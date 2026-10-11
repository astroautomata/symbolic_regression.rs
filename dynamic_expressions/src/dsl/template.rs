//! Template expressions, as in SymbolicRegression.jl's `TemplateStructure`: a combine formula
//! written in the DSL that calls subexpressions (`f(x1, x2)`), their derivatives (`D(f, 1)(x)`),
//! and indexes learnable parameter vectors (`p[class]`).

use core::fmt;

use ndarray::{Array2, ArrayView2};
use num_traits::Float;
use rustc_hash::FxHashMap;

use super::diff::{PartialForm, differentiate_calls, differentiate_closed};
use super::ir::{Graph, Node, NodeId, OpIr};
use super::kernels::Kernel;
use super::lower::{Lowered, TemplateScope, lower_formula};
use super::program::Program;
use super::{DslError, Span, parse_formula};
use crate::compile::{EvalPlan, compile_plan};
use crate::dispatch::SrcRef;
use crate::evaluate::{EvalOptions, eval_plan_array_into};
use crate::expression::PostfixExpr;
use crate::node::PNode;
use crate::strings::{StringTreeOptions, format_g, string_tree};
use crate::traits::{OpId, OperatorSet};

/// Why a template cannot be evaluated at all, as opposed to evaluating to an invalid result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TemplateEvalError {
    /// The formula's value is the same for every row, which SymbolicRegression.jl rejects.
    ScalarReturn,
    /// `D` differentiates a subexpression that uses this operator, which has no IR.
    UnsupportedOperator(String),
}

impl fmt::Display for TemplateEvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ScalarReturn => write!(
                f,
                "the template formula returns a scalar; it must return one value per row, using a feature or \
                 calling a subexpression on one"
            ),
            Self::UnsupportedOperator(name) => write!(
                f,
                "`D` cannot differentiate operator `{name}`; only built-in and DSL-defined operators support it"
            ),
        }
    }
}

impl std::error::Error for TemplateEvalError {}

const CHECKED: EvalOptions = EvalOptions {
    check_finite: true,
    early_exit: true,
};

/// The fixed part of a template expression: subexpression names and arities, parameter vector
/// names and lengths, the feature names the formula reads, and the compiled formula.
#[derive(Clone, Debug)]
pub struct TemplateStructure {
    expressions: Vec<String>,
    num_features: Vec<usize>,
    parameters: Vec<(String, usize)>,
    variable_names: Vec<String>,
    graph: Graph,
    root: NodeId,
    /// The formula's value is the same for every row, so it cannot be evaluated.
    scalar: bool,
    combine: String,
}

/// Reusable buffers for [`TemplateStructure::eval_into`], and one compiled derivative per `D`
/// call.
#[derive(Clone, Debug)]
pub struct TemplateWorkspace<T> {
    cols: Vec<Vec<T>>,
    sub_x: Array2<T>,
    scratch: Array2<T>,
    derivatives: Vec<Option<Derivative<T>>>,
    temps: Vec<T>,
}

impl<T> Default for TemplateWorkspace<T> {
    fn default() -> Self {
        Self {
            cols: Vec::new(),
            sub_x: Array2::from_shape_vec((0, 0), Vec::new()).expect("empty array"),
            scratch: Array2::from_shape_vec((0, 0), Vec::new()).expect("empty array"),
            derivatives: Vec::new(),
            temps: Vec::new(),
        }
    }
}

/// `D(...(D(tree, wrt[0]), ...), wrt[k])` compiled for one call, with the leaves it reads and what
/// it was compiled from: the operator set type (which fixes the built-in kernels) and its DSL
/// definitions, the call, the tree's nodes, and which of its constants are 0 or 1 (see
/// [`const_kind`]). Those constants are folded, which can drop leaves from the derivative; the
/// other constants are program inputs, so new values of them reuse the program.
#[derive(Clone, Debug)]
struct Derivative<T> {
    ops_type: &'static str,
    expr: u16,
    wrt: Vec<u16>,
    nodes: Vec<PNode>,
    const_kinds: Vec<u8>,
    definitions: Vec<Lowered>,
    program: Program<T>,
    leaves: Leaves,
}

impl<T: Float> Derivative<T> {
    fn compiled_from(
        &self,
        ops_type: &str,
        expr: u16,
        wrt: &[u16],
        nodes: &[PNode],
        consts: &[T],
        definitions: &[Lowered],
    ) -> bool {
        self.expr == expr
            && self.wrt == wrt
            && self.nodes == nodes
            && self
                .const_kinds
                .iter()
                .copied()
                .eq(consts.iter().map(|&c| const_kind(c)))
            && self.ops_type == ops_type
            && self.definitions.len() == definitions.len()
            && self
                .definitions
                .iter()
                .zip(definitions)
                .all(|(a, b)| a.arity == b.arity && a.root == b.root && a.graph.nodes() == b.graph.nodes())
    }
}

#[derive(Copy, Clone)]
enum Val<T> {
    Const(T),
    Feature(usize),
    Col(usize),
}

impl TemplateStructure {
    /// Compiles `combine`, which reads the features `variable_names` (in dataset column order),
    /// calls the subexpressions `expressions`, indexes the parameter vectors `parameters` (name
    /// and length), and may call the DSL-defined operators of `ops`. Each subexpression's arity
    /// is inferred from its calls unless `num_features` gives them, as SymbolicRegression.jl's
    /// `num_features` does.
    pub fn new<S: AsRef<str>, Ops: OperatorSet>(
        combine: &str,
        expressions: &[S],
        num_features: Option<&[usize]>,
        variable_names: &[S],
        parameters: &[(S, usize)],
        ops: &Ops,
    ) -> Result<Self, DslError> {
        let whole = Span::new(0, combine.len());
        let err = |message: String| DslError::new(message, whole, combine);
        let expressions: Vec<String> = expressions.iter().map(|s| s.as_ref().to_string()).collect();
        let variable_names: Vec<String> = variable_names.iter().map(|s| s.as_ref().to_string()).collect();
        let parameters: Vec<(String, usize)> = parameters.iter().map(|(s, n)| (s.as_ref().to_string(), *n)).collect();

        if expressions.is_empty() {
            return Err(err("a template needs at least one subexpression".to_string()));
        }
        let mut seen: Vec<&str> = Vec::new();
        for name in expressions
            .iter()
            .chain(parameters.iter().map(|(n, _)| n))
            .chain(&variable_names)
        {
            if seen.contains(&name.as_str()) {
                return Err(err(format!(
                    "`{name}` names more than one subexpression, parameter vector, or feature"
                )));
            }
            seen.push(name);
        }
        if let Some(n) = num_features {
            if n.len() != expressions.len() {
                return Err(err(format!(
                    "{} subexpressions but {} feature counts",
                    expressions.len(),
                    n.len()
                )));
            }
        }

        let formula = parse_formula(combine)?;
        let lowered = lower_formula(
            &formula,
            &TemplateScope {
                variables: &variable_names,
                expressions: &expressions,
                parameters: &parameters,
                num_features,
            },
            ops.definitions(),
        )?;

        let num_features = match num_features {
            Some(n) => n.to_vec(),
            None => expressions
                .iter()
                .zip(&lowered.arities)
                .map(|(name, called)| {
                    called.map(usize::from).ok_or_else(|| {
                        err(format!(
                            "failed to infer the number of features used by `{name}`: the formula never calls it"
                        ))
                    })
                })
                .collect::<Result<_, _>>()?,
        };

        Ok(Self {
            expressions,
            num_features,
            parameters,
            variable_names,
            graph: lowered.graph,
            root: lowered.root,
            scalar: lowered.scalar,
            combine: combine.to_string(),
        })
    }

    pub fn combine(&self) -> &str {
        &self.combine
    }

    pub fn expressions(&self) -> &[String] {
        &self.expressions
    }

    /// The number of arguments each subexpression takes, in [`Self::expressions`] order.
    pub fn num_features(&self) -> &[usize] {
        &self.num_features
    }

    /// The most arguments any subexpression takes.
    pub fn max_features(&self) -> usize {
        self.num_features.iter().copied().max().unwrap_or(0)
    }

    pub fn parameters(&self) -> &[(String, usize)] {
        &self.parameters
    }

    /// The total length of all parameter vectors; [`Self::eval_into`] takes them concatenated.
    pub fn n_parameters(&self) -> usize {
        self.parameters.iter().map(|(_, n)| n).sum()
    }

    pub fn variable_names(&self) -> &[String] {
        &self.variable_names
    }

    /// Whether the formula calls `D`.
    pub fn uses_derivatives(&self) -> bool {
        self.graph
            .nodes()
            .iter()
            .any(|n| matches!(n, Node::Call { wrt, .. } if !wrt.is_empty()))
    }

    /// Whether some subexpression uses a feature beyond its arity, which makes the expression
    /// invalid.
    pub fn has_invalid_variables<T, Ops, const D: usize>(&self, trees: &[PostfixExpr<T, Ops, D>]) -> bool {
        trees
            .iter()
            .zip(&self.num_features)
            .any(|(tree, &n)| reads_beyond(tree, n))
    }

    /// Checks that every parameter index that depends only on the features is a valid 1-based
    /// index for every row of `x`.
    pub fn check_indices<T: Float>(&self, x: ArrayView2<'_, T>) -> Result<(), String> {
        let n_rows = x.ncols();
        for node in self.graph.nodes() {
            let Node::Param { vector, index } = node else {
                continue;
            };
            let (name, len) = &self.parameters[usize::from(*vector)];
            for row in 0..n_rows {
                let Some(i) = self.feature_value(*index, x, row) else {
                    break;
                };
                if i.fract() != 0.0 || i < 1.0 || i > *len as f64 {
                    return Err(format!(
                        "row {} indexes `{name}` with {i}, but its indices are the integers 1 to {len}",
                        row + 1
                    ));
                }
            }
        }
        Ok(())
    }

    /// The value of `id` at `row` when it depends only on features and constants.
    fn feature_value<T: Float>(&self, id: NodeId, x: ArrayView2<'_, T>, row: usize) -> Option<f64> {
        let node = self.graph.node(id);
        let args: Option<Vec<f64>> = node.args().iter().map(|&a| self.feature_value(a, x, row)).collect();
        Some(match node {
            Node::Const(bits) => f64::from_bits(*bits),
            Node::Arg(i) => x[[usize::from(*i), row]].to_f64()?,
            Node::Op { kernel, .. } => kernel.eval_scalar(&args?),
            Node::Partial { .. } | Node::Call { .. } | Node::Param { .. } => return None,
        })
    }

    /// Checks that `D` can differentiate every operator in `operators`: built-in and DSL-defined
    /// operators have IR, Rust closures do not.
    pub fn check_operators<Ops: OperatorSet>(&self, ops: &Ops, operators: &[OpId]) -> Result<(), TemplateEvalError> {
        if !self.uses_derivatives() {
            return Ok(());
        }
        match operators.iter().find(|&&op| ops.ir(op).is_none()) {
            Some(&op) => Err(TemplateEvalError::UnsupportedOperator(ops.name(op).to_string())),
            None => Ok(()),
        }
    }

    /// Compiles one evaluation plan per subexpression.
    pub fn compile_plans<T, Ops, const D: usize>(&self, trees: &[PostfixExpr<T, Ops, D>]) -> Vec<EvalPlan<D>> {
        trees
            .iter()
            .map(|t| compile_plan(&t.nodes, self.max_features(), t.consts.len()))
            .collect()
    }

    /// Evaluates the template on `x` (features by rows) into `out`. `params` holds the parameter
    /// vectors concatenated in declaration order. Returns `Ok(false)` when the expression is
    /// invalid: a subexpression uses a feature beyond its arity or its call's arguments, a
    /// parameter index is not an integer in range, or any intermediate value is not finite, as
    /// SymbolicRegression.jl's `ValidVector`. Fails when the formula returns a scalar, or `D`
    /// differentiates an operator without IR.
    #[allow(clippy::too_many_arguments)]
    pub fn eval_into<T, Ops, const D: usize>(
        &self,
        out: &mut [T],
        trees: &[PostfixExpr<T, Ops, D>],
        plans: &[EvalPlan<D>],
        params: &[T],
        ops: &Ops,
        x: ArrayView2<'_, T>,
        ws: &mut TemplateWorkspace<T>,
    ) -> Result<bool, TemplateEvalError>
    where
        T: Float,
        Ops: OperatorSet<T = T>,
    {
        debug_assert_eq!(trees.len(), self.expressions.len());
        assert_eq!(
            params.len(),
            self.n_parameters(),
            "expected {} parameters, got {}",
            self.n_parameters(),
            params.len()
        );
        assert!(x.is_standard_layout(), "X must be contiguous");
        let x_data = x.as_slice().expect("X must be contiguous");
        let n = x.ncols();
        assert_eq!(out.len(), n);
        if self.scalar {
            return Err(TemplateEvalError::ScalarReturn);
        }
        if self.has_invalid_variables(trees) {
            return Ok(false);
        }

        let nodes = self.graph.nodes();
        ws.cols.resize_with(nodes.len(), Vec::new);
        ws.derivatives.resize_with(nodes.len(), || None);
        let mut vals: Vec<Val<T>> = Vec::with_capacity(nodes.len());
        for (i, node) in nodes.iter().enumerate() {
            let (done, rest) = ws.cols.split_at_mut(i);
            let col = &mut rest[0];
            let src = |v: Val<T>| -> SrcRef<'_, T> {
                match v {
                    Val::Const(c) => SrcRef::Const(c),
                    Val::Feature(f) => SrcRef::Slice(&x_data[f * n..(f + 1) * n]),
                    Val::Col(c) => SrcRef::Slice(&done[c]),
                }
            };
            let args: Vec<SrcRef<'_, T>> = node.args().iter().map(|a| src(vals[a.index()])).collect();
            let v = match node {
                Node::Const(bits) => Val::Const(T::from(f64::from_bits(*bits)).expect("f64 converts to T")),
                Node::Arg(f) => Val::Feature(usize::from(*f)),
                Node::Op { kernel, .. } => {
                    col.resize(n, T::zero());
                    if !(kernel.eval_fn::<T>())(col, &args, &CHECKED) {
                        return Ok(false);
                    }
                    Val::Col(i)
                }
                Node::Partial { .. } => unreachable!("formulas are never differentiated"),
                Node::Param { vector, .. } => {
                    let offset: usize = self.parameters[..usize::from(*vector)].iter().map(|(_, l)| l).sum();
                    let len = self.parameters[usize::from(*vector)].1;
                    let entries = &params[offset..offset + len];
                    col.resize(n, T::zero());
                    for (r, o) in col.iter_mut().enumerate() {
                        let idx = match args[0] {
                            SrcRef::Slice(s) => s[r],
                            SrcRef::Const(c) => c,
                        };
                        let Some(k) = index_of(idx, len) else {
                            return Ok(false);
                        };
                        *o = entries[k];
                    }
                    Val::Col(i)
                }
                Node::Call { expr, wrt, .. } => {
                    let k = usize::from(*expr);
                    let tree = &trees[k];
                    if reads_beyond(tree, args.len()) {
                        return Ok(false);
                    }
                    col.resize(n, T::zero());
                    let ok = if wrt.is_empty() {
                        fill_rows(&mut ws.sub_x, &args, n);
                        eval_plan_array_into(col, &plans[k], tree, ops, ws.sub_x.view(), &mut ws.scratch, &CHECKED)
                    } else {
                        let d = derivative(&mut ws.derivatives[i], *expr, wrt, tree, ops)?;
                        let finite = |s: &SrcRef<'_, T>| match s {
                            SrcRef::Slice(s) => s.iter().all(|v| v.is_finite()),
                            SrcRef::Const(c) => c.is_finite(),
                        };
                        let finite_leaves = d.leaves.features.iter().all(|&f| finite(&args[usize::from(f)]))
                            && d.leaves.consts.iter().all(|&c| tree.consts[usize::from(c)].is_finite());
                        if !finite_leaves {
                            return Ok(false);
                        }
                        let mut inputs: Vec<SrcRef<'_, T>> = tree.consts.iter().map(|&c| SrcRef::Const(c)).collect();
                        inputs.extend_from_slice(&args);
                        ws.temps.resize(d.program.n_temps() * n, T::zero());
                        d.program.run(col, &inputs, &mut ws.temps);
                        col.iter().all(|v| v.is_finite())
                    };
                    if !ok {
                        return Ok(false);
                    }
                    Val::Col(i)
                }
            };
            vals.push(v);
        }

        match vals[self.root.index()] {
            Val::Const(c) => out.fill(c),
            Val::Feature(f) => out.copy_from_slice(&x_data[f * n..(f + 1) * n]),
            Val::Col(c) => out.copy_from_slice(&ws.cols[c]),
        }
        Ok(out.iter().all(|v| v.is_finite()))
    }

    /// Evaluates the template once, compiling plans and buffers for the call.
    pub fn eval<T, Ops, const D: usize>(
        &self,
        trees: &[PostfixExpr<T, Ops, D>],
        params: &[T],
        ops: &Ops,
        x: ArrayView2<'_, T>,
    ) -> Result<(Vec<T>, bool), TemplateEvalError>
    where
        T: Float,
        Ops: OperatorSet<T = T>,
    {
        let mut out = vec![T::zero(); x.ncols()];
        if self.scalar {
            return Err(TemplateEvalError::ScalarReturn);
        }
        if self.has_invalid_variables(trees) {
            return Ok((out, false));
        }
        let plans = self.compile_plans(trees);
        let ok = self.eval_into(
            &mut out,
            trees,
            &plans,
            params,
            ops,
            x,
            &mut TemplateWorkspace::default(),
        )?;
        Ok((out, ok))
    }

    /// Prints the template as SymbolicRegression.jl does: each subexpression with its arguments
    /// as `#1`, `#2`, ..., then each parameter vector, as `f = #1 * #2; p = [1, 2]`. Pretty
    /// printing puts each on its own line and shortens vectors longer than 5.
    pub fn string<T, Ops, const D: usize>(
        &self,
        trees: &[PostfixExpr<T, Ops, D>],
        params: &[T],
        ops: &Ops,
        pretty: bool,
        print_precision: Option<usize>,
    ) -> String
    where
        T: Float + core::fmt::Display,
        Ops: OperatorSet,
    {
        let names: Vec<String> = (1..=self.max_features()).map(|i| format!("#{i}")).collect();
        let opts = || StringTreeOptions {
            variable_names: Some(&names),
            pretty,
            print_precision,
        };
        let constant = |v: &T| match print_precision {
            Some(p) => format_g(v.to_f64().unwrap_or(f64::NAN), p),
            None => v.to_string(),
        };
        let mut parts: Vec<(String, String)> = self
            .expressions
            .iter()
            .zip(trees)
            .map(|(name, tree)| (name.clone(), string_tree(tree, ops, opts())))
            .collect();
        let mut offset = 0;
        for (name, len) in &self.parameters {
            let entries = &params[offset..offset + len];
            offset += len;
            let shown: Vec<String> = if pretty && *len > 5 {
                let mut s: Vec<String> = entries[..3].iter().map(constant).collect();
                s.push("...".to_string());
                s.push(constant(&entries[len - 1]));
                s
            } else {
                entries.iter().map(constant).collect()
            };
            parts.push((name.clone(), format!("[{}]", shown.join(", "))));
        }

        let last = parts.len() - 1;
        let lines: Vec<String> = parts
            .into_iter()
            .enumerate()
            .map(|(i, (name, s))| {
                let prefix = match () {
                    _ if !pretty || last == 0 => "",
                    _ if i == 0 => "╭ ",
                    _ if i == last => "╰ ",
                    _ => "├ ",
                };
                format!("{prefix}{name} = {s}")
            })
            .collect();
        lines.join(if pretty { "\n" } else { "; " })
    }
}

/// The 0-based entry a 1-based parameter index selects.
fn index_of<T: Float>(i: T, len: usize) -> Option<usize> {
    let i = i.to_f64()?;
    (i.fract() == 0.0 && i >= 1.0 && i <= len as f64).then(|| i as usize - 1)
}

/// Stacks the argument columns of a call as the rows of `sub_x`.
fn fill_rows<T: Float>(sub_x: &mut Array2<T>, args: &[SrcRef<'_, T>], n: usize) {
    if sub_x.dim() != (args.len(), n) {
        *sub_x = Array2::zeros((args.len(), n));
    }
    for (mut row, arg) in sub_x.rows_mut().into_iter().zip(args) {
        match arg {
            SrcRef::Slice(s) => row.as_slice_mut().expect("rows are contiguous").copy_from_slice(s),
            SrcRef::Const(c) => row.fill(*c),
        }
    }
}

/// Whether `tree` uses a feature at or beyond `n`.
fn reads_beyond<T, Ops, const D: usize>(tree: &PostfixExpr<T, Ops, D>, n: usize) -> bool {
    tree.nodes
        .iter()
        .any(|node| matches!(node, PNode::Var { feature } if usize::from(*feature) >= n))
}

/// The program for `D(...(D(tree, wrt[0]), ...), wrt[k])` and the leaves it reads, compiled into
/// `slot` unless it already holds them for the same subexpression, operators, `D` call, and
/// constant kinds.
fn derivative<'w, T, Ops, const D: usize>(
    slot: &'w mut Option<Derivative<T>>,
    expr: u16,
    wrt: &[u16],
    tree: &PostfixExpr<T, Ops, D>,
    ops: &Ops,
) -> Result<&'w Derivative<T>, TemplateEvalError>
where
    T: Float,
    Ops: OperatorSet<T = T>,
{
    let ops_type = core::any::type_name::<Ops>();
    let definitions = ops.definitions();
    let cached = slot
        .as_ref()
        .is_some_and(|d| d.compiled_from(ops_type, expr, wrt, &tree.nodes, &tree.consts, definitions));
    if !cached {
        let folded: Vec<Option<f64>> = tree
            .consts
            .iter()
            .map(|&c| (const_kind(c) < 2).then(|| c.to_f64().expect("0 and 1 convert to f64")))
            .collect();
        let (program, leaves) = compile_derivative(tree, ops, wrt, &folded)?;
        *slot = Some(Derivative {
            ops_type,
            expr,
            wrt: wrt.to_vec(),
            nodes: tree.nodes.clone(),
            const_kinds: tree.consts.iter().map(|&c| const_kind(c)).collect(),
            definitions: definitions.to_vec(),
            program,
            leaves,
        });
    }
    Ok(slot.as_ref().expect("the slot was just filled"))
}

/// 0 for a zero constant, 1 for a constant of one, and 2 for any other.
fn const_kind<T: Float>(c: T) -> u8 {
    if c.is_zero() {
        0
    } else if c == T::one() {
        1
    } else {
        2
    }
}

/// Lowers `tree` to IR, with constant `i` as input `i` unless `folded[i]` gives its value, feature
/// `f` as input `consts.len() + f`, and each operator as an opaque call, differentiates it with
/// respect to each feature in `wrt`, and compiles the result. As in SymbolicRegression.jl's `D`,
/// each operator's partials are taken on the operator alone (see [`differentiate_calls`]) and
/// joined by the chain rule. A feature the tree cannot read has derivative 0.
fn compile_derivative<T, Ops, const D: usize>(
    tree: &PostfixExpr<T, Ops, D>,
    ops: &Ops,
    wrt: &[u16],
    folded: &[Option<f64>],
) -> Result<(Program<T>, Leaves), TemplateEvalError>
where
    T: Float,
    Ops: OperatorSet<T = T>,
{
    let n_consts = tree.consts.len();
    let input = |i: usize| u16::try_from(i).expect("a subexpression has at most 65535 inputs");
    let mut functions = Functions::default();
    let mut g = Graph::default();
    let mut stack: Vec<NodeId> = Vec::with_capacity(tree.nodes.len());
    for node in &tree.nodes {
        let id = match *node {
            PNode::Var { feature } => g.arg(input(n_consts + usize::from(feature))),
            PNode::Const { idx } => match folded[usize::from(idx)] {
                Some(c) => g.constant(c),
                None => g.arg(idx),
            },
            PNode::Op { arity, op } => {
                let args = stack.split_off(stack.len() - usize::from(arity));
                let expr = functions.of(ops, OpId { arity, id: op })?;
                g.intern(Node::Call {
                    expr,
                    wrt: Vec::new(),
                    args,
                })
            }
        };
        stack.push(id);
    }
    let mut root = stack.pop().expect("a tree has a value");
    for &w in wrt {
        root = match u16::try_from(n_consts + usize::from(w)) {
            Ok(arg) => differentiate_calls(&mut g, root, arg, &mut |f, w| functions.partial_form(f, w)),
            Err(_) => g.constant(0.0),
        };
    }
    let (g, root) = functions.inline(&g, root);
    Ok((Program::compile(&g, root), input_leaves(&g, root, n_consts)))
}

/// The operators of one tree as functions of their arguments (`Arg(i)`), and the partial
/// derivatives `D` needs, each taken on the function alone.
#[derive(Default)]
struct Functions {
    ops: Vec<OpId>,
    graphs: FxHashMap<(u16, Vec<u16>), (Graph, NodeId)>,
}

impl Functions {
    /// The function for `op`.
    fn of<Ops: OperatorSet>(&mut self, ops: &Ops, op: OpId) -> Result<u16, TemplateEvalError> {
        if let Some(f) = self.ops.iter().position(|&o| o == op) {
            return Ok(f as u16);
        }
        let f = u16::try_from(self.ops.len()).expect("a tree has at most 65535 distinct operators");
        let function = match ops.ir(op) {
            Some(OpIr::Kernel(kernel)) => {
                let mut g = Graph::default();
                let args: Vec<NodeId> = (0..u16::from(op.arity)).map(|i| g.arg(i)).collect();
                // DynamicDiff differentiates the `abs` operator to `sign`, which is 0 at 0; inside
                // a custom operator, ForwardDiff's `abs` rule applies instead.
                if kernel == Kernel::Abs {
                    let mut sign = Graph::default();
                    let x = sign.arg(0);
                    let root = sign.op(Kernel::Sign, vec![x]);
                    self.graphs.insert((f, vec![0]), (sign, root));
                }
                let root = g.op(kernel, args);
                (g, root)
            }
            Some(OpIr::Defined(def)) => (def.graph.clone(), def.root),
            None => return Err(TemplateEvalError::UnsupportedOperator(ops.name(op).to_string())),
        };
        self.ops.push(op);
        self.graphs.insert((f, Vec::new()), function);
        Ok(f)
    }

    /// Function `f` differentiated with respect to each of its arguments in `wrt`.
    fn get(&mut self, f: u16, wrt: &[u16]) -> &(Graph, NodeId) {
        let key = (f, wrt.to_vec());
        if !self.graphs.contains_key(&key) {
            let (&last, rest) = wrt.split_last().expect("every operator's function is stored");
            let (mut g, root) = self.get(f, rest).clone();
            let d = differentiate_closed(&mut g, root, last);
            let (g, roots) = g.extract(&[d]);
            self.graphs.insert(key.clone(), (g, roots[0]));
        }
        &self.graphs[&key]
    }

    fn partial_form(&mut self, f: u16, wrt: &[u16]) -> PartialForm {
        let (g, root) = self.get(f, wrt);
        match g.node(*root) {
            Node::Const(bits) => PartialForm::Const(f64::from_bits(*bits)),
            Node::Arg(i) => PartialForm::Arg(*i),
            _ => PartialForm::Other,
        }
    }

    /// `root` of `g` with every call replaced by its function's graph.
    fn inline(&mut self, g: &Graph, root: NodeId) -> (Graph, NodeId) {
        let reachable = g.reachable(&[root]);
        let mut out = Graph::default();
        let mut map = vec![NodeId(u32::MAX); g.len()];
        for (i, node) in g.nodes().iter().enumerate() {
            if !reachable[i] {
                continue;
            }
            let args: Vec<NodeId> = node.args().iter().map(|a| map[a.index()]).collect();
            map[i] = match node {
                Node::Call { expr, wrt, .. } => {
                    let (callee, callee_root) = self.get(*expr, wrt);
                    out.inline(callee, *callee_root, &args)
                }
                Node::Op { kernel, .. } => out.op(*kernel, args),
                Node::Const(_) | Node::Arg(_) => out.intern(node.clone()),
                Node::Partial { .. } | Node::Param { .. } => unreachable!("a tree lowers to calls of its operators"),
            };
        }
        (out, map[root.index()])
    }
}

/// The features and constants a compiled derivative reads, ascending. The derivative is invalid
/// where one of them is not finite.
#[derive(Clone, Debug, Default)]
struct Leaves {
    features: Vec<u16>,
    consts: Vec<u16>,
}

/// The inputs `root` of `g` depends on: constant `i` is input `i` and feature `f` is input
/// `n_consts + f`.
fn input_leaves(g: &Graph, root: NodeId, n_consts: usize) -> Leaves {
    let mut leaves = Leaves::default();
    for (node, live) in g.nodes().iter().zip(g.reachable(&[root])) {
        if let (true, Node::Arg(i)) = (live, node) {
            match usize::from(*i).checked_sub(n_consts) {
                Some(f) => leaves.features.push(f as u16),
                None => leaves.consts.push(*i),
            }
        }
    }
    leaves.features.sort_unstable();
    leaves.consts.sort_unstable();
    leaves
}
