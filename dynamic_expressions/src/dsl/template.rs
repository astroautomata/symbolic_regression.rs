//! Template expressions, as in SymbolicRegression.jl's `TemplateStructure`: a combine formula
//! written in the DSL that calls subexpressions (`f(x1, x2)`), their derivatives (`D(f, 1)(x)`),
//! and indexes learnable parameter vectors (`p[class]`).

use ndarray::{Array2, ArrayView2};
use num_traits::Float;

use super::dual::{Dual, Jet};
use super::ir::{Graph, Node, NodeId};
use super::lower::{TemplateScope, lower_formula};
use super::{DslError, Span, parse_formula};
use crate::compile::{EvalPlan, compile_plan};
use crate::dispatch::SrcRef;
use crate::evaluate::{EvalOptions, eval_plan_array_into};
use crate::expression::PostfixExpr;
use crate::node::PNode;
use crate::strings::{StringTreeOptions, format_g, string_tree};
use crate::traits::{OpId, OperatorSet};

/// `D` may be nested this deep, as in `D(D(D(f, 1), 1), 1)`.
pub const MAX_DERIVATIVE_ORDER: usize = 3;

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
    combine: String,
}

/// Reusable buffers for [`TemplateStructure::eval_into`].
#[derive(Clone, Debug)]
pub struct TemplateWorkspace<T> {
    cols: Vec<Vec<T>>,
    sub_x: Array2<T>,
    scratch: Array2<T>,
}

impl<T> Default for TemplateWorkspace<T> {
    fn default() -> Self {
        Self {
            cols: Vec::new(),
            sub_x: Array2::from_shape_vec((0, 0), Vec::new()).expect("empty array"),
            scratch: Array2::from_shape_vec((0, 0), Vec::new()).expect("empty array"),
        }
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
    /// calls the subexpressions `expressions`, and indexes the parameter vectors `parameters`
    /// (name and length). Each subexpression's arity is inferred from its calls unless
    /// `num_features` gives them, in which case every call must agree.
    pub fn new<S: AsRef<str>>(
        combine: &str,
        expressions: &[S],
        num_features: Option<&[usize]>,
        variable_names: &[S],
        parameters: &[(S, usize)],
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
            },
        )?;

        let mut arities = Vec::with_capacity(expressions.len());
        for (k, (name, called)) in expressions.iter().zip(&lowered.arities).enumerate() {
            let declared = num_features.map(|n| n[k]);
            let arity = match (called, declared) {
                (Some(c), Some(d)) if usize::from(*c) != d => {
                    return Err(err(format!(
                        "`{name}` is declared with {d} features but called with {c} arguments"
                    )));
                }
                (Some(c), _) => usize::from(*c),
                (None, Some(d)) => d,
                (None, None) => {
                    return Err(err(format!(
                        "failed to infer the number of features used by `{name}`: the formula never calls it"
                    )));
                }
            };
            arities.push(arity);
        }

        for node in lowered.graph.nodes() {
            if let Node::Call { wrt, .. } = node {
                if wrt.len() > MAX_DERIVATIVE_ORDER {
                    return Err(err(format!(
                        "`D` is nested {} deep; at most {MAX_DERIVATIVE_ORDER} is supported",
                        wrt.len()
                    )));
                }
            }
        }

        Ok(Self {
            expressions,
            num_features: arities,
            parameters,
            variable_names,
            graph: lowered.graph,
            root: lowered.root,
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
        trees.iter().zip(&self.num_features).any(|(tree, &n)| {
            tree.nodes
                .iter()
                .any(|node| matches!(node, PNode::Var { feature } if usize::from(*feature) >= n))
        })
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

    /// Checks that `ops` can evaluate every operator on dual numbers, which `D` needs.
    pub fn check_operators<Ops: OperatorSet>(&self, ops: &Ops, operators: &[OpId]) -> Result<(), String> {
        if !self.uses_derivatives() {
            return Ok(());
        }
        for &op in operators {
            let args = vec![Dual::constant(1.0f64); usize::from(op.arity)];
            if ops.eval_scalar(op, &args).is_none() {
                return Err(format!(
                    "`D` cannot differentiate operator `{}`; only built-in and DSL-defined operators support it",
                    ops.name(op)
                ));
            }
        }
        Ok(())
    }

    /// Compiles one evaluation plan per subexpression.
    pub fn compile_plans<T, Ops, const D: usize>(&self, trees: &[PostfixExpr<T, Ops, D>]) -> Vec<EvalPlan<D>> {
        trees
            .iter()
            .map(|t| compile_plan(&t.nodes, self.max_features(), t.consts.len()))
            .collect()
    }

    /// Evaluates the template on `x` (features by rows) into `out`. `params` holds the parameter
    /// vectors concatenated in declaration order. Returns `false` when the expression is invalid:
    /// a subexpression uses a feature beyond its arity, a parameter index is not an integer in
    /// range, or any intermediate value is not finite, as SymbolicRegression.jl's `ValidVector`.
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
    ) -> bool
    where
        T: Float,
        Ops: OperatorSet<T = T>,
    {
        debug_assert_eq!(trees.len(), self.expressions.len());
        debug_assert_eq!(params.len(), self.n_parameters());
        assert!(x.is_standard_layout(), "X must be contiguous");
        let x_data = x.as_slice().expect("X must be contiguous");
        let n = x.ncols();
        assert_eq!(out.len(), n);
        if self.has_invalid_variables(trees) {
            return false;
        }

        let nodes = self.graph.nodes();
        ws.cols.resize_with(nodes.len(), Vec::new);
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
                        return false;
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
                            return false;
                        };
                        *o = entries[k];
                    }
                    Val::Col(i)
                }
                Node::Call { expr, wrt, .. } => {
                    let k = usize::from(*expr);
                    col.resize(n, T::zero());
                    let ok = if wrt.is_empty() {
                        fill_rows(&mut ws.sub_x, &args, n);
                        eval_plan_array_into(
                            col,
                            &plans[k],
                            &trees[k],
                            ops,
                            ws.sub_x.view(),
                            &mut ws.scratch,
                            &CHECKED,
                        )
                    } else {
                        let tree = &trees[k];
                        match wrt.len() {
                            1 => eval_jet::<Dual<T>, T, Ops, D>(col, tree, ops, &args, wrt),
                            2 => eval_jet::<Dual<Dual<T>>, T, Ops, D>(col, tree, ops, &args, wrt),
                            _ => eval_jet::<Dual<Dual<Dual<T>>>, T, Ops, D>(col, tree, ops, &args, wrt),
                        }
                    };
                    if !ok {
                        return false;
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
        out.iter().all(|v| v.is_finite())
    }

    /// Evaluates the template once, compiling plans and buffers for the call.
    pub fn eval<T, Ops, const D: usize>(
        &self,
        trees: &[PostfixExpr<T, Ops, D>],
        params: &[T],
        ops: &Ops,
        x: ArrayView2<'_, T>,
    ) -> (Vec<T>, bool)
    where
        T: Float,
        Ops: OperatorSet<T = T>,
    {
        let mut out = vec![T::zero(); x.ncols()];
        let plans = self.compile_plans(trees);
        let ok = self.eval_into(
            &mut out,
            trees,
            &plans,
            params,
            ops,
            x,
            &mut TemplateWorkspace::default(),
        );
        (out, ok)
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

/// Evaluates `D(...(D(tree, wrt[0]), ...), wrt[k])` at every row by running the tree's operators on
/// nested dual numbers.
fn eval_jet<J, T, Ops, const D: usize>(
    out: &mut [T],
    tree: &PostfixExpr<T, Ops, D>,
    ops: &Ops,
    args: &[SrcRef<'_, T>],
    wrt: &[u16],
) -> bool
where
    J: Jet<T>,
    T: Float,
    Ops: OperatorSet<T = T>,
{
    let seeds: Vec<Vec<bool>> = (0..args.len())
        .map(|a| wrt.iter().map(|&w| usize::from(w) == a).collect())
        .collect();
    let mut stack: Vec<J> = Vec::with_capacity(tree.nodes.len());
    let mut operands: Vec<J> = Vec::with_capacity(D);
    for (r, o) in out.iter_mut().enumerate() {
        stack.clear();
        for node in &tree.nodes {
            let v = match *node {
                PNode::Var { feature } => {
                    let f = usize::from(feature);
                    let x = match args[f] {
                        SrcRef::Slice(s) => s[r],
                        SrcRef::Const(c) => c,
                    };
                    J::seed(x, &seeds[f])
                }
                PNode::Const { idx } => J::seed(tree.consts[usize::from(idx)], &[false; MAX_DERIVATIVE_ORDER]),
                PNode::Op { arity, op } => {
                    operands.clear();
                    operands.extend(stack.drain(stack.len() - usize::from(arity)..));
                    ops.eval_scalar(OpId { arity, id: op }, &operands)
                        .expect("template setup checks that every operator supports `D`")
                }
            };
            stack.push(v);
        }
        *o = stack.pop().expect("a tree has a value").top();
        if !o.is_finite() {
            return false;
        }
    }
    true
}
