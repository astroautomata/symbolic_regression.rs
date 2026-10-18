//! SymbolicRegression.jl's dimensional analysis (`DimensionalAnalysis.jl`): an expression is
//! evaluated on the first data row with each value carrying SI dimensions. A constant is a
//! wildcard that takes whatever dimensions make its expression consistent, unless
//! `dimensionless_constants_only` is set.
//!
//! An operator propagates dimensions only through the methods Julia's `WildcardQuantity` has:
//! `+`, `-`, `*`, `/`, `^`, `sqrt`, `cbrt`, `abs`, and `inv`, and Julia functions built from
//! them (`square`, `cube`, `identity`, and DSL operators whose bodies use only those on their
//! arguments). Any other operator, a DSL literal, or a call mixing a dimensioned value with a
//! plain number has no such method; its result is consistent only when every argument is a
//! wildcard, and is then a dimensionless value.

use dynamic_expressions::dispatch::{EvalKernelCtx, SrcRef};
use dynamic_expressions::dsl::ir::{Node, OpIr};
use dynamic_expressions::dsl::kernels::Kernel;
use dynamic_expressions::{EvalOptions, OpId, OperatorSet, PNode, PostfixExpr};
use ndarray::ArrayView2;
use num_traits::Float;

use crate::units::{DENOM, Dimensions, Units};

/// SymbolicRegression.jl's `WildcardQuantity`: a value with SI dimensions, whether those
/// dimensions are still free (`wildcard`), and whether the expression so far is inconsistent.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct WildcardQuantity<T> {
    pub value: T,
    pub dims: Dimensions,
    pub wildcard: bool,
    pub violates: bool,
}

impl<T: Float> WildcardQuantity<T> {
    pub fn new(value: T, dims: Dimensions, wildcard: bool) -> Self {
        Self {
            value,
            dims,
            wildcard,
            violates: false,
        }
    }

    fn violation() -> Self {
        Self {
            value: T::one(),
            dims: Dimensions::default(),
            wildcard: false,
            violates: true,
        }
    }
}

/// An argument inside an operator call: a dimensioned value, or a plain number (a stripped
/// wildcard or a DSL literal).
#[derive(Copy, Clone, Debug)]
enum Arg<T> {
    Quantity(WildcardQuantity<T>),
    Number(T),
}

/// SymbolicRegression.jl's `violates_dimensional_constraints`, evaluated on the first row of `x`
/// (features by rows).
pub(crate) fn violates_dimensional_constraints<T: Float, Ops: OperatorSet<T = T>, const D: usize>(
    tree: &PostfixExpr<T, Ops, D>,
    units: &Units,
    x: ArrayView2<'_, T>,
    ops: &Ops,
    dimensionless_constants_only: bool,
) -> bool {
    let out = dimensional_eval(tree, units, x, ops, !dimensionless_constants_only);
    out.violates || units.y.as_ref().is_some_and(|y| !out.wildcard && out.dims != y.dims)
}

pub(crate) fn dimensional_eval<T: Float, Ops: OperatorSet<T = T>, const D: usize>(
    tree: &PostfixExpr<T, Ops, D>,
    units: &Units,
    x: ArrayView2<'_, T>,
    ops: &Ops,
    allow_wildcards: bool,
) -> WildcardQuantity<T> {
    dynamic_expressions::node_utils::tree_mapreduce(
        &tree.nodes,
        |n| match *n {
            PNode::Const { idx } => {
                WildcardQuantity::new(tree.consts[usize::from(idx)], Dimensions::default(), allow_wildcards)
            }
            PNode::Var { feature } => {
                let f = usize::from(feature);
                let unit = &units.x[f];
                WildcardQuantity::new(x[[f, 0]] * T::from(unit.scale).unwrap(), unit.dims, false)
            }
            PNode::Op { .. } => unreachable!("branch node in leaf mapper"),
        },
        |n| match *n {
            PNode::Op { arity, op } => OpId { arity, id: op },
            _ => unreachable!("leaf node in branch mapper"),
        },
        |op, args| degn_eval(ops, op, args),
    )
}

/// SymbolicRegression.jl's `degn_eval`: tries the operator with every argument dimensioned, then
/// with wildcard arguments stripped to plain numbers, in Julia's lattice order; with every
/// argument stripped (all wildcards), the result is the operator's plain value.
pub(crate) fn degn_eval<T: Float, Ops: OperatorSet<T = T>>(
    ops: &Ops,
    op: OpId,
    args: &[WildcardQuantity<T>],
) -> WildcardQuantity<T> {
    if let Some(v) = args.iter().find(|a| a.violates) {
        return *v;
    }
    if args.iter().any(|a| !a.value.is_finite()) {
        return WildcardQuantity::violation();
    }
    let n = args.len();
    let all = (1usize << n) - 1;
    for stripped in 0..=all {
        let is_stripped = |j: usize| stripped >> (n - 1 - j) & 1 == 1;
        if (0..n).any(|j| is_stripped(j) && !args[j].wildcard) {
            continue;
        }
        if stripped == all {
            let values: Vec<T> = args.iter().map(|a| a.value).collect();
            return WildcardQuantity::new(plain_value(ops, op, &values), Dimensions::default(), false);
        }
        let inputs: Vec<Arg<T>> = (0..n)
            .map(|j| {
                if is_stripped(j) {
                    Arg::Number(args[j].value)
                } else {
                    Arg::Quantity(args[j])
                }
            })
            .collect();
        if let Some(Arg::Quantity(w)) = call(ops.ir(op), &inputs) {
            if !w.violates {
                return w;
            }
        }
    }
    WildcardQuantity::violation()
}

/// The operator on plain numbers, as the search evaluates it.
fn plain_value<T: Float, Ops: OperatorSet<T = T>>(ops: &Ops, op: OpId, values: &[T]) -> T {
    let mut out = [T::zero()];
    let args: Vec<SrcRef<'_, T>> = values.iter().map(|&v| SrcRef::Const(v)).collect();
    let opts = EvalOptions {
        check_finite: false,
        early_exit: false,
    };
    ops.eval(
        op,
        EvalKernelCtx {
            out: &mut out,
            args: &args,
            opts: &opts,
        },
    );
    out[0]
}

/// Calls an operator given as IR; `None` when Julia would find no method for these arguments.
fn call<T: Float>(ir: Option<OpIr<'_>>, args: &[Arg<T>]) -> Option<Arg<T>> {
    match ir? {
        OpIr::Kernel(kernel) => apply(kernel, args),
        OpIr::Defined(lowered) => {
            let mut values: Vec<Arg<T>> = Vec::with_capacity(lowered.graph.len());
            for node in &lowered.graph.nodes()[..=lowered.root.index()] {
                let v = match node {
                    Node::Const(bits) => Arg::Number(T::from(f64::from_bits(*bits)).unwrap()),
                    Node::Arg(i) => args[usize::from(*i)],
                    Node::Op { kernel, args: ids } => {
                        let inputs: Vec<Arg<T>> = ids.iter().map(|id| values[id.index()]).collect();
                        apply(*kernel, &inputs)?
                    }
                    Node::Partial { .. } | Node::Call { .. } | Node::Param { .. } => {
                        unreachable!("operator definitions hold only constants, arguments, and kernels")
                    }
                };
                values.push(v);
            }
            Some(values[lowered.root.index()])
        }
    }
}

/// One kernel: plain numbers evaluate as numbers; dimensioned values go through the kernel's
/// `WildcardQuantity` method, if it has one; a mix has no method. Julia's `ifelse` and `?:` take
/// any branch values, so a plain-number condition selects a branch of either kind.
fn apply<T: Float>(kernel: Kernel, args: &[Arg<T>]) -> Option<Arg<T>> {
    if let (Kernel::IfElse, Arg::Number(c)) = (kernel, args[0]) {
        return Some(if c > T::zero() { args[1] } else { args[2] });
    }
    if let Some(values) = args
        .iter()
        .map(|a| match a {
            Arg::Number(v) => Some(*v),
            Arg::Quantity(_) => None,
        })
        .collect::<Option<Vec<T>>>()
    {
        return Some(Arg::Number(kernel.eval_scalar(&values)));
    }
    let w: Vec<WildcardQuantity<T>> = args
        .iter()
        .map(|a| match a {
            Arg::Quantity(w) => Some(*w),
            Arg::Number(_) => None,
        })
        .collect::<Option<_>>()?;
    quantity_method(kernel, &w).map(Arg::Quantity)
}

/// The kernel's method on `WildcardQuantity`, if Julia has one (`DimensionalAnalysis.jl`).
fn quantity_method<T: Float>(kernel: Kernel, args: &[WildcardQuantity<T>]) -> Option<WildcardQuantity<T>> {
    if !has_quantity_method(kernel) {
        return None;
    }
    // Each method returns its first inconsistent argument unchanged.
    if let Some(v) = args.iter().find(|a| a.violates) {
        return Some(*v);
    }
    Some(match kernel {
        Kernel::Add => add(args[0], args[1], false),
        Kernel::Sub => add(args[0], args[1], true),
        Kernel::Mul => mul(args[0], args[1], false),
        Kernel::Div => mul(args[0], args[1], true),
        Kernel::Square => mul(args[0], args[0], false),
        Kernel::Cube => mul(mul(args[0], args[0], false), args[0], false),
        Kernel::Pow => {
            let (l, r) = (args[0], args[1]);
            let free = |w: WildcardQuantity<T>| w.dims.is_dimensionless() || w.wildcard;
            if free(l) && free(r) {
                WildcardQuantity::new(kernel.eval_scalar(&[l.value, r.value]), Dimensions::default(), false)
            } else {
                WildcardQuantity::violation()
            }
        }
        Kernel::Sqrt => unary(args[0], kernel, args[0].dims.pow(DENOM / 2)),
        Kernel::Cbrt => unary(args[0], kernel, args[0].dims.pow(DENOM / 3)),
        Kernel::Abs => unary(args[0], kernel, args[0].dims),
        Kernel::Inv => unary(args[0], kernel, args[0].dims.inv()),
        Kernel::Identity => args[0],
        _ => unreachable!("`has_quantity_method` lists every method"),
    })
}

fn has_quantity_method(kernel: Kernel) -> bool {
    matches!(
        kernel,
        Kernel::Add
            | Kernel::Sub
            | Kernel::Mul
            | Kernel::Div
            | Kernel::Pow
            | Kernel::Sqrt
            | Kernel::Cbrt
            | Kernel::Abs
            | Kernel::Inv
            | Kernel::Identity
            | Kernel::Square
            | Kernel::Cube
    )
}

fn unary<T: Float>(w: WildcardQuantity<T>, kernel: Kernel, dims: Dimensions) -> WildcardQuantity<T> {
    WildcardQuantity::new(kernel.eval_scalar(&[w.value]), dims, w.wildcard)
}

fn mul<T: Float>(l: WildcardQuantity<T>, r: WildcardQuantity<T>, divide: bool) -> WildcardQuantity<T> {
    let (value, dims) = if divide {
        (l.value / r.value, l.dims / r.dims)
    } else {
        (l.value * r.value, l.dims * r.dims)
    };
    WildcardQuantity::new(value, dims, l.wildcard || r.wildcard)
}

/// A wildcard operand takes the other operand's dimensions; two wildcards with different
/// dimensions give a dimensionless wildcard.
fn add<T: Float>(l: WildcardQuantity<T>, r: WildcardQuantity<T>, subtract: bool) -> WildcardQuantity<T> {
    let value = if subtract { l.value - r.value } else { l.value + r.value };
    if l.dims == r.dims {
        WildcardQuantity::new(value, l.dims, l.wildcard && r.wildcard)
    } else if l.wildcard && r.wildcard {
        WildcardQuantity::new(value, Dimensions::default(), true)
    } else if l.wildcard {
        WildcardQuantity::new(value, r.dims, false)
    } else if r.wildcard {
        WildcardQuantity::new(value, l.dims, false)
    } else {
        WildcardQuantity::violation()
    }
}
