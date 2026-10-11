//! Solving an expression for one of its subtrees: SymbolicRegression.jl's `approx_inverse` and
//! `eval_inverse_tree_array`.

use dynamic_expressions::node::PNode;
use dynamic_expressions::node_utils::subtree_sizes;
use dynamic_expressions::{EvalOptions, Metadata, OpId, OperatorSet, PostfixExpr, XColumns, eval_tree_array};
use num_traits::Float;

/// Solves `op(args) = y` for one argument: called as `f(y, args)`, where `args` holds every
/// argument of the operator and the slot being solved for is ignored.
pub type InverseFn<T> = fn(T, &[T]) -> T;

fn safe_log<T: Float>(x: T) -> T {
    if x <= T::zero() { T::nan() } else { x.ln() }
}

fn safe_pow<T: Float>(x: T, y: T) -> T {
    if x == T::zero() && y < T::zero() {
        T::nan()
    } else {
        x.powf(y)
    }
}

/// Julia's `mod(x, 2)`, floored.
fn mod2<T: Float>(x: T) -> T {
    let two = T::one() + T::one();
    x - two * (x / two).floor()
}

/// SymbolicRegression.jl's `approx_inverse`: a function that inverts the built-in operator `name`
/// (with `arity` arguments) in its argument `position`, on its principal branch. Inverses
/// return NaN outside their domain. `None` for operators SymbolicRegression.jl does not invert.
///
/// ```
/// use symbolic_regression::backsolve::approx_inverse;
///
/// let asin = approx_inverse::<f64>("sin", 1, 0).unwrap();
/// assert_eq!(asin(0.5, &[0.0]), 0.5f64.asin());
/// // `2.0 ^ x = 8.0` gives `x = 3.0`:
/// let log_base = approx_inverse::<f64>("pow", 2, 1).unwrap();
/// assert!((log_base(8.0, &[2.0, 0.0]) - 3.0).abs() < 1e-12);
/// assert!(approx_inverse::<f64>("abs", 1, 0).is_none());
/// ```
pub fn approx_inverse<T: Float>(name: &str, arity: u8, position: usize) -> Option<InverseFn<T>> {
    let f: InverseFn<T> = match (arity, name, position) {
        (1, "sin", _) => |y, _| if y.abs() <= T::one() { y.asin() } else { T::nan() },
        (1, "asin", _) => |y, _| y.sin(),
        (1, "cos", _) => |y, _| if y.abs() <= T::one() { y.acos() } else { T::nan() },
        (1, "acos", _) => |y, _| y.cos(),
        (1, "tan", _) => |y, _| y.atan(),
        (1, "atan", _) => |y, _| y.tan(),
        (1, "sinh", _) => |y, _| y.asinh(),
        (1, "asinh", _) => |y, _| y.sinh(),
        (1, "cosh", _) => |y, _| if y < T::one() { T::nan() } else { y.acosh() },
        (1, "acosh", _) => |y, _| y.cosh(),
        (1, "tanh", _) => |y, _| (mod2(y + T::one()) - T::one()).atanh(),
        (1, "atanh", _) => |y, _| y.tanh(),
        (1, "square", _) => |y, _| if y < T::zero() { T::nan() } else { y.sqrt() },
        (1, "sqrt", _) => |y, _| y * y,
        (1, "cube", _) => |y, _| y.cbrt(),
        (1, "cbrt", _) => |y, _| y * y * y,
        (1, "exp", _) => |y, _| safe_log(y),
        (1, "log", _) => |y, _| y.exp(),
        (1, "exp2", _) => |y, _| if y <= T::zero() { T::nan() } else { y.log2() },
        (1, "log2", _) => |y, _| y.exp2(),
        (1, "log10", _) => |y, _| T::from(10.0).expect("Float can represent 10.0").powf(y),
        (1, "expm1", _) => |y, _| if y <= -T::one() { T::nan() } else { y.ln_1p() },
        (1, "log1p", _) => |y, _| y.exp() - T::one(),
        (1, "neg", _) => |y, _| -y,
        (1, "inv", _) => |y, _| y.recip(),
        (1, "identity", _) => |y, _| y,
        (2, "add", 0) => |y, a| y - a[1],
        (2, "add", 1) => |y, a| y - a[0],
        (2, "sub", 0) => |y, a| y + a[1],
        (2, "sub", 1) => |y, a| a[0] - y,
        (2, "mul", 0) => |y, a| y / a[1],
        (2, "mul", 1) => |y, a| y / a[0],
        (2, "div", 0) => |y, a| y * a[1],
        (2, "div", 1) => |y, a| a[0] / y,
        (2, "pow", 0) => |y, a| safe_pow(y, a[1].recip()),
        (2, "pow", 1) => |y, a| safe_log(y) / safe_log(a[0]),
        (3, "fma", 0) => |y, a| (y - a[2]) / a[1],
        (3, "fma", 1) => |y, a| (y - a[2]) / a[0],
        (3, "fma", 2) => |y, a| y - a[0] * a[1],
        _ => return None,
    };
    Some(f)
}

/// [`approx_inverse`] of an operator in `ops`; operators defined at runtime have no inverse.
pub(crate) fn op_inverse<Ops: OperatorSet>(ops: &Ops, op: OpId, position: usize) -> Option<InverseFn<Ops::T>> {
    if ops.is_custom(op) {
        return None;
    }
    approx_inverse(ops.name(op), op.arity, position)
}

/// The subtree of `tree` spanning postfix positions `start..=end`, as its own expression.
pub(crate) fn subtree<T: Float, Ops, const D: usize>(
    tree: &PostfixExpr<T, Ops, D>,
    start: usize,
    end: usize,
) -> PostfixExpr<T, Ops, D> {
    let mut sub = PostfixExpr::new(
        tree.nodes[start..=end].to_vec(),
        tree.consts.clone(),
        Metadata::default(),
    );
    dynamic_expressions::compress_constants(&mut sub);
    sub
}

/// SymbolicRegression.jl's `eval_inverse_tree_array_masked`: the values the subtree rooted at
/// postfix position `node` must take for `tree` to output `y`, and a per-row validity mask.
///
/// Row `i` is valid only if `y[i]` is finite, every sibling subtree along the path from the root is
/// finite there, and every inversion step produced a finite value. A structural failure (an
/// operator without an inverse, or `node` not in `tree`) gives an all-false mask.
pub fn eval_inverse_tree_array_masked<'a, T, Ops, const D: usize>(
    tree: &PostfixExpr<T, Ops, D>,
    ops: &Ops,
    x: impl Into<XColumns<'a, T>>,
    node: usize,
    y: &[T],
) -> (Vec<T>, Vec<bool>)
where
    T: Float + 'a,
    Ops: OperatorSet<T = T>,
{
    let x = x.into();
    let n = y.len();
    let mut y = y.to_vec();
    let mut valid: Vec<bool> = y.iter().map(|v| v.is_finite()).collect();
    let sizes = subtree_sizes(&tree.nodes);
    let opts = EvalOptions {
        check_finite: false,
        early_exit: false,
    };
    let mut siblings: Vec<Vec<T>> = Vec::with_capacity(D);
    let mut args = [T::zero(); D];
    let mut root = tree.nodes.len() - 1;
    while root != node {
        let PNode::Op { arity, op } = tree.nodes[root] else {
            return (y, vec![false; n]);
        };
        let a = arity as usize;
        let mut ranges = [(0usize, 0usize); D];
        let mut end = root;
        for range in ranges[..a].iter_mut().rev() {
            let last = end - 1;
            end = last + 1 - sizes[last];
            *range = (end, last);
        }
        let Some(pos) = ranges[..a].iter().position(|&(s, e)| s <= node && node <= e) else {
            return (y, vec![false; n]);
        };
        let Some(inverse) = op_inverse(ops, OpId { arity, id: op }, pos) else {
            return (y, vec![false; n]);
        };

        siblings.clear();
        for (k, &(s, e)) in ranges[..a].iter().enumerate() {
            if k == pos {
                siblings.push(Vec::new());
                continue;
            }
            let (values, complete) = eval_tree_array(&subtree(tree, s, e), ops, x, &opts);
            if !complete {
                return (y, vec![false; n]);
            }
            for (ok, v) in valid.iter_mut().zip(&values) {
                *ok &= v.is_finite();
            }
            siblings.push(values);
        }
        if !valid.contains(&true) {
            return (y, valid);
        }

        for (row, (yr, ok)) in y.iter_mut().zip(valid.iter_mut()).enumerate() {
            for (k, sibling) in siblings.iter().enumerate() {
                if k != pos {
                    args[k] = sibling[row];
                }
            }
            *yr = inverse(*yr, &args[..a]);
            *ok &= yr.is_finite();
        }
        if !valid.contains(&true) {
            return (y, valid);
        }
        root = ranges[pos].1;
    }
    (y, valid)
}

/// SymbolicRegression.jl's `eval_inverse_tree_array`: [`eval_inverse_tree_array_masked`], valid
/// only if every row is.
pub fn eval_inverse_tree_array<'a, T, Ops, const D: usize>(
    tree: &PostfixExpr<T, Ops, D>,
    ops: &Ops,
    x: impl Into<XColumns<'a, T>>,
    node: usize,
    y: &[T],
) -> (Vec<T>, bool)
where
    T: Float + 'a,
    Ops: OperatorSet<T = T>,
{
    let (values, valid) = eval_inverse_tree_array_masked(tree, ops, x, node, y);
    (values, valid.iter().all(|&v| v))
}
