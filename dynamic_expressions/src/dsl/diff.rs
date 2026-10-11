use super::ir::{Graph, Node, NodeId};
use super::kernels::Kernel;

/// Adds `d(root)/d(Arg(wrt))` to `graph` and returns its id.
///
/// `Add`, `Sub`, `Mul`, `Div`, and `Neg` are differentiated symbolically, `IfElse` per branch,
/// comparisons and logical operators have derivative 0, and every other kernel uses the chain rule
/// with its `Partial` nodes. Zero terms are dropped as they are built.
pub fn differentiate(graph: &mut Graph, root: NodeId, wrt: u16) -> NodeId {
    derivative(graph, root, wrt, Rules::Kernel)
}

/// [`differentiate`] with each partial written in other kernels instead of a `Partial` node, so
/// the result can be differentiated again. The partials are ForwardDiff's, as SymbolicRegression.jl's
/// `D` takes them of one operator with its arguments as dual numbers: 0 where a safe operator
/// returns NaN, 0 for `x^y` with respect to `x` when `y` is 0, and DiffRules' `min` and `max`,
/// which give a tie to the first argument of `min` and the second of `max`. `IfElse` conditions
/// compare derivatives at ties, as comparisons of dual numbers do.
pub fn differentiate_closed(graph: &mut Graph, root: NodeId, wrt: u16) -> NodeId {
    derivative(graph, root, wrt, Rules::Closed)
}

/// Differentiates a graph whose operators are `Call { expr, wrt, args }` nodes: function `expr`,
/// differentiated with respect to each of its arguments in `wrt`, applied to `args`. The chain rule
/// adds `Call` nodes for the functions' partials, which stay opaque, so each operator is
/// differentiated on its own, as SymbolicRegression.jl's `D` does. `is_zero_partial(expr, wrt)`
/// tells which partials are 0, so their terms are dropped.
pub fn differentiate_calls(
    graph: &mut Graph,
    root: NodeId,
    wrt: u16,
    is_zero_partial: &mut dyn FnMut(u16, &[u16]) -> bool,
) -> NodeId {
    derivative(graph, root, wrt, Rules::Calls(is_zero_partial))
}

enum Rules<'a> {
    Kernel,
    Closed,
    Calls(&'a mut dyn FnMut(u16, &[u16]) -> bool),
}

fn derivative(graph: &mut Graph, root: NodeId, wrt: u16, mut rules: Rules<'_>) -> NodeId {
    let n = root.index() + 1;
    let mut reachable = graph.reachable(&[root]);
    reachable.truncate(n);
    let closed = !matches!(rules, Rules::Kernel);

    let mut d: Vec<NodeId> = Vec::with_capacity(n);
    for (i, &live) in reachable.iter().enumerate() {
        if !live {
            d.push(NodeId(u32::MAX));
            continue;
        }
        let di = match (graph.nodes()[i].clone(), &mut rules) {
            (Node::Const(_), _) => zero(graph),
            (Node::Arg(k), _) => graph.constant(if k == wrt { 1.0 } else { 0.0 }),
            (Node::Op { kernel, args }, _) => op_derivative(graph, kernel, &args, &d, closed),
            (Node::Call { expr, wrt: w, args }, Rules::Calls(is_zero_partial)) => {
                let mut total = zero(graph);
                for (j, arg) in (0u16..).zip(&args) {
                    let da = d[arg.index()];
                    if is_zero(graph, da) {
                        continue;
                    }
                    let mut partial_wrt = w.clone();
                    partial_wrt.push(j);
                    if is_zero_partial(expr, &partial_wrt) {
                        continue;
                    }
                    let partial = graph.intern(Node::Call {
                        expr,
                        wrt: partial_wrt,
                        args: args.clone(),
                    });
                    let term = mul(graph, partial, da);
                    total = add(graph, total, term);
                }
                total
            }
            (Node::Partial { .. }, _) => unreachable!("partial nodes are never differentiated"),
            (Node::Call { .. } | Node::Param { .. }, _) => unreachable!("template nodes are never differentiated"),
        };
        d.push(di);
    }
    d[root.index()]
}

fn op_derivative(g: &mut Graph, kernel: Kernel, args: &[NodeId], d: &[NodeId], closed: bool) -> NodeId {
    let da = |j: usize| d[args[j].index()];
    match kernel {
        Kernel::Add => add(g, da(0), da(1)),
        Kernel::Sub => sub(g, da(0), da(1)),
        Kernel::Neg => neg(g, da(0)),
        Kernel::Mul => {
            let left = mul(g, da(0), args[1]);
            let right = mul(g, args[0], da(1));
            add(g, left, right)
        }
        Kernel::Div => {
            let (a, b) = (args[0], args[1]);
            let first = div(g, da(0), b);
            let num = mul(g, a, da(1));
            let den = mul(g, b, b);
            let second = div(g, num, den);
            sub(g, first, second)
        }
        Kernel::IfElse => {
            let (then, otherwise) = (da(1), da(2));
            if is_zero(g, then) && is_zero(g, otherwise) {
                zero(g)
            } else {
                let cond = if closed {
                    lift_comparisons(g, args[0], d)
                } else {
                    args[0]
                };
                g.op(Kernel::IfElse, vec![cond, then, otherwise])
            }
        }
        k if k.returns_bool() => zero(g),
        k => {
            let dargs: Vec<NodeId> = (0..args.len()).map(da).collect();
            let mut total = zero(g);
            for (j, idx) in (0..args.len()).zip(0u8..) {
                if is_zero(g, da(j)) {
                    continue;
                }
                let partial = if closed {
                    closed_partial(g, k, idx, args, &dargs)
                } else {
                    g.partial(k, idx, args.to_vec())
                };
                let term = mul(g, partial, da(j));
                total = add(g, total, term);
            }
            total
        }
    }
}

/// `kernel`'s partial derivative with respect to argument `idx` at `args`, built from other
/// kernels, as ForwardDiff takes it of SymbolicRegression.jl's operator with the other arguments
/// fixed: the kernel's own partial formula, with ForwardDiff's rules where they differ. `dargs`
/// are the arguments' derivatives, which the operator's comparisons see at ties (see [`compare`]).
fn closed_partial(g: &mut Graph, kernel: Kernel, idx: u8, args: &[NodeId], dargs: &[NodeId]) -> NodeId {
    use Kernel as K;
    let x = args[0];
    let zero = g.constant(0.0);
    let one = g.constant(1.0);
    let minus_one = g.constant(-1.0);
    let two = g.constant(2.0);
    let three = g.constant(3.0);
    let ln2 = g.constant(core::f64::consts::LN_2);
    let ln10 = g.constant(core::f64::consts::LN_10);
    let indicator = |g: &mut Graph, cond: NodeId| g.op(K::IfElse, vec![cond, one, zero]);
    // `partial`, or 0 where `cond` holds. Where a safe operator returns NaN, it returns a constant,
    // whose partials are 0.
    let zero_if = |g: &mut Graph, cond: NodeId, partial: NodeId| g.op(K::IfElse, vec![cond, zero, partial]);
    // `x` compared with the constant `bound` in SymbolicRegression.jl's definition of the operator.
    let x_is = |g: &mut Graph, kernel: Kernel, bound: NodeId| compare(g, kernel, [x, bound], [dargs[0], zero]);
    // `partial` where `x` is in the domain of a safe operator, `-1 <= x <= 1`, and 0 elsewhere.
    let in_unit_interval = |g: &mut Graph, partial: NodeId| {
        let above = x_is(g, K::GreaterEqual, minus_one);
        let below = x_is(g, K::LessEqual, one);
        let inside = g.op(K::LogicalAnd, vec![above, below]);
        g.op(K::IfElse, vec![inside, partial, zero])
    };
    let where_x_is = |g: &mut Graph, kernel: Kernel, bound: NodeId, partial: NodeId| {
        let inside = x_is(g, kernel, bound);
        g.op(K::IfElse, vec![inside, partial, zero])
    };
    match kernel {
        K::Cos => {
            let s = g.op(K::Sin, vec![x]);
            neg(g, s)
        }
        K::Sin => g.op(K::Cos, vec![x]),
        K::Tan | K::Tanh => {
            let c = g.op(if kernel == K::Tan { K::Cos } else { K::Cosh }, vec![x]);
            let c2 = mul(g, c, c);
            div(g, one, c2)
        }
        K::Asin | K::Acos => {
            let x2 = mul(g, x, x);
            let r = sub(g, one, x2);
            let s = g.op(K::Sqrt, vec![r]);
            let partial = div(g, if kernel == K::Asin { one } else { minus_one }, s);
            in_unit_interval(g, partial)
        }
        K::Atan => {
            let x2 = mul(g, x, x);
            let den = add(g, one, x2);
            div(g, one, den)
        }
        K::Sinh => g.op(K::Cosh, vec![x]),
        K::Cosh => g.op(K::Sinh, vec![x]),
        K::Asinh => {
            let x2 = mul(g, x, x);
            let r = add(g, x2, one);
            let s = g.op(K::Sqrt, vec![r]);
            div(g, one, s)
        }
        K::Acosh => {
            let below = sub(g, x, one);
            let above = add(g, x, one);
            let a = g.op(K::Sqrt, vec![below]);
            let b = g.op(K::Sqrt, vec![above]);
            let den = mul(g, a, b);
            let partial = div(g, one, den);
            where_x_is(g, K::GreaterEqual, one, partial)
        }
        K::Atanh => {
            let x2 = mul(g, x, x);
            let den = sub(g, one, x2);
            let partial = div(g, one, den);
            in_unit_interval(g, partial)
        }
        K::Sec => {
            let s = g.op(K::Sec, vec![x]);
            let t = g.op(K::Tan, vec![x]);
            mul(g, s, t)
        }
        K::Csc => {
            let c = g.op(K::Csc, vec![x]);
            let nc = neg(g, c);
            let t = g.op(K::Cot, vec![x]);
            mul(g, nc, t)
        }
        K::Cot => {
            let s = g.op(K::Sin, vec![x]);
            let s2 = mul(g, s, s);
            div(g, minus_one, s2)
        }
        K::Exp | K::Expm1 => g.op(K::Exp, vec![x]),
        K::Exp2 => {
            let e = g.op(K::Exp2, vec![x]);
            mul(g, e, ln2)
        }
        K::Log => {
            let partial = div(g, one, x);
            where_x_is(g, K::Greater, zero, partial)
        }
        K::Log2 | K::Log10 => {
            let den = mul(g, x, if kernel == K::Log2 { ln2 } else { ln10 });
            let partial = div(g, one, den);
            where_x_is(g, K::Greater, zero, partial)
        }
        K::Log1p => {
            let den = add(g, one, x);
            let partial = div(g, one, den);
            where_x_is(g, K::Greater, minus_one, partial)
        }
        K::Sqrt => {
            let s = g.op(K::Sqrt, vec![x]);
            let den = mul(g, two, s);
            let partial = div(g, one, den);
            where_x_is(g, K::GreaterEqual, zero, partial)
        }
        K::Cbrt => {
            let c = g.op(K::Cbrt, vec![x]);
            let c2 = g.op(K::Square, vec![c]);
            let den = mul(g, three, c2);
            div(g, one, den)
        }
        // ForwardDiff's `abs` rule: -1 where `signbit(x)`, which is `1/x < 0`, and 1 elsewhere.
        K::Abs => {
            let inv = g.op(K::Inv, vec![x]);
            let negative = g.op(K::Less, vec![inv, zero]);
            g.op(K::IfElse, vec![negative, minus_one, one])
        }
        K::Abs2 | K::Square => mul(g, two, x),
        K::Cube => {
            let t = mul(g, three, x);
            mul(g, t, x)
        }
        K::Inv => {
            let x2 = mul(g, x, x);
            div(g, minus_one, x2)
        }
        K::Sign => zero,
        K::Identity => one,
        K::Pow => {
            let y = args[1];
            let p = g.op(K::Pow, vec![x, y]);
            // `safe_pow` returns NaN, a constant, where the kernel's value is NaN for numeric
            // arguments, except that a dual `x` is never zero, so `0^y` is not NaN for it.
            let p_nan = g.op(K::NotEqual, vec![p, p]);
            let x_number = g.op(K::Equal, vec![x, x]);
            let y_number = g.op(K::Equal, vec![y, y]);
            let numbers = g.op(K::LogicalAnd, vec![x_number, y_number]);
            let mut nan_region = g.op(K::LogicalAnd, vec![p_nan, numbers]);
            let partial = if idx == 0 {
                let negative = g.op(K::Less, vec![x, zero]);
                nan_region = g.op(K::LogicalAnd, vec![nan_region, negative]);
                // ForwardDiff's `x^y` for a dual `x` and `y == 0` has partial 0.
                let y_minus_one = match g.const_value(y) {
                    Some(c) => g.constant(c - 1.0),
                    None => sub(g, y, one),
                };
                let lowered = g.op(K::Pow, vec![x, y_minus_one]);
                let partial = mul(g, y, lowered);
                let constant_power = g.op(K::Equal, vec![y, zero]);
                zero_if(g, constant_power, partial)
            } else {
                // ForwardDiff's `x^y` for a dual `y` has partial 0 when `x == 0` and `y > 0`.
                let l = g.op(K::Log, vec![x]);
                let partial = mul(g, p, l);
                let x_zero = g.op(K::Equal, vec![x, zero]);
                let y_positive = g.op(K::Greater, vec![y, zero]);
                let flat = g.op(K::LogicalAnd, vec![x_zero, y_positive]);
                zero_if(g, flat, partial)
            };
            zero_if(g, nan_region, partial)
        }
        K::Atan2 => {
            let (y, x) = (args[0], args[1]);
            let x2 = mul(g, x, x);
            let y2 = mul(g, y, y);
            let den = add(g, x2, y2);
            let num = if idx == 0 { x } else { neg(g, y) };
            div(g, num, den)
        }
        // DiffRules: `min(a, b)` has partials `!(a > b)` and `a > b`; `max(a, b)` the reverse.
        K::Min | K::Max => {
            let a_greater = g.op(K::Greater, vec![args[0], args[1]]);
            if (kernel == K::Max) == (idx == 0) {
                indicator(g, a_greater)
            } else {
                g.op(K::IfElse, vec![a_greater, zero, one])
            }
        }
        K::Fma => [args[1], args[0], one][usize::from(idx)],
        // `clamp(x, lo, hi) = ifelse(x > hi, hi, ifelse(x < lo, lo, x))`.
        K::Clamp => {
            let (lo, hi) = (args[1], args[2]);
            let above = compare(g, K::Greater, [x, hi], [dargs[0], dargs[2]]);
            if idx == 2 {
                return indicator(g, above);
            }
            let below = compare(g, K::Less, [x, lo], [dargs[0], dargs[1]]);
            let below = if idx == 1 {
                below
            } else {
                g.op(K::LogicalNot, vec![below])
            };
            let not_above = g.op(K::LogicalNot, vec![above]);
            let picked = g.op(K::LogicalAnd, vec![not_above, below]);
            indicator(g, picked)
        }
        // `relu(x) = (x > 0) * x` and `cond(x, y) = (x > 0) * y`.
        K::Relu => {
            let positive = x_is(g, K::Greater, zero);
            indicator(g, positive)
        }
        K::Cond => {
            if idx == 0 {
                zero
            } else {
                let positive = x_is(g, K::Greater, zero);
                indicator(g, positive)
            }
        }
        K::Neg | K::Add | K::Sub | K::Mul | K::Div | K::IfElse => {
            unreachable!("`{kernel:?}` is differentiated symbolically")
        }
        k => {
            debug_assert!(k.returns_bool());
            zero
        }
    }
}

/// `cond` as ForwardDiff evaluates it on dual numbers (see [`compare`]). NaN tests (`u != u`)
/// compare values only, as `isnan` does.
fn lift_comparisons(g: &mut Graph, cond: NodeId, d: &[NodeId]) -> NodeId {
    use Kernel as K;
    let Node::Op { kernel, args } = g.node(cond).clone() else {
        return cond;
    };
    match kernel {
        K::LogicalAnd | K::LogicalOr | K::LogicalNot => {
            let lifted: Vec<NodeId> = args.iter().map(|&a| lift_comparisons(g, a, d)).collect();
            if lifted == args { cond } else { g.op(kernel, lifted) }
        }
        K::Equal | K::NotEqual | K::Less | K::LessEqual | K::Greater | K::GreaterEqual if args[0] != args[1] => {
            compare(g, kernel, [args[0], args[1]], [d[args[0].index()], d[args[1].index()]])
        }
        _ => cond,
    }
}

/// The comparison `kernel` of `u` and `v`, with derivatives `du` and `dv`, as on dual numbers:
/// when the values are equal it also compares the derivatives, so it holds just past the point.
fn compare(g: &mut Graph, kernel: Kernel, [u, v]: [NodeId; 2], [du, dv]: [NodeId; 2]) -> NodeId {
    use Kernel as K;
    let cond = g.op(kernel, vec![u, v]);
    if du == dv {
        return cond;
    }
    let on_partials = g.op(kernel, vec![du, dv]);
    match kernel {
        K::Equal => g.op(K::LogicalAnd, vec![cond, on_partials]),
        K::NotEqual => g.op(K::LogicalOr, vec![cond, on_partials]),
        _ => {
            let strict = match kernel {
                K::Less | K::LessEqual => K::Less,
                _ => K::Greater,
            };
            let by_value = g.op(strict, vec![u, v]);
            let tie = g.op(K::Equal, vec![u, v]);
            let by_partials = g.op(K::LogicalAnd, vec![tie, on_partials]);
            g.op(K::LogicalOr, vec![by_value, by_partials])
        }
    }
}

fn zero(g: &mut Graph) -> NodeId {
    g.constant(0.0)
}

fn is_zero(g: &Graph, id: NodeId) -> bool {
    g.const_value(id) == Some(0.0)
}

fn is_one(g: &Graph, id: NodeId) -> bool {
    g.const_value(id) == Some(1.0)
}

fn add(g: &mut Graph, a: NodeId, b: NodeId) -> NodeId {
    if is_zero(g, a) {
        b
    } else if is_zero(g, b) {
        a
    } else {
        g.op(Kernel::Add, vec![a, b])
    }
}

fn sub(g: &mut Graph, a: NodeId, b: NodeId) -> NodeId {
    if is_zero(g, b) {
        a
    } else if is_zero(g, a) {
        neg(g, b)
    } else {
        g.op(Kernel::Sub, vec![a, b])
    }
}

fn mul(g: &mut Graph, a: NodeId, b: NodeId) -> NodeId {
    if is_zero(g, a) || is_zero(g, b) {
        zero(g)
    } else if is_one(g, a) {
        b
    } else if is_one(g, b) {
        a
    } else {
        g.op(Kernel::Mul, vec![a, b])
    }
}

fn div(g: &mut Graph, a: NodeId, b: NodeId) -> NodeId {
    if is_zero(g, a) {
        zero(g)
    } else if is_one(g, b) {
        a
    } else {
        g.op(Kernel::Div, vec![a, b])
    }
}

fn neg(g: &mut Graph, a: NodeId) -> NodeId {
    match g.const_value(a) {
        Some(c) => g.constant(-c),
        None => g.op(Kernel::Neg, vec![a]),
    }
}
