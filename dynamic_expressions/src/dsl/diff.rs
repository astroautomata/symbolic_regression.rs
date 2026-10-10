use super::ir::{Graph, Node, NodeId};
use super::kernels::Kernel;

/// Adds `d(root)/d(Arg(wrt))` to `graph` and returns its id.
///
/// `Add`, `Sub`, `Mul`, `Div`, and `Neg` are differentiated symbolically, `IfElse` per branch,
/// comparisons and logical operators have derivative 0, and every other kernel uses the chain rule
/// with its `Partial` nodes. A `Partial` node is differentiated through its closed form in other
/// kernels, so results can be differentiated again. Zero terms are dropped as they are built.
pub fn differentiate(graph: &mut Graph, root: NodeId, wrt: u16) -> NodeId {
    let mut d: Vec<Option<NodeId>> = Vec::new();
    let mut stack = vec![root];
    while let Some(&id) = stack.last() {
        d.resize(graph.len(), None);
        if d[id.index()].is_some() {
            stack.pop();
            continue;
        }
        let node = graph.node(id).clone();
        let deps = match &node {
            Node::Partial { kernel, idx, args } => vec![closed_partial(graph, *kernel, *idx, args)],
            node => node.args().to_vec(),
        };
        d.resize(graph.len(), None);
        let pending: Vec<NodeId> = deps.iter().copied().filter(|a| d[a.index()].is_none()).collect();
        if !pending.is_empty() {
            stack.extend(pending);
            continue;
        }
        let di = match node {
            Node::Const(_) => zero(graph),
            Node::Arg(k) => graph.constant(if k == wrt { 1.0 } else { 0.0 }),
            Node::Op { kernel, args } => op_derivative(graph, kernel, &args, &d),
            Node::Partial { .. } => d[deps[0].index()].expect("the closed form is differentiated first"),
            Node::Call { .. } | Node::Param { .. } => unreachable!("template nodes are never differentiated"),
        };
        d[id.index()] = Some(di);
        stack.pop();
    }
    d[root.index()].expect("the root is differentiated")
}

fn op_derivative(g: &mut Graph, kernel: Kernel, args: &[NodeId], d: &[Option<NodeId>]) -> NodeId {
    let da = |j: usize| d[args[j].index()].expect("arguments are differentiated first");
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
                g.op(Kernel::IfElse, vec![args[0], then, otherwise])
            }
        }
        k if k.returns_bool() => zero(g),
        k => {
            let mut total = zero(g);
            for (j, idx) in (0..args.len()).zip(0u8..) {
                if is_zero(g, da(j)) {
                    continue;
                }
                let partial = g.partial(k, idx, args.to_vec());
                let term = mul(g, partial, da(j));
                total = add(g, total, term);
            }
            total
        }
    }
}

/// `kernel`'s partial derivative with respect to argument `idx` at `args`, built from other
/// kernels with the same formula and guards as the kernel's own partial.
fn closed_partial(g: &mut Graph, kernel: Kernel, idx: u8, args: &[NodeId]) -> NodeId {
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
    let outside = |g: &mut Graph, bound: NodeId, inside: NodeId| {
        let cond = g.op(K::LessEqual, vec![x, bound]);
        g.op(K::IfElse, vec![cond, zero, inside])
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
            div(g, if kernel == K::Asin { one } else { minus_one }, s)
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
            div(g, one, den)
        }
        K::Atanh => {
            let x2 = mul(g, x, x);
            let den = sub(g, one, x2);
            div(g, one, den)
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
            let inside = div(g, one, x);
            outside(g, zero, inside)
        }
        K::Log2 | K::Log10 => {
            let den = mul(g, x, if kernel == K::Log2 { ln2 } else { ln10 });
            let inside = div(g, one, den);
            outside(g, zero, inside)
        }
        K::Log1p => {
            let den = add(g, one, x);
            let inside = div(g, one, den);
            outside(g, minus_one, inside)
        }
        K::Sqrt => {
            let s = g.op(K::Sqrt, vec![x]);
            let den = mul(g, two, s);
            div(g, one, den)
        }
        K::Cbrt => {
            let c = g.op(K::Cbrt, vec![x]);
            let c2 = g.op(K::Square, vec![c]);
            let den = mul(g, three, c2);
            div(g, one, den)
        }
        K::Abs => g.op(K::Sign, vec![x]),
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
            if idx == 0 {
                let y_minus_one = match g.const_value(y) {
                    Some(c) => g.constant(c - 1.0),
                    None => sub(g, y, one),
                };
                let p = g.op(K::Pow, vec![x, y_minus_one]);
                mul(g, y, p)
            } else {
                let p = g.op(K::Pow, vec![x, y]);
                let l = g.op(K::Log, vec![x]);
                mul(g, p, l)
            }
        }
        K::Atan2 => {
            let (y, x) = (args[0], args[1]);
            let x2 = mul(g, x, x);
            let y2 = mul(g, y, y);
            let den = add(g, x2, y2);
            let num = if idx == 0 { x } else { neg(g, y) };
            div(g, num, den)
        }
        K::Min | K::Max => {
            let (a, b) = if idx == 0 {
                (args[0], args[1])
            } else {
                (args[1], args[0])
            };
            let (wins, loses) = if kernel == K::Min {
                (K::Less, K::Greater)
            } else {
                (K::Greater, K::Less)
            };
            let half = g.constant(0.5);
            let nan = g.constant(f64::NAN);
            let tie = g.op(K::Equal, vec![a, b]);
            let tie_or_nan = g.op(K::IfElse, vec![tie, half, nan]);
            let lost = g.op(loses, vec![a, b]);
            let not_won = g.op(K::IfElse, vec![lost, zero, tie_or_nan]);
            let won = g.op(wins, vec![a, b]);
            g.op(K::IfElse, vec![won, one, not_won])
        }
        K::Fma => [args[1], args[0], one][usize::from(idx)],
        K::Clamp => {
            let (lo, hi) = (args[1], args[2]);
            let below = g.op(K::Less, vec![x, lo]);
            let above = g.op(K::Greater, vec![x, hi]);
            match idx {
                0 => {
                    let out = g.op(K::LogicalOr, vec![below, above]);
                    g.op(K::IfElse, vec![out, zero, one])
                }
                1 => indicator(g, below),
                _ => indicator(g, above),
            }
        }
        K::Relu => {
            let positive = g.op(K::Greater, vec![x, zero]);
            indicator(g, positive)
        }
        K::Cond => {
            if idx == 0 {
                zero
            } else {
                let positive = g.op(K::Greater, vec![x, zero]);
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
