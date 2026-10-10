use super::ir::{Graph, Node, NodeId};
use super::kernels::Kernel;

/// Adds `d(root)/d(Arg(wrt))` to `graph` and returns its id.
///
/// `Add`, `Sub`, `Mul`, `Div`, and `Neg` are differentiated symbolically, `IfElse` per branch,
/// comparisons and logical operators have derivative 0, and every other kernel uses the chain rule
/// with its `Partial` nodes. Zero terms are dropped as they are built.
pub fn differentiate(graph: &mut Graph, root: NodeId, wrt: u16) -> NodeId {
    let n = root.index() + 1;
    let mut reachable = graph.reachable(&[root]);
    reachable.truncate(n);

    let mut d: Vec<NodeId> = Vec::with_capacity(n);
    for (i, &live) in reachable.iter().enumerate() {
        if !live {
            d.push(NodeId(u32::MAX));
            continue;
        }
        let di = match graph.nodes()[i].clone() {
            Node::Const(_) => zero(graph),
            Node::Arg(k) => graph.constant(if k == wrt { 1.0 } else { 0.0 }),
            Node::Op { kernel, args } => op_derivative(graph, kernel, &args, &d),
            Node::Partial { .. } => unreachable!("partial nodes are never differentiated"),
            Node::Call { .. } | Node::Param { .. } => unreachable!("template nodes are never differentiated"),
        };
        d.push(di);
    }
    d[root.index()]
}

fn op_derivative(g: &mut Graph, kernel: Kernel, args: &[NodeId], d: &[NodeId]) -> NodeId {
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
