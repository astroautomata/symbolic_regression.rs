use proptest::prelude::*;
use proptest::strategy::{BoxedStrategy, Union};

use crate::node::PNode;

#[derive(Clone, Debug)]
pub enum GenExpr {
    Var(u16),
    Const(u16),
    Op { arity: u8, op: u16, children: Vec<GenExpr> },
}

impl GenExpr {
    pub fn to_postfix(&self, out: &mut Vec<PNode>) {
        match self {
            GenExpr::Var(feature) => out.push(PNode::Var { feature: *feature }),
            GenExpr::Const(idx) => out.push(PNode::Const { idx: *idx }),
            GenExpr::Op { arity, op, children } => {
                for child in children {
                    child.to_postfix(out);
                }
                out.push(PNode::Op { arity: *arity, op: *op });
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn arb_expr(
    n_features: usize,
    n_consts: usize,
    unary_ops: Vec<u16>,
    binary_ops: Vec<u16>,
    ternary_ops: Vec<u16>,
    max_depth: u32,
    max_size: u32,
    max_branch: u32,
) -> impl Strategy<Value = GenExpr> {
    let mut leaf_choices: Vec<BoxedStrategy<GenExpr>> = Vec::new();
    if n_features > 0 {
        leaf_choices.push((0u16..(n_features as u16)).prop_map(GenExpr::Var).boxed());
    }
    if n_consts > 0 {
        leaf_choices.push((0u16..(n_consts as u16)).prop_map(GenExpr::Const).boxed());
    }
    debug_assert!(
        !leaf_choices.is_empty(),
        "arb_expr requires at least one feature or constant"
    );
    let leaf = Union::new(leaf_choices);

    leaf.prop_recursive(max_depth, max_size, max_branch, move |inner| {
        let mut choices: Vec<BoxedStrategy<GenExpr>> = Vec::new();

        if !unary_ops.is_empty() {
            let ops = unary_ops.clone();
            choices.push(
                (prop::sample::select(ops), inner.clone())
                    .prop_map(|(op, child)| GenExpr::Op {
                        arity: 1,
                        op,
                        children: vec![child],
                    })
                    .boxed(),
            );
        }

        if !binary_ops.is_empty() {
            let ops = binary_ops.clone();
            choices.push(
                (prop::sample::select(ops), prop::collection::vec(inner.clone(), 2))
                    .prop_map(|(op, children)| GenExpr::Op { arity: 2, op, children })
                    .boxed(),
            );
        }

        if !ternary_ops.is_empty() {
            let ops = ternary_ops.clone();
            choices.push(
                (prop::sample::select(ops), prop::collection::vec(inner.clone(), 3))
                    .prop_map(|(op, children)| GenExpr::Op { arity: 3, op, children })
                    .boxed(),
            );
        }

        debug_assert!(!choices.is_empty(), "arb_expr requires at least one operator");
        Union::new(choices)
    })
}

#[allow(clippy::too_many_arguments)]
pub fn arb_postfix_nodes(
    n_features: usize,
    n_consts: usize,
    unary_ops: Vec<u16>,
    binary_ops: Vec<u16>,
    ternary_ops: Vec<u16>,
    max_depth: u32,
    max_size: u32,
    max_branch: u32,
) -> impl Strategy<Value = Vec<PNode>> {
    arb_expr(
        n_features,
        n_consts,
        unary_ops,
        binary_ops,
        ternary_ops,
        max_depth,
        max_size,
        max_branch,
    )
    .prop_map(|expr| {
        let mut nodes = Vec::new();
        expr.to_postfix(&mut nodes);
        nodes
    })
}
