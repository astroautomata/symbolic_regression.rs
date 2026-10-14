#![cfg(feature = "serde")]

use dynamic_expressions::PNode;
use dynamic_expressions::expression::{Metadata, PostfixExpr};
use serde_json::json;
use symbolic_regression::{Expression, OneOrMany};

type Expr<T> = Expression<T, (), 2>;

fn tree<T>(nodes: Vec<PNode>, consts: Vec<T>, names: &[&str]) -> PostfixExpr<T, (), 2> {
    let meta = Metadata {
        variable_names: names.iter().map(|name| name.to_string()).collect(),
    };
    PostfixExpr::new(nodes, consts, meta)
}

fn x0_plus_c0() -> Vec<PNode> {
    vec![
        PNode::Var { feature: 0 },
        PNode::Const { idx: 0 },
        PNode::Op { arity: 2, op: 3 },
    ]
}

fn assert_same<T: PartialEq + std::fmt::Debug>(left: &Expr<T>, right: &Expr<T>) {
    assert_eq!(
        matches!(left.trees, OneOrMany::Single(_)),
        matches!(right.trees, OneOrMany::Single(_))
    );
    assert_eq!(left.trees.len(), right.trees.len());
    for (a, b) in left.trees.iter().zip(right.trees.iter()) {
        assert_eq!(a.nodes, b.nodes);
        assert_eq!(a.consts, b.consts);
        assert_eq!(a.meta.variable_names, b.meta.variable_names);
    }
    assert_eq!(left.params, right.params);
}

#[test]
fn single_tree_f64_has_stable_json_layout_and_round_trips() {
    let expr: Expr<f64> = Expression {
        trees: OneOrMany::Single(tree(x0_plus_c0(), vec![1.5], &["x", "y"])),
        params: vec![0.25, -2.0],
    };

    let value = serde_json::to_value(&expr).unwrap();
    assert_eq!(
        value,
        json!({
            "trees": {"Single": {
                "nodes": [
                    {"Var": {"feature": 0}},
                    {"Const": {"idx": 0}},
                    {"Op": {"arity": 2, "op": 3}}
                ],
                "consts": [1.5],
                "meta": {"variable_names": ["x", "y"]}
            }},
            "params": [0.25, -2.0]
        })
    );

    let back: Expr<f64> = serde_json::from_value(value).unwrap();
    assert_same(&expr, &back);
}

#[test]
fn multiple_trees_f32_round_trip() {
    let expr: Expr<f32> = Expression {
        trees: OneOrMany::Multiple(vec![
            tree(x0_plus_c0(), vec![0.5], &["a"]),
            tree(
                vec![
                    PNode::Const { idx: 1 },
                    PNode::Var { feature: 1 },
                    PNode::Op { arity: 1, op: 7 },
                    PNode::Op { arity: 2, op: 0 },
                ],
                vec![3.0, -1.25],
                &["a", "b"],
            ),
        ]),
        params: vec![4.5],
    };

    let text = serde_json::to_string(&expr).unwrap();
    let back: Expr<f32> = serde_json::from_str(&text).unwrap();
    assert!(matches!(&back.trees, OneOrMany::Multiple(trees) if trees.len() == 2));
    assert_same(&expr, &back);
}

#[test]
fn unknown_node_variant_is_rejected() {
    let value = json!({
        "trees": {"Single": {
            "nodes": [{"Param": {"idx": 0}}],
            "consts": [],
            "meta": {"variable_names": ["x"]}
        }},
        "params": []
    });
    assert!(serde_json::from_value::<Expr<f64>>(value).is_err());
}

#[test]
fn malformed_node_fields_are_rejected() {
    let value = json!({
        "trees": {"Single": {
            "nodes": [{"Op": {"arity": 300, "op": 0}}],
            "consts": [],
            "meta": {"variable_names": []}
        }},
        "params": []
    });
    assert!(serde_json::from_value::<Expr<f64>>(value).is_err());
}
