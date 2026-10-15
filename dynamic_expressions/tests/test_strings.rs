mod common;

use common::expr_readme_like;
use dynamic_expressions::strings::{StringTreeOptions, string_tree};
use dynamic_expressions::{HasOp, Metadata, PNode, PostfixExpr};

dynamic_expressions::op!(Add for f64 {
    display: "+",
    infix: "+",
    commutative: true,
    associative: true,
    eval: |[x, y]| { x + y },
    partial: |[_x, _y], _idx| { 1.0 },
});

dynamic_expressions::op!(Oplus for f64 {
    display: "⊕",
    infix: "⊕",
    eval: |[x, y, z]| { x + y + z },
    partial: |[_x, _y, _z], _idx| { 1.0 },
});

dynamic_expressions::opset! {
    Infix3Ops for f64 { Add, Oplus }
}

fn var_infix3(feature: u16) -> PostfixExpr<f64, Infix3Ops, 3> {
    PostfixExpr::new(vec![PNode::Var { feature }], vec![], Metadata::default())
}

fn add_infix3(x: PostfixExpr<f64, Infix3Ops, 3>, y: PostfixExpr<f64, Infix3Ops, 3>) -> PostfixExpr<f64, Infix3Ops, 3> {
    dynamic_expressions::expression_algebra::__apply_postfix::<f64, Infix3Ops, 3, 2>(
        <Infix3Ops as HasOp<Add>>::ID,
        [x, y],
    )
}

fn oplus3(
    x: PostfixExpr<f64, Infix3Ops, 3>,
    y: PostfixExpr<f64, Infix3Ops, 3>,
    z: PostfixExpr<f64, Infix3Ops, 3>,
) -> dynamic_expressions::PostfixExpr<f64, Infix3Ops, 3> {
    dynamic_expressions::expression_algebra::__apply_postfix::<f64, Infix3Ops, 3, 3>(
        <Infix3Ops as HasOp<Oplus>>::ID,
        [x, y, z],
    )
}

#[test]
fn string_tree_matches_expected() {
    let ex = expr_readme_like();
    let s = string_tree(&ex, &common::TestOps::new(), StringTreeOptions::default());
    assert_eq!(s, "x0 * cos(x1 - 3.2)");
}

#[test]
fn string_tree_uses_variable_names() {
    let mut ex = expr_readme_like();
    ex.meta.variable_names = vec!["x".into(), "y".into()];
    let s = string_tree(&ex, &common::TestOps::new(), StringTreeOptions::default());
    assert_eq!(s, "x * cos(y - 3.2)");
}

#[test]
fn infix_preserves_child_parens_for_precedence() {
    let mut a = common::var(0);
    a.meta.variable_names = vec!["a".into(), "b".into(), "c".into()];
    let b = common::var(1);
    let c = common::var(2);

    let expr = (a + b) * c;
    let s = string_tree(&expr, &common::TestOps::new(), StringTreeOptions::default());
    assert_eq!(s, "(a + b) * c");
}

#[test]
fn ternary_infix_custom_op_preserves_child_grouping() {
    let mut a = var_infix3(0);
    a.meta.variable_names = vec!["a".into(), "b".into(), "c".into()];
    let b = var_infix3(1);
    let c = var_infix3(2);

    // (a + b) ⊕ (b + c) ⊕ a
    //
    // If the "infix join" path strips child outer parens, this can degrade to:
    // a + b ⊕ b + c ⊕ a  (ambiguous / wrong-looking).
    let expr = oplus3(add_infix3(a.clone(), b.clone()), add_infix3(b, c), a);

    let s = string_tree(&expr, &Infix3Ops::new(), StringTreeOptions::default());
    assert_eq!(s, "(a + b) ⊕ (b + c) ⊕ a");
}

#[test]
fn default_string_variable_uses_name_and_falls_back() {
    let names = vec!["hello".to_string()];
    assert_eq!(
        dynamic_expressions::strings::default_string_variable(0, Some(&names)),
        "hello"
    );

    let empty: Vec<String> = vec![];
    assert_eq!(
        dynamic_expressions::strings::default_string_variable(0, Some(&empty)),
        "x0"
    );
}

#[test]
fn symbolic_tokens_print_infix_and_names_print_as_calls() {
    use dynamic_expressions::operator_enum::builtin;
    use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;

    type Ops = BuiltinOpsF64;
    let ops = Ops::new();
    let binary = |id: u16| -> PostfixExpr<f64, Ops, 3> {
        PostfixExpr::new(
            vec![
                PNode::Var { feature: 0 },
                PNode::Var { feature: 1 },
                PNode::Op { arity: 2, op: id },
            ],
            vec![],
            Metadata::default(),
        )
    };
    let pretty = || StringTreeOptions {
        pretty: true,
        ..Default::default()
    };

    let greater = binary(<Ops as HasOp<builtin::Greater>>::ID);
    assert_eq!(
        string_tree(&greater, &ops, StringTreeOptions::default()),
        "greater(x0, x1)"
    );
    assert_eq!(string_tree(&greater, &ops, pretty()), "x0 > x1");

    let add = binary(<Ops as HasOp<builtin::Add>>::ID);
    assert_eq!(string_tree(&add, &ops, StringTreeOptions::default()), "x0 + x1");
    assert_eq!(string_tree(&add, &ops, pretty()), "x0 + x1");

    let pow = binary(<Ops as HasOp<builtin::Pow>>::ID);
    assert_eq!(string_tree(&pow, &ops, StringTreeOptions::default()), "x0 ^ x1");

    let and = binary(<Ops as HasOp<builtin::LogicalAnd>>::ID);
    assert_eq!(
        string_tree(&and, &ops, StringTreeOptions::default()),
        "logical_and(x0, x1)"
    );
    assert_eq!(string_tree(&and, &ops, pretty()), "x0 && x1");

    let atan2 = binary(<Ops as HasOp<builtin::Atan2>>::ID);
    assert_eq!(string_tree(&atan2, &ops, pretty()), "atan2(x0, x1)");
}
