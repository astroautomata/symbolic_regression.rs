//! Ports of SymbolicRegression.jl testitems about printing (see tools/parity_tests.toml).

use dynamic_expressions::expression_algebra::__apply_postfix;
use dynamic_expressions::operator_enum::builtin::*;
use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
use dynamic_expressions::operators::{cos, sin};
use dynamic_expressions::strings::{StringTreeOptions, string_tree};
use dynamic_expressions::{Metadata, OperatorSet, PNode, PostfixExpr};

type Expr<Ops> = PostfixExpr<f64, Ops, 3>;

fn var<Ops>(feature: u16) -> Expr<Ops> {
    PostfixExpr::new(vec![PNode::Var { feature }], vec![], Metadata::default())
}

fn call<Ops: OperatorSet, const A: usize>(ops: &Ops, name: &str, args: [Expr<Ops>; A]) -> Expr<Ops> {
    let op = ops
        .lookup_with_arity(name, A as u8)
        .unwrap_or_else(|e| panic!("{name}: {e:?}"));
    __apply_postfix::<f64, Ops, 3, A>(op.id, args)
}

fn named<Ops>(mut expr: Expr<Ops>, names: &[&str]) -> Expr<Ops> {
    expr.meta.variable_names = names.iter().map(|s| s.to_string()).collect();
    expr
}

/// test/unit/basic/test_print.jl | "Test printing"
#[test]
fn printing_nested_operators_with_variable_names() {
    let ops = BuiltinOpsF64::new();
    let tree = || (sin(cos(sin(cos(var(0)) * var(2)) * 3.0) * -0.5) + 2.0) * 5.0;

    let s = string_tree(&named(tree(), &["x1", "x2", "x3"]), &ops, StringTreeOptions::default());
    assert_eq!(s, "(sin(cos(sin(cos(x1) * x3) * 3) * -0.5) + 2) * 5");
    let s = string_tree(&named(tree(), &["v1", "v2", "v3"]), &ops, StringTreeOptions::default());
    assert_eq!(s, "(sin(cos(sin(cos(v1) * v3) * 3) * -0.5) + 2) * 5");

    for name in ["log", "log2", "log10", "log1p", "sqrt", "acosh"] {
        let tree = named(call(&ops, name, [var(0)]), &["x1"]);
        assert_eq!(
            string_tree(&tree, &ops, StringTreeOptions::default()),
            format!("{name}(x1)")
        );
    }
    for name in ["pow", "^"] {
        let tree = named(call(&ops, name, [var(0), var(1)]), &["x1", "x2"]);
        assert_eq!(string_tree(&tree, &ops, StringTreeOptions::default()), "x1 ^ x2");
    }
}

dynamic_expressions::opset! {
    pub FirstOps for f64 { Cos, Add, Sub, Mul, Div }
}

dynamic_expressions::opset! {
    pub SecondOps for f64 { Sin, Div, Mul, Sub, Add }
}

/// test/unit/basic/test_pretty_printing.jl | "pretty print expression"
#[test]
fn print_uses_the_given_operator_set() {
    let first = FirstOps::new();
    let ex: Expr<FirstOps> = named(cos(var(0)) + var(1) * var(1), &["x", "y"]);
    assert_eq!(
        string_tree(&ex, &first, StringTreeOptions::default()),
        "cos(x) + (y * y)"
    );

    let unnamed = PostfixExpr::<f64, FirstOps, 3>::new(ex.nodes.clone(), vec![], Metadata::default());
    assert_eq!(
        string_tree(&unnamed, &first, StringTreeOptions::default()),
        "cos(x0) + (x1 * x1)"
    );

    let same_ids = PostfixExpr::<f64, SecondOps, 3>::new(ex.nodes.clone(), vec![], ex.meta.clone());
    assert_eq!(
        string_tree(&same_ids, &SecondOps::new(), StringTreeOptions::default()),
        "sin(x) / (y - y)"
    );
}

/// test/unit/basic/test_pretty_printing.jl | "pretty print vs serialization for comparison operators"
#[test]
fn comparison_operators_pretty_vs_serialized() {
    let ops = BuiltinOpsF64::new();
    let pretty = StringTreeOptions {
        pretty: true,
        ..Default::default()
    };
    for (name, symbol) in [("greater", ">"), ("greater_equal", ">=")] {
        let ex = named(call(&ops, name, [var(0), var(1)]), &["x1", "x2"]);
        assert_eq!(string_tree(&ex, &ops, pretty.clone()), format!("x1 {symbol} x2"));
        assert_eq!(
            string_tree(&ex, &ops, StringTreeOptions::default()),
            format!("{name}(x1, x2)")
        );
    }
}
