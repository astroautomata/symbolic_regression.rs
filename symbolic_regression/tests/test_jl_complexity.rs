//! Ports of SymbolicRegression.jl testitems about complexity (see tools/parity_tests.toml).

use dynamic_expressions::expression_algebra::__apply_postfix;
use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
use dynamic_expressions::operators::sin;
use dynamic_expressions::{Metadata, OperatorSet, PNode, PostfixExpr};
use symbolic_regression::{Options, compute_complexity};

type Ops = BuiltinOpsF64;
type Expr = PostfixExpr<f64, Ops, 3>;

fn x(feature: u16) -> Expr {
    PostfixExpr::new(vec![PNode::Var { feature }], vec![], Metadata::default())
}

fn c(value: f64) -> Expr {
    PostfixExpr::new(vec![PNode::Const { idx: 0 }], vec![value], Metadata::default())
}

/// `sin((x1 + x2 + x3)^2.3)`
fn tree() -> Expr {
    let pow = Ops::new().lookup("^").unwrap();
    sin(__apply_postfix::<f64, Ops, 3, 2>(pow.id, [x(0) + x(1) + x(2), c(2.3)]))
}

fn options(overrides: &[(&str, f32)]) -> Options<f64, Ops, 3> {
    let operator_set = Ops::new();
    let mut options = Options::<f64, Ops, 3> {
        operators: operator_set
            .from_names(["+", "sub", "*", "/", "^", "cos", "sin"])
            .unwrap(),
        ..Default::default()
    };
    for &(name, cost) in overrides {
        let op = options.operators.set.lookup(name).unwrap();
        options.operator_complexity_overrides.insert(op, cost);
    }
    options
}

/// test/unit/operators-core/test_complexity.jl | "Test complexity evaluation"
#[test]
fn complexity_with_operator_overrides() {
    let tree = tree();
    assert_eq!(compute_complexity(&tree.nodes, &options(&[])), 8);
    assert_eq!(compute_complexity(&tree.nodes, &options(&[("sin", 3.0)])), 10);
    assert_eq!(
        compute_complexity(&tree.nodes, &options(&[("sin", 3.0), ("+", 2.0)])),
        12
    );
    let real_costs = options(&[("sin", 3.0), ("+", 2.0), ("^", 3.2)]);
    assert_eq!(compute_complexity(&tree.nodes, &real_costs), 14);
}

/// test/unit/operators-core/test_complexity.jl | "Test other things about complexity"
#[test]
fn complexity_with_variable_and_constant_costs() {
    let tree = tree();
    let mut opts = options(&[("sin", 3.0), ("+", 2.0)]);
    opts.complexity_of_variables = 2.0;
    assert_eq!(compute_complexity(&tree.nodes, &opts), 12 + 3);
    opts.complexity_of_constants = 2.0;
    assert_eq!(compute_complexity(&tree.nodes, &opts), 12 + 3 + 1);
    opts.complexity_of_constants = 2.6;
    assert_eq!(compute_complexity(&tree.nodes, &opts), 12 + 3 + 1 + 1);

    let tree = x(0) + x(1) * x(2);
    let mut opts = options(&[("+", 5.0), ("*", 2.0)]);
    opts.variable_complexities = Some(vec![1.0, 2.0, 3.0]);
    assert_eq!(compute_complexity(&tree.nodes, &opts), 1 + 5 + 2 + 2 + 3);
    let mut opts = options(&[("+", 5.0), ("*", 2.0)]);
    opts.complexity_of_variables = 2.0;
    assert_eq!(compute_complexity(&tree.nodes, &opts), 2 + 5 + 2 + 2 + 2);
}
