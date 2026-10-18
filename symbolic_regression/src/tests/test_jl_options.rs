//! Ports of SymbolicRegression.jl testitems about operator options (see tools/parity_tests.toml).

use dynamic_expressions::dsl::CustomOps;
use dynamic_expressions::expression_algebra::__apply_postfix;
use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
use dynamic_expressions::{LookupError, Metadata, OperatorSet, PNode, PostfixExpr};

use crate::Options;
use crate::check_constraints::check_constraints;

type Ops = BuiltinOpsF64<CustomOps<f64>>;

fn with_max5() -> Ops {
    let mut custom = CustomOps::default();
    custom.define("max5(a, b, c, d, e) = max(a, b, c, d, e)").unwrap();
    BuiltinOpsF64::new().with_custom(custom)
}

fn x<const D: usize>(feature: u16) -> PostfixExpr<f64, Ops, D> {
    PostfixExpr::new(vec![PNode::Var { feature }], vec![], Metadata::default())
}

fn call<const D: usize, const A: usize>(
    ops: &Ops,
    name: &str,
    args: [PostfixExpr<f64, Ops, D>; A],
) -> PostfixExpr<f64, Ops, D> {
    let op = ops.lookup_with_arity(name, A as u8).unwrap();
    __apply_postfix::<f64, Ops, D, A>(op.id, args)
}

/// test/unit/basic/test_options.jl | "Test with_max_degree_from_context"
#[test]
fn max_arity_follows_operator_set() {
    let builtin = BuiltinOpsF64::new();
    let highest_arity = |names: &[&str]| {
        let selected = builtin.from_names::<3, _>(names.iter().copied()).unwrap();
        let nops = [1, 2, 3].map(|arity| selected.nops(arity));
        (nops, (1..=3).rev().find(|&arity| selected.nops(arity) > 0))
    };
    assert_eq!(highest_arity(&["sin", "cos", "+", "*", "sub"]), ([2, 3, 0], Some(2)));
    assert_eq!(highest_arity(&["sin", "cos"]), ([2, 0, 0], Some(1)));
    assert_eq!(
        highest_arity(&["sin", "cos", "+", "*", "sub", "fma", "clamp"]),
        ([2, 3, 2], Some(3))
    );

    assert_eq!(builtin.max_arity(), 3);
    let ops = with_max5();
    assert_eq!(ops.max_arity(), 5);
    assert_eq!(ops.ops_with_arity(5).count(), 1);
    assert_eq!(ops.ops_with_arity(4).count(), 0);
    let options = Options::<f64, Ops, 5> {
        operators: ops.from_names(["sin", "+", "max5"]).unwrap(),
        ..Default::default()
    };
    for op in options.operators.ops_by_arity.iter().flatten() {
        assert!(!options.op_constraints.limits.contains_key(op));
    }
}

/// test/unit/basic/test_options.jl | "Test operator appears in multiple degrees error"
#[test]
fn operator_in_two_arities_needs_an_arity() {
    let ops = BuiltinOpsF64::new();
    match ops.lookup("-") {
        Err(LookupError::Ambiguous { candidates, .. }) => {
            assert_eq!(candidates, vec!["neg".to_string(), "sub".to_string()]);
        }
        other => panic!("expected an ambiguity, got {other:?}"),
    }
    let sub = ops.lookup_with_arity("-", 2).unwrap();
    assert_eq!(ops.name(sub), "sub");

    let operator_set = BuiltinOpsF64::new().with_custom(CustomOps::default());
    let mut options = Options::<f64, Ops, 3> {
        operators: operator_set.from_names(["neg", "sub", "+"]).unwrap(),
        ..Default::default()
    };
    let sub = options.operators.set.lookup_with_arity("-", 2).unwrap();
    options.op_constraints.set_op_arg_constraint(sub, 0, 1);
    let ops = &options.operators.set;
    let neg_of_sum = call(ops, "neg", [x(0) + x(1)]);
    let sub_of_sum = call(ops, "sub", [x(0) + x(1), x(1)]);
    assert!(check_constraints(&neg_of_sum, &options, options.maxsize));
    assert!(!check_constraints(&sub_of_sum, &options, options.maxsize));
}

/// test/unit/basic/test_options.jl | "Test build_constraints with pre-processed vector format"
#[test]
fn default_constraints_cover_every_arity() {
    let operator_set = with_max5();
    let mut options = Options::<f64, Ops, 5> {
        operators: operator_set
            .from_names(["sin", "cos", "+", "*", "sub", "max5"])
            .unwrap(),
        ..Default::default()
    };
    let ops = &options.operators.set;
    let simple = call(ops, "max5", [call(ops, "sin", [x(0)]), x(1), x(0), x(1), x(0)]);
    let nested_last = call(ops, "max5", [x(0), x(1), x(0), x(1), x(0) + x(1)]);
    assert!(check_constraints(&simple, &options, options.maxsize));
    assert!(check_constraints(&nested_last, &options, options.maxsize));

    let max5 = options.operators.set.lookup("max5").unwrap();
    options.op_constraints.set_op_arg_constraint(max5, 4, 1);
    assert!(check_constraints(&simple, &options, options.maxsize));
    assert!(!check_constraints(&nested_last, &options, options.maxsize));
}
