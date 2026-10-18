//! Ports of SymbolicRegression.jl testitems about built-in operators (see tools/parity_tests.toml).

use dynamic_expressions::dsl::CustomOps;
use dynamic_expressions::expression_algebra::__apply_postfix;
use dynamic_expressions::operator_enum::builtin::*;
use dynamic_expressions::operator_enum::presets::{BuiltinOpsF32, BuiltinOpsF64};
use dynamic_expressions::{
    EvalOptions, Metadata, Operator, OperatorSet, Operators, PNode, PostfixExpr, eval_tree_array,
};
use ndarray::Array2;
use num_traits::Float;

fn un<T: Float, Op: Operator<T, 1>>(x: T) -> T {
    Op::eval(&[x])
}

fn bin<T: Float, Op: Operator<T, 2>>(x: T, y: T) -> T {
    Op::eval(&[x, y])
}

fn generic_semantics<T: Float + core::fmt::Debug>() {
    let t = |v: f64| T::from(v).unwrap();
    let close = |a: T, b: T| (a - b).abs() < t(1e-6);
    let (val, val2) = (t(0.5), t(3.2));
    assert!(close(un::<T, Log>(val), val.ln()));
    assert!(un::<T, Log>(-val).is_nan());
    assert!(close(un::<T, Log2>(val), val.log2()));
    assert!(un::<T, Log2>(-val).is_nan());
    assert!(close(un::<T, Log10>(val), val.log10()));
    assert!(un::<T, Log10>(-val).is_nan());
    assert!(close(un::<T, Log1p>(val), val.ln_1p()));
    assert!(close(un::<T, Acosh>(val2), val2.acosh()));
    assert!(un::<T, Acosh>(-val2).is_nan());
    assert!(close(un::<T, Asin>(val), val.asin()));
    assert!(un::<T, Asin>(val2).is_nan());
    assert!(close(un::<T, Acos>(val), val.acos()));
    assert!(un::<T, Acos>(val2).is_nan());
    assert!(close(un::<T, Atanh>(val), val.atanh()));
    assert!(un::<T, Atanh>(val2).is_nan());
    assert_eq!(un::<T, Neg>(-val), val);
    assert_eq!(un::<T, Sqrt>(val), val.sqrt());
    assert!(un::<T, Sqrt>(-val).is_nan());
    assert_eq!(bin::<T, Mul>(val, val2), val * val2);
    assert_eq!(bin::<T, Add>(val, val2), val + val2);
    assert_eq!(bin::<T, Sub>(val, val2), val - val2);
    assert_eq!(un::<T, Square>(val), val * val);
    assert_eq!(un::<T, Cube>(val), val * val * val);
    assert!(bin::<T, Pow>(t(0.0), t(-1.0)).is_nan());
    assert!(bin::<T, Pow>(-val, val2).is_nan());
    assert!(bin::<T, Pow>(-val, -val2).is_nan());
    assert!(bin::<T, Pow>(t(0.0), -val2).is_nan());
    assert!(close(bin::<T, Pow>(val, val2), val.powf(val2)));
    assert!(close(bin::<T, Pow>(val, -val2), val.powf(-val2)));
    assert!(!bin::<T, Pow>(t(-1.0), t(2.0)).is_nan());
    assert!(bin::<T, Pow>(t(-1.0), t(2.1)).is_nan());
    assert!(un::<T, Log>(t(0.0)).is_nan());
    assert!(un::<T, Log2>(t(0.0)).is_nan());
    assert!(un::<T, Log10>(t(0.0)).is_nan());
    assert!(un::<T, Log1p>(t(-2.0)).is_nan());
    assert_eq!(bin::<T, Greater>(val, val2), t(0.0));
    assert_eq!(bin::<T, Greater>(val2, val), t(1.0));
    assert_eq!(un::<T, Relu>(-val), t(0.0));
    assert_eq!(un::<T, Relu>(val), val);
    assert_eq!(bin::<T, LogicalOr>(val, val2), t(1.0));
    assert_eq!(bin::<T, LogicalOr>(t(0.0), val2), t(1.0));
    assert_eq!(bin::<T, LogicalAnd>(t(0.0), val2), t(0.0));
    assert_eq!(bin::<T, Cond>(val, val2), val2);
    assert_eq!(bin::<T, Cond>(-val, val2), t(0.0));
}

/// test/integration/ext/loopvectorization/test_operators.jl | "Generic operator tests"
#[test]
fn generic_operator_semantics() {
    generic_semantics::<f32>();
    generic_semantics::<f64>();
}

const SR_NAMES: [&str; 25] = [
    "plus",
    "sub",
    "mult",
    "/",
    "^",
    "greater",
    "less",
    "greater_equal",
    "less_equal",
    "logical_or",
    "logical_and",
    "cond",
    "square",
    "cube",
    "log",
    "log2",
    "log10",
    "log1p",
    "sqrt",
    "asin",
    "acos",
    "atanh",
    "acosh",
    "neg",
    "relu",
];

fn every_selected_op_evaluates<Ops: OperatorSet>(ops: Ops) {
    let selected: Operators<Ops, 3> = Operators::from_names(ops, SR_NAMES).unwrap();
    let ops = &selected.set;
    assert_eq!(selected.nops(1) + selected.nops(2), SR_NAMES.len());
    let mut rng = fastrand::Rng::with_seed(7);
    let data: Array2<Ops::T> = Array2::from_shape_fn((2, 64), |_| {
        <Ops::T as num_traits::NumCast>::from(rng.f64() * 6.0 - 3.0).unwrap()
    });
    let leaf = |feature| PostfixExpr::<Ops::T, Ops, 3>::new(vec![PNode::Var { feature }], vec![], Metadata::default());
    for (arity, ids) in (1u8..).zip(&selected.ops_by_arity) {
        for op in ids {
            let tree = match arity {
                1 => __apply_postfix::<Ops::T, Ops, 3, 1>(op.id, [leaf(0)]),
                _ => __apply_postfix::<Ops::T, Ops, 3, 2>(op.id, [leaf(0), leaf(1)]),
            };
            let opts = EvalOptions {
                check_finite: true,
                early_exit: false,
            };
            let (out, _) = eval_tree_array(&tree, ops, data.view(), &opts);
            assert!(
                out.iter().all(|v| !v.is_infinite()),
                "{} returned an infinity",
                ops.name(*op)
            );
        }
    }
}

/// test/integration/ext/loopvectorization/test_operators.jl | "Built-in operators pass validation"
#[test]
fn builtin_operator_names_resolve_and_evaluate() {
    let ops = BuiltinOpsF64::new();
    assert_eq!(ops.lookup("^").unwrap(), ops.lookup("pow").unwrap());
    assert_eq!(ops.lookup("plus").unwrap(), ops.lookup("+").unwrap());
    assert_eq!(ops.lookup("mult").unwrap(), ops.lookup("*").unwrap());
    for (name, display) in [
        ("greater", ">"),
        ("less", "<"),
        ("greater_equal", ">="),
        ("less_equal", "<="),
    ] {
        assert_eq!(ops.display(ops.lookup(name).unwrap()), display);
    }
    every_selected_op_evaluates(BuiltinOpsF32::new());
    every_selected_op_evaluates(ops);
}

/// test/integration/ext/loopvectorization/test_operators.jl | "Operators with wrong type fail"
#[test]
fn custom_operator_returning_bool_is_rejected() {
    let mut custom = CustomOps::<f64>::default();
    let err = custom.define("my_bad_op(x) = x > 0").unwrap_err();
    assert_eq!(
        err.message,
        "operator `my_bad_op` returns Bool; custom operators must return a number"
    );
    custom.define("my_ok_op(x) = (x > 0) * x").unwrap();
}

/// test/integration/ext/loopvectorization/test_operators.jl | "user_provided_operators applies safe operator mappings"
#[test]
fn selected_operators_use_safe_semantics() {
    type E = PostfixExpr<f64, BuiltinOpsF64, 3>;
    let ops = BuiltinOpsF64::new();
    let x0 = || E::new(vec![PNode::Var { feature: 0 }], vec![], Metadata::default());
    let c = |v: f64| E::new(vec![PNode::Const { idx: 0 }], vec![v], Metadata::default());
    let id = |name: &str, arity: u8| ops.lookup_with_arity(name, arity).unwrap().id;
    let cases: [(E, f64); 4] = [
        (__apply_postfix::<f64, BuiltinOpsF64, 3, 1>(id("log", 1), [x0()]), -1.0),
        (__apply_postfix::<f64, BuiltinOpsF64, 3, 1>(id("log", 1), [x0()]), 0.0),
        (__apply_postfix::<f64, BuiltinOpsF64, 3, 1>(id("sqrt", 1), [x0()]), -1.0),
        (
            __apply_postfix::<f64, BuiltinOpsF64, 3, 2>(id("^", 2), [x0(), c(-1.0)]),
            0.0,
        ),
    ];
    let opts = EvalOptions {
        check_finite: true,
        early_exit: false,
    };
    for (tree, input) in cases {
        let data = Array2::from_elem((1, 4), input);
        let (out, complete) = eval_tree_array(&tree, &ops, data.view(), &opts);
        assert!(out.iter().all(|v| v.is_nan()), "input {input}: {out:?}");
        assert!(!complete, "input {input}");
    }
}
