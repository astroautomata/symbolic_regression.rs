//! Ports of SymbolicRegression.jl testitems about custom operators (see tools/parity_tests.toml).

use dynamic_expressions::dsl::CustomOps;
use dynamic_expressions::expression_algebra::__apply_postfix;
use dynamic_expressions::operator_enum::presets::{BuiltinOpsF32, BuiltinOpsF64};
use dynamic_expressions::strings::{StringTreeOptions, string_tree};
use dynamic_expressions::{EvalOptions, Metadata, OperatorSet, PNode, PostfixExpr, eval_tree_array};
use ndarray::Array2;

type Ops32 = BuiltinOpsF32<CustomOps<f32>>;
type Ops64 = BuiltinOpsF64<CustomOps<f64>>;
type Expr<T, Ops> = PostfixExpr<T, Ops, 3>;

fn ops32(defs: &[&str]) -> Ops32 {
    let mut custom = CustomOps::default();
    for def in defs {
        custom.define(def).unwrap_or_else(|e| panic!("{def}: {e}"));
    }
    BuiltinOpsF32::new().with_custom(custom)
}

fn ops64(defs: &[&str]) -> Ops64 {
    let mut custom = CustomOps::default();
    for def in defs {
        custom.define(def).unwrap_or_else(|e| panic!("{def}: {e}"));
    }
    BuiltinOpsF64::new().with_custom(custom)
}

fn var<T, Ops>(feature: u16) -> Expr<T, Ops> {
    PostfixExpr::new(vec![PNode::Var { feature }], vec![], Metadata::default())
}

fn konst<T, Ops>(value: T) -> Expr<T, Ops> {
    PostfixExpr::new(vec![PNode::Const { idx: 0 }], vec![value], Metadata::default())
}

fn call<T, Ops: OperatorSet, const A: usize>(ops: &Ops, name: &str, args: [Expr<T, Ops>; A]) -> Expr<T, Ops> {
    let op = ops
        .lookup_with_arity(name, A as u8)
        .unwrap_or_else(|e| panic!("{name}: {e:?}"));
    __apply_postfix::<T, Ops, 3, A>(op.id, args)
}

fn data<T: num_traits::Float>(n_features: usize, n_rows: usize, seed: u64) -> Array2<T> {
    let mut rng = fastrand::Rng::with_seed(seed);
    Array2::from_shape_fn((n_features, n_rows), |_| T::from(rng.f64() * 4.0 - 2.0).unwrap())
}

/// test/unit/operators-core/test_custom_operators.jl | "Test custom operators"
#[test]
fn custom_operators_print_and_evaluate() {
    let ops = ops32(&[
        "op1(x, y) = x + y",
        "op2(x, y) = x^2 + 1 / ((y)^2 + 0.1)",
        "op3(x) = sin(x) + cos(x)",
    ]);
    let mut tree = call(
        &ops,
        "op1",
        [call(&ops, "op2", [var(0), var(1)]), call(&ops, "op3", [var(0)])],
    );
    tree.meta.variable_names = vec!["x1".into(), "x2".into()];
    assert_eq!(
        string_tree(&tree, &ops, StringTreeOptions::default()),
        "op1(op2(x1, x2), op3(x1))"
    );

    let x = data::<f32>(2, 10, 0);
    let (got, complete) = eval_tree_array(&tree, &ops, x.view(), &EvalOptions::default());
    assert!(complete);
    for (r, &g) in got.iter().enumerate() {
        let (x1, x2) = (x[(0, r)], x[(1, r)]);
        let want = (x1.powf(2.0) + 1.0 / (x2.powf(2.0) + 0.1)) + (x1.sin() + x1.cos());
        assert!((g - want).abs() <= 1e-5 * want.abs().max(1.0), "row {r}: {g} vs {want}");
    }

    let ops = ops64(&["my_func_a(x, y) = x^2 * y", "my_func_b(x) = x^3"]);
    let tree = call(
        &ops,
        "my_func_a",
        [
            call(&ops, "my_func_a", [var(1), konst(0.2)]),
            call(&ops, "my_func_b", [var(0)]),
        ],
    );
    let x = data::<f64>(2, 20, 1);
    let (got, complete) = eval_tree_array(&tree, &ops, x.view(), &EvalOptions::default());
    assert!(complete);
    for (r, &g) in got.iter().enumerate() {
        let (x1, x2) = (x[(0, r)], x[(1, r)]);
        let a = |x: f64, y: f64| x.powf(2.0) * y;
        let want = a(a(x2, 0.2), x1.powf(3.0));
        assert!(
            (g - want).abs() <= 1e-12 * want.abs().max(1.0),
            "row {r}: {g} vs {want}"
        );
    }

    let ops = ops64(&["my_func_c(x, y) = x * y + 0.3", "my_func_d(x) = x / (abs(x)^0.2 + 0.1)"]);
    let tree = call(
        &ops,
        "my_func_c",
        [
            call(&ops, "my_func_c", [var(1), konst(0.2)]),
            call(&ops, "my_func_d", [var(0)]),
        ],
    );
    let (got, complete) = eval_tree_array(&tree, &ops, x.view(), &EvalOptions::default());
    assert!(complete);
    for (r, &g) in got.iter().enumerate() {
        let (x1, x2) = (x[(0, r)], x[(1, r)]);
        let c = |x: f64, y: f64| x * y + 0.3;
        let want = c(c(x2, 0.2), x1 / (x1.abs().powf(0.2) + 0.1));
        assert!(
            (g - want).abs() <= 1e-12 * want.abs().max(1.0),
            "row {r}: {g} vs {want}"
        );
    }
}

/// test/unit/basic/test_options.jl | "Test operators stored globally"
#[test]
fn operator_sets_are_values_not_globals() {
    let plus_one = ops64(&["f(x) = x + 1"]);
    let times_two = ops64(&["f(x) = x * 2"]);
    let tree_a = call(&plus_one, "f", [var(0)]);
    let tree_b = call(&times_two, "f", [var(0)]);
    assert_eq!(tree_a.nodes, tree_b.nodes);

    let x = Array2::from_shape_vec((1, 2), vec![1.0, 3.0]).unwrap();
    let opts = EvalOptions::default();
    assert_eq!(eval_tree_array(&tree_a, &plus_one, x.view(), &opts).0, vec![2.0, 4.0]);
    assert_eq!(eval_tree_array(&tree_b, &times_two, x.view(), &opts).0, vec![2.0, 6.0]);
    assert_eq!(core::mem::size_of::<BuiltinOpsF64>(), 0);
}
