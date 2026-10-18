//! Ports of SymbolicRegression.jl testitems about evaluation (see tools/parity_tests.toml).

use dynamic_expressions::expression_algebra::__apply_postfix;
use dynamic_expressions::operator_enum::builtin::*;
use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
use dynamic_expressions::operators::{cos, sin};
use dynamic_expressions::{EvalOptions, Metadata, OperatorSet, PNode, PostfixExpr, eval_tree_array};
use ndarray::Array2;

dynamic_expressions::opset! {
    pub EvalOps for f32 { Cos, Sin, Add, Sub, Mul, Div }
}

type Expr = PostfixExpr<f32, EvalOps, 2>;

fn x(feature: u16) -> Expr {
    PostfixExpr::new(vec![PNode::Var { feature }], vec![], Metadata::default())
}

fn c(value: f32) -> Expr {
    PostfixExpr::new(vec![PNode::Const { idx: 0 }], vec![value], Metadata::default())
}

/// test/unit/evaluation/test_evaluation.jl | "Test validity of expression evaluation"
#[test]
fn fused_evaluation_paths_match_closures() {
    type Case = (fn() -> Expr, fn(f32, f32, f32) -> f32);
    let cases: [Case; 15] = [
        (|| x(0) * x(1), |x1, x2, _| x1 * x2),
        (|| x(0) * 3.0, |x1, _, _| x1 * 3.0),
        (|| c(3.0) * x(1), |_, x2, _| 3.0 * x2),
        (|| c(3.0) * 6.0, |_, _, _| 3.0 * 6.0),
        (|| x(0) * sin(x(1)), |x1, x2, _| x1 * x2.sin()),
        (|| c(3.0) * sin(x(1)), |_, x2, _| 3.0 * x2.sin()),
        (|| sin(x(0)) * x(1), |x1, x2, _| x1.sin() * x2),
        (|| sin(x(0)) * 3.0, |x1, _, _| x1.sin() * 3.0),
        (|| cos(x(0) * x(1)), |x1, x2, _| (x1 * x2).cos()),
        (|| cos(x(0) * 3.0), |x1, _, _| (x1 * 3.0).cos()),
        (|| cos(c(3.0) * x(1)), |_, x2, _| (3.0 * x2).cos()),
        (|| cos(c(3.0) * -0.5), |_, _, _| (3.0f32 * -0.5).cos()),
        (|| cos(sin(x(0))), |x1, _, _| x1.sin().cos()),
        (|| cos(sin(c(3.0))), |_, _, _| 3.0f32.sin().cos()),
        (
            || (sin(cos(sin(cos(x(0)) * x(2)) * 3.0) * -0.5) + 2.0) * 5.0,
            |x1, _, x3| ((((x1.cos() * x3).sin() * 3.0).cos() * -0.5).sin() + 2.0) * 5.0,
        ),
    ];

    let n = 100;
    let mut rng = fastrand::Rng::with_seed(0);
    let data = Array2::from_shape_fn((3, n), |_| rng.f32() * 4.0 - 2.0);
    let ops = EvalOps::new();
    for (i, (tree, truth)) in cases.iter().enumerate() {
        let (got, _) = eval_tree_array(&tree(), &ops, data.view(), &EvalOptions::default());
        for (r, &g) in got.iter().enumerate() {
            let want = truth(data[(0, r)], data[(1, r)], data[(2, r)]);
            assert!((g - want).abs() / n as f32 <= 1e-6, "case {i}, row {r}: {g} vs {want}");
        }
    }
}

/// test/unit/misc/test_hash.jl | "Test hash of tree"
#[test]
fn hash_of_tree() {
    type E = PostfixExpr<f64, BuiltinOpsF64, 3>;
    let ops = BuiltinOpsF64::new();
    let leaf = |node: PNode, consts: Vec<f64>| E::new(vec![node], consts, Metadata::default());
    let call = |name: &str, args: [E; 2]| {
        let op = ops.lookup_with_arity(name, 2).unwrap();
        __apply_postfix::<f64, BuiltinOpsF64, 3, 2>(op.id, args)
    };
    let tree = |root: &str| {
        let inner = leaf(PNode::Const { idx: 0 }, vec![3.0]) * cos(leaf(PNode::Var { feature: 0 }, vec![]));
        let squared = call("^", [inner, leaf(PNode::Const { idx: 0 }, vec![2.0])]);
        call(root, [squared, leaf(PNode::Const { idx: 0 }, vec![-1.2])])
    };
    let pow = tree("^");
    assert_eq!(pow.hash_nodes(), pow.clone().hash_nodes());
    assert_ne!(pow.hash_nodes(), tree("greater").hash_nodes());
}
