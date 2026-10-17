mod common;

use common::*;
use dynamic_expressions::{EvalOptions, eval_tree_array};
use rstest::rstest;

#[rstest]
#[case(true, true)]
#[case(true, false)]
#[case(false, false)]
fn eval_readme_like_matches_manual(#[case] check_finite: bool, #[case] early_exit: bool) {
    let expr = expr_readme_like();
    let (x_data, x) = make_x(2, 100);
    let opts = EvalOptions {
        check_finite,
        early_exit,
    };

    let n_rows = x.ncols();
    let mut out = vec![0.0f64; n_rows];
    let mut ctx = dynamic_expressions::EvalContext::<f64, 3>::new(n_rows);
    let ok = dynamic_expressions::eval_tree_array_into::<f64, TestOps, 3>(
        &mut out,
        &expr,
        &TestOps::new(),
        x.view(),
        &mut ctx,
        &opts,
    );

    // manual
    let mut manual = vec![0.0f64; n_rows];
    for row in 0..n_rows {
        let x1 = x_data[row];
        let x2 = x_data[n_rows + row];
        manual[row] = x1 * (x2 - 3.2).cos();
    }
    if check_finite {
        assert!(ok);
    }
    assert_close_vec(&out, &manual, 1e-12);
}

#[test]
fn eval_constant_only_fill() {
    let expr = c(1.25);
    let (_x_data, x) = make_x(2, 17);
    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };
    let (out, ok) = eval_tree_array::<f64, TestOps, 3>(&expr, &TestOps::new(), x.view(), &opts);
    assert!(ok);
    for v in out {
        assert_eq!(v, 1.25);
    }
}

#[test]
fn nan_detection_sets_complete_false() {
    // x1 / 0.0
    let expr = var(0) / 0.0;
    let (_x_data, x) = make_x(1, 10);
    let opts = EvalOptions {
        check_finite: true,
        early_exit: false,
    };
    let (_out, ok) = eval_tree_array::<f64, TestOps, 3>(&expr, &TestOps::new(), x.view(), &opts);
    assert!(!ok);
}

/// SymbolicRegression.jl rejects an expression whose operators read a non-finite feature or
/// constant, even where the value is finite, as for `relu(NaN) = 0`. Validity does not depend on
/// `early_exit`, which only decides whether evaluation stops, nor on whether the columns were
/// scanned up front with `XColumns::checked`, and the derivative evaluators agree.
#[test]
fn nonfinite_leaves_make_the_evaluation_incomplete() {
    use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
    use dynamic_expressions::{
        DiffContext, GradContext, Metadata, OperatorSet, PNode, PostfixExpr, XColumns, eval_diff_tree_array,
        eval_grad_tree_array,
    };
    let ops = BuiltinOpsF64::new();
    let relu = ops.lookup_with_arity("relu", 1).unwrap().id;
    let relu_of = |leaf: PNode, consts: Vec<f64>| {
        PostfixExpr::<f64, BuiltinOpsF64, 3>::new(
            vec![leaf, PNode::Op { arity: 1, op: relu }],
            consts,
            Metadata::default(),
        )
    };
    let finite = ndarray::Array2::from_shape_vec((1, 2), vec![-1.0, 2.0]).unwrap();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let x = ndarray::Array2::from_shape_vec((1, 2), vec![bad, 2.0]).unwrap();
        let cases = [
            (relu_of(PNode::Var { feature: 0 }, vec![]), x.view()),
            (relu_of(PNode::Const { idx: 0 }, vec![bad]), finite.view()),
        ];
        for (expr, x) in cases {
            for x in [XColumns::from(x), XColumns::checked(x)] {
                for (check_finite, early_exit) in [(true, true), (true, false), (false, false)] {
                    let opts = EvalOptions {
                        check_finite,
                        early_exit,
                    };
                    let what = format!(
                        "{bad} in {:?}, {x:?}, check_finite {check_finite}, early_exit {early_exit}",
                        expr.nodes
                    );
                    let (out, ok) = eval_tree_array(&expr, &ops, x, &opts);
                    assert_eq!(ok, !check_finite, "{what}");
                    if !early_exit && expr.consts.is_empty() {
                        assert_eq!(out[1], 2.0, "{what}");
                    }
                    let (_, _, ok) = eval_grad_tree_array(&expr, &ops, x, true, &mut GradContext::new(2), &opts);
                    assert_eq!(ok, !check_finite, "grad: {what}");
                    let (_, _, ok) = eval_diff_tree_array(&expr, &ops, x, 0, &mut DiffContext::new(2), &opts);
                    assert_eq!(ok, !check_finite, "diff: {what}");
                }
            }
        }
    }
}

/// A non-finite value in a feature the expression does not read leaves it valid, whether or not
/// the columns were scanned up front.
#[test]
fn unread_nonfinite_features_keep_the_evaluation_complete() {
    use dynamic_expressions::XColumns;
    let expr = var(0) * 2.0;
    let x = ndarray::Array2::from_shape_vec((2, 2), vec![1.0, 2.0, f64::NAN, 3.0]).unwrap();
    let opts = EvalOptions::default();
    for x in [XColumns::from(x.view()), XColumns::checked(x.view())] {
        let (out, ok) = eval_tree_array::<f64, TestOps, 3>(&expr, &TestOps::new(), x, &opts);
        assert!(ok, "{x:?}");
        assert_eq!(out, vec![2.0, 4.0]);
    }
}
