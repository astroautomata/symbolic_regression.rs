//! Ports of SymbolicRegression.jl testitems about template structures: compiling, evaluating, and
//! printing combine formulas (see tools/parity_tests.toml).

use std::panic::{AssertUnwindSafe, catch_unwind};

use dynamic_expressions::dsl::{TemplateStructure, TemplateWorkspace};
use dynamic_expressions::operator_enum::presets::{BuiltinOpsF32, BuiltinOpsF64};
use dynamic_expressions::operators::{cos, exp, log, sin};
use dynamic_expressions::strings::{StringTreeOptions, string_tree};
use dynamic_expressions::{Metadata, OperatorSet, PNode, PostfixExpr, lit};
use ndarray::Array2;

type Tree = PostfixExpr<f64, BuiltinOpsF64, 3>;

/// The subexpression argument `#i` (1-based).
fn arg(i: u16) -> Tree {
    PostfixExpr::new(vec![PNode::Var { feature: i - 1 }], vec![], Metadata::default())
}

fn arg32(i: u16) -> PostfixExpr<f32, BuiltinOpsF32, 3> {
    PostfixExpr::new(vec![PNode::Var { feature: i - 1 }], vec![], Metadata::default())
}

fn constant(v: f64) -> Tree {
    PostfixExpr::new(vec![PNode::Const { idx: 0 }], vec![v], Metadata::default())
}

fn template(src: &str, expressions: &[&str], variables: &[&str], parameters: &[(&str, usize)]) -> TemplateStructure {
    TemplateStructure::new(src, expressions, None, variables, parameters).unwrap_or_else(|e| panic!("`{src}`: {e}"))
}

fn error(
    src: &str,
    expressions: &[&str],
    num_features: Option<&[usize]>,
    variables: &[&str],
    parameters: &[(&str, usize)],
) -> String {
    TemplateStructure::new(src, expressions, num_features, variables, parameters)
        .map(|_| ())
        .expect_err(src)
        .message
}

/// Features by rows, one row per slice.
fn features<T: Copy>(rows: &[&[T]]) -> Array2<T> {
    Array2::from_shape_fn((rows.len(), rows[0].len()), |(f, i)| rows[f][i])
}

/// Evaluates on `rows`; `None` when SymbolicRegression.jl would return `nothing`.
fn eval(t: &TemplateStructure, trees: &[Tree], params: &[f64], rows: &[&[f64]]) -> Option<Vec<f64>> {
    let (out, ok) = t.eval(trees, params, &BuiltinOpsF64::new(), features(rows).view());
    ok.then_some(out)
}

fn assert_close(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() <= 1e-12 * w.abs().max(1.0), "{got:?} vs {want:?}");
    }
}

fn random(seed: u64, n: usize) -> Vec<f64> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..n).map(|_| rng.f64() * 4.0 - 2.0).collect()
}

/// test/integration/ext/loopvectorization/test_template_expression.jl | "Test invalid combiner functions"
/// test/integration/ext/mlj/templates/test_parametric_template_expressions.jl | "check_combiner_applicability errors"
#[test]
fn combine_formulas_that_do_not_match_the_declaration_fail_at_construction() {
    let msg = error("f(x) + p[1]", &["f"], None, &["x"], &[]);
    assert!(msg.contains("`p`"), "{msg}");
    let msg = error("f(x) + q[1]", &["f"], None, &["x"], &[("p", 1)]);
    assert!(msg.contains("`q`"), "{msg}");
    let msg = error("f(y)", &["f"], None, &["x"], &[]);
    assert!(msg.contains("`y`"), "{msg}");
    let msg = error("g(x)", &["f"], Some(&[1]), &["x"], &[]);
    assert!(msg.contains("`g`"), "{msg}");
}

/// test/integration/ext/loopvectorization/test_template_expression.jl | "Test get_variable_names"
#[test]
fn template_reads_its_variable_names() {
    let t = template("f(x1)", &["f"], &["x1", "x2"], &[]);
    assert_eq!(t.variable_names(), ["x1", "x2"]);
}

/// test/integration/ext/loopvectorization/test_template_expression.jl | "Test interface for TemplateExpression"
#[test]
fn nested_calls_infer_arities_and_print() {
    let t = template("f(f(f(x1))) - f(g(x2, x1))", &["f", "g"], &["x1", "x2", "x3"], &[]);
    assert_eq!(t.num_features(), [1, 2]);
    let trees = [arg(1), arg(2) * arg(2)];
    let ops = BuiltinOpsF64::new();
    assert_eq!(t.string(&trees, &[], &ops, false, None), "f = #1; g = #2 * #2");
    assert_eq!(t.string(&trees, &[], &ops, true, None), "╭ f = #1\n╰ g = #2 * #2");

    let (x1, x2, x3) = (random(1, 8), random(2, 8), random(3, 8));
    let want: Vec<f64> = x1.iter().map(|a| a - a * a).collect();
    assert_close(&eval(&t, &trees, &[], &[&x1, &x2, &x3]).unwrap(), &want);
}

/// test/integration/ext/loopvectorization/test_template_expression.jl | "Printing and evaluation of TemplateExpression"
#[test]
fn printing_and_evaluation() {
    let t = template("sin(f(x1, x2)) + g(x3)^2", &["f", "g"], &["x1", "x2", "x3"], &[]);
    let trees = [arg(1) * arg(2), arg(1)];
    assert_eq!(
        t.string(&trees, &[], &BuiltinOpsF64::new(), false, None),
        "f = #1 * #2; g = #1"
    );
    let (x1, x2, x3) = (random(1, 5), random(2, 5), random(3, 5));
    let want: Vec<f64> = (0..5).map(|i| (x1[i] * x2[i]).sin() + x3[i] * x3[i]).collect();
    assert_close(&eval(&t, &trees, &[], &[&x1, &x2, &x3]).unwrap(), &want);

    // The feature index is the argument position, whatever is passed.
    let call = |src, tree: Tree| {
        eval(
            &template(src, &["f"], &["x1", "x2", "x3"], &[]),
            &[tree],
            &[],
            &[&x1, &x2, &x3],
        )
    };
    assert_close(&call("f(x1)", arg(1)).unwrap(), &x1);
    assert_close(&call("f(x1, x2)", arg(2)).unwrap(), &x2);
    assert_close(&call("f(x2)", arg(1)).unwrap(), &x2);
    assert_close(&call("f(x3)", arg(1)).unwrap(), &x3);
    let x3_squared: Vec<f64> = x3.iter().map(|v| v * v).collect();
    assert_close(&call("f(x3, x3)", arg(1) * arg(2)).unwrap(), &x3_squared);
}

/// test/integration/ext/loopvectorization/test_template_expression.jl | "Test nothing return and type inference for
/// TemplateExpression"
#[test]
fn invalid_evaluations_return_nothing() {
    let t = template("1.0 + f(x1) / x1", &["f"], &["x1", "x2"], &[]);
    assert_eq!(eval(&t, &[arg(1)], &[], &[&[0.0], &[1.0]]), None);
    assert_close(&eval(&t, &[arg(1)], &[], &[&[1.0], &[2.0]]).unwrap(), &[2.0]);

    // An invalid argument makes the whole result invalid, even when the subexpression ignores it.
    let t = template("f(x1, log(x2))", &["f"], &["x1", "x2"], &[]);
    assert_close(&eval(&t, &[arg(1)], &[], &[&[1.0], &[2.0]]).unwrap(), &[1.0]);
    assert_eq!(eval(&t, &[arg(1)], &[], &[&[1.0], &[-1.0]]), None);
}

/// test/integration/ext/loopvectorization/test_template_expression.jl | "Test compatibility with power laws"
#[test]
fn power_of_negative_base_is_invalid() {
    let t = template("f(x1)^f(x2)", &["f"], &["x1", "x2"], &[]);
    let mut rng = fastrand::Rng::with_seed(0);
    let x1: Vec<f64> = (0..32).map(|_| -rng.f64()).collect();
    let x2: Vec<f64> = (0..32).map(|_| -rng.f64()).collect();
    assert_eq!(eval(&t, &[arg(1)], &[], &[&x1, &x2]), None);
}

/// test/integration/ext/loopvectorization/test_template_expression.jl | "Test feature constraints in
/// TemplateExpression"
#[test]
fn subexpressions_only_read_their_own_arguments() {
    let t = template("f(x1, x2) + g(x3)", &["f", "g"], &["x1", "x2", "x3"], &[]);
    let x: [&[f64]; 3] = [&[1.0], &[2.0], &[3.0]];
    assert_close(&eval(&t, &[arg(1) + arg(2), arg(1)], &[], &x).unwrap(), &[6.0]);
    assert_eq!(eval(&t, &[arg(1) + arg(3), arg(1)], &[], &x), None);
    assert_eq!(eval(&t, &[arg(1) + arg(2), arg(2)], &[], &x), None);
}

/// test/integration/ext/loopvectorization/test_template_expression.jl | "Test invalid structure"
#[test]
fn inconsistent_or_missing_calls_fail_at_construction() {
    let msg = error("f(x1) + f(x1, x2)", &["f"], None, &["x1", "x2"], &[]);
    assert!(msg.contains("`f`"), "{msg}");
    let msg = error("f(x1)", &["f", "g"], None, &["x1", "x2"], &[]);
    assert!(
        msg.contains("failed to infer the number of features used by `g`"),
        "{msg}"
    );
}

/// test/integration/ext/loopvectorization/test_template_expression.jl | "Test argument-less template structure"
#[test]
fn argument_less_subexpressions() {
    let t = template("f() * x1", &["f"], &["x1", "x2"], &[]);
    assert_eq!(t.num_features(), [0]);
    assert_close(&eval(&t, &[constant(3.0)], &[], &[&[1.0], &[2.0]]).unwrap(), &[3.0]);

    let t = template("f() + g(x1, x2)", &["f", "g"], &["x1", "x2"], &[]);
    assert_eq!(t.num_features(), [0, 2]);
    let trees = [constant(3.0), arg(1) + arg(2)];
    assert_close(&eval(&t, &trees, &[], &[&[1.0], &[2.0]]).unwrap(), &[6.0]);
}

/// test/integration/ext/loopvectorization/test_template_expression.jl | "Test TemplateExpression with differential
/// operator"
#[test]
fn derivative_operator() {
    let t = template("f(x1) + D(g, 1)(x2, x3)", &["f", "g"], &["x1", "x2", "x3"], &[]);
    let trees = [arg(1), cos(arg(1) - arg(2)) + lit(2.5) * arg(1)];
    let x: [&[f64]; 3] = [&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]];
    let want: Vec<f64> = (0..2).map(|i| x[0][i] - (x[1][i] - x[2][i]).sin() + 2.5).collect();
    assert_close(&eval(&t, &trees, &[], &x).unwrap(), &want);
}

/// test/integration/ext/loopvectorization/test_template_expression.jl | "Test literal_pow with ValidVector"
#[test]
fn literal_powers() {
    let x: [&[f64]; 1] = [&[2.0, 3.0, 4.0]];
    let power = |src| template(src, &["f"], &["x"], &[]);
    let with_f = |src: &str| TemplateStructure::new(src, &["f"], Some(&[1]), &["x"], &[] as &[(&str, usize)]).unwrap();
    assert_eq!(power("f(x)^2").num_features(), [1]);
    assert_close(&eval(&with_f("x^2"), &[arg(1)], &[], &x).unwrap(), &[4.0, 9.0, 16.0]);
    assert_close(&eval(&with_f("x^3"), &[arg(1)], &[], &x).unwrap(), &[8.0, 27.0, 64.0]);
    assert_eq!(eval(&with_f("x^2"), &[arg(1)], &[], &[&[2.0, f64::NAN, 4.0]]), None);
}

/// test/integration/ext/loopvectorization/test_template_expression.jl | "Test nan behavior with argument-less
/// expressions"
#[test]
fn argument_less_nan_is_invalid() {
    let c = || constant(3.0);
    let zero_over_zero = (c() / c() - 1.0) / (c() / c() - 1.0);
    let t = template("f() * x1", &["f"], &["x1", "x2"], &[]);
    assert_eq!(eval(&t, &[zero_over_zero], &[], &[&[1.0], &[2.0]]), None);
}

/// test/integration/ext/loopvectorization/test_template_expression.jl | "Test higher-order derivatives of safe_log with
/// DynamicDiff"
#[test]
fn higher_order_derivatives_of_log() {
    let at = |src, x: f64| eval(&template(src, &["f"], &["x"], &[]), &[log(arg(1))], &[], &[&[x]]);
    assert_eq!(at("f(x)", 2.0), Some(vec![2.0_f64.ln()]));
    assert_eq!(at("D(f, 1)(x)", 2.0), Some(vec![0.5]));
    assert_eq!(at("D(D(f, 1), 1)(x)", 2.0), Some(vec![-0.25]));

    assert_eq!(at("f(x)", -1.0), None);
    assert_eq!(at("D(f, 1)(x)", -1.0), Some(vec![0.0]));
    assert_eq!(at("D(D(f, 1), 1)(x)", -1.0), Some(vec![0.0]));
}

/// test/integration/ext/loopvectorization/test_template_expression.jl | "Test Float32/Float64 type conversion in
/// TemplateExpression"
#[test]
fn float32_data_with_float64_literals() {
    let t = template("0.5 * f(x1, x2)", &["f"], &["x1", "x2"], &[]);
    let tree = arg32(1) + arg32(2);
    let x = features::<f32>(&[&[1.0, 2.0], &[3.0, 4.0]]);
    let (out, ok): (Vec<f32>, bool) = t.eval(&[tree], &[], &BuiltinOpsF32::new(), x.view());
    assert!(ok);
    assert_eq!(out, [2.0_f32, 3.0]);
}

/// test/integration/ext/mlj/templates/test_parametric_template_expressions.jl | "creation & parameter counts"
#[test]
fn creation_and_parameter_counts() {
    let t = template("f(x) + p[1] + p[2]", &["f"], &["x"], &[("p", 2)]);
    assert_eq!(t.num_features(), [1]);
    assert_eq!(t.parameters(), [("p".to_string(), 2)]);
    assert_eq!(t.n_parameters(), 2);
    assert_eq!(t.expressions(), ["f"]);
}

/// test/integration/ext/mlj/templates/test_parametric_template_expressions.jl | "error conditions"
#[test]
fn wrong_parameter_counts_are_rejected() {
    let with_p = template("f(x) + p[1] + p[2]", &["f"], &["x"], &[("p", 2)]);
    let without = template("f(x)", &["f"], &["x"], &[]);
    for (t, params, msg) in [
        (&with_p, vec![1.0], "expected 2 parameters, got 1"),
        (&without, vec![1.0], "expected 0 parameters, got 1"),
    ] {
        let err = catch_unwind(AssertUnwindSafe(|| eval(t, &[arg(1)], &params, &[&[1.0]]))).unwrap_err();
        let got = err
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| err.downcast_ref::<&str>().copied())
            .unwrap();
        assert!(got.contains(msg), "{got}");
    }
}

/// test/integration/ext/mlj/templates/test_parametric_template_expressions.jl | "basic evaluation"
#[test]
fn basic_parameter_evaluation() {
    let t = template("f(x) + p[1] + p[2]", &["f"], &["x"], &[("p", 2)]);
    assert_close(&eval(&t, &[arg(1)], &[1.0, 2.0], &[&[5.0, 6.0]]).unwrap(), &[8.0, 9.0]);
}

/// test/integration/ext/mlj/templates/test_parametric_template_expressions.jl | "indexed evaluation"
#[test]
fn indexed_parameter_evaluation() {
    let t = template(
        "f(x) * p[i] - g(y) * p[i + 3]",
        &["f", "g"],
        &["x", "y", "i"],
        &[("p", 6)],
    );
    let trees = [cos(arg(1) * 3.1 - 0.5), exp(lit(1.0) - arg(1) * arg(1))];
    let mut rng = fastrand::Rng::with_seed(0);
    let x: Vec<f64> = (0..10).map(|_| rng.f64()).collect();
    let y: Vec<f64> = (0..10).map(|_| rng.f64()).collect();
    let i: Vec<f64> = (0..10).map(|_| rng.usize(1..=3) as f64).collect();
    let p: Vec<f64> = (0..6).map(|_| rng.f64()).collect();
    let want: Vec<f64> = (0..10)
        .map(|j| {
            let k = i[j] as usize;
            (x[j] * 3.1 - 0.5).cos() * p[k - 1] - (1.0 - y[j] * y[j]).exp() * p[k + 2]
        })
        .collect();
    assert_close(&eval(&t, &trees, &p, &[&x, &y, &i]).unwrap(), &want);
}

/// test/integration/ext/mlj/templates/test_parametric_template_expressions.jl | "Zero-parameter edge case"
#[test]
fn zero_parameter_template() {
    let t = template("f(x)", &["f"], &["x"], &[]);
    assert!(t.parameters().is_empty());
    assert_eq!(eval(&t, &[arg(1)], &[], &[&[10.0]]), Some(vec![10.0]));
}

/// test/integration/ext/mlj/templates/test_parametric_template_expressions.jl | "Non-Float64 parameter types"
#[test]
fn float32_parameters() {
    let t = template("f(x) + p[1] + p[2]", &["f"], &["x"], &[("p", 2)]);
    let tree = arg32(1) * 2.0;
    let x = features::<f32>(&[&[2.0]]);
    let (out, ok): (Vec<f32>, bool) = t.eval(&[tree], &[10.0, 20.0], &BuiltinOpsF32::new(), x.view());
    assert!(ok);
    assert_eq!(out, [34.0_f32]);
}

/// test/integration/ext/mlj/templates/test_parametric_template_expressions.jl | "multi-parameter expressions"
#[test]
fn multi_parameter_evaluation() {
    let t = template(
        "p1[1] * f(x)^2 + p1[2] * f(x) + p1[3] + p2[1] * g(y)^2 + p2[2] * g(y)",
        &["f", "g"],
        &["x", "y"],
        &[("p1", 3), ("p2", 2)],
    );
    assert_eq!(t.num_features(), [1, 1]);
    assert_eq!(t.parameters(), [("p1".to_string(), 3), ("p2".to_string(), 2)]);
    let params = [1.0, 2.0, 3.0, -1.0, -2.0];
    let out = eval(&t, &[arg(1), arg(1) + 0.0], &params, &[&[2.0, 3.0], &[4.0, 5.0]]).unwrap();
    assert_close(&out, &[-13.0, -17.0]);
}

/// test/integration/ext/mlj/templates/test_parametric_template_expressions.jl | "printing"
#[test]
fn printing_parameters() {
    let ops = BuiltinOpsF64::new();
    let t = template("f(x) + p[1] + p[2]", &["f"], &["x"], &[("p", 2)]);
    assert_eq!(
        t.string(&[arg(1)], &[1.0, 2.0], &ops, false, None),
        "f = #1; p = [1, 2]"
    );

    let t = template("f(x) + p1[1] + p2[1]", &["f"], &["x"], &[("p1", 10), ("p2", 1)]);
    let mut params = vec![1.0; 10];
    params.push(2.0);
    assert_eq!(
        t.string(&[arg(1)], &params, &ops, false, None),
        "f = #1; p1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]; p2 = [2]"
    );
}

/// test/integration/ext/mlj/templates/test_parametric_template_expressions.jl | "indexed multi-parameter expressions"
#[test]
fn indexed_multi_parameter_evaluation() {
    let t = template(
        "f(x, p1[i1], p2[i2]) + p3[1] * y",
        &["f"],
        &["x", "y", "i1", "i2"],
        &[("p1", 8), ("p2", 3), ("p3", 1)],
    );
    let tree = arg(1) * arg(2) + arg(3) * 1.5;
    let mut rng = fastrand::Rng::with_seed(0);
    let params: Vec<f64> = (0..12).map(|_| rng.f64()).collect();
    let (p1, p2, p3) = (&params[..8], &params[8..11], params[11]);
    let x: Vec<f64> = (0..32).map(|_| rng.f64()).collect();
    let y: Vec<f64> = (0..32).map(|_| rng.f64()).collect();
    let i1: Vec<f64> = (0..32).map(|_| rng.usize(1..=8) as f64).collect();
    let i2: Vec<f64> = (0..32).map(|_| rng.usize(1..=3) as f64).collect();
    let want: Vec<f64> = (0..32)
        .map(|j| x[j] * p1[i1[j] as usize - 1] + p2[i2[j] as usize - 1] * 1.5 + p3 * y[j])
        .collect();
    assert_close(&eval(&t, &[tree], &params, &[&x, &y, &i1, &i2]).unwrap(), &want);
}

/// test/unit/expressions/test_composable_expression.jl | "Test ComposableExpression"
#[test]
fn subexpression_reads_its_first_argument() {
    let t = template("f(x, y)", &["f"], &["x", "y"], &[]);
    let (x, y) = (random(1, 32), random(2, 32));
    assert_eq!(eval(&t, &[arg(1)], &[], &[&x, &y]), Some(x));
}

/// test/unit/expressions/test_composable_expression.jl | "Test interface for ComposableExpression"
#[test]
fn subexpressions_compose_with_themselves() {
    let t = template("f(f(x1, x2), f(x1, x2))", &["f"], &["x1", "x2"], &[]);
    let (x1, x2) = (random(1, 16), random(2, 16));
    let want: Vec<f64> = (0..16)
        .map(|i| {
            let f = x1[i] * x2[i].sin();
            f * f.sin()
        })
        .collect();
    assert_close(&eval(&t, &[arg(1) * sin(arg(2))], &[], &[&x1, &x2]).unwrap(), &want);
}

/// test/unit/expressions/test_composable_expression.jl | "Cover other operators"
#[test]
fn comparison_operators_print_as_infix() {
    let ops = BuiltinOpsF64::new();
    let pretty = StringTreeOptions {
        pretty: true,
        variable_names: Some(&["x1".to_string(), "x2".to_string()]),
        ..Default::default()
    };
    let call = |name: &str, rhs: Tree| {
        let op = ops.lookup_with_arity(name, 2).unwrap();
        dynamic_expressions::expression_algebra::__apply_postfix::<f64, BuiltinOpsF64, 3, 2>(op.id, [arg(1), rhs])
    };
    for (name, symbol) in [
        ("greater", ">"),
        ("greater_equal", ">="),
        ("less", "<"),
        ("less_equal", "<="),
    ] {
        assert_eq!(
            string_tree(&call(name, arg(2)), &ops, pretty.clone()),
            format!("x1 {symbol} x2")
        );
    }
    assert_eq!(string_tree(&call("greater", constant(1.0)), &ops, pretty), "x1 > 1");
}

/// test/unit/expressions/test_composable_expression.jl | "Test error handling"
#[test]
fn formula_operators_on_columns() {
    let x = [1.0, 2.0, 3.0];
    let y = [0.0, 2.0, 4.0];
    let formula = |src: &str, rows: &[&[f64]]| {
        let t = TemplateStructure::new(src, &["f"], Some(&[1]), &["x", "y"], &[] as &[(&str, usize)]).unwrap();
        eval(&t, &[arg(1)], &[], rows)
    };
    let at = |src| formula(src, &[&x, &y]).unwrap();
    assert_close(&at("x + 2.0"), &[3.0, 4.0, 5.0]);
    assert_close(&at("2.0 + x"), &[3.0, 4.0, 5.0]);
    assert_close(&at("x * 2.0"), &[2.0, 4.0, 6.0]);
    assert_close(&at("2.0 * x"), &[2.0, 4.0, 6.0]);
    assert_close(&at("x > y"), &[1.0, 0.0, 0.0]);
    assert_close(&at("x < y"), &[0.0, 0.0, 1.0]);
    assert_close(&at("x >= y"), &[1.0, 1.0, 0.0]);
    assert_close(&at("x <= y"), &[0.0, 1.0, 1.0]);
    assert_close(&at("x > 1.5"), &[0.0, 1.0, 1.0]);
    assert_close(&at("1.5 > x"), &[1.0, 0.0, 0.0]);
    assert_close(&at("max(x, y)"), &[1.0, 2.0, 4.0]);
    assert_close(&at("min(x, y)"), &[0.0, 2.0, 3.0]);
    assert_close(&at("max(x, 2.0)"), &[2.0, 2.0, 3.0]);
    assert_close(&at("min(x, 2.0)"), &[1.0, 2.0, 2.0]);
    assert_close(&at("sin(x)"), &x.map(f64::sin));
    assert_close(&at("cos(x)"), &x.map(f64::cos));
    assert_close(&at("atan(x)"), &x.map(f64::atan));
    assert_close(&at("abs(x)"), &[1.0, 2.0, 3.0]);
    assert_close(&at("-x"), &[-1.0, -2.0, -3.0]);
    assert_close(
        &at("atan(y, x)"),
        &[0.0_f64.atan2(1.0), 2.0_f64.atan2(2.0), 4.0_f64.atan2(3.0)],
    );

    let invalid = [1.0, f64::NAN, 3.0];
    assert_eq!(formula("x + 2.0", &[&invalid, &y]), None);
    assert_eq!(formula("2.0 + x", &[&invalid, &y]), None);
    assert_eq!(formula("sin(x)", &[&invalid, &y]), None);
}

/// test/unit/expressions/test_composable_expression.jl | "Test validity propagation with NaN"
#[test]
fn division_by_zero_invalidates_the_result() {
    let t = template("f(x1, x2)", &["f"], &["x1", "x2"], &[]);
    let tree = || lit(1.0) + arg(2) / arg(1);
    assert_close(&eval(&t, &[tree()], &[], &[&[1.0], &[2.0]]).unwrap(), &[3.0]);
    assert_close(
        &eval(&t, &[tree()], &[], &[&[1.0, 1.0], &[2.0, 2.0]]).unwrap(),
        &[3.0, 3.0],
    );
    assert_eq!(eval(&t, &[tree()], &[], &[&[1.0, 0.0], &[2.0, 2.0]]), None);
}

/// test/unit/expressions/test_composable_expression.jl | "Test Number inputs"
#[test]
fn subexpressions_accept_number_arguments() {
    let call = |src| TemplateStructure::new(src, &["f"], None, &["x1"], &[] as &[(&str, usize)]).unwrap();
    let tree = || arg(1) + arg(2);
    assert_close(
        &eval(&call("f(2.0, 3.0) * x1"), &[tree()], &[], &[&[1.0]]).unwrap(),
        &[5.0],
    );
    assert_eq!(eval(&call("f(NaN, 3.0) * x1"), &[tree()], &[], &[&[1.0]]), None);
    assert_close(&eval(&call("f(x1, 2.0)"), &[tree()], &[], &[&[1.0]]).unwrap(), &[3.0]);
    assert_close(
        &eval(&call("f(x1, 2.0)"), &[tree()], &[], &[&[1.0, 1.0]]).unwrap(),
        &[3.0, 3.0],
    );
    assert_eq!(eval(&call("f(x1, 2.0)"), &[tree()], &[], &[&[1.0, f64::NAN]]), None);
}

/// test/unit/templates/test_template_expression_buffer.jl | "TemplateExpression uses one call-time evaluation arena"
#[test]
fn one_workspace_serves_every_template() {
    let ops = BuiltinOpsF64::new();
    let trees = [arg(1) * arg(1) + 1.0, lit(2.0) * arg(1) - 0.5];
    let x = features(&[&[0.5, 1.0, 2.0, 3.0], &[-2.0, -1.0, 0.0, 4.0]]);
    let (x1, x2) = (x.row(0).to_vec(), x.row(1).to_vec());
    let f = |v: f64| v * v + 1.0;
    let g = |v: f64| 2.0 * v - 0.5;
    let cases: [(&str, &[&str], Vec<f64>); 4] = [
        (
            "f(x1) + D(f, 1)(x1)",
            &["f"],
            x1.iter().map(|&v| f(v) + 2.0 * v).collect(),
        ),
        ("f(x1) + g(x1)", &["f", "g"], x1.iter().map(|&v| f(v) + g(v)).collect()),
        ("f(x1) + f(x2)", &["f"], (0..4).map(|i| f(x1[i]) + f(x2[i])).collect()),
        ("f(g(x1))", &["f", "g"], x1.iter().map(|&v| f(g(v))).collect()),
    ];
    let mut ws = TemplateWorkspace::default();
    for _ in 0..2 {
        for (src, expressions, want) in &cases {
            let t = template(src, expressions, &["x1", "x2"], &[]);
            let trees = &trees[..expressions.len()];
            let plans = t.compile_plans(trees);
            let mut out = vec![0.0; 4];
            assert!(t.eval_into(&mut out, trees, &plans, &[], &ops, x.view(), &mut ws));
            let (fresh, ok) = t.eval(trees, &[], &ops, x.view());
            assert!(ok);
            assert_eq!(out, fresh, "{src}");
            assert_close(&out, want);
        }
    }
}

/// test/unit/templates/test_template_expression_string.jl | "template expression string representation"
#[test]
fn compact_and_pretty_strings() {
    let ops = BuiltinOpsF64::new();
    let t = template("f(x) + p[1] + p[2]", &["f"], &["x"], &[("p", 2)]);
    let compact = t.string(&[arg(1)], &[1.0, 2.0], &ops, false, None);
    assert!(
        compact.contains("f = ") && compact.contains("; p = ") && compact.contains("[1, 2]"),
        "{compact}"
    );
    let pretty = t.string(&[arg(1)], &[1.0, 2.0], &ops, true, None);
    assert!(
        pretty.contains("╭ f = ") && pretty.contains("\n╰ p = ") && pretty.contains("[1, 2]"),
        "{pretty}"
    );

    let t = template("f(x) + p[1]", &["f"], &["x"], &[("p", 6)]);
    let long = t.string(&[arg(1)], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &ops, true, None);
    assert!(long.contains("[1, 2, 3, ..., 6]"), "{long}");
}

/// test/unit/templates/test_template_macro.jl | "Basic @template_spec macro functionality"
#[test]
fn template_with_parameters_and_category_indices() {
    let t = template(
        "p1[class] * x1^2 + f(x1, x2, p2[class]) - g(p3[1] * x1)",
        &["f", "g"],
        &["x1", "x2", "class"],
        &[("p1", 10), ("p2", 10), ("p3", 1)],
    );
    assert_eq!(t.expressions(), ["f", "g"]);
    assert_eq!(t.num_features(), [3, 1]);
    assert_eq!(t.n_parameters(), 21);
    let out = eval(&t, &[arg(1), arg(1)], &[0.0; 21], &[&[1.0], &[2.0], &[1.0]]).unwrap();
    assert_close(&out, &[1.0]);
}

/// test/unit/templates/test_template_macro.jl | "Template macro error handling"
#[test]
fn template_needs_a_subexpression() {
    let msg = error("x", &[], None, &["x"], &[("p1", 1)]);
    assert!(msg.contains("at least one subexpression"), "{msg}");
}

/// test/unit/templates/test_template_macro.jl | "Template macro with complex structure"
#[test]
fn template_reusing_one_parameter_vector() {
    let t = template(
        "coeff[class] * x * base(x, coeff[class]) + modifier(y, coeff[class])",
        &["base", "modifier"],
        &["x", "y", "class"],
        &[("coeff", 5)],
    );
    assert_eq!(t.parameters(), [("coeff".to_string(), 5)]);
    let out = eval(&t, &[arg(1), arg(1)], &[1.0; 5], &[&[2.0], &[3.0], &[4.0]]).unwrap();
    assert_close(&out, &[7.0]);
}

/// test/unit/templates/test_template_macro.jl | "Template macro without parameters"
#[test]
fn template_without_parameters() {
    let t = template("x1^2 + f(x1, x2) - g(x1)", &["f", "g"], &["x1", "x2"], &[]);
    assert_eq!(t.expressions(), ["f", "g"]);
    assert_eq!(t.n_parameters(), 0);
    assert_close(&eval(&t, &[arg(1), arg(1)], &[], &[&[1.0], &[2.0]]).unwrap(), &[1.0]);
}

/// test/unit/templates/test_template_macro.jl | "Template macro with num_features parameter"
#[test]
fn declared_num_features_must_match_the_calls() {
    let t = TemplateStructure::new(
        "x1^2 + f(x1, x2)",
        &["f"],
        Some(&[2]),
        &["x1", "x2"],
        &[] as &[(&str, usize)],
    )
    .unwrap();
    assert_eq!(t.num_features(), [2]);
    let msg = error("x1^2 + f(x1, x2)", &["f"], Some(&[5]), &["x1", "x2"], &[]);
    assert!(
        msg.contains("declared with 5 features but called with 2 arguments"),
        "{msg}"
    );
}

/// test/unit/templates/test_template_prototype.jl | "Default Float64 inference is unchanged by prototype option"
#[test]
fn num_features_inference() {
    assert_eq!(
        template("f(x1, x2) + g(x3)", &["f", "g"], &["x1", "x2", "x3"], &[]).num_features(),
        [2, 1]
    );
    assert_eq!(
        template("f(x1) + g()", &["f", "g"], &["x1"], &[]).num_features(),
        [1, 0]
    );
    let declared = TemplateStructure::new("x1", &["f"], Some(&[1]), &["x1"], &[] as &[(&str, usize)]).unwrap();
    assert_eq!(declared.num_features(), [1]);
}

/// test/unit/templates/test_template_prototype.jl | "TemplateStructure requires disjoint expression and parameter
/// names"
#[test]
fn expression_and_parameter_names_are_disjoint() {
    let msg = error("f(x)", &["f"], Some(&[1]), &["x"], &[("f", 1)]);
    assert!(msg.contains("`f` names more than one"), "{msg}");
}
