//! Template evaluation beyond SymbolicRegression.jl's testitems: `D` of any order, DSL-defined
//! operators in subexpressions and formulas, the derivative cache, and templates that cannot be
//! evaluated at all.

use dynamic_expressions::dsl::{
    CustomOps, TemplateEvalError, TemplateStructure, TemplateWorkspace, parse_operator_tokens,
};
use dynamic_expressions::operator_enum::builtin::{Add, Mul};
use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
use dynamic_expressions::operators::{pow, sin};
use dynamic_expressions::{Metadata, OperatorSet, PNode, PostfixExpr, lit};
use ndarray::Array2;

dynamic_expressions::op!(Twice for f64 {
    eval: |[x]| { 2.0 * x },
    partial: |[_x], _idx| { 2.0 },
});

dynamic_expressions::opset! {
    RawOps for f64 { Add, Mul, Twice }
}

type Tree = PostfixExpr<f64, BuiltinOpsF64, 3>;
type CustomTree = PostfixExpr<f64, BuiltinOpsF64<CustomOps<f64>>, 3>;

/// The subexpression argument `#i` (1-based).
fn arg(i: u16) -> Tree {
    PostfixExpr::new(vec![PNode::Var { feature: i - 1 }], vec![], Metadata::default())
}

fn constant(v: f64) -> Tree {
    PostfixExpr::new(vec![PNode::Const { idx: 0 }], vec![v], Metadata::default())
}

fn template<Ops: OperatorSet>(
    src: &str,
    expressions: &[&str],
    num_features: Option<&[usize]>,
    variables: &[&str],
    parameters: &[(&str, usize)],
    ops: &Ops,
) -> TemplateStructure {
    TemplateStructure::new(src, expressions, num_features, variables, parameters, ops)
        .unwrap_or_else(|e| panic!("`{src}`: {e}"))
}

/// Features by rows, one row per slice.
fn features(rows: &[&[f64]]) -> Array2<f64> {
    Array2::from_shape_fn((rows.len(), rows[0].len()), |(f, i)| rows[f][i])
}

/// Evaluates the one-subexpression template `t` on `rows`; `Ok(None)` when the result is invalid.
fn eval<Ops: OperatorSet<T = f64>>(
    t: &TemplateStructure,
    tree: &PostfixExpr<f64, Ops, 3>,
    params: &[f64],
    ops: &Ops,
    rows: &[&[f64]],
) -> Result<Option<Vec<f64>>, TemplateEvalError> {
    let (out, ok) = t.eval(core::slice::from_ref(tree), params, ops, features(rows).view())?;
    Ok(ok.then_some(out))
}

fn assert_close(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() <= 1e-12 * w.abs().max(1.0), "{got:?} vs {want:?}");
    }
}

/// `D(...D(f, i)..., i)(args)`, nested `order` deep.
fn nested_d(order: usize, i: usize, args: &str) -> String {
    let d = (0..order).fold("f".to_string(), |f, _| format!("D({f}, {i})"));
    format!("{d}({args})")
}

/// `f` with a unary custom operator `name` applied to `#1`.
fn custom_call(ops: &BuiltinOpsF64<CustomOps<f64>>, name: &str) -> CustomTree {
    let op = ops.lookup_with_arity(name, 1).unwrap();
    PostfixExpr::new(
        vec![PNode::Var { feature: 0 }, PNode::Op { arity: 1, op: op.id }],
        vec![],
        Metadata::default(),
    )
}

fn custom_ops(definitions: &[&str]) -> BuiltinOpsF64<CustomOps<f64>> {
    let (custom, _) = parse_operator_tokens::<f64, _>(definitions).unwrap();
    BuiltinOpsF64::new().with_custom(custom)
}

#[test]
fn derivatives_of_any_order() {
    let ops = BuiltinOpsF64::new();
    let x = || arg(1);
    let power = |n: usize| (1..n).fold(x(), |acc, _| acc * x());
    let tree = lit(3.0) * power(7) - lit(2.0) * power(4) + x();
    let xs = [-1.5, -0.5, 0.75, 2.0];
    let at = |order: usize| {
        let t = template(&nested_d(order, 1, "x"), &["f"], None, &["x"], &[], &ops);
        eval(&t, &tree, &[], &ops, &[&xs]).unwrap().unwrap()
    };
    assert_close(&at(4), &xs.map(|v| 2520.0 * v * v * v - 48.0));
    assert_close(&at(6), &xs.map(|v| 15120.0 * v));
    assert_close(&at(7), &[15120.0; 4]);
    assert_close(&at(8), &[0.0; 4]);
}

#[test]
fn mixed_partial_derivatives() {
    let ops = BuiltinOpsF64::new();
    let tree = sin(arg(1) * arg(2)) + arg(1) * arg(1) * arg(1) * arg(2) * arg(2);
    let (xs, ys) = ([0.3, -1.2, 2.0], [0.7, 0.4, -1.1]);
    let at = |src: &str| {
        let t = template(src, &["f"], None, &["x", "y"], &[], &ops);
        eval(&t, &tree, &[], &ops, &[&xs, &ys]).unwrap().unwrap()
    };
    let d12: Vec<f64> = (0..3)
        .map(|i| {
            let (x, y) = (xs[i], ys[i]);
            (x * y).cos() - x * y * (x * y).sin() + 6.0 * x * x * y
        })
        .collect();
    let d122: Vec<f64> = (0..3)
        .map(|i| {
            let (x, y) = (xs[i], ys[i]);
            -2.0 * x * (x * y).sin() - x * x * y * (x * y).cos() + 6.0 * x * x
        })
        .collect();
    assert_close(&at("D(D(f, 1), 2)(x, y)"), &d12);
    assert_close(&at("D(D(f, 2), 1)(x, y)"), &d12);
    assert_close(&at("D(D(D(f, 1), 2), 2)(x, y)"), &d122);
}

#[test]
fn derivative_with_respect_to_an_absent_argument_is_zero() {
    let ops = BuiltinOpsF64::new();
    let tree = arg(1) * 2.5 + 1.0;
    let xs = [1.0, -2.0];
    for src in ["x + D(f, 3)(x)", "x + D(f, 65535)(x)", "x + D(D(f, 1), 2)(x)"] {
        let t = template(src, &["f"], None, &["x"], &[], &ops);
        assert_eq!(t.num_features(), [1]);
        assert_close(&eval(&t, &tree, &[], &ops, &[&xs]).unwrap().unwrap(), &xs);
    }
}

#[test]
fn custom_operators_are_differentiated_and_callable_from_formulas() {
    let ops = custom_ops(&["cube3(x) = x * x * x", "softplus(x) = log(1 + exp(x))"]);
    let xs: [f64; 3] = [-1.5, 0.5, 2.0];
    let t = template("D(D(f, 1), 1)(x) + softplus(x)", &["f"], None, &["x"], &[], &ops);
    let want = xs.map(|v| 6.0 * v + (1.0 + v.exp()).ln());
    assert_close(
        &eval(&t, &custom_call(&ops, "cube3"), &[], &ops, &[&xs])
            .unwrap()
            .unwrap(),
        &want,
    );

    let t = template("D(f, 1)(x)", &["f"], None, &["x"], &[], &ops);
    let want = xs.map(|v| v.exp() / (1.0 + v.exp()));
    assert_close(
        &eval(&t, &custom_call(&ops, "softplus"), &[], &ops, &[&xs])
            .unwrap()
            .unwrap(),
        &want,
    );

    let err = TemplateStructure::new("f(x) + softplus(x)", &["f"], None, &["x"], &[], &BuiltinOpsF64::new())
        .map(|_| ())
        .expect_err("`softplus` is not defined without the custom operators");
    assert!(err.message.contains("unknown function `softplus`"), "{}", err.message);
}

#[test]
fn integer_powers_of_negative_bases_have_finite_derivatives() {
    let ops = BuiltinOpsF64::new();
    let tree = pow(arg(1), constant(3.0));
    let xs = [-2.0, -0.5, 1.5];
    let t = template("D(f, 1)(x) + D(D(f, 1), 1)(x)", &["f"], None, &["x"], &[], &ops);
    let want = xs.map(|v| 3.0 * v * v + 6.0 * v);
    assert_close(&eval(&t, &tree, &[], &ops, &[&xs]).unwrap().unwrap(), &want);
}

#[test]
fn cached_derivatives_follow_constants_and_operator_definitions() {
    let x = features(&[&[1.0, 2.0, 3.0]]);
    let mut out = [0.0; 3];
    let mut ws = TemplateWorkspace::default();

    let ops = BuiltinOpsF64::new();
    let t = template("D(f, 1)(x)", &["f"], None, &["x"], &[], &ops);
    for c in [2.0, 5.0] {
        let trees = [arg(1) * c];
        let plans = t.compile_plans(&trees);
        assert!(
            t.eval_into(&mut out, &trees, &plans, &[], &ops, x.view(), &mut ws)
                .unwrap()
        );
        assert_eq!(out, [c; 3]);
    }

    let mut ws = TemplateWorkspace::default();
    for (definition, derivative) in [
        ("g(x) = x * x", [2.0, 4.0, 6.0]),
        ("g(x) = x * x * x", [3.0, 12.0, 27.0]),
    ] {
        let ops = custom_ops(&[definition]);
        let t = template("D(f, 1)(x)", &["f"], None, &["x"], &[], &ops);
        let trees = [custom_call(&ops, "g")];
        let plans = t.compile_plans(&trees);
        assert!(
            t.eval_into(&mut out, &trees, &plans, &[], &ops, x.view(), &mut ws)
                .unwrap()
        );
        assert_eq!(out, derivative, "{definition}");
    }
}

#[test]
fn scalar_formulas_fail_at_evaluation() {
    let ops = BuiltinOpsF64::new();
    let rows: [&[f64]; 1] = [&[1.0, 2.0]];
    let cases = [
        (template("f()", &["f"], None, &["x"], &[], &ops), constant(3.0)),
        (
            template("f(2.0, 3.0)", &["f"], None, &["x"], &[], &ops),
            arg(1) + arg(2),
        ),
        (
            template("p[1] + 2", &["f"], Some(&[1]), &["x"], &[("p", 1)], &ops),
            arg(1),
        ),
        (template("f()", &["f"], None, &["x"], &[], &ops), arg(1)),
    ];
    for (t, tree) in cases {
        let params = vec![1.0; t.n_parameters()];
        assert_eq!(
            eval(&t, &tree, &params, &ops, &rows),
            Err(TemplateEvalError::ScalarReturn),
            "{}",
            t.combine()
        );
        let mut out = [0.0; 2];
        let trees = [tree];
        let x = features(&rows);
        let result = t.eval_into(
            &mut out,
            &trees,
            &[],
            &params,
            &ops,
            x.view(),
            &mut TemplateWorkspace::default(),
        );
        assert_eq!(result, Err(TemplateEvalError::ScalarReturn), "{}", t.combine());
    }

    let t = template("f(x)", &["f"], None, &["x"], &[], &ops);
    assert_eq!(eval(&t, &constant(3.0), &[], &ops, &rows), Ok(Some(vec![3.0, 3.0])));
}

#[test]
fn rust_operators_cannot_be_differentiated() {
    let ops = RawOps::new();
    let twice = ops.lookup_with_arity("twice", 1).unwrap();
    let tree = PostfixExpr::<f64, RawOps, 3>::new(
        vec![PNode::Var { feature: 0 }, PNode::Op { arity: 1, op: twice.id }],
        vec![],
        Metadata::default(),
    );
    let rows: [&[f64]; 1] = [&[1.0, 2.0]];

    let plain = template("f(x)", &["f"], None, &["x"], &[], &ops);
    assert_eq!(eval(&plain, &tree, &[], &ops, &rows), Ok(Some(vec![2.0, 4.0])));
    assert_eq!(plain.check_operators(&ops, &[twice]), Ok(()));

    let derivative = template("D(f, 1)(x)", &["f"], None, &["x"], &[], &ops);
    let err = TemplateEvalError::UnsupportedOperator(ops.name(twice).to_string());
    assert_eq!(eval(&derivative, &tree, &[], &ops, &rows), Err(err.clone()));
    assert_eq!(derivative.check_operators(&ops, &[twice]), Err(err.clone()));
    assert!(err.to_string().contains("`D` cannot differentiate operator"), "{err}");
}
