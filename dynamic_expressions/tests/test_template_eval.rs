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

/// `D` applied to `tree` per `src` at one point; `None` when the result is invalid.
fn d_at(src: &str, variables: &[&str], tree: &Tree, point: &[f64]) -> Option<f64> {
    let ops = BuiltinOpsF64::new();
    let t = template(src, &["f"], None, variables, &[], &ops);
    let rows: Vec<&[f64]> = point.iter().map(core::slice::from_ref).collect();
    eval(&t, tree, &[], &ops, &rows).unwrap().map(|v| v[0])
}

fn assert_points(cases: &[(&str, f64, [Option<f64>; 3])], tree: impl Fn(&str) -> Tree) {
    for (name, x, want) in cases {
        for (order, want) in (1..=3).zip(want) {
            let got = d_at(&nested_d(order, 1, "x"), &["x"], &tree(name), &[*x]);
            match (got, want) {
                (Some(g), Some(w)) => assert_close(&[g], &[*w]),
                _ => assert_eq!(got, *want, "D^{order} {name} at {x}"),
            }
        }
    }
}

/// Expected values are SymbolicRegression.jl f97764fb's `D` (DynamicDiff 0.3.2, ForwardDiff
/// 1.4.6) on `safe_pow(x, c)`.
#[test]
fn constant_powers_follow_forward_diff() {
    let cases = [
        ("2", 0.0, [Some(0.0), Some(2.0), Some(0.0)]),
        ("2", -2.0, [Some(-4.0), Some(2.0), Some(0.0)]),
        ("2", 1.5, [Some(3.0), Some(2.0), Some(0.0)]),
        ("0", 0.0, [Some(0.0), Some(0.0), Some(0.0)]),
        ("0", -2.0, [Some(0.0), Some(0.0), Some(0.0)]),
        ("1", 0.0, [Some(1.0), Some(0.0), Some(0.0)]),
        ("1", -2.0, [Some(1.0), Some(0.0), Some(0.0)]),
        ("3", 0.0, [Some(0.0), Some(0.0), Some(6.0)]),
        ("3", -2.0, [Some(12.0), Some(-12.0), Some(6.0)]),
        ("-1", 0.0, [None, None, None]),
        ("-1", 2.0, [Some(-0.25), Some(0.25), Some(-0.375)]),
        ("0.5", -2.0, [Some(0.0), Some(0.0), Some(0.0)]),
        ("0.5", 0.0, [None, None, None]),
        ("0.5", 4.0, [Some(0.25), Some(-0.03125), Some(0.01171875)]),
        ("2.5", 0.0, [Some(0.0), Some(0.0), None]),
    ];
    assert_points(&cases, |c| pow(arg(1), constant(c.parse().unwrap())));
}

/// SymbolicRegression.jl's `D` on `safe_pow(x, p)`: the partial with respect to `x` is 0 at
/// `p = 0` only while `p` stays fixed, so `D(D(f, 1), 2)` keeps the mixed derivative `1/x`.
#[test]
fn mixed_power_derivatives_follow_forward_diff() {
    let tree = pow(arg(1), arg(2));
    let ln2 = 2.0_f64.ln();
    // (x, p, [d1, d2, d1d2, d2d1, d2d2, d1d1])
    let cases: [(f64, f64, [Option<f64>; 6]); 5] = [
        (
            2.0,
            0.0,
            [Some(0.0), Some(ln2), Some(0.5), Some(0.5), Some(ln2 * ln2), Some(0.0)],
        ),
        (
            2.0,
            1.0,
            [
                Some(1.0),
                Some(2.0 * ln2),
                Some(1.0 + ln2),
                Some(1.0 + ln2),
                Some(2.0 * ln2 * ln2),
                Some(0.0),
            ],
        ),
        (
            2.0,
            2.5,
            [
                Some(7.0710678118654755),
                Some(3.921032573874189),
                Some(7.729717842088926),
                Some(7.729717842088926),
                Some(2.7178526734645994),
                Some(5.303300858899107),
            ],
        ),
        (0.0, 0.0, [Some(0.0), None, None, None, Some(0.0), Some(0.0)]),
        (0.0, 1.0, [Some(1.0), Some(0.0), None, None, Some(0.0), Some(0.0)]),
    ];
    let srcs = [
        "D(f, 1)(x, p)",
        "D(f, 2)(x, p)",
        "D(D(f, 1), 2)(x, p)",
        "D(D(f, 2), 1)(x, p)",
        "D(D(f, 2), 2)(x, p)",
        "D(D(f, 1), 1)(x, p)",
    ];
    for (x, p, want) in cases {
        for (src, want) in srcs.iter().zip(want) {
            let got = d_at(src, &["x", "p"], &tree, &[x, p]);
            match (got, want) {
                (Some(g), Some(w)) => assert_close(&[g], &[w]),
                _ => assert_eq!(got, want, "{src} at x={x}, p={p}"),
            }
        }
    }
}

/// SymbolicRegression.jl's safe operators return a NaN constant outside their domains, so `D`
/// of every order is 0 there; ForwardDiff's comparisons put a boundary inside the domain when the
/// dual number moves into it.
#[test]
fn safe_operators_have_zero_derivatives_outside_their_domains() {
    use dynamic_expressions::operators::{acos, acosh, asin, atanh, log, log1p, log2, log10, sqrt};
    let zeros = [Some(0.0); 3];
    let invalid = [None; 3];
    let cases = [
        ("sqrt", -1.0, zeros),
        ("sqrt", 0.0, invalid),
        ("sqrt", 4.0, [Some(0.25), Some(-0.03125), Some(0.01171875)]),
        ("log", -1.0, zeros),
        ("log", 0.0, invalid),
        ("log2", -1.0, zeros),
        ("log10", -1.0, zeros),
        ("log1p", -2.0, zeros),
        ("log1p", -1.0, invalid),
        ("asin", 2.0, zeros),
        ("asin", 1.0, zeros),
        ("asin", -1.0, invalid),
        ("acos", 2.0, zeros),
        ("acos", 1.0, zeros),
        ("acosh", 0.5, zeros),
        ("acosh", 1.0, invalid),
        ("atanh", 2.0, zeros),
        ("atanh", 1.0, zeros),
    ];
    assert_points(&cases, |name| {
        let x = arg(1);
        match name {
            "sqrt" => sqrt(x),
            "log" => log(x),
            "log2" => log2(x),
            "log10" => log10(x),
            "log1p" => log1p(x),
            "asin" => asin(x),
            "acos" => acos(x),
            "acosh" => acosh(x),
            _ => atanh(x),
        }
    });

    let at_minus_one = |src: &str, tree: Tree| d_at(src, &["x"], &tree, &[-1.0]);
    assert_eq!(at_minus_one("D(f, 1)(x)", sin(sqrt(arg(1)))), None);
    assert_eq!(at_minus_one("D(D(f, 1), 1)(x)", sqrt(arg(1)) * arg(1)), Some(0.0));
    assert_eq!(at_minus_one("D(f, 1)(x)", log(arg(1)) * arg(1)), None);
    assert_eq!(at_minus_one("D(D(f, 1), 1)(x)", log(arg(1)) * arg(1)), Some(0.0));
}

/// DiffRules gives `min` ties to its first argument and `max` ties to its second. DynamicDiff
/// differentiates `abs` to `sign`, which is 0 at 0.
#[test]
fn min_max_and_abs_partials_follow_diffrules() {
    use dynamic_expressions::operators::{abs, max, min};
    let points = [[1.0, 1.0], [1.0, 2.0], [2.0, 1.0]];
    let check = |tree: Tree, src: &str, want: [f64; 3]| {
        for (point, want) in points.iter().zip(want) {
            assert_eq!(d_at(src, &["x", "y"], &tree, point), Some(want), "{src} at {point:?}");
        }
    };
    check(min(arg(1), arg(2)), "D(f, 1)(x, y)", [1.0, 1.0, 0.0]);
    check(min(arg(1), arg(2)), "D(f, 2)(x, y)", [0.0, 0.0, 1.0]);
    check(max(arg(1), arg(2)), "D(f, 1)(x, y)", [0.0, 0.0, 1.0]);
    check(max(arg(1), arg(2)), "D(f, 2)(x, y)", [1.0, 1.0, 0.0]);
    check(min(arg(1) * arg(1), arg(2)), "D(D(f, 1), 1)(x, y)", [2.0, 2.0, 0.0]);
    check(max(arg(1) * arg(1), arg(2)), "D(D(f, 1), 1)(x, y)", [0.0, 0.0, 2.0]);
    assert_eq!(
        d_at("D(f, 1)(x)", &["x"], &min(arg(1), constant(1.0)), &[1.0]),
        Some(1.0)
    );
    assert_eq!(
        d_at("D(f, 1)(x)", &["x"], &max(constant(1.0), arg(1)), &[1.0]),
        Some(1.0)
    );
    assert_eq!(d_at("D(f, 1)(x)", &["x"], &abs(arg(1)), &[0.0]), Some(0.0));
}

/// `tree` with the unary operator `name` applied to it.
fn apply<Ops: OperatorSet>(ops: &Ops, name: &str, mut tree: PostfixExpr<f64, Ops, 3>) -> PostfixExpr<f64, Ops, 3> {
    let op = ops.lookup_with_arity(name, 1).unwrap();
    tree.nodes.push(PNode::Op { arity: 1, op: op.id });
    tree
}

fn assert_d(got: Option<f64>, want: Option<f64>, what: &str) {
    match (got, want) {
        (Some(g), Some(w)) => assert_close(&[g], &[w]),
        _ => assert_eq!(got, want, "{what}"),
    }
}

/// SymbolicRegression.jl's `D` differentiates each operator on its own, with its argument's
/// derivative seeded at +1, and multiplies by the argument's actual derivative. So a comparison
/// tie at an operator's boundary resolves by +1 even when the argument decreases. Expected values
/// are SymbolicRegression.jl f97764fb's `D` (DynamicDiff 0.3.2, ForwardDiff 1.4.6).
#[test]
fn operators_are_differentiated_at_their_own_boundaries() {
    let ops = custom_ops(&["s(u) = u < 0 ? NaN : sqrt(u)", "c(u) = ifelse(u > 0, u, 0)"]);
    let x = || CustomTree::new(vec![PNode::Var { feature: 0 }], vec![], Metadata::default());
    // (tree, x, [D, D^2])
    let cases = [
        (apply(&ops, "s", -x()), 0.0, [None, None]),
        (apply(&ops, "s", -x()), -1.0, [Some(-0.5), Some(-0.25)]),
        (apply(&ops, "s", -x()), -4.0, [Some(-0.25), Some(-0.03125)]),
        (apply(&ops, "s", -x()), 1.0, [Some(0.0), Some(0.0)]),
        (apply(&ops, "s", x()), 0.0, [None, None]),
        (apply(&ops, "c", -x()), 0.0, [Some(-1.0), Some(0.0)]),
        (apply(&ops, "c", -x()), -1.0, [Some(-1.0), Some(0.0)]),
        (apply(&ops, "c", -x()), 1.0, [Some(0.0), Some(0.0)]),
        (apply(&ops, "c", x()), 0.0, [Some(1.0), Some(0.0)]),
    ];
    for (tree, x0, want) in cases {
        for (order, want) in (1..=2).zip(want) {
            let t = template(&nested_d(order, 1, "x"), &["f"], None, &["x"], &[], &ops);
            let got = eval(&t, &tree, &[], &ops, &[&[x0]]).unwrap().map(|v| v[0]);
            assert_d(got, want, &format!("D^{order} of {:?} at {x0}", tree.nodes));
        }
    }

    use dynamic_expressions::operators::{asin, sqrt};
    let builtin = [
        (sqrt(-arg(1)), 0.0, [None, None]),
        (asin(-arg(1)), -1.0, [Some(0.0), Some(0.0)]),
        (asin(-arg(1)), 1.0, [None, None]),
    ];
    for (tree, x0, want) in builtin {
        for (order, want) in (1..=2).zip(want) {
            let got = d_at(&nested_d(order, 1, "x"), &["x"], &tree, &[x0]);
            assert_d(got, want, &format!("D^{order} of {:?} at {x0}", tree.nodes));
        }
    }

    let tree = pow(arg(1), -arg(2));
    let ln2 = 2.0_f64.ln();
    let srcs = [
        "D(f, 1)(x, p)",
        "D(f, 2)(x, p)",
        "D(D(f, 1), 2)(x, p)",
        "D(D(f, 2), 1)(x, p)",
        "D(D(f, 2), 2)(x, p)",
    ];
    let cases = [
        (0.0, [Some(0.0), None, None, None, Some(0.0)]),
        (2.0, [Some(0.0), Some(-ln2), Some(-0.5), Some(-0.5), Some(ln2 * ln2)]),
    ];
    for (x0, want) in cases {
        for (src, want) in srcs.iter().zip(want) {
            assert_d(
                d_at(src, &["x", "p"], &tree, &[x0, 0.0]),
                want,
                &format!("{src} at x={x0}, p=0"),
            );
        }
    }
}

/// SymbolicRegression.jl's `relu`, `cond`, safe operators, and Julia's `clamp` compare their
/// arguments, so on ForwardDiff's dual numbers a tie at a boundary resolves by the arguments'
/// derivatives: +1 for the argument `D` differentiates an operator by, and the chain's derivative
/// inside a custom operator. `abs` follows DynamicDiff's `sign` as an operator and ForwardDiff's
/// `signbit` rule inside a custom operator. Expected values are SymbolicRegression.jl f97764fb's
/// `D` (DynamicDiff 0.3.2, ForwardDiff 1.4.6).
#[test]
fn nonsmooth_operators_follow_forward_diff_at_their_boundaries() {
    use dynamic_expressions::operators::{abs, clamp};
    let builtin = BuiltinOpsF64::new();
    let relu = |tree: Tree| apply(&builtin, "relu", tree);
    // (tree, x, [D, D^2])
    let unary = [
        (relu(arg(1)), 0.0, [Some(1.0), Some(0.0)]),
        (relu(arg(1)), -0.0, [Some(1.0), Some(0.0)]),
        (relu(-arg(1)), 0.0, [Some(-1.0), Some(0.0)]),
        (abs(arg(1)), 0.0, [Some(0.0), Some(0.0)]),
        (abs(arg(1)), -0.0, [Some(0.0), Some(0.0)]),
        (abs(arg(1)), -2.0, [Some(-1.0), Some(0.0)]),
        (abs(-arg(1)), 0.0, [Some(0.0), Some(0.0)]),
    ];
    for (tree, x0, want) in unary {
        for (order, want) in (1..=2).zip(want) {
            let got = d_at(&nested_d(order, 1, "x"), &["x"], &tree, &[x0]);
            assert_d(got, want, &format!("D^{order} of {:?} at {x0}", tree.nodes));
        }
    }

    let cond_op = builtin.lookup_with_arity("cond", 2).unwrap();
    let cond = Tree::new(
        vec![
            PNode::Var { feature: 0 },
            PNode::Var { feature: 1 },
            PNode::Op {
                arity: 2,
                op: cond_op.id,
            },
        ],
        vec![],
        Metadata::default(),
    );
    for src in [
        "D(f, 1)(x, y)",
        "D(f, 2)(x, y)",
        "D(D(f, 2), 1)(x, y)",
        "D(D(f, 2), 2)(x, y)",
    ] {
        for x0 in [0.0, -0.0] {
            assert_d(
                d_at(src, &["x", "y"], &cond, &[x0, 3.0]),
                Some(0.0),
                &format!("{src} of cond at {x0}"),
            );
        }
    }

    let tree = clamp(arg(1), arg(2), arg(3));
    let srcs = [
        "D(f, 1)(x, lo, hi)",
        "D(f, 2)(x, lo, hi)",
        "D(f, 3)(x, lo, hi)",
        "D(D(f, 1), 1)(x, lo, hi)",
        "D(D(f, 2), 1)(x, lo, hi)",
        "D(D(f, 3), 3)(x, lo, hi)",
    ];
    let cases = [
        ([0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        ([1.0, 0.0, 1.0], [0.0; 6]),
        ([0.5, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ([1.0, 1.0, 1.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        ([2.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        ([-1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    ];
    for (point, want) in cases {
        for (src, want) in srcs.iter().zip(want) {
            let got = d_at(src, &["x", "lo", "hi"], &tree, &point);
            assert_d(got, Some(want), &format!("{src} at {point:?}"));
        }
    }

    let ops = custom_ops(&[
        "qs(u) = sqrt(-u)",
        "ql(u) = log(-u)",
        "qa(u) = asin(-u)",
        "qh(u) = acosh(-u)",
        "qr(u) = relu(-u)",
        "qc(u) = clamp(2 * u, 0, u)",
        "qabs(u) = abs(u)",
    ]);
    let custom = [
        ("qs", 0.0, [Some(0.0), Some(0.0)]),
        ("ql", 0.0, [Some(0.0), Some(0.0)]),
        ("qa", 1.0, [Some(0.0), Some(0.0)]),
        ("qa", -1.0, [None, None]),
        ("qh", -1.0, [Some(0.0), Some(0.0)]),
        ("qr", 0.0, [Some(0.0), Some(0.0)]),
        ("qc", 0.0, [Some(1.0), Some(0.0)]),
        ("qabs", 0.0, [Some(1.0), Some(0.0)]),
        ("qabs", -0.0, [Some(-1.0), Some(0.0)]),
        ("qabs", -2.0, [Some(-1.0), Some(0.0)]),
    ];
    for (name, x0, want) in custom {
        for (order, want) in (1..=2).zip(want) {
            let t = template(&nested_d(order, 1, "x"), &["f"], None, &["x"], &[], &ops);
            let got = eval(&t, &custom_call(&ops, name), &[], &ops, &[&[x0]])
                .unwrap()
                .map(|v| v[0]);
            assert_d(got, want, &format!("D^{order} {name} at {x0}"));
        }
    }
}

/// `clamp` is Julia's, `x > hi ? hi : x < lo ? lo : x`, so SymbolicRegression.jl evaluates it
/// with inverted bounds. Expected values are SymbolicRegression.jl f97764fb's.
#[test]
fn clamp_accepts_inverted_bounds() {
    use dynamic_expressions::operators::clamp;
    let tree = clamp(arg(1), arg(2), arg(3));
    let srcs = [
        "f(x, lo, hi)",
        "D(f, 1)(x, lo, hi)",
        "D(f, 2)(x, lo, hi)",
        "D(f, 3)(x, lo, hi)",
    ];
    let cases = [
        ([0.5, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
        ([2.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
        ([-1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]),
    ];
    for (point, want) in cases {
        for (src, want) in srcs.iter().zip(want) {
            let got = d_at(src, &["x", "lo", "hi"], &tree, &point);
            assert_d(got, Some(want), &format!("{src} at {point:?}"));
        }
    }

    let ops = BuiltinOpsF64::new();
    let t = template("f(x)", &["f"], None, &["x"], &[], &ops);
    let tree = clamp(arg(1), constant(1.0), constant(0.0));
    let got = eval(&t, &tree, &[], &ops, &[&[0.5, 2.0, -1.0]]).unwrap();
    assert_eq!(got, Some(vec![0.0, 0.0, 1.0]));
}

/// `min` and `max` are Julia's, so a NaN argument makes the result NaN and SymbolicRegression.jl
/// rejects the expression, in a subexpression or in the combining formula.
#[test]
fn min_and_max_propagate_nan() {
    use dynamic_expressions::Operator;
    use dynamic_expressions::operator_enum::builtin::{Max, Min};
    use dynamic_expressions::operators::{max, min};
    let min_of = |x: f64, y: f64| <Min as Operator<f64, 2>>::eval(&[x, y]);
    let max_of = |x: f64, y: f64| <Max as Operator<f64, 2>>::eval(&[x, y]);
    assert!(min_of(f64::NAN, 1.0).is_nan() && min_of(1.0, f64::NAN).is_nan());
    assert!(max_of(f64::NAN, 1.0).is_nan() && max_of(1.0, f64::NAN).is_nan());
    assert_eq!((min_of(1.0, 2.0), max_of(1.0, 2.0)), (1.0, 2.0));
    assert!(min_of(0.0, -0.0).is_sign_negative() && min_of(-0.0, 0.0).is_sign_negative());
    assert!(max_of(0.0, -0.0).is_sign_positive() && max_of(-0.0, 0.0).is_sign_positive());
    assert_eq!(min_of(f64::INFINITY, f64::INFINITY), f64::INFINITY);
    assert_eq!(max_of(f64::NEG_INFINITY, f64::NEG_INFINITY), f64::NEG_INFINITY);

    let ops = BuiltinOpsF64::new();
    let xs = [1.0, -2.0];
    for (combine, tree) in [
        ("f(x)", min(arg(1), constant(f64::NAN))),
        ("f(x)", max(constant(f64::NAN), arg(1))),
        ("min(NaN, f(x))", arg(1)),
        ("max(f(x), NaN)", arg(1)),
    ] {
        let t = template(combine, &["f"], None, &["x"], &[], &ops);
        assert_eq!(
            eval(&t, &tree, &[], &ops, &[&xs]).unwrap(),
            None,
            "{combine} with {:?}",
            tree.nodes
        );
    }
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

    // An unused constant shifts the features among the program's inputs.
    let used = arg(1) * arg(1) * 3.0;
    let padded = PostfixExpr::new(used.nodes.clone(), vec![3.0, 7.0], Metadata::default());
    for tree in [used.clone(), padded, used] {
        let trees = [tree];
        let plans = t.compile_plans(&trees);
        assert!(
            t.eval_into(&mut out, &trees, &plans, &[], &ops, x.view(), &mut ws)
                .unwrap()
        );
        assert_eq!(out, [6.0, 12.0, 18.0]);
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
