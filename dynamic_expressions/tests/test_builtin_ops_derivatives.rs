use dynamic_expressions::Operator;
use dynamic_expressions::operator_enum::builtin::*;

fn fd_unary<Op: Operator<f64, 1>>(x: f64) -> f64 {
    let eps = 1e-7;
    let vp = <Op as Operator<f64, 1>>::eval(&[x + eps]);
    let vm = <Op as Operator<f64, 1>>::eval(&[x - eps]);
    (vp - vm) / (2.0 * eps)
}

fn check_unary<Op: Operator<f64, 1>>(x: f64, tol: f64) {
    let d = <Op as Operator<f64, 1>>::partial(&[x], 0);
    let fd = fd_unary::<Op>(x);
    assert!((d - fd).abs() <= tol.max(tol * fd.abs()), "d={} fd={} (x={})", d, fd, x);
}

fn fd_binary_x<Op: Operator<f64, 2>>(x: f64, y: f64) -> f64 {
    let eps = 1e-7;
    let vp = <Op as Operator<f64, 2>>::eval(&[x + eps, y]);
    let vm = <Op as Operator<f64, 2>>::eval(&[x - eps, y]);
    (vp - vm) / (2.0 * eps)
}

fn fd_binary_y<Op: Operator<f64, 2>>(x: f64, y: f64) -> f64 {
    let eps = 1e-7;
    let vp = <Op as Operator<f64, 2>>::eval(&[x, y + eps]);
    let vm = <Op as Operator<f64, 2>>::eval(&[x, y - eps]);
    (vp - vm) / (2.0 * eps)
}

fn check_binary<Op: Operator<f64, 2>>(x: f64, y: f64, tol: f64) {
    let dx = <Op as Operator<f64, 2>>::partial(&[x, y], 0);
    let dy = <Op as Operator<f64, 2>>::partial(&[x, y], 1);
    let fdx = fd_binary_x::<Op>(x, y);
    let fdy = fd_binary_y::<Op>(x, y);
    assert!(
        (dx - fdx).abs() <= tol.max(tol * fdx.abs()),
        "dx={} fdx={} (x={}, y={})",
        dx,
        fdx,
        x,
        y
    );
    assert!(
        (dy - fdy).abs() <= tol.max(tol * fdy.abs()),
        "dy={} fdy={} (x={}, y={})",
        dy,
        fdy,
        x,
        y
    );
}

#[test]
fn unary_derivatives_match_finite_difference() {
    // Choose points away from singularities / domain boundaries.
    check_unary::<Sin>(0.3, 1e-6);
    check_unary::<Cos>(0.3, 1e-6);
    check_unary::<Tan>(0.3, 1e-5);

    check_unary::<Asin>(0.2, 1e-5);
    check_unary::<Acos>(0.2, 1e-5);
    check_unary::<Atan>(0.2, 1e-6);

    check_unary::<Sinh>(0.2, 1e-6);
    check_unary::<Cosh>(0.2, 1e-6);
    check_unary::<Tanh>(0.2, 1e-6);

    check_unary::<Asinh>(0.2, 1e-6);
    check_unary::<Acosh>(2.0, 1e-6);
    check_unary::<Atanh>(0.2, 1e-5);

    check_unary::<Sec>(0.3, 1e-5);
    check_unary::<Csc>(1.2, 1e-5);
    check_unary::<Cot>(1.2, 1e-5);

    check_unary::<Exp>(0.2, 1e-6);
    check_unary::<Exp2>(0.2, 1e-6);
    check_unary::<Expm1>(0.2, 1e-6);

    check_unary::<Log>(1.3, 1e-6);
    check_unary::<Log2>(1.3, 1e-6);
    check_unary::<Log10>(1.3, 1e-6);
    check_unary::<Log1p>(0.2, 1e-6);

    check_unary::<Sqrt>(2.0, 1e-6);
    check_unary::<Cbrt>(2.0, 1e-5);

    check_unary::<Inv>(2.0, 1e-6);
    check_unary::<Abs2>(-0.7, 1e-6);

    check_unary::<Relu>(0.7, 1e-6);
    check_unary::<Relu>(-0.7, 1e-6);
    check_unary::<Square>(-0.7, 1e-6);
    check_unary::<Cube>(-0.7, 1e-6);
}

#[test]
fn binary_derivatives_match_finite_difference() {
    check_binary::<Add>(0.3, -0.7, 1e-6);
    check_binary::<Sub>(0.3, -0.7, 1e-6);
    check_binary::<Mul>(0.3, -0.7, 1e-6);
    check_binary::<Div>(0.3, 1.7, 1e-6);

    check_binary::<Atan2>(0.3, 1.7, 1e-6);
    check_binary::<Pow>(1.3, 0.7, 1e-5);

    check_binary::<Cond>(0.3, -0.7, 1e-6);
    check_binary::<Cond>(-0.3, -0.7, 1e-6);
}

fn unary(op: impl Fn(&[f64; 1]) -> f64, x: f64) -> f64 {
    op(&[x])
}

#[test]
fn log_and_pow_follow_symbolic_regression_domains() {
    for x in [0.0, -0.0, -1.5, f64::NEG_INFINITY] {
        assert!(unary(<Log as Operator<f64, 1>>::eval, x).is_nan(), "log({x})");
        assert!(unary(<Log2 as Operator<f64, 1>>::eval, x).is_nan(), "log2({x})");
        assert!(unary(<Log10 as Operator<f64, 1>>::eval, x).is_nan(), "log10({x})");
    }
    assert_eq!(unary(<Log as Operator<f64, 1>>::eval, 1.0), 0.0);
    assert_eq!(unary(<Log2 as Operator<f64, 1>>::eval, 8.0), 3.0);
    assert_eq!(unary(<Log10 as Operator<f64, 1>>::eval, 100.0), 2.0);

    assert!(unary(<Log1p as Operator<f64, 1>>::eval, -1.0).is_nan());
    assert!(unary(<Log1p as Operator<f64, 1>>::eval, -2.0).is_nan());
    assert_eq!(unary(<Log1p as Operator<f64, 1>>::eval, 0.0), 0.0);
    assert!(<Log1p as Operator<f32, 1>>::eval(&[-1.0f32]).is_nan());

    assert!(<Pow as Operator<f64, 2>>::eval(&[0.0, -1.0]).is_nan());
    assert!(<Pow as Operator<f64, 2>>::eval(&[0.0, -0.5]).is_nan());
    assert_eq!(<Pow as Operator<f64, 2>>::eval(&[0.0, 0.0]), 1.0);
    assert_eq!(<Pow as Operator<f64, 2>>::eval(&[0.0, 2.0]), 0.0);
    assert_eq!(<Pow as Operator<f64, 2>>::eval(&[-2.0, 3.0]), -8.0);
    assert!(<Pow as Operator<f64, 2>>::eval(&[-2.0, 0.5]).is_nan());
}

#[test]
fn comparison_and_logical_ops_return_zero_or_one() {
    let cases = [(1.0, 2.0), (2.0, 1.0), (1.0, 1.0), (f64::NAN, 1.0)];
    type Case = (&'static str, fn(&[f64; 2]) -> f64, [f64; 4]);
    let expected: [Case; 6] = [
        ("greater", <Greater as Operator<f64, 2>>::eval, [0.0, 1.0, 0.0, 0.0]),
        ("less", <Less as Operator<f64, 2>>::eval, [1.0, 0.0, 0.0, 0.0]),
        (
            "greater_equal",
            <GreaterEqual as Operator<f64, 2>>::eval,
            [0.0, 1.0, 1.0, 0.0],
        ),
        (
            "less_equal",
            <LessEqual as Operator<f64, 2>>::eval,
            [1.0, 0.0, 1.0, 0.0],
        ),
        ("equal", <Equal as Operator<f64, 2>>::eval, [0.0, 0.0, 1.0, 0.0]),
        ("not_equal", <NotEqual as Operator<f64, 2>>::eval, [1.0, 1.0, 0.0, 1.0]),
    ];
    for (name, eval, want) in expected {
        for ((x, y), w) in cases.iter().zip(want) {
            assert_eq!(eval(&[*x, *y]), w, "{name}({x}, {y})");
        }
    }
    assert_eq!(<Greater as Operator<f64, 2>>::partial(&[2.0, 1.0], 0), 0.0);
    assert_eq!(<Greater as Operator<f64, 2>>::partial(&[2.0, 1.0], 1), 0.0);

    let and = <LogicalAnd as Operator<f64, 2>>::eval;
    let or = <LogicalOr as Operator<f64, 2>>::eval;
    let not = <LogicalNot as Operator<f64, 1>>::eval;
    assert_eq!(
        [and(&[1.0, 2.0]), and(&[1.0, 0.0]), and(&[-1.0, -1.0])],
        [1.0, 0.0, 0.0]
    );
    assert_eq!([or(&[1.0, 0.0]), or(&[0.0, -1.0]), or(&[0.0, 0.5])], [1.0, 0.0, 1.0]);
    assert_eq!(
        [not(&[1.0]), not(&[0.0]), not(&[-1.0]), not(&[f64::NAN])],
        [0.0, 1.0, 1.0, 1.0]
    );
    assert_eq!(<LogicalAnd as Operator<f64, 2>>::partial(&[1.0, 2.0], 1), 0.0);
    assert_eq!(<LogicalNot as Operator<f64, 1>>::partial(&[1.0], 0), 0.0);
}

#[test]
fn select_ops_route_values_and_partials() {
    let ifelse = <IfElse as Operator<f64, 3>>::eval;
    assert_eq!(ifelse(&[1.0, 2.0, 3.0]), 2.0);
    assert_eq!(ifelse(&[0.0, 2.0, 3.0]), 3.0);
    assert_eq!(ifelse(&[f64::NAN, 2.0, 3.0]), 3.0);
    let d = |args: [f64; 3]| [0, 1, 2].map(|i| <IfElse as Operator<f64, 3>>::partial(&args, i));
    assert_eq!(d([1.0, 2.0, 3.0]), [0.0, 1.0, 0.0]);
    assert_eq!(d([-1.0, 2.0, 3.0]), [0.0, 0.0, 1.0]);

    let cond = <Cond as Operator<f64, 2>>::eval;
    assert_eq!(cond(&[1.0, 5.0]), 5.0);
    assert_eq!(cond(&[0.0, 5.0]), 0.0);
    assert_eq!(<Cond as Operator<f64, 2>>::partial(&[1.0, 5.0], 0), 0.0);
    assert_eq!(<Cond as Operator<f64, 2>>::partial(&[1.0, 5.0], 1), 1.0);
    assert_eq!(<Cond as Operator<f64, 2>>::partial(&[-1.0, 5.0], 1), 0.0);

    assert_eq!(unary(<Relu as Operator<f64, 1>>::eval, -2.0), 0.0);
    assert_eq!(unary(<Relu as Operator<f64, 1>>::eval, 2.0), 2.0);
    assert_eq!(<Relu as Operator<f64, 1>>::partial(&[0.0], 0), 0.0);
}

#[test]
fn nonsmooth_ops_have_reasonable_partials_off_boundaries() {
    // Abs: derivative is ±1 away from 0.
    assert_eq!(<Abs as Operator<f64, 1>>::partial(&[2.0], 0), 1.0);
    assert_eq!(<Abs as Operator<f64, 1>>::partial(&[-2.0], 0), -1.0);
    assert_eq!(<Abs as Operator<f64, 1>>::partial(&[0.0], 0), 0.0);

    // Sign/Identity are simple.
    assert_eq!(<Sign as Operator<f64, 1>>::partial(&[2.0], 0), 0.0);
    assert_eq!(<Identity as Operator<f64, 1>>::partial(&[2.0], 0), 1.0);

    // Min/Max: away from ties they are 0/1.
    assert_eq!(<Min as Operator<f64, 2>>::partial(&[1.0, 2.0], 0), 1.0);
    assert_eq!(<Min as Operator<f64, 2>>::partial(&[1.0, 2.0], 1), 0.0);
    assert_eq!(<Max as Operator<f64, 2>>::partial(&[1.0, 2.0], 0), 0.0);
    assert_eq!(<Max as Operator<f64, 2>>::partial(&[1.0, 2.0], 1), 1.0);
    assert_eq!(<Min as Operator<f64, 2>>::partial(&[2.0, 1.0], 0), 0.0);
    assert_eq!(<Min as Operator<f64, 2>>::partial(&[2.0, 1.0], 1), 1.0);
    assert_eq!(<Max as Operator<f64, 2>>::partial(&[2.0, 1.0], 0), 1.0);
    assert_eq!(<Max as Operator<f64, 2>>::partial(&[2.0, 1.0], 1), 0.0);
    // Reverse order exercises the other eval branch.
    assert_eq!(<Min as Operator<f64, 2>>::eval(&[2.0, 1.0]), 1.0);
    assert_eq!(<Max as Operator<f64, 2>>::eval(&[2.0, 1.0]), 2.0);
    // Tie splits gradient evenly.
    assert_eq!(<Min as Operator<f64, 2>>::partial(&[2.0, 2.0], 0), 0.5);
    assert_eq!(<Min as Operator<f64, 2>>::partial(&[2.0, 2.0], 1), 0.5);
    assert_eq!(<Max as Operator<f64, 2>>::partial(&[2.0, 2.0], 0), 0.5);
    assert_eq!(<Max as Operator<f64, 2>>::partial(&[2.0, 2.0], 1), 0.5);

    // Clamp: inside bounds, derivative wrt x is 1; below/above it's 0.
    let args = [0.0, -1.0, 1.0];
    assert_eq!(<Clamp as Operator<f64, 3>>::partial(&args, 0), 1.0);
    let args = [-2.0, -1.0, 1.0];
    assert_eq!(<Clamp as Operator<f64, 3>>::partial(&args, 0), 0.0);
    assert_eq!(<Clamp as Operator<f64, 3>>::partial(&args, 1), 1.0);
    // When x >= lo, derivative wrt lo is 0.
    let args = [0.0, -1.0, 1.0];
    assert_eq!(<Clamp as Operator<f64, 3>>::partial(&args, 1), 0.0);
    let args = [2.0, -1.0, 1.0];
    assert_eq!(<Clamp as Operator<f64, 3>>::partial(&args, 0), 0.0);
    assert_eq!(<Clamp as Operator<f64, 3>>::partial(&args, 2), 1.0);
    // When x <= hi, derivative wrt hi is 0.
    let args = [0.0, -1.0, 1.0];
    assert_eq!(<Clamp as Operator<f64, 3>>::partial(&args, 2), 0.0);
}

#[test]
fn sign_matches_julia_at_zero_and_nan() {
    let sign = |x: f64| <Sign as Operator<f64, 1>>::eval(&[x]);
    assert_eq!(sign(2.5), 1.0);
    assert_eq!(sign(-2.5), -1.0);
    assert_eq!(sign(0.0).to_bits(), 0.0f64.to_bits());
    assert_eq!(sign(-0.0).to_bits(), (-0.0f64).to_bits());
    assert!(sign(f64::NAN).is_nan());
    assert_eq!(<Sign as Operator<f32, 1>>::eval(&[0.0f32]), 0.0);
}
