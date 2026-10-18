use dynamic_expressions::EvalOptions;
use dynamic_expressions::dispatch::{DiffKernelCtx, GradKernelCtx, GradRef, SrcRef};
use dynamic_expressions::dsl::lower::{Lowered, lower};
use dynamic_expressions::dsl::{DslFloat, ScalarFn, parse_definition};

fn lower_with(src: &str, customs: &[Lowered]) -> Lowered {
    let def = parse_definition(src).unwrap_or_else(|e| panic!("{src:?}: {e}"));
    lower(&def, customs).unwrap_or_else(|e| panic!("{src:?}: {e}"))
}

const NO_CHECK: EvalOptions = EvalOptions {
    check_finite: false,
    early_exit: false,
};

fn compile<T: DslFloat>(src: &str) -> ScalarFn<T> {
    ScalarFn::from_lowered(&lower_with(src, &[]))
}

fn eval_cols<T: DslFloat>(f: &ScalarFn<T>, cols: &[Vec<T>]) -> Vec<T> {
    let args: Vec<SrcRef<'_, T>> = cols.iter().map(|c| SrcRef::Slice(c.as_slice())).collect();
    let mut out = vec![T::zero(); cols[0].len()];
    assert!(f.eval(&mut out, &args, &NO_CHECK));
    out
}

/// `d f / d arg_j` per row, through `ScalarFn::diff` with a unit direction.
fn partial_cols(f: &ScalarFn<f64>, cols: &[Vec<f64>], j: usize) -> Vec<f64> {
    let n = cols[0].len();
    let args: Vec<SrcRef<'_, f64>> = cols.iter().map(|c| SrcRef::Slice(c.as_slice())).collect();
    let dargs: Vec<SrcRef<'_, f64>> = (0..cols.len())
        .map(|k| SrcRef::Const(if k == j { 1.0 } else { 0.0 }))
        .collect();
    let (mut val, mut der) = (vec![0.0; n], vec![0.0; n]);
    f.diff(DiffKernelCtx {
        out_val: &mut val,
        out_der: &mut der,
        args: &args,
        dargs: &dargs,
        opts: &NO_CHECK,
    });
    assert_eq!(val, eval_cols(f, cols));
    der
}

/// All partials at once through `ScalarFn::grad` with one basis direction per argument.
fn grad_cols(f: &ScalarFn<f64>, cols: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = cols[0].len();
    let args: Vec<SrcRef<'_, f64>> = cols.iter().map(|c| SrcRef::Slice(c.as_slice())).collect();
    let arg_grads: Vec<GradRef<'_, f64>> = (0..cols.len()).map(GradRef::Basis).collect();
    let mut val = vec![0.0; n];
    let mut grad = vec![f64::NAN; n * cols.len()];
    f.grad(GradKernelCtx {
        out_val: &mut val,
        out_grad: &mut grad,
        args: &args,
        arg_grads: &arg_grads,
        n_dir: cols.len(),
        n_rows: n,
        opts: &NO_CHECK,
    });
    grad.chunks(n).map(<[f64]>::to_vec).collect()
}

/// Checks every partial against central finite differences (h = 1e-6, rel tol 1e-6) on random
/// rows drawn from `ranges`, one range per argument.
#[track_caller]
fn check_partials_fd(src: &str, ranges: &[(f64, f64)]) {
    let f = compile::<f64>(src);
    assert_eq!(usize::from(f.arity()), ranges.len(), "{src}");
    let mut rng = fastrand::Rng::with_seed(17);
    let n = 64;
    let cols: Vec<Vec<f64>> = ranges
        .iter()
        .map(|&(lo, hi)| (0..n).map(|_| lo + (hi - lo) * rng.f64()).collect())
        .collect();
    let grads = grad_cols(&f, &cols);
    for (j, grad) in grads.iter().enumerate() {
        let analytic = partial_cols(&f, &cols, j);
        assert_eq!(&analytic, grad, "{src}: diff and grad disagree for argument {j}");
        for r in 0..n {
            let x = cols[j][r];
            let h = 1e-6 * x.abs().max(1.0);
            let at = |dx: f64| {
                let point: Vec<Vec<f64>> = cols
                    .iter()
                    .enumerate()
                    .map(|(k, c)| vec![if k == j { c[r] + dx } else { c[r] }])
                    .collect();
                eval_cols(&f, &point)[0]
            };
            let fd = (at(h) - at(-h)) / (2.0 * h);
            let err = (analytic[r] - fd).abs();
            assert!(
                err <= 1e-6 * fd.abs().max(1.0),
                "{src}: d/darg{j} at row {r} (args {:?}): analytic {} vs finite difference {fd}",
                cols.iter().map(|c| c[r]).collect::<Vec<_>>(),
                analytic[r]
            );
        }
    }
}

#[test]
fn programs_unary_kernels_match_finite_differences() {
    const ALL: (f64, f64) = (-3.0, 3.0);
    const POS: (f64, f64) = (0.1, 3.0);
    const NEG: (f64, f64) = (-3.0, -0.1);
    const UNIT: (f64, f64) = (-0.9, 0.9);
    for (src, range) in [
        ("f(x) = cos(x)", ALL),
        ("f(x) = sin(x)", ALL),
        ("f(x) = tan(x)", (-1.2, 1.2)),
        ("f(x) = asin(x)", UNIT),
        ("f(x) = acos(x)", UNIT),
        ("f(x) = atan(x)", ALL),
        ("f(x) = sinh(x)", ALL),
        ("f(x) = cosh(x)", ALL),
        ("f(x) = tanh(x)", ALL),
        ("f(x) = asinh(x)", ALL),
        ("f(x) = acosh(x)", (1.1, 3.0)),
        ("f(x) = atanh(x)", UNIT),
        ("f(x) = sec(x)", (-1.2, 1.2)),
        ("f(x) = csc(x)", (0.2, 2.9)),
        ("f(x) = cot(x)", (0.2, 2.9)),
        ("f(x) = exp(x)", ALL),
        ("f(x) = log(x)", POS),
        ("f(x) = log2(x)", POS),
        ("f(x) = log10(x)", POS),
        ("f(x) = log1p(x)", (-0.9, 3.0)),
        ("f(x) = exp2(x)", ALL),
        ("f(x) = expm1(x)", ALL),
        ("f(x) = sqrt(x)", POS),
        ("f(x) = cbrt(x)", POS),
        ("f(x) = abs(x)", POS),
        ("f(x) = abs(x)", NEG),
        ("f(x) = abs2(x)", ALL),
        ("f(x) = inv(x)", POS),
        ("f(x) = sign(x)", NEG),
        ("f(x) = identity(x)", ALL),
        ("f(x) = neg(x)", ALL),
        ("f(x) = -x", ALL),
        ("f(x) = relu(x)", POS),
        ("f(x) = relu(x)", NEG),
        ("f(x) = square(x)", ALL),
        ("f(x) = cube(x)", ALL),
        ("f(x) = 1 / x", POS),
    ] {
        check_partials_fd(src, &[range]);
    }
}

#[test]
fn programs_binary_and_ternary_kernels_match_finite_differences() {
    const ALL: (f64, f64) = (-3.0, 3.0);
    const POS: (f64, f64) = (0.1, 3.0);
    const LOW: (f64, f64) = (0.0, 1.0);
    const HIGH: (f64, f64) = (2.0, 3.0);
    for (src, ranges) in [
        ("f(x, y) = x + y", vec![ALL, ALL]),
        ("f(x, y) = x - y", vec![ALL, ALL]),
        ("f(x, y) = x * y", vec![ALL, ALL]),
        ("f(x, y) = x / y", vec![ALL, (0.5, 3.0)]),
        ("f(x, y) = x ^ y", vec![POS, (-2.0, 2.0)]),
        ("f(y, x) = atan(y, x)", vec![ALL, (0.5, 3.0)]),
        ("f(y, x) = atan2(y, x)", vec![ALL, (-3.0, -0.5)]),
        ("f(x, y) = min(x, y)", vec![LOW, HIGH]),
        ("f(x, y) = min(x, y)", vec![HIGH, LOW]),
        ("f(x, y) = max(x, y)", vec![LOW, HIGH]),
        ("f(x, y) = max(x, y)", vec![HIGH, LOW]),
        ("f(a, b, c) = min(a, b, c)", vec![HIGH, LOW, (4.0, 5.0)]),
        ("f(x, y) = cond(x, y)", vec![POS, ALL]),
        ("f(x, y) = cond(x, y)", vec![(-3.0, -0.1), ALL]),
        ("f(x, y, z) = fma(x, y, z)", vec![ALL, ALL, ALL]),
        (
            "f(x, lo, hi) = clamp(x, lo, hi)",
            vec![(-0.5, 0.5), (-3.0, -1.0), (1.0, 3.0)],
        ),
        (
            "f(x, lo, hi) = clamp(x, lo, hi)",
            vec![(-0.5, 0.5), (1.0, 2.0), (3.0, 4.0)],
        ),
        (
            "f(x, lo, hi) = clamp(x, lo, hi)",
            vec![(3.5, 4.0), (1.0, 2.0), (2.5, 3.0)],
        ),
        ("f(x, y) = x * exp(-y^2)", vec![ALL, ALL]),
        ("f(x, y) = sin(x)*cos(y) + exp(-(x - y)^2)/(1 + x^2)", vec![ALL, ALL]),
    ] {
        check_partials_fd(src, &ranges);
    }
}

#[test]
fn programs_ifelse_and_comparisons() {
    check_partials_fd("f(x) = x > 0 ? x^2 : -x", &[(0.1, 3.0)]);
    check_partials_fd("f(x) = x > 0 ? x^2 : -x", &[(-3.0, -0.1)]);
    check_partials_fd(
        "f(t, a, b) = ifelse(t > 0, a * b, a - b)",
        &[(0.1, 1.0), (-3.0, 3.0), (-3.0, 3.0)],
    );
    check_partials_fd(
        "f(t, a, b) = ifelse(t > 0, a * b, a - b)",
        &[(-1.0, -0.1), (-3.0, 3.0), (-3.0, 3.0)],
    );

    let cols = vec![vec![2.0, -2.0, 0.5], vec![1.0, 1.0, 1.0]];
    let f = compile::<f64>("f(x) = x > 0 ? x^2 : -x");
    assert_eq!(partial_cols(&f, &cols[..1], 0), vec![4.0, -1.0, 1.0]);
    for src in [
        "f(x, y) = x > y",
        "f(x, y) = x < y",
        "f(x, y) = x >= y",
        "f(x, y) = x <= y",
        "f(x, y) = x == y",
        "f(x, y) = x != y",
        "f(x, y) = x > 0 && y > 0",
        "f(x, y) = x > 0 || !(y > 0)",
    ] {
        let f = compile::<f64>(src);
        assert!(f.returns_bool(), "{src}");
        assert_eq!(f.partial_program(0).n_steps(), 0, "{src}");
        assert_eq!(partial_cols(&f, &cols, 0), vec![0.0; 3], "{src}");
        assert_eq!(partial_cols(&f, &cols, 1), vec![0.0; 3], "{src}");
    }
    let f = compile::<f64>("f(x, y) = x > y");
    assert_eq!(eval_cols(&f, &cols), vec![1.0, 0.0, 0.0]);
    let f = compile::<f64>("f(x, y) = x > 0 && y > 0 ? 1 : 2");
    assert!(!f.returns_bool());
    assert_eq!(eval_cols(&f, &cols), vec![1.0, 2.0, 1.0]);
}

#[test]
fn programs_reuse_temporaries() {
    let f = compile::<f64>("f(x) = exp(sin(cos(x)))");
    assert_eq!(f.value_program().n_steps(), 3);
    assert!(f.value_program().n_temps() <= 2);
    let f = compile::<f64>("f(x) = exp(sin(cos(tan(sinh(x)))))");
    assert_eq!(f.value_program().n_steps(), 5);
    assert!(f.value_program().n_temps() <= 2);
    let f = compile::<f64>("f(x) = sin(x) * sin(x)");
    assert_eq!(f.value_program().n_steps(), 2);
    let f = compile::<f64>("f(x, y) = (x - y)^2 + exp(x - y)");
    assert_eq!(f.value_program().n_steps(), 4);
    let x = vec![0.3, -1.2, 2.0];
    assert_eq!(
        eval_cols(
            &compile::<f64>("f(x) = exp(sin(cos(tan(sinh(x)))))"),
            std::slice::from_ref(&x)
        ),
        x.iter()
            .map(|v: &f64| v.sinh().tan().cos().sin().exp())
            .collect::<Vec<_>>()
    );
}

#[test]
fn programs_leaf_roots_and_const_arguments() {
    let f = compile::<f64>("f(x) = x");
    assert_eq!(f.value_program().n_steps(), 0);
    assert_eq!(eval_cols(&f, &[vec![1.0, 2.0]]), vec![1.0, 2.0]);
    let f = compile::<f64>("f(x) = 3");
    assert_eq!(eval_cols(&f, &[vec![1.0, 2.0]]), vec![3.0, 3.0]);

    let f = compile::<f64>("f(x, y) = x * y + 1");
    let x = [1.0, 2.0, 3.0];
    let mut out = [0.0; 3];
    assert!(f.eval(
        &mut out,
        &[SrcRef::Slice(&x), SrcRef::Const(2.0)],
        &EvalOptions::default()
    ));
    assert_eq!(out, [3.0, 5.0, 7.0]);
    assert!(f.eval(
        &mut out,
        &[SrcRef::Const(0.5), SrcRef::Const(4.0)],
        &EvalOptions::default()
    ));
    assert_eq!(out, [3.0, 3.0, 3.0]);
    let f = compile::<f64>("f(x, y) = y");
    assert!(f.eval(
        &mut out,
        &[SrcRef::Slice(&x), SrcRef::Const(4.0)],
        &EvalOptions::default()
    ));
    assert_eq!(out, [4.0, 4.0, 4.0]);
}

#[test]
fn programs_f32_and_f64() {
    let x: Vec<f32> = vec![0.25, -1.5, 2.0];
    let y: Vec<f32> = vec![1.0, 0.5, -0.75];
    let f = compile::<f32>("f(x, y) = x * exp(-y^2) + 1f-20");
    let expected: Vec<f32> = x
        .iter()
        .zip(&y)
        .map(|(&x, &y)| x * (-(y.powf(2.0))).exp() + 1e-20f32)
        .collect();
    assert_eq!(eval_cols(&f, &[x.clone(), y.clone()]), expected);

    let f = compile::<f64>("f(x, y) = x * exp(-y^2)");
    let (x64, y64): (Vec<f64>, Vec<f64>) = (
        x.iter().map(|&v| v as f64).collect(),
        y.iter().map(|&v| v as f64).collect(),
    );
    let expected: Vec<f64> = x64
        .iter()
        .zip(&y64)
        .map(|(&x, &y)| x * (-(y.powf(2.0))).exp())
        .collect();
    assert_eq!(eval_cols(&f, &[x64, y64]), expected);

    let f = compile::<f32>("f(x) = x * 1e300");
    assert_eq!(eval_cols(&f, &[vec![1.0f32]]), vec![f32::INFINITY]);
}

#[test]
fn programs_finiteness_rules() {
    let f = compile::<f64>("f(x) = 1 / x");
    let x = [1.0, 0.0, 2.0];
    let mut out = [0.0; 3];
    assert!(!f.eval(&mut out, &[SrcRef::Slice(&x)], &EvalOptions::default()));
    assert!(out.iter().all(|v| v.is_nan()));
    let keep_going = EvalOptions {
        check_finite: true,
        early_exit: false,
    };
    assert!(!f.eval(&mut out, &[SrcRef::Slice(&x)], &keep_going));
    assert_eq!(out, [1.0, f64::INFINITY, 0.5]);
    assert!(f.eval(&mut out, &[SrcRef::Slice(&x)], &NO_CHECK));
    assert!(f.eval(&mut out, &[SrcRef::Slice(&[1.0, 4.0, 2.0])], &EvalOptions::default()));

    let (mut val, mut der) = ([0.0; 3], [0.0; 3]);
    let ok = f.diff(DiffKernelCtx {
        out_val: &mut val,
        out_der: &mut der,
        args: &[SrcRef::Slice(&x)],
        dargs: &[SrcRef::Const(1.0)],
        opts: &EvalOptions::default(),
    });
    assert!(!ok);
    assert!(val.iter().chain(&der).all(|v| v.is_nan()));

    let mut grad = [0.0; 3];
    let ok = f.grad(GradKernelCtx {
        out_val: &mut val,
        out_grad: &mut grad,
        args: &[SrcRef::Slice(&x)],
        arg_grads: &[GradRef::Basis(0)],
        n_dir: 1,
        n_rows: 3,
        opts: &keep_going,
    });
    assert!(!ok);
    assert_eq!(val, [1.0, f64::INFINITY, 0.5]);
    assert_eq!(grad[0], -1.0);
}

#[test]
fn programs_chain_rule_through_argument_derivatives() {
    let f = compile::<f64>("f(x, y) = x * y");
    let (x, y) = ([1.0, 2.0], [3.0, 5.0]);
    let (dx, dy) = ([0.5, -1.0], [2.0, 0.25]);
    let (mut val, mut der) = ([0.0; 2], [0.0; 2]);
    assert!(f.diff(DiffKernelCtx {
        out_val: &mut val,
        out_der: &mut der,
        args: &[SrcRef::Slice(&x), SrcRef::Slice(&y)],
        dargs: &[SrcRef::Slice(&dx), SrcRef::Slice(&dy)],
        opts: &EvalOptions::default(),
    }));
    assert_eq!(val, [3.0, 10.0]);
    assert_eq!(der, [3.0 * 0.5 + 1.0 * 2.0, -5.0 + 2.0 * 0.25]);

    // Two directions: a dir-major slab for x, nothing for y.
    let x_grad = [1.0, 2.0, 10.0, 20.0];
    let mut grad = [f64::NAN; 4];
    assert!(f.grad(GradKernelCtx {
        out_val: &mut val,
        out_grad: &mut grad,
        args: &[SrcRef::Slice(&x), SrcRef::Slice(&y)],
        arg_grads: &[GradRef::Slice(&x_grad), GradRef::Zero],
        n_dir: 2,
        n_rows: 2,
        opts: &EvalOptions::default(),
    }));
    assert_eq!(grad, [3.0, 10.0, 30.0, 100.0]);
}
