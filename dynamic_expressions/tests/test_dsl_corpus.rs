//! Every scalar DSL string passed to a PySR option in PySR's docs and tests at commit c15a9eb.
//! Custom operators compile and match hand-written Rust; losses, early-stop conditions, and
//! template formulas parse here and compile in later PRs.

use dynamic_expressions::EvalOptions;
use dynamic_expressions::custom::CustomTable;
use dynamic_expressions::dispatch::{DiffKernelCtx, EvalKernelCtx, SrcRef};
use dynamic_expressions::dsl::{CustomOps, parse_definition, parse_formula, parse_operator_tokens};

const N_ROWS: usize = 257;
const NO_CHECK: EvalOptions = EvalOptions {
    check_finite: false,
    early_exit: false,
};

struct OperatorCase {
    id: &'static str,
    src: &'static str,
    /// PySR locations, as `path:line`.
    locations: &'static str,
    arity: usize,
    reference: fn(&[f64]) -> f64,
    /// Input range per argument.
    range: (f64, f64),
    /// Rows where a central finite difference is not meaningful (kinks, poles, ties).
    near_kink: fn(&[f64]) -> bool,
}

fn no_kink(_: &[f64]) -> bool {
    false
}

const OPERATORS: &[OperatorCase] = &[
    OperatorCase {
        id: "O1",
        src: "inv(x) = 1/x",
        locations: "README.md:161; docs/src/examples/getting-started.md:34, :56",
        arity: 1,
        reference: |a| 1.0 / a[0],
        range: (-3.0, 3.0),
        near_kink: |a| a[0].abs() < 1e-1,
    },
    OperatorCase {
        id: "O2",
        src: "cos2(x)=cos(x)^2",
        locations: "README.md:271",
        arity: 1,
        reference: |a| a[0].cos().powf(2.0),
        range: (-3.0, 3.0),
        near_kink: no_kink,
    },
    OperatorCase {
        id: "O3",
        src: "myfunction(x) = x^2",
        locations: "docs/src/operators.md:66",
        arity: 1,
        reference: |a| a[0].powf(2.0),
        range: (-3.0, 3.0),
        near_kink: no_kink,
    },
    OperatorCase {
        id: "O4",
        src: "myotherfunction(x, y) = x^2*y",
        locations: "docs/src/operators.md:67",
        arity: 2,
        reference: |a| a[0].powf(2.0) * a[1],
        range: (-3.0, 3.0),
        near_kink: no_kink,
    },
    OperatorCase {
        id: "O5",
        src: "special(x, y) = x^2 + y",
        locations: "docs/src/options.md:52",
        arity: 2,
        reference: |a| a[0].powf(2.0) + a[1],
        range: (-3.0, 3.0),
        near_kink: no_kink,
    },
    OperatorCase {
        id: "O6",
        src: "ifelse(t, a, b) = t > 0 ? a : b",
        locations: "docs/src/examples/search-behaviour.md:96; docs/src/examples/objectives.md:197",
        arity: 3,
        reference: |a| if a[0] > 0.0 { a[1] } else { a[2] },
        range: (-3.0, 3.0),
        near_kink: |a| a[0].abs() < 1e-3,
    },
    OperatorCase {
        id: "O7",
        src: "min4(a, b, c, d) = min(a, b, c, d)",
        locations: "docs/src/examples/search-behaviour.md:97",
        arity: 4,
        reference: |a| a[0].min(a[1]).min(a[2]).min(a[3]),
        range: (-3.0, 3.0),
        near_kink: |a| (0..4).any(|i| (0..i).any(|j| (a[i] - a[j]).abs() < 1e-3)),
    },
    OperatorCase {
        id: "O8",
        src: "cos_approx(x) = 1 - x^2 / 2 + x^4 / 24 + x^6 / 720",
        locations: "pysr/test/test_jax.py:135; pysr/test/test_torch.py:232",
        arity: 1,
        reference: |a| 1.0 - a[0].powf(2.0) / 2.0 + a[0].powf(4.0) / 24.0 + a[0].powf(6.0) / 720.0,
        range: (-3.0, 3.0),
        near_kink: no_kink,
    },
    OperatorCase {
        id: "O9",
        src: "square_op(x) = x^2",
        locations: "pysr/test/test_main.py:449",
        arity: 1,
        reference: |a| a[0].powf(2.0),
        range: (-3.0, 3.0),
        near_kink: no_kink,
    },
    OperatorCase {
        id: "O10",
        src: "sq(x) = x^2",
        locations: "pysr/test/test_main.py:548, :648, :802",
        arity: 1,
        reference: |a| a[0].powf(2.0),
        range: (-3.0, 3.0),
        near_kink: no_kink,
    },
    OperatorCase {
        id: "O11",
        src: "square2(x) = x^2",
        locations: "pysr/test/test_main.py:2456",
        arity: 1,
        reference: |a| a[0].powf(2.0),
        range: (-3.0, 3.0),
        near_kink: no_kink,
    },
    OperatorCase {
        id: "O12",
        src: "α(x, y) = x - y",
        locations: "pysr/test/test_main.py:2513",
        arity: 2,
        reference: |a| a[0] - a[1],
        range: (-3.0, 3.0),
        near_kink: no_kink,
    },
    OperatorCase {
        id: "O13",
        src: "my_add(x, y) = x + y",
        locations: "pysr/test/test_main.py:2771",
        arity: 2,
        reference: |a| a[0] + a[1],
        range: (-3.0, 3.0),
        near_kink: no_kink,
    },
    OperatorCase {
        id: "O14",
        src: "my_sub(x, y) = x - y",
        locations: "pysr/test/test_main.py:2772",
        arity: 2,
        reference: |a| a[0] - a[1],
        range: (-3.0, 3.0),
        near_kink: no_kink,
    },
    OperatorCase {
        id: "O15",
        src: "my_mul(x, y) = x * y",
        locations: "pysr/test/test_main.py:2773",
        arity: 2,
        reference: |a| a[0] * a[1],
        range: (-3.0, 3.0),
        near_kink: no_kink,
    },
    OperatorCase {
        id: "O16",
        src: "my_cos(x) = cos(x)",
        locations: "pysr/test/test_main.py:2775",
        arity: 1,
        reference: |a| a[0].cos(),
        range: (-3.0, 3.0),
        near_kink: no_kink,
    },
];

/// `N_ROWS` rows per argument, uniform in `range`, from a fixed seed.
fn columns(arity: usize, range: (f64, f64)) -> Vec<Vec<f64>> {
    let mut rng = fastrand::Rng::with_seed(0x5EED);
    (0..arity)
        .map(|_| (0..N_ROWS).map(|_| range.0 + (range.1 - range.0) * rng.f64()).collect())
        .collect()
}

fn row(cols: &[Vec<f64>], r: usize) -> Vec<f64> {
    cols.iter().map(|c| c[r]).collect()
}

fn same_value(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a == b || (a - b).abs() <= 1e-12 * a.abs().max(b.abs())
}

fn compile(case: &OperatorCase) -> CustomOps<f64> {
    let mut ops = CustomOps::default();
    let defined = ops
        .define(case.src)
        .unwrap_or_else(|e| panic!("{} ({}): {e}", case.id, case.locations));
    assert_eq!(usize::from(defined.arity), case.arity, "{}", case.id);
    ops
}

fn eval(ops: &CustomOps<f64>, cols: &[Vec<f64>]) -> Vec<f64> {
    let args: Vec<SrcRef<'_, f64>> = cols.iter().map(|c| SrcRef::Slice(c)).collect();
    let mut out = vec![0.0; N_ROWS];
    ops.eval(
        0,
        EvalKernelCtx {
            out: &mut out,
            args: &args,
            opts: &NO_CHECK,
        },
    );
    out
}

/// The partial derivative with respect to argument `j`, through `CustomTable::diff`.
fn partial(ops: &CustomOps<f64>, cols: &[Vec<f64>], j: usize) -> Vec<f64> {
    let args: Vec<SrcRef<'_, f64>> = cols.iter().map(|c| SrcRef::Slice(c)).collect();
    let dargs: Vec<SrcRef<'_, f64>> = (0..cols.len())
        .map(|k| SrcRef::Const(if k == j { 1.0 } else { 0.0 }))
        .collect();
    let mut out_val = vec![0.0; N_ROWS];
    let mut out_der = vec![0.0; N_ROWS];
    ops.diff(
        0,
        DiffKernelCtx {
            out_val: &mut out_val,
            out_der: &mut out_der,
            args: &args,
            dargs: &dargs,
            opts: &NO_CHECK,
        },
    );
    out_der
}

#[test]
fn corpus_operators_match_hand_written_rust() {
    for case in OPERATORS {
        let ops = compile(case);
        let cols = columns(case.arity, case.range);
        let got = eval(&ops, &cols);
        for (r, &g) in got.iter().enumerate() {
            let x = row(&cols, r);
            let want = (case.reference)(&x);
            assert!(
                same_value(g, want),
                "{} `{}` ({}) at {x:?}: got {g}, want {want}",
                case.id,
                case.src,
                case.locations
            );
        }
    }
}

#[test]
fn corpus_operator_derivatives_match_central_differences() {
    for case in OPERATORS {
        let ops = compile(case);
        let cols = columns(case.arity, case.range);
        for j in 0..case.arity {
            let got = partial(&ops, &cols, j);
            for (r, &g) in got.iter().enumerate() {
                let x = row(&cols, r);
                if (case.near_kink)(&x) {
                    continue;
                }
                let h = 1e-6 * x[j].abs().max(1.0);
                let (mut hi, mut lo) = (x.clone(), x.clone());
                hi[j] += h;
                lo[j] -= h;
                let fd = ((case.reference)(&hi) - (case.reference)(&lo)) / (2.0 * h);
                assert!(
                    (g - fd).abs() <= 1e-6 * fd.abs().max(1.0),
                    "{} `{}` d/d{j} at {x:?}: got {g}, finite difference {fd}",
                    case.id,
                    case.src
                );
            }
        }
    }
}

/// Elementwise losses (compiled and checked in PR 2), with their PySR locations.
const LOSSES: &[(&str, &str)] = &[
    ("L1", "loss(prediction, target) = (prediction - target)^2"),
    ("L2", "myloss(x, y, w) = w * (x - y)^2"),
    ("L3", "myloss(x, y) = abs(x - y)^1.5"),
    ("L4", "f(x, y) = abs(x - y)^1.5"),
    ("L5", "loss(x, y) = abs(x * y)"),
    ("L6", "myloss(x, y, w) = w * abs(x - y)"),
    ("L7", "myloss(x, y, w) = abs(x - y)^2/w^2"),
    (
        "L8",
        "function loss_fnc(prediction, target)\n    scatter_loss = abs(log((abs(prediction)+1f-20) / (abs(target)+1f-20)))\n    sign_loss = 10 * (sign(prediction) - sign(target))^2\n    return scatter_loss + sign_loss\nend\n",
    ),
    ("L9", "(pred, target) -> pred"),
    ("L10", "my_loss(prediction, target) = (prediction - target)^2"),
    ("L11", "myloss_bad_arity(a) = a"),
    ("L12", "myloss2(prediction, target) = (prediction - target)^2"),
    (
        "L13",
        "myloss3(prediction, target, weights) = weights * (prediction - target)^2",
    ),
    (
        "L14",
        "(prediction::Float32, target::Float32) -> (prediction - target)^2",
    ),
    ("L15", "(pred, targ) -> pred"),
    ("L16", "g(x, y) = 0.0"),
    ("L17", "(prediction, target) -> 0.0"),
    ("L18", "(p, t) -> 0.0"),
];

/// Early-stop conditions (compiled and checked in PR 2).
const EARLY_STOPS: &[(&str, &str)] = &[
    ("E1", "stop_if(loss, complexity) = loss < 1e-6 && complexity < 10"),
    ("E2", "stop_if(loss, complexity) = loss < 1e-4 && complexity <= 5"),
    ("E3", "stop_if(loss, complexity) = loss < 1e-4 && complexity == 1"),
    ("E4", "stop_if(loss, complexity) = loss < 1e-10 && complexity == 1"),
    ("E5", "stop_if(loss, complexity) = loss < 1e-4 && complexity == 3"),
    ("E6", "stop_if_1(l, c) = l < 1e-8 && c <= 3"),
    ("E7", "stop_if_2(l, c) = l < 1e-8 && c <= 2"),
    ("E8", "stop_if(loss, complexity) = loss < 1e-4 && complexity == 2"),
    ("E9", "(loss, complexity) -> loss <= 1e-4 && complexity <= 6"),
    ("E10", "stop_if(loss, complexity) = loss < 0.05 && complexity == 2"),
    ("E11", "stop_if(loss, complexity) = loss < 1e-3 && complexity == 7"),
    ("E12", "stop_if(loss, complexity) = loss < 1e-10 && complexity == 6"),
    ("E13", "stop_if(loss, complexity) = loss < 1e-10 && complexity <= 3"),
    ("E14", "stop_if_under_n1(loss, complexity) = loss < -1.0"),
    ("E15", "stop_if(loss, complexity) = loss < 1e-4 && complexity <= 7"),
    ("E16", "stop_if(loss, complexity) = loss < 1e-4 && complexity <= 10"),
    ("E17", "stop_if(loss, complexity) = loss < 1e-6 && complexity <= 5"),
    ("E18", "(l, c) -> l < 1e-6 && c == 3"),
    ("E19", "(l, c) -> l < 1e-6 && c == 1"),
    ("E20", "checkpoint_early_stop(loss, complexity) = false"),
    ("E21", "(loss, complexity) -> loss < 1.0e-8 && complexity == 5"),
    ("E22", "(loss, complexity) -> false"),
];

/// Template combine formulas (compiled and checked in PR 3).
const TEMPLATES: &[(&str, &str)] = &[
    ("T1", "sin(f(x1, x2)) + g(x3)"),
    ("T2", "f(x1, x2, p1[category], p2[category])"),
    (
        "T3",
        "\n        v = shared(x1, x2, x3)\n        y1_predicted = v + f1(x1, x2, x3)\n        y2_predicted = v + f2(x1, x2, x3)\n        y3_predicted = v + f3(x1, x2, x3)\n\n        residuals = (\n            abs2(y1 - y1_predicted) +\n            abs2(y2 - y2_predicted) +\n            abs2(y3 - y3_predicted)\n        )\n\n        residuals\n    ",
    ),
    ("T4", "df = D(f, 1); df(x)"),
    ("T5", "sin(f(x1, x2)) + g(x3)^2"),
    ("T6", "sin(f(x, y))"),
    ("T7", "p[class] * x1^2 + f(x2)"),
    (
        "T8",
        "mu = f_mu(x1, x2, x3); logvar = f_logvar(x1, x2, x3); 0.5f0 * (logvar + (mu - y)^2 / exp(logvar)) - 2",
    ),
    ("T9", "f(x0, x1)"),
    ("T10", "p[1] * f(x) + p[2] * x + p[3]"),
    ("T11", "p[1] * f(x) + p[2]"),
    ("T12", "f(x)"),
    ("T13", "f(x, y) + g(z)"),
    ("T14", "p[1] * f(x)"),
    ("T15", "p1[1]*f(x,y) + p2[1]*g(z)"),
    ("T16", "f(var1) * g(var2)"),
    ("T17", "alpha*f(x) + beta*g(y)"),
    ("T18", "add_vectors(f(x1), g(x2))"),
    ("T19", "choose_parameter(p[1], f(x))"),
    ("T20", "f0(x) + f1(x) + f2(x) + f3(x) + f4(x)"),
];

#[test]
fn corpus_losses_and_early_stops_parse() {
    for (id, src) in LOSSES.iter().chain(EARLY_STOPS) {
        if let Err(e) = parse_definition(src) {
            panic!("{id} `{src}`: {e}");
        }
    }
}

#[test]
fn corpus_templates_parse() {
    for (id, src) in TEMPLATES {
        if let Err(e) = parse_formula(src) {
            panic!("{id} `{src}`: {e}");
        }
    }
}

#[test]
fn custom_value_type_strings_are_rejected_with_the_construct() {
    let cases = [
        (
            "nn_relu(a) = NNValue(max.(a.data, 0.0))",
            "broadcasting (`.`) is not supported",
        ),
        (
            "shift(a::Letter, b::Letter) = Letter(mod(a.code + b.code, 26))",
            "type `Letter` is not supported",
        ),
        (
            "cell_loss(prediction::Cell, target::Cell)::Float64 = prediction.v == target.v ? 0.0 : 1.0",
            "type `Cell` is not supported",
        ),
    ];
    for (src, message) in cases {
        let err = parse_definition(src).expect_err(src);
        assert!(err.message.contains(message), "`{src}`: {}", err.message);
    }
}

#[test]
fn external_package_functions_fail_at_setup_naming_the_function() {
    let cases = [
        ("p(i) = Primes.prime(round(i))", "`Primes.prime`"),
        ("myerf(x) = SpecialFunctions.erf(x)", "`SpecialFunctions.erf`"),
        ("myerf(x) = erf(x)", "unknown function `erf`"),
    ];
    for (src, name) in cases {
        let Err(err) = parse_operator_tokens::<f64, _>(["+", src]) else {
            panic!("`{src}` was accepted");
        };
        assert!(err.message.contains(name), "`{src}`: {}", err.message);
    }
}
