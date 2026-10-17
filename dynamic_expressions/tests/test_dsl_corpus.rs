//! Every scalar DSL string passed to a PySR option in PySR's docs and tests at commit c15a9eb.
//! Custom operators, elementwise losses, early-stop conditions, and template formulas compile and
//! match hand-written Rust. Template formulas that call functions defined only for custom value
//! types fail at setup.

use dynamic_expressions::custom::CustomTable;
use dynamic_expressions::dispatch::{DiffKernelCtx, EvalKernelCtx, SrcRef};
use dynamic_expressions::dsl::{CustomOps, ScalarFn, TemplateStructure, parse_definition, parse_operator_tokens};
use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
use dynamic_expressions::operators::cos;
use dynamic_expressions::{EvalOptions, PNode, PostfixExpr};
use ndarray::Array2;

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

struct LossCase {
    id: &'static str,
    src: &'static str,
    locations: &'static str,
    arity: usize,
    /// Hand-written loss of `(prediction, target[, weight])`.
    reference: fn(&[f64]) -> f64,
    /// Rows where a central finite difference in the prediction is not meaningful.
    near_kink: fn(&[f64]) -> bool,
}

fn julia_sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        x
    }
}

/// Elementwise losses. L11 is PySR's bad-arity example; it compiles as a function here, and
/// `symbolic_regression::parse_loss` rejects it as a loss.
const LOSSES: &[LossCase] = &[
    LossCase {
        id: "L1",
        src: "loss(prediction, target) = (prediction - target)^2",
        locations: "README.md:167; pysr/test/test_main.py:183",
        arity: 2,
        reference: |a| (a[0] - a[1]).powi(2),
        near_kink: no_kink,
    },
    LossCase {
        id: "L2",
        src: "myloss(x, y, w) = w * (x - y)^2",
        locations: "docs/src/examples/getting-started.md:127",
        arity: 3,
        reference: |a| a[2] * (a[0] - a[1]).powi(2),
        near_kink: no_kink,
    },
    LossCase {
        id: "L3",
        src: "myloss(x, y) = abs(x - y)^1.5",
        locations: "docs/src/options.md:246",
        arity: 2,
        reference: |a| (a[0] - a[1]).abs().powf(1.5),
        near_kink: |a| (a[0] - a[1]).abs() < 1e-3,
    },
    LossCase {
        id: "L4",
        src: "f(x, y) = abs(x - y)^1.5",
        locations: "docs/src/options.md:256",
        arity: 2,
        reference: |a| (a[0] - a[1]).abs().powf(1.5),
        near_kink: |a| (a[0] - a[1]).abs() < 1e-3,
    },
    LossCase {
        id: "L5",
        src: "loss(x, y) = abs(x * y)",
        locations: "docs/src/options.md:262",
        arity: 2,
        reference: |a| (a[0] * a[1]).abs(),
        near_kink: |a| a[0].abs() < 1e-3,
    },
    LossCase {
        id: "L6",
        src: "myloss(x, y, w) = w * abs(x - y)",
        locations: "docs/src/options.md:268; examples/pysr_demo.ipynb:579; skills/pysr/SKILL.md:88",
        arity: 3,
        reference: |a| a[2] * (a[0] - a[1]).abs(),
        near_kink: |a| (a[0] - a[1]).abs() < 1e-3,
    },
    LossCase {
        id: "L7",
        src: "myloss(x, y, w) = abs(x - y)^2/w^2",
        locations: "docs/src/options.md:275",
        arity: 3,
        reference: |a| (a[0] - a[1]).abs().powi(2) / a[2].powi(2),
        near_kink: no_kink,
    },
    LossCase {
        id: "L8",
        src: "function loss_fnc(prediction, target)\n    scatter_loss = abs(log((abs(prediction)+1f-20) / (abs(target)+1f-20)))\n    sign_loss = 10 * (sign(prediction) - sign(target))^2\n    return scatter_loss + sign_loss\nend\n",
        locations: "docs/src/examples/physics.md:24",
        arity: 2,
        reference: |a| {
            let scatter_loss = ((a[0].abs() + 1e-20) / (a[1].abs() + 1e-20)).ln().abs();
            let sign_loss = 10.0 * (julia_sign(a[0]) - julia_sign(a[1])).powi(2);
            scatter_loss + sign_loss
        },
        near_kink: |a| a[0].abs() < 1e-2 || (a[0].abs() - a[1].abs()).abs() < 1e-3,
    },
    LossCase {
        id: "L9",
        src: "(pred, target) -> pred",
        locations: "docs/src/examples/expression-specifications.md:222",
        arity: 2,
        reference: |a| a[0],
        near_kink: no_kink,
    },
    LossCase {
        id: "L10",
        src: "my_loss(prediction, target) = (prediction - target)^2",
        locations: "pysr/test/test_main.py:183",
        arity: 2,
        reference: |a| (a[0] - a[1]).powi(2),
        near_kink: no_kink,
    },
    LossCase {
        id: "L11",
        src: "myloss_bad_arity(a) = a",
        locations: "pysr/test/test_main.py:286",
        arity: 1,
        reference: |a| a[0],
        near_kink: no_kink,
    },
    LossCase {
        id: "L12",
        src: "myloss2(prediction, target) = (prediction - target)^2",
        locations: "pysr/test/test_main.py:303",
        arity: 2,
        reference: |a| (a[0] - a[1]).powi(2),
        near_kink: no_kink,
    },
    LossCase {
        id: "L13",
        src: "myloss3(prediction, target, weights) = weights * (prediction - target)^2",
        locations: "pysr/test/test_main.py:323",
        arity: 3,
        reference: |a| a[2] * (a[0] - a[1]).powi(2),
        near_kink: no_kink,
    },
    LossCase {
        id: "L14",
        src: "(prediction::Float32, target::Float32) -> (prediction - target)^2",
        locations: "pysr/test/test_main.py:333, :352",
        arity: 2,
        reference: |a| (a[0] - a[1]).powi(2),
        near_kink: no_kink,
    },
    LossCase {
        id: "L15",
        src: "(pred, targ) -> pred",
        locations: "pysr/test/test_main.py:1166",
        arity: 2,
        reference: |a| a[0],
        near_kink: no_kink,
    },
    LossCase {
        id: "L16",
        src: "g(x, y) = 0.0",
        locations: "pysr/test/test_main.py:2488",
        arity: 2,
        reference: |_| 0.0,
        near_kink: no_kink,
    },
    LossCase {
        id: "L17",
        src: "(prediction, target) -> 0.0",
        locations: "pysr/test/test_type_specs.py:241, :315, :1194, :1574",
        arity: 2,
        reference: |_| 0.0,
        near_kink: no_kink,
    },
    LossCase {
        id: "L18",
        src: "(p, t) -> 0.0",
        locations: "pysr/test/test_type_specs.py:1471",
        arity: 2,
        reference: |_| 0.0,
        near_kink: no_kink,
    },
];

struct EarlyStopCase {
    id: &'static str,
    src: &'static str,
    locations: &'static str,
    reference: fn(f64, f64) -> bool,
}

/// Early-stop conditions, taking `(loss, complexity)`.
const EARLY_STOPS: &[EarlyStopCase] = &[
    EarlyStopCase {
        id: "E1",
        src: "stop_if(loss, complexity) = loss < 1e-6 && complexity < 10",
        locations: "README.md:268; skills/pysr/SKILL.md:74",
        reference: |l, c| l < 1e-6 && c < 10.0,
    },
    EarlyStopCase {
        id: "E2",
        src: "stop_if(loss, complexity) = loss < 1e-4 && complexity <= 5",
        locations: "pysr/test/test_autodiff.py:32; pysr/test/test_main.py:1454",
        reference: |l, c| l < 1e-4 && c <= 5.0,
    },
    EarlyStopCase {
        id: "E3",
        src: "stop_if(loss, complexity) = loss < 1e-4 && complexity == 1",
        locations: "pysr/test/test_main.py:99, :109, :119, :1081",
        reference: |l, c| l < 1e-4 && c == 1.0,
    },
    EarlyStopCase {
        id: "E4",
        src: "stop_if(loss, complexity) = loss < 1e-10 && complexity == 1",
        locations: "pysr/test/test_main.py:147",
        reference: |l, c| l < 1e-10 && c == 1.0,
    },
    EarlyStopCase {
        id: "E5",
        src: "stop_if(loss, complexity) = loss < 1e-4 && complexity == 3",
        locations: "pysr/test/test_main.py:182, :461, :599",
        reference: |l, c| l < 1e-4 && c == 3.0,
    },
    EarlyStopCase {
        id: "E6",
        src: "stop_if_1(l, c) = l < 1e-8 && c <= 3",
        locations: "pysr/test/test_main.py:500 (f-string, case 1)",
        reference: |l, c| l < 1e-8 && c <= 3.0,
    },
    EarlyStopCase {
        id: "E7",
        src: "stop_if_2(l, c) = l < 1e-8 && c <= 2",
        locations: "pysr/test/test_main.py:500 (f-string, case 2)",
        reference: |l, c| l < 1e-8 && c <= 2.0,
    },
    EarlyStopCase {
        id: "E8",
        src: "stop_if(loss, complexity) = loss < 1e-4 && complexity == 2",
        locations: "pysr/test/test_main.py:554",
        reference: |l, c| l < 1e-4 && c == 2.0,
    },
    EarlyStopCase {
        id: "E9",
        src: "(loss, complexity) -> loss <= 1e-4 && complexity <= 6",
        locations: "pysr/test/test_main.py:584",
        reference: |l, c| l <= 1e-4 && c <= 6.0,
    },
    EarlyStopCase {
        id: "E10",
        src: "stop_if(loss, complexity) = loss < 0.05 && complexity == 2",
        locations: "pysr/test/test_main.py:654, :808",
        reference: |l, c| l < 0.05 && c == 2.0,
    },
    EarlyStopCase {
        id: "E11",
        src: "stop_if(loss, complexity) = loss < 1e-3 && complexity == 7",
        locations: "pysr/test/test_main.py:699",
        reference: |l, c| l < 1e-3 && c == 7.0,
    },
    EarlyStopCase {
        id: "E12",
        src: "stop_if(loss, complexity) = loss < 1e-10 && complexity == 6",
        locations: "pysr/test/test_main.py:863",
        reference: |l, c| l < 1e-10 && c == 6.0,
    },
    EarlyStopCase {
        id: "E13",
        src: "stop_if(loss, complexity) = loss < 1e-10 && complexity <= 3",
        locations: "pysr/test/test_main.py:1037",
        reference: |l, c| l < 1e-10 && c <= 3.0,
    },
    EarlyStopCase {
        id: "E14",
        src: "stop_if_under_n1(loss, complexity) = loss < -1.0",
        locations: "pysr/test/test_main.py:1168",
        reference: |l, _| l < -1.0,
    },
    EarlyStopCase {
        id: "E15",
        src: "stop_if(loss, complexity) = loss < 1e-4 && complexity <= 7",
        locations: "pysr/test/test_main.py:1180",
        reference: |l, c| l < 1e-4 && c <= 7.0,
    },
    EarlyStopCase {
        id: "E16",
        src: "stop_if(loss, complexity) = loss < 1e-4 && complexity <= 10",
        locations: "pysr/test/test_main.py:1200",
        reference: |l, c| l < 1e-4 && c <= 10.0,
    },
    EarlyStopCase {
        id: "E17",
        src: "stop_if(loss, complexity) = loss < 1e-6 && complexity <= 5",
        locations: "pysr/test/test_main.py:1264, :1276",
        reference: |l, c| l < 1e-6 && c <= 5.0,
    },
    EarlyStopCase {
        id: "E18",
        src: "(l, c) -> l < 1e-6 && c == 3",
        locations: "pysr/test/test_main.py:2852",
        reference: |l, c| l < 1e-6 && c == 3.0,
    },
    EarlyStopCase {
        id: "E19",
        src: "(l, c) -> l < 1e-6 && c == 1",
        locations: "pysr/test/test_main.py:2902",
        reference: |l, c| l < 1e-6 && c == 1.0,
    },
    EarlyStopCase {
        id: "E20",
        src: "checkpoint_early_stop(loss, complexity) = false",
        locations: "pysr/test/test_type_specs.py:624",
        reference: |_, _| false,
    },
    EarlyStopCase {
        id: "E21",
        src: "(loss, complexity) -> loss < 1.0e-8 && complexity == 5",
        locations: "pysr/test/test_type_specs.py:1140",
        reference: |l, c| l < 1.0e-8 && c == 5.0,
    },
    EarlyStopCase {
        id: "E22",
        src: "(loss, complexity) -> false",
        locations: "pysr/test/test_type_specs.py:391, :1413",
        reference: |_, _| false,
    },
];

fn compile_fn(id: &str, src: &str, locations: &str) -> ScalarFn<f64> {
    parse_definition(src)
        .and_then(|def| ScalarFn::compile(&def))
        .unwrap_or_else(|e| panic!("{id} ({locations}): {e}"))
}

/// `N_ROWS` rows per argument from a fixed seed: predictions and targets uniform in
/// [-3, 3), weights uniform in [0.1, 2).
fn loss_columns(arity: usize) -> Vec<Vec<f64>> {
    let mut rng = fastrand::Rng::with_seed(0x1055);
    (0..arity)
        .map(|k| {
            let (lo, hi) = if k == 2 { (0.1, 2.0) } else { (-3.0, 3.0) };
            (0..N_ROWS).map(|_| lo + (hi - lo) * rng.f64()).collect()
        })
        .collect()
}

fn slices(cols: &[Vec<f64>]) -> Vec<SrcRef<'_, f64>> {
    cols.iter().map(|c| SrcRef::Slice(c)).collect()
}

#[test]
fn corpus_losses_match_hand_written_rust() {
    for case in LOSSES {
        let f = compile_fn(case.id, case.src, case.locations);
        assert_eq!(usize::from(f.arity()), case.arity, "{}", case.id);
        assert!(!f.returns_bool(), "{}", case.id);
        let cols = loss_columns(case.arity);
        let mut got = vec![0.0; N_ROWS];
        f.eval(&mut got, &slices(&cols), &NO_CHECK);
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
fn corpus_loss_derivatives_match_central_differences() {
    for case in LOSSES {
        let f = compile_fn(case.id, case.src, case.locations);
        let cols = loss_columns(case.arity);
        let mut got = vec![0.0; N_ROWS];
        f.partial(0, &mut got, &slices(&cols));
        for (r, &g) in got.iter().enumerate() {
            let x = row(&cols, r);
            if (case.near_kink)(&x) {
                continue;
            }
            let h = 1e-6 * x[0].abs().max(1.0);
            let (mut hi, mut lo) = (x.clone(), x.clone());
            hi[0] += h;
            lo[0] -= h;
            let fd = ((case.reference)(&hi) - (case.reference)(&lo)) / (2.0 * h);
            assert!(
                (g - fd).abs() <= 1e-6 * fd.abs().max(1.0),
                "{} `{}` d/dprediction at {x:?}: got {g}, finite difference {fd}",
                case.id,
                case.src
            );
        }
    }
}

#[test]
fn corpus_early_stops_match_hand_written_rust() {
    let losses = [
        -2.0, -1.0, -0.5, 0.0, 1e-12, 1e-11, 1e-10, 2e-10, 1e-9, 1e-8, 2e-8, 1e-7, 1e-6, 2e-6, 1e-5, 1e-4, 2e-4, 1e-3,
        2e-3, 0.01, 0.05, 0.06, 0.5, 1.0, 2.0,
    ];
    let (loss_col, complexity_col): (Vec<f64>, Vec<f64>) = losses
        .iter()
        .flat_map(|&l| (0..=12).map(move |c| (l, f64::from(c))))
        .unzip();
    let cols = [loss_col, complexity_col];
    for case in EARLY_STOPS {
        let f = compile_fn(case.id, case.src, case.locations);
        assert_eq!(f.arity(), 2, "{}", case.id);
        assert!(f.returns_bool(), "{}", case.id);
        let mut got = vec![0.0; cols[0].len()];
        f.eval(&mut got, &slices(&cols), &NO_CHECK);
        for (r, &g) in got.iter().enumerate() {
            let (l, c) = (cols[0][r], cols[1][r]);
            let want = (case.reference)(l, c);
            assert_eq!(g, f64::from(u8::from(want)), "{} `{}` at ({l}, {c})", case.id, case.src);
        }
    }
}

/// Looks up a feature of the current row by name.
type Feature<'a> = &'a dyn Fn(&str) -> f64;

struct TemplateCase {
    id: &'static str,
    src: &'static str,
    expressions: &'static [&'static str],
    variables: &'static [&'static str],
    parameters: &'static [(&'static str, usize)],
    /// Variables holding 1-based category indices, with the number of categories.
    categories: &'static [(&'static str, usize)],
    /// The whole formula, from a feature lookup and the concatenated parameters.
    reference: fn(Feature<'_>, &[f64]) -> f64,
}

/// Subexpression `k` with `arity` arguments: `cos(#1 * w) + Σ_{i≥2} #i^2 * 0.5 / i + c`.
fn sub_tree(k: usize, arity: usize) -> PostfixExpr<f64, BuiltinOpsF64, 3> {
    let var = |f: u16| PostfixExpr::new(vec![PNode::Var { feature: f }], vec![], Default::default());
    let offset = 0.3 + 0.2 * k as f64;
    let mut tree = PostfixExpr::new(vec![PNode::Const { idx: 0 }], vec![offset], Default::default());
    if arity >= 1 {
        tree = cos(var(0) * (0.7 + 0.1 * k as f64)) + tree;
    }
    for i in 2..=arity {
        tree = tree + var(i as u16 - 1) * var(i as u16 - 1) * (0.5 / i as f64);
    }
    tree
}

fn sub(k: usize, args: &[f64]) -> f64 {
    let mut v = 0.3 + 0.2 * k as f64;
    if let Some(a) = args.first() {
        v += (a * (0.7 + 0.1 * k as f64)).cos();
    }
    for (i, a) in args.iter().enumerate().skip(1) {
        v += a * a * (0.5 / (i + 1) as f64);
    }
    v
}

/// Template combine formulas from PySR's docs and tests.
const TEMPLATES: &[TemplateCase] = &[
    TemplateCase {
        id: "T1",
        src: "sin(f(x1, x2)) + g(x3)",
        expressions: &["f", "g"],
        variables: &["x1", "x2", "x3"],
        parameters: &[],
        categories: &[],
        reference: |v, _| sub(0, &[v("x1"), v("x2")]).sin() + sub(1, &[v("x3")]),
    },
    TemplateCase {
        id: "T2",
        src: "f(x1, x2, p1[category], p2[category])",
        expressions: &["f"],
        variables: &["x1", "x2", "category"],
        parameters: &[("p1", 3), ("p2", 3)],
        categories: &[("category", 3)],
        reference: |v, p| {
            let c = v("category") as usize - 1;
            sub(0, &[v("x1"), v("x2"), p[c], p[3 + c]])
        },
    },
    TemplateCase {
        id: "T3",
        src: "\n        v = shared(x1, x2, x3)\n        y1_predicted = v + f1(x1, x2, x3)\n        y2_predicted = v + f2(x1, x2, x3)\n        y3_predicted = v + f3(x1, x2, x3)\n\n        residuals = (\n            abs2(y1 - y1_predicted) +\n            abs2(y2 - y2_predicted) +\n            abs2(y3 - y3_predicted)\n        )\n\n        residuals\n    ",
        expressions: &["shared", "f1", "f2", "f3"],
        variables: &["x1", "x2", "x3", "y1", "y2", "y3"],
        parameters: &[],
        categories: &[],
        reference: |v, _| {
            let x = [v("x1"), v("x2"), v("x3")];
            let s = sub(0, &x);
            (v("y1") - (s + sub(1, &x))).powi(2)
                + (v("y2") - (s + sub(2, &x))).powi(2)
                + (v("y3") - (s + sub(3, &x))).powi(2)
        },
    },
    TemplateCase {
        id: "T4",
        src: "df = D(f, 1); df(x)",
        expressions: &["f"],
        variables: &["x"],
        parameters: &[],
        categories: &[],
        reference: |v, _| -(0.7 * v("x")).sin() * 0.7,
    },
    TemplateCase {
        id: "T5",
        src: "sin(f(x1, x2)) + g(x3)^2",
        expressions: &["f", "g"],
        variables: &["x1", "x2", "x3"],
        parameters: &[],
        categories: &[],
        reference: |v, _| sub(0, &[v("x1"), v("x2")]).sin() + sub(1, &[v("x3")]).powi(2),
    },
    TemplateCase {
        id: "T6",
        src: "sin(f(x, y))",
        expressions: &["f"],
        variables: &["x", "y"],
        parameters: &[],
        categories: &[],
        reference: |v, _| sub(0, &[v("x"), v("y")]).sin(),
    },
    TemplateCase {
        id: "T7",
        src: "p[class] * x1^2 + f(x2)",
        expressions: &["f"],
        variables: &["x1", "x2", "class"],
        parameters: &[("p", 3)],
        categories: &[("class", 3)],
        reference: |v, p| p[v("class") as usize - 1] * v("x1").powi(2) + sub(0, &[v("x2")]),
    },
    TemplateCase {
        id: "T8",
        src: "mu = f_mu(x1, x2, x3); logvar = f_logvar(x1, x2, x3); 0.5f0 * (logvar + (mu - y)^2 / exp(logvar)) - 2",
        expressions: &["f_mu", "f_logvar"],
        variables: &["x1", "x2", "x3", "y"],
        parameters: &[],
        categories: &[],
        reference: |v, _| {
            let x = [v("x1"), v("x2"), v("x3")];
            let (mu, logvar) = (sub(0, &x), sub(1, &x));
            0.5 * (logvar + (mu - v("y")).powi(2) / logvar.exp()) - 2.0
        },
    },
    TemplateCase {
        id: "T9",
        src: "f(x0, x1)",
        expressions: &["f"],
        variables: &["x0", "x1"],
        parameters: &[],
        categories: &[],
        reference: |v, _| sub(0, &[v("x0"), v("x1")]),
    },
    TemplateCase {
        id: "T10",
        src: "p[1] * f(x) + p[2] * x + p[3]",
        expressions: &["f"],
        variables: &["x"],
        parameters: &[("p", 3)],
        categories: &[],
        reference: |v, p| p[0] * sub(0, &[v("x")]) + p[1] * v("x") + p[2],
    },
    TemplateCase {
        id: "T11",
        src: "p[1] * f(x) + p[2]",
        expressions: &["f"],
        variables: &["x"],
        parameters: &[("p", 2)],
        categories: &[],
        reference: |v, p| p[0] * sub(0, &[v("x")]) + p[1],
    },
    TemplateCase {
        id: "T12",
        src: "f(x)",
        expressions: &["f"],
        variables: &["x"],
        parameters: &[],
        categories: &[],
        reference: |v, _| sub(0, &[v("x")]),
    },
    TemplateCase {
        id: "T13",
        src: "f(x, y) + g(z)",
        expressions: &["f", "g"],
        variables: &["x", "y", "z"],
        parameters: &[],
        categories: &[],
        reference: |v, _| sub(0, &[v("x"), v("y")]) + sub(1, &[v("z")]),
    },
    TemplateCase {
        id: "T14",
        src: "p[1] * f(x)",
        expressions: &["f"],
        variables: &["x"],
        parameters: &[("p", 1)],
        categories: &[],
        reference: |v, p| p[0] * sub(0, &[v("x")]),
    },
    TemplateCase {
        id: "T15",
        src: "p1[1]*f(x,y) + p2[1]*g(z)",
        expressions: &["f", "g"],
        variables: &["x", "y", "z"],
        parameters: &[("p1", 1), ("p2", 1)],
        categories: &[],
        reference: |v, p| p[0] * sub(0, &[v("x"), v("y")]) + p[1] * sub(1, &[v("z")]),
    },
    TemplateCase {
        id: "T16",
        src: "f(var1) * g(var2)",
        expressions: &["f", "g"],
        variables: &["var1", "var2"],
        parameters: &[],
        categories: &[],
        reference: |v, _| sub(0, &[v("var1")]) * sub(1, &[v("var2")]),
    },
    TemplateCase {
        id: "T20",
        src: "f0(x) + f1(x) + f2(x) + f3(x) + f4(x)",
        expressions: &["f0", "f1", "f2", "f3", "f4"],
        variables: &["x"],
        parameters: &[],
        categories: &[],
        reference: |v, _| (0..5).map(|k| sub(k, &[v("x")])).sum(),
    },
];

fn compile_template(case: &TemplateCase) -> TemplateStructure {
    TemplateStructure::new(
        case.src,
        case.expressions,
        None,
        case.variables,
        case.parameters,
        &BuiltinOpsF64::new(),
    )
    .unwrap_or_else(|e| panic!("{} `{}`: {e}", case.id, case.src))
}

/// Random features in (-2, 2), with category columns drawn from `1..=n`.
fn template_inputs(case: &TemplateCase, seed: u64) -> Array2<f64> {
    let mut rng = fastrand::Rng::with_seed(seed);
    Array2::from_shape_fn((case.variables.len(), N_ROWS), |(f, _)| {
        match case.categories.iter().find(|(name, _)| *name == case.variables[f]) {
            Some((_, n)) => (rng.usize(0..*n) + 1) as f64,
            None => rng.f64() * 4.0 - 2.0,
        }
    })
}

#[test]
fn corpus_templates_match_hand_written_rust() {
    let ops = BuiltinOpsF64::new();
    for case in TEMPLATES {
        let template = compile_template(case);
        let trees: Vec<_> = template
            .num_features()
            .iter()
            .enumerate()
            .map(|(k, &arity)| sub_tree(k, arity))
            .collect();
        let params: Vec<f64> = (0..template.n_parameters()).map(|i| 0.5 + 0.25 * i as f64).collect();
        let x = template_inputs(case, 7);
        let (got, ok) = template.eval(&trees, &params, &ops, x.view()).unwrap();
        assert!(ok, "{} `{}` is invalid", case.id, case.src);
        for (r, &g) in got.iter().enumerate() {
            let value = |name: &str| {
                let f = case.variables.iter().position(|v| *v == name).expect("a variable");
                x[[f, r]]
            };
            let want = (case.reference)(&value, &params);
            assert!(
                (g - want).abs() <= 1e-12 * want.abs().max(1.0),
                "{} `{}` row {r}: got {g}, want {want}",
                case.id,
                case.src
            );
            assert!(
                (g - want).abs() <= 1e-12 * want.abs(),
                "{} `{}` row {r}: got {g}, want {want}, relative error {}",
                case.id,
                case.src,
                (g - want).abs() / want.abs()
            );
        }
    }
}

#[test]
fn corpus_template_derivative_matches_central_differences() {
    let case = TEMPLATES.iter().find(|c| c.id == "T4").expect("T4");
    let template = compile_template(case);
    let trees = vec![sub_tree(0, 1)];
    let x = template_inputs(case, 11);
    let (got, ok) = template.eval(&trees, &[], &BuiltinOpsF64::new(), x.view()).unwrap();
    assert!(ok);
    let h = 1e-5;
    for (r, &g) in got.iter().enumerate() {
        let x0 = x[[0, r]];
        let fd = (sub(0, &[x0 + h]) - sub(0, &[x0 - h])) / (2.0 * h);
        assert!(
            (g - fd).abs() <= 1e-8 * fd.abs().max(1.0),
            "row {r}: D(f, 1) = {g}, central difference {fd}"
        );
    }
}

#[test]
fn corpus_templates_for_custom_value_types_fail_at_setup_naming_the_function() {
    let cases = [
        (
            "T18",
            "add_vectors(f(x1), g(x2))",
            vec!["f", "g"],
            vec!["x1", "x2"],
            vec![],
            "`add_vectors`",
        ),
        (
            "T19",
            "choose_parameter(p[1], f(x))",
            vec!["f"],
            vec!["x"],
            vec![("p", 1)],
            "`choose_parameter`",
        ),
        (
            "T17",
            "alpha*f(x) + beta*g(y)",
            vec!["f", "g"],
            vec!["x", "y"],
            vec![("alpha", 1), ("beta", 2)],
            "`alpha` is a parameter vector",
        ),
    ];
    for (id, src, expressions, variables, parameters, name) in cases {
        let err = TemplateStructure::new(src, &expressions, None, &variables, &parameters, &BuiltinOpsF64::new())
            .expect_err(id);
        assert!(err.message.contains(name), "{id} `{src}`: {}", err.message);
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
