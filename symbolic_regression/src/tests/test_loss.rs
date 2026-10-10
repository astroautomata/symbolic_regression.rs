//! Every built-in loss against an independent formula from LossFunctions.jl v1.2.2 (the version
//! SymbolicRegression.jl resolves), cited as `src/<file>:<line>`. Distance losses take
//! `r = prediction - target` (src/losses.jl:6); margin losses take `a = target * prediction`
//! (src/losses.jl:9).

use crate::loss_functions::{BuiltinLoss, LossObject, parse_loss};

struct Case {
    src: &'static str,
    cite: &'static str,
    margin: bool,
    /// Loss of `r` (distance) or `a` (margin).
    reference: fn(f64) -> f64,
    /// Values of `r` or `a` where the loss has a kink.
    kinks: &'static [f64],
}

const CASES: &[Case] = &[
    Case {
        src: "LPDistLoss{3}()",
        cite: "losses/distance.jl:20",
        margin: false,
        reference: |r| r.abs().powi(3),
        kinks: &[],
    },
    Case {
        src: "LPDistLoss(2.5)",
        cite: "losses/distance.jl:20",
        margin: false,
        reference: |r| r.abs().powf(2.5),
        kinks: &[],
    },
    Case {
        src: "L1DistLoss()",
        cite: "losses/distance.jl:80",
        margin: false,
        reference: f64::abs,
        kinks: &[0.0],
    },
    Case {
        src: "L2DistLoss()",
        cite: "losses/distance.jl:125",
        margin: false,
        reference: |r| r.powi(2),
        kinks: &[],
    },
    Case {
        src: "PeriodicLoss(2.0)",
        cite: "losses/distance.jl:153, :159",
        margin: false,
        reference: |r| 1.0 - (r * (2.0 * std::f64::consts::PI / 2.0)).cos(),
        kinks: &[],
    },
    Case {
        src: "HuberLoss(0.7)",
        cite: "losses/distance.jl:213",
        margin: false,
        reference: |r| {
            if r.abs() <= 0.7 {
                0.5 * r * r
            } else {
                0.7 * r.abs() - 0.5 * 0.7 * 0.7
            }
        },
        kinks: &[-0.7, 0.7],
    },
    Case {
        src: "HuberLoss()",
        cite: "losses/distance.jl:210, :213",
        margin: false,
        reference: |r| if r.abs() <= 1.0 { 0.5 * r * r } else { r.abs() - 0.5 },
        kinks: &[-1.0, 1.0],
    },
    Case {
        src: "L1EpsilonInsLoss(0.3)",
        cite: "losses/distance.jl:288",
        margin: false,
        reference: |r| (r.abs() - 0.3).max(0.0),
        kinks: &[-0.3, 0.3],
    },
    Case {
        src: "EpsilonInsLoss(0.3)",
        cite: "losses/distance.jl:284, :288",
        margin: false,
        reference: |r| (r.abs() - 0.3).max(0.0),
        kinks: &[-0.3, 0.3],
    },
    Case {
        src: "L2EpsilonInsLoss(0.3)",
        cite: "losses/distance.jl:350",
        margin: false,
        reference: |r| (r.abs() - 0.3).max(0.0).powi(2),
        kinks: &[-0.3, 0.3],
    },
    Case {
        src: "LogitDistLoss()",
        cite: "losses/distance.jl:405",
        margin: false,
        reference: |r| -(4.0_f64).ln() - r + 2.0 * (1.0 + r.exp()).ln(),
        kinks: &[],
    },
    Case {
        src: "QuantileLoss(0.7)",
        cite: "losses/distance.jl:463",
        margin: false,
        reference: |r| r * (f64::from(u8::from(r > 0.0)) - 0.7),
        kinks: &[0.0],
    },
    Case {
        src: "LogCoshLoss()",
        cite: "losses/distance.jl:514-515, :517",
        margin: false,
        reference: |r| {
            let x = -2.0 * r;
            let softplus = if x > 0.0 {
                x + (-x).exp().ln_1p()
            } else {
                x.exp().ln_1p()
            };
            r + softplus - 2.0_f64.ln()
        },
        kinks: &[],
    },
    Case {
        src: "ZeroOneLoss()",
        cite: "losses/margin.jl:41",
        margin: true,
        reference: |a| if a < 0.0 { 1.0 } else { 0.0 },
        kinks: &[0.0],
    },
    Case {
        src: "PerceptronLoss()",
        cite: "losses/margin.jl:88",
        margin: true,
        reference: |a| (-a).max(0.0),
        kinks: &[0.0],
    },
    Case {
        src: "LogitMarginLoss()",
        cite: "losses/margin.jl:132",
        margin: true,
        reference: |a| (-a).exp().ln_1p(),
        kinks: &[],
    },
    Case {
        src: "L1HingeLoss()",
        cite: "losses/margin.jl:180",
        margin: true,
        reference: |a| (1.0 - a).max(0.0),
        kinks: &[1.0],
    },
    Case {
        src: "HingeLoss()",
        cite: "losses/margin.jl:178, :180",
        margin: true,
        reference: |a| (1.0 - a).max(0.0),
        kinks: &[1.0],
    },
    Case {
        src: "L2HingeLoss()",
        cite: "losses/margin.jl:228",
        margin: true,
        reference: |a| (1.0 - a).max(0.0).powi(2),
        kinks: &[1.0],
    },
    Case {
        src: "SmoothedL1HingeLoss(0.5)",
        cite: "losses/margin.jl:285",
        margin: true,
        reference: |a| {
            let gamma = 0.5;
            if a >= 1.0 - gamma {
                0.5 / gamma * (1.0 - a).max(0.0).powi(2)
            } else {
                1.0 - gamma / 2.0 - a
            }
        },
        kinks: &[0.5, 1.0],
    },
    Case {
        src: "ModifiedHuberLoss()",
        cite: "losses/margin.jl:346",
        margin: true,
        reference: |a| {
            if a >= -1.0 {
                (1.0 - a).max(0.0).powi(2)
            } else {
                -4.0 * a
            }
        },
        kinks: &[-1.0, 1.0],
    },
    Case {
        src: "L2MarginLoss()",
        cite: "losses/margin.jl:403",
        margin: true,
        reference: |a| (1.0 - a).powi(2),
        kinks: &[],
    },
    Case {
        src: "ExpLoss()",
        cite: "losses/margin.jl:452",
        margin: true,
        reference: |a| (-a).exp(),
        kinks: &[],
    },
    Case {
        src: "SigmoidLoss()",
        cite: "losses/margin.jl:500",
        margin: true,
        reference: |a| 1.0 - a.tanh(),
        kinks: &[],
    },
    Case {
        src: "DWDMarginLoss(1.5)",
        cite: "losses/margin.jl:558",
        margin: true,
        reference: |a| {
            let q: f64 = 1.5;
            if a <= q / (q + 1.0) {
                1.0 - a
            } else {
                (q.powf(q) / (q + 1.0).powf(q + 1.0)) / a.powf(q)
            }
        },
        kinks: &[0.6],
    },
];

fn close(a: f64, b: f64, rtol: f64) -> bool {
    a == b || (a - b).abs() <= rtol * a.abs().max(b.abs())
}

/// Prediction-target pairs: targets are uniform for distance losses and ±1 for margin losses, so
/// agreements take both signs.
fn points(margin: bool) -> Vec<(f64, f64)> {
    let mut rng = fastrand::Rng::with_seed(7);
    (0..500)
        .map(|_| {
            let prediction = rng.f64() * 6.0 - 3.0;
            let target = if margin {
                if rng.bool() { 1.0 } else { -1.0 }
            } else {
                rng.f64() * 6.0 - 3.0
            };
            (prediction, target)
        })
        .collect()
}

fn reference(case: &Case, prediction: f64, target: f64) -> f64 {
    let arg = if case.margin {
        target * prediction
    } else {
        prediction - target
    };
    (case.reference)(arg)
}

fn parse(case: &Case) -> LossObject<f64> {
    parse_loss::<f64>(case.src).unwrap_or_else(|e| panic!("{}: {e}", case.src))
}

#[test]
fn builtin_losses_match_lossfunctions_definitions() {
    for case in CASES {
        let loss = parse(case);
        let pts = points(case.margin);
        if case.margin {
            assert!(pts.iter().any(|&(p, t)| p * t < 0.0) && pts.iter().any(|&(p, t)| p * t > 1.0));
        }
        for &(p, t) in &pts {
            let got = loss.loss(&[p], &[t], None);
            let want = reference(case, p, t);
            assert!(
                close(got, want, 1e-12),
                "{} ({}) at ({p}, {t}): got {got}, want {want}",
                case.src,
                case.cite
            );
        }
    }
}

#[test]
fn builtin_loss_derivatives_match_central_differences() {
    for case in CASES {
        let loss = parse(case);
        for (p, t) in points(case.margin) {
            let arg = if case.margin { t * p } else { p - t };
            if case.kinks.iter().any(|k| (arg - k).abs() < 1e-3) {
                continue;
            }
            let h = 1e-6;
            let fd = (reference(case, p + h, t) - reference(case, p - h, t)) / (2.0 * h);
            let mut got = [0.0];
            loss.dloss_dyhat(&[p], &[t], None, &mut got);
            assert!(
                (got[0] - fd).abs() <= 1e-6 * fd.abs().max(1.0),
                "{} d/dprediction at ({p}, {t}): got {}, finite difference {fd}",
                case.src,
                got[0]
            );
        }
    }
}

#[test]
fn builtin_loss_derivatives_at_kinks_follow_lossfunctions() {
    let deriv = |src: &str, p: f64, t: f64| {
        let mut out = [0.0];
        parse_loss::<f64>(src).unwrap().dloss_dyhat(&[p], &[t], None, &mut out);
        out[0]
    };
    // losses/distance.jl:81, :21, :222, :292, :467; losses/margin.jl:42, :89, :181, :292, :349, :567
    assert_eq!(deriv("L1DistLoss()", 1.0, 1.0), 0.0);
    assert_eq!(deriv("LPDistLoss(1.5)", 1.0, 1.0), 0.0);
    assert_eq!(deriv("HuberLoss(0.5)", 1.5, 1.0), 0.5);
    assert_eq!(deriv("L1EpsilonInsLoss(0.5)", 1.5, 1.0), 0.0);
    assert_eq!(deriv("QuantileLoss(0.25)", 1.0, 1.0), -0.25);
    assert_eq!(deriv("PerceptronLoss()", 0.0, 1.0), 0.0);
    assert_eq!(deriv("L1HingeLoss()", 1.0, 1.0), 0.0);
    assert_eq!(deriv("SmoothedL1HingeLoss(0.5)", -1.0, -1.0), 0.0);
    assert_eq!(deriv("ModifiedHuberLoss()", -1.0, 1.0), -4.0);
    assert_eq!(deriv("ModifiedHuberLoss()", 1.0, -1.0), 4.0);
    assert_eq!(deriv("DWDMarginLoss(1.5)", 0.6, 1.0), -1.0);
    assert_eq!(deriv("ZeroOneLoss()", -2.0, 1.0), 0.0);
}

#[test]
fn builtin_losses_average_or_weight_like_lossfunctions() {
    let yhat = [1.0_f64, 0.0, 4.0];
    let y = [0.0_f64, 2.0, 3.0];
    let w = [1.0_f64, 3.0, 1.0];
    let l2 = BuiltinLoss::L2Dist;
    assert_eq!(crate::LossFn::<f64>::loss(&l2, &yhat, &y, None), 2.0);
    assert_eq!(
        crate::LossFn::<f64>::loss(&l2, &yhat, &y, Some(&w)),
        (1.0 + 12.0 + 1.0) / 5.0
    );

    let mut out = [0.0; 3];
    crate::LossFn::<f64>::dloss_dyhat(&l2, &yhat, &y, Some(&w), &mut out);
    for (got, want) in out.iter().zip([2.0 / 5.0, -12.0 / 5.0, 2.0 / 5.0]) {
        assert!(close(*got, want, 1e-15), "{out:?}");
    }
}

#[test]
fn builtin_constructors_accept_julia_forms() {
    let at = |src: &str| parse_loss::<f64>(src).unwrap().loss(&[2.5], &[0.5], None);
    assert_eq!(at("LPDistLoss{3}()"), 8.0);
    assert_eq!(at("LPDistLoss(3)"), 8.0);
    assert_eq!(at(" LPDistLoss{3.0}( ) "), 8.0);
    assert_eq!(at("HuberLoss(1)"), 1.5);
}

#[test]
fn builtin_constructor_errors_name_the_loss() {
    let cases = [
        ("PoissonLoss()", "unknown loss `PoissonLoss`"),
        (
            "HuberLoss(-1.0)",
            "`HuberLoss`: the crossover parameter must be strictly positive",
        ),
        ("QuantileLoss()", "`QuantileLoss` is constructed as `QuantileLoss(τ)`"),
        ("L2DistLoss(1.0)", "`L2DistLoss` is constructed as `L2DistLoss()`"),
        ("HuberLoss(d)", "`HuberLoss` parameters must be numbers, got `d`"),
    ];
    for (src, message) in cases {
        let err = parse_loss::<f64>(src)
            .err()
            .unwrap_or_else(|| panic!("`{src}` was accepted"));
        assert!(err.message.contains(message), "`{src}`: {}", err.message);
    }
}

#[test]
fn dsl_losses_weight_like_symbolicregression() {
    let yhat = [1.0_f64, 0.0, 4.0];
    let y = [0.0_f64, 2.0, 3.0];
    let w = [1.0_f64, 3.0, 1.0];

    let unweighted = parse_loss::<f64>("myloss(x, y) = abs(x - y)^1.5").unwrap();
    let want: f64 = [1.0_f64, 2.0, 1.0].iter().map(|r| r.powf(1.5)).sum::<f64>() / 3.0;
    assert!(close(unweighted.loss(&yhat, &y, None), want, 1e-15));
    assert!(unweighted.check_weights(false).is_ok());
    let err = unweighted.check_weights(true).unwrap_err();
    assert!(
        err.contains("(prediction, target, weight)") && err.contains("weights"),
        "{err}"
    );

    let weighted = parse_loss::<f64>("myloss(x, y, w) = w * abs(x - y)").unwrap();
    assert_eq!(weighted.loss(&yhat, &y, Some(&w)), (1.0 + 6.0 + 1.0) / 5.0);
    let mut out = [0.0; 3];
    weighted.dloss_dyhat(&yhat, &y, Some(&w), &mut out);
    for (got, want) in out.iter().zip([1.0 / 5.0, -3.0 / 5.0, 1.0 / 5.0]) {
        assert!(close(*got, want, 1e-15), "{out:?}");
    }
    assert!(weighted.check_weights(true).is_ok());
    assert!(weighted.check_weights(false).is_err());
}

#[test]
fn dsl_losses_sum_across_chunks() {
    let n = 1000;
    let yhat: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.5).cos()).collect();
    let loss = parse_loss::<f64>("(x, y) -> (x - y)^2").unwrap();
    let want = crate::LossFn::<f64>::loss(&BuiltinLoss::L2Dist, &yhat, &y, None);
    assert!(close(loss.loss(&yhat, &y, None), want, 1e-13));
}

#[test]
fn dsl_loss_errors_name_the_construct() {
    let cases = [
        (
            "myloss_bad_arity(a) = a",
            "takes (prediction, target) or (prediction, target, weight), but this one takes 1 argument",
        ),
        ("f(x, y) = x > y", "must return a number"),
        ("myloss(x, y) = erf(x - y)", "unknown function `erf`"),
        ("myloss(x, y) = SpecialFunctions.erf(x - y)", "`SpecialFunctions.erf`"),
        (
            "function my_custom_loss(tree, dataset, options)\n    out, completed = eval_tree_array(tree, dataset.X, options)\nend",
            "destructuring assignment (`a, b = ...`) is not supported",
        ),
        (
            "function my_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}\n    return zero(L)\nend",
            "type `Dataset` is not supported",
        ),
    ];
    for (src, message) in cases {
        let err = parse_loss::<f64>(src)
            .err()
            .unwrap_or_else(|| panic!("`{src}` was accepted"));
        assert!(err.message.contains(message), "`{src}`: {}", err.message);
    }
}
