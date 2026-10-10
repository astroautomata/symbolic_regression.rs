//! Ports of SymbolicRegression.jl's testitems for elementwise losses, `loss_scale`, and early
//! stopping: test/unit/misc/test_losses.jl, test/unit/misc/test_loss_scale.jl, and
//! test/unit/evaluation/test_early_stop.jl.

use dynamic_expressions::expression::PostfixExpr;
use dynamic_expressions::node::PNode;
use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
use ndarray::{Array1, Array2};

use super::common::{D, T, TestOps};
use crate::hall_of_fame::{HallOfFame, choose_best, compute_direct_score, compute_zero_centered_score, pareto_volume};
use crate::options::LossScale;
use crate::pop_member::PopMember;
use crate::search_utils::SearchEngine;
use crate::{Dataset, Options, parse_early_stop, parse_loss};

#[test]
fn custom_losses_match_their_definitions() {
    let mut rng = fastrand::Rng::with_seed(0);
    let x: Vec<f32> = (0..100).map(|_| rng.f32() * 4.0 - 2.0).collect();
    let y: Vec<f32> = (0..100).map(|_| rng.f32() * 4.0 - 2.0).collect();
    let w: Vec<f32> = (0..100).map(|_| rng.f32() * 2.0).collect();

    fn testl1(a: f32, b: f32) -> f32 {
        (a - b).abs()
    }
    fn customloss(a: f32, b: f32) -> f32 {
        (a - b).abs().powf(2.5)
    }
    let cases = [
        ("L1DistLoss()", "L1DistLoss()", testl1 as fn(f32, f32) -> f32),
        (
            "customloss(x, y) = abs(x - y)^2.5",
            "customloss(x, y, w) = w * (abs(x - y)^2.5)",
            customloss,
        ),
    ];
    for (unweighted, weighted, evaluator) in cases {
        let loss = parse_loss::<f32>(unweighted).unwrap();
        let want = x.iter().zip(&y).map(|(&a, &b)| evaluator(a, b)).sum::<f32>() / 100.0;
        let got = loss.loss(&x, &y, None);
        assert!((got - want).abs() < 1e-6, "{unweighted}: {got} vs {want}");

        let loss = parse_loss::<f32>(weighted).unwrap();
        let want = x
            .iter()
            .zip(&y)
            .zip(&w)
            .map(|((&a, &b), &wi)| wi * evaluator(a, b))
            .sum::<f32>()
            / w.iter().sum::<f32>();
        let got = loss.loss(&x, &y, Some(&w));
        assert!((got - want).abs() < 1e-6, "{weighted}: {got} vs {want}");
    }
}

#[test]
fn loss_scale_parameter_validation() {
    assert_eq!("log".parse::<LossScale>(), Ok(LossScale::Log));
    assert_eq!("linear".parse::<LossScale>(), Ok(LossScale::Linear));
    for bad in ["invalid", "cubic"] {
        let err = bad.parse::<LossScale>().unwrap_err();
        assert!(err.contains("`loss_scale` must be either log or linear"), "{err}");
    }

    assert_eq!(Options::<T, TestOps, D>::default().loss_scale, LossScale::Log);
    let linear = Options::<T, TestOps, D> {
        loss_scale: LossScale::Linear,
        ..Default::default()
    };
    assert_eq!(linear.loss_scale, LossScale::Linear);
}

#[test]
fn loss_scale_score_computation() {
    let close = |a: f64, b: f64| (a - b).abs() < 1e-5;
    assert!(close(compute_direct_score(0.5, 1.0, 1.0), 0.5));
    assert!(close(compute_direct_score(1.2, 1.0, 1.0), 0.0));
    assert!(close(compute_zero_centered_score(0.5, 1.0, 1.0), 2f64.ln()));
    assert!(close(compute_zero_centered_score(0.1, 1.0, 1.0), 10f64.ln()));
    assert!(close(compute_zero_centered_score(2.0, 1.0, 1.0), 0.0));
}

#[test]
fn loss_scale_in_choose_best() {
    let losses = [0.5, 0.3, 0.4, 0.2];
    let scores = [0.1, 0.8, 0.5, 0.3];
    assert_eq!(choose_best(&losses, &scores, LossScale::Log), Some(1));
    assert_eq!(choose_best(&losses, &scores, LossScale::Linear), Some(3));
    assert_eq!(choose_best::<f64>(&[], &[], LossScale::Log), None);
}

#[test]
fn loss_scale_in_pareto_volume() {
    let complexities = [1, 3, 5];
    assert!(pareto_volume(&[0.5, 0.3, 0.2], &complexities, 10, LossScale::Log) > 0.0);
    assert!(pareto_volume(&[0.5, 0.3, 0.2], &complexities, 10, LossScale::Linear) > 0.0);
    assert!(pareto_volume(&[0.1, -0.1, -0.5], &complexities, 10, LossScale::Linear) > 0.0);
}

fn hall_with_x1(loss: T) -> HallOfFame<T, TestOps, D> {
    let expr = PostfixExpr::<T, TestOps, D>::new(vec![PNode::Var { feature: 0 }], vec![], Default::default());
    let mut member = PopMember::from_expr_with_birth(0, expr, 2);
    member.complexity = 1;
    member.loss = loss;
    member.cost = loss;
    let mut hall = HallOfFame::new(30);
    hall.best_by_complexity[1] = Some(member);
    hall
}

#[test]
fn loss_scale_error_handling() {
    let hall = hall_with_x1(-1.0);

    let err = hall.pareto_scores(LossScale::Log).unwrap_err();
    assert!(err.contains("must be non-negative"), "{err}");
    assert!(err.contains("set the `loss_scale` to linear"), "{err}");

    let scores = hall.pareto_scores(LossScale::Linear).unwrap();
    assert_eq!(hall.pareto_front()[0].loss, -1.0);
    assert!(scores[0] >= 0.0);
}

#[cfg(feature = "progress")]
#[test]
fn pareto_curve_header_shows_score_only_for_log_scale() {
    use crate::progress_bars::{RenderOptions, format_hall_of_fame};

    let hall = hall_with_x1(0.5);
    let ops = Options::<T, TestOps, D>::default().operators.set;
    let table = |loss_scale, ansi| {
        format_hall_of_fame(&hall, &ops, 100, usize::MAX, RenderOptions { ansi, loss_scale }, 5).unwrap()
    };

    let log = table(LossScale::Log, false);
    for column in ["Complexity", "Loss", "Score", "Equation"] {
        assert!(log.contains(column), "{log}");
    }
    let linear = table(LossScale::Linear, false);
    for column in ["Complexity", "Loss", "Equation"] {
        assert!(linear.contains(column), "{linear}");
    }
    assert!(!linear.contains("Score"), "{linear}");

    for loss_scale in [LossScale::Log, LossScale::Linear] {
        let plain = console::strip_ansi_codes(&table(loss_scale, true)).into_owned();
        let lines: Vec<&str> = plain.lines().collect();
        let header_col = lines[1].find("Equation").unwrap();
        let row_col = lines[2].find("x0").unwrap();
        assert_eq!(header_col, row_col, "equation column misaligned:\n{plain}");
    }
}

#[cfg(feature = "progress")]
#[test]
fn pareto_curve_rejects_negative_losses_under_log_scale() {
    use crate::progress_bars::{RenderOptions, format_hall_of_fame};

    let hall = hall_with_x1(-1.0);
    let ops = Options::<T, TestOps, D>::default().operators.set;
    let table = |loss_scale| {
        format_hall_of_fame(
            &hall,
            &ops,
            100,
            usize::MAX,
            RenderOptions {
                ansi: false,
                loss_scale,
            },
            5,
        )
    };
    let err = table(LossScale::Log).unwrap_err();
    assert!(err.contains("negative loss (-1)"), "{err}");
    assert!(table(LossScale::Linear).unwrap().contains("-1.000e0"));
}

#[test]
fn early_stop_condition() {
    let mut rng = fastrand::Rng::with_seed(0);
    let n_rows = 100;
    let x = Array2::from_shape_fn((5, n_rows), |_| rng.f64() * 4.0 - 2.0);
    let y = Array1::from_iter((0..n_rows).map(|i| 2.0 * x[(3, i)].cos() + x[(0, i)] * x[(0, i)]));
    let dataset = Dataset::new(x, y);

    let src = "early_stop(loss, c) = ((loss <= 1e-10) && (c <= 10))";
    let early_stop = parse_early_stop::<f64>(src).unwrap();
    let options = Options::<f64, BuiltinOpsF64, 2> {
        operators: BuiltinOpsF64::new()
            .from_names(&["+", "*", "/", "sub", "cos", "exp"])
            .unwrap(),
        populations: 20,
        niterations: 1_000_000,
        seed: 0,
        deterministic: true,
        progress: false,
        early_stop_condition: Some(early_stop.clone()),
        ..Default::default()
    };

    let mut engine = SearchEngine::new(dataset, options);
    engine.step(usize::MAX).unwrap();
    assert!(engine.is_finished());
    assert!(engine.cycles_completed() < engine.total_cycles());
    assert!(
        engine
            .hall_of_fame()
            .members()
            .any(|m| early_stop(m.loss, m.complexity))
    );
}
