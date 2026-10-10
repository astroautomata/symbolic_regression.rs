//! Seeded searches driven by a loss and an early-stop condition written as strings.

use ndarray::{Array1, Array2};
use symbolic_regression::prelude::*;

type Ops = BuiltinOpsF64;

fn dataset() -> Dataset<f64> {
    let n_rows = 128usize;
    let mut rng = fastrand::Rng::with_seed(0);
    let x = Array2::from_shape_fn((2, n_rows), |_| rng.f64() * 6.0 - 3.0);
    let y = Array1::from_iter((0..n_rows).map(|i| x[(0, i)] * x[(0, i)] - 1.7 * x[(1, i)]));
    Dataset::new(x, y)
}

fn options() -> Options<f64, Ops, 2> {
    Options {
        operators: BuiltinOpsF64::new().from_names(["+", "sub", "*"]).unwrap(),
        seed: 0,
        deterministic: true,
        niterations: 40,
        populations: 8,
        maxsize: 12,
        progress: false,
        ..Default::default()
    }
}

#[test]
fn search_recovers_target_under_string_loss() {
    let options = Options {
        loss: parse_loss("myloss(x, y) = abs(x - y)^1.5").unwrap(),
        ..options()
    };
    let result = equation_search::<f64, Ops, 2>(&dataset(), &options);
    let best = result
        .hall_of_fame
        .pareto_front()
        .into_iter()
        .find(|m| m.loss <= 1e-9)
        .expect("no hall-of-fame member reached loss 1e-9");
    let equation = string_tree(&best.expr, &options.operators.set, Default::default());
    println!("complexity {} loss {:e} {equation}", best.complexity, best.loss);

    let x = Array2::from_shape_vec((2, 3), vec![0.5, -2.0, 2.5, 1.0, 0.25, -1.5]).unwrap();
    let (yhat, ok) = eval_tree_array(&best.expr, &options.operators.set, x.view(), &EvalOptions::default());
    assert!(ok);
    for (i, got) in yhat.iter().enumerate() {
        let want = x[(0, i)] * x[(0, i)] - 1.7 * x[(1, i)];
        assert!((got - want).abs() < 1e-4, "{equation} at column {i}: {got} vs {want}");
    }
}

#[test]
fn string_early_stop_ends_search_sooner() {
    let run = |early_stop: Option<&str>| {
        let options = Options {
            early_stop_condition: early_stop.map(|s| parse_early_stop(s).unwrap()),
            ..options()
        };
        let mut engine = SearchEngine::new(dataset(), options);
        engine.step(usize::MAX);
        (engine.cycles_completed(), engine.total_evals(), engine.total_cycles())
    };
    let (full_cycles, full_evals, total_cycles) = run(None);
    let (cycles, evals, _) = run(Some("f(loss, complexity) = loss < 1e-8 && complexity <= 12"));
    println!("without early stop: {full_cycles} cycles, {full_evals} evals; with: {cycles} cycles, {evals} evals");
    assert_eq!(full_cycles, total_cycles);
    assert!(cycles < full_cycles / 2, "{cycles} vs {full_cycles}");
    assert!(evals < full_evals / 2, "{evals} vs {full_evals}");
}

#[test]
#[should_panic(expected = "must accept (prediction, target, weight) when the dataset has weights")]
fn two_argument_loss_with_weights_fails_before_search() {
    let data = dataset();
    let weights = Some(Array1::ones(data.n_rows));
    let weighted = Dataset::with_weights_and_names(data.x.clone(), data.y.clone(), weights, Vec::new());
    let options = Options {
        loss: parse_loss("myloss2(prediction, target) = (prediction - target)^2").unwrap(),
        niterations: 1,
        ..options()
    };
    equation_search::<f64, Ops, 2>(&weighted, &options);
}

#[test]
fn strided_weights_score_like_contiguous_ones() {
    let data = dataset();
    let mut strided = Array1::from_iter((0..2 * data.n_rows).map(|i| 1.0 + (i / 2) as f64));
    strided.slice_collapse(ndarray::s![..;2]);
    assert!(strided.as_slice().is_none());
    let contiguous = strided.to_vec();

    let options = Options {
        loss: parse_loss("(x, y, w) -> w * abs(x - y)").unwrap(),
        niterations: 1,
        ..options()
    };
    let run = |weights: Array1<f64>| {
        let weighted = Dataset::with_weights_and_names(data.x.clone(), data.y.clone(), Some(weights), Vec::new());
        let result = equation_search::<f64, Ops, 2>(&weighted, &options);
        result
            .hall_of_fame
            .pareto_front()
            .iter()
            .map(|m| m.loss)
            .collect::<Vec<_>>()
    };
    assert_eq!(run(strided), run(Array1::from(contiguous)));
}
