//! A seeded search recovers a target that needs an operator written as a string.

use ndarray::{Array1, Array2};
use symbolic_regression::prelude::*;

#[test]
fn search_recovers_target_through_string_operator() {
    let n_rows = 128usize;
    let mut rng = fastrand::Rng::with_seed(0);
    let x = Array2::from_shape_fn((2, n_rows), |_| rng.f64() * 6.0 - 3.0);
    let y = Array1::from_iter((0..n_rows).map(|i| x[(0, i)] * (-x[(1, i)] * x[(1, i)]).exp()));
    let dataset = Dataset::new(x, y);

    let (custom, names) = parse_operator_tokens::<f64, _>(["+", "sub", "*", "f(x, y) = x*exp(-y^2)"]).unwrap();
    let operator_set = BuiltinOpsF64::new().with_custom(custom);
    let options = Options::<f64, _, 2> {
        operators: operator_set.from_names(&names).unwrap(),
        seed: 0,
        deterministic: true,
        niterations: 40,
        populations: 8,
        maxsize: 10,
        progress: false,
        ..Default::default()
    };

    let result = equation_search::<f64, BuiltinOpsF64<CustomOps<f64>>, 2>(&dataset, &options);
    let best = result
        .hall_of_fame
        .pareto_front()
        .into_iter()
        .find(|m| m.loss <= 1e-10)
        .expect("no hall-of-fame member reached loss 1e-10");
    let equation = string_tree(&best.expr, &options.operators.set, Default::default());
    println!(
        "seed {}: complexity {} loss {:e} {equation}",
        options.seed, best.complexity, best.loss
    );
    assert_eq!(equation, "f(x0, x1)");
}

#[test]
fn external_package_operator_fails_before_search() {
    let err = parse_operator_tokens::<f64, _>(["+", "p(i) = Primes.prime(round(i))"]).unwrap_err();
    assert!(err.message.contains("`Primes.prime`"), "{}", err.message);
}
