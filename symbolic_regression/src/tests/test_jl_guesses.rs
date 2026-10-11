//! Ports of SymbolicRegression.jl's test/unit/misc/test_guesses.jl (see tools/parity_tests.toml). Its
//! guesses name features `x1`, `x2`, ..., so the datasets here carry those names.

use dynamic_expressions::OperatorSet;
use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
use ndarray::{Array1, Array2};

use crate::dataset::Dataset;
use crate::guesses::Guess;
use crate::options::Options;
use crate::search_utils::equation_search;

type Ops = BuiltinOpsF64;
const D: usize = 2;

fn names(n: usize) -> Vec<String> {
    (1..=n).map(|i| format!("x{i}")).collect()
}

fn random_dataset(n_features: usize, n_rows: usize, seed: u64, f: impl Fn(&[f64]) -> f64) -> Dataset<f64> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let x = Array2::from_shape_fn((n_features, n_rows), |_| rng.f64() * 4.0 - 2.0);
    let y = Array1::from_iter(x.columns().into_iter().map(|c| f(&c.to_vec())));
    Dataset::with_weights_and_names(x, y, None, names(n_features))
}

fn options(operators: &[&str], guesses: &[&str]) -> Options<f64, Ops, D> {
    Options {
        operators: Ops::new().from_names(operators).unwrap(),
        guesses: vec![guesses.iter().map(|&g| Guess::from(g)).collect()],
        niterations: 0,
        populations: 2,
        progress: false,
        seed: 0,
        deterministic: true,
        ..Default::default()
    }
}

/// test/unit/misc/test_guesses.jl | "Guesses violating constraints do not enter the Hall of Fame"
#[test]
fn guesses_violating_constraints_do_not_enter_the_hall_of_fame() {
    let dataset = random_dataset(2, 8, 0, |_| 0.0);
    let mut options = options(&["+"], &["x1 + x2"]);
    let plus = options.operators.set.lookup_with_arity("+", 2).unwrap();
    options.op_constraints.set_op_arg_constraint(plus, 0, 0);
    options.op_constraints.set_op_arg_constraint(plus, 1, 0);

    let result = equation_search(std::slice::from_ref(&dataset), &options).unwrap();
    assert_eq!(result.halls_of_fame[0].members().count(), 0);
}
