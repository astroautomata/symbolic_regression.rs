//! A search resumed from its returned state keeps its hall of fame and evaluation count, and
//! refuses options that change what the saved state means.

use ndarray::{Array1, Array2};
use symbolic_regression::prelude::*;

type Ops = BuiltinOpsF64;

fn options(operators: &[&str], niterations: usize) -> Options<f64, Ops, 2> {
    Options {
        operators: Ops::new().from_names(operators).unwrap(),
        seed: 0,
        deterministic: true,
        niterations,
        populations: 4,
        maxsize: 12,
        progress: false,
        ..Default::default()
    }
}

#[test]
fn resumed_search_keeps_its_hall_of_fame_and_counters() {
    let n_rows = 64usize;
    let mut rng = fastrand::Rng::with_seed(0);
    let x = Array2::from_shape_fn((2, n_rows), |_| rng.f64() * 4.0 - 2.0);
    let y = Array1::from_iter((0..n_rows).map(|i| x[(0, i)] * x[(0, i)] * x[(1, i)] + (x[(1, i)]).cos()));
    let datasets = [Dataset::new(x, y)];
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let first_options = options(&["+", "sub", "*", "cos"], 3);
    let first = pool.install(|| equation_search(&datasets, &first_options).unwrap());
    let resume_options = options(&["+", "sub", "*", "cos"], 4);
    let resumed = pool.install(|| resume_search(first.clone(), &datasets, &resume_options).unwrap());

    assert!(
        resumed.total_evals > first.total_evals,
        "{} <= {}",
        resumed.total_evals,
        first.total_evals
    );
    let before = &first.halls_of_fame[0].best_by_complexity;
    let after = &resumed.halls_of_fame[0].best_by_complexity;
    for (c, (b, a)) in before.iter().zip(after).enumerate() {
        if let Some(b) = b {
            let a = a
                .as_ref()
                .unwrap_or_else(|| panic!("complexity {c} left the hall of fame"));
            assert!(a.cost <= b.cost, "complexity {c}: {} > {}", a.cost, b.cost);
        }
    }

    let mut engine = SearchEngine::resume(first.clone(), datasets.to_vec(), resume_options.clone()).unwrap();
    assert_eq!(engine.total_evals(), first.total_evals as u64);
    assert_eq!(engine.cycles_completed(), 0);
    assert_eq!(engine.total_cycles(), 4 * resume_options.populations);
    engine.step(1).unwrap();
    assert_eq!(engine.cycles_completed(), 1);
    assert!(engine.total_evals() > first.total_evals as u64);
}

#[test]
fn warm_start_rejects_changed_maxsize_or_operators() {
    let mut rng = fastrand::Rng::with_seed(1);
    let x = Array2::from_shape_fn((2, 16), |_| rng.f64());
    let y = x.row(0).to_owned() + x.row(1);
    let datasets = [Dataset::new(x, y)];
    let saved = equation_search(&datasets, &options(&["+", "*"], 1)).unwrap();

    let mut bigger = options(&["+", "*"], 1);
    bigger.maxsize = 20;
    let cases = [(bigger, "maxsize"), (options(&["+", "*", "sub"], 1), "operators")];
    for (changed, field) in cases {
        let err = resume_search(saved.clone(), &datasets, &changed).err().unwrap();
        assert_eq!(
            err,
            format!("Warm start incompatible due to changed field(s): {field}. Start a new search instead.")
        );
    }
}

/// Custom operators are numbered in the order they are defined, so a saved tree's operator ids
/// name other functions once the definitions are reordered.
#[test]
fn warm_start_rejects_renumbered_custom_operators() {
    let mut rng = fastrand::Rng::with_seed(2);
    let x = Array2::from_shape_fn((1, 16), |_| rng.f64());
    let y = x.row(0).mapv(|v| v + 1.0);
    let datasets = [Dataset::new(x, y)];
    let options = |definitions: [&str; 2]| {
        let (custom, _) = parse_operator_tokens::<f64, _>(definitions).unwrap();
        Options::<f64, BuiltinOpsF64<CustomOps<f64>>, 2> {
            operators: BuiltinOpsF64::new()
                .with_custom(custom)
                .from_names(["+", "f", "g"])
                .unwrap(),
            niterations: 1,
            populations: 2,
            progress: false,
            ..Default::default()
        }
    };
    let saved = equation_search(&datasets, &options(["f(x) = x + 1", "g(x) = x * 3"])).unwrap();
    let Err(err) = resume_search(saved, &datasets, &options(["g(x) = x * 3", "f(x) = x + 1"])) else {
        panic!("a search resumed with its custom operators renumbered");
    };
    assert_eq!(
        err,
        "Warm start incompatible due to changed field(s): operators. Start a new search instead."
    );
}

#[test]
fn warm_start_rejects_a_changed_feature_count_on_any_output() {
    let mut rng = fastrand::Rng::with_seed(3);
    let mut dataset = |n_features| {
        let x = Array2::from_shape_fn((n_features, 16), |_| rng.f64());
        let y = x.row(0).to_owned();
        Dataset::new(x, y)
    };
    let saved_datasets = [dataset(2), dataset(2)];
    let saved = equation_search(&saved_datasets, &options(&["+", "*"], 1)).unwrap();
    let datasets = [dataset(2), dataset(1)];
    let Err(err) = resume_search(saved, &datasets, &options(&["+", "*"], 1)) else {
        panic!("a search resumed with fewer features on its second output");
    };
    assert_eq!(
        err,
        "Warm start incompatible due to changed field(s): number of features. Start a new search instead."
    );
}

/// test/unit/evaluation/test_early_stop.jl | "State preservation with niterations=0". The saved
/// state comes from a search rather than hand-built populations, which the port does not expose.
#[test]
fn state_preservation_with_niterations_0() {
    let mut rng = fastrand::Rng::with_seed(42);
    let x = Array2::from_shape_fn((2, 10), |_| rng.f64() * 2.0 - 1.0);
    let y = x.row(0).to_owned() + x.row(1);
    let datasets = [Dataset::new(x, y)];
    let options = Options::<f64, Ops, 2> {
        operators: Ops::new().from_names(["+"]).unwrap(),
        niterations: 0,
        population_size: 5,
        populations: 2,
        maxsize: 5,
        tournament_selection_n: 2,
        progress: false,
        ..Default::default()
    };
    let saved = equation_search(&datasets, &options).unwrap();
    let result = resume_search(saved.clone(), &datasets, &options).unwrap();

    assert_eq!(result.populations[0].len(), 2);
    for (pop, saved_pop) in result.populations[0].iter().zip(&saved.populations[0]) {
        assert_eq!(pop.members.len(), 5);
        let hashes: Vec<u64> = pop.members.iter().map(|m| m.expr.hash_nodes()).collect();
        let saved_hashes: Vec<u64> = saved_pop.members.iter().map(|m| m.expr.hash_nodes()).collect();
        assert_eq!(hashes, saved_hashes);
    }
    assert!(result.warnings.is_empty(), "{:?}", result.warnings);
}
