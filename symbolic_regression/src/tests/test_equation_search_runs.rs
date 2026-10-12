use std::sync::mpsc;
use std::time::Duration;

use dynamic_expressions::strings::StringTreeOptions;
use ndarray::{Array1, Array2};

use super::common::{D, T, TestOps};
use crate::operator_library::OperatorLibrary;
use crate::{BuiltinMutation, Options, Parallelism, SearchEngine, equation_search, resume_search};

fn quadratic_dataset() -> crate::Dataset<T> {
    let x: Vec<T> = (0..64).map(|i| i as T / 64.0).collect();
    let y = x.iter().map(|xi| xi * xi + xi).collect();
    crate::Dataset::new(Array2::from_shape_vec((1, 64), x).unwrap(), Array1::from_vec(y))
}

fn small_options() -> Options<T, TestOps, D> {
    Options {
        save_to_file: false,
        progress: false,
        seed: Some(0),
        operators: OperatorLibrary::sr_default::<TestOps, D>(),
        populations: 4,
        population_size: 20,
        niterations: 2,
        ncycles_per_iteration: 10,
        maxsize: 10,
        ..Default::default()
    }
}

#[test]
fn searches_reject_mutation_lists_without_a_positive_weight() {
    let dataset = quadratic_dataset();
    for default_mutations in [vec![(BuiltinMutation::DoNothing, 0.0)], Vec::new()] {
        let options = Options {
            default_mutations: Some(default_mutations),
            ..small_options()
        };
        let message = "at least one mutation weight must be positive";
        let search = equation_search(std::slice::from_ref(&dataset), &options);
        assert_eq!(search.err().as_deref(), Some(message));
        let engine = SearchEngine::new(vec![dataset.clone()], options);
        assert_eq!(engine.err().as_deref(), Some(message));
    }
}

#[test]
fn deterministic_searches_require_serial_parallelism() {
    let dataset = quadratic_dataset();
    let mut options = Options {
        deterministic: true,
        parallelism: Parallelism::Serial,
        ..small_options()
    };
    let previous = equation_search(std::slice::from_ref(&dataset), &options).unwrap();
    assert!(options.validate().is_ok());
    options.parallelism = Parallelism::Multithreading;
    let message = "Determinism is only guaranteed for serial mode.";
    assert_eq!(options.validate().err().as_deref(), Some(message));
    assert_eq!(
        equation_search(std::slice::from_ref(&dataset), &options)
            .err()
            .as_deref(),
        Some(message)
    );
    assert_eq!(
        resume_search(previous.clone(), std::slice::from_ref(&dataset), &options)
            .err()
            .as_deref(),
        Some(message)
    );
    assert_eq!(
        SearchEngine::new(vec![dataset.clone()], options.clone())
            .err()
            .as_deref(),
        Some(message)
    );
    assert_eq!(
        SearchEngine::resume(previous, vec![dataset], options).err().as_deref(),
        Some(message)
    );
}

/// Only constant mutations, which have no weight on a tree without constants: the first such
/// tree a worker mutates panics, as SymbolicRegression.jl's `_sample_mutation` throws.
#[test]
fn a_panicking_worker_fails_the_search_instead_of_hanging_it() {
    let options = Options {
        default_mutations: Some(vec![(BuiltinMutation::Constant(Default::default()), 1.0)]),
        parallelism: Parallelism::Multithreading,
        ..small_options()
    };
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.install(|| equation_search(&[quadratic_dataset()], &options).map(|_| ()))
        }));
        let message = outcome.map_err(|payload| {
            payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
        });
        tx.send(message).unwrap();
    });
    let outcome = rx
        .recv_timeout(Duration::from_secs(60))
        .expect("the search did not finish within 60 s");
    assert_eq!(
        outcome.err(),
        Some(Some(
            "at least one conditioned mutation weight must be positive".to_string()
        ))
    );
}

/// A serial search runs one population at a time, so the size of the rayon pool it runs in cannot
/// change its result.
#[test]
fn serial_deterministic_searches_ignore_the_thread_count() {
    let options = Options {
        deterministic: true,
        parallelism: Parallelism::Serial,
        verbosity: Some(0),
        ..small_options()
    };
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        let result = pool.install(|| equation_search(&[quadratic_dataset()], &options).unwrap());
        let front: Vec<_> = result.halls_of_fame[0]
            .pareto_front()
            .iter()
            .map(|m| {
                let equation = m.expr.string(&options, StringTreeOptions::default());
                (m.complexity, m.loss.to_bits(), equation)
            })
            .collect();
        (front, result.total_evals.to_bits())
    };
    let one_thread = run(1);
    assert!(!one_thread.0.is_empty());
    assert_eq!(run(4), one_thread);
}

#[test]
fn equation_search_runs() {
    let n_rows = 64;
    let n_features = 1;
    let mut x = vec![0.0; n_rows];
    let mut y = vec![0.0; n_rows];
    for i in 0..n_rows {
        let xi = (i as T) / (n_rows as T);
        x[i] = xi;
        y[i] = xi * xi + xi;
    }
    let dataset = crate::Dataset::new(
        Array2::from_shape_vec((n_features, n_rows), x).unwrap(),
        Array1::from_vec(y),
    );

    let options = Options::<T, TestOps, D> {
        save_to_file: false,
        seed: Some(123),
        operators: OperatorLibrary::sr_default::<TestOps, D>(),
        populations: 2,
        population_size: 40,
        niterations: 2,
        ncycles_per_iteration: 10,
        maxsize: 10,
        maxdepth: Some(8),
        migration: false,
        hof_migration: false,
        optimizer_probability: 0.0,
        ..Default::default()
    };

    let result = equation_search::<T, TestOps, D>(std::slice::from_ref(&dataset), &options).unwrap();
    assert!(result.best[0].clone().unwrap().loss.is_finite());
}
