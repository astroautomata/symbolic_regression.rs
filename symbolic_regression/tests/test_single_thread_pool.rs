use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier, mpsc};
use std::time::{Duration, Instant};

use ndarray::{Array1, Array2};
use symbolic_regression::{Dataset, Options, equation_search};

symbolic_regression::op!(Square for f64 {
    eval: |[x]| { x * x },
    partial: |[x], _idx| { 2.0 * x },
});

symbolic_regression::opset! {
    SquareOps for f64 { Square }
}

fn run_with_deadline(test_name: &str) -> bool {
    const CHILD_TEST: &str = "SINGLE_THREAD_POOL_CHILD_TEST";
    if std::env::var(CHILD_TEST).as_deref() == Ok(test_name) {
        return true;
    }
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test_name, "--nocapture"])
        .env(CHILD_TEST, test_name)
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success(), "{test_name} child failed: {status}");
            return false;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("{test_name} did not finish within 10 s");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// A search called from outside a one-thread global pool must leave that thread available for cycles.
#[test]
fn search_finishes_outside_a_one_thread_pool() {
    if !run_with_deadline("search_finishes_outside_a_one_thread_pool") {
        return;
    }
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global().unwrap();
    let x: Vec<f64> = (0..32).map(|i| i as f64 * 0.1).collect();
    let y = x.iter().map(|v| v * v).collect();
    let dataset = Dataset::new(Array2::from_shape_vec((1, 32), x).unwrap(), Array1::from_vec(y));
    let options = Options::<f64, SquareOps, 1> {
        niterations: 2,
        populations: 2,
        population_size: 20,
        ncycles_per_iteration: 10,
        progress: false,
        operators: SquareOps::new().from_names(["square"]).unwrap(),
        ..Default::default()
    };
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let result = equation_search::<f64, SquareOps, 1>(&dataset, &options);
        tx.send(result.best.loss).unwrap();
    });
    let loss = rx
        .recv_timeout(Duration::from_secs(60))
        .expect("the search did not finish within 60 s");
    assert!(loss.is_finite());
}

#[test]
fn concurrent_searches_finish_outside_a_two_thread_pool() {
    if !run_with_deadline("concurrent_searches_finish_outside_a_two_thread_pool") {
        return;
    }
    rayon::ThreadPoolBuilder::new().num_threads(2).build_global().unwrap();
    let start = Arc::new(Barrier::new(2));
    let first_cycle = Arc::new(Barrier::new(2));
    let searches: Vec<_> = (0..2)
        .map(|_| {
            let start = Arc::clone(&start);
            let first_cycle = Arc::clone(&first_cycle);
            std::thread::spawn(move || {
                let x: Vec<f64> = (0..8).map(|i| i as f64 * 0.1).collect();
                let y = x.iter().map(|v| v * v).collect();
                let dataset = Dataset::new(Array2::from_shape_vec((1, 8), x).unwrap(), Array1::from_vec(y));
                let reached_first_cycle = Arc::new(AtomicBool::new(false));
                let reached = Arc::clone(&reached_first_cycle);
                let options = Options::<f64, SquareOps, 1> {
                    niterations: 2,
                    populations: 1,
                    population_size: 20,
                    ncycles_per_iteration: 1,
                    progress: false,
                    should_optimize_constants: false,
                    should_simplify: false,
                    operators: SquareOps::new().from_names(["square"]).unwrap(),
                    early_stop_condition: Some(Arc::new(move |_, _| {
                        // The callback runs on the coordinator. Neither search can finish its
                        // first cycle until the other coordinator is also running.
                        if !reached.swap(true, Ordering::Relaxed) {
                            first_cycle.wait();
                        }
                        false
                    })),
                    ..Default::default()
                };
                start.wait();
                assert!(rayon::current_thread_index().is_none());
                let result = equation_search::<f64, SquareOps, 1>(&dataset, &options);
                assert!(reached_first_cycle.load(Ordering::Relaxed));
                assert!(result.best.loss.is_finite());
            })
        })
        .collect();
    for search in searches {
        search.join().unwrap();
    }
}
