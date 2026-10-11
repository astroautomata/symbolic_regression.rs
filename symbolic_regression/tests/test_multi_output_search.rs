//! A search with a matrix target evolves one Pareto front per output.

use std::sync::Arc;

use ndarray::{Array2, Axis};
use symbolic_regression::prelude::*;

#[test]
fn two_output_search_recovers_each_target_on_its_own_front() {
    let n_rows = 100usize;
    let mut rng = fastrand::Rng::with_seed(0);
    let x = Array2::from_shape_fn((2, n_rows), |_| rng.f64() * 4.0 - 2.0);
    let y = ndarray::stack![
        Axis(0),
        x.row(0).to_owned() * x.row(1),
        x.row(0).to_owned() + x.row(1).to_owned() * x.row(1)
    ];
    let datasets = Dataset::multi_output(x, y, None, Vec::new());
    assert_eq!(datasets.len(), 2);
    assert_eq!(datasets[0].x.as_ptr(), datasets[1].x.as_ptr());

    let options = Options::<f64, BuiltinOpsF64, 2> {
        save_to_file: false,
        operators: BuiltinOpsF64::new().from_names(["+", "*"]).unwrap(),
        seed: 0,
        deterministic: true,
        niterations: 40,
        populations: 4,
        maxsize: 10,
        progress: false,
        early_stop_condition: Some(Arc::new(|l, c| l <= 1e-12 && c <= 5)),
        ..Default::default()
    };
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let result = pool.install(|| equation_search(&datasets, &options).unwrap());

    assert_eq!(result.halls_of_fame.len(), 2);
    assert_eq!(result.populations.len(), 2);
    let exact: Vec<String> = result
        .halls_of_fame
        .iter()
        .enumerate()
        .map(|(j, hall)| {
            let front = hall.pareto_front();
            for m in &front {
                println!(
                    "output {j}: {}\t{:e}\t{}",
                    m.complexity,
                    m.loss,
                    m.expr.string(&options, Default::default())
                );
            }
            let m = front
                .iter()
                .find(|m| m.loss <= 1e-12)
                .unwrap_or_else(|| panic!("output {j} has no member with loss 1e-12"));
            m.expr.string(&options, Default::default())
        })
        .collect();
    assert!(["x0 * x1", "x1 * x0"].contains(&exact[0].as_str()), "{}", exact[0]);
    assert!(
        ["x0 + (x1 * x1)", "(x1 * x1) + x0"].contains(&exact[1].as_str()),
        "{}",
        exact[1]
    );
}
