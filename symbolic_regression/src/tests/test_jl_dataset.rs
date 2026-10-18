//! Ports of SymbolicRegression.jl's test/unit/dataset testitems (see tools/parity_tests.toml).

use std::panic::catch_unwind;

use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
use fastrand::Rng;
use ndarray::{Array1, Array2};

use crate::dataset::Dataset;
use crate::options::Options;
use crate::search_utils::{LARGE_DATASET_WARNING, SearchEngine};

fn random(n_features: usize, n_rows: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut rng = Rng::with_seed(seed);
    let x = Array2::from_shape_fn((n_features, n_rows), |_| rng.f64());
    let y = Array1::from_shape_fn(n_rows, |_| rng.f64());
    (x, y)
}

/// test/unit/dataset/test_dataset.jl | "Dataset mismatched X/y dimensions". The port has no
/// dataset without a target.
#[test]
fn dataset_mismatched_x_y_dimensions() {
    let message = |x: Array2<f64>, y: Array1<f64>| {
        let err = catch_unwind(|| Dataset::new(x, y)).unwrap_err();
        err.downcast_ref::<String>().cloned().unwrap()
    };
    assert_eq!(
        message(Array2::zeros((32, 3)), Array1::zeros(32)),
        "Number of samples in `x` (x.ncols() = 3) does not match the length of `y` (32). `x` should have shape (n_features, n_rows)."
    );
    assert!(message(Array2::zeros((3, 32)), Array1::zeros(31)).contains("the length of `y` (31)"));
    let (x, y) = random(3, 32, 0);
    let dataset = Dataset::new(x, y);
    assert_eq!((dataset.n_features, dataset.n_rows), (3, 32));
}

/// test/unit/dataset/test_dataset.jl | "Vector output dataset". The port has no tuple-valued
/// targets; a search takes several targets as one dataset per output, sharing `x`.
#[test]
fn vector_output_dataset() {
    let (x, _) = random(3, 32, 0);
    let y = Array2::from_shape_fn((3, 32), |(j, i)| (j * 32 + i) as f64);
    let datasets = Dataset::multi_output(x.clone(), y.clone(), None, Vec::new());
    assert_eq!(datasets.len(), 3);
    for (j, d) in datasets.iter().enumerate() {
        assert_eq!(d.x, x);
        assert_eq!(d.y, y.row(j));
        assert_eq!(d.x.as_ptr(), datasets[0].x.as_ptr());
    }
}

/// test/unit/dataset/test_dataset.jl | "Large dataset warning". SymbolicRegression.jl logs the
/// warning (Configure.jl:113) once; the port records it once per search, whatever the number of
/// outputs. There is no `verbosity` option.
#[test]
fn large_dataset_warning() {
    let options = Options::<f64, BuiltinOpsF64, 2> {
        operators: BuiltinOpsF64::new().from_names(["+"]).unwrap(),
        niterations: 0,
        populations: 1,
        population_size: 2,
        progress: false,
        ..Default::default()
    };
    let warnings = |n_rows: usize, n_outputs: usize| {
        let y = Array2::zeros((n_outputs, n_rows));
        let datasets = Dataset::multi_output(Array2::zeros((1, n_rows)), y, None, Vec::new());
        SearchEngine::new(datasets, options.clone())
            .unwrap()
            .warnings()
            .to_vec()
    };
    assert!(warnings(50_000, 1).is_empty());
    assert_eq!(warnings(50_001, 1), [LARGE_DATASET_WARNING]);
    assert_eq!(warnings(50_001, 2), [LARGE_DATASET_WARNING]);
    assert!(LARGE_DATASET_WARNING.contains("more than 50,000"));
}

/// test/unit/dataset/test_batched_dataset.jl | "SubDataset properties". A batch is a buffer that
/// the search refills from the full dataset; the port has no dataset without a target.
#[test]
fn sub_dataset_properties() {
    let (x, y) = random(3, 32, 0);
    let weights = Array1::from_shape_fn(32, |i| i as f64);
    let names = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let dataset = Dataset::with_weights_and_names(x.clone(), y.clone(), Some(weights), names);

    let batch = |full: &Dataset<f64>, size: usize, seed: u64| {
        let mut batch = Dataset::make_batch_buffer(full, size);
        batch.resample_from(full, &mut Rng::with_seed(seed));
        batch
    };
    let batched = batch(&dataset, 16, 42);
    assert_eq!(batched.x.dim(), (3, 16));
    assert_eq!(batched.y.len(), 16);
    assert_eq!(batched.weights.as_ref().unwrap().len(), 16);
    assert_eq!(batched.n_rows, 16);
    assert_eq!(batched.n_features, dataset.n_features);
    assert_eq!(batched.variable_names, dataset.variable_names);
    assert_eq!(batched.avg_y, dataset.avg_y);
    for i in 0..16 {
        let row = (0..32).find(|&r| dataset.x.column(r) == batched.x.column(i)).unwrap();
        assert_eq!(batched.y[i], dataset.y[row]);
        assert_eq!(batched.weights.as_ref().unwrap()[i], row as f64);
    }

    assert_eq!(batch(&dataset, 16, 42).x, batched.x);

    let batched8 = batch(&dataset, 8, 0);
    assert_eq!(batched8.x.dim(), (3, 8));
    assert_eq!(batched8.y.len(), 8);
    assert_eq!(batched8.weights.as_ref().unwrap().len(), 8);

    assert!(batch(&Dataset::new(x, y), 16, 0).weights.is_none());
}
