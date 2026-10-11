use ndarray::{Array1, Array2};

use super::common::{D, T, TestOps};
use crate::operator_library::OperatorLibrary;
use crate::search_utils::SearchEngine;
use crate::{EarlyStop, Options};

#[test]
fn search_terminates_when_early_stop_condition_satisfied() {
    // y == 0 everywhere, so the trivial zero-constant member already in the initial population
    // produces loss == 0. Any non-trivial early-stop threshold should fire immediately.
    let dataset = crate::Dataset::new(
        Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]),
    );

    let options = Options::<T, TestOps, D> {
        operators: OperatorLibrary::sr_default::<TestOps, D>(),
        use_baseline: false,
        niterations: 100,
        populations: 2,
        population_size: 4,
        ncycles_per_iteration: 5,
        early_stop_condition: Some(EarlyStop::below::<T>(1e-6)),
        ..Default::default()
    };

    let mut engine = SearchEngine::<T, TestOps, D>::new(vec![dataset], options).unwrap();
    engine.step(usize::MAX).unwrap();

    // The search should bail out long before the full niterations budget.
    assert!(engine.is_finished());
    assert!(engine.cycles_completed() < 100 * 2);
}

#[test]
fn early_stop_thresholds_are_julia_number_literals() {
    for (src, threshold) in [
        ("1e-6", 1e-6),
        ("1f-6", f64::from(1e-6_f32)),
        (" -2.5 ", -2.5),
        ("+.5", 0.5),
        ("1_000.0", 1000.0),
    ] {
        let stop = crate::parse_early_stop::<f64>(src).unwrap_or_else(|e| panic!("{src}: {e}"));
        assert!(stop(threshold - threshold.abs() * 1e-9, 1), "{src}");
        assert!(!stop(threshold, 1), "{src}");
    }
    let err = crate::parse_early_stop::<f64>("1e").err().unwrap();
    assert!(err.message.contains("an exponent needs digits"), "{err}");
}
