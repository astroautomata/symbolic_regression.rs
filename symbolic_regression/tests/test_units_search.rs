//! Seeded searches with units: the dimensional penalty steers each search to the consistent one
//! of two equations that fit equally well.

use std::sync::Arc;

use dynamic_expressions::strings::StringTreeOptions;
use ndarray::{Array1, Array2};
use symbolic_regression::PopMember;
use symbolic_regression::prelude::*;
use symbolic_regression::units::UnitLabels;

type Ops = BuiltinOpsF64<CustomOps<f64>>;

fn options(tokens: &[&str], stop: fn(f64, usize) -> bool) -> Options<f64, Ops, 2> {
    let (custom, names) = parse_operator_tokens::<f64, _>(tokens.iter().copied()).unwrap();
    Options {
        operators: BuiltinOpsF64::new().with_custom(custom).from_names(&names).unwrap(),
        seed: 0,
        deterministic: true,
        niterations: 100,
        populations: 8,
        progress: false,
        early_stop_condition: Some(Arc::new(stop)),
        ..Default::default()
    }
}

/// Runs the search on one thread, so the seed fixes the result, and prints the Pareto front with
/// units as the hall of fame shows them.
fn search(dataset: &Dataset<f64>, options: &Options<f64, Ops, 2>) -> Vec<PopMember<f64, Ops, 2>> {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let result = pool.install(|| equation_search::<f64, Ops, 2>(std::slice::from_ref(dataset), options).unwrap());
    let units = dataset.units.as_ref().unwrap();
    let labels = UnitLabels::new(units, options.dimensionless_constants_only);
    println!("seed {}:", options.seed);
    let front = result.halls_of_fame[0].pareto_front();
    for m in &front {
        let equation = m.expr.string(
            options,
            StringTreeOptions {
                print_precision: Some(5),
                variable_suffixes: Some(&labels.variables),
                constant_suffix: labels.constant,
                ..Default::default()
            },
        );
        println!("  {}\t{:e}\ty{} = {equation}", m.complexity, m.loss, labels.target);
    }
    front
}

fn uniform(n_features: usize, n_rows: usize, seed: u64, scale: f64) -> Array2<f64> {
    let mut rng = fastrand::Rng::with_seed(seed);
    Array2::from_shape_fn((n_features, n_rows), |_| rng.f64() * scale)
}

/// test/integration/ext/dynamicquantities_units/test_units.jl | "Search with dimensional constraints"
#[test]
fn search_with_dimensional_constraints() {
    let x = uniform(1, 128, 0, 20.0);
    let y = Array1::from_iter(x.row(0).iter().map(|&v| v.cos() + v));
    let dataset = Dataset::new(x, y).with_units(Some(&["kg"]), Some("1")).unwrap();
    let options = options(&["sub", "*", "/", "custom_op(x, y) = x + y", "cos"], |l, c| {
        l < 1e-7 && c <= 8
    });
    let front = search(&dataset, &options);
    let best = front
        .iter()
        .find(|m| m.loss < 1e-7)
        .expect("a member with loss below 1e-7");
    // `custom_op(cos(1 * x1), 1 * x1)`: `cos` of a mass needs a constant to cancel the kilograms.
    assert!(
        best.complexity >= 8,
        "{}",
        best.expr.string(&options, Default::default())
    );
    for m in &front {
        let s = m.expr.string(&options, Default::default());
        assert!(!s.contains("cos(x0)"), "`cos` of a bare mass on the front: {s}");
    }
}

/// test/integration/ext/dynamicquantities_units/test_units.jl | "Search with dimensional constraints on output"
///
/// `x1` and `x2` hold the same values, so `x1 * x1` and `x2 * x2` fit `y` equally well; only
/// `x2 * x2` has the target's units.
#[test]
fn search_with_dimensional_constraints_on_output() {
    let x1 = uniform(1, 128, 0, 4.0).mapv(|v| v - 2.0);
    let x = ndarray::concatenate![ndarray::Axis(0), x1, x1];
    let y = Array1::from_iter(x.row(0).iter().map(|&v| v * v));
    let dataset = Dataset::new(x, y).with_units(Some(&["kg", "m"]), Some("m^2")).unwrap();
    let options = options(&["sub", "*", "/", "custom_op(x, y) = x + y", "cos"], |l, c| {
        l < 1e-7 && c == 3
    });
    let front = search(&dataset, &options);
    let best = front
        .iter()
        .find(|m| m.loss < 1e-7)
        .expect("a member with loss below 1e-7");
    assert_eq!(best.expr.string(&options, Default::default()), "x1 * x1");
}

/// The second search of the same testitem: `cbrt` and `sqrt` take kg^3 and kg^2 to kg.
#[test]
fn search_with_dimensional_constraints_on_output_through_roots() {
    let x = uniform(2, 128, 0, 4.0).mapv(|v| v - 2.0);
    let y = Array1::from_iter((0..128).map(|i| x[(0, i)].cbrt() + x[(1, i)].abs().sqrt()));
    let dataset = Dataset::new(x, y)
        .with_units(Some(&["kg^3", "kg^2"]), Some("kg"))
        .unwrap();
    let options = options(&["+", "*", "sqrt", "cbrt", "abs"], |l, c| l < 1e-7 && c == 6);
    let front = search(&dataset, &options);
    let best = front
        .iter()
        .find(|m| m.loss < 1e-7)
        .expect("a member with loss below 1e-7");
    let equation = best.expr.string(&options, Default::default());
    assert_eq!(best.complexity, 6, "{equation}");
    assert!(equation.contains("cbrt(") && equation.contains("sqrt("), "{equation}");
}

/// With `dimensionless_constants_only`, no constant can turn metres into seconds, so the search
/// ends on an equation in the second feature, though both features hold the same values.
#[test]
fn dimensionless_constants_only_picks_the_consistent_equation() {
    let x1 = uniform(1, 128, 1, 4.0).mapv(|v| v - 2.0);
    let x = ndarray::concatenate![ndarray::Axis(0), x1, x1];
    let y = Array1::from_iter(x.row(0).iter().map(|&v| 2.0 * v));
    let dataset = Dataset::new(x, y).with_units(Some(&["m", "s"]), Some("s")).unwrap();
    let options = Options {
        dimensionless_constants_only: true,
        ..options(&["+", "sub", "*", "/"], |l, c| l < 1e-7 && c <= 3)
    };
    let front = search(&dataset, &options);
    let best = front
        .iter()
        .find(|m| m.loss < 1e-7)
        .expect("a member with loss below 1e-7");
    let equation = best.expr.string(&options, Default::default());
    assert!(equation.contains("x1") && !equation.contains("x0"), "{equation}");
}
