//! Seeded searches over template expressions, including parameters indexed by a category column.

use std::sync::Arc;

use ndarray::{Array1, Array2};
use symbolic_regression::PopMember;
use symbolic_regression::prelude::*;

type Ops = BuiltinOpsF64;

fn template(
    combine: &str,
    expressions: &[&str],
    variable_names: &[&str],
    parameters: &[(&str, usize)],
) -> TemplateStructure {
    TemplateStructure::new(combine, expressions, None, variable_names, parameters, &Ops::new()).unwrap()
}

/// Options that stop the search once a member of at most `complexity` reaches loss `tol`.
fn options(
    template: TemplateStructure,
    operators: &[&str],
    niterations: usize,
    maxsize: usize,
    (tol, complexity): (f64, usize),
) -> Options<f64, Ops, 2> {
    Options {
        operators: BuiltinOpsF64::new().from_names(operators).unwrap(),
        seed: 0,
        deterministic: true,
        niterations,
        populations: 8,
        maxsize,
        progress: false,
        early_stop_condition: Some(Arc::new(move |l, c| l <= tol && c <= complexity)),
        template: Some(template),
        ..Default::default()
    }
}

/// Runs the search on a one-thread pool. Populations finish in a thread-dependent order, and the
/// search applies their results in that order, so only a serial search repeats exactly for a seed.
fn search(dataset: &Dataset<f64>, options: &Options<f64, Ops, 2>) -> SearchResult<f64, Ops, 2> {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    pool.install(|| equation_search::<f64, Ops, 2>(dataset, options).unwrap())
}

/// Prints the Pareto front and returns its simplest member with a loss at most `tol`.
fn simplest_below(
    result: &SearchResult<f64, Ops, 2>,
    options: &Options<f64, Ops, 2>,
    tol: f64,
) -> PopMember<f64, Ops, 2> {
    let front = result.hall_of_fame.pareto_front();
    println!("seed {}:", options.seed);
    for m in &front {
        println!(
            "  {}\t{:e}\t{}",
            m.complexity,
            m.loss,
            m.expr.string(options, Default::default())
        );
    }
    front
        .into_iter()
        .find(|m| m.loss <= tol)
        .unwrap_or_else(|| panic!("no hall-of-fame member reached loss {tol:e}"))
}

fn assert_close(got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() < tol, "{got:?} vs {want:?}");
    }
}

#[test]
fn search_recovers_template_combine() {
    let n_rows = 128usize;
    let mut rng = fastrand::Rng::with_seed(0);
    let x = Array2::from_shape_fn((3, n_rows), |_| rng.f64() * 4.0 - 2.0);
    let y = Array1::from_iter((0..n_rows).map(|i| (x[(0, i)] * x[(1, i)]).sin() + x[(2, i)] * x[(2, i)]));
    let options = options(
        template("sin(f(x1, x2)) + g(x3)", &["f", "g"], &["x1", "x2", "x3"], &[]),
        &["+", "sub", "*"],
        40,
        12,
        (1e-12, 6),
    );

    let result = search(&Dataset::new(x, y), &options);
    let best = simplest_below(&result, &options, 1e-12);
    let equation = best.expr.string(&options, Default::default());
    assert!(
        ["f = #1 * #2; g = #1 * #1", "f = #2 * #1; g = #1 * #1"].contains(&equation.as_str()),
        "{equation}"
    );
}

/// `f(x1) + p[class]`: per-class offsets, as SymbolicRegression.jl's former `ParametricExpression`.
#[test]
fn search_recovers_category_offsets() {
    let offsets = [0.5, -1.2, 2.0];
    let n_rows = 120usize;
    let mut rng = fastrand::Rng::with_seed(1);
    let x = Array2::from_shape_fn((2, n_rows), |(f, i)| {
        if f == 0 {
            rng.f64() * 4.0 - 2.0
        } else {
            (i % 3 + 1) as f64
        }
    });
    let y = Array1::from_iter((0..n_rows).map(|i| x[(0, i)] * x[(0, i)] + offsets[i % 3]));
    let options = options(
        template("f(x1) + p[class]", &["f"], &["x1", "class"], &[("p", 3)]),
        &["+", "sub", "*"],
        40,
        10,
        (1e-12, 3),
    );

    let result = search(&Dataset::new(x, y), &options);
    let best = simplest_below(&result, &options, 1e-12);
    assert_eq!(
        best.expr.string(&options, Default::default()).split(';').next(),
        Some("f = #1 * #1")
    );
    assert_close(&best.expr.params, &offsets, 1e-6);
}

/// The template the PySR migration guide gives for a `ParametricExpressionSpec(max_parameters=2)`
/// model. Both vectors reach `f` only as arguments, so `f` decides which one scales `x1`, which
/// shifts the output, and their signs: each learned vector is a true one up to sign, and the fitted
/// model has each class's true slope and offset.
#[test]
fn search_recovers_migration_guide_parameters() {
    let scale = [1.5, -0.5, 2.5];
    let shift = [0.3, -1.0, 2.0];
    let n_rows = 150usize;
    let mut rng = fastrand::Rng::with_seed(2);
    let x = Array2::from_shape_fn((3, n_rows), |(f, i)| {
        if f == 2 {
            (i % 3 + 1) as f64
        } else {
            rng.f64() * 4.0 - 2.0
        }
    });
    let y = Array1::from_iter((0..n_rows).map(|i| x[(0, i)] * scale[i % 3] + x[(1, i)] + shift[i % 3]));
    let options = options(
        template(
            "f(x1, x2, p1[category], p2[category])",
            &["f"],
            &["x1", "x2", "category"],
            &[("p1", 3), ("p2", 3)],
        ),
        &["+", "sub", "*"],
        100,
        12,
        (1e-12, 7),
    );

    let result = search(&Dataset::new(x, y), &options);
    let best = simplest_below(&result, &options, 1e-12);
    assert_eq!(best.complexity, 7);
    let params = &best.expr.params;
    let is_true_vector = |v: &[f64]| {
        [scale, shift].iter().any(|w| {
            [1.0, -1.0]
                .iter()
                .any(|s| v.iter().zip(w).all(|(a, b)| (a - s * b).abs() < 1e-6))
        })
    };
    assert!(params.chunks(3).all(is_true_vector), "{params:?}");

    let probe = Array2::from_shape_fn((3, 6), |(f, j)| match f {
        0 => (j / 3) as f64,
        1 => 0.0,
        _ => (j % 3 + 1) as f64,
    });
    let template = options.template.as_ref().unwrap();
    let (yhat, ok) = template
        .eval(&best.expr.trees, params, &options.operators.set, probe.view())
        .unwrap();
    assert!(ok);
    let slope: Vec<f64> = (0..3).map(|c| yhat[3 + c] - yhat[c]).collect();
    assert_close(&yhat[..3], &shift, 1e-6);
    assert_close(&slope, &scale, 1e-6);
}

/// SymbolicRegression.jl's "search with parametric template expressions": shared and per-class
/// parameters around two learned subexpressions.
#[test]
fn search_with_parametric_template_expression() {
    let n_rows = 200usize;
    let mut rng = fastrand::Rng::with_seed(3);
    let phase = [0.1, 1.5];
    let x = Array2::from_shape_fn((3, n_rows), |(f, _)| {
        if f == 2 {
            (rng.usize(0..2) + 1) as f64
        } else {
            rng.f64() * 4.0 - 2.0
        }
    });
    let y = Array1::from_iter((0..n_rows).map(|i| {
        let class = x[(2, i)] as usize - 1;
        2.0 * (x[(1, i)] + phase[class]).cos() + x[(0, i)] * x[(0, i)] - 2.0
    }));
    let options = options(
        template(
            "p1[1] * cos(f(x2) + p2[class]) + g(x1) - p1[2]",
            &["f", "g"],
            &["x1", "x2", "class"],
            &[("p1", 2), ("p2", 2)],
        ),
        &["+", "sub", "*", "/"],
        300,
        20,
        (1e-5, 9),
    );

    let dataset = Dataset::new(x, y);
    let result = search(&dataset, &options);
    let best = simplest_below(&result, &options, 1e-5);
    let template = options.template.as_ref().unwrap();
    let (yhat, ok) = template
        .eval(
            &best.expr.trees,
            &best.expr.params,
            &options.operators.set,
            dataset.x.view(),
        )
        .unwrap();
    assert!(ok);
    let mse = yhat.iter().zip(&dataset.y).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / n_rows as f64;
    assert!(mse < 1e-5, "mse {mse:e}");
}
