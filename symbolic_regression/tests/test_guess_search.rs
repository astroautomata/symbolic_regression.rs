//! Guesses enter each output's hall of fame with their constants optimized before the first cycle,
//! so a search with `niterations = 0` returns them.

use dynamic_expressions::OperatorSet;
use dynamic_expressions::strings::StringTreeOptions;
use ndarray::{Array1, Array2};
use symbolic_regression::PopMember;
use symbolic_regression::prelude::*;

fn data(f: impl Fn(f64, f64) -> f64) -> Dataset<f64> {
    let n_rows = 64usize;
    let mut rng = fastrand::Rng::with_seed(0);
    let x = Array2::from_shape_fn((2, n_rows), |_| rng.f64() * 4.0 - 2.0);
    let y = Array1::from_iter((0..n_rows).map(|i| f(x[(0, i)], x[(1, i)])));
    Dataset::new(x, y)
}

fn seeded<Ops: OperatorSet<T = f64> + Default>(
    operators: Operators<Ops, 2>,
    guesses: Vec<Guess<f64, Ops, 2>>,
) -> Options<f64, Ops, 2> {
    Options {
        operators,
        guesses: vec![guesses],
        niterations: 0,
        populations: 2,
        seed: 0,
        deterministic: true,
        progress: false,
        ..Default::default()
    }
}

fn search<Ops: OperatorSet<T = f64> + Default + Send + Sync>(
    dataset: &Dataset<f64>,
    options: &Options<f64, Ops, 2>,
) -> Result<SearchResult<f64, Ops, 2>, String> {
    equation_search(std::slice::from_ref(dataset), options)
}

/// The hall of fame's only member, which must fit the data exactly with `constants`.
fn assert_exact<Ops: OperatorSet<T = f64>>(result: &SearchResult<f64, Ops, 2>, complexity: usize, constants: &[f64]) {
    let members: Vec<&PopMember<f64, Ops, 2>> = result.halls_of_fame[0].members().collect();
    assert_eq!(
        members.len(),
        1,
        "only the guess enters the hall of fame before a cycle"
    );
    let m = members[0];
    assert_eq!(m.complexity, complexity);
    assert!(m.loss < 1e-12, "loss {:e}", m.loss);
    let got = m.expr.constants();
    assert_eq!(got.len(), constants.len());
    for (g, w) in got.iter().zip(constants) {
        assert!((g - w).abs() < 1e-6, "constants {got:?}, want {constants:?}");
    }
}

#[test]
fn plain_guess_enters_the_hall_of_fame_with_optimized_constants() {
    let dataset = data(|a, b| 2.5 * a * a + 3.0 * b);
    let operators = BuiltinOpsF64::new().from_names(["+", "*"]).unwrap();
    let options = seeded(operators, vec!["1.0*x0*x0 + 1.0*x1".into()]);
    let result = search(&dataset, &options).unwrap();
    assert_exact(&result, 9, &[2.5, 3.0]);
    assert_eq!(result.best[0].as_ref().unwrap().complexity, 9);
}

#[test]
fn template_guess_enters_the_hall_of_fame_with_optimized_constants_and_parameters() {
    let dataset = data(|a, b| 2.5 * a * a + 3.0 * b);
    let ops = BuiltinOpsF64::new();
    let template = TemplateStructure::new("f(a) + p[1] * b", &["f"], None, &["a", "b"], &[("p", 1)], &ops).unwrap();
    let mut options = seeded(
        ops.from_names(["+", "*"]).unwrap(),
        vec![Guess::Template(vec![
            ("f".to_string(), "1.0 * #1 * #1".into()),
            ("p".to_string(), vec![1.0].into()),
        ])],
    );
    options.template = Some(template);
    let result = search(&dataset, &options).unwrap();
    let m = result.halls_of_fame[0].members().next().unwrap();
    let rounded = StringTreeOptions {
        print_precision: Some(6),
        ..Default::default()
    };
    assert_eq!(m.expr.string(&options, rounded), "f = (2.5 * #1) * #1; p = [3]");
    assert_exact(&result, m.complexity, &[2.5, 3.0]);
}

#[test]
fn guess_with_a_dsl_custom_operator_enters_the_hall_of_fame() {
    let dataset = data(|a, b| 1.5 * a * (-b * b).exp());
    let (custom, names) = parse_operator_tokens::<f64, _>(["+", "*", "bump(x, y) = x*exp(-y^2)"]).unwrap();
    let operators = BuiltinOpsF64::new().with_custom(custom).from_names(&names).unwrap();
    let options = seeded(operators, vec!["1.0 * bump(x0, x1)".into()]);
    let result = search(&dataset, &options).unwrap();
    assert_exact(&result, 5, &[1.5]);
}

#[test]
fn guess_above_maxsize_stays_out_of_the_hall_of_fame_with_a_warning() {
    let dataset = data(|a, b| a * b);
    let mut options = seeded(
        BuiltinOpsF64::new().from_names(["+", "*"]).unwrap(),
        vec!["x0*x1 + x0*x1 + x0".into()],
    );
    options.maxsize = 7;
    let result = search(&dataset, &options).unwrap();
    assert_eq!(result.halls_of_fame[0].members().count(), 0);
    assert!(result.best[0].is_none());
    assert_eq!(
        result.warnings,
        ["Guess expression `((x0 * x1) + (x0 * x1)) + x0` has complexity 9 > maxsize (7)."]
    );
}

#[test]
fn invalid_guesses_fail_setup_naming_the_guess() {
    let dataset = data(|a, b| a * b);
    let (custom, names) = parse_operator_tokens::<f64, _>(["+", "*", "bump(x, y) = x*exp(-y^2)"]).unwrap();
    let operators = BuiltinOpsF64::new().with_custom(custom).from_names(&names).unwrap();
    let cases = [
        ("sin(x0)", "`sin` is not one of the search's operators"),
        ("bump(x0)", "the operator `bump` takes 2 arguments, not 1"),
        ("x0 * x2", "`x2` is not a variable; the variables are x0, x1"),
    ];
    for (guess, reason) in cases {
        let options = seeded(operators.clone(), vec![guess.into()]);
        let err = search(&dataset, &options).err().unwrap();
        assert!(err.starts_with(&format!("invalid guess `{guess}`: ")), "{err}");
        assert!(err.contains(reason), "{err}");
    }
}
