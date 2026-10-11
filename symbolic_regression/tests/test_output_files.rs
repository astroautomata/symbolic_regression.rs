use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::array;
use symbolic_regression::prelude::*;
use symbolic_regression::{Guess, SearchEngine, equation_search};

static NEXT_DIR: AtomicUsize = AtomicUsize::new(0);
struct TempDir(PathBuf);
impl TempDir {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "sr-outputs-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}
fn options(dir: &TempDir, run_id: &str) -> Options<f64, BuiltinOpsF64, 2> {
    Options {
        operators: BuiltinOpsF64::new().from_names(&["+", "*"]).unwrap(),
        populations: 1,
        population_size: 5,
        tournament_selection_n: 2,
        niterations: 3,
        ncycles_per_iteration: 1,
        maxsize: 20,
        deterministic: true,
        seed: 0,
        should_optimize_constants: false,
        progress: false,
        verbosity: Some(0),
        save_to_file: true,
        output_directory: Some(dir.0.to_str().unwrap().to_string()),
        run_id: Some(run_id.to_string()),
        input_stream: None,
        ..Default::default()
    }
}
fn dataset() -> Dataset<f64> {
    Dataset::with_weights_and_names(
        array![[1.0, 2.0, 3.0, 4.0]],
        array![1.0, 4.0, 9.0, 16.0],
        None,
        vec!["x1".into()],
    )
}
fn check_csv(
    path: PathBuf,
    options: &Options<f64, BuiltinOpsF64, 2>,
    data: &Dataset<f64>,
    hall: &symbolic_regression::HallOfFame<f64, BuiltinOpsF64, 2>,
) {
    let csv = std::fs::read_to_string(&path).unwrap();
    assert_eq!(csv, std::fs::read_to_string(path.with_extension("csv.bak")).unwrap());
    let mut lines = csv.lines();
    // SymbolicRegression.jl src/SearchUtils.jl:849-854 defines PySR's CSV columns and escaping.
    assert_eq!(lines.next().unwrap().as_bytes(), b"Complexity,Loss,Equation");
    let front = hall.pareto_front();
    assert!(!front.is_empty());
    let rows: Vec<_> = lines.collect();
    assert_eq!(rows.len(), front.len());
    for (row, member) in rows.iter().zip(front) {
        let fields: Vec<_> = row.splitn(3, ',').collect();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0].parse::<usize>().unwrap(), member.complexity);
        assert_eq!(fields[1].parse::<f64>().unwrap(), member.loss);
        let equation = member.expr.string(
            options,
            dynamic_expressions::strings::StringTreeOptions {
                variable_names: Some(&data.variable_names),
                julia_constants: true,
                ..Default::default()
            },
        );
        assert_eq!(fields[2], format!("\"{}\"", equation.replace('"', "\"\"")));
    }
}
#[test]
fn seeded_search_writes_pysr_single_and_multi_output_files() {
    let dir = TempDir::new();
    let mut opts = options(&dir, "single");
    opts.guesses = vec![vec![Guess::from("x1*x1")]];
    let data = dataset();
    let result = equation_search(std::slice::from_ref(&data), &opts).unwrap();
    assert_eq!(result.run_id, "single");
    check_csv(
        dir.0.join("single/hall_of_fame.csv"),
        &opts,
        &data,
        &result.halls_of_fame[0],
    );
    opts.run_id = Some("multi".into());
    opts.guesses = vec![vec![Guess::from("x1*x1")], vec![Guess::from("x1*x1")]];
    let result = equation_search(&[data.clone(), data.clone()], &opts).unwrap();
    for output in 1..=2 {
        check_csv(
            dir.0.join(format!("multi/hall_of_fame_output{output}.csv")),
            &opts,
            &data,
            &result.halls_of_fame[output - 1],
        );
    }
    assert!(!dir.0.join("multi/hall_of_fame.csv").exists());
}
#[test]
fn files_appear_after_a_cycle_before_search_completion() {
    let dir = TempDir::new();
    let mut engine = SearchEngine::new(vec![dataset()], options(&dir, "running")).unwrap();
    assert_eq!(engine.run_id(), "running");
    assert_eq!(engine.step(1).unwrap(), 1);
    assert!(!engine.is_finished());
    assert!(dir.0.join("running/hall_of_fame.csv").exists());
    assert!(dir.0.join("running/hall_of_fame.csv.bak").exists());
    engine.run_to_completion().unwrap();
}
#[test]
fn file_saving_with_zero_iterations() {
    let dir = TempDir::new();
    let mut opts = options(&dir, "zero");
    opts.niterations = 0;
    opts.guesses = vec![vec![Guess::from("2.0*x1*x1 + 3.0*x2 + 0.5")]];
    use rand::SeedableRng;
    use rand_distr::Distribution;
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let x = ndarray::Array2::from_shape_fn((2, 30), |_| rand_distr::StandardNormal.sample(&mut rng));
    let y = ndarray::Array1::from_shape_fn(30, |i| 2.0 * x[(0, i)] * x[(0, i)] + 3.0 * x[(1, i)] + 0.5);
    let data = Dataset::with_weights_and_names(x, y, None, vec!["x1".into(), "x2".into()]);
    let result = equation_search(&[data], &opts).unwrap();
    assert!(!result.halls_of_fame[0].pareto_front().is_empty());
    let content = std::fs::read_to_string(dir.0.join("zero/hall_of_fame.csv")).unwrap();
    assert!(content.contains("(((2.0 * x1) * x1) + (3.0 * x2)) + 0.5"));
    assert_eq!(
        content,
        std::fs::read_to_string(dir.0.join("zero/hall_of_fame.csv.bak")).unwrap()
    );
}
#[test]
fn hall_of_fame_csv_escapes_equation_quotes() {
    let dir = TempDir::new();
    let mut opts = options(&dir, "quoted");
    opts.niterations = 0;
    let mut tree = dynamic_expressions::PostfixExpr::<f64, BuiltinOpsF64, 2>::new(
        vec![dynamic_expressions::PNode::Var { feature: 0 }],
        vec![],
        Default::default(),
    );
    tree.meta.variable_names = vec![];
    opts.guesses = vec![vec![Guess::from(symbolic_regression::Expression::from(tree))]];
    let data = Dataset::with_weights_and_names(array![[1.0, 2.0]], array![1.0, 2.0], None, vec!["x\"quoted".into()]);
    equation_search(&[data], &opts).unwrap();
    let csv = std::fs::read_to_string(dir.0.join("quoted/hall_of_fame.csv")).unwrap();
    assert_eq!(csv, "Complexity,Loss,Equation\n1,0.0,\"x\"\"quoted\"\n");
}
#[test]
fn generated_run_id_is_reachable_and_matches_julia_shape() {
    let dir = TempDir::new();
    let mut opts = options(&dir, "unused");
    opts.run_id = None;
    opts.niterations = 0;
    let result = equation_search(&[dataset()], &opts).unwrap();
    let id = &result.run_id;
    assert_eq!(id.len(), 22);
    assert_eq!(&id[8..9], "_");
    assert_eq!(&id[15..16], "_");
    assert!(id[..8].bytes().all(|b| b.is_ascii_digit()));
    assert!(id[9..15].bytes().all(|b| b.is_ascii_digit()));
    assert!(id[16..].bytes().all(|b| b.is_ascii_alphanumeric()));
    assert!(dir.0.join(id).join("hall_of_fame.csv").exists());
}
