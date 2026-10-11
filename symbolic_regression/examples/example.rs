use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use symbolic_regression::prelude::*;

// Mirrors `SymbolicRegression.jl/example.jl`.

fn main() {
    fn usage() {
        eprintln!("Usage: example [--niterations <n>]");
    }

    let mut args = std::env::args().skip(1);
    let mut niterations: usize = 1_000;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                usage();
                return;
            }
            "-n" | "--niterations" => {
                let value = args.next().unwrap_or_else(|| {
                    eprintln!("Missing value for {arg}");
                    usage();
                    std::process::exit(2);
                });
                niterations = value.parse().unwrap_or_else(|_| {
                    eprintln!("Expected `--niterations` as an integer, got: {value}");
                    std::process::exit(2);
                });
            }
            _ => {
                eprintln!("Unknown arg: {arg}");
                usage();
                std::process::exit(2);
            }
        }
    }

    const N_FEATURES: usize = 5;
    const D: usize = 3;
    let n_rows = 100;

    let mut rng = StdRng::seed_from_u64(0);

    let mut x = Array2::zeros((N_FEATURES, n_rows));
    let mut y = Array1::zeros(n_rows);

    for i in 0..n_rows {
        for j in 0..N_FEATURES {
            x[(j, i)] = rng.random_range(-3.0f32..3.0f32);
        }
        let x1 = x[(1, i)];
        let x4 = x[(4, i)];
        y[i] = 2.0 * x4.cos() + x1 * x1 - 2.0;
    }

    let dataset = Dataset::new(x, y);

    let operators = BuiltinOpsF32::new()
        .from_names(["cos", "exp", "sin", "+", "sub", "*", "/"])
        .unwrap();

    let options = Options::<f32, BuiltinOpsF32, D> {
        operators,
        niterations,
        ..Default::default()
    };

    let result = equation_search::<_, BuiltinOpsF32, D>(std::slice::from_ref(&dataset), &options).unwrap();
    let dominating = result.halls_of_fame[0].pareto_front();

    println!("Final Pareto front:");
    println!("Complexity\tMSE\tEquation");
    for member in dominating {
        let equation = member.expr.string(&options, Default::default());
        println!("{}\t{}\t{}", member.complexity, member.loss, equation);
    }
    // To evaluate the expression, use:
    // let tree = dominating
    // .last()
    // .unwrap()
    // .expr.trees[0]
    // .clone();
    // let _ = eval_tree_array::<f32, BuiltinOpsF32, D>(
    // &tree,
    // &options.operators.set,
    // dataset.x.view(),
    // &EvalOptions::default(),
    // );
}
