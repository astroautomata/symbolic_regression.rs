use dynamic_expressions::dsl::CustomOps;
use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
use ndarray::{Array1, Array2};
use symbolic_regression::{Dataset, Options, Parallelism, equation_search, hall_of_fame_rows};

fn main() {
    let x = Array2::from_shape_fn((2, 64), |(feature, i)| {
        if feature == 0 {
            (i as f64 - 32.0) / 16.0
        } else {
            ((i * 13 % 64) as f64 - 32.0) / 16.0
        }
    });
    let y = Array1::from_iter((0..64).map(|i| x[(0, i)] * x[(0, i)] + x[(1, i)]));
    let dataset = Dataset::new(x, y);
    let options = Options::<f64, BuiltinOpsF64<CustomOps<f64>>, 3> {
        operators: BuiltinOpsF64::new()
            .with_custom(CustomOps::default())
            .from_names(["+", "*"])
            .unwrap(),
        seed: Some(17),
        deterministic: true,
        parallelism: Parallelism::Serial,
        populations: 2,
        population_size: 24,
        tournament_selection_n: 4,
        ncycles_per_iteration: 64,
        niterations: 8,
        maxsize: 10,
        progress: false,
        verbosity: Some(0),
        save_to_file: false,
        input_stream: None,
        run_id: Some("python-parity".to_string()),
        ..Default::default()
    };
    let result = equation_search(std::slice::from_ref(&dataset), &options).unwrap();
    let rows = hall_of_fame_rows(&result.halls_of_fame[0], &dataset, &options);
    print!("{{\"pareto_fronts\":[[");
    for (index, (complexity, loss, equation)) in rows.iter().enumerate() {
        if index != 0 {
            print!(",");
        }
        print!("[{complexity},\"{:016x}\",{equation:?}]", loss.to_bits());
    }
    println!(
        "]],\"total_evals\":{},\"run_id\":{:?}}}",
        result.total_evals, result.run_id
    );
}
