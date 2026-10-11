use dynamic_expressions::OperatorSet;
use dynamic_expressions::operator_enum::presets::BuiltinOpsF32;
use fastrand::Rng;
use ndarray::{Array1, Array2};
use symbolic_regression::{
    Batching, Dataset, EarlyStop, MutationWeights, OptimizerAlgorithm, Options, Parallelism, equation_search,
    resume_search,
};

use crate as symbolic_regression;

fn seeded_data(seed: u64) -> Array2<f32> {
    let mut rng = Rng::with_seed(seed);
    Array2::from_shape_fn((5, 100), |_| {
        symbolic_regression::random::standard_normal(&mut rng) as f32
    })
}

fn options() -> Options<f32, BuiltinOpsF32, 2> {
    Options {
        operators: BuiltinOpsF32::new().from_names(["+", "*", "cos"]).unwrap(),
        population_size: 1000,
        populations: 4,
        ncycles_per_iteration: 300,
        niterations: 2,
        tournament_selection_n: 10,
        tournament_selection_p: 1.0,
        topn: 10,
        maxsize: 20,
        parsimony: 0.0,
        alpha: 0.1,
        annealing: true,
        fraction_replaced: 0.1,
        fraction_replaced_hof: 0.1,
        perturbation_factor: 1.0,
        probability_negate_constant: 0.01,
        optimizer_nrestarts: 3,
        optimizer_probability: 0.1,
        optimizer_iterations: 100,
        optimizer_algorithm: OptimizerAlgorithm::NelderMead,
        batch_size: Some(50),
        batching: Batching::Off,
        crossover_probability: 0.0,
        use_frequency: false,
        use_frequency_in_tournament: false,
        skip_mutation_failures: false,
        seed: Some(0),
        progress: false,
        early_stop_condition: Some(EarlyStop::below(1e-6)),
        mutation_weights: MutationWeights {
            mutate_constant: 10.0,
            mutate_operator: 1.0,
            add_node: 1.0,
            insert_node: 3.0,
            delete_node: 3.0,
            simplify: 0.01,
            randomize: 1.0,
            do_nothing: 1.0,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn mixed_configuration(case: usize) {
    let mut options = options();
    let weighted = case == 0 || case == 2;
    if case == 0 {
        options.progress = true;
        options.warmup_maxsize_by = 0.5;
        options.batching = Batching::On;
        options.optimizer_algorithm = OptimizerAlgorithm::BFGS;
        options.tournament_selection_p = 0.8;
        options.parallelism = Parallelism::Serial;
    } else if case == 3 {
        options.crossover_probability = 0.02;
        options.use_frequency_in_tournament = true;
    } else if case == 4 {
        options.crossover_probability = 0.02;
        options.skip_mutation_failures = true;
        options.use_frequency = true;
        options.use_frequency_in_tournament = true;
    }
    let x = seeded_data(0);
    let mut rng = Rng::with_seed(0);
    let weights = weighted.then(|| Array1::from_iter((0..100).map(|_| if rng.f64() > 0.5 { 1.0 } else { 0.0 })));
    let y = Array1::from_iter((0..100).map(|i| {
        let weight = weights.as_ref().map_or(1.0, |w| w[i]);
        2.0 * x[[3, i]].cos() * weight + (1.0 - weight) * 5.0 * x[[1, i]]
    }));
    let dataset = Dataset::with_weights_and_names(x, y, weights, vec![]);
    let result = equation_search(std::slice::from_ref(&dataset), &options).unwrap();
    let front = result.halls_of_fame[0].pareto_front();
    assert!(!front.is_empty());
    let best = front.last().unwrap();
    assert!(best.loss < 2e-2, "case {case}, loss {}", best.loss);
    let test_x = seeded_data(1);
    let test_y = Array1::from_iter((0..100).map(|i| 2.0 * test_x[[3, i]].cos()));
    let test_dataset = Dataset::new(test_x, test_y);
    let mut evaluator = symbolic_regression::pop_member::Evaluator::new(100);
    let test_data = symbolic_regression::dataset::TaggedDataset::new(&test_dataset, None);
    assert!(evaluator.eval(&best.expr, &best.plans, &options, test_data.x_columns()));
    let residual: f32 = evaluator
        .yhat
        .iter()
        .zip(test_dataset.y.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(residual < 2e-2, "case {case}, residual {residual}");
}

#[test]
fn mixed_batched_weighted_serial_warmup_bfgs() {
    mixed_configuration(0);
}
#[test]
fn mixed_multithreading_default_configuration() {
    mixed_configuration(2);
}
#[test]
fn mixed_multithreading_crossover_tournament_frequency() {
    mixed_configuration(3);
}
#[test]
fn mixed_multithreading_crossover_skip_failures_both_frequencies() {
    mixed_configuration(4);
}

#[test]
fn custom_variable_names_and_resumed_losses() {
    let mut options = options();
    options.early_stop_condition = None;
    options.skip_mutation_failures = true;
    let mul = options.operators.set.lookup("*").unwrap();
    let cos = options.operators.set.lookup("cos").unwrap();
    options.op_constraints.set_op_arg_constraint(mul, 1, 10);
    options.op_constraints.set_op_arg_constraint(cos, 0, 5);
    let x = seeded_data(0);
    let y = Array1::from_iter((0..100).map(|i| 2.0 * x[[3, i]].cos() - x[[1, i]]));
    let names = ["t1", "t2", "t3", "t4", "t5"].map(str::to_owned).to_vec();
    let dataset = Dataset::with_weights_and_names(x, y, None, names);
    let result = equation_search(std::slice::from_ref(&dataset), &options).unwrap();
    let best = result.halls_of_fame[0].pareto_front().pop().unwrap();
    assert!(best.loss < 2e-3, "loss {}", best.loss);
    options.niterations = 0;
    let resumed = resume_search(result, std::slice::from_ref(&dataset), &options).unwrap();
    let previous_loss = resumed.halls_of_fame[0].pareto_front().last().unwrap().loss;
    assert!(previous_loss < 2e-3);
    let printed = best.expr.string(
        &options,
        dynamic_expressions::StringTreeOptions {
            variable_names: Some(&dataset.variable_names),
            ..Default::default()
        },
    );
    assert!(printed.contains("t4") && printed.contains("t2"));
    options.loss = symbolic_regression::parse_loss("loss(prediction, target) = 0.1 * (prediction - target)^2").unwrap();
    let rescored = resume_search(resumed, std::slice::from_ref(&dataset), &options).unwrap();
    let new_loss = rescored.halls_of_fame[0].pareto_front().last().unwrap().loss;
    assert!((new_loss - previous_loss * 0.1).abs() <= previous_loss.abs() * 1e-4 + 1e-10);
}
