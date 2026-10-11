//! A search whose only mutation is backsolve rewrites the guess `x0 + 1.0` into an exact fit of
//! `y = 2 x0 + 1`, keeping every member within the size limit. Without constant
//! optimization, the same search without backsolve does not find the fit.

use ndarray::{Array1, Array2};
use symbolic_regression::prelude::*;

fn search(backsolve: f64, do_nothing: f64, collection: bool) -> SearchResult<f64, BuiltinOpsF64, 2> {
    let x = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let y = Array1::from_iter(x.row(0).iter().map(|v| 2.0 * v + 1.0));
    let dataset = Dataset::new(x, y);
    let mut options = Options {
        save_to_file: false,
        operators: BuiltinOpsF64::new().from_names(["+", "*", "sub", "sin"]).unwrap(),
        mutation_weights: MutationWeights {
            mutate_constant: 0.0,
            mutate_operator: 0.0,
            mutate_feature: 0.0,
            swap_operands: 0.0,
            rotate_tree: 0.0,
            add_node: 0.0,
            insert_node: 0.0,
            delete_node: 0.0,
            simplify: 0.0,
            randomize: 0.0,
            do_nothing,
            optimize: 0.0,
            form_connection: 0.0,
            break_connection: 0.0,
            backsolve,
        },
        guesses: vec![vec!["x0 + 1.0".into()]],
        population_size: 20,
        populations: 1,
        ncycles_per_iteration: 20,
        niterations: 1,
        maxsize: 20,
        crossover_probability: 0.0,
        should_optimize_constants: false,
        seed: Some(0),
        deterministic: true,
        progress: false,
        ..Default::default()
    };
    if collection {
        options.default_mutations = Some(vec![]);
        options.mutations = vec![
            (
                BuiltinMutation::Backsolve(symbolic_regression::backsolve::BacksolveOptions {
                    max_terms: 8,
                    ..Default::default()
                }),
                backsolve,
            ),
            (BuiltinMutation::DoNothing, do_nothing),
        ];
    }
    let result = equation_search(std::slice::from_ref(&dataset), &options).unwrap();
    for member in result.populations[0].iter().flat_map(|p| p.members.iter()) {
        let nodes = &member.expr.trees[0].nodes;
        assert!(nodes.len() <= options.maxsize, "complexity {}", nodes.len());
        assert!(!member.loss.is_nan());
    }
    result
}

fn best_loss(result: &SearchResult<f64, BuiltinOpsF64, 2>) -> f64 {
    result.halls_of_fame[0]
        .members()
        .map(|m| m.loss)
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn backsolve_search_fits_exactly() {
    let result = search(1.0, 0.0, false);
    assert!(result.halls_of_fame[0].members().next().is_some());
    assert!(best_loss(&result) < 1e-20, "best loss {:e}", best_loss(&result));

    let control = search(0.0, 1.0, false);
    assert!(
        best_loss(&control) > 1e-3,
        "control best loss {:e}",
        best_loss(&control)
    );
}

#[test]
fn backsolve_collection_search_fits_exactly() {
    let result = search(1.0, 0.0, true);
    assert!(best_loss(&result) < 1e-20);
    let control = search(0.0, 1.0, true);
    assert!(best_loss(&control) > 1e-3);
}
