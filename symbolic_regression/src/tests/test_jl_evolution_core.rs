use std::collections::BTreeSet;

use dynamic_expressions::{PNode, PostfixExpr, string_tree};
use fastrand::Rng;
use ndarray::{Array1, Array2};

use super::common::{D, T, TestOps, infix};
use crate::adaptive_parsimony::RunningSearchStatistics;
use crate::dataset::{Dataset, TaggedDataset};
use crate::expression::Expression;
use crate::mutate::{self, CrossoverCtx, NextGenerationCtx};
use crate::operator_library::OperatorLibrary;
use crate::pop_member::{Evaluator, PopMember};
use crate::population::Population;
use crate::{BuiltinMutation, Options, Parallelism, mutation_functions};

fn options() -> Options<T, TestOps, D> {
    Options {
        operators: OperatorLibrary::sr_default::<TestOps, D>(),
        progress: false,
        use_frequency: false,
        use_frequency_in_tournament: false,
        ..Default::default()
    }
}
fn tree(text: &str) -> PostfixExpr<T, TestOps, D> {
    infix(&TestOps::new(), text)
}
fn text(expr: &PostfixExpr<T, TestOps, D>) -> String {
    string_tree(expr, &TestOps::new(), Default::default())
}
fn data(n: usize, features: usize) -> Array2<f64> {
    let mut rng = Rng::with_seed(0);
    Array2::from_shape_fn((features, n), |_| crate::random::standard_normal(&mut rng))
}
fn mutate_member(
    member: &PopMember<T, TestOps, D>,
    dataset: &Dataset<T>,
    options: &Options<T, TestOps, D>,
) -> PopMember<T, TestOps, D> {
    let mut rng = Rng::with_seed(0);
    let mut evaluator = Evaluator::new(dataset.n_rows);
    let statistics = RunningSearchStatistics::new(options.maxsize, 10000);
    let (child, accepted, _) = mutate::next_generation(
        member,
        NextGenerationCtx {
            rng: &mut rng,
            dataset: TaggedDataset::new(dataset, None),
            temperature: 1.0,
            curmaxsize: options.maxsize,
            stats: &statistics,
            options,
            evaluator: &mut evaluator,
            _ops: std::marker::PhantomData,
        },
    );
    assert!(accepted);
    child
}

#[test]
fn optimizable_constants_include_template_parameters() {
    let expr = Expression::from(tree("x1 + 2.0"));
    assert_eq!(expr.constants(), vec![2.0]);
    assert_eq!(expr.n_constants(), 1);
    let mut template = Expression {
        trees: vec![tree("3.0")].into(),
        params: vec![2.0],
    };
    assert_eq!(template.constants(), vec![3.0, 2.0]);
    assert_eq!(template.n_constants(), 2);
    template.set_constants(&[4.0, 5.0]);
    assert_eq!(template.constants(), vec![4.0, 5.0]);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| template.set_constants(&[6.0]))).is_err());
}

#[test]
fn crossover_conserves_nodes_and_swaps_both_operators() {
    let a = tree("cos(x1) + (3.0 + x2)");
    let b = tree("exp(x1 - x2 * x2) + 10.0 * x3");
    let chars = |a: &str, b: &str| {
        let mut chars: Vec<_> = a
            .chars()
            .chain(b.chars())
            .filter(|c| !['(', ')', ' '].contains(c))
            .collect();
        chars.sort_unstable();
        chars
    };
    let expected = chars(&text(&a), &text(&b));
    let mut rng = Rng::with_seed(0);
    let (mut cos_flipped, mut exp_flipped, mut both) = (false, false, false);
    for _ in 0..1000 {
        let (c, d) = mutation_functions::crossover_trees(&mut rng, &a, &b);
        let (c, d) = (text(&c), text(&d));
        cos_flipped |= d.contains("cos");
        exp_flipped |= c.contains("exp");
        if d.contains("cos") && c.contains("exp") {
            both = true;
            assert!(!c.contains("cos"));
            assert!(!d.contains("exp"));
        }
        assert_eq!(chars(&c, &d), expected);
    }
    assert!(cos_flipped && exp_flipped && both);
}

#[test]
fn default_crossover_matches_direct_trees() {
    let options = options();
    let dataset = Dataset::new(data(16, 2), Array1::zeros(16));
    let a = PopMember::from_expr(tree("cos(x1) + x2"), 2, &options);
    let b = PopMember::from_expr(tree("x1 * x2"), 2, &options);
    let mut rng = Rng::with_seed(17);
    let mut evaluator = Evaluator::new(16);
    let (c, d, accepted, evals) = mutate::crossover_generation(
        &a,
        &b,
        CrossoverCtx {
            rng: &mut rng,
            dataset: TaggedDataset::new(&dataset, None),
            curmaxsize: options.maxsize,
            options: &options,
            evaluator: &mut evaluator,
            _ops: std::marker::PhantomData,
        },
    );
    assert!(accepted);
    assert_eq!(evals, 2.0);
    let (direct_c, direct_d) =
        mutation_functions::crossover_trees(&mut Rng::with_seed(17), &a.expr.trees[0], &b.expr.trees[0]);
    assert_eq!(text(&c.expr.trees[0]), text(&direct_c));
    assert_eq!(text(&d.expr.trees[0]), text(&direct_d));
}

#[test]
fn feature_mutation_ranges_and_template_features() {
    let mut rng = Rng::with_seed(0);
    let mut feature = tree("x1");
    mutation_functions::mutate_feature_in_place(&mut rng, &mut feature, 3);
    assert!(matches!(feature.nodes[0], PNode::Var { feature: 1..=2 }));
    let mut one = tree("x1");
    mutation_functions::mutate_feature_in_place(&mut rng, &mut one, 1);
    assert_eq!(one.nodes, tree("x1").nodes);
    let mut constant = tree("1.0");
    mutation_functions::mutate_feature_in_place(&mut rng, &mut constant, 3);
    assert_eq!(constant.consts, vec![1.0]);
    let mut options = options();
    assert_eq!(options.mutation_weights.mutate_feature, 0.1);
    assert_eq!(options.max_features(5), 5);
    assert_eq!(options.max_features(10), 10);
    options.template = Some(
        dynamic_expressions::dsl::TemplateStructure::new(
            "f(x1, x2) + g(x1, x3, x4)",
            &["f", "g"],
            None,
            &["x1", "x2", "x3", "x4"],
            &[],
            &options.operators.set,
        )
        .unwrap(),
    );
    assert_eq!(options.tree_features(0, 4), 2);
    assert_eq!(options.tree_features(1, 4), 3);
}

#[test]
fn optimization_mutation_fits_sine_constants() {
    let mut options = options();
    options.mutations = vec![(BuiltinMutation::Optimize, 1e30)];
    options.optimizer_iterations = 1000;
    let x = data(100, 5);
    let y = Array1::from_iter((0..100).map(|i| (x[[0, i]] * 2.1 + 0.8).sin() + x[[1, i]].powi(2)));
    let dataset = Dataset::new(x, y);
    let mut member = PopMember::from_expr(tree("sin(x1 * 1.9 + 0.2) + x2 * x2"), 5, &options);
    member.evaluate(&TaggedDataset::new(&dataset, None), &options, &mut Evaluator::new(100));
    let child = mutate_member(&member, &dataset, &options);
    let constants = child.expr.constants();
    for k in [0.0, 0.2, 0.5, 1.0] {
        assert!(((constants[0] * k + constants[1]).sin() - (2.1 * k + 0.8_f64).sin()).abs() < 1e-3);
    }
}

#[test]
fn tournament_selection_prefers_low_cost_in_both_orders() {
    let mut options = options();
    options.tournament_selection_n = 10;
    options.tournament_selection_p = 0.999;
    let statistics = RunningSearchStatistics::new(options.maxsize, 500);
    for reverse in [false, true] {
        let pop = Population::new(
            (0..10)
                .map(|i| {
                    let mut member = PopMember::from_expr(tree("x1 * 3.2"), 1, &options);
                    member.cost = if reverse { 1.0 - i as f64 / 9.0 } else { i as f64 / 9.0 };
                    member.loss = 1.0;
                    member
                })
                .collect(),
        );
        let mut rng = Rng::with_seed(0);
        let mean = (0..100)
            .map(|_| crate::selection::best_of_sample(&mut rng, &pop, &statistics, &options).cost)
            .sum::<f64>()
            / 100.0;
        assert!(mean < 0.1);
    }
}

fn rotations(input: &str, count: usize) -> BTreeSet<String> {
    let expr = tree(input);
    let mut rng = Rng::with_seed(0);
    (0..count)
        .map(|_| {
            let mut out = expr.clone();
            mutation_functions::rotate_tree_in_place(&mut rng, &mut out);
            text(&out)
        })
        .collect()
}
fn expressions(inputs: &[&str]) -> BTreeSet<String> {
    inputs.iter().map(|s| text(&tree(s))).collect()
}

#[test]
fn basic_tree_rotation_outputs() {
    assert_eq!(rotations("x1", 100), expressions(&["x1"]));
    assert_eq!(rotations("x1 + x2", 100), expressions(&["x1 + x2"]));
    assert!(rotations("1.5 * x1 + x2", 100).is_subset(&expressions(&[
        "1.5 * (x1 + x2)",
        "(1.5 + x1) * x2",
        "(1.5 + x2) * x1"
    ])));
}

#[test]
fn complex_tree_rotation_outputs() {
    assert_eq!(
        rotations("1.5 * x1 + 2.5 / x3", 300),
        expressions(&[
            "1.5 * (x1 + 2.5 / x3)",
            "(1.5 * x1 + 2.5) / x3",
            "(1.5 + 2.5 / x3) * x1",
            "2.5 / (1.5 * x1 + x3)",
        ])
    );
    assert_eq!(
        rotations("1.5 * exp(x1) + 2.5 / x3", 300),
        expressions(&[
            "(1.5 * exp(x1) + 2.5) / x3",
            "1.5 * (exp(x1) + 2.5 / x3)",
            "exp(1.5 * x1) + 2.5 / x3",
            "(1.5 + 2.5 / x3) * exp(x1)",
            "2.5 / (1.5 * exp(x1) + x3)",
        ])
    );
    assert_eq!(
        rotations("exp(1.5 * x1 + 2.5 / x3)", 500),
        expressions(&[
            "exp(1.5 * x1) + 2.5 / x3",
            "exp(2.5 / (1.5 * x1 + x3))",
            "exp((1.5 * x1 + 2.5) / x3)",
            "exp(1.5 * (x1 + 2.5 / x3))",
            "exp((1.5 + 2.5 / x3) * x1)",
            "1.5 * x1 + exp(2.5 / x3)",
        ])
    );
}

#[test]
fn search_statistics_window_and_normalization() {
    let mut stats = RunningSearchStatistics::new(30, 500);
    for seed in 1..=1000 {
        stats.update_frequencies(Rng::with_seed(seed).usize(1..=10));
    }
    stats.normalize();
    assert_eq!(stats.frequencies.iter().sum::<f64>(), 1030.0);
    assert!((stats.normalized_frequencies.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!(stats.normalized_frequencies[4] > stats.normalized_frequencies[14]);
    stats.move_window();
    stats.normalize();
    assert!((stats.frequencies.iter().sum::<f64>() - 500.0).abs() < 1e-10);
    assert!(
        stats.normalized_frequencies[..5].iter().sum::<f64>() > stats.normalized_frequencies[9..15].iter().sum::<f64>()
    );
    for seed in 1..=500 {
        stats.update_frequencies(Rng::with_seed(seed).usize(10..=15));
    }
    stats.move_window();
    stats.normalize();
    assert!((stats.frequencies.iter().sum::<f64>() - 500.0).abs() < 1e-10);
    assert!(
        stats.normalized_frequencies[..5].iter().sum::<f64>() < stats.normalized_frequencies[9..15].iter().sum::<f64>()
    );
}

#[test]
fn simplify_mutation_refreshes_cost() {
    let mut options = options();
    options.parsimony = 0.5;
    options.should_simplify = Some(true);
    options.default_mutations = Some(vec![]);
    options.mutations = vec![(BuiltinMutation::Simplify, 1.0)];
    let x = data(64, 1);
    let y = x.row(0).to_owned();
    let dataset = Dataset::new(x, y);
    let mut parent = PopMember::from_expr(tree("(x1 + 1.0) + 2.0"), 1, &options);
    parent.evaluate(&TaggedDataset::new(&dataset, None), &options, &mut Evaluator::new(64));
    let child = mutate_member(&parent, &dataset, &options);
    assert!(child.complexity < parent.complexity);
    let expected = |m: &PopMember<T, TestOps, D>| {
        crate::loss_functions::loss_to_cost(m.loss, m.complexity, options.parsimony, options.use_baseline, None)
    };
    assert_eq!(parent.cost, expected(&parent));
    assert_eq!(child.cost, expected(&child));
    assert!(
        (parent.cost - child.cost - (parent.complexity - child.complexity) as f64 * options.parsimony).abs() < 1e-12
    );
}

#[test]
fn migration_inserts_candidate() {
    let options = options();
    let mut rng = Rng::with_seed(0);
    let mut pop = Population::new(
        (0..100)
            .map(|_| {
                PopMember::from_expr(
                    mutation_functions::random_expr(&mut rng, &options.operators, 5, 10),
                    5,
                    &options,
                )
            })
            .collect(),
    );
    let candidate = PopMember::from_expr(tree("1.0 + x2 * 3.2"), 5, &options);
    assert!(
        !pop.members
            .iter()
            .any(|m| text(&m.expr.trees[0]) == text(&candidate.expr.trees[0]))
    );
    crate::migration::migrate_into(&mut pop, std::slice::from_ref(&candidate), 0.5, &mut rng, true);
    assert!(
        pop.members
            .iter()
            .any(|m| text(&m.expr.trees[0]) == text(&candidate.expr.trees[0]))
    );
}

#[test]
fn bottom_k_population_matches_stable_sort() {
    let options = options();
    for n in [1, 5, 20, 50, 100, 1000] {
        for seed in 1..=10 {
            for kind in 0..3 {
                let mut rng = Rng::with_seed(seed);
                let costs: Vec<f64> = (0..n)
                    .map(|_| match kind {
                        0 => (rng.f32() * 2.0 - 1.0) as f64,
                        1 => rng.f64() * 2.0 - 1.0,
                        _ => rng.i32(-10..=10) as f64,
                    })
                    .collect();
                let pop = Population::new(
                    costs
                        .iter()
                        .enumerate()
                        .map(|(i, &cost)| {
                            let mut m = PopMember::from_expr_with_birth(i as u64, tree("x1"), 1, &options);
                            m.cost = cost;
                            m
                        })
                        .collect(),
                );
                let mut expected: Vec<_> = costs.iter().copied().enumerate().collect();
                expected.sort_by(|a, b| a.1.total_cmp(&b.1));
                for k in [1, 2, 3, 5, 10, 20, 50, 100] {
                    if k <= n {
                        let actual = crate::migration::best_sub_pop(&pop, k);
                        assert_eq!(
                            actual.iter().map(|m| (m.birth as usize, m.cost)).collect::<Vec<_>>(),
                            expected[..k]
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn deterministic_search_repeats_best_equation() {
    use dynamic_expressions::operator_enum::presets::BuiltinOpsF32;
    let x = data(1000, 2).mapv(|v| v as f32) * 2.0;
    let y = Array1::from_iter((0..1000).map(|i| 3.0 * x[[1, i]].cos() + x[[0, i]].powi(2) - 2.0));
    let dataset = Dataset::new(x, y);
    let options = Options::<f32, BuiltinOpsF32, 2> {
        operators: BuiltinOpsF32::new().from_names(["+", "sub", "*", "/", "cos"]).unwrap(),
        crossover_probability: 0.0,
        max_evals: 10000,
        deterministic: true,
        seed: 0,
        niterations: 5,
        parallelism: Parallelism::Serial,
        progress: false,
        ..Default::default()
    };
    let results: Vec<_> = (0..2)
        .map(|_| {
            crate::equation_search(std::slice::from_ref(&dataset), &options)
                .unwrap()
                .halls_of_fame[0]
                .pareto_front()
                .last()
                .unwrap()
                .expr
                .string(&options, Default::default())
        })
        .collect();
    assert_eq!(results[0], results[1]);
}

#[test]
fn nelder_mead_uses_affine_simplex_without_gradients() {
    use crate::optim::{EvalBudget, Objective, OptimOptions, nelder_mead_minimize};
    struct Quadratic {
        points: Vec<Vec<f64>>,
    }
    impl Objective for Quadratic {
        fn f_only(&mut self, x: &[f64], budget: &mut EvalBudget) -> Option<f64> {
            budget.f_calls += 1;
            self.points.push(x.to_vec());
            Some((x[0] - 3.0).powi(2) + (x[1] + 2.0).powi(2))
        }
        fn fg(&mut self, _: &[f64], _: &mut [f64], _: &mut EvalBudget) -> Option<f64> {
            panic!("NelderMead does not request gradients")
        }
    }
    let mut objective = Quadratic { points: vec![] };
    let result = nelder_mead_minimize(&[1.0, 2.0], &mut objective, OptimOptions::default()).unwrap();
    assert_eq!(
        &objective.points[..3],
        &[vec![1.0, 2.0], vec![1.525, 2.0], vec![1.0, 3.025]]
    );
    assert!(result.minimum < 1e-6);
    assert!((result.minimizer[0] - 3.0).abs() < 1e-3);
    assert!((result.minimizer[1] + 2.0).abs() < 1e-3);
    assert_eq!(result.f_calls, objective.points.len());
    let limited = nelder_mead_minimize(
        &[1.0, 2.0],
        &mut objective,
        OptimOptions {
            f_calls_limit: 3,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(limited.f_calls, 3);
}

#[test]
fn optimizer_gradient_budget_keeps_the_reported_loss_consistent() {
    use crate::optim::{BackTracking, EvalBudget, Objective, OptimOptions, bfgs_minimize};
    struct Quadratic;
    impl Objective for Quadratic {
        fn f_only(&mut self, x: &[f64], budget: &mut EvalBudget) -> Option<f64> {
            budget.f_calls += 1;
            Some((x[0] - 1.0).powi(2) + 10.0 * (x[1] + 2.0).powi(2))
        }
        fn fg(&mut self, x: &[f64], g: &mut [f64], budget: &mut EvalBudget) -> Option<f64> {
            g[0] = 2.0 * (x[0] - 1.0);
            g[1] = 20.0 * (x[1] + 2.0);
            self.f_only(x, budget)
        }
    }
    let result = bfgs_minimize(
        &[0.0, 0.0],
        &mut Quadratic,
        OptimOptions {
            g_calls_limit: 1,
            ..Default::default()
        },
        BackTracking::default(),
    )
    .unwrap();
    let expected = (result.minimizer[0] - 1.0).powi(2) + 10.0 * (result.minimizer[1] + 2.0).powi(2);
    assert_eq!(result.minimum, expected);
}

#[test]
fn subnormal_mutation_weights_select_the_only_positive_entry() {
    let weights = crate::MutationWeights {
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
        do_nothing: f64::from_bits(1),
        optimize: 0.0,
        form_connection: 0.0,
        break_connection: 0.0,
    };
    for seed in 0..100 {
        assert_eq!(
            mutate::sample_mutation(&mut Rng::with_seed(seed), &weights),
            mutate::MutationChoice::DoNothing
        );
    }
}
