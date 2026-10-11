use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
use fastrand::Rng;
use ndarray::{Array1, Array2};
use serde_json::Value;

use super::common::infix;
use crate::adaptive_parsimony::{RunningSearchStatistics, mutation_acceptance_multiplier, tournament_cost_multiplier};
use crate::dataset::TaggedDataset;
use crate::mutate::{self, MutationChoice, NextGenerationCtx};
use crate::mutation_functions::mutate_constant_in_place;
use crate::plugins::{MutationEvent, MutationWeightsState, check_plugins};
use crate::pop_member::{Evaluator, PopMember};
use crate::population::Population;
use crate::regularized_evolution::{RegEvolCtx, reg_evol_cycle};
use crate::single_iteration::{IterationCtx, annealing_temperature, s_r_cycle};
use crate::stop_controller::StopController;
use crate::tracing::{IterationTrace, StepTrace};
use crate::{
    AdaptiveMutationWeights, Dataset, HallOfFame, MutationBurst, MutationWeights, Options, Reward, SearchEngine,
    equation_search,
};

type T = f64;
type Ops = BuiltinOpsF64;
const D: usize = 3;
type Opts = Options<T, Ops, D>;

const C: usize = MutationChoice::MutateConstant as usize;
const O: usize = MutationChoice::MutateOperator as usize;

fn binary_ops() -> crate::Operators<Ops, D> {
    Ops::new().from_names::<D, _>(["+", "*"]).expect("valid opset")
}

fn only(choice: MutationChoice) -> MutationWeights {
    let mut weights = MutationWeights::default();
    for c in MutationChoice::ALL {
        *c.weight_mut(&mut weights) = 0.0;
    }
    *choice.weight_mut(&mut weights) = 1.0;
    weights
}

/// Options without annealing or adaptive parsimony, so every valid child is accepted.
fn plain_options(mutation_weights: MutationWeights) -> Opts {
    Options {
        operators: binary_ops(),
        mutation_weights,
        annealing: false,
        use_frequency: false,
        use_frequency_in_tournament: false,
        crossover_probability: 0.0,
        tournament_selection_n: 1,
        population_size: 2,
        progress: false,
        save_to_file: false,
        deterministic: true,
        ..Default::default()
    }
}

fn dataset(y: impl Fn(f64) -> f64) -> Dataset<T> {
    let x: Vec<T> = (0..16).map(|i| i as T / 4.0).collect();
    let y: Vec<T> = x.iter().map(|&v| y(v)).collect();
    Dataset::new(Array2::from_shape_vec((1, x.len()), x).unwrap(), Array1::from_vec(y))
}

fn member(options: &Opts, data: &Dataset<T>, src: &str) -> PopMember<T, Ops, D> {
    let expr = infix::<Ops, D>(&options.operators.set, src);
    let mut m = PopMember::from_expr_with_birth(0, expr, data.n_features, options);
    m.evaluate(
        &TaggedDataset::new(data, None),
        options,
        &mut Evaluator::new(data.n_rows),
    );
    m
}

fn event(accepted: bool, before_cost: f64, after_cost: f64, before_loss: f64, after_loss: f64) -> MutationEvent {
    MutationEvent {
        accepted,
        before_cost,
        before_loss,
        after: Some((after_cost, after_loss)),
    }
}

fn state_with(kinds: &[MutationChoice]) -> MutationWeightsState {
    MutationWeightsState {
        active: MutationChoice::ALL.map(|c| kinds.contains(&c)),
        ..Default::default()
    }
}

fn assert_close(a: f64, b: f64) {
    assert!((a - b).abs() <= 1e-12 * b.abs().max(1.0), "{a} != {b}");
}

/// One `reg_evol_cycle` over `members`, returning the population, the evaluations, and the
/// adaptive-mutation-weights statistics.
fn run_reg_evol(
    options: &Opts,
    data: &Dataset<T>,
    members: Vec<PopMember<T, Ops, D>>,
    trace: Option<&mut IterationTrace<T, Ops, D>>,
) -> (Population<T, Ops, D>, f64, MutationWeightsState) {
    let mut evaluator = Evaluator::new(data.n_rows);
    let mut rng = Rng::with_seed(0);
    let stats = RunningSearchStatistics::new(options.maxsize, 100_000).fork();
    let controller = StopController::from_options(options);
    let mut weights_state = MutationWeightsState::default();
    let mut best_seen = HallOfFame::new(options.maxsize);
    let mut pop = Population::new(members);
    let evals = reg_evol_cycle(
        &mut pop,
        RegEvolCtx {
            rng: &mut rng,
            dataset: TaggedDataset::new(data, None),
            temperature: 1.0,
            curmaxsize: options.maxsize,
            stats: &stats,
            options,
            evaluator: &mut evaluator,
            weights_state: &mut weights_state,
            best_seen: &mut best_seen,
            controller: &controller,
            trace,
            _ops: core::marker::PhantomData,
        },
    );
    (pop, evals, weights_state)
}

fn engine_options(populations: usize) -> Opts {
    Options {
        operators: binary_ops(),
        populations,
        population_size: 20,
        ncycles_per_iteration: 5,
        niterations: 4,
        progress: false,
        save_to_file: false,
        deterministic: true,
        ..Default::default()
    }
}

#[test]
fn adaptive_mutation_weights_defaults_and_state() {
    let options = Opts::default();
    let config = AdaptiveMutationWeights::default();
    assert_eq!(options.adaptive_mutation_weights, Some(config));
    assert_eq!(config.reward, Reward::Cost);
    assert_eq!(config.adaptation_strength, 0.5);
    assert_eq!((config.smoothing, config.floor), (0.02, 0.05));
    assert_eq!(options.mutation_burst, None);
    assert!(options.annealing && options.use_frequency && options.use_frequency_in_tournament);

    let state = MutationWeightsState::default();
    assert_eq!(state.attempts.len(), MutationChoice::ALL.len());
    assert!(state.active[MutationChoice::Backsolve as usize]);
    assert!(state.multipliers.iter().all(|&m| m == 1.0));
}

#[test]
fn adaptive_mutation_weights_update_sampled_kind_and_normalize() {
    let config = AdaptiveMutationWeights {
        smoothing: 0.2,
        reward: Reward::Loss,
        ..Default::default()
    };
    let mut state = state_with(&[MutationChoice::MutateConstant, MutationChoice::MutateOperator]);
    state.record(&config, MutationChoice::MutateConstant, event(true, 1.0, 0.5, 1.0, 0.5));

    let updated = 0.8 + 0.2 * ((2.0 / 3.0) / (3.0 / 5.0));
    let normalizer = (updated + 1.0) / 2.0;
    assert_close(state.multipliers[C], updated / normalizer);
    assert_close(state.multipliers[O], 1.0 / normalizer);
    assert_close((state.multipliers[C] + state.multipliers[O]) / 2.0, 1.0);
}

#[test]
fn adaptive_mutation_weights_regularize_in_log_space() {
    let mut base = only(MutationChoice::MutateConstant);
    base.mutate_constant = 2.0;
    base.mutate_operator = 3.0;
    let conditioned = |adaptation_strength: f64| {
        let config = AdaptiveMutationWeights {
            adaptation_strength,
            ..Default::default()
        };
        let mut state = MutationWeightsState::default();
        state.multipliers[C] = 4.0;
        state.multipliers[O] = 0.25;
        let mut weights = base.clone();
        state.condition_weights(&config, &mut weights);
        (weights.mutate_constant, weights.mutate_operator)
    };
    let check = |(a, b): (f64, f64), (ea, eb): (f64, f64)| {
        assert_close(a, ea);
        assert_close(b, eb);
    };
    check(conditioned(0.0), (2.0, 3.0));
    check(conditioned(0.25), (2.0 * 2f64.sqrt(), 3.0 / 2f64.sqrt()));
    check(conditioned(0.5), (4.0, 1.5));
    check(conditioned(1.0), (8.0, 0.75));
}

#[test]
fn skipped_mutation_kinds_stay_out_of_adaptive_weights_accounting() {
    let config = AdaptiveMutationWeights::default();
    let mut state = MutationWeightsState::default();
    for skipped in [MutationChoice::Simplify, MutationChoice::DoNothing] {
        state.record(&config, skipped, event(true, 1.0, 0.5, 1.0, 0.5));
    }
    assert!(state.attempts.iter().all(|&a| a == 0.0));

    state.record(&config, MutationChoice::MutateConstant, event(true, 1.0, 0.5, 1.0, 0.5));
    assert_eq!(state.attempts[C], 1.0);
    for skipped in [MutationChoice::Simplify, MutationChoice::DoNothing] {
        assert_eq!(state.attempts[skipped as usize], 0.0);
        assert_eq!(state.multipliers[skipped as usize], 1.0);
    }
}

#[test]
fn adaptive_mutation_weights_reward_the_configured_objective() {
    let improves_cost_only = event(true, 1.0, 0.5, 1.0, 2.0);
    for (reward, successes) in [(Reward::Cost, 1.0), (Reward::Loss, 0.0)] {
        let config = AdaptiveMutationWeights {
            reward,
            ..Default::default()
        };
        let mut state = state_with(&[MutationChoice::MutateConstant]);
        state.record(&config, MutationChoice::MutateConstant, improves_cost_only);
        assert_eq!(state.successes[C], successes, "{reward:?}");
    }
}

#[test]
fn plugin_configurations_validate_parameter_domains() {
    let weights = AdaptiveMutationWeights::default();
    assert!(weights.check().is_ok());
    for bad in [
        AdaptiveMutationWeights {
            smoothing: -0.1,
            ..weights
        },
        AdaptiveMutationWeights {
            smoothing: 1.1,
            ..weights
        },
        AdaptiveMutationWeights { floor: 0.0, ..weights },
        AdaptiveMutationWeights { floor: 1.1, ..weights },
        AdaptiveMutationWeights {
            adaptation_strength: -0.1,
            ..weights
        },
        AdaptiveMutationWeights {
            adaptation_strength: 1.1,
            ..weights
        },
    ] {
        assert!(bad.check().is_err(), "{bad:?}");
    }

    let burst = MutationBurst::default();
    assert!(burst.check().is_ok());
    for bad in [
        MutationBurst {
            retry_attempts: 0,
            ..burst
        },
        MutationBurst {
            compound_probability: -0.1,
            ..burst
        },
        MutationBurst {
            compound_probability: 1.1,
            ..burst
        },
        MutationBurst {
            compound_max_steps: 0,
            ..burst
        },
    ] {
        assert!(bad.check().is_err(), "{bad:?}");
    }

    assert!(check_plugins(&Opts::default()).is_ok());
    for alpha in [0.0, -0.1, f64::INFINITY, f64::NAN] {
        assert!(
            check_plugins(&Opts {
                alpha,
                ..Default::default()
            })
            .is_err(),
            "{alpha}"
        );
    }
    assert!(
        check_plugins(&Opts {
            alpha: 0.0,
            annealing: false,
            ..Default::default()
        })
        .is_ok()
    );
    let invalid = Opts {
        mutation_burst: Some(MutationBurst {
            retry_attempts: 0,
            ..burst
        }),
        save_to_file: false,
        ..Default::default()
    };
    assert!(SearchEngine::new(vec![dataset(|x| x)], invalid).is_err());
}

#[test]
fn annealing_temperature_follows_the_requested_cycle_count() {
    assert_eq!(annealing_temperature(0, 3), 1.0);
    assert_eq!(annealing_temperature(1, 3), 0.5);
    assert_eq!(annealing_temperature(2, 3), 0.0);
    assert_eq!(annealing_temperature(0, 1), 1.0);
    for i in 0..550 {
        // Julia's `LinRange(1.0, 0.0, 550)[i + 1]`.
        let (start, stop, t) = (1.0, 0.0, i as f64 / 549.0);
        let expected = (1.0 - t) * start + t * stop;
        assert_eq!(annealing_temperature(i, 550).to_bits(), expected.to_bits(), "cycle {i}");
    }
}

#[test]
fn annealing_temperature_scales_constant_perturbations() {
    let options = Opts {
        perturbation_factor: 1.0,
        ..plain_options(MutationWeights::default())
    };
    let max_factor = |temperature: f64| {
        let mut rng = Rng::with_seed(0);
        let mut largest: f64 = 0.0;
        for _ in 0..500 {
            let mut expr = infix::<Ops, D>(&options.operators.set, "x1 * 1.0");
            assert!(mutate_constant_in_place(&mut rng, &mut expr, temperature, &options));
            let factor = expr.consts[0].abs();
            largest = largest.max(factor.max(1.0 / factor));
        }
        largest
    };
    // The largest factor is `perturbation_factor * temperature + 1.1`.
    assert!(max_factor(0.0) <= 1.1 + 1e-12);
    assert!(max_factor(0.25) <= 1.35 + 1e-12);
    assert!(max_factor(1.0) > 1.35);
}

#[test]
fn adaptive_mutation_weights_attribute_the_sampled_kind() {
    let config = AdaptiveMutationWeights::default();
    let mut state = MutationWeightsState::default();
    state.record(&config, MutationChoice::MutateOperator, event(true, 1.0, 0.5, 1.0, 0.5));
    for c in MutationChoice::ALL {
        let expected = if c == MutationChoice::MutateOperator { 1.0 } else { 0.0 };
        assert_eq!(state.attempts[c as usize], expected, "{c:?}");
        assert_eq!(state.successes[c as usize], expected, "{c:?}");
    }
}

#[test]
fn mutation_burst_retries_until_accepted() {
    let burst = MutationBurst {
        retry_attempts: 4,
        compound_probability: 0.0,
        compound_max_steps: 1,
    };
    let mut n_calls = 0;
    let (_, accepted, _) = burst.run(&mut Rng::with_seed(0), &0, |_, &parent| {
        n_calls += 1;
        (parent, n_calls >= 3)
    });
    assert!(accepted);
    assert_eq!(n_calls, 3);
}

#[test]
fn mutation_burst_retry_stops_at_budget() {
    let burst = MutationBurst {
        retry_attempts: 4,
        compound_probability: 0.0,
        compound_max_steps: 1,
    };
    let mut n_calls = 0;
    let (_, accepted, _) = burst.run(&mut Rng::with_seed(0), &0, |_, &parent| {
        n_calls += 1;
        (parent, false)
    });
    assert!(!accepted);
    assert_eq!(n_calls, 4);
}

#[test]
fn mutation_burst_compound_chains_on_success() {
    let burst = MutationBurst {
        retry_attempts: 1,
        compound_probability: 1.0,
        compound_max_steps: 3,
    };
    let mut n_calls = 0;
    let (member, accepted, selected) = burst.run(&mut Rng::with_seed(0), &0, |_, &parent| {
        n_calls += 1;
        (parent + 1, true)
    });
    assert!(accepted);
    assert_eq!(n_calls, 3);
    assert_eq!(member, 3);
    assert_eq!(selected, 2);
}

#[test]
fn mutation_burst_compound_stops_on_rejection() {
    let burst = MutationBurst {
        retry_attempts: 1,
        compound_probability: 1.0,
        compound_max_steps: 3,
    };
    let mut n_calls = 0;
    let (_, accepted, _) = burst.run(&mut Rng::with_seed(0), &0, |_, &parent| {
        n_calls += 1;
        (parent, false)
    });
    assert!(!accepted);
    assert_eq!(n_calls, 1);
}

#[test]
fn mutation_burst_runs_every_compound_mutation() {
    let options = Opts {
        mutation_burst: Some(MutationBurst {
            retry_attempts: 1,
            compound_probability: 1.0,
            compound_max_steps: 3,
        }),
        ..plain_options(only(MutationChoice::RotateTree))
    };
    let data = dataset(|_| 0.0);
    let members = vec![member(&options, &data, "x1"), member(&options, &data, "x1")];
    let (_, evals, state) = run_reg_evol(&options, &data, members, None);
    let attempts = 3.0 * options.population_size as f64;
    assert_eq!(state.attempts[MutationChoice::RotateTree as usize], attempts);
    assert_eq!(evals, attempts);
}

#[test]
fn mutation_burst_retries_rejected_mutations() {
    let options = Opts {
        mutation_burst: Some(MutationBurst {
            retry_attempts: 3,
            compound_probability: 0.0,
            compound_max_steps: 1,
        }),
        maxsize: 1,
        skip_mutation_failures: true,
        ..plain_options(only(MutationChoice::InsertNode))
    };
    let data = dataset(|x| x);
    let members = vec![member(&options, &data, "x1"), member(&options, &data, "x1")];
    let births: Vec<u64> = members.iter().map(|m| m.birth).collect();
    let (pop, _, state) = run_reg_evol(&options, &data, members, None);
    assert_eq!(
        state.attempts[MutationChoice::InsertNode as usize],
        3.0 * options.population_size as f64
    );
    assert_eq!(pop.members.iter().map(|m| m.birth).collect::<Vec<_>>(), births);
}

#[test]
fn next_generation_accepts_valid_mutations_without_plugins() {
    let options = Opts {
        adaptive_mutation_weights: None,
        ..plain_options(only(MutationChoice::MutateConstant))
    };
    let data = dataset(|_| 0.0);
    let parent = member(&options, &data, "1.0");
    let (_, accepted, _) = mutate::next_generation(
        &parent,
        NextGenerationCtx {
            rng: &mut Rng::with_seed(0),
            dataset: TaggedDataset::new(&data, None),
            temperature: 1.0,
            curmaxsize: options.maxsize,
            stats: &RunningSearchStatistics::new(options.maxsize, 100_000).fork(),
            options: &options,
            evaluator: &mut Evaluator::new(data.n_rows),
            population: &[],
            weights_state: &mut MutationWeightsState::default(),
            _ops: core::marker::PhantomData,
        },
    );
    assert!(accepted);
}

#[test]
fn reg_evol_cycle_counts_every_burst_attempt() {
    let options = Opts {
        mutation_burst: Some(MutationBurst {
            retry_attempts: 1,
            compound_probability: 1.0,
            compound_max_steps: 2,
        }),
        ..plain_options(only(MutationChoice::RotateTree))
    };
    let data = dataset(|_| 0.0);
    let (_, evals, _) = run_reg_evol(&options, &data, vec![member(&options, &data, "x1")], None);
    assert_eq!(evals, 2.0 * TaggedDataset::new(&data, None).dataset_fraction());
}

#[test]
fn mutation_burst_keeps_accepted_intermediate_members_in_hall_of_fame() {
    let options = Opts {
        mutation_burst: Some(MutationBurst {
            retry_attempts: 1,
            compound_probability: 1.0,
            compound_max_steps: 2,
        }),
        ncycles_per_iteration: 1,
        ..plain_options(only(MutationChoice::AddNode))
    };
    let data = dataset(|x| x);
    let mut pop = Population::new(vec![member(&options, &data, "x1")]);
    let stats = RunningSearchStatistics::new(options.maxsize, 100_000).fork();
    let controller = StopController::from_options(&options);
    let full = TaggedDataset::new(&data, None);
    let mut ctx = IterationCtx {
        rng: &mut Rng::with_seed(0),
        full_dataset: full,
        curmaxsize: options.maxsize,
        stats: &stats,
        options: &options,
        evaluator: &mut Evaluator::new(data.n_rows),
        grad_ctx: &mut dynamic_expressions::GradContext::new(data.n_rows),
        weights_state: &mut MutationWeightsState::default(),
        controller: &controller,
        trace: None,
        _ops: core::marker::PhantomData,
    };
    let (_, best_seen) = s_r_cycle(&mut pop, &mut ctx, full);

    // Each accepted add-node step adds a binary operator and a leaf.
    assert_eq!(pop.members[0].complexity, 5);
    assert!(best_seen.best_by_complexity[3].is_some());
    assert!(best_seen.best_by_complexity[1].is_none());
}

#[test]
fn population_weights_state_persists_across_dispatches() {
    let data = dataset(|x| x * x + x);
    let mut engine = SearchEngine::new(vec![data], engine_options(1)).unwrap();
    let total = |engine: &SearchEngine<T, Ops, D>| engine.weights_state(0, 0).attempts.iter().sum::<f64>();
    assert_eq!(total(&engine), 0.0);
    engine.step(1).unwrap();
    let first = total(&engine);
    assert!(first > 0.0);
    engine.step(1).unwrap();
    assert!(total(&engine) > first);
}

#[test]
fn population_state_differs_from_output_state() {
    let options = engine_options(2);
    let maxsize = options.maxsize;
    let data = dataset(|x| x * x + x);
    let mut engine = SearchEngine::new(vec![data], options).unwrap();
    assert_eq!(
        engine.parsimony_stats(0).frequencies.iter().sum::<f64>(),
        maxsize as f64
    );
    engine.step(2).unwrap();

    let head = engine.parsimony_stats(0);
    assert!(head.frequencies.iter().sum::<f64>() > maxsize as f64);
    assert_close(head.fork().normalized_frequencies.iter().sum(), 1.0);
    assert_ne!(
        engine.weights_state(0, 0).multipliers,
        engine.weights_state(0, 1).multipliers
    );
}

#[test]
fn adaptive_mutation_weights_change_sampled_weights_over_a_seeded_search() {
    let options = engine_options(1);
    let config = options.adaptive_mutation_weights.unwrap();
    let base = options.mutation_weights.clone();
    let mut engine = SearchEngine::new(vec![dataset(|x| x * x + x)], options).unwrap();
    engine.step(2).unwrap();

    let state = engine.weights_state(0, 0);
    let active: Vec<usize> = (0..MutationChoice::ALL.len()).filter(|&i| state.active[i]).collect();
    let mean = active.iter().map(|&i| state.multipliers[i]).sum::<f64>() / active.len() as f64;
    assert_close(mean, 1.0);
    assert!(active.iter().any(|&i| (state.multipliers[i] - 1.0).abs() > 1e-6));
    assert_eq!(state.multipliers[MutationChoice::Simplify as usize], 1.0);

    let mut conditioned = base.clone();
    state.condition_weights(&config, &mut conditioned);
    let mut base = base;
    for c in MutationChoice::ALL {
        let expected = *c.weight_mut(&mut base) * state.multipliers[c as usize].sqrt();
        assert_close(*c.weight_mut(&mut conditioned), expected);
    }
}

#[test]
fn search_with_every_mutation_plugin_returns_a_hall_of_fame() {
    let options = Opts {
        operators: Ops::new()
            .from_names::<D, _>(["+", "sub", "*", "/", "sin", "cos"])
            .unwrap(),
        populations: 4,
        population_size: 20,
        ncycles_per_iteration: 20,
        niterations: 3,
        progress: false,
        save_to_file: false,
        deterministic: true,
        adaptive_mutation_weights: Some(AdaptiveMutationWeights {
            smoothing: 0.02,
            floor: 0.05,
            ..Default::default()
        }),
        mutation_burst: Some(MutationBurst::default()),
        ..Default::default()
    };
    let result = equation_search(&[dataset(|x| 2.0 * x + x.sin())], &options).unwrap();
    assert!(result.halls_of_fame[0].members().next().is_some());
}

#[test]
fn annealing_acceptance_probability_matches_formula() {
    let options = Opts {
        alpha: 0.5,
        use_frequency: false,
        ..Default::default()
    };
    let stats = RunningSearchStatistics::new(options.maxsize, 100_000).fork();
    let prob = mutate::acceptance_probability(&options, &stats, 0.4, (1.0, 1.3), (3, 5));
    assert_close(prob, (-(1.3f64 - 1.0) / (0.4 * 0.5)).exp());
    assert!(mutate::acceptance_probability(&options, &stats, 0.4, (1.3, 1.0), (3, 5)) > 1.0);

    let off = Opts {
        annealing: false,
        ..options.clone()
    };
    assert_eq!(
        mutate::acceptance_probability(&off, &stats, 0.4, (1.0, 1.3), (3, 5)),
        1.0
    );

    let mut frequent_small = RunningSearchStatistics::new(options.maxsize, 100_000);
    for _ in 0..10 {
        frequent_small.update_frequencies(3);
    }
    let frequent_small = frequent_small.fork();
    let with_frequency = Opts {
        use_frequency: true,
        ..options.clone()
    };
    let parsimony = frequent_small.freq(3) / frequent_small.freq(5);
    assert_close(
        mutate::acceptance_probability(&with_frequency, &frequent_small, 0.4, (1.0, 1.3), (3, 5)),
        (-(1.3f64 - 1.0) / (0.4 * 0.5)).exp() * parsimony,
    );
}

#[test]
fn annealing_changes_seeded_acceptance_counts() {
    let data = dataset(|x| 2.0 * x);
    let count_accepted = |annealing: bool| {
        let options = Opts {
            annealing,
            ..plain_options(MutationWeights::default())
        };
        let parent = member(&options, &data, "x1 + x1");
        let stats = RunningSearchStatistics::new(options.maxsize, 100_000).fork();
        let mut rng = Rng::with_seed(0);
        let mut evaluator = Evaluator::new(data.n_rows);
        let mut weights_state = MutationWeightsState::default();
        (0..300)
            .filter(|_| {
                mutate::next_generation(
                    &parent,
                    NextGenerationCtx {
                        rng: &mut rng,
                        dataset: TaggedDataset::new(&data, None),
                        temperature: 0.1,
                        curmaxsize: options.maxsize,
                        stats: &stats,
                        options: &options,
                        evaluator: &mut evaluator,
                        population: &[],
                        weights_state: &mut weights_state,
                        _ops: core::marker::PhantomData,
                    },
                )
                .1
            })
            .count()
    };
    let with_annealing = count_accepted(true);
    let without = count_accepted(false);
    assert!(with_annealing < without, "{with_annealing} vs {without}");
}

#[test]
fn adaptive_parsimony_options_toggle_each_hook() {
    let options = Opts::default();
    assert!(options.use_frequency && options.use_frequency_in_tournament);

    let mut stats = RunningSearchStatistics::new(options.maxsize, 100_000);
    for _ in 0..10 {
        stats.update_frequencies(1);
    }
    let stats = stats.fork();
    for (use_frequency, use_frequency_in_tournament) in [(false, true), (true, false), (false, false)] {
        let o = Opts {
            use_frequency,
            use_frequency_in_tournament,
            ..Default::default()
        };
        assert_eq!(
            tournament_cost_multiplier(&stats, &o, 1) > 1.0,
            use_frequency_in_tournament
        );
        assert_eq!(mutation_acceptance_multiplier(&stats, &o, 1, 3) > 1.0, use_frequency);
    }
}

#[test]
fn adaptive_mutation_weights_stay_enabled_without_the_other_plugins() {
    let options = Opts {
        use_frequency: false,
        use_frequency_in_tournament: false,
        annealing: false,
        ..Default::default()
    };
    assert_eq!(
        options.adaptive_mutation_weights,
        Some(AdaptiveMutationWeights::default())
    );
}

#[test]
fn adaptive_parsimony_hooks_match_the_legacy_formula() {
    let options = Opts {
        adaptive_parsimony_scaling: 30.0,
        maxsize: 20,
        ..Default::default()
    };
    let mut head = RunningSearchStatistics::new(options.maxsize, 100_000);
    for size in [1, 1, 1, 1, 1, 3, 3, 5] {
        head.update_frequencies(size);
    }

    let worker = head.fork();
    assert_close(worker.normalized_frequencies.iter().sum(), 1.0);
    let old_frequency = worker.normalized_frequencies[1];
    for _ in 0..100 {
        head.update_frequencies(2);
    }
    assert_eq!(worker.normalized_frequencies[1], old_frequency);
    let refreshed = head.fork();
    assert!(refreshed.normalized_frequencies[1] > old_frequency);

    let mult1 = tournament_cost_multiplier(&worker, &options, 1);
    let mult5 = tournament_cost_multiplier(&worker, &options, 5);
    assert!(mult1 > mult5);
    assert!(mult1 > 1.0);
    assert_close(mult1, (30.0 * worker.normalized_frequencies[0]).exp());
    assert!(mutation_acceptance_multiplier(&worker, &options, 1, 5) > 1.0);

    let tournament_off = Opts {
        use_frequency_in_tournament: false,
        ..options.clone()
    };
    let mutation_off = Opts {
        use_frequency: false,
        ..options.clone()
    };
    assert_eq!(tournament_cost_multiplier(&worker, &tournament_off, 1), 1.0);
    assert_eq!(mutation_acceptance_multiplier(&worker, &mutation_off, 1, 5), 1.0);
    assert_eq!(tournament_cost_multiplier(&worker, &options, 31), 1.0);
}

#[test]
fn adaptive_parsimony_search_runs_end_to_end() {
    let data = dataset(|x| 2.0 * x + 1.0);
    for use_frequency in [true, false] {
        let options = Opts {
            operators: Ops::new().from_names::<D, _>(["+", "*", "cos"]).unwrap(),
            populations: 2,
            niterations: 2,
            ncycles_per_iteration: 20,
            progress: false,
            save_to_file: false,
            use_frequency,
            ..Default::default()
        };
        let result = equation_search(std::slice::from_ref(&data), &options).unwrap();
        assert!(result.best[0].as_ref().unwrap().loss.is_finite());
    }
}

#[test]
fn adaptive_parsimony_state_is_per_output() {
    let options = engine_options(1);
    let maxsize = options.maxsize;
    let mut engine = SearchEngine::new(vec![dataset(|x| x * x), dataset(|x| x + 1.0)], options).unwrap();
    engine.step(2).unwrap();
    let (first, second) = (engine.parsimony_stats(0), engine.parsimony_stats(1));
    assert_eq!(first.frequencies.len(), maxsize);
    assert_eq!(second.frequencies.len(), maxsize);
    assert_eq!(first.fork().normalized_frequencies.len(), maxsize);
    assert_ne!(first.frequencies, second.frequencies);
}

/// The `"mutate"` and `"death"` events of a trace, as `(parent ref, event)` pairs.
fn trace_events(trace: &IterationTrace<T, Ops, D>, options: &Opts, kind: &str) -> (Value, Vec<(String, Value)>) {
    let record: Value = serde_json::from_str(&trace.to_json(options).replace(":Infinity", ":\"Infinity\"")).unwrap();
    let mut events = Vec::new();
    for (reference, entry) in record["mutations"].as_object().unwrap() {
        for event in entry["events"].as_array().unwrap() {
            if event["type"] == kind {
                events.push((reference.clone(), event.clone()));
            }
        }
    }
    (record, events)
}

#[test]
fn mutation_burst_traces_every_compound_mutation() {
    let options = Opts {
        mutation_burst: Some(MutationBurst {
            retry_attempts: 1,
            compound_probability: 1.0,
            compound_max_steps: 3,
        }),
        use_tracing: true,
        ..plain_options(only(MutationChoice::DoNothing))
    };
    let data = dataset(|_| 0.0);
    let pop = Population::new(vec![member(&options, &data, "x1"), member(&options, &data, "x1")]);
    let mut trace = IterationTrace::start(1, 1, 0, &pop);
    run_reg_evol(&options, &data, pop.members, Some(&mut trace));

    let (record, events) = trace_events(&trace, &options, "mutate");
    assert_eq!(events.len(), 3 * options.population_size);
    let selected = events.iter().filter(|(_, e)| e["selected"] == true).count();
    assert_eq!(selected, options.population_size);
    for (parent, event) in &events {
        let child = &record["mutations"][event["child"].to_string()];
        assert_eq!(child["parent"].to_string(), *parent);
        assert_eq!(event["mutation"]["type"], "identity");
    }
}

#[test]
fn trace_captures_rejected_retry_branches() {
    let options = Opts {
        mutation_burst: Some(MutationBurst {
            retry_attempts: 3,
            compound_probability: 0.0,
            compound_max_steps: 1,
        }),
        maxsize: 1,
        skip_mutation_failures: true,
        use_tracing: true,
        ..plain_options(only(MutationChoice::InsertNode))
    };
    let data = dataset(|x| x);
    let pop = Population::new(vec![member(&options, &data, "x1"), member(&options, &data, "x1")]);
    let mut trace = IterationTrace::start(1, 1, 0, &pop);
    run_reg_evol(&options, &data, pop.members, Some(&mut trace));

    let (_, mutations) = trace_events(&trace, &options, "mutate");
    assert_eq!(mutations.len(), 3 * options.population_size);
    let selected = mutations.iter().filter(|(_, e)| e["selected"] == true).count();
    assert_eq!(selected, options.population_size);
    assert!(mutations.iter().all(|(_, e)| e["mutation"]["result"] == "reject"));
    assert!(trace_events(&trace, &options, "death").1.is_empty());
}

#[test]
fn trace_captures_accepted_mutations() {
    let options = Opts {
        adaptive_mutation_weights: None,
        use_tracing: true,
        ..plain_options(only(MutationChoice::MutateConstant))
    };
    let data = dataset(|_| 0.0);
    let parent = member(&options, &data, "1.0");
    let mut step_trace = StepTrace::default();
    let (_, accepted, _) = mutate::next_generation_traced(
        &parent,
        NextGenerationCtx {
            rng: &mut Rng::with_seed(0),
            dataset: TaggedDataset::new(&data, None),
            temperature: 1.0,
            curmaxsize: options.maxsize,
            stats: &RunningSearchStatistics::new(options.maxsize, 100_000).fork(),
            options: &options,
            evaluator: &mut Evaluator::new(data.n_rows),
            population: &[],
            weights_state: &mut MutationWeightsState::default(),
            _ops: core::marker::PhantomData,
        },
        &mut step_trace,
    );
    assert!(accepted);
    assert_eq!(step_trace.outcome, Some(("accept", "pass")));
}

#[test]
fn accepted_crossover_children_enter_the_cycle_hall_of_fame() {
    let options = Opts {
        crossover_probability: 1.0,
        tournament_selection_n: 2,
        ncycles_per_iteration: 1,
        ..plain_options(MutationWeights::default())
    };
    let data = dataset(|x| x);
    let mut pop = Population::new(vec![
        member(&options, &data, "x1 + 1.0"),
        member(&options, &data, "x1 * x1"),
    ]);
    let initial: Vec<i64> = pop.members.iter().map(|m| m.reference).collect();
    let stats = RunningSearchStatistics::new(options.maxsize, 100_000).fork();
    let controller = StopController::from_options(&options);
    let mut best_seen = HallOfFame::new(options.maxsize);
    let mut children = Vec::new();
    let mut rng = Rng::with_seed(0);
    for _ in 0..20 {
        reg_evol_cycle(
            &mut pop,
            RegEvolCtx {
                rng: &mut rng,
                dataset: TaggedDataset::new(&data, None),
                temperature: 1.0,
                curmaxsize: options.maxsize,
                stats: &stats,
                options: &options,
                evaluator: &mut Evaluator::new(data.n_rows),
                weights_state: &mut MutationWeightsState::default(),
                best_seen: &mut best_seen,
                controller: &controller,
                trace: None,
                _ops: core::marker::PhantomData,
            },
        );
        children.extend(pop.members.iter().filter(|m| !initial.contains(&m.reference)).cloned());
    }
    // Every crossover child that entered the population also entered the cycle's hall of fame.
    let mut expected = HallOfFame::new(options.maxsize);
    expected.update_from_members(&children, &options);
    for (c, best) in expected.best_by_complexity.iter().enumerate() {
        if let Some(best) = best {
            let seen = best_seen.best_by_complexity[c].as_ref().expect("complexity seen");
            assert!(seen.cost <= best.cost, "complexity {c}");
        }
    }
    assert!(!children.is_empty());
}
