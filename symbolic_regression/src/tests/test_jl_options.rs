//! Ports of SymbolicRegression.jl testitems about operator options (see tools/parity_tests.toml).

use dynamic_expressions::dsl::CustomOps;
use dynamic_expressions::expression_algebra::__apply_postfix;
use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
use dynamic_expressions::{LookupError, Metadata, OperatorSet, PNode, PostfixExpr};

use crate::Options;
use crate::check_constraints::check_constraints;

type Ops = BuiltinOpsF64<CustomOps<f64>>;

fn with_max5() -> Ops {
    let mut custom = CustomOps::default();
    custom.define("max5(a, b, c, d, e) = max(a, b, c, d, e)").unwrap();
    BuiltinOpsF64::new().with_custom(custom)
}

fn x<const D: usize>(feature: u16) -> PostfixExpr<f64, Ops, D> {
    PostfixExpr::new(vec![PNode::Var { feature }], vec![], Metadata::default())
}

fn call<const D: usize, const A: usize>(
    ops: &Ops,
    name: &str,
    args: [PostfixExpr<f64, Ops, D>; A],
) -> PostfixExpr<f64, Ops, D> {
    let op = ops.lookup_with_arity(name, A as u8).unwrap();
    __apply_postfix::<f64, Ops, D, A>(op.id, args)
}

/// test/unit/basic/test_options.jl | "Test with_max_degree_from_context"
#[test]
fn max_arity_follows_operator_set() {
    let builtin = BuiltinOpsF64::new();
    let highest_arity = |names: &[&str]| {
        let selected = builtin.from_names::<3, _>(names.iter().copied()).unwrap();
        let nops = [1, 2, 3].map(|arity| selected.nops(arity));
        (nops, (1..=3).rev().find(|&arity| selected.nops(arity) > 0))
    };
    assert_eq!(highest_arity(&["sin", "cos", "+", "*", "sub"]), ([2, 3, 0], Some(2)));
    assert_eq!(highest_arity(&["sin", "cos"]), ([2, 0, 0], Some(1)));
    assert_eq!(
        highest_arity(&["sin", "cos", "+", "*", "sub", "fma", "clamp"]),
        ([2, 3, 2], Some(3))
    );

    assert_eq!(builtin.max_arity(), 3);
    let ops = with_max5();
    assert_eq!(ops.max_arity(), 5);
    assert_eq!(ops.ops_with_arity(5).count(), 1);
    assert_eq!(ops.ops_with_arity(4).count(), 0);
    let options = Options::<f64, Ops, 5> {
        operators: ops.from_names(["sin", "+", "max5"]).unwrap(),
        ..Default::default()
    };
    for op in options.operators.ops_by_arity.iter().flatten() {
        assert!(!options.op_constraints.limits.contains_key(op));
    }
}

/// test/unit/basic/test_options.jl | "Test operator appears in multiple degrees error"
#[test]
fn operator_in_two_arities_needs_an_arity() {
    let ops = BuiltinOpsF64::new();
    match ops.lookup("-") {
        Err(LookupError::Ambiguous { candidates, .. }) => {
            assert_eq!(candidates, vec!["neg".to_string(), "sub".to_string()]);
        }
        other => panic!("expected an ambiguity, got {other:?}"),
    }
    let sub = ops.lookup_with_arity("-", 2).unwrap();
    assert_eq!(ops.name(sub), "sub");

    let operator_set = BuiltinOpsF64::new().with_custom(CustomOps::default());
    let mut options = Options::<f64, Ops, 3> {
        operators: operator_set.from_names(["neg", "sub", "+"]).unwrap(),
        ..Default::default()
    };
    let sub = options.operators.set.lookup_with_arity("-", 2).unwrap();
    options.op_constraints.set_op_arg_constraint(sub, 0, 1);
    let ops = &options.operators.set;
    let neg_of_sum = call(ops, "neg", [x(0) + x(1)]);
    let sub_of_sum = call(ops, "sub", [x(0) + x(1), x(1)]);
    assert!(check_constraints(&neg_of_sum, &options, options.maxsize));
    assert!(!check_constraints(&sub_of_sum, &options, options.maxsize));
}

/// test/unit/basic/test_options.jl | "Test build_constraints with pre-processed vector format"
#[test]
fn default_constraints_cover_every_arity() {
    let operator_set = with_max5();
    let mut options = Options::<f64, Ops, 5> {
        operators: operator_set
            .from_names(["sin", "cos", "+", "*", "sub", "max5"])
            .unwrap(),
        ..Default::default()
    };
    let ops = &options.operators.set;
    let simple = call(ops, "max5", [call(ops, "sin", [x(0)]), x(1), x(0), x(1), x(0)]);
    let nested_last = call(ops, "max5", [x(0), x(1), x(0), x(1), x(0) + x(1)]);
    assert!(check_constraints(&simple, &options, options.maxsize));
    assert!(check_constraints(&nested_last, &options, options.maxsize));

    let max5 = options.operators.set.lookup("max5").unwrap();
    options.op_constraints.set_op_arg_constraint(max5, 4, 1);
    assert!(check_constraints(&simple, &options, options.maxsize));
    assert!(!check_constraints(&nested_last, &options, options.maxsize));
}

#[test]
fn options_optimizer_and_loss_scale() {
    use crate::{LossScale, OptimizerAlgorithm};
    let mut options = Options::<f64, Ops, 2> {
        optimizer_iterations: 16,
        optimizer_f_calls_limit: 100,
        optimizer_x_abstol: 1e-16,
        ..Default::default()
    };
    assert_eq!(options.optimizer_iterations, 16);
    assert_eq!(options.optimizer_f_calls_limit, 100);
    assert_eq!(options.optimizer_x_abstol, 1e-16);
    options.optimizer_iterations = 32;
    options.optimizer_g_calls_limit = 50;
    options.optimizer_f_reltol = 1e-16;
    assert_eq!(options.optimizer_g_calls_limit, 50);
    assert_eq!(options.optimizer_f_reltol, 1e-16);
    assert_eq!(options.optimizer_algorithm, OptimizerAlgorithm::BFGS);
    for (text, expected) in [("log", LossScale::Log), ("linear", LossScale::Linear)] {
        assert_eq!(text.parse::<LossScale>().unwrap(), expected);
    }
    for invalid in ["invalid", "cubic"] {
        assert!(invalid.parse::<LossScale>().is_err());
    }
}

#[test]
fn versioned_default_profiles() {
    use crate::{Batching, DefaultsProfile};
    let legacy = Options::<f64, Ops, 2>::with_defaults(DefaultsProfile::Legacy);
    let v1 = Options::<f64, Ops, 2>::with_defaults(DefaultsProfile::V1);
    let v2 = Options::<f64, Ops, 2>::with_defaults(DefaultsProfile::V2);
    let current = Options::<f64, Ops, 2>::default();
    // SymbolicRegression.jl Options.jl:1284-1389 selects all three profiles.
    assert_eq!(legacy.maxsize, 20);
    assert_eq!(legacy.populations, 15);
    assert_eq!(legacy.population_size, 33);
    assert_eq!(legacy.ncycles_per_iteration, 550);
    assert_eq!(legacy.parsimony, 0.0032);
    assert_eq!(legacy.adaptive_parsimony_scaling, 20.0);
    assert_eq!(legacy.crossover_probability, 0.066);
    assert!(!legacy.annealing);
    assert_eq!(legacy.alpha, 0.1);
    assert_eq!(legacy.perturbation_factor, 0.076);
    assert_eq!(legacy.probability_negate_constant, 0.01);
    assert_eq!(legacy.tournament_selection_n, 12);
    assert_eq!(legacy.tournament_selection_p, 0.86);
    assert_eq!(legacy.fraction_replaced_hof, 0.035);
    assert_eq!(v1.crossover_probability, 0.0259);
    assert_eq!(v2.crossover_probability, 0.20);
    assert_eq!(v1.batching, Batching::Off);
    assert_eq!(v2.batching, Batching::Auto);
    assert_eq!(v1.batch_size, Some(50));
    assert_eq!(v2.batch_size, None);
    macro_rules! unchanged {
        ($($field:ident),*) => { $(assert_eq!(v1.$field, v2.$field); assert_eq!(current.$field, v2.$field);)* };
    }
    unchanged!(
        maxsize,
        maxdepth,
        populations,
        population_size,
        ncycles_per_iteration,
        parsimony,
        warmup_maxsize_by,
        adaptive_parsimony_scaling,
        annealing,
        alpha,
        perturbation_factor,
        probability_negate_constant,
        tournament_selection_n,
        tournament_selection_p,
        fraction_replaced,
        fraction_replaced_hof,
        fraction_replaced_guesses,
        topn
    );
    assert_eq!(current.crossover_probability, v2.crossover_probability);
    assert_eq!(current.batching, v2.batching);
    assert_eq!(current.batch_size, v2.batch_size);
    assert_eq!(v1.default_mutations(), v2.default_mutations());
    assert_eq!(current.default_mutations(), v2.default_mutations());
    // Options.jl:1344,1382-1385; unset depth and simplification resolve at 877-886,1011.
    assert_eq!(current.adaptive_parsimony_scaling, 1040.0);
    assert_eq!(current.maxdepth, None);
    assert_eq!(current.effective_maxdepth(), 30);
    assert_eq!(current.should_simplify, None);
    assert!(current.effective_should_simplify());
}

#[test]
fn automatic_batching_options() {
    use crate::Batching;
    let mut options = Options::<f64, Ops, 2>::default();
    assert!(!options.use_batching(1000));
    assert!(options.use_batching(1001));
    assert_eq!(
        [1000, 1001, 4999, 5000, 49999, 50000].map(|n| options.get_batch_size(n)),
        [1000, 128, 128, 256, 256, 512]
    );
    assert!(!options.batching_required(1000));
    options.batching = Batching::On;
    assert!(options.use_batching(1000));
    assert_eq!(options.get_batch_size(1000), 1000);
    assert!(!options.batching_required(1000));
    options.batching = Batching::Off;
    options.batch_size = Some(64);
    assert!(!options.use_batching(1001));
    assert_eq!(options.get_batch_size(50000), 64);
    options.batching = Batching::On;
    options.batch_size = Some(2000);
    assert_eq!(options.get_batch_size(1000), 1000);
    assert_eq!(options.get_batch_size(5000), 2000);
    assert!(!options.batching_required(1000));
    options.batch_size = Some(0);
    assert!(options.validate().unwrap_err().contains("batch_size"));
}

#[test]
fn mutation_collection_overrides() {
    use crate::{BuiltinMutation as M, ConstantMutation, Crossover};
    let mut options = Options::<f64, Ops, 2> {
        mutations: vec![(M::Constant(ConstantMutation::default()), 0.0)],
        ..Default::default()
    };
    let resolved: Vec<_> = options.resolved_mutations().collect();
    assert_eq!(resolved[0], options.mutations[0]);
    assert_eq!(resolved.len(), options.default_mutations().len());
    assert_eq!(resolved.iter().filter(|(m, _)| matches!(m, M::Constant(_))).count(), 1);
    options.default_mutations = Some(vec![]);
    assert_eq!(options.resolved_mutations().collect::<Vec<_>>(), options.mutations);
    let custom = ConstantMutation {
        perturbation_factor: 0.2,
        probability_negate: 0.3,
    };
    options.mutations = vec![(M::Constant(custom), 0.25)];
    options.default_mutations = Some(vec![
        (M::Constant(ConstantMutation::default()), 0.75),
        (M::Operator, 1.0),
    ]);
    assert_eq!(
        options.resolved_mutations().collect::<Vec<_>>(),
        vec![(M::Constant(custom), 0.25), (M::Operator, 1.0)]
    );
    assert_eq!(
        options.resolved_crossovers().collect::<Vec<_>>(),
        vec![(Crossover::Subtree, 1.0)]
    );
    options.crossovers = vec![(Crossover::Subtree, 0.5)];
    assert_eq!(options.resolved_crossovers().collect::<Vec<_>>(), options.crossovers);
    options.default_crossovers = Some(vec![]);
    options.crossovers.clear();
    assert_eq!(options.resolved_crossovers().count(), 0);
}

#[test]
fn dependent_defaults_and_validation() {
    let mut options = Options::<f64, Ops, 2> {
        maxsize: 42,
        ..Default::default()
    };
    assert_eq!(options.effective_maxdepth(), 42);
    options.maxdepth = Some(7);
    assert_eq!(options.effective_maxdepth(), 7);
    let add = options.operators.set.lookup("+").unwrap();
    options.op_constraints.set_op_arg_constraint(add, 0, 1);
    assert!(!options.effective_should_simplify());
    options.should_simplify = Some(true);
    assert!(options.effective_should_simplify());
    options.maxsize = 3;
    assert!(options.validate().unwrap_err().contains("maxsize"));
    options.maxsize = 4;
    options.warmup_maxsize_by = -0.1;
    assert!(options.validate().unwrap_err().contains("warmup"));
    options.warmup_maxsize_by = 0.0;
    options.tournament_selection_n = options.population_size;
    assert!(options.validate().unwrap_err().contains("tournament"));
    options.tournament_selection_n -= 1;
    assert!(options.validate().is_ok());
}
