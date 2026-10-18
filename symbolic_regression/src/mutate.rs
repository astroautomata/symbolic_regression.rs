use std::ops::AddAssign;

use dynamic_expressions::node::PNode;
use fastrand::Rng;
use num_traits::Float;

use crate::adaptive_parsimony::RunningSearchStatistics;
use crate::check_constraints::check_expression;
use crate::constant_optimization::{OptimizeConstantsCtx, optimize_constants};
use crate::dataset::TaggedDataset;
use crate::expression::Expression;
use crate::loss_functions::loss_to_cost;
use crate::options::{MutationWeights, Options};
use crate::plugins::{MutationEvent, MutationWeightsState};
use crate::pop_member::{Evaluator, PopMember};
use crate::random::{usize_range, usize_range_inclusive};
use crate::selection::weighted_index;
use crate::tracing::StepTrace;
use crate::{backsolve, mutation_functions};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum MutationChoice {
    MutateConstant,
    MutateOperator,
    MutateFeature,
    SwapOperands,
    RotateTree,
    AddNode,
    InsertNode,
    DeleteNode,
    Simplify,
    Randomize,
    DoNothing,
    Optimize,
    Backsolve,
}

impl MutationChoice {
    /// Every kind, in sampling order.
    pub const ALL: [MutationChoice; 13] = [
        MutationChoice::MutateConstant,
        MutationChoice::MutateOperator,
        MutationChoice::MutateFeature,
        MutationChoice::SwapOperands,
        MutationChoice::RotateTree,
        MutationChoice::AddNode,
        MutationChoice::InsertNode,
        MutationChoice::DeleteNode,
        MutationChoice::Simplify,
        MutationChoice::Randomize,
        MutationChoice::DoNothing,
        MutationChoice::Optimize,
        MutationChoice::Backsolve,
    ];

    pub(crate) fn weight_mut(self, weights: &mut MutationWeights) -> &mut f64 {
        match self {
            MutationChoice::MutateConstant => &mut weights.mutate_constant,
            MutationChoice::MutateOperator => &mut weights.mutate_operator,
            MutationChoice::MutateFeature => &mut weights.mutate_feature,
            MutationChoice::SwapOperands => &mut weights.swap_operands,
            MutationChoice::RotateTree => &mut weights.rotate_tree,
            MutationChoice::AddNode => &mut weights.add_node,
            MutationChoice::InsertNode => &mut weights.insert_node,
            MutationChoice::DeleteNode => &mut weights.delete_node,
            MutationChoice::Simplify => &mut weights.simplify,
            MutationChoice::Randomize => &mut weights.randomize,
            MutationChoice::DoNothing => &mut weights.do_nothing,
            MutationChoice::Optimize => &mut weights.optimize,
            MutationChoice::Backsolve => &mut weights.backsolve,
        }
    }
}

impl MutationChoice {
    pub(crate) fn from_builtin(mutation: crate::options::BuiltinMutation) -> Self {
        match mutation {
            crate::options::BuiltinMutation::Constant(_) => Self::MutateConstant,
            crate::options::BuiltinMutation::Operator => Self::MutateOperator,
            crate::options::BuiltinMutation::Feature => Self::MutateFeature,
            crate::options::BuiltinMutation::SwapOperands => Self::SwapOperands,
            crate::options::BuiltinMutation::RotateTree => Self::RotateTree,
            crate::options::BuiltinMutation::AddNode => Self::AddNode,
            crate::options::BuiltinMutation::InsertNode => Self::InsertNode,
            crate::options::BuiltinMutation::DeleteNode => Self::DeleteNode,
            crate::options::BuiltinMutation::Simplify => Self::Simplify,
            crate::options::BuiltinMutation::Randomize => Self::Randomize,
            crate::options::BuiltinMutation::DoNothing => Self::DoNothing,
            crate::options::BuiltinMutation::Optimize => Self::Optimize,
            crate::options::BuiltinMutation::Backsolve(_) => Self::Backsolve,
        }
    }
    fn weight(self, weights: &MutationWeights) -> f64 {
        match self {
            Self::MutateConstant => weights.mutate_constant,
            Self::MutateOperator => weights.mutate_operator,
            Self::MutateFeature => weights.mutate_feature,
            Self::SwapOperands => weights.swap_operands,
            Self::RotateTree => weights.rotate_tree,
            Self::AddNode => weights.add_node,
            Self::InsertNode => weights.insert_node,
            Self::DeleteNode => weights.delete_node,
            Self::Simplify => weights.simplify,
            Self::Randomize => weights.randomize,
            Self::DoNothing => weights.do_nothing,
            Self::Optimize => weights.optimize,
            Self::Backsolve => weights.backsolve,
        }
    }
}

fn configured_weights<T: Float, Ops, const D: usize>(options: &Options<T, Ops, D>) -> MutationWeights {
    if options.mutations.is_empty() && options.default_mutations.is_none() {
        return options.mutation_weights.clone();
    }
    let mut weights = MutationWeights {
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
        do_nothing: 0.0,
        optimize: 0.0,
        backsolve: 0.0,
        form_connection: 0.0,
        break_connection: 0.0,
    };
    for (mutation, weight) in options.resolved_mutations() {
        *MutationChoice::from_builtin(mutation).weight_mut(&mut weights) += weight;
    }
    weights
}

pub struct NextGenerationCtx<'a, T: Float + AddAssign, Ops, const D: usize> {
    pub rng: &'a mut Rng,
    pub dataset: TaggedDataset<'a, T>,
    pub temperature: f64,
    pub curmaxsize: usize,
    pub stats: &'a RunningSearchStatistics,
    pub options: &'a Options<T, Ops, D>,
    pub evaluator: &'a mut Evaluator<T, D>,
    /// The population `member` was drawn from; the backsolve mutation takes library terms from it.
    pub population: &'a [PopMember<T, Ops, D>],
    /// The population's adaptive-mutation-weights statistics, read and updated when
    /// `options.adaptive_mutation_weights` is set.
    pub weights_state: &'a mut MutationWeightsState,
    pub _ops: core::marker::PhantomData<Ops>,
}

pub struct CrossoverCtx<'a, T: Float, Ops, const D: usize> {
    pub rng: &'a mut Rng,
    pub dataset: TaggedDataset<'a, T>,
    pub curmaxsize: usize,
    pub options: &'a Options<T, Ops, D>,
    pub evaluator: &'a mut Evaluator<T, D>,
    pub _ops: core::marker::PhantomData<Ops>,
}

fn count_constants(nodes: &[PNode]) -> usize {
    nodes.iter().filter(|n| matches!(n, PNode::Const { .. })).count()
}

/// SymbolicRegression.jl's `get_contents_for_mutation`: the tree a mutation edits, one of a
/// template's subexpressions at random.
fn tree_to_mutate(rng: &mut Rng, n_trees: usize) -> usize {
    if n_trees == 1 { 0 } else { usize_range(rng, 0..n_trees) }
}

pub fn condition_mutation_weights<T: Float + AddAssign, Ops, const D: usize>(
    weights: &mut MutationWeights,
    member: &PopMember<T, Ops, D>,
    options: &Options<T, Ops, D>,
    curmaxsize: usize,
    nfeatures: usize,
) {
    if options.template.is_some() {
        // SymbolicRegression.jl's template method skips the leaf checks and keeps the
        // mutate-constant weight unscaled.
        if nfeatures <= 1 {
            weights.mutate_feature = 0.0;
        }
        if member.complexity >= curmaxsize {
            weights.add_node = 0.0;
            weights.insert_node = 0.0;
        }
        if !options.effective_should_simplify() {
            weights.simplify = 0.0;
        }
        return;
    }

    let tree = &member.expr.trees[0];
    let tree_is_leaf = tree
        .nodes
        .iter()
        .all(|n| matches!(n, PNode::Var { .. } | PNode::Const { .. }));
    if tree_is_leaf {
        weights.mutate_operator = 0.0;
        weights.swap_operands = 0.0;
        weights.delete_node = 0.0;
        weights.simplify = 0.0;
        if tree.consts.is_empty() {
            weights.optimize = 0.0;
            weights.mutate_constant = 0.0;
        } else {
            weights.mutate_feature = 0.0;
        }
        return;
    }

    if !tree.nodes.iter().any(mutation_functions::is_swappable_op) {
        weights.swap_operands = 0.0;
    }

    let nconst = count_constants(&tree.nodes);
    weights.mutate_constant *= (nconst.min(8) as f64) / 8.0;

    if nfeatures <= 1 {
        weights.mutate_feature = 0.0;
    }

    let complexity = member.complexity;
    if complexity >= curmaxsize {
        weights.add_node = 0.0;
        weights.insert_node = 0.0;
    }

    if !options.effective_should_simplify() {
        weights.simplify = 0.0;
    }
}

pub fn sample_mutation(rng: &mut Rng, weights: &MutationWeights) -> MutationChoice {
    let mut weights = weights.clone();
    let w = MutationChoice::ALL.map(|c| *c.weight_mut(&mut weights));
    MutationChoice::ALL[weighted_index(rng, &w)]
}

enum MutationResult<T: Float + AddAssign, Ops, const D: usize> {
    ProposedExpr { expr: Expression<T, Ops, D>, evals: f64 },
    ProposedMember { member: PopMember<T, Ops, D>, evals: f64 },
    Failed { evals: f64 },
}

struct MutationApplyCtx<'a, 'd, T: Float + AddAssign, Ops, const D: usize> {
    rng: &'a mut Rng,
    member: &'a PopMember<T, Ops, D>,
    expr: Expression<T, Ops, D>,
    dataset: TaggedDataset<'d, T>,
    temperature: f64,
    curmaxsize: usize,
    options: &'a Options<T, Ops, D>,
    evaluator: &'a mut Evaluator<T, D>,
    trace: &'a mut StepTrace,
    population: &'a [PopMember<T, Ops, D>],
    selected_mutation: Option<crate::options::BuiltinMutation>,
}

/// SymbolicRegression.jl's `mutate_constant` for a candidate: a template with parameters mutates
/// its parameters instead of a tree constant half the time, and always when no tree has a
/// constant.
pub(crate) fn mutate_constant<T: Float, Ops, const D: usize>(
    rng: &mut Rng,
    expr: &mut Expression<T, Ops, D>,
    temperature: f64,
    options: &Options<T, Ops, D>,
    mutation: crate::options::ConstantMutation,
) {
    let mutate_tree = expr.params.is_empty() || (expr.trees.iter().any(|t| !t.consts.is_empty()) && rng.bool());
    if mutate_tree {
        let k = tree_to_mutate(rng, expr.trees.len());
        let _ = mutation_functions::mutate_constant_in_place(rng, &mut expr.trees[k], temperature, mutation);
    } else {
        let lengths: Vec<usize> = options
            .template
            .as_ref()
            .expect("only templates have parameters")
            .parameters()
            .iter()
            .map(|(_, n)| *n)
            .collect();
        mutation_functions::mutate_parameters_in_place(rng, &mut expr.params, &lengths, temperature, mutation);
    }
}

impl MutationChoice {
    #[allow(clippy::too_many_arguments)]
    fn apply<T: Float + num_traits::FromPrimitive + num_traits::ToPrimitive + AddAssign, Ops, const D: usize>(
        self,
        ctx: MutationApplyCtx<'_, '_, T, Ops, D>,
    ) -> MutationResult<T, Ops, D>
    where
        Ops: dynamic_expressions::OperatorSet<T = T>,
    {
        let MutationApplyCtx {
            rng,
            member,
            mut expr,
            dataset,
            temperature,
            curmaxsize,
            options,
            evaluator,
            trace,
            population,
            selected_mutation,
        } = ctx;
        let n_features = dataset.n_features;
        let edits_one_tree = !matches!(
            self,
            MutationChoice::MutateConstant
                | MutationChoice::Simplify
                | MutationChoice::DoNothing
                | MutationChoice::Optimize
                | MutationChoice::Backsolve
        );
        let k = if edits_one_tree {
            tree_to_mutate(rng, expr.trees.len())
        } else {
            0
        };
        let local_features = options.tree_features(k, n_features);
        let tree = &mut expr.trees[k];

        match self {
            MutationChoice::MutateConstant => {
                trace.kind = "mutate_constant";
                mutate_constant(
                    rng,
                    &mut expr,
                    temperature,
                    options,
                    match selected_mutation {
                        Some(crate::options::BuiltinMutation::Constant(c)) => c,
                        _ => options.constant_mutation(),
                    },
                );
                MutationResult::ProposedExpr { expr, evals: 0.0 }
            }
            MutationChoice::MutateOperator => {
                trace.kind = "mutate_operator";
                let _ = mutation_functions::mutate_operator_in_place(rng, tree, &options.operators);
                MutationResult::ProposedExpr { expr, evals: 0.0 }
            }
            MutationChoice::MutateFeature => {
                trace.kind = "mutate_feature";
                mutation_functions::mutate_feature_in_place(rng, tree, local_features);
                MutationResult::ProposedExpr { expr, evals: 0.0 }
            }
            MutationChoice::SwapOperands => {
                trace.kind = "swap_operands";
                let _ = mutation_functions::swap_operands_in_place(rng, tree);
                MutationResult::ProposedExpr { expr, evals: 0.0 }
            }
            MutationChoice::RotateTree => {
                trace.kind = "rotate_tree";
                let _ = mutation_functions::rotate_tree_in_place(rng, tree);
                MutationResult::ProposedExpr { expr, evals: 0.0 }
            }
            MutationChoice::AddNode => {
                if rng.bool() {
                    trace.kind = "add_node:append";
                    let _ =
                        mutation_functions::append_random_op_in_place(rng, tree, &options.operators, local_features);
                } else {
                    trace.kind = "add_node:prepend";
                    let _ =
                        mutation_functions::prepend_random_op_in_place(rng, tree, &options.operators, local_features);
                }
                MutationResult::ProposedExpr { expr, evals: 0.0 }
            }
            MutationChoice::InsertNode => {
                trace.kind = "insert_node";
                let _ = mutation_functions::insert_random_op_in_place(rng, tree, &options.operators, local_features);
                MutationResult::ProposedExpr { expr, evals: 0.0 }
            }
            MutationChoice::DeleteNode => {
                trace.kind = "delete_node";
                let _ = mutation_functions::delete_random_op_in_place(rng, tree);
                MutationResult::ProposedExpr { expr, evals: 0.0 }
            }
            MutationChoice::Simplify => {
                trace.kind = "simplify";
                for tree in expr.trees.iter_mut() {
                    let _ = dynamic_expressions::simplify_in_place(tree, &options.operators.set, &evaluator.eval_opts);
                }
                let mut out = PopMember::from_expr(expr, n_features, options);
                out.parent = member.reference;

                // Match the intended behavior (and current SymbolicRegression.jl main):
                // simplify returns immediately and keeps the old loss, but refreshes complexity/cost.
                out.loss = member.loss;
                out.complexity = options.complexity(&out.expr);
                out.cost = loss_to_cost(
                    out.loss,
                    out.complexity,
                    options.parsimony,
                    options.use_baseline,
                    dataset.baseline_loss,
                );

                MutationResult::ProposedMember {
                    member: out,
                    evals: 0.0,
                }
            }
            MutationChoice::Randomize => {
                trace.kind = "randomize";
                // Match SymbolicRegression.jl: sample a *uniform* random size in 1:curmaxsize.
                let max_size = curmaxsize.max(1).min(options.maxsize.max(1));
                let target_size = usize_range_inclusive(rng, 1..=max_size);
                *tree = mutation_functions::random_expr(rng, &options.operators, local_features, target_size);
                MutationResult::ProposedExpr { expr, evals: 0.0 }
            }
            MutationChoice::DoNothing => {
                // Match SymbolicRegression.jl: identity mutation is accepted immediately and keeps
                // the old loss/cost.
                *trace = StepTrace::identity();
                let mut out = PopMember::from_expr(expr, n_features, options);
                out.parent = member.reference;
                out.plans = member.plans.clone();
                out.complexity = member.complexity;
                out.loss = member.loss;
                out.cost = member.cost;
                MutationResult::ProposedMember {
                    member: out,
                    evals: 0.0,
                }
            }
            MutationChoice::Optimize => {
                trace.kind = "optimize";
                let mut out = PopMember::from_expr(expr, n_features, options);

                // Match SymbolicRegression.jl: optimize returns immediately with loss/cost already
                // computed by constant optimization.
                out.plans = member.plans.clone();
                out.complexity = member.complexity;
                out.loss = member.loss;
                out.cost = member.cost;
                out.birth = member.birth;
                out.reference = member.reference;
                out.parent = member.parent;

                let mut grad_ctx = dynamic_expressions::GradContext::new(dataset.n_rows);
                let (_improved, evals) = optimize_constants(
                    rng,
                    &mut out,
                    OptimizeConstantsCtx {
                        dataset,
                        options,
                        evaluator,
                        grad_ctx: &mut grad_ctx,
                    },
                );

                MutationResult::ProposedMember { member: out, evals }
            }
            MutationChoice::Backsolve => {
                trace.kind = "backsolve";
                match backsolve::backsolve_mutation(
                    rng,
                    member,
                    dataset,
                    curmaxsize,
                    options,
                    evaluator,
                    population,
                    match selected_mutation {
                        Some(crate::options::BuiltinMutation::Backsolve(params)) => params,
                        _ => options.backsolve,
                    },
                ) {
                    (Ok(member), evals) => MutationResult::ProposedMember { member, evals },
                    (Err(reason), evals) => {
                        trace.outcome = Some(("reject", reason));
                        MutationResult::Failed { evals }
                    }
                }
            }
        }
    }
}

/// `options.mutation_weights` conditioned on `member`, as SymbolicRegression.jl's
/// `next_generation` does: feature mutations see the features a tree may use, not the dataset's.
pub(crate) fn next_generation_weights<T: Float + AddAssign, Ops, const D: usize>(
    member: &PopMember<T, Ops, D>,
    options: &Options<T, Ops, D>,
    curmaxsize: usize,
    n_features: usize,
) -> MutationWeights
where
    Ops: dynamic_expressions::OperatorSet<T = T>,
{
    let mut weights = configured_weights(options);
    condition_mutation_weights(
        &mut weights,
        member,
        options,
        curmaxsize,
        options.max_features(n_features),
    );
    weights
}

#[cfg(any(test, feature = "bench"))]
pub fn next_generation<
    T: Float + num_traits::FromPrimitive + num_traits::ToPrimitive + AddAssign,
    Ops,
    const D: usize,
>(
    member: &PopMember<T, Ops, D>,
    ctx: NextGenerationCtx<'_, T, Ops, D>,
) -> (PopMember<T, Ops, D>, bool, f64)
where
    Ops: dynamic_expressions::OperatorSet<T = T>,
{
    next_generation_traced(member, ctx, &mut StepTrace::default())
}

/// [`next_generation`], recording the attempt in `trace` as SymbolicRegression.jl's `tmp_trace`.
pub(crate) fn next_generation_traced<
    T: Float + num_traits::FromPrimitive + num_traits::ToPrimitive + AddAssign,
    Ops,
    const D: usize,
>(
    member: &PopMember<T, Ops, D>,
    ctx: NextGenerationCtx<'_, T, Ops, D>,
    trace: &mut StepTrace,
) -> (PopMember<T, Ops, D>, bool, f64)
where
    Ops: dynamic_expressions::OperatorSet<T = T>,
{
    let NextGenerationCtx {
        rng,
        dataset,
        temperature,
        curmaxsize,
        stats,
        options,
        evaluator,
        population,
        weights_state,
        ..
    } = ctx;

    let mut weights = next_generation_weights(member, options, curmaxsize, dataset.n_features);
    if let Some(config) = &options.adaptive_mutation_weights {
        weights_state.condition_weights(config, &mut weights);
    }
    let (choice, selected_mutation) = if options.mutations.is_empty() && options.default_mutations.is_none() {
        (sample_mutation(rng, &weights), None)
    } else {
        let raw = configured_weights(options);
        let conditioned = |(mutation, weight): (crate::options::BuiltinMutation, f64)| {
            let choice = MutationChoice::from_builtin(mutation);
            let base = choice.weight(&raw);
            if base == 0.0 {
                0.0
            } else {
                weight * (choice.weight(&weights) / base)
            }
        };
        let total: f64 = options.resolved_mutations().map(conditioned).sum();
        assert!(
            total.is_finite() && total > 0.0,
            "at least one conditioned mutation weight must be positive"
        );
        let target = (rng.f64() * total).min(total.next_down());
        let mut cumulative = 0.0;
        let selected = options
            .resolved_mutations()
            .find(|&entry| {
                let weight = conditioned(entry);
                if weight <= 0.0 {
                    return false;
                }
                cumulative += weight;
                target < cumulative
            })
            .expect("positive mutation weight");
        (MutationChoice::from_builtin(selected.0), Some(selected.0))
    };

    let (baby, accepted, evals, after) = mutate_and_accept(
        choice,
        member,
        rng,
        dataset,
        temperature,
        curmaxsize,
        stats,
        options,
        evaluator,
        trace,
        population,
        selected_mutation,
    );
    if let Some(config) = &options.adaptive_mutation_weights {
        let event = MutationEvent {
            accepted,
            before_cost: member.cost.to_f64().unwrap_or(f64::INFINITY),
            before_loss: member.loss.to_f64().unwrap_or(f64::INFINITY),
            after,
        };
        weights_state.record(config, choice, event);
    }
    (baby, accepted, evals)
}

/// SymbolicRegression.jl's acceptance probability of a valid child: the product of the
/// annealing multiplier `exp(-(after_cost - before_cost) / (temperature * alpha))` and the
/// adaptive parsimony multiplier, each `1` when its option is off.
pub(crate) fn acceptance_probability<T: Float, Ops, const D: usize>(
    options: &Options<T, Ops, D>,
    stats: &RunningSearchStatistics,
    temperature: f64,
    (before_cost, after_cost): (f64, f64),
    (parent_complexity, child_complexity): (usize, usize),
) -> f64 {
    let mut prob = 1.0f64;
    if options.annealing {
        let delta = after_cost - before_cost;
        prob *= (-delta / (temperature * options.alpha)).exp();
    }
    prob * crate::adaptive_parsimony::mutation_acceptance_multiplier(
        stats,
        options,
        parent_complexity,
        child_complexity,
    )
}

/// A mutation's child, whether it was accepted, its evaluations, and the evaluated child's
/// `(cost, loss)`.
type MutationOutcome<T, Ops, const D: usize> = (PopMember<T, Ops, D>, bool, f64, Option<(f64, f64)>);

/// Applies `choice` to `member` and decides acceptance; a rejected child is the parent's copy.
#[allow(clippy::too_many_arguments)]
fn mutate_and_accept<T, Ops, const D: usize>(
    choice: MutationChoice,
    member: &PopMember<T, Ops, D>,
    rng: &mut Rng,
    dataset: TaggedDataset<'_, T>,
    temperature: f64,
    curmaxsize: usize,
    stats: &RunningSearchStatistics,
    options: &Options<T, Ops, D>,
    evaluator: &mut Evaluator<T, D>,
    trace: &mut StepTrace,
    population: &[PopMember<T, Ops, D>],
    selected_mutation: Option<crate::options::BuiltinMutation>,
) -> MutationOutcome<T, Ops, D>
where
    T: Float + num_traits::FromPrimitive + num_traits::ToPrimitive + AddAssign,
    Ops: dynamic_expressions::OperatorSet<T = T>,
{
    let before_cost = member.cost.to_f64().unwrap_or(f64::INFINITY);
    let n_features = dataset.n_features;
    let parent_copy = || {
        let mut copy = PopMember::from_expr(member.expr.clone(), n_features, options);
        copy.parent = member.reference;
        copy.complexity = member.complexity;
        copy.loss = member.loss;
        copy.cost = member.cost;
        copy
    };

    let max_attempts = 10;
    let mut successful = false;
    let mut tree = member.expr.clone();
    let mut evals = 0.0f64;

    for _ in 0..max_attempts {
        let outcome = choice.apply(MutationApplyCtx {
            rng,
            member,
            expr: member.expr.clone(),
            dataset,
            temperature,
            curmaxsize,
            options,
            evaluator,
            trace: &mut *trace,
            population,
            selected_mutation,
        });
        match outcome {
            MutationResult::ProposedExpr { expr, evals: e } => {
                evals += e;
                if check_expression(&expr, options, curmaxsize) {
                    successful = true;
                    tree = expr;
                    break;
                }
            }
            MutationResult::ProposedMember {
                member: out,
                evals: delta_evals,
            } => {
                evals += delta_evals;
                let after = (
                    out.cost.to_f64().unwrap_or(f64::INFINITY),
                    out.loss.to_f64().unwrap_or(f64::INFINITY),
                );
                return (out, true, evals, Some(after));
            }
            MutationResult::Failed { evals: e } => {
                evals += e;
                break;
            }
        }
    }

    if !successful {
        trace.outcome.get_or_insert(("reject", "failed_constraint_check"));
        return (parent_copy(), false, evals, None);
    }

    let mut baby = PopMember::from_expr(tree, n_features, options);
    baby.parent = member.reference;
    let _ok = baby.evaluate(&dataset, options, evaluator);
    evals += dataset.dataset_fraction();
    let after_cost = baby.cost.to_f64().unwrap_or(f64::INFINITY);
    if after_cost.is_nan() {
        trace.outcome = Some(("reject", "nan_loss"));
        return (parent_copy(), false, evals, None);
    }
    let after = Some((after_cost, baby.loss.to_f64().unwrap_or(f64::INFINITY)));

    let prob = acceptance_probability(
        options,
        stats,
        temperature,
        (before_cost, after_cost),
        (member.complexity, baby.complexity),
    );
    if prob < rng.f64() {
        trace.outcome = Some(("reject", "acceptance"));
        return (parent_copy(), false, evals, after);
    }

    trace.outcome = Some(("accept", "pass"));
    (baby, true, evals, after)
}

/// SymbolicRegression.jl's `crossover_trees`: each parent contributes one tree, a random
/// subexpression of a template, chosen independently; children keep their parent's other trees
/// and parameters.
pub(crate) fn crossover_expressions<T: Float, Ops, const D: usize>(
    rng: &mut Rng,
    a: &Expression<T, Ops, D>,
    b: &Expression<T, Ops, D>,
) -> (Expression<T, Ops, D>, Expression<T, Ops, D>) {
    let ka = tree_to_mutate(rng, a.trees.len());
    let kb = tree_to_mutate(rng, b.trees.len());
    let (ta, tb) = mutation_functions::crossover_trees(rng, &a.trees[ka], &b.trees[kb]);
    let (mut child_a, mut child_b) = (a.clone(), b.clone());
    child_a.trees[ka] = ta;
    child_b.trees[kb] = tb;
    (child_a, child_b)
}

#[cfg(test)]
pub fn crossover_generation<T: Float + AddAssign, Ops, const D: usize>(
    member1: &PopMember<T, Ops, D>,
    member2: &PopMember<T, Ops, D>,
    ctx: CrossoverCtx<'_, T, Ops, D>,
) -> (PopMember<T, Ops, D>, PopMember<T, Ops, D>, bool, f64)
where
    Ops: dynamic_expressions::OperatorSet<T = T>,
{
    crossover_generation_traced(member1, member2, ctx, &mut StepTrace::default())
}

/// [`crossover_generation`], recording the crossover in `trace` as SymbolicRegression.jl does.
pub(crate) fn crossover_generation_traced<T: Float + AddAssign, Ops, const D: usize>(
    member1: &PopMember<T, Ops, D>,
    member2: &PopMember<T, Ops, D>,
    ctx: CrossoverCtx<'_, T, Ops, D>,
    trace: &mut StepTrace,
) -> (PopMember<T, Ops, D>, PopMember<T, Ops, D>, bool, f64)
where
    Ops: dynamic_expressions::OperatorSet<T = T>,
{
    let CrossoverCtx {
        rng,
        dataset,
        curmaxsize,
        options,
        evaluator,
        ..
    } = ctx;

    if !options.resolved_crossovers().any(|(_, weight)| weight > 0.0) {
        panic!("at least one crossover weight must be positive");
    }
    let max_tries = 10;
    let mut tries = 0;
    loop {
        let (c1_expr, c2_expr) = crossover_expressions(rng, &member1.expr, &member2.expr);
        trace.kind = "subtree_crossover";
        tries += 1;
        if check_expression(&c1_expr, options, curmaxsize) && check_expression(&c2_expr, options, curmaxsize) {
            let mut baby1 = PopMember::from_expr(c1_expr, dataset.n_features, options);
            let mut baby2 = PopMember::from_expr(c2_expr, dataset.n_features, options);
            baby1.parent = member1.reference;
            baby2.parent = member2.reference;
            let _ = baby1.evaluate(&dataset, options, evaluator);
            let _ = baby2.evaluate(&dataset, options, evaluator);
            trace.outcome = Some(("accept", "pass"));
            return (baby1, baby2, true, 2.0 * dataset.dataset_fraction());
        }
        if tries >= max_tries {
            trace.outcome = Some(("reject", "failed_constraint_check"));
            let baby1 = member1.clone();
            let baby2 = member2.clone();
            return (baby1, baby2, false, 0.0);
        }
    }
}
