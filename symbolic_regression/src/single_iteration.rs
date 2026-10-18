use std::ops::AddAssign;

use fastrand::Rng;
use num_traits::{Float, FromPrimitive, ToPrimitive};

use crate::adaptive_parsimony::RunningSearchStatistics;
use crate::constant_optimization::{OptimizeConstantsCtx, optimize_constants};
use crate::dataset::TaggedDataset;
use crate::hall_of_fame::HallOfFame;
use crate::options::Options;
use crate::plugins::MutationWeightsState;
use crate::pop_member::{Evaluator, generate_reference};
use crate::population::Population;
use crate::regularized_evolution::{RegEvolCtx, reg_evol_cycle};
use crate::stop_controller::StopController;
use crate::tracing::IterationTrace;

pub struct IterationCtx<'a, T: Float + AddAssign, Ops, const D: usize> {
    pub rng: &'a mut Rng,
    pub full_dataset: TaggedDataset<'a, T>,
    pub curmaxsize: usize,
    pub stats: &'a RunningSearchStatistics,
    pub options: &'a Options<T, Ops, D>,
    pub evaluator: &'a mut Evaluator<T, D>,
    pub grad_ctx: &'a mut dynamic_expressions::GradContext<T, D>,
    pub weights_state: &'a mut MutationWeightsState,
    pub controller: &'a StopController,
    pub trace: Option<&'a mut IterationTrace<T, Ops, D>>,
    pub _ops: core::marker::PhantomData<Ops>,
}

/// SymbolicRegression.jl's `SimulatedAnnealingPlugin` temperature at (0-based) cycle `i` of
/// `ncycles`: `LinRange(1.0, 0.0, ncycles)[i + 1]`, or `1` for a single cycle.
pub(crate) fn annealing_temperature(i: usize, ncycles: usize) -> f64 {
    if ncycles <= 1 {
        return 1.0;
    }
    let t = (i as f64) / ((ncycles - 1) as f64);
    1.0 - t
}

pub fn s_r_cycle<T, Ops, const D: usize>(
    pop: &mut Population<T, Ops, D>,
    ctx: &mut IterationCtx<'_, T, Ops, D>,
    eval_dataset: TaggedDataset<'_, T>,
) -> (f64, HallOfFame<T, Ops, D>)
where
    T: Float + FromPrimitive + ToPrimitive + AddAssign,
    Ops: dynamic_expressions::OperatorSet<T = T>,
{
    let ncycles = ctx.options.ncycles_per_iteration.max(1);
    let mut num_evals = 0.0;
    let mut best_seen = HallOfFame::new(ctx.options.maxsize);

    for i in 0..ncycles {
        if ctx.controller.is_cancelled() {
            break;
        }
        let temperature = if ctx.options.annealing {
            annealing_temperature(i, ncycles)
        } else {
            1.0
        };
        num_evals += reg_evol_cycle(
            pop,
            RegEvolCtx {
                rng: ctx.rng,
                dataset: eval_dataset,
                temperature,
                curmaxsize: ctx.curmaxsize,
                stats: ctx.stats,
                options: ctx.options,
                evaluator: ctx.evaluator,
                weights_state: ctx.weights_state,
                best_seen: &mut best_seen,
                controller: ctx.controller,
                trace: ctx.trace.as_deref_mut(),
                _ops: core::marker::PhantomData,
            },
        );
        for m in &pop.members {
            best_seen.consider_unchecked(m, ctx.options);
        }
    }
    (num_evals, best_seen)
}

pub fn optimize_and_simplify_population<T, Ops, const D: usize>(
    pop: &mut Population<T, Ops, D>,
    ctx: &mut IterationCtx<'_, T, Ops, D>,
    opt_dataset: TaggedDataset<'_, T>,
) -> f64
where
    T: Float + FromPrimitive + ToPrimitive + AddAssign,
    Ops: dynamic_expressions::OperatorSet<T = T>,
{
    let mut num_evals = 0.0;

    if ctx.options.effective_should_simplify() {
        for m in &mut pop.members {
            if ctx.controller.is_cancelled() {
                return num_evals;
            }
            let mut changed = false;
            for tree in m.expr.trees.iter_mut() {
                changed |=
                    dynamic_expressions::simplify_in_place(tree, &ctx.options.operators.set, &ctx.evaluator.eval_opts);
            }
            if changed {
                m.rebuild_plans(ctx.full_dataset.n_features, ctx.options);
            }
        }
    }

    // Which members were constant-optimized, kept only for the trace.
    let mut optimized = vec![false; if ctx.trace.is_some() { pop.len() } else { 0 }];
    if ctx.options.should_optimize_constants && ctx.options.optimizer_probability > 0.0 {
        ctx.evaluator.ensure_n_rows(opt_dataset.n_rows);
        ctx.grad_ctx.n_rows = opt_dataset.n_rows;
        for (j, m) in pop.members.iter_mut().enumerate() {
            if ctx.controller.is_cancelled() {
                return num_evals;
            }
            if ctx.rng.f64() < ctx.options.optimizer_probability {
                let (improved, evals) = optimize_constants(
                    ctx.rng,
                    m,
                    OptimizeConstantsCtx {
                        dataset: opt_dataset,
                        options: ctx.options,
                        evaluator: ctx.evaluator,
                        grad_ctx: ctx.grad_ctx,
                    },
                );
                let _ = improved;
                num_evals += evals;
                if let Some(o) = optimized.get_mut(j) {
                    *o = true;
                }
            }
        }
    }

    // Match SymbolicRegression.jl `finalize_costs`: only re-evaluate on the full dataset when
    // batching is enabled (i.e., members were evolved on a batch and need final losses/costs).
    if ctx.options.batching_required(ctx.full_dataset.n_rows) {
        ctx.evaluator.ensure_n_rows(ctx.full_dataset.n_rows);
        for m in &mut pop.members {
            if ctx.controller.is_cancelled() {
                return num_evals;
            }
            let _ = m.evaluate(&ctx.full_dataset, ctx.options, ctx.evaluator);
            num_evals += 1.0;
        }
    }

    // SymbolicRegression.jl gives every member a new reference once the cycle has tuned it.
    for (j, m) in pop.members.iter_mut().enumerate() {
        let old_ref = m.reference;
        m.parent = old_ref;
        m.reference = generate_reference();
        if let Some(trace) = ctx.trace.as_deref_mut() {
            trace.optimization(m, old_ref, optimized[j]);
        }
    }
    num_evals
}
