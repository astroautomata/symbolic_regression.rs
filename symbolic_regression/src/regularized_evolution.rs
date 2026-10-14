use std::ops::AddAssign;

use fastrand::Rng;
use num_traits::{Float, FromPrimitive, ToPrimitive};

use crate::adaptive_parsimony::RunningSearchStatistics;
use crate::dataset::TaggedDataset;
use crate::hall_of_fame::HallOfFame;
use crate::mutate::{self, CrossoverCtx, NextGenerationCtx};
use crate::options::Options;
use crate::plugins::MutationWeightsState;
use crate::pop_member::{Evaluator, PopMember};
use crate::population::Population;
use crate::selection::best_of_sample;
use crate::stop_controller::StopController;
use crate::tracing::{IterationTrace, StepTrace};

pub struct RegEvolCtx<'a, T: Float + AddAssign, Ops, const D: usize> {
    pub rng: &'a mut Rng,
    pub dataset: TaggedDataset<'a, T>,
    pub temperature: f64,
    pub curmaxsize: usize,
    pub stats: &'a RunningSearchStatistics,
    pub options: &'a Options<T, Ops, D>,
    pub evaluator: &'a mut Evaluator<T, D>,
    pub weights_state: &'a mut MutationWeightsState,
    /// The cycle's best members, which accepted crossover children and every accepted mutation
    /// of a burst also enter.
    pub best_seen: &'a mut HallOfFame<T, Ops, D>,
    pub controller: &'a StopController,
    pub trace: Option<&'a mut IterationTrace<T, Ops, D>>,
    pub _ops: core::marker::PhantomData<Ops>,
}

pub fn reg_evol_cycle<T, Ops, const D: usize>(
    pop: &mut Population<T, Ops, D>,
    mut ctx: RegEvolCtx<'_, T, Ops, D>,
) -> f64
where
    T: Float + FromPrimitive + ToPrimitive + AddAssign,
    Ops: dynamic_expressions::OperatorSet<T = T>,
{
    let mut num_evals = 0.0;
    let n_evol_cycles = ((pop.len() as f64) / (ctx.options.tournament_selection_n as f64)).ceil() as usize;

    for _ in 0..n_evol_cycles {
        if ctx.controller.is_cancelled() {
            break;
        }
        if ctx.rng.f64() > ctx.options.crossover_probability {
            let allstar = best_of_sample(ctx.rng, pop, ctx.stats, ctx.options);
            let tracing = ctx.trace.is_some();
            let mut burst_steps = Vec::new();
            let mut step = |rng: &mut Rng, parent: &PopMember<T, Ops, D>| {
                let mut step_trace = StepTrace::default();
                let (baby, accepted, evals) = mutate::next_generation_traced(
                    parent,
                    NextGenerationCtx {
                        rng,
                        dataset: ctx.dataset,
                        temperature: ctx.temperature,
                        curmaxsize: ctx.curmaxsize,
                        stats: ctx.stats,
                        options: ctx.options,
                        evaluator: &mut *ctx.evaluator,
                        population: &pop.members,
                        weights_state: &mut *ctx.weights_state,
                        _ops: core::marker::PhantomData,
                    },
                    &mut step_trace,
                );
                num_evals += evals;
                (baby, accepted, step_trace)
            };
            let (baby, accepted, step_trace, selected) = match ctx.options.mutation_burst {
                None => {
                    let (baby, accepted, step_trace) = step(ctx.rng, &allstar);
                    (baby, accepted, step_trace, 0)
                }
                Some(burst) => {
                    let (baby, accepted, selected) = burst.run(ctx.rng, &allstar, |rng, parent| {
                        let (baby, accepted, step_trace) = step(rng, parent);
                        if accepted {
                            ctx.best_seen.consider_unchecked(&baby, ctx.options);
                        }
                        if tracing {
                            burst_steps.push((parent.clone(), baby.clone(), step_trace));
                        }
                        (baby, accepted)
                    });
                    (baby, accepted, StepTrace::default(), selected)
                }
            };
            let should_replace = accepted || !ctx.options.skip_mutation_failures;
            if let Some(trace) = ctx.trace.as_deref_mut() {
                let replaced = should_replace.then(|| &pop.members[pop.oldest_index()]);
                if ctx.options.mutation_burst.is_none() {
                    trace.mutation_attempts(&[(&allstar, &baby, step_trace)], replaced, 0);
                } else {
                    let steps: Vec<_> = burst_steps.iter().map(|(p, c, t)| (p, c, *t)).collect();
                    trace.mutation_attempts(&steps, replaced, selected);
                }
            }
            if !should_replace {
                continue;
            }
            pop.replace_oldest(baby);
        } else {
            let allstar1 = best_of_sample(ctx.rng, pop, ctx.stats, ctx.options);
            let allstar2 = best_of_sample(ctx.rng, pop, ctx.stats, ctx.options);
            let mut step_trace = StepTrace::default();
            let (baby1, baby2, accepted, tmp) = mutate::crossover_generation_traced(
                &allstar1,
                &allstar2,
                CrossoverCtx {
                    rng: ctx.rng,
                    dataset: ctx.dataset,
                    curmaxsize: ctx.curmaxsize,
                    options: ctx.options,
                    evaluator: ctx.evaluator,
                    _ops: core::marker::PhantomData,
                },
                &mut step_trace,
            );
            num_evals += tmp;
            if accepted {
                ctx.best_seen.consider_unchecked(&baby1, ctx.options);
                ctx.best_seen.consider_unchecked(&baby2, ctx.options);
            }
            if !accepted && ctx.options.skip_mutation_failures {
                continue;
            }
            if let Some(trace) = ctx.trace.as_deref_mut() {
                let (i1, i2) = pop.two_oldest_indices();
                trace.crossover(
                    &allstar1,
                    &allstar2,
                    &baby1,
                    &baby2,
                    &pop.members[i1],
                    &pop.members[i2],
                    step_trace,
                );
            }
            pop.replace_two_oldest(baby1, baby2);
        }
    }

    num_evals
}
