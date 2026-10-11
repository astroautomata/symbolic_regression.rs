use std::fmt::Display;
use std::ops::AddAssign;

use dynamic_expressions::OperatorSet;
use dynamic_expressions::dsl::TemplateEvalError;
use dynamic_expressions::dsl::ir::OpIr;
use dynamic_expressions::expression::PostfixExpr;
use fastrand::Rng;
use num_traits::{Float, FromPrimitive, ToPrimitive};

use crate::adaptive_parsimony::RunningSearchStatistics;
use crate::check_constraints::check_expression;
use crate::dataset::{Dataset, TaggedDataset};
use crate::expression::Expression;
use crate::guesses::{ParsedGuesses, seed_members};
use crate::hall_of_fame::HallOfFame;
use crate::loss_functions::baseline_loss_from_zero_expression;
use crate::options::Options;
use crate::pop_member::{Evaluator, PopMember};
use crate::population::Population;
use crate::progress_bars::SearchProgress;
use crate::random::{shuffle, standard_normal};
use crate::stop_controller::StopController;
use crate::tracing::IterationTrace;
use crate::{migration, single_iteration, tracing, warmup};

/// What a search found, with the state SymbolicRegression.jl's `return_state` returns: pass it to
/// [`resume_search`] or [`SearchEngine::resume`] to continue the search.
pub struct SearchResult<T: Float, Ops, const D: usize> {
    /// One hall of fame per output, in the order of the datasets.
    pub halls_of_fame: Vec<HallOfFame<T, Ops, D>>,
    /// Each output's equation SymbolicRegression.jl reports as best: see [`HallOfFame::best`].
    pub best: Vec<Option<PopMember<T, Ops, D>>>,
    /// `populations[j][i]` is population `i` of output `j` as the search left it.
    pub populations: Vec<Vec<Population<T, Ops, D>>>,
    /// The warnings SymbolicRegression.jl logs while setting up the search.
    pub warnings: Vec<String>,
    /// Evaluations, in units of the full dataset as `max_evals` counts them, including those of
    /// every search this one continues.
    pub total_evals: f64,
    /// Each population's generator, so that a warm start continues its random stream.
    rngs: Vec<Vec<Rng>>,
    key: WarmStartKey,
}

impl<T: Float, Ops, const D: usize> Clone for SearchResult<T, Ops, D> {
    fn clone(&self) -> Self {
        Self {
            halls_of_fame: self.halls_of_fame.clone(),
            best: self.best.clone(),
            populations: self.populations.clone(),
            warnings: self.warnings.clone(),
            total_evals: self.total_evals,
            rngs: self.rngs.clone(),
            key: self.key.clone(),
        }
    }
}

/// The options a warm start must keep, as SymbolicRegression.jl's `check_warm_start_compatibility`
/// lists them, and the shape of the data the populations were evolved on. Each is written out so
/// that two keys compare field by field.
#[derive(Clone, Debug, PartialEq)]
struct WarmStartKey(Vec<(&'static str, String)>);

impl WarmStartKey {
    fn new<T: Float, Ops: OperatorSet, const D: usize>(datasets: &[Dataset<T>], options: &Options<T, Ops, D>) -> Self {
        let set = &options.operators.set;
        let operators: Vec<String> = options
            .operators
            .ops_by_arity
            .iter()
            .flatten()
            .map(|&op| match set.ir(op) {
                Some(OpIr::Defined(def)) => {
                    format!(
                        "{}/{} #{} {:?} {:?}",
                        set.name(op),
                        op.arity,
                        op.id,
                        def.graph.nodes(),
                        def.root
                    )
                }
                _ => format!("{}/{} #{}", set.name(op), op.arity, op.id),
            })
            .collect();
        let mut op_constraints: Vec<_> = options
            .op_constraints
            .limits
            .iter()
            .map(|(op, limits)| (op.arity, op.id, limits.to_vec()))
            .collect();
        op_constraints.sort();
        let mut nested_constraints: Vec<_> = options
            .nested_constraints
            .limits
            .iter()
            .map(|(op, nested)| {
                let nested: Vec<_> = nested.iter().map(|(inner, n)| (inner.arity, inner.id, *n)).collect();
                (op.arity, op.id, nested)
            })
            .collect();
        nested_constraints.sort();
        let mut overrides: Vec<_> = options
            .operator_complexity_overrides
            .iter()
            .map(|(op, c)| (op.arity, op.id, c.to_bits()))
            .collect();
        overrides.sort();
        let template = options.template.as_ref().map(|t| {
            (
                t.combine().to_string(),
                t.expressions().to_vec(),
                t.num_features().to_vec(),
                t.parameters().to_vec(),
                t.variable_names().to_vec(),
            )
        });
        Self(vec![
            ("operators", format!("{operators:?}")),
            ("op_constraints", format!("{op_constraints:?}")),
            ("nested_constraints", format!("{nested_constraints:?}")),
            (
                "complexity_mapping",
                format!(
                    "{:?}",
                    (
                        options.complexity_of_constants,
                        options.complexity_of_variables,
                        &options.variable_complexities,
                        overrides,
                    )
                ),
            ),
            (
                "dimensionless_constants_only",
                options.dimensionless_constants_only.to_string(),
            ),
            ("maxsize", options.maxsize.to_string()),
            ("maxdepth", options.maxdepth.to_string()),
            ("populations", options.populations.to_string()),
            ("population_size", options.population_size.to_string()),
            ("expression_type", template.is_some().to_string()),
            ("expression_options", format!("{template:?}")),
            ("number of outputs", datasets.len().to_string()),
            (
                "number of features",
                format!("{:?}", datasets.iter().map(|d| d.n_features).collect::<Vec<_>>()),
            ),
        ])
    }

    /// SymbolicRegression.jl's `WarmStartIncompatibleError`, naming every changed field.
    fn check(&self, new: &Self) -> Result<(), String> {
        let changed: Vec<&str> = self
            .0
            .iter()
            .zip(&new.0)
            .filter(|(old, new)| old != new)
            .map(|(old, _)| old.0)
            .collect();
        if changed.is_empty() {
            return Ok(());
        }
        Err(format!(
            "Warm start incompatible due to changed field(s): {}. Start a new search instead.",
            changed.join(", ")
        ))
    }
}

fn usable_rayon_threads() -> usize {
    let pool_threads = rayon::current_num_threads();
    let in_rayon_pool = rayon::current_thread_index().is_some();
    if in_rayon_pool {
        pool_threads.saturating_sub(1)
    } else {
        pool_threads
    }
}

struct SearchCounters {
    total_cycles: usize,
    cycles_started: usize,
    cycles_completed: usize,
}

impl SearchCounters {
    fn new(total_cycles: usize) -> Self {
        Self {
            total_cycles,
            cycles_started: 0,
            cycles_completed: 0,
        }
    }

    fn cycles_remaining(&self) -> usize {
        self.total_cycles.saturating_sub(self.cycles_completed)
    }

    fn cycles_remaining_start_for_next_dispatch(&mut self) -> usize {
        let remaining = self.total_cycles.saturating_sub(self.cycles_started);
        self.cycles_started += 1;
        remaining
    }

    fn mark_completed(&mut self) -> usize {
        self.cycles_completed += 1;
        self.cycles_remaining()
    }
}

struct SearchTaskResult<T: Float + AddAssign, Ops, const D: usize> {
    output: usize,
    pop_idx: usize,
    evals: f64,
    best_seen: HallOfFame<T, Ops, D>,
    best_sub_pop: Vec<PopMember<T, Ops, D>>,
    pop_state: PopState<T, Ops, D>,
    /// The cycle's `"iteration"` trace record, when tracing.
    trace: Option<String>,
}

pub(crate) struct PopState<T: Float + AddAssign, Ops, const D: usize> {
    pub(crate) pop: Population<T, Ops, D>,
    pub(crate) evaluator: Evaluator<T, D>,
    pub(crate) grad_ctx: dynamic_expressions::GradContext<T, D>,
    pub(crate) rng: Rng,
    pub(crate) batch_dataset: Option<Dataset<T>>,
    /// Cycles this population has run in this search, which numbers its trace records.
    pub(crate) iteration: usize,
}

impl<T: Float + AddAssign, Ops, const D: usize> PopState<T, Ops, D> {
    fn new(pop: Population<T, Ops, D>, rng: Rng, n_rows: usize) -> Self {
        Self {
            pop,
            evaluator: Evaluator::new(n_rows),
            grad_ctx: dynamic_expressions::GradContext::new(n_rows),
            rng,
            batch_dataset: None,
            iteration: 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_iteration_phase<'a, F, Ret>(
        &'a mut self,
        f: F,
        full_dataset: TaggedDataset<'a, T>,
        options: &'a Options<T, Ops, D>,
        curmaxsize: usize,
        stats: &'a RunningSearchStatistics,
        controller: &'a StopController,
        trace: Option<&'a mut IterationTrace<T, Ops, D>>,
    ) -> Ret
    where
        F: FnOnce(
            &mut Population<T, Ops, D>,
            &mut single_iteration::IterationCtx<'a, T, Ops, D>,
            TaggedDataset<'a, T>,
        ) -> Ret,
    {
        let phase_dataset = if options.batching {
            let full_data = full_dataset.data();
            if full_data.n_rows == 0 {
                panic!("Cannot batch from an empty dataset (n_rows = 0).");
            }
            let bs = options.batch_size.max(1);
            let needs_new = match self.batch_dataset.as_ref() {
                None => true,
                Some(b) => b.n_rows != bs || b.n_features != full_data.n_features,
            };
            if needs_new {
                self.batch_dataset = Some(Dataset::make_batch_buffer(full_data, bs));
            }
            let batch = self.batch_dataset.as_mut().expect("set above");
            batch.resample_from(full_data, &mut self.rng);
            TaggedDataset::for_batch(batch, full_dataset.baseline_loss, full_data.n_rows)
        } else {
            full_dataset
        };

        let mut ctx = single_iteration::IterationCtx {
            rng: &mut self.rng,
            full_dataset,
            curmaxsize,
            stats,
            options,
            evaluator: &mut self.evaluator,
            grad_ctx: &mut self.grad_ctx,
            controller,
            trace,
            _ops: core::marker::PhantomData,
        };

        f(&mut self.pop, &mut ctx, phase_dataset)
    }
}

/// One output's search: SymbolicRegression.jl evolves separate populations for each output, and
/// migrates only among them.
struct OutputSearch<T: Float + AddAssign, Ops, const D: usize> {
    counters: SearchCounters,
    stats: RunningSearchStatistics,
    pops: Vec<Option<PopState<T, Ops, D>>>,
    best_sub_pops: Vec<Vec<PopMember<T, Ops, D>>>,
    /// The parsed guesses, which migrate into each population after every cycle.
    seeds: Vec<PopMember<T, Ops, D>>,
}

/// SymbolicRegression.jl's warning for a dataset of more than 50,000 rows, logged once per search.
pub(crate) const LARGE_DATASET_WARNING: &str = "You are using a dataset with more than 50,000 datapoints. Symbolic regression rarely benefits from this many points; consider subsampling to 10,000 points or fewer. If you have high noise, denoise the data first rather than using more points.";

/// Rejects options that cannot search `dataset` (see [`Options::check_dataset`]), before the
/// search starts.
fn check_dataset<T: Float, Ops: dynamic_expressions::OperatorSet, const D: usize>(
    dataset: &Dataset<T>,
    options: &Options<T, Ops, D>,
) {
    if let Err(message) = options.check_dataset(dataset) {
        panic!("{message}");
    }
}

/// Runs the whole search, one output per dataset, as SymbolicRegression.jl's `equation_search`
/// does with a vector of datasets. Fails when a guess is invalid or when the Pareto front cannot
/// be scored under `Options::loss_scale` (see [`HallOfFame::check_loss_scale`]).
pub fn equation_search<T, Ops, const D: usize>(
    datasets: &[Dataset<T>],
    options: &Options<T, Ops, D>,
) -> Result<SearchResult<T, Ops, D>, String>
where
    T: Float + AddAssign + FromPrimitive + ToPrimitive + Display + Send + Sync,
    Ops: OperatorSet<T = T> + Send + Sync,
{
    Search::start(datasets, options, None)?.run_to_completion(datasets, options)
}

/// Continues `previous` for `options.niterations` more iterations, as SymbolicRegression.jl's
/// `equation_search` does with `saved_state`: the populations and halls of fame are rescored on
/// `datasets` with `options`, which may change the loss and most search settings but not those
/// [`SearchEngine::resume`] checks.
pub fn resume_search<T, Ops, const D: usize>(
    previous: SearchResult<T, Ops, D>,
    datasets: &[Dataset<T>],
    options: &Options<T, Ops, D>,
) -> Result<SearchResult<T, Ops, D>, String>
where
    T: Float + AddAssign + FromPrimitive + ToPrimitive + Display + Send + Sync,
    Ops: OperatorSet<T = T> + Send + Sync,
{
    Search::start(datasets, options, Some(previous))?.run_to_completion(datasets, options)
}

struct SearchCore<T: Float + AddAssign, Ops, const D: usize> {
    outputs: Vec<OutputSearch<T, Ops, D>>,
    /// One per output, apart from `outputs` so that callers can borrow them as a slice.
    halls: Vec<HallOfFame<T, Ops, D>>,
    /// Evaluations in this search, which `max_evals` limits.
    total_evals: f64,
    progress: SearchProgress,
    order_rng: Rng,
    cur_iter: usize,
    /// `(output, population)` pairs, shuffled once, as SymbolicRegression.jl orders its tasks.
    task_order: Vec<(usize, usize)>,
    next_task: usize,
    progress_finished: bool,
}

impl<T: Float + AddAssign, Ops, const D: usize> SearchCore<T, Ops, D> {
    fn prepare_iteration_state(&mut self, niterations: usize) {
        if self.cur_iter >= niterations {
            return;
        }
        if !self.task_order.is_empty() && self.next_task < self.task_order.len() {
            return;
        }

        // Matches SymbolicRegression.jl `equation_search`: `task_order` is built and shuffled
        // exactly once at the start of the search, then cycled through forever.
        if self.task_order.is_empty() {
            self.task_order = self
                .outputs
                .iter()
                .enumerate()
                .flat_map(|(j, out)| (0..out.pops.len()).map(move |i| (j, i)))
                .collect();
            shuffle(&mut self.order_rng, &mut self.task_order);
        }
        self.next_task = 0;
        self.cur_iter += 1;
    }

    fn cycles_remaining(&self) -> usize {
        self.outputs.iter().map(|o| o.counters.cycles_remaining()).sum()
    }

    #[inline]
    fn finish_progress_if_needed(&mut self) {
        if !self.progress_finished {
            self.progress.finish();
            self.progress_finished = true;
        }
    }
}

impl<T, Ops, const D: usize> SearchCore<T, Ops, D>
where
    T: Float + FromPrimitive + ToPrimitive + Display + AddAssign + Send + Sync,
    Ops: OperatorSet<T = T> + Send + Sync,
{
    fn step(
        &mut self,
        datasets: &[Dataset<T>],
        baselines: &[Option<T>],
        options: &Options<T, Ops, D>,
        controller: &StopController,
        n_cycles: usize,
    ) -> Result<usize, String> {
        if n_cycles == 0 {
            return Ok(0);
        }

        let is_finished = |core: &Self| core.cycles_remaining() == 0 || controller.is_cancelled();

        if is_finished(self) {
            self.finish_progress_if_needed();
            return Ok(0);
        }

        let usable_threads = usable_rayon_threads();
        let need_inline = usable_threads == 0;
        let n_tasks: usize = self.outputs.iter().map(|o| o.pops.len()).sum();
        let n_workers = usable_threads.min(n_tasks).max(1);

        let (completed_total, error) = rayon::in_place_scope(|scope| {
            let (result_tx, result_rx) = std::sync::mpsc::channel::<SearchTaskResult<T, Ops, D>>();
            let mut in_flight = 0usize;
            let mut completed_total = 0usize;
            let mut stop_dispatching = false;
            let mut error: Option<String> = None;

            while completed_total < n_cycles && (!stop_dispatching || in_flight > 0) {
                if is_finished(self) {
                    stop_dispatching = true;
                }

                if controller.should_stop(self.total_evals) {
                    stop_dispatching = true;
                    controller.cancel();
                }

                if !stop_dispatching && in_flight == 0 && self.next_task >= self.task_order.len() {
                    self.prepare_iteration_state(options.niterations);
                    if self.next_task >= self.task_order.len() {
                        stop_dispatching = true;
                    }
                }

                while !stop_dispatching
                    && in_flight < n_workers
                    && completed_total + in_flight < n_cycles
                    && self.next_task < self.task_order.len()
                {
                    if is_finished(self) {
                        stop_dispatching = true;
                        break;
                    }

                    if controller.should_stop(self.total_evals) {
                        stop_dispatching = true;
                        controller.cancel();
                        break;
                    }

                    let (output, pop_idx) = self.task_order[self.next_task];
                    self.next_task += 1;

                    let out = &mut self.outputs[output];
                    let Some(pop_state) = out.pops[pop_idx].take() else {
                        continue;
                    };

                    let cycles_remaining_start = out.counters.cycles_remaining_start_for_next_dispatch();
                    let curmaxsize =
                        warmup::get_cur_maxsize(options, out.counters.total_cycles, cycles_remaining_start);
                    let mut stats_snapshot = out.stats.clone();
                    stats_snapshot.normalize();
                    let (dataset, baseline_loss) = (&datasets[output], baselines[output]);

                    if need_inline {
                        // Inline exec avoids deadlock in 1-thread pools, while preserving the same
                        // "send -> recv -> apply" completion-driven semantics as the parallel path.
                        let full_dataset = TaggedDataset::new(dataset, baseline_loss);
                        let res = execute_task(
                            full_dataset,
                            options,
                            (output, pop_idx),
                            curmaxsize,
                            stats_snapshot,
                            pop_state,
                            controller,
                        );
                        let _ = result_tx.send(res);
                    } else {
                        let result_tx = result_tx.clone();
                        scope.spawn(move |_| {
                            let full_dataset = TaggedDataset::new(dataset, baseline_loss);
                            let res = execute_task(
                                full_dataset,
                                options,
                                (output, pop_idx),
                                curmaxsize,
                                stats_snapshot,
                                pop_state,
                                controller,
                            );
                            let _ = result_tx.send(res);
                        });
                    }
                    in_flight += 1;
                }

                if in_flight == 0 {
                    if stop_dispatching {
                        break;
                    }
                    continue;
                }

                let res = result_rx.recv().expect("worker result channel closed early");
                in_flight -= 1;
                let applied = self.apply_task_result(options, res, controller);
                completed_total += 1;
                if let Err(message) = applied {
                    error.get_or_insert(message);
                    stop_dispatching = true;
                }

                if controller.should_stop(self.total_evals) {
                    stop_dispatching = true;
                    controller.cancel();
                }
            }

            (completed_total, error)
        });

        if error.is_some() || is_finished(self) {
            self.finish_progress_if_needed();
        }

        error.map_or(Ok(completed_total), Err)
    }

    /// Folds a finished cycle into its output's search. Fails once the Pareto front cannot be
    /// scored under `Options::loss_scale`, where SymbolicRegression.jl's `format_hall_of_fame`
    /// throws.
    fn apply_task_result(
        &mut self,
        options: &Options<T, Ops, D>,
        res: SearchTaskResult<T, Ops, D>,
        controller: &StopController,
    ) -> Result<(), String> {
        self.total_evals += res.evals;

        let (output, pop_idx) = (res.output, res.pop_idx);
        let hall = &mut self.halls[output];
        let OutputSearch {
            counters,
            stats,
            pops,
            best_sub_pops,
            seeds,
        } = &mut self.outputs[output];
        best_sub_pops[pop_idx] = res.best_sub_pop;
        let st = pops[pop_idx].insert(res.pop_state);
        if let Some(record) = &res.trace {
            tracing::write_record(&options.tracing_file, record, true)?;
        }

        stats.update_from_population(st.pop.members.iter().map(|m| m.complexity));
        stats.move_window();

        for m in res.best_seen.members() {
            hall.consider(m, options);
        }
        for m in &st.pop.members {
            hall.consider(m, options);
        }

        if options.migration {
            let mut candidates: Vec<PopMember<T, Ops, D>> = Vec::new();
            for (i, v) in best_sub_pops.iter().enumerate() {
                if i != pop_idx {
                    candidates.extend(v.iter().cloned());
                }
            }
            migration::migrate_into(
                &mut st.pop,
                &candidates,
                options.fraction_replaced,
                &mut st.rng,
                options.deterministic,
            );
        }

        if options.hof_migration {
            let dominating = hall.pareto_front();
            migration::migrate_into(
                &mut st.pop,
                &dominating,
                options.fraction_replaced_hof,
                &mut st.rng,
                options.deterministic,
            );
        }

        migration::migrate_into(
            &mut st.pop,
            seeds,
            options.fraction_replaced_guesses,
            &mut st.rng,
            options.deterministic,
        );

        // Matches SymbolicRegression.jl `check_for_loss_threshold`: cancel as soon as every
        // output's hall of fame has a member that satisfies the user's early-stop callback.
        if let Some(f) = &options.early_stop_condition {
            if self.halls.iter().all(|h| h.any_member_satisfies(f.as_ref())) {
                controller.cancel();
            }
        }

        let cycles_remaining = counters.mark_completed();
        let hall = &self.halls[output];
        hall.check_loss_scale(options.loss_scale)?;
        self.progress.on_cycle_complete(
            hall,
            options,
            self.total_evals as u64,
            cycles_remaining,
            options.print_precision,
        )
    }
}

/// A search over borrowed datasets and options, which [`equation_search`] runs directly and
/// [`SearchEngine`] runs over its own copies.
struct Search<T: Float + AddAssign, Ops, const D: usize> {
    core: SearchCore<T, Ops, D>,
    baselines: Vec<Option<T>>,
    controller: StopController,
    warnings: Vec<String>,
    /// Evaluations by the searches this one continues.
    prior_evals: f64,
    key: WarmStartKey,
}

impl<T, Ops, const D: usize> Search<T, Ops, D>
where
    T: Float + FromPrimitive + ToPrimitive + Display + AddAssign,
    Ops: OperatorSet<T = T>,
{
    fn start(
        datasets: &[Dataset<T>],
        options: &Options<T, Ops, D>,
        previous: Option<SearchResult<T, Ops, D>>,
    ) -> Result<Self, String> {
        if datasets.is_empty() {
            return Err("a search needs at least one dataset".to_string());
        }
        for dataset in datasets {
            check_dataset(dataset, options);
        }
        let key = WarmStartKey::new(datasets, options);
        if let Some(previous) = &previous {
            previous.key.check(&key)?;
        }
        let baselines: Vec<Option<T>> = datasets
            .iter()
            .map(|d| {
                if options.use_baseline {
                    baseline_loss_from_zero_expression::<T, Ops, D>(d, &options.operators.set, options.loss.as_ref())
                } else {
                    None
                }
            })
            .collect();
        let tagged: Vec<TaggedDataset<'_, T>> = datasets
            .iter()
            .zip(&baselines)
            .map(|(d, &b)| TaggedDataset::new(d, b))
            .collect();

        let controller = StopController::from_options(options);
        if options.use_tracing {
            tracing::write_record(&options.tracing_file, &tracing::search_record(options), false)?;
        }
        let mut warnings = Vec::new();
        if datasets.iter().any(|d| d.n_rows > 50_000) {
            warnings.push(LARGE_DATASET_WARNING.to_string());
        }
        let ParsedGuesses {
            members: seeds,
            warnings: guess_warnings,
        } = seed_members(&tagged, options)?;
        warnings.extend(guess_warnings);

        let (prior_evals, mut saved) = match previous {
            Some(p) => (p.total_evals, Some((p.halls_of_fame, p.populations, p.rngs))),
            None => (0.0, None),
        };
        let mut total_evals = 0.0;
        let mut halls = Vec::with_capacity(datasets.len());
        let mut outputs = Vec::with_capacity(datasets.len());
        for (j, (dataset, seeds)) in tagged.iter().zip(seeds).enumerate() {
            let mut evaluator = Evaluator::new(dataset.n_rows);
            let (mut hall, mut saved_pops) = match &mut saved {
                Some((halls, pops, rngs)) => {
                    let mut hall = std::mem::replace(&mut halls[j], HallOfFame::new(0));
                    for m in hall.best_by_complexity.iter_mut().flatten() {
                        m.evaluate(dataset, options, &mut evaluator);
                    }
                    let pops = std::mem::take(&mut pops[j]);
                    let rngs = std::mem::take(&mut rngs[j]);
                    (hall, pops.into_iter().zip(rngs).map(Some).collect())
                }
                None => (HallOfFame::new(options.maxsize), Vec::new()),
            };
            for seed in &seeds {
                if check_expression(&seed.expr, options, options.maxsize) {
                    hall.consider(seed, options);
                }
            }
            let mut pops = Vec::with_capacity(options.populations);
            for i in 0..options.populations {
                if controller.is_cancelled() {
                    break;
                }
                let state = match saved_pops.get_mut(i).and_then(Option::take) {
                    Some((mut pop, rng)) if pop.len() == options.population_size => {
                        for m in &mut pop.members {
                            m.evaluate(dataset, options, &mut evaluator);
                        }
                        PopState::new(pop, rng, dataset.n_rows)
                    }
                    saved_pop => {
                        if saved_pop.is_some() {
                            warnings.push(format!(
                                "Recreating population (output={}, population={}), as the saved one doesn't have the correct number of members.",
                                j + 1,
                                i + 1
                            ));
                        }
                        let rng = Rng::with_seed(options.seed.wrapping_add((j * options.populations + i) as u64));
                        let Some(state) =
                            new_population(*dataset, options, &controller, rng, i == 0).map_err(|e| e.to_string())?
                        else {
                            break;
                        };
                        total_evals += state.pop.len() as f64 * dataset.dataset_fraction();
                        state
                    }
                };
                pops.push(Some(state));
            }
            let best_sub_pops = pops
                .iter()
                .map(|st| migration::best_sub_pop(&st.as_ref().expect("population exists").pop, options.topn))
                .collect();
            outputs.push(OutputSearch {
                counters: SearchCounters::new(options.niterations * pops.len()),
                stats: RunningSearchStatistics::new(options.maxsize, 100_000),
                pops,
                best_sub_pops,
                seeds,
            });
            halls.push(hall);
        }

        let total_cycles = outputs.iter().map(|o| o.counters.total_cycles).sum();
        let mut progress = SearchProgress::new(options, datasets.len() == 1, total_cycles, datasets[0].units.as_ref());
        for warning in &warnings {
            progress.warn(warning);
        }
        progress.set_initial_evals(total_evals as u64);

        let core = SearchCore {
            outputs,
            halls,
            total_evals,
            progress,
            order_rng: Rng::with_seed(options.seed ^ 0x9e37_79b9_7f4a_7c15),
            cur_iter: 0,
            task_order: Vec::new(),
            next_task: 0,
            progress_finished: false,
        };

        Ok(Self {
            core,
            baselines,
            controller,
            warnings,
            prior_evals,
            key,
        })
    }

    fn is_finished(&self) -> bool {
        self.core.cycles_remaining() == 0 || self.controller.is_cancelled()
    }

    fn best(&self, options: &Options<T, Ops, D>) -> Result<Vec<Option<PopMember<T, Ops, D>>>, String> {
        self.core.halls.iter().map(|h| h.best(options.loss_scale)).collect()
    }

    fn step(&mut self, datasets: &[Dataset<T>], options: &Options<T, Ops, D>, n_cycles: usize) -> Result<usize, String>
    where
        T: Send + Sync,
        Ops: Send + Sync,
    {
        self.core
            .step(datasets, &self.baselines, options, &self.controller, n_cycles)
    }

    fn run_to_completion(
        mut self,
        datasets: &[Dataset<T>],
        options: &Options<T, Ops, D>,
    ) -> Result<SearchResult<T, Ops, D>, String>
    where
        T: Send + Sync,
        Ops: Send + Sync,
    {
        let stepped = self.step(datasets, options, usize::MAX);
        self.core.finish_progress_if_needed();
        stepped?;
        let best = self.best(options)?;
        let (populations, rngs) = self
            .core
            .outputs
            .into_iter()
            .map(|out| {
                out.pops
                    .into_iter()
                    .map(|st| {
                        let st = st.expect("every cycle has finished");
                        (st.pop, st.rng)
                    })
                    .unzip()
            })
            .unzip();
        Ok(SearchResult {
            halls_of_fame: self.core.halls,
            best,
            populations,
            warnings: self.warnings,
            total_evals: self.prior_evals + self.core.total_evals,
            rngs,
            key: self.key,
        })
    }
}

/// A search that runs a few cycles at a time, as the web UI drives it.
pub struct SearchEngine<T: Float + AddAssign, Ops, const D: usize> {
    datasets: Vec<Dataset<T>>,
    options: Options<T, Ops, D>,
    search: Search<T, Ops, D>,
}

impl<T, Ops, const D: usize> SearchEngine<T, Ops, D>
where
    T: Float + FromPrimitive + ToPrimitive + Display + AddAssign,
    Ops: OperatorSet<T = T>,
{
    /// Builds the initial populations of a search with one output per dataset, as
    /// SymbolicRegression.jl's `equation_search` does with a vector of datasets. Fails when a guess
    /// is invalid or the template cannot be evaluated.
    pub fn new(datasets: Vec<Dataset<T>>, options: Options<T, Ops, D>) -> Result<Self, String> {
        let search = Search::start(&datasets, &options, None)?;
        Ok(Self {
            datasets,
            options,
            search,
        })
    }

    /// Continues `previous`, as SymbolicRegression.jl's `equation_search` does with `saved_state`:
    /// its halls of fame and populations are rescored on `datasets` with `options`, and each
    /// population keeps its random stream. Fails, as SymbolicRegression.jl's
    /// `check_warm_start_compatibility` does, when `options` change the operators, constraints,
    /// complexities, `dimensionless_constants_only`, `maxsize`, `maxdepth`, `populations`,
    /// `population_size`, or the template, or when the number of outputs or features changes.
    pub fn resume(
        previous: SearchResult<T, Ops, D>,
        datasets: Vec<Dataset<T>>,
        options: Options<T, Ops, D>,
    ) -> Result<Self, String> {
        let search = Search::start(&datasets, &options, Some(previous))?;
        Ok(Self {
            datasets,
            options,
            search,
        })
    }

    pub fn total_cycles(&self) -> usize {
        self.search.core.outputs.iter().map(|o| o.counters.total_cycles).sum()
    }

    pub fn cycles_completed(&self) -> usize {
        self.search
            .core
            .outputs
            .iter()
            .map(|o| o.counters.cycles_completed)
            .sum()
    }

    /// Evaluations, including those of every search this one continues.
    pub fn total_evals(&self) -> u64 {
        (self.search.prior_evals + self.search.core.total_evals) as u64
    }

    pub fn is_finished(&self) -> bool {
        self.search.is_finished()
    }

    /// One hall of fame per output.
    pub fn halls_of_fame(&self) -> &[HallOfFame<T, Ops, D>] {
        &self.search.core.halls
    }

    /// Each output's best equation: see [`HallOfFame::best`].
    pub fn best(&self) -> Result<Vec<Option<PopMember<T, Ops, D>>>, String> {
        self.search.best(&self.options)
    }

    pub fn datasets(&self) -> &[Dataset<T>] {
        &self.datasets
    }

    pub fn options(&self) -> &Options<T, Ops, D> {
        &self.options
    }

    /// The warnings SymbolicRegression.jl logs while setting up the search.
    pub fn warnings(&self) -> &[String] {
        &self.search.warnings
    }

    /// Runs up to `n_cycles` more cycles and returns how many completed. Fails, after the cycles
    /// already running finish, once the Pareto front cannot be scored under `Options::loss_scale`
    /// (see [`HallOfFame::check_loss_scale`]).
    pub fn step(&mut self, n_cycles: usize) -> Result<usize, String>
    where
        T: Send + Sync,
        Ops: Send + Sync,
    {
        self.search.step(&self.datasets, &self.options, n_cycles)
    }

    pub fn run_to_completion(self) -> Result<SearchResult<T, Ops, D>, String>
    where
        T: Send + Sync,
        Ops: Send + Sync,
    {
        self.search.run_to_completion(&self.datasets, &self.options)
    }
}

fn execute_task<T, Ops, const D: usize>(
    full_dataset: TaggedDataset<'_, T>,
    options: &Options<T, Ops, D>,
    (output, pop_idx): (usize, usize),
    curmaxsize: usize,
    stats: RunningSearchStatistics,
    mut pop_state: PopState<T, Ops, D>,
    controller: &StopController,
) -> SearchTaskResult<T, Ops, D>
where
    T: Float + FromPrimitive + ToPrimitive + Display + AddAssign,
    Ops: OperatorSet<T = T>,
{
    if controller.is_cancelled() {
        return SearchTaskResult {
            output,
            pop_idx,
            evals: 0.0,
            best_seen: HallOfFame::new(options.maxsize),
            best_sub_pop: migration::best_sub_pop(&pop_state.pop, options.topn),
            pop_state,
            trace: None,
        };
    }
    let mut trace = options
        .use_tracing
        .then(|| IterationTrace::start(output + 1, pop_idx + 1, pop_state.iteration, &pop_state.pop));
    pop_state.iteration += 1;
    let (evals1, best_seen) = pop_state.run_iteration_phase(
        single_iteration::s_r_cycle,
        full_dataset,
        options,
        curmaxsize,
        &stats,
        controller,
        trace.as_mut(),
    );

    let evals2 = pop_state.run_iteration_phase(
        single_iteration::optimize_and_simplify_population,
        full_dataset,
        options,
        curmaxsize,
        &stats,
        controller,
        trace.as_mut(),
    );
    let evals = evals1.max(0.0) + evals2.max(0.0);

    let best_sub_pop = migration::best_sub_pop(&pop_state.pop, options.topn);

    SearchTaskResult {
        output,
        pop_idx,
        evals,
        best_seen,
        best_sub_pop,
        pop_state,
        trace: trace.map(|t| t.to_json(options)),
    }
}

/// Builds and scores a random population; `None` when the search is cancelled before a member
/// exists. As in SymbolicRegression.jl, its members reach the hall of fame only after a cycle. With
/// `check_template`, fails when the template cannot be evaluated, which the first member's
/// evaluation finds.
fn new_population<T, Ops, const D: usize>(
    full_dataset: TaggedDataset<'_, T>,
    options: &Options<T, Ops, D>,
    controller: &StopController,
    mut rng: Rng,
    check_template: bool,
) -> Result<Option<PopState<T, Ops, D>>, TemplateEvalError>
where
    T: Float + FromPrimitive + ToPrimitive + AddAssign,
    Ops: OperatorSet<T = T>,
{
    let dataset = full_dataset.data();
    let mut evaluator = Evaluator::new(dataset.n_rows);
    let nlength = 3usize;
    let mut members = Vec::with_capacity(options.population_size);
    for _ in 0..options.population_size {
        if controller.is_cancelled() {
            break;
        }
        let tree = crate::mutation_functions::random_expr_append_ops(
            &mut rng,
            &options.operators,
            options.max_features(dataset.n_features),
            nlength,
            options.maxsize,
        );
        let mut m = PopMember::from_expr(new_expression(tree, options, &mut rng), dataset.n_features, options);
        if check_template && options.template.is_some() && members.is_empty() {
            evaluator.eval_checked(&m.expr, &m.plans, options, dataset.x.view())?;
        }
        let _ = m.evaluate(&full_dataset, options, &mut evaluator);
        members.push(m);
    }
    if members.is_empty() {
        return Ok(None);
    }
    let mut state = PopState::new(Population::new(members), rng, dataset.n_rows);
    state.evaluator = evaluator;
    Ok(Some(state))
}

/// SymbolicRegression.jl's `create_expression`: a template starts with a copy of `tree` in every
/// subexpression and parameters drawn from a standard normal.
pub(crate) fn new_expression<T: Float, Ops, const D: usize>(
    tree: PostfixExpr<T, Ops, D>,
    options: &Options<T, Ops, D>,
    rng: &mut Rng,
) -> Expression<T, Ops, D> {
    let Some(template) = &options.template else {
        return tree.into();
    };
    Expression {
        trees: std::iter::repeat_n(tree, template.expressions().len()).collect(),
        params: (0..template.n_parameters())
            .map(|_| T::from(standard_normal(rng)).expect("a float converts to T"))
            .collect(),
    }
}

#[cfg(test)]
mod batching_search_tests {
    use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
    use ndarray::{Array1, Array2};

    use super::SearchEngine;
    use crate::Options;
    use crate::dataset::Dataset;

    #[test]
    fn search_engine_allocates_batch_buffer_when_batching_enabled() {
        type T = f64;
        type Ops = BuiltinOpsF64;
        const D: usize = 3;

        let n_rows = 20;
        let n_features = 2;
        let mut x = Array2::<T>::zeros((n_features, n_rows));
        for row in 0..n_rows {
            x[(0, row)] = row as T;
            x[(1, row)] = (row as T) + 100.0;
        }
        let y = Array1::from_iter((0..n_rows).map(|i| i as T));
        let dataset = Dataset::new(x, y);

        let operators = Ops::new().from_names::<D, _>(["sin", "+", "*"]).expect("valid opset");
        let options: Options<T, Ops, D> = Options {
            operators,
            batching: true,
            batch_size: 5,
            niterations: 1,
            ncycles_per_iteration: 1,
            populations: 1,
            population_size: 6,
            progress: false,
            should_optimize_constants: false,
            should_simplify: false,
            deterministic: true,
            ..Default::default()
        };

        let mut engine = SearchEngine::<T, Ops, D>::new(vec![dataset], options.clone()).unwrap();
        assert!(
            engine.search.core.outputs[0].pops[0]
                .as_ref()
                .expect("pop exists")
                .batch_dataset
                .is_none(),
            "batch buffer should be allocated lazily"
        );

        let _ = engine.step(1);

        let pop_state = engine.search.core.outputs[0].pops[0].as_ref().expect("pop exists");
        let batch = pop_state.batch_dataset.as_ref().expect("batch buffer created");
        assert_eq!(batch.n_rows, options.batch_size.max(1));
        assert_eq!(batch.n_features, engine.datasets[0].n_features);
    }

    #[test]
    fn search_engine_does_not_allocate_batch_buffer_when_batching_disabled() {
        type T = f64;
        type Ops = BuiltinOpsF64;
        const D: usize = 3;

        let n_rows = 20;
        let n_features = 2;
        let mut x = Array2::<T>::zeros((n_features, n_rows));
        for row in 0..n_rows {
            x[(0, row)] = row as T;
            x[(1, row)] = (row as T) + 100.0;
        }
        let y = Array1::from_iter((0..n_rows).map(|i| i as T));
        let dataset = Dataset::new(x, y);

        let operators = Ops::new().from_names::<D, _>(["sin", "+", "*"]).expect("valid opset");
        let options: Options<T, Ops, D> = Options {
            operators,
            batching: false,
            batch_size: 5,
            niterations: 1,
            ncycles_per_iteration: 1,
            populations: 1,
            population_size: 6,
            progress: false,
            should_optimize_constants: false,
            should_simplify: false,
            deterministic: true,
            ..Default::default()
        };

        let mut engine = SearchEngine::<T, Ops, D>::new(vec![dataset], options).unwrap();
        let _ = engine.step(1);

        let pop_state = engine.search.core.outputs[0].pops[0].as_ref().expect("pop exists");
        assert!(pop_state.batch_dataset.is_none());
    }
}
