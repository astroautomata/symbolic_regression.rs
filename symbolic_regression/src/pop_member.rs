use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::time::{SystemTime, UNIX_EPOCH};

use dynamic_expressions::dsl::{TemplateEvalError, TemplateWorkspace};
use dynamic_expressions::{EvalOptions, EvalPlan, XColumns};
use num_traits::Float;
#[cfg(target_arch = "wasm32")]
use web_time::{SystemTime, UNIX_EPOCH};

use crate::dataset::TaggedDataset;
use crate::expression::{Expression, OneOrMany};
use crate::loss_functions::loss_to_cost;
use crate::options::Options;

#[derive(Debug)]
pub struct PopMember<T: Float, Ops, const D: usize> {
    pub birth: u64,
    pub expr: Expression<T, Ops, D>,
    /// One evaluation plan per tree of `expr`, with a single plan stored inline.
    pub plans: OneOrMany<EvalPlan<D>>,
    pub complexity: usize,
    pub loss: T,
    pub cost: T,
}

static PSEUDO_TIME: OnceLock<AtomicU64> = OnceLock::new();

fn pseudo_time() -> &'static AtomicU64 {
    PSEUDO_TIME.get_or_init(|| AtomicU64::new(0))
}

pub(crate) fn get_birth_order(deterministic: bool) -> u64 {
    if deterministic {
        // SymbolicRegression.jl: `pseudo_time[] += 1; return pseudo_time[]`
        return pseudo_time().fetch_add(1, Ordering::Relaxed).saturating_add(1);
    }

    // SymbolicRegression.jl: `round(Int, 1e7 * time())`
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock must be after UNIX_EPOCH");
    let secs = dur.as_secs();
    let nanos = dur.subsec_nanos() as u64;
    // Round to the nearest 100ns tick (for positive values, ties round up).
    let ticks_1e7 = nanos.saturating_add(50) / 100;
    secs.saturating_mul(10_000_000).saturating_add(ticks_1e7)
}

#[cfg(test)]
pub(crate) fn reset_pseudo_time_for_tests() {
    pseudo_time().store(0, Ordering::Relaxed);
}

impl<T: Float, Ops, const D: usize> Clone for PopMember<T, Ops, D> {
    fn clone(&self) -> Self {
        Self {
            birth: self.birth,
            expr: self.expr.clone(),
            plans: self.plans.clone(),
            complexity: self.complexity,
            loss: self.loss,
            cost: self.cost,
        }
    }
}

pub struct Evaluator<T: Float, const D: usize> {
    pub eval_opts: EvalOptions,
    pub yhat: Vec<T>,
    pub scratch: ndarray::Array2<T>,
    pub template: TemplateWorkspace<T>,
}

impl<T: Float, const D: usize> Evaluator<T, D> {
    pub fn new(n_rows: usize) -> Self {
        Self {
            eval_opts: EvalOptions {
                check_finite: true,
                early_exit: true,
            },
            yhat: vec![T::zero(); n_rows],
            scratch: ndarray::Array2::zeros((0, 0)),
            template: TemplateWorkspace::default(),
        }
    }

    pub fn ensure_n_rows(&mut self, n_rows: usize) {
        if self.yhat.len() != n_rows {
            self.yhat.resize(n_rows, T::zero());
        }
    }

    /// Evaluates `expr` on `x` into `yhat`; `false` when the result is invalid. The search calls
    /// [`Self::eval_checked`] once before any other evaluation, so a template cannot fail here.
    pub fn eval<'a, Ops>(
        &mut self,
        expr: &Expression<T, Ops, D>,
        plans: &[EvalPlan<D>],
        options: &Options<T, Ops, D>,
        x: impl Into<XColumns<'a, T>>,
    ) -> bool
    where
        T: 'a,
        Ops: dynamic_expressions::OperatorSet<T = T>,
    {
        self.eval_checked(expr, plans, options, x)
            .expect("the search checks that the template can be evaluated before it starts")
    }

    /// [`Self::eval`], failing when the template cannot be evaluated at all.
    pub fn eval_checked<'a, Ops>(
        &mut self,
        expr: &Expression<T, Ops, D>,
        plans: &[EvalPlan<D>],
        options: &Options<T, Ops, D>,
        x: impl Into<XColumns<'a, T>>,
    ) -> Result<bool, TemplateEvalError>
    where
        T: 'a,
        Ops: dynamic_expressions::OperatorSet<T = T>,
    {
        let x = x.into();
        match &options.template {
            None => Ok(dynamic_expressions::eval_plan_array_into(
                &mut self.yhat,
                &plans[0],
                &expr.trees[0],
                &options.operators.set,
                x,
                &mut self.scratch,
                &self.eval_opts,
            )),
            Some(template) => template.eval_into(
                &mut self.yhat,
                &expr.trees,
                plans,
                &expr.params,
                &options.operators.set,
                x.view(),
                &mut self.template,
            ),
        }
    }
}

impl<T: Float, Ops, const D: usize> PopMember<T, Ops, D>
where
    Ops: dynamic_expressions::OperatorSet<T = T>,
{
    pub fn from_expr(expr: impl Into<Expression<T, Ops, D>>, n_features: usize, options: &Options<T, Ops, D>) -> Self {
        Self::from_expr_with_birth(get_birth_order(options.deterministic), expr, n_features, options)
    }

    pub fn from_expr_with_birth(
        birth: u64,
        expr: impl Into<Expression<T, Ops, D>>,
        n_features: usize,
        options: &Options<T, Ops, D>,
    ) -> Self {
        let expr = expr.into();
        let plans = expr.compile_plans(options.max_features(n_features));
        Self {
            birth,
            expr,
            plans,
            complexity: 0,
            loss: T::infinity(),
            cost: T::infinity(),
        }
    }

    pub fn rebuild_plans(&mut self, n_features: usize, options: &Options<T, Ops, D>) {
        self.plans = self.expr.compile_plans(options.max_features(n_features));
    }

    pub fn evaluate(
        &mut self,
        dataset: &TaggedDataset<'_, T>,
        options: &Options<T, Ops, D>,
        evaluator: &mut Evaluator<T, D>,
    ) -> bool {
        evaluator.ensure_n_rows(dataset.n_rows);
        let ok = evaluator.eval(&self.expr, &self.plans, options, dataset.x_columns());

        self.complexity = options.complexity(&self.expr);

        if !ok {
            self.loss = T::infinity();
            self.cost = T::infinity();
            return false;
        }

        let loss = options
            .loss
            .loss(&evaluator.yhat, dataset.y.as_slice().unwrap(), dataset.weights_slice());
        if loss.is_nan() {
            self.loss = loss;
            self.cost = T::nan();
            return false;
        }
        if !loss.is_finite() {
            self.loss = T::infinity();
            self.cost = T::infinity();
            return false;
        }
        self.loss = loss;

        self.cost = loss_to_cost(
            loss,
            self.complexity,
            options.parsimony,
            options.use_baseline,
            dataset.baseline_loss,
        );
        true
    }
}
