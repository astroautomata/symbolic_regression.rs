use std::ops::AddAssign;

use dynamic_expressions::utils::ZipEq;
use dynamic_expressions::{EvalOptions, EvalPlan, GradContext, OperatorSet, XColumns};
use fastrand::Rng;
use num_traits::{Float, FromPrimitive, ToPrimitive};

use crate::dataset::{Dataset, TaggedDataset};
use crate::expression::Expression;
use crate::optim::{BackTracking, Objective, OptimOptions, bfgs_minimize, newton_1d_minimize};
use crate::options::Options;
use crate::pop_member::{Evaluator, PopMember, get_birth_order};
use crate::random::standard_normal;

struct EvalWorkspace<'a, T: Float + AddAssign, Ops, const D: usize> {
    dataset: &'a Dataset<T>,
    x_columns: XColumns<'a, T>,
    options: &'a Options<T, Ops, D>,
    evaluator: &'a mut Evaluator<T, D>,
    grad_ctx: &'a mut GradContext<T, D>,
    eval_opts: EvalOptions,
    dloss_dyhat: Vec<T>,
}

impl<'a, T: Float + FromPrimitive + AddAssign, Ops: OperatorSet<T = T>, const D: usize> EvalWorkspace<'a, T, Ops, D> {
    fn new(
        dataset: TaggedDataset<'a, T>,
        options: &'a Options<T, Ops, D>,
        evaluator: &'a mut Evaluator<T, D>,
        grad_ctx: &'a mut GradContext<T, D>,
    ) -> Self {
        let eval_opts = EvalOptions {
            check_finite: true,
            early_exit: true,
        };

        Self {
            dataset: dataset.data(),
            x_columns: dataset.x_columns(),
            options,
            evaluator,
            grad_ctx,
            eval_opts,
            dloss_dyhat: vec![T::zero(); dataset.n_rows],
        }
    }

    fn loss_only(&mut self, plans: &[EvalPlan<D>], expr: &Expression<T, Ops, D>) -> Option<f64> {
        if !self.evaluator.eval(expr, plans, self.options, self.x_columns) {
            return None;
        }

        let loss = self.options.loss.loss(
            &self.evaluator.yhat,
            self.dataset.y.as_slice().unwrap(),
            self.dataset.weights_slice(),
        );
        if !loss.is_finite() {
            return None;
        }

        Some(loss.to_f64().unwrap_or(f64::INFINITY))
    }

    /// The loss and its gradient in `grad_out`. A single tree is differentiated exactly; a
    /// template uses central finite differences, as SymbolicRegression.jl's default
    /// (`autodiff_backend=nothing`) optimizer does.
    fn loss_and_grad(
        &mut self,
        plans: &[EvalPlan<D>],
        expr: &mut Expression<T, Ops, D>,
        grad_out: &mut [f64],
    ) -> Option<f64> {
        if self.options.template.is_some() {
            return self.loss_and_finite_difference_grad(plans, expr, grad_out);
        }
        let tree = &expr.trees[0];
        let n_params = tree.consts.len();
        let n_rows = self.dataset.n_rows;
        debug_assert_eq!(grad_out.len(), n_params);
        debug_assert_eq!(self.dloss_dyhat.len(), n_rows);

        let (yhat, dy_dc, ok) = dynamic_expressions::eval_grad_tree_array(
            tree,
            &self.options.operators.set,
            self.x_columns,
            false,
            self.grad_ctx,
            &self.eval_opts,
        );
        if !ok {
            return None;
        }
        if yhat.iter().any(|v| !v.is_finite()) {
            return None;
        }

        let loss = self
            .options
            .loss
            .loss(&yhat, self.dataset.y.as_slice().unwrap(), self.dataset.weights_slice());
        if !loss.is_finite() {
            return None;
        }

        self.options.loss.dloss_dyhat(
            &yhat,
            self.dataset.y.as_slice().unwrap(),
            self.dataset.weights_slice(),
            &mut self.dloss_dyhat,
        );

        for (ci, gout) in grad_out.iter_mut().enumerate() {
            if ci >= n_params {
                break;
            }
            let base = ci * n_rows;
            let acc = self
                .dloss_dyhat
                .iter()
                .copied()
                .zip_eq(dy_dc.data[base..base + n_rows].iter().copied())
                .fold(T::zero(), |a, (dl, dc)| a + dl * dc);
            *gout = acc.to_f64().unwrap_or(f64::INFINITY);
        }

        Some(loss.to_f64().unwrap_or(f64::INFINITY))
    }

    fn loss_and_finite_difference_grad(
        &mut self,
        plans: &[EvalPlan<D>],
        expr: &mut Expression<T, Ops, D>,
        grad_out: &mut [f64],
    ) -> Option<f64> {
        let loss = self.loss_only(plans, expr)?;
        let mut values = expr.constants();
        // FiniteDiff.jl's central-difference step for the element type.
        let rel_step = T::epsilon().to_f64()?.cbrt();
        for (i, g) in grad_out.iter_mut().enumerate() {
            let x = values[i];
            let h = rel_step * x.to_f64()?.abs().max(1.0);
            values[i] = x + T::from_f64(h)?;
            expr.set_constants(&values);
            let up = self.loss_only(plans, expr);
            values[i] = x - T::from_f64(h)?;
            expr.set_constants(&values);
            let down = self.loss_only(plans, expr);
            values[i] = x;
            expr.set_constants(&values);
            *g = (up? - down?) / (2.0 * h);
        }
        Some(loss)
    }

    fn optimize_from_start(
        &mut self,
        start: &[f64],
        n_params: usize,
        member: &mut PopMember<T, Ops, D>,
        optim_opts: OptimOptions,
        ls: BackTracking,
    ) -> Option<crate::optim::OptimResult> {
        let mut obj = ConstObjective {
            plans: &member.plans,
            expr: &mut member.expr,
            workspace: self,
        };

        if n_params == 1 {
            newton_1d_minimize(start[0], &mut obj, optim_opts, ls)
        } else {
            bfgs_minimize(start, &mut obj, optim_opts, ls)
        }
    }
}

struct ConstObjective<'plan, 'expr, 'work, 'data, T: Float + AddAssign, Ops, const D: usize> {
    plans: &'plan [EvalPlan<D>],
    expr: &'expr mut Expression<T, Ops, D>,
    workspace: &'work mut EvalWorkspace<'data, T, Ops, D>,
}

impl<T: Float + FromPrimitive + AddAssign, Ops, const D: usize> ConstObjective<'_, '_, '_, '_, T, Ops, D> {
    fn set(&mut self, x: &[f64]) -> Option<()> {
        let expr = &mut *self.expr;
        debug_assert_eq!(x.len(), expr.n_constants());
        let slots = expr
            .trees
            .iter_mut()
            .flat_map(|t| t.consts.iter_mut())
            .chain(expr.params.iter_mut());
        for (dst, &src) in slots.zip(x) {
            *dst = T::from_f64(src)?;
        }
        Some(())
    }
}

impl<T: Float + FromPrimitive + AddAssign, Ops, const D: usize> Objective for ConstObjective<'_, '_, '_, '_, T, Ops, D>
where
    Ops: OperatorSet<T = T>,
{
    fn f_only(&mut self, x: &[f64], budget: &mut crate::optim::EvalBudget) -> Option<f64> {
        budget.f_calls += 1;
        self.set(x)?;
        self.workspace.loss_only(self.plans, self.expr)
    }

    fn fg(&mut self, x: &[f64], g_out: &mut [f64], budget: &mut crate::optim::EvalBudget) -> Option<f64> {
        budget.f_calls += 1;
        self.set(x)?;
        self.workspace.loss_and_grad(self.plans, self.expr, g_out)
    }
}

pub fn optimize_constants<T: Float + FromPrimitive + ToPrimitive + AddAssign, Ops, const D: usize>(
    rng: &mut Rng,
    member: &mut PopMember<T, Ops, D>,
    ctx: OptimizeConstantsCtx<'_, '_, T, Ops, D>,
) -> (bool, f64)
where
    Ops: OperatorSet<T = T>,
{
    let OptimizeConstantsCtx {
        dataset,
        options,
        evaluator,
        grad_ctx,
    } = ctx;
    evaluator.ensure_n_rows(dataset.n_rows);
    grad_ctx.n_rows = dataset.n_rows;
    let n_params = member.expr.n_constants();
    if n_params == 0 {
        return (false, 0.0);
    }

    let orig_consts = member.expr.constants();
    let orig_birth = member.birth;
    let orig_loss = member.loss;
    let orig_cost = member.cost;

    let mut workspace = EvalWorkspace::new(dataset, options, evaluator, grad_ctx);

    let baseline = match workspace.loss_only(&member.plans, &member.expr) {
        Some(v) => v,
        None => return (false, 0.0),
    };

    let x0: Vec<f64> = orig_consts.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();

    let mut best_x = x0.clone();
    let mut best_f = baseline;

    let optim_opts = OptimOptions {
        iterations: options.optimizer_iterations,
        f_calls_limit: options.optimizer_f_calls_limit,
        g_abstol: 1e-8,
    };
    let ls = BackTracking::default();

    // Matches SymbolicRegression.jl `ConstantOptimization.jl`: each f-call counts as one
    // `dataset_fraction(dataset)` toward `num_evals`.
    let eval_fraction = dataset.dataset_fraction();
    let mut n_evals: f64 = 0.0;

    {
        let res = workspace.optimize_from_start(&x0, n_params, member, optim_opts, ls);
        if let Some(res) = res {
            n_evals += (res.f_calls as f64) * eval_fraction;
            if res.minimum < best_f {
                best_f = res.minimum;
                best_x = res.minimizer;
            }
        }
    }

    // Restarts:
    for _ in 0..options.optimizer_nrestarts {
        let mut xt = x0.clone();
        for v in &mut xt {
            let eps: f64 = standard_normal(rng);
            *v *= 1.0 + 0.5 * eps;
        }

        let res = workspace.optimize_from_start(&xt, n_params, member, optim_opts, ls);
        if let Some(res) = res {
            n_evals += (res.f_calls as f64) * eval_fraction;
            if res.minimum < best_f {
                best_f = res.minimum;
                best_x = res.minimizer;
            }
        }
    }

    if best_f < baseline {
        let best: Vec<T> = best_x.iter().map(|&v| T::from_f64(v).unwrap_or_else(T::zero)).collect();
        member.expr.set_constants(&best);
        let ok = member.evaluate(&dataset, options, evaluator);
        if !ok {
            member.expr.set_constants(&orig_consts);
            member.birth = orig_birth;
            member.loss = orig_loss;
            member.cost = orig_cost;
            return (false, n_evals);
        }
        n_evals += eval_fraction;
        member.birth = get_birth_order(options.deterministic);
        (true, n_evals)
    } else {
        member.expr.set_constants(&orig_consts);
        member.birth = orig_birth;
        member.loss = orig_loss;
        member.cost = orig_cost;
        (false, n_evals)
    }
}

pub struct OptimizeConstantsCtx<'a, 'd, T: Float, Ops, const D: usize> {
    pub dataset: TaggedDataset<'d, T>,
    pub options: &'a Options<T, Ops, D>,
    pub evaluator: &'a mut Evaluator<T, D>,
    pub grad_ctx: &'a mut GradContext<T, D>,
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};

    use super::*;
    use crate::operator_library::OperatorLibrary;
    use crate::tests::common::{D, T, TestOps, infix};

    /// SymbolicRegression.jl optimizes constants on the loss without the dimensional penalty
    /// (`regularization=false`) and stores the member's loss with it.
    #[test]
    fn optimizer_objective_excludes_the_dimensional_penalty() {
        let x = Array2::from_shape_fn((1, 64), |(_, i)| i as T / 64.0);
        let y = x.row(0).mapv(|v| v + 3.0);
        let dataset = Dataset::new(x, Array1::from(y.to_vec()))
            .with_units(Some(&["m"]), Some("s"))
            .unwrap();
        let options = Options::<T, TestOps, D> {
            operators: OperatorLibrary::sr_default::<TestOps, D>(),
            deterministic: true,
            use_baseline: false,
            ..Default::default()
        };
        let mut member = PopMember::from_expr(infix::<TestOps, D>(&options.operators.set, "x1 + 1.0"), 1, &options);
        let mut evaluator = Evaluator::<T, D>::new(dataset.n_rows);
        let mut grad_ctx = GradContext::<T, D>::new(dataset.n_rows);
        let tagged = TaggedDataset::new(&dataset, None);

        let mut workspace = EvalWorkspace::new(tagged, &options, &mut evaluator, &mut grad_ctx);
        assert_eq!(workspace.loss_only(&member.plans, &member.expr), Some(4.0));
        assert!(member.evaluate(&tagged, &options, &mut evaluator));
        assert_eq!(member.loss, 1004.0);

        let ctx = OptimizeConstantsCtx {
            dataset: tagged,
            options: &options,
            evaluator: &mut evaluator,
            grad_ctx: &mut grad_ctx,
        };
        let (improved, _) = optimize_constants(&mut Rng::with_seed(0), &mut member, ctx);
        assert!(improved);
        assert!((member.loss - 1000.0).abs() < 1e-9, "{}", member.loss);
    }
}
