use crate::compile::{compile_plan, EvalPlan};
use crate::evaluate::{resolve_der_src, resolve_grad_src, resolve_val_src, EvalOptions};
use crate::expression::PostfixExpr;
use crate::node::Src;
use crate::operator_enum::scalar::{
    DiffKernelCtx, GradKernelCtx, GradRef, OpId, ScalarOpSet, SrcRef,
};
use crate::subtree_caching::SubtreeCache;
use ndarray::ArrayView2;
use num_traits::Float;

#[derive(Debug)]
pub struct DiffContext<T: Float, const D: usize> {
    pub val_scratch: Vec<Vec<T>>,
    pub der_scratch: Vec<Vec<T>>,
    pub n_rows: usize,
    pub plan: Option<EvalPlan<D>>,
    pub plan_nodes_len: usize,
    pub plan_n_consts: usize,
    pub plan_n_features: usize,
}

impl<T: Float, const D: usize> DiffContext<T, D> {
    pub fn new(n_rows: usize) -> Self {
        Self {
            val_scratch: Vec::new(),
            der_scratch: Vec::new(),
            n_rows,
            plan: None,
            plan_nodes_len: 0,
            plan_n_consts: 0,
            plan_n_features: 0,
        }
    }

    pub fn ensure_scratch(&mut self, n_slots: usize) {
        if self.val_scratch.len() < n_slots {
            self.val_scratch.resize_with(n_slots, Vec::new);
        }
        if self.der_scratch.len() < n_slots {
            self.der_scratch.resize_with(n_slots, Vec::new);
        }
        for slot in &mut self.val_scratch[..n_slots] {
            if slot.len() != self.n_rows {
                slot.resize(self.n_rows, T::zero());
            }
        }
        for slot in &mut self.der_scratch[..n_slots] {
            if slot.len() != self.n_rows {
                slot.resize(self.n_rows, T::zero());
            }
        }
    }
}

#[derive(Debug)]
pub struct GradContext<T: Float, const D: usize> {
    pub val_scratch: Vec<Vec<T>>,
    pub grad_scratch: Vec<Vec<T>>, // slot-major, each len n_dir*n_rows
    pub n_rows: usize,
    pub plan: Option<EvalPlan<D>>,
    pub plan_nodes_len: usize,
    pub plan_n_consts: usize,
    pub plan_n_features: usize,
}

impl<T: Float, const D: usize> GradContext<T, D> {
    pub fn new(n_rows: usize) -> Self {
        Self {
            val_scratch: Vec::new(),
            grad_scratch: Vec::new(),
            n_rows,
            plan: None,
            plan_nodes_len: 0,
            plan_n_consts: 0,
            plan_n_features: 0,
        }
    }

    pub fn ensure_scratch(&mut self, n_slots: usize, n_dir: usize) {
        if self.val_scratch.len() < n_slots {
            self.val_scratch.resize_with(n_slots, Vec::new);
        }
        if self.grad_scratch.len() < n_slots {
            self.grad_scratch.resize_with(n_slots, Vec::new);
        }
        for slot in &mut self.val_scratch[..n_slots] {
            if slot.len() != self.n_rows {
                slot.resize(self.n_rows, T::zero());
            }
        }
        let grad_len = n_dir * self.n_rows;
        for slot in &mut self.grad_scratch[..n_slots] {
            if slot.len() != grad_len {
                slot.resize(grad_len, T::zero());
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct GradMatrix<T> {
    pub data: Vec<T>,
    pub n_dir: usize,
    pub n_rows: usize,
}

fn nan_grad_return<T: Float>(n_rows: usize, n_dir: usize) -> (Vec<T>, GradMatrix<T>, bool) {
    (
        vec![T::nan(); n_rows],
        GradMatrix {
            data: vec![T::nan(); n_dir * n_rows],
            n_dir,
            n_rows,
        },
        false,
    )
}

pub fn eval_diff_tree_array<T, Ops, const D: usize>(
    expr: &PostfixExpr<T, Ops, D>,
    x: ArrayView2<'_, T>,
    direction: usize,
    ctx: &mut DiffContext<T, D>,
    opts: &EvalOptions,
) -> (Vec<T>, Vec<T>, bool)
where
    T: Float,
    Ops: ScalarOpSet<T>,
{
    assert!(
        x.is_standard_layout(),
        "X must be standard (row-major) layout"
    );
    assert!(direction < x.ncols());
    assert_eq!(ctx.n_rows, x.nrows());
    let n_rows = x.nrows();
    let x_data = x.as_slice().expect("X must be contiguous");
    let n_features = x.ncols();

    let needs_recompile = ctx.plan.is_none()
        || ctx.plan_nodes_len != expr.nodes.len()
        || ctx.plan_n_consts != expr.consts.len()
        || ctx.plan_n_features != x.ncols();
    if needs_recompile {
        ctx.plan = Some(compile_plan::<D>(&expr.nodes, x.ncols(), expr.consts.len()));
        ctx.plan_nodes_len = expr.nodes.len();
        ctx.plan_n_consts = expr.consts.len();
        ctx.plan_n_features = x.ncols();
    }
    let n_slots = ctx.plan.as_ref().unwrap().n_slots;
    ctx.ensure_scratch(n_slots);
    let plan: &EvalPlan<D> = ctx.plan.as_ref().unwrap();

    let mut complete = true;

    for instr in plan.instrs.iter().copied() {
        let dst_slot = instr.dst as usize;
        let arity = instr.arity as usize;

        let (val_before, val_rest) = ctx.val_scratch.split_at_mut(dst_slot);
        let (dst_val, val_after) = val_rest.split_first_mut().unwrap();
        let (der_before, der_rest) = ctx.der_scratch.split_at_mut(dst_slot);
        let (dst_der, der_after) = der_rest.split_first_mut().unwrap();

        let mut args_refs: [SrcRef<'_, T>; D] = core::array::from_fn(|_| SrcRef::Const(T::zero()));
        let mut dargs_refs: [SrcRef<'_, T>; D] = core::array::from_fn(|_| SrcRef::Const(T::zero()));
        for (j, (dst_a, dst_da)) in args_refs
            .iter_mut()
            .take(arity)
            .zip(dargs_refs.iter_mut().take(arity))
            .enumerate()
        {
            *dst_a = resolve_val_src(
                instr.args[j],
                x_data,
                n_features,
                &expr.consts,
                dst_slot,
                val_before,
                val_after,
            );
            *dst_da = resolve_der_src(instr.args[j], direction, dst_slot, der_before, der_after);
        }

        let ok = Ops::diff(
            OpId {
                arity: instr.arity,
                id: instr.op,
            },
            DiffKernelCtx {
                out_val: dst_val,
                out_der: dst_der,
                args: &args_refs[..arity],
                dargs: &dargs_refs[..arity],
                opts,
            },
        );
        complete &= ok;
        if opts.early_exit && !ok {
            return (vec![T::nan(); n_rows], vec![T::nan(); n_rows], false);
        }
    }

    match plan.root {
        Src::Var(f) => {
            let eval = (0..n_rows)
                .map(|row| x_data[row * n_features + (f as usize)])
                .collect();
            let der = if f as usize == direction {
                vec![T::one(); n_rows]
            } else {
                vec![T::zero(); n_rows]
            };
            (eval, der, complete)
        }
        Src::Const(c) => {
            let v = expr.consts[c as usize];
            if opts.check_finite && !v.is_finite() {
                complete = false;
                if opts.early_exit {
                    return (vec![T::nan(); n_rows], vec![T::nan(); n_rows], false);
                }
            }
            (vec![v; n_rows], vec![T::zero(); n_rows], complete)
        }
        Src::Slot(s) => {
            let eval = ctx.val_scratch[s as usize].clone();
            let der = ctx.der_scratch[s as usize].clone();
            (eval, der, complete)
        }
    }
}

pub fn eval_grad_tree_array<T, Ops, const D: usize>(
    expr: &PostfixExpr<T, Ops, D>,
    x: ArrayView2<'_, T>,
    variable: bool,
    ctx: &mut GradContext<T, D>,
    opts: &EvalOptions,
) -> (Vec<T>, GradMatrix<T>, bool)
where
    T: Float,
    Ops: ScalarOpSet<T>,
{
    assert!(
        x.is_standard_layout(),
        "X must be standard (row-major) layout"
    );
    assert_eq!(ctx.n_rows, x.nrows());
    let n_rows = x.nrows();
    let x_data = x.as_slice().expect("X must be contiguous");
    let n_features = x.ncols();
    let n_dir = if variable {
        x.ncols()
    } else {
        expr.consts.len()
    };

    let needs_recompile = ctx.plan.is_none()
        || ctx.plan_nodes_len != expr.nodes.len()
        || ctx.plan_n_consts != expr.consts.len()
        || ctx.plan_n_features != x.ncols();
    if needs_recompile {
        ctx.plan = Some(compile_plan::<D>(&expr.nodes, x.ncols(), expr.consts.len()));
        ctx.plan_nodes_len = expr.nodes.len();
        ctx.plan_n_consts = expr.consts.len();
        ctx.plan_n_features = x.ncols();
    }
    let n_slots = ctx.plan.as_ref().unwrap().n_slots;
    ctx.ensure_scratch(n_slots, n_dir);
    let plan: &EvalPlan<D> = ctx.plan.as_ref().unwrap();

    let mut complete = true;

    for instr in plan.instrs.iter().copied() {
        let dst_slot = instr.dst as usize;
        let arity = instr.arity as usize;

        let (val_before, val_rest) = ctx.val_scratch.split_at_mut(dst_slot);
        let (dst_val, val_after) = val_rest.split_first_mut().unwrap();
        let (grad_before, grad_rest) = ctx.grad_scratch.split_at_mut(dst_slot);
        let (dst_grad, grad_after) = grad_rest.split_first_mut().unwrap();

        let mut args_refs: [SrcRef<'_, T>; D] = core::array::from_fn(|_| SrcRef::Const(T::zero()));
        let mut arg_grads: [GradRef<'_, T>; D] = core::array::from_fn(|_| GradRef::Zero);
        for (j, (dst_a, dst_g)) in args_refs
            .iter_mut()
            .take(arity)
            .zip(arg_grads.iter_mut().take(arity))
            .enumerate()
        {
            *dst_a = resolve_val_src(
                instr.args[j],
                x_data,
                n_features,
                &expr.consts,
                dst_slot,
                val_before,
                val_after,
            );
            *dst_g = resolve_grad_src(instr.args[j], variable, dst_slot, grad_before, grad_after);
        }

        dst_grad.fill(T::zero());

        let ok = Ops::grad(
            OpId {
                arity: instr.arity,
                id: instr.op,
            },
            GradKernelCtx {
                out_val: dst_val,
                out_grad: dst_grad,
                args: &args_refs[..arity],
                arg_grads: &arg_grads[..arity],
                n_dir,
                n_rows,
                opts,
            },
        );
        complete &= ok;
        if opts.early_exit && !ok {
            return nan_grad_return::<T>(n_rows, n_dir);
        }
    }

    match plan.root {
        Src::Var(f) => {
            let eval: Vec<T> = (0..n_rows)
                .map(|row| x_data[row * n_features + (f as usize)])
                .collect();
            let mut grad = vec![T::zero(); n_dir * n_rows];
            if variable {
                let dir = f as usize;
                debug_assert!(dir < n_dir);
                grad[dir * n_rows..(dir + 1) * n_rows].fill(T::one());
            }
            (
                eval,
                GradMatrix {
                    data: grad,
                    n_dir,
                    n_rows,
                },
                complete,
            )
        }
        Src::Const(c) => {
            let v = expr.consts[c as usize];
            let eval = vec![v; n_rows];
            let mut grad = vec![T::zero(); n_dir * n_rows];
            if !variable {
                let dir = c as usize;
                debug_assert!(dir < n_dir);
                grad[dir * n_rows..(dir + 1) * n_rows].fill(T::one());
            }
            if opts.check_finite && !v.is_finite() {
                complete = false;
                if opts.early_exit {
                    return nan_grad_return::<T>(n_rows, n_dir);
                }
            }
            (
                eval,
                GradMatrix {
                    data: grad,
                    n_dir,
                    n_rows,
                },
                complete,
            )
        }
        Src::Slot(s) => {
            let eval = ctx.val_scratch[s as usize].clone();
            let grad = ctx.grad_scratch[s as usize].clone();
            (
                eval,
                GradMatrix {
                    data: grad,
                    n_dir,
                    n_rows,
                },
                complete,
            )
        }
    }
}

fn fill_nan<T: Float>(buf: &mut [T]) {
    buf.fill(T::nan());
}

/// Evaluate value and gradient for a *compiled* plan, writing into preallocated buffers.
///
/// This is an allocation-free analogue of [`eval_grad_tree_array`].
///
/// `out_val` must have length `n_rows`, and `out_grad` must have length `n_dir * n_rows`,
/// where `n_dir = if variable { n_features } else { n_consts }`.
pub fn eval_grad_plan_array_into<T, Ops, const D: usize>(
    out_val: &mut [T],
    out_grad: &mut [T],
    plan: &EvalPlan<D>,
    expr: &PostfixExpr<T, Ops, D>,
    x: ArrayView2<'_, T>,
    variable: bool,
    ctx: &mut GradContext<T, D>,
    opts: &EvalOptions,
) -> bool
where
    T: Float,
    Ops: ScalarOpSet<T>,
{
    assert!(
        x.is_standard_layout(),
        "X must be standard (row-major) layout"
    );
    assert_eq!(out_val.len(), x.nrows());

    let n_rows = x.nrows();
    let n_features = x.ncols();
    let n_dir = if variable { n_features } else { expr.consts.len() };
    assert_eq!(out_grad.len(), n_dir * n_rows);

    let x_data = x.as_slice().expect("X must be contiguous");

    // Ensure scratch.
    ctx.ensure_scratch(plan.n_slots, n_dir);

    let mut complete = true;

    for instr in plan.instrs.iter().copied() {
        let dst_slot = instr.dst as usize;
        let arity = instr.arity as usize;
        let (val_before, val_rest) = ctx.val_scratch.split_at_mut(dst_slot);
        let (dst_val, val_after) = val_rest.split_first_mut().unwrap();

        let (grad_before, grad_rest) = ctx.grad_scratch.split_at_mut(dst_slot);
        let (dst_grad_buf, grad_after) = grad_rest.split_first_mut().unwrap();
        let dst_grad: &mut [T] = &mut dst_grad_buf[..n_dir * n_rows];
        dst_grad.fill(T::zero());

        let mut args_refs: [SrcRef<'_, T>; D] =
            core::array::from_fn(|_| SrcRef::Const(T::zero()));
        let mut arg_grads: [GradRef<'_, T>; D] = core::array::from_fn(|_| GradRef::Zero);

        for (j, (dst, gdst)) in args_refs
            .iter_mut()
            .zip(arg_grads.iter_mut())
            .take(arity)
            .enumerate()
        {
            *dst = resolve_val_src(
                instr.args[j],
                x_data,
                n_features,
                &expr.consts,
                dst_slot,
                val_before,
                val_after,
            );
            *gdst = resolve_grad_src(instr.args[j], variable, dst_slot, grad_before, grad_after);
        }

        let ok = Ops::grad(
            OpId {
                arity: instr.arity,
                id: instr.op,
            },
            GradKernelCtx {
                out_val: dst_val,
                out_grad: dst_grad,
                args: &args_refs[..arity],
                arg_grads: &arg_grads[..arity],
                n_dir,
                n_rows,
                opts,
            },
        );
        complete &= ok;
        if opts.early_exit && !ok {
            fill_nan(out_val);
            fill_nan(out_grad);
            return false;
        }
    }

    // Write root into outputs.
    match plan.root {
        Src::Var(f) => {
            let offset = f as usize;
            for row in 0..n_rows {
                out_val[row] = x_data[row * n_features + offset];
            }
            out_grad.fill(T::zero());
            if variable {
                let dir = offset;
                out_grad[dir * n_rows..(dir + 1) * n_rows].fill(T::one());
            }
        }
        Src::Const(c) => {
            let v = expr.consts[c as usize];
            out_val.fill(v);
            out_grad.fill(T::zero());
            if !variable {
                let dir = c as usize;
                debug_assert!(dir < n_dir);
                out_grad[dir * n_rows..(dir + 1) * n_rows].fill(T::one());
            }
            if opts.check_finite && !v.is_finite() {
                complete = false;
                if opts.early_exit {
                    fill_nan(out_val);
                    fill_nan(out_grad);
                    return false;
                }
            }
        }
        Src::Slot(s) => {
            out_val.copy_from_slice(&ctx.val_scratch[s as usize]);
            out_grad.copy_from_slice(&ctx.grad_scratch[s as usize][..n_dir * n_rows]);
        }
    }

    complete
}

/// Like [`eval_grad_plan_array_into`], but reuses cached values for constant-free subtrees.
///
/// When `variable == false` (gradients w.r.t. constants), any subtree that does not depend on
/// constants has *zero* gradient. In this case we can reuse cached values and skip calling
/// expensive operator kernels for those subtrees.
///
/// `dataset_key` must identify the dataset contents; changing it invalidates the cache.
pub fn eval_grad_plan_array_into_cached<T, Ops, const D: usize>(
    out_val: &mut [T],
    out_grad: &mut [T],
    plan: &EvalPlan<D>,
    expr: &PostfixExpr<T, Ops, D>,
    x: ArrayView2<'_, T>,
    variable: bool,
    ctx: &mut GradContext<T, D>,
    opts: &EvalOptions,
    cache: &mut SubtreeCache<T>,
    dataset_key: u64,
) -> bool
where
    T: Float,
    Ops: ScalarOpSet<T>,
{
    // Only safe/meaningful when taking gradients w.r.t. constants.
    if variable {
        return eval_grad_plan_array_into::<T, Ops, D>(
            out_val, out_grad, plan, expr, x, variable, ctx, opts,
        );
    }

    assert!(
        x.is_standard_layout(),
        "X must be standard (row-major) layout"
    );
    assert_eq!(out_val.len(), x.nrows());

    let n_rows = x.nrows();
    let n_features = x.ncols();
    let n_dir = expr.consts.len();
    assert_eq!(out_grad.len(), n_dir * n_rows);

    let x_data = x.as_slice().expect("X must be contiguous");

    ctx.ensure_scratch(plan.n_slots, n_dir);
    cache.ensure(&expr.nodes, dataset_key, n_rows);

    let mut complete = true;

    for (instr_index, instr) in plan.instrs.iter().copied().enumerate() {
        let dst_slot = instr.dst as usize;
        let arity = instr.arity as usize;

        let (val_before, val_rest) = ctx.val_scratch.split_at_mut(dst_slot);
        let (dst_val, val_after) = val_rest.split_first_mut().unwrap();

        let (grad_before, grad_rest) = ctx.grad_scratch.split_at_mut(dst_slot);
        let (dst_grad_buf, grad_after) = grad_rest.split_first_mut().unwrap();
        let dst_grad: &mut [T] = &mut dst_grad_buf[..n_dir * n_rows];

        // Cached constant-free subtree: value from cache, grad = 0.
        if let Some(cached) = cache.get(instr_index) {
            dst_val.copy_from_slice(&cached.val);
            dst_grad.fill(T::zero());
            if opts.check_finite && !cached.all_finite {
                complete = false;
                if opts.early_exit {
                    fill_nan(out_val);
                    fill_nan(out_grad);
                    return false;
                }
            }
            continue;
        }

        dst_grad.fill(T::zero());

        let mut args_refs: [SrcRef<'_, T>; D] =
            core::array::from_fn(|_| SrcRef::Const(T::zero()));
        let mut arg_grads: [GradRef<'_, T>; D] = core::array::from_fn(|_| GradRef::Zero);

        for (j, (dst, gdst)) in args_refs
            .iter_mut()
            .zip(arg_grads.iter_mut())
            .take(arity)
            .enumerate()
        {
            *dst = resolve_val_src(
                instr.args[j],
                x_data,
                n_features,
                &expr.consts,
                dst_slot,
                val_before,
                val_after,
            );
            *gdst = resolve_grad_src(instr.args[j], variable, dst_slot, grad_before, grad_after);
        }

        let ok = Ops::grad(
            OpId {
                arity: instr.arity,
                id: instr.op,
            },
            GradKernelCtx {
                out_val: dst_val,
                out_grad: dst_grad,
                args: &args_refs[..arity],
                arg_grads: &arg_grads[..arity],
                n_dir,
                n_rows,
                opts,
            },
        );
        complete &= ok;
        if opts.early_exit && !ok {
            fill_nan(out_val);
            fill_nan(out_grad);
            return false;
        }

        // Populate cache lazily.
        cache.maybe_store(instr_index, dst_val);
    }

    match plan.root {
        Src::Var(f) => {
            let offset = f as usize;
            for row in 0..n_rows {
                out_val[row] = x_data[row * n_features + offset];
            }
            out_grad.fill(T::zero());
        }
        Src::Const(c) => {
            let v = expr.consts[c as usize];
            out_val.fill(v);
            out_grad.fill(T::zero());
            let dir = c as usize;
            debug_assert!(dir < n_dir);
            out_grad[dir * n_rows..(dir + 1) * n_rows].fill(T::one());
            if opts.check_finite && !v.is_finite() {
                complete = false;
                if opts.early_exit {
                    fill_nan(out_val);
                    fill_nan(out_grad);
                    return false;
                }
            }
        }
        Src::Slot(s) => {
            out_val.copy_from_slice(&ctx.val_scratch[s as usize]);
            out_grad.copy_from_slice(&ctx.grad_scratch[s as usize][..n_dir * n_rows]);
        }
    }

    complete
}
