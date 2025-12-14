use crate::eval::EvalOptions;
use crate::operators::builtin::BuiltinOp;
use num_traits::Float;

use super::{grad_at, GradKernelCtx, SrcRef, __maybe_mark_nonfinite, __src_val};

fn __all_finite<T: Float>(vals: &[T]) -> bool {
    vals.iter().all(|v| v.is_finite())
}

pub fn eval_nary<const A: usize, T: Float>(
    eval: fn(&[T; A]) -> T,
    out: &mut [T],
    args: &[SrcRef<'_, T>],
    opts: &EvalOptions,
) -> bool {
    debug_assert_eq!(args.len(), A);
    let check_finite = opts.check_finite;
    let early_exit = opts.early_exit;
    let mut complete = true;

    if args.iter().all(|a| matches!(a, SrcRef::Const(_))) {
        let vals: [T; A] = core::array::from_fn(|j| __src_val(args[j], 0));
        let v = eval(&vals);
        out.fill(v);
        if !check_finite {
            return true;
        }
        return v.is_finite();
    }

    let mut vals: [T; A] = core::array::from_fn(|_| T::zero());
    if check_finite && early_exit {
        for (row, outv) in out.iter_mut().enumerate() {
            for (j, v) in vals.iter_mut().enumerate() {
                *v = __src_val(args[j], row);
            }
            let v = eval(&vals);
            if !__maybe_mark_nonfinite(v, opts, &mut complete) {
                *outv = v;
                return false;
            }
            *outv = v;
        }
        return complete;
    }

    for (row, outv) in out.iter_mut().enumerate() {
        for (j, v) in vals.iter_mut().enumerate() {
            *v = __src_val(args[j], row);
        }
        let v = eval(&vals);
        *outv = v;
    }
    if !check_finite {
        return true;
    }
    __all_finite(out)
}

pub fn eval_apply<const A: usize, T: Float, Op: BuiltinOp<T, A>>(
    out: &mut [T],
    args: &[SrcRef<'_, T>],
    opts: &EvalOptions,
) -> bool {
    debug_assert_eq!(args.len(), A);
    let check_finite = opts.check_finite;
    let early_exit = opts.early_exit;
    let mut complete = true;

    if args.iter().all(|a| matches!(a, SrcRef::Const(_))) {
        let vals: [T; A] = core::array::from_fn(|j| __src_val(args[j], 0));
        let v = Op::eval(&vals);
        out.fill(v);
        if !check_finite {
            return true;
        }
        if !v.is_finite() {
            complete = false;
        }
        return complete;
    }

    let mut vals: [T; A] = core::array::from_fn(|_| T::zero());
    if check_finite && early_exit {
        for (row, outv) in out.iter_mut().enumerate() {
            for (j, v) in vals.iter_mut().enumerate() {
                *v = __src_val(args[j], row);
            }
            let v = Op::eval(&vals);
            if !__maybe_mark_nonfinite(v, opts, &mut complete) {
                *outv = v;
                return false;
            }
            *outv = v;
        }
        return complete;
    }

    for (row, outv) in out.iter_mut().enumerate() {
        for (j, v) in vals.iter_mut().enumerate() {
            *v = __src_val(args[j], row);
        }
        let v = Op::eval(&vals);
        *outv = v;
    }
    if !check_finite {
        return true;
    }
    __all_finite(out)
}

pub fn diff_nary<const A: usize, T: Float + core::ops::AddAssign>(
    eval: fn(&[T; A]) -> T,
    partial: fn(&[T; A], usize) -> T,
    out_val: &mut [T],
    out_der: &mut [T],
    args: &[SrcRef<'_, T>],
    dargs: &[SrcRef<'_, T>],
    opts: &EvalOptions,
) -> bool {
    debug_assert_eq!(args.len(), A);
    debug_assert_eq!(dargs.len(), A);
    let check_finite = opts.check_finite;
    let early_exit = opts.early_exit;
    let mut complete = true;

    let mut vals: [T; A] = core::array::from_fn(|_| T::zero());
    let mut dvals: [T; A] = core::array::from_fn(|_| T::zero());

    if check_finite && early_exit {
        for ((row, outv), outd) in out_val.iter_mut().enumerate().zip(out_der.iter_mut()) {
            for (v, src) in vals.iter_mut().zip(args.iter().copied()) {
                *v = __src_val(src, row);
            }
            for (dv, dsrc) in dvals.iter_mut().zip(dargs.iter().copied()) {
                *dv = __src_val(dsrc, row);
            }
            let v = eval(&vals);
            let mut d = T::zero();
            for (j, dv) in dvals.iter().enumerate() {
                d += partial(&vals, j) * *dv;
            }
            if !__maybe_mark_nonfinite(v, opts, &mut complete) {
                *outv = v;
                *outd = d;
                return false;
            }
            *outv = v;
            *outd = d;
        }
        return complete;
    }

    for ((row, outv), outd) in out_val.iter_mut().enumerate().zip(out_der.iter_mut()) {
        for (v, src) in vals.iter_mut().zip(args.iter().copied()) {
            *v = __src_val(src, row);
        }
        for (dv, dsrc) in dvals.iter_mut().zip(dargs.iter().copied()) {
            *dv = __src_val(dsrc, row);
        }
        let v = eval(&vals);
        let mut d = T::zero();
        for (j, dv) in dvals.iter().enumerate() {
            d += partial(&vals, j) * *dv;
        }
        *outv = v;
        *outd = d;
    }

    if !check_finite {
        return true;
    }
    __all_finite(out_val)
}

pub fn diff_apply<const A: usize, T: Float + core::ops::AddAssign, Op: BuiltinOp<T, A>>(
    out_val: &mut [T],
    out_der: &mut [T],
    args: &[SrcRef<'_, T>],
    dargs: &[SrcRef<'_, T>],
    opts: &EvalOptions,
) -> bool {
    debug_assert_eq!(args.len(), A);
    debug_assert_eq!(dargs.len(), A);
    let check_finite = opts.check_finite;
    let early_exit = opts.early_exit;
    let mut complete = true;

    let mut vals: [T; A] = core::array::from_fn(|_| T::zero());
    let mut dvals: [T; A] = core::array::from_fn(|_| T::zero());

    if check_finite && early_exit {
        for ((row, outv), outd) in out_val.iter_mut().enumerate().zip(out_der.iter_mut()) {
            for (v, src) in vals.iter_mut().zip(args.iter().copied()) {
                *v = __src_val(src, row);
            }
            for (dv, dsrc) in dvals.iter_mut().zip(dargs.iter().copied()) {
                *dv = __src_val(dsrc, row);
            }
            let v = Op::eval(&vals);
            let mut d = T::zero();
            for (j, dv) in dvals.iter().enumerate() {
                d += Op::partial(&vals, j) * *dv;
            }
            if !__maybe_mark_nonfinite(v, opts, &mut complete) {
                *outv = v;
                *outd = d;
                return false;
            }
            *outv = v;
            *outd = d;
        }
        return complete;
    }

    for ((row, outv), outd) in out_val.iter_mut().enumerate().zip(out_der.iter_mut()) {
        for (v, src) in vals.iter_mut().zip(args.iter().copied()) {
            *v = __src_val(src, row);
        }
        for (dv, dsrc) in dvals.iter_mut().zip(dargs.iter().copied()) {
            *dv = __src_val(dsrc, row);
        }
        let v = Op::eval(&vals);
        let mut d = T::zero();
        for (j, dv) in dvals.iter().enumerate() {
            d += Op::partial(&vals, j) * *dv;
        }
        *outv = v;
        *outd = d;
    }

    if !check_finite {
        return true;
    }
    __all_finite(out_val)
}

pub fn grad_nary<const A: usize, T: Float + core::ops::AddAssign>(
    eval: fn(&[T; A]) -> T,
    partial: fn(&[T; A], usize) -> T,
    ctx: GradKernelCtx<'_, '_, T>,
) -> bool {
    debug_assert_eq!(ctx.args.len(), A);
    debug_assert_eq!(ctx.arg_grads.len(), A);

    let check_finite = ctx.opts.check_finite;
    let early_exit = ctx.opts.early_exit;
    let mut complete = true;
    let mut vals: [T; A] = core::array::from_fn(|_| T::zero());

    if check_finite && early_exit {
        for (row, outv) in ctx.out_val.iter_mut().enumerate() {
            for (v, src) in vals.iter_mut().zip(ctx.args.iter().copied()) {
                *v = __src_val(src, row);
            }
            let v = eval(&vals);
            if !__maybe_mark_nonfinite(v, ctx.opts, &mut complete) {
                *outv = v;
                ctx.out_grad.fill(T::nan());
                return false;
            }
            *outv = v;
        }
    } else {
        for (row, outv) in ctx.out_val.iter_mut().enumerate() {
            for (v, src) in vals.iter_mut().zip(ctx.args.iter().copied()) {
                *v = __src_val(src, row);
            }
            let v = eval(&vals);
            *outv = v;
        }
    }

    for (dir, grad_dir) in ctx
        .out_grad
        .chunks_mut(ctx.n_rows)
        .enumerate()
        .take(ctx.n_dir)
    {
        for (row, outg) in grad_dir.iter_mut().enumerate() {
            for (v, src) in vals.iter_mut().zip(ctx.args.iter().copied()) {
                *v = __src_val(src, row);
            }
            let mut g = T::zero();
            for (j, ag) in ctx.arg_grads.iter().copied().enumerate() {
                g += partial(&vals, j) * grad_at(ag, dir, row, ctx.n_rows);
            }
            *outg = g;
        }
    }

    if !check_finite {
        return true;
    }
    if early_exit {
        return complete;
    }
    __all_finite(ctx.out_val)
}

pub fn grad_apply<const A: usize, T: Float + core::ops::AddAssign, Op: BuiltinOp<T, A>>(
    ctx: GradKernelCtx<'_, '_, T>,
) -> bool {
    debug_assert_eq!(ctx.args.len(), A);
    debug_assert_eq!(ctx.arg_grads.len(), A);

    let check_finite = ctx.opts.check_finite;
    let early_exit = ctx.opts.early_exit;
    let mut complete = true;
    let mut vals: [T; A] = core::array::from_fn(|_| T::zero());

    if check_finite && early_exit {
        for (row, outv) in ctx.out_val.iter_mut().enumerate() {
            for (v, src) in vals.iter_mut().zip(ctx.args.iter().copied()) {
                *v = __src_val(src, row);
            }
            let v = Op::eval(&vals);
            if !__maybe_mark_nonfinite(v, ctx.opts, &mut complete) {
                *outv = v;
                ctx.out_grad.fill(T::nan());
                return false;
            }
            *outv = v;
        }
    } else {
        for (row, outv) in ctx.out_val.iter_mut().enumerate() {
            for (v, src) in vals.iter_mut().zip(ctx.args.iter().copied()) {
                *v = __src_val(src, row);
            }
            let v = Op::eval(&vals);
            *outv = v;
        }
    }

    for (dir, grad_dir) in ctx
        .out_grad
        .chunks_mut(ctx.n_rows)
        .enumerate()
        .take(ctx.n_dir)
    {
        for (row, outg) in grad_dir.iter_mut().enumerate() {
            for (v, src) in vals.iter_mut().zip(ctx.args.iter().copied()) {
                *v = __src_val(src, row);
            }
            let mut g = T::zero();
            for (j, ag) in ctx.arg_grads.iter().copied().enumerate() {
                g += Op::partial(&vals, j) * grad_at(ag, dir, row, ctx.n_rows);
            }
            *outg = g;
        }
    }

    if !check_finite {
        return true;
    }
    if early_exit {
        return complete;
    }
    __all_finite(ctx.out_val)
}
