use crate::eval::EvalOptions;
use num_traits::Float;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct OpId {
    pub arity: u8,
    pub id: u16,
}

pub trait HasOp<Tag, const A: usize> {
    const ID: u16;
}

#[derive(Copy, Clone, Debug)]
pub enum SrcRef<'a, T> {
    Slice(&'a [T]),
    /// Strided view into a flat row-major matrix `data[row*stride + offset]`.
    Strided {
        data: &'a [T],
        offset: usize,
        stride: usize,
    },
    Const(T),
}

#[derive(Copy, Clone, Debug)]
pub enum GradRef<'a, T> {
    /// Dir-major gradient slab: `grad[dir*n_rows + row]`.
    Slice(&'a [T]),
    /// One-hot basis direction (value is 1 if `dir == basis_dir` else 0).
    Basis(usize),
    /// All zeros.
    Zero,
}

pub fn grad_at<T: Float>(g: GradRef<'_, T>, dir: usize, row: usize, n_rows: usize) -> T {
    match g {
        GradRef::Slice(s) => s[dir * n_rows + row],
        GradRef::Basis(k) => {
            if dir == k {
                T::one()
            } else {
                T::zero()
            }
        }
        GradRef::Zero => T::zero(),
    }
}

pub struct EvalKernelCtx<'a, 'b, T> {
    pub out: &'b mut [T],
    pub args: &'b [SrcRef<'a, T>],
    pub opts: &'b EvalOptions,
}

pub struct DiffKernelCtx<'a, 'b, T> {
    pub out_val: &'b mut [T],
    pub out_der: &'b mut [T],
    pub args: &'b [SrcRef<'a, T>],
    pub dargs: &'b [SrcRef<'a, T>],
    pub opts: &'b EvalOptions,
}

pub struct GradKernelCtx<'a, 'b, T> {
    pub out_val: &'b mut [T],
    /// Dir-major buffer: `out_grad[dir*n_rows + row]`.
    pub out_grad: &'b mut [T],
    pub args: &'b [SrcRef<'a, T>],
    pub arg_grads: &'b [GradRef<'a, T>],
    pub n_dir: usize,
    pub n_rows: usize,
    pub opts: &'b EvalOptions,
}

pub trait ScalarOpSet<T: Float> {
    fn eval(op: OpId, ctx: EvalKernelCtx<'_, '_, T>) -> bool;
    fn diff(op: OpId, ctx: DiffKernelCtx<'_, '_, T>) -> bool;
    fn grad(op: OpId, ctx: GradKernelCtx<'_, '_, T>) -> bool;
}

#[doc(hidden)]
pub fn __src_val<T: Float>(src: SrcRef<'_, T>, row: usize) -> T {
    match src {
        SrcRef::Slice(s) => s[row],
        SrcRef::Strided {
            data,
            offset,
            stride,
        } => data[row * stride + offset],
        SrcRef::Const(c) => c,
    }
}

#[doc(hidden)]
pub fn __maybe_mark_nonfinite<T: Float>(v: T, opts: &EvalOptions, complete: &mut bool) -> bool {
    if opts.check_finite && !v.is_finite() {
        *complete = false;
        if opts.early_exit {
            return false;
        }
    }
    true
}
