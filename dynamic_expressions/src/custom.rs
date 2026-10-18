use num_traits::Float;

use crate::dispatch::{DiffKernelCtx, EvalKernelCtx, GradKernelCtx};
use crate::traits::OpMeta;

/// Runtime operators appended after an operator set's built-ins. `local` ids start at 0; the
/// operator set adds its built-in count to form the global id.
pub trait CustomTable<T: Float>: Clone + Default + core::fmt::Debug + Send + Sync {
    /// Guarantees that every entry returned by `meta` has complexity 1.
    const UNIT_COMPLEXITY: bool = false;
    fn max_arity(&self) -> u8;
    fn ids_with_arity(&self, arity: u8) -> impl Iterator<Item = u16> + '_;
    /// `id` in the returned metadata is the local id.
    fn meta(&self, local: u16) -> Option<OpMeta<'_>>;
    fn eval(&self, local: u16, ctx: EvalKernelCtx<'_, '_, T>) -> bool;
    fn diff(&self, local: u16, ctx: DiffKernelCtx<'_, '_, T>) -> bool;
    fn grad(&self, local: u16, ctx: GradKernelCtx<'_, '_, T>) -> bool;
}

/// The empty table. Operator sets without runtime operators use it and stay zero-sized.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct NoCustom;

impl<T: Float> CustomTable<T> for NoCustom {
    const UNIT_COMPLEXITY: bool = true;

    #[inline]
    fn max_arity(&self) -> u8 {
        0
    }

    #[inline]
    fn ids_with_arity(&self, _arity: u8) -> impl Iterator<Item = u16> + '_ {
        core::iter::empty()
    }

    #[inline]
    fn meta(&self, _local: u16) -> Option<OpMeta<'_>> {
        None
    }

    #[inline]
    fn eval(&self, local: u16, _ctx: EvalKernelCtx<'_, '_, T>) -> bool {
        panic!("unknown custom op {local}")
    }

    #[inline]
    fn diff(&self, local: u16, _ctx: DiffKernelCtx<'_, '_, T>) -> bool {
        panic!("unknown custom op {local}")
    }

    #[inline]
    fn grad(&self, local: u16, _ctx: GradKernelCtx<'_, '_, T>) -> bool {
        panic!("unknown custom op {local}")
    }
}
