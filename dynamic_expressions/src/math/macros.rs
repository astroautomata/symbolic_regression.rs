macro_rules! unary_wrappers {
    ($( $fname:ident => $Op:ty ),* $(,)?) => {
        $(
            #[inline]
            #[must_use]
            pub fn $fname<T, Ops, const D: usize>(
                x: crate::expr::PostfixExpr<T, Ops, D>,
            ) -> crate::expr::PostfixExpr<T, Ops, D>
            where
                Ops: crate::operators::scalar::HasOp<$Op, 1>,
            {
                crate::algebra::__apply_postfix::<T, Ops, D, 1>(
                    <Ops as crate::operators::scalar::HasOp<$Op, 1>>::ID,
                    [x],
                )
            }
        )*
    };
}

macro_rules! binary_wrappers {
    ($( $fname:ident => $Op:ty ),* $(,)?) => {
        $(
            #[inline]
            #[must_use]
            pub fn $fname<T, Ops, const D: usize>(
                x: crate::expr::PostfixExpr<T, Ops, D>,
                y: crate::expr::PostfixExpr<T, Ops, D>,
            ) -> crate::expr::PostfixExpr<T, Ops, D>
            where
                Ops: crate::operators::scalar::HasOp<$Op, 2>,
            {
                crate::algebra::__apply_postfix::<T, Ops, D, 2>(
                    <Ops as crate::operators::scalar::HasOp<$Op, 2>>::ID,
                    [x, y],
                )
            }
        )*
    };
}

macro_rules! ternary_wrappers {
    ($( $fname:ident => $Op:ty ),* $(,)?) => {
        $(
            #[inline]
            #[must_use]
            pub fn $fname<T, Ops, const D: usize>(
                x: crate::expr::PostfixExpr<T, Ops, D>,
                y: crate::expr::PostfixExpr<T, Ops, D>,
                z: crate::expr::PostfixExpr<T, Ops, D>,
            ) -> crate::expr::PostfixExpr<T, Ops, D>
            where
                Ops: crate::operators::scalar::HasOp<$Op, 3>,
            {
                crate::algebra::__apply_postfix::<T, Ops, D, 3>(
                    <Ops as crate::operators::scalar::HasOp<$Op, 3>>::ID,
                    [x, y, z],
                )
            }
        )*
    };
}

pub(crate) use {binary_wrappers, ternary_wrappers, unary_wrappers};
