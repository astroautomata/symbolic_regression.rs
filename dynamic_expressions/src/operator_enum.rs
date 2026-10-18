// -------------------------------------------------------------------------------------------------
// Operator definition macro
// -------------------------------------------------------------------------------------------------

/// Define an operator marker type implementing [`crate::traits::Operator`] + [`crate::traits::OpTag`].
///
/// This is the same DSL used for builtins in `operator_enum::builtin`, but exported so downstream
/// crates can define their own operators with identical syntax.
#[macro_export]
macro_rules! op {
    (@count) => { 0usize };
    (@count $head:tt $(, $tail:tt)*) => { 1usize + $crate::op!(@count $($tail),*) };

    (@maybe $x:expr) => { Some($x) };
    (@maybe) => { None };

    (@name $Op:ident, $name:literal) => { $name };
    (@name $Op:ident) => { $crate::paste::paste! { stringify!([<$Op:snake>]) } };

    (@infix $v:expr) => { Some($v) };
    (@infix) => { None };
    (@commutative $v:expr) => { $v };
    (@commutative) => { false };
    (@associative $v:expr) => { $v };
    (@associative) => { false };
    (@complexity $v:expr) => { $v };
    (@complexity) => { 1u16 };

    (
        $(#[$meta:meta])*
        $Op:ident for<$T:ident> { $($body:tt)* }
    ) => {
        $crate::op!(
            $(#[$meta])*
            pub $Op for<$T> { $($body)* }
        );
    };

    (
        $(#[$meta:meta])*
        $vis:vis $Op:ident for<$T:ident> {
            $(name: $name:literal,)?
            $(display: $display:expr,)?
            $(infix: $infix:expr,)?
            $(aliases: [$($alias:expr),* $(,)?],)?
            $(commutative: $commutative:expr,)?
            $(associative: $associative:expr,)?
            $(complexity: $complexity:expr,)?
            eval: |[$($eval_arg:pat_param),+ $(,)?]| $eval_body:block,
            partial: |[$($partial_arg:pat_param),+ $(,)?], $idx:pat_param| $partial_body:block $(,)?
        }
    ) => {
        $(#[$meta])*
        $vis struct $Op;

        impl $crate::traits::OpTag for $Op {
            const ARITY: u8 = $crate::op!(@count $($eval_arg),+) as u8;
        }

        impl<$T: $crate::num_traits::Float> $crate::traits::Operator<$T, { $crate::op!(@count $($eval_arg),+) }> for $Op {
            const NAME: &'static str = $crate::op!(@name $Op $(, $name)?);
            const INFIX: Option<&'static str> = $crate::op!(@infix $($infix)?);
            const DISPLAY: &'static str = match ($crate::op!(@maybe $( $display )?), $crate::op!(@infix $($infix)?)) {
                (Some(d), _) => d,
                (None, Some(i)) => i,
                (None, None) => $crate::op!(@name $Op $(, $name)?),
            };
            const ALIASES: &'static [&'static str] = &[$($($alias),*,)?];
            const COMMUTATIVE: bool = $crate::op!(@commutative $($commutative)?);
            const ASSOCIATIVE: bool = $crate::op!(@associative $($associative)?);
            const COMPLEXITY: u16 = $crate::op!(@complexity $($complexity)?);

            #[inline]
            fn eval(args: &[$T; { $crate::op!(@count $($eval_arg),+) }]) -> $T {
                let [$($eval_arg),+] = *args;
                $eval_body
            }

            #[inline]
            fn partial(args: &[$T; { $crate::op!(@count $($partial_arg),+) }], idx: usize) -> $T {
                let [$($partial_arg),+] = *args;
                let $idx = idx;
                $partial_body
            }
        }
    };

    (
        $(#[$meta:meta])*
        $Op:ident for $t:ty { $($body:tt)* }
    ) => {
        $crate::op!(
            $(#[$meta])*
            pub $Op for $t { $($body)* }
        );
    };

    (
        $(#[$meta:meta])*
        $vis:vis $Op:ident for $t:ty {
            $(name: $name:literal,)?
            $(display: $display:expr,)?
            $(infix: $infix:expr,)?
            $(aliases: [$($alias:expr),* $(,)?],)?
            $(commutative: $commutative:expr,)?
            $(associative: $associative:expr,)?
            $(complexity: $complexity:expr,)?
            eval: |[$($eval_arg:pat_param),+ $(,)?]| $eval_body:block,
            partial: |[$($partial_arg:pat_param),+ $(,)?], $idx:pat_param| $partial_body:block $(,)?
        }
    ) => {
        $(#[$meta])*
        $vis struct $Op;

        impl $crate::traits::OpTag for $Op {
            const ARITY: u8 = $crate::op!(@count $($eval_arg),+) as u8;
        }

        impl $crate::traits::Operator<$t, { $crate::op!(@count $($eval_arg),+) }> for $Op {
            const NAME: &'static str = $crate::op!(@name $Op $(, $name)?);
            const INFIX: Option<&'static str> = $crate::op!(@infix $($infix)?);
            const DISPLAY: &'static str = match ($crate::op!(@maybe $( $display )?), $crate::op!(@infix $($infix)?)) {
                (Some(d), _) => d,
                (None, Some(i)) => i,
                (None, None) => $crate::op!(@name $Op $(, $name)?),
            };
            const ALIASES: &'static [&'static str] = &[$($($alias),*,)?];
            const COMMUTATIVE: bool = $crate::op!(@commutative $($commutative)?);
            const ASSOCIATIVE: bool = $crate::op!(@associative $($associative)?);
            const COMPLEXITY: u16 = $crate::op!(@complexity $($complexity)?);

            #[inline]
            fn eval(args: &[$t; { $crate::op!(@count $($eval_arg),+) }]) -> $t {
                let [$($eval_arg),+] = *args;
                $eval_body
            }

            #[inline]
            fn partial(args: &[$t; { $crate::op!(@count $($partial_arg),+) }], idx: usize) -> $t {
                let [$($partial_arg),+] = *args;
                let $idx = idx;
                $partial_body
            }
        }
    };
}

pub mod builtin {
    use num_traits::Float;

    fn two<T: Float>() -> T {
        T::one() + T::one()
    }

    fn three<T: Float>() -> T {
        T::one() + T::one() + T::one()
    }

    fn half<T: Float>() -> T {
        T::from(0.5).expect("Float can represent 0.5")
    }

    crate::op!(Cos for<T> {
        eval: |[x]| { x.cos() },
        partial: |[x], _idx| { -x.sin() },
    });

    crate::op!(Sin for<T> {
        eval: |[x]| { x.sin() },
        partial: |[x], _idx| { x.cos() },
    });

    crate::op!(Tan for<T> {
        eval: |[x]| { x.tan() },
        partial: |[x], _idx| {
            let c = x.cos();
            T::one() / (c * c)
        },
    });

    crate::op!(Asin for<T> {
        eval: |[x]| { x.asin() },
        partial: |[x], _idx| { T::one() / (T::one() - x * x).sqrt() },
    });

    crate::op!(Acos for<T> {
        eval: |[x]| { x.acos() },
        partial: |[x], _idx| { -T::one() / (T::one() - x * x).sqrt() },
    });

    crate::op!(Atan for<T> {
        eval: |[x]| { x.atan() },
        partial: |[x], _idx| { T::one() / (T::one() + x * x) },
    });

    crate::op!(Sinh for<T> {
        eval: |[x]| { x.sinh() },
        partial: |[x], _idx| { x.cosh() },
    });

    crate::op!(Cosh for<T> {
        eval: |[x]| { x.cosh() },
        partial: |[x], _idx| { x.sinh() },
    });

    crate::op!(Tanh for<T> {
        eval: |[x]| { x.tanh() },
        partial: |[x], _idx| {
            let c = x.cosh();
            T::one() / (c * c)
        },
    });

    crate::op!(Asinh for<T> {
        eval: |[x]| { x.asinh() },
        partial: |[x], _idx| { T::one() / (x * x + T::one()).sqrt() },
    });

    crate::op!(Acosh for<T> {
        eval: |[x]| { x.acosh() },
        partial: |[x], _idx| { T::one() / ((x - T::one()).sqrt() * (x + T::one()).sqrt()) },
    });

    crate::op!(Atanh for<T> {
        eval: |[x]| { x.atanh() },
        partial: |[x], _idx| { T::one() / (T::one() - x * x) },
    });

    crate::op!(Sec for<T> {
        eval: |[x]| { T::one() / x.cos() },
        partial: |[x], _idx| { (T::one() / x.cos()) * x.tan() },
    });

    crate::op!(Csc for<T> {
        eval: |[x]| { T::one() / x.sin() },
        partial: |[x], _idx| {
            let csc = T::one() / x.sin();
            let cot = T::one() / x.tan();
            -csc * cot
        },
    });

    crate::op!(Cot for<T> {
        eval: |[x]| { T::one() / x.tan() },
        partial: |[x], _idx| {
            let s = x.sin();
            -T::one() / (s * s)
        },
    });

    crate::op!(Exp for<T> {
        eval: |[x]| { x.exp() },
        partial: |[x], _idx| { x.exp() },
    });

    crate::op!(Log for<T> {
        eval: |[x]| { if x <= T::zero() { T::nan() } else { x.ln() } },
        partial: |[x], _idx| { T::one() / x },
    });

    crate::op!(Log2 for<T> {
        eval: |[x]| { if x <= T::zero() { T::nan() } else { x.log2() } },
        partial: |[x], _idx| { T::one() / (x * two::<T>().ln()) },
    });

    crate::op!(Log10 for<T> {
        eval: |[x]| { if x <= T::zero() { T::nan() } else { x.log10() } },
        partial: |[x], _idx| {
            let ten = T::from(10.0).expect("Float can represent 10.0");
            T::one() / (x * ten.ln())
        },
    });

    crate::op!(Log1p for<T> {
        eval: |[x]| { if x <= -T::one() { T::nan() } else { x.ln_1p() } },
        partial: |[x], _idx| { T::one() / (T::one() + x) },
    });

    crate::op!(Exp2 for<T> {
        eval: |[x]| { x.exp2() },
        partial: |[x], _idx| { x.exp2() * two::<T>().ln() },
    });

    crate::op!(Expm1 for<T> {
        eval: |[x]| { x.exp_m1() },
        partial: |[x], _idx| { x.exp() },
    });

    crate::op!(Sqrt for<T> {
        eval: |[x]| { x.sqrt() },
        partial: |[x], _idx| { T::one() / (two::<T>() * x.sqrt()) },
    });

    crate::op!(Cbrt for<T> {
        eval: |[x]| { x.cbrt() },
        partial: |[x], _idx| {
            T::one() / (three::<T>() * x.cbrt().powi(2))
        },
    });

    crate::op!(Abs for<T> {
        eval: |[x]| { x.abs() },
        partial: |[x], _idx| {
            if x > T::zero() { T::one() } else if x < T::zero() { -T::one() } else { T::zero() }
        },
    });

    crate::op!(Abs2 for<T> {
        eval: |[x]| { x * x },
        partial: |[x], _idx| { two::<T>() * x },
    });

    crate::op!(Inv for<T> {
        eval: |[x]| { x.recip() },
        partial: |[x], _idx| { -T::one() / (x * x) },
    });

    // Julia's sign: zero (keeping its sign bit) for ±0, NaN for NaN.
    crate::op!(Sign for<T> {
        eval: |[x]| { if x > T::zero() { T::one() } else if x < T::zero() { -T::one() } else { x } },
        partial: |[_x], _idx| { T::zero() },
    });

    crate::op!(Identity for<T> {
        eval: |[x]| { x },
        partial: |[_x], _idx| { T::one() },
    });

    crate::op!(Neg for<T> {
        infix: "-",
        eval: |[x]| { -x },
        partial: |[_x], _idx| { -T::one() },
    });

    crate::op!(Add for<T> {
        infix: "+",
        aliases: ["plus"],
        commutative: true,
        associative: true,
        eval: |[x, y]| { x + y },
        partial: |[_x, _y], _idx| { T::one() },
    });

    crate::op!(Sub for<T> {
        infix: "-",
        eval: |[x, y]| { x - y },
        partial: |[_x, _y], idx| { if idx == 0 { T::one() } else { -T::one() } },
    });

    crate::op!(Mul for<T> {
        infix: "*",
        aliases: ["mult"],
        commutative: true,
        associative: true,
        eval: |[x, y]| { x * y },
        partial: |[x, y], idx| { if idx == 0 { y } else { x } },
    });

    crate::op!(Div for<T> {
        infix: "/",
        eval: |[x, y]| { x / y },
        partial: |[x, y], idx| { if idx == 0 { T::one() / y } else { -x / (y * y) } },
    });

    crate::op!(Pow for<T> {
        infix: "^",
        eval: |[x, y]| { if x == T::zero() && y < T::zero() { T::nan() } else { x.powf(y) } },
        partial: |[x, y], idx| {
            if idx == 0 { y * x.powf(y - T::one()) } else { x.powf(y) * x.ln() }
        },
    });

    crate::op!(Atan2 for<T> {
        eval: |[y, x]| { y.atan2(x) },
        partial: |[y, x], idx| {
            let denom = x * x + y * y;
            if idx == 0 { x / denom } else { -y / denom }
        },
    });

    // Julia's `min` and `max`: NaN if either argument is NaN, and `-0.0` below `0.0`.
    crate::op!(Min for<T> {
        commutative: true,
        associative: true,
        eval: |[x, y]| {
            let diff = x - y;
            if x.is_nan() || y.is_nan() { diff } else if diff.is_sign_negative() { x } else { y }
        },
        partial: |[x, y], idx| {
            #[allow(clippy::collapsible_else_if)]
            if x.is_nan() || y.is_nan() {
                T::nan()
            } else if idx == 0 {
                if x < y { T::one() } else if x > y { T::zero() } else { half::<T>() }
            } else {
                if y < x { T::one() } else if y > x { T::zero() } else { half::<T>() }
            }
        },
    });

    crate::op!(Max for<T> {
        commutative: true,
        associative: true,
        eval: |[x, y]| {
            let diff = x - y;
            if x.is_nan() || y.is_nan() { diff } else if diff.is_sign_negative() { y } else { x }
        },
        partial: |[x, y], idx| {
            #[allow(clippy::collapsible_else_if)]
            if x.is_nan() || y.is_nan() {
                T::nan()
            } else if idx == 0 {
                if x > y { T::one() } else if x < y { T::zero() } else { half::<T>() }
            } else {
                if y > x { T::one() } else if y < x { T::zero() } else { half::<T>() }
            }
        },
    });

    crate::op!(Fma for<T> {
        eval: |[x, y, z]| { x.mul_add(y, z) },
        partial: |[x, y, _z], idx| { if idx == 0 { y } else if idx == 1 { x } else { T::one() } },
    });

    // Julia's `clamp(x, lo, hi) = x > hi ? hi : x < lo ? lo : x`, which accepts inverted bounds.
    crate::op!(Clamp for<T> {
        eval: |[x, lo, hi]| { if x > hi { hi } else if x < lo { lo } else { x } },
        partial: |[x, lo, hi], idx| {
            let picked = if x > hi { 2 } else if x < lo { 1 } else { 0 };
            if idx == picked { T::one() } else { T::zero() }
        },
    });

    crate::op!(Relu for<T> {
        eval: |[x]| { if x > T::zero() { x } else { T::zero() } },
        partial: |[x], _idx| { if x > T::zero() { T::one() } else { T::zero() } },
    });

    crate::op!(Square for<T> {
        eval: |[x]| { x * x },
        partial: |[x], _idx| { two::<T>() * x },
    });

    crate::op!(Cube for<T> {
        eval: |[x]| { x * x * x },
        partial: |[x], _idx| { three::<T>() * x * x },
    });

    crate::op!(Greater for<T> {
        name: "greater",
        display: ">",
        eval: |[x, y]| { if x > y { T::one() } else { T::zero() } },
        partial: |[_x, _y], _idx| { T::zero() },
    });

    crate::op!(Less for<T> {
        name: "less",
        display: "<",
        eval: |[x, y]| { if x < y { T::one() } else { T::zero() } },
        partial: |[_x, _y], _idx| { T::zero() },
    });

    crate::op!(GreaterEqual for<T> {
        name: "greater_equal",
        display: ">=",
        eval: |[x, y]| { if x >= y { T::one() } else { T::zero() } },
        partial: |[_x, _y], _idx| { T::zero() },
    });

    crate::op!(LessEqual for<T> {
        name: "less_equal",
        display: "<=",
        eval: |[x, y]| { if x <= y { T::one() } else { T::zero() } },
        partial: |[_x, _y], _idx| { T::zero() },
    });

    crate::op!(Equal for<T> {
        name: "equal",
        display: "==",
        eval: |[x, y]| { if x == y { T::one() } else { T::zero() } },
        partial: |[_x, _y], _idx| { T::zero() },
    });

    crate::op!(NotEqual for<T> {
        name: "not_equal",
        display: "!=",
        eval: |[x, y]| { if x != y { T::one() } else { T::zero() } },
        partial: |[_x, _y], _idx| { T::zero() },
    });

    crate::op!(LogicalAnd for<T> {
        name: "logical_and",
        display: "&&",
        eval: |[x, y]| { if x > T::zero() && y > T::zero() { T::one() } else { T::zero() } },
        partial: |[_x, _y], _idx| { T::zero() },
    });

    crate::op!(LogicalOr for<T> {
        name: "logical_or",
        display: "||",
        eval: |[x, y]| { if x > T::zero() || y > T::zero() { T::one() } else { T::zero() } },
        partial: |[_x, _y], _idx| { T::zero() },
    });

    crate::op!(LogicalNot for<T> {
        name: "logical_not",
        display: "!",
        eval: |[x]| { if x > T::zero() { T::zero() } else { T::one() } },
        partial: |[_x], _idx| { T::zero() },
    });

    crate::op!(Cond for<T> {
        name: "cond",
        eval: |[x, y]| { if x > T::zero() { y } else { T::zero() } },
        partial: |[x, _y], idx| { if idx == 1 && x > T::zero() { T::one() } else { T::zero() } },
    });

    crate::op!(IfElse for<T> {
        name: "ifelse",
        eval: |[c, a, b]| { if c > T::zero() { a } else { b } },
        partial: |[c, _a, _b], idx| {
            let taken = c > T::zero();
            match idx {
                0 => T::zero(),
                1 => if taken { T::one() } else { T::zero() },
                _ => if taken { T::zero() } else { T::one() },
            }
        },
    });
}

pub mod presets {
    use crate::operator_enum::builtin::*;
    use crate::opset;

    // A convenient, batteries-included opset so downstream crates (like `symbolic_regression`)
    // don't need to define their own `Ops` type for common scalar use cases.

    opset! {
        pub BuiltinOpsF32 for f32 {
            Abs, Abs2, Acos, Acosh, Asin, Asinh, Atan, Atanh, Cbrt, Cos, Cosh, Cot, Csc, Exp, Exp2, Expm1,
            Identity, Inv, Log, Log1p, Log2, Log10, Neg, Sec, Sign, Sin, Sinh, Sqrt, Tan, Tanh,
            Add, Atan2, Div, Max, Min, Mul, Pow, Sub,
            Clamp, Fma,
            Relu, Square, Cube,
            Greater, Less, GreaterEqual, LessEqual, LogicalAnd, LogicalOr, Cond,
        }
    }

    opset! {
        pub BuiltinOpsF64 for f64 {
            Abs, Abs2, Acos, Acosh, Asin, Asinh, Atan, Atanh, Cbrt, Cos, Cosh, Cot, Csc, Exp, Exp2, Expm1,
            Identity, Inv, Log, Log1p, Log2, Log10, Neg, Sec, Sign, Sin, Sinh, Sqrt, Tan, Tanh,
            Add, Atan2, Div, Max, Min, Mul, Pow, Sub,
            Clamp, Fma,
            Relu, Square, Cube,
            Greater, Less, GreaterEqual, LessEqual, LogicalAnd, LogicalOr, Cond,
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Operator set macros
// -------------------------------------------------------------------------------------------------

#[macro_export]
macro_rules! opset {
    (@count) => { 0usize };
    (@count $head:tt $(, $tail:tt)*) => { 1usize + $crate::opset!(@count $($tail),*) };

    // Operator type resolution:
    // - Default: the type is `OpName` (must be in scope)
    // - Override per op: `OpName = some::path::Type`
    (@op_ty $op_name:ident) => { $op_name };
    (@op_ty $op_name:ident = $op_path:path) => { $op_path };

    (@max_op_arity $op_name:ident $(= $op_path:path)?) => {
        <$crate::opset!(@op_ty $op_name $(= $op_path)?) as $crate::traits::OpTag>::ARITY
    };
    (@max_op_arity $op_name:ident $(= $op_path:path)?, $($rest:tt)+) => {{
        let a = <$crate::opset!(@op_ty $op_name $(= $op_path)?) as $crate::traits::OpTag>::ARITY;
        let b = $crate::opset!(@max_op_arity $($rest)+);
        if a > b { a } else { b }
    }};

    // New syntax:
    // opset! { pub MyOps for f64 { Sin, Cos, Square, ... } }
    (
        $(#[$meta:meta])* $vis:vis $Ops:ident for $t:ty {
            $($op_name:ident $(= $op_path:path)?),* $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Copy, Clone, Debug, Default)]
        $vis struct $Ops<C = $crate::custom::NoCustom> {
            pub custom: C,
        }

        $crate::paste::paste! {
            #[allow(non_snake_case)]
            mod [<__opset_ $Ops>] {
                #![allow(non_upper_case_globals)]

                use super::*;

                const fn count_arity<const N: usize>(arities: &[u8; N], target: u8) -> usize {
                    let mut i = 0usize;
                    let mut n = 0usize;
                    while i < N {
                        if arities[i] == target {
                            n += 1;
                        }
                        i += 1;
                    }
                    n
                }

                const fn filter_ids<const N: usize, const M: usize>(
                    arities: &[u8; N],
                    target: u8,
                    ids: &[u16; N],
                ) -> [u16; M] {
                    let mut out = [0u16; M];
                    let mut i = 0usize;
                    let mut j = 0usize;
                    while i < N {
                        if arities[i] == target {
                            out[j] = ids[i];
                            j += 1;
                        }
                        i += 1;
                    }
                    out
                }

                #[repr(u16)]
                #[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
                #[allow(non_camel_case_types)]
                pub(super) enum OpId { $($op_name,)* }

                $crate::paste::paste! {
                    $(
                        pub(super) const [<ID_ $op_name>]: u16 = OpId::$op_name as u16;
                    )*
                }

                pub(super) const N: usize = $crate::opset!(@count $($op_name),*);
                pub(super) const N_BUILTIN: u16 = N as u16;

                pub(super) const ARITIES: [u8; N] = [
                    $( <$crate::opset!(@op_ty $op_name $(= $op_path)?) as $crate::traits::OpTag>::ARITY, )*
                ];

                pub(super) const IDS: [u16; N] = [
                    $( OpId::$op_name as u16, )*
                ];

                pub(super) const MAX_ARITY: u8 = $crate::opset!(@max_op_arity $($op_name $(= $op_path)?),*);
                const _: () = {
                    if MAX_ARITY > 16 {
                        panic!("opset!: max arity > 16 is not supported");
                    }
                };

                const COUNT_1: usize = count_arity(&ARITIES, 1);
                const COUNT_2: usize = count_arity(&ARITIES, 2);
                const COUNT_3: usize = count_arity(&ARITIES, 3);
                const COUNT_4: usize = count_arity(&ARITIES, 4);
                const COUNT_5: usize = count_arity(&ARITIES, 5);
                const COUNT_6: usize = count_arity(&ARITIES, 6);
                const COUNT_7: usize = count_arity(&ARITIES, 7);
                const COUNT_8: usize = count_arity(&ARITIES, 8);
                const COUNT_9: usize = count_arity(&ARITIES, 9);
                const COUNT_10: usize = count_arity(&ARITIES, 10);
                const COUNT_11: usize = count_arity(&ARITIES, 11);
                const COUNT_12: usize = count_arity(&ARITIES, 12);
                const COUNT_13: usize = count_arity(&ARITIES, 13);
                const COUNT_14: usize = count_arity(&ARITIES, 14);
                const COUNT_15: usize = count_arity(&ARITIES, 15);
                const COUNT_16: usize = count_arity(&ARITIES, 16);

                const IDS_1: [u16; COUNT_1] = filter_ids(&ARITIES, 1, &IDS);
                const IDS_2: [u16; COUNT_2] = filter_ids(&ARITIES, 2, &IDS);
                const IDS_3: [u16; COUNT_3] = filter_ids(&ARITIES, 3, &IDS);
                const IDS_4: [u16; COUNT_4] = filter_ids(&ARITIES, 4, &IDS);
                const IDS_5: [u16; COUNT_5] = filter_ids(&ARITIES, 5, &IDS);
                const IDS_6: [u16; COUNT_6] = filter_ids(&ARITIES, 6, &IDS);
                const IDS_7: [u16; COUNT_7] = filter_ids(&ARITIES, 7, &IDS);
                const IDS_8: [u16; COUNT_8] = filter_ids(&ARITIES, 8, &IDS);
                const IDS_9: [u16; COUNT_9] = filter_ids(&ARITIES, 9, &IDS);
                const IDS_10: [u16; COUNT_10] = filter_ids(&ARITIES, 10, &IDS);
                const IDS_11: [u16; COUNT_11] = filter_ids(&ARITIES, 11, &IDS);
                const IDS_12: [u16; COUNT_12] = filter_ids(&ARITIES, 12, &IDS);
                const IDS_13: [u16; COUNT_13] = filter_ids(&ARITIES, 13, &IDS);
                const IDS_14: [u16; COUNT_14] = filter_ids(&ARITIES, 14, &IDS);
                const IDS_15: [u16; COUNT_15] = filter_ids(&ARITIES, 15, &IDS);
                const IDS_16: [u16; COUNT_16] = filter_ids(&ARITIES, 16, &IDS);

                pub(super) const BY_ARITY: [&[u16]; 17] = [
                    &[] as &[u16],
                    &IDS_1,
                    &IDS_2,
                    &IDS_3,
                    &IDS_4,
                    &IDS_5,
                    &IDS_6,
                    &IDS_7,
                    &IDS_8,
                    &IDS_9,
                    &IDS_10,
                    &IDS_11,
                    &IDS_12,
                    &IDS_13,
                    &IDS_14,
                    &IDS_15,
                    &IDS_16,
                ];

                pub(super) const META: [$crate::traits::OpMeta<'static>; N] = [
                    $(
                        $crate::traits::OpMeta {
                            arity: <$crate::opset!(@op_ty $op_name $(= $op_path)?) as $crate::traits::OpTag>::ARITY,
                            id: OpId::$op_name as u16,
                            name: <$crate::opset!(@op_ty $op_name $(= $op_path)?) as $crate::traits::Operator<
                                $t,
                                { <$crate::opset!(@op_ty $op_name $(= $op_path)?) as $crate::traits::OpTag>::ARITY as usize }
                            >>::NAME,
                            display: <$crate::opset!(@op_ty $op_name $(= $op_path)?) as $crate::traits::Operator<
                                $t,
                                { <$crate::opset!(@op_ty $op_name $(= $op_path)?) as $crate::traits::OpTag>::ARITY as usize }
                            >>::DISPLAY,
                            infix: <$crate::opset!(@op_ty $op_name $(= $op_path)?) as $crate::traits::Operator<
                                $t,
                                { <$crate::opset!(@op_ty $op_name $(= $op_path)?) as $crate::traits::OpTag>::ARITY as usize }
                            >>::INFIX,
                            aliases: <$crate::opset!(@op_ty $op_name $(= $op_path)?) as $crate::traits::Operator<
                                $t,
                                { <$crate::opset!(@op_ty $op_name $(= $op_path)?) as $crate::traits::OpTag>::ARITY as usize }
                            >>::ALIASES,
                            commutative: <$crate::opset!(@op_ty $op_name $(= $op_path)?) as $crate::traits::Operator<
                                $t,
                                { <$crate::opset!(@op_ty $op_name $(= $op_path)?) as $crate::traits::OpTag>::ARITY as usize }
                            >>::COMMUTATIVE,
                            associative: <$crate::opset!(@op_ty $op_name $(= $op_path)?) as $crate::traits::Operator<
                                $t,
                                { <$crate::opset!(@op_ty $op_name $(= $op_path)?) as $crate::traits::OpTag>::ARITY as usize }
                            >>::ASSOCIATIVE,
                            complexity: <$crate::opset!(@op_ty $op_name $(= $op_path)?) as $crate::traits::Operator<
                                $t,
                                { <$crate::opset!(@op_ty $op_name $(= $op_path)?) as $crate::traits::OpTag>::ARITY as usize }
                            >>::COMPLEXITY,
                        },
                    )*
                ];
            }
        }

        impl<C: $crate::custom::CustomTable<$t>> $crate::traits::OperatorSet for $Ops<C> {
            type T = $t;

            const UNIT_COMPLEXITY: bool = C::UNIT_COMPLEXITY && {
                let metadata = &$crate::paste::paste! { [<__opset_ $Ops>]::META };
                let mut unit = true;
                let mut i = 0;
                while i < metadata.len() {
                    unit = unit && metadata[i].complexity == 1;
                    i += 1;
                }
                unit
            };

            #[inline]
            fn max_arity(&self) -> u8 {
                let builtin = $crate::paste::paste! { [<__opset_ $Ops>]::MAX_ARITY };
                builtin.max(self.custom.max_arity())
            }

            #[inline]
            fn ops_with_arity(&self, arity: u8) -> impl Iterator<Item = u16> + '_ {
                let builtin: &'static [u16] = $crate::paste::paste! {
                    [<__opset_ $Ops>]::BY_ARITY
                        .get(arity as usize)
                        .copied()
                        .unwrap_or(&[])
                };
                let n = $crate::paste::paste! { [<__opset_ $Ops>]::N_BUILTIN };
                builtin
                    .iter()
                    .copied()
                    .chain(self.custom.ids_with_arity(arity).map(move |local| n + local))
            }

            #[inline]
            fn meta(&self, op: $crate::traits::OpId) -> Option<$crate::traits::OpMeta<'_>> {
                let n = $crate::paste::paste! { [<__opset_ $Ops>]::N_BUILTIN };
                if op.id < n {
                    $crate::paste::paste! {
                        [<__opset_ $Ops>]::META
                            .get(op.id as usize)
                            .filter(|m| m.arity == op.arity)
                            .copied()
                    }
                } else {
                    self.custom
                        .meta(op.id - n)
                        .filter(|m| m.arity == op.arity)
                        .map(|m| $crate::traits::OpMeta { id: op.id, ..m })
                }
            }

            #[inline]
            fn is_custom(&self, op: $crate::traits::OpId) -> bool {
                let n = $crate::paste::paste! { [<__opset_ $Ops>]::N_BUILTIN };
                op.id >= n && self.custom.meta(op.id - n).is_some_and(|m| m.arity == op.arity)
            }

            #[inline]
            fn eval(&self, op: $crate::traits::OpId, ctx: $crate::dispatch::EvalKernelCtx<'_, '_, $t>) -> bool {
                match op.id {
                    $(
                        $crate::paste::paste! { [<__opset_ $Ops>]::[<ID_ $op_name>] } => {
                            let expected = <$crate::opset!(@op_ty $op_name $(= $op_path)?) as $crate::traits::OpTag>::ARITY;
                            debug_assert_eq!(op.arity, expected, "mismatched arity for op id {}", op.id);
                            $crate::evaluate::kernels::eval_nary::<
                                { <$crate::opset!(@op_ty $op_name $(= $op_path)?) as $crate::traits::OpTag>::ARITY as usize },
                                $t,
                                $crate::opset!(@op_ty $op_name $(= $op_path)?)
                            >(ctx.out, ctx.args, ctx.opts)
                        }
                    )*
                    id => self.custom.eval(id - $crate::paste::paste! { [<__opset_ $Ops>]::N_BUILTIN }, ctx),
                }
            }

            #[inline]
            fn diff(&self, op: $crate::traits::OpId, ctx: $crate::dispatch::DiffKernelCtx<'_, '_, $t>) -> bool {
                match op.id {
                    $(
                        $crate::paste::paste! { [<__opset_ $Ops>]::[<ID_ $op_name>] } => {
                            let expected = <$crate::opset!(@op_ty $op_name $(= $op_path)?) as $crate::traits::OpTag>::ARITY;
                            debug_assert_eq!(op.arity, expected, "mismatched arity for op id {}", op.id);
                            $crate::evaluate::kernels::diff_nary::<
                                { <$crate::opset!(@op_ty $op_name $(= $op_path)?) as $crate::traits::OpTag>::ARITY as usize },
                                $t,
                                $crate::opset!(@op_ty $op_name $(= $op_path)?)
                            >(ctx.out_val, ctx.out_der, ctx.args, ctx.dargs, ctx.opts)
                        }
                    )*
                    id => self.custom.diff(id - $crate::paste::paste! { [<__opset_ $Ops>]::N_BUILTIN }, ctx),
                }
            }

            #[inline]
            fn grad(&self, op: $crate::traits::OpId, ctx: $crate::dispatch::GradKernelCtx<'_, '_, $t>) -> bool {
                match op.id {
                    $(
                        $crate::paste::paste! { [<__opset_ $Ops>]::[<ID_ $op_name>] } => {
                            let expected = <$crate::opset!(@op_ty $op_name $(= $op_path)?) as $crate::traits::OpTag>::ARITY;
                            debug_assert_eq!(op.arity, expected, "mismatched arity for op id {}", op.id);
                            $crate::evaluate::kernels::grad_nary::<
                                { <$crate::opset!(@op_ty $op_name $(= $op_path)?) as $crate::traits::OpTag>::ARITY as usize },
                                $t,
                                $crate::opset!(@op_ty $op_name $(= $op_path)?)
                            >(ctx)
                        }
                    )*
                    id => self.custom.grad(id - $crate::paste::paste! { [<__opset_ $Ops>]::N_BUILTIN }, ctx),
                }
            }

            fn ir(&self, op: $crate::traits::OpId) -> Option<$crate::dsl::ir::OpIr<'_>> {
                match op.id {
                    $(
                        $crate::paste::paste! { [<__opset_ $Ops>]::[<ID_ $op_name>] } => {
                            $crate::dsl::kernels::Kernel::of::<$crate::opset!(@op_ty $op_name $(= $op_path)?)>()
                                .map($crate::dsl::ir::OpIr::Kernel)
                        }
                    )*
                    id => self.custom.ir(id - $crate::paste::paste! { [<__opset_ $Ops>]::N_BUILTIN }),
                }
            }

            fn definitions(&self) -> &[$crate::dsl::lower::Lowered] {
                self.custom.definitions()
            }
        }

        $(
            impl<C> $crate::traits::HasOp<$crate::opset!(@op_ty $op_name $(= $op_path)?)> for $Ops<C> {
                const ID: u16 = $crate::paste::paste! { [<__opset_ $Ops>]::OpId::$op_name as u16 };
            }
        )*

        // Private opsets in tests and binaries need not use every constructor.
        #[allow(dead_code)]
        impl $Ops<$crate::custom::NoCustom> {
            pub const fn new() -> Self {
                Self {
                    custom: $crate::custom::NoCustom,
                }
            }

            pub fn with_custom<C: $crate::custom::CustomTable<$t>>(self, custom: C) -> $Ops<C> {
                $Ops { custom }
            }
        }

        #[allow(dead_code)]
        impl<C: $crate::custom::CustomTable<$t>> $Ops<C> {
            #[allow(clippy::wrong_self_convention)]
            pub fn from_names<const D: usize, I>(
                self,
                names: I,
            ) -> Result<$crate::Operators<Self, D>, $crate::OperatorSelectError>
            where
                I: IntoIterator,
                I::Item: AsRef<str>,
            {
                $crate::Operators::from_names(self, names)
            }
        }
    };
}
