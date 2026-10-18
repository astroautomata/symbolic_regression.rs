use core::any::TypeId;

use num_traits::Float;

use crate::dispatch::SrcRef;
use crate::evaluate::EvalOptions;
use crate::evaluate::kernels::{eval_nary, partial_nary};
use crate::operator_enum::builtin;
use crate::traits::{OpTag, Operator};

/// [`eval_nary`] for one kernel.
pub type EvalFn<T> = fn(&mut [T], &[SrcRef<'_, T>], &EvalOptions) -> bool;
/// [`partial_nary`] for one kernel.
pub type PartialFn<T> = fn(&mut [T], &[SrcRef<'_, T>], usize);

/// The largest arity among the kernels.
pub const MAX_KERNEL_ARITY: usize = 3;

macro_rules! kernels {
    ($($Op:ident),* $(,)?) => {
        /// Every built-in operator type in [`crate::operator_enum::builtin`], callable from the DSL.
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum Kernel {
            $($Op,)*
        }

        impl Kernel {
            pub const ALL: &'static [Kernel] = &[$(Kernel::$Op,)*];

            pub fn arity(self) -> u8 {
                match self {
                    $(Kernel::$Op => <builtin::$Op as OpTag>::ARITY,)*
                }
            }

            pub fn name(self) -> &'static str {
                match self {
                    $(Kernel::$Op => {
                        <builtin::$Op as Operator<f64, { <builtin::$Op as OpTag>::ARITY as usize }>>::NAME
                    })*
                }
            }

            pub fn aliases(self) -> &'static [&'static str] {
                match self {
                    $(Kernel::$Op => {
                        <builtin::$Op as Operator<f64, { <builtin::$Op as OpTag>::ARITY as usize }>>::ALIASES
                    })*
                }
            }

            pub fn eval_fn<T: Float>(self) -> EvalFn<T> {
                match self {
                    $(Kernel::$Op => eval_nary::<{ <builtin::$Op as OpTag>::ARITY as usize }, T, builtin::$Op>,)*
                }
            }

            pub fn partial_fn<T: Float>(self) -> PartialFn<T> {
                match self {
                    $(Kernel::$Op => partial_nary::<{ <builtin::$Op as OpTag>::ARITY as usize }, T, builtin::$Op>,)*
                }
            }

            /// Evaluates the kernel on one row.
            pub fn eval_scalar<T: Float>(self, args: &[T]) -> T {
                match self {
                    $(Kernel::$Op => {
                        const A: usize = <builtin::$Op as OpTag>::ARITY as usize;
                        <builtin::$Op as Operator<T, A>>::eval(&core::array::from_fn(|i| args[i]))
                    })*
                }
            }

            /// The kernel that the operator type `Op` is, if `Op` is one of the built-in types.
            pub fn of<Op: 'static>() -> Option<Kernel> {
                let id = TypeId::of::<Op>();
                $(if id == TypeId::of::<builtin::$Op>() {
                    return Some(Kernel::$Op);
                })*
                None
            }
        }
    };
}

kernels! {
    Cos, Sin, Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh, Asinh, Acosh, Atanh, Sec, Csc, Cot,
    Exp, Log, Log2, Log10, Log1p, Exp2, Expm1, Sqrt, Cbrt, Abs, Abs2, Inv, Sign, Identity, Neg,
    Add, Sub, Mul, Div, Pow, Atan2, Min, Max, Fma, Clamp,
    Greater, Less, GreaterEqual, LessEqual, Equal, NotEqual, LogicalAnd, LogicalOr, LogicalNot,
    IfElse, Cond, Relu, Square, Cube,
}

impl Kernel {
    /// The kernel callable as `name` with `arity` arguments.
    pub fn lookup(name: &str, arity: usize) -> Option<Kernel> {
        Self::ALL
            .iter()
            .copied()
            .find(|k| usize::from(k.arity()) == arity && k.answers_to(name))
    }

    /// The argument counts `name` accepts as a kernel call, ascending.
    pub fn arities_of(name: &str) -> Vec<u8> {
        let mut arities: Vec<u8> = Self::ALL
            .iter()
            .filter(|k| k.answers_to(name))
            .map(|k| k.arity())
            .collect();
        arities.sort_unstable();
        arities.dedup();
        arities
    }

    fn answers_to(self, name: &str) -> bool {
        self.name() == name || self.aliases().contains(&name)
    }

    /// Comparisons and logical operators: their 0/1 results are `Bool` in the DSL.
    pub fn returns_bool(self) -> bool {
        matches!(
            self,
            Self::Greater
                | Self::Less
                | Self::GreaterEqual
                | Self::LessEqual
                | Self::Equal
                | Self::NotEqual
                | Self::LogicalAnd
                | Self::LogicalOr
                | Self::LogicalNot
        )
    }
}
