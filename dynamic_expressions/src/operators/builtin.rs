use num_traits::Float;

pub trait BuiltinOp<T: Float, const A: usize> {
    const NAME: &'static str;
    const INFIX: Option<&'static str> = None;
    const DISPLAY: &'static str = Self::NAME;

    fn eval(args: &[T; A]) -> T;
    fn partial(args: &[T; A], idx: usize) -> T;
}

pub trait OpMeta<const A: usize> {
    const COMMUTATIVE: bool = false;
    const ASSOCIATIVE: bool = false;
    const COMPLEXITY: f32 = 1.0;
}

fn two<T: Float>() -> T {
    T::one() + T::one()
}

macro_rules! builtin_op {
    (@name $Op:ident, $name:literal) => {
        $name
    };
    (@name $Op:ident) => {
        crate::paste::paste! { stringify!([<$Op:snake>]) }
    };
    (@infix $v:expr) => {
        $v
    };
    (@infix) => {
        None
    };
    (@commutative $v:expr) => {
        $v
    };
    (@commutative) => {
        false
    };
    (@associative $v:expr) => {
        $v
    };
    (@associative) => {
        false
    };
    (@complexity $v:expr) => {
        $v
    };
    (@complexity) => {
        1.0f32
    };
    (
        $(#[$meta:meta])*
        $Op:ident : $A:literal {
            $(name: $name:literal,)?
            $(infix: $infix:expr,)?
            $(commutative: $commutative:expr,)?
            $(associative: $associative:expr,)?
            $(complexity: $complexity:expr,)?
            eval($args:ident) $eval:block,
            partial($pargs:ident, $idx:ident) $partial:block $(,)?
        }
    ) => {
        $(#[$meta])*
        pub struct $Op;

        impl OpMeta<$A> for $Op {
            const COMMUTATIVE: bool = builtin_op!(@commutative $($commutative)?);
            const ASSOCIATIVE: bool = builtin_op!(@associative $($associative)?);
            const COMPLEXITY: f32 = builtin_op!(@complexity $($complexity)?);
        }

        impl<T: Float> BuiltinOp<T, $A> for $Op {
            const NAME: &'static str = builtin_op!(@name $Op $(, $name)?);
            const INFIX: Option<&'static str> = builtin_op!(@infix $($infix)?);
            const DISPLAY: &'static str = match builtin_op!(@infix $($infix)?) {
                Some(s) => s,
                None => builtin_op!(@name $Op $(, $name)?),
            };

            fn eval(args: &[T; $A]) -> T {
                let $args = args;
                $eval
            }

            fn partial(args: &[T; $A], idx: usize) -> T {
                let $pargs = args;
                let $idx = idx;
                $partial
            }
        }
    };
}

builtin_op!(Cos: 1 {
    eval(args) { args[0].cos() },
    partial(args, idx) {
        match idx {
            0 => -args[0].sin(),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Sin: 1 {
    eval(args) { args[0].sin() },
    partial(args, idx) {
        match idx {
            0 => args[0].cos(),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Tan: 1 {
    eval(args) { args[0].tan() },
    partial(args, idx) {
        match idx {
            0 => {
                let c = args[0].cos();
                T::one() / (c * c)
            }
            _ => unreachable!(),
        }
    },
});

builtin_op!(Asin: 1 {
    eval(args) { args[0].asin() },
    partial(args, idx) {
        match idx {
            0 => T::one() / (T::one() - args[0] * args[0]).sqrt(),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Acos: 1 {
    eval(args) { args[0].acos() },
    partial(args, idx) {
        match idx {
            0 => -T::one() / (T::one() - args[0] * args[0]).sqrt(),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Atan: 1 {
    eval(args) { args[0].atan() },
    partial(args, idx) {
        match idx {
            0 => T::one() / (T::one() + args[0] * args[0]),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Sinh: 1 {
    eval(args) { args[0].sinh() },
    partial(args, idx) {
        match idx {
            0 => args[0].cosh(),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Cosh: 1 {
    eval(args) { args[0].cosh() },
    partial(args, idx) {
        match idx {
            0 => args[0].sinh(),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Tanh: 1 {
    eval(args) { args[0].tanh() },
    partial(args, idx) {
        match idx {
            0 => {
                let c = args[0].cosh();
                T::one() / (c * c)
            }
            _ => unreachable!(),
        }
    },
});

builtin_op!(Asinh: 1 {
    eval(args) { args[0].asinh() },
    partial(args, idx) {
        match idx {
            0 => T::one() / (args[0] * args[0] + T::one()).sqrt(),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Acosh: 1 {
    eval(args) { args[0].acosh() },
    partial(args, idx) {
        match idx {
            0 => {
                let x = args[0];
                T::one() / ((x - T::one()).sqrt() * (x + T::one()).sqrt())
            }
            _ => unreachable!(),
        }
    },
});

builtin_op!(Atanh: 1 {
    eval(args) { args[0].atanh() },
    partial(args, idx) {
        match idx {
            0 => T::one() / (T::one() - args[0] * args[0]),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Sec: 1 {
    eval(args) { T::one() / args[0].cos() },
    partial(args, idx) {
        match idx {
            0 => {
                let sec = T::one() / args[0].cos();
                sec * args[0].tan()
            }
            _ => unreachable!(),
        }
    },
});

builtin_op!(Csc: 1 {
    eval(args) { T::one() / args[0].sin() },
    partial(args, idx) {
        match idx {
            0 => {
                let csc = T::one() / args[0].sin();
                let cot = T::one() / args[0].tan();
                -csc * cot
            }
            _ => unreachable!(),
        }
    },
});

builtin_op!(Cot: 1 {
    eval(args) { T::one() / args[0].tan() },
    partial(args, idx) {
        match idx {
            0 => {
                let s = args[0].sin();
                -T::one() / (s * s)
            }
            _ => unreachable!(),
        }
    },
});

builtin_op!(Exp: 1 {
    eval(args) { args[0].exp() },
    partial(args, idx) {
        match idx {
            0 => args[0].exp(),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Log: 1 {
    eval(args) { args[0].ln() },
    partial(args, idx) {
        match idx {
            0 => T::one() / args[0],
            _ => unreachable!(),
        }
    },
});

builtin_op!(Log2: 1 {
    eval(args) { args[0].log2() },
    partial(args, idx) {
        match idx {
            0 => T::one() / (args[0] * two::<T>().ln()),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Log10: 1 {
    eval(args) { args[0].log10() },
    partial(args, idx) {
        match idx {
            0 => {
                let ten = T::from(10.0).expect("Float can represent 10.0");
                T::one() / (args[0] * ten.ln())
            }
            _ => unreachable!(),
        }
    },
});

builtin_op!(Log1p: 1 {
    eval(args) { args[0].ln_1p() },
    partial(args, idx) {
        match idx {
            0 => T::one() / (T::one() + args[0]),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Exp2: 1 {
    eval(args) { args[0].exp2() },
    partial(args, idx) {
        match idx {
            0 => args[0].exp2() * two::<T>().ln(),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Expm1: 1 {
    eval(args) { args[0].exp_m1() },
    partial(args, idx) {
        match idx {
            0 => args[0].exp(),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Sqrt: 1 {
    eval(args) { args[0].sqrt() },
    partial(args, idx) {
        match idx {
            0 => T::one() / (two::<T>() * args[0].sqrt()),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Cbrt: 1 {
    eval(args) { args[0].cbrt() },
    partial(args, idx) {
        match idx {
            0 => {
                let three = T::from(3.0).expect("Float can represent 3.0");
                T::one() / (three * args[0].cbrt().powi(2))
            }
            _ => unreachable!(),
        }
    },
});

builtin_op!(Abs: 1 {
    eval(args) { args[0].abs() },
    partial(args, idx) {
        match idx {
            0 => {
                let x = args[0];
                if x > T::zero() {
                    T::one()
                } else if x < T::zero() {
                    -T::one()
                } else {
                    T::zero()
                }
            }
            _ => unreachable!(),
        }
    },
});

builtin_op!(Abs2: 1 {
    eval(args) { args[0] * args[0] },
    partial(args, idx) {
        match idx {
            0 => two::<T>() * args[0],
            _ => unreachable!(),
        }
    },
});

builtin_op!(Inv: 1 {
    eval(args) { args[0].recip() },
    partial(args, idx) {
        match idx {
            0 => -T::one() / (args[0] * args[0]),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Sign: 1 {
    eval(args) { args[0].signum() },
    partial(_args, idx) {
        match idx {
            0 => T::zero(),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Identity: 1 {
    eval(args) { args[0] },
    partial(_args, idx) {
        match idx {
            0 => T::one(),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Neg: 1 {
    infix: Some("-"),
    eval(args) { -args[0] },
    partial(_args, idx) {
        match idx {
            0 => -T::one(),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Add: 2 {
    infix: Some("+"),
    commutative: true,
    associative: true,
    eval(args) { args[0] + args[1] },
    partial(_args, idx) {
        match idx {
            0 | 1 => T::one(),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Sub: 2 {
    infix: Some("-"),
    eval(args) { args[0] - args[1] },
    partial(_args, idx) {
        match idx {
            0 => T::one(),
            1 => -T::one(),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Mul: 2 {
    infix: Some("*"),
    commutative: true,
    associative: true,
    eval(args) { args[0] * args[1] },
    partial(args, idx) {
        match idx {
            0 => args[1],
            1 => args[0],
            _ => unreachable!(),
        }
    },
});

builtin_op!(Div: 2 {
    infix: Some("/"),
    eval(args) { args[0] / args[1] },
    partial(args, idx) {
        match idx {
            0 => T::one() / args[1],
            1 => -args[0] / (args[1] * args[1]),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Pow: 2 {
    eval(args) { args[0].powf(args[1]) },
    partial(args, idx) {
        let x = args[0];
        let y = args[1];
        let f = x.powf(y);
        match idx {
            0 => y * x.powf(y - T::one()),
            1 => f * x.ln(),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Atan2: 2 {
    eval(args) { args[0].atan2(args[1]) },
    partial(args, idx) {
        let y = args[0];
        let x = args[1];
        let denom = x * x + y * y;
        match idx {
            0 => x / denom,
            1 => -y / denom,
            _ => unreachable!(),
        }
    },
});

builtin_op!(Min: 2 {
    commutative: true,
    associative: true,
    eval(args) { args[0].min(args[1]) },
    partial(args, idx) {
        let half = T::from(0.5).expect("Float can represent 0.5");
        if args[0].is_nan() && args[1].is_nan() {
            return half;
        }
        if args[0].is_nan() {
            return match idx {
                0 => T::zero(),
                1 => T::one(),
                _ => unreachable!(),
            };
        }
        if args[1].is_nan() {
            return match idx {
                0 => T::one(),
                1 => T::zero(),
                _ => unreachable!(),
            };
        }
        match idx {
            0 => {
                if args[0] < args[1] {
                    T::one()
                } else if args[0] > args[1] {
                    T::zero()
                } else {
                    half
                }
            }
            1 => {
                if args[1] < args[0] {
                    T::one()
                } else if args[1] > args[0] {
                    T::zero()
                } else {
                    half
                }
            }
            _ => unreachable!(),
        }
    },
});

builtin_op!(Max: 2 {
    commutative: true,
    associative: true,
    eval(args) { args[0].max(args[1]) },
    partial(args, idx) {
        let half = T::from(0.5).expect("Float can represent 0.5");
        if args[0].is_nan() && args[1].is_nan() {
            return half;
        }
        if args[0].is_nan() {
            return match idx {
                0 => T::zero(),
                1 => T::one(),
                _ => unreachable!(),
            };
        }
        if args[1].is_nan() {
            return match idx {
                0 => T::one(),
                1 => T::zero(),
                _ => unreachable!(),
            };
        }
        match idx {
            0 => {
                if args[0] > args[1] {
                    T::one()
                } else if args[0] < args[1] {
                    T::zero()
                } else {
                    half
                }
            }
            1 => {
                if args[1] > args[0] {
                    T::one()
                } else if args[1] < args[0] {
                    T::zero()
                } else {
                    half
                }
            }
            _ => unreachable!(),
        }
    },
});

builtin_op!(Fma: 3 {
    eval(args) { args[0].mul_add(args[1], args[2]) },
    partial(args, idx) {
        match idx {
            0 => args[1],
            1 => args[0],
            2 => T::one(),
            _ => unreachable!(),
        }
    },
});

builtin_op!(Clamp: 3 {
    eval(args) {
        let x = args[0];
        let lo = args[1];
        let hi = args[2];
        // `Float::clamp` may panic if `lo > hi`, so keep old behavior as fallback.
        if lo <= hi {
            x.clamp(lo, hi)
        } else if x < lo {
            lo
        } else if x > hi {
            hi
        } else {
            x
        }
    },
    partial(args, idx) {
        let x = args[0];
        let lo = args[1];
        let hi = args[2];
        match idx {
            0 => {
                if x < lo || x > hi { T::zero() } else { T::one() }
            }
            1 => {
                if x < lo { T::one() } else { T::zero() }
            }
            2 => {
                if x > hi { T::one() } else { T::zero() }
            }
            _ => unreachable!(),
        }
    },
});
