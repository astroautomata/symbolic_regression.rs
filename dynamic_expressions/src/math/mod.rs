mod macros;

use macros::*;

use crate::operators::builtin::*;

unary_wrappers! {
    cos => Cos,
    sin => Sin,
    tan => Tan,
    asin => Asin,
    acos => Acos,
    atan => Atan,
    sinh => Sinh,
    cosh => Cosh,
    tanh => Tanh,
    asinh => Asinh,
    acosh => Acosh,
    atanh => Atanh,
    sec => Sec,
    csc => Csc,
    cot => Cot,
    exp => Exp,
    exp2 => Exp2,
    expm1 => Expm1,
    log => Log,
    log2 => Log2,
    log10 => Log10,
    log1p => Log1p,
    sqrt => Sqrt,
    cbrt => Cbrt,
    abs => Abs,
    abs2 => Abs2,
    inv => Inv,
    sign => Sign,
    identity => Identity,
    neg => Neg,
}

binary_wrappers! {
    div => Div,
    add => Add,
    sub => Sub,
    mul => Mul,
    pow => Pow,
    atan2 => Atan2,
    min => Min,
    max => Max,
}

ternary_wrappers! {
    fma => Fma,
    clamp => Clamp,
}
