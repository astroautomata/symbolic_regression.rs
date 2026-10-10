//! Forward-mode dual numbers. Template formulas evaluate `D(f, i)` by running a subexpression's
//! operators on them, which differentiates every operator the way ForwardDiff.jl does for
//! SymbolicRegression.jl: a guarded branch that returns a constant (such as `NaN` outside a
//! logarithm's domain) has derivative 0. Nesting `Dual<Dual<T>>` gives higher derivatives.

use core::cmp::Ordering;
use core::num::FpCategory;
use core::ops::{Add, Div, Mul, Neg, Rem, Sub};

use num_traits::{Float, Num, NumCast, One, ToPrimitive, Zero};

/// `re + eps·ε` with `ε² = 0`. Comparisons look at `re` only.
#[derive(Copy, Clone, Debug)]
pub struct Dual<S> {
    pub re: S,
    pub eps: S,
}

impl<S: Float> Dual<S> {
    pub fn new(re: S, eps: S) -> Self {
        Self { re, eps }
    }

    pub fn constant(re: S) -> Self {
        Self { re, eps: S::zero() }
    }

    /// `f(re) + f'(re)·eps·ε`.
    fn chain(self, value: S, derivative: S) -> Self {
        Self {
            re: value,
            eps: derivative * self.eps,
        }
    }
}

impl<S: Float> PartialEq for Dual<S> {
    fn eq(&self, other: &Self) -> bool {
        self.re == other.re
    }
}

impl<S: Float> PartialOrd for Dual<S> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.re.partial_cmp(&other.re)
    }
}

impl<S: Float> Neg for Dual<S> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.re, -self.eps)
    }
}

impl<S: Float> Add for Dual<S> {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.eps + o.eps)
    }
}

impl<S: Float> Sub for Dual<S> {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.eps - o.eps)
    }
}

impl<S: Float> Mul for Dual<S> {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Self::new(self.re * o.re, self.eps * o.re + self.re * o.eps)
    }
}

impl<S: Float> Div for Dual<S> {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        let re = self.re / o.re;
        Self::new(re, (self.eps - re * o.eps) / o.re)
    }
}

impl<S: Float> Rem for Dual<S> {
    type Output = Self;
    fn rem(self, o: Self) -> Self {
        let q = (self.re / o.re).trunc();
        Self::new(self.re % o.re, self.eps - q * o.eps)
    }
}

impl<S: Float> Zero for Dual<S> {
    fn zero() -> Self {
        Self::constant(S::zero())
    }
    /// Structural: both parts are zero.
    fn is_zero(&self) -> bool {
        self.re.is_zero() && self.eps.is_zero()
    }
}

impl<S: Float> One for Dual<S> {
    fn one() -> Self {
        Self::constant(S::one())
    }
}

impl<S: Float> Num for Dual<S> {
    type FromStrRadixErr = S::FromStrRadixErr;
    fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        S::from_str_radix(s, radix).map(Self::constant)
    }
}

impl<S: Float> ToPrimitive for Dual<S> {
    fn to_i64(&self) -> Option<i64> {
        self.re.to_i64()
    }
    fn to_u64(&self) -> Option<u64> {
        self.re.to_u64()
    }
    fn to_f64(&self) -> Option<f64> {
        self.re.to_f64()
    }
}

impl<S: Float> NumCast for Dual<S> {
    fn from<N: ToPrimitive>(n: N) -> Option<Self> {
        S::from(n).map(Self::constant)
    }
}

impl<S: Float> Float for Dual<S> {
    fn nan() -> Self {
        Self::constant(S::nan())
    }
    fn infinity() -> Self {
        Self::constant(S::infinity())
    }
    fn neg_infinity() -> Self {
        Self::constant(S::neg_infinity())
    }
    fn neg_zero() -> Self {
        Self::constant(S::neg_zero())
    }
    fn min_value() -> Self {
        Self::constant(S::min_value())
    }
    fn min_positive_value() -> Self {
        Self::constant(S::min_positive_value())
    }
    fn max_value() -> Self {
        Self::constant(S::max_value())
    }
    fn epsilon() -> Self {
        Self::constant(S::epsilon())
    }
    fn is_nan(self) -> bool {
        self.re.is_nan()
    }
    fn is_infinite(self) -> bool {
        self.re.is_infinite()
    }
    fn is_finite(self) -> bool {
        self.re.is_finite()
    }
    fn is_normal(self) -> bool {
        self.re.is_normal()
    }
    fn classify(self) -> FpCategory {
        self.re.classify()
    }
    fn floor(self) -> Self {
        Self::constant(self.re.floor())
    }
    fn ceil(self) -> Self {
        Self::constant(self.re.ceil())
    }
    fn round(self) -> Self {
        Self::constant(self.re.round())
    }
    fn trunc(self) -> Self {
        Self::constant(self.re.trunc())
    }
    fn fract(self) -> Self {
        Self::new(self.re.fract(), self.eps)
    }
    fn abs(self) -> Self {
        if self.re.is_sign_negative() { -self } else { self }
    }
    fn signum(self) -> Self {
        Self::constant(self.re.signum())
    }
    fn is_sign_positive(self) -> bool {
        self.re.is_sign_positive()
    }
    fn is_sign_negative(self) -> bool {
        self.re.is_sign_negative()
    }
    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }
    fn recip(self) -> Self {
        Self::one() / self
    }
    fn powi(self, n: i32) -> Self {
        if n == 0 {
            return Self::one();
        }
        let d = S::from(n).unwrap() * self.re.powi(n - 1);
        self.chain(self.re.powi(n), d)
    }
    fn powf(self, n: Self) -> Self {
        let value = self.re.powf(n.re);
        let mut eps = S::zero();
        if !self.eps.is_zero() {
            eps = n.re * self.re.powf(n.re - S::one()) * self.eps;
        }
        if !n.eps.is_zero() {
            eps = eps + value * self.re.ln() * n.eps;
        }
        Self::new(value, eps)
    }
    fn sqrt(self) -> Self {
        let r = self.re.sqrt();
        self.chain(r, S::one() / (r + r))
    }
    fn exp(self) -> Self {
        let e = self.re.exp();
        self.chain(e, e)
    }
    fn exp2(self) -> Self {
        let e = self.re.exp2();
        self.chain(e, e * S::from(core::f64::consts::LN_2).unwrap())
    }
    fn ln(self) -> Self {
        self.chain(self.re.ln(), self.re.recip())
    }
    fn log(self, base: Self) -> Self {
        self.ln() / base.ln()
    }
    fn log2(self) -> Self {
        self.chain(
            self.re.log2(),
            (self.re * S::from(core::f64::consts::LN_2).unwrap()).recip(),
        )
    }
    fn log10(self) -> Self {
        self.chain(
            self.re.log10(),
            (self.re * S::from(core::f64::consts::LN_10).unwrap()).recip(),
        )
    }
    fn max(self, o: Self) -> Self {
        if o.re.is_nan() || self.re >= o.re { self } else { o }
    }
    fn min(self, o: Self) -> Self {
        if o.re.is_nan() || self.re <= o.re { self } else { o }
    }
    fn abs_sub(self, o: Self) -> Self {
        if self.re <= o.re { Self::zero() } else { self - o }
    }
    fn cbrt(self) -> Self {
        let c = self.re.cbrt();
        self.chain(c, (S::from(3.0).unwrap() * c * c).recip())
    }
    fn hypot(self, o: Self) -> Self {
        (self * self + o * o).sqrt()
    }
    fn sin(self) -> Self {
        self.chain(self.re.sin(), self.re.cos())
    }
    fn cos(self) -> Self {
        self.chain(self.re.cos(), -self.re.sin())
    }
    fn tan(self) -> Self {
        let t = self.re.tan();
        self.chain(t, S::one() + t * t)
    }
    fn asin(self) -> Self {
        self.chain(self.re.asin(), (S::one() - self.re * self.re).sqrt().recip())
    }
    fn acos(self) -> Self {
        self.chain(self.re.acos(), -(S::one() - self.re * self.re).sqrt().recip())
    }
    fn atan(self) -> Self {
        self.chain(self.re.atan(), (S::one() + self.re * self.re).recip())
    }
    fn atan2(self, o: Self) -> Self {
        let den = self.re * self.re + o.re * o.re;
        Self::new(self.re.atan2(o.re), (o.re * self.eps - self.re * o.eps) / den)
    }
    fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }
    fn exp_m1(self) -> Self {
        self.chain(self.re.exp_m1(), self.re.exp())
    }
    fn ln_1p(self) -> Self {
        self.chain(self.re.ln_1p(), (S::one() + self.re).recip())
    }
    fn sinh(self) -> Self {
        self.chain(self.re.sinh(), self.re.cosh())
    }
    fn cosh(self) -> Self {
        self.chain(self.re.cosh(), self.re.sinh())
    }
    fn tanh(self) -> Self {
        let t = self.re.tanh();
        self.chain(t, S::one() - t * t)
    }
    fn asinh(self) -> Self {
        self.chain(self.re.asinh(), (self.re * self.re + S::one()).sqrt().recip())
    }
    fn acosh(self) -> Self {
        self.chain(self.re.acosh(), (self.re * self.re - S::one()).sqrt().recip())
    }
    fn atanh(self) -> Self {
        self.chain(self.re.atanh(), (S::one() - self.re * self.re).recip())
    }
    fn integer_decode(self) -> (u64, i16, i8) {
        self.re.integer_decode()
    }
}

/// A scalar that carries derivatives along `ORDER` seeded directions: `T` itself (order 0) or
/// nested duals. `D(D(f, i), j)` evaluates `f` on order-2 jets.
pub trait Jet<T: Float>: Float {
    const ORDER: usize;
    /// `x` with derivative 1 along each direction `k` where `seeded[k]` (innermost first).
    fn seed(x: T, seeded: &[bool]) -> Self;
    /// The mixed partial along all `ORDER` directions.
    fn top(self) -> T;
}

impl<T: Float> Jet<T> for Dual<T> {
    const ORDER: usize = 1;
    fn seed(x: T, seeded: &[bool]) -> Self {
        Dual::new(x, if seeded[0] { T::one() } else { T::zero() })
    }
    fn top(self) -> T {
        self.eps
    }
}

impl<T: Float> Jet<T> for Dual<Dual<T>> {
    const ORDER: usize = 2;
    fn seed(x: T, seeded: &[bool]) -> Self {
        let outer = if seeded[1] { T::one() } else { T::zero() };
        Dual::new(<Dual<T> as Jet<T>>::seed(x, seeded), Dual::constant(outer))
    }
    fn top(self) -> T {
        self.eps.eps
    }
}

impl<T: Float> Jet<T> for Dual<Dual<Dual<T>>> {
    const ORDER: usize = 3;
    fn seed(x: T, seeded: &[bool]) -> Self {
        let outer = if seeded[2] { T::one() } else { T::zero() };
        Dual::new(
            <Dual<Dual<T>> as Jet<T>>::seed(x, seeded),
            Dual::constant(Dual::constant(outer)),
        )
    }
    fn top(self) -> T {
        self.eps.eps.eps
    }
}
