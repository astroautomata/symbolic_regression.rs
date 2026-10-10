use std::sync::Arc;

use dynamic_expressions::dispatch::SrcRef;
use dynamic_expressions::dsl::{DslError, DslFloat, ScalarFn, Span, parse_definition};
use dynamic_expressions::utils::ZipEq;
use dynamic_expressions::{EvalOptions, OperatorSet};
use ndarray::Array2;
use num_traits::Float;

use crate::dataset::Dataset;

pub trait LossFn<T: Float>: Send + Sync {
    fn loss(&self, yhat: &[T], y: &[T], w: Option<&[T]>) -> T;
    fn dloss_dyhat(&self, yhat: &[T], y: &[T], w: Option<&[T]>, out: &mut [T]);

    /// Whether this loss can score a dataset with (`weighted`) or without weights. The search
    /// checks this before it starts.
    fn check_weights(&self, _weighted: bool) -> Result<(), String> {
        Ok(())
    }
}

pub fn baseline_loss_from_zero_expression<T: Float, Ops, const D: usize>(
    dataset: &Dataset<T>,
    ops: &Ops,
    loss: &dyn LossFn<T>,
) -> Option<T>
where
    Ops: OperatorSet<T = T>,
{
    let expr: dynamic_expressions::expression::PostfixExpr<T, Ops, D> =
        dynamic_expressions::expression::PostfixExpr::zero();
    let plan = dynamic_expressions::compile_plan(&expr.nodes, dataset.n_features, expr.consts.len());

    let mut yhat = vec![T::zero(); dataset.n_rows];
    let mut scratch = Array2::<T>::zeros((0, 0));

    let eval_opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };
    let ok = dynamic_expressions::eval_plan_array_into(
        &mut yhat,
        &plan,
        &expr,
        ops,
        dataset.x.view(),
        &mut scratch,
        &eval_opts,
    );
    if !ok {
        return None;
    }

    let base = loss.loss(&yhat, dataset.y_slice(), dataset.weights_slice());
    base.is_finite().then_some(base)
}

pub fn loss_to_cost<T: Float>(
    loss: T,
    complexity: usize,
    parsimony: f64,
    use_baseline: bool,
    baseline_loss: Option<T>,
) -> T {
    let floor = T::from(0.01).unwrap();

    let denom = if use_baseline {
        match baseline_loss {
            Some(b) if b.is_finite() && b >= floor => b,
            _ => floor,
        }
    } else {
        floor
    };

    let parsimony = T::from(parsimony).unwrap_or_else(T::zero);
    loss / denom + parsimony * T::from(complexity).unwrap_or_else(T::zero)
}

pub type LossObject<T> = Arc<dyn LossFn<T> + Send + Sync>;

/// The elementwise losses SymbolicRegression.jl re-exports from LossFunctions.jl, under the same
/// names without the `Loss` suffix. Distance losses act on `r = prediction - target`; margin
/// losses act on the agreement `a = target * prediction`, for targets in {-1, +1}.
///
/// Without weights the search minimizes the mean loss; with weights, the weighted sum divided by
/// the sum of the weights.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum BuiltinLoss {
    /// `|r|^p`
    LPDist { p: f64 },
    /// `|r|`
    L1Dist,
    /// `r^2`
    L2Dist,
    /// `1 - cos(2πr / circumference)`
    Periodic { circumference: f64 },
    /// `r^2 / 2` for `|r| <= d`, else `d|r| - d^2 / 2`
    Huber { d: f64 },
    /// `max(0, |r| - eps)`
    L1EpsilonIns { eps: f64 },
    /// `max(0, |r| - eps)^2`
    L2EpsilonIns { eps: f64 },
    /// `-log(4e^r / (1 + e^r)^2)`
    LogitDist,
    /// `r * ((r > 0) - tau)`
    Quantile { tau: f64 },
    /// `log(cosh(r))`
    LogCosh,
    /// `1` for `a < 0`, else `0`
    ZeroOne,
    /// `max(0, -a)`
    Perceptron,
    /// `log(1 + e^-a)`
    LogitMargin,
    /// `max(0, 1 - a)`
    L1Hinge,
    /// `max(0, 1 - a)^2`
    L2Hinge,
    /// `max(0, 1 - a)^2 / (2 gamma)` for `a >= 1 - gamma`, else `1 - gamma/2 - a`
    SmoothedL1Hinge { gamma: f64 },
    /// `max(0, 1 - a)^2` for `a >= -1`, else `-4a`
    ModifiedHuber,
    /// `(1 - a)^2`
    L2Margin,
    /// `e^-a`
    Exp,
    /// `1 - tanh(a)`
    Sigmoid,
    /// `1 - a` for `a <= q / (q + 1)`, else `q^q / ((q + 1)^(q + 1) a^q)`
    DWDMargin { q: f64 },
}

/// Julia's `sign`: `±1`, or `x` itself for zeros and NaN.
fn sign<T: Float>(x: T) -> T {
    if x == T::zero() || x.is_nan() {
        x
    } else {
        T::one().copysign(x)
    }
}

/// Julia's `max(0, x)`, which propagates NaN.
fn max_zero<T: Float>(x: T) -> T {
    if x < T::zero() { T::zero() } else { x }
}

fn indicator<T: Float>(condition: bool) -> T {
    if condition { T::one() } else { T::zero() }
}

impl BuiltinLoss {
    /// The loss of one prediction, following LossFunctions.jl v1.2.2.
    #[inline(always)]
    pub fn value<T: Float>(&self, prediction: T, target: T) -> T {
        let c = |v: f64| T::from(v).unwrap();
        let (zero, one, two) = (T::zero(), T::one(), c(2.0));
        let r = prediction - target;
        let a = target * prediction;
        match *self {
            Self::LPDist { p } => r.abs().powf(c(p)),
            Self::L1Dist => r.abs(),
            Self::L2Dist => r * r,
            Self::Periodic { circumference } => one - (r * c(2.0 * core::f64::consts::PI / circumference)).cos(),
            Self::Huber { d } => {
                let d = c(d);
                if r.abs() <= d {
                    c(0.5) * r * r
                } else {
                    d * r.abs() - c(0.5) * d * d
                }
            }
            Self::L1EpsilonIns { eps } => max_zero(r.abs() - c(eps)),
            Self::L2EpsilonIns { eps } => {
                let m = max_zero(r.abs() - c(eps));
                m * m
            }
            Self::LogitDist => -c(4.0).ln() - r + two * (one + r.exp()).ln(),
            Self::Quantile { tau } => r * (indicator::<T>(r > zero) - c(tau)),
            Self::LogCosh => {
                let x = -two * r;
                let softplus = if x > zero {
                    x + (-x).exp().ln_1p()
                } else {
                    x.exp().ln_1p()
                };
                r + softplus - two.ln()
            }
            Self::ZeroOne => indicator(sign(a) < zero),
            Self::Perceptron => max_zero(-a),
            Self::LogitMargin => (-a).exp().ln_1p(),
            Self::L1Hinge => max_zero(one - a),
            Self::L2Hinge => {
                if a >= one {
                    zero
                } else {
                    (one - a) * (one - a)
                }
            }
            Self::SmoothedL1Hinge { gamma } => {
                let gamma = c(gamma);
                if a >= one - gamma {
                    let m = max_zero(one - a);
                    c(0.5) / gamma * (m * m)
                } else {
                    one - gamma / two - a
                }
            }
            Self::ModifiedHuber => {
                if a >= -one {
                    let m = max_zero(one - a);
                    m * m
                } else {
                    -c(4.0) * a
                }
            }
            Self::L2Margin => (one - a) * (one - a),
            Self::Exp => (-a).exp(),
            Self::Sigmoid => one - a.tanh(),
            Self::DWDMargin { q } => {
                if a <= c(q / (q + 1.0)) {
                    one - a
                } else {
                    c(q.powf(q) / (q + 1.0).powf(q + 1.0)) / a.powf(c(q))
                }
            }
        }
    }

    /// The derivative of [`Self::value`] with respect to the prediction, following
    /// LossFunctions.jl v1.2.2 `deriv`, including its choices at kinks.
    #[inline(always)]
    pub fn deriv<T: Float>(&self, prediction: T, target: T) -> T {
        let c = |v: f64| T::from(v).unwrap();
        let (zero, one, two) = (T::zero(), T::one(), c(2.0));
        let r = prediction - target;
        let a = target * prediction;
        let margin = |d: T| target * d;
        match *self {
            Self::LPDist { p } => {
                if r == zero {
                    zero
                } else {
                    c(p) * r * r.abs().powf(c(p - 2.0))
                }
            }
            Self::L1Dist => sign(r),
            Self::L2Dist => two * r,
            Self::Periodic { circumference } => {
                let k = c(2.0 * core::f64::consts::PI / circumference);
                k * (r * k).sin()
            }
            Self::Huber { d } => {
                let d = c(d);
                if r.abs() <= d { r } else { d * sign(r) }
            }
            Self::L1EpsilonIns { eps } => {
                if r.abs() <= c(eps) {
                    zero
                } else {
                    sign(r)
                }
            }
            Self::L2EpsilonIns { eps } => {
                if r.abs() <= c(eps) {
                    zero
                } else {
                    two * sign(r) * (r.abs() - c(eps))
                }
            }
            Self::LogitDist => (r / two).tanh(),
            Self::Quantile { tau } => indicator::<T>(r > zero) - c(tau),
            Self::LogCosh => r.tanh(),
            Self::ZeroOne => margin(zero),
            Self::Perceptron => margin(if a >= zero { zero } else { -one }),
            Self::LogitMargin => margin(-one / (one + a.exp())),
            Self::L1Hinge => margin(if a >= one { zero } else { -one }),
            Self::L2Hinge => margin(if a >= one { zero } else { two * (a - one) }),
            Self::SmoothedL1Hinge { gamma } => {
                let gamma = c(gamma);
                margin(if a >= one - gamma {
                    if a >= one { zero } else { (a - one) / gamma }
                } else {
                    -one
                })
            }
            Self::ModifiedHuber => margin(if a >= -one {
                if a > one { zero } else { two * a - two }
            } else {
                -c(4.0)
            }),
            Self::L2Margin => margin(two * (a - one)),
            Self::Exp => margin(-(-a).exp()),
            Self::Sigmoid => {
                let sech = one / a.cosh();
                margin(-(sech * sech))
            }
            Self::DWDMargin { q } => margin(if a <= c(q / (q + 1.0)) {
                -one
            } else {
                -c((q / (q + 1.0)).powf(q + 1.0)) / a.powf(c(q + 1.0))
            }),
        }
    }
}

#[inline(always)]
fn mean_of<T: Float>(yhat: &[T], y: &[T], w: Option<&[T]>, point: impl Fn(T, T) -> T) -> T {
    let pairs = yhat.iter().zip_eq(y);
    match w {
        None => pairs.fold(T::zero(), |acc, (&p, &t)| acc + point(p, t)) / T::from(y.len()).unwrap(),
        Some(w) => {
            let sum = pairs
                .zip_eq(w)
                .fold(T::zero(), |acc, ((&p, &t), &wi)| acc + wi * point(p, t));
            sum / w.iter().fold(T::zero(), |acc, &wi| acc + wi)
        }
    }
}

#[inline(always)]
fn mean_derivative_into<T: Float>(yhat: &[T], y: &[T], w: Option<&[T]>, out: &mut [T], point: impl Fn(T, T) -> T) {
    let rows = out.iter_mut().zip_eq(yhat).zip_eq(y);
    match w {
        None => {
            let inv = T::from(y.len()).unwrap().recip();
            for ((o, &p), &t) in rows {
                *o = inv * point(p, t);
            }
        }
        Some(w) => {
            let inv = w.iter().fold(T::zero(), |acc, &wi| acc + wi).recip();
            for (((o, &p), &t), &wi) in rows.zip_eq(w) {
                *o = wi * inv * point(p, t);
            }
        }
    }
}

/// A built-in loss whose variant is written as a literal inside `value` and `deriv`, so the
/// variant `match` folds away and each loss compiles to its own vectorizable row loop, reached
/// directly through the `LossObject` vtable.
struct Specialized<V, D> {
    value: V,
    deriv: D,
}

impl<T: Float, V, D> LossFn<T> for Specialized<V, D>
where
    V: Fn(T, T) -> T + Send + Sync,
    D: Fn(T, T) -> T + Send + Sync,
{
    fn loss(&self, yhat: &[T], y: &[T], w: Option<&[T]>) -> T {
        mean_of(yhat, y, w, &self.value)
    }

    fn dloss_dyhat(&self, yhat: &[T], y: &[T], w: Option<&[T]>, out: &mut [T]) {
        mean_derivative_into(yhat, y, w, out, &self.deriv)
    }
}

macro_rules! specialize {
    ($loss:expr; $($pattern:pat => $variant:expr),* $(,)?) => {
        match $loss {
            $($pattern => Arc::new(Specialized {
                value: move |a: T, b: T| $variant.value(a, b),
                deriv: move |a: T, b: T| $variant.deriv(a, b),
            }),)*
        }
    };
}

impl BuiltinLoss {
    /// This loss averaged over rows, or weighted by the dataset's weights, as SymbolicRegression.jl
    /// does.
    pub fn into_loss<T: Float>(self) -> LossObject<T> {
        use BuiltinLoss as L;
        specialize!(self;
            L::LPDist { p } => L::LPDist { p },
            L::L1Dist => L::L1Dist,
            L::L2Dist => L::L2Dist,
            L::Periodic { circumference } => L::Periodic { circumference },
            L::Huber { d } => L::Huber { d },
            L::L1EpsilonIns { eps } => L::L1EpsilonIns { eps },
            L::L2EpsilonIns { eps } => L::L2EpsilonIns { eps },
            L::LogitDist => L::LogitDist,
            L::Quantile { tau } => L::Quantile { tau },
            L::LogCosh => L::LogCosh,
            L::ZeroOne => L::ZeroOne,
            L::Perceptron => L::Perceptron,
            L::LogitMargin => L::LogitMargin,
            L::L1Hinge => L::L1Hinge,
            L::L2Hinge => L::L2Hinge,
            L::SmoothedL1Hinge { gamma } => L::SmoothedL1Hinge { gamma },
            L::ModifiedHuber => L::ModifiedHuber,
            L::L2Margin => L::L2Margin,
            L::Exp => L::Exp,
            L::Sigmoid => L::Sigmoid,
            L::DWDMargin { q } => L::DWDMargin { q },
        )
    }
}

/// An elementwise loss written in the DSL: `f(prediction, target)`, averaged over rows, or
/// `f(prediction, target, weight)`, summed and divided by the sum of the weights, as in
/// SymbolicRegression.jl.
#[derive(Clone, Debug)]
pub struct DslLoss<T> {
    func: ScalarFn<T>,
}

const NO_CHECK: EvalOptions = EvalOptions {
    check_finite: false,
    early_exit: false,
};

/// Rows per call when summing a DSL loss, so the values fit in a stack buffer.
const LOSS_CHUNK: usize = 256;

impl<T: DslFloat> DslLoss<T> {
    fn weighted(&self) -> bool {
        self.func.arity() == 3
    }

    fn args<'a>(&self, yhat: &'a [T], y: &'a [T], w: Option<&'a [T]>) -> ([SrcRef<'a, T>; 3], usize) {
        if let Err(message) = self.check_weights(w.is_some()) {
            panic!("{message}");
        }
        let w = w.map_or(SrcRef::Const(T::zero()), SrcRef::Slice);
        (
            [SrcRef::Slice(yhat), SrcRef::Slice(y), w],
            usize::from(self.func.arity()),
        )
    }

    fn denominator(&self, n: usize, w: Option<&[T]>) -> T {
        match w {
            Some(w) => w.iter().fold(T::zero(), |acc, &wi| acc + wi),
            None => T::from(n).unwrap(),
        }
    }
}

fn rows<'a, T>(arg: SrcRef<'a, T>, start: usize, end: usize) -> SrcRef<'a, T> {
    match arg {
        SrcRef::Slice(s) => SrcRef::Slice(&s[start..end]),
        SrcRef::Const(c) => SrcRef::Const(c),
    }
}

impl<T: DslFloat> LossFn<T> for DslLoss<T> {
    fn loss(&self, yhat: &[T], y: &[T], w: Option<&[T]>) -> T {
        let (args, n_args) = self.args(yhat, y, w);
        let mut values = [T::zero(); LOSS_CHUNK];
        let mut sum = T::zero();
        for start in (0..y.len()).step_by(LOSS_CHUNK) {
            let end = (start + LOSS_CHUNK).min(y.len());
            let chunk = args.map(|a| rows(a, start, end));
            let out = &mut values[..end - start];
            self.func.eval(out, &chunk[..n_args], &NO_CHECK);
            sum = out.iter().fold(sum, |acc, &v| acc + v);
        }
        sum / self.denominator(y.len(), w)
    }

    fn dloss_dyhat(&self, yhat: &[T], y: &[T], w: Option<&[T]>, out: &mut [T]) {
        let (args, n_args) = self.args(yhat, y, w);
        self.func.partial(0, out, &args[..n_args]);
        let inv = self.denominator(y.len(), w).recip();
        for o in out.iter_mut() {
            *o = *o * inv;
        }
    }

    fn check_weights(&self, weighted: bool) -> Result<(), String> {
        match (self.weighted(), weighted) {
            (false, true) => Err(
                "the elementwise loss must accept (prediction, target, weight) when the dataset has weights".to_string(),
            ),
            (true, false) => Err(
                "the elementwise loss takes a weight, but the dataset has no weights; define it as f(prediction, target)"
                    .to_string(),
            ),
            _ => Ok(()),
        }
    }
}

/// Parses an elementwise loss written as in PySR's `elementwise_loss`: a built-in constructor such
/// as `"L1DistLoss()"`, `"HuberLoss(1.5)"`, or `"LPDistLoss{3}()"` (see [`BuiltinLoss`]), or a
/// definition such as `"myloss(x, y) = abs(x - y)^1.5"` or `"(x, y, w) -> w * abs(x - y)"`.
///
/// ```
/// use symbolic_regression::parse_loss;
///
/// let loss = parse_loss::<f64>("myloss(x, y, w) = w * abs(x - y)").unwrap();
/// assert_eq!(loss.loss(&[1.0, 4.0], &[0.0, 0.0], Some(&[1.0, 3.0])), 3.25);
/// let err = parse_loss::<f64>("PoissonLoss()").err().unwrap();
/// assert!(err.message.contains("unknown loss `PoissonLoss`"));
/// ```
pub fn parse_loss<T: DslFloat>(src: &str) -> Result<LossObject<T>, DslError> {
    if let Some(builtin) = parse_builtin_loss(src)? {
        return Ok(builtin.into_loss());
    }
    let def = parse_definition(src)?;
    let n = def.params.len();
    if n != 2 && n != 3 {
        let plural = if n == 1 { "" } else { "s" };
        return Err(DslError::new(
            format!(
                "an elementwise loss takes (prediction, target) or (prediction, target, weight), but this one takes {n} argument{plural}"
            ),
            def.span,
            src,
        ));
    }
    let func = ScalarFn::compile(&def)?;
    if func.returns_bool() {
        return Err(DslError::new(
            "an elementwise loss must return a number, but this one returns Bool",
            def.span,
            src,
        ));
    }
    Ok(Arc::new(DslLoss { func }))
}

/// Parses `Name(args...)` or `Name{p}()` as a [`BuiltinLoss`] constructor. Returns `None` when
/// `src` is not a constructor call, such as a definition `f(x, y) = ...`.
fn parse_builtin_loss(src: &str) -> Result<Option<BuiltinLoss>, DslError> {
    let start = src.len() - src.trim_start().len();
    let s = src.trim();
    let name_len = s
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(s.len());
    let (name, rest) = s.split_at(name_len);
    let name_span = Span::new(start, start + name_len);
    let err = |message: String| DslError::new(message, name_span, src);

    let (type_param, rest) = match rest.strip_prefix('{') {
        Some(r) => match r.split_once('}') {
            Some((p, r)) => (Some(p), r),
            None => return Ok(None),
        },
        None => (None, rest),
    };
    let Some(args) = rest.strip_prefix('(').and_then(|r| r.strip_suffix(')')) else {
        return Ok(None);
    };
    if name.is_empty() || args.contains(['(', ')', '=']) {
        return Ok(None);
    }
    let Some(&(_, usage)) = BUILTIN_LOSS_USAGE.iter().find(|(n, _)| *n == name) else {
        return Err(err(format!(
            "unknown loss `{name}`; the built-in losses are {}",
            BUILTIN_LOSS_USAGE
                .iter()
                .map(|(_, u)| *u)
                .collect::<Vec<_>>()
                .join(", ")
        )));
    };

    let number = |text: &str| {
        text.trim()
            .parse::<f64>()
            .map_err(|_| err(format!("`{name}` parameters must be numbers, got `{}`", text.trim())))
    };
    let args: Vec<f64> = if args.trim().is_empty() {
        Vec::new()
    } else {
        args.split(',').map(number).collect::<Result<_, _>>()?
    };
    let type_param = type_param.map(number).transpose()?;
    let positive = |v: f64, what: &str| {
        if v > 0.0 {
            Ok(v)
        } else {
            Err(err(format!("`{name}`: {what} must be strictly positive, got {v}")))
        }
    };

    use BuiltinLoss as L;
    let loss = match (name, type_param, args.as_slice()) {
        ("LPDistLoss", Some(p), []) | ("LPDistLoss", None, &[p]) => L::LPDist { p },
        ("L1DistLoss", None, []) => L::L1Dist,
        ("L2DistLoss", None, []) => L::L2Dist,
        ("PeriodicLoss", None, []) => L::Periodic { circumference: 1.0 },
        ("PeriodicLoss", None, &[c]) => L::Periodic {
            circumference: positive(c, "the circumference")?,
        },
        ("HuberLoss", None, []) => L::Huber { d: 1.0 },
        ("HuberLoss", None, &[d]) => L::Huber {
            d: positive(d, "the crossover parameter")?,
        },
        ("L1EpsilonInsLoss" | "EpsilonInsLoss", None, &[eps]) => L::L1EpsilonIns {
            eps: positive(eps, "ε")?,
        },
        ("L2EpsilonInsLoss", None, &[eps]) => L::L2EpsilonIns {
            eps: positive(eps, "ε")?,
        },
        ("LogitDistLoss", None, []) => L::LogitDist,
        ("QuantileLoss", None, &[tau]) => L::Quantile { tau },
        ("LogCoshLoss", None, []) => L::LogCosh,
        ("ZeroOneLoss", None, []) => L::ZeroOne,
        ("PerceptronLoss", None, []) => L::Perceptron,
        ("LogitMarginLoss", None, []) => L::LogitMargin,
        ("L1HingeLoss" | "HingeLoss", None, []) => L::L1Hinge,
        ("L2HingeLoss", None, []) => L::L2Hinge,
        ("SmoothedL1HingeLoss", None, &[gamma]) => L::SmoothedL1Hinge {
            gamma: positive(gamma, "γ")?,
        },
        ("ModifiedHuberLoss", None, []) => L::ModifiedHuber,
        ("L2MarginLoss", None, []) => L::L2Margin,
        ("ExpLoss", None, []) => L::Exp,
        ("SigmoidLoss", None, []) => L::Sigmoid,
        ("DWDMarginLoss", None, &[q]) => L::DWDMargin { q: positive(q, "q")? },
        _ => return Err(err(format!("`{name}` is constructed as {usage}"))),
    };
    Ok(Some(loss))
}

const BUILTIN_LOSS_USAGE: &[(&str, &str)] = &[
    ("LPDistLoss", "`LPDistLoss{P}()` or `LPDistLoss(P)`"),
    ("L1DistLoss", "`L1DistLoss()`"),
    ("L2DistLoss", "`L2DistLoss()`"),
    ("PeriodicLoss", "`PeriodicLoss(c)`"),
    ("HuberLoss", "`HuberLoss(d)`"),
    ("L1EpsilonInsLoss", "`L1EpsilonInsLoss(ϵ)`"),
    ("EpsilonInsLoss", "`EpsilonInsLoss(ϵ)`"),
    ("L2EpsilonInsLoss", "`L2EpsilonInsLoss(ϵ)`"),
    ("LogitDistLoss", "`LogitDistLoss()`"),
    ("QuantileLoss", "`QuantileLoss(τ)`"),
    ("LogCoshLoss", "`LogCoshLoss()`"),
    ("ZeroOneLoss", "`ZeroOneLoss()`"),
    ("PerceptronLoss", "`PerceptronLoss()`"),
    ("LogitMarginLoss", "`LogitMarginLoss()`"),
    ("L1HingeLoss", "`L1HingeLoss()`"),
    ("HingeLoss", "`HingeLoss()`"),
    ("L2HingeLoss", "`L2HingeLoss()`"),
    ("SmoothedL1HingeLoss", "`SmoothedL1HingeLoss(γ)`"),
    ("ModifiedHuberLoss", "`ModifiedHuberLoss()`"),
    ("L2MarginLoss", "`L2MarginLoss()`"),
    ("ExpLoss", "`ExpLoss()`"),
    ("SigmoidLoss", "`SigmoidLoss()`"),
    ("DWDMarginLoss", "`DWDMarginLoss(q)`"),
];
