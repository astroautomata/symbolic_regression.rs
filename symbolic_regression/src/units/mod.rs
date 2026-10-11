//! Physical units, as SymbolicRegression.jl reads them with DynamicQuantities: each unit string
//! parses to SI dimensions, which dimensional analysis compares, and to symbolic units, which the
//! hall of fame prints (`km` stays `km`).

mod registry;

use std::fmt::Write;

use registry::{ENTRIES, N_UNITS};

/// DynamicQuantities stores each exponent as a `FixedRational{Int32, 25200}`: a numerator over
/// this denominator.
pub const DENOM: i32 = 25200;

/// SI dimension exponents, in units of `1/DENOM`, in DynamicQuantities' order: length (m), mass
/// (kg), time (s), current (A), temperature (K), luminous intensity (cd), amount (mol).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Dimensions(pub [i32; 7]);

impl Dimensions {
    pub const NAMES: [&'static str; 7] = ["m", "kg", "s", "A", "K", "cd", "mol"];

    pub fn is_dimensionless(self) -> bool {
        self.0 == [0; 7]
    }

    pub fn inv(self) -> Self {
        Self(self.0.map(i32::wrapping_neg))
    }

    /// `self ^ (p / DENOM)`, truncating each exponent toward zero as `FixedRational`
    /// multiplication does.
    pub fn pow(self, p: i32) -> Self {
        Self(self.0.map(|n| fixed_mul(n, p)))
    }
}

impl std::ops::Mul for Dimensions {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i].wrapping_add(other.0[i])))
    }
}

impl std::ops::Div for Dimensions {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i].wrapping_sub(other.0[i])))
    }
}

fn fixed_mul(n: i32, p: i32) -> i32 {
    (i64::from(n) * i64::from(p) / i64::from(DENOM)) as i32
}

/// One registered unit or constant.
pub(crate) struct Entry {
    name: &'static str,
    display: &'static str,
    scale: f64,
    dims: Dimensions,
}

/// A unit as SymbolicRegression.jl keeps it for a feature or the target.
#[derive(Clone, Debug, PartialEq)]
pub struct Unit {
    /// The SI value of the unit: `1000` for `km`.
    pub scale: f64,
    pub dims: Dimensions,
    symbolic: Symbolic,
}

/// A symbolic quantity: a value times registry entries raised to exponents (in `1/DENOM`), sorted
/// by entry.
#[derive(Clone, Debug, PartialEq)]
struct Symbolic {
    value: f64,
    terms: Vec<(usize, i32)>,
}

impl Unit {
    pub fn dimensionless() -> Self {
        Self {
            scale: 1.0,
            dims: Dimensions::default(),
            symbolic: Symbolic {
                value: 1.0,
                terms: Vec::new(),
            },
        }
    }

    /// Parses a unit string as DynamicQuantities' `uparse` and `sym_uparse` do: registered units
    /// and prefixes (`km/s^2`), `Constants.G`, numbers, `*`, `/`, `^`, `//`, `+`, `-`, and
    /// parentheses. An empty string is dimensionless, as in SymbolicRegression.jl.
    pub fn parse(src: &str) -> Result<Self, String> {
        if src.trim().is_empty() {
            return Ok(Self::dimensionless());
        }
        let err = |message: String| format!("cannot parse unit `{src}`: {message}");
        let tokens = lex(src).map_err(err)?;
        let mut parser = Parser {
            tokens: &tokens,
            pos: 0,
        };
        let value = parser.sum().map_err(err)?;
        if let Some(token) = parser.tokens.get(parser.pos) {
            return Err(err(format!("unexpected `{token}`")));
        }
        let q = value.into_quantity();
        Ok(Self {
            scale: q.si,
            dims: q.dims,
            symbolic: Symbolic {
                value: q.sym,
                terms: q.terms,
            },
        })
    }

    /// The unit as SymbolicRegression.jl prints it after a variable: `[m s⁻¹]`, `[2.0 km]` when
    /// its symbolic value is not 1, and nothing for a dimensionless unit of value 1.
    pub fn label(&self) -> String {
        let terms = self
            .symbolic
            .terms
            .iter()
            .map(|&(i, p)| {
                let name = ENTRIES[i].display;
                if p == DENOM {
                    name.to_string()
                } else {
                    format!("{name}{}", superscript(p))
                }
            })
            .collect::<Vec<_>>()
            .join(" ");
        if self.symbolic.value == 1.0 {
            if terms.is_empty() {
                String::new()
            } else {
                format!("[{terms}]")
            }
        } else {
            format!("[{} {terms}]", julia_float(self.symbolic.value))
        }
    }
}

/// The units of a dataset's features and target, as SymbolicRegression.jl's `X_units` and
/// `y_units`.
#[derive(Clone, Debug, PartialEq)]
pub struct Units {
    pub x: Vec<Unit>,
    pub y: Option<Unit>,
}

impl Units {
    /// SymbolicRegression.jl's rules: `x_units` needs one unit per feature, and target units
    /// alone make every feature dimensionless. `None` when neither is given.
    pub fn parse<S: AsRef<str>>(
        n_features: usize,
        x_units: Option<&[S]>,
        y_units: Option<&str>,
    ) -> Result<Option<Self>, String> {
        let y = y_units.map(Unit::parse).transpose()?;
        let x = match x_units {
            Some(units) => units
                .iter()
                .map(|u| Unit::parse(u.as_ref()))
                .collect::<Result<Vec<_>, _>>()?,
            None if y.is_some() => vec![Unit::dimensionless(); n_features],
            None => return Ok(None),
        };
        if x.len() != n_features {
            return Err(format!(
                "Number of features ({n_features}) does not match number of units ({})",
                x.len()
            ));
        }
        Ok(Some(Self { x, y }))
    }
}

/// The unit annotations SymbolicRegression.jl prints in the hall of fame: `y[kg] = x1[m] * 3.2[?]`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct UnitLabels {
    /// Each feature's [`Unit::label`].
    pub variables: Vec<String>,
    /// `[?]` after each constant, whose units are free, unless constants are dimensionless.
    pub constant: &'static str,
    /// The target's label, or `[?]` when only the features have units.
    pub target: String,
}

impl UnitLabels {
    pub fn new(units: &Units, dimensionless_constants_only: bool) -> Self {
        Self {
            variables: units.x.iter().map(Unit::label).collect(),
            constant: if dimensionless_constants_only { "" } else { "[?]" },
            target: units.y.as_ref().map_or("[?]".to_string(), Unit::label),
        }
    }
}

/// `p / DENOM` as DynamicQuantities prints an exponent: `²`, `⁻¹`, `¹ᐟ²`.
fn superscript(p: i32) -> String {
    let plain = if p % DENOM == 0 {
        (p / DENOM).to_string()
    } else {
        let g = gcd(p.unsigned_abs(), DENOM as u32) as i32;
        format!("{}/{}", p / g, DENOM / g)
    };
    plain
        .chars()
        .map(|c| match c {
            '-' => '⁻',
            '/' => 'ᐟ',
            d => ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'][d.to_digit(10).expect("a digit") as usize],
        })
        .collect()
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// `x` as Julia prints a `Float64`: shortest digits, `.0` on integers, and exponent notation
/// outside `1e-4 <= |x| < 1e6`.
fn julia_float(x: f64) -> String {
    let sci = format!("{x:e}");
    let (mantissa, exp) = sci.split_once('e').expect("`{:e}` has an exponent");
    let exp: i32 = exp.parse().expect("an integer exponent");
    if x == 0.0 || !x.is_finite() || (-4..6).contains(&exp) {
        return format!("{x:?}");
    }
    let mut out = mantissa.to_string();
    if !out.contains('.') {
        out.push_str(".0");
    }
    let _ = write!(out, "e{exp}");
    out
}

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Int(i64),
    Float(f64),
    Name(String),
    Sym(&'static str),
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::Int(i) => write!(f, "{i}"),
            Token::Float(x) => write!(f, "{x}"),
            Token::Name(s) => f.write_str(s),
            Token::Sym(s) => f.write_str(s),
        }
    }
}

fn lex(src: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut chars = src.char_indices().peekable();
    while let Some(&(start, c)) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
        } else if c.is_ascii_digit() || (c == '.' && src[start + 1..].starts_with(|d: char| d.is_ascii_digit())) {
            let mut end = start;
            let mut float = false;
            while let Some(&(i, d)) = chars.peek() {
                let exponent_sign = (d == '+' || d == '-') && src[..i].ends_with(['e', 'E']);
                if d.is_ascii_digit() || exponent_sign {
                    end = i + d.len_utf8();
                } else if d == '.' || d == 'e' || d == 'E' {
                    float = true;
                    end = i + 1;
                } else {
                    break;
                }
                chars.next();
            }
            let text = &src[start..end];
            tokens.push(if float {
                Token::Float(text.parse().map_err(|_| format!("invalid number `{text}`"))?)
            } else {
                Token::Int(text.parse().map_err(|_| format!("invalid number `{text}`"))?)
            });
        } else if c.is_alphabetic() || c == '_' {
            let mut end = start;
            while let Some(&(i, d)) = chars.peek() {
                if !(d.is_alphanumeric() || d == '_') {
                    break;
                }
                end = i + d.len_utf8();
                chars.next();
            }
            tokens.push(Token::Name(src[start..end].to_string()));
        } else {
            let sym = ["//", "*", "/", "^", "+", "-", "(", ")", "."]
                .into_iter()
                .find(|s| src[start..].starts_with(s))
                .ok_or_else(|| format!("unexpected `{c}`"))?;
            for _ in 0..sym.len() {
                chars.next();
            }
            tokens.push(Token::Sym(sym));
        }
    }
    Ok(tokens)
}

/// A dimensioned value: SI (`si`, `dims`) and symbolic (`sym`, `terms`) at once.
#[derive(Clone, Debug)]
struct Quantity {
    si: f64,
    dims: Dimensions,
    sym: f64,
    terms: Vec<(usize, i32)>,
}

impl Quantity {
    fn number(x: f64) -> Self {
        Self {
            si: x,
            dims: Dimensions::default(),
            sym: x,
            terms: Vec::new(),
        }
    }

    fn entry(i: usize) -> Self {
        let e = &ENTRIES[i];
        Self {
            si: e.scale,
            dims: e.dims,
            sym: 1.0,
            terms: vec![(i, DENOM)],
        }
    }

    fn is_dimensionless(&self) -> bool {
        self.dims.is_dimensionless() && self.terms.is_empty()
    }
}

/// A Julia value inside a unit expression: numbers keep Julia's integer, rational, and float
/// types until they meet a unit, since an exponent's type decides how it is rationalized.
#[derive(Clone, Debug)]
enum Value {
    Int(i64),
    Rational(i64, i64),
    Float(f64),
    Quantity(Quantity),
}

impl Value {
    fn as_f64(&self) -> f64 {
        match self {
            Value::Int(i) => *i as f64,
            Value::Rational(n, d) => *n as f64 / *d as f64,
            Value::Float(x) => *x,
            Value::Quantity(q) => q.si,
        }
    }

    fn into_quantity(self) -> Quantity {
        match self {
            Value::Quantity(q) => q,
            v => Quantity::number(v.as_f64()),
        }
    }
}

fn rational(n: i64, d: i64) -> Result<Value, String> {
    if d == 0 {
        return Err("division by zero in `//`".to_string());
    }
    let g = gcd_i64(n, d) * d.signum();
    Ok(Value::Rational(n / g, d / g))
}

fn gcd_i64(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.unsigned_abs(), b.unsigned_abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a as i64
}

fn merge_terms(a: &[(usize, i32)], b: &[(usize, i32)], sign: i32) -> Vec<(usize, i32)> {
    let mut out = a.to_vec();
    for &(i, p) in b {
        match out.binary_search_by_key(&i, |&(j, _)| j) {
            Ok(k) => out[k].1 = out[k].1.wrapping_add(sign * p),
            Err(k) => out.insert(k, (i, sign * p)),
        }
    }
    out.retain(|&(_, p)| p != 0);
    out
}

fn product(a: Value, b: Value, divide: bool) -> Result<Value, String> {
    Ok(match (a, b) {
        (a @ Value::Quantity(_), b) | (a, b @ Value::Quantity(_)) => {
            let (a, b) = (a.into_quantity(), b.into_quantity());
            let sign = if divide { -1 } else { 1 };
            Value::Quantity(Quantity {
                si: if divide { a.si / b.si } else { a.si * b.si },
                dims: if divide { a.dims / b.dims } else { a.dims * b.dims },
                sym: if divide { a.sym / b.sym } else { a.sym * b.sym },
                terms: merge_terms(&a.terms, &b.terms, sign),
            })
        }
        (a @ Value::Float(_), b) | (a, b @ Value::Float(_)) => Value::Float(if divide {
            a.as_f64() / b.as_f64()
        } else {
            a.as_f64() * b.as_f64()
        }),
        (Value::Int(x), Value::Int(y)) if !divide => Value::Int(x.wrapping_mul(y)),
        (Value::Int(x), Value::Int(y)) => Value::Float(x as f64 / y as f64),
        (a, b) => {
            let (an, ad) = as_ratio(&a);
            let (bn, bd) = as_ratio(&b);
            if divide {
                rational(an * bd, ad * bn)?
            } else {
                rational(an * bn, ad * bd)?
            }
        }
    })
}

fn as_ratio(v: &Value) -> (i64, i64) {
    match v {
        Value::Int(i) => (*i, 1),
        Value::Rational(n, d) => (*n, *d),
        _ => unreachable!("only integers and rationals are ratios"),
    }
}

fn sum(a: Value, b: Value, subtract: bool) -> Result<Value, String> {
    let sign = if subtract { -1.0 } else { 1.0 };
    Ok(match (a, b) {
        (a @ Value::Quantity(_), b) | (a, b @ Value::Quantity(_)) => {
            let (a, b) = (a.into_quantity(), b.into_quantity());
            if a.dims != b.dims || a.terms != b.terms {
                return Err("`+` and `-` need matching units".to_string());
            }
            Value::Quantity(Quantity {
                si: a.si + sign * b.si,
                sym: a.sym + sign * b.sym,
                ..a
            })
        }
        (a @ Value::Float(_), b) | (a, b @ Value::Float(_)) => Value::Float(a.as_f64() + sign * b.as_f64()),
        (Value::Int(x), Value::Int(y)) => Value::Int(if subtract { x.wrapping_sub(y) } else { x.wrapping_add(y) }),
        (a, b) => {
            let (an, ad) = as_ratio(&a);
            let (bn, bd) = as_ratio(&b);
            rational(an * bd + if subtract { -bn * ad } else { bn * ad }, ad * bd)?
        }
    })
}

fn negate(v: Value) -> Value {
    match v {
        Value::Int(i) => Value::Int(i.wrapping_neg()),
        Value::Rational(n, d) => Value::Rational(-n, d),
        Value::Float(x) => Value::Float(-x),
        Value::Quantity(q) => Value::Quantity(Quantity {
            si: -q.si,
            sym: -q.sym,
            ..q
        }),
    }
}

fn power(base: Value, exp: Value) -> Result<Value, String> {
    let exp = match exp {
        Value::Quantity(q) if q.is_dimensionless() => Value::Float(q.si),
        Value::Quantity(_) => return Err("an exponent must be dimensionless".to_string()),
        e => e,
    };
    match base {
        Value::Quantity(q) => {
            // DynamicQuantities rationalizes the exponent, then raises the value to the
            // rationalized power.
            let p = match exp {
                Value::Int(i) => i.wrapping_mul(i64::from(DENOM)) as i32,
                Value::Rational(n, d) => (i128::from(n) * i128::from(DENOM) / i128::from(d)) as i32,
                Value::Float(x) if x.fract() == 0.0 => (x.round() as i64).wrapping_mul(i64::from(DENOM)) as i32,
                Value::Float(x) => {
                    let (n, d) = rationalize(x);
                    (i128::from(n) * i128::from(DENOM) / i128::from(d)) as i32
                }
                Value::Quantity(_) => unreachable!("converted above"),
            };
            let value_power = match exp {
                Value::Int(i) => i as f64,
                _ => f64::from(p) / f64::from(DENOM),
            };
            let mut terms: Vec<(usize, i32)> = q.terms.iter().map(|&(i, e)| (i, fixed_mul(e, p))).collect();
            terms.retain(|&(_, e)| e != 0);
            Ok(Value::Quantity(Quantity {
                si: q.si.powf(value_power),
                dims: q.dims.pow(p),
                sym: q.sym.powf(value_power),
                terms,
            }))
        }
        Value::Int(b) => Ok(match exp {
            Value::Int(e) if e >= 0 => Value::Int(b.wrapping_pow(e as u32)),
            e => Value::Float((b as f64).powf(e.as_f64())),
        }),
        Value::Rational(n, d) => match exp {
            Value::Int(e) if e >= 0 => rational(n.wrapping_pow(e as u32), d.wrapping_pow(e as u32)),
            Value::Int(e) => rational(
                d.wrapping_pow(e.unsigned_abs() as u32),
                n.wrapping_pow(e.unsigned_abs() as u32),
            ),
            e => Ok(Value::Float((n as f64 / d as f64).powf(e.as_f64()))),
        },
        Value::Float(b) => Ok(Value::Float(b.powf(exp.as_f64()))),
    }
}

/// Julia's `rationalize(Int, x)`: the simplest fraction within `eps(x)` of `x`.
fn rationalize(x: f64) -> (i64, i64) {
    let tol = {
        let exponent = ((x.abs().to_bits() >> 52) & 0x7ff) as i32 - 1023;
        2f64.powi(exponent - 52)
    };
    let (mut p, mut q) = (if x < 0.0 { -1i64 } else { 1 }, 0i64);
    let (mut pp, mut qq) = (0i64, 1i64);
    let mut x = x.abs();
    let mut a = x.trunc();
    let mut r = x - a;
    let mut y = 1.0;
    let (mut nt, mut t, mut tt) = (tol, 0.0, tol);
    while r > nt {
        let ia = a as i64;
        let (Some(np), Some(nq)) = (
            ia.checked_mul(p).and_then(|v| v.checked_add(pp)),
            ia.checked_mul(q).and_then(|v| v.checked_add(qq)),
        ) else {
            return (p, q);
        };
        (p, pp) = (np, p);
        (q, qq) = (nq, q);
        (x, y) = (y, r);
        r = x % y;
        a = ((x - r) / y).round();
        (t, tt) = (nt, t);
        nt = a * t + tt;
    }
    let a = ((x - tt) / (y + t)).ceil() as i64;
    match (
        a.checked_mul(p).and_then(|v| v.checked_add(pp)),
        a.checked_mul(q).and_then(|v| v.checked_add(qq)),
    ) {
        (Some(np), Some(nq)) => (np, nq),
        _ => (p, q),
    }
}

/// Precedence follows Julia: `+ -` < `* /` < unary `- +` < `//` < numeric juxtaposition (`2m`) <
/// `^`, with `^` right-associative.
struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn eat(&mut self, sym: &'static str) -> bool {
        if self.peek() == Some(&Token::Sym(sym)) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn sum(&mut self) -> Result<Value, String> {
        let mut lhs = self.product()?;
        loop {
            if self.eat("+") {
                lhs = sum(lhs, self.product()?, false)?;
            } else if self.eat("-") {
                lhs = sum(lhs, self.product()?, true)?;
            } else {
                return Ok(lhs);
            }
        }
    }

    fn product(&mut self) -> Result<Value, String> {
        let mut lhs = self.unary()?;
        loop {
            if self.eat("*") {
                lhs = product(lhs, self.unary()?, false)?;
            } else if self.eat("/") {
                lhs = product(lhs, self.unary()?, true)?;
            } else {
                return Ok(lhs);
            }
        }
    }

    fn unary(&mut self) -> Result<Value, String> {
        if self.eat("-") {
            Ok(negate(self.unary()?))
        } else if self.eat("+") {
            self.unary()
        } else {
            self.rational()
        }
    }

    fn rational(&mut self) -> Result<Value, String> {
        let mut lhs = self.juxtaposition()?;
        while self.eat("//") {
            let rhs = self.juxtaposition()?;
            lhs = match (&lhs, &rhs) {
                (Value::Int(_) | Value::Rational(..), Value::Int(_) | Value::Rational(..)) => {
                    let (an, ad) = as_ratio(&lhs);
                    let (bn, bd) = as_ratio(&rhs);
                    rational(an * bd, ad * bn)?
                }
                _ => return Err("`//` needs integers".to_string()),
            };
        }
        Ok(lhs)
    }

    fn juxtaposition(&mut self) -> Result<Value, String> {
        let lhs = self.power()?;
        let number = matches!(self.tokens.get(self.pos - 1), Some(Token::Int(_) | Token::Float(_)));
        if number && matches!(self.peek(), Some(Token::Name(_) | Token::Sym("("))) {
            return product(lhs, self.power()?, false);
        }
        Ok(lhs)
    }

    fn power(&mut self) -> Result<Value, String> {
        let base = self.atom()?;
        if !self.eat("^") {
            return Ok(base);
        }
        let exp = if self.eat("-") {
            negate(self.power()?)
        } else {
            self.eat("+");
            self.power()?
        };
        power(base, exp)
    }

    fn atom(&mut self) -> Result<Value, String> {
        let Some(token) = self.peek().cloned() else {
            return Err("unexpected end".to_string());
        };
        self.pos += 1;
        match token {
            Token::Int(i) => Ok(Value::Int(i)),
            Token::Float(x) => Ok(Value::Float(x)),
            Token::Sym("(") => {
                let v = self.sum()?;
                if !self.eat(")") {
                    return Err("missing `)`".to_string());
                }
                Ok(v)
            }
            Token::Name(name) if name == "Constants" => {
                if !self.eat(".") {
                    return Err("expected `.` after `Constants`".to_string());
                }
                match self.peek().cloned() {
                    Some(Token::Name(c)) => {
                        self.pos += 1;
                        let i = (N_UNITS..ENTRIES.len())
                            .find(|&i| ENTRIES[i].name == c)
                            .ok_or_else(|| format!("unknown constant `Constants.{c}`"))?;
                        Ok(Value::Quantity(Quantity::entry(i)))
                    }
                    _ => Err("expected a constant name after `Constants.`".to_string()),
                }
            }
            Token::Name(name) => match ENTRIES[..N_UNITS].iter().position(|e| e.name == name) {
                Some(i) => Ok(Value::Quantity(Quantity::entry(i))),
                None if ENTRIES[N_UNITS..].iter().any(|e| e.name == name) => {
                    Err(format!("`{name}` is a constant, not a unit; write `Constants.{name}`"))
                }
                None => Err(format!("unknown unit `{name}`")),
            },
            token => Err(format!("unexpected `{token}`")),
        }
    }
}
