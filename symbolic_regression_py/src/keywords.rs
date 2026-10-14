//! SymbolicRegression.jl's keyword arguments mapped onto [`Options`]: those of `Options(; ...)`
//! and the search settings `equation_search` takes, such as `niterations` and `parallelism`.

use std::fmt::Display;
use std::ops::AddAssign;

use dynamic_expressions::dsl::{CustomOps, DslFloat, TemplateStructure, parse_operator_tokens};
use dynamic_expressions::operator_enum::presets::{BuiltinOpsF32, BuiltinOpsF64};
use dynamic_expressions::{OpId, OperatorSelectError, OperatorSet, Operators};
use num_traits::{FromPrimitive, ToPrimitive};
use serde::Serialize;
use serde::de::DeserializeOwned;
use symbolic_regression::{
    AdaptiveMutationWeights, Batching, BuiltinMutation, ConstantMutation, Crossover, DefaultsProfile, EarlyStop, Guess,
    InputStream, LossScale, MutationBurst, OptimizerAlgorithm, Options, Parallelism, Reward, TemplateGuessValue,
    parse_early_stop, parse_loss,
};

/// The largest operator arity the search supports: that of the built-in ternary operators.
pub(crate) const D: usize = 3;

/// A float type the search runs in, with an operator set holding every built-in operator and the
/// custom operators the options define.
pub(crate) trait Precision:
    DslFloat + AddAssign + FromPrimitive + ToPrimitive + Display + Serialize + DeserializeOwned
{
    type Ops: OperatorSet<T = Self> + Default + Send + Sync + 'static;
    fn operator_set(custom: CustomOps<Self>) -> Self::Ops;
}

impl Precision for f32 {
    type Ops = BuiltinOpsF32<CustomOps<f32>>;
    fn operator_set(custom: CustomOps<f32>) -> Self::Ops {
        BuiltinOpsF32::new().with_custom(custom)
    }
}

impl Precision for f64 {
    type Ops = BuiltinOpsF64<CustomOps<f64>>;
    fn operator_set(custom: CustomOps<f64>) -> Self::Ops {
        BuiltinOpsF64::new().with_custom(custom)
    }
}

pub(crate) type SearchOptions<T> = Options<T, <T as Precision>::Ops, D>;
type SearchGuesses<T> = Vec<Vec<Guess<T, <T as Precision>::Ops, D>>>;

/// A keyword argument's value, as Python passes it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<Value>),
    Dict(Vec<(Value, Value)>),
}

impl Value {
    fn describe(&self) -> String {
        match self {
            Value::None => "None".into(),
            Value::Bool(b) => if *b { "True" } else { "False" }.into(),
            Value::Int(i) => i.to_string(),
            Value::Float(f) => f.to_string(),
            Value::Str(s) => format!("{s:?}"),
            Value::List(_) => "a list".into(),
            Value::Dict(_) => "a dict".into(),
        }
    }
}

fn expected(key: &str, what: &str, value: &Value) -> String {
    format!("`{key}` must be {what}, got {}", value.describe())
}

fn to_bool(key: &str, value: Value) -> Result<bool, String> {
    match value {
        Value::Bool(b) => Ok(b),
        other => Err(expected(key, "True or False", &other)),
    }
}

fn to_int(key: &str, value: Value) -> Result<i64, String> {
    match value {
        Value::Int(i) => Ok(i),
        other => Err(expected(key, "an integer", &other)),
    }
}

fn to_unsigned<N: TryFrom<i64>>(key: &str, value: Value) -> Result<N, String> {
    let i = to_int(key, value)?;
    N::try_from(i).map_err(|_| format!("`{key}` must be a non-negative integer, got {i}"))
}

fn to_usize(key: &str, value: Value) -> Result<usize, String> {
    to_unsigned(key, value)
}

fn to_u64(key: &str, value: Value) -> Result<u64, String> {
    to_unsigned(key, value)
}

fn to_i32(key: &str, value: Value) -> Result<i32, String> {
    let i = to_int(key, value)?;
    i32::try_from(i).map_err(|_| format!("`{key}` is out of range, got {i}"))
}

fn to_f64(key: &str, value: Value) -> Result<f64, String> {
    let f = match value {
        Value::Int(i) => i as f64,
        Value::Float(f) => f,
        other => return Err(expected(key, "a number", &other)),
    };
    if !f.is_finite() {
        return Err(format!("`{key}` must be finite, got {f}"));
    }
    Ok(f)
}

fn to_f32(key: &str, value: Value) -> Result<f32, String> {
    let f = to_f64(key, value)? as f32;
    if !f.is_finite() {
        return Err(format!("`{key}` is outside the finite float32 range"));
    }
    Ok(f)
}

fn nonnegative(key: &str, f: f64) -> Result<f64, String> {
    if f < 0.0 {
        return Err(format!("`{key}` must be non-negative, got {f}"));
    }
    Ok(f)
}

fn to_nonnegative(key: &str, value: Value) -> Result<f64, String> {
    nonnegative(key, to_f64(key, value)?)
}

fn to_cost(key: &str, value: Value) -> Result<f32, String> {
    nonnegative(key, f64::from(to_f32(key, value)?)).map(|f| f as f32)
}

fn to_probability(key: &str, value: Value) -> Result<f64, String> {
    let p = to_f64(key, value)?;
    if !(0.0..=1.0).contains(&p) {
        return Err(format!("`{key}` must be between 0 and 1, got {p}"));
    }
    Ok(p)
}

fn to_probability32(key: &str, value: Value) -> Result<f32, String> {
    to_probability(key, value).map(|p| p as f32)
}

fn to_positive(key: &str, value: Value) -> Result<usize, String> {
    let n = to_usize(key, value)?;
    if n == 0 {
        return Err(format!("`{key}` must be positive"));
    }
    Ok(n)
}

fn to_precision<T: Precision>(key: &str, value: Value) -> Result<T, String> {
    let f = to_f64(key, value)?;
    let converted = T::from_f64(f).ok_or_else(|| format!("`{key}` is outside the selected precision"))?;
    if !converted.is_finite() {
        return Err(format!("`{key}` is outside the finite range of the selected precision"));
    }
    Ok(converted)
}

fn to_string(key: &str, value: Value) -> Result<String, String> {
    match value {
        Value::Str(s) => Ok(s),
        other => Err(expected(key, "a string", &other)),
    }
}

fn to_list(key: &str, value: Value) -> Result<Vec<Value>, String> {
    match value {
        Value::List(items) => Ok(items),
        other => Err(expected(key, "a list", &other)),
    }
}

fn to_dict(key: &str, value: Value) -> Result<Vec<(Value, Value)>, String> {
    match value {
        Value::Dict(items) => Ok(items),
        other => Err(expected(key, "a dict", &other)),
    }
}

fn to_strings(key: &str, value: Value) -> Result<Vec<String>, String> {
    to_list(key, value)?
        .into_iter()
        .map(|item| to_string(key, item))
        .collect()
}

/// A dict whose keys are strings, such as the parameters of a mutation or plugin.
fn to_named(key: &str, value: Value) -> Result<Vec<(String, Value)>, String> {
    to_dict(key, value)?
        .into_iter()
        .map(|(name, value)| Ok((to_string(key, name)?, value)))
        .collect()
}

/// Keyword arguments, which each setting takes out; any left over are errors. `None` means the
/// default, as `nothing` does in SymbolicRegression.jl.
pub(crate) struct Keywords(Vec<(String, Value)>);

impl Keywords {
    pub(crate) fn new(pairs: Vec<(String, Value)>) -> Self {
        Self(pairs)
    }

    fn take(&mut self, key: &str) -> Option<Value> {
        let i = self.0.iter().position(|(k, _)| k == key)?;
        Some(self.0.remove(i).1).filter(|v| *v != Value::None)
    }

    fn finish(self) -> Result<(), String> {
        let Some((key, _)) = self.0.first() else {
            return Ok(());
        };
        Err(match JULIA_ONLY.iter().find(|(name, _)| name == key) {
            Some((_, reason)) => {
                format!("`{key}` is a SymbolicRegression.jl option that the Rust engine does not support: {reason}")
            }
            None => format!("unknown option `{key}`"),
        })
    }
}

/// SymbolicRegression.jl keywords that the Rust engine cannot honor, and why.
pub(crate) const JULIA_ONLY: &[(&str, &str)] = &[
    (
        "loss_function",
        "it takes a Julia function of the whole expression; use `elementwise_loss`",
    ),
    (
        "loss_function_expression",
        "it takes a Julia function of the whole expression; use `elementwise_loss`",
    ),
    (
        "complexity_mapping",
        "it takes a Julia function; use the `complexity_of_*` options",
    ),
    ("operator_enum_constructor", "it takes a Julia type"),
    ("popmember_type", "it takes a Julia type"),
    ("loss_type", "it takes a Julia type"),
    ("turbo", "it selects LoopVectorization.jl"),
    ("bumper", "it selects Bumper.jl"),
    (
        "autodiff_backend",
        "it selects a Julia automatic differentiation package",
    ),
    ("define_helper_functions", "it defines Julia functions"),
    ("logger", "it takes a Julia logger"),
    ("external_stop", "the Python binding handles cancellation and Ctrl-C"),
    ("numprocs", "multiprocessing runs Julia worker processes"),
    ("procs", "multiprocessing runs Julia worker processes"),
    ("addprocs_function", "multiprocessing runs Julia worker processes"),
    ("heap_size_hint_in_bytes", "multiprocessing runs Julia worker processes"),
    ("worker_timeout", "multiprocessing runs Julia worker processes"),
    ("worker_imports", "multiprocessing runs Julia worker processes"),
    ("runtests", "it tests Julia worker processes"),
    ("return_state", "the search always returns its state"),
    ("npopulations", "it is a deprecated alias; use `populations`"),
    ("npop", "it is a deprecated alias; use `population_size`"),
    ("expression_type", "it is deprecated; use `expression_spec`"),
    ("expression_options", "it is deprecated; use `expression_spec`"),
    ("node_type", "it is deprecated; use `expression_spec`"),
    ("output_file", "it is deprecated; use `output_directory`"),
    ("fast_cycle", "it is deprecated"),
    ("bin_constraints", "it is deprecated; use `constraints`"),
    ("una_constraints", "it is deprecated; use `constraints`"),
];

const DEFAULT_BINARY_OPERATORS: [&str; 4] = ["+", "-", "/", "*"];

/// SymbolicRegression.jl's `Options(; kws...)`, with the `equation_search` settings that the Rust
/// engine keeps in `Options`. Fails, naming the keyword, on an unknown or unsupported keyword or a
/// value of the wrong kind.
pub(crate) fn options<T: Precision>(mut kw: Keywords) -> Result<SearchOptions<T>, String> {
    let mut o = match kw.take("defaults") {
        Some(version) => SearchOptions::<T>::with_defaults(defaults_profile(&to_string("defaults", version)?)?),
        None => SearchOptions::<T>::default(),
    };
    o.operators = operators::<T>(&mut kw)?;

    macro_rules! set {
        (Some $convert:ident: $($field:ident),+ $(,)?) => {
            $(if let Some(v) = kw.take(stringify!($field)) {
                o.$field = Some($convert(stringify!($field), v)?);
            })+
        };
        ($convert:ident: $($field:ident),+ $(,)?) => {
            $(if let Some(v) = kw.take(stringify!($field)) {
                o.$field = $convert(stringify!($field), v)?;
            })+
        };
    }
    set!(to_usize: ncycles_per_iteration, optimizer_nrestarts, optimizer_iterations, optimizer_f_calls_limit,
        optimizer_g_calls_limit, topn, niterations, maxsize);
    set!(to_cost: warmup_maxsize_by);
    set!(to_positive: populations, population_size, tournament_selection_n, print_precision);
    set!(to_nonnegative: parsimony, adaptive_parsimony_scaling, alpha, perturbation_factor,
        timeout_in_seconds, dimensional_constraint_penalty, optimizer_g_abstol, optimizer_x_abstol, optimizer_f_reltol);
    set!(to_probability: crossover_probability, probability_negate_constant, optimizer_probability,
        fraction_replaced, fraction_replaced_hof, fraction_replaced_guesses);
    set!(to_cost: complexity_of_constants);
    set!(to_probability32: tournament_selection_p);
    set!(to_bool: dimensionless_constants_only, use_frequency, use_frequency_in_tournament, skip_mutation_failures,
        annealing, should_optimize_constants, migration, hof_migration, deterministic, save_to_file, progress,
        use_tracing, use_baseline);
    set!(to_u64: max_evals);
    set!(Some to_positive: maxdepth, terminal_width);
    set!(Some to_usize: batch_size);
    set!(Some to_bool: should_simplify);
    set!(Some to_i32: verbosity);
    set!(Some to_u64: seed);
    set!(Some to_string: output_directory, run_id);
    if let Some(v) = kw.take("output_style") {
        o.output_style = match to_string("output_style", v)?.as_str() {
            "auto" => symbolic_regression::OutputStyle::Auto,
            "plain" => symbolic_regression::OutputStyle::Plain,
            "ansi" => symbolic_regression::OutputStyle::Ansi,
            other => {
                return Err(format!(
                    "`output_style` must be \"auto\", \"plain\", or \"ansi\", got {other:?}"
                ));
            }
        };
    }

    if let Some(v) = kw.take("tracing_file") {
        o.tracing_file = to_string("tracing_file", v)?.into();
    }
    if let Some(v) = kw.take("loss_scale") {
        o.loss_scale = to_string("loss_scale", v)?.parse::<LossScale>()?;
    }
    if let Some(v) = kw.take("optimizer_algorithm") {
        o.optimizer_algorithm = match to_string("optimizer_algorithm", v)?.as_str() {
            "BFGS" => OptimizerAlgorithm::BFGS,
            "NelderMead" => OptimizerAlgorithm::NelderMead,
            other => {
                return Err(format!(
                    "`optimizer_algorithm` must be \"BFGS\" or \"NelderMead\", got {other:?}"
                ));
            }
        };
    }
    if let Some(v) = kw.take("optimizer_options") {
        for (name, value) in to_named("optimizer_options", v)? {
            let key = format!("optimizer_options[{name:?}]");
            match name.as_str() {
                "iterations" => o.optimizer_iterations = to_usize(&key, value)?,
                "f_calls_limit" => o.optimizer_f_calls_limit = to_usize(&key, value)?,
                "g_calls_limit" => o.optimizer_g_calls_limit = to_usize(&key, value)?,
                "g_abstol" => o.optimizer_g_abstol = to_nonnegative(&key, value)?,
                "x_abstol" => o.optimizer_x_abstol = to_nonnegative(&key, value)?,
                "f_reltol" => o.optimizer_f_reltol = to_nonnegative(&key, value)?,
                other => {
                    return Err(format!(
                        "`optimizer_options` has no setting `{other}` in the Rust engine"
                    ));
                }
            }
        }
    }
    if let Some(v) = kw.take("batching") {
        o.batching = match v {
            Value::Bool(true) => Batching::On,
            Value::Bool(false) => Batching::Off,
            Value::Str(s) if s == "auto" => Batching::Auto,
            other => return Err(expected("batching", "True, False, or \"auto\"", &other)),
        };
    }
    if let Some(v) = kw.take("parallelism") {
        o.parallelism = match to_string("parallelism", v)?.as_str() {
            "serial" => Parallelism::Serial,
            "multithreading" => Parallelism::Multithreading,
            "multiprocessing" => {
                return Err(
                    "`parallelism=\"multiprocessing\"` runs Julia worker processes; use \"multithreading\"".into(),
                );
            }
            other => {
                return Err(format!(
                    "`parallelism` must be \"serial\" or \"multithreading\", got {other:?}"
                ));
            }
        };
    }
    if let Some(v) = kw.take("input_stream") {
        o.input_stream = match to_string("input_stream", v)?.as_str() {
            #[cfg(unix)]
            "stdin" => Some(InputStream::Stdin),
            #[cfg(not(unix))]
            "stdin" => return Err("`input_stream=\"stdin\"` is unsupported on this platform; use \"devnull\"".into()),
            "devnull" => None,
            other => {
                return Err(format!(
                    "`input_stream` must be \"stdin\" or \"devnull\", got {other:?}"
                ));
            }
        };
    }
    if let Some(v) = kw.take("elementwise_loss") {
        o.loss = parse_loss::<T>(&to_string("elementwise_loss", v)?).map_err(|e| format!("`elementwise_loss`: {e}"))?;
    }
    if let Some(v) = kw.take("early_stop_condition") {
        o.early_stop_condition = Some(match v {
            Value::Str(src) => {
                if let Some(threshold) = dynamic_expressions::dsl::lexer::number_literal(&src) {
                    EarlyStop::below(to_precision::<T>("early_stop_condition", Value::Float(threshold))?)
                } else {
                    parse_early_stop::<T>(&src).map_err(|e| format!("`early_stop_condition`: {e}"))?
                }
            }
            number => EarlyStop::below(to_precision::<T>("early_stop_condition", number)?),
        });
    }
    if let Some(v) = kw.take("complexity_of_variables") {
        match v {
            Value::List(items) => {
                let costs = items.into_iter().map(|c| to_cost("complexity_of_variables", c));
                o.variable_complexities = Some(costs.collect::<Result<_, _>>()?);
            }
            number => o.complexity_of_variables = to_cost("complexity_of_variables", number)?,
        }
    }
    if let Some(v) = kw.take("complexity_of_operators") {
        for (name, cost) in to_named("complexity_of_operators", v)? {
            let cost = to_cost("complexity_of_operators", cost)?;
            for op in selected(&o.operators, "complexity_of_operators", &name)? {
                o.operator_complexity_overrides.insert(op, cost);
            }
        }
    }
    if let Some(v) = kw.take("constraints") {
        set_constraints(&mut o, v)?;
    }
    if let Some(v) = kw.take("nested_constraints") {
        for (outer, inner) in to_named("nested_constraints", v)? {
            let roots = selected(&o.operators, "nested_constraints", &outer)?;
            for (inner, max) in to_named("nested_constraints", inner)? {
                let max = u8::try_from(to_int("nested_constraints", max.clone())?)
                    .map_err(|_| expected("nested_constraints", "a count from 0 to 255", &max))?;
                for nested in selected(&o.operators, "nested_constraints", &inner)? {
                    for &root in &roots {
                        o.nested_constraints.add_nested_constraint(root, nested, max);
                    }
                }
            }
        }
    }
    if let Some(v) = kw.take("expression_spec") {
        o.template = Some(template(v, &o.operators.set)?);
    }

    let mutation_weights = kw.take("mutation_weights");
    let default_mutations = kw.take("default_mutations");
    if mutation_weights.is_some() && default_mutations.is_some() {
        return Err("`mutation_weights` and `default_mutations` cannot be used together.".into());
    }
    if let Some(v) = mutation_weights {
        set_mutation_weights(&mut o, v)?;
    }
    if let Some(v) = default_mutations {
        o.default_mutations = Some(weighted("default_mutations", v, mutation)?);
    }
    if let Some(v) = kw.take("mutations") {
        o.mutations = weighted("mutations", v, mutation)?;
    }
    if let Some(v) = kw.take("default_crossovers") {
        o.default_crossovers = Some(weighted("default_crossovers", v, crossover)?);
    }
    if let Some(v) = kw.take("crossovers") {
        o.crossovers = weighted("crossovers", v, crossover)?;
    }
    let plugins = kw.take("plugins");
    let default_plugins = kw.take("default_plugins");
    set_plugins(&mut o, plugins, default_plugins)?;

    kw.finish()?;
    if o.tournament_selection_p == 0.0 {
        return Err("`tournament_selection_p` must be positive".into());
    }
    o.validate()?;
    if o.annealing && o.alpha <= 0.0 {
        return Err("`alpha` must be finite and positive.".into());
    }
    let total_weight: f64 = o.resolved_mutations().map(|(_, weight)| weight).sum();
    if !total_weight.is_finite() {
        return Err("the sum of mutation weights must be finite".into());
    }
    let total_crossover_weight: f64 = o.resolved_crossovers().map(|(_, weight)| weight).sum();
    if !total_crossover_weight.is_finite() {
        return Err("the sum of crossover weights must be finite".into());
    }
    if o.crossover_probability > 0.0 && !o.resolved_crossovers().any(|(_, w)| w > 0.0) {
        return Err("at least one crossover weight must be positive when crossover_probability > 0".into());
    }
    Ok(o)
}

/// SymbolicRegression.jl's `defaults`, a version such as `"0.24.5"`: versions before 1.0.0 select
/// the legacy hyperparameters, and 2.0.0 prereleases the current ones.
fn defaults_profile(version: &str) -> Result<DefaultsProfile, String> {
    let invalid = || format!("`defaults` must be a version such as \"1.0.0\", got {version:?}");
    let metadata = |s: &str| {
        s.split('.')
            .all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-'))
    };
    let version = match version.split_once('+') {
        Some((release, build)) if metadata(build) => release,
        Some(_) => return Err(invalid()),
        None => version,
    };
    let (release, prerelease) = match version.split_once('-') {
        Some((release, suffix)) if metadata(suffix) => (release, true),
        Some(_) => return Err(invalid()),
        None => (version, false),
    };
    let mut parts = release.split('.').map(|p| {
        if p.is_empty() || !p.bytes().all(|b| b.is_ascii_digit()) {
            return Err(invalid());
        }
        p.parse::<u64>().map_err(|_| invalid())
    });
    let major = parts.next().ok_or_else(invalid)??;
    let minor = parts.next().transpose()?.unwrap_or(0);
    let patch = parts.next().transpose()?.unwrap_or(0);
    if parts.next().is_some() {
        return Err(invalid());
    }
    Ok(match (major, minor, patch) {
        (0, ..) => DefaultsProfile::Legacy,
        (1, 0, 0) if prerelease => DefaultsProfile::Legacy,
        (1, ..) => DefaultsProfile::V1,
        _ => DefaultsProfile::V2,
    })
}

/// `operators`, which maps each arity to its operators as SymbolicRegression.jl's
/// `OperatorEnum(1 => ..., 2 => ...)` does, or else `binary_operators` and `unary_operators`.
/// Each operator is a built-in name or symbol (`"cos"`, `"+"`) or a definition (`"inv(x) = 1/x"`).
fn operators<T: Precision>(kw: &mut Keywords) -> Result<Operators<T::Ops, D>, String> {
    let by_arity = kw.take("operators");
    let binary = kw.take("binary_operators");
    let unary = kw.take("unary_operators");
    let tokens: Vec<(u8, String)> = match by_arity {
        Some(_) if binary.is_some() || unary.is_some() => {
            return Err("`operators` cannot be combined with `binary_operators` or `unary_operators`".into());
        }
        Some(by_arity) => {
            let mut tokens = Vec::new();
            for (arity, list) in to_dict("operators", by_arity)? {
                let arity = match arity {
                    Value::Int(a @ 1..=3) => a as u8,
                    other => return Err(expected("each key of `operators`", "an arity from 1 to 3", &other)),
                };
                tokens.extend(to_strings("operators", list)?.into_iter().map(|t| (arity, t)));
            }
            tokens
        }
        None => {
            let binary = match binary {
                Some(v) => to_strings("binary_operators", v)?,
                None => DEFAULT_BINARY_OPERATORS.map(String::from).into(),
            };
            let unary = unary.map_or(Ok(Vec::new()), |v| to_strings("unary_operators", v))?;
            let unary = unary.into_iter().map(|t| (1, t));
            unary.chain(binary.into_iter().map(|t| (2, t))).collect()
        }
    };
    let (custom, names) =
        parse_operator_tokens::<T, _>(tokens.iter().map(|(_, t)| t)).map_err(|e| format!("`operators`: {e}"))?;
    let mut operators = Operators::new(T::operator_set(custom));
    for (&(arity, _), name) in tokens.iter().zip(&names) {
        let op = operators
            .set
            .lookup_with_arity(name, arity)
            .map_err(|e| match operators.set.lookup(name) {
                Ok(op) => {
                    let list = ["unary", "binary", "ternary"][usize::from(arity) - 1];
                    let plural = if op.arity == 1 { "" } else { "s" };
                    format!(
                        "operator `{name}` takes {} argument{plural}, but it is in the {list} operator list",
                        op.arity
                    )
                }
                Err(_) => OperatorSelectError::Lookup(e).to_string(),
            })?;
        if operators.ops_by_arity[usize::from(arity) - 1].contains(&op) {
            return Err(OperatorSelectError::Duplicate(name.clone()).to_string());
        }
        operators.push(op);
    }
    if operators.ops_by_arity.iter().all(Vec::is_empty) {
        return Err(OperatorSelectError::Empty.to_string());
    }
    Ok(operators)
}

/// The search's operators that `name` names, in every arity.
fn selected<Ops: OperatorSet>(operators: &Operators<Ops, D>, key: &str, name: &str) -> Result<Vec<OpId>, String> {
    let ops: Vec<OpId> = operators
        .set
        .lookup_all(name)
        .into_iter()
        .filter(|op| operators.ops_by_arity[usize::from(op.arity) - 1].contains(op))
        .collect();
    if ops.is_empty() {
        return Err(format!(
            "`{key}` names `{name}`, which is not one of the search's operators"
        ));
    }
    Ok(ops)
}

/// `constraints`: each operator's largest argument complexities, an integer for a unary operator
/// and one per argument otherwise, with -1 for no limit.
fn set_constraints<T: Precision>(o: &mut SearchOptions<T>, value: Value) -> Result<(), String> {
    let key = "constraints";
    let entries = to_dict(key, value)?;
    let by_arity = entries.first().is_some_and(|(name, _)| matches!(name, Value::Int(_)));
    for (name, limits) in entries {
        if by_arity {
            let arity = match name {
                Value::Int(a @ 1..=3) => a as u8,
                other => return Err(expected(key, "integer arity keys from 1 to 3", &other)),
            };
            if limits == Value::None {
                continue;
            }
            let limits = to_list(key, limits)?;
            let ops = &o.operators.ops_by_arity[usize::from(arity) - 1];
            if limits.len() != ops.len() {
                return Err(format!(
                    "`constraints[{arity}]` needs {} entries, one per selected operator, got {}",
                    ops.len(),
                    limits.len()
                ));
            }
            for (&op, limits) in ops.iter().zip(limits) {
                argument_limits(&mut o.op_constraints, op, limits)?;
            }
        } else {
            let name = to_string(key, name)?;
            let arity = match &limits {
                Value::List(items) => items.len(),
                _ => 1,
            };
            let op = selected(&o.operators, key, &name)?
                .into_iter()
                .find(|op| usize::from(op.arity) == arity)
                .ok_or_else(|| format!("`constraints` gives `{name}` {arity} limit(s), but no operator `{name}` of the search takes {arity} argument(s)"))?;
            argument_limits(&mut o.op_constraints, op, limits)?;
        }
    }
    Ok(())
}

fn argument_limits(
    constraints: &mut symbolic_regression::OpConstraints<D>,
    op: OpId,
    limits: Value,
) -> Result<(), String> {
    let limits = match limits {
        Value::List(items) => items,
        single => vec![single],
    };
    if limits.len() != usize::from(op.arity) {
        return Err(format!(
            "`constraints` needs {} argument limits for this operator, got {}",
            op.arity,
            limits.len()
        ));
    }
    for (arg, limit) in limits.into_iter().enumerate() {
        match to_int("constraints", limit)? {
            -1 => {}
            limit => {
                let limit = u16::try_from(limit).map_err(|_| {
                    format!("`constraints` limits must be -1 or an integer from 0 to 65535, got {limit}")
                })?;
                constraints.set_op_arg_constraint(op, arg, limit);
            }
        }
    }
    Ok(())
}

/// `expression_spec`: SymbolicRegression.jl's `TemplateExpressionSpec`, as PySR's
/// `TemplateExpressionSpec` holds it.
fn template<Ops: OperatorSet>(value: Value, ops: &Ops) -> Result<TemplateStructure, String> {
    let key = "expression_spec";
    let (mut combine, mut expressions, mut variable_names, mut parameters) = (None, None, None, Vec::new());
    let mut num_features = None;
    for (name, value) in to_named(key, value)? {
        match name.as_str() {
            "combine" => combine = Some(to_string("expression_spec[\"combine\"]", value)?),
            "expressions" => expressions = Some(to_strings("expression_spec[\"expressions\"]", value)?),
            "variable_names" => variable_names = Some(to_strings("expression_spec[\"variable_names\"]", value)?),
            "num_features" if value == Value::None => {}
            "num_features" => {
                num_features = Some(
                    to_list("expression_spec[\"num_features\"]", value)?
                        .into_iter()
                        .map(|n| to_usize("expression_spec[\"num_features\"]", n))
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
            "parameters" if value == Value::None => {}
            "parameters" => {
                for (p, len) in to_named("expression_spec[\"parameters\"]", value)? {
                    parameters.push((p, to_usize("expression_spec[\"parameters\"]", len)?));
                }
            }
            other => return Err(format!("`expression_spec` has no field `{other}`")),
        }
    }
    let missing = |field: &str| format!("`expression_spec` needs `{field}`");
    TemplateStructure::new(
        &combine.ok_or_else(|| missing("combine"))?,
        &expressions.ok_or_else(|| missing("expressions"))?,
        num_features.as_deref(),
        &variable_names.ok_or_else(|| missing("variable_names"))?,
        &parameters,
        ops,
    )
    .map_err(|e| format!("`expression_spec`: {e}"))
}

fn set_mutation_weights<T: Precision>(o: &mut SearchOptions<T>, value: Value) -> Result<(), String> {
    o.mutation_weights = symbolic_regression::MutationWeights::default();
    let w = &mut o.mutation_weights;
    for (name, weight) in to_named("mutation_weights", value)? {
        let slot = match name.as_str() {
            "mutate_constant" => &mut w.mutate_constant,
            "mutate_operator" => &mut w.mutate_operator,
            "mutate_feature" => &mut w.mutate_feature,
            "swap_operands" => &mut w.swap_operands,
            "rotate_tree" => &mut w.rotate_tree,
            "add_node" => &mut w.add_node,
            "insert_node" => &mut w.insert_node,
            "delete_node" => &mut w.delete_node,
            "simplify" => &mut w.simplify,
            "randomize" => &mut w.randomize,
            "do_nothing" => &mut w.do_nothing,
            "optimize" => &mut w.optimize,
            "form_connection" | "break_connection" => {
                return Err(format!(
                    "`mutation_weights[{name:?}]` edits graph expressions, which the Rust engine does not search"
                ));
            }
            "backsolve" => &mut w.backsolve,
            other => return Err(format!("`mutation_weights` has no field `{other}`")),
        };
        *slot = to_nonnegative(&format!("mutation_weights[{name:?}]"), weight)?;
    }
    Ok(())
}

/// A list of `(item, weight)` pairs, such as SymbolicRegression.jl's `mutations`.
fn weighted<I>(key: &str, value: Value, item: fn(&str, Value) -> Result<I, String>) -> Result<Vec<(I, f64)>, String> {
    to_list(key, value)?
        .into_iter()
        .map(|pair| match pair {
            Value::List(pair) if pair.len() == 2 => {
                let [entry, weight]: [Value; 2] = pair.try_into().expect("two elements");
                Ok((item(key, entry)?, to_nonnegative(key, weight)?))
            }
            other => Err(expected(key, "a list of (item, weight) pairs", &other)),
        })
        .collect()
}

/// A mutation or plugin: its SymbolicRegression.jl type name, or a dict of that name under
/// `"type"` and the constructor's keyword arguments.
fn typed(key: &str, value: Value) -> Result<(String, Vec<(String, Value)>), String> {
    match value {
        Value::Str(name) => Ok((name, Vec::new())),
        Value::Dict(_) => {
            let mut fields = to_named(key, value)?;
            let i = fields
                .iter()
                .position(|(name, _)| name == "type")
                .ok_or_else(|| format!("each dict in `{key}` needs a \"type\""))?;
            let name = to_string(key, fields.remove(i).1)?;
            Ok((name, fields))
        }
        other => Err(expected(key, "a type name or a dict with a \"type\"", &other)),
    }
}

fn unknown_field(type_name: &str, field: &str) -> String {
    format!("`{type_name}` has no field `{field}`")
}

fn mutation(key: &str, value: Value) -> Result<BuiltinMutation, String> {
    let (name, fields) = typed(key, value)?;
    let mutation = match name.as_str() {
        "ConstantMutation" => {
            // SymbolicRegression.jl's `ConstantMutation()` defaults, not `Options`' perturbation.
            let mut c = ConstantMutation {
                perturbation_factor: 0.086,
                probability_negate: 0.01,
            };
            for (field, value) in fields {
                match field.as_str() {
                    "perturbation_factor" => c.perturbation_factor = to_nonnegative(key, value)?,
                    "probability_negate" => c.probability_negate = to_probability(key, value)?,
                    other => return Err(unknown_field(&name, other)),
                }
            }
            return Ok(BuiltinMutation::Constant(c));
        }
        "BacksolveMutation" => {
            let mut b = symbolic_regression::backsolve::BacksolveOptions::default();
            for (field, value) in fields {
                match field.as_str() {
                    "max_library_size" => b.max_library_size = to_positive(key, value)?,
                    "max_terms" => b.max_terms = to_positive(key, value)?,
                    "min_improvement" => b.min_improvement = to_probability(key, value)?,
                    "node_attempts" => b.node_attempts = to_positive(key, value)?,
                    other => return Err(unknown_field(&name, other)),
                }
            }
            return Ok(BuiltinMutation::Backsolve(b));
        }
        "OperatorMutation" => BuiltinMutation::Operator,
        "FeatureMutation" => BuiltinMutation::Feature,
        "SwapOperandsMutation" => BuiltinMutation::SwapOperands,
        "RotateTreeMutation" => BuiltinMutation::RotateTree,
        "AddNodeMutation" => BuiltinMutation::AddNode,
        "InsertNodeMutation" => BuiltinMutation::InsertNode,
        "DeleteNodeMutation" => BuiltinMutation::DeleteNode,
        "SimplifyMutation" => BuiltinMutation::Simplify,
        "RandomizeMutation" => BuiltinMutation::Randomize,
        "DoNothingMutation" => BuiltinMutation::DoNothing,
        "OptimizeMutation" => BuiltinMutation::Optimize,
        "FormConnectionMutation" | "BreakConnectionMutation" => {
            return Err(format!(
                "`{name}` edits graph expressions, which the Rust engine does not search"
            ));
        }
        other => return Err(format!("`{key}` has an unknown mutation `{other}`")),
    };
    match fields.first() {
        Some((field, _)) => Err(unknown_field(&name, field)),
        None => Ok(mutation),
    }
}

fn crossover(key: &str, value: Value) -> Result<Crossover, String> {
    let (name, fields) = typed(key, value)?;
    if name != "SubtreeCrossover" {
        return Err(format!("`{key}` has an unknown crossover `{name}`"));
    }
    match fields.first() {
        Some((field, _)) => Err(unknown_field(&name, field)),
        None => Ok(Crossover::Subtree),
    }
}

/// SymbolicRegression.jl's `plugins` and `default_plugins`: the defaults, built from `annealing`,
/// `alpha`, `use_frequency`, and `use_frequency_in_tournament` unless given, follow the plugins,
/// leaving out any whose type a plugin already has.
fn set_plugins<T: Precision>(
    o: &mut SearchOptions<T>,
    plugins: Option<Value>,
    default_plugins: Option<Value>,
) -> Result<(), String> {
    let mut chosen = match plugins {
        Some(v) => to_list("plugins", v)?
            .into_iter()
            .map(|p| plugin("plugins", p))
            .collect::<Result<Vec<_>, _>>()?,
        None => Vec::new(),
    };
    for (i, p) in chosen.iter().enumerate() {
        if chosen[..i].iter().any(|q| q.same_type(p)) {
            return Err(format!("`plugins` has more than one `{}`", p.type_name()));
        }
    }
    let defaults = match default_plugins {
        Some(v) => to_list("default_plugins", v)?
            .into_iter()
            .map(|p| plugin("default_plugins", p))
            .collect::<Result<Vec<_>, _>>()?,
        None => {
            let mut defaults = Vec::new();
            if o.annealing {
                defaults.push(Plugin::SimulatedAnnealing { alpha: o.alpha });
            }
            if o.use_frequency || o.use_frequency_in_tournament {
                defaults.push(Plugin::AdaptiveParsimony {
                    tournament: o.use_frequency_in_tournament,
                    mutation_acceptance: o.use_frequency,
                });
            }
            defaults.push(Plugin::AdaptiveMutationWeights(AdaptiveMutationWeights::default()));
            defaults
        }
    };
    for (i, p) in defaults.iter().enumerate() {
        if defaults[..i].iter().any(|q| q.same_type(p)) {
            return Err(format!("`default_plugins` has more than one `{}`", p.type_name()));
        }
    }
    for p in defaults {
        if !chosen.iter().any(|q| q.same_type(&p)) {
            chosen.push(p);
        }
    }
    o.annealing = false;
    (o.use_frequency, o.use_frequency_in_tournament) = (false, false);
    o.adaptive_mutation_weights = None;
    o.mutation_burst = None;
    for p in chosen {
        match p {
            Plugin::SimulatedAnnealing { alpha } => (o.annealing, o.alpha) = (true, alpha),
            Plugin::AdaptiveParsimony {
                tournament,
                mutation_acceptance,
            } => (o.use_frequency_in_tournament, o.use_frequency) = (tournament, mutation_acceptance),
            Plugin::AdaptiveMutationWeights(w) => o.adaptive_mutation_weights = Some(w),
            Plugin::MutationBurst(b) => o.mutation_burst = Some(b),
        }
    }
    Ok(())
}

enum Plugin {
    SimulatedAnnealing {
        alpha: f64,
    },
    AdaptiveParsimony {
        tournament: bool,
        mutation_acceptance: bool,
    },
    AdaptiveMutationWeights(AdaptiveMutationWeights),
    MutationBurst(MutationBurst),
}

impl Plugin {
    fn same_type(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }

    fn type_name(&self) -> &'static str {
        match self {
            Plugin::SimulatedAnnealing { .. } => "SimulatedAnnealingPlugin",
            Plugin::AdaptiveParsimony { .. } => "AdaptiveParsimonyPlugin",
            Plugin::AdaptiveMutationWeights(_) => "AdaptiveMutationWeightsPlugin",
            Plugin::MutationBurst(_) => "MutationBurstPlugin",
        }
    }
}

fn plugin(key: &str, value: Value) -> Result<Plugin, String> {
    let (name, fields) = typed(key, value)?;
    let mut plugin = match name.as_str() {
        "SimulatedAnnealingPlugin" => Plugin::SimulatedAnnealing { alpha: 0.1 },
        "AdaptiveParsimonyPlugin" => Plugin::AdaptiveParsimony {
            tournament: true,
            mutation_acceptance: true,
        },
        "AdaptiveMutationWeightsPlugin" => Plugin::AdaptiveMutationWeights(AdaptiveMutationWeights::default()),
        "MutationBurstPlugin" => Plugin::MutationBurst(MutationBurst::default()),
        other => return Err(format!("`{key}` has an unknown plugin `{other}`")),
    };
    for (field, value) in fields {
        match (&mut plugin, field.as_str()) {
            (Plugin::SimulatedAnnealing { alpha }, "alpha") => *alpha = to_f64(key, value)?,
            (Plugin::AdaptiveParsimony { tournament, .. }, "tournament") => *tournament = to_bool(key, value)?,
            (
                Plugin::AdaptiveParsimony {
                    mutation_acceptance, ..
                },
                "mutation_acceptance",
            ) => *mutation_acceptance = to_bool(key, value)?,
            (Plugin::AdaptiveMutationWeights(w), "smoothing") => w.smoothing = to_f64(key, value)?,
            (Plugin::AdaptiveMutationWeights(w), "floor") => w.floor = to_f64(key, value)?,
            (Plugin::AdaptiveMutationWeights(w), "adaptation_strength") => w.adaptation_strength = to_f64(key, value)?,
            (Plugin::AdaptiveMutationWeights(w), "reward") => {
                w.reward = match to_string(key, value)?.as_str() {
                    "cost" => Reward::Cost,
                    "loss" => Reward::Loss,
                    _ => return Err("`reward` must be either `:cost` or `:loss`.".into()),
                }
            }
            (Plugin::MutationBurst(b), "retry_attempts") => b.retry_attempts = to_usize(key, value)?,
            (Plugin::MutationBurst(b), "compound_probability") => b.compound_probability = to_f64(key, value)?,
            (Plugin::MutationBurst(b), "compound_max_steps") => b.compound_max_steps = to_usize(key, value)?,
            (_, other) => return Err(unknown_field(&name, other)),
        }
    }
    match &plugin {
        Plugin::SimulatedAnnealing { alpha } if *alpha <= 0.0 => {
            return Err("`alpha` must be finite and positive.".into());
        }
        Plugin::AdaptiveMutationWeights(w) => w.check()?,
        Plugin::MutationBurst(b) => b.check()?,
        _ => {}
    }
    Ok(plugin)
}

/// SymbolicRegression.jl's `guesses`, as PySR takes them: a list of guesses for a single output,
/// or one list per output. A guess is an equation, or for a template search a dict of each
/// subexpression's equation and parameter vectors.
pub(crate) fn guesses<T: Precision>(value: Value, nout: usize) -> Result<SearchGuesses<T>, String> {
    let key = "guesses";
    let lists = to_list(key, value)?;
    let lists = if nout == 1 && !matches!(lists.first(), Some(Value::List(_))) {
        vec![Value::List(lists)]
    } else {
        lists
    };
    if lists.len() != nout {
        return Err(format!(
            "`guesses` has {} lists, but the search has {nout} outputs",
            lists.len()
        ));
    }
    lists
        .into_iter()
        .map(|list| to_list(key, list)?.into_iter().map(guess).collect())
        .collect()
}

fn guess<T: Precision>(value: Value) -> Result<Guess<T, T::Ops, D>, String> {
    let key = "guesses";
    match value {
        Value::Str(src) => Ok(Guess::Equation(src)),
        Value::Dict(_) => {
            let parts = to_named(key, value)?.into_iter().map(|(name, part)| {
                let part = match part {
                    Value::Str(src) => TemplateGuessValue::Equation(src),
                    Value::List(values) => TemplateGuessValue::Parameter(
                        values
                            .into_iter()
                            .map(|v| to_precision::<T>(key, v))
                            .collect::<Result<_, String>>()?,
                    ),
                    other => return Err(expected(key, "an equation or a parameter vector", &other)),
                };
                Ok((name, part))
            });
            Ok(Guess::Template(parts.collect::<Result<_, String>>()?))
        }
        other => Err(expected(key, "an equation or a dict", &other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_dsl_and_nested_constraints() {
        let o = search_options([
            ("unary_operators", strs(["cos", "exp", "inv(x) = 1/x"])),
            (
                "nested_constraints",
                dict(vec![("cos", dict(vec![("exp", Value::Int(2))]))]),
            ),
            ("elementwise_loss", Value::Str("loss(x, y) = abs(x-y)".into())),
            (
                "early_stop_condition",
                Value::Str("(loss, complexity) -> loss < 0.1 && complexity < 8".into()),
            ),
        ])
        .unwrap();
        let cos = o.operators.set.lookup("cos").unwrap();
        let exp = o.operators.set.lookup("exp").unwrap();
        assert_eq!(o.nested_constraints.limits[&cos], vec![(exp, 2)]);
        assert_eq!(o.loss.loss(&[3.0], &[1.0], None), 2.0);
        let stop = o.early_stop_condition.unwrap();
        assert!(stop(0.05, 7));
        assert!(!stop(0.05, 8));
        for (key, value) in [
            ("unary_operators", strs(["bad(x) = unknown(x)"])),
            ("elementwise_loss", Value::Str("loss(x) = x".into())),
            (
                "early_stop_condition",
                Value::Str("(loss, complexity) -> loss + complexity".into()),
            ),
            (
                "nested_constraints",
                dict(vec![("cos", dict(vec![("exp", Value::Int(256))]))]),
            ),
            ("complexity_of_operators", dict(vec![("missing", Value::Float(1.0))])),
        ] {
            assert!(search_options([(key, value)]).is_err(), "{key}");
        }
        assert!(
            guesses::<f32>(
                Value::List(vec![dict(vec![("p", Value::List(vec![Value::Float(f64::MAX)]))])]),
                1
            )
            .is_err()
        );
    }

    fn strs<const N: usize>(items: [&str; N]) -> Value {
        Value::List(items.iter().map(|s| Value::Str(s.to_string())).collect())
    }

    fn search_options<const N: usize>(pairs: [(&str, Value); N]) -> Result<SearchOptions<f64>, String> {
        options::<f64>(Keywords::new(
            pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        ))
    }

    fn error<const N: usize>(pairs: [(&str, Value); N]) -> String {
        search_options(pairs).err().expect("the options are invalid")
    }

    fn dict(pairs: Vec<(&str, Value)>) -> Value {
        Value::Dict(pairs.into_iter().map(|(k, v)| (Value::Str(k.into()), v)).collect())
    }

    #[test]
    fn rejects_unsupported_keys_even_when_none() {
        for &(key, _) in JULIA_ONLY {
            assert!(error([(key, Value::None)]).contains(key));
        }
        assert!(error([("unknown", Value::None)]).contains("unknown option"));
        for key in ["form_connection", "break_connection"] {
            for value in [Value::None, Value::Float(0.0), Value::Float(1.0)] {
                assert!(error([("mutation_weights", dict(vec![(key, value)]))]).contains("graph"));
            }
        }
    }

    #[test]
    fn rejects_invalid_numeric_settings_before_search() {
        for (key, value) in [
            ("populations", Value::Int(0)),
            ("tournament_selection_n", Value::Int(0)),
            ("maxdepth", Value::Int(0)),
            ("complexity_of_constants", Value::Float(-1.0)),
            ("complexity_of_constants", Value::Float(f64::MAX)),
            ("parsimony", Value::Float(f64::NAN)),
            ("crossover_probability", Value::Float(1.1)),
            ("tournament_selection_p", Value::Float(0.0)),
            ("batch_size", Value::Int(0)),
            ("alpha", Value::Float(0.0)),
        ] {
            assert!(search_options([(key, value)]).is_err(), "{key}");
        }
        assert_eq!(error([("maxsize", Value::Int(3))]), "maxsize must be greater than 3");
        assert_eq!(error([("batch_size", Value::Int(0))]), "batch_size must be positive");
        assert_eq!(
            error([("deterministic", Value::Bool(true))]),
            "Determinism is only guaranteed for serial mode."
        );
        assert!(
            search_options([
                ("deterministic", Value::Bool(true)),
                ("parallelism", Value::Str("serial".into()))
            ])
            .is_ok()
        );
        assert!(
            search_options([
                ("deterministic", Value::Bool(true)),
                ("parallelism", Value::Str("serial".into())),
                ("seed", Value::Int(7))
            ])
            .is_ok()
        );
    }

    #[test]
    fn maps_arity_indexed_constraints_and_engine_settings() {
        let o = search_options([
            ("binary_operators", strs(["+"])),
            ("unary_operators", strs(["cos"])),
            (
                "constraints",
                Value::Dict(vec![
                    (Value::Int(1), Value::List(vec![Value::Int(2)])),
                    (
                        Value::Int(2),
                        Value::List(vec![Value::List(vec![Value::Int(-1), Value::Int(3)])]),
                    ),
                ]),
            ),
            ("use_baseline", Value::Bool(false)),
            ("output_style", Value::Str("plain".into())),
        ])
        .unwrap();
        let cos = o.operators.set.lookup("cos").unwrap();
        let add = o.operators.set.lookup("+").unwrap();
        assert_eq!(o.op_constraints.limits[&cos], [Some(2), None, None]);
        assert_eq!(o.op_constraints.limits[&add], [None, Some(3), None]);
        assert!(!o.use_baseline);
        assert_eq!(o.output_style, symbolic_regression::OutputStyle::Plain);
        assert!(search_options([("constraints", Value::Dict(vec![(Value::Int(2), Value::List(vec![]))]))]).is_err());
    }

    #[test]
    fn validates_weights_plugins_and_preserves_default_overrides() {
        for weight in [Value::Float(-1.0), Value::Float(f64::INFINITY)] {
            assert!(search_options([("mutation_weights", dict(vec![("add_node", weight)]))]).is_err());
        }
        for plugin in [
            dict(vec![
                ("type", Value::Str("AdaptiveMutationWeightsPlugin".into())),
                ("floor", Value::Float(0.0)),
            ]),
            dict(vec![
                ("type", Value::Str("MutationBurstPlugin".into())),
                ("retry_attempts", Value::Int(0)),
            ]),
        ] {
            assert!(search_options([("plugins", Value::List(vec![plugin]))]).is_err());
        }
        let o = search_options([
            (
                "mutations",
                Value::List(vec![Value::List(vec![
                    Value::Str("AddNodeMutation".into()),
                    Value::Float(7.0),
                ])]),
            ),
            (
                "default_mutations",
                Value::List(vec![
                    Value::List(vec![Value::Str("AddNodeMutation".into()), Value::Float(2.0)]),
                    Value::List(vec![Value::Str("DoNothingMutation".into()), Value::Float(1.0)]),
                ]),
            ),
        ])
        .unwrap();
        let resolved: Vec<_> = o.resolved_mutations().collect();
        assert!(matches!(
            resolved[..],
            [(BuiltinMutation::AddNode, 7.0), (BuiltinMutation::DoNothing, 1.0)]
        ));
    }

    #[test]
    fn template_dict_can_supply_explicit_subexpression_arities() {
        let o = search_options([(
            "expression_spec",
            dict(vec![
                ("combine", Value::Str("f(x) + p[1]".into())),
                ("expressions", strs(["f", "unused"])),
                ("variable_names", strs(["x"])),
                ("parameters", dict(vec![("p", Value::Int(2))])),
                ("num_features", Value::List(vec![Value::Int(1), Value::Int(2)])),
            ]),
        )])
        .unwrap();
        let template = o.template.unwrap();
        assert_eq!(template.num_features(), &[1, 2]);
        assert_eq!(template.n_parameters(), 2);
    }

    #[test]
    fn keywords_set_options_by_their_symbolic_regression_jl_names() {
        let o = search_options([
            ("binary_operators", strs(["+", "-", "*", "^"])),
            ("unary_operators", strs(["-", "inv(x) = 1/x"])),
            ("maxsize", Value::Int(12)),
            ("seed", Value::Int(3)),
            ("maxdepth", Value::None),
            ("batching", Value::Str("auto".into())),
            (
                "constraints",
                Value::Dict(vec![(
                    Value::Str("^".into()),
                    Value::List(vec![Value::Int(-1), Value::Int(1)]),
                )]),
            ),
            (
                "complexity_of_operators",
                Value::Dict(vec![(Value::Str("-".into()), Value::Float(2.0))]),
            ),
            (
                "optimizer_options",
                Value::Dict(vec![(Value::Str("iterations".into()), Value::Int(4))]),
            ),
            ("optimizer_iterations", Value::Int(16)),
            (
                "plugins",
                Value::List(vec![Value::Dict(vec![
                    (Value::Str("type".into()), Value::Str("MutationBurstPlugin".into())),
                    (Value::Str("retry_attempts".into()), Value::Int(2)),
                ])]),
            ),
            ("annealing", Value::Bool(false)),
            (
                "mutations",
                Value::List(vec![Value::List(vec![
                    Value::Str("BacksolveMutation".into()),
                    Value::Float(0.5),
                ])]),
            ),
        ])
        .unwrap();
        assert_eq!(
            (o.maxsize, o.seed, o.maxdepth, o.batching),
            (12, Some(3), None, Batching::Auto)
        );
        assert_eq!((o.operators.nops(1), o.operators.nops(2)), (2, 4));
        let pow = o.operators.set.lookup_with_arity("^", 2).unwrap();
        assert_eq!(o.op_constraints.limits[&pow], [None, Some(1), None]);
        assert_eq!(o.operator_complexity_overrides.len(), 2);
        assert_eq!(o.optimizer_iterations, 4);
        assert_eq!(o.mutation_burst.map(|b| b.retry_attempts), Some(2));
        assert!(!o.annealing && o.use_frequency && o.adaptive_mutation_weights.is_some());
        assert!(matches!(o.mutations[..], [(BuiltinMutation::Backsolve(_), w)] if w == 0.5));

        assert_eq!(
            error([("maxsize", Value::Str("10".into()))]),
            "`maxsize` must be an integer, got \"10\""
        );
        assert_eq!(
            error([("not_an_option", Value::Int(1))]),
            "unknown option `not_an_option`"
        );
        assert_eq!(
            error([("turbo", Value::Bool(false))]),
            "`turbo` is a SymbolicRegression.jl option that the Rust engine does not support: it selects LoopVectorization.jl"
        );
        assert_eq!(
            error([("unary_operators", strs(["+"]))]),
            "operator `+` takes 2 arguments, but it is in the unary operator list"
        );
        assert_eq!(
            error([
                ("mutation_weights", Value::Dict(vec![])),
                ("default_mutations", Value::List(vec![]))
            ]),
            "`mutation_weights` and `default_mutations` cannot be used together."
        );
        assert_eq!(defaults_profile("0.24.5"), Ok(DefaultsProfile::Legacy));
        assert_eq!(defaults_profile("1.2"), Ok(DefaultsProfile::V1));
        assert_eq!(defaults_profile("2.0.0-DEV"), Ok(DefaultsProfile::V2));
        assert_eq!(defaults_profile("2.0.0+build.1"), Ok(DefaultsProfile::V2));
        for version in ["", "1.0.0-", "1.0.0+", "+1.0", "2.0.0.1"] {
            assert!(defaults_profile(version).is_err());
        }
    }

    #[test]
    fn partial_mutation_weights_use_global_defaults() {
        let o = search_options([
            ("defaults", Value::Str("0.24.5".into())),
            ("mutation_weights", dict(vec![("add_node", Value::Float(1.0))])),
        ])
        .unwrap();
        let defaults = symbolic_regression::MutationWeights::default();
        assert_eq!(o.mutation_weights.add_node, 1.0);
        assert_eq!(o.mutation_weights.insert_node, defaults.insert_node);
        assert_eq!(o.mutation_weights.rotate_tree, defaults.rotate_tree);
        assert_eq!(o.mutation_weights.mutate_operator, defaults.mutate_operator);
    }
}

/// test/unit/basic/test_options.jl | "Test operators parameter conflicts"
#[cfg(test)]
#[test]
fn operators_parameter_conflicts() {
    fn strs<const N: usize>(items: [&str; N]) -> Value {
        Value::List(items.iter().map(|s| Value::Str(s.to_string())).collect())
    }
    fn search_options<const N: usize>(pairs: [(&str, Value); N]) -> Result<SearchOptions<f64>, String> {
        options::<f64>(Keywords::new(
            pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        ))
    }
    fn error<const N: usize>(pairs: [(&str, Value); N]) -> String {
        search_options(pairs).err().expect("the options are invalid")
    }
    let operators = || {
        Value::Dict(vec![
            (Value::Int(1), strs(["sin", "cos"])),
            (Value::Int(2), strs(["+", "*", "-"])),
        ])
    };
    let conflict = "`operators` cannot be combined with `binary_operators` or `unary_operators`";
    assert_eq!(
        error([("operators", operators()), ("binary_operators", strs(["+", "*"]))]),
        conflict
    );
    assert_eq!(
        error([("operators", operators()), ("unary_operators", strs(["sin"]))]),
        conflict
    );
    assert_eq!(
        error([
            ("operators", operators()),
            ("operator_enum_constructor", Value::Str("OperatorEnum".into()))
        ]),
        "`operator_enum_constructor` is a SymbolicRegression.jl option that the Rust engine does not support: it takes a Julia type"
    );
    let alone = search_options([("operators", operators())]).unwrap();
    assert_eq!((alone.operators.nops(1), alone.operators.nops(2)), (2, 3));
}
