use std::fmt::Display;
use std::ops::AddAssign;

use dynamic_expressions::dsl::{TemplateStructure, parse_expression};
use dynamic_expressions::strings::{StringTreeOptions, default_string_variable};
use dynamic_expressions::{GradContext, OperatorSet, PNode};
use fastrand::Rng;
use num_traits::{Float, FromPrimitive, ToPrimitive};

use crate::constant_optimization::{OptimizeConstantsCtx, optimize_constants};
use crate::dataset::{Dataset, TaggedDataset};
use crate::expression::Expression;
use crate::loss_functions::baseline_loss_from_zero_expression;
use crate::options::Options;
use crate::pop_member::{Evaluator, PopMember};

/// One of SymbolicRegression.jl's `guesses`, an equation that seeds the search.
///
/// ```
/// use symbolic_regression::{Guess, TemplateGuessValue};
///
/// let plain: Guess<f64, (), 2> = "2.0*x0*x0 + 3.0*x1 + 0.5".into();
/// let template: Guess<f64, (), 2> = Guess::Template(vec![
///     ("f".to_string(), "#1 * #2".into()),
///     ("p".to_string(), TemplateGuessValue::Parameter(vec![5.0, 10.0])),
/// ]);
/// ```
pub enum Guess<T, Ops, const D: usize> {
    /// An equation over the dataset's variable names, or `x0`, `x1`, ... when it has none. See
    /// [`parse_expression`] for the syntax.
    Equation(String),
    /// A template search's guess, written as PySR's dict guesses are: each subexpression by name, as
    /// an equation over its arguments `#1`, `#2`, ..., and parameter vectors by name. Omitted
    /// parameters start at zero.
    Template(Vec<(String, TemplateGuessValue<T>)>),
    /// An expression, used as it is.
    Expression(Expression<T, Ops, D>),
}

/// The value of one name in a [`Guess::Template`].
#[derive(Clone, Debug, PartialEq)]
pub enum TemplateGuessValue<T> {
    Equation(String),
    Parameter(Vec<T>),
}

impl<T: Clone, Ops, const D: usize> Clone for Guess<T, Ops, D> {
    fn clone(&self) -> Self {
        match self {
            Self::Equation(s) => Self::Equation(s.clone()),
            Self::Template(entries) => Self::Template(entries.clone()),
            Self::Expression(e) => Self::Expression(e.clone()),
        }
    }
}

impl<T, Ops, const D: usize> From<&str> for Guess<T, Ops, D> {
    fn from(src: &str) -> Self {
        Self::Equation(src.to_string())
    }
}

impl<T, Ops, const D: usize> From<String> for Guess<T, Ops, D> {
    fn from(src: String) -> Self {
        Self::Equation(src)
    }
}

impl<T, Ops, const D: usize> From<Expression<T, Ops, D>> for Guess<T, Ops, D> {
    fn from(expr: Expression<T, Ops, D>) -> Self {
        Self::Expression(expr)
    }
}

impl<T> From<&str> for TemplateGuessValue<T> {
    fn from(src: &str) -> Self {
        Self::Equation(src.to_string())
    }
}

impl<T> From<Vec<T>> for TemplateGuessValue<T> {
    fn from(values: Vec<T>) -> Self {
        Self::Parameter(values)
    }
}

/// The guesses of `options` as members of each output's search, with constants optimized, and the
/// warnings SymbolicRegression.jl's `parse_guesses` logs. Fails on the first guess that does not
/// parse, naming it, or when `options.guesses` does not hold one list per dataset.
pub fn parse_guesses<T, Ops, const D: usize>(
    datasets: &[Dataset<T>],
    options: &Options<T, Ops, D>,
) -> Result<ParsedGuesses<T, Ops, D>, String>
where
    T: Float + FromPrimitive + ToPrimitive + AddAssign + Display,
    Ops: OperatorSet<T = T>,
{
    let tagged: Vec<TaggedDataset<'_, T>> = datasets
        .iter()
        .map(|d| {
            let baseline = if options.use_baseline {
                baseline_loss_from_zero_expression::<T, Ops, D>(d, &options.operators.set, options.loss.as_ref())
            } else {
                None
            };
            TaggedDataset::new(d, baseline)
        })
        .collect();
    seed_members(&tagged, options, options.seed.unwrap_or_else(|| fastrand::u64(..)))
}

/// One list of seed members per output, and SymbolicRegression.jl's warnings about them.
pub struct ParsedGuesses<T: Float, Ops, const D: usize> {
    pub members: Vec<Vec<PopMember<T, Ops, D>>>,
    pub warnings: Vec<String>,
}

pub(crate) fn seed_members<T, Ops, const D: usize>(
    datasets: &[TaggedDataset<'_, T>],
    options: &Options<T, Ops, D>,
    seed: u64,
) -> Result<ParsedGuesses<T, Ops, D>, String>
where
    T: Float + FromPrimitive + ToPrimitive + AddAssign + Display,
    Ops: OperatorSet<T = T>,
{
    let mut parsed = ParsedGuesses {
        members: datasets.iter().map(|_| Vec::new()).collect(),
        warnings: Vec::new(),
    };
    if options.guesses.is_empty() {
        return Ok(parsed);
    }
    if options.guesses.len() != datasets.len() {
        return Err(format!(
            "`guesses` must hold one list of guesses per output: it has {} lists for {} outputs",
            options.guesses.len(),
            datasets.len()
        ));
    }
    let mut rng = Rng::with_seed(seed ^ 0x6775_6573_7365_7321);
    for ((guesses, dataset), members) in options.guesses.iter().zip(datasets).zip(&mut parsed.members) {
        let mut evaluator = Evaluator::new(dataset.n_rows);
        let mut grad_ctx = GradContext::new(dataset.n_rows);
        for guess in guesses {
            let invalid = |message: String| format!("invalid guess {}: {message}", describe(guess, options));
            let expr = guess_expression(guess, dataset, options).map_err(invalid)?;
            let mut member = PopMember::from_expr(expr, dataset.n_features, options);
            evaluator.ensure_n_rows(dataset.n_rows);
            evaluator
                .eval_checked(&member.expr, &member.plans, options, dataset.x_columns())
                .map_err(|e| invalid(e.to_string()))?;
            member.evaluate(dataset, options, &mut evaluator);
            if options.should_optimize_constants {
                let ctx = OptimizeConstantsCtx {
                    dataset: *dataset,
                    options,
                    evaluator: &mut evaluator,
                    grad_ctx: &mut grad_ctx,
                };
                optimize_constants(&mut rng, &mut member, ctx);
            }
            if member.complexity > options.maxsize {
                parsed.warnings.push(format!(
                    "Guess expression `{}` has complexity {} > maxsize ({}).",
                    member.expr.string(options, StringTreeOptions::default()),
                    member.complexity,
                    options.maxsize
                ));
            }
            members.push(member);
        }
    }
    Ok(parsed)
}

fn describe<T: Float + Display, Ops: OperatorSet, const D: usize>(
    guess: &Guess<T, Ops, D>,
    options: &Options<T, Ops, D>,
) -> String {
    match guess {
        Guess::Equation(src) => format!("`{src}`"),
        Guess::Template(entries) => {
            let entries: Vec<String> = entries
                .iter()
                .map(|(name, value)| match value {
                    TemplateGuessValue::Equation(src) => format!("{name} = {src:?}"),
                    TemplateGuessValue::Parameter(values) => {
                        let values: Vec<String> = values.iter().map(T::to_string).collect();
                        format!("{name} = [{}]", values.join(", "))
                    }
                })
                .collect();
            format!("({})", entries.join(", "))
        }
        Guess::Expression(expr) => format!("`{}`", expr.string(options, StringTreeOptions::default())),
    }
}

fn guess_expression<T: Float, Ops: OperatorSet, const D: usize>(
    guess: &Guess<T, Ops, D>,
    dataset: &Dataset<T>,
    options: &Options<T, Ops, D>,
) -> Result<Expression<T, Ops, D>, String> {
    let variable_names: Vec<String> = (0..dataset.n_features)
        .map(|i| default_string_variable(i as u16, Some(&dataset.variable_names)))
        .collect();
    match (guess, &options.template) {
        (Guess::Equation(src), None) => parse_expression(src, &options.operators, &variable_names)
            .map(Expression::from)
            .map_err(|e| e.to_string()),
        (Guess::Equation(_), Some(template)) => Err(format!(
            "a template search takes guesses that give each of its expressions ({})",
            template.expressions().join(", ")
        )),
        (Guess::Template(_), None) => Err("only a template search takes a template guess".to_string()),
        (Guess::Template(entries), Some(template)) => {
            template_guess(entries, template, &variable_names, dataset.n_features, options)
        }
        (Guess::Expression(expr), _) => {
            check_shape(expr, dataset.n_features, options)?;
            Ok(expr.clone())
        }
    }
}

/// SymbolicRegression.jl's `parse_expression` for a template guess: its names must be the
/// template's, every expression must be given as a string over `#1`, `#2`, ..., which must not
/// mention the dataset's variables, and every given parameter must be a vector of its length.
fn template_guess<T: Float, Ops: OperatorSet, const D: usize>(
    entries: &[(String, TemplateGuessValue<T>)],
    template: &TemplateStructure,
    variable_names: &[String],
    n_features: usize,
    options: &Options<T, Ops, D>,
) -> Result<Expression<T, Ops, D>, String> {
    let expressions = template.expressions();
    let parameters = template.parameters();
    let allowed: Vec<&str> = expressions
        .iter()
        .map(String::as_str)
        .chain(parameters.iter().map(|(name, _)| name.as_str()))
        .collect();
    for (i, (key, _)) in entries.iter().enumerate() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!(
                "`{key}` is not one of the template's expressions or parameters ({})",
                allowed.join(", ")
            ));
        }
        if entries[..i].iter().any(|(k, _)| k == key) {
            return Err(format!("`{key}` is given twice"));
        }
    }
    let find = |name: &str| entries.iter().find(|(key, _)| key == name).map(|(_, value)| value);
    let mut trees = Vec::with_capacity(expressions.len());
    for (k, name) in expressions.iter().enumerate() {
        let src = match find(name) {
            Some(TemplateGuessValue::Equation(src)) => src,
            Some(TemplateGuessValue::Parameter(_)) => {
                return Err(format!("the expression `{name}` must be an equation string"));
            }
            None => {
                return Err(format!(
                    "the guess must give every expression of the template ({})",
                    expressions.join(", ")
                ));
            }
        };
        if let Some(var) = variable_names.iter().find(|var| contains_word(src, var)) {
            return Err(format!(
                "found the variable name `{var}` in the expression `{name}`; write its arguments as `#1`, `#2`, ..."
            ));
        }
        let n_args = options.tree_features(k, n_features);
        let src = placeholders(src, n_args).map_err(|e| format!("in the expression `{name}`: {e}"))?;
        let args: Vec<String> = (1..=n_args).map(|i| format!("__arg_{i}")).collect();
        let tree = parse_expression(&src, &options.operators, &args)
            .map_err(|e| format!("in the expression `{name}`: {e}"))?;
        trees.push(tree);
    }
    let mut params = Vec::with_capacity(template.n_parameters());
    for (name, len) in parameters {
        match find(name) {
            None => params.extend(std::iter::repeat_n(T::zero(), *len)),
            Some(TemplateGuessValue::Parameter(values)) if values.len() == *len => params.extend_from_slice(values),
            Some(TemplateGuessValue::Parameter(values)) => {
                return Err(format!(
                    "the parameter `{name}` must have length {len}, got {}",
                    values.len()
                ));
            }
            Some(TemplateGuessValue::Equation(_)) => {
                return Err(format!("the parameter `{name}` must be a vector"));
            }
        }
    }
    Ok(Expression {
        trees: trees.into(),
        params,
    })
}

/// Whether `word` appears in `src` between word boundaries, as the regex `\bword\b` finds it.
fn contains_word(src: &str, word: &str) -> bool {
    let is_word = |c: char| c.is_alphanumeric() || c == '_';
    src.match_indices(word).any(|(i, _)| {
        !src[..i].chars().next_back().is_some_and(is_word) && !src[i + word.len()..].chars().next().is_some_and(is_word)
    })
}

/// `src` with each argument placeholder `#N` renamed to the variable `__arg_N`, checking that
/// `1 <= N <= n_args`.
fn placeholders(src: &str, n_args: usize) -> Result<String, String> {
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    while let Some(i) = rest.find('#') {
        out.push_str(&rest[..i]);
        let digits = rest[i + 1..]
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(rest.len() - i - 1);
        let number = &rest[i + 1..i + 1 + digits];
        match number.parse::<usize>() {
            Ok(n) if (1..=n_args).contains(&n) => out.push_str(&format!("__arg_{n}")),
            Ok(n) => return Err(format!("`#{n}` is not one of its {n_args} arguments")),
            Err(_) => return Err("`#` must be followed by an argument number, as in `#1`".to_string()),
        }
        rest = &rest[i + 1 + digits..];
    }
    out.push_str(rest);
    Ok(out)
}

/// An expression guess must have the search's number of trees and parameters, and read only the
/// features each tree takes.
fn check_shape<T: Float, Ops: OperatorSet, const D: usize>(
    expr: &Expression<T, Ops, D>,
    n_features: usize,
    options: &Options<T, Ops, D>,
) -> Result<(), String> {
    let (n_trees, n_params) = options
        .template
        .as_ref()
        .map_or((1, 0), |t| (t.expressions().len(), t.n_parameters()));
    if expr.trees.len() != n_trees || expr.params.len() != n_params {
        return Err(format!(
            "the expression has {} trees and {} parameters, but the search's expressions have {n_trees} and {n_params}",
            expr.trees.len(),
            expr.params.len()
        ));
    }
    for (k, tree) in expr.trees.iter().enumerate() {
        let n = options.tree_features(k, n_features);
        if let Some(feature) = tree.nodes.iter().find_map(|node| match *node {
            PNode::Var { feature } if usize::from(feature) >= n => Some(feature),
            _ => None,
        }) {
            return Err(format!("the expression reads feature {feature} of {n}"));
        }
    }
    Ok(())
}
