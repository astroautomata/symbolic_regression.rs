//! Ports of SymbolicRegression.jl's guess testitems: test/unit/misc/test_guesses.jl, and the
//! template guesses of test/unit/expressions/test_expression_builder.jl,
//! test/unit/templates/test_template_parameter_value_types.jl, and
//! test/integration/ext/loopvectorization/test_template_expression.jl (see tools/parity_tests.toml).
//! SymbolicRegression.jl names features `x1`, `x2`, ..., so the datasets here carry those names.

use dynamic_expressions::dsl::{CustomOps, TemplateStructure, parse_operator_tokens};
use dynamic_expressions::operator_enum::presets::{BuiltinOpsF32, BuiltinOpsF64};
use dynamic_expressions::strings::StringTreeOptions;
use dynamic_expressions::{Metadata, OperatorSet, Operators, PNode, PostfixExpr};
use ndarray::{Array1, Array2, array};

use crate::dataset::Dataset;
use crate::expression::Expression;
use crate::guesses::{Guess, TemplateGuessValue, parse_guesses};
use crate::options::Options;
use crate::search_utils::equation_search;

type Ops = BuiltinOpsF64;
type COps = BuiltinOpsF64<CustomOps<f64>>;
const D: usize = 2;

fn names(n: usize) -> Vec<String> {
    (1..=n).map(|i| format!("x{i}")).collect()
}

fn random_dataset(n_features: usize, n_rows: usize, seed: u64, mut f: impl FnMut(&[f64]) -> f64) -> Dataset<f64> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let x = Array2::from_shape_fn((n_features, n_rows), |_| rng.f64() * 4.0 - 2.0);
    let y = Array1::from_iter(x.columns().into_iter().map(|c| f(&c.to_vec())));
    Dataset::with_weights_and_names(x, y, None, names(n_features))
}

/// `X = [1.0 2.0; 3.0 4.0]` and `y = [5.0, 6.0]`, the dataset most of the testitems parse against.
fn small_dataset() -> Dataset<f64> {
    Dataset::with_weights_and_names(array![[1.0, 2.0], [3.0, 4.0]], array![5.0, 6.0], None, names(2))
}

fn builtin(operators: &[&str]) -> Operators<Ops, D> {
    Ops::new().from_names(operators).unwrap()
}

fn custom(tokens: &[&str]) -> Operators<COps, D> {
    let (custom, names) = parse_operator_tokens::<f64, _>(tokens.iter().copied()).unwrap();
    BuiltinOpsF64::new().with_custom(custom).from_names(&names).unwrap()
}

fn seeded<O: OperatorSet<T = f64> + Default>(
    operators: Operators<O, D>,
    guesses: Vec<Guess<f64, O, D>>,
) -> Options<f64, O, D> {
    Options {
        operators,
        guesses: vec![guesses],
        niterations: 0,
        populations: 2,
        progress: false,
        seed: 0,
        deterministic: true,
        ..Default::default()
    }
}

fn template<O: OperatorSet>(
    combine: &str,
    expressions: &[&str],
    variable_names: &[&str],
    parameters: &[(&str, usize)],
    options: &Options<f64, O, D>,
) -> TemplateStructure {
    TemplateStructure::new(
        combine,
        expressions,
        None,
        variable_names,
        parameters,
        &options.operators.set,
    )
    .unwrap()
}

fn tguess<O>(entries: &[(&str, TemplateGuessValue<f64>)]) -> Guess<f64, O, D> {
    Guess::Template(entries.iter().map(|(k, v)| (k.to_string(), v.clone())).collect())
}

fn eq(src: &str) -> TemplateGuessValue<f64> {
    src.into()
}

fn param(values: &[f64]) -> TemplateGuessValue<f64> {
    values.to_vec().into()
}

/// The loss of the best member of the Pareto front of output `j`.
fn best_loss<O>(result: &crate::SearchResult<f64, O, D>, j: usize) -> f64 {
    result.halls_of_fame[j]
        .pareto_front()
        .iter()
        .map(|m| m.loss)
        .fold(f64::INFINITY, f64::min)
}

/// test/unit/misc/test_guesses.jl | "Single output, single guess"
#[test]
fn single_output_single_guess() {
    let dataset = random_dataset(2, 50, 0, |x| 2.0 * x[0] * x[0] + 3.0 * x[1] + 0.5);
    let good = seeded(builtin(&["+", "*"]), vec!["2.0*x1*x1 + 3.0*x2 + 0.5".into()]);
    let result = equation_search(std::slice::from_ref(&dataset), &good).unwrap();
    assert!(best_loss(&result, 0) < 1e-10);

    let bad = seeded(builtin(&["+", "*"]), vec!["1.0*x1 + 1.0*x2 + 0.0".into()]);
    let result = equation_search(std::slice::from_ref(&dataset), &bad).unwrap();
    assert!(best_loss(&result, 0) >= 1e-10);
}

/// test/unit/misc/test_guesses.jl | "Guesses violating constraints do not enter the Hall of Fame"
#[test]
fn guesses_violating_constraints_do_not_enter_the_hall_of_fame() {
    let dataset = random_dataset(2, 8, 0, |_| 0.0);
    let mut options = seeded(builtin(&["+"]), vec!["x1 + x2".into()]);
    let plus = options.operators.set.lookup_with_arity("+", 2).unwrap();
    options.op_constraints.set_op_arg_constraint(plus, 0, 0);
    options.op_constraints.set_op_arg_constraint(plus, 1, 0);

    let result = equation_search(std::slice::from_ref(&dataset), &options).unwrap();
    assert_eq!(result.halls_of_fame[0].members().count(), 0);
}

/// test/unit/misc/test_guesses.jl | "parse_guesses with NamedTuple"
#[test]
fn parse_guesses_with_named_tuple() {
    let dataset = small_dataset();
    let mut options = seeded(
        builtin(&["+", "sub", "*"]),
        vec![tguess(&[("f", eq("2.0 * #1")), ("g", eq("1.5 * #2"))])],
    );
    options.template = Some(template(
        "f(x1, x2) + g(x1, x2)",
        &["f", "g"],
        &["x1", "x2"],
        &[],
        &options,
    ));
    let parsed = parse_guesses(std::slice::from_ref(&dataset), &options).unwrap();
    assert_eq!(parsed.members.len(), 1);
    assert_eq!(parsed.members[0].len(), 1);
    assert_eq!(parsed.members[0][0].expr.trees.len(), 2);

    options.guesses = vec![vec![
        tguess(&[("f", eq("#1")), ("g", eq("#2"))]),
        tguess(&[("f", eq("2.0 * #1")), ("g", eq("1.5 * #2"))]),
    ]];
    let parsed = parse_guesses(std::slice::from_ref(&dataset), &options).unwrap();
    assert_eq!(parsed.members.len(), 1);
    assert_eq!(parsed.members[0].len(), 2);
    assert!(parsed.members[0].iter().all(|m| m.expr.trees.len() == 2));
}

/// test/unit/misc/test_guesses.jl | "parse_guesses with NamedTuple and parameters"
#[test]
fn parse_guesses_with_named_tuple_and_parameters() {
    let dataset = small_dataset();
    let mut options = seeded(builtin(&["+", "sub", "*"]), vec![tguess(&[("f", eq("#1 * #2"))])]);
    options.template = Some(template(
        "f(x1, x2) + p[1] * x1 + p[2]",
        &["f"],
        &["x1", "x2"],
        &[("p", 2)],
        &options,
    ));
    let parsed = parse_guesses(std::slice::from_ref(&dataset), &options).unwrap();
    assert_eq!(parsed.members[0][0].expr.params.len(), 2);
}

/// test/unit/misc/test_guesses.jl | "NamedTuple guesses with different variable names"
#[test]
fn named_tuple_guesses_with_different_variable_names() {
    let mut dataset = random_dataset(3, 20, 0, |x| x[0] + 2.0 * x[1] - x[2]);
    dataset.variable_names = vec!["input1".into(), "input2".into(), "input3".into()];
    let mut options = seeded(
        builtin(&["+", "sub", "*"]),
        vec![tguess(&[("term1", eq("#1 + 2.0 * #2")), ("term2", eq("-1.0 * #3"))])],
    );
    options.template = Some(template(
        "term1(input1, input2, input3) + term2(input1, input2, input3)",
        &["term1", "term2"],
        &["input1", "input2", "input3"],
        &[],
        &options,
    ));
    let result = equation_search(std::slice::from_ref(&dataset), &options).unwrap();
    assert!(best_loss(&result, 0) < 1e-8);
}

/// test/unit/misc/test_guesses.jl | "Float32 dataset with Float64 guess literals"
#[test]
fn float32_dataset_with_float64_guess_literals() {
    let dataset =
        Dataset::<f32>::with_weights_and_names(array![[1.0, 2.0], [3.0, 4.0]], array![5.0, 6.0], None, names(2));
    for guess in [
        "4.561253 - ((x1 - x2) * 0.18459733)",
        "4.561253f0 - ((x1 - x2) * 0.18459733f0)",
    ] {
        let options = Options::<f32, BuiltinOpsF32, D> {
            operators: BuiltinOpsF32::new().from_names(["+", "sub", "*", "/"]).unwrap(),
            guesses: vec![vec![guess.into()]],
            deterministic: true,
            progress: false,
            ..Default::default()
        };
        let parsed = parse_guesses(std::slice::from_ref(&dataset), &options).unwrap();
        assert_eq!(parsed.members.len(), 1);
        assert_eq!(parsed.members[0].len(), 1, "{guess}");
    }
}

const PYTHAG_OPS: [&str; 9] = [
    "+",
    "sub",
    "*",
    "/",
    "pythag_pos(x, y) = sqrt(x^2 + y^2)",
    "pythag_neg(x, y) = x^2 - y^2 < 0 ? NaN : sqrt(x^2 - y^2)",
    "sin",
    "cos",
    "custom_sin(x) = sin(x) + 0.1",
];

/// test/unit/misc/test_guesses.jl | "Custom operators in string guesses"
#[test]
fn custom_operators_in_string_guesses() {
    let dataset = Dataset::with_weights_and_names(
        array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        array![7.0, 8.0, 9.0],
        None,
        names(2),
    );
    let lists: [&[&str]; 3] = [
        &["pythag_pos(x1, x2) + custom_sin(x1)"],
        &["pythag_pos(x1, 4.51352 - ((x2 - x1) * 0.07425507))"],
        &["pythag_pos(x1, x2)", "pythag_neg(x1, x2) + 0.5", "custom_sin(x1) * x2"],
    ];
    for guesses in lists {
        let options = seeded(custom(&PYTHAG_OPS), guesses.iter().map(|&g| g.into()).collect());
        let parsed = parse_guesses(std::slice::from_ref(&dataset), &options).unwrap();
        assert_eq!(parsed.members.len(), 1);
        assert_eq!(parsed.members[0].len(), guesses.len());
    }
}

/// test/unit/misc/test_guesses.jl | "Custom operators in equation_search guesses"
#[test]
fn custom_operators_in_equation_search_guesses() {
    let mut noise = fastrand::Rng::with_seed(1);
    let dataset = random_dataset(2, 30, 0, |x| {
        (x[0] * x[0] + x[1] * x[1]).sqrt() + 0.01 * (noise.f64() * 2.0 - 1.0)
    });
    for guess in ["pythag_pos(x1, x2)", "pythag_pos(x1, x2) + 0.0"] {
        let options = seeded(custom(&PYTHAG_OPS[..8]), vec![guess.into()]);
        let result = equation_search(std::slice::from_ref(&dataset), &options).unwrap();
        assert!(best_loss(&result, 0) < 1e-2, "{guess}");
    }
}

/// test/unit/misc/test_guesses.jl | "Custom operators error handling in guesses"
#[test]
fn custom_operators_error_handling_in_guesses() {
    let dataset = small_dataset();
    let cases = [
        ("pythag_pos(x1)", "the operator `pythag_pos` takes 2 arguments, not 1"),
        (
            "nonexistent_op(x1, x2)",
            "`nonexistent_op` is not one of the search's operators",
        ),
    ];
    for (guess, reason) in cases {
        let options = seeded(custom(&PYTHAG_OPS[..5]), vec![guess.into()]);
        let err = parse_guesses(std::slice::from_ref(&dataset), &options).err().unwrap();
        assert!(err.starts_with(&format!("invalid guess `{guess}`: ")), "{err}");
        assert!(err.contains(reason), "{err}");
    }
}

/// test/unit/misc/test_guesses.jl | "Custom operators with NamedTuple guesses"
#[test]
fn custom_operators_with_named_tuple_guesses() {
    let dataset = small_dataset();
    let mut options = seeded(
        custom(&[
            "+",
            "sub",
            "*",
            "pythag_pos(x, y) = sqrt(x^2 + y^2)",
            "custom_mul(x, y) = x * y * 1.1",
        ]),
        vec![tguess(&[
            ("f", eq("pythag_pos(#1, #2)")),
            ("g", eq("custom_mul(#1, #2)")),
        ])],
    );
    options.template = Some(template(
        "f(x1, x2) + g(x1, x2)",
        &["f", "g"],
        &["x1", "x2"],
        &[],
        &options,
    ));
    let parsed = parse_guesses(std::slice::from_ref(&dataset), &options).unwrap();
    assert_eq!(parsed.members[0].len(), 1);
    assert_eq!(
        parsed.members[0][0].expr.string(&options, Default::default()),
        "f = pythag_pos(#1, #2); g = custom_mul(#1, #2)"
    );
}

/// test/unit/misc/test_guesses.jl | "Smoke test migration with multiple outputs and templates"
#[test]
fn smoke_test_migration_with_multiple_outputs_and_templates() {
    let mut rng = fastrand::Rng::with_seed(0);
    let x = Array2::from_shape_fn((2, 20), |_| rng.f64() * 4.0 - 2.0);
    let y = ndarray::stack![
        ndarray::Axis(0),
        x.row(0).to_owned() * 2.0 + x.row(1),
        x.row(0).to_owned() - x.row(1)
    ];
    let datasets = Dataset::multi_output(x, y, None, names(2));
    let mut options = seeded(builtin(&["+", "sub", "*"]), Vec::new());
    options.template = Some(template("f(x1, x2)", &["f"], &["x1", "x2"], &[], &options));
    options.fraction_replaced_guesses = 0.5;
    options.niterations = 1;
    options.guesses = vec![
        vec![tguess(&[("f", eq("1.9 * #1 + #2"))])],
        vec![tguess(&[("f", eq("#1 - #2"))])],
    ];
    let result = equation_search(&datasets, &options).unwrap();
    for j in 0..2 {
        assert!(best_loss(&result, j) < 0.01, "output {j}");
    }
}

/// test/unit/misc/test_guesses.jl | "parse_guesses with mix of strings and expression objects"
#[test]
fn parse_guesses_with_mix_of_strings_and_expression_objects() {
    let dataset = small_dataset();
    let leaf = |node, consts| Expression::from(PostfixExpr::new(vec![node], consts, Metadata::default()));
    let feature = leaf(PNode::Var { feature: 0 }, vec![]);
    let constant = leaf(PNode::Const { idx: 0 }, vec![1.0]);
    let options = seeded(
        builtin(&["+", "sub", "*", "/", "sin", "cos"]),
        vec![feature.clone().into(), "x1 - x2".into(), constant.into()],
    );
    let parsed = parse_guesses(std::slice::from_ref(&dataset), &options).unwrap();
    assert_eq!(parsed.members.len(), 1);
    let members = &parsed.members[0];
    assert_eq!(members.len(), 3);

    assert_eq!(members[0].expr.trees[0].nodes, feature.trees[0].nodes);
    assert!(members[0].expr.constants().is_empty());
    let names = names(2);
    let named = StringTreeOptions {
        variable_names: Some(&names),
        ..Default::default()
    };
    assert_eq!(members[1].expr.string(&options, named), "x1 - x2");
    assert_ne!(members[2].expr.constants(), [1.0]);
}

/// test/unit/misc/test_guesses.jl | "maxsize warning"
#[test]
fn maxsize_warning() {
    let dataset = small_dataset();
    let mut options = seeded(
        builtin(&["+", "sub", "*", "/"]),
        vec!["x1 * x2 + x1 * x2 + x1 * x2 + x1 * x2 + x1 * x2".into()],
    );
    options.maxsize = 7;
    let warnings = parse_guesses(std::slice::from_ref(&dataset), &options)
        .unwrap()
        .warnings;
    assert_eq!(warnings.len(), 1);
    assert!(
        warnings[0].contains("complexity") && warnings[0].contains("maxsize"),
        "{warnings:?}"
    );

    options.guesses = vec![vec!["x1 + x2".into()]];
    let warnings = parse_guesses(std::slice::from_ref(&dataset), &options)
        .unwrap()
        .warnings;
    assert!(warnings.is_empty(), "{warnings:?}");
}

/// test/unit/misc/test_guesses.jl | "Vector of vectors input for single output"
#[test]
fn vector_of_vectors_input_for_single_output() {
    let dataset = small_dataset();
    let options = seeded(builtin(&["+", "sub"]), vec!["x1 + x2".into(), "x1 - x2".into()]);
    let parsed = parse_guesses(std::slice::from_ref(&dataset), &options).unwrap();
    assert_eq!(parsed.members.len(), 1);
    assert_eq!(parsed.members[0].len(), 2);
}

/// test/unit/misc/test_guesses.jl | "Multiple outputs guesses format validation"
#[test]
fn multiple_outputs_guesses_format_validation() {
    let datasets = [random_dataset(2, 10, 0, |x| x[0]), random_dataset(2, 10, 1, |x| x[1])];
    let options = seeded(builtin(&["+", "sub"]), vec!["x1 + x2".into(), "x1 - x2".into()]);
    let err = parse_guesses(&datasets, &options).err().unwrap();
    assert_eq!(
        err,
        "`guesses` must hold one list of guesses per output: it has 1 lists for 2 outputs"
    );
}

/// test/unit/misc/test_guesses.jl | "Template guesses preserve supplied parameters"
#[test]
fn template_guesses_preserve_supplied_parameters() {
    let supplied = vec![5.0, 10.0];
    let x = array![[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 1.0, 2.0]];
    let y = Array1::from_iter(x.columns().into_iter().map(|c| c[0] * supplied[c[1] as usize - 1]));
    let dataset = Dataset::with_weights_and_names(x, y, None, names(2));
    let mut options = seeded(builtin(&["*"]), vec![tguess(&[("f", eq("#1 * #2")), ("p", eq("x1"))])]);
    options.template = Some(template(
        "f(x, p[category])",
        &["f"],
        &["x", "category"],
        &[("p", 2)],
        &options,
    ));
    options.should_optimize_constants = false;
    let err = parse_guesses(std::slice::from_ref(&dataset), &options).err().unwrap();
    assert!(err.contains("the parameter `p` must be a vector"), "{err}");

    options.guesses = vec![vec![tguess(&[("f", eq("#1 * #2")), ("p", param(&supplied))])]];
    let parsed = parse_guesses(std::slice::from_ref(&dataset), &options).unwrap();
    assert_eq!(parsed.members[0][0].expr.params, supplied);

    let result = equation_search(std::slice::from_ref(&dataset), &options).unwrap();
    assert!(best_loss(&result, 0) < 1e-12);
}

/// test/unit/expressions/test_expression_builder.jl | "NamedTuple support in parse_expression"
#[test]
fn named_tuple_support_in_parse_expression() {
    let dataset = small_dataset();
    let mut options = seeded(builtin(&["+", "sub", "*", "/", "cos", "sin"]), Vec::new());
    options.template = Some(template(
        "f(x1, x2) * g(x1, x2)",
        &["f", "g"],
        &["x1", "x2"],
        &[],
        &options,
    ));
    options.should_optimize_constants = false;
    let cases = [
        (("#1 + 1.0", "#2 - 0.5"), "f = #1 + 1; g = #2 - 0.5"),
        (("cos(#1)", "sin(#2)"), "f = cos(#1); g = sin(#2)"),
    ];
    for ((f, g), printed) in cases {
        options.guesses = vec![vec![tguess(&[("f", eq(f)), ("g", eq(g))])]];
        let parsed = parse_guesses(std::slice::from_ref(&dataset), &options).unwrap();
        let expr = &parsed.members[0][0].expr;
        assert_eq!(expr.trees.len(), 2);
        let precise = StringTreeOptions {
            print_precision: Some(3),
            ..Default::default()
        };
        assert_eq!(expr.string(&options, precise), printed);
    }
}

/// test/unit/templates/test_template_parameter_value_types.jl | "template parser splits and
/// validates flat parameter entries". SymbolicRegression.jl's separate `parameters=` keyword of
/// `parse_expression` has no counterpart: a guess gives its parameters by name.
#[test]
fn template_parser_splits_and_validates_flat_parameter_entries() {
    let dataset = Dataset::with_weights_and_names(array![[1.0, 2.0]], array![0.0, 0.0], None, names(1));
    let mut options = seeded(builtin(&["+"]), Vec::new());
    options.template = Some(template(
        "f(x) + p[1] + q[1]",
        &["f"],
        &["x"],
        &[("p", 2), ("q", 1)],
        &options,
    ));
    options.should_optimize_constants = false;
    let mut parse = |entries: &[(&str, TemplateGuessValue<f64>)]| {
        options.guesses = vec![vec![tguess(entries)]];
        parse_guesses(std::slice::from_ref(&dataset), &options).map(|p| p.members[0][0].expr.params.clone())
    };

    assert_eq!(parse(&[("f", eq("#1"))]).unwrap(), [0.0, 0.0, 0.0]);
    assert_eq!(
        parse(&[("q", param(&[3.0])), ("f", eq("#1"))]).unwrap(),
        [0.0, 0.0, 3.0]
    );
    let errors = [
        (
            parse(&[("f", eq("#1")), ("unknown", param(&[1.0]))]),
            "`unknown` is not one of the template's expressions or parameters (f, p, q)",
        ),
        (
            parse(&[("f", eq("#1")), ("p", param(&[1.0]))]),
            "the parameter `p` must have length 2, got 1",
        ),
        (
            parse(&[("f", eq("#1")), ("p", eq("1.0"))]),
            "the parameter `p` must be a vector",
        ),
    ];
    for (result, reason) in errors {
        let err = result.err().unwrap();
        assert!(err.contains(reason), "{err}");
    }
}

/// test/integration/ext/loopvectorization/test_template_expression.jl | "TemplateExpression guess
/// validation"
#[test]
fn template_expression_guess_validation() {
    let mut dataset = small_dataset();
    let mut options = seeded(
        builtin(&["+", "sub", "*", "/"]),
        vec![tguess(&[("f", eq("x1 + x2")), ("g", eq("x1 * x2"))])],
    );
    options.template = Some(template(
        "f(x1, x2) + g(x1, x2)",
        &["f", "g"],
        &["x1", "x2"],
        &[],
        &options,
    ));
    let err = parse_guesses(std::slice::from_ref(&dataset), &options).err().unwrap();
    assert!(
        err.contains("found the variable name `x1` in the expression `f`; write its arguments as `#1`, `#2`, ..."),
        "{err}"
    );

    dataset.variable_names = vec!["alpha".into(), "beta".into()];
    options.guesses = vec![vec![tguess(&[("f", eq("alpha + beta")), ("g", eq("alpha * beta"))])]];
    let err = parse_guesses(std::slice::from_ref(&dataset), &options).err().unwrap();
    assert!(
        err.contains("found the variable name `alpha` in the expression `f`"),
        "{err}"
    );
}
