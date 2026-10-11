//! Every guess string in SymbolicRegression.jl's tests and docs (f97764fb) and PySR's docs and tests
//! (c15a9eb), found by searching for `guess`. Each parses through `parse_guesses` or is rejected
//! as SymbolicRegression.jl rejects it, and each parsed guess evaluates as a hand-written closure
//! does, within 1e-12 relative error. SymbolicRegression.jl names features `x1`, `x2`, ... and PySR
//! `x0`, `x1`, ...; each case's dataset carries the names its source uses.

use dynamic_expressions::dsl::{CustomOps, TemplateStructure, parse_operator_tokens};
use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
use ndarray::{Array1, Array2, array};

use crate::dataset::Dataset;
use crate::guesses::{Guess, TemplateGuessValue, parse_guesses};
use crate::options::Options;
use crate::pop_member::Evaluator;

type Ops = BuiltinOpsF64<CustomOps<f64>>;
const D: usize = 2;

/// The operators of every case, with SymbolicRegression.jl's test definitions of the custom ones.
/// `pythag_neg` is written with a ternary, since the DSL has no assignment inside an expression.
const OPERATORS: [&str; 12] = [
    "+",
    "sub",
    "*",
    "/",
    "sin",
    "cos",
    "log",
    "sqrt",
    "pythag_pos(x, y) = sqrt(x^2 + y^2)",
    "pythag_neg(x, y) = x^2 - y^2 < 0 ? NaN : sqrt(x^2 - y^2)",
    "custom_sin(x) = sin(x) + 0.1",
    "custom_mul(x, y) = x * y * 1.1",
];

const SR: &str = "SymbolicRegression.jl/";
const PYSR: &str = "PySR/";

const JL2: &[&str] = &["x1", "x2"];
const JL6: &[&str] = &["x1", "x2", "x3", "x4", "x5", "x6"];
const PY2: &[&str] = &["x0", "x1"];
const PY3: &[&str] = &["x0", "x1", "x2"];

fn pythag(a: f64, b: f64) -> f64 {
    (a * a + b * b).sqrt()
}

/// `(source, guess, variable names, expected value on a row of features)`.
type PlainCase = (&'static str, &'static str, &'static [&'static str], fn(&[f64]) -> f64);

const PLAIN: &[PlainCase] = &[
    (
        "test/unit/misc/test_guesses.jl:14",
        "2.0*x1*x1 + 3.0*x2 + 0.5",
        JL2,
        |r| 2.0 * r[0] * r[0] + 3.0 * r[1] + 0.5,
    ),
    ("test/unit/misc/test_guesses.jl:21", "1.0*x1 + 1.0*x2 + 0.0", JL2, |r| {
        r[0] + r[1]
    }),
    ("test/unit/misc/test_guesses.jl:39", "x1 + x2", JL2, |r| r[0] + r[1]),
    (
        "test/unit/misc/test_guesses.jl:176",
        "4.561253 - ((x1 - x2) * 0.18459733)",
        JL2,
        |r| 4.561253 - (r[0] - r[1]) * 0.18459733,
    ),
    (
        "test/unit/misc/test_guesses.jl:186",
        "4.561253f0 - ((x1 - x2) * 0.18459733f0)",
        JL2,
        |r| f64::from(4.561253f32) - (r[0] - r[1]) * f64::from(0.18459733f32),
    ),
    (
        "test/unit/misc/test_guesses.jl:218",
        "pythag_pos(x1, x2) + custom_sin(x1)",
        JL2,
        |r| pythag(r[0], r[1]) + r[0].sin() + 0.1,
    ),
    (
        "test/unit/misc/test_guesses.jl:228",
        "pythag_pos(x1, 4.51352 - ((x2 - x1) * 0.07425507))",
        JL2,
        |r| pythag(r[0], 4.51352 - (r[1] - r[0]) * 0.07425507),
    ),
    ("test/unit/misc/test_guesses.jl:239", "pythag_pos(x1, x2)", JL2, |r| {
        pythag(r[0], r[1])
    }),
    (
        "test/unit/misc/test_guesses.jl:239",
        "pythag_neg(x1, x2) + 0.5",
        JL2,
        |r| {
            let d = r[0] * r[0] - r[1] * r[1];
            if d < 0.0 { f64::NAN } else { d.sqrt() + 0.5 }
        },
    ),
    ("test/unit/misc/test_guesses.jl:239", "custom_sin(x1) * x2", JL2, |r| {
        (r[0].sin() + 0.1) * r[1]
    }),
    ("test/unit/misc/test_guesses.jl:271", "pythag_pos(x1, x2)", JL2, |r| {
        pythag(r[0], r[1])
    }),
    (
        "test/unit/misc/test_guesses.jl:279",
        "pythag_pos(x1, x2) + 0.0",
        JL2,
        |r| pythag(r[0], r[1]),
    ),
    ("test/unit/misc/test_guesses.jl:394", "x1 - x2", JL2, |r| r[0] - r[1]),
    (
        "test/unit/misc/test_guesses.jl:440",
        "x1 * x2 + x1 * x2 + x1 * x2 + x1 * x2 + x1 * x2",
        JL2,
        |r| 5.0 * r[0] * r[1],
    ),
    ("test/unit/misc/test_guesses.jl:451", "x1 + x2", JL2, |r| r[0] + r[1]),
    ("test/unit/misc/test_guesses.jl:467", "x1 + x2", JL2, |r| r[0] + r[1]),
    ("test/unit/misc/test_guesses.jl:467", "x1 - x2", JL2, |r| r[0] - r[1]),
    (
        "test/unit/misc/test_guesses.jl:514",
        "2.0*x1*x1 + 3.0*x2 + 0.5",
        JL2,
        |r| 2.0 * r[0] * r[0] + 3.0 * r[1] + 0.5,
    ),
    (
        "test/unit/backsolve/test_backsolve_mutation.jl:466",
        "x1 + 1.0",
        JL2,
        |r| r[0] + 1.0,
    ),
    ("test/unit/basic/test_abstract_popmember.jl:200", "x1 - x2", JL2, |r| {
        r[0] - r[1]
    }),
    ("test/unit/basic/test_abstract_popmember.jl:240", "x1 - x2", JL2, |r| {
        r[0] - r[1]
    }),
    ("test/unit/basic/test_abstract_popmember.jl:254", "x1 - x2", JL2, |r| {
        r[0] - r[1]
    }),
    (
        "docs/src/examples.md:496",
        "sin(x1 * x2) + cos(x3) * x4 + x5 / (x6 * x6 + 0.9)",
        JL6,
        |r| (r[0] * r[1]).sin() + r[2].cos() * r[3] + r[4] / (r[5] * r[5] + 0.9),
    ),
];

const PYSR_PLAIN: &[PlainCase] = &[
    ("docs/src/migration.md:195", "sin(x1 * 2.1)", PY2, |r| {
        (r[1] * 2.1).sin()
    }),
    (
        "docs/src/examples/instrumentation.md:196",
        "(mass / sqrt(1 - (velocity * velocity))) - mass",
        &["mass", "velocity"],
        |r| r[0] / (1.0 - r[1] * r[1]).sqrt() - r[0],
    ),
    ("pysr/test/test_main.py:1252", "2.0*x0 + 3.0*x1 + 0.5", PY2, |r| {
        2.0 * r[0] + 3.0 * r[1] + 0.5
    }),
    ("pysr/test/test_main.py:1252", "x0 + x1", PY2, |r| r[0] + r[1]),
    (
        "pysr/test/test_main.py:1263",
        "2.0*feature1 + 3.0*feature2 + 0.5",
        &["feature1", "feature2"],
        |r| 2.0 * r[0] + 3.0 * r[1] + 0.5,
    ),
    ("pysr/test/test_main.py:1275", "2.0*x0 + x1", PY2, |r| 2.0 * r[0] + r[1]),
    ("pysr/test/test_main.py:1275", "x0 - x1", PY2, |r| r[0] - r[1]),
    ("pysr/test/test_main.py:1453", "x0 + x1", PY2, |r| r[0] + r[1]),
    ("pysr/test/test_main.py:1453", "x0 - x1", PY2, |r| r[0] - r[1]),
    ("pysr/test/test_main.py:1469", "x0 * x1 + x2", PY3, |r| {
        r[0] * r[1] + r[2]
    }),
    ("pysr/test/test_main.py:1478", "x1 * x2 + x0", PY3, |r| {
        r[1] * r[2] + r[0]
    }),
    ("pysr/test/test_main.py:1493", "log(x0) + 1.0 * x1", PY2, |r| {
        r[0].ln() + r[1]
    }),
];

/// A template: its combine formula, expressions, variable names, and parameter lengths.
type Template = (
    &'static str,
    &'static [&'static str],
    &'static [&'static str],
    &'static [(&'static str, usize)],
);

const F_PLUS_G: Template = ("f(x1, x2) + g(x1, x2)", &["f", "g"], JL2, &[]);
const F_X1_X2: Template = ("f(x1, x2)", &["f"], JL2, &[]);
const F_CATEGORY: Template = ("f(x, p[category])", &["f"], &["x", "category"], &[("p", 2)]);
const PY_F: Template = ("f(x0, x1)", &["f"], PY2, &[]);
const PY_AFFINE: Template = ("p[1] * f(x) + p[2]", &["f"], &["x"], &[("p", 2)]);

#[derive(Clone, Copy, Debug)]
enum Value {
    Ex(&'static str),
    Param(&'static [f64]),
}
use Value::{Ex, Param};

type Entries = &'static [(&'static str, Value)];

/// `(source, template, entries, variable names, whether the second feature is a category index,
/// expected value on a row of features)`.
type TemplateCase = (
    &'static str,
    Template,
    Entries,
    &'static [&'static str],
    bool,
    fn(&[f64]) -> f64,
);

const TEMPLATES: &[TemplateCase] = &[
    (
        "test/unit/misc/test_guesses.jl:62",
        F_PLUS_G,
        &[("f", Ex("2.0 * #1")), ("g", Ex("1.5 * #2"))],
        JL2,
        false,
        |r| 2.0 * r[0] + 1.5 * r[1],
    ),
    (
        "test/unit/misc/test_guesses.jl:85",
        F_PLUS_G,
        &[("f", Ex("#1")), ("g", Ex("#2"))],
        JL2,
        false,
        |r| r[0] + r[1],
    ),
    (
        "test/unit/misc/test_guesses.jl:85",
        F_PLUS_G,
        &[("f", Ex("2.0 * #1")), ("g", Ex("1.5 * #2"))],
        JL2,
        false,
        |r| 2.0 * r[0] + 1.5 * r[1],
    ),
    (
        "test/unit/misc/test_guesses.jl:114",
        ("f(x1, x2) + p[1] * x1 + p[2]", &["f"], JL2, &[("p", 2)]),
        &[("f", Ex("#1 * #2"))],
        JL2,
        false,
        |r| r[0] * r[1],
    ),
    (
        "test/unit/misc/test_guesses.jl:147",
        (
            "term1(input1, input2, input3) + term2(input1, input2, input3)",
            &["term1", "term2"],
            &["input1", "input2", "input3"],
            &[],
        ),
        &[("term1", Ex("#1 + 2.0 * #2")), ("term2", Ex("-1.0 * #3"))],
        &["input1", "input2", "input3"],
        false,
        |r| r[0] + 2.0 * r[1] - r[2],
    ),
    (
        "test/unit/misc/test_guesses.jl:336",
        F_PLUS_G,
        &[("f", Ex("pythag_pos(#1, #2)")), ("g", Ex("custom_mul(#1, #2)"))],
        JL2,
        false,
        |r| pythag(r[0], r[1]) + r[0] * r[1] * 1.1,
    ),
    (
        "test/unit/misc/test_guesses.jl:375",
        F_X1_X2,
        &[("f", Ex("1.9 * #1 + #2"))],
        JL2,
        false,
        |r| 1.9 * r[0] + r[1],
    ),
    (
        "test/unit/misc/test_guesses.jl:375",
        F_X1_X2,
        &[("f", Ex("#1 - #2"))],
        JL2,
        false,
        |r| r[0] - r[1],
    ),
    (
        "test/unit/misc/test_guesses.jl:657",
        F_CATEGORY,
        &[("f", Ex("#1 * #2")), ("p", Param(&[5.0, 10.0]))],
        JL2,
        true,
        |r| r[0] * [5.0, 10.0][r[1] as usize - 1],
    ),
    (
        "test/unit/misc/test_guesses.jl:678",
        F_CATEGORY,
        &[("f", Ex("#1 * #2")), ("p", Param(&[5.0, 10.0]))],
        JL2,
        true,
        |r| r[0] * [5.0, 10.0][r[1] as usize - 1],
    ),
    // The docs give the guess without its template; this one has a one-argument `f`, a two-argument
    // `g`, and a parameter vector `p` of length 2, as the guess needs.
    (
        "docs/src/examples.md:514",
        ("p[1] * f(x1) + p[2] * g(x1, x2)", &["f", "g"], JL2, &[("p", 2)]),
        &[
            ("f", Ex("cos(#1) + 0.1")),
            ("g", Ex("sin(#2) - 0.9")),
            ("p", Param(&[5.0, 10.0])),
        ],
        JL2,
        false,
        |r| 5.0 * (r[0].cos() + 0.1) + 10.0 * (r[1].sin() - 0.9),
    ),
];

const PYSR_TEMPLATES: &[TemplateCase] = &[
    (
        "pysr/test/test_main.py:1292",
        PY_F,
        &[("f", Ex("#1 + #2"))],
        PY2,
        false,
        |r| r[0] + r[1],
    ),
    (
        "pysr/test/test_main.py:1311",
        (
            "p[1] * f(x) + p[2] * x + p[3]",
            &["f"],
            &["x"],
            &[("p", 3), ("unused", 1)],
        ),
        &[("f", Ex("#1 * #1")), ("p", Param(&[5.0, 10.0, 0.8]))],
        &["x"],
        false,
        |r| 5.0 * r[0] * r[0] + 10.0 * r[0] + 0.8,
    ),
    (
        "pysr/test/test_main.py:1331",
        PY_F,
        &[("f", Ex("#1 + #2"))],
        PY2,
        false,
        |r| r[0] + r[1],
    ),
    (
        "pysr/test/test_main.py:1332",
        PY_F,
        &[("f", Ex("#1 - #2"))],
        PY2,
        false,
        |r| r[0] - r[1],
    ),
    (
        "pysr/test/test_main.py:1352",
        PY_AFFINE,
        &[("f", Ex("#1")), ("p", Param(&[1.0, 0.0]))],
        &["x"],
        false,
        |r| r[0],
    ),
    (
        "pysr/test/test_main.py:1357",
        PY_AFFINE,
        &[("f", Ex("#1")), ("p", Param(&[3.0, 4.0]))],
        &["x"],
        false,
        |r| 3.0 * r[0] + 4.0,
    ),
    (
        "pysr/test/test_main.py:1373",
        PY_AFFINE,
        &[("f", Ex("#1")), ("p", Param(&[2.0, 1.0]))],
        &["x"],
        false,
        |r| 2.0 * r[0] + 1.0,
    ),
    (
        "pysr/test/test_main.py:1374",
        PY_AFFINE,
        &[("f", Ex("#1")), ("p", Param(&[-3.0, 0.5]))],
        &["x"],
        false,
        |r| -3.0 * r[0] + 0.5,
    ),
];

#[derive(Clone, Copy)]
enum Rejected {
    Plain(&'static str),
    Template(Template, Entries),
}

/// `(source, guess, variable names, the reason the error must give)`.
type RejectCase = (&'static str, Rejected, &'static [&'static str], &'static str);

const REJECTED: &[RejectCase] = &[
    (
        "SymbolicRegression.jl/test/unit/misc/test_guesses.jl:304",
        Rejected::Plain("pythag_pos(x1)"),
        JL2,
        "the operator `pythag_pos` takes 2 arguments, not 1",
    ),
    (
        "SymbolicRegression.jl/test/unit/misc/test_guesses.jl:309",
        Rejected::Plain("nonexistent_op(x1, x2)"),
        JL2,
        "`nonexistent_op` is not one of the search's operators",
    ),
    (
        "SymbolicRegression.jl/test/unit/misc/test_guesses.jl:634",
        Rejected::Plain("MissingVec2(1.0, 2.0)"),
        JL2,
        "`MissingVec2` is not one of the search's operators",
    ),
    (
        "SymbolicRegression.jl/test/unit/misc/test_guesses.jl:660",
        Rejected::Template(F_CATEGORY, &[("f", Ex("#1 * #2")), ("p", Ex("x1"))]),
        JL2,
        "the parameter `p` must be a vector",
    ),
    (
        "SymbolicRegression.jl/test/integration/ext/loopvectorization/test_template_expression.jl:645",
        Rejected::Template(F_PLUS_G, &[("f", Ex("x1 + x2")), ("g", Ex("x1 * x2"))]),
        JL2,
        "found the variable name `x1` in the expression `f`",
    ),
    (
        "SymbolicRegression.jl/test/integration/ext/loopvectorization/test_template_expression.jl:655",
        Rejected::Template(F_PLUS_G, &[("f", Ex("alpha + beta")), ("g", Ex("alpha * beta"))]),
        &["alpha", "beta"],
        "found the variable name `alpha` in the expression `f`",
    ),
    (
        "PySR/pysr/test/test_main.py:1395",
        Rejected::Template(
            PY_AFFINE,
            &[("f", Ex("#1")), ("missing", Ex("#1")), ("p", Param(&[1.0, 2.0]))],
        ),
        &["x"],
        "`missing` is not one of the template's expressions or parameters (f, p)",
    ),
    (
        "PySR/pysr/test/test_main.py:1396",
        Rejected::Template(PY_AFFINE, &[("p", Param(&[1.0, 2.0]))]),
        &["x"],
        "the guess must give every expression of the template (f)",
    ),
    (
        "PySR/pysr/test/test_main.py:1398",
        Rejected::Template(PY_AFFINE, &[("f", Ex("#1")), ("p", Param(&[1.0]))]),
        &["x"],
        "the parameter `p` must have length 2, got 1",
    ),
];

/// `(source, guess lists, number of outputs, members parsed per output or the error)`.
type ListCase = (
    &'static str,
    &'static [&'static [&'static str]],
    usize,
    Result<&'static [usize], &'static str>,
);

const LISTS: &[ListCase] = &[
    (
        "SymbolicRegression.jl/test/unit/misc/test_guesses.jl:486",
        &[&["x1 + x2", "x1 - x2"]],
        2,
        Err("`guesses` must hold one list of guesses per output: it has 1 lists for 2 outputs"),
    ),
    (
        "PySR/pysr/test/test_main.py:1415",
        &[&["x0", "x1"]],
        2,
        Err("`guesses` must hold one list of guesses per output: it has 1 lists for 2 outputs"),
    ),
    (
        "PySR/pysr/test/test_main.py:1423",
        &[&["x0"]],
        2,
        Err("`guesses` must hold one list of guesses per output: it has 1 lists for 2 outputs"),
    ),
    (
        "PySR/pysr/test/test_main.py:1442",
        &[&["x0 + x1"], &["x0 - x1"]],
        1,
        Err("`guesses` must hold one list of guesses per output: it has 2 lists for 1 outputs"),
    ),
    (
        "PySR/pysr/test/test_main.py:1453",
        &[&["x0 + x1", "x0 - x1"]],
        1,
        Ok(&[2]),
    ),
    (
        "PySR/pysr/test/test_main.py:1275",
        &[&["2.0*x0 + x1"], &["x0 - x1"]],
        2,
        Ok(&[1, 1]),
    ),
    ("PySR/pysr/test/test_main.py:1508", &[&[]], 1, Ok(&[0])),
];

/// Guesses outside the corpus, with the reason.
const EXCLUDED: &[(&str, &str, &str)] = &[
    (
        "SymbolicRegression.jl/test/unit/misc/test_guesses.jl:609",
        "Vec2(1.0, 2.0)",
        "custom value types are not supported",
    ),
    (
        "SymbolicRegression.jl/test/unit/misc/test_guesses.jl:629",
        "(; f=\"Vec2(3.0, 4.0)\")",
        "custom value types are not supported",
    ),
    (
        "PySR/docs/src/examples/value-types.md:413",
        "add_vectors(x1, Vec2([1.0, 2.0]))",
        "custom value types are not supported",
    ),
    (
        "PySR/pysr/test/test_type_specs.py:290",
        "transform(x0)",
        "custom value types are not supported",
    ),
    (
        "PySR/pysr/test/test_type_specs.py:1621",
        "add_vectors(x0, {type_name}([1.5, -2.5]))",
        "custom value types are not supported",
    ),
    (
        "PySR/pysr/test/test_type_specs.py:1652",
        "{\"f\": concat(#1, {type_name}(\"!\"))}",
        "custom value types are not supported",
    ),
    (
        "PySR/pysr/test/test_type_specs.py:1686",
        "{\"f\": \"#1\", \"p\": [\"!\"]}",
        "custom value types are not supported",
    ),
    (
        "PySR/pysr/test/test_type_specs.py:1717",
        "{\"f\": \"#1\", \"p\": parameter_values}",
        "custom value types are not supported",
    ),
    (
        "PySR/pysr/test/test_type_specs.py:1757",
        "{\"f\": \"#1\", \"p\": [(2.0, \"two\")]}",
        "custom value types are not supported",
    ),
    (
        "PySR/pysr/test/test_type_specs.py:1771",
        "identity_{type_name}(2.0)",
        "custom value types are not supported",
    ),
    (
        "PySR/pysr/test/test_type_specs.py:1786",
        "MissingGuessValue(\"!\")",
        "custom value types are not supported",
    ),
    (
        "PySR/pysr/test/test_main.py:1397",
        "{\"f\": [\"#1\"], \"p\": [1.0, 2.0]}",
        "a list in place of an expression string cannot be written as a `Guess`",
    ),
    (
        "PySR/pysr/test/test_main.py:1399",
        "{\"f\": \"#1\", \"p\": 1.0}",
        "PySR rejects it in Python; a scalar parameter cannot be written as a `Guess`",
    ),
    (
        "PySR/pysr/test/test_main.py:1400",
        "{\"f\": \"#1\", \"p\": [[1.0, 2.0]]}",
        "PySR rejects it in Python; a nested parameter cannot be written as a `Guess`",
    ),
    (
        "PySR/pysr/test/test_main.py:1432",
        "\"x0 + x1\"",
        "PySR rejects a bare string in Python; `Options::guesses` cannot hold one",
    ),
];

/// Five rows of six features. The second feature lies in (0, 1) so that `1 - velocity^2` is
/// positive; for a category template it holds the indices 1 and 2 instead.
fn dataset(names: &[&str], category: bool) -> Dataset<f64> {
    let mut x: Array2<f64> = array![
        [0.7, 1.3, 2.1, 0.9, 1.7],
        [0.3, 0.6, 0.2, 0.8, 0.5],
        [1.1, -0.4, 0.5, 2.2, -1.3],
        [-0.8, 0.9, 1.6, -0.2, 0.4],
        [2.0, -1.5, 0.3, 1.2, 0.8],
        [0.6, 1.4, -0.9, 0.1, 1.9],
    ];
    if category {
        x.row_mut(1).assign(&array![1.0, 2.0, 1.0, 2.0, 1.0]);
    }
    let x = x.slice(ndarray::s![..names.len(), ..]).to_owned();
    let names = names.iter().map(|n| n.to_string()).collect();
    Dataset::with_weights_and_names(x, Array1::zeros(5), None, names)
}

fn options(template: Option<Template>, guess: Guess<f64, Ops, D>) -> Options<f64, Ops, D> {
    let (custom, names) = parse_operator_tokens::<f64, _>(OPERATORS).unwrap();
    let set = BuiltinOpsF64::new().with_custom(custom);
    let template = template.map(|(combine, expressions, variables, parameters)| {
        TemplateStructure::new(combine, expressions, None, variables, parameters, &set).unwrap()
    });
    Options {
        operators: set.from_names(&names).unwrap(),
        template,
        guesses: vec![vec![guess]],
        should_optimize_constants: false,
        progress: false,
        ..Default::default()
    }
}

fn template_guess(entries: Entries) -> Guess<f64, Ops, D> {
    Guess::Template(
        entries
            .iter()
            .map(|&(name, value)| {
                let value = match value {
                    Ex(src) => TemplateGuessValue::Equation(src.to_string()),
                    Param(values) => TemplateGuessValue::Parameter(values.to_vec()),
                };
                (name.to_string(), value)
            })
            .collect(),
    )
}

fn close(got: f64, want: f64) -> bool {
    if !got.is_finite() || !want.is_finite() {
        return got == want || (got.is_nan() && want.is_nan());
    }
    (got - want).abs() <= 1e-12 * got.abs().max(want.abs())
}

#[test]
fn close_requires_matching_finiteness() {
    for (got, want) in [
        (f64::INFINITY, 1.0),
        (1.0, f64::INFINITY),
        (f64::INFINITY, f64::NEG_INFINITY),
        (f64::NAN, 1.0),
    ] {
        assert!(!close(got, want), "{got} vs {want}");
    }
    assert!(close(1.0, 1.0 + 1e-13));
    assert!(close(f64::NAN, f64::NAN));
    assert!(close(f64::INFINITY, f64::INFINITY));
}

fn assert_evaluates(source: &str, dataset: &Dataset<f64>, options: &Options<f64, Ops, D>, expected: fn(&[f64]) -> f64) {
    let parsed = parse_guesses(std::slice::from_ref(dataset), options).unwrap_or_else(|e| panic!("{source}: {e}"));
    let member = &parsed.members[0][0];
    let mut evaluator = Evaluator::new(dataset.n_rows);
    evaluator.eval(&member.expr, &member.plans, options, dataset.x.view());
    for (i, column) in dataset.x.columns().into_iter().enumerate() {
        let (got, want) = (evaluator.yhat[i], expected(&column.to_vec()));
        assert!(close(got, want), "{source}: row {i}: {got} vs {want}");
    }
}

#[test]
fn plain_guesses_evaluate_as_written() {
    let cases = PLAIN
        .iter()
        .map(|c| (SR, c))
        .chain(PYSR_PLAIN.iter().map(|c| (PYSR, c)));
    for (repo, &(source, guess, names, expected)) in cases {
        let source = format!("{repo}{source}: `{guess}`");
        assert_evaluates(&source, &dataset(names, false), &options(None, guess.into()), expected);
    }
}

#[test]
fn template_guesses_evaluate_as_written() {
    let cases = TEMPLATES
        .iter()
        .map(|c| (SR, c))
        .chain(PYSR_TEMPLATES.iter().map(|c| (PYSR, c)));
    for (repo, &(source, template, entries, names, category, expected)) in cases {
        let source = format!("{repo}{source}");
        let options = options(Some(template), template_guess(entries));
        assert_evaluates(&source, &dataset(names, category), &options, expected);
    }
}

#[test]
fn rejected_guesses_fail_with_their_reason() {
    for &(source, guess, names, reason) in REJECTED {
        let options = match guess {
            Rejected::Plain(src) => options(None, src.into()),
            Rejected::Template(template, entries) => options(Some(template), template_guess(entries)),
        };
        let err = parse_guesses(&[dataset(names, false)], &options)
            .err()
            .unwrap_or_else(|| panic!("{source}: parsed"));
        assert!(err.starts_with("invalid guess "), "{source}: {err}");
        assert!(err.contains(reason), "{source}: {err}");
    }
}

#[test]
fn guess_lists_match_the_outputs() {
    for &(source, lists, n_outputs, expected) in LISTS {
        let mut options = options(None, "x0".into());
        options.guesses = lists
            .iter()
            .map(|list| list.iter().map(|&g| g.into()).collect())
            .collect();
        let datasets = vec![dataset(PY2, false); n_outputs];
        let parsed = parse_guesses(&datasets, &options).map(|p| p.members.iter().map(Vec::len).collect::<Vec<_>>());
        assert_eq!(parsed.as_deref(), expected.map_err(String::from).as_deref(), "{source}");
    }
}

#[test]
fn every_corpus_entry_is_listed_once() {
    let mut keys: Vec<String> = PLAIN
        .iter()
        .map(|c| format!("{SR}{}{}", c.0, c.1))
        .chain(PYSR_PLAIN.iter().map(|c| format!("{PYSR}{}{}", c.0, c.1)))
        .chain(TEMPLATES.iter().map(|c| format!("{SR}{}{:?}", c.0, c.2)))
        .chain(PYSR_TEMPLATES.iter().map(|c| format!("{PYSR}{}", c.0)))
        .chain(REJECTED.iter().map(|c| c.0.to_string()))
        .chain(LISTS.iter().map(|c| c.0.to_string()))
        .chain(EXCLUDED.iter().map(|c| c.0.to_string()))
        .collect();
    let n = keys.len();
    keys.sort();
    keys.dedup();
    assert_eq!(keys.len(), n);
    assert!(EXCLUDED.iter().all(|(_, _, reason)| !reason.is_empty()));
}
