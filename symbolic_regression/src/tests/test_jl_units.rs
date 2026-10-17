//! SymbolicRegression.jl's units testitems (test/integration/ext/dynamicquantities_units).

use dynamic_expressions::dsl::{CustomOps, parse_operator_tokens};
use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
use dynamic_expressions::strings::{StringTreeOptions, string_tree};
use dynamic_expressions::{OperatorSet, PostfixExpr};
use ndarray::{Array1, Array2};

use super::common::infix;
use crate::dataset::Dataset;
use crate::dimensional_analysis::{WildcardQuantity, degn_eval, violates_dimensional_constraints};
use crate::random::standard_normal;
use crate::units::{DENOM, Dimensions, Unit, UnitLabels};

type Ops = BuiltinOpsF64<CustomOps<f64>>;

fn operators(tokens: &[&str]) -> Ops {
    let (custom, _) = parse_operator_tokens::<f64, _>(tokens.iter().copied()).unwrap();
    BuiltinOpsF64::new().with_custom(custom)
}

/// `randn(n_features, n_rows)` with a seeded generator.
fn randn(n_features: usize, n_rows: usize, seed: u64) -> Array2<f64> {
    let mut rng = fastrand::Rng::with_seed(seed);
    Array2::from_shape_fn((n_features, n_rows), |_| standard_normal(&mut rng))
}

fn dataset(n_features: usize, x_units: &[&str], y_units: &str) -> Dataset<f64> {
    Dataset::new(randn(n_features, 64, 0), Array1::zeros(64))
        .with_units(Some(x_units), Some(y_units))
        .unwrap()
}

fn violates(ops: &Ops, data: &Dataset<f64>, src: &str, dimensionless_constants_only: bool) -> bool {
    let tree: PostfixExpr<f64, Ops, 2> = infix(ops, src);
    violates_dimensional_constraints(
        &tree,
        data.units.as_ref().unwrap(),
        data.x.view(),
        ops,
        dimensionless_constants_only,
    )
}

fn length() -> Dimensions {
    Dimensions([DENOM, 0, 0, 0, 0, 0, 0])
}

fn mass() -> Dimensions {
    Dimensions([0, DENOM, 0, 0, 0, 0, 0])
}

/// test/integration/ext/dynamicquantities_units/test_units.jl | "Dimensional analysis"
#[test]
fn dimensional_analysis() {
    let units: Vec<Unit> = ["m", "1", "kg"].iter().map(|s| Unit::parse(s).unwrap()).collect();
    assert_eq!(
        units.iter().map(|u| u.dims).collect::<Vec<_>>(),
        [length(), Dimensions::default(), mass()]
    );
    assert_eq!(units.iter().map(Unit::label).collect::<Vec<_>>(), ["[m]", "", "[kg]"]);
    assert_eq!(Unit::parse("1.5").unwrap().label(), "[1.5 ]");
    assert_eq!(Unit::parse("").unwrap(), Unit::dimensionless());

    let ops = operators(&[
        "-",
        "*",
        "/",
        "custom_op(x, y) = x + y",
        "^",
        "cos",
        "cbrt",
        "sqrt",
        "abs",
        "inv",
    ]);
    let data = Dataset::new(randn(3, 100, 0), Array1::zeros(100))
        .with_units(Some(&["m", "1", "kg"]), Some("1"))
        .unwrap();
    let data_units = data.units.as_ref().unwrap();
    assert_eq!(data_units.x, units);
    assert_eq!(data_units.y, Some(Unit::dimensionless()));

    let good = [
        "3.2",
        "3.2 * x1 / x1",
        "1.0 * (3.2 * x1 - x2 * x1)",
        "3.2 * x1 - x2",
        "cos(3.2 * x1)",
        "cos(0.9 * x1 - 0.5 * x2)",
        "1.0 * (x1 - 0.5 * (x3 * (cos(0.9 * x1 - 0.5 * x2) - 1.2)))",
        "1.0 * custom_op(x1, x1)",
        "1.0 * custom_op(x1, 2.1 * x3)",
        "1.0 * custom_op(custom_op(x1, 2.1 * x3), x1)",
        "1.0 * custom_op(custom_op(x1, 2.1 * x3), 0.9 * x1)",
        "x2",
        "1.0 * x1",
        "1.0 * x3",
        "(1.0 * x1) ^ 3.2",
        "1.0 * (cbrt(x3 * x3 * x3) - x3)",
        "1.0 * (sqrt(x3 * x3) - x3)",
        "1.0 * (sqrt(abs(x3) * abs(x3)) - x3)",
        "inv(x2)",
        "1.0 * inv(x1)",
        "x3 * inv(x3)",
    ];
    let bad = [
        "x1",
        "x3",
        "x1 - x3",
        "1.0 * cos(x1)",
        "1.0 * cos(x1 - 0.5 * x2)",
        "1.0 * (x1 - (x3 * (cos(0.9 * x1 - 0.5 * x2) - 1.2)))",
        "1.0 * custom_op(x1, x3)",
        "1.0 * custom_op(custom_op(x1, 2.1 * x3), x3)",
        "1.0 * cos(0.8606301 / x1) / cos(custom_op(cos(x1), 3.2263336))",
        "1.0 * x1 ^ 3.2",
        "1.0 * (1.0 * x1) ^ x1",
        "1.0 * (cbrt(x3 * x3) - x3)",
        "1.0 * (sqrt(abs(x3)) - x3)",
        "inv(x3)",
        "inv(x1)",
        "x1 * inv(x3)",
    ];
    for src in good {
        assert!(!violates(&ops, &data, src, false), "{src} should be consistent");
    }
    for src in bad {
        assert!(violates(&ops, &data, src, false), "{src} should violate");
    }
}

/// test/integration/ext/dynamicquantities_units/test_units.jl | "Operator compatibility"
///
/// The Julia testitem calls `square`, `plus`, `greater`, and the rest on DynamicQuantities
/// `Quantity` values; the port has no quantity-valued evaluation. These are the assertions that
/// hold for dimensional analysis, on dimensioned (non-wildcard) arguments.
#[test]
fn operator_compatibility() {
    let ops = BuiltinOpsF64::new();
    let m = |v: f64| WildcardQuantity::new(v, length(), false);
    let s = |v: f64| WildcardQuantity::new(v, Dimensions([0, 0, DENOM, 0, 0, 0, 0]), false);
    let eval = |name: &str, args: &[WildcardQuantity<f64>]| {
        degn_eval(&ops, ops.lookup_with_arity(name, args.len() as u8).unwrap(), args)
    };
    assert_eq!(
        eval("square", &[m(1.0)]),
        WildcardQuantity::new(1.0, length().pow(2 * DENOM), false)
    );
    assert_eq!(
        eval("cube", &[m(1.0)]),
        WildcardQuantity::new(1.0, length().pow(3 * DENOM), false)
    );
    assert_eq!(eval("+", &[m(1.0), m(1.0)]), m(2.0));
    assert!(eval("+", &[m(1.0), s(1.0)]).violates);
    assert_eq!(eval("-", &[m(1.0), m(1.0)]), m(0.0));
    assert!(eval("-", &[m(1.0), s(1.0)]).violates);
    assert_eq!(
        eval("*", &[m(1.0), m(1.0)]),
        WildcardQuantity::new(1.0, length().pow(2 * DENOM), false)
    );
    assert_eq!(
        eval("*", &[m(1.0), s(1.0)]),
        WildcardQuantity::new(1.0, length() * s(1.0).dims, false)
    );
    assert!(eval("^", &[m(1.0), m(1.0)]).violates);
}

/// test/integration/ext/dynamicquantities_units/test_units.jl | "Should error on mismatched units"
#[test]
fn mismatched_units_are_rejected() {
    let err = Dataset::new(randn(11, 50, 0), Array1::zeros(50))
        .with_units(Some(&["m", "1"]), Some("kg"))
        .unwrap_err();
    assert!(err.contains("Number of features"), "{err}");
}

/// test/integration/ext/dynamicquantities_units/test_units.jl | "Should print units"
#[test]
fn units_are_printed() {
    let ops = operators(&["+", "-", "*", "/", "cos", "sin"]);
    let names: Vec<String> = ["x₁", "x₂", "x₃", "x₄", "x₅"].map(String::from).to_vec();
    let print = |data: &Dataset<f64>, src: &str, dimensionless_constants_only: bool| {
        let labels = UnitLabels::new(data.units.as_ref().unwrap(), dimensionless_constants_only);
        let tree: PostfixExpr<f64, Ops, 2> = infix(&ops, src);
        string_tree(
            &tree,
            &ops,
            StringTreeOptions {
                variable_names: Some(&names),
                print_precision: Some(5),
                variable_suffixes: Some(&labels.variables),
                constant_suffix: labels.constant,
                ..Default::default()
            },
        )
    };
    let data = dataset(5, &["m^3", "km/s", "kg", "1", "1"], "kg");
    let tree = "1.0 * (x1 + x2 * x3 * 5.32) - cos(1.5 * (x1 - 0.5))";
    assert_eq!(
        print(&data, tree, false),
        "(1[?] * (x₁[m³] + ((x₂[s⁻¹ km] * x₃[kg]) * 5.32[?]))) - cos(1.5[?] * (x₁[m³] - 0.5[?]))"
    );
    assert_eq!(print(&data, "x5 * 3.2", false), "x₅ * 3.2[?]");
    let data2 = dataset(5, &["1.5", "1.9", "2.0", "3.0", "5.0*m"], "kg");
    assert_eq!(print(&data2, "x5 * 3.2", false), "x₅[5.0 m] * 3.2[?]");
    assert_eq!(print(&data2, "x5 * 3.2", true), "x₅[5.0 m] * 3.2");
}

/// test/integration/ext/dynamicquantities_units/test_units.jl | "Dimensionless constants"
#[test]
fn dimensionless_constants() {
    let ops = operators(&["+", "-", "*", "/", "cos", "sin"]);
    let data = dataset(5, &["m^3", "km/s", "kg", "hr", "1"], "kg");
    // Julia's `cube(x)` and `square(x)` on expression nodes build `x * x * x` and `x * x`.
    let valid = [
        "1.5 * x1 / (x2 * x2 * x2 * (x4 * x4 * x4)) * x3",
        "x3",
        "x3 * x3 / x3 + x3",
    ];
    for src in valid {
        assert!(!violates(&ops, &data, src, true), "{src}");
    }
    let invalid = ["1.5", "1.5 * x1", "x3 - 1.0 * x1"];
    for src in invalid {
        assert!(violates(&ops, &data, src, true), "{src}");
        assert!(!violates(&ops, &data, src, false), "{src} with wildcard constants");
    }
}

/// test/integration/ext/dynamicquantities_units/test_units.jl | "Miscellaneous tests of unit interface"
///
/// The `@maybe_return_call` checks: a valid call returns its value, and a call Julia has no method
/// for (here, a dimensioned value added to a DSL literal) is caught. The MLJ unit-extraction
/// checks are Julia-only.
#[test]
fn unit_interface() {
    let ops = operators(&["+", "f(x) = x + 1"]);
    let speed = Unit::parse("m/s").unwrap().dims;
    let x = WildcardQuantity::new(1.0, speed, true);
    let add = ops.lookup_with_arity("+", 2).unwrap();
    assert_eq!(degn_eval(&ops, add, &[x, x]).value, 2.0);
    let f = ops.lookup_with_arity("f", 1).unwrap();
    assert!(degn_eval(&ops, f, &[WildcardQuantity::new(1.0, speed, false)]).violates);
    assert_eq!(
        degn_eval(&ops, f, &[x]),
        WildcardQuantity::new(2.0, Dimensions::default(), false)
    );
}
