//! Unit strings parse as DynamicQuantities 1.13.0 parses them, and dimensions propagate through
//! each operator as SymbolicRegression.jl's `degn_eval` propagates them.

use dynamic_expressions::dsl::parse_operator_tokens;
use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
use dynamic_expressions::{OpId, OperatorSet};

use crate::dimensional_analysis::{WildcardQuantity, degn_eval};
use crate::units::{DENOM, Dimensions, Unit};

/// Exponents in whole units, in DynamicQuantities' order (m, kg, s, A, K, cd, mol).
fn dims(e: [i32; 7]) -> Dimensions {
    Dimensions(e.map(|e| e * DENOM))
}

/// Every unit string in SymbolicRegression.jl's (f97764fb) and PySR's (c15a9eb) tests and docs,
/// with the scale, SI dimensions, and symbolic label (`format_dimensions(sym_uparse(s))`) that
/// DynamicQuantities 1.13.0 gives. The units are registered in DynamicQuantities' `src/units.jl`:
/// `m`, `g`, `s`, `A` at 45-48, the prefixed `km` at 54, `kg` at 55, `hr` at 179; and
/// `src/constants.jl`: `G` at 91, `M_sun` at 150, `R_earth` at 152. Derived dimensions follow
/// `*`, `/`, and `^` (`src/math.jl` 4-36, 96-140).
#[rustfmt::skip]
const DOCUMENTED: &[(&str, &str, f64, [i32; 7], &str)] = &[
    ("SR.jl test/integration/ext/dynamicquantities_units/test_units.jl:23", "m", 1.0, [1, 0, 0, 0, 0, 0, 0], "[m]"),
    ("SR.jl test/integration/ext/dynamicquantities_units/test_units.jl:23", "1", 1.0, [0, 0, 0, 0, 0, 0, 0], ""),
    ("SR.jl test/integration/ext/dynamicquantities_units/test_units.jl:23", "kg", 1.0, [0, 1, 0, 0, 0, 0, 0], "[kg]"),
    ("SR.jl test/integration/ext/dynamicquantities_units/test_units.jl:36", "", 1.0, [0, 0, 0, 0, 0, 0, 0], ""),
    ("SR.jl test/integration/ext/dynamicquantities_units/test_units.jl:156", "m^2", 1.0, [2, 0, 0, 0, 0, 0, 0], "[m²]"),
    ("SR.jl test/integration/ext/dynamicquantities_units/test_units.jl:157", "m^3", 1.0, [3, 0, 0, 0, 0, 0, 0], "[m³]"),
    ("SR.jl test/integration/ext/dynamicquantities_units/test_units.jl:159", "s", 1.0, [0, 0, 1, 0, 0, 0, 0], "[s]"),
    ("SR.jl test/integration/ext/dynamicquantities_units/test_units.jl:163", "m*s", 1.0, [1, 0, 1, 0, 0, 0, 0], "[m s]"),
    ("SR.jl test/integration/ext/dynamicquantities_units/test_units.jl:240", "kg^3", 1.0, [0, 3, 0, 0, 0, 0, 0], "[kg³]"),
    ("SR.jl test/integration/ext/dynamicquantities_units/test_units.jl:240", "kg^2", 1.0, [0, 2, 0, 0, 0, 0, 0], "[kg²]"),
    ("SR.jl test/integration/ext/dynamicquantities_units/test_units.jl:332", "km/s", 1000.0, [1, 0, -1, 0, 0, 0, 0], "[s⁻¹ km]"),
    ("SR.jl test/integration/ext/dynamicquantities_units/test_units.jl:404", "hr", 3600.0, [0, 0, 1, 0, 0, 0, 0], "[hr]"),
    ("SR.jl test/integration/ext/dynamicquantities_units/test_units.jl:446", "m/s", 1.0, [1, 0, -1, 0, 0, 0, 0], "[m s⁻¹]"),
    ("SR.jl docs/src/examples.md:168", "Constants.M_sun", 1.98840987e30, [0, 1, 0, 0, 0, 0, 0], "[M_sun]"),
    ("SR.jl docs/src/examples.md:170", "Constants.R_earth", 6.3781e6, [1, 0, 0, 0, 0, 0, 0], "[R_earth]"),
    ("SR.jl docs/src/examples.md:172", "Constants.G", 6.6743e-11, [3, -1, -2, 0, 0, 0, 0], "[G]"),
    ("PySR docs/src/examples/physics.md:67", "kg * m / s^2", 1.0, [1, 1, -2, 0, 0, 0, 0], "[m s⁻² kg]"),
    ("PySR docs/src/examples/physics.md:183", "s^-1", 1.0, [0, 0, -1, 0, 0, 0, 0], "[s⁻¹]"),
    ("PySR docs/src/examples/physics.md:183", "m^-1*s^-1", 1.0, [-1, 0, -1, 0, 0, 0, 0], "[m⁻¹ s⁻¹]"),
    ("PySR docs/src/examples/physics.md:183", "m*s^-2", 1.0, [1, 0, -2, 0, 0, 0, 0], "[m s⁻²]"),
    ("PySR pysr/test/test_main.py:2825", "s^2", 1.0, [0, 0, 2, 0, 0, 0, 0], "[s²]"),
    ("PySR pysr/test/test_main.py:2826", "km", 1000.0, [1, 0, 0, 0, 0, 0, 0], "[km]"),
    ("PySR pysr/test/test_main.py:2870", "A", 1.0, [0, 0, 0, 1, 0, 0, 0], "[A]"),
    ("PySR pysr/test/test_main.py:2871", "m*A", 1.0, [1, 0, 0, 1, 0, 0, 0], "[m A]"),
];

/// Strings that repeat a documented one: SR.jl test_units.jl:111, 216, 323, 332, 404 and
/// src/SymbolicRegression.jl:556; PySR physics.md:66 and test_main.py:2786, 2808, 2813.
const REPEATED: &[&str] = &[
    "kg",
    "1",
    "m",
    "m^2",
    "m^3",
    "Constants.M_sun",
    "Constants.R_earth",
    "m/s",
    "s",
    "",
];

/// Further `uparse`/`sym_uparse` results from DynamicQuantities 1.13.0, for the grammar:
/// rational and float exponents, numeric factors, constants that share a unit's name, Unicode
/// units, and Julia's float printing in labels.
#[rustfmt::skip]
const GRAMMAR: &[(&str, f64, [i32; 7], &str)] = &[
    ("m^0.5", 1.0, [12600, 0, 0, 0, 0, 0, 0], "[m¹ᐟ²]"),
    ("m^(1//2)", 1.0, [12600, 0, 0, 0, 0, 0, 0], "[m¹ᐟ²]"),
    ("m^(1/3)", 1.0, [8400, 0, 0, 0, 0, 0, 0], "[m¹ᐟ³]"),
    ("m^(-1//3)", 1.0, [-8400, 0, 0, 0, 0, 0, 0], "[m⁻¹ᐟ³]"),
    ("m^(1/7)", 1.0, [3600, 0, 0, 0, 0, 0, 0], "[m¹ᐟ⁷]"),
    ("m^-1.5", 1.0, [-37800, 0, 0, 0, 0, 0, 0], "[m⁻³ᐟ²]"),
    ("kg^(2/3)", 1.0, [0, 16800, 0, 0, 0, 0, 0], "[kg²ᐟ³]"),
    ("km^0.5", 31.622776601683793, [12600, 0, 0, 0, 0, 0, 0], "[km¹ᐟ²]"),
    ("(m/s)^2", 1.0, [50400, 0, -50400, 0, 0, 0, 0], "[m² s⁻²]"),
    ("2*km", 2000.0, [25200, 0, 0, 0, 0, 0, 0], "[2.0 km]"),
    ("2m", 2.0, [25200, 0, 0, 0, 0, 0, 0], "[2.0 m]"),
    ("-m", -1.0, [25200, 0, 0, 0, 0, 0, 0], "[-1.0 m]"),
    ("+m", 1.0, [25200, 0, 0, 0, 0, 0, 0], "[m]"),
    ("m - m", 0.0, [25200, 0, 0, 0, 0, 0, 0], "[0.0 m]"),
    ("1000", 1000.0, [0, 0, 0, 0, 0, 0, 0], "[1000.0 ]"),
    ("1.5", 1.5, [0, 0, 0, 0, 0, 0, 0], "[1.5 ]"),
    ("1e-5*m", 1.0e-5, [25200, 0, 0, 0, 0, 0, 0], "[1.0e-5 m]"),
    ("1e16*m", 1.0e16, [25200, 0, 0, 0, 0, 0, 0], "[1.0e16 m]"),
    ("100000*s", 100000.0, [0, 0, 25200, 0, 0, 0, 0], "[100000.0 s]"),
    ("Constants.h", 6.62607015e-34, [50400, 25200, -25200, 0, 0, 0, 0], "[h_constant]"),
    ("h", 3600.0, [0, 0, 25200, 0, 0, 0, 0], "[h]"),
    ("Constants.c^2", 8.987551787368176e16, [50400, 0, -50400, 0, 0, 0, 0], "[c²]"),
    ("J/s", 1.0, [50400, 25200, -75600, 0, 0, 0, 0], "[s⁻¹ J]"),
    ("N", 1.0, [25200, 25200, -50400, 0, 0, 0, 0], "[N]"),
    ("μm", 1.0e-6, [25200, 0, 0, 0, 0, 0, 0], "[μm]"),
    ("us", 1.0e-6, [0, 0, 25200, 0, 0, 0, 0], "[us]"),
    ("Ω", 1.0, [50400, 25200, -75600, -50400, 0, 0, 0], "[Ω]"),
    ("deg", 0.017453292519943295, [0, 0, 0, 0, 0, 0, 0], "[deg]"),
    ("mol/L", 999.9999999999998, [-75600, 0, 0, 0, 0, 0, 25200], "[mol L⁻¹]"),
    ("1/s", 1.0, [0, 0, -25200, 0, 0, 0, 0], "[s⁻¹]"),
];

fn assert_unit(src: &str, scale: f64, want: Dimensions, label: &str, origin: &str) {
    let unit = Unit::parse(src).unwrap_or_else(|e| panic!("{origin}: {e}"));
    assert!(
        (unit.scale - scale).abs() <= 1e-15 * scale.abs(),
        "{origin}: `{src}` scale {} != {scale}",
        unit.scale
    );
    assert_eq!(unit.dims, want, "{origin}: `{src}` dimensions");
    assert_eq!(unit.label(), label, "{origin}: `{src}` label");
}

#[test]
fn documented_unit_strings_match_dynamic_quantities() {
    for &(origin, src, scale, exponents, label) in DOCUMENTED {
        assert_unit(src, scale, dims(exponents), label, origin);
    }
    for src in REPEATED {
        assert!(
            DOCUMENTED.iter().any(|row| row.1 == *src),
            "`{src}` has a documented row"
        );
    }
}

#[test]
fn unit_grammar_matches_dynamic_quantities() {
    for &(src, scale, numerators, label) in GRAMMAR {
        assert_unit(src, scale, Dimensions(numerators), label, "uparse");
    }
}

#[test]
fn bad_unit_strings_name_the_problem() {
    for (src, message) in [
        ("foo", "unknown unit `foo`"),
        ("c", "`c` is a constant, not a unit; write `Constants.c`"),
        ("Constants.foo", "unknown constant `Constants.foo`"),
        ("m^m", "an exponent must be dimensionless"),
        ("m + s", "`+` and `-` need matching units"),
        ("m s", "unexpected `s`"),
        ("m^", "unexpected end"),
        ("m @ s", "unexpected `@`"),
    ] {
        assert_eq!(
            Unit::parse(src).unwrap_err(),
            format!("cannot parse unit `{src}`: {message}")
        );
    }
}

/// One `WildcardQuantity` from the table: `value unit wildcard`.
fn quantity(spec: &str) -> WildcardQuantity<f64> {
    let [value, unit, wildcard] = spec.split(' ').collect::<Vec<_>>()[..] else {
        panic!("bad argument `{spec}`");
    };
    let unit = Unit::parse(unit).unwrap();
    WildcardQuantity::new(value.parse::<f64>().unwrap() * unit.scale, unit.dims, wildcard == "1")
}

/// Every row of `data/units_operator_table.txt`, generated from SymbolicRegression.jl f97764fb's
/// `degn_eval`: each built-in operator the port shares with it, and DSL
/// operators, which SymbolicRegression.jl receives from PySR as Julia functions. Rows where Julia
/// raises an error are skipped: there the DSL returns NaN (`(-2.0)^0.5`), and a call that strips a
/// wildcard but returns a plain number counts as having no method (`ifelse(x > 0, x, y)`).
#[test]
fn operator_propagation_matches_symbolic_regression_jl() {
    let table = include_str!("data/units_operator_table.txt");
    let definitions: Vec<&str> = table
        .lines()
        .map(|l| l.split(" | ").next().unwrap())
        .filter(|op| op.contains('='))
        .collect();
    let mut tokens: Vec<&str> = definitions.clone();
    tokens.dedup();
    let (custom, _) = parse_operator_tokens::<f64, _>(tokens).unwrap();
    let ops = BuiltinOpsF64::new().with_custom(custom);
    let (mut checked, mut skipped) = (0, 0);
    for line in table.lines() {
        let [op, args, want] = line.split(" | ").collect::<Vec<_>>()[..] else {
            panic!("bad row `{line}`");
        };
        if want.starts_with("throws") {
            skipped += 1;
            continue;
        }
        let args: Vec<WildcardQuantity<f64>> = args.split(" ; ").map(quantity).collect();
        let arity = args.len() as u8;
        let id: OpId = if op.contains('=') {
            ops.lookup_with_arity(op.split('(').next().unwrap(), arity).unwrap()
        } else {
            ops.lookup_with_arity(op, arity)
                .unwrap_or_else(|e| panic!("`{op}`: {e:?}"))
        };
        let got = degn_eval(&ops, id, &args);
        let [violates, wildcard, dimensions, value] = want.split(' ').collect::<Vec<_>>()[..] else {
            panic!("bad result in `{line}`");
        };
        assert_eq!(got.violates, violates == "1", "{line}: {got:?}");
        if !got.violates {
            let numerators: Vec<i32> = dimensions.split(',').map(|d| d.parse().unwrap()).collect();
            assert_eq!(got.dims.0.to_vec(), numerators, "{line}: {got:?}");
            assert_eq!(got.wildcard, wildcard == "1", "{line}: {got:?}");
            let want: f64 = value.parse().unwrap();
            let close =
                (got.value.is_nan() && want.is_nan()) || (got.value - want).abs() <= 1e-12 * want.abs().max(1.0);
            assert!(close, "{line}: {got:?}");
        }
        checked += 1;
    }
    assert_eq!((checked, skipped), (1645, 15));
}

/// Julia checks `Rational` arithmetic and the conversion of exponents to `FixedRational{Int32}`,
/// so an overflow throws; here it is an error naming the string, never a panic or abort.
#[test]
fn overflowing_unit_strings_are_errors() {
    let deep = format!("{}m{}", "(".repeat(100_000), ")".repeat(100_000));
    let negations = format!("{}m", "-".repeat(100_000));
    let exponents = format!("m{}", "^m".repeat(100_000));
    for src in [
        "1//(9223372036854775807+1)*m",
        "(9223372036854775807//2)*(9223372036854775807//3)*m",
        "((9223372036854775807//2)+(9223372036854775807//3))*m",
        "-(-9223372036854775807//1-1//1)*m",
        "(2//3)^100*m",
        "m^(9223372036854775807//2)",
        "m^100000",
        "m^9223372036854775807",
        "(m^20000)^20000",
        "m^1e9",
        &deep,
        &negations,
        &exponents,
    ] {
        let err = Unit::parse(src).expect_err(&src[..src.len().min(40)]);
        assert!(
            err.starts_with(&format!("cannot parse unit `{src}`: ")),
            "{}",
            &err[..err.len().min(200)]
        );
    }
}

/// DynamicQuantities rounds `x * 25200` to the nearest `Int32` (fixed_rational.jl:118).
#[test]
fn float_exponents_round_to_the_fixed_denominator() {
    for (src, numerator) in [("m^0.33333", 8400), ("m^0.12345", 3111), ("m^(1/7)", 3600)] {
        assert_eq!(Unit::parse(src).unwrap().dims.0[0], numerator, "`{src}`");
    }
    for src in ["m^1e309", "m^(1e309-1e309)"] {
        assert!(Unit::parse(src).unwrap_err().ends_with("is not finite"), "`{src}`");
    }
}

/// Julia normalizes identifiers to NFC and then applies `Base.Unicode.julia_chartransform`.
#[test]
fn unit_names_normalize_as_julia_identifiers() {
    for (written, canonical) in [
        ("\u{2126}", "Ω"),
        ("m\u{2126}", "mΩ"),
        ("\u{00B5}m", "μm"),
        ("\u{212B}", "Å"),
        ("A\u{030A}", "Å"),
        ("\u{212A}", "K"),
        ("m^\u{2212}1", "m^-1"),
    ] {
        assert_eq!(
            Unit::parse(written).unwrap(),
            Unit::parse(canonical).unwrap(),
            "`{written}`"
        );
    }
}

/// The functions DynamicQuantities defines on quantities (`src/math.jl`), evaluated as its
/// parsers evaluate a call; any other function is an error naming it.
#[test]
fn unit_strings_call_dynamic_quantities_functions() {
    let sqrt_milliohm = Unit::parse("sqrt(mΩ)").unwrap();
    assert!((0.9 * sqrt_milliohm.scale - 0.02846049894151541).abs() < 1e-15);
    assert_eq!(sqrt_milliohm.dims, Dimensions([25200, 12600, -37800, -25200, 0, 0, 0]));
    assert_eq!(sqrt_milliohm.label(), "[mΩ¹ᐟ²]");
    for (src, scale, numerators, label) in [
        ("cbrt(m^3)", 1.0, [25200, 0, 0, 0, 0, 0, 0], "[m]"),
        ("inv(s)", 1.0, [0, 0, -25200, 0, 0, 0, 0], "[s⁻¹]"),
        ("abs2(km)", 1.0e6, [50400, 0, 0, 0, 0, 0, 0], "[km²]"),
        ("abs(-2)*m", 2.0, [25200, 0, 0, 0, 0, 0, 0], "[2.0 m]"),
        (
            "exp(m/m)*m",
            std::f64::consts::E,
            [25200, 0, 0, 0, 0, 0, 0],
            "[2.718281828459045 m]",
        ),
        (
            "atan(1, 1)*m",
            std::f64::consts::FRAC_PI_4,
            [25200, 0, 0, 0, 0, 0, 0],
            "[0.7853981633974483 m]",
        ),
        ("m^sqrt(4)", 1.0, [50400, 0, 0, 0, 0, 0, 0], "[m²]"),
        ("round(2.5)*m", 2.0, [25200, 0, 0, 0, 0, 0, 0], "[2.0 m]"),
        ("mod(7, 3)*m", 1.0, [25200, 0, 0, 0, 0, 0, 0], "[m]"),
        ("div(km, s)", 1000.0, [25200, 0, -25200, 0, 0, 0, 0], "[s⁻¹ km]"),
        ("factorial(4)*m", 24.0, [25200, 0, 0, 0, 0, 0, 0], "[24.0 m]"),
    ] {
        assert_unit(src, scale, Dimensions(numerators), label, "math.jl");
    }
    for (src, message) in [
        ("exp(m)", "`exp` needs a dimensionless argument"),
        ("sqrt(-1)", "`sqrt` is not defined at -1"),
        ("sqrt(m, s)", "`sqrt` takes 1 argument, got 2"),
        ("atan(m, s)", "`atan` needs arguments with the same units"),
        ("factorial(21)", "integer overflow"),
        (
            "gcd(4, 6)",
            "`gcd` is not a function DynamicQuantities defines on quantities",
        ),
        ("m(2)", "`m` is not a function DynamicQuantities defines on quantities"),
    ] {
        assert_eq!(
            Unit::parse(src).unwrap_err(),
            format!("cannot parse unit `{src}`: {message}")
        );
    }
}
