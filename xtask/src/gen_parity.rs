//! `cargo xtask gen-parity`: regenerate tools/parity_tests.toml from the pinned SymbolicRegression.jl test suite.
//!
//! Fields already recorded for an entry (status, pr, rust, excluded, note) are kept; only `line` is
//! refreshed. A new entry is excluded when a rule in EXCLUSIONS matches it; otherwise it starts as
//! "todo" in the feature-area PR that AREA_RULES assigns.

use std::collections::HashMap;
use std::path::Path;
use std::process::ExitCode;

use crate::mapping::{self, Entry, Key};

const LV: &str = "LoopVectorization (Julia-only acceleration)";
const DISTRIBUTED: &str = "Distributed multiprocessing (not supported)";
const NOT_FLOAT: &str = "Rust engine requires T: Float";
const VALUE_TYPES: &str = "custom value types (not supported)";

/// (path prefix, testitem name or None for every testitem under the prefix, value); first match wins.
type Rule<T> = (&'static str, Option<&'static str>, T);

const EXCLUSIONS: &[Rule<&str>] = &[
    ("test/integration/aqua/", None, "Aqua (Julia package quality checks)"),
    ("test/integration/jet/", None, "JET (Julia static analysis)"),
    ("test/integration/ext/mlj/core/", None, "MLJ (Julia-only interface)"),
    ("test/integration/ext/mlj/mti/", None, "MLJ (Julia-only interface)"),
    (
        "test/unit/basic/test_lite_interface.jl",
        None,
        "MLJ (Julia-only interface): MLJModelInterface machine API",
    ),
    (
        "test/unit/misc/test_loss_scale.jl",
        Some("loss_scale in MLJ interface"),
        "MLJ (Julia-only interface)",
    ),
    (
        "test/integration/ad/forwarddiff/",
        None,
        "ForwardDiff (Julia autodiff backend)",
    ),
    ("test/integration/ad/zygote/", None, "Zygote (Julia autodiff backend)"),
    (
        "test/integration/ad/mooncake/",
        None,
        "Mooncake (Julia autodiff backend)",
    ),
    (
        "test/integration/ad/mooncakev1/",
        None,
        "Mooncake (Julia autodiff backend)",
    ),
    (
        "test/integration/ext/loopvectorization/test_operators.jl",
        Some("Safe operators are compatible with ForwardDiff"),
        "ForwardDiff (Julia autodiff backend)",
    ),
    (
        "test/integration/ext/loopvectorization/test_operators.jl",
        Some("Turbo mode matches regular mode"),
        LV,
    ),
    ("test/integration/ext/loopvectorization/test_nan_detection.jl", None, LV),
    ("test/integration/ext/loopvectorization/test_turbo_nan.jl", None, LV),
    (
        "test/integration/ext/loopvectorization/test_template_expression.jl",
        Some("Test eval_context with turbo mode"),
        LV,
    ),
    (
        "test/integration/ext/loopvectorization/test_mixed.jl",
        Some("Search with multiprocessing & batching & multi-output & use_frequency & string-specified parallelism"),
        DISTRIBUTED,
    ),
    (
        "test/integration/ext/loopvectorization/test_mixed.jl",
        Some("Search with multiprocessing & default hyperparameters & Float64 type & turbo"),
        DISTRIBUTED,
    ),
    (
        "test/integration/ext/symbolicutils/",
        None,
        "SymbolicUtils (Julia-only symbolic algebra)",
    ),
    (
        "test/unit/precompile/test_precompilation.jl",
        None,
        "precompilation (Julia-only)",
    ),
    ("test/unit/parallel/", None, DISTRIBUTED),
    (
        "test/unit/operators-core/test_abstract_numbers.jl",
        Some("Abstract numbers"),
        NOT_FLOAT,
    ),
    ("test/unit/operators-core/test_abstract_numbers.jl", None, VALUE_TYPES),
    (
        "test/unit/operators-core/test_complexity.jl",
        Some("Custom complexity mapping"),
        "complexity_mapping (not supported)",
    ),
    ("test/unit/precompile/test_integer_evaluation.jl", None, NOT_FLOAT),
    (
        "test/integration/ext/loopvectorization/test_operators.jl",
        Some("Built-in operators pass validation for complex numbers"),
        NOT_FLOAT,
    ),
    (
        "test/integration/ext/loopvectorization/test_operators.jl",
        Some("Incompatibilities are caught"),
        NOT_FLOAT,
    ),
];

const AREA_RULES: &[Rule<i64>] = &[
    (
        "test/unit/basic/test_pretty_printing.jl",
        Some("pretty print expression"),
        1,
    ),
    (
        "test/unit/basic/test_pretty_printing.jl",
        Some("pretty print vs serialization for comparison operators"),
        1,
    ),
    (
        "test/unit/basic/test_options.jl",
        Some("Test operators parameter conflicts"),
        7,
    ),
    (
        "test/unit/basic/test_options.jl",
        Some("Test operators stored globally"),
        1,
    ),
    (
        "test/unit/basic/test_options.jl",
        Some("Test with_max_degree_from_context"),
        1,
    ),
    (
        "test/unit/basic/test_options.jl",
        Some("Test operator appears in multiple degrees error"),
        1,
    ),
    (
        "test/unit/basic/test_options.jl",
        Some("Test build_constraints with pre-processed vector format"),
        1,
    ),
    ("test/unit/basic/test_print.jl", None, 1),
    ("test/unit/operators-core/", None, 1),
    ("test/unit/constraints/", None, 1),
    ("test/unit/evaluation/test_evaluation.jl", None, 1),
    ("test/unit/misc/test_hash.jl", None, 1),
    ("test/integration/ext/loopvectorization/test_operators.jl", None, 1),
    ("test/unit/evaluation/test_early_stop.jl", None, 2),
    ("test/unit/misc/test_losses.jl", None, 2),
    ("test/unit/misc/test_loss_scale.jl", None, 2),
    ("test/unit/misc/test_custom_objectives.jl", None, 2),
    ("test/unit/expressions/", None, 3),
    ("test/unit/templates/", None, 3),
    (
        "test/integration/ext/loopvectorization/test_template_expression.jl",
        None,
        3,
    ),
    ("test/integration/ext/mlj/templates/", None, 3),
    ("test/integration/ext/mlj/parametric_search/", None, 3),
    ("test/integration/ext/dynamicquantities_units/", None, 4),
    ("test/unit/misc/test_guesses.jl", None, 5),
    ("test/unit/dataset/", None, 5),
    ("test/integration/ext/mlj/", None, 7),
    ("test/", None, 6),
];

fn first_match<T: Copy>(rules: &[Rule<T>], path: &str, name: &str) -> Option<T> {
    rules
        .iter()
        .find(|(prefix, item, _)| path.starts_with(prefix) && item.is_none_or(|item| item == name))
        .map(|&(_, _, value)| value)
}

/// The entry for a testitem the mapping does not list yet.
fn new_entry(file: String, line: i64, name: String) -> Entry {
    let (pr, status, excluded) = match first_match(EXCLUSIONS, &file, &name) {
        Some(reason) => (None, "excluded", Some(reason.to_owned())),
        None => (first_match(AREA_RULES, &file, &name), "todo", None),
    };
    Entry {
        file,
        line,
        name,
        pr,
        status: status.to_owned(),
        rust: vec![],
        excluded,
        note: None,
    }
}

pub fn regenerate(julia_repo: &Path) -> Result<ExitCode, String> {
    let path = mapping::toml_path();
    let mut existing: HashMap<Key, Entry> = HashMap::new();
    if path.exists() {
        existing.extend(mapping::load(&path)?.entries.into_iter().map(|e| (e.key(), e)));
    }
    let entries: Vec<Entry> = mapping::extract(julia_repo)?
        .into_iter()
        .map(
            |(file, line, name)| match existing.remove(&(file.clone(), name.clone())) {
                Some(entry) => Entry {
                    file,
                    line,
                    name,
                    ..entry
                },
                None => new_entry(file, line, name),
            },
        )
        .collect();
    mapping::write(&entries, &path)?;
    println!("wrote {} entries to {}", entries.len(), path.display());
    Ok(ExitCode::SUCCESS)
}
