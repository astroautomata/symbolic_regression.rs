//! SymbolicRegression.jl's test/unit/misc/test_tracing.jl.

use dynamic_expressions::expression::{Metadata, PostfixExpr};
use dynamic_expressions::node::PNode;
use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
use ndarray::{Array1, Array2};
use serde_json::{Value, json};

use crate::pop_member::PopMember;
use crate::population::Population;
use crate::tracing::{IterationTrace, StepTrace};
use crate::{Dataset, Options, equation_search};

type Ops = BuiltinOpsF64;

fn options() -> Options<f64, Ops, 2> {
    Options {
        operators: Ops::new().from_names(["+", "*"]).unwrap(),
        use_tracing: true,
        save_to_file: false,
        ..Default::default()
    }
}

fn member(reference: i64, parent: i64, loss: f64, options: &Options<f64, Ops, 2>) -> PopMember<f64, Ops, 2> {
    let leaf = PostfixExpr::new(vec![PNode::Var { feature: 0 }], Vec::new(), Metadata::default());
    let mut m = PopMember::from_expr_with_birth(0, leaf, 1, options);
    m.reference = reference;
    m.parent = parent;
    m.loss = loss;
    m.cost = loss;
    m.complexity = 1;
    m
}

fn record(trace: &IterationTrace<f64, Ops, 2>, options: &Options<f64, Ops, 2>) -> Value {
    let line = trace.to_json(options).replace(":Infinity", ":\"Infinity\"");
    serde_json::from_str(&line).unwrap()
}

fn event_types(entry: &Value) -> Vec<&str> {
    entry["events"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["type"].as_str().unwrap())
        .collect()
}

/// A member that survives from the start of the cycle without an entry in `mutations` gets one
/// copied from the starting population.
fn check_optimization_of_untouched_member(starting: Vec<PopMember<f64, Ops, 2>>) {
    let options = options();
    let mut trace = IterationTrace::start(1, 1, 0, &Population::new(starting));
    let tuned = member(2, 1, 1.0, &options);

    trace.optimization(&tuned, 1, false);

    let record = record(&trace, &options);
    let mutations = record["mutations"].as_object().unwrap();
    assert!(mutations.contains_key("2"));
    let old_entry = &mutations["1"];
    let start = &record["members"][0];
    for key in ["tree", "cost", "loss", "parent"] {
        assert_eq!(old_entry[key], start[key], "{key}");
    }
    assert_eq!(old_entry["parent"], json!(0));
    assert_eq!(event_types(old_entry), ["tuning", "death"]);
    assert_eq!(old_entry["events"][0]["child"], json!(2));
    assert_eq!(old_entry["events"][0]["mutation"], json!({"type": "simplification"}));
}

#[test]
fn trace_optimization_seeds_missing_old_ref() {
    check_optimization_of_untouched_member(vec![member(1, 0, 2.0, &options())]);
}

#[test]
fn trace_optimization_tolerates_a_reference_held_by_several_slots() {
    // One reference occupying several slots, as migration leaves it.
    let old = member(1, 0, 2.0, &options());
    check_optimization_of_untouched_member(vec![old.clone(), old]);
}

#[test]
fn step_traces_record_type_result_and_reason() {
    let options = options();
    let parent = member(1, -1, f64::INFINITY, &options);
    let child = member(2, 1, 1.0, &options);
    let replaced = member(3, -1, 3.0, &options);
    let mut trace = IterationTrace::start(1, 1, 4, &Population::new(vec![parent.clone(), replaced.clone()]));
    let step = StepTrace {
        kind: "mutate_constant",
        outcome: Some(("accept", "pass")),
    };
    trace.mutation_attempts(
        &[(&parent, &child, step), (&parent, &child, StepTrace::identity())],
        Some(&replaced),
        1,
    );

    assert!(trace.to_json(&options).contains("\"loss\":Infinity"));
    let record = record(&trace, &options);
    assert_eq!(record["iteration"], json!(4));
    let events = record["mutations"]["1"]["events"].as_array().unwrap();
    assert_eq!(
        events[0]["mutation"],
        json!({"type": "mutate_constant", "result": "accept", "reason": "pass"})
    );
    assert_eq!(
        events[1]["mutation"],
        json!({"type": "identity", "result": "accept", "reason": "identity"})
    );
    assert_eq!(events[0]["selected"], json!(false));
    assert_eq!(events[1]["selected"], json!(true));
    assert_eq!(event_types(&record["mutations"]["3"]), ["death"]);
    assert_eq!(event_types(&record["mutations"]["2"]), Vec::<&str>::new());
}

#[test]
fn tracing_search_pipeline_smoke_test() {
    let dir = std::env::temp_dir().join(format!("sr_tracing_pipeline_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut rng = fastrand::Rng::with_seed(0);
    let dataset = Dataset::new(
        Array2::from_shape_fn((1, 20), |_| rng.f64() - 0.5),
        Array1::from_shape_fn(20, |_| rng.f64() - 0.5),
    );
    let options = Options {
        population_size: 20,
        populations: 2,
        niterations: 1,
        progress: false,
        tracing_file: dir.join("trace.jsonl"),
        ..options()
    };
    let result = equation_search(&[dataset], &options);
    let lines = std::fs::read_to_string(&options.tracing_file).map(|s| s.lines().count());
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(result.is_ok());
    assert_eq!(lines.unwrap(), 1 + 2);
}
