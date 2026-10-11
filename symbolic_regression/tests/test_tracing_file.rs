//! SymbolicRegression.jl's `use_tracing` file: test/integration/ext/json_tracing/test_tracing.jl,
//! with each record's schema transcribed from SymbolicRegression.jl's src/Tracing.jl.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use dynamic_expressions::OperatorSet;
use ndarray::{Array1, Array2};
use serde_json::{Map, Value};
use symbolic_regression::prelude::*;

const POPULATIONS: usize = 2;
const POPULATION_SIZE: usize = 100;
const NITERATIONS: usize = 5;

/// A fresh directory under the system temp directory, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("sr_tracing_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn dataset() -> Dataset<f32> {
    let n_rows = 1000;
    let mut rng = fastrand::Rng::with_seed(0);
    let x = Array2::from_shape_fn((2, n_rows), |_| 2.0 * (rng.f32() * 2.0 - 1.0));
    let y = Array1::from_shape_fn(n_rows, |i| 3.0 * x[[1, i]].cos() + x[[0, i]] * x[[0, i]] - 2.0);
    Dataset::new(x, y)
}

fn options(tracing_file: &Path, use_tracing: bool) -> Options<f32, BuiltinOpsF32, 2> {
    let mut options = Options::<f32, BuiltinOpsF32, 2> {
        operators: BuiltinOpsF32::new().from_names(["+", "*", "/", "sub", "cos"]).unwrap(),
        use_tracing,
        tracing_file: tracing_file.to_path_buf(),
        populations: POPULATIONS,
        population_size: POPULATION_SIZE,
        maxsize: 20,
        niterations: NITERATIONS,
        seed: 0,
        deterministic: true,
        progress: false,
        ..Default::default()
    };
    let cos = options.operators.set.lookup("cos").unwrap();
    options.operator_complexity_overrides.insert(cos, 2.0);
    options
}

/// Parses a line as JSON, reading the `NaN`, `Infinity`, and `-Infinity` that JSON.jl writes with
/// `allownan=true` as strings.
fn parse_line(line: &str) -> Map<String, Value> {
    let line = line
        .replace(":NaN", ":\"NaN\"")
        .replace(":-Infinity", ":\"-Infinity\"")
        .replace(":Infinity", ":\"Infinity\"");
    match serde_json::from_str(&line) {
        Ok(Value::Object(record)) => record,
        other => panic!("not a JSON object: {other:?}\n{line}"),
    }
}

fn keys(object: &Map<String, Value>) -> HashSet<&str> {
    object.keys().map(String::as_str).collect()
}

fn set<'a>(names: &[&'a str]) -> HashSet<&'a str> {
    names.iter().copied().collect()
}

fn int(v: &Value) -> i64 {
    v.as_i64().unwrap_or_else(|| panic!("not an integer: {v}"))
}

fn is_float(v: &Value) -> bool {
    v.is_number() || ["NaN", "Infinity", "-Infinity"].contains(&v.as_str().unwrap_or(""))
}

fn assert_time(event: &Map<String, Value>) {
    assert!(event["time"].as_f64().unwrap() > 0.0, "{event:?}");
}

/// A step trace: `trace_mutation_type!` sets `type`, and `trace_mutation_result!` sets `result`
/// and `reason` together (src/Tracing.jl:28-46).
fn assert_step_trace(trace: &Map<String, Value>) {
    let k = keys(trace);
    assert!(
        k == set(&["type"]) || k == set(&["type", "result", "reason"]),
        "{trace:?}"
    );
    assert!(!trace["type"].as_str().unwrap().is_empty());
    if let Some(result) = trace.get("result") {
        assert!(["accept", "reject"].contains(&result.as_str().unwrap()), "{trace:?}");
        assert!(!trace["reason"].as_str().unwrap().is_empty());
    }
}

#[test]
fn tracing_file_follows_the_symbolic_regression_jl_schema() {
    let dir = TempDir::new("schema");
    let tracing_file = dir.0.join("pysr_trace.jsonl");
    let options = options(&tracing_file, true);
    equation_search(&[dataset()], &options).unwrap();

    let contents = std::fs::read_to_string(&tracing_file).unwrap();
    assert!(contents.ends_with('\n'));
    let records: Vec<_> = contents.lines().map(parse_line).collect();
    let (search, iterations) = records.split_first().unwrap();

    // The "search" record: src/Tracing.jl:48-57.
    assert_eq!(
        keys(search),
        set(&["schema_version", "record_type", "started_at", "options"])
    );
    assert_eq!(int(&search["schema_version"]), 1);
    assert_eq!(search["record_type"], "search");
    assert!(search["started_at"].as_f64().unwrap() > 0.0);
    assert!(search["options"].as_str().unwrap().contains("Options"));

    // One "iteration" record per population cycle, written as each cycle finishes:
    // src/SymbolicRegression.jl:1147 and src/Tracing.jl:68-81.
    assert_eq!(iterations.len(), POPULATIONS * NITERATIONS);
    let mut iteration_numbers: HashMap<i64, Vec<i64>> = HashMap::new();
    let mut total_entries = 0;
    let mut initial_refs = HashSet::new();
    let mut traced_refs = HashSet::new();
    for record in iterations {
        assert_eq!(
            keys(record),
            set(&[
                "schema_version",
                "record_type",
                "output",
                "population",
                "iteration",
                "started_at",
                "members",
                "mutations"
            ])
        );
        assert_eq!(int(&record["schema_version"]), 1);
        assert_eq!(record["record_type"], "iteration");
        assert_eq!(int(&record["output"]), 1);
        let population = int(&record["population"]);
        let iteration = int(&record["iteration"]);
        assert!((1..=POPULATIONS as i64).contains(&population));
        iteration_numbers.entry(population).or_default().push(iteration);
        assert!(record["started_at"].as_f64().unwrap() > 0.0);

        // Members of the population the cycle starts from: src/Tracing.jl:83-95.
        let members = record["members"].as_array().unwrap();
        assert_eq!(members.len(), POPULATION_SIZE);
        for member in members {
            let member = member.as_object().unwrap();
            assert_eq!(
                keys(member),
                set(&["tree", "loss", "cost", "complexity", "birth", "ref", "parent"])
            );
            assert!(member["tree"].is_string());
            assert!(is_float(&member["loss"]) && is_float(&member["cost"]), "{member:?}");
            assert!(int(&member["complexity"]) >= 1);
            assert!(member["birth"].is_u64());
            assert!(int(&member["ref"]) >= 0);
            if iteration == 0 {
                // The random initial population has no parents: src/Population.jl:81-111.
                assert_eq!(int(&member["parent"]), -1);
                initial_refs.insert(int(&member["ref"]));
            }
        }

        // Members the cycle touched, keyed by `string(ref)`: src/Tracing.jl:97-109, or copied
        // from `members` with their events by `trace_optimization!`: src/Tracing.jl:131-135.
        let mutations = record["mutations"].as_object().unwrap();
        total_entries += mutations.len();
        let entry_ref = |r: i64| {
            mutations
                .get(&r.to_string())
                .unwrap_or_else(|| panic!("ref {r} has no entry"))
                .as_object()
                .unwrap()
        };
        for (key, entry) in mutations {
            let r: i64 = key.parse().unwrap();
            assert!(r >= 0);
            traced_refs.insert(r);
            let entry = entry.as_object().unwrap();
            let k = keys(entry);
            let short = set(&["events", "tree", "cost", "loss", "parent"]);
            let long = set(&["events", "tree", "cost", "loss", "parent", "complexity", "birth", "ref"]);
            assert!(k == short || k == long, "{entry:?}");
            if k == long {
                assert_eq!(int(&entry["ref"]), r);
            }
            assert!(entry["tree"].is_string());
            assert!(is_float(&entry["loss"]) && is_float(&entry["cost"]));
            int(&entry["parent"]);

            for event in entry["events"].as_array().unwrap() {
                let event = event.as_object().unwrap();
                assert_time(event);
                match event["type"].as_str().unwrap() {
                    // src/Tracing.jl:159-166
                    "mutate" => {
                        assert_eq!(keys(event), set(&["type", "time", "child", "selected", "mutation"]));
                        assert!(event["selected"].is_boolean());
                        let mutation = event["mutation"].as_object().unwrap();
                        assert_step_trace(mutation);
                        let child_ref = int(&event["child"]);
                        let child = entry_ref(child_ref);
                        if mutation["type"] == "optimize" {
                            // Constant optimization returns the parent itself: src/Mutate.jl:945-959.
                            assert_eq!(child_ref, r);
                        } else {
                            assert_eq!(int(&child["parent"]), r, "{event:?}");
                        }
                    }
                    // src/Tracing.jl:201-213
                    "crossover" => {
                        assert_eq!(
                            keys(event),
                            set(&["type", "time", "parent1", "parent2", "child1", "child2", "details"])
                        );
                        let details = event["details"].as_object().unwrap();
                        assert_step_trace(details);
                        assert_eq!(details["type"], "subtree_crossover");
                        let (p1, p2) = (int(&event["parent1"]), int(&event["parent2"]));
                        assert!(r == p1 || r == p2);
                        let (c1, c2) = (int(&event["child1"]), int(&event["child2"]));
                        if details["result"] == "accept" {
                            assert_eq!(int(&entry_ref(c1)["parent"]), p1);
                            assert_eq!(int(&entry_ref(c2)["parent"]), p2);
                        } else {
                            // A failed crossover returns its parents: src/Crossover.jl:186-190.
                            assert_eq!((c1, c2), (p1, p2));
                        }
                    }
                    // src/Tracing.jl:124-129, 136
                    "tuning" => {
                        assert_eq!(keys(event), set(&["type", "time", "child", "mutation"]));
                        let mutation = event["mutation"].as_object().unwrap();
                        assert_eq!(keys(mutation), set(&["type"]));
                        assert!(
                            ["simplification", "simplification_and_optimization"]
                                .contains(&mutation["type"].as_str().unwrap())
                        );
                        assert_eq!(int(&entry_ref(int(&event["child"]))["parent"]), r);
                    }
                    // src/Tracing.jl:130, 169-170, 210-215
                    "death" => assert_eq!(keys(event), set(&["type", "time"])),
                    other => panic!("unknown event type {other}"),
                }
            }
        }
        // Every member of the next population is a tuned member of this cycle:
        // src/SingleIteration.jl:112-121.
        let tuned = mutations
            .values()
            .flat_map(|e| e["events"].as_array().unwrap())
            .filter(|e| e["type"] == "tuning")
            .count();
        assert_eq!(tuned, POPULATION_SIZE);
    }
    for numbers in iteration_numbers.values_mut() {
        numbers.sort_unstable();
        assert_eq!(*numbers, (0..numbers.len() as i64).collect::<Vec<_>>());
    }
    assert_eq!(iteration_numbers.len(), POPULATIONS);
    assert!(total_entries > 1000, "{total_entries}");

    // Later populations hold tuned members of earlier cycles, or migrants that cycles traced.
    for record in iterations.iter().filter(|r| int(&r["iteration"]) > 0) {
        for member in record["members"].as_array().unwrap() {
            let r = int(&member["ref"]);
            assert!(
                traced_refs.contains(&r) || initial_refs.contains(&r),
                "ref {r} was never traced"
            );
        }
    }
}

#[test]
fn tracing_changes_no_result_and_disabled_tracing_writes_nothing() {
    let dir = TempDir::new("identical");
    let tracing_file = dir.0.join("trace.jsonl");
    let run = |use_tracing: bool| {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        pool.install(|| equation_search(&[dataset()], &options(&tracing_file, use_tracing)).unwrap())
    };
    let summary = |result: &SearchResult<f32, BuiltinOpsF32, 2>| {
        let opts = options(&tracing_file, false);
        let front: Vec<_> = result.halls_of_fame[0]
            .pareto_front()
            .iter()
            .map(|m| (m.expr.string(&opts, Default::default()), m.loss.to_bits()))
            .collect();
        let pops: Vec<_> = result.populations[0]
            .iter()
            .flat_map(|p| &p.members)
            .map(|m| {
                (
                    m.expr.string(&opts, Default::default()),
                    m.loss.to_bits(),
                    m.cost.to_bits(),
                )
            })
            .collect();
        (front, pops)
    };

    let off = run(false);
    assert!(!tracing_file.exists());
    let on = run(true);
    assert!(tracing_file.exists());
    assert_eq!(summary(&off), summary(&on));
}
