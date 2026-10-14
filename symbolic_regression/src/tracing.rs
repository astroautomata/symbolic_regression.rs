//! SymbolicRegression.jl's search tracing (`use_tracing`): a JSON Lines file with one `"search"`
//! record when a search starts, then one `"iteration"` record per population cycle with the
//! population the cycle started from and the events of every member the cycle touched.

use std::collections::HashMap;
use std::fmt::{Display, Write as _};
use std::io::Write as _;
use std::path::Path;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{SystemTime, UNIX_EPOCH};

use dynamic_expressions::OperatorSet;
use dynamic_expressions::strings::StringTreeOptions;
use num_traits::Float;
#[cfg(target_arch = "wasm32")]
use web_time::{SystemTime, UNIX_EPOCH};

use crate::expression::Expression;
use crate::options::Options;
use crate::pop_member::PopMember;
use crate::population::Population;

pub const OPEN_TRACING_FILE_ERROR_PREFIX: &str = "cannot open the tracing file ";
pub const WRITE_TRACING_FILE_ERROR_PREFIX: &str = "cannot write the tracing file ";

/// SymbolicRegression.jl's `TRACE_SCHEMA_VERSION`.
const TRACE_SCHEMA_VERSION: u32 = 1;

/// SymbolicRegression.jl's `time()`: seconds since the Unix epoch.
fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock must be after UNIX_EPOCH")
        .as_secs_f64()
}

/// The trace of one mutation or crossover attempt: its type and, once decided, its result and
/// the reason for it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct StepTrace {
    pub(crate) kind: &'static str,
    pub(crate) outcome: Option<(&'static str, &'static str)>,
}

impl StepTrace {
    /// SymbolicRegression.jl's `trace_identity_mutation!`.
    pub(crate) fn identity() -> Self {
        Self {
            kind: "identity",
            outcome: Some(("accept", "identity")),
        }
    }

    fn write_json(&self, out: &mut String) {
        out.push_str("{\"type\":");
        write_json_str(out, self.kind);
        if let Some((result, reason)) = self.outcome {
            out.push_str(",\"result\":");
            write_json_str(out, result);
            out.push_str(",\"reason\":");
            write_json_str(out, reason);
        }
        out.push('}');
    }
}

struct TracedMember<T: Float, Ops, const D: usize> {
    expr: Expression<T, Ops, D>,
    loss: T,
    cost: T,
    complexity: usize,
    birth: u64,
    reference: i64,
    parent: i64,
}

impl<T: Float, Ops, const D: usize> From<&PopMember<T, Ops, D>> for TracedMember<T, Ops, D> {
    fn from(m: &PopMember<T, Ops, D>) -> Self {
        Self {
            expr: m.expr.clone(),
            loss: m.loss,
            cost: m.cost,
            complexity: m.complexity,
            birth: m.birth,
            reference: m.reference,
            parent: m.parent,
        }
    }
}

enum Event {
    Mutate {
        time: f64,
        child: i64,
        selected: bool,
        mutation: StepTrace,
    },
    Crossover {
        time: f64,
        parent1: i64,
        parent2: i64,
        child1: i64,
        child2: i64,
        details: StepTrace,
    },
    Tuning {
        time: f64,
        child: i64,
        mutation_type: &'static str,
    },
    Death {
        time: f64,
    },
}

/// One mutation attempt: the parent, the child it produced, and the attempt's trace.
pub(crate) type TracedStep<'a, T, Ops, const D: usize> =
    (&'a PopMember<T, Ops, D>, &'a PopMember<T, Ops, D>, StepTrace);

/// One entry of an iteration's `"mutations"` map.
struct MemberHistory<T: Float, Ops, const D: usize> {
    member: TracedMember<T, Ops, D>,
    /// Copied from the iteration's starting population, so it also records `complexity`,
    /// `birth`, and `ref`.
    from_start: bool,
    events: Vec<Event>,
}

/// SymbolicRegression.jl's `"iteration"` record for one population cycle.
pub(crate) struct IterationTrace<T: Float, Ops, const D: usize> {
    output: usize,
    population: usize,
    iteration: usize,
    started_at: f64,
    members: Vec<TracedMember<T, Ops, D>>,
    mutations: Vec<MemberHistory<T, Ops, D>>,
    index: HashMap<i64, usize>,
}

impl<T: Float, Ops, const D: usize> IterationTrace<T, Ops, D> {
    /// SymbolicRegression.jl's `trace_iteration_start!`; `output` and `population` count from 1,
    /// `iteration` from 0.
    pub(crate) fn start(output: usize, population: usize, iteration: usize, pop: &Population<T, Ops, D>) -> Self {
        Self {
            output,
            population,
            iteration,
            started_at: now(),
            members: pop.members.iter().map(TracedMember::from).collect(),
            mutations: Vec::new(),
            index: HashMap::new(),
        }
    }

    /// SymbolicRegression.jl's `_trace_member!`: adds `member`'s entry unless its `ref` has one.
    fn record(&mut self, member: &PopMember<T, Ops, D>) {
        if !self.index.contains_key(&member.reference) {
            self.index.insert(member.reference, self.mutations.len());
            self.mutations.push(MemberHistory {
                member: member.into(),
                from_start: false,
                events: Vec::new(),
            });
        }
    }

    fn push_event(&mut self, reference: i64, event: Event) {
        let i = self.index[&reference];
        self.mutations[i].events.push(event);
    }

    /// SymbolicRegression.jl's `trace_mutation_attempts!`: each `(parent, child, trace)` step
    /// adds a `"mutate"` event to its parent, marking step `selected` (counting from 0) as the
    /// one the population kept; `replaced` is the member the child replaces, if any, which dies.
    pub(crate) fn mutation_attempts(
        &mut self,
        steps: &[TracedStep<'_, T, Ops, D>],
        replaced: Option<&PopMember<T, Ops, D>>,
        selected: usize,
    ) {
        if let Some(m) = replaced {
            self.record(m);
        }
        for &(parent, child, _) in steps {
            self.record(parent);
            self.record(child);
        }
        for (i, &(parent, child, mutation)) in steps.iter().enumerate() {
            let event = Event::Mutate {
                time: now(),
                child: child.reference,
                selected: i == selected,
                mutation,
            };
            self.push_event(parent.reference, event);
        }
        if let Some(m) = replaced {
            self.push_event(m.reference, Event::Death { time: now() });
        }
    }

    /// SymbolicRegression.jl's `trace_crossover!`: both parents get the `"crossover"` event, and
    /// the two members the children replace die.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn crossover(
        &mut self,
        parent1: &PopMember<T, Ops, D>,
        parent2: &PopMember<T, Ops, D>,
        child1: &PopMember<T, Ops, D>,
        child2: &PopMember<T, Ops, D>,
        replaced1: &PopMember<T, Ops, D>,
        replaced2: &PopMember<T, Ops, D>,
        details: StepTrace,
    ) {
        for m in [parent1, parent2, child1, child2, replaced1, replaced2] {
            self.record(m);
        }
        let time = now();
        let event = || Event::Crossover {
            time,
            parent1: parent1.reference,
            parent2: parent2.reference,
            child1: child1.reference,
            child2: child2.reference,
            details,
        };
        self.push_event(parent1.reference, event());
        self.push_event(parent2.reference, event());
        self.push_event(replaced1.reference, Event::Death { time: now() });
        self.push_event(replaced2.reference, Event::Death { time: now() });
    }

    /// SymbolicRegression.jl's `trace_optimization!`: `member` took a new `ref` after the
    /// population was simplified and, if `optimized`, constant-optimized. Its former `ref`
    /// records a `"tuning"` event to the new one, then dies; a member untouched since the start
    /// of the cycle gets its entry from the starting population.
    pub(crate) fn optimization(&mut self, member: &PopMember<T, Ops, D>, old_ref: i64, optimized: bool) {
        self.record(member);
        if !self.index.contains_key(&old_ref) {
            let start = self
                .members
                .iter()
                .find(|m| m.reference == old_ref)
                .expect("an optimized member is in the population the cycle started from or one it created");
            let member = TracedMember {
                expr: start.expr.clone(),
                ..*start
            };
            self.index.insert(old_ref, self.mutations.len());
            self.mutations.push(MemberHistory {
                member,
                from_start: true,
                events: Vec::new(),
            });
        }
        let mutation_type = if optimized {
            "simplification_and_optimization"
        } else {
            "simplification"
        };
        let tuning = Event::Tuning {
            time: now(),
            child: member.reference,
            mutation_type,
        };
        self.push_event(old_ref, tuning);
        self.push_event(old_ref, Event::Death { time: now() });
    }

    /// The record as one JSON line, without the newline.
    pub(crate) fn to_json(&self, options: &Options<T, Ops, D>) -> String
    where
        T: Display,
        Ops: OperatorSet<T = T>,
    {
        let mut out = String::new();
        write!(
            out,
            "{{\"schema_version\":{TRACE_SCHEMA_VERSION},\"record_type\":\"iteration\",\"output\":{},\"population\":{},\"iteration\":{},\"started_at\":",
            self.output, self.population, self.iteration
        )
        .unwrap();
        write_json_float(&mut out, self.started_at);
        out.push_str(",\"members\":[");
        for (i, m) in self.members.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push('{');
            m.write_fields(&mut out, options, true);
            out.push('}');
        }
        out.push_str("],\"mutations\":{");
        for (i, h) in self.mutations.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            write!(out, "\"{}\":{{", h.member.reference).unwrap();
            h.member.write_fields(&mut out, options, h.from_start);
            out.push_str(",\"events\":[");
            for (k, e) in h.events.iter().enumerate() {
                if k > 0 {
                    out.push(',');
                }
                e.write_json(&mut out);
            }
            out.push_str("]}");
        }
        out.push_str("}}");
        out
    }
}

impl<T: Float, Ops, const D: usize> TracedMember<T, Ops, D> {
    /// `tree`, `loss`, `cost`, and `parent`, with `complexity`, `birth`, and `ref` when `full`, as
    /// SymbolicRegression.jl's `_trace_population` writes them.
    fn write_fields(&self, out: &mut String, options: &Options<T, Ops, D>, full: bool)
    where
        T: Display,
        Ops: OperatorSet<T = T>,
    {
        out.push_str("\"tree\":");
        write_json_str(out, &self.expr.string(options, StringTreeOptions::default()));
        out.push_str(",\"loss\":");
        write_json_float(out, self.loss);
        out.push_str(",\"cost\":");
        write_json_float(out, self.cost);
        if full {
            write!(
                out,
                ",\"complexity\":{},\"birth\":{},\"ref\":{}",
                self.complexity, self.birth, self.reference
            )
            .unwrap();
        }
        write!(out, ",\"parent\":{}", self.parent).unwrap();
    }
}

impl Event {
    fn write_json(&self, out: &mut String) {
        match *self {
            Event::Mutate {
                time,
                child,
                selected,
                mutation,
            } => {
                out.push_str("{\"type\":\"mutate\",\"time\":");
                write_json_float(out, time);
                write!(out, ",\"child\":{child},\"selected\":{selected},\"mutation\":").unwrap();
                mutation.write_json(out);
            }
            Event::Crossover {
                time,
                parent1,
                parent2,
                child1,
                child2,
                details,
            } => {
                out.push_str("{\"type\":\"crossover\",\"time\":");
                write_json_float(out, time);
                write!(
                    out,
                    ",\"parent1\":{parent1},\"parent2\":{parent2},\"child1\":{child1},\"child2\":{child2},\"details\":"
                )
                .unwrap();
                details.write_json(out);
            }
            Event::Tuning {
                time,
                child,
                mutation_type,
            } => {
                out.push_str("{\"type\":\"tuning\",\"time\":");
                write_json_float(out, time);
                write!(out, ",\"child\":{child},\"mutation\":{{\"type\":\"{mutation_type}\"}}").unwrap();
            }
            Event::Death { time } => {
                out.push_str("{\"type\":\"death\",\"time\":");
                write_json_float(out, time);
            }
        }
        out.push('}');
    }
}

/// SymbolicRegression.jl's `"search"` record, which `initialize_trace!` writes when a search
/// starts.
pub(crate) fn search_record<T: Float, Ops, const D: usize>(options: &Options<T, Ops, D>) -> String {
    let mut out = format!("{{\"schema_version\":{TRACE_SCHEMA_VERSION},\"record_type\":\"search\",\"started_at\":");
    write_json_float(&mut out, now());
    out.push_str(",\"options\":");
    write_json_str(&mut out, &options.describe());
    out.push('}');
    out
}

/// Writes `record` and a newline to `path`, appending or, as when a search starts, truncating.
pub(crate) fn write_record(path: &Path, record: &str, append: bool) -> Result<(), String> {
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(append)
        .truncate(!append)
        .open(path)
        .map_err(|e| format!("{OPEN_TRACING_FILE_ERROR_PREFIX}{}: {e}", path.display()))?;
    writeln!(file, "{record}").map_err(|e| format!("{WRITE_TRACING_FILE_ERROR_PREFIX}{}: {e}", path.display()))
}

fn write_json_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if u32::from(c) < 0x20 => write!(out, "\\u{:04x}", u32::from(c)).unwrap(),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// A number as JSON.jl writes it with `allownan=true`: non-finite values as `NaN`, `Infinity`,
/// and `-Infinity`.
fn write_json_float<T: Float + Display>(out: &mut String, x: T) {
    if x.is_nan() {
        out.push_str("NaN");
    } else if x.is_infinite() {
        out.push_str(if x > T::zero() { "Infinity" } else { "-Infinity" });
    } else {
        write!(out, "{x}").unwrap();
    }
}
