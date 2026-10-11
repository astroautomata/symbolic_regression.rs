use std::io::Cursor;

use csv::ReaderBuilder;
use dynamic_expressions::dsl::{CustomOps, TemplateStructure, parse_operator_tokens};
use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
use dynamic_expressions::strings::StringTreeOptions;
use dynamic_expressions::utils::ZipEq;
use dynamic_expressions::{EvalOptions, OpId, OperatorSet, Operators, eval_tree_array};
use fastrand::Rng;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use serde_wasm_bindgen::{from_value, to_value};
use symbolic_regression::{Dataset, Options, SearchEngine, WasmOptionsShim, choose_best, parse_early_stop, parse_loss};
use wasm_bindgen::prelude::*;

type WasmOps = BuiltinOpsF64<CustomOps<f64>>;

/// Operator tokens from the web UI, one list per arity. A token is a built-in name or symbol
/// (`"cos"`, `"+"`) or a custom operator definition (`"inv(x) = 1/x"`); each operator must take as
/// many arguments as its list says.
fn operators_from_tokens(
    unary: &[String],
    binary: &[String],
    ternary: &[String],
) -> Result<Operators<WasmOps, 3>, String> {
    let tokens: Vec<(u8, &str)> = (1u8..)
        .zip([unary, binary, ternary])
        .flat_map(|(arity, list)| {
            list.iter().map(move |s| match (arity, s.as_str()) {
                (1, "-") => (arity, "neg"),
                (2, "-") => (arity, "sub"),
                (_, s) => (arity, s),
            })
        })
        .collect();
    let (custom, names) = parse_operator_tokens::<f64, _>(tokens.iter().map(|&(_, t)| t)).map_err(|e| e.to_string())?;
    let operators = BuiltinOpsF64::new()
        .with_custom(custom)
        .from_names(&names)
        .map_err(|e| e.to_string())?;
    for (&(arity, _), name) in tokens.iter().zip(&names) {
        let op = operators.set.lookup(name).expect("from_names resolved every name");
        if op.arity != arity {
            let list = ["unary", "binary", "ternary"][usize::from(arity) - 1];
            let plural = if op.arity == 1 { "" } else { "s" };
            return Err(format!(
                "operator `{name}` takes {} argument{plural}, but it is in the {list} operator list",
                op.arity
            ));
        }
    }
    Ok(operators)
}

type FullDatasetParts = (Dataset<f64>, Vec<String>, Option<Array1<f64>>);
type SelectedRows = (Array2<f64>, Array1<f64>, Option<Array1<f64>>);

#[cfg(feature = "panic-hook")]
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn init_thread_pool(num_threads: usize) -> Result<(), JsValue> {
    wasm_bindgen_futures::JsFuture::from(wasm_bindgen_rayon::init_thread_pool(num_threads))
        .await
        .map(|_| ())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WasmOpInfo {
    pub arity: u8,
    pub name: String,
    pub display: String,
    pub infix: Option<String>,
    pub commutative: bool,
    pub associative: bool,
    pub complexity: u16,
}

#[wasm_bindgen]
pub fn builtin_operator_registry() -> Result<JsValue, JsValue> {
    let opset = BuiltinOpsF64::new();
    let mut ops = Vec::new();
    for arity in 1..=opset.max_arity() {
        for id in opset.ops_with_arity(arity) {
            let op = OpId { arity, id };
            ops.push(WasmOpInfo {
                arity,
                name: opset.name(op).to_string(),
                display: opset.display(op).to_string(),
                infix: opset.infix(op).map(|s| s.to_string()),
                commutative: opset.commutative(op),
                associative: opset.associative(op),
                complexity: opset.complexity(op),
            });
        }
    }
    to_value(&ops).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn default_search_options() -> Result<JsValue, JsValue> {
    let opts = WasmSearchOptions::default();
    to_value(&opts).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn default_core_options() -> Result<JsValue, JsValue> {
    to_value(&WasmOptionsShim::default()).map_err(|e| JsValue::from_str(&e.to_string()))
}

fn shuffle<T>(rng: &mut Rng, values: &mut [T]) {
    if values.len() <= 1 {
        return;
    }
    for i in (1..values.len()).rev() {
        let j = rng.usize(0..=i);
        values.swap(i, j);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct WasmSearchOptions {
    // Dataset / UI plumbing
    pub has_headers: bool,
    pub x_columns: Option<Vec<usize>>,
    pub y_column: Option<usize>,
    pub weights_column: Option<usize>,
    pub validation_fraction: f64,

    /// PySR's `elementwise_loss`: a built-in such as `"L1DistLoss()"` or `"HuberLoss(1.5)"`, or a
    /// definition such as `"f(x, y) = abs(x - y)^1.5"` (with a third weight argument when the data
    /// has a weights column).
    pub loss: String,
    /// PySR's `early_stop_condition`: a loss threshold such as `"1e-6"`, or a definition taking
    /// `(loss, complexity)` such as `"f(loss, complexity) = loss < 1e-6 && complexity < 10"`.
    pub early_stop_condition: Option<String>,
    /// A template expression, as PySR's `TemplateExpressionSpec`; `None` searches plain expressions.
    pub template: Option<WasmTemplateOptions>,

    #[serde(flatten)]
    pub core: WasmOptionsShim,
}

/// A template expression: the `combine` formula calls the subexpressions `expressions` on the
/// features and may index the parameter vectors `parameters`, given as `(name, length)` in
/// declaration order, by a category feature, as in `f(x1) + p[x2]`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WasmTemplateOptions {
    pub combine: String,
    pub expressions: Vec<String>,
    /// Each subexpression's number of arguments; inferred from the calls in `combine` when absent.
    #[serde(default)]
    pub num_features: Option<Vec<usize>>,
    /// The names `combine` uses for the features, in column order; the selected columns' CSV
    /// headers when absent.
    #[serde(default)]
    pub variable_names: Option<Vec<String>>,
    #[serde(default)]
    pub parameters: Vec<(String, usize)>,
}

impl Default for WasmSearchOptions {
    fn default() -> Self {
        Self {
            has_headers: true,
            x_columns: None,
            y_column: None,
            weights_column: None,
            validation_fraction: 0.0,

            loss: "L2DistLoss()".to_string(),
            early_stop_condition: None,
            template: None,
            core: WasmOptionsShim::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EquationPoint {
    pub id: String,
    pub complexity: usize,
    pub loss: f64,
    pub cost: f64,
    /// The Pareto score, as in SymbolicRegression.jl's hall-of-fame table.
    pub score: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EquationSummary {
    pub id: String,
    pub complexity: usize,
    pub loss: f64,
    pub cost: f64,
    /// The Pareto score, as in SymbolicRegression.jl's hall-of-fame table.
    pub score: f64,
    /// Display-precision equation (matches SymbolicRegression.jl `print_precision`, default 5
    /// significant digits). Use this for tables / inline rendering.
    pub equation: String,
    /// Full-precision equation, suitable for copy-to-clipboard.
    pub equation_full: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchSnapshot {
    pub total_cycles: usize,
    pub cycles_completed: usize,
    pub total_evals: u64,
    /// The equation SymbolicRegression.jl reports as best (`choose_best` under `loss_scale`).
    pub best: Option<EquationSummary>,
    pub pareto_points: Vec<EquationPoint>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WasmMetrics {
    pub n: usize,
    pub mse: f64,
    pub mae: f64,
    pub rmse: f64,
    pub r2: f64,
    pub corr: f64,
    pub min_abs_err: f64,
    pub max_abs_err: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WasmEvalResult {
    pub metrics: WasmMetrics,
    pub yhat: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WasmSplitIndices {
    pub train: Vec<usize>,
    pub val: Vec<usize>,
}

#[wasm_bindgen]
pub struct WasmSearch {
    engine: SearchEngine<f64, WasmOps, 3>,
    val_dataset: Option<Dataset<f64>>,
    split: WasmSplitIndices,
    pareto_k: usize,
}

#[wasm_bindgen]
impl WasmSearch {
    #[wasm_bindgen(constructor)]
    pub fn new(
        csv_text: String,
        opts: JsValue,
        unary_tokens: JsValue,
        binary_tokens: JsValue,
        ternary_tokens: JsValue,
    ) -> Result<WasmSearch, JsValue> {
        let opts: WasmSearchOptions = if opts.is_undefined() || opts.is_null() {
            WasmSearchOptions::default()
        } else {
            from_value(opts).map_err(|e| JsValue::from_str(&e.to_string()))?
        };

        let unary: Vec<String> = from_value(unary_tokens).unwrap_or_default();
        let binary: Vec<String> = from_value(binary_tokens).unwrap_or_default();
        let ternary: Vec<String> = from_value(ternary_tokens).unwrap_or_default();

        Self::build(&csv_text, &opts, &unary, &binary, &ternary).map_err(|e| JsValue::from_str(&e))
    }

    pub fn set_pareto_k(&mut self, k: usize) {
        self.pareto_k = k;
    }

    pub fn get_split_indices(&self) -> Result<JsValue, JsValue> {
        to_value(&self.split).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn is_finished(&self) -> bool {
        self.engine.is_finished()
    }

    pub fn step(&mut self, n_cycles: usize) -> Result<JsValue, JsValue> {
        self.engine.step(n_cycles).map_err(|e| JsValue::from_str(&e))?;
        let snap = snapshot(&self.engine, self.pareto_k).map_err(|e| JsValue::from_str(&e))?;
        to_value(&snap).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn best_equations(&self, k: usize) -> Result<JsValue, JsValue> {
        let (front, _) = scored_front(&self.engine).map_err(|e| JsValue::from_str(&e))?;
        let out: Vec<EquationSummary> = front[front.len().saturating_sub(k)..]
            .iter()
            .map(|(m, score)| summary_from_member(&self.engine, m, *score))
            .collect();
        to_value(&out).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn evaluate_member(&self, member_id: String, which: String) -> Result<JsValue, JsValue> {
        let out = self.evaluate(&member_id, &which).map_err(|e| JsValue::from_str(&e))?;
        to_value(&out).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

impl WasmSearch {
    fn build(
        csv_text: &str,
        opts: &WasmSearchOptions,
        unary: &[String],
        binary: &[String],
        ternary: &[String],
    ) -> Result<WasmSearch, String> {
        let operators = operators_from_tokens(unary, binary, ternary)?;

        let (headers, rows) = parse_csv_to_rows(csv_text, opts.has_headers)?;

        let (dataset_all, variable_names, weights_all) = build_full_dataset(&headers, &rows, opts)?;

        let split = make_split_indices(dataset_all.n_rows, opts.validation_fraction, opts.core.seed);
        let (train, val) = split_train_val(&dataset_all, weights_all.as_ref(), &variable_names, &split)?;

        let options = options_from_wasm(opts, operators, &variable_names)?;
        options.check_dataset(&train)?;
        if let Some(val) = &val {
            options.check_dataset(val)?;
        }

        let engine = SearchEngine::<f64, WasmOps, 3>::new(train, options)?;

        Ok(WasmSearch {
            engine,
            val_dataset: val,
            split,
            pareto_k: 250,
        })
    }

    fn evaluate(&self, member_id: &str, which: &str) -> Result<WasmEvalResult, String> {
        let member_id: u64 = member_id
            .parse()
            .map_err(|_| "member_id must be a base-10 integer string".to_string())?;
        let dataset = match which {
            "train" => self.engine.dataset(),
            "val" => self.val_dataset.as_ref().ok_or("no validation dataset")?,
            _ => return Err("which must be \"train\" or \"val\"".to_string()),
        };

        let m = find_member_by_id(&self.engine, member_id).ok_or("member not found")?;

        let yhat = eval_member_on_dataset(dataset, self.engine.options(), &m)?;
        let metrics = compute_metrics(&yhat, dataset);
        Ok(WasmEvalResult { metrics, yhat })
    }
}

fn options_from_wasm(
    opts: &WasmSearchOptions,
    operators: Operators<WasmOps, 3>,
    variable_names: &[String],
) -> Result<Options<f64, WasmOps, 3>, String> {
    let error = |e: dynamic_expressions::dsl::DslError| e.to_string();
    let template = opts
        .template
        .as_ref()
        .map(|t| template_from_wasm(t, variable_names, &operators.set))
        .transpose()?;
    let loss = parse_loss::<f64>(&opts.loss).map_err(error)?;
    let early_stop_condition = opts
        .early_stop_condition
        .as_deref()
        .map(parse_early_stop::<f64>)
        .transpose()
        .map_err(error)?;

    let mut out = Options::<f64, WasmOps, 3> {
        operators,
        ..Default::default()
    };

    opts.core.apply_to(&mut out);
    // Keep browser output clean (and default-features disables progress anyway).
    out.progress = false;
    out.loss = loss;
    out.early_stop_condition = early_stop_condition;
    out.template = template;

    Ok(out)
}

fn template_from_wasm(
    t: &WasmTemplateOptions,
    variable_names: &[String],
    ops: &WasmOps,
) -> Result<TemplateStructure, String> {
    TemplateStructure::new(
        &t.combine,
        &t.expressions,
        t.num_features.as_deref(),
        t.variable_names.as_deref().unwrap_or(variable_names),
        &t.parameters,
        ops,
    )
    .map_err(|e| e.to_string())
}

type ScoredMember = (symbolic_regression::PopMember<f64, WasmOps, 3>, f64);

/// The Pareto front with each member's score, and the index of the member `choose_best` picks
/// under `loss_scale`.
fn scored_front(engine: &SearchEngine<f64, WasmOps, 3>) -> Result<(Vec<ScoredMember>, Option<usize>), String> {
    let loss_scale = engine.options().loss_scale;
    let scores = engine.hall_of_fame().pareto_scores(loss_scale)?;
    let front = engine.hall_of_fame().pareto_front();
    let losses: Vec<f64> = front.iter().map(|m| m.loss).collect();
    let best = choose_best(&losses, &scores, loss_scale);
    Ok((front.into_iter().zip(scores).collect(), best))
}

fn snapshot(engine: &SearchEngine<f64, WasmOps, 3>, pareto_k: usize) -> Result<SearchSnapshot, String> {
    let (front, best) = scored_front(engine)?;
    let best = best.map(|i| summary_from_member(engine, &front[i].0, front[i].1));
    let pareto_points: Vec<EquationPoint> = front[front.len().saturating_sub(pareto_k)..]
        .iter()
        .map(|(m, score)| EquationPoint {
            id: m.expr.hash_nodes().to_string(),
            complexity: m.complexity,
            loss: m.loss,
            cost: m.cost,
            score: *score,
        })
        .collect();

    Ok(SearchSnapshot {
        total_cycles: engine.total_cycles(),
        cycles_completed: engine.cycles_completed(),
        total_evals: engine.total_evals(),
        best,
        pareto_points,
    })
}

fn summary_from_member(
    engine: &SearchEngine<f64, WasmOps, 3>,
    m: &symbolic_regression::PopMember<f64, WasmOps, 3>,
    score: f64,
) -> EquationSummary {
    let string = |print_precision| {
        m.expr.string(
            engine.options(),
            StringTreeOptions {
                variable_names: Some(engine.dataset().variable_names.as_slice()),
                pretty: false,
                print_precision,
            },
        )
    };
    let equation = string(Some(engine.options().print_precision));
    let equation_full = string(None);
    EquationSummary {
        id: m.expr.hash_nodes().to_string(),
        complexity: m.complexity,
        loss: m.loss,
        cost: m.cost,
        score,
        equation,
        equation_full,
    }
}

fn find_member_by_id(
    engine: &SearchEngine<f64, WasmOps, 3>,
    member_id: u64,
) -> Option<symbolic_regression::PopMember<f64, WasmOps, 3>> {
    engine
        .hall_of_fame()
        .members()
        .find(|m| m.expr.hash_nodes() == member_id)
        .cloned()
}

/// The member's predictions on `dataset`: the whole template when there is one.
fn eval_member_on_dataset(
    dataset: &Dataset<f64>,
    options: &Options<f64, WasmOps, 3>,
    m: &symbolic_regression::PopMember<f64, WasmOps, 3>,
) -> Result<Vec<f64>, String> {
    let ops = &options.operators.set;
    let x = dataset.x.view();
    let (yhat, ok) = match &options.template {
        None => eval_tree_array(&m.expr.trees[0], ops, x, &EvalOptions::default()),
        Some(template) => template
            .eval(&m.expr.trees, &m.expr.params, ops, x)
            .map_err(|e| e.to_string())?,
    };
    if !ok {
        return Err("evaluation failed".to_string());
    }
    Ok(yhat)
}

fn compute_metrics(yhat: &[f64], dataset: &Dataset<f64>) -> WasmMetrics {
    let y = dataset.y.as_slice().unwrap_or(&[]);
    let w = dataset.weights_slice();
    let n = y.len();

    let mut mse = 0.0;
    let mut mae = 0.0;
    let mut sum_w = 0.0;
    let mut min_abs = f64::INFINITY;
    let mut max_abs = 0.0;

    match w {
        None => {
            for (&yh, &yi) in yhat.iter().zip_eq(y) {
                let r = yh - yi;
                let ar = r.abs();
                mse += r * r;
                mae += ar;
                if ar < min_abs {
                    min_abs = ar;
                }
                if ar > max_abs {
                    max_abs = ar;
                }
            }
            if n > 0 {
                mse /= n as f64;
                mae /= n as f64;
            }
        }
        Some(w) => {
            for ((&yh, &yi), &wi) in yhat.iter().zip_eq(y).zip_eq(w) {
                let r = yh - yi;
                let ar = r.abs();
                sum_w += wi;
                mse += wi * r * r;
                mae += wi * ar;
                if ar < min_abs {
                    min_abs = ar;
                }
                if ar > max_abs {
                    max_abs = ar;
                }
            }
            if sum_w > 0.0 {
                mse /= sum_w;
                mae /= sum_w;
            } else {
                mse = 0.0;
                mae = 0.0;
            }
        }
    }

    let rmse = mse.sqrt();
    let (r2, corr) = r2_and_corr(yhat, y, w);

    WasmMetrics {
        n,
        mse,
        mae,
        rmse,
        r2,
        corr,
        min_abs_err: if min_abs.is_finite() { min_abs } else { 0.0 },
        max_abs_err: max_abs,
    }
}

fn r2_and_corr(yhat: &[f64], y: &[f64], w: Option<&[f64]>) -> (f64, f64) {
    if y.is_empty() || yhat.len() != y.len() {
        return (f64::NAN, f64::NAN);
    }

    // Weighted mean for R^2; correlation unweighted (for UI purposes).
    let y_mean = match w {
        None => y.iter().sum::<f64>() / (y.len() as f64),
        Some(w) => {
            let sum_w = w.iter().sum::<f64>();
            if sum_w == 0.0 {
                0.0
            } else {
                y.iter().zip_eq(w).map(|(&yi, &wi)| yi * wi).sum::<f64>() / sum_w
            }
        }
    };

    let mut sse = 0.0;
    let mut sst = 0.0;
    match w {
        None => {
            for (&yh, &yi) in yhat.iter().zip_eq(y) {
                let r = yh - yi;
                sse += r * r;
                let d = yi - y_mean;
                sst += d * d;
            }
        }
        Some(w) => {
            for ((&yh, &yi), &wi) in yhat.iter().zip_eq(y).zip_eq(w) {
                let r = yh - yi;
                sse += wi * r * r;
                let d = yi - y_mean;
                sst += wi * d * d;
            }
        }
    }
    let r2 = if sst == 0.0 { 0.0 } else { 1.0 - (sse / sst) };

    // Unweighted Pearson correlation.
    let yh_mean = yhat.iter().sum::<f64>() / (yhat.len() as f64);
    let mut cov = 0.0;
    let mut vy = 0.0;
    let mut vyh = 0.0;
    for (&yh, &yi) in yhat.iter().zip_eq(y) {
        let dyh = yh - yh_mean;
        let dy = yi - y_mean;
        cov += dyh * dy;
        vyh += dyh * dyh;
        vy += dy * dy;
    }
    let corr = if vy == 0.0 || vyh == 0.0 {
        0.0
    } else {
        cov / (vy.sqrt() * vyh.sqrt())
    };

    (r2, corr)
}

fn parse_csv_to_rows(csv_text: &str, has_headers: bool) -> Result<(Vec<String>, Vec<Vec<f64>>), String> {
    let mut rdr = ReaderBuilder::new()
        .trim(csv::Trim::All)
        .has_headers(has_headers)
        .from_reader(Cursor::new(csv_text.as_bytes()));

    let headers: Vec<String> = if has_headers {
        rdr.headers()
            .map_err(|e| e.to_string())?
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        Vec::new()
    };

    let mut rows: Vec<Vec<f64>> = Vec::new();
    for rec in rdr.records() {
        let rec = rec.map_err(|e| e.to_string())?;
        let mut row: Vec<f64> = Vec::with_capacity(rec.len());
        for s in rec.iter() {
            let v = s
                .parse::<f64>()
                .map_err(|e| format!("failed to parse {s:?} as f64: {e}"))?;
            row.push(v);
        }
        rows.push(row);
    }
    Ok((headers, rows))
}

fn build_full_dataset(
    headers: &[String],
    rows: &[Vec<f64>],
    opts: &WasmSearchOptions,
) -> Result<FullDatasetParts, String> {
    if rows.is_empty() {
        return Err("CSV had no data rows".to_string());
    }
    let n_rows = rows.len();
    let n_cols = rows[0].len();
    if n_cols < 2 {
        return Err("CSV must have at least 2 columns".to_string());
    }
    for (i, row) in rows.iter().enumerate() {
        if row.len() != n_cols {
            return Err(format!(
                "row {i} has {found} columns but expected {n_cols}",
                found = row.len()
            ));
        }
    }

    let y_col = opts.y_column.unwrap_or(n_cols - 1);
    if y_col >= n_cols {
        return Err(format!("y_column out of range: {y_col} (n_cols={n_cols})"));
    }

    let w_col = opts.weights_column;
    if let Some(wc) = w_col {
        if wc >= n_cols {
            return Err(format!("weights_column out of range: {wc} (n_cols={n_cols})"));
        }
        if wc == y_col {
            return Err("weights_column must be different from y_column".to_string());
        }
    }

    let mut x_cols = match &opts.x_columns {
        Some(v) => v.clone(),
        None => (0..n_cols).filter(|&c| c != y_col && Some(c) != w_col).collect(),
    };
    x_cols.sort_unstable();
    x_cols.dedup();
    if x_cols.is_empty() {
        return Err("x_columns is empty".to_string());
    }
    if x_cols.iter().any(|&c| c >= n_cols) {
        return Err("x_columns contains an out-of-range index".to_string());
    }
    if x_cols.contains(&y_col) {
        return Err("x_columns must not include y_column".to_string());
    }
    if let Some(wc) = w_col {
        if x_cols.contains(&wc) {
            return Err("x_columns must not include weights_column".to_string());
        }
    }

    let n_features = x_cols.len();

    let mut x = Array2::<f64>::zeros((n_features, n_rows));
    let mut y = Array1::<f64>::zeros(n_rows);
    let mut w = w_col.map(|_| Array1::<f64>::zeros(n_rows));

    for (i, row) in rows.iter().enumerate() {
        for (j, &c) in x_cols.iter().enumerate() {
            x[(j, i)] = row[c];
        }
        y[i] = row[y_col];
        if let (Some(wc), Some(w_arr)) = (w_col, w.as_mut()) {
            w_arr[i] = row[wc];
        }
    }

    let variable_names: Vec<String> = if opts.has_headers && headers.len() == n_cols {
        x_cols.iter().map(|&c| headers[c].clone()).collect()
    } else {
        (0..n_features).map(|i| format!("x{}", i)).collect()
    };

    let dataset = Dataset::with_weights_and_names(x, y, None, variable_names.clone());
    Ok((dataset, variable_names, w))
}

fn make_split_indices(n_rows: usize, validation_fraction: f64, seed: u64) -> WasmSplitIndices {
    if n_rows == 0 {
        return WasmSplitIndices {
            train: Vec::new(),
            val: Vec::new(),
        };
    }

    let vf = validation_fraction.clamp(0.0, 0.9);
    let n_val = ((vf * (n_rows as f64)).round() as usize).min(n_rows);
    if n_val == 0 {
        return WasmSplitIndices {
            train: (0..n_rows).collect(),
            val: Vec::new(),
        };
    }

    let mut idx: Vec<usize> = (0..n_rows).collect();
    let mut rng = Rng::with_seed(seed ^ 0x6a09_e667_f3bc_c909);
    shuffle(&mut rng, &mut idx);

    let val = idx[..n_val].to_vec();
    let train = idx[n_val..].to_vec();

    WasmSplitIndices { train, val }
}

fn split_train_val(
    dataset_all: &Dataset<f64>,
    weights_all: Option<&Array1<f64>>,
    variable_names: &[String],
    split: &WasmSplitIndices,
) -> Result<(Dataset<f64>, Option<Dataset<f64>>), String> {
    let (x_train, y_train, w_train) = select_rows(&dataset_all.x, &dataset_all.y, weights_all, &split.train)?;
    let train = Dataset::with_weights_and_names(x_train, y_train, w_train, variable_names.to_vec());

    if split.val.is_empty() {
        return Ok((train, None));
    }
    let (x_val, y_val, w_val) = select_rows(&dataset_all.x, &dataset_all.y, weights_all, &split.val)?;
    let val = Dataset::with_weights_and_names(x_val, y_val, w_val, variable_names.to_vec());
    Ok((train, Some(val)))
}

fn select_rows(
    x: &Array2<f64>,
    y: &Array1<f64>,
    w: Option<&Array1<f64>>,
    indices: &[usize],
) -> Result<SelectedRows, String> {
    let (n_features, n_rows) = x.dim();
    for &i in indices {
        if i >= n_rows {
            return Err(format!("index out of range while splitting: {i} (n_rows={n_rows})"));
        }
    }

    let mut xo = Array2::<f64>::zeros((n_features, indices.len()));
    let mut yo = Array1::<f64>::zeros(indices.len());
    let mut wo = w.map(|_| Array1::<f64>::zeros(indices.len()));

    for (i_new, &i_old) in indices.iter().enumerate() {
        for j in 0..n_features {
            xo[(j, i_new)] = x[(j, i_old)];
        }
        yo[i_new] = y[i_old];
        if let (Some(w_in), Some(w_out)) = (w, wo.as_mut()) {
            w_out[i_new] = w_in[i_old];
        }
    }
    Ok((xo, yo, wo))
}

#[cfg(test)]
mod tests {
    use super::WasmSearchOptions;

    #[test]
    fn wasm_options_shim_defaults_match_known_spec_values() {
        let core = WasmSearchOptions::default().core;
        assert_eq!(core.seed, 0);
        assert_eq!(core.niterations, 100);
        assert_eq!(core.populations, 31);
        assert_eq!(core.population_size, 27);
        assert_eq!(core.maxsize, 30);
        assert_eq!(core.maxdepth, 30);
        assert!(core.progress);
        assert!(!core.batching);
    }

    #[test]
    fn wasm_search_options_flattened_serde_shape_has_no_core_field() {
        let opts = WasmSearchOptions::default();
        let v = serde_json::to_value(&opts).unwrap();
        let obj = v.as_object().unwrap();
        assert!(obj.contains_key("seed"));
        assert!(obj.contains_key("mutation_weights"));
        assert!(!obj.contains_key("core"));
    }

    #[test]
    fn wasm_search_options_read_loss_strings_and_loss_scale() {
        let opts: WasmSearchOptions = serde_json::from_value(serde_json::json!({
            "loss": "myloss(x, y, w) = w * abs(x - y)",
            "early_stop_condition": "f(loss, complexity) = loss < 1e-6 && complexity < 10",
            "loss_scale": "linear",
        }))
        .unwrap();
        assert_eq!(opts.loss, "myloss(x, y, w) = w * abs(x - y)");
        assert_eq!(opts.core.loss_scale, symbolic_regression::LossScale::Linear);
        let options = symbolic_regression::Options::<f64, super::WasmOps, 3> {
            loss: symbolic_regression::parse_loss(&opts.loss).unwrap(),
            ..Default::default()
        };
        assert!(options.loss.check_weights(false).is_err());
        assert!(options.loss.check_weights(true).is_ok());

        assert_eq!(WasmSearchOptions::default().loss, "L2DistLoss()");
        assert!(WasmSearchOptions::default().early_stop_condition.is_none());
        assert!(serde_json::from_value::<WasmSearchOptions>(serde_json::json!({ "loss_scale": "cubic" })).is_err());
    }

    #[test]
    fn wasm_options_shim_apply_to_patches_core_options() {
        let mut core = WasmSearchOptions::default().core;
        core.seed = 123;
        core.niterations = 9;
        core.progress = false;

        let mut opt: symbolic_regression::Options<f64, super::WasmOps, 3> = Default::default();
        core.apply_to(&mut opt);

        assert_eq!(opt.seed, 123);
        assert_eq!(opt.niterations, 9);
        assert!(!opt.progress);
    }
    fn tokens(v: &[&str]) -> Vec<String> {
        v.iter().map(|t| t.to_string()).collect()
    }

    #[test]
    fn operator_tokens_accept_definitions() {
        use dynamic_expressions::OperatorSet;

        let operators =
            super::operators_from_tokens(&tokens(&["cos", "-", "inv(x) = 1/x"]), &tokens(&["+", "-"]), &[]).unwrap();
        assert_eq!(operators.nops(1), 3);
        assert_eq!(operators.nops(2), 2);
        let inv = operators.set.lookup("inv").unwrap();
        assert!(operators.set.is_custom(inv));
        assert!(operators.ops_by_arity[0].contains(&inv));
    }

    #[test]
    fn operator_tokens_must_match_their_list_arity() {
        let err = super::operators_from_tokens(&tokens(&["f(x, y) = x * y"]), &[], &[]).unwrap_err();
        assert_eq!(
            err,
            "operator `f` takes 2 arguments, but it is in the unary operator list"
        );
        let err = super::operators_from_tokens(&[], &tokens(&["+", "cos"]), &[]).unwrap_err();
        assert_eq!(
            err,
            "operator `cos` takes 1 argument, but it is in the binary operator list"
        );
    }

    #[test]
    fn operator_tokens_reject_external_package_functions() {
        let err = super::operators_from_tokens(&tokens(&["myerf(x) = SpecialFunctions.erf(x)"]), &[], &[]).unwrap_err();
        assert!(err.contains("`SpecialFunctions.erf`"), "{err}");
        let err = super::operators_from_tokens(&[], &tokens(&["p(a, b) = Primes.prime(a) + b"]), &[]).unwrap_err();
        assert!(err.contains("`Primes.prime`"), "{err}");
    }

    fn engine(loss: &str, loss_scale: symbolic_regression::LossScale) -> super::SearchEngine<f64, super::WasmOps, 3> {
        let x = ndarray::Array2::from_shape_fn((1, 32), |(_, i)| i as f64 / 8.0);
        let y = x.row(0).mapv(|v| v * v - 1.0);
        let options = super::Options::<f64, super::WasmOps, 3> {
            operators: super::operators_from_tokens(&[], &tokens(&["+", "*"]), &[]).unwrap(),
            loss: super::parse_loss(loss).unwrap(),
            loss_scale,
            niterations: 2,
            populations: 4,
            progress: false,
            ..Default::default()
        };
        super::SearchEngine::new(super::Dataset::new(x, y), options).unwrap()
    }

    #[test]
    fn search_errors_on_negative_losses_under_log_scale() {
        use symbolic_regression::LossScale;

        let mut search = engine("(x, y) -> abs(x - y) - 100", LossScale::Log);
        let err = super::snapshot(&search, 10).unwrap_err();
        assert!(err.contains("negative loss"), "{err}");
        assert!(search.step(usize::MAX).unwrap_err().contains("negative loss"));

        let mut search = engine("(x, y) -> abs(x - y) - 100", LossScale::Linear);
        search.step(usize::MAX).unwrap();
        assert!(super::snapshot(&search, 10).unwrap().best.unwrap().loss < 0.0);
    }

    #[test]
    fn snapshot_rows_carry_scores_and_best_is_the_choose_best_pick() {
        use symbolic_regression::{LossScale, choose_best};

        for loss_scale in [LossScale::Log, LossScale::Linear] {
            let mut search = engine("L2DistLoss()", loss_scale);
            search.step(usize::MAX).unwrap();
            let snap = super::snapshot(&search, usize::MAX).unwrap();
            let scores: Vec<f64> = snap.pareto_points.iter().map(|p| p.score).collect();
            assert_eq!(scores, search.hall_of_fame().pareto_scores(loss_scale).unwrap());

            let losses: Vec<f64> = snap.pareto_points.iter().map(|p| p.loss).collect();
            let pick = &snap.pareto_points[choose_best(&losses, &scores, loss_scale).unwrap()];
            let best = snap.best.unwrap();
            assert_eq!((&best.id, best.score), (&pick.id, pick.score), "{loss_scale:?}");

            let last_two = super::snapshot(&search, 2).unwrap().pareto_points;
            let ids = |points: &[super::EquationPoint]| points.iter().map(|p| p.id.clone()).collect::<Vec<_>>();
            assert_eq!(ids(&last_two), ids(&snap.pareto_points[snap.pareto_points.len() - 2..]));
        }
    }

    /// `y = 2x + p[c]` with `p = [0.5, -1.5]`, for 40 rows alternating category 1 and 2.
    fn template_csv() -> String {
        let mut csv = "x,c,y\n".to_string();
        for i in 0..40 {
            let (x, c) = (f64::from(i) / 10.0 - 2.0, 1 + i % 2);
            let p = [0.5, -1.5][c as usize - 1];
            csv.push_str(&format!("{x},{c},{}\n", 2.0 * x + p));
        }
        csv
    }

    fn template_options(template: serde_json::Value) -> WasmSearchOptions {
        serde_json::from_value(serde_json::json!({
            "niterations": 3,
            "populations": 2,
            "population_size": 20,
            "seed": 1,
            "template": template,
        }))
        .unwrap()
    }

    fn template_search(template: serde_json::Value) -> Result<super::WasmSearch, String> {
        super::WasmSearch::build(
            &template_csv(),
            &template_options(template),
            &[],
            &tokens(&["+", "*"]),
            &[],
        )
    }

    #[test]
    fn template_option_is_absent_from_old_option_objects() {
        assert!(WasmSearchOptions::default().template.is_none());
        assert!(serde_json::to_value(WasmSearchOptions::default()).unwrap()["template"].is_null());
        for old in [
            serde_json::json!({ "loss": "L1DistLoss()" }),
            serde_json::json!({ "template": null }),
        ] {
            assert!(
                serde_json::from_value::<WasmSearchOptions>(old)
                    .unwrap()
                    .template
                    .is_none()
            );
        }
        let t = template_options(serde_json::json!({ "combine": "f(x) + p[c]", "expressions": ["f"] }))
            .template
            .unwrap();
        assert_eq!((t.num_features, t.variable_names), (None, None));
        assert!(t.parameters.is_empty());
    }

    #[test]
    fn template_arities_are_inferred_unless_given() {
        let ops = super::operators_from_tokens(&[], &tokens(&["+", "*"]), &[]).unwrap();
        let spec = |num_features: Option<Vec<usize>>| super::WasmTemplateOptions {
            combine: "f(a, b) + g(a)".to_string(),
            expressions: tokens(&["f", "g"]),
            num_features,
            variable_names: None,
            parameters: Vec::new(),
        };
        let names = tokens(&["a", "b"]);
        let arities = |s| {
            super::template_from_wasm(&s, &names, &ops.set)
                .unwrap()
                .num_features()
                .to_vec()
        };
        assert_eq!(arities(spec(None)), [2, 1]);
        assert_eq!(arities(spec(Some(vec![3, 1]))), [3, 1]);
    }

    #[test]
    fn template_names_default_to_the_selected_headers() {
        let ops = super::operators_from_tokens(&[], &tokens(&["+"]), &[]).unwrap();
        let mut spec = super::WasmTemplateOptions {
            combine: "f(u, v)".to_string(),
            expressions: tokens(&["f"]),
            num_features: None,
            variable_names: Some(tokens(&["u", "v"])),
            parameters: Vec::new(),
        };
        let headers = tokens(&["a", "b"]);
        let t = super::template_from_wasm(&spec, &headers, &ops.set).unwrap();
        assert_eq!(t.variable_names(), ["u", "v"]);
        spec.variable_names = None;
        let err = super::template_from_wasm(&spec, &headers, &ops.set).unwrap_err();
        assert!(err.contains('u'), "{err}");
        spec.combine = "f(a, b)".to_string();
        let t = super::template_from_wasm(&spec, &headers, &ops.set).unwrap();
        assert_eq!(t.variable_names(), ["a", "b"]);
    }

    #[test]
    fn template_parameters_keep_their_declaration_order() {
        let t = template_options(serde_json::json!({
            "combine": "f(x) + q[c] * p[c]",
            "expressions": ["f"],
            "parameters": [["q", 2], ["p", 3]],
        }))
        .template
        .unwrap();
        assert_eq!(t.parameters, [("q".to_string(), 2), ("p".to_string(), 3)]);
        let ops = super::operators_from_tokens(&[], &tokens(&["+"]), &[]).unwrap();
        let t = super::template_from_wasm(&t, &tokens(&["x", "c"]), &ops.set).unwrap();
        assert_eq!(t.parameters(), [("q".to_string(), 2), ("p".to_string(), 3)]);
    }

    #[test]
    fn invalid_templates_fail_at_setup() {
        let cases = [
            (serde_json::json!({ "combine": "f(z)", "expressions": ["f"] }), "`z`"),
            (
                serde_json::json!({ "combine": "f(x)", "expressions": ["f"], "num_features": [1, 2] }),
                "1 subexpressions but 2 feature counts",
            ),
            (
                serde_json::json!({ "combine": "f(x)", "expressions": ["f"], "variable_names": ["x"] }),
                "the template reads 1 features (x), but the dataset has 2",
            ),
            (
                serde_json::json!({ "combine": "f(x) + p[c]", "expressions": ["f"], "parameters": [["p", 1]] }),
                "indexes `p` with 2",
            ),
        ];
        for (template, want) in cases {
            let err = template_search(template.clone()).err().unwrap();
            assert!(err.contains(want), "{template}: {err}");
        }
    }

    #[test]
    fn template_results_print_and_predict_the_whole_template() {
        let mut search = template_search(serde_json::json!({
            "combine": "f(x) + p[c]",
            "expressions": ["f"],
            "parameters": [["p", 2]],
        }))
        .unwrap();
        search.engine.step(usize::MAX).unwrap();
        let options = search.engine.options();
        let template = options.template.as_ref().unwrap();
        let x = search.engine.dataset().x.view();
        let front = super::snapshot(&search.engine, usize::MAX).unwrap().pareto_points;
        assert!(!front.is_empty());
        for point in &front {
            let m = super::find_member_by_id(&search.engine, point.id.parse().unwrap()).unwrap();
            let summary = super::summary_from_member(&search.engine, &m, point.score);
            let want = template.string(&m.expr.trees, &m.expr.params, &options.operators.set, false, None);
            assert_eq!(summary.equation_full, want);
            assert!(want.starts_with("f = ") && want.contains("; p = ["), "{want}");

            let result = search.evaluate(&point.id, "train").unwrap();
            let (yhat, ok) = template
                .eval(&m.expr.trees, &m.expr.params, &options.operators.set, x)
                .unwrap();
            assert!(ok);
            assert_eq!(result.yhat, yhat);
            assert!(
                (result.metrics.mse - m.loss).abs() <= 1e-9 * m.loss.max(1.0),
                "{} vs {}",
                result.metrics.mse,
                m.loss
            );
        }
    }
}
