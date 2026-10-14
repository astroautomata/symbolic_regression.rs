use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::Duration;

use ndarray::{Array1, Array2, ArrayView2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyKeyboardInterrupt, PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use serde::{Deserialize, Serialize};
use symbolic_regression::{
    CREATE_OUTPUT_DIRECTORY_ERROR_PREFIX, Dataset, Evaluator, ExternalStop, OPEN_TRACING_FILE_ERROR_PREFIX,
    SearchResult, WRITE_HALL_OF_FAME_ERROR_PREFIX, WRITE_TRACING_FILE_ERROR_PREFIX, equation_search, hall_of_fame_rows,
    resume_search,
};

use crate::keywords::{self, D, Keywords, Precision, SearchOptions, Value};

/// How often a running search checks for signals such as Ctrl-C.
const SIGNAL_POLL: Duration = Duration::from_millis(100);

// Bump when the serialized layout of Saved, StateData, or any transitive field type changes.
const STATE_FORMAT: u32 = 2;

fn value_error(message: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(message.to_string())
}

fn search_error(message: String) -> PyErr {
    if [
        CREATE_OUTPUT_DIRECTORY_ERROR_PREFIX,
        WRITE_HALL_OF_FAME_ERROR_PREFIX,
        OPEN_TRACING_FILE_ERROR_PREFIX,
        WRITE_TRACING_FILE_ERROR_PREFIX,
    ]
    .iter()
    .any(|prefix| message.starts_with(prefix))
    {
        PyOSError::new_err(message)
    } else {
        value_error(message)
    }
}

/// A search state for one float type: what the search left, and each output's hall-of-fame rows.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
struct Saved<T: Precision> {
    result: SearchResult<T, T::Ops, D>,
    fronts: Vec<Vec<(usize, f64, String)>>,
    interrupted: bool,
    n_features: usize,
    keywords: Vec<(String, Value)>,
}

#[derive(Clone, Serialize, Deserialize)]
enum StateData {
    F32(Saved<f32>),
    F64(Saved<f64>),
}

impl StateData {
    fn dtype(&self) -> &'static str {
        match self {
            StateData::F32(_) => f32::DTYPE,
            StateData::F64(_) => f64::DTYPE,
        }
    }
}

/// A float type that Python arrays and search states hold.
trait Stored: Precision + numpy::Element {
    const DTYPE: &'static str;
    fn wrap(saved: Saved<Self>) -> StateData;
    fn saved(data: &StateData) -> Option<&Saved<Self>>;
}

impl Stored for f32 {
    const DTYPE: &'static str = "float32";
    fn wrap(saved: Saved<f32>) -> StateData {
        StateData::F32(saved)
    }
    fn saved(data: &StateData) -> Option<&Saved<f32>> {
        match data {
            StateData::F32(saved) => Some(saved),
            StateData::F64(_) => None,
        }
    }
}

impl Stored for f64 {
    const DTYPE: &'static str = "float64";
    fn wrap(saved: Saved<f64>) -> StateData {
        StateData::F64(saved)
    }
    fn saved(data: &StateData) -> Option<&Saved<f64>> {
        match data {
            StateData::F64(saved) => Some(saved),
            StateData::F32(_) => None,
        }
    }
}

/// What a search found, and the state that `equation_search(..., saved_state=state)` continues.
/// Pickles.
#[pyclass(module = "symbolic_regression_rs", frozen)]
struct SearchState {
    data: StateData,
}

macro_rules! with_saved {
    ($data:expr, $saved:ident => $body:expr) => {
        match $data {
            StateData::F32($saved) => $body,
            StateData::F64($saved) => $body,
        }
    };
}

#[pymethods]
impl SearchState {
    /// Each output's Pareto front as its hall-of-fame file lists it: `(complexity, loss,
    /// equation)` rows from simplest to most complex.
    #[getter]
    fn pareto_fronts(&self) -> Vec<Vec<(usize, f64, String)>> {
        with_saved!(&self.data, s => s.fronts.clone())
    }

    /// The directory beneath `output_directory` that holds this run's files.
    #[getter]
    fn run_id(&self) -> String {
        with_saved!(&self.data, s => s.result.run_id.clone())
    }

    /// Evaluations, in units of the full dataset, including those of every search this one
    /// continues.
    #[getter]
    fn total_evals(&self) -> f64 {
        with_saved!(&self.data, s => s.result.total_evals)
    }

    /// The warnings SymbolicRegression.jl logs while setting up the search.
    #[getter]
    fn warnings(&self) -> Vec<String> {
        with_saved!(&self.data, s => s.result.warnings.clone())
    }

    /// Whether a signal such as Ctrl-C stopped the search early.
    #[getter]
    fn interrupted(&self) -> bool {
        with_saved!(&self.data, s => s.interrupted)
    }

    /// Evaluates the selected Pareto expression on (n_samples, n_features) inputs.
    #[pyo3(signature = (X, *, output=0, complexity=None))]
    #[allow(non_snake_case)]
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        X: &Bound<'py, PyAny>,
        output: i64,
        complexity: Option<i64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let output = usize::try_from(output).map_err(|_| value_error("output must be nonnegative"))?;
        let complexity = complexity
            .map(|c| usize::try_from(c).map_err(|_| value_error("complexity must be nonnegative")))
            .transpose()?;
        match &self.data {
            StateData::F32(saved) => evaluate_saved(py, saved, X, output, complexity).map(Bound::into_any),
            StateData::F64(saved) => evaluate_saved(py, saved, X, output, complexity).map(Bound::into_any),
        }
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyBytes>,))> {
        let mut bytes =
            postcard::to_extend(&(env!("CARGO_PKG_VERSION"), &self.data), vec![0; 8]).map_err(value_error)?;
        let checksum = crc32fast::hash(&bytes[8..]);
        bytes[..4].copy_from_slice(&STATE_FORMAT.to_le_bytes());
        bytes[4..8].copy_from_slice(&checksum.to_le_bytes());
        let load = py.import("symbolic_regression_rs")?.getattr("_load_state")?;
        Ok((load, (PyBytes::new(py, &bytes),)))
    }
}

/// Unpickles a [`SearchState`].
#[pyfunction]
fn _load_state(bytes: &[u8]) -> PyResult<SearchState> {
    let header = bytes
        .get(..8)
        .ok_or_else(|| value_error("search state header is truncated"))?;
    let format = u32::from_le_bytes(header[..4].try_into().expect("four header bytes"));
    if format != STATE_FORMAT {
        return Err(value_error(format!(
            "unsupported search state format {format}; expected {STATE_FORMAT}"
        )));
    }
    let checksum = u32::from_le_bytes(header[4..8].try_into().expect("four checksum bytes"));
    if crc32fast::hash(&bytes[8..]) != checksum {
        return Err(value_error("search state checksum does not match"));
    }
    let (version, rest) = postcard::take_from_bytes::<String>(&bytes[8..]).map_err(value_error)?;
    if version != env!("CARGO_PKG_VERSION") {
        return Err(value_error(format!(
            "this search state was saved by symbolic-regression-rs {version}, which this version ({}) cannot load",
            env!("CARGO_PKG_VERSION")
        )));
    }
    let (data, rest) = postcard::take_from_bytes::<StateData>(rest).map_err(value_error)?;
    if !rest.is_empty() {
        return Err(value_error("search state has trailing bytes"));
    }
    Ok(SearchState { data })
}

fn evaluate_saved<'py, T: Stored>(
    py: Python<'py>,
    saved: &Saved<T>,
    x: &Bound<'py, PyAny>,
    output: usize,
    complexity: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<T>>> {
    let front = saved
        .fronts
        .get(output)
        .ok_or_else(|| value_error(format!("output {output} does not exist")))?;
    let complexity = match complexity {
        Some(c) if front.iter().any(|row| row.0 == c) => c,
        Some(c) => return Err(value_error(format!("complexity {c} is not a returned Pareto row"))),
        None => {
            front
                .last()
                .ok_or_else(|| value_error("output has no Pareto expressions"))?
                .0
        }
    };
    let member = saved
        .result
        .halls_of_fame
        .get(output)
        .and_then(|hall| hall.best_by_complexity.get(complexity))
        .and_then(Option::as_ref)
        .ok_or_else(|| value_error("Pareto expression is missing from the hall of fame"))?;
    let x = x.cast::<PyArray2<T>>().map_err(|_| {
        value_error(format!(
            "X must be a 2-D {} NumPy array, as the state is; got {}",
            T::DTYPE,
            describe(x).unwrap_or_default()
        ))
    })?;
    let x = x.readonly();
    let x = x.as_array();
    if x.ncols() != saved.n_features {
        return Err(value_error(format!(
            "X has {} features; the state requires {}",
            x.ncols(),
            saved.n_features
        )));
    }
    let x = x.t().as_standard_layout().into_owned();
    let predictions = py
        .detach(move || -> Result<Vec<T>, String> {
            let options = keywords::evaluation_options::<T>(saved.keywords.clone())?;
            if let Some(template) = &options.template {
                template.check_indices(x.view())?;
            }
            let mut evaluator = Evaluator::<T, D>::new(x.ncols());
            let valid = evaluator
                .eval_checked(&member.expr, &member.plans, &options, x.view())
                .map_err(|err| err.to_string())?;
            if !valid || evaluator.yhat.iter().any(|value| !value.is_finite()) {
                return Err("expression evaluation produced invalid or nonfinite predictions".into());
            }
            Ok(evaluator.yhat)
        })
        .map_err(value_error)?;
    Ok(predictions.into_pyarray(py))
}

/// A Python keyword value as [`Value`]: lists and tuples become lists, and NumPy arrays and
/// scalars become what their `tolist()` returns.
fn value(key: &str, obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    Ok(if obj.is_none() {
        Value::None
    } else if let Ok(b) = obj.cast::<PyBool>() {
        Value::Bool(b.is_true())
    } else if obj.is_instance_of::<PyInt>() {
        match obj.extract::<i64>() {
            Ok(integer) => Value::Int(integer),
            Err(_) => Value::UInt(
                obj.extract::<u64>()
                    .map_err(|_| value_error(format!("`{key}` has an integer too large for 64 bits")))?,
            ),
        }
    } else if obj.is_instance_of::<PyFloat>() {
        Value::Float(obj.extract()?)
    } else if obj.is_instance_of::<PyString>() {
        Value::Str(obj.extract()?)
    } else if let Ok(dict) = obj.cast::<PyDict>() {
        Value::Dict(
            dict.iter()
                .map(|(k, v)| Ok((value(key, &k)?, value(key, &v)?)))
                .collect::<PyResult<_>>()?,
        )
    } else if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        Value::List(
            obj.try_iter()?
                .map(|item| value(key, &item?))
                .collect::<PyResult<_>>()?,
        )
    } else if obj.hasattr("tolist")? {
        value(key, &obj.call_method0("tolist")?)?
    } else {
        return Err(value_error(format!(
            "`{key}` cannot take a value of type {}",
            obj.get_type().name()?
        )));
    })
}

/// The array's shape and dtype, as an error message describes it.
fn describe(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(match obj.cast::<PyUntypedArray>() {
        Ok(array) => format!("a {}-D {} array", array.ndim(), array.dtype()),
        Err(_) => format!("a {}", obj.get_type().name()?),
    })
}

/// Runs a search, as SymbolicRegression.jl's `equation_search` does. `X` has shape
/// `(n_samples, n_features)` and `y` shape `(n_samples,)`, or `(n_samples, n_outputs)` for one
/// search per output; both are float32 or float64 arrays, which select the search's float type.
/// Every other keyword is a SymbolicRegression.jl option, by its name in `Options` or
/// `equation_search`. Ctrl-C stops the search after its running cycles finish and returns what it
/// found.
#[pyfunction(name = "equation_search")]
#[pyo3(signature = (
    X, y, *, weights=None, variable_names=None, display_variable_names=None, y_variable_names=None,
    X_units=None, y_units=None, guesses=None, saved_state=None, **options
))]
#[allow(non_snake_case, clippy::too_many_arguments)]
fn equation_search_py(
    py: Python<'_>,
    X: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    weights: Option<&Bound<'_, PyAny>>,
    variable_names: Option<Vec<String>>,
    display_variable_names: Option<Vec<String>>,
    y_variable_names: Option<&Bound<'_, PyAny>>,
    X_units: Option<Vec<String>>,
    y_units: Option<&Bound<'_, PyAny>>,
    guesses: Option<&Bound<'_, PyAny>>,
    saved_state: Option<PyRef<'_, SearchState>>,
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<SearchState> {
    let args = SearchArgs {
        y,
        weights,
        variable_names: variable_names.unwrap_or_default(),
        display_variable_names,
        y_variable_names,
        x_units: X_units,
        y_units,
        guesses,
        saved_state: saved_state.as_ref().map(|s| &s.data),
        options,
    };
    if let Ok(x) = X.cast::<PyArray2<f64>>() {
        search(py, x.readonly().as_array(), args)
    } else if let Ok(x) = X.cast::<PyArray2<f32>>() {
        search(py, x.readonly().as_array(), args)
    } else {
        Err(value_error(format!(
            "X must be a 2-D float32 or float64 NumPy array of shape (n_samples, n_features), got {}",
            describe(X)?
        )))
    }
}

struct SearchArgs<'a, 'py> {
    y: &'a Bound<'py, PyAny>,
    weights: Option<&'a Bound<'py, PyAny>>,
    variable_names: Vec<String>,
    display_variable_names: Option<Vec<String>>,
    y_variable_names: Option<&'a Bound<'py, PyAny>>,
    x_units: Option<Vec<String>>,
    y_units: Option<&'a Bound<'py, PyAny>>,
    guesses: Option<&'a Bound<'py, PyAny>>,
    saved_state: Option<&'a StateData>,
    options: Option<&'a Bound<'py, PyDict>>,
}

/// `name` as an array of `x`'s dtype with one sample per row, as an `(n_outputs, n_samples)`
/// array: a 1-D array is one output.
fn per_output<T: Stored>(name: &str, obj: &Bound<'_, PyAny>, n_samples: usize) -> PyResult<(Array2<T>, bool)> {
    let array = obj.cast::<PyArrayDyn<T>>().map_err(|_| {
        let got = describe(obj).unwrap_or_default();
        value_error(format!("{name} must be a {} NumPy array, as X is; got {got}", T::DTYPE))
    })?;
    let array = array.readonly();
    let view = array.as_array();
    let (columns, multi) = match *view.shape() {
        [n] if n == n_samples => (view.insert_axis(Axis(0)).into_dimensionality().expect("2-D"), false),
        [n, _] if n == n_samples => (view.into_dimensionality().expect("2-D").reversed_axes(), true),
        _ => {
            return Err(value_error(format!(
                "{name} must have shape (n_samples,) or (n_samples, n_outputs) with n_samples = {n_samples}, as X has; got shape {:?}",
                view.shape()
            )));
        }
    };
    let columns: ArrayView2<T> = columns;
    Ok((columns.as_standard_layout().into_owned(), multi))
}

fn search<T: Stored>(py: Python<'_>, x: ArrayView2<'_, T>, args: SearchArgs<'_, '_>) -> PyResult<SearchState> {
    let (n_samples, n_features) = x.dim();
    if n_samples == 0 || n_features == 0 {
        return Err(value_error("X must have at least one sample and one feature"));
    }
    let (y, multi_output) = per_output::<T>("y", args.y, n_samples)?;
    let nout = y.nrows();
    if nout == 0 {
        return Err(value_error("y must have at least one output"));
    }
    let weights = match args.weights {
        Some(w) => {
            let (w, weights_multi_output) = per_output::<T>("weights", w, n_samples)?;
            if weights_multi_output != multi_output || w.dim() != y.dim() {
                return Err(value_error("weights must have the shape of y"));
            }
            Some(w)
        }
        None => None,
    };
    if !args.variable_names.is_empty() && args.variable_names.len() != n_features {
        return Err(value_error(format!(
            "variable_names has {} names, but X has {n_features} features",
            args.variable_names.len()
        )));
    }

    let x = x.t().as_standard_layout().into_owned();
    let mut datasets = if multi_output {
        Dataset::multi_output(x, y, weights, args.variable_names)
    } else {
        let y: Array1<T> = y.index_axis_move(Axis(0), 0);
        let w = weights.map(|w| w.index_axis_move(Axis(0), 0));
        vec![Dataset::with_weights_and_names(x, y, w, args.variable_names)]
    };
    let output_strings = |name: &str, obj: Option<&Bound<'_, PyAny>>| -> PyResult<Option<Vec<String>>> {
        obj.map(|obj| {
            if multi_output {
                obj.extract::<Vec<String>>()
            } else {
                obj.extract::<String>().map(|value| vec![value])
            }
            .map_err(|err| {
                value_error(format!(
                    "{name} needs {}: {err}",
                    if multi_output {
                        "one string per output"
                    } else {
                        "a string"
                    }
                ))
            })
        })
        .transpose()
    };
    let y_names = output_strings("y_variable_names", args.y_variable_names)?;
    let y_units = output_strings("y_units", args.y_units)?;
    for (name, list) in [("y_variable_names", &y_names), ("y_units", &y_units)] {
        if list.as_ref().is_some_and(|l| l.len() != nout) {
            return Err(value_error(format!("{name} needs one entry per output ({nout})")));
        }
    }
    if args.x_units.as_ref().is_some_and(|u| u.len() != n_features) {
        return Err(value_error(format!(
            "X_units needs one unit per feature ({n_features})"
        )));
    }
    for (j, dataset) in datasets.iter_mut().enumerate() {
        let mut d = std::mem::replace(dataset, Dataset::new(Array2::zeros((0, 0)), Array1::zeros(0)));
        if args.display_variable_names.is_some() || y_names.is_some() {
            let display = args
                .display_variable_names
                .clone()
                .unwrap_or_else(|| d.display_variable_names.clone());
            d = d
                .with_display_names(display, y_names.as_ref().map(|n| n[j].clone()))
                .map_err(value_error)?;
        }
        if args.x_units.is_some() || y_units.is_some() {
            d = d
                .with_units(args.x_units.as_deref(), y_units.as_ref().map(|u| u[j].as_str()))
                .map_err(value_error)?;
        }
        *dataset = d;
    }

    let keywords = match args.options {
        Some(options) => options
            .iter()
            .map(|(k, v)| {
                let key: String = k.extract()?;
                let v = value(&key, &v)?;
                Ok((key, v))
            })
            .collect::<PyResult<Vec<_>>>()?,
        None => Vec::new(),
    };
    let mut options: SearchOptions<T> = keywords::options::<T>(Keywords::new(keywords.clone())).map_err(value_error)?;
    if options
        .variable_complexities
        .as_ref()
        .is_some_and(|v| v.len() != n_features)
    {
        return Err(value_error(format!(
            "complexity_of_variables needs one entry per feature ({n_features})"
        )));
    }
    if let Some(guesses) = args.guesses {
        options.guesses = keywords::guesses::<T>(value("guesses", guesses)?, nout).map_err(value_error)?;
    }
    let previous = match args.saved_state {
        None => None,
        Some(data) => Some(
            T::saved(data)
                .ok_or_else(|| {
                    value_error(format!(
                        "saved_state was searched in {}, but X is {}",
                        data.dtype(),
                        T::DTYPE
                    ))
                })?
                .result
                .clone(),
        ),
    };
    let stop = ExternalStop::default();
    options.external_stop = Some(stop.clone());

    let (result, interrupted) = run(py, &datasets, &options, previous, &stop)?;
    let result = result.map_err(search_error)?;
    let fronts = result
        .halls_of_fame
        .iter()
        .zip(&datasets)
        .map(|(hall, dataset)| hall_of_fame_rows(hall, dataset, &options))
        .collect();
    Ok(SearchState {
        data: T::wrap(Saved {
            result,
            fronts,
            interrupted,
            n_features,
            keywords: keywords::evaluation_keywords(keywords),
        }),
    })
}

type Outcome<T> = Result<SearchResult<T, <T as Precision>::Ops, D>, String>;

/// Runs the search on another thread without the GIL, checking for signals meanwhile. A
/// `KeyboardInterrupt` stops the search, which still returns what it found; any other exception a
/// signal handler raises stops it too, and is raised once it has.
fn run<T: Stored>(
    py: Python<'_>,
    datasets: &[Dataset<T>],
    options: &SearchOptions<T>,
    previous: Option<SearchResult<T, T::Ops, D>>,
    stop: &ExternalStop,
) -> PyResult<(Outcome<T>, bool)> {
    std::thread::scope(|scope| {
        let (sender, mut receiver) = mpsc::channel::<Outcome<T>>();
        let worker = scope.spawn(move || {
            let outcome = match previous {
                Some(previous) => resume_search(previous, datasets, options),
                None => equation_search(datasets, options),
            };
            let _ = sender.send(outcome);
        });
        let mut interrupted = false;
        let mut raised: Option<PyErr> = None;
        let outcome = loop {
            let (back, received) = py.detach(move || {
                let received = receiver.recv_timeout(SIGNAL_POLL);
                (receiver, received)
            });
            receiver = back;
            if let Err(err) = py.check_signals() {
                stop.request();
                if err.is_instance_of::<PyKeyboardInterrupt>(py) {
                    interrupted = true;
                } else {
                    raised.get_or_insert(err);
                }
            }
            match received {
                Ok(outcome) => break outcome,
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => match worker.join() {
                    Err(panic) => std::panic::resume_unwind(panic),
                    Ok(()) => unreachable!("the search thread sends its outcome before it finishes"),
                },
            }
        };
        match raised {
            Some(err) => Err(err),
            None => Ok((outcome, interrupted)),
        }
    })
}

#[pymodule]
fn symbolic_regression_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SearchState>()?;
    m.add_function(wrap_pyfunction!(equation_search_py, m)?)?;
    m.add_function(wrap_pyfunction!(_load_state, m)?)?;
    m.add(
        "SUPPORTED_OPTIONS",
        PyTuple::new(m.py(), keywords::SUPPORTED_OPTIONS.iter().copied())?,
    )?;
    let julia_only = PyDict::new(m.py());
    for &(name, reason) in keywords::JULIA_ONLY {
        julia_only.set_item(name, reason)?;
    }
    m.add("JULIA_ONLY_OPTIONS", julia_only)?;
    Ok(())
}
