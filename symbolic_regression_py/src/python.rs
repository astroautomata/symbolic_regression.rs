use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::Duration;

use ndarray::{Array1, Array2, ArrayView2, Axis};
use numpy::{PyArray2, PyArrayDyn, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyKeyboardInterrupt, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use serde::{Deserialize, Serialize};
use symbolic_regression::{Dataset, ExternalStop, SearchResult, equation_search, hall_of_fame_rows, resume_search};

use crate::keywords::{self, D, Keywords, Precision, SearchOptions, Value};

/// How often a running search checks for signals such as Ctrl-C.
const SIGNAL_POLL: Duration = Duration::from_millis(100);

fn value_error(message: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(message.to_string())
}

/// A search state for one float type: what the search left, and each output's hall-of-fame rows.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
struct Saved<T: Precision> {
    result: SearchResult<T, T::Ops, D>,
    fronts: Vec<Vec<(usize, f64, String)>>,
    interrupted: bool,
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

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyBytes>,))> {
        let bytes = postcard::to_stdvec(&(env!("CARGO_PKG_VERSION"), &self.data)).map_err(value_error)?;
        let load = py.import("symbolic_regression_rs")?.getattr("_load_state")?;
        Ok((load, (PyBytes::new(py, &bytes),)))
    }
}

/// Unpickles a [`SearchState`].
#[pyfunction]
fn _load_state(bytes: &[u8]) -> PyResult<SearchState> {
    let (version, rest) = postcard::take_from_bytes::<String>(bytes).map_err(value_error)?;
    if version != env!("CARGO_PKG_VERSION") {
        return Err(value_error(format!(
            "this search state was saved by symbolic-regression-rs {version}, which this version ({}) cannot load",
            env!("CARGO_PKG_VERSION")
        )));
    }
    let data = postcard::from_bytes(rest).map_err(value_error)?;
    Ok(SearchState { data })
}

/// A Python keyword value as [`Value`]: lists and tuples become lists, and NumPy arrays and
/// scalars become what their `tolist()` returns.
fn value(key: &str, obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    Ok(if obj.is_none() {
        Value::None
    } else if let Ok(b) = obj.cast::<PyBool>() {
        Value::Bool(b.is_true())
    } else if obj.is_instance_of::<PyInt>() {
        Value::Int(
            obj.extract()
                .map_err(|_| value_error(format!("`{key}` has an integer too large for 64 bits")))?,
        )
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
            let (w, _) = per_output::<T>("weights", w, n_samples)?;
            if w.dim() != y.dim() {
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
    let y_names: Option<Vec<String>> = match args.y_variable_names {
        None => None,
        Some(names) if multi_output => Some(names.extract()?),
        Some(name) => Some(vec![name.extract()?]),
    };
    let y_units: Option<Vec<String>> = match args.y_units {
        None => None,
        Some(units) if multi_output && !units.is_instance_of::<PyString>() => Some(units.extract()?),
        Some(unit) => Some(vec![unit.extract()?; nout]),
    };
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
    let mut options: SearchOptions<T> = keywords::options::<T>(Keywords::new(keywords)).map_err(value_error)?;
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
    let result = result.map_err(value_error)?;
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
            if !interrupted && raised.is_none() {
                if let Err(err) = py.check_signals() {
                    stop.request();
                    if err.is_instance_of::<PyKeyboardInterrupt>(py) {
                        interrupted = true;
                    } else {
                        raised = Some(err);
                    }
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
    Ok(())
}
