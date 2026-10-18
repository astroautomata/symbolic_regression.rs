import csv
import json
import os
import pickle
import signal
import struct
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from symbolic_regression_rs import SearchState, equation_search


OPTIONS = dict(
    binary_operators=["+", "*"],
    unary_operators=[],
    seed=17,
    deterministic=True,
    parallelism="serial",
    populations=2,
    population_size=24,
    tournament_selection_n=4,
    ncycles_per_iteration=64,
    niterations=8,
    maxsize=10,
    progress=False,
    verbosity=0,
    save_to_file=False,
    input_stream="devnull",
    run_id="python-parity",
)


def data(dtype=np.float64):
    i = np.arange(64, dtype=dtype)
    X = np.column_stack(((i - 32) / 16, ((i * 13) % 64 - 32) / 16))
    return X, X[:, 0] * X[:, 0] + X[:, 1]


def search(X, y, **options):
    return equation_search(X, y, **(OPTIONS | options))


def exact_fronts(state):
    return [
        [(complexity, struct.pack("!d", loss), equation) for complexity, loss, equation in front]
        for front in state.pareto_fronts
    ]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_builtin_target_recovery(dtype):
    X, _ = data(dtype)
    y = X[:, 0] * X[:, 0]
    state = search(X, y, binary_operators=["*"], niterations=12)
    assert isinstance(state, SearchState)
    assert len(state.pareto_fronts) == 1
    assert min(row[1] for row in state.pareto_fronts[0]) < 1e-10
    assert state.run_id == "python-parity"
    assert state.total_evals > 0
    assert state.warnings == []
    assert state.interrupted is False
    assert [row[0] for row in state.pareto_fronts[0]] == sorted(
        row[0] for row in state.pareto_fronts[0]
    )


def test_custom_dsl_operator_recovery():
    X, _ = data()
    y = X[:, 0] * X[:, 0]
    state = search(
        X, y, binary_operators=[], unary_operators=["square(x) = x*x"], niterations=12
    )
    best = min(state.pareto_fronts[0], key=lambda row: row[1])
    assert best[1] < 1e-12
    assert "square(x0)" in best[2]


@pytest.mark.parametrize(
    "loss,weighted",
    [
        ("L1DistLoss()", False),
        ("loss(prediction, target) = (prediction - target)^2 + 0.25", False),
        ("loss(prediction, target, weight) = weight*(prediction - target)^2 + 0.25", True),
    ],
)
def test_string_loss(loss, weighted):
    X, y = data()
    options = dict(elementwise_loss=loss, guesses=["x0*x0+x1"], niterations=0)
    if weighted:
        options["weights"] = np.full_like(y, 2)
    state = search(X, y, **options)
    expected = 0.0 if loss == "L1DistLoss()" else (0.125 if weighted else 0.25)
    assert min(row[1] for row in state.pareto_fronts[0]) == expected


def test_string_early_stop_returns_before_full_search():
    X, y = data()
    options = dict(guesses=["x0*x0+x1"], niterations=12)
    full = search(X, y, **options)
    stopped = search(
        X, y, **options,
        early_stop_condition="stop(loss, complexity) = loss < 1e-12 && complexity <= 6",
    )
    assert min(row[1] for row in stopped.pareto_fronts[0]) == 0.0
    assert stopped.total_evals < full.total_evals
    assert stopped.interrupted is False


def test_template_and_parameter_guesses():
    X, _ = data()
    y = X[:, 0] * X[:, 0] + 2
    state = search(
        X, y,
        expression_spec=dict(
            combine="f(x0) + p[1]",
            expressions=["f"], variable_names=["x0", "x1"], parameters={"p": 1},
        ),
        guesses=[{"f": "#1*#1", "p": [2.0]}],
        niterations=0,
    )
    best = min(state.pareto_fronts[0], key=lambda row: row[1])
    assert best[1] == 0.0
    assert "f =" in best[2]
    assert "p = [2" in best[2]


def test_dimensionally_invalid_equations_are_penalized():
    X, _ = data()
    y = X[:, 0].copy()
    options = dict(guesses=["x0"], niterations=0, dimensionless_constants_only=True)
    plain = search(X, y, **options)
    penalized = search(
        X, y, **options, X_units=["m", "m"], y_units="s", dimensional_constraint_penalty=1000.0
    )
    assert min(row[1] for row in plain.pareto_fronts[0]) == 0.0
    assert penalized.pareto_fronts[0]
    assert min(row[1] for row in penalized.pareto_fronts[0]) >= 1000.0


def test_multi_output_guesses_names_and_weights(tmp_path, capfd):
    X, y = data()
    Y = np.column_stack((y, X[:, 0] * X[:, 1]))
    state = search(
        X, Y, weights=np.ones_like(Y), variable_names=["a", "b"],
        display_variable_names=["A", "B"], y_variable_names=["u", "v"],
        guesses=[["a*a+b"], ["a*b"]], niterations=1, verbosity=1,
        save_to_file=True, output_directory=str(tmp_path),
    )
    assert len(state.pareto_fronts) == 2
    assert [min(row[1] for row in front) for front in state.pareto_fronts] == [0.0, 0.0]
    assert all("a" in front[-1][2] for front in state.pareto_fronts)
    terminal = capfd.readouterr().out
    first_output, second_output = terminal.split("Best equations for output 2")
    assert "Best equations for output 1" in first_output
    assert any("u = " in line and "A" in line and "B" in line for line in first_output.splitlines())
    assert any("v = " in line and "A" in line and "B" in line for line in second_output.splitlines())
    for output, front in enumerate(state.pareto_fronts, start=1):
        path = tmp_path / state.run_id / f"hall_of_fame_output{output}.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert [(int(r["Complexity"]), float(r["Loss"]), r["Equation"]) for r in rows] == front
        assert "a" in rows[-1]["Equation"] and "b" in rows[-1]["Equation"]
        assert all("A" not in r["Equation"] and "B" not in r["Equation"] for r in rows)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", ["fortran", "strided"])
def test_X_layout_preserves_exact_search(dtype, layout):
    X, y = data(dtype)
    expected = search(X, y)
    if layout == "fortran":
        other = np.asfortranarray(X)
        assert other.flags.f_contiguous and not other.flags.c_contiguous
    else:
        storage = np.empty((X.shape[0] * 2, X.shape[1] * 2), dtype=dtype)
        other = storage[::2, ::2]
        other[:] = X
        assert not other.flags.c_contiguous and not other.flags.f_contiguous
    actual = search(other, y)
    assert exact_fronts(actual) == exact_fronts(expected)
    assert actual.total_evals == expected.total_evals



def test_saved_state_incompatible_resume_preserves_engine_error():
    X, y = data()
    state = search(X, y, niterations=1)
    with pytest.raises(ValueError) as error:
        search(X, y, saved_state=state, population_size=25)
    assert str(error.value) == (
        "Warm start incompatible due to changed field(s): population_size. "
        "Start a new search instead."
    )


def test_pickle_preserves_exact_state_and_deterministic_resume():
    X, y = data()
    first = search(X, y, niterations=3)
    encoded = pickle.dumps(first, protocol=5)
    restored = pickle.loads(encoded)
    assert pickle.dumps(restored, protocol=5) == encoded
    assert exact_fronts(restored) == exact_fronts(first)
    assert restored.total_evals == first.total_evals
    assert restored.run_id == first.run_id
    resumed = search(X, y, saved_state=first, niterations=4)
    resumed_pickle = search(X, y, saved_state=restored, niterations=4)
    assert exact_fronts(resumed_pickle) == exact_fronts(resumed)
    assert resumed_pickle.total_evals == resumed.total_evals > first.total_evals
    assert resumed_pickle.run_id == resumed.run_id == first.run_id
    assert resumed_pickle.warnings == resumed.warnings


@pytest.mark.parametrize("seed", [0, 1, 17])
def test_pickle_resume_matches_in_a_fresh_process(tmp_path, seed):
    X, y = data()
    options = OPTIONS | dict(seed=seed, ncycles_per_iteration=8, niterations=1)
    first = equation_search(X, y, **options)
    saved_path = tmp_path / "saved.pickle"
    saved_path.write_bytes(pickle.dumps(first, protocol=5))
    options["niterations"] = 12
    expected = equation_search(X, y, saved_state=first, **options)
    result_path = tmp_path / "resumed.pickle"
    code = """
import json, pickle, sys
import numpy as np
from symbolic_regression_rs import equation_search
i = np.arange(64, dtype=np.float64)
X = np.column_stack(((i - 32) / 16, ((i * 13) % 64 - 32) / 16))
y = X[:, 0] * X[:, 0] + X[:, 1]
with open(sys.argv[1], 'rb') as stream:
    saved = pickle.load(stream)
state = equation_search(X, y, saved_state=saved, **json.loads(sys.argv[3]))
with open(sys.argv[2], 'wb') as stream:
    pickle.dump(state, stream, protocol=5)
"""
    subprocess.run(
        [sys.executable, "-c", code, str(saved_path), str(result_path), json.dumps(options)],
        check=True, capture_output=True, text=True, timeout=30,
    )
    resumed = pickle.loads(result_path.read_bytes())
    assert exact_fronts(resumed) == exact_fronts(expected)
    assert resumed.total_evals == expected.total_evals
    assert resumed.run_id == expected.run_id
    assert resumed.warnings == expected.warnings


def test_csv_header_rows_and_backup(tmp_path):
    X, y = data()
    state = search(
        X, y, output_directory=str(tmp_path), run_id="csv", save_to_file=True,
        guesses=["x0*x0+x1"], niterations=1,
    )
    path = tmp_path / state.run_id / "hall_of_fame.csv"
    assert path.read_bytes().splitlines()[0] == b"Complexity,Loss,Equation"
    assert path.read_bytes() == path.with_suffix(".csv.bak").read_bytes()
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [(int(r["Complexity"]), float(r["Loss"]), r["Equation"]) for r in rows] == state.pareto_fronts[0]


@pytest.mark.parametrize("dtype", [np.int64, np.float16, np.complex128, object])
def test_invalid_X_dtype(dtype):
    X, y = data()
    with pytest.raises(ValueError, match="X must be a 2-D float32 or float64 NumPy array"):
        search(X.astype(dtype), y)


@pytest.mark.parametrize("name", ["y", "weights"])
@pytest.mark.parametrize("dtype", [np.int64, np.float16, np.complex128, object])
def test_invalid_output_or_weight_dtype(name, dtype):
    X, y = data()
    with pytest.raises(ValueError, match=f"{name} must be a float64 NumPy array, as X is"):
        if name == "y":
            search(X, y.astype(dtype))
        else:
            search(X, y, weights=np.ones(y.shape, dtype=dtype))


@pytest.mark.parametrize("name", ["y", "weights"])
def test_mismatched_float_dtypes(name):
    X, y = data(np.float32)
    with pytest.raises(ValueError, match=f"{name} must be a float32 NumPy array, as X is"):
        if name == "y":
            search(X, y.astype(np.float64))
        else:
            search(X, y, weights=np.ones(y.shape, dtype=np.float64))


def test_saved_state_dtype_mismatch():
    X, y = data(np.float32)
    state = search(X, y, niterations=0)
    with pytest.raises(ValueError, match="saved_state was searched in float32, but X is float64"):
        search(X.astype(np.float64), y.astype(np.float64), saved_state=state)


@pytest.mark.parametrize(
    "shape_X,shape_y",
    [((0, 2), (0,)), ((3, 0), (3,)), ((3, 2), (3, 0))],
)
def test_empty_array_dimensions(shape_X, shape_y):
    with pytest.raises(ValueError, match="at least one|non.?empty"):
        search(np.empty(shape_X), np.empty(shape_y))


def test_input_shapes():
    X, y = data()
    with pytest.raises(ValueError, match="X must be a 2-D"):
        search(X[:, 0], y)
    with pytest.raises(ValueError, match="y must have shape"):
        search(X, y[:-1])
    with pytest.raises(ValueError, match="weights must have the shape of y"):
        search(X, y, weights=np.ones((len(y), 2)))


def test_variable_complexity_count_mismatch():
    X, y = data()
    with pytest.raises(ValueError, match="complexity_of_variables"):
        search(X, y, complexity_of_variables=[1.0])


@pytest.mark.parametrize(
    "option,message",
    [
        ({"not_an_option": 1}, "unknown option `not_an_option`"),
        ({"turbo": False}, "`turbo` is a SymbolicRegression.jl option that the Rust engine does not support: it selects LoopVectorization.jl"),
        ({"loss_function": "f(x, y) = x"}, "`loss_function` is a SymbolicRegression.jl option that the Rust engine does not support: it takes a Julia function of the whole expression; use `elementwise_loss`"),
    ],
)
def test_option_error_strings_are_preserved(option, message):
    X, y = data()
    with pytest.raises(ValueError) as error:
        search(X, y, **option)
    assert str(error.value) == message


@pytest.mark.parametrize("individual", ["binary_operators", "unary_operators"])
def test_operators_parameter_conflicts(individual):
    X, y = data()
    options = {k: v for k, v in OPTIONS.items() if k not in ("binary_operators", "unary_operators")}
    with pytest.raises(ValueError) as error:
        equation_search(X, y, **options, operators={1: ["sin", "cos"], 2: ["+", "*", "-"]}, **{individual: ["+"] if individual == "binary_operators" else ["sin"]})
    assert str(error.value) == "`operators` cannot be combined with `binary_operators` or `unary_operators`"


def test_another_python_thread_progresses_during_search():
    X, y = data()
    finished = threading.Event()
    ticks = []

    def heartbeat():
        while not finished.wait(0.005):
            ticks.append(time.monotonic())

    worker = threading.Thread(target=heartbeat)
    worker.start()
    started = time.monotonic()
    try:
        search(X, y, niterations=1_000_000, timeout_in_seconds=1.0)
        ended = time.monotonic()
    finally:
        finished.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert ended - started >= 0.5
    assert sum(started + 0.15 < tick < ended - 0.15 for tick in ticks) >= 10


def test_interrupt_returns_partial_search_state(tmp_path):
    code = """
import pickle, signal, sys
if sys.platform == "win32":
    signal.signal(signal.SIGBREAK, signal.default_int_handler)
import numpy as np
from symbolic_regression_rs import equation_search
X = np.arange(128, dtype=np.float64).reshape(64, 2) / 32
print('ready', flush=True)
state = equation_search(X, X[:, 0]**2 + X[:, 1], binary_operators=['+', '*'],
    seed=17, deterministic=True, parallelism='serial', populations=2,
    population_size=24, tournament_selection_n=4, ncycles_per_iteration=64,
    niterations=1000000, maxsize=10, progress=False, verbosity=0,
    save_to_file=False, input_stream='devnull', timeout_in_seconds=10.0)
with open(sys.argv[1], 'wb') as stream:
    pickle.dump(state, stream, protocol=5)
"""
    path = tmp_path / "partial.pickle"
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(path)], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    watchdog = threading.Timer(30, process.kill)
    watchdog.start()
    try:
        assert process.stdout.readline().strip() == "ready"
        time.sleep(0.5)
        process.send_signal(signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGINT)
        stdout, stderr = process.communicate(timeout=20)
    finally:
        watchdog.cancel()
        if process.poll() is None:
            process.kill()
            process.wait()
    assert process.returncode == 0, stdout + stderr
    state = pickle.loads(path.read_bytes())
    assert state.interrupted is True
    assert state.total_evals > 0
    assert state.pareto_fronts[0]
    X = np.arange(128, dtype=np.float64).reshape(64, 2) / 32
    resumed = search(
        X, X[:, 0]**2 + X[:, 1], saved_state=state, niterations=1, run_id=state.run_id
    )
    assert resumed.total_evals > state.total_evals
    assert resumed.run_id == state.run_id
    assert resumed.interrupted is False


def test_python_front_matches_rust_example_exactly():
    executable = os.environ.get("SYMBOLIC_REGRESSION_RUST_EXAMPLE")
    assert executable, "Set SYMBOLIC_REGRESSION_RUST_EXAMPLE to the built python_parity executable"
    completed = subprocess.run([executable], check=True, capture_output=True, text=True, timeout=60)
    rust = json.loads(completed.stdout)
    X, y = data()
    python = search(X, y)
    expected = [[(row[0], bytes.fromhex(row[1]), row[2]) for row in front] for front in rust["pareto_fronts"]]
    assert exact_fronts(python) == expected
    assert python.total_evals == rust["total_evals"]
    assert python.run_id == rust["run_id"]
