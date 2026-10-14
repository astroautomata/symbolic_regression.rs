import csv
import os
import pickle
import signal
import struct
import subprocess
import sys
import threading
import time
import zlib

import numpy as np
import pytest

import symbolic_regression_rs as sr


OPTIONS = dict(
    binary_operators=["+", "*"], unary_operators=[], seed=17,
    deterministic=True, parallelism="serial", populations=1, population_size=16,
    tournament_selection_n=4, ncycles_per_iteration=4, niterations=0, maxsize=10,
    progress=False, verbosity=0, save_to_file=False, input_stream="devnull",
)


def data(dtype=np.float64):
    x = np.arange(32, dtype=dtype)
    X = np.column_stack(((x - 16) / 8, ((x * 13) % 32 - 16) / 8))
    return X, X[:, 0] * X[:, 0] + X[:, 1]


def search(X, y, **options):
    return sr.equation_search(X, y, **(OPTIONS | options))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_evaluate_front_complexities_dtype_and_pickle(dtype):
    X, y = data(dtype)
    state = search(X, y, guesses=["x0", "x0*x0+x1"])
    restored = pickle.loads(pickle.dumps(state, protocol=5))
    for candidate in [state, restored]:
        front = candidate.pareto_fronts[0]
        assert front[-1][1] == 0.0
        np.testing.assert_array_equal(candidate.evaluate(X), y)
        assert candidate.evaluate(X).dtype == dtype
        for complexity, loss, _ in front:
            prediction = candidate.evaluate(X, complexity=complexity)
            assert prediction.shape == (X.shape[0],)
            assert prediction.dtype == dtype
            np.testing.assert_allclose(np.mean((prediction - y) ** 2), loss, rtol=1e-5, atol=1e-6)
            np.testing.assert_array_equal(prediction, restored.evaluate(X, complexity=complexity))
        missing = next(c for c in range(1, 20) if c not in [row[0] for row in front])
        with pytest.raises(ValueError, match="complexity"):
            candidate.evaluate(X, complexity=missing)
        with pytest.raises(ValueError, match="output"):
            candidate.evaluate(X, output=1)
        with pytest.raises(ValueError):
            candidate.evaluate(X, output=-1)
        with pytest.raises(ValueError):
            candidate.evaluate(X, complexity=-1)
        with pytest.raises(ValueError, match="feature"):
            candidate.evaluate(X[:, :1])
        with pytest.raises(ValueError, match="float32|float64"):
            candidate.evaluate(X.astype(np.int64))
        with pytest.raises(ValueError, match="float32|float64"):
            candidate.evaluate(X.astype(np.float64 if dtype == np.float32 else np.float32))
        with pytest.raises(TypeError):
            candidate.evaluate(X, index=0)
        np.testing.assert_array_equal(candidate.evaluate(X[::-1]), y[::-1])
        assert candidate.evaluate(X[:0]).shape == (0,)


def test_evaluate_matches_csv_on_new_inputs(tmp_path):
    X, y = data()
    state = search(X, y, guesses=["x0*x0+x1"], niterations=1,
                   save_to_file=True, output_directory=str(tmp_path), run_id="evaluate-csv")
    new_X = np.array([[-3.0, 0.25], [0.125, -4.0], [5.0, 2.0], [-2.5, 3.0]])
    with (tmp_path / state.run_id / "hall_of_fame.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [(int(row["Complexity"]), float(row["Loss"]), row["Equation"]) for row in rows] == state.pareto_fronts[0]
    for row in rows:
        expected = eval(row["Equation"], {"__builtins__": {}},
                        {"x0": new_X[:, 0], "x1": new_X[:, 1]})
        np.testing.assert_allclose(
            state.evaluate(new_X, complexity=int(row["Complexity"])),
            np.broadcast_to(expected, (len(new_X),)), rtol=1e-12, atol=1e-12,
        )


def test_evaluate_multiple_outputs_and_custom_operators_after_pickle():
    X, _ = data()
    y = np.column_stack((X[:, 0] ** 2, X[:, 1] ** 2))
    state = search(X, y, binary_operators=[], unary_operators=["square(x) = x*x"],
                   guesses=[["square(x0)"], ["square(x1)"]])
    for candidate in [state, pickle.loads(pickle.dumps(state))]:
        for output in range(2):
            np.testing.assert_array_equal(candidate.evaluate(X, output=output), y[:, output])


def test_evaluate_template_parameters_and_indices_after_pickle():
    X, _ = data()
    X[:, 1] = np.arange(len(X)) % 2 + 1
    y = X[:, 0] ** 2 + np.where(X[:, 1] == 1, 2.0, 3.0)
    state = search(
        X, y, unary_operators=["square(x) = x*x"],
        expression_spec=dict(combine="f(x0) + p[x1]", expressions=["f"],
                             variable_names=["x0", "x1"], parameters={"p": 2}),
        guesses=[{"f": "square(#1)", "p": [2.0, 3.0]}],
    )
    for candidate in [state, pickle.loads(pickle.dumps(state))]:
        np.testing.assert_array_equal(candidate.evaluate(X), y)
        new_X = np.array([[-3.0, 2.0], [0.125, 1.0], [5.0, 2.0], [-2.5, 1.0]])
        expected = new_X[:, 0] ** 2 + np.array([3.0, 2.0, 3.0, 2.0])
        np.testing.assert_array_equal(candidate.evaluate(new_X), expected)
        invalid = X.copy()
        invalid[0, 1] = 3
        with pytest.raises(ValueError, match="indices|indexes"):
            candidate.evaluate(invalid)


@pytest.mark.parametrize("template", [False, True])
def test_pickle_evaluation_excludes_machine_local_options(tmp_path, template):
    X, _ = data()
    X[:, 1] = np.arange(len(X)) % 2 + 1
    options = dict(unary_operators=["square(x) = x*x"], guesses=["square(x0)+x1"])
    y = X[:, 0] ** 2 + X[:, 1]
    new_X = np.array([[-3.0, 2.0], [0.125, 1.0], [5.0, 2.0], [-2.5, 1.0]])
    expected = new_X[:, 0] ** 2 + new_X[:, 1]
    if template:
        options.update(
            expression_spec=dict(combine="f(x0) + p[x1]", expressions=["f"],
                                 variable_names=["x0", "x1"], parameters={"p": 2}),
            guesses=[{"f": "square(#1)", "p": [2.0, 3.0]}],
        )
        y = X[:, 0] ** 2 + np.where(X[:, 1] == 1, 2.0, 3.0)
        expected = new_X[:, 0] ** 2 + np.array([3.0, 2.0, 3.0, 2.0])
    output_directory = str(tmp_path / "machine-local-output")
    tracing_file = str(tmp_path / "machine-local-trace.jsonl")
    state = search(
        X, y, **options, niterations=1, save_to_file=True,
        output_directory=output_directory, run_id="portable-state",
        use_tracing=True, tracing_file=tracing_file,
        input_stream="devnull" if os.name == "nt" else "stdin",
    )
    restored = pickle.loads(pickle.dumps(state, protocol=5))
    for candidate in [state, restored]:
        np.testing.assert_array_equal(candidate.evaluate(new_X), expected)
    _, (encoded,) = state.__reduce__()
    encoded = bytes(encoded)
    for option in ["input_stream", "output_directory", "tracing_file"]:
        assert option.encode() not in encoded
    for path in [output_directory, tracing_file]:
        assert path.encode() not in encoded


def test_evaluate_invalid_numeric_result_is_an_error():
    X = np.arange(1, 17, dtype=np.float64).reshape(-1, 1)
    state = search(X, 1 / X[:, 0], binary_operators=[], unary_operators=["inv"], guesses=["inv(x0)"])
    assert state.pareto_fronts[0][-1][1] == 0.0
    with pytest.raises(ValueError, match="invalid|finite"):
        state.evaluate(np.zeros_like(X))


def test_state_rejects_trailing_bytes():
    X, y = data()
    state = search(X, y, guesses=["x0*x0+x1"])
    loader, (encoded,) = state.__reduce__()
    with pytest.raises(ValueError):
        loader(bytes(encoded) + b"\x00")


def test_state_serialized_layout_guard():
    code = r'''
import struct

import numpy as np
import symbolic_regression_rs as sr

X = np.array([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float64)
state = sr.equation_search(
    X, X[:, 0], binary_operators=["+"], unary_operators=[], guesses=["x0"],
    seed=7, deterministic=True, parallelism="serial", populations=1,
    population_size=2, tournament_selection_n=1, ncycles_per_iteration=0,
    niterations=0, maxsize=4, should_optimize_constants=False,
    progress=False, verbosity=0, save_to_file=False, input_stream="devnull",
    run_id="state-format-fixture",
)
assert state.pareto_fronts == [[(1, 0.0, "x0")]]
np.testing.assert_array_equal(state.evaluate(X), X[:, 0])
_, (encoded,) = state.__reduce__()
encoded = bytes(encoded)
print(len(encoded), *struct.unpack("<II", encoded[:8]))
'''
    observed = tuple(map(int, subprocess.check_output([sys.executable, "-c", code], text=True).split()))
    assert observed == (559, 2, 1831551450), (
        "Serialized search state changed. If the serialized layout changed, bump STATE_FORMAT "
        "before recording the new fixture length and checksum. "
        f"Observed (length, format, checksum): {observed}"
    )


def test_state_header_checksum_and_strict_payload():
    X, y = data()
    state = search(X, y, guesses=["x0*x0+x1"])
    loader, (encoded,) = state.__reduce__()
    encoded = bytes(encoded)
    format_version, checksum = struct.unpack("<II", encoded[:8])
    assert format_version == 2
    assert checksum == zlib.crc32(encoded[8:])
    restored = loader(encoded)
    assert pickle.dumps(restored, protocol=5) == pickle.dumps(state, protocol=5)
    assert restored.pareto_fronts == state.pareto_fronts
    corrupt = bytearray(encoded)
    corrupt[-1] ^= 1
    extended = encoded + b"\x00"
    extended_checked = struct.pack("<II", format_version, zlib.crc32(extended[8:])) + extended[8:]
    wrong_format = struct.pack("<I", format_version + 1) + encoded[4:]
    wrong_version = bytearray(encoded[8:])
    assert wrong_version[0] > 0
    wrong_version[1] ^= 1
    wrong_version = struct.pack("<II", format_version, zlib.crc32(wrong_version)) + wrong_version
    for payload in [bytes(corrupt), encoded[:4], encoded[:-1], extended,
                    extended_checked, wrong_format, wrong_version, encoded[8:]]:
        with pytest.raises(ValueError):
            loader(payload)


@pytest.mark.parametrize("keyword", ["y_variable_names", "y_units"])
@pytest.mark.parametrize("value", [123, [123], ["one"], "one", ["one", "two", "three"]])
def test_output_metadata_errors_are_value_errors(keyword, value):
    X, y = data()
    with pytest.raises(ValueError):
        search(X, np.column_stack((y, y)), **{keyword: value})


@pytest.mark.parametrize("keyword", ["y_variable_names", "y_units"])
def test_single_output_metadata_extraction_is_value_error(keyword):
    X, y = data()
    with pytest.raises(ValueError):
        search(X, y, **{keyword: 123})
    state = search(X, y[:, None], **{keyword: ["1" if keyword == "y_units" else "one"]})
    assert len(state.pareto_fronts) == 1


def test_weights_preserve_y_dimensionality():
    X, y = data()
    with pytest.raises(ValueError, match="shape"):
        search(X, y[:, None], weights=np.ones_like(y))
    with pytest.raises(ValueError, match="shape"):
        search(X, y, weights=np.ones_like(y[:, None]))


@pytest.mark.parametrize("tracing", [False, True])
def test_output_io_errors_are_os_errors(tmp_path, tracing):
    X, y = data()
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    options = dict(use_tracing=True, tracing_file=str(blocker / "trace.jsonl")) if tracing else dict(
        output_directory=str(blocker), save_to_file=True,
    )
    with pytest.raises(OSError, match="cannot (create|open|write)"):
        search(X, y, niterations=1, **options)


def test_seed_supports_full_unsigned_range():
    X, y = data()
    assert search(X, y, seed=2**64 - 1).pareto_fronts
    with pytest.raises(ValueError, match="64 bits"):
        search(X, y, seed=2**64)


def test_keyword_metadata_and_none_validation():
    assert isinstance(sr.SUPPORTED_OPTIONS, tuple)
    assert len(sr.SUPPORTED_OPTIONS) == len(set(sr.SUPPORTED_OPTIONS))
    assert "seed" in sr.SUPPORTED_OPTIONS
    assert "expression_spec" in sr.SUPPORTED_OPTIONS
    assert isinstance(sr.JULIA_ONLY_OPTIONS, dict)
    assert all(isinstance(reason, str) and reason for reason in sr.JULIA_ONLY_OPTIONS.values())
    assert {"extra", "multithreaded"} <= sr.JULIA_ONLY_OPTIONS.keys()
    X, y = data()
    for keyword in ["not_an_option", *sr.JULIA_ONLY_OPTIONS]:
        with pytest.raises(ValueError):
            search(X, y, **{keyword: None})


def test_two_interrupts_are_consumed_while_search_drains():
    code = r'''
import signal, sys
import numpy as np
from symbolic_regression_rs import equation_search
count = 0
def interrupt(signum, frame):
    global count
    count += 1
    print("SIGNAL", count, flush=True)
    raise KeyboardInterrupt
signal.signal(signal.SIGINT, interrupt)
if hasattr(signal, "SIGBREAK"):
    signal.signal(signal.SIGBREAK, interrupt)
X = np.linspace(0.1, 1, 100000).reshape(-1, 1)
slow = "function loss(prediction, target); v = prediction; "
slow += "v = sin(v); " * 48
slow += "return (prediction-target)^2 + v; end"
print("READY", flush=True)
state = equation_search(
    X, X[:, 0], binary_operators=["+", "*"], unary_operators=[],
    elementwise_loss=slow,
    populations=1, population_size=64, tournament_selection_n=4,
    niterations=100000, ncycles_per_iteration=1, parallelism="serial",
    progress=False, verbosity=0, save_to_file=False, input_stream="devnull",
)
assert state.interrupted
assert count == 2, count
print("RETURNED", flush=True)
'''
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen([sys.executable, "-u", "-c", code], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, creationflags=creationflags)
    lines = []
    ready = threading.Event()
    first_signal = threading.Event()

    def read_stdout():
        for line in process.stdout:
            lines.append(line.strip())
            if line.strip() == "READY":
                ready.set()
            if line.strip() == "SIGNAL 1":
                first_signal.set()

    reader = threading.Thread(target=read_stdout, daemon=True)
    reader.start()
    try:
        assert ready.wait(20), lines
        time.sleep(0.25)
        interrupt = signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT
        process.send_signal(interrupt)
        if not first_signal.wait(10):
            returncode = process.poll()
            if returncode is None:
                process.kill()
                process.wait()
            reader.join(timeout=5)
            pytest.fail(f"first signal was not handled: returncode={returncode}, stdout={lines}, stderr={process.stderr.read()}")
        process.send_signal(interrupt)
        process.wait(timeout=90)
        reader.join(timeout=5)
        stderr = process.stderr.read()
        assert process.returncode == 0, (lines, stderr)
        assert lines[-3:] == ["SIGNAL 1", "SIGNAL 2", "RETURNED"]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        reader.join(timeout=5)
        process.stdout.close()
        process.stderr.close()
