<p align="center">
  <img
    src="https://raw.githubusercontent.com/MilesCranmer/SymbolicRegression.jl/refs/heads/master/docs/src/assets/logo.png"
    height="200"
    alt="SymbolicRegression.jl logo"
  />
</p>

<h1 align="center">symbolic_regression.rs</h1>

Rust port of [`SymbolicRegression.jl`](https://github.com/MilesCranmer/SymbolicRegression.jl) with support for WebAssembly.

Try out a fully browser-based demo of WebAssembly-compiled symbolic regression [here](https://astroautomata.com/symbolic_regression.rs/).

> [!WARNING]
> This package is an **experiment**. The API is not stabilized, and you should expect large breaking changes in the syntax.
> This library is not ready for use.

This workspace contains four crates:

| Crate | crates.io | CI |
|---|---|---|
| [`symbolic_regression`](./symbolic_regression) | [![crates.io](https://img.shields.io/crates/v/symbolic_regression)](https://crates.io/crates/symbolic_regression) | [![CI (symbolic_regression)](https://github.com/astro-automata/symbolic_regression.rs/actions/workflows/ci-symbolic-regression.yml/badge.svg?branch=main)](https://github.com/astro-automata/symbolic_regression.rs/actions/workflows/ci-symbolic-regression.yml) |
| [`dynamic_expressions`](./dynamic_expressions) | [![crates.io](https://img.shields.io/crates/v/dynamic_expressions)](https://crates.io/crates/dynamic_expressions) | [![CI (dynamic_expressions)](https://github.com/astro-automata/symbolic_regression.rs/actions/workflows/ci-dynamic-expressions.yml/badge.svg?branch=main)](https://github.com/astro-automata/symbolic_regression.rs/actions/workflows/ci-dynamic-expressions.yml) |
| [`symbolic_regression_wasm`](./symbolic_regression_wasm) | [![crates.io](https://img.shields.io/crates/v/symbolic_regression_wasm)](https://crates.io/crates/symbolic_regression_wasm) | [![CI (Web UI)](https://github.com/astro-automata/symbolic_regression.rs/actions/workflows/ci-web.yml/badge.svg?branch=main)](https://github.com/astro-automata/symbolic_regression.rs/actions/workflows/ci-web.yml) |
| [`symbolic_regression_py`](./symbolic_regression_py) | Python wheel: `symbolic-regression-rs` | [![CI (Python)](https://github.com/astroautomata/symbolic_regression.rs/actions/workflows/ci-symbolic-regression.yml/badge.svg?branch=main)](https://github.com/astroautomata/symbolic_regression.rs/actions/workflows/ci-symbolic-regression.yml) |

## Python

Build and install the standalone Python wheel from this checkout with Rust, Python 3.10 or later, and [maturin](https://www.maturin.rs/):

```bash
pip install "maturin>=1.9.4,<2"
maturin build --release --manifest-path symbolic_regression_py/Cargo.toml --out wheelhouse
pip install wheelhouse/*.whl
```

The distribution is `symbolic-regression-rs`; import it as `symbolic_regression_rs`. The wheel uses Python's stable ABI and needs no Julia installation. `X` has shape `(n_samples, n_features)` and `y` has shape `(n_samples,)` or `(n_samples, n_outputs)`. Both arrays, and optional `weights`, must have the same NumPy `float32` or `float64` dtype. Weights must match both the shape and number of dimensions of `y`. For two-dimensional `y`, `y_variable_names` and `y_units` each require one string per output; for one-dimensional `y`, each takes a single string.

```python
import numpy as np
from symbolic_regression_rs import equation_search

X = np.random.default_rng(0).uniform(-2, 2, (64, 2))
y = X[:, 0]**2 + X[:, 1]
state = equation_search(
    X, y, binary_operators=["+", "*"], niterations=40, populations=4,
    seed=0, deterministic=True, parallelism="serial", save_to_file=False,
)
print(state.pareto_fronts[0])  # (complexity, loss, equation) rows
prediction = state.evaluate(X)  # Last Pareto row, with the state's dtype
```

Keywords follow SymbolicRegression.jl's `Options` and `equation_search`, including string operators, `elementwise_loss`, `early_stop_condition`, units, and guesses. `symbolic_regression_rs.SUPPORTED_OPTIONS` lists accepted option names, and `JULIA_ONLY_OPTIONS` maps rejected names to reasons. Julia-only and unknown options raise `ValueError`, including when their value is `None`. Output-file failures raise `OSError`. Pickle a `SearchState` to save it, then pass the restored state as `saved_state` with compatible options to continue. Pickles require the same package version and state format; damaged or extended payloads are rejected. Ctrl-C returns partial results with `state.interrupted=True`, and further interrupts are consumed while the search finishes its running work. See [the Python example](./symbolic_regression_py/examples/example.py).

`state.evaluate(X, output=0, complexity=None)` evaluates a stored expression directly. `X` must have the state's dtype and feature count. `output` selects an output, and `complexity` selects an exact complexity from that output's returned Pareto rows; omitting it selects the last row, which has the lowest loss. Custom operators, template formulas, and fitted template parameters survive pickling. Only operator and template definitions are saved for evaluation; machine-local input and output settings are excluded. Invalid template indices and nonfinite predictions raise `ValueError`.

Rust-specific settings are `output_style`, `use_baseline`, and `optimizer_g_calls_limit`, `optimizer_g_abstol`, `optimizer_x_abstol`, and `optimizer_f_reltol`; the optimizer settings also accept the corresponding names inside `optimizer_options`.

## Low-level API

Execute `examples/example.rs`, which is the standard example from the [`SymbolicRegression.jl` README](https://github.com/MilesCranmer/SymbolicRegression.jl).

```bash
cargo run -p symbolic_regression --example example --release
```


The code executed is:

```rust
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use symbolic_regression::prelude::*;

// Mirrors `SymbolicRegression.jl/example.jl`.

fn main() {
    const D: usize = 3;
    let n_features = 5;
    let n_rows = 100;

    let mut rng = StdRng::seed_from_u64(0);

    let mut x = Array2::zeros((n_features, n_rows));
    let mut y = Array1::zeros(n_rows);

    for i in 0..n_rows {
        for j in 0..n_features {
            x[(j, i)] = rng.random_range(-3.0f32..3.0f32);
        }
        let x1 = x[(1, i)];
        let x4 = x[(4, i)];
        y[i] = 2.0 * x4.cos() + x1 * x1 - 2.0;
    }

    let dataset = Dataset::new(x, y);

    let operators = BuiltinOpsF32::new().from_names(["cos", "exp", "sin", "+", "sub", "*", "/"]).unwrap();

    let options = Options::<f32, BuiltinOpsF32, D> {
        operators,
        niterations: 200,
        ..Default::default()
    };

    let result = equation_search::<f32, BuiltinOpsF32, D>(std::slice::from_ref(&dataset), &options).unwrap();
    let dominating = result.halls_of_fame[0].pareto_front();

    println!("Final Pareto front:");
    println!("Complexity\tMSE\tEquation");
    for member in dominating {
        let equation = member.expr.string(&options, Default::default());
        println!("{}\t{}\t{}", member.complexity, member.loss, equation);
    }
    // To evaluate the expression, use:
    /*
        let tree = dominating
            .last()
            .unwrap()
            .expr.trees[0]
            .clone();
        let _ = eval_tree_array::<f32, BuiltinOpsF32, D>(
            &tree,
            &options.operators.set,
            dataset.x.view(),
            &EvalOptions::default(),
        );
    */
}
```

Searches write `hall_of_fame.csv` and an identical `.bak` file beneath
`output_directory/run_id` after each population cycle and at completion. Multi-output
searches write `hall_of_fame_output1.csv`, `hall_of_fame_output2.csv`, and so on.
`output_directory` defaults to `outputs`; `SearchResult::run_id` and
`SearchEngine::run_id()` expose the generated identifier. Set `save_to_file: false`
to disable file output. WebAssembly has no filesystem: there `save_to_file` defaults to
false, and a search with `save_to_file` or `use_tracing` set fails at setup.

`SrLogger::new(callback)` records population complexities, Pareto
summaries, equations, and evaluation counts, with each population as its cycle left it,
before migration. It logs step zero, every 100 calls, and the final cycle of a search that
runs to completion; a stopped search logs no extra entry. `SrLogger::with_interval` selects
another interval; nonpositive intervals disable logging. Clones share the logger's step counter. Set `verbosity`
to `Some(0)` to suppress printing, or leave it `None` for normal output.
`Dataset::with_display_names` sets terminal feature and target names while CSV files
keep `variable_names`. `terminal_width` controls the terminal table width.
`ExternalStop::request()` asks one search to stop gracefully at its next completed
population cycle. Unix searches monitor stdin for `q` followed by Enter or a
control-C byte, and stop monitoring once stdin closes or fails; `input_stream: None`
disables monitoring. `InputStream::channel()`
and `InputStream::nonblocking()` accept other input sources on every platform.

## Custom operators

Define custom operators with `op!`, then build an operator set with `opset!`:

```rust
use symbolic_regression::prelude::*;

op!(Square for f64 {
    eval: |[x]| { x * x },
    partial: |[x], _idx| { 2.0 * x },
});

op!(Exp for f64 {
    eval: |[x]| { x.exp() },
    partial: |[x], _idx| { x.exp() },
});

op!(Add for f64 {
    infix: "+",
    commutative: true,
    associative: true,
    eval: |[x, y]| { x + y },
    partial: |[_x, _y], _idx| { 1.0 },
});

op!(Sub for f64 {
    infix: "-",    // optional
    complexity: 2, // optional
    eval: |[x, y]| { x - y },
    partial: |[_x, _y], idx| { if idx == 0 { 1.0 } else { -1.0 } },
});

opset! {
    pub MyOps for f64 {
        Square,
        Exp,
        Add,
        Sub,
    }
}

let operators = MyOps::new().from_names(["square", "exp", "add", "sub"]).unwrap();
let options = Options::<f64, MyOps, _> { operators, ..Default::default() };
```

Operators can also be written as strings in the Julia syntax that PySR uses, such as `"inv(x) = 1/x"` or `"f(x, y) = x * exp(-y^2)"`. They are parsed, differentiated, and compiled once, when the operator set is built; errors such as unknown functions are reported then. `parse_operator_tokens` accepts built-in names and definitions together, and the definitions become the `custom` table of the operator set:

```rust
use symbolic_regression::prelude::*;

let (custom, names) = parse_operator_tokens::<f64, _>(["+", "*", "cos", "inv(x) = 1/x"]).unwrap();
let operator_set = BuiltinOpsF64::new().with_custom(custom);
let options = Options::<f64, _, 2> {
    operators: operator_set.from_names(&names).unwrap(),
    ..Default::default()
};
```

The WASM binding (`WasmSearch`) accepts the same definitions in its operator lists.

## Losses and early stopping

`Options::loss` defaults to `L2DistLoss()`. `parse_loss` reads PySR's `elementwise_loss` strings: a built-in loss from SymbolicRegression.jl by its constructor, such as `"L1DistLoss()"`, `"HuberLoss(1.5)"`, `"LPDistLoss{3}()"`, or a classification loss such as `"L1HingeLoss()"` (see `BuiltinLoss`; `BuiltinLoss::into_loss` builds the same loss without a string), or a definition taking `(prediction, target)`, or `(prediction, target, weight)` for datasets with weights. Definitions are differentiated once for constant optimization. `parse_early_stop` reads `early_stop_condition`: a loss threshold such as `"1e-6"`, or a definition of `(loss, complexity)` that returns Bool. `loss_scale` sets how the Pareto front is scored: `LossScale::Log` (the default) needs non-negative losses, and `LossScale::Linear` allows negative ones. As in SymbolicRegression.jl, a negative loss on the Pareto front under `LossScale::Log` is an error: `SearchEngine::step`, `equation_search`, and `SearchEngine::best` return it. `SearchResult::best` and `SearchEngine::best` are the equation `choose_best` picks under `loss_scale`.

```rust
use symbolic_regression::prelude::*;

let options = Options::<f64, BuiltinOpsF64, 2> {
    loss: parse_loss("myloss(x, y, w) = w * abs(x - y)").unwrap(),
    early_stop_condition: Some(parse_early_stop("stop_if(loss, complexity) = loss < 1e-6 && complexity < 10").unwrap()),
    loss_scale: LossScale::Linear,
    ..Default::default()
};
```

`WasmSearch` takes the same strings as its `loss` and `early_stop_condition` options, and `loss_scale` as `"log"` or `"linear"`.

`Options::default()` uses the current SymbolicRegression.jl hyperparameters. Rust operator sets are generic, so callers select operators explicitly. `seed: None` draws a fresh random seed for each search; `seed: Some(value)` repeats its random stream. `Options::with_defaults(DefaultsProfile::Legacy)` selects the pre-1.0 profile, and `DefaultsProfile::V1` selects the 1.x profile. `Batching::Auto` samples datasets with more than 1,000 rows; an unset `batch_size` selects 128, 256, or 512 rows as the dataset grows. An unset `maxdepth` follows `maxsize`, and unset `should_simplify` enables simplification only when no constraints are configured. `optimizer_algorithm` selects `OptimizerAlgorithm::BFGS` or `OptimizerAlgorithm::NelderMead`. Weighted `mutations` and `crossovers` replace defaults of the same built-in kind; `default_mutations: Some(vec![])` and `default_crossovers: Some(vec![])` disable their defaults.

Search halls of fame retain only expressions within `maxsize` that satisfy the configured constraints, including every template subexpression.

## Template expressions

`Options::template` fixes the shape of the equation, as PySR's `TemplateExpressionSpec` does. The combine formula is a string in the same syntax; it calls subexpressions that the search evolves, takes derivatives with `D(f, i)`, and indexes learnable parameter vectors. Indexing a parameter vector by a category column (`p[class]`, with classes numbered from 1) gives each class its own constants, which replaces SymbolicRegression.jl's former `ParametricExpression`:

```rust
use symbolic_regression::prelude::*;

let template = TemplateStructure::new(
    "f(x1, x2, p1[category], p2[category])",
    &["f"],
    None,
    &["x1", "x2", "category"],
    &[("p1", 3), ("p2", 3)],
    &BuiltinOpsF64::new(),
)
.unwrap();
let options = Options::<f64, BuiltinOpsF64, 2> {
    template: Some(template),
    ..Default::default()
};
```

`variable_names` names the dataset's features in column order, and `num_features` (here `None`) may fix each subexpression's arity instead of inferring it from the calls. The formula may call the DSL-defined operators of the operator set passed last, and `D` differentiates built-in and DSL-defined operators to any order; operators written as Rust closures cannot be differentiated, which the search reports before it starts. A formula whose value is the same for every row, such as `f()`, fails when evaluated. Each member's `expr` holds one tree per subexpression (`trees`) and the parameter vectors concatenated in declaration order (`params`); `expr.string(&options, ..)` prints it as `f = #1 * #3; p1 = [..]; p2 = [..]`.

`Expression::trees` and `PopMember::plans` use `OneOrMany`: one component is stored inline, and multiple components use a vector. Use `.into()` to convert a `Vec` to this storage.

## Units

`Dataset::with_units` attaches units to the features and the target, written as PySR takes them: `"kg"`, `"m/s^2"`, `"km"`, `"Constants.G"`, or `""` for dimensionless. The parser follows DynamicQuantities' `uparse`, with its units, prefixes, constants, and the functions it defines on quantities (`"sqrt(mΩ)"`); a bad string fails with the offending token. Target units alone make every feature dimensionless.

```rust
use ndarray::{Array1, Array2};
use symbolic_regression::prelude::*;

let dataset = Dataset::<f64>::new(Array2::zeros((2, 8)), Array1::zeros(8))
    .with_units(Some(&["m", "s"]), Some("m/s"))
    .unwrap();
let options = Options::<f64, BuiltinOpsF64, 2> {
    dimensional_constraint_penalty: 1000.0,
    dimensionless_constants_only: false,
    ..Default::default()
};
```

As in SymbolicRegression.jl, constants take whatever units make an expression consistent unless `dimensionless_constants_only` is set, and each dimensionally inconsistent expression has `dimensional_constraint_penalty` (default 1000) added to its loss. Constant optimization minimizes the loss without the penalty. The hall of fame prints units: `y[m s⁻¹] = x0[m] / (x1[s] * 1.02[?])`, where `[?]` marks a constant whose units are free. Template expressions do not take units. `WasmSearch` takes `x_units` and `y_units` options.

## WASM

This workspace includes a thin `wasm-bindgen` wrapper crate (`symbolic_regression_wasm`) at `symbolic_regression_wasm/` and a minimal browser UI at `web_ui/` (Vite + WebWorker).

```bash
rustup target add wasm32-unknown-unknown

# one-time
cargo install wasm-pack

# build the wasm package into the Vite app
cd symbolic_regression_wasm
wasm-pack build --target web --out-dir ../web_ui/src/pkg

# run the dev server
cd ../web_ui
npm install
npm run dev
```

See `web_ui/README.md` for details.
