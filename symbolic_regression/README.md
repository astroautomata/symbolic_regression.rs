# symbolic_regression

[![crates.io](https://img.shields.io/crates/v/symbolic_regression)](https://crates.io/crates/symbolic_regression)

Rust port of the core engine from `SymbolicRegression.jl` (regularized evolution + Pareto hall-of-fame),
built on top of the `dynamic_expressions` crate in this workspace.

For a repo-level overview and examples, see `README.md` at the repo root.

## Run the example

```bash
cargo run -p symbolic_regression --example example --release
```

## Complexity

`complexity_of_constants`, `complexity_of_variables`, `variable_complexities`, and `operator_complexity_overrides` are `f32`, so costs may be fractional. A tree's complexity is the sum of its node costs rounded to the nearest integer (ties to even), as in `SymbolicRegression.jl`.
