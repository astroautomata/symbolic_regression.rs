# symbolic_regression

[![crates.io](https://img.shields.io/crates/v/symbolic_regression)](https://crates.io/crates/symbolic_regression)

Rust port of the core engine from `SymbolicRegression.jl` (regularized evolution + Pareto hall-of-fame),
built on top of the `dynamic_expressions` crate in this workspace.

For a repo-level overview and examples, see `README.md` at the repo root.

## CLI (`symreg`)

This crate optionally provides a CLI binary named `symreg` behind the `cli` feature.

Build and run it from the repo root:

```bash
cargo run -p symbolic_regression --features cli --bin symreg -- data.csv --y target
```

## Run the example

```bash
cargo run -p symbolic_regression --example example --release
```
