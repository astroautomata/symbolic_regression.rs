# dynamic_expressions

[![crates.io](https://img.shields.io/crates/v/dynamic_expressions)](https://crates.io/crates/dynamic_expressions)

Fast batched evaluation + forward-mode derivatives for symbolic expressions (Rust port of `DynamicExpressions.jl`).

This crate is the evaluation backend used by `symbolic_regression`.

## Data layout

Evaluation APIs take `ndarray::ArrayView2<'_, T>` for `X` with shape `(n_features, n_rows)`.

## Operator semantics

Built-in operators follow `SymbolicRegression.jl` at the edges of their domains:

- `x ^ y` is NaN for `0 ^ y` with `y < 0` and for a negative base with a non-integer exponent.
- `log`, `log2`, and `log10` are NaN for `x <= 0`, and `log1p` for `x <= -1`.
- `sign(±0)` is `±0`.

## Printing

- `pow` prints infix, as `x ^ y`.
- A `PostfixExpr` stores operator ids, so it prints together with its operator set: `string_tree(&expr, &operators, StringTreeOptions::default())`.
