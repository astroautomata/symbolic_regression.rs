//! Python bindings for `symbolic_regression`, imported as `symbolic_regression_rs`: a search
//! function taking SymbolicRegression.jl's keyword arguments, and the picklable state it returns.

mod keywords;
// Unit tests cover the keyword mapping without linking Python, which extension modules leave to
// the interpreter that imports them.
#[cfg(not(test))]
mod python;
