mod bfgs;
mod linalg;
mod line_search;
mod options;

pub(crate) use bfgs::{bfgs_minimize, newton_1d_minimize};
pub(crate) use options::{BackTracking, EvalBudget, Objective, OptimOptions};

#[cfg(test)]
mod tests;
