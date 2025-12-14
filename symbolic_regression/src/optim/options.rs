#[derive(Clone, Copy, Debug)]
pub(crate) struct OptimOptions {
    pub iterations: usize,
    pub f_calls_limit: usize,
    pub g_abstol: f64,
}

impl Default for OptimOptions {
    fn default() -> Self {
        Self {
            iterations: 1000,
            f_calls_limit: 0,
            g_abstol: 1e-8,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BackTracking {
    pub c1: f64,
    pub rho_hi: f64,
    pub rho_lo: f64,
    pub iterations: usize,
    pub order: u8,
    pub maxstep: f64,
}

impl Default for BackTracking {
    fn default() -> Self {
        Self {
            c1: 1e-4,
            rho_hi: 0.5,
            rho_lo: 0.1,
            iterations: 1000,
            order: 3,
            maxstep: f64::INFINITY,
        }
    }
}

#[derive(Default, Clone, Copy, Debug)]
pub(crate) struct EvalBudget {
    pub f_calls: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct OptimResult {
    pub minimizer: Vec<f64>,
    pub minimum: f64,
    pub f_calls: usize,
}

pub(crate) trait Objective {
    fn f_only(&mut self, x: &[f64], budget: &mut EvalBudget) -> Option<f64>;
    fn fg(&mut self, x: &[f64], g_out: &mut [f64], budget: &mut EvalBudget) -> Option<f64>;
}
