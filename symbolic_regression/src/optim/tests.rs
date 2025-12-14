use super::bfgs::{bfgs_minimize, newton_1d_minimize};
use super::line_search::{backtracking_linesearch, LineSearchInput};
use super::{BackTracking, EvalBudget, Objective, OptimOptions};

struct Quad2D;

impl Objective for Quad2D {
    fn f_only(&mut self, x: &[f64], budget: &mut EvalBudget) -> Option<f64> {
        budget.f_calls += 1;
        let t0 = x[0] - 1.0;
        let t1 = x[1] + 2.0;
        Some(t0 * t0 + 10.0 * t1 * t1)
    }

    fn fg(&mut self, x: &[f64], g_out: &mut [f64], budget: &mut EvalBudget) -> Option<f64> {
        budget.f_calls += 1;
        let t0 = x[0] - 1.0;
        let t1 = x[1] + 2.0;
        g_out[0] = 2.0 * t0;
        g_out[1] = 20.0 * t1;
        Some(t0 * t0 + 10.0 * t1 * t1)
    }
}

struct Quad1D;

impl Objective for Quad1D {
    fn f_only(&mut self, x: &[f64], budget: &mut EvalBudget) -> Option<f64> {
        budget.f_calls += 1;
        let t = x[0] - 3.0;
        Some(t * t)
    }

    fn fg(&mut self, x: &[f64], g_out: &mut [f64], budget: &mut EvalBudget) -> Option<f64> {
        budget.f_calls += 1;
        let t = x[0] - 3.0;
        let f = t * t;
        g_out[0] = 2.0 * t;
        Some(f)
    }
}

#[test]
fn backtracking_satisfies_armijo_for_quadratic() {
    let ls = BackTracking::default();
    let x = [0.0];
    let s = [6.0];
    let phi0 = 9.0;
    let dphi0 = -36.0;
    let mut x_new = [0.0];
    let mut phi_at = |xv: &[f64]| {
        let t = xv[0] - 3.0;
        Some(t * t)
    };
    let (alpha, phi) = backtracking_linesearch(
        &ls,
        LineSearchInput {
            x: &x,
            s: &s,
            alpha0: 1.0,
            phi0,
            dphi0,
        },
        &mut x_new,
        &mut phi_at,
    )
    .expect("line search failed");
    assert!(alpha > 0.0 && alpha < 1.0);
    assert!(phi <= phi0 + ls.c1 * alpha * dphi0);
}

#[test]
fn bfgs_minimizes_simple_quadratic() {
    let opts = OptimOptions {
        iterations: 50,
        f_calls_limit: 0,
        g_abstol: 1e-10,
    };
    let ls = BackTracking::default();
    let mut obj = Quad2D;
    let res = bfgs_minimize(&[0.0, 0.0], &mut obj, opts, ls).unwrap();
    assert!((res.minimizer[0] - 1.0).abs() < 1e-6);
    assert!((res.minimizer[1] + 2.0).abs() < 1e-6);
}

#[test]
fn newton_1d_minimizes_quadratic() {
    let opts = OptimOptions {
        iterations: 25,
        f_calls_limit: 0,
        g_abstol: 1e-10,
    };
    let ls = BackTracking::default();
    let mut obj = Quad1D;
    let res = newton_1d_minimize(0.0, &mut obj, opts, ls).unwrap();
    assert!((res.minimizer[0] - 3.0).abs() < 1e-6);
}
