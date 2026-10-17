//! Rust-only module (no direct Julia file): lightweight optimization routines.

use dynamic_expressions::utils::ZipEq;

#[derive(Clone, Copy, Debug)]
pub(crate) struct OptimOptions {
    pub iterations: usize,
    pub f_calls_limit: usize,
    pub g_abstol: f64,
    pub g_calls_limit: usize,
    pub x_abstol: f64,
    pub f_reltol: f64,
}

impl Default for OptimOptions {
    fn default() -> Self {
        Self {
            iterations: 1000,
            f_calls_limit: 0,
            g_abstol: 1e-8,
            g_calls_limit: 0,
            x_abstol: 0.0,
            f_reltol: 0.0,
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

pub(crate) fn inf_norm(v: &[f64]) -> f64 {
    v.iter().copied().map(f64::abs).fold(0.0, |a, b| a.max(b))
}

pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().copied().zip_eq(b.iter().copied()).map(|(x, y)| x * y).sum()
}

pub(crate) fn axpy_into(out: &mut [f64], x: &[f64], alpha: f64, s: &[f64]) {
    for ((o, &xi), &si) in out.iter_mut().zip_eq(x).zip_eq(s) {
        *o = xi + alpha * si;
    }
}

pub(crate) fn matvec(out: &mut [f64], a: &[f64], x: &[f64]) {
    let n = x.len();
    debug_assert_eq!(a.len(), n * n);
    debug_assert_eq!(out.len(), n);

    for i in 0..n {
        let row = &a[i * n..(i + 1) * n];
        out[i] = row
            .iter()
            .copied()
            .zip_eq(x.iter().copied())
            .map(|(aa, xx)| aa * xx)
            .sum();
    }
}

fn quadratic_step(alpha: f64, phi: f64, phi0: f64, dphi0: f64) -> Option<f64> {
    let denom = 2.0 * (phi - phi0 - dphi0 * alpha);
    if !denom.is_finite() || denom == 0.0 {
        return None;
    }
    let out = -(dphi0 * alpha * alpha) / denom;
    if out.is_finite() { Some(out) } else { None }
}

fn cubic_step(alpha1: f64, phi1: f64, alpha2: f64, phi2: f64, phi0: f64, dphi0: f64) -> Option<f64> {
    let d1 = phi1 - phi0 - dphi0 * alpha1;
    let d2 = phi2 - phi0 - dphi0 * alpha2;
    let denom = alpha1 * alpha1 * alpha2 * alpha2 * (alpha2 - alpha1);
    if !denom.is_finite() || denom == 0.0 {
        return None;
    }
    let a = (alpha1 * alpha1 * d2 - alpha2 * alpha2 * d1) / denom;
    let b = (-alpha1 * alpha1 * alpha1 * d2 + alpha2 * alpha2 * alpha2 * d1) / denom;
    if !a.is_finite() || !b.is_finite() {
        return None;
    }

    if a.abs() <= f64::EPSILON {
        let denom_b = 2.0 * b;
        if denom_b == 0.0 || !denom_b.is_finite() {
            return None;
        }
        let out = -dphi0 / denom_b;
        return if out.is_finite() { Some(out) } else { None };
    }

    let disc = (b * b - 3.0 * a * dphi0).max(0.0);
    let sqrt_disc = disc.sqrt();
    let out = (-b + sqrt_disc) / (3.0 * a);
    if out.is_finite() { Some(out) } else { None }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct LineSearchInput<'a> {
    pub x: &'a [f64],
    pub s: &'a [f64],
    pub alpha0: f64,
    pub phi0: f64,
    pub dphi0: f64,
}

pub(crate) fn backtracking_linesearch(
    ls: &BackTracking,
    input: LineSearchInput<'_>,
    x_new: &mut [f64],
    phi_at: &mut impl FnMut(&[f64]) -> Option<f64>,
) -> Option<(f64, f64)> {
    let x = input.x;
    let s = input.s;
    let phi0 = input.phi0;
    let dphi0 = input.dphi0;
    let mut alpha = input.alpha0;

    if !alpha.is_finite() || alpha <= 0.0 {
        return None;
    }

    if ls.maxstep.is_finite() {
        let step_norm = inf_norm(s);
        if step_norm > 0.0 {
            alpha = alpha.min(ls.maxstep / step_norm);
        }
    }

    let mut alpha_prev = alpha;
    let mut phi_prev: Option<f64> = None;

    // If phi(alpha) is not finite, keep halving until it is.
    let finite_max = ((f64::EPSILON.ln() / 2.0_f64.ln()).abs().ceil() as usize).max(1);
    let mut phi_alpha: f64;
    {
        let mut ok = None;
        let mut a = alpha;
        for _ in 0..finite_max {
            axpy_into(x_new, x, a, s);
            ok = phi_at(x_new);
            if matches!(ok, Some(v) if v.is_finite()) {
                break;
            }
            a *= 0.5;
            if a <= 0.0 {
                ok = None;
                break;
            }
        }
        alpha = a;
        phi_alpha = ok?;
        if !phi_alpha.is_finite() {
            return None;
        }
    }

    let armijo_rhs = |a: f64| phi0 + ls.c1 * a * dphi0;

    for iter in 0..ls.iterations {
        if phi_alpha <= armijo_rhs(alpha) {
            return Some((alpha, phi_alpha));
        }

        let mut alpha_tmp = if iter == 0 || ls.order < 3 {
            quadratic_step(alpha, phi_alpha, phi0, dphi0)
        } else if let Some(phi_prev_v) = phi_prev {
            cubic_step(alpha, phi_alpha, alpha_prev, phi_prev_v, phi0, dphi0)
        } else {
            quadratic_step(alpha, phi_alpha, phi0, dphi0)
        }
        .unwrap_or(alpha * 0.5);

        // Clamp to [rho_lo*alpha, rho_hi*alpha]
        let lo = ls.rho_lo * alpha;
        let hi = ls.rho_hi * alpha;
        if alpha_tmp > hi {
            alpha_tmp = hi;
        }
        if alpha_tmp < lo {
            alpha_tmp = lo;
        }

        alpha_prev = alpha;
        phi_prev = Some(phi_alpha);

        alpha = alpha_tmp;
        if !alpha.is_finite() || alpha <= 0.0 {
            return None;
        }

        axpy_into(x_new, x, alpha, s);
        phi_alpha = phi_at(x_new)?;
        if !phi_alpha.is_finite() {
            return None;
        }
    }

    None
}

/// Optim.jl's affine simplex and dimension-adaptive Nelder-Mead coefficients.
pub(crate) fn nelder_mead_minimize(x0: &[f64], obj: &mut impl Objective, opts: OptimOptions) -> Option<OptimResult> {
    let n = x0.len();
    assert!(n > 0);
    let m = n + 1;
    let mut budget = EvalBudget::default();
    let mut simplex = vec![x0.to_vec(); m];
    for j in 0..n {
        simplex[j + 1][j] = 1.5 * x0[j] + 0.025;
    }
    let mut values = vec![f64::INFINITY; m];
    let mut evaluate = |x: &[f64], budget: &mut EvalBudget| {
        if opts.f_calls_limit != 0 && budget.f_calls >= opts.f_calls_limit {
            return None;
        }
        Some(obj.f_only(x, budget).unwrap_or(f64::INFINITY))
    };
    for (value, x) in values.iter_mut().zip(&simplex) {
        if let Some(f) = evaluate(x, &mut budget) {
            *value = f;
        } else {
            break;
        }
    }
    let mut order: Vec<usize> = (0..m).collect();
    let mut centroid = vec![0.0; n];
    let mut reflect = vec![0.0; n];
    let mut candidate = vec![0.0; n];
    let expansion = 1.0 + 2.0 / n as f64;
    let contraction = 0.75 - 0.5 / n as f64;
    let shrink = 1.0 - 1.0 / n as f64;
    for _ in 0..opts.iterations {
        order.sort_unstable_by(|&a, &b| values[a].total_cmp(&values[b]));
        let best = order[0];
        let worst = order[n];
        let mean = values.iter().sum::<f64>() / m as f64;
        let spread = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / m as f64).sqrt();
        if spread <= opts.g_abstol || (opts.f_calls_limit != 0 && budget.f_calls >= opts.f_calls_limit) {
            break;
        }
        centroid.fill(0.0);
        for &i in &order[..n] {
            for (c, &v) in centroid.iter_mut().zip(&simplex[i]) {
                *c += v / n as f64;
            }
        }
        for j in 0..n {
            reflect[j] = 2.0 * centroid[j] - simplex[worst][j];
        }
        let Some(f_reflect) = evaluate(&reflect, &mut budget) else {
            break;
        };
        let mut do_shrink = false;
        if f_reflect < values[best] {
            for j in 0..n {
                candidate[j] = centroid[j] + expansion * (reflect[j] - centroid[j]);
            }
            let f_expand = evaluate(&candidate, &mut budget);
            if f_expand.is_some_and(|f| f < f_reflect) {
                simplex[worst].copy_from_slice(&candidate);
                values[worst] = f_expand.unwrap();
            } else {
                simplex[worst].copy_from_slice(&reflect);
                values[worst] = f_reflect;
            }
        } else if f_reflect < values[order[n - 1]] {
            simplex[worst].copy_from_slice(&reflect);
            values[worst] = f_reflect;
        } else {
            let outside = f_reflect < values[worst];
            let factor = if outside { contraction } else { -contraction };
            for j in 0..n {
                candidate[j] = centroid[j] + factor * (reflect[j] - centroid[j]);
            }
            let Some(f_contract) = evaluate(&candidate, &mut budget) else {
                break;
            };
            let threshold = if outside { f_reflect } else { values[worst] };
            if f_contract < threshold {
                simplex[worst].copy_from_slice(&candidate);
                values[worst] = f_contract;
            } else {
                do_shrink = true;
            }
        }
        if do_shrink {
            for &i in &order[1..] {
                for j in 0..n {
                    candidate[j] = simplex[best][j] + shrink * (simplex[i][j] - simplex[best][j]);
                }
                let Some(f) = evaluate(&candidate, &mut budget) else {
                    break;
                };
                simplex[i].copy_from_slice(&candidate);
                values[i] = f;
            }
        }
    }
    order.sort_unstable_by(|&a, &b| values[a].total_cmp(&values[b]));
    centroid.fill(0.0);
    for &i in &order[..n] {
        for (c, &v) in centroid.iter_mut().zip(&simplex[i]) {
            *c += v / n as f64;
        }
    }
    let best = order[0];
    if let Some(f) = evaluate(&centroid, &mut budget) {
        if f < values[best] {
            return Some(OptimResult {
                minimizer: centroid,
                minimum: f,
                f_calls: budget.f_calls,
            });
        }
    }
    Some(OptimResult {
        minimizer: simplex.swap_remove(best),
        minimum: values[best],
        f_calls: budget.f_calls,
    })
}

pub(crate) fn bfgs_minimize(
    x0: &[f64],
    obj: &mut impl Objective,
    opts: OptimOptions,
    ls: BackTracking,
) -> Option<OptimResult> {
    let n = x0.len();
    let mut budget = EvalBudget::default();

    let mut x = x0.to_vec();
    let mut x_prev = vec![0.0; n];
    let mut x_ls = vec![0.0; n];

    let mut g = vec![0.0; n];
    let mut g_prev = vec![0.0; n];
    let mut s = vec![0.0; n];
    let mut u = vec![0.0; n];
    let mut dx = vec![0.0; n];
    let mut dg = vec![0.0; n];

    let mut inv_h = vec![0.0; n * n];
    for i in 0..n {
        inv_h[i * n + i] = 1.0;
    }

    let mut g_calls = 1;
    let mut fx = obj.fg(&x, &mut g, &mut budget)?;
    if !fx.is_finite() {
        return None;
    }

    for _ in 0..opts.iterations {
        if opts.f_calls_limit != 0 && budget.f_calls >= opts.f_calls_limit {
            break;
        }

        let g_inf = inf_norm(&g);
        if !g_inf.is_finite() || g_inf <= opts.g_abstol {
            break;
        }

        matvec(&mut s, &inv_h, &g);
        for v in &mut s {
            *v = -*v;
        }
        let mut dphi0 = dot(&g, &s);
        if !dphi0.is_finite() || dphi0 >= 0.0 {
            inv_h.fill(0.0);
            for i in 0..n {
                inv_h[i * n + i] = 1.0;
            }
            for (dst, gi) in s.iter_mut().zip_eq(&g) {
                *dst = -gi;
            }
            dphi0 = dot(&g, &s);
        }
        if !dphi0.is_finite() || dphi0 >= 0.0 {
            break;
        }

        let phi0 = fx;
        let mut phi_at = |x_trial: &[f64]| -> Option<f64> {
            if opts.f_calls_limit != 0 && budget.f_calls >= opts.f_calls_limit {
                return None;
            }
            let v = obj.f_only(x_trial, &mut budget)?;
            if v.is_finite() { Some(v) } else { None }
        };

        let (_alpha, accepted_fx) = backtracking_linesearch(
            &ls,
            LineSearchInput {
                x: &x,
                s: &s,
                alpha0: 1.0,
                phi0,
                dphi0,
            },
            &mut x_ls,
            &mut phi_at,
        )?;

        x_prev.copy_from_slice(&x);
        g_prev.copy_from_slice(&g);
        x.copy_from_slice(&x_ls);

        if (opts.g_calls_limit != 0 && g_calls >= opts.g_calls_limit)
            || (opts.f_calls_limit != 0 && budget.f_calls >= opts.f_calls_limit)
        {
            fx = accepted_fx;
            break;
        }
        g_calls += 1;
        fx = obj.fg(&x, &mut g, &mut budget)?;
        if !fx.is_finite() {
            break;
        }

        // BFGS update (inverse Hessian):
        for ((dst, xi), xpi) in dx.iter_mut().zip_eq(&x).zip_eq(&x_prev) {
            *dst = xi - xpi;
        }
        for ((dst, gcur), gpi) in dg.iter_mut().zip_eq(&g).zip_eq(&g_prev) {
            *dst = gcur - gpi;
        }

        if (opts.x_abstol > 0.0 && inf_norm(&dx) <= opts.x_abstol)
            || (opts.f_reltol > 0.0 && (fx - phi0).abs() <= opts.f_reltol * fx.abs())
        {
            break;
        }
        let dx_dg = dot(&dx, &dg);
        let curvature_eps = 1e-12;
        if !(dx_dg.is_finite() && dx_dg > curvature_eps) {
            inv_h.fill(0.0);
            for i in 0..n {
                inv_h[i * n + i] = 1.0;
            }
            continue;
        }
        if dx_dg > 0.0 {
            matvec(&mut u, &inv_h, &dg);
            let dg_u = dot(&dg, &u);
            if dg_u.is_finite() {
                let c1 = (dx_dg + dg_u) / (dx_dg * dx_dg);
                let c2 = 1.0 / dx_dg;
                for i in 0..n {
                    for j in 0..n {
                        inv_h[i * n + j] += c1 * dx[i] * dx[j] - c2 * (u[i] * dx[j] + dx[i] * u[j]);
                    }
                }
                // Keep symmetry; floating point drift can accumulate.
                for i in 0..n {
                    for j in (i + 1)..n {
                        let a = 0.5 * (inv_h[i * n + j] + inv_h[j * n + i]);
                        inv_h[i * n + j] = a;
                        inv_h[j * n + i] = a;
                    }
                }
            }
        }
    }

    Some(OptimResult {
        minimizer: x,
        minimum: fx,
        f_calls: budget.f_calls,
    })
}

pub(crate) fn newton_1d_minimize(
    x0: f64,
    obj: &mut impl Objective,
    opts: OptimOptions,
    ls: BackTracking,
) -> Option<OptimResult> {
    let mut budget = EvalBudget::default();
    let mut x = x0;
    let mut g_calls = 0;
    let mut best_x = x0;
    let mut best_f = f64::INFINITY;
    let mut g_tmp = [0.0f64];

    for _ in 0..opts.iterations {
        if opts.f_calls_limit != 0 && budget.f_calls >= opts.f_calls_limit {
            break;
        }
        let x_vec = [x];
        if opts.g_calls_limit != 0 && g_calls >= opts.g_calls_limit {
            break;
        }
        g_calls += 1;
        let f = obj.fg(&x_vec, &mut g_tmp, &mut budget)?;
        let g = g_tmp[0];
        if !f.is_finite() || !g.is_finite() {
            return None;
        }
        if f < best_f {
            best_f = f;
            best_x = x;
        }
        if g.abs() <= opts.g_abstol {
            break;
        }

        // Finite-difference Hessian estimate:
        let eps = f64::EPSILON.sqrt() * (x.abs() + 1.0);
        let x_plus = [x + eps];
        let x_minus = [x - eps];
        if opts.g_calls_limit != 0 && g_calls + 2 > opts.g_calls_limit {
            break;
        }
        g_calls += 2;
        let _ = obj.fg(&x_plus, &mut g_tmp, &mut budget)?;
        let g_plus = g_tmp[0];
        let _ = obj.fg(&x_minus, &mut g_tmp, &mut budget)?;
        let g_minus = g_tmp[0];
        let h = (g_plus - g_minus) / (2.0 * eps);
        let mut h_mod = h.abs().max(1e-12);
        if !h_mod.is_finite() {
            h_mod = 1e-12;
        }

        let mut s = -g / h_mod;
        let mut dphi0 = g * s;
        if !dphi0.is_finite() || dphi0 >= 0.0 {
            s = -g;
            dphi0 = g * s;
        }
        if !dphi0.is_finite() || dphi0 >= 0.0 {
            break;
        }

        let phi0 = f;
        let mut phi_at = |x_t: &[f64]| -> Option<f64> {
            if opts.f_calls_limit != 0 && budget.f_calls >= opts.f_calls_limit {
                return None;
            }
            let v = obj.f_only(x_t, &mut budget)?;
            if v.is_finite() { Some(v) } else { None }
        };

        let x_vec = [x];
        let s_vec = [s];
        let mut x_new = [x];
        let (alpha, accepted_fx) = backtracking_linesearch(
            &ls,
            LineSearchInput {
                x: &x_vec,
                s: &s_vec,
                alpha0: 1.0,
                phi0,
                dphi0,
            },
            &mut x_new,
            &mut phi_at,
        )?;
        let step = alpha * s;
        x += step;
        if accepted_fx < best_f {
            best_x = x;
            best_f = accepted_fx;
        }
        if (opts.x_abstol > 0.0 && step.abs() <= opts.x_abstol)
            || (opts.f_reltol > 0.0 && (accepted_fx - phi0).abs() <= opts.f_reltol * accepted_fx.abs())
        {
            break;
        }
    }

    Some(OptimResult {
        minimizer: vec![best_x],
        minimum: best_f,
        f_calls: budget.f_calls,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
            ..Default::default()
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
            ..Default::default()
        };
        let ls = BackTracking::default();
        let mut obj = Quad1D;
        let res = newton_1d_minimize(0.0, &mut obj, opts, ls).unwrap();
        assert!((res.minimizer[0] - 3.0).abs() < 1e-6);
    }

    struct Quad3DOffDiag;

    impl Objective for Quad3DOffDiag {
        fn f_only(&mut self, x: &[f64], budget: &mut EvalBudget) -> Option<f64> {
            budget.f_calls += 1;
            // f(x) = 0.5 x^T A x - b^T x
            // A = [[2,1,0],[1,2,0],[0,0,3]] (SPD), b = [1,0,2]
            let a00 = 2.0;
            let a01 = 1.0;
            let a11 = 2.0;
            let a22 = 3.0;
            let b0 = 1.0;
            let b1 = 0.0;
            let b2 = 2.0;

            let x0 = x[0];
            let x1 = x[1];
            let x2 = x[2];

            let ax0 = a00 * x0 + a01 * x1;
            let ax1 = a01 * x0 + a11 * x1;
            let ax2 = a22 * x2;

            let xtax = x0 * ax0 + x1 * ax1 + x2 * ax2;
            Some(0.5 * xtax - (b0 * x0 + b1 * x1 + b2 * x2))
        }

        fn fg(&mut self, x: &[f64], g_out: &mut [f64], budget: &mut EvalBudget) -> Option<f64> {
            budget.f_calls += 1;
            let a00 = 2.0;
            let a01 = 1.0;
            let a11 = 2.0;
            let a22 = 3.0;
            let b0 = 1.0;
            let b1 = 0.0;
            let b2 = 2.0;

            let x0 = x[0];
            let x1 = x[1];
            let x2 = x[2];

            let ax0 = a00 * x0 + a01 * x1;
            let ax1 = a01 * x0 + a11 * x1;
            let ax2 = a22 * x2;

            g_out[0] = ax0 - b0;
            g_out[1] = ax1 - b1;
            g_out[2] = ax2 - b2;

            let xtax = x0 * ax0 + x1 * ax1 + x2 * ax2;
            Some(0.5 * xtax - (b0 * x0 + b1 * x1 + b2 * x2))
        }
    }

    #[test]
    fn bfgs_minimizes_spd_quadratic_with_off_diagonal() {
        let opts = OptimOptions {
            iterations: 100,
            f_calls_limit: 0,
            g_abstol: 1e-10,
            ..Default::default()
        };
        let ls = BackTracking::default();
        let mut obj = Quad3DOffDiag;
        let res = bfgs_minimize(&[0.5, -0.5, 0.0], &mut obj, opts, ls).unwrap();
        // Solve A x = b:
        // [2 1] [x0] = [1]  => x0=2/3, x1=-1/3 ; x2 = 2/3
        assert!((res.minimizer[0] - (2.0 / 3.0)).abs() < 1e-6);
        assert!((res.minimizer[1] - (-1.0 / 3.0)).abs() < 1e-6);
        assert!((res.minimizer[2] - (2.0 / 3.0)).abs() < 1e-6);
        assert!(res.minimum.is_finite());
    }
}
