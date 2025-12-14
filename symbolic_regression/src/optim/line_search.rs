use super::linalg::{axpy_into, inf_norm};
use super::options::BackTracking;

fn quadratic_step(alpha: f64, phi: f64, phi0: f64, dphi0: f64) -> Option<f64> {
    let denom = 2.0 * (phi - phi0 - dphi0 * alpha);
    if !denom.is_finite() || denom == 0.0 {
        return None;
    }
    let out = -(dphi0 * alpha * alpha) / denom;
    if out.is_finite() {
        Some(out)
    } else {
        None
    }
}

fn cubic_step(
    alpha1: f64,
    phi1: f64,
    alpha2: f64,
    phi2: f64,
    phi0: f64,
    dphi0: f64,
) -> Option<f64> {
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
    if out.is_finite() {
        Some(out)
    } else {
        None
    }
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
