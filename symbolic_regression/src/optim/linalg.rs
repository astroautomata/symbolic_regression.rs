pub(crate) fn inf_norm(v: &[f64]) -> f64 {
    v.iter().copied().map(f64::abs).fold(0.0, |a, b| a.max(b))
}

pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .copied()
        .zip(b.iter().copied())
        .map(|(x, y)| x * y)
        .sum()
}

pub(crate) fn axpy_into(out: &mut [f64], x: &[f64], alpha: f64, s: &[f64]) {
    for ((o, &xi), &si) in out.iter_mut().zip(x.iter()).zip(s.iter()) {
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
            .zip(x.iter().copied())
            .map(|(aa, xx)| aa * xx)
            .sum();
    }
}
