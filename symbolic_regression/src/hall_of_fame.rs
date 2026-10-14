use num_traits::Float;

use crate::loss_functions::max_zero;
use crate::options::{LossScale, Options};
use crate::pop_member::PopMember;

#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(serialize = "T: serde::Serialize", deserialize = "T: serde::Deserialize<'de>"))
)]
pub struct HallOfFame<T: Float, Ops, const D: usize> {
    pub best_by_complexity: Vec<Option<PopMember<T, Ops, D>>>,
}

impl<T: Float, Ops, const D: usize> Clone for HallOfFame<T, Ops, D> {
    fn clone(&self) -> Self {
        Self {
            best_by_complexity: self.best_by_complexity.clone(),
        }
    }
}

impl<T: Float, Ops, const D: usize> HallOfFame<T, Ops, D> {
    pub fn new(max_complexity: usize) -> Self {
        Self {
            best_by_complexity: vec![None; max_complexity + 1],
        }
    }

    pub fn consider(&mut self, member: &PopMember<T, Ops, D>, options: &Options<T, Ops, D>) {
        let c = member.complexity;
        // Match SymbolicRegression.jl `s_r_cycle`: keep entries with `0 < c <= maxsize`.
        if c == 0 || c > options.maxsize {
            return;
        }
        if c >= self.best_by_complexity.len() {
            return;
        }
        match &self.best_by_complexity[c] {
            None => self.best_by_complexity[c] = Some(member.clone()),
            Some(best) => {
                if member
                    .cost
                    .partial_cmp(&best.cost)
                    .unwrap_or(std::cmp::Ordering::Greater)
                    == std::cmp::Ordering::Less
                {
                    self.best_by_complexity[c] = Some(member.clone());
                }
            }
        }
    }

    pub fn update_from_members(&mut self, members: &[PopMember<T, Ops, D>], options: &Options<T, Ops, D>) {
        for m in members {
            self.consider(m, options);
        }
    }

    pub fn members(&self) -> impl Iterator<Item = &PopMember<T, Ops, D>> {
        self.best_by_complexity.iter().flatten()
    }

    /// Matches SymbolicRegression.jl `check_for_loss_threshold`: returns true if any existing
    /// member satisfies the callback. Caller passes `(loss, complexity) -> bool`.
    pub fn any_member_satisfies(&self, f: &dyn Fn(T, usize) -> bool) -> bool {
        self.best_by_complexity
            .iter()
            .flatten()
            .any(|m| f(m.loss, m.complexity))
    }

    /// The members with a lower loss than every simpler member, from simplest to most complex.
    pub(crate) fn front(&self) -> impl Iterator<Item = &PopMember<T, Ops, D>> {
        let mut best_loss = T::infinity();
        self.best_by_complexity.iter().flatten().filter(move |m| {
            let improves = m.loss < best_loss;
            if improves {
                best_loss = m.loss;
            }
            improves
        })
    }

    pub fn pareto_front(&self) -> Vec<PopMember<T, Ops, D>> {
        self.front().cloned().collect()
    }

    /// SymbolicRegression.jl's `format_hall_of_fame` check: scoring with `LossScale::Log` needs
    /// every loss on the Pareto front to be non-negative.
    pub fn check_loss_scale(&self, loss_scale: LossScale) -> Result<(), String> {
        if loss_scale == LossScale::Linear {
            return Ok(());
        }
        match self.front().find(|m| m.loss < T::zero()) {
            Some(m) => Err(format!(
                "the Pareto front has a negative loss ({}). Your loss function must be non-negative. To allow negative losses, set the `loss_scale` to linear, or consider wrapping your loss inside an exponential.",
                m.loss.to_f64().unwrap_or(f64::NAN)
            )),
            None => Ok(()),
        }
    }

    /// The score of each [`Self::pareto_front`] member, as SymbolicRegression.jl's
    /// `format_hall_of_fame` computes it: the first member scores 0, and each later one scores
    /// the drop in loss (`LossScale::Linear`) or log-loss (`LossScale::Log`) per unit of
    /// complexity since the previous member. Fails as [`Self::check_loss_scale`] does.
    pub fn pareto_scores(&self, loss_scale: LossScale) -> Result<Vec<T>, String> {
        self.check_loss_scale(loss_scale)?;
        let mut last: Option<(T, usize)> = None;
        Ok(self
            .front()
            .map(|m| {
                let score = match last {
                    None => T::zero(),
                    Some((last_loss, last_complexity)) => {
                        let delta_c = T::from(m.complexity).unwrap() - T::from(last_complexity).unwrap();
                        match loss_scale {
                            LossScale::Linear => compute_direct_score(m.loss, last_loss, delta_c),
                            LossScale::Log => compute_zero_centered_score(m.loss, last_loss, delta_c),
                        }
                    }
                };
                last = Some((m.loss, m.complexity));
                score
            })
            .collect())
    }

    /// The Pareto front member [`choose_best`] picks, as SymbolicRegression.jl reports the best
    /// equation; `None` when no member has a finite loss. Fails as [`Self::check_loss_scale`] does.
    pub fn best(&self, loss_scale: LossScale) -> Result<Option<PopMember<T, Ops, D>>, String> {
        let scores = self.pareto_scores(loss_scale)?;
        let losses: Vec<T> = self.front().map(|m| m.loss).collect();
        Ok(choose_best(&losses, &scores, loss_scale).and_then(|i| self.front().nth(i).cloned()))
    }
}

/// Julia's `eps(x)`: the distance from `|x|` to the next larger float.
fn spacing<T: Float>(x: T) -> T {
    let x = x.abs();
    if !x.is_finite() {
        return T::nan();
    }
    if x < T::min_positive_value() {
        return T::min_positive_value() * T::epsilon();
    }
    let (_, exp, _) = x.integer_decode();
    let two = T::one() + T::one();
    let half = i32::from(exp) / 2;
    two.powi(half) * two.powi(i32::from(exp) - half)
}

pub fn compute_direct_score<T: Float>(cur_loss: T, last_loss: T, delta_c: T) -> T {
    max_zero(-(cur_loss - last_loss) / delta_c)
}

pub fn compute_zero_centered_score<T: Float>(cur_loss: T, last_loss: T, delta_c: T) -> T {
    let log_ratio = (max_zero(cur_loss / last_loss) + spacing(cur_loss)).ln();
    max_zero(-log_ratio / delta_c)
}

/// The index of the equation SymbolicRegression.jl's `choose_best` (and PySR's
/// `model_selection="best"`) picks: with `LossScale::Log`, the highest score among equations whose
/// loss is within 1.5 times the lowest; with `LossScale::Linear`, the lowest loss. `None` when
/// there are no equations.
pub fn choose_best<T: Float>(losses: &[T], scores: &[T], loss_scale: LossScale) -> Option<usize> {
    let first_best = |better: &dyn Fn(T, T) -> bool, key: &dyn Fn(usize) -> T| {
        (1..losses.len()).fold(0, |best, i| if better(key(i), key(best)) { i } else { best })
    };
    if losses.is_empty() {
        return None;
    }
    Some(match loss_scale {
        LossScale::Linear => first_best(&|a, b| a < b, &|i| losses[i]),
        LossScale::Log => {
            let threshold = T::from(1.5).unwrap() * losses.iter().copied().fold(T::infinity(), T::min);
            let key = |i: usize| {
                if losses[i] <= threshold {
                    scores[i]
                } else {
                    T::neg_infinity()
                }
            };
            first_best(&|a, b| a > b, &key)
        }
    })
}

/// SymbolicRegression.jl's `pareto_volume`: the area of the convex hull of the Pareto front in
/// (log10 complexity, loss) space, with log10 losses for `LossScale::Log`, closed off at
/// complexity `maxsize + 1`.
pub fn pareto_volume<T: Float>(losses: &[T], complexities: &[usize], maxsize: usize, loss_scale: LossScale) -> f64 {
    if losses.is_empty() {
        return 0.0;
    }
    let mut points: Vec<[f64; 2]> = losses
        .iter()
        .zip(complexities)
        .map(|(&loss, &c)| {
            let y = match loss_scale {
                LossScale::Linear => loss,
                LossScale::Log => (loss + T::epsilon()).log10(),
            };
            [(c as f64).log10(), y.to_f64().unwrap()]
        })
        .collect();
    let min_y = points.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
    points.push([((maxsize + 1) as f64).log10(), min_y]);
    let max_y = points.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max);
    let max_x = points.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max);
    points.push([max_x, max_y]);
    convex_hull_area(&convex_hull(&points))
}

/// Gift wrapping, as in SymbolicRegression.jl's `convex_hull`.
fn convex_hull(points: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let is_left_of = |p: [f64; 2], start: [f64; 2], end: [f64; 2]| {
        (end[0] - start[0]) * (p[1] - start[1]) - (end[1] - start[1]) * (p[0] - start[0]) > 0.0
    };
    let mut cur = points
        .iter()
        .copied()
        .fold(points[0], |a, p| if p[0] < a[0] { p } else { a });
    let mut hull = Vec::new();
    loop {
        hull.push(cur);
        let mut end = points[0];
        for &candidate in points {
            if end == cur || is_left_of(candidate, cur, end) {
                end = candidate;
            }
        }
        cur = end;
        if end == hull[0] {
            return hull;
        }
    }
}

fn convex_hull_area(hull: &[[f64; 2]]) -> f64 {
    let twice: f64 = (0..hull.len())
        .map(|i| {
            let j = (i + 1) % hull.len();
            hull[i][0] * hull[j][1] - hull[j][0] * hull[i][1]
        })
        .sum();
    twice.abs() / 2.0
}
