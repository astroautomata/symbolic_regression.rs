//! SymbolicRegression.jl's built-in `AdaptiveMutationWeightsPlugin` and `MutationBurstPlugin`,
//! configured as data in [`Options`]. Simulated annealing and adaptive parsimony, the other two
//! built-in plugins, are configured by `Options::annealing`/`alpha` and
//! `Options::use_frequency`/`use_frequency_in_tournament`.

use fastrand::Rng;
use num_traits::Float;

use crate::mutate::MutationChoice;
use crate::options::{MutationWeights, Options};

/// The objective [`AdaptiveMutationWeights`] counts improvements in.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub enum Reward {
    #[default]
    Cost,
    Loss,
}

/// SymbolicRegression.jl's `AdaptiveMutationWeightsPlugin`: each population learns a multiplier
/// for every mutation kind from how often that kind strictly improves the reward objective. After
/// each mutation, the sampled kind's multiplier moves by an exponential moving average toward its
/// smoothed success rate relative to the mean rate, clamped to `[floor, 1/floor]`, and the
/// multipliers are renormalized to unit mean. Sampling multiplies each weight by
/// `multiplier^adaptation_strength`. Simplify and do-nothing mutations take no part.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AdaptiveMutationWeights {
    /// The moving average's factor, in `[0, 1]`.
    pub smoothing: f64,
    /// The clamp on a kind's relative success rate, in `(0, 1]`.
    pub floor: f64,
    pub reward: Reward,
    /// The learned multiplier's strength in log space, in `[0, 1]`: zero keeps the configured
    /// weights, one applies the multipliers in full.
    pub adaptation_strength: f64,
}

impl Default for AdaptiveMutationWeights {
    fn default() -> Self {
        Self {
            smoothing: 0.02,
            floor: 0.05,
            reward: Reward::Cost,
            adaptation_strength: 0.5,
        }
    }
}

impl AdaptiveMutationWeights {
    pub fn check(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.smoothing) {
            return Err("`smoothing` must be between 0 and 1.".to_string());
        }
        if !(self.floor > 0.0 && self.floor <= 1.0) {
            return Err("`floor` must be in (0, 1].".to_string());
        }
        if !(0.0..=1.0).contains(&self.adaptation_strength) {
            return Err("`adaptation_strength` must be between 0 and 1.".to_string());
        }
        Ok(())
    }
}

/// SymbolicRegression.jl's `MutationBurstPlugin`: a rejected mutation is retried on the same
/// parent, up to `retry_attempts` attempts in total; after an accepted one, with probability
/// `compound_probability` another mutation is chained onto the result, up to
/// `compound_max_steps` accepted mutations in total.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MutationBurst {
    pub retry_attempts: usize,
    pub compound_probability: f64,
    pub compound_max_steps: usize,
}

impl Default for MutationBurst {
    fn default() -> Self {
        Self {
            retry_attempts: 4,
            compound_probability: 0.25,
            compound_max_steps: 2,
        }
    }
}

impl MutationBurst {
    pub fn check(&self) -> Result<(), String> {
        if self.retry_attempts < 1 {
            return Err("`retry_attempts` must be at least 1.".to_string());
        }
        if !(0.0..=1.0).contains(&self.compound_probability) {
            return Err("`compound_probability` must be between 0 and 1.".to_string());
        }
        if self.compound_max_steps < 1 {
            return Err("`compound_max_steps` must be at least 1.".to_string());
        }
        Ok(())
    }

    /// Runs one burst from `parent`. Each call to `step` is one mutation attempt, returning the
    /// child and whether it was accepted. Returns the selected attempt (the last one, or the last
    /// accepted one once a compound step is rejected) and its index among the attempts.
    pub(crate) fn run<M>(
        &self,
        rng: &mut Rng,
        parent: &M,
        mut step: impl FnMut(&mut Rng, &M) -> (M, bool),
    ) -> (M, bool, usize) {
        let (mut member, mut accepted) = step(rng, parent);
        let (mut selected, mut n_attempts) = (0, 1);
        for _ in 1..self.retry_attempts {
            if accepted {
                break;
            }
            (member, accepted) = step(rng, parent);
            selected = n_attempts;
            n_attempts += 1;
        }
        if !accepted {
            return (member, accepted, selected);
        }
        let mut n_steps = 1;
        while n_steps < self.compound_max_steps && rng.f64() < self.compound_probability {
            let (next, next_accepted) = step(rng, &member);
            n_attempts += 1;
            if !next_accepted {
                break;
            }
            member = next;
            selected = n_attempts - 1;
            n_steps += 1;
        }
        (member, true, selected)
    }
}

/// Fails when a built-in plugin's configuration is outside its domain.
pub(crate) fn check_plugins<T: Float, Ops, const D: usize>(options: &Options<T, Ops, D>) -> Result<(), String> {
    if options.annealing && !(options.alpha.is_finite() && options.alpha > 0.0) {
        return Err("`alpha` must be finite and positive.".to_string());
    }
    if let Some(weights) = &options.adaptive_mutation_weights {
        weights.check()?;
    }
    if let Some(burst) = &options.mutation_burst {
        burst.check()?;
    }
    Ok(())
}

/// One mutation's outcome: `after` holds the child's `(cost, loss)` when it was evaluated.
#[derive(Copy, Clone, Debug)]
pub(crate) struct MutationEvent {
    pub(crate) accepted: bool,
    pub(crate) before_cost: f64,
    pub(crate) before_loss: f64,
    pub(crate) after: Option<(f64, f64)>,
}

const N_KINDS: usize = MutationChoice::ALL.len();

/// One population's [`AdaptiveMutationWeights`] statistics, indexed by [`MutationChoice`]. Each
/// population keeps its own across the search, as SymbolicRegression.jl's worker plugin state.
#[derive(Clone, Debug)]
pub struct MutationWeightsState {
    pub(crate) attempts: [f64; N_KINDS],
    pub(crate) successes: [f64; N_KINDS],
    pub(crate) multipliers: [f64; N_KINDS],
    pub(crate) active: [bool; N_KINDS],
}

impl Default for MutationWeightsState {
    fn default() -> Self {
        Self {
            attempts: [0.0; N_KINDS],
            successes: [0.0; N_KINDS],
            multipliers: [1.0; N_KINDS],
            active: MutationChoice::ALL.map(|c| !matches!(c, MutationChoice::Simplify | MutationChoice::DoNothing)),
        }
    }
}

impl MutationWeightsState {
    /// Multiplies each of `weights` by its kind's learned multiplier.
    pub(crate) fn condition_weights(&self, config: &AdaptiveMutationWeights, weights: &mut MutationWeights) {
        let exponent = config.adaptation_strength;
        if exponent == 0.0 {
            return;
        }
        for choice in MutationChoice::ALL {
            let m = self.multipliers[choice as usize];
            *choice.weight_mut(weights) *= if exponent == 0.5 { m.sqrt() } else { m.powf(exponent) };
        }
    }

    /// Accounts one mutation of kind `choice`.
    pub(crate) fn record(&mut self, config: &AdaptiveMutationWeights, choice: MutationChoice, event: MutationEvent) {
        let idx = choice as usize;
        if !self.active[idx] {
            return;
        }
        self.attempts[idx] += 1.0;
        let (before, after) = match config.reward {
            Reward::Cost => (event.before_cost, event.after.map(|a| a.0)),
            Reward::Loss => (event.before_loss, event.after.map(|a| a.1)),
        };
        if event.accepted && after.is_some_and(|a| a < before) {
            self.successes[idx] += 1.0;
        }
        let (mut total_successes, mut total_attempts, mut n) = (0.0, 0.0, 0.0);
        for i in 0..N_KINDS {
            if self.active[i] {
                total_successes += self.successes[i];
                total_attempts += self.attempts[i];
                n += 1.0;
            }
        }
        let mean_rate = (total_successes + n) / (total_attempts + 2.0 * n);
        let rate = (self.successes[idx] + 1.0) / (self.attempts[idx] + 2.0);
        let target = (rate / mean_rate).clamp(config.floor, 1.0 / config.floor);
        self.multipliers[idx] = (1.0 - config.smoothing) * self.multipliers[idx] + config.smoothing * target;

        let mut active_sum = 0.0;
        for i in 0..N_KINDS {
            if self.active[i] {
                active_sum += self.multipliers[i];
            }
        }
        let mean_multiplier = active_sum / n;
        for i in 0..N_KINDS {
            if self.active[i] {
                self.multipliers[i] /= mean_multiplier;
            }
        }
    }
}
