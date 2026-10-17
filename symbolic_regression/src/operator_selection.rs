use dynamic_expressions::{OpId, Operators};
use fastrand::Rng;

use crate::random::usize_range;

pub trait OperatorsSampling {
    fn total_ops_up_to(&self, max_arity: usize) -> usize;
    fn sample_arity(&self, rng: &mut Rng, max_arity: usize) -> usize;
    fn sample_op(&self, rng: &mut Rng, arity: usize) -> OpId;
}

impl<Ops, const D: usize> OperatorsSampling for Operators<Ops, D> {
    fn total_ops_up_to(&self, max_arity: usize) -> usize {
        let max_arity = max_arity.min(D);
        (1..=max_arity).map(|a| self.nops(a)).sum()
    }

    fn sample_arity(&self, rng: &mut Rng, max_arity: usize) -> usize {
        let max_arity = max_arity.min(D);
        let total: usize = (1..=max_arity).map(|a| self.nops(a)).sum();
        assert!(total > 0, "no operators available up to arity={max_arity}");
        let mut r = usize_range(rng, 0..total);
        for arity in 1..=max_arity {
            let n = self.nops(arity);
            if r < n {
                return arity;
            }
            r -= n;
        }
        unreachable!()
    }

    fn sample_op(&self, rng: &mut Rng, arity: usize) -> OpId {
        let v = &self.ops_by_arity[arity - 1];
        let i = usize_range(rng, 0..v.len());
        v[i]
    }
}
