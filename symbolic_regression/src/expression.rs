use dynamic_expressions::expression::PostfixExpr;
use dynamic_expressions::strings::{StringTreeOptions, string_tree};
use dynamic_expressions::{EvalPlan, OperatorSet, compile_plan};
use num_traits::Float;

use crate::options::Options;

/// A candidate equation: one tree per template subexpression, or a single tree in a search without
/// a template, and the template's parameter vectors, concatenated in declaration order.
#[derive(Debug)]
pub struct Expression<T, Ops, const D: usize> {
    pub trees: Vec<PostfixExpr<T, Ops, D>>,
    pub params: Vec<T>,
}

impl<T: Clone, Ops, const D: usize> Clone for Expression<T, Ops, D> {
    fn clone(&self) -> Self {
        Self {
            trees: self.trees.clone(),
            params: self.params.clone(),
        }
    }
}

impl<T, Ops, const D: usize> From<PostfixExpr<T, Ops, D>> for Expression<T, Ops, D> {
    fn from(tree: PostfixExpr<T, Ops, D>) -> Self {
        Self {
            trees: vec![tree],
            params: Vec::new(),
        }
    }
}

impl<T: Float, Ops, const D: usize> Expression<T, Ops, D> {
    /// Every constant and parameter, in the order constant optimization sees them: each tree's
    /// constants in turn, then the parameters.
    pub fn constants(&self) -> Vec<T> {
        let mut out: Vec<T> = self.trees.iter().flat_map(|t| t.consts.iter().copied()).collect();
        out.extend_from_slice(&self.params);
        out
    }

    /// Writes `values`, ordered as [`Self::constants`] returns them.
    pub fn set_constants(&mut self, values: &[T]) {
        assert_eq!(values.len(), self.n_constants(), "wrong number of constants");
        let mut rest = values;
        for tree in &mut self.trees {
            let (mine, tail) = rest.split_at(tree.consts.len());
            tree.consts.copy_from_slice(mine);
            rest = tail;
        }
        self.params.copy_from_slice(rest);
    }

    pub fn n_constants(&self) -> usize {
        self.trees.iter().map(|t| t.consts.len()).sum::<usize>() + self.params.len()
    }

    /// A hash of the trees' structure, ignoring constants and parameters.
    pub fn hash_nodes(&self) -> u64 {
        self.trees.iter().fold(0, |h, t| h.rotate_left(5) ^ t.hash_nodes())
    }

    /// One evaluation plan per tree, for data with `n_features` features (a template's widest
    /// subexpression).
    pub fn compile_plans(&self, n_features: usize) -> Vec<EvalPlan<D>> {
        self.trees
            .iter()
            .map(|t| compile_plan(&t.nodes, n_features, t.consts.len()))
            .collect()
    }

    /// Prints the expression: the tree, or a template as SymbolicRegression.jl prints it
    /// (`f = #1 * #2; p = [1, 2]`), ignoring `opts.variable_names`.
    pub fn string(&self, options: &Options<T, Ops, D>, opts: StringTreeOptions<'_>) -> String
    where
        T: core::fmt::Display,
        Ops: OperatorSet,
    {
        let ops = &options.operators.set;
        match &options.template {
            None => string_tree(&self.trees[0], ops, opts),
            Some(template) => template.string(&self.trees, &self.params, ops, opts.pretty, opts.print_precision),
        }
    }
}
