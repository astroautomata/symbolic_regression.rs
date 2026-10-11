use dynamic_expressions::expression::PostfixExpr;
use dynamic_expressions::strings::{StringTreeOptions, string_tree};
use dynamic_expressions::{EvalPlan, OperatorSet, compile_plan};
use num_traits::Float;

use crate::options::Options;

/// Stores one expression component inline, or multiple components in a vector.
#[derive(Clone, Debug)]
pub enum OneOrMany<T> {
    Single(T),
    Multiple(Vec<T>),
}

impl<T> std::ops::Deref for OneOrMany<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        match self {
            Self::Single(value) => std::slice::from_ref(value),
            Self::Multiple(values) => values,
        }
    }
}

impl<T> std::ops::DerefMut for OneOrMany<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        match self {
            Self::Single(value) => std::slice::from_mut(value),
            Self::Multiple(values) => values,
        }
    }
}

impl<T> FromIterator<T> for OneOrMany<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut iter = iter.into_iter();
        let Some(first) = iter.next() else {
            return Self::Multiple(Vec::new());
        };
        let Some(second) = iter.next() else {
            return Self::Single(first);
        };
        let mut values = Vec::with_capacity(iter.size_hint().0 + 2);
        values.push(first);
        values.push(second);
        values.extend(iter);
        Self::Multiple(values)
    }
}

impl<T> From<Vec<T>> for OneOrMany<T> {
    fn from(mut values: Vec<T>) -> Self {
        if values.len() == 1 {
            Self::Single(values.pop().expect("one component"))
        } else {
            Self::Multiple(values)
        }
    }
}

/// A candidate equation: one tree per template subexpression, or a single tree in a search without
/// a template, and the template's parameter vectors, concatenated in declaration order.
#[derive(Debug)]
pub struct Expression<T, Ops, const D: usize> {
    pub trees: OneOrMany<PostfixExpr<T, Ops, D>>,
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
            trees: OneOrMany::Single(tree),
            params: Vec::new(),
        }
    }
}

impl<T: Float, Ops, const D: usize> Expression<T, Ops, D> {
    /// Every constant and parameter, in the order constant optimization sees them: each tree's
    /// constants in turn, then the parameters.
    pub fn constants(&self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.n_constants());
        for tree in self.trees.iter() {
            out.extend_from_slice(&tree.consts);
        }
        out.extend_from_slice(&self.params);
        out
    }

    /// Writes `values`, ordered as [`Self::constants`] returns them.
    pub fn set_constants(&mut self, values: &[T]) {
        assert_eq!(values.len(), self.n_constants(), "wrong number of constants");
        let mut rest = values;
        for tree in self.trees.iter_mut() {
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
    pub fn compile_plans(&self, n_features: usize) -> OneOrMany<EvalPlan<D>> {
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

#[cfg(test)]
mod tests {
    use dynamic_expressions::PNode;
    use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;

    use super::*;
    use crate::PopMember;

    #[test]
    fn plain_trees_and_plans_stay_inline_on_clone() {
        let options = Options::<f64, BuiltinOpsF64, 2>::default();
        let tree = PostfixExpr::new(vec![PNode::Var { feature: 0 }], vec![], Default::default());
        let member = PopMember::from_expr(tree, 1, &options);
        for member in [member.clone(), member] {
            assert!(matches!(&member.expr.trees, OneOrMany::Single(_)));
            assert!(matches!(&member.plans, OneOrMany::Single(_)));
        }
    }

    #[test]
    fn part_collection_and_clone_preserve_slices() {
        let empty: OneOrMany<usize> = std::iter::empty().collect();
        assert!(empty.is_empty());
        let single: OneOrMany<_> = [1].into_iter().collect();
        assert!(matches!(&single, OneOrMany::Single(1)));
        let mut many: OneOrMany<_> = [1, 2, 3].into_iter().collect();
        let copy = many.clone();
        many[1] = 9;
        assert_eq!(&*many, &[1, 9, 3]);
        assert_eq!(&*copy, &[1, 2, 3]);
        let converted: OneOrMany<_> = vec![1].into();
        assert!(matches!(&converted, OneOrMany::Single(1)));
    }
}
