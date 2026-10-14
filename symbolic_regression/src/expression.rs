use dynamic_expressions::expression::PostfixExpr;
use dynamic_expressions::strings::{StringTreeOptions, string_tree};
use dynamic_expressions::{EvalPlan, OperatorSet, compile_plan};
use num_traits::Float;

use crate::options::Options;

/// Stores one expression component inline, or multiple components in a vector.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(serialize = "T: serde::Serialize", deserialize = "T: serde::Deserialize<'de>"))
)]
#[derive(Debug)]
pub struct Expression<T, Ops, const D: usize> {
    #[cfg_attr(feature = "serde", serde(with = "tree_serde"))]
    pub trees: OneOrMany<PostfixExpr<T, Ops, D>>,
    pub params: Vec<T>,
}

/// Serializes `dynamic_expressions` trees from this crate, so that crate needs no serde support.
/// Serialization borrows the trees; deserialization rebuilds them through `PostfixExpr::new`.
#[cfg(feature = "serde")]
mod tree_serde {
    use dynamic_expressions::{Metadata, PNode, PostfixExpr};
    use serde::ser::SerializeStruct;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use super::OneOrMany;

    #[allow(dead_code)]
    #[derive(Serialize, Deserialize)]
    #[serde(remote = "PNode", rename = "PNode")]
    enum PNodeDef {
        Var { feature: u16 },
        Const { idx: u16 },
        Op { arity: u8, op: u16 },
    }

    #[allow(dead_code)]
    #[derive(Serialize, Deserialize)]
    #[serde(remote = "Metadata", rename = "Metadata")]
    struct MetadataDef {
        variable_names: Vec<String>,
    }

    struct NodesRef<'a>(&'a [PNode]);

    impl Serialize for NodesRef<'_> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serializer.collect_seq(self.0.iter().map(NodeRef))
        }
    }

    struct NodeRef<'a>(&'a PNode);

    impl Serialize for NodeRef<'_> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            PNodeDef::serialize(self.0, serializer)
        }
    }

    struct MetadataRef<'a>(&'a Metadata);

    impl Serialize for MetadataRef<'_> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            MetadataDef::serialize(self.0, serializer)
        }
    }

    struct TreeRef<'a, T, Ops, const D: usize>(&'a PostfixExpr<T, Ops, D>);

    impl<T: Serialize, Ops, const D: usize> Serialize for TreeRef<'_, T, Ops, D> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let tree = self.0;
            let mut state = serializer.serialize_struct("PostfixExpr", 3)?;
            state.serialize_field("nodes", &NodesRef(&tree.nodes))?;
            state.serialize_field("consts", &tree.consts)?;
            state.serialize_field("meta", &MetadataRef(&tree.meta))?;
            state.end()
        }
    }

    struct TreesRef<'a, T, Ops, const D: usize>(&'a [PostfixExpr<T, Ops, D>]);

    impl<T: Serialize, Ops, const D: usize> Serialize for TreesRef<'_, T, Ops, D> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serializer.collect_seq(self.0.iter().map(TreeRef))
        }
    }

    pub fn serialize<S: Serializer, T: Serialize, Ops, const D: usize>(
        trees: &OneOrMany<PostfixExpr<T, Ops, D>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match trees {
            OneOrMany::Single(tree) => serializer.serialize_newtype_variant("OneOrMany", 0, "Single", &TreeRef(tree)),
            OneOrMany::Multiple(trees) => {
                serializer.serialize_newtype_variant("OneOrMany", 1, "Multiple", &TreesRef(trees))
            }
        }
    }

    #[derive(Deserialize)]
    #[serde(transparent)]
    struct Node(#[serde(with = "PNodeDef")] PNode);

    #[derive(Deserialize)]
    #[serde(rename = "PostfixExpr", bound = "T: Deserialize<'de>")]
    struct Tree<T> {
        nodes: Vec<Node>,
        consts: Vec<T>,
        #[serde(with = "MetadataDef")]
        meta: Metadata,
    }

    impl<T> Tree<T> {
        fn into_expr<Ops, const D: usize>(self) -> PostfixExpr<T, Ops, D> {
            PostfixExpr::new(
                self.nodes.into_iter().map(|node| node.0).collect(),
                self.consts,
                self.meta,
            )
        }
    }

    pub fn deserialize<'de, De: Deserializer<'de>, T: Deserialize<'de>, Ops, const D: usize>(
        deserializer: De,
    ) -> Result<OneOrMany<PostfixExpr<T, Ops, D>>, De::Error> {
        Ok(match OneOrMany::<Tree<T>>::deserialize(deserializer)? {
            OneOrMany::Single(tree) => OneOrMany::Single(tree.into_expr()),
            OneOrMany::Multiple(trees) => OneOrMany::Multiple(trees.into_iter().map(Tree::into_expr).collect()),
        })
    }
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
