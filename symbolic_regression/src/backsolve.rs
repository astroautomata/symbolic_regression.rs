//! SymbolicRegression.jl's backsolve mutation (experimental): solve an expression for one of its
//! subtrees, then replace that subtree with a sparse weighted sum of library terms fitted to the
//! solved values.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

use dynamic_expressions::node::PNode;
use dynamic_expressions::node_utils::subtree_sizes;
use dynamic_expressions::{EvalOptions, Metadata, OpId, OperatorSet, PostfixExpr, XColumns, eval_tree_array};
use fastrand::Rng;
use ndarray::ArrayView2;
use num_traits::Float;

use crate::check_constraints::check_constraints;
use crate::complexity::compute_complexity;
use crate::dataset::{Dataset, TaggedDataset};
use crate::evaluate_inverse::subtree;
pub use crate::evaluate_inverse::{InverseFn, approx_inverse, eval_inverse_tree_array, eval_inverse_tree_array_masked};
use crate::options::Options;
use crate::pop_member::{Evaluator, PopMember};
use crate::random::usize_range;

/// SymbolicRegression.jl's `BacksolveMutation` parameters. The mutation's weight is
/// `MutationWeights::backsolve`.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "serde", serde(default))]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BacksolveOptions {
    /// Maximum number of basis terms in the library; at least one more than the number of features.
    pub max_library_size: usize,
    /// Maximum number of terms in the fitted weighted sum.
    pub max_terms: usize,
    /// Minimum relative reduction of the residual norm required to add another term.
    pub min_improvement: f64,
    /// Random subtrees each mutation tries before it fails.
    pub node_attempts: usize,
}

impl Default for BacksolveOptions {
    fn default() -> Self {
        Self {
            max_library_size: 500,
            max_terms: 8,
            min_improvement: 1e-3,
            node_attempts: 8,
        }
    }
}

/// Dense square matrix, row-major.
struct Square<T> {
    n: usize,
    data: Vec<T>,
}

impl<T: Float> Square<T> {
    fn at(&self, i: usize, j: usize) -> T {
        self.data[i * self.n + j]
    }

    fn select(&self, idx: &[usize]) -> Self {
        let data = idx
            .iter()
            .flat_map(|&i| idx.iter().map(move |&j| self.at(i, j)))
            .collect();
        Self { n: idx.len(), data }
    }
}

/// Solves `a x = b` by Cholesky factorization; `None` when `a` is not numerically positive
/// definite.
fn cholesky_solve<T: Float>(a: &Square<T>, b: &[T]) -> Option<Vec<T>> {
    let n = a.n;
    let mut l = vec![T::zero(); n * n];
    for j in 0..n {
        let mut d = a.at(j, j);
        for p in 0..j {
            d = d - l[j * n + p] * l[j * n + p];
        }
        if d.is_nan() || d <= T::zero() {
            return None;
        }
        let d = d.sqrt();
        l[j * n + j] = d;
        for i in j + 1..n {
            let mut s = a.at(i, j);
            for p in 0..j {
                s = s - l[i * n + p] * l[j * n + p];
            }
            l[i * n + j] = s / d;
        }
    }
    let mut z = b.to_vec();
    for i in 0..n {
        for p in 0..i {
            z[i] = z[i] - l[i * n + p] * z[p];
        }
        z[i] = z[i] / l[i * n + i];
    }
    for i in (0..n).rev() {
        for p in i + 1..n {
            z[i] = z[i] - l[p * n + i] * z[p];
        }
        z[i] = z[i] / l[i * n + i];
    }
    Some(z)
}

/// Solves the column-normalized normal equations `g_ij / (c_i c_j) x = g / c`, adding an
/// escalating ridge when the system is numerically singular. All zeros when no ridge works.
fn solve_gram<T: Float>(g: &Square<T>, rhs: &[T], col_norms: &[T]) -> Vec<T> {
    let n = g.n;
    let mut a = Square {
        n,
        data: (0..n * n)
            .map(|k| g.data[k] / (col_norms[k / n] * col_norms[k % n]))
            .collect(),
    };
    let b: Vec<T> = rhs.iter().zip(col_norms).map(|(&r, &c)| r / c).collect();
    if let Some(x) = cholesky_solve(&a, &b) {
        return x;
    }
    let hundred = T::from(100.0).unwrap();
    let diagonal: Vec<T> = (0..n).map(|i| a.at(i, i)).collect();
    let mut ridge = T::from(1e-10).unwrap();
    for _ in 0..3 {
        for (i, &d) in diagonal.iter().enumerate() {
            a.data[i * n + i] = d + ridge;
        }
        if let Some(x) = cholesky_solve(&a, &b) {
            return x;
        }
        ridge = ridge * hundred;
    }
    vec![T::zero(); n]
}

fn column_norms<T: Float>(g: &Square<T>) -> Vec<T> {
    (0..g.n).map(|i| g.at(i, i).max(T::zero()).sqrt()).collect()
}

fn gram<T: Float>(columns: &[&[T]]) -> Square<T> {
    let n = columns.len();
    let mut data = vec![T::zero(); n * n];
    for i in 0..n {
        for j in i..n {
            let v = dot(columns[i], columns[j]);
            data[i * n + j] = v;
            data[j * n + i] = v;
        }
    }
    Square { n, data }
}

fn dot<T: Float>(a: &[T], b: &[T]) -> T {
    a.iter().zip(b).fold(T::zero(), |s, (&x, &y)| s + x * y)
}

/// SymbolicRegression.jl's `stlsq`: sequential thresholded least squares for
/// `theta * coefficients ~= y`, with `theta` of shape `(n_samples, n_features)`. Returns the
/// coefficients and whether any is nonzero; all zeros when `y` does not match `theta`.
pub fn stlsq<T: Float>(theta: ArrayView2<'_, T>, y: &[T], lambda: f64, max_iter: usize) -> (Vec<T>, bool) {
    let (n_samples, n_features) = theta.dim();
    if y.len() != n_samples {
        return (vec![T::zero(); n_features], false);
    }
    let columns: Vec<Vec<T>> = theta.columns().into_iter().map(|c| c.to_vec()).collect();
    let refs: Vec<&[T]> = columns.iter().map(Vec::as_slice).collect();
    let g = gram(&refs);
    let rhs: Vec<T> = refs.iter().map(|c| dot(c, y)).collect();

    let tol = T::epsilon();
    let threshold = T::from(lambda).unwrap();
    let col_norms: Vec<T> = column_norms(&g).into_iter().map(|c| c.max(tol)).collect();
    let mut coefficients = solve_gram(&g, &rhs, &col_norms);
    for _ in 0..max_iter {
        let active: Vec<usize> = (0..n_features)
            .filter(|&i| coefficients[i].is_nan() || coefficients[i].abs() >= threshold)
            .collect();
        if active.is_empty() {
            return (vec![T::zero(); n_features], false);
        }
        let pick = |v: &[T]| active.iter().map(|&i| v[i]).collect::<Vec<T>>();
        let solved = solve_gram(&g.select(&active), &pick(&rhs), &pick(&col_norms));
        let mut next = vec![T::zero(); n_features];
        for (&i, &c) in active.iter().zip(&solved) {
            next[i] = c;
        }
        let change = next
            .iter()
            .zip(&coefficients)
            .fold(T::zero(), |s, (&a, &b)| s + (a - b) * (a - b))
            .sqrt();
        coefficients = next;
        if change < tol * T::from(10.0).unwrap() {
            break;
        }
    }
    for (c, &norm) in coefficients.iter_mut().zip(&col_norms) {
        *c = *c / norm;
    }
    let success = coefficients.iter().any(|c| c.abs() > tol * T::from(100.0).unwrap());
    (coefficients, success)
}

/// Inputs to [`greedy_forward_selection`] beyond the Gram system.
struct SelectionLimits<'a> {
    term_costs: &'a [usize],
    join_cost: usize,
    budget: Option<usize>,
    max_terms: usize,
    min_improvement: f64,
}

/// SymbolicRegression.jl's `greedy_forward_selection_gram`: orthogonal matching pursuit on the
/// Gram matrix `g`, cross vector `rhs`, and `yy = |y|^2`, adding the column most correlated with the
/// residual while its complexity fits the budget and it shrinks the residual norm by at least a
/// relative `min_improvement`.
fn greedy_forward_selection<T: Float>(g: &Square<T>, rhs: &[T], yy: T, limits: &SelectionLimits<'_>) -> (Vec<T>, bool) {
    let n = rhs.len();
    let mut coefficients = vec![T::zero(); n];
    if n == 0 || g.n != n {
        return (coefficients, false);
    }
    let tol = T::epsilon();
    let raw_norms = column_norms(g);
    let usable: Vec<bool> = raw_norms.iter().map(|&c| c > tol).collect();
    if !usable.contains(&true) {
        return (coefficients, false);
    }
    let col_norms: Vec<T> = raw_norms.into_iter().map(|c| c.max(tol)).collect();
    let g_norm = Square {
        n,
        data: (0..n * n)
            .map(|k| g.data[k] / (col_norms[k / n] * col_norms[k % n]))
            .collect(),
    };
    let rhs_norm: Vec<T> = rhs.iter().zip(&col_norms).map(|(&r, &c)| r / c).collect();
    let min_improvement = T::from(limits.min_improvement).unwrap();

    let mut remaining = limits.budget.unwrap_or(usize::MAX);
    let mut active: Vec<usize> = Vec::with_capacity(limits.max_terms);
    let mut x_active: Vec<T> = Vec::new();
    let mut residual_norm = yy.max(T::zero()).sqrt();
    let y_scale = residual_norm.max(T::one());
    let mut corr_resid = rhs_norm.clone();

    while active.len() < limits.max_terms {
        let extra = if active.is_empty() { 0 } else { limits.join_cost };
        let mut best: Option<usize> = None;
        let mut best_corr = T::zero();
        for j in 0..n {
            if !usable[j] || active.contains(&j) || limits.term_costs[j].saturating_add(extra) > remaining {
                continue;
            }
            let corr = corr_resid[j].abs();
            if corr > best_corr {
                best_corr = corr;
                best = Some(j);
            }
        }
        let Some(best) = best else {
            break;
        };

        let mut trial = active.clone();
        trial.push(best);
        let pick = |v: &[T]| trial.iter().map(|&i| v[i]).collect::<Vec<T>>();
        let x_trial = solve_gram(&g.select(&trial), &pick(rhs), &pick(&col_norms));
        let g_trial = g_norm.select(&trial);
        let quad = (0..trial.len()).fold(T::zero(), |s, i| {
            s + x_trial[i] * (0..trial.len()).fold(T::zero(), |r, j| r + g_trial.at(i, j) * x_trial[j])
        });
        let trial_norm = (yy - (T::one() + T::one()) * dot(&x_trial, &pick(&rhs_norm)) + quad)
            .max(T::zero())
            .sqrt();

        let improvement = (residual_norm - trial_norm) / residual_norm.max(tol * y_scale);
        if improvement.is_nan() || improvement < min_improvement {
            break;
        }

        remaining -= limits.term_costs[best] + extra;
        active = trial;
        x_active = x_trial;
        residual_norm = trial_norm;

        if residual_norm <= tol * T::from(100.0).unwrap() * y_scale {
            break;
        }
        for (j, c) in corr_resid.iter_mut().enumerate() {
            *c = rhs_norm[j]
                - active
                    .iter()
                    .zip(&x_active)
                    .fold(T::zero(), |s, (&k, &x)| s + g_norm.at(j, k) * x);
        }
    }

    if active.is_empty() {
        return (coefficients, false);
    }
    for (&k, &x) in active.iter().zip(&x_active) {
        coefficients[k] = x / col_norms[k];
    }
    let success = coefficients.iter().any(|c| c.abs() > tol * T::from(100.0).unwrap());
    (coefficients, success)
}

/// The library terms a backsolve fit combines, each with its values on the dataset.
pub struct BasisLibrary<T, Ops, const D: usize> {
    /// Distinct finite terms: the constant `1`, each feature, then subtrees of population members.
    pub terms: Vec<(PostfixExpr<T, Ops, D>, Vec<T>)>,
    /// How many candidate terms were evaluated to build the library.
    pub n_evaluated: usize,
}

/// SymbolicRegression.jl's `build_basis_library`: the constant `1`, every feature, then every
/// subtree (in pre-order) of the `top_k` lowest-loss members of `population`, capped at
/// `max_library_size` terms. Terms that are non-finite on the data or repeat an earlier term's
/// values are dropped. Fails when `max_library_size < 1 + n_features`.
pub fn build_basis_library<T, Ops, const D: usize>(
    dataset: &Dataset<T>,
    options: &Options<T, Ops, D>,
    n_features: usize,
    population: &[PopMember<T, Ops, D>],
    max_library_size: usize,
    top_k: usize,
) -> Result<BasisLibrary<T, Ops, D>, String>
where
    T: Float,
    Ops: OperatorSet<T = T>,
{
    if max_library_size < 1 + n_features {
        return Err(format!(
            "the backsolve library needs max_library_size >= 1 + the number of features; got max_library_size={max_library_size} with {n_features} features"
        ));
    }
    let leaf = |node: PNode, consts: Vec<T>| PostfixExpr::new(vec![node], consts, Metadata::default());
    let mut candidates = Vec::with_capacity(max_library_size);
    candidates.push(leaf(PNode::Const { idx: 0 }, vec![T::one()]));
    for feature in 0..n_features {
        candidates.push(leaf(
            PNode::Var {
                feature: feature as u16,
            },
            Vec::new(),
        ));
    }

    let mut members: Vec<&PopMember<T, Ops, D>> = population.iter().collect();
    members.sort_by(|a, b| {
        a.loss
            .partial_cmp(&b.loss)
            .unwrap_or_else(|| a.loss.is_nan().cmp(&b.loss.is_nan()))
    });
    'members: for member in members.into_iter().take(top_k) {
        for tree in member.expr.trees.iter() {
            for (start, end) in preorder_subtrees(&tree.nodes) {
                if candidates.len() >= max_library_size {
                    break 'members;
                }
                candidates.push(subtree(tree, start, end));
            }
        }
    }

    let x = XColumns::from(dataset.x.view());
    let opts = EvalOptions::default();
    let n_evaluated = candidates.len();
    let mut terms: Vec<(PostfixExpr<T, Ops, D>, Vec<T>)> = Vec::with_capacity(n_evaluated);
    let mut seen: HashMap<u64, usize> = HashMap::new();
    for term in candidates {
        let (values, complete) = eval_tree_array(&term, &options.operators.set, x, &opts);
        if !complete || values.iter().any(|v| !v.is_finite()) {
            continue;
        }
        let mut hasher = DefaultHasher::new();
        for v in &values {
            v.integer_decode().hash(&mut hasher);
        }
        let h = hasher.finish();
        if seen.get(&h).is_some_and(|&j| terms[j].1 == values) {
            continue;
        }
        seen.insert(h, terms.len());
        terms.push((term, values));
    }
    Ok(BasisLibrary { terms, n_evaluated })
}

/// Postfix `(start, end)` spans of every subtree, root first, each node before its children.
fn preorder_subtrees(nodes: &[PNode]) -> Vec<(usize, usize)> {
    let sizes = subtree_sizes(nodes);
    let mut out = Vec::with_capacity(nodes.len());
    let mut stack = vec![nodes.len() - 1];
    while let Some(root) = stack.pop() {
        out.push((root + 1 - sizes[root], root));
        if let PNode::Op { arity, .. } = nodes[root] {
            let mut end = root;
            let mut children = Vec::with_capacity(arity as usize);
            for _ in 0..arity {
                children.push(end - 1);
                end -= sizes[end - 1];
            }
            stack.extend(children);
        }
    }
    out
}

/// The selected built-in `+` and `*`, which a backsolve fit needs to build its weighted sum.
pub(crate) fn weighted_sum_ops<T: Float, Ops: OperatorSet, const D: usize>(
    options: &Options<T, Ops, D>,
) -> Option<(OpId, OpId)> {
    let binary = options.operators.ops_by_arity.get(1)?;
    let find = |name: &str| {
        binary
            .iter()
            .copied()
            .find(|&op| !options.operators.set.is_custom(op) && options.operators.set.name(op) == name)
    };
    Some((find("add")?, find("mul")?))
}

/// SymbolicRegression.jl's `combine_trees_weighted_sum`: `c1 * t1 + c2 * t2 + ...` over the terms
/// with a nonzero coefficient, writing `t` for `1 * t`. `None` when no coefficient is nonzero or
/// `+` or `*` is not among the selected operators.
pub fn combine_trees_weighted_sum<T: Float, Ops: OperatorSet, const D: usize>(
    terms: &[&PostfixExpr<T, Ops, D>],
    coefficients: &[T],
    options: &Options<T, Ops, D>,
) -> Option<PostfixExpr<T, Ops, D>> {
    let (add, mul) = weighted_sum_ops(options)?;
    let tol = T::epsilon() * T::from(100.0).unwrap();
    let mut nodes = Vec::new();
    let mut consts = Vec::new();
    let mut n_terms = 0;
    for (term, &c) in terms.iter().zip(coefficients) {
        if c.is_nan() || c.abs() <= tol {
            continue;
        }
        let scaled = (c - T::one()).abs() >= tol;
        if scaled {
            nodes.push(PNode::Const {
                idx: consts.len() as u16,
            });
            consts.push(c);
        }
        let offset = consts.len() as u16;
        nodes.extend(term.nodes.iter().map(|&n| match n {
            PNode::Const { idx } => PNode::Const { idx: idx + offset },
            other => other,
        }));
        consts.extend_from_slice(&term.consts);
        if scaled {
            nodes.push(PNode::Op { arity: 2, op: mul.id });
        }
        n_terms += 1;
        if n_terms > 1 {
            nodes.push(PNode::Op { arity: 2, op: add.id });
        }
    }
    (n_terms > 0).then(|| PostfixExpr::new(nodes, consts, Metadata::default()))
}

/// SymbolicRegression.jl's `BacksolveSetup`: a basis library with its Gram matrix and the
/// complexity each term adds as a weighted-sum addend, shared by the fits of one mutation.
pub struct BacksolveSetup<T, Ops, const D: usize> {
    pub basis: BasisLibrary<T, Ops, D>,
    pub backsolve: BacksolveOptions,
    gram: Square<T>,
    term_costs: Vec<usize>,
    join_cost: usize,
    addend_overhead: usize,
}

/// SymbolicRegression.jl's `prepare_backsolve_setup`; see [`build_basis_library`].
pub fn prepare_backsolve_setup<T, Ops, const D: usize>(
    dataset: &Dataset<T>,
    options: &Options<T, Ops, D>,
    n_features: usize,
    population: &[PopMember<T, Ops, D>],
    backsolve: BacksolveOptions,
    top_k: usize,
) -> Result<BacksolveSetup<T, Ops, D>, String>
where
    T: Float,
    Ops: OperatorSet<T = T>,
{
    let basis = build_basis_library(
        dataset,
        options,
        n_features,
        population,
        backsolve.max_library_size,
        top_k,
    )?;
    let columns: Vec<&[T]> = basis.terms.iter().map(|(_, v)| v.as_slice()).collect();
    let gram = gram(&columns);
    let constant = [PNode::Const { idx: 0 }];
    let constant_cost = compute_complexity(&constant, options);
    let binary_cost = |op: OpId| {
        let nodes = [constant[0], constant[0], PNode::Op { arity: 2, op: op.id }];
        compute_complexity(&nodes, options).saturating_sub(2 * constant_cost)
    };
    let (add, mul) = weighted_sum_ops(options).ok_or("backsolve needs `+` and `*` among the operators")?;
    let addend_overhead = constant_cost + binary_cost(mul);
    let term_costs = basis
        .terms
        .iter()
        .map(|(t, _)| compute_complexity(&t.nodes, options) + addend_overhead)
        .collect();
    Ok(BacksolveSetup {
        basis,
        backsolve,
        gram,
        term_costs,
        join_cost: binary_cost(add),
        addend_overhead,
    })
}

/// SymbolicRegression.jl's `fit_sparse_expression`: a weighted sum of library terms, chosen by
/// greedy forward selection, that fits `target`.
///
/// `extra_term` (typically the subtree being replaced) joins the library for this fit. Rows where
/// `valid_mask` is false take the extra term's own values as their target, or are dropped without
/// an extra term; fewer than `min(10, rows)` valid rows fail the fit. The result's complexity stays
/// within `max_complexity`, and it must evaluate to finite values.
pub fn fit_sparse_expression<T, Ops, const D: usize>(
    setup: &BacksolveSetup<T, Ops, D>,
    target: &[T],
    dataset: &Dataset<T>,
    options: &Options<T, Ops, D>,
    valid_mask: Option<&[bool]>,
    extra_term: Option<&PostfixExpr<T, Ops, D>>,
    max_complexity: Option<usize>,
) -> Option<PostfixExpr<T, Ops, D>>
where
    T: Float,
    Ops: OperatorSet<T = T>,
{
    let n_rows = target.len();
    if valid_mask.is_some_and(|m| m.iter().filter(|&&v| v).count() < n_rows.min(10)) {
        return None;
    }
    let x = XColumns::from(dataset.x.view());
    let opts = EvalOptions::default();
    let columns: Vec<&[T]> = setup.basis.terms.iter().map(|(_, v)| v.as_slice()).collect();

    let extra_values = match extra_term {
        Some(term) => {
            let (values, complete) = eval_tree_array(term, &options.operators.set, x, &opts);
            if !complete || values.iter().any(|v| !v.is_finite()) {
                return None;
            }
            Some(values)
        }
        None => None,
    };

    let (g, rhs, yy) = match (valid_mask, &extra_values) {
        (Some(mask), None) => {
            let rows: Vec<usize> = (0..n_rows).filter(|&i| mask[i]).collect();
            let masked: Vec<Vec<T>> = columns.iter().map(|c| rows.iter().map(|&i| c[i]).collect()).collect();
            let masked_target: Vec<T> = rows.iter().map(|&i| target[i]).collect();
            let refs: Vec<&[T]> = masked.iter().map(Vec::as_slice).collect();
            let rhs = refs.iter().map(|c| dot(c, &masked_target)).collect();
            (gram(&refs), rhs, dot(&masked_target, &masked_target))
        }
        (_, extra) => {
            let target_fit: Vec<T> = match (valid_mask, extra) {
                (Some(mask), Some(extra)) => (0..n_rows)
                    .map(|i| if mask[i] { target[i] } else { extra[i] })
                    .collect(),
                _ => target.to_vec(),
            };
            let mut rhs: Vec<T> = columns.iter().map(|c| dot(c, &target_fit)).collect();
            let g = match extra {
                None => Square {
                    n: setup.gram.n,
                    data: setup.gram.data.clone(),
                },
                Some(extra) => {
                    let p = setup.gram.n;
                    let cross: Vec<T> = columns.iter().map(|c| dot(c, extra)).collect();
                    let mut data = Vec::with_capacity((p + 1) * (p + 1));
                    for (row, &c) in setup.gram.data.chunks(p).zip(&cross) {
                        data.extend_from_slice(row);
                        data.push(c);
                    }
                    data.extend_from_slice(&cross);
                    data.push(dot(extra, extra));
                    rhs.push(dot(extra, &target_fit));
                    Square { n: p + 1, data }
                }
            };
            (g, rhs, dot(&target_fit, &target_fit))
        }
    };

    let mut terms: Vec<&PostfixExpr<T, Ops, D>> = setup.basis.terms.iter().map(|(t, _)| t).collect();
    let mut term_costs = setup.term_costs.clone();
    if let Some(term) = extra_term {
        terms.push(term);
        term_costs.push(compute_complexity(&term.nodes, options) + setup.addend_overhead);
    }

    let (coefficients, success) = greedy_forward_selection(
        &g,
        &rhs,
        yy,
        &SelectionLimits {
            term_costs: &term_costs,
            join_cost: setup.join_cost,
            budget: max_complexity,
            max_terms: setup.backsolve.max_terms,
            min_improvement: setup.backsolve.min_improvement,
        },
    );
    if !success {
        return None;
    }
    let result = combine_trees_weighted_sum(&terms, &coefficients, options)?;
    if max_complexity.is_some_and(|m| compute_complexity(&result.nodes, options) > m) {
        return None;
    }
    let (predicted, complete) = eval_tree_array(&result, &options.operators.set, x, &opts);
    if !complete || predicted.iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some(result)
}

/// `tree` with the subtree at postfix positions `start..=end` replaced by `sub`.
fn replace_subtree<T: Float, Ops, const D: usize>(
    tree: &PostfixExpr<T, Ops, D>,
    start: usize,
    end: usize,
    sub: &PostfixExpr<T, Ops, D>,
) -> PostfixExpr<T, Ops, D> {
    let offset = tree.consts.len() as u16;
    let mut nodes = Vec::with_capacity(tree.nodes.len() - (end + 1 - start) + sub.nodes.len());
    nodes.extend_from_slice(&tree.nodes[..start]);
    nodes.extend(sub.nodes.iter().map(|&n| match n {
        PNode::Const { idx } => PNode::Const { idx: idx + offset },
        other => other,
    }));
    nodes.extend_from_slice(&tree.nodes[end + 1..]);
    let mut consts = tree.consts.clone();
    consts.extend_from_slice(&sub.consts);
    let mut out = PostfixExpr::new(nodes, consts, tree.meta.clone());
    dynamic_expressions::compress_constants(&mut out);
    out
}

/// One backsolve rewrite: solve `tree` for the subtree at `node`, then fit a weighted sum to the
/// solved values within the size budget `curmaxsize` leaves for it.
struct Rewriter<'a, T: Float, Ops, const D: usize> {
    dataset: &'a Dataset<T>,
    options: &'a Options<T, Ops, D>,
    population: &'a [PopMember<T, Ops, D>],
    backsolve: BacksolveOptions,
    top_k: usize,
    curmaxsize: usize,
    setup: Option<BacksolveSetup<T, Ops, D>>,
}

impl<T: Float, Ops: OperatorSet<T = T>, const D: usize> Rewriter<'_, T, Ops, D> {
    fn rewrite(
        &mut self,
        tree: &PostfixExpr<T, Ops, D>,
        node: usize,
        evals: &mut f64,
        fraction: f64,
    ) -> Option<PostfixExpr<T, Ops, D>> {
        let ops = &self.options.operators.set;
        let y = self.dataset.y.as_slice().expect("y is contiguous");
        let (target, valid) = eval_inverse_tree_array_masked(tree, ops, self.dataset.x.view(), node, y);
        *evals += fraction;
        if valid.iter().filter(|&&v| v).count() < y.len().min(10) {
            return None;
        }
        let sizes = subtree_sizes(&tree.nodes);
        let start = node + 1 - sizes[node];
        let node_complexity = compute_complexity(&tree.nodes[start..=node], self.options);
        let parent_complexity = compute_complexity(&tree.nodes, self.options);
        let budget = (self.curmaxsize + node_complexity)
            .checked_sub(parent_complexity)
            .filter(|&b| b >= 1)?;
        if self.setup.is_none() {
            let setup = prepare_backsolve_setup(
                self.dataset,
                self.options,
                self.dataset.n_features,
                self.population,
                self.backsolve,
                self.top_k,
            )
            .expect("the search checks the backsolve options before it starts");
            *evals += setup.basis.n_evaluated as f64 * fraction;
            self.setup = Some(setup);
        }
        let setup = self.setup.as_ref().expect("built above");
        let old = subtree(tree, start, node);
        let fitted = fit_sparse_expression(
            setup,
            &target,
            self.dataset,
            self.options,
            Some(&valid),
            Some(&old),
            Some(budget),
        );
        *evals += 2.0 * fraction;
        Some(replace_subtree(tree, start, node, &fitted?))
    }
}

/// SymbolicRegression.jl's `backsolve_rewrite_random_node`: one rewrite of a random non-root
/// subtree, or `tree` unchanged when it is a single node or the rewrite fails.
pub fn backsolve_rewrite_random_node<T, Ops, const D: usize>(
    rng: &mut Rng,
    tree: &PostfixExpr<T, Ops, D>,
    dataset: &Dataset<T>,
    options: &Options<T, Ops, D>,
    population: &[PopMember<T, Ops, D>],
    curmaxsize: usize,
) -> PostfixExpr<T, Ops, D>
where
    T: Float,
    Ops: OperatorSet<T = T>,
{
    if tree.nodes.len() == 1 || weighted_sum_ops(options).is_none() {
        return tree.clone();
    }
    let node = usize_range(rng, 0..tree.nodes.len() - 1);
    let mut rewriter = Rewriter {
        dataset,
        options,
        population,
        backsolve: options.backsolve,
        top_k: 10,
        curmaxsize,
        setup: None,
    };
    rewriter
        .rewrite(tree, node, &mut 0.0, 1.0)
        .unwrap_or_else(|| tree.clone())
}

/// The backsolve mutation of `member`, as SymbolicRegression.jl's `mutate!` for
/// `BacksolveMutation`: up to `node_attempts` rewrites of random non-root subtrees, returning the
/// first child that satisfies the constraints and costs less than its parent, with the
/// evaluations spent. Library terms come from `population`.
pub(crate) fn backsolve_mutation<T, Ops, const D: usize>(
    rng: &mut Rng,
    member: &PopMember<T, Ops, D>,
    dataset: TaggedDataset<'_, T>,
    curmaxsize: usize,
    options: &Options<T, Ops, D>,
    evaluator: &mut Evaluator<T, D>,
    population: &[PopMember<T, Ops, D>],
) -> (Option<PopMember<T, Ops, D>>, f64)
where
    T: Float,
    Ops: OperatorSet<T = T>,
{
    let mut evals = 0.0;
    if options.template.is_some() || weighted_sum_ops(options).is_none() {
        return (None, evals);
    }
    let tree = &member.expr.trees[0];
    if tree.nodes.len() == 1 {
        return (None, evals);
    }
    let fraction = dataset.dataset_fraction();
    let mut rewriter = Rewriter {
        dataset: dataset.data(),
        options,
        population,
        backsolve: options.backsolve,
        top_k: population.len(),
        curmaxsize,
        setup: None,
    };
    let n_features = dataset.n_features;
    for _ in 0..options.backsolve.node_attempts {
        let node = usize_range(rng, 0..tree.nodes.len() - 1);
        let Some(candidate) = rewriter.rewrite(tree, node, &mut evals, fraction) else {
            continue;
        };
        if !check_constraints(&candidate, options, curmaxsize) {
            continue;
        }
        let mut child = PopMember::from_expr(candidate, n_features, options);
        child.evaluate(&dataset, options, evaluator);
        evals += fraction;

        let mut parent_cost = member.cost;
        if dataset.full_n_rows != dataset.n_rows {
            let mut parent = member.clone();
            parent.evaluate(&dataset, options, evaluator);
            parent_cost = parent.cost;
            evals += fraction;
        }
        let improved = if parent_cost.is_finite() {
            let margin = T::from(1e-10).unwrap() * parent_cost.abs().max(T::one());
            child.cost < parent_cost - margin
        } else {
            child.cost.is_finite()
        };
        if improved {
            return (Some(child), evals);
        }
    }
    (None, evals)
}
