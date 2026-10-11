use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
use dynamic_expressions::{EvalOptions, Metadata, OperatorSet, Operators, PNode, PostfixExpr, eval_tree_array};
use fastrand::Rng;
use ndarray::{Array1, Array2, array};

use super::common::infix;
use crate::adaptive_parsimony::RunningSearchStatistics;
use crate::backsolve::{
    BacksolveOptions, approx_inverse, backsolve_rewrite_random_node, build_basis_library, combine_trees_weighted_sum,
    eval_inverse_tree_array, eval_inverse_tree_array_masked, fit_sparse_expression, prepare_backsolve_setup, stlsq,
};
use crate::check_constraints::check_constraints;
use crate::dataset::TaggedDataset;
use crate::mutate::{self, NextGenerationCtx};
use crate::pop_member::Evaluator;
use crate::{Dataset, MutationWeights, Options, PopMember};

type Ops = BuiltinOpsF64;
type Scalar = fn(f64) -> f64;

fn options_with<const N: usize>(unary: &[&str], binary: &[&str], ternary: &[&str]) -> Options<f64, Ops, N> {
    let mut operators = Operators::<Ops, N>::new(Ops::new());
    for (arity, names) in [(1u8, unary), (2, binary), (3, ternary)] {
        for name in names {
            let op = operators.set.lookup_with_arity(name, arity).unwrap();
            operators.push(op);
        }
    }
    Options {
        operators,
        ..Default::default()
    }
}

fn options(unary: &[&str], binary: &[&str]) -> Options<f64, Ops, 2> {
    options_with(unary, binary, &[])
}

fn expr<const N: usize>(options: &Options<f64, Ops, N>, src: &str) -> PostfixExpr<f64, Ops, N> {
    infix(&options.operators.set, src)
}

fn values<const N: usize>(
    options: &Options<f64, Ops, N>,
    tree: &PostfixExpr<f64, Ops, N>,
    x: &Array2<f64>,
) -> Vec<f64> {
    let (out, ok) = eval_tree_array(tree, &options.operators.set, x.view(), &EvalOptions::default());
    assert!(ok, "evaluation failed");
    out
}

fn close(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= 1e-8 * x.abs().max(y.abs()).max(1.0))
}

fn mse(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>() / a.len() as f64
}

fn residual(theta: &Array2<f64>, coefficients: &[f64], y: &[f64]) -> f64 {
    let fit = theta.dot(&Array1::from(coefficients.to_vec()));
    fit.iter().zip(y).map(|(f, v)| (f - v) * (f - v)).sum::<f64>().sqrt()
}

fn member(
    options: &Options<f64, Ops, 2>,
    dataset: &Dataset<f64>,
    tree: PostfixExpr<f64, Ops, 2>,
    loss: f64,
) -> PopMember<f64, Ops, 2> {
    let mut m = PopMember::from_expr(tree, dataset.n_features, options);
    m.loss = loss;
    m
}

fn two_feature_data() -> (Array2<f64>, Vec<f64>, Dataset<f64>) {
    let x = array![[1.0, 2.0, 3.0, 4.0], [0.5, 1.0, 1.5, 2.0]];
    let y = vec![3.5, 7.0, 10.5, 14.0];
    let dataset = Dataset::new(x.clone(), Array1::from(y.clone()));
    (x, y, dataset)
}

#[test]
fn stlsq_algorithm_basic_functionality() {
    let theta = array![
        [1.0, 0.1, 0.2],
        [2.0, -0.3, 0.1],
        [3.0, 0.4, -0.5],
        [4.0, -0.2, 0.3],
        [5.0, 0.5, -0.1]
    ];
    let y: Vec<f64> = theta
        .rows()
        .into_iter()
        .map(|r| 2.0 * r[0] + 0.0 * r[1] + 3.0 * r[2])
        .collect();
    let (c, ok) = stlsq(theta.view(), &y, 0.1, 10);
    assert!(ok);
    assert!((c[0] - 2.0).abs() < 1e-6);
    assert!(c[1].abs() < 1e-6);
    assert!((c[2] - 3.0).abs() < 1e-6);

    let theta2 = array![[1.0, 2.0], [3.0, 4.0]];
    let (c2, ok2) = stlsq(theta2.view(), &[0.001, 0.002], 1.0, 10);
    assert!(!ok2);
    assert!(c2.iter().all(|&v| v == 0.0));

    let (c3, ok3) = stlsq(theta2.view(), &[1.0, 2.0, 3.0], 0.01, 10);
    assert!(!ok3);
    assert_eq!(c3.len(), 2);
    assert!(c3.iter().all(|&v| v == 0.0));

    let theta4 = array![[1.0, 0.5, 0.1], [2.0, 1.0, 0.2], [3.0, 1.5, 0.3]];
    let y4 = [1.0, 2.0, 3.0];
    let (c4, ok4) = stlsq(theta4.view(), &y4, 0.05, 20);
    assert!(ok4);
    assert!(residual(&theta4, &c4, &y4) < 1e-3);

    let theta5 = array![[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]];
    let (c5, ok5) = stlsq(theta5.view(), &y4, 0.01, 10);
    assert!(ok5);
    assert!(residual(&theta5, &c5, &y4) < 1e-8);
}

#[test]
fn tree_combination_with_weighted_sum() {
    let options = options(&["sin"], &["+", "*"]);
    let x = array![[1.0, 2.0, 3.0]];
    let x1 = expr(&options, "x1");
    let one = expr(&options, "1.0");
    let combine = |trees: &[&PostfixExpr<f64, Ops, 2>], c: &[f64], o: &Options<f64, Ops, 2>| {
        combine_trees_weighted_sum(trees, c, o)
    };

    let r1 = combine(&[&x1], &[1.0], &options).unwrap();
    assert!(close(&values(&options, &r1, &x), &[1.0, 2.0, 3.0]));
    let r2 = combine(&[&x1], &[2.5], &options).unwrap();
    assert!(close(&values(&options, &r2, &x), &[2.5, 5.0, 7.5]));
    let r3 = combine(&[&x1, &one], &[2.0, 3.0], &options).unwrap();
    assert!(close(&values(&options, &r3, &x), &[5.0, 7.0, 9.0]));
    assert!(combine(&[&x1, &one], &[0.0, 0.0], &options).is_none());
    let r5 = combine(&[&x1, &one], &[0.0, 5.0], &options).unwrap();
    assert!(close(&values(&options, &r5, &x), &[5.0, 5.0, 5.0]));

    let no_add = self::options(&["sin"], &["*"]);
    assert!(combine(&[&x1, &one], &[2.0, 3.0], &no_add).is_none());
    let no_mult = self::options(&["sin"], &["+"]);
    assert!(combine(&[&x1, &one], &[2.0, 3.0], &no_mult).is_none());
    assert!(combine(&[&x1], &[2.5], &no_mult).is_none());

    let r9 = combine(&[&x1, &one], &[1.0, 2.0], &options).unwrap();
    assert!(close(&values(&options, &r9, &x), &[3.0, 4.0, 5.0]));
}

#[test]
fn fit_sparse_expression_full_pipeline() {
    let options = options(&["sin", "cos"], &["+", "*"]);
    assert_eq!(options.backsolve.max_library_size, 500);
    let (x, y, dataset) = two_feature_data();
    let setup = prepare_backsolve_setup(&dataset, &options, 2, &[], options.backsolve, 10).unwrap();
    let result = fit_sparse_expression(&setup, &y, &dataset, &options, None, None, None).unwrap();
    assert!(mse(&values(&options, &result, &x), &y) < 0.1);
}

#[test]
fn build_basis_library_terms() {
    let options = options(&["sin", "cos"], &["+", "*", "-"]);
    let (_, _, dataset) = two_feature_data();
    let build = |population: &[PopMember<f64, Ops, 2>], max_library_size: usize, top_k: usize| {
        build_basis_library(&dataset, &options, 2, population, max_library_size, top_k)
    };

    assert!(build(&[], 2, 10).is_err());

    let basis = build(&[], 200, 10).unwrap();
    assert_eq!(basis.terms.len(), 3);
    assert!(basis.terms.iter().all(|(_, v)| v.len() == 4));

    let tree1 = expr(&options, "x1 + x2");
    let tree2 = expr(&options, "sin(x1)");
    let tree3 = expr(&options, "x1");
    let pop = [
        member(&options, &dataset, tree1.clone(), 1.0),
        member(&options, &dataset, tree2.clone(), 2.0),
        member(&options, &dataset, tree3.clone(), 3.0),
    ];
    let basis2 = build(&pop, 200, 10).unwrap();
    assert!(basis2.terms.len() > 3);

    let single_tree = [member(&options, &dataset, tree1.clone(), 0.1)];
    assert!(build(&single_tree, 200, 10).unwrap().terms.len() > 3);

    let dup = [
        member(&options, &dataset, tree1.clone(), 1.0),
        member(&options, &dataset, tree1.clone(), 2.0),
    ];
    let basis_dup = build(&dup, 200, 10).unwrap();
    let basis_single = build(&dup[..1], 200, 10).unwrap();
    assert_eq!(basis_dup.terms.len(), basis_single.terms.len());

    let pop_topk = [
        member(&options, &dataset, tree1, 10.0),
        member(&options, &dataset, tree2, 10.0),
        member(&options, &dataset, tree3, 0.1),
    ];
    let basis_topk = build(&pop_topk, 200, 1).unwrap();
    let basis_all = build(&pop_topk, 200, 10).unwrap();
    assert!(basis_topk.terms.len() <= basis_all.terms.len());

    assert!(build(&pop, 4, 10).unwrap().terms.len() <= 4);
}

#[test]
fn fit_sparse_expression_with_population_for_backsolve() {
    let options = options(&["sin", "cos"], &["+", "*"]);
    let (x, y, dataset) = two_feature_data();
    let pop = [
        member(&options, &dataset, expr(&options, "x1 + 1.0"), 1.0),
        member(&options, &dataset, expr(&options, "x2 + 2.0"), 2.0),
    ];
    let setup = prepare_backsolve_setup(&dataset, &options, 2, &pop, options.backsolve, 10).unwrap();
    let result = fit_sparse_expression(&setup, &y, &dataset, &options, None, None, None).unwrap();
    assert!(mse(&values(&options, &result, &x), &y) < 1.0);
}

#[test]
fn integration_backsolve_rewrite_with_sparse_fit() {
    let options = options(&["sin", "cos"], &["+", "*", "-"]);
    let (x, _, dataset) = two_feature_data();
    let tree = expr(&options, "(x1 + x2) * sin(x1)");
    let mut rng = Rng::with_seed(2);
    let mutated = backsolve_rewrite_random_node(&mut rng, &tree, &dataset, &options, &[], options.maxsize);
    assert_ne!(values(&options, &mutated, &x), values(&options, &tree, &x));

    let simple = expr(&options, "x1");
    let unchanged = backsolve_rewrite_random_node(&mut rng, &simple, &dataset, &options, &[], options.maxsize);
    assert_eq!(unchanged.nodes, simple.nodes);
}

#[test]
fn edge_cases_and_error_handling() {
    let x = array![[1.0, 2.0, 3.0]];
    let y = [1.0, 2.0, 3.0];
    let dataset = Dataset::new(x.clone(), Array1::from(y.to_vec()));
    let empty = options(&[], &[]);
    assert!(prepare_backsolve_setup(&dataset, &empty, 1, &[], empty.backsolve, 10).is_err());

    let strict = Options {
        backsolve: BacksolveOptions {
            min_improvement: 1e10,
            ..Default::default()
        },
        ..options(&["sin"], &["+", "*"])
    };
    let setup = prepare_backsolve_setup(&dataset, &strict, 1, &[], strict.backsolve, 10).unwrap();
    assert!(fit_sparse_expression(&setup, &y, &dataset, &strict, None, None, None).is_none());

    let (_, ok_empty) = stlsq(Array2::<f64>::zeros((0, 3)).view(), &[], 0.01, 10);
    assert!(!ok_empty);
    let (c_single, _) = stlsq(array![[1.0, 2.0, 3.0]].view(), &[6.0], 0.01, 10);
    assert_eq!(c_single.len(), 3);

    let y_inf = [1.0, f64::INFINITY, 3.0];
    let dataset_inf = Dataset::new(x, Array1::from(y_inf.to_vec()));
    let opts = options(&["sin"], &["+", "*"]);
    let setup = prepare_backsolve_setup(&dataset_inf, &opts, 1, &[], opts.backsolve, 10).unwrap();
    assert!(fit_sparse_expression(&setup, &y_inf, &dataset_inf, &opts, None, None, None).is_none());
}

fn same(a: f64, b: f64) -> bool {
    a == b || (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12 * a.abs().max(1.0)
}

#[test]
fn inverse_functions() {
    let pairs: [(&str, Scalar); 20] = [
        ("sin", |v| if v.abs() <= 1.0 { v.asin() } else { f64::NAN }),
        ("asin", f64::sin),
        ("cos", |v| if v.abs() <= 1.0 { v.acos() } else { f64::NAN }),
        ("acos", f64::cos),
        ("tan", f64::atan),
        ("atan", f64::tan),
        ("cosh", |v| if v < 1.0 { f64::NAN } else { v.acosh() }),
        ("acosh", f64::cosh),
        ("tanh", |v| ((v + 1.0).rem_euclid(2.0) - 1.0).atanh()),
        ("atanh", f64::tanh),
        ("square", |v| if v < 0.0 { f64::NAN } else { v.sqrt() }),
        ("exp", |v| if v <= 0.0 { f64::NAN } else { v.ln() }),
        ("log", f64::exp),
        ("log2", f64::exp2),
        ("exp2", |v| if v <= 0.0 { f64::NAN } else { v.log2() }),
        ("log10", |v| 10f64.powf(v)),
        ("sqrt", |v| v * v),
        ("cube", f64::cbrt),
        ("cbrt", |v| v * v * v),
        ("neg", |v| -v),
    ];
    for (name, expected) in &pairs {
        let inverse = approx_inverse::<f64>(name, 1, 0).unwrap_or_else(|| panic!("{name} has an inverse"));
        for v in [-2.5, -0.9, -0.3, 0.0, 0.2, 0.7, 1.0, 1.5, 3.0] {
            assert!(same(inverse(v, &[0.0]), expected(v)), "{name} at {v}");
        }
    }
    let exp1m = approx_inverse::<f64>("log1p", 1, 0).unwrap();
    assert!(same(exp1m(1.0, &[0.0]), 1f64.exp() - 1.0));
    let log1p = approx_inverse::<f64>("expm1", 1, 0).unwrap();
    assert!(same(log1p(exp1m(0.3, &[0.0]), &[0.0]), 0.3));
    assert!(approx_inverse::<f64>("abs", 1, 0).is_none());
    assert!(approx_inverse::<f64>("relu", 1, 0).is_none());

    let div_by_2 = approx_inverse::<f64>("mul", 2, 0).unwrap();
    assert_eq!(div_by_2(6.0, &[0.0, 2.0]), 3.0);
    let log_base_2 = approx_inverse::<f64>("pow", 2, 1).unwrap();
    assert!(same(log_base_2(8.0, &[2.0, 0.0]), 3.0));
    let sqrt = approx_inverse::<f64>("pow", 2, 0).unwrap();
    assert_eq!(sqrt(9.0, &[0.0, 2.0]), 3.0);
    for position in 0..2 {
        assert!(approx_inverse::<f64>("mod", 2, position).is_none());
    }

    let args = [2.0, 3.0, 4.0];
    for (position, expected) in [(0, 2.0), (1, 3.0), (2, 4.0)] {
        let inverse = approx_inverse::<f64>("fma", 3, position).unwrap();
        assert!((inverse(10.0, &args) - expected).abs() < 1e-10);
    }
    assert!(approx_inverse::<f64>("clamp", 3, 1).is_none());
}

#[test]
fn evaluate_inverse_trees() {
    let options = options(&["sin"], &["+", "*"]);
    let ops = &options.operators.set;
    let x = array![[1.0]];

    let (inv, ok) = eval_inverse_tree_array(&expr(&options, "sin(x1)"), ops, x.view(), 0, &[0.5]);
    assert!(ok);
    assert!((inv[0] - 0.5f64.asin()).abs() < 1e-10);

    let (inv, ok) = eval_inverse_tree_array(&expr(&options, "x1 + 2.0"), ops, x.view(), 0, &[5.0]);
    assert!(ok);
    assert!((inv[0] - 3.0).abs() < 1e-10);

    let (inv, ok) = eval_inverse_tree_array(&expr(&options, "sin(x1 * 2.0)"), ops, x.view(), 0, &[0.5]);
    assert!(ok);
    assert!((inv[0] - 0.5f64.asin() / 2.0).abs() < 1e-10);

    let x3 = array![[1.0, 2.0, 3.0]];
    let (inv, ok) = eval_inverse_tree_array(&expr(&options, "x1 + 1.0"), ops, x3.view(), 0, &[2.0, 3.0, 4.0]);
    assert!(ok);
    assert!(close(&inv, &[1.0, 2.0, 3.0]));

    let (inv, ok) = eval_inverse_tree_array(&expr(&options, "x1"), ops, x.view(), 5, &[1.0]);
    assert!(!ok);
    assert_eq!(inv, vec![1.0]);

    let div = self::options(&[], &["+", "/"]);
    let tree = expr(&div, "(x1 + 1.0 / x2) + 2.0 / x2");
    let xs = array![[0.0, 0.0, 0.0], [0.0, 1.0, 2.0]];
    let (inv, valid) = eval_inverse_tree_array_masked(&tree, &div.operators.set, xs.view(), 0, &[20.0; 3]);
    assert_eq!(valid, vec![false, true, true]);
    assert!(close(&inv[1..], &[17.0, 18.5]));

    let fma = options_with::<3>(&[], &["/"], &["fma"]);
    let fma_id = fma.operators.ops_by_arity[2][0].id;
    let leaves = |nodes: Vec<PNode>| PostfixExpr::<f64, Ops, 3>::new(nodes, vec![1.0, 2.0], Metadata::default());
    let tree = leaves(vec![
        PNode::Var { feature: 0 },
        PNode::Var { feature: 1 },
        PNode::Var { feature: 2 },
        PNode::Op { arity: 3, op: fma_id },
    ]);
    let x321 = array![[3.0], [2.0], [1.0]];
    for (node, expected) in [(0, 3.0), (1, 2.0), (2, 1.0)] {
        let (inv, ok) = eval_inverse_tree_array(&tree, &fma.operators.set, x321.view(), node, &[7.0]);
        assert!(ok);
        assert!((inv[0] - expected).abs() < 1e-10);
    }

    let div_id = fma.operators.ops_by_arity[1][0].id;
    let tree = leaves(vec![
        PNode::Const { idx: 0 },
        PNode::Var { feature: 1 },
        PNode::Op { arity: 2, op: div_id },
        PNode::Const { idx: 1 },
        PNode::Var { feature: 1 },
        PNode::Op { arity: 2, op: div_id },
        PNode::Var { feature: 0 },
        PNode::Op { arity: 3, op: fma_id },
    ]);
    let (inv, valid) = eval_inverse_tree_array_masked(&tree, &fma.operators.set, xs.view(), 6, &[10.0; 3]);
    assert_eq!(valid, vec![false, true, true]);
    assert!(close(&inv[1..], &[8.0, 9.5]));
}

#[test]
fn operators_without_inverses_fail_gracefully() {
    let options = options(&["abs", "sin"], &["+", "*"]);
    let x = array![[1.0, 2.0, 3.0]];
    let dataset = Dataset::new(x.clone(), array![2.0, 3.0, 4.0]);
    let tree = expr(&options, "abs(x1) + 1.0");
    let (_, ok) = eval_inverse_tree_array(&tree, &options.operators.set, x.view(), 0, &[2.0, 3.0, 4.0]);
    assert!(!ok);

    let (custom, names) = crate::prelude::parse_operator_tokens::<f64, _>(["+", "*", "inv(x) = 1/x"]).unwrap();
    let custom_options = Options::<f64, _, 2> {
        operators: BuiltinOpsF64::new().with_custom(custom).from_names(&names).unwrap(),
        ..Default::default()
    };
    let tree = infix::<_, 2>(&custom_options.operators.set, "inv(x1)");
    let (_, valid) = eval_inverse_tree_array_masked(&tree, &custom_options.operators.set, x.view(), 0, &[1.0; 3]);
    assert_eq!(valid, vec![false; 3]);
    let mut rng = Rng::with_seed(0);
    let same_tree = backsolve_rewrite_random_node(&mut rng, &tree, &dataset, &custom_options, &[], 30);
    assert_eq!(same_tree.nodes, tree.nodes);
}

#[test]
fn backsolve_rewrite_random_node_cases() {
    let options = options(&["sin", "cos"], &["+", "*", "-", "/"]);
    let x = array![[1.0, 2.0, 3.0]];
    let dataset = Dataset::new(x, array![2.0, 3.0, 4.0]);
    let mut rng = Rng::with_seed(0);
    let tree = expr(&options, "sin(x1) + 2.0");
    let mutated = backsolve_rewrite_random_node(&mut rng, &tree, &dataset, &options, &[], options.maxsize);
    assert!(!mutated.nodes.is_empty());

    let single_options = self::options(&["sin"], &["+"]);
    let single = expr(&single_options, "1.0");
    let one_row = Dataset::new(array![[1.0]], array![1.0]);
    let unchanged = backsolve_rewrite_random_node(&mut rng, &single, &one_row, &single_options, &[], 30);
    assert_eq!(
        (unchanged.nodes, unchanged.consts),
        (single.nodes.clone(), single.consts.clone())
    );

    let plain = self::options(&["sin"], &["+", "*"]);
    let out_of_range = Dataset::new(array![[1.0, 2.0]], array![10.0, 20.0]);
    let tree = expr(&plain, "sin(x1)");
    let unchanged = backsolve_rewrite_random_node(&mut rng, &tree, &out_of_range, &plain, &[], 30);
    assert_eq!(unchanged.nodes, tree.nodes);
}

#[test]
fn backsolve_rewrite_random_node_ternary_tree() {
    let options = options_with::<3>(&[], &["+", "*"], &["fma"]);
    let fma_id = options.operators.ops_by_arity[2][0].id;
    let x = array![[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]];
    let dataset = Dataset::new(x.clone(), x.row(0).mapv(|v| 2.0 * v + 3.0));
    let tree = PostfixExpr::<f64, Ops, 3>::new(
        vec![
            PNode::Var { feature: 0 },
            PNode::Const { idx: 0 },
            PNode::Const { idx: 1 },
            PNode::Op { arity: 3, op: fma_id },
        ],
        vec![2.0, 7.0],
        Metadata::default(),
    );
    let base = values(&options, &tree, &x);
    let mut rng = Rng::with_seed(3);
    let mut n_changed = 0;
    for _ in 0..20 {
        let mutated = backsolve_rewrite_random_node(&mut rng, &tree, &dataset, &options, &[], options.maxsize);
        if !close(&values(&options, &mutated, &x), &base) {
            n_changed += 1;
        }
    }
    assert!(n_changed > 0);
}

#[test]
fn mutation_weights_backsolve_field() {
    assert_eq!(MutationWeights::default().backsolve, 0.0);
    let on = MutationWeights {
        backsolve: 0.5,
        ..Default::default()
    };
    assert_eq!(on.backsolve, 0.5);
}

#[test]
fn backsolve_options() {
    let defaults = Options::<f64, Ops, 2>::default();
    assert_eq!(defaults.backsolve, BacksolveOptions::default());
    assert_eq!(
        defaults.backsolve,
        BacksolveOptions {
            max_library_size: 500,
            max_terms: 8,
            min_improvement: 1e-3,
            node_attempts: 8,
        }
    );
    assert_eq!(defaults.mutation_weights.backsolve, 0.0);

    let custom = Options::<f64, Ops, 2> {
        backsolve: BacksolveOptions {
            max_terms: 4,
            ..Default::default()
        },
        ..Default::default()
    };
    assert_eq!(custom.backsolve.max_terms, 4);

    let dataset = Dataset::new(Array2::zeros((3, 5)), Array1::zeros(5));
    let mut small = Options::<f64, Ops, 2> {
        backsolve: BacksolveOptions {
            max_library_size: 3,
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(small.check_dataset(&dataset).is_ok(), "backsolve is off");
    small.mutation_weights.backsolve = 1.0;
    assert!(small.check_dataset(&dataset).is_err());
    small.backsolve.max_library_size = 4;
    assert!(small.check_dataset(&dataset).is_ok());
}

/// SymbolicRegression.jl's "backsolve in mutation pipeline": with only backsolve enabled,
/// `next_generation` rewrites `x1 + 1.0` into an exact fit of `y = 2 x1`, and every child it
/// returns satisfies the constraints with a finite loss.
#[test]
fn backsolve_fires_in_next_generation() {
    let x = array![[1.0, 2.0, 3.0]];
    let dataset = Dataset::new(x, array![2.0, 4.0, 6.0]);
    let options = Options {
        mutation_weights: MutationWeights {
            mutate_constant: 0.0,
            mutate_operator: 0.0,
            mutate_feature: 0.0,
            swap_operands: 0.0,
            rotate_tree: 0.0,
            add_node: 0.0,
            insert_node: 0.0,
            delete_node: 0.0,
            simplify: 0.0,
            randomize: 0.0,
            do_nothing: 0.0,
            optimize: 0.0,
            form_connection: 0.0,
            break_connection: 0.0,
            backsolve: 1.0,
        },
        annealing: false,
        use_frequency: false,
        maxsize: 10,
        ..options(&["sin"], &["+", "*"])
    };
    let tagged = TaggedDataset::new(&dataset, None);
    let mut evaluator = Evaluator::<f64, 2>::new(dataset.n_rows);
    let mut parent = PopMember::from_expr(expr(&options, "x1 + 1.0"), 1, &options);
    parent.evaluate(&tagged, &options, &mut evaluator);
    let population = [parent.clone()];
    let mut stats = RunningSearchStatistics::new(options.maxsize, 1000);
    stats.normalize();
    let mut rng = Rng::with_seed(0);

    let mut fired = 0;
    for _ in 0..10 {
        let (child, accepted, evals) = mutate::next_generation(
            &parent,
            NextGenerationCtx {
                rng: &mut rng,
                dataset: tagged,
                temperature: 1.0,
                curmaxsize: options.maxsize,
                stats: &stats,
                options: &options,
                evaluator: &mut evaluator,
                population: &population,
                _ops: core::marker::PhantomData,
            },
        );
        assert!(evals >= 1.0, "an inversion is counted");
        assert!(check_constraints(&child.expr.trees[0], &options, options.maxsize));
        assert!(child.loss.is_finite());
        if accepted {
            fired += 1;
            assert!(child.cost < parent.cost);
            assert!(child.loss < 1e-20, "loss {:e}", child.loss);
            assert_ne!(child.expr.trees[0].nodes, parent.expr.trees[0].nodes);
        }
    }
    assert!(fired > 0);
}
