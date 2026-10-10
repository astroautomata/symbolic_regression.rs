//! Ports of SymbolicRegression.jl testitems about template expressions inside the search:
//! constants and parameters, constraints, losses, mutation, crossover, and new candidates (see
//! tools/parity_tests.toml).

use dynamic_expressions::dsl::TemplateStructure;
use dynamic_expressions::expression::PostfixExpr;
use dynamic_expressions::node::PNode;
use dynamic_expressions::operator_enum::presets::BuiltinOpsF32;
use dynamic_expressions::operators::exp;
use dynamic_expressions::{Metadata, OperatorSet};
use fastrand::Rng;
use ndarray::{Array1, Array2, Axis};

use super::common::{D, T, TestOps};
use crate::check_constraints::check_expression;
use crate::dataset::{Dataset, TaggedDataset};
use crate::expression::Expression;
use crate::mutate::{
    CrossoverCtx, condition_mutation_weights, crossover_expressions, crossover_generation, mutate_constant,
};
use crate::options::{MutationWeights, Options};
use crate::pop_member::{Evaluator, PopMember};
use crate::random::standard_normal;
use crate::search_utils::new_expression;
use crate::{NestedConstraints, OperatorLibrary};

type Expr = Expression<T, TestOps, D>;

/// The subexpression argument `#i` (1-based).
fn arg(i: u16) -> PostfixExpr<T, TestOps, D> {
    PostfixExpr::new(vec![PNode::Var { feature: i - 1 }], vec![], Metadata::default())
}

fn constant(v: T) -> PostfixExpr<T, TestOps, D> {
    PostfixExpr::new(vec![PNode::Const { idx: 0 }], vec![v], Metadata::default())
}

fn options(
    combine: &str,
    expressions: &[&str],
    variables: &[&str],
    parameters: &[(&str, usize)],
) -> Options<T, TestOps, D> {
    let operators = OperatorLibrary::sr_default::<TestOps, D>();
    let template = TemplateStructure::new(combine, expressions, None, variables, parameters, &operators.set).unwrap();
    Options {
        operators,
        template: Some(template),
        ..Default::default()
    }
}

/// The parameter vectors of `expr`, split by the template's lengths.
fn vectors<'a>(expr: &'a Expr, options: &Options<T, TestOps, D>) -> Vec<&'a [T]> {
    let mut rest = expr.params.as_slice();
    let mut out = Vec::new();
    for (_, n) in options.template.as_ref().unwrap().parameters() {
        let (v, tail) = rest.split_at(*n);
        out.push(v);
        rest = tail;
    }
    out
}

fn eval(expr: &Expr, options: &Options<T, TestOps, D>, rows: &[&[T]]) -> Vec<T> {
    let x = Array2::from_shape_fn((rows.len(), rows[0].len()), |(f, i)| rows[f][i]);
    let (out, ok) = options
        .template
        .as_ref()
        .unwrap()
        .eval(&expr.trees, &expr.params, &options.operators.set, x.view())
        .unwrap();
    assert!(ok);
    out
}

/// test/integration/ext/loopvectorization/test_template_expression.jl | "Test parameter handling in scalar constants"
#[test]
fn parameters_are_scalar_constants_after_tree_constants() {
    let mut expr = Expr {
        trees: vec![arg(1)],
        params: vec![1.0, 2.0],
    };
    let mut constants = expr.constants();
    assert_eq!(constants, [1.0, 2.0]);
    let n = constants.len();
    constants[n - 2..].copy_from_slice(&[3.0, 4.0]);
    expr.set_constants(&constants);
    assert_eq!(expr.params, [3.0, 4.0]);
}

/// test/integration/ext/loopvectorization/test_template_expression.jl | "Test constraints checking in
/// TemplateExpression"
#[test]
fn nested_constraints_apply_to_each_subexpression() {
    let mut options = options("f(x1, x2)", &["f"], &["x1", "x2"], &[]);
    options.operators = TestOps::new().from_names(["+", "sub", "*", "/", "exp"]).unwrap();
    let exp_id = options.operators.set.lookup_with_arity("exp", 1).unwrap();
    let mut nested = NestedConstraints::default();
    nested.add_nested_constraint(exp_id, exp_id, 1);
    options.nested_constraints = nested;

    let valid = Expr::from(exp(arg(1)));
    let invalid = Expr::from(exp(exp(exp(arg(1)))));
    assert!(check_expression(&valid, &options, 20));
    assert!(!check_expression(&invalid, &options, 20));
}

/// `y = 2 x1 + x2^2` on three rows, as SymbolicRegression.jl's loss tests use.
fn loss_dataset() -> Dataset<f32> {
    let x = Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0]).unwrap();
    let y = Array1::from_iter((0..3).map(|i| 2.0 * x[(0, i)] + x[(1, i)] * x[(1, i)]));
    Dataset::new(x, y)
}

fn loss_of<const N: usize>(
    member: &mut PopMember<f32, BuiltinOpsF32, 2>,
    data: TaggedDataset<'_, f32>,
    options: &Options<f32, BuiltinOpsF32, 2>,
) -> f32 {
    assert!(member.evaluate(&data, options, &mut Evaluator::new(N)));
    member.loss
}

/// test/integration/ext/loopvectorization/test_template_expression.jl | "loss_function_expression with expressions and
/// templates" test/integration/ext/loopvectorization/test_template_expression.jl | "New batching syntax"
#[test]
fn template_loss_on_the_dataset_and_a_batch() {
    let options = Options::<f32, BuiltinOpsF32, 2> {
        operators: BuiltinOpsF32::new().from_names(["+", "*"]).unwrap(),
        template: Some(
            TemplateStructure::new(
                "f(x1) + p[1] + p[2]",
                &["f"],
                None,
                &["x1", "x2", "x3"],
                &[("p", 2)],
                &BuiltinOpsF32::new(),
            )
            .unwrap(),
        ),
        ..Default::default()
    };
    let x1 = PostfixExpr::new(vec![PNode::Var { feature: 0 }], vec![], Metadata::default());
    let expr = Expression {
        trees: vec![x1],
        params: vec![1.0, 2.0],
    };
    let full = loss_dataset();
    let mut member = PopMember::from_expr(expr, full.n_features, &options);

    let want = [4.0_f32 - 6.0, 5.0 - 13.0, 6.0 - 22.0]
        .iter()
        .map(|r| r * r)
        .sum::<f32>()
        / 3.0;
    assert_eq!(
        loss_of::<3>(&mut member, TaggedDataset::new(&full, None), &options),
        want
    );

    let batch = Dataset::new(full.x.select(Axis(1), &[0, 1]), full.y.select(Axis(0), &[0, 1]));
    let want = [4.0_f32 - 6.0, 5.0 - 13.0].iter().map(|r| r * r).sum::<f32>() / 2.0;
    let got = loss_of::<2>(
        &mut member,
        TaggedDataset::for_batch(&batch, None, full.n_rows),
        &options,
    );
    assert_eq!(got, want);
}

/// test/integration/ext/loopvectorization/test_template_expression.jl | "Test Float32/Float64 type conversion in
/// TemplateExpression"
#[test]
fn float32_template_loss() {
    let options = Options::<f32, BuiltinOpsF32, 2> {
        operators: BuiltinOpsF32::new().from_names(["+", "*", "/", "sub"]).unwrap(),
        template: Some(
            TemplateStructure::new(
                "0.5 * f(x1, x2)",
                &["f"],
                None,
                &["x1", "x2"],
                &[],
                &BuiltinOpsF32::new(),
            )
            .unwrap(),
        ),
        ..Default::default()
    };
    let var =
        |f| PostfixExpr::<f32, BuiltinOpsF32, 2>::new(vec![PNode::Var { feature: f }], vec![], Metadata::default());
    let dataset = Dataset::new(
        Array2::from_shape_vec((2, 2), vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap(),
        Array1::from_vec(vec![2.0_f32, 3.0]),
    );
    let mut member = PopMember::from_expr(var(0) + var(1), 2, &options);
    let loss: f32 = loss_of::<2>(&mut member, TaggedDataset::new(&dataset, None), &options);
    assert_eq!(loss, 0.0);
}

/// Mutates a copy of `expr` `n` times with one RNG, reporting which parameter vectors ever changed.
fn vectors_changed(expr: &Expr, options: &Options<T, TestOps, D>, n: usize) -> Vec<bool> {
    let mut rng = Rng::with_seed(0);
    let original = vectors(expr, options);
    let mut changed = vec![false; original.len()];
    for _ in 0..n {
        let mut mutated = expr.clone();
        mutate_constant(&mut rng, &mut mutated, 1.0, options);
        for (k, v) in vectors(&mutated, options).iter().enumerate() {
            changed[k] |= *v != original[k];
        }
    }
    changed
}

/// test/integration/ext/mlj/templates/test_parametric_template_expressions.jl | "parameters get mutated"
#[test]
fn constant_mutation_changes_parameters() {
    let options = options("f(x) + p[1] + p[2]", &["f"], &["x"], &[("p", 2)]);
    let expr = Expr {
        trees: vec![arg(1)],
        params: vec![1.0, 2.0],
    };
    assert_eq!(vectors_changed(&expr, &options, 50), [true]);
}

/// test/integration/ext/mlj/templates/test_parametric_template_expressions.jl | "Preallocated copying with parameters"
#[test]
fn copying_carries_current_parameters() {
    let options = options("f(x) + p[1] + p[2]", &["f"], &["x"], &[("p", 2)]);
    let mut expr = Expr {
        trees: vec![arg(1)],
        params: vec![10.0, 20.0],
    };
    let mut copy = expr.clone();
    expr.params.copy_from_slice(&[100.0, 200.0]);
    copy.clone_from(&expr);
    assert_eq!(copy.params, [100.0, 200.0]);
    assert_eq!(eval(&expr, &options, &[&[1.0]]), [301.0]);
    assert_eq!(eval(&copy, &options, &[&[2.0]]), [302.0]);
}

/// test/integration/ext/mlj/templates/test_parametric_template_expressions.jl | "multi-parameter expressions"
#[test]
fn multi_parameter_constants_mutation_and_copying() {
    let options = options(
        "p1[1] * f(x)^2 + p1[2] * f(x) + p1[3] + p2[1] * g(y)^2 + p2[2] * g(y)",
        &["f", "g"],
        &["x", "y"],
        &[("p1", 3), ("p2", 2)],
    );
    let mut expr = Expr {
        trees: vec![arg(1), arg(1) + 0.0],
        params: vec![1.0, 2.0, 3.0, -1.0, -2.0],
    };
    assert_eq!(expr.constants(), [0.0, 1.0, 2.0, 3.0, -1.0, -2.0]);
    assert_eq!(eval(&expr, &options, &[&[2.0, 3.0], &[4.0, 5.0]]), [-13.0, -17.0]);
    assert_eq!(vectors_changed(&expr, &options, 50), [true, true]);

    let mut copy = expr.clone();
    expr.params.copy_from_slice(&[10.0, 20.0, 30.0, -10.0, -20.0]);
    copy.clone_from(&expr);
    assert_eq!(vectors(&copy, &options), [&[10.0, 20.0, 30.0][..], &[-10.0, -20.0][..]]);
}

/// test/unit/expressions/test_template_inner_expression_type.jl | "TemplateExpressionSpec initializes template
/// parameters"
#[test]
fn new_candidates_copy_the_tree_and_draw_normal_parameters() {
    let options = options(
        "f(x) + weights[1] + bias[1]",
        &["f"],
        &["x"],
        &[("weights", 2), ("bias", 1)],
    );
    let mut rng = Rng::with_seed(7);
    let mut expected_rng = rng.clone();
    let created = new_expression(constant(0.0), &options, &mut rng);
    assert_eq!(created.trees.len(), 1);
    assert_eq!(created.trees[0].consts, [0.0]);
    let expected: Vec<T> = (0..3).map(|_| standard_normal(&mut expected_rng)).collect();
    assert_eq!(created.params, expected);
    assert_eq!(
        vectors(&created, &options).iter().map(|v| v.len()).collect::<Vec<_>>(),
        [2, 1]
    );

    let options = Options::<T, TestOps, D> {
        template: None,
        ..options
    };
    let plain = new_expression(constant(0.0), &options, &mut rng);
    assert_eq!(plain.trees.len(), 1);
    assert!(plain.params.is_empty());
}

/// test/unit/templates/test_template_expression_mutation.jl | "template expression parameter mutation"
#[test]
fn parameter_mutation_changes_one_vector_and_conditions_weights() {
    let options = options(
        "f(x1, x2) * p1[1] + g(x3) * p2[1]",
        &["f", "g"],
        &["x1", "x2", "x3"],
        &[("p1", 2), ("p2", 3)],
    );
    let expr = Expr {
        trees: vec![arg(1), arg(1)],
        params: vec![1.0, 2.0, 3.0, 4.0, 5.0],
    };
    assert_eq!(vectors_changed(&expr, &options, 50), [true, true]);

    let original = vectors(&expr, &options);
    let mut rng = Rng::with_seed(1);
    for _ in 0..20 {
        let mut mutated = expr.clone();
        mutate_constant(&mut rng, &mut mutated, 1.0, &options);
        let now = vectors(&mutated, &options);
        assert!((now[0] != original[0]) ^ (now[1] != original[1]), "{now:?}");
    }

    let configured = Options {
        perturbation_factor: 10.0,
        probability_negate_constant: 1.0,
        ..options.clone()
    };
    let (mut a, mut b) = (expr.clone(), expr.clone());
    mutate_constant(&mut Rng::with_seed(1), &mut a, 1.0, &configured);
    mutate_constant(&mut Rng::with_seed(1), &mut b, 1.0, &options);
    assert_ne!(a.params, b.params);

    let options = Options {
        should_simplify: false,
        ..options
    };
    let member = PopMember::from_expr(expr, 3, &options);
    let mut weights = MutationWeights::default();
    condition_mutation_weights(&mut weights, &member, &options, 0, 1);
    assert_eq!(weights.mutate_feature, 0.0);
    assert_eq!(weights.add_node, 0.0);
    assert_eq!(weights.insert_node, 0.0);
    assert_eq!(weights.simplify, 0.0);
}

/// test/unit/templates/test_template_expression_mutation.jl | "template crossover isolates recipient candidate state"
#[test]
fn crossover_children_own_their_state() {
    let options = options("f(x) + g(x) * p[1]", &["f", "g"], &["x"], &[("p", 2)]);
    let parent = |f, g, p: [T; 2]| Expr {
        trees: vec![constant(f), constant(g)],
        params: p.to_vec(),
    };
    let (parent1, parent2) = (parent(1.0, 2.0, [3.0, 4.0]), parent(5.0, 6.0, [7.0, 8.0]));
    let (values1, values2) = (parent1.constants(), parent2.constants());

    let (mut child1, mut child2) = crossover_expressions(&mut Rng::with_seed(3), &parent1, &parent2);
    assert_eq!(child1.params, parent1.params);
    assert_eq!(child2.params, parent2.params);
    for (child, offset) in [(&mut child1, 10.0), (&mut child2, 20.0)] {
        let shifted: Vec<T> = child.constants().iter().map(|v| v + offset).collect();
        child.set_constants(&shifted);
        assert_eq!(child.constants(), shifted);
        assert_eq!(parent1.constants(), values1);
        assert_eq!(parent2.constants(), values2);
    }

    let dataset = Dataset::new(Array2::zeros((1, 4)), Array1::zeros(4));
    let member = |expr| {
        let mut m = PopMember::from_expr(expr, 1, &options);
        m.complexity = options.complexity(&m.expr);
        m
    };
    let (_, _, accepted, _) = crossover_generation(
        &member(parent1),
        &member(parent2),
        CrossoverCtx {
            rng: &mut Rng::with_seed(0),
            dataset: TaggedDataset::new(&dataset, None),
            curmaxsize: 100,
            options: &options,
            evaluator: &mut Evaluator::new(4),
            _ops: core::marker::PhantomData,
        },
    );
    assert!(accepted);
}
