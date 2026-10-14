//! Halls of fame admit only members that satisfy the search's constraints, as SymbolicRegression.jl's
//! checked `update_hall_of_fame!` does, even when a population still holds an invalid tree.

use dynamic_expressions::{HasOp, OpId};
use ndarray::{Array1, Array2};
use symbolic_regression::prelude::*;
use symbolic_regression::{HallOfFame, PopMember, parse_guesses};

type Ops = BuiltinOpsF64;

fn mul() -> OpId {
    <Ops as HasOp<Mul>>::op_id()
}

fn cubic() -> Dataset<f64> {
    let n_rows = 64usize;
    let mut rng = fastrand::Rng::with_seed(0);
    let x = Array2::from_shape_fn((2, n_rows), |_| rng.f64() * 4.0 - 2.0);
    let y = Array1::from_iter((0..n_rows).map(|i| x[(0, i)] * x[(0, i)] * x[(0, i)]));
    Dataset::new(x, y)
}

/// Serial, seeded options over `operators` that forbid any `*` inside a `*`.
fn no_nested_mul(operators: &[&str]) -> Options<f64, Ops, 2> {
    let mut options = Options {
        save_to_file: false,
        operators: Ops::new().from_names(operators).unwrap(),
        should_simplify: Some(false),
        seed: Some(0),
        deterministic: true,
        parallelism: Parallelism::Serial,
        progress: false,
        ..Default::default()
    };
    options.nested_constraints.add_nested_constraint(mul(), mul(), 0);
    options
}

fn mul_count(tree: &PostfixExpr<f64, Ops, 2>) -> usize {
    let mul = mul();
    tree.nodes
        .iter()
        .filter(|n| matches!(n, PNode::Op { arity, op } if *arity == mul.arity && *op == mul.id))
        .count()
}

/// The guesses evaluated as members, without any constraint check.
fn members(options: &mut Options<f64, Ops, 2>, guesses: Vec<Guess<f64, Ops, 2>>) -> Vec<PopMember<f64, Ops, 2>> {
    options.guesses = vec![guesses];
    let mut parsed = parse_guesses(std::slice::from_ref(&cubic()), options).unwrap();
    options.guesses.clear();
    parsed.members.pop().unwrap()
}
/// The complexities the hall of fame keeps after considering `members`.
fn admitted(options: &Options<f64, Ops, 2>, members: &[PopMember<f64, Ops, 2>]) -> Vec<usize> {
    let mut hall = HallOfFame::new(options.maxsize);
    for m in members {
        hall.consider(m, options);
    }
    hall.members().map(|m| m.complexity).collect()
}

/// With only `*` and no `*` allowed inside a `*`, every hall entry has at most one `*`,
/// even when initial random populations contain nested multiplication.
#[test]
fn search_halls_exclude_nested_multiplication() {
    let mut options = no_nested_mul(&["*"]);
    options.niterations = 4;
    options.populations = 4;
    options.population_size = 30;
    options.ncycles_per_iteration = 100;
    let result = equation_search::<f64, Ops, 2>(std::slice::from_ref(&cubic()), &options).unwrap();

    let hall = &result.halls_of_fame[0];
    assert!(hall.members().next().is_some(), "the hall of fame is empty");
    for m in hall.members().chain(&hall.pareto_front()).chain(&result.best[0]) {
        for tree in m.expr.trees.iter() {
            assert!(
                mul_count(tree) <= 1,
                "nested `*` in the hall of fame: {}",
                m.expr.string(&options, Default::default())
            );
        }
    }
}

#[test]
fn hall_rejects_nested_constraint_violation() {
    let mut options = no_nested_mul(&["*"]);
    let members = members(&mut options, vec!["x0 * x0 * x0".into(), "x0 * x1".into()]);
    assert_eq!(admitted(&options, &members), [members[1].complexity]);
}

#[test]
fn hall_rejects_trees_deeper_than_maxdepth() {
    let mut options = no_nested_mul(&["*"]);
    options.nested_constraints = Default::default();
    options.maxdepth = Some(2);
    let members = members(&mut options, vec!["x0 * x0 * x0".into(), "x0 * x1".into()]);
    assert_eq!(admitted(&options, &members), [members[1].complexity]);
}

/// A template member is admitted only when every subexpression satisfies the constraints.
#[test]
fn hall_rejects_template_subexpression_violation() {
    let mut options = no_nested_mul(&["+", "*"]);
    options.template =
        Some(TemplateStructure::new("f(a) + g(b)", &["f", "g"], None, &["a", "b"], &[], &Ops::new()).unwrap());
    let template = |f: &str| Guess::Template(vec![("f".to_string(), f.into()), ("g".to_string(), "#1".into())]);
    let members = members(&mut options, vec![template("#1 * #1 * #1"), template("#1 * #1")]);
    assert_eq!(admitted(&options, &members), [members[1].complexity]);
}
