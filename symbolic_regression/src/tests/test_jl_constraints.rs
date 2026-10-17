//! Ports of SymbolicRegression.jl testitems about constraints (see tools/parity_tests.toml).

use dynamic_expressions::dsl::CustomOps;
use dynamic_expressions::expression_algebra::__apply_postfix;
use dynamic_expressions::node_utils::count_depth;
use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
use dynamic_expressions::{Metadata, OperatorSet, PNode, PostfixExpr};

use crate::check_constraints::check_constraints;
use crate::{NestedConstraints, Options};

type Ops = BuiltinOpsF64<CustomOps<f64>>;
type Expr = PostfixExpr<f64, Ops, 3>;

fn operator_set(defs: &[&str]) -> Ops {
    let mut custom = CustomOps::default();
    for def in defs {
        custom.define(def).unwrap();
    }
    BuiltinOpsF64::new().with_custom(custom)
}

fn options(operator_set: Ops, names: &[&str]) -> Options<f64, Ops, 3> {
    Options {
        operators: operator_set.from_names(names).unwrap(),
        ..Default::default()
    }
}

fn x(feature: u16) -> Expr {
    PostfixExpr::new(vec![PNode::Var { feature }], vec![], Metadata::default())
}

fn c(value: f64) -> Expr {
    PostfixExpr::new(vec![PNode::Const { idx: 0 }], vec![value], Metadata::default())
}

fn call<const A: usize>(options: &Options<f64, Ops, 3>, name: &str, args: [Expr; A]) -> Expr {
    let op = options.operators.set.lookup_with_arity(name, A as u8).unwrap();
    __apply_postfix::<f64, Ops, 3, A>(op.id, args)
}

fn ok(expr: &Expr, options: &Options<f64, Ops, 3>) -> bool {
    check_constraints(expr, options, options.maxsize)
}

/// test/unit/constraints/test_constraints.jl | "Test constraints interface"
#[test]
fn constraints_interface() {
    let mut opts = options(
        operator_set(&["_inv(x) = 1 / x"]),
        &["+", "*", "^", "/", "greater", "_inv"],
    );
    let inv = opts.operators.set.lookup("_inv").unwrap();
    opts.op_constraints.set_op_arg_constraint(inv, 0, 4);
    let tree = call(
        &opts,
        "greater",
        [
            call(&opts, "^", [c(3.0) * call(&opts, "_inv", [x(0)]), c(2.0)]),
            c(-1.2),
        ],
    );
    let violating = call(&opts, "_inv", [tree.clone()]);
    assert!(ok(&tree, &opts));
    assert!(!ok(&violating, &opts));

    let complex_expr = x(0) * x(0) * x(0) * x(0) * x(0);
    let nested_inv = call(&opts, "greater", [x(0), call(&opts, "_inv", [complex_expr])]);
    assert!(!ok(&nested_inv, &opts));

    let builtin = || operator_set(&[]);
    let mut opts = options(builtin(), &["+", "*"]);
    opts.maxsize = 5;
    let tree = x(0) + x(1) * x(2);
    let violating = c(5.1) * tree.clone();
    assert!(ok(&tree, &opts));
    assert!(!ok(&violating, &opts));

    let mul = opts.operators.set.lookup("*").unwrap();
    opts.operator_complexity_overrides.insert(mul, 3.0);
    assert!(!ok(&tree, &opts));
    opts.operator_complexity_overrides.insert(mul, 0.0);
    assert!(ok(&violating, &opts));

    let mut opts = options(builtin(), &["+", "*", "cos"]);
    opts.maxsize = 100;
    opts.maxdepth = Some(3);
    let tree = (x(0) + x(1)) + (x(2) + x(0));
    assert_eq!(count_depth(&tree.nodes), 3);
    assert!(ok(&tree, &opts));
    let tree = (x(0) + x(1)) + (x(2) + x(0)) * x(0);
    assert_eq!(count_depth(&tree.nodes), 4);
    assert!(!ok(&tree, &opts));
    let tree = call(&opts, "cos", [call(&opts, "cos", [x(0)])]);
    assert_eq!(count_depth(&tree.nodes), 3);
    assert!(ok(&tree, &opts));
    let tree = call(&opts, "cos", [tree]);
    assert_eq!(count_depth(&tree.nodes), 4);
    assert!(!ok(&tree, &opts));

    let mut opts = options(operator_set(&["_inv2(x) = 1 / x"]), &["+", "*", "_inv2"]);
    opts.maxsize = 10;
    let inv2 = opts.operators.set.lookup("_inv2").unwrap();
    opts.op_constraints.set_op_arg_constraint(inv2, 0, 5);
    let deep = x(0) * x(0) * x(0) * x(0) * x(0) * x(0);
    assert!(!ok(&call(&opts, "_inv2", [deep]), &opts));
}

/// test/unit/constraints/test_nested_constraints.jl | "Test nested constraint checking"
#[test]
fn nested_constraint_checking() {
    let base = || options(operator_set(&[]), &["+", "*", "/", "sub", "cos", "exp"]);
    let with = |rules: &[(&str, u8, &str, u8, u8)]| {
        let mut opts = base();
        let mut nested = NestedConstraints::default();
        for &(root, root_arity, inner, inner_arity, max) in rules {
            let root = opts.operators.set.lookup_with_arity(root, root_arity).unwrap();
            let inner = opts.operators.set.lookup_with_arity(inner, inner_arity).unwrap();
            nested.add_nested_constraint(root, inner, max);
        }
        opts.nested_constraints = nested;
        opts
    };
    let o = base();
    let cos = |e| call(&o, "cos", [e]);
    let exp = |e| call(&o, "exp", [e]);

    let tree = cos(exp(exp(exp(exp(x(0))))));
    assert!(!ok(&tree, &with(&[("cos", 1, "exp", 1, 3)])));
    assert!(ok(&tree, &with(&[("cos", 1, "exp", 1, 4)])));

    let tree = cos(exp(x(0)) + exp(exp(exp(exp(x(0))))));
    assert!(!ok(&tree, &with(&[("cos", 1, "exp", 1, 3)])));
    assert!(ok(&tree, &with(&[("cos", 1, "exp", 1, 4)])));

    let tree = cos(exp(x(0)) + exp(exp(x(0) + exp(exp(exp(x(0)))))));
    assert!(!ok(&tree, &with(&[("cos", 1, "+", 2, 1)])));
    assert!(ok(&tree, &with(&[("cos", 1, "+", 2, 2)])));

    let tree = cos(cos(x(0))) + cos(x(0)) + exp(cos(x(0)));
    assert!(ok(&tree, &base()));
    assert!(!ok(&tree, &with(&[("cos", 1, "cos", 1, 0)])));
    assert!(ok(&tree, &with(&[("cos", 1, "cos", 1, 1)])));
    assert!(ok(&tree, &with(&[("cos", 1, "exp", 1, 0)])));
    assert!(!ok(&tree, &with(&[("exp", 1, "cos", 1, 0)])));
    assert!(!ok(&tree, &with(&[("+", 2, "+", 2, 0)])));
}
