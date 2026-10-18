#[test]
fn lookup_with_dash_is_ambiguous_without_arity() {
    use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
    use dynamic_expressions::{LookupError, OperatorSet};

    let err = BuiltinOpsF64::new().lookup("-").unwrap_err();
    match err {
        LookupError::Ambiguous { token, candidates } => {
            assert_eq!(token, "-");
            assert!(candidates.iter().any(|c| c.eq_ignore_ascii_case("neg")));
            assert!(candidates.iter().any(|c| c.eq_ignore_ascii_case("sub")));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn lookup_with_dash_and_arity_selects_sub() {
    use dynamic_expressions::OperatorSet;
    use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;

    let op = BuiltinOpsF64::new().lookup_with_arity("-", 2).unwrap();
    assert_eq!(op.arity, 2);
    assert!(BuiltinOpsF64::new().name(op).eq_ignore_ascii_case("sub"));
}

#[test]
fn lookup_with_dash_and_arity_selects_neg() {
    use dynamic_expressions::OperatorSet;
    use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;

    let op = BuiltinOpsF64::new().lookup_with_arity("-", 1).unwrap();
    assert_eq!(op.arity, 1);
    assert!(BuiltinOpsF64::new().name(op).eq_ignore_ascii_case("neg"));
}

#[test]
fn name_returns_unknown_op_for_invalid_opid() {
    use dynamic_expressions::OperatorSet;
    use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;

    assert_eq!(
        BuiltinOpsF64::new().name(dynamic_expressions::OpId { arity: 9, id: 0 }),
        "unknown_op"
    );
    assert_eq!(
        BuiltinOpsF64::new().name(dynamic_expressions::OpId { arity: 1, id: 999 }),
        "unknown_op"
    );
}

#[test]
fn aliases_and_new_builtins_resolve_by_token() {
    use dynamic_expressions::operator_enum::builtin;
    use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
    use dynamic_expressions::{HasOp, OperatorSet};

    let ops = BuiltinOpsF64::new();
    assert_eq!(
        ops.lookup("plus").unwrap(),
        <BuiltinOpsF64 as HasOp<builtin::Add>>::op_id()
    );
    assert_eq!(
        ops.lookup("mult").unwrap(),
        <BuiltinOpsF64 as HasOp<builtin::Mul>>::op_id()
    );
    assert_eq!(
        ops.lookup("^").unwrap(),
        <BuiltinOpsF64 as HasOp<builtin::Pow>>::op_id()
    );
    assert_eq!(ops.lookup(">").unwrap(), ops.lookup("greater").unwrap());
    assert_eq!(ops.lookup("<=").unwrap(), ops.lookup("less_equal").unwrap());
    assert_eq!(ops.lookup("&&").unwrap(), ops.lookup("logical_and").unwrap());
    for name in ["relu", "square", "cube", "greater_equal", "less", "logical_or", "cond"] {
        assert!(ops.lookup(name).is_ok(), "{name}");
    }
    for name in ["equal", "not_equal", "logical_not", "ifelse"] {
        assert!(ops.lookup(name).is_err(), "{name} is not in the presets");
    }
}

#[test]
fn preset_ids_of_existing_operators_are_stable() {
    use dynamic_expressions::operator_enum::builtin;
    use dynamic_expressions::operator_enum::presets::{BuiltinOpsF32, BuiltinOpsF64};
    use dynamic_expressions::{HasOp, OperatorSet};

    assert_eq!(<BuiltinOpsF64 as HasOp<builtin::Abs>>::ID, 0);
    assert_eq!(<BuiltinOpsF64 as HasOp<builtin::Fma>>::ID, 39);
    assert_eq!(<BuiltinOpsF64 as HasOp<builtin::Relu>>::ID, 40);
    assert_eq!(<BuiltinOpsF64 as HasOp<builtin::Cond>>::ID, 49);
    assert_eq!(<BuiltinOpsF32 as HasOp<builtin::Fma>>::ID, 39);
    let ops = BuiltinOpsF64::new();
    let n: usize = (1..=ops.max_arity()).map(|a| ops.ops_with_arity(a).count()).sum();
    assert_eq!(n, 50);
}
