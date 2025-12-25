#[test]
fn lookup_prefers_binary_sub_for_dash() {
    use dynamic_expressions::OperatorSet;
    use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;

    let op = BuiltinOpsF64::lookup("-").unwrap();
    assert_eq!(op.arity, 2);
    assert!(BuiltinOpsF64::name(op).eq_ignore_ascii_case("sub"));
}
