use dynamic_expressions::dsl::ir::{Graph, Node, NodeId, Ty};
use dynamic_expressions::dsl::kernels::Kernel;
use dynamic_expressions::dsl::lower::{Lowered, lower};
use dynamic_expressions::dsl::{CustomOps, Span, parse_definition};

fn span_of(src: &str, needle: &str) -> Span {
    let start = src.find(needle).unwrap_or_else(|| panic!("{needle:?} not in {src:?}"));
    Span::new(start, start + needle.len())
}

fn ir_string(graph: &Graph, id: NodeId) -> String {
    match graph.node(id) {
        Node::Const(bits) => format!("{}", f64::from_bits(*bits)),
        Node::Arg(i) => format!("#{i}"),
        Node::Op { kernel, args } => {
            let args: Vec<String> = args.iter().map(|a| ir_string(graph, *a)).collect();
            format!("({} {})", kernel.name(), args.join(" "))
        }
        Node::Partial { kernel, idx, args } => {
            let args: Vec<String> = args.iter().map(|a| ir_string(graph, *a)).collect();
            format!("(d{idx}{} {})", kernel.name(), args.join(" "))
        }
    }
}

fn lower_with(src: &str, customs: &[Lowered]) -> Lowered {
    let def = parse_definition(src).unwrap_or_else(|e| panic!("{src:?}: {e}"));
    lower(&def, customs).unwrap_or_else(|e| panic!("{src:?}: {e}"))
}

fn ir(src: &str) -> String {
    let l = lower_with(src, &[]);
    ir_string(&l.graph, l.root)
}

#[track_caller]
fn assert_lower_error(src: &str, customs: &[Lowered], message: &str, span: Span) {
    let def = parse_definition(src).unwrap_or_else(|e| panic!("{src:?}: {e}"));
    let err = lower(&def, customs).expect_err(src);
    assert_eq!(err.message, message, "{src:?}");
    assert_eq!(err.span, span, "{src:?}: {err}");
}

#[test]
fn lower_operators_map_to_kernels() {
    assert_eq!(
        ir("f(x, y) = x + y - x * y / x ^ y"),
        "(sub (add #0 #1) (div (mul #0 #1) (pow #0 #1)))"
    );
    assert_eq!(
        ir("f(x, y) = (x < y) + (x <= y) + (x > y) + (x >= y) + (x == y) + (x != y)"),
        "(add (add (add (add (add (less #0 #1) (less_equal #0 #1)) (greater #0 #1)) (greater_equal #0 #1)) (equal #0 #1)) (not_equal #0 #1))"
    );
    assert_eq!(
        ir("f(x, y) = x > 0 && !(y > 0) || false"),
        "(logical_or (logical_and (greater #0 0) (logical_not (greater #1 0))) 0)"
    );
    assert_eq!(ir("f(t, a, b) = t > 0 ? a : b"), "(ifelse (greater #0 0) #1 #2)");
    assert_eq!(ir("f(t, a, b) = ifelse(t > 0, a, b)"), "(ifelse (greater #0 0) #1 #2)");
    assert_eq!(ir("f(x) = -x"), "(neg #0)");
    assert_eq!(ir("f(x) = +x"), "#0");
    assert_eq!(ir("f() = -2"), "-2");
    assert_eq!(ir("f() = -pi"), format!("{}", -core::f64::consts::PI));
    assert_eq!(
        ir("f(x) = sin(cos(x)) + abs2(x) + relu(x) + square(x) + cond(x, x)"),
        "(add (add (add (add (sin (cos #0)) (abs2 #0)) (relu #0)) (square #0)) (cond #0 #0))"
    );
    assert_eq!(ir("f(x, y) = greater(x, y)"), "(greater #0 #1)");
}

#[test]
fn lower_names_and_shadowing() {
    assert_eq!(ir("f() = pi"), format!("{}", core::f64::consts::PI));
    assert_eq!(ir("f() = π"), format!("{}", core::f64::consts::PI));
    assert_eq!(ir("f() = ℯ"), format!("{}", core::f64::consts::E));
    assert_eq!(ir("f() = 1_000"), "1000");
    assert_eq!(ir("f() = Inf"), "inf");
    assert_eq!(ir("f() = NaN"), "NaN");
    assert_eq!(ir("f(pi) = pi * 2"), "(mul #0 2)");
    assert_eq!(
        ir("function f(x)\n    Inf = x + 1\n    Inf * Inf\nend"),
        "(mul (add #0 1) (add #0 1))"
    );
}

#[test]
fn lower_locals_and_return() {
    assert_eq!(
        ir("function f(x)\n    a = x + 1\n    local b = a * a\n    b - a\nend"),
        "(sub (mul (add #0 1) (add #0 1)) (add #0 1))"
    );
    assert_eq!(ir("function f(x)\n    return x\n    x^2\nend"), "#0");
    assert_eq!(ir("function f(x)\n    y = sin(x)\nend"), "(sin #0)");
    assert_eq!(
        ir("function f(x)\n    x = x + 1; x = x * 2\n    x\nend"),
        "(mul (add #0 1) 2)"
    );
}

#[test]
fn lower_common_subexpressions_share_nodes() {
    let l = lower_with("f(x) = sin(x) * sin(x)", &[]);
    assert_eq!(l.graph.len(), 3, "{:?}", l.graph.nodes());
    let l = lower_with("f(x, y) = (x - y)^2 + exp(x - y)", &[]);
    let subs = l
        .graph
        .nodes()
        .iter()
        .filter(|n| {
            matches!(
                n,
                Node::Op {
                    kernel: Kernel::Sub,
                    ..
                }
            )
        })
        .count();
    assert_eq!(subs, 1);
}

#[test]
fn lower_inlines_earlier_custom_operators() {
    let sq = lower_with("sq(x) = x^2", &[]);
    let g = lower_with("g(x, y) = sq(x + y) + sq(x)", std::slice::from_ref(&sq));
    assert_eq!(ir_string(&g.graph, g.root), "(add (pow (add #0 #1) 2) (pow #0 2))");
    let h = lower_with("h(a) = g(a, sq(a))", &[sq.clone(), g]);
    assert_eq!(
        ir_string(&h.graph, h.root),
        "(add (pow (add #0 (pow #0 2)) 2) (pow #0 2))"
    );
    let inv = lower_with("inv(x) = 2 / x", &[]);
    let k = lower_with("k(x) = inv(x)", &[inv]);
    assert_eq!(ir_string(&k.graph, k.root), "(div 2 #0)");
    assert_eq!(ir("k(x) = inv(x)"), "(inv #0)");
    assert_lower_error(
        "k(x) = sq(x, x)",
        &[sq],
        "`sq` takes 1 argument, got 2",
        Span::new(7, 15),
    );
}

#[test]
fn lower_bool_and_num_types() {
    assert_eq!(lower_with("f(x) = x > 0", &[]).ty, Ty::Bool);
    assert_eq!(lower_with("f() = true", &[]).ty, Ty::Bool);
    assert_eq!(lower_with("f(x) = !(x > 0) || false", &[]).ty, Ty::Bool);
    assert_eq!(lower_with("f(x) = x > 0 ? true : false", &[]).ty, Ty::Bool);
    assert_eq!(lower_with("f(x) = x > 0 ? 1 : true", &[]).ty, Ty::Num);
    assert_eq!(lower_with("f(x) = (x > 0) + 1", &[]).ty, Ty::Num);
    assert_eq!(lower_with("f(x) = -(x > 0)", &[]).ty, Ty::Num);
    assert_eq!(lower_with("f(x) = x", &[]).ty, Ty::Num);
    // SymbolicRegression.jl's named comparison and logic operators return floats.
    for src in [
        "f(x) = greater(x, 0)",
        "f(x, y) = less_equal(x, y)",
        "f(x, y) = logical_and(x, y)",
        "f(x) = logical_not(x)",
    ] {
        assert_eq!(lower_with(src, &[]).ty, Ty::Num, "{src}");
    }
    let mut custom = CustomOps::<f64>::default();
    custom.define("step(x) = greater(x, 0)").unwrap();

    let msg = "expected a Bool condition, found a number";
    let src = "f(x) = x && true";
    assert_lower_error(src, &[], msg, Span::new(7, 8));
    let src = "f(x) = true || x + 1";
    assert_lower_error(src, &[], msg, span_of(src, "x + 1"));
    let src = "f(x) = !x";
    assert_lower_error(src, &[], msg, Span::new(8, 9));
    let src = "f(x) = x ? 1 : 2";
    assert_lower_error(src, &[], msg, Span::new(7, 8));
    let src = "f(x) = ifelse(x, 1, 2)";
    assert_lower_error(src, &[], msg, Span::new(14, 15));
}

#[test]
fn lower_atan_and_variadic_min_max() {
    assert_eq!(ir("f(y, x) = atan(y, x)"), "(atan2 #0 #1)");
    assert_eq!(ir("f(x) = atan(x)"), "(atan #0)");
    assert_eq!(ir("f(a, b) = min(a, b)"), "(min #0 #1)");
    assert_eq!(ir("f(a, b, c, d) = min(a, b, c, d)"), "(min (min (min #0 #1) #2) #3)");
    assert_eq!(ir("f(a, b, c) = max(a, b, c)"), "(max (max #0 #1) #2)");
    let src = "f(x) = atan(x, x, x)";
    assert_lower_error(
        src,
        &[],
        "`atan` takes 1 or 2 arguments, got 3",
        span_of(src, "atan(x, x, x)"),
    );
    let src = "f(x) = max(x)";
    assert_lower_error(
        src,
        &[],
        "`max` takes 2 or more arguments, got 1",
        span_of(src, "max(x)"),
    );
    let src = "f(x) = ifelse(x > 0, x)";
    assert_lower_error(
        src,
        &[],
        "`ifelse` takes 3 arguments, got 2",
        span_of(src, "ifelse(x > 0, x)"),
    );
}

#[test]
fn lower_errors() {
    let src = "f(x) = foo(x) + 1";
    assert_lower_error(src, &[], "unknown function `foo`", span_of(src, "foo"));
    let src = "f(x) = x + z";
    assert_lower_error(src, &[], "unknown name `z`", span_of(src, "z"));
    let src = "f(x, y) = cos(x, y)";
    assert_lower_error(src, &[], "`cos` takes 1 argument, got 2", span_of(src, "cos(x, y)"));
    let src = "f(x) = fma(x)";
    assert_lower_error(src, &[], "`fma` takes 3 arguments, got 1", span_of(src, "fma(x)"));
    let src = "f(x) = 1 + f(x - 1)";
    assert_lower_error(src, &[], "operator `f` cannot call itself", span_of(src, "f(x - 1)"));
    let src = "function f(x)\nend";
    assert_lower_error(src, &[], "function body has no value", Span::new(0, src.len()));
}
