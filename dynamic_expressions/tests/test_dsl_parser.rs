use dynamic_expressions::dsl::{
    CustomOps, DefKind, Expr, ExprKind, Span, Stmt, UnaryOp, parse_definition, parse_formula, parse_operator_tokens,
};

fn sexpr(e: &Expr) -> String {
    match &e.kind {
        ExprKind::Num(v) => format!("{v}"),
        ExprKind::Bool(b) => b.to_string(),
        ExprKind::Name(n) => n.clone(),
        ExprKind::Unary { op, arg } => {
            let op = match op {
                UnaryOp::Neg => "-",
                UnaryOp::Plus => "+",
                UnaryOp::Not => "!",
            };
            format!("({op} {})", sexpr(arg))
        }
        ExprKind::Binary { op, lhs, rhs } => format!("({op:?} {} {})", sexpr(lhs), sexpr(rhs)),
        ExprKind::Ternary { cond, then, otherwise } => {
            format!("(? {} {} {})", sexpr(cond), sexpr(then), sexpr(otherwise))
        }
        ExprKind::Call { callee, args } => {
            let args: Vec<String> = args.iter().map(sexpr).collect();
            format!("({} {})", callee.name, args.join(" "))
        }
        ExprKind::Index { base, index } => format!("([] {} {})", sexpr(base), sexpr(index)),
    }
}

fn stmt_sexpr(stmt: &Stmt) -> String {
    match stmt {
        Stmt::Assign { local, name, value } => {
            format!(
                "({}= {} {})",
                if *local { "local " } else { "" },
                name.name,
                sexpr(value)
            )
        }
        Stmt::Return(e) => format!("(return {})", sexpr(e)),
        Stmt::Expr(e) => sexpr(e),
    }
}

/// The body of a one-expression definition `f(x...) = <body>`.
fn body(src: &str) -> String {
    let def = parse_definition(src).unwrap_or_else(|e| panic!("{src:?}: {e}"));
    let stmts: Vec<String> = def.body.iter().map(stmt_sexpr).collect();
    stmts.join("; ")
}

fn span_of(src: &str, needle: &str) -> Span {
    let start = src.find(needle).unwrap_or_else(|| panic!("{needle:?} not in {src:?}"));
    Span::new(start, start + needle.len())
}

#[track_caller]
fn assert_def_error(src: &str, message: &str, span: Span) {
    let err = parse_definition(src).expect_err(src);
    assert_eq!(err.message, message, "{src:?}");
    assert_eq!(err.span, span, "{src:?}: {err}");
}

#[test]
fn parser_literals() {
    assert_eq!(body("f() = 1"), "1");
    assert_eq!(body("f() = .5"), "0.5");
    assert_eq!(body("f() = 1e-6"), "0.000001");
    let def = parse_definition("f() = 1f-20").unwrap();
    match &def.body[..] {
        [
            Stmt::Expr(Expr {
                kind: ExprKind::Num(v), ..
            }),
        ] => assert_eq!(*v, 1e-20f32 as f64),
        other => panic!("{other:?}"),
    }
    assert_eq!(body("f() = true && !false"), "(And true (! false))");
    assert_eq!(body("f() = pi + Inf - NaN"), "(Sub (Add pi Inf) NaN)");
}

#[test]
fn parser_precedence() {
    assert_eq!(body("f(x) = -x^2"), "(- (Pow x 2))");
    assert_eq!(body("f() = 2^-1"), "(Pow 2 (- 1))");
    assert_eq!(body("f(x, y, z) = x^y^z"), "(Pow x (Pow y z))");
    assert_eq!(body("f(a, b, c) = a - b - c"), "(Sub (Sub a b) c)");
    assert_eq!(body("f(a, b, c) = a / b * c"), "(Mul (Div a b) c)");
    assert_eq!(body("f(a, b, c) = a + b * c"), "(Add a (Mul b c))");
    assert_eq!(body("f(a, b) = -a * +b"), "(Mul (- a) (+ b))");
    assert_eq!(
        body("f(a, b, c, d, e) = a < b && c >= d || !e"),
        "(Or (And (Lt a b) (Ge c d)) (! e))"
    );
}

#[test]
fn parser_ternary_nests_to_the_right() {
    assert_eq!(body("f(a, b, c, d, e) = a ? b : c ? d : e"), "(? a b (? c d e))");
    assert_eq!(body("f(a, b, c, d, e) = a ? b ? c : d : e"), "(? a (? b c d) e)");
    assert_eq!(body("f(t, a, b) = t > 0 ? a : b"), "(? (Gt t 0) a b)");
}

#[test]
fn parser_definition_forms() {
    let def = parse_definition("α(x, y) = x - y").unwrap();
    assert_eq!(def.kind, DefKind::Named);
    assert_eq!(def.name.as_ref().unwrap().name, "α");
    assert_eq!(def.name.as_ref().unwrap().span, Span::new(0, 2));
    assert_eq!(def.params.len(), 2);

    let def = parse_definition("(prediction::Float32, target::Float32) -> (prediction - target)^2").unwrap();
    assert_eq!(def.kind, DefKind::Lambda);
    assert!(def.name.is_none());
    assert_eq!(def.params[0].ty.as_ref().unwrap().name, "Float32");
    assert_eq!(body("(prediction::Float32, target) -> prediction"), "prediction");

    let def = parse_definition("x -> x^2").unwrap();
    assert_eq!(def.kind, DefKind::Lambda);
    assert_eq!(def.params[0].name.name, "x");

    let def = parse_definition("g(x::T, y::T)::T where {T<:Real} = x * y").unwrap();
    assert_eq!(def.return_type.as_ref().unwrap().name, "T");
    assert_eq!(def.type_vars[0].bound.as_ref().unwrap().name, "Real");
    assert_eq!(body("h(x::T) where T<:AbstractFloat = x"), "x");
    assert_eq!(body("h(x::Number)::Float64 = x"), "x");
    assert_eq!(body("k() = 0.0"), "0");
}

#[test]
fn parser_function_blocks() {
    let src = "function loss_fnc(prediction, target)\n    a = abs(prediction) # comment\n\n    local b = 10 * a; c = b + 1\n    return b + c\nend\n";
    let def = parse_definition(src).unwrap();
    assert_eq!(def.kind, DefKind::Function);
    assert_eq!(def.name.as_ref().unwrap().name, "loss_fnc");
    assert_eq!(
        body(src),
        "(= a (abs prediction)); (local = b (Mul 10 a)); (= c (Add b 1)); (return (Add b c))"
    );
    assert_eq!(def.span, Span::new(0, src.len() - 1));

    assert_eq!(body("function f(x) x^2 end"), "(Pow x 2)");
    assert_eq!(body("function f(x::T) where T\n    x\nend"), "x");
    assert_eq!(
        body("function f(x)::Float64\n    y = (x +\n        1)\n    y\nend"),
        "(= y (Add x 1)); y"
    );
}

#[test]
fn parser_newline_continuation() {
    assert_eq!(body("f(x) = x +\n    1"), "(Add x 1)");
    assert_eq!(body("f(x) = (x\n    + 1)"), "(Add x 1)");
    assert_eq!(body("f(x) =\n    x"), "x");
    assert_eq!(body("\n\n  f(x) = x  # trailing\n\n"), "x");
    assert_def_error("f(x) = x\n+ 1", "expected end of input, found `+`", Span::new(9, 10));
}

#[test]
fn parser_formulas() {
    let f = parse_formula("df = D(f, 1); df(x)").unwrap();
    let stmts: Vec<String> = f.stmts.iter().map(stmt_sexpr).collect();
    assert_eq!(stmts, ["(= df (D f 1))", "(df x)"]);

    let f = parse_formula("p1[1]*f(x,y) + p2[category]").unwrap();
    assert_eq!(
        stmt_sexpr(&f.stmts[0]),
        "(Add (Mul ([] p1 1) (f x y)) ([] p2 category))"
    );

    let src = "\n    v = shared(x1)\n    r = (\n        abs2(y1 - v) +\n        abs2(y2 - v)\n    )\n\n    r\n";
    let f = parse_formula(src).unwrap();
    let stmts: Vec<String> = f.stmts.iter().map(stmt_sexpr).collect();
    assert_eq!(
        stmts,
        [
            "(= v (shared x1))",
            "(= r (Add (abs2 (Sub y1 v)) (abs2 (Sub y2 v))))",
            "r"
        ]
    );

    let err = parse_formula("  \n ").unwrap_err();
    assert_eq!(err.message, "expected an expression, found end of input");
    let err = parse_formula("a b").unwrap_err();
    assert_eq!(err.message, "expected `;` or a new line, found `b`");
    assert_eq!(err.span, Span::new(2, 3));
}

#[test]
fn parser_rejects_unsupported_constructs() {
    let src = "f(x) = x .+ 1";
    assert_def_error(src, "broadcasting (`.`) is not supported", span_of(src, ".+"));
    let src = "f(x) = x .* 2";
    assert_def_error(src, "broadcasting (`.`) is not supported", span_of(src, ".*"));
    let src = "f(a) = max.(a, 0)";
    assert_def_error(src, "broadcasting (`.`) is not supported", span_of(src, "max.("));
    let src = "f(a) = a.data + 1";
    assert_def_error(src, "property access (`a.b`) is not supported", span_of(src, "a.data"));

    let src = "function f(x)\n    for i in 1:3\n        x = x + i\n    end\n    x\nend";
    let start = src.find("for").unwrap();
    let end = src.find("end").unwrap() + 3;
    assert_def_error(src, "`for` loops are not supported", Span::new(start, end));
    let src = "function f(x)\n    while x > 1\n        x = x / 2\n    end\n    x\nend";
    let start = src.find("while").unwrap();
    let end = src.find("end").unwrap() + 3;
    assert_def_error(src, "`while` loops are not supported", Span::new(start, end));
    let src = "function f(x)\n    if x > 0\n        return x\n    else\n        return 0\n    end\nend";
    let start = src.find("if").unwrap();
    let end = src.find("end").unwrap() + 3;
    assert_def_error(
        src,
        "`if` blocks are not supported; use `c ? a : b` or `ifelse(c, a, b)`",
        Span::new(start, end),
    );
    let src = "begin\n    f(x) = x\nend";
    assert_def_error(src, "`begin` blocks are not supported", Span::new(0, src.len()));

    let src = "f(x) = [x, 1]";
    assert_def_error(src, "array literals are not supported", span_of(src, "[x, 1]"));
    let src = "f(x) = \"x\"";
    assert_def_error(src, "strings are not supported", span_of(src, "\"x\""));
    let src = "f(x::Letter) = x";
    assert_def_error(
        src,
        "type `Letter` is not supported; use Float32, Float64, Real, or a `where` type variable",
        span_of(src, "Letter"),
    );
    let src = "f(x)::Int = x";
    assert_def_error(
        src,
        "type `Int` is not supported; use Float32, Float64, Real, or a `where` type variable",
        span_of(src, "Int"),
    );
    let src = "f(x) = 0 < x < 1";
    assert_def_error(
        src,
        "chained comparisons are not supported; use `a < b && b < c`",
        span_of(src, "0 < x < 1"),
    );
    let src = "f(x) = 2x + 1";
    assert_def_error(
        src,
        "implicit multiplication is not supported; write `2 * x`",
        span_of(src, "2x"),
    );
    let src = "f() = 1e";
    assert_def_error(
        src,
        "invalid number `1e`; an exponent needs digits, as in `1e-3`",
        span_of(src, "1e"),
    );
    let src = "f(x, y, x) = x";
    assert_def_error(src, "parameter `x` appears more than once", Span::new(8, 9));
    let src = "(a, a) -> a";
    assert_def_error(src, "parameter `a` appears more than once", Span::new(4, 5));
    let src = "f(x) = 2(x + 1)";
    assert_def_error(
        src,
        "implicit multiplication is not supported; write `2 * x`",
        span_of(src, "2("),
    );
    let src = "f(p) = p[1] + 1";
    assert_def_error(
        src,
        "indexing is only allowed in template formulas",
        span_of(src, "p[1]"),
    );
    let src = "f(g) = D(g, 1)";
    assert_def_error(src, "`D` is only allowed in template formulas", span_of(src, "D(g, 1)"));
    let src = "f(x) = (x, 1)";
    assert_def_error(src, "expected `)`, found `,`", span_of(src, ","));
    let src = "f(x) = x +";
    assert_def_error(src, "expected an expression, found end of input", Span::new(10, 10));
    let src = "f(x = x";
    assert_def_error(src, "expected `)`, found `=`", span_of(src, "="));
    let src = "1 + x";
    assert_def_error(
        src,
        "expected a definition such as `f(x) = ...`, found `1`",
        Span::new(0, 1),
    );
}

#[test]
fn parser_error_display() {
    let err = parse_definition("f(a) = a.data").unwrap_err();
    assert_eq!(
        err.to_string(),
        "property access (`a.b`) is not supported at line 1, column 8\nf(a) = a.data\n       ^^^^^^"
    );
}

fn chain(separator: &str, term: &str, terms: usize) -> String {
    format!("f(x) = {}", vec![term; terms].join(separator))
}

#[test]
fn parser_bounds_nesting_and_length_instead_of_overflowing_the_stack() {
    let deep = 100_000;
    let too_deep = "expression is too long or too deeply nested";
    for src in [
        format!("f(x) = {}x{}", "(".repeat(deep), ")".repeat(deep)),
        format!("f(x) = {}x", "-".repeat(deep)),
        format!("f(x) = {}", "x^".repeat(deep) + "x"),
        format!("f(x) = {}x", "x > 0 ? ".repeat(deep) + &"x : ".repeat(deep)),
        chain(" + ", "x", 60_000),
        chain(" * ", "x", 60_000),
        chain(" && ", "x > 0", 60_000),
        chain(" || ", "x > 0", 60_000),
        chain(" + ", "x", 257),
    ] {
        let err = parse_definition(&src).unwrap_err();
        assert_eq!(err.message, too_deep);
    }
    let err = parse_formula(&format!("p{}", "[1]".repeat(deep))).unwrap_err();
    assert_eq!(err.message, too_deep);
}

/// The largest accepted expressions parse, compile, clone, and drop on a 1 MiB stack, the WASM
/// default.
#[test]
fn largest_accepted_expressions_fit_a_small_stack() {
    let sources = [
        chain(" + ", "x", 256),
        chain(" && ", "x > 0", 128).replacen("= ", "= ifelse(", 1) + ", x, -x)",
        format!("f(x) = {}x{}", "sin(".repeat(30), ")".repeat(30)),
        format!("f(x) = {}x{}", "(".repeat(31), ")".repeat(31)),
    ];
    std::thread::Builder::new()
        .stack_size(1 << 20)
        .spawn(move || {
            for src in sources {
                let def = parse_definition(&src).unwrap();
                drop(def.clone());
                CustomOps::<f64>::default().define(&src).unwrap();
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn operator_tokens_recognize_every_definition_form() {
    let (custom, names) =
        parse_operator_tokens::<f64, _>(["+", "cos", "sq(x) = x*x", "function cube3(x)\n    return x*x*x\nend"])
            .unwrap();
    assert_eq!(names, ["+", "cos", "sq", "cube3"]);
    assert_eq!(custom.len(), 2);
    let err = parse_operator_tokens::<f64, _>(["x -> x^2"]).unwrap_err();
    assert!(err.message.contains("custom operators need a name"), "{}", err.message);
}
