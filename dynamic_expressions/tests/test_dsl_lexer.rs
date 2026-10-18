use dynamic_expressions::dsl::lexer::{Keyword, Tok, tokenize};
use dynamic_expressions::dsl::{DslError, Span};

fn toks(src: &str) -> Vec<Tok> {
    tokenize(src).into_iter().map(|t| t.tok).collect()
}

fn num(src: &str) -> f64 {
    match toks(src).as_slice() {
        [Tok::Num(v), Tok::Eof] => *v,
        other => panic!("{src:?} lexed as {other:?}"),
    }
}

fn ident(s: &str) -> Tok {
    Tok::Ident(s.to_string())
}

#[test]
fn lexer_number_literals() {
    assert_eq!(num("1"), 1.0);
    assert_eq!(num("0.5"), 0.5);
    assert_eq!(num(".5"), 0.5);
    assert_eq!(num("2."), 2.0);
    assert_eq!(num("1e-6"), 1e-6);
    assert_eq!(num("1.0e-8"), 1.0e-8);
    assert_eq!(num("3E+2"), 300.0);
    assert_eq!(num("1f0"), 1.0);
    assert_eq!(num("0.5f0"), 0.5);
    assert_eq!(num("1f-20"), 1e-20f32 as f64);
    assert_ne!(num("1f-20"), 1e-20);
    assert_eq!(num("2.5f3"), 2500.0);
    assert_eq!(num("1_000"), 1000.0);
    assert_eq!(num("1_000.000_5"), 1000.0005);
    assert_eq!(num("2_5e1_0"), 25e10);
    assert_eq!(toks("1_"), vec![Tok::Num(1.0), ident("_"), Tok::Eof]);
}

#[test]
fn lexer_exponent_without_digits_is_a_malformed_number() {
    for (src, text) in [
        ("1e", "1e"),
        ("2E+", "2E+"),
        ("1f", "1f"),
        ("2exp(x)", "2e"),
        ("1.5e-x", "1.5e-"),
    ] {
        let bad = Tok::Bad(format!(
            "invalid number `{text}`; an exponent needs digits, as in `1e-3`"
        ));
        assert_eq!(toks(src)[0], bad, "{src:?}");
        assert_eq!(tokenize(src)[0].span, Span::new(0, text.len()), "{src:?}");
    }
}

#[test]
fn lexer_operators() {
    assert_eq!(
        toks("a+b-c*d/e^f"),
        vec![
            ident("a"),
            Tok::Plus,
            ident("b"),
            Tok::Minus,
            ident("c"),
            Tok::Star,
            ident("d"),
            Tok::Slash,
            ident("e"),
            Tok::Caret,
            ident("f"),
            Tok::Eof
        ]
    );
    assert_eq!(
        toks("< <= > >= == != && || ! ? : :: <: = ->"),
        vec![
            Tok::Lt,
            Tok::Le,
            Tok::Gt,
            Tok::Ge,
            Tok::EqEq,
            Tok::Ne,
            Tok::AndAnd,
            Tok::OrOr,
            Tok::Bang,
            Tok::Question,
            Tok::Colon,
            Tok::ColonColon,
            Tok::SubType,
            Tok::Assign,
            Tok::Arrow,
            Tok::Eof
        ]
    );
}

#[test]
fn lexer_identifiers_and_keywords() {
    assert_eq!(
        toks("α(x, y) = x - y"),
        vec![
            ident("α"),
            Tok::LParen,
            ident("x"),
            Tok::Comma,
            ident("y"),
            Tok::RParen,
            Tok::Assign,
            ident("x"),
            Tok::Minus,
            ident("y"),
            Tok::Eof
        ]
    );
    assert_eq!(toks("_a1 b!"), vec![ident("_a1"), ident("b!"), Tok::Eof]);
    assert_eq!(toks("x!=y"), vec![ident("x"), Tok::Ne, ident("y"), Tok::Eof]);
    assert_eq!(
        toks("function end return local where true false"),
        vec![
            Tok::Kw(Keyword::Function),
            Tok::Kw(Keyword::End),
            Tok::Kw(Keyword::Return),
            Tok::Kw(Keyword::Local),
            Tok::Kw(Keyword::Where),
            Tok::Kw(Keyword::True),
            Tok::Kw(Keyword::False),
            Tok::Eof
        ]
    );
}

#[test]
fn lexer_spans_are_byte_offsets() {
    let tokens = tokenize("α + 10");
    assert_eq!(tokens[0].span, Span::new(0, 2));
    assert_eq!(tokens[1].span, Span::new(3, 4));
    assert_eq!(tokens[2].span, Span::new(5, 7));
    assert_eq!(tokens[3].span, Span::new(7, 7));
}

#[test]
fn lexer_newlines_end_statements_only_at_depth_zero() {
    assert_eq!(
        toks("\n\n  a = 1\n\n\nb\n"),
        vec![
            ident("a"),
            Tok::Assign,
            Tok::Num(1.0),
            Tok::Newline,
            ident("b"),
            Tok::Newline,
            Tok::Eof
        ]
    );
    assert_eq!(
        toks("(a\n+ b)"),
        vec![Tok::LParen, ident("a"), Tok::Plus, ident("b"), Tok::RParen, Tok::Eof]
    );
    assert_eq!(
        toks("f(a,\nb)[\n1]"),
        vec![
            ident("f"),
            Tok::LParen,
            ident("a"),
            Tok::Comma,
            ident("b"),
            Tok::RParen,
            Tok::LBracket,
            Tok::Num(1.0),
            Tok::RBracket,
            Tok::Eof
        ]
    );
}

#[test]
fn lexer_newline_after_continuation_tokens_is_ignored() {
    for (src, op) in [
        ("a +\nb", Tok::Plus),
        ("a *\nb", Tok::Star),
        ("a &&\nb", Tok::AndAnd),
        ("a ?\nb", Tok::Question),
        ("a :\nb", Tok::Colon),
        ("a =\nb", Tok::Assign),
        ("a ->\nb", Tok::Arrow),
        ("a ,\nb", Tok::Comma),
        ("a <=\nb", Tok::Le),
    ] {
        assert_eq!(toks(src), vec![ident("a"), op, ident("b"), Tok::Eof], "{src:?}");
    }
    assert_eq!(
        toks("a\n+ b"),
        vec![ident("a"), Tok::Newline, Tok::Plus, ident("b"), Tok::Eof]
    );
}

#[test]
fn lexer_comments_run_to_end_of_line() {
    assert_eq!(
        toks("a # comment ( [\nb # trailing"),
        vec![ident("a"), Tok::Newline, ident("b"), Tok::Eof]
    );
}

#[test]
fn lexer_rejected_constructs_become_bad_tokens() {
    let cases = [
        ("x .+ y", "broadcasting (`.`) is not supported", Span::new(2, 4)),
        ("max.(a, 0)", "broadcasting (`.`) is not supported", Span::new(0, 5)),
        ("a.data", "property access (`a.b`) is not supported", Span::new(0, 6)),
        ("\"abc\"", "strings are not supported", Span::new(0, 5)),
        ("a & b", "unexpected character `&`", Span::new(2, 3)),
    ];
    for (src, message, span) in cases {
        let bad = tokenize(src)
            .into_iter()
            .find(|t| matches!(t.tok, Tok::Bad(_)))
            .unwrap_or_else(|| panic!("{src:?} produced no bad token"));
        assert_eq!(bad.tok, Tok::Bad(message.to_string()), "{src:?}");
        assert_eq!(bad.span, span, "{src:?}");
    }
}

#[test]
fn error_display_points_at_the_span() {
    let err = DslError::new("unknown name `z`", Span::new(15, 16), "f(x) = x +\n    z * 2");
    assert_eq!(
        err.to_string(),
        "unknown name `z` at line 2, column 5\n    z * 2\n    ^"
    );
    let err = DslError::new("bad", Span::new(0, 4), "abcdef");
    assert_eq!(err.to_string(), "bad at line 1, column 1\nabcdef\n^^^^");
    let err = DslError::new("expected `)`, found end of input", Span::new(4, 4), "f(x(");
    assert_eq!(
        err.to_string(),
        "expected `)`, found end of input at line 1, column 5\nf(x(\n    ^"
    );
}
