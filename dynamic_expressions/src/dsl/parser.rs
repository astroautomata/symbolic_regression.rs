use super::ast::{BinaryOp, DefKind, Definition, Expr, ExprKind, Formula, Ident, Param, Stmt, TypeVar, UnaryOp};
use super::lexer::{Keyword, Tok, Token, tokenize};
use super::{DslError, Span};

const BUILTIN_TYPES: [&str; 5] = ["Float32", "Float64", "Real", "Number", "AbstractFloat"];

/// Every recursive grammar cycle passes through `expr` or `unary` (a parenthesized level passes
/// through both). Bounding their nesting bounds the parser's recursion, and bounding node height
/// bounds every later walk of the tree, including left-nested chains such as `a + b + ... + z`
/// that the parser builds in a loop. Running out of stack aborts the process rather than
/// panicking, and WASM stacks are small.
const MAX_NESTING: u32 = 64;
const MAX_HEIGHT: u32 = 256;
const TOO_DEEP: &str = "expression is too long or too deeply nested";

/// Parses a custom operator, loss, or early-stop condition: `f(x) = ...`, `(x, y) -> ...`, or
/// `function f(x) ... end`.
pub fn parse_definition(src: &str) -> Result<Definition, DslError> {
    let mut p = Parser::new(src, false);
    let def = match &p.cur().tok {
        Tok::Kw(Keyword::Function) => p.function_block()?,
        Tok::LParen => p.paren_lambda()?,
        Tok::Ident(_) if p.nth(1).tok == Tok::Arrow => p.ident_lambda()?,
        Tok::Ident(_) => p.named_def()?,
        _ => return Err(p.unexpected("a definition such as `f(x) = ...`")),
    };
    p.skip_separators();
    p.expect_eof()?;
    Ok(def)
}

/// Parses a template combine formula: statements separated by `;` or newlines, the last being
/// the value. Formulas may index parameters (`p[i]`) and use `D(f, i)`.
pub fn parse_formula(src: &str) -> Result<Formula, DslError> {
    let mut p = Parser::new(src, true);
    let mut stmts = Vec::new();
    loop {
        p.skip_separators();
        if p.cur().tok == Tok::Eof {
            break;
        }
        stmts.push(p.stmt()?);
        if !matches!(p.peek()?, Tok::Newline | Tok::Semi | Tok::Eof) {
            return Err(p.unexpected("`;` or a new line"));
        }
    }
    let Some(last) = stmts.last() else {
        return Err(p.unexpected("an expression"));
    };
    let span = stmt_span(&stmts[0]).to(stmt_span(last));
    Ok(Formula {
        stmts,
        span,
        source: src.to_string(),
    })
}

fn stmt_span(stmt: &Stmt) -> Span {
    match stmt {
        Stmt::Assign { name, value, .. } => name.span.to(value.span),
        Stmt::Return(e) | Stmt::Expr(e) => e.span,
    }
}

struct Parser<'s> {
    src: &'s str,
    toks: Vec<Token>,
    pos: usize,
    template: bool,
    depth: u32,
}

impl<'s> Parser<'s> {
    fn new(src: &'s str, template: bool) -> Self {
        Self {
            src,
            toks: tokenize(src),
            pos: 0,
            template,
            depth: 0,
        }
    }

    fn err(&self, message: impl Into<String>, span: Span) -> DslError {
        DslError::new(message, span, self.src)
    }

    fn cur(&self) -> &Token {
        self.nth(0)
    }

    fn nested<R>(&mut self, f: impl FnOnce(&mut Self) -> Result<R, DslError>) -> Result<R, DslError> {
        if self.depth == MAX_NESTING {
            return Err(self.err(TOO_DEEP, self.cur().span));
        }
        self.depth += 1;
        let result = f(self);
        self.depth -= 1;
        result
    }

    fn node(&self, kind: ExprKind, span: Span) -> Result<Expr, DslError> {
        let expr = Expr::new(kind, span);
        if expr.height > MAX_HEIGHT {
            return Err(self.err(TOO_DEEP, span));
        }
        Ok(expr)
    }

    fn nth(&self, n: usize) -> &Token {
        let last = self.toks.len() - 1;
        &self.toks[(self.pos + n).min(last)]
    }

    /// The current token, or the error it stands for.
    fn peek(&self) -> Result<&Tok, DslError> {
        let token = self.cur();
        match &token.tok {
            Tok::Bad(message) => Err(self.err(message.clone(), token.span)),
            tok => Ok(tok),
        }
    }

    fn bump(&mut self) -> Span {
        let span = self.cur().span;
        if self.pos < self.toks.len() - 1 {
            self.pos += 1;
        }
        span
    }

    fn eat(&mut self, tok: &Tok) -> Result<bool, DslError> {
        if self.peek()? == tok {
            self.bump();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn expect(&mut self, tok: Tok) -> Result<Span, DslError> {
        if self.peek()? == &tok {
            Ok(self.bump())
        } else {
            Err(self.unexpected(&tok.describe()))
        }
    }

    fn expect_eof(&self) -> Result<(), DslError> {
        match self.peek()? {
            Tok::Eof => Ok(()),
            _ => Err(self.unexpected("end of input")),
        }
    }

    fn skip_separators(&mut self) {
        while matches!(self.cur().tok, Tok::Newline | Tok::Semi) {
            self.bump();
        }
    }

    /// The error for the current token when `expected` was required there. Rejected keywords and
    /// bad tokens report their own message instead.
    fn unexpected(&self, expected: &str) -> DslError {
        let token = self.cur();
        match &token.tok {
            Tok::Bad(message) => self.err(message.clone(), token.span),
            Tok::Kw(kw @ (Keyword::For | Keyword::While | Keyword::If | Keyword::Begin)) => self.rejected_block(*kw),
            tok => self.err(format!("expected {expected}, found {}", tok.describe()), token.span),
        }
    }

    /// The error for a `for`/`while`/`if`/`begin` block at the current token, spanning up to its
    /// matching `end`.
    fn rejected_block(&self, kw: Keyword) -> DslError {
        let message = match kw {
            Keyword::For => "`for` loops are not supported",
            Keyword::While => "`while` loops are not supported",
            Keyword::If => "`if` blocks are not supported; use `c ? a : b` or `ifelse(c, a, b)`",
            _ => "`begin` blocks are not supported",
        };
        let start = self.cur().span;
        let mut depth = 0usize;
        let mut end = start;
        for token in &self.toks[self.pos..] {
            match token.tok {
                Tok::Kw(k) if k.opens_block() => depth += 1,
                Tok::Kw(Keyword::End) => depth -= 1,
                Tok::Eof => break,
                _ => {}
            }
            end = token.span;
            if depth == 0 {
                break;
            }
        }
        self.err(message, start.to(end))
    }

    /// The span from the bracket at the current token through its matching closer (or the end of
    /// input when unbalanced).
    fn bracketed_span(&self) -> Span {
        let start = self.cur().span;
        let mut depth = 0usize;
        let mut end = start;
        for token in &self.toks[self.pos..] {
            match token.tok {
                Tok::LParen | Tok::LBracket | Tok::LBrace => depth += 1,
                Tok::RParen | Tok::RBracket | Tok::RBrace => depth -= 1,
                Tok::Eof => break,
                _ => {}
            }
            end = token.span;
            if depth == 0 {
                break;
            }
        }
        start.to(end)
    }

    fn ident(&mut self, what: &str) -> Result<Ident, DslError> {
        match self.peek()? {
            Tok::Ident(name) => {
                let name = name.clone();
                let span = self.bump();
                Ok(Ident { name, span })
            }
            _ => Err(self.unexpected(what)),
        }
    }

    // ---- Definitions ----

    fn named_def(&mut self) -> Result<Definition, DslError> {
        let name = self.ident("a name")?;
        let params = self.param_list()?;
        let (return_type, type_vars) = self.signature_tail()?;
        self.check_types(&params, return_type.as_ref(), &type_vars)?;
        self.expect(Tok::Assign)?;
        let body = self.expr()?;
        let span = name.span.to(body.span);
        Ok(Definition {
            kind: DefKind::Named,
            name: Some(name),
            params,
            return_type,
            type_vars,
            body: vec![Stmt::Expr(body)],
            span,
            source: self.src.to_string(),
        })
    }

    fn paren_lambda(&mut self) -> Result<Definition, DslError> {
        let start = self.cur().span;
        let params = self.param_list()?;
        self.check_types(&params, None, &[])?;
        self.expect(Tok::Arrow)?;
        let body = self.expr()?;
        Ok(self.lambda(params, start.to(body.span), body))
    }

    fn ident_lambda(&mut self) -> Result<Definition, DslError> {
        let name = self.ident("a parameter name")?;
        self.expect(Tok::Arrow)?;
        let body = self.expr()?;
        let span = name.span.to(body.span);
        Ok(self.lambda(vec![Param { name, ty: None }], span, body))
    }

    fn lambda(&self, params: Vec<Param>, span: Span, body: Expr) -> Definition {
        Definition {
            kind: DefKind::Lambda,
            name: None,
            params,
            return_type: None,
            type_vars: Vec::new(),
            body: vec![Stmt::Expr(body)],
            span,
            source: self.src.to_string(),
        }
    }

    fn function_block(&mut self) -> Result<Definition, DslError> {
        let start = self.bump();
        let name = self.ident("a function name")?;
        let params = self.param_list()?;
        let (return_type, type_vars) = self.signature_tail()?;
        self.check_types(&params, return_type.as_ref(), &type_vars)?;
        let mut body = Vec::new();
        let end = loop {
            self.skip_separators();
            match self.peek()? {
                Tok::Kw(Keyword::End) => break self.bump(),
                Tok::Eof => return Err(self.unexpected("`end`")),
                _ => {}
            }
            body.push(self.stmt()?);
            if !matches!(self.peek()?, Tok::Newline | Tok::Semi | Tok::Kw(Keyword::End)) {
                return Err(self.unexpected("`;`, a new line, or `end`"));
            }
        };
        Ok(Definition {
            kind: DefKind::Function,
            name: Some(name),
            params,
            return_type,
            type_vars,
            body,
            span: start.to(end),
            source: self.src.to_string(),
        })
    }

    /// `( [param {, param}] )`
    fn param_list(&mut self) -> Result<Vec<Param>, DslError> {
        self.expect(Tok::LParen)?;
        let mut params = Vec::new();
        if self.eat(&Tok::RParen)? {
            return Ok(params);
        }
        loop {
            let name = self.ident("a parameter name")?;
            if params.iter().any(|p: &Param| p.name.name == name.name) {
                return Err(self.err(format!("parameter `{}` appears more than once", name.name), name.span));
            }
            let ty = if self.eat(&Tok::ColonColon)? {
                Some(self.type_name()?)
            } else {
                None
            };
            params.push(Param { name, ty });
            if !self.eat(&Tok::Comma)? {
                break;
            }
        }
        self.expect(Tok::RParen)?;
        Ok(params)
    }

    /// A type annotation. Only plain names are accepted, so parametric types such as
    /// `Dataset{T,L}` are rejected by name here rather than as a stray `{`.
    fn type_name(&mut self) -> Result<Ident, DslError> {
        let ty = self.ident("a type")?;
        if *self.peek()? == Tok::LBrace {
            return Err(self.err(
                format!(
                    "type `{}` is not supported; use Float32, Float64, Real, or a `where` type variable",
                    ty.name
                ),
                ty.span,
            ));
        }
        Ok(ty)
    }

    /// `[:: Type] [where T | where {T <: Bound, ...}]`
    fn signature_tail(&mut self) -> Result<(Option<Ident>, Vec<TypeVar>), DslError> {
        let return_type = if self.eat(&Tok::ColonColon)? {
            Some(self.type_name()?)
        } else {
            None
        };
        let mut type_vars = Vec::new();
        if self.eat(&Tok::Kw(Keyword::Where))? {
            let braced = self.eat(&Tok::LBrace)?;
            loop {
                let name = self.ident("a type variable")?;
                let bound = if self.eat(&Tok::SubType)? {
                    Some(self.ident("a type")?)
                } else {
                    None
                };
                type_vars.push(TypeVar { name, bound });
                if !(braced && self.eat(&Tok::Comma)?) {
                    break;
                }
            }
            if braced {
                self.expect(Tok::RBrace)?;
            }
        }
        Ok((return_type, type_vars))
    }

    fn check_types(
        &self,
        params: &[Param],
        return_type: Option<&Ident>,
        type_vars: &[TypeVar],
    ) -> Result<(), DslError> {
        let annotations = params
            .iter()
            .filter_map(|p| p.ty.as_ref())
            .chain(return_type)
            .chain(type_vars.iter().filter_map(|v| v.bound.as_ref()));
        for ty in annotations {
            let known = BUILTIN_TYPES.contains(&ty.name.as_str()) || type_vars.iter().any(|v| v.name.name == ty.name);
            if !known {
                return Err(self.err(
                    format!(
                        "type `{}` is not supported; use Float32, Float64, Real, or a `where` type variable",
                        ty.name
                    ),
                    ty.span,
                ));
            }
        }
        Ok(())
    }

    // ---- Statements ----

    fn stmt(&mut self) -> Result<Stmt, DslError> {
        match self.peek()? {
            Tok::Kw(Keyword::Local) => {
                self.bump();
                let name = self.ident("a variable name")?;
                self.expect(Tok::Assign)?;
                let value = self.expr()?;
                Ok(Stmt::Assign {
                    local: true,
                    name,
                    value,
                })
            }
            Tok::Kw(Keyword::Return) => {
                self.bump();
                Ok(Stmt::Return(self.expr()?))
            }
            Tok::Ident(_) if self.nth(1).tok == Tok::Comma => Err(self.err(
                "destructuring assignment (`a, b = ...`) is not supported",
                self.cur().span,
            )),
            Tok::Ident(_) if self.nth(1).tok == Tok::Assign => {
                let name = self.ident("a variable name")?;
                self.bump();
                let value = self.expr()?;
                Ok(Stmt::Assign {
                    local: false,
                    name,
                    value,
                })
            }
            _ => Ok(Stmt::Expr(self.expr()?)),
        }
    }

    // ---- Expressions ----

    fn expr(&mut self) -> Result<Expr, DslError> {
        self.nested(Self::ternary)
    }

    fn ternary(&mut self) -> Result<Expr, DslError> {
        let cond = self.or()?;
        if !self.eat(&Tok::Question)? {
            return Ok(cond);
        }
        let then = self.expr()?;
        self.expect(Tok::Colon)?;
        let otherwise = self.expr()?;
        let span = cond.span.to(otherwise.span);
        self.node(
            ExprKind::Ternary {
                cond: Box::new(cond),
                then: Box::new(then),
                otherwise: Box::new(otherwise),
            },
            span,
        )
    }

    fn binary(&self, op: BinaryOp, lhs: Expr, rhs: Expr) -> Result<Expr, DslError> {
        let span = lhs.span.to(rhs.span);
        self.node(
            ExprKind::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            },
            span,
        )
    }

    fn or(&mut self) -> Result<Expr, DslError> {
        let mut lhs = self.and()?;
        while self.eat(&Tok::OrOr)? {
            let rhs = self.and()?;
            lhs = self.binary(BinaryOp::Or, lhs, rhs)?;
        }
        Ok(lhs)
    }

    fn and(&mut self) -> Result<Expr, DslError> {
        let mut lhs = self.cmp()?;
        while self.eat(&Tok::AndAnd)? {
            let rhs = self.cmp()?;
            lhs = self.binary(BinaryOp::And, lhs, rhs)?;
        }
        Ok(lhs)
    }

    fn cmp_op(&self) -> Result<Option<BinaryOp>, DslError> {
        Ok(Some(match self.peek()? {
            Tok::Lt => BinaryOp::Lt,
            Tok::Le => BinaryOp::Le,
            Tok::Gt => BinaryOp::Gt,
            Tok::Ge => BinaryOp::Ge,
            Tok::EqEq => BinaryOp::Eq,
            Tok::Ne => BinaryOp::Ne,
            _ => return Ok(None),
        }))
    }

    fn cmp(&mut self) -> Result<Expr, DslError> {
        let lhs = self.add()?;
        let Some(op) = self.cmp_op()? else {
            return Ok(lhs);
        };
        self.bump();
        let rhs = self.add()?;
        if self.cmp_op()?.is_some() {
            self.bump();
            let third = self.add()?;
            return Err(self.err(
                "chained comparisons are not supported; use `a < b && b < c`",
                lhs.span.to(third.span),
            ));
        }
        self.binary(op, lhs, rhs)
    }

    fn add(&mut self) -> Result<Expr, DslError> {
        let mut lhs = self.mul()?;
        loop {
            let op = match self.peek()? {
                Tok::Plus => BinaryOp::Add,
                Tok::Minus => BinaryOp::Sub,
                _ => return Ok(lhs),
            };
            self.bump();
            let rhs = self.mul()?;
            lhs = self.binary(op, lhs, rhs)?;
        }
    }

    fn mul(&mut self) -> Result<Expr, DslError> {
        let mut lhs = self.unary()?;
        loop {
            let op = match self.peek()? {
                Tok::Star => BinaryOp::Mul,
                Tok::Slash => BinaryOp::Div,
                _ => return Ok(lhs),
            };
            self.bump();
            let rhs = self.unary()?;
            lhs = self.binary(op, lhs, rhs)?;
        }
    }

    fn unary(&mut self) -> Result<Expr, DslError> {
        self.nested(Self::prefixed)
    }

    fn prefixed(&mut self) -> Result<Expr, DslError> {
        let op = match self.peek()? {
            Tok::Minus => UnaryOp::Neg,
            Tok::Plus => UnaryOp::Plus,
            Tok::Bang => UnaryOp::Not,
            _ => return self.power(),
        };
        let start = self.bump();
        let arg = self.unary()?;
        let span = start.to(arg.span);
        self.node(ExprKind::Unary { op, arg: Box::new(arg) }, span)
    }

    fn power(&mut self) -> Result<Expr, DslError> {
        let base = self.postfix()?;
        if !self.eat(&Tok::Caret)? {
            return Ok(base);
        }
        let exponent = self.unary()?;
        self.binary(BinaryOp::Pow, base, exponent)
    }

    fn postfix(&mut self) -> Result<Expr, DslError> {
        let mut base = self.primary()?;
        loop {
            match (self.peek()?, &base.kind) {
                (Tok::LParen, ExprKind::Name(name)) => {
                    let callee = Ident {
                        name: name.clone(),
                        span: base.span,
                    };
                    if callee.name == "D" && !self.template {
                        return Err(self.err(
                            "`D` is only allowed in template formulas",
                            base.span.to(self.bracketed_span()),
                        ));
                    }
                    let args = self.call_args()?;
                    let span = base.span.to(self.toks[self.pos - 1].span);
                    base = self.node(ExprKind::Call { callee, args }, span)?;
                }
                (Tok::LParen, ExprKind::Call { .. } | ExprKind::Apply { .. }) if self.template => {
                    let args = self.call_args()?;
                    let span = base.span.to(self.toks[self.pos - 1].span);
                    let func = Box::new(base);
                    base = self.node(ExprKind::Apply { func, args }, span)?;
                }
                (Tok::LBracket, _) => {
                    if !self.template {
                        return Err(self.err(
                            "indexing is only allowed in template formulas",
                            base.span.to(self.bracketed_span()),
                        ));
                    }
                    self.bump();
                    let index = self.expr()?;
                    let end = self.expect(Tok::RBracket)?;
                    let span = base.span.to(end);
                    base = self.node(
                        ExprKind::Index {
                            base: Box::new(base),
                            index: Box::new(index),
                        },
                        span,
                    )?;
                }
                _ => break,
            }
        }
        let next = self.cur();
        let juxtaposed =
            matches!(next.tok, Tok::Ident(_) | Tok::LParen | Tok::Num(_)) && next.span.start == base.span.end;
        if matches!(base.kind, ExprKind::Num(_)) && juxtaposed {
            return Err(self.err(
                "implicit multiplication is not supported; write `2 * x`",
                base.span.to(next.span),
            ));
        }
        Ok(base)
    }

    fn call_args(&mut self) -> Result<Vec<Expr>, DslError> {
        self.expect(Tok::LParen)?;
        let mut args = Vec::new();
        if self.eat(&Tok::RParen)? {
            return Ok(args);
        }
        loop {
            args.push(self.expr()?);
            if !self.eat(&Tok::Comma)? {
                break;
            }
        }
        self.expect(Tok::RParen)?;
        Ok(args)
    }

    fn primary(&mut self) -> Result<Expr, DslError> {
        let kind = match self.peek()? {
            Tok::Num(v) => ExprKind::Num(*v),
            Tok::Ident(name) => ExprKind::Name(name.clone()),
            Tok::Kw(Keyword::True) => ExprKind::Bool(true),
            Tok::Kw(Keyword::False) => ExprKind::Bool(false),
            Tok::LParen => {
                let start = self.bump();
                let inner = self.expr()?;
                let end = self.expect(Tok::RParen)?;
                return Ok(Expr {
                    span: start.to(end),
                    ..inner
                });
            }
            Tok::LBracket => return Err(self.err("array literals are not supported", self.bracketed_span())),
            _ => return Err(self.unexpected("an expression")),
        };
        let span = self.bump();
        Ok(Expr::new(kind, span))
    }
}
