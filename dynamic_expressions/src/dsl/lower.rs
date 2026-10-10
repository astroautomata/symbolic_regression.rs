use super::ast::{BinaryOp, Definition, Expr, ExprKind, Formula, Ident, Stmt, UnaryOp};
use super::ir::{Graph, Node, NodeId, Ty};
use super::kernels::Kernel;
use super::{DslError, Span};

/// A definition lowered to IR: `Arg(i)` is parameter `i` and `root` is the returned value.
#[derive(Clone, Debug)]
pub struct Lowered {
    pub name: Option<String>,
    pub arity: u8,
    pub graph: Graph,
    pub root: NodeId,
    pub ty: Ty,
    /// The span of the expression that produces the returned value.
    pub root_span: Span,
}

/// The names a template formula sees: feature columns (`Arg(i)`), subexpressions, and parameter
/// vectors with their lengths.
pub(crate) struct TemplateScope<'a> {
    pub variables: &'a [String],
    pub expressions: &'a [String],
    pub parameters: &'a [(String, usize)],
}

/// A template formula lowered to IR, with the number of arguments each subexpression is called
/// with (`None` for one that is never called).
pub(crate) struct LoweredFormula {
    pub graph: Graph,
    pub root: NodeId,
    pub arities: Vec<Option<u8>>,
}

#[derive(Copy, Clone, Debug)]
struct Value {
    id: NodeId,
    ty: Ty,
}

/// What a name refers to. Only template formulas bind subexpressions and parameter vectors.
#[derive(Clone, Debug)]
enum Binding {
    Value(Value),
    /// Subexpression `expr`, differentiated with respect to each argument in `wrt`.
    Func {
        expr: u16,
        wrt: Vec<u16>,
    },
    Params(u16),
}

/// Lowers `def` to IR. Calls to `customs` (earlier custom operators) are inlined.
pub fn lower(def: &Definition, customs: &[Lowered]) -> Result<Lowered, DslError> {
    let mut cx = Lowerer::new(def.source(), customs, def.name.as_ref().map(|n| n.name.as_str()), None);
    let arity = u8::try_from(def.params.len())
        .map_err(|_| cx.err("operators take at most 255 arguments".to_string(), def.span))?;
    for (i, param) in def.params.iter().enumerate() {
        let id = cx.graph.arg(i as u16);
        cx.bind(&param.name.name, Binding::Value(Value { id, ty: Ty::Num }));
    }

    let Some((value, root_span)) = cx.body(&def.body)? else {
        return Err(cx.err("function body has no value".to_string(), def.span));
    };

    let (graph, roots) = cx.graph.extract(&[value.id]);
    Ok(Lowered {
        name: def.name.as_ref().map(|n| n.name.clone()),
        arity,
        graph,
        root: roots[0],
        ty: value.ty,
        root_span,
    })
}

/// Lowers a template combine formula. `Arg(i)` is feature column `i`.
pub(crate) fn lower_formula(formula: &Formula, scope: &TemplateScope<'_>) -> Result<LoweredFormula, DslError> {
    let mut cx = Lowerer::new(formula.source(), &[], None, Some(scope));
    for (i, name) in scope.variables.iter().enumerate() {
        let id = cx.graph.arg(i as u16);
        cx.bind(name, Binding::Value(Value { id, ty: Ty::Num }));
    }
    for (i, name) in (0u16..).zip(scope.expressions) {
        cx.bind(
            name,
            Binding::Func {
                expr: i,
                wrt: Vec::new(),
            },
        );
    }
    for (i, (name, _)) in (0u16..).zip(scope.parameters) {
        cx.bind(name, Binding::Params(i));
    }

    let Some((value, span)) = cx.body(&formula.stmts)? else {
        return Err(cx.err("the formula has no value".to_string(), formula.span));
    };
    let (graph, roots) = cx.graph.extract(&[value.id]);
    if !varies(&graph, roots[0]) {
        return Err(cx.err(
            "the formula's value is the same for every row; it must use a feature or call a subexpression on one"
                .to_string(),
            span,
        ));
    }
    Ok(LoweredFormula {
        graph,
        root: roots[0],
        arities: cx.arities,
    })
}

/// Whether the value at `root` can differ between rows. Every other formula value is a scalar
/// in SymbolicRegression.jl, which rejects a scalar result.
fn varies(graph: &Graph, root: NodeId) -> bool {
    let mut row_dependent = Vec::with_capacity(graph.len());
    for node in graph.nodes() {
        let v = match node {
            Node::Const(_) => false,
            Node::Arg(_) => true,
            node => node.args().iter().any(|a| row_dependent[a.index()]),
        };
        row_dependent.push(v);
    }
    row_dependent[root.index()]
}

struct Lowerer<'a> {
    src: &'a str,
    graph: Graph,
    /// Parameters, locals, and template names; later entries shadow earlier ones.
    env: Vec<(String, Binding)>,
    customs: &'a [Lowered],
    self_name: Option<&'a str>,
    template: Option<&'a TemplateScope<'a>>,
    /// Template formulas only: the argument count of each subexpression's calls.
    arities: Vec<Option<u8>>,
}

impl<'a> Lowerer<'a> {
    fn new(
        src: &'a str,
        customs: &'a [Lowered],
        self_name: Option<&'a str>,
        template: Option<&'a TemplateScope<'a>>,
    ) -> Self {
        Self {
            src,
            graph: Graph::default(),
            env: Vec::new(),
            customs,
            self_name,
            template,
            arities: vec![None; template.map_or(0, |t| t.expressions.len())],
        }
    }

    fn err(&self, message: String, span: Span) -> DslError {
        DslError::new(message, span, self.src)
    }

    fn bind(&mut self, name: &str, binding: Binding) {
        self.env.push((name.to_string(), binding));
    }

    fn lookup(&self, name: &str) -> Option<&Binding> {
        self.env.iter().rev().find(|(n, _)| n == name).map(|(_, b)| b)
    }

    /// Lowers statements; the value is the last expression, assignment, or `return`.
    fn body(&mut self, stmts: &[Stmt]) -> Result<Option<(Value, Span)>, DslError> {
        let mut result = None;
        for stmt in stmts {
            match stmt {
                Stmt::Assign { name, value, .. } => {
                    if let Some(func) = self.func(value)? {
                        self.bind(&name.name, func);
                        result = None;
                        continue;
                    }
                    let v = self.expr(value)?;
                    self.bind(&name.name, Binding::Value(v));
                    result = Some((v, value.span));
                }
                Stmt::Return(e) => {
                    result = Some((self.expr(e)?, e.span));
                    break;
                }
                Stmt::Expr(e) => result = Some((self.expr(e)?, e.span)),
            }
        }
        Ok(result)
    }

    fn num(id: NodeId) -> Value {
        Value { id, ty: Ty::Num }
    }

    fn boolean(id: NodeId) -> Value {
        Value { id, ty: Ty::Bool }
    }

    /// Lowers `e`, which is used as a condition and so must be `Bool`.
    fn condition(&mut self, e: &Expr) -> Result<NodeId, DslError> {
        let v = self.expr(e)?;
        if v.ty != Ty::Bool {
            return Err(self.err("expected a Bool condition, found a number".to_string(), e.span));
        }
        Ok(v.id)
    }

    fn select(&mut self, cond: NodeId, then: Value, otherwise: Value) -> Value {
        let id = self.graph.op(Kernel::IfElse, vec![cond, then.id, otherwise.id]);
        let ty = if then.ty == Ty::Bool && otherwise.ty == Ty::Bool {
            Ty::Bool
        } else {
            Ty::Num
        };
        Value { id, ty }
    }

    fn expr(&mut self, e: &Expr) -> Result<Value, DslError> {
        match &e.kind {
            ExprKind::Num(v) => Ok(Self::num(self.graph.constant(*v))),
            ExprKind::Bool(b) => Ok(Self::boolean(self.graph.constant(if *b { 1.0 } else { 0.0 }))),
            ExprKind::Name(name) => self.name(name, e.span),
            ExprKind::Unary { op, arg } => match op {
                UnaryOp::Plus => Ok(Self::num(self.expr(arg)?.id)),
                UnaryOp::Neg => {
                    let a = self.expr(arg)?;
                    let id = match self.graph.const_value(a.id) {
                        Some(c) => self.graph.constant(-c),
                        None => self.graph.op(Kernel::Neg, vec![a.id]),
                    };
                    Ok(Self::num(id))
                }
                UnaryOp::Not => {
                    let a = self.condition(arg)?;
                    Ok(Self::boolean(self.graph.op(Kernel::LogicalNot, vec![a])))
                }
            },
            ExprKind::Binary { op, lhs, rhs } => self.binary(*op, lhs, rhs),
            ExprKind::Ternary { cond, then, otherwise } => {
                let c = self.condition(cond)?;
                let a = self.expr(then)?;
                let b = self.expr(otherwise)?;
                Ok(self.select(c, a, b))
            }
            ExprKind::Call { callee, args } => self.call(callee, args, e.span),
            ExprKind::Apply { func, args } => match self.func(func)? {
                Some(Binding::Func { expr, wrt }) => self.apply(expr, wrt, args, e.span),
                _ => Err(self.err(
                    "only subexpressions and their derivatives can be called".to_string(),
                    func.span,
                )),
            },
            ExprKind::Index { base, index } => self.index(base, index, e.span),
        }
    }

    /// Resolves `e` to a subexpression or one of its derivatives, if it denotes one.
    fn func(&mut self, e: &Expr) -> Result<Option<Binding>, DslError> {
        if self.template.is_none() {
            return Ok(None);
        }
        match &e.kind {
            ExprKind::Name(name) => Ok(self.lookup(name).filter(|b| matches!(b, Binding::Func { .. })).cloned()),
            ExprKind::Call { callee, args } if callee.name == "D" && self.lookup("D").is_none() => {
                if args.len() != 2 {
                    return Err(self.arity_error("D", "2", false, args.len(), e.span));
                }
                let Some(Binding::Func { expr, mut wrt }) = self.func(&args[0])? else {
                    return Err(self.err(
                        "the first argument of `D` must be a subexpression, as in `D(f, 1)`".to_string(),
                        args[0].span,
                    ));
                };
                let i = match args[1].kind {
                    ExprKind::Num(v) if v >= 1.0 && v.fract() == 0.0 && v <= f64::from(u16::MAX) => v as u16,
                    _ => {
                        return Err(self.err(
                            "the second argument of `D` must be a positive integer argument index".to_string(),
                            args[1].span,
                        ));
                    }
                };
                wrt.push(i - 1);
                Ok(Some(Binding::Func { expr, wrt }))
            }
            _ => Ok(None),
        }
    }

    fn apply(&mut self, expr: u16, wrt: Vec<u16>, args: &[Expr], span: Span) -> Result<Value, DslError> {
        let scope = self.template.expect("only template formulas bind subexpressions");
        let name = &scope.expressions[usize::from(expr)];
        let n = u8::try_from(args.len())
            .map_err(|_| self.err("subexpressions take at most 255 arguments".to_string(), span))?;
        match self.arities[usize::from(expr)] {
            Some(a) if a != n => {
                return Err(self.err(
                    format!("inconsistent number of arguments passed to `{name}`: {a} and {n}"),
                    span,
                ));
            }
            _ => self.arities[usize::from(expr)] = Some(n),
        }
        if let Some(&i) = wrt.iter().find(|&&i| i >= u16::from(n)) {
            return Err(self.err(
                format!(
                    "`D` differentiates `{name}` with respect to argument {}, but it takes {n}",
                    i + 1
                ),
                span,
            ));
        }
        let ids = self.args(args)?.iter().map(|v| v.id).collect();
        Ok(Self::num(self.graph.intern(Node::Call { expr, wrt, args: ids })))
    }

    fn index(&mut self, base: &Expr, index: &Expr, span: Span) -> Result<Value, DslError> {
        let ExprKind::Name(name) = &base.kind else {
            return Err(self.err("only parameter vectors can be indexed".to_string(), base.span));
        };
        let Some(Binding::Params(vector)) = self.lookup(name).cloned() else {
            return Err(self.err(format!("`{name}` is not a parameter vector"), base.span));
        };
        let len = self
            .template
            .expect("only template formulas bind parameters")
            .parameters[usize::from(vector)]
        .1;
        let i = self.expr(index)?;
        if let Some(c) = self.graph.const_value(i.id) {
            if c.fract() != 0.0 || c < 1.0 || c > len as f64 {
                return Err(self.err(
                    format!("index {c} is out of bounds for `{name}`, which has {len} entries"),
                    span,
                ));
            }
        }
        Ok(Self::num(self.graph.intern(Node::Param { vector, index: i.id })))
    }

    fn name(&mut self, name: &str, span: Span) -> Result<Value, DslError> {
        match self.lookup(name) {
            Some(Binding::Value(v)) => return Ok(*v),
            Some(Binding::Func { .. }) => {
                return Err(self.err(format!("`{name}` is a subexpression; call it, as in `{name}(x)`"), span));
            }
            Some(Binding::Params(_)) => {
                return Err(self.err(
                    format!("`{name}` is a parameter vector; index it, as in `{name}[1]`"),
                    span,
                ));
            }
            None => {}
        }
        let value = match name {
            "pi" | "π" => core::f64::consts::PI,
            "ℯ" => core::f64::consts::E,
            "Inf" => f64::INFINITY,
            "NaN" => f64::NAN,
            _ => return Err(self.err(format!("unknown name `{name}`"), span)),
        };
        Ok(Self::num(self.graph.constant(value)))
    }

    fn binary(&mut self, op: BinaryOp, lhs: &Expr, rhs: &Expr) -> Result<Value, DslError> {
        let kernel = match op {
            BinaryOp::And | BinaryOp::Or => {
                let a = self.condition(lhs)?;
                let b = self.condition(rhs)?;
                let kernel = if op == BinaryOp::And {
                    Kernel::LogicalAnd
                } else {
                    Kernel::LogicalOr
                };
                return Ok(Self::boolean(self.graph.op(kernel, vec![a, b])));
            }
            BinaryOp::Add => Kernel::Add,
            BinaryOp::Sub => Kernel::Sub,
            BinaryOp::Mul => Kernel::Mul,
            BinaryOp::Div => Kernel::Div,
            BinaryOp::Pow => Kernel::Pow,
            BinaryOp::Lt => Kernel::Less,
            BinaryOp::Le => Kernel::LessEqual,
            BinaryOp::Gt => Kernel::Greater,
            BinaryOp::Ge => Kernel::GreaterEqual,
            BinaryOp::Eq => Kernel::Equal,
            BinaryOp::Ne => Kernel::NotEqual,
        };
        let a = self.expr(lhs)?;
        let b = self.expr(rhs)?;
        let id = self.graph.op(kernel, vec![a.id, b.id]);
        Ok(Value {
            id,
            ty: if kernel.returns_bool() { Ty::Bool } else { Ty::Num },
        })
    }

    fn args(&mut self, args: &[Expr]) -> Result<Vec<Value>, DslError> {
        args.iter().map(|a| self.expr(a)).collect()
    }

    fn call(&mut self, callee: &Ident, args: &[Expr], span: Span) -> Result<Value, DslError> {
        let name = callee.name.as_str();
        let n = args.len();
        if self.self_name == Some(name) {
            return Err(self.err(format!("operator `{name}` cannot call itself"), span));
        }

        if let Some(Binding::Func { expr, wrt }) = self.lookup(name).cloned() {
            return self.apply(expr, wrt, args, span);
        }
        if name == "D" && self.template.is_some() && self.lookup("D").is_none() {
            return Err(self.err("`D(f, i)` is a function; call it, as in `D(f, 1)(x)`".to_string(), span));
        }

        if let Some(custom) = self.customs.iter().rev().find(|c| c.name.as_deref() == Some(name)) {
            if n != usize::from(custom.arity) {
                return Err(self.arity_error(name, &custom.arity.to_string(), custom.arity == 1, n, span));
            }
            let ids: Vec<NodeId> = self.args(args)?.iter().map(|v| v.id).collect();
            let id = self.graph.inline(&custom.graph, custom.root, &ids);
            return Ok(Value { id, ty: custom.ty });
        }

        match (name, n) {
            ("ifelse", 3) => {
                let c = self.condition(&args[0])?;
                let a = self.expr(&args[1])?;
                let b = self.expr(&args[2])?;
                return Ok(self.select(c, a, b));
            }
            ("ifelse", _) => return Err(self.arity_error(name, "3", false, n, span)),
            ("min" | "max", 2..) => {
                let kernel = if name == "min" { Kernel::Min } else { Kernel::Max };
                let values = self.args(args)?;
                let id = values[1..]
                    .iter()
                    .fold(values[0].id, |acc, v| self.graph.op(kernel, vec![acc, v.id]));
                return Ok(Self::num(id));
            }
            ("min" | "max", _) => return Err(self.arity_error(name, "2 or more", false, n, span)),
            ("atan", 2) => {
                let values = self.args(args)?;
                return Ok(Self::num(
                    self.graph.op(Kernel::Atan2, vec![values[0].id, values[1].id]),
                ));
            }
            _ => {}
        }

        // Called by name, comparison and logic kernels return 0.0 or 1.0, as SymbolicRegression.jl's
        // `greater` and `logical_and` do; only the operator syntax (`>`, `&&`) produces a Bool.
        if let Some(kernel) = Kernel::lookup(name, n) {
            let ids = self.args(args)?.iter().map(|v| v.id).collect();
            return Ok(Self::num(self.graph.op(kernel, ids)));
        }

        let mut arities = Kernel::arities_of(name);
        if name == "atan" {
            arities.push(2);
        }
        match arities.as_slice() {
            [] => Err(self.err(format!("unknown function `{name}`"), callee.span)),
            [one] => Err(self.arity_error(name, &one.to_string(), *one == 1, n, span)),
            many => {
                let list: Vec<String> = many.iter().map(u8::to_string).collect();
                Err(self.arity_error(name, &list.join(" or "), false, n, span))
            }
        }
    }

    fn arity_error(&self, name: &str, expected: &str, singular: bool, got: usize, span: Span) -> DslError {
        let noun = if singular { "argument" } else { "arguments" };
        self.err(format!("`{name}` takes {expected} {noun}, got {got}"), span)
    }
}
