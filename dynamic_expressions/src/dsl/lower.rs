use super::ast::{BinaryOp, Definition, Expr, ExprKind, Ident, Stmt, UnaryOp};
use super::ir::{Graph, NodeId, Ty};
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

#[derive(Copy, Clone, Debug)]
struct Value {
    id: NodeId,
    ty: Ty,
}

/// Lowers `def` to IR. Calls to `customs` (earlier custom operators) are inlined.
pub fn lower(def: &Definition, customs: &[Lowered]) -> Result<Lowered, DslError> {
    let mut cx = Lowerer {
        src: def.source(),
        graph: Graph::default(),
        env: Vec::new(),
        customs,
        self_name: def.name.as_ref().map(|n| n.name.as_str()),
    };
    let arity = u8::try_from(def.params.len())
        .map_err(|_| cx.err("operators take at most 255 arguments".to_string(), def.span))?;
    for (i, param) in def.params.iter().enumerate() {
        let id = cx.graph.arg(i as u16);
        cx.env.push((param.name.name.clone(), Value { id, ty: Ty::Num }));
    }

    let mut result = None;
    for stmt in &def.body {
        match stmt {
            Stmt::Assign { name, value, .. } => {
                let v = cx.expr(value)?;
                cx.env.push((name.name.clone(), v));
                result = Some((v, value.span));
            }
            Stmt::Return(e) => {
                result = Some((cx.expr(e)?, e.span));
                break;
            }
            Stmt::Expr(e) => result = Some((cx.expr(e)?, e.span)),
        }
    }
    let Some((value, root_span)) = result else {
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

struct Lowerer<'a> {
    src: &'a str,
    graph: Graph,
    /// Parameters and locals; later entries shadow earlier ones.
    env: Vec<(String, Value)>,
    customs: &'a [Lowered],
    self_name: Option<&'a str>,
}

impl Lowerer<'_> {
    fn err(&self, message: String, span: Span) -> DslError {
        DslError::new(message, span, self.src)
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
            ExprKind::Index { .. } => {
                Err(self.err("indexing is only allowed in template formulas".to_string(), e.span))
            }
        }
    }

    fn name(&mut self, name: &str, span: Span) -> Result<Value, DslError> {
        if let Some((_, v)) = self.env.iter().rev().find(|(n, _)| n == name) {
            return Ok(*v);
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
