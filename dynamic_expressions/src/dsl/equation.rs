use num_traits::Float;

use super::ast::{BinaryOp, Expr, ExprKind, UnaryOp};
use super::{DslError, Span, parse_expr};
use crate::expression::{Metadata, PostfixExpr};
use crate::node::PNode;
use crate::select::Operators;
use crate::traits::{OpId, OperatorSet};

/// Parses an equation string into a tree, as DynamicExpressions' `parse_expression` reads one for
/// SymbolicRegression.jl's `guesses`: each call or infix operator must be one of the selected
/// `operators` taking that many arguments, each name one of `variable_names`, and each number a
/// constant. A `-` written directly before a number is part of the number (`-1.0`), as in Julia;
/// `a + b + c` nests to the left. Comparisons call the operators that print as them (`x1 < x2`
/// calls `less`); `&&`, `||`, and `a ? b : c` are not calls, so they are rejected as in Julia.
///
/// ```
/// use dynamic_expressions::dsl::parse_expression;
/// use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
/// use dynamic_expressions::{Operators, PostfixExpr};
///
/// let operators: Operators<_, 2> = BuiltinOpsF64::new().from_names(["+", "*", "cos"]).unwrap();
/// let names = ["x1".to_string(), "x2".to_string()];
/// let tree: PostfixExpr<f64, _, 2> = parse_expression("2.0*x1 + cos(x2)", &operators, &names).unwrap();
/// assert_eq!(tree.consts, vec![2.0]);
/// let err = parse_expression::<f64, _, 2>("sin(x1)", &operators, &names).unwrap_err();
/// assert_eq!(err.message, "`sin` is not one of the search's operators");
/// ```
pub fn parse_expression<T: Float, Ops: OperatorSet, const D: usize>(
    src: &str,
    operators: &Operators<Ops, D>,
    variable_names: &[String],
) -> Result<PostfixExpr<T, Ops, D>, DslError> {
    let expr = parse_expr(src)?;
    let mut lowering = Lowering {
        src,
        operators,
        variable_names,
        nodes: Vec::new(),
        consts: Vec::new(),
    };
    lowering.lower(&expr)?;
    Ok(PostfixExpr::new(lowering.nodes, lowering.consts, Metadata::default()))
}

struct Lowering<'a, T, Ops, const D: usize> {
    src: &'a str,
    operators: &'a Operators<Ops, D>,
    variable_names: &'a [String],
    nodes: Vec<PNode>,
    consts: Vec<T>,
}

impl<T: Float, Ops: OperatorSet, const D: usize> Lowering<'_, T, Ops, D> {
    fn err(&self, message: impl Into<String>, span: Span) -> DslError {
        DslError::new(message, span, self.src)
    }

    fn lower(&mut self, e: &Expr) -> Result<(), DslError> {
        match &e.kind {
            ExprKind::Num(v) => self.constant(*v, e.span),
            ExprKind::Bool(b) => self.constant(if *b { 1.0 } else { 0.0 }, e.span),
            ExprKind::Name(name) => {
                let Some(feature) = self.variable_names.iter().position(|v| v == name) else {
                    return Err(self.err(
                        format!(
                            "`{name}` is not a variable; the variables are {}",
                            self.variable_names.join(", ")
                        ),
                        e.span,
                    ));
                };
                let feature = u16::try_from(feature).map_err(|_| self.err("too many variables", e.span))?;
                self.nodes.push(PNode::Var { feature });
                Ok(())
            }
            ExprKind::Unary { op, arg } => {
                if let (UnaryOp::Neg | UnaryOp::Plus, ExprKind::Num(v)) = (op, &arg.kind) {
                    if self.is_signed_literal(e.span, arg.span) {
                        return self.constant(if *op == UnaryOp::Neg { -v } else { *v }, e.span);
                    }
                }
                let name = match op {
                    UnaryOp::Neg => "-",
                    UnaryOp::Plus => "+",
                    UnaryOp::Not => "!",
                };
                self.call(name, &[arg.as_ref()], e.span)
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let name = match op {
                    BinaryOp::Add => "+",
                    BinaryOp::Sub => "-",
                    BinaryOp::Mul => "*",
                    BinaryOp::Div => "/",
                    BinaryOp::Pow => "^",
                    BinaryOp::Lt => "<",
                    BinaryOp::Le => "<=",
                    BinaryOp::Gt => ">",
                    BinaryOp::Ge => ">=",
                    BinaryOp::Eq => "==",
                    BinaryOp::Ne => "!=",
                    BinaryOp::And | BinaryOp::Or => {
                        return Err(self.err(
                            "`&&` and `||` are not function calls; use an operator such as `logical_and`",
                            e.span,
                        ));
                    }
                };
                self.call(name, &[lhs.as_ref(), rhs.as_ref()], e.span)
            }
            ExprKind::Call { callee, args } => {
                let args: Vec<&Expr> = args.iter().collect();
                self.call(&callee.name, &args, e.span)
            }
            ExprKind::Ternary { .. } => Err(self.err(
                "`a ? b : c` is not a function call; use an operator such as `cond`",
                e.span,
            )),
            ExprKind::Index { .. } | ExprKind::Apply { .. } => {
                Err(self.err("indexing and `D` are only allowed in template formulas", e.span))
            }
        }
    }

    /// Julia reads a sign written directly before a number literal as part of it: `-1.0` is a
    /// constant, while `- 1.0`, `-(1.0)`, and `-x` call `-`.
    fn is_signed_literal(&self, sign: Span, number: Span) -> bool {
        number.start == sign.start + 1 && self.src[number.start..].starts_with(|c: char| c.is_ascii_digit() || c == '.')
    }

    fn constant(&mut self, v: f64, span: Span) -> Result<(), DslError> {
        let idx = u16::try_from(self.consts.len()).map_err(|_| self.err("too many constants", span))?;
        self.consts
            .push(T::from(v).ok_or_else(|| self.err("the number does not fit the data type", span))?);
        self.nodes.push(PNode::Const { idx });
        Ok(())
    }

    fn call(&mut self, name: &str, args: &[&Expr], span: Span) -> Result<(), DslError> {
        let op = self.operator(name, args.len(), span)?;
        for arg in args {
            self.lower(arg)?;
        }
        self.nodes.push(PNode::Op {
            arity: op.arity,
            op: op.id,
        });
        Ok(())
    }

    /// The selected operator `name` names with `arity` arguments, by its name, printed form, infix
    /// symbol, or an alias. A custom operator wins over a built-in of the same name, as in
    /// operator selection.
    fn operator(&self, name: &str, arity: usize, span: Span) -> Result<OpId, DslError> {
        let set = &self.operators.set;
        let names = |op: &&OpId| {
            set.meta(**op).is_some_and(|m| {
                m.name == name || m.display == name || m.infix == Some(name) || m.aliases.contains(&name)
            })
        };
        let selected = || self.operators.ops_by_arity.iter().flatten().filter(names);
        let mut matches: Vec<OpId> = selected()
            .filter(|op| usize::from(op.arity) == arity)
            .copied()
            .collect();
        if matches.len() > 1 {
            matches.retain(|&op| set.is_custom(op));
        }
        match matches[..] {
            [op] => Ok(op),
            [] => {
                let arities: Vec<String> = selected().map(|op| op.arity.to_string()).collect();
                Err(self.err(
                    if arities.is_empty() {
                        format!("`{name}` is not one of the search's operators")
                    } else {
                        format!(
                            "the operator `{name}` takes {} arguments, not {arity}",
                            arities.join(" or ")
                        )
                    },
                    span,
                ))
            }
            _ => Err(self.err(
                format!("`{name}` names more than one operator taking {arity} arguments"),
                span,
            )),
        }
    }
}
