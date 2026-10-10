use super::Span;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ident {
    pub name: String,
    pub span: Span,
}

/// A parameter with its optional `::Type` annotation. Annotations are checked and then ignored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Param {
    pub name: Ident,
    pub ty: Option<Ident>,
}

/// One entry of a `where` clause: `T` or `T<:Real`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypeVar {
    pub name: Ident,
    pub bound: Option<Ident>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DefKind {
    /// `f(x) = expr`
    Named,
    /// `(x, y) -> expr` or `x -> expr`
    Lambda,
    /// `function f(x) ... end`
    Function,
}

/// A parsed scalar function: a custom operator, loss, or early-stop condition.
#[derive(Clone, Debug, PartialEq)]
pub struct Definition {
    pub kind: DefKind,
    /// `None` for lambdas.
    pub name: Option<Ident>,
    pub params: Vec<Param>,
    pub return_type: Option<Ident>,
    pub type_vars: Vec<TypeVar>,
    pub body: Vec<Stmt>,
    pub span: Span,
    pub(crate) source: String,
}

impl Definition {
    pub fn source(&self) -> &str {
        &self.source
    }
}

/// A parsed template combine formula: statements whose last one is the value.
#[derive(Clone, Debug, PartialEq)]
pub struct Formula {
    pub stmts: Vec<Stmt>,
    pub span: Span,
    pub(crate) source: String,
}

impl Formula {
    pub fn source(&self) -> &str {
        &self.source
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    Assign { local: bool, name: Ident, value: Expr },
    Return(Expr),
    Expr(Expr),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
    /// Levels of nodes from this one down to its deepest leaf. The parser bounds it, so every
    /// recursive walk of the tree (lowering, cloning, dropping) has bounded stack depth.
    pub(crate) height: u32,
}

impl Expr {
    pub(crate) fn new(kind: ExprKind, span: Span) -> Self {
        let below = match &kind {
            ExprKind::Num(_) | ExprKind::Bool(_) | ExprKind::Name(_) => 0,
            ExprKind::Unary { arg, .. } => arg.height,
            ExprKind::Binary { lhs, rhs, .. } => lhs.height.max(rhs.height),
            ExprKind::Ternary { cond, then, otherwise } => cond.height.max(then.height).max(otherwise.height),
            ExprKind::Call { args, .. } => args.iter().map(|a| a.height).max().unwrap_or(0),
            ExprKind::Index { base, index } => base.height.max(index.height),
            ExprKind::Apply { func, args } => args.iter().map(|a| a.height).fold(func.height, u32::max),
        };
        Self {
            kind,
            span,
            height: below + 1,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExprKind {
    Num(f64),
    Bool(bool),
    Name(String),
    Unary {
        op: UnaryOp,
        arg: Box<Expr>,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Ternary {
        cond: Box<Expr>,
        then: Box<Expr>,
        otherwise: Box<Expr>,
    },
    Call {
        callee: Ident,
        args: Vec<Expr>,
    },
    /// `p[i]`; only template formulas allow it.
    Index {
        base: Box<Expr>,
        index: Box<Expr>,
    },
    /// `D(f, 1)(x)`: a call of a function-valued expression; only template formulas allow it.
    Apply {
        func: Box<Expr>,
        args: Vec<Expr>,
    },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Plus,
    Not,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    And,
    Or,
}
