//! A small Julia-compatible scalar language for custom operators, losses, early-stop conditions,
//! and template formulas.

pub mod ast;
#[doc(hidden)]
pub mod lexer;
mod parser;

use core::fmt;

pub use self::ast::{BinaryOp, DefKind, Definition, Expr, ExprKind, Formula, Ident, Param, Stmt, TypeVar, UnaryOp};
pub use self::parser::{parse_definition, parse_formula};

/// Byte offsets into the source string, `start..end`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// The smallest span covering both `self` and `other`.
    pub fn to(self, other: Span) -> Span {
        Span::new(self.start.min(other.start), self.end.max(other.end))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DslError {
    pub message: String,
    pub span: Span,
    source: String,
}

impl DslError {
    pub fn new(message: impl Into<String>, span: Span, source: &str) -> Self {
        Self {
            message: message.into(),
            span,
            source: source.to_string(),
        }
    }
}

impl fmt::Display for DslError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let start = self.span.start.min(self.source.len());
        let line_start = self.source[..start].rfind('\n').map_or(0, |i| i + 1);
        let line_end = self.source[start..].find('\n').map_or(self.source.len(), |i| start + i);
        let line = &self.source[line_start..line_end];
        let line_no = self.source[..line_start].matches('\n').count() + 1;
        let before = &self.source[line_start..start];
        let column = before.chars().count() + 1;
        let underlined = self.source[start..self.span.end.clamp(start, line_end)].chars().count();

        writeln!(f, "{} at line {line_no}, column {column}", self.message)?;
        writeln!(f, "{line}")?;
        let pad: String = before.chars().map(|c| if c == '\t' { '\t' } else { ' ' }).collect();
        write!(f, "{pad}{}", "^".repeat(underlined.max(1)))
    }
}

impl std::error::Error for DslError {}
