use super::ir::Ty;
use super::lower::{Lowered, lower};
use super::program::{DslFloat, ScalarFn};
use super::{DslError, parse_definition};
use crate::custom::CustomTable;
use crate::dispatch::{DiffKernelCtx, EvalKernelCtx, GradKernelCtx};
use crate::traits::OpMeta;

/// Custom operators compiled from DSL definitions. Local id `i` is the `i`-th definition.
#[derive(Clone, Debug)]
pub struct CustomOps<T> {
    lowered: Vec<Lowered>,
    funcs: Vec<ScalarFn<T>>,
}

impl<T> Default for CustomOps<T> {
    fn default() -> Self {
        Self {
            lowered: Vec::new(),
            funcs: Vec::new(),
        }
    }
}

/// The operator a definition added.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Defined {
    pub name: String,
    pub arity: u8,
}

impl<T: DslFloat> CustomOps<T> {
    /// Parses and compiles one named definition, such as `inv(x) = 1/x`. Its body may call
    /// built-in kernels and operators defined earlier in this table.
    pub fn define(&mut self, src: &str) -> Result<Defined, DslError> {
        let def = parse_definition(src)?;
        let Some(name) = def.name.clone() else {
            return Err(DslError::new(
                "custom operators need a name, as in `f(x) = ...`",
                def.span,
                src,
            ));
        };
        if self
            .lowered
            .iter()
            .any(|l| l.name.as_deref() == Some(name.name.as_str()))
        {
            return Err(DslError::new(
                format!("operator `{}` is already defined", name.name),
                name.span,
                src,
            ));
        }
        if def.params.is_empty() {
            return Err(DslError::new(
                format!("operator `{}` needs at least one argument", name.name),
                def.span,
                src,
            ));
        }
        let lowered = lower(&def, &self.lowered)?;
        if lowered.ty == Ty::Bool {
            return Err(DslError::new(
                format!(
                    "operator `{}` returns Bool; custom operators must return a number",
                    name.name
                ),
                lowered.root_span,
                src,
            ));
        }
        let defined = Defined {
            name: name.name,
            arity: lowered.arity,
        };
        self.funcs.push(ScalarFn::from_lowered(&lowered));
        self.lowered.push(lowered);
        Ok(defined)
    }

    pub fn len(&self) -> usize {
        self.lowered.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lowered.is_empty()
    }
}

/// Whether an operator token is DSL source (`"inv(x) = 1/x"`, `"function f(x) ... end"`,
/// `"x -> x^2"`) rather than an operator name or symbol (`"cos"`, `"+"`, `"<="`). No built-in name
/// or symbol contains `(` or `->`.
pub fn is_definition(token: &str) -> bool {
    token.contains('(') || token.contains("->")
}

/// Splits PySR-style operator tokens into built-in names (`"cos"`, `"+"`) and definitions
/// (`"inv(x) = 1/x"`). Returns the compiled definitions and the token names to select, in token
/// order; a definition contributes the name it defines.
pub fn parse_operator_tokens<T, I>(tokens: I) -> Result<(CustomOps<T>, Vec<String>), DslError>
where
    T: DslFloat,
    I: IntoIterator,
    I::Item: AsRef<str>,
{
    let mut custom = CustomOps::default();
    let mut names = Vec::new();
    for token in tokens {
        let token = token.as_ref();
        if is_definition(token) {
            names.push(custom.define(token)?.name);
        } else {
            names.push(token.trim().to_string());
        }
    }
    Ok((custom, names))
}

impl<T: DslFloat> CustomTable<T> for CustomOps<T> {
    const UNIT_COMPLEXITY: bool = true;

    fn max_arity(&self) -> u8 {
        self.lowered.iter().map(|l| l.arity).max().unwrap_or(0)
    }

    fn ids_with_arity(&self, arity: u8) -> impl Iterator<Item = u16> + '_ {
        (0u16..)
            .zip(&self.lowered)
            .filter(move |(_, l)| l.arity == arity)
            .map(|(id, _)| id)
    }

    fn meta(&self, local: u16) -> Option<OpMeta<'_>> {
        let lowered = self.lowered.get(usize::from(local))?;
        let name = lowered.name.as_deref()?;
        Some(OpMeta {
            arity: lowered.arity,
            id: local,
            name,
            display: name,
            infix: None,
            aliases: &[],
            commutative: false,
            associative: false,
            complexity: 1,
        })
    }

    fn eval(&self, local: u16, ctx: EvalKernelCtx<'_, '_, T>) -> bool {
        self.funcs[usize::from(local)].eval(ctx.out, ctx.args, ctx.opts)
    }

    fn diff(&self, local: u16, ctx: DiffKernelCtx<'_, '_, T>) -> bool {
        self.funcs[usize::from(local)].diff(ctx)
    }

    fn grad(&self, local: u16, ctx: GradKernelCtx<'_, '_, T>) -> bool {
        self.funcs[usize::from(local)].grad(ctx)
    }
}
