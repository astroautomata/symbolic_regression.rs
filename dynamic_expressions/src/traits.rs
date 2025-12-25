use num_traits::Float;

pub trait Operator<T: Float, const A: usize> {
    const NAME: &'static str;
    const DISPLAY: &'static str = Self::NAME;
    const INFIX: Option<&'static str> = None;
    const COMMUTATIVE: bool = false;
    const ASSOCIATIVE: bool = false;
    const COMPLEXITY: u16 = 1;

    fn eval(args: &[T; A]) -> T;
    fn partial(args: &[T; A], idx: usize) -> T;
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct OpId {
    pub arity: u8,
    pub id: u16,
}

/// Associate a fixed arity with an operator marker type.
///
/// Rust can't (in general) recover `A` from `Tag: Operator<T, A>`, so we attach arity directly to
/// the marker type for ergonomic typed ID lookup (e.g. `<Ops as HasOp<Add>>::ID`).
pub trait OpTag {
    const ARITY: u8;
}

pub trait HasOp<Tag: OpTag> {
    const ID: u16;

    #[inline]
    fn op_id() -> OpId {
        OpId {
            arity: Tag::ARITY,
            id: Self::ID,
        }
    }
}

#[derive(Debug, Clone)]
pub enum LookupError {
    Unknown(String),
    Ambiguous {
        token: String,
        candidates: Vec<&'static str>,
    },
}

/// An active set of operators.
///
/// Design goals:
/// - **Static:** the set is known at compile time (no runtime registration).
/// - **Inlinable:** implementations are expected to use `match (arity, id)`-style dispatch to retrieve operator
///   metadata and (elsewhere) to dispatch evaluation.
/// - **Single entry point:** this trait is the place to ask "what operators exist?" and "what are their
///   names/tokens/metadata?".
pub trait OperatorSet: Sized {
    type T: Float;

    /// Maximum arity supported by this set. (Typically small, e.g. 1–3.)
    const MAX_ARITY: u8;

    /// List the active operator IDs for a given arity.
    fn ops_with_arity(arity: u8) -> &'static [u16];

    /// Canonical operator name. Implementations typically forward to `<Op as Operator<T, A>>::NAME`
    /// for the operator corresponding to `(op.arity, op.id)`.
    fn name(op: OpId) -> &'static str;

    /// Display name (for pretty-printing). Typically `<Op as Operator<T, A>>::DISPLAY`.
    fn display(op: OpId) -> &'static str;

    /// Optional infix token, e.g. `"+"` or `"-"`. Typically `<Op as Operator<T, A>>::INFIX`.
    fn infix(op: OpId) -> Option<&'static str>;
    fn commutative(op: OpId) -> bool;
    fn associative(op: OpId) -> bool;
    fn complexity(op: OpId) -> u16;

    /// Evaluate `op` elementwise over rows, writing into `ctx.out`.
    ///
    /// Returns `true` if evaluation completed without encountering non-finite values (subject to
    /// `EvalOptions` in `ctx.opts`).
    fn eval(op: OpId, ctx: crate::dispatch::EvalKernelCtx<'_, '_, Self::T>) -> bool;

    /// Compute value and directional derivative elementwise, writing into `ctx.out_val` and
    /// `ctx.out_der`.
    fn diff(op: OpId, ctx: crate::dispatch::DiffKernelCtx<'_, '_, Self::T>) -> bool;

    /// Compute value and gradients elementwise, writing into `ctx.out_val` and `ctx.out_grad`.
    fn grad(op: OpId, ctx: crate::dispatch::GradKernelCtx<'_, '_, Self::T>) -> bool;

    #[inline]
    fn matches_token(op: OpId, tok: &str) -> bool {
        let t = tok.trim();
        t.eq_ignore_ascii_case(Self::name(op)) || t == Self::display(op) || Self::infix(op).is_some_and(|s| t == s)
    }

    #[inline]
    fn for_each_op(mut f: impl FnMut(OpId)) {
        for arity in 1..=Self::MAX_ARITY {
            for &id in Self::ops_with_arity(arity) {
                f(OpId { arity, id });
            }
        }
    }

    fn lookup_all(token: &str) -> Vec<OpId> {
        let mut out = Vec::new();
        Self::for_each_op(|op| {
            if Self::matches_token(op, token) {
                out.push(op);
            }
        });
        out
    }

    fn lookup(token: &str) -> Result<OpId, LookupError> {
        let matches = Self::lookup_all(token);
        match matches.as_slice() {
            [] => Err(LookupError::Unknown(token.trim().to_string())),
            [single] => Ok(*single),
            _ => {
                // Common CLI ambiguity: "-" can be unary neg or binary sub. Prefer binary sub.
                let t = token.trim();
                if t == "-" {
                    if let Some(op) = matches
                        .iter()
                        .copied()
                        .find(|op| op.arity == 2 && Self::name(*op).eq_ignore_ascii_case("sub"))
                    {
                        return Ok(op);
                    }
                }

                Err(LookupError::Ambiguous {
                    token: t.to_string(),
                    candidates: matches.iter().map(|op| Self::name(*op)).collect(),
                })
            }
        }
    }

    fn lookup_with_arity(token: &str, arity: u8) -> Result<OpId, LookupError> {
        let mut matches = Vec::new();
        for &id in Self::ops_with_arity(arity) {
            let op = OpId { arity, id };
            if Self::matches_token(op, token) {
                matches.push(op);
            }
        }
        match matches.as_slice() {
            [] => Err(LookupError::Unknown(token.trim().to_string())),
            [single] => Ok(*single),
            _ => Err(LookupError::Ambiguous {
                token: token.trim().to_string(),
                candidates: matches.iter().map(|op| Self::name(*op)).collect(),
            }),
        }
    }
}
