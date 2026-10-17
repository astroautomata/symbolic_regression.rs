use num_traits::Float;

pub trait Operator<T: Float, const A: usize> {
    const NAME: &'static str;
    const DISPLAY: &'static str = Self::NAME;
    const INFIX: Option<&'static str> = None;
    const ALIASES: &'static [&'static str] = &[];
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

/// Operator sets build `OpMeta<'static>` tables for built-ins; runtime operators borrow their
/// tokens from the operator-set value.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct OpMeta<'a> {
    pub arity: u8,
    pub id: u16,
    pub name: &'a str,
    pub display: &'a str,
    pub infix: Option<&'a str>,
    pub aliases: &'a [&'a str],
    pub commutative: bool,
    pub associative: bool,
    pub complexity: u16,
}

/// Associate a fixed arity with an operator marker type.
///
/// Rust can't (in general) infer `A` from `Tag: Operator<T, A>`, so we attach arity directly.
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
    Ambiguous { token: String, candidates: Vec<String> },
}

/// Operator-set abstraction.
///
/// - "What ops exist?" -> `ops_with_arity`
/// - "What are their tokens / properties?" -> `meta` (plus default helpers)
/// - "How do I eval/diff/grad?" -> `eval/diff/grad` (dispatch to kernels)
pub trait OperatorSet: Sized {
    type T: Float;

    /// Guarantees unit complexity for every registered operator, including runtime operators.
    const UNIT_COMPLEXITY: bool = false;

    fn max_arity(&self) -> u8;

    fn ops_with_arity(&self, arity: u8) -> impl Iterator<Item = u16> + '_;

    /// The only required metadata entrypoint.
    fn meta(&self, op: OpId) -> Option<OpMeta<'_>>;

    /// Operators defined at runtime. They shadow built-ins with the same token in `lookup`.
    fn is_custom(&self, op: OpId) -> bool;

    // ---- Convenience defaults derived from meta() ----

    #[inline]
    fn name(&self, op: OpId) -> &str {
        self.meta(op).map(|m| m.name).unwrap_or("unknown_op")
    }

    #[inline]
    fn display(&self, op: OpId) -> &str {
        self.meta(op).map(|m| m.display).unwrap_or("unknown_op")
    }

    #[inline]
    fn infix(&self, op: OpId) -> Option<&str> {
        self.meta(op).and_then(|m| m.infix)
    }

    #[inline]
    fn commutative(&self, op: OpId) -> bool {
        self.meta(op).is_some_and(|m| m.commutative)
    }

    #[inline]
    fn associative(&self, op: OpId) -> bool {
        self.meta(op).is_some_and(|m| m.associative)
    }

    #[inline]
    fn complexity(&self, op: OpId) -> u16 {
        self.meta(op).map(|m| m.complexity).unwrap_or(0)
    }

    // ---- Kernel dispatch ----

    fn eval(&self, op: OpId, ctx: crate::dispatch::EvalKernelCtx<'_, '_, Self::T>) -> bool;
    fn diff(&self, op: OpId, ctx: crate::dispatch::DiffKernelCtx<'_, '_, Self::T>) -> bool;
    fn grad(&self, op: OpId, ctx: crate::dispatch::GradKernelCtx<'_, '_, Self::T>) -> bool;

    // ---- Token lookup ----

    #[inline]
    fn matches_token(&self, op: OpId, tok: &str) -> bool {
        let t = tok.trim();
        let Some(meta) = self.meta(op) else {
            return false;
        };

        t.eq_ignore_ascii_case(meta.name)
            || t == meta.display
            || meta.infix.is_some_and(|s| t == s)
            || meta.aliases.iter().any(|a| t.eq_ignore_ascii_case(a))
    }

    #[inline]
    fn for_each_op(&self, mut f: impl FnMut(OpId)) {
        for arity in 1..=self.max_arity() {
            for id in self.ops_with_arity(arity) {
                f(OpId { arity, id });
            }
        }
    }

    fn lookup_all(&self, token: &str) -> Vec<OpId> {
        let mut out = Vec::new();
        self.for_each_op(|op| {
            if self.matches_token(op, token) {
                out.push(op);
            }
        });
        out
    }

    fn lookup(&self, token: &str) -> Result<OpId, LookupError> {
        resolve_lookup(self, token, self.lookup_all(token))
    }

    fn lookup_with_arity(&self, token: &str, arity: u8) -> Result<OpId, LookupError> {
        let matches = self
            .ops_with_arity(arity)
            .map(|id| OpId { arity, id })
            .filter(|&op| self.matches_token(op, token))
            .collect();
        resolve_lookup(self, token, matches)
    }
}

fn resolve_lookup<Ops: OperatorSet>(ops: &Ops, token: &str, mut matches: Vec<OpId>) -> Result<OpId, LookupError> {
    if matches.len() > 1 && matches.iter().any(|&op| ops.is_custom(op)) {
        matches.retain(|&op| ops.is_custom(op));
    }
    match matches.as_slice() {
        [] => Err(LookupError::Unknown(token.trim().to_string())),
        [single] => Ok(*single),
        _ => Err(LookupError::Ambiguous {
            token: token.trim().to_string(),
            candidates: matches.iter().map(|&op| ops.name(op).to_string()).collect(),
        }),
    }
}
