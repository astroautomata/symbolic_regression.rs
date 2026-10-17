use std::collections::HashSet;
use std::fmt;

use crate::traits::{LookupError, OpId, OperatorSet};

/// An operator set together with the operators selected from it. Keeping both in one value means
/// the selected ids always refer to the set that evaluates and prints them.
#[derive(Clone, Debug)]
pub struct Operators<Ops, const D: usize> {
    pub set: Ops,
    pub ops_by_arity: [Vec<OpId>; D],
}

#[derive(Debug, Clone)]
pub enum OperatorSelectError {
    Lookup(LookupError),
    ArityTooLarge { token: String, arity: u8, max_arity: usize },
    Duplicate(String),
    Empty,
}

impl fmt::Display for OperatorSelectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OperatorSelectError::Lookup(e) => write!(f, "{e:?}"),
            OperatorSelectError::ArityTooLarge {
                token,
                arity,
                max_arity,
            } => write!(
                f,
                "operator token {token:?} has arity={arity} which exceeds D={max_arity}"
            ),
            OperatorSelectError::Duplicate(tok) => write!(f, "duplicate operator token {tok:?}"),
            OperatorSelectError::Empty => write!(f, "no operators provided"),
        }
    }
}

impl std::error::Error for OperatorSelectError {}

impl<Ops, const D: usize> Operators<Ops, D> {
    /// No operators selected yet from `set`.
    pub fn new(set: Ops) -> Self {
        Self {
            set,
            ops_by_arity: std::array::from_fn(|_| Vec::new()),
        }
    }

    pub fn push(&mut self, op: OpId) {
        let arity = op.arity as usize;
        assert!((1..=D).contains(&arity));
        self.ops_by_arity[arity - 1].push(op);
    }

    pub fn nops(&self, arity: usize) -> usize {
        self.ops_by_arity[arity - 1].len()
    }

    pub fn from_names<I>(set: Ops, names: I) -> Result<Self, OperatorSelectError>
    where
        Ops: OperatorSet,
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        let mut iter = names.into_iter().peekable();
        if iter.peek().is_none() {
            return Err(OperatorSelectError::Empty);
        }

        let mut out = Self::new(set);
        let mut seen: HashSet<(u8, u16)> = HashSet::new();

        for tok in iter {
            let tok = tok.as_ref();
            let op = out.set.lookup(tok).map_err(OperatorSelectError::Lookup)?;
            if (op.arity as usize) > D {
                return Err(OperatorSelectError::ArityTooLarge {
                    token: tok.to_string(),
                    arity: op.arity,
                    max_arity: D,
                });
            }
            let key = (op.arity, op.id);
            if !seen.insert(key) {
                return Err(OperatorSelectError::Duplicate(tok.to_string()));
            }
            out.push(op);
        }

        Ok(out)
    }
}

impl<Ops: Default, const D: usize> Default for Operators<Ops, D> {
    fn default() -> Self {
        Self::new(Ops::default())
    }
}

#[cfg(test)]
mod tests {
    use crate::operator_enum::builtin;
    use crate::operator_enum::presets::BuiltinOpsF64;
    use crate::{HasOp, Operators};

    crate::op!(Square for f64 {
        eval: |[x]| { x * x },
        partial: |[x], _idx| { 2.0 * x },
    });

    crate::opset! {
        V2Ops for f64 { Square }
    }

    #[test]
    fn from_names_requires_disambiguation_for_dash() {
        let ops: Result<Operators<_, 3>, _> = BuiltinOpsF64::new().from_names(["-"]);
        assert!(ops.is_err());

        let ops: Operators<_, 3> = BuiltinOpsF64::new().from_names(["neg", "sub"]).unwrap();
        assert_eq!(ops.nops(1), 1);
        assert_eq!(ops.ops_by_arity[0][0], <BuiltinOpsF64 as HasOp<builtin::Neg>>::op_id());
        assert_eq!(ops.nops(2), 1);
        assert_eq!(ops.ops_by_arity[1][0], <BuiltinOpsF64 as HasOp<builtin::Sub>>::op_id());
    }

    #[test]
    fn v2_opset_dsl_builds_operator_set() {
        let ops: Operators<_, 3> = V2Ops::new().from_names(["square"]).unwrap();
        assert_eq!(ops.nops(1), 1);
    }
}
