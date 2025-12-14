use crate::operators::scalar::OpId;

pub trait OpNames {
    fn op_name(op: OpId) -> &'static str;

    fn op_pretty_name(op: OpId) -> &'static str {
        Self::op_name(op)
    }
}
