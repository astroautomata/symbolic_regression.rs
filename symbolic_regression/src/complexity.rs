use std::collections::HashMap;

use dynamic_expressions::node::PNode;
use dynamic_expressions::{OpId, OperatorSet};
use num_traits::Float;

use crate::options::Options;

pub(crate) fn compute_custom_complexity_checked<T: Float, Ops: OperatorSet, const D: usize>(
    nodes: &[PNode],
    options: &Options<T, Ops, D>,
    op_arg_limits: Option<&HashMap<OpId, [Option<u16>; D]>>,
) -> Option<usize> {
    let mut st: Vec<f32> = Vec::with_capacity(nodes.len().min(256));

    for n in nodes {
        match *n {
            PNode::Var { feature } => {
                let idx = feature as usize;
                let c = options
                    .variable_complexities
                    .as_ref()
                    .and_then(|v| v.get(idx))
                    .copied()
                    .unwrap_or(options.complexity_of_variables);
                st.push(c);
            }
            PNode::Const { .. } => st.push(options.complexity_of_constants),
            PNode::Op { arity, op } => {
                let a = arity as usize;
                let mut child = [0.0f32; D];
                for j in (0..a).rev() {
                    child[j] = st.pop().unwrap_or(0.0);
                }

                let oid = OpId { arity, id: op };
                if let Some(limits) = op_arg_limits.and_then(|m| m.get(&oid)) {
                    for j in 0..a {
                        let Some(lim) = limits[j] else {
                            continue;
                        };
                        if rounded(child[j]) > usize::from(lim) {
                            return None;
                        }
                    }
                }

                let base = options
                    .operator_complexity_overrides
                    .get(&oid)
                    .copied()
                    .unwrap_or_else(|| f32::from(options.operators.set.complexity(oid)));
                st.push(base + child.iter().sum::<f32>());
            }
        }
    }

    if st.len() != 1 {
        return None;
    }
    Some(rounded(st[0]))
}

/// Julia's `round(Int, x)`: nearest integer, ties to even.
fn rounded(complexity: f32) -> usize {
    complexity.round_ties_even() as usize
}

pub fn compute_complexity<T: Float, Ops: OperatorSet, const D: usize>(
    nodes: &[PNode],
    options: &Options<T, Ops, D>,
) -> usize {
    if options.uses_default_complexity() {
        return nodes.len();
    }

    compute_custom_complexity_checked(nodes, options, None).unwrap_or(0)
}
