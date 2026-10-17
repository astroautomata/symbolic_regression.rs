use std::hash::{Hash, Hasher};

use rustc_hash::FxHasher;

use crate::node::{PNode, Src};

#[derive(Clone, Debug)]
pub struct EvalPlan<const D: usize> {
    pub instrs: Vec<Instr<D>>,
    pub n_slots: usize,
    pub root: Src,
    pub hash: u64,
}

#[derive(Copy, Clone, Debug)]
pub struct Instr<const D: usize> {
    pub arity: u8,
    pub op: u16,
    pub args: [Src; D],
    pub dst: u16,
    /// Marks the first occurrence of each variable or constant leaf.
    pub leaf_args: [bool; D],
}

pub fn compile_plan<const D: usize>(nodes: &[PNode], n_features: usize, n_consts: usize) -> EvalPlan<D> {
    assert!(
        n_features <= (u16::MAX as usize),
        "n_features={} exceeds u16::MAX",
        n_features
    );
    assert!(
        n_consts <= (u16::MAX as usize),
        "n_consts={} exceeds u16::MAX",
        n_consts
    );
    let n_features_u16 = n_features as u16;
    let n_consts_u16 = n_consts as u16;

    let mut stack: Vec<(Src, bool)> = Vec::new();
    let n_ops = nodes.iter().filter(|n| matches!(n, PNode::Op { .. })).count();
    let mut instrs: Vec<Instr<D>> = Vec::with_capacity(n_ops);

    let mut free_slots: Vec<u16> = Vec::new();
    let mut next_slot: u16 = 0;
    let mut max_slot: u16 = 0;

    let alloc_slot = |free_slots: &mut Vec<u16>, next_slot: &mut u16, max_slot: &mut u16| -> u16 {
        let s = free_slots.pop().unwrap_or_else(|| {
            let s = *next_slot;
            *next_slot += 1;
            s
        });
        *max_slot = (*max_slot).max(s + 1);
        s
    };

    for (index, node) in nodes.iter().enumerate() {
        match *node {
            PNode::Var { feature } => {
                assert!(feature < n_features_u16, "Var index out of bounds");
                stack.push((Src::Var(feature), !nodes[..index].contains(node)));
            }
            PNode::Const { idx } => {
                assert!(idx < n_consts_u16, "Const index out of bounds");
                stack.push((Src::Const(idx), !nodes[..index].contains(node)));
            }
            PNode::Op { arity, op } => {
                let arity_u8 = arity;
                let arity = arity as usize;
                assert!(arity >= 1 && arity <= D, "Unsupported arity {} (D={})", arity, D);

                let mut args: [Src; D] = [Src::Const(0); D];
                let mut leaf_args = [false; D];
                for j in (0..arity).rev() {
                    (args[j], leaf_args[j]) = stack.pop().expect("stack underflow (op)");
                }

                let dst = alloc_slot(&mut free_slots, &mut next_slot, &mut max_slot);
                instrs.push(Instr {
                    arity: arity_u8,
                    op,
                    args,
                    dst,
                    leaf_args,
                });

                for src in args.iter().take(arity) {
                    if let Src::Slot(s) = *src {
                        free_slots.push(s);
                    }
                }

                stack.push((Src::Slot(dst), false));
            }
        }
    }

    assert_eq!(stack.len(), 1, "Postfix did not reduce to a single root");
    let root = stack.pop().unwrap().0;
    let n_slots = max_slot as usize;
    let mut hasher = FxHasher::default();
    nodes.hash(&mut hasher);
    let hash = hasher.finish();
    EvalPlan {
        instrs,
        n_slots,
        root,
        hash,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaf_flags_preserve_sources_and_mark_each_leaf_once() {
        let plan = compile_plan::<2>(
            &[
                PNode::Var { feature: 0 },
                PNode::Var { feature: 0 },
                PNode::Const { idx: 0 },
                PNode::Op { arity: 2, op: 0 },
                PNode::Op { arity: 2, op: 0 },
            ],
            1,
            1,
        );
        assert_eq!(plan.instrs[0].args, [Src::Var(0), Src::Const(0)]);
        assert_eq!(plan.instrs[0].leaf_args, [false, true]);
        assert_eq!(plan.instrs[1].args[0], Src::Var(0));
        assert_eq!(plan.instrs[1].leaf_args, [true, false]);
        assert_eq!(plan.instrs.capacity(), 2);
        let copy = plan.clone();
        assert_eq!(copy.instrs[1].leaf_args, [true, false]);
    }

    #[test]
    fn root_leaves_need_no_instruction_allocation() {
        for (node, features, constants, root) in [
            (PNode::Var { feature: 0 }, 1, 0, Src::Var(0)),
            (PNode::Const { idx: 0 }, 0, 1, Src::Const(0)),
        ] {
            let plan = compile_plan::<0>(&[node], features, constants);
            assert!(plan.instrs.is_empty());
            assert_eq!(plan.instrs.capacity(), 0);
            assert_eq!(plan.root, root);
        }
    }
}
