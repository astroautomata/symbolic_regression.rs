use dynamic_expressions::custom::CustomTable;
use dynamic_expressions::dispatch::{DiffKernelCtx, EvalKernelCtx, GradKernelCtx};
use dynamic_expressions::evaluate::kernels::{diff_nary, eval_nary, grad_nary};
use dynamic_expressions::operator_enum::builtin;
use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
use dynamic_expressions::strings::{StringTreeOptions, string_tree};
use dynamic_expressions::{
    DiffContext, EvalOptions, GradContext, HasOp, OpId, OpMeta, OperatorSet, PNode, PostfixExpr, eval_diff_tree_array,
    eval_grad_tree_array, eval_tree_array,
};
use ndarray::Array2;

dynamic_expressions::op!(Twice for f64 {
    eval: |[x]| { 2.0 * x },
    partial: |[_x], _idx| { 2.0 },
});

/// One runtime unary operator computing `2x`, under a configurable name.
#[derive(Clone, Debug)]
struct ScaleTable {
    name: &'static str,
}

impl Default for ScaleTable {
    fn default() -> Self {
        Self { name: "twice" }
    }
}

impl CustomTable<f64> for ScaleTable {
    fn max_arity(&self) -> u8 {
        1
    }

    fn ids_with_arity(&self, arity: u8) -> impl Iterator<Item = u16> + '_ {
        (arity == 1).then_some(0).into_iter()
    }

    fn meta(&self, local: u16) -> Option<OpMeta<'_>> {
        (local == 0).then_some(OpMeta {
            arity: 1,
            id: 0,
            name: self.name,
            display: self.name,
            infix: None,
            aliases: &[],
            commutative: false,
            associative: false,
            complexity: 1,
        })
    }

    fn eval(&self, local: u16, ctx: EvalKernelCtx<'_, '_, f64>) -> bool {
        assert_eq!(local, 0);
        eval_nary::<1, f64, Twice>(ctx.out, ctx.args, ctx.opts)
    }

    fn diff(&self, local: u16, ctx: DiffKernelCtx<'_, '_, f64>) -> bool {
        assert_eq!(local, 0);
        diff_nary::<1, f64, Twice>(ctx.out_val, ctx.out_der, ctx.args, ctx.dargs, ctx.opts)
    }

    fn grad(&self, local: u16, ctx: GradKernelCtx<'_, '_, f64>) -> bool {
        assert_eq!(local, 0);
        grad_nary::<1, f64, Twice>(ctx)
    }
}

type Ops = BuiltinOpsF64<ScaleTable>;
const D: usize = 3;

fn ops() -> Ops {
    BuiltinOpsF64::new().with_custom(ScaleTable::default())
}

fn n_builtins() -> u16 {
    let builtin = BuiltinOpsF64::new();
    (1..=builtin.max_arity())
        .map(|a| builtin.ops_with_arity(a).count() as u16)
        .sum()
}

fn twice_id(ops: &Ops) -> OpId {
    ops.lookup("twice").unwrap()
}

/// `twice(x0 * c0)` with `c0 = 3`.
fn twice_of_scaled_x0(ops: &Ops) -> PostfixExpr<f64, Ops, D> {
    let twice = twice_id(ops);
    PostfixExpr::new(
        vec![
            PNode::Var { feature: 0 },
            PNode::Const { idx: 0 },
            PNode::Op {
                arity: 2,
                op: <Ops as HasOp<builtin::Mul>>::ID,
            },
            PNode::Op { arity: 1, op: twice.id },
        ],
        vec![3.0],
        Default::default(),
    )
}

fn x_data() -> Array2<f64> {
    Array2::from_shape_vec((1, 4), vec![-1.5, 0.0, 0.25, 2.0]).unwrap()
}

#[test]
fn custom_op_gets_the_first_id_after_builtins() {
    let ops = ops();
    let twice = twice_id(&ops);
    assert_eq!(
        twice,
        OpId {
            arity: 1,
            id: n_builtins()
        }
    );
    assert!(ops.is_custom(twice));
    assert!(!ops.is_custom(BuiltinOpsF64::new().lookup("sin").unwrap()));
    assert!(!ops.is_custom(OpId { arity: 2, id: twice.id }));
}

#[test]
fn custom_op_meta_and_arity_listing() {
    let ops = ops();
    let twice = twice_id(&ops);
    assert_eq!(
        ops.meta(twice),
        Some(OpMeta {
            arity: 1,
            id: twice.id,
            name: "twice",
            display: "twice",
            infix: None,
            aliases: &[],
            commutative: false,
            associative: false,
            complexity: 1,
        })
    );
    assert_eq!(ops.name(twice), "twice");
    assert_eq!(ops.complexity(twice), 1);
    assert_eq!(ops.meta(OpId { arity: 2, id: twice.id }), None);
    assert_eq!(
        ops.meta(OpId {
            arity: 1,
            id: twice.id + 1
        }),
        None
    );

    assert!(ops.ops_with_arity(1).any(|id| id == twice.id));
    assert!(!ops.ops_with_arity(2).any(|id| id == twice.id));
    assert_eq!(
        ops.ops_with_arity(1).count(),
        BuiltinOpsF64::new().ops_with_arity(1).count() + 1
    );
    assert_eq!(ops.max_arity(), BuiltinOpsF64::new().max_arity());
}

#[test]
fn custom_op_evaluates_through_fallback_arm() {
    let ops = ops();
    let expr = twice_of_scaled_x0(&ops);
    let x = x_data();
    let opts = EvalOptions::default();

    let (y, ok) = eval_tree_array(&expr, &ops, x.view(), &opts);
    assert!(ok);
    assert_eq!(y, vec![-9.0, 0.0, 1.5, 12.0]);
}

#[test]
fn custom_op_differentiates_through_fallback_arm() {
    let ops = ops();
    let expr = twice_of_scaled_x0(&ops);
    let x = x_data();
    let n_rows = x.ncols();
    let opts = EvalOptions::default();

    let mut dctx = DiffContext::<f64, D>::new(n_rows);
    let (y, dy, ok) = eval_diff_tree_array(&expr, &ops, x.view(), 0, &mut dctx, &opts);
    assert!(ok);
    assert_eq!(y, vec![-9.0, 0.0, 1.5, 12.0]);
    assert_eq!(dy, vec![6.0; n_rows]);

    let mut gctx = GradContext::<f64, D>::new(n_rows);
    let (_y, grad_x, ok) = eval_grad_tree_array(&expr, &ops, x.view(), true, &mut gctx, &opts);
    assert!(ok);
    assert_eq!(grad_x.n_dir, 1);
    assert_eq!(grad_x.data, vec![6.0; n_rows]);

    let (_y, grad_c, ok) = eval_grad_tree_array(&expr, &ops, x.view(), false, &mut gctx, &opts);
    assert!(ok);
    assert_eq!(grad_c.n_dir, 1);
    assert_eq!(grad_c.data, vec![-3.0, 0.0, 0.5, 4.0]);
}

#[test]
fn custom_op_dispatches_diff_kernel() {
    let ops = ops();
    let twice = twice_id(&ops);
    let x = [1.0, -2.0];
    let dx = [0.5, 3.0];
    let mut out_val = [0.0; 2];
    let mut out_der = [0.0; 2];
    let ok = ops.diff(
        twice,
        DiffKernelCtx {
            out_val: &mut out_val,
            out_der: &mut out_der,
            args: &[dynamic_expressions::dispatch::SrcRef::Slice(&x)],
            dargs: &[dynamic_expressions::dispatch::SrcRef::Slice(&dx)],
            opts: &EvalOptions::default(),
        },
    );
    assert!(ok);
    assert_eq!(out_val, [2.0, -4.0]);
    assert_eq!(out_der, [1.0, 6.0]);
}

#[test]
fn custom_op_prints_by_name() {
    let ops = ops();
    let twice = twice_id(&ops);
    let expr: PostfixExpr<f64, Ops, D> = PostfixExpr::new(
        vec![PNode::Var { feature: 0 }, PNode::Op { arity: 1, op: twice.id }],
        vec![],
        Default::default(),
    );
    assert_eq!(string_tree(&expr, &ops, StringTreeOptions::default()), "twice(x0)");
    let pretty = StringTreeOptions {
        pretty: true,
        ..Default::default()
    };
    assert_eq!(string_tree(&expr, &ops, pretty), "twice(x0)");
}

#[test]
fn custom_op_shadows_builtin_with_same_token() {
    let builtin = BuiltinOpsF64::new();
    let builtin_inv = builtin.lookup("inv").unwrap();
    assert!(!builtin.is_custom(builtin_inv));

    let ops = BuiltinOpsF64::new().with_custom(ScaleTable { name: "inv" });
    assert_eq!(ops.lookup_all("inv").len(), 2);

    let inv = ops.lookup("inv").unwrap();
    assert_eq!(
        inv,
        OpId {
            arity: 1,
            id: n_builtins()
        }
    );
    assert!(ops.is_custom(inv));
    assert_eq!(ops.lookup_with_arity("inv", 1).unwrap(), inv);
    assert_ne!(inv, builtin_inv);
}

#[test]
fn opset_without_custom_operators_is_zero_sized() {
    assert_eq!(size_of::<BuiltinOpsF64>(), 0);
    assert_eq!(
        size_of::<dynamic_expressions::operator_enum::presets::BuiltinOpsF32>(),
        0
    );
}
