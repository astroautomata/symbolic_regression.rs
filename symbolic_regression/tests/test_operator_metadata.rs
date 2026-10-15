//! Operator metadata complexity counts toward expression complexity.

use dynamic_expressions::expression_algebra::__apply_postfix;
use dynamic_expressions::operator_enum::builtin::Add;
use dynamic_expressions::{Metadata, OperatorSet, PNode, PostfixExpr};
use symbolic_regression::{Options, compute_complexity};

dynamic_expressions::op!(Heavy for f64 {
    complexity: 3,
    eval: |[x]| { x },
    partial: |[_x], _idx| { 1.0 },
});

dynamic_expressions::opset! {
    pub HeavyOps for f64 { Heavy, Add }
}

#[test]
fn complexity_uses_operator_metadata() {
    let operator_set = HeavyOps::new();
    let options = Options::<f64, HeavyOps, 2> {
        operators: operator_set.from_names(["heavy", "add"]).unwrap(),
        ..Default::default()
    };
    let heavy = options.operators.set.lookup("heavy").unwrap();
    let leaf = PostfixExpr::<f64, HeavyOps, 2>::new(vec![PNode::Var { feature: 0 }], vec![], Metadata::default());
    let tree = __apply_postfix::<f64, HeavyOps, 2, 1>(heavy.id, [leaf]);
    assert_eq!(compute_complexity(&tree.nodes, &options), 3 + 1);
}

#[derive(Clone, Debug, Default)]
struct WeightedTable {
    set: HeavyOps,
    complexity: u16,
}

impl dynamic_expressions::custom::CustomTable<f64> for WeightedTable {
    fn max_arity(&self) -> u8 {
        1
    }
    fn ids_with_arity(&self, arity: u8) -> impl Iterator<Item = u16> + '_ {
        (arity == 1).then_some(0).into_iter()
    }
    fn meta(&self, local: u16) -> Option<dynamic_expressions::OpMeta<'_>> {
        let mut meta = self.set.meta(dynamic_expressions::OpId { arity: 1, id: local })?;
        meta.name = "runtime_heavy";
        meta.display = meta.name;
        meta.complexity = self.complexity;
        Some(meta)
    }
    fn eval(&self, local: u16, ctx: dynamic_expressions::dispatch::EvalKernelCtx<'_, '_, f64>) -> bool {
        self.set.eval(dynamic_expressions::OpId { arity: 1, id: local }, ctx)
    }
    fn diff(&self, local: u16, ctx: dynamic_expressions::dispatch::DiffKernelCtx<'_, '_, f64>) -> bool {
        self.set.diff(dynamic_expressions::OpId { arity: 1, id: local }, ctx)
    }
    fn grad(&self, local: u16, ctx: dynamic_expressions::dispatch::GradKernelCtx<'_, '_, f64>) -> bool {
        self.set.grad(dynamic_expressions::OpId { arity: 1, id: local }, ctx)
    }
}

#[test]
fn unit_complexity_guarantee_includes_marker_and_runtime_metadata() {
    use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
    assert!(std::hint::black_box(<BuiltinOpsF64 as OperatorSet>::UNIT_COMPLEXITY));
    assert!(!std::hint::black_box(<HeavyOps as OperatorSet>::UNIT_COMPLEXITY));
    type Ops = BuiltinOpsF64<WeightedTable>;
    assert!(!std::hint::black_box(Ops::UNIT_COMPLEXITY));
    let set = BuiltinOpsF64::new().with_custom(WeightedTable {
        set: HeavyOps::new(),
        complexity: 3,
    });
    let mut opts = Options::<f64, Ops, 2> {
        operators: set.from_names(["runtime_heavy"]).unwrap(),
        ..Default::default()
    };
    let op = opts.operators.set.lookup("runtime_heavy").unwrap();
    let nodes = [PNode::Var { feature: 0 }, PNode::Op { arity: 1, op: op.id }];
    assert!(!opts.uses_default_complexity());
    assert_eq!(compute_complexity(&nodes, &opts), 4);
    opts.operators.set.custom.complexity = 1;
    assert!(opts.uses_default_complexity());
    assert_eq!(compute_complexity(&nodes, &opts), 2);
    opts.operators.set.custom.complexity = 5;
    assert!(!opts.uses_default_complexity());
    assert_eq!(compute_complexity(&nodes, &opts), 6);
    opts.operator_complexity_overrides.insert(op, 2);
    opts.variable_complexities = Some(vec![4]);
    assert_eq!(compute_complexity(&nodes, &opts), 6);
    let mut plain = Options::<f64, BuiltinOpsF64, 2> {
        complexity_of_constants: 3,
        ..Default::default()
    };
    assert_eq!(compute_complexity(&[PNode::Const { idx: 0 }], &plain), 3);
    plain.complexity_of_variables = 4;
    assert_eq!(compute_complexity(&[PNode::Var { feature: 0 }], &plain), 4);
}
