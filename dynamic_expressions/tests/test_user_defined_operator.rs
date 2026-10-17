use dynamic_expressions::expression::Metadata;
use dynamic_expressions::node::PNode;
use dynamic_expressions::operator_enum::builtin;
use dynamic_expressions::operator_enum::builtin::*;
use dynamic_expressions::{EvalOptions, HasOp, OpId, OperatorSet, PostfixExpr, eval_tree_array};
use ndarray::Array2;

dynamic_expressions::op!(Square for<T> {
    eval: |[x]| { x * x },
    partial: |[x], _idx| { (T::one() + T::one()) * x },
});

dynamic_expressions::opset! {
    UserOps for f64 { Sin, Square, Add }
}

fn var(feature: u16) -> PostfixExpr<f64, UserOps, 2> {
    PostfixExpr::new(vec![PNode::Var { feature }], vec![], Metadata::default())
}

fn square(x: PostfixExpr<f64, UserOps, 2>) -> PostfixExpr<f64, UserOps, 2> {
    dynamic_expressions::expression_algebra::__apply_postfix::<f64, UserOps, 2, 1>(<UserOps as HasOp<Square>>::ID, [x])
}

#[test]
fn opset_can_mix_builtin_and_user_defined_ops() {
    let square_op = <UserOps as HasOp<Square>>::op_id();
    assert_eq!(
        square_op,
        OpId {
            arity: 1,
            id: <UserOps as HasOp<Square>>::ID
        }
    );
    assert_eq!(UserOps::new().lookup("square").unwrap(), square_op);

    let sin = <UserOps as HasOp<builtin::Sin>>::op_id();
    assert_eq!(UserOps::new().lookup("sin").unwrap(), sin);

    let x = Array2::from_shape_vec((1, 4), vec![-2.0, -1.0, 0.5, 3.0]).unwrap();
    let expr = square(var(0));

    let opts = EvalOptions {
        check_finite: true,
        early_exit: true,
    };
    let (out, ok) = eval_tree_array::<f64, UserOps, 2>(&expr, &UserOps::new(), x.view(), &opts);
    assert!(ok);
    assert_eq!(out, vec![4.0, 1.0, 0.25, 9.0]);
}
