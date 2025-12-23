use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use dynamic_expressions::{
    compile_plan, eval_grad_tree_array, eval_plan_array_into, eval_plan_array_into_cached,
    eval_tree_array_into, EvalContext, EvalOptions, GradContext, PNode, PostfixExpr, SubtreeCache,
};
use dynamic_expressions::operator_enum::presets::BuiltinOpsF64;
use dynamic_expressions::operator_registry::OpRegistry;
use ndarray::Array2;
use std::sync::Arc;

dynamic_expressions::opset! {
    pub struct ReadmeOps<f64>;
    ops {
        (1, Op1) { Cos, }
        (2, Op2) { Add, Sub, Mul, }
    }
}

fn var(feature: u16) -> PostfixExpr<f64, ReadmeOps, 2> {
    PostfixExpr::new(vec![PNode::Var { feature }], vec![], Default::default())
}

fn build_expr() -> PostfixExpr<f64, ReadmeOps, 2> {
    // x1 * cos(x2 - 3.2)
    use dynamic_expressions::math::cos;
    var(0) * cos(var(1) - 3.2)
}

fn update_consts(consts: &mut [f64], tick: u64) {
    let t = tick as f64 * 0.001;
    for (i, c) in consts.iter_mut().enumerate() {
        *c = (t + i as f64).sin() * 0.5;
    }
}

fn build_complex_expr(n_features: usize) -> PostfixExpr<f64, BuiltinOpsF64, 2> {
    let add = <BuiltinOpsF64 as OpRegistry>::lookup_with_arity("+", 2)
        .expect("missing add")
        .op
        .id;
    let sub = <BuiltinOpsF64 as OpRegistry>::lookup_with_arity("-", 2)
        .expect("missing sub")
        .op
        .id;
    let mul = <BuiltinOpsF64 as OpRegistry>::lookup_with_arity("*", 2)
        .expect("missing mul")
        .op
        .id;
    let sin = <BuiltinOpsF64 as OpRegistry>::lookup_with_arity("sin", 1)
        .expect("missing sin")
        .op
        .id;
    let cos = <BuiltinOpsF64 as OpRegistry>::lookup_with_arity("cos", 1)
        .expect("missing cos")
        .op
        .id;

    let mut nodes = Vec::new();

    // Constant-free subtree: a deep fold over variables with alternating unary ops.
    nodes.push(PNode::Var { feature: 0 });
    for i in 1..n_features {
        nodes.push(PNode::Var { feature: i as u16 });
        let op = if i % 2 == 0 { add } else { mul };
        nodes.push(PNode::Op { arity: 2, op });
        let uop = if i % 2 == 0 { sin } else { cos };
        nodes.push(PNode::Op { arity: 1, op: uop });
    }

    // Constant-dependent subtree.
    nodes.push(PNode::Const { idx: 0 });
    nodes.push(PNode::Var { feature: 0 });
    nodes.push(PNode::Op { arity: 2, op: mul });
    nodes.push(PNode::Const { idx: 1 });
    nodes.push(PNode::Op { arity: 2, op: add });
    nodes.push(PNode::Const { idx: 2 });
    nodes.push(PNode::Op { arity: 2, op: sub });
    nodes.push(PNode::Op { arity: 1, op: sin });

    // Combine subtrees.
    nodes.push(PNode::Op { arity: 2, op: add });

    PostfixExpr::new(nodes, vec![0.1, -0.2, 0.3], Default::default())
}

fn eval_naive(expr: &PostfixExpr<f64, ReadmeOps, 2>, x: ndarray::ArrayView2<'_, f64>) -> Vec<f64> {
    let n_rows = x.nrows();
    let mut out = vec![0.0f64; n_rows];

    for row in 0..n_rows {
        let x1 = x[(row, 0)];
        let x2 = x[(row, 1)];
        let mut stack: Vec<f64> = Vec::with_capacity(expr.nodes.len());
        for node in &expr.nodes {
            match *node {
                PNode::Var { feature: 0 } => stack.push(x1),
                PNode::Var { feature: 1 } => stack.push(x2),
                PNode::Var { feature } => panic!("unexpected feature {}", feature),
                PNode::Const { idx } => stack.push(expr.consts[usize::from(idx)]),
                PNode::Op { arity: 1, op } => {
                    let a = stack.pop().unwrap();
                    match op {
                        x if x == (Op1::Cos as u16) => stack.push(a.cos()),
                        _ => panic!("unknown unary op {}", op),
                    }
                }
                PNode::Op { arity: 2, op } => {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    match op {
                        x if x == (Op2::Add as u16) => stack.push(a + b),
                        x if x == (Op2::Sub as u16) => stack.push(a - b),
                        x if x == (Op2::Mul as u16) => stack.push(a * b),
                        _ => panic!("unknown binary op {}", op),
                    }
                }
                PNode::Op { arity, op } => panic!("unsupported arity {} op {}", arity, op),
            }
        }
        out[row] = stack.pop().unwrap();
    }

    out
}

fn bench_readme_like(c: &mut Criterion) {
    let n_features = 2usize;
    let n_rows = 100usize;
    let mut data = vec![0.0f64; n_features * n_rows];
    for row in 0..n_rows {
        for feature in 0..n_features {
            data[row * n_features + feature] = (row as f64 + 1.0) * (feature as f64 + 1.0) * 0.001;
        }
    }
    let x = Array2::from_shape_vec((n_rows, n_features), data).unwrap();
    let x_data = x.as_slice().unwrap();
    let x_view = x.view();
    let expr = build_expr();
    let opts = EvalOptions {
        check_finite: false,
        early_exit: false,
    };

    let mut group = c.benchmark_group("readme_like");
    group.bench_with_input(
        BenchmarkId::new("naive_stack_eval", n_rows),
        &n_rows,
        |b, _| b.iter(|| eval_naive(&expr, x_view)),
    );
    group.bench_with_input(BenchmarkId::new("hardcoded", n_rows), &n_rows, |b, _| {
        b.iter(|| {
            let mut out = vec![0.0f64; n_rows];
            for row in 0..n_rows {
                let x1 = x_data[row * n_features];
                let x2 = x_data[row * n_features + 1];
                out[row] = x1 * (x2 - 3.2).cos();
            }
            out
        })
    });
    group.bench_with_input(
        BenchmarkId::new("hardcoded_optimized", n_rows),
        &n_rows,
        |b, _| {
            let mut out = vec![0.0f64; n_rows];
            b.iter(|| {
                for row in 0..n_rows {
                    let x1 = x_data[row * n_features];
                    let x2 = x_data[row * n_features + 1];
                    out[row] = x1 * (x2 - 3.2).cos();
                }
            })
        },
    );
    group.bench_with_input(BenchmarkId::new("dynamic_eval", n_rows), &n_rows, |b, _| {
        let mut out = vec![0.0f64; n_rows];
        let mut ctx = EvalContext::<f64, 2>::new(n_rows);
        b.iter(|| {
            let _ok =
                eval_tree_array_into::<f64, ReadmeOps, 2>(&mut out, &expr, x_view, &mut ctx, &opts);
        })
    });
    group.bench_with_input(
        BenchmarkId::new("dynamic_eval_mutate_op", n_rows),
        &n_rows,
        |b, _| {
            let mut out = vec![0.0f64; n_rows];
            let mut ctx = EvalContext::<f64, 2>::new(n_rows);
            let mut which: u16 = 0;
            b.iter(|| {
                let mut ex = expr.clone();
                // Mutate the root binary op among {Add, Sub, Mul}.
                let op = match which % 3 {
                    0 => Op2::Add as u16,
                    1 => Op2::Sub as u16,
                    _ => Op2::Mul as u16,
                };
                which = which.wrapping_add(1);
                if let Some(PNode::Op {
                    arity: 2,
                    op: ref mut id,
                }) = ex.nodes.last_mut()
                {
                    *id = op;
                }
                let _ok = eval_tree_array_into::<f64, ReadmeOps, 2>(
                    &mut out, &ex, x_view, &mut ctx, &opts,
                );
            })
        },
    );
    group.finish();

    // Derivatives (matches README's "Derivatives" section).
    let mut group = c.benchmark_group("readme_like_derivatives");
    group.bench_with_input(
        BenchmarkId::new("grad_variables", n_rows),
        &n_rows,
        |b, _| {
            let mut gctx = GradContext::<f64, 2>::new(n_rows);
            b.iter(|| {
                let _ = eval_grad_tree_array::<f64, ReadmeOps, 2>(
                    &expr, x_view, true, &mut gctx, &opts,
                );
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("grad_mutate_op", n_rows),
        &n_rows,
        |b, _| {
            let mut gctx = GradContext::<f64, 2>::new(n_rows);
            let mut which: u16 = 0;
            b.iter(|| {
                let mut ex = expr.clone();
                // Mutate the root binary op among {Add, Sub, Mul}.
                let op = match which % 3 {
                    0 => Op2::Add as u16,
                    1 => Op2::Sub as u16,
                    _ => Op2::Mul as u16,
                };
                which = which.wrapping_add(1);
                if let Some(PNode::Op {
                    arity: 2,
                    op: ref mut id,
                }) = ex.nodes.last_mut()
                {
                    *id = op;
                }

                let _ =
                    eval_grad_tree_array::<f64, ReadmeOps, 2>(&ex, x_view, true, &mut gctx, &opts);
            })
        },
    );
    group.bench_with_input(
        BenchmarkId::new("grad_constants", n_rows),
        &n_rows,
        |b, _| {
            let mut gctx = GradContext::<f64, 2>::new(n_rows);
            b.iter(|| {
                let _ = eval_grad_tree_array::<f64, ReadmeOps, 2>(
                    &expr, x_view, false, &mut gctx, &opts,
                );
            })
        },
    );
    group.finish();
}

fn bench_subtree_cache_complex(c: &mut Criterion) {
    let n_features = 8usize;
    let n_rows = 4096usize;
    let mut data = vec![0.0f64; n_features * n_rows];
    for row in 0..n_rows {
        for feature in 0..n_features {
            let idx = row * n_features + feature;
            data[idx] = ((row as f64 + 1.0) * (feature as f64 + 1.0)).sin() * 0.1;
        }
    }
    let x = Arc::new(Array2::from_shape_vec((n_rows, n_features), data).unwrap());

    let expr = Arc::new(build_complex_expr(n_features));
    let plan = Arc::new(compile_plan::<2>(
        &expr.nodes,
        n_features,
        expr.consts.len(),
    ));
    let opts = EvalOptions {
        check_finite: true,
        early_exit: false,
    };

    let mut group = c.benchmark_group("subtree_cache_complex");
    group.bench_function("uncached_eval", {
        let x = Arc::clone(&x);
        let expr = Arc::clone(&expr);
        let plan = Arc::clone(&plan);
        move |b| {
            let x_view = x.view();
            let plan = (*plan).clone();
            let mut expr = (*expr).clone();
            let mut scratch = Vec::new();
            let mut out = vec![0.0f64; n_rows];
            let mut tick = 0u64;
            b.iter(|| {
                tick = tick.wrapping_add(1);
                update_consts(&mut expr.consts, tick);
                let _ok = eval_plan_array_into(
                    &mut out,
                    &plan,
                    &expr,
                    x_view,
                    &mut scratch,
                    &opts,
                );
            })
        }
    });

    group.bench_function("cached_eval", {
        let x = Arc::clone(&x);
        let expr = Arc::clone(&expr);
        let plan = Arc::clone(&plan);
        move |b| {
            let x_view = x.view();
            let plan = (*plan).clone();
            let mut expr = (*expr).clone();
            let mut scratch = Vec::new();
            let mut out = vec![0.0f64; n_rows];
            let mut cache = SubtreeCache::new(n_rows, 64 * 1024 * 1024);
            let dataset_key = 0xC0FFEEu64;
            let _ = eval_plan_array_into_cached(
                &mut out,
                &plan,
                &expr,
                x_view,
                &mut scratch,
                &opts,
                &mut cache,
                dataset_key,
            );
            let mut tick = 0u64;
            b.iter(|| {
                tick = tick.wrapping_add(1);
                update_consts(&mut expr.consts, tick);
                let _ok = eval_plan_array_into_cached(
                    &mut out,
                    &plan,
                    &expr,
                    x_view,
                    &mut scratch,
                    &opts,
                    &mut cache,
                    dataset_key,
                );
            })
        }
    });

    group.finish();
}

criterion_group!(benches, bench_readme_like, bench_subtree_cache_complex);
criterion_main!(benches);
