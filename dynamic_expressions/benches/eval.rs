use criterion::{criterion_group, criterion_main};
use dynamic_expressions::evaluate::EvalOptions;
use dynamic_expressions::expression::PostfixExpr;
use dynamic_expressions::node::PNode;
use dynamic_expressions::operator_enum::builtin::*;
use dynamic_expressions::{OperatorSet, opset};
use ndarray::Array2;
use num_traits::Float;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const N_FEATURES: usize = 5;
const TREE_SIZE: usize = 20;
const N_TREES: usize = 100;
const ROW_COUNTS: [usize; 4] = [64, 256, 1000, 10000];

const BENCH_OP_NAMES: [&str; 6] = ["cos", "exp", "add", "sub", "mul", "div"];

const BUILTIN_OP_NAMES: [&str; 40] = [
    "abs", "abs2", "acos", "acosh", "asin", "asinh", "atan", "atanh", "cbrt", "cos", "cosh", "cot", "csc", "exp",
    "exp2", "expm1", "identity", "inv", "log", "log1p", "log2", "log10", "neg", "sec", "sign", "sin", "sinh", "sqrt",
    "tan", "tanh", "add", "atan2", "div", "max", "min", "mul", "pow", "sub", "clamp", "fma",
];

opset! {
    pub BenchOpsF32 for f32 { Cos, Exp, Add, Sub, Mul, Div }
}

opset! {
    pub BenchOpsF64 for f64 { Cos, Exp, Add, Sub, Mul, Div }
}

fn random_leaf<T: Float, R: Rng>(rng: &mut R, n_features: usize, consts: &mut Vec<T>) -> PNode {
    if rng.random_bool(0.5) {
        let val: T = T::from(rng.random_range(-2.0..2.0)).unwrap();
        let idx: u16 = consts.len().try_into().expect("too many constants");
        consts.push(val);
        PNode::Const { idx }
    } else {
        let feature: u16 = rng
            .random_range(0..n_features)
            .try_into()
            .expect("feature index overflow");
        PNode::Var { feature }
    }
}

fn all_ops_by_arity<Ops: OperatorSet, const D: usize>(ops: &Ops) -> [Vec<u16>; D] {
    core::array::from_fn(|arity_minus_one| {
        let arity = (arity_minus_one + 1) as u8;
        ops.ops_with_arity(arity).collect()
    })
}

/// Operators picked by name, so adding operators to an opset does not change the workload.
fn ops_by_arity_from_names<Ops: OperatorSet, const D: usize>(ops: &Ops, names: &[&str]) -> [Vec<u16>; D] {
    let mut by_arity: [Vec<u16>; D] = core::array::from_fn(|_| Vec::new());
    for name in names {
        let op = ops
            .lookup(name)
            .unwrap_or_else(|e| panic!("bench operator {name}: {e:?}"));
        by_arity[usize::from(op.arity) - 1].push(op.id);
    }
    by_arity
}

fn gen_random_tree_fixed_size<T: Float, Ops: OperatorSet<T = T>, const D: usize, R: Rng>(
    rng: &mut R,
    target_size: usize,
    n_features: usize,
    ops_by_arity: &[Vec<u16>; D],
) -> PostfixExpr<T, Ops, D> {
    assert!(target_size >= 1);
    let mut nodes = Vec::with_capacity(target_size);
    let mut consts: Vec<T> = Vec::new();
    nodes.push(random_leaf(rng, n_features, &mut consts));

    while nodes.len() < target_size {
        let rem = target_size - nodes.len();
        let max_arity = rem.min(D);
        if max_arity == 0 {
            break;
        }
        let arity = rng.random_range(1..=max_arity) as u8;

        let Some(choices) = ops_by_arity.get(usize::from(arity) - 1) else {
            break;
        };
        if choices.is_empty() {
            break;
        }
        let op = choices[rng.random_range(0..choices.len())];

        let leaves: Vec<_> = nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| matches!(n, PNode::Var { .. } | PNode::Const { .. }).then_some(i))
            .collect();
        if leaves.is_empty() {
            break;
        }
        let pos = leaves[rng.random_range(0..leaves.len())];

        let mut repl = Vec::with_capacity(usize::from(arity) + 1);
        for _ in 0..arity {
            repl.push(random_leaf(rng, n_features, &mut consts));
        }
        repl.push(PNode::Op { arity, op });
        nodes.splice(pos..=pos, repl);
    }

    PostfixExpr::new(nodes, consts, Default::default())
}

fn make_data<T: Float>(n_rows: usize) -> Array2<T> {
    // Column-major layout for evaluation: shape = (n_features, n_rows).
    let mut data: Vec<T> = Vec::with_capacity(N_FEATURES * n_rows);
    for feature in 0..N_FEATURES {
        for row in 0..n_rows {
            let v = (row as f64 * 0.01) + (feature as f64 * 0.1);
            data.push(T::from(v).unwrap());
        }
    }
    Array2::from_shape_vec((N_FEATURES, n_rows), data).unwrap()
}

fn bench_eval_group<T, Ops, const D: usize>(c: &mut criterion::Criterion, ops: &Ops, type_name: &str, op_names: &[&str])
where
    T: Float + core::ops::AddAssign + Send + Sync,
    Ops: OperatorSet<T = T> + Send + Sync,
{
    let mut rng = StdRng::seed_from_u64(0);
    let ops_by_arity = ops_by_arity_from_names::<Ops, D>(ops, op_names);
    let trees: Vec<PostfixExpr<T, Ops, D>> = (0..N_TREES)
        .map(|_| gen_random_tree_fixed_size(&mut rng, TREE_SIZE, N_FEATURES, &ops_by_arity))
        .collect();
    let opts = EvalOptions {
        check_finite: false,
        early_exit: false,
    };

    let mut group = c.benchmark_group(format!("evaluation/{type_name}"));
    for n_rows in ROW_COUNTS {
        let data = make_data::<T>(n_rows);
        let x_view = data.view();
        group.bench_function(criterion::BenchmarkId::new("eval", n_rows), |b| {
            let mut out = vec![T::zero(); n_rows];
            let mut ctx = dynamic_expressions::EvalContext::<T, D>::new(n_rows);
            b.iter(|| {
                for tree in &trees {
                    let _ = dynamic_expressions::eval_tree_array_into(&mut out, tree, ops, x_view, &mut ctx, &opts);
                }
            })
        });

        group.bench_function(criterion::BenchmarkId::new("derivative", n_rows), |b| {
            let mut gctx = dynamic_expressions::GradContext::<T, D>::new(n_rows);
            b.iter(|| {
                for tree in &trees {
                    let _ = dynamic_expressions::eval_grad_tree_array(tree, ops, x_view, true, &mut gctx, &opts);
                }
            })
        });
    }
    group.finish();
}

fn bench_utilities<T, Ops, const D: usize>(c: &mut criterion::Criterion, ops: &Ops, type_name: &str)
where
    T: Float + Send + Sync,
    Ops: OperatorSet<T = T> + Send + Sync,
{
    let mut rng = StdRng::seed_from_u64(1);
    let ops_by_arity = all_ops_by_arity::<Ops, D>(ops);
    let mut trees: Vec<PostfixExpr<T, Ops, D>> = (0..N_TREES)
        .map(|_| gen_random_tree_fixed_size(&mut rng, TREE_SIZE, N_FEATURES, &ops_by_arity))
        .collect();
    let eval_opts = EvalOptions {
        check_finite: false,
        early_exit: false,
    };

    let mut group = c.benchmark_group(format!("utilities/{type_name}"));
    group.bench_function(criterion::BenchmarkId::from_parameter("clone"), |b| {
        b.iter(|| trees.to_vec())
    });

    group.bench_function(criterion::BenchmarkId::from_parameter("simplify"), |b| {
        b.iter(|| {
            for tree in &mut trees {
                let _ = dynamic_expressions::simplify_in_place(tree, ops, &eval_opts);
            }
        })
    });

    group.bench_function(criterion::BenchmarkId::from_parameter("combine_operators"), |b| {
        b.iter(|| {
            for tree in &mut trees {
                let _ = dynamic_expressions::combine_operators_in_place(tree, ops);
            }
        })
    });

    group.bench_function(criterion::BenchmarkId::from_parameter("counting"), |b| {
        b.iter(|| {
            for tree in &trees {
                let _ = dynamic_expressions::node_utils::count_nodes(&tree.nodes);
                let _ = dynamic_expressions::node_utils::count_depth(&tree.nodes);
                let _ = dynamic_expressions::node_utils::count_constant_nodes(&tree.nodes);
            }
        })
    });

    group.bench_function(criterion::BenchmarkId::from_parameter("compress_constants"), |b| {
        b.iter_with_setup(
            || trees.clone(),
            |mut cloned_trees| {
                for tree in &mut cloned_trees {
                    let _ = dynamic_expressions::compress_constants(tree);
                }
            },
        )
    });

    group.finish();
}

fn bench_count_depth_sizes<T, Ops, const D: usize>(c: &mut criterion::Criterion, ops: &Ops, type_name: &str)
where
    T: Float + Send + Sync,
    Ops: OperatorSet<T = T> + Send + Sync,
{
    let mut rng = StdRng::seed_from_u64(2);
    let ops_by_arity = all_ops_by_arity::<Ops, D>(ops);
    let sizes = [8usize, 20, 64, 128];
    let mut group = c.benchmark_group(format!("utilities/{type_name}/count_depth_sizes"));

    for size in sizes {
        let trees: Vec<PostfixExpr<T, Ops, D>> = (0..N_TREES)
            .map(|_| gen_random_tree_fixed_size(&mut rng, size, N_FEATURES, &ops_by_arity))
            .collect();

        group.bench_function(criterion::BenchmarkId::from_parameter(size), |b| {
            b.iter(|| {
                for tree in &trees {
                    let _ = dynamic_expressions::node_utils::count_depth(&tree.nodes);
                }
            })
        });
    }

    group.finish();
}

fn criterion_benchmark(c: &mut criterion::Criterion) {
    bench_eval_group::<f32, BenchOpsF32, 2>(c, &BenchOpsF32::new(), "Float32", &BENCH_OP_NAMES);
    bench_eval_group::<f64, BenchOpsF64, 2>(c, &BenchOpsF64::new(), "Float64", &BENCH_OP_NAMES);

    bench_utilities::<f32, BenchOpsF32, 2>(c, &BenchOpsF32::new(), "Float32");
    bench_utilities::<f64, BenchOpsF64, 2>(c, &BenchOpsF64::new(), "Float64");
    bench_count_depth_sizes::<f32, BenchOpsF32, 2>(c, &BenchOpsF32::new(), "Float32");
    bench_count_depth_sizes::<f64, BenchOpsF64, 2>(c, &BenchOpsF64::new(), "Float64");

    // Builtin opsets include ternary operators; benchmark them as well.
    bench_eval_group::<f32, dynamic_expressions::operator_enum::presets::BuiltinOpsF32, 3>(
        c,
        &dynamic_expressions::operator_enum::presets::BuiltinOpsF32::new(),
        "BuiltinF32",
        &BUILTIN_OP_NAMES,
    );
    bench_eval_group::<f64, dynamic_expressions::operator_enum::presets::BuiltinOpsF64, 3>(
        c,
        &dynamic_expressions::operator_enum::presets::BuiltinOpsF64::new(),
        "BuiltinF64",
        &BUILTIN_OP_NAMES,
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
