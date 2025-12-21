use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use dynamic_expressions::PostfixExpression;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Duration;
use symbolic_regression::compute_complexity;
use symbolic_regression::{check_constraints, equation_search, optimize_constants, PopMember};
use symbolic_regression::{Dataset, MutationWeights, OperatorLibrary, Options};

type T = f32;
const D: usize = 3;
type Ops = dynamic_expressions::operator_enum::presets::BuiltinOpsF32;
const N_FEATURES: usize = 5;

fn make_options(seed: u64) -> Options<T, D> {
    let mutation_weights = MutationWeights {
        swap_operands: 0.0,
        form_connection: 0.0,
        break_connection: 0.0,
        ..Default::default()
    };

    Options {
        seed,
        niterations: 15,
        populations: 1,
        population_size: 64,
        ncycles_per_iteration: 380,
        batch_size: 50,
        maxsize: 30,
        maxdepth: 30,
        parsimony: 0.0,
        perturbation_factor: 0.129,
        probability_negate_constant: 0.00743,
        tournament_selection_n: 15,
        tournament_selection_p: 1.0,
        crossover_probability: 0.0259,
        mutation_weights,
        operators: OperatorLibrary::sr_default::<Ops, D>(),
        progress: false,
        ..Default::default()
    }
}

fn make_dataset(seed: u64) -> (Dataset<T>, Vec<T>) {
    let n = 1_000usize;
    let n_features = 5usize;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x: Vec<T> = Vec::with_capacity(n * n_features);
    for _ in 0..n * n_features {
        x.push(rng.random_range(-5.0f32..5.0f32) as T);
    }
    let x_arr = Array2::from_shape_vec((n, n_features), x).unwrap();

    let eqn = |row: &[T]| -> T {
        let a = (2.13f32 as T * row[0]).cos();
        let b = row[1] * row[2].abs().powf(0.9f32 as T) * (0.5f32 as T);
        let c = row[3].abs().powf(1.5f32 as T) * (0.3f32 as T);
        a + b - c
    };

    let mut y = Vec::with_capacity(n);
    for r in 0..n {
        let row = x_arr.row(r);
        y.push(eqn(row.as_slice().unwrap()) + (0.1f32 as T) * rng.random_range(-1.0f32..1.0f32));
    }

    (Dataset::new(x_arr, Array1::from_vec(y.clone())), y)
}

fn bench_search(c: &mut Criterion) {
    let seeds = [1u64];
    let mut group = c.benchmark_group("search");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(2));
    group.warm_up_time(Duration::from_secs(1));
    for &seed in &seeds {
        let (dataset, _) = make_dataset(seed);
        let options = make_options(seed);
        group.bench_function(BenchmarkId::new("equation_search", seed), |b| {
            b.iter(|| {
                let _res = equation_search::<T, Ops, D>(&dataset, &options);
            })
        });
    }
    group.finish();
}

fn bench_optimize_constants(c: &mut Criterion) {
    let (dataset, _) = make_dataset(9);
    let options = make_options(9);
    let mut rng = StdRng::seed_from_u64(42);
    let expr = symbolic_regression::random_expr::<T, Ops, D, _>(
        &mut rng,
        &options.operators,
        dataset.n_features,
        12,
    );
    let member = PopMember::from_expr(
        symbolic_regression::MemberId(0),
        None,
        0,
        expr,
        dataset.n_features,
    );

    let mut group = c.benchmark_group("utils/optimize_constants");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(2));
    group.warm_up_time(Duration::from_secs(1));
    group.bench_function("optimize_constants_single", |b| {
        b.iter(|| {
            let mut evaluator = symbolic_regression::Evaluator::new(dataset.n_rows);
            let mut grad_ctx = dynamic_expressions::GradContext::<T, D>::new(dataset.n_rows);
            let mut next_birth = 0u64;

            let mut m = member.clone();
            let ctx = symbolic_regression::OptimizeConstantsCtx {
                dataset: symbolic_regression::TaggedDataset::new(&dataset, None),
                options: &options,
                evaluator: &mut evaluator,
                grad_ctx: &mut grad_ctx,
                next_birth: &mut next_birth,
            };

            let _ = optimize_constants(&mut rng, &mut m, ctx);
        })
    });
    group.finish();
}

fn bench_complexity_and_constraints(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(7);
    let options = make_options(7);
    let trees: Vec<_> = (0..20)
        .map(|_| {
            symbolic_regression::random_expr::<T, Ops, D, _>(
                &mut rng,
                &options.operators,
                N_FEATURES,
                20,
            )
        })
        .collect();

    let mut group = c.benchmark_group("utils");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(2));
    group.warm_up_time(Duration::from_secs(1));
    group.bench_function("compute_complexity_x20", |b| {
        b.iter(|| {
            for tree in &trees {
                let _ = compute_complexity(tree.nodes(), &options);
            }
        })
    });

    group.bench_function("check_constraints_x20", |b| {
        b.iter(|| {
            for tree in &trees {
                let _ = check_constraints(tree, &options, options.maxsize);
            }
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_search,
    bench_optimize_constants,
    bench_complexity_and_constraints
);
criterion_main!(benches);
