//! Benchmarks for the `pool` family on [`Tensor`] and [`Matrix`], and for
//! the [`pool_sum`], [`pool_max`], [`pool_min`], [`pool_avg`] (and
//! `_mat` equivalents) free helper functions.
//!
//! `pool` walks every kernel window in the input, calls the closure on it,
//! and writes the closure's return value into the result. The interesting
//! comparison is the serial loop (Tensor/Matrix::pool) vs. the rayon
//! parallel variant (Tensor/Matrix::pool_mt).
//!
//! We use `pool_sum` as the closure so the per-window work is proportional
//! to the kernel size — a no-op closure would make the rayon overhead the
//! whole story.

use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};
use tensor_math::definitions::matrix::Matrix;
use tensor_math::definitions::shape::Shape;

use tensor_math::utilities::tensor::{
    pool_avg, pool_max, pool_min, pool_sum,
};

use super::bench_utils::{drain_f64, drain_mat_f64, matrix, tensor_from_shape};

use tensor_math::utilities::matrix::{
    pool_avg_mat, pool_max_mat, pool_min_mat, pool_sum_mat,
};

// ---------------------------------------------------------------------------
// Tensor::pool / pool_mt
// ---------------------------------------------------------------------------
//
// Kernel and stride shapes are chosen so the input tensor is partitioned
// into a moderate grid of non-overlapping windows: this gives both
// implementations real per-window work and lets the parallel variant
// benefit from many rayon tasks.

fn bench_pool_tensor(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool/tensor_2d");
    for &(rows, cols) in &[
        (256usize, 256usize),
        (512, 512),
        (1024, 1024),
    ] {
        let a = tensor_from_shape(&[rows, cols], 1.0);
        let kernel = Shape::new(vec![8, 8]);
        let stride = Shape::new(vec![8, 8]);
        // Throughput counts input elements touched (every window is read
        // once).
        group.throughput(Throughput::Elements((rows * cols) as u64));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("st", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a
                    .pool(pool_sum::<f64>, &kernel, &stride, 0.0)
                    .expect("pool must succeed");
                std::hint::black_box(drain_f64(&r));
            });
        });
        group.bench_with_input(BenchmarkId::new("mt", &label), &label, |bench, _| {
            let pool_fn = pool_sum::<f64>;
            bench.iter(|| {
                let r = a
                    .pool_mt(&pool_fn, &kernel, &stride, 0.0)
                    .expect("pool_mt must succeed");
                std::hint::black_box(drain_f64(&r));
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Tensor::pool_indexed / pool_indexed_mt
// ---------------------------------------------------------------------------

fn bench_pool_indexed_tensor(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_indexed/tensor_2d");
    for &(rows, cols) in &[
        (256usize, 256usize),
        (512, 512),
        (1024, 1024),
    ] {
        let a = tensor_from_shape(&[rows, cols], 1.0);
        let kernel = Shape::new(vec![8, 8]);
        let stride = Shape::new(vec![8, 8]);
        group.throughput(Throughput::Elements((rows * cols) as u64));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("st", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a
                    .pool_indexed(
                        |_idx, t| pool_sum::<f64>(t),
                        &kernel,
                        &stride,
                        0.0,
                    )
                    .expect("pool_indexed must succeed");
                std::hint::black_box(drain_f64(&r));
            });
        });
        group.bench_with_input(BenchmarkId::new("mt", &label), &label, |bench, _| {
            let pool_fn = |idx: Vec<usize>, t: tensor_math::definitions::tensor::Tensor<f64>| {
                pool_sum::<f64>(t) + idx[0] as f64
            };
            bench.iter(|| {
                let r = a
                    .pool_indexed_mt(&pool_fn, &kernel, &stride, 0.0)
                    .expect("pool_indexed_mt must succeed");
                std::hint::black_box(drain_f64(&r));
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Matrix::pool / pool_mt
// ---------------------------------------------------------------------------

fn bench_pool_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool/matrix");
    for &(rows, cols) in &[
        (256usize, 256usize),
        (512, 512),
        (1024, 1024),
    ] {
        let a = matrix(rows, cols, 1.0);
        let kernel = (8usize, 8usize);
        let stride = (8usize, 8usize);
        group.throughput(Throughput::Elements((rows * cols) as u64));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("st", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a
                    .pool(pool_sum_mat::<f64>, kernel, stride, 0.0)
                    .expect("pool must succeed");
                std::hint::black_box(drain_mat_f64(&r));
            });
        });
        group.bench_with_input(BenchmarkId::new("mt", &label), &label, |bench, _| {
            let pool_fn = pool_sum_mat::<f64>;
            bench.iter(|| {
                let r = a
                    .pool_mt(&pool_fn, kernel, stride, 0.0)
                    .expect("pool_mt must succeed");
                std::hint::black_box(drain_mat_f64(&r));
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Matrix::pool_indexed / pool_indexed_mt
// ---------------------------------------------------------------------------

fn bench_pool_indexed_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_indexed/matrix");
    for &(rows, cols) in &[
        (256usize, 256usize),
        (512, 512),
        (1024, 1024),
    ] {
        let a = matrix(rows, cols, 1.0);
        let kernel = (8usize, 8usize);
        let stride = (8usize, 8usize);
        group.throughput(Throughput::Elements((rows * cols) as u64));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("st", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a
                    .pool_indexed(
                        |_idx, m| pool_sum_mat::<f64>(m),
                        kernel,
                        stride,
                        0.0,
                    )
                    .expect("pool_indexed must succeed");
                std::hint::black_box(drain_mat_f64(&r));
            });
        });
        group.bench_with_input(BenchmarkId::new("mt", &label), &label, |bench, _| {
            let pool_fn = |(r, _c): (usize, usize), m: Matrix<f64>| {
                pool_sum_mat::<f64>(m) + r as f64
            };
            bench.iter(|| {
                let r = a
                    .pool_indexed_mt(&pool_fn, kernel, stride, 0.0)
                    .expect("pool_indexed_mt must succeed");
                std::hint::black_box(drain_mat_f64(&r));
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// pool_sum / pool_max / pool_min / pool_avg (Tensor free functions)
// ---------------------------------------------------------------------------
//
// These are the building-block closures used by callers of `pool`/`pool_mt`.
// We measure them in isolation so the pool benchmark numbers can be
// interpreted as "kernel evaluation + scheduling" rather than "scheduling".

fn bench_pool_helpers_tensor(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_helpers/tensor");
    for &n in &[1_000usize, 100_000, 1_000_000] {
        let a = tensor_from_shape(&[n], 1.0);
        group.throughput(Throughput::Elements(n as u64));

        let label = format!("len{n}");

        group.bench_with_input(BenchmarkId::new("sum", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = pool_sum::<f64>(a.clone());
                std::hint::black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("min", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = pool_min::<f64>(a.clone());
                std::hint::black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("max", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = pool_max::<f64>(a.clone());
                std::hint::black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("avg", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = pool_avg::<f64>(a.clone());
                std::hint::black_box(r);
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// pool_sum_mat / pool_max_mat / pool_min_mat / pool_avg_mat
// ---------------------------------------------------------------------------

fn bench_pool_helpers_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_helpers/matrix");
    for &n in &[128usize, 512, 2048] {
        let a = matrix(n, n, 1.0);
        group.throughput(Throughput::Elements((n * n) as u64));

        let label = format!("{n}x{n}");

        group.bench_with_input(BenchmarkId::new("sum", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = pool_sum_mat::<f64>(a.clone());
                std::hint::black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("min", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = pool_min_mat::<f64>(a.clone());
                std::hint::black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("max", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = pool_max_mat::<f64>(a.clone());
                std::hint::black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("avg", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = pool_avg_mat::<f64>(a.clone());
                std::hint::black_box(r);
            });
        });
    }
    group.finish();
}

criterion_group!(
    name = pool_benches;
    config = Criterion::default().sample_size(10);
    targets =
        bench_pool_tensor,
        bench_pool_indexed_tensor,
        bench_pool_matrix,
        bench_pool_indexed_matrix,
        bench_pool_helpers_tensor,
        bench_pool_helpers_matrix,
);
// `criterion_group!` above declares `pub fn pool_benches()`.
