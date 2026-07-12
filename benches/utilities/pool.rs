//! Benchmarks for the `pool` family on [`Tensor`] and [`Matrix`], and for
//! the [`pool_sum`], [`pool_max`], [`pool_min`], [`pool_avg`] (and
//! `_mat` equivalents) free helper functions.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tensor_math::definitions::matrix::Matrix;
use tensor_math::definitions::shape::Shape;
use tensor_math::definitions::tensor::Tensor;
use tensor_math::shape;
use tensor_math::utilities::tensor::{
    pool_avg, pool_max, pool_min, pool_sum,
};

use tensor_math::utilities::matrix::{
    pool_avg_mat, pool_max_mat, pool_min_mat, pool_sum_mat,
};

fn bench_pool_tensor(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool/tensor_2d");
    for &(rows, cols) in &[
        (256usize, 256usize),
        (512, 512),
        (1024, 1024),
    ] {
        let a = Tensor::from_value(&shape![rows, cols], 1.0);
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
                black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("mt", &label), &label, |bench, _| {
            let pool_fn = pool_sum::<f64>;
            bench.iter(|| {
                let r = a
                    .pool_mt(&pool_fn, &kernel, &stride, 0.0)
                    .expect("pool_mt must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_pool_indexed_tensor(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_indexed/tensor_2d");
    for &(rows, cols) in &[
        (256usize, 256usize),
        (512, 512),
        (1024, 1024),
    ] {
        let a = Tensor::from_value(&shape![rows, cols], 1.0);
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
                black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("mt", &label), &label, |bench, _| {
            let pool_fn = |idx: Vec<usize>, t: Tensor<f64>| {
                pool_sum::<f64>(t) + idx[0] as f64
            };
            bench.iter(|| {
                let r = a
                    .pool_indexed_mt(&pool_fn, &kernel, &stride, 0.0)
                    .expect("pool_indexed_mt must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_pool_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool/matrix");
    for &(rows, cols) in &[
        (256usize, 256usize),
        (512, 512),
        (1024, 1024),
    ] {
        let a = Matrix::from_value(rows, cols, 1.0);
        let kernel = (8usize, 8usize);
        let stride = (8usize, 8usize);
        group.throughput(Throughput::Elements((rows * cols) as u64));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("st", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a
                    .pool(pool_sum_mat::<f64>, kernel, stride, 0.0)
                    .expect("pool must succeed");
                black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("mt", &label), &label, |bench, _| {
            let pool_fn = pool_sum_mat::<f64>;
            bench.iter(|| {
                let r = a
                    .pool_mt(&pool_fn, kernel, stride, 0.0)
                    .expect("pool_mt must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_pool_indexed_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_indexed/matrix");
    for &(rows, cols) in &[
        (256usize, 256usize),
        (512, 512),
        (1024, 1024),
    ] {
        let a = Matrix::from_value(rows, cols, 1.0);
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
                black_box(r);
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
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_pool_helpers_tensor(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_helpers/tensor");
    for &n in &[1_000usize, 100_000, 1_000_000] {
        let a = Tensor::from_value(&shape![n], 1.0);
        group.throughput(Throughput::Elements(n as u64));

        let label = format!("len{n}");

        group.bench_with_input(BenchmarkId::new("sum", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = pool_sum::<f64>(a.clone());
                black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("min", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = pool_min::<f64>(a.clone());
                black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("max", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = pool_max::<f64>(a.clone());
                black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("avg", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = pool_avg::<f64>(a.clone());
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_pool_helpers_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_helpers/matrix");
    for &n in &[128usize, 512, 2048] {
        let a = Matrix::from_value(n, n, 1.0);
        group.throughput(Throughput::Elements((n * n) as u64));

        let label = format!("{n}x{n}");

        group.bench_with_input(BenchmarkId::new("sum", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = pool_sum_mat::<f64>(a.clone());
                black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("min", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = pool_min_mat::<f64>(a.clone());
                black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("max", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = pool_max_mat::<f64>(a.clone());
                black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("avg", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = pool_avg_mat::<f64>(a.clone());
                black_box(r);
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

criterion_main!(pool_benches);
