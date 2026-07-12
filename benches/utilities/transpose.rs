//! Benchmarks for the single-threaded vs. multi-threaded `transpose`
//! implementations of [`Tensor`] and [`Matrix`].

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tensor_math::definitions::tensor::Tensor;
use tensor_math::definitions::shape::Shape;
use tensor_math::definitions::transpose::Transpose;
use tensor_math::{shape, transpose};
use tensor_math::definitions::matrix::Matrix;

fn bench_transpose_1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose/tensor_1d");
    let perm: Transpose = transpose![0];
    for &n in &[1_000usize, 100_000, 1_000_000, 10_000_000] {
        let a = Tensor::from_value(
            &shape![n],
            1.0
        );
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("st", n), &n, |bench, _| {
            bench.iter(|| {
                let r = a.transpose(&perm).expect("transpose must succeed");
                black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("mt", n), &n, |bench, _| {
            bench.iter(|| {
                let r = a.transpose_mt(&perm).expect("transpose must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_transpose_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose/tensor_2d");
    let perm: Transpose = transpose![1, 0];
    for &(rows, cols) in &[
        (32usize, 32usize),
        (128, 128),
        (512, 512),
        (1024, 1024),
        (2048, 4096),
    ] {
        let a = Tensor::from_value(&shape![rows, cols], 1.0);
        let elems = (rows * cols) as u64;
        group.throughput(Throughput::Elements(elems));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("st", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.transpose(&perm).expect("transpose must succeed");
                black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("mt", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.transpose_mt(&perm).expect("transpose must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_transpose_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose/tensor_3d");
    let perm: Transpose = transpose![2, 0, 1];
    for &(d0, d1, d2) in &[
        (8usize, 64usize, 64usize),
        (16, 128, 128),
        (32, 256, 256),
        (64, 256, 256),
    ] {
        let a = Tensor::from_value(&shape![d0, d1, d2], 1.0);
        let elems = (d0 * d1 * d2) as u64;
        group.throughput(Throughput::Elements(elems));

        let label = format!("{d0}x{d1}x{d2}");

        group.bench_with_input(BenchmarkId::new("st", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.transpose(&perm).expect("transpose must succeed");
                black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("mt", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.transpose_mt(&perm).expect("transpose must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_transpose_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose/matrix");
    for &(rows, cols) in &[
        (32usize, 32usize),
        (128, 128),
        (512, 512),
        (1024, 1024),
        (2048, 4096),
    ] {
        let a = Matrix::from_value(rows, cols, 1.0);
        group.throughput(Throughput::Elements((rows * cols) as u64));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("st", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.transpose();
                black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("mt", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.transpose_mt();
                black_box(r);
            });
        });
    }
    group.finish();
}

criterion_group!(
    name = transpose_benches;
    config = Criterion::default().sample_size(10);
    targets =
        bench_transpose_1d,
        bench_transpose_2d,
        bench_transpose_3d,
        bench_transpose_matrix,
);

criterion_main!(transpose_benches);
