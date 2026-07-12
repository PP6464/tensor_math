//! Benchmarks for the single-threaded vs. multi-threaded `concat` implementations
//! of [`Tensor`] and [`Matrix`].

use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tensor_math::definitions::tensor::Tensor;
use tensor_math::definitions::matrix::Matrix;
use tensor_math::definitions::shape::Shape;
use tensor_math::shape;

fn bench_concat_1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("concat/tensor_1d");
    for &n in &[1_000usize, 100_000, 1_000_000, 10_000_000] {
        let a = Tensor::from_value(&shape![n], 1.0);
        let b = Tensor::from_value(&shape![n], 2.0);
        group.throughput(Throughput::Elements(2 * n as u64));

        group.bench_with_input(BenchmarkId::new("st", n), &n, |bench, _| {
            bench.iter(|| {
                let r = a.concat(&b, 0).expect("concat must succeed");
                black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("mt", n), &n, |bench, _| {
            bench.iter(|| {
                let r = a.concat_mt(&b, 0).expect("concat must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_concat_2d_axis1(c: &mut Criterion) {
    let mut group = c.benchmark_group("concat/tensor_2d_axis1");
    for &(rows, cols) in &[
        (32usize, 32usize),
        (128, 128),
        (512, 512),
        (1024, 1024),
        (2048, 4096),
    ] {
        let a = Tensor::from_value(&shape![rows, cols], 1.0);
        let b = Tensor::from_value(&shape![rows, cols], 2.0);
        let elems = (2 * rows * cols) as u64;
        group.throughput(Throughput::Elements(elems));

        group.bench_with_input(
            BenchmarkId::new("st", format!("{rows}x{cols}")),
            &(rows, cols),
            |bench, _| {
                bench.iter(|| {
                    let r = a.concat(&b, 1).expect("concat must succeed");
                    black_box(r);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("mt", format!("{rows}x{cols}")),
            &(rows, cols),
            |bench, _| {
                bench.iter(|| {
                    let r = a.concat_mt(&b, 1).expect("concat must succeed");
                    black_box(r);
                });
            },
        );
    }
    group.finish();
}

fn bench_concat_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("concat/tensor_3d");
    for &(d0, d1, d2) in &[
        (8usize, 64usize, 64usize),
        (16, 128, 128),
        (32, 256, 256),
        (64, 256, 256),
    ] {
        let a = Tensor::from_value(&shape![d0, d1, d2], 1.0);
        let b = Tensor::from_value(&shape![d0, d1, d2], 1.0);
        let elems = (2 * d0 * d1 * d2) as u64;
        group.throughput(Throughput::Elements(elems));

        let label = format!("{d0}x{d1}x{d2}_axis2");

        group.bench_with_input(BenchmarkId::new("st", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.concat(&b, 2).expect("concat must succeed");
                black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("mt", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.concat_mt(&b, 2).expect("concat must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_concat_cols(c: &mut Criterion) {
    let mut group = c.benchmark_group("concat/matrix_cols");
    for &(rows, cols) in &[
        (32usize, 32usize),
        (128, 128),
        (512, 512),
        (1024, 1024),
        (2048, 4096),
    ] {
        let a = Matrix::from_value(rows, cols, 1.0);
        let b = Matrix::from_value(rows, cols, 2.0);
        group.throughput(Throughput::Elements((2 * rows * cols) as u64));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("st", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.concat_cols(&b).expect("concat must succeed");
                black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("mt", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.concat_cols_mt(&b).expect("concat must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_concat_rows(c: &mut Criterion) {
    let mut group = c.benchmark_group("concat/matrix_rows");
    for &(rows, cols) in &[
        (32usize, 32usize),
        (128, 128),
        (512, 512),
        (1024, 1024),
        (2048, 4096),
    ] {
        let a = Matrix::from_value(rows, cols, 1.0);
        let b = Matrix::from_value(rows, cols, 2.0);
        group.throughput(Throughput::Elements((2 * rows * cols) as u64));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("st", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.concat_rows(&b).expect("concat must succeed");
                black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("mt", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.concat_rows_mt(&b).expect("concat must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

criterion_group!(
    name = concat_benches;
    config = Criterion::default().sample_size(10);
    targets =
        bench_concat_1d,
        bench_concat_2d_axis1,
        bench_concat_3d,
        bench_concat_cols,
        bench_concat_rows,
);

criterion_main!(concat_benches);
