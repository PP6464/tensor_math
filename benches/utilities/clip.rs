//! Benchmarks for `Tensor::clip` / `par_clip` and `Matrix::clip` / `par_clip`.

use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tensor_math::definitions::matrix::Matrix;
use tensor_math::definitions::tensor::Tensor;
use tensor_math::definitions::shape::Shape;
use tensor_math::shape;

fn bench_clip_tensor(c: &mut Criterion) {
    let mut group = c.benchmark_group("clip/tensor_2d");
    for &(rows, cols) in &[
        (32usize, 32usize),
        (128, 128),
        (512, 512),
        (1024, 1024),
        (2048, 4096),
    ] {
        let a = Tensor::from_value(
            &shape![rows, cols],
            1.0
        );
        let elems = (rows * cols) as u64;
        group.throughput(Throughput::Elements(elems));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("st", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.clip(-0.5, 0.5);
                black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("mt", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.par_clip(-0.5, 0.5);
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_clip_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("clip/matrix");
    for &(rows, cols) in &[
        (32, 32),
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
                let r = a.clip(-0.5, 0.5);
                black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("mt", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.par_clip(-0.5, 0.5);
                black_box(r);
            });
        });
    }
    group.finish();
}

criterion_group!(
    name = clip_benches;
    config = Criterion::default().sample_size(10);
    targets = bench_clip_tensor, bench_clip_matrix,
);
