//! Benchmarks for the shape-only operations [`Tensor::reshape`],
//! [`Tensor::flatten`], and [`Matrix::reshape`].

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tensor_math::definitions::matrix::Matrix;
use tensor_math::definitions::shape::Shape;
use tensor_math::definitions::tensor::Tensor;
use tensor_math::shape;

fn bench_reshape_tensor(c: &mut Criterion) {
    let mut group = c.benchmark_group("reshape/tensor");
    for &n in &[1_000usize, 100_000, 1_000_000] {
        // 1-D → 2-D square
        let a = Tensor::from_value(&shape![n], 1.0);
        let side = (n as f64).sqrt() as usize;
        // Use sizes that exactly multiply to n.
        let new_shape = Shape::new(vec![side, n / side]);
        let label = format!("1d_{n}->2d");

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("1d_to_2d", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.clone().reshape(&new_shape).expect("reshape must succeed");
                black_box(r);
            });
        });

        // 2-D → 1-D collapse
        let a2 = Tensor::from_value(&shape![side, n / side], 1.0);
        let flat = Shape::new(vec![n]);
        let label2 = format!("2d_{n}->1d");

        group.bench_with_input(BenchmarkId::new("2d_to_1d", &label2), &label2, |bench, _| {
            bench.iter(|| {
                let r = a2.clone().reshape(&flat).expect("reshape must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_flatten_tensor(c: &mut Criterion) {
    let mut group = c.benchmark_group("flatten/tensor");
    for &n in &[1_000usize, 100_000, 1_000_000] {
        let a = Tensor::from_value(&shape![1, n], 1.0);
        group.throughput(Throughput::Elements(n as u64));

        let label = format!("1x{n}");

        group.bench_with_input(BenchmarkId::new("axis0", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.clone().flatten(0).expect("flatten must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_reshape_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("reshape/matrix");
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

        group.bench_with_input(BenchmarkId::new("swap", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.clone().reshape(cols, rows).expect("reshape must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

criterion_group!(
    name = reshape_flatten_benches;
    config = Criterion::default().sample_size(10);
    targets =
        bench_reshape_tensor,
        bench_flatten_tensor,
        bench_reshape_matrix,
);

criterion_main!(reshape_flatten_benches);
