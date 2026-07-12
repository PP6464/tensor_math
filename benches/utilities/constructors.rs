//! Benchmarks for the various zero/one/random tensor & matrix constructors.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tensor_math::definitions::matrix::Matrix;
use tensor_math::definitions::shape::Shape;
use tensor_math::utilities::matrix::{eye, identity};

fn bench_constructors_tensor(c: &mut Criterion) {
    let mut group = c.benchmark_group("constructors/tensor");
    for &n in &[1_000usize, 100_000, 1_000_000] {
        let shape = Shape::new(vec![n]);
        let elems = n as u64;
        group.throughput(Throughput::Elements(elems));

        let label = format!("len{n}");

        group.bench_with_input(BenchmarkId::new("from_value", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = tensor_math::definitions::tensor::Tensor::from_value(&shape, 1.0);
                black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("from_shape", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = tensor_math::definitions::tensor::Tensor::<f64>::from_shape(&shape);
                black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("zeros", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = tensor_math::definitions::tensor::Tensor::<f64>::zeros(&shape);
                black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("rand", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = tensor_math::definitions::tensor::Tensor::<f64>::rand(&shape);
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_constructors_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("constructors/matrix");
    for &n in &[64usize, 256, 1024, 4096] {
        let elems = (n * n) as u64;
        group.throughput(Throughput::Elements(elems));

        let label = format!("{n}x{n}");

        group.bench_with_input(BenchmarkId::new("from_value", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = Matrix::from_value(n, n, 1.0);
                black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("from_shape", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = Matrix::<f64>::from_shape(n, n);
                black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("zeros", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = Matrix::<f64>::zeros(n, n);
                black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("rand", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = Matrix::<f64>::rand(n, n);
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_identity_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("constructors/identity");
    for &n in &[64usize, 256, 1024] {
        let elems = (n * n) as u64;
        group.throughput(Throughput::Elements(elems));

        let label = format!("{n}x{n}");

        group.bench_with_input(BenchmarkId::new("identity", &label), &label, |bench, _| {
            bench.iter(|| {
                let r: Matrix<f64> = identity(n);
                black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("eye", &label), &label, |bench, _| {
            bench.iter(|| {
                let r: Matrix<f64> = eye(n);
                black_box(r);
            });
        });
    }
    group.finish();
}

criterion_group!(
    name = constructors_benches;
    config = Criterion::default().sample_size(10);
    targets =
        bench_constructors_tensor,
        bench_constructors_matrix,
        bench_identity_matrix,
);

criterion_main!(constructors_benches);
