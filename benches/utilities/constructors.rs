//! Benchmarks for the various zero/one/random tensor & matrix constructors.
//!
//! These benchmarks are short on the per-iteration work and so the bulk of
//! the time is in `vec![value; n]` allocation. We still include them
//! because (a) constructors are part of the public API and (b) they are
//! the standard way to build inputs for the more interesting ops.

use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};
use tensor_math::definitions::matrix::Matrix;
use tensor_math::definitions::shape::Shape;
use tensor_math::utilities::matrix::{eye, identity};

use super::bench_utils::{drain_f64, drain_mat_f64, matrix};

// ---------------------------------------------------------------------------
// Tensor constructors
// ---------------------------------------------------------------------------

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
                std::hint::black_box(drain_f64(&r));
            });
        });
        group.bench_with_input(BenchmarkId::new("from_shape", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = tensor_math::definitions::tensor::Tensor::from_shape(&shape);
                std::hint::black_box(drain_f64(&r));
            });
        });
        group.bench_with_input(BenchmarkId::new("zeros", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = tensor_math::definitions::tensor::Tensor::zeros(&shape);
                std::hint::black_box(drain_f64(&r));
            });
        });
        group.bench_with_input(BenchmarkId::new("rand", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = tensor_math::definitions::tensor::Tensor::rand(&shape);
                std::hint::black_box(drain_f64(&r));
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Matrix constructors
// ---------------------------------------------------------------------------

fn bench_constructors_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("constructors/matrix");
    for &n in &[64usize, 256, 1024, 4096] {
        let elems = (n * n) as u64;
        group.throughput(Throughput::Elements(elems));

        let label = format!("{n}x{n}");

        group.bench_with_input(BenchmarkId::new("from_value", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = matrix(n, n, 1.0);
                std::hint::black_box(drain_mat_f64(&r));
            });
        });
        group.bench_with_input(BenchmarkId::new("from_shape", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = Matrix::from_shape(n, n);
                std::hint::black_box(drain_mat_f64(&r));
            });
        });
        group.bench_with_input(BenchmarkId::new("zeros", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = Matrix::zeros(n, n);
                std::hint::black_box(drain_mat_f64(&r));
            });
        });
        group.bench_with_input(BenchmarkId::new("rand", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = Matrix::rand(n, n);
                std::hint::black_box(drain_mat_f64(&r));
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Matrix::identity / Matrix::eye
// ---------------------------------------------------------------------------
//
// `identity` and `eye` are the same function under two names — but we still
// benchmark them separately so the two entries are visible in the output
// (mirrors the public API).

fn bench_identity_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("constructors/identity");
    for &n in &[64usize, 256, 1024] {
        let elems = (n * n) as u64;
        group.throughput(Throughput::Elements(elems));

        let label = format!("{n}x{n}");

        group.bench_with_input(BenchmarkId::new("identity", &label), &label, |bench, _| {
            bench.iter(|| {
                let r: Matrix<f64> = identity(n);
                std::hint::black_box(drain_mat_f64(&r));
            });
        });
        group.bench_with_input(BenchmarkId::new("eye", &label), &label, |bench, _| {
            bench.iter(|| {
                let r: Matrix<f64> = eye(n);
                std::hint::black_box(drain_mat_f64(&r));
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
// `criterion_group!` above declares `pub fn constructors_benches()`.
