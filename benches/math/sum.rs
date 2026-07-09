//! Benchmarks for `Tensor::sum` / `Tensor::sum_mt`, `Matrix::sum` /
//! `Matrix::sum_mt`, and `Matrix::trace` / `Matrix::trace_mt`.

use std::hint::black_box;

use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};
use num::complex::Complex64;

use tensor_math::definitions::matrix::Matrix;
use tensor_math::definitions::shape::Shape;
use tensor_math::definitions::tensor::Tensor;
use tensor_math::shape;

fn label_shape(shape: &Shape) -> String {
    let mut s = String::new();
    for i in 0..shape.rank() {
        s.push_str(&format!("{}x", shape[i]));
    }
    s.pop();
    s
}

fn random_complex_matrix(rows: usize, cols: usize) -> Matrix<Complex64> {
    let re = Matrix::<f64>::rand(rows, cols);
    let im = Matrix::<f64>::rand(rows, cols);
    re.into_complex() + im.into_complex().par_map(|v| v * Complex64::I)
}

fn random_complex_tensor(shape: &Shape) -> Tensor<Complex64> {
    let re = Tensor::<f64>::rand(shape);
    let im = Tensor::<f64>::rand(shape);
    re.into_complex() + im.into_complex().par_map(|v| v * Complex64::I)
}

fn bench_f64_tensor_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum");

    for shape in &[
        shape![1000],
        shape![200, 200],
        shape![100, 100, 100],
        shape![1000000],
    ] {
        let a = Tensor::<f64>::rand(shape);
        group.throughput(Throughput::Elements(shape.element_count() as u64));

        let label = label_shape(shape);

        group.bench_with_input(BenchmarkId::new("f64/tensor", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.sum();
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_f64_tensor_sum_mt(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_mt");

    for shape in &[
        shape![1000],
        shape![200, 200],
        shape![100, 100, 100],
        shape![1000000],
    ] {
        let a = Tensor::<f64>::rand(shape);
        group.throughput(Throughput::Elements(shape.element_count() as u64));

        let label = label_shape(shape);

        group.bench_with_input(BenchmarkId::new("f64/tensor", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.sum_mt();
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_c64_tensor_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum");

    for shape in &[
        shape![1000],
        shape![200, 200],
        shape![100, 100, 100],
        shape![1000000],
    ] {
        let a = random_complex_tensor(shape);
        group.throughput(Throughput::Elements(shape.element_count() as u64));

        let label = label_shape(shape);

        group.bench_with_input(BenchmarkId::new("c64/tensor", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.sum();
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_c64_tensor_sum_mt(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_mt");

    for shape in &[
        shape![1000],
        shape![200, 200],
        shape![100, 100, 100],
        shape![1000000],
    ] {
        let a = random_complex_tensor(shape);
        group.throughput(Throughput::Elements(shape.element_count() as u64));

        let label = label_shape(shape);

        group.bench_with_input(BenchmarkId::new("c64/tensor", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.sum_mt();
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_f64_matrix_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum");

    for &(rows, cols) in &[(100, 100), (1000, 1000), (100, 10000), (10000, 100)] {
        let a = Matrix::<f64>::rand(rows, cols);
        group.throughput(Throughput::Elements((rows * cols) as u64));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("f64/matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.sum();
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_f64_matrix_sum_mt(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_mt");

    for &(rows, cols) in &[(100, 100), (1000, 1000), (100, 10000), (10000, 100)] {
        let a = Matrix::<f64>::rand(rows, cols);
        group.throughput(Throughput::Elements((rows * cols) as u64));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("f64/matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.sum_mt();
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_c64_matrix_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum");

    for &(rows, cols) in &[(100, 100), (1000, 1000), (100, 10000), (10000, 100)] {
        let a = random_complex_matrix(rows, cols);
        group.throughput(Throughput::Elements((rows * cols) as u64));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("c64/matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.sum();
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_c64_matrix_sum_mt(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_mt");

    for &(rows, cols) in &[(100, 100), (1000, 1000), (100, 10000), (10000, 100)] {
        let a = random_complex_matrix(rows, cols);
        group.throughput(Throughput::Elements((rows * cols) as u64));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("c64/matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.sum_mt();
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_f64_matrix_trace(c: &mut Criterion) {
    let mut group = c.benchmark_group("trace");

    for &n in &[16, 64, 256, 1024, 4096] {
        let a = Matrix::<f64>::rand(n, n);
        // Trace visits `n` diagonal elements.
        group.throughput(Throughput::Elements(n as u64));

        let label = format!("{n}x{n}");

        group.bench_with_input(BenchmarkId::new("f64/matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.trace().expect("trace must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_f64_matrix_trace_mt(c: &mut Criterion) {
    let mut group = c.benchmark_group("trace_mt");

    for &n in &[16, 64, 256, 1024, 4096] {
        let a = Matrix::<f64>::rand(n, n);
        group.throughput(Throughput::Elements(n as u64));

        let label = format!("{n}x{n}");

        group.bench_with_input(BenchmarkId::new("f64/matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.trace_mt().expect("trace_mt must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_c64_matrix_trace(c: &mut Criterion) {
    let mut group = c.benchmark_group("trace");

    for &n in &[16, 64, 256, 1024, 4096] {
        let a = random_complex_matrix(n, n);
        group.throughput(Throughput::Elements(n as u64));

        let label = format!("{n}x{n}");

        group.bench_with_input(BenchmarkId::new("c64/matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.trace().expect("trace must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_c64_matrix_trace_mt(c: &mut Criterion) {
    let mut group = c.benchmark_group("trace_mt");

    for &n in &[16, 64, 256, 1024, 4096] {
        let a = random_complex_matrix(n, n);
        group.throughput(Throughput::Elements(n as u64));

        let label = format!("{n}x{n}");

        group.bench_with_input(BenchmarkId::new("c64/matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.trace_mt().expect("trace_mt must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

criterion_group!(
    name = sum_benches;
    config = Criterion::default().sample_size(10);
    targets =
        bench_f64_tensor_sum,
        bench_f64_tensor_sum_mt,
        bench_c64_tensor_sum,
        bench_c64_tensor_sum_mt,
        bench_f64_matrix_sum,
        bench_f64_matrix_sum_mt,
        bench_c64_matrix_sum,
        bench_c64_matrix_sum_mt,
        bench_f64_matrix_trace,
        bench_f64_matrix_trace_mt,
        bench_c64_matrix_trace,
        bench_c64_matrix_trace_mt,
);
