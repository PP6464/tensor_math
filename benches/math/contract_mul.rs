//! Benchmarks for `contract_mul` and `mat_mul` (matrix multiplication) on
//! `Matrix<f64>` / `Matrix<Complex64>` and `Tensor<f64>` / `Tensor<Complex64>`.
//!
//! Both the single-threaded and multi-threaded (`_mt`) variants are exercised.

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

fn bench_f64_matrix_contract_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("contract_mul");

    for &(rows, inner, cols) in &[
        (32, 32, 32),
        (128, 128, 128),
        (512, 512, 512),
        (1024, 1024, 1024),
    ] {
        let a = Matrix::<f64>::rand(rows, inner);
        let b = Matrix::<f64>::rand(inner, cols);
        // One fused multiply-add per output element.
        group.throughput(Throughput::Elements((2 * rows * inner * cols) as u64));

        let label = format!("{rows}x{inner}x{cols}");

        group.bench_with_input(BenchmarkId::new("f64/matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.contract_mul(&b).expect("contract_mul must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_f64_matrix_contract_mul_mt(c: &mut Criterion) {
    let mut group = c.benchmark_group("contract_mul_mt");

    for &(rows, inner, cols) in &[
        (32, 32, 32),
        (128, 128, 128),
        (512, 512, 512),
        (1024, 1024, 1024),
    ] {
        let a = Matrix::<f64>::rand(rows, inner);
        let b = Matrix::<f64>::rand(inner, cols);
        group.throughput(Throughput::Elements((2 * rows * inner * cols) as u64));

        let label = format!("{rows}x{inner}x{cols}");

        group.bench_with_input(BenchmarkId::new("f64/matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a
                    .contract_mul_mt(&b)
                    .expect("contract_mul_mt must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_f64_matrix_mat_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("mat_mul");

    for &(rows, inner, cols) in &[
        (32, 32, 32),
        (128, 128, 128),
        (512, 512, 512),
        (1024, 1024, 1024),
    ] {
        let a = Matrix::<f64>::rand(rows, inner);
        let b = Matrix::<f64>::rand(inner, cols);
        group.throughput(Throughput::Elements((2 * rows * inner * cols) as u64));

        let label = format!("{rows}x{inner}x{cols}");

        group.bench_with_input(BenchmarkId::new("f64/matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.mat_mul(&b).expect("mat_mul must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_f64_matrix_mat_mul_mt(c: &mut Criterion) {
    let mut group = c.benchmark_group("mat_mul_mt");

    for &(rows, inner, cols) in &[
        (32, 32, 32),
        (128, 128, 128),
        (512, 512, 512),
        (1024, 1024, 1024),
    ] {
        let a = Matrix::<f64>::rand(rows, inner);
        let b = Matrix::<f64>::rand(inner, cols);
        group.throughput(Throughput::Elements((2 * rows * inner * cols) as u64));

        let label = format!("{rows}x{inner}x{cols}");

        group.bench_with_input(BenchmarkId::new("f64/matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.mat_mul_mt(&b).expect("mat_mul_mt must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_c64_matrix_contract_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("contract_mul");

    for &(rows, inner, cols) in &[
        (32, 32, 32),
        (128, 128, 128),
        (512, 512, 512),
        (1024, 1024, 1024),
    ] {
        let a = random_complex_matrix(rows, inner);
        let b = random_complex_matrix(inner, cols);
        // Each complex multiply is 4 real multiplies + 2 real adds, and the
        // dot-product reduction adds another inner contribution per element.
        group.throughput(Throughput::Elements((2 * rows * inner * cols) as u64));

        let label = format!("{rows}x{inner}x{cols}");

        group.bench_with_input(BenchmarkId::new("c64/matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.contract_mul(&b).expect("contract_mul must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_c64_matrix_contract_mul_mt(c: &mut Criterion) {
    let mut group = c.benchmark_group("contract_mul_mt");

    for &(rows, inner, cols) in &[
        (32, 32, 32),
        (128, 128, 128),
        (512, 512, 512),
        (1024, 1024, 1024),
    ] {
        let a = random_complex_matrix(rows, inner);
        let b = random_complex_matrix(inner, cols);
        group.throughput(Throughput::Elements((2 * rows * inner * cols) as u64));

        let label = format!("{rows}x{inner}x{cols}");

        group.bench_with_input(BenchmarkId::new("c64/matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a
                    .contract_mul_mt(&b)
                    .expect("contract_mul_mt must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_c64_matrix_mat_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("mat_mul");

    for &(rows, inner, cols) in &[
        (32, 32, 32),
        (128, 128, 128),
        (512, 512, 512),
        (1024, 1024, 1024),
    ] {
        let a = random_complex_matrix(rows, inner);
        let b = random_complex_matrix(inner, cols);
        group.throughput(Throughput::Elements((2 * rows * inner * cols) as u64));

        let label = format!("{rows}x{inner}x{cols}");

        group.bench_with_input(BenchmarkId::new("c64/matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.mat_mul(&b).expect("mat_mul must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_c64_matrix_mat_mul_mt(c: &mut Criterion) {
    let mut group = c.benchmark_group("mat_mul_mt");

    for &(rows, inner, cols) in &[
        (32, 32, 32),
        (128, 128, 128),
        (512, 512, 512),
        (1024, 1024, 1024),
    ] {
        let a = random_complex_matrix(rows, inner);
        let b = random_complex_matrix(inner, cols);
        group.throughput(Throughput::Elements((2 * rows * inner * cols) as u64));

        let label = format!("{rows}x{inner}x{cols}");

        group.bench_with_input(BenchmarkId::new("c64/matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.mat_mul_mt(&b).expect("mat_mul_mt must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_f64_tensor_contract_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("contract_mul");

    for shape in &[
        shape![32, 32, 32],
        shape![128, 128, 128],
        shape![512, 512, 512],
        shape![1024, 1024, 1024],
    ] {
        let inner = shape[shape.rank() - 1];
        let a = Tensor::<f64>::rand(shape);
        let b = Tensor::<f64>::rand(&shape![inner, 32, 32]);
        // Output has shape a.shape[..rank-1] x 32 x 32.
        let out_elems = a.shape().element_count() / inner * 32 * 32;
        group.throughput(Throughput::Elements((2 * out_elems * inner) as u64));

        let label = format!("{}-x{}x32", label_shape(&a.shape()), inner);

        group.bench_with_input(BenchmarkId::new("f64/tensor", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.contract_mul(&b).expect("contract_mul must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_f64_tensor_contract_mul_mt(c: &mut Criterion) {
    let mut group = c.benchmark_group("contract_mul_mt");

    for shape in &[
        shape![32, 32, 32],
        shape![128, 128, 128],
        shape![512, 512, 512],
        shape![1024, 1024, 1024],
    ] {
        let inner = shape[shape.rank() - 1];
        let a = Tensor::<f64>::rand(shape);
        let b = Tensor::<f64>::rand(&shape![inner, 32, 32]);
        let out_elems = a.shape().element_count() / inner * 32 * 32;
        group.throughput(Throughput::Elements((2 * out_elems * inner) as u64));

        let label = format!("{}-x{}x32", label_shape(&a.shape()), inner);

        group.bench_with_input(BenchmarkId::new("f64/tensor", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a
                    .contract_mul_mt(&b)
                    .expect("contract_mul_mt must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_c64_tensor_contract_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("contract_mul");

    for shape in &[
        shape![32, 32, 32],
        shape![128, 128, 128],
        shape![512, 512, 512],
        shape![1024, 1024, 1024],
    ] {
        let inner = shape[shape.rank() - 1];
        let a = random_complex_tensor(shape);
        let b = random_complex_tensor(&shape![inner, 32, 32]);
        let out_elems = a.shape().element_count() / inner * 32 * 32;
        group.throughput(Throughput::Elements((2 * out_elems * inner) as u64));

        let label = format!("{}-x{}x32", label_shape(&a.shape()), inner);

        group.bench_with_input(BenchmarkId::new("c64/tensor", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.contract_mul(&b).expect("contract_mul must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_c64_tensor_contract_mul_mt(c: &mut Criterion) {
    let mut group = c.benchmark_group("contract_mul_mt");

    for shape in &[
        shape![32, 32, 32],
        shape![128, 128, 128],
        shape![512, 512, 512],
        shape![1024, 1024, 1024],
    ] {
        let inner = shape[shape.rank() - 1];
        let a = random_complex_tensor(shape);
        let b = random_complex_tensor(&shape![inner, 32, 32]);
        let out_elems = a.shape().element_count() / inner * 32 * 32;
        group.throughput(Throughput::Elements((2 * out_elems * inner) as u64));

        let label = format!("{}-x{}x32", label_shape(&a.shape()), inner);

        group.bench_with_input(BenchmarkId::new("c64/tensor", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a
                    .contract_mul_mt(&b)
                    .expect("contract_mul_mt must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

criterion_group!(
    name = contract_mul_benches;
    config = Criterion::default().sample_size(10);
    targets =
        bench_f64_matrix_contract_mul,
        bench_f64_matrix_contract_mul_mt,
        bench_f64_matrix_mat_mul,
        bench_f64_matrix_mat_mul_mt,
        bench_c64_matrix_contract_mul,
        bench_c64_matrix_contract_mul_mt,
        bench_c64_matrix_mat_mul,
        bench_c64_matrix_mat_mul_mt,
        bench_f64_tensor_contract_mul,
        bench_f64_tensor_contract_mul_mt,
        bench_c64_tensor_contract_mul,
        bench_c64_tensor_contract_mul_mt,
);
