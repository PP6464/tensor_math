//! Benchmarks for `corr` / `conv` (and the matrix variants `corr_cols`,
//! `corr_rows`, `corr_axes`) on `Matrix<f64>` / `Matrix<Complex64>` and
//! `Tensor<f64>` / `Tensor<Complex64>`.
//!
//! Both the single-threaded and multi-threaded (`_mt`) variants are exercised.

use std::collections::HashSet;
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
fn bench_f64_matrix_corr(c: &mut Criterion) {
    let mut group = c.benchmark_group("corr");

    for &(rows, k_rows, cols, k_cols) in &[
        (32, 3, 32, 3),
        (64, 3, 64, 3),
        (128, 5, 128, 5),
        (256, 7, 256, 7),
    ] {
        let a = Matrix::<f64>::rand(rows, cols);
        let b = Matrix::<f64>::rand(k_rows, k_cols);
        // (rows + k_rows - 1) * (cols + k_cols - 1) output elements.
        let out = (rows + k_rows - 1) * (cols + k_cols - 1);
        group.throughput(Throughput::Elements(out as u64));

        let label = format!("{rows}x{cols}*{k_rows}x{k_cols}");

        group.bench_with_input(BenchmarkId::new("f64/matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.corr(&b).expect("corr must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_f64_matrix_corr_mt(c: &mut Criterion) {
    let mut group = c.benchmark_group("corr_mt");

    for &(rows, k_rows, cols, k_cols) in &[
        (32, 3, 32, 3),
        (64, 3, 64, 3),
        (128, 5, 128, 5),
        (256, 7, 256, 7),
    ] {
        let a = Matrix::<f64>::rand(rows, cols);
        let b = Matrix::<f64>::rand(k_rows, k_cols);
        let out = (rows + k_rows - 1) * (cols + k_cols - 1);
        group.throughput(Throughput::Elements(out as u64));

        let label = format!("{rows}x{cols}*{k_rows}x{k_cols}");

        group.bench_with_input(BenchmarkId::new("f64/matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.corr_mt(&b).expect("corr_mt must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_f64_matrix_conv(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv");

    for &(rows, k_rows, cols, k_cols) in &[
        (32, 3, 32, 3),
        (64, 3, 64, 3),
        (128, 5, 128, 5),
        (256, 7, 256, 7),
    ] {
        let a = Matrix::<f64>::rand(rows, cols);
        let b = Matrix::<f64>::rand(k_rows, k_cols);
        let out = (rows + k_rows - 1) * (cols + k_cols - 1);
        group.throughput(Throughput::Elements(out as u64));

        let label = format!("{rows}x{cols}*{k_rows}x{k_cols}");

        group.bench_with_input(BenchmarkId::new("f64/matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.conv(&b).expect("conv must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_f64_matrix_conv_mt(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv_mt");

    for &(rows, k_rows, cols, k_cols) in &[
        (32, 3, 32, 3),
        (64, 3, 64, 3),
        (128, 5, 128, 5),
        (256, 7, 256, 7),
    ] {
        let a = Matrix::<f64>::rand(rows, cols);
        let b = Matrix::<f64>::rand(k_rows, k_cols);
        let out = (rows + k_rows - 1) * (cols + k_cols - 1);
        group.throughput(Throughput::Elements(out as u64));

        let label = format!("{rows}x{cols}*{k_rows}x{k_cols}");

        group.bench_with_input(BenchmarkId::new("f64/matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.conv_mt(&b).expect("conv_mt must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_c64_matrix_corr(c: &mut Criterion) {
    let mut group = c.benchmark_group("corr");

    for &(rows, k_rows, cols, k_cols) in &[
        (32, 3, 32, 3),
        (64, 3, 64, 3),
        (128, 5, 128, 5),
        (256, 7, 256, 7),
    ] {
        let a = random_complex_matrix(rows, cols);
        let b = random_complex_matrix(k_rows, k_cols);
        let out = (rows + k_rows - 1) * (cols + k_cols - 1);
        group.throughput(Throughput::Elements(out as u64));

        let label = format!("{rows}x{cols}*{k_rows}x{k_cols}");

        group.bench_with_input(BenchmarkId::new("c64/matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.corr(&b).expect("corr must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_c64_matrix_corr_mt(c: &mut Criterion) {
    let mut group = c.benchmark_group("corr_mt");

    for &(rows, k_rows, cols, k_cols) in &[
        (32, 3, 32, 3),
        (64, 3, 64, 3),
        (128, 5, 128, 5),
        (256, 7, 256, 7),
    ] {
        let a = random_complex_matrix(rows, cols);
        let b = random_complex_matrix(k_rows, k_cols);
        let out = (rows + k_rows - 1) * (cols + k_cols - 1);
        group.throughput(Throughput::Elements(out as u64));

        let label = format!("{rows}x{cols}*{k_rows}x{k_cols}");

        group.bench_with_input(BenchmarkId::new("c64/matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.corr_mt(&b).expect("corr_mt must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_c64_matrix_conv(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv");

    for &(rows, k_rows, cols, k_cols) in &[
        (32, 3, 32, 3),
        (64, 3, 64, 3),
        (128, 5, 128, 5),
        (256, 7, 256, 7),
    ] {
        let a = random_complex_matrix(rows, cols);
        let b = random_complex_matrix(k_rows, k_cols);
        let out = (rows + k_rows - 1) * (cols + k_cols - 1);
        group.throughput(Throughput::Elements(out as u64));

        let label = format!("{rows}x{cols}*{k_rows}x{k_cols}");

        group.bench_with_input(BenchmarkId::new("c64/matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.conv(&b).expect("conv must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_c64_matrix_conv_mt(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv_mt");

    for &(rows, k_rows, cols, k_cols) in &[
        (32, 3, 32, 3),
        (64, 3, 64, 3),
        (128, 5, 128, 5),
        (256, 7, 256, 7),
    ] {
        let a = random_complex_matrix(rows, cols);
        let b = random_complex_matrix(k_rows, k_cols);
        let out = (rows + k_rows - 1) * (cols + k_cols - 1);
        group.throughput(Throughput::Elements(out as u64));

        let label = format!("{rows}x{cols}*{k_rows}x{k_cols}");

        group.bench_with_input(BenchmarkId::new("c64/matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.conv_mt(&b).expect("conv_mt must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_f64_matrix_corr_cols(c: &mut Criterion) {
    let mut group = c.benchmark_group("corr_cols");

    for &(rows, k_rows, cols) in &[(32, 3, 16), (64, 5, 32), (128, 7, 32), (256, 9, 32)] {
        let a = Matrix::<f64>::rand(rows, cols);
        let b = Matrix::<f64>::rand(k_rows, cols);
        let out = (rows + k_rows - 1) * cols;
        group.throughput(Throughput::Elements(out as u64));

        let label = format!("{rows}x{cols}*{k_rows}");

        group.bench_with_input(BenchmarkId::new("f64/matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.corr_cols(&b).expect("corr_cols must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_f64_matrix_corr_cols_mt(c: &mut Criterion) {
    let mut group = c.benchmark_group("corr_cols_mt");

    for &(rows, k_rows, cols) in &[(32, 3, 16), (64, 5, 32), (128, 7, 32), (256, 9, 32)] {
        let a = Matrix::<f64>::rand(rows, cols);
        let b = Matrix::<f64>::rand(k_rows, cols);
        let out = (rows + k_rows - 1) * cols;
        group.throughput(Throughput::Elements(out as u64));

        let label = format!("{rows}x{cols}*{k_rows}");

        group.bench_with_input(BenchmarkId::new("f64/matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a
                    .corr_cols_mt(&b)
                    .expect("corr_cols_mt must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_c64_matrix_corr_cols(c: &mut Criterion) {
    let mut group = c.benchmark_group("corr_cols");

    for &(rows, k_rows, cols) in &[(32, 3, 16), (64, 5, 32), (128, 7, 32), (256, 9, 32)] {
        let a = random_complex_matrix(rows, cols);
        let b = random_complex_matrix(k_rows, cols);
        let out = (rows + k_rows - 1) * cols;
        group.throughput(Throughput::Elements(out as u64));

        let label = format!("{rows}x{cols}*{k_rows}");

        group.bench_with_input(BenchmarkId::new("c64/matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.corr_cols(&b).expect("corr_cols must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_c64_matrix_corr_cols_mt(c: &mut Criterion) {
    let mut group = c.benchmark_group("corr_cols_mt");

    for &(rows, k_rows, cols) in &[(32, 3, 16), (64, 5, 32), (128, 7, 32), (256, 9, 32)] {
        let a = random_complex_matrix(rows, cols);
        let b = random_complex_matrix(k_rows, cols);
        let out = (rows + k_rows - 1) * cols;
        group.throughput(Throughput::Elements(out as u64));

        let label = format!("{rows}x{cols}*{k_rows}");

        group.bench_with_input(BenchmarkId::new("c64/matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a
                    .corr_cols_mt(&b)
                    .expect("corr_cols_mt must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_f64_matrix_corr_rows(c: &mut Criterion) {
    let mut group = c.benchmark_group("corr_rows");

    for &(rows, cols, k_cols) in &[(16, 32, 3), (32, 64, 5), (32, 128, 7), (32, 256, 9)] {
        let a = Matrix::<f64>::rand(rows, cols);
        let b = Matrix::<f64>::rand(rows, k_cols);
        let out = rows * (cols + k_cols - 1);
        group.throughput(Throughput::Elements(out as u64));

        let label = format!("{rows}x{cols}*{k_cols}");

        group.bench_with_input(BenchmarkId::new("f64/matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.corr_rows(&b).expect("corr_rows must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_f64_matrix_corr_rows_mt(c: &mut Criterion) {
    let mut group = c.benchmark_group("corr_rows_mt");

    for &(rows, cols, k_cols) in &[(16, 32, 3), (32, 64, 5), (32, 128, 7), (32, 256, 9)] {
        let a = Matrix::<f64>::rand(rows, cols);
        let b = Matrix::<f64>::rand(rows, k_cols);
        let out = rows * (cols + k_cols - 1);
        group.throughput(Throughput::Elements(out as u64));

        let label = format!("{rows}x{cols}*{k_cols}");

        group.bench_with_input(BenchmarkId::new("f64/matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a
                    .corr_rows_mt(&b)
                    .expect("corr_rows_mt must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_c64_matrix_corr_rows(c: &mut Criterion) {
    let mut group = c.benchmark_group("corr_rows");

    for &(rows, cols, k_cols) in &[(16, 32, 3), (32, 64, 5), (32, 128, 7), (32, 256, 9)] {
        let a = random_complex_matrix(rows, cols);
        let b = random_complex_matrix(rows, k_cols);
        let out = rows * (cols + k_cols - 1);
        group.throughput(Throughput::Elements(out as u64));

        let label = format!("{rows}x{cols}*{k_cols}");

        group.bench_with_input(BenchmarkId::new("c64/matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.corr_rows(&b).expect("corr_rows must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_c64_matrix_corr_rows_mt(c: &mut Criterion) {
    let mut group = c.benchmark_group("corr_rows_mt");

    for &(rows, cols, k_cols) in &[(16, 32, 3), (32, 64, 5), (32, 128, 7), (32, 256, 9)] {
        let a = random_complex_matrix(rows, cols);
        let b = random_complex_matrix(rows, k_cols);
        let out = rows * (cols + k_cols - 1);
        group.throughput(Throughput::Elements(out as u64));

        let label = format!("{rows}x{cols}*{k_cols}");

        group.bench_with_input(BenchmarkId::new("c64/matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a
                    .corr_rows_mt(&b)
                    .expect("corr_rows_mt must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_f64_tensor_corr(c: &mut Criterion) {
    let mut group = c.benchmark_group("corr");

    for (a_shape, b_shape) in &[
        (shape![32, 32], shape![3, 3]),
        (shape![64, 64], shape![3, 3]),
        (shape![128, 128], shape![5, 5]),
        (shape![16, 16, 16], shape![3, 3, 3]),
    ] {
        let a = Tensor::<f64>::rand(a_shape);
        let b = Tensor::<f64>::rand(b_shape);
        let out = a.shape().element_count() / b.shape().element_count()
            * (0..a_shape.rank())
                .map(|i| a_shape[i] + b_shape[i] - 1)
                .product::<usize>();
        group.throughput(Throughput::Elements(out as u64));

        let label = format!("{}*{}", label_shape(a_shape), label_shape(b_shape));

        group.bench_with_input(BenchmarkId::new("f64/tensor", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.corr(&b).expect("corr must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_f64_tensor_corr_mt(c: &mut Criterion) {
    let mut group = c.benchmark_group("corr_mt");

    for (a_shape, b_shape) in &[
        (shape![32, 32], shape![3, 3]),
        (shape![64, 64], shape![3, 3]),
        (shape![128, 128], shape![5, 5]),
        (shape![16, 16, 16], shape![3, 3, 3]),
    ] {
        let a = Tensor::<f64>::rand(a_shape);
        let b = Tensor::<f64>::rand(b_shape);
        let out = a.shape().element_count() / b.shape().element_count()
            * (0..a_shape.rank())
                .map(|i| a_shape[i] + b_shape[i] - 1)
                .product::<usize>();
        group.throughput(Throughput::Elements(out as u64));

        let label = format!("{}*{}", label_shape(a_shape), label_shape(b_shape));

        group.bench_with_input(BenchmarkId::new("f64/tensor", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.corr_mt(&b).expect("corr_mt must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_f64_tensor_conv(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv");

    for (a_shape, b_shape) in &[
        (shape![32, 32], shape![3, 3]),
        (shape![64, 64], shape![3, 3]),
        (shape![128, 128], shape![5, 5]),
        (shape![16, 16, 16], shape![3, 3, 3]),
    ] {
        let a = Tensor::<f64>::rand(a_shape);
        let b = Tensor::<f64>::rand(b_shape);
        let out = a.shape().element_count() / b.shape().element_count()
            * (0..a_shape.rank())
                .map(|i| a_shape[i] + b_shape[i] - 1)
                .product::<usize>();
        group.throughput(Throughput::Elements(out as u64));

        let label = format!("{}*{}", label_shape(a_shape), label_shape(b_shape));

        group.bench_with_input(BenchmarkId::new("f64/tensor", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.conv(&b).expect("conv must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_f64_tensor_conv_mt(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv_mt");

    for (a_shape, b_shape) in &[
        (shape![32, 32], shape![3, 3]),
        (shape![64, 64], shape![3, 3]),
        (shape![128, 128], shape![5, 5]),
        (shape![16, 16, 16], shape![3, 3, 3]),
    ] {
        let a = Tensor::<f64>::rand(a_shape);
        let b = Tensor::<f64>::rand(b_shape);
        let out = a.shape().element_count() / b.shape().element_count()
            * (0..a_shape.rank())
                .map(|i| a_shape[i] + b_shape[i] - 1)
                .product::<usize>();
        group.throughput(Throughput::Elements(out as u64));

        let label = format!("{}*{}", label_shape(a_shape), label_shape(b_shape));

        group.bench_with_input(BenchmarkId::new("f64/tensor", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.conv_mt(&b).expect("conv_mt must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_c64_tensor_corr(c: &mut Criterion) {
    let mut group = c.benchmark_group("corr");

    for (a_shape, b_shape) in &[
        (shape![32, 32], shape![3, 3]),
        (shape![64, 64], shape![3, 3]),
        (shape![128, 128], shape![5, 5]),
        (shape![16, 16, 16], shape![3, 3, 3]),
    ] {
        let a = random_complex_tensor(a_shape);
        let b = random_complex_tensor(b_shape);
        let out = a.shape().element_count() / b.shape().element_count()
            * (0..a_shape.rank())
                .map(|i| a_shape[i] + b_shape[i] - 1)
                .product::<usize>();
        group.throughput(Throughput::Elements(out as u64));

        let label = format!("{}*{}", label_shape(a_shape), label_shape(b_shape));

        group.bench_with_input(BenchmarkId::new("c64/tensor", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.corr(&b).expect("corr must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_c64_tensor_corr_mt(c: &mut Criterion) {
    let mut group = c.benchmark_group("corr_mt");

    for (a_shape, b_shape) in &[
        (shape![32, 32], shape![3, 3]),
        (shape![64, 64], shape![3, 3]),
        (shape![128, 128], shape![5, 5]),
        (shape![16, 16, 16], shape![3, 3, 3]),
    ] {
        let a = random_complex_tensor(a_shape);
        let b = random_complex_tensor(b_shape);
        let out = a.shape().element_count() / b.shape().element_count()
            * (0..a_shape.rank())
                .map(|i| a_shape[i] + b_shape[i] - 1)
                .product::<usize>();
        group.throughput(Throughput::Elements(out as u64));

        let label = format!("{}*{}", label_shape(a_shape), label_shape(b_shape));

        group.bench_with_input(BenchmarkId::new("c64/tensor", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.corr_mt(&b).expect("corr_mt must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_c64_tensor_conv(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv");

    for (a_shape, b_shape) in &[
        (shape![32, 32], shape![3, 3]),
        (shape![64, 64], shape![3, 3]),
        (shape![128, 128], shape![5, 5]),
        (shape![16, 16, 16], shape![3, 3, 3]),
    ] {
        let a = random_complex_tensor(a_shape);
        let b = random_complex_tensor(b_shape);
        let out = a.shape().element_count() / b.shape().element_count()
            * (0..a_shape.rank())
                .map(|i| a_shape[i] + b_shape[i] - 1)
                .product::<usize>();
        group.throughput(Throughput::Elements(out as u64));

        let label = format!("{}*{}", label_shape(a_shape), label_shape(b_shape));

        group.bench_with_input(BenchmarkId::new("c64/tensor", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.conv(&b).expect("conv must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_c64_tensor_conv_mt(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv_mt");

    for (a_shape, b_shape) in &[
        (shape![32, 32], shape![3, 3]),
        (shape![64, 64], shape![3, 3]),
        (shape![128, 128], shape![5, 5]),
        (shape![16, 16, 16], shape![3, 3, 3]),
    ] {
        let a = random_complex_tensor(a_shape);
        let b = random_complex_tensor(b_shape);
        let out = a.shape().element_count() / b.shape().element_count()
            * (0..a_shape.rank())
                .map(|i| a_shape[i] + b_shape[i] - 1)
                .product::<usize>();
        group.throughput(Throughput::Elements(out as u64));

        let label = format!("{}*{}", label_shape(a_shape), label_shape(b_shape));

        group.bench_with_input(BenchmarkId::new("c64/tensor", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.conv_mt(&b).expect("conv_mt must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn axes_last_only(rank: usize) -> HashSet<usize> {
    let mut axes = HashSet::new();
    axes.insert(rank - 1);
    axes
}

fn bench_f64_tensor_corr_axes(c: &mut Criterion) {
    let mut group = c.benchmark_group("corr_axes");

    for (a_shape, b_shape) in &[
        (shape![32, 32, 32], shape![32, 32, 5]),
        (shape![64, 64, 64], shape![64, 64, 5]),
        (shape![16, 16, 128], shape![16, 16, 7]),
    ] {
        let a = Tensor::<f64>::rand(a_shape);
        let b = Tensor::<f64>::rand(b_shape);
        let rank = a_shape.rank();
        let axes = axes_last_only(rank);
        // Output elements = a.shape[0..rank-1].product() * (a.shape[rank-1] + b.shape[rank-1] - 1).
        let mut out = 1usize;
        for i in 0..rank - 1 {
            out *= a_shape[i];
        }
        out *= a_shape[rank - 1] + b_shape[rank - 1] - 1;
        group.throughput(Throughput::Elements(out as u64));

        let label = format!("{}*{}/axes={{{}}}", label_shape(a_shape), label_shape(b_shape), rank - 1);

        group.bench_with_input(BenchmarkId::new("f64/tensor", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.corr_axes(&b, &axes).expect("corr_axes must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_f64_tensor_corr_axes_mt(c: &mut Criterion) {
    let mut group = c.benchmark_group("corr_axes_mt");

    for (a_shape, b_shape) in &[
        (shape![32, 32, 32], shape![32, 32, 5]),
        (shape![64, 64, 64], shape![64, 64, 5]),
        (shape![16, 16, 128], shape![16, 16, 7]),
    ] {
        let a = Tensor::<f64>::rand(a_shape);
        let b = Tensor::<f64>::rand(b_shape);
        let rank = a_shape.rank();
        let axes = axes_last_only(rank);
        let mut out = 1usize;
        for i in 0..rank - 1 {
            out *= a_shape[i];
        }
        out *= a_shape[rank - 1] + b_shape[rank - 1] - 1;
        group.throughput(Throughput::Elements(out as u64));

        let label = format!("{}*{}/axes={{{}}}", label_shape(a_shape), label_shape(b_shape), rank - 1);

        group.bench_with_input(BenchmarkId::new("f64/tensor", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a
                    .corr_axes_mt(&b, &axes)
                    .expect("corr_axes_mt must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_c64_tensor_corr_axes(c: &mut Criterion) {
    let mut group = c.benchmark_group("corr_axes");

    for (a_shape, b_shape) in &[
        (shape![32, 32, 32], shape![32, 32, 5]),
        (shape![64, 64, 64], shape![64, 64, 5]),
        (shape![16, 16, 128], shape![16, 16, 7]),
    ] {
        let a = random_complex_tensor(a_shape);
        let b = random_complex_tensor(b_shape);
        let rank = a_shape.rank();
        let axes = axes_last_only(rank);
        let mut out = 1usize;
        for i in 0..rank - 1 {
            out *= a_shape[i];
        }
        out *= a_shape[rank - 1] + b_shape[rank - 1] - 1;
        group.throughput(Throughput::Elements(out as u64));

        let label = format!("{}*{}/axes={{{}}}", label_shape(a_shape), label_shape(b_shape), rank - 1);

        group.bench_with_input(BenchmarkId::new("c64/tensor", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.corr_axes(&b, &axes).expect("corr_axes must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_c64_tensor_corr_axes_mt(c: &mut Criterion) {
    let mut group = c.benchmark_group("corr_axes_mt");

    for (a_shape, b_shape) in &[
        (shape![32, 32, 32], shape![32, 32, 5]),
        (shape![64, 64, 64], shape![64, 64, 5]),
        (shape![16, 16, 128], shape![16, 16, 7]),
    ] {
        let a = random_complex_tensor(a_shape);
        let b = random_complex_tensor(b_shape);
        let rank = a_shape.rank();
        let axes = axes_last_only(rank);
        let mut out = 1usize;
        for i in 0..rank - 1 {
            out *= a_shape[i];
        }
        out *= a_shape[rank - 1] + b_shape[rank - 1] - 1;
        group.throughput(Throughput::Elements(out as u64));

        let label = format!("{}*{}/axes={{{}}}", label_shape(a_shape), label_shape(b_shape), rank - 1);

        group.bench_with_input(BenchmarkId::new("c64/tensor", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a
                    .corr_axes_mt(&b, &axes)
                    .expect("corr_axes_mt must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

criterion_group!(
    name = corr_conv_benches;
    config = Criterion::default().sample_size(10);
    targets =
        bench_f64_matrix_corr,
        bench_f64_matrix_corr_mt,
        bench_f64_matrix_conv,
        bench_f64_matrix_conv_mt,
        bench_c64_matrix_corr,
        bench_c64_matrix_corr_mt,
        bench_c64_matrix_conv,
        bench_c64_matrix_conv_mt,
        bench_f64_matrix_corr_cols,
        bench_f64_matrix_corr_cols_mt,
        bench_c64_matrix_corr_cols,
        bench_c64_matrix_corr_cols_mt,
        bench_f64_matrix_corr_rows,
        bench_f64_matrix_corr_rows_mt,
        bench_c64_matrix_corr_rows,
        bench_c64_matrix_corr_rows_mt,
        bench_f64_tensor_corr,
        bench_f64_tensor_corr_mt,
        bench_f64_tensor_conv,
        bench_f64_tensor_conv_mt,
        bench_c64_tensor_corr,
        bench_c64_tensor_corr_mt,
        bench_c64_tensor_conv,
        bench_c64_tensor_conv_mt,
        bench_f64_tensor_corr_axes,
        bench_f64_tensor_corr_axes_mt,
        bench_c64_tensor_corr_axes,
        bench_c64_tensor_corr_axes_mt,
);
