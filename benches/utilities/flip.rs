//! Benchmarks for the `flip` family on [`Tensor`] and [`Matrix`].
//!
//! The single-argument `flip` / `flip_mt` variants are convenience wrappers
//! around `flip_axes` / `flip_axes_mt` (or `flip_rows` / `flip_cols` for
//! `Matrix`); the benchmarks exercise both forms. Sizes span small/medium/
//! large so the rayon crossover is visible in the output.

use std::collections::HashSet;

use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};

use super::bench_utils::{drain_f64, drain_mat_f64, matrix, tensor_from_shape};

// ---------------------------------------------------------------------------
// Tensor::flip / flip_mt (flip every axis)
// ---------------------------------------------------------------------------

fn bench_flip_tensor(c: &mut Criterion) {
    let mut group = c.benchmark_group("flip/tensor_2d_all");
    for &(rows, cols) in &[
        (32usize, 32usize),
        (128, 128),
        (512, 512),
        (1024, 1024),
        (2048, 4096),
    ] {
        let a = tensor_from_shape(&[rows, cols], 1.0);
        let elems = (rows * cols) as u64;
        group.throughput(Throughput::Elements(elems));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("st", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.flip();
                std::hint::black_box(drain_f64(&r));
            });
        });
        group.bench_with_input(BenchmarkId::new("mt", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.flip_mt();
                std::hint::black_box(drain_f64(&r));
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Tensor::flip_axes / flip_axes_mt (flip a subset of axes)
// ---------------------------------------------------------------------------

fn bench_flip_axes_tensor(c: &mut Criterion) {
    let mut group = c.benchmark_group("flip/tensor_3d_axes");
    // Flip just the last two axes, leaving the leading axis untouched.
    let axes: HashSet<usize> = [1usize, 2usize].into_iter().collect();
    for &(d0, d1, d2) in &[
        (8usize, 64usize, 64usize),
        (16, 128, 128),
        (32, 256, 256),
        (64, 256, 256),
    ] {
        let a = tensor_from_shape(&[d0, d1, d2], 1.0);
        let elems = (d0 * d1 * d2) as u64;
        group.throughput(Throughput::Elements(elems));

        let label = format!("{d0}x{d1}x{d2}");

        group.bench_with_input(BenchmarkId::new("st", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.flip_axes(&axes).expect("flip_axes must succeed");
                std::hint::black_box(drain_f64(&r));
            });
        });
        group.bench_with_input(BenchmarkId::new("mt", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.flip_axes_mt(&axes).expect("flip_axes_mt must succeed");
                std::hint::black_box(drain_f64(&r));
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Matrix::flip_rows / flip_rows_mt
// ---------------------------------------------------------------------------

fn bench_flip_rows_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("flip/matrix_rows");
    for &(rows, cols) in &[
        (32usize, 32usize),
        (128, 128),
        (512, 512),
        (1024, 1024),
        (2048, 4096),
    ] {
        let a = matrix(rows, cols, 1.0);
        group.throughput(Throughput::Elements((rows * cols) as u64));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("st", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.flip_rows();
                std::hint::black_box(drain_mat_f64(&r));
            });
        });
        group.bench_with_input(BenchmarkId::new("mt", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.flip_rows_mt();
                std::hint::black_box(drain_mat_f64(&r));
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Matrix::flip_cols / flip_cols_mt
// ---------------------------------------------------------------------------

fn bench_flip_cols_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("flip/matrix_cols");
    for &(rows, cols) in &[
        (32usize, 32usize),
        (128, 128),
        (512, 512),
        (1024, 1024),
        (2048, 4096),
    ] {
        let a = matrix(rows, cols, 1.0);
        group.throughput(Throughput::Elements((rows * cols) as u64));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("st", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.flip_cols();
                std::hint::black_box(drain_mat_f64(&r));
            });
        });
        group.bench_with_input(BenchmarkId::new("mt", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.flip_cols_mt();
                std::hint::black_box(drain_mat_f64(&r));
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Matrix::flip / flip_mt (flip both axes)
// ---------------------------------------------------------------------------

fn bench_flip_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("flip/matrix_all");
    for &(rows, cols) in &[
        (32usize, 32usize),
        (128, 128),
        (512, 512),
        (1024, 1024),
        (2048, 4096),
    ] {
        let a = matrix(rows, cols, 1.0);
        group.throughput(Throughput::Elements((rows * cols) as u64));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("st", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.flip();
                std::hint::black_box(drain_mat_f64(&r));
            });
        });
        group.bench_with_input(BenchmarkId::new("mt", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.flip_mt();
                std::hint::black_box(drain_mat_f64(&r));
            });
        });
    }
    group.finish();
}

criterion_group!(
    name = flip_benches;
    config = Criterion::default().sample_size(10);
    targets =
        bench_flip_tensor,
        bench_flip_axes_tensor,
        bench_flip_rows_matrix,
        bench_flip_cols_matrix,
        bench_flip_matrix,
);
// `criterion_group!` above declares `pub fn flip_benches()`.
