//! Benchmarks for the `slice` family on [`Tensor`] and [`Matrix`].
//!
//! Slicing copies the requested sub-rectangle into a fresh tensor/matrix
//! (immutable `slice`) or hands out a mutable view (`slice_mut`). The
//! interesting comparison is between full-size and partial slices, and
//! between copying and mutating in place.

use std::ops::Range;

use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};

use super::bench_utils::{drain_f64, drain_mat_f64, matrix, tensor_from_shape};

// ---------------------------------------------------------------------------
// Tensor::slice
// ---------------------------------------------------------------------------
//
// We slice a 2-D tensor along both axes. Sizes span small/medium/large and
// the slice ratio is held at roughly 1/8th per axis to give a meaningful
// region without making the result smaller than the work of setting up the
// benchmark.

fn bench_slice_tensor(c: &mut Criterion) {
    let mut group = c.benchmark_group("slice/tensor_2d");
    for &(rows, cols) in &[
        (128usize, 128usize),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ] {
        let a = tensor_from_shape(&[rows, cols], 1.0);
        let r0 = rows / 4..(3 * rows) / 4;
        let c0 = cols / 4..(3 * cols) / 4;
        let elems = ((3 * rows / 4 - rows / 4) * (3 * cols / 4 - cols / 4)) as u64;
        group.throughput(Throughput::Elements(elems));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("st", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.slice(&[r0.clone(), c0.clone()]).expect("slice must succeed");
                std::hint::black_box(drain_f64(&r));
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Tensor::slice_mut (touch every element of the view so it cannot be elided)
// ---------------------------------------------------------------------------

fn bench_slice_mut_tensor(c: &mut Criterion) {
    let mut group = c.benchmark_group("slice_mut/tensor_2d");
    for &(rows, cols) in &[
        (128usize, 128usize),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ] {
        let elems = ((3 * rows / 4 - rows / 4) * (3 * cols / 4 - cols / 4)) as u64;
        group.throughput(Throughput::Elements(elems));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("st", &label), &label, |bench, _| {
            bench.iter(|| {
                let mut a = tensor_from_shape(&[rows, cols], 1.0);
                let view = a
                    .slice_mut(&[
                        rows / 4..(3 * rows) / 4,
                        cols / 4..(3 * cols) / 4,
                    ])
                    .expect("slice_mut must succeed");
                // Read every element via Index to make sure the view is
                // observably used; the result is fed to `black_box`.
                let mut acc: f64 = 0.0;
                for i in 0..view.shape()[0] {
                    for j in 0..view.shape()[1] {
                        if let Some(v) = view.get(&[i, j]) {
                            acc += *v;
                        }
                    }
                }
                std::hint::black_box(acc);
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Matrix::slice
// ---------------------------------------------------------------------------

fn bench_slice_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("slice/matrix");
    for &(rows, cols) in &[
        (128usize, 128usize),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ] {
        let a = matrix(rows, cols, 1.0);
        let r0: Range<usize> = rows / 4..(3 * rows) / 4;
        let c0: Range<usize> = cols / 4..(3 * cols) / 4;
        let elems = ((3 * rows / 4 - rows / 4) * (3 * cols / 4 - cols / 4)) as u64;
        group.throughput(Throughput::Elements(elems));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("st", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.slice(r0.clone(), c0.clone()).expect("slice must succeed");
                std::hint::black_box(drain_mat_f64(&r));
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Matrix::slice_mut
// ---------------------------------------------------------------------------

fn bench_slice_mut_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("slice_mut/matrix");
    for &(rows, cols) in &[
        (128usize, 128usize),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ] {
        let elems = ((3 * rows / 4 - rows / 4) * (3 * cols / 4 - cols / 4)) as u64;
        group.throughput(Throughput::Elements(elems));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("st", &label), &label, |bench, _| {
            bench.iter(|| {
                let mut a = matrix(rows, cols, 1.0);
                let view = a
                    .slice_mut(rows / 4..(3 * rows) / 4, cols / 4..(3 * cols) / 4)
                    .expect("slice_mut must succeed");
                let mut acc: f64 = 0.0;
                for i in 0..view.rows() {
                    for j in 0..view.cols() {
                        if let Some(v) = view.get((i, j)) {
                            acc += *v;
                        }
                    }
                }
                std::hint::black_box(acc);
            });
        });
    }
    group.finish();
}

criterion_group!(
    name = slice_benches;
    config = Criterion::default().sample_size(10);
    targets =
        bench_slice_tensor,
        bench_slice_mut_tensor,
        bench_slice_matrix,
        bench_slice_mut_matrix,
);
// `criterion_group!` above declares `pub fn slice_benches()`.
