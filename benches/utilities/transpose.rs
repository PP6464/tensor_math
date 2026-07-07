//! Benchmarks for the single-threaded vs. multi-threaded `transpose`
//! implementations of [`Tensor`] and [`Matrix`].
//!
//! Each pair (`transpose` vs `transpose_mt`) is exercised across a range of
//! sizes so the crossover point where rayon actually starts to pay off is
//! visible in the output. To prevent the optimiser from eliding the work,
//! every benchmark reduces the resulting tensor/matrix to a single `f64`
//! (sum of elements).
//!
//! Notes on the permutations used:
//! * 1-D tensors: a 1-axis permutation is forced (`[0]`) to keep parity with
//!   the API; both implementations are essentially identity, but the pair is
//!   included for completeness.
//! * 2-D tensors: the swap `[1, 0]` exercises the non-trivial permutation
//!   path.
//! * 3-D tensors: `[2, 0, 1]` is a cyclic permutation that touches every
//!   stride in both implementations, mirroring the size of the work done in
//!   `concat/tensor_3d`.
//! * Matrices: `Matrix::transpose` is a fixed swap, so the pair measures the
//!   scalar vs. parallel path of the same operation.

use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};

use tensor_math::definitions::transpose::Transpose;
use tensor_math::transpose;

use super::bench_utils::{drain_f64, drain_mat_f64, matrix, tensor_1d, tensor_from_shape};

// ---------------------------------------------------------------------------
// 1-D tensor transpose
// ---------------------------------------------------------------------------
//
// A 1-D tensor has only one axis, so any valid permutation is `[0]` and the
// work is essentially zero. We still keep the pair so the comparison shows
// up in the output at small sizes.

fn bench_transpose_1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose/tensor_1d");
    let perm: Transpose = transpose![0];
    for &n in &[1_000usize, 100_000, 1_000_000, 10_000_000] {
        let a = tensor_1d(n, 1.0);
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("st", n), &n, |bench, _| {
            bench.iter(|| {
                let r = a.transpose(&perm).expect("transpose must succeed");
                std::hint::black_box(drain_f64(&r));
            });
        });
        group.bench_with_input(BenchmarkId::new("mt", n), &n, |bench, _| {
            bench.iter(|| {
                let r = a.transpose_mt(&perm).expect("transpose must succeed");
                std::hint::black_box(drain_f64(&r));
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// 2-D tensor transpose
// ---------------------------------------------------------------------------
//
// A 2-D `[rows, cols]` tensor transposed by `[1, 0]` is the matrix case in
// disguise. Sizes span small/medium/large to expose the rayon crossover.

fn bench_transpose_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose/tensor_2d");
    let perm: Transpose = transpose![1, 0];
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
                let r = a.transpose(&perm).expect("transpose must succeed");
                std::hint::black_box(drain_f64(&r));
            });
        });
        group.bench_with_input(BenchmarkId::new("mt", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.transpose_mt(&perm).expect("transpose must succeed");
                std::hint::black_box(drain_f64(&r));
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// 3-D tensor transpose
// ---------------------------------------------------------------------------
//
// A 3-D tensor transposed by `[2, 0, 1]` rotates the axes so that every
// stride is involved in the copy. Sizes mirror `concat/tensor_3d` for an
// apples-to-apples comparison of where rayon helps.

fn bench_transpose_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose/tensor_3d");
    let perm: Transpose = transpose![2, 0, 1];
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
                let r = a.transpose(&perm).expect("transpose must succeed");
                std::hint::black_box(drain_f64(&r));
            });
        });
        group.bench_with_input(BenchmarkId::new("mt", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.transpose_mt(&perm).expect("transpose must succeed");
                std::hint::black_box(drain_f64(&r));
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Matrix transpose
// ---------------------------------------------------------------------------
//
// `Matrix::transpose` has no permutation argument — it is always a swap
// of rows and columns — so the pair directly compares the scalar copy loop
// against the rayon parallel copy.

fn bench_transpose_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose/matrix");
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
                let r = a.transpose();
                std::hint::black_box(drain_mat_f64(&r));
            });
        });
        group.bench_with_input(BenchmarkId::new("mt", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.transpose_mt();
                std::hint::black_box(drain_mat_f64(&r));
            });
        });
    }
    group.finish();
}

criterion_group!(
    name = transpose_benches;
    config = Criterion::default().sample_size(10);
    targets =
        bench_transpose_1d,
        bench_transpose_2d,
        bench_transpose_3d,
        bench_transpose_matrix,
);
// `criterion_group!` above declares `pub fn transpose_benches()`. The crate
// root imports it from `utilities::transpose::transpose_benches` to wire up
// `criterion_main!`.
