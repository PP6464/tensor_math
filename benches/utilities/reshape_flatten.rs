//! Benchmarks for the shape-only operations [`Tensor::reshape`],
//! [`Tensor::flatten`], and [`Matrix::reshape`].
//!
//! These functions are *not* compute-bound — they just construct a new
//! `Tensor` / `Matrix` over the existing element buffer. The benchmark is
//! useful as a baseline (every larger benchmark that takes a tensor or
//! matrix pays these costs once) and to catch accidental copies if a future
//! change breaks the "no data movement" guarantee.

use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};

use tensor_math::definitions::shape::Shape;

use super::bench_utils::{drain_f64, drain_mat_f64, matrix, tensor_1d, tensor_from_shape};

// ---------------------------------------------------------------------------
// Tensor::reshape (n×n → 1-D, n×n×4 → 2×(n×n×2))
// ---------------------------------------------------------------------------
//
// We benchmark two flavours of `reshape`:
// * 1-D → n-D (re-shape into a square or block layout).
// * n-D → 1-D (collapse).

fn bench_reshape_tensor(c: &mut Criterion) {
    let mut group = c.benchmark_group("reshape/tensor");
    for &n in &[1_000usize, 100_000, 1_000_000] {
        // 1-D → 2-D square
        let a = tensor_1d(n, 1.0);
        let side = (n as f64).sqrt() as usize;
        // Use sizes that exactly multiply to n.
        let new_shape = Shape::new(vec![side, n / side]);
        let label = format!("1d_{n}->2d");

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("1d_to_2d", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.clone().reshape(&new_shape).expect("reshape must succeed");
                std::hint::black_box(drain_f64(&r));
            });
        });

        // 2-D → 1-D collapse
        let a2 = tensor_from_shape(&[side, n / side], 1.0);
        let flat = Shape::new(vec![n]);
        let label2 = format!("2d_{n}->1d");

        group.bench_with_input(BenchmarkId::new("2d_to_1d", &label2), &label2, |bench, _| {
            bench.iter(|| {
                let r = a2.clone().reshape(&flat).expect("reshape must succeed");
                std::hint::black_box(drain_f64(&r));
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Tensor::flatten (requires the chosen axis to be size 1)
// ---------------------------------------------------------------------------
//
// Flatten a `[1, n]` tensor on axis 0, producing a 1-D tensor of length `n`.
// Sizes mirror `reshape/tensor` for direct comparison.

fn bench_flatten_tensor(c: &mut Criterion) {
    let mut group = c.benchmark_group("flatten/tensor");
    for &n in &[1_000usize, 100_000, 1_000_000] {
        let a = tensor_from_shape(&[1, n], 1.0);
        group.throughput(Throughput::Elements(n as u64));

        let label = format!("1x{n}");

        group.bench_with_input(BenchmarkId::new("axis0", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.clone().flatten(0).expect("flatten must succeed");
                std::hint::black_box(drain_f64(&r));
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Matrix::reshape (rows, cols) → (cols, rows) and back
// ---------------------------------------------------------------------------

fn bench_reshape_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("reshape/matrix");
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

        group.bench_with_input(BenchmarkId::new("swap", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.clone().reshape(cols, rows).expect("reshape must succeed");
                std::hint::black_box(drain_mat_f64(&r));
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
// `criterion_group!` above declares `pub fn reshape_flatten_benches()`.
