//! Benchmarks for `Tensor::clip` / `par_clip` and `Matrix::clip` / `par_clip`.
//!
//! `clip` is a pure element-wise transform, so the cost is dominated by
//! per-element branching and (for `par_*`) rayon overhead. Sizes span
//! small/medium/large so the rayon crossover is visible in the output.

use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};

use super::bench_utils::{drain_f64, drain_mat_f64, matrix, tensor_from_shape};

// ---------------------------------------------------------------------------
// Tensor::clip / par_clip
// ---------------------------------------------------------------------------

fn bench_clip_tensor(c: &mut Criterion) {
    let mut group = c.benchmark_group("clip/tensor_2d");
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
                let r = a.clip(-0.5, 0.5);
                std::hint::black_box(drain_f64(&r));
            });
        });
        group.bench_with_input(BenchmarkId::new("mt", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.par_clip(-0.5, 0.5);
                std::hint::black_box(drain_f64(&r));
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Matrix::clip / par_clip
// ---------------------------------------------------------------------------

fn bench_clip_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("clip/matrix");
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
                let r = a.clip(-0.5, 0.5);
                std::hint::black_box(drain_mat_f64(&r));
            });
        });
        group.bench_with_input(BenchmarkId::new("mt", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.par_clip(-0.5, 0.5);
                std::hint::black_box(drain_mat_f64(&r));
            });
        });
    }
    group.finish();
}

criterion_group!(
    name = clip_benches;
    config = Criterion::default().sample_size(10);
    targets = bench_clip_tensor, bench_clip_matrix,
);
// `criterion_group!` above declares `pub fn clip_benches()`.
