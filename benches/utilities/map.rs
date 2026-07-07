//! Benchmarks for the `map` family on [`Tensor`] and [`Matrix`].
//!
//! There are four variants on each type:
//! * `map`         — owns the input, closure takes `T` by value (st).
//! * `map_refs`    — borrows the input, closure takes `&T` (st).
//! * `par_map`     — like `map` but parallel.
//! * `par_map_refs`— like `map_refs` but parallel.
//!
//! The work is identical in all four cases (one `f64`-returning closure call
//! per element), so the benchmarks make the rayon overhead visible across
//! sizes.

use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};

use super::bench_utils::{drain_f64, drain_mat_f64, matrix, tensor_from_shape};

// ---------------------------------------------------------------------------
// Tensor::map / map_refs / par_map / par_map_refs
// ---------------------------------------------------------------------------

fn bench_map_tensor(c: &mut Criterion) {
    let mut group = c.benchmark_group("map/tensor_2d");
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

        // `map` consumes the tensor, so we clone up front to keep the loop
        // re-runnable by Criterion.
        group.bench_with_input(BenchmarkId::new("st/map", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.clone().map(|x| x * 2.0 + 1.0);
                std::hint::black_box(drain_f64(&r));
            });
        });
        group.bench_with_input(BenchmarkId::new("st/map_refs", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.map_refs(|x| *x * 2.0 + 1.0);
                std::hint::black_box(drain_f64(&r));
            });
        });
        group.bench_with_input(BenchmarkId::new("mt/par_map", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.clone().par_map(|x| x * 2.0 + 1.0);
                std::hint::black_box(drain_f64(&r));
            });
        });
        group.bench_with_input(
            BenchmarkId::new("mt/par_map_refs", &label),
            &label,
            |bench, _| {
                bench.iter(|| {
                    let r = a.par_map_refs(|x| *x * 2.0 + 1.0);
                    std::hint::black_box(drain_f64(&r));
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Matrix::map / map_refs / par_map / par_map_refs
// ---------------------------------------------------------------------------

fn bench_map_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("map/matrix");
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

        group.bench_with_input(BenchmarkId::new("st/map", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.clone().map(|x| x * 2.0 + 1.0);
                std::hint::black_box(drain_mat_f64(&r));
            });
        });
        group.bench_with_input(BenchmarkId::new("st/map_refs", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.map_refs(|x| *x * 2.0 + 1.0);
                std::hint::black_box(drain_mat_f64(&r));
            });
        });
        group.bench_with_input(BenchmarkId::new("mt/par_map", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.clone().par_map(|x| x * 2.0 + 1.0);
                std::hint::black_box(drain_mat_f64(&r));
            });
        });
        group.bench_with_input(
            BenchmarkId::new("mt/par_map_refs", &label),
            &label,
            |bench, _| {
                bench.iter(|| {
                    let r = a.par_map_refs(|x| *x * 2.0 + 1.0);
                    std::hint::black_box(drain_mat_f64(&r));
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    name = map_benches;
    config = Criterion::default().sample_size(10);
    targets = bench_map_tensor, bench_map_matrix,
);
// `criterion_group!` above declares `pub fn map_benches()`.
