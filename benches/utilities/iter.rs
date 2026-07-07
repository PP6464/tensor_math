//! Benchmarks for the `enumerated_iter` family on [`Tensor`] and [`Matrix`].
//!
//! Each variant walks the full element list and is used pervasively by other
//! `utilities` operations. The interesting comparison is between the serial
//! iterators (`enumerated_iter`, `enumerated_iter_mut`) and their rayon
//! equivalents (`enumerated_par_iter`, `enumerated_par_iter_mut`); the
//! `mut` variants write to the result so we get to measure the parallel
//! write path as well.
//!
//! To prevent the optimiser from eliding the work, every iteration drains
//! the visited elements into a `u64` checksum fed to `black_box`.

use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};

use super::bench_utils::{matrix, tensor_from_shape, drain_iter, drain_iter_mut};

// ---------------------------------------------------------------------------
// Tensor::enumerated_iter / iter_mut / par_iter / par_iter_mut
// ---------------------------------------------------------------------------

fn bench_iter_tensor(c: &mut Criterion) {
    use rayon::iter::ParallelIterator;

    let mut group = c.benchmark_group("iter/tensor_2d");
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

        group.bench_with_input(BenchmarkId::new("st/iter", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = drain_iter(a.enumerated_iter());
                std::hint::black_box(r);
            });
        });
        group.bench_with_input(
            BenchmarkId::new("st/iter_mut", &label),
            &label,
            |bench, _| {
                let mut a = a.clone();
                bench.iter(|| {
                    let r = drain_iter_mut(a.enumerated_iter_mut());
                    std::hint::black_box(r);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("mt/par_iter", &label),
            &label,
            |bench, _| {
                bench.iter(|| {
                    // `enumerated_par_iter` yields `(Vec<usize>, T)`, not
                    // `&T`, so we collect into a checksum via `map` +
                    // `reduce` instead of borrowing.
                    let r = a
                        .enumerated_par_iter()
                        .map(|(_, v)| v.to_bits() as u64)
                        .reduce(|| 0u64, |a, b| a.wrapping_add(b));
                    std::hint::black_box(r);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("mt/par_iter_mut", &label),
            &label,
            |bench, _| {
                let mut a = a.clone();
                bench.iter(|| {
                    let r = a
                        .enumerated_par_iter_mut()
                        .map(|(_, v)| (*v).to_bits() as u64)
                        .reduce(|| 0u64, |a, b| a.wrapping_add(b));
                    std::hint::black_box(r);
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Matrix::enumerated_iter / iter_mut / par_iter / par_iter_mut
// ---------------------------------------------------------------------------

fn bench_iter_matrix(c: &mut Criterion) {
    use rayon::iter::ParallelIterator;

    let mut group = c.benchmark_group("iter/matrix");
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

        group.bench_with_input(BenchmarkId::new("st/iter", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = drain_iter(a.enumerated_iter().map(|((r, c), v)| (vec![r, c], v)));
                std::hint::black_box(r);
            });
        });
        group.bench_with_input(
            BenchmarkId::new("st/iter_mut", &label),
            &label,
            |bench, _| {
                let mut a = a.clone();
                bench.iter(|| {
                    let r = drain_iter_mut(a.enumerated_iter_mut().map(|((r, c), v)| (vec![r, c], v)));
                    std::hint::black_box(r);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("mt/par_iter", &label),
            &label,
            |bench, _| {
                bench.iter(|| {
                    let r = a
                        .enumerated_par_iter()
                        .map(|(_, v)| v.to_bits() as u64)
                        .reduce(|| 0u64, |a, b| a.wrapping_add(b));
                    std::hint::black_box(r);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("mt/par_iter_mut", &label),
            &label,
            |bench, _| {
                let mut a = a.clone();
                bench.iter(|| {
                    let r = a
                        .enumerated_par_iter_mut()
                        .map(|(_, v)| (*v).to_bits() as u64)
                        .reduce(|| 0u64, |a, b| a.wrapping_add(b));
                    std::hint::black_box(r);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    name = iter_benches;
    config = Criterion::default().sample_size(10);
    targets = bench_iter_tensor, bench_iter_matrix,
);
// `criterion_group!` above declares `pub fn iter_benches()`.
