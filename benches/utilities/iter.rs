//! Benchmarks for the `enumerated_iter` family on [`Tensor`] and [`Matrix`].

use std::hint::black_box;
use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};
use tensor_math::definitions::matrix::Matrix;
use tensor_math::definitions::tensor::Tensor;
use tensor_math::definitions::shape::Shape;
use tensor_math::shape;

/// Drain a sequential `enumerated_iter` into a checksum.
#[inline]
pub fn drain_iter<I>(it: I) -> u64
where
    I: Iterator<Item = (Vec<usize>, f64)>,
{
    let mut acc: u64 = 0;
    for (_, v) in it {
        acc = acc.wrapping_add(v.to_bits());
    }
    acc
}

/// Drain a sequential `enumerated_iter_mut` into a checksum.
#[inline]
pub fn drain_iter_mut<'a, I>(it: I) -> u64
where
    I: Iterator<Item = (Vec<usize>, &'a mut f64)>,
{
    let mut acc: u64 = 0;
    for (_, v) in it {
        acc = acc.wrapping_add((*v).to_bits());
    }
    acc
}

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
        let a = Tensor::from_value(&shape![rows, cols], 1.0);
        let elems = (rows * cols) as u64;
        group.throughput(Throughput::Elements(elems));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("st/iter", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = drain_iter(a.enumerated_iter());
                black_box(r);
            });
        });
        group.bench_with_input(
            BenchmarkId::new("st/iter_mut", &label),
            &label,
            |bench, _| {
                let mut a = a.clone();
                bench.iter(|| {
                    let r = drain_iter_mut(a.enumerated_iter_mut());
                    black_box(r);
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
                        .map(|(_, v)| v.to_bits())
                        .reduce(|| 0u64, |a, b| a.wrapping_add(b));
                    black_box(r);
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
                        .map(|(_, v)| (*v).to_bits())
                        .reduce(|| 0u64, |a, b| a.wrapping_add(b));
                    black_box(r);
                });
            },
        );
    }
    group.finish();
}

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
        let a = Matrix::from_value(rows, cols, 1.0);
        group.throughput(Throughput::Elements((rows * cols) as u64));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("st/iter", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = drain_iter(a.enumerated_iter().map(|((r, c), v)| (vec![r, c], v)));
                black_box(r);
            });
        });
        group.bench_with_input(
            BenchmarkId::new("st/iter_mut", &label),
            &label,
            |bench, _| {
                let mut a = a.clone();
                bench.iter(|| {
                    let r = drain_iter_mut(a.enumerated_iter_mut().map(|((r, c), v)| (vec![r, c], v)));
                    black_box(r);
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
                        .map(|(_, v)| v.to_bits())
                        .reduce(|| 0u64, |a, b| a.wrapping_add(b));
                    black_box(r);
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
                        .map(|(_, v)| (*v).to_bits())
                        .reduce(|| 0u64, |a, b| a.wrapping_add(b));
                    black_box(r);
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
