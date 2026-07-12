//! Benchmarks for the `map` family on [`Tensor`] and [`Matrix`].

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tensor_math::definitions::matrix::Matrix;
use tensor_math::definitions::tensor::Tensor;
use tensor_math::definitions::shape::Shape;
use tensor_math::shape;

fn bench_map_tensor(c: &mut Criterion) {
    let mut group = c.benchmark_group("map/tensor_2d");
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

        // `map` consumes the tensor, so we clone up front to keep the loop
        // re-runnable by Criterion.
        group.bench_with_input(BenchmarkId::new("st/map", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.clone().map(|x| x * 2.0 + 1.0);
                black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("st/map_refs", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.map_refs(|x| *x * 2.0 + 1.0);
                black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("mt/par_map", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.clone().par_map(|x| x * 2.0 + 1.0);
                black_box(r);
            });
        });
        group.bench_with_input(
            BenchmarkId::new("mt/par_map_refs", &label),
            &label,
            |bench, _| {
                bench.iter(|| {
                    let r = a.par_map_refs(|x| *x * 2.0 + 1.0);
                    black_box(r);
                });
            },
        );
    }
    group.finish();
}

fn bench_map_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("map/matrix");
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

        group.bench_with_input(BenchmarkId::new("st/map", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.clone().map(|x| x * 2.0 + 1.0);
                black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("st/map_refs", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.map_refs(|x| *x * 2.0 + 1.0);
                black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("mt/par_map", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.clone().par_map(|x| x * 2.0 + 1.0);
                black_box(r);
            });
        });
        group.bench_with_input(
            BenchmarkId::new("mt/par_map_refs", &label),
            &label,
            |bench, _| {
                bench.iter(|| {
                    let r = a.par_map_refs(|x| *x * 2.0 + 1.0);
                    black_box(r);
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

criterion_main!(map_benches);
