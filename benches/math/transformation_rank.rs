//! Benchmarks for `Matrix::<f64/Complex64>::transformation_rank`.

use std::hint::black_box;
use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};
use num::complex::Complex64;
use tensor_math::definitions::matrix::Matrix;

fn bench_f64_transformation_rank(c: &mut Criterion) {
    let mut group = c.benchmark_group("transformation_rank");
    for (rows, cols) in [
        (32, 16),
        (32, 32),
        (32, 64),
        (64, 32),
        (64, 64),
        (64, 128),
        (128, 64),
        (128, 128),
        (128, 256),
    ] {
        let a = Matrix::<f64>::rand(rows, cols);
        group.throughput(Throughput::Elements((rows * cols) as u64));

        let label = format!(
            "{}/{rows}x{cols}",
            if rows > cols {
                "tall"
            } else if rows == cols {
                "square"
            } else {
                "wide"
            }
        );

        group.bench_with_input(BenchmarkId::new("c64", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.transformation_rank();
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_c64_transformation_rank(c: &mut Criterion) {
    let mut group = c.benchmark_group("transformation_rank");
    for (rows, cols) in [
        (32, 16),
        (32, 32),
        (32, 64),
        (64, 32),
        (64, 64),
        (64, 128),
        (128, 64),
        (128, 128),
        (128, 256),
    ] {
        let a = Matrix::<f64>::rand(rows, cols).into_complex();
        let b = Matrix::<f64>::rand(rows, cols).into_complex();
        let c = a + b * Complex64::I;
        group.throughput(Throughput::Elements((rows * cols) as u64));

        let label = format!(
            "{}/{rows}x{cols}",
            if rows > cols {
                "tall"
            } else if rows == cols {
                "square"
            } else {
                "wide"
            }
        );

        group.bench_with_input(BenchmarkId::new("c64", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = c.transformation_rank();
                black_box(r);
            });
        });
    }
    group.finish();
}

criterion_group!(
    name = transformation_rank_benches;
    config = Criterion::default().sample_size(10);
    targets =
        bench_f64_transformation_rank,
        bench_c64_transformation_rank,
);
