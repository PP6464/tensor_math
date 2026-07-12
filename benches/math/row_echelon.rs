//! Benchmarks for `Matrix<f64>::row_echelon` / `reduced_row_echelon`
//! and `Matrix<Complex64>::row_echelon` / `reduced_row_echelon`.

use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use num::complex::Complex64;
use tensor_math::definitions::matrix::Matrix;

fn bench_f64_row_echelon(c: &mut Criterion) {
    let mut group = c.benchmark_group("row_echelon");
    for &(rows, cols) in &[
        (32, 16),
        (32, 32),
        (32, 64),
        (128, 64),
        (128, 128),
        (128, 256),
        (512, 256),
        (512, 512),
        (512, 1024),
        (1024, 512),
        (1024, 1024),
        (1024, 2048),
        (2048, 1024),
        (2048, 2048),
        (2048, 4096),
    ] {
        let a = Matrix::<f64>::rand(rows, cols);
        group.throughput(Throughput::Elements((rows * cols) as u64));

        let label = format!("{rows}x{cols}");
        let name = if rows > cols {
            "tall"
        } else if rows == cols {
            "square"
        } else {
            "wide"
        };
        let name = format!("f64/{name}");

        group.bench_with_input(BenchmarkId::new(name, &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.clone().row_echelon();
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_f64_reduced_row_echelon(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduced_row_echelon");
    for &(rows, cols) in &[
        (32, 16),
        (32, 32),
        (32, 64),
        (128, 64),
        (128, 128),
        (128, 256),
        (512, 256),
        (512, 512),
        (512, 1024),
        (1024, 512),
        (1024, 1024),
        (1024, 2048),
        (2048, 1024),
        (2048, 2048),
        (2048, 4096),
    ] {
        let a = Matrix::<f64>::rand(rows, cols);
        group.throughput(Throughput::Elements((rows * cols) as u64));

        let label = format!("{rows}x{cols}");
        let name = if rows > cols {
            "tall"
        } else if rows == cols {
            "square"
        } else {
            "wide"
        };
        let name = format!("f64/{name}");

        group.bench_with_input(BenchmarkId::new(name, &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.clone().reduced_row_echelon();
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_c64_row_echelon(c: &mut Criterion) {
    let mut group = c.benchmark_group("row_echelon");
    for &(rows, cols) in &[
        (32, 16),
        (32, 32),
        (32, 64),
        (128, 64),
        (128, 128),
        (128, 256),
        (512, 256),
        (512, 512),
        (512, 1024),
        (1024, 512),
        (1024, 1024),
        (1024, 2048),
        (2048, 1024),
        (2048, 2048),
        (2048, 4096),
    ] {
        let a = Matrix::<f64>::rand(rows, cols);
        let b = Matrix::<f64>::rand(rows, cols);
        let c = a.into_complex() + b.into_complex().par_map(|v| v * Complex64::I);
        group.throughput(Throughput::Elements((rows * cols) as u64));

        let label = format!("{rows}x{cols}");
        let name = if rows > cols {
            "tall"
        } else if rows == cols {
            "square"
        } else {
            "wide"
        };
        let name = format!("c64/{name}");

        group.bench_with_input(BenchmarkId::new(name, &label), &label, |bench, _| {
            bench.iter(|| {
                let r = c.clone().row_echelon();
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_c64_reduced_row_echelon(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduced_row_echelon");
    for &(rows, cols) in &[
        (32, 16),
        (32, 32),
        (32, 64),
        (128, 64),
        (128, 128),
        (128, 256),
        (512, 256),
        (512, 512),
        (512, 1024),
        (1024, 512),
        (1024, 1024),
        (1024, 2048),
        (2048, 1024),
        (2048, 2048),
        (2048, 4096),
    ] {
        let a = Matrix::<f64>::rand(rows, cols);
        let b = Matrix::<f64>::rand(rows, cols);
        let c = a.into_complex() + b.into_complex().par_map(|v| v * Complex64::I);
        group.throughput(Throughput::Elements((rows * cols) as u64));

        let label = format!("{rows}x{cols}");
        let name = if rows > cols {
            "tall"
        } else if rows == cols {
            "square"
        } else {
            "wide"
        };
        let name = format!("c64/{name}");

        group.bench_with_input(BenchmarkId::new(name, &label), &label, |bench, _| {
            bench.iter(|| {
                let r = c.clone().reduced_row_echelon();
                black_box(r);
            });
        });
    }
    group.finish();
}

criterion_group!(
    name = row_echelon_benches;
    config = Criterion::default().sample_size(10);
    targets =
        bench_f64_row_echelon,
        bench_c64_row_echelon,
        bench_f64_reduced_row_echelon,
        bench_c64_reduced_row_echelon,
);

criterion_main!(row_echelon_benches);
