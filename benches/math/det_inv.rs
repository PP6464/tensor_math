//! Benchmarks for `Matrix::<f64/Complex64>::det()` and `Matrix::<f64/Complex64>::inv()`

use std::hint::black_box;
use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};
use num::complex::Complex64;
use tensor_math::definitions::matrix::Matrix;

fn bench_f64_det(c: &mut Criterion) {
    let mut group = c.benchmark_group("det");
    for ord in [32, 64, 128, 256, 512, 1024] {
        let a = Matrix::<f64>::rand(ord, ord);
        let label = format!("{ord}x{ord}");
        group.throughput(Throughput::Elements((ord * ord) as u64));
        group.bench_with_input(BenchmarkId::new("f64", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.det().expect("det must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}
fn bench_c64_det(c: &mut Criterion) {
    let mut group = c.benchmark_group("det");
    for ord in [32, 64, 128, 256, 512, 1024] {
        let a = Matrix::<f64>::rand(ord, ord).into_complex();
        let b = Matrix::<f64>::rand(ord, ord).into_complex();
        let c = a + b * Complex64::I;
        let label = format!("{ord}x{ord}");
        group.throughput(Throughput::Elements((ord * ord) as u64));
        group.bench_with_input(BenchmarkId::new("c64", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = c.det().expect("det must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}
fn bench_f64_inv(c: &mut Criterion) {
    let mut group = c.benchmark_group("inv");
    for ord in [32, 64, 128, 256, 512, 1024] {
        let a = Matrix::<f64>::rand(ord, ord);
        let label = format!("{ord}x{ord}");
        group.throughput(Throughput::Elements((ord * ord) as u64));
        group.bench_with_input(BenchmarkId::new("f64", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.inv().expect("inv must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}
fn bench_c64_inv(c: &mut Criterion) {
    let mut group = c.benchmark_group("inv");
    for ord in [32, 64, 128, 256, 512, 1024] {
        let a = Matrix::<f64>::rand(ord, ord).into_complex();
        let b = Matrix::<f64>::rand(ord, ord).into_complex();
        let c = a + b * Complex64::I;
        let label = format!("{ord}x{ord}");
        group.throughput(Throughput::Elements((ord * ord) as u64));
        group.bench_with_input(BenchmarkId::new("c64", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = c.inv().expect("inv must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

criterion_group!(
    name = det_inv_benches;
    config = Criterion::default().sample_size(10);
    targets =
        bench_f64_det,
        bench_c64_det,
        bench_f64_inv,
        bench_c64_inv,
);