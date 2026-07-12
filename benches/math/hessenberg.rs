//! Benchmarks for `Matrix::<f64/Compelx64>::upper_hessenberg` and `lower_hessenberg`.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use num::complex::Complex64;
use tensor_math::definitions::matrix::Matrix;

fn bench_f64_upper_hessenberg(c: &mut Criterion) {
    let mut group = c.benchmark_group("upper_hessenberg");
    for ord in [32, 64, 128, 256, 512, 1024] {
        let a = Matrix::<f64>::rand(ord, ord);
        group.throughput(Throughput::Elements((ord * ord) as u64));

        let label = format!("{ord}x{ord}");

        group.bench_with_input(BenchmarkId::new("c64", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.upper_hessenberg().expect("hessenberg must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_c64_upper_hessenberg(c: &mut Criterion) {
    let mut group = c.benchmark_group("upper_hessenberg");
    for ord in [32, 64, 128, 256, 512, 1024] {
        let a = Matrix::<f64>::rand(ord, ord).into_complex();
        let b = Matrix::<f64>::rand(ord, ord).into_complex();
        let c = a + b * Complex64::I;
        group.throughput(Throughput::Elements((ord * ord) as u64));

        let label = format!("{ord}x{ord}");

        group.bench_with_input(BenchmarkId::new("c64", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = c.upper_hessenberg().expect("hessenberg must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_f64_lower_hessenberg(c: &mut Criterion) {
    let mut group = c.benchmark_group("lower_hessenberg");
    for ord in [32, 64, 128, 256, 512, 1024] {
        let a = Matrix::<f64>::rand(ord, ord);
        group.throughput(Throughput::Elements((ord * ord) as u64));

        let label = format!("{ord}x{ord}");

        group.bench_with_input(BenchmarkId::new("f64", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.lower_hessenberg().expect("hessenberg must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_c64_lower_hessenberg(c: &mut Criterion) {
    let mut group = c.benchmark_group("lower_hessenberg");
    for ord in [32, 64, 128, 256, 512, 1024] {
        let a = Matrix::<f64>::rand(ord, ord).into_complex();
        let b = Matrix::<f64>::rand(ord, ord).into_complex();
        let c = a + b * Complex64::I;
        group.throughput(Throughput::Elements((ord * ord) as u64));

        let label = format!("{ord}x{ord}");

        group.bench_with_input(BenchmarkId::new("c64", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = c.lower_hessenberg().expect("hessenberg must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_f64_upper_hessenberg_h(c: &mut Criterion) {
    let mut group = c.benchmark_group("upper_hessenberg_h");
    for ord in [32, 64, 128, 256, 512, 1024] {
        let a = Matrix::<f64>::rand(ord, ord);
        group.throughput(Throughput::Elements((ord * ord) as u64));

        let label = format!("{ord}x{ord}");

        group.bench_with_input(BenchmarkId::new("c64", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.upper_hessenberg_h().expect("hessenberg must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_c64_upper_hessenberg_h(c: &mut Criterion) {
    let mut group = c.benchmark_group("upper_hessenberg_h");
    for ord in [32, 64, 128, 256, 512, 1024] {
        let a = Matrix::<f64>::rand(ord, ord).into_complex();
        let b = Matrix::<f64>::rand(ord, ord).into_complex();
        let c = a + b * Complex64::I;
        group.throughput(Throughput::Elements((ord * ord) as u64));

        let label = format!("{ord}x{ord}");

        group.bench_with_input(BenchmarkId::new("c64", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = c.upper_hessenberg_h().expect("hessenberg must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_f64_lower_hessenberg_h(c: &mut Criterion) {
    let mut group = c.benchmark_group("lower_hessenberg_h");
    for ord in [32, 64, 128, 256, 512, 1024] {
        let a = Matrix::<f64>::rand(ord, ord);
        group.throughput(Throughput::Elements((ord * ord) as u64));

        let label = format!("{ord}x{ord}");

        group.bench_with_input(BenchmarkId::new("f64", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.lower_hessenberg_h().expect("hessenberg must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_c64_lower_hessenberg_h(c: &mut Criterion) {
    let mut group = c.benchmark_group("lower_hessenberg_h");
    for ord in [32, 64, 128, 256, 512, 1024] {
        let a = Matrix::<f64>::rand(ord, ord).into_complex();
        let b = Matrix::<f64>::rand(ord, ord).into_complex();
        let c = a + b * Complex64::I;
        group.throughput(Throughput::Elements((ord * ord) as u64));

        let label = format!("{ord}x{ord}");

        group.bench_with_input(BenchmarkId::new("c64", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = c.lower_hessenberg_h().expect("hessenberg must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_f64_upper_hessenberg_accumulate_q(c: &mut Criterion) {
    let mut group = c.benchmark_group("upper_hessenberg_accumulate_q");
    for ord in [32, 64, 128, 256, 512, 1024] {
        let a = Matrix::<f64>::rand(ord, ord);
        let (_, reflectors) = a.upper_hessenberg_h().expect("hessenberg must succeed");
        group.throughput(Throughput::Elements((ord * ord) as u64));

        let label = format!("{ord}x{ord}");

        group.bench_with_input(BenchmarkId::new("c64", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = reflectors.accumulate_q();
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_c64_upper_hessenberg_accumulate_q(c: &mut Criterion) {
    let mut group = c.benchmark_group("upper_hessenberg_accumulate_q");
    for ord in [32, 64, 128, 256, 512, 1024] {
        let a = Matrix::<f64>::rand(ord, ord).into_complex();
        let b = Matrix::<f64>::rand(ord, ord).into_complex();
        let c = a + b * Complex64::I;
        let (_, reflectors) = c.upper_hessenberg_h().expect("hessenberg must succeed");
        group.throughput(Throughput::Elements((ord * ord) as u64));

        let label = format!("{ord}x{ord}");

        group.bench_with_input(BenchmarkId::new("c64", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = reflectors.accumulate_q();
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_f64_lower_hessenberg_accumulate_q(c: &mut Criterion) {
    let mut group = c.benchmark_group("lower_hessenberg_accumulate_q");
    for ord in [32, 64, 128, 256, 512, 1024] {
        let a = Matrix::<f64>::rand(ord, ord);
        let (_, reflectors) = a.lower_hessenberg_h().expect("hessenberg must succeed");
        group.throughput(Throughput::Elements((ord * ord) as u64));

        let label = format!("{ord}x{ord}");

        group.bench_with_input(BenchmarkId::new("f64", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = reflectors.accumulate_q();
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_c64_lower_hessenberg_accumulate_q(c: &mut Criterion) {
    let mut group = c.benchmark_group("lower_hessenberg_accumulate_q");
    for ord in [32, 64, 128, 256, 512, 1024] {
        let a = Matrix::<f64>::rand(ord, ord).into_complex();
        let b = Matrix::<f64>::rand(ord, ord).into_complex();
        let c = a + b * Complex64::I;
        let (_, reflectors) = c.lower_hessenberg_h().expect("hessenberg must succeed");
        group.throughput(Throughput::Elements((ord * ord) as u64));

        let label = format!("{ord}x{ord}");

        group.bench_with_input(BenchmarkId::new("c64", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = reflectors.accumulate_q();
                black_box(r);
            });
        });
    }
    group.finish();
}

criterion_group!(
    name = hessenberg_benches;
    config = Criterion::default().sample_size(10);
    targets =
        bench_f64_upper_hessenberg,
        bench_f64_upper_hessenberg_h,
        bench_f64_upper_hessenberg_accumulate_q,
        bench_c64_upper_hessenberg,
        bench_c64_upper_hessenberg_h,
        bench_c64_upper_hessenberg_accumulate_q,
        bench_f64_lower_hessenberg,
        bench_f64_lower_hessenberg_h,
        bench_f64_lower_hessenberg_accumulate_q,
        bench_c64_lower_hessenberg,
        bench_c64_lower_hessenberg_h,
        bench_c64_lower_hessenberg_accumulate_q,
);

criterion_main!(hessenberg_benches);
