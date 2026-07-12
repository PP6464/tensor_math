//! Benchmarks for the internal FFT functions.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use num::complex::Complex64;
use rand::RngExt;

use tensor_math::utilities::internal_functions::{
    bluestein_fft_vec, fft_vec, ifft_vec, radix_2_fft_vec,
};

/// Build a length-`n` vector of random `Complex64` values.
fn rand_vec(n: usize) -> Vec<Complex64> {
    let mut rng = rand::rng();
    (0..n)
        .map(|_| Complex64 {
            re: rng.random::<f64>(),
            im: rng.random::<f64>(),
        })
        .collect()
}

/// Powers of two, used to exercise `radix_2_fft_vec` and the radix-2 path of
/// `fft_vec`/`ifft_vec`.
const POWERS_OF_TWO: &[usize] = &[
    16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
];

/// Non-powers-of-two sizes, used to exercise `bluestein_fft_vec` and the
/// Bluestein dispatch path of `fft_vec`/`ifft_vec`.
const NON_POWERS_OF_TWO: &[usize] = &[15, 17, 31, 33, 100, 255, 1000, 4095, 4097, 12_000];

fn bench_radix_2_fft_vec(c: &mut Criterion) {
    let mut group = c.benchmark_group("radix_2_fft_vec");

    for &n in POWERS_OF_TWO {
        let x = rand_vec(n);
        group.throughput(Throughput::Elements(n as u64));

        let label = n.to_string();

        group.bench_with_input(BenchmarkId::new("c64", &label), &label, |b, _| {
            b.iter(|| {
                let r = radix_2_fft_vec(black_box(&x));
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_bluestein_fft_vec(c: &mut Criterion) {
    let mut group = c.benchmark_group("bluestein_fft_vec");

    for &n in NON_POWERS_OF_TWO {
        let x = rand_vec(n);
        group.throughput(Throughput::Elements(n as u64));

        let label = n.to_string();

        group.bench_with_input(BenchmarkId::new("c64", &label), &label, |b, _| {
            b.iter(|| {
                let r = bluestein_fft_vec(black_box(&x));
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_fft_vec_pow2(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_vec_pow2");

    for &n in POWERS_OF_TWO {
        let x = rand_vec(n);
        group.throughput(Throughput::Elements(n as u64));

        let label = n.to_string();

        group.bench_with_input(BenchmarkId::new("c64", &label), &label, |b, _| {
            b.iter(|| {
                let r = fft_vec(black_box(&x));
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_fft_vec_bluestein(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_vec_bluestein");

    for &n in NON_POWERS_OF_TWO {
        let x = rand_vec(n);
        group.throughput(Throughput::Elements(n as u64));

        let label = n.to_string();

        group.bench_with_input(BenchmarkId::new("c64", &label), &label, |b, _| {
            b.iter(|| {
                let r = fft_vec(black_box(&x));
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_ifft_vec_pow2(c: &mut Criterion) {
    let mut group = c.benchmark_group("ifft_vec_pow2");

    for &n in POWERS_OF_TWO {
        let x = rand_vec(n);
        group.throughput(Throughput::Elements(n as u64));

        let label = n.to_string();

        group.bench_with_input(BenchmarkId::new("c64", &label), &label, |b, _| {
            b.iter(|| {
                let r = ifft_vec(black_box(&x));
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_ifft_vec_bluestein(c: &mut Criterion) {
    let mut group = c.benchmark_group("ifft_vec_bluestein");

    for &n in NON_POWERS_OF_TWO {
        let x = rand_vec(n);
        group.throughput(Throughput::Elements(n as u64));

        let label = n.to_string();

        group.bench_with_input(BenchmarkId::new("c64", &label), &label, |b, _| {
            b.iter(|| {
                let r = ifft_vec(black_box(&x));
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_fft_round_trip_pow2(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_roundtrip_pow2");

    for &n in POWERS_OF_TWO {
        let x = rand_vec(n);
        group.throughput(Throughput::Elements((2 * n) as u64));

        let label = n.to_string();

        group.bench_with_input(BenchmarkId::new("c64", &label), &label, |b, _| {
            b.iter(|| {
                let fwd = fft_vec(black_box(&x));
                let inv = ifft_vec(black_box(&fwd));
                black_box(inv);
            });
        });
    }

    group.finish();
}

fn bench_fft_round_trip_bluestein(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_roundtrip_bluestein");

    for &n in NON_POWERS_OF_TWO {
        let x = rand_vec(n);
        group.throughput(Throughput::Elements((2 * n) as u64));

        let label = n.to_string();

        group.bench_with_input(BenchmarkId::new("c64", &label), &label, |b, _| {
            b.iter(|| {
                let fwd = fft_vec(black_box(&x));
                let inv = ifft_vec(black_box(&fwd));
                black_box(inv);
            });
        });
    }

    group.finish();
}

criterion_group!(
    name = fft_internal_benches;
    config = Criterion::default().sample_size(10);
    targets =
        bench_radix_2_fft_vec,
        bench_bluestein_fft_vec,
        bench_fft_vec_pow2,
        bench_fft_vec_bluestein,
        bench_ifft_vec_pow2,
        bench_ifft_vec_bluestein,
        bench_fft_round_trip_pow2,
        bench_fft_round_trip_bluestein,
);

criterion_main!(fft_internal_benches);
