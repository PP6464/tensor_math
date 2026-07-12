//! Benchmarks for `Matrix<Complex64>::eigendecompose`.

use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tensor_math::definitions::matrix::Matrix;

fn bench_eigendecompose(c: &mut Criterion) {
    let mut group = c.benchmark_group("eigen");
    for ord in [25, 50, 100] {
        let mat = Matrix::rand(ord, ord).into_complex();
        group.throughput(Throughput::Elements((ord * ord) as u64));
        let label = format!("{ord}x{ord}");
        group.bench_with_input(BenchmarkId::new("decompose", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = mat.eigendecompose().expect("eigen must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_eigenvalues(c: &mut Criterion) {
    let mut group = c.benchmark_group("eigen");
    for ord in [25, 50, 100] {
        let mat = Matrix::rand(ord, ord).into_complex();
        group.throughput(Throughput::Elements((ord * ord) as u64));
        let label = format!("{ord}x{ord}");
        group.bench_with_input(BenchmarkId::new("values", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = mat.eigenvalues().expect("eigen must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

criterion_group!(
    name = eigen_benches;
    config = Criterion::default().sample_size(10);
    targets = 
        bench_eigendecompose,
        bench_eigenvalues,
);

criterion_main!(eigen_benches);
