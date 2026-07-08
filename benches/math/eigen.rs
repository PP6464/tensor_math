//! Benchmarks for `Matrix<Complex64>::eigendecompose`.

use std::hint::black_box;
use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};
use tensor_math::definitions::matrix::Matrix;

fn bench_eigen(c: &mut Criterion) {
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

criterion_group!(
    name = eigen_benches;
    config = Criterion::default().sample_size(10);
    targets = bench_eigen,
);