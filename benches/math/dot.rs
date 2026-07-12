//! Benchmarks for the `dot` single-threaded and multi-threaded implementations

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use tensor_math::definitions::matrix::Matrix;
use tensor_math::definitions::shape::Shape;
use tensor_math::definitions::tensor::Tensor;
use tensor_math::shape;

fn bench_tensor_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot");

    for shape in &[
        shape![50, 100, 20],
        shape![500, 1000, 200],
        shape![200, 100],
        shape![2000, 1000],
        shape![1000],
        shape![1000000],
    ] {
        let mut label: String = "".to_string();
        for i in 0..shape.rank() {
            label.push_str(&format!("{}x", shape[i]));
        }
        // Remove last x
        label.remove(label.len() - 1);
        let a = Tensor::<f64>::rand(shape);
        group.throughput(Throughput::Elements(2 * shape.element_count() as u64));

        group.bench_with_input(BenchmarkId::new("tensor", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.dot(&a).expect("dot must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_tensor_dot_mt(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_mt");

    for shape in &[
        shape![50, 100, 20],
        shape![500, 1000, 200],
        shape![200, 100],
        shape![2000, 1000],
        shape![1000],
        shape![1000000],
    ] {
        let mut label: String = "".to_string();
        for i in 0..shape.rank() {
            label.push_str(&format!("{}x", shape[i]));
        }
        // Remove last x
        label.remove(label.len() - 1);
        let a = Tensor::<f64>::rand(shape);
        group.throughput(Throughput::Elements(2 * shape.element_count() as u64));

        group.bench_with_input(BenchmarkId::new("tensor", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.dot_mt(&a).expect("dot must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_matrix_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot");

    for (row, col) in &[
        (200, 100),
        (100, 200),
        (2000, 1000),
        (1000, 2000),
        (10000, 20000),
        (20000, 10000),
    ] {
        let label = format!("{row}x{col}");
        let a = Matrix::<f64>::rand(*row, *col);
        group.throughput(Throughput::Elements((2 * row * col) as u64));

        group.bench_with_input(BenchmarkId::new("matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.dot(&a).expect("dot must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

fn bench_matrix_dot_mt(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_mt");

    for (row, col) in &[
        (200, 100),
        (100, 200),
        (2000, 1000),
        (1000, 2000),
        (10000, 20000),
        (20000, 10000),
    ] {
        let label = format!("{row}x{col}");
        let a = Matrix::<f64>::rand(*row, *col);
        group.throughput(Throughput::Elements((2 * row * col) as u64));

        group.bench_with_input(BenchmarkId::new("matrix", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.dot_mt(&a).expect("dot must succeed");
                black_box(r);
            });
        });
    }

    group.finish();
}

criterion_group!(
    name = dot_benches;
    config = Criterion::default().sample_size(10);
    targets =
        bench_tensor_dot,
        bench_matrix_dot,
        bench_tensor_dot_mt,
        bench_matrix_dot_mt,
);

criterion_main!(dot_benches);
