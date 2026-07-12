//! Benchmarks for the `slice` family on [`Tensor`] and [`Matrix`].

use std::hint::black_box;
use std::ops::Range;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tensor_math::definitions::matrix::Matrix;
use tensor_math::definitions::tensor::Tensor;
use tensor_math::definitions::shape::Shape;
use tensor_math::shape;

fn bench_slice_tensor(c: &mut Criterion) {
    let mut group = c.benchmark_group("slice/tensor_2d");
    for &(rows, cols) in &[
        (128usize, 128usize),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ] {
        let a = Tensor::from_value(&shape![rows, cols], 1.0);
        let r0 = rows / 4..(3 * rows) / 4;
        let c0 = cols / 4..(3 * cols) / 4;
        let elems = ((3 * rows / 4 - rows / 4) * (3 * cols / 4 - cols / 4)) as u64;
        group.throughput(Throughput::Elements(elems));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("st", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.slice(&[r0.clone(), c0.clone()]).expect("slice must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_slice_mut_tensor(c: &mut Criterion) {
    let mut group = c.benchmark_group("slice_mut/tensor_2d");
    for &(rows, cols) in &[
        (128usize, 128usize),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ] {
        let elems = ((3 * rows / 4 - rows / 4) * (3 * cols / 4 - cols / 4)) as u64;
        group.throughput(Throughput::Elements(elems));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("st", &label), &label, |bench, _| {
            bench.iter(|| {
                let mut a = Tensor::from_value(&shape![rows, cols], 1.0);
                let view = a
                    .slice_mut(&[
                        rows / 4..(3 * rows) / 4,
                        cols / 4..(3 * cols) / 4,
                    ])
                    .expect("slice_mut must succeed");
                // Read every element via Index to make sure the view is
                // observably used; the result is fed to `black_box`.
                let mut acc: f64 = 0.0;
                for i in 0..view.shape()[0] {
                    for j in 0..view.shape()[1] {
                        if let Some(v) = view.get(&[i, j]) {
                            acc += *v;
                        }
                    }
                }
                black_box(acc);
            });
        });
    }
    group.finish();
}

fn bench_slice_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("slice/matrix");
    for &(rows, cols) in &[
        (128usize, 128usize),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ] {
        let a = Matrix::from_value(rows, cols, 1.0);
        let r0: Range<usize> = rows / 4..(3 * rows) / 4;
        let c0: Range<usize> = cols / 4..(3 * cols) / 4;
        let elems = ((3 * rows / 4 - rows / 4) * (3 * cols / 4 - cols / 4)) as u64;
        group.throughput(Throughput::Elements(elems));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("st", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.slice(r0.clone(), c0.clone()).expect("slice must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_slice_mut_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("slice_mut/matrix");
    for &(rows, cols) in &[
        (128usize, 128usize),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ] {
        let elems = ((3 * rows / 4 - rows / 4) * (3 * cols / 4 - cols / 4)) as u64;
        group.throughput(Throughput::Elements(elems));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("st", &label), &label, |bench, _| {
            bench.iter(|| {
                let mut a = Matrix::from_value(rows, cols, 1.0);
                let view = a
                    .slice_mut(rows / 4..(3 * rows) / 4, cols / 4..(3 * cols) / 4)
                    .expect("slice_mut must succeed");
                let mut acc: f64 = 0.0;
                for i in 0..view.rows() {
                    for j in 0..view.cols() {
                        if let Some(v) = view.get((i, j)) {
                            acc += *v;
                        }
                    }
                }
                black_box(acc);
            });
        });
    }
    group.finish();
}

criterion_group!(
    name = slice_benches;
    config = Criterion::default().sample_size(10);
    targets =
        bench_slice_tensor,
        bench_slice_mut_tensor,
        bench_slice_matrix,
        bench_slice_mut_matrix,
);

criterion_main!(slice_benches);
