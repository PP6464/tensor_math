//! Benchmarks for the single-threaded vs. multi-threaded `concat` implementations
//! of [`Tensor`] and [`Matrix`].
//!
//! Each pair (`concat` vs `concat_mt`) is exercised across a range of sizes so
//! that the crossover point where rayon actually starts to pay off is visible
//! in the output. To prevent the optimiser from eliding the work, every
//! benchmark reduces the resulting tensor to a single `f64` (sum of elements).

use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};

use tensor_math::definitions::matrix::Matrix;
use tensor_math::definitions::shape::Shape;
use tensor_math::definitions::tensor::Tensor;

/// Helper: drain a tensor into a single `f64` so the result is observably used.
#[inline]
fn drain_f64(t: &Tensor<f64>) -> f64 {
    t.elements().iter().copied().sum()
}

/// Helper: touch every element of a matrix and return a checksum, so the
/// optimiser cannot elide the work. The result is fed to `black_box` by the
/// caller.
#[inline]
fn drain_mat_f64(m: &Matrix<f64>) -> f64 {
    let mut acc = 0.0f64;
    for i in 0..m.rows() {
        for j in 0..m.cols() {
            if let Some(v) = m.get((i, j)) {
                acc += *v;
            }
        }
    }
    acc
}

/// Build a 1-D tensor of `n` zeros with the given capacity already filled
/// with `value`.
fn tensor_1d(n: usize, value: f64) -> Tensor<f64> {
    let shape = Shape::new(vec![n]);
    let elements = vec![value; n];
    Tensor::new(&shape, elements).expect("1-D tensor construction cannot fail")
}

/// Build an `n`-D tensor of zeros with the given shape and fill it with `value`.
fn tensor_from_shape(shape_dims: &[usize], value: f64) -> Tensor<f64> {
    let total: usize = shape_dims.iter().product();
    let shape = Shape::new(shape_dims.to_vec());
    let elements = vec![value; total];
    Tensor::new(&shape, elements).expect("tensor construction cannot fail")
}

/// Build an `rows x cols` matrix filled with `value`.
fn matrix(rows: usize, cols: usize, value: f64) -> Matrix<f64> {
    let elements = vec![value; rows * cols];
    Matrix::new(rows, cols, elements).expect("matrix construction cannot fail")
}

// ---------------------------------------------------------------------------
// 1-D tensor concatenation
// ---------------------------------------------------------------------------
//
// `Tensor::concat` along axis 0 of a 1-D tensor is the trivial `extend` path,
// so the two implementations are essentially the same work. We still keep the
// pair to make the comparison explicit in the output.

fn bench_concat_1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("concat/tensor_1d");
    for &n in &[1_000usize, 100_000, 1_000_000, 10_000_000] {
        let a = tensor_1d(n, 1.0);
        let b = tensor_1d(n, 2.0);
        group.throughput(Throughput::Elements(2 * n as u64));

        group.bench_with_input(BenchmarkId::new("st", n), &n, |bench, _| {
            bench.iter(|| {
                let r = a.concat(&b, 0).expect("concat must succeed");
                std::hint::black_box(drain_f64(&r));
            });
        });
        group.bench_with_input(BenchmarkId::new("mt", n), &n, |bench, _| {
            bench.iter(|| {
                let r = a.concat_mt(&b, 0).expect("concat must succeed");
                std::hint::black_box(drain_f64(&r));
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// 2-D tensor concatenation
// ---------------------------------------------------------------------------
//
// Here axis 1 (column-wise) exercises the chunked-copy path in both
// implementations. Sizes span small/medium/large to expose the rayon
// crossover.

fn bench_concat_2d_axis1(c: &mut Criterion) {
    let mut group = c.benchmark_group("concat/tensor_2d_axis1");
    for &(rows, cols) in &[
        (32usize, 32usize),
        (128, 128),
        (512, 512),
        (1024, 1024),
        (2048, 4096),
    ] {
        let a = tensor_from_shape(&[rows, cols], 1.0);
        let b = tensor_from_shape(&[rows, cols], 2.0);
        let elems = (2 * rows * cols) as u64;
        group.throughput(Throughput::Elements(elems));

        group.bench_with_input(
            BenchmarkId::new("st", format!("{rows}x{cols}")),
            &(rows, cols),
            |bench, _| {
                bench.iter(|| {
                    let r = a.concat(&b, 1).expect("concat must succeed");
                    std::hint::black_box(drain_f64(&r));
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("mt", format!("{rows}x{cols}")),
            &(rows, cols),
            |bench, _| {
                bench.iter(|| {
                    let r = a.concat_mt(&b, 1).expect("concat must succeed");
                    std::hint::black_box(drain_f64(&r));
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// 3-D tensor concatenation
// ---------------------------------------------------------------------------
//
// For a 3-D tensor the `concat` path operates on chunks of size
// `strides[axis-1]`, so even a moderate shape produces enough work to keep
// rayon busy.

fn bench_concat_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("concat/tensor_3d");
    for &(d0, d1, d2) in &[
        (8usize, 64usize, 64usize),
        (16, 128, 128),
        (32, 256, 256),
        (64, 256, 256),
    ] {
        let a = tensor_from_shape(&[d0, d1, d2], 1.0);
        let b = tensor_from_shape(&[d0, d1, d2], 2.0);
        let elems = (2 * d0 * d1 * d2) as u64;
        group.throughput(Throughput::Elements(elems));

        let label = format!("{d0}x{d1}x{d2}_axis2");

        group.bench_with_input(BenchmarkId::new("st", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.concat(&b, 2).expect("concat must succeed");
                std::hint::black_box(drain_f64(&r));
            });
        });
        group.bench_with_input(BenchmarkId::new("mt", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.concat_mt(&b, 2).expect("concat must succeed");
                std::hint::black_box(drain_f64(&r));
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Matrix: concat_cols (axis 1) and concat_rows (axis 0)
// ---------------------------------------------------------------------------

fn bench_concat_cols(c: &mut Criterion) {
    let mut group = c.benchmark_group("concat/matrix_cols");
    for &(rows, cols) in &[
        (32usize, 32usize),
        (128, 128),
        (512, 512),
        (1024, 1024),
        (2048, 4096),
    ] {
        let a = matrix(rows, cols, 1.0);
        let b = matrix(rows, cols, 2.0);
        group.throughput(Throughput::Elements((2 * rows * cols) as u64));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("st", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.concat_cols(&b).expect("concat must succeed");
                std::hint::black_box(drain_mat_f64(&r));
            });
        });
        group.bench_with_input(BenchmarkId::new("mt", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.concat_cols_mt(&b).expect("concat must succeed");
                std::hint::black_box(drain_mat_f64(&r));
            });
        });
    }
    group.finish();
}

fn bench_concat_rows(c: &mut Criterion) {
    let mut group = c.benchmark_group("concat/matrix_rows");
    for &(rows, cols) in &[
        (32usize, 32usize),
        (128, 128),
        (512, 512),
        (1024, 1024),
        (2048, 4096),
    ] {
        let a = matrix(rows, cols, 1.0);
        let b = matrix(rows, cols, 2.0);
        group.throughput(Throughput::Elements((2 * rows * cols) as u64));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("st", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.concat_rows(&b).expect("concat must succeed");
                std::hint::black_box(drain_mat_f64(&r));
            });
        });
        group.bench_with_input(BenchmarkId::new("mt", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.concat_rows_mt(&b).expect("concat must succeed");
                std::hint::black_box(drain_mat_f64(&r));
            });
        });
    }
    group.finish();
}

criterion_group!(
    name = concat_benches;
    config = Criterion::default().sample_size(10);
    targets =
        bench_concat_1d,
        bench_concat_2d_axis1,
        bench_concat_3d,
        bench_concat_cols,
        bench_concat_rows,
);
// `criterion_group!` above declares `pub fn concat_benches()`. The crate
// root imports it from `utilities::concat::concat_benches` to wire up
// `criterion_main!`.
