//! Benchmarks for every public FFT operation in `src/math/fft_ops.rs`.

use std::collections::HashSet;
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use num::complex::Complex64;
use tensor_math::definitions::matrix::Matrix;
use tensor_math::definitions::shape::Shape;
use tensor_math::definitions::tensor::Tensor;
use tensor_math::shape;

/// Build a `Matrix<Complex64>` with both real and imaginary parts populated.
fn rand_complex_matrix(rows: usize, cols: usize) -> Matrix<Complex64> {
    let a = Matrix::<f64>::rand(rows, cols);
    let b = Matrix::<f64>::rand(rows, cols);
    a.into_complex() + b.into_complex().par_map(|v| v * Complex64::I)
}

/// Build a `Tensor<Complex64>` with both real and imaginary parts populated.
fn rand_complex_tensor(s: &Shape) -> Tensor<Complex64> {
    let a = Tensor::<f64>::rand(s);
    let b = Tensor::<f64>::rand(s);
    a.into_complex() + b.into_complex().par_map(|v| v * Complex64::I)
}

fn shape_label(s: &Shape) -> String {
    let mut label = String::new();
    for i in 0..s.rank() {
        label.push_str(&format!("{}x", s[i]));
    }
    label.pop();
    label
}

fn bench_tensor_f64_fft_single_axis(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_single_axis");
    let shapes = [
        shape![1024],
        shape![4096],
        shape![16384],
        shape![64, 64],
        shape![256, 256],
        shape![64, 64, 64],
    ];

    for s in &shapes {
        let t = Tensor::<f64>::rand(s);
        group.throughput(Throughput::Elements(s.element_count() as u64));
        let label = shape_label(s);

        for axis in 0..s.rank() {
            let bench_label = format!("{label}/axis{axis}");
            group.bench_with_input(
                BenchmarkId::new("f64/tensor", &bench_label),
                &bench_label,
                |b, _| {
                    b.iter(|| {
                        let r = t.fft_single_axis(axis).expect("fft must succeed");
                        black_box(r);
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_tensor_f64_fft_axes(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_axes");
    // For each shape test every non-empty subset of axes.
    let shapes = [
        (shape![1024], HashSet::from([0usize])),
        (shape![64, 64], HashSet::from([0usize, 1])),
        (shape![64, 64, 64], HashSet::from([0usize, 2])),
        (shape![64, 64, 64], HashSet::from([0usize, 1, 2])),
    ];

    for (s, axes) in &shapes {
        let t = Tensor::<f64>::rand(s);
        group.throughput(Throughput::Elements(s.element_count() as u64));
        let mut label = shape_label(s);
        let mut axes_label = String::from("axes=");
        let mut sorted: Vec<_> = axes.iter().copied().collect();
        sorted.sort();
        for (i, a) in sorted.iter().enumerate() {
            if i > 0 {
                axes_label.push(',');
            }
            axes_label.push_str(&a.to_string());
        }
        label.push('/');
        label.push_str(&axes_label);

        group.bench_with_input(BenchmarkId::new("f64/tensor", &label), &label, |b, _| {
            b.iter(|| {
                let r = t.fft_axes(axes).expect("fft must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_tensor_f64_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft");
    let shapes = [
        shape![1024],
        shape![4096],
        shape![64, 64],
        shape![256, 256],
        shape![32, 32, 32],
    ];

    for s in &shapes {
        let t = Tensor::<f64>::rand(s);
        group.throughput(Throughput::Elements(s.element_count() as u64));
        let label = shape_label(s);

        group.bench_with_input(
            BenchmarkId::new("f64/tensor", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let r = t.fft().expect("fft must succeed");
                    black_box(r);
                });
            },
        );
    }
    group.finish();
}

fn bench_tensor_f64_ifft_single_axis(c: &mut Criterion) {
    let mut group = c.benchmark_group("ifft_single_axis");
    let shapes = [
        shape![1024],
        shape![4096],
        shape![16384],
        shape![64, 64],
        shape![256, 256],
        shape![64, 64, 64],
    ];

    for s in &shapes {
        let t = Tensor::<f64>::rand(s);
        group.throughput(Throughput::Elements(s.element_count() as u64));
        let label = shape_label(s);

        for axis in 0..s.rank() {
            let bench_label = format!("{label}/axis{axis}");
            group.bench_with_input(
                BenchmarkId::new("f64/tensor", &bench_label),
                &bench_label,
                |b, _| {
                    b.iter(|| {
                        let r = t.ifft_single_axis(axis).expect("ifft must succeed");
                        black_box(r);
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_tensor_f64_ifft_axes(c: &mut Criterion) {
    let mut group = c.benchmark_group("ifft_axes");
    let shapes = [
        (shape![1024], HashSet::from([0usize])),
        (shape![64, 64], HashSet::from([0usize, 1])),
        (shape![64, 64, 64], HashSet::from([0usize, 2])),
        (shape![64, 64, 64], HashSet::from([0usize, 1, 2])),
    ];

    for (s, axes) in &shapes {
        let t = Tensor::<f64>::rand(s);
        group.throughput(Throughput::Elements(s.element_count() as u64));
        let mut label = shape_label(s);
        let mut axes_label = String::from("axes=");
        let mut sorted: Vec<_> = axes.iter().copied().collect();
        sorted.sort();
        for (i, a) in sorted.iter().enumerate() {
            if i > 0 {
                axes_label.push(',');
            }
            axes_label.push_str(&a.to_string());
        }
        label.push('/');
        label.push_str(&axes_label);

        group.bench_with_input(BenchmarkId::new("f64/tensor", &label), &label, |b, _| {
            b.iter(|| {
                let r = t.ifft_axes(axes).expect("ifft must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_tensor_f64_ifft(c: &mut Criterion) {
    let mut group = c.benchmark_group("ifft");
    let shapes = [
        shape![1024],
        shape![4096],
        shape![64, 64],
        shape![256, 256],
        shape![32, 32, 32],
    ];

    for s in &shapes {
        let t = Tensor::<f64>::rand(s);
        group.throughput(Throughput::Elements(s.element_count() as u64));
        let label = shape_label(s);

        group.bench_with_input(
            BenchmarkId::new("f64/tensor", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let r = t.ifft().expect("ifft must succeed");
                    black_box(r);
                });
            },
        );
    }
    group.finish();
}

fn bench_tensor_f64_fft_corr_axes(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_corr_axes");
    let shapes = [
        (shape![256], shape![5]),
        (shape![128, 128], shape![5, 5]),
        (shape![32, 32, 32], shape![5, 5, 5]),
    ];

    for (s, k) in &shapes {
        let a = Tensor::<f64>::rand(s);
        let b = Tensor::<f64>::rand(k);
        group.throughput(Throughput::Elements(s.element_count() as u64));
        let label = format!(
            "{}__{}",
            shape_label(s),
            shape_label(k)
        );
        let axes: HashSet<usize> = (0..s.rank()).collect();

        group.bench_with_input(BenchmarkId::new("f64/tensor", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a
                    .fft_corr_axes(&b, &axes)
                    .expect("fft_corr_axes must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_tensor_f64_fft_conv_axes(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_conv_axes");
    let shapes = [
        (shape![256], shape![5]),
        (shape![128, 128], shape![5, 5]),
        (shape![32, 32, 32], shape![5, 5, 5]),
    ];

    for (s, k) in &shapes {
        let a = Tensor::<f64>::rand(s);
        let b = Tensor::<f64>::rand(k);
        group.throughput(Throughput::Elements(s.element_count() as u64));
        let label = format!("{}__{}", shape_label(s), shape_label(k));
        let axes: HashSet<usize> = (0..s.rank()).collect();

        group.bench_with_input(BenchmarkId::new("f64/tensor", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a
                    .fft_conv_axes(&b, &axes)
                    .expect("fft_conv_axes must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_tensor_f64_fft_corr(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_corr");
    let shapes = [
        (shape![256], shape![5]),
        (shape![128, 128], shape![5, 5]),
        (shape![32, 32, 32], shape![5, 5, 5]),
    ];

    for (s, k) in &shapes {
        let a = Tensor::<f64>::rand(s);
        let b = Tensor::<f64>::rand(k);
        group.throughput(Throughput::Elements(s.element_count() as u64));
        let label = format!("{}__{}", shape_label(s), shape_label(k));

        group.bench_with_input(BenchmarkId::new("f64/tensor", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.fft_corr(&b).expect("fft_corr must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_tensor_f64_fft_conv(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_conv");
    let shapes = [
        (shape![256], shape![5]),
        (shape![128, 128], shape![5, 5]),
        (shape![32, 32, 32], shape![5, 5, 5]),
    ];

    for (s, k) in &shapes {
        let a = Tensor::<f64>::rand(s);
        let b = Tensor::<f64>::rand(k);
        group.throughput(Throughput::Elements(s.element_count() as u64));
        let label = format!("{}__{}", shape_label(s), shape_label(k));

        group.bench_with_input(BenchmarkId::new("f64/tensor", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.fft_conv(&b).expect("fft_conv must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_matrix_f64_fft_rows(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_rows");
    for &(rows, cols) in &[(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)] {
        let m = Matrix::<f64>::rand(rows, cols);
        group.throughput(Throughput::Elements((rows * cols) as u64));
        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("f64/matrix", &label), &label, |b, _| {
            b.iter(|| {
                let r = m.fft_rows();
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_matrix_f64_fft_cols(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_cols");
    for &(rows, cols) in &[(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)] {
        let m = Matrix::<f64>::rand(rows, cols);
        group.throughput(Throughput::Elements((rows * cols) as u64));
        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("f64/matrix", &label), &label, |b, _| {
            b.iter(|| {
                let r = m.fft_cols();
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_matrix_f64_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft");
    for &(rows, cols) in &[(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)] {
        let m = Matrix::<f64>::rand(rows, cols);
        group.throughput(Throughput::Elements((rows * cols) as u64));
        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("f64/matrix", &label), &label, |b, _| {
            b.iter(|| {
                let r = m.fft();
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_matrix_f64_ifft_rows(c: &mut Criterion) {
    let mut group = c.benchmark_group("ifft_rows");
    for &(rows, cols) in &[(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)] {
        let m = Matrix::<f64>::rand(rows, cols);
        group.throughput(Throughput::Elements((rows * cols) as u64));
        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("f64/matrix", &label), &label, |b, _| {
            b.iter(|| {
                let r = m.ifft_rows();
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_matrix_f64_ifft_cols(c: &mut Criterion) {
    let mut group = c.benchmark_group("ifft_cols");
    for &(rows, cols) in &[(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)] {
        let m = Matrix::<f64>::rand(rows, cols);
        group.throughput(Throughput::Elements((rows * cols) as u64));
        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("f64/matrix", &label), &label, |b, _| {
            b.iter(|| {
                let r = m.ifft_cols();
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_matrix_f64_ifft(c: &mut Criterion) {
    let mut group = c.benchmark_group("ifft");
    for &(rows, cols) in &[(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)] {
        let m = Matrix::<f64>::rand(rows, cols);
        group.throughput(Throughput::Elements((rows * cols) as u64));
        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("f64/matrix", &label), &label, |b, _| {
            b.iter(|| {
                let r = m.clone().ifft();
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_matrix_f64_fft_conv_cols(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_conv_cols");
    for &(rows, cols) in &[(32, 32), (64, 64), (128, 128), (256, 256)] {
        let m = Matrix::<f64>::rand(rows, cols);
        let k = Matrix::<f64>::rand(5, cols);
        group.throughput(Throughput::Elements((rows * cols) as u64));
        let label = format!("{rows}x{cols}__5x{cols}");

        group.bench_with_input(BenchmarkId::new("f64/matrix", &label), &label, |b, _| {
            b.iter(|| {
                let r = m
                    .fft_conv_cols(&k)
                    .expect("fft_conv_cols must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_matrix_f64_fft_corr_cols(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_corr_cols");
    for &(rows, cols) in &[(32, 32), (64, 64), (128, 128), (256, 256)] {
        let m = Matrix::<f64>::rand(rows, cols);
        let k = Matrix::<f64>::rand(5, cols);
        group.throughput(Throughput::Elements((rows * cols) as u64));
        let label = format!("{rows}x{cols}__5x{cols}");

        group.bench_with_input(BenchmarkId::new("f64/matrix", &label), &label, |b, _| {
            b.iter(|| {
                let r = m
                    .fft_corr_cols(&k)
                    .expect("fft_corr_cols must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_matrix_f64_fft_conv_rows(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_conv_rows");
    for &(rows, cols) in &[(32, 32), (64, 64), (128, 128), (256, 256)] {
        let m = Matrix::<f64>::rand(rows, cols);
        let k = Matrix::<f64>::rand(rows, 5);
        group.throughput(Throughput::Elements((rows * cols) as u64));
        let label = format!("{rows}x{cols}__{rows}x5");

        group.bench_with_input(BenchmarkId::new("f64/matrix", &label), &label, |b, _| {
            b.iter(|| {
                let r = m
                    .fft_conv_rows(&k)
                    .expect("fft_conv_rows must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_matrix_f64_fft_corr_rows(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_corr_rows");
    for &(rows, cols) in &[(32, 32), (64, 64), (128, 128), (256, 256)] {
        let m = Matrix::<f64>::rand(rows, cols);
        let k = Matrix::<f64>::rand(rows, 5);
        group.throughput(Throughput::Elements((rows * cols) as u64));
        let label = format!("{rows}x{cols}__{rows}x5");

        group.bench_with_input(BenchmarkId::new("f64/matrix", &label), &label, |b, _| {
            b.iter(|| {
                let r = m
                    .fft_corr_rows(&k)
                    .expect("fft_corr_rows must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_matrix_f64_fft_conv(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_conv");
    for &(rows, cols) in &[(32, 32), (64, 64), (128, 128), (256, 256)] {
        let m = Matrix::<f64>::rand(rows, cols);
        let k = Matrix::<f64>::rand(5, 5);
        group.throughput(Throughput::Elements((rows * cols) as u64));
        let label = format!("{rows}x{cols}__5x5");

        group.bench_with_input(BenchmarkId::new("f64/matrix", &label), &label, |b, _| {
            b.iter(|| {
                let r = m.fft_conv(&k);
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_matrix_f64_fft_corr(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_corr");
    for &(rows, cols) in &[(32, 32), (64, 64), (128, 128), (256, 256)] {
        let m = Matrix::<f64>::rand(rows, cols);
        let k = Matrix::<f64>::rand(5, 5);
        group.throughput(Throughput::Elements((rows * cols) as u64));
        let label = format!("{rows}x{cols}__5x5");

        group.bench_with_input(BenchmarkId::new("f64/matrix", &label), &label, |b, _| {
            b.iter(|| {
                let r = m.fft_corr(&k);
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_tensor_c64_fft_single_axis(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_single_axis");
    let shapes = [
        shape![1024],
        shape![4096],
        shape![16384],
        shape![64, 64],
        shape![256, 256],
        shape![64, 64, 64],
    ];

    for s in &shapes {
        let t = rand_complex_tensor(s);
        group.throughput(Throughput::Elements(s.element_count() as u64));
        let label = shape_label(s);

        for axis in 0..s.rank() {
            let bench_label = format!("{label}/axis{axis}");
            group.bench_with_input(
                BenchmarkId::new("c64/tensor", &bench_label),
                &bench_label,
                |b, _| {
                    b.iter(|| {
                        let r = t.fft_single_axis(axis).expect("fft must succeed");
                        black_box(r);
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_tensor_c64_fft_axes(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_axes");
    let shapes = [
        (shape![1024], HashSet::from([0usize])),
        (shape![64, 64], HashSet::from([0usize, 1])),
        (shape![64, 64, 64], HashSet::from([0usize, 2])),
        (shape![64, 64, 64], HashSet::from([0usize, 1, 2])),
    ];

    for (s, axes) in &shapes {
        let t = rand_complex_tensor(s);
        group.throughput(Throughput::Elements(s.element_count() as u64));
        let mut label = shape_label(s);
        let mut axes_label = String::from("axes=");
        let mut sorted: Vec<_> = axes.iter().copied().collect();
        sorted.sort();
        for (i, a) in sorted.iter().enumerate() {
            if i > 0 {
                axes_label.push(',');
            }
            axes_label.push_str(&a.to_string());
        }
        label.push('/');
        label.push_str(&axes_label);

        group.bench_with_input(BenchmarkId::new("c64/tensor", &label), &label, |b, _| {
            b.iter(|| {
                let r = t.fft_axes(axes).expect("fft must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_tensor_c64_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft");
    let shapes = [
        shape![1024],
        shape![4096],
        shape![64, 64],
        shape![256, 256],
        shape![32, 32, 32],
    ];

    for s in &shapes {
        let t = rand_complex_tensor(s);
        group.throughput(Throughput::Elements(s.element_count() as u64));
        let label = shape_label(s);

        group.bench_with_input(
            BenchmarkId::new("c64/tensor", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let r = t.fft().expect("fft must succeed");
                    black_box(r);
                });
            },
        );
    }
    group.finish();
}

fn bench_tensor_c64_ifft_single_axis(c: &mut Criterion) {
    let mut group = c.benchmark_group("ifft_single_axis");
    let shapes = [
        shape![1024],
        shape![4096],
        shape![16384],
        shape![64, 64],
        shape![256, 256],
        shape![64, 64, 64],
    ];

    for s in &shapes {
        let t = rand_complex_tensor(s);
        group.throughput(Throughput::Elements(s.element_count() as u64));
        let label = shape_label(s);

        for axis in 0..s.rank() {
            let bench_label = format!("{label}/axis{axis}");
            group.bench_with_input(
                BenchmarkId::new("c64/tensor", &bench_label),
                &bench_label,
                |b, _| {
                    b.iter(|| {
                        let r = t.ifft_single_axis(axis).expect("ifft must succeed");
                        black_box(r);
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_tensor_c64_ifft_axes(c: &mut Criterion) {
    let mut group = c.benchmark_group("ifft_axes");
    let shapes = [
        (shape![1024], HashSet::from([0usize])),
        (shape![64, 64], HashSet::from([0usize, 1])),
        (shape![64, 64, 64], HashSet::from([0usize, 2])),
        (shape![64, 64, 64], HashSet::from([0usize, 1, 2])),
    ];

    for (s, axes) in &shapes {
        let t = rand_complex_tensor(s);
        group.throughput(Throughput::Elements(s.element_count() as u64));
        let mut label = shape_label(s);
        let mut axes_label = String::from("axes=");
        let mut sorted: Vec<_> = axes.iter().copied().collect();
        sorted.sort();
        for (i, a) in sorted.iter().enumerate() {
            if i > 0 {
                axes_label.push(',');
            }
            axes_label.push_str(&a.to_string());
        }
        label.push('/');
        label.push_str(&axes_label);

        group.bench_with_input(BenchmarkId::new("c64/tensor", &label), &label, |b, _| {
            b.iter(|| {
                let r = t.ifft_axes(axes).expect("ifft must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_tensor_c64_ifft(c: &mut Criterion) {
    let mut group = c.benchmark_group("ifft");
    let shapes = [
        shape![1024],
        shape![4096],
        shape![64, 64],
        shape![256, 256],
        shape![32, 32, 32],
    ];

    for s in &shapes {
        let t = rand_complex_tensor(s);
        group.throughput(Throughput::Elements(s.element_count() as u64));
        let label = shape_label(s);

        group.bench_with_input(
            BenchmarkId::new("c64/tensor", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let r = t.ifft().expect("ifft must succeed");
                    black_box(r);
                });
            },
        );
    }
    group.finish();
}

fn bench_tensor_c64_fft_corr_axes(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_corr_axes");
    let shapes = [
        (shape![256], shape![5]),
        (shape![128, 128], shape![5, 5]),
        (shape![32, 32, 32], shape![5, 5, 5]),
    ];

    for (s, k) in &shapes {
        let a = rand_complex_tensor(s);
        let b = rand_complex_tensor(k);
        group.throughput(Throughput::Elements(s.element_count() as u64));
        let label = format!("{}__{}", shape_label(s), shape_label(k));
        let axes: HashSet<usize> = (0..s.rank()).collect();

        group.bench_with_input(BenchmarkId::new("c64/tensor", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a
                    .fft_corr_axes(&b, &axes)
                    .expect("fft_corr_axes must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_tensor_c64_fft_conv_axes(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_conv_axes");
    let shapes = [
        (shape![256], shape![5]),
        (shape![128, 128], shape![5, 5]),
        (shape![32, 32, 32], shape![5, 5, 5]),
    ];

    for (s, k) in &shapes {
        let a = rand_complex_tensor(s);
        let b = rand_complex_tensor(k);
        group.throughput(Throughput::Elements(s.element_count() as u64));
        let label = format!("{}__{}", shape_label(s), shape_label(k));
        let axes: HashSet<usize> = (0..s.rank()).collect();

        group.bench_with_input(BenchmarkId::new("c64/tensor", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a
                    .fft_conv_axes(&b, &axes)
                    .expect("fft_conv_axes must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_tensor_c64_fft_corr(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_corr");
    let shapes = [
        (shape![256], shape![5]),
        (shape![128, 128], shape![5, 5]),
        (shape![32, 32, 32], shape![5, 5, 5]),
    ];

    for (s, k) in &shapes {
        let a = rand_complex_tensor(s);
        let b = rand_complex_tensor(k);
        group.throughput(Throughput::Elements(s.element_count() as u64));
        let label = format!("{}__{}", shape_label(s), shape_label(k));

        group.bench_with_input(BenchmarkId::new("c64/tensor", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.fft_corr(&b).expect("fft_corr must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_tensor_c64_fft_conv(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_conv");
    let shapes = [
        (shape![256], shape![5]),
        (shape![128, 128], shape![5, 5]),
        (shape![32, 32, 32], shape![5, 5, 5]),
    ];

    for (s, k) in &shapes {
        let a = rand_complex_tensor(s);
        let b = rand_complex_tensor(k);
        group.throughput(Throughput::Elements(s.element_count() as u64));
        let label = format!("{}__{}", shape_label(s), shape_label(k));

        group.bench_with_input(BenchmarkId::new("c64/tensor", &label), &label, |bench, _| {
            bench.iter(|| {
                let r = a.fft_conv(&b).expect("fft_conv must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Matrix<Complex64> FFT benches
// ---------------------------------------------------------------------------

fn bench_matrix_c64_fft_rows(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_rows");
    for &(rows, cols) in &[(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)] {
        let m = rand_complex_matrix(rows, cols);
        group.throughput(Throughput::Elements((rows * cols) as u64));
        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("c64/matrix", &label), &label, |b, _| {
            b.iter(|| {
                let r = m.fft_rows();
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_matrix_c64_fft_cols(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_cols");
    for &(rows, cols) in &[(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)] {
        let m = rand_complex_matrix(rows, cols);
        group.throughput(Throughput::Elements((rows * cols) as u64));
        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("c64/matrix", &label), &label, |b, _| {
            b.iter(|| {
                let r = m.fft_cols();
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_matrix_c64_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft");
    for &(rows, cols) in &[(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)] {
        let m = rand_complex_matrix(rows, cols);
        group.throughput(Throughput::Elements((rows * cols) as u64));
        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("c64/matrix", &label), &label, |b, _| {
            b.iter(|| {
                let r = m.fft();
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_matrix_c64_ifft_rows(c: &mut Criterion) {
    let mut group = c.benchmark_group("ifft_rows");
    for &(rows, cols) in &[(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)] {
        let m = rand_complex_matrix(rows, cols);
        group.throughput(Throughput::Elements((rows * cols) as u64));
        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("c64/matrix", &label), &label, |b, _| {
            b.iter(|| {
                let r = m.ifft_rows();
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_matrix_c64_ifft_cols(c: &mut Criterion) {
    let mut group = c.benchmark_group("ifft_cols");
    for &(rows, cols) in &[(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)] {
        let m = rand_complex_matrix(rows, cols);
        group.throughput(Throughput::Elements((rows * cols) as u64));
        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("c64/matrix", &label), &label, |b, _| {
            b.iter(|| {
                let r = m.ifft_cols();
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_matrix_c64_ifft(c: &mut Criterion) {
    let mut group = c.benchmark_group("ifft");
    for &(rows, cols) in &[(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)] {
        let m = rand_complex_matrix(rows, cols);
        group.throughput(Throughput::Elements((rows * cols) as u64));
        let label = format!("{rows}x{cols}");

        group.bench_with_input(BenchmarkId::new("c64/matrix", &label), &label, |b, _| {
            b.iter(|| {
                let r = m.clone().ifft();
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_matrix_c64_fft_conv_cols(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_conv_cols");
    for &(rows, cols) in &[(32, 32), (64, 64), (128, 128), (256, 256)] {
        let m = rand_complex_matrix(rows, cols);
        let k = rand_complex_matrix(5, cols);
        group.throughput(Throughput::Elements((rows * cols) as u64));
        let label = format!("{rows}x{cols}__5x{cols}");

        group.bench_with_input(BenchmarkId::new("c64/matrix", &label), &label, |b, _| {
            b.iter(|| {
                let r = m
                    .fft_conv_cols(&k)
                    .expect("fft_conv_cols must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_matrix_c64_fft_corr_cols(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_corr_cols");
    for &(rows, cols) in &[(32, 32), (64, 64), (128, 128), (256, 256)] {
        let m = rand_complex_matrix(rows, cols);
        let k = rand_complex_matrix(5, cols);
        group.throughput(Throughput::Elements((rows * cols) as u64));
        let label = format!("{rows}x{cols}__5x{cols}");

        group.bench_with_input(BenchmarkId::new("c64/matrix", &label), &label, |b, _| {
            b.iter(|| {
                let r = m
                    .fft_corr_cols(&k)
                    .expect("fft_corr_cols must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_matrix_c64_fft_conv_rows(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_conv_rows");
    for &(rows, cols) in &[(32, 32), (64, 64), (128, 128), (256, 256)] {
        let m = rand_complex_matrix(rows, cols);
        let k = rand_complex_matrix(rows, 5);
        group.throughput(Throughput::Elements((rows * cols) as u64));
        let label = format!("{rows}x{cols}__{rows}x5");

        group.bench_with_input(BenchmarkId::new("c64/matrix", &label), &label, |b, _| {
            b.iter(|| {
                let r = m
                    .fft_conv_rows(&k)
                    .expect("fft_conv_rows must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_matrix_c64_fft_corr_rows(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_corr_rows");
    for &(rows, cols) in &[(32, 32), (64, 64), (128, 128), (256, 256)] {
        let m = rand_complex_matrix(rows, cols);
        let k = rand_complex_matrix(rows, 5);
        group.throughput(Throughput::Elements((rows * cols) as u64));
        let label = format!("{rows}x{cols}__{rows}x5");

        group.bench_with_input(BenchmarkId::new("c64/matrix", &label), &label, |b, _| {
            b.iter(|| {
                let r = m
                    .fft_corr_rows(&k)
                    .expect("fft_corr_rows must succeed");
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_matrix_c64_fft_conv(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_conv");
    for &(rows, cols) in &[(32, 32), (64, 64), (128, 128), (256, 256)] {
        let m = rand_complex_matrix(rows, cols);
        let k = rand_complex_matrix(5, 5);
        group.throughput(Throughput::Elements((rows * cols) as u64));
        let label = format!("{rows}x{cols}__5x5");

        group.bench_with_input(BenchmarkId::new("c64/matrix", &label), &label, |b, _| {
            b.iter(|| {
                let r = m.fft_conv(&k);
                black_box(r);
            });
        });
    }
    group.finish();
}

fn bench_matrix_c64_fft_corr(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_corr");
    for &(rows, cols) in &[(32, 32), (64, 64), (128, 128), (256, 256)] {
        let m = rand_complex_matrix(rows, cols);
        let k = rand_complex_matrix(5, 5);
        group.throughput(Throughput::Elements((rows * cols) as u64));
        let label = format!("{rows}x{cols}__5x5");

        group.bench_with_input(BenchmarkId::new("c64/matrix", &label), &label, |b, _| {
            b.iter(|| {
                let r = m.fft_corr(&k);
                black_box(r);
            });
        });
    }
    group.finish();
}

criterion_group!(
    name = fft_benches;
    config = Criterion::default().sample_size(10);
    targets =
        bench_tensor_f64_fft_single_axis,
        bench_tensor_f64_fft_axes,
        bench_tensor_f64_fft,
        bench_tensor_f64_ifft_single_axis,
        bench_tensor_f64_ifft_axes,
        bench_tensor_f64_ifft,
        bench_tensor_f64_fft_corr_axes,
        bench_tensor_f64_fft_conv_axes,
        bench_tensor_f64_fft_corr,
        bench_tensor_f64_fft_conv,
        bench_matrix_f64_fft_rows,
        bench_matrix_f64_fft_cols,
        bench_matrix_f64_fft,
        bench_matrix_f64_ifft_rows,
        bench_matrix_f64_ifft_cols,
        bench_matrix_f64_ifft,
        bench_matrix_f64_fft_conv_cols,
        bench_matrix_f64_fft_corr_cols,
        bench_matrix_f64_fft_conv_rows,
        bench_matrix_f64_fft_corr_rows,
        bench_matrix_f64_fft_conv,
        bench_matrix_f64_fft_corr,
        bench_tensor_c64_fft_single_axis,
        bench_tensor_c64_fft_axes,
        bench_tensor_c64_fft,
        bench_tensor_c64_ifft_single_axis,
        bench_tensor_c64_ifft_axes,
        bench_tensor_c64_ifft,
        bench_tensor_c64_fft_corr_axes,
        bench_tensor_c64_fft_conv_axes,
        bench_tensor_c64_fft_corr,
        bench_tensor_c64_fft_conv,
        bench_matrix_c64_fft_rows,
        bench_matrix_c64_fft_cols,
        bench_matrix_c64_fft,
        bench_matrix_c64_ifft_rows,
        bench_matrix_c64_ifft_cols,
        bench_matrix_c64_ifft,
        bench_matrix_c64_fft_conv_cols,
        bench_matrix_c64_fft_corr_cols,
        bench_matrix_c64_fft_conv_rows,
        bench_matrix_c64_fft_corr_rows,
        bench_matrix_c64_fft_conv,
        bench_matrix_c64_fft_corr,
);

criterion_main!(fft_benches);
