//! Benchmarks for the elementwise math operations declared in
//! `src/math/f64_c64_ops.rs`.

use std::hint::black_box;

use criterion::{criterion_group, BenchmarkId, Criterion, Throughput};
use num::complex::Complex64;

use tensor_math::definitions::matrix::Matrix;
use tensor_math::definitions::shape::Shape;
use tensor_math::definitions::tensor::Tensor;
use tensor_math::definitions::transpose::Transpose;

/// Standard sweep of tensor shapes used for the f64 benches.
const TENSOR_SHAPES: &[&[usize]] = &[
    &[1000usize],
    &[200, 200],
    &[100, 100, 100],
    &[1_000_000usize],
];

/// Standard sweep of matrix shapes used for the f64 benches.
const MATRIX_SHAPES: &[(usize, usize)] = &[
    (100, 100),
    (1000, 1000),
    (100, 10_000),
    (10_000, 100),
];

fn label_shape(shape: &Shape) -> String {
    let mut s = String::new();
    for i in 0..shape.rank() {
        s.push_str(&format!("{}x", shape[i]));
    }
    s.pop();
    s
}

fn random_complex_tensor(shape: &Shape) -> Tensor<Complex64> {
    let re = Tensor::<f64>::rand(shape);
    let im = Tensor::<f64>::rand(shape);
    re.into_complex() + im.into_complex().par_map(|v| v * Complex64::I)
}

fn random_complex_matrix(rows: usize, cols: usize) -> Matrix<Complex64> {
    let re = Matrix::<f64>::rand(rows, cols);
    let im = Matrix::<f64>::rand(rows, cols);
    re.into_complex() + im.into_complex().par_map(|v| v * Complex64::I)
}

fn bench_f64_tensor_elementwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("f64_tensor_elementwise");

    for shape in TENSOR_SHAPES {
        let s = Shape::new(shape.to_vec());
        let n = s.element_count();
        group.throughput(Throughput::Elements(n as u64));

        let label = label_shape(&s);

        // --- into_complex ---
        group.bench_with_input(
            BenchmarkId::new("into_complex", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let t = Tensor::<f64>::rand(&s);
                    let r = t.into_complex();
                    black_box(r);
                });
            },
        );

        // --- simple unary elementwise (one benchmark each) ---
        macro_rules! unary {
            ($name:ident) => {
                group.bench_with_input(
                    BenchmarkId::new(stringify!($name), &label),
                    &label,
                    |b, _| {
                        b.iter(|| {
                            let t = Tensor::<f64>::rand(&s);
                            let r = t.$name();
                            black_box(r);
                        });
                    },
                );
            };
        }

        unary!(exp);
        unary!(ln);
        unary!(log2);
        unary!(log10);
        unary!(sqrt);
        unary!(cbrt);
        unary!(recip);
        unary!(abs);
        unary!(sin);
        unary!(cos);
        unary!(tan);
        unary!(asin);
        unary!(acos);
        unary!(atan);
        unary!(sinh);
        unary!(cosh);
        unary!(tanh);
        unary!(asinh);
        unary!(acosh);
        unary!(atanh);
        unary!(sigmoid);
        unary!(relu);
    }

    group.finish();
}

fn bench_f64_tensor_param_elementwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("f64_tensor_elementwise");

    for shape in TENSOR_SHAPES {
        let s = Shape::new(shape.to_vec());
        let n = s.element_count();
        group.throughput(Throughput::Elements(n as u64));

        let label = label_shape(&s);

        // `log(n)`, `exp_base_n(n)`, `pow(n)`, `leaky_relu(alpha)` all take a
        // scalar parameter; the value is essentially irrelevant to throughput.
        group.bench_with_input(
            BenchmarkId::new("log", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let t = Tensor::<f64>::rand(&s);
                    let r = t.log(2.0);
                    black_box(r);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("exp_base_n", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let t = Tensor::<f64>::rand(&s);
                    let r = t.exp_base_n(2.0);
                    black_box(r);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("pow", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let t = Tensor::<f64>::rand(&s);
                    let r = t.pow(2.0);
                    black_box(r);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("leaky_relu", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let t = Tensor::<f64>::rand(&s);
                    let r = t.leaky_relu(0.01);
                    black_box(r);
                });
            },
        );
    }

    group.finish();
}

// Methods that depend on a global sum/magnitude: softmax, norm_l1, norm_l2,
// born_probabilities, plus the scalar returns mag / mag_2.
fn bench_f64_tensor_aggregates(c: &mut Criterion) {
    let mut group = c.benchmark_group("f64_tensor_aggregates");

    for shape in TENSOR_SHAPES {
        let s = Shape::new(shape.to_vec());
        let n = s.element_count();
        group.throughput(Throughput::Elements(n as u64));

        let label = label_shape(&s);

        group.bench_with_input(
            BenchmarkId::new("softmax", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let t = Tensor::<f64>::rand(&s);
                    let r = t.softmax();
                    black_box(r);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("norm_l1", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let t = Tensor::<f64>::rand(&s);
                    let r = t.norm_l1();
                    black_box(r);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("norm_l2", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let t = Tensor::<f64>::rand(&s);
                    let r = t.norm_l2();
                    black_box(r);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("born_probabilities", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let t = Tensor::<f64>::rand(&s);
                    let r = t.born_probabilities();
                    black_box(r);
                });
            },
        );

        // `mag` / `mag_2` return scalars; throughput is "1 element op per call"
        // in the sense that each call still walks every element.
        group.bench_with_input(
            BenchmarkId::new("mag", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let t = Tensor::<f64>::rand(&s);
                    let r = t.mag();
                    black_box(r);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("mag_2", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let t = Tensor::<f64>::rand(&s);
                    let r = t.mag_2();
                    black_box(r);
                });
            },
        );
    }

    group.finish();
}

fn bench_f64_matrix_elementwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("f64_matrix_elementwise");

    for &(rows, cols) in MATRIX_SHAPES {
        let n = rows * cols;
        group.throughput(Throughput::Elements(n as u64));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(
            BenchmarkId::new("into_complex", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let m = Matrix::<f64>::rand(rows, cols);
                    let r = m.into_complex();
                    black_box(r);
                });
            },
        );

        macro_rules! unary {
            ($name:ident) => {
                group.bench_with_input(
                    BenchmarkId::new(stringify!($name), &label),
                    &label,
                    |b, _| {
                        b.iter(|| {
                            let m = Matrix::<f64>::rand(rows, cols);
                            let r = m.$name();
                            black_box(r);
                        });
                    },
                );
            };
        }

        unary!(exp);
        unary!(ln);
        unary!(log2);
        unary!(log10);
        unary!(sqrt);
        unary!(cbrt);
        unary!(recip);
        unary!(abs);
        unary!(sin);
        unary!(cos);
        unary!(tan);
        unary!(asin);
        unary!(acos);
        unary!(atan);
        unary!(sinh);
        unary!(cosh);
        unary!(tanh);
        unary!(asinh);
        unary!(acosh);
        unary!(atanh);
        unary!(sigmoid);
        unary!(relu);
    }

    group.finish();
}

fn bench_f64_matrix_param_elementwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("f64_matrix_elementwise");

    for &(rows, cols) in MATRIX_SHAPES {
        let n = rows * cols;
        group.throughput(Throughput::Elements(n as u64));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(
            BenchmarkId::new("log", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let m = Matrix::<f64>::rand(rows, cols);
                    let r = m.log(2.0);
                    black_box(r);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("exp_base_n", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let m = Matrix::<f64>::rand(rows, cols);
                    let r = m.exp_base_n(2.0);
                    black_box(r);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("pow", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let m = Matrix::<f64>::rand(rows, cols);
                    let r = m.pow(2.0);
                    black_box(r);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("leaky_relu", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let m = Matrix::<f64>::rand(rows, cols);
                    let r = m.leaky_relu(0.01);
                    black_box(r);
                });
            },
        );
    }

    group.finish();
}

fn bench_f64_matrix_aggregates(c: &mut Criterion) {
    let mut group = c.benchmark_group("f64_matrix_aggregates");

    for &(rows, cols) in MATRIX_SHAPES {
        let n = rows * cols;
        group.throughput(Throughput::Elements(n as u64));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(
            BenchmarkId::new("softmax", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let m = Matrix::<f64>::rand(rows, cols);
                    let r = m.softmax();
                    black_box(r);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("norm_l1", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let m = Matrix::<f64>::rand(rows, cols);
                    let r = m.norm_l1();
                    black_box(r);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("norm_l2", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let m = Matrix::<f64>::rand(rows, cols);
                    let r = m.norm_l2();
                    black_box(r);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("born_probabilities", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let m = Matrix::<f64>::rand(rows, cols);
                    let r = m.born_probabilities();
                    black_box(r);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("mag", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let m = Matrix::<f64>::rand(rows, cols);
                    let r = m.mag();
                    black_box(r);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("mag_2", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let m = Matrix::<f64>::rand(rows, cols);
                    let r = m.mag_2();
                    black_box(r);
                });
            },
        );
    }

    group.finish();
}

fn bench_c64_tensor_elementwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("c64_tensor_elementwise");

    for shape in TENSOR_SHAPES {
        let s = Shape::new(shape.to_vec());
        let n = s.element_count();
        group.throughput(Throughput::Elements(n as u64));

        let label = label_shape(&s);

        group.bench_with_input(
            BenchmarkId::new("re", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let t = random_complex_tensor(&s);
                    let r = t.re();
                    black_box(r);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("im", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let t = random_complex_tensor(&s);
                    let r = t.im();
                    black_box(r);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("conj", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let t = random_complex_tensor(&s);
                    let r = t.conj();
                    black_box(r);
                });
            },
        );

        macro_rules! unary {
            ($name:ident) => {
                group.bench_with_input(
                    BenchmarkId::new(stringify!($name), &label),
                    &label,
                    |b, _| {
                        b.iter(|| {
                            let t = random_complex_tensor(&s);
                            let r = t.$name();
                            black_box(r);
                        });
                    },
                );
            };
        }

        unary!(exp);
        unary!(ln);
        unary!(log2);
        unary!(log10);
        unary!(sqrt);
        unary!(cbrt);
        unary!(recip);
        unary!(abs);
        unary!(sin);
        unary!(cos);
        unary!(tan);
        unary!(asin);
        unary!(acos);
        unary!(atan);
        unary!(sinh);
        unary!(cosh);
        unary!(tanh);
        unary!(asinh);
        unary!(acosh);
        unary!(atanh);
    }

    group.finish();
}

fn bench_c64_tensor_param_elementwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("c64_tensor_elementwise");

    for shape in TENSOR_SHAPES {
        let s = Shape::new(shape.to_vec());
        let n = s.element_count();
        group.throughput(Throughput::Elements(n as u64));

        let label = label_shape(&s);

        // `log(n)`, `exp_base_n(n)`, `pow(n)` take a scalar parameter.
        group.bench_with_input(
            BenchmarkId::new("log", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let t = random_complex_tensor(&s);
                    let r = t.log(2.0);
                    black_box(r);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("exp_base_n", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let t = random_complex_tensor(&s);
                    let r = t.exp_base_n(Complex64::new(2.0, 0.0));
                    black_box(r);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("pow", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let t = random_complex_tensor(&s);
                    let r = t.pow(Complex64::new(2.0, 0.0));
                    black_box(r);
                });
            },
        );
    }

    group.finish();
}

fn bench_c64_tensor_conj_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("c64_tensor_conj_transpose");

    // `conj_transpose` needs a 2-D+ tensor with a non-trivial `Transpose`.
    for shape in &[&[64usize, 64][..], &[32, 32, 32], &[16, 16, 16, 16]] {
        let s = Shape::new(shape.to_vec());
        let n = s.element_count();
        group.throughput(Throughput::Elements(n as u64));

        let label = label_shape(&s);

        // Reverse the leading two axes: [1, 0, 2, 3, ...].
        let mut perm: Vec<usize> = (0..s.rank()).collect();
        if perm.len() >= 2 {
            perm.swap(0, 1);
        }
        let transpose = Transpose::new(&perm).expect("transpose must be valid");

        group.bench_with_input(
            BenchmarkId::new("conj_transpose", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let t = random_complex_tensor(&s);
                    let r = t
                        .conj_transpose(&transpose)
                        .expect("conj_transpose must succeed");
                    black_box(r);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("conj_transpose_mt", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let t = random_complex_tensor(&s);
                    let r = t
                        .conj_transpose_mt(&transpose)
                        .expect("conj_transpose_mt must succeed");
                    black_box(r);
                });
            },
        );
    }

    group.finish();
}

fn bench_c64_tensor_aggregates(c: &mut Criterion) {
    let mut group = c.benchmark_group("c64_tensor_aggregates");

    for shape in TENSOR_SHAPES {
        let s = Shape::new(shape.to_vec());
        let n = s.element_count();
        group.throughput(Throughput::Elements(n as u64));

        let label = label_shape(&s);

        group.bench_with_input(
            BenchmarkId::new("norm_l1", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let t = random_complex_tensor(&s);
                    let r = t.norm_l1();
                    black_box(r);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("norm_l2", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let t = random_complex_tensor(&s);
                    let r = t.norm_l2();
                    black_box(r);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("born_probabilities", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let t = random_complex_tensor(&s);
                    let r = t.born_probabilities();
                    black_box(r);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("mag", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let t = random_complex_tensor(&s);
                    let r = t.mag();
                    black_box(r);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("mag_2", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let t = random_complex_tensor(&s);
                    let r = t.mag_2();
                    black_box(r);
                });
            },
        );
    }

    group.finish();
}

fn bench_c64_matrix_elementwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("c64_matrix_elementwise");

    for &(rows, cols) in MATRIX_SHAPES {
        let n = rows * cols;
        group.throughput(Throughput::Elements(n as u64));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(
            BenchmarkId::new("re", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let m = random_complex_matrix(rows, cols);
                    let r = m.re();
                    black_box(r);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("im", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let m = random_complex_matrix(rows, cols);
                    let r = m.im();
                    black_box(r);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("conj", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let m = random_complex_matrix(rows, cols);
                    let r = m.conj();
                    black_box(r);
                });
            },
        );

        macro_rules! unary {
            ($name:ident) => {
                group.bench_with_input(
                    BenchmarkId::new(stringify!($name), &label),
                    &label,
                    |b, _| {
                        b.iter(|| {
                            let m = random_complex_matrix(rows, cols);
                            let r = m.$name();
                            black_box(r);
                        });
                    },
                );
            };
        }

        unary!(exp);
        unary!(ln);
        unary!(log2);
        unary!(log10);
        unary!(sqrt);
        unary!(cbrt);
        unary!(recip);
        unary!(abs);
        unary!(sin);
        unary!(cos);
        unary!(tan);
        unary!(asin);
        unary!(acos);
        unary!(atan);
        unary!(sinh);
        unary!(cosh);
        unary!(tanh);
        unary!(asinh);
        unary!(acosh);
        unary!(atanh);
    }

    group.finish();
}

fn bench_c64_matrix_param_elementwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("c64_matrix_elementwise");

    for &(rows, cols) in MATRIX_SHAPES {
        let n = rows * cols;
        group.throughput(Throughput::Elements(n as u64));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(
            BenchmarkId::new("log", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let m = random_complex_matrix(rows, cols);
                    let r = m.log(2.0);
                    black_box(r);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("exp_base_n", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let m = random_complex_matrix(rows, cols);
                    let r = m.exp_base_n(Complex64::new(2.0, 0.0));
                    black_box(r);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("pow", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let m = random_complex_matrix(rows, cols);
                    let r = m.pow(Complex64::new(2.0, 0.0));
                    black_box(r);
                });
            },
        );
    }

    group.finish();
}

fn bench_c64_matrix_conj_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("c64_matrix_conj_transpose");

    for &(rows, cols) in &[(64usize, 64usize), (128, 256), (1024, 1024)] {
        let n = rows * cols;
        group.throughput(Throughput::Elements(n as u64));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(
            BenchmarkId::new("conj_transpose", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let m = random_complex_matrix(rows, cols);
                    let r = m.conj_transpose();
                    black_box(r);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("conj_transpose_mt", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let m = random_complex_matrix(rows, cols);
                    let r = m.conj_transpose_mt();
                    black_box(r);
                });
            },
        );
    }

    group.finish();
}

fn bench_c64_matrix_aggregates(c: &mut Criterion) {
    let mut group = c.benchmark_group("c64_matrix_aggregates");

    for &(rows, cols) in MATRIX_SHAPES {
        let n = rows * cols;
        group.throughput(Throughput::Elements(n as u64));

        let label = format!("{rows}x{cols}");

        group.bench_with_input(
            BenchmarkId::new("norm_l1", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let m = random_complex_matrix(rows, cols);
                    let r = m.norm_l1();
                    black_box(r);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("norm_l2", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let m = random_complex_matrix(rows, cols);
                    let r = m.norm_l2();
                    black_box(r);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("born_probabilities", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let m = random_complex_matrix(rows, cols);
                    let r = m.born_probabilities();
                    black_box(r);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("mag", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let m = random_complex_matrix(rows, cols);
                    let r = m.mag();
                    black_box(r);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("mag_2", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    let m = random_complex_matrix(rows, cols);
                    let r = m.mag_2();
                    black_box(r);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    name = f64_c64_ops_benches;
    config = Criterion::default().sample_size(10);
    targets =
        bench_f64_tensor_elementwise,
        bench_f64_tensor_param_elementwise,
        bench_f64_tensor_aggregates,
        bench_f64_matrix_elementwise,
        bench_f64_matrix_param_elementwise,
        bench_f64_matrix_aggregates,
        bench_c64_tensor_elementwise,
        bench_c64_tensor_param_elementwise,
        bench_c64_tensor_conj_transpose,
        bench_c64_tensor_aggregates,
        bench_c64_matrix_elementwise,
        bench_c64_matrix_param_elementwise,
        bench_c64_matrix_conj_transpose,
        bench_c64_matrix_aggregates,
);
