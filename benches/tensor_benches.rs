use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use num::complex::Complex64;
use tensor_math::tensor::tensor::{Tensor, Shape, Matrix};
use tensor_math::tensor::tensor_math::{identity, pool_avg, pool_avg_mat, Transpose};
use tensor_math::shape;

pub fn bench_concat_mt(c: &mut Criterion) {
    let t1: Tensor<f64> = Tensor::rand(&shape![100, 100, 100]);
    let t2: Tensor<f64> = Tensor::rand(&shape![100, 2000, 100]);

    c.bench_function("concat_mt", |b| {
        b.iter(|| {
            t1.concat_mt(black_box(&t2), black_box(1)).unwrap();
        })
    });
}

pub fn bench_concat_mat_mt(c: &mut Criterion) {
    let m1 = Matrix::<f64>::rand(100, 100);
    let m2 = Matrix::<f64>::rand(200, 100);

    c.bench_function("concat_mat_mt", |b| {
        b.iter(|| {
            m1.concat_mt(black_box(&m2), black_box(0)).unwrap();
        });
    });
}

pub fn bench_transpose_mt(c: &mut Criterion) {
    let t1: Tensor<f64> = Tensor::rand(&shape![100, 50, 20]);

    c.bench_function("transpose_mt", |b| {
        b.iter(|| {
            t1.transpose_mt(black_box(&Transpose::new(&vec![2, 0, 1]).unwrap())).unwrap();
        })
    });
}

pub fn bench_transpose_mat_mt(c: &mut Criterion) {
    let m1 = Matrix::<f64>::rand(100, 20);

    c.bench_function("transpose_mat_mt", |b| {
        b.iter(|| {
            m1.transpose_mt();
        });
    });
}

pub fn bench_contract_mul_mt(c: &mut Criterion) {
    let t1: Tensor<f64> = Tensor::rand(&shape![100, 100]);
    let t2: Tensor<f64> = Tensor::rand(&shape![100, 100]);

    c.bench_function("contract_mul_mt", |b| {
        b.iter(|| {
            t1.contract_mul_mt(black_box(&t2)).unwrap();
        })
    });
}

pub fn bench_contract_mul_mat_mt(c: &mut Criterion) {
    let m1 = Matrix::<f64>::rand(100, 50);
    let m2= Matrix::<f64>::rand(50, 200);

    c.bench_function("contract_mul_mt_mat", |b| {
        b.iter(|| {
            m1.contract_mul_mt(black_box(&m2)).unwrap();
        });
    });
}

pub fn bench_kronecker_mt(c: &mut Criterion) {
    let t1: Tensor<f64> = Tensor::rand(&shape![10, 15]);
    let t2: Tensor<f64> = Tensor::rand(&shape![10, 20, 10]);

    c.bench_function("kronecker_mt", |b| {
        b.iter(|| {
            t1.kronecker_mt(black_box(&t2));
        })
    });
}

pub fn bench_kronecker_mat_mt(c: &mut Criterion) {
    let m1 = Matrix::<f64>::rand(100, 100);
    let m2 = Matrix::<f64>::rand(200, 55);

    c.bench_function("kronecker_mat_mt", |b| {
        b.iter(|| {
            m1.kronecker_mt(black_box(&m2));
        });
    });
}

pub fn bench_pool_mt(c: &mut Criterion) {
    let t1: Tensor<f64> = Tensor::rand(&shape![50, 50, 50]);

    c.bench_function("pool_mt", |b| {
        b.iter(|| {
            t1.pool_mt(black_box(&pool_avg), black_box(&shape![11, 13, 17]), black_box(&shape![5, 7, 3]));
        });
    });
}

pub fn bench_pool_mat_mt(c: &mut Criterion) {
    let m1 = Matrix::<f64>::rand(200, 200);

    c.bench_function("pool_mat_mt", |b| {
        b.iter(|| {
            m1.pool_mt(&black_box(pool_avg_mat), black_box((20, 15)), black_box((10, 5)))
        });
    });
}

pub fn bench_det(c: &mut Criterion) {
    let m1 = Matrix::<f64>::rand(100, 100);

    c.bench_function("det", |b| {
        b.iter(|| {
            m1.det();
        })
    });
}

pub fn bench_inv(c: &mut Criterion) {
    let mut m1 = Matrix::<f64>::rand(100, 100);

    if m1.det() == 0.0 {
        m1 = m1 + identity(100);
    }

    c.bench_function("inv", |b| {
        b.iter(|| {
            m1.inv().expect("");
        })
    });
}

pub fn bench_eigendecompose(c: &mut Criterion) {
    let m1 = Matrix::<f64>::rand(20, 20).transform_elementwise(|x| Complex64 { re: x, im: 0.0 });
    let m2 = Matrix::<f64>::rand(20, 20).transform_elementwise(|x| Complex64 { re: 0.0, im: x });
    let m = &m1 + &m2;

    c.bench_function("eigendecompose", |b| {
        b.iter(|| {
            m.eigendecompose();
        })
    });
}

pub fn bench_tensor_fft(c: &mut Criterion) {
    let t1 = Tensor::<f64>::rand(&shape![256, 50, 20]).transform_elementwise(|x| Complex64 { re: x, im: 0.0 });
    let t2 = Tensor::<f64>::rand(&shape![256, 50, 20]).transform_elementwise(|x| Complex64 { re: 0.0, im: x });
    let t = &t1 + &t2;

    c.bench_function("tensor_fft", |b| {
        b.iter(|| {
            t.fft();
        })
    });
}

pub fn bench_mat_fft(c: &mut Criterion) {
    let m1 = Matrix::<f64>::rand(256, 100).transform_elementwise(|x| Complex64 { re: x, im: 0.0 });
    let m2 = Matrix::<f64>::rand(256, 100).transform_elementwise(|x| Complex64 { re: 0.0, im: x });
    let m = &m1 + &m2;

    c.bench_function("mat_fft", |b| {
        b.iter(|| {
            m.fft();
        })
    });
}

criterion_group!(
    benches,
    bench_concat_mt,
    bench_concat_mat_mt,
    bench_transpose_mt,
    bench_transpose_mat_mt,
    bench_contract_mul_mt,
    bench_contract_mul_mat_mt,
    bench_kronecker_mt,
    bench_kronecker_mat_mt,
    bench_pool_mt,
    bench_pool_mat_mt,
    bench_det,
    bench_inv,
    bench_eigendecompose,
    bench_tensor_fft,
    bench_mat_fft,
);
criterion_main!(benches);