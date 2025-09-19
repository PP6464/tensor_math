use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use tensor_math::tensor::tensor::{Tensor, Shape};
use tensor_math::tensor::tensor_math::Transpose;
use tensor_math::ts;

pub fn bench_concat_mt(c: &mut Criterion) {
    let t1: Tensor<f64> = Tensor::rand(&ts![100, 100, 100]);
    let t2: Tensor<f64> = Tensor::rand(&ts![100, 2000, 100]);

    c.bench_function("concat_mt", |b| {
        b.iter(|| {
            t1.concat_mt(black_box(&t2), black_box(1)).unwrap();
        })
    });
}

pub fn bench_transpose(c: &mut Criterion) {
    let t1: Tensor<f64> = Tensor::rand(&ts![100, 100, 100]);

    c.bench_function("transpose", |b| {
        b.iter(|| {
            t1.clone().transpose(&Transpose::new(&vec![2, 0, 1]).unwrap()).unwrap();
        })
    });
}

pub fn bench_contract_mul_mt(c: &mut Criterion) {
    let t1: Tensor<f64> = Tensor::rand(&ts![100, 100]);
    let t2: Tensor<f64> = Tensor::rand(&ts![100, 100]);

    c.bench_function("contract_mul_mt", |b| {
        b.iter(|| {
            t1.contract_mul_mt(black_box(&t2)).unwrap();
        })
    });
}


criterion_group!(benches, bench_concat_mt, bench_transpose, bench_contract_mul_mt);
criterion_main!(benches);