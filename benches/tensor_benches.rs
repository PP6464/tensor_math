use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use tensor_math::tensor::tensor::{Tensor, Shape};
use tensor_math::ts;

pub fn bench_concat(c: &mut Criterion) {
    let t1: Tensor<f64> = Tensor::rand(&ts![100, 100, 100]);
    let t2: Tensor<f64> = Tensor::rand(&ts![100, 2000, 100]);

    c.bench_function("concat", |b| {
        b.iter(|| {
            t1.concat(black_box(&t2), black_box(1)).unwrap();
        })
    });
}


criterion_group!(benches, bench_concat);
criterion_main!(benches);