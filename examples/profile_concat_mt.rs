// examples/profile_concat_mt.rs
use std::hint::black_box;
use tensor_math::definitions::shape::Shape;
use tensor_math::definitions::tensor::Tensor;
use tensor_math::shape;

pub fn main() {
    let a = Tensor::from_value(&shape![64, 256, 256], 1.0);
    let b = Tensor::from_value(&shape![64, 256, 256], 2.0);

    // Warm up (page faults, allocator caching, thread pool spin-up) so the
    // measured loop isn't dominated by one-time setup costs.
    for _ in 0..5 {
        let r = a.concat_mt(&b, 2).expect("concat must succeed");
        black_box(&r);
    }

    let iterations = 1000;
    for _ in 0..iterations {
        let r = a.concat_mt(&b, 2).expect("concat must succeed");
        black_box(&r);
    }
}