use std::hint::black_box;
use tensor_math::definitions::matrix::Matrix;

pub fn main() {
    let a = Matrix::<f64>::rand(128, 128);

    // Warm up (page faults, allocator caching, thread pool spin-up) so the
    // measured loop isn't dominated by one-time setup costs.
    for _ in 0..5 {
        let r = a.upper_hessenberg().expect("hessenberg must succeed");
        black_box(&r);
    }

    let iterations = 20;
    for _ in 0..iterations {
        let r = a.upper_hessenberg().expect("hessenberg must succeed");
        black_box(&r);
    }
}