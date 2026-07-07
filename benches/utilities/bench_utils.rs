//! Helpers shared across the utility-level benchmarks.
//!
//! These are kept in their own module so that each `*_benches` file (e.g.
//! `concat`, `transpose`) can construct inputs and drain results in the same
//! way without duplicating boilerplate. The functions are deliberately small
//! and `#[inline]`-friendly — they exist for convenience, not for performance.

use tensor_math::definitions::matrix::Matrix;
use tensor_math::definitions::shape::Shape;
use tensor_math::definitions::tensor::Tensor;

/// Drain a tensor into a single `f64` so the result is observably used and the
/// optimiser cannot elide the work. The caller should still `black_box` the
/// return value.
#[inline]
pub fn drain_f64(t: &Tensor<f64>) -> f64 {
    t.elements().iter().copied().sum()
}

/// Touch every element of a matrix and return a checksum, so the optimiser
/// cannot elide the work. The result is fed to `black_box` by the caller.
#[inline]
pub fn drain_mat_f64(m: &Matrix<f64>) -> f64 {
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

/// Build a 1-D tensor of `n` elements filled with `value`.
pub fn tensor_1d(n: usize, value: f64) -> Tensor<f64> {
    let shape = Shape::new(vec![n]);
    let elements = vec![value; n];
    Tensor::new(&shape, elements).expect("1-D tensor construction cannot fail")
}

/// Build an `n`-D tensor with the given shape filled with `value`.
pub fn tensor_from_shape(shape_dims: &[usize], value: f64) -> Tensor<f64> {
    let total: usize = shape_dims.iter().product();
    let shape = Shape::new(shape_dims.to_vec());
    let elements = vec![value; total];
    Tensor::new(&shape, elements).expect("tensor construction cannot fail")
}

/// Build an `rows x cols` matrix filled with `value`.
pub fn matrix(rows: usize, cols: usize, value: f64) -> Matrix<f64> {
    let elements = vec![value; rows * cols];
    Matrix::new(rows, cols, elements).expect("matrix construction cannot fail")
}

/// Drain a sequential `enumerated_iter` into a checksum.
#[inline]
pub fn drain_iter<I>(it: I) -> u64
where
    I: Iterator<Item = (Vec<usize>, f64)>,
{
    let mut acc: u64 = 0;
    for (_, v) in it {
        acc = acc.wrapping_add(v.to_bits());
    }
    acc
}

/// Drain a sequential `enumerated_iter_mut` into a checksum.
#[inline]
pub fn drain_iter_mut<'a, I>(it: I) -> u64
where
    I: Iterator<Item = (Vec<usize>, &'a mut f64)>,
{
    let mut acc: u64 = 0;
    for (_, v) in it {
        acc = acc.wrapping_add((*v).to_bits());
    }
    acc
}
