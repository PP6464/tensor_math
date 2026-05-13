use std::ops::Add;
use num::Zero;
use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::definitions::tensor::Tensor;

impl<T: Add<Output = T> + Clone + Zero> Tensor<T> {
    /// Compute the sum of a `Tensor`
    pub fn sum(&self) -> T {
        self.iter().cloned().fold(T::zero(), |acc, x| acc + x)
    }
}

impl<T: Add<Output = T> + Clone + Zero> Matrix<T> {
    /// Compute the sum of a `Matrix`
    pub fn sum(&self) -> T {
        self.tensor.sum()
    }

    /// Computes the trace of a matrix
    pub fn trace(self: &Matrix<T>) -> Result<T, TensorErrors> {
        if !self.is_square() {
            return Err(TensorErrors::NonSquareMatrix);
        }

        let mut sum = T::zero();

        for i in 1..self.shape.0.iter().min().unwrap().clone() {
            sum = sum.add(self[&[i, i]].clone());
        }

        Ok(sum)
    }
}
