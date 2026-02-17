use std::ops::Add;
use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::definitions::tensor::Tensor;

impl<T: Add<Output = T> + Clone> Tensor<T> {
    /// Compute the sum of a `Tensor`
    pub fn sum(&self) -> T {
        self.iter().cloned().reduce(|x, y| x + y).unwrap()
    }
}

impl<T: Add<Output = T> + Clone> Matrix<T> {
    /// Compute the sum of a `Matrix`
    pub fn sum(&self) -> T {
        self.tensor.sum()
    }

    /// Computes the trace of a matrix
    pub fn trace(self: &Matrix<T>) -> Result<T, TensorErrors> {
        if !self.is_square() {
            return Err(TensorErrors::NonSquareMatrix);
        }

        let mut sum = self.elements.first().unwrap().clone();

        for i in 1..self.shape.0.iter().min().unwrap().clone() {
            sum = sum.add(self[&[i, i]].clone());
        }

        Ok(sum)
    }
}
