use rayon::iter::ParallelIterator;
use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::definitions::tensor::Tensor;
use num::Zero;
use std::ops::Add;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator};

impl<T: Add<Output = T> + Clone + Zero> Tensor<T> {
    /// Compute the sum of a tensor
    pub fn sum(&self) -> T {
        self.iter().cloned().fold(T::zero(), |acc, x| acc + x)
    }
}

impl<T: Add<Output = T> + Clone + Zero + Send + Sync> Tensor<T> {
    /// Compute the sum of a tensor
    pub fn sum_mt(&self) -> T {
        self.par_iter().cloned().reduce(|| T::zero(), |acc, x| acc + x)
    }
}

impl<T: Add<Output = T> + Clone + Zero> Matrix<T> {
    /// Compute the sum of a matrix
    pub fn sum(&self) -> T {
        self.tensor.sum()
    }

    /// Computes the trace of a matrix
    pub fn trace(self: &Matrix<T>) -> Result<T, TensorErrors> {
        if !self.is_square() {
            return Err(TensorErrors::NonSquareMatrix);
        }

        let mut sum = T::zero();

        for i in 0..self.shape.0.iter().min().unwrap().clone() {
            sum = sum.add(self[&[i, i]].clone());
        }

        Ok(sum)
    }
}

impl<T: Add<Output = T> + Clone + Zero + Send + Sync> Matrix<T> {
    /// Compute the sum of a matrix
    pub fn sum_mt(&self) -> T {
        self.tensor.sum_mt()
    }

    /// Computes the trace of a matrix
    pub fn trace_mt(self: &Matrix<T>) -> Result<T, TensorErrors> {
        if !self.is_square() {
            return Err(TensorErrors::NonSquareMatrix);
        }

        let sum = (0..self.rows)
            .into_par_iter()
            .map(|i| self[&[i, i]].clone())
            .reduce(|| T::zero(), |acc, x| acc + x);

        Ok(sum)
    }
}
