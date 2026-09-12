use std::ops::{Add, AddAssign};
use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::definitions::tensor::Tensor;
use num::Zero;
use rayon::iter::ParallelIterator;
use rayon::prelude::ParallelSlice;

fn sum_slice<T: AddAssign + Add<Output = T> + Clone + Zero>(slice: &[T]) -> T {
    // Use four accumulators to help auto vectorisation
    let mut acc = [T::zero(), T::zero(), T::zero(), T::zero()];

    unsafe {
        for chunk in slice.chunks_exact(4) {
            *acc.get_unchecked_mut(0) += chunk.get_unchecked(0).clone();
            *acc.get_unchecked_mut(1) += chunk.get_unchecked(1).clone();
            *acc.get_unchecked_mut(2) += chunk.get_unchecked(2).clone();
            *acc.get_unchecked_mut(3) += chunk.get_unchecked(3).clone();
        }

        let mut sum = acc.into_iter().reduce(T::add).map_or_else(T::zero, std::convert::identity);

        let remainder = slice.len() % 4;

        for i in 0..remainder {
            sum = sum + slice.get_unchecked(slice.len() - remainder + i).clone();
        }

        sum
    }
}

fn sum_slice_mt<T: AddAssign + Add<Output = T> + Clone + Zero + Send + Sync>(slice: &[T]) -> T {
    slice
        .par_chunks(4096)
        .map(|chunk| sum_slice(chunk))
        .reduce(|| T::zero(), |acc, x| acc + x)
}

impl<T: AddAssign + Add<Output = T> + Clone + Zero> Tensor<T> {
    /// Compute the sum of a tensor
    pub fn sum(&self) -> T {
        sum_slice(&self.elements)
    }
}

impl<T: AddAssign + Add<Output = T> + Clone + Zero + Send + Sync> Tensor<T> {
    /// Compute the sum of a tensor
    pub fn sum_mt(&self) -> T {
        sum_slice_mt(&self.elements)
    }
}

impl<T: AddAssign + Add<Output = T> + Clone + Zero> Matrix<T> {
    /// Compute the sum of a matrix
    pub fn sum(&self) -> T {
        sum_slice(&self.elements)
    }

    /// Computes the trace of a matrix
    pub fn trace(self: &Matrix<T>) -> Result<T, TensorErrors> {
        if !self.is_square() {
            return Err(TensorErrors::NonSquareMatrix);
        }

        let mut diag_slice = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            diag_slice.push(self[&[i, i]].clone());
        }

        Ok(sum_slice(&diag_slice))
    }
}

impl<T: AddAssign + Add<Output = T> + Clone + Zero + Send + Sync> Matrix<T> {
    /// Compute the sum of a matrix
    pub fn sum_mt(&self) -> T {
        sum_slice_mt(&self.elements)
    }

    /// Computes the trace of a matrix
    pub fn trace_mt(self: &Matrix<T>) -> Result<T, TensorErrors> {
        if !self.is_square() {
            return Err(TensorErrors::NonSquareMatrix);
        }

        let mut diag_slice = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            diag_slice.push(self[&[i, i]].clone());
        }

        Ok(sum_slice_mt(&diag_slice))
    }
}
