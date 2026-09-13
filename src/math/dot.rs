use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::definitions::tensor::Tensor;
use crate::utilities::internal_functions::{dot_vectors, dot_vectors_mt};
use num::Zero;
use std::ops::{Add, AddAssign, Mul};

impl<T> Tensor<T> {
    /// Computes the dot product of two tensors, i.e. the element-wise product, then the sum of the result.
    /// This fails if the tensors do not have the same shape.
    pub fn dot(&self, other: &Tensor<T>) -> Result<T, TensorErrors>
    where
        T: AddAssign + Clone + Mul<Output = T> + Zero,
    {
        if self.shape != other.shape {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape.clone(),
                shape_2: other.shape.clone(),
                op: "dot",
            });
        }

        Ok(dot_vectors(self.elements(), &other.elements()))
    }

    /// Computes the dot product of two tensors, i.e. the element-wise product, then the sum of the result, without validity checking.
    pub(crate) unsafe fn dot_unchecked(&self, other: &Tensor<T>) -> T
    where
        T: AddAssign + Clone + Mul<Output = T> + Zero,
    {
        dot_vectors(self.elements(), &other.elements())
    }

    /// Computes the dot product of two tensors, i.e. the element-wise product, then the sum of the result.
    /// This fails if the tensors do not have the same shape.
    pub fn dot_mt(&self, other: &Tensor<T>) -> Result<T, TensorErrors>
    where
        T: AddAssign + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
    {
        if self.shape != other.shape {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape.clone(),
                shape_2: other.shape.clone(),
                op: "dot",
            });
        }

        Ok(dot_vectors_mt(self.elements(), other.elements()))
    }

    /// Computes the dot product of two tensors, i.e. the element-wise product, then the sum of the result, without validity checking.
    pub(crate) unsafe fn dot_unchecked_mt(&self, other: &Tensor<T>) -> T
    where
        T: AddAssign + Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
    {
        dot_vectors_mt(self.elements(), other.elements())
    }
}

impl<T> Matrix<T> {
    /// Computes the dot product of the two matrices, i.e. the elementwise product, then the sum of the result.
    /// This fails if the matrices do not have the same shape.
    pub fn dot(&self, other: &Matrix<T>) -> Result<T, TensorErrors>
    where
        T: Clone + Add<Output = T> + Mul<Output = T> + Zero + AddAssign,
    {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: other.shape(),
                op: "dot",
            });
        }

        Ok(dot_vectors(self.elements(), other.elements()))
    }

    /// Computes the dot product of the two matrices, i.e. the elementwise product, then the sum of the result, without validity checking.
    pub(crate) unsafe fn dot_unchecked(&self, other: &Matrix<T>) -> T
    where
        T: Clone + Add<Output = T> + Mul<Output = T> + Zero + AddAssign,
    {
        dot_vectors(self.elements(), other.elements())
    }

    /// Computes the dot product of two matrices, i.e. the element-wise product, then the sum of the result.
    /// This fails if the matrices do not have the same shape.
    pub fn dot_mt(&self, other: &Matrix<T>) -> Result<T, TensorErrors>
    where
        T: AddAssign + Clone + Add<Output = T> + Mul<Output = T> + Send + Sync + Zero,
    {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: other.shape(),
                op: "dot_mt",
            });
        }

        Ok(dot_vectors_mt(self.elements(), other.elements()))
    }

    /// Computes the dot product of two matrices, i.e. the element-wise product, then the sum of the result, without validity checking.
    pub(crate) unsafe fn dot_unchecked_mt(&self, other: &Matrix<T>) -> T
    where
        T: AddAssign + Clone + Add<Output = T> + Mul<Output = T> + Send + Sync + Zero,
    {
        dot_vectors_mt(self.elements(), other.elements())
    }
}
