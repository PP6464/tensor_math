use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator, IndexedParallelIterator};
use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::definitions::shape::Shape;
use crate::shape;
use crate::utilities::internal_functions::dot_vectors;
use num::Zero;
use std::ops::{Add, AddAssign, Mul};
use rayon::slice::ParallelSlice;

impl<T> Matrix<T> {
    /// Computes the product of this matrix with a vector.
    /// This fails if the length of the vector does not match `self.cols`.
    pub fn mat_vec_mul(self, v: &[T]) -> Result<Vec<T>, TensorErrors>
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero,
    {
        if v.len() != self.cols {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: shape![v.len()],
                op: "mat_vec_mul",
            });
        }

        let mut res: Vec<T> = Vec::with_capacity(v.len());

        self.chunks(self.cols)
            .for_each(|chunk| res.push(dot_vectors(chunk, v)));

        Ok(res)
    }

    /// Computes the product of this matrix with a vector without bounds checking.
    pub(crate) unsafe fn mat_vec_mul_unchecked(self, v: &[T]) -> Vec<T>
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero,
    {
        let mut res: Vec<T> = Vec::with_capacity(v.len());

        self.chunks(self.cols)
            .for_each(|chunk| res.push(dot_vectors(chunk, v)));

        res
    }

    /// Computes the product of this matrix with a vector.
    /// This fails if the length of the vector does not match `self.cols`.
    pub fn mat_vec_mul_mt(self, v: &[T]) -> Result<Vec<T>, TensorErrors>
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero + Send + Sync,
    {
        if v.len() != self.cols {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: shape![v.len()],
                op: "mat_vec_mul",
            });
        }

        let mut res: Vec<T> = Vec::with_capacity(v.len());
        let buf = res.spare_capacity_mut();

        self.par_chunks(self.cols)
            .zip(buf.par_iter_mut())
            .for_each(|(chunk, out)| {
                out.write(dot_vectors(chunk, v));
            });

        Ok(res)
    }

    /// Computes the product of this matrix with a vector without bounds checking.
    pub(crate) unsafe fn mat_vec_mul_mt_unchecked(self, v: &[T]) -> Vec<T>
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero + Send + Sync,
    {
        let mut res: Vec<T> = Vec::with_capacity(v.len());

        self.chunks(self.cols)
            .for_each(|chunk| res.push(dot_vectors(chunk, v)));

        res
    }
}
