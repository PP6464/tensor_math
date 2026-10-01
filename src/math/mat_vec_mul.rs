use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator, IndexedParallelIterator};
use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::definitions::shape::Shape;
use crate::shape;
use crate::utilities::internal_functions::dot_vectors;
use num::Zero;
use std::ops::{Add, AddAssign, Mul};

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

        self.iter_rows().for_each(|chunk| res.push(dot_vectors(chunk, v)));

        Ok(res)
    }

    /// Computes the product of this matrix with a vector without bounds checking.
    pub(crate) unsafe fn mat_vec_mul_unchecked(self, v: &[T]) -> Vec<T>
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero,
    {
        let mut res: Vec<T> = Vec::with_capacity(v.len());

        self.iter_rows().for_each(|chunk| res.push(dot_vectors(chunk, v)));

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

        self.par_iter_rows()
            .zip(buf.par_iter_mut())
            .for_each(|(chunk, out)| {
                out.write(dot_vectors(chunk, v));
            });
        
        unsafe {
            res.set_len(self.rows);
        }

        Ok(res)
    }

    /// Computes the product of this matrix with a vector without bounds checking.
    pub(crate) unsafe fn mat_vec_mul_unchecked_mt(self, v: &[T]) -> Vec<T>
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero + Send + Sync,
    {
        let mut res: Vec<T> = Vec::with_capacity(v.len());
        let buf = res.spare_capacity_mut();

        self.par_iter_rows()
            .zip(buf.par_iter_mut())
            .for_each(|(chunk, out)| {
                out.write(dot_vectors(chunk, v));
            });
        
        unsafe {
            res.set_len(self.rows);
        }

        res
    }

    /// Computes the product of a vector with this matrix.
    /// This fails if the length of the vector does not match `self.rows`.
    pub fn vec_mat_mul(self, v: &[T]) -> Result<Vec<T>, TensorErrors>
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero,
    {
        if v.len() != self.rows {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: shape![v.len()],
                op: "vec_mat_mul",
            });
        }

        let mut res: Vec<T> = Vec::with_capacity(v.len());

        self.transpose().iter_rows().for_each(|chunk| res.push(dot_vectors(v, chunk)));

        Ok(res)
    }

    /// Computes the product of a vector with this matrix without bounds checking.
    pub(crate) unsafe fn vec_mat_mul_unchecked(self, v: &[T]) -> Vec<T>
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero,
    {
        let mut res: Vec<T> = Vec::with_capacity(v.len());

        self.transpose().iter_rows().for_each(|chunk| res.push(dot_vectors(v, chunk)));

        res
    }

    /// Computes the product of a vector with this matrix.
    /// This fails if the length of the vector does not match `self.rows`.
    pub fn vec_mat_mul_mt(self, v: &[T]) -> Result<Vec<T>, TensorErrors>
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero + Send + Sync,
    {
        if v.len() != self.rows {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: shape![v.len()],
                op: "mat_vec_mul",
            });
        }

        let mut res: Vec<T> = Vec::with_capacity(v.len());
        let buf = res.spare_capacity_mut();
        
        let cols = self.cols;

        self
            .transpose_mt()
            .par_iter_rows()
            .zip(buf.par_iter_mut())
            .for_each(|(chunk, out)| {
                out.write(dot_vectors(chunk, v));
            });

        unsafe {
            res.set_len(cols);
        }

        Ok(res)
    }

    /// Computes the product of a vector with this matrix without bounds checking.
    pub(crate) unsafe fn vec_mat_mul_unchecked_mt(self, v: &[T]) -> Vec<T>
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero + Send + Sync,
    {
        let mut res: Vec<T> = Vec::with_capacity(v.len());
        let buf = res.spare_capacity_mut();
        
        let cols = self.cols;

        self
            .transpose_mt()
            .par_iter_rows()
            .zip(buf.par_iter_mut())
            .for_each(|(chunk, out)| {
                out.write(dot_vectors(chunk, v));
            });

        unsafe {
            res.set_len(cols);
        }

        res
    }
}
