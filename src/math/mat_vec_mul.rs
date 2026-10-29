use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::definitions::matrix_slice::MatrixSlice;
use crate::shape;
use crate::utilities::internal_functions::dot_vectors;
use num::Zero;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
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

        Ok(unsafe { self.mat_vec_mul_unchecked(v) })
    }

    /// Computes the product of this matrix with a vector without bounds checking.
    pub(crate) unsafe fn mat_vec_mul_unchecked(self, v: &[T]) -> Vec<T>
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero,
    {
        let mut res: Vec<T> = Vec::with_capacity(self.rows);

        self.iter_rows()
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

        Ok(unsafe { self.mat_vec_mul_unchecked_mt(v) })
    }

    /// Computes the product of this matrix with a vector without bounds checking.
    pub(crate) unsafe fn mat_vec_mul_unchecked_mt(self, v: &[T]) -> Vec<T>
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero + Send + Sync,
    {
        let mut res: Vec<T> = Vec::with_capacity(self.rows);
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

        Ok(unsafe { self.vec_mat_mul_unchecked(v) })
    }

    /// Computes the product of a vector with this matrix without bounds checking.
    pub(crate) unsafe fn vec_mat_mul_unchecked(self, v: &[T]) -> Vec<T>
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero,
    {
        let mut res: Vec<T> = Vec::with_capacity(self.cols);

        self.transpose()
            .iter_rows()
            .for_each(|chunk| res.push(dot_vectors(v, chunk)));

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

        Ok(unsafe { self.vec_mat_mul_unchecked_mt(v) })
    }

    /// Computes the product of a vector with this matrix without bounds checking.
    pub(crate) unsafe fn vec_mat_mul_unchecked_mt(self, v: &[T]) -> Vec<T>
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero + Send + Sync,
    {
        let mut res: Vec<T> = Vec::with_capacity(self.cols);
        let buf = res.spare_capacity_mut();

        let cols = self.cols;

        self.transpose_mt()
            .par_iter_rows()
            .zip(buf.par_iter_mut())
            .for_each(|(chunk, out)| {
                out.write(dot_vectors(v, chunk));
            });

        unsafe {
            res.set_len(cols);
        }

        res
    }

    /// Computes the product of this matrix with a vector, and stores the result into `out`.
    /// This fails if the length of the vector does not match `self.cols` or if the length of `out.len() != self.rows`.
    pub fn mat_vec_mul_into_slice(self, v: &[T], out: &mut [T]) -> Result<(), TensorErrors>
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero,
    {
        if v.len() != self.cols {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: shape![v.len()],
                op: "mat_vec_mul_into_slice",
            });
        }

        if out.len() != self.rows {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: shape![self.rows],
                shape_2: shape![out.len()],
                op: "mat_vec_mul_into_slice",
            });
        }

        unsafe { self.mat_vec_mul_unchecked_into_slice(v, out) };
        Ok(())
    }

    /// Computes the product of this matrix with a vector and stores into `out` without bounds checking.
    pub(crate) unsafe fn mat_vec_mul_unchecked_into_slice(self, v: &[T], out: &mut [T])
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero,
    {
        out.iter_mut()
            .zip(self.iter_rows())
            .for_each(|(out_elem, chunk)| *out_elem = dot_vectors(chunk, v));
    }

    /// Computes the product of this matrix with a vector and stores the result into `out`.
    /// This fails if the length of the vector does not match `self.cols` or if `out.len() != self.rows`.
    pub fn mat_vec_mul_mt_into_slice(self, v: &[T], out: &mut [T]) -> Result<(), TensorErrors>
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero + Send + Sync,
    {
        if v.len() != self.cols {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: shape![v.len()],
                op: "mat_vec_mul_into_slice",
            });
        }

        if out.len() != self.rows {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: shape![self.rows],
                shape_2: shape![out.len()],
                op: "mat_vec_mul_into_slice",
            });
        }

        unsafe { self.mat_vec_mul_unchecked_mt_into_slice(v, out) };
        Ok(())
    }

    /// Computes the product of this matrix with a vector and stores the result into `out` without bounds checking.
    pub(crate) unsafe fn mat_vec_mul_unchecked_mt_into_slice(self, v: &[T], out: &mut [T])
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero + Send + Sync,
    {
        self.par_iter_rows()
            .zip(out.par_iter_mut())
            .for_each(|(chunk, out)| {
                *out = dot_vectors(chunk, v);
            });
    }

    /// Computes the product of a vector with this matrix and stores the result into `out`.
    /// This fails if the length of the vector does not match `self.rows` or if `out.len() != self.cols`.
    pub fn vec_mat_mul_into_slice(self, v: &[T], out: &mut [T]) -> Result<(), TensorErrors>
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero,
    {
        if v.len() != self.rows {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: shape![v.len()],
                op: "vec_mat_mul_into_slice",
            });
        }

        if out.len() != self.cols {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: shape![self.cols],
                shape_2: shape![out.len()],
                op: "vec_mat_mul_into_slice",
            });
        }

        unsafe { self.vec_mat_mul_unchecked_into_slice(v, out) };
        Ok(())
    }

    /// Computes the product of a vector with this matrix and stores it into `out` without bounds checking.
    pub(crate) unsafe fn vec_mat_mul_unchecked_into_slice(self, v: &[T], out: &mut [T])
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero,
    {
        out.iter_mut()
            .zip(self.transpose().iter_rows())
            .for_each(|(out_elem, chunk)| *out_elem = dot_vectors(v, chunk));
    }

    /// Computes the product of a vector with this matrix and stores it into `out`.
    /// This fails if the length of the vector does not match `self.rows` or if `out.len() != self.cols`.
    pub fn vec_mat_mul_mt_into_slice(self, v: &[T], out: &mut [T]) -> Result<(), TensorErrors>
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero + Send + Sync,
    {
        if v.len() != self.rows {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: shape![v.len()],
                op: "vec_mat_mul_into_slice",
            });
        }

        if out.len() != self.cols {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: shape![self.cols],
                shape_2: shape![out.len()],
                op: "vec_mat_mul_into_slice",
            });
        }

        unsafe { self.vec_mat_mul_unchecked_mt_into_slice(v, out) };
        Ok(())
    }

    /// Computes the product of a vector with this matrix and stores it into `out` without bounds checking.
    pub(crate) unsafe fn vec_mat_mul_unchecked_mt_into_slice(self, v: &[T], out: &mut [T])
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero + Send + Sync,
    {
        out.par_iter_mut()
            .zip(self.transpose().par_iter_rows())
            .for_each(|(out_elem, chunk)| *out_elem = dot_vectors(v, chunk));
    }
}

impl<T> MatrixSlice<'_, T> {
    /// Computes the product of this matrix slice with a vector.
    /// This fails if the length of the vector does not match `self.cols()`.
    pub fn mat_vec_mul(&self, v: &[T]) -> Result<Vec<T>, TensorErrors>
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero,
    {
        if v.len() != self.cols() {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: shape![v.len()],
                op: "mat_vec_mul",
            });
        }

        Ok(unsafe { self.mat_vec_mul_unchecked(v) })
    }

    /// Computes the product of this matrix slice with a vector without bounds checking.
    pub(crate) unsafe fn mat_vec_mul_unchecked(&self, v: &[T]) -> Vec<T>
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero,
    {
        let mut res: Vec<T> = Vec::with_capacity(self.rows());

        self.iter_rows()
            .for_each(|chunk| res.push(dot_vectors(chunk, v)));

        res
    }

    /// Computes the product of this matrix slice with a vector.
    /// This fails if the length of the vector does not match `self.cols()`.
    pub fn mat_vec_mul_mt(&self, v: &[T]) -> Result<Vec<T>, TensorErrors>
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero + Send + Sync,
    {
        if v.len() != self.cols() {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: shape![v.len()],
                op: "mat_vec_mul",
            });
        }

        Ok(unsafe { self.mat_vec_mul_unchecked_mt(v) })
    }

    /// Computes the product of this matrix slice with a vector without bounds checking.
    pub(crate) unsafe fn mat_vec_mul_unchecked_mt(&self, v: &[T]) -> Vec<T>
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero + Send + Sync,
    {
        let mut res: Vec<T> = Vec::with_capacity(self.rows());
        let buf = res.spare_capacity_mut();

        self.par_iter_rows()
            .zip(buf.par_iter_mut())
            .for_each(|(chunk, out)| {
                out.write(dot_vectors(chunk, v));
            });

        unsafe {
            res.set_len(self.rows());
        }

        res
    }

    /// Computes the product of a vector with this matrix slice.
    /// This fails if the length of the vector does not match `self.rows()`.
    pub fn vec_mat_mul(&self, v: &[T]) -> Result<Vec<T>, TensorErrors>
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero,
    {
        if v.len() != self.rows() {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: shape![v.len()],
                op: "vec_mat_mul",
            });
        }

        Ok(unsafe { self.vec_mat_mul_unchecked(v) })
    }

    /// Computes the product of a vector with this matrix slice without bounds checking.
    pub(crate) unsafe fn vec_mat_mul_unchecked(&self, v: &[T]) -> Vec<T>
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero,
    {
        let mut res: Vec<T> = Vec::with_capacity(self.cols());

        self.clone_into_matrix()
            .transpose()
            .iter_rows()
            .for_each(|chunk| res.push(dot_vectors(v, chunk)));

        res
    }

    /// Computes the product of a vector with this matrix slice.
    /// This fails if the length of the vector does not match `self.rows()`.
    pub fn vec_mat_mul_mt(&self, v: &[T]) -> Result<Vec<T>, TensorErrors>
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero + Send + Sync,
    {
        if v.len() != self.rows() {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: shape![v.len()],
                op: "mat_vec_mul",
            });
        }

        Ok(unsafe { self.vec_mat_mul_unchecked_mt(v) })
    }

    /// Computes the product of a vector with this matrix slice without bounds checking.
    pub(crate) unsafe fn vec_mat_mul_unchecked_mt(&self, v: &[T]) -> Vec<T>
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero + Send + Sync,
    {
        let mut res: Vec<T> = Vec::with_capacity(self.cols());
        let buf = res.spare_capacity_mut();

        let cols = self.cols();

        self.clone_into_matrix()
            .transpose_mt()
            .par_iter_rows()
            .zip(buf.par_iter_mut())
            .for_each(|(chunk, out)| {
                out.write(dot_vectors(v, chunk));
            });

        unsafe {
            res.set_len(cols);
        }

        res
    }

    /// Computes the product of this matrix slice with a vector, and stores the result into `out`.
    /// This fails if the length of the vector does not match `self.cols()` or if the length of `out.len() != self.rows()`.
    pub fn mat_vec_mul_into_slice(&self, v: &[T], out: &mut [T]) -> Result<(), TensorErrors>
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero,
    {
        if v.len() != self.cols() {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: shape![v.len()],
                op: "mat_vec_mul_into_slice",
            });
        }

        if out.len() != self.rows() {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: shape![self.rows()],
                shape_2: shape![out.len()],
                op: "mat_vec_mul_into_slice",
            });
        }

        unsafe { self.mat_vec_mul_unchecked_into_slice(v, out) };
        Ok(())
    }

    /// Computes the product of this matrix slice with a vector and stores into `out` without bounds checking.
    pub(crate) unsafe fn mat_vec_mul_unchecked_into_slice(&self, v: &[T], out: &mut [T])
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero,
    {
        out.iter_mut()
            .zip(self.iter_rows())
            .for_each(|(out_elem, chunk)| *out_elem = dot_vectors(chunk, v));
    }

    /// Computes the product of this matrix slice with a vector and stores the result into `out`.
    /// This fails if the length of the vector does not match `self.cols()` or if `out.len() != self.rows()`.
    pub fn mat_vec_mul_mt_into_slice(&self, v: &[T], out: &mut [T]) -> Result<(), TensorErrors>
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero + Send + Sync,
    {
        if v.len() != self.cols() {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: shape![v.len()],
                op: "mat_vec_mul_into_slice",
            });
        }

        if out.len() != self.rows() {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: shape![self.rows()],
                shape_2: shape![out.len()],
                op: "mat_vec_mul_into_slice",
            });
        }

        unsafe { self.mat_vec_mul_unchecked_mt_into_slice(v, out) };
        Ok(())
    }

    /// Computes the product of this matrix slice with a vector and stores the result into `out` without bounds checking.
    pub(crate) unsafe fn mat_vec_mul_unchecked_mt_into_slice(&self, v: &[T], out: &mut [T])
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero + Send + Sync,
    {
        self.par_iter_rows()
            .zip(out.par_iter_mut())
            .for_each(|(chunk, out)| {
                *out = dot_vectors(chunk, v);
            });
    }

    /// Computes the product of a vector with this matrix slice and stores the result into `out`.
    /// This fails if the length of the vector does not match `self.rows()` or if `out.len() != self.cols()`.
    pub fn vec_mat_mul_into_slice(&self, v: &[T], out: &mut [T]) -> Result<(), TensorErrors>
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero,
    {
        if v.len() != self.rows() {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: shape![v.len()],
                op: "vec_mat_mul_into_slice",
            });
        }

        if out.len() != self.cols() {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: shape![self.cols()],
                shape_2: shape![out.len()],
                op: "vec_mat_mul_into_slice",
            });
        }

        unsafe { self.vec_mat_mul_unchecked_into_slice(v, out) };
        Ok(())
    }

    /// Computes the product of a vector with this matrix slice and stores it into `out` without bounds checking.
    pub(crate) unsafe fn vec_mat_mul_unchecked_into_slice(&self, v: &[T], out: &mut [T])
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero,
    {
        out.iter_mut()
            .zip(self.clone_into_matrix().transpose().iter_rows())
            .for_each(|(out_elem, chunk)| *out_elem = dot_vectors(v, chunk));
    }

    /// Computes the product of a vector with this matrix slice and stores it into `out`.
    /// This fails if the length of the vector does not match `self.rows()` or if `out.len() != self.cols()`.
    pub fn vec_mat_mul_mt_into_slice(&self, v: &[T], out: &mut [T]) -> Result<(), TensorErrors>
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero + Send + Sync,
    {
        if v.len() != self.rows() {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: shape![v.len()],
                op: "vec_mat_mul_into_slice",
            });
        }

        if out.len() != self.cols() {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: shape![self.cols()],
                shape_2: shape![out.len()],
                op: "vec_mat_mul_into_slice",
            });
        }

        unsafe { self.vec_mat_mul_unchecked_mt_into_slice(v, out) };
        Ok(())
    }

    /// Computes the product of a vector with this matrix slice and stores it into `out` without bounds checking.
    pub(crate) unsafe fn vec_mat_mul_unchecked_mt_into_slice(&self, v: &[T], out: &mut [T])
    where
        T: Clone + Mul<Output = T> + AddAssign + Add<Output = T> + Zero + Send + Sync,
    {
        out.par_iter_mut()
            .zip(self.clone_into_matrix().transpose().par_iter_rows())
            .for_each(|(out_elem, chunk)| *out_elem = dot_vectors(v, chunk));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix() -> Matrix<i32> {
        Matrix::new(2, 3, vec![1, 2, 3, 4, 5, 6]).unwrap()
    }

    #[test]
    fn matrix_vector_products_and_output_slices() {
        let m = matrix();
        assert_eq!(m.clone().mat_vec_mul(&[2, 3, 4]), Ok(vec![20, 47]));
        assert_eq!(m.clone().mat_vec_mul_mt(&[2, 3, 4]), Ok(vec![20, 47]));
        assert_eq!(m.clone().vec_mat_mul(&[2, 3]), Ok(vec![14, 19, 24]));
        assert_eq!(m.clone().vec_mat_mul_mt(&[2, 3]), Ok(vec![14, 19, 24]));

        let mut out = [0, 0];
        m.clone()
            .mat_vec_mul_into_slice(&[2, 3, 4], &mut out)
            .unwrap();
        assert_eq!(out, [20, 47]);
        m.clone()
            .mat_vec_mul_mt_into_slice(&[2, 3, 4], &mut out)
            .unwrap();
        assert_eq!(out, [20, 47]);

        let mut out = [0, 0, 0];
        m.clone().vec_mat_mul_into_slice(&[2, 3], &mut out).unwrap();
        assert_eq!(out, [14, 19, 24]);
        m.clone()
            .vec_mat_mul_mt_into_slice(&[2, 3], &mut out)
            .unwrap();
        assert_eq!(out, [14, 19, 24]);

        let empty_rows = Matrix::<i32>::new(0, 3, vec![]).unwrap();
        assert_eq!(empty_rows.clone().mat_vec_mul(&[1, 2, 3]), Ok(vec![]));
        assert_eq!(empty_rows.clone().mat_vec_mul_mt(&[1, 2, 3]), Ok(vec![]));
        empty_rows
            .mat_vec_mul_into_slice(&[1, 2, 3], &mut [])
            .unwrap();

        assert!(matches!(
            m.clone().mat_vec_mul(&[1, 2]),
            Err(TensorErrors::IncompatibleShapes { .. })
        ));
        assert!(matches!(
            m.vec_mat_mul_mt(&[1]),
            Err(TensorErrors::IncompatibleShapes { .. })
        ));
    }

    #[test]
    fn matrix_slice_vector_products_and_output_slices() {
        let m = matrix();
        let slice = m.slice(0..2, 1..3).unwrap();
        assert_eq!(slice.mat_vec_mul(&[2, 3]), Ok(vec![13, 28]));
        assert_eq!(slice.mat_vec_mul_mt(&[2, 3]), Ok(vec![13, 28]));
        assert_eq!(slice.vec_mat_mul(&[2, 3]), Ok(vec![19, 24]));
        assert_eq!(slice.vec_mat_mul_mt(&[2, 3]), Ok(vec![19, 24]));

        let mut out = [0, 0];
        slice.mat_vec_mul_into_slice(&[2, 3], &mut out).unwrap();
        assert_eq!(out, [13, 28]);
        slice.mat_vec_mul_mt_into_slice(&[2, 3], &mut out).unwrap();
        assert_eq!(out, [13, 28]);

        let mut out = [0, 0];
        slice.vec_mat_mul_into_slice(&[2, 3], &mut out).unwrap();
        assert_eq!(out, [19, 24]);
        slice.vec_mat_mul_mt_into_slice(&[2, 3], &mut out).unwrap();
        assert_eq!(out, [19, 24]);

        assert!(matches!(
            slice.mat_vec_mul(&[1]),
            Err(TensorErrors::IncompatibleShapes { .. })
        ));
        assert!(matches!(
            slice.vec_mat_mul_mt(&[1]),
            Err(TensorErrors::IncompatibleShapes { .. })
        ));
        assert!(matches!(
            slice.mat_vec_mul_into_slice(&[1, 2], &mut []),
            Err(TensorErrors::IncompatibleShapes { .. })
        ));
    }

    #[test]
    fn matrix_and_slice_vector_operations_reject_all_incompatible_lengths() {
        let matrix = matrix();
        assert!(matrix.clone().mat_vec_mul(&[1, 2]).is_err());
        assert!(matrix.clone().mat_vec_mul_mt(&[1, 2]).is_err());
        assert!(matrix.clone().vec_mat_mul(&[1]).is_err());
        assert!(matrix.clone().vec_mat_mul_mt(&[1]).is_err());

        assert!(matrix
            .clone()
            .mat_vec_mul_into_slice(&[1, 2], &mut [0, 0])
            .is_err());
        assert!(matrix
            .clone()
            .mat_vec_mul_into_slice(&[1, 2, 3], &mut [0])
            .is_err());
        assert!(matrix
            .clone()
            .mat_vec_mul_mt_into_slice(&[1, 2], &mut [0, 0])
            .is_err());
        assert!(matrix
            .clone()
            .mat_vec_mul_mt_into_slice(&[1, 2, 3], &mut [0])
            .is_err());
        assert!(matrix
            .clone()
            .vec_mat_mul_into_slice(&[1], &mut [0, 0, 0])
            .is_err());
        assert!(matrix
            .clone()
            .vec_mat_mul_into_slice(&[1, 2], &mut [0])
            .is_err());
        assert!(matrix
            .clone()
            .vec_mat_mul_mt_into_slice(&[1], &mut [0, 0, 0])
            .is_err());
        assert!(matrix
            .clone()
            .vec_mat_mul_mt_into_slice(&[1, 2], &mut [0])
            .is_err());

        let slice = matrix.slice(0..2, 0..3).unwrap();
        assert!(slice.mat_vec_mul(&[1, 2]).is_err());
        assert!(slice.mat_vec_mul_mt(&[1, 2]).is_err());
        assert!(slice.vec_mat_mul(&[1]).is_err());
        assert!(slice.vec_mat_mul_mt(&[1]).is_err());
        assert!(slice.mat_vec_mul_into_slice(&[1, 2], &mut [0, 0]).is_err());
        assert!(slice.mat_vec_mul_into_slice(&[1, 2, 3], &mut [0]).is_err());
        assert!(slice
            .mat_vec_mul_mt_into_slice(&[1, 2], &mut [0, 0])
            .is_err());
        assert!(slice
            .mat_vec_mul_mt_into_slice(&[1, 2, 3], &mut [0])
            .is_err());
        assert!(slice.vec_mat_mul_into_slice(&[1], &mut [0, 0, 0]).is_err());
        assert!(slice.vec_mat_mul_into_slice(&[1, 2], &mut [0]).is_err());
        assert!(slice
            .vec_mat_mul_mt_into_slice(&[1], &mut [0, 0, 0])
            .is_err());
        assert!(slice.vec_mat_mul_mt_into_slice(&[1, 2], &mut [0]).is_err());
    }

    #[test]
    fn vector_products_handle_zero_dimensions_for_matrices_and_slices() {
        let matrix = Matrix::<i32>::new(2, 0, vec![]).unwrap();
        assert_eq!(matrix.clone().mat_vec_mul(&[]), Ok(vec![0, 0]));
        assert_eq!(matrix.clone().mat_vec_mul_mt(&[]), Ok(vec![0, 0]));
        assert_eq!(matrix.clone().vec_mat_mul(&[3, 4]), Ok(vec![]));
        assert_eq!(matrix.clone().vec_mat_mul_mt(&[3, 4]), Ok(vec![]));
        let mut mat_out = [1, 1];
        matrix
            .clone()
            .mat_vec_mul_into_slice(&[], &mut mat_out)
            .unwrap();
        matrix
            .clone()
            .mat_vec_mul_mt_into_slice(&[], &mut mat_out)
            .unwrap();
        assert_eq!(mat_out, [0, 0]);
        let mut vec_out: [i32; 0] = [];
        matrix
            .clone()
            .vec_mat_mul_into_slice(&[3, 4], &mut vec_out)
            .unwrap();
        matrix
            .vec_mat_mul_mt_into_slice(&[3, 4], &mut vec_out)
            .unwrap();

        let matrix = Matrix::<i32>::new(0, 3, vec![]).unwrap();
        assert_eq!(matrix.clone().mat_vec_mul(&[1, 2, 3]), Ok(vec![]));
        assert_eq!(matrix.clone().mat_vec_mul_mt(&[1, 2, 3]), Ok(vec![]));
        assert_eq!(matrix.clone().vec_mat_mul(&[]), Ok(vec![0, 0, 0]));
        assert_eq!(matrix.clone().vec_mat_mul_mt(&[]), Ok(vec![0, 0, 0]));
        let mut vec_out = [1, 1, 1];
        matrix
            .clone()
            .vec_mat_mul_into_slice(&[], &mut vec_out)
            .unwrap();
        matrix.vec_mat_mul_mt_into_slice(&[], &mut vec_out).unwrap();
        assert_eq!(vec_out, [0, 0, 0]);

        let backing = Matrix::<i32>::new(3, 4, vec![0; 12]).unwrap();
        for (rows, cols) in [(0..0, 1..4), (1..3, 2..2), (0..0, 0..0)] {
            let slice = backing.slice(rows, cols).unwrap();
            let lhs = vec![1; slice.cols()];
            let rhs = vec![1; slice.rows()];
            assert_eq!(slice.mat_vec_mul(&lhs).unwrap().len(), slice.rows());
            assert_eq!(slice.mat_vec_mul_mt(&lhs).unwrap().len(), slice.rows());
            assert_eq!(slice.vec_mat_mul(&rhs).unwrap().len(), slice.cols());
            assert_eq!(slice.vec_mat_mul_mt(&rhs).unwrap().len(), slice.cols());
        }
    }
}

#[cfg(test)]
mod noncommutative_mat_vec_tests {
    use num::Zero;
    use std::ops::{Add, AddAssign, Mul};

    use tensor_math::definitions::matrix::Matrix;

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct NC(i32, i32);

    impl Mul for NC {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            Self(self.0 * rhs.0, self.1 * rhs.0 + rhs.1)
        }
    }

    impl Add for NC {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            Self(self.0 + rhs.0, self.1 + rhs.1)
        }
    }

    impl AddAssign for NC {
        fn add_assign(&mut self, rhs: Self) {
            self.0 += rhs.0;
            self.1 += rhs.1;
        }
    }

    impl Zero for NC {
        fn zero() -> Self {
            Self(0, 0)
        }

        fn is_zero(&self) -> bool {
            *self == Self::zero()
        }
    }

    fn nc(value: i32) -> NC {
        NC(value, 1)
    }

    #[test]
    fn matrix_vector_multiplication_preserves_noncommutative_operand_order() {
        let matrix = Matrix::new(2, 2, vec![nc(1), nc(2), nc(3), nc(4)]).unwrap();
        let vector = [nc(5), nc(6)];

        let mat_vec = vec![NC(17, 13), NC(39, 13)];
        let vec_mat = vec![NC(23, 6), NC(34, 8)];
        assert_eq!(matrix.clone().mat_vec_mul(&vector).unwrap(), mat_vec);
        assert_eq!(matrix.clone().mat_vec_mul_mt(&vector).unwrap(), mat_vec);
        assert_eq!(matrix.clone().vec_mat_mul(&vector).unwrap(), vec_mat);
        assert_eq!(matrix.clone().vec_mat_mul_mt(&vector).unwrap(), vec_mat);

        let mut output = [NC::zero(), NC::zero()];
        matrix
            .clone()
            .mat_vec_mul_into_slice(&vector, &mut output)
            .unwrap();
        assert_eq!(output.as_slice(), mat_vec);
        matrix
            .clone()
            .mat_vec_mul_mt_into_slice(&vector, &mut output)
            .unwrap();
        assert_eq!(output.as_slice(), mat_vec);
        matrix
            .clone()
            .vec_mat_mul_into_slice(&vector, &mut output)
            .unwrap();
        assert_eq!(output.as_slice(), vec_mat);
        matrix
            .vec_mat_mul_mt_into_slice(&vector, &mut output)
            .unwrap();
        assert_eq!(output.as_slice(), vec_mat);

        let mut slice_backing = Matrix::new(4, 4, vec![NC(0, 0); 16]).unwrap();
        slice_backing[(1, 1)] = nc(1);
        slice_backing[(1, 2)] = nc(2);
        slice_backing[(2, 1)] = nc(3);
        slice_backing[(2, 2)] = nc(4);
        let slice = slice_backing.slice(1..3, 1..3).unwrap();
        assert_eq!(slice.mat_vec_mul(&vector).unwrap(), mat_vec);
        assert_eq!(slice.mat_vec_mul_mt(&vector).unwrap(), mat_vec);
        assert_eq!(slice.vec_mat_mul(&vector).unwrap(), vec_mat);
        assert_eq!(slice.vec_mat_mul_mt(&vector).unwrap(), vec_mat);
    }
}
