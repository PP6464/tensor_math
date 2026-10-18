use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::definitions::matrix_slice_mut::MatrixSliceMut;
use crate::definitions::shape::Shape;
use crate::shape;
use crate::utilities::internal_functions::dot_vectors;
use num::Zero;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use std::ops::{Add, AddAssign, Mul, SubAssign};
/*
--------------------------------------------
GEMM functions
--------------------------------------------
*/

impl<T: Add<Output = T> + AddAssign + Mul<Output = T> + Zero + Clone> Matrix<T> {
    /// Computes the product of the matrices and adds it to this matrix.
    /// This fails if the matrices are not multiplicatively compatible or if the result
    /// does not have the same shape as this matrix.
    pub fn gemm_add(&mut self, m1: Matrix<T>, m2: Matrix<T>) -> Result<(), TensorErrors> {
        if m1.cols != m2.rows {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: m1.shape(),
                shape_2: m2.shape(),
                op: "gemm_add",
            });
        }

        if m1.rows != self.rows || m2.cols != self.cols {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: shape![m1.rows, m2.cols],
                shape_2: self.shape(),
                op: "gemm_add",
            });
        }

        unsafe { self.gemm_add_unchecked(m1, m2) };
        Ok(())
    }

    /// Computes the products of the matrices and adds it to the matrix without bounds checking.
    pub(crate) unsafe fn gemm_add_unchecked(&mut self, m1: Matrix<T>, m2: Matrix<T>) {
        let m2_transpose = m2.transpose();

        self.iter_rows_mut()
            .zip(m1.iter_rows())
            .for_each(|(out_row, row)| {
                out_row
                    .iter_mut()
                    .zip(m2_transpose.iter_rows())
                    .for_each(|(out, col)| {
                        *out += dot_vectors(row, col);
                    })
            });
    }

    /// Computes the product of the matrices and adds it to this matrix.
    /// This fails if the matrices are not multiplicatively compatible or if the result
    /// does not have the same shape as this matrix.
    pub fn gemm_add_mt(&mut self, m1: Matrix<T>, m2: Matrix<T>) -> Result<(), TensorErrors>
    where
        T: Send + Sync,
    {
        if m1.cols != m2.rows {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: m1.shape(),
                shape_2: m2.shape(),
                op: "gemm_add_mt",
            });
        }

        if m1.rows != self.rows || m2.cols != self.cols {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: shape![m1.rows, m2.cols],
                shape_2: self.shape(),
                op: "gemm_add_mt",
            });
        }

        unsafe { self.gemm_add_unchecked_mt(m1, m2) };
        Ok(())
    }

    /// Computes the products of the matrices and adds it to the matrix without bounds checking.
    pub(crate) unsafe fn gemm_add_unchecked_mt(&mut self, m1: Matrix<T>, m2: Matrix<T>)
    where
        T: Send + Sync,
    {
        let m2_transpose = m2.transpose_mt();

        self.par_iter_rows_mut()
            .zip(m1.par_iter_rows())
            .for_each(|(out_row, row)| {
                out_row
                    .par_iter_mut()
                    .zip(m2_transpose.par_iter_rows())
                    .for_each(|(out, col)| {
                        *out += dot_vectors(row, col);
                    })
            });
    }

    /// Computes the product of the matrices and subtracts it from this matrix.
    /// This fails if the matrices are not multiplicatively compatible or if the result
    /// does not have the same shape as this matrix.
    pub fn gemm_sub(&mut self, m1: Matrix<T>, m2: Matrix<T>) -> Result<(), TensorErrors>
    where
        T: SubAssign,
    {
        if m1.cols != m2.rows {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: m1.shape(),
                shape_2: m2.shape(),
                op: "gemm_sub",
            });
        }

        if m1.rows != self.rows || m2.cols != self.cols {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: shape![m1.rows, m2.cols],
                shape_2: self.shape(),
                op: "gemm_sub",
            });
        }

        unsafe { self.gemm_sub_unchecked(m1, m2) };
        Ok(())
    }

    /// Computes the products of the matrices and subtracts it from the matrix without bounds checking.
    pub(crate) unsafe fn gemm_sub_unchecked(&mut self, m1: Matrix<T>, m2: Matrix<T>)
    where
        T: SubAssign,
    {
        let m2_transpose = m2.transpose();

        self.iter_rows_mut()
            .zip(m1.iter_rows())
            .for_each(|(out_row, row)| {
                out_row
                    .iter_mut()
                    .zip(m2_transpose.iter_rows())
                    .for_each(|(out, col)| {
                        *out -= dot_vectors(row, col);
                    })
            });
    }

    /// Computes the product of the matrices and subtracts it from this matrix.
    /// This fails if the matrices are not multiplicatively compatible or if the result
    /// does not have the same shape as this matrix.
    pub fn gemm_sub_mt(&mut self, m1: Matrix<T>, m2: Matrix<T>) -> Result<(), TensorErrors>
    where
        T: Send + Sync + SubAssign,
    {
        if m1.cols != m2.rows {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: m1.shape(),
                shape_2: m2.shape(),
                op: "gemm_sub_mt",
            });
        }

        if m1.rows != self.rows || m2.cols != self.cols {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: shape![m1.rows, m2.cols],
                shape_2: self.shape(),
                op: "gemm_sub_mt",
            });
        }

        unsafe { self.gemm_sub_unchecked_mt(m1, m2) };
        Ok(())
    }

    /// Computes the products of the matrices and subtracts it from the matrix without bounds checking.
    pub(crate) unsafe fn gemm_sub_unchecked_mt(&mut self, m1: Matrix<T>, m2: Matrix<T>)
    where
        T: Send + Sync + SubAssign,
    {
        let m2_transpose = m2.transpose_mt();

        self.par_iter_rows_mut()
            .zip(m1.par_iter_rows())
            .for_each(|(out_row, row)| {
                out_row
                    .par_iter_mut()
                    .zip(m2_transpose.par_iter_rows())
                    .for_each(|(out, col)| {
                        *out -= dot_vectors(row, col);
                    })
            });
    }
}

impl<T: Add<Output = T> + AddAssign + Mul<Output = T> + Zero + Clone> MatrixSliceMut<'_, T> {
    /// Computes the product of the matrices and adds it to this matrix slice.
    /// This fails if the matrices are not multiplicatively compatible or if the result
    /// does not have the same shape as this matrix.
    pub fn gemm_add(&mut self, m1: Matrix<T>, m2: Matrix<T>) -> Result<(), TensorErrors> {
        if m1.cols != m2.rows {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: m1.shape(),
                shape_2: m2.shape(),
                op: "gemm_add",
            });
        }

        if m1.rows != self.rows() || m2.cols != self.cols() {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: shape![m1.rows, m2.cols],
                shape_2: self.shape(),
                op: "gemm_add",
            });
        }

        unsafe { self.gemm_add_unchecked(m1, m2) };
        Ok(())
    }

    /// Computes the products of the matrices and adds it to the matrix slice without bounds checking.
    pub(crate) unsafe fn gemm_add_unchecked(&mut self, m1: Matrix<T>, m2: Matrix<T>) {
        let m2_transpose = m2.transpose();

        self.iter_rows_mut()
            .zip(m1.iter_rows())
            .for_each(|(out_row, row)| {
                out_row
                    .iter_mut()
                    .zip(m2_transpose.iter_rows())
                    .for_each(|(out, col)| {
                        *out += dot_vectors(row, col);
                    })
            });
    }

    /// Computes the product of the matrices and adds it to this matrix slice.
    /// This fails if the matrices are not multiplicatively compatible or if the result
    /// does not have the same shape as this matrix.
    pub fn gemm_add_mt(&mut self, m1: Matrix<T>, m2: Matrix<T>) -> Result<(), TensorErrors>
    where
        T: Send + Sync,
    {
        if m1.cols != m2.rows {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: m1.shape(),
                shape_2: m2.shape(),
                op: "gemm_add_mt",
            });
        }

        if m1.rows != self.rows() || m2.cols != self.cols() {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: shape![m1.rows, m2.cols],
                shape_2: self.shape(),
                op: "gemm_add_mt",
            });
        }

        unsafe { self.gemm_add_unchecked_mt(m1, m2) };
        Ok(())
    }

    /// Computes the products of the matrices and adds it to the matrix slice without bounds checking.
    pub(crate) unsafe fn gemm_add_unchecked_mt(&mut self, m1: Matrix<T>, m2: Matrix<T>)
    where
        T: Send + Sync,
    {
        let m2_transpose = m2.transpose_mt();

        self.par_iter_rows_mut()
            .zip(m1.par_iter_rows())
            .for_each(|(out_row, row)| {
                out_row
                    .into_par_iter()
                    .zip(m2_transpose.par_iter_rows())
                    .for_each(|(out, col)| {
                        *out += dot_vectors(row, col);
                    })
            });
    }

    /// Computes the product of the matrices and subtracts it from this matrix slice.
    /// This fails if the matrices are not multiplicatively compatible or if the result
    /// does not have the same shape as this matrix.
    pub fn gemm_sub(&mut self, m1: Matrix<T>, m2: Matrix<T>) -> Result<(), TensorErrors>
    where
        T: SubAssign,
    {
        if m1.cols != m2.rows {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: m1.shape(),
                shape_2: m2.shape(),
                op: "gemm_sub",
            });
        }

        if m1.rows != self.rows() || m2.cols != self.cols() {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: shape![m1.rows, m2.cols],
                shape_2: self.shape(),
                op: "gemm_sub",
            });
        }

        unsafe { self.gemm_sub_unchecked(m1, m2) };
        Ok(())
    }

    /// Computes the products of the matrices and subtracts it from the matrix slice without bounds checking.
    pub(crate) unsafe fn gemm_sub_unchecked(&mut self, m1: Matrix<T>, m2: Matrix<T>)
    where
        T: SubAssign,
    {
        let m2_transpose = m2.transpose();

        self.iter_rows_mut()
            .zip(m1.iter_rows())
            .for_each(|(out_row, row)| {
                out_row
                    .iter_mut()
                    .zip(m2_transpose.iter_rows())
                    .for_each(|(out, col)| {
                        *out -= dot_vectors(row, col);
                    })
            });
    }

    /// Computes the product of the matrices and subtracts it from this matrix slice.
    /// This fails if the matrices are not multiplicatively compatible or if the result
    /// does not have the same shape as this matrix.
    pub fn gemm_sub_mt(&mut self, m1: Matrix<T>, m2: Matrix<T>) -> Result<(), TensorErrors>
    where
        T: Send + Sync + SubAssign,
    {
        if m1.cols != m2.rows {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: m1.shape(),
                shape_2: m2.shape(),
                op: "gemm_sub_mt",
            });
        }

        if m1.rows != self.rows() || m2.cols != self.cols() {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: shape![m1.rows, m2.cols],
                shape_2: self.shape(),
                op: "gemm_sub_mt",
            });
        }

        unsafe { self.gemm_sub_unchecked_mt(m1, m2) };
        Ok(())
    }

    /// Computes the products of the matrices and subtracts it from the matrix slice without bounds checking.
    pub(crate) unsafe fn gemm_sub_unchecked_mt(&mut self, m1: Matrix<T>, m2: Matrix<T>)
    where
        T: Send + Sync + SubAssign,
    {
        let m2_transpose = m2.transpose_mt();

        self.par_iter_rows_mut()
            .zip(m1.par_iter_rows())
            .for_each(|(out_row, row)| {
                out_row
                    .into_par_iter()
                    .zip(m2_transpose.par_iter_rows())
                    .for_each(|(out, col)| {
                        *out -= dot_vectors(row, col);
                    })
            });
    }
}
