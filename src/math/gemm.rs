use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::definitions::matrix_slice_mut::MatrixSliceMut;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn operands() -> (Matrix<i32>, Matrix<i32>) {
        (
            Matrix::new(2, 2, vec![1, 2, 3, 4]).unwrap(),
            Matrix::new(2, 2, vec![5, 6, 7, 8]).unwrap(),
        )
    }

    #[test]
    fn matrix_gemm_add_and_sub_serial_and_parallel() {
        let (left, right) = operands();
        let mut output = Matrix::new(2, 2, vec![10, 10, 10, 10]).unwrap();
        output.gemm_add(left.clone(), right.clone()).unwrap();
        assert_eq!(output.elements(), &[29, 32, 53, 60]);
        output.gemm_sub(left.clone(), right.clone()).unwrap();
        assert_eq!(output.elements(), &[10, 10, 10, 10]);

        output.gemm_add_mt(left.clone(), right.clone()).unwrap();
        assert_eq!(output.elements(), &[29, 32, 53, 60]);
        output.gemm_sub_mt(left.clone(), right.clone()).unwrap();
        assert_eq!(output.elements(), &[10, 10, 10, 10]);

        let mut empty = Matrix::<i32>::new(0, 2, vec![]).unwrap();
        empty
            .gemm_add(
                Matrix::new(0, 3, vec![]).unwrap(),
                Matrix::new(3, 2, vec![0; 6]).unwrap(),
            )
            .unwrap();
        empty
            .gemm_sub_mt(
                Matrix::new(0, 3, vec![]).unwrap(),
                Matrix::new(3, 2, vec![0; 6]).unwrap(),
            )
            .unwrap();
        assert!(empty.elements().is_empty());

        assert!(matches!(
            output.gemm_add(
                Matrix::new(2, 3, vec![0; 6]).unwrap(),
                Matrix::new(2, 2, vec![0; 4]).unwrap()
            ),
            Err(TensorErrors::IncompatibleShapes { .. })
        ));
        assert!(matches!(
            output.gemm_sub_mt(
                Matrix::new(2, 2, vec![0; 4]).unwrap(),
                Matrix::new(2, 3, vec![0; 6]).unwrap()
            ),
            Err(TensorErrors::IncompatibleShapes { .. })
        ));
    }

    #[test]
    fn matrix_slice_gemm_add_and_sub_serial_and_parallel() {
        let (left, right) = operands();
        let mut backing = Matrix::new(4, 4, vec![0; 16]).unwrap();
        {
            let mut slice = backing.slice_mut(1..3, 1..3).unwrap();
            slice.gemm_add(left.clone(), right.clone()).unwrap();
            assert_eq!(slice.clone_into_matrix().elements(), &[19, 22, 43, 50]);
            slice.gemm_sub(left.clone(), right.clone()).unwrap();
            assert_eq!(slice.clone_into_matrix().elements(), &[0, 0, 0, 0]);
            slice.gemm_add_mt(left.clone(), right.clone()).unwrap();
            slice.gemm_sub_mt(left.clone(), right.clone()).unwrap();
            assert_eq!(slice.clone_into_matrix().elements(), &[0, 0, 0, 0]);

            assert!(matches!(
                slice.gemm_add(
                    Matrix::new(2, 3, vec![0; 6]).unwrap(),
                    Matrix::new(2, 2, vec![0; 4]).unwrap()
                ),
                Err(TensorErrors::IncompatibleShapes { .. })
            ));
            assert!(matches!(
                slice.gemm_sub_mt(
                    Matrix::new(2, 2, vec![0; 4]).unwrap(),
                    Matrix::new(2, 3, vec![0; 6]).unwrap()
                ),
                Err(TensorErrors::IncompatibleShapes { .. })
            ));
        }
        assert_eq!(backing.elements().len(), 16);
    }

    #[test]
    fn gemm_handles_zero_inner_and_empty_output_dimensions() {
        let mut output = Matrix::new(2, 3, vec![1, 2, 3, 4, 5, 6]).unwrap();
        let left = Matrix::<i32>::new(2, 0, vec![]).unwrap();
        let right = Matrix::<i32>::new(0, 3, vec![]).unwrap();

        output.gemm_add(left.clone(), right.clone()).unwrap();
        output.gemm_sub_mt(left, right).unwrap();
        assert_eq!(output.elements(), &[1, 2, 3, 4, 5, 6]);

        let mut empty_rows = Matrix::<i32>::new(0, 3, vec![]).unwrap();
        empty_rows
            .gemm_add_mt(
                Matrix::new(0, 2, vec![]).unwrap(),
                Matrix::new(2, 3, vec![1; 6]).unwrap(),
            )
            .unwrap();
        let mut empty_cols = Matrix::<i32>::new(2, 0, vec![]).unwrap();
        empty_cols
            .gemm_sub(
                Matrix::new(2, 3, vec![1; 6]).unwrap(),
                Matrix::new(3, 0, vec![]).unwrap(),
            )
            .unwrap();
        assert!(empty_rows.elements().is_empty());
        assert!(empty_cols.elements().is_empty());

        let mut slice_backing = Matrix::new(4, 5, vec![9; 20]).unwrap();
        let mut slice = slice_backing.slice_mut(1..3, 1..4).unwrap();
        slice
            .gemm_add_mt(
                Matrix::new(2, 0, vec![]).unwrap(),
                Matrix::new(0, 3, vec![]).unwrap(),
            )
            .unwrap();
        assert_eq!(slice.clone_into_matrix().elements(), &[9; 6]);
    }
}

#[cfg(test)]
mod noncommutative_gemm_tests {
    use num::Zero;
    use std::ops::{Add, AddAssign, Mul, SubAssign};

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

    impl SubAssign for NC {
        fn sub_assign(&mut self, rhs: Self) {
            self.0 -= rhs.0;
            self.1 -= rhs.1;
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

    fn operands() -> (Matrix<NC>, Matrix<NC>) {
        (
            Matrix::new(2, 3, vec![nc(1), nc(2), nc(3), nc(4), nc(5), nc(6)]).unwrap(),
            Matrix::new(3, 2, vec![nc(7), nc(8), nc(9), nc(10), nc(11), nc(12)]).unwrap(),
        )
    }

    fn expected() -> Matrix<NC> {
        Matrix::new(2, 2, vec![NC(58, 30), NC(64, 33), NC(139, 30), NC(154, 33)]).unwrap()
    }

    #[test]
    fn gemm_preserves_noncommutative_operand_order_serial_parallel_and_slices() {
        let (left, right) = operands();
        let expected = expected();

        let mut out = Matrix::new(2, 2, vec![NC::zero(); 4]).unwrap();
        out.gemm_add(left.clone(), right.clone()).unwrap();
        assert_eq!(out, expected);
        out.gemm_sub(left.clone(), right.clone()).unwrap();
        assert_eq!(out, Matrix::new(2, 2, vec![NC::zero(); 4]).unwrap());
        out.gemm_add_mt(left.clone(), right.clone()).unwrap();
        assert_eq!(out, expected);
        out.gemm_sub_mt(left.clone(), right.clone()).unwrap();
        assert_eq!(out, Matrix::new(2, 2, vec![NC::zero(); 4]).unwrap());

        let mut backing = Matrix::new(4, 4, vec![NC::zero(); 16]).unwrap();
        let mut slice = backing.slice_mut(1..3, 1..3).unwrap();
        slice.gemm_add(left.clone(), right.clone()).unwrap();
        assert_eq!(slice.clone_into_matrix(), expected);
        slice.gemm_sub(left.clone(), right.clone()).unwrap();
        slice.gemm_add_mt(left.clone(), right.clone()).unwrap();
        assert_eq!(slice.clone_into_matrix(), expected);
        slice.gemm_sub_mt(left, right).unwrap();
        assert_eq!(
            slice.clone_into_matrix(),
            Matrix::new(2, 2, vec![NC::zero(); 4]).unwrap()
        );
    }
}
