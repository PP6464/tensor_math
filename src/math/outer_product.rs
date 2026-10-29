use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::definitions::matrix_slice_mut::MatrixSliceMut;
use crate::shape;
use rayon::iter::ParallelIterator;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator};
use rayon::slice::ParallelSliceMut;
use std::ops::{AddAssign, Mul, SubAssign};
/*
--------------------------------------------
* Outer product functions
--------------------------------------------
*/

/// Computes the outer product of two vectors.
pub fn outer_product<T: Mul<Output = T> + Clone>(v1: &[T], v2: &[T]) -> Matrix<T> {
    let (l1, l2) = (v1.len(), v2.len());
    let mut elements = Vec::with_capacity(l1 * l2);

    for e1 in v1.iter() {
        for e2 in v2.iter() {
            elements.push(e1.clone() * e2.clone());
        }
    }

    Matrix {
        rows: l1,
        cols: l2,
        elements,
    }
}

/// Computes the outer product of two vectors.
pub fn outer_product_mt<T: Mul<Output = T> + Clone + Send + Sync>(v1: &[T], v2: &[T]) -> Matrix<T> {
    let (l1, l2) = (v1.len(), v2.len());
    if l1 == 0 || l2 == 0 {
        return Matrix {
            rows: l1,
            cols: l2,
            elements: Vec::new(),
        };
    }

    let mut elements = Vec::with_capacity(l1 * l2);
    let buf = elements.spare_capacity_mut();

    v1.par_iter()
        .zip(buf.par_chunks_mut(l2))
        .for_each(|(e1, row)| {
            for (e2, out) in v2.iter().zip(row.iter_mut()) {
                out.write(e1.clone() * e2.clone());
            }
        });

    unsafe {
        elements.set_len(l1 * l2);
    }

    Matrix {
        rows: l1,
        cols: l2,
        elements,
    }
}

/*
--------------------------------------------
* Rank-1 updates
--------------------------------------------
*/

impl<T> Matrix<T> {
    /// Does a rank 1 addition update with the outer product of v1 and v2.
    /// This fails if the outer product of v1 and v2 would not match the shape of `self`.
    pub fn rank1_update_add(&mut self, v1: &[T], v2: &[T]) -> Result<(), TensorErrors>
    where
        T: Mul<Output = T> + Clone + AddAssign,
    {
        if self.rows != v1.len() || self.cols != v2.len() {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: shape![v1.len(), v2.len()],
                op: "rank1_update",
            });
        }

        unsafe { self.rank1_update_add_unchecked(v1, v2) };
        Ok(())
    }

    /// Does a rank 1 addition update with the outer product of v1 and v2 without validity checking.
    pub(crate) unsafe fn rank1_update_add_unchecked(&mut self, v1: &[T], v2: &[T])
    where
        T: Mul<Output = T> + Clone + AddAssign,
    {
        if self.rows == 0 || self.cols == 0 {
            return;
        }

        for (row, a) in self.chunks_mut(v2.len()).zip(v1) {
            for (out, b) in row.iter_mut().zip(v2) {
                *out += a.clone() * b.clone();
            }
        }
    }

    /// Does a rank 1 addition update with the outer product of v1 and v2.
    /// This fails if the outer product of v1 and v2 would not match the shape of `self`.
    pub fn rank1_update_add_mt(&mut self, v1: &[T], v2: &[T]) -> Result<(), TensorErrors>
    where
        T: Mul<Output = T> + Clone + AddAssign + Send + Sync,
    {
        if self.rows != v1.len() || self.cols != v2.len() {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: shape![v1.len(), v2.len()],
                op: "rank1_update",
            });
        }

        unsafe { self.rank1_update_add_unchecked_mt(v1, v2) };
        Ok(())
    }

    /// Does a rank 1 addition update with the outer product of v1 and v2 without validity checking.
    pub(crate) unsafe fn rank1_update_add_unchecked_mt(&mut self, v1: &[T], v2: &[T])
    where
        T: Mul<Output = T> + Clone + AddAssign + Send + Sync,
    {
        if self.rows == 0 || self.cols == 0 {
            return;
        }

        self.par_chunks_mut(v2.len())
            .zip(v1.par_iter())
            .for_each(|(out_row, e1)| {
                out_row.iter_mut().zip(v2).for_each(|(out, e2)| {
                    *out += e1.clone() * e2.clone();
                });
            });
    }

    /// Does a rank 1 subtraction update with the outer product of v1 and v2.
    /// This fails if the outer product of v1 and v2 would not match the shape of `self`.
    pub fn rank1_update_sub(&mut self, v1: &[T], v2: &[T]) -> Result<(), TensorErrors>
    where
        T: Mul<Output = T> + Clone + SubAssign,
    {
        if self.rows != v1.len() || self.cols != v2.len() {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: shape![v1.len(), v2.len()],
                op: "rank1_update",
            });
        }

        unsafe { self.rank1_update_sub_unchecked(v1, v2) };
        Ok(())
    }

    /// Does a rank 1 subtraction update with the outer product of v1 and v2 without validity checking.
    pub(crate) unsafe fn rank1_update_sub_unchecked(&mut self, v1: &[T], v2: &[T])
    where
        T: Mul<Output = T> + Clone + SubAssign,
    {
        if self.rows == 0 || self.cols == 0 {
            return;
        }

        for (row, a) in self.chunks_mut(v2.len()).zip(v1) {
            for (out, b) in row.iter_mut().zip(v2) {
                *out -= a.clone() * b.clone();
            }
        }
    }

    /// Does a rank 1 subtraction update with the outer product of v1 and v2.
    /// This fails if the outer product of v1 and v2 would not match the shape of `self`.
    pub fn rank1_update_sub_mt(&mut self, v1: &[T], v2: &[T]) -> Result<(), TensorErrors>
    where
        T: Mul<Output = T> + Clone + SubAssign + Send + Sync,
    {
        if self.rows != v1.len() || self.cols != v2.len() {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: shape![v1.len(), v2.len()],
                op: "rank1_update",
            });
        }

        unsafe { self.rank1_update_sub_unchecked_mt(v1, v2) };
        Ok(())
    }

    /// Does a rank 1 subtraction update with the outer product of v1 and v2 without validity checking.
    pub(crate) unsafe fn rank1_update_sub_unchecked_mt(&mut self, v1: &[T], v2: &[T])
    where
        T: Mul<Output = T> + Clone + SubAssign + Send + Sync,
    {
        if self.rows == 0 || self.cols == 0 {
            return;
        }

        self.par_chunks_mut(v2.len())
            .zip(v1.par_iter())
            .for_each(|(out_row, e1)| {
                out_row.iter_mut().zip(v2).for_each(|(out, e2)| {
                    *out -= e1.clone() * e2.clone();
                });
            });
    }
}

impl<T> MatrixSliceMut<'_, T> {
    /// Does a rank 1 addition update with the outer product of v1 and v2.
    /// This fails if the outer product of v1 and v2 would not match the shape of `self`.
    pub fn rank1_update_add(&mut self, v1: &[T], v2: &[T]) -> Result<(), TensorErrors>
    where
        T: Mul<Output = T> + Clone + AddAssign,
    {
        if self.rows() != v1.len() || self.cols() != v2.len() {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: shape![v1.len(), v2.len()],
                op: "rank1_update",
            });
        }

        unsafe { self.rank1_update_add_unchecked(v1, v2) };
        Ok(())
    }

    /// Does a rank 1 addition update with the outer product of v1 and v2 without validity checking.
    pub(crate) unsafe fn rank1_update_add_unchecked(&mut self, v1: &[T], v2: &[T])
    where
        T: Mul<Output = T> + Clone + AddAssign,
    {
        if self.rows() == 0 || self.cols() == 0 {
            return;
        }

        for (row, a) in self.iter_rows_mut().zip(v1) {
            for (out, b) in row.iter_mut().zip(v2) {
                *out += a.clone() * b.clone();
            }
        }
    }

    /// Does a rank 1 addition update with the outer product of v1 and v2.
    /// This fails if the outer product of v1 and v2 would not match the shape of `self`.
    pub fn rank1_update_add_mt(&mut self, v1: &[T], v2: &[T]) -> Result<(), TensorErrors>
    where
        T: Mul<Output = T> + Clone + AddAssign + Send + Sync,
    {
        if self.rows() != v1.len() || self.cols() != v2.len() {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: shape![v1.len(), v2.len()],
                op: "rank1_update",
            });
        }

        unsafe { self.rank1_update_add_unchecked_mt(v1, v2) };
        Ok(())
    }

    /// Does a rank 1 addition update with the outer product of v1 and v2 without validity checking.
    pub(crate) unsafe fn rank1_update_add_unchecked_mt(&mut self, v1: &[T], v2: &[T])
    where
        T: Mul<Output = T> + Clone + AddAssign + Send + Sync,
    {
        if self.rows() == 0 || self.cols() == 0 {
            return;
        }

        self.par_iter_rows_mut()
            .zip(v1.par_iter())
            .for_each(|(row, e1)| {
                row.iter_mut().zip(v2).for_each(|(out, e2)| {
                    *out += e1.clone() * e2.clone();
                });
            });
    }

    /// Does a rank 1 subtraction update with the outer product of v1 and v2.
    /// This fails if the outer product of v1 and v2 would not match the shape of `self`.
    pub fn rank1_update_sub(&mut self, v1: &[T], v2: &[T]) -> Result<(), TensorErrors>
    where
        T: Mul<Output = T> + Clone + SubAssign,
    {
        if self.rows() != v1.len() || self.cols() != v2.len() {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: shape![v1.len(), v2.len()],
                op: "rank1_update",
            });
        }

        unsafe { self.rank1_update_sub_unchecked(v1, v2) };
        Ok(())
    }

    /// Does a rank 1 subtraction update with the outer product of v1 and v2 without validity checking.
    pub(crate) unsafe fn rank1_update_sub_unchecked(&mut self, v1: &[T], v2: &[T])
    where
        T: Mul<Output = T> + Clone + SubAssign,
    {
        if self.rows() == 0 || self.cols() == 0 {
            return;
        }

        for (row, a) in self.iter_rows_mut().zip(v1) {
            for (out, b) in row.iter_mut().zip(v2) {
                *out -= a.clone() * b.clone();
            }
        }
    }

    /// Does a rank 1 subtraction update with the outer product of v1 and v2.
    /// This fails if the outer product of v1 and v2 would not match the shape of `self`.
    pub fn rank1_update_sub_mt(&mut self, v1: &[T], v2: &[T]) -> Result<(), TensorErrors>
    where
        T: Mul<Output = T> + Clone + SubAssign + Send + Sync,
    {
        if self.rows() != v1.len() || self.cols() != v2.len() {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: shape![v1.len(), v2.len()],
                op: "rank1_update",
            });
        }

        unsafe { self.rank1_update_sub_unchecked_mt(v1, v2) };
        Ok(())
    }

    /// Does a rank 1 subtraction update with the outer product of v1 and v2 without validity checking.
    pub(crate) unsafe fn rank1_update_sub_unchecked_mt(&mut self, v1: &[T], v2: &[T])
    where
        T: Mul<Output = T> + Clone + SubAssign + Send + Sync,
    {
        if self.rows() == 0 || self.cols() == 0 {
            return;
        }

        self.par_iter_rows_mut()
            .zip(v1.par_iter())
            .for_each(|(row, e1)| {
                row.iter_mut().zip(v2).for_each(|(out, e2)| {
                    *out -= e1.clone() * e2.clone();
                });
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outer_products_cover_normal_and_empty_vectors() {
        let expected = Matrix::new(2, 3, vec![4, 4, 6, 8, 8, 12]).unwrap();
        assert_eq!(outer_product(&[2, 4], &[2, 2, 3]), expected);
        assert_eq!(outer_product_mt(&[2, 4], &[2, 2, 3]), expected);

        let empty = Matrix::<i32>::new(2, 0, vec![]).unwrap();
        assert_eq!(outer_product(&[1, 2], &[]), empty);
        assert_eq!(outer_product_mt(&[1, 2], &[]), empty);
        assert_eq!(
            outer_product::<i32>(&[], &[1, 2]),
            Matrix::new(0, 2, vec![]).unwrap()
        );
        assert_eq!(
            outer_product_mt::<i32>(&[], &[1, 2]),
            Matrix::new(0, 2, vec![]).unwrap()
        );
        assert_eq!(
            outer_product::<i32>(&[], &[]),
            Matrix::new(0, 0, vec![]).unwrap()
        );
        assert_eq!(
            outer_product_mt::<i32>(&[], &[]),
            Matrix::new(0, 0, vec![]).unwrap()
        );
    }

    #[test]
    fn matrix_rank_one_updates_serial_parallel_and_shape_errors() {
        let (v1, v2) = ([1, 2], [3, 4]);
        let mut matrix = Matrix::new(2, 2, vec![10, 20, 30, 40]).unwrap();
        matrix.rank1_update_add(&v1, &v2).unwrap();
        assert_eq!(matrix.elements(), &[13, 24, 36, 48]);
        matrix.rank1_update_sub(&v1, &v2).unwrap();
        assert_eq!(matrix.elements(), &[10, 20, 30, 40]);
        matrix.rank1_update_add_mt(&v1, &v2).unwrap();
        matrix.rank1_update_sub_mt(&v1, &v2).unwrap();
        assert_eq!(matrix.elements(), &[10, 20, 30, 40]);

        assert!(matches!(
            matrix.rank1_update_add(&[1], &v2),
            Err(TensorErrors::IncompatibleShapes { .. })
        ));
        assert!(matches!(
            matrix.rank1_update_sub_mt(&v1, &[1]),
            Err(TensorErrors::IncompatibleShapes { .. })
        ));

        let mut backing = Matrix::new(4, 4, vec![0; 16]).unwrap();
        let mut slice = backing.slice_mut(1..3, 1..3).unwrap();
        slice.rank1_update_add(&v1, &v2).unwrap();
        assert_eq!(slice.clone_into_matrix().elements(), &[3, 4, 6, 8]);
        slice.rank1_update_sub(&v1, &v2).unwrap();
        slice.rank1_update_add_mt(&v1, &v2).unwrap();
        slice.rank1_update_sub_mt(&v1, &v2).unwrap();
        assert_eq!(slice.clone_into_matrix().elements(), &[0, 0, 0, 0]);
        assert!(matches!(
            slice.rank1_update_add_mt(&[1], &v2),
            Err(TensorErrors::IncompatibleShapes { .. })
        ));
    }

    #[test]
    fn rank_one_updates_handle_zero_dimensions() {
        let mut zero_columns = Matrix::<i32>::new(2, 0, vec![]).unwrap();
        zero_columns.rank1_update_add(&[1, 2], &[]).unwrap();
        zero_columns.rank1_update_sub_mt(&[1, 2], &[]).unwrap();
        assert!(zero_columns.elements().is_empty());

        let mut zero_rows = Matrix::<i32>::new(0, 3, vec![]).unwrap();
        zero_rows.rank1_update_add_mt(&[], &[1, 2, 3]).unwrap();
        zero_rows.rank1_update_sub(&[], &[1, 2, 3]).unwrap();
        assert!(zero_rows.elements().is_empty());

        let mut backing = Matrix::<i32>::new(3, 4, vec![0; 12]).unwrap();
        let mut slice = backing.slice_mut(1..3, 2..2).unwrap();
        slice.rank1_update_add_mt(&[1, 2], &[]).unwrap();
        slice.rank1_update_sub(&[1, 2], &[]).unwrap();
        assert_eq!(slice.clone_into_matrix().elements(), &[]);
    }
}

#[cfg(test)]
mod noncommutative_outer_product_tests {
    use std::ops::{AddAssign, Mul, SubAssign};

    use tensor_math::definitions::matrix::Matrix;

    use super::{outer_product, outer_product_mt};

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct NC(i32, i32);

    impl Mul for NC {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            Self(self.0 * rhs.0, self.1 * rhs.0 + rhs.1)
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

    fn nc(value: i32) -> NC {
        NC(value, 1)
    }

    fn vectors() -> ([NC; 2], [NC; 2]) {
        ([nc(1), nc(2)], [nc(3), nc(4)])
    }

    fn expected() -> Matrix<NC> {
        Matrix::new(2, 2, vec![NC(3, 4), NC(4, 5), NC(6, 4), NC(8, 5)]).unwrap()
    }

    #[test]
    fn outer_products_preserve_noncommutative_operand_order() {
        let (left, right) = vectors();
        assert_eq!(outer_product(&left, &right), expected());
        assert_eq!(outer_product_mt(&left, &right), expected());
        assert_ne!(outer_product(&left, &right), outer_product(&right, &left));
    }

    #[test]
    fn rank_one_updates_preserve_noncommutative_operand_order() {
        let (left, right) = vectors();
        let mut matrix = Matrix::new(2, 2, vec![nc(10); 4]).unwrap();
        matrix.rank1_update_add(&left, &right).unwrap();
        assert_eq!(
            matrix,
            Matrix::new(2, 2, vec![NC(13, 5), NC(14, 6), NC(16, 5), NC(18, 6)]).unwrap()
        );

        matrix.rank1_update_sub_mt(&left, &right).unwrap();
        assert_eq!(matrix, Matrix::new(2, 2, vec![nc(10); 4]).unwrap());
        matrix.rank1_update_add_mt(&left, &right).unwrap();
        matrix.rank1_update_sub(&left, &right).unwrap();
        assert_eq!(matrix, Matrix::new(2, 2, vec![nc(10); 4]).unwrap());

        let mut backing = Matrix::new(4, 4, vec![NC(0, 0); 16]).unwrap();
        let mut slice = backing.slice_mut(1..3, 1..3).unwrap();
        slice.rank1_update_add_mt(&left, &right).unwrap();
        assert_eq!(slice.clone_into_matrix(), expected());
        slice.rank1_update_sub(&left, &right).unwrap();
        assert_eq!(
            slice.clone_into_matrix(),
            Matrix::new(2, 2, vec![NC(0, 0); 4]).unwrap()
        );
    }
}
