use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::definitions::shape::Shape;
use crate::shape;
use rayon::iter::ParallelIterator;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator};
use rayon::slice::ParallelSliceMut;
use std::ops::{AddAssign, Mul, SubAssign};
use crate::definitions::matrix_slice_mut::MatrixSliceMut;
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

        for (row, a) in self.chunks_mut(v2.len()).zip(v1) {
            for (out, b) in row.iter_mut().zip(v2) {
                *out += a.clone() * b.clone();
            }
        }

        Ok(())
    }

    /// Does a rank 1 addition update with the outer product of v1 and v2 without validity checking.
    pub(crate) unsafe fn rank1_update_add_unchecked(&mut self, v1: &[T], v2: &[T])
    where
        T: Mul<Output = T> + Clone + AddAssign,
    {
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

        self.par_chunks_mut(v2.len())
            .zip(v1.par_iter())
            .for_each(|(out_row, e1)| {
                out_row.iter_mut().zip(v2).for_each(|(out, e2)| {
                    *out += e1.clone() * e2.clone();
                });
            });

        Ok(())
    }

    /// Does a rank 1 addition update with the outer product of v1 and v2 without validity checking.
    pub(crate) unsafe fn rank1_update_add_unchecked_mt(&mut self, v1: &[T], v2: &[T])
    where
        T: Mul<Output = T> + Clone + AddAssign + Send + Sync,
    {
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

        for (row, a) in self.chunks_mut(v2.len()).zip(v1) {
            for (out, b) in row.iter_mut().zip(v2) {
                *out -= a.clone() * b.clone();
            }
        }

        Ok(())
    }

    /// Does a rank 1 subtraction update with the outer product of v1 and v2 without validity checking.
    pub(crate) unsafe fn rank1_update_sub_unchecked(&mut self, v1: &[T], v2: &[T])
    where
        T: Mul<Output = T> + Clone + SubAssign,
    {
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

        self.par_chunks_mut(v2.len())
            .zip(v1.par_iter())
            .for_each(|(out_row, e1)| {
                out_row.iter_mut().zip(v2).for_each(|(out, e2)| {
                    *out -= e1.clone() * e2.clone();
                });
            });

        Ok(())
    }

    /// Does a rank 1 subtraction update with the outer product of v1 and v2 without validity checking.
    pub(crate) unsafe fn rank1_update_sub_unchecked_mt(&mut self, v1: &[T], v2: &[T])
    where
        T: Mul<Output = T> + Clone + SubAssign + Send + Sync,
    {
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

        for (row, a) in self.iter_rows_mut().zip(v1) {
            for (out, b) in row.into_iter().zip(v2) {
                *out += a.clone() * b.clone();
            }
        }

        Ok(())
    }

    /// Does a rank 1 addition update with the outer product of v1 and v2 without validity checking.
    pub(crate) unsafe fn rank1_update_add_unchecked(&mut self, v1: &[T], v2: &[T])
    where
        T: Mul<Output = T> + Clone + AddAssign,
    {
        for (row, a) in self.iter_rows_mut().zip(v1) {
            for (out, b) in row.into_iter().zip(v2) {
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

        self.par_iter_rows_mut()
            .zip(v1.par_iter())
            .for_each(|(row, e1)| {
                row.iter_mut().zip(v2).for_each(|(out, e2)| {
                    *out += e1.clone() * e2.clone();
                });
            });

        Ok(())
    }

    /// Does a rank 1 addition update with the outer product of v1 and v2 without validity checking.
    pub(crate) unsafe fn rank1_update_add_unchecked_mt(&mut self, v1: &[T], v2: &[T])
    where
        T: Mul<Output = T> + Clone + AddAssign + Send + Sync,
    {
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

        for (row, a) in self.iter_rows_mut().zip(v1) {
            for (out, b) in row.into_iter().zip(v2) {
                *out -= a.clone() * b.clone();
            }
        }

        Ok(())
    }

    /// Does a rank 1 subtraction update with the outer product of v1 and v2 without validity checking.
    pub(crate) unsafe fn rank1_update_sub_unchecked(&mut self, v1: &[T], v2: &[T])
    where
        T: Mul<Output = T> + Clone + SubAssign,
    {
        for (row, a) in self.iter_rows_mut().zip(v1) {
            for (out, b) in row.into_iter().zip(v2) {
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

        self.par_iter_rows_mut()
            .zip(v1.par_iter())
            .for_each(|(row, e1)| {
                row.iter_mut().zip(v2).for_each(|(out, e2)| {
                    *out -= e1.clone() * e2.clone();
                });
            });

        Ok(())
    }

    /// Does a rank 1 subtraction update with the outer product of v1 and v2 without validity checking.
    pub(crate) unsafe fn rank1_update_sub_unchecked_mt(&mut self, v1: &[T], v2: &[T])
    where
        T: Mul<Output = T> + Clone + SubAssign + Send + Sync,
    {
        self.par_iter_rows_mut()
            .zip(v1.par_iter())
            .for_each(|(row, e1)| {
                row.iter_mut().zip(v2).for_each(|(out, e2)| {
                    *out -= e1.clone() * e2.clone();
                });
            });
    }
}
