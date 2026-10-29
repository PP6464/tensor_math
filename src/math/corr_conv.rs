use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::definitions::shape::Shape;
use crate::definitions::tensor::Tensor;
use crate::definitions::transpose::Transpose;
use crate::shape;
use num::Zero;
use std::collections::HashSet;
use std::ops::{Add, AddAssign, Mul};

impl<T: Clone + Add<Output = T> + Mul<Output = T> + Zero + AddAssign> Tensor<T> {
    /// Computes the correlation of two tensors across specified axes.
    /// This fails if the ranks of the tensors do not match, if the rank is zero, if either tensor is empty, if any axis is out of bounds, or if the shapes are incompatible on non-correlated axes.
    pub fn corr_axes(
        self,
        other: Tensor<T>,
        axes: &HashSet<usize>,
    ) -> Result<Tensor<T>, TensorErrors> {
        if self.rank() != other.rank() {
            return Err(TensorErrors::RanksDoNotMatch(self.rank(), other.rank()));
        }

        if self.rank() == 0 {
            return Err(TensorErrors::RankZero { op: "Correlation" });
        }

        if self.is_empty() || other.is_empty() {
            return Err(TensorErrors::TensorEmpty { op: "Correlation" });
        }

        let rank = self.rank();

        for &axis in axes {
            if axis >= rank {
                return Err(TensorErrors::AxisOutOfBounds { axis, rank });
            }
        }

        let mut perm_vec = Vec::with_capacity(rank);

        for i in 0..rank {
            if !axes.contains(&i) {
                perm_vec.push(i);
                if self.shape[i] != other.shape[i] {
                    return Err(TensorErrors::IncompatibleShapes {
                        shape_1: self.shape(),
                        shape_2: other.shape(),
                        op: "corr_axes",
                    });
                }
            }
        }

        perm_vec.extend(axes.iter());
        let perm = Transpose::new(&perm_vec)?;
        let inv_perm = perm.clone().inverse();

        let self_perm = self.transpose(&perm)?;
        let other_perm = other.transpose(&perm)?;

        let padded_shape_vec = self_perm
            .shape
            .0
            .iter()
            .zip(other_perm.shape.0.iter())
            .enumerate()
            .map(|(i, (&s, &o))| {
                if i >= rank - axes.len() {
                    s + 2 * (o - 1)
                } else {
                    s
                }
            })
            .collect::<Vec<_>>();

        let padded_shape = Shape::new(padded_shape_vec);

        let res_shape_vec = self_perm
            .shape
            .0
            .iter()
            .zip(other_perm.shape.0.iter())
            .enumerate()
            .map(|(i, (&s, &o))| if i >= rank - axes.len() { s + o - 1 } else { s })
            .collect::<Vec<_>>();

        let kernel_shape_vec = other_perm
            .shape
            .0
            .iter()
            .enumerate()
            .map(|(i, &o)| if i >= rank - axes.len() { o } else { 1 })
            .collect::<Vec<_>>();

        let kernel_shape = Shape::new(kernel_shape_vec);

        let mut self_padded = Self::zeros(padded_shape);
        unsafe {
            self_padded
                .slice_unchecked_mut(
                    &self_perm
                        .shape
                        .0
                        .iter()
                        .zip(other_perm.shape.0.iter())
                        .enumerate()
                        .map(|(i, (&s, &o))| {
                            if i >= rank - axes.len() {
                                o - 1..o - 1 + s
                            } else {
                                0..s
                            }
                        })
                        .collect::<Vec<_>>(),
                )
                .set_all(&self_perm)?;
        }

        unsafe {
            Ok(self_padded
                .pool_indexed(
                    |index, t| {
                        if t.shape() == kernel_shape {
                            (other_perm
                                .slice_unchecked(
                                    &other_perm
                                        .shape
                                        .0
                                        .iter()
                                        .enumerate()
                                        .map(|(i, &o)| {
                                            if i >= rank - axes.len() {
                                                0..o
                                            } else {
                                                index[i]..index[i] + 1
                                            }
                                        })
                                        .collect::<Vec<_>>(),
                                )
                                .clone_into_tensor()
                                * t.clone_into_tensor())
                            .sum()
                        } else {
                            T::zero()
                        }
                    },
                    &kernel_shape,
                    &Shape::new(vec![1; rank]),
                )?
                .slice_unchecked(&res_shape_vec.iter().map(|&x| 0..x).collect::<Vec<_>>())
                .clone_into_tensor()
                .transpose_unchecked(&inv_perm))
        }
    }

    /// Computes the convolution of two tensors across specified axes.
    /// This fails if the ranks of the tensors do not match, if the rank is zero, if either tensor is empty, if any axis is out of bounds, or if the shapes are incompatible on non-convolved axes.
    pub fn conv_axes(
        self,
        other: Tensor<T>,
        axes: &HashSet<usize>,
    ) -> Result<Tensor<T>, TensorErrors> {
        self.corr_axes(other.flip_axes(axes)?, axes)
    }

    /// Computes the correlation of two tensors across all axes.
    /// This fails if the ranks of the tensors do not match, if the rank is zero, or if either tensor is empty.
    pub fn corr(self, other: Tensor<T>) -> Result<Tensor<T>, TensorErrors> {
        if self.rank() != other.rank() {
            return Err(TensorErrors::RanksDoNotMatch(self.rank(), other.rank()));
        }

        if self.rank() == 0 {
            return Err(TensorErrors::RankZero { op: "Correlation" });
        }

        if self.is_empty() || other.is_empty() {
            return Err(TensorErrors::TensorEmpty { op: "Correlation" });
        }

        let res_shape = Shape::new(
            self.shape
                .0
                .iter()
                .zip(other.shape.0.iter())
                .map(|(&s, &o)| s + o - 1)
                .collect::<Vec<_>>(),
        );

        let padded_shape = Shape::new(
            self.shape
                .0
                .iter()
                .zip(other.shape.0.iter())
                .map(|(&s, &o)| s + 2 * (o - 1))
                .collect::<Vec<_>>(),
        );

        let mut self_padded = Self::zeros(padded_shape);

        self_padded
            .slice_mut(
                &(0..self.rank())
                    .map(|i| other.shape[i] - 1..other.shape[i] + self.shape[i] - 1)
                    .collect::<Vec<_>>(),
            )?
            .set_all(&self)?;

        unsafe {
            Ok(self_padded
                .pool(
                    |t| {
                        if t.shape() == other.shape {
                            (&other * t.clone_into_tensor()).sum()
                        } else {
                            T::zero()
                        }
                    },
                    &other.shape,
                    &Shape::new(vec![1; self.rank()]),
                )?
                .slice_unchecked(&res_shape.0.iter().map(|&x| 0..x).collect::<Vec<_>>())
                .clone_into_tensor())
        }
    }

    /// Computes the convolution of two matrices across all axes.
    /// This fails if the underlying correlation fails, typically due to rank mismatch or empty tensors.
    pub fn conv(self, other: Tensor<T>) -> Result<Tensor<T>, TensorErrors> {
        self.corr(other.flip())
    }

    /// Computes the correlation of two tensors across specified axes on multiple threads.
    /// This fails if the ranks of the tensors do not match, if the rank is zero, if either tensor is empty, if any axis is out of bounds, or if the shapes are incompatible on non-correlated axes.
    pub fn corr_axes_mt(
        self,
        other: Tensor<T>,
        axes: &HashSet<usize>,
    ) -> Result<Tensor<T>, TensorErrors>
    where
        T: Send + Sync,
    {
        if self.rank() != other.rank() {
            return Err(TensorErrors::RanksDoNotMatch(self.rank(), other.rank()));
        }

        if self.rank() == 0 {
            return Err(TensorErrors::RankZero { op: "Correlation" });
        }

        if self.is_empty() || other.is_empty() {
            return Err(TensorErrors::TensorEmpty { op: "Correlation" });
        }

        let rank = self.rank();

        for &axis in axes {
            if axis >= rank {
                return Err(TensorErrors::AxisOutOfBounds { axis, rank });
            }
        }

        let mut perm_vec = Vec::with_capacity(rank);

        for i in 0..rank {
            if !axes.contains(&i) {
                perm_vec.push(i);
                if self.shape[i] != other.shape[i] {
                    return Err(TensorErrors::IncompatibleShapes {
                        shape_1: self.shape,
                        shape_2: other.shape,
                        op: "corr_axes_mt",
                    });
                }
            }
        }

        perm_vec.extend(axes.iter());
        let perm = Transpose::new(&perm_vec)?;
        let inv_perm = perm.clone().inverse();

        let self_perm = self.transpose(&perm)?;
        let other_perm = other.transpose(&perm)?;

        let padded_shape_vec = self_perm
            .shape
            .0
            .iter()
            .zip(other_perm.shape.0.iter())
            .enumerate()
            .map(|(i, (&s, &o))| {
                if i >= rank - axes.len() {
                    s + 2 * (o - 1)
                } else {
                    s
                }
            })
            .collect::<Vec<_>>();

        let padded_shape = Shape::new(padded_shape_vec);

        let res_shape_vec = self_perm
            .shape
            .0
            .iter()
            .zip(other_perm.shape.0.iter())
            .enumerate()
            .map(|(i, (&s, &o))| if i >= rank - axes.len() { s + o - 1 } else { s })
            .collect::<Vec<_>>();

        let kernel_shape_vec = other_perm
            .shape
            .0
            .iter()
            .enumerate()
            .map(|(i, &o)| if i >= rank - axes.len() { o } else { 1 })
            .collect::<Vec<_>>();

        let kernel_shape = Shape::new(kernel_shape_vec);

        let mut self_padded = Self::zeros(padded_shape);
        unsafe {
            self_padded
                .slice_unchecked_mut(
                    &self_perm
                        .shape
                        .0
                        .iter()
                        .zip(other_perm.shape.0.iter())
                        .enumerate()
                        .map(|(i, (&s, &o))| {
                            if i >= rank - axes.len() {
                                o - 1..o - 1 + s
                            } else {
                                0..s
                            }
                        })
                        .collect::<Vec<_>>(),
                )
                .set_all(&self_perm)?;
        }

        unsafe {
            Ok(self_padded
                .pool_indexed_mt(
                    &|index, t| {
                        if t.shape() == kernel_shape {
                            (other_perm
                                .slice_unchecked(
                                    &other_perm
                                        .shape
                                        .0
                                        .iter()
                                        .enumerate()
                                        .map(|(i, &o)| {
                                            if i >= rank - axes.len() {
                                                0..o
                                            } else {
                                                index[i]..index[i] + 1
                                            }
                                        })
                                        .collect::<Vec<_>>(),
                                )
                                .clone_into_tensor()
                                * t.clone_into_tensor())
                            .sum()
                        } else {
                            T::zero()
                        }
                    },
                    &kernel_shape,
                    &Shape::new(vec![1; rank]),
                )?
                .slice_unchecked(&res_shape_vec.iter().map(|&x| 0..x).collect::<Vec<_>>())
                .clone_into_tensor()
                .transpose_unchecked(&inv_perm))
        }
    }

    /// Computes the convolution of two tensors across specified axes on multiple threads.
    /// This fails if the ranks of the tensors do not match, if the rank is zero, if either tensor is empty, if any axis is out of bounds, or if the shapes are incompatible on non-correlated axes.
    pub fn conv_axes_mt(
        self,
        other: Tensor<T>,
        axes: &HashSet<usize>,
    ) -> Result<Tensor<T>, TensorErrors>
    where
        T: Send + Sync, {
        self.corr_axes_mt(other.flip_axes_mt(axes)?, axes)
    }

    /// Computes the correlation of two tensors across all axes on multiple threads.
    /// This fails if the ranks of the tensors do not match, if the rank is zero, or if either tensor is empty.
    pub fn corr_mt(self, other: Tensor<T>) -> Result<Tensor<T>, TensorErrors>
    where
        T: Send + Sync,
    {
        if self.rank() != other.rank() {
            return Err(TensorErrors::RanksDoNotMatch(self.rank(), other.rank()));
        }

        if self.rank() == 0 {
            return Err(TensorErrors::RankZero { op: "Correlation" });
        }

        if self.is_empty() || other.is_empty() {
            return Err(TensorErrors::TensorEmpty { op: "Correlation" });
        }

        let res_shape = Shape::new(
            self.shape
                .0
                .iter()
                .zip(other.shape.0.iter())
                .map(|(&s, &o)| s + o - 1)
                .collect::<Vec<_>>(),
        );

        let padded_shape = Shape::new(
            self.shape
                .0
                .iter()
                .zip(other.shape.0.iter())
                .map(|(&s, &o)| s + 2 * (o - 1))
                .collect::<Vec<_>>(),
        );

        let mut self_padded = Self::zeros(padded_shape);

        unsafe {
            self_padded
                .slice_unchecked_mut(
                    &(0..self.rank())
                        .map(|i| other.shape[i] - 1..other.shape[i] + self.shape[i] - 1)
                        .collect::<Vec<_>>(),
                )
                .set_all(&self)?;
        }

        unsafe {
            Ok(self_padded
                .pool_mt(
                    &|t| {
                        if t.shape() == other.shape {
                            (&other * t.clone_into_tensor()).sum()
                        } else {
                            T::zero()
                        }
                    },
                    &other.shape,
                    &Shape::new(vec![1; self.rank()]),
                )?
                .slice_unchecked(&res_shape.0.iter().map(|&x| 0..x).collect::<Vec<_>>())
                .clone_into_tensor())
        }
    }

    /// Computes the convolution of two tensors across all axes on multiple threads.
    /// This fails if the underlying correlation fails, typically due to rank mismatch or empty tensors.
    pub fn conv_mt(self, other: Tensor<T>) -> Result<Tensor<T>, TensorErrors>
    where
        T: Send + Sync,
    {
        self.corr_mt(other.flip())
    }
}

impl<T: Clone + Add<Output = T> + Mul<Output = T> + Zero + AddAssign> Matrix<T> {
    /// Computes the correlations of two matrices across the columns.
    /// This fails if the number of columns in the two matrices do not match, or if either matrix is empty.
    pub fn corr_cols(self, other: Matrix<T>) -> Result<Matrix<T>, TensorErrors> {
        if self.cols != other.cols {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: other.shape(),
                op: "corr_cols",
            });
        }

        if self.is_empty() || other.is_empty() {
            return Err(TensorErrors::TensorEmpty { op: "Correlation" });
        }

        let mut self_padded = Self::zeros(self.rows + 2 * (other.rows - 1), self.cols);
        self_padded
            .slice_mut(other.rows - 1..other.rows - 1 + self.rows, 0..self.cols)?
            .set_all(&self)?;

        unsafe {
            Ok(self_padded
                .pool_indexed(
                    |(_, c), m| {
                        if m.shape() == shape![other.rows, 1] {
                            (m.clone_into_matrix()
                                * other
                                    .slice_unchecked(0..other.rows, c..c + 1)
                                    .clone_into_matrix())
                            .sum()
                        } else {
                            T::zero()
                        }
                    },
                    (other.rows, 1),
                    (1, 1),
                )?
                .slice_unchecked(0..self.rows + other.rows - 1, 0..self.cols)
                .clone_into_matrix())
        }
    }

    /// Computes the convolution of two matrices across the columns.
    /// This fails if the number of columns in the two matrices do not match, or if either matrix is empty.
    pub fn conv_cols(self, other: Matrix<T>) -> Result<Matrix<T>, TensorErrors> {
        self.corr_cols(other.flip_cols())
    }

    /// Computes the correlations of two matrices across the rows.
    /// This fails if the number of rows in the two matrices do not match, or if either matrix is empty.
    pub fn corr_rows(self, other: Matrix<T>) -> Result<Matrix<T>, TensorErrors> {
        if self.rows != other.rows {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: other.shape(),
                op: "corr_rows",
            });
        }

        if self.is_empty() || other.is_empty() {
            return Err(TensorErrors::TensorEmpty { op: "Correlation" });
        }

        let mut self_padded = Self::zeros(self.rows, self.cols + 2 * (other.cols - 1));
        self_padded
            .slice_mut(0..self.rows, other.cols - 1..other.cols - 1 + self.cols)?
            .set_all(&self)?;

        unsafe {
            Ok(self_padded
                .pool_indexed(
                    |(r, _), m| {
                        if m.shape() == shape![1, other.cols] {
                            (m.clone_into_matrix()
                                * other
                                    .slice_unchecked(r..r + 1, 0..other.cols)
                                    .clone_into_matrix())
                            .sum()
                        } else {
                            T::zero()
                        }
                    },
                    (1, other.cols),
                    (1, 1),
                )?
                .slice_unchecked(0..self.rows, 0..self.cols + other.cols - 1)
                .clone_into_matrix())
        }
    }

    /// Computes the convolution of two matrices across the rows.
    /// This fails if the number of rows in the two matrices do not match, or if either matrix is empty.
    pub fn conv_rows(self, other: Matrix<T>) -> Result<Matrix<T>, TensorErrors> {
        self.corr_rows(other.flip_rows())
    }

    /// Computes the correlation of two matrices across rows and columns.
    /// This fails if either matrix is empty.
    pub fn corr(self, other: Matrix<T>) -> Result<Matrix<T>, TensorErrors> {
        if self.is_empty() || other.is_empty() {
            return Err(TensorErrors::TensorEmpty { op: "Correlation" });
        }

        let (padded_rows, padded_cols) = (
            self.rows + 2 * (other.rows - 1),
            self.cols + 2 * (other.cols - 1),
        );
        let (res_rows, res_cols) = (self.rows + other.rows - 1, self.cols + other.cols - 1);

        let mut self_padded = Self::zeros(padded_rows, padded_cols);

        self_padded
            .slice_mut(
                other.rows - 1..other.rows - 1 + self.rows,
                other.cols - 1..other.cols - 1 + self.cols,
            )?
            .set_all(&self)?;

        unsafe {
            Ok(self_padded
                .pool(
                    |t| {
                        if t.shape() == other.shape() {
                            (&other * t.clone_into_matrix()).sum()
                        } else {
                            T::zero()
                        }
                    },
                    (other.rows, other.cols),
                    (1, 1),
                )?
                .slice_unchecked(0..res_rows, 0..res_cols)
                .clone_into_matrix())
        }
    }

    /// Computes the convolution of two matrices across rows and columns.
    /// This fails if the underlying correlation fails, typically due to empty matrices.
    pub fn conv(self, other: Matrix<T>) -> Result<Matrix<T>, TensorErrors> {
        self.corr(other.flip())
    }

    /// Computes the correlation of two tensors across all axes on multiple threads.
    /// This fails if either matrix is empty.
    pub fn corr_mt(self, other: Matrix<T>) -> Result<Matrix<T>, TensorErrors>
    where
        T: Send + Sync,
    {
        if self.is_empty() || other.is_empty() {
            return Err(TensorErrors::TensorEmpty { op: "Correlation" });
        }

        let (padded_rows, padded_cols) = (
            self.rows + 2 * (other.rows - 1),
            self.cols + 2 * (other.cols - 1),
        );
        let (res_rows, res_cols) = (self.rows + other.rows - 1, self.cols + other.cols - 1);

        let mut self_padded = Self::zeros(padded_rows, padded_cols);

        self_padded
            .slice_mut(
                other.rows - 1..other.rows - 1 + self.rows,
                other.cols - 1..other.cols - 1 + self.cols,
            )?
            .set_all(&self)?;

        unsafe {
            Ok(self_padded
                .pool_mt(
                    &|t| {
                        if t.shape() == other.shape() {
                            (&other * t.clone_into_matrix()).sum()
                        } else {
                            T::zero()
                        }
                    },
                    (other.rows, other.cols),
                    (1, 1),
                )?
                .slice_unchecked(0..res_rows, 0..res_cols)
                .clone_into_matrix())
        }
    }

    /// Computes the convolution of two matrices across rows and columns.
    /// This fails if the underlying correlation fails, typically due to empty matrices.
    pub fn conv_mt(self, other: Matrix<T>) -> Result<Matrix<T>, TensorErrors>
    where
        T: Send + Sync,
    {
        self.corr_mt(other.flip())
    }

    /// Computes the correlations of two matrices across the columns.
    /// This fails if the number of columns in the two matrices do not match, or if either matrix is empty.
    pub fn corr_cols_mt(self, other: Matrix<T>) -> Result<Matrix<T>, TensorErrors>
    where
        T: Send + Sync,
    {
        if self.cols != other.cols {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: other.shape(),
                op: "corr_cols_mt",
            });
        }

        if self.is_empty() || other.is_empty() {
            return Err(TensorErrors::TensorEmpty { op: "Correlation" });
        }

        let mut self_padded = Self::zeros(self.rows + 2 * (other.rows - 1), self.cols);
        self_padded
            .slice_mut(other.rows - 1..other.rows - 1 + self.rows, 0..self.cols)?
            .set_all(&self)?;

        unsafe {
            Ok(self_padded
                .pool_indexed_mt(
                    &|(_, c), m| {
                        if m.shape() == shape![other.rows, 1] {
                            (m.clone_into_matrix()
                                * other
                                    .slice_unchecked(0..other.rows, c..c + 1)
                                    .clone_into_matrix())
                            .sum()
                        } else {
                            T::zero()
                        }
                    },
                    (other.rows, 1),
                    (1, 1),
                )?
                .slice_unchecked(0..self.rows + other.rows - 1, 0..self.cols)
                .clone_into_matrix())
        }
    }

    /// Computes the convolution of two matrices across the columns.
    /// This fails if the number of columns in the two matrices do not match, or if either matrix is empty.
    pub fn conv_cols_mt(self, other: Matrix<T>) -> Result<Matrix<T>, TensorErrors>
    where
        T: Send + Sync, {
        self.corr_cols_mt(other.flip_cols_mt())
    }

    /// Computes the correlations of two matrices across the rows.
    /// This fails if the number of rows in the two matrices do not match, or if either matrix is empty.
    pub fn corr_rows_mt(self, other: Matrix<T>) -> Result<Matrix<T>, TensorErrors>
    where
        T: Send + Sync,
    {
        if self.rows != other.rows {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: other.shape(),
                op: "corr_rows_mt",
            });
        }

        if self.is_empty() || other.is_empty() {
            return Err(TensorErrors::TensorEmpty { op: "Correlation" });
        }

        let mut self_padded = Self::zeros(self.rows, self.cols + 2 * (other.cols - 1));
        self_padded
            .slice_mut(0..self.rows, other.cols - 1..other.cols - 1 + self.cols)?
            .set_all(&self)?;

        unsafe {
            Ok(self_padded
                .pool_indexed_mt(
                    &|(r, _), m| {
                        if m.shape() == shape![1, other.cols] {
                            (m.clone_into_matrix()
                                * other
                                    .slice_unchecked(r..r + 1, 0..other.cols)
                                    .clone_into_matrix())
                            .sum()
                        } else {
                            T::zero()
                        }
                    },
                    (1, other.cols),
                    (1, 1),
                )?
                .slice_unchecked(0..self.rows, 0..self.cols + other.cols - 1)
                .clone_into_matrix())
        }
    }

    /// Computes the convolution of two matrices across the rows.
    /// This fails if the number of rows in the two matrices do not match, or if either matrix is empty.
    pub fn conv_rows_mt(self, other: Matrix<T>) -> Result<Matrix<T>, TensorErrors>
    where
        T: Send + Sync, {
        self.corr_rows_mt(other.flip_rows_mt())
    }
}


#[cfg(test)]
mod corr_conv_tests {
    use float_cmp::approx_eq;
    use num::complex::Complex64;
    use num::ToPrimitive;
    use std::collections::HashSet;
    use tensor_math::definitions::errors::TensorErrors;
    use tensor_math::definitions::matrix::Matrix;
    use tensor_math::definitions::tensor::Tensor;
    use tensor_math::shape;

    #[test]
    fn cannot_corr_conv_empty_tensors() {
        let empty = Tensor::<f64>::new(shape![0, 5], vec![]).unwrap();
        let normal = Tensor::<f64>::new(shape![2, 5], vec![1.0; 10]).unwrap();

        assert!(matches!(
            empty.clone().corr(normal.clone()).unwrap_err(),
            TensorErrors::TensorEmpty { .. }
        ));
        assert!(matches!(
            normal.clone().corr(empty.clone()).unwrap_err(),
            TensorErrors::TensorEmpty { .. }
        ));
        assert!(matches!(
            empty.clone().conv(normal.clone()).unwrap_err(),
            TensorErrors::TensorEmpty { .. }
        ));
        assert!(matches!(
            empty.clone().corr_axes(normal.clone(), &HashSet::from([1])).unwrap_err(),
            TensorErrors::TensorEmpty { .. }
        ));
    }

    #[test]
    fn cannot_corr_conv_mt_empty_tensors() {
        let empty = Tensor::<f64>::new(shape![0, 5], vec![]).unwrap();
        let normal = Tensor::<f64>::new(shape![2, 5], vec![1.0; 10]).unwrap();

        assert!(matches!(
            empty.clone().corr_mt(normal.clone()).unwrap_err(),
            TensorErrors::TensorEmpty { .. }
        ));
        assert!(matches!(
            empty.clone().conv_mt(normal.clone()).unwrap_err(),
            TensorErrors::TensorEmpty { .. }
        ));
        assert!(matches!(
            empty
                .corr_axes_mt(normal.clone(), &HashSet::from([1]))
                .unwrap_err(),
            TensorErrors::TensorEmpty { .. }
        ));
    }

    #[test]
    fn cannot_corr_conv_empty_matrices() {
        let empty = Matrix::<f64>::new(0, 5, vec![]).unwrap();
        let empty2 = Matrix::<f64>::new(5, 0, vec![]).unwrap();
        let normal = Matrix::<f64>::new(2, 5, vec![1.0; 10]).unwrap();
        let normal2 = Matrix::<f64>::new(5, 2, vec![1.0; 10]).unwrap();

        assert!(matches!(
            empty.clone().corr(normal.clone()).unwrap_err(),
            TensorErrors::TensorEmpty { .. }
        ));
        assert!(matches!(
            empty.clone().conv(normal.clone()).unwrap_err(),
            TensorErrors::TensorEmpty { .. }
        ));
        assert!(matches!(
            empty2.clone().corr_rows(normal2.clone()).unwrap_err(),
            TensorErrors::TensorEmpty { .. }
        ));
        assert!(matches!(
            empty.clone().corr_cols(normal.clone()).unwrap_err(),
            TensorErrors::TensorEmpty { .. }
        ));
    }

    #[test]
    fn cannot_corr_conv_mt_empty_matrices() {
        let empty = Matrix::<f64>::new(0, 5, vec![]).unwrap();
        let empty2 = Matrix::<f64>::new(5, 0, vec![]).unwrap();
        let normal = Matrix::<f64>::new(2, 5, vec![1.0; 10]).unwrap();
        let normal2 = Matrix::<f64>::new(5, 2, vec![1.0; 10]).unwrap();

        assert!(matches!(
            empty.clone().corr_mt(normal.clone()).unwrap_err(),
            TensorErrors::TensorEmpty { .. }
        ));
        assert!(matches!(
            empty.clone().conv(normal.clone()).unwrap_err(),
            TensorErrors::TensorEmpty { .. }
        ));
        assert!(matches!(
            empty2.clone().corr_rows_mt(normal2.clone()).unwrap_err(),
            TensorErrors::TensorEmpty { .. }
        ));
        assert!(matches!(
            empty.clone().corr_cols_mt(normal.clone()).unwrap_err(),
            TensorErrors::TensorEmpty { .. }
        ));
    }

    #[test]
    fn cannot_corr_conv_scalar_tensors() {
        let scalar = Tensor::<f64>::new(shape![], vec![1.0]).unwrap();
        let normal = Tensor::<f64>::new(shape![1], vec![1.0]).unwrap();

        assert!(matches!(
            scalar.clone().corr(scalar.clone()).unwrap_err(),
            TensorErrors::RankZero { .. }
        ));
        assert!(matches!(
            scalar.clone().corr_mt(scalar.clone()).unwrap_err(),
            TensorErrors::RankZero { .. }
        ));
        // Rank mismatch is checked first if ranks differ
        assert!(matches!(
            scalar.clone().corr(normal.clone()).unwrap_err(),
            TensorErrors::RanksDoNotMatch(..)
        ));
    }

    #[test]
    fn cannot_corr_conv_mt_scalar_tensors() {
        let scalar = Tensor::<f64>::new(shape![], vec![1.0]).unwrap();
        let normal = Tensor::<f64>::new(shape![1], vec![1.0]).unwrap();

        assert!(matches!(
            scalar.clone().corr_mt(scalar.clone()).unwrap_err(),
            TensorErrors::RankZero { .. }
        ));
        assert!(matches!(
            scalar.clone().conv_mt(scalar.clone()).unwrap_err(),
            TensorErrors::RankZero { .. }
        ));
        assert!(matches!(
            scalar
                .clone()
                .corr_axes_mt(scalar.clone(), &HashSet::from([]))
                .unwrap_err(),
            TensorErrors::RankZero { .. }
        ));
        assert!(matches!(
            scalar.clone().corr_mt(normal.clone()).unwrap_err(),
            TensorErrors::RanksDoNotMatch(..)
        ));
    }

    #[test]
    fn invalid_corr_tensors() {
        let t1 = Tensor::<i32>::new(shape![1, 2, 3, 4], vec![1; 24]).unwrap();
        let t2 = Tensor::<i32>::new(shape![1, 2, 3], vec![1; 6]).unwrap();

        let err = t1.clone().corr(t2.clone()).unwrap_err();
        assert!(matches!(err, TensorErrors::RanksDoNotMatch(_, _)));

        let err = t1.clone().corr_mt(t2.clone()).unwrap_err();
        assert!(matches!(err, TensorErrors::RanksDoNotMatch(_, _)));
    }

    #[test]
    fn corr_tensors() {
        let t1 = Tensor::<i32>::new(shape![2, 3, 4], (0..24).collect()).unwrap();
        let t2 = Tensor::<i32>::new(shape![1, 2, 3], (0..6).collect()).unwrap();

        let res = t1.clone().corr(t2.clone()).unwrap();
        let ans = t1
            .map(|x| Complex64::new(x as f64, 0.0))
            .fft_conv(t2.clone().map(|x| Complex64::new(x as f64, 0.0)).flip_mt())
            .unwrap()
            .map(|c| c.re.round().to_i32().unwrap());

        assert_eq!(res, ans);
    }

    #[test]
    fn corr_mat() {
        let m1 = Matrix::<i32>::new(4, 6, (0..24).collect()).unwrap();
        let m2 = Matrix::<i32>::new(2, 3, (0..6).collect()).unwrap();

        let res = m1.clone().corr(m2.clone()).unwrap();
        let ans = m1
            .map(|x| Complex64::new(x as f64, 0.0))
            .fft_conv(m2.clone().map(|x| Complex64::new(x as f64, 0.0)).flip_mt())
            .map(|c| c.re.round().to_i32().unwrap());

        assert_eq!(res, ans);
    }

    #[test]
    fn corr_mt_tensors() {
        let t1 = Tensor::<f64>::rand(shape![10, 20]).clip(-100.0, 100.0);
        let t2 = Tensor::<f64>::rand(shape![10, 20]).clip(-100.0, 100.0);

        let res_mt = t1.clone().corr_mt(t2.clone()).unwrap();
        let res = t1.clone().corr(t2.clone()).unwrap();

        assert_eq!(res, res_mt);
    }

    #[test]
    fn corr_mt_mat() {
        let m1 = Matrix::<f64>::rand(10, 20).clip(-100.0, 100.0);
        let m2 = Matrix::<f64>::rand(10, 20).clip(-100.0, 100.0);

        let res_mt = m1.clone().corr_mt(m2.clone());
        let res = m1.clone().corr(m2.clone());

        assert_eq!(res, res_mt);
    }

    #[test]
    fn corr_complex_mat() {
        let m1 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);
        let m2 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);
        let m3 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);
        let m4 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);

        let in1 = m1.clone().map(|x| Complex64::new(x, 0.0)) + m2.clone().map(|x| Complex64::new(0.0, x));
        let in2 = m3.clone().map(|x| Complex64::new(x, 0.0)) + m4.clone().map(|x| Complex64::new(0.0, x));

        assert_eq!(
            in1.clone().fft_corr_cols(in2.clone()).unwrap(),
            in1.clone().fft_conv_cols(in2.clone().flip_cols_mt()).unwrap()
        );
        assert_eq!(
            in1.clone().fft_corr_rows(in2.clone()).unwrap(),
            in1.clone().fft_conv_rows(in2.clone().flip_rows_mt()).unwrap()
        );
        assert_eq!(in1.clone().fft_corr(in2.clone()), in1.clone().fft_conv(in2.clone().flip_mt()));
    }

    #[test]
    fn corr_complex_tensor() {
        let t1 = Tensor::<f64>::rand(shape![10, 2, 5]).clip(-100.0, 100.0);
        let t2 = Tensor::<f64>::rand(shape![10, 2, 5]).clip(-100.0, 100.0);
        let t3 = Tensor::<f64>::rand(shape![10, 2, 5]).clip(-100.0, 100.0);
        let t4 = Tensor::<f64>::rand(shape![10, 2, 5]).clip(-100.0, 100.0);

        let mut axes = HashSet::new();
        axes.insert(1);

        let in1 = t1.clone().map(|x| Complex64::new(x, 0.0)) + t2.clone().map(|x| Complex64::new(0.0, x));
        let in2 = t3.clone().map(|x| Complex64::new(x, 0.0)) + t4.clone().map(|x| Complex64::new(0.0, x));

        assert!(approx_eq!(
            Tensor<Complex64>,
            in1.clone().fft_corr_axes(in2.clone(), &axes).unwrap(),
            in1.clone().fft_conv_axes(in2.clone().flip_axes_mt(&axes).unwrap(), &axes)
                .unwrap()
        ));
        assert!(approx_eq!(
            Tensor<Complex64>,
            in1.clone().fft_corr(in2.clone()).unwrap(),
            in1.clone().fft_conv(in2.clone().flip_mt()).unwrap(),
            epsilon = 1e-10
        ));
    }

    #[test]
    fn conv_mat() {
        let m1 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);
        let m2 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);
        let m3 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);
        let m4 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);

        let in1 = m1.clone().map(|x| Complex64::new(x, 0.0)) + m2.clone().map(|x| Complex64::new(0.0, x));
        let in2 = m3.clone().map(|x| Complex64::new(x, 0.0)) + m4.clone().map(|x| Complex64::new(0.0, x));

        assert!(approx_eq!(
            Matrix<Complex64>,
            in1.clone().conv(in2.clone()).unwrap(),
            in1.clone().fft_conv(in2.clone()),
            epsilon = 1e-10
        ));
    }

    #[test]
    fn conv_tensor() {
        let t1 = Tensor::<f64>::rand(shape![10, 2, 5]).clip(-100.0, 100.0);
        let t2 = Tensor::<f64>::rand(shape![10, 2, 5]).clip(-100.0, 100.0);
        let t3 = Tensor::<f64>::rand(shape![10, 2, 5]).clip(-100.0, 100.0);
        let t4 = Tensor::<f64>::rand(shape![10, 2, 5]).clip(-100.0, 100.0);

        let in1 = t1.clone().map(|x| Complex64::new(x, 0.0)) + t2.clone().map(|x| Complex64::new(0.0, x));
        let in2 = t3.clone().map(|x| Complex64::new(x, 0.0)) + t4.clone().map(|x| Complex64::new(0.0, x));

        assert!(approx_eq!(
            Tensor<Complex64>,
            in1.clone().conv(in2.clone()).unwrap(),
            in1.clone().fft_conv(in2.clone()).unwrap(),
            epsilon = 1e-10
        ));
    }

    #[test]
    fn conv_mat_mt() {
        let m1 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);
        let m2 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);

        let res_mt = m1.clone().conv_mt(m2.clone());
        let res = m1.clone().conv(m2.clone());

        assert_eq!(res_mt, res);
    }

    #[test]
    fn conv_tensor_mt() {
        let t1 = Tensor::<f64>::rand(shape![10, 20, 5]).clip(-100.0, 100.0);
        let t2 = Tensor::<f64>::rand(shape![10, 10, 1]).clip(-100.0, 100.0);

        let res_mt = t1.clone().conv_mt(t2.clone()).unwrap();
        let res = t1.clone().conv(t2.clone()).unwrap();

        assert_eq!(res_mt, res);
    }

    #[test]
    fn corr_conv_axes_tensor() {
        let t1 = Tensor::<f64>::rand(shape![10, 2, 5]).clip(-100.0, 100.0);
        let t2 = Tensor::<f64>::rand(shape![10, 2, 5]).clip(-100.0, 100.0);
        let t3 = Tensor::<f64>::rand(shape![10, 2, 5]).clip(-100.0, 100.0);
        let t4 = Tensor::<f64>::rand(shape![10, 2, 5]).clip(-100.0, 100.0);

        let in1 = t1.clone().map(|x| Complex64::new(x, 0.0)) + t2.clone().map(|x| Complex64::new(0.0, x));
        let in2 = t3.clone().map(|x| Complex64::new(x, 0.0)) + t4.clone().map(|x| Complex64::new(0.0, x));

        let mut axes = HashSet::new();
        axes.insert(0);
        axes.insert(2);

        let res = in1.clone().corr_axes(in2.clone(), &axes).unwrap();
        let ans = in1.clone().fft_corr_axes(in2.clone(), &axes).unwrap();

        let res_conv = in1.clone().conv_axes(in2.clone().flip_axes(&axes).unwrap(), &axes).unwrap();
        let ans_conv = in1.clone().fft_conv_axes(in2.clone().flip_axes(&axes).unwrap(), &axes).unwrap();

        assert!(approx_eq!(Tensor<Complex64>, ans, res, epsilon = 1e-10));
        assert!(approx_eq!(Tensor<Complex64>, ans_conv, res_conv, epsilon = 1e-10));
    }

    #[test]
    fn corr_conv_axes_mt_tensor() {
        let t1 = Tensor::<f64>::rand(shape![10, 2, 5]).clip(-100.0, 100.0);
        let t2 = Tensor::<f64>::rand(shape![10, 2, 5]).clip(-100.0, 100.0);
        let t3 = Tensor::<f64>::rand(shape![10, 2, 5]).clip(-100.0, 100.0);
        let t4 = Tensor::<f64>::rand(shape![10, 2, 5]).clip(-100.0, 100.0);

        let in1 = t1.clone().map(|x| Complex64::new(x, 0.0)) + t2.clone().map(|x| Complex64::new(0.0, x));
        let in2 = t3.clone().map(|x| Complex64::new(x, 0.0)) + t4.clone().map(|x| Complex64::new(0.0, x));

        let mut axes = HashSet::new();
        axes.insert(0);
        axes.insert(2);

        let res = in1.clone().corr_axes_mt(in2.clone(), &axes).unwrap();
        let ans = in1.clone().fft_corr_axes(in2.clone(), &axes).unwrap();

        let res_conv = in1.clone().conv_axes(in2.clone().flip_axes(&axes).unwrap(), &axes).unwrap();
        let ans_conv = in1.clone().fft_conv_axes(in2.clone().flip_axes(&axes).unwrap(), &axes).unwrap();

        assert!(approx_eq!(Tensor<Complex64>, ans, res, epsilon = 1e-10));
        assert!(approx_eq!(Tensor<Complex64>, ans_conv, res_conv, epsilon = 1e-10));
    }

    #[test]
    fn corr_conv_axes_mat() {
        let m1 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);
        let m2 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);
        let m3 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);
        let m4 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);

        let in1 = m1.clone().map(|x| Complex64::new(x, 0.0)) + m2.clone().map(|x| Complex64::new(0.0, x));
        let in2 = m3.clone().map(|x| Complex64::new(x, 0.0)) + m4.clone().map(|x| Complex64::new(0.0, x));

        assert!(approx_eq!(
            Matrix<Complex64>,
            in1.clone().fft_corr_rows(in2.clone()).unwrap(),
            in1.clone().corr_rows(in2.clone()).unwrap(),
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Matrix<Complex64>,
            in1.clone().fft_corr_cols(in2.clone()).unwrap(),
            in1.clone().corr_cols(in2.clone()).unwrap(),
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Matrix<Complex64>,
            in1.clone().fft_conv(in2.clone()),
            in1.clone().conv(in2.clone()).unwrap(),
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Matrix<Complex64>,
            in1.clone().fft_conv_rows(in2.clone()).unwrap(),
            in1.clone().conv_rows(in2.clone()).unwrap(),
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Matrix<Complex64>,
            in1.clone().fft_conv_cols(in2.clone()).unwrap(),
            in1.clone().conv_cols(in2.clone()).unwrap(),
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Matrix<Complex64>,
            in1.clone().fft_conv(in2.clone()),
            in1.clone().conv(in2.clone()).unwrap(),
            epsilon = 1e-10
        ));
    }

    #[test]
    fn corr_conv_axes_mt_mat() {
        let m1 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);
        let m2 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);
        let m3 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);
        let m4 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);

        let in1 = m1.clone().map(|x| Complex64::new(x, 0.0)) + m2.clone().map(|x| Complex64::new(0.0, x));
        let in2 = m3.clone().map(|x| Complex64::new(x, 0.0)) + m4.clone().map(|x| Complex64::new(0.0, x));

        assert!(approx_eq!(
            Matrix<Complex64>,
            in1.clone().fft_corr_rows(in2.clone()).unwrap(),
            in1.clone().corr_rows_mt(in2.clone()).unwrap(),
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Matrix<Complex64>,
            in1.clone().fft_corr_cols(in2.clone()).unwrap(),
            in1.clone().corr_cols_mt(in2.clone()).unwrap(),
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Matrix<Complex64>,
            in1.clone().fft_corr(in2.clone()),
            in1.clone().corr_mt(in2.clone()).unwrap(),
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Matrix<Complex64>,
            in1.clone().fft_conv_rows(in2.clone()).unwrap(),
            in1.clone().conv_rows_mt(in2.clone()).unwrap(),
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Matrix<Complex64>,
            in1.clone().fft_conv_cols(in2.clone()).unwrap(),
            in1.clone().conv_cols_mt(in2.clone()).unwrap(),
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Matrix<Complex64>,
            in1.clone().fft_conv(in2.clone()),
            in1.clone().conv_mt(in2.clone()).unwrap(),
            epsilon = 1e-10
        ));
    }

    #[test]
    fn invalid_corr_conv_axes_tensors_ranks_do_not_match() {
        let t1 = Tensor::<f64>::rand(shape![10, 2, 5]).clip(-100.0, 100.0);
        let t2 = Tensor::<f64>::rand(shape![10, 10]).clip(-100.0, 100.0);

        let err = t1.clone().corr_axes(t2.clone(), &HashSet::new()).unwrap_err();
        assert!(matches!(err, TensorErrors::RanksDoNotMatch(..)));

        let err = t1.clone().conv_axes(t2.clone(), &HashSet::new()).unwrap_err();
        assert!(matches!(err, TensorErrors::RanksDoNotMatch(..)));

        let err = t1.clone().corr_axes_mt(t2.clone(), &HashSet::new()).unwrap_err();
        assert!(matches!(err, TensorErrors::RanksDoNotMatch(..)));

        let err = t1.clone().conv_axes_mt(t2.clone(), &HashSet::new()).unwrap_err();
        assert!(matches!(err, TensorErrors::RanksDoNotMatch(..)));
    }

    #[test]
    fn invalid_corr_conv_tensor_incompatible_shapes() {
        let t1 = Tensor::<f64>::rand(shape![10, 10, 10]).clip(-100.0, 100.0);
        let t2 = Tensor::<f64>::rand(shape![10, 20, 10]).clip(-100.0, 100.0);

        let err = t1.clone().corr_axes(t2.clone(), &HashSet::new()).unwrap_err();
        assert!(matches!(err, TensorErrors::IncompatibleShapes { .. }));

        let err = t1.clone().corr_axes_mt(t2.clone(), &HashSet::new()).unwrap_err();
        assert!(matches!(err, TensorErrors::IncompatibleShapes { .. }));

        let err = t1.clone().conv_axes(t2.clone(), &HashSet::new()).unwrap_err();
        assert!(matches!(err, TensorErrors::IncompatibleShapes { .. }));

        let err = t1.clone().conv_axes_mt(t2.clone(), &HashSet::new()).unwrap_err();
        assert!(matches!(err, TensorErrors::IncompatibleShapes { .. }));
    }

    #[test]
    fn invalid_corr_conv_mat_incompatible_shapes() {
        let m1 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);
        let m2 = Matrix::<f64>::rand(10, 20).clip(-100.0, 100.0);
        let m3 = Matrix::<f64>::rand(20, 10).clip(-100.0, 100.0);

        let err = m1.clone().corr_cols(m2.clone()).unwrap_err();
        assert!(matches!(err, TensorErrors::IncompatibleShapes { .. }));

        let err = m1.clone().corr_cols_mt(m2.clone()).unwrap_err();
        assert!(matches!(err, TensorErrors::IncompatibleShapes { .. }));

        let err = m1.clone().corr_rows(m3.clone()).unwrap_err();
        assert!(matches!(err, TensorErrors::IncompatibleShapes { .. }));

        let err = m1.clone().corr_rows_mt(m3.clone()).unwrap_err();
        assert!(matches!(err, TensorErrors::IncompatibleShapes { .. }));

        let err = m1.clone().conv_cols(m2.clone()).unwrap_err();
        assert!(matches!(err, TensorErrors::IncompatibleShapes { .. }));

        let err = m1.clone().conv_cols_mt(m2.clone()).unwrap_err();
        assert!(matches!(err, TensorErrors::IncompatibleShapes { .. }));

        let err = m1.clone().conv_rows(m3.clone()).unwrap_err();
        assert!(matches!(err, TensorErrors::IncompatibleShapes { .. }));

        let err = m1.clone().conv_rows_mt(m3.clone()).unwrap_err();
        assert!(matches!(err, TensorErrors::IncompatibleShapes { .. }));
    }

    #[test]
    fn invalid_corr_conv_axes_out_of_bounds() {
        let t1 = Tensor::<f64>::rand(shape![10, 10, 10]).clip(-100.0, 100.0);
        let t2 = Tensor::<f64>::rand(shape![10, 10, 10]).clip(-100.0, 100.0);
        let mut axes = HashSet::new();

        axes.insert(5);

        let err = t1.clone().corr_axes(t2.clone(), &axes).unwrap_err();
        assert!(matches!(err, TensorErrors::AxisOutOfBounds { .. }));

        let err = t1.clone().corr_axes_mt(t2.clone(), &axes).unwrap_err();
        assert!(matches!(err, TensorErrors::AxisOutOfBounds { .. }));

        let err = t1.clone().conv_axes(t2.clone(), &axes).unwrap_err();
        assert!(matches!(err, TensorErrors::AxisOutOfBounds { .. }));

        let err = t1.clone().conv_axes_mt(t2.clone(), &axes).unwrap_err();
        assert!(matches!(err, TensorErrors::AxisOutOfBounds { .. }));
    }

    #[test]
    fn invalid_because_rank_0() {
        let t1 = Tensor::<f64>::new(shape![], vec![1.0]).unwrap();
        let t2 = Tensor::<f64>::new(shape![], vec![1.0]).unwrap();

        let err = t1.clone().corr_axes(t2.clone(), &HashSet::new()).unwrap_err();
        assert!(matches!(err, TensorErrors::RankZero { .. }));

        let err = t1.clone().corr_axes_mt(t2.clone(), &HashSet::new()).unwrap_err();
        assert!(matches!(err, TensorErrors::RankZero { .. }));

        let err = t1.clone().corr(t2.clone()).unwrap_err();
        assert!(matches!(err, TensorErrors::RankZero { .. }));

        let err = t1.clone().corr_mt(t2.clone()).unwrap_err();
        assert!(matches!(err, TensorErrors::RankZero { .. }));

        let err = t1.clone().conv_axes(t2.clone(), &HashSet::new()).unwrap_err();
        assert!(matches!(err, TensorErrors::RankZero { .. }));

        let err = t1.clone().conv_axes_mt(t2.clone(), &HashSet::new()).unwrap_err();
        assert!(matches!(err, TensorErrors::RankZero { .. }));

        let err = t1.clone().conv(t2.clone()).unwrap_err();
        assert!(matches!(err, TensorErrors::RankZero { .. }));

        let err = t1.clone().conv_mt(t2.clone()).unwrap_err();
        assert!(matches!(err, TensorErrors::RankZero { .. }));
    }
}
