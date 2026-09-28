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
}
