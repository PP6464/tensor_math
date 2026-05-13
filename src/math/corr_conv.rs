use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::definitions::shape::Shape;
use crate::definitions::tensor::Tensor;
use crate::definitions::transpose::Transpose;
use crate::shape;
use num::Zero;
use std::collections::HashSet;
use std::ops::{Add, Mul};

impl<T: Clone + Add<Output = T> + Mul<Output = T> + Zero> Tensor<T> {
    /// Computes the correlation of two tensors across specified axes
    pub fn corr_axes(
        &self,
        other: &Tensor<T>,
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
                    return Err(TensorErrors::ShapesIncompatible);
                }
            }
        }

        perm_vec.extend(axes.iter());
        let perm = Transpose::new(&perm_vec)?;
        let inv_perm = perm.inverse();

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

        let mut self_padded = Self::zeros(&padded_shape);
        self_padded
            .slice_mut(
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
            )?
            .set_all(&self_perm)?;

        Ok(self_padded
            .pool_indexed(
                |index, t| {
                    if t.shape == kernel_shape {
                        (other_perm
                            .slice(
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
                            .unwrap()
                            * t)
                            .sum()
                    } else {
                        T::zero()
                    }
                },
                &kernel_shape,
                &Shape::new(vec![1; rank]),
                T::zero(),
            )
            .unwrap()
            .slice(&res_shape_vec.iter().map(|&x| 0..x).collect::<Vec<_>>())?
            .transpose(&inv_perm)?)
    }

    /// Computes the correlation of two tensors across all axes
    pub fn corr(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorErrors> {
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

        let mut self_padded = Self::zeros(&padded_shape);

        self_padded
            .slice_mut(
                &(0..self.rank())
                    .map(|i| other.shape[i] - 1..other.shape[i] + self.shape[i] - 1)
                    .collect::<Vec<_>>(),
            )?
            .set_all(self)?;

        Ok(self_padded
            .pool(
                |t| {
                    if t.shape == other.shape {
                        (other * t).sum()
                    } else {
                        T::zero()
                    }
                },
                &other.shape,
                &Shape::new(vec![1; self.rank()]),
                T::zero(),
            )?
            .slice(&res_shape.0.iter().map(|&x| 0..x).collect::<Vec<_>>())?)
    }

    /// Computes the convolution of two matrices across all axes
    pub fn conv(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorErrors> {
        self.corr(&other.flip())
    }
}

impl<T: Clone + Add<Output = T> + Mul<Output = T> + Zero> Matrix<T> {
    /// Computes the correlations of two matrices across the columns
    pub fn corr_cols(&self, other: &Matrix<T>) -> Result<Matrix<T>, TensorErrors> {
        if self.cols != other.cols {
            return Err(TensorErrors::ShapesIncompatible);
        }

        if self.is_empty() || other.is_empty() {
            return Err(TensorErrors::TensorEmpty { op: "Correlation" });
        }

        let mut self_padded = Self::zeros(self.rows + 2 * (other.rows - 1), self.cols);
        self_padded
            .slice_mut(other.rows - 1..other.rows - 1 + self.rows, 0..self.cols)?
            .set_all(&self)?;

        Ok(self_padded
            .pool_indexed(
                |(_, c), m| {
                    if m.shape == shape![other.rows, 1] {
                        (m * other.slice(0..other.rows, c..c + 1).unwrap()).sum()
                    } else {
                        T::zero()
                    }
                },
                (other.rows, 1),
                (1, 1),
                T::zero(),
            )?
            .slice(0..self.rows + other.rows - 1, 0..self.cols)?)
    }

    /// Computes the correlations of two matrices across the rows
    pub fn corr_rows(&self, other: &Matrix<T>) -> Result<Matrix<T>, TensorErrors> {
        if self.rows != other.rows {
            return Err(TensorErrors::ShapesIncompatible);
        }

        if self.is_empty() || other.is_empty() {
            return Err(TensorErrors::TensorEmpty { op: "Correlation" });
        }

        let mut self_padded = Self::zeros(self.rows, self.cols + 2 * (other.cols - 1));
        self_padded
            .slice_mut(0..self.rows, other.cols - 1..other.cols - 1 + self.cols)?
            .set_all(&self)?;

        Ok(self_padded
            .pool_indexed(
                |(r, _), m| {
                    if m.shape == shape![1, other.cols] {
                        (m * other.slice(r..r + 1, 0..other.cols).unwrap()).sum()
                    } else {
                        T::zero()
                    }
                },
                (1, other.cols),
                (1, 1),
                T::zero(),
            )?
            .slice(0..self.rows, 0..self.cols + other.cols - 1)?)
    }

    /// Computes the correlation of two matrices across rows and columns
    pub fn corr(&self, other: &Matrix<T>) -> Result<Matrix<T>, TensorErrors> {
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
            .set_all(self)?;

        Ok(self_padded
            .pool(
                |t| {
                    if t.shape == other.shape {
                        (other * t).sum()
                    } else {
                        T::zero()
                    }
                },
                (other.rows, other.cols),
                (1, 1),
                T::zero(),
            )?
            .slice(0..res_rows, 0..res_cols)?)
    }

    /// Computes the convolution of two matrices across rows and columns
    pub fn conv(&self, other: &Matrix<T>) -> Result<Matrix<T>, TensorErrors> {
        self.corr(&other.flip())
    }
}

impl<T: Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync> Tensor<T> {
    /// Computes the correlation of two tensors across specified axes on multiple threads
    pub fn corr_axes_mt(
        &self,
        other: &Tensor<T>,
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
                    return Err(TensorErrors::ShapesIncompatible);
                }
            }
        }

        perm_vec.extend(axes.iter());
        let perm = Transpose::new(&perm_vec)?;
        let inv_perm = perm.inverse();

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

        let mut self_padded = Self::zeros(&padded_shape);
        self_padded
            .slice_mut(
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
            )?
            .set_all(&self_perm)?;

        Ok(self_padded
            .pool_indexed_mt(
                &|index, t| {
                    if t.shape == kernel_shape {
                        (other_perm
                            .slice(
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
                            .unwrap()
                            * t)
                            .sum()
                    } else {
                        T::zero()
                    }
                },
                &kernel_shape,
                &Shape::new(vec![1; rank]),
                T::zero(),
            )?
            .slice(&res_shape_vec.iter().map(|&x| 0..x).collect::<Vec<_>>())?
            .transpose(&inv_perm)?)
    }

    /// Computes the correlation of two tensors across all axes on multiple threads
    pub fn corr_mt(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorErrors> {
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

        let mut self_padded = Self::zeros(&padded_shape);

        self_padded
            .slice_mut(
                &(0..self.rank())
                    .map(|i| other.shape[i] - 1..other.shape[i] + self.shape[i] - 1)
                    .collect::<Vec<_>>(),
            )?
            .set_all(self)?;

        Ok(self_padded
            .pool_mt(
                &|t| {
                    if t.shape == other.shape {
                        (other * t).sum()
                    } else {
                        T::zero()
                    }
                },
                &other.shape,
                &Shape::new(vec![1; self.rank()]),
                T::zero(),
            )?
            .slice(&res_shape.0.iter().map(|&x| 0..x).collect::<Vec<_>>())?)
    }

    /// Computes the convolution of two tensors across all axes on multiple threads
    pub fn conv_mt(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorErrors> {
        self.corr_mt(&other.flip())
    }
}

impl<T: Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync> Matrix<T> {
    /// Computes the correlation of two tensors across all axes on multiple threads
    pub fn corr_mt(&self, other: &Matrix<T>) -> Result<Matrix<T>, TensorErrors> {
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
            .set_all(self)?;

        Ok(self_padded
            .pool_mt(
                &|t| {
                    if t.shape == other.shape {
                        (other * t).sum()
                    } else {
                        T::zero()
                    }
                },
                (other.rows, other.cols),
                (1, 1),
                T::zero(),
            )?
            .slice(0..res_rows, 0..res_cols)?)
    }

    /// Computes the convolution of two matrices across rows and columns\
    pub fn conv_mt(&self, other: &Matrix<T>) -> Result<Matrix<T>, TensorErrors> {
        self.corr_mt(&other.flip())
    }

    /// Computes the correlations of two matrices across the columns
    pub fn corr_cols_mt(&self, other: &Matrix<T>) -> Result<Matrix<T>, TensorErrors> {
        if self.cols != other.cols {
            return Err(TensorErrors::ShapesIncompatible);
        }

        if self.is_empty() || other.is_empty() {
            return Err(TensorErrors::TensorEmpty { op: "Correlation" });
        }

        let mut self_padded = Self::zeros(self.rows + 2 * (other.rows - 1), self.cols);
        self_padded
            .slice_mut(other.rows - 1..other.rows - 1 + self.rows, 0..self.cols)?
            .set_all(&self)?;

        Ok(self_padded
            .pool_indexed_mt(
                &|(_, c), m| {
                    if m.shape == shape![other.rows, 1] {
                        (m * other.slice(0..other.rows, c..c + 1).unwrap()).sum()
                    } else {
                        T::zero()
                    }
                },
                (other.rows, 1),
                (1, 1),
                T::zero(),
            )?
            .slice(0..self.rows + other.rows - 1, 0..self.cols)?)
    }

    /// Computes the correlations of two matrices across the rows
    pub fn corr_rows_mt(&self, other: &Matrix<T>) -> Result<Matrix<T>, TensorErrors> {
        if self.rows != other.rows {
            return Err(TensorErrors::ShapesIncompatible);
        }

        if self.is_empty() || other.is_empty() {
            return Err(TensorErrors::TensorEmpty { op: "Correlation" });
        }

        let mut self_padded = Self::zeros(self.rows, self.cols + 2 * (other.cols - 1));
        self_padded
            .slice_mut(0..self.rows, other.cols - 1..other.cols - 1 + self.cols)?
            .set_all(&self)?;

        Ok(self_padded
            .pool_indexed_mt(
                &|(r, _), m| {
                    if m.shape == shape![1, other.cols] {
                        (m * other.slice(r..r + 1, 0..other.cols).unwrap()).sum()
                    } else {
                        T::zero()
                    }
                },
                (1, other.cols),
                (1, 1),
                T::zero(),
            )?
            .slice(0..self.rows, 0..self.cols + other.cols - 1)?)
    }
}
