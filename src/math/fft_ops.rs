use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::definitions::shape::Shape;
use crate::definitions::tensor::Tensor;
use crate::definitions::transpose::Transpose;
use crate::utilities::internal_functions::{fft_vec, ifft_vec};
use num::complex::Complex64;
use rayon::iter::ParallelIterator;
use rayon::prelude::ParallelSlice;
use std::collections::HashSet;

impl Tensor<Complex64> {
    /// Computes an FFT along a single axis.
    /// Fails if the tensor is rank zero or the axis is out of bounds.
    pub fn fft_single_axis(&self, axis: usize) -> Result<Tensor<Complex64>, TensorErrors> {
        if self.rank() == 0 {
            return Err(TensorErrors::RankZero {
                op: "fft_single_axis",
            });
        }
        if axis >= self.rank() {
            return Err(TensorErrors::AxisOutOfBounds {
                axis,
                rank: self.rank(),
            });
        }

        let transpose = Transpose::identity(self.rank()).swap_axes(self.rank() - 1, axis)?;
        self.transpose_mt(&transpose)?
            .par_chunks_exact(self.shape[axis])
            .map(fft_vec)
            .flatten()
            .collect::<Tensor<_>>()
            .reshape(&transpose.new_shape(self.shape())?)?
            .transpose_mt(&transpose.inverse())
    }

    /// Computes an FFT along a list of axes
    /// Fails if the tensor is rank zero or if any of the axes are out of bounds.
    pub fn fft_axes(&self, axes: &HashSet<usize>) -> Result<Tensor<Complex64>, TensorErrors> {
        if self.rank() == 0 {
            return Err(TensorErrors::RankZero { op: "fft_axes" });
        }

        let mut res = self.clone();

        for &axis in axes {
            res = res.fft_single_axis(axis)?;
        }

        Ok(res)
    }

    /// Computes an FFT along all the axes.
    /// Fails if the tensor is rank zero.
    pub fn fft(&self) -> Result<Tensor<Complex64>, TensorErrors> {
        if self.rank() == 0 {
            return Err(TensorErrors::RankZero { op: "fft" });
        }
        self.fft_axes(&(0..self.rank()).collect())
    }

    /// Computes an inverse FFT along a single axis.
    /// Fails if the tensor is rank zero or the axis is out of bounds.
    pub fn ifft_single_axis(&self, axis: usize) -> Result<Tensor<Complex64>, TensorErrors> {
        if self.rank() == 0 {
            return Err(TensorErrors::RankZero {
                op: "fft_single_axis",
            });
        }
        if axis >= self.rank() {
            return Err(TensorErrors::AxisOutOfBounds {
                axis,
                rank: self.rank(),
            });
        }

        let transpose = Transpose::identity(self.rank()).swap_axes(self.rank() - 1, axis)?;

        self.transpose_mt(&transpose)?
            .par_chunks_exact(self.shape[axis])
            .map(ifft_vec)
            .flatten()
            .collect::<Tensor<_>>()
            .reshape(&transpose.new_shape(self.shape())?)?
            .transpose_mt(&transpose.inverse())
    }

    /// Computes an inverse FFT along a list of axes
    /// Fails if the tensor is rank zero or if any of the axes are out of bounds.
    pub fn ifft_axes(&self, axes: &HashSet<usize>) -> Result<Tensor<Complex64>, TensorErrors> {
        if self.rank() == 0 {
            return Err(TensorErrors::RankZero { op: "ifft_axes" });
        }

        let mut res = self.clone();

        for &axis in axes {
            res = res.ifft_single_axis(axis)?;
        }

        Ok(res)
    }

    /// Computes an inverse FFT along all the axes.
    /// Fails if the tensor is rank zero.
    pub fn ifft(&self) -> Result<Tensor<Complex64>, TensorErrors> {
        if self.rank() == 0 {
            return Err(TensorErrors::RankZero { op: "ifft" });
        }
        self.ifft_axes(&(0..self.rank()).collect())
    }

    /// Computes the correlation of this and another tensor along a specified list of axes.
    /// Fails if the tensors have different ranks, if non-correlation shapes do not match, or if the tensor is rank zero.
    pub fn fft_corr_axes(
        &self,
        other: &Tensor<Complex64>,
        axes: &HashSet<usize>,
    ) -> Result<Tensor<Complex64>, TensorErrors> {
        if self.rank() == 0 {
            return Err(TensorErrors::RankZero {
                op: "fft_corr_axes",
            });
        }
        self.fft_conv_axes(&other.flip_axes_mt(axes)?, axes)
    }

    /// Computes the convolution of this and another tensor along a specified list of axes.
    /// Fails if the tensors have different ranks, if non-convolution shapes do not match, or if the tensor is rank zero.
    pub fn fft_conv_axes(
        &self,
        other: &Tensor<Complex64>,
        axes: &HashSet<usize>,
    ) -> Result<Tensor<Complex64>, TensorErrors> {
        if self.rank() == 0 {
            return Err(TensorErrors::RankZero {
                op: "fft_conv_axes",
            });
        }
        // Check that the shapes match on all non-convolution axes
        if self.rank() != other.rank() {
            return Err(TensorErrors::RanksDoNotMatch(self.rank(), other.rank()));
        }

        let rank = self.rank();
        let k = axes.len();

        let mut perm_vec = Vec::with_capacity(rank);

        for i in 0..rank {
            if !axes.contains(&i) {
                if self.shape[i] != other.shape[i] {
                    return Err(TensorErrors::ShapesIncompatible);
                }
                perm_vec.push(i);
            }
        }

        perm_vec.extend(axes);
        let perm = Transpose::new(&perm_vec)?;
        let inv_perm = perm.inverse();

        // Pad the tensors as required
        let mut new_shape = self.shape().0.clone();

        for axis in axes {
            new_shape[*axis] += other.shape[*axis] - 1;
        }

        let new_shape = Shape::new(new_shape);

        let mut self_padded = Self::zeros(&new_shape);
        let mut other_padded = Self::zeros(&new_shape);

        self_padded
            .slice_mut(
                self.shape
                    .0
                    .iter()
                    .map(|x| 0..*x)
                    .collect::<Vec<_>>()
                    .as_slice(),
            )?
            .set_all(&self)?;
        other_padded
            .slice_mut(
                other
                    .shape
                    .0
                    .iter()
                    .map(|x| 0..*x)
                    .collect::<Vec<_>>()
                    .as_slice(),
            )?
            .set_all(&other)?;

        let self_fft = self_padded
            .transpose_mt(&perm)?
            .fft_axes(&(1..=k).map(|i| rank - i).collect())?;

        let other_fft = other_padded
            .transpose_mt(&perm)?
            .fft_axes(&(1..=k).map(|i| rank - i).collect())?;

        let res = self_fft * other_fft;

        res.ifft_axes(&(1..=k).map(|i| rank - i).collect())?
            .transpose_mt(&inv_perm)
    }

    /// Computes the correlation of this and another tensor.
    /// Fails if the tensor is rank zero or if ranks do not match.
    pub fn fft_corr(&self, other: &Tensor<Complex64>) -> Result<Tensor<Complex64>, TensorErrors> {
        if self.rank() == 0 {
            return Err(TensorErrors::RankZero { op: "fft_corr" });
        }
        self.fft_conv(&other.flip_mt())
    }

    /// Computes the convolution of this and another tensor.
    /// Fails if the tensor is rank zero or if ranks do not match.
    pub fn fft_conv(&self, other: &Tensor<Complex64>) -> Result<Tensor<Complex64>, TensorErrors> {
        if self.rank() == 0 {
            return Err(TensorErrors::RankZero { op: "fft_conv" });
        }
        if self.rank() != other.rank() {
            return Err(TensorErrors::RanksDoNotMatch(self.rank(), other.rank()));
        }

        let mut new_shape = self.shape().0.clone();

        for axis in 0..self.rank() {
            new_shape[axis] += other.shape[axis] - 1;
        }

        let new_shape = Shape::new(new_shape);

        let mut self_padded = Self::zeros(&new_shape);
        let mut other_padded = Self::zeros(&new_shape);

        self_padded
            .slice_mut(
                self.shape
                    .0
                    .iter()
                    .map(|x| 0..*x)
                    .collect::<Vec<_>>()
                    .as_slice(),
            )?
            .set_all(&self)?;
        other_padded
            .slice_mut(
                other
                    .shape
                    .0
                    .iter()
                    .map(|x| 0..*x)
                    .collect::<Vec<_>>()
                    .as_slice(),
            )?
            .set_all(&other)?;

        let self_fft = self_padded.fft()?;

        let other_fft = other_padded.fft()?;

        Ok((self_fft * other_fft).ifft()?)
    }
}

impl Matrix<Complex64> {
    /// Computes the FFT along the rows
    pub fn fft_rows(&self) -> Matrix<Complex64> {
        self.par_chunks_exact(self.cols)
            .map(fft_vec)
            .flatten()
            .collect::<Matrix<_>>()
            .reshape(self.rows, self.cols)
            .unwrap()
    }

    /// Computes the FFT along the columns
    pub fn fft_cols(&self) -> Matrix<Complex64> {
        self.transpose_mt()
            .par_chunks_exact(self.rows)
            .map(fft_vec)
            .flatten()
            .collect::<Matrix<_>>()
            .reshape(self.cols, self.rows)
            .unwrap()
            .transpose_mt()
    }

    /// Computes an FFT along the rows and the columns
    pub fn fft(&self) -> Matrix<Complex64> {
        self.fft_rows().fft_cols()
    }

    /// Computes an IFFT along the rows
    pub fn ifft_rows(&self) -> Matrix<Complex64> {
        self.par_chunks_exact(self.cols)
            .map(ifft_vec)
            .flatten()
            .collect::<Matrix<_>>()
            .reshape(self.rows, self.cols)
            .unwrap()
    }

    /// Computes an IFFT along the columns
    pub fn ifft_cols(&self) -> Matrix<Complex64> {
        self.transpose_mt()
            .par_chunks_exact(self.rows)
            .map(ifft_vec)
            .flatten()
            .collect::<Matrix<_>>()
            .reshape(self.cols, self.rows)
            .unwrap()
            .transpose_mt()
    }

    /// Computes an IFFT along the rows and the columns
    pub fn ifft(self) -> Matrix<Complex64> {
        self.ifft_rows().ifft_cols()
    }

    /// Computes convolution along the columns
    pub fn fft_conv_cols(
        &self,
        other: &Matrix<Complex64>,
    ) -> Result<Matrix<Complex64>, TensorErrors> {
        if self.cols != other.cols {
            return Err(TensorErrors::ShapesIncompatible);
        }

        let self_padded = self.concat_rows_mt(&Self::zeros(other.rows - 1, self.cols))?;
        let other_padded = other.concat_rows_mt(&Self::zeros(self.rows - 1, other.cols))?;

        Ok((self_padded.fft_cols() * other_padded.fft_cols()).ifft_cols())
    }

    /// Computes correlation along the columns
    pub fn fft_corr_cols(
        &self,
        other: &Matrix<Complex64>,
    ) -> Result<Matrix<Complex64>, TensorErrors> {
        self.fft_conv_cols(&other.flip_cols_mt())
    }

    /// Computes convolution along the rows
    pub fn fft_conv_rows(
        &self,
        other: &Matrix<Complex64>,
    ) -> Result<Matrix<Complex64>, TensorErrors> {
        if self.rows != other.rows {
            return Err(TensorErrors::ShapesIncompatible);
        }

        let self_padded = self.concat_cols_mt(&Self::zeros(self.rows, other.cols - 1))?;
        let other_padded = other.concat_cols_mt(&Self::zeros(other.rows, self.cols - 1))?;

        Ok((self_padded.fft_rows() * other_padded.fft_rows()).ifft_rows())
    }

    /// Computes correlation along the columns
    pub fn fft_corr_rows(
        &self,
        other: &Matrix<Complex64>,
    ) -> Result<Matrix<Complex64>, TensorErrors> {
        self.fft_conv_rows(&other.flip_rows_mt())
    }

    /// Computes convolution of two matrices
    pub fn fft_conv(&self, other: &Matrix<Complex64>) -> Matrix<Complex64> {
        let mut self_padded = Self::zeros(self.rows + other.rows - 1, self.cols + other.cols - 1);
        let mut other_padded = Self::zeros(self.rows + other.rows - 1, self.cols + other.cols - 1);

        self_padded
            .slice_mut(0..self.rows, 0..self.cols)
            .unwrap()
            .set_all(&self)
            .unwrap();
        other_padded
            .slice_mut(0..other.rows, 0..other.cols)
            .unwrap()
            .set_all(&other)
            .unwrap();

        (self_padded.fft() * other_padded.fft()).ifft()
    }

    /// Computes correlation of two matrices
    pub fn fft_corr(&self, other: &Matrix<Complex64>) -> Matrix<Complex64> {
        self.fft_conv(&other.flip_mt())
    }
}

impl Tensor<f64> {
    /// Computes an FFT along a single axis.
    pub fn fft_single_axis(&self, axis: usize) -> Result<Tensor<Complex64>, TensorErrors> {
        self.par_map_refs(|x| Complex64 {
            re: x.clone(),
            im: 0.0,
        })
        .fft_single_axis(axis)
    }

    /// Computes an FFT along a list of axes
    pub fn fft_axes(&self, axes: &HashSet<usize>) -> Result<Tensor<Complex64>, TensorErrors> {
        self.par_map_refs(|x| Complex64 {
            re: x.clone(),
            im: 0.0,
        })
        .fft_axes(axes)
    }

    /// Computes an FFT along all the axes.
    pub fn fft(&self) -> Result<Tensor<Complex64>, TensorErrors> {
        self.par_map_refs(|x| Complex64 {
            re: x.clone(),
            im: 0.0,
        })
        .fft()
    }

    /// Computes an inverse FFT along a single axis.
    pub fn ifft_single_axis(&self, axis: usize) -> Result<Tensor<Complex64>, TensorErrors> {
        self.par_map_refs(|x| Complex64 {
            re: x.clone(),
            im: 0.0,
        })
        .ifft_single_axis(axis)
    }

    /// Computes an inverse FFT along a list of axes
    pub fn ifft_axes(&self, axes: &HashSet<usize>) -> Result<Tensor<Complex64>, TensorErrors> {
        self.par_map_refs(|x| Complex64 {
            re: x.clone(),
            im: 0.0,
        })
        .ifft_axes(axes)
    }

    /// Computes an inverse FFT along all the axes.
    pub fn ifft(&self) -> Result<Tensor<Complex64>, TensorErrors> {
        self.par_map_refs(|x| Complex64 {
            re: x.clone(),
            im: 0.0,
        })
        .ifft()
    }

    /// Computes the correlation of this and another tensor along a specified list of axes.
    pub fn fft_corr_axes(
        &self,
        other: &Tensor<f64>,
        axes: &HashSet<usize>,
    ) -> Result<Tensor<Complex64>, TensorErrors> {
        let self_c = self.par_map_refs(|x| Complex64 {
            re: x.clone(),
            im: 0.0,
        });
        let other_c = other.par_map_refs(|x| Complex64 {
            re: x.clone(),
            im: 0.0,
        });
        self_c.fft_corr_axes(&other_c, axes)
    }

    /// Computes the convolution of this and another tensor along a specified list of axes.
    pub fn fft_conv_axes(
        &self,
        other: &Tensor<f64>,
        axes: &HashSet<usize>,
    ) -> Result<Tensor<Complex64>, TensorErrors> {
        let self_c = self.par_map_refs(|x| Complex64 {
            re: x.clone(),
            im: 0.0,
        });
        let other_c = other.par_map_refs(|x| Complex64 {
            re: x.clone(),
            im: 0.0,
        });
        self_c.fft_conv_axes(&other_c, axes)
    }

    /// Computes the correlation of this and another tensor.
    pub fn fft_corr(&self, other: &Tensor<f64>) -> Result<Tensor<Complex64>, TensorErrors> {
        let self_c = self.par_map_refs(|x| Complex64 {
            re: x.clone(),
            im: 0.0,
        });
        let other_c = other.par_map_refs(|x| Complex64 {
            re: x.clone(),
            im: 0.0,
        });
        self_c.fft_corr(&other_c)
    }

    /// Computes the convolution of this and another tensor.
    pub fn fft_conv(&self, other: &Tensor<f64>) -> Result<Tensor<Complex64>, TensorErrors> {
        let self_c = self.par_map_refs(|x| Complex64 {
            re: x.clone(),
            im: 0.0,
        });
        let other_c = other.par_map_refs(|x| Complex64 {
            re: x.clone(),
            im: 0.0,
        });
        self_c.fft_conv(&other_c)
    }
}

impl Matrix<f64> {
    /// Computes the FFT along the rows
    pub fn fft_rows(&self) -> Matrix<Complex64> {
        self.par_map_refs(|x| Complex64 {
            re: x.clone(),
            im: 0.0,
        })
        .fft_rows()
    }

    /// Computes the FFT along the columns
    pub fn fft_cols(&self) -> Matrix<Complex64> {
        self.par_map_refs(|x| Complex64 {
            re: x.clone(),
            im: 0.0,
        })
        .fft_cols()
    }

    /// Computes an FFT along the rows and the columns
    pub fn fft(&self) -> Matrix<Complex64> {
        self.par_map_refs(|x| Complex64 {
            re: x.clone(),
            im: 0.0,
        })
        .fft()
    }

    /// Computes an IFFT along the rows
    pub fn ifft_rows(&self) -> Matrix<Complex64> {
        self.par_map_refs(|x| Complex64 {
            re: x.clone(),
            im: 0.0,
        })
        .ifft_rows()
    }

    /// Computes an IFFT along the columns
    pub fn ifft_cols(&self) -> Matrix<Complex64> {
        self.par_map_refs(|x| Complex64 {
            re: x.clone(),
            im: 0.0,
        })
        .ifft_cols()
    }

    /// Computes an IFFT along the rows and the columns
    pub fn ifft(self) -> Matrix<Complex64> {
        self.par_map_refs(|x| Complex64 {
            re: x.clone(),
            im: 0.0,
        })
        .ifft()
    }

    /// Computes convolution along the columns
    pub fn fft_conv_cols(&self, other: &Matrix<f64>) -> Result<Matrix<Complex64>, TensorErrors> {
        let self_c = self.par_map_refs(|x| Complex64 {
            re: x.clone(),
            im: 0.0,
        });
        let other_c = other.par_map_refs(|x| Complex64 {
            re: x.clone(),
            im: 0.0,
        });
        self_c.fft_conv_cols(&other_c)
    }

    /// Computes correlation along the columns
    pub fn fft_corr_cols(&self, other: &Matrix<f64>) -> Result<Matrix<Complex64>, TensorErrors> {
        let self_c = self.par_map_refs(|x| Complex64 {
            re: x.clone(),
            im: 0.0,
        });
        let other_c = other.par_map_refs(|x| Complex64 {
            re: x.clone(),
            im: 0.0,
        });
        self_c.fft_corr_cols(&other_c)
    }

    /// Computes convolution along the rows
    pub fn fft_conv_rows(&self, other: &Matrix<f64>) -> Result<Matrix<Complex64>, TensorErrors> {
        let self_c = self.par_map_refs(|x| Complex64 {
            re: x.clone(),
            im: 0.0,
        });
        let other_c = other.par_map_refs(|x| Complex64 {
            re: x.clone(),
            im: 0.0,
        });
        self_c.fft_conv_rows(&other_c)
    }

    /// Computes correlation along the rows
    pub fn fft_corr_rows(&self, other: &Matrix<f64>) -> Result<Matrix<Complex64>, TensorErrors> {
        let self_c = self.par_map_refs(|x| Complex64 {
            re: x.clone(),
            im: 0.0,
        });
        let other_c = other.par_map_refs(|x| Complex64 {
            re: x.clone(),
            im: 0.0,
        });
        self_c.fft_corr_rows(&other_c)
    }

    /// Computes convolution of two matrices
    pub fn fft_conv(&self, other: &Matrix<f64>) -> Matrix<Complex64> {
        let self_c = self.par_map_refs(|x| Complex64 {
            re: x.clone(),
            im: 0.0,
        });
        let other_c = other.par_map_refs(|x| Complex64 {
            re: x.clone(),
            im: 0.0,
        });
        self_c.fft_conv(&other_c)
    }

    /// Computes correlation of two matrices
    pub fn fft_corr(&self, other: &Matrix<f64>) -> Matrix<Complex64> {
        let self_c = self.par_map_refs(|x| Complex64 {
            re: x.clone(),
            im: 0.0,
        });
        let other_c = other.par_map_refs(|x| Complex64 {
            re: x.clone(),
            im: 0.0,
        });
        self_c.fft_corr(&other_c)
    }
}
