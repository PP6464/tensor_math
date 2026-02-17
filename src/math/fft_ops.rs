use std::collections::HashSet;
use std::sync::{Arc, Mutex, RwLock};
use std::thread::scope;
use num::complex::Complex64;
use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::definitions::shape::Shape;
use crate::definitions::tensor::Tensor;
use crate::definitions::transpose::Transpose;
use crate::utilities::internal_functions::{fft_vec, ifft_vec};

impl Tensor<Complex64> {
    /// Computes an FFT along a single axis.
    pub fn fft_single_axis(&self, axis: usize) -> Result<Tensor<Complex64>, TensorErrors> {
        if axis >= self.rank() {
            return Err(TensorErrors::AxisOutOfBounds {
                axis,
                rank: self.rank(),
            })
        }

        let res_shape = self.shape();
        let res_mutexes = Arc::new(
            RwLock::new(
                Tensor::new(
                    res_shape,
                    (0..res_shape.element_count()).map(|_| Mutex::new(Complex64::ZERO)).collect(),
                )?,
            )
        );

        // The shape of the rest of the tensor, i.e. what tensor we move along
        // when taking slices to FFT and then write into the result
        let mut fft_along_shape_vec = res_shape.0.clone();
        fft_along_shape_vec.remove(axis);

        let fft_along_shape = Shape::new(fft_along_shape_vec)?;
        let axis_len = res_shape[axis];

        scope(|s| {
            for index in fft_along_shape.indices() {
                let res_arc = res_mutexes.clone();

                s.spawn(move || {
                    let res_read = res_arc.read().unwrap();

                    let mut ranges = index.iter().map(|x| *x..x+1).collect::<Vec<_>>();
                    ranges.insert(axis, 0..axis_len);
                    let in_vec = self.slice(&ranges).unwrap().elements;
                    let out_vec = fft_vec(&in_vec);

                    for i in 0..axis_len {
                        let mut idx = index.clone();
                        idx.insert(axis, i);

                        let mut write_lock = res_read[&idx].lock().unwrap();
                        *write_lock = out_vec[i];
                    }
                });
            }
        });

        let res_read = res_mutexes.read().unwrap();
        res_read.iter().map(|x| x.lock().unwrap().clone()).collect::<Tensor<_>>().reshape(&res_shape)
    }

    /// Computes an FFT along a list of axes
    pub fn fft_axes(&self, axes: &HashSet<usize>) -> Result<Tensor<Complex64>, TensorErrors> {
        let mut res = self.clone();

        for &axis in axes {
            res = res.fft_single_axis(axis)?;
        }

        Ok(res)
    }

    /// Computes an FFT along all the axes
    pub fn fft(&self) -> Tensor<Complex64> {
        self.fft_axes(&(0..self.rank()).collect()).unwrap()
    }

    /// Computes an inverse FFT along a single axis.
    pub fn ifft_single_axis(&self, axis: usize) -> Result<Tensor<Complex64>, TensorErrors> {
        if axis >= self.rank() {
            return Err(TensorErrors::AxisOutOfBounds {
                axis,
                rank: self.rank(),
            })
        }

        let res_shape = self.shape();
        let res_mutexes = Arc::new(
            RwLock::new(
                Tensor::new(
                    res_shape,
                    (0..res_shape.element_count()).map(|_| Mutex::new(Complex64::ZERO)).collect(),
                )?,
            )
        );

        // The shape of the rest of the tensor, i.e. what tensor we move along
        // when taking slices to FFT and then write into the result
        let mut fft_along_shape_vec = res_shape.0.clone();
        fft_along_shape_vec.remove(axis);

        let fft_along_shape = Shape::new(fft_along_shape_vec).unwrap();
        let axis_len = res_shape[axis];

        scope(|s| {
            for index in fft_along_shape.indices() {
                let res_arc = res_mutexes.clone();

                s.spawn(move || {
                    let res_read = res_arc.read().unwrap();

                    let mut ranges = index.iter().map(|x| *x..x+1).collect::<Vec<_>>();
                    ranges.insert(axis, 0..axis_len);
                    let in_vec = self.slice(&ranges).unwrap().elements;
                    let out_vec = ifft_vec(&in_vec);

                    for i in 0..axis_len {
                        let mut idx = index.clone();
                        idx.insert(axis, i);

                        let mut write_lock = res_read[&idx].lock().unwrap();
                        *write_lock = out_vec[i];
                    }
                });
            }
        });

        let res_read = res_mutexes.read().unwrap();
        res_read.iter().map(|x| x.lock().unwrap().clone()).collect::<Tensor<_>>().reshape(&res_shape)
    }

    /// Computes an inverse FFT along a list of axes
    pub fn ifft_axes(&self, axes: &HashSet<usize>) -> Result<Tensor<Complex64>, TensorErrors> {
        let mut res = self.clone();

        for &axis in axes {
            res = res.ifft_single_axis(axis)?;
        }

        Ok(res)
    }

    /// Computes an inverse FFT along all the axes
    pub fn ifft(&self) -> Tensor<Complex64> {
        self.ifft_axes(&(0..self.rank()).collect()).unwrap()
    }

    /// Computes the correlation of this and another tensor along a specified list of axes.
    pub fn fft_corr_axes(&self, other: &Tensor<Complex64>, axes: &HashSet<usize>) -> Result<Tensor<Complex64>, TensorErrors> {
        self.fft_conv_axes(&other.flip_axes_mt(axes)?, axes)
    }

    /// Computes the convolution of this and another tensor along a specified list of axes.
    pub fn fft_conv_axes(&self, other: &Tensor<Complex64>, axes: &HashSet<usize>) -> Result<Tensor<Complex64>, TensorErrors> {
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

        let new_shape = Shape::new(new_shape)?;

        let mut self_padded = Self::zeros(&new_shape);
        let mut other_padded = Self::zeros(&new_shape);

        self_padded
            .slice_mut(self.shape.0.iter().map(|x| 0..*x).collect::<Vec<_>>().as_slice())?
            .set_all(&self)?;
        other_padded
            .slice_mut(other.shape.0.iter().map(|x| 0..*x).collect::<Vec<_>>().as_slice())?
            .set_all(&other)?;

        let self_fft = self_padded
            .transpose_mt(&perm)?
            .fft_axes(&(1..=k).map(|i| rank - i).collect())?;

        let other_fft = other_padded
            .transpose_mt(&perm)?
            .fft_axes(&(1..=k).map(|i| rank - i).collect())?;

        let res = self_fft * other_fft;

        res.ifft_axes(&(1..=k).map(|i| rank - i).collect())?.transpose_mt(&inv_perm)
    }

    /// Computes the correlation of this and another tensor
    pub fn fft_corr(&self, other: &Tensor<Complex64>) -> Result<Tensor<Complex64>, TensorErrors> {
        self.fft_conv(&other.flip_mt())
    }

    /// Computes the convolution of this and another tensor
    pub fn fft_conv(&self, other: &Tensor<Complex64>) -> Result<Tensor<Complex64>, TensorErrors> {
        if self.rank() != other.rank() {
            return Err(TensorErrors::RanksDoNotMatch(self.rank(), other.rank()));
        }

        let mut new_shape = self.shape().0.clone();

        for axis in 0..self.rank() {
            new_shape[axis] += other.shape[axis] - 1;
        }

        let new_shape = Shape::new(new_shape).unwrap();

        let mut self_padded = Self::zeros(&new_shape);
        let mut other_padded = Self::zeros(&new_shape);

        self_padded
            .slice_mut(self.shape.0.iter().map(|x| 0..*x).collect::<Vec<_>>().as_slice())?
            .set_all(&self)?;
        other_padded
            .slice_mut(other.shape.0.iter().map(|x| 0..*x).collect::<Vec<_>>().as_slice())?
            .set_all(&other)?;

        let self_fft = self_padded.fft();

        let other_fft = other_padded.fft();

        Ok((self_fft * other_fft).ifft())
    }
}

impl Matrix<Complex64> {
    /// Computes the FFT along the rows
    pub fn fft_rows(&self) -> Matrix<Complex64> {
        let res_mutexes = Arc::new(
            RwLock::new(
                Matrix::new(
                    self.rows, self.cols,
                    (0..self.shape.element_count()).map(|_| Mutex::new(Complex64::ZERO)).collect(),
                ).unwrap(),
            )
        );

        scope(|s| {
            for i in 0..self.rows {
                let res_arc = res_mutexes.clone();

                s.spawn(move || {
                    let res_read = res_arc.read().unwrap();

                    let in_vec = &self.slice(i..i+1, 0..self.cols).unwrap().elements;
                    let out_vec = fft_vec(&in_vec);

                    for j in 0..self.cols {
                        let mut write_lock = res_read[(i, j)].lock().unwrap();

                        *write_lock = out_vec[j].clone();
                    }
                });
            }
        });

        let res_read = res_mutexes.read().unwrap();
        res_read.iter().map(|m| m.lock().unwrap().clone()).collect::<Matrix<_>>().reshape(self.rows, self.cols).unwrap()
    }

    /// Computes the FFT along the columns
    pub fn fft_cols(&self) -> Matrix<Complex64> {
        let res_mutexes = Arc::new(
            RwLock::new(
                Matrix::new(
                    self.rows, self.cols,
                    (0..self.shape.element_count()).map(|_| Mutex::new(Complex64::ZERO)).collect(),
                ).unwrap(),
            )
        );

        scope(|s| {
            for i in 0..self.cols {
                let res_arc = res_mutexes.clone();

                s.spawn(move || {
                    let res_read = res_arc.read().unwrap();

                    let in_vec = &self.slice(0..self.rows, i..i+1).unwrap().elements;
                    let out_vec = fft_vec(&in_vec);

                    for j in 0..self.rows {
                        let mut write_lock = res_read[(j, i)].lock().unwrap();

                        *write_lock = out_vec[j].clone();
                    }
                });
            }
        });

        let res_read = res_mutexes.read().unwrap();
        res_read.iter().map(|m| m.lock().unwrap().clone()).collect::<Matrix<_>>().reshape(self.rows, self.cols).unwrap()
    }

    /// Computes an FFT along the rows and the columns
    pub fn fft(&self) -> Matrix<Complex64> {
        self.fft_rows().fft_cols()
    }

    /// Computes an IFFT along the rows
    pub fn ifft_rows(&self) -> Matrix<Complex64> {
        let res_mutexes = Arc::new(
            RwLock::new(
                Matrix::new(
                    self.rows, self.cols,
                    (0..self.shape.element_count()).map(|_| Mutex::new(Complex64::ZERO)).collect(),
                ).unwrap(),
            )
        );

        scope(|s| {
            for i in 0..self.rows {
                let res_arc = res_mutexes.clone();

                s.spawn(move || {
                    let res_read = res_arc.read().unwrap();

                    let in_vec = &self.slice(i..i+1, 0..self.cols).unwrap().elements;
                    let out_vec = ifft_vec(&in_vec);

                    for j in 0..self.cols {
                        let mut write_lock = res_read[(i, j)].lock().unwrap();

                        *write_lock = out_vec[j].clone();
                    }
                });
            }
        });

        let res_read = res_mutexes.read().unwrap();
        res_read.iter().map(|m| m.lock().unwrap().clone()).collect::<Matrix<_>>().reshape(self.rows, self.cols).unwrap()
    }

    /// Computes an IFFT along the columns
    pub fn ifft_cols(&self) -> Matrix<Complex64> {
        let res_mutexes = Arc::new(
            RwLock::new(
                Matrix::new(
                    self.rows, self.cols,
                    (0..self.shape.element_count()).map(|_| Mutex::new(Complex64::ZERO)).collect(),
                ).unwrap(),
            )
        );

        scope(|s| {
            for i in 0..self.cols {
                let res_arc = res_mutexes.clone();

                s.spawn(move || {
                    let res_read = res_arc.read().unwrap();

                    let in_vec = &self.slice(0..self.rows, i..i+1).unwrap().elements;
                    let out_vec = ifft_vec(&in_vec);

                    for j in 0..self.rows {
                        let mut write_lock = res_read[(j, i)].lock().unwrap();

                        *write_lock = out_vec[j].clone();
                    }
                });
            }
        });

        let res_read = res_mutexes.read().unwrap();
        res_read.iter().map(|m| m.lock().unwrap().clone()).collect::<Matrix<_>>().reshape(self.rows, self.cols).unwrap()
    }

    /// Computes an IFFT along the rows and the columns
    pub fn ifft(self) -> Matrix<Complex64> {
        self.ifft_rows().ifft_cols()
    }

    /// Computes convolution along the columns
    pub fn fft_conv_cols(&self, other: &Matrix<Complex64>) -> Result<Matrix<Complex64>, TensorErrors> {
        if self.cols != other.cols {
            return Err(TensorErrors::ShapesIncompatible);
        }

        let self_padded = self.concat_mt(&Self::zeros(other.rows - 1, self.cols)?, 0)?;
        let other_padded = other.concat_mt(&Self::zeros(self.rows - 1, other.cols)?, 0)?;

        Ok((self_padded.fft_cols() * other_padded.fft_cols()).ifft_cols())
    }

    /// Computes correlation along the columns
    pub fn fft_corr_cols(&self, other: &Matrix<Complex64>) -> Result<Matrix<Complex64>, TensorErrors> {
        self.fft_conv_cols(&other.flip_cols_mt())
    }

    /// Computes convolution along the rows
    pub fn fft_conv_rows(&self, other: &Matrix<Complex64>) -> Result<Matrix<Complex64>, TensorErrors> {
        if self.rows != other.rows {
            return Err(TensorErrors::ShapesIncompatible);
        }

        let self_padded = self.concat_mt(&Self::zeros(self.rows, other.cols - 1)?, 1)?;
        let other_padded = other.concat_mt(&Self::zeros(other.rows, self.cols - 1)?, 1)?;

        Ok((self_padded.fft_rows() * other_padded.fft_rows()).ifft_rows())
    }

    /// Computes correlation along the columns
    pub fn fft_corr_rows(&self, other: &Matrix<Complex64>) -> Result<Matrix<Complex64>, TensorErrors> {
        self.fft_conv_rows(&other.flip_rows_mt())
    }

    /// Computes convolution of two matrices
    pub fn fft_conv(&self, other: &Matrix<Complex64>) -> Matrix<Complex64> {
        let mut self_padded = Self::zeros(self.rows + other.rows - 1, self.cols + other.cols - 1).unwrap();
        let mut other_padded = Self::zeros(self.rows + other.rows - 1, self.cols + other.cols - 1).unwrap();

        self_padded.slice_mut(0..self.rows, 0..self.cols).unwrap().set_all(&self).unwrap();
        other_padded.slice_mut(0..other.rows, 0..other.cols).unwrap().set_all(&other).unwrap();

        (self_padded.fft() * other_padded.fft()).ifft()
    }

    /// Computes correlation of two matrices
    pub fn fft_corr(&self, other: &Matrix<Complex64>) -> Matrix<Complex64> {
        self.fft_conv(&other.flip_mt())
    }
}
