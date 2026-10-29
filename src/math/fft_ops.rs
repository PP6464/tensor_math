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
    pub fn fft_single_axis(self, axis: usize) -> Result<Tensor<Complex64>, TensorErrors> {
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
        let shape = self.shape();

        let transpose =
            unsafe { Transpose::identity(self.rank()).swap_axes_unchecked(self.rank() - 1, axis) };
        unsafe {
            self.transpose_unchecked_mt(&transpose)
                .par_chunks(shape[axis])
                .map(fft_vec)
                .flatten()
                .collect::<Tensor<_>>()
                .reshape_unchecked(transpose.new_shape_unchecked(shape))
                .transpose_mt(&transpose)
        }
    }

    /// Computes an FFT along a list of axes
    /// Fails if the tensor is rank zero or if any of the axes are out of bounds.
    pub fn fft_axes(mut self, axes: &HashSet<usize>) -> Result<Tensor<Complex64>, TensorErrors> {
        if self.rank() == 0 {
            return Err(TensorErrors::RankZero {
                op: "fft_single_axis",
            });
        }

        for &axis in axes {
            self = self.fft_single_axis(axis)?;
        }

        Ok(self)
    }

    /// Computes an FFT along all the axes.
    /// Fails if the tensor is rank zero.
    pub fn fft(self) -> Result<Tensor<Complex64>, TensorErrors> {
        let rank = self.rank();
        self.fft_axes(&(0..rank).collect())
    }

    /// Computes an inverse FFT along a single axis.
    /// Fails if the tensor is rank zero or the axis is out of bounds.
    pub fn ifft_single_axis(self, axis: usize) -> Result<Tensor<Complex64>, TensorErrors> {
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

        let shape = self.shape();
        let transpose =
            unsafe { Transpose::identity(self.rank()).swap_axes_unchecked(self.rank() - 1, axis) };

        unsafe {
            self.transpose_unchecked_mt(&transpose)
                .par_chunks(shape[axis])
                .map(ifft_vec)
                .flatten()
                .collect::<Tensor<_>>()
                .reshape_unchecked(transpose.new_shape_unchecked(shape))
                .transpose_mt(&transpose.inverse())
        }
    }

    /// Computes an inverse FFT along a list of axes
    /// Fails if the tensor is rank zero or if any of the axes are out of bounds.
    pub fn ifft_axes(mut self, axes: &HashSet<usize>) -> Result<Tensor<Complex64>, TensorErrors> {
        if self.rank() == 0 {
            return Err(TensorErrors::RankZero { op: "ifft_axes" });
        }

        for &axis in axes {
            self = self.ifft_single_axis(axis)?;
        }

        Ok(self)
    }

    /// Computes an inverse FFT along all the axes.
    /// Fails if the tensor is rank zero.
    pub fn ifft(self) -> Result<Tensor<Complex64>, TensorErrors> {
        let rank = self.rank();
        self.ifft_axes(&(0..rank).collect())
    }

    /// Computes the correlation of this and another tensor along a specified list of axes.
    /// Fails if the tensors have different ranks, if non-correlation shapes do not match, or if the tensor is rank zero.
    pub fn fft_corr_axes(
        self,
        other: Tensor<Complex64>,
        axes: &HashSet<usize>,
    ) -> Result<Tensor<Complex64>, TensorErrors> {
        self.fft_conv_axes(other.flip_axes(axes)?, axes)
    }

    /// Computes the convolution of this and another tensor along a specified list of axes.
    /// Fails if the tensors have different ranks, if non-convolution shapes do not match, or if the tensor is rank zero.
    pub fn fft_conv_axes(
        self,
        other: Tensor<Complex64>,
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
                    return Err(TensorErrors::IncompatibleShapes {
                        shape_1: self.shape(),
                        shape_2: other.shape(),
                        op: "fft_conv_axes",
                    });
                }
                perm_vec.push(i);
            }
        }

        perm_vec.extend(axes);
        let perm = Transpose::new(&perm_vec)?;
        let inv_perm = perm.clone().inverse();

        // Pad the tensors as required
        let mut new_shape = self.shape().0.clone();

        for axis in axes {
            new_shape[*axis] += other.shape[*axis] - 1;
        }

        let new_shape = Shape::new(new_shape);

        let mut self_padded = Self::zeros(new_shape.clone());
        let mut other_padded = Self::zeros(new_shape);

        unsafe {
            self_padded
                .slice_unchecked_mut(
                    self.shape
                        .0
                        .iter()
                        .map(|x| 0..*x)
                        .collect::<Vec<_>>()
                        .as_slice(),
                )
                .set_all(&self)?;
            other_padded
                .slice_unchecked_mut(
                    other
                        .shape
                        .0
                        .iter()
                        .map(|x| 0..*x)
                        .collect::<Vec<_>>()
                        .as_slice(),
                )
                .set_all(&other)?;

            let self_fft = self_padded
                .transpose_unchecked_mt(&perm)
                .fft_axes(&(1..=k).map(|i| rank - i).collect())?;

            let other_fft = other_padded
                .transpose_unchecked_mt(&perm)
                .fft_axes(&(1..=k).map(|i| rank - i).collect())?;

            let res = self_fft * other_fft;

            res.ifft_axes(&(1..=k).map(|i| rank - i).collect())?
                .transpose_mt(&inv_perm)
        }
    }

    /// Computes the correlation of this and another tensor.
    /// Fails if the tensor is rank zero or if ranks do not match.
    pub fn fft_corr(self, other: Tensor<Complex64>) -> Result<Tensor<Complex64>, TensorErrors> {
        if self.rank() == 0 {
            return Err(TensorErrors::RankZero { op: "fft_corr" });
        }
        self.fft_conv(other.flip_mt())
    }

    /// Computes the convolution of this and another tensor.
    /// Fails if the tensor is rank zero or if ranks do not match.
    pub fn fft_conv(self, other: Tensor<Complex64>) -> Result<Tensor<Complex64>, TensorErrors> {
        if self.rank() == 0 {
            return Err(TensorErrors::RankZero { op: "fft_conv" });
        }
        if self.rank() != other.rank() {
            return Err(TensorErrors::RanksDoNotMatch(self.rank(), other.rank()));
        }

        let mut new_shape = self.shape().0.clone();

        for (axis, new_shape_elem) in new_shape.iter_mut().enumerate() {
            *new_shape_elem += other.shape[axis] - 1;
        }

        let new_shape = Shape::new(new_shape);

        let mut self_padded = Self::zeros(new_shape.clone());
        let mut other_padded = Self::zeros(new_shape);

        unsafe {
            self_padded
                .slice_unchecked_mut(
                    self.shape
                        .0
                        .iter()
                        .map(|x| 0..*x)
                        .collect::<Vec<_>>()
                        .as_slice(),
                )
                .set_all(&self)?;
            other_padded
                .slice_unchecked_mut(
                    other
                        .shape
                        .0
                        .iter()
                        .map(|x| 0..*x)
                        .collect::<Vec<_>>()
                        .as_slice(),
                )
                .set_all(&other)?;
        }

        let self_fft = self_padded.fft()?;

        let other_fft = other_padded.fft()?;

        (self_fft * other_fft).ifft()
    }
}

impl Matrix<Complex64> {
    /// Computes the FFT along the rows
    pub fn fft_rows(self) -> Matrix<Complex64> {
        self.par_chunks(self.cols)
            .map(fft_vec)
            .flatten()
            .collect::<Matrix<_>>()
            .reshape(self.rows, self.cols)
            .unwrap()
    }

    /// Computes the FFT along the columns
    pub fn fft_cols(self) -> Matrix<Complex64> {
        let (rows, cols) = (self.rows, self.cols);
        self.transpose_mt()
            .par_chunks(rows)
            .map(fft_vec)
            .flatten()
            .collect::<Matrix<_>>()
            .reshape(cols, rows)
            .unwrap()
            .transpose_mt()
    }

    /// Computes an FFT along the rows and the columns
    pub fn fft(self) -> Matrix<Complex64> {
        self.fft_rows().fft_cols()
    }

    /// Computes an IFFT along the rows
    pub fn ifft_rows(&self) -> Matrix<Complex64> {
        self.par_chunks(self.cols)
            .map(ifft_vec)
            .flatten()
            .collect::<Matrix<_>>()
            .reshape(self.rows, self.cols)
            .unwrap()
    }

    /// Computes an IFFT along the columns
    pub fn ifft_cols(self) -> Matrix<Complex64> {
        let (rows, cols) = (self.rows, self.cols);

        self.transpose_mt()
            .par_chunks(rows)
            .map(ifft_vec)
            .flatten()
            .collect::<Matrix<_>>()
            .reshape(cols, rows)
            .unwrap()
            .transpose_mt()
    }

    /// Computes an IFFT along the rows and the columns
    pub fn ifft(self) -> Matrix<Complex64> {
        self.ifft_rows().ifft_cols()
    }

    /// Computes convolution along the columns
    pub fn fft_conv_cols(
        self,
        other: Matrix<Complex64>,
    ) -> Result<Matrix<Complex64>, TensorErrors> {
        if self.cols != other.cols {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: other.shape(),
                op: "fft_conv_cols",
            });
        }

        let (self_rows, self_cols) = (self.rows, self.cols);
        let (other_rows, other_cols) = (other.rows, other.cols);

        let self_padded = self.concat_along_cols(Self::zeros(other_rows - 1, self_cols))?;
        let other_padded = other.concat_along_cols(Self::zeros(self_rows - 1, other_cols))?;

        Ok((self_padded.fft_cols() * other_padded.fft_cols()).ifft_cols())
    }

    /// Computes correlation along the columns
    pub fn fft_corr_cols(
        self,
        other: Matrix<Complex64>,
    ) -> Result<Matrix<Complex64>, TensorErrors> {
        self.fft_conv_cols(other.flip_cols_mt())
    }

    /// Computes convolution along the rows
    pub fn fft_conv_rows(
        self,
        other: Matrix<Complex64>,
    ) -> Result<Matrix<Complex64>, TensorErrors> {
        if self.rows != other.rows {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: other.shape(),
                op: "fft_conv_rows",
            });
        }

        let (self_rows, self_cols) = (self.rows, self.cols);
        let (other_rows, other_cols) = (other.rows, other.cols);

        let self_padded = self.concat_along_rows(Self::zeros(self_rows, other_cols - 1))?;
        let other_padded = other.concat_along_rows(Self::zeros(other_rows, self_cols - 1))?;

        Ok((self_padded.fft_rows() * other_padded.fft_rows()).ifft_rows())
    }

    /// Computes correlation along the columns
    pub fn fft_corr_rows(
        self,
        other: Matrix<Complex64>,
    ) -> Result<Matrix<Complex64>, TensorErrors> {
        self.fft_conv_rows(other.flip_rows_mt())
    }

    /// Computes convolution of two matrices
    pub fn fft_conv(self, other: Matrix<Complex64>) -> Matrix<Complex64> {
        let mut self_padded = Self::zeros(self.rows + other.rows - 1, self.cols + other.cols - 1);
        let mut other_padded = Self::zeros(self.rows + other.rows - 1, self.cols + other.cols - 1);

        unsafe {
            self_padded
                .slice_unchecked_mut(0..self.rows, 0..self.cols)
                .set_all(&self)
                .unwrap();
            other_padded
                .slice_unchecked_mut(0..other.rows, 0..other.cols)
                .set_all(&other)
                .unwrap();
        }

        (self_padded.fft() * other_padded.fft()).ifft()
    }

    /// Computes correlation of two matrices
    pub fn fft_corr(self, other: Matrix<Complex64>) -> Matrix<Complex64> {
        self.fft_conv(other.flip_mt())
    }
}

impl Tensor<f64> {
    /// Computes an FFT along a single axis.
    pub fn fft_single_axis(&self, axis: usize) -> Result<Tensor<Complex64>, TensorErrors> {
        self.par_map_refs(|x| Complex64 { re: *x, im: 0.0 })
            .fft_single_axis(axis)
    }

    /// Computes an FFT along a list of axes
    pub fn fft_axes(&self, axes: &HashSet<usize>) -> Result<Tensor<Complex64>, TensorErrors> {
        self.par_map_refs(|x| Complex64 { re: *x, im: 0.0 })
            .fft_axes(axes)
    }

    /// Computes an FFT along all the axes.
    pub fn fft(&self) -> Result<Tensor<Complex64>, TensorErrors> {
        self.par_map_refs(|x| Complex64 { re: *x, im: 0.0 }).fft()
    }

    /// Computes an inverse FFT along a single axis.
    pub fn ifft_single_axis(&self, axis: usize) -> Result<Tensor<Complex64>, TensorErrors> {
        self.par_map_refs(|x| Complex64 { re: *x, im: 0.0 })
            .ifft_single_axis(axis)
    }

    /// Computes an inverse FFT along a list of axes
    pub fn ifft_axes(&self, axes: &HashSet<usize>) -> Result<Tensor<Complex64>, TensorErrors> {
        self.par_map_refs(|x| Complex64 { re: *x, im: 0.0 })
            .ifft_axes(axes)
    }

    /// Computes an inverse FFT along all the axes.
    pub fn ifft(&self) -> Result<Tensor<Complex64>, TensorErrors> {
        self.par_map_refs(|x| Complex64 { re: *x, im: 0.0 }).ifft()
    }

    /// Computes the correlation of this and another tensor along a specified list of axes.
    pub fn fft_corr_axes(
        &self,
        other: &Tensor<f64>,
        axes: &HashSet<usize>,
    ) -> Result<Tensor<Complex64>, TensorErrors> {
        let self_c = self.par_map_refs(|x| Complex64 { re: *x, im: 0.0 });
        let other_c = other.par_map_refs(|x| Complex64 { re: *x, im: 0.0 });
        self_c.fft_corr_axes(other_c, axes)
    }

    /// Computes the convolution of this and another tensor along a specified list of axes.
    pub fn fft_conv_axes(
        &self,
        other: &Tensor<f64>,
        axes: &HashSet<usize>,
    ) -> Result<Tensor<Complex64>, TensorErrors> {
        let self_c = self.par_map_refs(|x| Complex64 { re: *x, im: 0.0 });
        let other_c = other.par_map_refs(|x| Complex64 { re: *x, im: 0.0 });
        self_c.fft_conv_axes(other_c, axes)
    }

    /// Computes the correlation of this and another tensor.
    pub fn fft_corr(&self, other: &Tensor<f64>) -> Result<Tensor<Complex64>, TensorErrors> {
        let self_c = self.par_map_refs(|x| Complex64 { re: *x, im: 0.0 });
        let other_c = other.par_map_refs(|x| Complex64 { re: *x, im: 0.0 });
        self_c.fft_corr(other_c)
    }

    /// Computes the convolution of this and another tensor.
    pub fn fft_conv(&self, other: &Tensor<f64>) -> Result<Tensor<Complex64>, TensorErrors> {
        let self_c = self.par_map_refs(|x| Complex64 { re: *x, im: 0.0 });
        let other_c = other.par_map_refs(|x| Complex64 { re: *x, im: 0.0 });
        self_c.fft_conv(other_c)
    }
}

impl Matrix<f64> {
    /// Computes the FFT along the rows
    pub fn fft_rows(&self) -> Matrix<Complex64> {
        self.par_map_refs(|x| Complex64 { re: *x, im: 0.0 })
            .fft_rows()
    }

    /// Computes the FFT along the columns
    pub fn fft_cols(&self) -> Matrix<Complex64> {
        self.par_map_refs(|x| Complex64 { re: *x, im: 0.0 })
            .fft_cols()
    }

    /// Computes an FFT along the rows and the columns
    pub fn fft(&self) -> Matrix<Complex64> {
        self.par_map_refs(|x| Complex64 { re: *x, im: 0.0 }).fft()
    }

    /// Computes an IFFT along the rows
    pub fn ifft_rows(&self) -> Matrix<Complex64> {
        self.par_map_refs(|x| Complex64 { re: *x, im: 0.0 })
            .ifft_rows()
    }

    /// Computes an IFFT along the columns
    pub fn ifft_cols(&self) -> Matrix<Complex64> {
        self.par_map_refs(|x| Complex64 { re: *x, im: 0.0 })
            .ifft_cols()
    }

    /// Computes an IFFT along the rows and the columns
    pub fn ifft(self) -> Matrix<Complex64> {
        self.par_map_refs(|x| Complex64 { re: *x, im: 0.0 }).ifft()
    }

    /// Computes convolution along the columns
    pub fn fft_conv_cols(&self, other: &Matrix<f64>) -> Result<Matrix<Complex64>, TensorErrors> {
        let self_c = self.par_map_refs(|x| Complex64 { re: *x, im: 0.0 });
        let other_c = other.par_map_refs(|x| Complex64 { re: *x, im: 0.0 });
        self_c.fft_conv_cols(other_c)
    }

    /// Computes correlation along the columns
    pub fn fft_corr_cols(&self, other: &Matrix<f64>) -> Result<Matrix<Complex64>, TensorErrors> {
        let self_c = self.par_map_refs(|x| Complex64 { re: *x, im: 0.0 });
        let other_c = other.par_map_refs(|x| Complex64 { re: *x, im: 0.0 });
        self_c.fft_corr_cols(other_c)
    }

    /// Computes convolution along the rows
    pub fn fft_conv_rows(&self, other: &Matrix<f64>) -> Result<Matrix<Complex64>, TensorErrors> {
        let self_c = self.par_map_refs(|x| Complex64 { re: *x, im: 0.0 });
        let other_c = other.par_map_refs(|x| Complex64 { re: *x, im: 0.0 });
        self_c.fft_conv_rows(other_c)
    }

    /// Computes correlation along the rows
    pub fn fft_corr_rows(&self, other: &Matrix<f64>) -> Result<Matrix<Complex64>, TensorErrors> {
        let self_c = self.par_map_refs(|x| Complex64 { re: *x, im: 0.0 });
        let other_c = other.par_map_refs(|x| Complex64 { re: *x, im: 0.0 });
        self_c.fft_corr_rows(other_c)
    }

    /// Computes convolution of two matrices
    pub fn fft_conv(&self, other: &Matrix<f64>) -> Matrix<Complex64> {
        let self_c = self.par_map_refs(|x| Complex64 { re: *x, im: 0.0 });
        let other_c = other.par_map_refs(|x| Complex64 { re: *x, im: 0.0 });
        self_c.fft_conv(other_c)
    }

    /// Computes correlation of two matrices
    pub fn fft_corr(&self, other: &Matrix<f64>) -> Matrix<Complex64> {
        let self_c = self.par_map_refs(|x| Complex64 { re: *x, im: 0.0 });
        let other_c = other.par_map_refs(|x| Complex64 { re: *x, im: 0.0 });
        self_c.fft_corr(other_c)
    }
}

#[cfg(test)]
mod fft_ops_tests {
    use float_cmp::approx_eq;
    use num::complex::Complex64;
    use num::{FromPrimitive, Zero};
    use std::collections::HashSet;
    use tensor_math::definitions::errors::TensorErrors;
    use tensor_math::definitions::matrix::Matrix;
    use tensor_math::definitions::tensor::Tensor;
    use tensor_math::shape;

    #[test]
    fn fft_ops_fail_on_rank_zero_tensors() {
        let t_empty = Tensor::<Complex64>::new(shape![], vec![Complex64::zero()]).unwrap();
        let other = Tensor::<Complex64>::new(shape![], vec![Complex64::zero()]).unwrap();
        let axes = HashSet::from([0]);

        assert!(matches!(
            t_empty.clone().fft_single_axis(0),
            Err(TensorErrors::RankZero { .. })
        ));
        assert!(matches!(
            t_empty.clone().fft_axes(&axes),
            Err(TensorErrors::RankZero { .. })
        ));
        assert!(matches!(
            t_empty.clone().fft(),
            Err(TensorErrors::RankZero { .. })
        ));

        assert!(matches!(
            t_empty.clone().ifft_single_axis(0),
            Err(TensorErrors::RankZero { .. })
        ));
        assert!(matches!(
            t_empty.clone().ifft_axes(&axes),
            Err(TensorErrors::RankZero { .. })
        ));
        assert!(matches!(
            t_empty.clone().ifft(),
            Err(TensorErrors::RankZero { .. })
        ));

        assert!(matches!(
            t_empty.clone().fft_conv(other.clone()),
            Err(TensorErrors::RankZero { .. })
        ));
        assert!(matches!(
            t_empty.clone().fft_conv_axes(other.clone(), &axes),
            Err(TensorErrors::RankZero { .. })
        ));
        assert!(matches!(
            t_empty.clone().fft_corr(other.clone()),
            Err(TensorErrors::RankZero { .. })
        ));
        assert!(matches!(
            t_empty.clone().fft_corr_axes(other.clone(), &axes),
            Err(TensorErrors::AxisOutOfBounds { .. })
        ));
    }

    #[test]
    fn mat_fft() {
        let m1 = Matrix::<Complex64>::new(
            2,
            3,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(3.0, 2.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(1.0, -1.0),
                Complex64::new(2.0, -3.0),
            ],
        )
        .unwrap();
        let ans_cols = Matrix::<Complex64>::new(
            2,
            3,
            vec![
                Complex64::new(1.0, 1.0),
                Complex64::new(3.0, -2.0),
                Complex64::new(5.0, -1.0),
                Complex64::new(1.0, -1.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 5.0),
            ],
        )
        .unwrap();
        let ans_rows = Matrix::<Complex64>::new(
            2,
            3,
            vec![
                Complex64::new(6.0, 1.0),
                Complex64::new(-4.09807621, 0.3660254),
                Complex64::new(1.09807621, -1.3660254),
                Complex64::new(3.0, -3.0),
                Complex64::new(0.23205081, 3.8660254),
                Complex64::new(-3.23205081, 2.1339746),
            ],
        )
        .unwrap();
        let ans_entire = Matrix::<Complex64>::new(
            2,
            3,
            vec![
                Complex64::new(9.0, -2.0),
                Complex64::new(-3.8660254, 4.23205081),
                Complex64::new(-2.1339746, 0.76794919),
                Complex64::new(3.0, 4.0),
                Complex64::new(-4.33012702, -3.5),
                Complex64::new(4.33012702, -3.5),
            ],
        )
        .unwrap();

        let res_rows = m1.clone().fft_rows();
        let res_cols = m1.clone().fft_cols();
        let res_entire = m1.clone().fft();

        for i in 0..2 {
            for j in 0..3 {
                assert!(approx_eq!(
                    f64,
                    res_rows[(i, j)].re,
                    ans_rows[(i, j)].re,
                    epsilon = 1e-5
                ));
                assert!(approx_eq!(
                    f64,
                    res_rows[(i, j)].im,
                    ans_rows[(i, j)].im,
                    epsilon = 1e-5
                ));
                assert!(approx_eq!(
                    f64,
                    res_cols[(i, j)].re,
                    ans_cols[(i, j)].re,
                    epsilon = 1e-5
                ));
                assert!(approx_eq!(
                    f64,
                    res_cols[(i, j)].im,
                    ans_cols[(i, j)].im,
                    epsilon = 1e-5
                ));
                assert!(approx_eq!(
                    f64,
                    res_entire[(i, j)].re,
                    ans_entire[(i, j)].re,
                    epsilon = 1e-5
                ));
                assert!(approx_eq!(
                    f64,
                    res_entire[(i, j)].im,
                    ans_entire[(i, j)].im,
                    epsilon = 1e-5
                ));
            }
        }
    }

    #[test]
    fn mat_ifft() {
        let m1 = Matrix::<Complex64>::new(
            2,
            3,
            vec![
                Complex64::new(9.0, -2.0),
                Complex64::new(-3.8660254, 4.23205081),
                Complex64::new(-2.1339746, 0.76794919),
                Complex64::new(3.0, 4.0),
                Complex64::new(-4.33012702, -3.5),
                Complex64::new(4.33012702, -3.5),
            ],
        )
        .unwrap();

        let ans_cols = Matrix::<Complex64>::new(
            2,
            3,
            vec![
                Complex64::new(6.0, 1.0),
                Complex64::new(-4.09807621, 0.3660254),
                Complex64::new(1.09807621, -1.3660254),
                Complex64::new(3.0, -3.0),
                Complex64::new(0.23205081, 3.8660254),
                Complex64::new(-3.23205081, 2.1339746),
            ],
        )
        .unwrap();
        let ans_rows = Matrix::<Complex64>::new(
            2,
            3,
            vec![
                Complex64::new(1.0, 1.0),
                Complex64::new(3.0, -2.0),
                Complex64::new(5.0, -1.0),
                Complex64::new(1.0, -1.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 5.0),
            ],
        )
        .unwrap();
        let ans_entire = Matrix::<Complex64>::new(
            2,
            3,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(3.0, 2.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(1.0, -1.0),
                Complex64::new(2.0, -3.0),
            ],
        )
        .unwrap();

        let res_rows = m1.clone().ifft_rows();
        let res_cols = m1.clone().ifft_cols();
        let res_entire = m1.clone().ifft();

        for i in 0..2 {
            for j in 0..3 {
                assert!(approx_eq!(
                    f64,
                    res_rows[(i, j)].re,
                    ans_rows[(i, j)].re,
                    epsilon = 1e-5
                ));
                assert!(approx_eq!(
                    f64,
                    res_rows[(i, j)].im,
                    ans_rows[(i, j)].im,
                    epsilon = 1e-5
                ));
                assert!(approx_eq!(
                    f64,
                    res_cols[(i, j)].re,
                    ans_cols[(i, j)].re,
                    epsilon = 1e-5
                ));
                assert!(approx_eq!(
                    f64,
                    res_cols[(i, j)].im,
                    ans_cols[(i, j)].im,
                    epsilon = 1e-5
                ));
                assert!(approx_eq!(
                    f64,
                    res_entire[(i, j)].re,
                    ans_entire[(i, j)].re,
                    epsilon = 1e-5
                ));
                assert!(approx_eq!(
                    f64,
                    res_entire[(i, j)].im,
                    ans_entire[(i, j)].im,
                    epsilon = 1e-5
                ));
            }
        }
    }

    #[test]
    fn tensor_fft() {
        let t1 = Tensor::<Complex64>::new(
            shape![3, 2, 4],
            (0..24)
                .map(|i| Complex64::new(i as f64, i as f64 * if i % 2 == 0 { 1.0 } else { -1.0 }))
                .collect(),
        )
        .unwrap();

        let ans_axes_1_2 = Tensor::<Complex64>::new(
            shape![3, 2, 4],
            vec![
                Complex64::new(28.0, -4.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-4.0, 28.0),
                Complex64::new(-8.0, -8.0),
                Complex64::new(-16.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, -16.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(92.0, -4.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-4.0, 92.0),
                Complex64::new(-8.0, -8.0),
                Complex64::new(-16.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, -16.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(156.0, -4.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-4.0, 156.0),
                Complex64::new(-8.0, -8.0),
                Complex64::new(-16.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, -16.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .unwrap();
        let ans_axis_0 = Tensor::<Complex64>::new(
            shape![3, 2, 4],
            vec![
                Complex64::new(24.0, 24.0),
                Complex64::new(27.0, -27.0),
                Complex64::new(30.0, 30.0),
                Complex64::new(33.0, -33.0),
                Complex64::new(36.0, 36.0),
                Complex64::new(39.0, -39.0),
                Complex64::new(42.0, 42.0),
                Complex64::new(45.0, -45.0),
                Complex64::new(-18.92820323, -5.07179677),
                Complex64::new(-5.07179677, 18.92820323),
                Complex64::new(-18.92820323, -5.07179677),
                Complex64::new(-5.07179677, 18.92820323),
                Complex64::new(-18.92820323, -5.07179677),
                Complex64::new(-5.07179677, 18.92820323),
                Complex64::new(-18.92820323, -5.07179677),
                Complex64::new(-5.07179677, 18.92820323),
                Complex64::new(-5.07179677, -18.92820323),
                Complex64::new(-18.92820323, 5.07179677),
                Complex64::new(-5.07179677, -18.92820323),
                Complex64::new(-18.92820323, 5.07179677),
                Complex64::new(-5.07179677, -18.92820323),
                Complex64::new(-18.92820323, 5.07179677),
                Complex64::new(-5.07179677, -18.92820323),
                Complex64::new(-18.92820323, 5.07179677),
            ],
        )
        .unwrap();
        let ans_entire = Tensor::<Complex64>::new(
            shape![3, 2, 4],
            vec![
                Complex64::new(276.0, -12.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-12.0, 276.0),
                Complex64::new(-24.0, -24.0),
                Complex64::new(-48.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, -48.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-96.0, 55.42562584),
                Complex64::new(0.0, 0.0),
                Complex64::new(-55.42562584, -96.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-96.0, -55.42562584),
                Complex64::new(0.0, 0.0),
                Complex64::new(55.42562584, -96.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .unwrap();

        let res_axes_1_2 = t1.clone().fft_axes(&HashSet::from([1, 2])).unwrap();
        let res_axis_0 = t1.clone().fft_single_axis(0).unwrap();
        let res_entire = t1.clone().fft().unwrap();

        for i in 0..3 {
            for j in 0..2 {
                for k in 0..4 {
                    assert!(approx_eq!(
                        f64,
                        res_axes_1_2[&[i, j, k]].re,
                        ans_axes_1_2[&[i, j, k]].re,
                        epsilon = 1e-5
                    ));
                    assert!(approx_eq!(
                        f64,
                        res_axes_1_2[&[i, j, k]].im,
                        ans_axes_1_2[&[i, j, k]].im,
                        epsilon = 1e-5
                    ));
                    assert!(approx_eq!(
                        f64,
                        res_axis_0[&[i, j, k]].re,
                        ans_axis_0[&[i, j, k]].re,
                        epsilon = 1e-5
                    ));
                    assert!(approx_eq!(
                        f64,
                        res_axis_0[&[i, j, k]].im,
                        ans_axis_0[&[i, j, k]].im,
                        epsilon = 1e-5
                    ));
                    assert!(approx_eq!(
                        f64,
                        res_entire[&[i, j, k]].re,
                        ans_entire[&[i, j, k]].re,
                        epsilon = 1e-5
                    ));
                    assert!(approx_eq!(
                        f64,
                        res_entire[&[i, j, k]].im,
                        ans_entire[&[i, j, k]].im,
                        epsilon = 1e-5
                    ));
                }
            }
        }
    }

    #[test]
    fn tensor_ifft() {
        let ans_entire = Tensor::<Complex64>::new(
            shape![3, 2, 4],
            (0..24)
                .map(|i| Complex64::new(i as f64, i as f64 * if i % 2 == 0 { 1.0 } else { -1.0 }))
                .collect(),
        )
        .unwrap();
        let ans_axis_0 = Tensor::<Complex64>::new(
            shape![3, 2, 4],
            vec![
                Complex64::new(28.0, -4.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-4.0, 28.0),
                Complex64::new(-8.0, -8.0),
                Complex64::new(-16.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, -16.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(92.0, -4.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-4.0, 92.0),
                Complex64::new(-8.0, -8.0),
                Complex64::new(-16.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, -16.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(156.0, -4.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-4.0, 156.0),
                Complex64::new(-8.0, -8.0),
                Complex64::new(-16.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, -16.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .unwrap();
        let ans_axes_1_2 = Tensor::<Complex64>::new(
            shape![3, 2, 4],
            vec![
                Complex64::new(24.0, 24.0),
                Complex64::new(27.0, -27.0),
                Complex64::new(30.0, 30.0),
                Complex64::new(33.0, -33.0),
                Complex64::new(36.0, 36.0),
                Complex64::new(39.0, -39.0),
                Complex64::new(42.0, 42.0),
                Complex64::new(45.0, -45.0),
                Complex64::new(-18.92820323, -5.07179677),
                Complex64::new(-5.07179677, 18.92820323),
                Complex64::new(-18.92820323, -5.07179677),
                Complex64::new(-5.07179677, 18.92820323),
                Complex64::new(-18.92820323, -5.07179677),
                Complex64::new(-5.07179677, 18.92820323),
                Complex64::new(-18.92820323, -5.07179677),
                Complex64::new(-5.07179677, 18.92820323),
                Complex64::new(-5.07179677, -18.92820323),
                Complex64::new(-18.92820323, 5.07179677),
                Complex64::new(-5.07179677, -18.92820323),
                Complex64::new(-18.92820323, 5.07179677),
                Complex64::new(-5.07179677, -18.92820323),
                Complex64::new(-18.92820323, 5.07179677),
                Complex64::new(-5.07179677, -18.92820323),
                Complex64::new(-18.92820323, 5.07179677),
            ],
        )
        .unwrap();
        let t1 = Tensor::<Complex64>::new(
            shape![3, 2, 4],
            vec![
                Complex64::new(276.0, -12.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-12.0, 276.0),
                Complex64::new(-24.0, -24.0),
                Complex64::new(-48.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, -48.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-96.0, 55.42562584),
                Complex64::new(0.0, 0.0),
                Complex64::new(-55.42562584, -96.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-96.0, -55.42562584),
                Complex64::new(0.0, 0.0),
                Complex64::new(55.42562584, -96.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .unwrap();

        let res_axes_1_2 = t1.clone().ifft_axes(&HashSet::from([1, 2])).unwrap();
        let res_axis_0 = t1.clone().ifft_single_axis(0).unwrap();
        let res_entire = t1.clone().ifft().unwrap();

        for i in 0..3 {
            for j in 0..2 {
                for k in 0..4 {
                    assert!(approx_eq!(
                        f64,
                        res_axes_1_2[&[i, j, k]].re,
                        ans_axes_1_2[&[i, j, k]].re,
                        epsilon = 1e-5
                    ));
                    assert!(approx_eq!(
                        f64,
                        res_axes_1_2[&[i, j, k]].im,
                        ans_axes_1_2[&[i, j, k]].im,
                        epsilon = 1e-5
                    ));
                    assert!(approx_eq!(
                        f64,
                        res_axis_0[&[i, j, k]].re,
                        ans_axis_0[&[i, j, k]].re,
                        epsilon = 1e-5
                    ));
                    assert!(approx_eq!(
                        f64,
                        res_axis_0[&[i, j, k]].im,
                        ans_axis_0[&[i, j, k]].im,
                        epsilon = 1e-5
                    ));
                    assert!(approx_eq!(
                        f64,
                        res_entire[&[i, j, k]].re,
                        ans_entire[&[i, j, k]].re,
                        epsilon = 1e-5
                    ));
                    assert!(approx_eq!(
                        f64,
                        res_entire[&[i, j, k]].im,
                        ans_entire[&[i, j, k]].im,
                        epsilon = 1e-5
                    ));
                }
            }
        }
    }

    #[test]
    fn invalid_tensor_fft_corr_conv_axes_different_ranks() {
        let t1 = Tensor::<Complex64>::new(
            shape![1, 2, 3],
            (0..6)
                .map(|x| Complex64 {
                    re: x as f64,
                    im: 0.0,
                })
                .collect(),
        )
        .unwrap();
        let t2 = Tensor::<Complex64>::new(
            shape![2, 3],
            (0..6)
                .map(|x| Complex64 {
                    re: x as f64,
                    im: 0.0,
                })
                .collect(),
        )
        .unwrap();

        let err = t1
            .clone()
            .fft_conv_axes(t2.clone(), &HashSet::new())
            .unwrap_err();
        assert!(matches!(err, TensorErrors::RanksDoNotMatch(_, _)));
        let err = t1.fft_corr_axes(t2.clone(), &HashSet::new()).unwrap_err();
        assert!(matches!(err, TensorErrors::RanksDoNotMatch(_, _)));
    }

    #[test]
    fn invalid_tensor_fft_ifft_axis_out_of_bound() {
        let t1 = Tensor::<f64>::rand(shape![4, 2, 3]);

        let err = t1
            .clone()
            .map(Complex64::from)
            .fft_single_axis(4)
            .unwrap_err();
        assert!(matches!(err, TensorErrors::AxisOutOfBounds { .. }));

        let err = t1
            .clone()
            .map(Complex64::from)
            .ifft_single_axis(4)
            .unwrap_err();
        assert!(matches!(err, TensorErrors::AxisOutOfBounds { .. }));
    }

    #[test]
    fn invalid_tensor_fft_corr_conv_axes_incompatible_shapes() {
        let t1 = Tensor::<Complex64>::new(
            shape![1, 2, 3],
            (0..6)
                .map(|x| Complex64 {
                    re: x as f64,
                    im: 0.0,
                })
                .collect(),
        )
        .unwrap();
        let t2 = Tensor::<Complex64>::new(
            shape![1, 3, 2],
            (0..6)
                .map(|x| Complex64 {
                    re: x as f64,
                    im: 0.0,
                })
                .collect(),
        )
        .unwrap();
        let mut axes = HashSet::new();
        axes.insert(0);

        let err = t1.clone().fft_conv_axes(t2.clone(), &axes).unwrap_err();
        assert!(matches!(err, TensorErrors::IncompatibleShapes { .. }));
        let err = t1.fft_corr_axes(t2.clone(), &axes).unwrap_err();
        assert!(matches!(err, TensorErrors::IncompatibleShapes { .. }));
    }

    #[test]
    fn tensor_fft_corr_conv_axes() {
        let t1 = Tensor::<Complex64>::new(
            shape![4, 2, 3],
            (0..24)
                .map(|x| Complex64 {
                    re: x as f64,
                    im: x as f64,
                })
                .collect(),
        )
        .unwrap();
        let t2 = Tensor::<Complex64>::new(
            shape![1, 2, 3],
            (0..6)
                .map(|x| Complex64 {
                    re: x as f64,
                    im: -x as f64,
                })
                .collect(),
        )
        .unwrap();
        let mut axes = HashSet::new();
        axes.insert(0);
        axes.insert(2);

        let res = t1.clone().fft_conv_axes(t2.clone(), &axes).unwrap();
        let res_corr = t1
            .fft_corr_axes(t2.flip_axes(&axes).unwrap(), &axes)
            .unwrap();

        let ans = Tensor::<Complex64>::new(
            shape![4, 2, 5],
            vec![
                0, 0, 2, 8, 8, 18, 48, 92, 80, 50, 0, 12, 38, 44, 32, 54, 132, 236, 188, 110, 0,
                24, 74, 80, 56, 90, 216, 380, 296, 170, 0, 36, 110, 116, 80, 126, 300, 524, 404,
                230,
            ]
            .iter()
            .map(|x| Complex64::from_f64(*x as f64).unwrap())
            .collect(),
        )
        .unwrap();

        assert!(approx_eq!(
            Tensor<Complex64>,
            res,
            ans.clone(),
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Tensor<Complex64>,
            res_corr,
            ans,
            epsilon = 1e-10
        ));
    }

    #[test]
    fn invalid_tensor_fft_corr_conv_diff_ranks() {
        let t1 = Tensor::<Complex64>::new(
            shape![1, 2, 3],
            (0..6)
                .map(|x| Complex64 {
                    re: x as f64,
                    im: 0.0,
                })
                .collect(),
        )
        .unwrap();
        let t2 = Tensor::<Complex64>::new(
            shape![2, 3],
            (0..6)
                .map(|x| Complex64 {
                    re: x as f64,
                    im: 0.0,
                })
                .collect(),
        )
        .unwrap();

        let err = t1.clone().fft_conv(t2.clone()).unwrap_err();
        assert!(matches!(err, TensorErrors::RanksDoNotMatch(_, _)));
        let err = t1.fft_corr(t2.clone()).unwrap_err();
        assert!(matches!(err, TensorErrors::RanksDoNotMatch(_, _)));
    }

    #[test]
    fn tensor_fft_conv() {
        let t1 = Tensor::<Complex64>::new(
            shape![4, 2, 3],
            (0..24)
                .map(|x| Complex64 {
                    re: x as f64,
                    im: x as f64,
                })
                .collect(),
        )
        .unwrap();
        let t2 = Tensor::<Complex64>::new(
            shape![1, 2, 3],
            (0..6)
                .map(|x| Complex64 {
                    re: x as f64,
                    im: -x as f64,
                })
                .collect(),
        )
        .unwrap();

        let res = t1.fft_conv(t2.clone()).unwrap();

        let ans = Tensor::<Complex64>::new(
            shape![4, 3, 5],
            vec![
                0, 0, 2, 8, 8, 0, 12, 40, 52, 40, 18, 48, 92, 80, 50, 0, 12, 38, 44, 32, 36, 108,
                220, 196, 124, 54, 132, 236, 188, 110, 0, 24, 74, 80, 56, 72, 204, 400, 340, 208,
                90, 216, 380, 296, 170, 0, 36, 110, 116, 80, 108, 300, 580, 484, 292, 126, 300,
                524, 404, 230,
            ]
            .iter()
            .map(|x| Complex64::from_f64(*x as f64).unwrap())
            .collect(),
        )
        .unwrap();

        assert!(approx_eq!(Tensor<Complex64>, res, ans, epsilon = 1e-10));
    }

    #[test]
    fn invalid_fft_mat_corr_conv_cols_different_cols() {
        let m1 = Matrix::<Complex64>::zeros(5, 1);
        let m2 = Matrix::<Complex64>::zeros(3, 2);

        let err = m1.clone().fft_conv_cols(m2.clone()).unwrap_err();
        assert!(matches!(err, TensorErrors::IncompatibleShapes { .. }));
        let err = m1.fft_corr_cols(m2.clone()).unwrap_err();
        assert!(matches!(err, TensorErrors::IncompatibleShapes { .. }));
    }

    #[test]
    fn invalid_fft_mat_corr_conv_rows_different_rows() {
        let m1 = Matrix::<Complex64>::zeros(2, 2);
        let m2 = Matrix::<Complex64>::zeros(1, 1);

        let err = m1.clone().fft_conv_rows(m2.clone()).unwrap_err();
        assert!(matches!(err, TensorErrors::IncompatibleShapes { .. }));
        let err = m1.fft_corr_rows(m2.clone()).unwrap_err();
        assert!(matches!(err, TensorErrors::IncompatibleShapes { .. }));
    }

    #[test]
    fn fft_mat_corr_conv_rows() {
        let m1 = Matrix::<Complex64>::new(
            3,
            2,
            (0..6)
                .map(|x| Complex64 {
                    re: x as f64,
                    im: 0.0,
                })
                .collect(),
        )
        .unwrap();
        let m2 = Matrix::<Complex64>::new(
            3,
            5,
            (0..15)
                .map(|x| Complex64 {
                    re: 0.0,
                    im: x as f64,
                })
                .collect(),
        )
        .unwrap();

        let res = m1.clone().fft_conv_rows(m2.clone()).unwrap();
        let res_corr = m1.fft_corr_rows(m2.flip_rows_mt()).unwrap();

        let ans = Matrix::<Complex64>::new(
            3,
            6,
            vec![
                0, 0, 1, 2, 3, 4, 10, 27, 32, 37, 42, 27, 40, 94, 103, 112, 121, 70,
            ]
            .iter()
            .map(|x| Complex64 {
                re: 0.0,
                im: *x as f64,
            })
            .collect(),
        )
        .unwrap();

        assert!(approx_eq!(
            Matrix<Complex64>,
            ans.clone(),
            res,
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Matrix<Complex64>,
            ans,
            res_corr,
            epsilon = 1e-10
        ));
    }

    #[test]
    fn fft_mat_corr_conv_cols() {
        let m1 = Matrix::<Complex64>::new(
            2,
            3,
            (0..6)
                .map(|x| Complex64 {
                    re: x as f64,
                    im: 0.0,
                })
                .collect(),
        )
        .unwrap();
        let m2 = Matrix::<Complex64>::new(
            5,
            3,
            (0..15)
                .map(|x| Complex64 {
                    re: 0.0,
                    im: x as f64,
                })
                .collect(),
        )
        .unwrap();

        let res = m1.clone().fft_conv_cols(m2.clone()).unwrap();
        let res_corr = m1.fft_corr_cols(m2.flip_cols().clone()).unwrap();

        let ans = Matrix::<Complex64>::new(
            6,
            3,
            vec![
                0, 1, 4, 0, 8, 20, 9, 23, 41, 18, 38, 62, 27, 53, 83, 36, 52, 70,
            ]
            .iter()
            .map(|x| Complex64 {
                re: 0.0,
                im: *x as f64,
            })
            .collect(),
        )
        .unwrap();

        assert!(approx_eq!(
            Matrix<Complex64>,
            ans.clone(),
            res,
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Matrix<Complex64>,
            ans,
            res_corr,
            epsilon = 1e-10
        ));
    }

    #[test]
    fn fft_mat_corr_conv() {
        let m1 = Matrix::<Complex64>::new(
            2,
            3,
            (0..6)
                .map(|x| Complex64 {
                    re: x as f64,
                    im: 0.0,
                })
                .collect(),
        )
        .unwrap();
        let m2 = Matrix::<Complex64>::new(
            5,
            3,
            (0..15)
                .map(|x| Complex64 {
                    re: 0.0,
                    im: x as f64,
                })
                .collect(),
        )
        .unwrap();

        let res = m1.clone().fft_conv(m2.clone());
        let res_corr = m1.fft_corr(m2.flip().clone());

        let ans = Matrix::<Complex64>::new(
            6,
            5,
            vec![
                0, 0, 1, 4, 4, 0, 6, 20, 26, 20, 9, 30, 65, 62, 41, 18, 54, 110, 98, 62, 27, 78,
                155, 134, 83, 36, 87, 154, 121, 70,
            ]
            .iter()
            .map(|x| Complex64 {
                re: 0.0,
                im: *x as f64,
            })
            .collect(),
        )
        .unwrap();

        assert!(approx_eq!(
            Matrix<Complex64>,
            ans.clone(),
            res_corr,
            epsilon = 1e-10
        ));
        assert!(approx_eq!(Matrix<Complex64>, ans, res, epsilon = 1e-10));
    }

    fn as_complex_tensor(values: &Tensor<f64>) -> Tensor<Complex64> {
        values.map_refs(|value| Complex64::new(*value, 0.0))
    }

    fn as_complex_matrix(values: &Matrix<f64>) -> Matrix<Complex64> {
        values.map_refs(|value| Complex64::new(*value, 0.0))
    }

    #[test]
    fn f64_tensor_fft_and_ifft_variants() {
        let values = Tensor::new(shape![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let complex_values = as_complex_tensor(&values);
        let axes = HashSet::from([1]);
        let no_axes = HashSet::new();

        assert!(approx_eq!(
            Tensor<Complex64>,
            values.fft_single_axis(0).unwrap(),
            complex_values.clone().fft_single_axis(0).unwrap(),
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Tensor<Complex64>,
            values.fft_axes(&axes).unwrap(),
            complex_values.clone().fft_axes(&axes).unwrap(),
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Tensor<Complex64>,
            values.fft().unwrap(),
            complex_values.clone().fft().unwrap(),
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Tensor<Complex64>,
            values.ifft_single_axis(1).unwrap(),
            complex_values.clone().ifft_single_axis(1).unwrap(),
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Tensor<Complex64>,
            values.ifft_axes(&axes).unwrap(),
            complex_values.clone().ifft_axes(&axes).unwrap(),
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Tensor<Complex64>,
            values.ifft().unwrap(),
            complex_values.clone().ifft().unwrap(),
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Tensor<Complex64>,
            values.fft_axes(&no_axes).unwrap(),
            complex_values.clone(),
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Tensor<Complex64>,
            values.ifft_axes(&no_axes).unwrap(),
            complex_values,
            epsilon = 1e-10
        ));
    }

    #[test]
    fn f64_tensor_fft_convolution_and_correlation_variants() {
        let lhs = Tensor::new(shape![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let rhs = Tensor::new(shape![2, 2], vec![2.0, 1.0, 0.0, 3.0]).unwrap();
        let lhs_complex = as_complex_tensor(&lhs);
        let rhs_complex = as_complex_tensor(&rhs);
        let axes = HashSet::from([1]);

        assert!(approx_eq!(
            Tensor<Complex64>,
            lhs.fft_conv_axes(&rhs, &axes).unwrap(),
            lhs_complex
                .clone()
                .fft_conv_axes(rhs_complex.clone(), &axes)
                .unwrap(),
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Tensor<Complex64>,
            lhs.fft_corr_axes(&rhs, &axes).unwrap(),
            lhs_complex
                .clone()
                .fft_corr_axes(rhs_complex.clone(), &axes)
                .unwrap(),
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Tensor<Complex64>,
            lhs.fft_conv(&rhs).unwrap(),
            lhs_complex.clone().fft_conv(rhs_complex.clone()).unwrap(),
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Tensor<Complex64>,
            lhs.fft_corr(&rhs).unwrap(),
            lhs_complex.fft_corr(rhs_complex).unwrap(),
            epsilon = 1e-10
        ));
    }

    #[test]
    fn f64_tensor_fft_variants_preserve_errors() {
        let scalar = Tensor::new(shape![], vec![2.0]).unwrap();
        let scalar_other = Tensor::new(shape![], vec![1.0]).unwrap();
        let no_axes = HashSet::new();

        assert!(scalar.fft_single_axis(0).is_err());
        assert!(scalar.fft_axes(&no_axes).is_err());
        assert!(scalar.fft().is_err());
        assert!(scalar.ifft_single_axis(0).is_err());
        assert!(scalar.ifft_axes(&no_axes).is_err());
        assert!(scalar.ifft().is_err());
        assert!(scalar.fft_conv_axes(&scalar_other, &no_axes).is_err());
        assert!(scalar.fft_corr_axes(&scalar_other, &no_axes).is_err());
        assert!(scalar.fft_conv(&scalar_other).is_err());
        assert!(scalar.fft_corr(&scalar_other).is_err());

        let values = Tensor::new(shape![2, 3], vec![1.0; 6]).unwrap();
        assert!(matches!(
            values.fft_single_axis(2),
            Err(TensorErrors::AxisOutOfBounds { .. })
        ));
        assert!(matches!(
            values.ifft_single_axis(2),
            Err(TensorErrors::AxisOutOfBounds { .. })
        ));
        let mismatched_non_convolution_axis = Tensor::new(shape![3, 2], vec![1.0; 6]).unwrap();
        assert!(values
            .fft_conv_axes(&mismatched_non_convolution_axis, &HashSet::from([1]))
            .is_err());
        assert!(values
            .fft_corr_axes(&mismatched_non_convolution_axis, &HashSet::from([1]))
            .is_err());
    }

    #[test]
    fn f64_matrix_fft_and_ifft_variants() {
        let values = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let complex_values = as_complex_matrix(&values);

        assert!(approx_eq!(
            Matrix<Complex64>,
            values.fft_rows(),
            complex_values.clone().fft_rows(),
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Matrix<Complex64>,
            values.fft_cols(),
            complex_values.clone().fft_cols(),
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Matrix<Complex64>,
            values.fft(),
            complex_values.clone().fft(),
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Matrix<Complex64>,
            values.ifft_rows(),
            complex_values.clone().ifft_rows(),
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Matrix<Complex64>,
            values.ifft_cols(),
            complex_values.clone().ifft_cols(),
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Matrix<Complex64>,
            values.ifft(),
            complex_values.ifft(),
            epsilon = 1e-10
        ));
    }

    #[test]
    fn f64_matrix_fft_convolution_and_correlation_variants() {
        let lhs = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let rhs_rows = Matrix::new(2, 2, vec![2.0, 1.0, 0.0, 3.0]).unwrap();
        let rhs_cols = Matrix::new(1, 3, vec![3.0, 1.0, 2.0]).unwrap();
        let rhs_all = Matrix::new(1, 2, vec![2.0, 1.0]).unwrap();
        let lhs_complex = as_complex_matrix(&lhs);

        assert!(approx_eq!(
            Matrix<Complex64>,
            lhs.fft_conv_cols(&rhs_cols).unwrap(),
            lhs_complex
                .clone()
                .fft_conv_cols(as_complex_matrix(&rhs_cols))
                .unwrap(),
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Matrix<Complex64>,
            lhs.fft_corr_cols(&rhs_cols).unwrap(),
            lhs_complex
                .clone()
                .fft_corr_cols(as_complex_matrix(&rhs_cols))
                .unwrap(),
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Matrix<Complex64>,
            lhs.fft_conv_rows(&rhs_rows).unwrap(),
            lhs_complex
                .clone()
                .fft_conv_rows(as_complex_matrix(&rhs_rows))
                .unwrap(),
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Matrix<Complex64>,
            lhs.fft_corr_rows(&rhs_rows).unwrap(),
            lhs_complex
                .clone()
                .fft_corr_rows(as_complex_matrix(&rhs_rows))
                .unwrap(),
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Matrix<Complex64>,
            lhs.fft_conv(&rhs_all),
            lhs_complex.clone().fft_conv(as_complex_matrix(&rhs_all)),
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Matrix<Complex64>,
            lhs.fft_corr(&rhs_all),
            lhs_complex.fft_corr(as_complex_matrix(&rhs_all)),
            epsilon = 1e-10
        ));
    }

    #[test]
    fn f64_matrix_fft_convolution_variants_reject_incompatible_shapes() {
        let values = Matrix::new(2, 3, vec![1.0; 6]).unwrap();

        assert!(values
            .fft_conv_cols(&Matrix::new(2, 2, vec![1.0; 4]).unwrap())
            .is_err());
        assert!(values
            .fft_corr_cols(&Matrix::new(2, 2, vec![1.0; 4]).unwrap())
            .is_err());
        assert!(values
            .fft_conv_rows(&Matrix::new(1, 3, vec![1.0; 3]).unwrap())
            .is_err());
        assert!(values
            .fft_corr_rows(&Matrix::new(1, 3, vec![1.0; 3]).unwrap())
            .is_err());
    }
}
