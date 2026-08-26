use crate::definitions::errors::TensorErrors;
use crate::definitions::errors::TensorErrors::IncompatibleShapes;
use crate::definitions::matrix::Matrix;
use crate::definitions::matrix_slice::MatrixSlice;
use crate::definitions::matrix_slice_mut::MatrixSliceMut;
use crate::definitions::traits::IntoMatrix;
use crate::definitions::transpose::Transpose;
use crate::transpose;
use num::{One, ToPrimitive, Zero};
use rand::distr::{Distribution, StandardUniform};
use rand::RngExt;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};
use std::cmp::min;
use std::ops::{Add, Div, Range};
/*
--------------------------------------------
* Matrix utility constructors
--------------------------------------------
*/

impl<T> Matrix<T> {
    /// Creates a tensor from a single value with specified shape.
    pub fn from_value(rows: usize, cols: usize, value: T) -> Self
    where
        T: Clone,
    {
        let elements = vec![value; rows * cols];
        Matrix {
            rows,
            cols,
            elements,
        }
    }

    /// Generate a tensor of the specified shape filled with random values.
    pub fn rand(rows: usize, cols: usize) -> Matrix<T>
    where
        StandardUniform: Distribution<T>,
    {
        let mut elements = Vec::with_capacity(rows * cols);
        let mut buf = elements.spare_capacity_mut();
        let mut rng = rand::rng();

        buf.iter_mut().for_each(|e| {
            e.write(rng.random());
        });

        unsafe {
            elements.set_len(rows * cols);
        }

        Matrix {
            rows,
            cols,
            elements,
        }
    }

    /// Constructs a tensor of the specified shape filled with `T::default()`.
    pub fn from_shape(rows: usize, cols: usize) -> Matrix<T>
    where
        T: Default + Clone,
    {
        let elements = vec![T::default(); rows * cols];
        Matrix {
            rows,
            cols,
            elements,
        }
    }

    /// Returns a matrix of the specified shape filled with `T::zero()`.
    pub fn zeros(rows: usize, cols: usize) -> Matrix<T>
    where
        T: Zero + Clone,
    {
        Matrix::from_value(rows, cols, T::zero())
    }
}

/// Constructs an identity matrix of `T` values of the given size.
pub fn identity<T: Zero + One + Clone>(n: usize) -> Matrix<T> {
    let mut t = Matrix::zeros(n, n);

    for i in 0..n {
        t[&[i, i]] = T::one();
    }

    t
}

/// Constructs an identity matrix of `T` values of the given size.
pub fn eye<T: Zero + One + Clone>(n: usize) -> Matrix<T> {
    let mut t = Matrix::zeros(n, n);

    for i in 0..n {
        t[&[i, i]] = T::one();
    }

    t
}

impl<T: Default> Default for Matrix<T> {
    fn default() -> Self {
        Matrix {
            rows: 1,
            cols: 1,
            elements: vec![T::default()],
        }
    }
}

/*
--------------------------------------------
* Matrix utility functions
--------------------------------------------
*/

impl<T> Matrix<T> {
    /// Reshapes the matrix.
    /// This will fail if `new_rows * new_cols != self.element_count()`.
    pub fn reshape(self, new_rows: usize, new_cols: usize) -> Result<Matrix<T>, TensorErrors> {
        if new_rows * new_cols != self.shape().element_count() {
            return Err(TensorErrors::ShapeSizeDoesNotMatch);
        }

        Ok(Matrix {
            elements: self.elements,
            rows: new_rows,
            cols: new_cols,
        })
    }

    /// Applies the given function over the entire tensor elementwise by consuming the elements.
    pub fn map<F>(self, f: impl FnMut(T) -> F) -> Matrix<F> {
        let rows = self.rows;
        let cols = self.cols;

        Matrix {
            elements: self.elements.into_iter().map(f).collect(),
            rows,
            cols,
        }
    }

    /// Applies the given function over the entire tensor elementwise by consuming the elements.
    pub fn map_refs<F>(&self, f: impl FnMut(&T) -> F) -> Matrix<F> {
        let rows = self.rows;
        let cols = self.cols;

        Matrix {
            elements: self.iter().map(f).collect(),
            rows,
            cols,
        }
    }

    /// Applies the given function over the entire tensor elementwise by consuming the elements.
    pub fn par_map<F: Send>(self, f: impl Fn(T) -> F + Send + Sync) -> Matrix<F>
    where
        T: Send + Sync,
    {
        let rows = self.rows;
        let cols = self.cols;

        Matrix {
            elements: self.elements.into_par_iter().map(f).collect(),
            rows,
            cols,
        }
    }

    /// Applies the given function over the entire tensor elementwise by consuming the elements.
    pub fn par_map_refs<F: Send>(&self, f: impl Fn(&T) -> F + Send + Sync) -> Matrix<F>
    where
        T: Send + Sync,
    {
        let rows = self.rows;
        let cols = self.cols;

        Matrix {
            elements: self.par_iter().map(f).collect(),
            rows,
            cols,
        }
    }

    /// Returns an immutable cloned slice to a specified region of the matrix.
    /// This fails if for either range, `range.start > range.end`,
    /// or if the indices include an out-of-bounds index.
    pub fn slice(
        &self,
        rows_range: Range<usize>,
        cols_range: Range<usize>,
    ) -> Result<MatrixSlice<T>, TensorErrors> {
        if rows_range.end > self.rows {
            return Err(TensorErrors::SliceIndicesOutOfBounds {
                start: rows_range.start,
                end: rows_range.end,
                length: self.rows,
                axis: 0,
            });
        }

        if rows_range.start > rows_range.end {
            return Err(TensorErrors::InvalidInterval {
                max: rows_range.end as f64,
                min: rows_range.start as f64,
            });
        }

        if cols_range.end > self.cols {
            return Err(TensorErrors::SliceIndicesOutOfBounds {
                start: cols_range.start,
                end: cols_range.end,
                length: self.cols,
                axis: 1,
            });
        }

        if cols_range.start > cols_range.end {
            return Err(TensorErrors::InvalidInterval {
                max: cols_range.end as f64,
                min: cols_range.start as f64,
            });
        }

        Ok(MatrixSlice {
            orig: self,
            start: (rows_range.start, cols_range.start),
            end: (rows_range.end, cols_range.end),
        })
    }

    /// Returns a slice covering the entire matrix.
    pub fn as_matrix_slice(&self) -> MatrixSlice<T> {
        MatrixSlice {
            orig: self,
            start: (0, 0),
            end: (self.rows, self.cols),
        }
    }

    /// Returns a mutable slice to a specific region of the matrix.
    /// This fails if for either range, `range.start > range.end`,
    /// or if the indices include an out-of-bounds index.
    pub fn slice_mut(
        &mut self,
        rows_range: Range<usize>,
        cols_range: Range<usize>,
    ) -> Result<MatrixSliceMut<'_, T>, TensorErrors> {
        if rows_range.end > self.rows {
            return Err(TensorErrors::SliceIndicesOutOfBounds {
                start: rows_range.start,
                end: rows_range.end,
                length: self.rows,
                axis: 0,
            });
        }

        if rows_range.start > rows_range.end {
            return Err(TensorErrors::InvalidInterval {
                max: rows_range.end as f64,
                min: rows_range.start as f64,
            });
        }

        if cols_range.end > self.cols {
            return Err(TensorErrors::SliceIndicesOutOfBounds {
                start: cols_range.start,
                end: cols_range.end,
                length: self.cols,
                axis: 1,
            });
        }

        if cols_range.start > cols_range.end {
            return Err(TensorErrors::InvalidInterval {
                max: cols_range.end as f64,
                min: cols_range.start as f64,
            });
        }

        Ok(MatrixSliceMut {
            orig: self,
            start: (rows_range.start, cols_range.start),
            end: (rows_range.end, cols_range.end),
        })
    }

    /// Returns an iterator that is enumerated with matrix indices.
    pub fn enumerated_iter(&self) -> impl Iterator<Item = ((usize, usize), &T)> {
        let cols = self.cols();
        self.iter()
            .enumerate()
            .map(move |(index, elem)| ((index / cols, index % cols), elem))
    }

    /// Returns a parallel iterator that is enumerated with matrix indices.
    pub fn enumerated_par_iter(&self) -> impl ParallelIterator<Item = ((usize, usize), &T)>
    where
        T: Send + Sync,
    {
        let cols = self.cols();
        self.par_iter()
            .enumerate()
            .map(move |(index, elem)| ((index / cols, index % cols), elem))
    }

    /// Returns a mutable iterator that is enumerated with matrix indices.
    pub fn enumerated_iter_mut(&mut self) -> impl Iterator<Item = ((usize, usize), &mut T)> {
        let cols = self.cols();
        self.iter_mut()
            .enumerate()
            .map(move |(index, elem)| ((index / cols, index % cols), elem))
    }

    /// Returns a parallel mutable iterator that is enumerated with matrix indices.
    pub fn enumerated_par_iter_mut(
        &mut self,
    ) -> impl ParallelIterator<Item = ((usize, usize), &mut T)>
    where
        T: Send + Sync,
    {
        let cols = self.cols();
        self.par_iter_mut()
            .enumerate()
            .map(move |(index, elem)| ((index / cols, index % cols), elem))
    }

    /// Sets all the values in the mutable matrix to the given values.
    /// This fails if the shape of the values does not match the matrix's shape.
    pub fn set_all(&mut self, values: &Matrix<T>) -> Result<(), TensorErrors>
    where
        T: Clone,
    {
        if self.rows() != values.rows() || self.cols() != values.cols() {
            return Err(IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: values.shape(),
                op: "set_all",
            });
        }

        for (index, value) in values.enumerated_iter() {
            unsafe { *self.get_unchecked_mut(index) = value.clone() }
        }

        Ok(())
    }

    /// Sets all the values in the mutable matrix to the given values.
    /// This fails if the shape of the values does not match the matrix's shape.
    pub fn set_all_from_slice(&mut self, values: &MatrixSlice<T>) -> Result<(), TensorErrors>
    where
        T: Clone,
    {
        if self.rows() != values.rows() || self.cols() != values.cols() {
            return Err(IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: values.shape(),
                op: "set_all",
            });
        }

        for (index, value) in values.enumerated_iter() {
            unsafe { *self.get_unchecked_mut(index) = value.clone() }
        }

        Ok(())
    }
}

impl<T: Clone> Matrix<T> {
    /// Concatenates two matrices along the columns.
    /// This fails if the number of columns do not match.
    pub fn concat_cols(&self, other: &Matrix<T>) -> Result<Matrix<T>, TensorErrors> {
        let res = self.tensor.concat(&other.tensor, 1)?;
        let res_shape = res.shape.clone();

        Ok(Matrix {
            tensor: res,
            rows: res_shape[0],
            cols: res_shape[1],
        })
    }

    /// Concatenates two matrices along the rows.
    /// This fails if the number of rows do not match.
    pub fn concat_rows(&self, other: &Matrix<T>) -> Result<Matrix<T>, TensorErrors> {
        let res = self.tensor.concat(&other.tensor, 0)?;
        let res_shape = res.shape.clone();

        Ok(Matrix {
            tensor: res,
            rows: res_shape[0],
            cols: res_shape[1],
        })
    }

    /// Transpose a matrix and returns the result.
    pub fn transpose(&self) -> Matrix<T> {
        self.tensor
            .transpose(&transpose![1, 0])
            .unwrap()
            .try_into()
            .unwrap()
    }

    /// Flips the columns of a matrix.
    pub fn flip_rows(&self) -> Matrix<T> {
        let mut res = self.clone();

        for ((row, col), v) in self.enumerated_iter() {
            res[(row, self.cols - col - 1)] = v;
        }

        res
    }

    /// Flips the rows of a matrix.
    pub fn flip_cols(&self) -> Matrix<T> {
        let mut res = self.clone();

        for ((row, col), v) in self.enumerated_iter() {
            res[(self.rows - row - 1, col)] = v
        }

        res
    }

    /// Flips a matrix along both the rows and the columns.
    pub fn flip(&self) -> Matrix<T> {
        let mut res = self.clone();

        for ((row, col), v) in self.enumerated_iter() {
            res[(self.rows - row - 1, self.cols - col - 1)] = v;
        }

        res
    }

    /// Pools a `Matrix<T>` into a `Matrix<O>` using a custom pooling function.
    /// The custom function will take a `Matrix<T>` that corresponds to the slice that the kernel covers.
    /// If the kernel is hanging over the edge of the matrix, then only the bit of the matrix that fits is included.
    /// This will fail if either the kernel or stride shape contains 0, or if the matrix is empty.
    pub fn pool<O: Clone>(
        &self,
        pool_fn: impl Fn(Matrix<T>) -> O,
        kernel_shape: (usize, usize),
        stride_shape: (usize, usize),
        init: O,
    ) -> Result<Matrix<O>, TensorErrors> {
        if kernel_shape.0 == 0 || kernel_shape.1 == 0 || stride_shape.0 == 0 || stride_shape.1 == 0
        {
            return Err(TensorErrors::ShapeContainsZero);
        }

        if self.elements.is_empty() {
            return Err(TensorErrors::TensorEmpty { op: "Pooling" });
        }

        let res_shape = (
            self.rows.div_ceil(stride_shape.0),
            self.cols.div_ceil(stride_shape.1),
        );
        let mut res = Matrix::<O>::from_value(res_shape.0, res_shape.1, init);

        for (pos, val) in res.enumerated_iter_mut() {
            let start_pos = (pos.0 * stride_shape.0, pos.1 * stride_shape.1);
            let end_pos = (
                min(start_pos.0 + kernel_shape.0, self.rows),
                min(start_pos.1 + kernel_shape.1, self.cols),
            );

            let indices = (start_pos.0..end_pos.0, start_pos.1..end_pos.1);
            let value = pool_fn(self.slice(indices.0, indices.1)?.into_matrix());

            *val = value;
        }

        Ok(res)
    }

    /// Pools a `Matrix<T>` into a `Matrix<O>` using a custom pooling function with the index.
    /// The custom function will take a `Matrix<T>` that corresponds to the slice that the kernel covers.
    /// If the kernel is hanging over the edge of the matrix, then only the bit of the matrix that fits is included.
    /// This will fail if either the kernel or stride shape contains 0, or if the matrix is empty.
    pub fn pool_indexed<O: Clone>(
        &self,
        pool_fn: impl Fn((usize, usize), Matrix<T>) -> O,
        kernel_shape: (usize, usize),
        stride_shape: (usize, usize),
        init: O,
    ) -> Result<Matrix<O>, TensorErrors> {
        if kernel_shape.0 == 0 || kernel_shape.1 == 0 || stride_shape.0 == 0 || stride_shape.1 == 0
        {
            return Err(TensorErrors::ShapeContainsZero);
        }

        if self.elements.is_empty() {
            return Err(TensorErrors::TensorEmpty { op: "Pooling" });
        }

        let res_shape = (
            self.rows.div_ceil(stride_shape.0),
            self.cols.div_ceil(stride_shape.1),
        );
        let mut res = Matrix::<O>::from_value(res_shape.0, res_shape.1, init);

        for (pos, val) in res.enumerated_iter_mut() {
            let start_pos = (pos.0 * stride_shape.0, pos.1 * stride_shape.1);
            let end_pos = (
                min(start_pos.0 + kernel_shape.0, self.rows),
                min(start_pos.1 + kernel_shape.1, self.cols),
            );

            let indices = (start_pos.0..end_pos.0, start_pos.1..end_pos.1);
            let value = pool_fn(start_pos, self.slice(indices.0, indices.1)?.into_matrix());

            *val = value;
        }

        Ok(res)
    }
}

impl<T: Clone + Send + Sync> Matrix<T> {
    /// Concatenates two matrices along the columns.
    /// This fails if the number of columns do not match.
    pub fn concat_cols_mt(&self, other: &Matrix<T>) -> Result<Matrix<T>, TensorErrors> {
        let res_tensor = self.tensor.concat_mt(&other.tensor, 1)?;

        res_tensor.try_into()
    }

    /// Concatenates a matrix along the rows.
    /// This fails if the number of rows do not match.
    pub fn concat_rows_mt(&self, other: &Matrix<T>) -> Result<Matrix<T>, TensorErrors> {
        let res_tensor = self.tensor.concat_mt(&other.tensor, 0)?;

        res_tensor.try_into()
    }

    /// Flips the columns of a matrix (using multiple threads).
    pub fn flip_cols_mt(&self) -> Matrix<T> {
        let mut res = self.clone();

        res.enumerated_par_iter_mut()
            .for_each(|((row, col), elem)| {
                *elem = self[(self.rows - row - 1, col)].clone();
            });

        res
    }

    /// Flips the rows of a matrix (using multiple threads).
    pub fn flip_rows_mt(&self) -> Matrix<T> {
        let mut res = self.clone();

        res.enumerated_par_iter_mut()
            .for_each(|((row, col), elem)| {
                *elem = self[(row, self.cols - col - 1)].clone();
            });

        res
    }

    /// Flips the matrix along both the rows and columns (using multiple threads).
    pub fn flip_mt(&self) -> Matrix<T> {
        let mut res = self.clone();

        res.enumerated_par_iter_mut()
            .for_each(|((row, col), elem)| {
                *elem = self[(self.rows - row - 1, self.cols - col - 1)].clone();
            });

        res
    }

    /// Transposes a matrix (using multiple threads).
    pub fn transpose_mt(&self) -> Matrix<T> {
        let mut new_matrix = self.clone().reshape(self.cols, self.rows).unwrap();

        new_matrix
            .enumerated_par_iter_mut()
            .for_each(|((row, col), elem)| {
                *elem = self[(col, row)].clone();
            });

        new_matrix
    }

    /// Pools a `Matrix<T>` into a `Matrix<O>` using a custom pooling function.
    /// The custom function will take a `Matrix<T>` that corresponds to the slice that the kernel covers.
    /// If the kernel is hanging over the edge of the matrix, then only the bit of the matrix that fits is included.
    /// As this is multithreaded, a reference to the pooling function is expected.
    /// This will fail if either the kernel or stride shape contains 0, or if the matrix is empty.
    pub fn pool_mt<O: Clone + Send + Sync>(
        &self,
        pool_fn: &(impl Fn(Matrix<T>) -> O + Sync),
        kernel_shape: (usize, usize),
        stride_shape: (usize, usize),
        init: O,
    ) -> Result<Matrix<O>, TensorErrors> {
        if kernel_shape.0 == 0 || kernel_shape.1 == 0 || stride_shape.0 == 0 || stride_shape.1 == 0
        {
            return Err(TensorErrors::ShapeContainsZero);
        }

        if self.elements.is_empty() {
            return Err(TensorErrors::TensorEmpty { op: "Pooling" });
        }

        let res_shape = (
            self.rows.div_ceil(stride_shape.0),
            self.cols.div_ceil(stride_shape.1),
        );

        let mut result = Matrix::<O>::from_value(res_shape.0, res_shape.1, init);

        result.enumerated_par_iter_mut().for_each(|(index, elem)| {
            let self_pos = (index.0 * stride_shape.0, index.1 * stride_shape.1);
            let self_end_pos = (
                min(self_pos.0 + kernel_shape.0, self.rows),
                min(self_pos.1 + kernel_shape.1, self.cols),
            );

            *elem = pool_fn(
                self.slice(self_pos.0..self_end_pos.0, self_pos.1..self_end_pos.1)
                    .unwrap()
                    .into_matrix(),
            );
        });

        Ok(result)
    }

    /// Pools a `Matrix<T>` into a `Matrix<O>` using a custom pooling function with the index.
    /// The custom function will take a `Matrix<T>` that corresponds to the slice that the kernel covers.
    /// If the kernel is hanging over the edge of the matrix, then only the bit of the matrix that fits is included.
    /// As this is multithreaded, a reference to the pooling function is expected.
    /// This will fail if either the kernel or stride shape contains 0, or if the matrix is empty.
    pub fn pool_indexed_mt<O: Clone + Send + Sync>(
        &self,
        pool_fn: &(impl Fn((usize, usize), Matrix<T>) -> O + Sync),
        kernel_shape: (usize, usize),
        stride_shape: (usize, usize),
        init: O,
    ) -> Result<Matrix<O>, TensorErrors> {
        if kernel_shape.0 == 0 || kernel_shape.1 == 0 || stride_shape.0 == 0 || stride_shape.1 == 0
        {
            return Err(TensorErrors::ShapeContainsZero);
        }

        if self.elements.is_empty() {
            return Err(TensorErrors::TensorEmpty { op: "Pooling" });
        }

        let res_shape = (
            self.rows.div_ceil(stride_shape.0),
            self.cols.div_ceil(stride_shape.1),
        );

        let mut result = Matrix::<O>::from_value(res_shape.0, res_shape.1, init);

        result.enumerated_par_iter_mut().for_each(|(index, elem)| {
            let self_pos = (index.0 * stride_shape.0, index.1 * stride_shape.1);
            let self_end_pos = (
                min(self_pos.0 + kernel_shape.0, self.rows),
                min(self_pos.1 + kernel_shape.1, self.cols),
            );

            *elem = pool_fn(
                index,
                self.slice(self_pos.0..self_end_pos.0, self_pos.1..self_end_pos.1)
                    .unwrap()
                    .into_matrix(),
            );
        });

        Ok(result)
    }
}

impl<T: PartialOrd + Clone> Matrix<T> {
    /// Clips the values in the matrix between `[min, max]`
    pub fn clip(self, min: T, max: T) -> Matrix<T> {
        self.map(|val| {
            if val <= min {
                min.clone()
            } else if val >= max {
                max.clone()
            } else {
                val
            }
        })
    }
}

impl<T: PartialOrd + Clone + Send + Sync> Matrix<T> {
    /// Clips the values in the matrix between `[min, max]`
    pub fn par_clip(self, min: T, max: T) -> Matrix<T> {
        self.par_map(|val| {
            if val <= min {
                min.clone()
            } else if val >= max {
                max.clone()
            } else {
                val
            }
        })
    }
}

/// Default pooling function to sum the values.
pub fn pool_sum_mat<T: Add<Output = T> + Clone>(m: Matrix<T>) -> T {
    m.iter().cloned().reduce(T::add).unwrap()
}

/// Default pooling function to find the minimum.
pub fn pool_min_mat<T: PartialOrd + Clone>(m: Matrix<T>) -> T {
    let mut min = m.first().unwrap().clone();

    for i in m.iter() {
        if *i < min {
            min = i.clone();
        }
    }

    min
}

/// Default pooling function to find the maximum.
pub fn pool_max_mat<T: PartialOrd + Clone>(m: Matrix<T>) -> T {
    let mut max = m.first().unwrap().clone();

    for i in m.iter() {
        if *i > max {
            max = i.clone();
        }
    }

    max
}

/// Default pooling function to find the average.
/// The total number of elements is the total number of elements in the input
/// so this may vary if the kernel is hanging over the edge of the tensor.
pub fn pool_avg_mat<T: Add<Output = T> + Div<f64, Output = T> + Clone>(m: Matrix<T>) -> T {
    let sum = pool_sum_mat(m.clone());
    let elems = m.shape().element_count().to_f64().unwrap();

    sum / elems
}
