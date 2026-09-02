use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::definitions::matrix_slice::MatrixSlice;
use crate::definitions::matrix_slice_mut::MatrixSliceMut;
use num::{One, ToPrimitive, Zero};
use rand::distr::{Distribution, StandardUniform};
use rand::RngExt;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};
use rayon::slice::ParallelSliceMut;
use std::cmp::min;
use std::mem::MaybeUninit;
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
* Basic matrix utility functions
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

    /// Reshapes the tensor without checking compatibility of shapes.
    pub(crate) unsafe fn reshape_unchecked(self, new_rows: usize, new_cols: usize) -> Matrix<T> {
        Matrix {
            elements: self.elements,
            rows: new_rows,
            cols: new_cols,
        }
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
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: values.shape(),
                op: "set_all",
            });
        }

        self.elements.clone_from_slice(values);

        Ok(())
    }

    /// Sets all the values in the mutable matrix to the given values.
    /// This fails if the shape of the values does not match the matrix's shape.
    pub fn set_all_from_slice(&mut self, values: &MatrixSlice<T>) -> Result<(), TensorErrors>
    where
        T: Clone,
    {
        if self.rows() != values.rows() || self.cols() != values.cols() {
            return Err(TensorErrors::IncompatibleShapes {
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

    /// Slices the matrix without checking bounds.
    pub(crate) unsafe fn slice_unchecked(
        &self,
        rows_range: Range<usize>,
        cols_range: Range<usize>,
    ) -> MatrixSlice<T> {
        MatrixSlice {
            orig: self,
            start: (rows_range.start, cols_range.start),
            end: (rows_range.end, cols_range.end),
        }
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

    /// Slices the matrix mutably without checking bounds.
    pub(crate) unsafe fn slice_unchecked_mut(
        &'_ mut self,
        rows_range: Range<usize>,
        cols_range: Range<usize>,
    ) -> MatrixSliceMut<'_, T> {
        MatrixSliceMut {
            orig: self,
            start: (rows_range.start, cols_range.start),
            end: (rows_range.end, cols_range.end),
        }
    }

    /// Flips the contents of the rows of a matrix.
    pub fn flip_rows(mut self) -> Matrix<T> {
        let cols = self.cols;

        self.elements.chunks_mut(cols).for_each(|c| c.reverse());

        self
    }

    /// Flips the contents of the rows of a matrix.
    pub fn flip_rows_mt(mut self) -> Matrix<T>
    where
        T: Send + Sync,
    {
        let cols = self.cols;

        self.elements.par_chunks_mut(cols).for_each(|c| c.reverse());

        self
    }

    /// Flips the contents of the columns of a matrix.
    pub fn flip_cols(mut self) -> Matrix<T> {
        let cols = self.cols;
        let rows = self.rows;

        let (mut low, mut high) = (0, rows - 1);

        unsafe {
            while low < high {
                let (top, bottom) = self.elements.split_at_mut_unchecked(high * cols);
                let row_low = top.get_unchecked_mut(low * cols..low * cols + cols);
                let row_high = bottom.get_unchecked_mut(..cols);
                row_low.swap_with_slice(row_high);
                low += 1;
                high -= 1;
            }
        }

        self
    }

    /// Flips the contents of the columns of a matrix.
    pub fn flip_cols_mt(mut self) -> Matrix<T>
    where
        T: Send + Sync,
    {
        let cols = self.cols;
        let mid = self.rows / 2;

        unsafe {
            let (top, rest) = self.elements.split_at_mut_unchecked(mid * cols);
            let bottom = rest.get_unchecked_mut(rest.len() - mid * cols..);

            top.par_chunks_mut(cols)
                .zip(bottom.par_chunks_mut(cols).rev())
                .for_each(|(t, b)| t.swap_with_slice(b));
        }

        self
    }

    /// Flips a matrix along both the rows and the columns.
    pub fn flip(mut self) -> Matrix<T> {
        let cols = self.cols;
        let rows = self.rows;

        let (mut low, mut high) = (0, rows - 1);

        unsafe {
            while low < high {
                let (top, bottom) = self.elements.split_at_mut_unchecked(high * cols);
                let row_low = top.get_unchecked_mut(low * cols..low * cols + cols);
                let row_high = bottom.get_unchecked_mut(..cols);
                row_low.swap_with_slice(row_high);

                // Flip the row contents as well
                row_low.reverse();
                row_high.reverse();

                low += 1;
                high -= 1;
            }

            if low == high {
                // Need to flip the middle row
                self.elements
                    .get_unchecked_mut(low * cols..low * cols + cols)
                    .reverse();
            }
        }

        self
    }

    /// Flips a matrix along both the rows and the columns.
    pub fn flip_mt(mut self) -> Matrix<T>
    where
        T: Send + Sync,
    {
        let cols = self.cols;
        let mid = self.rows / 2;

        unsafe {
            let (top, rest) = self.elements.split_at_mut_unchecked(mid * cols);
            let bottom = rest.get_unchecked_mut(rest.len() - mid * cols..);

            top.par_chunks_mut(cols)
                .zip(bottom.par_chunks_mut(cols).rev())
                .for_each(|(t, b)| {
                    t.swap_with_slice(b);
                    // Need to flip along the rows too
                    t.reverse();
                    b.reverse();
                });

            if self.rows % 2 == 0 {
                // Need to flip the middle row
                self.elements
                    .get_unchecked_mut(mid * cols..mid * cols + cols)
                    .reverse();
            }
        }

        self
    }

    /// Transpose a matrix.
    pub fn transpose(mut self) -> Matrix<T> {
        let mut new_elements = Vec::<T>::with_capacity(self.elements.len());
        let buf = new_elements.spare_capacity_mut();

        self.elements
            .into_iter()
            .enumerate()
            .for_each(|(index, e)| unsafe {
                let i = index / self.cols;
                let j = index % self.cols;
                let new_index = j * self.rows + i;
                buf.get_unchecked_mut(new_index).write(e);
            });

        unsafe {
            new_elements.set_len(self.rows * self.cols);
        }

        Matrix {
            elements: new_elements,
            rows: self.cols,
            cols: self.rows,
        }
    }

    /// Transposes a matrix.
    pub fn transpose_mt(self) -> Matrix<T>
    where
        T: Send + Sync,
    {
        struct ThreadSafePtr<O>(*mut O);

        impl<O> ThreadSafePtr<O> {
            fn add(&self, offset: usize) -> ThreadSafePtr<O> {
                unsafe { ThreadSafePtr(self.0.add(offset)) }
            }

            fn write(&self, value: O) {
                unsafe { self.0.write(value) }
            }
        }

        unsafe impl<O> Sync for ThreadSafePtr<O> {}
        unsafe impl<O> Send for ThreadSafePtr<O> {}

        let mut new_elements = Vec::<T>::with_capacity(self.elements.len());
        let buf = new_elements.spare_capacity_mut();
        let buf_start = ThreadSafePtr(buf.as_mut_ptr());

        self.elements
            .into_par_iter()
            .enumerate()
            .for_each(|(index, e)| unsafe {
                let i = index / self.cols;
                let j = index % self.cols;
                let new_index = j * self.rows + i;
                let to_insert = MaybeUninit::new(e);
                buf_start.add(new_index).write(to_insert);
            });

        unsafe {
            new_elements.set_len(self.rows * self.cols);
        }

        Matrix {
            elements: new_elements,
            rows: self.cols,
            cols: self.rows,
        }
    }

    /// Concatenates two matrices along the columns.
    /// This fails if the number of columns do not match.
    pub fn concat_cols(mut self, other: Matrix<T>) -> Result<Matrix<T>, TensorErrors> {
        if self.cols != other.cols {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: other.shape(),
                op: "concat_cols",
            });
        }

        self.elements.extend(other.elements.into_iter());

        Ok(Matrix {
            rows: self.rows + other.rows,
            cols: self.cols,
            elements: self.elements,
        })
    }

    /// Concatenates two matrices along the columns without checking for shape compatibility.
    pub(crate) unsafe fn concat_cols_unchecked(mut self, other: Matrix<T>) -> Matrix<T> {
        self.elements.extend(other.elements.into_iter());

        Matrix {
            rows: self.rows + other.rows,
            cols: self.cols,
            elements: self.elements,
        }
    }

    /// Concatenates two matrices along the rows.
    /// This fails if the number of rows do not match.
    pub fn concat_rows(self, other: Matrix<T>) -> Result<Matrix<T>, TensorErrors> {
        if self.rows != other.rows {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: other.shape(),
                op: "concat_rows",
            });
        }

        let mut new_elements = Vec::with_capacity(self.elements.len() + other.elements.len());
        let mut self_iter = self.elements.into_iter();
        let mut other_iter = other.elements.into_iter();

        (0..self.rows).for_each(|_| {
            new_elements.extend(self_iter.by_ref().take(self.cols));
            new_elements.extend(other_iter.by_ref().take(other.cols));
        });

        Ok(Matrix {
            rows: self.rows,
            cols: self.cols + other.cols,
            elements: new_elements,
        })
    }

    /// Concatenates two matrices along the rows without checking for shape compatibility.
    pub(crate) unsafe fn concat_rows_unchecked(self, other: Matrix<T>) -> Matrix<T> {
        let mut new_elements = Vec::with_capacity(self.elements.len() + other.elements.len());
        let mut self_iter = self.elements.into_iter();
        let mut other_iter = other.elements.into_iter();

        (0..self.rows).for_each(|_| {
            new_elements.extend(self_iter.by_ref().take(self.cols));
            new_elements.extend(other_iter.by_ref().take(other.cols));
        });

        Matrix {
            rows: self.rows,
            cols: self.cols + other.cols,
            elements: new_elements,
        }
    }

    /// Pools a `Matrix<T>` into a `Matrix<O>` using a custom pooling function.
    /// The custom function will take a `MatrixSlice<T>` that corresponds to the slice that the kernel covers.
    /// If the kernel is hanging over the edge of the matrix, then only the bit of the matrix that fits is included.
    /// This will fail if either the kernel or stride shape contains 0, or if the matrix is empty.
    pub fn pool<O>(
        &self,
        pool_fn: impl Fn(MatrixSlice<T>) -> O,
        kernel_shape: (usize, usize),
        stride_shape: (usize, usize),
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
        let mut res = Vec::with_capacity(res_shape.0 * res_shape.1);
        unsafe { res.set_len(res_shape.0 * res_shape.1); }
        let mut res = Matrix {
            rows: res_shape.0,
            cols: res_shape.1,
            elements: res,
        };

        for (pos, val) in res.enumerated_iter_mut() {
            let start_pos = (pos.0 * stride_shape.0, pos.1 * stride_shape.1);
            let end_pos = (
                min(start_pos.0 + kernel_shape.0, self.rows),
                min(start_pos.1 + kernel_shape.1, self.cols),
            );

            let indices = (start_pos.0..end_pos.0, start_pos.1..end_pos.1);
            let value = unsafe { pool_fn(self.slice_unchecked(indices.0, indices.1)) };

            *val = value;
        }

        Ok(res)
    }

    /// Pools a `Matrix<T>` into a `Matrix<O>` using a custom pooling function with the index.
    /// The custom function will take a `MatrixSlice<T>` that corresponds to the slice that the kernel covers.
    /// If the kernel is hanging over the edge of the matrix, then only the bit of the matrix that fits is included.
    /// This will fail if either the kernel or stride shape contains 0, or if the matrix is empty.
    pub fn pool_indexed<O>(
        &self,
        pool_fn: impl Fn((usize, usize), MatrixSlice<T>) -> O,
        kernel_shape: (usize, usize),
        stride_shape: (usize, usize),
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
        let mut res = Vec::with_capacity(res_shape.0 * res_shape.1);
        unsafe { res.set_len(res_shape.0 * res_shape.1); }
        let mut res = Matrix {
            rows: res_shape.0,
            cols: res_shape.1,
            elements: res,
        };

        for (pos, val) in res.enumerated_iter_mut() {
            let start_pos = (pos.0 * stride_shape.0, pos.1 * stride_shape.1);
            let end_pos = (
                min(start_pos.0 + kernel_shape.0, self.rows),
                min(start_pos.1 + kernel_shape.1, self.cols),
            );

            let indices = (start_pos.0..end_pos.0, start_pos.1..end_pos.1);
            let value = unsafe { pool_fn(start_pos, self.slice_unchecked(indices.0, indices.1)) };

            *val = value;
        }

        Ok(res)
    }

    /// Pools a `Matrix<T>` into a `Matrix<O>` using a custom pooling function.
    /// The custom function will take a `MatrixSlice<T>` that corresponds to the slice that the kernel covers.
    /// If the kernel is hanging over the edge of the matrix, then only the bit of the matrix that fits is included.
    /// As this is multithreaded, a reference to the pooling function is expected.
    /// This will fail if either the kernel or stride shape contains 0, or if the matrix is empty.
    pub fn pool_mt<O: Send + Sync>(
        &self,
        pool_fn: &(impl Fn(MatrixSlice<T>) -> O + Sync),
        kernel_shape: (usize, usize),
        stride_shape: (usize, usize),
    ) -> Result<Matrix<O>, TensorErrors> where T: Send + Sync {
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

        let mut res = Vec::with_capacity(res_shape.0 * res_shape.1);
        unsafe { res.set_len(res_shape.0 * res_shape.1); }
        let mut res = Matrix {
            rows: res_shape.0,
            cols: res_shape.1,
            elements: res,
        };

        res.enumerated_par_iter_mut().for_each(|(index, elem)| unsafe {
            let self_pos = (index.0 * stride_shape.0, index.1 * stride_shape.1);
            let self_end_pos = (
                min(self_pos.0 + kernel_shape.0, self.rows),
                min(self_pos.1 + kernel_shape.1, self.cols),
            );

            *elem = pool_fn(
                self.slice_unchecked(self_pos.0..self_end_pos.0, self_pos.1..self_end_pos.1),
            );
        });

        Ok(res)
    }

    /// Pools a `Matrix<T>` into a `Matrix<O>` using a custom pooling function with the index.
    /// The custom function will take a `MatrixSlice<T>` that corresponds to the slice that the kernel covers.
    /// If the kernel is hanging over the edge of the matrix, then only the bit of the matrix that fits is included.
    /// As this is multithreaded, a reference to the pooling function is expected.
    /// This will fail if either the kernel or stride shape contains 0, or if the matrix is empty.
    pub fn pool_indexed_mt<O: Clone + Send + Sync>(
        &self,
        pool_fn: &(impl Fn((usize, usize), MatrixSlice<T>) -> O + Sync),
        kernel_shape: (usize, usize),
        stride_shape: (usize, usize),
    ) -> Result<Matrix<O>, TensorErrors> where T: Send + Sync {
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

        let mut res = Vec::with_capacity(res_shape.0 * res_shape.1);
        unsafe { res.set_len(res_shape.0 * res_shape.1); }
        let mut res = Matrix {
            rows: res_shape.0,
            cols: res_shape.1,
            elements: res,
        };

        res.enumerated_par_iter_mut().for_each(|(index, elem)| unsafe {
            let self_pos = (index.0 * stride_shape.0, index.1 * stride_shape.1);
            let self_end_pos = (
                min(self_pos.0 + kernel_shape.0, self.rows),
                min(self_pos.1 + kernel_shape.1, self.cols),
            );

            *elem = pool_fn(
                index,
                self.slice_unchecked(self_pos.0..self_end_pos.0, self_pos.1..self_end_pos.1),
            );
        });

        Ok(res)
    }

    /// Clips the values in the matrix between `[min, max]`.
    pub fn clip(self, min: T, max: T) -> Matrix<T> where T: PartialOrd + Clone {
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

    /// Clips the values in the matrix between `[min, max]`.
    pub fn par_clip(self, min: T, max: T) -> Matrix<T> where T: PartialOrd + Clone + Send + Sync {
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
pub fn pool_sum_mat<T: Add<Output = T> + Clone>(m: MatrixSlice<T>) -> Option<T> {
    m.iter().cloned().reduce(T::add)
}

/// Default pooling function to find the minimum.
pub fn pool_min_mat<T: PartialOrd + Clone>(m: MatrixSlice<T>) -> Option<T> {
    m.iter().reduce(|a, b| if a < b { a } else { b }).cloned()
}

/// Default pooling function to find the maximum.
pub fn pool_max_mat<T: PartialOrd + Clone>(m: MatrixSlice<T>) -> Option<T> {
    m.iter().reduce(|a, b| if a > b { a } else { b }).cloned()
}

/// Default pooling function to find the average.
/// The total number of elements is the total number of elements in the input
/// so this may vary if the kernel is hanging over the edge of the tensor.
pub fn pool_avg_mat<T: Add<Output = T> + Div<f64, Output = T> + Clone>(m: MatrixSlice<T>) -> Option<T> {
    let count = m.shape().element_count().to_f64();
    let sum = pool_sum_mat(m);
    match (sum, count) {
        (Some(s), Some(c)) => Some(s / c),
        _ => None,
    }
}
