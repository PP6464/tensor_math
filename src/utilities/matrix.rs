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
        let buf = elements.spare_capacity_mut();
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

        Ok(unsafe { self.reshape_unchecked(new_rows, new_cols) })
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
    ) -> Result<MatrixSlice<'_, T>, TensorErrors> {
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

        Ok(unsafe { self.slice_unchecked(rows_range, cols_range) })
    }

    /// Slices the matrix without checking bounds.
    pub(crate) unsafe fn slice_unchecked(
        &self,
        rows_range: Range<usize>,
        cols_range: Range<usize>,
    ) -> MatrixSlice<'_, T> {
        MatrixSlice {
            orig: self,
            start: (rows_range.start, cols_range.start),
            end: (rows_range.end, cols_range.end),
        }
    }

    /// Returns a slice covering the entire matrix.
    pub fn as_matrix_slice(&self) -> MatrixSlice<'_, T> {
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

        Ok(unsafe { self.slice_unchecked_mut(rows_range, cols_range) })
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

    /// Returns a mutable slice covering the entire matrix.
    pub fn as_matrix_slice_mut(&mut self) -> MatrixSliceMut<'_, T> {
        MatrixSliceMut {
            start: (0, 0),
            end: (self.rows, self.cols),
            orig: self,
        }
    }

    /// Flips the contents of the rows of a matrix.
    pub fn flip_rows(mut self) -> Matrix<T> {
        let cols = self.cols;

        if cols == 0 || self.rows == 0 {
            return self;
        }

        self.elements.chunks_mut(cols).for_each(|c| c.reverse());

        self
    }

    /// Flips the contents of the rows of a matrix.
    pub fn flip_rows_mt(mut self) -> Matrix<T>
    where
        T: Send + Sync,
    {
        let cols = self.cols;

        if cols == 0 || self.rows == 0 {
            return self;
        }

        self.elements.par_chunks_mut(cols).for_each(|c| c.reverse());

        self
    }

    /// Flips the contents of the columns of a matrix.
    pub fn flip_cols(mut self) -> Matrix<T> {
        let cols = self.cols;
        let rows = self.rows;

        if rows == 0 || cols == 0 {
            return self;
        }

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

        if self.rows == 0 || self.cols == 0 {
            return self;
        }

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

        if self.rows == 0 || self.cols == 0 {
            return self;
        }

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

        if self.rows == 0 || self.cols == 0 {
            return self;
        }

        unsafe {
            let (top, rest) = self.elements.split_at_mut_unchecked(mid * cols);
            let bottom = rest.get_unchecked_mut(rest.len() - mid * cols..);

            top.par_chunks_mut(cols)
                .zip(bottom.par_chunks_mut(cols).rev())
                .for_each(|(t, b)| {
                    t.swap_with_slice(b);
                    // Need to flip the rows too
                    t.reverse();
                    b.reverse();
                });

            if !self.rows.is_multiple_of(2) {
                // Need to flip the middle row
                self.elements
                    .get_unchecked_mut(mid * cols..mid * cols + cols)
                    .reverse();
            }
        }

        self
    }

    /// Transpose a matrix.
    pub fn transpose(self) -> Matrix<T> {
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
            .for_each(|(index, e)| {
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
    pub fn concat_along_cols(self, other: Matrix<T>) -> Result<Matrix<T>, TensorErrors> {
        if self.cols != other.cols {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: other.shape(),
                op: "concat_cols",
            });
        }

        Ok(unsafe { self.concat_along_cols_unchecked(other) })
    }

    /// Concatenates two matrices along the columns without checking for shape compatibility.
    pub(crate) unsafe fn concat_along_cols_unchecked(mut self, other: Matrix<T>) -> Matrix<T> {
        self.elements.extend(other.elements);

        Matrix {
            rows: self.rows + other.rows,
            cols: self.cols,
            elements: self.elements,
        }
    }

    /// Concatenates two matrices along the rows.
    /// This fails if the number of rows do not match.
    pub fn concat_along_rows(self, other: Matrix<T>) -> Result<Matrix<T>, TensorErrors> {
        if self.rows != other.rows {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: other.shape(),
                op: "concat_rows",
            });
        }

        Ok(unsafe { self.concat_along_rows_unchecked(other) })
    }

    /// Concatenates two matrices along the rows without checking for shape compatibility.
    pub(crate) unsafe fn concat_along_rows_unchecked(self, other: Matrix<T>) -> Matrix<T> {
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
        // We will initialise this vector here so we can disable the error
        #[allow(clippy::uninit_vec)]
        unsafe {
            res.set_len(res_shape.0 * res_shape.1);
        }
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
        // We will initialise this vector here so we can disable the error
        #[allow(clippy::uninit_vec)]
        unsafe {
            res.set_len(res_shape.0 * res_shape.1);
        }
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
    ) -> Result<Matrix<O>, TensorErrors>
    where
        T: Send + Sync,
    {
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
        // We will initialise this vector here so we can disable the error
        #[allow(clippy::uninit_vec)]
        unsafe {
            res.set_len(res_shape.0 * res_shape.1);
        }
        let mut res = Matrix {
            rows: res_shape.0,
            cols: res_shape.1,
            elements: res,
        };

        res.enumerated_par_iter_mut()
            .for_each(|(index, elem)| unsafe {
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
    ) -> Result<Matrix<O>, TensorErrors>
    where
        T: Send + Sync,
    {
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
        // We will initialise this vector here so we can disable the error
        #[allow(clippy::uninit_vec)]
        unsafe {
            res.set_len(res_shape.0 * res_shape.1);
        }
        let mut res = Matrix {
            rows: res_shape.0,
            cols: res_shape.1,
            elements: res,
        };

        res.enumerated_par_iter_mut()
            .for_each(|(index, elem)| unsafe {
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
    pub fn clip(self, min: T, max: T) -> Matrix<T>
    where
        T: PartialOrd + Clone,
    {
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
    pub fn par_clip(self, min: T, max: T) -> Matrix<T>
    where
        T: PartialOrd + Clone + Send + Sync,
    {
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
pub fn pool_avg_mat<T: Add<Output = T> + Div<f64, Output = T> + Clone>(
    m: MatrixSlice<T>,
) -> Option<T> {
    let count = m.shape().element_count().to_f64();
    let sum = pool_sum_mat(m);
    match (sum, count) {
        (Some(s), Some(c)) => Some(s / c),
        _ => None,
    }
}

#[cfg(test)]
mod matrix_utils_tests {
    use rayon::iter::ParallelIterator;
    use tensor_math::definitions::errors::TensorErrors;
    use tensor_math::definitions::matrix::Matrix;
    use tensor_math::utilities::matrix::{
        pool_avg_mat, pool_max_mat, pool_min_mat, pool_sum_mat, identity, eye,
    };

    #[test]
    fn concat() {
        let m1 = Matrix::new(3, 3, (0..9).collect()).unwrap();
        let m2 = Matrix::new(3, 2, (9..15).collect()).unwrap();
        let res = m1.clone().concat_along_rows(m2.clone()).unwrap();
        let ans =
            Matrix::new(3, 5, vec![0, 1, 2, 9, 10, 3, 4, 5, 11, 12, 6, 7, 8, 13, 14]).unwrap();
        assert_eq!(ans, res);
    }

    #[test]
    fn concat_with_empty_matrix() {
        let m1 = Matrix::new(3, 3, (0..9).collect()).unwrap();
        let m2 = Matrix::new(3, 0, vec![]).unwrap();
        let res = m1.clone().concat_along_rows(m2.clone()).unwrap();
        assert_eq!(res, m1);

        let m3 = Matrix::new(0, 3, vec![]).unwrap();
        let res = m1.clone().concat_along_cols(m3.clone()).unwrap();
        assert_eq!(res, m1);
    }

    #[test]
    fn concat_multithreaded_with_empty_matrix() {
        let m1 = Matrix::new(30, 30, (0..900).collect()).unwrap();
        let m2 = Matrix::new(30, 0, vec![]).unwrap();
        let res = m1.clone().concat_along_rows(m2.clone()).unwrap();
        assert_eq!(res, m1);

        let m3 = Matrix::new(0, 30, vec![]).unwrap();
        let res = m1.clone().concat_along_cols(m3.clone()).unwrap();
        assert_eq!(res, m1);
    }

    #[test]
    fn invalid_concat() {
        let m1 = Matrix::new(3, 3, (0..9).collect()).unwrap();
        let m2 = Matrix::new(3, 2, (0..6).collect()).unwrap();
        assert!(matches!(
            m1.clone().concat_along_cols(m2.clone()).unwrap_err(),
            TensorErrors::IncompatibleShapes { .. }
        ));

        let m3 = Matrix::new(2, 3, (0..6).collect()).unwrap();
        assert!(matches!(
            m1.clone().concat_along_rows(m3.clone()).unwrap_err(),
            TensorErrors::IncompatibleShapes { .. }
        ));
    }

    #[test]
    fn reshape_correctly() {
        let m1 = (0..6).collect::<Matrix<_>>();
        let ans = Matrix::new(2, 3, (0..6).collect()).unwrap();
        assert_eq!(m1.clone().reshape(2, 3).unwrap(), ans);

        let m2 = Matrix::<usize>::new(0, 2, vec![]).unwrap();
        let ans = m2.clone().reshape(0, 100).unwrap();
        assert_eq!(ans, Matrix::new(0, 100, vec![]).unwrap());
    }

    #[test]
    fn invalid_reshape() {
        let m1 = Matrix::new(3, 3, (0..9).collect()).unwrap();
        assert_eq!(
            m1.clone().reshape(2, 3).unwrap_err(),
            TensorErrors::ShapeSizeDoesNotMatch
        );

        let m2 = Matrix::<usize>::new(3, 0, vec![]).unwrap();
        assert_eq!(
            m2.clone().reshape(3, 1).unwrap_err(),
            TensorErrors::ShapeSizeDoesNotMatch
        )
    }

    #[test]
    fn flip_axes_with_empty_matrix() {
        let m1 = Matrix::<i32>::new(0, 3, vec![]).unwrap();
        assert_eq!(m1.clone().flip_cols(), m1);
        assert_eq!(m1.clone().flip_rows(), m1);
        assert_eq!(m1.clone().flip(), m1);

        assert_eq!(m1.clone().flip_cols_mt(), m1);
        assert_eq!(m1.clone().flip_rows_mt(), m1);
        assert_eq!(m1.clone().flip_mt(), m1);

        let m2 = Matrix::<i32>::new(3, 0, vec![]).unwrap();
        assert_eq!(m2.clone().flip_cols(), m2);
        assert_eq!(m2.clone().flip_rows(), m2);
        assert_eq!(m2.clone().flip(), m2);
    }

    #[test]
    fn collect_empty_matrix() {
        let empty_vec: Vec<i32> = vec![];
        let m: Matrix<i32> = empty_vec.into_iter().collect();

        assert_eq!(m.rows(), 1);
        assert_eq!(m.cols(), 0);
        assert!(m.is_empty());
    }

    #[test]
    fn random() {
        Matrix::<i32>::rand(2, 3);
    }

    #[test]
    fn transform_elementwise() {
        let m1 = Matrix::<f64>::new(2, 2, (0..4).map(f64::from).collect()).unwrap();
        let m1 = m1.clone().map(f64::sqrt);
        let ans = Matrix::<f64>::new(2, 2, vec![0.0, 1.0, 2.0_f64.sqrt(), 3.0_f64.sqrt()]).unwrap();
        assert!((ans - m1).into_iter().map(f64::abs).sum::<f64>() < 1e-15);
    }

    #[test]
    fn slicing() {
        let m1 = Matrix::<f64>::new(4, 4, (0..16).map(f64::from).collect()).unwrap();
        let slice = m1.slice(1..3, 1..3).unwrap();
        let ans = Matrix::<f64>::new(2, 2, vec![5.0, 6.0, 9.0, 10.0]).unwrap();
        assert_eq!(slice.clone_into_matrix(), ans);
    }

    #[test]
    fn invalid_slice_out_of_bounds() {
        let m1 = Matrix::<f64>::new(4, 4, (0..16).map(f64::from).collect()).unwrap();
        let err = m1.slice(1..5, 1..2).unwrap_err();
        assert!(matches!(err, TensorErrors::SliceIndicesOutOfBounds { .. }));
        let err = m1.slice(1..2, 1..5).unwrap_err();
        assert!(matches!(err, TensorErrors::SliceIndicesOutOfBounds { .. }));
    }

    #[test]
    fn slicing_mut() {
        let mut m1 = Matrix::<f64>::new(4, 4, (0..16).map(f64::from).collect()).unwrap();
        let mut slice = m1.slice_mut(1..3, 1..3).unwrap();
        slice[(0, 0)] = 69.0;
        slice[(0, 1)] = 42.0;
        slice[&[1, 0]] = -20.0;
        slice[&[1, 1]] = 91.0;
        let ans = Matrix::<f64>::new(
            4,
            4,
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 69.0, 42.0, 7.0, 8.0, -20.0, 91.0, 11.0, 12.0, 13.0, 14.0,
                15.0,
            ],
        )
        .unwrap();
        assert_eq!(m1, ans);
    }

    #[test]
    fn set_all() {
        let mut m1 = Matrix::<f64>::new(4, 4, (0..16).map(f64::from).collect()).unwrap();
        let mut slice_mut = m1.slice_mut(1..3, 1..3).unwrap();
        let inserted = Matrix::<f64>::new(2, 2, (0..4).map(f64::from).collect()).unwrap();
        slice_mut.set_all(&inserted).unwrap();

        let ans = Matrix::<f64>::new(
            4,
            4,
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 7.0, 8.0, 2.0, 3.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            ],
        )
        .unwrap();
        assert_eq!(m1, ans);
    }

    #[test]
    #[should_panic]
    fn set_all_fail() {
        let mut m1 = Matrix::<f64>::new(4, 4, (0..16).map(f64::from).collect()).unwrap();
        let mut slice_mut = m1.slice_mut(1..3, 1..3).unwrap();

        slice_mut.set_all(&Matrix::zeros(2, 1)).unwrap();
    }

    #[test]
    #[should_panic]
    fn slice_mut_out_of_bounds() {
        let mut m1 = Matrix::<f64>::new(4, 4, (0..16).map(f64::from).collect()).unwrap();
        let mut slice = m1.slice_mut(1..3, 1..3).unwrap();
        slice[&[3, 1]] = 69.0;
    }
    #[test]
    fn transpose() {
        let m1 = Matrix::<i32>::new(3, 3, (0..9).collect()).unwrap();
        let t = m1.clone().transpose();
        let ans = Matrix::<i32>::new(3, 3, vec![0, 3, 6, 1, 4, 7, 2, 5, 8]).unwrap();

        assert_eq!(ans, t);
    }

    #[test]
    fn transpose_mt() {
        let m1 = Matrix::<i32>::rand(10, 5);

        let ans = m1.clone().transpose();
        let mt_ans = m1.clone().transpose_mt();

        assert_eq!(mt_ans, ans);
    }

    #[test]
    fn clipping() {
        let m1 = Matrix::<i32>::new(3, 3, (0..9).collect()).unwrap();
        let res = m1.clone().clip(3, 6);
        let ans = Matrix::new(3, 3, vec![3, 3, 3, 3, 4, 5, 6, 6, 6]).unwrap();

        assert_eq!(res, ans);
    }

    #[test]
    fn clipping_empty() {
        let m1 = Matrix::<i32>::new(0, 3, vec![]).unwrap();
        let res = m1.clone().clip(3, 6);
        let ans = Matrix::new(0, 3, vec![]).unwrap();

        assert_eq!(res, ans);

        let m1 = Matrix::<i32>::new(3, 0, vec![]).unwrap();
        let res = m1.clone().clip(3, 6);
        let ans = Matrix::new(3, 0, vec![]).unwrap();

        assert_eq!(res, ans);

        let m1 = Matrix::<i32>::new(0, 0, vec![]).unwrap();
        let res = m1.clone().clip(3, 6);
        let ans = Matrix::new(0, 0, vec![]).unwrap();

        assert_eq!(res, ans);
    }

    #[test]
    fn pool() {
        let m1 = Matrix::<f64>::new(3, 3, (0..9).map(f64::from).collect()).unwrap();
        let sum = m1.clone().pool(pool_sum_mat, (2, 2), (2, 2)).unwrap();
        let min = m1.clone().pool(pool_min_mat, (2, 2), (1, 2)).unwrap();
        let avg = m1.clone().pool(pool_avg_mat, (1, 1), (1, 1)).unwrap();
        let max = m1.clone().pool(pool_max_mat, (1, 1), (2, 2)).unwrap();

        let sum_ans = Matrix::<f64>::new(2, 2, vec![8.0, 7.0, 13.0, 8.0])
            .unwrap()
            .map(Some);
        let min_ans = Matrix::<f64>::new(3, 2, vec![0.0, 2.0, 3.0, 5.0, 6.0, 8.0])
            .unwrap()
            .map(Some);
        let avg_ans = m1.clone().map(Some);
        let max_ans = Matrix::<f64>::new(2, 2, vec![0.0, 2.0, 6.0, 8.0])
            .unwrap()
            .map(Some);

        assert_eq!(sum, sum_ans);
        assert_eq!(min, min_ans);
        assert_eq!(avg, avg_ans);
        assert_eq!(max, max_ans);
    }

    #[test]
    fn pool_mt() {
        let m1 = Matrix::<f64>::rand(20, 20);

        let ans = m1.clone().pool(pool_avg_mat, (5, 4), (2, 3)).unwrap();
        let mt_ans = m1.clone().pool_mt(&pool_avg_mat, (5, 4), (2, 3)).unwrap();

        assert_eq!(ans, mt_ans);
    }

    #[test]
    fn flip() {
        let m1 = Matrix::<i32>::new(2, 3, vec![0, 1, 2, 3, 4, 5]).unwrap();
        let res_cols = m1.clone().flip_cols();
        let res_rows = m1.clone().flip_rows();
        let res = m1.clone().flip();

        let ans_cols = Matrix::<i32>::new(2, 3, vec![3, 4, 5, 0, 1, 2]).unwrap();
        let ans_rows = Matrix::<i32>::new(2, 3, vec![2, 1, 0, 5, 4, 3]).unwrap();
        let ans = Matrix::<i32>::new(2, 3, vec![5, 4, 3, 2, 1, 0]).unwrap();

        assert_eq!(res_cols, ans_cols);
        assert_eq!(res_rows, ans_rows);
        assert_eq!(res, ans);
    }

    #[test]
    fn flip_mt() {
        let m1 = Matrix::<i32>::rand(10, 20);
        let m2 = Matrix::<i32>::rand(10, 21);
        let m3 = Matrix::<i32>::rand(11, 11);
        let m4 = Matrix::<i32>::rand(11, 10);

        let res_cols_1 = m1.clone().flip_cols();
        let res_cols_2 = m2.clone().flip_cols();
        let res_cols_3 = m3.clone().flip_cols();
        let res_cols_4 = m4.clone().flip_cols();
        let res_rows_1 = m1.clone().flip_rows();
        let res_rows_2 = m2.clone().flip_rows();
        let res_rows_3 = m3.clone().flip_rows();
        let res_rows_4 = m4.clone().flip_rows();
        let res_1 = m1.clone().flip();
        let res_2 = m2.clone().flip();
        let res_3 = m3.clone().flip();
        let res_4 = m4.clone().flip();

        let res_cols_mt_1 = m1.clone().flip_cols_mt();
        let res_cols_mt_2 = m2.clone().flip_cols_mt();
        let res_cols_mt_3 = m3.clone().flip_cols_mt();
        let res_cols_mt_4 = m4.clone().flip_cols_mt();
        let res_rows_mt_1 = m1.clone().flip_rows_mt();
        let res_rows_mt_2 = m2.clone().flip_rows_mt();
        let res_rows_mt_3 = m3.clone().flip_rows_mt();
        let res_rows_mt_4 = m4.clone().flip_rows_mt();
        let res_mt_1 = m1.clone().flip_mt();
        let res_mt_2 = m2.clone().flip_mt();
        let res_mt_3 = m3.clone().flip_mt();
        let res_mt_4 = m4.clone().flip_mt();

        assert_eq!(res_cols_1, res_cols_mt_1);
        assert_eq!(res_cols_2, res_cols_mt_2);
        assert_eq!(res_cols_3, res_cols_mt_3);
        assert_eq!(res_cols_4, res_cols_mt_4);
        assert_eq!(res_rows_1, res_rows_mt_1);
        assert_eq!(res_rows_2, res_rows_mt_2);
        assert_eq!(res_rows_3, res_rows_mt_3);
        assert_eq!(res_rows_4, res_rows_mt_4);
        assert_eq!(res_1, res_mt_1);
        assert_eq!(res_2, res_mt_2);
        assert_eq!(res_3, res_mt_3);
        assert_eq!(res_4, res_mt_4);
    }

    #[test]
    fn invalid_slice_mut_index_ranges_invalid() {
        let mut m1 = Matrix::<i32>::new(3, 3, (0..9).collect()).unwrap();

        let err = m1.slice_mut(10..2, 1..2).unwrap_err();
        assert!(matches!(err, TensorErrors::InvalidInterval { .. }));

        let err = m1.slice_mut(1..2, 10..2).unwrap_err();
        assert!(matches!(err, TensorErrors::InvalidInterval { .. }));
    }

    #[test]
    fn invalid_slice_mut_indices_out_of_bounds() {
        let mut m1 = Matrix::<i32>::new(3, 3, (0..9).collect()).unwrap();

        let err = m1.slice_mut(4..10, 1..2).unwrap_err();
        assert!(matches!(err, TensorErrors::SliceIndicesOutOfBounds { .. }));

        let err = m1.slice_mut(1..2, 4..10).unwrap_err();
        assert!(matches!(err, TensorErrors::SliceIndicesOutOfBounds { .. }));
    }

    #[test]
    fn pooling_invalid_shape() {
        let m1 = Matrix::<f64>::new(3, 3, (0..9).map(f64::from).collect()).unwrap();

        let err = m1.clone().pool(pool_avg_mat, (0, 1), (1, 1)).unwrap_err();
        assert!(matches!(err, TensorErrors::ShapeContainsZero));

        let err = m1
            .pool_indexed(|_, m| pool_max_mat(m), (1, 1), (0, 1))
            .unwrap_err();
        assert!(matches!(err, TensorErrors::ShapeContainsZero));

        let err = m1
            .clone()
            .pool_mt(&pool_min_mat, (1, 0), (1, 1))
            .unwrap_err();
        assert!(matches!(err, TensorErrors::ShapeContainsZero));

        let err = m1
            .pool_indexed_mt(&|_, m| pool_sum_mat(m), (1, 1), (1, 0))
            .unwrap_err();
        assert!(matches!(err, TensorErrors::ShapeContainsZero));

        let m2 = Matrix::<f64>::new(0, 3, vec![]).unwrap();

        let err = m2.clone().pool(pool_avg_mat, (1, 1), (1, 1)).unwrap_err();
        assert!(matches!(err, TensorErrors::TensorEmpty { .. }));

        let err = m2
            .pool_indexed(|_, m| pool_sum_mat(m), (1, 1), (1, 1))
            .unwrap_err();
        assert!(matches!(err, TensorErrors::TensorEmpty { .. }));

        let err = m2
            .clone()
            .pool_mt(&pool_min_mat, (1, 1), (1, 1))
            .unwrap_err();
        assert!(matches!(err, TensorErrors::TensorEmpty { .. }));

        let err = m2
            .pool_indexed_mt(&|_, m| pool_max_mat(m), (1, 1), (1, 1))
            .unwrap_err();
        assert!(matches!(err, TensorErrors::TensorEmpty { .. }));
    }

    #[test]
    fn slice_with_empty_indices() {
        let m1 = Matrix::<f64>::new(4, 4, (0..16).map(f64::from).collect()).unwrap();
        let slice = m1.slice(1..1, 1..3).unwrap();
        assert_eq!(slice.rows(), 0);
        assert_eq!(slice.cols(), 2);

        let slice = m1.slice(1..3, 1..1).unwrap();
        assert_eq!(slice.rows(), 2);
        assert_eq!(slice.cols(), 0);
    }

    #[test]
    fn enumerated_iter() {
        let m1 = Matrix::new(2, 2, vec![10, 20, 30, 40]).unwrap();
        let expected = vec![((0, 0), 10), ((0, 1), 20), ((1, 0), 30), ((1, 1), 40)];
        let res: Vec<_> = m1.enumerated_iter().map(|(idx, x)| (idx, *x)).collect();
        assert_eq!(res, expected);

        let m2 = Matrix::new(0, 2, vec![]).unwrap();
        let res2: Vec<((usize, usize), i32)> =
            m2.enumerated_iter().map(|(idx, x)| (idx, *x)).collect();
        assert!(res2.is_empty());
    }

    #[test]
    fn enumerated_iter_mut() {
        let mut m1 = Matrix::new(2, 2, vec![10, 20, 30, 40]).unwrap();
        for ((r, c), val) in m1.enumerated_iter_mut() {
            *val += r + c;
        }
        let ans = Matrix::new(2, 2, vec![10, 21, 31, 42]).unwrap();
        assert_eq!(m1, ans);

        let mut m2 = Matrix::<usize>::new(0, 2, vec![]).unwrap();
        let res2 = m2.enumerated_iter_mut().collect::<Vec<_>>();
        assert!(res2.is_empty());
    }

    #[test]
    fn enumerated_iter_empty_variants() {
        let m_row_empty = Matrix::<i32>::new(0, 3, vec![]).unwrap();
        assert_eq!(m_row_empty.enumerated_iter().count(), 0);

        let m_col_empty = Matrix::<i32>::new(3, 0, vec![]).unwrap();
        assert_eq!(m_col_empty.enumerated_iter().count(), 0);

        let m_both_empty = Matrix::<i32>::new(0, 0, vec![]).unwrap();
        assert_eq!(m_both_empty.enumerated_iter().count(), 0);
    }

    #[test]
    fn enumerated_iter_mut_empty_variants() {
        let mut m_row_empty = Matrix::<i32>::new(0, 3, vec![]).unwrap();
        assert_eq!(m_row_empty.enumerated_iter_mut().count(), 0);

        let mut m_col_empty = Matrix::<i32>::new(3, 0, vec![]).unwrap();
        assert_eq!(m_col_empty.enumerated_iter_mut().count(), 0);

        let mut m_both_empty = Matrix::<i32>::new(0, 0, vec![]).unwrap();
        assert_eq!(m_both_empty.enumerated_iter_mut().count(), 0);
    }

    #[test]
    fn enumerated_par_iter() {
        let m1 = Matrix::new(2, 2, vec![10, 20, 30, 40]).unwrap();
        let expected = vec![((0, 0), 10), ((0, 1), 20), ((1, 0), 30), ((1, 1), 40)];
        let res: Vec<_> = m1
            .enumerated_par_iter()
            .map(|(idx, x)| (idx, *x))
            .collect();
        assert_eq!(res, expected);

        let m2 = Matrix::new(0, 2, vec![]).unwrap();
        let res2: Vec<((usize, usize), i32)> = m2
            .enumerated_par_iter()
            .map(|(idx, x)| (idx, *x))
            .collect();
        assert!(res2.is_empty());
    }

    #[test]
    fn enumerated_par_iter_mut() {
        let mut m1 = Matrix::new(2, 2, vec![10, 20, 30, 40]).unwrap();
        m1.enumerated_par_iter_mut()
            .for_each(|((r, c), val)| *val += r + c);
        let ans = Matrix::new(2, 2, vec![10, 21, 31, 42]).unwrap();
        assert_eq!(m1, ans);

        let mut m2 = Matrix::<usize>::new(0, 2, vec![]).unwrap();
        let res2 = m2.enumerated_par_iter_mut().collect::<Vec<_>>();
        assert!(res2.is_empty());
    }

    #[test]
    fn enumerated_par_iter_empty_variants() {
        let m_row_empty = Matrix::<i32>::new(0, 3, vec![]).unwrap();
        assert_eq!(m_row_empty.enumerated_par_iter().count(), 0);

        let m_col_empty = Matrix::<i32>::new(3, 0, vec![]).unwrap();
        assert_eq!(m_col_empty.enumerated_par_iter().count(), 0);

        let m_both_empty = Matrix::<i32>::new(0, 0, vec![]).unwrap();
        assert_eq!(m_both_empty.enumerated_par_iter().count(), 0);
    }

    #[test]
    fn enumerated_par_iter_mut_empty_variants() {
        let mut m_row_empty = Matrix::<i32>::new(0, 3, vec![]).unwrap();
        assert_eq!(m_row_empty.enumerated_par_iter_mut().count(), 0);

        let mut m_col_empty = Matrix::<i32>::new(3, 0, vec![]).unwrap();
        assert_eq!(m_col_empty.enumerated_par_iter_mut().count(), 0);

        let mut m_both_empty = Matrix::<i32>::new(0, 0, vec![]).unwrap();
        assert_eq!(m_both_empty.enumerated_par_iter_mut().count(), 0);
    }

    #[test]
    fn default_matrix() {
        let m: Matrix<i32> = Default::default();
        assert_eq!(m.rows(), 1);
        assert_eq!(m.cols(), 1);
        assert_eq!(m.elements(), &vec![0]);
    }

    #[test]
    fn identity_and_eye_empty() {
        let identity: Matrix<i32> = identity(0);
        assert_eq!(identity.rows(), 0);
        assert_eq!(identity.cols(), 0);
        assert_eq!(identity.elements(), &vec![]);

        let eye: Matrix<i32> = eye(0);
        assert_eq!(eye.rows(), 0);
        assert_eq!(eye.cols(), 0);
        assert_eq!(eye.elements(), &vec![]);
    }

    #[test]
    fn identity_and_eye_normal() {
        let identity: Matrix<i32> = identity(3);
        assert_eq!(identity.rows(), 3);
        assert_eq!(identity.cols(), 3);
        assert_eq!(identity.elements(), &vec![1, 0, 0, 0, 1, 0, 0, 0, 1]);

        let eye: Matrix<i32> = eye(3);
        assert_eq!(eye.rows(), 3);
        assert_eq!(eye.cols(), 3);
        assert_eq!(eye.elements(), &vec![1, 0, 0, 0, 1, 0, 0, 0, 1]);
    }

    #[test]
    fn from_shape_empty() {
        let m: Matrix<i32> = Matrix::from_shape(0, 0);
        assert_eq!(m.rows(), 0);
        assert_eq!(m.cols(), 0);
        assert_eq!(m.elements(), &vec![]);

        let m: Matrix<i32> = Matrix::from_shape(0, 5);
        assert_eq!(m.rows(), 0);
        assert_eq!(m.cols(), 5);
        assert_eq!(m.elements(), &vec![]);

        let m: Matrix<i32> = Matrix::from_shape(5, 0);
        assert_eq!(m.rows(), 5);
        assert_eq!(m.cols(), 0);
        assert_eq!(m.elements(), &vec![]);
    }

    #[test]
    fn from_shape_normal() {
        let m: Matrix<i32> = Matrix::from_shape(2, 3);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert_eq!(m.elements(), &vec![0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn map_tests() {
        let m: Matrix<i32> = Matrix::from_shape(2, 3);
        let m_mapped = m.clone().map(|x| x + 1);
        assert_eq!(m_mapped.rows(), 2);
        assert_eq!(m_mapped.cols(), 3);
        assert_eq!(m_mapped.elements(), &vec![1, 1, 1, 1, 1, 1]);

        let m_par_mapped = m.clone().par_map(|x| x + 1);
        assert_eq!(m_par_mapped.rows(), 2);
        assert_eq!(m_par_mapped.cols(), 3);
        assert_eq!(m_par_mapped.elements(), &vec![1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn map_ref_tests() {
        let m: Matrix<i32> = Matrix::from_shape(2, 3);
        let m_mapped = m.clone().map_refs(|x| x + 1);
        assert_eq!(m_mapped.rows(), 2);
        assert_eq!(m_mapped.cols(), 3);
        assert_eq!(m_mapped.elements(), &vec![1, 1, 1, 1, 1, 1]);

        let m_par_mapped = m.clone().par_map_refs(|x| x + 1);
        assert_eq!(m_par_mapped.rows(), 2);
        assert_eq!(m_par_mapped.cols(), 3);
        assert_eq!(m_par_mapped.elements(), &vec![1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn valid_set_all() {
        let mut m1 = Matrix::<f64>::new(4, 4, (0..16).map(f64::from).collect()).unwrap();
        let m2 = Matrix::<f64>::new(4, 4, (16..32).map(f64::from).collect()).unwrap();
        m1.set_all(&m2).unwrap();
        assert_eq!(m1, m2);
    }

    #[test]
    fn invalid_set_all() {
        let mut m1 = Matrix::<f64>::new(4, 4, (0..16).map(f64::from).collect()).unwrap();
        let m2 = Matrix::<f64>::new(2, 2, (16..20).map(f64::from).collect()).unwrap();
        let result = m1.set_all(&m2);
        assert!(matches!(result, Err(TensorErrors::IncompatibleShapes { .. })));
    }

    #[test]
    fn valid_set_all_from_slice() {
        let mut m1 = Matrix::<f64>::new(4, 4, (0..16).map(f64::from).collect()).unwrap();
        let m2 = Matrix::<f64>::new(4, 4, (16..32).map(f64::from).collect()).unwrap();
        m1.set_all_from_slice(&m2.as_matrix_slice()).unwrap();
        assert_eq!(m1, m2);
    }

    #[test]
    fn invalid_set_all_from_slice() {
        let mut m1 = Matrix::<f64>::new(4, 4, (0..16).map(f64::from).collect()).unwrap();
        let m2 = Matrix::<f64>::new(2, 2, (16..20).map(f64::from).collect()).unwrap();
        let result = m1.set_all_from_slice(&m2.as_matrix_slice());
        assert!(matches!(result, Err(TensorErrors::IncompatibleShapes { .. })));
    }

    #[test]
    fn as_matrix_slice() {
        let m1 = Matrix::<f64>::new(4, 4, (0..16).map(f64::from).collect()).unwrap();
        let slice = m1.as_matrix_slice();
        assert_eq!(slice.rows(), 4);
        assert_eq!(slice.cols(), 4);
        assert_eq!(slice.iter().copied().collect::<Vec<_>>(), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
    }

    #[test]
    fn as_matrix_slice_mut() {
        let mut m1 = Matrix::<f64>::new(4, 4, (0..16).map(f64::from).collect()).unwrap();
        let mut slice_mut = m1.as_matrix_slice_mut();
        assert_eq!(slice_mut.rows(), 4);
        assert_eq!(slice_mut.cols(), 4);
        assert_eq!(slice_mut.iter().copied().collect::<Vec<_>>(), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        slice_mut.iter_mut().for_each(|x| *x += 1.0);
        assert_eq!(m1.iter().copied().collect::<Vec<_>>(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
    }

    #[test]
    fn par_clip_empty() {
        let m1 = Matrix::<i32>::new(0, 3, vec![]).unwrap();
        let res = m1.clone().par_clip(3, 6);
        assert_eq!(res.rows(), 0);
        assert_eq!(res.cols(), 3);
        assert_eq!(res.elements(), &vec![]);

        let m1 = Matrix::<i32>::new(3, 0, vec![]).unwrap();
        let res = m1.clone().par_clip(3, 6);
        assert_eq!(res.rows(), 3);
        assert_eq!(res.cols(), 0);
        assert_eq!(res.elements(), &vec![]);

        let m1 = Matrix::<i32>::new(0, 0, vec![]).unwrap();
        let res = m1.clone().par_clip(3, 6);
        assert_eq!(res.rows(), 0);
        assert_eq!(res.cols(), 0);
        assert_eq!(res.elements(), &vec![]);
    }

    #[test]
    fn par_clip_normal() {
        let m1 = Matrix::<i32>::new(3, 3, (0..9).collect()).unwrap();
        let res = m1.clone().par_clip(3, 6);
        let ans = Matrix::new(3, 3, vec![3, 3, 3, 3, 4, 5, 6, 6, 6]).unwrap();
        assert_eq!(res, ans);
    }

    #[test]
    fn slice_invalid_intervals() {
        let m1 = Matrix::<f64>::new(4, 4, (0..16).map(f64::from).collect()).unwrap();
        let err = m1.slice(3..2, 1..3).unwrap_err();
        assert!(matches!(err, TensorErrors::InvalidInterval { .. }));

        let err = m1.slice(1..3, 3..2).unwrap_err();
        assert!(matches!(err, TensorErrors::InvalidInterval { .. }));
    }

    #[test]
    fn all_pools_produce_same_result() {
        let m1 = Matrix::<f64>::new(4, 4, (0..16).map(f64::from).collect()).unwrap();
        let normal = m1.clone().pool(pool_sum_mat, (2, 2), (2, 2)).unwrap();
        let normal_mt = m1.clone().pool_mt(&pool_sum_mat, (2, 2), (2, 2)).unwrap();
        let indexed = m1.clone().pool_indexed(|_, s| pool_sum_mat(s), (2, 2), (2, 2)).unwrap();
        let indexed_mt = m1.clone().pool_indexed_mt(&|_, s| pool_sum_mat(s), (2, 2), (2, 2)).unwrap();

        assert_eq!(normal, normal_mt);
        assert_eq!(normal, indexed);
        assert_eq!(normal, indexed_mt);
    }
}
