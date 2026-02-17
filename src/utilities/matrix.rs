use std::cmp::min;
use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::definitions::matrix_slice_mut::MatrixSliceMut;
use crate::definitions::shape::Shape;
use crate::definitions::tensor::Tensor;
use crate::definitions::traits::IntoMatrix;
use crate::definitions::transpose::Transpose;
use crate::{shape, transpose};
use num::{One, ToPrimitive, Zero};
use rand::distr::{Distribution, StandardUniform};
use rand::Fill;
use std::ops::{Add, Div, Range};
use std::sync::Arc;
use std::thread::scope;

impl<T> Matrix<T> {
    pub fn reshape(self, new_rows: usize, new_cols: usize) -> Result<Matrix<T>, TensorErrors> {
        Ok(Matrix {
            tensor: self.tensor.reshape(&shape![new_rows, new_cols])?,
            rows: new_rows,
            cols: new_cols,
        })
    }

    pub fn map<F>(self, f: impl FnMut(T) -> F) -> Matrix<F> {
        let rows = self.rows;
        let cols = self.cols;

        Matrix {
            tensor: self.tensor.map(f),
            rows,
            cols,
        }
    }
    
    pub fn map_refs<F>(&self, f: impl FnMut(&T) -> F) -> Matrix<F> {
        let rows = self.rows;
        let cols = self.cols;
        
        Matrix {
            tensor: self.tensor.map_refs(f),
            rows,
            cols
        }
    }
}

impl<T: Clone> Matrix<T> {
    /// Creates a matrix from a single value with specified shape.
    pub fn from_value(rows: usize, cols: usize, value: T) -> Result<Matrix<T>, TensorErrors> {
        Ok(Matrix {
            tensor: Tensor::from_value(&Shape::new(vec![rows, cols])?, value),
            rows,
            cols,
        })
    }

    /// Concatenates two matrices along the specified axis (0 or 1)
    pub fn concat(&self, other: &Matrix<T>, dim: usize) -> Result<Matrix<T>, TensorErrors> {
        let res = self.tensor.concat(&other.tensor, dim)?;
        let res_shape = res.shape.clone();

        Ok(Matrix {
            tensor: res,
            rows: res_shape[0],
            cols: res_shape[1],
        })
    }

    /// Gives an enumerated iter with matrix indices
    pub fn enumerated_iter(&self) -> impl Iterator<Item = ((usize, usize), T)> + use<'_, T> {
        self.tensor.enumerated_iter().map(|(i, x)| ((i[0], i[1]), x))
    }

    /// Gives a mutable enumerated iter with matrix indices
    pub fn enumerated_iter_mut(&mut self) -> impl Iterator<Item = ((usize, usize), &mut T)> + use<'_, T> {
        self.tensor.enumerated_iter_mut().map(|(i, x)| ((i[0], i[1]), x))
    }

    /// Gives an immutable cloned slice to a certain part of the matrix
    pub fn slice(&self, rows_range: Range<usize>, cols_range: Range<usize>) -> Result<Matrix<T>, TensorErrors> {
        Ok(Matrix {
            tensor: self.tensor.slice(&[rows_range.clone(), cols_range.clone()])?,
            rows: rows_range.len(),
            cols: cols_range.len(),
        })
    }

    /// Gives a mutable slice to a certain part of the matrix
    pub fn slice_mut(&mut self, rows_range: Range<usize>, cols_range: Range<usize>) -> Result<MatrixSliceMut<'_, T>, TensorErrors> {
        if rows_range.end > self.rows {
            return Err(TensorErrors::SliceIndicesOutOfBounds {
                start: rows_range.start,
                end: rows_range.end,
                length: self.rows,
                axis: 0,
            });
        }

        if rows_range.start >= rows_range.end {
            return Err(TensorErrors::InvalidNonEmptyInterval {
                max: rows_range.end as f64,
                min: rows_range.start as f64,
            })
        }

        if cols_range.end > self.cols {
            return Err(TensorErrors::SliceIndicesOutOfBounds {
                start: cols_range.start,
                end: cols_range.end,
                length: self.cols,
                axis: 1,
            });
        }

        if cols_range.start >= cols_range.end {
            return Err(TensorErrors::InvalidNonEmptyInterval {
                max: cols_range.end as f64,
                min: cols_range.start as f64,
            })
        }

        Ok(MatrixSliceMut {
            orig: self,
            start: (rows_range.start, cols_range.start),
            end: (rows_range.end, cols_range.end),
        })
    }

    /// Transpose a matrix and returns the result
    pub fn transpose(&self) -> Matrix<T> {
        self.tensor.transpose(&transpose![1, 0]).unwrap().try_into().unwrap()
    }

    /// Flips the columns of a matrix
    pub fn flip_rows(&self) -> Matrix<T> {
        let mut res = self.clone();

        for ((row, col), v) in self.enumerated_iter() {
            res[(row, self.cols - col - 1)] = v;
        }

        res
    }

    /// Flips the rows of a matrix
    pub fn flip_cols(&self) -> Matrix<T> {
        let mut res = self.clone();

        for ((row, col), v) in self.enumerated_iter() {
            res[(self.rows - row - 1, col)] = v
        }

        res
    }

    /// Flips a matrix
    pub fn flip(&self) -> Matrix<T> {
        let mut res = self.clone();

        for ((row, col), v) in self.enumerated_iter() {
            res[(self.rows - row - 1, self.cols - col - 1)] = v;
        }

        res
    }

    /// Pools a `Matrix<T>` into a `Matrix<O>` using a custom pooling function.
    /// The custom function will take a `Matrix<T>` that corresponds to the slice that the kernel covers.
    /// If the kernel is hanging over the edge of the tensor, then only the bits of the tensor that fit are included.
    /// This is reflected in the shape of the input tensor.
    /// Default functions for `max` and `avg` are given as well.
    pub fn pool<O: Clone>(
        &self,
        pool_fn: impl Fn(Matrix<T>) -> O,
        kernel_shape: (usize, usize),
        stride_shape: (usize, usize), 
        init: O,
    ) -> Result<Matrix<O>, TensorErrors> {
        if kernel_shape.0 == 0 || kernel_shape.1 == 0 || stride_shape.0 == 0 || stride_shape.1 == 0 {
            return Err(TensorErrors::ShapeContainsZero);
        }

        let res_shape = (self.rows.div_ceil(stride_shape.0), self.cols.div_ceil(stride_shape.1));
        let mut res = Matrix::<O>::from_value(res_shape.0, res_shape.1, init).unwrap();

        for (pos, val) in res.enumerated_iter_mut() {
            let start_pos = (pos.0 * stride_shape.0, pos.1 * stride_shape.1);
            let end_pos = (min(start_pos.0 + kernel_shape.0, self.rows), min(start_pos.1 + kernel_shape.1, self.cols));

            let indices = (start_pos.0..end_pos.0, start_pos.1..end_pos.1);
            let value = pool_fn(self.slice(indices.0, indices.1)?);

            *val = value;
        }

        Ok(res)
    }

    /// Pools a `Matrix<T>` into a `Matrix<O>` using a custom pooling function with the index.
    /// The custom function will take a `Matrix<T>` that corresponds to the slice that the kernel covers.
    /// If the kernel is hanging over the edge of the tensor, then only the bits of the tensor that fit are included.
    /// This is reflected in the shape of the input tensor.
    /// Default functions for `max` and `avg` are given as well.
    pub fn pool_indexed<O: Clone>(
        &self,
        pool_fn: impl Fn((usize, usize), Matrix<T>) -> O,
        kernel_shape: (usize, usize), 
        stride_shape: (usize, usize), 
        init: O,
    ) -> Result<Matrix<O>, TensorErrors> {
        if kernel_shape.0 == 0 || kernel_shape.1 == 0 || stride_shape.0 == 0 || stride_shape.1 == 0 {
            return Err(TensorErrors::ShapeContainsZero);
        }

        let res_shape = (self.rows.div_ceil(stride_shape.0), self.cols.div_ceil(stride_shape.1));
        let mut res = Matrix::<O>::from_value(res_shape.0, res_shape.1, init).unwrap();

        for (pos, val) in res.enumerated_iter_mut() {
            let start_pos = (pos.0 * stride_shape.0, pos.1 * stride_shape.1);
            let end_pos = (min(start_pos.0 + kernel_shape.0, self.rows), min(start_pos.1 + kernel_shape.1, self.cols));

            let indices = (start_pos.0..end_pos.0, start_pos.1..end_pos.1);
            let value = pool_fn(start_pos, self.slice(indices.0, indices.1)?);

            *val = value;
        }

        Ok(res)
    }
}

impl<T: Clone + Send + Sync> Matrix<T> {
    /// Concatenates two matrices along the specified axis (0 or 1)
    pub fn concat_mt(&self, other: &Matrix<T>, dim: usize) -> Result<Matrix<T>, TensorErrors> {
        let res_tensor = self.tensor.concat_mt(&other.tensor, dim)?;

        res_tensor.try_into()
    }

    /// Flips the columns of a matrix
    pub fn flip_cols_mt(&self) -> Matrix<T> {
        let self_arc = Arc::new(self);
        let mut res = self.clone();

        scope(|s| {
            for (i, chunk) in res.chunks_mut(1).enumerate() {
                let self_ref = self_arc.clone();
                let mat_index = self_arc.shape.tensor_index(i);

                s.spawn(move || {
                    chunk[0] = self_ref[(self.rows - 1 - mat_index[0], mat_index[1])].clone();
                });
            }
        });

        res
    }

    /// Flips the columns of a matrix
    pub fn flip_rows_mt(&self) -> Matrix<T> {
        let self_arc = Arc::new(self);
        let mut res = self.clone();

        scope(|s| {
            for (i, chunk) in res.chunks_mut(1).enumerate() {
                let self_ref = self_arc.clone();
                let mat_index = self_arc.shape.tensor_index(i);

                s.spawn(move || {
                    chunk[0] = self_ref[(mat_index[0], self.cols - 1 - mat_index[1])].clone();
                });
            }
        });

        res
    }

    /// Flips the columns of a matrix
    pub fn flip_mt(&self) -> Matrix<T> {
        let self_arc = Arc::new(self);
        let mut res = self.clone();

        scope(|s| {
            for (i, chunk) in res.chunks_mut(1).enumerate() {
                let self_ref = self_arc.clone();
                let mat_index = self_arc.shape.tensor_index(i);

                s.spawn(move || {
                    chunk[0] = self_ref[(self.rows - 1 - mat_index[0], self.cols - 1 - mat_index[1])].clone();
                });
            }
        });

        res
    }

    /// Transposes a `Matrix` using a multithreaded implementation and returns the result
    pub fn transpose_mt(&self) -> Matrix<T> {
        let new_shape = (self.cols, self.rows);
        let mut new_matrix = self.clone().reshape(new_shape.0, new_shape.1).unwrap();

        let elems_per_thread = 20;  // Number of elements a single thread handles

        scope(|s| {
            for (i, elem) in new_matrix.chunks_mut(elems_per_thread).enumerate() {
                s.spawn(move || {
                    for j in 0..elems_per_thread {
                        let k = i * elems_per_thread + j;
                        let new_index = &shape![new_shape.0, new_shape.1].tensor_index(k);
                        let old_index = (new_index[1], new_index[0]);

                        if old_index.0 >= self.rows || old_index.1 >= self.cols {
                            continue
                        }

                        elem[j] = self[old_index].clone();
                    }
                });
            }
        });

        new_matrix
    }

    /// Pools a `Matrix<T>` into a `Matrix<O>` using a custom pooling function.
    /// The custom function will take a `Matrix<T>` that corresponds to the slice that the kernel covers.
    /// If the kernel is hanging over the edge of the tensor, then only the bits of the tensor that fit are included.
    /// This is reflected in the shape of the input tensor.
    /// Default functions for `max` and `avg` are given as well.
    /// This is a multithreaded implementation.
    pub fn pool_mt<O: Clone + Send + Sync>(
        &self,
        pool_fn: &(impl Fn(Matrix<T>) -> O + Sync),
        kernel_shape: (usize, usize),
        stride_shape: (usize, usize), 
        init: O,
    ) -> Result<Matrix<O>, TensorErrors> {
        if kernel_shape.0 == 0 || kernel_shape.1 == 0 || stride_shape.0 == 0 || stride_shape.1 == 0 {
            return Err(TensorErrors::ShapeContainsZero);
        }

        let res_shape = (self.rows.div_ceil(stride_shape.0), self.cols.div_ceil(stride_shape.1));

        let mut result = Matrix::<O>::from_value(res_shape.0, res_shape.1, init).unwrap();

        scope(|s| {
            let res_chunks = result.chunks_mut(1);

            for (i, chunk) in res_chunks.enumerate() {
                s.spawn(move || {
                    let res_pos = &shape![res_shape.0, res_shape.1].tensor_index(i);
                    let self_pos = (res_pos[0] * stride_shape.0, res_pos[1] * stride_shape.1);
                    let self_end_pos = (
                        min(self_pos.0 + kernel_shape.0, self.rows),
                        min(self_pos.1 + kernel_shape.1, self.cols),
                    );

                    let indices = (self_pos.0..self_end_pos.0, self_pos.1..self_end_pos.1);

                    chunk[0] = pool_fn(self.slice(indices.0, indices.1).unwrap());
                });
            }
        });

        Ok(result)
    }

    /// Pools a `Matrix<T>` into a `Matrix<O>` using a custom pooling function with the index.
    /// The custom function will take a `Matrix<T>` that corresponds to the slice that the kernel covers.
    /// If the kernel is hanging over the edge of the tensor, then only the bits of the tensor that fit are included.
    /// This is reflected in the shape of the input tensor.
    /// Default functions for `max` and `avg` are given as well.
    /// This is a multithreaded implementation.
    pub fn pool_indexed_mt<O: Clone + Send + Sync>(
        &self,
        pool_fn: &(impl Fn((usize, usize), Matrix<T>) -> O + Sync),
        kernel_shape: (usize, usize),
        stride_shape: (usize, usize), 
        init: O,
    ) -> Result<Matrix<O>, TensorErrors> {
        if kernel_shape.0 == 0 || kernel_shape.1 == 0 || stride_shape.0 == 0 || stride_shape.1 == 0 {
            return Err(TensorErrors::ShapeContainsZero);
        }

        let res_shape = (self.rows.div_ceil(stride_shape.0), self.cols.div_ceil(stride_shape.1));

        let mut result = Matrix::<O>::from_value(res_shape.0, res_shape.1, init).unwrap();

        scope(|s| {
            let res_chunks = result.chunks_mut(1);

            for (i, chunk) in res_chunks.enumerate() {
                s.spawn(move || {
                    let res_pos = shape![res_shape.0, res_shape.1].tensor_index(i);
                    let self_pos = (res_pos[0] * stride_shape.0, res_pos[1] * stride_shape.1);
                    let self_end_pos = (
                        min(self_pos.0 + kernel_shape.0, self.rows),
                        min(self_pos.1 + kernel_shape.1, self.cols),
                    );

                    let indices = (self_pos.0..self_end_pos.0, self_pos.1..self_end_pos.1);

                    chunk[0] = pool_fn(self_pos, self.slice(indices.0, indices.1).unwrap());
                });
            }
        });

        Ok(result)
    }
}

impl<T: Default + Clone> Matrix<T> {
    pub fn from_shape(rows: usize, cols: usize) -> Result<Matrix<T>, TensorErrors> {
        Ok(Matrix {
            tensor: Tensor::<T>::from_shape(&Shape::new(vec![rows, cols])?),
            rows,
            cols,
        })
    }
}

impl<T: Default + Clone> Matrix<T> where StandardUniform: Distribution<T>, [T]: Fill {
    pub fn rand(rows: usize, cols: usize) -> Matrix<T> {
        Matrix {
            tensor: Tensor::<T>::rand(&shape![rows, cols]),
            rows,
            cols,
        }
    }
}

impl<T: Zero + Clone> Matrix<T> {
    pub fn zeros(rows: usize, cols: usize) -> Result<Matrix<T>, TensorErrors> {
        Ok(Matrix::from_value(rows, cols, T::zero())?)
    }
}

impl<T: Clone> IntoMatrix<T> for Vec<T> {
    /// Converts an iterator into a matrix of shape (1, length_of_iter)
    fn into_matrix(self) -> Matrix<T> {
        Matrix::new(1, self.len(), self).unwrap()
    }
}

impl<T: PartialOrd + Clone> Matrix<T> {
    /// Clips the values in the `Matrix` between [min, max\]
    pub fn clip(&self, min: T, max: T) -> Matrix<T> {
        let mut res = self.clone();

        for val in res.iter_mut() {
            if *val <= min {
                *val = min.clone();
            } else if *val >= max {
                *val = max.clone();
            }
        }

        res
    }
}

/// Constructs an identity matrix of `T` values of the given size
pub fn identity<T: Zero + One + Clone>(n: usize) -> Result<Matrix<T>, TensorErrors> {
    let mut t = Matrix::zeros(n, n)?;

    for i in 0..n {
        t[&[i, i]] = T::one();
    }

    Ok(t)
}

/// Constructs an identity matrix of `T` values of the given size
pub fn eye<T: Zero + One + Clone>(n: usize) -> Result<Matrix<T>, TensorErrors> {
    let mut t = Matrix::zeros(n, n)?;

    for i in 0..n {
        t[&[i, i]] = T::one();
    }

    Ok(t)
}

/// Default pooling function to sum the values
pub fn pool_sum_mat<T: Add<Output = T> + Clone>(m: Matrix<T>) -> T {
    m.iter().cloned().reduce(T::add).unwrap()
}

/// Default pooling function to find the minimum
pub fn pool_min_mat<T: PartialOrd + Clone>(m: Matrix<T>) -> T {
    let mut min = m.first().unwrap().clone();

    for i in m.iter() {
        if *i < min {
            min = i.clone();
        }
    }

    min
}

/// Default pooling function to find the maximum
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
/// Bear in mind the total number of elements is the total number of elements in the input,
/// so if you want the total number of elements to stay the same even for overhanging
///  input tensors then you will need to write your own version.
pub fn pool_avg_mat<T: Add<Output = T> + Div<f64, Output = T> + Clone>(m: Matrix<T>) -> T {
    let sum = pool_sum_mat(m.clone());
    let elems = m.shape().element_count().to_f64().unwrap();

    sum / elems
}
