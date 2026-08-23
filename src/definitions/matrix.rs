use crate::definitions::errors::TensorErrors;
use crate::definitions::shape::Shape;
use crate::definitions::strides::Strides;
use crate::definitions::tensor::Tensor;
use crate::definitions::traits::IntoTensor;
use crate::{mat_addr, shape};
use rayon::iter::{FromParallelIterator, IntoParallelIterator};
use rayon::iter::ParallelIterator;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::vec::IntoIter;
/*
--------------------------------------------
* Matrix definition
--------------------------------------------
*/

/// This struct represents a matrix, i.e. a rank 2 tensor.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Matrix<T> {
    pub(crate) elements: Vec<T>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

impl<T> Matrix<T> {
    /// Returns a new matrix with the given rows and columns and specified elements.
    /// This fails if `elements.len() != rows * cols`.
    pub fn new(rows: usize, cols: usize, elements: Vec<T>) -> Result<Matrix<T>, TensorErrors> {
        if rows * cols != elements.len() {
            return Err(TensorErrors::ShapeSizeDoesNotMatch);
        }
        Ok(Matrix {
            elements,
            rows,
            cols,
        })
    }

    /// Returns the number of rows in the matrix.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns in the matrix.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Gives the shape of the matrix.
    pub fn shape(&self) -> Shape {
        shape![self.rows, self.cols]
    }

    /// Checks if the matrix is square.
    pub fn is_square(&self) -> bool {
        self.cols == self.rows
    }

    /// Returns the underlying elements.
    pub fn elements(&self) -> &[T] {
        &self.elements
    }

    /// Returns a mutable reference to the underlying elements.
    pub fn elements_mut(&mut self) -> &mut [T] {
        &mut self.elements
    }

    /// Gets the element at the specified indices, returning `None` if the indices are out of bounds.
    pub fn get(&self, indices: (usize, usize)) -> Option<&T> {
        if indices.0 >= self.rows || indices.1 >= self.cols {
            return None;
        }

        unsafe { Some(self.elements.get_unchecked(mat_addr!(indices, self.cols))) }
    }

    /// Gets the element at the specified indices without bounds checking.
    pub(crate) unsafe fn get_unchecked(&self, indices: (usize, usize)) -> &T {
        self.elements.get_unchecked(mat_addr!(indices, self.cols))
    }

    /// Gets a mutable reference to the element at the specified indices, returning `None` if the indices are out of bounds.
    pub fn get_mut(&mut self, indices: (usize, usize)) -> Option<&mut T> {
        if indices.0 >= self.rows || indices.1 >= self.cols {
            return None;
        }

        unsafe {
            Some(
                self.elements
                    .get_unchecked_mut(mat_addr!(indices, self.cols)),
            )
        }
    }

    /// Gets a mutable reference to the element at the specified indices without bounds checking.
    pub(crate) unsafe fn get_unchecked_mut(&mut self, indices: (usize, usize)) -> &mut T {
        self.elements
            .get_unchecked_mut(mat_addr!(indices, self.cols))
    }

    /// Consumes the matrix and returns an iterator.
    pub fn into_iter(self) -> IntoIter<T> {
        self.elements.into_iter()
    }
}

/*
--------------------------------------------
* Matrix indexing
--------------------------------------------
*/

impl<T> Index<&[usize; 2]> for Matrix<T> {
    type Output = T;

    fn index(&self, index: &[usize; 2]) -> &Self::Output {
        assert!(
            index[0] < self.rows && index[1] < self.cols,
            "Indices out of bounds: {:?}",
            index
        );

        unsafe {
            self.elements
                .get_unchecked(mat_addr!((index[0], index[1]), self.cols))
        }
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        assert!(
            index.0 < self.rows && index.1 < self.cols,
            "Indices out of bounds: {:?}",
            index
        );

        unsafe { self.elements.get_unchecked(mat_addr!(index, self.cols)) }
    }
}

impl<T> IndexMut<&[usize; 2]> for Matrix<T> {
    fn index_mut(&mut self, index: &[usize; 2]) -> &mut Self::Output {
        assert!(
            index[0] < self.rows && index[1] < self.cols,
            "Indices out of bounds: {:?}",
            index
        );

        unsafe {
            self.elements
                .get_unchecked_mut(mat_addr!((index[0], index[1]), self.cols))
        }
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        assert!(
            index.0 < self.rows && index.1 < self.cols,
            "Indices out of bounds: {:?}",
            index
        );

        unsafe { self.elements.get_unchecked_mut(mat_addr!(index, self.cols)) }
    }
}

/*
--------------------------------------------
* Conversion between Matrix and Tensor
--------------------------------------------
*/

impl<T> IntoTensor<T> for Matrix<T> {
    fn into_tensor(self) -> Tensor<T> {
        Tensor {
            shape: shape![self.rows, self.cols],
            strides: Strides(vec![self.cols, 1]),
            elements: self.elements,
        }
    }
}

/*
--------------------------------------------
* Deref and DerefMut implementations
--------------------------------------------
*/

impl<T> Deref for Matrix<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.elements
    }
}

impl<T> DerefMut for Matrix<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.elements
    }
}

/*
--------------------------------------------
* Default implementation
--------------------------------------------
*/

impl<T: Default + Clone> Default for Matrix<T> {
    /// Returns a single-element matrix with the single element being `T::default()`.
    fn default() -> Self {
        Matrix {
            elements: vec![T::default()],
            rows: 1,
            cols: 1,
        }
    }
}

/*
--------------------------------------------
* Collection from iterators
--------------------------------------------
*/

impl<T> FromIterator<T> for Matrix<T> {
    /// Converts an iterator into a matrix of shape `(1, iter.len())`.
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let elems = iter.into_iter().collect::<Vec<_>>();

        Matrix {
            rows: 1,
            cols: elems.len(),
            elements: elems,
        }
    }
}

impl<T: Send> FromParallelIterator<T> for Matrix<T> {
    /// Converts a parallel iterator into a matrix of shape `(1, iter.len())`.
    fn from_par_iter<I>(par_iter: I) -> Self
    where
        I: IntoParallelIterator<Item = T>,
    {
        let elements: Vec<T> = par_iter.into_par_iter().collect();
        Matrix {
            cols: elements.len(),
            rows: 1,
            elements,
        }
    }
}
