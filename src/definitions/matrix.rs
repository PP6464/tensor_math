use crate::definitions::chunk::{Chunk, ChunkMut};
use crate::definitions::errors::TensorErrors;
use crate::definitions::shape::Shape;
use crate::definitions::strides::Strides;
use crate::definitions::tensor::Tensor;
use crate::definitions::traits::{IntoTensor, MatrixLike, MatrixLikeMut, TryIntoMatrix};
use crate::{mat_addr, shape};
use rayon::iter::{FromParallelIterator, IntoParallelIterator};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use rayon::slice::{ParallelSlice, ParallelSliceMut};
use std::ops::{Deref, DerefMut, Index, IndexMut};

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

    /// Returns the underlying elements
    pub fn elements(&self) -> &[T] {
        &self.elements
    }

    /// Returns a mutable reference to the underlying elements
    pub fn elements_mut(&mut self) -> &mut [T] {
        &mut self.elements
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

impl<T> TryIntoMatrix<T> for Tensor<T> {
    type Error = TensorErrors;

    fn try_into_matrix(self) -> Result<Matrix<T>, TensorErrors> {
        if self.shape.rank() != 2 {
            return Err(TensorErrors::RanksDoNotMatch(self.shape.rank(), 2));
        }

        Ok(Matrix {
            elements: self.elements,
            rows: self.shape[0],
            cols: self.shape[1],
        })
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
    /// Converts a parallel iterator into a matrix of shape `(1, iter.len())`
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

/*
--------------------------------------------
* Matrix-like trait implementations
--------------------------------------------
*/

impl<T> MatrixLike<T> for Matrix<T> {
    fn shape(&self) -> Shape {
        shape![self.rows, self.cols]
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn is_square(&self) -> bool {
        self.cols == self.rows
    }

    fn get(&self, indices: (usize, usize)) -> Option<&T> {
        if indices.0 >= self.rows || indices.1 >= self.cols {
            return None;
        }

        unsafe { Some(self.elements.get_unchecked(mat_addr!(indices, self.cols))) }
    }

    unsafe fn get_unchecked(&self, indices: (usize, usize)) -> &T {
        self.elements.get_unchecked(mat_addr!(indices, self.cols))
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T>
    where
        T: 'a,
    {
        self.elements.iter()
    }

    fn par_iter<'a>(&'a self) -> impl IndexedParallelIterator<Item = &'a T>
    where
        T: 'a + Send + Sync,
    {
        self.elements().par_iter()
    }

    fn chunks<'a>(&'a self, n: usize) -> impl Iterator<Item = Chunk<'a, T>>
    where
        T: 'a,
    {
        self.elements.chunks(n).map(Chunk::Contiguous)
    }

    fn par_chunks<'a>(&'a self, n: usize) -> impl IndexedParallelIterator<Item = Chunk<'a, T>>
    where
        T: Send + Sync + 'a,
    {
        self.elements.par_chunks(n).map(Chunk::Contiguous)
    }
}

impl<T> MatrixLikeMut<T> for Matrix<T> {
    fn get_mut(&mut self, indices: (usize, usize)) -> Option<&mut T> {
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

    unsafe fn get_unchecked_mut(&mut self, indices: (usize, usize)) -> &mut T {
        self.elements
            .get_unchecked_mut(mat_addr!(indices, self.cols))
    }

    fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &mut T>
    where
        T: 'a,
    {
        self.elements_mut().iter_mut()
    }

    fn par_iter_mut<'a>(&'a mut self) -> impl IndexedParallelIterator<Item = &'a mut T>
    where
        T: 'a + Send + Sync,
    {
        self.elements_mut().par_iter_mut()
    }

    fn chunks_mut<'a>(&'a mut self, n: usize) -> impl Iterator<Item = ChunkMut<'a, T>>
    where
        T: 'a,
    {
        self.elements.chunks_mut(n).map(ChunkMut::Contiguous)
    }

    fn par_chunks_mut<'a>(
        &'a mut self,
        n: usize,
    ) -> impl IndexedParallelIterator<Item = ChunkMut<'a, T>>
    where
        T: Send + Sync + 'a,
    {
        self.elements.par_chunks_mut(n).map(ChunkMut::Contiguous)
    }
}
