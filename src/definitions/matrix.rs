use crate::definitions::chunk::{Chunk, ChunkMut};
use crate::definitions::errors::TensorErrors;
use crate::definitions::shape::Shape;
use crate::definitions::tensor::Tensor;
use crate::definitions::traits::{IntoTensor, MatrixLike, MatrixLikeMut};
use crate::shape;
use rayon::iter::{FromParallelIterator, IntoParallelIterator};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::slice::Iter;
use std::vec::IntoIter;

/// This struct represents a matrix, i.e. a rank 2 tensor.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Matrix<T> {
    pub(crate) tensor: Tensor<T>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

impl<T> Matrix<T> {
    /// Returns a new matrix with the given rows and columns and specified elements.
    /// This fails if `elements.len() != rows * cols`.
    pub fn new(rows: usize, cols: usize, elements: Vec<T>) -> Result<Matrix<T>, TensorErrors> {
        Ok(Matrix {
            tensor: Tensor::new(&shape![rows, cols], elements)?,
            rows,
            cols,
        })
    }

    /// Returns the shape of the matrix.
    pub fn shape(&self) -> Shape {
        shape![self.rows, self.cols]
    }

    /// Returns the number of rows of the matrix.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns of the matrix.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Gets the element at an index if it is in bounds, otherwise returns None.
    pub fn get(&self, indices: (usize, usize)) -> Option<&T> {
        self.tensor.get(&[indices.0, indices.1])
    }

    /// Gets a mutable reference to the element at an index if it is in bounds,
    /// otherwise returns None.
    pub fn get_mut(&mut self, indices: (usize, usize)) -> Option<&mut T> {
        self.tensor.get_mut(&[indices.0, indices.1])
    }
}

impl<T> IntoTensor<T> for Matrix<T> {
    fn into_tensor(self) -> Tensor<T> {
        self.tensor
    }
}

impl<T> TryFrom<Tensor<T>> for Matrix<T> {
    type Error = TensorErrors;

    /// Converts the tensor into a matrix.
    /// This fails if `tensor.rank() != 2`.
    fn try_from(tensor: Tensor<T>) -> Result<Self, Self::Error> {
        if tensor.rank() != 2 {
            return Err(TensorErrors::ShapesIncompatible);
        }

        Ok(Matrix {
            rows: tensor.shape[0],
            cols: tensor.shape[1],
            tensor,
        })
    }
}

impl<T> Index<&[usize; 2]> for Matrix<T> {
    type Output = T;

    fn index(&self, index: &[usize; 2]) -> &Self::Output {
        &self.tensor[index]
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.tensor[&[index.0, index.1]]
    }
}

impl<T> IndexMut<&[usize; 2]> for Matrix<T> {
    fn index_mut(&mut self, index: &[usize; 2]) -> &mut Self::Output {
        &mut self.tensor[index]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.tensor[&[index.0, index.1]]
    }
}

impl<T> Deref for Matrix<T> {
    type Target = Tensor<T>;

    fn deref(&self) -> &Self::Target {
        &self.tensor
    }
}

impl<T> DerefMut for Matrix<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.tensor
    }
}

impl<T: Default + Clone> Default for Matrix<T> {
    /// Returns a single-element matrix with the single element being `T::default()`.
    fn default() -> Self {
        Matrix {
            tensor: Tensor::<T>::default().reshape(&shape![1, 1]).unwrap(),
            rows: 1,
            cols: 1,
        }
    }
}

impl<T> IntoIterator for Matrix<T> {
    type Item = T;
    type IntoIter = IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.tensor.into_iter()
    }
}

impl<T> FromIterator<T> for Matrix<T> {
    /// Converts an iterator into a matrix of shape `(1, iter.len())`.
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let elems = iter.into_iter().collect::<Vec<_>>();

        Matrix {
            rows: 1,
            cols: elems.len(),
            tensor: Tensor::new(&shape![1, elems.len()], elems).unwrap(),
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
        Matrix::new(1, elements.len(), elements).unwrap()
    }
}

impl<'a, T: Clone> From<Iter<'a, T>> for Matrix<T> {
    /// Converts an iterator into a matrix of shape `(1, value.len())`.
    fn from(value: Iter<'a, T>) -> Self {
        let elements: Vec<T> = value.map(|x| x.clone()).collect();
        Matrix {
            rows: 1,
            cols: elements.len(),
            tensor: Tensor::new(&shape![1, elements.len()], elements).unwrap(),
        }
    }
}

impl<T> MatrixLike<T> for Matrix<T> {
    fn shape(&self) -> Shape {
        self.shape()
    }

    fn rows(&self) -> usize {
        self.rows()
    }

    fn cols(&self) -> usize {
        self.cols()
    }

    fn is_square(&self) -> bool {
        self.cols == self.rows
    }

    fn get(&self, indices: (usize, usize)) -> Option<&T> {
        self.get(indices)
    }

    unsafe fn get_unchecked(&self, indices: (usize, usize)) -> &T {
        self.elements.get_unchecked(indices.0 * self.rows + indices.1)
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T>
    where
        T: 'a,
    {
        self.elements.iter()
    }

    fn par_iter<'a>(&'a self) -> impl IndexedParallelIterator<Item=&'a T>
    where
        T: 'a + Send + Sync
    {
        self.elements().par_iter()
    }

    fn chunks<'a>(&'a self, n: usize) -> impl Iterator<Item=Chunk<'a, T>>
    where
        T: 'a
    {
        self.elements.chunks(n).map(Chunk::Contiguous)
    }

    fn par_chunks<'a>(&'a self, n: usize) -> impl IndexedParallelIterator<Item=Chunk<'a, T>>
    where
        T: Send + Sync + 'a
    {
        self.elements.par_chunks(n).map(Chunk::Contiguous)
    }
}

impl<T> MatrixLikeMut<T> for Matrix<T> {
    fn get_mut(&mut self, indices: (usize, usize)) -> Option<&mut T> {
        self.get_mut(indices)
    }

    unsafe fn get_unchecked_mut(&mut self, indices: (usize, usize)) -> &mut T {
        let flat_index = indices.0 * self.rows + indices.1;
        self.elements.get_unchecked_mut(flat_index)
    }

    fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &mut T>
    where
        T: 'a,
    {
        self.elements_mut().iter_mut()
    }

    fn par_iter_mut<'a>(&'a mut self) -> impl IndexedParallelIterator<Item=&'a mut T>
    where
        T: 'a + Send + Sync
    {
        self.elements_mut().par_iter_mut()
    }

    fn chunks_mut<'a>(&'a mut self, n: usize) -> impl Iterator<Item=ChunkMut<'a, T>>
    where
        T: 'a
    {
        self.elements.chunks_mut(n).map(ChunkMut::Contiguous)
    }

    fn par_chunks_mut<'a>(&'a mut self, n: usize) -> impl IndexedParallelIterator<Item=ChunkMut<'a, T>>
    where
        T: Send + Sync + 'a
    {
        self.elements.par_chunks_mut(n).map(ChunkMut::Contiguous)
    }
}
