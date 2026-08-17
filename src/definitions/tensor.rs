use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::definitions::shape::Shape;
use crate::definitions::strides::Strides;
use crate::definitions::traits::{TensorLike, TensorLikeMut};
use crate::shape;
use crate::utilities::internal_functions::dot_vectors;
use rayon::iter::{FromParallelIterator, IntoParallelIterator};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::slice::Iter;
use std::vec::IntoIter;
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
use crate::definitions::chunk::{Chunk, ChunkMut};

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Tensor<T> {
    pub(crate) shape: Shape,
    pub(crate) strides: Strides,
    pub(crate) elements: Vec<T>,
}

impl<T> Tensor<T> {
    /// Returns a new tensor with the specified shape and elements.
    /// This fails if the number of elements does not match `shape.element_count()`.
    pub fn new(shape: &Shape, elements: Vec<T>) -> Result<Self, TensorErrors> {
        if shape.element_count() != elements.len() {
            return Err(TensorErrors::ShapeSizeDoesNotMatch);
        }

        let strides = Strides::from_shape(shape);

        Ok(Tensor {
            shape: shape.clone(),
            strides,
            elements,
        })
    }

    /// Gets the element at the specified element if the index is in bounds, otherwise returns None.
    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        if indices.len() != self.rank() {
            return None;
        }

        self.elements
            .get(dot_vectors(&indices.to_vec(), &self.strides.0))
    }

    /// Gets a mutable reference to the element at the specified index if it is in bounds,
    /// otherwise returns None.
    pub fn get_mut(&mut self, indices: &[usize]) -> Option<&mut T> {
        if indices.len() != self.rank() {
            return None;
        }

        self.elements
            .get_mut(dot_vectors(&indices.to_vec(), &self.strides.0))
    }

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns the rank of the tensor.
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Returns the elements of the tensor.
    pub fn elements(&self) -> &[T] {
        &self.elements
    }
    
    /// Returns a mutable reference to the elements of the tensor.
    pub fn elements_mut(&mut self) -> &mut [T] { &mut self.elements }
}

impl<T> Index<&[usize]> for Tensor<T> {
    type Output = T;

    fn index(&self, index: &[usize]) -> &Self::Output {
        assert_eq!(
            self.rank(),
            index.len(),
            "Shape dimension and index dimension do not match"
        );
        for i in 0..self.rank() {
            assert!(
                index[i] < self.shape[i],
                "Index for dimension {i} out of bounds: index {}, shape {}",
                index[i],
                self.shape[i]
            );
        }

        let addr = dot_vectors(&self.strides.clone().0, &index);
        &self.elements[addr]
    }
}

impl<T> IndexMut<&[usize]> for Tensor<T> {
    fn index_mut(&mut self, index: &[usize]) -> &mut T {
        assert_eq!(
            self.rank(),
            index.len(),
            "Shape dimension and index dimension do not match"
        );
        for i in 0..self.rank() {
            assert!(
                index[i] < self.shape[i],
                "Index for dimension {i} out of bounds: index {}, shape {}",
                index[i],
                self.shape[i]
            );
        }

        let addr = dot_vectors(&self.strides.clone().0, &index.to_vec());
        &mut self.elements[addr]
    }
}

impl<T> IntoIterator for Tensor<T> {
    type Item = T;
    type IntoIter = IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.elements.into_iter()
    }
}

impl<'a, T: Clone> From<Iter<'a, T>> for Tensor<T> {
    /// Converts an iterator into a tensor of shape `(value.len())`
    fn from(value: Iter<'a, T>) -> Self {
        let elements: Vec<T> = value.map(|x| x.clone()).collect();
        Tensor::new(&shape![elements.len()], elements).unwrap()
    }
}

impl<T> FromIterator<T> for Tensor<T> {
    /// Converts an iterator into a tensor of shape `(iter.len())`
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let elements: Vec<T> = iter.into_iter().collect();
        Tensor::new(&shape![elements.len()], elements).unwrap()
    }
}

impl<T: Send> FromParallelIterator<T> for Tensor<T> {
    /// Converts a parallel iterator into a tensor of shape `(iter.len())`
    fn from_par_iter<I>(par_iter: I) -> Self
    where
        I: IntoParallelIterator<Item = T>,
    {
        let elements: Vec<T> = par_iter.into_par_iter().collect();
        Tensor::new(&shape![elements.len()], elements).unwrap()
    }
}

impl<T> From<Matrix<T>> for Tensor<T> {
    fn from(value: Matrix<T>) -> Self {
        value.tensor
    }
}

impl<T> Deref for Tensor<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.elements.as_slice()
    }
}

impl<T> DerefMut for Tensor<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.elements.as_mut_slice()
    }
}

impl<T> TensorLike<T> for Tensor<T> {
    fn shape(&self) -> Shape {
        self.shape.clone()
    }

    fn rank(&self) -> usize {
        self.rank()
    }

    fn elements(&self) -> &[T] {
        self.elements()
    }

    fn get(&self, indices: &[usize]) -> Option<&T> {
        self.get(indices)
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
        self.elements.par_iter()
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

impl<T> TensorLikeMut<T> for Tensor<T> {
    fn elements_mut(&mut self) -> &mut [T] {
        self.elements_mut()
    }

    fn get_mut(&mut self, indices: &[usize]) -> Option<&mut T> {
        self.get_mut(indices)
    }

    fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut T>
    where
        T: 'a,
    {
        self.elements_mut().iter_mut()
    }

    fn par_iter_mut<'a>(&'a mut self) -> impl IndexedParallelIterator<Item = &'a mut T>
    where
        T: 'a + Send + Sync,
    {
        self.elements.par_iter_mut()
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
