use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::definitions::shape::Shape;
use crate::definitions::strides::Strides;
use crate::definitions::traits::{IntoTensor, TryIntoMatrix};
use crate::shape;
use crate::utilities::internal_functions::dot_vectors;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::{
    FromParallelIterator, IntoParallelIterator
    , ParallelIterator,
};
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::vec::IntoIter;
/*
--------------------------------------------
* Tensor definition
--------------------------------------------
*/

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Tensor<T> {
    pub(crate) shape: Shape,
    pub(crate) strides: Strides,
    pub(crate) elements: Vec<T>,
}

impl<T> Tensor<T> {
    /// Returns a new tensor with the specified shape and elements.
    /// This fails if the number of elements does not match `shape.element_count()`.
    pub fn new(shape: Shape, elements: Vec<T>) -> Result<Self, TensorErrors> {
        if shape.element_count() != elements.len() {
            return Err(TensorErrors::ShapeSizeDoesNotMatch);
        }

        let strides = Strides::from_shape(&shape);

        Ok(Tensor {
            shape,
            strides,
            elements,
        })
    }

    /// Returns the elements of the tensor.
    pub fn elements(&self) -> &[T] {
        &self.elements
    }

    /// Returns a mutable reference to the elements of the tensor.
    pub fn elements_mut(&mut self) -> &mut [T] {
        &mut self.elements
    }

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> Shape {
        self.shape.clone()
    }

    /// Returns the rank of the tensor.
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Gets the element at the specified indices, returning `None` if the indices are out of bounds.
    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        if indices.len() != self.rank() {
            return None;
        }
        for i in 0..self.rank() {
            if indices[i] >= self.shape[i] {
                return None;
            }
        }
        let addr = dot_vectors(&self.strides.0, indices);
        unsafe { Some(self.elements.get_unchecked(addr)) }
    }

    /// Gets the element at the specified indices without bounds checking.
    pub(crate) unsafe fn get_unchecked(&self, indices: &[usize]) -> &T {
        self.elements
            .get_unchecked(dot_vectors(&self.strides.0, indices))
    }

    /// Gets a mutable reference to the element at the specified indices, returning `None` if the indices are out of bounds.
    pub fn get_mut(&mut self, indices: &[usize]) -> Option<&mut T> {
        if indices.len() != self.rank() {
            return None;
        }
        for i in 0..self.rank() {
            if indices[i] >= self.shape[i] {
                return None;
            }
        }
        let addr = dot_vectors(&self.strides.0, indices);
        unsafe { Some(self.elements.get_unchecked_mut(addr)) }
    }

    /// Gets a mutable reference to the element at the specified indices without bounds checking.
    pub(crate) unsafe fn get_unchecked_mut(&mut self, indices: &[usize]) -> &mut T {
        self.elements
            .get_unchecked_mut(dot_vectors(&self.strides.0, indices))
    }

    /// Consumes the tensor and returns an iterator over its elements.
    pub fn into_iter(self) -> IntoIter<T> {
        self.elements.into_iter()
    }

    /// Consumes the tensor and returns a parallel iterator over its elements.
    pub fn into_par_iter(self) -> impl ParallelIterator<Item = T>
    where
        T: Send + Sync,
    {
        self.elements.into_par_iter()
    }
}

/*
--------------------------------------------
* Tensor indexing
--------------------------------------------
*/

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
        unsafe { self.elements.get_unchecked(addr) }
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
        unsafe { self.elements.get_unchecked_mut(addr) }
    }
}

/*
--------------------------------------------
* Conversion between Matrix and Tensor
--------------------------------------------
*/

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
* Conversion from Vector
--------------------------------------------
*/

impl<T> IntoTensor<T> for Vec<T> {
    fn into_tensor(self) -> Tensor<T> {
        Tensor {
            shape: shape![self.len()],
            strides: Strides(vec![1]),
            elements: self,
        }
    }
}

/*
--------------------------------------------
* Collection from iterators
--------------------------------------------
*/

impl<T> FromIterator<T> for Tensor<T> {
    /// Converts an iterator into a tensor of shape `(iter.len())`
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let elements: Vec<T> = iter.into_iter().collect();
        Tensor {
            shape: shape![elements.len()],
            strides: Strides(vec![1]),
            elements,
        }
    }
}

impl<T: Send> FromParallelIterator<T> for Tensor<T> {
    /// Converts a parallel iterator into a tensor of shape `(iter.len())`
    fn from_par_iter<I>(par_iter: I) -> Self
    where
        I: IntoParallelIterator<Item = T>,
    {
        let elements = par_iter.into_par_iter().collect::<Vec<_>>();
        Tensor {
            shape: shape![elements.len()],
            strides: Strides(vec![1]),
            elements,
        }
    }
}

/*
--------------------------------------------
* Deref and DerefMut implementations
--------------------------------------------
*/

impl<T> Deref for Tensor<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.elements()
    }
}

impl<T> DerefMut for Tensor<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.elements_mut()
    }
}
