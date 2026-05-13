use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::definitions::shape::Shape;
use crate::definitions::strides::Strides;
use crate::shape;
use crate::utilities::internal_functions::dot_vectors;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::slice::Iter;
use std::vec::IntoIter;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Tensor<T> {
    pub(crate) shape: Shape,
    pub(crate) strides: Strides,
    pub(crate) elements: Vec<T>,
}

impl<T> Tensor<T> {
    /// Returns a new tensor with the specified shape and elements.
    ///
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

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns the rank of the tensor.
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Returns the elements of the tensor.
    pub fn elements(&self) -> &Vec<T> {
        &self.elements
    }
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

        let addr = dot_vectors(&self.strides.clone().into(), &index.to_vec());
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

        let addr = dot_vectors(&self.strides.clone().into(), &index.to_vec());
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
