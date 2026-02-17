use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::slice::Iter;
use std::vec::IntoIter;
use crate::definitions::errors::TensorErrors;
use crate::definitions::shape::Shape;
use crate::definitions::tensor::Tensor;
use crate::definitions::traits::{IntoTensor};
use crate::shape;

/// This struct represents a matrix, i.e. a rank 2 tensor.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Matrix<T> {
    pub(crate) tensor: Tensor<T>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

impl<T> Matrix<T> {
    /// Returns a new matrix with the given rows and columns and specified elements if possible
    pub fn new(rows: usize, cols: usize, elements: Vec<T>) -> Result<Matrix<T>, TensorErrors> {
        Ok(Matrix {
            tensor: Tensor::new(&Shape::new(vec![rows, cols])?, elements)?,
            rows,
            cols,
        })
    }

    /// Gives whether the matrix is square or not
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// Gives the shape of the matrix
    pub fn shape(&self) -> Shape {
        shape![self.rows, self.cols]
    }

    /// Gives the number of rows of the matrix
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Gives the number of columns of the matrix
    pub fn cols(&self) -> usize {
        self.cols
    }
    
    /// Gets the element at an index if it is in bounds, otherwise returns None
    pub fn get(&self, indices: (usize, usize)) -> Option<&T> {
        self.tensor.get(&[indices.0, indices.1])
    }
}

impl<T> IntoTensor<T> for Matrix<T> {
    fn into_tensor(self) -> Tensor<T> {
        self.tensor
    }
}

impl<T> TryFrom<Tensor<T>> for Matrix<T> {
    type Error = TensorErrors;
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

impl<T> From<IntoIter<T>> for Matrix<T> {
    /// Converts an `IntoIter<T>` into a `Matrix<T>` of shape (1, length_of_iter).
    /// This will panic if the iterator is empty since the shape will contain a 0.
    fn from(value: IntoIter<T>) -> Self {
        let l = value.len();

        Matrix {
            tensor: Tensor::from(value).reshape(&shape![1, l]).unwrap(),
            rows: 1,
            cols: l,
        }
    }
}

impl<T> FromIterator<T> for Matrix<T> {
    /// Converts an `IntoIter<T>` into a `Matrix<T>` of shape (1, length_of_iter)
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let elems = iter.into_iter().collect::<Vec<_>>();

        Matrix {
            rows: 1,
            cols: elems.len(),
            tensor: Tensor::new(&shape![1, elems.len()], elems).unwrap(),
        }
    }
}

impl<'a, T: Clone> From<Iter<'a, T>> for Matrix<T> {
    /// Converts an iterator into a matrix of shape (1, length_of_iter)
    /// This will panic if the iterator is empty as the shape will contain a 0.
    fn from(value: Iter<'a, T>) -> Self {
        let elements: Vec<T> = value.map(|x| x.clone()).collect();
        Matrix {
            rows: 1,
            cols: elements.len(),
            tensor: Tensor::new(&shape![1, elements.len()], elements).unwrap(),
        }
    }
}
