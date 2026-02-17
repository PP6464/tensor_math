use std::ops::{Index, IndexMut};
use crate::definitions::errors::TensorErrors;
use crate::definitions::errors::TensorErrors::SliceIncompatibleShape;
use crate::definitions::matrix::Matrix;
use crate::definitions::shape::Shape;
use crate::definitions::traits::IntoMatrix;
use crate::shape;

#[derive(Debug, Eq, PartialEq)]
pub struct MatrixSliceMut<'a, T> {
    pub(crate) orig: &'a mut Matrix<T>,
    pub(crate) start: (usize, usize),
    pub(crate) end: (usize, usize),
}

impl<'a, T: Clone> MatrixSliceMut<'a, T> {
    /// Sets all the values in the mutable slice to the values in the given input
    pub fn set_all(&mut self, values: &Matrix<T>) -> Result<(), TensorErrors> {
        if self.end.0 - self.start.0 != values.rows || self.end.1 - self.start.1 != values.cols {
            return Err(SliceIncompatibleShape {
                slice_shape: shape![self.end.0 - self.start.0, self.end.1 - self.start.1],
                tensor_shape: values.shape.clone(),
            });
        }
        
        for (index, value) in values.enumerated_iter() {
            self[index] = value
        }
        
        Ok(())
    }
    
    /// Gets the value at the specified index, returns None otherwise
    pub fn get(&self, indices: (usize, usize)) -> Option<&T> {
        let orig_index = (indices.0 + self.start.0, indices.1 + self.start.1);
        
        if orig_index.0 >= self.end.0 || orig_index.1 >= self.end.1 {
            return None;
        }
        
        self.orig.get(orig_index)
    }
}
impl<T> Index<(usize, usize)> for MatrixSliceMut<'_, T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        assert!(index.0 + self.start.0 < self.end.0);
        assert!(index.1 + self.start.1 < self.end.1);

        &self.orig[(self.start.0 + index.0, self.start.1 + index.1)]
    }
}

impl<T> Index<&[usize; 2]> for MatrixSliceMut<'_, T> {
    type Output = T;

    fn index(&self, index: &[usize; 2]) -> &Self::Output {
        assert!(self.start.0 + index[0] < self.end.0);
        assert!(self.start.1 + index[1] < self.end.1);

        &self.orig[(self.start.0 + index[0], self.start.1 + index[1])]
    }
}

impl<T> IndexMut<(usize, usize)> for MatrixSliceMut<'_, T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        assert!(index.0 + self.start.0 < self.end.0);
        assert!(index.1 + self.start.1 < self.end.1);

        &mut self.orig[(self.start.0 + index.0, self.start.1 + index.1)]
    }
}

impl<T> IndexMut<&[usize; 2]> for MatrixSliceMut<'_, T> {
    fn index_mut(&mut self, index: &[usize; 2]) -> &mut Self::Output {
        assert!(self.start.0 + index[0] < self.end.0);
        assert!(self.start.1 + index[1] < self.end.1);

        &mut self.orig[(self.start.0 + index[0], self.start.1 + index[1])]
    }
}

impl<T: Clone> IntoMatrix<T> for MatrixSliceMut<'_, T> {
    fn into_matrix(self) -> Matrix<T> {
        self
            .orig
            .slice(self.start.0..self.end.0, self.start.1..self.end.1)
            .unwrap()
    }
}
