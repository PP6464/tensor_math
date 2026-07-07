use crate::definitions::errors::TensorErrors;
use crate::definitions::errors::TensorErrors::SliceIncompatibleShape;
use crate::definitions::shape::Shape;
use crate::definitions::tensor::Tensor;
use crate::definitions::traits::IntoTensor;
use std::ops::{Index, IndexMut};

#[derive(Debug, Eq, PartialEq)]
pub struct TensorSliceMut<'a, T> {
    pub(crate) orig: &'a mut Tensor<T>,
    pub(crate) start: Vec<usize>,
    pub(crate) end: Vec<usize>,
}

impl<T> TensorSliceMut<'_, T> {
    /// Returns the rank of the tensor slice.
    pub fn rank(&self) -> usize {
        self.end.len()
    }

    /// Returns the shape of the tensor slice.
    pub fn shape(&self) -> Shape {
        Shape::new(
            self.end
                .iter()
                .zip(self.start.iter())
                .map(|(e, s)| e - s)
                .collect(),
        )
    }
    
    /// Returns the start position of the tensor slice.
    pub fn start(&self) -> &[usize] {
        &self.start
    }
    
    /// Returns the end position of the tensor slice.
    pub fn end(&self) -> &[usize] {
        &self.end
    }
}

impl<'a, T: Clone> TensorSliceMut<'a, T> {
    /// Sets all the values in the mutable slice to the given values.
    /// This fails if the shape of the values does not match the slice shape.
    pub fn set_all(&mut self, values: &Tensor<T>) -> Result<(), TensorErrors> {
        let slice_shape = Shape::new(
            self.end
                .iter()
                .zip(self.start.iter())
                .map(|(e, s)| e - s)
                .collect(),
        );

        if slice_shape != values.shape {
            return Err(SliceIncompatibleShape {
                slice_shape: self
                    .start
                    .iter()
                    .zip(self.end.iter())
                    .map(|(&x, &y)| y - x)
                    .collect::<Shape>(),
                tensor_shape: values.shape.clone(),
            });
        }

        for (index, value) in values.enumerated_iter() {
            self[index.as_slice()] = value;
        }

        Ok(())
    }

    /// Gets the value at the specified index if it exits, otherwise returns None
    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        if indices.len() != self.orig.rank() {
            return None;
        }

        let orig_index = indices
            .iter()
            .zip(self.start.iter())
            .map(|(x, y)| x + y)
            .collect::<Vec<usize>>();

        for i in 0..self.orig.rank() {
            if self.end[i] <= orig_index[i] {
                return None;
            }
        }

        self.orig.get(orig_index.as_slice())
    }
}

impl<T> Index<&[usize]> for TensorSliceMut<'_, T> {
    type Output = T;

    fn index(&self, index: &[usize]) -> &Self::Output {
        let actual_index = self
            .start
            .iter()
            .zip(index.iter())
            .enumerate()
            .map(|(i, (a, b))| {
                assert!(a + b < self.end[i]);
                a + b
            })
            .collect::<Vec<usize>>();

        &self.orig[actual_index.as_slice()]
    }
}

impl<T> IndexMut<&[usize]> for TensorSliceMut<'_, T> {
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        let actual_index = self
            .start
            .iter()
            .zip(index.iter())
            .enumerate()
            .map(|(i, (a, b))| {
                assert!(a + b < self.end[i]);
                a + b
            })
            .collect::<Vec<usize>>();

        &mut self.orig[actual_index.as_slice()]
    }
}

impl<T: Clone> IntoTensor<T> for TensorSliceMut<'_, T> {
    fn into_tensor(self) -> Tensor<T> {
        self.orig
            .slice(
                &self
                    .start
                    .iter()
                    .zip(self.end.iter())
                    .map(|(x, y)| *x..*y)
                    .collect::<Vec<_>>(),
            )
            .unwrap()
    }
}
