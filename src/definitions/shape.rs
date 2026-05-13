use crate::definitions::errors::TensorErrors;
use crate::definitions::strides::Strides;
use crate::utilities::internal_functions::dot_vectors;
use std::fmt::Display;
use std::ops::{Index, IndexMut};

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Shape(pub(crate) Vec<usize>);

impl Shape {
    /// Constructs a new shape.
    pub fn new(shape: Vec<usize>) -> Self {
        Shape(shape)
    }

    /// This returns the number of dimensions of the shape.
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// The number of elements a tensor of this shape should have.
    pub fn element_count(&self) -> usize {
        self.0.iter().product()
    }

    /// This gives a list of all indices that are valid for a tensor with this shape.
    pub fn indices(&self) -> Vec<Vec<usize>> {
        (0..self.element_count())
            .map(|i| self.tensor_index(i).unwrap())
            .collect()
    }

    /// Sets the value at a given axis.
    ///
    /// This fails if the axis is out of bounds.
    pub fn set(&mut self, axis: usize, value: usize) -> Result<(), TensorErrors> {
        if axis >= self.rank() {
            return Err(TensorErrors::AxisOutOfBounds {
                axis,
                rank: self.rank(),
            });
        }

        self.0[axis] = value;

        Ok(())
    }

    /// Gets the length at a specified axis, if it exists, otherwise returns None.
    pub fn get(&self, axis: usize) -> Option<usize> {
        self.0.get(axis).copied()
    }

    /// This gives the address for a corresponding shape index.
    ///
    /// This fails if the index is out of bounds.
    pub fn address(&self, index: Vec<usize>) -> Result<usize, TensorErrors> {
        if index.len() != self.rank() {
            return Err(TensorErrors::IndicesInvalidForRank(
                index.len(),
                self.rank(),
            ));
        }

        for (i, &v) in index.iter().enumerate() {
            if v >= self[i] {
                return Err(TensorErrors::IndexOutOfBounds {
                    index: v,
                    axis: i,
                    length: self[i],
                });
            }
        }

        Ok(dot_vectors(&Strides::from_shape(self).0, &index))
    }

    /// Computes the tensor index for a given address.
    ///
    /// This fails if the address is out of bounds.
    pub fn tensor_index(&self, address: usize) -> Result<Vec<usize>, TensorErrors> {
        if address >= self.element_count() {
            return Err(TensorErrors::AddressOutOfBounds(address));
        }

        let mut index_vec = Vec::with_capacity(self.rank());
        let mut remainder = address;
        let strides = Strides::from_shape(self);

        for j in strides.0.iter() {
            let floored_div = remainder / j;
            index_vec.push(floored_div);
            remainder = remainder % j;
        }

        Ok(index_vec)
    }
}

impl Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for Shape {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl FromIterator<usize> for Shape {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        Shape(Vec::from_iter(iter))
    }
}

impl From<Vec<usize>> for Shape {
    fn from(shape: Vec<usize>) -> Self {
        Shape::new(shape)
    }
}

impl Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0.as_slice())
    }
}

#[macro_export]
/// Creates a shape from varargs of type usize.
macro_rules! shape {
    ($($shape_dimensions:expr),*$(,)?) => {
        Shape::new(vec![$($shape_dimensions),*])
    };
}
