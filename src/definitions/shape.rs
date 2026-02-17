use crate::definitions::errors::TensorErrors;
use crate::definitions::strides::Strides;
use crate::utilities::internal_functions::dot_vectors;
use std::fmt::Display;
use std::ops::Index;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Shape(pub(crate) Vec<usize>);

impl Shape {
    pub fn new(shape: Vec<usize>) -> Result<Self, TensorErrors> {
        if shape.is_empty() {
            return Err(TensorErrors::ShapeNoDimensions);
        }

        if shape.contains(&0) {
            return Err(TensorErrors::ShapeContainsZero);
        }

        Ok(Shape(shape))
    }

    /// This returns the number of dimensions of the shape
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// The size of the elements a tensor of this shape would have
    pub fn element_count(&self) -> usize {
        self.0.iter().product()
    }

    /// This gives a list of all indices that are valid for a tensor with this shape
    pub fn indices(&self) -> Vec<Vec<usize>> {
        (0..self.element_count()).map(|i| self.tensor_index(i)).collect()
    }

    /// This sets the value at a given axis
    pub fn set(&mut self, axis: usize, value: usize) -> Result<(), TensorErrors> {
        if value == 0 {
            return Err(TensorErrors::ShapeContainsZero);
        }

        if axis >= self.rank() {
            return Err(TensorErrors::AxisOutOfBounds {
                axis,
                rank: self.rank(),
            });
        }

        self.0[axis] = value;

        Ok(())
    }

    /// Gets the length at a specified axis, if it exists, otherwise returns None
    pub fn get(&self, axis: usize) -> Option<usize> {
        self.0.get(axis).copied()
    }

    /// This gives the address for a corresponding shape index
    pub fn address_of(&self, index: Vec<usize>) -> Result<usize, TensorErrors> {
        if index.len() != self.rank() {
            return Err(TensorErrors::IndicesInvalidForRank(index.len(), self.rank()));
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

    /// Computes the tensor index for a given address (also takes the shape of the tensor)
    /// E.g. for a tensor of shape (2, 3, 2), address 4 in the data would correspond to the index (0, 2, 0)
    /// and the address 11 would correspond to (1, 2, 1) etc.
    pub fn tensor_index(&self, address: usize) -> Vec<usize> {
        let mut index_vec = Vec::with_capacity(self.rank());
        let mut remainder = address;
        let strides = Strides::from_shape(self);

        for j in strides.0.iter() {
            let floored_div = remainder / j;
            index_vec.push(floored_div);
            remainder = remainder % j;
        }

        index_vec
    }
}

impl Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl TryFrom<Vec<usize>> for Shape {
    type Error = TensorErrors;

    fn try_from(shape: Vec<usize>) -> Result<Self, Self::Error> {
        Shape::new(shape)
    }
}

impl Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0.as_slice())
    }
}

#[macro_export]
/// Creates a shape from varargs of type usize
/// Assumes the arguments form a valid shape so
/// will panic! if the arguments are invalid instead
/// of returning a `Result` type
macro_rules! shape {
    ($($shape_dimensions:expr),*$(,)?) => {
        Shape::new(vec![$($shape_dimensions),*]).unwrap()
    };
}
