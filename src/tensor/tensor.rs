use std::ops::{Add, Index, IndexMut, Mul};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ShapeValidationError {
    #[error("Shape vector contains 0")]
    ShapeContainsZero,
    #[error("Shape vector has no dimensions")]
    ShapeNoDimensions,
}

#[derive(Debug, Error)]
pub enum TensorCreationErrors {
    #[error("Shape vector indicates a size different to that of the data provided")]
    ShapeSizeDoesNotMatch,
}

pub(crate) fn dot_vectors<T: Add<Output = T> + Mul<Output = T> + Clone>(vec1: &Vec<T>, vec2: &Vec<T>) -> T
{
    vec1.iter()
        .cloned()
        .zip(vec2.iter().cloned())
        .map(|(x, y)| x * y)
        .reduce(T::add)
        .unwrap()
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Shape(pub(crate) Vec<usize>);
impl Shape {
    pub fn new(shape: Vec<usize>) -> Result<Self, ShapeValidationError> {
        if shape.is_empty() {
            return Err(ShapeValidationError::ShapeNoDimensions);
        }

        if shape.contains(&0) {
            return Err(ShapeValidationError::ShapeContainsZero);
        }

        Ok(Shape(shape))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// The size of the data a tensor of this shape would have
    pub fn data_len(&self) -> usize {
        self.0.iter().product()
    }
}
impl Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
impl IndexMut<usize> for Shape {
    fn index_mut(&mut self, index: usize) -> &mut <Self as Index<usize>>::Output {
        &mut self.0[index]
    }
}
impl TryFrom<Vec<usize>> for Shape {
    type Error = ShapeValidationError;

    fn try_from(shape: Vec<usize>) -> Result<Self, Self::Error> {
        Shape::new(shape)
    }
}

/// Indexing products are effectively cache to help index faster
/// For example, if I want the value at (1, 0, 3) for a tensor of shape (5, 2, 4)
/// Then the index in data would be: 1 * (2 * 4) + 0 * (4) + 3
/// So you could make an index_products vector, which would be:
/// index_products = [2*4,4,1\]; (just put a 1 at the end)
/// so that all you have to do is:
/// addr = dot_vectors(index_vector, index_products)
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct IndexProducts(pub(crate) Vec<usize>);
impl IndexProducts {
    pub(crate) fn from_shape(shape: &Shape) -> IndexProducts {
        let mut index_products: Vec<usize> = vec![1; shape.len()];

        for i in 1..shape.len() {
            let current_index = shape.len() - 1 - i;
            index_products[current_index] =
                shape[current_index + 1] * index_products[current_index + 1];
        }

        IndexProducts(index_products)
    }
}
impl Into<Vec<usize>> for IndexProducts {
    fn into(self) -> Vec<usize> {
        self.0
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Tensor<T> {
    shape: Shape,
    index_products: IndexProducts,
    data: Vec<T>,
}
impl<T: Clone> Tensor<T> {
    pub fn from_value(shape: Shape, value: T) -> Self {
        let data = vec![value; shape.data_len()];
        Tensor::new(shape, data).unwrap()
    }
}
impl<T> Tensor<T> {
    pub fn new(shape: Shape, data: Vec<T>) -> Result<Self, TensorCreationErrors> {
        if shape.data_len() != data.len() {
            return Err(TensorCreationErrors::ShapeSizeDoesNotMatch);
        }

        let index_products = IndexProducts::from_shape(&shape);

        Ok(Tensor {
            shape,
            index_products,
            data,
        })
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }
}
impl<T: Default + Clone> Tensor<T> {
    pub fn from_shape(shape: &Shape) -> Tensor<T> {
        let data = vec![T::default(); shape.data_len()];
        Tensor {
            data,
            shape: shape.clone(),
            index_products: IndexProducts::from_shape(shape),
        }
    }
}
impl<T> Index<&[usize]> for Tensor<T> {
    type Output = T;

    fn index(&self, index: &[usize]) -> &Self::Output {
        assert_eq!(
            self.shape.len(),
            index.len(),
            "Shape dimension and index dimension do not match"
        );
        for i in 0..self.shape.len() {
            if index[i] >= self.shape[i] {
                panic!("Index for dimension {i} out of bounds: index {}, shape {}", index[i], self.shape[i]);
            }
        }

        let addr = dot_vectors(&self.index_products.clone().into(), &index.to_vec());
        &self.data[addr]
    }
}
impl<T> IndexMut<&[usize]> for Tensor<T> {
    fn index_mut(&mut self, index: &[usize]) -> &mut T {
        assert_eq!(
            self.shape.len(),
            index.len(),
            "Shape dimension and index dimension do not match"
        );
        for i in 0..self.shape.len() {
            if index[i] >= self.shape[i] {
                panic!("Index for dimension {i} out of bounds: index {}, shape {}", index[i], self.shape[i]);
            }
        }

        let addr = dot_vectors(&self.index_products.clone().into(), &index.to_vec());
        &mut self.data[addr]
    }
}
