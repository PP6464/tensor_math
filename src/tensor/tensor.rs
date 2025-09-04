use std::ops::{Add, Index, IndexMut, Mul};
use std::slice::Iter;
use std::vec::IntoIter;
use rand::distr::{Distribution, StandardUniform};
use rand::{Fill, Rng};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ShapeValidationError {
    #[error("Shape vector contains 0")]
    ShapeContainsZero,
    #[error("Shape vector has no dimensions")]
    ShapeNoDimensions,
}

#[derive(Debug, Error)]
pub enum TensorUtilErrors {
    #[error("Shape vector indicates a size different to that of the elements provided")]
    ShapeSizeDoesNotMatch,
    #[error("Shape vector is not 1 on dimension {0} so cannot flatten on this dimension")]
    DimIsNotOne(usize),
    #[error("Dimension {dim} greater than number of dimensions in shape: {max_dim}")]
    DimOutOfBounds { dim: usize, max_dim: usize },
    #[error("Dimensions are not compatible for concatenation")]
    ShapesIncompatibleForConcatenation,
}

pub(crate) fn dot_vectors<T: Add<Output = T> + Mul<Output = T> + Clone>(
    vec1: &Vec<T>,
    vec2: &Vec<T>,
) -> T {
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

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// The size of the elements a tensor of this shape would have
    pub fn element_count(&self) -> usize {
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

#[macro_export]
/// Creates a shape from varargs of type usize
/// Assumes the arguments form a valid shape so
/// will panic! if the arguments are valid instead
/// of returning a `Result` type
macro_rules! ts {
    ($($shape_dimensions:expr),*$(,)?) => {
        Shape::new(vec![$($shape_dimensions),*]).unwrap()
    };
}

/// Indexing products are effectively cache to help index faster
/// For example, if I want the value at (1, 0, 3) for a tensor of shape (5, 2, 4)
/// Then the index in elements would be: 1 * (2 * 4) + 0 * (4) + 3
/// So you could make an index_products vector, which would be:
/// index_products = [2*4,4,1\]; (just put a 1 at the end)
/// so that all you have to do is:
/// addr = dot_vectors(index_vector, index_products)
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct IndexProducts(pub(crate) Vec<usize>);
impl IndexProducts {
    pub(crate) fn from_shape(shape: &Shape) -> IndexProducts {
        let mut index_products: Vec<usize> = vec![1; shape.rank()];

        for i in 1..shape.rank() {
            let current_index = shape.rank() - 1 - i;
            index_products[current_index] =
                shape[current_index + 1] * index_products[current_index + 1];
        }

        IndexProducts(index_products)
    }
}
impl Index<usize> for IndexProducts {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
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
    elements: Vec<T>,
}
impl<T> Tensor<T> {
    pub fn new(shape: &Shape, elements: Vec<T>) -> Result<Self, TensorUtilErrors> {
        if shape.element_count() != elements.len() {
            return Err(TensorUtilErrors::ShapeSizeDoesNotMatch);
        }

        let index_products = IndexProducts::from_shape(shape);

        Ok(Tensor {
            shape: shape.clone(),
            index_products,
            elements,
        })
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn elements(&self) -> &Vec<T> {
        &self.elements
    }

    pub fn reshape(&mut self, new_shape: &Shape) -> Result<(), TensorUtilErrors> {
        if new_shape.element_count() != self.shape.element_count() {
            return Err(TensorUtilErrors::ShapeSizeDoesNotMatch);
        }

        self.shape = new_shape.clone();
        self.index_products = IndexProducts::from_shape(new_shape);
        Ok(())
    }
    pub fn flatten(&mut self, dim: usize) -> Result<(), TensorUtilErrors> {
        if dim >= self.shape.rank() {
            return Err(TensorUtilErrors::DimOutOfBounds {
                dim,
                max_dim: self.shape.rank(),
            });
        }

        if self.shape[dim] != 1 {
            return Err(TensorUtilErrors::DimIsNotOne(dim));
        }

        self.shape.0.remove(dim);
        self.index_products = IndexProducts::from_shape(&self.shape);
        Ok(())
    }
}
impl<T: Clone> Tensor<T> {
    pub fn from_value(shape: &Shape, value: T) -> Self {
        let elements = vec![value; shape.element_count()];
        Tensor::new(shape, elements).unwrap()
    }

    pub fn concat(&self, other: &Tensor<T>, dim: usize) -> Result<Tensor<T>, TensorUtilErrors> {
        if self.shape.rank() < other.shape.rank() {
            return Err(TensorUtilErrors::ShapesIncompatibleForConcatenation);
        }
        let mut resultant_shape: Vec<usize> = Vec::with_capacity(self.shape.rank());

        // Ensure shapes match on every dim that is not the dim along which to concatenate
        for i in 0..self.shape.rank() {
            if i == dim {
                resultant_shape.push(self.shape[i] + other.shape[i]);
                continue;
            }

            if self.shape[i] != other.shape[i] {
                return Err(TensorUtilErrors::ShapesIncompatibleForConcatenation);
            }

            resultant_shape.push(self.shape[i]);
        }

        let resultant_shape: Shape = resultant_shape.try_into().unwrap();
        let mut resultant_elements: Vec<T> = Vec::with_capacity(resultant_shape.element_count());

        if dim == 0 {
            // If the dimension is 0 we can just merge the elements one after another
            resultant_elements = self.elements.clone();
            resultant_elements.extend(other.elements.clone());
            return Ok(Tensor::new(&resultant_shape, resultant_elements)?);
        }

        let mut self_chunks = self.elements.chunks(self.index_products[dim - 1]);
        let mut other_chunks = other.elements.chunks(other.index_products[dim - 1]);

        // Merge together chunks from self and other in the correct manner to get
        // the result for concatenating self and other (in that order)
        // Note self_chunks and other_chunks have the same length
        // Because their shapes are the same in the dimensions to the left of the concatenation dim
        for _ in 0..self_chunks.len() {
            resultant_elements.extend_from_slice(self_chunks.next().unwrap());
            resultant_elements.extend_from_slice(other_chunks.next().unwrap());
        }

        let result = Tensor::new(&resultant_shape, resultant_elements)?;

        Ok(result)
    }
}
impl<T: Default + Clone> Tensor<T> {
    pub fn from_shape(shape: &Shape) -> Tensor<T> {
        let elements = vec![T::default(); shape.element_count()];
        Tensor {
            elements,
            shape: shape.clone(),
            index_products: IndexProducts::from_shape(shape),
        }
    }
}
/// Generate a `Tensor` full of random values of type `T`
/// For any type `T` that implements `rand::distr::Distribution`
impl<T: Default + Clone> Tensor<T> where StandardUniform: Distribution<T>, [T]: Fill {
    pub(crate) fn rand(shape: &Shape) -> Tensor<T> {
        let mut elements = vec![T::default(); shape.element_count()];
        let mut rng = rand::rng();

        rng.fill(elements.as_mut_slice());

        Tensor::new(shape, elements).unwrap()
    }
}
impl<T> Index<&[usize]> for Tensor<T> {
    type Output = T;

    fn index(&self, index: &[usize]) -> &Self::Output {
        assert_eq!(
            self.shape.rank(),
            index.len(),
            "Shape dimension and index dimension do not match"
        );
        for i in 0..self.shape.rank() {
            if index[i] >= self.shape[i] {
                panic!(
                    "Index for dimension {i} out of bounds: index {}, shape {}",
                    index[i], self.shape[i]
                );
            }
        }

        let addr = dot_vectors(&self.index_products.clone().into(), &index.to_vec());
        &self.elements[addr]
    }
}
impl<T> IndexMut<&[usize]> for Tensor<T> {
    fn index_mut(&mut self, index: &[usize]) -> &mut T {
        assert_eq!(
            self.shape.rank(),
            index.len(),
            "Shape dimension and index dimension do not match"
        );
        for i in 0..self.shape.rank() {
            if index[i] >= self.shape[i] {
                panic!(
                    "Index for dimension {i} out of bounds: index {}, shape {}",
                    index[i], self.shape[i]
                );
            }
        }

        let addr = dot_vectors(&self.index_products.clone().into(), &index.to_vec());
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
/// Converts an iterator into a `Tensor` of shape (length_of_iter)
impl<'a, T: Clone> From<Iter<'a, T>> for Tensor<T> {
    fn from(value: Iter<'a, T>) -> Self {
        let elements: Vec<T> = value.map(|x| x.clone()).collect();
        Tensor::new(&ts![elements.len()], elements).unwrap()
    }
}
/// Converts an `IntoIter<T>` into a `Tensor<T>` of shape (length_of_iter)
impl<T> From<IntoIter<T>> for Tensor<T> {
    fn from(value: IntoIter<T>) -> Self {
        let elements: Vec<T> = value.collect();
        Tensor::new(&ts![elements.len()], elements).unwrap()
    }
}