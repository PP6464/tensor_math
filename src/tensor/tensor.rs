use rand::distr::{Distribution, StandardUniform};
use rand::{Fill, Rng};
use std::ops::{Add, Deref, DerefMut, Index, IndexMut, Mul, Range};
use std::slice::Iter;
use std::thread;
use std::vec::IntoIter;
use num::Zero;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TensorErrors {
    #[error("Shape vector indicates a size different to that of the elements provided")]
    ShapeSizeDoesNotMatch,
    #[error("Shape vector is not 1 on dimension {0} so cannot flatten on this dimension")]
    DimIsNotOne(usize),
    #[error("Dimension {dim} greater than number of dimensions in shape: {max_dim}")]
    DimOutOfBounds { dim: usize, max_dim: usize },
    #[error("Shapes are not compatible")]
    ShapesIncompatible,
    #[error("Transposition permutation invalid")]
    TransposePermutationInvalid,
    #[error("Determinant is zero")]
    DeterminantZero,
    #[error("Shape vector contains 0")]
    ShapeContainsZero,
    #[error("Shape vector has no dimensions")]
    ShapeNoDimensions,
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

/// Computes the tensor index for a given address (also takes the shape of the `Tensor`)
/// E.g. for a `Tensor` of shape (2, 3, 2), address 4 in the data would correspond to the index (0, 2, 0)
/// and the address 11 would correspond to (1, 2, 1) etc.
pub fn tensor_index(address: usize, shape: &Shape) -> Vec<usize> {
    let mut index_vec = Vec::with_capacity(shape.rank());
    let mut remainder = address;
    let strides = Strides::from_shape(shape);

    for j in strides.0.iter() {
        let floored_div = remainder / j;
        index_vec.push(floored_div);
        remainder = remainder % j;
    }

    index_vec
}

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
        (0..self.element_count()).map(|i| tensor_index(i, self)).collect()
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
    type Error = TensorErrors;

    fn try_from(shape: Vec<usize>) -> Result<Self, Self::Error> {
        Shape::new(shape)
    }
}

#[macro_export]
/// Creates a shape from varargs of type usize
/// Assumes the arguments form a valid shape so
/// will panic! if the arguments are valid instead
/// of returning a `Result` type
macro_rules! shape {
    ($($shape_dimensions:expr),*$(,)?) => {
        Shape::new(vec![$($shape_dimensions),*]).unwrap()
    };
}

/// Cache the strides required to index the tensor.
/// addr = dot_vectors(index_vector, strides)
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct Strides(pub(crate) Vec<usize>);
impl Strides {
    pub(crate) fn from_shape(shape: &Shape) -> Strides {
        let mut strides: Vec<usize> = vec![1; shape.rank()];

        for i in 1..shape.rank() {
            let current_index = shape.rank() - 1 - i;
            strides[current_index] =
                shape[current_index + 1] * strides[current_index + 1];
        }

        Strides(strides)
    }
}
impl Index<usize> for Strides {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
impl Into<Vec<usize>> for Strides {
    fn into(self) -> Vec<usize> {
        self.0
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Tensor<T> {
    pub(crate) shape: Shape,
    pub(crate) strides: Strides,
    pub(crate) elements: Vec<T>,
}
#[derive(Debug, Eq, PartialEq)]
pub struct TensorSliceMut<'a, T> {
    pub(crate) orig: &'a mut Tensor<T>,
    pub(crate) start: Vec<usize>,
    pub(crate) end: Vec<usize>,
}
impl<'a, T: Clone> TensorSliceMut<'a, T> {
    /// Sets all the values in the mutable slice to the given values
    pub fn set_all(&mut self, values: &Tensor<T>) {
        let slice_shape = Shape::new(
            self.end.iter().zip(self.start.iter()).map(|(e, s)| e - s).collect(),
        ).unwrap();

        assert_eq!(slice_shape, values.shape, "Slice shape and values shape are not the same");

        for (index, value) in values.enumerated_iter() {
            self[index.as_slice()] = value;
        }
    }
}
impl<T> Index<&[usize]> for TensorSliceMut<'_, T> {
    type Output = T;

    fn index(&self, index: &[usize]) -> &Self::Output {
        let actual_index = self.start.iter().zip(index.iter()).enumerate().map(|(i, (a, b))| {
            assert!(a + b < self.end[i]);
            a + b
        }).collect::<Vec<usize>>();

        &self.orig[actual_index.as_slice()]
    }
}
impl<T> IndexMut<&[usize]> for TensorSliceMut<'_, T> {
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        let actual_index = self.start.iter().zip(index.iter()).enumerate().map(|(i, (a, b))| {
            assert!(a + b < self.end[i]);
            a + b
        }).collect::<Vec<usize>>();

        &mut self.orig[actual_index.as_slice()]
    }
}
impl<T> Tensor<T> {
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

    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn elements(&self) -> &Vec<T> {
        &self.elements
    }

    /// Reshape a `Tensor`, consuming it and returning the new one
    pub fn reshape(self, new_shape: &Shape) -> Result<Tensor<T>, TensorErrors> {
        if new_shape.element_count() != self.shape.element_count() {
            return Err(TensorErrors::ShapeSizeDoesNotMatch);
        }

        Ok(Tensor::new(new_shape, self.elements)?)
    }

    /// Reshape a `Tensor` in-place
    pub fn reshape_in_place(&mut self, new_shape: &Shape) -> Result<(), TensorErrors> {
        if new_shape.element_count() != self.shape.element_count() {
            return Err(TensorErrors::ShapeSizeDoesNotMatch);
        }

        self.shape = new_shape.clone();
        self.strides = Strides::from_shape(new_shape);
        Ok(())
    }

    /// Flatten a `Tensor` on a given dimension, consuming it and returning the new one
    pub fn flatten(self, dim: usize) -> Result<Tensor<T>, TensorErrors> {
        if dim >= self.rank() {
            return Err(TensorErrors::DimOutOfBounds {
                dim,
                max_dim: self.rank(),
            });
        }

        if self.shape[dim] != 1 {
            return Err(TensorErrors::DimIsNotOne(dim));
        }

        let mut copy = self.shape;
        copy.0.remove(dim);
        Ok(Tensor::new(&copy, self.elements)?)
    }

    /// Flatten a `Tensor` on a given dimension in-place
    pub fn flatten_in_place(&mut self, dim: usize) -> Result<(), TensorErrors> {
        if dim >= self.rank() {
            return Err(TensorErrors::DimOutOfBounds {
                dim,
                max_dim: self.rank(),
            });
        }

        if self.shape[dim] != 1 {
            return Err(TensorErrors::DimIsNotOne(dim));
        }

        self.shape.0.remove(dim);
        self.strides = Strides::from_shape(&self.shape);
        Ok(())
    }

    /// Apply a transformation to a `Tensor` element-wise, consuming the original and returning the result
    pub fn transform_elementwise<O>(self, closure: impl FnMut(T) -> O) -> Tensor<O> {
        let shape = self.shape().clone();
        let new_elements = self.elements.into_iter().map(closure).collect();
        Tensor::new(&shape, new_elements).unwrap()
    }
}
impl<T: Clone> Tensor<T> {
    /// Creates a tensor from a single value with specified shape
    pub fn from_value(shape: &Shape, value: T) -> Self {
        let elements = vec![value; shape.element_count()];
        Tensor::new(shape, elements).unwrap()
    }

    /// Concatenates a `Tensor` with another `Tensor` along the specified dimension
    pub fn concat(&self, other: &Tensor<T>, dim: usize) -> Result<Tensor<T>, TensorErrors> {
        if self.rank() < other.rank() {
            return Err(TensorErrors::ShapesIncompatible);
        }
        let mut resultant_shape: Vec<usize> = Vec::with_capacity(self.rank());

        // Ensure shapes match on every dim that is not the dim along which to concatenate
        for i in 0..self.rank() {
            if i == dim {
                resultant_shape.push(self.shape[i] + other.shape[i]);
                continue;
            }

            if self.shape[i] != other.shape[i] {
                return Err(TensorErrors::ShapesIncompatible);
            }

            resultant_shape.push(self.shape[i]);
        }

        let resultant_shape: Shape = resultant_shape.try_into()?;
        let mut resultant_elements: Vec<T> = Vec::with_capacity(resultant_shape.element_count());

        if dim == 0 {
            // If the dimension is 0 we can just merge the elements one after another
            resultant_elements = self.elements.clone();
            resultant_elements.extend(other.elements.clone());
            return Ok(Tensor::new(&resultant_shape, resultant_elements)?);
        }

        let mut self_chunks = self.elements.chunks(self.strides[dim - 1]);
        let mut other_chunks = other.elements.chunks(other.strides[dim - 1]);

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

    /// Give an iterable that is enumerated with tensor indices
    pub fn enumerated_iter(&self) -> impl Iterator<Item = (Vec<usize>, T)> + '_ {
        self.elements
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, element)| (tensor_index(index, &self.shape), element))
    }

    /// Give a mutable iterable that is enumerated with tensor indices
    pub fn enumerated_iter_mut(&mut self) -> impl Iterator<Item = (Vec<usize>, &mut T)> + '_  {
        self.elements
            .iter_mut()
            .enumerate()
            .map(|(index, element)| (tensor_index(index, &self.shape), element))
    }

    /// Gives a cloned immutable slice to a region in the tensor specified
    /// by an array of range of indices for each dimension of the tensor
    pub fn slice(&self, indices: &[Range<usize>]) -> Tensor<T> {
        assert_eq!(indices.len(), self.rank(), "Slice must have the same number of ranges as the rank of the tensor");
        for (i, range) in indices.iter().enumerate() {
            assert!(range.end <= self.shape[i], "Index range for dimension {i} out of bounds: size = {}, range = {}..{}", self.shape[i], range.start, range.end);
        }

        let start = indices.iter().map(|range| range.start).collect::<Vec<usize>>();

        let res_shape = Shape::new(indices.iter().map(|r| { r.end - r.start }).collect()).unwrap();
        let mut res = Tensor::<T>::from_value(&res_shape, self.first().unwrap().clone());

        for (pos, val) in res.enumerated_iter_mut() {
            let orig_index = pos.iter().zip(start.iter()).map(|(x, y)| x + y).collect::<Vec<usize>>();

            *val = self[&orig_index.as_slice()].clone();
        }

        res
    }

    /// Gives a mutable slice to a tensor in the specified range of indices.
    pub fn slice_mut(&'_ mut self, indices: &[Range<usize>]) -> TensorSliceMut<'_, T> {
        assert_eq!(indices.len(), self.rank(), "Slice must have the same number of ranges as the rank of the tensor");
        for (i, range) in indices.iter().enumerate() {
            assert!(range.end <= self.shape[i], "Index range for dimension {i} out of bounds: size = {}, range = {}..{}", self.shape[i], range.start, range.end);
        }

        let start = indices.iter().map(|range| range.start).collect::<Vec<usize>>();
        let end = indices.iter().map(|range| range.end).collect::<Vec<usize>>();

        TensorSliceMut {
            start,
            end,
            orig: self,
        }
    }
}
impl<T: Clone + Send + Sync> Tensor<T> {
    /// Concatenates a `Tensor` with another `Tensor` along the specified dimension.
    /// This spawns multiple threads to handle writing to the result at the same time.
    pub fn concat_mt(&self, other: &Tensor<T>, dim: usize) -> Result<Tensor<T>, TensorErrors> {
        if self.rank() < other.rank() {
            return Err(TensorErrors::ShapesIncompatible);
        }
        let mut resultant_shape: Vec<usize> = Vec::with_capacity(self.rank());

        // Ensure shapes match on every dim that is not the dim along which to concatenate
        for i in 0..self.rank() {
            if i == dim {
                resultant_shape.push(self.shape[i] + other.shape[i]);
                continue;
            }

            if self.shape[i] != other.shape[i] {
                return Err(TensorErrors::ShapesIncompatible);
            }

            resultant_shape.push(self.shape[i]);
        }

        let resultant_shape: Shape = resultant_shape.try_into()?;

        if dim == 0 {
            // If the dimension is 0 we can just merge the elements one after another
            let mut res_elems = self.elements.clone();
            res_elems.extend(other.elements.clone());
            return Ok(Tensor::new(&resultant_shape, res_elems)?);
        }

        let mut res_elems = vec![self.first().unwrap().clone(); resultant_shape.element_count()];

        let self_chunk_size = self.strides[dim - 1];
        let other_chunk_size = other.strides[dim - 1];

        let chunks_per_thread = 5; // The number of chunks a single thread manages

        // Merge together chunks from self and other in the correct manner to get
        // the result for concatenating self and other (in that order)
        // Note self_chunks and other_chunks have the same length
        // Because their shapes are the same in the dimensions to the left of the concatenation dim

        thread::scope(|s| {
            let thread_chunk_size_self = self_chunk_size * chunks_per_thread;
            let thread_chunk_size_other = other_chunk_size * chunks_per_thread;

            let thread_chunks_self = self.elements.chunks(thread_chunk_size_self);
            let thread_chunks_other = other.elements.chunks(thread_chunk_size_other);

            let thread_chunks = thread_chunks_self.zip(thread_chunks_other);

            let mut res_elem_chunks = res_elems.chunks_mut(thread_chunk_size_self + thread_chunk_size_other);

            for (self_thread_chunk, other_thread_chunk) in thread_chunks {
                let res_chunk = res_elem_chunks.next().unwrap();

                s.spawn(|| {
                    let mut self_chunks = self_thread_chunk.chunks(self_chunk_size);
                    let mut other_chunks = other_thread_chunk.chunks(other_chunk_size);
                    let actual_count = self_chunks.len();  // On the last chunk it may not have as many as chunks_per_thread chunks
                    
                    for j in 0..actual_count {
                        let current_self_chunk = self_chunks.next().unwrap();
                        let current_other_chunk = other_chunks.next().unwrap();

                        let start = j * (self_chunk_size + other_chunk_size);

                        for (k, self_val) in current_self_chunk.iter().enumerate() {
                            res_chunk[start + k] = self_val.clone();
                        }

                        for (k, other_val) in current_other_chunk.iter().enumerate() {
                            res_chunk[start + self_chunk_size + k] = other_val.clone();
                        }
                    }
                });
            }
        });

        let result = Tensor::new(&resultant_shape, res_elems)?;
        Ok(result)
    }
}
impl<T: Default + Clone> Tensor<T> {
    pub fn from_shape(shape: &Shape) -> Tensor<T> {
        let elements = vec![T::default(); shape.element_count()];
        Tensor {
            elements,
            shape: shape.clone(),
            strides: Strides::from_shape(shape),
        }
    }
}
/// Generate a `Tensor` full of random values of type `T`
/// For any type `T` that implements `rand::distr::Distribution`
impl<T: Default + Clone> Tensor<T> where StandardUniform: Distribution<T>, [T]: Fill {
    pub fn rand(shape: &Shape) -> Tensor<T> {
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
            self.rank(),
            index.len(),
            "Shape dimension and index dimension do not match"
        );
        for i in 0..self.rank() {
            if index[i] >= self.shape[i] {
                panic!(
                    "Index for dimension {i} out of bounds: index {}, shape {}",
                    index[i], self.shape[i]
                );
            }
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
            if index[i] >= self.shape[i] {
                panic!(
                    "Index for dimension {i} out of bounds: index {}, shape {}",
                    index[i], self.shape[i]
                );
            }
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
/// Converts an iterator into a `Tensor` of shape (length_of_iter)
impl<'a, T: Clone> From<Iter<'a, T>> for Tensor<T> {
    fn from(value: Iter<'a, T>) -> Self {
        let elements: Vec<T> = value.map(|x| x.clone()).collect();
        Tensor::new(&shape![elements.len()], elements).unwrap()
    }
}
/// Converts an `IntoIter<T>` into a `Tensor<T>` of shape (length_of_iter)
impl<T> From<IntoIter<T>> for Tensor<T> {
    fn from(value: IntoIter<T>) -> Self {
        let elements: Vec<T> = value.collect();
        Tensor::new(&shape![elements.len()], elements).unwrap()
    }
}
/// Converts an `Iterator<T>` into a `Tensor<T>` of shape (length_of_iter)
impl<T> FromIterator<T> for Tensor<T> {
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
    fn deref_mut(&mut self) -> &mut Self::Target { self.elements.as_mut_slice() }
}
/// This trait allows you to specify that something can be infallibly converted into a type of `Tensor<T>`.
/// It has the `into_tensor` method that converts the value into a `Tensor<T>`, consuming it. Bear in mind
/// that this does then automatically derive an implementation for `TryIntoTensor`.
pub trait IntoTensor<T> {
    fn into_tensor(self) -> Tensor<T>;
}
/// This trait allows you to specify that something can be fallibly converted into a type of `Tensor<T>`.
/// It has the `try_into_tensor` method that attempts to convert the value into a `Tensor<T>`, consuming it
/// and returning an error value if not possible.
pub trait TryIntoTensor<T> {
    type Error;

    fn try_into_tensor(self) -> Result<Tensor<T>, Self::Error>;
}
impl<T, O: IntoTensor<T>> TryIntoTensor<T> for O {
    type Error = ();

    fn try_into_tensor(self) -> Result<Tensor<T>, Self::Error> {
        Ok(self.into_tensor())
    }
}
impl<T: Clone> IntoTensor<T> for TensorSliceMut<'_, T> {
    fn into_tensor(self) -> Tensor<T> {
        self.orig.slice(self.start.iter().zip(self.end.iter()).map(|(x, y)| *x..*y).collect::<Vec<Range<usize>>>().as_slice())
    }
}
/// This converts the vector into a 1-d tensor
impl<T: Clone> IntoTensor<T> for Vec<T> {
    fn into_tensor(self) -> Tensor<T> {
        self.iter().into()
    }
}
impl<T: Default + Clone> Default for Tensor<T> {
    fn default() -> Self {
        Tensor::from_shape(&shape![1])
    }
}
impl<T: Zero + Clone> Tensor<T> {
    pub fn zeros(shape: &Shape) -> Tensor<T> {
        Tensor::from_value(shape, T::zero())
    }
}

/// This struct represents a matrix, i.e. a rank 2 tensor.
/// The underlying tensor is just `matrix.tensor`
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Matrix<T> {
    pub(crate) tensor: Tensor<T>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
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
impl<T> Matrix<T> {
    pub fn new(rows: usize, cols: usize, elements: Vec<T>) -> Result<Matrix<T>, TensorErrors> {
        Ok(Matrix {
            tensor: Tensor::new(&Shape::new(vec![rows, cols])?, elements)?,
            rows,
            cols,
        })
    }

    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    pub fn shape(&self) -> Shape {
        shape![self.rows, self.cols]
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn reshape(self, new_rows: usize, new_cols: usize) -> Result<Matrix<T>, TensorErrors> {
        Ok(Matrix {
            tensor: self.tensor.reshape(&shape![new_rows, new_cols])?,
            rows: new_rows,
            cols: new_cols,
        })
    }

    pub fn reshape_in_place(&mut self, new_rows: usize, new_cols: usize) -> Result<(), TensorErrors> {
        self.tensor.reshape_in_place(&shape![new_rows, new_cols])?;
        self.rows = new_rows;
        self.cols = new_cols;

        Ok(())
    }

    pub fn transform_elementwise<F>(self, f: impl FnMut(T) -> F) -> Matrix<F> {
        let rows = self.rows;
        let cols = self.cols;

        Matrix {
            tensor: self.tensor.transform_elementwise(f),
            rows,
            cols,
        }
    }
}
impl<T: Clone> Matrix<T> {
    /// Creates a matrix from a single value with specified shape.
    /// Will panic if one of rows or cols is 0
    pub fn from_value(rows: usize, cols: usize, value: T) -> Matrix<T> {
        Matrix {
            tensor: Tensor::from_value(&shape![rows, cols], value),
            rows,
            cols,
        }
    }

    /// Concatenates two matrices along the specified dim (0 or 1)
    pub fn concat(&self, other: &Matrix<T>, dim: usize) -> Result<Matrix<T>, TensorErrors> {
        let res = self.tensor.concat(&other.tensor, dim)?;
        let res_shape = res.shape.clone();

        Ok(Matrix {
            tensor: res,
            rows: res_shape[0],
            cols: res_shape[1],
        })
    }

    /// Gives an enumerated iter with matrix indices
    pub fn enumerated_iter(&self) -> impl Iterator<Item = ((usize, usize), T)> + use<'_, T> {
        self.tensor.enumerated_iter().map(|(i, x)| ((i[0], i[1]), x))
    }

    /// Gives a mutable enumerated iter with matrix indices
    pub fn enumerated_iter_mut(&mut self) -> impl Iterator<Item = ((usize, usize), &mut T)> + use<'_, T> {
        self.tensor.enumerated_iter_mut().map(|(i, x)| ((i[0], i[1]), x))
    }

    /// Gives an immutable cloned slice to a certain part of the matrix
    pub fn slice(&self, rows_range: Range<usize>, cols_range: Range<usize>) -> Matrix<T> {
        Matrix {
            tensor: self.tensor.slice(&[rows_range.clone(), cols_range.clone()]),
            rows: rows_range.len(),
            cols: cols_range.len(),
        }
    }

    /// Gives a mutable slic to a certain part of the matrix
    pub fn slice_mut(&mut self, rows_range: Range<usize>, cols_range: Range<usize>) -> MatrixSliceMut<'_, T> {
        MatrixSliceMut {
            orig: self,
            start: (rows_range.start, cols_range.start),
            end: (rows_range.end, cols_range.end),
        }
    }
}
impl<T: Clone + Send + Sync> Matrix<T> {
    pub fn concat_mt(&self, other: &Matrix<T>, dim: usize) -> Result<Matrix<T>, TensorErrors> {
        let res_tensor = self.tensor.concat_mt(&other.tensor, dim)?;

        res_tensor.try_into()
    }
}
impl<T: Default + Clone> Matrix<T> {
    pub fn from_shape(rows: usize, cols: usize) -> Matrix<T> {
        Matrix {
            tensor: Tensor::<T>::from_shape(&shape![rows, cols]),
            rows,
            cols,
        }
    }
}
impl<T: Default + Clone> Matrix<T> where StandardUniform: Distribution<T>, [T]: Fill {
    pub fn rand(rows: usize, cols: usize) -> Matrix<T> {
        Matrix {
            tensor: Tensor::<T>::rand(&shape![rows, cols]),
            rows,
            cols,
        }
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
impl<T: Zero + Clone> Matrix<T> {
    pub fn zeros(rows: usize, cols: usize) -> Matrix<T> {
        Matrix::from_value(rows, cols, T::zero())
    }
}
pub struct MatrixSliceMut<'a, T> {
    pub(crate) orig: &'a mut Matrix<T>,
    pub(crate) start: (usize, usize),
    pub(crate) end: (usize, usize),
}
impl<'a, T: Clone> MatrixSliceMut<'a, T> {
    /// Sets all the values in the mutable slice to the values in the given input
    pub fn set_all(&mut self, values: &Matrix<T>) {
        assert!(self.end.0 - self.start.0 == values.rows && self.end.1 - self.start.1 == values.cols, "Mutable slice shape and input shape do not match");

        for (index, value) in values.enumerated_iter() {
            self[index] = value
        }
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
/// Converts an `IntoIter<T>` into a `Matrix<T>` of shape (1, length_of_iter)
impl<T> From<IntoIter<T>> for Matrix<T> {
    fn from(value: IntoIter<T>) -> Self {
        let l = value.len();

        Matrix {
            tensor: value.into(),
            rows: 1,
            cols: l,
        }
    }
}
/// Converts an `IntoIter<T>` into a `Matrix<T>` of shape (1, length_of_iter)
impl<T> FromIterator<T> for Matrix<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let elems = iter.into_iter().collect::<Vec<_>>();

        Matrix {
            rows: 1,
            cols: elems.len(),
            tensor: Tensor::new(&shape![1, elems.len()], elems).unwrap(),
        }
    }
}
/// Converts an iterator into a `Matrix<T>` of shape (1, length_of_iter)
impl<'a, T: Clone> From<Iter<'a, T>> for Matrix<T> {
    fn from(value: Iter<'a, T>) -> Self {
        let elements: Vec<T> = value.map(|x| x.clone()).collect();
        Matrix {
            rows: 1,
            cols: elements.len(),
            tensor: Tensor::new(&shape![elements.len()], elements).unwrap(),
        }
    }
}
/// This trait allows you to specify that something can be infallibly converted into a `Matrix<T>`.
/// Bear in mind that this does automatically derive an implementation for `TryIntoMatrix<T>`.
pub trait IntoMatrix<T> {
    fn into_matrix(self) -> Matrix<T>;
}
/// This trait allows you to specify that something can be fallibly converted into a `Matrix<T>`,
/// returning an error value if it is not possible.
pub trait TryIntoMatrix<T> {
    type Error;

    fn try_into_matrix(self) -> Result<Matrix<T>, Self::Error>;
}
impl<T, O: IntoMatrix<T>> TryIntoMatrix<T> for O {
    type Error = ();

    fn try_into_matrix(self) -> Result<Matrix<T>, Self::Error> {
        Ok(self.into_matrix())
    }
}
impl<T: Clone> IntoMatrix<T> for Vec<T> {
    fn into_matrix(self) -> Matrix<T> {
        Matrix::new(1, self.len(), self).unwrap()
    }
}
impl<T: Clone> IntoMatrix<T> for MatrixSliceMut<'_, T> {
    fn into_matrix(self) -> Matrix<T> {
        self.orig.slice(self.start.0..self.end.0, self.start.1..self.end.1)
    }
}