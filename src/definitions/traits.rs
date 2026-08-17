use crate::definitions::chunk::{Chunk, ChunkMut};
use crate::definitions::matrix::Matrix;
use crate::definitions::shape::Shape;
use crate::definitions::tensor::Tensor;
use rayon::iter::IndexedParallelIterator;
use std::ops::{Index, IndexMut};

/// This trait allows you to specify that something can be infallibly converted into a tensor.
/// This automatically derives an implementation for `TryIntoTensor`.
pub trait IntoTensor<T> {
    fn into_tensor(self) -> Tensor<T>;
}

/// This trait allows you to specify that something can be fallibly converted into a tensor.
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

/// This trait allows you to specify that something can be infallibly converted into a matrix.
/// This automatically derives an implementation for `TryIntoMatrix`.
pub trait IntoMatrix<T> {
    fn into_matrix(self) -> Matrix<T>;
}

/// This trait allows you to specify that something can be fallibly converted into a matrix.
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

/// Implemented by anything that behaves like an immutable tensor: it exposes its shape,
/// rank, and a way to read its elements. Supports `[]` indexing via the standard
/// `Index` impls that concrete types provide, and provides borrowing and consuming
/// iteration over its elements.
pub trait TensorLike<T>: for<'a> Index<&'a [usize], Output = T> {
    /// Returns the shape of the tensor-like value.
    fn shape(&self) -> Shape;

    /// Returns the rank (number of dimensions) of the tensor-like value.
    fn rank(&self) -> usize;

    /// Returns the elements of the tensor-like value as a slice.
    fn elements(&self) -> &[T];

    /// Returns a reference to the element at the given indices if it is in bounds,
    /// otherwise returns None.
    fn get(&self, indices: &[usize]) -> Option<&T>;

    /// Returns an iterator over references to the elements of the tensor-like value.
    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T>
    where
        T: 'a;

    /// Returns a parallel iterator over the references.
    fn par_iter<'a>(&'a self) -> impl IndexedParallelIterator<Item = &'a T>
    where
        T: 'a + Send + Sync;

    /// Returns an iterator over chunks of the tensor-like value
    fn chunks<'a>(&'a self, n: usize) -> impl Iterator<Item = Chunk<'a, T>>
    where
        T: 'a;

    /// Returns a parallel iterator over chunks of the tensor-like value
    fn par_chunks<'a>(&'a self, n: usize) -> impl IndexedParallelIterator<Item = Chunk<'a, T>>
    where
        T: Send + Sync + 'a;
}

/// Implemented by anything that behaves like a mutable tensor: in addition to the
/// `TensorLike` interface, it provides a way to obtain a mutable view of its elements.
/// Supports `[]` indexing for both read and write via the standard `Index` / `IndexMut`
/// impls that concrete types provide, and provides mutable borrowing and consuming
/// iteration over its elements.
pub trait TensorLikeMut<T>: TensorLike<T> + for<'a> IndexMut<&'a [usize]> {
    /// Returns the elements of the tensor-like value as a mutable slice.
    fn elements_mut(&mut self) -> &mut [T];

    /// Returns a mutable reference to the element at the given indices if it is in bounds,
    /// otherwise returns None.
    fn get_mut(&mut self, indices: &[usize]) -> Option<&mut T>;

    /// Returns an iterator over mutable references to the elements of the tensor-like value.
    fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &mut T>
    where
        T: 'a;

    /// Returns a parallel mutable iterator over the references.
    fn par_iter_mut<'a>(&'a mut self) -> impl IndexedParallelIterator<Item = &'a mut T>
    where
        T: 'a + Send + Sync;

    /// Returns a mutable iterator over chunks of the tensor-like value
    fn chunks_mut<'a>(&'a mut self, n: usize) -> impl Iterator<Item = ChunkMut<'a, T>>
    where
        T: 'a;

    /// Returns a parallel mutable iterator over chunks of the tensor-like value
    fn par_chunks_mut<'a>(&'a mut self, n: usize) -> impl IndexedParallelIterator<Item = ChunkMut<'a, T>>
    where
        T: Send + Sync + 'a;
}

/// Implemented by anything that behaves like an immutable matrix: it exposes its shape,
/// row count, column count, and a way to read its elements. Supports `[]` indexing via
/// the standard `Index` impls that concrete types provide, and provides borrowing and
/// consuming iteration over its elements.
pub trait MatrixLike<T>:
    Index<(usize, usize), Output = T> + for<'a> Index<&'a [usize; 2], Output = T>
{
    /// Returns the shape of the matrix-like value.
    fn shape(&self) -> Shape;

    /// Returns the number of rows of the matrix-like value.
    fn rows(&self) -> usize;

    /// Returns the number of columns of the matrix-like value.
    fn cols(&self) -> usize;

    /// Returns a reference to the element at the given indices if it is in bounds,
    /// otherwise returns None.
    fn get(&self, indices: (usize, usize)) -> Option<&T>;

    /// Returns an iterator over references to the elements of the matrix-like value.
    fn iter<'a>(&'a self) -> impl Iterator<Item = &T>
    where
        T: 'a;

    /// Returns a parallel iterator over the references.
    fn par_iter<'a>(&'a self) -> impl IndexedParallelIterator<Item = &'a T>
    where
        T: 'a + Send + Sync;

    /// Returns an iterator over chunks of the tensor-like value
    fn chunks<'a>(&'a self, n: usize) -> impl Iterator<Item = Chunk<'a, T>>
    where
        T: 'a;

    /// Returns a parallel iterator over chunks of the tensor-like value
    fn par_chunks<'a>(&'a self, n: usize) -> impl IndexedParallelIterator<Item = Chunk<'a, T>>
    where
        T: Send + Sync + 'a;
}

/// Implemented by anything that behaves like a mutable matrix: in addition to the
/// `MatrixLike` interface, it provides a way to obtain a mutable reference to its elements.
/// Supports `[]` indexing for both read and write via the standard `Index` / `IndexMut`
/// impls that concrete types provide, and provides mutable borrowing and consuming
/// iteration over its elements.
pub trait MatrixLikeMut<T>:
    MatrixLike<T> + IndexMut<(usize, usize)> + for<'a> IndexMut<&'a [usize; 2]>
{
    /// Returns a mutable reference to the element at the given indices if it is in bounds,
    /// otherwise returns None.
    fn get_mut(&mut self, indices: (usize, usize)) -> Option<&mut T>;

    /// Returns an iterator over mutable references to the elements of the matrix-like value.
    fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &mut T>
    where
        T: 'a;

    /// Returns a parallel mutable iterator over the references.
    fn par_iter_mut<'a>(&'a mut self) -> impl IndexedParallelIterator<Item = &'a mut T>
    where
        T: 'a + Send + Sync;

    /// Returns a mutable iterator over chunks of the tensor-like value
    fn chunks_mut<'a>(&'a mut self, n: usize) -> impl Iterator<Item = ChunkMut<'a, T>>
    where
        T: 'a;

    /// Returns a parallel mutable iterator over chunks of the tensor-like value
    fn par_chunks_mut<'a>(&'a mut self, n: usize) -> impl IndexedParallelIterator<Item = ChunkMut<'a, T>>
    where
        T: Send + Sync + 'a;
}
