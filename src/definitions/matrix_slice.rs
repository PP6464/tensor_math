use crate::definitions::chunk::{Chunk, ChunkMut};
use crate::definitions::errors::TensorErrors;
use crate::definitions::errors::TensorErrors::SliceIncompatibleShape;
use crate::definitions::matrix::Matrix;
use crate::definitions::shape::Shape;
use crate::definitions::traits::{IntoMatrix, IntoTensor, MatrixLike, MatrixLikeMut};
use crate::shape;
use rayon::iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

#[derive(Debug, Eq, PartialEq)]
pub struct MatrixSlice<'a, T> {
    pub(crate) orig: &'a Matrix<T>,
    pub(crate) start: (usize, usize),
    pub(crate) end: (usize, usize),
}

impl<T> MatrixSlice<'_, T> {
    /// Returns the number of rows of the matrix slice.
    pub fn rows(&self) -> usize {
        self.end.0 - self.start.0
    }

    /// Returns the number of columns of the matrix slice.
    pub fn cols(&self) -> usize {
        self.end.1 - self.start.1
    }

    /// Returns the start position of the matrix slice.
    pub fn start(&self) -> (usize, usize) {
        self.start
    }

    /// Returns the end position of the matrix slice.
    pub fn end(&self) -> (usize, usize) {
        self.end
    }

    /// Returns the shape of the matrix slice.
    pub fn shape(&self) -> Shape {
        shape![self.rows(), self.cols()]
    }

    /// Applies the given function to each element of the slice.
    pub fn for_each(&self, mut closure: impl FnMut(&T)) {
        let rows = self.rows();
        let cols = self.cols();
        for i in 0..rows * cols {
            closure(&self[(i / cols, i % cols)]);
        }
    }

    /// Applies the given function to each element of the slice with the index in the slice.
    pub fn enumerated_for_each(&self, mut closure: impl FnMut((usize, usize), &T)) {
        let rows = self.rows();
        let cols = self.cols();
        for i in 0..rows * cols {
            closure((i / cols, i % cols), &self[(i / cols, i % cols)]);
        }
    }
}

impl<'a, T: Clone> MatrixSlice<'a, T> {
    /// Gets the value at the specified index, returning None if the index is out of bounds.
    pub fn get(&self, indices: (usize, usize)) -> Option<&T> {
        let orig_index = (indices.0 + self.start.0, indices.1 + self.start.1);

        if orig_index.0 >= self.end.0 || orig_index.1 >= self.end.1 {
            return None;
        }

        self.orig.get(orig_index)
    }
}
impl<T> Index<(usize, usize)> for MatrixSlice<'_, T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        assert!(index.0 + self.start.0 < self.end.0);
        assert!(index.1 + self.start.1 < self.end.1);

        &self.orig[(self.start.0 + index.0, self.start.1 + index.1)]
    }
}

impl<T> Index<&[usize; 2]> for MatrixSlice<'_, T> {
    type Output = T;

    fn index(&self, index: &[usize; 2]) -> &Self::Output {
        assert!(self.start.0 + index[0] < self.end.0);
        assert!(self.start.1 + index[1] < self.end.1);

        &self.orig[(self.start.0 + index[0], self.start.1 + index[1])]
    }
}

impl<T: Clone> IntoMatrix<T> for MatrixSlice<'_, T> {
    fn into_matrix(self) -> Matrix<T> {
        let mut elements = Vec::with_capacity(self.rows() * self.cols());
        for i in 0..self.rows() * self.cols() {
            elements.push(self[(i / self.cols(), i % self.cols())].clone());
        }

        Matrix {
            rows: self.rows(),
            cols: self.cols(),
            tensor: elements.into_tensor(),
        }
    }
}

impl<'a, T: Clone> MatrixLike<T> for MatrixSlice<'a, T> {
    fn shape(&self) -> Shape {
        MatrixSlice::shape(self)
    }

    fn rows(&self) -> usize {
        MatrixSlice::rows(self)
    }

    fn cols(&self) -> usize {
        MatrixSlice::cols(self)
    }

    fn get(&self, indices: (usize, usize)) -> Option<&T> {
        MatrixSlice::get(self, indices)
    }

    fn iter<'b>(&'b self) -> impl Iterator<Item = &'b T>
    where
        T: 'b,
    {
        struct SliceIter<'b, T> {
            slice: &'b MatrixSlice<'b, T>,
            flat_index: usize,
        }

        impl<'b, T> Iterator for SliceIter<'b, T> {
            type Item = &'b T;

            fn next(&mut self) -> Option<Self::Item> {
                let shape = self.slice.shape();
                let res = match shape.tensor_index(self.flat_index) {
                    Ok(i) => Some(&self.slice[(i[0], i[1])]),
                    Err(_) => None,
                };

                self.flat_index += 1;

                res
            }
        }

        SliceIter {
            slice: self,
            flat_index: 0,
        }
    }

    fn par_iter<'b>(&'b self) -> impl IndexedParallelIterator<Item = &'b T>
    where
        T: 'b + Send + Sync,
    {
        let shape = self.shape();
        (0..shape.element_count()).into_par_iter().map(move |i| {
            let tensor_index = shape.tensor_index(i).unwrap();
            &self[(tensor_index[0], tensor_index[1])]
        })
    }

    fn chunks<'b>(&'b self, n: usize) -> impl Iterator<Item = Chunk<'b, T>>
    where
        T: 'b,
    {
        let mut it = self.iter();
        std::iter::from_fn(move || {
            let v: Vec<&'b T> = it.by_ref().take(n).collect();
            (!v.is_empty()).then_some(Chunk::NonContiguous(v))
        })
    }

    fn par_chunks<'b>(&'b self, n: usize) -> impl IndexedParallelIterator<Item = Chunk<'b, T>>
    where
        T: Send + Sync + 'b,
    {
        self.par_iter().chunks(n).map(Chunk::NonContiguous)
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct MatrixSliceMut<'a, T> {
    pub(crate) orig: &'a mut Matrix<T>,
    pub(crate) start: (usize, usize),
    pub(crate) end: (usize, usize),
}

impl<T> MatrixSliceMut<'_, T> {
    /// Returns the number of rows of the matrix slice.
    pub fn rows(&self) -> usize {
        self.end.0 - self.start.0
    }

    /// Returns the number of columns of the matrix slice.
    pub fn cols(&self) -> usize {
        self.end.1 - self.start.1
    }

    /// Returns the start position of the matrix slice.
    pub fn start(&self) -> (usize, usize) {
        self.start
    }

    /// Returns the end position of the matrix slice.
    pub fn end(&self) -> (usize, usize) {
        self.end
    }

    /// Returns the shape of the matrix slice.
    pub fn shape(&self) -> Shape {
        shape![self.rows(), self.cols()]
    }

    /// Applies the given function to each element of the slice.
    pub fn for_each_mut(&mut self, mut closure: impl FnMut(&mut T)) {
        let rows = self.rows();
        let cols = self.cols();
        for i in 0..rows * cols {
            closure(&mut self[(i / cols, i % cols)]);
        }
    }

    /// Applies the given function to each element of the slice with the index in the slice.
    pub fn enumerated_for_each_mut(&mut self, mut closure: impl FnMut((usize, usize), &mut T)) {
        let rows = self.rows();
        let cols = self.cols();
        for i in 0..rows * cols {
            closure((i / cols, i % cols), &mut self[(i / cols, i % cols)]);
        }
    }
}

impl<'a, T: Clone> MatrixSliceMut<'a, T> {
    /// Sets all the values in the mutable slice to the values in the given input.
    /// This fails if the input does not have the same shape as the slice.
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

    /// Gets the value at the specified index, returning None if the index is out of bounds.
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
        let mut elements = Vec::with_capacity(self.rows() * self.cols());
        for i in 0..self.rows() * self.cols() {
            elements.push(self[(i / self.cols(), i % self.cols())].clone());
        }

        Matrix {
            rows: self.rows(),
            cols: self.cols(),
            tensor: elements.into_tensor(),
        }
    }
}

impl<'a, T: Clone> MatrixLike<T> for MatrixSliceMut<'a, T> {
    fn shape(&self) -> Shape {
        MatrixSliceMut::shape(self)
    }

    fn rows(&self) -> usize {
        MatrixSliceMut::rows(self)
    }

    fn cols(&self) -> usize {
        MatrixSliceMut::cols(self)
    }

    fn get(&self, indices: (usize, usize)) -> Option<&T> {
        MatrixSliceMut::get(self, indices)
    }

    fn iter<'b>(&'b self) -> impl Iterator<Item = &'b T>
    where
        T: 'b,
    {
        struct SliceIter<'b, T> {
            slice: &'b MatrixSliceMut<'b, T>,
            flat_index: usize,
        }

        impl<'b, T> Iterator for SliceIter<'b, T> {
            type Item = &'b T;

            fn next(&mut self) -> Option<Self::Item> {
                let shape = self.slice.shape();
                let res = match shape.tensor_index(self.flat_index) {
                    Ok(i) => Some(&self.slice[(i[0], i[1])]),
                    Err(_) => None,
                };

                self.flat_index += 1;

                res
            }
        }

        SliceIter {
            slice: self,
            flat_index: 0,
        }
    }

    fn par_iter<'b>(&'b self) -> impl IndexedParallelIterator<Item = &'b T>
    where
        T: 'b + Send + Sync,
    {
        let shape = self.shape();
        (0..shape.element_count()).into_par_iter().map(move |i| {
            let tensor_index = shape.tensor_index(i).unwrap();
            &self[(tensor_index[0], tensor_index[1])]
        })
    }

    fn chunks<'b>(&'b self, n: usize) -> impl Iterator<Item = Chunk<'b, T>>
    where
        T: 'b,
    {
        let mut it = self.iter();
        std::iter::from_fn(move || {
            let v: Vec<&'b T> = it.by_ref().take(n).collect();
            (!v.is_empty()).then_some(Chunk::NonContiguous(v))
        })
    }

    fn par_chunks<'b>(&'b self, n: usize) -> impl IndexedParallelIterator<Item = Chunk<'b, T>>
    where
        T: Send + Sync + 'b,
    {
        self.par_iter().chunks(n).map(Chunk::NonContiguous)
    }
}

impl<'a, T: Clone> MatrixLikeMut<T> for MatrixSliceMut<'a, T> {
    fn get_mut(&mut self, indices: (usize, usize)) -> Option<&mut T> {
        self.orig.get_mut(indices)
    }

    fn iter_mut<'b>(&'b mut self) -> impl Iterator<Item = &'b mut T>
    where
        T: 'b,
    {
        struct SliceIterMut<'b, T> {
            base: *mut T,
            flat_index: usize,
            shape: &'b Shape,
        }

        impl<'b, T: 'b> Iterator for SliceIterMut<'b, T> {
            type Item = &'b mut T;

            fn next(&mut self) -> Option<Self::Item> {
                self.shape.tensor_index(self.flat_index).ok()?;
                self.flat_index += 1;
                let offset = self.flat_index - 1;
                Some(unsafe { &mut *self.base.add(offset) })
            }
        }

        SliceIterMut {
            base: self.orig.elements_mut().as_mut_ptr(),
            flat_index: 0,
            shape: &self.orig.shape,
        }
    }

    fn par_iter_mut<'b>(&'b mut self) -> impl IndexedParallelIterator<Item = &'b mut T>
    where
        T: 'b + Send + Sync,
    {
        let orig_shape = self.orig.shape.clone();
        let slice_shape = self.shape().clone();
        let base = self.orig.elements_mut().as_mut_ptr();
        let count = orig_shape.element_count();

        struct MatrixSliceMutIter<'b, T> {
            slice_start: (usize, usize), // Start of the entire slice
            start: usize,                // Flat index of start (within entire slice)
            end: usize,                  // Flat exclusive end of subslice (within entire slice)
            base: *mut T,                // start for the entire original tensor
            slice_shape: Shape,          // Shape of the entire slice
            orig_shape: Shape,           // Shape of entire original matrix
            _marker: PhantomData<&'b T>,
        }

        unsafe impl<'b, T> Send for MatrixSliceMutIter<'b, T> {}
        unsafe impl<'b, T> Sync for MatrixSliceMutIter<'b, T> {}

        struct ParIter<'b, T> {
            c: &'b T,
        }

        impl<'b, T: 'b> Iterator for MatrixSliceMutIter<'b, T> {
            type Item = &'b mut T;

            fn next(&mut self) -> Option<Self::Item> {
                if self.start >= self.end {
                    return None;
                }

                let orig_index = self
                    .slice_shape
                    .tensor_index(self.start)
                    .ok()?
                    .iter()
                    .zip([self.slice_start.0, self.slice_start.1].iter())
                    .map(|(a, b)| a + b)
                    .collect::<Vec<_>>();
                let res = unsafe {
                    self.base
                        .add(self.orig_shape.address(orig_index).ok()?)
                        .as_mut()
                };

                self.start += 1;

                res
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                (self.end - self.start, Some(self.end - self.start))
            }
        }
        impl<'b, T: 'b> DoubleEndedIterator for MatrixSliceMutIter<'b, T> {
            fn next_back(&mut self) -> Option<Self::Item> {
                if self.end <= self.start {
                    return None;
                }

                self.end -= 1;

                let orig_index = self
                    .slice_shape
                    .tensor_index(self.end)
                    .ok()?
                    .iter()
                    .zip([self.slice_start.0, self.slice_start.1].iter())
                    .map(|(a, b)| a + b)
                    .collect::<Vec<_>>();

                unsafe {
                    self.base
                        .add(self.orig_shape.address(orig_index).ok()?)
                        .as_mut()
                }
            }
        }
        impl<'b, T: 'b> ExactSizeIterator for MatrixSliceMutIter<'b, T> {}

        impl<'b, T: 'b> Producer for MatrixSliceMutIter<'b, T> {
            type Item = &'b mut T;
            type IntoIter = MatrixSliceMutIter<'b, T>;

            fn into_iter(self) -> Self::IntoIter {
                self
            }

            fn split_at(self, index: usize) -> (Self, Self) {
                (
                    MatrixSliceMutIter {
                        slice_start: self.slice_start,
                        start: self.start,
                        end: self.start + index,
                        base: self.base,
                        slice_shape: self.slice_shape.clone(),
                        orig_shape: self.orig_shape.clone(),
                        _marker: Default::default(),
                    },
                    MatrixSliceMutIter {
                        slice_start: self.slice_start,
                        start: self.start + index,
                        end: self.end,
                        base: self.base,
                        slice_shape: self.slice_shape,
                        orig_shape: self.orig_shape,
                        _marker: Default::default(),
                    },
                )
            }
        }

        impl<'b, T> ParallelIterator for MatrixSliceMutIter<'b, T>
        where
            T: Send + Sync + 'b,
        {
            type Item = &'b mut T;

            fn drive_unindexed<C>(self, consumer: C) -> C::Result
            where
                C: UnindexedConsumer<Self::Item>,
            {
                bridge(self, consumer)
            }
        }

        impl<'b, T: Send + Sync + 'b> IndexedParallelIterator for MatrixSliceMutIter<'b, T> {
            fn len(&self) -> usize {
                self.end - self.start
            }

            fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> C::Result {
                bridge(self, consumer)
            }

            fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
                callback.callback(self)
            }
        }

        MatrixSliceMutIter {
            slice_start: self.start,
            start: 0,
            end: count,
            base,
            slice_shape,
            orig_shape,
            _marker: Default::default(),
        }
    }

    fn chunks_mut<'b>(&'b mut self, n: usize) -> impl Iterator<Item = ChunkMut<'b, T>>
    where
        T: 'b,
    {
        let mut it = self.iter_mut();
        std::iter::from_fn(move || {
            let v: Vec<&'b mut T> = it.by_ref().take(n).collect();
            (!v.is_empty()).then_some(ChunkMut::NonContiguous(v))
        })
    }

    fn par_chunks_mut<'b>(
        &'b mut self,
        n: usize,
    ) -> impl IndexedParallelIterator<Item = ChunkMut<'b, T>>
    where
        T: Send + Sync + 'b,
    {
        self.par_iter_mut().chunks(n).map(ChunkMut::NonContiguous)
    }
}
