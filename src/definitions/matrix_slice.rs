use crate::definitions::chunk::{Chunk, ChunkMut};
use crate::definitions::matrix::Matrix;
use crate::definitions::shape::Shape;
use crate::definitions::traits::{IntoMatrix, MatrixLike, MatrixLikeMut};
use crate::shape;
use rayon::iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

/*
--------------------------------------------
* Immutable matrix slice definition
--------------------------------------------
*/

#[derive(Debug, Eq, PartialEq)]
pub struct MatrixSlice<'a, T> {
    pub(crate) orig: &'a Matrix<T>,
    pub(crate) start: (usize, usize),
    pub(crate) end: (usize, usize),
}

impl<T> MatrixSlice<'_, T> {
    /// Returns the start position of the matrix slice.
    pub fn start(&self) -> (usize, usize) {
        self.start
    }

    /// Returns the end position of the matrix slice.
    pub fn end(&self) -> (usize, usize) {
        self.end
    }

    /// Applies the given function to each element of the slice.
    pub fn for_each(&self, mut closure: impl FnMut(&T)) {
        let rows = self.rows();
        let cols = self.cols();
        for i in 0..rows * cols {
            unsafe {
                closure(self.get_unchecked((i / cols, i % cols)));
            }
        }
    }

    /// Applies the given function to each element of the slice with the index in the slice.
    pub fn enumerated_for_each(&self, mut closure: impl FnMut((usize, usize), &T)) {
        let rows = self.rows();
        let cols = self.cols();
        for i in 0..rows * cols {
            unsafe {
                closure(
                    (i / cols, i % cols),
                    self.get_unchecked((i / cols, i % cols)),
                );
            }
        }
    }
}

/*
--------------------------------------------
* Indexing for immutable matrix slices
--------------------------------------------
*/

impl<T> Index<(usize, usize)> for MatrixSlice<'_, T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        assert!(index.0 + self.start.0 < self.end.0);
        assert!(index.1 + self.start.1 < self.end.1);

        unsafe {
            self.orig
                .get_unchecked((self.start.0 + index.0, self.start.1 + index.1))
        }
    }
}

impl<T> Index<&[usize; 2]> for MatrixSlice<'_, T> {
    type Output = T;

    fn index(&self, index: &[usize; 2]) -> &Self::Output {
        assert!(self.start.0 + index[0] < self.end.0);
        assert!(self.start.1 + index[1] < self.end.1);

        unsafe {
            self.orig
                .get_unchecked((self.start.0 + index[0], self.start.1 + index[1]))
        }
    }
}

/*
--------------------------------------------
* Conversion into Matrix
--------------------------------------------
*/

impl<T: Clone> IntoMatrix<T> for MatrixSlice<'_, T> {
    fn into_matrix(self) -> Matrix<T> {
        let mut elements = Vec::with_capacity(self.rows() * self.cols());
        for i in 0..self.rows() * self.cols() {
            elements.push(self[(i / self.cols(), i % self.cols())].clone());
        }

        Matrix {
            rows: self.rows(),
            cols: self.cols(),
            elements,
        }
    }
}

impl<T: Clone> MatrixSlice<'_, T> {
    /// Converts the matrix slice into a new matrix.
    pub fn clone_to_matrix(&self) -> Matrix<T> {
        let mut elements = Vec::with_capacity(self.rows() * self.cols());
        for i in 0..self.rows() * self.cols() {
            elements.push(self[(i / self.cols(), i % self.cols())].clone());
        }

        Matrix {
            rows: self.rows(),
            cols: self.cols(),
            elements,
        }
    }
}

/*
--------------------------------------------
* Matrix-like trait implementations
--------------------------------------------
*/

impl<'a, T> MatrixLike<T> for MatrixSlice<'a, T> {
    fn shape(&self) -> Shape {
        shape![self.rows(), self.cols()]
    }

    fn rows(&self) -> usize {
        self.end.0 - self.start.0
    }

    fn cols(&self) -> usize {
        self.end.1 - self.start.1
    }

    fn is_square(&self) -> bool {
        self.rows() == self.cols()
    }

    fn get(&self, indices: (usize, usize)) -> Option<&T> {
        if indices.0 >= self.rows() || indices.1 >= self.cols() {
            return None;
        }

        unsafe { Some(self.get_unchecked((indices.0, indices.1))) }
    }

    unsafe fn get_unchecked(&self, indices: (usize, usize)) -> &T {
        self.orig
            .get_unchecked((indices.0 + self.start.0, indices.1 + self.start.1))
    }

    fn iter<'b>(&'b self) -> impl Iterator<Item = &'b T>
    where
        T: 'b,
    {
        struct SliceIter<'b, T> {
            slice: &'b MatrixSlice<'b, T>,
            shape: Shape,
            flat_index: usize,
        }

        impl<'b, T> Iterator for SliceIter<'b, T> {
            type Item = &'b T;

            fn next(&mut self) -> Option<Self::Item> {
                let res = match self.shape.tensor_index(self.flat_index) {
                    Ok(i) => Some(unsafe { self.slice.get_unchecked((i[0], i[1])) }),
                    Err(_) => None,
                };

                self.flat_index += 1;

                res
            }
        }

        SliceIter {
            slice: self,
            shape: self.shape(),
            flat_index: 0,
        }
    }

    fn par_iter<'b>(&'b self) -> impl IndexedParallelIterator<Item = &'b T>
    where
        T: 'b + Send + Sync,
    {
        let shape = self.shape();
        (0..shape.element_count())
            .into_par_iter()
            .map(move |i| unsafe { self.get_unchecked((i / self.cols(), i % self.cols())) })
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

/*
--------------------------------------------
* Mutable matrix slice definition
--------------------------------------------
*/

#[derive(Debug, Eq, PartialEq)]
pub struct MatrixSliceMut<'a, T> {
    pub(crate) orig: &'a mut Matrix<T>,
    pub(crate) start: (usize, usize),
    pub(crate) end: (usize, usize),
}

impl<T> MatrixSliceMut<'_, T> {
    /// Returns the start position of the matrix slice.
    pub fn start(&self) -> (usize, usize) {
        self.start
    }

    /// Returns the end position of the matrix slice.
    pub fn end(&self) -> (usize, usize) {
        self.end
    }

    /// Applies the given function to each element of the slice.
    pub fn for_each_mut(&mut self, mut closure: impl FnMut(&mut T)) {
        let rows = self.rows();
        let cols = self.cols();
        for i in 0..rows * cols {
            unsafe {
                closure(self.get_unchecked_mut((i / cols, i % cols)));
            }
        }
    }

    /// Applies the given function to each element of the slice with the index in the slice.
    pub fn enumerated_for_each_mut(&mut self, mut closure: impl FnMut((usize, usize), &mut T)) {
        let rows = self.rows();
        let cols = self.cols();
        for i in 0..rows * cols {
            unsafe {
                closure(
                    (i / cols, i % cols),
                    self.get_unchecked_mut((i / cols, i % cols)),
                );
            }
        }
    }
}

/*
--------------------------------------------
* Indexing for mutable matrix slices
--------------------------------------------
*/

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

/*
--------------------------------------------
* Conversion into Matrix
--------------------------------------------
*/

impl<T: Clone> IntoMatrix<T> for MatrixSliceMut<'_, T> {
    fn into_matrix(self) -> Matrix<T> {
        let mut elements = Vec::with_capacity(self.rows() * self.cols());
        for i in 0..self.rows() * self.cols() {
            unsafe {
                elements.push(
                    self.get_unchecked((i / self.cols(), i % self.cols()))
                        .clone(),
                );
            }
        }

        Matrix {
            rows: self.rows(),
            cols: self.cols(),
            elements,
        }
    }
}

impl<T: Clone> MatrixSliceMut<'_, T> {
    /// Converts the matrix slice into a new matrix.
    pub fn clone_to_matrix(&self) -> Matrix<T> {
        let mut elements = Vec::with_capacity(self.rows() * self.cols());
        for i in 0..self.rows() * self.cols() {
            elements.push(self[(i / self.cols(), i % self.cols())].clone());
        }

        Matrix {
            rows: self.rows(),
            cols: self.cols(),
            elements,
        }
    }
}

/*
--------------------------------------------
* Matrix-like trait implementations
--------------------------------------------
*/

impl<'a, T> MatrixLike<T> for MatrixSliceMut<'a, T> {
    fn shape(&self) -> Shape {
        shape![self.rows(), self.cols()]
    }

    fn rows(&self) -> usize {
        self.end.0 - self.start.0
    }

    fn cols(&self) -> usize {
        self.end.1 - self.start.1
    }

    fn is_square(&self) -> bool {
        self.rows() == self.cols()
    }

    fn get(&self, indices: (usize, usize)) -> Option<&T> {
        let orig_index = (indices.0 + self.start.0, indices.1 + self.start.1);

        if orig_index.0 >= self.end.0 || orig_index.1 >= self.end.1 {
            return None;
        }

        unsafe { Some(self.orig.get_unchecked(orig_index)) }
    }

    unsafe fn get_unchecked(&self, indices: (usize, usize)) -> &T {
        self.orig
            .get_unchecked((indices.0 + self.start.0, indices.1 + self.start.1))
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
                    Ok(i) => Some(unsafe { self.slice.get_unchecked((i[0], i[1])) }),
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
        (0..shape.element_count())
            .into_par_iter()
            .map(move |i| unsafe { self.get_unchecked((i / self.cols(), i % self.cols())) })
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

impl<'a, T> MatrixLikeMut<T> for MatrixSliceMut<'a, T> {
    fn get_mut(&mut self, indices: (usize, usize)) -> Option<&mut T> {
        if indices.0 >= self.rows() || indices.1 >= self.cols() {
            return None;
        }

        unsafe {
            Some(
                self.orig
                    .get_unchecked_mut((indices.0 + self.start.0, indices.1 + self.start.1)),
            )
        }
    }

    unsafe fn get_unchecked_mut(&mut self, indices: (usize, usize)) -> &mut T {
        self.orig
            .get_unchecked_mut((indices.0 + self.start.0, indices.1 + self.start.1))
    }

    fn iter_mut<'b>(&'b mut self) -> impl Iterator<Item = &'b mut T>
    where
        T: 'b,
    {
        struct SliceIterMut<'b, T> {
            base: *mut T,
            flat_index: usize,
            len: usize,
            _marker: PhantomData<&'b T>,
        }

        impl<'b, T: 'b> Iterator for SliceIterMut<'b, T> {
            type Item = &'b mut T;

            fn next(&mut self) -> Option<Self::Item> {
                (self.flat_index < self.len).then_some(unsafe {
                    self.flat_index += 1;
                    self.base.add(self.flat_index - 1).as_mut()?
                })
            }
        }

        SliceIterMut {
            base: self.orig.elements_mut().as_mut_ptr(),
            flat_index: 0,
            len: self.shape().element_count(),
            _marker: Default::default(),
        }
    }

    fn par_iter_mut<'b>(&'b mut self) -> impl IndexedParallelIterator<Item = &'b mut T>
    where
        T: 'b + Send + Sync,
    {
        let orig_shape = self.orig.shape().clone();
        let slice_shape = self.shape().clone();
        let base = self.orig.elements_mut().as_mut_ptr();
        let count = orig_shape.element_count();

        struct MatrixSliceMutIter<'b, T> {
            slice_start: (usize, usize), // Start of the entire slice
            start: usize,                // Flat index of start (within entire slice)
            end: usize,                  // Flat exclusive end of subslice (within entire slice)
            base: *mut T,                // start for the entire original matrix
            slice_shape: Shape,          // Shape of the entire slice
            orig_shape: Shape,           // Shape of entire original matrix
            _marker: PhantomData<&'b T>,
        }

        unsafe impl<'b, T> Send for MatrixSliceMutIter<'b, T> {}
        unsafe impl<'b, T> Sync for MatrixSliceMutIter<'b, T> {}

        impl<'b, T: 'b> Iterator for MatrixSliceMutIter<'b, T> {
            type Item = &'b mut T;

            fn next(&mut self) -> Option<Self::Item> {
                if self.start >= self.end {
                    return None;
                }

                let orig_index = unsafe {
                    self.slice_shape
                        .tensor_index_unchecked(self.start)
                        .iter()
                        .zip([self.slice_start.0, self.slice_start.1].iter())
                        .map(|(a, b)| a + b)
                        .collect::<Vec<_>>()
                };
                let res = unsafe {
                    self.base
                        .add(self.orig_shape.address_unchecked(orig_index))
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

                unsafe {
                    let orig_index = self
                        .slice_shape
                        .tensor_index_unchecked(self.end)
                        .iter()
                        .zip([self.slice_start.0, self.slice_start.1].iter())
                        .map(|(a, b)| a + b)
                        .collect::<Vec<_>>();

                    self.base
                        .add(self.orig_shape.address_unchecked(orig_index))
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

        impl<'b, T: Send + Sync + 'b> ParallelIterator for MatrixSliceMutIter<'b, T> {
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
