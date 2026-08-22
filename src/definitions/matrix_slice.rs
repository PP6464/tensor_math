use crate::definitions::matrix::Matrix;
use crate::definitions::shape::Shape;
use crate::shape;
use rayon::iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::marker::PhantomData;
use std::ops::{Deref, Index, IndexMut};

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

    /// Returns the shape of the matrix slice.
    pub fn shape(&self) -> Shape {
        shape![self.rows(), self.cols()]
    }

    /// Returns the number of rows in the matrix slice.
    pub fn rows(&self) -> usize {
        self.end.0 - self.start.0
    }

    /// Returns the number of columns in the matrix slice.
    pub fn cols(&self) -> usize {
        self.end.1 - self.start.1
    }

    /// Checks if the matrix slice is square.
    pub fn is_square(&self) -> bool {
        self.rows() == self.cols()
    }

    /// Gets the element at the specified indices.
    pub fn get(&self, indices: (usize, usize)) -> Option<&T> {
        if indices.0 >= self.rows() || indices.1 >= self.cols() {
            return None;
        }

        unsafe { Some(self.get_unchecked((indices.0, indices.1))) }
    }

    /// Gets the element at the specified indices without bounds checking.
    pub(crate) unsafe fn get_unchecked(&self, indices: (usize, usize)) -> &T {
        self.orig
            .get_unchecked((indices.0 + self.start.0, indices.1 + self.start.1))
    }

    /// Returns an iterator over the elements of the matrix slice.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        struct SliceIter<'a, T> {
            slice: &'a MatrixSlice<'a, T>,
            shape: Shape,
            flat_index: usize,
        }

        impl<'a, T> Iterator for SliceIter<'a, T> {
            type Item = &'a T;

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

    /// Returns a parallel iterator over the elements of the matrix slice.
    pub fn par_iter(&self) -> impl IndexedParallelIterator<Item = &T>
    where
        T: Send + Sync,
    {
        let shape = self.shape();
        (0..shape.element_count())
            .into_par_iter()
            .map(move |i| unsafe { self.get_unchecked((i / self.cols(), i % self.cols())) })
    }

    /// Returns an iterator over chunks of the elements of the matrix slice.
    pub fn chunks(&self, n: usize) -> impl Iterator<Item = Vec<&T>> {
        let mut it = self.iter();
        std::iter::from_fn(move || {
            let v: Vec<&T> = it.by_ref().take(n).collect();
            (!v.is_empty()).then_some(v)
        })
    }

    /// Returns a parallel iterator over chunks of the elements of the matrix slice.
    pub fn par_chunks(&self, n: usize) -> impl IndexedParallelIterator<Item = Vec<&T>> {
        self.par_iter().chunks(n)
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

impl<T: Clone> MatrixSlice<'_, T> {
    /// Converts the matrix slice into a new matrix.
    pub fn clone_into_matrix(&self) -> Matrix<T> {
        let mut elements = Vec::with_capacity(self.rows() * self.cols());
        for i in self.start.0..self.end.0 {
            let start = i * self.orig.cols + self.start.1;
            let end = i * self.orig.cols + self.end.1;
            unsafe { elements.extend_from_slice(&self.orig.elements.get_unchecked(start..end)); }
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
    /// Gets a mutable reference to the element at the specified indices.
    pub fn get_mut(&mut self, indices: (usize, usize)) -> Option<&mut T> {
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

    /// Gets a mutable reference to the element at the specified indices without bounds checking.
    pub(crate) unsafe fn get_unchecked_mut(&mut self, indices: (usize, usize)) -> &mut T {
        self.orig
            .get_unchecked_mut((indices.0 + self.start.0, indices.1 + self.start.1))
    }

    /// Returns an iterator over mutable references to the elements of the matrix slice.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        struct SliceIterMut<'a, T> {
            base: *mut T,
            flat_index: usize,
            len: usize,
            _marker: PhantomData<&'a T>,
        }

        impl<'a, T: 'a> Iterator for SliceIterMut<'a, T> {
            type Item = &'a mut T;

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

    /// Returns a parallel iterator over mutable references to the elements of the matrix slice.
    pub fn par_iter_mut(&mut self) -> impl IndexedParallelIterator<Item = &mut T>
    where
        T: Send + Sync,
    {
        let orig_shape = self.orig.shape().clone();
        let slice_shape = self.shape().clone();
        let base = self.orig.elements_mut().as_mut_ptr();
        let count = orig_shape.element_count();

        struct MatrixSliceMutIter<'a, T> {
            slice_start: (usize, usize), // Start of the entire slice
            start: usize,                // Flat index of start (within entire slice)
            end: usize,                  // Flat exclusive end of subslice (within entire slice)
            base: *mut T,                // start for the entire original matrix
            slice_shape: Shape,          // Shape of the entire slice
            orig_shape: Shape,           // Shape of entire original matrix
            _marker: PhantomData<&'a T>,
        }

        unsafe impl<'a, T> Send for MatrixSliceMutIter<'a, T> {}
        unsafe impl<'a, T> Sync for MatrixSliceMutIter<'a, T> {}

        impl<'a, T: 'a> Iterator for MatrixSliceMutIter<'a, T> {
            type Item = &'a mut T;

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
        impl<'a, T: 'a> DoubleEndedIterator for MatrixSliceMutIter<'a, T> {
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
        impl<'a, T: 'a> ExactSizeIterator for MatrixSliceMutIter<'a, T> {}

        impl<'a, T: 'a> Producer for MatrixSliceMutIter<'a, T> {
            type Item = &'a mut T;
            type IntoIter = MatrixSliceMutIter<'a, T>;

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

        impl<'a, T: Send + Sync + 'a> ParallelIterator for MatrixSliceMutIter<'a, T> {
            type Item = &'a mut T;

            fn drive_unindexed<C>(self, consumer: C) -> C::Result
            where
                C: UnindexedConsumer<Self::Item>,
            {
                bridge(self, consumer)
            }
        }

        impl<'a, T: Send + Sync + 'a> IndexedParallelIterator for MatrixSliceMutIter<'a, T> {
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

    /// Returns an iterator over mutable chunks of the elements of the matrix slice.
    pub fn chunks_mut<'a>(&'a mut self, n: usize) -> impl Iterator<Item = Vec<&mut T>> {
        let mut it = self.iter_mut();
        std::iter::from_fn(move || {
            let v: Vec<&'a mut T> = it.by_ref().take(n).collect();
            (!v.is_empty()).then_some(v)
        })
    }

    /// Returns a parallel iterator over mutable chunks of the elements of the matrix slice.
    pub fn par_chunks_mut(&mut self, n: usize) -> impl IndexedParallelIterator<Item = Vec<&mut T>>
    where
        T: Send + Sync,
    {
        self.par_iter_mut().chunks(n)
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
* Deref implementation
--------------------------------------------
*/

impl<'a, T> Deref for MatrixSliceMut<'a, T> {
    type Target = MatrixSlice<'a, T>;

    fn deref(&self) -> &Self::Target {
        &MatrixSlice {
            orig: self.orig,
            start: self.start,
            end: self.end,
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
