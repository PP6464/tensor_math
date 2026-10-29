use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::definitions::matrix_slice::MatrixSlice;
use crate::definitions::shape::Shape;
use crate::shape;
use rayon::iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

/*
--------------------------------------------
* Mutable matrix slice definition
--------------------------------------------
*/

#[derive(Debug, Eq, PartialEq, Hash)]
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

    /// Returns this slice as an immutable slice.
    pub fn as_immut(&self) -> MatrixSlice<'_, T> {
        MatrixSlice {
            orig: self.orig,
            start: self.start,
            end: self.end,
        }
    }

    /// Uses the immutable version of this slice in a closure, returning the result of the closure.
    pub fn use_as_immut<R>(&self, f: impl FnOnce(MatrixSlice<'_, T>) -> R) -> R {
        f(self.as_immut())
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

    /// Gets a mutable reference to the element at the specified indices.
    pub fn get_mut(&mut self, indices: (usize, usize)) -> Option<&mut T> {
        if indices.0 >= self.rows() || indices.1 >= self.cols() {
            return None;
        }

        unsafe {
            Some(
                self.get_unchecked_mut((indices.0, indices.1)),
            )
        }
    }

    /// Gets a mutable reference to the element at the specified indices without bounds checking.
    pub(crate) unsafe fn get_unchecked_mut(&mut self, indices: (usize, usize)) -> &mut T {
        self.orig
            .get_unchecked_mut((indices.0 + self.start.0, indices.1 + self.start.1))
    }

    /// Returns an iterator over the elements of the matrix slice.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &T> + DoubleEndedIterator {
        (0..self.rows() * self.cols())
            .map(move |i| unsafe { self.get_unchecked((i / self.cols(), i % self.cols())) })
    }

    /// Returns a parallel iterator over the elements of the matrix slice.
    pub fn par_iter(&self) -> impl IndexedParallelIterator<Item = &T>
    where
        T: Send + Sync,
    {
        (0..self.rows() * self.cols())
            .into_par_iter()
            .map(move |i| unsafe { self.get_unchecked((i / self.cols(), i % self.cols())) })
    }

    /// Returns an iterator over mutable references to the elements of the matrix slice.
    pub fn iter_mut(&mut self) -> impl ExactSizeIterator<Item = &mut T> + DoubleEndedIterator {
        let orig_shape = self.orig.shape().clone();
        let slice_shape = self.shape().clone();
        let base = self.orig.elements_mut().as_mut_ptr();
        let count = slice_shape.element_count();

        #[derive(Debug)]
        struct MatrixSliceMutIter<'a, T> {
            slice_start: (usize, usize), // Start of the entire slice
            start: usize,                // Flat index of start (within entire slice)
            end: usize,                  // Flat exclusive end of subslice (within entire slice)
            base: *mut T,                // start for the entire original matrix
            slice_shape: Shape,          // Shape of the entire slice
            orig_shape: Shape,           // Shape of entire original matrix
            _marker: PhantomData<&'a T>,
        }

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
                        .add(self.orig_shape.address_unchecked(&orig_index))
                        .as_mut()
                };

                self.start += 1;

                res
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                (self.end - self.start, Some(self.end - self.start))
            }
        }
        impl<'a, T: 'a> ExactSizeIterator for MatrixSliceMutIter<'a, T> {}
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
                        .add(self.orig_shape.address_unchecked(&orig_index))
                        .as_mut()
                }
            }
        }

        MatrixSliceMutIter {
            slice_start: self.start,
            start: 0,
            end: count,
            base,
            slice_shape,
            orig_shape,
            _marker: PhantomData,
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
        let count = slice_shape.element_count();

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
                        .add(self.orig_shape.address_unchecked(&orig_index))
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
                        .add(self.orig_shape.address_unchecked(&orig_index))
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

    /// Returns an iterator over chunks of the elements of the matrix slice.
    pub fn chunks(&self, n: usize) -> impl Iterator<Item = Vec<&T>> {
        let mut it = self.iter();
        std::iter::from_fn(move || {
            let v: Vec<&T> = it.by_ref().take(n).collect();
            (!v.is_empty()).then_some(v)
        })
    }

    /// Returns a parallel iterator over chunks of the elements of the matrix slice.
    pub fn par_chunks(&self, n: usize) -> impl IndexedParallelIterator<Item = Vec<&T>>
    where
        T: Send + Sync,
    {
        self.par_iter().chunks(n)
    }

    /// Returns an iterator over mutable chunks of the elements of the matrix slice.
    pub fn chunks_mut(&mut self, n: usize) -> impl Iterator<Item = Vec<&mut T>> {
        let mut it = self.iter_mut();
        std::iter::from_fn(move || {
            let v: Vec<&mut T> = it.by_ref().take(n).collect();
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

    /// Swaps the elements at the indices.
    pub fn swap(
        &mut self,
        index1: (usize, usize),
        index2: (usize, usize),
    ) -> Result<(), TensorErrors> {
        self.orig.swap(
            (index1.0 + self.start.0, index1.1 + self.start.1),
            (index2.0 + self.start.0, index2.1 + self.start.1),
        )
    }

    /// Returns an iterator over the rows.
    pub fn iter_rows(&self) -> impl ExactSizeIterator<Item = &[T]> + DoubleEndedIterator {
        let orig_cols = self.orig.cols;
        let start_col = self.start.1;
        let end_col = self.end.1;
        (self.start.0..self.end.0).map(move |row| unsafe {
            self.orig
                .elements
                .get_unchecked(row * orig_cols + start_col..row * orig_cols + end_col)
        })
    }

    /// Returns a parallel iterator over the rows.
    pub fn par_iter_rows(&self) -> impl IndexedParallelIterator<Item = &[T]>
    where
        T: Send + Sync,
    {
        let orig_cols = self.orig.cols;
        let start_col = self.start.1;
        let end_col = self.end.1;
        (self.start.0..self.end.0)
            .into_par_iter()
            .map(move |row| unsafe {
                self.orig
                    .elements
                    .get_unchecked(row * orig_cols + start_col..row * orig_cols + end_col)
            })
    }

    /// Returns a mutable iterator over the rows.
    pub fn iter_rows_mut(
        &mut self,
    ) -> impl ExactSizeIterator<Item = &mut [T]> + DoubleEndedIterator + '_ {
        let rows = self.end.0 - self.start.0;
        let orig_cols = self.orig.cols;
        let start_row = self.start.0;
        let start_col = self.start.1;
        let width = self.end.1 - self.start.1;
        let elements = self.orig.elements.as_mut_ptr();
        (0..rows).map(move |row| {
            // The selected row ranges are disjoint; zero-width slices are empty.
            unsafe {
                std::slice::from_raw_parts_mut(
                    elements.add((start_row + row) * orig_cols + start_col),
                    width,
                )
            }
        })
    }

    /// Returns a parallel mutable iterator over the rows.
    pub fn par_iter_rows_mut(&mut self) -> impl IndexedParallelIterator<Item = &mut [T]>
    where
        T: Send + Sync,
    {
        let rows = self.end.0 - self.start.0;
        let orig_cols = self.orig.cols;
        let start_row = self.start.0;
        let start_col = self.start.1;
        let width = self.end.1 - self.start.1;
        let mut elements = self.orig.elements.as_mut_slice();
        let mut row_slices = Vec::with_capacity(rows);

        let (_, remaining) = elements.split_at_mut(start_row * orig_cols);
        elements = remaining;

        for _ in 0..rows {
            let (row, remaining) = elements.split_at_mut(orig_cols);
            let (_, columns) = row.split_at_mut(start_col);
            let (selected, _) = columns.split_at_mut(width);
            row_slices.push(selected);
            elements = remaining;
        }

        row_slices.into_par_iter()
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

impl<T: Clone> MatrixSliceMut<'_, T> {
    /// Converts the matrix slice into a new matrix.
    pub fn clone_into_matrix(&self) -> Matrix<T> {
        let mut elements = Vec::with_capacity(self.rows() * self.cols());
        for i in self.start.0..self.end.0 {
            let start = i * self.orig.cols + self.start.1;
            let end = i * self.orig.cols + self.end.1;
            unsafe {
                elements.extend_from_slice(self.orig.elements.get_unchecked(start..end));
            }
        }

        Matrix {
            rows: self.rows(),
            cols: self.cols(),
            elements,
        }
    }
}

#[cfg(test)]
mod matrix_slice_mut_tests {
    use rayon::iter::{IndexedParallelIterator, ParallelIterator};
    use tensor_math::definitions::errors::TensorErrors;
    use tensor_math::definitions::matrix::Matrix;
    use tensor_math::shape;

    #[test]
    fn slice_mut_dimensions() {
        let mut m1 = Matrix::new(5, 5, (0..25).collect()).unwrap();
        let slice = m1.slice_mut(1..4, 2..5).unwrap();

        assert_eq!(slice.rows(), 3);
        assert_eq!(slice.cols(), 3);
        assert_eq!(slice.shape(), shape![3, 3]);
    }

    #[test]
    fn empty_slice_mut() {
        let mut m1 = Matrix::<usize>::new(0, 0, vec![]).unwrap();
        let slice = m1.slice_mut(0..0, 0..0).unwrap();
        assert_eq!(slice.end(), (0, 0));
        assert_eq!(slice.start(), (0, 0));
        assert_eq!(slice.rows(), 0);
        assert_eq!(slice.cols(), 0);
    }

    #[test]
    fn index_mut_slice() {
        let mut m1 = Matrix::new(3, 3, (0..9).collect()).unwrap();

        let mut slice = m1.slice_mut(1..3, 1..3).unwrap();

        assert_eq!(slice[&[0, 0]], 4);
        assert_eq!(slice[(0, 1)], 5);

        slice[(1, 0)] = -1;
        slice[(1, 1)] = -2;

        let ans = Matrix::new(3, 3, vec![0, 1, 2, 3, 4, 5, 6, -1, -2]).unwrap();

        assert_eq!(m1, ans);
    }

    #[test]
    fn convert_into_mat() {
        let mut m1 = Matrix::new(3, 3, (0..9).collect()).unwrap();

        let mut slice = m1.slice_mut(1..3, 1..3).unwrap();
        slice[(0, 1)] = 10;

        let ans = Matrix::new(2, 2, vec![4, 10, 7, 8]).unwrap();

        assert_eq!(slice.clone_into_matrix(), ans);

        let slice = m1.slice_mut(1..3, 1..3).unwrap();

        assert_eq!(slice.clone_into_matrix(), ans);
    }

    #[test]
    fn get_in_slice() {
        let mut m1 = Matrix::new(3, 3, (0..9).collect()).unwrap();

        let slice = m1.slice_mut(1..3, 1..3).unwrap();

        assert_eq!(slice.get((0, 0)), Some(&4));
        assert_eq!(slice.get((2, 2)), None);
    }

    #[test]
    fn rows_and_cols() {
        let mut m1 = Matrix::new(5, 8, (0..40).collect()).unwrap();
        let slice = m1.slice_mut(1..4, 2..7).unwrap();

        assert_eq!(slice.rows(), 3);
        assert_eq!(slice.cols(), 5);
        assert_eq!(slice.shape(), shape![3, 5]);
    }

    #[test]
    fn for_each_mut_basic() {
        let mut m1 = Matrix::new(3, 3, (0..9).collect()).unwrap();

        let mut slice = m1.slice_mut(1..3, 1..3).unwrap();
        slice.iter_mut().for_each(|x| *x += 10);

        let ans = Matrix::new(3, 3, vec![0, 1, 2, 3, 14, 15, 6, 17, 18]).unwrap();

        assert_eq!(m1, ans);
    }

    #[test]
    fn for_each_mut_empty() {
        let mut m1 = Matrix::<usize>::new(0, 0, vec![]).unwrap();
        let mut slice = m1.slice_mut(0..0, 0..0).unwrap();
        let mut count = 0;
        slice.iter_mut().for_each(|_| count += 1);
        assert_eq!(count, 0);
    }

    #[test]
    fn for_each_mut_rectangular() {
        let mut m1 = Matrix::new(4, 4, (0..16).collect()).unwrap();

        let mut slice = m1.slice_mut(1..3, 0..4).unwrap();
        slice.iter_mut().for_each(|x| *x = 0);

        let ans = Matrix::new(
            4,
            4,
            vec![0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 12, 13, 14, 15],
        )
        .unwrap();

        assert_eq!(m1, ans);
    }

    #[test]
    fn enumerated_for_each_mut_basic() {
        let mut m1 = Matrix::new(3, 3, (0..9).collect()).unwrap();

        let mut slice = m1.slice_mut(1..3, 1..3).unwrap();
        slice.enumerated_iter_mut().for_each(|(idx, x)| {
            // idx should be relative to the slice: [0..2, 0..2]
            *x = ((idx.0 + 1) * 10 + (idx.1 + 1)) as i32;
        });

        let ans = Matrix::new(3, 3, vec![0, 1, 2, 3, 11, 12, 6, 21, 22]).unwrap();

        assert_eq!(m1, ans);
    }

    #[test]
    fn enumerated_for_each_mut_empty() {
        let mut m1 = Matrix::<usize>::new(0, 0, vec![]).unwrap();
        let mut slice = m1.slice_mut(0..0, 0..0).unwrap();

        let mut count = 0;
        slice.enumerated_iter_mut().for_each(|_| count += 1);
        assert_eq!(count, 0);
    }

    #[test]
    fn zero_column_slices_yield_one_empty_row_per_row() {
        let matrix = Matrix::new(3, 0, Vec::<i32>::new()).unwrap();
        let slice = matrix.slice(1..3, 0..0).unwrap();

        assert_eq!(
            slice.iter_rows().map(<[i32]>::len).collect::<Vec<_>>(),
            [0; 2]
        );
        assert_eq!(
            slice.par_iter_rows().map(<[i32]>::len).collect::<Vec<_>>(),
            [0; 2]
        );
    }

    #[test]
    fn zero_column_slices_mut_yield_one_empty_row_per_row() {
        let mut matrix = Matrix::new(3, 0, Vec::<i32>::new()).unwrap();
        let mut slice = matrix.slice_mut(1..3, 0..0).unwrap();

        assert_eq!(
            slice
                .iter_rows_mut()
                .map(|row| row.len())
                .collect::<Vec<_>>(),
            [0; 2]
        );
        assert_eq!(
            slice
                .par_iter_rows_mut()
                .map(|row| row.len())
                .collect::<Vec<_>>(),
            [0; 2]
        );
    }

    #[test]
    fn square_and_zero_dimension_slice_shapes() {
        let mut matrix = Matrix::new(3, 4, (0..12).collect()).unwrap();

        let slice = matrix.slice_mut(0..2, 1..3).unwrap();
        assert!(slice.is_square());
        assert_eq!(slice.shape(), shape![2, 2]);
        drop(slice);

        let no_rows = matrix.slice_mut(1..1, 0..4).unwrap();
        assert_eq!(no_rows.start(), (1, 0));
        assert_eq!(no_rows.end(), (1, 4));
        assert_eq!(no_rows.shape(), shape![0, 4]);
        assert!(!no_rows.is_square());
        drop(no_rows);

        let no_columns = matrix.slice_mut(0..3, 2..2).unwrap();
        assert_eq!(no_columns.shape(), shape![3, 0]);
        assert!(!no_columns.is_square());
    }

    #[test]
    fn immutable_view_methods_read_the_selected_region() {
        let mut matrix = Matrix::new(3, 4, (0..12).collect()).unwrap();
        let slice = matrix.slice_mut(1..3, 1..3).unwrap();
        let view = slice.as_immut();

        assert_eq!(view.start(), (1, 1));
        assert_eq!(view.end(), (3, 3));
        assert_eq!(view.iter().copied().collect::<Vec<_>>(), vec![5, 6, 9, 10]);

        let sum = slice.use_as_immut(|view| view.iter().copied().sum::<i32>());
        assert_eq!(sum, 30);
    }

    #[test]
    fn get_and_get_mut_check_each_axis_and_handle_empty_regions() {
        let mut matrix = Matrix::new(3, 3, (0..9).collect()).unwrap();
        let mut slice = matrix.slice_mut(1..3, 1..3).unwrap();

        assert_eq!(slice.get((0, 0)), Some(&4));
        assert_eq!(slice.get((2, 0)), None);
        assert_eq!(slice.get((0, 2)), None);
        *slice.get_mut((1, 1)).unwrap() = 99;
        assert_eq!(slice.get((1, 1)), Some(&99));
        assert!(slice.get_mut((2, 0)).is_none());
        assert!(slice.get_mut((0, 2)).is_none());
        drop(slice);
        assert_eq!(matrix[(2, 2)], 99);

        let mut no_rows_matrix = Matrix::<i32>::new(0, 3, vec![]).unwrap();
        let mut no_rows = no_rows_matrix.slice_mut(0..0, 0..3).unwrap();
        assert_eq!(no_rows.get((0, 0)), None);
        assert!(no_rows.get_mut((0, 0)).is_none());
        drop(no_rows);

        let mut no_columns_matrix = Matrix::<i32>::new(3, 0, vec![]).unwrap();
        let mut no_columns = no_columns_matrix.slice_mut(0..3, 0..0).unwrap();
        assert_eq!(no_columns.get((0, 0)), None);
        assert!(no_columns.get_mut((0, 0)).is_none());
    }

    #[test]
    fn element_iterators_cover_selected_elements_in_row_major_order() {
        let mut matrix = Matrix::new(3, 4, (0..12).collect()).unwrap();
        let slice = matrix.slice_mut(1..3, 1..3).unwrap();
        assert_eq!(slice.iter().copied().collect::<Vec<_>>(), vec![5, 6, 9, 10]);
        assert_eq!(
            slice.par_iter().copied().collect::<Vec<_>>(),
            vec![5, 6, 9, 10]
        );
        assert_eq!(slice.iter().next_back(), Some(&10));
        assert_eq!(slice.par_iter().len(), 4);
        drop(slice);

        let mut matrix = Matrix::new(3, 4, (0..12).collect()).unwrap();
        let mut slice = matrix.slice_mut(1..3, 1..3).unwrap();
        slice.iter_mut().for_each(|element| *element += 10);
        drop(slice);
        assert_eq!(
            matrix.elements(),
            &[0, 1, 2, 3, 4, 15, 16, 7, 8, 19, 20, 11]
        );

        let mut slice = matrix.slice_mut(1..3, 1..3).unwrap();
        slice.par_iter_mut().for_each(|element| *element *= 2);
        drop(slice);
        assert_eq!(
            matrix.elements(),
            &[0, 1, 2, 3, 4, 30, 32, 7, 8, 38, 40, 11]
        );
    }

    #[test]
    fn reversed_iterators_work_as_expected() {
        let mut matrix = Matrix::new(3, 4, (0..12).collect()).unwrap();
        let slice = matrix.slice_mut(1..3, 1..3).unwrap();
        assert_eq!(
            slice.iter().rev().copied().collect::<Vec<_>>(),
            vec![10, 9, 6, 5]
        );
        assert_eq!(
            slice.par_iter().rev().copied().collect::<Vec<_>>(),
            vec![10, 9, 6, 5]
        );
        drop(slice);

        let mut slice = matrix.slice_mut(1..3, 1..3).unwrap();
        slice.iter_mut().rev().for_each(|element| *element += 10);
        drop(slice);
        assert_eq!(
            matrix.elements(),
            &[0, 1, 2, 3, 4, 15, 16, 7, 8, 19, 20, 11]
        );

        let mut slice = matrix.slice_mut(1..3, 1..3).unwrap();
        slice.par_iter_mut().rev().for_each(|element| *element *= 2);
        drop(slice);
        assert_eq!(
            matrix.elements(),
            &[0, 1, 2, 3, 4, 30, 32, 7, 8, 38, 40, 11]
        );
    }

    #[test]
    fn element_iterators_are_empty_for_zero_sized_regions() {
        for (rows, cols, selected_rows, selected_cols) in
            [(0, 3, 0..0, 0..3), (3, 0, 0..3, 0..0), (0, 0, 0..0, 0..0)]
        {
            let mut matrix = Matrix::<i32>::new(rows, cols, vec![]).unwrap();
            let mut slice = matrix
                .slice_mut(selected_rows.clone(), selected_cols.clone())
                .unwrap();
            assert_eq!(slice.iter().count(), 0);
            assert_eq!(slice.par_iter().count(), 0);
            assert_eq!(slice.iter_mut().count(), 0);
            assert_eq!(slice.par_iter_mut().count(), 0);
        }
    }

    #[test]
    fn immutable_and_mutable_chunks_cover_elements() {
        let mut matrix = Matrix::new(2, 3, (0..6).collect()).unwrap();
        let slice = matrix.slice_mut(0..2, 1..3).unwrap();

        assert_eq!(
            slice
                .chunks(3)
                .map(|chunk| chunk.into_iter().copied().collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            vec![vec![1, 2, 4], vec![5]]
        );
        assert_eq!(
            slice
                .par_chunks(3)
                .map(|chunk| chunk.into_iter().copied().collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            vec![vec![1, 2, 4], vec![5]]
        );
        assert_eq!(slice.chunks(0).count(), 0);
        drop(slice);

        let mut slice = matrix.slice_mut(0..2, 1..3).unwrap();
        for chunk in slice.chunks_mut(2) {
            for element in chunk {
                *element += 10;
            }
        }
        drop(slice);
        assert_eq!(matrix.elements(), &[0, 11, 12, 3, 14, 15]);

        let mut slice = matrix.slice_mut(0..2, 1..3).unwrap();
        slice
            .par_chunks_mut(2)
            .for_each(|chunk| chunk.into_iter().for_each(|element| *element *= 2));
        drop(slice);
        assert_eq!(matrix.elements(), &[0, 22, 24, 3, 28, 30]);
    }

    #[test]
    fn chunks_are_empty_for_zero_sized_regions() {
        let mut no_rows_matrix = Matrix::<i32>::new(0, 3, vec![]).unwrap();
        let no_rows = no_rows_matrix.slice_mut(0..0, 0..3).unwrap();
        assert_eq!(no_rows.chunks(2).count(), 0);
        assert_eq!(no_rows.par_chunks(2).count(), 0);
        drop(no_rows);

        let mut no_columns_matrix = Matrix::<i32>::new(3, 0, vec![]).unwrap();
        let no_columns = no_columns_matrix.slice_mut(0..3, 0..0).unwrap();
        assert_eq!(no_columns.chunks(2).count(), 0);
        assert_eq!(no_columns.par_chunks(2).count(), 0);
        drop(no_columns);

        let mut no_rows = no_rows_matrix.slice_mut(0..0, 0..3).unwrap();
        assert_eq!(no_rows.chunks_mut(2).count(), 0);
        assert_eq!(no_rows.par_chunks_mut(2).count(), 0);
        drop(no_rows);
        let mut no_columns = no_columns_matrix.slice_mut(0..3, 0..0).unwrap();
        assert_eq!(no_columns.chunks_mut(2).count(), 0);
        assert_eq!(no_columns.par_chunks_mut(2).count(), 0);
    }

    #[test]
    fn swap_exchanges_elements_and_reports_matrix_bounds_errors() {
        let mut matrix = Matrix::new(3, 3, (0..9).collect()).unwrap();
        let mut slice = matrix.slice_mut(1..3, 1..3).unwrap();

        slice.swap((0, 0), (1, 1)).unwrap();
        assert_eq!(slice[(0, 0)], 8);
        assert_eq!(slice[(1, 1)], 4);
        slice.swap((0, 0), (0, 0)).unwrap();
        assert_eq!(
            slice.swap((1, 1), (2, 1)),
            Err(TensorErrors::IndexOutOfBounds {
                index: 3,
                axis: 0,
                length: 3
            })
        );
    }

    #[test]
    fn row_iterators_cover_only_selected_rows_and_columns() {
        let mut matrix = Matrix::new(3, 4, (0..12).collect()).unwrap();
        let slice = matrix.slice_mut(1..3, 1..3).unwrap();
        let expected = vec![&[5, 6][..], &[9, 10][..]];
        assert_eq!(slice.iter_rows().collect::<Vec<_>>(), expected);
        assert_eq!(slice.par_iter_rows().collect::<Vec<_>>(), expected);
        assert_eq!(slice.iter_rows().next_back(), Some(&[9, 10][..]));
        drop(slice);

        let mut slice = matrix.slice_mut(1..3, 1..3).unwrap();
        for row in slice.iter_rows_mut() {
            row[0] = -row[0];
        }
        drop(slice);
        assert_eq!(matrix.elements(), &[0, 1, 2, 3, 4, -5, 6, 7, 8, -9, 10, 11]);

        let mut slice = matrix.slice_mut(1..3, 1..3).unwrap();
        slice.par_iter_rows_mut().for_each(|row| row[1] = -row[1]);
        drop(slice);
        assert_eq!(
            matrix.elements(),
            &[0, 1, 2, 3, 4, -5, -6, 7, 8, -9, -10, 11]
        );

        let mut matrix = Matrix::<i32>::new(0, 3, vec![]).unwrap();
        let mut slice = matrix.slice_mut(0..0, 0..3).unwrap();
        assert_eq!(slice.iter_rows().count(), 0);
        assert_eq!(slice.par_iter_rows().count(), 0);
        assert_eq!(slice.iter_rows_mut().count(), 0);
        assert_eq!(slice.par_iter_rows_mut().count(), 0);
    }

    #[test]
    fn both_mutable_index_forms_update_the_matrix() {
        let mut matrix = Matrix::new(2, 3, (0..6).collect()).unwrap();
        {
            let mut slice = matrix.slice_mut(0..2, 1..3).unwrap();
            slice[(0, 0)] = 10;
            slice[&[1, 1]] = 20;
            assert_eq!(slice[(0, 0)], 10);
            assert_eq!(slice[&[1, 1]], 20);
        }

        assert_eq!(matrix.elements(), &[0, 10, 2, 3, 4, 20]);
    }

    #[test]
    #[should_panic]
    fn tuple_mutable_index_panics_outside_slice() {
        let mut matrix = Matrix::new(2, 2, vec![1, 2, 3, 4]).unwrap();
        let mut slice = matrix.slice_mut(0..1, 0..1).unwrap();
        slice[(1, 0)] = 10;
    }

    #[test]
    #[should_panic]
    fn array_mutable_index_panics_outside_slice() {
        let mut matrix = Matrix::new(2, 2, vec![1, 2, 3, 4]).unwrap();
        let mut slice = matrix.slice_mut(0..1, 0..1).unwrap();
        slice[&[0, 1]] = 10;
    }

    #[test]
    fn clone_into_matrix_preserves_zero_dimension_slice_shapes() {
        let mut matrix = Matrix::new(3, 4, (0..12).collect()).unwrap();
        assert_eq!(
            matrix.slice_mut(1..1, 0..4).unwrap().clone_into_matrix(),
            Matrix::new(0, 4, vec![]).unwrap()
        );
        assert_eq!(
            matrix.slice_mut(0..3, 2..2).unwrap().clone_into_matrix(),
            Matrix::new(3, 0, vec![]).unwrap()
        );

        let mut no_rows = Matrix::<i32>::new(0, 3, vec![]).unwrap();
        assert_eq!(
            no_rows.slice_mut(0..0, 1..3).unwrap().clone_into_matrix(),
            Matrix::new(0, 2, vec![]).unwrap()
        );
    }
}
