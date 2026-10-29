use crate::definitions::matrix::Matrix;
use crate::definitions::shape::Shape;
use crate::shape;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::ops::Index;
/*
--------------------------------------------
* Immutable matrix slice definition
--------------------------------------------
*/

#[derive(Debug, Eq, PartialEq, Hash, Clone)]
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
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &T> + DoubleEndedIterator {
        (0..self.rows() * self.cols())
            .map(move |i| unsafe { self.get_unchecked((i / self.cols(), i % self.cols())) })
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
    pub fn par_chunks(&self, n: usize) -> impl IndexedParallelIterator<Item = Vec<&T>>
    where
        T: Send + Sync,
    {
        self.par_iter().chunks(n)
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
mod matrix_slice_tests {
    use rayon::iter::{IndexedParallelIterator, ParallelIterator};
    use tensor_math::definitions::matrix::Matrix;
    use tensor_math::shape;

    #[test]
    fn slice_bounds_and_dimensions() {
        let matrix = Matrix::new(3, 4, (0..12).collect()).unwrap();
        let slice = matrix.slice(1..3, 1..4).unwrap();

        assert_eq!(slice.start(), (1, 1));
        assert_eq!(slice.end(), (3, 4));
        assert_eq!(slice.rows(), 2);
        assert_eq!(slice.cols(), 3);
        assert_eq!(slice.shape(), shape![2, 3]);
        assert!(!slice.is_square());
        assert!(matrix.slice(1..1, 1..1).unwrap().is_square());
        assert!(matrix.slice(0..2, 0..2).unwrap().is_square());
        let no_rows = matrix.slice(1..1, 0..4).unwrap();
        assert_eq!(no_rows.shape(), shape![0, 4]);
        assert_eq!(no_rows.iter_rows().count(), 0);
        let no_columns = matrix.slice(0..3, 2..2).unwrap();
        assert_eq!(no_columns.shape(), shape![3, 0]);
        assert_eq!(
            no_columns.iter_rows().map(<[i32]>::len).collect::<Vec<_>>(),
            vec![0, 0, 0]
        );
    }

    #[test]
    fn empty_shapes_keep_their_dimensions() {
        let no_rows = Matrix::<i32>::new(0, 3, vec![]).unwrap();
        let row_empty = no_rows.slice(0..0, 0..3).unwrap();
        assert_eq!(row_empty.shape(), shape![0, 3]);
        assert_eq!(row_empty.rows(), 0);
        assert_eq!(row_empty.cols(), 3);
        assert_eq!(row_empty.start(), (0, 0));
        assert_eq!(row_empty.end(), (0, 3));
        assert!(row_empty.iter().next().is_none());
        assert!(row_empty.par_iter().count() == 0);

        let no_columns = Matrix::<i32>::new(3, 0, vec![]).unwrap();
        let column_empty = no_columns.slice(1..3, 0..0).unwrap();
        assert_eq!(column_empty.shape(), shape![2, 0]);
        assert_eq!(column_empty.rows(), 2);
        assert_eq!(column_empty.cols(), 0);
        assert!(column_empty.iter().next().is_none());
        assert_eq!(column_empty.par_iter().count(), 0);

        let empty = Matrix::<i32>::new(0, 0, vec![]).unwrap();
        let empty_slice = empty.slice(0..0, 0..0).unwrap();
        assert_eq!(empty_slice.shape(), shape![0, 0]);
        assert!(empty_slice.is_square());
        assert_eq!(empty_slice.iter().len(), 0);
    }

    #[test]
    fn get_returns_relative_elements_and_none_outside_slice() {
        let matrix = Matrix::new(3, 4, (0..12).collect()).unwrap();
        let slice = matrix.slice(1..3, 1..4).unwrap();

        assert_eq!(slice.get((0, 0)), Some(&5));
        assert_eq!(slice.get((1, 2)), Some(&11));
        assert_eq!(slice.get((2, 0)), None);
        assert_eq!(slice.get((0, 3)), None);

        let empty = Matrix::<i32>::new(3, 0, vec![]).unwrap();
        assert_eq!(empty.slice(0..3, 0..0).unwrap().get((0, 0)), None);
    }

    #[test]
    fn iterators_yield_elements_in_row_major_order() {
        let matrix = Matrix::new(3, 4, (0..12).collect()).unwrap();
        let slice = matrix.slice(1..3, 1..3).unwrap();

        assert_eq!(slice.iter().copied().collect::<Vec<_>>(), vec![5, 6, 9, 10]);
        assert_eq!(
            slice.par_iter().copied().collect::<Vec<_>>(),
            vec![5, 6, 9, 10]
        );
        assert_eq!(slice.iter().next_back(), Some(&10));
        assert_eq!(slice.par_iter().len(), 4);

        let no_rows = Matrix::new(0, 3, Vec::<i32>::new()).unwrap();
        assert_eq!(no_rows.slice(0..0, 1..3).unwrap().iter().count(), 0);
        let no_columns = Matrix::new(3, 0, Vec::<i32>::new()).unwrap();
        assert_eq!(no_columns.slice(0..3, 0..0).unwrap().par_iter().count(), 0);
    }

    #[test]
    fn chunks_cover_elements_and_empty_slices() {
        let matrix = Matrix::new(2, 3, (0..6).collect()).unwrap();
        let slice = matrix.slice(0..2, 1..3).unwrap();

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

        let no_rows = Matrix::<i32>::new(0, 3, vec![]).unwrap();
        assert_eq!(no_rows.slice(0..0, 0..3).unwrap().chunks(2).count(), 0);
        assert_eq!(no_rows.slice(0..0, 0..3).unwrap().par_chunks(2).count(), 0);
        let no_columns = Matrix::<i32>::new(2, 0, vec![]).unwrap();
        assert_eq!(no_columns.slice(0..2, 0..0).unwrap().chunks(2).count(), 0);
        assert_eq!(
            no_columns.slice(0..2, 0..0).unwrap().par_chunks(2).count(),
            0
        );
    }

    #[test]
    fn row_iterators_return_slice_columns_for_each_selected_row() {
        let matrix = Matrix::new(3, 4, (0..12).collect()).unwrap();
        let slice = matrix.slice(1..3, 1..3).unwrap();
        let expected = vec![&[5, 6][..], &[9, 10][..]];

        assert_eq!(slice.iter_rows().collect::<Vec<_>>(), expected);
        assert_eq!(slice.par_iter_rows().collect::<Vec<_>>(), expected);
        assert_eq!(slice.iter_rows().next_back(), Some(&[9, 10][..]));

        let no_rows = Matrix::<i32>::new(0, 3, vec![]).unwrap();
        assert_eq!(no_rows.slice(0..0, 0..3).unwrap().iter_rows().count(), 0);
        let no_columns = Matrix::<i32>::new(3, 0, vec![]).unwrap();
        assert_eq!(
            no_columns
                .slice(1..3, 0..0)
                .unwrap()
                .par_iter_rows()
                .map(<[i32]>::len)
                .collect::<Vec<_>>(),
            vec![0, 0]
        );
    }

    #[test]
    fn indexing_supports_tuple_and_array_indices() {
        let matrix = Matrix::new(3, 4, (0..12).collect()).unwrap();
        let slice = matrix.slice(1..3, 1..4).unwrap();

        assert_eq!(slice[(0, 0)], 5);
        assert_eq!(slice[(1, 2)], 11);
        assert_eq!(slice[&[0, 2]], 7);
    }

    #[test]
    #[should_panic]
    fn tuple_indexing_panics_outside_slice() {
        let matrix = Matrix::new(2, 2, vec![1, 2, 3, 4]).unwrap();
        let slice = matrix.slice(0..1, 0..1).unwrap();
        let _ = slice[(1, 0)];
    }

    #[test]
    #[should_panic]
    fn array_indexing_panics_outside_slice() {
        let matrix = Matrix::new(2, 2, vec![1, 2, 3, 4]).unwrap();
        let slice = matrix.slice(0..1, 0..1).unwrap();
        let _ = slice[&[0, 1]];
    }

    #[test]
    fn clone_into_matrix_copies_slice_and_preserves_empty_shapes() {
        let matrix = Matrix::new(3, 4, (0..12).collect()).unwrap();
        let cloned = matrix.slice(1..3, 1..3).unwrap().clone_into_matrix();
        assert_eq!(cloned, Matrix::new(2, 2, vec![5, 6, 9, 10]).unwrap());
        assert_eq!(
            matrix.slice(1..1, 0..4).unwrap().clone_into_matrix(),
            Matrix::new(0, 4, vec![]).unwrap()
        );
        assert_eq!(
            matrix.slice(0..3, 2..2).unwrap().clone_into_matrix(),
            Matrix::new(3, 0, vec![]).unwrap()
        );

        let no_rows = Matrix::<i32>::new(0, 3, vec![]).unwrap();
        assert_eq!(
            no_rows.slice(0..0, 1..3).unwrap().clone_into_matrix(),
            Matrix::new(0, 2, vec![]).unwrap()
        );
        let no_columns = Matrix::<i32>::new(3, 0, vec![]).unwrap();
        assert_eq!(
            no_columns.slice(1..3, 0..0).unwrap().clone_into_matrix(),
            Matrix::new(2, 0, vec![]).unwrap()
        );
        let empty = Matrix::<i32>::new(0, 0, vec![]).unwrap();
        assert_eq!(
            empty.slice(0..0, 0..0).unwrap().clone_into_matrix(),
            Matrix::new(0, 0, vec![]).unwrap()
        );
    }
}
