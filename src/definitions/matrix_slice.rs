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
        (0..self.rows() * self.cols())
            .into_iter()
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
                elements.extend_from_slice(&self.orig.elements.get_unchecked(start..end));
            }
        }

        Matrix {
            rows: self.rows(),
            cols: self.cols(),
            elements,
        }
    }
}
