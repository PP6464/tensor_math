use crate::definitions::matrix::Matrix;
use crate::definitions::matrix_slice::MatrixSlice;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;

/*
--------------------------------------------
* Immutable matrix slice utilities
--------------------------------------------
*/

impl<T> MatrixSlice<'_, T> {
    /// Applies the given function over the entire matrix elementwise by reference.
    pub fn map_refs<F>(&self, f: impl FnMut(&T) -> F) -> Matrix<F> {
        unsafe { self.iter().map(f).collect::<Matrix<_>>().reshape_unchecked(self.rows(), self.cols()) }
    }

    /// Applies the given function over the entire matrix elementwise by reference.
    pub fn par_map_refs<F: Send>(&self, f: impl Fn(&T) -> F + Send + Sync) -> Matrix<F>
    where
        T: Send + Sync,
    {
        unsafe { self.par_iter().map(f).collect::<Matrix<_>>().reshape_unchecked(self.rows(), self.cols()) }
    }

    /// Applies the given function over the entire matrix elementwise by reference, enumerated with matrix indices.
    pub fn enumerated_map_refs<F>(&self, f: impl FnMut(((usize, usize), &T)) -> F) -> Matrix<F> {
        unsafe { self.enumerated_iter().map(f).collect::<Matrix<_>>().reshape_unchecked(self.rows(), self.cols()) }
    }

    /// Applies the given function over the entire matrix elementwise by reference, enumerated with matrix indices.
    pub fn enumerated_par_map_refs<F: Send>(&self, f: impl Fn(((usize, usize), &T)) -> F + Send + Sync) -> Matrix<F>
    where
        T: Send + Sync,
    {
        unsafe { self.enumerated_par_iter().map(f).collect::<Matrix<_>>().reshape_unchecked(self.rows(), self.cols()) }
    }

    /// Returns an iterator that is enumerated with matrix indices.
    pub fn enumerated_iter(&self) -> impl Iterator<Item = ((usize, usize), &T)> {
        let cols = self.cols();
        self.iter()
            .enumerate()
            .map(move |(index, elem)| ((index / cols, index % cols), elem))
    }

    /// Returns a parallel iterator that is enumerated with matrix indices.
    pub fn enumerated_par_iter(&self) -> impl ParallelIterator<Item = ((usize, usize), &T)>
    where
        T: Send + Sync,
    {
        let cols = self.cols();
        self.par_iter()
            .enumerate()
            .map(move |(index, elem)| ((index / cols, index % cols), elem))
    }
}
