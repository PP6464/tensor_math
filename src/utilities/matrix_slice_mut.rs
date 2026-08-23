use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::definitions::matrix_slice::MatrixSlice;
use crate::definitions::matrix_slice_mut::MatrixSliceMut;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;

/*
--------------------------------------------
* Mutable matrix slice utilities
--------------------------------------------
*/

impl<T> MatrixSliceMut<'_, T> {
    /// Applies the given function over the entire matrix elementwise by reference.
    pub fn map_refs<F>(&self, f: impl FnMut(&T) -> F) -> Matrix<F> {
        self.iter().map(f).collect()
    }

    /// Applies the given function over the entire matrix elementwise by reference.
    pub fn par_map_refs<F: Send>(&self, f: impl Fn(&T) -> F + Send + Sync) -> Matrix<F>
    where
        T: Send + Sync,
    {
        self.par_iter().map(f).collect()
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

    /// Returns a mutable iterator that is enumerated with matrix indices.
    pub fn enumerated_iter_mut(&mut self) -> impl Iterator<Item = ((usize, usize), &mut T)> {
        let cols = self.cols();
        self.iter_mut()
            .enumerate()
            .map(move |(index, elem)| ((index / cols, index % cols), elem))
    }

    /// Returns a parallel mutable iterator that is enumerated with matrix indices.
    pub fn enumerated_par_iter_mut(
        &mut self,
    ) -> impl ParallelIterator<Item = ((usize, usize), &mut T)>
    where
        T: Send + Sync,
    {
        let cols = self.cols();
        self.par_iter_mut()
            .enumerate()
            .map(move |(index, elem)| ((index / cols, index % cols), elem))
    }

    /// Sets all the values in the mutable matrix slice to the given values.
    /// This fails if the shape of the values does not match the matrix slice's shape.
    pub fn set_all(&mut self, values: &Matrix<T>) -> Result<(), TensorErrors>
    where
        T: Clone,
    {
        if self.rows() != values.rows() || self.cols() != values.cols() {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: values.shape(),
                op: "set_all",
            });
        }

        for (index, value) in values.enumerated_iter() {
            unsafe { *self.get_unchecked_mut(index) = value.clone() }
        }

        Ok(())
    }

    /// Sets all the values in the mutable matrix slice to the given values.
    /// This fails if the shape of the values does not match the matrix slice's shape.
    pub fn set_all_from_slice(&mut self, values: &MatrixSlice<T>) -> Result<(), TensorErrors>
    where
        T: Clone,
    {
        if self.rows() != values.rows() || self.cols() != values.cols() {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: values.shape(),
                op: "set_all",
            });
        }

        for (index, value) in values.enumerated_iter() {
            unsafe { *self.get_unchecked_mut(index) = value.clone() }
        }

        Ok(())
    }
}
