use crate::definitions::errors::TensorErrors;
use crate::definitions::errors::TensorErrors::IncompatibleShapes;
use crate::definitions::tensor::Tensor;
use crate::definitions::tensor_slice::TensorSlice;
use crate::definitions::tensor_slice_mut::TensorSliceMut;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;

/*
--------------------------------------------
* Mutable tensor slice utilities
--------------------------------------------
*/

impl<T> TensorSliceMut<'_, T> {
    /// Applies the given function over the entire tensor elementwise by reference.
    pub fn map_refs<F>(&self, f: impl FnMut(&T) -> F) -> Tensor<F> {
        self.iter().map(f).collect()
    }

    /// Applies the given function over the entire tensor elementwise by reference.
    pub fn par_map_refs<F: Send>(&self, f: impl Fn(&T) -> F + Send + Sync) -> Tensor<F>
    where
        T: Send + Sync,
    {
        self.par_iter().map(f).collect()
    }

    /// Returns an iterator that is enumerated with tensor indices.
    pub fn enumerated_iter(&self) -> impl Iterator<Item = (Vec<usize>, &T)> {
        let shape = self.shape();
        unsafe {
            self.iter()
                .enumerate()
                .map(move |(index, elem)| (shape.tensor_index_unchecked(index), elem))
        }
    }

    /// Returns a parallel iterator that is enumerated with tensor indices.
    pub fn enumerated_par_iter(&self) -> impl ParallelIterator<Item = (Vec<usize>, &T)> + '_
    where
        T: Send + Sync,
    {
        let shape = self.shape();
        unsafe {
            self.par_iter()
                .enumerate()
                .map(move |(index, elem)| (shape.tensor_index_unchecked(index), elem))
        }
    }

    /// Returns a mutable iterator that is enumerated with tensor indices.
    pub fn enumerated_iter_mut(&mut self) -> impl Iterator<Item = (Vec<usize>, &mut T)> + '_ {
        let shape = self.shape();
        unsafe {
            self.iter_mut()
                .enumerate()
                .map(move |(index, elem)| (shape.tensor_index_unchecked(index), elem))
        }
    }

    /// Returns a parallel mutable iterator that is enumerated with tensor indices.
    pub fn enumerated_par_iter_mut(
        &mut self,
    ) -> impl ParallelIterator<Item = (Vec<usize>, &mut T)> + '_
    where
        T: Send + Sync,
    {
        let shape = self.shape();
        unsafe {
            self.par_iter_mut()
                .enumerate()
                .map(move |(index, elem)| (shape.tensor_index_unchecked(index), elem))
        }
    }

    /// Sets all the values in the tensor slice to the given values.
    /// This fails if the shape of the values does not match the tensor slice's shape.
    pub fn set_all(&mut self, values: &Tensor<T>) -> Result<(), TensorErrors>
    where
        T: Clone,
    {
        if self.shape() != values.shape() {
            return Err(IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: values.shape(),
                op: "set_all",
            });
        }

        for (index, value) in values.enumerated_iter() {
            unsafe { *self.get_unchecked_mut(&index) = value.clone() }
        }

        Ok(())
    }

    /// Sets all the values in the tensor slice to the given values.
    /// This fails if the shape of the values does not match the tensor slice's shape.
    pub fn set_all_from_slice(&mut self, values: &TensorSlice<T>) -> Result<(), TensorErrors>
    where
        T: Clone,
    {
        if self.shape() != values.shape() {
            return Err(IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: values.shape(),
                op: "set_all",
            });
        }

        for (index, value) in values.enumerated_iter() {
            unsafe { *self.get_unchecked_mut(&index) = value.clone() }
        }

        Ok(())
    }
}
