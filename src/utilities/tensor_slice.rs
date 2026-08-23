use crate::definitions::tensor::Tensor;
use crate::definitions::tensor_slice::TensorSlice;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;

/*
--------------------------------------------
* Immutable tensor slice utilities
--------------------------------------------
*/

impl<T> TensorSlice<'_, T> {
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
}
