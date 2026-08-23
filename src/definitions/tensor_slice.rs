use crate::definitions::shape::Shape;
use crate::definitions::strides::Strides;
use crate::definitions::tensor::Tensor;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::ops::Index;

/*
--------------------------------------------
* Immutable tensor slice definition
--------------------------------------------
*/

#[derive(Debug, Eq, PartialEq)]
pub struct TensorSlice<'a, T> {
    pub(crate) orig: &'a Tensor<T>,
    pub(crate) start: Vec<usize>,
    pub(crate) end: Vec<usize>,
}

impl<T> TensorSlice<'_, T> {
    /// Returns the start position of the tensor slice.
    pub fn start(&self) -> &[usize] {
        &self.start
    }

    /// Returns the end position of the tensor slice.
    pub fn end(&self) -> &[usize] {
        &self.end
    }

    /// Returns the shape of the tensor slice.
    pub fn shape(&self) -> Shape {
        Shape::new(
            self.end
                .iter()
                .zip(self.start.iter())
                .map(|(e, s)| e - s)
                .collect(),
        )
    }

    /// Returns the rank of the tensor slice.
    pub fn rank(&self) -> usize {
        self.end.len()
    }

    /// Gets a reference to the element at the specified indices in the tensor slice, returning `None` if the indices are out of bounds.
    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        if indices.len() != self.orig.rank() {
            return None;
        }

        let orig_index = indices
            .iter()
            .zip(self.start.iter())
            .map(|(x, y)| x + y)
            .collect::<Vec<usize>>();

        for i in 0..self.orig.rank() {
            if self.end[i] <= orig_index[i] {
                return None;
            }
        }

        unsafe { Some(self.orig.get_unchecked(orig_index.as_slice())) }
    }

    /// Gets a reference to the element at the specified indices in the tensor slice without bounds checking.
    pub(crate) unsafe fn get_unchecked(&self, indices: &[usize]) -> &T {
        self.orig.get_unchecked(
            &indices
                .iter()
                .zip(self.start.iter())
                .map(|(x, y)| x + y)
                .collect::<Vec<_>>(),
        )
    }

    /// Returns an iterator over the elements of the tensor slice.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        let shape = self.shape();
        (0..shape.element_count())
            .map(move |i| unsafe { self.get_unchecked(&shape.tensor_index_unchecked(i)) })
    }

    /// Returns a parallel iterator over the elements of the tensor slice.
    pub fn par_iter(&self) -> impl IndexedParallelIterator<Item = &T>
    where
        T: Send + Sync,
    {
        let shape = self.shape();
        unsafe {
            (0..shape.element_count())
                .into_par_iter()
                .map(move |i| self.get_unchecked(&shape.tensor_index_unchecked(i)))
        }
    }

    /// Returns an iterator over chunks of elements.
    pub fn chunks(&self, n: usize) -> impl Iterator<Item = Vec<&T>> {
        let mut it = self.iter();
        std::iter::from_fn(move || {
            let v: Vec<&T> = it.by_ref().take(n).collect();
            (!v.is_empty()).then_some(v)
        })
    }

    /// Returns a parallel iterator over chunks of elements.
    pub fn par_chunks(&self, n: usize) -> impl IndexedParallelIterator<Item = Vec<&T>>
    where
        T: Send + Sync,
    {
        self.par_iter().chunks(n)
    }
}

/*
--------------------------------------------
* Indexing for tensor slices
--------------------------------------------
*/

impl<T> Index<&[usize]> for TensorSlice<'_, T> {
    type Output = T;

    fn index(&self, index: &[usize]) -> &Self::Output {
        let actual_index = self
            .start
            .iter()
            .zip(index.iter())
            .enumerate()
            .map(|(i, (a, b))| {
                assert!(a + b < self.end[i]);
                a + b
            })
            .collect::<Vec<usize>>();

        unsafe { self.orig.get_unchecked(actual_index.as_slice()) }
    }
}

/*
--------------------------------------------
* Conversion into Tensor
--------------------------------------------
*/

impl<T: Clone> TensorSlice<'_, T> {
    /// Clones the slice into a new tensor.
    pub fn clone_into_tensor(&self) -> Tensor<T> {
        if self.rank() == 0 {
            return Tensor {
                shape: Shape::new(vec![]),
                strides: Strides::from_shape(&Shape::new(vec![])),
                elements: self.orig.elements.clone(),
            };
        }

        let mut elements = Vec::with_capacity(self.shape().element_count());

        unsafe {
            let per_chunk_delta = self.orig.strides.0.get_unchecked(0);
            let slice_start_pos = self.orig.shape.address_unchecked(&self.start);
            let n_chunks = self
                .shape()
                .0
                .get_unchecked(0..self.rank() - 1)
                .iter()
                .product();

            for i in 0..n_chunks {
                let start = slice_start_pos + i * per_chunk_delta;
                let end = start + self.shape().0.get_unchecked(self.rank() - 1);
                elements.extend_from_slice(self.orig.elements.get_unchecked(start..end));
            }
        }

        Tensor {
            shape: self.shape(),
            strides: Strides::from_shape(&self.shape()),
            elements,
        }
    }
}
