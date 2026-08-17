use crate::definitions::chunk::{Chunk, ChunkMut};
use crate::definitions::errors::TensorErrors;
use crate::definitions::errors::TensorErrors::SliceIncompatibleShape;
use crate::definitions::shape::Shape;
use crate::definitions::tensor::Tensor;
use crate::definitions::traits::{IntoTensor, TensorLike, TensorLikeMut};
use rayon::iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

#[derive(Debug, Eq, PartialEq)]
pub struct TensorSlice<'a, T> {
    pub(crate) orig: &'a Tensor<T>,
    pub(crate) start: Vec<usize>,
    pub(crate) end: Vec<usize>,
}

impl<T> TensorSlice<'_, T> {
    /// Returns the rank of the tensor slice.
    pub fn rank(&self) -> usize {
        self.end.len()
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

    /// Returns the start position of the tensor slice.
    pub fn start(&self) -> &[usize] {
        &self.start
    }

    /// Returns the end position of the tensor slice.
    pub fn end(&self) -> &[usize] {
        &self.end
    }

    /// Applies the given function to each element of the slice.
    pub fn for_each(&self, mut closure: impl FnMut(&T)) {
        let shape = self.shape();
        for i in 0..shape.element_count() {
            closure(&self[&shape.tensor_index(i).unwrap()]);
        }
    }

    /// Applies the given function to each element of the slice with the index in the slice.
    pub fn enumerated_for_each(&self, mut closure: impl FnMut(&[usize], &T)) {
        let shape = self.shape();
        for i in 0..shape.element_count() {
            let index = shape.tensor_index(i).unwrap();
            closure(&index, &self[&index]);
        }
    }
}

impl<'a, T: Clone> TensorSlice<'a, T> {
    /// Gets the value at the specified index if it exits, otherwise returns None
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

        self.orig.get(orig_index.as_slice())
    }
}

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

        &self.orig[actual_index.as_slice()]
    }
}

impl<T: Clone> IntoTensor<T> for TensorSlice<'_, T> {
    fn into_tensor(self) -> Tensor<T> {
        let mut elements = Vec::with_capacity(self.shape().element_count());
        for i in 0..self.shape().element_count() {
            elements.push(self[&self.shape().tensor_index(i).unwrap()].clone());
        }

        Tensor::new(&self.shape(), elements).unwrap()
    }
}

impl<'a, T: Clone> TensorLike<T> for TensorSlice<'a, T> {
    fn shape(&self) -> Shape {
        TensorSlice::shape(self)
    }

    fn rank(&self) -> usize {
        TensorSlice::rank(self)
    }

    fn elements(&self) -> &[T] {
        self.orig.elements()
    }

    fn get(&self, indices: &[usize]) -> Option<&T> {
        TensorSlice::get(self, indices)
    }

    unsafe fn get_unchecked(&self, indices: &[usize]) -> &T {
        self.orig.get_unchecked(
            &indices
                .iter()
                .zip(self.start.iter())
                .map(|(x, y)| x + y)
                .collect::<Vec<_>>(),
        )
    }

    fn iter<'b>(&'b self) -> impl Iterator<Item = &'b T>
    where
        T: 'b,
    {
        struct SliceIter<'b, T> {
            slice: &'b TensorSlice<'b, T>,
            flat_index: usize,
        }

        impl<'b, T> Iterator for SliceIter<'b, T> {
            type Item = &'b T;

            fn next(&mut self) -> Option<Self::Item> {
                let shape = self.slice.shape();
                let res = match shape.tensor_index(self.flat_index) {
                    Ok(i) => Some(&self.slice[&i]),
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
            .map(move |i| &self[&shape.tensor_index(i).unwrap()])
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

#[derive(Debug, Eq, PartialEq)]
pub struct TensorSliceMut<'a, T> {
    pub(crate) orig: &'a mut Tensor<T>,
    pub(crate) start: Vec<usize>,
    pub(crate) end: Vec<usize>,
}

impl<T> TensorSliceMut<'_, T> {
    /// Returns the rank of the tensor slice.
    pub fn rank(&self) -> usize {
        self.end.len()
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

    /// Returns the start position of the tensor slice.
    pub fn start(&self) -> &[usize] {
        &self.start
    }

    /// Returns the end position of the tensor slice.
    pub fn end(&self) -> &[usize] {
        &self.end
    }

    /// Applies the given function to each element of the slice.
    pub fn for_each_mut(&mut self, mut closure: impl FnMut(&mut T)) {
        let shape = self.shape();
        for i in 0..shape.element_count() {
            closure(&mut self[&shape.tensor_index(i).unwrap()]);
        }
    }

    /// Applies the given function to each element of the slice with the index in the slice.
    pub fn enumerated_for_each_mut(&mut self, mut closure: impl FnMut(&[usize], &mut T)) {
        let shape = self.shape();
        for i in 0..shape.element_count() {
            let index = shape.tensor_index(i).unwrap();
            closure(&index, &mut self[&index]);
        }
    }
}

impl<'a, T: Clone> TensorSliceMut<'a, T> {
    /// Sets all the values in the mutable slice to the given values.
    /// This fails if the shape of the values does not match the slice shape.
    pub fn set_all(&mut self, values: &Tensor<T>) -> Result<(), TensorErrors> {
        let slice_shape = Shape::new(
            self.end
                .iter()
                .zip(self.start.iter())
                .map(|(e, s)| e - s)
                .collect(),
        );

        if slice_shape != values.shape {
            return Err(SliceIncompatibleShape {
                slice_shape: self
                    .start
                    .iter()
                    .zip(self.end.iter())
                    .map(|(&x, &y)| y - x)
                    .collect::<Shape>(),
                tensor_shape: values.shape.clone(),
            });
        }

        for (index, value) in values.enumerated_iter() {
            self[index.as_slice()] = value;
        }

        Ok(())
    }

    /// Gets the value at the specified index if it exits, otherwise returns None
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

        self.orig.get(orig_index.as_slice())
    }
}

impl<T> Index<&[usize]> for TensorSliceMut<'_, T> {
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

        &self.orig[actual_index.as_slice()]
    }
}

impl<T> IndexMut<&[usize]> for TensorSliceMut<'_, T> {
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
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

        &mut self.orig[actual_index.as_slice()]
    }
}

impl<T: Clone> IntoTensor<T> for TensorSliceMut<'_, T> {
    fn into_tensor(self) -> Tensor<T> {
        let mut elements = Vec::with_capacity(self.shape().element_count());
        for i in 0..self.shape().element_count() {
            elements.push(self[&self.shape().tensor_index(i).unwrap()].clone());
        }

        Tensor::new(&self.shape(), elements).unwrap()
    }
}

impl<'a, T: Clone> TensorLike<T> for TensorSliceMut<'a, T> {
    fn shape(&self) -> Shape {
        TensorSliceMut::shape(self)
    }

    fn rank(&self) -> usize {
        TensorSliceMut::rank(self)
    }

    fn elements(&self) -> &[T] {
        self.orig.elements()
    }

    fn get(&self, indices: &[usize]) -> Option<&T> {
        TensorSliceMut::get(self, indices)
    }

    unsafe fn get_unchecked(&self, indices: &[usize]) -> &T {
        self.orig.get_unchecked(
            &indices
                .iter()
                .zip(self.start.iter())
                .map(|(x, y)| x + y)
                .collect::<Vec<_>>(),
        )
    }

    fn iter<'b>(&'b self) -> impl Iterator<Item = &'b T>
    where
        T: 'b,
    {
        struct SliceIter<'b, T> {
            slice: &'b TensorSliceMut<'b, T>,
            flat_index: usize,
        }

        impl<'b, T> Iterator for SliceIter<'b, T> {
            type Item = &'b T;

            fn next(&mut self) -> Option<Self::Item> {
                let shape = self.slice.shape();
                let res = match shape.tensor_index(self.flat_index) {
                    Ok(i) => Some(&self.slice[&i]),
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
            .map(move |i| &self[&shape.tensor_index(i).unwrap()])
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

impl<'a, T: Clone> TensorLikeMut<T> for TensorSliceMut<'a, T> {
    fn elements_mut(&mut self) -> &mut [T] {
        self.orig.elements_mut()
    }

    fn get_mut(&mut self, indices: &[usize]) -> Option<&mut T> {
        self.orig.get_mut(indices)
    }

    unsafe fn get_unchecked_mut(&mut self, indices: &[usize]) -> &mut T {
        self.orig.get_unchecked_mut(
            &indices
                .iter()
                .zip(self.start.iter())
                .map(|(x, y)| x + y)
                .collect::<Vec<_>>(),
        )
    }

    fn iter_mut<'b>(&'b mut self) -> impl Iterator<Item = &'b mut T>
    where
        T: 'b,
    {
        struct SliceIterMut<'b, T> {
            base: *mut T,
            flat_index: usize,
            shape: &'b Shape,
        }

        impl<'b, T: 'b> Iterator for SliceIterMut<'b, T> {
            type Item = &'b mut T;

            fn next(&mut self) -> Option<Self::Item> {
                self.shape.tensor_index(self.flat_index).ok()?;
                self.flat_index += 1;
                let offset = self.flat_index - 1;
                Some(unsafe { &mut *self.base.add(offset) })
            }
        }

        SliceIterMut {
            base: self.orig.elements_mut().as_mut_ptr(),
            flat_index: 0,
            shape: &self.orig.shape,
        }
    }

    fn par_iter_mut<'b>(&'b mut self) -> impl IndexedParallelIterator<Item = &'b mut T>
    where
        T: 'b + Send + Sync,
    {
        let orig_shape = self.orig.shape.clone();
        let slice_shape = self.shape().clone();
        let base = self.orig.elements_mut().as_mut_ptr();
        let count = orig_shape.element_count();

        struct TensorSliceMutIter<'b, T> {
            slice_start: Vec<usize>, // Start of the entire slice
            start: usize,            // Flat index of start (within entire slice)
            end: usize,              // Flat exclusive end of subslice (within entire slice)
            base: *mut T,            // start for the entire original tensor
            slice_shape: Shape,      // Shape of the entire slice
            orig_shape: Shape,       // Shape of entire original tensor
            _marker: PhantomData<&'b T>,
        }

        unsafe impl<'b, T> Send for TensorSliceMutIter<'b, T> {}
        unsafe impl<'b, T> Sync for TensorSliceMutIter<'b, T> {}

        struct ParIter<'b, T> {
            c: &'b T,
        }

        impl<'b, T: 'b> Iterator for TensorSliceMutIter<'b, T> {
            type Item = &'b mut T;

            fn next(&mut self) -> Option<Self::Item> {
                if self.start >= self.end {
                    return None;
                }

                let orig_index = self
                    .slice_shape
                    .tensor_index(self.start)
                    .ok()?
                    .iter()
                    .zip(self.slice_start.iter())
                    .map(|(a, b)| a + b)
                    .collect::<Vec<_>>();
                let res = unsafe {
                    self.base
                        .add(self.orig_shape.address(orig_index).ok()?)
                        .as_mut()
                };

                self.start += 1;

                res
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                (self.end - self.start, Some(self.end - self.start))
            }
        }
        impl<'b, T: 'b> DoubleEndedIterator for TensorSliceMutIter<'b, T> {
            fn next_back(&mut self) -> Option<Self::Item> {
                if self.end <= self.start {
                    return None;
                }

                self.end -= 1;

                let orig_index = self
                    .slice_shape
                    .tensor_index(self.end)
                    .ok()?
                    .iter()
                    .zip(self.slice_start.iter())
                    .map(|(a, b)| a + b)
                    .collect::<Vec<_>>();

                unsafe {
                    self.base
                        .add(self.orig_shape.address(orig_index).ok()?)
                        .as_mut()
                }
            }
        }
        impl<'b, T: 'b> ExactSizeIterator for TensorSliceMutIter<'b, T> {}

        impl<'b, T: 'b> Producer for TensorSliceMutIter<'b, T> {
            type Item = &'b mut T;
            type IntoIter = TensorSliceMutIter<'b, T>;

            fn into_iter(self) -> Self::IntoIter {
                self
            }

            fn split_at(self, index: usize) -> (Self, Self) {
                (
                    TensorSliceMutIter {
                        slice_start: self.slice_start.clone(),
                        start: self.start,
                        end: self.start + index,
                        base: self.base,
                        slice_shape: self.slice_shape.clone(),
                        orig_shape: self.orig_shape.clone(),
                        _marker: Default::default(),
                    },
                    TensorSliceMutIter {
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

        impl<'b, T> ParallelIterator for TensorSliceMutIter<'b, T>
        where
            T: Send + Sync + 'b,
        {
            type Item = &'b mut T;

            fn drive_unindexed<C>(self, consumer: C) -> C::Result
            where
                C: UnindexedConsumer<Self::Item>,
            {
                bridge(self, consumer)
            }
        }

        impl<'b, T: Send + Sync + 'b> IndexedParallelIterator for TensorSliceMutIter<'b, T> {
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

        TensorSliceMutIter {
            slice_start: self.start.clone(),
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
