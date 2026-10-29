/*
--------------------------------------------
* Mutable tensor slice definition
--------------------------------------------
*/

use crate::definitions::errors::TensorErrors;
use crate::definitions::shape::Shape;
use crate::definitions::strides::Strides;
use crate::definitions::tensor::Tensor;
use crate::definitions::tensor_slice::TensorSlice;
use rayon::iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

#[derive(Debug, Eq, PartialEq, Hash)]
pub struct TensorSliceMut<'a, T> {
    pub(crate) orig: &'a mut Tensor<T>,
    pub(crate) start: Vec<usize>,
    pub(crate) end: Vec<usize>,
}

impl<T> TensorSliceMut<'_, T> {
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

    /// Returns this slice as an immutable slice.
    pub fn as_immut(&self) -> TensorSlice<'_, T> {
        TensorSlice {
            orig: self.orig,
            start: self.start.clone(),
            end: self.end.clone(),
        }
    }

    /// Uses the immutable version of this slice in a closure, returning the result of the closure.
    pub fn use_as_immut<R>(&self, f: impl FnOnce(TensorSlice<'_, T>) -> R) -> R {
        f(self.as_immut())
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

        for (self_end, orig_index) in self.end.iter().zip(orig_index.iter()) {
            if *self_end <= *orig_index {
                return None;
            }
        }

        Some(unsafe { self.get_unchecked(indices) })
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

    /// Gets a mutable reference to the element at the specified indices in the tensor slice, returning `None` if the indices are out of bounds.
    pub fn get_mut(&mut self, indices: &[usize]) -> Option<&mut T> {
        if indices.len() != self.orig.rank() {
            return None;
        }

        let orig_index = indices
            .iter()
            .zip(self.start.iter())
            .map(|(x, y)| x + y)
            .collect::<Vec<usize>>();

        for (self_end, orig_index) in self.end.iter().zip(orig_index.iter()) {
            if *self_end <= *orig_index {
                return None;
            }
        }

        Some(unsafe { self.get_unchecked_mut(indices) })
    }

    /// Gets a mutable reference to the element at the specified indices in the tensor slice without bounds checking.
    pub(crate) unsafe fn get_unchecked_mut(&mut self, indices: &[usize]) -> &mut T {
        self.orig.get_unchecked_mut(
            &indices
                .iter()
                .zip(self.start.iter())
                .map(|(x, y)| x + y)
                .collect::<Vec<_>>(),
        )
    }

    /// Returns an iterator over the elements of the tensor slice.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &T> + DoubleEndedIterator {
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

    /// Returns a mutable iterator over the elements of the tensor slice.
    pub fn iter_mut(&mut self) -> impl ExactSizeIterator<Item = &mut T> + DoubleEndedIterator {
        struct SliceIterMut<'a, T> {
            base: *mut T,            // Pointer to the start of the original tensor's elements
            flat_index: usize,       // Current flat index in the slice
            len: usize,              // Total number of elements in the slice
            orig_shape: Shape,       // Shape of the original tensor
            slice_shape: Shape,      // Shape of the slice
            slice_start: Vec<usize>, // Start of the slice in the original tensor
            _marker: PhantomData<&'a mut T>,
        }

        impl<'a, T: 'a> Iterator for SliceIterMut<'a, T> {
            type Item = &'a mut T;

            fn next(&mut self) -> Option<Self::Item> {
                if self.flat_index >= self.len {
                    return None;
                }

                self.flat_index += 1;

                Some(unsafe {
                    &mut *self.base.add(
                        self.orig_shape.address_unchecked(
                            &self
                                .slice_shape
                                .tensor_index_unchecked(self.flat_index - 1)
                                .iter()
                                .zip(self.slice_start.iter())
                                .map(|(a, b)| a + b)
                                .collect::<Vec<_>>(),
                        ),
                    )
                })
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                (self.len - self.flat_index, Some(self.len - self.flat_index))
            }
        }
        impl<'a, T: 'a> ExactSizeIterator for SliceIterMut<'a, T> {}
        impl<'a, T: 'a> DoubleEndedIterator for SliceIterMut<'a, T> {
            fn next_back(&mut self) -> Option<Self::Item> {
                if self.flat_index >= self.len {
                    return None;
                }

                self.len -= 1;

                Some(unsafe {
                    &mut *self.base.add(
                        self.orig_shape.address_unchecked(
                            &self
                                .slice_shape
                                .tensor_index_unchecked(self.len)
                                .iter()
                                .zip(self.slice_start.iter())
                                .map(|(a, b)| a + b)
                                .collect::<Vec<_>>(),
                        ),
                    )
                })
            }
        }

        SliceIterMut {
            base: self.orig.elements_mut().as_mut_ptr(),
            flat_index: 0,
            len: self.shape().element_count(),
            orig_shape: self.orig.shape(),
            slice_shape: self.shape(),
            slice_start: self.start.clone(),
            _marker: Default::default(),
        }
    }

    /// Returns a parallel mutable iterator over the elements of the tensor slice.
    pub fn par_iter_mut(&mut self) -> impl IndexedParallelIterator<Item = &mut T>
    where
        T: Send + Sync,
    {
        let orig_shape = self.orig.shape.clone();
        let slice_shape = self.shape().clone();
        let base = self.orig.elements_mut().as_mut_ptr();
        let count = slice_shape.element_count();

        struct TensorSliceMutIter<'a, T> {
            slice_start: Vec<usize>, // Start of the entire slice
            start: usize,            // Flat index of start (within entire slice)
            end: usize,              // Flat exclusive end of subslice (within entire slice)
            base: *mut T,            // start for the entire original tensor
            slice_shape: Shape,      // Shape of the entire slice
            orig_shape: Shape,       // Shape of entire original tensor
            _marker: PhantomData<&'a T>,
        }

        unsafe impl<'a, T> Send for TensorSliceMutIter<'a, T> {}
        unsafe impl<'a, T> Sync for TensorSliceMutIter<'a, T> {}

        impl<'a, T: 'a> Iterator for TensorSliceMutIter<'a, T> {
            type Item = &'a mut T;

            fn next(&mut self) -> Option<Self::Item> {
                if self.start >= self.end {
                    return None;
                }

                let res = unsafe {
                    let orig_index = self
                        .slice_shape
                        .tensor_index_unchecked(self.start)
                        .iter()
                        .zip(self.slice_start.iter())
                        .map(|(a, b)| a + b)
                        .collect::<Vec<_>>();
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
        impl<'a, T: 'a> DoubleEndedIterator for TensorSliceMutIter<'a, T> {
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
                        .zip(self.slice_start.iter())
                        .map(|(a, b)| a + b)
                        .collect::<Vec<_>>();
                    self.base
                        .add(self.orig_shape.address_unchecked(&orig_index))
                        .as_mut()
                }
            }
        }
        impl<'a, T: 'a> ExactSizeIterator for TensorSliceMutIter<'a, T> {}

        impl<'a, T: 'a> Producer for TensorSliceMutIter<'a, T> {
            type Item = &'a mut T;
            type IntoIter = TensorSliceMutIter<'a, T>;

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

        impl<'a, T> ParallelIterator for TensorSliceMutIter<'a, T>
        where
            T: Send + Sync + 'a,
        {
            type Item = &'a mut T;

            fn drive_unindexed<C>(self, consumer: C) -> C::Result
            where
                C: UnindexedConsumer<Self::Item>,
            {
                bridge(self, consumer)
            }
        }

        impl<'a, T: Send + Sync + 'a> IndexedParallelIterator for TensorSliceMutIter<'a, T> {
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

    /// Returns an iterator over chunks of mutable elements.
    pub fn chunks_mut(&mut self, n: usize) -> impl Iterator<Item = Vec<&mut T>> {
        let mut it = self.iter_mut();
        std::iter::from_fn(move || {
            let v: Vec<&mut T> = it.by_ref().take(n).collect();
            (!v.is_empty()).then_some(v)
        })
    }

    /// Returns a parallel iterator over chunks of mutable elements.
    pub fn par_chunks_mut(&mut self, n: usize) -> impl IndexedParallelIterator<Item = Vec<&mut T>>
    where
        T: Send + Sync,
    {
        self.par_iter_mut().chunks(n)
    }

    /// Swaps the elements at the specified indices.
    pub fn swap(&mut self, index1: &[usize], index2: &[usize]) -> Result<(), TensorErrors> {
        let rank = self.rank();
        if index1.len() != rank {
            return Err(TensorErrors::IndicesInvalidForRank(index1.len(), rank));
        }
        if index2.len() != rank {
            return Err(TensorErrors::IndicesInvalidForRank(index2.len(), rank));
        }
        let slice_shape = self.shape();
        for axis in 0..rank {
            if index1[axis] >= slice_shape[axis] {
                return Err(TensorErrors::IndexOutOfBounds {
                    index: index1[axis],
                    axis,
                    length: slice_shape[axis],
                });
            }
            if index2[axis] >= slice_shape[axis] {
                return Err(TensorErrors::IndexOutOfBounds {
                    index: index2[axis],
                    axis,
                    length: slice_shape[axis],
                });
            }
        }

        self.orig.swap(
            &index1
                .iter()
                .zip(self.start.iter())
                .map(|(x, y)| x + y)
                .collect::<Vec<_>>(),
            &index2
                .iter()
                .zip(self.start.iter())
                .map(|(x, y)| x + y)
                .collect::<Vec<_>>(),
        )
    }
}

/*
--------------------------------------------
* Indexing for mutable tensor slices
--------------------------------------------
*/

impl<T> Index<&[usize]> for TensorSliceMut<'_, T> {
    type Output = T;

    fn index(&self, index: &[usize]) -> &Self::Output {
        assert_eq!(
            index.len(),
            self.rank(),
            "Index rank does not match tensor slice rank"
        );
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

impl<T> IndexMut<&[usize]> for TensorSliceMut<'_, T> {
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        assert_eq!(
            index.len(),
            self.rank(),
            "Index rank does not match tensor slice rank"
        );
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

        unsafe { self.orig.get_unchecked_mut(actual_index.as_slice()) }
    }
}

/*
--------------------------------------------
* Conversion into Tensor
--------------------------------------------
*/

impl<T: Clone> TensorSliceMut<'_, T> {
    /// Clones the slice into a new tensor.
    pub fn clone_into_tensor(&self) -> Tensor<T> {
        if self.rank() == 0 {
            return Tensor {
                shape: Shape::new(vec![]),
                strides: Strides::from_shape(&Shape::new(vec![])),
                elements: self.orig.elements.clone(),
            };
        }

        let slice_shape = self.shape();
        let mut elements = Vec::with_capacity(slice_shape.element_count());
        for i in 0..slice_shape.element_count() {
            let mut index = unsafe { slice_shape.tensor_index_unchecked(i) };
            for (index_i, start_i) in index.iter_mut().zip(self.start.iter()) {
                *index_i += start_i;
            }
            let address = unsafe { self.orig.shape.address_unchecked(&index) };
            elements.push(unsafe { self.orig.elements.get_unchecked(address) }.clone());
        }

        Tensor {
            shape: slice_shape.clone(),
            strides: Strides::from_shape(&slice_shape),
            elements,
        }
    }
}

#[cfg(test)]
mod tensor_slice_mut_tests {
    use rayon::iter::{IndexedParallelIterator, ParallelIterator};
    use tensor_math::definitions::errors::TensorErrors;
    use tensor_math::definitions::tensor::Tensor;
    use tensor_math::shape;

    #[test]
    fn slice_rank_and_shape() {
        let shape = shape![4, 5, 6];
        let mut tensor = Tensor::<i32>::from_shape(shape);
        let slice = tensor.slice_mut(&[1..4, 0..2, 3..6]).unwrap();

        assert_eq!(slice.rank(), 3);
        assert_eq!(slice.shape(), shape![3, 2, 3]);
    }

    #[test]
    #[should_panic]
    fn invalid_index() {
        let shape = shape![2, 3, 4];
        let mut tensor = Tensor::<i32>::from_shape(shape);
        let slice = tensor.slice_mut(&[0..1, 0..1, 0..1]).unwrap();

        slice[&[1, 0, 0]];
    }

    #[test]
    fn index_mut() {
        let shape = shape![2, 3, 4];
        let mut tensor = Tensor::<i32>::from_shape(shape);
        let mut slice = tensor.slice_mut(&[0..1, 0..1, 0..1]).unwrap();

        assert_eq!(slice[&[0, 0, 0]], 0);

        slice[&[0, 0, 0]] = 67;

        assert_eq!(slice[&[0, 0, 0]], 67);
    }

    #[test]
    fn get_is_safe_index() {
        let shape = shape![2, 3, 4];
        let mut tensor = Tensor::<i32>::from_shape(shape);
        let slice = tensor.slice_mut(&[0..1, 0..1, 0..1]).unwrap();

        assert_eq!(slice.get(&[0, 0]), None);
        assert_eq!(slice.get(&[0, 0, 0]), Some(&0));
    }

    #[test]
    fn get_mut_is_safe_index_mut() {
        let shape = shape![2, 3, 4];
        let mut tensor = Tensor::<i32>::from_shape(shape);
        let mut slice = tensor.slice_mut(&[0..1, 0..1, 0..1]).unwrap();

        assert_eq!(slice.get_mut(&[0, 0]), None);
        assert_eq!(slice.get_mut(&[0, 0, 0]).copied(), Some(0));
        slice.get_mut(&[0, 0, 0]).map(|x| *x = 42);

        assert_eq!(tensor, Tensor::new(shape![2, 3, 4], vec![
            42, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,

            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
        ]).unwrap());
    }

    #[test]
    fn convert_into_tensor() {
        let shape = shape![2, 3, 4];
        let mut tensor = Tensor::<i32>::from_shape(shape);
        let slice = tensor.slice_mut(&[0..1, 0..1, 0..1]).unwrap();
        let ans = Tensor::new(shape![1, 1, 1], vec![0]).unwrap();

        assert_eq!(slice.clone_into_tensor(), ans);

        let slice = tensor.slice_mut(&[0..1, 0..1, 0..1]).unwrap();

        assert_eq!(slice.clone_into_tensor(), ans);
    }

    #[test]
    fn slicing_on_scalar_tensor() {
        let mut tensor = Tensor::<i32>::new(shape![], vec![42]).unwrap();
        let mut slice = tensor.slice_mut(&[]).unwrap();

        assert_eq!(slice.get(&[]), Some(&42));
        slice[&[]] = 10;
        assert_eq!(
            slice.clone_into_tensor(),
            Tensor::new(shape![], vec![10]).unwrap()
        );
    }

    #[test]
    fn slicing_on_empty_tensor() {
        let mut tensor = Tensor::<i32>::new(shape![0, 3], vec![]).unwrap();
        let slice = tensor.slice_mut(&[0..0, 1..2]).unwrap();

        assert_eq!(slice.start(), vec![0, 1]);
        assert_eq!(slice.end(), vec![0, 2]);
        assert_eq!(slice.clone_into_tensor().shape(), shape![0, 1]);

        let mut tensor2 = Tensor::<i32>::new(shape![3, 0], vec![]).unwrap();
        let slice2 = tensor2.slice_mut(&[1..2, 0..0]).unwrap();

        assert_eq!(slice2.start(), vec![1, 0]);
        assert_eq!(slice2.end(), vec![2, 0]);
        assert_eq!(slice2.clone_into_tensor().shape(), shape![1, 0]);
    }

    #[test]
    fn slice_shape() {
        let shape = shape![10, 20, 30];
        let mut tensor = Tensor::<i32>::from_shape(shape);

        // Regular slice
        let slice = tensor.slice_mut(&[2..5, 0..10, 15..30]).unwrap();
        assert_eq!(slice.shape(), shape![3, 10, 15]);

        // Zero-length dimension
        let slice = tensor.slice_mut(&[0..10, 5..5, 0..30]).unwrap();
        assert_eq!(slice.shape(), shape![10, 0, 30]);
    }

    #[test]
    fn for_each_mut_basic() {
        let shape = shape![2, 3, 4];
        let mut tensor = Tensor::<i32>::from_shape(shape);
        let mut slice = tensor.slice_mut(&[0..2, 1..3, 1..4]).unwrap();
        // slice shape: [2, 2, 3]

        slice.iter_mut().for_each(|x| *x = 1);

        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    let val = *tensor.get(&[i, j, k]).unwrap();
                    if j >= 1 && j < 3 && k >= 1 && k < 4 {
                        assert_eq!(val, 1, "Expected 1 at [{}, {}, {}]", i, j, k);
                    } else {
                        assert_eq!(val, 0, "Expected 0 at [{}, {}, {}]", i, j, k);
                    }
                }
            }
        }
    }

    #[test]
    fn for_each_mut_scalar() {
        let mut tensor = Tensor::<i32>::new(shape![], vec![42]).unwrap();
        let mut slice = tensor.slice_mut(&[]).unwrap();

        slice.iter_mut().for_each(|x| *x += 1);
        assert_eq!(*tensor.get(&[]).unwrap(), 43);
    }

    #[test]
    fn for_each_mut_empty() {
        let mut tensor = Tensor::<i32>::new(shape![3, 0], vec![]).unwrap();
        let mut slice = tensor.slice_mut(&[1..2, 0..0]).unwrap();

        let mut count = 0;
        slice.iter_mut().for_each(|_| count += 1);
        assert_eq!(count, 0);
    }
    #[test]
    fn enumerated_for_each_mut_basic() {
        let shape = shape![2, 3, 4];
        let mut tensor = Tensor::<i32>::from_shape(shape);
        let mut slice = tensor.slice_mut(&[0..2, 1..3, 1..4]).unwrap();
        // slice shape: [2, 2, 3]

        slice.enumerated_iter_mut().for_each(|(idx, x)| {
            // idx should be relative to the slice: [0..2, 0..2, 0..3]
            *x = (idx[0] * 100 + idx[1] * 10 + idx[2] + 1) as i32;
        });

        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    let val = *tensor.get(&[i, j, k]).unwrap();
                    if j >= 1 && j < 3 && k >= 1 && k < 4 {
                        let expected = (i * 100 + (j - 1) * 10 + (k - 1) + 1) as i32;
                        assert_eq!(
                            val, expected,
                            "Mismatch at tensor index [{}, {}, {}]",
                            i, j, k
                        );
                    } else {
                        assert_eq!(
                            val, 0,
                            "Expected zero at tensor index [{}, {}, {}]",
                            i, j, k
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn enumerated_for_each_mut_scalar() {
        let mut tensor = Tensor::<i32>::new(shape![], vec![42]).unwrap();
        let mut slice = tensor.slice_mut(&[]).unwrap();

        slice.enumerated_iter_mut().for_each(|(idx, x)| {
            assert_eq!(idx.len(), 0);
            *x += 1;
        });
        assert_eq!(*tensor.get(&[]).unwrap(), 43);
    }

    #[test]
    fn enumerated_for_each_mut_empty() {
        let mut tensor = Tensor::<i32>::new(shape![3, 0], vec![]).unwrap();
        let mut slice = tensor.slice_mut(&[1..2, 0..0]).unwrap();

        let mut count = 0;
        slice.enumerated_iter_mut().for_each(|(_, _)| count += 1);
        assert_eq!(count, 0);
    }

    #[test]
    fn immutable_views_and_checked_mutable_access_cover_scalar_and_empty_slices() {
        let mut tensor = Tensor::new(shape![2, 3], (0..6).collect()).unwrap();
        {
            let mut slice = tensor.slice_mut(&[1..2, 1..3]).unwrap();
            assert_eq!(slice.start(), &[1, 1]);
            assert_eq!(slice.end(), &[2, 3]);
            assert_eq!(slice.shape(), shape![1, 2]);
            assert_eq!(slice.rank(), 2);
            assert_eq!(
                slice.as_immut().iter().copied().collect::<Vec<_>>(),
                vec![4, 5]
            );
            assert_eq!(
                slice.use_as_immut(|view| view.get(&[0, 1]).copied()),
                Some(5)
            );
            assert_eq!(slice.get(&[]), None);
            assert_eq!(slice.get(&[0]), None);
            assert_eq!(slice.get(&[0, 0]), Some(&4));
            assert_eq!(slice.get(&[1, 0]), None);
            assert_eq!(slice.get(&[0, 2]), None);
            *slice.get_mut(&[0, 0]).unwrap() = 40;
            assert!(slice.get_mut(&[1, 0]).is_none());
            assert!(slice.get_mut(&[0, 2]).is_none());
        }
        assert_eq!(tensor.get(&[1, 1]), Some(&40));

        let mut scalar = Tensor::new(shape![], vec![7]).unwrap();
        let scalar_slice = scalar.slice_mut(&[]).unwrap();
        assert_eq!(scalar_slice.rank(), 0);
        assert_eq!(scalar_slice.shape(), shape![]);
        assert_eq!(scalar_slice.as_immut().get(&[]), Some(&7));
        assert_eq!(scalar_slice.get(&[]), Some(&7));
        drop(scalar_slice);

        let mut empty = Tensor::<i32>::new(shape![0, 3], vec![]).unwrap();
        let mut empty_slice = empty.slice_mut(&[0..0, 1..3]).unwrap();
        assert_eq!(empty_slice.shape(), shape![0, 2]);
        assert_eq!(empty_slice.get(&[0, 0]), None);
        assert!(empty_slice.get_mut(&[0, 0]).is_none());
    }

    #[test]
    fn iterators_visit_only_slice_elements_and_mutate_them() {
        let mut tensor = Tensor::new(shape![2, 3, 4], (0..24).collect()).unwrap();
        let slice = tensor.slice_mut(&[0..2, 1..3, 1..3]).unwrap();
        let expected = vec![5, 6, 9, 10, 17, 18, 21, 22];
        assert_eq!(slice.iter().copied().collect::<Vec<_>>(), expected);
        assert_eq!(slice.par_iter().copied().collect::<Vec<_>>(), expected);
        assert_eq!(slice.iter().next_back(), Some(&22));
        assert_eq!(slice.par_iter().len(), 8);
        drop(slice);

        let mut slice = tensor.slice_mut(&[0..2, 1..3, 1..3]).unwrap();
        slice.iter_mut().for_each(|value| *value += 100);
        drop(slice);
        assert_eq!(tensor.get(&[0, 1, 1]), Some(&105));
        assert_eq!(tensor.get(&[0, 1, 3]), Some(&7));

        let mut slice = tensor.slice_mut(&[0..2, 1..3, 1..3]).unwrap();
        slice.par_iter_mut().for_each(|value| *value += 1000);
        drop(slice);
        assert_eq!(tensor.get(&[0, 1, 1]), Some(&1105));
        assert_eq!(tensor.get(&[1, 2, 2]), Some(&1122));
        assert_eq!(tensor.get(&[0, 0, 0]), Some(&0));
    }

    #[test]
    fn parallel_mutable_iterator_visits_scalar_once_and_empty_slice_never() {
        let mut scalar = Tensor::new(shape![], vec![4]).unwrap();
        scalar
            .slice_mut(&[])
            .unwrap()
            .par_iter_mut()
            .for_each(|value| *value += 1);
        assert_eq!(scalar.get(&[]), Some(&5));

        for (tensor_shape, ranges) in [
            (shape![0, 2], vec![0..0, 0..2]),
            (shape![2, 0], vec![0..2, 0..0]),
            (shape![0, 0], vec![0..0, 0..0]),
        ] {
            let mut tensor = Tensor::<i32>::new(tensor_shape, vec![]).unwrap();
            let mut slice = tensor.slice_mut(&ranges).unwrap();
            assert_eq!(slice.par_iter_mut().count(), 0);
            assert_eq!(slice.iter_mut().count(), 0);
        }
    }

    #[test]
    fn chunks_cover_scalar_and_regular_mutable_slices() {
        let mut tensor = Tensor::new(shape![2, 3], (0..6).collect()).unwrap();
        let slice = tensor.slice_mut(&[0..2, 1..3]).unwrap();
        let expected = vec![vec![1, 2, 4], vec![5]];
        assert_eq!(
            slice
                .chunks(3)
                .map(|chunk| chunk.into_iter().copied().collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(
            slice
                .par_chunks(3)
                .map(|chunk| chunk.into_iter().copied().collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            expected
        );
        drop(slice);

        let mut slice = tensor.slice_mut(&[0..2, 1..3]).unwrap();
        slice
            .chunks_mut(2)
            .for_each(|chunk| chunk.into_iter().for_each(|value| *value += 10));
        drop(slice);
        assert_eq!(tensor.elements(), &[0, 11, 12, 3, 14, 15]);

        let mut slice = tensor.slice_mut(&[0..2, 1..3]).unwrap();
        slice
            .par_chunks_mut(2)
            .for_each(|chunk| chunk.into_iter().for_each(|value| *value *= 2));
        drop(slice);
        assert_eq!(tensor.elements(), &[0, 22, 24, 3, 28, 30]);

        let mut scalar = Tensor::new(shape![], vec![8]).unwrap();
        assert_eq!(scalar.slice_mut(&[]).unwrap().chunks(1).count(), 1);
        assert_eq!(scalar.slice_mut(&[]).unwrap().par_chunks(1).count(), 1);
    }

    #[test]
    fn chunks_mut_are_empty_for_all_empty_dimension_shapes() {
        for (tensor_shape, ranges) in [
            (shape![0, 2], vec![0..0, 0..2]),
            (shape![2, 0], vec![0..2, 0..0]),
            (shape![0, 0], vec![0..0, 0..0]),
        ] {
            let mut tensor = Tensor::<i32>::new(tensor_shape, vec![]).unwrap();
            let mut slice = tensor.slice_mut(&ranges).unwrap();
            assert_eq!(slice.chunks(2).count(), 0);
            assert_eq!(slice.par_chunks(2).count(), 0);
            assert_eq!(slice.chunks_mut(2).count(), 0);
            assert_eq!(slice.par_chunks_mut(2).count(), 0);
        }
    }

    #[test]
    fn swap_checks_slice_relative_indices_and_supports_rank_zero() {
        let mut tensor = Tensor::new(shape![3, 3], (0..9).collect()).unwrap();
        {
            let mut slice = tensor.slice_mut(&[1..3, 1..3]).unwrap();
            slice.swap(&[0, 0], &[1, 1]).unwrap();
            assert_eq!(slice[&[0, 0]], 8);
            assert_eq!(slice[&[1, 1]], 4);
            assert_eq!(
                slice.swap(&[0], &[0, 0]),
                Err(TensorErrors::IndicesInvalidForRank(1, 2))
            );
            assert_eq!(
                slice.swap(&[0, 0], &[0, 2]),
                Err(TensorErrors::IndexOutOfBounds {
                    axis: 1,
                    length: 2,
                    index: 2
                })
            );
            assert_eq!(
                slice.swap(&[2, 0], &[0, 0]),
                Err(TensorErrors::IndexOutOfBounds {
                    axis: 0,
                    length: 2,
                    index: 2
                })
            );
            assert_eq!(
                slice.swap(&[0, 0], &[0]),
                Err(TensorErrors::IndicesInvalidForRank(1, 2))
            );
        }
        assert_eq!(tensor.elements(), &[0, 1, 2, 3, 8, 5, 6, 7, 4]);

        let mut scalar = Tensor::new(shape![], vec![42]).unwrap();
        scalar.slice_mut(&[]).unwrap().swap(&[], &[]).unwrap();
        assert_eq!(scalar.get(&[]), Some(&42));
    }

    #[test]
    fn mutable_indexing_supports_scalar_and_checks_exact_rank() {
        let mut tensor = Tensor::new(shape![2, 2], vec![0, 1, 2, 3]).unwrap();
        {
            let mut slice = tensor.slice_mut(&[1..2, 0..2]).unwrap();
            slice[&[0, 1]] = 30;
            assert_eq!(slice[&[0, 1]], 30);
        }
        assert_eq!(tensor.get(&[1, 1]), Some(&30));

        let mut scalar = Tensor::new(shape![], vec![7]).unwrap();
        scalar.slice_mut(&[]).unwrap()[&[]] = 8;
        assert_eq!(scalar.get(&[]), Some(&8));
    }

    #[test]
    #[should_panic]
    fn immutable_indexing_panics_for_short_index_rank() {
        let mut tensor = Tensor::new(shape![2, 2], vec![0, 1, 2, 3]).unwrap();
        let slice = tensor.slice_mut(&[0..2, 0..2]).unwrap();
        let _ = slice[&[0]];
    }

    #[test]
    #[should_panic]
    fn mutable_indexing_panics_for_short_index_rank() {
        let mut tensor = Tensor::new(shape![2, 2], vec![0, 1, 2, 3]).unwrap();
        let mut slice = tensor.slice_mut(&[0..2, 0..2]).unwrap();
        slice[&[0]] = 1;
    }

    #[test]
    #[should_panic]
    fn mutable_indexing_panics_for_long_index_rank() {
        let mut tensor = Tensor::new(shape![2, 2], vec![0, 1, 2, 3]).unwrap();
        let mut slice = tensor.slice_mut(&[0..2, 0..2]).unwrap();
        slice[&[0, 0, 0]] = 1;
    }

    #[test]
    fn clone_into_tensor_handles_scalar_and_each_empty_dimension_position() {
        let mut scalar = Tensor::new(shape![], vec![42]).unwrap();
        assert_eq!(
            scalar.slice_mut(&[]).unwrap().clone_into_tensor(),
            Tensor::new(shape![], vec![42]).unwrap()
        );

        let mut tensor = Tensor::new(shape![2, 3, 2], (0..12).collect()).unwrap();
        assert_eq!(
            tensor
                .slice_mut(&[0..2, 1..3, 0..2])
                .unwrap()
                .clone_into_tensor(),
            Tensor::new(shape![2, 2, 2], vec![2, 3, 4, 5, 8, 9, 10, 11]).unwrap()
        );
        let mut tensor = Tensor::new(shape![2, 3, 2], (0..12).collect()).unwrap();
        assert_eq!(
            tensor
                .slice_mut(&[0..2, 1..1, 0..2])
                .unwrap()
                .clone_into_tensor(),
            Tensor::new(shape![2, 0, 2], vec![]).unwrap()
        );
    }

    #[test]
    fn reversed_iterators_work_as_expected() {
        let mut tensor = Tensor::new(shape![2, 3], (0..6).collect()).unwrap();
        let slice = tensor.slice_mut(&[0..2, 1..3]).unwrap();
        let expected = vec![5, 4, 2, 1];
        assert_eq!(slice.iter().rev().copied().collect::<Vec<_>>(), expected);
        assert_eq!(
            slice.par_iter().rev().copied().collect::<Vec<_>>(),
            expected
        );

        let mut slice = tensor.slice_mut(&[0..2, 1..3]).unwrap();
        slice.iter_mut().rev().for_each(|value| *value += 10);
        drop(slice);
        assert_eq!(tensor.elements(), &[0, 11, 12, 3, 14, 15]);

        let mut slice = tensor.slice_mut(&[0..2, 1..3]).unwrap();
        slice.par_iter_mut().rev().for_each(|value| *value *= 2);
        drop(slice);
        assert_eq!(tensor.elements(), &[0, 22, 24, 3, 28, 30]);
    }
}
