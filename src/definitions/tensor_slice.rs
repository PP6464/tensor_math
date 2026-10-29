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

#[derive(Debug, Eq, PartialEq, Hash, Clone)]
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
mod tensor_slice_tests {
    use rayon::iter::{IndexedParallelIterator, ParallelIterator};
    use tensor_math::definitions::tensor::Tensor;
    use tensor_math::shape;

    #[test]
    fn slice_bounds_shape_rank_and_scalar_shape() {
        let tensor = Tensor::new(shape![3, 4], (0..12).collect()).unwrap();
        let slice = tensor.slice(&[1..3, 1..4]).unwrap();

        assert_eq!(slice.start(), &[1, 1]);
        assert_eq!(slice.end(), &[3, 4]);
        assert_eq!(slice.shape(), shape![2, 3]);
        assert_eq!(slice.rank(), 2);

        let scalar = Tensor::new(shape![], vec![5]).unwrap();
        let scalar_slice = scalar.slice(&[]).unwrap();
        assert_eq!(scalar_slice.start(), &[]);
        assert_eq!(scalar_slice.end(), &[]);
        assert_eq!(scalar_slice.shape(), shape![]);
        assert_eq!(scalar_slice.rank(), 0);
    }

    #[test]
    fn get_checks_rank_and_slice_relative_bounds() {
        let tensor = Tensor::new(shape![3, 4], (0..12).collect()).unwrap();
        let slice = tensor.slice(&[1..3, 1..4]).unwrap();

        assert_eq!(slice.get(&[0, 0]), Some(&5));
        assert_eq!(slice.get(&[1, 2]), Some(&11));
        assert_eq!(slice.get(&[]), None);
        assert_eq!(slice.get(&[0]), None);
        assert_eq!(slice.get(&[2, 0]), None);
        assert_eq!(slice.get(&[0, 3]), None);

        let scalar = Tensor::new(shape![], vec![42]).unwrap();
        assert_eq!(scalar.slice(&[]).unwrap().get(&[]), Some(&42));

        let empty = Tensor::<i32>::new(shape![2, 0, 3], vec![]).unwrap();
        assert_eq!(
            empty.slice(&[0..2, 0..0, 0..3]).unwrap().get(&[0, 0, 0]),
            None
        );
    }

    #[test]
    fn element_iterators_are_row_major_and_support_rank_zero() {
        let tensor = Tensor::new(shape![2, 3, 4], (0..24).collect()).unwrap();
        let slice = tensor.slice(&[0..2, 1..3, 1..4]).unwrap();
        let expected = vec![5, 6, 7, 9, 10, 11, 17, 18, 19, 21, 22, 23];

        assert_eq!(slice.iter().copied().collect::<Vec<_>>(), expected);
        assert_eq!(slice.par_iter().copied().collect::<Vec<_>>(), expected);
        assert_eq!(slice.iter().next_back(), Some(&23));
        assert_eq!(slice.par_iter().len(), 12);

        let scalar = Tensor::new(shape![], vec![42]).unwrap();
        let scalar_slice = scalar.slice(&[]).unwrap();
        assert_eq!(scalar_slice.iter().copied().collect::<Vec<_>>(), vec![42]);
        assert_eq!(
            scalar_slice.par_iter().copied().collect::<Vec<_>>(),
            vec![42]
        );
    }

    #[test]
    fn empty_dimension_slices_have_empty_element_iterators() {
        for (tensor_shape, ranges, slice_shape) in [
            (shape![0, 2], vec![0..0, 0..2], shape![0, 2]),
            (shape![2, 0], vec![0..2, 0..0], shape![2, 0]),
            (shape![0, 0], vec![0..0, 0..0], shape![0, 0]),
        ] {
            let tensor = Tensor::<i32>::new(tensor_shape, vec![]).unwrap();
            let slice = tensor.slice(&ranges).unwrap();
            assert_eq!(slice.shape(), slice_shape);
            assert_eq!(slice.iter().count(), 0);
            assert_eq!(slice.par_iter().count(), 0);
        }
    }

    #[test]
    fn chunks_cover_elements_and_empty_slices() {
        let tensor = Tensor::new(shape![2, 3], (0..6).collect()).unwrap();
        let slice = tensor.slice(&[0..2, 1..3]).unwrap();
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
        assert_eq!(slice.chunks(0).count(), 0);
        let scalar = Tensor::new(shape![], vec![8]).unwrap();
        assert_eq!(scalar.slice(&[]).unwrap().chunks(2).count(), 1);
        assert_eq!(scalar.slice(&[]).unwrap().par_chunks(2).count(), 1);

        let empty = Tensor::<i32>::new(shape![0, 2], vec![]).unwrap();
        let empty_slice = empty.slice(&[0..0, 0..2]).unwrap();
        assert_eq!(empty_slice.chunks(2).count(), 0);
        assert_eq!(empty_slice.par_chunks(2).count(), 0);
    }

    #[test]
    fn indexing_reads_slice_relative_indices_and_rank_zero() {
        let tensor = Tensor::new(shape![3, 4], (0..12).collect()).unwrap();
        let slice = tensor.slice(&[1..3, 1..4]).unwrap();
        assert_eq!(slice[&[0, 0]], 5);
        assert_eq!(slice[&[1, 2]], 11);

        let scalar = Tensor::new(shape![], vec![42]).unwrap();
        assert_eq!(scalar.slice(&[]).unwrap()[&[]], 42);
    }

    #[test]
    #[should_panic]
    fn indexing_panics_outside_slice() {
        let tensor = Tensor::new(shape![2, 2], vec![1, 2, 3, 4]).unwrap();
        let slice = tensor.slice(&[0..1, 0..1]).unwrap();
        let _ = slice[&[1, 0]];
    }

    #[test]
    #[should_panic]
    fn indexing_panics_for_short_index_rank() {
        let tensor = Tensor::new(shape![2, 2], vec![1, 2, 3, 4]).unwrap();
        let slice = tensor.slice(&[0..2, 0..2]).unwrap();
        let _ = slice[&[0]];
    }

    #[test]
    #[should_panic]
    fn indexing_panics_for_long_index_rank() {
        let tensor = Tensor::new(shape![2, 2], vec![1, 2, 3, 4]).unwrap();
        let slice = tensor.slice(&[0..2, 0..2]).unwrap();
        let _ = slice[&[0, 0, 0]];
    }

    #[test]
    fn clone_into_tensor_preserves_scalar_and_zero_dimension_shapes() {
        let scalar = Tensor::new(shape![], vec![42]).unwrap();
        assert_eq!(
            scalar.slice(&[]).unwrap().clone_into_tensor(),
            Tensor::new(shape![], vec![42]).unwrap()
        );

        let tensor = Tensor::new(shape![3, 4], (0..12).collect()).unwrap();
        assert_eq!(
            tensor.slice(&[1..3, 1..3]).unwrap().clone_into_tensor(),
            Tensor::new(shape![2, 2], vec![5, 6, 9, 10]).unwrap()
        );
        let rank_three = Tensor::new(shape![2, 3, 2], (0..12).collect()).unwrap();
        assert_eq!(
            rank_three
                .slice(&[0..2, 1..3, 0..2])
                .unwrap()
                .clone_into_tensor(),
            Tensor::new(shape![2, 2, 2], vec![2, 3, 4, 5, 8, 9, 10, 11]).unwrap()
        );
        assert_eq!(
            tensor.slice(&[1..1, 0..4]).unwrap().clone_into_tensor(),
            Tensor::new(shape![0, 4], vec![]).unwrap()
        );
        assert_eq!(
            tensor.slice(&[0..3, 2..2]).unwrap().clone_into_tensor(),
            Tensor::new(shape![3, 0], vec![]).unwrap()
        );
        let empty = Tensor::<i32>::new(shape![0, 3], vec![]).unwrap();
        assert_eq!(
            empty.slice(&[0..0, 0..3]).unwrap().clone_into_tensor(),
            Tensor::new(shape![0, 3], vec![]).unwrap()
        );
    }
}
