use crate::definitions::errors::TensorErrors;
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
        unsafe {
            self.iter()
                .map(f)
                .collect::<Tensor<_>>()
                .reshape_unchecked(self.shape())
        }
    }

    /// Applies the given function over the entire tensor elementwise by reference.
    pub fn par_map_refs<F: Send>(&self, f: impl Fn(&T) -> F + Send + Sync) -> Tensor<F>
    where
        T: Send + Sync,
    {
        unsafe {
            self.par_iter()
                .map(f)
                .collect::<Tensor<_>>()
                .reshape_unchecked(self.shape())
        }
    }

    /// Applies the given function over the entire tensor elementwise by reference, enumerated with tensor indices.
    pub fn enumerated_map_refs<F>(&self, f: impl FnMut((Vec<usize>, &T)) -> F) -> Tensor<F> {
        unsafe {
            self.enumerated_iter()
                .map(f)
                .collect::<Tensor<_>>()
                .reshape_unchecked(self.shape())
        }
    }

    /// Applies the given function over the entire tensor elementwise by reference, enumerated with tensor indices.
    pub fn enumerated_par_map_refs<F: Send>(
        &self,
        f: impl Fn((Vec<usize>, &T)) -> F + Send + Sync,
    ) -> Tensor<F>
    where
        T: Send + Sync,
    {
        unsafe {
            self.enumerated_par_iter()
                .map(f)
                .collect::<Tensor<_>>()
                .reshape_unchecked(self.shape())
        }
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

    /// Gives another immutable slice relative to the current slice.
    /// This fails if the resultant slice would be out of bounds or have a negative dimension.
    pub fn reslice(
        &self,
        start_delta: &[isize],
        end_delta: &[isize],
    ) -> Result<TensorSlice<'_, T>, TensorErrors> {
        if start_delta.len() != self.start.len() {
            return Err(TensorErrors::RanksDoNotMatch(
                start_delta.len(),
                self.start.len(),
            ));
        }

        if end_delta.len() != self.end.len() {
            return Err(TensorErrors::RanksDoNotMatch(
                end_delta.len(),
                self.end.len(),
            ));
        }

        let new_start = self
            .start
            .iter()
            .zip(start_delta.iter())
            .map(|(x, y)| *x as isize + *y)
            .collect::<Vec<_>>();
        let new_end = self
            .end
            .iter()
            .zip(end_delta.iter())
            .map(|(x, y)| *x as isize + *y)
            .collect::<Vec<_>>();

        if new_start.iter().any(|x| *x < 0) {
            return Err(TensorErrors::ReslicingNegativePosition("start", new_start));
        }

        if new_end.iter().any(|x| *x < 0) {
            return Err(TensorErrors::ReslicingNegativePosition("end", new_end));
        }

        self.orig.slice(
            &new_start
                .into_iter()
                .zip(new_end)
                .map(|(x, y)| x as usize..y as usize)
                .collect::<Vec<_>>(),
        )
    }
}

#[cfg(test)]
mod tensor_slice_utility_tests {
    use rayon::iter::ParallelIterator;
    use tensor_math::definitions::errors::TensorErrors;
    use tensor_math::definitions::tensor::Tensor;
    use tensor_math::shape;

    #[test]
    fn map_variants_preserve_slice_shape_and_transform_values() {
        let tensor = Tensor::new(shape![3, 4], (0..12).collect()).unwrap();
        let slice = tensor.slice(&[1..3, 1..4]).unwrap();
        let expected = Tensor::new(shape![2, 3], vec![10, 12, 14, 18, 20, 22]).unwrap();

        assert_eq!(slice.map_refs(|value| value * 2), expected);
        assert_eq!(slice.par_map_refs(|value| value * 2), expected);
        assert_eq!(
            slice.enumerated_map_refs(|(index, value)| value + index.iter().sum::<usize>() as i32),
            Tensor::new(shape![2, 3], vec![5, 7, 9, 10, 12, 14]).unwrap()
        );
        assert_eq!(
            slice.enumerated_par_map_refs(
                |(index, value)| value + index.iter().sum::<usize>() as i32
            ),
            Tensor::new(shape![2, 3], vec![5, 7, 9, 10, 12, 14]).unwrap()
        );
    }

    #[test]
    fn enumerated_iterators_use_slice_relative_indices() {
        let tensor = Tensor::new(shape![3, 4], (0..12).collect()).unwrap();
        let slice = tensor.slice(&[1..3, 1..4]).unwrap();
        let expected = vec![
            (vec![0, 0], 5),
            (vec![0, 1], 6),
            (vec![0, 2], 7),
            (vec![1, 0], 9),
            (vec![1, 1], 10),
            (vec![1, 2], 11),
        ];

        assert_eq!(
            slice
                .enumerated_iter()
                .map(|(index, value)| (index, *value))
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(
            slice
                .enumerated_par_iter()
                .map(|(index, value)| (index, *value))
                .collect::<Vec<_>>(),
            expected
        );

        let scalar = Tensor::new(shape![], vec![42]).unwrap();
        let scalar_slice = scalar.slice(&[]).unwrap();
        assert_eq!(
            scalar_slice.map_refs(|value| value + 1),
            Tensor::new(shape![], vec![43]).unwrap()
        );
        assert_eq!(
            scalar_slice.par_map_refs(|value| value + 1),
            Tensor::new(shape![], vec![43]).unwrap()
        );
        assert_eq!(
            scalar_slice.enumerated_map_refs(|(index, value)| value + index.len() as i32 + 1),
            Tensor::new(shape![], vec![43]).unwrap()
        );
        assert_eq!(
            scalar_slice.enumerated_par_map_refs(|(index, value)| value + index.len() as i32 + 1),
            Tensor::new(shape![], vec![43]).unwrap()
        );
        assert_eq!(
            scalar_slice
                .enumerated_iter()
                .map(|(index, value)| (index, *value))
                .collect::<Vec<_>>(),
            vec![(vec![], 42)]
        );
        assert_eq!(scalar_slice.enumerated_par_iter().count(), 1);
    }

    #[test]
    fn all_map_and_enumeration_variants_handle_empty_dimensions() {
        for (shape, ranges, expected_shape) in [
            (shape![0, 3], vec![0..0, 0..3], shape![0, 3]),
            (shape![3, 0], vec![0..3, 0..0], shape![3, 0]),
            (shape![0, 0], vec![0..0, 0..0], shape![0, 0]),
        ] {
            let tensor = Tensor::<i32>::new(shape, vec![]).unwrap();
            let slice = tensor.slice(&ranges).unwrap();
            assert_eq!(slice.map_refs(|value| *value).shape(), expected_shape);
            assert_eq!(slice.par_map_refs(|value| *value).shape(), expected_shape);
            assert_eq!(
                slice.enumerated_map_refs(|(_, value)| *value).shape(),
                expected_shape
            );
            assert_eq!(
                slice.enumerated_par_map_refs(|(_, value)| *value).shape(),
                expected_shape
            );
            assert_eq!(slice.enumerated_iter().count(), 0);
            assert_eq!(slice.enumerated_par_iter().count(), 0);
        }
    }

    #[test]
    fn reslice_adjusts_bounds_in_both_directions() {
        let tensor = Tensor::new(shape![5, 6], (0..30).collect()).unwrap();
        let slice = tensor.slice(&[1..4, 1..5]).unwrap();
        let resliced = slice.reslice(&vec![1, -1], &vec![-1, -1]).unwrap();

        assert_eq!(resliced.start(), &[2, 0]);
        assert_eq!(resliced.end(), &[3, 4]);
        assert_eq!(resliced.shape(), shape![1, 4]);
        assert_eq!(
            resliced.map_refs(|value| *value),
            Tensor::new(shape![1, 4], vec![12, 13, 14, 15]).unwrap()
        );
    }

    #[test]
    fn reslice_rejects_negative_start_and_end_positions() {
        let tensor = Tensor::new(shape![4, 5], (0..20).collect()).unwrap();

        assert_eq!(
            tensor
                .slice(&[1..3, 1..4])
                .unwrap()
                .reslice(&vec![-2, 0], &vec![0, 0])
                .unwrap_err(),
            TensorErrors::ReslicingNegativePosition("start", vec![-1, 1])
        );
        assert_eq!(
            tensor
                .slice(&[1..3, 1..4])
                .unwrap()
                .reslice(&vec![0, 0], &vec![-4, 0])
                .unwrap_err(),
            TensorErrors::ReslicingNegativePosition("end", vec![-1, 4])
        );
    }

    #[test]
    fn reslice_rejects_rank_mismatch_and_out_of_parent_bounds() {
        let tensor = Tensor::new(shape![4, 5], (0..20).collect()).unwrap();

        assert_eq!(
            tensor
                .slice(&[1..3, 1..4])
                .unwrap()
                .reslice(&vec![0], &vec![0, 0])
                .unwrap_err(),
            TensorErrors::RanksDoNotMatch(1, 2)
        );
        assert_eq!(
            tensor
                .slice(&[1..3, 1..4])
                .unwrap()
                .reslice(&vec![0, 0], &vec![0, 2])
                .unwrap_err(),
            TensorErrors::SliceIndicesOutOfBounds {
                start: 1,
                end: 6,
                axis: 1,
                length: 5
            }
        );
        assert!(matches!(
            tensor
                .slice(&[1..3, 1..4])
                .unwrap()
                .reslice(&vec![3, 0], &vec![0, 0]),
            Err(TensorErrors::InvalidInterval { .. })
        ));
    }

    #[test]
    fn reslice_indices_invalid_for_rank() {
        let t1 = Tensor::<i32>::new(shape![2, 3], (0..6).collect()).unwrap();
        let err = t1.slice(&vec![0..1, 0..1]).unwrap().reslice(&vec![0, 0, 0], &vec![1, 1]).unwrap_err();
        assert!(matches!(err, TensorErrors::RanksDoNotMatch(..)));
        let err = t1.slice(&vec![0..1, 0..1]).unwrap().reslice(&vec![0, 0], &vec![1, 1, 1]).unwrap_err();
        assert!(matches!(err, TensorErrors::RanksDoNotMatch(..)));
    }
}
