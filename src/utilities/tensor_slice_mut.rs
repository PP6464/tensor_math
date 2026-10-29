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
}

impl<'a, T> TensorSliceMut<'a, T> {
    /// Gives another immutable slice relative to the current slice.
    /// This fails if the resultant slice would be out of bounds or have a negative dimension.
    pub fn reslice_mut(
        self,
        start_delta: &[isize],
        end_delta: &[isize],
    ) -> Result<TensorSliceMut<'a, T>, TensorErrors> {
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

        self.orig.slice_mut(
            &new_start
                .into_iter()
                .zip(new_end)
                .map(|(x, y)| x as usize..y as usize)
                .collect::<Vec<_>>(),
        )
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

#[cfg(test)]
mod tensor_slice_mut_utility_tests {
    use rayon::iter::ParallelIterator;
    use tensor_math::definitions::errors::TensorErrors;
    use tensor_math::definitions::tensor::Tensor;
    use tensor_math::shape;

    #[test]
    fn immutable_maps_and_enumeration_preserve_slice_relative_indices() {
        let mut tensor = Tensor::new(shape![3, 4], (0..12).collect()).unwrap();
        let slice = tensor.slice_mut(&[1..3, 1..4]).unwrap();
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
        assert_eq!(
            slice
                .enumerated_iter()
                .map(|(index, value)| (index, *value))
                .collect::<Vec<_>>(),
            vec![
                (vec![0, 0], 5),
                (vec![0, 1], 6),
                (vec![0, 2], 7),
                (vec![1, 0], 9),
                (vec![1, 1], 10),
                (vec![1, 2], 11),
            ]
        );
        assert_eq!(slice.enumerated_par_iter().count(), 6);
    }

    #[test]
    fn mutable_enumerated_iterators_update_only_selected_values() {
        let mut tensor = Tensor::new(shape![2, 3], (0..6).collect()).unwrap();
        {
            let mut slice = tensor.slice_mut(&[0..2, 1..3]).unwrap();
            slice.enumerated_iter_mut().for_each(|(index, value)| {
                *value = (index[0] * 10 + index[1]) as i32;
            });
        }
        assert_eq!(tensor.elements(), &[0, 0, 1, 3, 10, 11]);

        {
            let mut slice = tensor.slice_mut(&[0..2, 1..3]).unwrap();
            slice
                .enumerated_par_iter_mut()
                .for_each(|(index, value)| *value += (index[0] + index[1]) as i32);
        }
        assert_eq!(tensor.elements(), &[0, 0, 2, 3, 11, 13]);
    }

    #[test]
    fn mutable_maps_and_enumeration_handle_scalar_and_empty_slices() {
        let mut scalar = Tensor::new(shape![], vec![5]).unwrap();
        {
            let mut slice = scalar.slice_mut(&[]).unwrap();
            assert_eq!(slice.rank(), 0);
            assert_eq!(
                slice.map_refs(|value| value + 1),
                Tensor::new(shape![], vec![6]).unwrap()
            );
            assert_eq!(
                slice.par_map_refs(|value| value + 1),
                Tensor::new(shape![], vec![6]).unwrap()
            );
            assert_eq!(
                slice.enumerated_map_refs(|(index, value)| {
                    assert!(index.is_empty());
                    value + 1
                }),
                Tensor::new(shape![], vec![6]).unwrap()
            );
            assert_eq!(
                slice.enumerated_par_map_refs(|(index, value)| value + index.len() as i32 + 1),
                Tensor::new(shape![], vec![6]).unwrap()
            );
            slice.enumerated_iter_mut().for_each(|(index, value)| {
                assert!(index.is_empty());
                *value += 1;
            });
            slice
                .enumerated_par_iter_mut()
                .for_each(|(index, value)| *value += index.len() as i32 + 1);
        }
        assert_eq!(scalar.elements(), &[7]);

        for (tensor_shape, ranges, slice_shape) in [
            (shape![0, 3], vec![0..0, 0..3], shape![0, 3]),
            (shape![3, 0], vec![0..3, 0..0], shape![3, 0]),
            (shape![0, 0], vec![0..0, 0..0], shape![0, 0]),
        ] {
            let mut tensor = Tensor::<i32>::new(tensor_shape, vec![]).unwrap();
            let slice = tensor.slice_mut(&ranges).unwrap();
            assert_eq!(slice.map_refs(|value| *value).shape(), slice_shape);
            assert_eq!(slice.par_map_refs(|value| *value).shape(), slice_shape);
            assert_eq!(
                slice.enumerated_map_refs(|(_, value)| *value).shape(),
                slice_shape
            );
            assert_eq!(
                slice.enumerated_par_map_refs(|(_, value)| *value).shape(),
                slice_shape
            );
            assert_eq!(slice.enumerated_iter().count(), 0);
            assert_eq!(slice.enumerated_par_iter().count(), 0);
        }
    }

    #[test]
    fn set_all_and_set_all_from_slice_require_equal_shapes() {
        let mut tensor = Tensor::new(shape![3, 3], (0..9).collect()).unwrap();
        {
            let mut slice = tensor.slice_mut(&[1..3, 1..3]).unwrap();
            slice
                .set_all(&Tensor::new(shape![2, 2], vec![10, 11, 12, 13]).unwrap())
                .unwrap();
        }
        assert_eq!(tensor.elements(), &[0, 1, 2, 3, 10, 11, 6, 12, 13]);

        let replacement = Tensor::new(shape![2, 2], vec![20, 21, 22, 23]).unwrap();
        let replacement_slice = replacement.slice(&[0..2, 0..2]).unwrap();
        {
            let mut slice = tensor.slice_mut(&[1..3, 1..3]).unwrap();
            slice.set_all_from_slice(&replacement_slice).unwrap();
            assert_eq!(
                slice.set_all(&Tensor::new(shape![1, 4], vec![1; 4]).unwrap()),
                Err(TensorErrors::IncompatibleShapes {
                    shape_1: shape![2, 2],
                    shape_2: shape![1, 4],
                    op: "set_all"
                })
            );
            assert_eq!(
                slice.set_all_from_slice(
                    &Tensor::new(shape![1, 4], vec![1; 4])
                        .unwrap()
                        .as_tensor_slice()
                ),
                Err(TensorErrors::IncompatibleShapes {
                    shape_1: shape![2, 2],
                    shape_2: shape![1, 4],
                    op: "set_all"
                })
            );
        }
        assert_eq!(tensor.elements(), &[0, 1, 2, 3, 20, 21, 6, 22, 23]);
    }

    #[test]
    fn set_all_accepts_scalar_and_matching_empty_shapes() {
        let mut scalar = Tensor::new(shape![], vec![0]).unwrap();
        scalar
            .slice_mut(&[])
            .unwrap()
            .set_all(&Tensor::new(shape![], vec![9]).unwrap())
            .unwrap();
        assert_eq!(scalar.elements(), &[9]);
        let scalar_source = Tensor::new(shape![], vec![11]).unwrap();
        scalar
            .slice_mut(&[])
            .unwrap()
            .set_all_from_slice(&scalar_source.as_tensor_slice())
            .unwrap();
        assert_eq!(scalar.elements(), &[11]);

        for (tensor_shape, ranges, target_shape) in [
            (shape![0, 3], vec![0..0, 0..3], shape![0, 3]),
            (shape![3, 0], vec![0..3, 0..0], shape![3, 0]),
        ] {
            let mut tensor = Tensor::<i32>::new(tensor_shape, vec![]).unwrap();
            tensor
                .slice_mut(&ranges)
                .unwrap()
                .set_all(&Tensor::new(target_shape, vec![]).unwrap())
                .unwrap();
        }
    }

    #[test]
    fn reslice_adjusts_bounds_and_keeps_mutations_within_result() {
        let mut tensor = Tensor::new(shape![4, 5], (0..20).collect()).unwrap();
        {
            let slice = tensor.slice_mut(&[1..3, 1..4]).unwrap();
            let mut resliced = slice.reslice_mut(&vec![1, 0], &vec![1, 1]).unwrap();

            assert_eq!(resliced.start(), &[2, 1]);
            assert_eq!(resliced.end(), &[4, 5]);
            assert_eq!(resliced.shape(), shape![2, 4]);
            assert_eq!(
                resliced.map_refs(|value| *value),
                Tensor::new(shape![2, 4], (11..15).chain(16..20).collect()).unwrap()
            );
            resliced[&[0, 0]] = 100;
        }
        assert_eq!(tensor[&[2, 1]], 100);
        assert_eq!(tensor[&[1, 1]], 6);
    }

    #[test]
    fn reslice_rejects_invalid_ranks_intervals_and_parent_bounds() {
        let mut tensor = Tensor::new(shape![4, 5], (0..20).collect()).unwrap();

        assert_eq!(
            tensor
                .slice_mut(&[1..3, 1..4])
                .unwrap()
                .reslice_mut(&vec![0], &vec![0, 0])
                .unwrap_err(),
            TensorErrors::RanksDoNotMatch(1, 2)
        );
        assert!(matches!(
            tensor
                .slice_mut(&[1..3, 1..4])
                .unwrap()
                .reslice_mut(&vec![0, 0], &vec![3, 0]),
            Err(TensorErrors::SliceIndicesOutOfBounds { axis: 0, .. })
        ));
        assert!(matches!(
            tensor
                .slice_mut(&[1..3, 1..4])
                .unwrap()
                .reslice_mut(&vec![3, 0], &vec![0, 0]),
            Err(TensorErrors::InvalidInterval { .. })
        ));
        assert_eq!(
            tensor
                .slice_mut(&[1..3, 1..4])
                .unwrap()
                .reslice_mut(&vec![-2, 0], &vec![0, 0])
                .unwrap_err(),
            TensorErrors::ReslicingNegativePosition("start", vec![-1, 1])
        );
        assert_eq!(
            tensor
                .slice_mut(&[1..3, 1..4])
                .unwrap()
                .reslice_mut(&vec![0, 0], &vec![-4, 0])
                .unwrap_err(),
            TensorErrors::ReslicingNegativePosition("end", vec![-1, 4])
        );
    }

    #[test]
    fn reslice_indices_invalid_for_rank() {
        let mut t1 = Tensor::<i32>::new(shape![2, 3], (0..6).collect()).unwrap();
        let err = t1.slice_mut(&vec![0..1, 0..1]).unwrap().reslice_mut(&vec![0, 0, 0], &vec![1, 1]).unwrap_err();
        assert!(matches!(err, TensorErrors::RanksDoNotMatch(..)));
        let err = t1.slice_mut(&vec![0..1, 0..1]).unwrap().reslice_mut(&vec![0, 0], &vec![1, 1, 1]).unwrap_err();
        assert!(matches!(err, TensorErrors::RanksDoNotMatch(..)));
    }
}
