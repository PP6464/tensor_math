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
        unsafe {
            self.iter()
                .map(f)
                .collect::<Matrix<_>>()
                .reshape_unchecked(self.rows(), self.cols())
        }
    }

    /// Applies the given function over the entire matrix elementwise by reference.
    pub fn par_map_refs<F: Send>(&self, f: impl Fn(&T) -> F + Send + Sync) -> Matrix<F>
    where
        T: Send + Sync,
    {
        unsafe {
            self.par_iter()
                .map(f)
                .collect::<Matrix<_>>()
                .reshape_unchecked(self.rows(), self.cols())
        }
    }

    /// Applies the given function over the entire matrix elementwise by reference, enumerated with matrix indices.
    pub fn enumerated_map_refs<F>(&self, f: impl FnMut(((usize, usize), &T)) -> F) -> Matrix<F> {
        unsafe {
            self.enumerated_iter()
                .map(f)
                .collect::<Matrix<_>>()
                .reshape_unchecked(self.rows(), self.cols())
        }
    }

    /// Applies the given function over the entire matrix elementwise by reference, enumerated with matrix indices.
    pub fn enumerated_par_map_refs<F: Send>(
        &self,
        f: impl Fn(((usize, usize), &T)) -> F + Send + Sync,
    ) -> Matrix<F>
    where
        T: Send + Sync,
    {
        unsafe {
            self.enumerated_par_iter()
                .map(f)
                .collect::<Matrix<_>>()
                .reshape_unchecked(self.rows(), self.cols())
        }
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
    pub fn enumerated_iter_mut(&mut self) -> impl Iterator<Item = ((usize, usize), &mut T)>
    where
        T: std::fmt::Debug,
    {
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

impl<'a, T> MatrixSliceMut<'a, T> {
    /// Converts into another mutable slice relative to the current slice.
    /// This fails if the resultant slice would be out of bounds or have a negative dimension.
    pub fn reslice_mut(
        self,
        start_delta: (isize, isize),
        end_delta: (isize, isize),
    ) -> Result<MatrixSliceMut<'a, T>, TensorErrors> {
        let new_start = (
            self.start.0 as isize + start_delta.0,
            self.start.1 as isize + start_delta.1,
        );
        let new_end = (
            self.end.0 as isize + end_delta.0,
            self.end.1 as isize + end_delta.1,
        );

        if new_start.0 < 0 || new_start.1 < 0 {
            return Err(TensorErrors::ReslicingNegativePosition(
                "start",
                vec![new_start.0, new_start.1],
            ));
        }

        if new_end.0 < 0 || new_end.1 < 0 {
            return Err(TensorErrors::ReslicingNegativePosition(
                "end",
                vec![new_end.0, new_end.1],
            ));
        }

        self.orig.slice_mut(
            new_start.0 as usize..new_end.0 as usize,
            new_start.1 as usize..new_end.1 as usize,
        )
    }
}

#[cfg(test)]
mod matrix_slice_mut_utility_tests {
    use rayon::iter::ParallelIterator;
    use tensor_math::definitions::errors::TensorErrors;
    use tensor_math::definitions::matrix::Matrix;
    use tensor_math::shape;

    #[test]
    fn map_variants_keep_slice_dimensions_and_use_relative_coordinates() {
        let mut matrix = Matrix::new(3, 4, (0..12).collect()).unwrap();
        let slice = matrix.slice_mut(1..3, 1..4).unwrap();
        assert_eq!(
            slice.map_refs(|value| value * 2),
            Matrix::new(2, 3, vec![10, 12, 14, 18, 20, 22]).unwrap()
        );
        assert_eq!(
            slice.par_map_refs(|value| value * 2),
            Matrix::new(2, 3, vec![10, 12, 14, 18, 20, 22]).unwrap()
        );
        assert_eq!(
            slice.enumerated_map_refs(|((row, col), value)| value + (row + col) as i32),
            Matrix::new(2, 3, vec![5, 7, 9, 10, 12, 14]).unwrap()
        );
        assert_eq!(
            slice.enumerated_par_map_refs(|((row, col), value)| { value + (row + col) as i32 }),
            Matrix::new(2, 3, vec![5, 7, 9, 10, 12, 14]).unwrap()
        );
        assert_eq!(
            slice
                .enumerated_iter()
                .map(|(index, value)| (index, *value))
                .collect::<Vec<_>>(),
            vec![
                ((0, 0), 5),
                ((0, 1), 6),
                ((0, 2), 7),
                ((1, 0), 9),
                ((1, 1), 10),
                ((1, 2), 11),
            ]
        );
        assert_eq!(slice.enumerated_par_iter().count(), 6);
    }

    #[test]
    fn mutable_enumerated_iterators_update_selected_elements_only() {
        let mut matrix = Matrix::new(2, 3, (0..6).collect()).unwrap();
        {
            let mut slice = matrix.slice_mut(0..2, 1..3).unwrap();
            slice.enumerated_iter_mut().for_each(|((row, col), value)| {
                *value = (row * 10 + col) as i32;
            });
        }
        assert_eq!(matrix.elements(), &[0, 0, 1, 3, 10, 11]);

        {
            let mut slice = matrix.slice_mut(0..2, 1..3).unwrap();
            slice
                .enumerated_par_iter_mut()
                .for_each(|((row, col), value)| *value += (row + col) as i32);
        }
        assert_eq!(matrix.elements(), &[0, 0, 2, 3, 11, 13]);
    }

    #[test]
    fn mapped_and_enumerated_empty_slices_keep_their_shapes() {
        for (rows, cols, row_range, col_range) in
            [(0, 3, 0..0, 0..3), (3, 0, 0..3, 0..0), (0, 0, 0..0, 0..0)]
        {
            let mut matrix = Matrix::<i32>::new(rows, cols, vec![]).unwrap();
            let mut slice = matrix.slice_mut(row_range, col_range).unwrap();
            let expected = Matrix::new(slice.rows(), slice.cols(), vec![]).unwrap();
            assert_eq!(slice.map_refs(|value| *value), expected);
            assert_eq!(slice.par_map_refs(|value| *value), expected);
            assert_eq!(slice.enumerated_map_refs(|(_, value)| *value), expected);
            assert_eq!(slice.enumerated_par_map_refs(|(_, value)| *value), expected);
            assert_eq!(slice.enumerated_iter().count(), 0);
            assert_eq!(slice.enumerated_par_iter().count(), 0);
            assert_eq!(slice.enumerated_iter_mut().count(), 0);
            assert_eq!(slice.enumerated_par_iter_mut().count(), 0);
        }
    }

    #[test]
    fn set_all_and_set_all_from_slice_copy_matching_shapes_and_reject_mismatches() {
        let mut matrix = Matrix::new(3, 3, (0..9).collect()).unwrap();
        {
            let mut slice = matrix.slice_mut(1..3, 1..3).unwrap();
            slice
                .set_all(&Matrix::new(2, 2, vec![10, 11, 12, 13]).unwrap())
                .unwrap();
        }
        assert_eq!(matrix.elements(), &[0, 1, 2, 3, 10, 11, 6, 12, 13]);

        let source = Matrix::new(2, 2, vec![20, 21, 22, 23]).unwrap();
        let source_slice = source.slice(0..2, 0..2).unwrap();
        {
            let mut slice = matrix.slice_mut(1..3, 1..3).unwrap();
            slice.set_all_from_slice(&source_slice).unwrap();
            let mismatch = Matrix::new(1, 4, vec![1; 4]).unwrap();
            assert_eq!(
                slice.set_all(&mismatch),
                Err(TensorErrors::IncompatibleShapes {
                    shape_1: shape![2, 2],
                    shape_2: mismatch.shape(),
                    op: "set_all"
                })
            );
            let mismatch_slice = mismatch.slice(0..1, 0..4).unwrap();
            assert_eq!(
                slice.set_all_from_slice(&mismatch_slice),
                Err(TensorErrors::IncompatibleShapes {
                    shape_1: shape![2, 2],
                    shape_2: mismatch.shape(),
                    op: "set_all"
                })
            );
        }
        assert_eq!(matrix.elements(), &[0, 1, 2, 3, 20, 21, 6, 22, 23]);
    }

    #[test]
    fn set_all_accepts_empty_matrices_and_zero_sized_slices() {
        for (rows, cols) in [(0, 0), (0, 3), (3, 0)] {
            let mut matrix = Matrix::<i32>::new(rows, cols, vec![]).unwrap();
            let mut slice = matrix.slice_mut(0..rows, 0..cols).unwrap();
            slice
                .set_all(&Matrix::new(rows, cols, vec![]).unwrap())
                .unwrap();
            let source = Matrix::<i32>::new(rows, cols, vec![]).unwrap();
            let source_slice = source.slice(0..rows, 0..cols).unwrap();
            slice.set_all_from_slice(&source_slice).unwrap();
        }
    }

    #[test]
    fn reslice_mut_adjusts_bounds_and_keeps_mutations_within_result() {
        let mut matrix = Matrix::new(4, 5, (0..20).collect()).unwrap();
        {
            let slice = matrix.slice_mut(1..3, 1..4).unwrap();
            let mut resliced = slice.reslice_mut((1, 0), (1, 1)).unwrap();

            assert_eq!(resliced.start(), (2, 1));
            assert_eq!(resliced.end(), (4, 5));
            assert_eq!(resliced.shape(), shape![2, 4]);
            assert_eq!(
                resliced.map_refs(|value| *value),
                Matrix::new(2, 4, (11..15).chain(16..20).collect()).unwrap()
            );
            resliced[(0, 0)] = 100;
        }
        assert_eq!(matrix[(2, 1)], 100);
        assert_eq!(matrix[(1, 1)], 6);
    }

    #[test]
    fn reslice_mut_rejects_invalid_intervals_and_parent_bounds() {
        let mut matrix = Matrix::new(4, 5, (0..20).collect()).unwrap();

        assert!(matches!(
            matrix
                .slice_mut(1..3, 1..4)
                .unwrap()
                .reslice_mut((0, 0), (2, 2)),
            Err(TensorErrors::SliceIndicesOutOfBounds { axis: 0, .. })
        ));
        assert!(matches!(
            matrix
                .slice_mut(1..3, 1..4)
                .unwrap()
                .reslice_mut((3, 0), (0, 0)),
            Err(TensorErrors::InvalidInterval { .. })
        ));
        assert_eq!(
            matrix
                .slice_mut(1..3, 1..4)
                .unwrap()
                .reslice_mut((-2, 0), (0, 0))
                .unwrap_err(),
            TensorErrors::ReslicingNegativePosition("start", vec![-1, 1])
        );
        assert_eq!(
            matrix
                .slice_mut(1..3, 1..4)
                .unwrap()
                .reslice_mut((0, 0), (-4, 0))
                .unwrap_err(),
            TensorErrors::ReslicingNegativePosition("end", vec![-1, 4])
        );
    }
}
