use crate::definitions::matrix::Matrix;
use crate::definitions::matrix_slice::MatrixSlice;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;
use tensor_math::definitions::errors::TensorErrors;
/*
--------------------------------------------
* Immutable matrix slice utilities
--------------------------------------------
*/

impl<T> MatrixSlice<'_, T> {
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

    /// Gives another immutable slice relative to the current slice.
    /// This fails if the resultant slice would be out of bounds or have a negative dimension.
    pub fn reslice(
        &self,
        start_delta: (isize, isize),
        end_delta: (isize, isize),
    ) -> Result<MatrixSlice<'_, T>, TensorErrors> {
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

        self.orig.slice(
            new_start.0 as usize..new_end.0 as usize,
            new_start.1 as usize..new_end.1 as usize,
        )
    }
}

#[cfg(test)]
mod matrix_slice_utility_tests {
    use rayon::iter::ParallelIterator;
    use tensor_math::definitions::errors::TensorErrors;
    use tensor_math::definitions::matrix::Matrix;

    #[test]
    fn maps_and_enumerated_maps_keep_slice_dimensions() {
        let matrix = Matrix::new(3, 4, (0..12).collect()).unwrap();
        let slice = matrix.slice(1..3, 1..4).unwrap();
        let expected = Matrix::new(2, 3, vec![10, 12, 14, 18, 20, 22]).unwrap();

        assert_eq!(slice.map_refs(|value| value * 2), expected);
        assert_eq!(slice.par_map_refs(|value| value * 2), expected);
        let expected_enumerated = Matrix::new(2, 3, vec![5, 7, 9, 10, 12, 14]).unwrap();
        assert_eq!(
            slice.enumerated_map_refs(|((row, col), value)| value + (row + col) as i32),
            expected_enumerated
        );
        assert_eq!(
            slice.enumerated_par_map_refs(|((row, col), value)| value + (row + col) as i32),
            expected_enumerated
        );
    }

    #[test]
    fn enumerated_iterators_report_slice_relative_coordinates() {
        let matrix = Matrix::new(3, 4, (0..12).collect()).unwrap();
        let slice = matrix.slice(1..3, 1..4).unwrap();
        let expected = vec![
            ((0, 0), 5),
            ((0, 1), 6),
            ((0, 2), 7),
            ((1, 0), 9),
            ((1, 1), 10),
            ((1, 2), 11),
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
    }

    #[test]
    fn map_and_enumeration_variants_handle_zero_sized_shapes() {
        for (rows, cols, row_range, col_range) in
            [(0, 3, 0..0, 0..3), (3, 0, 0..3, 0..0), (0, 0, 0..0, 0..0)]
        {
            let matrix = Matrix::<i32>::new(rows, cols, vec![]).unwrap();
            let slice = matrix.slice(row_range, col_range).unwrap();

            let mapped = Matrix::new(slice.rows(), slice.cols(), vec![]).unwrap();
            assert_eq!(slice.map_refs(|value| *value), mapped);
            assert_eq!(slice.par_map_refs(|value| *value), mapped);
            assert_eq!(slice.enumerated_map_refs(|(_, value)| *value), mapped);
            assert_eq!(slice.enumerated_par_map_refs(|(_, value)| *value), mapped);
            assert_eq!(slice.enumerated_iter().count(), 0);
            assert_eq!(slice.enumerated_par_iter().count(), 0);
        }
    }

    #[test]
    fn reslice_adjusts_bounds_in_both_directions() {
        let matrix = Matrix::new(5, 6, (0..30).collect()).unwrap();
        let slice = matrix.slice(1..4, 1..5).unwrap();
        let resliced = slice.reslice((1, -1), (-1, -1)).unwrap();

        assert_eq!(resliced.start(), (2, 0));
        assert_eq!(resliced.end(), (3, 4));
        assert_eq!(resliced.shape(), tensor_math::shape![1, 4]);
        assert_eq!(
            resliced.map_refs(|value| *value),
            Matrix::new(1, 4, vec![12, 13, 14, 15]).unwrap()
        );
    }

    #[test]
    fn reslice_rejects_negative_start_and_end_positions() {
        let matrix = Matrix::new(4, 5, (0..20).collect()).unwrap();

        assert_eq!(
            matrix
                .slice(1..3, 1..4)
                .unwrap()
                .reslice((-2, 0), (0, 0))
                .unwrap_err(),
            TensorErrors::ReslicingNegativePosition("start", vec![-1, 1])
        );
        assert_eq!(
            matrix
                .slice(1..3, 1..4)
                .unwrap()
                .reslice((0, 0), (-4, 0))
                .unwrap_err(),
            TensorErrors::ReslicingNegativePosition("end", vec![-1, 4])
        );
    }

    #[test]
    fn reslice_rejects_negative_dimensions_and_out_of_parent_bounds() {
        let matrix = Matrix::new(4, 5, (0..20).collect()).unwrap();

        assert!(matches!(
            matrix.slice(1..3, 1..4).unwrap().reslice((3, 0), (0, 0)),
            Err(TensorErrors::InvalidInterval { .. })
        ));
        assert_eq!(
            matrix
                .slice(1..3, 1..4)
                .unwrap()
                .reslice((0, 0), (0, 2))
                .unwrap_err(),
            TensorErrors::SliceIndicesOutOfBounds {
                start: 1,
                end: 6,
                axis: 1,
                length: 5
            }
        );
    }
}
