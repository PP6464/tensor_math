use crate::definitions::matrix::Matrix;
use float_cmp::approx_eq;
use num::complex::{Complex64, ComplexFloat};

impl Matrix<f64> {
    /// Gives whether the matrix is in row echelon form or not.
    pub fn is_row_echelon(&self) -> bool {
        let mut all_zero_rows = false;
        let mut prev_pivot_col: i32 = -1;

        for i in 0..self.rows {
            let current_row = unsafe { self.slice_unchecked(i..i + 1, 0..self.cols) };

            if all_zero_rows && !current_row.iter().all(|x| approx_eq!(f64, *x, 0.0)) {
                return false; // There is a row below a row of all 0 that is not itself all 0
            }

            if current_row.iter().all(|x| approx_eq!(f64, *x, 0.0)) {
                all_zero_rows = true;
                continue;
            }

            let current_pivot_col = current_row
                .iter()
                .position(|&x| !approx_eq!(f64, x, 0.0))
                .unwrap();

            if current_pivot_col as i32 <= prev_pivot_col {
                return false;
            }

            prev_pivot_col = current_pivot_col as i32;
        }

        true
    }

    /// Gives whether the matrix is in reduced row echelon form or not.
    pub fn is_reduced_row_echelon(&self) -> bool {
        let mut pivot = (0, 0);

        while pivot.0 < self.rows && pivot.1 < self.cols {
            let pivot_val = self[pivot];

            // Note everything below must be 0
            if pivot.0 < self.rows - 1 {
                let below_slice =
                    unsafe { self.slice_unchecked(pivot.0 + 1..self.rows, pivot.1..pivot.1 + 1) };

                if below_slice.iter().any(|x| !approx_eq!(f64, *x, 0.0)) {
                    return false;
                }
            }

            if approx_eq!(f64, pivot_val, 0.0) {
                // If this is the last element in the row then there are no suitable other pivots
                if pivot.1 == self.cols - 1 {
                    // This row is all 0 otherwise, so just check everything below is all 0 as well
                    return unsafe { self.slice_unchecked(pivot.0 + 1..self.rows, 0..self.cols) }
                        .iter()
                        .all(|x| approx_eq!(f64, *x, 0.0));
                }

                // Check for all 0, otherwise just move to the first non-zero element
                let right_slice =
                    unsafe { self.slice_unchecked(pivot.0..pivot.0 + 1, pivot.1 + 1..self.cols) };
                let option_pivot = right_slice.iter().position(|&x| !approx_eq!(f64, x, 0.0));

                match option_pivot {
                    Some(index) => {
                        pivot = (pivot.0, pivot.1 + 1 + index);
                        continue;
                    }
                    None => {
                        // This row is all 0, check rows below
                        return unsafe {
                            self.slice_unchecked(pivot.0 + 1..self.rows, 0..self.cols)
                        }
                        .iter()
                        .all(|x| approx_eq!(f64, *x, 0.0));
                    }
                }
            }

            // Note everything to the left should be 0
            if pivot.1 > 0 {
                let left_slice = unsafe { self.slice_unchecked(pivot.0..pivot.0 + 1, 0..pivot.1) };

                if left_slice.iter().any(|x| !approx_eq!(f64, *x, 0.0)) {
                    return false;
                }
            }

            // Similarly everything above must be 0
            if pivot.0 > 0 {
                let above_slice = unsafe { self.slice_unchecked(0..pivot.0, pivot.1..pivot.1 + 1) };

                if above_slice.iter().any(|x| !approx_eq!(f64, *x, 0.0)) {
                    return false;
                }
            }

            // The pivot value must also be normalised
            if !approx_eq!(f64, pivot_val, 1.0) {
                return false;
            }

            // Move pivot by 1 diagonally since as we have reached here this is a normal pivot
            pivot = (pivot.0 + 1, pivot.1 + 1);
        }

        true
    }

    /// Computes the REF form of a matrix.
    /// As this is REF the leading entries will not be normalised.
    pub(crate) fn tracked_row_echelon(mut self) -> (Matrix<f64>, i32, Vec<(usize, usize)>) {
        let mut det_scale = 1;
        let mut pivot = (0usize, 0usize);
        let mut pivots = Vec::with_capacity(self.rows);

        while pivot.0 < self.rows && pivot.1 < self.cols {
            // Find the largest usable pivot
            let (row, &value) =
                unsafe { self.slice_unchecked(pivot.0..self.rows, pivot.1..pivot.1 + 1) }
                    .iter()
                    .enumerate()
                    .max_by(|(_, &x), (_, &y)| x.abs().total_cmp(&y.abs()))
                    .unwrap();

            // If the entire column is 0 we skip to the next column
            if approx_eq!(f64, value, 0.0) {
                pivot.1 += 1;
                continue;
            }

            // Swap the pivot row with the current row if necessary
            // Note that the row is an offset from the current pivot row
            if row != 0 {
                unsafe {
                    self.swap_rows_unchecked(pivot.0, pivot.0 + row);
                }
                det_scale *= -1;
            }

            pivots.push(pivot);

            // Eliminate rows below
            // We can do this as a rank1_update_sub
            unsafe {
                let vec1 = self
                    .slice_unchecked(pivot.0 + 1..self.rows, pivot.1..pivot.1 + 1)
                    .iter()
                    .map(|&x| x / value)
                    .collect::<Vec<_>>();
                let vec2 = self
                    .slice_unchecked(pivot.0..pivot.0 + 1, pivot.1..self.cols)
                    .iter()
                    .copied()
                    .collect::<Vec<_>>();

                self.slice_unchecked_mut(pivot.0 + 1..self.rows, pivot.1..self.cols)
                    .rank1_update_sub_unchecked_mt(&vec1, &vec2);
            }

            pivot = (pivot.0 + 1, pivot.1 + 1);
        }

        (self, det_scale, pivots)
    }

    /// Computes the row echelon form of a matrix.
    /// As this is REF the leading entries will not be normalised.
    pub fn row_echelon(self) -> Matrix<f64> {
        self.tracked_row_echelon().0
    }

    /// Computes the reduced row echelon form of a matrix.
    pub fn reduced_row_echelon(self) -> Matrix<f64> {
        let (mut ref_form, _, pivots) = self.tracked_row_echelon();
        let cols = ref_form.cols;

        // Go backwards through the pivots: normalise row and eliminate above
        for &pivot in pivots.iter().rev() {
            let val = ref_form[pivot];

            // Normalise row
            unsafe {
                ref_form
                    .slice_unchecked_mut(pivot.0..pivot.0 + 1, pivot.1..cols)
                    .iter_mut()
                    .for_each(|x| *x /= val);
            }

            // Eliminate above
            let row = unsafe { ref_form.slice_unchecked(pivot.0..pivot.0 + 1, pivot.1..cols) }
                .iter()
                .copied()
                .collect::<Vec<_>>();
            let col = unsafe { ref_form.slice_unchecked(0..pivot.0, pivot.1..pivot.1 + 1) }
                .iter()
                .copied()
                .collect::<Vec<_>>();

            unsafe {
                ref_form
                    .slice_unchecked_mut(0..pivot.0, pivot.1..cols)
                    .rank1_update_sub_unchecked_mt(&col, &row);
            }
        }

        ref_form
    }
}

impl Matrix<Complex64> {
    /// Gives whether the matrix is in row echelon form or not.
    pub fn is_row_echelon(&self) -> bool {
        let mut all_zero_rows = false;
        let mut prev_pivot_col: i32 = -1;

        for i in 0..self.rows {
            let current_row = unsafe { self.slice_unchecked(i..i + 1, 0..self.cols) };

            if all_zero_rows && !current_row.iter().all(|x| approx_eq!(f64, (*x).abs(), 0.0)) {
                return false; // There is a row below a row of all 0 that is not itself all 0
            }

            if current_row.iter().all(|x| approx_eq!(f64, (*x).abs(), 0.0)) {
                all_zero_rows = true;
                continue;
            }

            let current_pivot_col = current_row
                .iter()
                .position(|&x| !approx_eq!(f64, x.abs(), 0.0))
                .unwrap();

            if current_pivot_col as i32 <= prev_pivot_col {
                return false;
            }

            prev_pivot_col = current_pivot_col as i32;
        }

        true
    }

    /// Gives whether the matrix is in reduced row echelon form or not.
    pub fn is_reduced_row_echelon(&self) -> bool {
        let mut pivot = (0, 0);

        while pivot.0 < self.rows && pivot.1 < self.cols {
            let pivot_val = self[pivot];

            // Note everything below must be 0
            if pivot.0 < self.rows - 1 {
                let below_slice =
                    unsafe { self.slice_unchecked(pivot.0 + 1..self.rows, pivot.1..pivot.1 + 1) };

                if below_slice
                    .iter()
                    .any(|x| !approx_eq!(f64, (*x).abs(), 0.0))
                {
                    return false;
                }
            }

            if approx_eq!(f64, pivot_val.abs(), 0.0) {
                // If this is the last element in the row then there are no suitable other pivots
                if pivot.1 == self.cols - 1 {
                    // This row is all 0 so just check everything below is all 0 as well
                    unsafe {
                        return self
                            .slice_unchecked(pivot.0 + 1..self.rows, 0..self.cols)
                            .iter()
                            .all(|x| approx_eq!(f64, (*x).abs(), 0.0));
                    }
                }

                // Check for all 0, otherwise just move to the first non-zero element
                let right_slice =
                    unsafe { self.slice_unchecked(pivot.0..pivot.0 + 1, pivot.1 + 1..self.cols) };
                let option_pivot = right_slice
                    .iter()
                    .enumerate()
                    .find(|(_, &x)| !approx_eq!(f64, x.abs(), 0.0));

                match option_pivot {
                    Some((index, _)) => {
                        pivot = (pivot.0, pivot.1 + 1 + index);
                        continue;
                    }
                    None => {
                        // This row is all 0, check rows below
                        unsafe {
                            return self
                                .slice_unchecked(pivot.0 + 1..self.rows, 0..self.cols)
                                .iter()
                                .all(|x| approx_eq!(f64, (*x).abs(), 0.0));
                        }
                    }
                }
            }

            // Note everything to the left should be 0
            if pivot.1 > 0 {
                let left_slice = unsafe { self.slice_unchecked(pivot.0..pivot.0 + 1, 0..pivot.1) };

                if left_slice.iter().any(|x| !approx_eq!(f64, (*x).abs(), 0.0)) {
                    return false;
                }
            }

            // Similarly everything above must be 0
            if pivot.0 > 0 {
                let above_slice = unsafe { self.slice_unchecked(0..pivot.0, pivot.1..pivot.1 + 1) };

                if above_slice
                    .iter()
                    .any(|x| !approx_eq!(f64, (*x).abs(), 0.0))
                {
                    return false;
                }
            }

            // The pivot value must also be normalised
            if !approx_eq!(f64, pivot_val.re, 1.0) || !approx_eq!(f64, pivot_val.im, 0.0) {
                return false;
            }

            // Move pivot by 1 diagonally since as we have reached here this is a normal pivot
            pivot = (pivot.0 + 1, pivot.1 + 1);
        }

        true
    }

    /// Computes the row echelon form of a matrix.
    /// As this is REF the leading entries will not be normalised.
    pub(crate) fn tracked_row_echelon(mut self) -> (Matrix<Complex64>, i32, Vec<(usize, usize)>) {
        let mut det_scale = 1;
        let mut pivot = (0usize, 0usize);
        let mut pivots = Vec::with_capacity(self.rows);

        while pivot.0 < self.rows && pivot.1 < self.cols {
            // Find the largest usable pivot
            let (row, &value) =
                unsafe { self.slice_unchecked(pivot.0..self.rows, pivot.1..pivot.1 + 1) }
                    .iter()
                    .enumerate()
                    .max_by(|(_, &x), (_, &y)| x.abs().total_cmp(&y.abs()))
                    .unwrap();

            // If the entire column is 0 we skip to the next column
            if approx_eq!(f64, value.norm_sqr(), 0.0) {
                pivot.1 += 1;
                continue;
            }

            // Swap the pivot row with the current row if necessary
            // Note the row is an offset to the current pivot row
            if row != 0 {
                unsafe {
                    self.swap_rows_unchecked(pivot.0, pivot.0 + row);
                }
                det_scale *= -1;
            }

            pivots.push(pivot);

            // Eliminate rows below
            // We can do this as a rank1_update_sub.
            unsafe {
                let vec1 = self
                    .slice_unchecked(pivot.0 + 1..self.rows, pivot.1..pivot.1 + 1)
                    .iter()
                    .map(|&x| x / value)
                    .collect::<Vec<_>>();
                let vec2 = self
                    .slice_unchecked(pivot.0..pivot.0 + 1, pivot.1..self.cols)
                    .iter()
                    .copied()
                    .collect::<Vec<_>>();

                self.slice_unchecked_mut(pivot.0 + 1..self.rows, pivot.1..self.cols)
                    .rank1_update_sub_unchecked_mt(&vec1, &vec2);
            }

            pivot = (pivot.0 + 1, pivot.1 + 1);
        }

        (self, det_scale, pivots)
    }

    /// Computes the row echelon form of a matrix.
    /// As this is REF the leading entries will not be normalised.
    pub fn row_echelon(self) -> Matrix<Complex64> {
        self.tracked_row_echelon().0
    }

    /// Computes the reduced row echelon form of a matrix
    pub fn reduced_row_echelon(self) -> Matrix<Complex64> {
        let (mut ref_form, _, pivots) = self.tracked_row_echelon();
        let cols = ref_form.cols;

        // Go backwards through the pivots: normalise row and eliminate above
        for &pivot in pivots.iter().rev() {
            let val = ref_form[pivot];

            // Normalise row
            unsafe {
                ref_form
                    .slice_unchecked_mut(pivot.0..pivot.0 + 1, pivot.1..cols)
                    .iter_mut()
                    .for_each(|x| *x /= val);
            }

            // Eliminate above
            let row = unsafe { ref_form.slice_unchecked(pivot.0..pivot.0 + 1, pivot.1..cols) }
                .iter()
                .copied()
                .collect::<Vec<_>>();
            let col = unsafe { ref_form.slice_unchecked(0..pivot.0, pivot.1..pivot.1 + 1) }
                .iter()
                .copied()
                .collect::<Vec<_>>();

            unsafe {
                ref_form
                    .slice_unchecked_mut(0..pivot.0, pivot.1..cols)
                    .rank1_update_sub_unchecked_mt(&col, &row);
            }
        }

        ref_form
    }
}

#[cfg(test)]
mod row_echelon_tests {
    use float_cmp::approx_eq;
    use num::complex::Complex64;
    use tensor_math::definitions::matrix::Matrix;
    use tensor_math::utilities::matrix::identity;

    #[test]
    fn empty_matrix_tests() {
        let matrices = vec![
            Matrix::<f64>::new(0, 0, vec![]).unwrap(),
            Matrix::<f64>::new(0, 5, vec![]).unwrap(),
            Matrix::<f64>::new(5, 0, vec![]).unwrap(),
        ];

        for m in matrices {
            assert!(m.is_row_echelon());
            assert!(m.is_reduced_row_echelon());
            assert_eq!(m.clone().row_echelon(), m);
            assert_eq!(m.clone().reduced_row_echelon(), m);

            let mc = m.into_complex();
            assert!(mc.is_row_echelon());
            assert!(mc.is_reduced_row_echelon());
            assert_eq!(mc.clone().row_echelon(), mc);
            assert_eq!(mc.clone().reduced_row_echelon(), mc);
        }
    }

    #[test]
    fn f64_row_echelon_predicates_cover_skipped_columns_and_zero_rows() {
        let valid = Matrix::<f64>::new(
            4,
            5,
            vec![
                0.0, 2.0, 3.0, 0.0, 1.0, 0.0, 0.0, 0.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        assert!(valid.is_row_echelon());

        let zero_row_before_nonzero =
            Matrix::<f64>::new(3, 3, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
        assert!(!zero_row_before_nonzero.is_row_echelon());

        let repeated_pivot_column =
            Matrix::<f64>::new(2, 3, vec![1.0, 2.0, 0.0, 3.0, 4.0, 5.0]).unwrap();
        assert!(!repeated_pivot_column.is_row_echelon());
    }

    #[test]
    fn c64_row_echelon_predicate_handles_valid_skipped_columns() {
        let complex = Matrix::<Complex64>::new(
            3,
            4,
            vec![
                Complex64::ZERO,
                Complex64::new(1.0, 2.0),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::new(3.0, 1.0),
                Complex64::new(4.0, -2.0),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
            ],
        )
        .unwrap();
        assert!(complex.is_row_echelon());
    }

    #[test]
    fn f64_reduced_row_echelon_predicate_checks_pivot_columns() {
        let valid = Matrix::<f64>::new(
            3,
            5,
            vec![
                0.0, 1.0, 2.0, 0.0, 3.0, //
                0.0, 0.0, 0.0, 1.0, 4.0, //
                0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        assert!(valid.is_reduced_row_echelon());

        let nonzero_above_pivot = Matrix::<f64>::new(2, 2, vec![1.0, 2.0, 0.0, 1.0]).unwrap();
        assert!(!nonzero_above_pivot.is_reduced_row_echelon());

        let nonzero_left_of_pivot =
            Matrix::<f64>::new(2, 3, vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]).unwrap();
        assert!(!nonzero_left_of_pivot.is_reduced_row_echelon());
    }

    #[test]
    fn f64_reduced_row_echelon_predicate_rejects_non_unit_pivots() {
        let non_unit_pivot = Matrix::<f64>::new(1, 1, vec![2.0]).unwrap();
        assert!(!non_unit_pivot.is_reduced_row_echelon());
    }

    #[test]
    fn c64_reduced_row_echelon_predicate_rejects_non_unit_pivots() {
        let complex_non_unit_pivot =
            Matrix::<Complex64>::new(1, 1, vec![Complex64::new(0.0, 2.0)]).unwrap();
        assert!(!complex_non_unit_pivot.is_reduced_row_echelon());
    }

    #[test]
    fn f64_reduced_row_echelon_normalizes_single_row_inputs() {
        let one_row = Matrix::<f64>::new(1, 3, vec![0.0, 2.0, 4.0]).unwrap();
        assert!(one_row.clone().row_echelon().is_row_echelon());
        assert_eq!(
            one_row.reduced_row_echelon(),
            Matrix::new(1, 3, vec![0.0, 1.0, 2.0]).unwrap()
        );
    }

    #[test]
    fn f64_echelon_predicates_cover_single_column_inputs() {
        let one_column = Matrix::<f64>::new(3, 1, vec![0.0, 2.0, 4.0]).unwrap();
        assert!(!one_column.is_row_echelon());
        assert!(!one_column.is_reduced_row_echelon());

        let echelon_column = Matrix::<f64>::new(3, 1, vec![1.0, 0.0, 0.0]).unwrap();
        assert!(echelon_column.is_row_echelon());
        assert!(echelon_column.is_reduced_row_echelon());
    }

    #[test]
    fn c64_echelon_predicates_cover_rank_deficient_inputs() {
        let rank_deficient = Matrix::<Complex64>::new(
            3,
            4,
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(6.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .unwrap();
        assert!(!rank_deficient.is_row_echelon());
        assert!(!rank_deficient.is_reduced_row_echelon());
    }

    #[test]
    fn f64_ref_test() {
        let input = Matrix::<f64>::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let ref_form = input.clone().row_echelon();
        assert!(ref_form.is_row_echelon());
        assert!(approx_eq!(
            Matrix<f64>,
            ref_form.reduced_row_echelon(),
            input.reduced_row_echelon()
        ));
    }

    #[test]
    fn c64_ref_test() {
        let input = Matrix::<Complex64>::new(
            3,
            3,
            vec![
                Complex64::new(1.0, 2.0),
                Complex64::new(-4.0, 0.0),
                Complex64::new(3.0, 1.0),
                Complex64::new(1.0, 2.0),
                Complex64::new(-4.0, 0.0),
                Complex64::new(3.0, 1.0),
                Complex64::new(2.0, -5.0),
                Complex64::new(0.1, -3.5),
                Complex64::new(0.0, 1.0),
            ],
        )
        .unwrap();
        let ref_form = input.clone().row_echelon();
        assert!(ref_form.is_row_echelon());
        assert!(approx_eq!(
            Matrix<Complex64>,
            ref_form.reduced_row_echelon(),
            input.reduced_row_echelon(),
            epsilon = 1e-10
        ));
    }

    #[test]
    fn f64_is_ref_test() {
        let not_ref = Matrix::<f64>::new(
            3,
            5,
            vec![
                0.5, -5.0, 1.2, -3.4, 1.2, 0.2, -2.0, 0.0, 0.2, 1.7, 0.0, -1.0, 0.0, -3.6, 1.1,
            ],
        )
        .unwrap();
        assert!(!not_ref.is_row_echelon());

        let ref_form = Matrix::<f64>::new(
            3,
            4,
            vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        assert!(ref_form.is_row_echelon());

        let zero_row_before_nonzero =
            Matrix::<f64>::new(3, 3, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
        assert!(!zero_row_before_nonzero.is_row_echelon());
    }

    #[test]
    fn c64_is_ref_test() {
        let not_ref = Matrix::<Complex64>::new(
            4,
            3,
            vec![
                Complex64::new(1.0, -2.0),
                Complex64::ZERO,
                Complex64::new(1.8, -3.1),
                Complex64::new(0.0, -0.0),
                Complex64::ZERO,
                Complex64::new(1.9, -2.1),
                Complex64::new(4.2, 3.0),
                Complex64::ZERO,
                Complex64::new(9.0, -2.2),
                Complex64::new(-1.5, 2.5),
                Complex64::ZERO,
                Complex64::new(8.0, -8.6),
            ],
        )
        .unwrap();
        assert!(!not_ref.is_row_echelon());

        let ref_form = Matrix::<Complex64>::new(
            5,
            3,
            vec![
                Complex64::new(1.0, 2.0),
                Complex64::new(-4.0, 0.0),
                Complex64::new(3.0, 1.0),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::new(3.0, 1.0),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
            ],
        )
        .unwrap();
        assert!(ref_form.is_row_echelon());

        let zero_row_before_nonzero = Matrix::<Complex64>::new(
            3,
            3,
            vec![
                Complex64::ONE,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ONE,
            ],
        )
        .unwrap();
        assert!(!zero_row_before_nonzero.is_row_echelon());
    }

    #[test]
    fn f64_is_rref_test() {
        let valid =
            Matrix::<f64>::new(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        assert!(valid.is_reduced_row_echelon());

        let nonzero_above_pivot = Matrix::<f64>::new(2, 2, vec![1.0, 2.0, 0.0, 1.0]).unwrap();
        assert!(!nonzero_above_pivot.is_reduced_row_echelon());

        let non_unit_pivot = Matrix::<f64>::new(1, 1, vec![2.0]).unwrap();
        assert!(!non_unit_pivot.is_reduced_row_echelon());

        let all_zero_bottom = Matrix::<f64>::new(
            4,
            5,
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        assert!(all_zero_bottom.is_reduced_row_echelon());

        let left_not_zero = Matrix::<f64>::new(
            3,
            4,
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        .unwrap();
        assert!(!left_not_zero.is_reduced_row_echelon());
    }

    #[test]
    fn c64_is_rref_test() {
        let valid = Matrix::<Complex64>::new(
            3,
            3,
            vec![
                Complex64::ONE,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ONE,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
            ],
        )
        .unwrap();
        assert!(valid.is_reduced_row_echelon());

        let non_unit_pivot =
            Matrix::<Complex64>::new(1, 1, vec![Complex64::new(0.0, 2.0)]).unwrap();
        assert!(!non_unit_pivot.is_reduced_row_echelon());

        let all_zero_bottom = Matrix::<Complex64>::new(
            4,
            5,
            vec![
                Complex64::ONE,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ONE,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ONE,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
            ],
        )
        .unwrap();
        assert!(all_zero_bottom.is_reduced_row_echelon());

        let left_not_zero = Matrix::<Complex64>::new(
            3,
            4,
            vec![
                Complex64::ONE,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ONE,
                Complex64::ONE,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ONE,
                Complex64::ZERO,
            ],
        )
        .unwrap();
        assert!(!left_not_zero.is_reduced_row_echelon());
    }

    #[test]
    fn f64_rref_test() {
        let input = Matrix::<f64>::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(approx_eq!(
            Matrix<f64>,
            input.reduced_row_echelon(),
            identity(2),
            epsilon = 1e-10
        ));

        let one_row = Matrix::<f64>::new(1, 3, vec![0.0, 2.0, 4.0]).unwrap();
        assert_eq!(
            one_row.reduced_row_echelon(),
            Matrix::new(1, 3, vec![0.0, 1.0, 2.0]).unwrap()
        );
    }

    #[test]
    fn c64_rref_test() {
        let input = Matrix::<Complex64>::new(
            3,
            3,
            vec![
                Complex64::new(1.0, 2.0),
                Complex64::new(-4.0, 0.0),
                Complex64::new(3.0, 1.0),
                Complex64::new(1.0, 2.0),
                Complex64::new(-4.0, 0.0),
                Complex64::new(3.0, 1.0),
                Complex64::new(2.0, -5.0),
                Complex64::new(0.1, -3.5),
                Complex64::new(0.0, 1.0),
            ],
        )
        .unwrap();
        let expected = Matrix::<Complex64>::new(
            3,
            3,
            vec![
                Complex64::ONE,
                Complex64::ZERO,
                Complex64 {
                    re: 0.2678687248670384,
                    im: -0.01050719937735134,
                },
                Complex64::ZERO,
                Complex64::ONE,
                Complex64 {
                    re: -0.6777792190945647,
                    im: -0.1186924374108186,
                },
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::ZERO,
            ],
        )
        .unwrap();
        assert!(approx_eq!(
            Matrix<Complex64>,
            input.reduced_row_echelon(),
            expected,
            epsilon = 1e-15
        ));
    }
}
