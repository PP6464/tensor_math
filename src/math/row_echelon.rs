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

            if (current_pivot_col as i32) <= prev_pivot_col {
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
                        if pivot.0 + 1 == self.rows {
                            return true;
                        }

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

        while pivot.0 < self.rows - 1 && pivot.1 < self.cols {
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
            if row != pivot.0 {
                unsafe {
                    self.swap_rows_unchecked(pivot.0, row);
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

            if current_pivot_col <= prev_pivot_col as usize {
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
                    // If this is the last row then return true because we are done checking everything else
                    if pivot.0 == self.rows - 1 {
                        return true;
                    }

                    // This row is all 0 otherwise, so just check everything below is all 0 as well
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
                        if pivot.0 + 1 == self.rows {
                            return true;
                        }

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

        while pivot.0 < self.rows - 1 && pivot.1 < self.cols {
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
            if row != pivot.0 {
                unsafe {
                    self.swap_rows_unchecked(pivot.0, row);
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
