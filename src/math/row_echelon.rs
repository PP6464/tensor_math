use float_cmp::approx_eq;
use num::complex::{Complex64, ComplexFloat};
use crate::definitions::matrix::Matrix;

impl Matrix<f64> {
    /// Gives whether the matrix is in row echelon form or not
    pub fn is_row_echelon(&self) -> bool {
        let mut all_zero_rows = false;
        let mut prev_pivot_col: i32 = -1;

        for i in 0..self.rows {
            let current_row = self.slice(i..i+1, 0..self.cols).unwrap();

            if all_zero_rows && !current_row.iter().all(|x| approx_eq!(f64, *x, 0.0)) {
                return false; // There is a row below a row of all 0 that is not itself all 0
            }

            if current_row.iter().all(|x| approx_eq!(f64, *x, 0.0)) {
                all_zero_rows = true;
                continue
            }

            let (current_pivot_col, _) = current_row.iter().enumerate().find(|(_, &x)| !approx_eq!(f64, x, 0.0)).unwrap();

            if (current_pivot_col as i32) <= prev_pivot_col {
                return false;
            }

            prev_pivot_col = current_pivot_col as i32;
        }

        true
    }

    /// Gives whether the matrix is in reduced row echelon form or not
    pub fn is_reduced_row_echelon(&self) -> bool {
        let mut pivot = (0, 0);

        while pivot.0 < self.rows && pivot.1 < self.cols {
            let pivot_val = self[pivot];

            // Note everything below must be 0
            if pivot.0 < self.rows - 1 {
                let below_slice = self.slice(pivot.0 + 1..self.rows, pivot.1..pivot.1 + 1).unwrap();

                if below_slice.iter().any(|x| !approx_eq!(f64, *x, 0.0)) {
                    return false;
                }
            }

            if approx_eq!(f64, pivot_val, 0.0) {
                // If this is the last element in the row then there are no suitable other pivots
                if pivot.1 == self.cols - 1 {
                    // If this is the last row then return true because we are done checking everything else
                    if pivot.0 == self.rows - 1 {
                        return true
                    }

                    // This row is all 0 otherwise, so just check everything below is all 0 as well
                    return self.slice(pivot.0 + 1..self.rows, 0..self.cols).unwrap().iter().all(|x| approx_eq!(f64, *x, 0.0));
                }

                // Check for all 0, otherwise just move to the first non-zero element
                let right_slice = self.slice(pivot.0..pivot.0 + 1, pivot.1 + 1..self.cols).unwrap();
                let option_pivot = right_slice.iter().enumerate().find(|(_, &x)| !approx_eq!(f64, x, 0.0));

                match option_pivot {
                    Some((index, _)) => {
                        pivot = (pivot.0, pivot.1 + 1 + index);
                        continue
                    }
                    None => {
                        // This row is all 0, check rows below
                        if pivot.0 + 1 == self.rows {
                            return true
                        }

                        return self.slice(pivot.0 + 1..self.rows, 0..self.cols).unwrap().iter().all(|x| approx_eq!(f64, *x, 0.0));
                    }
                }
            }

            // Note everything to the left should be 0
            if pivot.1 > 0 {
                let left_slice = self.slice(pivot.0..pivot.0 + 1, 0..pivot.1).unwrap();

                if left_slice.iter().any(|x| !approx_eq!(f64, *x, 0.0)) {
                    return false;
                }
            }

            // Similarly everything above must be 0
            if pivot.0 > 0 {
                let above_slice = self.slice(0..pivot.0, pivot.1..pivot.1 + 1);

                if above_slice.unwrap().iter().any(|x| !approx_eq!(f64, *x, 0.0)) {
                    return false;
                }
            }

            // Move pivot by 1 diagonally since as we have reached here this is a normal pivot
            pivot = (pivot.0 + 1, pivot.1 + 1);
        }

        true
    }

    /// Computes the REF form of a matrix.
    /// This does not require that the leading entries of the rows are normalised.
    pub(crate) fn tracked_row_echelon(&self) -> (Matrix<f64>, i32) {
        let mut res = self.clone();
        let mut det_scale = 1;
        let mut pivot = (0usize, 0usize);

        while pivot.0 < res.rows - 1 && pivot.1 < res.cols {
            // Identify if we can use this position as a pivot value
            let pivot_val = res[pivot];

            if approx_eq!(f64, pivot_val, 0.0) {
                // The pivot value is 0

                // Check if any of the other rows below have a non-zero value
                // at this pivot column and if so then use that row's value instead
                let slice_below = res.slice(pivot.0 + 1..res.rows, pivot.1..pivot.1 + 1).unwrap();

                let (index, max_abs) = slice_below.iter().enumerate().rev().max_by(|(_, &x), (_, &y)| { x.abs().total_cmp(&y.abs()) }).unwrap();

                if approx_eq!(f64, max_abs.abs(), 0.0) {
                    // There is no suitable pivot on this row
                    pivot = (pivot.0, pivot.1 + 1);
                    continue
                }

                // There is a suitable pivot on this column in another row, so need to swap those rows
                let chosen_copy = res.slice(index + pivot.0 + 1..index + pivot.0 + 2, pivot.1..res.cols).unwrap();
                let current_copy = res.slice(pivot.0..pivot.0 + 1, pivot.1..res.cols).unwrap();

                res.slice_mut(pivot.0..pivot.0 + 1, pivot.1..res.cols).unwrap().set_all(&chosen_copy).unwrap();
                res.slice_mut(index + pivot.0 + 1..index + pivot.0 + 2, pivot.1..res.cols).unwrap().set_all(&current_copy).unwrap();

                // Now multiply determinant scale factor by -1 because we swapped rows
                det_scale *= -1;

                // Now we can resume with normal Gauss-Jordan elimination
                continue
            } else {
                // Eliminate all rows below
                for i in pivot.0 + 1..res.rows {
                    let val_for_row = res[(i, pivot.1)];
                    let new_row = res.slice(i..i+1, pivot.1..res.cols).unwrap() - res.slice(pivot.0..pivot.0 + 1, pivot.1..res.cols).unwrap() * (val_for_row) / pivot_val;
                    res.slice_mut(i..i+1, pivot.1..res.cols).unwrap().set_all(&new_row).unwrap();
                }

                // We slide the pivot one down and to the right in the normal case
                pivot = (pivot.0 + 1, pivot.1 + 1);
            }
        }

        (res, det_scale)
    }

    /// Computes the row echelon form of a matrix.
    /// This does not require that the leading entries of the rows are normalised.
    pub fn row_echelon(&self) -> Matrix<f64> {
        self.tracked_row_echelon().0
    }

    /// Computes the reduced row echelon form of a matrix
    pub fn reduced_row_echelon(&self) -> Matrix<f64> {
        let mut res = self.clone();

        let mut pivot = (0usize, 0usize);

        while pivot.0 < res.rows && pivot.1 < res.cols {
            // Identify if we can use this position as a pivot value
            let pivot_val = res[pivot];

            if approx_eq!(f64, pivot_val, 0.0) {
                // The pivot value is 0

                // If this is the last row then there are no other rows to swap with,
                // so we must see directly where the first usable pivot is in the last row.
                // If such a pivot does not exist then we stop here.
                if pivot.0 == res.rows - 1 {
                    if pivot.1 == res.cols - 1 {
                        // There is nothing to check
                        break
                    }

                    let rest_of_row = res.slice(pivot.0..res.rows, pivot.1 + 1..res.cols).unwrap();

                    let (index, max_abs) = rest_of_row.iter().enumerate().rev().max_by(|(_, &x), (_, &y)| { x.abs().total_cmp(&y.abs()) }).unwrap();

                    if !approx_eq!(f64, *max_abs, 0.0) {
                        // There is a suitable pivot so change the position of the pivot to it
                        pivot = (pivot.0, pivot.1 + 1 + index);
                        continue
                    }

                    // Otherwise we are done here
                    break
                }

                // Check if any of the other rows below have a non-zero value
                // at this pivot column and if so then use that row's value instead
                let slice_below = res.slice(pivot.0 + 1..res.rows, pivot.1..pivot.1 + 1).unwrap();

                let (index, max_abs) = slice_below.iter().enumerate().max_by(|(_, &x), (_, &y)| { x.abs().total_cmp(&y.abs()) }).unwrap();

                if approx_eq!(f64, max_abs.abs(), 0.0) {
                    // There is no suitable pivot on this row
                    pivot = (pivot.0, pivot.1 + 1);
                    continue
                }

                // There is a suitable pivot on this column in another row, so need to swap those rows
                let chosen_copy = res.slice(index + pivot.0 + 1..index + pivot.0 + 2, pivot.1..res.cols).unwrap();
                let current_copy = res.slice(pivot.0..pivot.0 + 1, pivot.1..res.cols).unwrap();

                res.slice_mut(pivot.0..pivot.0 + 1, pivot.1..res.cols).unwrap().set_all(&chosen_copy).unwrap();
                res.slice_mut(index + pivot.0 + 1..index + pivot.0 + 2, pivot.1..res.cols).unwrap().set_all(&current_copy).unwrap();

                // Now we can resume with normal Gauss-Jordan elimination
                continue
            } else {
                // Normalise the row
                let norm_row = res.slice(pivot.0 .. pivot.0 + 1, pivot.1 .. res.cols).unwrap() / pivot_val;
                res.slice_mut(pivot.0..pivot.0 + 1, pivot.1..res.cols).unwrap().set_all(&norm_row).unwrap();
            }

            // Eliminate all other rows
            for i in 0..res.rows {
                if i == pivot.0 {
                    continue
                }

                let val_for_row = res[(i, pivot.1)];
                let new_row = res.slice(i..i+1, pivot.1..res.cols).unwrap() - res.slice(pivot.0..pivot.0 + 1, pivot.1..res.cols).unwrap() * val_for_row;
                res.slice_mut(i..i + 1, pivot.1..res.cols).unwrap().set_all(&new_row).unwrap();
            }

            // We slide the pivot one down and to the right as we are in the normal case
            pivot = (pivot.0 + 1, pivot.1 + 1);
        }

        res
    }
}

impl Matrix<Complex64> {
    /// Gives whether the matrix is in row echelon form or not
    pub fn is_row_echelon(&self) -> bool {
        let mut all_zero_rows = false;
        let mut prev_pivot_col: i32 = -1;

        for i in 0..self.rows {
            let current_row = self.slice(i..i+1, 0..self.cols).unwrap();

            if all_zero_rows && !current_row.iter().all(|x| approx_eq!(f64, (*x).abs(), 0.0)) {
                return false; // There is a row below a row of all 0 that is not itself all 0
            }

            if current_row.iter().all(|x| approx_eq!(f64, (*x).abs(), 0.0)) {
                all_zero_rows = true;
                continue
            }

            let (current_pivot_col, _) = current_row.iter().enumerate().find(|(_, &x)| !approx_eq!(f64, x.abs(), 0.0)).unwrap();

            if (current_pivot_col as i32) <= prev_pivot_col {
                return false;
            }

            prev_pivot_col = current_pivot_col as i32;
        }

        true
    }

    /// Gives whether the matrix is in reduced row echelon form or not
    pub fn is_reduced_row_echelon(&self) -> bool {
        let mut pivot = (0, 0);

        while pivot.0 < self.rows && pivot.1 < self.cols {
            let pivot_val = self[pivot];

            // Note everything below must be 0
            if pivot.0 < self.rows - 1 {
                let below_slice = self.slice(pivot.0 + 1..self.rows, pivot.1..pivot.1 + 1).unwrap();

                if below_slice.iter().any(|x| !approx_eq!(f64, (*x).abs(), 0.0)) {
                    return false;
                }
            }

            if approx_eq!(f64, pivot_val.abs(), 0.0) {
                // If this is the last element in the row then there are no suitable other pivots
                if pivot.1 == self.cols - 1 {
                    // If this is the last row then return true because we are done checking everything else
                    if pivot.0 == self.rows - 1 {
                        return true
                    }

                    // This row is all 0 otherwise, so just check everything below is all 0 as well
                    return self.slice(pivot.0 + 1..self.rows, 0..self.cols).unwrap().iter().all(|x| approx_eq!(f64, (*x).abs(), 0.0));
                }

                // Check for all 0, otherwise just move to the first non-zero element
                let right_slice = self.slice(pivot.0..pivot.0 + 1, pivot.1 + 1..self.cols).unwrap();
                let option_pivot = right_slice.iter().enumerate().find(|(_, &x)| !approx_eq!(f64, x.abs(), 0.0));

                match option_pivot {
                    Some((index, _)) => {
                        pivot = (pivot.0, pivot.1 + 1 + index);
                        continue
                    }
                    None => {
                        // This row is all 0, check rows below
                        if pivot.0 + 1 == self.rows {
                            return true
                        }

                        return self.slice(pivot.0 + 1..self.rows, 0..self.cols).unwrap().iter().all(|x| approx_eq!(f64, (*x).abs(), 0.0));
                    }
                }
            }

            // Note everything to the left should be 0
            if pivot.1 > 0 {
                let left_slice = self.slice(pivot.0..pivot.0 + 1, 0..pivot.1).unwrap();

                if left_slice.iter().any(|x| !approx_eq!(f64, (*x).abs(), 0.0)) {
                    return false;
                }
            }

            // Similarly everything above must be 0
            if pivot.0 > 0 {
                let above_slice = self.slice(0..pivot.0, pivot.1..pivot.1 + 1).unwrap();

                if above_slice.iter().any(|x| !approx_eq!(f64, (*x).abs(), 0.0)) {
                    return false;
                }
            }

            // Move pivot by 1 diagonally since as we have reached here this is a normal pivot
            pivot = (pivot.0 + 1, pivot.1 + 1);
        }

        true
    }

    /// Computes the row echelon form of a matrix.
    /// This does not require that the leading entries are normalised.
    /// This returns a tuple of the result and 1 if an even number of rows
    /// have been swapped or -1 if an odd number of rows have been swapped
    pub(crate) fn tracked_row_echelon(&self) -> (Matrix<Complex64>, i32) {
        let mut res = self.clone();
        let mut det_scale = 1;

        let mut pivot = (0usize, 0usize);

        while pivot.0 < res.rows - 1 && pivot.1 < res.cols {
            // Identify if we can use this position as a pivot value
            let pivot_val = res[pivot];

            if approx_eq!(f64, pivot_val.abs(), 0.0) {
                // The pivot value is 0

                // Check if any of the other rows below have a non-zero value
                // at this pivot column and if so then use that row's value instead
                let slice_below = res.slice(pivot.0 + 1..res.rows, pivot.1..pivot.1 + 1).unwrap();

                let (index, max_abs) = slice_below.iter().enumerate().rev().max_by(|(_, &x), (_, &y)| { x.abs().total_cmp(&y.abs()) }).unwrap();

                if approx_eq!(f64, max_abs.abs(), 0.0) {
                    // There is no suitable pivot on this row
                    pivot = (pivot.0, pivot.1 + 1);
                    continue
                }

                // There is a suitable pivot on this column in another row, so need to swap those rows
                let chosen_copy = res.slice(index + pivot.0 + 1..index + pivot.0 + 2, pivot.1..res.cols).unwrap();
                let current_copy = res.slice(pivot.0..pivot.0 + 1, pivot.1..res.cols).unwrap();

                res.slice_mut(pivot.0..pivot.0 + 1, pivot.1..res.cols).unwrap().set_all(&chosen_copy).unwrap();
                res.slice_mut(index + pivot.0 + 1..index + pivot.0 + 2, pivot.1..res.cols).unwrap().set_all(&current_copy).unwrap();

                // Multiply the determinant scale factor by -1 because we swapped a row
                det_scale *= -1;

                // Now we can resume with normal Gauss-Jordan elimination
                continue
            } else {
                // Eliminate all rows below
                for i in pivot.0 + 1..res.rows {
                    let val_for_row = res[(i, pivot.1)];
                    let new_row = res.slice(i..i+1, pivot.1..res.cols).unwrap() - res.slice(pivot.0..pivot.0 + 1, pivot.1..res.cols).unwrap() * (val_for_row) / pivot_val;
                    res.slice_mut(i..i + 1, pivot.1..res.cols).unwrap().set_all(&new_row).unwrap();
                }

                // We slide the pivot one down and to the right in the normal case
                pivot = (pivot.0 + 1, pivot.1 + 1);
            }
        }

        (res, det_scale)
    }

    /// Computes the row echelon form of a matrix.
    /// This does not require that the leading entries are normalised.
    pub fn row_echelon(&self) -> Matrix<Complex64> {
        self.tracked_row_echelon().0
    }

    /// Computes the reduced row echelon form of a matrix
    pub fn reduced_row_echelon(&self) -> Matrix<Complex64> {
        let mut res = self.clone();

        let mut pivot = (0usize, 0usize);

        while pivot.0 < res.rows && pivot.1 < res.cols {
            // Identify if we can use this position as a pivot value
            let pivot_val = res[pivot];

            if approx_eq!(f64, pivot_val.abs(), 0.0) {
                // The pivot value is 0

                // If this is the last row then there are no other rows to swap with,
                // so we must see directly where the first usable pivot is in the last row.
                // If such a pivot does not exist then we stop here.
                if pivot.0 == res.rows - 1 {
                    if pivot.1 == res.cols - 1 {
                        // There is nothing to check
                        break
                    }

                    let rest_of_row = res.slice(pivot.0..res.rows, pivot.1 + 1..res.cols).unwrap();

                    let (index, max_abs) = rest_of_row.iter().enumerate().rev().max_by(|(_, &x), (_, &y)| { x.abs().total_cmp(&y.abs()) }).unwrap();

                    if !approx_eq!(f64, max_abs.abs(), 0.0) {
                        // There is a suitable pivot so change the position of the pivot to it
                        pivot = (pivot.0, pivot.1 + 1 + index);
                        continue
                    }

                    // Otherwise we are done here
                    break
                }

                // Check if any of the other rows below have a non-zero value
                // at this pivot column and if so then use that row's value instead
                let slice_below = res.slice(pivot.0 + 1..res.rows, pivot.1..pivot.1 + 1).unwrap();

                let (index, max_abs) = slice_below.iter().enumerate().max_by(|(_, &x), (_, &y)| { x.abs().total_cmp(&y.abs()) }).unwrap();

                if approx_eq!(f64, max_abs.abs(), 0.0) {
                    // There is no suitable pivot on this row
                    pivot = (pivot.0, pivot.1 + 1);
                    continue
                }

                // There is a suitable pivot on this column in another row, so need to swap those rows
                let chosen_copy = res.slice(index + pivot.0 + 1..index + pivot.0 + 2, pivot.1..res.cols).unwrap();
                let current_copy = res.slice(pivot.0..pivot.0 + 1, pivot.1..res.cols).unwrap();

                res.slice_mut(pivot.0..pivot.0 + 1, pivot.1..res.cols).unwrap().set_all(&chosen_copy).unwrap();
                res.slice_mut(index + pivot.0 + 1..index + pivot.0 + 2, pivot.1..res.cols).unwrap().set_all(&current_copy).unwrap();

                // Now we can resume with normal Gauss-Jordan elimination
                continue
            } else {
                // Normalise the row
                let norm_row = res.slice(pivot.0 .. pivot.0 + 1, pivot.1 .. res.cols).unwrap() / pivot_val;
                res.slice_mut(pivot.0 .. pivot.0 + 1, pivot.1 .. res.cols).unwrap().set_all(&norm_row).unwrap();
            }

            // Eliminate all other rows
            for i in 0..res.rows {
                if i == pivot.0 {
                    continue
                }

                let val_for_row = res[(i, pivot.1)];
                let new_row = res.slice(i..i+1, pivot.1..res.cols).unwrap() - res.slice(pivot.0..pivot.0 + 1, pivot.1..res.cols).unwrap() * val_for_row;
                res.slice_mut(i..i+1, pivot.1..res.cols).unwrap().set_all(&new_row).unwrap();
            }

            // We slide the pivot one down and to the right as we are in the normal case
            pivot = (pivot.0 + 1, pivot.1 + 1);
        }

        res
    }
}
