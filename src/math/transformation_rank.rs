use crate::definitions::matrix::Matrix;
use float_cmp::approx_eq;
use num::complex::{Complex64, ComplexFloat};

impl Matrix<f64> {
    /// Gives the rank of the transformation represented by this matrix.
    pub fn transformation_rank(&self) -> usize {
        let mut all_zero_rows = 0usize;
        let ref_form = self.row_echelon();
        let rows = self.rows;

        unsafe {
            for i in (0..rows).rev() {
                let row = ref_form.slice_unchecked(i..i + 1, 0..self.cols);

                if !row.iter().all(|x| approx_eq!(f64, *x, 0.0)) {
                    return rows - all_zero_rows;
                }

                all_zero_rows += 1;
            }
        }

        0
    }
}

impl Matrix<Complex64> {
    /// Gives the rank of the transformation represented by this matrix.
    pub fn transformation_rank(&self) -> usize {
        let mut all_zero_rows = 0usize;
        let ref_form = self.row_echelon();
        let rows = self.rows;

        unsafe {
            for i in (0..rows).rev() {
                let row = ref_form.slice_unchecked(i..i + 1, 0..self.cols);

                if !row.iter().all(|x| approx_eq!(f64, x.abs(), 0.0)) {
                    return rows - all_zero_rows;
                }

                all_zero_rows += 1;
            }
        }

        0
    }
}
