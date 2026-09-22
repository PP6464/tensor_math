use crate::definitions::matrix::Matrix;
use num::complex::Complex64;

impl Matrix<f64> {
    /// Gives the rank of the transformation represented by this matrix.
    pub fn transformation_rank(self) -> usize {
        self.tracked_row_echelon().2.len()
    }
}

impl Matrix<Complex64> {
    /// Gives the rank of the transformation represented by this matrix.
    pub fn transformation_rank(self) -> usize {
        self.tracked_row_echelon().2.len()
    }
}
