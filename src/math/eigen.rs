use num::complex::{Complex64, ComplexFloat};
use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::math::polynomials::solve_quadratic;
use crate::utilities::matrix::identity;

impl Matrix<Complex64> {
    /// Returns the eigendecomposition for a matrix in the form `(values, vectors)`
    /// where `values` is a vector of eigenvalues and `vectors` is a matrix where
    /// the columns are the eigenvectors the matrix.
    pub fn eigendecompose(&self) -> Result<(Vec<Complex64>, Matrix<Complex64>), TensorErrors> {
        if !self.is_square() {
            return Err(TensorErrors::NonSquareMatrix);
        }

        if self.rows == 1 {
            return Ok((self.elements.clone(), self.clone()));
        }

        let (mut h, mut q) = self.upper_hessenberg()?;
        let ord = self.rows;
        let mut diff = Complex64::ONE;
        let mut prev_ref = h[(ord - 1, ord - 1)];
        let mut iters = 0;

        while diff.abs() > 1e-15 && iters < 1000 {
            iters += 1;

            // Calculate the Wilkinson shift
            let bottom_right_mat = h.slice(ord - 2..ord, ord - 2..ord)?;
            let bottom_right = h[(ord - 1, ord - 1)];
            
            let roots = solve_quadratic(&[
                bottom_right_mat.det()?,
                -bottom_right_mat.trace()?,
                Complex64::ONE,
            ])?;

            let dist0 = (roots[0] - bottom_right).abs();
            let dist1 = (roots[1] - bottom_right).abs();

            let mut ws = roots[0];

            if dist1 < dist0 {
                ws = roots[1]
            }
            
            let shifted = h.clone() - identity(ord)? * ws;
            let (qs, rs) = shifted.householder();

            h = rs.contract_mul_mt(&qs)? + identity(ord)? * ws;
            q = q.contract_mul_mt(&qs)?;

            diff = h[(ord - 1, ord - 1)] - prev_ref;
            prev_ref = h[(ord - 1, ord - 1)];
        }

        let mut eigenvalues = Vec::with_capacity(ord);
        for i in 0..ord {
            eigenvalues.push(h[(i, i)]);
        }

        Ok((eigenvalues, q))
    }
}