use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::math::polynomials::solve_quadratic;
use crate::utilities::matrix::identity;
use num::complex::{Complex64, ComplexFloat};

impl Matrix<Complex64> {
    /// Returns the eigendecomposition for a matrix in the form `(values, vectors)`
    /// where `values` is a vector of eigenvalues and `vectors` is a matrix where
    /// the columns are the eigenvectors the matrix.
    /// This fails if the matrix is not square or if the process could not converge on eigenvalues.
    pub fn eigendecompose(&self) -> Result<(Vec<Complex64>, Matrix<Complex64>), TensorErrors> {
        if !self.is_square() {
            return Err(TensorErrors::NonSquareMatrix);
        }

        if self.rows <= 1 {
            return Ok((self.elements.clone(), self.clone()));
        }

        let (mut h, mut q) = self.upper_hessenberg()?;
        let ord = self.rows;
        let mut m = ord;
        let mut iters_since_deflation = 0;

        let mut eigenvalues = Vec::with_capacity(self.rows);
        let buf = eigenvalues.spare_capacity_mut();

        while m > 1 {
            // Check if we can deflate
            if h[(m - 1, m - 2)].abs() < 1e-15 {
                buf[m - 1].write(h[(m - 1, m - 1)]);
                m -= 1;
                iters_since_deflation = 0;
                continue;
            }

            // Fail if we have not converged
            if iters_since_deflation >= 30 {
                return Err(TensorErrors::EigenDecompositionDidNotConverge);
            }

            iters_since_deflation += 1;

            // Calculate the Wilkinson shift
            let bottom_right_mat = h.slice(m - 2..m, m - 2..m)?;
            let bottom_right = h[(m - 1, m - 1)];

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

            let shifted = h.slice(0..m, 0..m)? - identity(m) * ws;
            let (qs, rs) = shifted.householder();

            h
                .slice_mut(0..m, 0..m)?
                .set_all(&(rs.contract_mul_mt(&qs)? + identity(m) * ws))?;
            let slice_copy = q.slice(0..ord, 0..m)?;
            q
                .slice_mut(0..ord, 0..m)?
                .set_all(&slice_copy.mat_mul_mt(&qs)?)?;
        }

        // At this point we must have converged on all eigenvalues
        buf[0].write(h[(0, 0)]);

        unsafe {
            eigenvalues.set_len(ord);
        }

        Ok((eigenvalues, q))
    }

    /// Returns the eigenvalues for a matrix.
    /// This fails if the matrix is not square or if the process could not converge on eigenvalues.
    pub fn eigenvalues(&self) -> Result<Vec<Complex64>, TensorErrors> {
        if !self.is_square() {
            return Err(TensorErrors::NonSquareMatrix);
        }

        if self.rows <= 1 {
            return Ok(self.elements.clone());
        }

        let (mut h, _) = self.upper_hessenberg_h()?;
        let ord = self.rows;
        let mut m = ord;
        let mut iters_since_deflation = 0;

        let mut eigenvalues = Vec::with_capacity(self.rows);
        let buf = eigenvalues.spare_capacity_mut();

        while m > 1 {
            // Check if we can deflate
            if h[(m - 1, m - 2)].abs() < 1e-15 {
                buf[m - 1].write(h[(m - 1, m - 1)]);
                m -= 1;
                iters_since_deflation = 0;
                continue;
            }

            // Fail if we have not converged
            if iters_since_deflation >= 30 {
                return Err(TensorErrors::EigenDecompositionDidNotConverge);
            }

            iters_since_deflation += 1;

            // Calculate the Wilkinson shift
            let bottom_right_mat = h.slice(m - 2..m, m - 2..m)?;
            let bottom_right = h[(m - 1, m - 1)];

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

            let shifted = h.slice(0..m, 0..m)? - identity(m) * ws;
            let (qs, rs) = shifted.householder();

            h
                .slice_mut(0..m, 0..m)?
                .set_all(&(rs.contract_mul_mt(&qs)? + identity(m) * ws))?;
        }

        // At this point we must have converged on all eigenvalues
        buf[0].write(h[(0, 0)]);

        unsafe {
            eigenvalues.set_len(ord);
        }

        Ok(eigenvalues)
    }
}

impl Matrix<f64> {
    /// Returns the eigendecomposition for a matrix in the form `(values, vectors)`
    /// where `values` is a vector of eigenvalues and `vectors` is a matrix where
    /// the columns are the eigenvectors the matrix. The entries of both will be `Complex64`.
    /// This fails if the matrix is not square.
    pub fn eigendecompose(&self) -> Result<(Vec<Complex64>, Matrix<Complex64>), TensorErrors> {
        self.par_map_refs(|x| Complex64 { re: x.clone(), im: 0.0 }).eigendecompose()
    }

    /// Returns the eigenvalues for a matrix.
    /// This fails if the matrix is not square or if the process could not converge on eigenvalues.
    pub fn eigenvalues(&self) -> Result<Vec<Complex64>, TensorErrors> {
        self.par_map_refs(|x| Complex64 { re: x.clone(), im: 0.0 }).eigenvalues()
    }
}
