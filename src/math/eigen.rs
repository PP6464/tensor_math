use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::math::polynomials::solve_quadratic;
use num::complex::{Complex64, ComplexFloat};

impl Matrix<Complex64> {
    /// Returns the eigendecomposition for a matrix in the form `(values, vectors)`
    /// where `values` is a vector of eigenvalues and `vectors` is a matrix where
    /// the columns are the eigenvectors the matrix.
    /// This fails if the matrix is not square or if the process could not converge on eigenvalues.
    pub fn eigendecompose(self) -> Result<(Vec<Complex64>, Matrix<Complex64>), TensorErrors> {
        if !self.is_square() {
            return Err(TensorErrors::NonSquareMatrix);
        }

        if self.rows <= 1 {
            return Ok((self.elements.clone(), self.clone()));
        }

        let ord = self.rows;
        let (mut h, mut q) = self.upper_hessenberg()?;
        let mut m = ord;
        let mut iters_since_deflation = 0;

        let mut eigenvalues = Vec::with_capacity(ord);
        let buf = eigenvalues.spare_capacity_mut();

        unsafe {
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
                let bottom_right_mat =
                    h.slice_unchecked(m - 2..m, m - 2..m).clone_into_matrix();
                let bottom_right = h[(m - 1, m - 1)];

                let trace = bottom_right_mat.trace()?;

                let roots = solve_quadratic(&[bottom_right_mat.det()?, -trace, Complex64::ONE])?;

                let dist0 = (roots[0] - bottom_right).abs();
                let dist1 = (roots[1] - bottom_right).abs();

                let mut ws = roots[0];

                if dist1 < dist0 {
                    ws = roots[1]
                }

                // Shift the matrix and perform a QR decomposition
                let mut shifted = h.slice_unchecked(0..m, 0..m).clone_into_matrix();
                for i in 0..m {
                    shifted[(i, i)] -= ws;
                }
                let (mut rs, reflectors) = shifted.householder_r();
                reflectors.apply_right_unchecked(&mut rs);
                for i in 0..m {
                    rs[(i, i)] += ws;
                }

                h.slice_unchecked_mut(0..m, 0..m).set_all(&rs)?;
                let mut slice_copy = q.slice_unchecked(0..ord, 0..m).clone_into_matrix();
                reflectors.apply_right_unchecked(&mut slice_copy);
                q.slice_unchecked_mut(0..ord, 0..m).set_all(&slice_copy)?;
            }
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
    pub fn eigenvalues(self) -> Result<Vec<Complex64>, TensorErrors> {
        if !self.is_square() {
            return Err(TensorErrors::NonSquareMatrix);
        }

        if self.rows <= 1 {
            return Ok(self.elements.clone());
        }

        let ord = self.rows;
        let (mut h, _) = self.upper_hessenberg_h()?;
        let mut m = ord;
        let mut iters_since_deflation = 0;

        let mut eigenvalues = Vec::with_capacity(ord);
        let buf = eigenvalues.spare_capacity_mut();

        unsafe {
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
                let bottom_right_mat = h.slice_unchecked(m - 2..m, m - 2..m).clone_into_matrix();
                let bottom_right = h[(m - 1, m - 1)];

                let trace = bottom_right_mat.trace()?;
                let roots = solve_quadratic(&[bottom_right_mat.det()?, -trace, Complex64::ONE])?;

                let dist0 = (roots[0] - bottom_right).abs();
                let dist1 = (roots[1] - bottom_right).abs();

                let mut ws = roots[0];

                if dist1 < dist0 {
                    ws = roots[1]
                }

                let mut shifted = h.slice_unchecked(0..m, 0..m).clone_into_matrix();
                for i in 0..m {
                    shifted[(i, i)] -= ws;
                }
                let (mut rs, reflectors) = shifted.householder_r();
                reflectors.apply_right_unchecked(&mut rs);
                for i in 0..m {
                    rs[(i, i)] += ws;
                }

                h.slice_unchecked_mut(0..m, 0..m).set_all(&rs)?;
            }
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
    pub fn eigendecompose(self) -> Result<(Vec<Complex64>, Matrix<Complex64>), TensorErrors> {
        self.par_map_refs(|x| Complex64 {
            re: *x,
            im: 0.0,
        })
        .eigendecompose()
    }

    /// Returns the eigenvalues for a matrix.
    /// This fails if the matrix is not square or if the process could not converge on eigenvalues.
    pub fn eigenvalues(self) -> Result<Vec<Complex64>, TensorErrors> {
        self.par_map_refs(|x| Complex64 {
            re: *x,
            im: 0.0,
        })
        .eigenvalues()
    }
}
