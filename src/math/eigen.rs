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
        self.par_map_refs(|x| Complex64 { re: *x, im: 0.0 })
            .eigendecompose()
    }

    /// Returns the eigenvalues for a matrix.
    /// This fails if the matrix is not square or if the process could not converge on eigenvalues.
    pub fn eigenvalues(self) -> Result<Vec<Complex64>, TensorErrors> {
        self.par_map_refs(|x| Complex64 { re: *x, im: 0.0 })
            .eigenvalues()
    }
}

#[cfg(test)]
mod eigen_tests {
    use float_cmp::approx_eq;
    use num::complex::{Complex64, ComplexFloat};
    use tensor_math::definitions::errors::TensorErrors;
    use tensor_math::definitions::matrix::Matrix;
    use tensor_math::utilities::matrix::eye;

    fn assert_eigenvalues_match(mut actual: Vec<Complex64>, expected: &[Complex64]) {
        assert_eq!(actual.len(), expected.len());

        for expected_value in expected {
            let (index, distance) = actual
                .iter()
                .enumerate()
                .map(|(index, actual_value)| (index, (*actual_value - *expected_value).norm()))
                .min_by(|left, right| left.1.total_cmp(&right.1))
                .unwrap();
            assert!(
                distance <= 1e-10,
                "expected eigenvalue {expected_value:?}, got {:?}",
                actual[index]
            );
            actual.remove(index);
        }
    }

    #[test]
    fn zero_by_zero_case() {
        let m = Matrix::<Complex64>::new(0, 0, vec![]).unwrap();
        let (vals, vecs) = m.clone().eigendecompose().unwrap();
        assert!(vals.is_empty());
        assert_eq!(vecs.rows(), 0);
        assert_eq!(vecs.cols(), 0);

        let vals_only = m.clone().eigenvalues().unwrap();
        assert_eq!(vals, vals_only);
    }

    #[test]
    fn one_by_one_case() {
        let m1 = Matrix::<f64>::new(1, 1, vec![1.0]).unwrap();
        let mc = m1.clone().map(Complex64::from);

        let (vals, vecs) = mc.clone().eigendecompose().unwrap();
        assert_eq!(vals, vec![Complex64::ONE]);
        assert_eq!(vecs, eye(1));

        let vals_only = mc.clone().eigenvalues().unwrap();
        assert_eq!(vals.len(), vals_only.len());
        assert!(approx_eq!(
            f64,
            vals[0].re,
            vals_only[0].re,
            epsilon = 1e-15
        ));
        assert!(approx_eq!(
            f64,
            vals[0].im,
            vals_only[0].im,
            epsilon = 1e-15
        ));
    }

    #[test]
    fn eigenvalues() {
        let ms = vec![
            Matrix::<Complex64>::new(
                3,
                3,
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(ComplexFloat::sqrt(0.5), 0.0),
                    Complex64::new(-ComplexFloat::sqrt(0.5), 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(ComplexFloat::sqrt(0.5), 0.0),
                    Complex64::new(ComplexFloat::sqrt(0.5), 0.0),
                ],
            )
            .unwrap(),
            Matrix::<Complex64>::new(
                2,
                2,
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(1.0, 0.0),
                ],
            )
            .unwrap(),
        ];

        for m in ms {
            let (vals, vecs) = m.clone().eigendecompose().unwrap();
            let vals_only = m.clone().eigenvalues().unwrap();
            assert_eq!(vals.len(), vals_only.len());
            for (&v1, &v2) in vals.iter().zip(vals_only.iter()) {
                assert!(approx_eq!(f64, v1.re, v2.re, epsilon = 1e-15));
                assert!(approx_eq!(f64, v1.im, v2.im, epsilon = 1e-15));
            }

            let ord = m.rows();

            for i in 0..ord {
                let vec = vecs.slice(0..ord, i..i + 1).unwrap();
                let val = vals[i];

                assert!(approx_eq!(
                    Matrix<Complex64>,
                    vec.clone_into_matrix() * val,
                    m.clone().mat_mul_mt(vec.clone_into_matrix()).unwrap(),
                    epsilon = 1e-10
                ));
            }
        }
    }

    #[test]
    fn invalid_eigendecomposition_non_square() {
        let m = Matrix::<Complex64>::new(2, 1, vec![Complex64::ONE, Complex64::ZERO]).unwrap();

        let err = m.clone().eigendecompose().unwrap_err();
        assert_eq!(err, TensorErrors::NonSquareMatrix);

        let err_only = m.clone().eigenvalues().unwrap_err();
        assert_eq!(err_only, TensorErrors::NonSquareMatrix);
    }

    #[test]
    fn f64_eigenvalues_only_handles_empty_singleton_and_diagonal_matrices() {
        let empty = Matrix::<f64>::new(0, 0, vec![]).unwrap();
        assert!(empty.eigenvalues().unwrap().is_empty());

        let singleton = Matrix::<f64>::new(1, 1, vec![-2.5]).unwrap();
        assert_eigenvalues_match(
            singleton.eigenvalues().unwrap(),
            &[Complex64::new(-2.5, 0.0)],
        );

        let diagonal =
            Matrix::<f64>::new(3, 3, vec![4.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.5]).unwrap();
        assert_eigenvalues_match(
            diagonal.eigenvalues().unwrap(),
            &[
                Complex64::new(4.0, 0.0),
                Complex64::new(-2.0, 0.0),
                Complex64::new(0.5, 0.0),
            ],
        );
    }

    #[test]
    fn f64_eigenvalues_only_handles_real_and_complex_spectra() {
        let symmetric = Matrix::<f64>::new(2, 2, vec![2.0, 1.0, 1.0, 2.0]).unwrap();
        assert_eigenvalues_match(
            symmetric.eigenvalues().unwrap(),
            &[Complex64::new(3.0, 0.0), Complex64::new(1.0, 0.0)],
        );

        let rotation = Matrix::<f64>::new(2, 2, vec![0.0, -1.0, 1.0, 0.0]).unwrap();
        assert_eigenvalues_match(
            rotation.eigenvalues().unwrap(),
            &[Complex64::new(0.0, 1.0), Complex64::new(0.0, -1.0)],
        );
    }

    #[test]
    fn complex_eigenvalues_only_handles_empty_singleton_and_triangular_matrices() {
        let empty = Matrix::<Complex64>::new(0, 0, vec![]).unwrap();
        assert!(empty.eigenvalues().unwrap().is_empty());

        let singleton = Matrix::<Complex64>::new(1, 1, vec![Complex64::new(2.0, -3.0)]).unwrap();
        assert_eigenvalues_match(
            singleton.eigenvalues().unwrap(),
            &[Complex64::new(2.0, -3.0)],
        );

        let triangular = Matrix::<Complex64>::new(
            3,
            3,
            vec![
                Complex64::new(1.0, 2.0),
                Complex64::ZERO,
                Complex64::ZERO,
                Complex64::new(-3.0, 0.5),
                Complex64::new(-2.0, -1.0),
                Complex64::ZERO,
                Complex64::new(4.0, -2.0),
                Complex64::new(0.25, 3.0),
                Complex64::new(0.5, 4.0),
            ],
        )
        .unwrap();
        assert_eigenvalues_match(
            triangular.eigenvalues().unwrap(),
            &[
                Complex64::new(1.0, 2.0),
                Complex64::new(-2.0, -1.0),
                Complex64::new(0.5, 4.0),
            ],
        );
    }

    #[test]
    fn eigenvalues_only_rejects_non_square_f64_matrices() {
        let rectangular = Matrix::<f64>::new(2, 1, vec![1.0, 2.0]).unwrap();
        assert_eq!(
            rectangular.eigenvalues().unwrap_err(),
            TensorErrors::NonSquareMatrix
        );
    }
}
