#[cfg(test)]
mod eigen_tests {
    use tensor_math::definitions::errors::TensorErrors;
    use tensor_math::definitions::matrix::Matrix;
    use tensor_math::utilities::matrix::eye;
    use float_cmp::approx_eq;
    use num::complex::{Complex64, ComplexFloat};
    
    #[test]
    fn zero_by_zero_case() {
        let m = Matrix::<Complex64>::new(0, 0, vec![]).unwrap();
        let (vals, vecs) = m.eigendecompose().unwrap();
        assert!(vals.is_empty());
        assert_eq!(vecs.rows(), 0);
        assert_eq!(vecs.cols(), 0);

        let vals_only = m.eigenvalues().unwrap();
        assert_eq!(vals, vals_only);
    }

    #[test]
    fn one_by_one_case() {
        let m1 = Matrix::<f64>::new(1, 1, vec![1.0]).unwrap();
        let mc = m1.map(Complex64::from);

        let (vals, vecs) = mc.eigendecompose().unwrap();
        assert_eq!(vals, vec![Complex64::ONE]);
        assert_eq!(vecs, eye(1));

        let vals_only = mc.eigenvalues().unwrap();
        assert_eq!(vals.len(), vals_only.len());
        assert!(approx_eq!(f64, vals[0].re, vals_only[0].re, epsilon = 1e-15));
        assert!(approx_eq!(f64, vals[0].im, vals_only[0].im, epsilon = 1e-15));
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
            let (vals, vecs) = m.eigendecompose().unwrap();
            let vals_only = m.eigenvalues().unwrap();
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
                    vec.clone() * val,
                    m.contract_mul_mt(&vec).unwrap(),
                    epsilon = 1e-10
                ));
            }
        }
    }

    #[test]
    fn invalid_eigendecomposition_non_square() {
        let m = Matrix::<Complex64>::new(2, 1, vec![Complex64::ONE, Complex64::ZERO]).unwrap();

        let err = m.eigendecompose().unwrap_err();
        assert_eq!(err, TensorErrors::NonSquareMatrix);

        let err_only = m.eigenvalues().unwrap_err();
        assert_eq!(err_only, TensorErrors::NonSquareMatrix);
    }
}
