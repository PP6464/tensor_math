#[cfg(test)]
mod eigen_tests {
    use crate::definitions::matrix::Matrix;
    use float_cmp::approx_eq;
    use num::complex::{Complex64, ComplexFloat};
    use crate::definitions::errors::TensorErrors;
    use crate::utilities::matrix::eye;

    #[test]
    fn one_by_one_case() {
        let m1 = Matrix::<f64>::new(1, 1, vec![1.0]).unwrap();

        let (vals, vecs) = m1.map(Complex64::from).eigendecompose().unwrap();
        assert_eq!(vals, vec![Complex64::ONE]);
        assert_eq!(vecs, eye(1).unwrap());
    }

    #[test]
    fn eigenvalues() {
        let ms = vec![
            Matrix::<Complex64>::new(
                3, 3,
                vec![
                    Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0), Complex64::new(ComplexFloat::sqrt(0.5), 0.0), Complex64::new(-ComplexFloat::sqrt(0.5), 0.0),
                    Complex64::new(0.0, 0.0), Complex64::new(ComplexFloat::sqrt(0.5), 0.0), Complex64::new(ComplexFloat::sqrt(0.5), 0.0),
                ],
            ).unwrap(),
            Matrix::<Complex64>::new(
                2, 2,
                vec![
                    Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0),
                    Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0),
                ],
            ).unwrap(),
        ];

        for m in ms {
            let (vals, vecs) = m.eigendecompose().unwrap();
            let ord = m.rows;

            for i in 0..ord {
                let vec = vecs.slice(0..ord, i..i + 1).unwrap();
                let val = vals[i];

                assert!(approx_eq!(Matrix<Complex64>, vec.clone() * val, m.contract_mul_mt(&vec).unwrap(), epsilon = 1e-13));
            }
        }
    }

    #[test]
    fn invalid_eigendecomposition_non_square() {
        let err = Matrix::<Complex64>::new(2, 1, vec![Complex64::ONE, Complex64::ZERO]).unwrap().eigendecompose().unwrap_err();
        assert_eq!(err, TensorErrors::NonSquareMatrix);
    }
}