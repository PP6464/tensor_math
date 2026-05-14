#[cfg(test)]
mod det_inv_tests {
    use crate::definitions::errors::TensorErrors;
    use crate::definitions::matrix::Matrix;
    use crate::definitions::traits::IntoMatrix;
    use crate::math::det_inv::{det_slow, inv_slow};
    use crate::utilities::matrix::eye;
    use float_cmp::approx_eq;
    use num::complex::Complex64;
    use num::FromPrimitive;

    #[test]
    fn invalid_inversion_square_matrix_only() {
        let m1 = Matrix::<f64>::new(3, 2, (0..6).map(f64::from).collect()).unwrap();

        let err = inv_slow(&m1).unwrap_err();
        match err {
            TensorErrors::NonSquareMatrix => {},
            _ => panic!("Incorrect error"),
        }

        let err = m1.inv().unwrap_err();
        match err {
            TensorErrors::NonSquareMatrix => {}
            _ => panic!("Incorrect error"),
        }

        let m2 = Matrix::new(3, 2,  (0..6).map(Complex64::from_i32).map(|x| x.unwrap()).collect()).unwrap();
        let err = m2.inv().unwrap_err();

        match err {
            TensorErrors::NonSquareMatrix => {},
            _ => panic!("Incorrect error"),
        };
    }

    #[test]
    fn determinant() {
        let m1 = Matrix::<i32>::new(3, 3, vec![5, -2, 1, 8, 9, -5, 1, 0, 2]).unwrap();
        let m2 = vec![1].into_matrix();
        let m3 = Matrix::<f64>::zeros(10, 10);

        assert_eq!(det_slow(&m2).unwrap(), 1);
        assert_eq!(det_slow(&m1).unwrap(), 123);
        assert!(approx_eq!(f64, m1.map(|x| x as f64).det().unwrap(), 123.0));
        assert_eq!(m3.det().unwrap(), 0.0);
        assert_eq!(m3.into_complex().det().unwrap(), 0.0.into());
    }

    #[test]
    fn invalid_det_square_matrix_only() {
        let m1 = Matrix::<f64>::new(3, 2, (0..6).map(f64::from).collect()).unwrap();

        let err = det_slow(&m1).unwrap_err();
        match err {
            TensorErrors::NonSquareMatrix => {},
            _ => panic!("Incorrect error"),
        };

        let err = m1.det().unwrap_err();
        match err {
            TensorErrors::NonSquareMatrix => {},
            _ => panic!("Incorrect error"),
        };

        let m2 = Matrix::new(3, 2, (0..6).map(Complex64::from_i32).map(|x| x.unwrap()).collect()).unwrap();

        let err = m2.det().unwrap_err();
        match err {
            TensorErrors::NonSquareMatrix => {},
            _ => panic!("Incorrect error"),
        };
    }

    #[test]
    fn inverse() {
        let m1 = Matrix::<f64>::new(
            3,
            3,
            vec![3.0, 4.0, 5.0, 2.0, -1.0, 4.0, 3.0, -5.0, -10.0],
        )
            .unwrap();
        let inverse = inv_slow(&m1).unwrap();
        let fast_inverse = m1.inv().unwrap();
        let ans = Matrix::<f64>::new(
            3,
            3,
            vec![
                10.0 / 61.0,
                5.0 / 61.0,
                7.0 / 61.0,
                32.0 / 183.0,
                -15.0 / 61.0,
                -2.0 / 183.0,
                -7.0 / 183.0,
                9.0 / 61.0,
                -11.0 / 183.0,
            ],
        )
            .unwrap();

        assert!(approx_eq!(Matrix<f64>, inverse.clone(), ans.clone(), epsilon = 1e-15));
        assert!(approx_eq!(Matrix<f64>, fast_inverse, ans, epsilon = 1e-15));
        assert!(m1
            .contract_mul(&inverse)
            .unwrap()
            .enumerated_iter()
            .all(|(i, x)| { approx_eq!(f64, x, eye(3)[i], epsilon = 1e-15) }));

        let m2 = Matrix::new(
            2, 2,
            vec![
                Complex64 { re: 1.0, im: 1.0 }, Complex64 { re: 1.0, im: 0.0 },
                Complex64 { re: -1.0, im: 1.0 }, Complex64 { re: 0.0, im: -1.0 },
            ]
        ).unwrap();

        let slow_res = inv_slow(&m2).unwrap();
        let fast_res = m2.inv().unwrap();
        let ans = Matrix::new(
            2, 2,
            vec![
                Complex64 { re: 0.25, im: -0.25 }, Complex64 { re: -0.25, im: -0.25 },
                Complex64 { re: 0.5, im: 0.0 }, Complex64 { re: 0.0, im: 0.5 },
            ],
        ).unwrap();

        assert!(approx_eq!(Matrix<Complex64>, slow_res.clone(), ans.clone(), epsilon = 1e-15));
        assert!(approx_eq!(Matrix<Complex64>, fast_res, ans, epsilon = 1e-15));
        assert!(m2
            .contract_mul(&slow_res)
            .unwrap()
            .enumerated_iter()
            .all(|(i, x)| {
                approx_eq!(f64, x.re, eye(3)[i], epsilon = 1e-15);
                approx_eq!(f64, x.im, 0.0, epsilon = 1e-15)
            }));
    }

    #[test]
    fn inverse_det_0() {
        let m1 = Matrix::<f64>::new(3, 3, vec![0.0; 9]).unwrap();
        let m2 = Matrix::new(
            2, 2,
            vec![
                Complex64 { re: 1.0, im: 1.0 }, Complex64 { re: 1.0, im: 0.0 },
                Complex64 { re: 1.0, im: 1.0 }, Complex64 { re: 1.0, im: 0.0 },
            ],
        ).unwrap();

        let err = inv_slow(&m1).unwrap_err();
        match err {
            TensorErrors::DeterminantZero => {},
            _ => panic!("Incorrect error"),
        }
        
        let err = m1.inv().unwrap_err();
        match err {
            TensorErrors::DeterminantZero => {},
            _ => panic!("Incorrect error"),
        }

        let err = m2.inv().unwrap_err();
        match err {
            TensorErrors::DeterminantZero => {}
            _ => panic!("Incorrect error"),
        }
    }
    
    #[test]
    fn empty_matrix_det_inv() {
        let m_f64 = Matrix::<f64>::new(0, 0, vec![]).unwrap();
        let m_complex = Matrix::<Complex64>::new(0, 0, vec![]).unwrap();

        // f64 tests
        assert_eq!(m_f64.det().unwrap(), 1.0);
        assert_eq!(det_slow(&m_f64).unwrap(), 1.0);
        assert_eq!(m_f64.inv().unwrap().rows, 0);
        assert_eq!(inv_slow(&m_f64).unwrap().rows, 0);

        // Complex64 tests
        assert_eq!(m_complex.det().unwrap(), Complex64::new(1.0, 0.0));
        assert_eq!(m_complex.inv().unwrap().rows, 0);
        assert_eq!(inv_slow(&m_complex).unwrap().rows, 0);
    }
}
