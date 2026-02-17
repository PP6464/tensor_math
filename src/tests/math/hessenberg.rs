#[cfg(test)]
mod hessenberg_tests {
    use float_cmp::approx_eq;
    use num::complex::Complex64;
    use crate::definitions::errors::TensorErrors;
    use crate::definitions::matrix::Matrix;
    use crate::utilities::matrix::eye;

    #[test]
    fn upper_hessenberg() {
        let m1 = Matrix::<Complex64>::new(
            3, 3,
            vec![
                Complex64 { re: 3.0, im: 0.0 }, Complex64 { re: 2.0, im: 0.0 }, Complex64 { re: 0.0, im: 4.0 },
                Complex64 { re: 0.0, im: 0.0 }, Complex64 { re: 0.0, im: 5.0 }, Complex64 { re: 0.0, im: 6.0 },
                Complex64 { re: -5.0, im: 4.0 }, Complex64 { re: 7.0, im: 0.0 }, Complex64 { re: 3.0, im: 0.0 },
            ],
        ).unwrap();
        let (h1, q1) = m1.upper_hessenberg().unwrap();
        let h1_ans = Matrix::<Complex64>::new(
            3, 3,
            vec![
                Complex64 { re: 3.0, im: 0.0 }, Complex64 { re: 2.498780190217697, im: 3.123475237772121 }, Complex64 { re: 1.5617376188860606, im: 1.2493900951088488 },
                Complex64 { re: -6.4031242374328485, im: 0.0 }, Complex64 { re: 3.0, im: 0.0 }, Complex64 { re: 1.536585365853658, im: 6.829268292682927 },
                Complex64 { re: 0.0, im: 0.0 }, Complex64 { re: 5.853658536585366, im: 1.3170731707317078 }, Complex64 { re: 0.0, im: 5.0 },
            ],
        ).unwrap();

        assert!(approx_eq!(Matrix<Complex64>, h1.clone(), h1_ans, epsilon = 1e-13));
        assert!(approx_eq!(Matrix<Complex64>, h1, q1.clone().conj_transpose().contract_mul_mt(&m1).unwrap().contract_mul_mt(&q1).unwrap(), epsilon = 1e-15));

        let m2 = Matrix::<f64>::new(
            4, 4,
            vec![
                1.0, 0.5, 0.25, 0.125,
                0.3, 0.6, 0.9, 1.2,
                0.4, 0.8, 1.2, 1.6,
                -5.0, 10.0, -15.0, 20.0,
            ],
        ).unwrap();

        let (h2, q2) = m2.upper_hessenberg().unwrap();
        let h2_ans = Matrix::<f64>::new(
            4, 4,
            vec![
                1.0, 0.07462778926574923, -0.5099565599826192, 0.25000000000000017,
                -5.024937810560445, 20.21584158415841, 4.158415841584154, 16.92558260547191,
                0.0, -3.8415841584158397, 1.5841584158415822, -1.5920595043359822,
                0.0, 0.0, -6.661338147750939e-16, 6.661338147750939e-16,
            ]
        ).unwrap();

        assert!(approx_eq!(Matrix<f64>, h2.clone(), h2_ans, epsilon = 1e-13));
        assert!(approx_eq!(Matrix<f64>, h2, q2.clone().transpose().contract_mul_mt(&m2).unwrap().contract_mul_mt(&q2).unwrap(), epsilon = 1e-13));
    }

    #[test]
    fn lower_hessenberg() {
        let m1 = Matrix::<Complex64>::new(
            3, 3,
            vec![
                Complex64 { re: 3.0, im: 0.0 }, Complex64 { re: 2.0, im: 0.0 }, Complex64 { re: 0.0, im: 4.0 },
                Complex64 { re: 0.0, im: 0.0 }, Complex64 { re: 0.0, im: 5.0 }, Complex64 { re: 0.0, im: 6.0 },
                Complex64 { re: -5.0, im: 4.0 }, Complex64 { re: 7.0, im: 0.0 }, Complex64 { re: 3.0, im: 0.0 },
            ],
        ).unwrap();
        let (h1, q1) = m1.lower_hessenberg().unwrap();
        let h1_ans = Matrix::<Complex64>::new(
            3, 3,
            vec![
                Complex64 { re: 3.0, im: 0.0 }, Complex64 { re: -4.472135954999581, im: -0.0 }, Complex64 { re: 0.0, im: -2.220446049250313e-16 },
                Complex64 { re: 3.5777087639996643, im: 4.4721359549995805 }, Complex64 { re: 4.800000000000005, im: 3.8000000000000074 }, Complex64 { re: -7.600000000000007, im: -2.400000000000002 },
                Complex64 { re: -2.23606797749979, im: 1.788854381999832 }, Complex64 { re: 0.6000000000000011, im: -3.6000000000000014 }, Complex64 { re: -1.8000000000000005, im: 1.2000000000000008 }],
        ).unwrap();

        assert!(approx_eq!(Matrix<Complex64>, h1.clone(), h1_ans, epsilon = 1e-13));
        assert!(approx_eq!(Matrix<Complex64>, h1, q1.clone().conj_transpose().contract_mul_mt(&m1).unwrap().contract_mul_mt(&q1).unwrap(), epsilon = 1e-15));

        let m2 = Matrix::<f64>::new(
            4, 4,
            vec![
                1.0, 0.5, 0.25, 0.125,
                0.3, 0.6, 0.9, 1.2,
                0.4, 0.8, 1.2, 1.6,
                -5.0, 10.0, -15.0, 20.0,
            ],
        ).unwrap();

        let (h2, q2) = m2.lower_hessenberg().unwrap();
        let h2_ans = Matrix::<f64>::new(
            4, 4,
            vec![
                1.0, -0.57282196186948, 0.0, 0.0,
                0.6546536707079773, 3.142857142857143, -6.375441828093274, 0.0,
                -4.448647121025878, -4.805429258075459, 21.194716078900154, -5.948528383276557,
                -2.242981801980734, -3.7073689241171914, 12.403596185570388, -2.5375732217573193]

        ).unwrap();

        assert!(approx_eq!(Matrix<f64>, h2.clone(), h2_ans, epsilon = 1e-13));
        assert!(approx_eq!(Matrix<f64>, h2, q2.clone().transpose().contract_mul_mt(&m2).unwrap().contract_mul_mt(&q2).unwrap(), epsilon = 1e-13));
    }

    #[test]
    fn invalid_hessenberg_non_square() {
        let m1 = Matrix::<f64>::rand(2, 3);
        let m2  = m1.clone().map(Complex64::from);

        let err = m1.upper_hessenberg().unwrap_err();
        assert_eq!(err, TensorErrors::NonSquareMatrix);

        let err = m1.lower_hessenberg().unwrap_err();
        assert_eq!(err, TensorErrors::NonSquareMatrix);

        let err = m2.upper_hessenberg().unwrap_err();
        assert_eq!(err, TensorErrors::NonSquareMatrix);

        let err = m2.lower_hessenberg().unwrap_err();
        assert_eq!(err, TensorErrors::NonSquareMatrix);
    }

    #[test]
    fn hessenberg_trivial_cases() {
        let m1 = Matrix::<f64>::rand(2, 2);
        let m2 = Matrix::<f64>::rand(1, 1);

        assert_eq!(m1.upper_hessenberg().unwrap().0, m1);
        assert_eq!(m2.upper_hessenberg().unwrap().0, m2);
    }

    #[test]
    fn hessenberg_of_zero() {
        let m1 = Matrix::<f64>::zeros(5, 5).unwrap();

        let (h, q) = m1.upper_hessenberg().unwrap();

        assert_eq!(h, Matrix::zeros(5, 5).unwrap());
        assert_eq!(q, eye(5).unwrap());
    }
}
