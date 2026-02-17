#[cfg(test)]
mod row_echelon_tests {
    use crate::definitions::matrix::Matrix;
    use float_cmp::approx_eq;
    use num::complex::Complex64;
    use crate::utilities::matrix::{eye, identity};

    #[test]
    fn ref_test() {
        let m1 = Matrix::<f64>::new(
            3, 5,
            vec![
                0.5, -5.0, 1.2, -3.4, 1.2,
                0.2, -2.0, 0.0, 0.2, 1.7,
                0.0, -1.0, 0.0, -3.6, 1.1,
            ],
        ).unwrap();

        let m1_ref = m1.row_echelon();
        assert!(m1_ref.is_row_echelon());
        assert!(approx_eq!(Matrix<f64>, m1_ref.reduced_row_echelon(), m1.reduced_row_echelon()));

        let m2 = Matrix::<Complex64>::new(
            4, 3,
            vec![
                Complex64::new(1.0, -2.0), Complex64::ZERO, Complex64::new(1.8, -3.1),
                Complex64::new(2.3, -1.0), Complex64::ZERO, Complex64::new(1.9, -2.1),
                Complex64::new(4.2, 3.0), Complex64::ZERO, Complex64::new(9.0, -2.2),
                Complex64::new(-1.5, 2.5), Complex64::ZERO, Complex64::new(8.0, -8.6),
            ],
        ).unwrap();

        let m2_ref = m2.row_echelon();
        assert!(approx_eq!(Matrix<Complex64>, m2_ref.reduced_row_echelon(), m2.reduced_row_echelon(), epsilon = 1e-14));

        let m3 = Matrix::<f64>::new(
            2, 2,
            vec![
                1.0, 2.0,
                3.0, 4.0,
            ],
        ).unwrap();

        let m3_ref = m3.row_echelon();
        assert!(m3_ref.is_row_echelon());
        assert!(approx_eq!(Matrix<f64>, m3_ref.reduced_row_echelon(), m3.reduced_row_echelon()));

        let m4 = Matrix::<Complex64>::new(
            3, 3,
            vec![
                Complex64::new(1.0, 2.0), Complex64::new(-4.0, 0.0), Complex64::new(3.0, 1.0),
                Complex64::new(1.0, 2.0), Complex64::new(-4.0, 0.0), Complex64::new(3.0, 1.0),
                Complex64::new(2.0, -5.0), Complex64::new(0.1, -3.5), Complex64::new(0.0, 1.0),
            ],
        ).unwrap();

        let m4_ref = m4.row_echelon();
        assert!(m4_ref.is_row_echelon());
        assert!(approx_eq!(Matrix<Complex64>, m4_ref.reduced_row_echelon(), m4.reduced_row_echelon()));
    }

    #[test]
    fn is_ref_test() {
        let m1 = Matrix::<f64>::new(
            3, 5,
            vec![
                0.5, -5.0, 1.2, -3.4, 1.2,
                0.2, -2.0, 0.0, 0.2, 1.7,
                0.0, -1.0, 0.0, -3.6, 1.1,
            ],
        ).unwrap();

        assert!(!m1.is_row_echelon());

        let m2 = Matrix::<Complex64>::new(
            4, 3,
            vec![
                Complex64::new(1.0, -2.0), Complex64::ZERO, Complex64::new(1.8, -3.1),
                Complex64::new(0.0, -0.0), Complex64::ZERO, Complex64::new(1.9, -2.1),
                Complex64::new(4.2, 3.0), Complex64::ZERO, Complex64::new(9.0, -2.2),
                Complex64::new(-1.5, 2.5), Complex64::ZERO, Complex64::new(8.0, -8.6),
            ],
        ).unwrap();

        assert!(!m2.is_row_echelon());

        let m3 = Matrix::<f64>::new(
            3, 4,
            vec![
                1.0, 2.0, 3.0, 4.0,
                0.0, 0.0, 1.0, 3.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        ).unwrap();

        assert!(m3.is_row_echelon());

        let m4 = Matrix::<Complex64>::new(
            5, 3,
            vec![
                Complex64::new(1.0, 2.0), Complex64::new(-4.0, 0.0), Complex64::new(3.0, 1.0),
                Complex64::new(0.0, 0.0), Complex64::new(-0.0, 0.0), Complex64::new(3.0, 1.0),
                Complex64::new(0.0, 0.0), Complex64::new(-0.0, 0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, -0.0), Complex64::new(0.0, -0.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, -0.0), Complex64::new(0.0, -0.0), Complex64::new(0.0, 0.0),
            ],
        ).unwrap();

        assert!(m4.is_row_echelon());

        let m5 = Matrix::<f64>::new(
            3, 3,
            vec![
                1.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 1.0,
            ],
        ).unwrap();

        assert!(!m5.is_row_echelon());
        assert!(!m5.map(Complex64::from).is_row_echelon());
    }

    #[test]
    fn is_rref_test() {
        let m1 = Matrix::<f64>::new(
            3, 5,
            vec![
                0.5, -5.0, 1.2, -3.4, 1.2,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, -1.0, 0.0, -3.6, 1.1,
            ],
        ).unwrap();

        let m1_rref = m1.reduced_row_echelon();

        assert!(!m1.is_reduced_row_echelon());
        assert!(m1_rref.is_reduced_row_echelon());
        assert!(m1_rref.into_complex().is_reduced_row_echelon());

        let m2 = Matrix::<Complex64>::new(
            4, 3,
            vec![
                Complex64::new(1.0, -2.0), Complex64::ZERO, Complex64::new(1.8, -3.1),
                Complex64::new(2.3, -1.0), Complex64::ZERO, Complex64::new(1.9, -2.1),
                Complex64::new(4.2, 3.0), Complex64::ZERO, Complex64::new(9.0, -2.2),
                Complex64::new(-1.5, 2.5), Complex64::ZERO, Complex64::new(8.0, -8.6),
            ],
        ).unwrap();

        let m2_rref = m2.reduced_row_echelon();

        assert!(!m2.is_reduced_row_echelon());
        assert!(m2_rref.is_reduced_row_echelon());

        let m3 = Matrix::<f64>::new(
            2, 2,
            vec![
                1.0, 2.0,
                3.0, 4.0,
            ],
        ).unwrap();

        let m3_rref = m3.reduced_row_echelon();

        assert!(!m3.is_reduced_row_echelon());
        assert!(m3_rref.is_reduced_row_echelon());
        assert!(m3_rref.into_complex().is_reduced_row_echelon());

        let m4 = Matrix::<Complex64>::new(
            3, 3,
            vec![
                Complex64::new(1.0, 2.0), Complex64::new(-4.0, 0.0), Complex64::new(3.0, 1.0),
                Complex64::new(1.0, 2.0), Complex64::new(-4.0, 0.0), Complex64::new(3.0, 1.0),
                Complex64::new(2.0, -5.0), Complex64::new(0.1, -3.5), Complex64::new(0.0, 1.0),
            ],
        ).unwrap();

        let m4_rref = m4.reduced_row_echelon();

        assert!(!m4.is_reduced_row_echelon());
        assert!(m4_rref.is_reduced_row_echelon());

        let m5 = Matrix::<f64>::new(
            3, 3,
            vec![
                1.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
            ],
        ).unwrap();

        assert!(m5.is_reduced_row_echelon());
        assert!(m5.into_complex().is_reduced_row_echelon());

        let m6 = Matrix::<f64>::new(
            3, 4,
            vec![
                1.0, 0.0, 1.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ],
        ).unwrap();

        assert!(!m6.is_reduced_row_echelon());
        assert!(!m6.into_complex().is_reduced_row_echelon());

        let m7 = Matrix::<f64>::new(
            3, 4,
            vec![
                1.0, 0.0, 1.0, 0.0,
                1.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ],
        ).unwrap();

        assert!(!m7.is_reduced_row_echelon());
        assert!(!m7.into_complex().is_reduced_row_echelon());

        let m8 = Matrix::<f64>::new(
            3, 3,
            vec![
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 0.0,
            ],
        ).unwrap();

        assert!(m8.is_reduced_row_echelon());
        assert!(m8.into_complex().is_reduced_row_echelon());
    }

    #[test]
    fn rref_test() {
        let m1 = Matrix::<f64>::new(
            3, 5,
            vec![
                0.5, -5.0, 1.2, -3.4, 1.2,
                0.2, -2.0, 0.0, 0.2, 1.7,
                0.0, -1.0, 0.0, -3.6, 1.1,
            ],
        ).unwrap();

        let m1_rref = m1.reduced_row_echelon();
        let m1_rref_ans = eye(3).unwrap().concat_mt(&Matrix::<f64>::new(
            3, 2,
            vec![
                37.0, -2.5,
                3.6, -1.1,
                -3.25, -61.0/24.0,
            ],
        ).unwrap(), 1).unwrap();

        assert!(approx_eq!(Matrix<f64>, m1_rref, m1_rref_ans, epsilon = 1e-15));

        let m2 = Matrix::<Complex64>::new(
            4, 3,
            vec![
                Complex64::new(1.0, -2.0), Complex64::ZERO, Complex64::new(1.8, -3.1),
                Complex64::new(2.3, -1.0), Complex64::ZERO, Complex64::new(1.9, -2.1),
                Complex64::new(4.2, 3.0), Complex64::ZERO, Complex64::new(9.0, -2.2),
                Complex64::new(-1.5, 2.5), Complex64::ZERO, Complex64::new(8.0, -8.6),
            ],
        ).unwrap();

        let m2_rref = m2.reduced_row_echelon();
        let m2_rref_ans = Matrix::<Complex64>::new(
            4, 3,
            vec![
                Complex64::ONE, Complex64::ZERO, Complex64::ZERO,
                Complex64::ZERO, Complex64::ZERO, Complex64::ONE,
                Complex64::ZERO, Complex64::ZERO, Complex64::ZERO,
                Complex64::ZERO, Complex64::ZERO, Complex64::ZERO,
            ],
        ).unwrap();

        assert!(approx_eq!(Matrix<Complex64>, m2_rref, m2_rref_ans, epsilon = 1e-15));

        let m3 = Matrix::<f64>::new(
            2, 2,
            vec![
                1.0, 2.0,
                3.0, 4.0,
            ],
        ).unwrap();

        let m3_rref = m3.reduced_row_echelon();
        let m3_rref_ans = identity(2).unwrap();

        assert!(approx_eq!(Matrix<f64>, m3_rref, m3_rref_ans, epsilon = 1e-15));

        let m4 = Matrix::<Complex64>::new(
            3, 3,
            vec![
                Complex64::new(1.0, 2.0), Complex64::new(-4.0, 0.0), Complex64::new(3.0, 1.0),
                Complex64::new(1.0, 2.0), Complex64::new(-4.0, 0.0), Complex64::new(3.0, 1.0),
                Complex64::new(2.0, -5.0), Complex64::new(0.1, -3.5), Complex64::new(0.0, 1.0),
            ],
        ).unwrap();

        let m4_rref = m4.reduced_row_echelon();
        let m4_rref_ans = Matrix::<Complex64>::new(
            3, 3,
            vec![
                Complex64::ONE, Complex64::ZERO, Complex64 { re: 0.2678687248670384, im: -0.01050719937735134 },
                Complex64::ZERO, Complex64::ONE, Complex64 { re: -0.6777792190945647, im: -0.1186924374108186 },
                Complex64::ZERO, Complex64::ZERO, Complex64::ZERO,
            ],
        ).unwrap();

        assert!(approx_eq!(Matrix<Complex64>, m4_rref, m4_rref_ans, epsilon = 1e-15));

        let m5 = Matrix::<f64>::new(
            3, 6,
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 2.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 5.0, 5.0,
            ],
        ).unwrap();

        let m5_rref = m5.reduced_row_echelon();
        let m5_rref_ans = Matrix::<f64>::new(
            3, 6,
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
            ],
        ).unwrap();

        println!("{:?}", m5_rref);
        println!("{:?}", m5_rref_ans);

        assert!(approx_eq!(Matrix<f64>, m5_rref, m5_rref_ans.clone(), epsilon = 1e-15));

        let m5_rref = m5.into_complex().reduced_row_echelon();
        let m5_rref_ans = m5_rref_ans.into_complex();

        assert!(approx_eq!(Matrix<Complex64>, m5_rref, m5_rref_ans, epsilon = 1e-15));

        let m6 = Matrix::<f64>::new(
            3, 6,
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 2.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ).unwrap();

        let m6_rref = m6.reduced_row_echelon();
        let m6_rref_ans = Matrix::<f64>::new(
            3, 6,
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ).unwrap();

        println!("{:?}", m6_rref);
        println!("{:?}", m6_rref_ans);

        assert!(approx_eq!(Matrix<f64>, m6_rref, m6_rref_ans.clone(), epsilon = 1e-15));

        let m6_rref = m6.into_complex().reduced_row_echelon();
        let m6_rref_ans = m6_rref_ans.into_complex();

        assert!(approx_eq!(Matrix<Complex64>, m6_rref, m6_rref_ans, epsilon = 1e-15));

        let m7 = Matrix::<f64>::new(
            3, 3,
            vec![
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 0.0,
            ],
        ).unwrap();

        let m7_rref = m7.reduced_row_echelon();
        let m7_rref_ans = m7.clone();

        assert_eq!(m7_rref, m7_rref_ans);
        assert_eq!(m7.into_complex().reduced_row_echelon(), m7_rref_ans.into_complex());
    }
}