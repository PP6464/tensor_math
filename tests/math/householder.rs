#[cfg(test)]
mod householder_tests {
    use float_cmp::{ApproxEq, F64Margin, FloatMargin};
    use num::complex::{Complex64, ComplexFloat};
    use tensor_math::definitions::matrix::Matrix;
    use tensor_math::definitions::shape::Shape;
    use tensor_math::definitions::tensor::Tensor;
    use tensor_math::shape;
    use tensor_math::utilities::matrix::eye;

    #[test]
    fn test_householder() {
        let m1 = Matrix::new(3, 3, vec![4.0, 1.0, 1.0, 1.0, 3.0, 0.0, 1.0, 0.0, 2.0]).unwrap();

        let (q1, r1) = m1.householder();

        for i in 0..r1.shape()[0] {
            if i >= r1.shape()[1] - 1 {
                continue;
            }

            assert!(r1
                .slice(i + 1..r1.shape()[0], i..i + 1)
                .unwrap()
                .iter()
                .all(|x| { x.abs() <= 1e-10 }));
        }
        assert!(q1
            .contract_mul(&r1)
            .unwrap()
            .approx_eq(m1.clone(), F64Margin::default().epsilon(1e-10)));

        let m2: Matrix<Complex64> = Tensor::<Complex64>::new(
            &shape![3, 2],
            vec![
                Complex64 { re: 4.0, im: 1.0 },
                Complex64 { re: -5.0, im: -2.0 },
                Complex64 { re: 5.0, im: -4.0 },
                Complex64 { re: 5.0, im: 3.0 },
                Complex64 { re: 0.0, im: 0.0 },
                Complex64 { re: 1.0, im: -1.0 },
            ],
        )
        .unwrap()
        .try_into()
        .unwrap();
        let (q2, r2) = m2.householder();

        for i in 0..r2.shape()[0] {
            if i >= r2.shape()[1] - 1 {
                continue;
            }

            assert!(r2
                .slice(i + 1..r2.shape()[0], i..i + 1)
                .unwrap()
                .iter()
                .all(|x| x.abs() <= 1e-10));
        }
        assert!(q2
            .contract_mul(&r2)
            .unwrap()
            .approx_eq(m2, F64Margin::default().epsilon(1e-10)));

        let m3: Matrix<Complex64> = Tensor::<Complex64>::new(
            &shape![2, 3],
            vec![
                Complex64 { re: -4.0, im: -1.0 },
                Complex64 { re: 5.0, im: -3.0 },
                Complex64 { re: 2.0, im: -4.0 },
                Complex64 { re: -5.0, im: 2.0 },
                Complex64 { re: 2.0, im: -1.0 },
                Complex64 { re: 4.0, im: -1.0 },
            ],
        )
        .unwrap()
        .try_into()
        .unwrap();
        let (q3, r3) = m3.householder();

        for i in 0..(r3.shape()[0] - 1) {
            if i >= r3.shape()[1] - 1 {
                continue;
            }
            assert!(r3
                .slice(i + 1..r3.shape()[0], i..i + 1)
                .unwrap()
                .iter()
                .all(|x| x.abs() <= 1e-10));
        }
        assert!(q3
            .contract_mul(&r3)
            .unwrap()
            .approx_eq(m3, F64Margin::default().epsilon(1e-10)));
    }

    #[test]
    fn householder_of_zeros() {
        let m1 = Matrix::<f64>::zeros(3, 3);
        let m2 = Matrix::<Complex64>::zeros(3, 3);

        assert_eq!(m1.householder(), (eye(3), m1));
        assert_eq!(m2.householder(), (eye(3), m2));
    }

    #[test]
    fn householder_intermediate_zero_reflector() {
        let m1 = Matrix::new(
            4,
            3,
            vec![
                1.0, 2.0, 3.0,
                1.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                1.0, 0.0, 1.0,
            ],
        )
        .unwrap();
        let (q1, r1) = m1.householder();

        for i in 0..r1.rows() {
            if i >= r1.cols() - 1 {
                continue;
            }
            assert!(r1
                .slice(i + 1..r1.rows(), i..i + 1)
                .unwrap()
                .iter()
                .all(|x| { x.abs() <= 1e-10 }));
        }
        assert!(q1
            .contract_mul(&r1)
            .unwrap()
            .approx_eq(m1, F64Margin::default().epsilon(1e-10)));

        let m2= Matrix::<Complex64>::new(
            4, 3,
            vec![
                Complex64 { re: 1.0, im: 0.0 }, Complex64 { re: 0.5, im: 1.0 }, Complex64 { re: 0.0, im: 0.0 },
                Complex64 { re: 2.0, im: 1.0 }, Complex64 { re: 0.0, im: 0.0 }, Complex64 { re: 0.0, im: 0.0 },
                Complex64 { re: 3.0, im: -2.0 }, Complex64 { re: 0.0, im: 0.0 }, Complex64 { re: 0.0, im: 0.0 },
                Complex64 { re: 3.0, im: -2.0 }, Complex64 { re: 0.0, im: 0.0 }, Complex64 { re: 2.0, im: -10.0 },
            ],
        )
        .unwrap();
        let (q2, r2) = m2.householder();

        for i in 0..r2.shape()[0] {
            if i >= r2.shape()[1] - 1 {
                continue;
            }
            assert!(r2
                .slice(i + 1..r2.shape()[0], i..i + 1)
                .unwrap()
                .iter()
                .all(|x| x.abs() <= 1e-10));
        }
        assert!(q2
            .contract_mul(&r2)
            .unwrap()
            .approx_eq(m2, F64Margin::default().epsilon(1e-10)));
    }

    #[test]
    fn householder_empty_matrices() {
        let m1 = Matrix::<f64>::zeros(0, 0);
        let (q1, r1) = m1.householder();
        assert_eq!(q1, eye(0));
        assert_eq!(r1, m1);

        let m2 = Matrix::<f64>::zeros(3, 0);
        let (q2, r2) = m2.householder();
        assert_eq!(q2, eye(3));
        assert_eq!(r2, m2);

        let m3 = Matrix::<f64>::zeros(0, 3);
        let (q3, r3) = m3.householder();
        assert_eq!(q3, eye(0));
        assert_eq!(r3, m3);
    }
}
