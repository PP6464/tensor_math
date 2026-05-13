#[cfg(test)]
mod transformation_rank_tests {
    use num::complex::Complex64;
    use crate::definitions::matrix::Matrix;
    use crate::utilities::matrix::identity;

    #[test]
    fn transformation_rank() {
        let m1: Matrix<f64> = identity(3);
        assert_eq!(m1.transformation_rank(), 3);

        let mut m2 = identity(4);
        m2.slice_mut(1..3, 0..4).unwrap().set_all(&Matrix::<f64>::zeros(2, 4)).unwrap();
        assert_eq!(m2.transformation_rank(), 2);

        let m3 = Matrix::<Complex64>::new(
            3, 4,
            vec![
                Complex64::new(1.0, 2.0), Complex64::new(-4.0, 0.0), Complex64::new(3.0, 1.0), Complex64::new(3.0, 1.0),
                Complex64::new(1.0, 3.0), Complex64::new(-4.0, 1.0), Complex64::new(3.2, -1.0), Complex64::new(3.0, 1.0),
                Complex64::new(1.0, 4.0), Complex64::new(-3.0, 0.0), Complex64::new(3.1, 1.5), Complex64::new(3.0, 1.0),
            ],
        ).unwrap();
        assert_eq!(m3.transformation_rank(), 3);

        let m4 = Matrix::<Complex64>::new(
            3, 3,
            vec![
                Complex64::new(1.0, 2.0), Complex64::new(3.0, 1.0), Complex64::new(3.0, 1.0),
                Complex64::new(1.0, 3.0), Complex64::new(3.0, 1.0), Complex64::new(3.0, 1.0),
                Complex64::new(1.0, 4.0), Complex64::new(3.0, 1.0), Complex64::new(3.0, 1.0),
            ],
        ).unwrap();

        assert_eq!(m4.transformation_rank(), 2);

        let m5 = Matrix::<f64>::zeros(3, 3);
        assert_eq!(m5.transformation_rank(), 0);

        let m6 = Matrix::<Complex64>::zeros(3, 3);
        assert_eq!(m6.transformation_rank(), 0);
    }
}