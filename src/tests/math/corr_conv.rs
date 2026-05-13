#[cfg(test)]
mod corr_conv_tests {
    use crate::definitions::matrix::Matrix;
    use crate::definitions::shape::Shape;
    use crate::definitions::tensor::Tensor;
    use crate::shape;
    use float_cmp::approx_eq;
    use num::complex::Complex64;
    use std::collections::HashSet;
    use num::ToPrimitive;
    use crate::definitions::errors::TensorErrors;
    
    #[test]
    fn cannot_corr_conv_empty_tensors() {
        let empty = Tensor::<f64>::new(&shape![0, 5], vec![]).unwrap();
        let normal = Tensor::<f64>::new(&shape![2, 5], vec![1.0; 10]).unwrap();

        assert!(matches!(empty.corr(&normal).unwrap_err(), TensorErrors::TensorEmpty { .. }));
        assert!(matches!(normal.corr(&empty).unwrap_err(), TensorErrors::TensorEmpty { .. }));
        assert!(matches!(empty.conv(&normal).unwrap_err(), TensorErrors::TensorEmpty { .. }));
        assert!(matches!(empty.corr_axes(&normal, &HashSet::from([1])).unwrap_err(), TensorErrors::TensorEmpty { .. }));
    }
    
    #[test]
    fn cannot_corr_conv_mt_empty_tensors() {
        let empty = Tensor::<f64>::new(&shape![0, 5], vec![]).unwrap();
        let normal = Tensor::<f64>::new(&shape![2, 5], vec![1.0; 10]).unwrap();

        assert!(matches!(empty.corr_mt(&normal).unwrap_err(), TensorErrors::TensorEmpty { .. }));
        assert!(matches!(empty.conv_mt(&normal).unwrap_err(), TensorErrors::TensorEmpty { .. }));
        assert!(matches!(empty.corr_axes_mt(&normal, &HashSet::from([1])).unwrap_err(), TensorErrors::TensorEmpty { .. }));
    }

    #[test]
    fn cannot_corr_conv_empty_matrices() {
        let empty = Matrix::<f64>::new(0, 5, vec![]).unwrap();
        let empty2 = Matrix::<f64>::new(5, 0, vec![]).unwrap();
        let normal = Matrix::<f64>::new(2, 5, vec![1.0; 10]).unwrap();
        let normal2 = Matrix::<f64>::new(5, 2, vec![1.0; 10]).unwrap();

        assert!(matches!(empty.corr(&normal).unwrap_err(), TensorErrors::TensorEmpty { .. }));
        assert!(matches!(empty.conv(&normal).unwrap_err(), TensorErrors::TensorEmpty { .. }));
        assert!(matches!(empty2.corr_rows(&normal2).unwrap_err(), TensorErrors::TensorEmpty { .. }));
        assert!(matches!(empty.corr_cols(&normal).unwrap_err(), TensorErrors::TensorEmpty { .. }));
    }
    
    #[test]
    fn cannot_corr_conv_mt_empty_matrices() {
        let empty = Matrix::<f64>::new(0, 5, vec![]).unwrap();
        let empty2 = Matrix::<f64>::new(5, 0, vec![]).unwrap();
        let normal = Matrix::<f64>::new(2, 5, vec![1.0; 10]).unwrap();
        let normal2 = Matrix::<f64>::new(5, 2, vec![1.0; 10]).unwrap();

        assert!(matches!(empty.corr_mt(&normal).unwrap_err(), TensorErrors::TensorEmpty { .. }));
        assert!(matches!(empty.conv(&normal).unwrap_err(), TensorErrors::TensorEmpty { .. }));
        assert!(matches!(empty2.corr_rows_mt(&normal2).unwrap_err(), TensorErrors::TensorEmpty { .. }));
        assert!(matches!(empty.corr_cols_mt(&normal).unwrap_err(), TensorErrors::TensorEmpty { .. }));
    }

    #[test]
    fn cannot_corr_conv_scalar_tensors() {
        let scalar = Tensor::<f64>::new(&shape![], vec![1.0]).unwrap();
        let normal = Tensor::<f64>::new(&shape![1], vec![1.0]).unwrap();

        assert!(matches!(scalar.corr(&scalar).unwrap_err(), TensorErrors::RankZero { .. }));
        assert!(matches!(scalar.corr_mt(&scalar).unwrap_err(), TensorErrors::RankZero { .. }));
        // Rank mismatch is checked first if ranks differ
        assert!(matches!(scalar.corr(&normal).unwrap_err(), TensorErrors::RanksDoNotMatch(..)));
    }
    
    #[test]
    fn cannot_corr_conv_mt_scalar_tensors() {
        let scalar = Tensor::<f64>::new(&shape![], vec![1.0]).unwrap();
        let normal = Tensor::<f64>::new(&shape![1], vec![1.0]).unwrap();

        assert!(matches!(scalar.corr_mt(&scalar).unwrap_err(), TensorErrors::RankZero { .. }));
        assert!(matches!(scalar.conv_mt(&scalar).unwrap_err(), TensorErrors::RankZero { .. }));
        assert!(matches!(scalar.corr_axes_mt(&scalar, &HashSet::from([])).unwrap_err(), TensorErrors::RankZero { .. }));
        assert!(matches!(scalar.corr_mt(&normal).unwrap_err(), TensorErrors::RanksDoNotMatch(..)));
    }

    #[test]
    fn invalid_corr_tensors() {
        let t1 = Tensor::<i32>::new(
            &shape![1, 2, 3, 4],
            vec![1; 24],
        ).unwrap();
        let t2 = Tensor::<i32>::new(
            &shape![1, 2, 3],
            vec![1; 6],
        ).unwrap();

        let err = t1.corr(&t2).unwrap_err();
        match err {
            TensorErrors::RanksDoNotMatch(_, _) => {},
            _ => panic!("Incorrect error"),
        };

        let err = t1.corr_mt(&t2).unwrap_err();
        match err {
            TensorErrors::RanksDoNotMatch(_, _) => {},
            _ => panic!("Incorrect error"),
        };
    }

    #[test]
    fn corr_tensors() {
        let t1 = Tensor::<i32>::new(
            &shape![2, 3, 4],
            (0..24).collect(),
        ).unwrap();
        let t2 = Tensor::<i32>::new(
            &shape![1, 2, 3],
            (0..6).collect(),
        ).unwrap();

        let res = t1.corr(&t2).unwrap();
        let ans = t1
            .map(|x| Complex64::new(x as f64, 0.0))
            .fft_conv(&t2.map(|x| Complex64::new(x as f64, 0.0)).flip_mt())
            .unwrap()
            .map(|c| c.re.round().to_i32().unwrap());

        assert_eq!(res, ans);
    }

    #[test]
    fn corr_mat() {
        let m1 = Matrix::<i32>::new(
            4, 6,
            (0..24).collect(),
        ).unwrap();
        let m2 = Matrix::<i32>::new(
            2, 3,
            (0..6).collect(),
        ).unwrap();

        let res = m1.corr(&m2).unwrap();
        let ans = m1
            .map(|x| Complex64::new(x as f64, 0.0))
            .fft_conv(&m2.map(|x| Complex64::new(x as f64, 0.0)).flip_mt())
            .map(|c| c.re.round().to_i32().unwrap());

        assert_eq!(res, ans);
    }

    #[test]
    fn corr_mt_tensors() {
        let t1 = Tensor::<f64>::rand(&shape![10, 20]).clip(-100.0, 100.0);
        let t2 = Tensor::<f64>::rand(&shape![10, 20]).clip(-100.0, 100.0);

        let res_mt = t1.corr_mt(&t2).unwrap();
        let res = t1.corr(&t2).unwrap();

        assert_eq!(res, res_mt);
    }

    #[test]
    fn corr_mt_mat() {
        let m1 = Matrix::<f64>::rand(10, 20).clip(-100.0, 100.0);
        let m2 = Matrix::<f64>::rand(10, 20).clip(-100.0, 100.0);

        let res_mt = m1.corr_mt(&m2);
        let res = m1.corr(&m2);

        assert_eq!(res, res_mt);
    }

    #[test]
    fn corr_complex_mat() {
        let m1 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);
        let m2 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);
        let m3 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);
        let m4 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);

        let in1 = m1.map(|x| Complex64::new(x, 0.0)) + m2.map(|x| Complex64::new(0.0, x));
        let in2 = m3.map(|x| Complex64::new(x, 0.0)) + m4.map(|x| Complex64::new(0.0, x));

        assert_eq!(in1.fft_corr_cols(&in2).unwrap(), in1.fft_conv_cols(&in2.flip_cols_mt()).unwrap());
        assert_eq!(in1.fft_corr_rows(&in2).unwrap(), in1.fft_conv_rows(&in2.flip_rows_mt()).unwrap());
        assert_eq!(in1.fft_corr(&in2), in1.fft_conv(&in2.flip_mt()));
    }

    #[test]
    fn corr_complex_tensor() {
        let t1 = Tensor::<f64>::rand(&shape![10, 2, 5]).clip(-100.0, 100.0);
        let t2 = Tensor::<f64>::rand(&shape![10, 2, 5]).clip(-100.0, 100.0);
        let t3 = Tensor::<f64>::rand(&shape![10, 2, 5]).clip(-100.0, 100.0);
        let t4 = Tensor::<f64>::rand(&shape![10, 2, 5]).clip(-100.0, 100.0);

        let mut axes = HashSet::new();
        axes.insert(1);

        let in1 = t1.map(|x| Complex64::new(x, 0.0)) + t2.map(|x| Complex64::new(0.0, x));
        let in2 = t3.map(|x| Complex64::new(x, 0.0)) + t4.map(|x| Complex64::new(0.0, x));

        assert!(approx_eq!(Tensor<Complex64>, in1.fft_corr_axes(&in2, &axes).unwrap(), in1.fft_conv_axes(&in2.flip_axes_mt(&axes).unwrap(), &axes).unwrap()));
        assert!(approx_eq!(Tensor<Complex64>, in1.fft_corr(&in2).unwrap(), in1.fft_conv(&in2.flip_mt()).unwrap(), epsilon = 1e-10));
    }

    #[test]
    fn conv_mat() {
        let m1 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);
        let m2 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);
        let m3 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);
        let m4 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);

        let in1 = m1.map(|x| Complex64::new(x, 0.0)) + m2.map(|x| Complex64::new(0.0, x));
        let in2 = m3.map(|x| Complex64::new(x, 0.0)) + m4.map(|x| Complex64::new(0.0, x));

        assert!(approx_eq!(Matrix<Complex64>, in1.conv(&in2).unwrap(), in1.fft_conv(&in2), epsilon = 1e-10));
    }

    #[test]
    fn conv_tensor() {
        let t1 = Tensor::<f64>::rand(&shape![10, 2, 5]).clip(-100.0, 100.0);
        let t2 = Tensor::<f64>::rand(&shape![10, 2, 5]).clip(-100.0, 100.0);
        let t3 = Tensor::<f64>::rand(&shape![10, 2, 5]).clip(-100.0, 100.0);
        let t4 = Tensor::<f64>::rand(&shape![10, 2, 5]).clip(-100.0, 100.0);

        let in1 = t1.map(|x| Complex64::new(x, 0.0)) + t2.map(|x| Complex64::new(0.0, x));
        let in2 = t3.map(|x| Complex64::new(x, 0.0)) + t4.map(|x| Complex64::new(0.0, x));

        assert!(approx_eq!(Tensor<Complex64>, in1.conv(&in2).unwrap(), in1.fft_conv(&in2).unwrap(), epsilon = 1e-10));
    }

    #[test]
    fn conv_mat_mt() {
        let m1 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);
        let m2 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);

        let res_mt = m1.conv_mt(&m2);
        let res = m1.conv(&m2);

        assert_eq!(res_mt, res);
    }

    #[test]
    fn conv_tensor_mt() {
        let t1 = Tensor::<f64>::rand(&shape![10, 20, 5]).clip(-100.0, 100.0);
        let t2 = Tensor::<f64>::rand(&shape![10, 10, 1]).clip(-100.0, 100.0);

        let res_mt = t1.conv_mt(&t2).unwrap();
        let res = t1.conv(&t2).unwrap();

        assert_eq!(res_mt, res);
    }

    #[test]
    fn corr_axes_tensor() {
        let t1 = Tensor::<f64>::rand(&shape![10, 2, 5]).clip(-100.0, 100.0);
        let t2 = Tensor::<f64>::rand(&shape![10, 2, 5]).clip(-100.0, 100.0);
        let t3 = Tensor::<f64>::rand(&shape![10, 2, 5]).clip(-100.0, 100.0);
        let t4 = Tensor::<f64>::rand(&shape![10, 2, 5]).clip(-100.0, 100.0);

        let in1 = t1.map(|x| Complex64::new(x, 0.0)) + t2.map(|x| Complex64::new(0.0, x));
        let in2 = t3.map(|x| Complex64::new(x, 0.0)) + t4.map(|x| Complex64::new(0.0, x));

        let mut axes = HashSet::new();
        axes.insert(0);
        axes.insert(2);

        let res = in1.corr_axes(&in2, &axes).unwrap();
        let ans = in1.fft_corr_axes(&in2, &axes).unwrap();

        assert!(approx_eq!(Tensor<Complex64>, ans, res, epsilon = 1e-10));
    }

    #[test]
    fn corr_axes_mt_tensor() {
        let t1 = Tensor::<f64>::rand(&shape![10, 2, 5]).clip(-100.0, 100.0);
        let t2 = Tensor::<f64>::rand(&shape![10, 2, 5]).clip(-100.0, 100.0);
        let t3 = Tensor::<f64>::rand(&shape![10, 2, 5]).clip(-100.0, 100.0);
        let t4 = Tensor::<f64>::rand(&shape![10, 2, 5]).clip(-100.0, 100.0);

        let in1 = t1.map(|x| Complex64::new(x, 0.0)) + t2.map(|x| Complex64::new(0.0, x));
        let in2 = t3.map(|x| Complex64::new(x, 0.0)) + t4.map(|x| Complex64::new(0.0, x));

        let mut axes = HashSet::new();
        axes.insert(0);
        axes.insert(2);

        let res = in1.corr_axes_mt(&in2, &axes).unwrap();
        let ans = in1.fft_corr_axes(&in2, &axes).unwrap();

        assert!(approx_eq!(Tensor<Complex64>, ans, res, epsilon = 1e-10));
    }

    #[test]
    fn corr_axes_mat() {
        let m1 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);
        let m2 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);
        let m3 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);
        let m4 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);

        let in1 = m1.map(|x| Complex64::new(x, 0.0)) + m2.map(|x| Complex64::new(0.0, x));
        let in2 = m3.map(|x| Complex64::new(x, 0.0)) + m4.map(|x| Complex64::new(0.0, x));

        assert!(approx_eq!(Matrix<Complex64>, in1.fft_corr_rows(&in2).unwrap(), in1.corr_rows(&in2).unwrap(), epsilon = 1e-10));
        assert!(approx_eq!(Matrix<Complex64>, in1.fft_corr_cols(&in2).unwrap(), in1.corr_cols(&in2).unwrap(), epsilon = 1e-10));
        assert!(approx_eq!(Matrix<Complex64>, in1.fft_corr(&in2), in1.corr(&in2).unwrap(), epsilon = 1e-10));
    }

    #[test]
    fn corr_axes_mt_mat() {
        let m1 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);
        let m2 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);
        let m3 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);
        let m4 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);

        let in1 = m1.map(|x| Complex64::new(x, 0.0)) + m2.map(|x| Complex64::new(0.0, x));
        let in2 = m3.map(|x| Complex64::new(x, 0.0)) + m4.map(|x| Complex64::new(0.0, x));

        assert!(approx_eq!(Matrix<Complex64>, in1.fft_corr_rows(&in2).unwrap(), in1.corr_rows_mt(&in2).unwrap(), epsilon = 1e-10));
        assert!(approx_eq!(Matrix<Complex64>, in1.fft_corr_cols(&in2).unwrap(), in1.corr_cols_mt(&in2).unwrap(), epsilon = 1e-10));
        assert!(approx_eq!(Matrix<Complex64>, in1.fft_corr(&in2), in1.corr_mt(&in2).unwrap(), epsilon = 1e-10));
    }

    #[test]
    fn invalid_corr_axes_tensors_ranks_do_not_match() {
        let t1 = Tensor::<f64>::rand(&shape![10, 2, 5]).clip(-100.0, 100.0);
        let t2 = Tensor::<f64>::rand(&shape![10, 10]).clip(-100.0, 100.0);

        let err = t1.corr_axes(&t2, &HashSet::new()).unwrap_err();
        match err {
            TensorErrors::RanksDoNotMatch(..) => {}
            _ => panic!("Incorrect error")
        }

        let err = t1.corr_axes_mt(&t2, &HashSet::new()).unwrap_err();
        match err {
            TensorErrors::RanksDoNotMatch(..) => {}
            _ => panic!("Incorrect error")
        }
    }

    #[test]
    fn invalid_corr_tensor_incompatible_shapes() {
        let t1 = Tensor::<f64>::rand(&shape![10, 10, 10]).clip(-100.0, 100.0);
        let t2 = Tensor::<f64>::rand(&shape![10, 20, 10]).clip(-100.0, 100.0);

        let err = t1.corr_axes(&t2, &HashSet::new()).unwrap_err();
        match err {
            TensorErrors::ShapesIncompatible => {},
            _ => panic!("Incorrect error"),
        };

        let err = t1.corr_axes_mt(&t2, &HashSet::new()).unwrap_err();
        match err {
            TensorErrors::ShapesIncompatible => {},
            _ => panic!("Incorrect error"),
        };
    }

    #[test]
    fn invalid_corr_mat_incompatible_shapes() {
        let m1 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);
        let m2 = Matrix::<f64>::rand(10, 20).clip(-100.0, 100.0);
        let m3 = Matrix::<f64>::rand(20, 10).clip(-100.0, 100.0);

        let err = m1.corr_cols(&m2).unwrap_err();
        match err {
            TensorErrors::ShapesIncompatible => {},
            _ => panic!("Incorrect error"),
        };

        let err = m1.corr_cols_mt(&m2).unwrap_err();
        match err {
            TensorErrors::ShapesIncompatible => {},
            _ => panic!("Incorrect error"),
        };

        let err = m1.corr_rows(&m3).unwrap_err();
        match err {
            TensorErrors::ShapesIncompatible => {},
            _ => panic!("Incorrect error"),
        };

        let err = m1.corr_rows_mt(&m3).unwrap_err();
        match err {
            TensorErrors::ShapesIncompatible => {},
            _ => panic!("Incorrect error"),
        };
    }

    #[test]
    fn invalid_corr_axes_out_of_bounds() {
        let t1 = Tensor::<f64>::rand(&shape![10, 10, 10]).clip(-100.0, 100.0);
        let t2 = Tensor::<f64>::rand(&shape![10, 10, 10]).clip(-100.0, 100.0);
        let mut axes = HashSet::new();

        axes.insert(5);

        let err = t1.corr_axes(&t2, &axes).unwrap_err();
        match err {
            TensorErrors::AxisOutOfBounds { .. } => {},
            _ => panic!("Incorrect error"),
        };

        let err = t1.corr_axes_mt(&t2, &axes).unwrap_err();
        match err {
            TensorErrors::AxisOutOfBounds { .. } => {},
            _ => panic!("Incorrect error"),
        };
    }
}