#[cfg(test)]
mod fft_ops_tests {
    use crate::definitions::matrix::Matrix;
    use crate::definitions::shape::Shape;
    use crate::definitions::tensor::Tensor;
    use crate::shape;
    use float_cmp::approx_eq;
    use num::complex::Complex64;
    use num::FromPrimitive;
    use std::collections::HashSet;
    use crate::definitions::errors::TensorErrors;

    #[test]
    fn mat_fft() {
        let m1 = Matrix::<Complex64>::new(
            2, 3,
            vec![
                Complex64::new(1.0, 0.0), Complex64::new(2.0, -1.0), Complex64::new(3.0, 2.0),
                Complex64::new(0.0, 1.0), Complex64::new(1.0, -1.0), Complex64::new(2.0, -3.0),
            ],
        ).unwrap();
        let ans_cols = Matrix::<Complex64>::new(
            2, 3,
            vec![
                Complex64::new(1.0, 1.0), Complex64::new(3.0, -2.0), Complex64::new(5.0, -1.0),
                Complex64::new(1.0, -1.0), Complex64::new(1.0, 0.0), Complex64::new(1.0, 5.0),
            ],
        ).unwrap();
        let ans_rows = Matrix::<Complex64>::new(
            2, 3,
            vec![
                Complex64::new(6.0, 1.0), Complex64::new(-4.09807621, 0.3660254), Complex64::new(1.09807621, -1.3660254),
                Complex64::new(3.0, -3.0), Complex64::new(0.23205081, 3.8660254), Complex64::new(-3.23205081, 2.1339746),
            ],
        ).unwrap();
        let ans_entire = Matrix::<Complex64>::new(
            2, 3,
            vec![
                Complex64::new(9.0, -2.0), Complex64::new(-3.8660254, 4.23205081), Complex64::new(-2.1339746, 0.76794919),
                Complex64::new(3.0, 4.0), Complex64::new(-4.33012702, -3.5), Complex64::new(4.33012702, -3.5),
            ],
        ).unwrap();

        let res_rows = m1.fft_rows();
        let res_cols = m1.fft_cols();
        let res_entire = m1.fft();

        for i in 0..2 {
            for j in 0..3 {
                assert!(approx_eq!(f64, res_rows[(i, j)].re, ans_rows[(i, j)].re, epsilon = 1e-5));
                assert!(approx_eq!(f64, res_rows[(i, j)].im, ans_rows[(i, j)].im, epsilon = 1e-5));
                assert!(approx_eq!(f64, res_cols[(i, j)].re, ans_cols[(i, j)].re, epsilon = 1e-5));
                assert!(approx_eq!(f64, res_cols[(i, j)].im, ans_cols[(i, j)].im, epsilon = 1e-5));
                assert!(approx_eq!(f64, res_entire[(i, j)].re, ans_entire[(i, j)].re, epsilon = 1e-5));
                assert!(approx_eq!(f64, res_entire[(i, j)].im, ans_entire[(i, j)].im, epsilon = 1e-5));
            }
        }
    }

    #[test]
    fn mat_ifft() {
        let m1 = Matrix::<Complex64>::new(
            2, 3,
            vec![
                Complex64::new(9.0, -2.0), Complex64::new(-3.8660254, 4.23205081), Complex64::new(-2.1339746, 0.76794919),
                Complex64::new(3.0, 4.0), Complex64::new(-4.33012702, -3.5), Complex64::new(4.33012702, -3.5),
            ],
        ).unwrap();

        let ans_cols = Matrix::<Complex64>::new(
            2, 3,
            vec![
                Complex64::new(6.0, 1.0), Complex64::new(-4.09807621, 0.3660254), Complex64::new(1.09807621, -1.3660254),
                Complex64::new(3.0, -3.0), Complex64::new(0.23205081, 3.8660254), Complex64::new(-3.23205081, 2.1339746),
            ],
        ).unwrap();
        let ans_rows = Matrix::<Complex64>::new(
            2, 3,
            vec![
                Complex64::new(1.0, 1.0), Complex64::new(3.0, -2.0), Complex64::new(5.0, -1.0),
                Complex64::new(1.0, -1.0), Complex64::new(1.0, 0.0), Complex64::new(1.0, 5.0),
            ],
        ).unwrap();
        let ans_entire = Matrix::<Complex64>::new(
            2, 3,
            vec![
                Complex64::new(1.0, 0.0), Complex64::new(2.0, -1.0), Complex64::new(3.0, 2.0),
                Complex64::new(0.0, 1.0), Complex64::new(1.0, -1.0), Complex64::new(2.0, -3.0),
            ],
        ).unwrap();

        let res_rows = m1.ifft_rows();
        let res_cols = m1.ifft_cols();
        let res_entire = m1.ifft();

        for i in 0..2 {
            for j in 0..3 {
                assert!(approx_eq!(f64, res_rows[(i, j)].re, ans_rows[(i, j)].re, epsilon = 1e-5));
                assert!(approx_eq!(f64, res_rows[(i, j)].im, ans_rows[(i, j)].im, epsilon = 1e-5));
                assert!(approx_eq!(f64, res_cols[(i, j)].re, ans_cols[(i, j)].re, epsilon = 1e-5));
                assert!(approx_eq!(f64, res_cols[(i, j)].im, ans_cols[(i, j)].im, epsilon = 1e-5));
                assert!(approx_eq!(f64, res_entire[(i, j)].re, ans_entire[(i, j)].re, epsilon = 1e-5));
                assert!(approx_eq!(f64, res_entire[(i, j)].im, ans_entire[(i, j)].im, epsilon = 1e-5));
            }
        }
    }

    #[test]
    fn tensor_fft() {
        let t1 = Tensor::<Complex64>::new(
            &shape![3, 2, 4],
            (0..24).map(|i| Complex64::new(i as f64, i as f64 * if i % 2 == 0 { 1.0 } else { -1.0 })).collect(),
        ).unwrap();

        let ans_axes_1_2 = Tensor::<Complex64>::new(
            &shape![3, 2, 4],
            vec![
                Complex64::new(28.0, -4.0), Complex64::new(0.0, 0.0), Complex64::new(-4.0, 28.0), Complex64::new(-8.0, -8.0),
                Complex64::new(-16.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, -16.0), Complex64::new(0.0, 0.0),

                Complex64::new(92.0, -4.0), Complex64::new(0.0, 0.0), Complex64::new(-4.0, 92.0), Complex64::new(-8.0, -8.0),
                Complex64::new(-16.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, -16.0), Complex64::new(0.0, 0.0),

                Complex64::new(156.0, -4.0), Complex64::new(0.0, 0.0), Complex64::new(-4.0, 156.0), Complex64::new(-8.0, -8.0),
                Complex64::new(-16.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, -16.0), Complex64::new(0.0, 0.0),
            ],
        ).unwrap();
        let ans_axis_0 = Tensor::<Complex64>::new(
            &shape![3, 2, 4],
            vec![
                Complex64::new(24.0, 24.0), Complex64::new(27.0, -27.0), Complex64::new(30.0, 30.0), Complex64::new(33.0, -33.0),
                Complex64::new(36.0, 36.0), Complex64::new(39.0, -39.0), Complex64::new(42.0, 42.0), Complex64::new(45.0, -45.0),

                Complex64::new(-18.92820323, -5.07179677), Complex64::new(-5.07179677, 18.92820323), Complex64::new(-18.92820323,-5.07179677), Complex64::new(-5.07179677, 18.92820323),
                Complex64::new(-18.92820323, -5.07179677), Complex64::new(-5.07179677, 18.92820323), Complex64::new(-18.92820323,-5.07179677), Complex64::new(-5.07179677, 18.92820323),

                Complex64::new(-5.07179677, -18.92820323), Complex64::new(-18.92820323, 5.07179677), Complex64::new(-5.07179677, -18.92820323), Complex64::new(-18.92820323, 5.07179677),
                Complex64::new(-5.07179677, -18.92820323), Complex64::new(-18.92820323, 5.07179677), Complex64::new(-5.07179677, -18.92820323), Complex64::new(-18.92820323, 5.07179677),
            ],
        ).unwrap();
        let ans_entire = Tensor::<Complex64>::new(
            &shape![3, 2, 4],
            vec![
                Complex64::new(276.0, -12.0), Complex64::new(0.0, 0.0), Complex64::new(-12.0, 276.0), Complex64::new(-24.0, -24.0),
                Complex64::new(-48.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, -48.0), Complex64::new(0.0, 0.0),

                Complex64::new(-96.0, 55.42562584), Complex64::new(0.0, 0.0), Complex64::new(-55.42562584, -96.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),

                Complex64::new(-96.0, -55.42562584), Complex64::new(0.0, 0.0), Complex64::new(55.42562584, -96.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            ],
        ).unwrap();

        let res_axes_1_2 = t1.fft_axes(&HashSet::from([1, 2])).unwrap();
        let res_axis_0 = t1.fft_single_axis(0).unwrap();
        let res_entire = t1.fft();

        for i in 0..3 {
            for j in 0..2 {
                for k in 0..4 {
                    assert!(approx_eq!(f64, res_axes_1_2[&[i, j, k]].re, ans_axes_1_2[&[i, j, k]].re, epsilon = 1e-5));
                    assert!(approx_eq!(f64, res_axes_1_2[&[i, j, k]].im, ans_axes_1_2[&[i, j, k]].im, epsilon = 1e-5));
                    assert!(approx_eq!(f64, res_axis_0[&[i, j, k]].re, ans_axis_0[&[i, j, k]].re, epsilon = 1e-5));
                    assert!(approx_eq!(f64, res_axis_0[&[i, j, k]].im, ans_axis_0[&[i, j, k]].im, epsilon = 1e-5));
                    assert!(approx_eq!(f64, res_entire[&[i, j, k]].re, ans_entire[&[i, j, k]].re, epsilon = 1e-5));
                    assert!(approx_eq!(f64, res_entire[&[i, j, k]].im, ans_entire[&[i, j, k]].im, epsilon = 1e-5));
                }
            }
        }
    }

    #[test]
    fn tensor_ifft() {
        let ans_entire = Tensor::<Complex64>::new(
            &shape![3, 2, 4],
            (0..24).map(|i| Complex64::new(i as f64, i as f64 * if i % 2 == 0 { 1.0 } else { -1.0 })).collect(),
        ).unwrap();
        let ans_axis_0 = Tensor::<Complex64>::new(
            &shape![3, 2, 4],
            vec![
                Complex64::new(28.0, -4.0), Complex64::new(0.0, 0.0), Complex64::new(-4.0, 28.0), Complex64::new(-8.0, -8.0),
                Complex64::new(-16.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, -16.0), Complex64::new(0.0, 0.0),

                Complex64::new(92.0, -4.0), Complex64::new(0.0, 0.0), Complex64::new(-4.0, 92.0), Complex64::new(-8.0, -8.0),
                Complex64::new(-16.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, -16.0), Complex64::new(0.0, 0.0),

                Complex64::new(156.0, -4.0), Complex64::new(0.0, 0.0), Complex64::new(-4.0, 156.0), Complex64::new(-8.0, -8.0),
                Complex64::new(-16.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, -16.0), Complex64::new(0.0, 0.0),
            ],
        ).unwrap();
        let ans_axes_1_2 = Tensor::<Complex64>::new(
            &shape![3, 2, 4],
            vec![
                Complex64::new(24.0, 24.0), Complex64::new(27.0, -27.0), Complex64::new(30.0, 30.0), Complex64::new(33.0, -33.0),
                Complex64::new(36.0, 36.0), Complex64::new(39.0, -39.0), Complex64::new(42.0, 42.0), Complex64::new(45.0, -45.0),

                Complex64::new(-18.92820323, -5.07179677), Complex64::new(-5.07179677, 18.92820323), Complex64::new(-18.92820323,-5.07179677), Complex64::new(-5.07179677, 18.92820323),
                Complex64::new(-18.92820323, -5.07179677), Complex64::new(-5.07179677, 18.92820323), Complex64::new(-18.92820323,-5.07179677), Complex64::new(-5.07179677, 18.92820323),

                Complex64::new(-5.07179677, -18.92820323), Complex64::new(-18.92820323, 5.07179677), Complex64::new(-5.07179677, -18.92820323), Complex64::new(-18.92820323, 5.07179677),
                Complex64::new(-5.07179677, -18.92820323), Complex64::new(-18.92820323, 5.07179677), Complex64::new(-5.07179677, -18.92820323), Complex64::new(-18.92820323, 5.07179677),
            ],
        ).unwrap();
        let t1 = Tensor::<Complex64>::new(
            &shape![3, 2, 4],
            vec![
                Complex64::new(276.0, -12.0), Complex64::new(0.0, 0.0), Complex64::new(-12.0, 276.0), Complex64::new(-24.0, -24.0),
                Complex64::new(-48.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, -48.0), Complex64::new(0.0, 0.0),

                Complex64::new(-96.0, 55.42562584), Complex64::new(0.0, 0.0), Complex64::new(-55.42562584, -96.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),

                Complex64::new(-96.0, -55.42562584), Complex64::new(0.0, 0.0), Complex64::new(55.42562584, -96.0), Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0),
            ],
        ).unwrap();

        let res_axes_1_2 = t1.ifft_axes(&HashSet::from([1, 2])).unwrap();
        let res_axis_0 = t1.ifft_single_axis(0).unwrap();
        let res_entire = t1.ifft();

        for i in 0..3 {
            for j in 0..2 {
                for k in 0..4 {
                    assert!(approx_eq!(f64, res_axes_1_2[&[i, j, k]].re, ans_axes_1_2[&[i, j, k]].re, epsilon = 1e-5));
                    assert!(approx_eq!(f64, res_axes_1_2[&[i, j, k]].im, ans_axes_1_2[&[i, j, k]].im, epsilon = 1e-5));
                    assert!(approx_eq!(f64, res_axis_0[&[i, j, k]].re, ans_axis_0[&[i, j, k]].re, epsilon = 1e-5));
                    assert!(approx_eq!(f64, res_axis_0[&[i, j, k]].im, ans_axis_0[&[i, j, k]].im, epsilon = 1e-5));
                    assert!(approx_eq!(f64, res_entire[&[i, j, k]].re, ans_entire[&[i, j, k]].re, epsilon = 1e-5));
                    assert!(approx_eq!(f64, res_entire[&[i, j, k]].im, ans_entire[&[i, j, k]].im, epsilon = 1e-5));
                }
            }
        }
    }

    #[test]
    fn invalid_tensor_fft_conv_axes_different_ranks() {
        let t1 = Tensor::<Complex64>::new(
            &shape![1, 2, 3],
            (0..6).map(|x| Complex64 { re: x as f64, im: 0.0 }).collect(),
        ).unwrap();
        let t2 = Tensor::<Complex64>::new(
            &shape![2, 3],
            (0..6).map(|x| Complex64 { re: x as f64, im: 0.0 }).collect(),
        ).unwrap();

        let err = t1.fft_conv_axes(&t2, &HashSet::new()).unwrap_err();
        match err {
            TensorErrors::RanksDoNotMatch(_, _) => {},
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn invalid_tensor_fft_ifft_axis_out_of_bound() {
        let t1 = Tensor::<f64>::rand(&shape![4, 2, 3]);

        let err = t1.clone().map(Complex64::from).fft_single_axis(4).unwrap_err();
        match err {
            TensorErrors::AxisOutOfBounds { .. } => {},
            _ => panic!("Incorrect error"),
        }

        let err = t1.clone().map(Complex64::from).ifft_single_axis(4).unwrap_err();
        match err {
            TensorErrors::AxisOutOfBounds { .. } => {},
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn invalid_tensor_fft_conv_axes_incompatible_shapes() {
        let t1 = Tensor::<Complex64>::new(
            &shape![1, 2, 3],
            (0..6).map(|x| Complex64 { re: x as f64, im: 0.0 }).collect(),
        ).unwrap();
        let t2 = Tensor::<Complex64>::new(
            &shape![1, 3, 2],
            (0..6).map(|x| Complex64 { re: x as f64, im: 0.0 }).collect(),
        ).unwrap();
        let mut axes = HashSet::new();
        axes.insert(0);

        let err = t1.fft_conv_axes(&t2, &axes).unwrap_err();
        match err {
            TensorErrors::ShapesIncompatible => {},
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn tensor_fft_conv_axes() {
        let t1 = Tensor::<Complex64>::new(
            &shape![4, 2, 3],
            (0..24).map(|x| Complex64 { re: x as f64, im: x as f64 }).collect(),
        ).unwrap();
        let t2 = Tensor::<Complex64>::new(
            &shape![1, 2, 3],
            (0..6).map(|x| Complex64 { re: x as f64, im: -x as f64 }).collect(),
        ).unwrap();
        let mut axes = HashSet::new();
        axes.insert(0);
        axes.insert(2);

        let res = t1.fft_conv_axes(&t2, &axes).unwrap();

        let ans = Tensor::<Complex64>::new(
            &shape![4, 2, 5],
            vec![
                0, 0, 2, 8, 8,
                18, 48, 92, 80, 50,
                0, 12, 38, 44, 32,
                54, 132, 236, 188, 110,
                0, 24, 74, 80, 56,
                90, 216, 380, 296, 170,
                0, 36, 110, 116, 80,
                126, 300, 524, 404, 230,
            ].iter().map(|x| Complex64::from_f64(*x as f64).unwrap()).collect()
        ).unwrap();

        assert!(approx_eq!(Tensor<Complex64>, res, ans, epsilon = 1e-10));
    }

    #[test]
    fn invalid_tensor_fft_conv_diff_ranks() {
        let t1 = Tensor::<Complex64>::new(
            &shape![1, 2, 3],
            (0..6).map(|x| Complex64 { re: x as f64, im: 0.0 }).collect(),
        ).unwrap();
        let t2 = Tensor::<Complex64>::new(
            &shape![2, 3],
            (0..6).map(|x| Complex64 { re: x as f64, im: 0.0 }).collect(),
        ).unwrap();

        let err = t1.fft_conv(&t2).unwrap_err();
        match err {
            TensorErrors::RanksDoNotMatch(_, _) => {},
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn tensor_fft_conv() {
        let t1 = Tensor::<Complex64>::new(
            &shape![4, 2, 3],
            (0..24).map(|x| Complex64 { re: x as f64, im: x as f64 }).collect(),
        ).unwrap();
        let t2 = Tensor::<Complex64>::new(
            &shape![1, 2, 3],
            (0..6).map(|x| Complex64 { re: x as f64, im: -x as f64 }).collect(),
        ).unwrap();

        let res = t1.fft_conv(&t2).unwrap();

        let ans = Tensor::<Complex64>::new(
            &shape![4, 3, 5],
            vec![
                0, 0, 2, 8, 8,
                0, 12, 40, 52, 40,
                18, 48, 92, 80, 50,

                0, 12, 38, 44, 32,
                36, 108, 220, 196, 124,
                54, 132, 236, 188, 110,

                0, 24, 74, 80, 56,
                72, 204, 400, 340, 208,
                90, 216, 380, 296, 170,

                0, 36, 110, 116, 80,
                108, 300, 580, 484, 292,
                126, 300, 524, 404, 230,
            ].iter().map(|x| Complex64::from_f64(*x as f64).unwrap()).collect()
        ).unwrap();

        assert!(approx_eq!(Tensor<Complex64>, res, ans, epsilon = 1e-10));
    }

    #[test]
    fn invalid_fft_mat_conv_cols_different_cols() {
        let m1 = Matrix::<Complex64>::zeros(5, 1);
        let m2 = Matrix::<Complex64>::zeros(3, 2);

        let err = m1.fft_conv_cols(&m2).unwrap_err();
        match err {
            TensorErrors::ShapesIncompatible => {},
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn invalid_fft_mat_conv_rows_different_rows() {
        let m1 = Matrix::<Complex64>::zeros(2, 2);
        let m2 = Matrix::<Complex64>::zeros(1, 1);

        let err = m1.fft_conv_rows(&m2).unwrap_err();
        match err {
            TensorErrors::ShapesIncompatible => {},
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn fft_mat_conv_rows() {
        let m1 = Matrix::<Complex64>::new(
            3, 2,
            (0..6).map(|x| Complex64 { re: x as f64, im: 0.0 }).collect(),
        ).unwrap();
        let m2 = Matrix::<Complex64>::new(
            3, 5,
            (0..15).map(|x| Complex64 { re: 0.0, im: x as f64 }).collect(),
        ).unwrap();

        let res = m1.fft_conv_rows(&m2).unwrap();

        let ans = Matrix::<Complex64>::new(
            3, 6,
            vec![
                0, 0, 1, 2, 3, 4,
                10, 27, 32, 37, 42, 27,
                40, 94, 103, 112, 121, 70,
            ].iter().map(|x| Complex64 { re: 0.0, im: *x as f64 }).collect(),
        ).unwrap();

        assert!(approx_eq!(Matrix<Complex64>, ans, res, epsilon = 1e-10));
    }

    #[test]
    fn fft_mat_conv_cols() {
        let m1 = Matrix::<Complex64>::new(
            2, 3,
            (0..6).map(|x| Complex64 { re: x as f64, im: 0.0 }).collect(),
        ).unwrap();
        let m2 = Matrix::<Complex64>::new(
            5, 3,
            (0..15).map(|x| Complex64 { re: 0.0, im: x as f64 }).collect(),
        ).unwrap();

        let res = m1.fft_conv_cols(&m2).unwrap();

        let ans = Matrix::<Complex64>::new(
            6, 3,
            vec![
                0, 1, 4,
                0, 8, 20,
                9, 23, 41,
                18, 38, 62,
                27, 53, 83,
                36, 52, 70,
            ].iter().map(|x| Complex64 { re: 0.0, im: *x as f64 }).collect(),
        ).unwrap();

        assert!(approx_eq!(Matrix<Complex64>, ans, res, epsilon = 1e-10));
    }

    #[test]
    fn fft_mat_conv() {
        let m1 = Matrix::<Complex64>::new(
            2, 3,
            (0..6).map(|x| Complex64 { re: x as f64, im: 0.0 }).collect(),
        ).unwrap();
        let m2 = Matrix::<Complex64>::new(
            5, 3,
            (0..15).map(|x| Complex64 { re: 0.0, im: x as f64 }).collect(),
        ).unwrap();

        let res = m1.fft_conv(&m2);

        let ans = Matrix::<Complex64>::new(
            6, 5,
            vec![
                0, 0, 1, 4, 4,
                0, 6, 20, 26, 20,
                9, 30, 65, 62, 41,
                18, 54, 110, 98, 62,
                27, 78, 155, 134, 83,
                36, 87, 154, 121, 70,
            ].iter().map(|x| Complex64 { re: 0.0, im: *x as f64 }).collect(),
        ).unwrap();

        assert!(approx_eq!(Matrix<Complex64>, ans, res, epsilon = 1e-10));
    }
}