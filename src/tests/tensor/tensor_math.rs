#[cfg(test)]
mod tensor_math_tests {
    use std::collections::HashSet;
    use crate::tensor::tensor::{Matrix, Tensor};
    use crate::tensor::tensor::{Shape, TensorErrors};
    use crate::tensor::tensor_math::{bluestein_fft_vec, det_slow, fft_vec, gaussian_pdf_cov_mat, gaussian_pdf_multi_sigma, gaussian_pdf_single_sigma, gaussian_sample, identity, ifft_vec, inv_slow, pool_avg, pool_avg_mat, pool_max, pool_max_mat, pool_min, pool_min_mat, pool_sum, pool_sum_mat, radix_2_fft_vec, solve_cubic, solve_quadratic, solve_quartic, Transpose};
    use crate::{shape, transpose};
    use float_cmp::{approx_eq, assert_approx_eq, ApproxEq, F64Margin, FloatMargin};
    use num::complex::{Complex64, ComplexFloat};
    use std::f64::consts::PI;
    use std::ops::Add;
    use num::{FromPrimitive, ToPrimitive};
    use num::traits::real::Real;

    #[test]
    fn add_tensors() {
        let t1 = Tensor::<i32>::new(&shape![2, 3], vec![0, 0, 0, 2, 3, -1]).unwrap();
        let t2 = Tensor::<i32>::new(&shape![2, 3], vec![1, 1, 1, 2, 4, -10]).unwrap();
        let ans = Tensor::<i32>::new(&shape![2, 3], vec![1, 1, 1, 4, 7, -11]).unwrap();
        assert_eq!(ans, t1 + t2);
    }

    #[test]
    fn mat_operators() {
        let m1 = -identity::<Complex64>(2) / <f64 as Into<Complex64>>::into(2.0);
        let m2 = identity::<Complex64>(2) * (Complex64::ONE + Complex64::I);
        let ans = Matrix::new(2, 2, vec![Complex64 { re: 0.5, im: 1.0 }, Complex64 { re: 0.0, im: 0.0 }, Complex64 { re: 0.0, im: 0.0 }, Complex64 { re: 0.5, im: 1.0 }]).unwrap();

        assert_eq!(m1 + m2, ans);
    }

    #[test]
    fn subtract_tensors() {
        let t1 = Tensor::<i32>::new(&shape![2, 3], vec![0, 0, 0, 2, 3, -1]).unwrap();
        let t2 = Tensor::<i32>::new(&shape![2, 3], vec![1, 1, 1, 2, 4, -10]).unwrap();
        let ans = Tensor::<i32>::new(&shape![2, 3], vec![1, 1, 1, 0, 1, -9]).unwrap();
        assert_eq!(ans, t2 - t1);
    }

    #[test]
    fn multiply_tensors() {
        let t1 = Tensor::<i32>::new(&shape![2, 3], vec![0, 0, 0, 2, 3, -1]).unwrap();
        let t2 = &Tensor::<i32>::new(&shape![2, 3], vec![1, 1, 1, 2, 4, -10]).unwrap();
        let ans = Tensor::<i32>::new(&shape![2, 3], vec![0, 0, 0, 4, 12, 10]).unwrap();
        assert_eq!(ans, t1 * t2);
    }

    #[test]
    fn divide_tensors() {
        let t1 = Tensor::<i32>::new(&shape![2, 3], vec![0, 0, 0, 2, 5, -1]).unwrap();
        let t2 = &Tensor::<i32>::new(&shape![2, 3], vec![1, 1, 1, 2, 4, -10]).unwrap();
        let ans = Tensor::<i32>::new(&shape![2, 3], vec![0, 0, 0, 1, 1, 0]).unwrap();
        assert_eq!(ans, t1 / t2);
    }

    #[test]
    #[should_panic]
    fn incompatible_shapes_for_elementwise_bin_op() {
        let t1 = Tensor::<i32>::rand(&shape![2, 3]);
        let t2 = Tensor::<i32>::rand(&shape![2, 2]);

        let _ = t1 + t2;
    }

    #[test]
    #[should_panic]
    fn incompatible_shapes_for_elementwise_bin_op_mat() {
        let m1 = Matrix::<i32>::rand(2, 2);
        let m2 = Matrix::<i32>::rand(3, 3);

        let _ = m1 + m2;
    }

    #[test]
    #[should_panic]
    fn divide_by_zero() {
        let t1 = Tensor::<i32>::rand(&shape![2, 2]);
        let mut t2 = Tensor::<i32>::rand(&shape![2, 2]);
        t2[&[1, 1]] = 0;
        let _ = t1 / t2;
    }

    #[test]
    #[should_panic]
    fn divide_mat_by_zero() {
        let m1 = Matrix::new(2, 2, vec![1, 2, 3, 4]).unwrap();
        let _ = m1 / 0;
    }

    #[test]
    fn single_element_implementations() {
        let t1 = Tensor::<i32>::new(&shape![2, 3], vec![0, 1, 2, -1, 4, -10]).unwrap();
        let val = 5;

        let ans_add = Tensor::<i32>::new(&shape![2, 3], vec![5, 6, 7, 4, 9, -5]).unwrap();
        let ans_sub = Tensor::<i32>::new(&shape![2, 3], vec![-5, -4, -3, -6, -1, -15]).unwrap();
        let ans_mul = Tensor::<i32>::new(&shape![2, 3], vec![0, 5, 10, -5, 20, -50]).unwrap();
        let ans_div = Tensor::<i32>::new(&shape![2, 3], vec![0, 0, 0, 0, 0, -2]).unwrap();
        assert_eq!(ans_add, &t1 + val);
        assert_eq!(ans_sub, t1.clone() - &val);
        assert_eq!(ans_mul, &t1 * val);
        assert_eq!(ans_div, t1 / val);
    }

    #[test]
    fn unary_minus() {
        let t1 = Tensor::<i32>::new(&shape![2, 3], vec![1, 3, -4, 4, 5, 2]).unwrap();
        let ans = Tensor::<i32>::new(&shape![2, 3], vec![-1, -3, 4, -4, -5, -2]).unwrap();

        assert_eq!(ans, -t1);
    }

    #[test]
    fn transpose() {
        let t1 = Tensor::<i32>::new(&shape![2, 3, 4], (0..24).collect()).unwrap();
        let mut t2 = Tensor::<i32>::new(&shape![2, 3], (0..6).collect()).unwrap();
        let transposed_t1 = t1
            .clone()
            .transpose(&transpose![0, 2, 1])
            .unwrap();
        t2.transpose_in_place(&Transpose::new(&vec![1, 0]).unwrap())
            .unwrap();
        let transposed_t2 = t2.clone();
        let ans1 = Tensor::<i32>::new(
            &shape![2, 4, 3],
            vec![
                0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19,
                23,
            ],
        )
        .unwrap();
        let ans2 = Tensor::<i32>::new(&shape![3, 2], vec![0, 3, 1, 4, 2, 5]).unwrap();

        assert_eq!(ans1, transposed_t1);
        assert_eq!(ans2, transposed_t2);
    }

    #[test]
    fn transpose_mt() {
        let t1 = Tensor::<i32>::rand(&shape![10, 60, 5]);
        let transpose = transpose![1, 2, 0];

        let ans = t1.transpose(&transpose).unwrap();
        let mt_ans = t1.transpose_mt(&transpose).unwrap();

        assert_eq!(mt_ans, ans);
    }

    #[test]
    fn transpose_in_place() {
        let mut t1 = Tensor::<i32>::rand(&shape![10, 20, 5]);
        let transpose = transpose![2, 0, 1];

        let ans = t1.clone().transpose(&transpose).unwrap();
        t1.transpose_in_place(&transpose).unwrap();

        assert_eq!(ans, t1);
    }

    #[test]
    fn transpose_in_place_mt() {
        let mut t1 = Tensor::<i32>::rand(&shape![10, 20, 5]);
        let transpose = transpose![2, 0, 1];

        let ans = t1.clone().transpose(&transpose).unwrap();
        t1.transpose_in_place_mt(&transpose).unwrap();

        assert_eq!(ans, t1);
    }

    #[test]
    fn transpose_mat() {
        let m1 = Matrix::<i32>::new(3, 3, (0..9).collect()).unwrap();
        let t = m1.transpose();
        let ans = Matrix::<i32>::new(3, 3, vec![0, 3, 6, 1, 4, 7, 2, 5, 8]).unwrap();

        assert_eq!(ans, t);
    }

    #[test]
    fn transpose_mat_mt() {
        let m1 = Matrix::<i32>::rand(10, 5);

        let ans = m1.transpose();
        let mt_ans = m1.transpose_mt();

        assert_eq!(mt_ans, ans);
    }

    #[test]
    fn transpose_in_place_mat() {
        let mut m1 = Matrix::<i32>::rand(10, 5);

        let ans = m1.clone().transpose();
        m1.transpose_in_place();

        assert_eq!(ans, m1);
    }

    #[test]
    fn transpose_in_place_mat_mt() {
        let mut m1 = Matrix::<i32>::rand(10, 5);
        let ans = m1.transpose();
        m1.transpose_in_place_mt();
        assert_eq!(ans, m1);
    }

    #[test]
    fn tensor_contraction_multiplication() {
        let t1 = Tensor::<i32>::new(&shape![2, 3], (0..6).collect()).unwrap();
        let t2 = Tensor::<i32>::new(&shape![3, 3], (0..9).collect()).unwrap();
        let ans = Tensor::<i32>::new(&shape![2, 3], vec![15, 18, 21, 42, 54, 66]).unwrap();
        assert_eq!(ans, t1.contract_mul(&t2).unwrap());

        let t3 = Tensor::<i32>::new(&shape![2, 3], vec![5, 10, 2, 3, -2, 1]).unwrap();
        let t4 = Tensor::<i32>::new(&shape![3, 2, 2], (0..12).collect()).unwrap();
        let ans = Tensor::<i32>::new(&shape![2, 2, 2], vec![56, 73, 90, 107, 0, 2, 4, 6]).unwrap();
        assert_eq!(ans, t3.contract_mul(&t4).unwrap());
    }

    #[test]
    fn matrix_mul() {
        let m1 = Matrix::<i32>::new(3, 3, (0..9).collect()).unwrap();
        let m2 = Matrix::<i32>::new(3, 2, (0..6).collect()).unwrap();

        let res = m1.contract_mul(&m2).unwrap();
        let ans = Matrix::new(3, 2, vec![10, 13, 28, 40, 46, 67]).unwrap();

        assert_eq!(ans, res);
    }

    #[test]
    fn tensor_contraction_multiplication_shape_invalid() {
        let t1 = Tensor::<i32>::new(&shape![2, 3], (0..6).collect()).unwrap();
        let t2 = Tensor::<i32>::new(&shape![2, 2], (0..4).collect()).unwrap();
        t1.contract_mul(&t2).expect_err("Invalid shapes");
    }

    #[test]
    fn invalid_shapes_for_mat_mul() {
        let m1 = Matrix::<i32>::rand(3, 3);
        let m2 = Matrix::<i32>::rand(2, 3);
        m1.contract_mul(&m2).expect_err("Should've panicked");
    }

    #[test]
    fn test_kronecker_product() {
        let t1 = Tensor::<i32>::new(&shape![2, 3], (0..6).collect()).unwrap();
        let t2 = Tensor::<i32>::new(&shape![5, 2, 2], (0..20).collect()).unwrap();
        let mut ans_vec = vec![0; 20];
        ans_vec.extend(0..20);
        ans_vec.extend((0..20).map(|i| i * 2));
        ans_vec.extend((0..20).map(|i| i * 3));
        ans_vec.extend((0..20).map(|i| i * 4));
        ans_vec.extend((0..20).map(|i| i * 5));
        let ans = Tensor::<i32>::new(&shape![10, 6, 2], ans_vec).unwrap();
        assert_eq!(t1.kronecker(&t2), ans);
    }

    #[test]
    fn kronecker_mt() {
        let t1 = Tensor::<i32>::new(&shape![2, 3], (0..6).collect()).unwrap();
        let t2 = Tensor::<i32>::new(&shape![5, 2, 2], (0..20).collect()).unwrap();

        let ans = t1.kronecker(&t2);
        let mt_ans = t1.kronecker_mt(&t2);

        assert_eq!(mt_ans, ans);
    }

    #[test]
    fn test_mat_kronecker_product() {
        let m1 = Matrix::<i32>::new(30, 30, (0..900).collect()).unwrap();
        let m2 = Matrix::<i32>::new(30, 20, (0..600).collect()).unwrap();

        let ans = m1.tensor.kronecker(&m2.tensor).try_into().unwrap();
        let res = m1.kronecker(&m2);

        assert_eq!(res, ans);
    }

    #[test]
    fn kronecker_mt_mat() {
        let m1 = Matrix::<i32>::rand(30, 30).clip(-10, 10);
        let m2 = Matrix::<i32>::rand(30, 30).clip(-10, 10);

        let ans = m1.kronecker(&m2);
        let mt_ans = m1.kronecker_mt(&m2);

        assert_eq!(mt_ans, ans);
    }

    #[test]
    fn test_trace() {
        let m1 = Matrix::<i32>::new(2, 2, (0..4).collect()).unwrap();
        assert_eq!(3, m1.trace());
    }

    #[test]
    #[should_panic]
    fn invalid_trace_non_square() {
        let m1 = Matrix::<i32>::new(2, 3, (0..6).collect()).unwrap();
        m1.trace();
    }

    #[test]
    fn determinant() {
        let m1 = Matrix::<i32>::new(3, 3, vec![5, -2, 1, 8, 9, -5, 1, 0, 2]).unwrap();
        assert_eq!(det_slow(&m1), 123);
        assert!(approx_eq!(f64, m1.transform_elementwise(|x| x as f64).det(), 123.0));
    }

    #[test]
    #[should_panic]
    fn invalid_det_square_matrix_only() {
        let m1 = Matrix::<i32>::new(3, 2, (0..6).collect()).unwrap();
        det_slow(&m1);
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
            .all(|(i, x)| { approx_eq!(f64, x, identity(3)[i], epsilon = 1e-15) }));
    }

    #[test]
    fn inverse_det_0() {
        let m1 = Matrix::<f64>::new(3, 3, vec![0.0; 9]).unwrap();
        inv_slow(&m1).expect_err(format!("{:?}", TensorErrors::DeterminantZero).as_str());
        m1.inv().expect_err(format!("{:?}", TensorErrors::DeterminantZero).as_str());
    }

    #[test]
    #[should_panic]
    fn invalid_inversion_square_matrix_only() {
        let m1 = Matrix::<i32>::new(3, 2, (0..6).collect()).unwrap();
        inv_slow(&m1).expect("Should've panicked");
    }

    #[test]
    fn clipping() {
        let t1 = Tensor::<i32>::new(
            &shape![3, 3, 3],
            (0..27).collect(),
        ).unwrap();
        let t1 = t1.clip(5, 10);
        let ans = Tensor::<i32>::new(
            &shape![3, 3, 3],
            vec![5, 5, 5, 5, 5, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        ).unwrap();

        assert_eq!(t1, ans);
    }

    #[test]
    fn clipping_mat() {
        let m1 = Matrix::<i32>::new(3, 3, (0..9).collect()).unwrap();
        let res = m1.clip(3, 6);
        let ans = Matrix::new(
            3,
            3,
            vec![3, 3, 3, 3, 4, 5, 6, 6, 6],
        ).unwrap();

        assert_eq!(res, ans);
    }

    #[test]
    fn pool_tensor() {
        let t1: Tensor<f64> = Tensor::<i32>::new(
            &shape![3, 3, 3],
            vec![
                1, 5, -1, 2, 3, -5, 12, 10, -10, 1, -4, 2, 9, 6, 8, -1, 0, -1, -8, 7, 4, 5, 1, 2,
                -5, 3, 1,
            ],
        )
        .unwrap()
        .transform_elementwise(|x| x.into());

        let avg_pool = t1.pool(pool_avg, &shape![2, 2, 2], &shape![2, 2, 2]);
        let sum_pool = t1.pool(pool_sum, &shape![2, 2, 2], &shape![2, 2, 2]);
        let max_pool = t1.pool(pool_max, &shape![2, 2, 2], &shape![1, 1, 1]);
        let min_pool = t1.pool(pool_min, &shape![3, 1, 1], &shape![3, 1, 1]);

        let sum_ans = Tensor::<f64>::new(
            &shape![2, 2, 2],
            vec![23.0, 4.0, 21.0, -11.0, 5.0, 6.0, -2.0, 1.0],
        )
        .unwrap();
        let avg_ans = Tensor::<f64>::new(
            &shape![2, 2, 2],
            vec![2.875, 1.0, 5.25, -5.5, 1.25, 3.0, -1.0, 1.0],
        )
        .unwrap();
        let max_ans = Tensor::<f64>::new(
            &shape![3, 3, 3],
            vec![
                9.0, 8.0, 8.0, 12.0, 10.0, 8.0, 12.0, 10.0, -1.0, 9.0, 8.0, 8.0, 9.0, 8.0, 8.0,
                3.0, 3.0, 1.0, 7.0, 7.0, 4.0, 5.0, 3.0, 2.0, 3.0, 3.0, 1.0,
            ],
        )
        .unwrap();
        let min_ans = Tensor::<f64>::new(
            &shape![1, 3, 3],
            vec![-8.0, -4.0, -1.0, 2.0, 1.0, -5.0, -5.0, 0.0, -10.0],
        )
        .unwrap();

        assert_eq!(avg_pool, avg_ans);
        assert_eq!(sum_pool, sum_ans);
        assert_eq!(max_pool, max_ans);
        assert_eq!(min_pool, min_ans);
    }

    #[test]
    fn pool_tensor_mt() {
        let t1 = Tensor::<i32>::rand(&shape![200, 10, 10]);

        let ans = t1.pool(pool_min, &shape![10, 30, 1], &shape![10, 10, 10]);
        let mt_ans = t1.pool_mt(&pool_min, &shape![10, 30, 1], &shape![10, 10, 10]);

        assert_eq!(ans, mt_ans);
    }

    #[test]
    fn pool_mat() {
        let m1 = Matrix::<f64>::new(3, 3, (0..9).map(f64::from).collect()).unwrap();
        let sum = m1.pool(pool_sum_mat, (2, 2), (2, 2));
        let min = m1.pool(pool_min_mat, (2, 2), (1, 2));
        let avg = m1.pool(pool_avg_mat, (1, 1), (1, 1));
        let max = m1.pool(pool_max_mat, (1, 1), (2, 2));

        let sum_ans = Matrix::<f64>::new(2, 2, vec![8.0, 7.0, 13.0, 8.0]).unwrap();
        let min_ans = Matrix::<f64>::new(3, 2, vec![0.0, 2.0, 3.0, 5.0, 6.0, 8.0]).unwrap();
        let avg_ans = m1.clone();
        let max_ans = Matrix::<f64>::new(2, 2, vec![0.0, 2.0, 6.0, 8.0]).unwrap();

        assert_eq!(sum, sum_ans);
        assert_eq!(min, min_ans);
        assert_eq!(avg, avg_ans);
        assert_eq!(max, max_ans);
    }

    #[test]
    fn pool_mat_mt() {
        let m1 = Matrix::<f64>::rand(20, 20);

        let ans = m1.pool(pool_avg_mat, (5, 4), (2, 3));
        let mt_ans = m1.pool_mt(&pool_avg_mat, (5, 4), (2, 3));

        assert_eq!(ans, mt_ans);
    }

    #[test]
    fn normalised() {
        let t1 = Tensor::<f64>::new(
            &shape![3, 3, 3],
            (0..27).map(|x| x as f64).collect::<Vec<f64>>(),
        )
        .unwrap();

        let t1_norm_l1 = t1.clone().norm_l1();
        let t1_norm_l2 = t1.norm_l2();

        let t1_norm_l1_ans = Tensor::<f64>::new(
            &shape![3, 3, 3],
            (0..27).map(|x| (x as f64) / 351.0).collect::<Vec<f64>>(),
        )
        .unwrap();
        let t1_norm_l2_ans = Tensor::<f64>::new(
            &shape![3, 3, 3],
            (0..27)
                .map(|x| (x as f64) / ComplexFloat::sqrt(6201.0))
                .collect::<Vec<f64>>(),
        )
        .unwrap();

        assert_eq!(t1_norm_l1, t1_norm_l1_ans);
        assert_eq!(t1_norm_l2, t1_norm_l2_ans);

        let t2 = Tensor::<Complex64>::new(
            &&shape![2, 3],
            (0..6)
                .map(|x| Complex64 {
                    re: x as f64,
                    im: x as f64,
                })
                .collect(),
        )
        .unwrap();

        let t2_norm_l1 = t2.clone().norm_l1();
        let t2_norm_l2 = t2.norm_l2();

        let t2_norm_l1_ans = Tensor::<Complex64>::new(
            &shape![2, 3],
            (0..6)
                .map(|x| Complex64 {
                    re: (x as f64) / (15.0 * f64::sqrt(2.0)),
                    im: (x as f64) / (15.0 * f64::sqrt(2.0)),
                })
                .collect(),
        )
        .unwrap();
        let t2_norm_l2_ans = Tensor::<Complex64>::new(
            &shape![2, 3],
            (0..6)
                .map(|x| Complex64 {
                    re: (x as f64) / ComplexFloat::sqrt(110.0),
                    im: (x as f64) / ComplexFloat::sqrt(110.0),
                })
                .collect(),
        )
        .unwrap();

        assert_approx_eq!(
            f64,
            (t2_norm_l1_ans - t2_norm_l1)
                .transform_elementwise(Complex64::abs)
                .sum(),
            0.0,
            epsilon = 1e-15
        );
        assert_approx_eq!(
            f64,
            (t2_norm_l2_ans - t2_norm_l2)
                .transform_elementwise(Complex64::abs)
                .sum(),
            0.0,
            epsilon = 1e-15
        );
    }

    #[test]
    fn rand_gaussian_sample() {
        let t1 = gaussian_sample(1.0, &shape![3, 3, 3], -10.0, 10.0);

        assert!(t1.iter().all(|x| -10.0 <= *x));
        assert!(t1.iter().all(|x| 10.0 >= *x));

        println!("{:?}", t1);
    }

    #[test]
    #[should_panic]
    fn rand_gaussian_sample_invalid_min_more_than_max() {
        gaussian_sample(1.0, &shape![3, 3, 3], 10.0, -10.0);
    }

    #[test]
    #[should_panic]
    fn rand_gaussian_sample_invalid_min_eq_max() {
        gaussian_sample(1.0, &shape![3, 3, 3], 10.0, 10.0);
    }

    #[test]
    #[should_panic]
    fn rand_gaussian_sample_invalid_neg_sigma() {
        gaussian_sample(-1.0, &shape![3, 3, 3], -100.0, -10.0);
    }

    #[test]
    fn single_sigma_gaussian_pdf() {
        let t1 = gaussian_pdf_single_sigma(0.5, &shape![5, 3]);
        let ans = (Tensor::<f64>::new(
            &shape![5, 3],
            vec![
                5.0, 4.0, 5.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 5.0, 4.0, 5.0,
            ],
        )
        .unwrap()
            * -2.0)
            .exp()
            * 2.0
            / PI;
        assert_approx_eq!(
            f64,
            (t1 - ans).transform_elementwise(f64::abs).sum(),
            0.0,
            epsilon = 1e-15
        );
    }

    #[test]
    fn multi_sigma_gaussian_pdf() {
        let t1 = gaussian_pdf_multi_sigma(vec![0.25, 0.4], &shape![5, 3]);
        let ans = (Tensor::<f64>::new(
            &shape![5, 3],
            vec![
                35.125, 32.0, 35.125, 11.125, 8.0, 11.125, 3.125, 0.0, 3.125, 11.125, 8.0, 11.125,
                35.125, 32.0, 35.125,
            ],
        )
        .unwrap()
            * -1.0)
            .exp()
            * 5.0
            / PI;
        assert_approx_eq!(
            f64,
            (t1 - ans).transform_elementwise(f64::abs).sum(),
            0.0,
            epsilon = 1e-15
        );
    }

    #[test]
    #[should_panic]
    fn invalid_multi_sigma_gaussian_pdf_zero_sigma() {
        gaussian_pdf_multi_sigma(vec![0.0, 0.5], &shape![1, 1]);
    }

    #[test]
    #[should_panic]
    fn invalid_multi_sigma_gaussian_pdf_negative_sigma() {
        gaussian_pdf_multi_sigma(vec![0.5, -0.5], &shape![1, 1]);
    }

    #[test]
    #[should_panic]
    fn invalid_multi_sigma_gaussian_pdf_sigma_len_invalid() {
        gaussian_pdf_multi_sigma(vec![0.1, 0.5], &shape![1, 1, 3]);
    }

    #[test]
    fn gaussian_covariance_matrix_tensor() {
        let cov = Matrix::<f64>::new(
            2, 2,
            vec![
                1.0, 0.5,
                0.5, 1.0
            ],
        ).unwrap();

        let ans = Tensor::<f64>::new(
            &shape![3, 4],
            vec![
                0.057228531823873385, 0.11146595955293902, 0.057228531823873385, 0.007745039563599408,
                0.041006034909973794, 0.1555632781262252, 0.1555632781262252, 0.041006034909973794,
                0.007745039563599408, 0.057228531823873385, 0.11146595955293902, 0.057228531823873385,
            ],
        ).unwrap();
        assert!(approx_eq!(Tensor<f64>, gaussian_pdf_cov_mat(cov, &shape![3, 4]), ans, epsilon = 1e-15));
    }

    #[test]
    #[should_panic]
    fn invalid_cov_mat_non_square() {
        gaussian_pdf_cov_mat(Matrix::<f64>::new(2, 1, vec![0.0, 1.0]).unwrap(), &shape![1, 1]);
    }

    #[test]
    #[should_panic]
    fn invalid_cov_mat_shape_not_right_dim() {
        gaussian_pdf_cov_mat(identity(2), &shape![1]);
    }

    #[test]
    #[should_panic]
    fn invalid_cov_mat_not_positive_definite() {
        gaussian_pdf_cov_mat(Matrix::<f64>::new(2, 2, vec![0.0, 1.0, 1.0, 0.0]).unwrap(), &shape![1, 1]);
    }

    #[test]
    fn test_householder() {
        let m1: Matrix<f64> = Tensor::<i32>::new(
            &shape![9],
            vec![
                    4, 1, 1,
                    1, 3, 0,
                    1, 0, 2,
                ],
            )
            .unwrap()
            .iter()
            .map(|x| f64::from_i32(*x).unwrap())
            .collect::<Matrix<f64>>()
            .reshape(3, 3)
            .unwrap();

        let (q1, r1) = m1.householder();

        for i in 0..r1.shape[0] {
            if i >= r1.shape[1] - 1 {
                continue
            }

            assert!(r1.slice(i+1..r1.shape[0], i..i+1).iter().all(|x| { x.abs() <= 1e-10 }));
        }
        assert!(q1.contract_mul(&r1).unwrap().tensor.approx_eq(m1.clone().tensor, F64Margin::default().epsilon(1e-10)));

        let m2: Matrix<Complex64> = Tensor::<Complex64>::new(
            &shape![3, 2],
            vec![
                Complex64 { re: 4.0, im: 1.0 }, Complex64 { re: -5.0, im: -2.0 },
                Complex64 { re: 5.0, im: -4.0 }, Complex64 { re: 5.0, im: 3.0 },
                Complex64 { re: 0.0, im: 0.0 }, Complex64 { re: 1.0, im: -1.0 },
            ],
        ).unwrap().try_into().unwrap();
        let (q2, r2) = m2.householder();

        for i in 0..r2.shape[0] {
            if i >= r2.shape[1] - 1 {
                continue
            }

            assert!(r2.slice(i+1..r2.shape[0], i..i+1).iter().all(|x| x.abs() <= 1e-10));
        }
        assert!(q2.contract_mul(&r2).unwrap().tensor.approx_eq(m2.tensor, F64Margin::default().epsilon(1e-10)));

        let m3: Matrix<Complex64> = Tensor::<Complex64>::new(
            &shape![2, 3],
            vec![
                Complex64 { re: -4.0, im: -1.0 }, Complex64 { re: 5.0, im: -3.0 }, Complex64 { re: 2.0, im: -4.0 },
                Complex64 { re: -5.0, im: 2.0 }, Complex64 { re: 2.0, im: -1.0 }, Complex64 { re: 4.0, im: -1.0 },
            ],
        ).unwrap().try_into().unwrap();
        let (q3, r3) = m3.householder();

        for i in 0..(r3.shape[0] - 1) {
            if i >= r3.shape[1] - 1 {
                continue
            }
            assert!(r3.slice(i+1..r3.shape[0], i..i+1).iter().all(|x| x.abs() <= 1e-10));
        }
        assert!(q3.contract_mul(&r3).unwrap().tensor.approx_eq(m3.tensor, F64Margin::default().epsilon(1e-10)));
    }

    #[test]
    fn solve_quadratic_poly() {
        let coefficients = [
            Complex64 { re: 1.0, im: 2.0 },
            Complex64 { re: 0.0, im: -2.0 },
            Complex64 { re: 5.0, im: 0.0 },
        ];
        let roots = solve_quadratic(&coefficients);

        assert_eq!(roots.len(), 2);
        assert_approx_eq!(f64, coefficients.iter().enumerate().map(|(i, c)| c * roots[0].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 1e-15);
        assert_approx_eq!(f64, coefficients.iter().enumerate().map(|(i, c)| c * roots[1].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn solve_cubic_poly() {
        let coefficients = [
            Complex64 { re: 4.0, im: 0.0 },
            Complex64 { re: 3.0, im: 0.0 },
            Complex64 { re: 2.0, im: 0.0 },
            Complex64 { re: 1.0, im: 0.0 },
        ];
        let roots = solve_cubic(&coefficients);

        assert_eq!(roots.len(), 3);
        assert_approx_eq!(f64, coefficients.iter().enumerate().map(|(i, c)| c * roots[0].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 2e-15);
        assert_approx_eq!(f64, coefficients.iter().enumerate().map(|(i, c)| c * roots[1].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 2e-15);
        assert_approx_eq!(f64, coefficients.iter().enumerate().map(|(i, c)| c * roots[2].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 2e-15);

        let coefficients2 = [
            Complex64 { re: 45.0, im: -2.0 },
            Complex64 { re: -3.0, im: 4.0 },
            Complex64 { re: 2.0, im: 3.0 },
            Complex64 { re: 1.0, im: 10.0 },
        ];
        let roots2 = solve_cubic(&coefficients2);

        assert_eq!(roots2.len(), 3);
        assert_approx_eq!(f64, coefficients2.iter().enumerate().map(|(i, c)| c * roots2[0].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 2e-11);
        assert_approx_eq!(f64, coefficients2.iter().enumerate().map(|(i, c)| c * roots2[1].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 2e-11);
        assert_approx_eq!(f64, coefficients2.iter().enumerate().map(|(i, c)| c * roots2[2].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 2e-11);
    }

    #[test]
    fn solve_quartic_poly() {
        let coefficients = [
            Complex64 { re: 4.5, im: -2.0 },
            Complex64 { re: 3.4, im: 5.0 },
            Complex64 { re: 2.1, im: 3.0 },
            Complex64 { re: 1.2, im: 2.0 },
            Complex64 { re: 1.1, im: -10.0 },
        ];

        let roots = solve_quartic(&coefficients);
        assert_eq!(roots.len(), 4);

        assert_approx_eq!(f64, coefficients.iter().enumerate().map(|(i, c)| c * roots[0].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 2e-10);
        assert_approx_eq!(f64, coefficients.iter().enumerate().map(|(i, c)| c * roots[1].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 2e-10);
        assert_approx_eq!(f64, coefficients.iter().enumerate().map(|(i, c)| c * roots[2].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 2e-10);
        assert_approx_eq!(f64, coefficients.iter().enumerate().map(|(i, c)| c * roots[3].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 2e-10);
    }

    #[test]
    fn contract_mul_mt() {
        let t1 = Tensor::<f64>::rand(&shape![10, 3, 20]);
        let t2 = Tensor::<f64>::rand(&shape![20, 10]);

        let ans = t1.clone().contract_mul(&t2).unwrap();
        let mt_ans = t1.contract_mul_mt(&t2).unwrap();

        assert_eq!(ans, mt_ans);
    }

    #[test]
    fn mat_mul_mt() {
        let m1 = Matrix::<f64>::rand(10, 10);
        let m2 = Matrix::<f64>::rand(10, 10);

        let ans = m1.contract_mul_mt(&m2).unwrap();
        let mt_ans = m1.mat_mul_mt(&m2).unwrap();

        assert_eq!(ans, mt_ans);
    }

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
        let m1_rref_ans = identity(3).concat_mt(&Matrix::<f64>::new(
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
        let m3_rref_ans = identity(2);

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
    }

    #[test]
    fn transformation_rank_mat() {
        let m1: Matrix<f64> = identity(3);
        assert_eq!(m1.transformation_rank(), 3);

        let mut m2 = identity(4);
        m2.slice_mut(1..3, 0..4).set_all(&Matrix::<f64>::zeros(2, 4));
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
    }

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
        let (h1, q1) = m1.upper_hessenberg();
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

        let (h2, q2) = m2.upper_hessenberg();
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
        let (h1, q1) = m1.lower_hessenberg();
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

        let (h2, q2) = m2.lower_hessenberg();
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
            let (vals, vecs) = m.eigendecompose();
            let ord = m.rows;

            for i in 0..ord {
                let vec = vecs.slice(0..ord, i..i + 1);
                let val = vals[i];

                assert!(approx_eq!(Matrix<Complex64>, vec.clone() * val, m.contract_mul_mt(&vec).unwrap(), epsilon = 1e-13));
            }
        }
    }

    #[test]
    #[should_panic]
    fn invalid_eigendecomposition_non_square() {
        Matrix::<Complex64>::new(2, 1, vec![Complex64::ONE, Complex64::ZERO]).unwrap().eigendecompose();
    }

    #[test]
    fn radix_2_vec_fft_test() {
        let v1 = vec![
            Complex64::new(1.0, 0.0), Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 2.0), Complex64::new(4.0, -5.0),
            Complex64::new(6.0, 7.0), Complex64::new(8.0, -4.0),
            Complex64::new(2.0, 1.0), Complex64::new(0.0, 0.0),
        ];
        let v2 = vec![
            Complex64 { re: 0.0, im: 1.0 },
            Complex64 { re: -0.1453217681275245, im: -3.6026214877097886 },
            Complex64 { re: -1.3968022466674206, im: 0.2212317420824741 },
            Complex64 { re: 6.39622598104463, im: -0.29714171603742257 },
            Complex64 { re: -2.3576391889522714, im: 10.365400978964416 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
        ];
        let res1 = radix_2_fft_vec(&v1);
        let res2 = radix_2_fft_vec(&v2);

        let ans1 = vec![
            Complex64 { re: 26.0, im: 0.0 },
            Complex64 { re: -12.485281374238571, im: -0.9289321881345245 },
            Complex64 { re: 2.0, im: -2.0000000000000018 },
            Complex64 { re: -0.34314575050761853, im: -10.242640687119287 },
            Complex64 { re: -2.0, im: 20.0 },
            Complex64 { re: 4.485281374238571, im: -15.071067811865476 },
            Complex64 { re: 2.0, im: 10.000000000000002 },
            Complex64 { re: -11.656854249492381, im: -1.7573593128807126 },
        ];
        let ans2 = vec![
            Complex64 { re: 2.4964627772974133, im: 7.686869517299678 },
            Complex64 { re: 10.194430302685799, im: -4.794067509264503 },
            Complex64 { re: -4.804250848251911, im: -14.725982651422731 },
            Complex64 { re: -18.400909719645895, im: 0.9514637996861768 },
            Complex64 { re: -4.266316713957217, im: 17.685716986054096 },
            Complex64 { re: 13.946932312659404, im: 5.899646309853949 },
            Complex64 { re: 4.004431213373595, im: -12.84493282150859 },
            Complex64 { re: -15.4761803961386, im: -4.600523633014343 },
            Complex64 { re: -10.005345648536798, im: 15.4863959247941 },
            Complex64 { re: 8.873863904133222, im: 13.797591498439782 },
            Complex64 { re: 9.961992710321402, im: -1.211214813171261 },
            Complex64 { re: -0.041646627012200454, im: -2.0042344264809056 },
            Complex64 { re: 2.3446428293875154, im: 4.602621487709789 },
            Complex64 { re: 8.446377396379242, im: -1.472613543220143 },
            Complex64 { re: 0.2683836803659996, im: -8.679473629755085 },
            Complex64 { re: -7.542867173060969, im: 0.2227375039999857 },
        ];

        for i in 0..8 {
            assert!(approx_eq!(f64, res1[i].re, ans1[i].re, epsilon = 1e-15));
            assert!(approx_eq!(f64, res1[i].im, ans1[i].im, epsilon = 1e-15));
        }

        for i in 0..16 {
            assert!(approx_eq!(f64, res2[i].re, ans2[i].re, epsilon = 1e-10));
            assert!(approx_eq!(f64, res2[i].im, ans2[i].im, epsilon = 1e-10));
        }
    }

    #[test]
    #[should_panic]
    fn invalid_radix_2_vec_fft_len_not_pow_2() {
        radix_2_fft_vec(&vec![Complex64 { re: 0.0, im: 0.0 }, Complex64 { re: 0.0, im: 1.0 }, Complex64 { re: 0.0, im: 1.0 }]);
    }

    #[test]
    fn bluestein_fft_test() {
        let v1 = vec![
            Complex64::new(0.0, 1.0),
            Complex64::new(2.0, -3.0),
            Complex64::new(1.0, -1.0),
            Complex64::new(5.0, -4.0),
            Complex64::new(8.0, -7.0)
        ];
        let res = bluestein_fft_vec(&v1);
        let ans = vec![
            Complex64 { re: 16.0, im: - 14.0},
            Complex64 { re: 3.8036497995578236, im: 10.012395135066075 },
            Complex64 { re: -6.738096517215357, im:  7.267570420448962 },
            Complex64 { re: -5.734039437784222, im: 7.822599523300513 },
            Complex64 { re: -7.331513844558244, im: -6.102565078815551 },
        ];

        for i in 0..5 {
            assert!(approx_eq!(f64, res[i].re, ans[i].re, epsilon = 1e-10));
            assert!(approx_eq!(f64, res[i].im, ans[i].im, epsilon = 1e-10));
        }
    }

    #[test]
    fn fft_test() {
        let v1 = vec![
            Complex64::new(1.0, 0.0), Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 2.0), Complex64::new(4.0, -5.0),
            Complex64::new(6.0, 7.0), Complex64::new(8.0, -4.0),
            Complex64::new(2.0, 1.0), Complex64::new(0.0, 0.0),
        ];
        let v2 = vec![
            Complex64 { re: 0.0, im: 1.0 },
            Complex64 { re: -0.1453217681275245, im: -3.6026214877097886 },
            Complex64 { re: -1.3968022466674206, im: 0.2212317420824741 },
            Complex64 { re: 6.39622598104463, im: -0.29714171603742257 },
            Complex64 { re: -2.3576391889522714, im: 10.365400978964416 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
        ];
        let v3 = vec![
            Complex64::new(0.0, 1.0),
            Complex64::new(2.0, -3.0),
            Complex64::new(1.0, -1.0),
            Complex64::new(5.0, -4.0),
            Complex64::new(8.0, -7.0),
        ];

        let res1 = fft_vec(&v1);
        let res2 = fft_vec(&v2);
        let res3 = fft_vec(&v3);

        let ans1 = vec![
            Complex64 { re: 26.0, im: 0.0 },
            Complex64 { re: -12.485281374238571, im: -0.9289321881345245 },
            Complex64 { re: 2.0, im: -2.0000000000000018 },
            Complex64 { re: -0.34314575050761853, im: -10.242640687119287 },
            Complex64 { re: -2.0, im: 20.0 },
            Complex64 { re: 4.485281374238571, im: -15.071067811865476 },
            Complex64 { re: 2.0, im: 10.000000000000002 },
            Complex64 { re: -11.656854249492381, im: -1.7573593128807126 },
        ];
        let ans2 = vec![
            Complex64 { re: 2.4964627772974133, im: 7.686869517299678 },
            Complex64 { re: 10.194430302685799, im: -4.794067509264503 },
            Complex64 { re: -4.804250848251911, im: -14.725982651422731 },
            Complex64 { re: -18.400909719645895, im: 0.9514637996861768 },
            Complex64 { re: -4.266316713957217, im: 17.685716986054096 },
            Complex64 { re: 13.946932312659404, im: 5.899646309853949 },
            Complex64 { re: 4.004431213373595, im: -12.84493282150859 },
            Complex64 { re: -15.4761803961386, im: -4.600523633014343 },
            Complex64 { re: -10.005345648536798, im: 15.4863959247941 },
            Complex64 { re: 8.873863904133222, im: 13.797591498439782 },
            Complex64 { re: 9.961992710321402, im: -1.211214813171261 },
            Complex64 { re: -0.041646627012200454, im: -2.0042344264809056 },
            Complex64 { re: 2.3446428293875154, im: 4.602621487709789 },
            Complex64 { re: 8.446377396379242, im: -1.472613543220143 },
            Complex64 { re: 0.2683836803659996, im: -8.679473629755085 },
            Complex64 { re: -7.542867173060969, im: 0.2227375039999857 },
        ];
        let ans3 = vec![
            Complex64 { re: 16.0, im: - 14.0},
            Complex64 { re: 3.8036497995578236, im: 10.012395135066075 },
            Complex64 { re: -6.738096517215357, im:  7.267570420448962 },
            Complex64 { re: -5.734039437784222, im: 7.822599523300513 },
            Complex64 { re: -7.331513844558244, im: -6.102565078815551 },
        ];

        for i in 0..8 {
            assert!(approx_eq!(f64, res1[i].re, ans1[i].re, epsilon = 1e-15));
            assert!(approx_eq!(f64, res1[i].im, ans1[i].im, epsilon = 1e-15));
        }

        for i in 0..16 {
            assert!(approx_eq!(f64, res2[i].re, ans2[i].re, epsilon = 1e-10));
            assert!(approx_eq!(f64, res2[i].im, ans2[i].im, epsilon = 1e-10));
        }

        for i in 0..5 {
            assert!(approx_eq!(f64, res3[i].re, ans3[i].re, epsilon = 1e-10));
            assert!(approx_eq!(f64, res3[i].im, ans3[i].im, epsilon = 1e-10));
        }
    }

    #[test]
    fn ifft_test() {
        let v1 = vec![
            Complex64 { re: 26.0, im: 0.0 },
            Complex64 { re: -12.485281374238571, im: -0.9289321881345245 },
            Complex64 { re: 2.0, im: -2.0000000000000018 },
            Complex64 { re: -0.34314575050761853, im: -10.242640687119287 },
            Complex64 { re: -2.0, im: 20.0 },
            Complex64 { re: 4.485281374238571, im: -15.071067811865476 },
            Complex64 { re: 2.0, im: 10.000000000000002 },
            Complex64 { re: -11.656854249492381, im: -1.7573593128807126 },
        ];
        let v2 = vec![
            Complex64 { re: 2.4964627772974133, im: 7.686869517299678 },
            Complex64 { re: 10.194430302685799, im: -4.794067509264503 },
            Complex64 { re: -4.804250848251911, im: -14.725982651422731 },
            Complex64 { re: -18.400909719645895, im: 0.9514637996861768 },
            Complex64 { re: -4.266316713957217, im: 17.685716986054096 },
            Complex64 { re: 13.946932312659404, im: 5.899646309853949 },
            Complex64 { re: 4.004431213373595, im: -12.84493282150859 },
            Complex64 { re: -15.4761803961386, im: -4.600523633014343 },
            Complex64 { re: -10.005345648536798, im: 15.4863959247941 },
            Complex64 { re: 8.873863904133222, im: 13.797591498439782 },
            Complex64 { re: 9.961992710321402, im: -1.211214813171261 },
            Complex64 { re: -0.041646627012200454, im: -2.0042344264809056 },
            Complex64 { re: 2.3446428293875154, im: 4.602621487709789 },
            Complex64 { re: 8.446377396379242, im: -1.472613543220143 },
            Complex64 { re: 0.2683836803659996, im: -8.679473629755085 },
            Complex64 { re: -7.542867173060969, im: 0.2227375039999857 },
        ];
        let v3 = vec![
            Complex64 { re: 16.0, im: - 14.0},
            Complex64 { re: 3.8036497995578236, im: 10.012395135066075 },
            Complex64 { re: -6.738096517215357, im:  7.267570420448962 },
            Complex64 { re: -5.734039437784222, im: 7.822599523300513 },
            Complex64 { re: -7.331513844558244, im: -6.102565078815551 },
        ];

        let ans1 = vec![
            Complex64::new(1.0, 0.0), Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 2.0), Complex64::new(4.0, -5.0),
            Complex64::new(6.0, 7.0), Complex64::new(8.0, -4.0),
            Complex64::new(2.0, 1.0), Complex64::new(0.0, 0.0),
        ];
        let ans2 = vec![
            Complex64 { re: 0.0, im: 1.0 },
            Complex64 { re: -0.1453217681275245, im: -3.6026214877097886 },
            Complex64 { re: -1.3968022466674206, im: 0.2212317420824741 },
            Complex64 { re: 6.39622598104463, im: -0.29714171603742257 },
            Complex64 { re: -2.3576391889522714, im: 10.365400978964416 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
        ];
        let ans3 = vec![
            Complex64::new(0.0, 1.0),
            Complex64::new(2.0, -3.0),
            Complex64::new(1.0, -1.0),
            Complex64::new(5.0, -4.0),
            Complex64::new(8.0, -7.0),
        ];

        let res1 = ifft_vec(&v1);
        let res2 = ifft_vec(&v2);
        let res3 = ifft_vec(&v3);

        for i in 0..8 {
            assert!(approx_eq!(f64, res1[i].re, ans1[i].re, epsilon = 1e-10));
            assert!(approx_eq!(f64, res1[i].im, ans1[i].im, epsilon = 1e-10));
        }

        for i in 0..16 {
            assert!(approx_eq!(f64, res2[i].re, ans2[i].re, epsilon = 1e-10));
            assert!(approx_eq!(f64, res2[i].im, ans2[i].im, epsilon = 1e-10));
        }

        for i in 0..5 {
            assert!(approx_eq!(f64, res3[i].re, ans3[i].re, epsilon = 1e-10));
            assert!(approx_eq!(f64, res3[i].im, ans3[i].im, epsilon = 1e-10));
        }
    }

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

        let res_axes_1_2 = t1.fft_axes(vec![1, 2]);
        let res_axis_0 = t1.fft_single_axis(0);
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

        let res_axes_1_2 = t1.ifft_axes(vec![1, 2]);
        let res_axis_0 = t1.ifft_single_axis(0);
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
    #[should_panic]
    fn invalid_tensor_fft_conv_axes_different_ranks() {
        let t1 = Tensor::<Complex64>::new(
            &shape![1, 2, 3],
            (0..6).map(|x| Complex64 { re: x as f64, im: 0.0 }).collect(),
        ).unwrap();
        let t2 = Tensor::<Complex64>::new(
            &shape![2, 3],
            (0..6).map(|x| Complex64 { re: x as f64, im: 0.0 }).collect(),
        ).unwrap();

        t1.fft_conv_axes(&t2, &HashSet::new());
    }

    #[test]
    #[should_panic]
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

        t1.fft_conv_axes(&t2, &axes);
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

        let res = t1.fft_conv_axes(&t2, &axes);

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
    #[should_panic]
    fn invalid_tensor_fft_conv_diff_ranks() {
        let t1 = Tensor::<Complex64>::new(
            &shape![1, 2, 3],
            (0..6).map(|x| Complex64 { re: x as f64, im: 0.0 }).collect(),
        ).unwrap();
        let t2 = Tensor::<Complex64>::new(
            &shape![2, 3],
            (0..6).map(|x| Complex64 { re: x as f64, im: 0.0 }).collect(),
        ).unwrap();

        t1.fft_conv(&t2);
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

        let res = t1.fft_conv(&t2);

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
    #[should_panic]
    fn invalid_fft_mat_conv_cols_different_cols() {
        let m1 = Matrix::<Complex64>::zeros(5, 1);
        let m2 = Matrix::<Complex64>::zeros(3, 2);

        m1.fft_conv_cols(&m2);
    }

    #[test]
    #[should_panic]
    fn invalid_fft_mat_conv_rows_different_rows() {
        let m1 = Matrix::<Complex64>::zeros(2, 2);
        let m2 = Matrix::<Complex64>::zeros(1, 1);

        m1.fft_conv_cols(&m2);
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

        let res = m1.fft_conv_rows(&m2);

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

        let res = m1.fft_conv_cols(&m2);

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

    #[test]
    #[should_panic]
    fn invalid_flip_axes_tensor() {
        let t1 = Tensor::<i32>::new(
            &shape![1, 2, 3, 4],
            vec![1; 24],
        ).unwrap();

        let mut axes = HashSet::new();
        axes.insert(4);

        t1.flip_axes(&axes);
    }

    #[test]
    #[should_panic]
    fn invalid_flip_axes_mt_tensor() {
        let t1 = Tensor::<i32>::new(
            &shape![1, 2, 3, 4],
            vec![1; 24],
        ).unwrap();

        let mut axes = HashSet::new();
        axes.insert(4);

        t1.flip_axes_mt(&axes);
    }

    #[test]
    fn flip_tensor() {
        let t1 = Tensor::<i32>::new(
            &shape![2, 3, 4],
            (0..24).collect(),
        ).unwrap();
        let mut axes = HashSet::new();
        axes.insert(0);
        axes.insert(2);

        let res_axes = t1.flip_axes(&axes);
        let res = t1.flip();

        let ans_axes = Tensor::<i32>::new(
            &shape![2, 3, 4],
              vec![
                  15, 14, 13, 12,
                  19, 18, 17, 16,
                  23, 22, 21, 20,

                  3, 2, 1, 0,
                  7, 6, 5, 4,
                  11, 10, 9, 8,
              ],
        ).unwrap();
        let ans = Tensor::<i32>::new(
            &shape![2, 3, 4],
            vec![
                23, 22, 21, 20,
                19, 18, 17, 16,
                15, 14, 13, 12,

                11, 10, 9, 8,
                7, 6, 5, 4,
                3, 2, 1, 0,
            ],
        ).unwrap();

        assert_eq!(res_axes, ans_axes);
        assert_eq!(res, ans);
    }

    #[test]
    fn flip_matrix() {
        let m1 = Matrix::<i32>::new(
            2, 3,
            vec![
                0, 1, 2,
                3, 4, 5,
            ],
        ).unwrap();
        let res_cols = m1.flip_cols();
        let res_rows = m1.flip_rows();
        let res = m1.flip();

        let ans_cols = Matrix::<i32>::new(
            2, 3,
            vec![
                3, 4, 5,
                0, 1, 2,
            ],
        ).unwrap();
        let ans_rows = Matrix::<i32>::new(
            2, 3,
            vec![
                2, 1, 0,
                5, 4, 3,
            ],
        ).unwrap();
        let ans = Matrix::<i32>::new(
            2, 3,
            vec![
                5, 4, 3,
                2, 1, 0,
            ],
        ).unwrap();

        assert_eq!(res_cols, ans_cols);
        assert_eq!(res_rows, ans_rows);
        assert_eq!(res, ans);
    }

    #[test]
    fn flip_mt_tensor() {
        let t1 = Tensor::<i32>::rand(&shape![10, 20, 10]);
        let mut axes = HashSet::new();
        axes.insert(0);
        axes.insert(1);

        let res_axes = t1.flip_axes(&axes);
        let res_axes_mt = t1.flip_axes_mt(&axes);

        let res = t1.flip();
        let res_mt = t1.flip_mt();

        assert_eq!(res_axes, res_axes_mt);
        assert_eq!(res, res_mt);
    }

    #[test]
    fn flip_mt_mat() {
        let m1 = Matrix::<i32>::rand(10, 20);

        let res_cols = m1.flip_cols();
        let res_rows = m1.flip_rows();
        let res = m1.flip();

        let res_cols_mt = m1.flip_cols_mt();
        let res_rows_mt = m1.flip_rows_mt();
        let res_mt = m1.flip_mt();

        assert_eq!(res_cols, res_cols_mt);
        assert_eq!(res_rows, res_rows_mt);
        assert_eq!(res, res_mt);
    }

    #[test]
    #[should_panic]
    fn invalid_corr_tensors() {
        let t1 = Tensor::<i32>::new(
            &shape![1, 2, 3, 4],
            vec![1; 24],
        ).unwrap();
        let t2 = Tensor::<i32>::new(
            &shape![1, 2, 3],
            vec![1; 6],
        ).unwrap();

        t1.corr(&t2);
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

        let res = t1.corr(&t2);
        let ans = t1
            .transform_elementwise(|x| Complex64::new(x as f64, 0.0))
            .fft_conv(&t2.transform_elementwise(|x| Complex64::new(x as f64, 0.0)).flip_mt())
            .transform_elementwise(|c| c.re.round().to_i32().unwrap());

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

        let res = m1.corr(&m2);
        let ans = m1
            .transform_elementwise(|x| Complex64::new(x as f64, 0.0))
            .fft_conv(&m2.transform_elementwise(|x| Complex64::new(x as f64, 0.0)).flip_mt())
            .transform_elementwise(|c| c.re.round().to_i32().unwrap());

        assert_eq!(res, ans);
    }

    #[test]
    fn corr_mt_tensors() {
        let t1 = Tensor::<f64>::rand(&shape![10, 20]).clip(-100.0, 100.0);
        let t2 = Tensor::<f64>::rand(&shape![10, 20]).clip(-100.0, 100.0);

        let res_mt = t1.corr_mt(&t2);
        let res = t1.corr(&t2);

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

        let in1 = m1.transform_elementwise(|x| Complex64::new(x, 0.0)) + m2.transform_elementwise(|x| Complex64::new(0.0, x));
        let in2 = m3.transform_elementwise(|x| Complex64::new(x, 0.0)) + m4.transform_elementwise(|x| Complex64::new(0.0, x));

        assert_eq!(in1.fft_corr_cols(&in2), in1.fft_conv_cols(&in2.flip_cols_mt()));
        assert_eq!(in1.fft_corr_rows(&in2), in1.fft_conv_rows(&in2.flip_rows_mt()));
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

        let in1 = t1.transform_elementwise(|x| Complex64::new(x, 0.0)) + t2.transform_elementwise(|x| Complex64::new(0.0, x));
        let in2 = t3.transform_elementwise(|x| Complex64::new(x, 0.0)) + t4.transform_elementwise(|x| Complex64::new(0.0, x));

        assert_eq!(in1.fft_corr_axes(&in2, &axes), in1.fft_conv_axes(&in2.flip_axes_mt(&axes), &axes));
        assert_eq!(in1.fft_corr(&in2), in1.fft_conv(&in2.flip_mt()));
    }

    #[test]
    fn conv_mat() {
        let m1 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);
        let m2 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);
        let m3 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);
        let m4 = Matrix::<f64>::rand(10, 10).clip(-100.0, 100.0);

        let in1 = m1.transform_elementwise(|x| Complex64::new(x, 0.0)) + m2.transform_elementwise(|x| Complex64::new(0.0, x));
        let in2 = m3.transform_elementwise(|x| Complex64::new(x, 0.0)) + m4.transform_elementwise(|x| Complex64::new(0.0, x));

        assert!(approx_eq!(Matrix<Complex64>, in1.conv(&in2), in1.fft_conv(&in2), epsilon = 1e-10));
    }

    #[test]
    fn conv_tensor() {
        let t1 = Tensor::<f64>::rand(&shape![10, 2, 5]).clip(-100.0, 100.0);
        let t2 = Tensor::<f64>::rand(&shape![10, 2, 5]).clip(-100.0, 100.0);
        let t3 = Tensor::<f64>::rand(&shape![10, 2, 5]).clip(-100.0, 100.0);
        let t4 = Tensor::<f64>::rand(&shape![10, 2, 5]).clip(-100.0, 100.0);

        let in1 = t1.transform_elementwise(|x| Complex64::new(x, 0.0)) + t2.transform_elementwise(|x| Complex64::new(0.0, x));
        let in2 = t3.transform_elementwise(|x| Complex64::new(x, 0.0)) + t4.transform_elementwise(|x| Complex64::new(0.0, x));

        assert!(approx_eq!(Tensor<Complex64>, in1.conv(&in2), in1.fft_conv(&in2), epsilon = 1e-10));
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

        let res_mt = t1.conv_mt(&t2);
        let res = t1.conv(&t2);

        assert_eq!(res_mt, res);
    }
}
