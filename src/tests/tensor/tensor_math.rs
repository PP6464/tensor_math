#[cfg(test)]
mod tensor_math_tests {
    use crate::tensor::tensor::{Matrix, Tensor};
    use crate::tensor::tensor::{Shape, TensorErrors};
    use crate::tensor::tensor_math::{det, gaussian_pdf_multi_sigma, gaussian_pdf_single_sigma, gaussian_sample, identity, inv, pool_avg, pool_avg_mat, pool_max, pool_max_mat, pool_min, pool_min_mat, pool_sum, pool_sum_mat, solve_cubic, solve_quadratic, solve_quartic, trace, Transpose};
    use crate::{shape, transpose};
    use float_cmp::{approx_eq, assert_approx_eq, ApproxEq, F64Margin, FloatMargin};
    use num::complex::{Complex64, ComplexFloat};
    use std::f64::consts::PI;
    use std::ops::Add;
    use num::FromPrimitive;

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
        assert_eq!(3, trace(&m1));
    }

    #[test]
    #[should_panic]
    fn invalid_trace_non_square() {
        let m1 = Matrix::<i32>::new(2, 3, (0..6).collect()).unwrap();
        trace(&m1);
    }

    #[test]
    fn determinant() {
        let m1 = Matrix::<i32>::new(3, 3, vec![5, -2, 1, 8, 9, -5, 1, 0, 2]).unwrap();
        assert_eq!(det(&m1), 123);
    }

    #[test]
    #[should_panic]
    fn invalid_det_square_matrix_only() {
        let m1 = Matrix::<i32>::new(3, 2, (0..6).collect()).unwrap();
        det(&m1);
    }

    #[test]
    fn inverse() {
        let m1 = Matrix::<f64>::new(
            3,
            3,
            vec![3.0, 4.0, 5.0, 2.0, -1.0, 4.0, 3.0, -5.0, -10.0],
        )
        .unwrap();
        let inverse = inv(&m1).unwrap();
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

        assert_eq!(inverse, ans);
        assert!(m1
            .contract_mul(&inverse)
            .unwrap()
            .enumerated_iter()
            .all(|(i, x)| { approx_eq!(f64, x, identity(3)[i], epsilon = 1e-15) }));
    }

    #[test]
    fn inverse_det_0() {
        let m1 = Matrix::<f64>::new(3, 3, vec![0.0; 9]).unwrap();
        inv(&m1).expect_err(format!("{:?}", TensorErrors::DeterminantZero).as_str());
    }

    #[test]
    #[should_panic]
    fn invalid_inversion_square_matrix_only() {
        let m1 = Matrix::<i32>::new(3, 2, (0..6).collect()).unwrap();
        inv(&m1).expect("TODO: panic message");
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
                .map(|x| (x as f64) / 6201.0.sqrt())
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
                    re: (x as f64) / 110.0.sqrt(),
                    im: (x as f64) / 110.0.sqrt(),
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
    fn test_householder() {
        let m1: Matrix<Complex64> = Tensor::<i32>::new(
            &shape![9],
            vec![
                    4, 1, 1,
                    1, 3, 0,
                    1, 0, 2,
                ],
            )
            .unwrap()
            .iter()
            .map(|x| Complex64::from_i32(*x).unwrap())
            .collect::<Tensor<Complex64>>()
            .reshape(&shape![3, 3])
            .unwrap()
            .try_into()
            .unwrap();

        let (q1, r1) = m1.householder();

        for i in 0..r1.shape[0] {
            if i >= r1.shape[1] - 1 {
                continue
            }

            assert!(r1.slice(i+1..r1.shape[0], i..i+1).iter().all(|x| {
                if x.abs() > 1e-10 {
                    println!("Failed for: {:?}", x.abs());
                }
                x.abs() <= 1e-10 }));
        }
        assert!(q1.contract_mul(&r1).unwrap().tensor.approx_eq(m1.clone().tensor, F64Margin::default().epsilon(1e-10)));

        println!("Passed test 1: square");

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


        println!("Passed test 2: tall");

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

        println!("Passed test 3: wide");
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
        println!("m2 ref: {m2_ref:?}");
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
}
