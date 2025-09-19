#[cfg(test)]
mod tensor_math_tests {
    use crate::tensor::tensor::{Tensor};
    use crate::tensor::tensor::{Shape, TensorErrors};
    use crate::tensor::tensor_math::{det, gaussian_pdf_multi_sigma, gaussian_pdf_single_sigma, gaussian_sample, householder, identity, inv, kronecker_product, pool, pool_avg, pool_max, pool_min, pool_sum, solve_cubic, solve_quadratic, solve_quartic, trace, Transpose};
    use crate::ts;
    use float_cmp::{approx_eq, assert_approx_eq, ApproxEq, F64Margin, FloatMargin};
    use num::complex::{Complex64, ComplexFloat};
    use std::f64::consts::PI;
    use std::ops::Add;
    use num::FromPrimitive;

    #[test]
    fn add_tensors() {
        let t1 = Tensor::<i32>::new(&ts![2, 3], vec![0, 0, 0, 2, 3, -1]).unwrap();
        let t2 = Tensor::<i32>::new(&ts![2, 3], vec![1, 1, 1, 2, 4, -10]).unwrap();
        let ans = Tensor::<i32>::new(&ts![2, 3], vec![1, 1, 1, 4, 7, -11]).unwrap();
        assert_eq!(ans, t1 + t2);
    }

    #[test]
    fn subtract_tensors() {
        let t1 = Tensor::<i32>::new(&ts![2, 3], vec![0, 0, 0, 2, 3, -1]).unwrap();
        let t2 = Tensor::<i32>::new(&ts![2, 3], vec![1, 1, 1, 2, 4, -10]).unwrap();
        let ans = Tensor::<i32>::new(&ts![2, 3], vec![1, 1, 1, 0, 1, -9]).unwrap();
        assert_eq!(ans, t2 - t1);
    }

    #[test]
    fn multiply_tensors() {
        let t1 = Tensor::<i32>::new(&ts![2, 3], vec![0, 0, 0, 2, 3, -1]).unwrap();
        let t2 = &Tensor::<i32>::new(&ts![2, 3], vec![1, 1, 1, 2, 4, -10]).unwrap();
        let ans = Tensor::<i32>::new(&ts![2, 3], vec![0, 0, 0, 4, 12, 10]).unwrap();
        assert_eq!(ans, t1 * t2);
    }

    #[test]
    fn divide_tensors() {
        let t1 = Tensor::<i32>::new(&ts![2, 3], vec![0, 0, 0, 2, 5, -1]).unwrap();
        let t2 = &Tensor::<i32>::new(&ts![2, 3], vec![1, 1, 1, 2, 4, -10]).unwrap();
        let ans = Tensor::<i32>::new(&ts![2, 3], vec![0, 0, 0, 1, 1, 0]).unwrap();
        assert_eq!(ans, t1 / t2);
    }

    #[test]
    #[should_panic]
    fn incompatible_shapes_for_elementwise_bin_op() {
        let t1 = Tensor::<i32>::rand(&ts![2, 3]);
        let t2 = Tensor::<i32>::rand(&ts![2, 2]);

        let _ = t1 + t2;
    }

    #[test]
    #[should_panic]
    fn divide_by_zero() {
        let t1 = Tensor::<i32>::rand(&ts![2, 3]);
        let mut t2 = Tensor::<i32>::rand(&ts![2, 2]);
        t2[&[1, 1]] = 0;
        let _ = t1 / t2;
    }

    #[test]
    fn single_element_implementations() {
        let t1 = Tensor::<i32>::new(&ts![2, 3], vec![0, 1, 2, -1, 4, -10]).unwrap();
        let val = 5;

        let ans_add = Tensor::<i32>::new(&ts![2, 3], vec![5, 6, 7, 4, 9, -5]).unwrap();
        let ans_sub = Tensor::<i32>::new(&ts![2, 3], vec![-5, -4, -3, -6, -1, -15]).unwrap();
        let ans_mul = Tensor::<i32>::new(&ts![2, 3], vec![0, 5, 10, -5, 20, -50]).unwrap();
        let ans_div = Tensor::<i32>::new(&ts![2, 3], vec![0, 0, 0, 0, 0, -2]).unwrap();
        assert_eq!(ans_add, &t1 + val);
        assert_eq!(ans_sub, t1.clone() - &val);
        assert_eq!(ans_mul, &t1 * val);
        assert_eq!(ans_div, t1 / val);
    }

    #[test]
    fn unary_minus() {
        let t1 = Tensor::<i32>::new(&ts![2, 3], vec![1, 3, -4, 4, 5, 2]).unwrap();
        let ans = Tensor::<i32>::new(&ts![2, 3], vec![-1, -3, 4, -4, -5, -2]).unwrap();

        assert_eq!(ans, -t1);
    }

    #[test]
    fn transpose() {
        let t1 = Tensor::<i32>::new(&ts![2, 3, 4], (0..24).collect()).unwrap();
        let mut t2 = Tensor::<i32>::new(&ts![2, 3], (0..6).collect()).unwrap();
        let transposed_t1 = t1
            .clone()
            .transpose(&Transpose::new(&vec![0, 2, 1]).unwrap())
            .unwrap();
        t2.transpose_in_place(&Transpose::new(&vec![1, 0]).unwrap())
            .unwrap();
        let transposed_t2 = t2.clone();
        let ans1 = Tensor::<i32>::new(
            &ts![2, 4, 3],
            vec![
                0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19,
                23,
            ],
        )
        .unwrap();
        let ans2 = Tensor::<i32>::new(&ts![3, 2], vec![0, 3, 1, 4, 2, 5]).unwrap();

        assert_eq!(ans1, transposed_t1);
        assert_eq!(ans2, transposed_t2);
    }

    #[test]
    fn tensor_contraction_multiplication() {
        let t1 = Tensor::<i32>::new(&ts![2, 3], (0..6).collect()).unwrap();
        let t2 = Tensor::<i32>::new(&ts![3, 3], (0..9).collect()).unwrap();
        let ans = Tensor::<i32>::new(&ts![2, 3], vec![15, 18, 21, 42, 54, 66]).unwrap();
        assert_eq!(ans, t1.contract_mul(&t2).unwrap());

        let t3 = Tensor::<i32>::new(&ts![2, 3], vec![5, 10, 2, 3, -2, 1]).unwrap();
        let t4 = Tensor::<i32>::new(&ts![3, 2, 2], (0..12).collect()).unwrap();
        let ans = Tensor::<i32>::new(&ts![2, 2, 2], vec![56, 73, 90, 107, 0, 2, 4, 6]).unwrap();
        assert_eq!(ans, t3.contract_mul(&t4).unwrap());
    }

    #[test]
    fn tensor_contraction_multiplication_shape_invalid() {
        let t1 = Tensor::<i32>::new(&ts![2, 3], (0..6).collect()).unwrap();
        let t2 = Tensor::<i32>::new(&ts![2, 2], (0..4).collect()).unwrap();
        t1.contract_mul(&t2).expect_err("Invalid shapes");
    }

    #[test]
    fn test_kronecker_product() {
        let t1 = Tensor::<i32>::new(&ts![2, 3], (0..6).collect()).unwrap();
        let t2 = Tensor::<i32>::new(&ts![5, 2, 2], (0..20).collect()).unwrap();
        let mut ans_vec = vec![0; 20];
        ans_vec.extend(0..20);
        ans_vec.extend((0..20).map(|i| i * 2));
        ans_vec.extend((0..20).map(|i| i * 3));
        ans_vec.extend((0..20).map(|i| i * 4));
        ans_vec.extend((0..20).map(|i| i * 5));
        let ans = Tensor::<i32>::new(&ts![10, 6, 2], ans_vec).unwrap();
        assert_eq!(kronecker_product(&t1, &t2), ans);
    }

    #[test]
    fn test_trace() {
        let t1 = Tensor::<i32>::new(&ts![2, 2], (0..4).collect()).unwrap();
        assert_eq!(3, trace(&t1));
    }

    #[test]
    #[should_panic]
    fn invalid_trace_not_mat() {
        let t1 = Tensor::<i32>::new(&ts![2, 2, 3], (0..12).collect()).unwrap();
        trace(&t1);
    }

    #[test]
    #[should_panic]
    fn invalid_trace_non_square() {
        let t1 = Tensor::<i32>::new(&ts![2, 3], (0..6).collect()).unwrap();
        trace(&t1);
    }

    #[test]
    fn determinant() {
        let t1 = Tensor::<i32>::new(&ts![3, 3], vec![5, -2, 1, 8, 9, -5, 1, 0, 2]).unwrap();
        assert_eq!(det(&t1), 123);
    }

    #[test]
    #[should_panic]
    fn invalid_det_matrix_only() {
        let t1 = Tensor::<i32>::new(&ts![3, 3, 3], (0..27).collect()).unwrap();
        det(&t1);
    }

    #[test]
    #[should_panic]
    fn invalid_det_square_matrix_only() {
        let t1 = Tensor::<i32>::new(&ts![3, 2], (0..6).collect()).unwrap();
        det(&t1);
    }

    #[test]
    fn inverse() {
        let t1 = Tensor::<f64>::new(
            &ts![3, 3],
            vec![3.0, 4.0, 5.0, 2.0, -1.0, 4.0, 3.0, -5.0, -10.0],
        )
        .unwrap();
        let inverse = inv(&t1).unwrap();
        let ans = Tensor::<f64>::new(
            &ts![3, 3],
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
        assert!(t1
            .contract_mul(&inverse)
            .unwrap()
            .enumerated_iter()
            .all(|(i, x)| { approx_eq!(f64, x, identity(3)[&i], epsilon = 1e-15) }));
    }

    #[test]
    fn inverse_det_0() {
        let t1 = Tensor::<f64>::new(&ts![3, 3], vec![0.0; 9]).unwrap();
        inv(&t1).expect_err(format!("{:?}", TensorErrors::DeterminantZero).as_str());
    }

    #[test]
    #[should_panic]
    fn invalid_inversion_matrix_only() {
        let t1 = Tensor::<i32>::new(&ts![3, 3, 3], (0..27).collect()).unwrap();
        inv(&t1).expect("TODO: panic message");
    }

    #[test]
    #[should_panic]
    fn invalid_inversion_square_matrix_only() {
        let t1 = Tensor::<i32>::new(&ts![3, 2], (0..6).collect()).unwrap();
        inv(&t1).expect("TODO: panic message");
    }

    #[test]
    fn pool_tensor() {
        let t1: Tensor<f64> = Tensor::<i32>::new(
            &ts![3, 3, 3],
            vec![
                1, 5, -1, 2, 3, -5, 12, 10, -10, 1, -4, 2, 9, 6, 8, -1, 0, -1, -8, 7, 4, 5, 1, 2,
                -5, 3, 1,
            ],
        )
        .unwrap()
        .transform_elementwise(|x| x.into());

        let avg_pool = pool(&t1, pool_avg, &ts![2, 2, 2], &ts![2, 2, 2]);
        let sum_pool = pool(&t1, pool_sum, &ts![2, 2, 2], &ts![2, 2, 2]);
        let max_pool = pool(&t1, pool_max, &ts![2, 2, 2], &ts![1, 1, 1]);
        let min_pool = pool(&t1, pool_min, &ts![3, 1, 1], &ts![3, 1, 1]);

        let sum_ans = Tensor::<f64>::new(
            &ts![2, 2, 2],
            vec![23.0, 4.0, 21.0, -11.0, 5.0, 6.0, -2.0, 1.0],
        )
        .unwrap();
        let avg_ans = Tensor::<f64>::new(
            &ts![2, 2, 2],
            vec![2.875, 1.0, 5.25, -5.5, 1.25, 3.0, -1.0, 1.0],
        )
        .unwrap();
        let max_ans = Tensor::<f64>::new(
            &ts![3, 3, 3],
            vec![
                9.0, 8.0, 8.0, 12.0, 10.0, 8.0, 12.0, 10.0, -1.0, 9.0, 8.0, 8.0, 9.0, 8.0, 8.0,
                3.0, 3.0, 1.0, 7.0, 7.0, 4.0, 5.0, 3.0, 2.0, 3.0, 3.0, 1.0,
            ],
        )
        .unwrap();
        let min_ans = Tensor::<f64>::new(
            &ts![1, 3, 3],
            vec![-8.0, -4.0, -1.0, 2.0, 1.0, -5.0, -5.0, 0.0, -10.0],
        )
        .unwrap();

        assert_eq!(avg_pool, avg_ans);
        assert_eq!(sum_pool, sum_ans);
        assert_eq!(max_pool, max_ans);
        assert_eq!(min_pool, min_ans);
    }

    #[test]
    fn normalised() {
        let t1 = Tensor::<f64>::new(
            &ts![3, 3, 3],
            (0..27).map(|x| x as f64).collect::<Vec<f64>>(),
        )
        .unwrap();

        let t1_norm_l1 = t1.clone().norm_l1();
        let t1_norm_l2 = t1.norm_l2();

        let t1_norm_l1_ans = Tensor::<f64>::new(
            &ts![3, 3, 3],
            (0..27).map(|x| (x as f64) / 351.0).collect::<Vec<f64>>(),
        )
        .unwrap();
        let t1_norm_l2_ans = Tensor::<f64>::new(
            &ts![3, 3, 3],
            (0..27)
                .map(|x| (x as f64) / 6201.0.sqrt())
                .collect::<Vec<f64>>(),
        )
        .unwrap();

        assert_eq!(t1_norm_l1, t1_norm_l1_ans);
        assert_eq!(t1_norm_l2, t1_norm_l2_ans);

        let t2 = Tensor::<Complex64>::new(
            &&ts![2, 3],
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
            &ts![2, 3],
            (0..6)
                .map(|x| Complex64 {
                    re: (x as f64) / (15.0 * f64::sqrt(2.0)),
                    im: (x as f64) / (15.0 * f64::sqrt(2.0)),
                })
                .collect(),
        )
        .unwrap();
        let t2_norm_l2_ans = Tensor::<Complex64>::new(
            &ts![2, 3],
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
        let t1 = gaussian_sample(1.0, &ts![3, 3, 3], -10.0, 10.0);

        assert!(t1.iter().all(|x| -10.0 <= *x));
        assert!(t1.iter().all(|x| 10.0 >= *x));

        println!("{:?}", t1);
    }

    #[test]
    #[should_panic]
    fn rand_gaussian_sample_invalid_min_more_than_max() {
        gaussian_sample(1.0, &ts![3, 3, 3], 10.0, -10.0);
    }

    #[test]
    #[should_panic]
    fn rand_gaussian_sample_invalid_min_eq_max() {
        gaussian_sample(1.0, &ts![3, 3, 3], 10.0, 10.0);
    }

    #[test]
    #[should_panic]
    fn rand_gaussian_sample_invalid_neg_sigma() {
        gaussian_sample(-1.0, &ts![3, 3, 3], -100.0, -10.0);
    }

    #[test]
    fn single_sigma_gaussian_pdf() {
        let t1 = gaussian_pdf_single_sigma(0.5, &ts![5, 3]);
        let ans = (Tensor::<f64>::new(
            &ts![5, 3],
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
        let t1 = gaussian_pdf_multi_sigma(vec![0.25, 0.4], &ts![5, 3]);
        let ans = (Tensor::<f64>::new(
            &ts![5, 3],
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
        gaussian_pdf_multi_sigma(vec![0.0, 0.5], &ts![1, 1]);
    }

    #[test]
    #[should_panic]
    fn invalid_multi_sigma_gaussian_pdf_negative_sigma() {
        gaussian_pdf_multi_sigma(vec![0.5, -0.5], &ts![1, 1]);
    }

    #[test]
    #[should_panic]
    fn invalid_multi_sigma_gaussian_pdf_sigma_len_invalid() {
        gaussian_pdf_multi_sigma(vec![0.1, 0.5], &ts![1, 1, 3]);
    }

    #[test]
    fn test_householder() {
        let t1: Tensor<Complex64> = Tensor::<i32>::new(
                &ts![9],
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
            .reshape(&ts![3, 3])
            .unwrap();

        let (q1, r1) = householder(&t1);

        for i in 0..r1.shape[0] {
            if i >= r1.shape[1] - 1 {
                continue
            }
            assert!(r1.slice(&[i+1..r1.shape[0], i..i+1]).iter().all(|x| x.abs() <= 1e-10));
        }
        assert!(q1.contract_mul(&r1).unwrap().approx_eq(t1, F64Margin::default().epsilon(1e-10)));

        println!("Passed test 1: square");

        let t2 = Tensor::<Complex64>::new(
            &ts![3, 2],
            vec![
                Complex64 { re: 4.0, im: 1.0 }, Complex64 { re: -5.0, im: -2.0 },
                Complex64 { re: 5.0, im: -4.0 }, Complex64 { re: 5.0, im: 3.0 },
                Complex64 { re: 0.0, im: 0.0 }, Complex64 { re: 1.0, im: -1.0 },
            ],
        ).unwrap();
        let (q2, r2) = householder(&t2);

        for i in 0..r2.shape[0] {
            if i >= r2.shape[1] - 1 {
                continue
            }
            assert!(r2.slice(&[i+1..r2.shape[0], i..i+1]).iter().all(|x| x.abs() <= 1e-10));
        }
        assert!(q2.contract_mul(&r2).unwrap().approx_eq(t2, F64Margin::default().epsilon(1e-10)));


        println!("Passed test 2: tall");

        let t3 = Tensor::<Complex64>::new(
            &ts![2, 3],
            vec![
                Complex64 { re: -4.0, im: -1.0 }, Complex64 { re: 5.0, im: -3.0 }, Complex64 { re: 2.0, im: -4.0 },
                Complex64 { re: -5.0, im: 2.0 }, Complex64 { re: 2.0, im: -1.0 }, Complex64 { re: 4.0, im: -1.0 },
            ],
        ).unwrap();
        let (q3, r3) = householder(&t3);

        for i in 0..(r3.shape[0] - 1) {
            if i >= r3.shape[1] - 1 {
                continue
            }
            assert!(r3.slice(&[i+1..r3.shape[0], i..i+1]).iter().all(|x| x.abs() <= 1e-10));
        }
        assert!(q3.contract_mul(&r3).unwrap().approx_eq(t3, F64Margin::default().epsilon(1e-10)));

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
        let t1 = Tensor::<Complex64>::new(
            &ts![10, 30, 20],
            (0..6000).map(|i| Complex64::new(i as f64, i as f64)).collect(),
        ).unwrap();
        let t2 = Tensor::<Complex64>::new(
            &ts![20, 100],
            (0..2000).map(|i| Complex64::new(i as f64, i as f64)).collect(),
        ).unwrap();

        let ans = t1.clone().contract_mul(&t2).unwrap();
        let mt_ans = t1.contract_mul_mt(&t2).unwrap();

        assert_eq!(ans, mt_ans);
    }
}
