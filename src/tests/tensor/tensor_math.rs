#[cfg(test)]
mod tensor_math_tests {
    use crate::tensor::tensor::Shape;
    use crate::tensor::tensor::Tensor;
    use crate::tensor::tensor_math::{kronecker_product, Transpose};
    use crate::ts;

    #[test]
    fn add_tensors() {
        let t1 = Tensor::<i32>::new(
            &ts![2, 3],
            vec![
                0, 0, 0,
                2, 3, -1,
            ],
        ).unwrap();
        let t2 = Tensor::<i32>::new(
            &ts![2, 3],
            vec![
                1, 1, 1,
                2, 4, -10,
            ],
        ).unwrap();
        let ans = Tensor::<i32>::new(
            &ts![2, 3],
            vec![
                1, 1, 1,
                4, 7, -11,
            ],
        ).unwrap();
        assert_eq!(ans, t1 + t2);
    }

    #[test]
    fn subtract_tensors() {
        let t1 = Tensor::<i32>::new(
            &ts![2, 3],
            vec![
                0, 0, 0,
                2, 3, -1,
            ],
        ).unwrap();
        let t2 = Tensor::<i32>::new(
            &ts![2, 3],
            vec![
                1, 1, 1,
                2, 4, -10,
            ],
        ).unwrap();
        let ans = Tensor::<i32>::new(
            &ts![2, 3],
            vec![
                1, 1, 1,
                0, 1, -9,
            ],
        ).unwrap();
        assert_eq!(ans, t2 - t1);
    }

    #[test]
    fn multiply_tensors() {
        let t1 = Tensor::<i32>::new(
            &ts![2, 3],
            vec![
                0, 0, 0,
                2, 3, -1,
            ],
        ).unwrap();
        let t2 = &Tensor::<i32>::new(
            &ts![2, 3],
            vec![
                1, 1, 1,
                2, 4, -10,
            ],
        ).unwrap();
        let ans = Tensor::<i32>::new(
            &ts![2, 3],
            vec![
                0, 0, 0,
                4, 12, 10,
            ],
        ).unwrap();
        assert_eq!(ans, t1 * t2);
    }

    #[test]
    fn divide_tensors() {
        let t1 = Tensor::<i32>::new(
            &ts![2, 3],
            vec![
                0, 0, 0,
                2, 5, -1,
            ],
        ).unwrap();
        let t2 = &Tensor::<i32>::new(
            &ts![2, 3],
            vec![
                1, 1, 1,
                2, 4, -10,
            ],
        ).unwrap();
        let ans = Tensor::<i32>::new(
            &ts![2, 3],
            vec![
                0, 0, 0,
                1, 1, 0,
            ],
        ).unwrap();
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
        let t1 = Tensor::<i32>::new(
            &ts![2, 3],
            vec![
                0, 1, 2,
                -1, 4, -10,
            ],
        ).unwrap();
        let val = 5;

        let ans_add = Tensor::<i32>::new(
            &ts![2, 3],
            vec![
                5, 6, 7,
                4, 9, -5,
            ],
        ).unwrap();
        let ans_sub = Tensor::<i32>::new(
            &ts![2, 3],
            vec![
                -5, -4, -3,
                -6, -1, -15,
            ],
        ).unwrap();
        let ans_mul = Tensor::<i32>::new(
            &ts![2, 3],
            vec![
                0, 5, 10,
                -5, 20, -50,
            ],
        ).unwrap();
        let ans_div = Tensor::<i32>::new(
            &ts![2, 3],
            vec![
                0, 0, 0,
                0, 0, -2
            ],
        ).unwrap();
        assert_eq!(ans_add, &t1 + val);
        assert_eq!(ans_sub, t1.clone() - &val);
        assert_eq!(ans_mul, &t1 * val);
        assert_eq!(ans_div, t1 / val);
    }

    #[test]
    fn unary_minus() {
        let t1 = Tensor::<i32>::new(
            &ts![2, 3],
            vec![
                1, 3, -4,
                4, 5, 2,
            ],
        ).unwrap();
        let ans = Tensor::<i32>::new(
            &ts![2, 3],
            vec![
                -1, -3, 4,
                -4, -5, -2,
            ],
        ).unwrap();

        assert_eq!(ans, -t1);
    }

    #[test]
    fn transpose() {
        let t1 = Tensor::<i32>::new(
            &ts![2, 3, 4],
            (0..24).collect()
        ).unwrap();
        let mut t2 = Tensor::<i32>::new(
            &ts![2, 3],
            (0..6).collect()
        ).unwrap();
        let transposed_t1 = t1.clone().transpose(&Transpose::new(&vec![0, 2, 1]).unwrap()).unwrap();
        t2.transpose_in_place(&Transpose::new(&vec![1, 0]).unwrap()).unwrap();
        let transposed_t2 = t2.clone();
        let ans1 = Tensor::<i32>::new(
            &ts![2, 4, 3],
            vec![
                0, 4, 8,
                1, 5, 9,
                2, 6, 10,
                3, 7, 11,
                12, 16, 20,
                13, 17, 21,
                14, 18, 22,
                15, 19, 23,
            ],
        ).unwrap();
        let ans2 = Tensor::<i32>::new(
            &ts![3, 2],
            vec![
                0, 3,
                1, 4,
                2, 5,
            ],
        ).unwrap();

        assert_eq!(ans1, transposed_t1);
        assert_eq!(ans2, transposed_t2);
    }

    #[test]
    fn tensor_contraction_multiplication() {
        let t1 = Tensor::<i32>::new(
            &ts![2, 3],
            (0..6).collect()
        ).unwrap();
        let t2 = Tensor::<i32>::new(
            &ts![3, 3],
            (0..9).collect()
        ).unwrap();
        let ans = Tensor::<i32>::new(
            &ts![2, 3],
            vec![
                15, 18, 21,
                42, 54, 66
            ],
        ).unwrap();
        assert_eq!(ans, t1.contract_mul(&t2).unwrap());

        let t3 = Tensor::<i32>::new(
            &ts![2, 3],
            vec![
                5, 10, 2,
                3, -2, 1,
            ],
        ).unwrap();
        let t4 = Tensor::<i32>::new(
            &ts![3, 2, 2],
            (0..12).collect(),
        ).unwrap();
        let ans = Tensor::<i32>::new(
            &ts![2, 2, 2],
            vec![
                56, 73,
                90, 107,
                0, 2,
                4, 6,
            ],
        ).unwrap();
        assert_eq!(ans, t3.contract_mul(&t4).unwrap());
    }

    #[test]
    fn tensor_contraction_multiplication_shape_invalid() {
        let t1 = Tensor::<i32>::new(
            &ts![2, 3],
            (0..6).collect(),
        ).unwrap();
        let t2 = Tensor::<i32>::new(
          &ts![2, 2],
          (0..4).collect(),
        ).unwrap();
        t1.contract_mul(&t2).expect_err("Invalid shapes");
    }

    #[test]
    fn test_kronecker_product() {
        let t1 = Tensor::<i32>::new(
            &ts![2, 3],
            (0..6).collect(),
        ).unwrap();
        let t2 = Tensor::<i32>::new(
            &ts![5, 2, 2],
            (0..20).collect(),
        ).unwrap();
        let mut ans_vec = vec![0; 20];
        ans_vec.extend(0..20);
        ans_vec.extend((0..20).map(|i| i * 2));
        ans_vec.extend((0..20).map(|i| i * 3));
        ans_vec.extend((0..20).map(|i| i * 4));
        ans_vec.extend((0..20).map(|i| i * 5));
        let ans = Tensor::<i32>::new(
            &ts![10, 6, 2],
            ans_vec,
        ).unwrap();
        assert_eq!(kronecker_product(&t1, &t2), ans);
    }
}
