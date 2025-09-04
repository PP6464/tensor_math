#[cfg(test)]
mod tensor_math_tests {
    use crate::tensor::tensor::Shape;
    use crate::tensor::tensor::Tensor;
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
}
