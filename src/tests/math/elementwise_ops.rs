#[cfg(test)]
mod elementwise_ops_tests {
    use crate::definitions::matrix::Matrix;
    use crate::definitions::shape::Shape;
    use crate::definitions::tensor::Tensor;
    use crate::definitions::traits::IntoMatrix;
    use crate::shape;
    use crate::utilities::matrix::identity;
    use num::complex::Complex64;
    use crate::tests::test_helpers::assert_panics;

    #[test]
    fn add_tensors() {
        let t1 = Tensor::<i32>::new(&shape![2, 3], vec![0, 0, 0, 2, 3, -1]).unwrap();
        let t2 = Tensor::<i32>::new(&shape![2, 3], vec![1, 1, 1, 2, 4, -10]).unwrap();
        let ans = Tensor::<i32>::new(&shape![2, 3], vec![1, 1, 1, 4, 7, -11]).unwrap();
        assert_eq!(ans, t1 + t2);
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
    fn incompatible_shapes_for_elementwise_bin_op() {
        let t1 = Tensor::<i32>::rand(&shape![2, 3]);
        let t2 = Tensor::<i32>::rand(&shape![2, 2]);

        assert_panics(|| { let _ = &t1 + &t2; });
        assert_panics(|| { let _ = &t1 + t2.clone(); });
        assert_panics(|| { let _ = t1.clone() + &t2; });
        assert_panics(|| { let _ = t1 + t2; });
    }
    
    #[test]
    fn mat_operators() {
        let m1 = -identity::<Complex64>(2) / Complex64::from(2.0);
        let m2 = identity::<Complex64>(2) * (Complex64::ONE + Complex64::I);
        let ans = Matrix::new(
            2,
            2,
            vec![
                Complex64 { re: 0.5, im: 1.0 },
                Complex64 { re: 0.0, im: 0.0 },
                Complex64 { re: 0.0, im: 0.0 },
                Complex64 { re: 0.5, im: 1.0 },
            ],
        )
        .unwrap();

        assert_eq!(m1 + m2, ans);
    }

    #[test]
    fn incompatible_shapes_for_elementwise_bin_op_mat() {
        let m1 = Matrix::<i32>::rand(2, 2);
        let m2 = Matrix::<i32>::rand(3, 3);

        assert_panics(|| { let _ = &m1 + &m2; });
        assert_panics(|| { let _ = &m1 + m2.clone(); });
        assert_panics(|| { let _ = m1.clone() + &m2; });
        assert_panics(|| { let _ = m1 + m2; });
    }

    #[test]
    fn unary_minus() {
        let t1 = Tensor::<i32>::new(&shape![2, 3], vec![1, 3, -4, 4, 5, 2]).unwrap();
        let ans = Tensor::<i32>::new(&shape![2, 3], vec![-1, -3, 4, -4, -5, -2]).unwrap();

        assert_eq!(ans, -t1);
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
    fn mat_ref_and_single_val_op() {
        let m1 = vec![1, 2, 3, 4].into_matrix().reshape(2, 2).unwrap();

        let ans = vec![0, 1, 2, 3].into_matrix().reshape(2, 2).unwrap();

        assert_eq!(&m1 - 1, ans);
    }

    #[test]
    fn mat_and_single_val_ref_op() {
        let m1 = vec![1, 2, 3, 4].into_matrix().reshape(2, 2).unwrap();
        let ans = vec![0, 1, 2, 3].into_matrix().reshape(2, 2).unwrap();
        assert_eq!(m1 - &1, ans);
    }

    #[test]
    fn mat_and_mat_ref() {
        let m1 = vec![1, 2, 3, 4].into_matrix().reshape(2, 2).unwrap();
        let m2 = Matrix::from_value(2, 2, 1);
        let ans = vec![0, 1, 2, 3].into_matrix().reshape(2, 2).unwrap();

        assert_eq!(m1 - &m2, ans);
    }

    #[test]
    fn empty_tensor_ops() {
        let t1 = Tensor::<i32>::new(&shape![0, 3], vec![]).unwrap();
        let t2 = Tensor::<i32>::new(&shape![0, 3], vec![]).unwrap();
        let ans = Tensor::<i32>::new(&shape![0, 3], vec![]).unwrap();

        assert_eq!(&t1 + t2, ans);
        assert_eq!(&t1 * 5, ans);
    }

    #[test]
    fn empty_matrix_ops() {
        let m1 = Matrix::<f64>::new(0, 2, vec![]).unwrap();
        let m2 = Matrix::<f64>::new(0, 2, vec![]).unwrap();
        let ans = Matrix::<f64>::new(0, 2, vec![]).unwrap();

        assert_eq!(m1 + m2, ans);
    }

    #[test]
    fn rank_0_tensor_ops() {
        let t1 = Tensor::<i32>::new(&shape![], vec![10]).unwrap();
        let t2 = Tensor::<i32>::new(&shape![], vec![5]).unwrap();
        let ans_add = Tensor::<i32>::new(&shape![], vec![15]).unwrap();
        let ans_mul = Tensor::<i32>::new(&shape![], vec![50]).unwrap();

        assert_eq!(&t1 + &t2, ans_add);
        assert_eq!(t1 * t2, ans_mul);
    }
}
