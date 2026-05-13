#[cfg(test)]
mod contract_mul_tests {
    use crate::definitions::errors::TensorErrors;
    use crate::definitions::matrix::Matrix;
    use crate::definitions::shape::Shape;
    use crate::definitions::tensor::Tensor;
    use crate::shape;
    
    #[test]
    fn contract_mul_with_scalar_tensors() {
        let scalar = Tensor::new(&shape![], vec![5]).unwrap();
        let t1 = Tensor::new(&shape![2, 2], vec![1, 2, 3, 4]).unwrap();

        // Scalar * Tensor
        let res1 = scalar.contract_mul(&t1).unwrap();
        let ans1 = Tensor::new(&shape![2, 2], vec![5, 10, 15, 20]).unwrap();
        assert_eq!(res1, ans1);

        // Tensor * Scalar
        let res2 = t1.contract_mul(&scalar).unwrap();
        assert_eq!(res2, ans1);
    }
    
    #[test]
    fn contract_mul_mt_with_scalar_tensors() {
        let scalar = Tensor::new(&shape![], vec![5]).unwrap();
        let t1 = Tensor::new(&shape![2, 2], vec![1, 2, 3, 4]).unwrap();

        // Scalar * Tensor
        let res1 = scalar.contract_mul_mt(&t1).unwrap();
        let ans1 = Tensor::new(&shape![2, 2], vec![5, 10, 15, 20]).unwrap();
        assert_eq!(res1, ans1);

        // Tensor * Scalar
        let res2 = t1.contract_mul_mt(&scalar).unwrap();
        assert_eq!(res2, ans1);
    }
    
    #[test]
    fn contract_mul_with_empty_tensor() {
        let t1 = Tensor::<i32>::new(&shape![2, 0], vec![]).unwrap();
        let t2 = Tensor::<i32>::new(&shape![0, 3], vec![]).unwrap();

        let res = t1.contract_mul(&t2).unwrap();
        let ans = Tensor::<i32>::new(&shape![2, 3], vec![0, 0, 0, 0, 0, 0]).unwrap();
        assert_eq!(res, ans);

        let t3 = Tensor::<i32>::new(&shape![0, 2], vec![]).unwrap();
        let t4 = Tensor::<i32>::new(&shape![2, 2], vec![1, 2, 3, 4]).unwrap();
        let res2 = t3.contract_mul(&t4).unwrap();
        assert_eq!(res2.shape.0, vec![0, 2]);
        assert_eq!(res2.elements.len(), 0);
    }
    
    #[test]
    fn contract_mul_mt_with_empty_tensor() {
        let t1 = Tensor::<i32>::new(&shape![2, 0], vec![]).unwrap();
        let t2 = Tensor::<i32>::new(&shape![0, 3], vec![]).unwrap();

        let res = t1.contract_mul_mt(&t2).unwrap();
        let ans = Tensor::<i32>::new(&shape![2, 3], vec![0, 0, 0, 0, 0, 0]).unwrap();
        assert_eq!(res, ans);

        let t3 = Tensor::<i32>::new(&shape![0, 2], vec![]).unwrap();
        let t4 = Tensor::<i32>::new(&shape![2, 2], vec![1, 2, 3, 4]).unwrap();
        let res2 = t3.contract_mul_mt(&t4).unwrap();
        assert_eq!(res2.shape.0, vec![0, 2]);
        assert_eq!(res2.elements.len(), 0);
    }
    
    #[test]
    fn mat_mul_with_empty_matrix() {
        let m1 = Matrix::<i32>::new(2, 0, vec![]).unwrap();
        let m2 = Matrix::<i32>::new(0, 3, vec![]).unwrap();

        let res = m1.mat_mul(&m2).unwrap();
        let ans = Matrix::<i32>::new(2, 3, vec![0, 0, 0, 0, 0, 0]).unwrap();
        assert_eq!(res, ans);

        let m3 = Matrix::<i32>::new(0, 2, vec![]).unwrap();
        let m4 = Matrix::<i32>::new(2, 2, vec![1, 2, 3, 4]).unwrap();
        let res2 = m3.mat_mul(&m4).unwrap();
        assert_eq!(res2.rows, 0);
        assert_eq!(res2.cols, 2);
        assert_eq!(res2.tensor.elements.len(), 0);
    }
    
    #[test]
    fn mat_mul_mt_with_empty_matrix() {
        let m1 = Matrix::<i32>::new(2, 0, vec![]).unwrap();
        let m2 = Matrix::<i32>::new(0, 3, vec![]).unwrap();

        let res = m1.mat_mul_mt(&m2).unwrap();
        let ans = Matrix::<i32>::new(2, 3, vec![0, 0, 0, 0, 0, 0]).unwrap();
        assert_eq!(res, ans);

        let m3 = Matrix::<i32>::new(0, 2, vec![]).unwrap();
        let m4 = Matrix::<i32>::new(2, 2, vec![1, 2, 3, 4]).unwrap();
        let res2 = m3.mat_mul_mt(&m4).unwrap();
        assert_eq!(res2.rows, 0);
        assert_eq!(res2.cols, 2);
        assert_eq!(res2.tensor.elements.len(), 0);
    }
    
    #[test]
    fn dot_scalar_tensors() {
        let t1 = Tensor::new(&shape![], vec![5]).unwrap();
        let t2 = Tensor::new(&shape![], vec![10]).unwrap();

        assert_eq!(t1.dot(&t2).unwrap(), 50);
    }

    #[test]
    fn dot_empty_tensors() {
        let t1 = Tensor::<i32>::new(&shape![2, 0], vec![]).unwrap();
        let t2 = Tensor::<i32>::new(&shape![2, 0], vec![]).unwrap();

        assert_eq!(t1.dot(&t2).unwrap(), 0);
    }

    #[test]
    fn dot_empty_matrices() {
        let m1 = Matrix::<i32>::new(2, 0, vec![]).unwrap();
        let m2 = Matrix::<i32>::new(2, 0, vec![]).unwrap();
        let m3 = Matrix::<i32>::new(0, 1, vec![]).unwrap();
        let m4 = Matrix::<i32>::new(0, 1, vec![]).unwrap();

        assert_eq!(m1.dot(&m2).unwrap(), 0);
        assert_eq!(m3.dot(&m4).unwrap(), 0);
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

        let res = m1.mat_mul(&m2).unwrap();
        let ans = Matrix::new(3, 2, vec![10, 13, 28, 40, 46, 67]).unwrap();

        assert_eq!(ans, res);
    }

    #[test]
    fn tensor_contraction_multiplication_shape_invalid() {
        let t1 = Tensor::<i32>::new(&shape![2, 3], (0..6).collect()).unwrap();
        let t2 = Tensor::<i32>::new(&shape![2, 2], (0..4).collect()).unwrap();
        let err = t1.contract_mul(&t2).unwrap_err();
        assert_eq!(err, TensorErrors::ShapesIncompatible);
        let err = t1.contract_mul_mt(&t2).unwrap_err();
        assert_eq!(err, TensorErrors::ShapesIncompatible);
    }

    #[test]
    fn invalid_shapes_for_mat_mul() {
        let m1 = Matrix::<i32>::rand(3, 3);
        let m2 = Matrix::<i32>::rand(2, 3);
        let err = m1.mat_mul(&m2).unwrap_err();
        assert_eq!(err, TensorErrors::ShapesIncompatible)
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
    fn dot_matrices() {
        let m1 = Matrix::new(10, 10, (0..100).collect()).unwrap();
        let m2 = Matrix::new(10, 10, (0..100).collect()).unwrap();

        assert_eq!(m1.dot(&m2).unwrap(), (99 * 100 * 199) / 6)
    }

    #[test]
    fn invalid_dot() {
        let t1 = Tensor::new(
            &shape![1, 2, 3],
            (0..6).collect(),
        ).unwrap();

        let t2 = Tensor::new(
            &shape![2, 1, 3],
            (0..6).collect(),
        ).unwrap();

        let err = t1.dot(&t2).unwrap_err();
        match err {
            TensorErrors::ShapesIncompatible => {},
             _ => panic!("Incorrect error"),
        }

        let m1 = Matrix::new(
            3, 2,
            (0..6).collect(),
        ).unwrap();
        let m2 = Matrix::new(
            2, 3,
            (0..6).collect(),
        ).unwrap();

        let err = m1.dot(&m2).unwrap_err();

        match err {
            TensorErrors::ShapesIncompatible => {},
            _ => panic!("Incorrect error"),
        }
    }
}