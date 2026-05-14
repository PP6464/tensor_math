#[cfg(test)]
mod kronecker_tests {
    use crate::definitions::matrix::Matrix;
    use crate::definitions::shape::Shape;
    use crate::definitions::tensor::Tensor;
    use crate::shape;

    #[test]
    fn kronecker_product() {
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
        let t1 = Tensor::<i32>::new(&shape![5, 2, 2], (0..20).collect()).unwrap();
        let t2 = Tensor::<i32>::new(&shape![2, 3], (0..6).collect()).unwrap();

        let ans = t1.kronecker(&t2);
        let mt_ans = t1.kronecker_mt(&t2);

        assert_eq!(mt_ans, ans);

        let ans = t2.kronecker(&t1);
        let mt_ans = t2.kronecker_mt(&t1);

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
    fn kronecker_product_with_scalar_tensor() {
        let t = Tensor::<i32>::new(&shape![2, 2], vec![1, 2, 3, 4]).unwrap();
        let s = Tensor::<i32>::new(&shape![], vec![10]).unwrap();

        let ans_a_s = Tensor::<i32>::new(&shape![2, 2], vec![10, 20, 30, 40]).unwrap();
        assert_eq!(t.kronecker(&s), ans_a_s);
        assert_eq!(t.kronecker_mt(&s), ans_a_s);

        let ans_s_a = Tensor::<i32>::new(&shape![2, 2], vec![10, 20, 30, 40]).unwrap();
        assert_eq!(s.kronecker(&t), ans_s_a);
        assert_eq!(s.kronecker_mt(&t), ans_s_a);
    }
    
    #[test]
    fn kronecker_product_with_empty_matrices_and_tensors() {
        let empty_t = Tensor::<i32>::new(&shape![0, 2], vec![]).unwrap();
        let full_t = Tensor::<i32>::new(&shape![2, 2], vec![1, 2, 3, 4]).unwrap();

        // [0, 2] x [2, 2] -> [0, 4]
        let res1 = empty_t.kronecker(&full_t);
        assert_eq!(res1.shape, shape![0, 4]);
        assert_eq!(res1.elements.len(), 0);
        assert_eq!(empty_t.kronecker_mt(&full_t), res1);

        // [2, 2] x [0, 2] -> [0, 4]
        let res2 = full_t.kronecker(&empty_t);
        assert_eq!(res2.shape, shape![0, 4]);
        assert_eq!(res2.elements.len(), 0);
        assert_eq!(full_t.kronecker_mt(&empty_t), res2);

        let empty_m = Matrix::<i32>::new(0, 5, vec![]).unwrap();
        let full_m = Matrix::<i32>::new(2, 2, vec![1, 2, 3, 4]).unwrap();

        // (0x5) x (2x2) -> (0x10)
        let res_m1 = empty_m.kronecker(&full_m);
        assert_eq!(res_m1.rows, 0);
        assert_eq!(res_m1.cols, 10);
        assert_eq!(res_m1.tensor.elements.len(), 0);
        assert_eq!(empty_m.kronecker_mt(&full_m), res_m1);

        // (2x2) x (0x5) -> (0x10)
        let res_m2 = full_m.kronecker(&empty_m);
        assert_eq!(res_m2.rows, 0);
        assert_eq!(res_m2.cols, 10);
        assert_eq!(res_m2.tensor.elements.len(), 0);
        assert_eq!(full_m.kronecker_mt(&empty_m), res_m2);
    }
}