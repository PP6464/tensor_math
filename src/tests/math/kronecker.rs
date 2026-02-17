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
}