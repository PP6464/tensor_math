#[cfg(test)]
mod matrix_utils_tests {
    use crate::definitions::errors::TensorErrors;
    use crate::definitions::matrix::Matrix;
    use crate::utilities::matrix::{
        identity, pool_avg_mat, pool_max_mat, pool_min_mat, pool_sum_mat,
    };
    use num::complex::Complex64;

    #[test]
    fn concat() {
        let m1 = Matrix::new(3, 3, (0..9).collect()).unwrap();
        let m2 = Matrix::new(3, 2, (9..15).collect()).unwrap();
        let res = m1.concat_cols(&m2).unwrap();
        let ans =
            Matrix::new(3, 5, vec![0, 1, 2, 9, 10, 3, 4, 5, 11, 12, 6, 7, 8, 13, 14]).unwrap();
        assert_eq!(ans, res);
    }
    
    #[test]
    fn concat_with_empty_matrix() {
        let m1 = Matrix::new(3, 3, (0..9).collect()).unwrap();
        let m2 = Matrix::new(3, 0, vec![]).unwrap();
        let res = m1.concat_cols(&m2).unwrap();
        assert_eq!(res, m1);

        let m3 = Matrix::new(0, 3, vec![]).unwrap();
        let res = m1.concat_rows(&m3).unwrap();
        assert_eq!(res, m1);
    }
    
    #[test]
    fn concat_multithreaded_with_empty_matrix() {
        let m1 = Matrix::new(30, 30, (0..900).collect()).unwrap();
        let m2 = Matrix::new(30, 0, vec![]).unwrap();
        let res = m1.concat_cols_mt(&m2).unwrap();
        assert_eq!(res, m1);

        let m3 = Matrix::new(0, 30, vec![]).unwrap();
        let res = m1.concat_rows_mt(&m3).unwrap();
        assert_eq!(res, m1);
    }

    #[test]
    fn invalid_concat() {
        let m1 = Matrix::new(3, 3, (0..9).collect()).unwrap();
        let m2 = Matrix::new(3, 2, (0..6).collect()).unwrap();
        assert_eq!(
            m1.concat(&m2, 0).unwrap_err(),
            TensorErrors::ShapesIncompatible
        );
    }

    #[test]
    fn reshape_correctly() {
        let m1 = (0..6).collect::<Matrix<_>>();
        let ans = Matrix::new(2, 3, (0..6).collect()).unwrap();
        assert_eq!(m1.reshape(2, 3).unwrap(), ans);
        
        let m2 = Matrix::<usize>::new(0, 2, vec![]).unwrap();
        let ans = m2.reshape(0, 100).unwrap();
        assert_eq!(ans, Matrix::new(0, 100, vec![]).unwrap());
    }

    #[test]
    fn invalid_reshape() {
        let m1 = Matrix::new(3, 3, (0..9).collect()).unwrap();
        assert_eq!(
            m1.reshape(2, 3).unwrap_err(),
            TensorErrors::ShapeSizeDoesNotMatch
        );

        let m2 = Matrix::<usize>::new(3, 0, vec![]).unwrap();
        assert_eq!(
            m2.reshape(3, 1).unwrap_err(),
            TensorErrors::ShapeSizeDoesNotMatch
        )
    }
    
    #[test]
    fn flip_axes_with_empty_matrix() {
        let m1 = Matrix::<i32>::new(0, 3, vec![]).unwrap();
        assert_eq!(m1.flip_cols(), m1);
        assert_eq!(m1.flip_rows(), m1);
        assert_eq!(m1.flip(), m1);

        assert_eq!(m1.flip_cols_mt(), m1);
        assert_eq!(m1.flip_rows_mt(), m1);
        assert_eq!(m1.flip_mt(), m1);

        let m2 = Matrix::<i32>::new(3, 0, vec![]).unwrap();
        assert_eq!(m2.flip_cols(), m2);
        assert_eq!(m2.flip_rows(), m2);
        assert_eq!(m2.flip(), m2);
    }
    
    #[test]
    fn collect_empty_matrix() {
        let empty_vec: Vec<i32> = vec![];
        let m: Matrix<i32> = empty_vec.into_iter().collect();

        assert_eq!(m.rows, 1);
        assert_eq!(m.cols, 0);
        assert!(m.elements.is_empty());
    }

    #[test]
    fn random() {
        Matrix::<i32>::rand(2, 3);
    }

    #[test]
    fn transform_elementwise() {
        let m1 = Matrix::<f64>::new(2, 2, (0..4).map(f64::from).collect()).unwrap();
        let m1 = m1.map(f64::sqrt);
        let ans = Matrix::<f64>::new(2, 2, vec![0.0, 1.0, 2.0_f64.sqrt(), 3.0_f64.sqrt()]).unwrap();
        assert!((ans - m1).into_iter().map(f64::abs).sum::<f64>() < 1e-15);
    }

    #[test]
    fn slicing() {
        let m1 = Matrix::<f64>::new(4, 4, (0..16).map(f64::from).collect()).unwrap();
        let slice = m1.slice(1..3, 1..3).unwrap();
        let ans = Matrix::<f64>::new(2, 2, vec![5.0, 6.0, 9.0, 10.0]).unwrap();
        assert_eq!(slice, ans);
    }

    #[test]
    #[should_panic]
    fn invalid_slice_out_of_bounds() {
        let m1 = Matrix::<f64>::new(4, 4, (0..16).map(f64::from).collect()).unwrap();
        m1.slice(1..5, 1..2).unwrap();
    }

    #[test]
    fn slicing_mut() {
        let mut m1 = Matrix::<f64>::new(4, 4, (0..16).map(f64::from).collect()).unwrap();
        let mut slice = m1.slice_mut(1..3, 1..3).unwrap();
        slice[(0, 0)] = 69.0;
        slice[(0, 1)] = 42.0;
        slice[&[1, 0]] = -20.0;
        slice[&[1, 1]] = 91.0;
        let ans = Matrix::<f64>::new(
            4,
            4,
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 69.0, 42.0, 7.0, 8.0, -20.0, 91.0, 11.0, 12.0, 13.0, 14.0,
                15.0,
            ],
        )
        .unwrap();
        assert_eq!(m1, ans);
    }

    #[test]
    fn set_all() {
        let mut m1 = Matrix::<f64>::new(4, 4, (0..16).map(f64::from).collect()).unwrap();
        let mut slice_mut = m1.slice_mut(1..3, 1..3).unwrap();
        let inserted = Matrix::<f64>::new(2, 2, (0..4).map(f64::from).collect()).unwrap();
        slice_mut.set_all(&inserted).unwrap();

        let ans = Matrix::<f64>::new(
            4,
            4,
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 7.0, 8.0, 2.0, 3.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            ],
        )
        .unwrap();
        assert_eq!(m1, ans);
    }

    #[test]
    #[should_panic]
    fn set_all_fail() {
        let mut m1 = Matrix::<f64>::new(4, 4, (0..16).map(f64::from).collect()).unwrap();
        let mut slice_mut = m1.slice_mut(1..3, 1..3).unwrap();

        slice_mut.set_all(&Matrix::zeros(2, 1)).unwrap();
    }

    #[test]
    #[should_panic]
    fn slice_mut_out_of_bounds() {
        let mut m1 = Matrix::<f64>::new(4, 4, (0..16).map(f64::from).collect()).unwrap();
        let mut slice = m1.slice_mut(1..3, 1..3).unwrap();
        slice[&[3, 1]] = 69.0;
    }

    #[test]
    fn concat_multithreaded() {
        let m1 = identity::<Complex64>(10).reshape(20, 5).unwrap();
        let m2 = (0..500)
            .collect::<Matrix<_>>()
            .reshape(100, 5)
            .unwrap()
            .map(f64::from)
            .map(Complex64::from);

        let ans = m1.concat(&m2, 0).unwrap();
        let mt_ans = m1.concat_mt(&m2, 0).unwrap();

        assert_eq!(ans, mt_ans);
    }

    #[test]
    fn transpose() {
        let m1 = Matrix::<i32>::new(3, 3, (0..9).collect()).unwrap();
        let t = m1.transpose();
        let ans = Matrix::<i32>::new(3, 3, vec![0, 3, 6, 1, 4, 7, 2, 5, 8]).unwrap();

        assert_eq!(ans, t);
    }

    #[test]
    fn transpose_mt() {
        let m1 = Matrix::<i32>::rand(10, 5);

        let ans = m1.transpose();
        let mt_ans = m1.transpose_mt();

        assert_eq!(mt_ans, ans);
    }

    #[test]
    fn clipping() {
        let m1 = Matrix::<i32>::new(3, 3, (0..9).collect()).unwrap();
        let res = m1.clip(3, 6);
        let ans = Matrix::new(3, 3, vec![3, 3, 3, 3, 4, 5, 6, 6, 6]).unwrap();

        assert_eq!(res, ans);
    }

    #[test]
    fn pool() {
        let m1 = Matrix::<f64>::new(3, 3, (0..9).map(f64::from).collect()).unwrap();
        let sum = m1.pool(pool_sum_mat, (2, 2), (2, 2), 0.0).unwrap();
        let min = m1.pool(pool_min_mat, (2, 2), (1, 2), 0.0).unwrap();
        let avg = m1.pool(pool_avg_mat, (1, 1), (1, 1), 0.0).unwrap();
        let max = m1.pool(pool_max_mat, (1, 1), (2, 2), 0.0).unwrap();

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
    fn pool_mt() {
        let m1 = Matrix::<f64>::rand(20, 20);

        let ans = m1.pool(pool_avg_mat, (5, 4), (2, 3), 0.0).unwrap();
        let mt_ans = m1.pool_mt(&pool_avg_mat, (5, 4), (2, 3), 0.0).unwrap();

        assert_eq!(ans, mt_ans);
    }

    #[test]
    fn flip() {
        let m1 = Matrix::<i32>::new(2, 3, vec![0, 1, 2, 3, 4, 5]).unwrap();
        let res_cols = m1.flip_cols();
        let res_rows = m1.flip_rows();
        let res = m1.flip();

        let ans_cols = Matrix::<i32>::new(2, 3, vec![3, 4, 5, 0, 1, 2]).unwrap();
        let ans_rows = Matrix::<i32>::new(2, 3, vec![2, 1, 0, 5, 4, 3]).unwrap();
        let ans = Matrix::<i32>::new(2, 3, vec![5, 4, 3, 2, 1, 0]).unwrap();

        assert_eq!(res_cols, ans_cols);
        assert_eq!(res_rows, ans_rows);
        assert_eq!(res, ans);
    }

    #[test]
    fn flip_mt() {
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
    fn invalid_slice_mut_index_ranges_invalid() {
        let mut m1 = Matrix::<i32>::new(3, 3, (0..9).collect()).unwrap();

        let err = m1.slice_mut(10..2, 1..2).unwrap_err();
        match err {
            TensorErrors::InvalidInterval { .. } => {}
            _ => panic!("Incorrect error"),
        }

        let err = m1.slice_mut(1..2, 10..2).unwrap_err();
        match err {
            TensorErrors::InvalidInterval { .. } => {}
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn invalid_slice_mut_indices_out_of_bounds() {
        let mut m1 = Matrix::<i32>::new(3, 3, (0..9).collect()).unwrap();

        let err = m1.slice_mut(4..10, 1..2).unwrap_err();
        match err {
            TensorErrors::SliceIndicesOutOfBounds { .. } => {}
            _ => panic!("Incorrect error"),
        }

        let err = m1.slice_mut(1..2, 4..10).unwrap_err();
        match err {
            TensorErrors::SliceIndicesOutOfBounds { .. } => {}
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn pooling_invalid_shape() {
        let m1 = Matrix::<f64>::new(3, 3, (0..9).map(f64::from).collect()).unwrap();

        let err = m1.pool(pool_avg_mat, (0, 1), (1, 1), 0.0).unwrap_err();
        match err {
            TensorErrors::ShapeContainsZero => {}
            _ => panic!("Incorrect error"),
        }

        let err = m1
            .pool_indexed(|_, m| pool_max_mat(m), (1, 1), (0, 1), 0.0)
            .unwrap_err();
        match err {
            TensorErrors::ShapeContainsZero => {}
            _ => panic!("Incorrect error"),
        }

        let err = m1.pool_mt(&pool_min_mat, (1, 0), (1, 1), 0.0).unwrap_err();
        match err {
            TensorErrors::ShapeContainsZero => {}
            _ => panic!("Incorrect error"),
        }

        let err = m1
            .pool_indexed_mt(&|_, m| pool_sum_mat(m), (1, 1), (1, 0), 0.0)
            .unwrap_err();
        match err {
            TensorErrors::ShapeContainsZero => {}
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn slice_with_empty_indices() {
        let m1 = Matrix::<f64>::new(4, 4, (0..16).map(f64::from).collect()).unwrap();
        let slice = m1.slice(1..1, 1..3).unwrap();
        assert_eq!(slice.rows, 0);
        assert_eq!(slice.cols, 2);

        let slice = m1.slice(1..3, 1..1).unwrap();
        assert_eq!(slice.rows, 2);
        assert_eq!(slice.cols, 0);
    }
    
    #[test]
    fn enumerated_iter() {
        let m1 = Matrix::new(2, 2, vec![10, 20, 30, 40]).unwrap();
        let expected = vec![((0, 0), 10), ((0, 1), 20), ((1, 0), 30), ((1, 1), 40)];
        let res: Vec<_> = m1.enumerated_iter().collect();
        assert_eq!(res, expected);
        
        let m2 = Matrix::new(0, 2, vec![]).unwrap();
        let res2: Vec<((usize, usize), i32)> = m2.enumerated_iter().collect();
        assert!(res2.is_empty());
    }

    #[test]
    fn enumerated_iter_mut() {
        let mut m1 = Matrix::new(2, 2, vec![10, 20, 30, 40]).unwrap();
        for ((r, c), val) in m1.enumerated_iter_mut() {
            *val += r + c;
        }
        let ans = Matrix::new(2, 2, vec![10, 21, 31, 42]).unwrap();
        assert_eq!(m1, ans);

        let mut m2 = Matrix::<usize>::new(0, 2, vec![]).unwrap();
        let res2 = m2.enumerated_iter_mut().collect::<Vec<_>>();
        assert!(res2.is_empty());
    }
}
