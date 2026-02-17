#[cfg(test)]
mod matrix_definition_tests {
    use crate::definitions::shape::Shape;
use crate::definitions::errors::TensorErrors;
    use crate::definitions::matrix::Matrix;
    use crate::definitions::tensor::Tensor;
    use crate::definitions::traits::IntoTensor;
    use crate::shape;

    #[test]
    fn invalid_shape() {
        let err = Matrix::<i32>::new(0, 1, vec![]).unwrap_err();
        assert_eq!(err, TensorErrors::ShapeContainsZero);
    }

    #[test]
    fn invalid_shape_and_elements() {
        let err = Matrix::new(1,1, vec![1, 2]).unwrap_err();
        assert_eq!(err, TensorErrors::ShapeSizeDoesNotMatch);
    }

    #[test]
    fn is_square() {
        let m1 = Matrix::<i32>::from_shape(2, 2).unwrap();
        let m2 = Matrix::<i32>::from_shape(2, 3).unwrap();

        assert!(m1.is_square());
        assert!(!m2.is_square());
    }

    #[test]
    #[should_panic]
    fn invalid_index() {
        let m1 = Matrix::<i32>::from_shape(2, 2).unwrap();
        m1[(2, 2)];
    }

    #[test]
    fn get_is_safe_indexing() {
        let m1 = Matrix::<i32>::new(
            2, 2,
            vec![0, 1, 2, 3]
        ).unwrap();

        assert_eq!(m1.get((1, 1)), Some(&3));
        assert_eq!(m1.get((2, 2)), None);
    }

    #[test]
    fn rows_cols_gives_shape() {
        let m1 = Matrix::<i32>::new(
            2, 3,
            vec![0, 1, 2, 3, 4, 5],
        ).unwrap();

        assert_eq!(m1.rows(), 2);
        assert_eq!(m1.cols(), 3);
    }

    #[test]
    fn convert_into_tensor() {
        let m1 = Matrix::<i32>::new(
            2, 3,
            vec![0, 1, 2, 3, 4, 5],
        ).unwrap();
        let t1 = Tensor::<i32>::new(
            &shape![2, 3],
            vec![0, 1, 2, 3, 4, 5],
        ).unwrap();

        assert_eq!(m1.into_tensor(), t1);
    }

    #[test]
    fn default_matrix() {
        let m1 = Matrix::<i32>::default();
        assert_eq!(m1, Matrix::new(1, 1, vec![0]).unwrap());
    }

    #[test]
    fn from_into_iter() {
        let v1 = vec![1, 2, 3, 4, 5, 6];
        let m1 = Matrix::from(v1.into_iter());
        let ans = Matrix::new(
            1, 6,
            vec![1, 2, 3, 4, 5, 6],
        ).unwrap();

        assert_eq!(m1, ans);
    }

    #[test]
    fn from_iter() {
        let v1 = vec![1, 2, 3, 4, 5, 6];
        let m1 = Matrix::from(v1.iter());

        let ans = Matrix::new(
            1, 6,
            v1,
        ).unwrap();

        assert_eq!(m1, ans);
    }

    #[test]
    fn invalid_try_from_tensor() {
        let t1 = vec![1, 2, 3, 4, 5, 6].into_tensor();
        let err = Matrix::try_from(t1).unwrap_err();
        match err {
            TensorErrors::ShapesIncompatible => {},
            _ => panic!("Incorrect error"),
        }
    }
}