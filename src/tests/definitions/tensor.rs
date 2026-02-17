#[cfg(test)]
mod tensor_definition_tests {
    use crate::definitions::errors::TensorErrors;
    use crate::definitions::matrix::Matrix;
    use crate::definitions::shape::Shape;
    use crate::definitions::tensor::Tensor;
    use crate::shape;

    #[test]
    fn invalid_shape_size_and_data_length() {
        let shape = shape![2, 3, 4];
        let data = vec![1, 2, 3, 4];
        let err = Tensor::new(&shape, data).unwrap_err();
        assert_eq!(err, TensorErrors::ShapeSizeDoesNotMatch);
    }

    #[test]
    #[should_panic]
    fn invalid_index_out_of_bounds() {
        let shape = shape![2, 3, 4];
        let tensor = Tensor::<i32>::from_shape(&shape);

        tensor[&[1, 2, 4]];
    }

    #[test]
    #[should_panic]
    fn invalid_index_incorrect_rank() {
        let shape = shape![2, 3, 4];
        let tensor = Tensor::<i32>::from_shape(&shape);

        tensor[&[1, 2]];
    }

    #[test]
    #[should_panic]
    fn invalid_mut_index_out_of_bounds() {
        let shape = shape![2, 3, 4];
        let mut tensor = Tensor::<i32>::from_shape(&shape);

        tensor[&[1, 2, 4]] = 1;
    }

    #[test]
    #[should_panic]
    fn invalid_mut_index_incorrect_rank() {
        let shape = shape![2, 3, 4];
        let mut tensor = Tensor::<i32>::from_shape(&shape);

        tensor[&[1, 2]] = 1;
    }

    #[test]
    fn from_matrix() {
        let m1 = Matrix::new(3, 5, (0..15).collect()).unwrap();
        let t1 = Tensor::from(m1);
        let ans = Tensor::new(
            &shape![3, 5],
            (0..15).collect()
        ).unwrap();
        
        assert_eq!(t1, ans);
    }
}
