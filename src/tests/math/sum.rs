#[cfg(test)]
mod sum_tests {
    use crate::definitions::errors::TensorErrors;
    use crate::definitions::matrix::Matrix;
    use crate::definitions::shape::Shape;
    use crate::definitions::tensor::Tensor;
    use crate::shape;

    #[test]
    fn test_tensor_sum() {
        let tensor = Tensor::new(&shape![2, 2], vec![1, 2, 3, 4]).unwrap();
        assert_eq!(tensor.sum(), 10);
    }

    #[test]
    fn test_matrix_sum() {
        let matrix = vec![1.0, 2.0, 3.0, 4.0].into_iter().collect::<Matrix<_>>();
        assert_eq!(matrix.sum(), 10.0);
    }

    #[test]
    fn test_matrix_trace() {
        let matrix = Matrix::new(2, 2, vec![1, 2, 3, 4]).unwrap();
        assert_eq!(matrix.trace().unwrap(), 5);
    }

    #[test]
    fn test_matrix_trace_non_square() {
        let matrix = Matrix::new(2, 3, vec![1, 2, 3, 4, 5, 6]).unwrap();
        let result = matrix.trace();
        assert!(matches!(result, Err(TensorErrors::NonSquareMatrix)));
    }
    
    #[test]
    fn zero_by_zero_trace() {
        let matrix = Matrix::<usize>::new(0, 0, vec![]).unwrap();
        assert_eq!(matrix.trace().unwrap(), 0);
    }
    
    #[test]
    fn empty_matrix_tensor_sum() {
        let tensor = Tensor::<i32>::new(&shape![0], vec![]).unwrap();
        assert_eq!(tensor.sum(), 0);

        let matrix = Matrix::<f64>::new(0, 0, vec![]).unwrap();
        assert_eq!(matrix.sum(), 0.0);
    }
}
