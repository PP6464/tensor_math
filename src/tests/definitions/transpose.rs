#[cfg(test)]
mod transpose_tests {
    use crate::definitions::shape::Shape;
    use crate::definitions::transpose::Transpose;
    use crate::{shape, transpose};
    use crate::definitions::errors::TensorErrors;

    #[test]
    fn old_and_new_shapes() {
        let transpose = transpose![1, 0, 2];
        let shape = shape![3, 4, 5];

        let old_shape = transpose.old_shape(&shape).unwrap();
        let ans = transpose.inverse().new_shape(&shape).unwrap();

        assert_eq!(old_shape, ans);

        let shape = shape![2, 3];
        let err = transpose.old_shape(&shape).unwrap_err();
        match err {
            TensorErrors::TransposeIncompatibleRank { .. } => {},
            _ => panic!("Incorrect error"),
        }

        let err = transpose.inverse().new_shape(&shape).unwrap_err();
        match err {
            TensorErrors::TransposeIncompatibleRank { .. } => {},
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn invalid_perm() {
        let err = Transpose::new(&vec![4, 1]).unwrap_err();
        match err {
            TensorErrors::TransposePermutationInvalid => {},
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn indices_do_not_match_rank() {
        let transpose = transpose![1, 0, 2];
        let index = vec![1, 2];

        let err = transpose.new_index(&index).unwrap_err();
        match err {
            TensorErrors::TransposeIncompatibleRank { .. } => {},
            _ => panic!("Incorrect error"),
        }

        let err = transpose.old_index(&index).unwrap_err();
        match err {
            TensorErrors::TransposeIncompatibleRank { .. } => {},
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn default_perm() {
        let transpose = Transpose::default(3);
        let ans = transpose![0, 1, 2];

        assert_eq!(transpose, ans);
    }

    #[test]
    fn swap_axes() {
        let transpose = transpose![1, 0, 2];

        assert_eq!(transpose![0, 1, 2], transpose.swap_axes(0, 1).unwrap());

        let err = transpose.swap_axes(1, 10).unwrap_err();
        match err {
            TensorErrors::TransposePermutationInvalid => {},
            _ => panic!("Incorrect error"),
        }
    }
    
    #[test]
    fn transpose_for_scalar_tensor() {
        let transpose = Transpose::default(0);
        assert_eq!(transpose.permutation().len(), 0);

        let shape = shape![];
        let new_shape = transpose.new_shape(&shape).unwrap();
        assert_eq!(new_shape, shape![]);

        let index = vec![];
        let new_index = transpose.new_index(&index).unwrap();
        assert_eq!(new_index.len(), 0);

        assert_eq!(transpose.inverse(), transpose);
    }
}
