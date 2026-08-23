#[cfg(test)]
mod traits_tests {
    use tensor_math::definitions::matrix::Matrix;
    use tensor_math::definitions::shape::Shape;
    use tensor_math::definitions::tensor::Tensor;
    use tensor_math::definitions::traits::{MatrixLike, MatrixLikeMut, TensorLike, TensorLikeMut};

    fn rank_and_elem_count<T: TensorLike<i32>>(t: &T) -> (usize, usize) {
        (t.rank(), t.elements().len())
    }

    fn shape_rows_cols<T: MatrixLike<i32>>(m: &T) -> (Shape, usize, usize) {
        (m.shape(), m.rows(), m.cols())
    }

    fn bump_first_elem<T: TensorLikeMut<i32>>(t: &mut T) {
        if let Some(x) = t.get_mut(&[0, 0]) {
            *x += 1;
        }
    }

    fn bump_matrix_elem<T: MatrixLikeMut<i32>>(m: &mut T) {
        if let Some(x) = m.get_mut((0, 0)) {
            *x += 1;
        }
    }

    #[test]
    fn tensor_like_basics() {
        let t = Tensor::<i32>::new(&Shape::new(vec![2, 3]), (0..6).collect()).unwrap();
        assert_eq!(rank_and_elem_count(&t), (2, 6));
        assert_eq!(t.shape().clone(), Shape::new(vec![2, 3]));
        assert_eq!(t.get(&[0, 0]), Some(&0));
        assert_eq!(t.get(&[5, 0]), None);
    }

    #[test]
    fn tensor_like_mut_basics() {
        let mut t = Tensor::<i32>::new(&Shape::new(vec![2, 3]), (0..6).collect()).unwrap();
        bump_first_elem(&mut t);
        assert_eq!(t.get(&[0, 0]), Some(&1));
    }

    #[test]
    fn matrix_like_basics() {
        let m = Matrix::<i32>::new(2, 3, (0..6).collect()).unwrap();
        let (shape, rows, cols) = shape_rows_cols(&m);
        assert_eq!(shape, Shape::new(vec![2, 3]));
        assert_eq!(rows, 2);
        assert_eq!(cols, 3);
        assert_eq!(m.get((0, 0)), Some(&0));
        assert_eq!(m.get((5, 5)), None);
    }

    #[test]
    fn matrix_like_mut_basics() {
        let mut m = Matrix::<i32>::new(2, 3, (0..6).collect()).unwrap();
        bump_matrix_elem(&mut m);
        assert_eq!(m.get((0, 0)), Some(&1));
    }

    #[test]
    fn mutable_requires_immutable() {
        // A `TensorLikeMut` is automatically a `TensorLike`; pass it to the immutable fn.
        fn check<T: TensorLikeMut<i32>>(t: &mut T) {
            let _ = rank_and_elem_count(t);
        }
        let mut t = Tensor::<i32>::new(&Shape::new(vec![2, 3]), (0..6).collect()).unwrap();
        check(&mut t);

        // A `MatrixLikeMut` is automatically a `MatrixLike`.
        fn check_m<T: MatrixLikeMut<i32>>(m: &mut T) {
            let _ = shape_rows_cols(m);
        }
        let mut m = Matrix::<i32>::new(2, 3, (0..6).collect()).unwrap();
        check_m(&mut m);
    }
}
