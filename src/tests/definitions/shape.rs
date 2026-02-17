#[cfg(test)]
mod shape_definition_tests {
    use crate::definitions::errors::TensorErrors;
    use crate::definitions::shape::Shape;
    use crate::definitions::strides::Strides;
    use crate::shape;

    #[test]
    #[should_panic]
    fn invalid_index() {
        shape![1, 3][2];
    }

    #[test]
    fn strides() {
        let shape = shape![2, 3, 4];
        let index_products = Strides::from_shape(&shape);
        assert_eq!(vec![12, 4, 1], index_products.0);
    }

    #[test]
    fn invalid_shape_empty() {
        let err = Shape::new(vec![]).unwrap_err();
        match err {
            TensorErrors::ShapeNoDimensions => {},
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn invalid_shape_zero() {
        let err = Shape::new(vec![2, 0, 3]).unwrap_err();
        match err {
            TensorErrors::ShapeContainsZero => {},
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn display_impl() {
        let shape = shape![2, 3, 4];
        assert_eq!(format!("{}", shape), "[2, 3, 4]");
    }

    #[test]
    fn set_at_axis() {
        let mut shape = shape![1, 1, 1];
        shape.set(2, 2).unwrap();
        assert_eq!(shape, shape![1, 1, 2]);
        let err = shape.set(2, 0).unwrap_err();
        match err {
            TensorErrors::ShapeContainsZero => {},
            _ => panic!("Incorrect error"),
        }
        let err = shape.set(3, 2).unwrap_err();
        match err {
            TensorErrors::AxisOutOfBounds { axis: _, rank : _ } => {},
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn get_is_safe_index() {
        let shape = shape![1, 2, 3];
        assert_eq!(shape.get(1), Some(2));
        assert_eq!(shape.get(5), None);
    }

    #[test]
    fn address_of() {
        let shape = shape![4, 2, 3];
        let addr = shape.address_of(vec![2, 1, 2]).unwrap();
        assert_eq!(addr, 17);
        let err = shape.address_of(vec![1, 2]).unwrap_err();
        match err {
            TensorErrors::IndicesInvalidForRank(_, _) => {},
            _ => panic!("Incorrect error"),
        }
        let err = shape.address_of(vec![4, 0, 0]).unwrap_err();
        match err {
            TensorErrors::IndexOutOfBounds { index: _, axis: _, length: _ } => {},
            _ => panic!("Incorrect error"),
        }
    }
}
