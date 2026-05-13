#[cfg(test)]
mod shape_definition_tests {
    use crate::definitions::errors::TensorErrors;
    use crate::definitions::shape::Shape;
    use crate::definitions::strides::Strides;
    use crate::shape;

    #[test]
    fn can_create_empty_shape() {
        let shape = shape![];
        assert_eq!(shape.rank(), 0);
    }

    #[test]
    #[should_panic]
    fn invalid_index_empty_shape() {
        let shape = shape![];
        shape[0];
    }

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
    fn display_impl() {
        let shape = shape![2, 3, 4];
        assert_eq!(format!("{}", shape), "[2, 3, 4]");
    }

    #[test]
    fn set_at_axis() {
        let mut shape = shape![1, 1, 1];
        shape.set(2, 2).unwrap();
        assert_eq!(shape, shape![1, 1, 2]);
        let err = shape.set(3, 2).unwrap_err();
        match err {
            TensorErrors::AxisOutOfBounds { .. } => {}
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
    fn address() {
        let shape = shape![];
        let addr = shape.address(vec![]).unwrap();
        assert_eq!(addr, 0);
        let shape = shape![4, 2, 3];
        let addr = shape.address(vec![2, 1, 2]).unwrap();
        assert_eq!(addr, 17);
        let err = shape.address(vec![1, 2]).unwrap_err();
        match err {
            TensorErrors::IndicesInvalidForRank(_, _) => {}
            _ => panic!("Incorrect error"),
        }
        let err = shape.address(vec![4, 0, 0]).unwrap_err();
        match err {
            TensorErrors::IndexOutOfBounds {
                index: _,
                axis: _,
                length: _,
            } => {}
            _ => panic!("Incorrect error"),
        }
    }
    
    #[test]
    fn tensor_index() {
        let shape = shape![];
        let index = shape.tensor_index(0).unwrap();
        assert_eq!(index, vec![]);
        
        let shape = shape![0];
        let err = shape.tensor_index(0).unwrap_err();
        match err {
            TensorErrors::AddressOutOfBounds(_) => {}
            _ => panic!("Incorrect error"),
        }
        
        let shape = shape![4, 2, 3];
        let index = shape.tensor_index(17).unwrap();
        assert_eq!(index, vec![2, 1, 2]);

        let err = shape.tensor_index(24).unwrap_err();
        match err {
            TensorErrors::AddressOutOfBounds(_) => {}
            _ => panic!("Incorrect error"),
        }
    }
}
