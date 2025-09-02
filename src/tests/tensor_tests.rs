#[cfg(test)]
mod tests {
    use crate::tensor::tensor::{IndexProducts, Shape, Tensor};

    #[test]
    fn check_shape_products() {
        let shape = Shape::new(vec![2, 3, 4]).unwrap();
        let index_products = IndexProducts::from_shape(&shape);
        assert_eq!(vec![12, 4, 1], index_products.0);
    }

    #[test]
    #[should_panic]
    fn invalid_shape() {
        Shape::new(vec![2, 3, 0]).unwrap();
    }

    #[test]
    #[should_panic]
    fn invalid_shape_size_and_data_length() {
        let shape = Shape::new(vec![2, 3, 4]).unwrap();
        let data = vec![1, 2, 3, 4];
        Tensor::new(shape, data).unwrap();
    }
    
    #[test]
    #[should_panic]
    fn invalid_index() {
        let shape = Shape::new(vec![2, 3, 4]).unwrap();
        let tensor = Tensor::<i32>::from_shape(&shape);
        
        tensor[&[1,2,4]];
    }
}