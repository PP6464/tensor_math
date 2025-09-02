#[cfg(test)]
mod tests {
    use crate::tensor::tensor::{IndexProducts, Shape, Tensor, TensorUtilErrors};

    #[test]
    fn shape_products() {
        let shape = Shape::new(vec![2, 3, 4]).unwrap();
        let index_products = IndexProducts::from_shape(shape);
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
        let tensor = Tensor::<i32>::from_shape(shape);
        
        tensor[&[1,2,4]];
    }

    #[test]
    fn concat() {
        let t1 = Tensor::<i32>::from_shape(Shape::new(vec![4, 2, 3]).unwrap());
        let t2 = Tensor::<i32>::from_value(Shape::new(vec![4, 1, 3]).unwrap(), 1);
        let ans1 = Tensor::new(
            Shape::new(vec![4, 3, 3]).unwrap(),
            vec![
                0, 0, 0,
                0, 0, 0,
                1, 1, 1,
                0, 0, 0,
                0, 0, 0,
                1, 1, 1,
                0, 0, 0,
                0, 0, 0,
                1, 1, 1,
                0, 0, 0,
                0, 0, 0,
                1, 1, 1,
            ],
        ).unwrap();
        assert_eq!(ans1, t1.concat(&t2, 1).unwrap());

        let t3 = Tensor::<i32>::from_shape(Shape::new(vec![2, 3]).unwrap());
        let t4 = Tensor::<i32>::from_value(Shape::new(vec![1, 3]).unwrap(), -1);
        let ans2 = Tensor::new(
            Shape::new(vec![3, 3]).unwrap(),
            vec![
                0, 0, 0,
                0, 0, 0,
                -1, -1, -1,
            ],
        ).unwrap();
        assert_eq!(ans2, t3.concat(&t4, 0).unwrap());
    }

    #[test]
    #[should_panic]
    fn invalid_concat() {
        let t1 = Tensor::<i32>::from_shape(Shape::new(vec![4, 2, 3]).unwrap());
        let t2 = Tensor::<i32>::from_shape(Shape::new(vec![3, 1, 2]).unwrap());

        t1.concat(&t2, 0).expect("Should've panicked");
    }

    #[test]
    fn reshape_correctly() {
        let mut t1 = Tensor::<i32>::from_shape(Shape::new(vec![2, 3, 4]).unwrap());
        t1.reshape(Shape::new(vec![4, 6]).unwrap()).expect("Was a valid reshape but failed");

        assert_eq!(*t1.shape(), Shape::new(vec![4, 6]).unwrap());
    }

    #[test]
    #[should_panic]
    fn invalid_reshape() {
        let mut t1 = Tensor::<i32>::from_shape(Shape::new(vec![2, 3, 4]).unwrap());
        t1.reshape(Shape::new(vec![1,1,1,1,1,12]).unwrap()).expect("Should've panicked");
    }
    
    #[test]
    fn flatten_correctly() {
        let mut t1 = Tensor::<i32>::from_shape(Shape::new(vec![2, 3, 1, 4, 1]).unwrap());
        t1.flatten(2).expect("Valid flatten but failed");
        t1.flatten(3).expect("Valid flatten but failed");
        assert_eq!(*t1.shape(), Shape::new(vec![2, 3, 4]).unwrap());
    }

    #[test]
    fn invalid_flatten_dim_out_of_bounds() {
        let mut t1 = Tensor::<i32>::from_shape(Shape::new(vec![2, 3, 4]).unwrap());
        let error = t1.flatten(5).err().unwrap();

        match error {
            TensorUtilErrors::DimOutOfBounds { dim: _, max_dim: _ } => {},
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn invalid_flatten_dim_not_one() {
        let mut t1 = Tensor::<i32>::from_shape(Shape::new(vec![2, 3, 4]).unwrap());
        let error = t1.flatten(1).err().unwrap();

        match error {
            TensorUtilErrors::DimIsNotOne(_) => {},
            _ => panic!("Incorrect error"),
        }
    }
}