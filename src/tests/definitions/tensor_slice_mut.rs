#[cfg(test)]
mod tensor_slice_mut_tests {
    use crate::definitions::shape::Shape;
    use crate::definitions::tensor::Tensor;
    use crate::definitions::traits::{IntoTensor, TryIntoTensor};
    use crate::shape;

    #[test]
    #[should_panic]
    fn invalid_index() {
        let shape = shape![2, 3, 4];
        let mut tensor = Tensor::<i32>::from_shape(&shape);
        let slice = tensor.slice_mut(&[0..1, 0..1, 0..1]).unwrap();
    
        slice[&[1, 0, 0]];
    }
    
    #[test]
    fn index_mut() {
        let shape = shape![2, 3, 4];
        let mut tensor = Tensor::<i32>::from_shape(&shape);
        let mut slice = tensor.slice_mut(&[0..1, 0..1, 0..1]).unwrap();

        assert_eq!(slice[&[0, 0, 0]], 0);
        
        slice[&[0, 0, 0]] = 67;

        assert_eq!(slice[&[0, 0, 0]], 67);
    }
    
    #[test]
    fn get_is_safe_index() {
        let shape = shape![2, 3, 4];
        let mut tensor = Tensor::<i32>::from_shape(&shape);
        let slice = tensor.slice_mut(&[0..1, 0..1, 0..1]).unwrap();

        assert_eq!(slice.get(&[0, 0]), None);
        assert_eq!(slice.get(&[0, 0, 0]), Some(&0));
    }
    
    #[test]
    fn convert_into_tensor() {
        let shape = shape![2, 3, 4];
        let mut tensor = Tensor::<i32>::from_shape(&shape);
        let slice = tensor.slice_mut(&[0..1, 0..1, 0..1]).unwrap();
        let ans = Tensor::new(
            &shape![1, 1, 1],
            vec![0]
        ).unwrap();
        
        assert_eq!(slice.into_tensor(), ans);

        let slice = tensor.slice_mut(&[0..1, 0..1, 0..1]).unwrap();
        
        assert_eq!(slice.try_into_tensor().unwrap(), ans);
    }
}
