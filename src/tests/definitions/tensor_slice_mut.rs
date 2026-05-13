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
    
    #[test]
    fn slicing_on_scalar_tensor() {
        let mut tensor = Tensor::<i32>::new(&shape![], vec![42]).unwrap();
        let mut slice = tensor.slice_mut(&[]).unwrap();

        assert_eq!(slice.get(&[]), Some(&42));
        slice[&[]] = 10;
        assert_eq!(slice.into_tensor(), Tensor::new(&shape![], vec![10]).unwrap());
    }
    
    #[test]
    fn slicing_on_empty_tensor() {
        let mut tensor = Tensor::<i32>::new(&shape![0, 3], vec![]).unwrap();
        let slice = tensor.slice_mut(&[0..0, 1..2]).unwrap();

        assert_eq!(slice.start, vec![0, 1]);
        assert_eq!(slice.end, vec![0, 2]);
        assert_eq!(slice.into_tensor().shape().0, vec![0, 1]);

        let mut tensor2 = Tensor::<i32>::new(&shape![3, 0], vec![]).unwrap();
        let slice2 = tensor2.slice_mut(&[1..2, 0..0]).unwrap();

        assert_eq!(slice2.start, vec![1, 0]);
        assert_eq!(slice2.end, vec![2, 0]);
        assert_eq!(slice2.into_tensor().shape().0, vec![1, 0]);
    }
}
