#[cfg(test)]
mod tensor_slice_mut_tests {
    use tensor_math::definitions::shape::Shape;
    use tensor_math::definitions::tensor::Tensor;
    use tensor_math::definitions::traits::{IntoTensor, TryIntoTensor};
    use tensor_math::shape;

    #[test]
    fn slice_rank_and_shape() {
        let shape = shape![4, 5, 6];
        let mut tensor = Tensor::<i32>::from_shape(&shape);
        let slice = tensor.slice_mut(&[1..4, 0..2, 3..6]).unwrap();

        assert_eq!(slice.rank(), 3);
        assert_eq!(slice.shape(), shape![3, 2, 3]);
    }

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
        let ans = Tensor::new(&shape![1, 1, 1], vec![0]).unwrap();

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
        assert_eq!(
            slice.into_tensor(),
            Tensor::new(&shape![], vec![10]).unwrap()
        );
    }

    #[test]
    fn slicing_on_empty_tensor() {
        let mut tensor = Tensor::<i32>::new(&shape![0, 3], vec![]).unwrap();
        let slice = tensor.slice_mut(&[0..0, 1..2]).unwrap();

        assert_eq!(slice.start(), vec![0, 1]);
        assert_eq!(slice.end(), vec![0, 2]);
        assert_eq!(slice.into_tensor().shape(), &shape![0, 1]);

        let mut tensor2 = Tensor::<i32>::new(&shape![3, 0], vec![]).unwrap();
        let slice2 = tensor2.slice_mut(&[1..2, 0..0]).unwrap();

        assert_eq!(slice2.start(), vec![1, 0]);
        assert_eq!(slice2.end(), vec![2, 0]);
        assert_eq!(slice2.into_tensor().shape(), &shape![1, 0]);
    }

    #[test]
    fn slice_shape() {
        let shape = shape![10, 20, 30];
        let mut tensor = Tensor::<i32>::from_shape(&shape);

        // Regular slice
        let slice = tensor.slice_mut(&[2..5, 0..10, 15..30]).unwrap();
        assert_eq!(slice.shape(), shape![3, 10, 15]);

        // Zero-length dimension
        let slice = tensor.slice_mut(&[0..10, 5..5, 0..30]).unwrap();
        assert_eq!(slice.shape(), shape![10, 0, 30]);
    }

    #[test]
    fn for_each_mut_basic() {
        let shape = shape![2, 3, 4];
        let mut tensor = Tensor::<i32>::from_shape(&shape);
        let mut slice = tensor.slice_mut(&[0..2, 1..3, 1..4]).unwrap();
        // slice shape: [2, 2, 3]

        slice.for_each_mut(|x| *x = 1);

        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    let val = *tensor.get(&[i, j, k]).unwrap();
                    if j >= 1 && j < 3 && k >= 1 && k < 4 {
                        assert_eq!(val, 1, "Expected 1 at [{}, {}, {}]", i, j, k);
                    } else {
                        assert_eq!(val, 0, "Expected 0 at [{}, {}, {}]", i, j, k);
                    }
                }
            }
        }
    }

    #[test]
    fn for_each_mut_scalar() {
        let mut tensor = Tensor::<i32>::new(&shape![], vec![42]).unwrap();
        let mut slice = tensor.slice_mut(&[]).unwrap();

        slice.for_each_mut(|x| *x += 1);
        assert_eq!(*tensor.get(&[]).unwrap(), 43);
    }

    #[test]
    fn for_each_mut_empty() {
        let mut tensor = Tensor::<i32>::new(&shape![3, 0], vec![]).unwrap();
        let mut slice = tensor.slice_mut(&[1..2, 0..0]).unwrap();

        let mut count = 0;
        slice.for_each_mut(|_| count += 1);
        assert_eq!(count, 0);
    }
    #[test]
    fn enumerated_for_each_mut_basic() {
        let shape = shape![2, 3, 4];
        let mut tensor = Tensor::<i32>::from_shape(&shape);
        let mut slice = tensor.slice_mut(&[0..2, 1..3, 1..4]).unwrap();
        // slice shape: [2, 2, 3]

        slice.enumerated_for_each_mut(|idx, x| {
            // idx should be relative to the slice: [0..2, 0..2, 0..3]
            *x = (idx[0] * 100 + idx[1] * 10 + idx[2] + 1) as i32;
        });

        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    let val = *tensor.get(&[i, j, k]).unwrap();
                    if j >= 1 && j < 3 && k >= 1 && k < 4 {
                        let expected = (i * 100 + (j - 1) * 10 + (k - 1) + 1) as i32;
                        assert_eq!(
                            val, expected,
                            "Mismatch at tensor index [{}, {}, {}]",
                            i, j, k
                        );
                    } else {
                        assert_eq!(
                            val, 0,
                            "Expected zero at tensor index [{}, {}, {}]",
                            i, j, k
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn enumerated_for_each_mut_scalar() {
        let mut tensor = Tensor::<i32>::new(&shape![], vec![42]).unwrap();
        let mut slice = tensor.slice_mut(&[]).unwrap();

        slice.enumerated_for_each_mut(|idx, x| {
            assert_eq!(idx.len(), 0);
            *x += 1;
        });
        assert_eq!(*tensor.get(&[]).unwrap(), 43);
    }

    #[test]
    fn enumerated_for_each_mut_empty() {
        let mut tensor = Tensor::<i32>::new(&shape![3, 0], vec![]).unwrap();
        let mut slice = tensor.slice_mut(&[1..2, 0..0]).unwrap();

        let mut count = 0;
        slice.enumerated_for_each_mut(|_, _| count += 1);
        assert_eq!(count, 0);
    }
}
