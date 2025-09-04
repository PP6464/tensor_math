#[cfg(test)]
mod tensor_util_tests {
    use crate::tensor::tensor::{IndexProducts, Shape, Tensor, TensorUtilErrors};
    use crate::ts;

    #[test]
    fn shape_products() {
        let shape = ts![2, 3, 4];
        let index_products = IndexProducts::from_shape(&shape);
        assert_eq!(vec![12, 4, 1], index_products.0);
    }

    #[test]
    #[should_panic]
    fn invalid_shape_empty() {
        ts![];
    }

    #[test]
    #[should_panic]
    fn invalid_shape_zero() {
        ts![2, 0, 3];
    }

    #[test]
    #[should_panic]
    fn invalid_shape_size_and_data_length() {
        let shape = ts![2, 3, 4];
        let data = vec![1, 2, 3, 4];
        Tensor::new(&shape, data).unwrap();
    }

    #[test]
    #[should_panic]
    fn invalid_index() {
        let shape = ts![2, 3, 4];
        let tensor = Tensor::<i32>::from_shape(&shape);

        tensor[&[1, 2, 4]];
    }

    #[test]
    fn concat() {
        let t1 = Tensor::<i32>::from_shape(&ts![4, 2, 3]);
        let t2 = Tensor::<i32>::from_value(&ts![4, 1, 3], 1);
        let ans1 = Tensor::new(
            &ts![4, 3, 3],
            vec![
                0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
                0, 0, 0, 0, 0, 1, 1, 1,
            ],
        )
        .unwrap();
        assert_eq!(ans1, t1.concat(&t2, 1).unwrap());

        let t3 = Tensor::<i32>::from_shape(&ts![2, 3]);
        let t4 = Tensor::<i32>::from_value(&ts![1, 3], -1);
        let ans2 = Tensor::new(&ts![3, 3], vec![0, 0, 0, 0, 0, 0, -1, -1, -1]).unwrap();
        assert_eq!(ans2, t3.concat(&t4, 0).unwrap());
    }

    #[test]
    fn invalid_concat() {
        let t1 = Tensor::<i32>::from_shape(&ts![4, 2, 3]);
        let t2 = Tensor::<i32>::from_shape(&ts![3, 1, 2]);

        t1.concat(&t2, 0).expect_err("Should've panicked");
    }

    #[test]
    fn reshape_correctly() {
        let mut t1 = Tensor::<i32>::from_shape(&ts![2, 3, 4]);
        t1.reshape_in_place(&ts![4, 6])
            .expect("Was a valid reshape but failed");

        assert_eq!(*t1.shape(), ts![4, 6]);
    }

    #[test]
    fn invalid_reshape() {
        Tensor::<i32>::from_shape(&ts![2, 3, 4])
            .reshape(&ts![1, 1, 1, 1, 1, 12])
            .expect_err("Should've panicked");
    }

    #[test]
    fn flatten_correctly() {
        let mut t1 = Tensor::<i32>::from_shape(&ts![2, 3, 1, 4, 1])
            .flatten(2)
            .expect("Valid flatten but failed");
        t1.flatten_in_place(3).expect("Valid flatten but failed");
        assert_eq!(*t1.shape(), ts![2, 3, 4]);
    }

    #[test]
    fn invalid_flatten_dim_out_of_bounds() {
        let mut t1 = Tensor::<i32>::from_shape(&ts![2, 3, 4]);
        let error = t1.flatten_in_place(5).err().unwrap();

        match error {
            TensorUtilErrors::DimOutOfBounds { dim: _, max_dim: _ } => {}
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn invalid_flatten_dim_not_one() {
        let mut t1 = Tensor::<i32>::from_shape(&ts![2, 3, 4]);
        let error = t1.flatten_in_place(1).err().unwrap();

        match error {
            TensorUtilErrors::DimIsNotOne(_) => {}
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn tensor_iterator() {
        let shape = ts![2, 3];
        let t1 = Tensor::<i32>::from_shape(&shape);
        let iter1 = t1.clone().into_iter();
        let iter2 = t1.clone().into_iter();

        let mut count = 0;
        for x in iter1 {
            assert_eq!(x, 0);
            count += 1;
        }

        assert_eq!(count, shape.element_count());

        let mut t2: Tensor<i32> = iter2.into();
        assert_eq!(t2.shape(), &ts![shape.element_count()]);
        assert_eq!(t2.elements(), t1.elements());
        t2.reshape_in_place(&shape)
            .expect("Was a valid reshape but failed");
        assert_eq!(t2, t1);

        let shape2 = ts![5, 2];
        let iter3 = vec![0; shape2.element_count()];
        let mut t3: Tensor<i32> = iter3.iter().into();
        assert_eq!(t3, Tensor::from_value(&ts![shape2.element_count()], 0));
        t3.reshape_in_place(&shape2)
            .expect("Was a valid reshape but failed");
        assert_eq!(t3, Tensor::from_value(&shape2, 0));
    }

    #[test]
    fn random_tensor() {
        Tensor::<i32>::rand(&ts![2, 3]);
    }
}
