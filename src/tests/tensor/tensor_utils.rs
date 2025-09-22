#[cfg(test)]
mod tensor_util_tests {
    use std::f64::consts::PI;
    use num::complex::Complex64;
    use num::Float;
    use crate::tensor::tensor::{Strides, Shape, Tensor, TensorErrors, Matrix};
    use crate::shape;
    use crate::tensor::tensor_math::identity;

    #[test]
    fn shape_products() {
        let shape = shape![2, 3, 4];
        let index_products = Strides::from_shape(&shape);
        assert_eq!(vec![12, 4, 1], index_products.0);
    }

    #[test]
    #[should_panic]
    fn invalid_shape_empty() {
        shape![];
    }

    #[test]
    #[should_panic]
    fn invalid_shape_zero() {
        shape![2, 0, 3];
    }

    #[test]
    fn invalid_shape_mat() {
        Matrix::<i32>::new(0, 1, vec![]).expect_err("Should've panicked");
    }

    #[test]
    fn invalid_shape_size_and_data_length() {
        let shape = shape![2, 3, 4];
        let data = vec![1, 2, 3, 4];
        Tensor::new(&shape, data).expect_err("Should've panicked");
    }

    #[test]
    fn invalid_shape_and_elements_mat() {
        Matrix::new(1,1, vec![1, 2]).expect_err("Should've panicked");
    }
    
    #[test]
    fn mat_is_square() {
        let m1 = Matrix::<i32>::from_shape(2, 2);
        let m2 = Matrix::<i32>::from_shape(2, 3);
        
        assert!(m1.is_square());
        assert!(!m2.is_square());
    }

    #[test]
    #[should_panic]
    fn invalid_index() {
        let shape = shape![2, 3, 4];
        let tensor = Tensor::<i32>::from_shape(&shape);

        tensor[&[1, 2, 4]];
    }

    #[test]
    #[should_panic]
    fn mat_invalid_index() {
        let m1 = Matrix::<i32>::from_shape(2, 2);
        m1[(2, 2)];
    }

    #[test]
    fn concat() {
        let t1 = Tensor::<i32>::from_shape(&shape![4, 2, 3]);
        let t2 = Tensor::<i32>::from_value(&shape![4, 1, 3], 1);
        let ans1 = Tensor::new(
            &shape![4, 3, 3],
            vec![
                0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
                0, 0, 0, 0, 0, 1, 1, 1,
            ],
        )
        .unwrap();
        assert_eq!(ans1, t1.concat(&t2, 1).unwrap());

        let t3 = Tensor::<i32>::from_shape(&shape![2, 3]);
        let t4 = Tensor::<i32>::from_value(&shape![1, 3], -1);
        let ans2 = Tensor::new(&shape![3, 3], vec![0, 0, 0, 0, 0, 0, -1, -1, -1]).unwrap();
        assert_eq!(ans2, t3.concat(&t4, 0).unwrap());
    }
    
    #[test]
    fn mat_concat() {
        let m1 = Matrix::new(3,  3, (0..9).collect()).unwrap();
        let m2 = Matrix::new(3,  2, (9..15).collect()).unwrap();
        let res = m1.concat(&m2, 1).unwrap();
        let ans = Matrix::new(3, 5, vec![
            0, 1, 2, 9, 10,
            3, 4, 5, 11, 12,
            6, 7, 8, 13, 14,
        ]).unwrap();
        assert_eq!(ans, res);
    }

    #[test]
    fn invalid_concat() {
        let t1 = Tensor::<i32>::from_shape(&shape![4, 2, 3]);
        let t2 = Tensor::<i32>::from_shape(&shape![3, 1, 2]);

        t1.concat(&t2, 0).expect_err("Should've panicked");
    }
    
    #[test]
    fn invalid_concat_mat() {
        let m1 = Matrix::new(3, 3, (0..9).collect()).unwrap();
        let m2 = Matrix::new(3, 2, (0..6).collect()).unwrap();
        m1.concat(&m2, 0).expect_err("Should've panicked");
    }

    #[test]
    fn reshape_correctly() {
        let mut t1 = Tensor::<i32>::from_shape(&shape![2, 3, 4]);
        t1.reshape_in_place(&shape![4, 6])
            .expect("Was a valid reshape but failed");

        assert_eq!(*t1.shape(), shape![4, 6]);
    }
    
    #[test]
    fn reshape_correctly_mat() {
        let m1 = (0..6).collect::<Matrix<_>>();
        let ans = Matrix::new(2, 3, (0..6).collect()).unwrap();
        assert_eq!(m1.reshape(2, 3).unwrap(), ans);
    }

    #[test]
    fn invalid_reshape() {
        Tensor::<i32>::from_shape(&shape![2, 3, 4])
            .reshape(&shape![1, 1, 1, 1, 1, 12])
            .expect_err("Should've panicked");
    }
    
    #[test]
    fn invalid_reshape_mat() {
        let m1 = Matrix::new(3, 3, (0..9).collect()).unwrap();
        m1.reshape(2, 3).expect_err("Should've panicked");
    }

    #[test]
    fn flatten_correctly() {
        let mut t1 = Tensor::<i32>::from_shape(&shape![2, 3, 1, 4, 1])
            .flatten(2)
            .expect("Valid flatten but failed");
        t1.flatten_in_place(3).expect("Valid flatten but failed");
        assert_eq!(*t1.shape(), shape![2, 3, 4]);
    }

    #[test]
    fn invalid_flatten_dim_out_of_bounds() {
        let mut t1 = Tensor::<i32>::from_shape(&shape![2, 3, 4]);
        let error = t1.flatten_in_place(5).err().unwrap();

        match error {
            TensorErrors::DimOutOfBounds { dim: _, max_dim: _ } => {}
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn invalid_flatten_dim_not_one() {
        let mut t1 = Tensor::<i32>::from_shape(&shape![2, 3, 4]);
        let error = t1.flatten_in_place(1).err().unwrap();

        match error {
            TensorErrors::DimIsNotOne(_) => {}
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn tensor_iterator() {
        let shape = shape![2, 3];
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
        assert_eq!(t2.shape(), &shape![shape.element_count()]);
        assert_eq!(t2.elements(), t1.elements());
        t2.reshape_in_place(&shape)
            .expect("Was a valid reshape but failed");
        assert_eq!(t2, t1);

        let shape2 = shape![5, 2];
        let iter3 = vec![0; shape2.element_count()];
        let mut t3: Tensor<i32> = iter3.iter().into();
        assert_eq!(t3, Tensor::from_value(&shape![shape2.element_count()], 0));
        t3.reshape_in_place(&shape2)
            .expect("Was a valid reshape but failed");
        assert_eq!(t3, Tensor::from_value(&shape2, 0));
    }

    #[test]
    fn random_tensor() {
        Tensor::<i32>::rand(&shape![2, 3]);
    }
    
    #[test]
    fn random_mat() {
        Matrix::<i32>::rand(2, 3);
    }

    #[test]
    fn transform_elementwise() {
        let t1 = Tensor::<f64>::new(
            &shape![2, 3],
            vec![
                0.0, PI / 6.0, PI / 3.0,
                PI / 2.0, 2.0 * PI / 3.0, 5.0 * PI / 6.0
            ],
        ).unwrap();
        let transformed = t1.transform_elementwise(f64::cos);
        let ans = Tensor::<f64>::new(
            &shape![2, 3],
            vec![
                1.0, f64::sqrt(3.0) / 2.0, 0.5,
                0.0, -0.5, -f64::sqrt(3.0) / 2.0,
            ],
        ).unwrap();
        assert!((ans - transformed).into_iter().map(f64::abs).sum::<f64>() < 1e-6);
    }
    
    #[test]
    fn transform_elementwise_mat() {
        let m1 = Matrix::<f64>::new(2, 2, (0..4).map(f64::from).collect()).unwrap();
        let m1 = m1.transform_elementwise(f64::sqrt);
        let ans = Matrix::<f64>::new(2, 2, vec![0.0, 1.0, 2.0.sqrt(), 3.0.sqrt()]).unwrap();
        assert!((ans - m1).into_iter().map(f64::abs).sum::<f64>() < 1e-15);
    }

    #[test]
    fn slicing() {
        let t1 = Tensor::<i32>::new(
            &shape![3, 3, 3],
            (0..27).collect(),
        ).unwrap();

        let sliced = t1.slice(&[
            0..3,
            1..2,
            1..3,
        ]);
        let ans = Tensor::<i32>::new(
            &shape![3, 1, 2],
            vec![
                4, 5,
                13, 14,
                22, 23
            ],
        ).unwrap();

        assert_eq!(sliced, ans);
    }

    #[test]
    fn slicing_mat() {
        let m1 = Matrix::<f64>::new(4, 4, (0..16).map(f64::from).collect()).unwrap();
        let slice = m1.slice(1..3, 1..3);
        let ans = Matrix::<f64>::new(2, 2, vec![5.0, 6.0, 9.0, 10.0]).unwrap();
        assert_eq!(slice, ans);
    }

    #[test]
    #[should_panic]
    fn invalid_slice_out_of_bounds() {
        let t1 = Tensor::<i32>::from_shape(&shape![2, 3]);
        t1.slice(&[
            1..5,
            0..3,
        ]);
    }

    #[test]
    #[should_panic]
    fn invalid_slice_out_of_bounds_matrix() {
        let m1 = Matrix::<f64>::new(4, 4, (0..16).map(f64::from).collect()).unwrap();
        m1.slice(1..5, 1..2);
    }

    #[test]
    #[should_panic]
    fn invalid_slice_incorrect_rank() {
        let t1 = Tensor::<i32>::from_shape(&shape![2, 3]);
        t1.slice(&[]);
    }

    #[test]
    fn slicing_mut() {
        let mut t1 = Tensor::<i32>::new(
            &shape![3, 3, 3],
            (0..27).collect(),
        ).unwrap();
        let mut slice = t1.slice_mut(&[1..2, 1..2, 1..2]);
        slice[&[0, 0, 0]] = 69;
        let ans = Tensor::<i32>::new(
            &shape![3, 3, 3],
            (0..27).map(|x| if x != 13 { x } else { 69 }).collect(),
        ).unwrap();
        assert_eq!(t1, ans);
    }

    #[test]
    fn slicing_mut_mat() {
        let mut m1 = Matrix::<f64>::new(4, 4, (0..16).map(f64::from).collect()).unwrap();
        let mut slice = m1.slice_mut(1..3, 1..3);
        slice[(0, 0)] = 69.0;
        slice[(0, 1)] = 42.0;
        slice[&[1, 0]] = -20.0;
        slice[&[1, 1]] = 91.0;
        let ans = Matrix::<f64>::new(4, 4, vec![0.0, 1.0, 2.0, 3.0, 4.0, 69.0, 42.0, 7.0, 8.0, -20.0, 91.0, 11.0, 12.0, 13.0, 14.0, 15.0]).unwrap();
        assert_eq!(m1, ans);
    }

    #[test]
    #[should_panic]
    fn slice_mut_out_of_bounds() {
        let mut t1 = Tensor::<i32>::from_shape(&shape![2, 3]);
        let mut slice  = t1.slice_mut(&[0..1, 0..1]);

        slice[&[0, 1]] = 69;
    }

    #[test]
    #[should_panic]
    fn slice_mut_out_of_bounds_mat() {
        let mut m1 = Matrix::<f64>::new(4, 4, (0..16).map(f64::from).collect()).unwrap();
        let mut slice = m1.slice_mut(1..3, 1..3);
        slice[&[3, 1]] = 69.0;
    }

    #[test]
    fn concat_multithreaded() {
        let t1 = Tensor::<i32>::from_shape(&shape![20, 30]);
        let t2 = Tensor::<i32>::from_value(&shape![20, 20], 2);

        let ans = t1.concat(&t2, 1).unwrap();
        let mt_ans = t1.concat_mt(&t2, 1).unwrap();

        assert_eq!(ans, mt_ans);
    }

    #[test]
    fn concat_multithreaded_mat() {
        let m1 = identity::<Complex64>(10).reshape(20, 5).unwrap();
        let m2 = (0..500).collect::<Matrix<_>>().reshape(100, 5).unwrap().transform_elementwise(f64::from).transform_elementwise(Complex64::from);

        let ans  = m1.concat(&m2, 0).unwrap();
        let mt_ans = m1.concat_mt(&m2, 0).unwrap();
        
        assert_eq!(ans, mt_ans);
    }
}
