#[cfg(test)]
mod tensor_util_tests {
    use std::f64::consts::PI;
    use std::ops::Add;
    use float_cmp::assert_approx_eq;
    use num::complex::{Complex64, ComplexFloat};
    use crate::tensor::tensor::{Strides, Shape, Tensor, TensorErrors};
    use crate::tensor::tensor_math::{solve_cubic, solve_quadratic, solve_quartic};
    use crate::ts;

    #[test]
    fn shape_products() {
        let shape = ts![2, 3, 4];
        let index_products = Strides::from_shape(&shape);
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
            TensorErrors::DimOutOfBounds { dim: _, max_dim: _ } => {}
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn invalid_flatten_dim_not_one() {
        let mut t1 = Tensor::<i32>::from_shape(&ts![2, 3, 4]);
        let error = t1.flatten_in_place(1).err().unwrap();

        match error {
            TensorErrors::DimIsNotOne(_) => {}
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

    #[test]
    fn transform_elementwise() {
        let t1 = Tensor::<f64>::new(
            &ts![2, 3],
            vec![
                0.0, PI / 6.0, PI / 3.0,
                PI / 2.0, 2.0 * PI / 3.0, 5.0 * PI / 6.0
            ],
        ).unwrap();
        let transformed = t1.transform_elementwise(f64::cos);
        let ans = Tensor::<f64>::new(
            &ts![2, 3],
            vec![
                1.0, f64::sqrt(3.0) / 2.0, 0.5,
                0.0, -0.5, -f64::sqrt(3.0) / 2.0,
            ],
        ).unwrap();
        assert!((ans - transformed).into_iter().map(f64::abs).sum::<f64>() < 1e-6);
    }

    #[test]
    fn slicing() {
        let t1 = Tensor::<i32>::new(
            &ts![3, 3, 3],
            (0..27).collect(),
        ).unwrap();

        let sliced = t1.slice(&[
            0..3,
            1..2,
            1..3,
        ]);
        let ans = Tensor::<i32>::new(
            &ts![3, 1, 2],
            vec![
                4, 5,
                13, 14,
                22, 23
            ],
        ).unwrap();

        assert_eq!(sliced, ans);
    }

    #[test]
    #[should_panic]
    fn invalid_slice_out_of_bounds() {
        let t1 = Tensor::<i32>::from_shape(&ts![2, 3]);
        t1.slice(&[
            1..5,
            0..3,
        ]);
    }

    #[test]
    #[should_panic]
    fn invalid_slice_incorrect_rank() {
        let t1 = Tensor::<i32>::from_shape(&ts![2, 3]);
        t1.slice(&[]);
    }

    #[test]
    fn slicing_mut() {
        let mut t1 = Tensor::<i32>::new(
            &ts![3, 3, 3],
            (0..27).collect(),
        ).unwrap();
        let mut slice = t1.slice_mut(&[1..2, 1..2, 1..2]);
        slice[&[0, 0, 0]] = 69;
        let ans = Tensor::<i32>::new(
            &ts![3, 3, 3],
            (0..27).map(|x| if x != 13 { x } else { 69 }).collect(),
        ).unwrap();
        assert_eq!(t1, ans);
    }

    #[test]
    #[should_panic]
    fn slice_mut_out_of_bounds() {
        let mut t1 = Tensor::<i32>::from_shape(&ts![2, 3]);
        let mut slice  = t1.slice_mut(&[0..1, 0..1]);

        slice[&[0, 1]] = 69;
    }

    #[test]
    fn solve_quadratic_poly() {
        let coefficients = [
            Complex64 { re: 1.0, im: 2.0 },
            Complex64 { re: 0.0, im: -2.0 },
            Complex64 { re: 5.0, im: 0.0 },
        ];
        let roots = solve_quadratic(&coefficients);

        assert_eq!(roots.len(), 2);
        assert_approx_eq!(f64, coefficients.iter().enumerate().map(|(i, c)| c * roots[0].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 1e-15);
        assert_approx_eq!(f64, coefficients.iter().enumerate().map(|(i, c)| c * roots[1].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn solve_cubic_poly() {
        let coefficients = [
            Complex64 { re: 4.0, im: 0.0 },
            Complex64 { re: 3.0, im: 0.0 },
            Complex64 { re: 2.0, im: 0.0 },
            Complex64 { re: 1.0, im: 0.0 },
        ];
        let roots = solve_cubic(&coefficients);

        assert_eq!(roots.len(), 3);
        assert_approx_eq!(f64, coefficients.iter().enumerate().map(|(i, c)| c * roots[0].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 2e-15);
        assert_approx_eq!(f64, coefficients.iter().enumerate().map(|(i, c)| c * roots[1].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 2e-15);
        assert_approx_eq!(f64, coefficients.iter().enumerate().map(|(i, c)| c * roots[2].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 2e-15);

        let coefficients2 = [
            Complex64 { re: 45.0, im: -2.0 },
            Complex64 { re: -3.0, im: 4.0 },
            Complex64 { re: 2.0, im: 3.0 },
            Complex64 { re: 1.0, im: 10.0 },
        ];
        let roots2 = solve_cubic(&coefficients2);

        assert_eq!(roots2.len(), 3);
        assert_approx_eq!(f64, coefficients2.iter().enumerate().map(|(i, c)| c * roots2[0].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 2e-11);
        assert_approx_eq!(f64, coefficients2.iter().enumerate().map(|(i, c)| c * roots2[1].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 2e-11);
        assert_approx_eq!(f64, coefficients2.iter().enumerate().map(|(i, c)| c * roots2[2].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 2e-11);
    }

    #[test]
    fn solve_quartic_poly() {
        let coefficients = [
            Complex64 { re: 4.5, im: -2.0 },
            Complex64 { re: 3.4, im: 5.0 },
            Complex64 { re: 2.1, im: 3.0 },
            Complex64 { re: 1.2, im: 2.0 },
            Complex64 { re: 1.1, im: -10.0 },
        ];

        let roots = solve_quartic(&coefficients);
        assert_eq!(roots.len(), 4);

        assert_approx_eq!(f64, coefficients.iter().enumerate().map(|(i, c)| c * roots[0].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 2e-10);
        assert_approx_eq!(f64, coefficients.iter().enumerate().map(|(i, c)| c * roots[1].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 2e-10);
        assert_approx_eq!(f64, coefficients.iter().enumerate().map(|(i, c)| c * roots[2].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 2e-10);
        assert_approx_eq!(f64, coefficients.iter().enumerate().map(|(i, c)| c * roots[3].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 2e-10);
    }
}
