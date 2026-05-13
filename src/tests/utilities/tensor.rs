#[cfg(test)]
mod tensor_utils_tests {
    use crate::definitions::errors::TensorErrors;
    use crate::definitions::shape::Shape;
    use crate::definitions::tensor::Tensor;
    use crate::definitions::transpose::Transpose;
    use crate::utilities::tensor::{pool_avg, pool_max, pool_min, pool_sum};
    use crate::{shape, transpose};
    use std::collections::HashSet;
    use std::f64::consts::PI;

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
    fn invalid_concat() {
        let t1 = Tensor::<i32>::from_shape(&shape![4, 2, 3]);
        let t2 = Tensor::<i32>::from_shape(&shape![3, 1, 2]);

        assert_eq!(t1.concat(&t2, 0).unwrap_err(), TensorErrors::ShapesIncompatible);
    }

    #[test]
    fn reshape_correctly() {
        let t1 = Tensor::<i32>::from_shape(&shape![2, 3, 4]);
        let t1 = t1.reshape(&shape![4, 6])
            .expect("Was a valid reshape but failed");

        assert_eq!(*t1.shape(), shape![4, 6]);
    }

    #[test]
    fn invalid_reshape() {
        Tensor::<i32>::from_shape(&shape![2, 3, 4])
            .reshape(&shape![1, 1, 1, 1, 1, 12])
            .expect_err("Should've panicked");
    }

    #[test]
    fn flatten_correctly() {
        let t1 = Tensor::<i32>::from_shape(&shape![2, 3, 1, 4, 1])
            .flatten(2)
            .expect("Valid flatten but failed");
        let t1 = t1.flatten(3).expect("Valid flatten but failed");
        assert_eq!(*t1.shape(), shape![2, 3, 4]);
    }
    
    #[test]
    fn invalid_flatten_dim_out_of_bounds() {
        let t1 = Tensor::<i32>::from_shape(&shape![2, 3, 4]);
        let error = t1.flatten(5).err().unwrap();

        match error {
            TensorErrors::AxisOutOfBounds {
                axis: _,
                rank: _,
            } => {}
            _ => panic!("Incorrect error"),
        }
    }
    
    #[test]
    fn invalid_flatten_dim_not_one() {
        let t1 = Tensor::<i32>::from_shape(&shape![2, 3, 4]);
        let error = t1.flatten(1).err().unwrap();

        match error {
            TensorErrors::AxisIsNotOne(_) => {}
            _ => panic!("Incorrect error"),
        }
    }
    
    #[test]
    fn iterator() {
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

        let t2: Tensor<i32> = iter2.collect();
        assert_eq!(t2.shape(), &shape![shape.element_count()]);
        assert_eq!(t2.elements(), t1.elements());
        let t2 = t2.reshape(&shape)
            .expect("Was a valid reshape but failed");
        assert_eq!(t2, t1);

        let shape2 = shape![5, 2];
        let iter3 = vec![0; shape2.element_count()];
        let t3: Tensor<i32> = iter3.iter().into();
        assert_eq!(t3, Tensor::from_value(&shape![shape2.element_count()], 0));
        let t3 = t3.reshape(&shape2)
            .expect("Was a valid reshape but failed");
        assert_eq!(t3, Tensor::from_value(&shape2, 0));
    }
    
    #[test]
    fn random() {
        Tensor::<i32>::rand(&shape![2, 3]);
    }
    
    #[test]
    fn transform_elementwise() {
        let t1 = Tensor::<f64>::new(
            &shape![2, 3],
            vec![
                0.0,
                PI / 6.0,
                PI / 3.0,
                PI / 2.0,
                2.0 * PI / 3.0,
                5.0 * PI / 6.0,
            ],
        )
        .unwrap();
        let transformed = t1.map(f64::cos);
        let ans = Tensor::<f64>::new(
            &shape![2, 3],
            vec![
                1.0,
                f64::sqrt(3.0) / 2.0,
                0.5,
                0.0,
                -0.5,
                -f64::sqrt(3.0) / 2.0,
            ],
        )
        .unwrap();
        assert!((ans - transformed).into_iter().map(f64::abs).sum::<f64>() < 1e-6);
    }
    
    #[test]
    fn slicing() {
        let t1 = Tensor::<i32>::new(&shape![3, 3, 3], (0..27).collect()).unwrap();

        let sliced = t1.slice(&[0..3, 1..2, 1..3]).unwrap();
        let ans = Tensor::<i32>::new(&shape![3, 1, 2], vec![4, 5, 13, 14, 22, 23]).unwrap();

        assert_eq!(sliced, ans);
    }
    
    #[test]
    #[should_panic]
    fn invalid_slice_out_of_bounds() {
        let t1 = Tensor::<i32>::from_shape(&shape![2, 3]);
        t1.slice(&[1..5, 0..3]).unwrap();
    }
    
    #[test]
    #[should_panic]
    fn invalid_slice_incorrect_rank() {
        let t1 = Tensor::<i32>::from_shape(&shape![2, 3]);
        t1.slice(&[]).unwrap();
    }
    
    #[test]
    fn slicing_mut() {
        let mut t1 = Tensor::<i32>::new(&shape![3, 3, 3], (0..27).collect()).unwrap();
        let mut slice = t1.slice_mut(&[1..2, 1..2, 1..2]).unwrap();
        slice[&[0, 0, 0]] = 69;
        let ans = Tensor::<i32>::new(
            &shape![3, 3, 3],
            (0..27).map(|x| if x != 13 { x } else { 69 }).collect(),
        )
        .unwrap();
        assert_eq!(t1, ans);
    }
    
    #[test]
    fn set_all() {
        let mut t1 = Tensor::<i32>::new(&shape![3, 3, 3], (0..27).collect()).unwrap();
        let mut slice_mut = t1.slice_mut(&[0..2, 1..2, 0..2]).unwrap();
        let inserted = Tensor::<i32>::new(&shape![2, 1, 2], (0..4).collect()).unwrap();

        slice_mut.set_all(&inserted).unwrap();
        let ans = Tensor::<i32>::new(
            &shape![3, 3, 3],
            vec![
                0, 1, 2, 0, 1, 5, 6, 7, 8, 9, 10, 11, 2, 3, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26,
            ],
        )
        .unwrap();

        assert_eq!(t1, ans);
    }
    
    #[test]
    #[should_panic]
    fn set_all_fail() {
        let mut t1 = Tensor::<i32>::new(&shape![3, 3, 3], (0..27).collect()).unwrap();
        let mut slice_mut = t1.slice_mut(&[0..2, 0..2, 0..2]).unwrap();

        slice_mut.set_all(&Tensor::<i32>::zeros(&shape![1])).unwrap();
    }
    
    #[test]
    #[should_panic]
    fn slice_mut_out_of_bounds() {
        let mut t1 = Tensor::<i32>::from_shape(&shape![2, 3]);
        let mut slice = t1.slice_mut(&[0..1, 0..1]).unwrap();

        slice[&[0, 1]] = 69;
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
    fn transpose() {
        let t1 = Tensor::<i32>::new(&shape![2, 3, 4], (0..24).collect()).unwrap();
        let t2 = Tensor::<i32>::new(&shape![2, 3], (0..6).collect()).unwrap();
        let transposed_t1 = t1
            .clone()
            .transpose(&transpose![0, 2, 1])
            .unwrap();
        let t2 = t2.transpose(&Transpose::new(&vec![1, 0]).unwrap())
            .unwrap();
        let transposed_t2 = t2.clone();
        let ans1 = Tensor::<i32>::new(
            &shape![2, 4, 3],
            vec![
                0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19,
                23,
            ],
        )
            .unwrap();
        let ans2 = Tensor::<i32>::new(&shape![3, 2], vec![0, 3, 1, 4, 2, 5]).unwrap();

        assert_eq!(ans1, transposed_t1);
        assert_eq!(ans2, transposed_t2);
    }

    #[test]
    fn transpose_mt() {
        let t1 = Tensor::<i32>::rand(&shape![10, 60, 5]);
        let transpose = transpose![1, 2, 0];

        let ans = t1.transpose(&transpose).unwrap();
        let mt_ans = t1.transpose_mt(&transpose).unwrap();

        assert_eq!(mt_ans, ans);
    }

    #[test]
    fn clipping() {
        let t1 = Tensor::<i32>::new(
            &shape![3, 3, 3],
            (0..27).collect(),
        ).unwrap();
        let t1 = t1.clip(5, 10);
        let ans = Tensor::<i32>::new(
            &shape![3, 3, 3],
            vec![5, 5, 5, 5, 5, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        ).unwrap();

        assert_eq!(t1, ans);
    }

    #[test]
    fn pool() {
        let t1: Tensor<f64> = Tensor::<i32>::new(
            &shape![3, 3, 3],
            vec![
                1, 5, -1, 2, 3, -5, 12, 10, -10, 1, -4, 2, 9, 6, 8, -1, 0, -1, -8, 7, 4, 5, 1, 2,
                -5, 3, 1,
            ],
        )
            .unwrap()
            .map(|x| x.into());

        let avg_pool = t1.pool(pool_avg, &shape![2, 2, 2], &shape![2, 2, 2], 0.0).unwrap();
        let sum_pool = t1.pool(pool_sum, &shape![2, 2, 2], &shape![2, 2, 2], 0.0).unwrap();
        let max_pool = t1.pool(pool_max, &shape![2, 2, 2], &shape![1, 1, 1], 0.0).unwrap();
        let min_pool = t1.pool(pool_min, &shape![3, 1, 1], &shape![3, 1, 1], 0.0).unwrap();

        let sum_ans = Tensor::<f64>::new(
            &shape![2, 2, 2],
            vec![23.0, 4.0, 21.0, -11.0, 5.0, 6.0, -2.0, 1.0],
        )
            .unwrap();
        let avg_ans = Tensor::<f64>::new(
            &shape![2, 2, 2],
            vec![2.875, 1.0, 5.25, -5.5, 1.25, 3.0, -1.0, 1.0],
        )
            .unwrap();
        let max_ans = Tensor::<f64>::new(
            &shape![3, 3, 3],
            vec![
                9.0, 8.0, 8.0, 12.0, 10.0, 8.0, 12.0, 10.0, -1.0, 9.0, 8.0, 8.0, 9.0, 8.0, 8.0,
                3.0, 3.0, 1.0, 7.0, 7.0, 4.0, 5.0, 3.0, 2.0, 3.0, 3.0, 1.0,
            ],
        )
            .unwrap();
        let min_ans = Tensor::<f64>::new(
            &shape![1, 3, 3],
            vec![-8.0, -4.0, -1.0, 2.0, 1.0, -5.0, -5.0, 0.0, -10.0],
        )
            .unwrap();

        assert_eq!(avg_pool, avg_ans);
        assert_eq!(sum_pool, sum_ans);
        assert_eq!(max_pool, max_ans);
        assert_eq!(min_pool, min_ans);
    }

    #[test]
    fn pool_mt() {
        let t1 = Tensor::<i32>::rand(&shape![200, 10, 10]);

        let ans = t1.pool(pool_min, &shape![10, 30, 1], &shape![10, 10, 10], 0).unwrap();
        let mt_ans = t1.pool_mt(&pool_min, &shape![10, 30, 1], &shape![10, 10, 10], 0).unwrap();

        assert_eq!(ans, mt_ans);
    }

    #[test]
    fn invalid_flip_axes_mt() {
        let t1 = Tensor::<i32>::new(
            &shape![1, 2, 3, 4],
            vec![1; 24],
        ).unwrap();

        let mut axes = HashSet::new();
        axes.insert(4);

        let err = t1.flip_axes_mt(&axes).unwrap_err();
        match err { 
            TensorErrors::AxisOutOfBounds { axis: _, rank: _ } => {},
            _ => panic!("Incorrect error"),
        }
    }
    
    #[test]
    fn flip() {
        let t1 = Tensor::<i32>::new(
            &shape![2, 3, 4],
            (0..24).collect(),
        ).unwrap();
        let mut axes = HashSet::new();
        axes.insert(0);
        axes.insert(2);

        let res_axes = t1.flip_axes(&axes).unwrap();
        let res = t1.flip();

        let ans_axes = Tensor::<i32>::new(
            &shape![2, 3, 4],
            vec![
                15, 14, 13, 12,
                19, 18, 17, 16,
                23, 22, 21, 20,

                3, 2, 1, 0,
                7, 6, 5, 4,
                11, 10, 9, 8,
            ],
        ).unwrap();
        let ans = Tensor::<i32>::new(
            &shape![2, 3, 4],
            vec![
                23, 22, 21, 20,
                19, 18, 17, 16,
                15, 14, 13, 12,

                11, 10, 9, 8,
                7, 6, 5, 4,
                3, 2, 1, 0,
            ],
        ).unwrap();

        assert_eq!(res_axes, ans_axes);
        assert_eq!(res, ans);
    }
    
    #[test]
    fn flip_mt() {
        let t1 = Tensor::<i32>::rand(&shape![10, 20, 10]);
        let mut axes = HashSet::new();
        axes.insert(0);
        axes.insert(1);

        let res_axes = t1.flip_axes(&axes).unwrap();
        let res_axes_mt = t1.flip_axes_mt(&axes).unwrap();

        let res = t1.flip();
        let res_mt = t1.flip_mt();

        assert_eq!(res_axes, res_axes_mt);
        assert_eq!(res, res_mt);
    }

    #[test]
    fn invalid_flip_axes() {
        let t1 = Tensor::<i32>::new(
            &shape![1, 2, 3, 4],
            vec![1; 24],
        ).unwrap();

        let mut axes = HashSet::new();
        axes.insert(4);

        let err = t1.flip_axes(&axes).unwrap_err();
        match err {
            TensorErrors::AxisOutOfBounds { axis: _, rank: _ } => {},
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn transform_refs_elementwise() {
        let t1 = Tensor::new(&shape![2, 3, 4], (0..24).collect()).unwrap();
        let ans = Tensor::new(&shape![2, 3, 4], (1..25).collect()).unwrap();

        assert_eq!(t1.map(|x| x + &1), ans);
    }

    #[test]
    fn invalid_concat_diff_ranks() {
        let t1 = Tensor::<i32>::new(&shape![1, 2, 3, 4], (0..24).collect()).unwrap();
        let t2 = Tensor::<i32>::new(&shape![2, 3, 4], (0..24).collect()).unwrap();

        let err = t1.concat(&t2, 0).unwrap_err();
        match err {
            TensorErrors::RanksDoNotMatch(..) => {},
            _ => panic!("Incorrect error")
        }

        let err = t1.concat_mt(&t2, 0).unwrap_err();
        match err {
            TensorErrors::RanksDoNotMatch(..) => {},
            _ => panic!("Incorrect error")
        }
    }

    #[test]
    fn pooling_invalid_shape_rank() {
        let t1 = Tensor::<f64>::new(&shape![2, 4, 3], (0..24).map(f64::from).collect()).unwrap();

        let err = t1.pool(
            pool_avg,
            &shape![1, 1, 1],
            &shape![1, 1],
            0.0,
        ).unwrap_err();
        match err {
            TensorErrors::RanksDoNotMatch(..) => {},
            _ => panic!("Incorrect error")
        }

        let err = t1.pool(
            pool_avg,
            &shape![1, 1],
            &shape![1, 1, 1],
            0.0,
        ).unwrap_err();
        match err {
            TensorErrors::RanksDoNotMatch(..) => {},
            _ => panic!("Incorrect error")
        }

        let err = t1.pool_indexed(
            |_, m| { pool_max(m) },
            &shape![1, 1],
            &shape![1, 1, 1],
            0.0,
        ).unwrap_err();
        match err {
            TensorErrors::RanksDoNotMatch(..) => {},
            _ => panic!("Incorrect error")
        }

        let err = t1.pool_indexed(
            |_, m| { pool_max(m) },
            &shape![1, 1, 1],
            &shape![1, 1],
            0.0,
        ).unwrap_err();
        match err {
            TensorErrors::RanksDoNotMatch(..) => {},
            _ => panic!("Incorrect error")
        }

        let err = t1.pool_mt(
            &pool_min,
            &shape![1, 1],
            &shape![1, 1, 1],

            0.0,
        ).unwrap_err();
        match err {
            TensorErrors::RanksDoNotMatch(..) => {},
            _ => panic!("Incorrect error")
        }

        let err = t1.pool_mt(
            &pool_min,
            &shape![1, 1, 1],
            &shape![1, 1],

            0.0,
        ).unwrap_err();
        match err {
            TensorErrors::RanksDoNotMatch(..) => {},
            _ => panic!("Incorrect error")
        }

        let err = t1.pool_indexed_mt(
            &|_, m| { pool_sum(m) },
            &shape![1, 1, 1],
            &shape![1, 1],
            0.0,
        ).unwrap_err();
        match err {
            TensorErrors::RanksDoNotMatch(..) => {},
            _ => panic!("Incorrect error")
        }

        let err = t1.pool_indexed_mt(
            &|_, m| { pool_sum(m) },
            &shape![1, 1],
            &shape![1, 1, 1],
            0.0,
        ).unwrap_err();
        match err {
            TensorErrors::RanksDoNotMatch(..) => {},
            _ => panic!("Incorrect error")
        }
    }

    #[test]
    fn invalid_transpose_rank() {
        let t1 = Tensor::new(&shape![2, 3, 4], (0..24).collect()).unwrap();

        let err = t1.transpose(&Transpose::default(1)).unwrap_err();
        match err {
            TensorErrors::TransposeIncompatibleRank { .. } => {},
            _ => panic!("Incorrect error")
        }

        let err = t1.transpose_mt(&Transpose::default(4)).unwrap_err();
        match err {
            TensorErrors::TransposeIncompatibleRank { .. } => {},
            _ => panic!("Incorrect error")
        }
    }

    #[test]
    fn invalid_slicing_incorrect_ranges() {
        let mut t1 = Tensor::<i32>::new(&shape![1, 2, 3, 4], (0..24).collect()).unwrap();

        let err = t1.slice(&[0..1, 0..1, 1..0, 0..1]).unwrap_err();
        match err {
            TensorErrors::InvalidInterval { .. } => {},
            _ => panic!("Incorrect error"),
        }

        let err = t1.slice_mut(&[0..1, 0..1, 1..0, 0..1]).unwrap_err();
        match err {
            TensorErrors::InvalidInterval { .. } => {},
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn invalid_slicing_num_indices_wrong() {
        let mut t1 = Tensor::<i32>::new(&shape![1, 2, 3, 4], (0..24).collect()).unwrap();

        let err = t1.slice(&[0..1]).unwrap_err();
        match err {
            TensorErrors::SliceIncompatibleShape { .. } => {},
            _ => panic!("Incorrect error"),
        }

        let err = t1.slice_mut(&[0..1]).unwrap_err();
        match err {
            TensorErrors::SliceIncompatibleShape { .. } => {},
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn slice_mut_invalid_indices_out_of_bounds() {
        let mut t1 = Tensor::<i32>::new(&shape![1, 2, 3, 4], (0..24).collect()).unwrap();

        let err = t1.slice_mut(&[0..1, 0..1, 2..5, 0..3]).unwrap_err();
        match err {
            TensorErrors::SliceIndicesOutOfBounds { .. } => {},
            _ => panic!("Incorrect error"),
        }
    }
    
    #[test]
    fn slicing_scalar_tensor() {
        let t1 = Tensor::<i32>::new(&shape![], vec![42]).unwrap();
        let sliced = t1.slice(&[]).unwrap();
        assert_eq!(sliced, t1);
        assert_eq!(sliced.elements(), &[42]);
    }
    
    #[test]
    fn slicing_empty_tensor() {
        let t1 = Tensor::<i32>::new(&shape![0, 3], vec![]).unwrap();
        let sliced = t1.slice(&[0..0, 1..2]).unwrap();
        let ans = Tensor::<i32>::new(&shape![0, 1], vec![]).unwrap();

        assert_eq!(sliced, ans);
        assert!(sliced.elements().is_empty());
    }
    
    #[test]
    fn cannot_flatten_scalar_tensor() {
        let t1 = Tensor::<i32>::new(&shape![], vec![42]).unwrap();
        let error = t1.flatten(0).err().unwrap();

        match error {
            TensorErrors::AxisOutOfBounds { .. } => {}
            _ => panic!("Incorrect error"),
        }
    }
    
    #[test]
    fn flip_scalar_tensor() {
        let t1 = Tensor::new(&shape![], vec![42]).unwrap();

        assert_eq!(t1.flip(), t1);
        assert_eq!(t1.flip_mt(), t1);

        let axes = HashSet::new();
        assert_eq!(t1.flip_axes(&axes).unwrap(), t1);
        assert_eq!(t1.flip_axes_mt(&axes).unwrap(), t1);

        let mut invalid_axes = HashSet::new();
        invalid_axes.insert(0);
        let err = t1.flip_axes(&invalid_axes).unwrap_err();
        match err {
            TensorErrors::AxisOutOfBounds { .. } => {}
            _ => panic!("Incorrect error"),
        }
    }
    
    #[test]
    fn collect_empty_tensor() {
        let empty_vec: Vec<i32> = vec![];
        let t: Tensor<i32> = empty_vec.into_iter().collect();

        assert_eq!(t.rank(), 1);
        assert_eq!(t.shape()[0], 0);
        assert!(t.elements().is_empty());
    }
    
    #[test]
    fn enumerated_iter() {
        let mut t1 = Tensor::<i32>::new(&shape![2, 2], vec![1, 2, 3, 4]).unwrap();

        let expected = vec![
            (vec![0, 0], 1),
            (vec![0, 1], 2),
            (vec![1, 0], 3),
            (vec![1, 1], 4),
        ];

        let actual: Vec<(Vec<usize>, i32)> = t1.enumerated_iter().collect();
        assert_eq!(actual, expected);

        for (idx, val) in t1.enumerated_iter_mut() {
            *val += idx.iter().sum::<usize>() as i32;
        }

        let expected_mut = vec![
            (vec![0, 0], 1), // 1 + 0
            (vec![0, 1], 3), // 2 + 1
            (vec![1, 0], 4), // 3 + 1
            (vec![1, 1], 6), // 4 + 2
        ];
        let actual_mut: Vec<(Vec<usize>, i32)> =
            t1.enumerated_iter().map(|(i, v)| (i, v)).collect();
        assert_eq!(actual_mut, expected_mut);
    }
    
    #[test]
    fn concat_scalar_tensors() {
        let t1 = Tensor::new(&shape![], vec![1]).unwrap();
        let t2 = Tensor::new(&shape![], vec![2]).unwrap();

        let err = t1.concat(&t2, 0).unwrap_err();
        match err {
            TensorErrors::AxisOutOfBounds { .. } => {},
            _ => panic!("Incorrect error")
        }
    }

    #[test]
    fn concat_with_empty_tensor() {
        let t1 = Tensor::<i32>::from_shape(&shape![2, 3]);
        let t2 = Tensor::<i32>::new(&shape![0, 3], vec![]).unwrap();

        let res = t1.concat(&t2, 0).unwrap();
        assert_eq!(res, t1);

        let t3 = Tensor::<i32>::new(&shape![2, 0], vec![]).unwrap();
        let res2 = t1.concat(&t3, 1).unwrap();
        assert_eq!(res2, t1);
    }

    #[test]
    fn concat_multithreaded_with_empty_tensor() {
        let t1 = Tensor::<i32>::from_shape(&shape![100, 10]);
        let t2 = Tensor::<i32>::new(&shape![0, 10], vec![]).unwrap();

        let res = t1.concat_mt(&t2, 0).unwrap();
        assert_eq!(res, t1);
    }
}
