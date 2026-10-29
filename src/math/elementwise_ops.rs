use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::definitions::tensor::Tensor;
use crate::{impl_assign_op, impl_bin_op};
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

impl_bin_op!(Add, add);
impl_bin_op!(Mul, mul);
impl_bin_op!(Sub, sub);
impl_bin_op!(Div, div);
impl_bin_op!(Rem, rem);
impl_bin_op!(BitXor, bitxor);
impl_bin_op!(BitAnd, bitand);
impl_bin_op!(BitOr, bitor);

impl_assign_op!(AddAssign, add_assign);
impl_assign_op!(SubAssign, sub_assign);
impl_assign_op!(MulAssign, mul_assign);
impl_assign_op!(DivAssign, div_assign);
impl_assign_op!(RemAssign, rem_assign);
impl_assign_op!(BitXorAssign, bitxor_assign);
impl_assign_op!(BitAndAssign, bitand_assign);
impl_assign_op!(BitOrAssign, bitor_assign);

impl<T: Neg<Output = T> + Clone> Neg for Tensor<T> {
    type Output = Tensor<T>;

    fn neg(self) -> Self::Output {
        self.map(|x| -x)
    }
}
impl<T: Neg<Output = T> + Clone> Neg for Matrix<T> {
    type Output = Matrix<T>;

    fn neg(self) -> Self::Output {
        self.map(|x| -x)
    }
}

#[cfg(test)]
mod elementwise_ops_tests {
    use crate::test_helpers::assert_panics;
    use num::complex::Complex64;
    use tensor_math::definitions::matrix::Matrix;
    use tensor_math::definitions::tensor::Tensor;
    use tensor_math::definitions::traits::IntoMatrix;
    use tensor_math::shape;
    use tensor_math::utilities::matrix::identity;

    #[test]
    fn add_tensors() {
        let t1 = Tensor::<i32>::new(shape![2, 3], vec![0, 0, 0, 2, 3, -1]).unwrap();
        let t2 = Tensor::<i32>::new(shape![2, 3], vec![1, 1, 1, 2, 4, -10]).unwrap();
        let ans = Tensor::<i32>::new(shape![2, 3], vec![1, 1, 1, 4, 7, -11]).unwrap();
        assert_eq!(ans, t1 + t2);
    }

    #[test]
    fn subtract_tensors() {
        let t1 = Tensor::<i32>::new(shape![2, 3], vec![0, 0, 0, 2, 3, -1]).unwrap();
        let t2 = Tensor::<i32>::new(shape![2, 3], vec![1, 1, 1, 2, 4, -10]).unwrap();
        let ans = Tensor::<i32>::new(shape![2, 3], vec![1, 1, 1, 0, 1, -9]).unwrap();
        assert_eq!(ans, t2 - t1);
    }

    #[test]
    fn multiply_tensors() {
        let t1 = Tensor::<i32>::new(shape![2, 3], vec![0, 0, 0, 2, 3, -1]).unwrap();
        let t2 = &Tensor::<i32>::new(shape![2, 3], vec![1, 1, 1, 2, 4, -10]).unwrap();
        let ans = Tensor::<i32>::new(shape![2, 3], vec![0, 0, 0, 4, 12, 10]).unwrap();
        assert_eq!(ans, t1 * t2);
    }

    #[test]
    fn divide_tensors() {
        let t1 = Tensor::<i32>::new(shape![2, 3], vec![0, 0, 0, 2, 5, -1]).unwrap();
        let t2 = &Tensor::<i32>::new(shape![2, 3], vec![1, 1, 1, 2, 4, -10]).unwrap();
        let ans = Tensor::<i32>::new(shape![2, 3], vec![0, 0, 0, 1, 1, 0]).unwrap();
        assert_eq!(ans, t1 / t2);
    }

    #[test]
    fn incompatible_shapes_for_elementwise_bin_op() {
        let t1 = Tensor::<i32>::rand(shape![2, 3]);
        let t2 = Tensor::<i32>::rand(shape![2, 2]);

        assert_panics(|| {
            let _ = &t1 + &t2;
        });
        assert_panics(|| {
            let _ = &t1 + t2.clone();
        });
        assert_panics(|| {
            let _ = t1.clone() + &t2;
        });
        assert_panics(|| {
            let _ = t1 + t2;
        });
    }

    #[test]
    fn mat_operators() {
        let m1 = -identity::<Complex64>(2) / Complex64::from(2.0);
        let m2 = identity::<Complex64>(2) * (Complex64::ONE + Complex64::I);
        let ans = Matrix::new(
            2,
            2,
            vec![
                Complex64 { re: 0.5, im: 1.0 },
                Complex64 { re: 0.0, im: 0.0 },
                Complex64 { re: 0.0, im: 0.0 },
                Complex64 { re: 0.5, im: 1.0 },
            ],
        )
        .unwrap();

        assert_eq!(m1 + m2, ans);
    }

    #[test]
    fn incompatible_shapes_for_elementwise_bin_op_mat() {
        let m1 = Matrix::<i32>::rand(2, 2);
        let m2 = Matrix::<i32>::rand(3, 3);

        assert_panics(|| {
            let _ = &m1 + &m2;
        });
        assert_panics(|| {
            let _ = &m1 + m2.clone();
        });
        assert_panics(|| {
            let _ = m1.clone() + &m2;
        });
        assert_panics(|| {
            let _ = m1 + m2;
        });
    }

    #[test]
    fn unary_minus() {
        let t1 = Tensor::<i32>::new(shape![2, 3], vec![1, 3, -4, 4, 5, 2]).unwrap();
        let ans = Tensor::<i32>::new(shape![2, 3], vec![-1, -3, 4, -4, -5, -2]).unwrap();

        assert_eq!(ans, -t1);
    }

    #[test]
    #[should_panic]
    fn divide_by_zero() {
        let t1 = Tensor::<i32>::rand(shape![2, 2]);
        let mut t2 = Tensor::<i32>::rand(shape![2, 2]);
        t2[&[1, 1]] = 0;
        let _ = t1 / t2;
    }

    #[test]
    #[should_panic]
    fn divide_mat_by_zero() {
        let m1 = Matrix::new(2, 2, vec![1, 2, 3, 4]).unwrap();
        let _ = m1 / 0;
    }

    #[test]
    fn single_element_implementations() {
        let t1 = Tensor::<i32>::new(shape![2, 3], vec![0, 1, 2, -1, 4, -10]).unwrap();
        let val = 5;

        let ans_add = Tensor::<i32>::new(shape![2, 3], vec![5, 6, 7, 4, 9, -5]).unwrap();
        let ans_sub = Tensor::<i32>::new(shape![2, 3], vec![-5, -4, -3, -6, -1, -15]).unwrap();
        let ans_mul = Tensor::<i32>::new(shape![2, 3], vec![0, 5, 10, -5, 20, -50]).unwrap();
        let ans_div = Tensor::<i32>::new(shape![2, 3], vec![0, 0, 0, 0, 0, -2]).unwrap();
        assert_eq!(ans_add, &t1 + val);
        assert_eq!(ans_sub, t1.clone() - &val);
        assert_eq!(ans_mul, &t1 * val);
        assert_eq!(ans_div, t1 / val);
    }

    #[test]
    fn mat_ref_and_single_val_op() {
        let m1 = vec![1, 2, 3, 4].into_matrix().reshape(2, 2).unwrap();

        let ans = vec![0, 1, 2, 3].into_matrix().reshape(2, 2).unwrap();

        assert_eq!(&m1 - 1, ans);
    }

    #[test]
    fn mat_and_single_val_ref_op() {
        let m1 = vec![1, 2, 3, 4].into_matrix().reshape(2, 2).unwrap();
        let ans = vec![0, 1, 2, 3].into_matrix().reshape(2, 2).unwrap();
        assert_eq!(m1 - &1, ans);
    }

    #[test]
    fn mat_and_mat_ref() {
        let m1 = vec![1, 2, 3, 4].into_matrix().reshape(2, 2).unwrap();
        let m2 = Matrix::from_value(2, 2, 1);
        let ans = vec![0, 1, 2, 3].into_matrix().reshape(2, 2).unwrap();

        assert_eq!(m1 - &m2, ans);
    }

    #[test]
    fn empty_tensor_ops() {
        let t1 = Tensor::<i32>::new(shape![0, 3], vec![]).unwrap();
        let t2 = Tensor::<i32>::new(shape![0, 3], vec![]).unwrap();
        let ans = Tensor::<i32>::new(shape![0, 3], vec![]).unwrap();

        assert_eq!(&t1 + t2, ans);
        assert_eq!(&t1 * 5, ans);
    }

    #[test]
    fn empty_matrix_ops() {
        let m1 = Matrix::<f64>::new(0, 2, vec![]).unwrap();
        let m2 = Matrix::<f64>::new(0, 2, vec![]).unwrap();
        let ans = Matrix::<f64>::new(0, 2, vec![]).unwrap();

        assert_eq!(m1 + m2, ans);
    }

    #[test]
    fn rank_0_tensor_ops() {
        let t1 = Tensor::<i32>::new(shape![], vec![10]).unwrap();
        let t2 = Tensor::<i32>::new(shape![], vec![5]).unwrap();
        let ans_add = Tensor::<i32>::new(shape![], vec![15]).unwrap();
        let ans_mul = Tensor::<i32>::new(shape![], vec![50]).unwrap();

        assert_eq!(&t1 + &t2, ans_add);
        assert_eq!(t1 * t2, ans_mul);
    }

    #[test]
    fn remainder_and_bitwise_tensor_operators() {
        let lhs = Tensor::<i32>::new(shape![2, 2], vec![10, 6, 3, 2]).unwrap();
        let rhs = Tensor::<i32>::new(shape![2, 2], vec![3, 2, 2, 5]).unwrap();

        assert_eq!(
            Tensor::new(shape![2, 2], vec![1, 0, 1, 2]).unwrap(),
            &lhs % &rhs
        );
        assert_eq!(
            Tensor::new(shape![2, 2], vec![9, 4, 1, 7]).unwrap(),
            &lhs ^ &rhs
        );
        assert_eq!(
            Tensor::new(shape![2, 2], vec![2, 2, 2, 0]).unwrap(),
            &lhs & &rhs
        );
        assert_eq!(
            Tensor::new(shape![2, 2], vec![11, 6, 3, 7]).unwrap(),
            &lhs | &rhs
        );
    }

    #[test]
    fn elementwise_matrix_operators() {
        let lhs = Matrix::new(2, 2, vec![10, 6, 3, 2]).unwrap();
        let rhs = Matrix::new(2, 2, vec![3, 2, 2, 5]).unwrap();

        assert_eq!(Matrix::new(2, 2, vec![13, 8, 5, 7]).unwrap(), &lhs + &rhs);
        assert_eq!(Matrix::new(2, 2, vec![7, 4, 1, -3]).unwrap(), &lhs - &rhs);
        assert_eq!(Matrix::new(2, 2, vec![30, 12, 6, 10]).unwrap(), &lhs * &rhs);
        assert_eq!(Matrix::new(2, 2, vec![3, 3, 1, 0]).unwrap(), &lhs / &rhs);
        assert_eq!(Matrix::new(2, 2, vec![1, 0, 1, 2]).unwrap(), &lhs % &rhs);
        assert_eq!(Matrix::new(2, 2, vec![9, 4, 1, 7]).unwrap(), &lhs ^ &rhs);
        assert_eq!(Matrix::new(2, 2, vec![2, 2, 2, 0]).unwrap(), &lhs & &rhs);
        assert_eq!(Matrix::new(2, 2, vec![11, 6, 3, 7]).unwrap(), &lhs | &rhs);
    }

    #[test]
    fn remainder_and_bitwise_scalar_operators() {
        let tensor = Tensor::<i32>::new(shape![2], vec![10, 6]).unwrap();
        let matrix = Matrix::new(1, 2, vec![10, 6]).unwrap();

        assert_eq!(Tensor::new(shape![2], vec![1, 0]).unwrap(), &tensor % 3);
        assert_eq!(
            Tensor::new(shape![2], vec![1, 0]).unwrap(),
            tensor.clone() % &3
        );
        assert_eq!(Tensor::new(shape![2], vec![9, 5]).unwrap(), &tensor ^ &3);
        assert_eq!(
            Tensor::new(shape![2], vec![2, 2]).unwrap(),
            tensor.clone() & 3
        );
        assert_eq!(Tensor::new(shape![2], vec![11, 7]).unwrap(), tensor | &3);

        assert_eq!(Matrix::new(1, 2, vec![1, 0]).unwrap(), &matrix % 3);
        assert_eq!(Matrix::new(1, 2, vec![1, 0]).unwrap(), matrix.clone() % &3);
        assert_eq!(Matrix::new(1, 2, vec![9, 5]).unwrap(), &matrix ^ &3);
        assert_eq!(Matrix::new(1, 2, vec![2, 2]).unwrap(), matrix.clone() & 3);
        assert_eq!(Matrix::new(1, 2, vec![11, 7]).unwrap(), matrix | &3);
    }

    #[test]
    fn assignment_operators_for_tensors() {
        let lhs = Tensor::<i32>::new(shape![2, 2], vec![10, 6, 3, 2]).unwrap();
        let rhs = Tensor::<i32>::new(shape![2, 2], vec![3, 2, 2, 5]).unwrap();

        let mut value = lhs.clone();
        value += rhs.clone();
        assert_eq!(Tensor::new(shape![2, 2], vec![13, 8, 5, 7]).unwrap(), value);

        let mut value = lhs.clone();
        value -= rhs.clone();
        assert_eq!(Tensor::new(shape![2, 2], vec![7, 4, 1, -3]).unwrap(), value);

        let mut value = lhs.clone();
        value *= rhs.clone();
        assert_eq!(
            Tensor::new(shape![2, 2], vec![30, 12, 6, 10]).unwrap(),
            value
        );

        let mut value = lhs.clone();
        value /= rhs.clone();
        assert_eq!(Tensor::new(shape![2, 2], vec![3, 3, 1, 0]).unwrap(), value);

        let mut value = lhs.clone();
        value %= rhs.clone();
        assert_eq!(Tensor::new(shape![2, 2], vec![1, 0, 1, 2]).unwrap(), value);

        let mut value = lhs.clone();
        value ^= rhs.clone();
        assert_eq!(Tensor::new(shape![2, 2], vec![9, 4, 1, 7]).unwrap(), value);

        let mut value = lhs.clone();
        value &= rhs.clone();
        assert_eq!(Tensor::new(shape![2, 2], vec![2, 2, 2, 0]).unwrap(), value);

        let mut value = lhs;
        value |= rhs;
        assert_eq!(Tensor::new(shape![2, 2], vec![11, 6, 3, 7]).unwrap(), value);
    }

    #[test]
    fn assignment_operators_for_matrices() {
        let lhs = Matrix::new(2, 2, vec![10, 6, 3, 2]).unwrap();
        let rhs = Matrix::new(2, 2, vec![3, 2, 2, 5]).unwrap();

        let mut value = lhs.clone();
        value += rhs.clone();
        assert_eq!(Matrix::new(2, 2, vec![13, 8, 5, 7]).unwrap(), value);

        let mut value = lhs.clone();
        value -= rhs.clone();
        assert_eq!(Matrix::new(2, 2, vec![7, 4, 1, -3]).unwrap(), value);

        let mut value = lhs.clone();
        value *= rhs.clone();
        assert_eq!(Matrix::new(2, 2, vec![30, 12, 6, 10]).unwrap(), value);

        let mut value = lhs.clone();
        value /= rhs.clone();
        assert_eq!(Matrix::new(2, 2, vec![3, 3, 1, 0]).unwrap(), value);

        let mut value = lhs.clone();
        value %= rhs.clone();
        assert_eq!(Matrix::new(2, 2, vec![1, 0, 1, 2]).unwrap(), value);

        let mut value = lhs.clone();
        value ^= rhs.clone();
        assert_eq!(Matrix::new(2, 2, vec![9, 4, 1, 7]).unwrap(), value);

        let mut value = lhs.clone();
        value &= rhs.clone();
        assert_eq!(Matrix::new(2, 2, vec![2, 2, 2, 0]).unwrap(), value);

        let mut value = lhs;
        value |= rhs;
        assert_eq!(Matrix::new(2, 2, vec![11, 6, 3, 7]).unwrap(), value);
    }

    #[test]
    fn empty_shapes_for_elementwise_operators() {
        for shape in [shape![0, 3], shape![3, 0], shape![2, 0, 3]] {
            let lhs = Tensor::<i32>::new(shape.clone(), vec![]).unwrap();
            let rhs = Tensor::<i32>::new(shape.clone(), vec![]).unwrap();

            assert_eq!(Tensor::new(shape.clone(), vec![]).unwrap(), &lhs + &rhs);
            assert_eq!(Tensor::new(shape.clone(), vec![]).unwrap(), &lhs - &rhs);
            assert_eq!(Tensor::new(shape.clone(), vec![]).unwrap(), &lhs * &rhs);
            assert_eq!(Tensor::new(shape.clone(), vec![]).unwrap(), &lhs / &rhs);
            assert_eq!(Tensor::new(shape.clone(), vec![]).unwrap(), &lhs % &rhs);
            assert_eq!(Tensor::new(shape.clone(), vec![]).unwrap(), &lhs ^ &rhs);
            assert_eq!(Tensor::new(shape.clone(), vec![]).unwrap(), &lhs & &rhs);
            assert_eq!(Tensor::new(shape.clone(), vec![]).unwrap(), &lhs | &rhs);
            assert_eq!(Tensor::new(shape.clone(), vec![]).unwrap(), &lhs + 7);

            let mut assigned = lhs;
            assigned += rhs;
            assert_eq!(Tensor::new(shape, vec![]).unwrap(), assigned);
        }

        for (rows, cols) in [(0, 3), (3, 0)] {
            let lhs = Matrix::<i32>::new(rows, cols, vec![]).unwrap();
            let rhs = Matrix::<i32>::new(rows, cols, vec![]).unwrap();

            assert_eq!(Matrix::new(rows, cols, vec![]).unwrap(), &lhs + &rhs);
            assert_eq!(Matrix::new(rows, cols, vec![]).unwrap(), &lhs - &rhs);
            assert_eq!(Matrix::new(rows, cols, vec![]).unwrap(), &lhs * &rhs);
            assert_eq!(Matrix::new(rows, cols, vec![]).unwrap(), &lhs / &rhs);
            assert_eq!(Matrix::new(rows, cols, vec![]).unwrap(), &lhs % &rhs);
            assert_eq!(Matrix::new(rows, cols, vec![]).unwrap(), &lhs ^ &rhs);
            assert_eq!(Matrix::new(rows, cols, vec![]).unwrap(), &lhs & &rhs);
            assert_eq!(Matrix::new(rows, cols, vec![]).unwrap(), &lhs | &rhs);
            assert_eq!(Matrix::new(rows, cols, vec![]).unwrap(), &lhs + 7);

            let mut assigned = lhs;
            assigned += rhs;
            assert_eq!(Matrix::new(rows, cols, vec![]).unwrap(), assigned);
        }
    }

    #[test]
    fn rank_0_tensor_elementwise_and_assignment_operators() {
        let lhs = Tensor::<i32>::new(shape![], vec![10]).unwrap();
        let rhs = Tensor::<i32>::new(shape![], vec![3]).unwrap();

        assert_eq!(Tensor::new(shape![], vec![13]).unwrap(), &lhs + &rhs);
        assert_eq!(Tensor::new(shape![], vec![7]).unwrap(), &lhs - &rhs);
        assert_eq!(Tensor::new(shape![], vec![30]).unwrap(), &lhs * &rhs);
        assert_eq!(Tensor::new(shape![], vec![3]).unwrap(), &lhs / &rhs);
        assert_eq!(Tensor::new(shape![], vec![1]).unwrap(), &lhs % &rhs);
        assert_eq!(Tensor::new(shape![], vec![9]).unwrap(), &lhs ^ &rhs);
        assert_eq!(Tensor::new(shape![], vec![2]).unwrap(), &lhs & &rhs);
        assert_eq!(Tensor::new(shape![], vec![11]).unwrap(), &lhs | &rhs);
        assert_eq!(Tensor::new(shape![], vec![3]).unwrap(), &lhs / 3);

        let mut assigned = lhs.clone();
        assigned += rhs;
        assert_eq!(Tensor::new(shape![], vec![13]).unwrap(), assigned);
    }

    #[test]
    fn assignment_operators_reject_incompatible_shapes() {
        assert_panics(|| {
            let mut tensor = Tensor::<i32>::new(shape![0, 3], vec![]).unwrap();
            tensor += Tensor::<i32>::new(shape![0, 4], vec![]).unwrap();
        });

        assert_panics(|| {
            let mut matrix = Matrix::<i32>::new(0, 3, vec![]).unwrap();
            matrix += Matrix::<i32>::new(3, 0, vec![]).unwrap();
        });
    }

    #[test]
    fn unary_minus_for_matrices_and_empty_values() {
        let matrix = Matrix::new(2, 2, vec![1, -3, 4, 0]).unwrap();
        assert_eq!(Matrix::new(2, 2, vec![-1, 3, -4, 0]).unwrap(), -matrix);

        let empty_tensor = Tensor::<i32>::new(shape![2, 0], vec![]).unwrap();
        assert_eq!(Tensor::new(shape![2, 0], vec![]).unwrap(), -empty_tensor);

        let empty_matrix = Matrix::<i32>::new(2, 0, vec![]).unwrap();
        assert_eq!(Matrix::new(2, 0, vec![]).unwrap(), -empty_matrix);
    }
}
