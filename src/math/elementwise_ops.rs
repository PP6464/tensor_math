use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::definitions::tensor::Tensor;
use crate::impl_bin_op;
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Rem, Sub};

impl_bin_op!(Add, add);
impl_bin_op!(Mul, mul);
impl_bin_op!(Sub, sub);
impl_bin_op!(Div, div);
impl_bin_op!(Rem, rem);
impl_bin_op!(BitXor, bitxor);
impl_bin_op!(BitAnd, bitand);
impl_bin_op!(BitOr, bitor);

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
