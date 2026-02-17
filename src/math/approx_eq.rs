use float_cmp::{approx_eq, ApproxEq, F64Margin};
use num::complex::{Complex64, ComplexFloat};
use crate::definitions::matrix::Matrix;
use crate::definitions::tensor::Tensor;

impl ApproxEq for Tensor<f64> {
    type Margin = F64Margin;

    fn approx_eq<M: Into<Self::Margin>>(self, other: Self, margin: M) -> bool {
        if self.shape != other.shape { return false }

        let margin = margin.into();

        self.enumerated_iter().all(|(i, x)| {
            approx_eq!(f64, x, other[&i], margin.clone())
        })
    }
}

impl ApproxEq for Matrix<f64> {
    type Margin = F64Margin;

    fn approx_eq<M: Into<Self::Margin>>(self, other: Self, margin: M) -> bool {
        self.tensor.approx_eq(other.tensor, margin)
    }
}

impl ApproxEq for Tensor<Complex64> {
    type Margin = F64Margin;

    fn approx_eq<M: Into<Self::Margin>>(self, other: Self, margin: M) -> bool {
        if self.shape != other.shape { return false }

        let margin = margin.into();

        self.enumerated_iter().all(|(i, x)| {
            x.re().approx_eq(other[&i].re(), margin) && x.im().approx_eq(other[&i].im(), margin)
        })
    }
}

impl ApproxEq for Matrix<Complex64> {
    type Margin = F64Margin;

    fn approx_eq<M: Into<Self::Margin>>(self, other: Self, margin: M) -> bool {
        self.tensor.approx_eq(other.tensor, margin)
    }
}