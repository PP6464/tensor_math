use crate::definitions::matrix::Matrix;
use crate::definitions::tensor::Tensor;

/*
--------------------------------------------
Tensor Conversion Traits
--------------------------------------------
*/

/// This trait allows you to specify that something can be infallibly converted into a tensor.
/// This automatically derives an implementation for `TryIntoTensor`.
pub trait IntoTensor<T> {
    fn into_tensor(self) -> Tensor<T>;
}

/// This trait allows you to specify that something can be fallibly converted into a tensor.
pub trait TryIntoTensor<T> {
    type Error;

    fn try_into_tensor(self) -> Result<Tensor<T>, Self::Error>;
}

impl<T, O: IntoTensor<T>> TryIntoTensor<T> for O {
    type Error = ();

    fn try_into_tensor(self) -> Result<Tensor<T>, Self::Error> {
        Ok(self.into_tensor())
    }
}

/*
--------------------------------------------
* Matrix Conversion Traits
--------------------------------------------
*/

/// This trait allows you to specify that something can be infallibly converted into a matrix.
pub trait IntoMatrix<T> {
    fn into_matrix(self) -> Matrix<T>;
}

/// This trait allows you to specify that something can be fallibly converted into a matrix.
pub trait TryIntoMatrix<T> {
    type Error;

    fn try_into_matrix(self) -> Result<Matrix<T>, Self::Error>;
}
