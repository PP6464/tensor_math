use crate::definitions::matrix::Matrix;
use crate::definitions::tensor::Tensor;

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

/// This trait allows you to specify that something can be infallibly converted into a matrix.
/// This automatically derives an implementation for `TryIntoMatrix`.
pub trait IntoMatrix<T> {
    fn into_matrix(self) -> Matrix<T>;
}

/// This trait allows you to specify that something can be fallibly converted into a matrix.
pub trait TryIntoMatrix<T> {
    type Error;

    fn try_into_matrix(self) -> Result<Matrix<T>, Self::Error>;
}

impl<T, O: IntoMatrix<T>> TryIntoMatrix<T> for O {
    type Error = ();

    fn try_into_matrix(self) -> Result<Matrix<T>, Self::Error> {
        Ok(self.into_matrix())
    }
}
