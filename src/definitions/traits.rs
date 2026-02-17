use crate::definitions::matrix::Matrix;
use crate::definitions::tensor::Tensor;

/// This trait allows you to specify that something can be infallibly converted into a type of `Tensor<T>`.
/// It has the `into_tensor` method that converts the value into a `Tensor<T>`, consuming it. Bear in mind
/// that this does then automatically derive an implementation for `TryIntoTensor`.
pub trait IntoTensor<T> {
    fn into_tensor(self) -> Tensor<T>;
}
/// This trait allows you to specify that something can be fallibly converted into a type of `Tensor<T>`.
/// It has the `try_into_tensor` method that attempts to convert the value into a `Tensor<T>`, consuming it
/// and returning an error value if not possible.
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

/// This trait allows you to specify that something can be infallibly converted into a `Matrix<T>`.
/// Bear in mind that this does automatically derive an implementation for `TryIntoMatrix<T>`.
pub trait IntoMatrix<T> {
    fn into_matrix(self) -> Matrix<T>;
}

/// This trait allows you to specify that something can be fallibly converted into a `Matrix<T>`,
/// returning an error value if it is not possible.
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