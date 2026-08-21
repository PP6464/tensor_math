#[macro_export]
/// Creates a shape from varargs of type usize.
/// You will need to import the Shape struct, at `tensor_math::definitions::shape::Shape;`
macro_rules! shape {
    ($($shape_dimensions:expr),*$(,)?) => {
        Shape::new(vec![$($shape_dimensions),*])
    };
}

#[macro_export]
/// Constructs a transpose using the specified permutation.
/// Assumes the permutation is valid so will panic if it is not.
/// You will need to import the Transpose struct, at `tensor_math::definitions::transpose::Transpose;`
macro_rules! transpose {
    ($($x:expr),*$(,)?) => {
        Transpose::new(&vec![$($x),*]).unwrap()
    };
}

#[macro_export]
/// Implement an operation element-wise for tensors and matrices
/// Also allows you to implement operations with a tensor/matrix and a single value
/// By applying the operation between it and each element of the tensor/matrix in turn
macro_rules! impl_bin_op {
    ($op:ident, $op_fn:ident) => {
        impl<T: $op<Output = T> + Clone> $op<Tensor<T>> for Tensor<T> {
            type Output = Tensor<T>;

            fn $op_fn(self, rhs: Tensor<T>) -> Tensor<T> {
                assert_eq!(
                    self.shape(),
                    rhs.shape(),
                    "{}",
                    TensorErrors::ShapesIncompatible
                );

                let elements = self
                    .elements()
                    .into_iter()
                    .cloned()
                    .zip(rhs.elements().into_iter().cloned())
                    .map(|(a, b)| a.$op_fn(b))
                    .collect();

                Tensor::new(self.shape(), elements).unwrap()
            }
        }
        impl<T: $op<Output = T> + Clone> $op<Tensor<T>> for &Tensor<T> {
            type Output = Tensor<T>;

            fn $op_fn(self, rhs: Tensor<T>) -> Tensor<T> {
                assert_eq!(
                    self.shape(),
                    rhs.shape(),
                    "{}",
                    TensorErrors::ShapesIncompatible
                );

                let elements = self
                    .elements()
                    .into_iter()
                    .cloned()
                    .zip(rhs.elements().into_iter().cloned())
                    .map(|(a, b)| a.$op_fn(b))
                    .collect();

                Tensor::new(self.shape(), elements).unwrap()
            }
        }
        impl<T: $op<Output = T> + Clone> $op<&Tensor<T>> for &Tensor<T> {
            type Output = Tensor<T>;

            fn $op_fn(self, rhs: &Tensor<T>) -> Tensor<T> {
                assert_eq!(
                    self.shape(),
                    rhs.shape(),
                    "{}",
                    TensorErrors::ShapesIncompatible
                );

                let elements = self
                    .elements()
                    .into_iter()
                    .cloned()
                    .zip(rhs.elements().into_iter().cloned())
                    .map(|(a, b)| a.$op_fn(b))
                    .collect();

                Tensor::new(self.shape(), elements).unwrap()
            }
        }
        impl<T: $op<Output = T> + Clone> $op<&Tensor<T>> for Tensor<T> {
            type Output = Tensor<T>;

            fn $op_fn(self, rhs: &Tensor<T>) -> Tensor<T> {
                assert_eq!(
                    self.shape(),
                    rhs.shape(),
                    "{}",
                    TensorErrors::ShapesIncompatible
                );

                let elements = self
                    .elements()
                    .into_iter()
                    .cloned()
                    .zip(rhs.elements().into_iter().cloned())
                    .map(|(a, b)| a.$op_fn(b))
                    .collect();

                Tensor::new(self.shape(), elements).unwrap()
            }
        }
        impl<T: $op<Output = T> + Clone> $op<T> for &Tensor<T> {
            type Output = Tensor<T>;

            fn $op_fn(self, rhs: T) -> Tensor<T> {
                let elements = self
                    .elements()
                    .into_iter()
                    .cloned()
                    .map(|a| a.$op_fn(rhs.clone()))
                    .collect();

                Tensor::new(self.shape(), elements).unwrap()
            }
        }
        impl<T: $op<Output = T> + Clone> $op<T> for Tensor<T> {
            type Output = Tensor<T>;

            fn $op_fn(self, rhs: T) -> Tensor<T> {
                let elements = self
                    .elements()
                    .into_iter()
                    .cloned()
                    .map(|a| a.$op_fn(rhs.clone()))
                    .collect();

                Tensor::new(self.shape(), elements).unwrap()
            }
        }
        impl<T: $op<Output = T> + Clone> $op<&T> for Tensor<T> {
            type Output = Tensor<T>;

            fn $op_fn(self, rhs: &T) -> Tensor<T> {
                let elements = self
                    .elements()
                    .into_iter()
                    .cloned()
                    .map(|a| a.$op_fn(rhs.clone()))
                    .collect();

                Tensor::new(self.shape(), elements).unwrap()
            }
        }
        impl<T: $op<Output = T> + Clone> $op<&T> for &Tensor<T> {
            type Output = Tensor<T>;

            fn $op_fn(self, rhs: &T) -> Tensor<T> {
                let elements = self
                    .elements()
                    .into_iter()
                    .cloned()
                    .map(|a| a.$op_fn(rhs.clone()))
                    .collect();

                Tensor::new(self.shape(), elements).unwrap()
            }
        }
        impl<T: $op<Output = T> + Clone> $op<Matrix<T>> for Matrix<T> {
            type Output = Matrix<T>;

            fn $op_fn(self, rhs: Matrix<T>) -> Matrix<T> {
                assert_eq!(
                    self.shape(),
                    rhs.shape(),
                    "{}",
                    TensorErrors::ShapesIncompatible
                );

                let elements = self
                    .elements()
                    .into_iter()
                    .cloned()
                    .zip(rhs.elements().into_iter().cloned())
                    .map(|(a, b)| a.$op_fn(b))
                    .collect();

                Matrix::new(self.rows, self.cols, elements).unwrap()
            }
        }
        impl<T: $op<Output = T> + Clone> $op<Matrix<T>> for &Matrix<T> {
            type Output = Matrix<T>;

            fn $op_fn(self, rhs: Matrix<T>) -> Matrix<T> {
                assert_eq!(
                    self.shape(),
                    rhs.shape(),
                    "{}",
                    TensorErrors::ShapesIncompatible
                );

                let elements = self
                    .elements()
                    .into_iter()
                    .cloned()
                    .zip(rhs.elements().into_iter().cloned())
                    .map(|(a, b)| a.$op_fn(b))
                    .collect();

                Matrix::new(self.rows, self.cols, elements).unwrap()
            }
        }
        impl<T: $op<Output = T> + Clone> $op<&Matrix<T>> for &Matrix<T> {
            type Output = Matrix<T>;

            fn $op_fn(self, rhs: &Matrix<T>) -> Matrix<T> {
                assert_eq!(
                    self.shape(),
                    rhs.shape(),
                    "{}",
                    TensorErrors::ShapesIncompatible
                );

                let elements = self
                    .elements()
                    .into_iter()
                    .cloned()
                    .zip(rhs.elements().into_iter().cloned())
                    .map(|(a, b)| a.$op_fn(b))
                    .collect();

                Matrix::new(self.rows, self.cols, elements).unwrap()
            }
        }
        impl<T: $op<Output = T> + Clone> $op<&Matrix<T>> for Matrix<T> {
            type Output = Matrix<T>;

            fn $op_fn(self, rhs: &Matrix<T>) -> Matrix<T> {
                assert_eq!(
                    self.shape(),
                    rhs.shape(),
                    "{}",
                    TensorErrors::ShapesIncompatible
                );

                let elements = self
                    .elements()
                    .into_iter()
                    .cloned()
                    .zip(rhs.elements().into_iter().cloned())
                    .map(|(a, b)| a.$op_fn(b))
                    .collect();

                Matrix::new(self.rows, self.cols, elements).unwrap()
            }
        }
        impl<T: $op<Output = T> + Clone> $op<T> for &Matrix<T> {
            type Output = Matrix<T>;

            fn $op_fn(self, rhs: T) -> Matrix<T> {
                let elements = self
                    .elements()
                    .into_iter()
                    .cloned()
                    .map(|a| a.$op_fn(rhs.clone()))
                    .collect();

                Matrix::new(self.rows, self.cols, elements).unwrap()
            }
        }
        impl<T: $op<Output = T> + Clone> $op<T> for Matrix<T> {
            type Output = Matrix<T>;

            fn $op_fn(self, rhs: T) -> Matrix<T> {
                let elements = self
                    .elements()
                    .into_iter()
                    .cloned()
                    .map(|a| a.$op_fn(rhs.clone()))
                    .collect();

                Matrix::new(self.rows, self.cols, elements).unwrap()
            }
        }
        impl<T: $op<Output = T> + Clone> $op<&T> for Matrix<T> {
            type Output = Matrix<T>;

            fn $op_fn(self, rhs: &T) -> Matrix<T> {
                let elements = self
                    .elements()
                    .into_iter()
                    .cloned()
                    .map(|a| a.$op_fn(rhs.clone()))
                    .collect();

                Matrix::new(self.rows, self.cols, elements).unwrap()
            }
        }
        impl<T: $op<Output = T> + Clone> $op<&T> for &Matrix<T> {
            type Output = Matrix<T>;

            fn $op_fn(self, rhs: &T) -> Matrix<T> {
                let elements = self
                    .elements()
                    .into_iter()
                    .cloned()
                    .map(|a| a.$op_fn(rhs.clone()))
                    .collect();

                Matrix::new(self.rows, self.cols, elements).unwrap()
            }
        }
    };
}

pub(crate) mod internal_macros {
    #[macro_export]
    macro_rules! mat_addr {
        ($indices:expr, $cols:expr) => {
            $indices.0 * $cols + $indices.1
        };
    }
}
