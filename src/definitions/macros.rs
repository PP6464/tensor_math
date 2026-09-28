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
        impl<T: $op<Output = T>> $op<Tensor<T>> for Tensor<T> {
            type Output = Tensor<T>;

            fn $op_fn(self, rhs: Tensor<T>) -> Tensor<T> {
                assert_eq!(
                    self.shape(),
                    rhs.shape(),
                    "{}",
                    TensorErrors::IncompatibleShapes { shape_1: self.shape(), shape_2: rhs.shape(), op: "elementwise_op", }
                );

                let strides = self.strides.clone();
                let shape = self.shape.clone();

                let elements = self
                    .elements
                    .into_iter()
                    .zip(rhs.elements.into_iter())
                    .map(|(a, b)| a.$op_fn(b))
                    .collect();

                Tensor {
                    strides,
                    shape,
                    elements
                }
            }
        }
        impl<T: $op<Output = T> + Clone> $op<Tensor<T>> for &Tensor<T> {
            type Output = Tensor<T>;

            fn $op_fn(self, rhs: Tensor<T>) -> Tensor<T> {
                assert_eq!(
                    self.shape(),
                    rhs.shape(),
                    "{}",
                    TensorErrors::IncompatibleShapes { shape_1: self.shape(), shape_2: rhs.shape(), op: "elementwise_op", }
                );

                let strides = self.strides.clone();
                let shape = self.shape.clone();

                let elements = self
                    .elements
                    .iter()
                    .cloned()
                    .zip(rhs.elements.into_iter())
                    .map(|(a, b)| a.$op_fn(b))
                    .collect();

                Tensor {
                    strides,
                    shape,
                    elements
                }
            }
        }
        impl<T: $op<Output = T> + Clone> $op<&Tensor<T>> for &Tensor<T> {
            type Output = Tensor<T>;

            fn $op_fn(self, rhs: &Tensor<T>) -> Tensor<T> {
                assert_eq!(
                    self.shape(),
                    rhs.shape(),
                    "{}",
                    TensorErrors::IncompatibleShapes { shape_1: self.shape(), shape_2: rhs.shape(), op: "elementwise_op", }
                );

                let elements = self
                    .elements
                    .iter()
                    .cloned()
                    .zip(rhs.elements.iter().cloned())
                    .map(|(a, b)| a.$op_fn(b))
                    .collect();

                Tensor {
                    strides: self.strides.clone(),
                    shape: self.shape.clone(),
                    elements
                }
            }
        }
        impl<T: $op<Output = T> + Clone> $op<&Tensor<T>> for Tensor<T> {
            type Output = Tensor<T>;

            fn $op_fn(self, rhs: &Tensor<T>) -> Tensor<T> {
                assert_eq!(
                    self.shape(),
                    rhs.shape(),
                    "{}",
                    TensorErrors::IncompatibleShapes { shape_1: self.shape(), shape_2: rhs.shape(), op: "elementwise_op", }
                );

                let strides = self.strides.clone();
                let shape = self.shape.clone();

                let elements = self
                    .elements
                    .into_iter()
                    .zip(rhs.elements.iter().cloned())
                    .map(|(a, b)| a.$op_fn(b))
                    .collect();

                Tensor {
                    strides,
                    shape,
                    elements,
                }
            }
        }
        impl<T: $op<Output = T> + Clone> $op<T> for &Tensor<T> {
            type Output = Tensor<T>;

            fn $op_fn(self, rhs: T) -> Tensor<T> {
                self.map_refs(|x| x.clone().$op_fn(rhs.clone()))
            }
        }
        impl<T: $op<Output = T> + Clone> $op<T> for Tensor<T> {
            type Output = Tensor<T>;

            fn $op_fn(self, rhs: T) -> Tensor<T> {
                self.map(|x| x.$op_fn(rhs.clone()))
            }
        }
        impl<T: $op<Output = T> + Clone> $op<&T> for Tensor<T> {
            type Output = Tensor<T>;

            fn $op_fn(self, rhs: &T) -> Tensor<T> {
                self.map(|x| x.$op_fn(rhs.clone()))
            }
        }
        impl<T: $op<Output = T> + Clone> $op<&T> for &Tensor<T> {
            type Output = Tensor<T>;

            fn $op_fn(self, rhs: &T) -> Tensor<T> {
                self.map_refs(|x| x.clone().$op_fn(rhs.clone()))
            }
        }
        impl<T: $op<Output = T> + Clone> $op<Matrix<T>> for Matrix<T> {
            type Output = Matrix<T>;

            fn $op_fn(self, rhs: Matrix<T>) -> Matrix<T> {
                assert_eq!(
                    self.shape(),
                    rhs.shape(),
                    "{}",
                    TensorErrors::IncompatibleShapes { shape_1: self.shape(), shape_2: rhs.shape(), op: "elementwise_op", }
                );

                let (rows, cols) = (self.rows, self.cols);

                let elements = self
                    .elements
                    .into_iter()
                    .zip(rhs.elements.into_iter())
                    .map(|(a, b)| a.$op_fn(b))
                    .collect();

                Matrix {
                    rows,
                    cols,
                    elements
                }
            }
        }
        impl<T: $op<Output = T> + Clone> $op<Matrix<T>> for &Matrix<T> {
            type Output = Matrix<T>;

            fn $op_fn(self, rhs: Matrix<T>) -> Matrix<T> {
                assert_eq!(
                    self.shape(),
                    rhs.shape(),
                    "{}",
                    TensorErrors::IncompatibleShapes { shape_1: self.shape(), shape_2: rhs.shape(), op: "elementwise_op", }
                );

                let elements = self
                    .elements
                    .iter()
                    .cloned()
                    .zip(rhs.elements.into_iter())
                    .map(|(a, b)| a.$op_fn(b))
                    .collect();

                Matrix {
                    rows: self.rows,
                    cols: self.cols,
                    elements
                }
            }
        }
        impl<T: $op<Output = T> + Clone> $op<&Matrix<T>> for &Matrix<T> {
            type Output = Matrix<T>;

            fn $op_fn(self, rhs: &Matrix<T>) -> Matrix<T> {
                assert_eq!(
                    self.shape(),
                    rhs.shape(),
                    "{}",
                    TensorErrors::IncompatibleShapes { shape_1: self.shape(), shape_2: rhs.shape(), op: "elementwise_op", }
                );

                let elements = self
                    .elements
                    .iter()
                    .cloned()
                    .zip(rhs.elements.iter().cloned())
                    .map(|(a, b)| a.$op_fn(b))
                    .collect();

                Matrix {
                    rows: self.rows,
                    cols: self.cols,
                    elements,
                }
            }
        }
        impl<T: $op<Output = T> + Clone> $op<&Matrix<T>> for Matrix<T> {
            type Output = Matrix<T>;

            fn $op_fn(self, rhs: &Matrix<T>) -> Matrix<T> {
                assert_eq!(
                    self.shape(),
                    rhs.shape(),
                    "{}",
                    TensorErrors::IncompatibleShapes { shape_1: self.shape(), shape_2: rhs.shape(), op: "elementwise_op", }
                );

                let elements = self
                    .elements
                    .into_iter()
                    .zip(rhs.elements.iter().cloned())
                    .map(|(a, b)| a.$op_fn(b))
                    .collect();

                Matrix {
                    rows: rhs.rows,
                    cols: rhs.cols,
                    elements,
                }
            }
        }
        impl<T: $op<Output = T> + Clone> $op<T> for &Matrix<T> {
            type Output = Matrix<T>;

            fn $op_fn(self, rhs: T) -> Matrix<T> {
                self.map_refs(|x| x.clone().$op_fn(rhs.clone()))
            }
        }
        impl<T: $op<Output = T> + Clone> $op<T> for Matrix<T> {
            type Output = Matrix<T>;

            fn $op_fn(self, rhs: T) -> Matrix<T> {
                self.map(|x| x.$op_fn(rhs.clone()))
            }
        }
        impl<T: $op<Output = T> + Clone> $op<&T> for Matrix<T> {
            type Output = Matrix<T>;

            fn $op_fn(self, rhs: &T) -> Matrix<T> {
                self.map(|x| x.$op_fn(rhs.clone()))
            }
        }
        impl<T: $op<Output = T> + Clone> $op<&T> for &Matrix<T> {
            type Output = Matrix<T>;

            fn $op_fn(self, rhs: &T) -> Matrix<T> {
                self.map_refs(|x| x.clone().$op_fn(rhs.clone()))
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
