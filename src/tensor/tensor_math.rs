use crate::tensor::tensor::TensorErrors::DeterminantZero;
use crate::tensor::tensor::{dot_vectors, IntoMatrix, IntoTensor, Matrix, Shape, Strides, Tensor};
use crate::tensor::tensor::{tensor_index, TensorErrors};
use crate::shape;
use float_cmp::{approx_eq, ApproxEq, F64Margin};
use num::complex::{Complex64, ComplexFloat};
use num::{One, ToPrimitive, Zero};
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use std::cmp::min;
use std::f64::consts::PI;
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Range, Rem, Sub};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::scope;

/// Implement an operation elementwise for `Tensor` and `Matrix`
/// Also allows you to implement operations with a `Tensor`/`Matrix` and a single value
/// By applying the operation between it and each element of the `Tensor`/`Matrix` in turn
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
        self.transform_elementwise(|x| -x)
    }
}
impl<T: Neg<Output = T> + Clone> Neg for Matrix<T> {
    type Output = Matrix<T>;

    fn neg(self) -> Self::Output {
        self.transform_elementwise(|x| -x)
    }
}

/// Describes the parameters for a transpose
/// Specify the permutation vector as which axis
/// In the old axis goes to the current axis after
/// transposition. E.g.:
/// To transpose (a,b,c) to (c,a,b) the permutation would be [2, 0, 1\]
/// because we want axis 2 to be at axis 0, axis 0 to be at axis 1 etc.
pub struct Transpose {
    permutation: Vec<usize>,
}

impl Transpose {
    pub fn new(permutation: &Vec<usize>) -> Result<Self, TensorErrors> {
        let mut perm_copy = permutation.to_vec();
        perm_copy.sort();

        if perm_copy != (0..permutation.len()).collect::<Vec<usize>>() {
            return Err(TensorErrors::TransposePermutationInvalid);
        }

        Ok(Transpose {
            permutation: permutation.clone(),
        })
    }

    /// Gives an instance of `Transpose` that corresponds to a default permutation
    /// i.e. no axes are changed.
    pub fn default(n: usize) -> Self {
        Transpose::new((0..n).collect::<Vec<usize>>().as_ref()).unwrap()
    }

    /// Swap two axes
    pub fn swap_axes(&self, axis1: usize, axis2: usize) -> Result<Self, TensorErrors> {
        if axis1 >= self.permutation.len() || axis2 >= self.permutation.len() {
            return Err(TensorErrors::TransposePermutationInvalid);
        }

        let mut new_perm = self.permutation.clone();
        new_perm.swap(axis1, axis2);

        Transpose::new(&new_perm)
    }

    /// Swap two axes in-place
    pub fn swap_axes_in_place(&mut self, axis1: usize, axis2: usize) -> Result<(), TensorErrors> {
        if axis1 >= self.permutation.len() || axis2 >= self.permutation.len() {
            return Err(TensorErrors::TransposePermutationInvalid);
        }

        self.permutation.swap(axis1, axis2);

        Ok(())
    }

    pub fn new_shape(&self, old_shape: &Shape) -> Result<Shape, TensorErrors> {
        if old_shape.rank() != self.permutation.len() {
            return Err(TensorErrors::ShapesIncompatible);
        }

        let mut new_shape_vec = Vec::with_capacity(old_shape.rank());

        for old_pos in self.permutation.iter() {
            new_shape_vec.push(old_shape[*old_pos]);
        }

        Ok(Shape::new(new_shape_vec).unwrap())
    }

    pub fn new_index(&self, old_index: &[usize]) -> Result<Vec<usize>, TensorErrors> {
        if old_index.len() != self.permutation.len() {
            return Err(TensorErrors::ShapesIncompatible);
        }

        let mut new_index_vec = Vec::with_capacity(old_index.len());

        for old_pos in self.permutation.iter() {
            new_index_vec.push(old_index[*old_pos]);
        }

        Ok(new_index_vec)
    }
}

#[macro_export]
/// Creates an instance of `Transpose` using the specified permutation.
/// Assumes the permutation is valid so will panic if it is not.
macro_rules! transpose {
    ($($x:expr),*$(,)?) => {
        Transpose::new(&vec![$($x),*]).unwrap()
    };
}

impl<T: Clone> Tensor<T> {
    /// Transposes a `Tensor`
    pub fn transpose(&self, transpose: &Transpose) -> Result<Tensor<T>, TensorErrors> {
        if transpose.permutation.len() != self.shape().rank() {
            return Err(TensorErrors::ShapesIncompatible);
        }

        let new_shape = transpose.new_shape(self.shape())?;
        let new_strides = Strides::from_shape(&new_shape);
        let mut new_elements = self.elements().clone();

        for (old_index, elem) in self.enumerated_iter() {
            let new_index = transpose.new_index(&old_index)?;
            let new_addr = dot_vectors(&new_index, &new_strides.0);

            new_elements[new_addr] = elem;
        }

        Ok(Tensor::new(&new_shape, new_elements.to_vec())?)
    }

    /// Transpose a `Tensor` in-place
    pub fn transpose_in_place(&mut self, transpose: &Transpose) -> Result<(), TensorErrors> {
        let new_tensor = self.clone().transpose(&transpose)?;

        self.elements = new_tensor.elements;
        self.shape = new_tensor.shape;
        self.strides = new_tensor.strides;

        Ok(())
    }
}
impl<T: Clone> Matrix<T> {
    /// Transpose a `Matrix`
    pub fn transpose(&self) -> Matrix<T> {
        self.tensor.transpose(&transpose![1, 0]).unwrap().try_into().unwrap()
    }

    /// Transpose a `Matrix` in-place
    pub fn transpose_in_place(&mut self) {
        self.tensor = self.tensor.transpose(&transpose![1, 0]).unwrap();
    }
}
impl<T: Default + Clone> Tensor<T> {
    /// Pools a `Tensor<T>` into a `Tensor<O>` using a custom pooling function.
    /// The custom function will take a `Tensor<T>` that corresponds to the slice that the kernel covers.
    /// If the kernel is hanging over the edge of the tensor, then only the bits of the tensor that fit are included.
    /// This is reflected in the shape of the input tensor.
    /// Default functions for `max` and `avg` are given as well.
    pub fn pool<O: Default + Clone>(&self, pool_fn: impl Fn(Tensor<T>) -> O, kernel_shape: &Shape, stride_shape: &Shape) -> Tensor<O> {
        assert_eq!(
            kernel_shape.rank(),
            self.rank(),
            "Kernel shape rank and tensor shape rank are not the same"
        );
        assert_eq!(
            stride_shape.rank(),
            self.rank(),
            "Stride shape rank and tensor shape rank are not the same"
        );

        let res_shape = &Shape::new(
            self.shape()
                .0
                .iter()
                .cloned()
                .zip(stride_shape.0.iter().cloned())
                .map(|(x, y)| x.div_ceil(y))
                .collect::<Vec<usize>>(),
        )
            .unwrap();

        let mut result = Tensor::<O>::from_shape(res_shape);

        for (pos, val) in result.enumerated_iter_mut() {
            let start_pos = pos
                .iter()
                .zip(stride_shape.0.iter())
                .map(|(x, y)| x * y)
                .collect::<Vec<usize>>();
            let end_pos = start_pos
                .iter()
                .zip(kernel_shape.0.iter())
                .enumerate()
                .map(|(i, (x, y))| {
                    let shape_val = self.shape[i];

                    if x + y < shape_val {
                        x + y
                    } else {
                        shape_val
                    }
                })
                .collect::<Vec<usize>>();

            let indices = end_pos
                .iter()
                .zip(start_pos.iter())
                .map(|(x, y)| *y..*x)
                .collect::<Vec<_>>();

            *val = pool_fn(self.slice(indices.as_slice()));
        }

        result
    }
}
impl<T: Default + Clone> Matrix<T> {
    /// Pools a `Matrix<T>` into a `Matrix<O>` using a custom pooling function.
    /// The custom function will take a `Matrix<T>` that corresponds to the slice that the kernel covers.
    /// If the kernel is hanging over the edge of the tensor, then only the bits of the tensor that fit are included.
    /// This is reflected in the shape of the input tensor.
    /// Default functions for `max` and `avg` are given as well.
    pub fn pool<O: Default + Clone>(&self, pool_fn: impl Fn(Matrix<T>) -> O, kernel_shape: (usize, usize), stride_shape: (usize, usize)) -> Matrix<O> {
        assert!(kernel_shape.0 != 0 || kernel_shape.1 != 0, "Invalid kernel shape");
        assert!(stride_shape.0 != 0 || stride_shape.1 != 0, "Invalid stride shape");

        let res_shape = (self.rows.div_ceil(stride_shape.0), self.cols.div_ceil(stride_shape.1));
        let mut res = Matrix::<O>::from_shape(res_shape.0, res_shape.1);

        for (pos, val) in res.enumerated_iter_mut() {
            let start_pos = (pos.0 * stride_shape.0, pos.1 * stride_shape.1);
            let end_pos = (min(start_pos.0 + stride_shape.0, self.rows), min(start_pos.1 + stride_shape.1, self.cols));

            let indices = (start_pos.0..end_pos.0, start_pos.1..end_pos.1);

            *val = pool_fn(self.slice(indices.0, indices.1));
        }

        res
    }
}
impl<T: Add<Output = T> + Clone> Tensor<T> {
    /// Compute the sum of the tensor
    pub fn sum(&self) -> T {
        self.iter().cloned().reduce(T::add).unwrap()
    }
}
impl<T: Add<Output = T> + Clone> Matrix<T> {
    /// Compute the sum of a `Matrix`
    pub fn sum(&self) -> T {
        self.tensor.sum()
    }
}
impl<T: PartialOrd + Clone> Tensor<T> {
    /// Bounds the values between `min` and `max`
    /// consuming the original and returning the result
    pub fn clip(&self, min: T, max: T) -> Tensor<T> {
        let shape = self.shape();
        Tensor::new(
            shape,
            self.iter()
                .cloned()
                .map(|x| {
                    if x < min {
                        min.clone()
                    } else if x > max {
                        max.clone()
                    } else {
                        x
                    }
                })
                .collect(),
        )
        .unwrap()
    }
}
impl<T: PartialOrd + Clone> Matrix<T> {
    /// Clips the values in the `Matrix` between [min, max\]
    pub fn clip(&self, min: T, max: T) -> Matrix<T> {
        let mut res = self.clone();

        for val in res.iter_mut() {
            if *val <= min {
                *val = min.clone();
            } else if *val >= max {
                *val = max.clone();
            }
        }

        res
    }
}
impl<T: Clone + Mul<Output = T>> Tensor<T> {
    /// Implements the Kronecker product for any two things that can be converted to tensors.
    /// The Kronecker product scales the second tensor by each element of the first tensor,
    /// giving a `Tensor` of type `Tensor<T>`. This is then simplified into just `Tensor<T>`
    /// with the result having a shape that is the element-wise product of the two input
    /// tensors' shapes (if one has a lower rank than the other, then the rest of the larger rank
    /// tensor's shape values are inserted afterward).
    pub fn kronecker(&self, other: &Tensor<T>) -> Tensor<T> {
        let mut new_shape_vec = Vec::new();

        if self.rank() > other.rank() {
            for i in 0..other.rank() {
                new_shape_vec.push(self.shape[i] * other.shape[i]);
            }

            for i in other.rank()..self.rank() {
                new_shape_vec.push(self.shape[i]);
            }
        } else {
            for i in 0..self.rank() {
                new_shape_vec.push(self.shape[i] * other.shape[i]);
            }

            for i in self.rank()..other.rank() {
                new_shape_vec.push(other.shape[i]);
            }
        }

        let new_shape = Shape::new(new_shape_vec).unwrap();
        let mut new_elements = Vec::with_capacity(new_shape.element_count());

        for i in self.elements.iter().cloned() {
            let tensor = other * i;
            new_elements.extend(tensor.elements);
        }

        new_elements.into_tensor().reshape(&new_shape).unwrap()
    }
}
impl<T: Clone + Mul<Output = T>> Matrix<T> {
    pub fn kronecker(&self, other: &Matrix<T>) -> Matrix<T> {
        let res_tensor = self.tensor.kronecker(&other.tensor);
        Matrix {
            rows: res_tensor.shape[0],
            cols: res_tensor.shape[1],
            tensor: res_tensor,
        }
    }
}
impl<T: Clone + Mul<Output = T> + Send + Sync> Tensor<T> {
    /// Computes the Kronecker product using multiple threads
    pub fn kronecker_mt(&self, other: &Tensor<T>) -> Tensor<T> {
        let mut new_shape_vec = Vec::new();

        if self.rank() > other.rank() {
            for i in 0..other.rank() {
                new_shape_vec.push(self.shape[i] * other.shape[i]);
            }

            for i in other.rank()..self.rank() {
                new_shape_vec.push(self.shape[i]);
            }
        } else {
            for i in 0..self.rank() {
                new_shape_vec.push(self.shape[i] * other.shape[i]);
            }

            for i in self.rank()..other.rank() {
                new_shape_vec.push(other.shape[i]);
            }
        }

        let new_shape = Shape::new(new_shape_vec).unwrap();
        let mut new_elements = (0..new_shape.element_count()).map(|_| self.first().unwrap().clone()).collect::<Vec<T>>();

        scope(|s| {
            let mut new_elements_chunks = new_elements.chunks_mut(other.shape.element_count());

            for elem in self.iter() {
                let new_elements_chunk = new_elements_chunks.next().unwrap();

                s.spawn(move || {
                    let res = other * elem;

                    for (j, e) in res.iter().enumerate() {
                        new_elements_chunk[j] = e.clone();
                    }
                });
            }
        });

        new_elements.into_tensor().reshape(&new_shape).unwrap()
    }
}
impl<T: Clone + Mul<Output = T> + Send + Sync> Matrix<T> {
    /// Computes the Kronecker product using multiple threads
    pub fn kronecker_mt(&self, other: &Matrix<T>) -> Matrix<T> {
        let mut new_shape_vec = Vec::new();

        if self.rank() > other.rank() {
            for i in 0..other.rank() {
                new_shape_vec.push(self.shape[i] * other.shape[i]);
            }

            for i in other.rank()..self.rank() {
                new_shape_vec.push(self.shape[i]);
            }
        } else {
            for i in 0..self.rank() {
                new_shape_vec.push(self.shape[i] * other.shape[i]);
            }

            for i in self.rank()..other.rank() {
                new_shape_vec.push(other.shape[i]);
            }
        }

        let new_shape = Shape::new(new_shape_vec).unwrap();
        let mut new_elements = (0..new_shape.element_count()).map(|_| self.first().unwrap().clone()).collect::<Vec<T>>();

        scope(|s| {
            let mut new_elements_chunks = new_elements.chunks_mut(other.shape.element_count());

            for elem in self.iter() {
                let new_elements_chunk = new_elements_chunks.next().unwrap();

                s.spawn(move || {
                    let res = other * elem;

                    for (j, e) in res.iter().enumerate() {
                        new_elements_chunk[j] = e.clone();
                    }
                });
            }
        });

        new_elements.into_matrix().reshape(new_shape[0], new_shape[1]).unwrap()
    }
}
impl<T: Clone + Add<Output = T> + Mul<Output = T>> Tensor<T> {
    /// Perform tensor-contraction multiplication, which is a more general form of matrix multiplication
    /// A `Tensor` of shape (a,b,c) multiplied in this way by a `Tensor` of shape (c, d, e, f)
    /// will produce a resultant `Tensor` of shape (a, b, d, e, f) by the following formula:
    /// result(a, b, d, e, f) = sum(x=0, x=c) { first(a, b, x) * second(x, d, e, f) }
    pub fn contract_mul(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorErrors> {
        if self.shape.0.last().unwrap() != other.shape.0.first().unwrap() {
            return Err(TensorErrors::ShapesIncompatible);
        }

        let mut resultant_shape_vec = self
            .shape
            .0
            .iter()
            .take(self.rank() - 1)
            .cloned()
            .collect::<Vec<usize>>();

        resultant_shape_vec.extend(
            other
                .shape
                .0
                .iter()
                .rev()
                .take(other.rank() - 1)
                .rev()
                .cloned(),
        );
        let resultant_shape = Shape::new(resultant_shape_vec).unwrap();
        let mut resultant_elements: Vec<T> = Vec::with_capacity(resultant_shape.element_count());

        for i in 0..resultant_shape.element_count() {
            let index = tensor_index(i, &resultant_shape);
            let (self_chunk, other_chunk) = index.split_at(self.rank() - 1);
            let mut self_elements: Vec<T> = Vec::with_capacity(*self.shape.0.last().unwrap());
            let mut other_elements: Vec<T> = Vec::with_capacity(*other.shape.0.first().unwrap());

            for j in 0..*self.shape.0.last().unwrap() {
                let mut self_index = self_chunk.to_vec();
                self_index.push(j);

                self_elements.push(self[&self_index].clone());

                let mut other_index = other_chunk.to_vec();
                other_index.insert(0, j);
                other_elements.push(other[&other_index].clone());
            }

            resultant_elements.push(dot_vectors(&self_elements, &other_elements));
        }

        Ok(resultant_elements.into_tensor().reshape(&resultant_shape)?)
    }

    /// Computes the dot product of two tensors, i.e. the element-wise product, then the sum of the result
    pub fn dot(&self, other: &Tensor<T>) -> T {
        (self * other).sum()
    }
}
impl<T: Clone + Add<Output = T> + Mul<Output = T>> Matrix<T> {
    /// Does matrix multiplication with another `Matrix`
    pub fn contract_mul(&self, other: &Matrix<T>) -> Result<Matrix<T>, TensorErrors> {
        self.tensor.contract_mul(&other.tensor)?.try_into()
    }

    /// Does matrix multiplication with another matrix. This is just an alternate name for the method
    pub fn mat_mul(&self, other: &Matrix<T>) -> Result<Matrix<T>, TensorErrors> {
        self.contract_mul(other)
    }
}
impl<T: Clone + Add<Output = T> + Mul<Output = T> + Send + Sync> Tensor<T> {
    /// Does tensor contraction multiplication on multiple threads
    pub fn contract_mul_mt(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorErrors> {
        if self.shape.0.last().unwrap() != other.shape.0.first().unwrap() {
            return Err(TensorErrors::ShapesIncompatible);
        }

        let mut resultant_shape_vec = self
            .shape
            .0
            .iter()
            .take(self.rank() - 1)
            .cloned()
            .collect::<Vec<usize>>();

        resultant_shape_vec.extend(
            other
                .shape
                .0
                .iter()
                .rev()
                .take(other.rank() - 1)
                .rev()
                .cloned(),
        );
        let res_shape = Shape::new(resultant_shape_vec).unwrap();

        let res_mutexes = Arc::new(
            RwLock::new(
                Tensor::new(
                    &res_shape,
                    (0..res_shape.element_count())
                        .map(|_| Mutex::new(self.first().unwrap().clone()))
                        .collect(),
                )?,
            ),
        );

        let self_arc = Arc::new(self);
        let res_shape_arc = Arc::new(res_shape);
        let elems_per_thread = 10;  // The number of elements each thread is responsible for

        scope(|s| {
            for i in 0..res_shape_arc.clone().element_count() / elems_per_thread {
                let res_shape_arc_clone = res_shape_arc.clone();
                let res_arc_clone = res_mutexes.clone();

                let self_arc_clone = self_arc.clone();

                let start = i * elems_per_thread;
                let end = min((i + 1) * elems_per_thread, res_shape_arc.element_count());


                s.spawn(move || {
                    let res_read = res_arc_clone.read().unwrap();

                    for j in start..end {
                        let t_index = tensor_index(j, &res_shape_arc_clone);

                        let (self_part, other_part) = t_index.split_at(self_arc_clone.rank() - 1);

                        let mut self_indices = self_part.iter().map(|&x| x..x+1).collect::<Vec<Range<usize>>>();
                        self_indices.push(0..self_arc_clone.shape.0.last().unwrap().clone());

                        let mut other_indices = other_part.iter().map(|&x| x..x+1).collect::<Vec<Range<usize>>>();
                        other_indices.insert(0, 0..other.shape.0.first().unwrap().clone());

                        let self_elems = self_arc_clone.slice(&self_indices).reshape(&shape![self_arc_clone.shape.0.last().unwrap().clone()]).unwrap();
                        let other_elems = other.slice(&other_indices).reshape(&shape![other.shape.0.first().unwrap().clone()]).unwrap();

                        let elem_res = self_elems.dot(&other_elems);

                        let mut write_lock = res_read[&t_index].lock().unwrap();

                        *write_lock = elem_res;
                    }
                });
            }
        });

        let res_read = res_mutexes.read().unwrap();
        Ok(res_read.iter().map(|x| x.lock().unwrap().clone()).collect::<Tensor<T>>().reshape(&res_shape_arc.clone())?)
    }
}
impl<T: Clone + Add<Output = T> + Mul<Output = T> + Send + Sync> Matrix<T> {
    /// Does tensor contraction multiplication on multiple threads
    pub fn contract_mul_mt(&self, other: &Matrix<T>) -> Result<Matrix<T>, TensorErrors> {
        self.tensor.contract_mul_mt(&other.tensor)?.try_into()
    }

    /// Does tensor contraction multiplication on multiple threads. This is just an alternate name for the method
    pub fn mat_mul_mt(&self, other: &Matrix<T>) -> Result<Matrix<T>, TensorErrors> {
        self.contract_mul_mt(other)
    }
}

// Define a bunch of convenience mathematical functions for Tensor<f64> and Matrix<f64>
// They will consume the original tensor and return the result
impl Tensor<f64> {
    /// Exponentiates each element in the tensor
    pub fn exp(self) -> Tensor<f64> {
        self.transform_elementwise(f64::exp)
    }

    /// Raises n to the power of each element.
    /// This method uses `f64::powf` so beware of `f64::NaN` values
    /// if you have a negative value raised to the power of 0.5 for example
    pub fn exp_base_n(self, n: f64) -> Tensor<f64> {
        self.transform_elementwise(|x| f64::powf(n, x))
    }

    /// Raises each element to the power of n
    /// Like `exp_base_n` this can give `NaN` values if you aren't careful
    /// with what you are raising to which power
    pub fn pow(self, n: f64) -> Tensor<f64> {
        self.transform_elementwise(|x| f64::powf(x, n))
    }

    /// Computes the sin of each element
    pub fn sin(self) -> Tensor<f64> {
        self.transform_elementwise(f64::sin)
    }

    /// Computes the cos of each element
    pub fn cos(self) -> Tensor<f64> {
        self.transform_elementwise(f64::cos)
    }

    /// Computes the tan of each element
    pub fn tan(self) -> Tensor<f64> {
        self.transform_elementwise(f64::tan)
    }

    /// Computes the arcsin of each element
    pub fn asin(self) -> Tensor<f64> {
        self.transform_elementwise(f64::asin)
    }

    /// Computes the arccos of each element
    pub fn acos(self) -> Tensor<f64> {
        self.transform_elementwise(f64::acos)
    }

    /// Computes the atan of each element
    pub fn atan(self) -> Tensor<f64> {
        self.transform_elementwise(f64::atan)
    }

    /// Computes the reciprocal of each element.
    /// Use in conjunction with trigonometric functions to get sec(x), coth(x) etc.
    pub fn recip(self) -> Tensor<f64> {
        self.transform_elementwise(f64::recip)
    }

    /// Computes the sinh of each element
    pub fn sinh(self) -> Tensor<f64> {
        self.transform_elementwise(f64::sinh)
    }

    /// Computes the cosh of each element
    pub fn cosh(self) -> Tensor<f64> {
        self.transform_elementwise(f64::cosh)
    }

    /// Computes the tanh of each element
    pub fn tanh(self) -> Tensor<f64> {
        self.transform_elementwise(f64::tanh)
    }

    /// Computes the arsinh of each element
    pub fn asinh(self) -> Tensor<f64> {
        self.transform_elementwise(f64::asinh)
    }

    /// Computes the arcosh of each element
    pub fn acosh(self) -> Tensor<f64> {
        self.transform_elementwise(f64::acosh)
    }

    /// Computes the artanh of each element
    pub fn atanh(self) -> Tensor<f64> {
        self.transform_elementwise(f64::atanh)
    }

    /// Computes the sigmoid function for each element.
    /// Sigmoid(x) = 1 / (1 + exp(-x))
    pub fn sigmoid(self) -> Tensor<f64> {
        ((-self).exp() + 1.0).recip()
    }

    /// Computes the ReLU function for each element
    pub fn relu(self) -> Tensor<f64> {
        self.transform_elementwise(|x| if x > 0.0 { x } else { 0.0 })
    }

    /// Computes the leaky ReLU function for each element
    pub fn leaky_relu(self, alpha: f64) -> Tensor<f64> {
        self.transform_elementwise(|x| if x > 0.0 { x } else { alpha * x })
    }

    /// Applies softmax to the tensor
    pub fn softmax(self) -> Tensor<f64> {
        let new = self.exp();
        let sum = new.sum();
        new / sum
    }

    /// Normalises the tensor so the sum of magnitudes is 1
    pub fn norm_l1(self) -> Tensor<f64> {
        let mag = self.clone().transform_elementwise(|x| x.abs()).sum();
        self / mag
    }

    /// Normalises the tensor so the sum of the squares of the magnitudes is 1
    pub fn norm_l2(self) -> Tensor<f64> {
        let mag = self.clone().transform_elementwise(|x| x * x).sum().sqrt();
        self / mag
    }
}
impl Matrix<f64> {
    /// Exponentiates each element in the tensor
    pub fn exp(self) -> Matrix<f64> {
        self.transform_elementwise(f64::exp)
    }

    /// Raises n to the power of each element.
    /// This method uses `f64::powf` so beware of `f64::NaN` values
    /// if you have a negative value raised to the power of 0.5 for example
    pub fn exp_base_n(self, n: f64) -> Matrix<f64> {
        self.transform_elementwise(|x| f64::powf(n, x))
    }

    /// Raises each element to the power of n
    /// Like `exp_base_n` this can give `NaN` values if you aren't careful
    /// with what you are raising to which power
    pub fn pow(self, n: f64) -> Matrix<f64> {
        self.transform_elementwise(|x| f64::powf(x, n))
    }

    /// Computes the sin of each element
    pub fn sin(self) -> Matrix<f64> {
        self.transform_elementwise(f64::sin)
    }

    /// Computes the cos of each element
    pub fn cos(self) -> Matrix<f64> {
        self.transform_elementwise(f64::cos)
    }

    /// Computes the tan of each element
    pub fn tan(self) -> Matrix<f64> {
        self.transform_elementwise(f64::tan)
    }

    /// Computes the arcsin of each element
    pub fn asin(self) -> Matrix<f64> {
        self.transform_elementwise(f64::asin)
    }

    /// Computes the arccos of each element
    pub fn acos(self) -> Matrix<f64> {
        self.transform_elementwise(f64::acos)
    }

    /// Computes the atan of each element
    pub fn atan(self) -> Matrix<f64> {
        self.transform_elementwise(f64::atan)
    }

    /// Computes the reciprocal of each element.
    /// Use in conjunction with trigonometric functions to get sec(x), coth(x) etc.
    pub fn recip(self) -> Matrix<f64> {
        self.transform_elementwise(f64::recip)
    }

    /// Computes the sinh of each element
    pub fn sinh(self) -> Matrix<f64> {
        self.transform_elementwise(f64::sinh)
    }

    /// Computes the cosh of each element
    pub fn cosh(self) -> Matrix<f64> {
        self.transform_elementwise(f64::cosh)
    }

    /// Computes the tanh of each element
    pub fn tanh(self) -> Matrix<f64> {
        self.transform_elementwise(f64::tanh)
    }

    /// Computes the arsinh of each element
    pub fn asinh(self) -> Matrix<f64> {
        self.transform_elementwise(f64::asinh)
    }

    /// Computes the arcosh of each element
    pub fn acosh(self) -> Matrix<f64> {
        self.transform_elementwise(f64::acosh)
    }

    /// Computes the artanh of each element
    pub fn atanh(self) -> Matrix<f64> {
        self.transform_elementwise(f64::atanh)
    }

    /// Computes the sigmoid function for each element.
    /// Sigmoid(x) = 1 / (1 + exp(-x))
    pub fn sigmoid(self) -> Matrix<f64> {
        ((-self).exp() + 1.0).recip()
    }

    /// Computes the ReLU function for each element
    pub fn relu(self) -> Matrix<f64> {
        self.transform_elementwise(|x| if x > 0.0 { x } else { 0.0 })
    }

    /// Computes the leaky ReLU function for each element
    pub fn leaky_relu(self, alpha: f64) -> Matrix<f64> {
        self.transform_elementwise(|x| if x > 0.0 { x } else { alpha * x })
    }

    /// Applies softmax to the tensor
    pub fn softmax(self) -> Matrix<f64> {
        let new = self.exp();
        let sum = new.sum();
        new / sum
    }

    /// Normalises the tensor so the sum of magnitudes is 1
    pub fn norm_l1(self) -> Matrix<f64> {
        let mag = self.clone().transform_elementwise(|x| x.abs()).sum();
        self / mag
    }

    /// Normalises the tensor so the sum of the squares of the magnitudes is 1
    pub fn norm_l2(self) -> Matrix<f64> {
        let mag = self.clone().transform_elementwise(|x| x * x).sum().sqrt();
        self / mag
    }
}

impl ApproxEq for Tensor<f64> {
    type Margin = F64Margin;

    fn approx_eq<M: Into<Self::Margin>>(self, other: Self, margin: M) -> bool {
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

// Define a bunch of convenience mathematical functions for Tensor<Complex64> and Matrix<f64>
// They will consume the original tensor and return the result
impl Tensor<Complex64> {
    /// Computes the exponential of each element
    pub fn exp(self) -> Tensor<Complex64> {
        self.transform_elementwise(Complex64::exp)
    }

    /// Raises n to the power of each element
    pub fn exp_base_n(self, n: Complex64) -> Tensor<Complex64> {
        self.transform_elementwise(|x| n.powc(x))
    }

    /// Raises each element to the power of n
    pub fn pow(self, n: Complex64) -> Tensor<Complex64> {
        self.transform_elementwise(|x| x.powc(n))
    }

    /// Computes the sin of each element
    pub fn sin(self) -> Tensor<Complex64> {
        self.transform_elementwise(Complex64::sin)
    }

    /// Computes the cos of each element
    pub fn cos(self) -> Tensor<Complex64> {
        self.transform_elementwise(Complex64::cos)
    }

    /// Computes the tan of each element
    pub fn tan(self) -> Tensor<Complex64> {
        self.transform_elementwise(Complex64::tan)
    }

    /// Computes the arcsin of each element
    pub fn asin(self) -> Tensor<Complex64> {
        self.transform_elementwise(Complex64::asin)
    }

    /// Computes the arccos of each element
    pub fn acos(self) -> Tensor<Complex64> {
        self.transform_elementwise(Complex64::acos)
    }

    /// Computes the arctan of each element
    pub fn atan(self) -> Tensor<Complex64> {
        self.transform_elementwise(Complex64::atan)
    }

    /// Computes the sinh of each element
    pub fn sinh(self) -> Tensor<Complex64> {
        self.transform_elementwise(Complex64::sinh)
    }

    /// Computes the cosh of each element
    pub fn cosh(self) -> Tensor<Complex64> {
        self.transform_elementwise(Complex64::cosh)
    }

    /// Computes the tanh of each element
    pub fn tanh(self) -> Tensor<Complex64> {
        self.transform_elementwise(Complex64::tanh)
    }

    /// Computes the arsinh of each element
    pub fn asinh(self) -> Tensor<Complex64> {
        self.transform_elementwise(Complex64::asinh)
    }

    /// Computes the arcosh of each element
    pub fn acosh(self) -> Tensor<Complex64> {
        self.transform_elementwise(Complex64::acosh)
    }

    /// Computes the artanh of each element
    pub fn atanh(self) -> Tensor<Complex64> {
        self.transform_elementwise(Complex64::atanh)
    }

    /// Computes the reciprocal of each element.
    /// Use in conjunction with trigonometric functions to get sec(x), coth(x) etc.
    pub fn recip(self) -> Tensor<Complex64> {
        self.transform_elementwise(Complex64::recip)
    }

    /// Applies softmax to the tensor
    pub fn softmax(self) -> Tensor<Complex64> {
        let new = self.exp();
        let sum = new.sum();
        new / sum
    }

    /// Normalises the tensor so the sum of magnitudes is 1
    pub fn norm_l1(self) -> Tensor<Complex64> {
        let mag: Complex64 = self.clone().transform_elementwise(|x| x.abs()).sum().into();
        self / mag
    }

    /// Normalises the tensor so the sum of the squares of the magnitudes is 1
    pub fn norm_l2(self) -> Tensor<Complex64> {
        let mag: Complex64 = self
            .clone()
            .transform_elementwise(|x| (x * x).abs())
            .sum()
            .sqrt()
            .into();
        self / mag
    }
}
impl Matrix<Complex64> {
    /// Computes the exponential of each element
    pub fn exp(self) -> Matrix<Complex64> {
        self.transform_elementwise(Complex64::exp)
    }

    /// Raises n to the power of each element
    pub fn exp_base_n(self, n: Complex64) -> Matrix<Complex64> {
        self.transform_elementwise(|x| n.powc(x))
    }

    /// Raises each element to the power of n
    pub fn pow(self, n: Complex64) -> Matrix<Complex64> {
        self.transform_elementwise(|x| x.powc(n))
    }

    /// Computes the sin of each element
    pub fn sin(self) -> Matrix<Complex64> {
        self.transform_elementwise(Complex64::sin)
    }

    /// Computes the cos of each element
    pub fn cos(self) -> Matrix<Complex64> {
        self.transform_elementwise(Complex64::cos)
    }

    /// Computes the tan of each element
    pub fn tan(self) -> Matrix<Complex64> {
        self.transform_elementwise(Complex64::tan)
    }

    /// Computes the arcsin of each element
    pub fn asin(self) -> Matrix<Complex64> {
        self.transform_elementwise(Complex64::asin)
    }

    /// Computes the arccos of each element
    pub fn acos(self) -> Matrix<Complex64> {
        self.transform_elementwise(Complex64::acos)
    }

    /// Computes the arctan of each element
    pub fn atan(self) -> Matrix<Complex64> {
        self.transform_elementwise(Complex64::atan)
    }

    /// Computes the sinh of each element
    pub fn sinh(self) -> Matrix<Complex64> {
        self.transform_elementwise(Complex64::sinh)
    }

    /// Computes the cosh of each element
    pub fn cosh(self) -> Matrix<Complex64> {
        self.transform_elementwise(Complex64::cosh)
    }

    /// Computes the tanh of each element
    pub fn tanh(self) -> Matrix<Complex64> {
        self.transform_elementwise(Complex64::tanh)
    }

    /// Computes the arsinh of each element
    pub fn asinh(self) -> Matrix<Complex64> {
        self.transform_elementwise(Complex64::asinh)
    }

    /// Computes the arcosh of each element
    pub fn acosh(self) -> Matrix<Complex64> {
        self.transform_elementwise(Complex64::acosh)
    }

    /// Computes the artanh of each element
    pub fn atanh(self) -> Matrix<Complex64> {
        self.transform_elementwise(Complex64::atanh)
    }

    /// Computes the reciprocal of each element.
    /// Use in conjunction with trigonometric functions to get sec(x), coth(x) etc.
    pub fn recip(self) -> Matrix<Complex64> {
        self.transform_elementwise(Complex64::recip)
    }

    /// Applies softmax to the tensor
    pub fn softmax(self) -> Matrix<Complex64> {
        let new = self.exp();
        let sum = new.sum();
        new / sum
    }

    /// Normalises the tensor so the sum of magnitudes is 1
    pub fn norm_l1(self) -> Matrix<Complex64> {
        let mag: Complex64 = self.clone().transform_elementwise(|x| x.abs()).sum().into();
        self / mag
    }

    /// Normalises the tensor so the sum of the squares of the magnitudes is 1
    pub fn norm_l2(self) -> Matrix<Complex64> {
        let mag: Complex64 = self
            .clone()
            .transform_elementwise(|x| (x * x).abs())
            .sum()
            .sqrt()
            .into();
        self / mag
    }

    /// Computes the Householder transformation for the given matrix `t` (of shape (rows, cols)).
    /// Returns (Q, R), where Q is a unitary and Hermitian square matrix of shape (rows, rows)
    /// and R is an upper triangle matrix of shape (rows, cols) such that `Q.contract_mul(R) == t`.
    pub fn householder(&self) -> (Matrix<Complex64>, Matrix<Complex64>) {
        assert_eq!(self.shape().rank(), 2, "Only defined for matrices");

        let (rows, cols) = (self.shape()[0], self.shape()[1]);

        let mut r = self.clone();
        let mut q = identity::<Complex64>(rows);

        for k in 0..min(cols, rows) {
            let vec_bottom = r.slice(k..rows, k..k + 1);
            let alpha = -1.0
                * match vec_bottom[&[0, 0]] {
                Complex64::ZERO => Complex64::ONE,
                x => x / Complex64::abs(x),
            } * vec_bottom.iter().map(|x| <f64 as Into<Complex64>>::into((x * x).abs())).sum::<Complex64>().sqrt();
            let mut e1 = Matrix::<Complex64>::from_shape(vec_bottom.rows, vec_bottom.cols);
            e1[&[0, 0]] = Complex64::ONE;
            let v = vec_bottom - e1 * alpha;

            if v.iter().all(|x| approx_eq!(f64, x.abs(), 0.0)) {
                continue;
            }

            let u = v.norm_l2();
            let u_clone = u.clone();
            let u_t = u_clone.transpose();
            let u_star = u_t.iter().map(|x| x.conj()).collect::<Matrix<Complex64>>().reshape(u_t.rows, u_t.cols).unwrap();
            let h_sub = identity::<Complex64>(u.shape.0.first().unwrap().clone()) - u
                .contract_mul(&u_star)
                .unwrap()
                * Complex64 { re: 2.0, im: 0.0 };

            // Update R
            let r_slice_copy = r.slice(k..rows, k..cols);
            let mut r_slice_mut = r.slice_mut(k..rows, k..cols);
            let r_slice_res = h_sub.clone().contract_mul(&r_slice_copy).unwrap();

            for (pos, val) in r_slice_res.enumerated_iter() {
                r_slice_mut[pos] = val;
            }

            // Update Q
            let q_slice_copy = q.slice(0..rows, k..rows);
            let mut q_slice_mut = q.slice_mut(0..rows, k..rows);
            let q_slice_res = q_slice_copy.contract_mul(&h_sub).unwrap();

            for (pos, val) in q_slice_res.enumerated_iter() {
                q_slice_mut[pos] = val;
            }
        }

        (q, r)
    }
}

impl ApproxEq for Tensor<Complex64> {
    type Margin = F64Margin;

    fn approx_eq<M: Into<Self::Margin>>(self, other: Self, margin: M) -> bool {
        let margin = margin.into();
        
        self.enumerated_iter().all(|(i, x)| {
            approx_eq!(f64, x.re, other[&i].re, margin.clone()) && approx_eq!(f64, x.im, other[&i].im, margin.clone())
        })
    }
}

/// Default pooling function to sum the values
pub fn pool_sum<T: Add<Output = T> + Clone>(t: Tensor<T>) -> T {
    t.iter().cloned().reduce(T::add).unwrap()
}

/// Default pooling function to find the maximum value
pub fn pool_max<T: PartialOrd + Clone>(t: Tensor<T>) -> T {
    let mut max = t.first().unwrap().clone();

    for i in t.iter() {
        if *i > max {
            max = i.clone();
        }
    }

    max
}

/// Default pooling function to find the minimum value
pub fn pool_min<T: PartialOrd + Clone>(t: Tensor<T>) -> T {
    let mut min = t.first().unwrap().clone();

    for i in t.iter() {
        if *i < min {
            min = i.clone();
        }
    }

    min
}

/// Default pooling function to find the average.
/// Bear in mind the total number of elements is the total number of elements in the input,
/// so if you want the total number of elements to stay the same even for overhanging
///  input tensors then you will need to write your own version.
pub fn pool_avg<T: Add<Output = T> + Div<f64, Output = T> + Clone>(t: Tensor<T>) -> T {
    let elems = t.shape().element_count().to_f64().unwrap();
    let sum = pool_sum(t);
    sum / elems
}

/// Default pooling function to sum the values
pub fn pool_sum_mat<T: Add<Output = T> + Clone>(m: Matrix<T>) -> T {
    m.iter().cloned().reduce(T::add).unwrap()
}

/// Default pooling function to find the minimum
pub fn pool_min_mat<T: PartialOrd + Clone>(m: Matrix<T>) -> T {
    let mut min = m.first().unwrap().clone();

    for i in m.iter() {
        if *i < min {
            min = i.clone();
        }
    }

    min
}

/// Default pooling function to find the minimum
pub fn pool_max_mat<T: PartialOrd + Clone>(m: Matrix<T>) -> T {
    let mut max = m.first().unwrap().clone();

    for i in m.iter() {
        if *i > max {
            max = i.clone();
        }
    }

    max
}

/// Default pooling function to find the average.
/// Bear in mind the total number of elements is the total number of elements in the input,
/// so if you want the total number of elements to stay the same even for overhanging
///  input tensors then you will need to write your own version.
pub fn pool_avg_mat<T: Add<Output = T> + Div<f64, Output = T> + Clone>(m: Matrix<T>) -> T {
    let sum = pool_sum_mat(m.clone());
    let elems = m.shape().element_count().to_f64().unwrap();

    sum / elems
}

/// Solves a quadratic. The coefficients are entered as a `&[Complex64]` where the index of the
/// coefficient corresponds to the power of x, e.g. [1, 2, 3\] would be 1 + 2x + 3x².
pub fn solve_quadratic(coefficients: &[Complex64]) -> Vec<Complex64> {
    assert_eq!(coefficients.len(), 3, "Input must be a quadratic");

    let a = coefficients[2];
    let b = coefficients[1];
    let c = coefficients[0];

    let d = b.powi(2) - 4.0 * a * c;
    let mut roots = Vec::new();

    roots.push((-b + d.sqrt()) / (2.0 * a));
    roots.push((-b - d.sqrt()) / (2.0 * a));

    roots
}

/// Solves a cubic. The coefficients are entered as a `&[Complex64]` where the index of the
/// coefficient corresponds to the power of x, e.g. [1, 2, 3, 4\] would be 1 + 2x + 3x² + 4x³.
pub fn solve_cubic(coefficients: &[Complex64]) -> Vec<Complex64> {
    assert_eq!(coefficients.len(), 4, "Input must be a cubic");

    let a = coefficients[3];
    let b = coefficients[2];
    let c = coefficients[1];
    let d = coefficients[0];

    let q = (3.0 * a * c - b.powi(2)) / (9.0 * a.powi(2));
    let r = (9.0 * a * b * c - 27.0 * a.powi(2) * d - 2.0 * b.powi(3)) / (54.0 * a.powi(3));

    let d = r.powi(2) + q.powi(3);

    let s_cubed = r + d.sqrt();

    let omega = Complex64 { re: -0.5, im: f64::sqrt(3.0) / 2.0 };

    let s = s_cubed.cbrt();
    let t = (-q) / s;

    let offset = b / (3.0 * a);

    vec![
        s + t - offset,
        s * omega + t * omega.powi(2) - offset,
        s * omega.powi(2) + t * omega - offset,
    ]
}

/// Solves a quartic. The coefficients are entered as a `&[Complex64]` where the index of the
/// coefficient corresponds to the power of x, e.g. [1, 2, 3, 4, 5\] would be 1 + 2x + 3x² + 4x³ + 5x⁴.
pub fn solve_quartic(coefficients: &[Complex64]) -> Vec<Complex64> {
    assert_eq!(coefficients.len(), 5, "Input must be a quartic");

    let a = coefficients[4];
    let b = coefficients[3];
    let c = coefficients[2];
    let d = coefficients[1];
    let e = coefficients[0];

    let yi_squared = solve_cubic(&[
        -(-b.powi(3) + 4.0 * a * b * c - 8.0 * a.powi(2) * d).powi(2),
        3.0 * b.powi(4) + 16.0 * a.powi(2) * c.powi(2) + 16.0 * a.powi(2) * b * d - 16.0 * a * b.powi(2) * c - 64.0 * a.powi(3) * e,
        -(3.0 * b.powi(2) - 8.0 * a * c),
        1.0.into(),
    ]);

    let mut y0 = Complex64::ZERO;
    let mut y1 = Complex64::ZERO;
    let mut y2 = Complex64::ZERO;
    let mut found = false;

    for i in 0..=1 {
        for j in 0..=1 {
            for k in 0..=1 {
                y0 = yi_squared[0].sqrt() * (-1.0).powi(i);
                y1 = yi_squared[1].sqrt() * (-1.0).powi(j);
                y2 = yi_squared[2].sqrt() * (-1.0).powi(k);

                if (y0 * y1 * y2 - (-b.powi(3) + 4.0 * a * b * c - 8.0 * a.powi(2) * d)).abs() < 2e-10 {
                    found = true;
                    break
                }
            }
        }
    }

    assert!(found);

    vec![
        (-b + y0 + y1 + y2) / (4.0 * a),
        (-b + y0 - y1 - y2) / (4.0 * a),
        (-b - y0 + y1 - y2) / (4.0 * a),
        (-b - y0 - y1 + y2) / (4.0 * a),
    ]
}

// Matrix specific functions
/// Computes the trace of a matrix
pub fn trace<T: Add<Output = T> + Clone>(m: &Matrix<T>) -> T {
    assert!(m.is_square(), "Trace only defined for square matrices");

    let mut sum = m.elements.first().unwrap().clone();

    for i in 1..m.shape.0.iter().min().unwrap().clone() {
        sum = sum.add(m[&[i, i]].clone());
    }

    sum
}

/// Calculates the determinant for a matrix of values of type `T`.
pub fn det<T: Add<Output = T> + Mul<Output = T> + Sub<Output = T> + Clone + Zero>(
    m: &Matrix<T>,
) -> T {
    assert!(m.is_square(), "Determinant is only defined for square matrices");

    let ord = m.shape[0];

    if ord == 2 {
        return m[&[0, 0]].clone() * m[&[1, 1]].clone() - m[&[0, 1]].clone() * m[&[1, 0]].clone();
    }

    if ord == 1 {
        return m[&[0, 0]].clone();
    }

    let mut determinant = T::zero();

    for i in 0..ord {
        let is_minus = i % 2 != 0;

        if i == 0 {
            let slice = m.slice(1..ord, 1..ord);
            determinant = determinant + m[&[0, i]].clone() * det(&slice);

            continue;
        }

        if i == ord - 1 {
            let slice = m.slice(1..ord, 0..(ord - 1));

            if is_minus {
                determinant = determinant - m[&[0, i]].clone() * det(&slice);
            } else {
                determinant = determinant + m[&[0, i]].clone() * det(&slice);
            }

            continue;
        }

        let slice = m
            .slice(1..ord, 0..i)
            .concat(&m.slice(1..ord, i + 1..ord), 1)
            .unwrap();

        if is_minus {
            determinant = determinant - m[&[0, i]].clone() * det(&slice)
        } else {
            determinant = determinant + m[&[0, i]].clone() * det(&slice)
        }
    }

    determinant
}

/// Calculates the inverse for a matrix of values of type `T`.
/// If the determinant is 0 you will receive `TensorErrors::DeterminantZero`
pub fn inv<T>(m: &Matrix<T>) -> Result<Matrix<T>, TensorErrors>
where
    T: Add<Output = T>
        + Mul<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Neg<Output = T>
        + Clone
        + Zero
        + PartialEq,
{
    assert!(m.is_square(), "Inversion is only defined for square matrices");

    let ord = m.shape[0];
    let mut res = Matrix::<T>::from_value(m.rows, m.cols, T::zero());
    let d = det(&m);

    if d == T::zero() {
        return Err(DeterminantZero);
    }

    // Construct adjoint matrix

    // i is for which row we are on
    for i in 0..ord {
        // j is for which column we are on
        for j in 0..ord {
            let is_minus = (i + j) % 2 != 0;

            let slice = match (i, j) {
                (0, 0) => m.slice(1..ord, 1..ord),
                _ if (i, j) == (ord - 1, ord - 1) => m.slice(0..i, 0..j),
                _ if (i, j) == (0, ord - 1) => m.slice(1..ord, 0..j),
                _ if (i, j) == (ord - 1, 0) => m.slice(0..i, 1..ord),
                _ if i == 0 => m
                    .slice(1..ord, 0..j)
                    .concat(&m.slice(1..ord, j + 1..ord), 1)?,
                _ if i == ord - 1 => m
                    .slice(0..i, 0..j)
                    .concat(&m.slice(0..i, j + 1..ord), 1)?,
                _ if j == 0 => m
                    .slice(0..i, 1..ord)
                    .concat(&m.slice((i + 1)..ord, 1..ord), 0)?,
                _ if j == ord - 1 => m
                    .slice(0..i, 0..j)
                    .concat(&m.slice((i + 1)..ord, 0..j), 0)?,
                _ => {
                    let slice_top = m
                        .slice(0..i, 0..j)
                        .concat(&m.slice(0..i, (j + 1)..ord), 1)?;
                    let slice_bottom = m
                        .slice((i + 1)..ord, 0..j)
                        .concat(&m.slice((i + 1)..ord, (j + 1)..ord), 1)?;

                    slice_top.concat(&slice_bottom, 0)?
                }
            };

            res[&[j, i]] = if is_minus { -det(&slice) } else { det(&slice) };
        }
    }

    Ok(res / d)
}

/// Constructs an identity matrix of `T` values of the given size
pub fn identity<T: Zero + One + Clone>(n: usize) -> Matrix<T> {
    let mut t = Matrix::zeros(n, n);

    for i in 0..n {
        t[&[i, i]] = T::one();
    }

    t
}

/// Creates a tensor of values of the Gaussian pdf with a specified standard deviation.
/// The shape of the result is also specified by the user. The mean is the centre value,
/// but when tensors have an even number of elements in a certain axis, then the centre
/// is treated as being in between the two.
pub fn gaussian_pdf_single_sigma(sigma: f64, shape: &Shape) -> Tensor<f64> {
    assert!(sigma > 0.0, "Sigma must be positive");

    let mut res = Tensor::<f64>::from_shape(shape);
    let centre = shape
        .0
        .iter()
        .map(|x| (x - 1).to_f64().unwrap() / 2.0)
        .collect::<Vec<f64>>();

    for (pos, val) in res.enumerated_iter_mut() {
        let exponent = centre
            .iter()
            .zip(pos.iter())
            .map(|(x, y)| (x - y.to_f64().unwrap()).powi(2))
            .reduce(f64::add)
            .unwrap()
            * -1.0
            / (2.0 * sigma.powi(2));

        *val = f64::exp(exponent) / (sigma * f64::sqrt(2.0 * PI)).powi(shape.rank() as i32)
    }

    res
}

/// Creates a tensor of values of the Gaussian pdf with a specified list
/// of standard deviations, one for each axis of the tensor.
/// The shape of the result is also specified by the user. The mean is the centre value,
/// but when tensors have an even number of elements in a certain axis, then the centre
/// is treated as being in between the two.
pub fn gaussian_pdf_multi_sigma(sigma: Vec<f64>, shape: &Shape) -> Tensor<f64> {
    assert_eq!(
        sigma.len(),
        shape.rank(),
        "Sigma vector must have the same length as the rank of the tensor"
    );
    assert!(sigma.iter().all(|x| x > &0.0), "Sigma must be positive");

    let mut res = Tensor::<f64>::from_shape(shape);
    let centre = shape
        .0
        .iter()
        .map(|x| (x - 1).to_f64().unwrap() / 2.0)
        .collect::<Vec<f64>>();

    for (pos, val) in res.enumerated_iter_mut() {
        let mut exponent = 0.0;

        for (i, s) in sigma.iter().enumerate() {
            exponent -= (pos[i].to_f64().unwrap() - centre[i]).powi(2) / (2.0 * s * s)
        }

        let mut denominator = 1.0;

        for s in sigma.iter() {
            denominator *= s * f64::sqrt(2.0 * PI);
        }

        *val = f64::exp(exponent) / denominator;
    }

    res
}

/// Creates a tensor of values with the specified shape where the values are sampled from a
/// Gaussian distribution. The min and max values allow you to specify the range of the outputs.
/// The possible outputs will be 1001 different outputs spaced evenly across the interval [min, max\].
pub fn gaussian_sample(sigma: f64, shape: &Shape, min: f64, max: f64) -> Tensor<f64> {
    assert!(max > min, "Maximum must be greater than minimum");
    assert!(sigma > 0.0, "Sigma must be positive");

    let step_size = (max - min) / 1e3;

    let mut res = Tensor::<f64>::from_shape(shape);
    let dist = WeightedIndex::new(
        (0..=1000)
            .map(|x| f64::exp(-((x.to_f64().unwrap() - 500.0).powi(2)) / (2.0 * sigma.powi(2))))
            .collect::<Vec<f64>>(),
    )
    .unwrap();
    let mut rng = rand::rng();

    for val in res.iter_mut() {
        let index = dist.sample(&mut rng).to_f64().unwrap();
        *val = min + index * step_size;
    }

    res
}

// The following is WIP
// /// Creates a tensor of values of the Gaussian pdf with a specified covariance matrix.
// /// The shape of the result is also specified by the user. The mean is the centre value,
// /// but when tensors have an even number of elements in a certain axis, then the centre
// /// is treated as being in between the two.
// pub fn gaussian_pdf_mat_sigma(sigma: Tensor<f64>, shape: &Shape) -> Tensor<f64> {
//     assert_eq!(sigma.rank(), 2, "Sigma should be a matrix");
//     assert_eq!(sigma.shape[0], sigma.shape[1], "Sigma should be a square matrix");
//     assert!(sigma.iter().all(|x| x > &0.0), "All values in sigma must be positive");
//
//     let ord = sigma.shape[0];
//
//     assert_eq!(ord, shape.rank(), "Sigma matrix should have the same order as the rank of the result");
//
//     let mut res = Tensor::<f64>::from_shape(shape);
//     let sigma_inv = inv(&sigma).unwrap();
//
//     let centre = shape.0.iter().map(|x| { (x - 1).to_f64().unwrap() / 2.0 }).collect::<Vec<f64>>();
//
//     for (pos, val) in res.enumerated_iter_mut() {
//
//     }
//
//     res
// }
