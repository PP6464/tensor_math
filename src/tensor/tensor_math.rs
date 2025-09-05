use crate::tensor::tensor::{dot_vectors, IndexProducts, Shape, Tensor};
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Rem, Sub};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TensorMathErrors {
    #[error("Shapes are not compatible")]
    ShapesIncompatible,
    #[error("Transposition permutation invalid")]
    TransposePermutationInvalid,
}

/// Implement an operation elementwise
/// Also allows you to implement operations with a `Tensor` and a single value
/// By applying the operation between it and each element of the `Tensor` in turn
macro_rules! impl_bin_op {
    ($op:ident, $op_fn:ident) => {
        impl<T: $op<Output = T> + Clone> $op<Tensor<T>> for Tensor<T> {
            type Output = Tensor<T>;

            fn $op_fn(self, rhs: Tensor<T>) -> Tensor<T> {
                assert_eq!(
                    self.shape(),
                    rhs.shape(),
                    "{}",
                    TensorMathErrors::ShapesIncompatible
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
                    TensorMathErrors::ShapesIncompatible
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
                    TensorMathErrors::ShapesIncompatible
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
                    TensorMathErrors::ShapesIncompatible
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
        let elements = self.elements().into_iter().cloned().map(|a| -a).collect();

        Tensor::new(self.shape(), elements).unwrap()
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
    pub fn new(permutation: &Vec<usize>) -> Result<Self, TensorMathErrors> {
        let mut perm_copy = permutation.to_vec();
        perm_copy.sort();

        if perm_copy != (0..permutation.len()).collect::<Vec<usize>>() {
            return Err(TensorMathErrors::TransposePermutationInvalid);
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

    /// Swap two axes, consuming the original and returning the new one
    pub fn swap_axes(self, axis1: usize, axis2: usize) -> Result<Self, TensorMathErrors> {
        if axis1 >= self.permutation.len() || axis2 >= self.permutation.len() {
            return Err(TensorMathErrors::TransposePermutationInvalid);
        }

        let mut new_perm = self.permutation;
        new_perm.swap(axis1, axis2);

        Transpose::new(&new_perm)
    }

    /// Swap two axes in-place
    pub fn swap_axes_in_place(&mut self, axis1: usize, axis2: usize) -> Result<(), TensorMathErrors> {
        if axis1 >= self.permutation.len() || axis2 >= self.permutation.len() {
            return Err(TensorMathErrors::TransposePermutationInvalid);
        }

        self.permutation.swap(axis1, axis2);

        Ok(())
    }

    pub fn new_shape(&self, old_shape: &Shape) -> Result<Shape, TensorMathErrors> {
        if old_shape.rank() != self.permutation.len() {
            return Err(TensorMathErrors::ShapesIncompatible);
        }

        let mut new_shape_vec = Vec::with_capacity(old_shape.rank());

        for old_pos in self.permutation.iter() {
            new_shape_vec.push(old_shape[*old_pos]);
        }

        Ok(Shape::new(new_shape_vec).unwrap())
    }

    pub fn new_index(&self, old_index: &[usize]) -> Result<Vec<usize>, TensorMathErrors> {
        if old_index.len() != self.permutation.len() {
            return Err(TensorMathErrors::ShapesIncompatible);
        }

        let mut new_index_vec = Vec::with_capacity(old_index.len());

        for old_pos in self.permutation.iter() {
            new_index_vec.push(old_index[*old_pos]);
        }

        Ok(new_index_vec)
    }
}
impl<T : Clone> Tensor<T> {
    /// Transposes a `Tensor`, consuming it and returning the new one
    pub fn transpose(self, transpose: &Transpose) -> Result<Tensor<T>, TensorMathErrors> {
        if transpose.permutation.len() != self.shape().rank() {
            return Err(TensorMathErrors::ShapesIncompatible);
        }

        let new_shape = transpose.new_shape(self.shape())?;
        let new_index_products = IndexProducts::from_shape(&new_shape);
        let mut new_elements = self.elements().clone();

        for i in 0..new_shape.element_count() {
            let mut old_index_vec = Vec::with_capacity(new_shape.rank());
            let mut remainder = i;

            for j in self.index_products.0.iter() {
                let floored_div = remainder / j;
                old_index_vec.push(floored_div);
                remainder = remainder % j;
            }

            let new_index = transpose.new_index(&old_index_vec)?;
            let addr_in_new_elements = dot_vectors(&new_index, &new_index_products.0);

            new_elements[addr_in_new_elements] = self.elements[i].clone();
        }

        Ok(Tensor::new(&new_shape, new_elements.to_vec()).unwrap())
    }

    /// Transpose a `Tensor` in-place
    pub fn transpose_in_place(&mut self, transpose: &Transpose) -> Result<(), TensorMathErrors> {
        let new_tensor = self.clone().transpose(&transpose)?;

        self.elements = new_tensor.elements;
        self.shape = new_tensor.shape;
        self.index_products = new_tensor.index_products;

        Ok(())
    }
}
