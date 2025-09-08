use crate::tensor::tensor::{tensor_index, TensorErrors};
use crate::tensor::tensor::{dot_vectors, IndexProducts, Shape, Tensor};
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Rem, Sub};
use num::complex::{Complex64, ComplexFloat};
use crate::tensor::tensor::TensorErrors::DeterminantZero;
use crate::ts;

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

    /// Swap two axes, consuming the original and returning the new one
    pub fn swap_axes(self, axis1: usize, axis2: usize) -> Result<Self, TensorErrors> {
        if axis1 >= self.permutation.len() || axis2 >= self.permutation.len() {
            return Err(TensorErrors::TransposePermutationInvalid);
        }

        let mut new_perm = self.permutation;
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

impl<T: Clone> Tensor<T> {
    /// Transposes a `Tensor`, consuming it and returning the new one
    pub fn transpose(self, transpose: &Transpose) -> Result<Tensor<T>, TensorErrors> {
        if transpose.permutation.len() != self.shape().rank() {
            return Err(TensorErrors::ShapesIncompatible);
        }

        let new_shape = transpose.new_shape(self.shape())?;
        let new_index_products = IndexProducts::from_shape(&new_shape);
        let mut new_elements = self.elements().clone();

        for (old_index, elem) in self.enumerated_iter() {
            let new_index = transpose.new_index(&old_index)?;
            let new_addr = dot_vectors(&new_index, &new_index_products.0);

            new_elements[new_addr] = elem;
        }

        Ok(Tensor::new(&new_shape, new_elements.to_vec())?)
    }

    /// Transpose a `Tensor` in-place
    pub fn transpose_in_place(&mut self, transpose: &Transpose) -> Result<(), TensorErrors> {
        let new_tensor = self.clone().transpose(&transpose)?;

        self.elements = new_tensor.elements;
        self.shape = new_tensor.shape;
        self.index_products = new_tensor.index_products;

        Ok(())
    }
}
impl<T: Add<Output=T> + Clone> Tensor<T> {
    /// Compute the sum of the tensor
    pub fn sum(&self) -> T {
        self.iter().cloned().reduce(T::add).unwrap()
    }
}
impl<T: PartialOrd + Clone> Tensor<T> {
    /// Bounds the values between `min` and `max`
    /// consuming the original and returning the result
    pub fn clip(self, min: T, max: T) -> Tensor<T> {
        let shape = self.shape();
        Tensor::new(shape, self.iter().cloned().map(|x| if x < min { min.clone() } else if x > max { max.clone() } else { x }).collect()).unwrap()
    }
}
impl<T: Clone + Add<Output=T> + Mul<Output=T>> Tensor<T> {
    /// Perform tensor-contraction multiplication, which is a more general form of matrix multiplication
    /// A `Tensor` of shape (a,b,c) multiplied in this way by a `Tensor` of shape (c, d, e, f)
    /// will produce a resultant `Tensor` of shape (a, b, d, e, f) by the following formula:
    /// result(a, b, d, e, f) = sum(x=0, x=c) { first(a, b, x) * second(x, d, e, f) }
    /// This method consumes the original `Tensor` (but not the other one) and returns the result
    pub fn contract_mul(self, other: &Tensor<T>) -> Result<Tensor<T>, TensorErrors> {
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

        resultant_shape_vec
            .extend(
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
        let mut resultant_elements = Vec::with_capacity(resultant_shape.element_count());

        for i in 0..resultant_shape.element_count() {
            let index = tensor_index(i, &resultant_shape);
            let (self_chunk, other_chunk) = index.split_at(self.rank() - 1);
            let mut self_elements = Vec::with_capacity(*self.shape.0.last().unwrap());
            let mut other_elements = Vec::with_capacity(*other.shape.0.first().unwrap());

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

        Ok(Tensor::new(&resultant_shape, resultant_elements.to_vec())?)
    }
}

// More complex mathematical functions

/// Implements the Kronecker product for two tensors.
/// The Kronecker product scales the second tensor by each element of the first tensor,
/// giving a `Tensor` of type `Tensor<T>`. This is then simplified into just `Tensor<T>`
/// with the result having a shape that is the element-wise product of the two input
/// tensors' shapes (if one has a lower rank than the other, then the rest of the larger rank
/// tensor's shape values are inserted afterward).
/// Borrows both immutably and returns the result.
pub fn kronecker_product<T: Clone + Mul<Output=T>>(t1: &Tensor<T>, t2: &Tensor<T>) -> Tensor<T> {
    let mut new_shape_vec = Vec::new();

    if t1.rank() > t2.rank() {
        for i in 0..t2.rank() {
            new_shape_vec.push(t1.shape[i] * t2.shape[i]);
        }

        for i in t2.rank()..t1.rank() {
            new_shape_vec.push(t1.shape[i]);
        }
    } else {
        for i in 0..t1.rank() {
            new_shape_vec.push(t1.shape[i] * t2.shape[i]);
        }

        for i in t1.rank()..t2.rank() {
            new_shape_vec.push(t2.shape[i]);
        }
    }

    let new_shape = Shape::new(new_shape_vec).unwrap();
    let mut new_elements = Vec::with_capacity(new_shape.element_count());

    for i in t1.elements.iter().cloned() {
        let tensor = t2 * i;
        new_elements.extend(tensor.elements);
    }

    Tensor::new(&new_shape, new_elements).unwrap()
}

// Define a bunch of convenience mathematical functions for f64
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
}

// Define a bunch of convenience mathematical functions for Tensor<Complex64>
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
}

// Matrix specific functions
/// Computes the trace of a matrix
pub fn trace<T: Add<Output = T> + Clone>(t: &Tensor<T>) -> T {
    assert_eq!(t.rank(), 2, "This implementation of trace is only for matrices");
    assert_eq!(t.shape.0[0], t.shape.0[1], "Trace is only defined for square matrices");

    let mut sum = t.elements.first().unwrap().clone();

    for i in 1..t.shape.0.iter().min().unwrap().clone() {
        sum = sum.add(t[&[i, i]].clone());
    }

    sum
}

/// Calculates the determinant for a matrix of values of type `T`.
/// This assumes that `T::default()` returns a 0-value of type `T`,
/// which is the case for all common number types, and Complex64 as well.
pub fn det<T: Add<Output = T> + Mul<Output = T> + Sub<Output = T> + Clone + Default>(t: &Tensor<T>) -> T {
    assert_eq!(t.rank(), 2, "Determinant is only for matrices");
    assert_eq!(t.shape[0], t.shape[1], "Determinant is only defined for square matrices");

    let ord = t.shape[0];

    if ord == 2 {
        return t[&[0, 0]].clone() * t[&[1, 1]].clone() - t[&[0, 1]].clone() * t[&[1, 0]].clone();
    }

    if ord == 1 {
        return t[&[0, 0]].clone();
    }

    let mut determinant = T::default();

    for i in 0..ord {
        let is_minus = i % 2 != 0;

        let rest_of_shape = ts![ord - 1, ord - 1];
        let mut rest_of_tensor = Tensor::<T>::from_shape(&rest_of_shape);
        let mut skipped = false;

        // j is for which column we are on
        for j in 0..ord {
            if j == i { skipped = true; continue; }

            // k is for which row we are on
            for k in 1..ord {
                rest_of_tensor[&[k - 1, if skipped { j - 1 } else { j }]] = t[&[k, j]].clone();
            }
        }

        if is_minus {
            determinant = determinant - t[&[0, i]].clone() * det(&rest_of_tensor)
        } else {
            determinant = determinant + t[&[0, i]].clone() * det(&rest_of_tensor)
        }
    }

    determinant
}

/// Calculates the inverse for a matrix of values of type `T`.
/// This assumes that `T::default()` returns a 0-value of type `T`,
/// which is the case for all common number types, and Complex64 as well.
/// Beware of NaN values or panicking if the determinant is 0.
pub fn inv<T>(t: &Tensor<T>) -> Result<Tensor<T>, TensorErrors>
where T: Add<Output=T> + Mul<Output=T> + Sub<Output=T> + Div<Output=T>  + Neg<Output = T> + Clone + Default + PartialEq + std::fmt::Debug
{
    assert_eq!(t.rank(), 2, "Inversion is only defined for matrices");
    assert_eq!(t.shape[0], t.shape[1], "Inversion is only defined for square matrices");

    let ord = t.shape[0];
    let mut res = Tensor::<T>::from_shape(t.shape());
    let d = det(&t);

    if d == T::default() {
        return Err(DeterminantZero);
    }

    // Construct adjoint matrix
    let rest_of_shape = ts![ord - 1, ord - 1];
    let mut rest_of_tensor = Tensor::<T>::from_shape(&rest_of_shape);

    // i is for which row we are on
    for i in 0..ord {

        // j is for which column we are on
        for j in 0..ord {
            let is_minus = (i + j) % 2 != 0;
            let mut skipped_row = false;

            // k is for which row we are on when filling the rest_of tensor
            for k in 0..ord {
                if i == k { skipped_row = true; continue; }

                let mut skipped_col = false;

                // l is for which column we are on when filling the rest_of tensor
                for l in 0..ord {
                    if j == l { skipped_col = true; continue; }

                    rest_of_tensor[
                        &[
                            if skipped_row { k - 1 } else { k },
                            if skipped_col { l - 1 } else { l },
                        ]
                    ] = t[&[k, l]].clone();
                }
            }

            res[&[j, i]] = if is_minus { -det(&rest_of_tensor) } else { det(&rest_of_tensor) };
        }
    }

    Ok(res / d)
}

/// Constructs an identity matrix of `f64` values of the given size
pub fn identity(n: usize) -> Tensor<f64> {
    let mut t = Tensor::from_shape(&ts![n, n]);

    for i in 0..n {
        t[&[i, i]] = 1.0;
    }

    t
}
