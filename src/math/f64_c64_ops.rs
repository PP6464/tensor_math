use num::complex::{Complex64, ComplexFloat};
use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::definitions::tensor::Tensor;
use crate::definitions::transpose::Transpose;

impl Tensor<f64> {
    /// Converts this into a tensor of `Complex64` values
    pub fn into_complex(self) -> Tensor<Complex64> { self.map(Complex64::from) }
    
    /// Exponentiates each element in the tensor
    pub fn exp(self) -> Tensor<f64> {
        self.map(f64::exp)
    }
    
    /// Computes the natural logarithm of each element in the tensor
    pub fn ln(self) -> Tensor<f64> { self.map(f64::ln) }
    
    /// Computes the log base n of each element in the tensor
    pub fn log(self, n: f64) -> Tensor<f64> { self.map(|x| x.log(n)) }
    
    /// Computes the log base 2 of each element in the tensor
    pub fn log2(self) -> Tensor<f64> { self.map(f64::log2) }
    
    /// Computes the log base 10 of each element in the tensor
    pub fn log10(self) -> Tensor<f64> { self.map(f64::log10) }

    /// Raises n to the power of each element.
    /// This method uses `f64::powf` so beware of `f64::NaN` values
    /// if you have a negative value raised to the power of 0.5 for example
    pub fn exp_base_n(self, n: f64) -> Tensor<f64> {
        self.map(|x| f64::powf(n, x))
    }

    /// Raises each element to the power of n
    /// Like `exp_base_n` this can give `NaN` values if you aren't careful
    /// with what you are raising to which power
    pub fn pow(self, n: f64) -> Tensor<f64> {
        self.map(|x| f64::powf(x, n))
    }

    /// Computes the sin of each element
    pub fn sin(self) -> Tensor<f64> {
        self.map(f64::sin)
    }

    /// Computes the cos of each element
    pub fn cos(self) -> Tensor<f64> {
        self.map(f64::cos)
    }

    /// Computes the tan of each element
    pub fn tan(self) -> Tensor<f64> {
        self.map(f64::tan)
    }

    /// Computes the arcsin of each element
    pub fn asin(self) -> Tensor<f64> {
        self.map(f64::asin)
    }

    /// Computes the arccos of each element
    pub fn acos(self) -> Tensor<f64> {
        self.map(f64::acos)
    }

    /// Computes the atan of each element
    pub fn atan(self) -> Tensor<f64> {
        self.map(f64::atan)
    }

    /// Computes the reciprocal of each element.
    /// Use in conjunction with trigonometric functions to get sec(x), coth(x) etc.
    pub fn recip(self) -> Tensor<f64> {
        self.map(f64::recip)
    }

    /// Computes the sinh of each element
    pub fn sinh(self) -> Tensor<f64> {
        self.map(f64::sinh)
    }

    /// Computes the cosh of each element
    pub fn cosh(self) -> Tensor<f64> {
        self.map(f64::cosh)
    }

    /// Computes the tanh of each element
    pub fn tanh(self) -> Tensor<f64> {
        self.map(f64::tanh)
    }

    /// Computes the arsinh of each element
    pub fn asinh(self) -> Tensor<f64> {
        self.map(f64::asinh)
    }

    /// Computes the arcosh of each element
    pub fn acosh(self) -> Tensor<f64> {
        self.map(f64::acosh)
    }

    /// Computes the artanh of each element
    pub fn atanh(self) -> Tensor<f64> {
        self.map(f64::atanh)
    }

    /// Computes the sigmoid function for each element.
    /// Sigmoid(x) = 1 / (1 + exp(-x))
    pub fn sigmoid(self) -> Tensor<f64> {
        ((-self).exp() + 1.0).recip()
    }

    /// Computes the ReLU function for each element
    pub fn relu(self) -> Tensor<f64> {
        self.map(|x| if x > 0.0 { x } else { 0.0 })
    }

    /// Computes the leaky ReLU function for each element
    pub fn leaky_relu(self, alpha: f64) -> Tensor<f64> {
        self.map(|x| if x > 0.0 { x } else { alpha * x })
    }

    /// Applies softmax to the tensor
    pub fn softmax(self) -> Tensor<f64> {
        let new = self.exp();
        let sum = new.sum();
        new / sum
    }

    /// Computes the square root of every element
    pub fn sqrt(self) -> Tensor<f64> { self.map(f64::sqrt) }

    /// Computes the cube root of every element
    pub fn cbrt(self) -> Tensor<f64> { self.map(f64::cbrt) }

    /// Normalises the tensor so the sum of magnitudes is 1
    pub fn norm_l1(self) -> Tensor<f64> {
        let mag = self.clone().map(|x| x.abs()).sum();
        self / mag
    }

    /// Normalises the tensor so the sum of the squares of the magnitudes is 1
    pub fn norm_l2(self) -> Tensor<f64> {
        let mag = self.clone().map(|x| x * x).sum().sqrt();
        self / mag
    }

    /// Computes the sum of the squares of the values, then square roots the result
    pub fn mag(self) -> f64 { self.map(|x| x * x).sum().sqrt() }

    /// Computes the sum of squares of the values
    pub fn mag_2(self) -> f64 { self.map(|x| x * x).sum() }

    /// Computes the absolute value of every element
    pub fn abs(self) -> Tensor<f64> { self.map(|x| x.abs()) }

    /// Computes the squared norms of each element, then divides by the square of the magnitude
    pub fn born_probabilities(self) -> Tensor<f64> {
        let div_by = self.clone().mag_2();
        self.pow(2.0) / div_by
    }
}

impl Matrix<f64> {
    /// Converts this into a matrix of `Complex64` values
    pub fn into_complex(self) -> Matrix<Complex64> { self.map(Complex64::from) }
    
    /// Exponentiates each element in the tensor
    pub fn exp(self) -> Matrix<f64> {
        self.map(f64::exp)
    }

    /// Computes the natural logarithm of each element in the tensor
    pub fn ln(self) -> Matrix<f64> { self.map(f64::ln) }

    /// Computes the log base n of each element in the tensor
    pub fn log(self, n: f64) -> Matrix<f64> { self.map(|x| x.log(n)) }

    /// Computes the log base 2 of each element in the tensor
    pub fn log2(self) -> Matrix<f64> { self.map(f64::log2) }

    /// Computes the log base 10 of each element in the tensor
    pub fn log10(self) -> Matrix<f64> { self.map(f64::log10) }

    /// Raises n to the power of each element.
    /// This method uses `f64::powf` so beware of `f64::NaN` values
    /// if you have a negative value raised to the power of 0.5 for example
    pub fn exp_base_n(self, n: f64) -> Matrix<f64> {
        self.map(|x| f64::powf(n, x))
    }

    /// Raises each element to the power of n
    /// Like `exp_base_n` this can give `NaN` values if you aren't careful
    /// with what you are raising to which power
    pub fn pow(self, n: f64) -> Matrix<f64> {
        self.map(|x| f64::powf(x, n))
    }

    /// Computes the sin of each element
    pub fn sin(self) -> Matrix<f64> {
        self.map(f64::sin)
    }

    /// Computes the cos of each element
    pub fn cos(self) -> Matrix<f64> {
        self.map(f64::cos)
    }

    /// Computes the tan of each element
    pub fn tan(self) -> Matrix<f64> {
        self.map(f64::tan)
    }

    /// Computes the arcsin of each element
    pub fn asin(self) -> Matrix<f64> {
        self.map(f64::asin)
    }

    /// Computes the arccos of each element
    pub fn acos(self) -> Matrix<f64> {
        self.map(f64::acos)
    }

    /// Computes the atan of each element
    pub fn atan(self) -> Matrix<f64> {
        self.map(f64::atan)
    }

    /// Computes the reciprocal of each element.
    /// Use in conjunction with trigonometric functions to get sec(x), coth(x) etc.
    pub fn recip(self) -> Matrix<f64> {
        self.map(f64::recip)
    }

    /// Computes the sinh of each element
    pub fn sinh(self) -> Matrix<f64> {
        self.map(f64::sinh)
    }

    /// Computes the cosh of each element
    pub fn cosh(self) -> Matrix<f64> {
        self.map(f64::cosh)
    }

    /// Computes the tanh of each element
    pub fn tanh(self) -> Matrix<f64> {
        self.map(f64::tanh)
    }

    /// Computes the arsinh of each element
    pub fn asinh(self) -> Matrix<f64> {
        self.map(f64::asinh)
    }

    /// Computes the arcosh of each element
    pub fn acosh(self) -> Matrix<f64> {
        self.map(f64::acosh)
    }

    /// Computes the artanh of each element
    pub fn atanh(self) -> Matrix<f64> {
        self.map(f64::atanh)
    }

    /// Computes the sigmoid function for each element.
    /// Sigmoid(x) = 1 / (1 + exp(-x))
    pub fn sigmoid(self) -> Matrix<f64> {
        ((-self).exp() + 1.0).recip()
    }

    /// Computes the ReLU function for each element
    pub fn relu(self) -> Matrix<f64> {
        self.map(|x| if x > 0.0 { x } else { 0.0 })
    }

    /// Computes the leaky ReLU function for each element
    pub fn leaky_relu(self, alpha: f64) -> Matrix<f64> {
        self.map(|x| if x > 0.0 { x } else { alpha * x })
    }

    /// Applies softmax to the tensor
    pub fn softmax(self) -> Matrix<f64> {
        let new = self.exp();
        let sum = new.sum();
        new / sum
    }

    /// Computes the square root of every element
    pub fn sqrt(self) -> Matrix<f64> { self.map(f64::sqrt) }

    /// Computes the cube root of every element
    pub fn cbrt(self) -> Matrix<f64> { self.map(f64::cbrt) }

    /// Normalises the tensor so the sum of magnitudes is 1
    pub fn norm_l1(self) -> Matrix<f64> {
        let mag = self.clone().map(|x| x.abs()).sum();
        self / mag
    }

    /// Normalises the tensor so the sum of the squares of the magnitudes is 1
    pub fn norm_l2(self) -> Matrix<f64> {
        let mag = self.clone().map(|x| x * x).sum().sqrt();
        self / mag
    }

    /// Computes the sum of squares of values, then square roots the result
    pub fn mag(self) -> f64 { self.map(|x| x * x).sum().sqrt() }

    /// Computes the sum of squares of values
    pub fn mag_2(self) -> f64 { self.map(|x| x * x).sum() }

    /// Computes the absolute value of every element
    pub fn abs(self) -> Matrix<f64> { self.map(|x| x.abs()) }

    /// Computes the squared norms of each element, then divides by the square of the magnitude
    pub fn born_probabilities(self) -> Matrix<f64> {
        let div_by = self.clone().mag_2();
        self.pow(2.0) / div_by
    }
}

impl Tensor<Complex64> {
    /// Gives the real parts of all the elements
    pub fn re(self) -> Tensor<f64> { self.map(Complex64::re) }
    
    /// Gives the imaginary parts of all the elements
    pub fn im(self) -> Tensor<f64> { self.map(Complex64::im) }
    
    /// Computes the exponential of each element
    pub fn exp(self) -> Tensor<Complex64> {
        self.map(Complex64::exp)
    }

    /// Computes the natural logarithm of each element in the tensor
    pub fn ln(self) -> Tensor<Complex64> { self.map(Complex64::ln) }

    /// Computes the log base n of each element in the tensor
    pub fn log(self, n: f64) -> Tensor<Complex64> { self.map(|x| x.log(n)) }

    /// Computes the log base 2 of each element in the tensor
    pub fn log2(self) -> Tensor<Complex64> { self.map(Complex64::log2) }

    /// Computes the log base 10 of each element in the tensor
    pub fn log10(self) -> Tensor<Complex64> { self.map(Complex64::log10) }

    /// Raises n to the power of each element
    pub fn exp_base_n(self, n: Complex64) -> Tensor<Complex64> {
        self.map(|x| n.powc(x))
    }

    /// Raises each element to the power of n
    pub fn pow(self, n: Complex64) -> Tensor<Complex64> {
        self.map(|x| x.powc(n))
    }

    /// Computes the sin of each element
    pub fn sin(self) -> Tensor<Complex64> {
        self.map(Complex64::sin)
    }

    /// Computes the cos of each element
    pub fn cos(self) -> Tensor<Complex64> {
        self.map(Complex64::cos)
    }

    /// Computes the tan of each element
    pub fn tan(self) -> Tensor<Complex64> {
        self.map(Complex64::tan)
    }

    /// Computes the arcsin of each element
    pub fn asin(self) -> Tensor<Complex64> {
        self.map(Complex64::asin)
    }

    /// Computes the arccos of each element
    pub fn acos(self) -> Tensor<Complex64> {
        self.map(Complex64::acos)
    }

    /// Computes the arctan of each element
    pub fn atan(self) -> Tensor<Complex64> {
        self.map(Complex64::atan)
    }

    /// Computes the sinh of each element
    pub fn sinh(self) -> Tensor<Complex64> {
        self.map(Complex64::sinh)
    }

    /// Computes the cosh of each element
    pub fn cosh(self) -> Tensor<Complex64> {
        self.map(Complex64::cosh)
    }

    /// Computes the tanh of each element
    pub fn tanh(self) -> Tensor<Complex64> {
        self.map(Complex64::tanh)
    }

    /// Computes the arsinh of each element
    pub fn asinh(self) -> Tensor<Complex64> {
        self.map(Complex64::asinh)
    }

    /// Computes the arcosh of each element
    pub fn acosh(self) -> Tensor<Complex64> {
        self.map(Complex64::acosh)
    }

    /// Computes the artanh of each element
    pub fn atanh(self) -> Tensor<Complex64> {
        self.map(Complex64::atanh)
    }

    /// Computes the reciprocal of each element.
    /// Use in conjunction with trigonometric functions to get sec(x), coth(x) etc.
    pub fn recip(self) -> Tensor<Complex64> {
        self.map(Complex64::recip)
    }

    /// Computes the square root of every element
    pub fn sqrt(self) -> Tensor<Complex64> { self.map(Complex64::sqrt) }

    /// Computes the cube root of every element
    pub fn cbrt(self) -> Tensor<Complex64> { self.map(Complex64::cbrt) }

    /// Normalises the tensor so the sum of magnitudes is 1
    pub fn norm_l1(self) -> Tensor<Complex64> {
        let mag: Complex64 = self.clone().map(|x| x.abs()).sum().into();
        self / mag
    }

    /// Normalises the tensor so the sum of the squares of the magnitudes is 1
    pub fn norm_l2(self) -> Tensor<Complex64> {
        let mag: Complex64 = self
            .clone()
            .map(|x| (x * x).abs())
            .sum()
            .sqrt()
            .into();
        self / mag
    }

    /// Conjugates every element in the list
    pub fn conj(self) -> Tensor<Complex64> { self.map_refs(Complex64::conj) }

    /// Returns the conjugate transpose of a `Tensor<Complex64>`. This uses the multithreaded
    /// implementation of `transpose`.
    pub fn conj_transpose_mt(&self, transpose: &Transpose) -> Result<Tensor<Complex64>, TensorErrors> {
        Ok(self.transpose_mt(&transpose)?.map(|x| x.conj()))
    }

    /// Returns the conjugate transpose of a `Tensor<Complex64>`. This uses the single-threaded
    /// implementation of `transpose`.
    pub fn conj_transpose(&self, transpose: &Transpose) -> Result<Tensor<Complex64>, TensorErrors> {
        Ok(self.transpose(&transpose)?.map(|x| x.conj()))
    }

    /// Computes the sum of the square of the absolute values, then square roots the result
    pub fn mag(self) -> f64 { self.map(|x| (x * x).abs()).sum().sqrt() }

    /// Computes the sum of the square of the absolute values
    pub fn mag_2(self) -> f64 { self.map(|x| (x * x).abs()).sum() }

    /// Computes the absolute value of every element
    pub fn abs(self) -> Tensor<f64> { self.map(|x| x.abs()) }

    /// Computes the squared norms of each element, then divides by the square of the magnitude
    pub fn born_probabilities(self) -> Tensor<f64> {
        let div_by = self.clone().mag_2();
        self.abs().pow(2.0) / div_by
    }
}

impl Matrix<Complex64> {
    /// Gives the real parts of all the elements
    pub fn re(self) -> Matrix<f64> { self.map(Complex64::re) }
    
    /// Gives the imaginary parts of all the elements
    pub fn im(self) -> Matrix<f64> { self.map(Complex64::im) }
    
    /// Computes the exponential of each element
    pub fn exp(self) -> Matrix<Complex64> {
        self.map(Complex64::exp)
    }

    /// Computes the natural logarithm of each element in the tensor
    pub fn ln(self) -> Matrix<Complex64> { self.map(Complex64::ln) }

    /// Computes the log base n of each element in the tensor
    pub fn log(self, n: f64) -> Matrix<Complex64> { self.map(|x| x.log(n)) }

    /// Computes the log base 2 of each element in the tensor
    pub fn log2(self) -> Matrix<Complex64> { self.map(Complex64::log2) }

    /// Computes the log base 10 of each element in the tensor
    pub fn log10(self) -> Matrix<Complex64> { self.map(Complex64::log10) }

    /// Raises n to the power of each element
    pub fn exp_base_n(self, n: Complex64) -> Matrix<Complex64> {
        self.map(|x| n.powc(x))
    }

    /// Raises each element to the power of n
    pub fn pow(self, n: Complex64) -> Matrix<Complex64> {
        self.map(|x| x.powc(n))
    }

    /// Computes the sin of each element
    pub fn sin(self) -> Matrix<Complex64> {
        self.map(Complex64::sin)
    }

    /// Computes the cos of each element
    pub fn cos(self) -> Matrix<Complex64> {
        self.map(Complex64::cos)
    }

    /// Computes the tan of each element
    pub fn tan(self) -> Matrix<Complex64> {
        self.map(Complex64::tan)
    }

    /// Computes the arcsin of each element
    pub fn asin(self) -> Matrix<Complex64> {
        self.map(Complex64::asin)
    }

    /// Computes the arccos of each element
    pub fn acos(self) -> Matrix<Complex64> {
        self.map(Complex64::acos)
    }

    /// Computes the arctan of each element
    pub fn atan(self) -> Matrix<Complex64> {
        self.map(Complex64::atan)
    }

    /// Computes the sinh of each element
    pub fn sinh(self) -> Matrix<Complex64> {
        self.map(Complex64::sinh)
    }

    /// Computes the cosh of each element
    pub fn cosh(self) -> Matrix<Complex64> {
        self.map(Complex64::cosh)
    }

    /// Computes the tanh of each element
    pub fn tanh(self) -> Matrix<Complex64> {
        self.map(Complex64::tanh)
    }

    /// Computes the arsinh of each element
    pub fn asinh(self) -> Matrix<Complex64> {
        self.map(Complex64::asinh)
    }

    /// Computes the arcosh of each element
    pub fn acosh(self) -> Matrix<Complex64> {
        self.map(Complex64::acosh)
    }

    /// Computes the artanh of each element
    pub fn atanh(self) -> Matrix<Complex64> {
        self.map(Complex64::atanh)
    }

    /// Computes the reciprocal of each element.
    /// Use in conjunction with trigonometric functions to get sec(x), coth(x) etc.
    pub fn recip(self) -> Matrix<Complex64> {
        self.map(Complex64::recip)
    }

    /// Computes the square root of every element
    pub fn sqrt(self) -> Matrix<Complex64> { self.map(Complex64::sqrt) }

    /// Computes the cube root of every element
    pub fn cbrt(self) -> Matrix<Complex64> { self.map(Complex64::cbrt) }

    /// Normalises the tensor so the sum of magnitudes is 1
    pub fn norm_l1(self) -> Matrix<Complex64> {
        let mag: Complex64 = self.clone().map(|x| x.abs()).sum().into();
        self / mag
    }

    /// Normalises the tensor so the sum of the squares of the magnitudes is 1
    pub fn norm_l2(self) -> Matrix<Complex64> {
        let mag: Complex64 = self
            .clone()
            .map(|x| (x * x).abs())
            .sum()
            .sqrt()
            .into();
        self / mag
    }

    /// Conjugates every element in the matrix
    pub fn conj(self) -> Matrix<Complex64> { self.map_refs(Complex64::conj) }

    /// Returns the conjugate transpose of a `Matrix<Complex64>`. This uses the multithreaded
    /// implementation of `transpose`.
    pub fn conj_transpose_mt(&self) -> Matrix<Complex64> {
        self.transpose_mt().map(|x| x.conj())
    }

    /// Returns the conjugate transpose of a `Matrix<Complex64>`. This uses the single-threaded
    /// implementation of `transpose`.
    pub fn conj_transpose(&self) -> Matrix<Complex64> {
        self.transpose().map(|x| x.conj())
    }

    /// Computes the sum of the square of the absolute values, then square roots the result
    pub fn mag(self) -> f64 { self.map(|x| (x * x).abs()).sum().sqrt() }

    /// Computes the sum of the square of the absolute values
    pub fn mag_2(self) -> f64 { self.map(|x| (x * x).abs()).sum() }

    /// Computes the absolute value of every element
    pub fn abs(self) -> Matrix<f64> { self.map(|x| x.abs()) }

    /// Computes the squared norms of each element, then divides by the square of the magnitude
    pub fn born_probabilities(self) -> Matrix<f64> {
        let div_by = self.clone().mag_2();
        self.abs().pow(2.0) / div_by
    }
}
