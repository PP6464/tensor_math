use std::f64::consts::PI;
use std::ops::Add;
use float_cmp::approx_eq;
use num::complex::Complex64;
use num::ToPrimitive;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::definitions::shape::Shape;
use crate::definitions::tensor::Tensor;

/// Creates a tensor of values of the Gaussian pdf with a specified standard deviation.
/// The shape of the result is also specified by the user. The mean is the centre value,
/// but when tensors have an even number of elements in a certain axis, then the centre
/// is treated as being in between the two.
/// This fails if the standard deviation is not positive.
pub fn gaussian_pdf_single_sigma(sigma: f64, shape: &Shape) -> Result<Tensor<f64>, TensorErrors> {
    if sigma <= 0.0 {
        return Err(TensorErrors::NonPositiveSigma(sigma));
    }

    let mut res = Tensor::<f64>::from_shape(shape);

    if shape.element_count() == 0 {
        return Ok(res);
    }

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
            .fold(0.0, f64::add)
            * -1.0
            / (2.0 * sigma.powi(2));

        *val = f64::exp(exponent) / (sigma * f64::sqrt(2.0 * PI)).powi(shape.rank() as i32)
    }

    Ok(res)
}

/// Creates a tensor of values of the Gaussian pdf with a specified list
/// of standard deviations, one for each axis of the tensor.
/// The shape of the result is also specified by the user. The mean is the centre value,
/// but when tensors have an even number of elements in a certain axis, then the centre
/// is treated as being in between the two.
/// This fails if any of the standard deviations are not positive.
pub fn gaussian_pdf_multi_sigma(sigma: Vec<f64>, shape: &Shape) -> Result<Tensor<f64>, TensorErrors> {
    if sigma.len() != shape.rank() {
        return Err(TensorErrors::SigmaListLengthIncompatible(sigma.len(), shape.rank()));
    }
    
    if !sigma.iter().all(|x| x > &0.0) {
        return Err(TensorErrors::SigmaListNotAllPositive);
    }

    let mut res = Tensor::<f64>::from_shape(shape);

    if shape.element_count() == 0 {
        return Ok(res);
    }

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

    Ok(res)
}

/// Creates a tensor of values with the specified shape where the values are sampled from a
/// Gaussian distribution. The min and max values allow you to specify the range of the outputs.
/// The possible outputs will be 1001 different outputs spaced evenly across the interval [min, max\].
/// This fails if the standard deviation is not positive or if `max <= min`.
pub fn gaussian_sample(sigma: f64, shape: &Shape, min: f64, max: f64) -> Result<Tensor<f64>, TensorErrors> {
    if max <= min {
        return Err(TensorErrors::InvalidInterval {
            min,
            max,
        });
    }
    
    if sigma <= 0.0 {
        return Err(TensorErrors::NonPositiveSigma(sigma));
    }

    let step_size = (max - min) / 1e3;

    let mut res = Tensor::<f64>::from_shape(shape);

    if shape.element_count() == 0 {
        return Ok(res);
    }

    let dist = WeightedIndex::new(
        (0..=1000)
            .map(|x| f64::exp(-(x.to_f64().unwrap() - 500.0).powi(2)) / (2.0 * sigma.powi(2)))
            .collect::<Vec<f64>>(),
    )
        .unwrap();
    let mut rng = rand::rng();

    for val in res.iter_mut() {
        let index = dist.sample(&mut rng).to_f64().unwrap();
        *val = min + index * step_size;
    }

    Ok(res)
}

/// Creates a tensor of values of the Gaussian pdf with a specified covariance matrix.
/// The shape of the result is also specified by the user. The mean is the centre value,
/// but when tensors have an even number of elements in a certain axis, then the centre
/// is treated as being in between the two.
/// This fails if the standard deviation matrix is not positive definite.
pub fn gaussian_pdf_cov_mat(sigma: Matrix<f64>, shape: &Shape) -> Result<Tensor<f64>, TensorErrors> {
    if !sigma.is_square() {
        return Err(TensorErrors::NonSquareMatrix);
    }

    let ord = sigma.rows;

    if ord != shape.rank() {
        return Err(TensorErrors::RanksDoNotMatch(ord, shape.rank()));
    }

    let (vals, _) = sigma.clone().map(|x| Complex64::new(x, 0.0)).eigendecompose()?;
    
    if !vals.iter().all(|x| x.re > 0.0 && approx_eq!(f64, x.im, 0.0, epsilon = 1e-15)) {
        return Err(TensorErrors::CovMatNotPositiveDefinite);
    }

    let mut res = Tensor::<f64>::from_shape(shape);

    if shape.element_count() == 0 {
        return Ok(res);
    }

    let sigma_inv = sigma.inv()?;

    let centre = shape.0.iter().map(|x| { (x - 1).to_f64().unwrap() / 2.0 }).collect::<Vec<f64>>();

    let denom = (2.0 * PI).powf(0.5 * ord as f64) * sigma.det()?.sqrt();

    for (pos, val) in res.enumerated_iter_mut() {
        let offset = pos.iter().zip(centre.iter()).map(|(i, j)| *i as f64 - j).collect::<Matrix<f64>>();

        let exponent = -0.5 * offset.contract_mul_mt(&sigma_inv.contract_mul_mt(&offset.transpose_mt())?)?[(0, 0)];

        *val = exponent.exp() / denom;
    }

    Ok(res)
}