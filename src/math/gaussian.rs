use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::definitions::shape::Shape;
use crate::definitions::tensor::Tensor;
use float_cmp::approx_eq;
use num::complex::Complex64;
use num::ToPrimitive;
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use std::f64::consts::PI;
use std::ops::Add;

/// Creates a tensor of values of the Gaussian pdf with a specified standard deviation.
/// The shape of the result is also specified by the user. The mean is the centre value,
/// but when tensors have an even number of elements in a certain axis, then the centre
/// is treated as being in between the two.
/// This fails if the standard deviation is not positive.
pub fn gaussian_pdf_single_sigma(sigma: f64, shape: Shape) -> Result<Tensor<f64>, TensorErrors> {
    if sigma <= 0.0 {
        return Err(TensorErrors::NonPositiveSigma(sigma));
    }

    let mut res = Tensor::<f64>::from_shape(shape.clone());

    if shape.element_count() == 0 {
        return Ok(res);
    }

    let centre = shape
        .0
        .iter()
        .map(|x| (x - 1).to_f64().unwrap() / 2.0)
        .collect::<Vec<f64>>();

    for (pos, val) in res.enumerated_iter_mut() {
        let exponent = -centre
            .iter()
            .zip(pos.iter())
            .map(|(x, y)| (x - y.to_f64().unwrap()).powi(2))
            .fold(0.0, f64::add)
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
pub fn gaussian_pdf_multi_sigma(
    sigma: Vec<f64>,
    shape: Shape,
) -> Result<Tensor<f64>, TensorErrors> {
    if sigma.len() != shape.rank() {
        return Err(TensorErrors::SigmaListLengthIncompatible(
            sigma.len(),
            shape.rank(),
        ));
    }

    if !sigma.iter().all(|x| x > &0.0) {
        return Err(TensorErrors::SigmaListNotAllPositive);
    }

    let mut res = Tensor::<f64>::from_shape(shape.clone());

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
pub fn gaussian_sample(
    sigma: f64,
    shape: Shape,
    min: f64,
    max: f64,
) -> Result<Tensor<f64>, TensorErrors> {
    if max <= min {
        return Err(TensorErrors::InvalidInterval { min, max });
    }

    if sigma <= 0.0 {
        return Err(TensorErrors::NonPositiveSigma(sigma));
    }

    let step_size = (max - min) / 1e3;

    let mut res = Tensor::<f64>::from_shape(shape.clone());

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
pub fn gaussian_pdf_cov_mat(sigma: Matrix<f64>, shape: Shape) -> Result<Tensor<f64>, TensorErrors> {
    if !sigma.is_square() {
        return Err(TensorErrors::NonSquareMatrix);
    }

    let ord = sigma.rows;

    if ord != shape.rank() {
        return Err(TensorErrors::RanksDoNotMatch(ord, shape.rank()));
    }

    let vals = sigma
        .clone()
        .map(|x| Complex64::new(x, 0.0))
        .eigenvalues()?;

    if !vals
        .iter()
        .all(|x| x.re > 0.0 && approx_eq!(f64, x.im, 0.0, epsilon = 1e-15))
    {
        return Err(TensorErrors::CovMatNotPositiveDefinite);
    }

    let mut res = Tensor::<f64>::from_shape(shape.clone());

    if shape.element_count() == 0 {
        return Ok(res);
    }

    let sigma_inv = sigma.clone().inv()?;

    let centre = shape
        .0
        .iter()
        .map(|x| (x - 1).to_f64().unwrap() / 2.0)
        .collect::<Vec<f64>>();

    let denom = (2.0 * PI).powf(0.5 * ord as f64) * sigma.det()?.sqrt();

    for (pos, val) in res.enumerated_iter_mut() {
        let offset = pos
            .iter()
            .zip(centre.iter())
            .map(|(i, j)| *i as f64 - j)
            .collect::<Matrix<f64>>();

        let offset_t = offset.clone().transpose_mt();

        let exponent = -0.5 * offset.mat_mul_mt(sigma_inv.clone().mat_mul_mt(offset_t)?)?[(0, 0)];

        *val = exponent.exp() / denom;
    }

    Ok(res)
}

#[cfg(test)]
mod gaussian_distribution_tests {
    use float_cmp::{approx_eq, assert_approx_eq};
    use std::f64::consts::PI;
    use tensor_math::definitions::errors::TensorErrors;
    use tensor_math::definitions::matrix::Matrix;
    use tensor_math::definitions::tensor::Tensor;
    use tensor_math::math::gaussian::{
        gaussian_pdf_cov_mat, gaussian_pdf_multi_sigma, gaussian_pdf_single_sigma, gaussian_sample,
    };
    use tensor_math::shape;
    use tensor_math::utilities::matrix::eye;

    #[test]
    fn rand_gaussian_sample() {
        let t1 = gaussian_sample(1.0, shape![3, 3, 3], -10.0, 10.0).unwrap();

        assert!(t1.iter().all(|x| -10.0 <= *x));
        assert!(t1.iter().all(|x| 10.0 >= *x));

        println!("{:?}", t1);
    }

    #[test]
    fn rand_gaussian_sample_invalid_min_more_than_max() {
        let err = gaussian_sample(1.0, shape![3, 3, 3], 10.0, -10.0).unwrap_err();
        assert!(matches!(
            err,
            TensorErrors::InvalidInterval { min: _, max: _ }
        ));
    }

    #[test]
    fn rand_gaussian_sample_invalid_min_eq_max() {
        let err = gaussian_sample(1.0, shape![3, 3, 3], 10.0, 10.0).unwrap_err();
        assert!(matches!(
            err,
            TensorErrors::InvalidInterval { min: _, max: _ }
        ));
    }

    #[test]
    fn rand_gaussian_sample_invalid_neg_sigma() {
        let err = gaussian_sample(-1.0, shape![3, 3, 3], -100.0, -10.0).unwrap_err();
        assert!(matches!(err, TensorErrors::NonPositiveSigma(_)));
    }

    #[test]
    fn rand_gaussian_sample_invalid_zero_sigma() {
        let err = gaussian_sample(0.0, shape![2], -1.0, 1.0).unwrap_err();
        assert_eq!(err, TensorErrors::NonPositiveSigma(0.0));
    }

    #[test]
    fn single_sigma_gaussian_pdf() {
        let t1 = gaussian_pdf_single_sigma(0.5, shape![5, 3]).unwrap();
        let ans = (Tensor::<f64>::new(
            shape![5, 3],
            vec![
                5.0, 4.0, 5.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 5.0, 4.0, 5.0,
            ],
        )
        .unwrap()
            * -2.0)
            .exp()
            * 2.0
            / PI;
        assert_approx_eq!(f64, (t1 - ans).map(f64::abs).sum(), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn invalid_gaussian_single_sigma_negative() {
        let err = gaussian_pdf_single_sigma(-1.0, shape![1, 2, 3]).unwrap_err();
        assert!(matches!(err, TensorErrors::NonPositiveSigma(..)));
    }

    #[test]
    fn invalid_gaussian_single_sigma_zero() {
        let err = gaussian_pdf_single_sigma(0.0, shape![3]).unwrap_err();
        assert_eq!(err, TensorErrors::NonPositiveSigma(0.0));
    }

    #[test]
    fn gaussian_pdf_even_axis_is_centered_between_elements() {
        let t = gaussian_pdf_single_sigma(1.0, shape![2]).unwrap();
        let expected = (-0.125_f64).exp() / (2.0 * PI).sqrt();

        assert_approx_eq!(f64, t[&[0]], expected, epsilon = 1e-15);
        assert_approx_eq!(f64, t[&[1]], expected, epsilon = 1e-15);
    }

    #[test]
    fn multi_sigma_gaussian_pdf() {
        let t1 = gaussian_pdf_multi_sigma(vec![0.25, 0.4], shape![5, 3]).unwrap();
        let ans = (Tensor::<f64>::new(
            shape![5, 3],
            vec![
                35.125, 32.0, 35.125, 11.125, 8.0, 11.125, 3.125, 0.0, 3.125, 11.125, 8.0, 11.125,
                35.125, 32.0, 35.125,
            ],
        )
        .unwrap()
            * -1.0)
            .exp()
            * 5.0
            / PI;
        assert_approx_eq!(f64, (t1 - ans).map(f64::abs).sum(), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn invalid_multi_sigma_gaussian_pdf_zero_sigma() {
        let err = gaussian_pdf_multi_sigma(vec![0.0, 0.5], shape![1, 1]).unwrap_err();
        assert!(matches!(err, TensorErrors::SigmaListNotAllPositive));
    }

    #[test]
    fn invalid_multi_sigma_gaussian_pdf_negative_sigma() {
        let err = gaussian_pdf_multi_sigma(vec![0.5, -0.5], shape![1, 1]).unwrap_err();
        assert!(matches!(err, TensorErrors::SigmaListNotAllPositive));
    }

    #[test]
    fn invalid_multi_sigma_gaussian_pdf_sigma_len_invalid() {
        let err = gaussian_pdf_multi_sigma(vec![0.1, 0.5], shape![1, 1, 3]).unwrap_err();
        assert!(matches!(
            err,
            TensorErrors::SigmaListLengthIncompatible(_, _)
        ));
    }

    #[test]
    fn gaussian_covariance_matrix_tensor() {
        let cov = Matrix::<f64>::new(2, 2, vec![1.0, 0.5, 0.5, 1.0]).unwrap();

        let ans = Tensor::<f64>::new(
            shape![3, 4],
            vec![
                0.057228531823873385,
                0.11146595955293902,
                0.057228531823873385,
                0.007745039563599408,
                0.041006034909973794,
                0.1555632781262252,
                0.1555632781262252,
                0.041006034909973794,
                0.007745039563599408,
                0.057228531823873385,
                0.11146595955293902,
                0.057228531823873385,
            ],
        )
        .unwrap();
        assert!(approx_eq!(
            Tensor<f64>,
            gaussian_pdf_cov_mat(cov, shape![3, 4]).unwrap(),
            ans,
            epsilon = 1e-10
        ));
    }

    #[test]
    fn invalid_cov_mat_non_square() {
        let err = gaussian_pdf_cov_mat(
            Matrix::<f64>::new(2, 1, vec![0.0, 1.0]).unwrap(),
            shape![1, 1],
        )
        .unwrap_err();
        assert!(matches!(err, TensorErrors::NonSquareMatrix));
    }

    #[test]
    fn invalid_cov_mat_shape_not_right_dim() {
        let err = gaussian_pdf_cov_mat(eye(2), shape![1]).unwrap_err();
        assert!(matches!(err, TensorErrors::RanksDoNotMatch(_, _)));
    }

    #[test]
    fn invalid_cov_mat_not_positive_definite() {
        let err = gaussian_pdf_cov_mat(
            Matrix::<f64>::new(2, 2, vec![0.0, 1.0, 1.0, 0.0]).unwrap(),
            shape![1, 1],
        )
        .unwrap_err();
        assert!(matches!(err, TensorErrors::CovMatNotPositiveDefinite));
    }

    #[test]
    fn gaussian_pdf_single_sigma_rank_zero() {
        let t = gaussian_pdf_single_sigma(1.0, shape![]).unwrap();
        assert_eq!(t.shape().rank(), 0);
        assert_approx_eq!(f64, t[&[]], 1.0, epsilon = 1e-15);
    }

    #[test]
    fn gaussian_pdf_multi_sigma_rank_zero() {
        let t = gaussian_pdf_multi_sigma(vec![], shape![]).unwrap();
        assert_eq!(t.shape().rank(), 0);
        // Exponent 0, Denominator 1
        assert_approx_eq!(f64, t[&[]], 1.0, epsilon = 1e-15);
    }

    #[test]
    fn gaussian_sample_rank_zero() {
        let t = gaussian_sample(1.0, shape![], -10.0, 10.0).unwrap();
        assert_eq!(t.shape().rank(), 0);
        assert!(t[&[]] >= -10.0 && t[&[]] <= 10.0);
    }

    #[test]
    fn gaussian_pdf_cov_mat_rank_zero() {
        let cov = Matrix::<f64>::new(0, 0, vec![]).unwrap();
        let t = gaussian_pdf_cov_mat(cov, shape![]).unwrap();
        assert_eq!(t.shape().rank(), 0);
        assert_approx_eq!(f64, t[&[]], 1.0, epsilon = 1e-15);
    }

    #[test]
    fn gaussian_pdf_single_sigma_empty_tensor() {
        let t = gaussian_pdf_single_sigma(1.0, shape![0, 5]).unwrap();
        assert_eq!(t.len(), 0);
    }

    #[test]
    fn gaussian_pdf_multi_sigma_empty_tensor() {
        let t = gaussian_pdf_multi_sigma(vec![1.0, 1.0], shape![2, 0]).unwrap();
        assert_eq!(t.len(), 0);
    }

    #[test]
    fn gaussian_sample_empty_tensor() {
        let t = gaussian_sample(1.0, shape![0], -1.0, 1.0).unwrap();
        assert_eq!(t.len(), 0);
    }

    #[test]
    fn gaussian_pdf_cov_mat_empty_tensor() {
        let cov = eye(2);
        let t = gaussian_pdf_cov_mat(cov, shape![2, 0]).unwrap();
        assert_eq!(t.len(), 0);
    }
}
