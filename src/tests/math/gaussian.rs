#[cfg(test)]
mod gaussian_distribution_tests {
    use crate::definitions::matrix::Matrix;
    use crate::definitions::shape::Shape;
    use crate::definitions::tensor::Tensor;
    use crate::math::gaussian::{gaussian_pdf_cov_mat, gaussian_pdf_multi_sigma, gaussian_pdf_single_sigma, gaussian_sample};
    use crate::shape;
    use crate::utilities::matrix::eye;
    use float_cmp::{approx_eq, assert_approx_eq};
    use std::f64::consts::PI;
    use crate::definitions::errors::TensorErrors;

    #[test]
    fn rand_gaussian_sample() {
        let t1 = gaussian_sample(1.0, &shape![3, 3, 3], -10.0, 10.0).unwrap();

        assert!(t1.iter().all(|x| -10.0 <= *x));
        assert!(t1.iter().all(|x| 10.0 >= *x));

        println!("{:?}", t1);
    }

    #[test]
    fn rand_gaussian_sample_invalid_min_more_than_max() {
        let err = gaussian_sample(1.0, &shape![3, 3, 3], 10.0, -10.0).unwrap_err();
        match err {
            TensorErrors::InvalidInterval { min: _, max: _ } => {},
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn rand_gaussian_sample_invalid_min_eq_max() {
        let err = gaussian_sample(1.0, &shape![3, 3, 3], 10.0, 10.0).unwrap_err();
        match err {
            TensorErrors::InvalidInterval { min: _, max: _ } => {},
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn rand_gaussian_sample_invalid_neg_sigma() {
        let err = gaussian_sample(-1.0, &shape![3, 3, 3], -100.0, -10.0).unwrap_err();
        match err {
            TensorErrors::NonPositiveSigma(_) => {},
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn single_sigma_gaussian_pdf() {
        let t1 = gaussian_pdf_single_sigma(0.5, &shape![5, 3]).unwrap();
        let ans = (Tensor::<f64>::new(
            &shape![5, 3],
            vec![
                5.0, 4.0, 5.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 5.0, 4.0, 5.0,
            ],
        )
            .unwrap()
            * -2.0)
            .exp()
            * 2.0
            / PI;
        assert_approx_eq!(
            f64,
            (t1 - ans).map(f64::abs).sum(),
            0.0,
            epsilon = 1e-15
        );
    }
    
    #[test]
    fn invalid_gaussian_single_sigma_negative() {
        let err = gaussian_pdf_single_sigma(-1.0, &shape![1, 2, 3]).unwrap_err();
        match err {
            TensorErrors::NonPositiveSigma(..) => {},
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn multi_sigma_gaussian_pdf() {
        let t1 = gaussian_pdf_multi_sigma(vec![0.25, 0.4], &shape![5, 3]).unwrap();
        let ans = (Tensor::<f64>::new(
            &shape![5, 3],
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
        assert_approx_eq!(
            f64,
            (t1 - ans).map(f64::abs).sum(),
            0.0,
            epsilon = 1e-15
        );
    }

    #[test]
    fn invalid_multi_sigma_gaussian_pdf_zero_sigma() {
        let err = gaussian_pdf_multi_sigma(vec![0.0, 0.5], &shape![1, 1]).unwrap_err();
        match err {
            TensorErrors::SigmaListNotAllPositive => {},
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn invalid_multi_sigma_gaussian_pdf_negative_sigma() {
        let err = gaussian_pdf_multi_sigma(vec![0.5, -0.5], &shape![1, 1]).unwrap_err();
        match err {
            TensorErrors::SigmaListNotAllPositive => {},
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn invalid_multi_sigma_gaussian_pdf_sigma_len_invalid() {
        let err = gaussian_pdf_multi_sigma(vec![0.1, 0.5], &shape![1, 1, 3]).unwrap_err();
        match err {
            TensorErrors::SigmaListLengthIncompatible(_, _) => {},
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn gaussian_covariance_matrix_tensor() {
        let cov = Matrix::<f64>::new(
            2, 2,
            vec![
                1.0, 0.5,
                0.5, 1.0
            ],
        ).unwrap();

        let ans = Tensor::<f64>::new(
            &shape![3, 4],
            vec![
                0.057228531823873385, 0.11146595955293902, 0.057228531823873385, 0.007745039563599408,
                0.041006034909973794, 0.1555632781262252, 0.1555632781262252, 0.041006034909973794,
                0.007745039563599408, 0.057228531823873385, 0.11146595955293902, 0.057228531823873385,
            ],
        ).unwrap();
        assert!(approx_eq!(Tensor<f64>, gaussian_pdf_cov_mat(cov, &shape![3, 4]).unwrap(), ans, epsilon = 1e-15));
    }

    #[test]
    fn invalid_cov_mat_non_square() {
        let err = gaussian_pdf_cov_mat(Matrix::<f64>::new(2, 1, vec![0.0, 1.0]).unwrap(), &shape![1, 1]).unwrap_err();
        match err {
            TensorErrors::NonSquareMatrix => {},
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn invalid_cov_mat_shape_not_right_dim() {
        let err = gaussian_pdf_cov_mat(eye(2), &shape![1]).unwrap_err();
        match err {
            TensorErrors::RanksDoNotMatch(_, _) => {},
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn invalid_cov_mat_not_positive_definite() {
        let err = gaussian_pdf_cov_mat(Matrix::<f64>::new(2, 2, vec![0.0, 1.0, 1.0, 0.0]).unwrap(), &shape![1, 1]).unwrap_err();
        match err {
            TensorErrors::CovMatNotPositiveDefinite => {},
            _ => panic!("Incorrect error"),
        }
    }
    
    #[test]
    fn gaussian_pdf_single_sigma_rank_zero() {
        let t = gaussian_pdf_single_sigma(1.0, &shape![]).unwrap();
        assert_eq!(t.shape().rank(), 0);
        assert_approx_eq!(f64, t[&[]], 1.0, epsilon = 1e-15);
    }

    #[test]
    fn gaussian_pdf_multi_sigma_rank_zero() {
        let t = gaussian_pdf_multi_sigma(vec![], &shape![]).unwrap();
        assert_eq!(t.shape().rank(), 0);
        // Exponent 0, Denominator 1
        assert_approx_eq!(f64, t[&[]], 1.0, epsilon = 1e-15);
    }

    #[test]
    fn gaussian_sample_rank_zero() {
        let t = gaussian_sample(1.0, &shape![], -10.0, 10.0).unwrap();
        assert_eq!(t.shape().rank(), 0);
        assert!(t[&[]] >= -10.0 && t[&[]] <= 10.0);
    }

    #[test]
    fn gaussian_pdf_cov_mat_rank_zero() {
        let cov = Matrix::<f64>::new(0, 0, vec![]).unwrap();
        let t = gaussian_pdf_cov_mat(cov, &shape![]).unwrap();
        assert_eq!(t.shape().rank(), 0);
        assert_approx_eq!(f64, t[&[]], 1.0, epsilon = 1e-15);
    }
    
    #[test]
    fn gaussian_pdf_single_sigma_empty_tensor() {
        let t = gaussian_pdf_single_sigma(1.0, &shape![0, 5]).unwrap();
        assert_eq!(t.len(), 0);
    }

    #[test]
    fn gaussian_pdf_multi_sigma_empty_tensor() {
        let t = gaussian_pdf_multi_sigma(vec![1.0, 1.0], &shape![2, 0]).unwrap();
        assert_eq!(t.len(), 0);
    }

    #[test]
    fn gaussian_sample_empty_tensor() {
        let t = gaussian_sample(1.0, &shape![0], -1.0, 1.0).unwrap();
        assert_eq!(t.len(), 0);
    }

    #[test]
    fn gaussian_pdf_cov_mat_empty_tensor() {
        let cov = eye(2);
        let t = gaussian_pdf_cov_mat(cov, &shape![2, 0]).unwrap();
        assert_eq!(t.len(), 0);
    }
}
