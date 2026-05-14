#[cfg(test)]
mod f64_c64_ops_tests {
    use crate::definitions::errors::TensorErrors;
    use crate::definitions::matrix::Matrix;
    use crate::definitions::shape::Shape;
    use crate::definitions::tensor::Tensor;
    use crate::{shape, transpose};
    use float_cmp::{approx_eq, assert_approx_eq};
    use num::complex::{Complex64, ComplexFloat};
    use crate::definitions::traits::{IntoMatrix, IntoTensor};
    use crate::definitions::transpose::Transpose;

    fn setup() -> (Tensor<f64>, Tensor<Complex64>, Matrix<f64>, Matrix<Complex64>) {
        let t1 = Tensor::new(
            &shape![2, 3, 4],
            (0..24).map(f64::from).collect(),
        ).unwrap();
        let t2 = t1.map_refs(|&x| Complex64::new(x, x));

        let m1 = Matrix::new(
            6, 4,
            (0..24).map(f64::from).collect(),
        ).unwrap();
        let m2 = m1.map_refs(|&x| Complex64::new(x, -x));

        (t1, t2, m1, m2)
    }

    #[test]
    fn re() {
        let (_, t2, _, m2) = setup();

        let ans = (0..24).map(f64::from).collect::<Matrix<f64>>().reshape(m2.rows, m2.cols).unwrap();

        assert_eq!(ans, m2.re());
        assert_eq!(ans.into_tensor().reshape(t2.shape()).unwrap(), t2.re());
    }

    #[test]
    fn im() {
        let (_, t2, _, m2) = setup();

        let ans = (0..24).map(f64::from).collect::<Matrix<f64>>().reshape(m2.rows, m2.cols).unwrap();

        assert_eq!(ans, m2.im() * -1.0);
        assert_eq!(ans.into_tensor().reshape(t2.shape()).unwrap(), t2.im());
    }

    #[test]
    fn into_complex() {
        let (m1, _, t1, _) = setup();

        assert_eq!(m1.clone(), m1.into_complex().re());
        assert_eq!(t1.clone(), t1.into_complex().re());
    }

    #[test]
    fn test_trace() {
        let m1 = Matrix::<i32>::new(2, 2, (0..4).collect()).unwrap();
        assert_eq!(3, m1.trace().unwrap());
    }

    #[test]
    fn invalid_trace_non_square() {
        let m1 = Matrix::<i32>::new(2, 3, (0..6).collect()).unwrap();
        let err = m1.trace().unwrap_err();

        match err {
            TensorErrors::NonSquareMatrix => {},
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn normalised() {
        let t1 = Tensor::<f64>::new(
            &shape![3, 3, 3],
            (0..27).map(|x| x as f64).collect::<Vec<f64>>(),
        )
            .unwrap();

        let t1_norm_l1 = t1.clone().norm_l1();
        let t1_norm_l2 = t1.norm_l2();

        let t1_norm_l1_ans = Tensor::<f64>::new(
            &shape![3, 3, 3],
            (0..27).map(|x| (x as f64) / 351.0).collect::<Vec<f64>>(),
        )
            .unwrap();
        let t1_norm_l2_ans = Tensor::<f64>::new(
            &shape![3, 3, 3],
            (0..27)
                .map(|x| (x as f64) / ComplexFloat::sqrt(6201.0))
                .collect::<Vec<f64>>(),
        )
            .unwrap();

        assert_eq!(t1_norm_l1, t1_norm_l1_ans);
        assert_eq!(t1_norm_l2, t1_norm_l2_ans);

        let t2 = Tensor::<Complex64>::new(
            &&shape![2, 3],
            (0..6)
                .map(|x| Complex64 {
                    re: x as f64,
                    im: x as f64,
                })
                .collect(),
        )
            .unwrap();

        let t2_norm_l1 = t2.clone().norm_l1();
        let t2_norm_l2 = t2.norm_l2();

        let t2_norm_l1_ans = Tensor::<Complex64>::new(
            &shape![2, 3],
            (0..6)
                .map(|x| Complex64 {
                    re: (x as f64) / (15.0 * f64::sqrt(2.0)),
                    im: (x as f64) / (15.0 * f64::sqrt(2.0)),
                })
                .collect(),
        )
            .unwrap();
        let t2_norm_l2_ans = Tensor::<Complex64>::new(
            &shape![2, 3],
            (0..6)
                .map(|x| Complex64 {
                    re: (x as f64) / ComplexFloat::sqrt(110.0),
                    im: (x as f64) / ComplexFloat::sqrt(110.0),
                })
                .collect(),
        )
            .unwrap();

        assert_approx_eq!(
            f64,
            (t2_norm_l1_ans - t2_norm_l1)
                .map(Complex64::abs)
                .sum(),
            0.0,
            epsilon = 1e-15
        );
        assert_approx_eq!(
            f64,
            (t2_norm_l2_ans - t2_norm_l2)
                .map(Complex64::abs)
                .sum(),
            0.0,
            epsilon = 1e-15
        );
    }

    #[test]
    fn exp() {
        let (t1, t2, m1, m2) = setup();

        let (ans1, ans2, ans3, ans4) = (
            t1.elements.iter().map(|&x| f64::exp(x)).collect::<Tensor<_>>().reshape(t1.shape()).unwrap(),
            t2.elements.iter().map(|&x| Complex64::exp(x)).collect::<Tensor<_>>().reshape(t2.shape()).unwrap(),
            m1.elements.iter().map(|&x| f64::exp(x)).collect::<Matrix<_>>().reshape(m1.rows, m1.cols).unwrap(),
            m2.elements.iter().map(|&x| Complex64::exp(x)).collect::<Matrix<_>>().reshape(m2.rows, m2.cols).unwrap(),
        );

        assert!(approx_eq!(Tensor<f64>, t1.exp(), ans1, epsilon = 1e-15));
        assert!(approx_eq!(Tensor<Complex64>, t2.exp(), ans2, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<f64>, m1.exp(), ans3, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<Complex64>, m2.exp(), ans4, epsilon = 1e-15));
    }

    #[test]
    fn ln() {
        let (t1, t2, m1, m2) = setup();

        let (ans1, ans2, ans3, ans4) = (
            t1.elements.iter().map(|&x| f64::ln(x)).collect::<Tensor<_>>().reshape(t1.shape()).unwrap(),
            t2.elements.iter().map(|&x| Complex64::ln(x)).collect::<Tensor<_>>().reshape(t2.shape()).unwrap(),
            m1.elements.iter().map(|&x| f64::ln(x)).collect::<Matrix<_>>().reshape(m1.rows, m1.cols).unwrap(),
            m2.elements.iter().map(|&x| Complex64::ln(x)).collect::<Matrix<_>>().reshape(m2.rows, m2.cols).unwrap(),
        );

        assert!(approx_eq!(Tensor<f64>, t1.ln(), ans1, epsilon = 1e-15));
        assert!(approx_eq!(Tensor<Complex64>, t2.ln(), ans2, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<f64>, m1.ln(), ans3, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<Complex64>, m2.ln(), ans4, epsilon = 1e-15));
    }

    #[test]
    fn log() {
        let (t1, t2, m1, m2) = setup();

        let (ans1, ans2, ans3, ans4) = (
            t1.elements.iter().map(|&x| f64::log(x, 5.0)).collect::<Tensor<_>>().reshape(t1.shape()).unwrap(),
            t2.elements.iter().map(|&x| Complex64::log(x, 5.0)).collect::<Tensor<_>>().reshape(t2.shape()).unwrap(),
            m1.elements.iter().map(|&x| f64::log(x, 5.0)).collect::<Matrix<_>>().reshape(m1.rows, m1.cols).unwrap(),
            m2.elements.iter().map(|&x| Complex64::log(x, 5.0)).collect::<Matrix<_>>().reshape(m2.rows, m2.cols).unwrap(),
        );

        assert!(approx_eq!(Tensor<f64>, t1.log(5.0), ans1, epsilon = 1e-15));
        assert!(approx_eq!(Tensor<Complex64>, t2.log(5.0), ans2, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<f64>, m1.log(5.0), ans3, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<Complex64>, m2.log(5.0), ans4, epsilon = 1e-15));
    }

    #[test]
    fn log2() {
        let (t1, t2, m1, m2) = setup();

        let (ans1, ans2, ans3, ans4) = (
            t1.elements.iter().map(|&x| f64::log2(x)).collect::<Tensor<_>>().reshape(t1.shape()).unwrap(),
            t2.elements.iter().map(|&x| Complex64::log2(x)).collect::<Tensor<_>>().reshape(t2.shape()).unwrap(),
            m1.elements.iter().map(|&x| f64::log2(x)).collect::<Matrix<_>>().reshape(m1.rows, m1.cols).unwrap(),
            m2.elements.iter().map(|&x| Complex64::log2(x)).collect::<Matrix<_>>().reshape(m2.rows, m2.cols).unwrap(),
        );

        assert!(approx_eq!(Tensor<f64>, t1.log2(), ans1, epsilon = 1e-15));
        assert!(approx_eq!(Tensor<Complex64>, t2.log2(), ans2, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<f64>, m1.log2(), ans3, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<Complex64>, m2.log2(), ans4, epsilon = 1e-15));
    }

    #[test]
    fn log10() {
        let (t1, t2, m1, m2) = setup();

        let (ans1, ans2, ans3, ans4) = (
            t1.elements.iter().map(|&x| f64::log10(x)).collect::<Tensor<_>>().reshape(t1.shape()).unwrap(),
            t2.elements.iter().map(|&x| Complex64::log10(x)).collect::<Tensor<_>>().reshape(t2.shape()).unwrap(),
            m1.elements.iter().map(|&x| f64::log10(x)).collect::<Matrix<_>>().reshape(m1.rows, m1.cols).unwrap(),
            m2.elements.iter().map(|&x| Complex64::log10(x)).collect::<Matrix<_>>().reshape(m2.rows, m2.cols).unwrap(),
        );

        assert!(approx_eq!(Tensor<f64>, t1.log10(), ans1, epsilon = 1e-15));
        assert!(approx_eq!(Tensor<Complex64>, t2.log10(), ans2, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<f64>, m1.log10(), ans3, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<Complex64>, m2.log10(), ans4, epsilon = 1e-15));
    }

    #[test]
    fn exp_base_n() {
        let (t1, t2, m1, m2) = setup();

        let (ans1, ans2, ans3, ans4) = (
            t1.elements.iter().map(|&x| f64::powf(2.0, x)).collect::<Tensor<_>>().reshape(t1.shape()).unwrap(),
            t2.elements.iter().map(|&x| Complex64::powc(2.0.into(), x)).collect::<Tensor<_>>().reshape(t2.shape()).unwrap(),
            m1.elements.iter().map(|&x| f64::powf(2.0, x)).collect::<Matrix<_>>().reshape(m1.rows, m1.cols).unwrap(),
            m2.elements.iter().map(|&x| Complex64::powc(2.0.into(), x)).collect::<Matrix<_>>().reshape(m2.rows, m2.cols).unwrap(),
        );

        assert!(approx_eq!(Tensor<f64>, t1.exp_base_n(2.0), ans1, epsilon = 1e-15));
        assert!(approx_eq!(Tensor<Complex64>, t2.exp_base_n(2.0.into()), ans2, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<f64>, m1.exp_base_n(2.0), ans3, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<Complex64>, m2.exp_base_n(2.0.into()), ans4, epsilon = 1e-15));
    }

    #[test]
    fn pow() {
        let (t1, t2, m1, m2) = setup();

        let (ans1, ans2, ans3, ans4) = (
            t1.elements.iter().map(|&x| f64::powf(x, 1.5)).collect::<Tensor<_>>().reshape(t1.shape()).unwrap(),
            t2.elements.iter().map(|&x| Complex64::powf(x, 1.5)).collect::<Tensor<_>>().reshape(t2.shape()).unwrap(),
            m1.elements.iter().map(|&x| f64::powf(x, 1.5)).collect::<Matrix<_>>().reshape(m1.rows, m1.cols).unwrap(),
            m2.elements.iter().map(|&x| Complex64::powf(x, 1.5)).collect::<Matrix<_>>().reshape(m2.rows, m2.cols).unwrap(),
        );

        assert!(approx_eq!(Tensor<f64>, t1.pow(1.5), ans1, epsilon = 1e-10));
        assert!(approx_eq!(Tensor<Complex64>, t2.pow(1.5.into()), ans2, epsilon = 1e-10));
        assert!(approx_eq!(Matrix<f64>, m1.pow(1.5), ans3, epsilon = 1e-10));
        assert!(approx_eq!(Matrix<Complex64>, m2.pow(1.5.into()), ans4, epsilon = 1e-10));
    }

    #[test]
    fn sin() {
        let (t1, t2, m1, m2) = setup();

        let (ans1, ans2, ans3, ans4) = (
            t1.elements.iter().map(|&x| f64::sin(x)).collect::<Tensor<_>>().reshape(t1.shape()).unwrap(),
            t2.elements.iter().map(|&x| Complex64::sin(x)).collect::<Tensor<_>>().reshape(t2.shape()).unwrap(),
            m1.elements.iter().map(|&x| f64::sin(x)).collect::<Matrix<_>>().reshape(m1.rows, m1.cols).unwrap(),
            m2.elements.iter().map(|&x| Complex64::sin(x)).collect::<Matrix<_>>().reshape(m2.rows, m2.cols).unwrap(),
        );

        assert!(approx_eq!(Tensor<f64>, t1.sin(), ans1, epsilon = 1e-15));
        assert!(approx_eq!(Tensor<Complex64>, t2.sin(), ans2, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<f64>, m1.sin(), ans3, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<Complex64>, m2.sin(), ans4, epsilon = 1e-15));
    }

    #[test]
    fn cos() {
        let (t1, t2, m1, m2) = setup();

        let (ans1, ans2, ans3, ans4) = (
            t1.elements.iter().map(|&x| f64::cos(x)).collect::<Tensor<_>>().reshape(t1.shape()).unwrap(),
            t2.elements.iter().map(|&x| Complex64::cos(x)).collect::<Tensor<_>>().reshape(t2.shape()).unwrap(),
            m1.elements.iter().map(|&x| f64::cos(x)).collect::<Matrix<_>>().reshape(m1.rows, m1.cols).unwrap(),
            m2.elements.iter().map(|&x| Complex64::cos(x)).collect::<Matrix<_>>().reshape(m2.rows, m2.cols).unwrap(),
        );

        assert!(approx_eq!(Tensor<f64>, t1.cos(), ans1, epsilon = 1e-15));
        assert!(approx_eq!(Tensor<Complex64>, t2.cos(), ans2, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<f64>, m1.cos(), ans3, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<Complex64>, m2.cos(), ans4, epsilon = 1e-15));
    }

    #[test]
    fn tan() {
        let (t1, t2, m1, m2) = setup();

        let (ans1, ans2, ans3, ans4) = (
            t1.elements.iter().map(|&x| f64::tan(x)).collect::<Tensor<_>>().reshape(t1.shape()).unwrap(),
            t2.elements.iter().map(|&x| Complex64::tan(x)).collect::<Tensor<_>>().reshape(t2.shape()).unwrap(),
            m1.elements.iter().map(|&x| f64::tan(x)).collect::<Matrix<_>>().reshape(m1.rows, m1.cols).unwrap(),
            m2.elements.iter().map(|&x| Complex64::tan(x)).collect::<Matrix<_>>().reshape(m2.rows, m2.cols).unwrap(),
        );

        assert!(approx_eq!(Tensor<f64>, t1.tan(), ans1, epsilon = 1e-15));
        assert!(approx_eq!(Tensor<Complex64>, t2.tan(), ans2, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<f64>, m1.tan(), ans3, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<Complex64>, m2.tan(), ans4, epsilon = 1e-15));
    }

    #[test]
    fn asin() {
        let (t1, t2, m1, m2) = setup();

        let (ans1, ans2, ans3, ans4) = (
            t1.elements.iter().map(|&x| x.tanh().asin()).collect::<Tensor<_>>().reshape(t1.shape()).unwrap(),
            t2.elements.iter().map(|&x| x.tanh().asin()).collect::<Tensor<_>>().reshape(t2.shape()).unwrap(),
            m1.elements.iter().map(|&x| x.tanh().asin()).collect::<Matrix<_>>().reshape(m1.rows, m1.cols).unwrap(),
            m2.elements.iter().map(|&x| x.tanh().asin()).collect::<Matrix<_>>().reshape(m2.rows, m2.cols).unwrap(),
        );

        assert!(approx_eq!(Tensor<f64>, t1.tanh().asin(), ans1, epsilon = 1e-15));
        assert!(approx_eq!(Tensor<Complex64>, t2.tanh().asin(), ans2, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<f64>, m1.tanh().asin(), ans3, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<Complex64>, m2.tanh().asin(), ans4, epsilon = 1e-15));
    }

    #[test]
    fn acos() {
        let (t1, t2, m1, m2) = setup();

        let (ans1, ans2, ans3, ans4) = (
            t1.elements.iter().map(|&x| x.tanh().acos()).collect::<Tensor<_>>().reshape(t1.shape()).unwrap(),
            t2.elements.iter().map(|&x| x.tanh().acos()).collect::<Tensor<_>>().reshape(t2.shape()).unwrap(),
            m1.elements.iter().map(|&x| x.tanh().acos()).collect::<Matrix<_>>().reshape(m1.rows, m1.cols).unwrap(),
            m2.elements.iter().map(|&x| x.tanh().acos()).collect::<Matrix<_>>().reshape(m2.rows, m2.cols).unwrap(),
        );

        assert!(approx_eq!(Tensor<f64>, t1.tanh().acos(), ans1, epsilon = 1e-15));
        assert!(approx_eq!(Tensor<Complex64>, t2.tanh().acos(), ans2, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<f64>, m1.tanh().acos(), ans3, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<Complex64>, m2.tanh().acos(), ans4, epsilon = 1e-15));
    }

    #[test]
    fn atan() {
        let (t1, t2, m1, m2) = setup();

        let (ans1, ans2, ans3, ans4) = (
            t1.elements.iter().map(|&x| f64::atan(x)).collect::<Tensor<_>>().reshape(t1.shape()).unwrap(),
            t2.elements.iter().map(|&x| Complex64::atan(x)).collect::<Tensor<_>>().reshape(t2.shape()).unwrap(),
            m1.elements.iter().map(|&x| f64::atan(x)).collect::<Matrix<_>>().reshape(m1.rows, m1.cols).unwrap(),
            m2.elements.iter().map(|&x| Complex64::atan(x)).collect::<Matrix<_>>().reshape(m2.rows, m2.cols).unwrap(),
        );

        assert!(approx_eq!(Tensor<f64>, t1.atan(), ans1, epsilon = 1e-15));
        assert!(approx_eq!(Tensor<Complex64>, t2.atan(), ans2, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<f64>, m1.atan(), ans3, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<Complex64>, m2.atan(), ans4, epsilon = 1e-15));
    }

    #[test]
    fn recip() {
        let (t1, t2, m1, m2) = setup();

        let (ans1, ans2, ans3, ans4) = (
            t1.elements.iter().map(|&x| f64::recip(x + 1.0)).collect::<Tensor<_>>().reshape(t1.shape()).unwrap(),
            t2.elements.iter().map(|&x| Complex64::recip(x + Complex64::ONE)).collect::<Tensor<_>>().reshape(t2.shape()).unwrap(),
            m1.elements.iter().map(|&x| f64::recip(x + 1.0)).collect::<Matrix<_>>().reshape(m1.rows, m1.cols).unwrap(),
            m2.elements.iter().map(|&x| Complex64::recip(x + Complex64::ONE)).collect::<Matrix<_>>().reshape(m2.rows, m2.cols).unwrap(),
        );

        assert!(approx_eq!(Tensor<f64>, t1.map(|x| x + 1.0).recip(), ans1, epsilon = 1e-15));
        assert!(approx_eq!(Tensor<Complex64>, t2.map(|x| x + Complex64::ONE).recip(), ans2, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<f64>, m1.map(|x| x + 1.0).recip(), ans3, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<Complex64>, m2.map(|x| x + Complex64::ONE).recip(), ans4, epsilon = 1e-15));
    }

    #[test]
    fn sinh() {
        let (t1, t2, m1, m2) = setup();

        let (ans1, ans2, ans3, ans4) = (
            t1.elements.iter().map(|&x| f64::sinh(x)).collect::<Tensor<_>>().reshape(t1.shape()).unwrap(),
            t2.elements.iter().map(|&x| Complex64::sinh(x)).collect::<Tensor<_>>().reshape(t2.shape()).unwrap(),
            m1.elements.iter().map(|&x| f64::sinh(x)).collect::<Matrix<_>>().reshape(m1.rows, m1.cols).unwrap(),
            m2.elements.iter().map(|&x| Complex64::sinh(x)).collect::<Matrix<_>>().reshape(m2.rows, m2.cols).unwrap(),
        );

        assert!(approx_eq!(Tensor<f64>, t1.sinh(), ans1, epsilon = 1e-15));
        assert!(approx_eq!(Tensor<Complex64>, t2.sinh(), ans2, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<f64>, m1.sinh(), ans3, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<Complex64>, m2.sinh(), ans4, epsilon = 1e-15));
    }

    #[test]
    fn cosh() {
        let (t1, t2, m1, m2) = setup();

        let (ans1, ans2, ans3, ans4) = (
            t1.elements.iter().map(|&x| f64::cosh(x)).collect::<Tensor<_>>().reshape(t1.shape()).unwrap(),
            t2.elements.iter().map(|&x| Complex64::cosh(x)).collect::<Tensor<_>>().reshape(t2.shape()).unwrap(),
            m1.elements.iter().map(|&x| f64::cosh(x)).collect::<Matrix<_>>().reshape(m1.rows, m1.cols).unwrap(),
            m2.elements.iter().map(|&x| Complex64::cosh(x)).collect::<Matrix<_>>().reshape(m2.rows, m2.cols).unwrap(),
        );

        assert!(approx_eq!(Tensor<f64>, t1.cosh(), ans1, epsilon = 1e-15));
        assert!(approx_eq!(Tensor<Complex64>, t2.cosh(), ans2, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<f64>, m1.cosh(), ans3, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<Complex64>, m2.cosh(), ans4, epsilon = 1e-15));
    }

    #[test]
    fn tanh() {
        let (t1, t2, m1, m2) = setup();

        let (ans1, ans2, ans3, ans4) = (
            t1.elements.iter().map(|&x| f64::tanh(x)).collect::<Tensor<_>>().reshape(t1.shape()).unwrap(),
            t2.elements.iter().map(|&x| Complex64::tanh(x)).collect::<Tensor<_>>().reshape(t2.shape()).unwrap(),
            m1.elements.iter().map(|&x| f64::tanh(x)).collect::<Matrix<_>>().reshape(m1.rows, m1.cols).unwrap(),
            m2.elements.iter().map(|&x| Complex64::tanh(x)).collect::<Matrix<_>>().reshape(m2.rows, m2.cols).unwrap(),
        );

        assert!(approx_eq!(Tensor<f64>, t1.tanh(), ans1, epsilon = 1e-15));
        assert!(approx_eq!(Tensor<Complex64>, t2.tanh(), ans2, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<f64>, m1.tanh(), ans3, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<Complex64>, m2.tanh(), ans4, epsilon = 1e-15));
    }

    #[test]
    fn asinh() {
        let (t1, t2, m1, m2) = setup();

        let (ans1, ans2, ans3, ans4) = (
            t1.elements.iter().map(|&x| f64::asinh(x)).collect::<Tensor<_>>().reshape(t1.shape()).unwrap(),
            t2.elements.iter().map(|&x| Complex64::asinh(x)).collect::<Tensor<_>>().reshape(t2.shape()).unwrap(),
            m1.elements.iter().map(|&x| f64::asinh(x)).collect::<Matrix<_>>().reshape(m1.rows, m1.cols).unwrap(),
            m2.elements.iter().map(|&x| Complex64::asinh(x)).collect::<Matrix<_>>().reshape(m2.rows, m2.cols).unwrap(),
        );

        assert!(approx_eq!(Tensor<f64>, t1.asinh(), ans1, epsilon = 1e-15));
        assert!(approx_eq!(Tensor<Complex64>, t2.asinh(), ans2, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<f64>, m1.asinh(), ans3, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<Complex64>, m2.asinh(), ans4, epsilon = 1e-15));
    }

    #[test]
    fn acosh() {
        let (t1, t2, m1, m2) = setup();

        let (ans1, ans2, ans3, ans4) = (
            t1.elements.iter().map(|&x| (x + 1.0).acosh()).collect::<Tensor<_>>().reshape(t1.shape()).unwrap(),
            t2.elements.iter().map(|&x| (x + 1.0).acosh()).collect::<Tensor<_>>().reshape(t2.shape()).unwrap(),
            m1.elements.iter().map(|&x| (x + 1.0).acosh()).collect::<Matrix<_>>().reshape(m1.rows, m1.cols).unwrap(),
            m2.elements.iter().map(|&x| (x + 1.0).acosh()).collect::<Matrix<_>>().reshape(m2.rows, m2.cols).unwrap(),
        );

        assert!(approx_eq!(Tensor<f64>, (t1 + 1.0).acosh(), ans1, epsilon = 1e-15));
        assert!(approx_eq!(Tensor<Complex64>, (t2 + Complex64::ONE).acosh(), ans2, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<f64>, (m1 + 1.0).acosh(), ans3, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<Complex64>, (m2 + Complex64::ONE).acosh(), ans4, epsilon = 1e-15));
    }

    #[test]
    fn atanh() {
        let (t1, t2, m1, m2) = setup();

        let (ans1, ans2, ans3, ans4) = (
            t1.elements.iter().map(|&x| f64::atanh(x)).collect::<Tensor<_>>().reshape(t1.shape()).unwrap(),
            t2.elements.iter().map(|&x| Complex64::atanh(x)).collect::<Tensor<_>>().reshape(t2.shape()).unwrap(),
            m1.elements.iter().map(|&x| f64::atanh(x)).collect::<Matrix<_>>().reshape(m1.rows, m1.cols).unwrap(),
            m2.elements.iter().map(|&x| Complex64::atanh(x)).collect::<Matrix<_>>().reshape(m2.rows, m2.cols).unwrap(),
        );

        assert!(approx_eq!(Tensor<f64>, t1.atanh(), ans1, epsilon = 1e-15));
        assert!(approx_eq!(Tensor<Complex64>, t2.atanh(), ans2, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<f64>, m1.atanh(), ans3, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<Complex64>, m2.atanh(), ans4, epsilon = 1e-15));
    }

    #[test]
    fn sigmoid() {
        let (t1, _, m1, _) = setup();

        let ans = vec![
            6.14417460e-06, 1.67014218e-05, 4.53978687e-05, 1.23394576e-04,
            3.35350130e-04, 9.11051194e-04, 2.47262316e-03, 6.69285092e-03,
            1.79862100e-02, 4.74258732e-02, 1.19202922e-01, 2.68941421e-01,
            5.00000000e-01, 7.31058579e-01, 8.80797078e-01, 9.52574127e-01,
            9.82013790e-01, 9.93307149e-01, 9.97527377e-01, 9.99088949e-01,
            9.99664650e-01, 9.99876605e-01, 9.99954602e-01, 9.99983299e-01,
        ];

        assert!(approx_eq!(Tensor<f64>, (t1.clone() - 12.0).sigmoid(), ans.clone().into_tensor().reshape(t1.shape()).unwrap(), epsilon = 1e-5));
        assert!(approx_eq!(Matrix<f64>, (m1.clone() - 12.0).sigmoid(), ans.into_matrix().reshape(m1.rows, m1.cols).unwrap(), epsilon = 1e-5));
    }

    #[test]
    fn relu() {
        let (t1, _, m1, _) = setup();

        let (ans1, ans2) = (
            t1.elements.iter().map(|&x| f64::max(f64::atanh(x), 0.0)).collect::<Tensor<_>>().reshape(t1.shape()).unwrap(),
            m1.elements.iter().map(|&x| f64::max(f64::atanh(x), 0.0)).collect::<Matrix<_>>().reshape(m1.rows, m1.cols).unwrap(),
        );

        assert!(approx_eq!(Tensor<f64>, t1.atanh().relu(), ans1, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<f64>, m1.atanh().relu(), ans2, epsilon = 1e-15));
    }

    #[test]
    fn leaky_relu() {
        let (t1, _, m1, _) = setup();

        let (ans1, ans2) = (
            t1.elements.iter().map(|&x| if x > 0.0 { x } else { 0.01 * x }).collect::<Tensor<_>>().reshape(t1.shape()).unwrap(),
            m1.elements.iter().map(|&x| if x > 0.0 { x } else { 0.01 * x }).collect::<Matrix<_>>().reshape(m1.rows, m1.cols).unwrap(),
        );

        assert!(approx_eq!(Tensor<f64>, t1.leaky_relu(0.01), ans1, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<f64>, m1.leaky_relu(0.01), ans2, epsilon = 1e-15));
    }

    #[test]
    fn softmax() {
        let (t1, _, m1, _) = setup();
        let t1_shape = t1.shape().clone();
        let (m1_rows, m1_cols) = (m1.rows(), m1.cols());

        let ans = vec![
            6.48674509e-11, 1.76328013e-10, 4.79309234e-10, 1.30289758e-09,
            3.54164282e-09, 9.62718331e-09, 2.61693974e-08, 7.11357975e-08,
            1.93367146e-07, 5.25626399e-07, 1.42880069e-06, 3.88388295e-06,
            1.05574884e-05, 2.86982290e-05, 7.80098743e-05, 2.12052824e-04,
            5.76419338e-04, 1.56687021e-03, 4.25919482e-03, 1.15776919e-02,
            3.14714295e-02, 8.55482149e-02, 2.32544158e-01, 6.32120559e-01
        ];

        assert!(approx_eq!(Tensor<f64>, t1.softmax(), ans.clone().into_tensor().reshape(&t1_shape).unwrap(), epsilon = 1e-5));
        assert!(approx_eq!(Matrix<f64>, m1.softmax(), ans.clone().into_matrix().reshape(m1_rows, m1_cols).unwrap(), epsilon = 1e-5));
    }

    #[test]
    fn sqrt() {
        let (t1, t2, m1, m2) = setup();

        let (ans1, ans2, ans3, ans4) = (
            t1.elements.iter().map(|&x| f64::sqrt(x)).collect::<Tensor<_>>().reshape(t1.shape()).unwrap(),
            t2.elements.iter().map(|&x| Complex64::sqrt(x)).collect::<Tensor<_>>().reshape(t2.shape()).unwrap(),
            m1.elements.iter().map(|&x| f64::sqrt(x)).collect::<Matrix<_>>().reshape(m1.rows, m1.cols).unwrap(),
            m2.elements.iter().map(|&x| Complex64::sqrt(x)).collect::<Matrix<_>>().reshape(m2.rows, m2.cols).unwrap(),
        );

        assert!(approx_eq!(Tensor<f64>, t1.sqrt(), ans1, epsilon = 1e-15));
        assert!(approx_eq!(Tensor<Complex64>, t2.sqrt(), ans2, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<f64>, m1.sqrt(), ans3, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<Complex64>, m2.sqrt(), ans4, epsilon = 1e-15));
    }

    #[test]
    fn cbrt() {
        let (t1, t2, m1, m2) = setup();

        let (ans1, ans2, ans3, ans4) = (
            t1.elements.iter().map(|&x| f64::cbrt(x)).collect::<Tensor<_>>().reshape(t1.shape()).unwrap(),
            t2.elements.iter().map(|&x| Complex64::cbrt(x)).collect::<Tensor<_>>().reshape(t2.shape()).unwrap(),
            m1.elements.iter().map(|&x| f64::cbrt(x)).collect::<Matrix<_>>().reshape(m1.rows, m1.cols).unwrap(),
            m2.elements.iter().map(|&x| Complex64::cbrt(x)).collect::<Matrix<_>>().reshape(m2.rows, m2.cols).unwrap(),
        );

        assert!(approx_eq!(Tensor<f64>, t1.cbrt(), ans1, epsilon = 1e-15));
        assert!(approx_eq!(Tensor<Complex64>, t2.cbrt(), ans2, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<f64>, m1.cbrt(), ans3, epsilon = 1e-15));
        assert!(approx_eq!(Matrix<Complex64>, m2.cbrt(), ans4, epsilon = 1e-15));
    }

    #[test]
    fn norm_l1() {
        let (t1, t2, m1, m2) = setup();

        let (ans1, ans2, ans3, ans4) = (
            t1.elements.iter().map(|&x| x / 276.0).collect::<Tensor<_>>().reshape(t1.shape()).unwrap(),
            t2.elements.iter().map(|&x| x / (276.0 * 2.0.sqrt())).map(Complex64::from).collect::<Tensor<_>>().reshape(t2.shape()).unwrap(),
            m1.elements.iter().map(|&x| x / 276.0).collect::<Matrix<_>>().reshape(m1.rows, m1.cols).unwrap(),
            m2.elements.iter().map(|&x| x / (276.0 * 2.0.sqrt())).map(Complex64::from).collect::<Matrix<_>>().reshape(m2.rows, m2.cols).unwrap(),
        );

        assert!(approx_eq!(Tensor<f64>, t1.norm_l1(), ans1, epsilon = 1e-10));
        assert!(approx_eq!(Tensor<Complex64>, t2.norm_l1(), ans2, epsilon = 1e-10));
        assert!(approx_eq!(Matrix<f64>, m1.norm_l1(), ans3, epsilon = 1e-10));
        assert!(approx_eq!(Matrix<Complex64>, m2.norm_l1(), ans4, epsilon = 1e-10));
    }

    #[test]
    fn norm_l2() {
        let (t1, t2, m1, m2) = setup();

        let (ans1, ans2, ans3, ans4) = (
            t1.elements.iter().map(|&x| x / 4324.0.sqrt()).collect::<Tensor<_>>().reshape(t1.shape()).unwrap(),
            t2.elements.iter().map(|&x| x / 8648.0.sqrt()).collect::<Tensor<_>>().reshape(t2.shape()).unwrap(),
            m1.elements.iter().map(|&x| x / 4324.0.sqrt()).collect::<Matrix<_>>().reshape(m1.rows, m1.cols).unwrap(),
            m2.elements.iter().map(|&x| x / 8648.0.sqrt()).collect::<Matrix<_>>().reshape(m2.rows, m2.cols).unwrap(),
        );

        assert!(approx_eq!(Tensor<f64>, t1.norm_l2(), ans1, epsilon = 1e-10));
        assert!(approx_eq!(Tensor<Complex64>, t2.norm_l2(), ans2, epsilon = 1e-10));
        assert!(approx_eq!(Matrix<f64>, m1.norm_l2(), ans3, epsilon = 1e-10));
        assert!(approx_eq!(Matrix<Complex64>, m2.norm_l2(), ans4, epsilon = 1e-10));
    }

    #[test]
    fn conj() {
        let (_, t2, _, m2) = setup();

        let ans1 = t2.elements.iter().map(Complex64::conj).collect::<Tensor<_>>().reshape(t2.shape()).unwrap();
        let ans2 = m2.elements.iter().map(Complex64::conj).collect::<Matrix<_>>().reshape(m2.rows(), m2.cols()).unwrap();

        assert_eq!(ans1, t2.conj());
        assert_eq!(ans2, m2.conj());
    }

    #[test]
    fn conj_transpose() {
        let (_, t2, _, m2) = setup();

        let transpose = transpose![1, 0, 2];

        let ans1 = t2.elements.iter().map(Complex64::conj).collect::<Tensor<_>>().reshape(t2.shape()).unwrap().transpose(&transpose).unwrap();
        let ans2 = m2.elements.iter().map(Complex64::conj).collect::<Matrix<_>>().reshape(m2.rows(), m2.cols()).unwrap().transpose();

        assert_eq!(ans1, t2.conj_transpose(&transpose).unwrap());
        assert_eq!(ans1, t2.conj_transpose_mt(&transpose).unwrap());
        assert_eq!(ans2, m2.conj_transpose());
        assert_eq!(ans2, m2.conj_transpose_mt());
    }

    #[test]
    fn mag() {
        let (t1, t2, m1, m2) = setup();
        let (ans1, ans2, ans3, ans4) = (4324.0.sqrt(), 8648.0.sqrt(), 4324.0.sqrt(), 8648.0.sqrt());

        assert_eq!(ans1, t1.mag());
        assert_eq!(ans2, t2.mag());
        assert_eq!(ans3, m1.mag());
        assert_eq!(ans4, m2.mag());
    }

    #[test]
    fn mag_2() {
        let (t1, t2, m1, m2) = setup();
        let (ans1, ans2, ans3, ans4) = (4324.0, 8648.0, 4324.0, 8648.0);

        assert_eq!(ans1, t1.mag_2());
        assert_eq!(ans2, t2.mag_2());
        assert_eq!(ans3, m1.mag_2());
        assert_eq!(ans4, m2.mag_2());
    }

    #[test]
    fn abs() {
        let (t1, t2, m1, m2) = setup();

        assert_eq!(t1.clone().abs(), t1);
        assert_eq!(t2.clone().abs(), t2.elements.iter().map(|x| x.abs()).collect::<Tensor<_>>().reshape(t2.shape()).unwrap());
        assert_eq!(m1.clone().abs(), m1);
        assert_eq!(m2.clone().abs(), m2.elements.iter().map(|x| x.abs()).collect::<Matrix<_>>().reshape(m2.rows(), m2.cols()).unwrap());
    }

    #[test]
    fn born_probabilities() {
        let (t1, t2, m1, m2) = setup();
        let (ans1, ans2, ans3, ans4) = (
            t1.elements.iter().map(|x| x * x / 4324.0).collect::<Tensor<_>>().reshape(t1.shape()).unwrap(),
            t2.elements.iter().map(|x| (x * x).abs() / 8648.0).collect::<Tensor<_>>().reshape(t2.shape()).unwrap(),
            m1.elements.iter().map(|x| x * x / 4324.0).collect::<Matrix<_>>().reshape(m1.rows(), m1.cols()).unwrap(),
            m2.elements.iter().map(|x| (x * x).abs() / 8648.0).collect::<Matrix<_>>().reshape(m2.rows(), m2.cols()).unwrap(),
        );

        assert!(approx_eq!(Tensor<f64>, ans1, t1.born_probabilities()));
        assert!(approx_eq!(Tensor<f64>, ans2, t2.born_probabilities()));
        assert!(approx_eq!(Matrix<f64>, ans3, m1.born_probabilities()));
        assert!(approx_eq!(Matrix<f64>, ans4, m2.born_probabilities()));
    }
    
    #[test]
    fn rank_0_tensor_ops() {
        let val = Complex64::new(3.0, 4.0);
        let t = Tensor::new(&shape![], vec![val]).unwrap();

        // Real/Imag
        assert_eq!(t.clone().re().elements[0], 3.0);
        assert_eq!(t.clone().im().elements[0], 4.0);

        // Basic Math
        assert_approx_eq!(f64, t.clone().exp().elements[0].re, val.exp().re, epsilon = 1e-15);
        assert_approx_eq!(f64, t.clone().ln().elements[0].re, val.ln().re, epsilon = 1e-15);
        assert_approx_eq!(f64, t.clone().sin().elements[0].re, val.sin().re, epsilon = 1e-15);
        assert_approx_eq!(f64, t.clone().sqrt().elements[0].re, val.sqrt().re, epsilon = 1e-15);

        // Magnitude and Abs
        assert_approx_eq!(f64, t.clone().mag(), 5.0, epsilon = 1e-15);
        assert_approx_eq!(f64, t.clone().mag_2(), 25.0, epsilon = 1e-15);
        assert_approx_eq!(f64, t.clone().abs().elements[0], 5.0, epsilon = 1e-15);

        // Born probabilities (rank 0 sum is just the value itself / mag_2)
        let prob = t.clone().born_probabilities();
        assert_approx_eq!(f64, prob.elements[0], 1.0, epsilon = 1e-15);
    }
    
    #[test]
    fn empty_tensor_matrix_ops() {
        let t_empty = Tensor::<f64>::new(&shape![0], vec![]).unwrap();
        let m_empty = Matrix::<f64>::new(0, 0, vec![]).unwrap();

        // Mapping operations for f64
        assert_eq!(t_empty.clone().exp().elements.len(), 0);
        assert_eq!(t_empty.clone().ln().elements.len(), 0);
        assert_eq!(t_empty.clone().log(10.0).elements.len(), 0);
        assert_eq!(t_empty.clone().log2().elements.len(), 0);
        assert_eq!(t_empty.clone().log10().elements.len(), 0);
        assert_eq!(t_empty.clone().exp_base_n(2.0).elements.len(), 0);
        assert_eq!(t_empty.clone().pow(2.0).elements.len(), 0);
        assert_eq!(t_empty.clone().sin().elements.len(), 0);
        assert_eq!(t_empty.clone().cos().elements.len(), 0);
        assert_eq!(t_empty.clone().tan().elements.len(), 0);
        assert_eq!(t_empty.clone().asin().elements.len(), 0);
        assert_eq!(t_empty.clone().acos().elements.len(), 0);
        assert_eq!(t_empty.clone().atan().elements.len(), 0);
        assert_eq!(t_empty.clone().recip().elements.len(), 0);
        assert_eq!(t_empty.clone().sinh().elements.len(), 0);
        assert_eq!(t_empty.clone().cosh().elements.len(), 0);
        assert_eq!(t_empty.clone().tanh().elements.len(), 0);
        assert_eq!(t_empty.clone().asinh().elements.len(), 0);
        assert_eq!(t_empty.clone().acosh().elements.len(), 0);
        assert_eq!(t_empty.clone().atanh().elements.len(), 0);
        assert_eq!(t_empty.clone().sigmoid().elements.len(), 0);
        assert_eq!(t_empty.clone().relu().elements.len(), 0);
        assert_eq!(t_empty.clone().leaky_relu(0.1).elements.len(), 0);
        assert_eq!(t_empty.clone().sqrt().elements.len(), 0);
        assert_eq!(t_empty.clone().cbrt().elements.len(), 0);
        assert_eq!(t_empty.clone().abs().elements.len(), 0);
        assert_eq!(t_empty.clone().into_complex().elements.len(), 0);

        // Matrix f64 mapping
        assert_eq!(m_empty.clone().exp().elements.len(), 0);
        assert_eq!(m_empty.clone().sigmoid().elements.len(), 0);

        // Softmax on empty (nan sum usually, but check length)
        assert_eq!(t_empty.clone().softmax().elements.len(), 0);
        assert_eq!(m_empty.clone().softmax().elements.len(), 0);

        // Normalization on empty
        assert_eq!(t_empty.clone().norm_l1().elements.len(), 0);
        assert_eq!(t_empty.clone().norm_l2().elements.len(), 0);
        assert_eq!(t_empty.clone().born_probabilities().elements.len(), 0);

        // Reduction operations
        assert_eq!(t_empty.clone().mag(), 0.0);
        assert_eq!(t_empty.clone().mag_2(), 0.0);
        assert_eq!(m_empty.clone().mag(), 0.0);
        assert_eq!(m_empty.clone().mag_2(), 0.0);

        // Complex empty
        let t_c_empty = Tensor::<Complex64>::new(&shape![0, 5], vec![]).unwrap();
        let m_c_empty = Matrix::<Complex64>::new(0, 0, vec![]).unwrap();

        assert_eq!(t_c_empty.clone().re().elements.len(), 0);
        assert_eq!(t_c_empty.clone().im().elements.len(), 0);
        assert_eq!(t_c_empty.clone().exp().elements.len(), 0);
        assert_eq!(t_c_empty.clone().conj().elements.len(), 0);
        assert_eq!(t_c_empty.conj_transpose(&transpose![1, 0]).unwrap().elements.len(), 0);
        assert_eq!(m_c_empty.conj_transpose().elements.len(), 0);
        assert_eq!(t_c_empty.clone().mag(), 0.0);
        assert_eq!(t_c_empty.clone().mag_2(), 0.0);
        assert_eq!(t_c_empty.clone().abs().elements.len(), 0);
        assert_eq!(t_c_empty.clone().born_probabilities().elements.len(), 0);
    }

    #[test]
    fn test_nan_returns() {
        // 1. ln on negative f64
        let t_neg = Tensor::new(&shape![2], vec![-1.0, -2.0]).unwrap();
        assert!(t_neg.ln().elements.iter().all(|x| x.is_nan()));

        // 2. sqrt on negative f64
        let m_neg = Matrix::new(1, 2, vec![-4.0, -9.0]).unwrap();
        assert!(m_neg.sqrt().elements.iter().all(|x| x.is_nan()));

        // 3. inverse trig/hyperbolic out of range f64
        let t_range = Tensor::new(&shape![1], vec![2.0]).unwrap();
        assert!(t_range.clone().asin().elements[0].is_nan());
        assert!(t_range.acos().elements[0].is_nan());

        let t_acosh = Tensor::new(&shape![1], vec![0.5]).unwrap();
        assert!(t_acosh.acosh().elements[0].is_nan());

        // 4. Born probabilities on zero tensors (f64 and Complex64)
        let t_zero_f = Tensor::new(&shape![3], vec![0.0, 0.0, 0.0]).unwrap();
        assert!(t_zero_f.born_probabilities().elements.iter().all(|x| x.is_nan()));

        let t_zero_c = Tensor::new(&shape![2], vec![Complex64::ZERO, Complex64::ZERO]).unwrap();
        assert!(t_zero_c.born_probabilities().elements.iter().all(|x| x.is_nan()));

        // 5. Normalization on zeros
        let m_zero_f = Matrix::new(2, 2, vec![0.0; 4]).unwrap();
        assert!(m_zero_f.clone().norm_l1().elements.iter().all(|x| x.is_nan()));
        assert!(m_zero_f.norm_l2().elements.iter().all(|x| x.is_nan()));

        let m_zero_c = Matrix::new(2, 2, vec![Complex64::ZERO; 4]).unwrap();
        assert!(m_zero_c.norm_l1().elements.iter().all(|x| x.re.is_nan() && x.im.is_nan()));
    }
}
