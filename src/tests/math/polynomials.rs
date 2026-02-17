#[cfg(test)]
mod polynomials_tests {
    use std::ops::Add;
    use float_cmp::assert_approx_eq;
    use num::complex::{Complex64, ComplexFloat};
    use crate::definitions::errors::TensorErrors;
    use crate::math::polynomials::{solve_cubic, solve_quadratic, solve_quartic};

    #[test]
    fn leading_coefficient_is_zero() {
        let c1 = [
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let c2 = [
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let c3 = [
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];

        let err = solve_quadratic(&c1).unwrap_err();
        match err {
            TensorErrors::PolynomialLeadingCoefficientZero => {},
            _ => panic!("Incorrect error"),
        }

        let err = solve_cubic(&c2).unwrap_err();
        match err {
            TensorErrors::PolynomialLeadingCoefficientZero => {},
            _ => panic!("Incorrect error"),
        }

        let err = solve_quartic(&c3).unwrap_err();
        match err {
            TensorErrors::PolynomialLeadingCoefficientZero => {},
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn solve_quadratic_poly() {
        let coefficients = [
            Complex64 { re: 1.0, im: 2.0 },
            Complex64 { re: 0.0, im: -2.0 },
            Complex64 { re: 5.0, im: 0.0 },
        ];
        let roots = solve_quadratic(&coefficients).unwrap();

        assert_eq!(roots.len(), 2);
        assert_approx_eq!(f64, coefficients.iter().enumerate().map(|(i, c)| c * roots[0].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 1e-15);
        assert_approx_eq!(f64, coefficients.iter().enumerate().map(|(i, c)| c * roots[1].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn solve_cubic_poly() {
        let coefficients = [
            Complex64 { re: 4.0, im: 0.0 },
            Complex64 { re: 3.0, im: 0.0 },
            Complex64 { re: 2.0, im: 0.0 },
            Complex64 { re: 1.0, im: 0.0 },
        ];
        let roots = solve_cubic(&coefficients).unwrap();

        assert_eq!(roots.len(), 3);
        assert_approx_eq!(f64, coefficients.iter().enumerate().map(|(i, c)| c * roots[0].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 2e-15);
        assert_approx_eq!(f64, coefficients.iter().enumerate().map(|(i, c)| c * roots[1].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 2e-15);
        assert_approx_eq!(f64, coefficients.iter().enumerate().map(|(i, c)| c * roots[2].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 2e-15);

        let coefficients2 = [
            Complex64 { re: 45.0, im: -2.0 },
            Complex64 { re: -3.0, im: 4.0 },
            Complex64 { re: 2.0, im: 3.0 },
            Complex64 { re: 1.0, im: 10.0 },
        ];
        let roots2 = solve_cubic(&coefficients2).unwrap();

        assert_eq!(roots2.len(), 3);
        assert_approx_eq!(f64, coefficients2.iter().enumerate().map(|(i, c)| c * roots2[0].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 2e-11);
        assert_approx_eq!(f64, coefficients2.iter().enumerate().map(|(i, c)| c * roots2[1].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 2e-11);
        assert_approx_eq!(f64, coefficients2.iter().enumerate().map(|(i, c)| c * roots2[2].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 2e-11);
    }

    #[test]
    fn solve_quartic_poly() {
        let coefficients = [
            Complex64 { re: 4.5, im: -2.0 },
            Complex64 { re: 3.4, im: 5.0 },
            Complex64 { re: 2.1, im: 3.0 },
            Complex64 { re: 1.2, im: 2.0 },
            Complex64 { re: 1.1, im: -10.0 },
        ];

        let roots = solve_quartic(&coefficients).unwrap();
        assert_eq!(roots.len(), 4);

        assert_approx_eq!(f64, coefficients.iter().enumerate().map(|(i, c)| c * roots[0].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 2e-10);
        assert_approx_eq!(f64, coefficients.iter().enumerate().map(|(i, c)| c * roots[1].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 2e-10);
        assert_approx_eq!(f64, coefficients.iter().enumerate().map(|(i, c)| c * roots[2].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 2e-10);
        assert_approx_eq!(f64, coefficients.iter().enumerate().map(|(i, c)| c * roots[3].powi(i as i32)).reduce(Complex64::add).unwrap().abs(), 0.0, epsilon = 2e-10);
    }
}