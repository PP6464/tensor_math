use crate::definitions::errors::TensorErrors;
use float_cmp::approx_eq;
use num::complex::{Complex64, ComplexFloat};

/// Solves a quadratic. The coefficients are entered as a `&[Complex64; 3]` where the index of the
/// coefficient corresponds to the power of x, e.g. [1, 2, 3\] would be 1 + 2x + 3x².
pub fn solve_quadratic(coefficients: &[Complex64; 3]) -> Result<Vec<Complex64>, TensorErrors> {
    if approx_eq!(f64, coefficients[2].abs(), 0.0) {
        return Err(TensorErrors::PolynomialLeadingCoefficientZero);
    }

    let a = coefficients[2];
    let b = coefficients[1];
    let c = coefficients[0];

    let d = b.powi(2) - 4.0 * a * c;

    Ok(vec![
        (-b + d.sqrt()) / (2.0 * a),
        (-b - d.sqrt()) / (2.0 * a),
    ])
}

/// Solves a cubic. The coefficients are entered as a `&[Complex64; 4]` where the index of the
/// coefficient corresponds to the power of x, e.g. [1, 2, 3, 4\] would be 1 + 2x + 3x² + 4x³.
pub fn solve_cubic(coefficients: &[Complex64; 4]) -> Result<Vec<Complex64>, TensorErrors> {
    if approx_eq!(f64, coefficients[3].abs(), 0.0) {
        return Err(TensorErrors::PolynomialLeadingCoefficientZero);
    }

    let a = coefficients[3];
    let b = coefficients[2];
    let c = coefficients[1];
    let d = coefficients[0];

    let q = (3.0 * a * c - b.powi(2)) / (9.0 * a.powi(2));
    let r = (9.0 * a * b * c - 27.0 * a.powi(2) * d - 2.0 * b.powi(3)) / (54.0 * a.powi(3));

    let d = r.powi(2) + q.powi(3);

    let s_cubed = r + d.sqrt();

    let omega = Complex64 {
        re: -0.5,
        im: f64::sqrt(3.0) / 2.0,
    };

    let s = s_cubed.cbrt();
    let t = (-q) / s;

    let offset = b / (3.0 * a);

    Ok(vec![
        s + t - offset,
        s * omega + t * omega.powi(2) - offset,
        s * omega.powi(2) + t * omega - offset,
    ])
}

/// Solves a quartic. The coefficients are entered as a `&[Complex64; 5]` where the index of the
/// coefficient corresponds to the power of x, e.g. [1, 2, 3, 4, 5\] would be 1 + 2x + 3x² + 4x³ + 5x⁴.
pub fn solve_quartic(coefficients: &[Complex64; 5]) -> Result<Vec<Complex64>, TensorErrors> {
    if approx_eq!(f64, coefficients[4].abs(), 0.0) {
        return Err(TensorErrors::PolynomialLeadingCoefficientZero);
    }

    let a = coefficients[4];
    let b = coefficients[3];
    let c = coefficients[2];
    let d = coefficients[1];
    let e = coefficients[0];

    let yi_squared = solve_cubic(&[
        -(-b.powi(3) + 4.0 * a * b * c - 8.0 * a.powi(2) * d).powi(2),
        3.0 * b.powi(4) + 16.0 * a.powi(2) * c.powi(2) + 16.0 * a.powi(2) * b * d
            - 16.0 * a * b.powi(2) * c
            - 64.0 * a.powi(3) * e,
        -(3.0 * b.powi(2) - 8.0 * a * c),
        1.0.into(),
    ])?;

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

                if (y0 * y1 * y2 - (-b.powi(3) + 4.0 * a * b * c - 8.0 * a.powi(2) * d)).abs()
                    < 2e-10
                {
                    found = true;
                    break;
                }
            }
        }
    }

    assert!(found, "Roots could not be calculated");

    Ok(vec![
        (-b + y0 + y1 + y2) / (4.0 * a),
        (-b + y0 - y1 - y2) / (4.0 * a),
        (-b - y0 + y1 - y2) / (4.0 * a),
        (-b - y0 - y1 + y2) / (4.0 * a),
    ])
}


#[cfg(test)]
mod polynomials_tests {
    use float_cmp::assert_approx_eq;
    use num::complex::{Complex64, ComplexFloat};
    use std::ops::Add;
    use tensor_math::definitions::errors::TensorErrors;
    use tensor_math::math::polynomials::{solve_cubic, solve_quadratic, solve_quartic};

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
        assert!(matches!(err, TensorErrors::PolynomialLeadingCoefficientZero));

        let err = solve_cubic(&c2).unwrap_err();
        assert!(matches!(err, TensorErrors::PolynomialLeadingCoefficientZero));

        let err = solve_quartic(&c3).unwrap_err();
        assert!(matches!(err, TensorErrors::PolynomialLeadingCoefficientZero));
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
        assert_approx_eq!(
            f64,
            coefficients
                .iter()
                .enumerate()
                .map(|(i, c)| c * roots[0].powi(i as i32))
                .reduce(Complex64::add)
                .unwrap()
                .abs(),
            0.0,
            epsilon = 1e-15
        );
        assert_approx_eq!(
            f64,
            coefficients
                .iter()
                .enumerate()
                .map(|(i, c)| c * roots[1].powi(i as i32))
                .reduce(Complex64::add)
                .unwrap()
                .abs(),
            0.0,
            epsilon = 1e-15
        );
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
        assert_approx_eq!(
            f64,
            coefficients
                .iter()
                .enumerate()
                .map(|(i, c)| c * roots[0].powi(i as i32))
                .reduce(Complex64::add)
                .unwrap()
                .abs(),
            0.0,
            epsilon = 2e-15
        );
        assert_approx_eq!(
            f64,
            coefficients
                .iter()
                .enumerate()
                .map(|(i, c)| c * roots[1].powi(i as i32))
                .reduce(Complex64::add)
                .unwrap()
                .abs(),
            0.0,
            epsilon = 2e-15
        );
        assert_approx_eq!(
            f64,
            coefficients
                .iter()
                .enumerate()
                .map(|(i, c)| c * roots[2].powi(i as i32))
                .reduce(Complex64::add)
                .unwrap()
                .abs(),
            0.0,
            epsilon = 2e-15
        );

        let coefficients2 = [
            Complex64 { re: 45.0, im: -2.0 },
            Complex64 { re: -3.0, im: 4.0 },
            Complex64 { re: 2.0, im: 3.0 },
            Complex64 { re: 1.0, im: 10.0 },
        ];
        let roots2 = solve_cubic(&coefficients2).unwrap();

        assert_eq!(roots2.len(), 3);
        assert_approx_eq!(
            f64,
            coefficients2
                .iter()
                .enumerate()
                .map(|(i, c)| c * roots2[0].powi(i as i32))
                .reduce(Complex64::add)
                .unwrap()
                .abs(),
            0.0,
            epsilon = 2e-11
        );
        assert_approx_eq!(
            f64,
            coefficients2
                .iter()
                .enumerate()
                .map(|(i, c)| c * roots2[1].powi(i as i32))
                .reduce(Complex64::add)
                .unwrap()
                .abs(),
            0.0,
            epsilon = 2e-11
        );
        assert_approx_eq!(
            f64,
            coefficients2
                .iter()
                .enumerate()
                .map(|(i, c)| c * roots2[2].powi(i as i32))
                .reduce(Complex64::add)
                .unwrap()
                .abs(),
            0.0,
            epsilon = 2e-11
        );
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

        assert_approx_eq!(
            f64,
            coefficients
                .iter()
                .enumerate()
                .map(|(i, c)| c * roots[0].powi(i as i32))
                .reduce(Complex64::add)
                .unwrap()
                .abs(),
            0.0,
            epsilon = 2e-10
        );
        assert_approx_eq!(
            f64,
            coefficients
                .iter()
                .enumerate()
                .map(|(i, c)| c * roots[1].powi(i as i32))
                .reduce(Complex64::add)
                .unwrap()
                .abs(),
            0.0,
            epsilon = 2e-10
        );
        assert_approx_eq!(
            f64,
            coefficients
                .iter()
                .enumerate()
                .map(|(i, c)| c * roots[2].powi(i as i32))
                .reduce(Complex64::add)
                .unwrap()
                .abs(),
            0.0,
            epsilon = 2e-10
        );
        assert_approx_eq!(
            f64,
            coefficients
                .iter()
                .enumerate()
                .map(|(i, c)| c * roots[3].powi(i as i32))
                .reduce(Complex64::add)
                .unwrap()
                .abs(),
            0.0,
            epsilon = 2e-10
        );
    }
}
