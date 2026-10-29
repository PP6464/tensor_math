use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::utilities::matrix::identity;
use float_cmp::approx_eq;
use num::complex::Complex64;
use num::{One, Zero};
use std::ops::{Add, Div, Mul, Neg, Sub};

impl Matrix<f64> {
    /// Computes the determinant of the matrix.
    /// This fails if the matrix is not square.
    pub fn det(self) -> Result<f64, TensorErrors> {
        if !self.is_square() {
            return Err(TensorErrors::NonSquareMatrix);
        }

        let ord = self.rows;
        if ord == 0 {
            return Ok(1.0);
        }
        let (ref_form, det_scale, _) = self.tracked_row_echelon();
        let mut res = 1f64;

        for i in 0..ord {
            res *= ref_form[(i, i)];
        }

        Ok(res * det_scale as f64)
    }

    /// Computes the inverse of a matrix.
    /// This fails if the matrix is not square or if the determinant is 0.
    pub fn inv(self) -> Result<Matrix<f64>, TensorErrors> {
        if !self.is_square() {
            return Err(TensorErrors::NonSquareMatrix);
        }

        let ord = self.rows;
        if ord == 0 {
            return Ok(self.clone());
        }

        let a_i_rref =
            unsafe { self.concat_along_rows_unchecked(identity(ord)) }.reduced_row_echelon();
        let left = unsafe { a_i_rref.slice_unchecked(0..ord, 0..ord) };
        let right = unsafe { a_i_rref.slice_unchecked(0..ord, ord..2 * ord) };

        if !approx_eq!(Matrix<f64>, left.clone_into_matrix(), identity(ord)) {
            return Err(TensorErrors::DeterminantZero);
        }

        Ok(right.clone_into_matrix())
    }
}

impl Matrix<Complex64> {
    /// Computes the determinant of a matrix.
    /// This fails if the matrix is not square.
    pub fn det(self) -> Result<Complex64, TensorErrors> {
        if !self.is_square() {
            return Err(TensorErrors::NonSquareMatrix);
        }

        let ord = self.rows;
        if ord == 0 {
            return Ok(Complex64::ONE);
        }

        let (ref_form, det_scale, _) = self.tracked_row_echelon();
        let mut res = Complex64::ONE;

        for i in 0..ord {
            res *= ref_form[(i, i)];
        }

        Ok(res * Complex64::new(det_scale as f64, 0.0))
    }

    /// Computes the inverse of a matrix.
    /// This fails if the matrix is not square or if the determinant is 0.
    pub fn inv(self) -> Result<Matrix<Complex64>, TensorErrors> {
        if !self.is_square() {
            return Err(TensorErrors::NonSquareMatrix);
        }

        let ord = self.rows;
        if ord == 0 {
            return Ok(self.clone());
        }

        let a_i_rref =
            unsafe { self.concat_along_rows_unchecked(identity(ord)) }.reduced_row_echelon();
        let left = a_i_rref.slice(0..ord, 0..ord)?;
        let right = a_i_rref.slice(0..ord, ord..2 * ord)?;

        if !approx_eq!(Matrix<Complex64>, left.clone_into_matrix(), identity(ord)) {
            return Err(TensorErrors::DeterminantZero);
        }

        Ok(right.clone_into_matrix())
    }
}

/// Calculates the determinant for a matrix of values of type `T`.
/// This uses a slower method which is O(n!) for an n x n matrix but may be
/// useful for matrices of types that aren't f64 or Complex64.
/// This fails if the matrix is not square.
pub fn det_slow<T: Add<Output = T> + Mul<Output = T> + Sub<Output = T> + Clone + Zero + One>(
    m: &Matrix<T>,
) -> Result<T, TensorErrors> {
    if !m.is_square() {
        return Err(TensorErrors::NonSquareMatrix);
    }

    let ord = m.rows;

    if ord == 0 {
        return Ok(T::one());
    }

    if ord == 2 {
        return Ok(
            m[&[0, 0]].clone() * m[&[1, 1]].clone() - m[&[0, 1]].clone() * m[&[1, 0]].clone()
        );
    }

    if ord == 1 {
        return Ok(m[&[0, 0]].clone());
    }

    let mut determinant = T::zero();

    for i in 0..ord {
        let is_minus = i % 2 != 0;

        unsafe {
            if i == 0 {
                let slice = m.slice_unchecked(1..ord, 1..ord).clone_into_matrix();
                determinant = determinant + m[&[0, i]].clone() * det_slow(&slice)?;

                continue;
            }
        }

        unsafe {
            if i == ord - 1 {
                let slice = m.slice_unchecked(1..ord, 0..(ord - 1)).clone_into_matrix();

                if is_minus {
                    determinant = determinant - m[&[0, i]].clone() * det_slow(&slice)?;
                } else {
                    determinant = determinant + m[&[0, i]].clone() * det_slow(&slice)?;
                }

                continue;
            }
        }

        unsafe {
            let slice = m
                .slice_unchecked(1..ord, 0..i)
                .clone_into_matrix()
                .concat_along_rows_unchecked(
                    m.slice_unchecked(1..ord, i + 1..ord).clone_into_matrix(),
                );

            if is_minus {
                determinant = determinant - m[&[0, i]].clone() * det_slow(&slice)?
            } else {
                determinant = determinant + m[&[0, i]].clone() * det_slow(&slice)?
            }
        }
    }

    Ok(determinant)
}

/// Calculates the inverse for a matrix of values of type `T`.
/// This uses a slower implementation for det and is slower itself than using
/// REF/RREF, but note that this can be used on matrices that don't have REF/RREF
/// implemented for them.
/// This fails if the matrix is not square or has determinant 0.
pub fn inv_slow<T>(m: &Matrix<T>) -> Result<Matrix<T>, TensorErrors>
where
    T: Add<Output = T>
        + Mul<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Neg<Output = T>
        + Clone
        + Zero
        + One
        + PartialEq,
{
    if !m.is_square() {
        return Err(TensorErrors::NonSquareMatrix);
    }

    let ord = m.rows;

    if ord == 0 {
        return Ok(m.clone());
    }

    let mut res = Matrix::<T>::zeros(m.rows, m.cols);
    let d = det_slow(m)?;

    if d == T::zero() {
        return Err(TensorErrors::DeterminantZero);
    }

    // Construct adjoint matrix

    // i is for which row we are on
    for i in 0..ord {
        // j is for which column we are on
        for j in 0..ord {
            let is_minus = (i + j) % 2 != 0;

            unsafe {
                let slice = match (i, j) {
                    (0, 0) => m.slice_unchecked(1..ord, 1..ord).clone_into_matrix(),
                    _ if (i, j) == (ord - 1, ord - 1) => {
                        m.slice_unchecked(0..i, 0..j).clone_into_matrix()
                    }
                    _ if (i, j) == (0, ord - 1) => {
                        m.slice_unchecked(1..ord, 0..j).clone_into_matrix()
                    }
                    _ if (i, j) == (ord - 1, 0) => {
                        m.slice_unchecked(0..i, 1..ord).clone_into_matrix()
                    }
                    _ if i == 0 => m
                        .slice_unchecked(1..ord, 0..j)
                        .clone_into_matrix()
                        .concat_along_rows_unchecked(
                            m.slice_unchecked(1..ord, j + 1..ord).clone_into_matrix(),
                        ),
                    _ if i == ord - 1 => m
                        .slice_unchecked(0..i, 0..j)
                        .clone_into_matrix()
                        .concat_along_rows_unchecked(
                            m.slice_unchecked(0..i, j + 1..ord).clone_into_matrix(),
                        ),
                    _ if j == 0 => m
                        .slice_unchecked(0..i, 1..ord)
                        .clone_into_matrix()
                        .concat_along_cols_unchecked(
                            m.slice_unchecked((i + 1)..ord, 1..ord).clone_into_matrix(),
                        ),
                    _ if j == ord - 1 => m
                        .slice_unchecked(0..i, 0..j)
                        .clone_into_matrix()
                        .concat_along_cols_unchecked(
                            m.slice_unchecked((i + 1)..ord, 0..j).clone_into_matrix(),
                        ),
                    _ => {
                        let slice_top = m
                            .slice_unchecked(0..i, 0..j)
                            .clone_into_matrix()
                            .concat_along_rows_unchecked(
                                m.slice_unchecked(0..i, (j + 1)..ord).clone_into_matrix(),
                            );
                        let slice_bottom = m
                            .slice_unchecked((i + 1)..ord, 0..j)
                            .clone_into_matrix()
                            .concat_along_rows_unchecked(
                                m.slice_unchecked((i + 1)..ord, (j + 1)..ord)
                                    .clone_into_matrix(),
                            );

                        slice_top.concat_along_cols_unchecked(slice_bottom)
                    }
                };

                res[&[j, i]] = if is_minus {
                    -det_slow(&slice)?
                } else {
                    det_slow(&slice)?
                };
            }
        }
    }

    Ok(res / d)
}

#[cfg(test)]
mod det_inv_tests {
    use float_cmp::approx_eq;
    use num::complex::Complex64;
    use num::FromPrimitive;
    use tensor_math::definitions::errors::TensorErrors;
    use tensor_math::definitions::matrix::Matrix;
    use tensor_math::definitions::traits::IntoMatrix;
    use tensor_math::math::det_inv::{det_slow, inv_slow};
    use tensor_math::utilities::matrix::eye;

    #[test]
    fn invalid_inversion_square_matrix_only() {
        let m1 = Matrix::<f64>::new(3, 2, (0..6).map(f64::from).collect()).unwrap();

        let err = inv_slow(&m1).unwrap_err();
        assert!(matches!(err, TensorErrors::NonSquareMatrix));

        let err = m1.clone().inv().unwrap_err();
        assert!(matches!(err, TensorErrors::NonSquareMatrix));

        let m2 = Matrix::new(
            3,
            2,
            (0..6)
                .map(Complex64::from_i32)
                .map(|x| x.unwrap())
                .collect(),
        )
        .unwrap();
        let err = m2.clone().inv().unwrap_err();

        assert!(matches!(err, TensorErrors::NonSquareMatrix));
    }

    #[test]
    fn determinant() {
        let m1 = Matrix::<i32>::new(3, 3, vec![5, -2, 1, 8, 9, -5, 1, 0, 2]).unwrap();
        let m2 = vec![1].into_matrix();
        let m3 = Matrix::<f64>::zeros(10, 10);

        assert_eq!(det_slow(&m2).unwrap(), 1);
        assert_eq!(det_slow(&m1).unwrap(), 123);
        assert_eq!(det_slow(&m3).unwrap(), 0.0);
        assert!(approx_eq!(
            f64,
            m1.clone().map(|x| x as f64).det().unwrap(),
            123.0
        ));
        assert_eq!(m3.clone().det().unwrap(), 0.0);
        assert_eq!(m3.into_complex().det().unwrap(), 0.0.into());
    }

    #[test]
    fn invalid_det_square_matrix_only() {
        let m1 = Matrix::<f64>::new(3, 2, (0..6).map(f64::from).collect()).unwrap();

        let err = det_slow(&m1).unwrap_err();
        assert!(matches!(err, TensorErrors::NonSquareMatrix));


        let err = m1.clone().det().unwrap_err();
        assert!(matches!(err, TensorErrors::NonSquareMatrix));


        let m2 = Matrix::new(
            3,
            2,
            (0..6)
                .map(Complex64::from_i32)
                .map(|x| x.unwrap())
                .collect(),
        )
        .unwrap();

        let err = m2.clone().det().unwrap_err();
        assert!(matches!(err, TensorErrors::NonSquareMatrix));
    }

    #[test]
    fn inverse() {
        let m1 = Matrix::<f64>::new(3, 3, vec![3.0, 4.0, 5.0, 2.0, -1.0, 4.0, 3.0, -5.0, -10.0])
            .unwrap();
        let inverse = inv_slow(&m1).unwrap();
        let fast_inverse = m1.clone().inv().unwrap();
        let ans = Matrix::<f64>::new(
            3,
            3,
            vec![
                10.0 / 61.0,
                5.0 / 61.0,
                7.0 / 61.0,
                32.0 / 183.0,
                -15.0 / 61.0,
                -2.0 / 183.0,
                -7.0 / 183.0,
                9.0 / 61.0,
                -11.0 / 183.0,
            ],
        )
        .unwrap();

        assert!(approx_eq!(
            Matrix<f64>,
            inverse.clone(),
            ans.clone(),
            epsilon = 1e-15
        ));
        assert!(approx_eq!(Matrix<f64>, fast_inverse, ans, epsilon = 1e-15));
        assert!(m1
            .mat_mul(inverse.clone())
            .unwrap()
            .enumerated_iter()
            .all(|(i, x)| { approx_eq!(f64, *x, eye::<f64>(3)[i], epsilon = 1e-15) }));

        let m2 = Matrix::new(
            2,
            2,
            vec![
                Complex64 { re: 1.0, im: 1.0 },
                Complex64 { re: 1.0, im: 0.0 },
                Complex64 { re: -1.0, im: 1.0 },
                Complex64 { re: 0.0, im: -1.0 },
            ],
        )
        .unwrap();

        let slow_res = inv_slow(&m2).unwrap();
        let fast_res = m2.clone().inv().unwrap();
        let ans = Matrix::new(
            2,
            2,
            vec![
                Complex64 {
                    re: 0.25,
                    im: -0.25,
                },
                Complex64 {
                    re: -0.25,
                    im: -0.25,
                },
                Complex64 { re: 0.5, im: 0.0 },
                Complex64 { re: 0.0, im: 0.5 },
            ],
        )
        .unwrap();

        assert!(approx_eq!(
            Matrix<Complex64>,
            slow_res.clone(),
            ans.clone(),
            epsilon = 1e-15
        ));
        assert!(approx_eq!(
            Matrix<Complex64>,
            fast_res,
            ans,
            epsilon = 1e-15
        ));
        assert!(m2
            .mat_mul(slow_res.clone())
            .unwrap()
            .enumerated_iter()
            .all(|(i, x)| {
                approx_eq!(f64, x.re, eye(3)[i], epsilon = 1e-15);
                approx_eq!(f64, x.im, 0.0, epsilon = 1e-15)
            }));
    }

    #[test]
    fn inverse_det_0() {
        let m1 = Matrix::<f64>::new(3, 3, vec![0.0; 9]).unwrap();
        let m2 = Matrix::new(
            2,
            2,
            vec![
                Complex64 { re: 1.0, im: 1.0 },
                Complex64 { re: 1.0, im: 0.0 },
                Complex64 { re: 1.0, im: 1.0 },
                Complex64 { re: 1.0, im: 0.0 },
            ],
        )
        .unwrap();

        let err = inv_slow(&m1).unwrap_err();
        assert!(matches!(err, TensorErrors::DeterminantZero));

        let err = m1.clone().inv().unwrap_err();
        assert!(matches!(err, TensorErrors::DeterminantZero));

        let err = m2.clone().inv().unwrap_err();
        assert!(matches!(err, TensorErrors::DeterminantZero));
    }

    #[test]
    fn empty_matrix_det_inv() {
        let m_f64 = Matrix::<f64>::new(0, 0, vec![]).unwrap();
        let m_complex = Matrix::<Complex64>::new(0, 0, vec![]).unwrap();

        // f64 tests
        assert_eq!(m_f64.clone().det().unwrap(), 1.0);
        assert_eq!(det_slow(&m_f64).unwrap(), 1.0);
        assert_eq!(m_f64.clone().inv().unwrap().rows(), 0);
        assert_eq!(inv_slow(&m_f64).unwrap().rows(), 0);

        // Complex64 tests
        assert_eq!(m_complex.clone().det().unwrap(), Complex64::new(1.0, 0.0));
        assert_eq!(m_complex.clone().inv().unwrap().rows(), 0);
        assert_eq!(inv_slow(&m_complex).unwrap().rows(), 0);
    }
}
