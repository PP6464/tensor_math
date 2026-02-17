use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::utilities::matrix::identity;
use float_cmp::approx_eq;
use num::complex::Complex64;
use num::Zero;
use std::ops::{Add, Div, Mul, Neg, Sub};

impl Matrix<f64> {
    /// Computes the determinant of the matrix
    pub fn det(&self) -> Result<f64, TensorErrors> {
        if !self.is_square() {
            return Err(TensorErrors::NonSquareMatrix);
        }

        let ord = self.rows;
        let (ref_form, det_scale) = self.tracked_row_echelon();
        let mut res = 1f64;

        for i in 0..ord {
            res *= ref_form[(i, i)];
        }

        Ok(res * det_scale as f64)
    }

    /// Computes the inverse of a matrix.
    /// This returns a result in case the determinant is zero.
    pub fn inv(&self) -> Result<Matrix<f64>, TensorErrors> {
        if !self.is_square() {
            return Err(TensorErrors::NonSquareMatrix);
        }

        let ord = self.rows;

        let a_i_rref = self.concat_mt(&identity(ord)?, 1)?.reduced_row_echelon();
        let left = a_i_rref.slice(0..ord, 0..ord)?;
        let right = a_i_rref.slice(0..ord, ord..2 * ord)?;

        if !approx_eq!(Matrix<f64>, left, identity(ord)?) {
            return Err(TensorErrors::DeterminantZero);
        }

        Ok(right)
    }
}

impl Matrix<Complex64> {
    /// Computes the determinant of a matrix.
    pub fn det(&self) -> Result<Complex64, TensorErrors> {
        if !self.is_square() {
            return Err(TensorErrors::NonSquareMatrix);
        }

        let (ref_form, det_scale) = self.tracked_row_echelon();
        let ord = self.rows;
        let mut res = Complex64::ONE;

        for i in 0..ord {
            res *= ref_form[(i, i)];
        }

        Ok(res * Complex64::new(det_scale as f64, 0.0))
    }

    /// Computes the inverse of a matrix.
    /// This returns a result for in case the determinant was zero.
    pub fn inv(&self) -> Result<Matrix<Complex64>, TensorErrors> {
        if !self.is_square() {
            return Err(TensorErrors::NonSquareMatrix);
        }

        let ord = self.rows;

        let a_i_rref = self.concat_mt(&identity(ord)?, 1)?.reduced_row_echelon();
        let left = a_i_rref.slice(0..ord, 0..ord)?;
        let right = a_i_rref.slice(0..ord, ord..2 * ord)?;

        if !approx_eq!(Matrix<Complex64>, left, identity(ord)?) {
            return Err(TensorErrors::DeterminantZero);
        }

        Ok(right)
    }
}

/// Calculates the determinant for a matrix of values of type `T`.
/// This uses a slower method which is O(n!) for an n x n matrix but may be
/// useful for matrices of types that aren't f64 or Complex64.
pub fn det_slow<T: Add<Output = T> + Mul<Output = T> + Sub<Output = T> + Clone + Zero>(
    m: &Matrix<T>,
) -> Result<T, TensorErrors> {
    if !m.is_square() {
        return Err(TensorErrors::NonSquareMatrix);
    }

    let ord = m.shape[0];

    if ord == 2 {
        return Ok(m[&[0, 0]].clone() * m[&[1, 1]].clone() - m[&[0, 1]].clone() * m[&[1, 0]].clone());
    }

    if ord == 1 {
        return Ok(m[&[0, 0]].clone());
    }

    let mut determinant = T::zero();

    for i in 0..ord {
        let is_minus = i % 2 != 0;

        if i == 0 {
            let slice = m.slice(1..ord, 1..ord)?;
            determinant = determinant + m[&[0, i]].clone() * det_slow(&slice)?;

            continue;
        }

        if i == ord - 1 {
            let slice = m.slice(1..ord, 0..(ord - 1))?;

            if is_minus {
                determinant = determinant - m[&[0, i]].clone() * det_slow(&slice)?;
            } else {
                determinant = determinant + m[&[0, i]].clone() * det_slow(&slice)?;
            }

            continue;
        }

        let slice = m
            .slice(1..ord, 0..i)?
            .concat(&m.slice(1..ord, i + 1..ord).unwrap(), 1)?;

        if is_minus {
            determinant = determinant - m[&[0, i]].clone() * det_slow(&slice)?
        } else {
            determinant = determinant + m[&[0, i]].clone() * det_slow(&slice)?
        }
    }

    Ok(determinant)
}

/// Calculates the inverse for a matrix of values of type `T`.
/// If the determinant is 0 you will receive `TensorErrors::DeterminantZero`.
/// This uses a slower implementation for det and is slower itself than using
/// REF/RREF, but note that this can be used on matrices that don't have REF/RREF
/// implemented for them.
pub fn inv_slow<T>(m: &Matrix<T>) -> Result<Matrix<T>, TensorErrors>
where
    T: Add<Output = T>
        + Mul<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Neg<Output = T>
        + Clone
        + Zero
        + PartialEq,
{
    if !m.is_square() {
        return Err(TensorErrors::NonSquareMatrix);
    }

    let ord = m.shape[0];
    let mut res = Matrix::<T>::from_value(m.rows, m.cols, T::zero())?;
    let d = det_slow(&m)?;

    if d == T::zero() {
        return Err(TensorErrors::DeterminantZero);
    }

    // Construct adjoint matrix

    // i is for which row we are on
    for i in 0..ord {
        // j is for which column we are on
        for j in 0..ord {
            let is_minus = (i + j) % 2 != 0;

            let slice = match (i, j) {
                (0, 0) => m.slice(1..ord, 1..ord),
                _ if (i, j) == (ord - 1, ord - 1) => m.slice(0..i, 0..j),
                _ if (i, j) == (0, ord - 1) => m.slice(1..ord, 0..j),
                _ if (i, j) == (ord - 1, 0) => m.slice(0..i, 1..ord),
                _ if i == 0 => Ok(m
                    .slice(1..ord, 0..j)?
                    .concat(&m.slice(1..ord, j + 1..ord)?, 1)?),
                _ if i == ord - 1 => m.slice(0..i, 0..j)?.concat(&m.slice(0..i, j + 1..ord)?, 1),
                _ if j == 0 => Ok(m
                    .slice(0..i, 1..ord)
                    .unwrap()
                    .concat(&m.slice((i + 1)..ord, 1..ord)?, 0)?),
                _ if j == ord - 1 => Ok(m
                    .slice(0..i, 0..j)
                    .unwrap()
                    .concat(&m.slice((i + 1)..ord, 0..j)?, 0)?),
                _ => Ok({
                    let slice_top = m
                        .slice(0..i, 0..j)?
                        .concat(&m.slice(0..i, (j + 1)..ord)?, 1)?;
                    let slice_bottom = m
                        .slice((i + 1)..ord, 0..j)?
                        .concat(&m.slice((i + 1)..ord, (j + 1)..ord)?, 1)?;

                    slice_top.concat(&slice_bottom, 0)?
                })
            }?;

            res[&[j, i]] = if is_minus {
                -det_slow(&slice)?
            } else {
                det_slow(&slice)?
            };
        }
    }

    Ok(res / d)
}
