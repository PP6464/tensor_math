use crate::definitions::matrix::Matrix;
use crate::definitions::matrix_slice_mut::MatrixSliceMut;
use crate::utilities::matrix::identity;
use float_cmp::approx_eq;
use num::complex::Complex64;
use std::cmp::min;

/*
--------------------------------------------
* Helpers
--------------------------------------------
*/

/// B ← (I − 2uuᵀ) B, in place. `w` is scratch with len >= B.cols().
unsafe fn reflect_left_f64(b: &mut MatrixSliceMut<'_, f64>, u: &[f64], w: &mut [f64]) {
    let w = w.get_unchecked_mut(..b.cols());
    b.as_immut().vec_mat_mul_unchecked_mt_into_slice(u, w); // w = uᵀB
    w.iter_mut().for_each(|x| *x *= 2.0);
    b.rank1_update_sub_unchecked_mt(u, w); // B −= u·w
}

/// B ← (I − 2uu*) B, in place. `uc` (with len >= B.rows()) and `w` (with len >= B.cols()) are reusable scratch.
unsafe fn reflect_left_c64(
    b: &mut MatrixSliceMut<'_, Complex64>,
    u: &[Complex64],
    uc: &mut Vec<Complex64>,
    w: &mut [Complex64],
) {
    uc.clear();
    uc.extend(u.iter().map(|x| x.conj()));
    let w = w.get_unchecked_mut(..b.cols());
    b.as_immut().vec_mat_mul_unchecked_mt_into_slice(uc, w); // w = u*B
    w.iter_mut().for_each(|x| *x *= 2.0);
    b.rank1_update_sub_unchecked_mt(u, w); // B −= u·w  (no conj)
}

/// B ← B (I − 2uu*), in place. `u` has len B.cols(); `y` is scratch with len >= B.rows(); `uc` is reusable scratch for conj(u).
unsafe fn reflect_right_c64(
    b: &mut MatrixSliceMut<'_, Complex64>,
    u: &[Complex64],
    uc: &mut Vec<Complex64>,
    y: &mut [Complex64],
) {
    let y = &mut y[..b.rows()];
    b.as_immut().mat_vec_mul_unchecked_mt_into_slice(u, y); // y = Bu
    y.iter_mut().for_each(|x| *x *= 2.0);
    uc.clear();
    uc.extend(u.iter().map(|x| x.conj()));
    b.rank1_update_sub_unchecked_mt(y, uc);
}

/*
--------------------------------------------
* Householder reflectors
--------------------------------------------
*/

pub struct HouseholderReflectors<T> {
    /// One reflector vector per step, stored at the row offset it applies from.
    /// vectors\[k\] has length (rows - k).
    vectors: Vec<Matrix<T>>,
    rows: usize,
}

impl HouseholderReflectors<f64> {
    /// Accumulate the reflectors into a matrix Q
    pub fn accumulate_q(&self) -> Matrix<f64> {
        let n = self.rows;
        let mut w = vec![0.0; n];
        let mut q = identity::<f64>(self.rows);
        unsafe {
            for (k, u) in self.vectors.iter().enumerate().rev() {
                if u.rows == 0 {
                    continue;
                } // skip no-op steps
                let mut blk = q.slice_unchecked_mut(k..n, k..n);
                reflect_left_f64(&mut blk, &u.elements, &mut w);
            }
        }
        q
    }
}

impl HouseholderReflectors<Complex64> {
    /// Accumulate the reflectors into a matrix Q
    pub fn accumulate_q(&self) -> Matrix<Complex64> {
        let n = self.rows;
        let mut q = identity::<Complex64>(n);
        let mut w = vec![Complex64::ZERO; n];
        let mut uc = Vec::with_capacity(n);
        unsafe {
            for (k, u) in self.vectors.iter().enumerate().rev() {
                if u.rows == 0 {
                    continue;
                } // skip no-op steps
                let mut blk = q.slice_unchecked_mut(k..n, k..n);
                reflect_left_c64(&mut blk, &u.elements, &mut uc, &mut w);
            }
        }
        q
    }

    /// Applies the reflectors to the right of a matrix B, in place.
    pub(crate) unsafe fn apply_right_unchecked(&self, b: &mut Matrix<Complex64>) {
        let mut y = vec![Complex64::ZERO; b.rows];
        let mut uc = Vec::with_capacity(self.rows);
        for (k, u) in self.vectors.iter().enumerate() {
            if u.rows == 0 {
                continue;
            }
            let mut block = b.slice_unchecked_mut(0..b.rows, k..self.rows);
            reflect_right_c64(&mut block, &u.elements, &mut uc, &mut y);
        }
    }
}

/*
--------------------------------------------
* Householder methods
--------------------------------------------
*/

impl Matrix<f64> {
    /// Computes the Householder transformation for the given matrix `t` (of shape (rows, cols)).
    /// Returns (Q, R), where Q is a unitary matrix of shape (rows, rows)
    /// and R is an upper triangle matrix of shape (rows, cols) such that `Q.mat_mul(R) == t`.
    pub fn householder(self) -> (Matrix<f64>, Matrix<f64>) {
        let (r, reflectors) = self.householder_r();
        (reflectors.accumulate_q(), r)
    }

    /// Computes the Householder transformation for the given matrix `t` (of shape (rows, cols)).
    /// Returns (R, reflectors) where R is an upper triangle matrix of shape (rows, cols) and
    /// reflectors is something that can be accumulated to construct Q (with the provided method),
    /// where Q is a unitary matrix of shape (rows, rows) such that `Q.mat_mul(R) == t`.
    pub fn householder_r(mut self) -> (Matrix<f64>, HouseholderReflectors<f64>) {
        let (rows, cols) = (self.rows, self.cols);
        let mut vectors = Vec::with_capacity(min(rows, cols));
        let mut w = vec![0.0; cols];

        unsafe {
            for k in 0..min(cols, rows) {
                let mut u = (k..rows)
                    .map(|x| self.get_unchecked((x, k)))
                    .copied()
                    .collect::<Vec<_>>();
                let norm = u.iter().map(|&x| x * x).sum::<f64>().sqrt();
                let alpha = -(if u[0] == 0.0 { 1.0 } else { u[0].signum() }) * norm;
                u[0] -= alpha;
                let vnorm = u.iter().map(|&x| x * x).sum::<f64>().sqrt();
                if approx_eq!(f64, vnorm, 0.0) {
                    vectors.push(Matrix::from_shape(0, 1));
                    continue;
                }
                u.iter_mut().for_each(|x| *x /= vnorm);

                if k + 1 < cols {
                    let mut trail = self.slice_unchecked_mut(k..rows, k + 1..cols);
                    reflect_left_f64(&mut trail, &u, &mut w);
                }

                // Directly set column k to the Householder vector, with alpha at the top and zeros below
                *self.get_unchecked_mut((k, k)) = alpha;
                for i in k + 1..rows {
                    *self.get_unchecked_mut((i, k)) = 0.0;
                }

                vectors.push(Matrix {
                    rows: rows - k,
                    cols: 1,
                    elements: u,
                });
            }
        }

        (self, HouseholderReflectors { vectors, rows })
    }
}

impl Matrix<Complex64> {
    /// Computes the Householder transformation for the given matrix `t` (of shape (rows, cols)).
    /// Returns (Q, R), where Q is a unitary matrix of shape (rows, rows)
    /// and R is an upper triangle matrix of shape (rows, cols) such that `Q.mat_mul(R) == t`.
    pub fn householder(self) -> (Matrix<Complex64>, Matrix<Complex64>) {
        let (r, reflectors) = self.householder_r();
        (reflectors.accumulate_q(), r)
    }

    /// Computes the Householder transformation for the given matrix `t` (of shape (rows, cols)).
    /// Returns (R, reflectors) where R is an upper triangle matrix of shape (rows, cols) and
    /// reflectors is something that can be accumulated to construct Q (with the provided method),
    /// where Q is a unitary matrix of shape (rows, rows) such that `Q.mat_mul(R) == t`.
    pub fn householder_r(mut self) -> (Matrix<Complex64>, HouseholderReflectors<Complex64>) {
        let (rows, cols) = (self.rows, self.cols);
        let mut vectors = Vec::with_capacity(min(rows, cols));
        let mut w = vec![Complex64::ZERO; cols];
        let mut uc = Vec::with_capacity(rows);

        unsafe {
            for k in 0..min(rows, cols) {
                let mut u: Vec<Complex64> = (k..rows)
                    .map(|i| self.get_unchecked((i, k)))
                    .copied()
                    .collect();
                let norm = u.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
                let x0 = u[0];
                let phase = if x0 == Complex64::ZERO {
                    Complex64::ONE
                } else {
                    x0 / x0.norm()
                };
                let alpha = -phase * norm;
                u[0] -= alpha;
                let vnorm = u.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
                if approx_eq!(f64, vnorm, 0.0) {
                    vectors.push(Matrix::from_shape(0, 1));
                    continue;
                }
                u.iter_mut().for_each(|x| *x /= vnorm);

                if k + 1 < cols {
                    let mut trail = self.slice_mut(k..rows, (k + 1)..cols).unwrap();
                    reflect_left_c64(&mut trail, &u, &mut uc, &mut w);
                }
                self[(k, k)] = alpha;
                for i in k + 1..rows {
                    self[(i, k)] = Complex64::ZERO;
                }

                vectors.push(Matrix {
                    rows: rows - k,
                    cols: 1,
                    elements: u,
                });
            }
        }
        (self, HouseholderReflectors { vectors, rows })
    }
}
