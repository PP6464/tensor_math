use crate::definitions::matrix::Matrix;
use crate::utilities::matrix::identity;
use float_cmp::approx_eq;
use num::complex::{Complex64, ComplexFloat};
use std::cmp::min;

pub struct HouseholderReflectors<T> {
    /// One reflector vector per step, stored at the row offset it applies from.
    /// vectors\[k\] has length (rows - k).
    vectors: Vec<Matrix<T>>,
    rows: usize,
}

impl HouseholderReflectors<f64> {
    /// Accumulate the reflectors into a matrix Q
    pub fn accumulate_q(&self) -> Matrix<f64> {
        let mut q = identity::<f64>(self.rows);
        for (k, u) in self.vectors.iter().enumerate() {
            if u.rows == 0 { continue; } // skip no-op steps
            let u_t = u.transpose_mt();
            let q_slice_copy = q.slice(0..self.rows, k..self.rows).unwrap();
            let mut q_slice_mut = q.slice_mut(0..self.rows, k..self.rows).unwrap();
            let q_u = q_slice_copy.mat_mul_mt(u).unwrap();
            let q_slice_res = q_slice_copy - q_u.mat_mul(&u_t).unwrap() * 2.0;
            q_slice_mut.set_all(&q_slice_res).unwrap();
        }
        q
    }
}

impl HouseholderReflectors<Complex64> {
    /// Accumulate the reflectors into a matrix Q
    pub fn accumulate_q(&self) -> Matrix<Complex64> {
        let mut q = identity::<Complex64>(self.rows);
        for (k, u) in self.vectors.iter().enumerate() {
            if u.rows == 0 { continue; }
            let u_star = u.conj_transpose_mt();
            let q_slice_copy = q.slice(0..self.rows, k..self.rows).unwrap();
            let mut q_slice_mut = q.slice_mut(0..self.rows, k..self.rows).unwrap();
            let q_u = q_slice_copy.mat_mul_mt(u).unwrap();
            let q_slice_res = q_slice_copy - q_u.mat_mul_mt(&u_star).unwrap() * Complex64 { re: 2.0, im: 0.0 };
            q_slice_mut.set_all(&q_slice_res).unwrap();
        }
        q
    }
}

impl Matrix<f64> {
    /// Computes the Householder transformation for the given matrix `t` (of shape (rows, cols)).
    /// Returns (Q, R), where Q is a unitary matrix of shape (rows, rows)
    /// and R is an upper triangle matrix of shape (rows, cols) such that `Q.mat_mul(R) == t`.
    pub fn householder(&self) -> (Matrix<f64>, Matrix<f64>) {
        let (r, reflectors) = self.householder_r();
        (reflectors.accumulate_q(), r)
    }

    /// Computes the Householder transformation for the given matrix `t` (of shape (rows, cols)).
    /// Returns (R, reflectors) where R is an upper triangle matrix of shape (rows, cols) and
    /// reflectors is something that can be accumulated to construct Q (with the provided method),
    /// where Q is a unitary matrix of shape (rows, rows) such that `Q.mat_mul(R) == t`.
    pub fn householder_r(&self) -> (Matrix<f64>, HouseholderReflectors<f64>) {
        let (rows, cols) = (self.shape()[0], self.shape()[1]);

        let mut r = self.clone();
        let mut vectors = Vec::with_capacity(min(rows, cols));

        for k in 0..min(cols, rows) {
            let vec_bottom = r.slice(k..rows, k..k + 1).unwrap();
            let alpha =
                -1.0 * match vec_bottom[&[0, 0]] {
                    0.0 => 1.0,
                    x => x.signum(),
                } * vec_bottom.clone().mag();
            let mut e1 = Matrix::<f64>::from_shape(vec_bottom.rows, vec_bottom.cols);
            e1[&[0, 0]] = 1.0;
            let v = vec_bottom - e1 * alpha;

            if v.iter().all(|x| approx_eq!(f64, x.abs(), 0.0)) {
                vectors.push(Matrix::from_shape(0, 1));
                continue;
            }

            let u = v.norm_l2();
            let u_t = u.transpose_mt();

            // Update R
            // We can set the kth column directly
            r[(k, k)] = alpha;
            for i in (k+1)..rows {
                r[(i, k)] = 0.0;
            }

            if k + 1 < cols {
                let r_rest_copy = r.slice(k..rows, (k+1)..cols).unwrap();
                let mut r_rest_mut = r.slice_mut(k..rows, (k+1)..cols).unwrap();
                let u_t_r = u_t.mat_mul_mt(&r_rest_copy).unwrap();
                let r_rest_res = r_rest_copy - u.mat_mul_mt(&u_t_r).unwrap() * 2.0;
                r_rest_mut.set_all(&r_rest_res).expect("failed to set all");
            }

            // Add the new vector
            vectors.push(u);
        }

        (r, HouseholderReflectors { vectors, rows })
    }
}

impl Matrix<Complex64> {
    /// Computes the Householder transformation for the given matrix `t` (of shape (rows, cols)).
    /// Returns (Q, R), where Q is a unitary matrix of shape (rows, rows)
    /// and R is an upper triangle matrix of shape (rows, cols) such that `Q.mat_mul(R) == t`.
    pub fn householder(&self) -> (Matrix<Complex64>, Matrix<Complex64>) {
        let (r, reflectors) = self.householder_r();
        (reflectors.accumulate_q(), r)
    }

    /// Computes the Householder transformation for the given matrix `t` (of shape (rows, cols)).
    /// Returns (R, reflectors) where R is an upper triangle matrix of shape (rows, cols) and
    /// reflectors is something that can be accumulated to construct Q (with the provided method),
    /// where Q is a unitary matrix of shape (rows, rows) such that `Q.mat_mul(R) == t`.
    pub fn householder_r(&self) -> (Matrix<Complex64>, HouseholderReflectors<Complex64>) {
        let (rows, cols) = (self.shape()[0], self.shape()[1]);

        let mut r = self.clone();
        let mut vectors = Vec::with_capacity(min(rows, cols));

        for k in 0..min(cols, rows) {
            let vec_bottom = r.slice(k..rows, k..k + 1).unwrap();
            let alpha =
                -1.0 * match vec_bottom[&[0, 0]] {
                    Complex64::ZERO => Complex64::ONE,
                    x => x / Complex64::abs(x),
                } * vec_bottom.clone().mag();
            let mut e1 = Matrix::<Complex64>::from_shape(vec_bottom.rows, vec_bottom.cols);
            e1[&[0, 0]] = Complex64::ONE;
            let v = vec_bottom - e1 * alpha;

            if v.iter().all(|x| approx_eq!(f64, x.abs(), 0.0)) {
                vectors.push(Matrix::from_shape(0, 1));
                continue;
            }

            let u = v.norm_l2();
            let u_star = u.conj_transpose_mt();

            // Update R
            // We can set the kth column because we know what it is
            r[(k, k)] = alpha;
            for i in (k+1)..rows {
                r[(i, k)] = Complex64::ZERO;
            }

            if k + 1 < cols {
                let r_rest_copy = r.slice(k..rows, (k+1)..cols).unwrap();
                let mut r_rest_mut = r.slice_mut(k..rows, (k+1)..cols).unwrap();
                let u_star_r = u_star.mat_mul_mt(&r_rest_copy).unwrap();
                let r_rest_res = r_rest_copy - u.mat_mul_mt(&u_star_r).unwrap() * Complex64 { re: 2.0, im: 0.0 };
                r_rest_mut.set_all(&r_rest_res).expect("failed to set all");
            }

            // Update Q
            vectors.push(u);
        }

        (r, HouseholderReflectors { vectors, rows })
    }
}
