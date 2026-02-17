use std::cmp::min;
use float_cmp::approx_eq;
use num::complex::{Complex64, ComplexFloat};
use crate::definitions::matrix::Matrix;
use crate::utilities::matrix::identity;

impl Matrix<f64> {
    /// Computes the Householder transformation for the given matrix `t` (of shape (rows, cols)).
    /// Returns (Q, R), where Q is a unitary and Hermitian square matrix of shape (rows, rows)
    /// and R is an upper triangle matrix of shape (rows, cols) such that `Q.contract_mul(R) == t`.
    pub fn householder(&self) -> (Matrix<f64>, Matrix<f64>) {
        let (rows, cols) = (self.shape()[0], self.shape()[1]);

        let mut r = self.clone();
        let mut q = identity::<f64>(rows).unwrap();

        for k in 0..min(cols, rows) {
            let vec_bottom = r.slice(k..rows, k..k + 1).unwrap();
            let alpha = -1.0
                * match vec_bottom[&[0, 0]] {
                0.0 => 1.0,
                x => x.signum(),
            } * vec_bottom.iter().map(|x| x * x).sum::<f64>().sqrt();
            let mut e1 = Matrix::<f64>::from_shape(vec_bottom.rows, vec_bottom.cols).unwrap();
            e1[&[0, 0]] = 1.0;
            let v = vec_bottom - e1 * alpha;

            if v.iter().all(|x| approx_eq!(f64, x.abs(), 0.0)) {
                continue;
            }

            let u = v.norm_l2();
            let u_clone = u.clone();
            let u_t = u_clone.transpose_mt();
            let h_sub = identity::<f64>(u.shape.0.first().unwrap().clone()).unwrap() - u
                .contract_mul(&u_t)
                .unwrap()
                * 2.0;

            // Update R
            let r_slice_copy = r.slice(k..rows, k..cols).unwrap();
            let mut r_slice_mut = r.slice_mut(k..rows, k..cols).unwrap();
            let r_slice_res = h_sub.clone().contract_mul_mt(&r_slice_copy).unwrap();

            r_slice_mut.set_all(&r_slice_res).unwrap();

            // Update Q
            let q_slice_copy = q.slice(0..rows, k..rows).unwrap();
            let mut q_slice_mut = q.slice_mut(0..rows, k..rows).unwrap();
            let q_slice_res = q_slice_copy.contract_mul_mt(&h_sub).unwrap();

            q_slice_mut.set_all(&q_slice_res).unwrap();
        }

        (q, r)
    }
}

impl Matrix<Complex64> {
    /// Computes the Householder transformation for the given matrix `t` (of shape (rows, cols)).
    /// Returns (Q, R), where Q is a unitary and Hermitian square matrix of shape (rows, rows)
    /// and R is an upper triangle matrix of shape (rows, cols) such that `Q.contract_mul(R) == t`.
    pub fn householder(&self) -> (Matrix<Complex64>, Matrix<Complex64>) {
        let (rows, cols) = (self.shape()[0], self.shape()[1]);

        let mut r = self.clone();
        let mut q = identity::<Complex64>(rows).unwrap();

        for k in 0..min(cols, rows) {
            let vec_bottom = r.slice(k..rows, k..k + 1).unwrap();
            let alpha = -1.0
                * match vec_bottom[&[0, 0]] {
                Complex64::ZERO => Complex64::ONE,
                x => x / Complex64::abs(x),
            } * vec_bottom.iter().map(|x| Complex64::from((x * x).abs())).sum::<Complex64>().sqrt();
            let mut e1 = Matrix::<Complex64>::from_shape(vec_bottom.rows, vec_bottom.cols).unwrap();
            e1[&[0, 0]] = Complex64::ONE;
            let v = vec_bottom - e1 * alpha;

            if v.iter().all(|x| approx_eq!(f64, x.abs(), 0.0)) {
                continue;
            }

            let u = v.norm_l2();
            let u_clone = u.clone();
            let u_t = u_clone.transpose_mt();
            let u_star = u_t.iter().map(|x| x.conj()).collect::<Matrix<Complex64>>().reshape(u_t.rows, u_t.cols).unwrap();
            let h_sub = identity::<Complex64>(u.shape.0.first().unwrap().clone()).unwrap() - u
                .contract_mul(&u_star)
                .unwrap()
                * Complex64 { re: 2.0, im: 0.0 };

            // Update R
            let r_slice_copy = r.slice(k..rows, k..cols).unwrap();
            let mut r_slice_mut = r.slice_mut(k..rows, k..cols).unwrap();
            let r_slice_res = h_sub.clone().contract_mul_mt(&r_slice_copy).unwrap();

            r_slice_mut.set_all(&r_slice_res).unwrap();

            // Update Q
            let q_slice_copy = q.slice(0..rows, k..rows).unwrap();
            let mut q_slice_mut = q.slice_mut(0..rows, k..rows).unwrap();
            let q_slice_res = q_slice_copy.contract_mul_mt(&h_sub).unwrap();

            q_slice_mut.set_all(&q_slice_res).unwrap();
        }

        (q, r)
    }
}