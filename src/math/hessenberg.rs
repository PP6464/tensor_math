use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::utilities::matrix::identity;
use float_cmp::approx_eq;
use num::complex::{Complex64, ComplexFloat};

pub struct HessenbergReflectors<T> {
    /// One reflector vector per step, stored at the row offset it applies from.
    /// vectors\[k\] has length (rows - k - 1).
    vectors: Vec<Matrix<T>>,
    rows: usize,
}

impl HessenbergReflectors<f64> {
    /// Accumulate the reflectors into a matrix Q
    pub fn accumulate_q(&self) -> Matrix<f64> {
        let mut q = identity::<f64>(self.rows);
        for (k, u) in self.vectors.iter().enumerate() {
            if u.rows == 0 { continue; } // skip no-op steps (v was ~0)
            let u_t = u.transpose_mt();
            let q_slice_copy = q.slice(0..self.rows, (k + 1)..self.rows).unwrap();
            let mut q_slice_mut = q.slice_mut(0..self.rows, (k + 1)..self.rows).unwrap();
            let q_u = q_slice_copy.mat_mul_mt(u).unwrap();
            let q_slice_res = q_slice_copy - q_u.mat_mul(&u_t).unwrap() * 2.0;
            q_slice_mut.set_all(&q_slice_res).unwrap();
        }
        q
    }
}

impl HessenbergReflectors<Complex64> {
    /// Accumulate the reflectors into a matrix Q
    pub fn accumulate_q(&self) -> Matrix<Complex64> {
        let mut q = identity::<Complex64>(self.rows);
        for (k, u) in self.vectors.iter().enumerate() {
            if u.rows == 0 { continue; }
            let u_star = u.conj_transpose_mt();
            let q_slice_copy = q.slice(0..self.rows, (k + 1)..self.rows).unwrap();
            let mut q_slice_mut = q.slice_mut(0..self.rows, (k + 1)..self.rows).unwrap();
            let q_u = q_slice_copy.mat_mul_mt(u).unwrap();
            let q_slice_res = q_slice_copy - q_u.mat_mul_mt(&u_star).unwrap() * Complex64 { re: 2.0, im: 0.0 };
            q_slice_mut.set_all(&q_slice_res).unwrap();
        }
        q
    }
}

impl Matrix<f64> {
    /// Computes the upper Hessenberg form for square matrices.
    /// Returns (H, Q) where H is the Hessenberg form and Q is a unitary matrix such that
    /// `Q.mat_mul(H).mat_mul(Q.conj_transpose()) == self`.
    /// This fails if the matrix is not square.
    pub fn upper_hessenberg(&self) -> Result<(Matrix<f64>, Matrix<f64>), TensorErrors> {
        let (h, reflectors) = self.upper_hessenberg_h()?;
        Ok((h, reflectors.accumulate_q()))
    }

    /// Computes the upper Hessenberg form for square matrices.
    /// Returns (H, reflectors) where H is the Hessenberg form and reflectors is something
    /// that can be accumulated to construct Q (with the provided method), where Q is a unitary
    /// matrix such that `Q.mat_mul(H).mat_mul(Q.conj_transpose()) == self`.
    /// This fails if the matrix is not square.
    pub fn upper_hessenberg_h(&self) -> Result<(Matrix<f64>, HessenbergReflectors<f64>), TensorErrors> {
        if !self.is_square() {
            return Err(TensorErrors::NonSquareMatrix);
        }

        let ord = self.rows;

        // All 0x0, 1x1 and 2x2 matrices are trivially upper Hessenberg
        if ord < 3 {
            return Ok((self.clone(), HessenbergReflectors { vectors: Vec::new(), rows: ord } ));
        }

        let (mut h, mut vectors) = (self.clone(), Vec::with_capacity(ord - 2));

        for i in 0..ord - 2 {
            let vec_bottom = h.slice(i + 1..ord, i..i + 1)?;
            let alpha =
                -1.0 * match vec_bottom[&[0, 0]] {
                    0.0 => 1.0,
                    x => x.signum(),
                } * vec_bottom.clone().mag();
            let mut e1 = Matrix::<f64>::from_shape(vec_bottom.rows, vec_bottom.cols);
            e1[&[0, 0]] = 1.0;
            let v = vec_bottom - e1 * alpha;

            if v.iter().all(|x| approx_eq!(f64, *x, 0.0)) {
                vectors.push(Matrix::from_shape(0, 0));
                continue;
            }

            let u = v.norm_l2();
            let u_t = u.transpose_mt();
            // Start position for the reflector matrix
            let reflector_start = i + 1;

            // Update H
            // We can expand H_{n+1} = R* H_n R into H_{n+1} = H_n - 2 H_n u u* -2 u u* H_n
            //     + 4 u u* H_n u u*
            // Now if we define:
            // a = H_n u
            // b = u* H_n
            // s = u* a (this is just a scalar)
            // c = a - 2 s u
            // Then the formula becomes
            // H_{n+1} = H_n - 2 u b - 2 c u*
            // And we avoid massive matrix multiplications

            // Calculate a
            let a = h.slice(0..ord, reflector_start..ord)?.mat_mul_mt(&u)?;
            // Calculate b
            let b = u_t.mat_mul_mt(&h.slice(reflector_start..ord, 0..ord)?)?;
            // Calculate s
            let s = u.dot(&a.slice(reflector_start..ord, 0..1)?)?;
            // Calculate c
            let slice_copy = a.slice(reflector_start..ord, 0..1)?;
            let mut c = a;
            c
                .slice_mut(reflector_start..ord, 0..1)?
                .set_all(&(slice_copy - u.clone() * s * 2.0))?;
            // Subtract 2 u b
            let h_slice_copy = h.slice(reflector_start..ord, 0..ord)?;
            h
                .slice_mut(reflector_start..ord, 0..ord)?
                .set_all(&(h_slice_copy - u.mat_mul_mt(&(b * 2.0))?))?;
            // Subtract 2 c u*
            let h_slice_copy = h.slice(0..ord, reflector_start..ord)?;
            h
                .slice_mut(0..ord, reflector_start..ord)?
                .set_all(&(h_slice_copy - c.mat_mul_mt(&(u_t * 2.0))?))?;

            // Add the new vector
            vectors.push(u);
        }

        Ok((h, HessenbergReflectors { vectors, rows: ord }))
    }

    /// Computes the lower Hessenberg form for square matrices
    /// Returns (H, Q) where H is the Hessenberg form and Q is the accrued reflectors.
    /// This fails if the matrix is not square.
    pub fn lower_hessenberg(&self) -> Result<(Matrix<f64>, Matrix<f64>), TensorErrors> {
        // Note that Q_l = Q_u and H_l = (H_u) ^ T
        let (h_u, q_u) = self.transpose_mt().upper_hessenberg()?;
        Ok((h_u.transpose_mt(), q_u))
    }
}

impl Matrix<Complex64> {
    /// Computes the upper Hessenberg form for square matrices.
    /// Returns (H, Q) where H is the Hessenberg form and Q is a unitary matrix such that
    /// `Q.mat_mul(H).mat_mul(Q.conj_transpose()) == self`.
    /// This fails if the matrix is not square.
    pub fn upper_hessenberg(&self) -> Result<(Matrix<Complex64>, Matrix<Complex64>), TensorErrors> {
        let (h, reflectors) = self.upper_hessenberg_h()?;
        Ok((h, reflectors.accumulate_q()))
    }

    /// Computes the upper Hessenberg form for square matrices.
    /// Returns (H, reflectors) where H is the Hessenberg form and reflectors is something
    /// that can be accumulated to construct Q (with the provided method), where Q is a unitary
    /// matrix such that `Q.mat_mul(H).mat_mul(Q.conj_transpose()) == self`.
    /// This fails if the matrix is not square.
    pub fn upper_hessenberg_h(&self) -> Result<(Matrix<Complex64>, HessenbergReflectors<Complex64>), TensorErrors> {
        if !self.is_square() {
            return Err(TensorErrors::NonSquareMatrix);
        }

        let ord = self.rows;

        // All 0x0, 1x1 and 2x2 matrices are trivially upper Hessenberg
        if ord < 3 {
            return Ok((self.clone(), HessenbergReflectors { vectors: Vec::new(), rows: ord }));
        }

        let (mut h, mut vectors) = (self.clone(), Vec::with_capacity(ord - 2));

        for i in 0..ord - 2 {
            let vec_bottom = h.slice(i + 1..ord, i..i + 1)?;
            let alpha = Complex64::from(-1.0)
                * match vec_bottom[&[0, 0]] {
                    Complex64::ZERO => Complex64::ONE,
                    x => x / Complex64::abs(x),
                }
                * vec_bottom
                    .iter()
                    .map(|x| <f64 as Into<Complex64>>::into((x * x).abs()))
                    .sum::<Complex64>()
                    .sqrt();
            let mut e1 = Matrix::<Complex64>::from_shape(vec_bottom.rows, vec_bottom.cols);
            e1[&[0, 0]] = Complex64::ONE;
            let v = vec_bottom - e1 * alpha;

            if v.iter().all(|x| approx_eq!(f64, x.abs(), 0.0)) {
                continue;
            }

            let u = v.norm_l2();
            let u_star = u.conj_transpose_mt();
            // Start position for the reflector matrix
            let reflector_start = i + 1;

            // Update H
            // We can expand H_{n+1} = R* H_n R into H_{n+1} = H_n - 2 H_n u u* -2 u u* H_n
            //     + 4 u u* H_n u u*
            // Now if we define:
            // a = H_n u
            // b = u* H_n
            // s = u* a (this is just a scalar)
            // c = a - 2 s u
            // Then the formula becomes
            // H_{n+1} = H_n - 2 u b - 2 c u*
            // And we avoid massive matrix multiplications

            // Calculate a
            let a = h.slice(0..ord, reflector_start..ord)?.mat_mul_mt(&u)?;
            // Calculate b
            let b = u_star.mat_mul_mt(&h.slice(reflector_start..ord, 0..ord)?)?;
            // Calculate s
            let s = u.clone().conj().dot(&a.slice(reflector_start..ord, 0..1)?)?;
            // Calculate c
            let slice_copy = a.slice(reflector_start..ord, 0..1)?;
            let mut c = a;
            c
                .slice_mut(reflector_start..ord, 0..1)?
                .set_all(&(slice_copy - u.clone() * s * Complex64::new(2.0, 0.0)))?;
            // Subtract 2 u b
            let h_slice_copy = h.slice(reflector_start..ord, 0..ord)?;
            h
                .slice_mut(reflector_start..ord, 0..ord)?
                .set_all(&(h_slice_copy - u.mat_mul_mt(&(b * Complex64::new(2.0, 0.0)))?))?;
            // Subtract 2 c u*
            let h_slice_copy = h.slice(0..ord, reflector_start..ord)?;
            h
                .slice_mut(0..ord, reflector_start..ord)?
                .set_all(&(h_slice_copy - c.mat_mul_mt(&(u_star * Complex64::new(2.0, 0.0)))?))?;

            // Add the new vector
            vectors.push(u);
        }

        Ok((h, HessenbergReflectors { vectors, rows: ord }))
    }

    /// Computes the lower Hessenberg form for square matrices.
    /// Returns (H, Q) where H is the Hessenberg form and Q is the accrued reflectors.
    /// This fails if the matrix is not square.
    pub fn lower_hessenberg(&self) -> Result<(Matrix<Complex64>, Matrix<Complex64>), TensorErrors> {
        // Note that Q_l = Q_u and H_l = (H_u) ^ *
        let (h_u, q_u) = self.conj_transpose_mt().upper_hessenberg()?;
        Ok((h_u.conj_transpose_mt(), q_u))
    }
}
