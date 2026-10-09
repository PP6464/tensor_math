use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::definitions::matrix_slice_mut::MatrixSliceMut;
use crate::utilities::matrix::identity;
use float_cmp::approx_eq;
use num::complex::Complex64;
/*
--------------------------------------------
* Helpers
--------------------------------------------
*/

/// B ← (I − 2uuᵀ) B, in place. `w` is scratch with len >= B.cols().
unsafe fn reflect_left_f64(b: &mut MatrixSliceMut<'_, f64>, u: &[f64], w: &mut [f64]) {
    let w = w.get_unchecked_mut(..b.cols());
    b.as_immut().mat_vec_mul_unchecked_mt_into_slice(u, w); // w = uᵀB
    w.iter_mut().for_each(|x| *x *= 2.0);
    b.rank1_update_sub_unchecked_mt(u, w); // B −= u·w
}

/// B ← (I − 2uu*) B, in place. `uc` and `w` are reusable scratch.
unsafe fn reflect_left_c64(
    b: &mut MatrixSliceMut<'_, Complex64>,
    u: &[Complex64],
    uc: &mut Vec<Complex64>,
    w: &mut [Complex64],
) {
    uc.clear();
    uc.extend(u.iter().map(|x| x.conj()));
    let w = w.get_unchecked_mut(..b.cols());
    b.as_immut().mat_vec_mul_unchecked_mt_into_slice(uc, w); // w = u*B
    w.iter_mut().for_each(|x| *x *= 2.0);
    b.rank1_update_sub_unchecked_mt(u, w); // B −= u·w  (no conj)
}

/// B ← B (I − 2uuᵀ), in place. `y` is scratch with len >= B.rows().
unsafe fn reflect_right_f64(b: &mut MatrixSliceMut<'_, f64>, u: &[f64], y: &mut [f64]) {
    let y = &mut y[..b.rows()];
    b.as_immut().mat_vec_mul_unchecked_mt_into_slice(u, y); // y = B u
    y.iter_mut().for_each(|x| *x *= 2.0);
    b.rank1_update_sub_unchecked_mt(y, u); // B −= y·uᵀ
}

/// B ← B (I − 2uu*), in place. `uc` and `y` are reusable scratch.
unsafe fn reflect_right_c64(
    b: &mut MatrixSliceMut<'_, Complex64>,
    u: &[Complex64],
    uc: &mut Vec<Complex64>,
    y: &mut [Complex64],
) {
    let y = &mut y[..b.rows()];
    b.as_immut().mat_vec_mul_unchecked_mt_into_slice(u, y); // y = B u   (no conj)
    y.iter_mut().for_each(|x| *x *= 2.0);
    uc.clear();
    uc.extend(u.iter().map(|x| x.conj()));
    b.rank1_update_sub_unchecked_mt(y, uc); // B −= y·u*
}

/*
--------------------------------------------
* Hessenberg reflectors
--------------------------------------------
*/

pub struct HessenbergReflectors<T> {
    /// One reflector vector per step, stored at the row offset it applies from.
    /// vectors\[k\] has length (rows - k - 1).
    vectors: Vec<Matrix<T>>,
    rows: usize,
}

impl HessenbergReflectors<f64> {
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

impl HessenbergReflectors<Complex64> {
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
}

/*
--------------------------------------------
* Hessenberg methods
--------------------------------------------
*/

impl Matrix<f64> {
    /// Computes the upper Hessenberg form for square matrices.
    /// Returns (H, Q) where H is the Hessenberg form and Q is a unitary matrix such that
    /// `Q.mat_mul(H).mat_mul(Q.conj_transpose()) == self`.
    /// This fails if the matrix is not square.
    pub fn upper_hessenberg(self) -> Result<(Matrix<f64>, Matrix<f64>), TensorErrors> {
        let (h, reflectors) = self.upper_hessenberg_h()?;
        Ok((h, reflectors.accumulate_q()))
    }

    /// Computes the upper Hessenberg form for square matrices.
    /// Returns (H, reflectors) where H is the Hessenberg form and reflectors is something
    /// that can be accumulated to construct Q (with the provided method), where Q is a unitary
    /// matrix such that `Q.mat_mul(H).mat_mul(Q.conj_transpose()) == self`.
    /// This fails if the matrix is not square.
    pub fn upper_hessenberg_h(
        self,
    ) -> Result<(Matrix<f64>, HessenbergReflectors<f64>), TensorErrors> {
        if !self.is_square() {
            return Err(TensorErrors::NonSquareMatrix);
        }

        let ord = self.rows;

        // All 0x0, 1x1 and 2x2 matrices are trivially upper Hessenberg
        if ord < 3 {
            return Ok((
                self.clone(),
                HessenbergReflectors {
                    vectors: Vec::new(),
                    rows: ord,
                },
            ));
        }

        let (mut h, mut vectors) = (self, Vec::with_capacity(ord - 2));
        let mut w = vec![0.0; ord];
        let mut y = vec![0.0; ord];

        for i in 0..ord - 2 {
            let s = i + 1;
            let mut u: Vec<f64> = (s..ord).map(|r| h[(r, i)]).collect();
            let norm = u.iter().map(|x| x * x).sum::<f64>().sqrt();
            let alpha = -(if u[0] == 0.0 { 1.0 } else { u[0].signum() }) * norm;
            u[0] -= alpha;
            let vnorm = u.iter().map(|x| x * x).sum::<f64>().sqrt();
            if approx_eq!(f64, vnorm, 0.0) {
                vectors.push(Matrix::from_shape(0, 0));
                continue;
            }
            u.iter_mut().for_each(|x| *x /= vnorm);

            unsafe {
                // right: H[:, s..] ← H[:, s..] (I − 2uuᵀ)
                let mut right = h.slice_unchecked_mut(0..ord, s..ord);
                reflect_right_f64(&mut right, &u, &mut y);
            }
            unsafe {
                // left: H[s.., s..] ← (I − 2uuᵀ) H[s.., s..]
                let mut left = h.slice_unchecked_mut(s..ord, s..ord);
                reflect_left_f64(&mut left, &u, &mut w);
            }
            h[(s, i)] = alpha; // column i is known exactly
            for r in s + 1..ord {
                h[(r, i)] = 0.0;
            }

            vectors.push(Matrix {
                rows: ord - s,
                cols: 1,
                elements: u,
            });
        }
        Ok((h, HessenbergReflectors { vectors, rows: ord }))
    }

    /// Computes the lower Hessenberg form for square matrices.
    /// Returns (H, Q) where H is the Hessenberg form and Q is a unitary matrix such that
    /// `Q.mat_mul(H).mat_mul(Q.conj_transpose()) == self`.
    /// This fails if the matrix is not square.
    pub fn lower_hessenberg(self) -> Result<(Matrix<f64>, Matrix<f64>), TensorErrors> {
        // Note that Q_l = Q_u and H_l = (H_u) ^ T
        let (h_u, q) = self.transpose_mt().upper_hessenberg()?;
        Ok((h_u.transpose_mt(), q))
    }

    /// Computes the lower Hessenberg form for square matrices.
    /// Returns (H, reflectors) where H is the Hessenberg form and reflectors is something
    /// that can be accumulated to construct Q (with the provided method), where Q is a unitary
    /// matrix such that `Q.mat_mul(H).mat_mul(Q.conj_transpose()) == self`.
    /// This fails if the matrix is not square.
    pub fn lower_hessenberg_h(
        self,
    ) -> Result<(Matrix<f64>, HessenbergReflectors<f64>), TensorErrors> {
        // Note that Q_l = Q_u so reflectors are the same and H_l = (H_u) ^ T
        let (h_u, reflectors) = self.transpose_mt().upper_hessenberg_h()?;
        Ok((h_u.transpose_mt(), reflectors))
    }
}

impl Matrix<Complex64> {
    /// Computes the upper Hessenberg form for square matrices.
    /// Returns (H, Q) where H is the Hessenberg form and Q is a unitary matrix such that
    /// `Q.mat_mul(H).mat_mul(Q.conj_transpose()) == self`.
    /// This fails if the matrix is not square.
    pub fn upper_hessenberg(self) -> Result<(Matrix<Complex64>, Matrix<Complex64>), TensorErrors> {
        let (h, reflectors) = self.upper_hessenberg_h()?;
        Ok((h, reflectors.accumulate_q()))
    }

    /// Computes the upper Hessenberg form for square matrices.
    /// Returns (H, reflectors) where H is the Hessenberg form and reflectors is something
    /// that can be accumulated to construct Q (with the provided method), where Q is a unitary
    /// matrix such that `Q.mat_mul(H).mat_mul(Q.conj_transpose()) == self`.
    /// This fails if the matrix is not square.
    pub fn upper_hessenberg_h(
        self,
    ) -> Result<(Matrix<Complex64>, HessenbergReflectors<Complex64>), TensorErrors> {
        if !self.is_square() {
            return Err(TensorErrors::NonSquareMatrix);
        }

        let ord = self.rows;

        // All 0x0, 1x1 and 2x2 matrices are trivially upper Hessenberg
        if ord < 3 {
            return Ok((
                self.clone(),
                HessenbergReflectors {
                    vectors: Vec::new(),
                    rows: ord,
                },
            ));
        }

        let (mut h, mut vectors) = (self.clone(), Vec::with_capacity(ord - 2));
        let mut uc = Vec::with_capacity(ord);
        let mut w = vec![Complex64::ZERO; ord];
        let mut y = vec![Complex64::ZERO; ord];

        for i in 0..ord - 2 {
            let s = i + 1;
            let mut u: Vec<Complex64> = (s..ord).map(|r| h[(r, i)]).collect();
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
                vectors.push(Matrix::from_shape(0, 0)); // see the bug note below
                continue;
            }
            u.iter_mut().for_each(|x| *x /= vnorm);

            unsafe {
                // right: H[:, s..] ← H[:, s..] (I − 2uuᵀ)
                let mut right = h.slice_mut(0..ord, s..ord)?;
                reflect_right_c64(&mut right, &u, &mut uc, &mut y);
            }
            unsafe {
                // left: H[s.., s..] ← (I − 2uuᵀ) H[s.., s..]
                let mut left = h.slice_mut(s..ord, s..ord)?;
                reflect_left_c64(&mut left, &u, &mut uc, &mut w);
            }
            h[(s, i)] = alpha; // column i is known exactly
            for r in s + 1..ord {
                h[(r, i)] = Complex64::ZERO;
            }

            vectors.push(Matrix {
                rows: ord - s,
                cols: 1,
                elements: u,
            });
        }
        Ok((h, HessenbergReflectors { vectors, rows: ord }))
    }

    /// Computes the lower Hessenberg form for square matrices.
    /// Returns (H, Q) where H is the Hessenberg form and Q is a unitary matrix such that
    /// `Q.mat_mul(H).mat_mul(Q.conj_transpose()) == self`.
    /// This fails if the matrix is not square.
    pub fn lower_hessenberg(self) -> Result<(Matrix<Complex64>, Matrix<Complex64>), TensorErrors> {
        // Note that Q_l = Q_u and H_l = (H_u) ^ *
        let (h_u, q) = self.conj_transpose_mt().upper_hessenberg()?;
        Ok((h_u.conj_transpose_mt(), q))
    }

    /// Computes the lower Hessenberg form for square matrices.
    /// Returns (H, reflectors) where H is the Hessenberg form and reflectors is something
    /// that can be accumulated to construct Q (with the provided method), where Q is a unitary
    /// matrix such that `Q.mat_mul(H).mat_mul(Q.conj_transpose()) == self`.
    /// This fails if the matrix is not square.
    pub fn lower_hessenberg_h(
        self,
    ) -> Result<(Matrix<Complex64>, HessenbergReflectors<Complex64>), TensorErrors> {
        // Note that Q_l = Q_u so reflectors are the same and H_l = (H_u) ^ *
        let (h_u, reflectors) = self.conj_transpose_mt().upper_hessenberg_h()?;
        Ok((h_u.conj_transpose_mt(), reflectors))
    }
}
