use float_cmp::approx_eq;
use num::complex::{Complex64, ComplexFloat};
use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::utilities::matrix::identity;

impl Matrix<f64> {
    /// Computes the upper Hessenberg form for square matrices.
    /// Returns (H, Q) where H is the Hessenberg form and Q is the accrued reflectors.
    pub fn upper_hessenberg(&self) -> Result<(Matrix<f64>, Matrix<f64>), TensorErrors> {
        if !self.is_square() {
            return Err(TensorErrors::NonSquareMatrix);
        }

        let ord = self.rows;

        // All 1x1 and 2x2 matrices are trivially upper Hessenberg
        if ord < 3 {
            return Ok((self.clone(), identity(ord)));
        }

        let (mut h, mut q) = (self.clone(), identity(ord));

        for i in 0..ord - 2 {
            let vec_bottom = h.slice(i+1..ord, i..i + 1)?;
            let alpha = -1.0
                * match vec_bottom[&[0, 0]] {
                0.0 => 1.0,
                x => x.signum(),
            } * vec_bottom.iter().map(|x| x * x).sum::<f64>().sqrt();
            let mut e1 = Matrix::<f64>::from_shape(vec_bottom.rows, vec_bottom.cols)?;
            e1[&[0, 0]] = 1.0;
            let v = vec_bottom - e1 * alpha;

            if v.iter().all(|x| approx_eq!(f64, *x, 0.0)) {
                continue;
            }

            let u = v.norm_l2();
            let u_clone = u.clone();
            let u_t = u_clone.transpose_mt();
            let reflector = identity::<f64>(u.shape.0.first().unwrap().clone()) - u
                .contract_mul(&u_t)?
                * 2.0;
            let reflector_transpose = reflector.transpose_mt();
            let reflector_ord = reflector.rows;

            // Update Q
            let q_slice = q.slice(0..ord, ord - reflector_ord..ord)?;
            let q_slice_res = q_slice.contract_mul_mt(&reflector)?;
            let mut q_slice_mut = q.slice_mut(0..ord, ord - reflector_ord..ord)?;
            q_slice_mut.set_all(&q_slice_res)?;

            // Update H Part 1: postmultiply by the reflector
            let h_slice = h.slice(0..ord, ord - reflector_ord..ord)?;
            let h_slice_res = h_slice.contract_mul_mt(&reflector)?;
            let mut h_slice_mut = h.slice_mut(0..ord, ord - reflector_ord..ord)?;
            h_slice_mut.set_all(&h_slice_res)?;

            // Update H Part 2: premultiply by the transpose of the reflector
            let h_slice = h.slice(ord - reflector_ord..ord, 0..ord)?;
            let h_slice_res = reflector_transpose.contract_mul_mt(&h_slice)?;
            let mut h_slice_mut = h.slice_mut(ord - reflector_ord..ord, 0..ord)?;
            h_slice_mut.set_all(&h_slice_res)?;
        }

        Ok((h, q))
    }

    /// Computes the lower Hessenberg form for square matrices
    /// Returns (H, Q) where H is the Hessenberg form and Q is the accrued reflectors.
    pub fn lower_hessenberg(&self) -> Result<(Matrix<f64>, Matrix<f64>), TensorErrors> {
        // Note that Q_l = Q_u and H_l = (H_u) ^ T
        let (h_u, q_u) = self.transpose_mt().upper_hessenberg()?;
        Ok((h_u.transpose_mt(), q_u))
    }
}

impl Matrix<Complex64> {
    /// Computes the upper Hessenberg form for square matrices.
    /// Returns (H, Q) where H is the Hessenberg form and Q is the accrued reflectors.
    pub fn upper_hessenberg(&self) -> Result<(Matrix<Complex64>, Matrix<Complex64>), TensorErrors> {
        if !self.is_square() {
            return Err(TensorErrors::NonSquareMatrix);
        }

        let ord = self.rows;

        // All 1x1 and 2x2 matrices are trivially upper Hessenberg
        if ord < 3 {
            return Ok((self.clone(), identity(ord)));
        }

        let (mut h, mut q) = (self.clone(), identity(ord));

        for i in 0..ord - 2 {
            let vec_bottom = h.slice(i+1..ord, i..i + 1)?;
            let alpha = Complex64::from(-1.0)
                * match vec_bottom[&[0, 0]] {
                Complex64::ZERO => Complex64::ONE,
                x => x / Complex64::abs(x),
            } * vec_bottom.iter().map(|x| <f64 as Into<Complex64>>::into((x * x).abs())).sum::<Complex64>().sqrt();
            let mut e1 = Matrix::<Complex64>::from_shape(vec_bottom.rows, vec_bottom.cols)?;
            e1[&[0, 0]] = Complex64::ONE;
            let v = vec_bottom - e1 * alpha;

            if v.iter().all(|x| approx_eq!(f64, x.abs(), 0.0)) {
                continue;
            }

            let u = v.norm_l2();
            let u_clone = u.clone();
            let u_t = u_clone.transpose_mt();
            let u_star = u_t.iter().map(|x| x.conj()).collect::<Matrix<Complex64>>().reshape(u_t.rows, u_t.cols)?;
            let reflector = identity::<Complex64>(u.shape.0[0].clone()) - u.mat_mul_mt(&u_star)?
                * Complex64 { re: 2.0, im: 0.0 };
            let reflector_star = reflector.conj_transpose_mt();
            let reflector_ord = reflector.rows;

            // Update Q
            let q_slice = q.slice(0..ord, ord - reflector_ord..ord);
            let q_slice_res = q_slice?.contract_mul_mt(&reflector)?;
            let mut q_slice_mut = q.slice_mut(0..ord, ord - reflector_ord..ord)?;
            q_slice_mut.set_all(&q_slice_res)?;

            // Update H Part 1: postmultiply by the reflector
            let h_slice = h.slice(0..ord, ord - reflector_ord..ord)?;
            let h_slice_res = h_slice.contract_mul_mt(&reflector)?;
            let mut h_slice_mut = h.slice_mut(0..ord, ord - reflector_ord..ord)?;
            h_slice_mut.set_all(&h_slice_res)?;

            // Update H Part 2: premultiply by the conjugate transpose of the reflector
            let h_slice = h.slice(ord - reflector_ord..ord, 0..ord)?;
            let h_slice_res = reflector_star.contract_mul_mt(&h_slice)?;
            let mut h_slice_mut = h.slice_mut(ord - reflector_ord..ord, 0..ord)?;
            h_slice_mut.set_all(&h_slice_res)?;
        }

        Ok((h, q))
    }

    /// Computes the lower Hessenberg form for square matrices.
    /// Returns (H, Q) where H is the Hessenberg form and Q is the accrued reflectors.
    pub fn lower_hessenberg(&self) -> Result<(Matrix<Complex64>, Matrix<Complex64>), TensorErrors> {
        // Note that Q_l = Q_u and H_l = (H_u) ^ T
        let (h_u, q_u) = self.conj_transpose_mt().upper_hessenberg()?;
        Ok((h_u.conj_transpose_mt(), q_u))
    }
}
