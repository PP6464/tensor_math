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
    b.as_immut().vec_mat_mul_unchecked_mt_into_slice(u, w); // w = uᵀB
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
    b.as_immut().vec_mat_mul_unchecked_mt_into_slice(uc, w); // w = u*B
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

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
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
                let mut blk = q.slice_unchecked_mut(k+1..n, k+1..n);
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
                let mut blk = q.slice_unchecked_mut(k+1..n, k+1..n);
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

        let (mut h, mut vectors) = (self, Vec::with_capacity(ord - 2));
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
                vectors.push(Matrix::from_shape(0, 0));
                continue;
            }
            u.iter_mut().for_each(|x| *x /= vnorm);

            unsafe {
                // right: H[:, s..] ← H[:, s..] (I − 2uuᵀ)
                let mut right = h.slice_unchecked_mut(0..ord, s..ord);
                reflect_right_c64(&mut right, &u, &mut uc, &mut y);
            }
            unsafe {
                // left: H[s.., s..] ← (I − 2uuᵀ) H[s.., s..]
                let mut left = h.slice_unchecked_mut(s..ord, s..ord);
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

#[cfg(test)]
mod hessenberg_tests {
    use float_cmp::approx_eq;
    use num::complex::{Complex64, ComplexFloat};
    use tensor_math::definitions::errors::TensorErrors;
    use tensor_math::definitions::matrix::Matrix;
    use tensor_math::utilities::matrix::eye;

    fn assert_f64_hessenberg_surfaces(input: Matrix<f64>) {
        let (upper, upper_q) = input.clone().upper_hessenberg().unwrap();
        let (upper_from_reflectors, upper_reflectors) = input.clone().upper_hessenberg_h().unwrap();
        let upper_q_from_reflectors = upper_reflectors.accumulate_q();
        assert!(approx_eq!(
            Matrix<f64>,
            upper.clone(),
            upper_from_reflectors,
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Matrix<f64>,
            upper_q.clone(),
            upper_q_from_reflectors,
            epsilon = 1e-10
        ));
        for row in 0..upper.rows() {
            for col in 0..upper.cols() {
                if row > col + 1 {
                    assert!(approx_eq!(f64, upper[(row, col)], 0.0, epsilon = 1e-10));
                }
            }
        }

        let (lower, lower_q) = input.clone().lower_hessenberg().unwrap();
        let (lower_from_reflectors, lower_reflectors) = input.clone().lower_hessenberg_h().unwrap();
        let lower_q_from_reflectors = lower_reflectors.accumulate_q();
        assert!(approx_eq!(
            Matrix<f64>,
            lower.clone(),
            lower_from_reflectors,
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Matrix<f64>,
            lower_q.clone(),
            lower_q_from_reflectors,
            epsilon = 1e-10
        ));
        for row in 0..lower.rows() {
            for col in 0..lower.cols() {
                if col > row + 1 {
                    assert!(approx_eq!(f64, lower[(row, col)], 0.0, epsilon = 1e-10));
                }
            }
        }

        if input.rows() != 0 {
            for (h, q) in [(upper, upper_q), (lower, lower_q)] {
                let reconstructed = q
                    .clone()
                    .mat_mul(h.clone())
                    .unwrap()
                    .mat_mul(q.transpose())
                    .unwrap();
                assert!(approx_eq!(
                    Matrix<f64>,
                    reconstructed,
                    input.clone(),
                    epsilon = 1e-10
                ));
            }
        }
    }

    fn assert_complex_hessenberg_surfaces(input: Matrix<Complex64>) {
        let (upper, upper_q) = input.clone().upper_hessenberg().unwrap();
        let (upper_from_reflectors, upper_reflectors) = input.clone().upper_hessenberg_h().unwrap();
        let upper_q_from_reflectors = upper_reflectors.accumulate_q();
        assert!(approx_eq!(
            Matrix<Complex64>,
            upper.clone(),
            upper_from_reflectors,
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Matrix<Complex64>,
            upper_q.clone(),
            upper_q_from_reflectors,
            epsilon = 1e-10
        ));
        for row in 0..upper.rows() {
            for col in 0..upper.cols() {
                if row > col + 1 {
                    assert!(approx_eq!(
                        f64,
                        upper[(row, col)].abs(),
                        0.0,
                        epsilon = 1e-10
                    ));
                }
            }
        }

        let (lower, lower_q) = input.clone().lower_hessenberg().unwrap();
        let (lower_from_reflectors, lower_reflectors) = input.clone().lower_hessenberg_h().unwrap();
        let lower_q_from_reflectors = lower_reflectors.accumulate_q();
        assert!(approx_eq!(
            Matrix<Complex64>,
            lower.clone(),
            lower_from_reflectors,
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Matrix<Complex64>,
            lower_q.clone(),
            lower_q_from_reflectors,
            epsilon = 1e-10
        ));
        for row in 0..lower.rows() {
            for col in 0..lower.cols() {
                if col > row + 1 {
                    assert!(approx_eq!(
                        f64,
                        lower[(row, col)].abs(),
                        0.0,
                        epsilon = 1e-10
                    ));
                }
            }
        }

        if input.rows() != 0 {
            for (h, q) in [(upper, upper_q), (lower, lower_q)] {
                let reconstructed = q
                    .clone()
                    .mat_mul(h.clone())
                    .unwrap()
                    .mat_mul(q.conj_transpose())
                    .unwrap();
                assert!(approx_eq!(
                    Matrix<Complex64>,
                    reconstructed,
                    input.clone(),
                    epsilon = 1e-10
                ));
            }
        }
    }

    #[test]
    fn upper_hessenberg() {
        let m1 = Matrix::<Complex64>::new(
            3,
            3,
            vec![
                Complex64 { re: 3.0, im: 0.0 },
                Complex64 { re: 2.0, im: 0.0 },
                Complex64 { re: 0.0, im: 4.0 },
                Complex64 { re: 0.0, im: 0.0 },
                Complex64 { re: 0.0, im: 5.0 },
                Complex64 { re: 0.0, im: 6.0 },
                Complex64 { re: -5.0, im: 4.0 },
                Complex64 { re: 7.0, im: 0.0 },
                Complex64 { re: 3.0, im: 0.0 },
            ],
        )
        .unwrap();
        let (h1, q1) = m1.clone().upper_hessenberg().unwrap();
        let h1_ans = Matrix::<Complex64>::new(
            3,
            3,
            vec![
                Complex64 { re: 3.0, im: 0.0 },
                Complex64 {
                    re: 2.498780190217697,
                    im: 3.123475237772121,
                },
                Complex64 {
                    re: 1.5617376188860606,
                    im: 1.2493900951088488,
                },
                Complex64 {
                    re: -6.4031242374328485,
                    im: 0.0,
                },
                Complex64 { re: 3.0, im: 0.0 },
                Complex64 {
                    re: 1.536585365853658,
                    im: 6.829268292682927,
                },
                Complex64 { re: 0.0, im: 0.0 },
                Complex64 {
                    re: 5.853658536585366,
                    im: 1.3170731707317078,
                },
                Complex64 { re: 0.0, im: 5.0 },
            ],
        )
        .unwrap();
        
        assert!(approx_eq!(
            Matrix<Complex64>,
            h1.clone(),
            h1_ans,
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Matrix<Complex64>,
            h1,
            q1.clone()
                .conj_transpose()
                .mat_mul_mt(m1.clone())
                .unwrap()
                .mat_mul_mt(q1.clone())
                .unwrap(),
            epsilon = 1e-10
        ));

        let m2 = Matrix::<f64>::new(
            4,
            4,
            vec![
                1.0, 0.5, 0.25, 0.125, 0.3, 0.6, 0.9, 1.2, 0.4, 0.8, 1.2, 1.6, -5.0, 10.0, -15.0,
                20.0,
            ],
        )
        .unwrap();

        let (h2, q2) = m2.clone().upper_hessenberg().unwrap();
        let h2_ans = Matrix::<f64>::new(
            4,
            4,
            vec![
                1.0,
                0.07462778926574923,
                -0.5099565599826192,
                0.25000000000000017,
                -5.024937810560445,
                20.21584158415841,
                4.158415841584154,
                16.92558260547191,
                0.0,
                -3.8415841584158397,
                1.5841584158415822,
                -1.5920595043359822,
                0.0,
                0.0,
                -6.661338147750939e-16,
                6.661338147750939e-16,
            ],
        )
        .unwrap();

        assert!(approx_eq!(Matrix<f64>, h2.clone(), h2_ans, epsilon = 1e-10));
        assert!(approx_eq!(
            Matrix<f64>,
            h2,
            q2.clone()
                .transpose()
                .mat_mul_mt(m2.clone())
                .unwrap()
                .mat_mul_mt(q2.clone())
                .unwrap(),
            epsilon = 1e-10
        ));
    }

    #[test]
    fn lower_hessenberg() {
        let m1 = Matrix::<Complex64>::new(
            3,
            3,
            vec![
                Complex64 { re: 3.0, im: 0.0 },
                Complex64 { re: 2.0, im: 0.0 },
                Complex64 { re: 0.0, im: 4.0 },
                Complex64 { re: 0.0, im: 0.0 },
                Complex64 { re: 0.0, im: 5.0 },
                Complex64 { re: 0.0, im: 6.0 },
                Complex64 { re: -5.0, im: 4.0 },
                Complex64 { re: 7.0, im: 0.0 },
                Complex64 { re: 3.0, im: 0.0 },
            ],
        )
        .unwrap();
        let (h1, q1) = m1.clone().lower_hessenberg().unwrap();
        let h1_ans = Matrix::<Complex64>::new(
            3,
            3,
            vec![
                Complex64 { re: 3.0, im: 0.0 },
                Complex64 {
                    re: -4.472135954999581,
                    im: -0.0,
                },
                Complex64 {
                    re: 0.0,
                    im: -2.220446049250313e-16,
                },
                Complex64 {
                    re: 3.5777087639996643,
                    im: 4.4721359549995805,
                },
                Complex64 {
                    re: 4.800000000000005,
                    im: 3.8000000000000074,
                },
                Complex64 {
                    re: -7.600000000000007,
                    im: -2.400000000000002,
                },
                Complex64 {
                    re: -2.23606797749979,
                    im: 1.788854381999832,
                },
                Complex64 {
                    re: 0.6000000000000011,
                    im: -3.6000000000000014,
                },
                Complex64 {
                    re: -1.8000000000000005,
                    im: 1.2000000000000008,
                },
            ],
        )
        .unwrap();

        assert!(approx_eq!(
            Matrix<Complex64>,
            h1.clone(),
            h1_ans,
            epsilon = 1e-10
        ));
        assert!(approx_eq!(
            Matrix<Complex64>,
            h1,
            q1.clone()
                .conj_transpose()
                .mat_mul_mt(m1.clone())
                .unwrap()
                .mat_mul_mt(q1.clone())
                .unwrap(),
            epsilon = 1e-10
        ));

        let m2 = Matrix::<f64>::new(
            4,
            4,
            vec![
                1.0, 0.5, 0.25, 0.125, 0.3, 0.6, 0.9, 1.2, 0.4, 0.8, 1.2, 1.6, -5.0, 10.0, -15.0,
                20.0,
            ],
        )
        .unwrap();

        let (h2, q2) = m2.clone().lower_hessenberg().unwrap();
        let h2_ans = Matrix::<f64>::new(
            4,
            4,
            vec![
                1.0,
                -0.57282196186948,
                0.0,
                0.0,
                0.6546536707079773,
                3.142857142857143,
                -6.375441828093274,
                0.0,
                -4.448647121025878,
                -4.805429258075459,
                21.194716078900154,
                -5.948528383276557,
                -2.242981801980734,
                -3.7073689241171914,
                12.403596185570388,
                -2.5375732217573193,
            ],
        )
        .unwrap();

        assert!(approx_eq!(Matrix<f64>, h2.clone(), h2_ans, epsilon = 1e-10));
        assert!(approx_eq!(
            Matrix<f64>,
            h2,
            q2.clone()
                .transpose()
                .mat_mul_mt(m2.clone())
                .unwrap()
                .mat_mul_mt(q2.clone())
                .unwrap(),
            epsilon = 1e-10
        ));
    }

    #[test]
    fn invalid_hessenberg_non_square() {
        let m1 = Matrix::<f64>::rand(2, 3);
        let m2 = m1.clone().map(Complex64::from);

        let err = m1.clone().upper_hessenberg().unwrap_err();
        assert_eq!(err, TensorErrors::NonSquareMatrix);
        assert_eq!(
            m1.clone().upper_hessenberg_h().err().unwrap(),
            TensorErrors::NonSquareMatrix
        );

        let err = m1.clone().lower_hessenberg().unwrap_err();
        assert_eq!(err, TensorErrors::NonSquareMatrix);
        assert_eq!(
            m1.clone().lower_hessenberg_h().err().unwrap(),
            TensorErrors::NonSquareMatrix
        );

        let err = m2.clone().upper_hessenberg().unwrap_err();
        assert_eq!(err, TensorErrors::NonSquareMatrix);
        assert_eq!(
            m2.clone().upper_hessenberg_h().err().unwrap(),
            TensorErrors::NonSquareMatrix
        );

        let err = m2.clone().lower_hessenberg().unwrap_err();
        assert_eq!(err, TensorErrors::NonSquareMatrix);
        assert_eq!(
            m2.clone().lower_hessenberg_h().err().unwrap(),
            TensorErrors::NonSquareMatrix
        );
    }

    #[test]
    fn hessenberg_rejects_rectangular_empty_shapes() {
        for (rows, cols) in [(0, 1), (1, 0), (0, 3), (3, 0)] {
            let real = Matrix::<f64>::zeros(rows, cols);
            assert_eq!(
                real.clone().upper_hessenberg().unwrap_err(),
                TensorErrors::NonSquareMatrix
            );
            assert_eq!(
                real.clone().upper_hessenberg_h().err().unwrap(),
                TensorErrors::NonSquareMatrix
            );
            assert_eq!(
                real.clone().lower_hessenberg().unwrap_err(),
                TensorErrors::NonSquareMatrix
            );
            assert_eq!(
                real.lower_hessenberg_h().err().unwrap(),
                TensorErrors::NonSquareMatrix
            );

            let complex = Matrix::<Complex64>::zeros(rows, cols);
            assert_eq!(
                complex.clone().upper_hessenberg().unwrap_err(),
                TensorErrors::NonSquareMatrix
            );
            assert_eq!(
                complex.clone().upper_hessenberg_h().err().unwrap(),
                TensorErrors::NonSquareMatrix
            );
            assert_eq!(
                complex.clone().lower_hessenberg().unwrap_err(),
                TensorErrors::NonSquareMatrix
            );
            assert_eq!(
                complex.lower_hessenberg_h().err().unwrap(),
                TensorErrors::NonSquareMatrix
            );
        }
    }

    #[test]
    fn hessenberg_trivial_cases() {
        let m1 = Matrix::<f64>::rand(2, 2);
        let m2 = Matrix::<f64>::rand(1, 1);

        assert_eq!(m1.clone().upper_hessenberg().unwrap().0, m1);
        assert_eq!(m2.clone().upper_hessenberg().unwrap().0, m2);
    }

    #[test]
    fn hessenberg_of_zero() {
        let m1 = Matrix::<f64>::zeros(5, 5);

        let (h, q) = m1.clone().upper_hessenberg().unwrap();

        assert_eq!(h, Matrix::zeros(5, 5));
        assert_eq!(q, eye(5));
    }

    #[test]
    fn hessenberg_of_zero_by_zero() {
        let m1 = Matrix::<f64>::zeros(0, 0);
        let (h, q) = m1.clone().upper_hessenberg().unwrap();
        assert_eq!(h, Matrix::zeros(0, 0));
        assert_eq!(q, eye(0));
        let (h, q) = m1.clone().lower_hessenberg().unwrap();
        assert_eq!(h, Matrix::zeros(0, 0));
        assert_eq!(q, eye(0));
    }

    #[test]
    fn hessenberg_intermediate_zero_reflector() {
        let m1 = Matrix::<f64>::new(
            5,
            5,
            vec![
                4.0, 1.0, 2.0, 3.0, 1.0, 0.0, 3.0, 1.0, 1.0, -1.0, 1.0, 1.0, 5.0, 2.0, 3.0, 2.0,
                1.0, 1.0, 4.0, 4.0, 2.0, 0.0, 1.0, 4.0, -2.0,
            ],
        )
        .unwrap();
        let (h1, q1) = m1.clone().upper_hessenberg().unwrap();
        for i in 0..h1.rows() {
            for j in 0..h1.cols() {
                if i > j + 1 {
                    assert!(approx_eq!(f64, h1[&[i, j]], 0.0, epsilon = 1e-10));
                }
            }
        }
        assert!(approx_eq!(
            Matrix<f64>,
            h1,
            q1.clone()
                .transpose()
                .mat_mul_mt(m1.clone())
                .unwrap()
                .mat_mul_mt(q1.clone())
                .unwrap(),
            epsilon = 1e-10
        ));
        assert!(!approx_eq!(
            Matrix<f64>,
            q1.clone(),
            eye(5),
            epsilon = 1e-10
        ));

        let (h1, q1) = m1.clone().transpose_mt().lower_hessenberg().unwrap();
        for i in 0..h1.rows() {
            for j in 0..h1.cols() {
                if j > i + 1 {
                    assert!(approx_eq!(f64, h1[&[i, j]], 0.0, epsilon = 1e-10));
                }
            }
        }
        assert!(approx_eq!(
            Matrix<f64>,
            h1,
            q1.clone()
                .transpose()
                .mat_mul_mt(m1.clone().clone().transpose_mt())
                .unwrap()
                .mat_mul_mt(q1.clone())
                .unwrap(),
            epsilon = 1e-10
        ));
        assert!(!approx_eq!(
            Matrix<f64>,
            q1.clone(),
            eye(5),
            epsilon = 1e-10
        ));

        let m2: Matrix<Complex64> = Matrix::<Complex64>::new(
            5,
            5,
            vec![
                Complex64::new(4.0, 1.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(3.0, 1.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(3.0, 1.0),
                Complex64::new(-2.0, 0.0),
                Complex64::new(0.0, 3.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(3.0, 2.0),
                Complex64::new(1.0, 1.0),
                Complex64::new(1.0, -1.0),
                Complex64::new(-1.0, 1.0),
                Complex64::new(2.0, 1.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(2.0, 1.0),
                Complex64::new(2.0, 2.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(4.0, 2.0),
                Complex64::new(2.0, 0.0),
            ],
        )
        .unwrap();
        let (h2, q2) = m2.clone().upper_hessenberg().unwrap();
        for i in 0..h2.rows() {
            for j in 0..h2.cols() {
                if i > j + 1 {
                    assert!(approx_eq!(f64, h2[&[i, j]].abs(), 0.0, epsilon = 1e-10));
                }
            }
        }
        assert!(approx_eq!(
            Matrix<Complex64>,
            h2,
            q2.clone()
                .conj_transpose()
                .mat_mul_mt(m2.clone())
                .unwrap()
                .mat_mul_mt(q2.clone())
                .unwrap(),
            epsilon = 1e-10
        ));
        assert!(!approx_eq!(
            Matrix<Complex64>,
            q2.clone(),
            eye(5),
            epsilon = 1e-10
        ));

        let (h2, q2) = m2.clone().conj_transpose_mt().upper_hessenberg().unwrap();
        for i in 0..h2.rows() {
            for j in 0..h2.cols() {
                if i > j + 1 {
                    assert!(approx_eq!(f64, h2[&[i, j]].abs(), 0.0, epsilon = 1e-10));
                }
            }
        }
        assert!(approx_eq!(
            Matrix<Complex64>,
            h2,
            q2.clone()
                .conj_transpose()
                .mat_mul_mt(m2.clone().conj_transpose_mt())
                .unwrap()
                .mat_mul_mt(q2.clone())
                .unwrap(),
            epsilon = 1e-10
        ));
        assert!(!approx_eq!(
            Matrix<Complex64>,
            q2.clone(),
            eye(5),
            epsilon = 1e-10
        ));
    }

    #[test]
    fn hessenberg_public_surfaces_cover_edge_shapes_and_noop_reflectors() {
        for real in [
            Matrix::<f64>::new(0, 0, vec![]).unwrap(),
            Matrix::<f64>::new(1, 1, vec![2.0]).unwrap(),
            Matrix::<f64>::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            Matrix::<f64>::zeros(4, 4),
            Matrix::<f64>::new(
                4,
                4,
                vec![
                    1.0, 2.0, 0.0, 1.0, //
                    0.0, 3.0, 4.0, 0.0, //
                    0.0, 5.0, 6.0, 2.0, //
                    0.0, 7.0, 8.0, 9.0,
                ],
            )
            .unwrap(),
        ] {
            assert_f64_hessenberg_surfaces(real);
        }

        let c = |re, im| Complex64::new(re, im);
        for complex in [
            Matrix::<Complex64>::new(0, 0, vec![]).unwrap(),
            Matrix::<Complex64>::new(1, 1, vec![c(2.0, -1.0)]).unwrap(),
            Matrix::<Complex64>::new(
                2,
                2,
                vec![c(1.0, 1.0), c(2.0, -1.0), c(3.0, 2.0), c(4.0, -2.0)],
            )
            .unwrap(),
            Matrix::<Complex64>::zeros(4, 4),
            Matrix::<Complex64>::new(
                4,
                4,
                vec![
                    c(1.0, 1.0),
                    c(2.0, 0.0),
                    c(0.0, 1.0),
                    c(1.0, -1.0),
                    c(0.0, 0.0),
                    c(3.0, -1.0),
                    c(4.0, 1.0),
                    c(0.0, 2.0),
                    c(0.0, 0.0),
                    c(5.0, 2.0),
                    c(6.0, 0.0),
                    c(2.0, -1.0),
                    c(0.0, 0.0),
                    c(7.0, -2.0),
                    c(8.0, 1.0),
                    c(9.0, 0.0),
                ],
            )
            .unwrap(),
        ] {
            assert_complex_hessenberg_surfaces(complex);
        }
    }
}
