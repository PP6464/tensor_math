use crate::utilities::internal_functions::{for_each_chunk_mut, for_each_chunk_mut_pair};
use num::complex::Complex64;
use num::Zero;

const FFT_CHUNK_SIZE: usize = 4096;

/// Used to represent a plan for computing the FFT of a given length.
/// The plan contains precomputed twiddle factors and,
/// in the case of Bluestein's algorithm, the chirp filter and its FFT.
pub(crate) struct FFTPlan {
    n: usize,                 // Given length of the FFT
    l: usize,                 // padded length: == n (radix-2) or pow2 >= 2n-1 (Bluestein)
    twiddles: Vec<Complex64>, // l/2 entries: e^{-2πik/l}, each computed directly
    chirp: Vec<Complex64>,    // n entries: w_i = e^{-iπ i²/n}   (empty => radix-2 path)
    filter: Vec<Complex64>, // l entries: FFT of the chirp filter, bit-reversed order, scaled by 1/l
}

/// Computes the twiddle factors for a given length `l`.
fn make_twiddles(l: usize) -> Vec<Complex64> {
    (0..l / 2)
        .map(|k| Complex64::from_polar(1.0, -2.0 * std::f64::consts::PI * k as f64 / l as f64))
        .collect()
}

/// Computes the chirp filter for Bluestein's algorithm for a given length `n`.
/// The chirp filter is defined as w_i = e^{-iπ i²/n} for i = 0..<n.
fn make_chirp(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|i| {
            Complex64::from_polar(
                1.0,
                -std::f64::consts::PI * (i % (2 * n)).pow(2) as f64 / n as f64,
            )
        })
        .collect()
}

/// Computes the Decimation-In-Frequency (DIF) FFT of the input slice `x` in place, using the twiddle factors given.
/// The input is in the normal order, and the output is in bit-reversed order.
fn dif_fft(x: &mut [Complex64], twiddles: &[Complex64]) {
    let l = x.len();

    let mut half = l >> 1;
    while half >= 1 {
        let stride = (l >> 1) / half;
        let block_size = half << 1;
        for_each_chunk_mut(
            x,
            |_, chunk| {
                for block in chunk.chunks_exact_mut(block_size) {
                    let (low, high) = block.split_at_mut(half);
                    for_each_chunk_mut_pair(
                        low,
                        high,
                        |chunk_i, lows, highs| {
                            let start = chunk_i * FFT_CHUNK_SIZE;
                            for (i, (lo, hi)) in lows.iter_mut().zip(highs.iter_mut()).enumerate() {
                                let low = *lo; // Copy the value of *lo before modifying it
                                *lo += *hi;
                                *hi = (low - *hi) * twiddles[(start + i) * stride];
                            }
                        },
                        FFT_CHUNK_SIZE,
                    );
                }
            },
            block_size.max(FFT_CHUNK_SIZE),
        );
        half >>= 1;
    }
}

/// Computes the Decimation-In-Time (DIT) FFT of the input slice `x` in place, using the twiddle factors given.
/// The input is in bit-reversed order, and the output is in normal order. Note that setting CONJ uses conjugate
/// twiddle factors, which is equivalent to computing the inverse FFT (IFFT) up to a scaling factor of 1/n.
fn dit_fft<const CONJ: bool>(x: &mut [Complex64], twiddles: &[Complex64]) {
    let l = x.len();
    let mut half = 1;
    while half < l {
        let stride = l / (2 * half);
        let block_size = 2 * half;
        for_each_chunk_mut(
            x,
            |_, chunk| {
                for block in chunk.chunks_exact_mut(block_size) {
                    let (lo, hi) = block.split_at_mut(half);
                    for_each_chunk_mut_pair(
                        lo,
                        hi,
                        |chunk_i, lows, highs| {
                            let start = chunk_i * FFT_CHUNK_SIZE;
                            for (j, (x, y)) in lows.iter_mut().zip(highs.iter_mut()).enumerate() {
                                let w = if CONJ {
                                    twiddles[(start + j) * stride].conj()
                                } else {
                                    twiddles[(start + j) * stride]
                                };
                                let t = *y * w;
                                let u = *x;
                                *x = u + t;
                                *y = u - t;
                            }
                        },
                        FFT_CHUNK_SIZE,
                    );
                }
            },
            block_size.max(FFT_CHUNK_SIZE),
        );
        half <<= 1;
    }
}

/// Computes the bit-reversal permutation of the input slice `a` in place.
fn bit_reverse_permute(a: &mut [Complex64]) {
    let n = a.len(); // requires n >= 2, power of two
    let shift = usize::BITS - n.trailing_zeros();
    for i in 0..n {
        let r = i.reverse_bits() >> shift;
        if i < r {
            a.swap(i, r);
        }
    }
}

impl FFTPlan {
    /// Creates a new FFT plan for the given length.
    pub(crate) fn new(n: usize) -> Self {
        if n <= 1 || n.is_power_of_two() {
            return FFTPlan {
                n,
                l: n,
                twiddles: make_twiddles(n),
                chirp: vec![],
                filter: vec![],
            };
        }

        let l = (2 * n - 1).next_power_of_two();
        let twiddles = make_twiddles(l);
        let chirp = make_chirp(n);

        // Symmetric filter b[i] = b[l-i] = conj(w_i), then transform it ONCE.
        let mut filter = vec![Complex64::default(); l];
        filter[0] = chirp[0].conj();
        for i in 1..n {
            filter[i] = chirp[i].conj();
            filter[l - i] = chirp[i].conj();
        }

        dif_fft(&mut filter, &twiddles); // transform it once, left bit-reversed
        let scale = 1.0 / l as f64;
        filter.iter_mut().for_each(|f| *f *= scale); // fold the inverse's 1/l in here

        FFTPlan {
            n,
            l,
            twiddles,
            chirp,
            filter,
        }
    }

    /// Length of the per-worker scratch buffer this plan needs (0 on the radix-2 path).
    pub(crate) fn scratch_len(&self) -> usize {
        if self.chirp.is_empty() {
            0
        } else {
            self.l
        }
    }

    /// In-place forward FFT of one lane of length `n`.
    pub(crate) fn forward(&self, lane: &mut [Complex64], scratch: &mut [Complex64]) {
        debug_assert_eq!(lane.len(), self.n);
        if self.n <= 1 {
            return;
        }
        if self.chirp.is_empty() {
            bit_reverse_permute(lane);
            dit_fft::<false>(lane, &self.twiddles);
        } else {
            self.bluestein(lane, scratch);
        }
    }

    /// In-place inverse FFT via IFFT(x) = conj(FFT(conj(x))) / n, with no extra allocation.
    pub(crate) fn inverse(&self, lane: &mut [Complex64], scratch: &mut [Complex64]) {
        lane.iter_mut().for_each(|z| *z = z.conj());
        self.forward(lane, scratch);
        let s = 1.0 / self.n as f64;
        lane.iter_mut().for_each(|z| *z = z.conj() * s);
    }

    fn bluestein(&self, lane: &mut [Complex64], scratch: &mut [Complex64]) {
        let (n, l) = (self.n, self.l);
        let s = &mut scratch[..l];

        // u = x * chirp, zero padded to l (the only place the padding lives: the scratch)
        for ((d, &x), &w) in s[..n].iter_mut().zip(lane.iter()).zip(&self.chirp) {
            *d = x * w;
        }
        s[n..].fill(Complex64::zero());

        dif_fft(s, &self.twiddles); // natural -> bit-reversed
                                    // pointwise, same permuted order
        for (a, &f) in s.iter_mut().zip(&self.filter) {
            *a *= f;
        }
        dit_fft::<true>(s, &self.twiddles); // bit-reversed -> natural, conj twiddles

        for ((x, &r), &w) in lane.iter_mut().zip(s.iter()).zip(&self.chirp) {
            *x = r * w; // write the first n results back
        }
    }
}

#[cfg(test)]
mod fft_plan_tests {
    use std::f64::consts::FRAC_1_SQRT_2;

    #[test]
    fn bit_reverse_permute_test() {
        use super::bit_reverse_permute;
        use num::complex::Complex64;

        let mut a = (0..16)
            .map(|x| Complex64::new(x as f64, 0.0))
            .collect::<Vec<_>>();
        bit_reverse_permute(&mut a);
        assert_eq!(
            a,
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(8.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(12.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(10.0, 0.0),
                Complex64::new(6.0, 0.0),
                Complex64::new(14.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(9.0, 0.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(13.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(11.0, 0.0),
                Complex64::new(7.0, 0.0),
                Complex64::new(15.0, 0.0),
            ]
        );
    }

    #[test]
    fn make_twiddle_tests() {
        use super::make_twiddles;
        use num::complex::Complex64;

        let twiddles = make_twiddles(8);
        let expected = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(FRAC_1_SQRT_2, -FRAC_1_SQRT_2),
            Complex64::new(0.0, -1.0),
            Complex64::new(-FRAC_1_SQRT_2, -FRAC_1_SQRT_2),
        ];
        for (t, e) in twiddles.iter().zip(expected.iter()) {
            assert!((t.re - e.re).abs() < 1e-10);
            assert!((t.im - e.im).abs() < 1e-10);
        }
    }

    #[test]
    fn make_chirp_tests() {
        use super::make_chirp;
        use num::complex::Complex64;

        let chirp = make_chirp(4);
        let expected = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(FRAC_1_SQRT_2, -FRAC_1_SQRT_2),
            Complex64::new(-1.0, 0.0),
            Complex64::new(FRAC_1_SQRT_2, -FRAC_1_SQRT_2),
        ];
        for (c, e) in chirp.iter().zip(expected.iter()) {
            assert!((c.re - e.re).abs() < 1e-10);
            assert!((c.im - e.im).abs() < 1e-10);
        }
    }

    #[test]
    fn dit_fft_test() {
        use super::{dit_fft, make_twiddles};
        use num::complex::Complex64;

        let mut x = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(4.0, 0.0),
        ];
        let twiddles = make_twiddles(4);
        dit_fft::<false>(&mut x, &twiddles);
        let expected = vec![
            Complex64::new(10.0, 0.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(-2.0, 0.0),
            Complex64::new(-2.0, -2.0),
        ];
        for (a, e) in x.iter().zip(expected.iter()) {
            assert!((a.re - e.re).abs() < 1e-10);
            assert!((a.im - e.im).abs() < 1e-10);
        }
    }

    #[test]
    fn dif_fft_test() {
        use super::{dif_fft, make_twiddles};
        use num::complex::Complex64;

        let mut x = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ];
        let twiddles = make_twiddles(4);
        dif_fft(&mut x, &twiddles);
        let expected = vec![
            Complex64::new(10.0, 0.0),
            Complex64::new(-2.0, 0.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(-2.0, -2.0),
        ];
        for (a, e) in x.iter().zip(expected.iter()) {
            assert!((a.re - e.re).abs() < 1e-10);
            assert!((a.im - e.im).abs() < 1e-10);
        }
    }

    #[test]
    fn dit_conj_fft_test() {
        use super::{dit_fft, make_twiddles};
        use num::complex::Complex64;

        let mut x = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(4.0, 0.0),
        ];
        let twiddles = make_twiddles(4);
        dit_fft::<true>(&mut x, &twiddles);
        let expected = vec![
            Complex64::new(10.0, 0.0),
            Complex64::new(-2.0, -2.0),
            Complex64::new(-2.0, 0.0),
            Complex64::new(-2.0, 2.0),
        ];
        for (a, e) in x.iter().zip(expected.iter()) {
            assert!((a.re - e.re).abs() < 1e-10);
            assert!((a.im - e.im).abs() < 1e-10);
        }
    }

    #[test]
    fn round_trip_fft_test() {
        use super::FFTPlan;
        use num::complex::Complex64;

        let n = 8;
        let plan = FFTPlan::new(n);
        let mut x = (0..n)
            .map(|i| Complex64::new(i as f64, 0.0))
            .collect::<Vec<_>>();
        let mut scratch = vec![Complex64::default(); plan.scratch_len()];
        plan.forward(&mut x, &mut scratch);
        plan.inverse(&mut x, &mut scratch);
        for (i, &val) in x.iter().enumerate() {
            assert!((val.re - i as f64).abs() < 1e-10);
            assert!(val.im.abs() < 1e-10);
        }
    }

    #[test]
    fn radix2_fft_test() {
        use super::FFTPlan;
        use num::complex::Complex64;

        let n = 8; // power-of-two length
        let plan = FFTPlan::new(n);
        let mut x = (0..n)
            .map(|i| Complex64::new(i as f64, 0.0))
            .collect::<Vec<_>>();
        let mut scratch = vec![Complex64::default(); plan.scratch_len()];
        plan.forward(&mut x, &mut scratch);
        let ans = vec![
            Complex64::new(28.0, 0.0),
            Complex64::new(-4.0, 9.65685424949238),
            Complex64::new(-4.0, 4.0),
            Complex64::new(-4.0, 1.6568542494923806),
            Complex64::new(-4.0, 0.0),
            Complex64::new(-4.0, -1.6568542494923806),
            Complex64::new(-4.0, -4.0),
            Complex64::new(-4.0, -9.65685424949238),
        ];
        for (i, &val) in x.iter().enumerate() {
            assert!((val.re - ans[i].re).abs() < 1e-10);
            assert!((val.im - ans[i].im).abs() < 1e-10);
        }
    }

    #[test]
    fn bluestein_fft_test() {
        use super::FFTPlan;
        use num::complex::Complex64;

        let n = 60; // non-power-of-two length
        let plan = FFTPlan::new(n);
        let mut x = (0..n)
            .map(|i| Complex64::new(i as f64, 0.0))
            .collect::<Vec<_>>();
        let mut scratch = vec![Complex64::default(); plan.scratch_len()];
        plan.forward(&mut x, &mut scratch);
        plan.inverse(&mut x, &mut scratch);
        for (i, &val) in x.iter().enumerate() {
            assert!((val.re - i as f64).abs() < 1e-10);
            assert!(val.im.abs() < 1e-10);
        }
    }

    #[test]
    fn radix2_ifft_test() {
        use super::FFTPlan;
        use num::complex::Complex64;

        let n = 8; // power-of-two length
        let plan = FFTPlan::new(n);
        let mut x = (0..n)
            .map(|i| Complex64::new(i as f64, 0.0))
            .collect::<Vec<_>>();
        let mut scratch = vec![Complex64::default(); plan.scratch_len()];
        plan.inverse(&mut x, &mut scratch);
        plan.forward(&mut x, &mut scratch);
        for (i, &val) in x.iter().enumerate() {
            assert!((val.re - i as f64).abs() < 1e-10);
            assert!(val.im.abs() < 1e-10);
        }
    }

    #[test]
    fn bluestein_ifft_test() {
        use super::FFTPlan;
        use num::complex::Complex64;

        let n = 100; // non-power-of-two length
        let plan = FFTPlan::new(n);
        let mut x = (0..n)
            .map(|i| Complex64::new(i as f64, 0.0))
            .collect::<Vec<_>>();
        let mut scratch = vec![Complex64::default(); plan.scratch_len()];
        plan.inverse(&mut x, &mut scratch);
        plan.forward(&mut x, &mut scratch);
        for (i, &val) in x.iter().enumerate() {
            assert!((val.re - i as f64).abs() < 1e-10);
            assert!(val.im.abs() < 1e-10);
        }
    }

    #[test]
    fn large_fft_round_trip_test() {
        use super::FFTPlan;
        use num::complex::Complex64;

        for n in [8192, 4097] {
            let plan = FFTPlan::new(n);
            let mut x = (0..n)
                .map(|i| Complex64::new((i % 17) as f64, (i % 13) as f64))
                .collect::<Vec<_>>();
            let original = x.clone();
            let mut scratch = vec![Complex64::default(); plan.scratch_len()];

            plan.forward(&mut x, &mut scratch);
            plan.inverse(&mut x, &mut scratch);

            for (actual, expected) in x.iter().zip(original.iter()) {
                assert!((actual.re - expected.re).abs() < 1e-9);
                assert!((actual.im - expected.im).abs() < 1e-9);
            }
        }
    }
}
