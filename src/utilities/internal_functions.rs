use num::complex::Complex64;
use num::{One, Zero};
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use rayon::prelude::ParallelSliceMut;
use std::f64::consts::PI;
use std::ops::{Add, Mul};
use rayon::slice::ParallelSlice;

/// This computes the dot product of two vectors of any type `T` that implements `Add` and `Mul`
pub fn dot_vectors<T: Add<Output = T> + Mul<Output = T> + Zero + Clone>(
    vec1: &[T],
    vec2: &[T],
) -> T {
    // Use accumulators so that the compiler can optimise this when possible
    let mut acc = [T::zero(), T::zero(), T::zero(), T::zero()];

    let chunks = vec1.len() / 4;

    unsafe {
        for i in 0..chunks {
            let base = i * 4;
            *acc.get_unchecked_mut(0) = acc.get_unchecked(0).clone() + vec1.get_unchecked(base).clone() * vec2.get_unchecked(base).clone();
            *acc.get_unchecked_mut(1) = acc.get_unchecked(1).clone() + vec1.get_unchecked(base + 1).clone() * vec2.get_unchecked(base + 1).clone();
            *acc.get_unchecked_mut(2) = acc.get_unchecked(2).clone() + vec1.get_unchecked(base + 2).clone() * vec2.get_unchecked(base + 2).clone();
            *acc.get_unchecked_mut(3) = acc.get_unchecked(3).clone() + vec1.get_unchecked(base + 3).clone() * vec2.get_unchecked(base + 3).clone();
        }

        let mut total = acc.get_unchecked(0).clone()
            + acc.get_unchecked(1).clone()
            + acc.get_unchecked(2).clone()
            + acc.get_unchecked(3).clone();
        for i in (chunks * 4)..vec1.len() {
            total = total + vec1.get_unchecked(i).clone() * vec2.get_unchecked(i).clone();
        }
        total
    }
}

/// This computes the dot product of two vectors using many threads.
pub fn dot_vectors_mt<T: Add<Output = T> + Mul<Output = T> + Zero + Clone + Send + Sync>(
    vec1: &[T],
    vec2: &[T],
) -> T {
    vec1.par_chunks(4096)
        .zip(vec2.par_chunks(4096))
        .map(|(chunk1, chunk2)| dot_vectors(chunk1, chunk2))
        .reduce(|| T::zero(), |a, b| a + b)
}

/// This computes the FFT of a vector of Complex64 values
/// where the length of the vector is a power of 2. This
pub fn radix_2_fft_vec(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len();

    if n == 0 {
        return vec![];
    }

    let log2_n = n.trailing_zeros() as usize;
    let mut res = x.to_vec();

    let omega = Complex64::from_polar(1.0, -2.0 * PI / n as f64);

    // Bit-reverse indices
    res.par_iter_mut().enumerate().for_each(|(i, val)| {
        let mut orig = i;
        let mut rev = 0;

        for _ in 0..log2_n {
            rev <<= 1;
            rev |= orig & 1;
            orig >>= 1;
        }

        *val = x[rev];
    });

    // Compute twiddle factors
    let mut twiddle_factors: Vec<Complex64> = Vec::with_capacity(n);
    twiddle_factors.push(Complex64::one());

    for _ in 1..n {
        twiddle_factors.push(twiddle_factors.last().unwrap() * omega);
    }

    // FFT Butterfly
    for iters in 1..=log2_n {
        let half_len = (1 << iters) >> 1;

        res.par_chunks_mut(1 << iters).for_each(|chunk| {
            let (firsts, seconds) = chunk.split_at_mut(half_len);
            firsts
                .par_iter_mut()
                .zip(seconds.par_iter_mut())
                .enumerate()
                .for_each(|(i, (first_val, second_val))| {
                    let twiddle_index = (n >> iters) * (i & n - 1);
                    let twiddle = twiddle_factors[twiddle_index];

                    let first = first_val.clone();
                    let second = second_val.clone();

                    *first_val = first + twiddle * second;
                    *second_val = first - twiddle * second;
                });
        });
    }

    res
}

/// Computes an FFT for an arbitrarily long vector using the Bluestein method.
pub fn bluestein_fft_vec(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }

    let l = ((n << 1) - 1).next_power_of_two();

    let mut res = vec![Complex64::zero(); n];
    let mut u = vec![Complex64::zero(); l];
    let mut v = vec![Complex64::zero(); l];
    let mut v_star = vec![Complex64::zero(); n];

    for i in 0..n {
        let w_i = Complex64::from_polar(1.0, -PI * (i * i) as f64 / n as f64);

        u[i] = x[i] * w_i;
        v[i] = w_i.conj();
        v_star[i] = w_i;

        if i > 0 {
            v[l - i] = v[i];
        }
    }

    u = radix_2_fft_vec(&u);
    v = radix_2_fft_vec(&v);

    // Note that IFFT(x) = FFT(x.conj_each()).conj_each()
    let mut conv_res = (0..l).map(|i| (u[i] * v[i]).conj()).collect::<Vec<_>>();
    conv_res = radix_2_fft_vec(&conv_res)
        .iter()
        .map(|x| x.conj() / l as f64)
        .collect();

    for i in 0..n {
        res[i] = conv_res[i] * v_star[i];
    }

    res
}

/// Computes an FFT for an arbitrarily long vector, using radix 2 FFT
/// directly where appropriate, otherwise using the Bluestein method.
pub fn fft_vec(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }

    if n & n - 1 == 0 {
        radix_2_fft_vec(x)
    } else {
        bluestein_fft_vec(x)
    }
}

/// Computes an inverse FFT using the `fft` function and the identity
/// IFFT(x) === 1/N × FFT(x*)* where * means conjugating every element
/// and N is the size of the list x.
pub fn ifft_vec(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len() as f64;
    if n == 0.0 {
        return vec![];
    }
    fft_vec(&x.iter().map(|z| z.conj()).collect::<Vec<_>>())
        .iter()
        .map(|z| z.conj() / n)
        .collect()
}
