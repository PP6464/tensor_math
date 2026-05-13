use std::collections::HashSet;
use std::f64::consts::PI;
use std::ops::{Add, Mul};
use num::complex::Complex64;
use num::{One, Zero};

/// This computes the dot product of two vectors of any type `T` that implements `Add` and `Mul`
pub(crate) fn dot_vectors<T: Add<Output = T> + Mul<Output = T> + Zero + Clone>(
    vec1: &Vec<T>,
    vec2: &Vec<T>,
) -> T {
    vec1.iter()
        .cloned()
        .zip(vec2.iter().cloned())
        .map(|(x, y)| x * y)
        .fold(T::zero(), T::add)
}

/// This computes the FFT of a vector of Complex64 values
/// where the length of the vector is a power of 2. This
pub(crate) fn radix_2_fft_vec(x: &Vec<Complex64>) -> Vec<Complex64> {
    let n = x.len();

    if n == 0 {
        return vec![];
    }

    let log2_n = n.trailing_zeros() as usize;
    let mut res = x.clone();

    let omega = Complex64::from_polar(1.0, -2.0 * PI / n as f64);

    // Bit-reverse indices
    let mut swapped = HashSet::<usize>::new();

    for i in 0..n {
        if swapped.contains(&i) {
            continue;
        }

        let mut orig = i;
        let mut rev = 0;

        for _ in 0..log2_n {
            rev <<= 1;
            rev |= orig & 1;
            orig >>= 1;
        }

        swapped.insert(i);
        swapped.insert(rev);

        let orig_copy = res[i];
        res[i] = res[rev];
        res[rev] = orig_copy;
    }

    // Compute twiddle factors
    let mut twiddle_factors: Vec<Complex64> = Vec::with_capacity(n);
    twiddle_factors.push(Complex64::one());

    for _ in 1..n {
        twiddle_factors.push(twiddle_factors.last().unwrap() * omega);
    }

    // FFT Butterfly
    for iters in 1..=log2_n {
        let half_len = (1 << iters) >> 1;

        for chunk in 0..n >> iters {
            for i in 0..half_len {
                let first_pos = (chunk << iters) + i;
                let second_pos = (chunk << iters) + i + half_len;

                let twiddle_index = (n >> iters) * (i & n - 1);
                let twiddle = twiddle_factors[twiddle_index];

                let first = res[first_pos];
                let second = res[second_pos];

                res[first_pos] = first + twiddle * second;
                res[second_pos] = first - twiddle * second;
            }
        }
    }

    res
}

/// Computes an FFT for an arbitrarily long vector using the Bluestein method.
pub(crate) fn bluestein_fft_vec(x: &Vec<Complex64>) -> Vec<Complex64> {
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
    conv_res = radix_2_fft_vec(&conv_res).iter().map(|x| x.conj() / l as f64).collect();

    for i in 0..n {
        res[i] = conv_res[i] * v_star[i];
    }

    res
}

/// Computes an FFT for an arbitrarily long vector, using radix 2 FFT
/// directly where appropriate, otherwise using the Bluestein method.
pub(crate) fn fft_vec(x: &Vec<Complex64>) -> Vec<Complex64> {
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
pub(crate) fn ifft_vec(x: &Vec<Complex64>) -> Vec<Complex64> {
    let n = x.len() as f64;
    if n == 0.0 {
        return vec![];
    }
    fft_vec(&x.iter().map(|z| z.conj()).collect()).iter().map(|z| z.conj() / n).collect()
}