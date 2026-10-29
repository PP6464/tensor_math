use num::complex::Complex64;
use num::{One, Zero};
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use rayon::prelude::ParallelSliceMut;
use rayon::slice::ParallelSlice;
use std::f64::consts::PI;
use std::ops::{AddAssign, Mul};

/// This computes the dot product of two vectors of any type `T` that implements `Add` and `Mul`
pub fn dot_vectors<T: AddAssign + Mul<Output = T> + Zero + Clone>(vec1: &[T], vec2: &[T]) -> T {
    // Use accumulators so that the compiler can optimise this when possible
    let mut acc = [T::zero(), T::zero(), T::zero(), T::zero()];

    let chunks = vec1.len() / 4;

    unsafe {
        for i in 0..chunks {
            let base = i * 4;
            *acc.get_unchecked_mut(0) +=
                vec1.get_unchecked(base).clone() * vec2.get_unchecked(base).clone();
            *acc.get_unchecked_mut(1) +=
                vec1.get_unchecked(base + 1).clone() * vec2.get_unchecked(base + 1).clone();
            *acc.get_unchecked_mut(2) +=
                vec1.get_unchecked(base + 2).clone() * vec2.get_unchecked(base + 2).clone();
            *acc.get_unchecked_mut(3) +=
                vec1.get_unchecked(base + 3).clone() * vec2.get_unchecked(base + 3).clone();
        }

        let mut total = acc.get_unchecked(0).clone()
            + acc.get_unchecked(1).clone()
            + acc.get_unchecked(2).clone()
            + acc.get_unchecked(3).clone();
        for i in (chunks * 4)..vec1.len() {
            total += vec1.get_unchecked(i).clone() * vec2.get_unchecked(i).clone();
        }
        total
    }
}

/// This computes the dot product of two vectors using many threads.
pub fn dot_vectors_mt<T: AddAssign + Mul<Output = T> + Zero + Clone + Send + Sync>(
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
    res.par_chunks_mut(4096)
        .enumerate()
        .for_each(|(chunk_i, vals)| {
            for i in chunk_i * 4096..chunk_i * 4096 + vals.len() {
                let mut orig = i;
                let mut rev = 0;

                for _ in 0..log2_n {
                    rev <<= 1;
                    rev |= orig & 1;
                    orig >>= 1;
                }

                vals[i & 4095] = x[rev];
            }
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
        let n_chunks = n >> iters;

        // Only parallelise when the work is significant
        if n_chunks >= 8 {
            res.par_chunks_mut(1 << iters).for_each(|chunk| {
                let (firsts, seconds) = chunk.split_at_mut(half_len);

                if half_len >= 512 {
                    firsts
                        .par_iter_mut()
                        .zip(seconds.par_iter_mut())
                        .enumerate()
                        .for_each(|(i, (first_val, second_val))| {
                            let twiddle_index = (n >> iters) * (i & (n - 1));
                            let twiddle = twiddle_factors[twiddle_index];

                            let first = *first_val;
                            let second = *second_val;

                            *first_val = first + twiddle * second;
                            *second_val = first - twiddle * second;
                        });
                } else {
                    firsts
                        .iter_mut()
                        .zip(seconds.iter_mut())
                        .enumerate()
                        .for_each(|(i, (first_val, second_val))| {
                            let twiddle_index = (n >> iters) * (i & (n - 1));
                            let twiddle = twiddle_factors[twiddle_index];

                            let first = *first_val;
                            let second = *second_val;

                            *first_val = first + twiddle * second;
                            *second_val = first - twiddle * second;
                        });
                }
            });
        } else {
            res.chunks_mut(1 << iters).for_each(|chunk| {
                let (firsts, seconds) = chunk.split_at_mut(half_len);
                if half_len >= 512 {
                    firsts
                        .par_iter_mut()
                        .zip(seconds.par_iter_mut())
                        .enumerate()
                        .for_each(|(i, (first_val, second_val))| {
                            let twiddle_index = (n >> iters) * (i & (n - 1));
                            let twiddle = twiddle_factors[twiddle_index];

                            let first = *first_val;
                            let second = *second_val;

                            *first_val = first + twiddle * second;
                            *second_val = first - twiddle * second;
                        });
                } else {
                    firsts
                        .iter_mut()
                        .zip(seconds.iter_mut())
                        .enumerate()
                        .for_each(|(i, (first_val, second_val))| {
                            let twiddle_index = (n >> iters) * (i & (n - 1));
                            let twiddle = twiddle_factors[twiddle_index];

                            let first = *first_val;
                            let second = *second_val;

                            *first_val = first + twiddle * second;
                            *second_val = first - twiddle * second;
                        });
                }
            });
        }
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

    if n & (n - 1) == 0 {
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


#[cfg(test)]
mod internal_functions_tests {
    use float_cmp::approx_eq;
    use num::complex::Complex64;
    use tensor_math::utilities::internal_functions::{
        bluestein_fft_vec, dot_vectors, dot_vectors_mt, fft_vec, ifft_vec, radix_2_fft_vec,
    };

    #[test]
    fn radix_2_vec_fft_test() {
        let v1 = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 2.0),
            Complex64::new(4.0, -5.0),
            Complex64::new(6.0, 7.0),
            Complex64::new(8.0, -4.0),
            Complex64::new(2.0, 1.0),
            Complex64::new(0.0, 0.0),
        ];
        let v2 = vec![
            Complex64 { re: 0.0, im: 1.0 },
            Complex64 {
                re: -0.1453217681275245,
                im: -3.6026214877097886,
            },
            Complex64 {
                re: -1.3968022466674206,
                im: 0.2212317420824741,
            },
            Complex64 {
                re: 6.39622598104463,
                im: -0.29714171603742257,
            },
            Complex64 {
                re: -2.3576391889522714,
                im: 10.365400978964416,
            },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
        ];
        let res1 = radix_2_fft_vec(&v1);
        let res2 = radix_2_fft_vec(&v2);

        let ans1 = vec![
            Complex64 { re: 26.0, im: 0.0 },
            Complex64 {
                re: -12.485281374238571,
                im: -0.9289321881345245,
            },
            Complex64 {
                re: 2.0,
                im: -2.0000000000000018,
            },
            Complex64 {
                re: -0.34314575050761853,
                im: -10.242640687119287,
            },
            Complex64 { re: -2.0, im: 20.0 },
            Complex64 {
                re: 4.485281374238571,
                im: -15.071067811865476,
            },
            Complex64 {
                re: 2.0,
                im: 10.000000000000002,
            },
            Complex64 {
                re: -11.656854249492381,
                im: -1.7573593128807126,
            },
        ];
        let ans2 = vec![
            Complex64 {
                re: 2.4964627772974133,
                im: 7.686869517299678,
            },
            Complex64 {
                re: 10.194430302685799,
                im: -4.794067509264503,
            },
            Complex64 {
                re: -4.804250848251911,
                im: -14.725982651422731,
            },
            Complex64 {
                re: -18.400909719645895,
                im: 0.9514637996861768,
            },
            Complex64 {
                re: -4.266316713957217,
                im: 17.685716986054096,
            },
            Complex64 {
                re: 13.946932312659404,
                im: 5.899646309853949,
            },
            Complex64 {
                re: 4.004431213373595,
                im: -12.84493282150859,
            },
            Complex64 {
                re: -15.4761803961386,
                im: -4.600523633014343,
            },
            Complex64 {
                re: -10.005345648536798,
                im: 15.4863959247941,
            },
            Complex64 {
                re: 8.873863904133222,
                im: 13.797591498439782,
            },
            Complex64 {
                re: 9.961992710321402,
                im: -1.211214813171261,
            },
            Complex64 {
                re: -0.041646627012200454,
                im: -2.0042344264809056,
            },
            Complex64 {
                re: 2.3446428293875154,
                im: 4.602621487709789,
            },
            Complex64 {
                re: 8.446377396379242,
                im: -1.472613543220143,
            },
            Complex64 {
                re: 0.2683836803659996,
                im: -8.679473629755085,
            },
            Complex64 {
                re: -7.542867173060969,
                im: 0.2227375039999857,
            },
        ];

        for i in 0..8 {
            assert!(approx_eq!(f64, res1[i].re, ans1[i].re, epsilon = 1e-10));
            assert!(approx_eq!(f64, res1[i].im, ans1[i].im, epsilon = 1e-10));
        }

        for i in 0..16 {
            assert!(approx_eq!(f64, res2[i].re, ans2[i].re, epsilon = 1e-10));
            assert!(approx_eq!(f64, res2[i].im, ans2[i].im, epsilon = 1e-10));
        }
    }

    #[test]
    fn large_ffts_test() {
        let v1 = (0..10000)
            .map(|i| Complex64::new(i as f64, (i % 10) as f64))
            .collect::<Vec<_>>();
        let res1 = fft_vec(&v1);
        let original = ifft_vec(&res1);

        for i in 0..10000 {
            assert!(approx_eq!(
                f64,
                original[i].re,
                v1[i].re,
                epsilon = 1e-5
            ));
            assert!(approx_eq!(
                f64,
                original[i].im,
                v1[i].im,
                epsilon = 1e-5
            ));
        }

        let v2 = (0..1<<15)
            .map(|i| Complex64::new((i % 10) as f64, i as f64))
            .collect::<Vec<_>>();

        let res2 = fft_vec(&v2);
        let original2 = ifft_vec(&res2);
        for i in 0..1<<15 {
            assert!(approx_eq!(
                f64,
                original2[i].re,
                v2[i].re,
                epsilon = 1e-5
            ));
            assert!(approx_eq!(
                f64,
                original2[i].im,
                v2[i].im,
                epsilon = 1e-5
            ));
        }
    }

    #[test]
    fn bluestein_fft_test() {
        let v1 = vec![
            Complex64::new(0.0, 1.0),
            Complex64::new(2.0, -3.0),
            Complex64::new(1.0, -1.0),
            Complex64::new(5.0, -4.0),
            Complex64::new(8.0, -7.0),
        ];
        let res = bluestein_fft_vec(&v1);
        let ans = vec![
            Complex64 {
                re: 16.0,
                im: -14.0,
            },
            Complex64 {
                re: 3.8036497995578236,
                im: 10.012395135066075,
            },
            Complex64 {
                re: -6.738096517215357,
                im: 7.267570420448962,
            },
            Complex64 {
                re: -5.734039437784222,
                im: 7.822599523300513,
            },
            Complex64 {
                re: -7.331513844558244,
                im: -6.102565078815551,
            },
        ];

        for i in 0..5 {
            assert!(approx_eq!(f64, res[i].re, ans[i].re, epsilon = 1e-10));
            assert!(approx_eq!(f64, res[i].im, ans[i].im, epsilon = 1e-10));
        }
    }

    #[test]
    fn fft_test() {
        let v1 = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 2.0),
            Complex64::new(4.0, -5.0),
            Complex64::new(6.0, 7.0),
            Complex64::new(8.0, -4.0),
            Complex64::new(2.0, 1.0),
            Complex64::new(0.0, 0.0),
        ];
        let v2 = vec![
            Complex64 { re: 0.0, im: 1.0 },
            Complex64 {
                re: -0.1453217681275245,
                im: -3.6026214877097886,
            },
            Complex64 {
                re: -1.3968022466674206,
                im: 0.2212317420824741,
            },
            Complex64 {
                re: 6.39622598104463,
                im: -0.29714171603742257,
            },
            Complex64 {
                re: -2.3576391889522714,
                im: 10.365400978964416,
            },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
        ];
        let v3 = vec![
            Complex64::new(0.0, 1.0),
            Complex64::new(2.0, -3.0),
            Complex64::new(1.0, -1.0),
            Complex64::new(5.0, -4.0),
            Complex64::new(8.0, -7.0),
        ];

        let res1 = fft_vec(&v1);
        let res2 = fft_vec(&v2);
        let res3 = fft_vec(&v3);

        let ans1 = vec![
            Complex64 { re: 26.0, im: 0.0 },
            Complex64 {
                re: -12.485281374238571,
                im: -0.9289321881345245,
            },
            Complex64 {
                re: 2.0,
                im: -2.0000000000000018,
            },
            Complex64 {
                re: -0.34314575050761853,
                im: -10.242640687119287,
            },
            Complex64 { re: -2.0, im: 20.0 },
            Complex64 {
                re: 4.485281374238571,
                im: -15.071067811865476,
            },
            Complex64 {
                re: 2.0,
                im: 10.000000000000002,
            },
            Complex64 {
                re: -11.656854249492381,
                im: -1.7573593128807126,
            },
        ];
        let ans2 = vec![
            Complex64 {
                re: 2.4964627772974133,
                im: 7.686869517299678,
            },
            Complex64 {
                re: 10.194430302685799,
                im: -4.794067509264503,
            },
            Complex64 {
                re: -4.804250848251911,
                im: -14.725982651422731,
            },
            Complex64 {
                re: -18.400909719645895,
                im: 0.9514637996861768,
            },
            Complex64 {
                re: -4.266316713957217,
                im: 17.685716986054096,
            },
            Complex64 {
                re: 13.946932312659404,
                im: 5.899646309853949,
            },
            Complex64 {
                re: 4.004431213373595,
                im: -12.84493282150859,
            },
            Complex64 {
                re: -15.4761803961386,
                im: -4.600523633014343,
            },
            Complex64 {
                re: -10.005345648536798,
                im: 15.4863959247941,
            },
            Complex64 {
                re: 8.873863904133222,
                im: 13.797591498439782,
            },
            Complex64 {
                re: 9.961992710321402,
                im: -1.211214813171261,
            },
            Complex64 {
                re: -0.041646627012200454,
                im: -2.0042344264809056,
            },
            Complex64 {
                re: 2.3446428293875154,
                im: 4.602621487709789,
            },
            Complex64 {
                re: 8.446377396379242,
                im: -1.472613543220143,
            },
            Complex64 {
                re: 0.2683836803659996,
                im: -8.679473629755085,
            },
            Complex64 {
                re: -7.542867173060969,
                im: 0.2227375039999857,
            },
        ];
        let ans3 = vec![
            Complex64 {
                re: 16.0,
                im: -14.0,
            },
            Complex64 {
                re: 3.8036497995578236,
                im: 10.012395135066075,
            },
            Complex64 {
                re: -6.738096517215357,
                im: 7.267570420448962,
            },
            Complex64 {
                re: -5.734039437784222,
                im: 7.822599523300513,
            },
            Complex64 {
                re: -7.331513844558244,
                im: -6.102565078815551,
            },
        ];

        for i in 0..8 {
            assert!(approx_eq!(f64, res1[i].re, ans1[i].re, epsilon = 1e-10));
            assert!(approx_eq!(f64, res1[i].im, ans1[i].im, epsilon = 1e-10));
        }

        for i in 0..16 {
            assert!(approx_eq!(f64, res2[i].re, ans2[i].re, epsilon = 1e-10));
            assert!(approx_eq!(f64, res2[i].im, ans2[i].im, epsilon = 1e-10));
        }

        for i in 0..5 {
            assert!(approx_eq!(f64, res3[i].re, ans3[i].re, epsilon = 1e-10));
            assert!(approx_eq!(f64, res3[i].im, ans3[i].im, epsilon = 1e-10));
        }
    }

    #[test]
    fn ifft_test() {
        let v1 = vec![
            Complex64 { re: 26.0, im: 0.0 },
            Complex64 {
                re: -12.485281374238571,
                im: -0.9289321881345245,
            },
            Complex64 {
                re: 2.0,
                im: -2.0000000000000018,
            },
            Complex64 {
                re: -0.34314575050761853,
                im: -10.242640687119287,
            },
            Complex64 { re: -2.0, im: 20.0 },
            Complex64 {
                re: 4.485281374238571,
                im: -15.071067811865476,
            },
            Complex64 {
                re: 2.0,
                im: 10.000000000000002,
            },
            Complex64 {
                re: -11.656854249492381,
                im: -1.7573593128807126,
            },
        ];
        let v2 = vec![
            Complex64 {
                re: 2.4964627772974133,
                im: 7.686869517299678,
            },
            Complex64 {
                re: 10.194430302685799,
                im: -4.794067509264503,
            },
            Complex64 {
                re: -4.804250848251911,
                im: -14.725982651422731,
            },
            Complex64 {
                re: -18.400909719645895,
                im: 0.9514637996861768,
            },
            Complex64 {
                re: -4.266316713957217,
                im: 17.685716986054096,
            },
            Complex64 {
                re: 13.946932312659404,
                im: 5.899646309853949,
            },
            Complex64 {
                re: 4.004431213373595,
                im: -12.84493282150859,
            },
            Complex64 {
                re: -15.4761803961386,
                im: -4.600523633014343,
            },
            Complex64 {
                re: -10.005345648536798,
                im: 15.4863959247941,
            },
            Complex64 {
                re: 8.873863904133222,
                im: 13.797591498439782,
            },
            Complex64 {
                re: 9.961992710321402,
                im: -1.211214813171261,
            },
            Complex64 {
                re: -0.041646627012200454,
                im: -2.0042344264809056,
            },
            Complex64 {
                re: 2.3446428293875154,
                im: 4.602621487709789,
            },
            Complex64 {
                re: 8.446377396379242,
                im: -1.472613543220143,
            },
            Complex64 {
                re: 0.2683836803659996,
                im: -8.679473629755085,
            },
            Complex64 {
                re: -7.542867173060969,
                im: 0.2227375039999857,
            },
        ];
        let v3 = vec![
            Complex64 {
                re: 16.0,
                im: -14.0,
            },
            Complex64 {
                re: 3.8036497995578236,
                im: 10.012395135066075,
            },
            Complex64 {
                re: -6.738096517215357,
                im: 7.267570420448962,
            },
            Complex64 {
                re: -5.734039437784222,
                im: 7.822599523300513,
            },
            Complex64 {
                re: -7.331513844558244,
                im: -6.102565078815551,
            },
        ];

        let ans1 = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 2.0),
            Complex64::new(4.0, -5.0),
            Complex64::new(6.0, 7.0),
            Complex64::new(8.0, -4.0),
            Complex64::new(2.0, 1.0),
            Complex64::new(0.0, 0.0),
        ];
        let ans2 = vec![
            Complex64 { re: 0.0, im: 1.0 },
            Complex64 {
                re: -0.1453217681275245,
                im: -3.6026214877097886,
            },
            Complex64 {
                re: -1.3968022466674206,
                im: 0.2212317420824741,
            },
            Complex64 {
                re: 6.39622598104463,
                im: -0.29714171603742257,
            },
            Complex64 {
                re: -2.3576391889522714,
                im: 10.365400978964416,
            },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
            Complex64 { re: 0.0, im: 0.0 },
        ];
        let ans3 = vec![
            Complex64::new(0.0, 1.0),
            Complex64::new(2.0, -3.0),
            Complex64::new(1.0, -1.0),
            Complex64::new(5.0, -4.0),
            Complex64::new(8.0, -7.0),
        ];

        let res1 = ifft_vec(&v1);
        let res2 = ifft_vec(&v2);
        let res3 = ifft_vec(&v3);

        for i in 0..8 {
            assert!(approx_eq!(f64, res1[i].re, ans1[i].re, epsilon = 1e-10));
            assert!(approx_eq!(f64, res1[i].im, ans1[i].im, epsilon = 1e-10));
        }

        for i in 0..16 {
            assert!(approx_eq!(f64, res2[i].re, ans2[i].re, epsilon = 1e-10));
            assert!(approx_eq!(f64, res2[i].im, ans2[i].im, epsilon = 1e-10));
        }

        for i in 0..5 {
            assert!(approx_eq!(f64, res3[i].re, ans3[i].re, epsilon = 1e-10));
            assert!(approx_eq!(f64, res3[i].im, ans3[i].im, epsilon = 1e-10));
        }
    }

    #[test]
    fn dot_vectors_test() {
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![4.0, 5.0, 6.0];
        let res = dot_vectors(&v1, &v2);
        assert!(approx_eq!(f64, res, 32.0, epsilon = 1e-10));

        let v3 = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
        let v4 = vec![Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)];

        let res2 = dot_vectors(&v3, &v4);
        assert!(approx_eq!(f64, res2.re, -18.0, epsilon = 1e-10));
        assert!(approx_eq!(f64, res2.im, 68.0, epsilon = 1e-10));

        let (v5, v6): (Vec<usize>, Vec<usize>) = (vec![], vec![]);
        let res = dot_vectors(&v5, &v6);
        assert_eq!(res, 0);
    }

    #[test]
    fn dot_vectors_mt_test() {
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![4.0, 5.0, 6.0];
        let res = dot_vectors_mt(&v1, &v2);
        assert!(approx_eq!(f64, res, 32.0, epsilon = 1e-10));

        let v3 = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
        let v4 = vec![Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)];

        let res2 = dot_vectors_mt(&v3, &v4);
        assert!(approx_eq!(f64, res2.re, -18.0, epsilon = 1e-10));
        assert!(approx_eq!(f64, res2.im, 68.0, epsilon = 1e-10));

        let (v5, v6): (Vec<usize>, Vec<usize>) = (vec![], vec![]);
        let res = dot_vectors_mt(&v5, &v6);
        assert_eq!(res, 0);
    }

    #[test]
    fn large_dot_test() {
        let v1 = (0..10000).map(|i| i).collect::<Vec<_>>();
        let v2 = (0..10000).map(|i| i % 10).collect::<Vec<_>>();
        let res = dot_vectors(&v1, &v2);
        let res_mt = dot_vectors_mt(&v1, &v2);
        assert_eq!(res, 225060000);
        assert_eq!(res_mt, 225060000);
    }

    #[test]
    fn fft_empty_vec() {
        let v: Vec<Complex64> = vec![];
        assert_eq!(fft_vec(&v).len(), 0);
        assert_eq!(ifft_vec(&v).len(), 0);
        assert_eq!(radix_2_fft_vec(&v).len(), 0);
        assert_eq!(bluestein_fft_vec(&v).len(), 0);
    }
}

#[cfg(test)]
mod noncommutative_dot_tests {
    use std::ops::{Add, AddAssign, Mul};
    use num::traits::Zero;
    use super::{dot_vectors, dot_vectors_mt};

    /// A two-component value whose `Mul` implementation is non-commutative.
    ///
    /// `NC(a, b) * NC(c, d) = NC(a*c, b*c + d)`, so swapping operands changes
    /// the second component whenever `a != c`. `Add` is the standard component-wise
    /// sum, so it is commutative — that isolates the test to the multiplication step.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct NC(i32, i32);

    impl Mul for NC {
        type Output = NC;
        fn mul(self, rhs: Self) -> Self::Output {
            NC(self.0 * rhs.0, self.1 * rhs.0 + rhs.1)
        }
    }

    impl Add for NC {
        type Output = NC;
        fn add(self, rhs: Self) -> Self::Output {
            NC(self.0 + rhs.0, self.1 + rhs.1)
        }
    }

    impl AddAssign for NC {
        fn add_assign(&mut self, rhs: Self) {
            self.0 += rhs.0;
            self.1 += rhs.1;
        }
    }

    impl Zero for NC {
        fn zero() -> Self {
            NC(0, 0)
        }
        fn is_zero(&self) -> bool {
            *self == NC(0, 0)
        }
    }

    /// Tiny constructor that keeps the call sites compact.
    fn nc(v: i32) -> NC {
        NC(v, 1)
    }

    #[test]
    fn noncommutative_dot() {
        let v1 = vec![nc(1), nc(2)];
        let v2 = vec![nc(3), nc(4)];
        let res = dot_vectors(&v1, &v2);
        let res_swapped = dot_vectors(&v2, &v1);
        assert_eq!(res, NC(11, 9));
        assert_eq!(res_swapped, NC(11, 5));
    }

    #[test]
    fn noncommutative_dot_mt() {
        let v1 = vec![nc(1), nc(2)];
        let v2 = vec![nc(3), nc(4)];
        let res = dot_vectors_mt(&v1, &v2);
        let res_swapped = dot_vectors_mt(&v2, &v1);
        assert_eq!(res, NC(11, 9));
        assert_eq!(res_swapped, NC(11, 5));
    }
}
