use num::Zero;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::prelude::ParallelSliceMut;
use rayon::slice::ParallelSlice;
use std::ops::{AddAssign, Mul};

const PARALLEL_THRESHOLD: usize = 4096;

/// This computes the dot product of two vectors of any type `T` that implements `Add` and `Mul`
pub(crate) fn dot_vectors<T: AddAssign + Mul<Output = T> + Zero + Clone>(
    vec1: &[T],
    vec2: &[T],
) -> T {
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
pub(crate) fn dot_vectors_mt<T: AddAssign + Mul<Output = T> + Zero + Clone + Send + Sync>(
    vec1: &[T],
    vec2: &[T],
) -> T {
    vec1.par_chunks(4096)
        .zip(vec2.par_chunks(4096))
        .map(|(chunk1, chunk2)| dot_vectors(chunk1, chunk2))
        .reduce(|| T::zero(), |a, b| a + b)
}

/// Splits (mutable) work between parallel and sequential processing.
/// Change `PARALLEL_THRESHOLD` to adjust the threshold for parallel processing.
pub(crate) fn for_each_chunk_mut<T, F>(data: &mut [T], func: F, chunk_size: usize)
where
    T: Send + Sync,
    F: Fn(usize, &mut [T]) + Send + Sync,
{
    if chunk_size >= PARALLEL_THRESHOLD && data.len() >= PARALLEL_THRESHOLD {
        data.par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(i, chunk)| func(i, chunk));
    } else if data.len() >= PARALLEL_THRESHOLD {
        data.chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(i, chunk)| func(i, chunk));
    } else {
        func(0, data);
    }
}

/// Like for_each_chunk_mut, but for two mutable slices of the same length.
/// The function `func` is called with the index of the chunk and two mutable slices of the same length.
pub(crate) fn for_each_chunk_mut_pair<T, F>(
    first: &mut [T],
    second: &mut [T],
    func: F,
    chunk_size: usize,
) where
    T: Send + Sync,
    F: Fn(usize, &mut [T], &mut [T]) + Send + Sync,
{
    assert_eq!(first.len(), second.len());
    if chunk_size >= PARALLEL_THRESHOLD && first.len() >= PARALLEL_THRESHOLD {
        first
            .par_chunks_mut(chunk_size)
            .zip(second.par_chunks_mut(chunk_size))
            .enumerate()
            .for_each(|(i, (first, second))| func(i, first, second));
    } else if first.len() >= PARALLEL_THRESHOLD {
        first
            .chunks_mut(chunk_size)
            .zip(second.chunks_mut(chunk_size))
            .enumerate()
            .for_each(|(i, (first, second))| func(i, first, second));
    } else {
        func(0, first, second);
    }
}

#[cfg(test)]
mod internal_functions_tests {
    use crate::utilities::internal_functions::PARALLEL_THRESHOLD;
    use float_cmp::approx_eq;
    use num::complex::Complex64;
    use tensor_math::utilities::internal_functions::{
        dot_vectors, dot_vectors_mt, for_each_chunk_mut, for_each_chunk_mut_pair,
    };

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
    fn split_parallel_work_larger_chunk_size_than_threshold() {
        let f = |i: usize, chunk: &mut [i32]| {
            assert_eq!(chunk.len(), PARALLEL_THRESHOLD << 1);
            for x in chunk.iter_mut() {
                *x += i as i32;
            }
        };
        let mut data = vec![0; PARALLEL_THRESHOLD << 2];
        for_each_chunk_mut(&mut data, f, PARALLEL_THRESHOLD << 1);
        assert_eq!(
            data,
            (0..(PARALLEL_THRESHOLD << 2))
                .map(|i| (i / (PARALLEL_THRESHOLD << 1)) as i32)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn split_parallel_work_smaller_chunk_size_than_threshold() {
        let f = |i: usize, chunk: &mut [i32]| {
            assert_eq!(chunk.len(), PARALLEL_THRESHOLD >> 1);
            for x in chunk.iter_mut() {
                *x += i as i32;
            }
        };
        let mut data = vec![0; PARALLEL_THRESHOLD << 2];
        for_each_chunk_mut(&mut data, f, PARALLEL_THRESHOLD >> 1);
        assert_eq!(
            data,
            (0..(PARALLEL_THRESHOLD << 2))
                .map(|i| (i / (PARALLEL_THRESHOLD >> 1)) as i32)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn split_parallel_work_across_paired_chunks() {
        let mut first = vec![1; PARALLEL_THRESHOLD << 2];
        let mut second = vec![0; PARALLEL_THRESHOLD << 2];
        for_each_chunk_mut_pair(
            &mut first,
            &mut second,
            |i, first, second| {
                for (a, b) in first.iter().zip(second.iter_mut()) {
                    *b = *a + i as i32;
                }
            },
            PARALLEL_THRESHOLD,
        );
        assert_eq!(
            second,
            (0..(PARALLEL_THRESHOLD << 2))
                .map(|i| 1 + (i / PARALLEL_THRESHOLD) as i32)
                .collect::<Vec<_>>()
        );
    }
}

#[cfg(test)]
mod noncommutative_dot_tests {
    use super::{dot_vectors, dot_vectors_mt};
    use num::traits::Zero;
    use std::ops::{Add, AddAssign, Mul};

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
