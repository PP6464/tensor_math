use crate::definitions::matrix::Matrix;
use crate::definitions::shape::Shape;
use crate::definitions::strides::Strides;
use crate::definitions::tensor::Tensor;
use crate::utilities::internal_functions::dot_vectors;
use rayon::iter::ParallelIterator;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator};
use rayon::prelude::ParallelSliceMut;
use std::mem::MaybeUninit;
use std::ops::Mul;

impl<T: Clone + Mul<Output = T>> Tensor<T> {
    /// Computes the Kronecker product between the tensors.
    /// For tensors of differing ranks it pads the lower rank shape
    /// backwards with ones so that the ranks match.
    pub fn kronecker(&self, other: &Tensor<T>) -> Tensor<T> {
        let max_rank = self.rank().max(other.rank());
        let self_shape_padded = self
            .shape
            .0
            .iter()
            .copied()
            .chain((self.rank()..max_rank).map(|_| 1usize))
            .collect::<Shape>();
        let other_shape_padded = other
            .shape
            .0
            .iter()
            .copied()
            .chain((other.rank()..max_rank).map(|_| 1usize))
            .collect::<Shape>();
        let res_shape = self_shape_padded
            .0
            .iter()
            .zip(other_shape_padded.0.iter())
            .map(|(&x, &y)| x * y)
            .collect::<Shape>();
        let res_strides = Strides::from_shape(&res_shape);
        let mut elements = Vec::with_capacity(res_shape.element_count());
        let buf = elements.spare_capacity_mut();
        let buf_ptr = buf.as_mut_ptr();

        unsafe {
            for self_index in self_shape_padded.indices() {
                let self_element = self.get_unchecked(self_index.get_unchecked(..self.rank()));
                let base_offset = self_index
                    .iter()
                    .zip(other_shape_padded.0.iter())
                    .map(|(&x, &y)| x * y)
                    .collect::<Vec<_>>();

                for other_index in other_shape_padded.indices() {
                    let other_element =
                        other.get_unchecked(other_index.get_unchecked(..other.rank()));
                    let offset = base_offset
                        .iter()
                        .zip(other_index.iter())
                        .map(|(&x, &y)| x + y)
                        .collect::<Vec<_>>();

                    buf_ptr
                        .add(dot_vectors(&offset, &res_strides.0))
                        .as_mut_unchecked()
                        .write(self_element.clone() * other_element.clone());
                }
            }

            elements.set_len(res_shape.element_count());
        }

        Tensor {
            strides: res_strides,
            shape: res_shape,
            elements,
        }
    }

    /// Computes the Kronecker product between the tensors.
    /// For tensors of differing ranks it pads the lower rank shape
    /// backwards with ones so that the ranks match.
    pub fn kronecker_mt(&self, other: &Tensor<T>) -> Tensor<T>
    where
        T: Send + Sync,
    {
        struct ThreadSafePtr<O>(*mut O);
        unsafe impl<O> Send for ThreadSafePtr<O> {}
        unsafe impl<O> Sync for ThreadSafePtr<O> {}

        impl<O> ThreadSafePtr<O> {
            unsafe fn add(&self, offset: usize) -> ThreadSafePtr<O> {
                ThreadSafePtr(self.0.add(offset))
            }

            unsafe fn write(&mut self, val: O) {
                self.0.write(val)
            }
        }

        let max_rank = self.rank().max(other.rank());
        let self_shape_padded = self
            .shape
            .0
            .iter()
            .copied()
            .chain((self.rank()..max_rank).map(|_| 1usize))
            .collect::<Shape>();
        let other_shape_padded = other
            .shape
            .0
            .iter()
            .copied()
            .chain((other.rank()..max_rank).map(|_| 1usize))
            .collect::<Shape>();
        let res_shape = self_shape_padded
            .0
            .iter()
            .zip(other_shape_padded.0.iter())
            .map(|(&x, &y)| x * y)
            .collect::<Shape>();
        let res_strides = Strides::from_shape(&res_shape);
        let mut elements = Vec::with_capacity(res_shape.element_count());
        let buf = elements.spare_capacity_mut();
        let buf_ptr = ThreadSafePtr(buf.as_mut_ptr());

        unsafe {
            self_shape_padded
                .indices()
                .par_iter()
                .for_each(|self_index| {
                    let self_element = self.get_unchecked(self_index.get_unchecked(..self.rank()));
                    let base_offset = self_index
                        .iter()
                        .zip(other_shape_padded.0.iter())
                        .map(|(&x, &y)| x * y)
                        .collect::<Vec<_>>();

                    for other_index in other_shape_padded.indices() {
                        let other_element =
                            other.get_unchecked(other_index.get_unchecked(..other.rank()));
                        let offset = base_offset
                            .iter()
                            .zip(other_index.iter())
                            .map(|(&x, &y)| x + y)
                            .collect::<Vec<_>>();

                        buf_ptr
                            .add(dot_vectors(&offset, &res_strides.0))
                            .write(MaybeUninit::new(
                                self_element.clone() * other_element.clone(),
                            ));
                    }
                });
        }

        unsafe {
            elements.set_len(res_shape.element_count());
        }

        Tensor {
            strides: res_strides,
            shape: res_shape,
            elements,
        }
    }
}

impl<T: Clone + Mul<Output = T>> Matrix<T> {
    /// Computes the Kronecker product of the matrices.
    pub fn kronecker(&self, other: &Matrix<T>) -> Matrix<T> {
        let rows = self.rows * other.rows;
        let cols = self.cols * other.cols;
        let mut elements = Vec::with_capacity(rows * cols);

        if rows == 0 || cols == 0 {
            return Matrix {
                rows,
                cols,
                elements,
            };
        }

        let output = elements.spare_capacity_mut();
        for (output_row, row) in output[..rows * cols].chunks_mut(cols).enumerate() {
            let self_row = output_row / other.rows;
            let other_row = output_row % other.rows;
            let self_values = &self.elements[self_row * self.cols..(self_row + 1) * self.cols];
            let other_values =
                &other.elements[other_row * other.cols..(other_row + 1) * other.cols];

            for (self_col, self_value) in self_values.iter().enumerate() {
                for (other_col, other_value) in other_values.iter().enumerate() {
                    row[self_col * other.cols + other_col]
                        .write(self_value.clone() * other_value.clone());
                }
            }
        }

        unsafe {
            elements.set_len(rows * cols);
        }

        Matrix {
            rows,
            cols,
            elements,
        }
    }

    /// Computes the Kronecker product of the matrices.
    pub fn kronecker_mt(&self, other: &Matrix<T>) -> Matrix<T>
    where
        T: Send + Sync,
    {
        let rows = self.rows * other.rows;
        let cols = self.cols * other.cols;
        let mut elements = Vec::with_capacity(rows * cols);
        if rows == 0 || cols == 0 {
            return Matrix {
                rows,
                cols,
                elements,
            };
        }

        let output = elements.spare_capacity_mut();
        output[..rows * cols]
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(output_row, row)| {
                let self_row = output_row / other.rows;
                let other_row = output_row % other.rows;
                let self_values = &self.elements[self_row * self.cols..(self_row + 1) * self.cols];
                let other_values =
                    &other.elements[other_row * other.cols..(other_row + 1) * other.cols];

                for (self_col, self_value) in self_values.iter().enumerate() {
                    for (other_col, other_value) in other_values.iter().enumerate() {
                        row[self_col * other.cols + other_col]
                            .write(self_value.clone() * other_value.clone());
                    }
                }
            });

        unsafe {
            elements.set_len(rows * cols);
        }

        Matrix {
            rows,
            cols,
            elements,
        }
    }
}

#[cfg(test)]
mod kronecker_tests {
    use tensor_math::definitions::matrix::Matrix;
    use tensor_math::definitions::tensor::Tensor;
    use tensor_math::definitions::traits::IntoTensor;
    use tensor_math::definitions::traits::TryIntoMatrix;
    use tensor_math::shape;

    #[test]
    fn kronecker_product() {
        let t1 = Tensor::<i32>::new(shape![2, 2], vec![1, 2, 3, 4]).unwrap();
        let t2 = Tensor::<i32>::new(shape![2, 2], vec![5, 6, 7, 8]).unwrap();
        let expected = Tensor::<i32>::new(
            shape![4, 4],
            vec![5, 6, 10, 12, 7, 8, 14, 16, 15, 18, 20, 24, 21, 24, 28, 32],
        )
        .unwrap();
        assert_eq!(t1.kronecker(&t2), expected);
    }

    #[test]
    fn kronecker_mt() {
        let t1 = Tensor::<i32>::new(shape![5, 2, 2], (0..20).collect()).unwrap();
        let t2 = Tensor::<i32>::new(shape![2, 3], (0..6).collect()).unwrap();

        let ans = t1.kronecker(&t2);
        let mt_ans = t1.kronecker_mt(&t2);

        assert_eq!(mt_ans, ans);

        let ans = t2.kronecker(&t1);
        let mt_ans = t2.kronecker_mt(&t1);

        assert_eq!(mt_ans, ans);
    }

    #[test]
    fn test_mat_kronecker_product() {
        let m1 = Matrix::<i32>::new(2, 2, vec![1, 2, 3, 4]).unwrap();
        let m2 = Matrix::<i32>::new(2, 2, vec![5, 6, 7, 8]).unwrap();
        let expected = Matrix::<i32>::new(
            4,
            4,
            vec![5, 6, 10, 12, 7, 8, 14, 16, 15, 18, 20, 24, 21, 24, 28, 32],
        )
        .unwrap();

        assert_eq!(m1.kronecker(&m2), expected);
        assert_eq!(
            m1.kronecker(&m2),
            m1.into_tensor()
                .kronecker(&m2.into_tensor())
                .try_into_matrix()
                .unwrap()
        );
    }

    #[test]
    fn kronecker_mt_mat() {
        let m1 = Matrix::<i32>::new(2, 2, vec![1, 2, 3, 4]).unwrap();
        let m2 = Matrix::<i32>::new(2, 2, vec![5, 6, 7, 8]).unwrap();
        let expected = Matrix::<i32>::new(
            4,
            4,
            vec![5, 6, 10, 12, 7, 8, 14, 16, 15, 18, 20, 24, 21, 24, 28, 32],
        )
        .unwrap();

        assert_eq!(m1.kronecker_mt(&m2), expected);
    }

    #[test]
    fn kronecker_product_with_scalar_tensor() {
        let t = Tensor::<i32>::new(shape![2, 2], vec![1, 2, 3, 4]).unwrap();
        let s = Tensor::<i32>::new(shape![], vec![10]).unwrap();

        let ans_a_s = Tensor::<i32>::new(shape![2, 2], vec![10, 20, 30, 40]).unwrap();
        assert_eq!(t.kronecker(&s), ans_a_s);
        assert_eq!(t.kronecker_mt(&s), ans_a_s);

        let ans_s_a = Tensor::<i32>::new(shape![2, 2], vec![10, 20, 30, 40]).unwrap();
        assert_eq!(s.kronecker(&t), ans_s_a);
        assert_eq!(s.kronecker_mt(&t), ans_s_a);

        let scalar = Tensor::<i32>::new(shape![], vec![3]).unwrap();
        assert_eq!(
            scalar.kronecker(&Tensor::new(shape![], vec![4]).unwrap()),
            Tensor::new(shape![], vec![12]).unwrap()
        );
    }

    #[test]
    fn kronecker_product_pads_lower_rank_shape_on_the_right() {
        let vector = Tensor::<i32>::new(shape![2], vec![2, 3]).unwrap();
        let matrix = Tensor::<i32>::new(shape![2, 2], vec![1, 4, 5, 6]).unwrap();
        let expected = Tensor::<i32>::new(shape![4, 2], vec![2, 8, 10, 12, 3, 12, 15, 18]).unwrap();

        assert_eq!(vector.kronecker(&matrix), expected);
        assert_eq!(vector.kronecker_mt(&matrix), expected);
    }

    #[test]
    fn kronecker_product_with_empty_matrices_and_tensors() {
        let empty_t = Tensor::<i32>::new(shape![0, 2], vec![]).unwrap();
        let full_t = Tensor::<i32>::new(shape![2, 2], vec![1, 2, 3, 4]).unwrap();

        // [0, 2] x [2, 2] -> [0, 4]
        let res1 = empty_t.kronecker(&full_t);
        assert_eq!(res1.shape(), shape![0, 4]);
        assert_eq!(res1.len(), 0);
        assert_eq!(empty_t.kronecker_mt(&full_t), res1);

        // [2, 2] x [0, 2] -> [0, 4]
        let res2 = full_t.kronecker(&empty_t);
        assert_eq!(res2.shape(), shape![0, 4]);
        assert_eq!(res2.len(), 0);
        assert_eq!(full_t.kronecker_mt(&empty_t), res2);

        let empty_m = Matrix::<i32>::new(0, 5, vec![]).unwrap();
        let full_m = Matrix::<i32>::new(2, 2, vec![1, 2, 3, 4]).unwrap();

        // (0x5) x (2x2) -> (0x10)
        let res_m1 = empty_m.kronecker(&full_m);
        assert_eq!(res_m1.rows(), 0);
        assert_eq!(res_m1.cols(), 10);
        assert_eq!(res_m1.len(), 0);
        assert_eq!(empty_m.kronecker_mt(&full_m), res_m1);

        // (2x2) x (0x5) -> (0x10)
        let res_m2 = full_m.kronecker(&empty_m);
        assert_eq!(res_m2.rows(), 0);
        assert_eq!(res_m2.cols(), 10);
        assert_eq!(res_m2.len(), 0);
        assert_eq!(full_m.kronecker_mt(&empty_m), res_m2);

        for (lhs, rhs, expected_rows, expected_cols) in [
            (
                Matrix::<i32>::new(2, 0, vec![]).unwrap(),
                Matrix::<i32>::new(3, 2, vec![1; 6]).unwrap(),
                6,
                0,
            ),
            (
                Matrix::<i32>::new(0, 0, vec![]).unwrap(),
                Matrix::<i32>::new(2, 3, vec![1; 6]).unwrap(),
                0,
                0,
            ),
        ] {
            let expected = Matrix::<i32>::new(expected_rows, expected_cols, vec![]).unwrap();
            assert_eq!(lhs.kronecker(&rhs), expected);
            assert_eq!(lhs.kronecker_mt(&rhs), expected);
        }

        for (lhs, rhs) in [
            (
                Tensor::<i32>::new(shape![0, 2], vec![]).unwrap(),
                Tensor::<i32>::new(shape![3], vec![1, 2, 3]).unwrap(),
            ),
            (
                Tensor::<i32>::new(shape![2], vec![1, 2]).unwrap(),
                Tensor::<i32>::new(shape![0, 3], vec![]).unwrap(),
            ),
        ] {
            let serial = lhs.kronecker(&rhs);
            assert_eq!(lhs.kronecker_mt(&rhs), serial);
            assert_eq!(serial.len(), 0);
        }
    }
}

#[cfg(test)]
mod noncommutative_tests {
    use crate::definitions::matrix::Matrix;
    use crate::definitions::tensor::Tensor;
    use crate::shape;
    use num::Zero;
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
    fn kronecker_matrix_noncommutative() {
        // 2x2 * 3x2 -> 6x4
        let a = Matrix::<NC>::new(2, 2, vec![nc(1), nc(2), nc(3), nc(4)]).unwrap();
        let b = Matrix::<NC>::new(3, 2, vec![nc(5), nc(6), nc(7), nc(8), nc(9), nc(10)]).unwrap();

        // Each output block is a_ij * B. Rows are ordered by the row of A,
        // then the row of B; columns are ordered by the column of A, then B.
        let expected = Matrix::<NC>::new(
            6,
            4,
            vec![
                NC(5, 6),
                NC(6, 7),
                NC(10, 6),
                NC(12, 7),
                NC(7, 8),
                NC(8, 9),
                NC(14, 8),
                NC(16, 9),
                NC(9, 10),
                NC(10, 11),
                NC(18, 10),
                NC(20, 11),
                NC(15, 6),
                NC(18, 7),
                NC(20, 6),
                NC(24, 7),
                NC(21, 8),
                NC(24, 9),
                NC(28, 8),
                NC(32, 9),
                NC(27, 10),
                NC(30, 11),
                NC(36, 10),
                NC(40, 11),
            ],
        )
        .unwrap();

        let got = a.kronecker(&b);
        assert_eq!(got, expected);

        let a_kron_b = a.kronecker(&b);
        let b_kron_a = b.kronecker(&a);
        assert_ne!(a_kron_b, b_kron_a);
    }

    #[test]
    fn kronecker_matrix_noncommutative_mt() {
        let a = Matrix::<NC>::new(2, 2, vec![nc(1), nc(2), nc(3), nc(4)]).unwrap();
        let b = Matrix::<NC>::new(3, 2, vec![nc(5), nc(6), nc(7), nc(8), nc(9), nc(10)]).unwrap();

        let expected_ab = a.kronecker(&b);
        let expected_ba = b.kronecker(&a);

        let got_ab = a.kronecker_mt(&b);
        let got_ba = b.kronecker_mt(&a);

        assert_eq!(got_ab, expected_ab);
        assert_eq!(got_ba, expected_ba);
        assert_ne!(got_ab, got_ba);
    }

    #[test]
    fn kronecker_tensor_noncommutative() {
        // A: shape [2], B: shape [3]
        // The output preserves self * other operand order for every paired element.
        let a = Tensor::<NC>::new(shape![2], vec![nc(1), nc(2)]).unwrap();
        let b = Tensor::<NC>::new(shape![3], vec![nc(3), nc(4), nc(5)]).unwrap();

        // Hand-computed: A.kronecker(&B) = (B * a0) ++ (B * a1)
        //   B * NC(1, 1) = [NC(3, 1*3+1), NC(4, 1*4+1), NC(5, 1*5+1)]
        //                = [NC(3, 4),    NC(4, 5),    NC(5, 6)]
        //   B * NC(2, 1) = [NC(6, 1*3+1), NC(8, 1*4+1), NC(10, 1*5+1)]
        //                = [NC(6, 4),    NC(8, 5),     NC(10, 6)]
        let expected = Tensor::<NC>::new(
            shape![6],
            vec![NC(3, 4), NC(4, 5), NC(5, 6), NC(6, 4), NC(8, 5), NC(10, 6)],
        )
        .unwrap();

        let got = a.kronecker(&b);
        assert_eq!(got, expected);

        // Sanity: A.kronecker(&B) and B.kronecker(&A) must differ when Mul is
        // non-commutative — otherwise the implementation is silently swapping
        // the operands somewhere.
        let a_kron_b = a.kronecker(&b);
        let b_kron_a = b.kronecker(&a);
        assert_ne!(a_kron_b, b_kron_a);
    }

    #[test]
    fn kronecker_tensor_noncommutative_mt() {
        let a = Tensor::<NC>::new(shape![2], vec![nc(1), nc(2)]).unwrap();
        let b = Tensor::<NC>::new(shape![3], vec![nc(3), nc(4), nc(5)]).unwrap();

        let expected_ab = a.kronecker(&b);
        let expected_ba = b.kronecker(&a);

        let got_ab = a.kronecker_mt(&b);
        let got_ba = b.kronecker_mt(&a);

        // The multithreaded variant must match the single-threaded result for
        // both operand orderings, even though the per-element work is split
        // across independent tasks.
        assert_eq!(got_ab, expected_ab);
        assert_eq!(got_ba, expected_ba);
        assert_ne!(got_ab, got_ba);
    }
}
