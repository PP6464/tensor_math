//! Tests for multiplication-based operations against a non-commutative element type.
//!
//! The single-threaded and multi-threaded implementations of the multiplication
//! ops are otherwise tested with primitive numeric types where `a * b == b * a`,
//! so they cannot catch order-of-operations bugs. The `NC` type below makes
//! multiplication depend on the operand order (`x * y != y * x` in general),
//! so the tests can verify that each implementation actually multiplies the
//! operands in the documented order.

#[cfg(test)]
mod noncommutative_mul_tests {
    use std::ops::{Add, Mul};

    use num::Zero;

    use tensor_math::definitions::matrix::Matrix;
    use tensor_math::definitions::shape::Shape;
    use tensor_math::definitions::tensor::Tensor;
    use tensor_math::shape;

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

    // ---- Kronecker — Tensor -------------------------------------------------

    #[test]
    fn kronecker_tensor_noncommutative() {
        // A: shape [2], B: shape [3]
        // A.kronecker(&B) lays down a 0..B for every element of A in order, where
        // each block is "B scaled by that A element" (i.e. the kronecker
        // implementation does `other * scalar` per element of `self`).
        let a = Tensor::<NC>::new(&shape![2], vec![nc(1), nc(2)]).unwrap();
        let b = Tensor::<NC>::new(&shape![3], vec![nc(3), nc(4), nc(5)]).unwrap();

        // Hand-computed: A.kronecker(&B) = (B * a0) ++ (B * a1)
        //   B * NC(1, 1) = [NC(3, 1*3+1), NC(4, 1*4+1), NC(5, 1*5+1)]
        //                = [NC(3, 4),    NC(4, 5),    NC(5, 6)]
        //   B * NC(2, 1) = [NC(6, 1*3+1), NC(8, 1*4+1), NC(10, 1*5+1)]
        //                = [NC(6, 4),    NC(8, 5),     NC(10, 6)]
        let expected = Tensor::<NC>::new(
            &shape![6],
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
        let a = Tensor::<NC>::new(&shape![2], vec![nc(1), nc(2)]).unwrap();
        let b = Tensor::<NC>::new(&shape![3], vec![nc(3), nc(4), nc(5)]).unwrap();

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

    // ---- Kronecker — Matrix --------------------------------------------------

    #[test]
    fn kronecker_matrix_noncommutative() {
        // 2x2 * 3x2 -> 6x4
        let a = Matrix::<NC>::new(2, 2, vec![nc(1), nc(2), nc(3), nc(4)]).unwrap();
        let b = Matrix::<NC>::new(3, 2, vec![nc(5), nc(6), nc(7), nc(8), nc(9), nc(10)]).unwrap();

        // For each element a_ij in row-major order, the output appends the block
        // a_ij * B. Row-major of a: [NC(1,1), NC(2,1), NC(3,1), NC(4,1)].
        // B * NC(k, 1) = B with each element mapped via (b.0*k, b.1*k + 1):
        //   B * NC(1, 1) = [NC(5, 6), NC(6, 7), NC(7, 8), NC(8, 9), NC(9, 10), NC(10, 11)]
        //   B * NC(2, 1) = [NC(10, 6), NC(12, 7), NC(14, 8), NC(16, 9), NC(18, 10), NC(20, 11)]
        //   B * NC(3, 1) = [NC(15, 6), NC(18, 7), NC(21, 8), NC(24, 9), NC(27, 10), NC(30, 11)]
        //   B * NC(4, 1) = [NC(20, 6), NC(24, 7), NC(28, 8), NC(32, 9), NC(36, 10), NC(40, 11)]
        let expected = Matrix::<NC>::new(
            6,
            4,
            vec![
                NC(5, 6),
                NC(6, 7),
                NC(7, 8),
                NC(8, 9),
                NC(9, 10),
                NC(10, 11),
                NC(10, 6),
                NC(12, 7),
                NC(14, 8),
                NC(16, 9),
                NC(18, 10),
                NC(20, 11),
                NC(15, 6),
                NC(18, 7),
                NC(21, 8),
                NC(24, 9),
                NC(27, 10),
                NC(30, 11),
                NC(20, 6),
                NC(24, 7),
                NC(28, 8),
                NC(32, 9),
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

    // ---- Contract mul — Tensor ----------------------------------------------

    #[test]
    fn contract_mul_tensor_noncommutative() {
        // 2x3 * 3x2 -> 2x2 (matrix-style contraction)
        // A is row-major, B is row-major.
        let a = Tensor::<NC>::new(
            &shape![2, 3],
            vec![nc(1), nc(2), nc(3), nc(4), nc(5), nc(6)],
        )
        .unwrap();
        let b = Tensor::<NC>::new(
            &shape![3, 2],
            vec![nc(7), nc(8), nc(9), nc(10), nc(11), nc(12)],
        )
        .unwrap();

        // Hand-computed via NC(a,1) * NC(c,1) = NC(a*c, a*c + 1):
        //   A*B[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
        //            = NC(7, 8) + NC(18, 10) + NC(33, 12) = NC(58, 30)
        //   A*B[0,1] = NC(8, 9) + NC(20, 11) + NC(36, 13) = NC(64, 33)
        //   A*B[1,0] = NC(28, 8) + NC(45, 10) + NC(66, 12) = NC(139, 30)
        //   A*B[1,1] = NC(32, 9) + NC(50, 11) + NC(72, 13) = NC(154, 33)
        let expected = Tensor::<NC>::new(
            &shape![2, 2],
            vec![NC(58, 30), NC(64, 33), NC(139, 30), NC(154, 33)],
        )
        .unwrap();

        let got = a.contract_mul(&b).unwrap();
        assert_eq!(got, expected);

        // Sanity: A.contract_mul(&B) and B.contract_mul(&A) must differ.
        let a_b = a.contract_mul(&b).unwrap();
        let b_a = b.contract_mul(&a).unwrap();
        assert_ne!(a_b, b_a);
    }

    #[test]
    fn contract_mul_tensor_noncommutative_mt() {
        let a = Tensor::<NC>::new(
            &shape![2, 3],
            vec![nc(1), nc(2), nc(3), nc(4), nc(5), nc(6)],
        )
        .unwrap();
        let b = Tensor::<NC>::new(
            &shape![3, 2],
            vec![nc(7), nc(8), nc(9), nc(10), nc(11), nc(12)],
        )
        .unwrap();

        let a_b_single = a.contract_mul(&b).unwrap();
        let b_a_single = b.contract_mul(&a).unwrap();

        let a_b_mt = a.contract_mul_mt(&b).unwrap();
        let b_a_mt = b.contract_mul_mt(&a).unwrap();

        assert_eq!(a_b_mt, a_b_single);
        assert_eq!(b_a_mt, b_a_single);
        assert_ne!(a_b_mt, b_a_mt);
    }

    // ---- Contract mul / mat mul — Matrix ------------------------------------

    #[test]
    fn contract_mul_matrix_noncommutative() {
        let a = Matrix::<NC>::new(2, 3, vec![nc(1), nc(2), nc(3), nc(4), nc(5), nc(6)]).unwrap();
        let b = Matrix::<NC>::new(3, 2, vec![nc(7), nc(8), nc(9), nc(10), nc(11), nc(12)]).unwrap();

        let expected =
            Matrix::<NC>::new(2, 2, vec![NC(58, 30), NC(64, 33), NC(139, 30), NC(154, 33)])
                .unwrap();

        let got = a.contract_mul(&b).unwrap();
        assert_eq!(got, expected);

        let a_b = a.contract_mul(&b).unwrap();
        let b_a = b.contract_mul(&a).unwrap();
        assert_ne!(a_b, b_a);
    }

    #[test]
    fn contract_mul_matrix_noncommutative_mt() {
        let a = Matrix::<NC>::new(2, 3, vec![nc(1), nc(2), nc(3), nc(4), nc(5), nc(6)]).unwrap();
        let b = Matrix::<NC>::new(3, 2, vec![nc(7), nc(8), nc(9), nc(10), nc(11), nc(12)]).unwrap();

        let a_b_single = a.contract_mul(&b).unwrap();
        let b_a_single = b.contract_mul(&a).unwrap();

        let a_b_mt = a.contract_mul_mt(&b).unwrap();
        let b_a_mt = b.contract_mul_mt(&a).unwrap();

        assert_eq!(a_b_mt, a_b_single);
        assert_eq!(b_a_mt, b_a_single);
        assert_ne!(a_b_mt, b_a_mt);
    }

    #[test]
    fn mat_mul_alias_uses_contract_mul() {
        // mat_mul / mat_mul_mt are thin aliases of contract_mul. Verify that
        // they route through the same code for the non-commutative type so a
        // future refactor that diverges them would be caught.
        let a = Matrix::<NC>::new(2, 3, vec![nc(1), nc(2), nc(3), nc(4), nc(5), nc(6)]).unwrap();
        let b = Matrix::<NC>::new(3, 2, vec![nc(7), nc(8), nc(9), nc(10), nc(11), nc(12)]).unwrap();

        assert_eq!(a.mat_mul(&b).unwrap(), a.contract_mul(&b).unwrap());
        assert_eq!(a.mat_mul_mt(&b).unwrap(), a.contract_mul_mt(&b).unwrap());
    }

    // ---- Elementwise * (single-threaded) ------------------------------------

    #[test]
    fn elementwise_mul_tensor_noncommutative() {
        let t1 = Tensor::<NC>::new(&shape![2], vec![nc(1), nc(2)]).unwrap();
        let t2 = Tensor::<NC>::new(&shape![2], vec![nc(3), nc(4)]).unwrap();

        // NC(1,1)*NC(3,1)=NC(3, 1*3+1)=NC(3,4); NC(2,1)*NC(4,1)=NC(8, 1*4+1)=NC(8,5)
        let expected = Tensor::<NC>::new(&shape![2], vec![NC(3, 4), NC(8, 5)]).unwrap();

        assert_eq!((&t1 * &t2), expected);
        assert_ne!((&t1 * &t2), (&t2 * &t1));
    }

    #[test]
    fn elementwise_mul_matrix_noncommutative() {
        let m1 = Matrix::<NC>::new(2, 2, vec![nc(1), nc(2), nc(3), nc(4)]).unwrap();
        let m2 = Matrix::<NC>::new(2, 2, vec![nc(5), nc(6), nc(7), nc(8)]).unwrap();

        // Row-major products: NC(1,1)*NC(5,1)=NC(5,6), NC(2,1)*NC(6,1)=NC(12,7),
        //                     NC(3,1)*NC(7,1)=NC(21,8), NC(4,1)*NC(8,1)=NC(32,9)
        let expected =
            Matrix::<NC>::new(2, 2, vec![NC(5, 6), NC(12, 7), NC(21, 8), NC(32, 9)]).unwrap();

        assert_eq!((&m1 * &m2), expected);
        assert_ne!((&m1 * &m2), (&m2 * &m1));
    }

    // ---- Dot product (single-threaded) --------------------------------------

    #[test]
    fn dot_tensor_noncommutative() {
        // a.dot(b) = sum_k a[k] * b[k]  (uses elementwise Mul + sum)
        let a = Tensor::<NC>::new(&shape![2], vec![nc(1), nc(2)]).unwrap();
        let b = Tensor::<NC>::new(&shape![2], vec![nc(3), nc(4)]).unwrap();

        // NC(1,1)*NC(3,1) = NC(3, 4); NC(2,1)*NC(4,1) = NC(8, 5); sum = NC(11, 9)
        let got = a.dot(&b).unwrap();
        assert_eq!(got, NC(11, 9));

        // b.dot(a) = (b * a).sum():
        //   b * a (elementwise, b on the left):
        //     NC(3,1) * NC(1,1) = NC(3, 2);  NC(4,1) * NC(2,1) = NC(8, 3)
        //   sum = NC(11, 5)
        let got_swapped = b.dot(&a).unwrap();
        assert_eq!(got_swapped, NC(11, 5));

        assert_ne!(got, got_swapped);
    }

    #[test]
    fn dot_matrix_noncommutative() {
        let m1 = Matrix::<NC>::new(2, 2, vec![nc(1), nc(2), nc(3), nc(4)]).unwrap();
        let m2 = Matrix::<NC>::new(2, 2, vec![nc(5), nc(6), nc(7), nc(8)]).unwrap();

        // m1.dot(m2) = sum_k m1[k] * m2[k] over the row-major element list:
        //   NC(1,1)*NC(5,1) = NC(5, 6)
        //   NC(2,1)*NC(6,1) = NC(12, 7)
        //   NC(3,1)*NC(7,1) = NC(21, 8)
        //   NC(4,1)*NC(8,1) = NC(32, 9)
        // sum = NC(70, 30)
        let got = m1.dot(&m2).unwrap();
        assert_eq!(got, NC(70, 30));

        // m2.dot(m1) = NC(5, 2) + NC(12, 3) + NC(21, 4) + NC(32, 5) = NC(70, 14)
        let got_swapped = m2.dot(&m1).unwrap();
        assert_eq!(got_swapped, NC(70, 14));

        assert_ne!(got, got_swapped);
    }

    // ---- Dot product (multi-threaded) --------------------------------------
    #[test]
    fn dot_tensor_noncommutative_mt() {
        // a.dot(b) = sum_k a[k] * b[k]  (uses elementwise Mul + sum)
        let a = Tensor::<NC>::new(&shape![2], vec![nc(1), nc(2)]).unwrap();
        let b = Tensor::<NC>::new(&shape![2], vec![nc(3), nc(4)]).unwrap();

        // NC(1,1)*NC(3,1) = NC(3, 4); NC(2,1)*NC(4,1) = NC(8, 5); sum = NC(11, 9)
        let got = a.dot_mt(&b).unwrap();
        assert_eq!(got, NC(11, 9));

        // b.dot(a) = (b * a).sum():
        //   b * a (elementwise, b on the left):
        //     NC(3,1) * NC(1,1) = NC(3, 2);  NC(4,1) * NC(2,1) = NC(8, 3)
        //   sum = NC(11, 5)
        let got_swapped = b.dot_mt(&a).unwrap();
        assert_eq!(got_swapped, NC(11, 5));

        assert_ne!(got, got_swapped);
    }

    #[test]
    fn dot_matrix_noncommutative_mt() {
        let m1 = Matrix::<NC>::new(2, 2, vec![nc(1), nc(2), nc(3), nc(4)]).unwrap();
        let m2 = Matrix::<NC>::new(2, 2, vec![nc(5), nc(6), nc(7), nc(8)]).unwrap();

        // m1.dot(m2) = sum_k m1[k] * m2[k] over the row-major element list:
        //   NC(1,1)*NC(5,1) = NC(5, 6)
        //   NC(2,1)*NC(6,1) = NC(12, 7)
        //   NC(3,1)*NC(7,1) = NC(21, 8)
        //   NC(4,1)*NC(8,1) = NC(32, 9)
        // sum = NC(70, 30)
        let got = m1.dot_mt(&m2).unwrap();
        assert_eq!(got, NC(70, 30));

        // m2.dot(m1) = NC(5, 2) + NC(12, 3) + NC(21, 4) + NC(32, 5) = NC(70, 14)
        let got_swapped = m2.dot_mt(&m1).unwrap();
        assert_eq!(got_swapped, NC(70, 14));

        assert_ne!(got, got_swapped);
    }
}
