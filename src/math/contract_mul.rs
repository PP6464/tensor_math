use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::definitions::shape::Shape;
use crate::definitions::tensor::Tensor;
use crate::definitions::traits::IntoTensor;
use crate::shape;
use crate::utilities::internal_functions::dot_vectors;
use num::Zero;
use rayon::iter::ParallelIterator;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator};
use rayon::slice::ParallelSliceMut;
use std::ops::{AddAssign, Mul};

/*
--------------------------------------------
* Tensor contract multiplication
--------------------------------------------
*/

impl<T> Tensor<T> {
    /// Perform tensor-contraction multiplication,
    /// which is a more general form of matrix multiplication.
    /// E.g: A tensor of shape (a,b,c) multiplied in this way by a tensor of shape (c, d, e, f)
    /// will produce a tensor of shape (a, b, d, e, f) by the following formula:
    /// result[&[i, j, k, l, m\]\] = sum(x=0, x=c) { first[&[i, j, x\]\] * second[&[x, k, l, m\]\] }.
    /// This fails if the tensors have shapes that are not compatible.
    pub fn contract_mul(self, other: Tensor<T>) -> Result<Tensor<T>, TensorErrors>
    where
        T: AddAssign + Mul<Output = T> + Clone + Zero,
    {
        let self_shape = self.shape().0;
        let self_rank = self.rank();
        let other_shape = other.shape().0;
        let other_rank = other.rank();

        if self_rank > 0 && other_rank > 0 {
            let self_matrix_shape = shape![
                self_shape[..self_rank - 1].iter().product(),
                self_shape[self_rank - 1]
            ];
            let other_matrix_shape = shape![other_shape[0], other_shape[1..].iter().product()];

            if self_matrix_shape[1] != other_matrix_shape[0] {
                return Err(TensorErrors::IncompatibleShapes {
                    shape_1: self_matrix_shape,
                    shape_2: other_matrix_shape,
                    op: "mat_mul",
                });
            }
        }

        Ok(unsafe { self.contract_mul_unchecked(other) })
    }

    /// Perform tensor-contraction multiplication,
    /// which is a more general form of matrix multiplication.
    /// E.g: A tensor of shape (a,b,c) multiplied in this way by a tensor of shape (c, d, e, f)
    /// will produce a tensor of shape (a, b, d, e, f) by the following formula:
    /// result[&[i, j, k, l, m\]\] = sum(x=0, x=c) { first[&[i, j, x\]\] * second[&[x, k, l, m\]\] }.
    /// This does not check for validity.
    pub(crate) unsafe fn contract_mul_unchecked(mut self, mut other: Tensor<T>) -> Tensor<T>
    where
        T: AddAssign + Mul<Output = T> + Clone + Zero,
    {
        let self_shape = self.shape().0;
        let self_rank = self.rank();
        let other_shape = other.shape().0;
        let other_rank = other.rank();

        if self_rank == 0 {
            let e = self.elements.get_unchecked(0);
            return other.map(|x| e.clone() * x);
        }

        if other_rank == 0 {
            let e = other.elements.get_unchecked(0);
            return self * e;
        }

        self = self.reshape_unchecked(shape![
            self_shape[..self_rank - 1].iter().product(),
            self_shape[self_rank - 1]
        ]);
        let self_mat = Matrix {
            rows: self.shape[0],
            cols: self.shape[1],
            elements: self.elements,
        };
        other = other.reshape_unchecked(shape![other_shape[0], other_shape[1..].iter().product()]);
        let other_mat = Matrix {
            rows: other.shape[0],
            cols: other.shape[1],
            elements: other.elements,
        };

        let result_mat = self_mat.mat_mul_unchecked(other_mat);
        let result_shape = [
            self_shape[..self_rank - 1].to_vec(),
            other_shape[1..].to_vec(),
        ]
        .concat();

        result_mat
            .into_tensor()
            .reshape_unchecked(Shape::new(result_shape))
    }

    /// Perform tensor-contraction multiplication (using multiple threads),
    /// which is a more general form of matrix multiplication.
    /// E.g: A tensor of shape (a,b,c) multiplied in this way by a tensor of shape (c, d, e, f)
    /// will produce a tensor of shape (a, b, d, e, f) by the following formula:
    /// result[&[i, j, k, l, m\]\] = sum(x=0, x=c) { first[&[i, j, x\]\] * second[&[x, k, l, m\]\] }.
    /// This fails if the tensors have shapes that are not compatible.
    pub fn contract_mul_mt(self, other: Tensor<T>) -> Result<Tensor<T>, TensorErrors>
    where
        T: AddAssign + Mul<Output = T> + Clone + Zero + Send + Sync,
    {
        let self_shape = self.shape().0;
        let self_rank = self.rank();
        let other_shape = other.shape().0;
        let other_rank = other.rank();

        if self_rank > 0 && other_rank > 0 {
            let self_matrix_shape = shape![
                self_shape[..self_rank - 1].iter().product(),
                self_shape[self_rank - 1]
            ];
            let other_matrix_shape = shape![other_shape[0], other_shape[1..].iter().product()];

            if self_matrix_shape[1] != other_matrix_shape[0] {
                return Err(TensorErrors::IncompatibleShapes {
                    shape_1: self_matrix_shape,
                    shape_2: other_matrix_shape,
                    op: "mat_mul",
                });
            }
        }

        Ok(unsafe { self.contract_mul_unchecked_mt(other) })
    }

    /// Perform tensor-contraction multiplication,
    /// which is a more general form of matrix multiplication.
    /// E.g: A tensor of shape (a,b,c) multiplied in this way by a tensor of shape (c, d, e, f)
    /// will produce a tensor of shape (a, b, d, e, f) by the following formula:
    /// result[&[i, j, k, l, m\]\] = sum(x=0, x=c) { first[&[i, j, x\]\] * second[&[x, k, l, m\]\] }.
    /// This does not check for validity.
    pub(crate) unsafe fn contract_mul_unchecked_mt(mut self, mut other: Tensor<T>) -> Tensor<T>
    where
        T: AddAssign + Mul<Output = T> + Clone + Zero + Send + Sync,
    {
        let self_shape = self.shape().0;
        let self_rank = self.rank();
        let other_shape = other.shape().0;
        let other_rank = other.rank();

        if self_rank == 0 {
            let e = self.elements.get_unchecked(0);
            return other.map(|x| e.clone() * x);
        }

        if other_rank == 0 {
            let e = other.elements.get_unchecked(0);
            return self * e;
        }

        self = self.reshape_unchecked(shape![
            self_shape[..self_rank - 1].iter().product(),
            self_shape[self_rank - 1]
        ]);
        let self_mat = Matrix {
            rows: self.shape[0],
            cols: self.shape[1],
            elements: self.elements,
        };
        other = other.reshape_unchecked(shape![other_shape[0], other_shape[1..].iter().product()]);
        let other_mat = Matrix {
            rows: other.shape[0],
            cols: other.shape[1],
            elements: other.elements,
        };

        let result_mat = self_mat.mat_mul_unchecked_mt(other_mat);
        let result_shape = [
            self_shape[..self_rank - 1].to_vec(),
            other_shape[1..].to_vec(),
        ]
        .concat();

        result_mat
            .into_tensor()
            .reshape_unchecked(Shape::new(result_shape))
    }
}

/*
--------------------------------------------
* Matrix multiplication
--------------------------------------------
*/

impl<T> Matrix<T> {
    /// Does matrix multiplication with another matrix.
    /// This fails if the matrices are not multiplicatively compatible.
    pub fn mat_mul(self, other: Matrix<T>) -> Result<Matrix<T>, TensorErrors>
    where
        T: AddAssign + Mul<Output = T> + Clone + Zero,
    {
        if self.cols != other.rows {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: other.shape(),
                op: "mat_mul",
            });
        }

        Ok(unsafe { self.mat_mul_unchecked(other) })
    }

    /// Does matrix multiplication with another matrix without validity checking.
    pub(crate) unsafe fn mat_mul_unchecked(self, other: Matrix<T>) -> Matrix<T>
    where
        T: AddAssign + Mul<Output = T> + Clone + Zero,
    {
        let mut elements = Vec::with_capacity(self.rows * other.cols);
        let other_transpose = other.transpose();

        self.iter_rows().for_each(|row| {
            other_transpose.iter_rows().for_each(|col| {
                elements.push(dot_vectors(row, col));
            });
        });

        Matrix {
            rows: self.rows,
            cols: other_transpose.rows,
            elements,
        }
    }

    /// Does matrix multiplication on multiple threads.
    /// This fails if the matrices are not multiplicatively compatible.
    pub fn mat_mul_mt(self, other: Matrix<T>) -> Result<Matrix<T>, TensorErrors>
    where
        T: AddAssign + Mul<Output = T> + Clone + Zero + Send + Sync,
    {
        if self.cols != other.rows {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: other.shape(),
                op: "mat_mul",
            });
        }

        Ok(self.mat_mul_unchecked_mt(other))
    }

    /// Does matrix multiplication on multiple threads without validity checking.
    pub fn mat_mul_unchecked_mt(self, other: Matrix<T>) -> Matrix<T>
    where
        T: AddAssign + Mul<Output = T> + Clone + Zero + Send + Sync,
    {
        let mut elements = Vec::with_capacity(self.rows * other.cols);
        let buf = elements.spare_capacity_mut();
        let other_transpose = other.transpose_mt();

        self.par_iter_rows()
            .zip(buf.par_chunks_mut(other_transpose.rows))
            .for_each(|(row, outs)| {
                other_transpose
                    .par_iter_rows()
                    .zip(outs.par_iter_mut())
                    .for_each(|(col, out)| {
                        out.write(dot_vectors(row, col));
                    });
            });

        unsafe {
            elements.set_len(self.rows * other_transpose.rows);
        }

        Matrix {
            rows: self.rows,
            cols: other_transpose.rows,
            elements,
        }
    }
}

#[cfg(test)]
mod contract_mul_tests {
    use float_cmp::approx_eq;
    use tensor_math::definitions::errors::TensorErrors;
    use tensor_math::definitions::matrix::Matrix;
    use tensor_math::definitions::tensor::Tensor;
    use tensor_math::shape;

    #[test]
    fn contract_mul_with_scalar_tensors() {
        let scalar = Tensor::new(shape![], vec![5]).unwrap();
        let t1 = Tensor::new(shape![2, 2], vec![1, 2, 3, 4]).unwrap();

        // Scalar * Tensor
        let res1 = scalar.clone().contract_mul(t1.clone()).unwrap();
        let ans1 = Tensor::new(shape![2, 2], vec![5, 10, 15, 20]).unwrap();
        assert_eq!(res1, ans1);

        // Tensor * Scalar
        let res2 = t1.clone().contract_mul(scalar.clone()).unwrap();
        assert_eq!(res2, ans1);
    }

    #[test]
    fn contract_mul_mt_with_scalar_tensors() {
        let scalar = Tensor::new(shape![], vec![5]).unwrap();
        let t1 = Tensor::new(shape![2, 2], vec![1, 2, 3, 4]).unwrap();

        // Scalar * Tensor
        let res1 = scalar.clone().contract_mul_mt(t1.clone()).unwrap();
        let ans1 = Tensor::new(shape![2, 2], vec![5, 10, 15, 20]).unwrap();
        assert_eq!(res1, ans1);

        // Tensor * Scalar
        let res2 = t1.clone().contract_mul_mt(scalar.clone()).unwrap();
        assert_eq!(res2, ans1);
    }

    #[test]
    fn contract_mul_with_empty_tensor() {
        let t1 = Tensor::<i32>::new(shape![2, 0], vec![]).unwrap();
        let t2 = Tensor::<i32>::new(shape![0, 3], vec![]).unwrap();

        let res = t1.clone().contract_mul(t2.clone()).unwrap();
        let ans = Tensor::<i32>::new(shape![2, 3], vec![0, 0, 0, 0, 0, 0]).unwrap();
        assert_eq!(res, ans);

        let t3 = Tensor::<i32>::new(shape![0, 2], vec![]).unwrap();
        let t4 = Tensor::<i32>::new(shape![2, 2], vec![1, 2, 3, 4]).unwrap();
        let res2 = t3.clone().contract_mul(t4.clone()).unwrap();
        assert_eq!(res2.shape(), shape![0, 2]);
        assert_eq!(res2.len(), 0);
    }

    #[test]
    fn contract_mul_mt_with_empty_tensor() {
        let t1 = Tensor::<i32>::new(shape![2, 0], vec![]).unwrap();
        let t2 = Tensor::<i32>::new(shape![0, 3], vec![]).unwrap();

        let res = t1.clone().contract_mul_mt(t2.clone()).unwrap();
        let ans = Tensor::<i32>::new(shape![2, 3], vec![0, 0, 0, 0, 0, 0]).unwrap();
        assert_eq!(res, ans);

        let t3 = Tensor::<i32>::new(shape![0, 2], vec![]).unwrap();
        let t4 = Tensor::<i32>::new(shape![2, 2], vec![1, 2, 3, 4]).unwrap();
        let res2 = t3.clone().contract_mul_mt(t4.clone()).unwrap();
        assert_eq!(res2.shape(), shape![0, 2]);
        assert_eq!(res2.len(), 0);
    }

    #[test]
    fn mat_mul_with_empty_matrix() {
        let m1 = Matrix::<i32>::new(2, 0, vec![]).unwrap();
        let m2 = Matrix::<i32>::new(0, 3, vec![]).unwrap();

        let res = m1.clone().mat_mul(m2.clone()).unwrap();
        let ans = Matrix::<i32>::new(2, 3, vec![0, 0, 0, 0, 0, 0]).unwrap();
        assert_eq!(res, ans);

        let m3 = Matrix::<i32>::new(0, 2, vec![]).unwrap();
        let m4 = Matrix::<i32>::new(2, 2, vec![1, 2, 3, 4]).unwrap();
        let res2 = m3.clone().mat_mul(m4.clone()).unwrap();
        assert_eq!(res2.rows(), 0);
        assert_eq!(res2.cols(), 2);
        assert_eq!(res2.len(), 0);
    }

    #[test]
    fn mat_mul_mt_with_empty_matrix() {
        let m1 = Matrix::<i32>::new(2, 0, vec![]).unwrap();
        let m2 = Matrix::<i32>::new(0, 3, vec![]).unwrap();

        let res = m1.clone().mat_mul_mt(m2.clone()).unwrap();
        let ans = Matrix::<i32>::new(2, 3, vec![0, 0, 0, 0, 0, 0]).unwrap();
        assert_eq!(res, ans);

        let m3 = Matrix::<i32>::new(0, 2, vec![]).unwrap();
        let m4 = Matrix::<i32>::new(2, 2, vec![1, 2, 3, 4]).unwrap();
        let res2 = m3.clone().mat_mul_mt(m4.clone()).unwrap();
        assert_eq!(res2.rows(), 0);
        assert_eq!(res2.cols(), 2);
        assert_eq!(res2.len(), 0);
    }

    #[test]
    fn dot_scalar_tensors() {
        let t1 = Tensor::new(shape![], vec![5]).unwrap();
        let t2 = Tensor::new(shape![], vec![10]).unwrap();

        assert_eq!(t1.dot(&t2).unwrap(), 50);
    }

    #[test]
    fn dot_empty_tensors() {
        let t1 = Tensor::<i32>::new(shape![2, 0], vec![]).unwrap();
        let t2 = Tensor::<i32>::new(shape![2, 0], vec![]).unwrap();

        assert_eq!(t1.dot(&t2).unwrap(), 0);
    }

    #[test]
    fn dot_empty_matrices() {
        let m1 = Matrix::<i32>::new(2, 0, vec![]).unwrap();
        let m2 = Matrix::<i32>::new(2, 0, vec![]).unwrap();
        let m3 = Matrix::<i32>::new(0, 1, vec![]).unwrap();
        let m4 = Matrix::<i32>::new(0, 1, vec![]).unwrap();

        assert_eq!(m1.dot(&m2).unwrap(), 0);
        assert_eq!(m3.dot(&m4).unwrap(), 0);
    }

    #[test]
    fn tensor_contraction_multiplication() {
        let t1 = Tensor::<i32>::new(shape![2, 3], (0..6).collect()).unwrap();
        let t2 = Tensor::<i32>::new(shape![3, 3], (0..9).collect()).unwrap();
        let ans = Tensor::<i32>::new(shape![2, 3], vec![15, 18, 21, 42, 54, 66]).unwrap();
        assert_eq!(ans, t1.clone().contract_mul(t2.clone()).unwrap());

        let t3 = Tensor::<i32>::new(shape![2, 3], vec![5, 10, 2, 3, -2, 1]).unwrap();
        let t4 = Tensor::<i32>::new(shape![3, 2, 2], (0..12).collect()).unwrap();
        let ans = Tensor::<i32>::new(shape![2, 2, 2], vec![56, 73, 90, 107, 0, 2, 4, 6]).unwrap();
        assert_eq!(ans, t3.clone().contract_mul(t4.clone()).unwrap());
    }

    #[test]
    fn matrix_mul() {
        let m1 = Matrix::<i32>::new(3, 3, (0..9).collect()).unwrap();
        let m2 = Matrix::<i32>::new(3, 2, (0..6).collect()).unwrap();

        let res = m1.clone().mat_mul(m2.clone()).unwrap();
        let ans = Matrix::new(3, 2, vec![10, 13, 28, 40, 46, 67]).unwrap();

        assert_eq!(ans, res);
    }

    #[test]
    fn tensor_contraction_multiplication_shape_invalid() {
        let t1 = Tensor::<i32>::new(shape![2, 3], (0..6).collect()).unwrap();
        let t2 = Tensor::<i32>::new(shape![2, 2], (0..4).collect()).unwrap();
        let err = t1.clone().contract_mul(t2.clone()).unwrap_err();
        assert!(matches!(err, TensorErrors::IncompatibleShapes { .. }));
        let err = t1.clone().contract_mul_mt(t2.clone()).unwrap_err();
        assert!(matches!(err, TensorErrors::IncompatibleShapes { .. }));
    }

    #[test]
    fn invalid_shapes_for_mat_mul() {
        let m1 = Matrix::<i32>::rand(3, 3);
        let m2 = Matrix::<i32>::rand(2, 3);
        let err = m1.clone().mat_mul(m2.clone()).unwrap_err();
        let err_mt = m1.clone().mat_mul_mt(m2.clone()).unwrap_err();
        assert!(matches!(err, TensorErrors::IncompatibleShapes { .. }));
        assert!(matches!(err_mt, TensorErrors::IncompatibleShapes { .. }));
    }

    #[test]
    fn contract_mul_mt() {
        let t1 = Tensor::<f64>::rand(shape![10, 3, 20]);
        let t2 = Tensor::<f64>::rand(shape![20, 10]);

        let ans = t1.clone().contract_mul(t2.clone()).unwrap();
        let mt_ans = t1.clone().contract_mul_mt(t2.clone()).unwrap();

        assert!(approx_eq!(Tensor<f64>, ans, mt_ans, epsilon = 1e-10));
    }

    #[test]
    fn mat_mul_mt() {
        let m1 = Matrix::<f64>::rand(10, 10);
        let m2 = Matrix::<f64>::rand(10, 10);

        let ans = m1.clone().mat_mul(m2.clone()).unwrap();
        let mt_ans = m1.clone().mat_mul_mt(m2.clone()).unwrap();

        assert_eq!(ans, mt_ans);
    }

    #[test]
    fn dot_matrices() {
        let m1 = Matrix::new(10, 10, (0..100).collect()).unwrap();
        let m2 = Matrix::new(10, 10, (0..100).collect()).unwrap();

        assert_eq!(m1.dot(&m2).unwrap(), (99 * 100 * 199) / 6)
    }

    #[test]
    fn dot_mt_scalar_tensors() {
        let t1 = Tensor::new(shape![], vec![5]).unwrap();
        let t2 = Tensor::new(shape![], vec![10]).unwrap();

        assert_eq!(t1.dot_mt(&t2).unwrap(), 50);
    }

    #[test]
    fn dot_mt_empty_tensors() {
        let t1 = Tensor::<i32>::new(shape![2, 0], vec![]).unwrap();
        let t2 = Tensor::<i32>::new(shape![2, 0], vec![]).unwrap();

        assert_eq!(t1.dot_mt(&t2).unwrap(), 0);
    }

    #[test]
    fn dot_mt_empty_matrices() {
        let m1 = Matrix::<i32>::new(2, 0, vec![]).unwrap();
        let m2 = Matrix::<i32>::new(2, 0, vec![]).unwrap();

        assert_eq!(m1.dot_mt(&m2).unwrap(), 0);
    }

    #[test]
    fn dot_mt_correctness() {
        let t1 = Tensor::<f64>::rand(shape![100, 100]);
        let t2 = Tensor::<f64>::rand(shape![100, 100]);

        let ans = t1.dot(&t2).unwrap();
        let mt_ans = t1.dot_mt(&t2).unwrap();

        assert!(approx_eq!(f64, ans, mt_ans, epsilon = 1e-10));

        let m1 = Matrix::new(10, 10, (0..100).collect::<Vec<i32>>()).unwrap();
        let m2 = Matrix::new(10, 10, (0..100).collect::<Vec<i32>>()).unwrap();

        assert_eq!(m1.dot_mt(&m2).unwrap(), (99 * 100 * 199) / 6);
    }

    #[test]
    fn invalid_dot_mt() {
        let t1 = Tensor::new(shape![1, 2], vec![1, 2]).unwrap();
        let t2 = Tensor::new(shape![1, 3], vec![1, 2, 3]).unwrap();

        assert!(matches!(
            t1.dot_mt(&t2).unwrap_err(),
            TensorErrors::IncompatibleShapes { .. }
        ));

        let m1 = Matrix::<i32>::zeros(2, 2);
        let m2 = Matrix::<i32>::zeros(3, 3);

        assert!(matches!(
            m1.dot_mt(&m2).unwrap_err(),
            TensorErrors::IncompatibleShapes { .. }
        ));
    }

    #[test]
    fn invalid_dot() {
        let t1 = Tensor::new(shape![1, 2, 3], (0..6).collect()).unwrap();

        let t2 = Tensor::new(shape![2, 1, 3], (0..6).collect()).unwrap();

        let err = t1.dot(&t2).unwrap_err();
        assert!(matches!(err, TensorErrors::IncompatibleShapes { .. }));

        let m1 = Matrix::new(3, 2, (0..6).collect()).unwrap();
        let m2 = Matrix::new(2, 3, (0..6).collect()).unwrap();

        let err = m1.dot(&m2).unwrap_err();

        assert!(matches!(err, TensorErrors::IncompatibleShapes { .. }));
    }
}

// Tests for multiplication-based operations against a non-commutative element type.
//
// The single-threaded and multi-threaded implementations of the multiplication
// ops are otherwise tested with primitive numeric types where `a * b == b * a`,
// so they cannot catch order-of-operations bugs. The `NC` type below makes
// multiplication depend on the operand order (`x * y != y * x` in general),
// so the tests can verify that each implementation actually multiplies the
// operands in the documented order.

#[cfg(test)]
mod noncommutative_mul_tests {
    use std::ops::{Add, AddAssign, Mul};

    use num::Zero;

    use tensor_math::definitions::matrix::Matrix;
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

    // ---- Contract mul — Tensor ----------------------------------------------

    #[test]
    fn contract_mul_tensor_noncommutative() {
        // 2x3 * 3x2 -> 2x2 (matrix-style contraction)
        // A is row-major, B is row-major.
        let a = Tensor::<NC>::new(shape![2, 3], vec![nc(1), nc(2), nc(3), nc(4), nc(5), nc(6)])
            .unwrap();
        let b = Tensor::<NC>::new(
            shape![3, 2],
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
            shape![2, 2],
            vec![NC(58, 30), NC(64, 33), NC(139, 30), NC(154, 33)],
        )
        .unwrap();

        let got = a.clone().contract_mul(b.clone()).unwrap();
        assert_eq!(got, expected);

        // Sanity: A.contract_mul(B.clone()) and B.contract_mul(A.clone()) must differ.
        let a_b = a.clone().contract_mul(b.clone()).unwrap();
        let b_a = b.clone().contract_mul(a.clone()).unwrap();
        assert_ne!(a_b, b_a);
    }

    #[test]
    fn contract_mul_tensor_noncommutative_mt() {
        let a = Tensor::<NC>::new(shape![2, 3], vec![nc(1), nc(2), nc(3), nc(4), nc(5), nc(6)])
            .unwrap();
        let b = Tensor::<NC>::new(
            shape![3, 2],
            vec![nc(7), nc(8), nc(9), nc(10), nc(11), nc(12)],
        )
        .unwrap();

        let a_b_single = a.clone().contract_mul(b.clone()).unwrap();
        let b_a_single = b.clone().contract_mul(a.clone()).unwrap();

        let a_b_mt = a.clone().contract_mul_mt(b.clone()).unwrap();
        let b_a_mt = b.clone().contract_mul_mt(a.clone()).unwrap();

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

        let got = a.clone().mat_mul(b.clone()).unwrap();
        assert_eq!(got, expected);

        let a_b = a.clone().mat_mul(b.clone()).unwrap();
        let b_a = b.clone().mat_mul(a.clone()).unwrap();
        assert_ne!(a_b, b_a);
    }

    #[test]
    fn contract_mul_matrix_noncommutative_mt() {
        let a = Matrix::<NC>::new(2, 3, vec![nc(1), nc(2), nc(3), nc(4), nc(5), nc(6)]).unwrap();
        let b = Matrix::<NC>::new(3, 2, vec![nc(7), nc(8), nc(9), nc(10), nc(11), nc(12)]).unwrap();

        let a_b_single = a.clone().mat_mul(b.clone()).unwrap();
        let b_a_single = b.clone().mat_mul(a.clone()).unwrap();

        let a_b_mt = a.clone().mat_mul_mt(b.clone()).unwrap();
        let b_a_mt = b.clone().mat_mul_mt(a.clone()).unwrap();

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

        assert_eq!(
            a.clone().mat_mul(b.clone()).unwrap(),
            a.clone().mat_mul(b.clone()).unwrap()
        );
        assert_eq!(
            a.clone().mat_mul_mt(b.clone()).unwrap(),
            a.clone().mat_mul_mt(b.clone()).unwrap()
        );
    }

    // ---- Elementwise * (single-threaded) ------------------------------------

    #[test]
    fn elementwise_mul_tensor_noncommutative() {
        let t1 = Tensor::<NC>::new(shape![2], vec![nc(1), nc(2)]).unwrap();
        let t2 = Tensor::<NC>::new(shape![2], vec![nc(3), nc(4)]).unwrap();

        // NC(1,1)*NC(3,1)=NC(3, 1*3+1)=NC(3,4); NC(2,1)*NC(4,1)=NC(8, 1*4+1)=NC(8,5)
        let expected = Tensor::<NC>::new(shape![2], vec![NC(3, 4), NC(8, 5)]).unwrap();

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
        let a = Tensor::<NC>::new(shape![2], vec![nc(1), nc(2)]).unwrap();
        let b = Tensor::<NC>::new(shape![2], vec![nc(3), nc(4)]).unwrap();

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
        let a = Tensor::<NC>::new(shape![2], vec![nc(1), nc(2)]).unwrap();
        let b = Tensor::<NC>::new(shape![2], vec![nc(3), nc(4)]).unwrap();

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
