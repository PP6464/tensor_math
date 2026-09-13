use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::definitions::shape::Shape;
use crate::definitions::tensor::Tensor;
use crate::definitions::traits::IntoTensor;
use crate::shape;
use crate::utilities::internal_functions::dot_vectors;
use num::Zero;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator};
use rayon::iter::ParallelIterator;
use rayon::slice::{ParallelSlice, ParallelSliceMut};
use std::ops::{AddAssign, Mul};

/*
--------------------------------------------
* ?Tensor contract multiplication
--------------------------------------------
*/

impl<T> Tensor<T> {
    /// Perform tensor-contraction multiplication,
    /// which is a more general form of matrix multiplication.
    /// E.g: A tensor of shape (a,b,c) multiplied in this way by a tensor of shape (c, d, e, f)
    /// will produce a tensor of shape (a, b, d, e, f) by the following formula:
    /// result[&[i, j, k, l, m\]\] = sum(x=0, x=c) { first[&[i, j, x\]\] * second[&[x, k, l, m\]\] }.
    /// This fails if the tensors have shapes that are not compatible.
    pub fn contract_mul(mut self, mut other: Tensor<T>) -> Result<Tensor<T>, TensorErrors>
    where
        T: AddAssign + Mul<Output = T> + Clone + Zero,
    {
        let self_shape = self.shape().0;
        let self_rank = self.rank();
        let other_shape = other.shape().0;
        let other_rank = other.rank();

        unsafe {
            if self_rank == 0 {
                let e = self.elements.get_unchecked(0);
                return Ok(other.map(|x| e.clone() * x));
            }

            if other_rank == 0 {
                let e = other.elements.get_unchecked(0);
                return Ok(self * e);
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
            other =
                other.reshape_unchecked(shape![other_shape[0], other_shape[1..].iter().product()]);
            let other_mat = Matrix {
                rows: other.shape[0],
                cols: other.shape[1],
                elements: other.elements,
            };

            let result_mat = self_mat.mat_mul(other_mat)?;
            let result_shape = [
                self_shape[..self_rank - 1].to_vec(),
                other_shape[1..].to_vec(),
            ]
            .concat();

            Ok(result_mat
                .into_tensor()
                .reshape_unchecked(Shape::new(result_shape)))
        }
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
        other =
            other.reshape_unchecked(shape![other_shape[0], other_shape[1..].iter().product()]);
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
    pub fn contract_mul_mt(mut self, mut other: Tensor<T>) -> Result<Tensor<T>, TensorErrors>
    where
        T: AddAssign + Mul<Output = T> + Clone + Zero + Send + Sync,
    {
        let self_shape = self.shape().0;
        let self_rank = self.rank();
        let other_shape = other.shape().0;
        let other_rank = other.rank();

        unsafe {
            if self_rank == 0 {
                let e = self.elements.get_unchecked(0);
                return Ok(other.map(|x| e.clone() * x));
            }

            if other_rank == 0 {
                let e = other.elements.get_unchecked(0);
                return Ok(self * e);
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
            other =
                other.reshape_unchecked(shape![other_shape[0], other_shape[1..].iter().product()]);
            let other_mat = Matrix {
                rows: other.shape[0],
                cols: other.shape[1],
                elements: other.elements,
            };

            let result_mat = self_mat.mat_mul_mt(other_mat)?;
            let result_shape = [
                self_shape[..self_rank - 1].to_vec(),
                other_shape[1..].to_vec(),
            ]
                .concat();

            Ok(result_mat
                .into_tensor()
                .reshape_unchecked(Shape::new(result_shape)))
        }
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
        other =
            other.reshape_unchecked(shape![other_shape[0], other_shape[1..].iter().product()]);
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

        let mut elements = Vec::with_capacity(self.rows * other.cols);
        let other_transpose = other.transpose();

        self.chunks(self.cols).for_each(|row| {
            other_transpose
                .chunks(other_transpose.cols)
                .for_each(|col| {
                    elements.push(dot_vectors(row, col));
                });
        });

        Ok(Matrix {
            rows: self.rows,
            cols: other_transpose.rows,
            elements,
        })
    }

    /// Does matrix multiplication with another matrix without validity checking.
    pub(crate) unsafe fn mat_mul_unchecked(self, other: Matrix<T>) -> Matrix<T>
    where
        T: AddAssign + Mul<Output = T> + Clone + Zero,
    {
        let mut elements = Vec::with_capacity(self.rows * other.cols);
        let other_transpose = other.transpose();

        self.chunks(self.cols).for_each(|row| {
            other_transpose
                .chunks(other_transpose.cols)
                .for_each(|col| {
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

        let mut elements = Vec::with_capacity(self.rows * other.cols);
        let buf = elements.spare_capacity_mut();
        let other_transpose = other.transpose_mt();

        self.par_chunks(self.cols)
            .zip(buf.par_chunks_mut(self.rows))
            .for_each(|(row, outs)| {
                other_transpose
                    .par_chunks(other_transpose.cols)
                    .zip(outs.par_iter_mut())
                    .for_each(|(col, out)| {
                        out.write(dot_vectors(row, col));
                    });
            });

        Ok(Matrix {
            rows: self.rows,
            cols: other_transpose.rows,
            elements,
        })
    }

    /// Does matrix multiplication on multiple threads without validity checking.
    pub fn mat_mul_unchecked_mt(self, other: Matrix<T>) -> Matrix<T>
    where
        T: AddAssign + Mul<Output = T> + Clone + Zero + Send + Sync,
    {
        let mut elements = Vec::with_capacity(self.rows * other.cols);
        let buf = elements.spare_capacity_mut();
        let other_transpose = other.transpose_mt();

        self.par_chunks(self.cols)
            .zip(buf.par_chunks_mut(self.rows))
            .for_each(|(row, outs)| {
                other_transpose
                    .par_chunks(other_transpose.cols)
                    .zip(outs.par_iter_mut())
                    .for_each(|(col, out)| {
                        out.write(dot_vectors(row, col));
                    });
            });

        Matrix {
            rows: self.rows,
            cols: other_transpose.rows,
            elements,
        }
    }
}
