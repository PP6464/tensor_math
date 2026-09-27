use crate::definitions::matrix::Matrix;
use crate::definitions::shape::Shape;
use crate::definitions::strides::Strides;
use crate::definitions::tensor::Tensor;
use crate::utilities::internal_functions::dot_vectors;
use rayon::iter::ParallelIterator;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator};
use rayon::prelude::ParallelSliceMut;
use rayon::slice::ParallelSlice;
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

        for self_row in self.elements.chunks(self.cols) {
            for other_row in other.elements.chunks(other.cols) {
                for e in self_row {
                    elements.extend(other_row.iter().cloned().map(|x| e.clone() * x))
                }
            }
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
        let buf = elements.spare_capacity_mut();

        self.elements
            .par_chunks(self.cols)
            .zip(buf.par_chunks_mut(self.cols * other.cols))
            .for_each(|(self_row, out_row)| {
                self_row
                    .iter()
                    .zip(out_row.chunks_mut(other.cols))
                    .for_each(|(e, out_section)| {
                        other.elements.chunks(other.cols).for_each(|other_row| {
                            out_section
                                .iter_mut()
                                .zip(other_row.iter().cloned())
                                .for_each(|(out, f)| {
                                    out.write(e.clone() * f);
                                })
                        });
                    })
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
