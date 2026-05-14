use std::ops::{Mul};
use std::thread::scope;
use crate::definitions::matrix::Matrix;
use crate::definitions::shape::Shape;
use crate::definitions::tensor::Tensor;
use crate::definitions::traits::{IntoMatrix, IntoTensor};

impl<T: Clone + Mul<Output = T>> Tensor<T> {
    /// Implements the Kronecker product for any two things that can be converted to tensors.
    /// The Kronecker product scales the second tensor by each element of the first tensor,
    /// giving a `Tensor` of type `Tensor<T>`. This is then simplified into just `Tensor<T>`
    /// with the result having a shape that is the element-wise product of the two input
    /// tensors' shapes (if one has a lower rank than the other, then the rest of the larger rank
    /// tensor's shape values are inserted afterward).
    pub fn kronecker(&self, other: &Tensor<T>) -> Tensor<T> {
        let mut new_shape_vec = Vec::new();

        if self.rank() > other.rank() {
            for i in 0..other.rank() {
                new_shape_vec.push(self.shape[i] * other.shape[i]);
            }

            for i in other.rank()..self.rank() {
                new_shape_vec.push(self.shape[i]);
            }
        } else {
            for i in 0..self.rank() {
                new_shape_vec.push(self.shape[i] * other.shape[i]);
            }

            for i in self.rank()..other.rank() {
                new_shape_vec.push(other.shape[i]);
            }
        }

        let new_shape = Shape::new(new_shape_vec);
        let mut new_elements = Vec::with_capacity(new_shape.element_count());

        for i in self.elements.iter().cloned() {
            let tensor = other * i;
            new_elements.extend(tensor.elements);
        }

        new_elements.into_tensor().reshape(&new_shape).unwrap()
    }
}

impl<T: Clone + Mul<Output = T>> Matrix<T> {
    pub fn kronecker(&self, other: &Matrix<T>) -> Matrix<T> {
        let res_tensor = self.tensor.kronecker(&other.tensor);
        Matrix {
            rows: res_tensor.shape[0],
            cols: res_tensor.shape[1],
            tensor: res_tensor,
        }
    }
}

impl<T: Clone + Mul<Output = T> + Send + Sync> Tensor<T> {
    /// Computes the Kronecker product using multiple threads
    pub fn kronecker_mt(&self, other: &Tensor<T>) -> Tensor<T> {
        let mut new_shape_vec = Vec::new();

        if self.rank() > other.rank() {
            for i in 0..other.rank() {
                new_shape_vec.push(self.shape[i] * other.shape[i]);
            }

            for i in other.rank()..self.rank() {
                new_shape_vec.push(self.shape[i]);
            }
        } else {
            for i in 0..self.rank() {
                new_shape_vec.push(self.shape[i] * other.shape[i]);
            }

            for i in self.rank()..other.rank() {
                new_shape_vec.push(other.shape[i]);
            }
        }

        let new_shape = Shape::new(new_shape_vec);

        if new_shape.element_count() == 0 {
            return Tensor::new(&new_shape, vec![]).unwrap();
        }

        let mut new_elements = (0..new_shape.element_count()).map(|_| self.first().unwrap().clone()).collect::<Vec<T>>();

        scope(|s| {
            let mut new_elements_chunks = new_elements.chunks_mut(other.shape.element_count());

            for elem in self.iter() {
                let new_elements_chunk = new_elements_chunks.next().unwrap();

                s.spawn(move || {
                    let res = other * elem;

                    for (j, e) in res.iter().enumerate() {
                        new_elements_chunk[j] = e.clone();
                    }
                });
            }
        });

        new_elements.into_tensor().reshape(&new_shape).unwrap()
    }
}

impl<T: Clone + Mul<Output = T> + Send + Sync> Matrix<T> {
    /// Computes the Kronecker product using multiple threads
    pub fn kronecker_mt(&self, other: &Matrix<T>) -> Matrix<T> {
        let mut new_elements = (0..self.rows * self.cols * other.rows * other.cols)
            .map(|_| self[(0, 0)].clone())
            .collect::<Vec<T>>();

        if new_elements.len() == 0 {
            return Matrix::new(self.rows * other.rows, self.cols * other.cols, vec![]).unwrap();
        }

        scope(|s| {
            let mut new_elements_chunks = new_elements.chunks_mut(other.shape.element_count());

            for elem in self.iter() {
                let new_elements_chunk = new_elements_chunks.next().unwrap();

                s.spawn(move || {
                    let res = other * elem;

                    for (j, e) in res.iter().enumerate() {
                        new_elements_chunk[j] = e.clone();
                    }
                });
            }
        });

        new_elements.into_matrix().reshape(self.rows * other.rows, self.cols * other.cols).unwrap()
    }
}
