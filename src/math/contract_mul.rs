use std::cmp::min;
use std::ops::{Add, Mul};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::scope;
use num::Zero;
use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::definitions::shape::Shape;
use crate::definitions::tensor::Tensor;
use crate::definitions::traits::{IntoTensor};
use crate::shape;
use crate::utilities::internal_functions::dot_vectors;

impl<T: Clone + Add<Output = T> + Mul<Output = T> + Zero> Tensor<T> {
    /// Perform tensor-contraction multiplication,
    /// which is a more general form of matrix multiplication.
    /// E.g: A tensor of shape (a,b,c) multiplied in this way by a tensor of shape (c, d, e, f)
    /// will produce a tensor of shape (a, b, d, e, f) by the following formula:
    /// result[&[i, j, k, l, m\]\] = sum(x=0, x=c) { first[&[i, j, x\]\] * second[&[x, k, l, m\]\] }.
    pub fn contract_mul(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorErrors> {
        if self.rank() == 0 {
            return Ok(other.map_refs(|x| self.elements[0].clone() * x.clone()))
        }

        if other.rank() == 0 {
            return Ok(self * other.elements[0].clone())
        }

        if self.shape.0.last().unwrap() != other.shape.0.first().unwrap() {
            return Err(TensorErrors::ShapesIncompatible);
        }

        let mut resultant_shape_vec = self
            .shape
            .0
            .iter()
            .take(self.rank() - 1)
            .cloned()
            .collect::<Vec<usize>>();

        resultant_shape_vec.extend(
            other
                .shape
                .0
                .iter()
                .rev()
                .take(other.rank() - 1)
                .rev()
                .cloned(),
        );
        let resultant_shape: Shape = resultant_shape_vec.into();
        let mut resultant_elements: Vec<T> = Vec::with_capacity(resultant_shape.element_count());

        for i in 0..resultant_shape.element_count() {
            let index = resultant_shape.tensor_index(i)?;
            let (self_chunk, other_chunk) = index.split_at(self.rank() - 1);
            let mut self_elements: Vec<T> = Vec::with_capacity(*self.shape.0.last().unwrap());
            let mut other_elements: Vec<T> = Vec::with_capacity(*other.shape.0.first().unwrap());

            for j in 0..*self.shape.0.last().unwrap() {
                let mut self_index = self_chunk.to_vec();
                self_index.push(j);

                self_elements.push(self[&self_index].clone());

                let mut other_index = other_chunk.to_vec();
                other_index.insert(0, j);
                other_elements.push(other[&other_index].clone());
            }

            resultant_elements.push(dot_vectors(&self_elements, &other_elements));
        }

        resultant_elements.into_tensor().reshape(&resultant_shape)
    }

    /// Computes the dot product of two tensors, i.e. the element-wise product, then the sum of the result
    pub fn dot(&self, other: &Tensor<T>) -> Result<T, TensorErrors> {
        if self.shape != other.shape {
            return Err(TensorErrors::ShapesIncompatible);
        }

        Ok((self * other).sum())
    }
}

impl<T: Clone + Add<Output = T> + Mul<Output = T> + Zero> Matrix<T> {
    /// Does matrix multiplication with another matrix.
    pub fn contract_mul(&self, other: &Matrix<T>) -> Result<Matrix<T>, TensorErrors> {
        self.tensor.contract_mul(&other.tensor)?.try_into()
    }

    /// Does matrix multiplication with another matrix.
    pub fn mat_mul(&self, other: &Matrix<T>) -> Result<Matrix<T>, TensorErrors> {
        self.contract_mul(other)
    }
    
    /// Computes the dot product of the two matrices, i.e. the elementwise product, then the sum of the result.
    pub fn dot(&self, other: &Matrix<T>) -> Result<T, TensorErrors> {
        if self.shape != other.shape {
            return Err(TensorErrors::ShapesIncompatible);
        }
        
        Ok((self * other).sum())    
    }
}

impl<T: Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync> Tensor<T> {
    /// Perform tensor-contraction multiplication (using multiple threads),
    /// which is a more general form of matrix multiplication.
    /// E.g: A tensor of shape (a,b,c) multiplied in this way by a tensor of shape (c, d, e, f)
    /// will produce a tensor of shape (a, b, d, e, f) by the following formula:
    /// result[&[i, j, k, l, m\]\] = sum(x=0, x=c) { first[&[i, j, x\]\] * second[&[x, k, l, m\]\] }.
    pub fn contract_mul_mt(&self, other: &Tensor<T>) -> Result<Tensor<T>, TensorErrors> {
        if self.rank() == 0 {
            return Ok(other.map_refs(|x| self.elements[0].clone() * x.clone()));
        }

        if other.rank() == 0 {
            return Ok(self * other.elements[0].clone());
        }

        if self.shape.0.last().unwrap() != other.shape.0.first().unwrap() {
            return Err(TensorErrors::ShapesIncompatible);
        }

        let mut resultant_shape_vec = self
            .shape
            .0
            .iter()
            .take(self.rank() - 1)
            .cloned()
            .collect::<Vec<usize>>();

        resultant_shape_vec.extend(
            other
                .shape
                .0
                .iter()
                .rev()
                .take(other.rank() - 1)
                .rev()
                .cloned(),
        );
        let res_shape = resultant_shape_vec.into();

        let res_mutexes = Arc::new(
            RwLock::new(
                Tensor::new(
                    &res_shape,
                    (0..res_shape.element_count())
                        .map(|_| Mutex::new(T::zero()))
                        .collect(),
                )?,
            ),
        );

        let self_arc = Arc::new(self);
        let res_shape_arc = Arc::new(res_shape);
        let elems_per_thread = 1;  // The number of elements each thread is responsible for

        scope(|s| {
            for i in 0..res_shape_arc.clone().element_count() / elems_per_thread {
                let res_shape_arc_clone = res_shape_arc.clone();
                let res_arc_clone = res_mutexes.clone();

                let self_arc_clone = self_arc.clone();

                let start = i * elems_per_thread;
                let end = min((i + 1) * elems_per_thread, res_shape_arc.element_count());

                s.spawn(move || {
                    let res_read = res_arc_clone.read().unwrap();

                    for j in start..end {
                        let t_index = res_shape_arc_clone.tensor_index(j).unwrap();

                        let (self_part, other_part) = t_index.split_at(self_arc_clone.rank() - 1);

                        let mut self_indices = self_part.iter().map(|&x| x..x+1).collect::<Vec<_>>();
                        self_indices.push(0..self_arc_clone.shape.0.last().unwrap().clone());

                        let mut other_indices = other_part.iter().map(|&x| x..x+1).collect::<Vec<_>>();
                        other_indices.insert(0, 0..other.shape.0.first().unwrap().clone());

                        let self_elems = self_arc_clone.slice(&self_indices).unwrap().reshape(&shape![self_arc_clone.shape.0.last().unwrap().clone()]).unwrap();
                        let other_elems = other.slice(&other_indices).unwrap().reshape(&shape![other.shape.0.first().unwrap().clone()]).unwrap();

                        let elem_res = self_elems.dot(&other_elems).unwrap();

                        let mut write_lock = res_read[&t_index].lock().unwrap();

                        *write_lock = elem_res;
                    }
                });
            }
        });

        let res_read = res_mutexes.read().unwrap();
        res_read.iter().map(|x| x.lock().unwrap().clone()).collect::<Tensor<T>>().reshape(&res_shape_arc.clone())
    }
}

impl<T: Clone + Add<Output = T> + Mul<Output = T> + Send + Sync + Zero> Matrix<T> {
    /// Does matrix multiplication on multiple threads
    pub fn contract_mul_mt(&self, other: &Matrix<T>) -> Result<Matrix<T>, TensorErrors> {
        self.tensor.contract_mul_mt(&other.tensor)?.try_into()
    }

    /// Does matrix multiplication on multiple threads. This is just an alternate name for the method
    pub fn mat_mul_mt(&self, other: &Matrix<T>) -> Result<Matrix<T>, TensorErrors> {
        self.contract_mul_mt(other)
    }
}
