use crate::definitions::errors::TensorErrors;
use crate::definitions::shape::Shape;
use crate::definitions::strides::Strides;
use crate::definitions::tensor::Tensor;
use crate::definitions::tensor_slice_mut::TensorSliceMut;
use crate::definitions::traits::IntoTensor;
use crate::definitions::transpose::Transpose;
use crate::shape;
use crate::utilities::internal_functions::dot_vectors;
use num::{ToPrimitive, Zero};
use std::collections::HashSet;
use std::ops::{Add, Div, Range};
use std::sync::Arc;
use std::thread::scope;

impl<T> Tensor<T> {
    /// Reshape a `Tensor`, consuming it and returning the new one
    pub fn reshape(self, new_shape: &Shape) -> Result<Tensor<T>, TensorErrors> {
        if new_shape.element_count() != self.shape.element_count() {
            return Err(TensorErrors::ShapeSizeDoesNotMatch);
        }

        Ok(Tensor::new(new_shape, self.elements)?)
    }

    /// Flatten a `Tensor` on a given dimension, consuming it and returning the new one
    pub fn flatten(self, dim: usize) -> Result<Tensor<T>, TensorErrors> {
        if dim >= self.rank() {
            return Err(TensorErrors::AxisOutOfBounds {
                axis: dim,
                rank: self.rank(),
            });
        }

        if self.shape[dim] != 1 {
            return Err(TensorErrors::AxisIsNotOne(dim));
        }

        let mut copy = self.shape;
        copy.0.remove(dim);
        Ok(Tensor::new(&copy, self.elements)?)
    }

    /// Apply a transformation to a `Tensor` element-wise, consuming the original and returning the result
    pub fn map<O>(self, closure: impl FnMut(T) -> O) -> Tensor<O> {
        let shape = self.shape().clone();
        let new_elements = self.elements.into_iter().map(closure).collect();
        Tensor::new(&shape, new_elements).unwrap()
    }

    /// Apply a transformation to a tensor element-wise, but not consuming the original tensor
    pub fn map_refs<O>(&self, closure: impl FnMut(&T) -> O) -> Tensor<O> {
        let shape = self.shape().clone();
        let new_elements = self.elements.iter().map(closure).collect();
        Tensor::new(&shape, new_elements).unwrap()
    }
}

impl<T: Clone> Tensor<T> {
    /// Creates a tensor from a single value with specified shape
    pub fn from_value(shape: &Shape, value: T) -> Self {
        let elements = vec![value; shape.element_count()];
        Tensor::new(shape, elements).unwrap()
    }

    /// Concatenates a `Tensor` with another `Tensor` along the specified dimension
    pub fn concat(&self, other: &Tensor<T>, dim: usize) -> Result<Tensor<T>, TensorErrors> {
        if self.rank() != other.rank() {
            return Err(TensorErrors::RanksDoNotMatch(self.rank(), other.rank()));
        }
        let mut resultant_shape: Vec<usize> = Vec::with_capacity(self.rank());

        // Ensure shapes match on every dim that is not the dim along which to concatenate
        for i in 0..self.rank() {
            if i == dim {
                resultant_shape.push(self.shape[i] + other.shape[i]);
                continue;
            }

            if self.shape[i] != other.shape[i] {
                return Err(TensorErrors::ShapesIncompatible);
            }

            resultant_shape.push(self.shape[i]);
        }

        let resultant_shape: Shape = resultant_shape.try_into()?;
        let mut resultant_elements: Vec<T> = Vec::with_capacity(resultant_shape.element_count());

        if dim == 0 {
            // If the dimension is 0 we can just merge the elements one after another
            resultant_elements = self.elements.clone();
            resultant_elements.extend(other.elements.clone());
            return Ok(Tensor::new(&resultant_shape, resultant_elements)?);
        }

        let mut self_chunks = self.elements.chunks(self.strides[dim - 1]);
        let mut other_chunks = other.elements.chunks(other.strides[dim - 1]);

        // Merge together chunks from self and other in the correct manner to get
        // the result for concatenating self and other (in that order)
        // Note self_chunks and other_chunks have the same length
        // Because their shapes are the same in the dimensions to the left of the concatenation dim
        for _ in 0..self_chunks.len() {
            resultant_elements.extend_from_slice(self_chunks.next().unwrap());
            resultant_elements.extend_from_slice(other_chunks.next().unwrap());
        }

        let result = Tensor::new(&resultant_shape, resultant_elements)?;

        Ok(result)
    }

    /// Give an iterable that is enumerated with tensor indices
    pub fn enumerated_iter(&self) -> impl Iterator<Item = (Vec<usize>, T)> + '_ {
        self.elements
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, element)| (self.shape.tensor_index(index), element))
    }

    /// Give a mutable iterable that is enumerated with tensor indices
    pub fn enumerated_iter_mut(&mut self) -> impl Iterator<Item = (Vec<usize>, &mut T)> + '_  {
        self.elements
            .iter_mut()
            .enumerate()
            .map(|(index, element)| (self.shape.tensor_index(index), element))
    }

    /// Gives a cloned immutable slice to a region in the tensor specified
    /// by an array of range of indices for each dimension of the tensor
    pub fn slice(&self, indices: &[Range<usize>]) -> Result<Tensor<T>, TensorErrors> {
        // We have to check this up front because otherwise we would have subtraction with overflow
        for range in indices.iter() {
            if range.start >= range.end {
                return Err(TensorErrors::InvalidNonEmptyInterval {
                    max: range.end as f64,
                    min: range.start as f64,
                })
            }
        }

        let start = indices.iter().map(|range| range.start).collect::<Vec<usize>>();

        let res_shape = Shape::new(indices.iter().map(|r| { r.end - r.start }).collect())?;

        if indices.len() != self.rank() {
            return Err(TensorErrors::SliceIncompatibleShape {
                slice_shape: res_shape,
                tensor_shape: self.shape.clone(),
            });
        }

        for (i, range) in indices.iter().enumerate() {
            if range.end > self.shape[i] {
                return Err(TensorErrors::SliceIndicesOutOfBounds {
                    start: range.start,
                    end: range.end,
                    axis: i,
                    length: self.shape[i],
                });
            }
        }

        let mut res = Tensor::<T>::from_value(&res_shape, self.first().unwrap().clone());

        for (pos, val) in res.enumerated_iter_mut() {
            let orig_index = pos.iter().zip(start.iter()).map(|(x, y)| x + y).collect::<Vec<usize>>();

            *val = self[&orig_index.as_slice()].clone();
        }

        Ok(res)
    }

    /// Gives a mutable slice to a tensor in the specified range of indices.
    pub fn slice_mut(&'_ mut self, indices: &[Range<usize>]) -> Result<TensorSliceMut<'_, T>, TensorErrors> {
        if indices.len() != self.rank() {
            return Err(TensorErrors::SliceIncompatibleShape {
                slice_shape: Shape::new(indices.iter().map(|r| r.end - r.start).collect())?,
                tensor_shape: self.shape.clone(),
            });
        }

        for (i, range) in indices.iter().enumerate() {
            if range.end > self.shape[i] {
                return Err(TensorErrors::SliceIndicesOutOfBounds {
                    start: range.start,
                    end: range.end,
                    axis: i,
                    length: self.shape[i],
                });
            }
            
            if range.start >= range.end {
                return Err(TensorErrors::InvalidNonEmptyInterval {
                    max: range.end as f64,
                    min: range.start as f64,
                });
            }
        }

        let start = indices.iter().map(|range| range.start).collect::<Vec<usize>>();
        let end = indices.iter().map(|range| range.end).collect::<Vec<usize>>();

        Ok(TensorSliceMut {
            start,
            end,
            orig: self,
        })
    }

    /// Flips a tensor along a list of specified axes
    pub fn flip_axes(&self, axes: &HashSet<usize>) -> Result<Tensor<T>, TensorErrors> {
        for &axis in axes.iter() {
            if axis >= self.rank() {
                return Err(TensorErrors::AxisOutOfBounds {
                    axis,
                    rank: self.rank(),
                });
            }
        }

        let mut res = self.clone();

        for (i, e) in self.enumerated_iter() {
            let mut new_index = i.clone();

            for &axis in axes {
                new_index[axis] = self.shape[axis] - i[axis] - 1;
            }

            res[&new_index] = e;
        }

        Ok(res)
    }

    /// Flips a tensor along all axes
    pub fn flip(&self) -> Tensor<T> {
        self.flip_axes(&(0..self.rank()).collect()).unwrap()
    }

    /// Transposes a tensor and returns the result
    pub fn transpose(&self, transpose: &Transpose) -> Result<Tensor<T>, TensorErrors> {
        if transpose.permutation.len() != self.shape().rank() {
            return Err(TensorErrors::TransposeIncompatibleRank {
                rank: self.rank(),
                trank: transpose.permutation.len(),
            });
        }

        let new_shape = transpose.new_shape(self.shape())?;
        let new_strides = Strides::from_shape(&new_shape);
        let mut new_elements = self.elements().clone();

        for (old_index, elem) in self.enumerated_iter() {
            let new_index = transpose.new_index(&old_index)?;
            let new_addr = dot_vectors(&new_index, &new_strides.0);

            new_elements[new_addr] = elem;
        }

        Ok(Tensor::new(&new_shape, new_elements.to_vec())?)
    }

    /// Pools a `Tensor<T>` into a `Tensor<O>` using a custom pooling function.
    /// The custom function will take a `Tensor<T>` that corresponds to the slice that the kernel covers.
    /// If the kernel is hanging over the edge of the tensor, then only the bits of the tensor that fit are included.
    /// This is reflected in the shape of the input tensor.
    /// Default functions for `max` and `avg` are given as well.
    pub fn pool<O: Clone>(
        &self, 
        pool_fn: impl Fn(Tensor<T>) -> O,
        kernel_shape: &Shape, 
        stride_shape: &Shape, 
        init: O,
    ) -> Result<Tensor<O>, TensorErrors> {
        if kernel_shape.rank() != self.rank() {
            return Err(TensorErrors::RanksDoNotMatch(kernel_shape.rank(), self.rank()));
        }

        if stride_shape.rank() != self.rank() {
            return Err(TensorErrors::RanksDoNotMatch(stride_shape.rank(), self.rank()));
        }

        let res_shape = &Shape::new(
            self.shape()
                .0
                .iter()
                .cloned()
                .zip(stride_shape.0.iter().cloned())
                .map(|(x, y)| x.div_ceil(y))
                .collect::<Vec<usize>>(),
        )?;

        let mut result = Tensor::<O>::from_value(res_shape, init);

        for (pos, val) in result.enumerated_iter_mut() {
            let start_pos = pos
                .iter()
                .zip(stride_shape.0.iter())
                .map(|(x, y)| x * y)
                .collect::<Vec<usize>>();
            let end_pos = start_pos
                .iter()
                .zip(kernel_shape.0.iter())
                .enumerate()
                .map(|(i, (x, y))| {
                    let shape_val = self.shape[i];

                    if x + y < shape_val {
                        x + y
                    } else {
                        shape_val
                    }
                })
                .collect::<Vec<usize>>();

            let indices = end_pos
                .iter()
                .zip(start_pos.iter())
                .map(|(x, y)| *y..*x)
                .collect::<Vec<_>>();

            *val = pool_fn(self.slice(&indices)?);
        }

        Ok(result)
    }

    /// Pools a `Tensor<T>` into a `Tensor<O>` using a custom pooling function with the index.
    /// The custom function will take a `Tensor<T>` that corresponds to the slice that the kernel covers.
    /// If the kernel is hanging over the edge of the tensor, then only the bits of the tensor that fit are included.
    /// This is reflected in the shape of the input tensor.
    /// Default functions for `max` and `avg` are given as well.
    pub fn pool_indexed<O: Clone>(
        &self, 
        pool_fn: impl Fn(Vec<usize>, Tensor<T>) -> O,
        kernel_shape: &Shape, 
        stride_shape: &Shape,
        init: O,
    ) -> Result<Tensor<O>, TensorErrors> {
        if kernel_shape.rank() != self.rank() {
            return Err(TensorErrors::RanksDoNotMatch(kernel_shape.rank(), self.rank()));
        }

        if stride_shape.rank() != self.rank() {
            return Err(TensorErrors::RanksDoNotMatch(stride_shape.rank(), self.rank()));
        }

        let res_shape = &Shape::new(
            self.shape()
                .0
                .iter()
                .cloned()
                .zip(stride_shape.0.iter().cloned())
                .map(|(x, y)| x.div_ceil(y))
                .collect::<Vec<usize>>(),
        )?;

        let mut result = Tensor::<O>::from_value(res_shape, init);

        for (pos, val) in result.enumerated_iter_mut() {
            let start_pos = pos
                .iter()
                .zip(stride_shape.0.iter())
                .map(|(x, y)| x * y)
                .collect::<Vec<usize>>();
            let end_pos = start_pos
                .iter()
                .zip(kernel_shape.0.iter())
                .enumerate()
                .map(|(i, (x, y))| {
                    let shape_val = self.shape[i];

                    if x + y < shape_val {
                        x + y
                    } else {
                        shape_val
                    }
                })
                .collect::<Vec<usize>>();

            let indices = end_pos
                .iter()
                .zip(start_pos.iter())
                .map(|(x, y)| *y..*x)
                .collect::<Vec<_>>();

            *val = pool_fn(start_pos, self.slice(&indices)?);
        }

        Ok(result)
    }
}

impl<T: Clone + Send + Sync> Tensor<T> {
    /// Concatenates a `Tensor` with another `Tensor` along the specified axis.
    /// This spawns multiple threads to handle writing to the result at the same time.
    pub fn concat_mt(&self, other: &Tensor<T>, axis: usize) -> Result<Tensor<T>, TensorErrors> {
        if self.rank() != other.rank() {
            return Err(TensorErrors::RanksDoNotMatch(self.rank(), other.rank()));
        }
        let mut resultant_shape: Vec<usize> = Vec::with_capacity(self.rank());

        // Ensure shapes match on every dim that is not the dim along which to concatenate
        for i in 0..self.rank() {
            if i == axis {
                resultant_shape.push(self.shape[i] + other.shape[i]);
                continue;
            }

            if self.shape[i] != other.shape[i] {
                return Err(TensorErrors::ShapesIncompatible);
            }

            resultant_shape.push(self.shape[i]);
        }

        let resultant_shape: Shape = resultant_shape.try_into()?;

        if axis == 0 {
            // If the dimension is 0 we can just merge the elements one after another
            let mut res_elems = self.elements.clone();
            res_elems.extend(other.elements.clone());
            return Ok(Tensor::new(&resultant_shape, res_elems)?);
        }

        let mut res_elems = vec![self.first().unwrap().clone(); resultant_shape.element_count()];

        let self_chunk_size = self.strides[axis - 1];
        let other_chunk_size = other.strides[axis - 1];

        let chunks_per_thread = 5; // The number of chunks a single thread manages

        // Merge together chunks from self and other in the correct manner to get
        // the result for concatenating self and other (in that order)
        // Note self_chunks and other_chunks have the same length
        // Because their shapes are the same in the dimensions to the left of the concatenation dim

        scope(|s| {
            let thread_chunk_size_self = self_chunk_size * chunks_per_thread;
            let thread_chunk_size_other = other_chunk_size * chunks_per_thread;

            let thread_chunks_self = self.elements.chunks(thread_chunk_size_self);
            let thread_chunks_other = other.elements.chunks(thread_chunk_size_other);

            let thread_chunks = thread_chunks_self.zip(thread_chunks_other);

            let mut res_elem_chunks = res_elems.chunks_mut(thread_chunk_size_self + thread_chunk_size_other);

            for (self_thread_chunk, other_thread_chunk) in thread_chunks {
                let res_chunk = res_elem_chunks.next().unwrap();

                s.spawn(|| {
                    let mut self_chunks = self_thread_chunk.chunks(self_chunk_size);
                    let mut other_chunks = other_thread_chunk.chunks(other_chunk_size);
                    let actual_count = self_chunks.len();  // On the last chunk it may not have as many as chunks_per_thread chunks

                    for j in 0..actual_count {
                        let current_self_chunk = self_chunks.next().unwrap();
                        let current_other_chunk = other_chunks.next().unwrap();

                        let start = j * (self_chunk_size + other_chunk_size);

                        for (k, self_val) in current_self_chunk.iter().enumerate() {
                            res_chunk[start + k] = self_val.clone();
                        }

                        for (k, other_val) in current_other_chunk.iter().enumerate() {
                            res_chunk[start + self_chunk_size + k] = other_val.clone();
                        }
                    }
                });
            }
        });

        let result = Tensor::new(&resultant_shape, res_elems)?;
        Ok(result)
    }

    /// Flips a tensor along specified axes
    pub fn flip_axes_mt(&self, axes: &HashSet<usize>) -> Result<Tensor<T>, TensorErrors> {
        let mut res = self.clone();
        let self_arc = Arc::new(self.clone());

        for &axis in axes.iter() {
            if axis >= self.rank() {
                return Err(TensorErrors::AxisOutOfBounds {
                    axis,
                    rank: self.rank(),
                });
            }
        }

        scope(|s| {
            for (i, chunk) in res.chunks_mut(1).enumerate() {
                let self_ref = self_arc.clone();

                s.spawn(move || {
                    let mut new_index = self.shape.tensor_index(i);

                    for &axis in axes {
                        new_index[axis] = self.shape[axis] - new_index[axis] - 1;
                    }

                    chunk[0] = self_ref[&new_index].clone();
                });
            }
        });

        Ok(res)
    }

    /// Flips a tensor
    pub fn flip_mt(&self) -> Tensor<T> {
        self.flip_axes_mt(&(0..self.rank()).collect()).unwrap()
    }

    /// Transposes a `Tensor` using a multithreaded implementation and returns the result
    pub fn transpose_mt(&self, transpose: &Transpose) -> Result<Tensor<T>, TensorErrors> {
        if transpose.permutation.len() != self.shape().rank() {
            return Err(TensorErrors::TransposeIncompatibleRank {
                rank: self.rank(),
                trank: transpose.permutation.len()
            });
        }

        let new_shape = transpose.new_shape(self.shape())?;
        let mut new_tensor = self.clone().reshape(&new_shape)?;

        let elems_per_thread = 20;  // Number of elements a single thread handles

        scope(|s| {
            for (i, elem) in new_tensor.chunks_mut(elems_per_thread).enumerate() {
                let new_shape_clone = new_shape.clone();

                s.spawn(move || {
                    for j in 0..elems_per_thread {
                        let k = i * elems_per_thread + j;
                        let new_index = new_shape_clone.tensor_index(k);
                        let old_index = transpose.old_index(&new_index).unwrap();

                        if old_index.iter().enumerate().any(|(i, x)| x >= &self.shape[i]) {
                            continue;
                        }

                        elem[j] = self[&old_index].clone();
                    }
                });
            }
        });

        Ok(new_tensor)
    }

    /// Pools a `Tensor<T>` into a `Tensor<O>` using a custom pooling function with the index.
    /// The custom function will take a `Tensor<T>` that corresponds to the slice that the kernel covers.
    /// If the kernel is hanging over the edge of the tensor, then only the bits of the tensor that fit are included.
    /// This is reflected in the shape of the input tensor.
    /// Default functions for `max` and `avg` are given as well.
    /// This is a multithreaded implementation.
    pub fn pool_indexed_mt<O: Clone + Send + Sync>(
        &self, 
        pool_fn: &(impl Fn(Vec<usize>, Tensor<T>) -> O + Sync),
        kernel_shape: &Shape,
        stride_shape: &Shape,
        init: O,
    ) -> Result<Tensor<O>, TensorErrors> {
        if kernel_shape.rank() != self.rank() {
            return Err(TensorErrors::RanksDoNotMatch(kernel_shape.rank(), self.rank()));
        }

        if stride_shape.rank() != self.rank() {
            return Err(TensorErrors::RanksDoNotMatch(stride_shape.rank(), self.rank()));
        }

        let res_shape = &Shape::new(
            self.shape()
                .0
                .iter()
                .cloned()
                .zip(stride_shape.0.iter().cloned())
                .map(|(x, y)| x.div_ceil(y))
                .collect::<Vec<usize>>(),
        )?;

        let mut result = Tensor::<O>::from_value(res_shape, init);

        scope(|s| {
            let res_chunks = result.chunks_mut(1);

            for (i, chunk) in res_chunks.enumerate() {
                s.spawn(move || {
                    let res_pos = res_shape.tensor_index(i);
                    let self_pos = res_pos.into_tensor() * stride_shape.clone().0.into_tensor();
                    let self_end_pos = (&self_pos + &kernel_shape.clone().0.into_tensor()).iter().enumerate().map(|(i, x)| if x > &self.shape()[i] { self.shape()[i] } else { *x }).collect::<Vec<usize>>();

                    let indices = self_pos
                        .iter()
                        .zip(self_end_pos.iter())
                        .map(|(x, y)| *x..*y)
                        .collect::<Vec<_>>();

                    chunk[0] = pool_fn(self_pos.elements, self.slice(&indices).unwrap());
                });
            }
        });

        Ok(result)
    }

    /// Pools a `Tensor<T>` into a `Tensor<O>` using a custom pooling function.
    /// The custom function will take a `Tensor<T>` that corresponds to the slice that the kernel covers.
    /// If the kernel is hanging over the edge of the tensor, then only the bits of the tensor that fit are included.
    /// This is reflected in the shape of the input tensor.
    /// Default functions for `max` and `avg` are given as well.
    /// This is a multithreaded implementation.
    pub fn pool_mt<O: Clone + Send + Sync>(
        &self,
        pool_fn: &(impl Fn(Tensor<T>) -> O + Sync), 
        kernel_shape: &Shape, 
        stride_shape: &Shape,
        init: O,
    ) -> Result<Tensor<O>, TensorErrors> {
        if kernel_shape.rank() != self.rank() {
            return Err(TensorErrors::RanksDoNotMatch(kernel_shape.rank(), self.rank()));
        }

        if stride_shape.rank() != self.rank() {
            return Err(TensorErrors::RanksDoNotMatch(stride_shape.rank(), self.rank()));
        }

        let res_shape = &Shape::new(
            self.shape()
                .0
                .iter()
                .cloned()
                .zip(stride_shape.0.iter().cloned())
                .map(|(x, y)| x.div_ceil(y))
                .collect::<Vec<usize>>(),
        )?;

        let mut result = Tensor::<O>::from_value(res_shape, init);

        scope(|s| {
            let res_chunks = result.chunks_mut(1);

            for (i, chunk) in res_chunks.enumerate() {
                s.spawn(move || {
                    let res_pos = res_shape.tensor_index(i);
                    let self_pos = res_pos.into_tensor() * stride_shape.clone().0.into_tensor();
                    let self_end_pos = (&self_pos + &kernel_shape.clone().0.into_tensor()).iter().enumerate().map(|(i, x)| if x > &self.shape()[i] { self.shape()[i] } else { *x }).collect::<Vec<usize>>();

                    let indices = self_pos
                        .iter()
                        .zip(self_end_pos.iter())
                        .map(|(x, y)| *x..*y)
                        .collect::<Vec<_>>();

                    chunk[0] = pool_fn(self.slice(&indices).unwrap());
                });
            }
        });

        Ok(result)
    }
}

impl<T: Default + Clone> Tensor<T> {
    pub fn from_shape(shape: &Shape) -> Tensor<T> {
        let elements = vec![T::default(); shape.element_count()];
        Tensor {
            elements,
            shape: shape.clone(),
            strides: Strides::from_shape(shape),
        }
    }
}

impl<T: Clone> IntoTensor<T> for Vec<T> {
    /// This converts the vector into a 1-d tensor
    fn into_tensor(self) -> Tensor<T> {
        self.iter().into()
    }
}

impl<T: Default + Clone> Default for Tensor<T> {
    fn default() -> Self {
        Tensor::from_shape(&shape![1])
    }
}

impl<T: Zero + Clone> Tensor<T> {
    pub fn zeros(shape: &Shape) -> Tensor<T> {
        Tensor::from_value(shape, T::zero())
    }
}

impl<T: PartialOrd + Clone> Tensor<T> {
    /// Bounds the values between `min` and `max`
    /// consuming the original and returning the result
    pub fn clip(&self, min: T, max: T) -> Tensor<T> {
        let shape = self.shape();
        Tensor::new(
            shape,
            self.iter()
                .cloned()
                .map(|x| {
                    if x < min {
                        min.clone()
                    } else if x > max {
                        max.clone()
                    } else {
                        x
                    }
                })
                .collect(),
        )
            .unwrap()
    }
}

/// Default pooling function to sum the values
pub fn pool_sum<T: Add<Output = T> + Clone>(t: Tensor<T>) -> T {
    t.iter().cloned().reduce(T::add).unwrap()
}

/// Default pooling function to find the maximum value
pub fn pool_max<T: PartialOrd + Clone>(t: Tensor<T>) -> T {
    let mut max = t.first().unwrap().clone();

    for i in t.iter() {
        if *i > max {
            max = i.clone();
        }
    }

    max
}

/// Default pooling function to find the minimum value
pub fn pool_min<T: PartialOrd + Clone>(t: Tensor<T>) -> T {
    let mut min = t.first().unwrap().clone();

    for i in t.iter() {
        if *i < min {
            min = i.clone();
        }
    }

    min
}

/// Default pooling function to find the average.
/// Bear in mind the total number of elements is the total number of elements in the input,
/// so if you want the total number of elements to stay the same even for overhanging
///  input tensors then you will need to write your own version.
pub fn pool_avg<T: Add<Output = T> + Div<f64, Output = T> + Clone>(t: Tensor<T>) -> T {
    let elems = t.shape().element_count().to_f64().unwrap();
    let sum = pool_sum(t);
    sum / elems
}
