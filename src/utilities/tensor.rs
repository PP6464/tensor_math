use crate::definitions::errors::TensorErrors;
use crate::definitions::shape::Shape;
use crate::definitions::strides::Strides;
use crate::definitions::tensor::Tensor;
use crate::definitions::tensor_slice::TensorSlice;
use crate::definitions::tensor_slice_mut::TensorSliceMut;
use crate::definitions::traits::IntoTensor;
use crate::definitions::transpose::Transpose;
use crate::shape;
use num::{ToPrimitive, Zero};
use rand::distr::{Distribution, StandardUniform};
use rand::RngExt;
use rayon::prelude::*;
use std::collections::HashSet;
use std::mem::MaybeUninit;
use std::ops::{Add, Div, Range};
/*
--------------------------------------------
* Tensor utility constructors
--------------------------------------------
*/

impl<T> Tensor<T> {
    /// Creates a tensor from a single value with specified shape.
    pub fn from_value(shape: Shape, value: T) -> Self
    where
        T: Clone,
    {
        let elements = vec![value; shape.element_count()];
        Tensor {
            strides: Strides::from_shape(&shape),
            shape,
            elements,
        }
    }

    /// Generate a tensor of the specified shape filled with random values.
    pub fn rand(shape: Shape) -> Tensor<T>
    where
        StandardUniform: Distribution<T>,
    {
        let mut elements = Vec::with_capacity(shape.element_count());
        let buf = elements.spare_capacity_mut();
        let mut rng = rand::rng();

        buf.iter_mut().for_each(|e| {
            e.write(rng.random());
        });

        unsafe {
            elements.set_len(shape.element_count());
        }

        Tensor {
            strides: Strides::from_shape(&shape),
            shape,
            elements,
        }
    }

    /// Constructs a tensor of the specified shape filled with `T::default()`.
    pub fn from_shape(shape: Shape) -> Tensor<T>
    where
        T: Default + Clone,
    {
        let elements = vec![T::default(); shape.element_count()];
        Tensor {
            strides: Strides::from_shape(&shape),
            elements,
            shape,
        }
    }

    /// Returns a matrix of the specified shape filled with `T::zero()`.
    pub fn zeros(shape: Shape) -> Tensor<T>
    where
        T: Zero + Clone,
    {
        Tensor::from_value(shape, T::zero())
    }
}

impl<T: Default> Default for Tensor<T> {
    /// Returns a tensor with shape (1) and a single element of `T::default()`.
    fn default() -> Self {
        Tensor {
            shape: shape![1],
            strides: Strides(vec![1]),
            elements: vec![T::default()],
        }
    }
}

/*
--------------------------------------------
* Basic tensor utility functions
--------------------------------------------
*/

impl<T> Tensor<T> {
    /// Reshapes the tensor.
    /// This fails if `new_shape.element_count() != self.shape().element_count()`.
    pub fn reshape(self, new_shape: Shape) -> Result<Tensor<T>, TensorErrors> {
        if new_shape.element_count() != self.shape.element_count() {
            return Err(TensorErrors::ShapeSizeDoesNotMatch);
        }

        Ok(Tensor {
            strides: Strides::from_shape(&new_shape),
            shape: new_shape,
            elements: self.elements,
        })
    }

    /// Reshapes the tensor without checking compatibility of shapes.
    pub(crate) unsafe fn reshape_unchecked(self, new_shape: Shape) -> Tensor<T> {
        Tensor {
            strides: Strides::from_shape(&new_shape),
            shape: new_shape,
            elements: self.elements,
        }
    }

    /// Flatten a tensor on a given dimension.
    /// This fails if the shape of the tensor at the given axis is not 1,
    /// or if the axis is out-of-bounds.
    pub fn flatten(self, axis: usize) -> Result<Tensor<T>, TensorErrors> {
        if axis >= self.rank() {
            return Err(TensorErrors::AxisOutOfBounds {
                axis,
                rank: self.rank(),
            });
        }

        if self.shape[axis] != 1 {
            return Err(TensorErrors::AxisIsNotOne(axis));
        }

        let mut copy = self.shape;
        copy.0.remove(axis);
        Ok(Tensor {
            strides: Strides::from_shape(&copy),
            shape: copy,
            elements: self.elements,
        })
    }

    /// Applies the given function over the entire tensor elementwise by consuming the elements.
    pub fn map<O>(self, closure: impl FnMut(T) -> O) -> Tensor<O> {
        let shape = self.shape();
        unsafe {
            self.into_iter()
                .map(closure)
                .collect::<Tensor<_>>()
                .reshape_unchecked(shape)
        }
    }

    /// Applies the given function over the entire tensor elementwise by consuming the elements
    pub fn par_map<O: Send>(self, closure: impl Fn(T) -> O + Sync + Send) -> Tensor<O>
    where
        T: Send + Sync,
    {
        self.into_par_iter().map(closure).collect()
    }

    /// Applies the given function over the entire tensor elementwise by reference.
    pub fn map_refs<O>(&self, closure: impl FnMut(&T) -> O) -> Tensor<O> {
        let shape = self.shape();
        unsafe {
            self.iter()
                .map(closure)
                .collect::<Tensor<_>>()
                .reshape_unchecked(shape)
        }
    }

    /// Applies the given function over the entire tensor elementwise by reference.
    pub fn par_map_refs<O: Send>(&self, closure: impl Fn(&T) -> O + Sync + Send) -> Tensor<O>
    where
        T: Send + Sync,
    {
        self.iter().map(closure).collect()
    }

    /// Returns an iterator that is enumerated with tensor indices.
    pub fn enumerated_iter(&self) -> impl Iterator<Item = (Vec<usize>, &T)> {
        let shape = self.shape();
        unsafe {
            self.iter()
                .enumerate()
                .map(move |(index, elem)| (shape.tensor_index_unchecked(index), elem))
        }
    }

    /// Returns a parallel iterator that is enumerated with tensor indices.
    pub fn enumerated_par_iter(&self) -> impl ParallelIterator<Item = (Vec<usize>, &T)> + '_
    where
        T: Send + Sync,
    {
        let shape = self.shape();
        unsafe {
            self.par_iter()
                .enumerate()
                .map(move |(index, elem)| (shape.tensor_index_unchecked(index), elem))
        }
    }

    /// Returns a mutable iterator that is enumerated with tensor indices.
    pub fn enumerated_iter_mut(&mut self) -> impl Iterator<Item = (Vec<usize>, &mut T)> + '_ {
        let shape = self.shape();
        unsafe {
            self.iter_mut()
                .enumerate()
                .map(move |(index, elem)| (shape.tensor_index_unchecked(index), elem))
        }
    }

    /// Returns a parallel mutable iterator that is enumerated with tensor indices.
    pub fn enumerated_par_iter_mut(
        &mut self,
    ) -> impl ParallelIterator<Item = (Vec<usize>, &mut T)> + '_
    where
        T: Send + Sync,
    {
        let shape = self.shape();
        unsafe {
            self.par_iter_mut()
                .enumerate()
                .map(move |(index, elem)| (shape.tensor_index_unchecked(index), elem))
        }
    }

    /// Sets all the values in the tensor to the given values.
    /// This fails if the shape of the values does not match the tensor's shape.
    pub fn set_all(&mut self, values: &Tensor<T>) -> Result<(), TensorErrors>
    where
        T: Clone,
    {
        if self.shape() != values.shape() {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: values.shape(),
                op: "set_all",
            });
        }

        self.elements.clone_from_slice(values);

        Ok(())
    }

    /// Sets all the values in the tensor to the given values.
    /// This fails if the shape of the values does not match the tensor's shape.
    pub fn set_all_from_slice(&mut self, values: &TensorSlice<T>) -> Result<(), TensorErrors>
    where
        T: Clone,
    {
        if self.shape() != values.shape() {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: self.shape(),
                shape_2: values.shape(),
                op: "set_all",
            });
        }

        for (index, value) in values.enumerated_iter() {
            unsafe { *self.get_unchecked_mut(&index) = value.clone() }
        }

        Ok(())
    }

    /// Returns a cloned immutable slice to a specified region in the tensor.
    /// This fails if `range.start > range.end` for any index range,
    /// or if the region includes an out-of-bounds index.
    pub fn slice(&self, indices: &[Range<usize>]) -> Result<TensorSlice<T>, TensorErrors> {
        if indices.len() != self.rank() {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: indices.iter().map(|r| r.end - r.start).collect(),
                shape_2: self.shape.clone(),
                op: "slice_mut",
            });
        }

        for (i, range) in indices.iter().enumerate() {
            if range.start > range.end {
                return Err(TensorErrors::InvalidInterval {
                    max: range.end as f64,
                    min: range.start as f64,
                });
            }

            if range.end > self.shape[i] {
                return Err(TensorErrors::SliceIndicesOutOfBounds {
                    start: range.start,
                    end: range.end,
                    length: self.shape[i],
                    axis: i,
                });
            }
        }

        let (start, end) = indices.iter().map(|range| (range.start, range.end)).unzip();

        Ok(TensorSlice {
            start,
            orig: self,
            end,
        })
    }

    /// Slices the tensor without checking bounds.
    pub(crate) unsafe fn slice_unchecked(&self, indices: &[Range<usize>]) -> TensorSlice<T> {
        let (start, end) = indices.iter().map(|range| (range.start, range.end)).unzip();

        TensorSlice {
            start,
            orig: self,
            end,
        }
    }

    /// Returns a slice covering the entire tensor.
    pub fn as_tensor_slice(&self) -> TensorSlice<T> {
        TensorSlice {
            orig: self,
            start: vec![0; self.rank()],
            end: self.shape.0.clone(),
        }
    }

    /// Returns a mutable slice of a specified region in the tensor.
    /// This fails if `range.start > range.end` for any index range,
    /// or if the region includes an out-of-bounds index.
    pub fn slice_mut(
        &'_ mut self,
        indices: &[Range<usize>],
    ) -> Result<TensorSliceMut<'_, T>, TensorErrors> {
        if indices.len() != self.rank() {
            return Err(TensorErrors::IncompatibleShapes {
                shape_1: indices.iter().map(|r| r.end - r.start).collect(),
                shape_2: self.shape.clone(),
                op: "slice_mut",
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

            if range.start > range.end {
                return Err(TensorErrors::InvalidInterval {
                    max: range.end as f64,
                    min: range.start as f64,
                });
            }
        }

        let start = indices
            .iter()
            .map(|range| range.start)
            .collect::<Vec<usize>>();
        let end = indices
            .iter()
            .map(|range| range.end)
            .collect::<Vec<usize>>();

        Ok(TensorSliceMut {
            start,
            end,
            orig: self,
        })
    }

    /// Slices the tensor mutably without checking bounds.
    pub(crate) fn slice_unchecked_mut(
        &'_ mut self,
        indices: &[Range<usize>],
    ) -> TensorSliceMut<'_, T> {
        let (start, end) = indices.iter().map(|range| (range.start, range.end)).unzip();

        TensorSliceMut {
            start,
            orig: self,
            end,
        }
    }

    /// Returns a mutable slice covering the entire tensor.
    pub fn as_tensor_slice_mut(&mut self) -> TensorSliceMut<T> {
        TensorSliceMut {
            start: vec![0; self.rank()],
            end: self.shape.0.clone(),
            orig: self,
        }
    }

    /// Flips the tensor along a single axis.
    /// This fails if the axis is out of bounds.
    pub fn flip_axis(mut self, axis: usize) -> Result<Tensor<T>, TensorErrors> {
        if axis >= self.rank() {
            return Err(TensorErrors::AxisOutOfBounds {
                axis,
                rank: self.rank(),
            });
        }

        let inner = self.strides[axis];
        let outer = self.shape[axis];

        if axis == self.rank() - 1 {
            self.chunks_mut(outer).for_each(|c| c.reverse());
            return Ok(self);
        }

        self.elements
            .chunks_mut(inner * outer)
            .for_each(|outer_chunk| unsafe {
                let mid = outer / 2;
                let (top, rest) = outer_chunk.split_at_mut_unchecked(mid * inner);
                let bottom = rest.get_unchecked_mut(rest.len() - mid * inner..);

                top.chunks_mut(inner)
                    .zip(bottom.chunks_mut(inner).rev())
                    .for_each(|(t, b)| {
                        t.swap_with_slice(b);
                    });
            });

        Ok(self)
    }

    /// Flips the tensor along a single axis without bounds checking the axis.
    pub(crate) unsafe fn flip_axis_unchecked(mut self, axis: usize) -> Tensor<T> {
        let inner = self.strides[axis];
        let outer = self.shape[axis];

        if axis == self.rank() - 1 {
            self.chunks_mut(outer).for_each(|c| c.reverse());
            return self;
        }

        self.elements
            .chunks_mut(inner * outer)
            .for_each(|outer_chunk| {
                let mid = outer / 2;
                let (top, rest) = outer_chunk.split_at_mut_unchecked(mid * inner);
                let bottom = rest.get_unchecked_mut(rest.len() - mid * inner..);

                top.chunks_mut(inner)
                    .zip(bottom.chunks_mut(inner).rev())
                    .for_each(|(t, b)| {
                        t.swap_with_slice(b);
                    });
            });

        self
    }

    /// Flips the tensor along a single axis.
    /// This fails if the axis is out of bounds.
    pub fn flip_axis_mt(mut self, axis: usize) -> Result<Tensor<T>, TensorErrors>
    where
        T: Send + Sync,
    {
        if axis >= self.rank() {
            return Err(TensorErrors::AxisOutOfBounds {
                axis,
                rank: self.rank(),
            });
        }

        let inner = self.strides[axis];
        let outer = self.shape[axis];

        self.elements
            .par_chunks_mut(inner * outer)
            .for_each(|outer_chunk| unsafe {
                let mid = outer / 2;
                let (top, rest) = outer_chunk.split_at_mut_unchecked(mid * inner);
                let bottom = rest.get_unchecked_mut(rest.len() - mid * inner..);

                top.par_chunks_mut(inner)
                    .zip(bottom.par_chunks_mut(inner).rev())
                    .for_each(|(t, b)| {
                        t.swap_with_slice(b);
                    });
            });

        Ok(self)
    }

    /// Flips the tensor along a single axis without bounds checking the axis.
    pub(crate) unsafe fn flip_axis_unchecked_mt(mut self, axis: usize) -> Tensor<T>
    where
        T: Send + Sync,
    {
        let inner = self.strides[axis];
        let outer = self.shape[axis];

        self.elements
            .par_chunks_mut(inner * outer)
            .for_each(|outer_chunk| {
                let mid = outer / 2;
                let (top, rest) = outer_chunk.split_at_mut_unchecked(mid * inner);
                let bottom = rest.get_unchecked_mut(rest.len() - mid * inner..);

                top.par_chunks_mut(inner)
                    .zip(bottom.par_chunks_mut(inner).rev())
                    .for_each(|(t, b)| {
                        t.swap_with_slice(b);
                    });
            });

        self
    }

    /// Flips the tensor along the specified axes.
    /// This fails if any of the axes are out of bounds.
    pub fn flip_axes(mut self, axes: HashSet<usize>) -> Result<Tensor<T>, TensorErrors> {
        for &axis in axes.iter() {
            self = self.flip_axis(axis)?;
        }

        Ok(self)
    }

    /// Flips the tensor along the specified axes without bounds checking the axes.
    pub(crate) unsafe fn flip_axes_unchecked(mut self, axes: HashSet<usize>) -> Tensor<T> {
        for &axis in axes.iter() {
            unsafe {
                self = self.flip_axis_unchecked(axis);
            }
        }

        self
    }

    /// Flips the tensor along the specified axes.
    /// This fails if any of the axes are out of bounds.
    pub fn flip_axes_mt(mut self, axes: HashSet<usize>) -> Result<Tensor<T>, TensorErrors>
    where
        T: Send + Sync,
    {
        let rank = self.rank();

        for &axis in axes.iter() {
            if axis >= rank {
                return Err(TensorErrors::AxisOutOfBounds { axis, rank });
            }

            unsafe {
                self = self.flip_axis_unchecked_mt(axis);
            }
        }

        Ok(self)
    }

    /// Flips the tensor along the specified axes without bounds checking the axes.
    pub(crate) unsafe fn flip_axes_unchecked_mt(mut self, axes: HashSet<usize>) -> Tensor<T>
    where
        T: Send + Sync,
    {
        for &axis in axes.iter() {
            unsafe {
                self = self.flip_axis_unchecked_mt(axis);
            }
        }

        self
    }

    /// Flips the tensor along all axes.
    pub fn flip(self) -> Tensor<T> {
        let rank = self.rank();
        unsafe { self.flip_axes_unchecked((0..rank).collect()) }
    }

    /// Flips the tensor along all axes.
    pub fn flip_mt(self) -> Tensor<T>
    where
        T: Send + Sync,
    {
        let rank = self.rank();
        unsafe { self.flip_axes_unchecked_mt((0..rank).collect()) }
    }

    /// Transposes a tensor and returns the result.
    /// This fails if `self.rank() != transpose.permutation().len()`.
    pub fn transpose(self, transpose: Transpose) -> Result<Tensor<T>, TensorErrors> {
        if transpose.permutation.len() != self.shape().rank() {
            return Err(TensorErrors::TransposeIncompatibleRank {
                rank: self.rank(),
                trank: transpose.permutation.len(),
            });
        }

        let mut new_elements = Vec::with_capacity(self.shape().element_count());
        let buf = new_elements.spare_capacity_mut();

        let shape = self.shape;

        self.elements
            .into_iter()
            .enumerate()
            .for_each(|(index, elem)| unsafe {
                let new_index = transpose.new_index_unchecked(shape.tensor_index_unchecked(index));
                let new_address = shape.address_unchecked(&new_index);
                buf.get_unchecked_mut(new_address).write(elem);
            });

        unsafe {
            new_elements.set_len(shape.element_count());
            let new_shape = transpose.new_shape_unchecked(shape);

            Ok(Tensor {
                strides: Strides::from_shape(&new_shape),
                shape: new_shape,
                elements: new_elements,
            })
        }
    }

    /// Transposes a tensor without checking the rank of the transpose.
    pub(crate) fn transpose_unchecked(self, transpose: Transpose) -> Tensor<T> {
        let mut new_elements = Vec::with_capacity(self.shape().element_count());
        let buf = new_elements.spare_capacity_mut();

        let shape = self.shape;

        self.elements
            .into_iter()
            .enumerate()
            .for_each(|(index, elem)| unsafe {
                let new_index = transpose.new_index_unchecked(shape.tensor_index_unchecked(index));
                let new_address = shape.address_unchecked(&new_index);
                buf.get_unchecked_mut(new_address).write(elem);
            });

        unsafe {
            new_elements.set_len(shape.element_count());
            let new_shape = transpose.new_shape_unchecked(shape);

            Tensor {
                strides: Strides::from_shape(&new_shape),
                shape: new_shape,
                elements: new_elements,
            }
        }
    }

    /// Transposes a tensor and returns the result.
    /// This fails if `self.rank() != transpose.permutation().len()`.
    pub fn transpose_mt(self, transpose: Transpose) -> Result<Tensor<T>, TensorErrors>
    where
        T: Send + Sync,
    {
        struct ThreadSafePtr<O>(*mut O);

        impl<O> ThreadSafePtr<O> {
            fn add(&self, offset: usize) -> ThreadSafePtr<O> {
                unsafe { ThreadSafePtr(self.0.add(offset)) }
            }

            fn write(&self, value: O) {
                unsafe { self.0.write(value) }
            }
        }

        unsafe impl<O> Sync for ThreadSafePtr<O> {}
        unsafe impl<O> Send for ThreadSafePtr<O> {}

        if transpose.permutation.len() != self.shape().rank() {
            return Err(TensorErrors::TransposeIncompatibleRank {
                rank: self.rank(),
                trank: transpose.permutation.len(),
            });
        }

        let mut new_elements = Vec::with_capacity(self.shape().element_count());
        let buf = new_elements.spare_capacity_mut();
        let buf_ptr = ThreadSafePtr(buf.as_mut_ptr());

        let shape = self.shape;

        self.elements
            .into_par_iter()
            .enumerate()
            .for_each(|(index, elem)| unsafe {
                let new_index = transpose.new_index_unchecked(shape.tensor_index_unchecked(index));
                let new_address = shape.address_unchecked(&new_index);
                buf_ptr.add(new_address).write(MaybeUninit::new(elem));
            });

        unsafe {
            new_elements.set_len(shape.element_count());
            let new_shape = transpose.new_shape_unchecked(shape);

            Ok(Tensor {
                strides: Strides::from_shape(&new_shape),
                shape: new_shape,
                elements: new_elements,
            })
        }
    }

    /// Transposes a tensor and returns the result, without checking the rank of the transpose.
    pub(crate) fn transpose_unchecked_mt(self, transpose: Transpose) -> Tensor<T>
    where
        T: Send + Sync,
    {
        struct ThreadSafePtr<O>(*mut O);

        impl<O> ThreadSafePtr<O> {
            fn add(&self, offset: usize) -> ThreadSafePtr<O> {
                unsafe { ThreadSafePtr(self.0.add(offset)) }
            }

            fn write(&self, value: O) {
                unsafe { self.0.write(value) }
            }
        }

        unsafe impl<O> Sync for ThreadSafePtr<O> {}
        unsafe impl<O> Send for ThreadSafePtr<O> {}

        let mut new_elements = Vec::with_capacity(self.shape().element_count());
        let buf = new_elements.spare_capacity_mut();
        let buf_ptr = ThreadSafePtr(buf.as_mut_ptr());

        let shape = self.shape;

        self.elements
            .into_par_iter()
            .enumerate()
            .for_each(|(index, elem)| unsafe {
                let new_index = transpose.new_index_unchecked(shape.tensor_index_unchecked(index));
                let new_address = shape.address_unchecked(&new_index);
                buf_ptr.add(new_address).write(MaybeUninit::new(elem));
            });

        unsafe {
            new_elements.set_len(shape.element_count());
            let new_shape = transpose.new_shape_unchecked(shape);

            Tensor {
                strides: Strides::from_shape(&new_shape),
                shape: new_shape,
                elements: new_elements,
            }
        }
    }

    /// Concatenates a tensor with another tensor along the specified axis.
    /// This fails if `self.shape()[i] != other.shape()[i]` for all `i` that is not `axis`,
    /// or if the ranks do not match.
    pub fn concat(self, other: Tensor<T>, axis: usize) -> Result<Tensor<T>, TensorErrors> {
        if self.rank() != other.rank() {
            return Err(TensorErrors::RanksDoNotMatch(self.rank(), other.rank()));
        }

        let mut resultant_shape: Vec<usize> = Vec::with_capacity(self.rank());

        if axis >= self.rank() {
            return Err(TensorErrors::AxisOutOfBounds {
                axis,
                rank: self.rank(),
            });
        }

        for i in 0..self.rank() {
            if i == axis {
                resultant_shape.push(self.shape[i] + other.shape[i]);
                continue;
            }

            if self.shape[i] != other.shape[i] {
                return Err(TensorErrors::IncompatibleShapes {
                    shape_1: self.shape.clone(),
                    shape_2: other.shape.clone(),
                    op: "concat",
                });
            }

            resultant_shape.push(self.shape[i]);
        }

        let resultant_shape: Shape = resultant_shape.into();
        let mut new_elements =
            Vec::with_capacity(self.shape.element_count() + other.shape.element_count());

        let mut self_iter = self.elements.into_iter();
        let mut other_iter = other.elements.into_iter();

        for _ in 0..self.shape.element_count() / self.strides[axis] {
            new_elements.extend(self_iter.by_ref().take(self.strides[axis]));
            new_elements.extend(other_iter.by_ref().take(other.strides[axis]));
        }

        Ok(Tensor {
            strides: Strides::from_shape(&resultant_shape),
            shape: resultant_shape,
            elements: new_elements,
        })
    }

    /// Concatenates a tensor with another tensor along the specified axis without validation.
    pub(crate) fn concat_unchecked(self, other: Tensor<T>, axis: usize) -> Tensor<T> {
        let mut resultant_shape: Vec<usize> = self.shape.0.clone();
        resultant_shape[axis] += other.shape[axis];

        let resultant_shape: Shape = resultant_shape.into();
        let mut new_elements =
            Vec::with_capacity(self.shape.element_count() + other.shape.element_count());

        let mut self_iter = self.elements.into_iter();
        let mut other_iter = other.elements.into_iter();

        for _ in 0..self.shape.element_count() / self.strides[axis] {
            new_elements.extend(self_iter.by_ref().take(self.strides[axis]));
            new_elements.extend(other_iter.by_ref().take(other.strides[axis]));
        }

        Tensor {
            strides: Strides::from_shape(&resultant_shape),
            shape: resultant_shape,
            elements: new_elements,
        }
    }

    /// Bounds the values between `min` and `max`.
    pub fn clip(self, min: T, max: T) -> Tensor<T>
    where
        T: Clone + PartialOrd,
    {
        self.map(|val| {
            if val < min {
                min.clone()
            } else if val > max {
                max.clone()
            } else {
                val
            }
        })
    }

    /// Bounds the values between `min` and `max`
    pub fn par_clip(self, min: T, max: T) -> Tensor<T>
    where
        T: Clone + Send + Sync + PartialOrd,
    {
        self.par_map(|val| {
            if val < min {
                min.clone()
            } else if val > max {
                max.clone()
            } else {
                val
            }
        })
    }
}

impl<T> Tensor<T> {
    /// Pools a `Tensor<T>` into a `Tensor<O>` using a custom pooling function.
    /// The custom function will take a `TensorSlice<T>` that corresponds to the slice
    /// that the kernel covers. If the kernel is hanging over the edge of the tensor,
    /// then only the bit of the tensor that fits is included.
    /// This fails if the kernel shape or stride shape contains 0, or if either of their ranks
    /// do not match the rank of the input tensor, or if the tensor is empty or has rank 0.
    pub fn pool<O>(
        &self,
        pool_fn: impl Fn(TensorSlice<T>) -> O,
        kernel_shape: &Shape,
        stride_shape: &Shape,
    ) -> Result<Tensor<O>, TensorErrors> {
        if self.rank() == 0 {
            return Err(TensorErrors::RankZero { op: "Pooling" });
        }

        if self.elements.is_empty() {
            return Err(TensorErrors::TensorEmpty { op: "Pooling" });
        }

        if kernel_shape.rank() != self.rank() {
            return Err(TensorErrors::RanksDoNotMatch(
                kernel_shape.rank(),
                self.rank(),
            ));
        }

        if stride_shape.rank() != self.rank() {
            return Err(TensorErrors::RanksDoNotMatch(
                stride_shape.rank(),
                self.rank(),
            ));
        }

        if kernel_shape.0.contains(&0) || stride_shape.0.contains(&0) {
            return Err(TensorErrors::ShapeContainsZero);
        }

        let res_shape: Shape = self
            .shape()
            .0
            .iter()
            .cloned()
            .zip(stride_shape.0.iter().cloned())
            .map(|(x, y)| x.div_ceil(y))
            .collect();

        let mut res = Vec::with_capacity(res_shape.element_count());
        unsafe {
            res.set_len(res_shape.element_count());
        }
        let mut res = Tensor {
            strides: Strides::from_shape(&res_shape),
            shape: res_shape,
            elements: res,
        };

        for (pos, val) in res.enumerated_iter_mut() {
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

            unsafe {
                *val = pool_fn(self.slice_unchecked(&indices));
            }
        }

        Ok(res)
    }

    /// Pools a `Tensor<T>` into a `Tensor<O>` using a custom pooling function with the index.
    /// The custom function will take a `Tensor<T>` that corresponds to the slice
    /// that the kernel covers. If the kernel is hanging over the edge of the tensor,
    /// then only the bit of the tensor that fits is included.
    /// This fails if the kernel shape or stride shape contains 0, or if either of their ranks
    /// do not match the rank of the input tensor, or if the tensor is empty or has rank 0.
    pub fn pool_indexed<O: Clone>(
        &self,
        pool_fn: impl Fn(Vec<usize>, TensorSlice<T>) -> O,
        kernel_shape: &Shape,
        stride_shape: &Shape,
    ) -> Result<Tensor<O>, TensorErrors> {
        if self.rank() == 0 {
            return Err(TensorErrors::RankZero { op: "Pooling" });
        }

        if self.elements.is_empty() {
            return Err(TensorErrors::TensorEmpty { op: "Pooling" });
        }

        if kernel_shape.rank() != self.rank() {
            return Err(TensorErrors::RanksDoNotMatch(
                kernel_shape.rank(),
                self.rank(),
            ));
        }

        if stride_shape.rank() != self.rank() {
            return Err(TensorErrors::RanksDoNotMatch(
                stride_shape.rank(),
                self.rank(),
            ));
        }

        if kernel_shape.0.contains(&0) || stride_shape.0.contains(&0) {
            return Err(TensorErrors::ShapeContainsZero);
        }

        let res_shape: Shape = self
            .shape()
            .0
            .iter()
            .cloned()
            .zip(stride_shape.0.iter().cloned())
            .map(|(x, y)| x.div_ceil(y))
            .collect();

        let mut res = Vec::with_capacity(res_shape.element_count());
        unsafe {
            res.set_len(res_shape.element_count());
        }
        let mut res = Tensor {
            strides: Strides::from_shape(&res_shape),
            shape: res_shape,
            elements: res,
        };

        for (pos, val) in res.enumerated_iter_mut() {
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

            unsafe {
                *val = pool_fn(start_pos, self.slice_unchecked(&indices));
            }
        }

        Ok(res)
    }

    /// Pools a `Tensor<T>` into a `Tensor<O>` using a custom pooling function.
    /// The custom function will take a `TensorSlice<T>` that corresponds to the slice
    /// that the kernel covers. If the kernel is hanging over the edge of the tensor,
    /// then only the bit of the tensor that fits is included.
    /// As this is multithreaded, a reference to the pooling function is expected.
    /// This fails if the kernel shape or stride shape contains 0, or if either of their ranks
    /// do not match the rank of the input tensor, or if the tensor is empty or has rank 0.
    pub fn pool_mt<O: Send + Sync>(
        &self,
        pool_fn: &(impl Fn(TensorSlice<T>) -> O + Sync),
        kernel_shape: &Shape,
        stride_shape: &Shape,
    ) -> Result<Tensor<O>, TensorErrors>
    where
        T: Send + Sync,
    {
        if self.rank() == 0 {
            return Err(TensorErrors::RankZero { op: "Pooling" });
        }

        if self.elements.is_empty() {
            return Err(TensorErrors::TensorEmpty { op: "Pooling" });
        }

        if kernel_shape.rank() != self.rank() {
            return Err(TensorErrors::RanksDoNotMatch(
                kernel_shape.rank(),
                self.rank(),
            ));
        }

        if stride_shape.rank() != self.rank() {
            return Err(TensorErrors::RanksDoNotMatch(
                stride_shape.rank(),
                self.rank(),
            ));
        }

        if kernel_shape.0.contains(&0) || stride_shape.0.contains(&0) {
            return Err(TensorErrors::ShapeContainsZero);
        }

        let res_shape: Shape = self
            .shape()
            .0
            .iter()
            .cloned()
            .zip(stride_shape.0.iter().cloned())
            .map(|(x, y)| x.div_ceil(y))
            .collect();

        let mut res = Vec::with_capacity(res_shape.element_count());
        unsafe {
            res.set_len(res_shape.element_count());
        }
        let mut res = Tensor {
            strides: Strides::from_shape(&res_shape),
            shape: res_shape,
            elements: res,
        };

        res.enumerated_par_iter_mut().for_each(|(index, elem)| {
            let self_pos = index.clone().into_tensor() * stride_shape.clone().0.into_tensor();
            let self_end_pos = (&self_pos + &kernel_shape.clone().0.into_tensor())
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    if x > &self.shape()[i] {
                        self.shape()[i]
                    } else {
                        *x
                    }
                })
                .collect::<Vec<_>>();

            let indices = self_pos
                .iter()
                .zip(self_end_pos.iter())
                .map(|(x, y)| *x..*y)
                .collect::<Vec<_>>();

            unsafe {
                *elem = pool_fn(self.slice_unchecked(&indices));
            }
        });

        Ok(res)
    }

    /// Pools a `Tensor<T>` into a `Tensor<O>` using a custom pooling function with the index.
    /// The custom function will take a `TensorSlice<T>` that corresponds to the slice
    /// that the kernel covers. If the kernel is hanging over the edge of the tensor,
    /// then only the bit of the tensor that fits is included.
    /// As this is multithreaded, a reference to the pooling function is expected.
    /// This fails if the kernel shape or stride shape contains 0, or if either of their ranks
    /// do not match the rank of the input tensor, or if the tensor is empty or has rank 0.
    pub fn pool_indexed_mt<O: Send + Sync>(
        &self,
        pool_fn: &(impl Fn(Vec<usize>, TensorSlice<T>) -> O + Sync),
        kernel_shape: &Shape,
        stride_shape: &Shape,
    ) -> Result<Tensor<O>, TensorErrors>
    where
        T: Send + Sync,
    {
        if self.rank() == 0 {
            return Err(TensorErrors::RankZero { op: "Pooling" });
        }

        if self.elements.is_empty() {
            return Err(TensorErrors::TensorEmpty { op: "Pooling" });
        }

        if kernel_shape.rank() != self.rank() {
            return Err(TensorErrors::RanksDoNotMatch(
                kernel_shape.rank(),
                self.rank(),
            ));
        }

        if stride_shape.rank() != self.rank() {
            return Err(TensorErrors::RanksDoNotMatch(
                stride_shape.rank(),
                self.rank(),
            ));
        }

        if kernel_shape.0.contains(&0) || stride_shape.0.contains(&0) {
            return Err(TensorErrors::ShapeContainsZero);
        }

        let res_shape: Shape = self
            .shape()
            .0
            .iter()
            .cloned()
            .zip(stride_shape.0.iter().cloned())
            .map(|(x, y)| x.div_ceil(y))
            .collect();

        let mut res = Vec::with_capacity(res_shape.element_count());
        unsafe {
            res.set_len(res_shape.element_count());
        }
        let mut res = Tensor {
            strides: Strides::from_shape(&res_shape),
            shape: res_shape,
            elements: res,
        };

        res.enumerated_par_iter_mut().for_each(|(index, elem)| {
            let self_pos = index.clone().into_tensor() * stride_shape.clone().0.into_tensor();
            let self_end_pos = (&self_pos + &kernel_shape.clone().0.into_tensor())
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    if x > &self.shape()[i] {
                        self.shape()[i]
                    } else {
                        *x
                    }
                })
                .collect::<Vec<usize>>();

            let indices = self_pos
                .iter()
                .zip(self_end_pos.iter())
                .map(|(x, y)| *x..*y)
                .collect::<Vec<_>>();

            unsafe {
                *elem = pool_fn(index, self.slice_unchecked(&indices));
            }
        });

        Ok(res)
    }
}

/// Default pooling function to sum the values.
pub fn pool_sum<T: Add<Output = T> + Clone>(t: TensorSlice<T>) -> Option<T> {
    t.iter().cloned().reduce(T::add)
}

/// Default pooling function to find the maximum value.
pub fn pool_max<T: PartialOrd + Clone>(t: TensorSlice<T>) -> Option<T> {
    t.iter().reduce(|a, b| if a > b { a } else { b }).cloned()
}

/// Default pooling function to find the minimum value.
pub fn pool_min<T: PartialOrd + Clone>(t: TensorSlice<T>) -> Option<T> {
    t.iter().reduce(|a, b| if a < b { a } else { b }).cloned()
}

/// Default pooling function to find the average.
/// The total number of elements is the total number of elements in the input, so this may
/// vary if the kernel is hanging over the edge of the tensor.
pub fn pool_avg<T: Add<Output = T> + Div<f64, Output = T> + Clone>(t: TensorSlice<T>) -> Option<T> {
    let elems = t.shape().element_count().to_f64();
    let sum = pool_sum(t);
    match (sum, elems) {
        (Some(s), Some(e)) => Some(s / e),
        _ => None,
    }
}
