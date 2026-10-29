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

        Ok(unsafe { self.reshape_unchecked(new_shape) })
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
        let shape = self.shape();
        unsafe {
            self.into_par_iter()
                .map(closure)
                .collect::<Tensor<_>>()
                .reshape_unchecked(shape)
        }
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
        let shape = self.shape();
        unsafe {
            self.elements
                .par_iter()
                .map(closure)
                .collect::<Tensor<_>>()
                .reshape_unchecked(shape)
        }
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
    pub fn slice(&self, indices: &[Range<usize>]) -> Result<TensorSlice<'_, T>, TensorErrors> {
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

        Ok(unsafe { self.slice_unchecked(indices) })
    }

    /// Slices the tensor without checking bounds.
    pub(crate) unsafe fn slice_unchecked(&self, indices: &[Range<usize>]) -> TensorSlice<'_, T> {
        let (start, end) = indices.iter().map(|range| (range.start, range.end)).unzip();

        TensorSlice {
            start,
            orig: self,
            end,
        }
    }

    /// Returns a slice covering the entire tensor.
    pub fn as_tensor_slice(&self) -> TensorSlice<'_, T> {
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

        Ok(unsafe { self.slice_unchecked_mut(indices) })
    }

    /// Slices the tensor mutably without checking bounds.
    pub(crate) unsafe fn slice_unchecked_mut(
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
    pub fn as_tensor_slice_mut(&mut self) -> TensorSliceMut<'_, T> {
        TensorSliceMut {
            start: vec![0; self.rank()],
            end: self.shape.0.clone(),
            orig: self,
        }
    }

    /// Flips the tensor along a single axis.
    /// This fails if the axis is out of bounds.
    pub fn flip_axis(self, axis: usize) -> Result<Tensor<T>, TensorErrors> {
        if axis >= self.rank() {
            return Err(TensorErrors::AxisOutOfBounds {
                axis,
                rank: self.rank(),
            });
        }

        Ok(unsafe { self.flip_axis_unchecked(axis) })
    }

    /// Flips the tensor along a single axis without bounds checking the axis.
    pub(crate) unsafe fn flip_axis_unchecked(mut self, axis: usize) -> Tensor<T> {
        if self.elements.is_empty() {
            return self;
        }

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
    pub fn flip_axis_mt(self, axis: usize) -> Result<Tensor<T>, TensorErrors>
    where
        T: Send + Sync,
    {
        if axis >= self.rank() {
            return Err(TensorErrors::AxisOutOfBounds {
                axis,
                rank: self.rank(),
            });
        }

        Ok(unsafe { self.flip_axis_unchecked_mt(axis) })
    }

    /// Flips the tensor along a single axis without bounds checking the axis.
    pub(crate) unsafe fn flip_axis_unchecked_mt(mut self, axis: usize) -> Tensor<T>
    where
        T: Send + Sync,
    {
        if self.elements.is_empty() {
            return self;
        }

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
    pub fn flip_axes(self, axes: &HashSet<usize>) -> Result<Tensor<T>, TensorErrors> {
        let rank = self.rank();
        for &axis in axes.iter() {
            if axis >= rank {
                return Err(TensorErrors::AxisOutOfBounds { axis, rank });
            }
        }

        Ok(unsafe { self.flip_axes_unchecked(axes) })
    }

    /// Flips the tensor along the specified axes without bounds checking the axes.
    pub(crate) unsafe fn flip_axes_unchecked(mut self, axes: &HashSet<usize>) -> Tensor<T> {
        for &axis in axes.iter() {
            unsafe {
                self = self.flip_axis_unchecked(axis);
            }
        }

        self
    }

    /// Flips the tensor along the specified axes.
    /// This fails if any of the axes are out of bounds.
    pub fn flip_axes_mt(self, axes: &HashSet<usize>) -> Result<Tensor<T>, TensorErrors>
    where
        T: Send + Sync,
    {
        let rank = self.rank();

        for &axis in axes.iter() {
            if axis >= rank {
                return Err(TensorErrors::AxisOutOfBounds { axis, rank });
            }
        }

        Ok(unsafe { self.flip_axes_unchecked_mt(axes) })
    }

    /// Flips the tensor along the specified axes without bounds checking the axes.
    pub(crate) unsafe fn flip_axes_unchecked_mt(mut self, axes: &HashSet<usize>) -> Tensor<T>
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
        unsafe { self.flip_axes_unchecked(&(0..rank).collect()) }
    }

    /// Flips the tensor along all axes.
    pub fn flip_mt(self) -> Tensor<T>
    where
        T: Send + Sync,
    {
        let rank = self.rank();
        unsafe { self.flip_axes_unchecked_mt(&(0..rank).collect()) }
    }

    /// Transposes a tensor and returns the result.
    /// This fails if `self.rank() != transpose.permutation().len()`.
    pub fn transpose(self, transpose: &Transpose) -> Result<Tensor<T>, TensorErrors> {
        if transpose.permutation.len() != self.shape().rank() {
            return Err(TensorErrors::TransposeIncompatibleRank {
                rank: self.rank(),
                trank: transpose.permutation.len(),
            });
        }

        Ok(unsafe { self.transpose_unchecked(transpose) })
    }

    /// Transposes a tensor without checking the rank of the transpose.
    pub(crate) unsafe fn transpose_unchecked(self, transpose: &Transpose) -> Tensor<T> {
        let mut new_elements = Vec::with_capacity(self.shape().element_count());
        let buf = new_elements.spare_capacity_mut();

        let old_shape = self.shape;
        let new_shape = transpose.new_shape_unchecked(old_shape.clone());

        self.elements
            .into_iter()
            .enumerate()
            .for_each(|(index, elem)| unsafe {
                let old_index = old_shape.tensor_index_unchecked(index);
                let new_index = transpose.new_index_unchecked(old_index);
                let new_address = new_shape.address_unchecked(&new_index);
                buf.get_unchecked_mut(new_address).write(elem);
            });

        unsafe {
            new_elements.set_len(new_shape.element_count());

            Tensor {
                strides: Strides::from_shape(&new_shape),
                shape: new_shape,
                elements: new_elements,
            }
        }
    }

    /// Transposes a tensor and returns the result.
    /// This fails if `self.rank() != transpose.permutation().len()`.
    pub fn transpose_mt(self, transpose: &Transpose) -> Result<Tensor<T>, TensorErrors>
    where
        T: Send + Sync,
    {
        if transpose.permutation.len() != self.shape().rank() {
            return Err(TensorErrors::TransposeIncompatibleRank {
                rank: self.rank(),
                trank: transpose.permutation.len(),
            });
        }

        Ok(unsafe { self.transpose_unchecked_mt(transpose) })
    }

    /// Transposes a tensor and returns the result, without checking the rank of the transpose.
    pub(crate) unsafe fn transpose_unchecked_mt(self, transpose: &Transpose) -> Tensor<T>
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

        let old_shape = self.shape;
        let new_shape = transpose.new_shape_unchecked(old_shape.clone());

        self.elements
            .into_par_iter()
            .enumerate()
            .for_each(|(index, elem)| unsafe {
                let old_index = old_shape.tensor_index_unchecked(index);
                let new_index = transpose.new_index_unchecked(old_index);
                let new_address = new_shape.address_unchecked(&new_index);
                buf_ptr.add(new_address).write(MaybeUninit::new(elem));
            });

        unsafe {
            new_elements.set_len(new_shape.element_count());

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

        Ok(unsafe { self.concat_unchecked(other, axis) })
    }

    /// Concatenates a tensor with another tensor along the specified axis without validation.
    pub(crate) unsafe fn concat_unchecked(self, other: Tensor<T>, axis: usize) -> Tensor<T> {
        let mut resultant_shape: Vec<usize> = self.shape.0.clone();
        resultant_shape[axis] += other.shape[axis];

        let resultant_shape: Shape = resultant_shape.into();
        let mut new_elements =
            Vec::with_capacity(self.shape.element_count() + other.shape.element_count());

        if resultant_shape.element_count() == 0 {
            return Tensor {
                strides: Strides::from_shape(&resultant_shape),
                shape: resultant_shape,
                elements: new_elements,
            };
        }

        let inner = self.strides[axis];
        let outer = self.shape.0[..axis].iter().product::<usize>();
        let mut self_iter = self.elements.into_iter();
        let mut other_iter = other.elements.into_iter();

        for _ in 0..outer {
            for _ in 0..self.shape[axis] {
                new_elements.extend(self_iter.by_ref().take(inner));
            }
            for _ in 0..other.shape[axis] {
                new_elements.extend(other_iter.by_ref().take(inner));
            }
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
        // We will initialise this vector here so we can disable the error
        #[allow(clippy::uninit_vec)]
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
        // We will initialise this vector here so we can disable the error
        #[allow(clippy::uninit_vec)]
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
        // We will initialise this vector here so we can disable the error
        #[allow(clippy::uninit_vec)]
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
        // We will initialise this vector here so we can disable the error
        #[allow(clippy::uninit_vec)]
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

#[cfg(test)]
mod tensor_utils_tests {
    use rayon::prelude::*;
    use std::collections::HashSet;
    use std::f64::consts::PI;
    use tensor_math::definitions::errors::TensorErrors;
    use tensor_math::definitions::tensor::Tensor;
    use tensor_math::definitions::transpose::Transpose;
    use tensor_math::utilities::tensor::{pool_avg, pool_max, pool_min, pool_sum};
    use tensor_math::{shape, transpose};

    #[test]
    fn concat() {
        let t1 = Tensor::<i32>::from_shape(shape![4, 2, 3]);
        let t2 = Tensor::<i32>::from_value(shape![4, 1, 3], 1);
        let ans1 = Tensor::new(
            shape![4, 3, 3],
            vec![
                0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
                0, 0, 0, 0, 0, 1, 1, 1,
            ],
        )
        .unwrap();
        assert_eq!(ans1, t1.clone().concat(t2.clone(), 1).unwrap());

        let t3 = Tensor::<i32>::from_shape(shape![2, 3]);
        let t4 = Tensor::<i32>::from_value(shape![1, 3], -1);
        let ans2 = Tensor::new(shape![3, 3], vec![0, 0, 0, 0, 0, 0, -1, -1, -1]).unwrap();
        assert_eq!(ans2, t3.clone().concat(t4.clone(), 0).unwrap());
    }

    #[test]
    fn invalid_concat() {
        let t1 = Tensor::<i32>::from_shape(shape![4, 2, 3]);
        let t2 = Tensor::<i32>::from_shape(shape![3, 1, 2]);

        assert!(matches!(
            t1.clone().concat(t2.clone(), 0).unwrap_err(),
            TensorErrors::IncompatibleShapes { .. }
        ));
    }

    #[test]
    fn reshape_correctly() {
        let t1 = Tensor::<i32>::from_shape(shape![2, 3, 4]);
        let t1 = t1
            .reshape(shape![4, 6])
            .expect("Was a valid reshape but failed");

        assert_eq!(t1.shape(), shape![4, 6]);
    }

    #[test]
    fn invalid_reshape() {
        Tensor::<i32>::from_shape(shape![2, 3, 4])
            .reshape(shape![1, 1, 1, 1, 1, 12])
            .expect_err("Should've panicked");
    }

    #[test]
    fn flatten_correctly() {
        let t1 = Tensor::<i32>::from_shape(shape![2, 3, 1, 4, 1])
            .flatten(2)
            .expect("Valid flatten but failed");
        let t1 = t1.clone().flatten(3).expect("Valid flatten but failed");
        assert_eq!(t1.shape(), shape![2, 3, 4]);
    }

    #[test]
    fn invalid_flatten_dim_out_of_bounds() {
        let t1 = Tensor::<i32>::from_shape(shape![2, 3, 4]);
        let error = t1.clone().flatten(5).err().unwrap();

        assert!(matches!(
            error,
            TensorErrors::AxisOutOfBounds { axis: _, rank: _ }
        ));
    }

    #[test]
    fn invalid_flatten_dim_not_one() {
        let t1 = Tensor::<i32>::from_shape(shape![2, 3, 4]);
        let error = t1.clone().flatten(1).err().unwrap();

        assert!(matches!(error, TensorErrors::AxisIsNotOne(_)));
    }

    #[test]
    fn iterator() {
        let shape = shape![2, 3];
        let t1 = Tensor::<i32>::from_shape(shape.clone());
        let iter1 = t1.clone().into_iter();
        let iter2 = t1.clone().into_iter();

        let mut count = 0;
        for x in iter1 {
            assert_eq!(x, 0);
            count += 1;
        }

        assert_eq!(count, shape.element_count());

        let t2: Tensor<i32> = iter2.collect();
        assert_eq!(t2.shape(), shape![shape.element_count()]);
        assert_eq!(t2.elements(), t1.elements());
        let t2 = t2
            .clone()
            .reshape(shape.clone())
            .expect("Was a valid reshape but failed");
        assert_eq!(t2, t1);

        let shape2 = shape![5, 2];
        let iter3 = vec![0; shape2.element_count()];
        let t3: Tensor<i32> = iter3.iter().copied().collect();
        assert_eq!(t3, Tensor::from_value(shape![shape2.element_count()], 0));
        let t3 = t3
            .clone()
            .reshape(shape2.clone())
            .expect("Was a valid reshape but failed");
        assert_eq!(t3, Tensor::from_value(shape2.clone(), 0));
    }

    #[test]
    fn random() {
        Tensor::<i32>::rand(shape![2, 3]);
    }

    #[test]
    fn transform_elementwise() {
        let t1 = Tensor::<f64>::new(
            shape![2, 3],
            vec![
                0.0,
                PI / 6.0,
                PI / 3.0,
                PI / 2.0,
                2.0 * PI / 3.0,
                5.0 * PI / 6.0,
            ],
        )
        .unwrap();
        let transformed = t1.clone().map(f64::cos);
        let ans = Tensor::<f64>::new(
            shape![2, 3],
            vec![
                1.0,
                f64::sqrt(3.0) / 2.0,
                0.5,
                0.0,
                -0.5,
                -f64::sqrt(3.0) / 2.0,
            ],
        )
        .unwrap();
        assert!((ans - transformed).into_iter().map(f64::abs).sum::<f64>() < 1e-6);
    }

    #[test]
    fn slicing() {
        let t1 = Tensor::<i32>::new(shape![3, 3, 3], (0..27).collect()).unwrap();

        let sliced = t1.slice(&[0..3, 1..2, 1..3]).unwrap();
        let ans = Tensor::<i32>::new(shape![3, 1, 2], vec![4, 5, 13, 14, 22, 23]).unwrap();

        assert_eq!(sliced.clone_into_tensor(), ans);
    }

    #[test]
    #[should_panic]
    fn invalid_slice_out_of_bounds() {
        let t1 = Tensor::<i32>::from_shape(shape![2, 3]);
        t1.slice(&[1..5, 0..3]).unwrap();
    }

    #[test]
    #[should_panic]
    fn invalid_slice_incorrect_rank() {
        let t1 = Tensor::<i32>::from_shape(shape![2, 3]);
        t1.slice(&[]).unwrap();
    }

    #[test]
    fn slicing_mut() {
        let mut t1 = Tensor::<i32>::new(shape![3, 3, 3], (0..27).collect()).unwrap();
        let mut slice = t1.slice_mut(&[1..2, 1..2, 1..2]).unwrap();
        slice[&[0, 0, 0]] = 69;
        let ans = Tensor::<i32>::new(
            shape![3, 3, 3],
            (0..27).map(|x| if x != 13 { x } else { 69 }).collect(),
        )
        .unwrap();
        assert_eq!(t1, ans);
    }

    #[test]
    fn set_all() {
        let mut t1 = Tensor::<i32>::new(shape![3, 3, 3], (0..27).collect()).unwrap();
        let mut slice_mut = t1.slice_mut(&[0..2, 1..2, 0..2]).unwrap();
        let inserted = Tensor::<i32>::new(shape![2, 1, 2], (0..4).collect()).unwrap();

        slice_mut.set_all(&inserted).unwrap();
        let ans = Tensor::<i32>::new(
            shape![3, 3, 3],
            vec![
                0, 1, 2, 0, 1, 5, 6, 7, 8, 9, 10, 11, 2, 3, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26,
            ],
        )
        .unwrap();

        assert_eq!(t1, ans);
    }

    #[test]
    #[should_panic]
    fn set_all_fail() {
        let mut t1 = Tensor::<i32>::new(shape![3, 3, 3], (0..27).collect()).unwrap();
        let mut slice_mut = t1.slice_mut(&[0..2, 0..2, 0..2]).unwrap();

        slice_mut.set_all(&Tensor::<i32>::zeros(shape![1])).unwrap();
    }

    #[test]
    #[should_panic]
    fn slice_mut_out_of_bounds() {
        let mut t1 = Tensor::<i32>::from_shape(shape![2, 3]);
        let mut slice = t1.slice_mut(&[0..1, 0..1]).unwrap();

        slice[&[0, 1]] = 69;
    }

    #[test]
    fn concat_multithreaded() {
        let t1 = Tensor::<i32>::from_shape(shape![20, 30]);
        let t2 = Tensor::<i32>::from_value(shape![20, 20], 2);

        let ans = t1.clone().concat(t2.clone(), 1).unwrap();
        let mt_ans = t1.clone().concat(t2.clone(), 1).unwrap();

        assert_eq!(ans, mt_ans);
    }

    #[test]
    fn transpose() {
        let t1 = Tensor::<i32>::new(shape![2, 3, 4], (0..24).collect()).unwrap();
        let t2 = Tensor::<i32>::new(shape![2, 3], (0..6).collect()).unwrap();
        let transposed_t1 = t1.clone().transpose(&transpose![0, 2, 1]).unwrap();
        let t2 = t2
            .clone()
            .transpose(&Transpose::new(&vec![1, 0]).unwrap())
            .unwrap();
        let transposed_t2 = t2.clone();
        let ans1 = Tensor::<i32>::new(
            shape![2, 4, 3],
            vec![
                0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19,
                23,
            ],
        )
        .unwrap();
        let ans2 = Tensor::<i32>::new(shape![3, 2], vec![0, 3, 1, 4, 2, 5]).unwrap();

        assert_eq!(ans1, transposed_t1);
        assert_eq!(ans2, transposed_t2);
    }

    #[test]
    fn transpose_mt() {
        let t1 = Tensor::<i32>::rand(shape![10, 60, 5]);
        let transpose = transpose![1, 2, 0];

        let ans = t1.clone().transpose(&transpose).unwrap();
        let mt_ans = t1.clone().transpose_mt(&transpose).unwrap();

        assert_eq!(mt_ans, ans);
    }

    #[test]
    fn clipping() {
        let t1 = Tensor::<i32>::new(shape![3, 3, 3], (0..27).collect()).unwrap();
        let t1 = t1.clone().clip(5, 10);
        let ans = Tensor::<i32>::new(
            shape![3, 3, 3],
            vec![
                5, 5, 5, 5, 5, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                10, 10, 10, 10,
            ],
        )
        .unwrap();

        assert_eq!(t1, ans);
    }

    #[test]
    fn pool() {
        let t1: Tensor<f64> = Tensor::<i32>::new(
            shape![3, 3, 3],
            vec![
                1, 5, -1, 2, 3, -5, 12, 10, -10, 1, -4, 2, 9, 6, 8, -1, 0, -1, -8, 7, 4, 5, 1, 2,
                -5, 3, 1,
            ],
        )
        .unwrap()
        .map(|x| x.into());

        let avg_pool = t1
            .pool(pool_avg, &shape![2, 2, 2], &shape![2, 2, 2])
            .unwrap();
        let sum_pool = t1
            .pool(pool_sum, &shape![2, 2, 2], &shape![2, 2, 2])
            .unwrap();
        let max_pool = t1
            .pool(pool_max, &shape![2, 2, 2], &shape![1, 1, 1])
            .unwrap();
        let min_pool = t1
            .pool(pool_min, &shape![3, 1, 1], &shape![3, 1, 1])
            .unwrap();

        let sum_ans = Tensor::<f64>::new(
            shape![2, 2, 2],
            vec![23.0, 4.0, 21.0, -11.0, 5.0, 6.0, -2.0, 1.0],
        )
        .unwrap()
        .map(Some);
        let avg_ans = Tensor::<f64>::new(
            shape![2, 2, 2],
            vec![2.875, 1.0, 5.25, -5.5, 1.25, 3.0, -1.0, 1.0],
        )
        .unwrap()
        .map(Some);
        let max_ans = Tensor::<f64>::new(
            shape![3, 3, 3],
            vec![
                9.0, 8.0, 8.0, 12.0, 10.0, 8.0, 12.0, 10.0, -1.0, 9.0, 8.0, 8.0, 9.0, 8.0, 8.0,
                3.0, 3.0, 1.0, 7.0, 7.0, 4.0, 5.0, 3.0, 2.0, 3.0, 3.0, 1.0,
            ],
        )
        .unwrap()
        .map(Some);
        let min_ans = Tensor::<f64>::new(
            shape![1, 3, 3],
            vec![-8.0, -4.0, -1.0, 2.0, 1.0, -5.0, -5.0, 0.0, -10.0],
        )
        .unwrap()
        .map(Some);

        assert_eq!(avg_pool, avg_ans);
        assert_eq!(sum_pool, sum_ans);
        assert_eq!(max_pool, max_ans);
        assert_eq!(min_pool, min_ans);
    }

    #[test]
    fn pool_mt() {
        let t1 = Tensor::<i32>::rand(shape![200, 10, 10]);

        let ans = t1
            .pool(pool_min, &shape![10, 30, 1], &shape![10, 10, 10])
            .unwrap();
        let mt_ans = t1
            .pool_mt(&pool_min, &shape![10, 30, 1], &shape![10, 10, 10])
            .unwrap();

        assert_eq!(ans, mt_ans);
    }

    #[test]
    fn invalid_flip_axes_mt() {
        let t1 = Tensor::<i32>::new(shape![1, 2, 3, 4], vec![1; 24]).unwrap();

        let mut axes = HashSet::new();
        axes.insert(4);

        let err = t1.clone().flip_axes_mt(&axes).unwrap_err();
        match err {
            TensorErrors::AxisOutOfBounds { axis: _, rank: _ } => {}
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn flip() {
        let t1 = Tensor::<i32>::new(shape![2, 3, 4], (0..24).collect()).unwrap();
        let mut axes = HashSet::new();
        axes.insert(0);
        axes.insert(2);

        let res_axes = t1.clone().flip_axes(&axes).unwrap();
        let res = t1.clone().flip();

        let ans_axes = Tensor::<i32>::new(
            shape![2, 3, 4],
            vec![
                15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9,
                8,
            ],
        )
        .unwrap();
        let ans = Tensor::<i32>::new(
            shape![2, 3, 4],
            vec![
                23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
                0,
            ],
        )
        .unwrap();

        assert_eq!(res_axes, ans_axes);
        assert_eq!(res, ans);
    }

    #[test]
    fn flip_mt() {
        let t1 = Tensor::<i32>::rand(shape![10, 20, 10]);
        let mut axes = HashSet::new();
        axes.insert(0);
        axes.insert(1);

        let res_axes = t1.clone().flip_axes(&axes).unwrap();
        let res_axes_mt = t1.clone().flip_axes_mt(&axes).unwrap();

        let res = t1.clone().flip();
        let res_mt = t1.clone().flip_mt();

        assert_eq!(res_axes, res_axes_mt);
        assert_eq!(res, res_mt);
    }

    #[test]
    fn invalid_flip_axes() {
        let t1 = Tensor::<i32>::new(shape![1, 2, 3, 4], vec![1; 24]).unwrap();

        let mut axes = HashSet::new();
        axes.insert(4);

        let err = t1.clone().flip_axes(&axes).unwrap_err();
        match err {
            TensorErrors::AxisOutOfBounds { axis: _, rank: _ } => {}
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn transform_refs_elementwise() {
        let t1 = Tensor::new(shape![2, 3, 4], (0..24).collect()).unwrap();
        let ans = Tensor::new(shape![2, 3, 4], (1..25).collect()).unwrap();

        assert_eq!(t1.clone().map(|x| x + &1), ans);
    }

    #[test]
    fn invalid_concat_diff_ranks() {
        let t1 = Tensor::<i32>::new(shape![1, 2, 3, 4], (0..24).collect()).unwrap();
        let t2 = Tensor::<i32>::new(shape![2, 3, 4], (0..24).collect()).unwrap();

        let err = t1.clone().concat(t2.clone(), 0).unwrap_err();
        match err {
            TensorErrors::RanksDoNotMatch(..) => {}
            _ => panic!("Incorrect error"),
        }

        let err = t1.clone().concat(t2.clone(), 0).unwrap_err();
        match err {
            TensorErrors::RanksDoNotMatch(..) => {}
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn pooling_invalid_shape_rank() {
        let t1 = Tensor::<f64>::new(shape![2, 4, 3], (0..24).map(f64::from).collect()).unwrap();

        let err = t1
            .pool(pool_avg, &shape![1, 1, 1], &shape![1, 1])
            .unwrap_err();
        match err {
            TensorErrors::RanksDoNotMatch(..) => {}
            _ => panic!("Incorrect error"),
        }

        let err = t1
            .pool(pool_avg, &shape![1, 1], &shape![1, 1, 1])
            .unwrap_err();
        match err {
            TensorErrors::RanksDoNotMatch(..) => {}
            _ => panic!("Incorrect error"),
        }

        let err = t1
            .pool_indexed(|_, m| pool_max(m), &shape![1, 1], &shape![1, 1, 1])
            .unwrap_err();
        match err {
            TensorErrors::RanksDoNotMatch(..) => {}
            _ => panic!("Incorrect error"),
        }

        let err = t1
            .pool_indexed(|_, m| pool_max(m), &shape![1, 1, 1], &shape![1, 1])
            .unwrap_err();
        match err {
            TensorErrors::RanksDoNotMatch(..) => {}
            _ => panic!("Incorrect error"),
        }

        let err = t1
            .pool_mt(&pool_min, &shape![1, 1], &shape![1, 1, 1])
            .unwrap_err();
        match err {
            TensorErrors::RanksDoNotMatch(..) => {}
            _ => panic!("Incorrect error"),
        }

        let err = t1
            .pool_mt(&pool_min, &shape![1, 1, 1], &shape![1, 1])
            .unwrap_err();
        match err {
            TensorErrors::RanksDoNotMatch(..) => {}
            _ => panic!("Incorrect error"),
        }

        let err = t1
            .pool_indexed_mt(&|_, m| pool_sum(m), &shape![1, 1, 1], &shape![1, 1])
            .unwrap_err();
        match err {
            TensorErrors::RanksDoNotMatch(..) => {}
            _ => panic!("Incorrect error"),
        }

        let err = t1
            .pool_indexed_mt(&|_, m| pool_sum(m), &shape![1, 1], &shape![1, 1, 1])
            .unwrap_err();
        match err {
            TensorErrors::RanksDoNotMatch(..) => {}
            _ => panic!("Incorrect error"),
        }

        let scalar = Tensor::<f64>::new(shape![], vec![1.0]).unwrap();

        let err = scalar.pool(pool_avg, &shape![], &shape![]).unwrap_err();
        match err {
            TensorErrors::RankZero { .. } => {}
            _ => panic!("Incorrect error for scalar pooling"),
        }

        let err = scalar
            .pool_indexed(|_, m| pool_avg(m), &shape![], &shape![])
            .unwrap_err();
        match err {
            TensorErrors::RankZero { .. } => {}
            _ => panic!("Incorrect error for scalar pooling"),
        }

        let err = scalar.pool_mt(&pool_sum, &shape![], &shape![]).unwrap_err();
        match err {
            TensorErrors::RankZero { .. } => {}
            _ => panic!("Incorrect error for scalar pooling"),
        }

        let err = scalar
            .pool_indexed_mt(&|_, m| pool_avg(m), &shape![], &shape![])
            .unwrap_err();
        match err {
            TensorErrors::RankZero { .. } => {}
            _ => panic!("Incorrect error for scalar pooling"),
        }

        let empty = Tensor::<f64>::new(shape![0, 3], vec![]).unwrap();

        let err = empty
            .clone()
            .pool(pool_avg, &shape![], &shape![])
            .unwrap_err();
        match err {
            TensorErrors::TensorEmpty { .. } => {}
            _ => panic!("Incorrect error for scalar pooling"),
        }

        let err = empty
            .pool_indexed(|_, m| pool_avg(m), &shape![], &shape![])
            .unwrap_err();
        match err {
            TensorErrors::TensorEmpty { .. } => {}
            _ => panic!("Incorrect error for scalar pooling"),
        }

        let err = empty.pool_mt(&pool_sum, &shape![], &shape![]).unwrap_err();
        match err {
            TensorErrors::TensorEmpty { .. } => {}
            _ => panic!("Incorrect error for scalar pooling"),
        }

        let err = empty
            .pool_indexed_mt(&|_, m| pool_avg(m), &shape![], &shape![])
            .unwrap_err();
        match err {
            TensorErrors::TensorEmpty { .. } => {}
            _ => panic!("Incorrect error for scalar pooling"),
        }
    }

    #[test]
    fn invalid_transpose_rank() {
        let t1 = Tensor::new(shape![2, 3, 4], (0..24).collect()).unwrap();

        let err = t1.clone().transpose(&Transpose::identity(1)).unwrap_err();
        match err {
            TensorErrors::TransposeIncompatibleRank { .. } => {}
            _ => panic!("Incorrect error"),
        }

        let err = t1
            .clone()
            .transpose_mt(&Transpose::identity(4))
            .unwrap_err();
        match err {
            TensorErrors::TransposeIncompatibleRank { .. } => {}
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn invalid_slicing_incorrect_ranges() {
        let mut t1 = Tensor::<i32>::new(shape![1, 2, 3, 4], (0..24).collect()).unwrap();

        let err = t1.slice(&[0..1, 0..1, 1..0, 0..1]).unwrap_err();
        match err {
            TensorErrors::InvalidInterval { .. } => {}
            _ => panic!("Incorrect error"),
        }

        let err = t1.slice_mut(&[0..1, 0..1, 1..0, 0..1]).unwrap_err();
        match err {
            TensorErrors::InvalidInterval { .. } => {}
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn invalid_slicing_num_indices_wrong() {
        let mut t1 = Tensor::<i32>::new(shape![1, 2, 3, 4], (0..24).collect()).unwrap();

        let err = t1.slice(&[0..1]).unwrap_err();
        match err {
            TensorErrors::IncompatibleShapes { .. } => {}
            _ => panic!("Incorrect error"),
        }

        let err = t1.slice_mut(&[0..1]).unwrap_err();
        match err {
            TensorErrors::IncompatibleShapes { .. } => {}
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn slice_mut_invalid_indices_out_of_bounds() {
        let mut t1 = Tensor::<i32>::new(shape![1, 2, 3, 4], (0..24).collect()).unwrap();

        let err = t1.slice_mut(&[0..1, 0..1, 2..5, 0..3]).unwrap_err();
        match err {
            TensorErrors::SliceIndicesOutOfBounds { .. } => {}
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn slicing_scalar_tensor() {
        let t1 = Tensor::<i32>::new(shape![], vec![42]).unwrap();
        let sliced = t1.slice(&[]).unwrap();
        assert_eq!(sliced.clone_into_tensor(), t1);
        assert_eq!(sliced.iter().copied().collect::<Vec<_>>(), [42]);
    }

    #[test]
    fn slicing_empty_tensor() {
        let t1 = Tensor::<i32>::new(shape![0, 3], vec![]).unwrap();
        let sliced = t1.slice(&[0..0, 1..2]).unwrap();
        let ans = Tensor::<i32>::new(shape![0, 1], vec![]).unwrap();

        assert_eq!(sliced.clone_into_tensor(), ans);
        assert_eq!(sliced.iter().count(), 0);
    }

    #[test]
    fn cannot_flatten_scalar_tensor() {
        let t1 = Tensor::<i32>::new(shape![], vec![42]).unwrap();
        let error = t1.clone().flatten(0).err().unwrap();

        match error {
            TensorErrors::AxisOutOfBounds { .. } => {}
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn flip_scalar_tensor() {
        let t1 = Tensor::new(shape![], vec![42]).unwrap();

        assert_eq!(t1.clone().flip(), t1);
        assert_eq!(t1.clone().flip_mt(), t1);

        let axes = HashSet::new();
        assert_eq!(t1.clone().flip_axes(&axes).unwrap(), t1);
        assert_eq!(t1.clone().flip_axes_mt(&axes).unwrap(), t1);

        let mut invalid_axes = HashSet::new();
        invalid_axes.insert(0);
        let err = t1.clone().flip_axes(&invalid_axes).unwrap_err();
        match err {
            TensorErrors::AxisOutOfBounds { .. } => {}
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn collect_empty_tensor() {
        let empty_vec: Vec<i32> = vec![];
        let t: Tensor<i32> = empty_vec.into_iter().collect();

        assert_eq!(t.rank(), 1);
        assert_eq!(t.shape()[0], 0);
        assert!(t.elements().is_empty());
    }

    #[test]
    fn enumerated_iter() {
        let mut t1 = Tensor::<i32>::new(shape![2, 2], vec![1, 2, 3, 4]).unwrap();

        let expected = vec![
            (vec![0, 0], 1),
            (vec![0, 1], 2),
            (vec![1, 0], 3),
            (vec![1, 1], 4),
        ];

        let actual: Vec<(Vec<usize>, i32)> = t1
            .enumerated_iter()
            .map(|(idx, value)| (idx, *value))
            .collect();
        assert_eq!(actual, expected);

        for (idx, val) in t1.enumerated_iter_mut() {
            *val += idx.iter().sum::<usize>() as i32;
        }

        let expected_mut = vec![
            (vec![0, 0], 1), // 1 + 0
            (vec![0, 1], 3), // 2 + 1
            (vec![1, 0], 4), // 3 + 1
            (vec![1, 1], 6), // 4 + 2
        ];
        let actual_mut: Vec<(Vec<usize>, i32)> =
            t1.enumerated_iter().map(|(i, v)| (i, *v)).collect();
        assert_eq!(actual_mut, expected_mut);
    }

    #[test]
    fn enumerated_iter_rank_0() {
        let mut t1 = Tensor::<i32>::new(shape![], vec![42]).unwrap();

        let expected = vec![(vec![], 42)];
        let actual: Vec<(Vec<usize>, i32)> = t1
            .enumerated_iter()
            .map(|(idx, value)| (idx, *value))
            .collect();
        assert_eq!(actual, expected);

        for (idx, val) in t1.enumerated_iter_mut() {
            assert!(idx.is_empty());
            *val += 1;
        }
        assert_eq!(t1.elements()[0], 43);
    }

    #[test]
    fn enumerated_iter_empty() {
        let mut t1 = Tensor::<i32>::new(shape![0, 2], vec![]).unwrap();

        assert_eq!(t1.enumerated_iter().count(), 0);
        for (_, _) in t1.enumerated_iter_mut() {
            panic!("Should not iterate over empty tensor");
        }
    }

    #[test]
    fn enumerated_par_iter_and_mut() {
        let mut t1 = Tensor::<i32>::new(shape![2, 2], vec![1, 2, 3, 4]).unwrap();

        let expected = vec![
            (vec![0, 0], 1),
            (vec![0, 1], 2),
            (vec![1, 0], 3),
            (vec![1, 1], 4),
        ];

        let actual: Vec<(Vec<usize>, i32)> = t1
            .enumerated_par_iter()
            .map(|(idx, value)| (idx, *value))
            .collect();
        assert_eq!(actual, expected);

        t1.enumerated_par_iter_mut().for_each(|(idx, val)| {
            *val += idx.iter().sum::<usize>() as i32;
        });

        let expected_mut = vec![
            (vec![0, 0], 1),
            (vec![0, 1], 3),
            (vec![1, 0], 4),
            (vec![1, 1], 6),
        ];
        let actual_mut: Vec<(Vec<usize>, i32)> =
            t1.enumerated_par_iter().map(|(i, v)| (i, *v)).collect();
        assert_eq!(actual_mut, expected_mut);
    }

    #[test]
    fn enumerated_par_iter_rank_0_and_mut() {
        let mut t1 = Tensor::<i32>::new(shape![], vec![42]).unwrap();

        let expected = vec![(vec![], 42)];
        let actual: Vec<(Vec<usize>, i32)> = t1
            .enumerated_par_iter()
            .map(|(idx, value)| (idx, *value))
            .collect();
        assert_eq!(actual, expected);

        t1.enumerated_par_iter_mut().for_each(|(idx, val)| {
            assert!(idx.is_empty());
            *val += 1;
        });
        assert_eq!(t1.elements()[0], 43);
    }

    #[test]
    fn enumerated_par_iter_empty_and_mut() {
        let mut t1 = Tensor::<i32>::new(shape![0, 2], vec![]).unwrap();

        assert_eq!(t1.enumerated_par_iter().count(), 0);
        t1.enumerated_par_iter_mut().for_each(|(_, _)| {
            panic!("Should not iterate over empty tensor");
        });
    }

    #[test]
    fn concat_scalar_tensors() {
        let t1 = Tensor::new(shape![], vec![1]).unwrap();
        let t2 = Tensor::new(shape![], vec![2]).unwrap();

        let err = t1.clone().concat(t2.clone(), 0).unwrap_err();
        match err {
            TensorErrors::AxisOutOfBounds { .. } => {}
            _ => panic!("Incorrect error"),
        }
    }

    #[test]
    fn concat_with_empty_tensor() {
        let t1 = Tensor::<i32>::from_shape(shape![2, 3]);
        let t2 = Tensor::<i32>::new(shape![0, 3], vec![]).unwrap();

        let res = t1.clone().concat(t2.clone(), 0).unwrap();
        assert_eq!(res.shape(), shape![2, 3]);
        assert_eq!(res.elements(), t1.elements());

        let t3 = Tensor::<i32>::new(shape![2, 0], vec![]).unwrap();
        let res2 = t1.clone().concat(t3.clone(), 1).unwrap();
        assert_eq!(res2.shape(), shape![2, 3]);
        assert_eq!(res2.elements(), t1.elements());

        // Test concatenating into an empty tensor
        let res3 = t2.clone().concat(t1.clone(), 0).unwrap();
        assert_eq!(res3.shape(), shape![2, 3]);
        assert_eq!(res3.elements(), t1.elements());
    }

    #[test]
    fn concat_multithreaded_with_empty_tensor() {
        let t1 = Tensor::<i32>::from_shape(shape![100, 10]);
        let t2 = Tensor::<i32>::new(shape![0, 10], vec![]).unwrap();

        let res = t1.clone().concat(t2.clone(), 0).unwrap();
        assert_eq!(res.shape(), shape![100, 10]);
        assert_eq!(res.elements(), t1.elements());

        let res2 = t2.clone().concat(t1.clone(), 0).unwrap();
        assert_eq!(res2.shape(), shape![100, 10]);
        assert_eq!(res2.elements(), t1.elements());
    }

    #[test]
    fn constructors_maps_and_reshape_cover_scalar_and_empty_shapes() {
        let scalar = Tensor::<i32>::from_value(shape![], 7);
        assert_eq!(scalar.shape(), shape![]);
        assert_eq!(scalar.elements(), &[7]);
        assert_eq!(Tensor::<i32>::from_shape(shape![]).elements(), &[0]);
        assert_eq!(Tensor::<i32>::zeros(shape![0, 3]).elements(), &[]);
        assert_eq!(Tensor::<i32>::default().shape(), shape![1]);

        assert_eq!(
            scalar.clone().map(|value| value * 2),
            Tensor::new(shape![], vec![14]).unwrap()
        );
        assert_eq!(
            scalar.clone().par_map(|value| value * 2),
            Tensor::new(shape![], vec![14]).unwrap()
        );
        assert_eq!(
            scalar.map_refs(|value| value + 1),
            Tensor::new(shape![], vec![8]).unwrap()
        );
        assert_eq!(
            scalar.par_map_refs(|value| value + 1),
            Tensor::new(shape![], vec![8]).unwrap()
        );

        let empty = Tensor::<i32>::zeros(shape![2, 0, 3]);
        assert_eq!(
            empty.clone().map(|value| value + 1).shape(),
            shape![2, 0, 3]
        );
        assert_eq!(empty.clone().par_map(|value| value + 1), empty);
        assert_eq!(empty.map_refs(|value| value + 1), empty);
        assert_eq!(empty.par_map_refs(|value| value + 1), empty);
        assert_eq!(
            Tensor::new(shape![], vec![9])
                .unwrap()
                .reshape(shape![1])
                .unwrap(),
            Tensor::new(shape![1], vec![9]).unwrap()
        );
    }

    #[test]
    fn zero_sized_tensor_set_all_flip_transpose_and_clip_preserve_shape() {
        let mut empty = Tensor::<i32>::new(shape![2, 0, 3], vec![]).unwrap();
        empty
            .set_all(&Tensor::new(shape![2, 0, 3], vec![]).unwrap())
            .unwrap();
        let source = Tensor::<i32>::new(shape![2, 0, 3], vec![]).unwrap();
        empty.set_all_from_slice(&source.as_tensor_slice()).unwrap();
        assert_eq!(empty.clone().flip_axis(1).unwrap(), empty);
        assert_eq!(empty.clone().flip_axis_mt(1).unwrap(), empty);
        assert_eq!(
            empty
                .clone()
                .transpose(&transpose![2, 1, 0])
                .unwrap()
                .shape(),
            shape![3, 0, 2]
        );
        assert_eq!(
            empty
                .clone()
                .transpose_mt(&transpose![2, 1, 0])
                .unwrap()
                .shape(),
            shape![3, 0, 2]
        );
        assert_eq!(empty.clone().clip(-1, 1), empty);
        assert_eq!(empty.clone().par_clip(-1, 1), empty);

        let mut scalar = Tensor::new(shape![], vec![1]).unwrap();
        scalar
            .set_all(&Tensor::new(shape![], vec![2]).unwrap())
            .unwrap();
        assert_eq!(scalar.elements(), &[2]);
        let scalar_source = Tensor::new(shape![], vec![3]).unwrap();
        scalar
            .set_all_from_slice(&scalar_source.as_tensor_slice())
            .unwrap();
        assert_eq!(scalar.elements(), &[3]);
        assert_eq!(scalar.clone().clip(0, 2).elements(), &[2]);
    }

    #[test]
    fn valid_set_all() {
        let mut t1 = Tensor::<i32>::new(shape![2, 2], vec![1, 2, 3, 4]).unwrap();
        let t2 = Tensor::<i32>::new(shape![2, 2], vec![5, 6, 7, 8]).unwrap();
        t1.set_all(&t2).unwrap();
        let ans = Tensor::<i32>::new(shape![2, 2], vec![5, 6, 7, 8]).unwrap();
        assert_eq!(t1, ans);
    }

    #[test]
    fn invalid_set_all() {
        let mut t1 = Tensor::<i32>::new(shape![2, 2], vec![1, 2, 3, 4]).unwrap();
        let t2 = Tensor::<i32>::new(shape![3, 3], vec![5, 6, 7, 8, 9, 10, 11, 12, 13]).unwrap();
        let err = t1.set_all(&t2).unwrap_err();
        assert!(matches!(err, TensorErrors::IncompatibleShapes { .. }));
    }

    #[test]
    fn set_all_from_slice() {
        let mut t1 = Tensor::<i32>::new(shape![2, 2], vec![1, 2, 3, 4]).unwrap();
        let source = Tensor::<i32>::new(shape![2, 2], vec![5, 6, 7, 8]).unwrap();
        t1.set_all_from_slice(&source.as_tensor_slice()).unwrap();
        let ans = Tensor::<i32>::new(shape![2, 2], vec![5, 6, 7, 8]).unwrap();
        assert_eq!(t1, ans);
    }

    #[test]
    fn invalid_set_all_from_slice() {
        let mut t1 = Tensor::<i32>::new(shape![2, 2], vec![1, 2, 3, 4]).unwrap();
        let source = Tensor::<i32>::new(shape![3, 3], vec![5, 6, 7, 8, 9, 10, 11, 12, 13]).unwrap();
        let err = t1
            .set_all_from_slice(&source.as_tensor_slice())
            .unwrap_err();
        assert!(matches!(err, TensorErrors::IncompatibleShapes { .. }));
    }

    #[test]
    fn as_tensor_slice_and_slice_mut() {
        let mut t1 = Tensor::<i32>::new(shape![2, 2], vec![1, 2, 3, 4]).unwrap();
        let slice = t1.as_tensor_slice();
        assert_eq!(slice.shape(), shape![2, 2]);
        assert_eq!(slice.iter().copied().collect::<Vec<_>>(), vec![1, 2, 3, 4]);

        let mut slice_mut = t1.as_tensor_slice_mut();
        assert_eq!(slice_mut.shape(), shape![2, 2]);
        assert_eq!(
            slice_mut.iter().copied().collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );

        // Modify the mutable slice and check if the original tensor is updated
        slice_mut[&[0, 0]] = 10;
        assert_eq!(t1.elements(), &[10, 2, 3, 4]);
    }

    #[test]
    fn flip_invalid_axis_out_of_bounds() {
        let t1 = Tensor::<i32>::new(shape![2, 2], vec![1, 2, 3, 4]).unwrap();
        let err = t1.clone().flip_axis(2).unwrap_err();
        assert!(matches!(err, TensorErrors::AxisOutOfBounds { .. }));

        let err_mt = t1.flip_axis_mt(2).unwrap_err();
        assert!(matches!(err_mt, TensorErrors::AxisOutOfBounds { .. }));
    }

    #[test]
    fn concat_empty() {
        let t1 = Tensor::<i32>::new(shape![0, 2], vec![]).unwrap();
        let t2 = Tensor::<i32>::new(shape![0, 2], vec![]).unwrap();

        let res1 = t1.clone().concat(t2.clone(), 0).unwrap();
        let res2 = t1.clone().concat(t2.clone(), 1).unwrap();
        assert_eq!(res1.shape(), shape![0, 2]);
        assert_eq!(res1.elements(), &[]);
        assert_eq!(res2.shape(), shape![0, 4]);
        assert_eq!(res2.elements(), &[]);
    }

    #[test]
    fn par_clip() {
        let t1 = Tensor::<i32>::new(shape![3, 3], vec![1, 5, -1, 2, 3, -5, 12, 10, -10]).unwrap();
        let t1_clipped = t1.clone().par_clip(0, 10);
        let ans = Tensor::<i32>::new(shape![3, 3], vec![1, 5, 0, 2, 3, 0, 10, 10, 0]).unwrap();

        assert_eq!(t1_clipped, ans);
    }

    #[test]
    fn all_pools_are_same() {
        let t1 = Tensor::<i32>::new(
            shape![3, 3, 3],
            vec![
                1, 5, -1, 2, 3, -5, 12, 10, -10, 1, -4, 2, 9, 6, 8, -1, 0, -1, -8, 7, 4, 5, 1, 2,
                -5, 3, 1,
            ],
        )
        .unwrap();

        let normal = t1
            .pool(pool_max, &shape![2, 2, 2], &shape![2, 2, 2])
            .unwrap();
        let normal_mt = t1
            .pool_mt(&pool_max, &shape![2, 2, 2], &shape![2, 2, 2])
            .unwrap();
        let indexed = t1
            .pool_indexed(|_, m| pool_max(m), &shape![2, 2, 2], &shape![2, 2, 2])
            .unwrap();
        let indexed_mt = t1
            .pool_indexed_mt(&|_, m| pool_max(m), &shape![2, 2, 2], &shape![2, 2, 2])
            .unwrap();

        assert_eq!(normal, normal_mt);
        assert_eq!(normal, indexed);
        assert_eq!(normal, indexed_mt);
    }

    #[test]
    fn pool_kernel_shape_zero() {
        let t1 = Tensor::<i32>::new(
            shape![3, 3, 3],
            vec![
                1, 5, -1, 2, 3, -5, 12, 10, -10, 1, -4, 2, 9, 6, 8, -1, 0, -1, -8, 7, 4, 5, 1, 2,
                -5, 3, 1,
            ],
        )
        .unwrap();

        let err = t1
            .pool(pool_max, &shape![0, 2, 2], &shape![2, 2, 2])
            .unwrap_err();
        assert!(matches!(err, TensorErrors::ShapeContainsZero { .. }));

        let err_mt = t1
            .pool_mt(&pool_max, &shape![0, 2, 2], &shape![2, 2, 2])
            .unwrap_err();
        assert!(matches!(err_mt, TensorErrors::ShapeContainsZero { .. }));

        let err_indexed = t1
            .pool_indexed(|_, t| pool_max(t), &shape![0, 2, 2], &shape![2, 2, 2])
            .unwrap_err();
        assert!(matches!(err_indexed, TensorErrors::ShapeContainsZero { .. }));

        let err_indexed_mt = t1
            .pool_indexed_mt(&|_, t| pool_max(t), &shape![0, 2, 2], &shape![2, 2, 2])
            .unwrap_err();
        assert!(matches!(err_indexed_mt, TensorErrors::ShapeContainsZero { .. }));
    }
}
