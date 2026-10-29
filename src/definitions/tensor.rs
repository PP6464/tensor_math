use crate::definitions::errors::TensorErrors;
use crate::definitions::matrix::Matrix;
use crate::definitions::shape::Shape;
use crate::definitions::strides::Strides;
use crate::definitions::traits::{IntoTensor, TryIntoMatrix};
use crate::shape;
use crate::utilities::internal_functions::dot_vectors;
use rayon::iter::{FromParallelIterator, IntoParallelIterator, ParallelIterator};
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::vec::IntoIter;
/*
--------------------------------------------
* Tensor definition
--------------------------------------------
*/

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Tensor<T> {
    pub(crate) shape: Shape,
    pub(crate) strides: Strides,
    pub(crate) elements: Vec<T>,
}

impl<T> Tensor<T> {
    /// Returns a new tensor with the specified shape and elements.
    /// This fails if the number of elements does not match `shape.element_count()`.
    pub fn new(shape: Shape, elements: Vec<T>) -> Result<Self, TensorErrors> {
        if shape.element_count() != elements.len() {
            return Err(TensorErrors::ShapeSizeDoesNotMatch);
        }

        let strides = Strides::from_shape(&shape);

        Ok(Tensor {
            shape,
            strides,
            elements,
        })
    }

    /// Returns the elements of the tensor.
    pub fn elements(&self) -> &[T] {
        &self.elements
    }

    /// Returns a mutable reference to the elements of the tensor.
    pub fn elements_mut(&mut self) -> &mut [T] {
        &mut self.elements
    }

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> Shape {
        self.shape.clone()
    }

    /// Returns the rank of the tensor.
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Gets the element at the specified indices, returning `None` if the indices are out of bounds.
    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        if indices.len() != self.rank() {
            return None;
        }
        for (index_i, shape_i) in indices.iter().zip(self.shape.0.iter()) {
            if index_i >= shape_i {
                return None;
            }
        }
        Some(unsafe { self.get_unchecked(indices) })
    }

    /// Gets the element at the specified indices without bounds checking.
    pub(crate) unsafe fn get_unchecked(&self, indices: &[usize]) -> &T {
        self.elements
            .get_unchecked(dot_vectors(&self.strides.0, indices))
    }

    /// Gets a mutable reference to the element at the specified indices, returning `None` if the indices are out of bounds.
    pub fn get_mut(&mut self, indices: &[usize]) -> Option<&mut T> {
        if indices.len() != self.rank() {
            return None;
        }
        for (index_i, shape_i) in indices.iter().zip(self.shape.0.iter()) {
            if *index_i >= *shape_i {
                return None;
            }
        }
        Some(unsafe { self.get_unchecked_mut(indices) })
    }

    /// Gets a mutable reference to the element at the specified indices without bounds checking.
    pub(crate) unsafe fn get_unchecked_mut(&mut self, indices: &[usize]) -> &mut T {
        self.elements
            .get_unchecked_mut(dot_vectors(&self.strides.0, indices))
    }

    /// Consumes the tensor and returns a parallel iterator over its elements.
    pub fn into_par_iter(self) -> impl ParallelIterator<Item = T>
    where
        T: Send + Sync,
    {
        self.elements.into_par_iter()
    }

    /// Swaps the elements at the indices.
    pub fn swap(&mut self, index1: &[usize], index2: &[usize]) -> Result<(), TensorErrors> {
        if index1.len() != self.rank() {
            return Err(TensorErrors::IndicesInvalidForRank(
                index1.len(),
                self.rank(),
            ));
        }

        if index2.len() != self.rank() {
            return Err(TensorErrors::IndicesInvalidForRank(
                index2.len(),
                self.rank(),
            ));
        }

        for (si, (&i1, &i2)) in index1.iter().zip(index2.iter()).enumerate() {
            if i1 >= self.shape[si] {
                return Err(TensorErrors::IndexOutOfBounds {
                    axis: si,
                    length: self.shape[si],
                    index: i1,
                });
            }

            if i2 >= self.shape[si] {
                return Err(TensorErrors::IndexOutOfBounds {
                    axis: si,
                    length: self.shape[si],
                    index: i2,
                });
            }
        }

        unsafe {
            self.elements.swap(
                self.shape.address_unchecked(index1),
                self.shape.address_unchecked(index2),
            );
        }

        Ok(())
    }
}

/*
--------------------------------------------
* Tensor indexing
--------------------------------------------
*/

impl<T> Index<&[usize]> for Tensor<T> {
    type Output = T;

    fn index(&self, index: &[usize]) -> &Self::Output {
        assert_eq!(
            self.rank(),
            index.len(),
            "Shape dimension and index dimension do not match"
        );
        for (index_i, shape_i) in index.iter().zip(self.shape.0.iter()) {
            assert!(
                index_i < shape_i,
                "Index for dimension {index_i} out of bounds: index {}, shape {}",
                index_i,
                shape_i
            );
        }

        let addr = dot_vectors(&self.strides.clone().0, index);
        unsafe { self.elements.get_unchecked(addr) }
    }
}

impl<T> IndexMut<&[usize]> for Tensor<T> {
    fn index_mut(&mut self, index: &[usize]) -> &mut T {
        assert_eq!(
            self.rank(),
            index.len(),
            "Shape dimension and index dimension do not match"
        );
        for (index_i, shape_i) in index.iter().zip(self.shape.0.iter()) {
            assert!(
                index_i < shape_i,
                "Index for dimension {index_i} out of bounds: index {}, shape {}",
                index_i,
                shape_i
            );
        }

        let addr = dot_vectors(&self.strides.clone().0, index);
        unsafe { self.elements.get_unchecked_mut(addr) }
    }
}

/*
--------------------------------------------
* Conversion between Matrix and Tensor
--------------------------------------------
*/

impl<T> TryIntoMatrix<T> for Tensor<T> {
    type Error = TensorErrors;

    fn try_into_matrix(self) -> Result<Matrix<T>, TensorErrors> {
        if self.shape.rank() != 2 {
            return Err(TensorErrors::RanksDoNotMatch(self.shape.rank(), 2));
        }

        Ok(Matrix {
            elements: self.elements,
            rows: self.shape[0],
            cols: self.shape[1],
        })
    }
}

/*
--------------------------------------------
* Conversion from Vector
--------------------------------------------
*/

impl<T> IntoTensor<T> for Vec<T> {
    fn into_tensor(self) -> Tensor<T> {
        Tensor {
            shape: shape![self.len()],
            strides: Strides(vec![1]),
            elements: self,
        }
    }
}

/*
--------------------------------------------
* Conversion into iterator
--------------------------------------------
*/

impl<T> IntoIterator for Tensor<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.into_iter()
    }
}

/*
--------------------------------------------
* Collection from iterators
--------------------------------------------
*/

impl<T> FromIterator<T> for Tensor<T> {
    /// Converts an iterator into a tensor of shape `(iter.len())`
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let elements: Vec<T> = iter.into_iter().collect();
        Tensor {
            shape: shape![elements.len()],
            strides: Strides(vec![1]),
            elements,
        }
    }
}

impl<T: Send> FromParallelIterator<T> for Tensor<T> {
    /// Converts a parallel iterator into a tensor of shape `(iter.len())`
    fn from_par_iter<I>(par_iter: I) -> Self
    where
        I: IntoParallelIterator<Item = T>,
    {
        let elements = par_iter.into_par_iter().collect::<Vec<_>>();
        Tensor {
            shape: shape![elements.len()],
            strides: Strides(vec![1]),
            elements,
        }
    }
}

/*
--------------------------------------------
* Deref and DerefMut implementations
--------------------------------------------
*/

impl<T> Deref for Tensor<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.elements()
    }
}

impl<T> DerefMut for Tensor<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.elements_mut()
    }
}

#[cfg(test)]
mod tensor_definition_tests {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    use tensor_math::definitions::errors::TensorErrors;
    use tensor_math::definitions::matrix::Matrix;
    use tensor_math::definitions::tensor::Tensor;
    use tensor_math::definitions::traits::{IntoTensor, TryIntoMatrix};
    use tensor_math::shape;

    #[test]
    fn invalid_shape_size_and_data_length() {
        let shape = shape![2, 3, 4];
        let data = vec![1, 2, 3, 4];
        let err = Tensor::new(shape, data).unwrap_err();
        assert_eq!(err, TensorErrors::ShapeSizeDoesNotMatch);
    }

    #[test]
    #[should_panic]
    fn invalid_index_out_of_bounds() {
        let shape = shape![2, 3, 4];
        let tensor = Tensor::<i32>::from_shape(shape);

        tensor[&[1, 2, 4]];
    }

    #[test]
    #[should_panic]
    fn invalid_index_incorrect_rank() {
        let shape = shape![2, 3, 4];
        let tensor = Tensor::<i32>::from_shape(shape);

        tensor[&[1, 2]];
    }

    #[test]
    #[should_panic]
    fn invalid_mut_index_out_of_bounds() {
        let shape = shape![2, 3, 4];
        let mut tensor = Tensor::<i32>::from_shape(shape);

        tensor[&[1, 2, 4]] = 1;
    }

    #[test]
    #[should_panic]
    fn invalid_mut_index_incorrect_rank() {
        let shape = shape![2, 3, 4];
        let mut tensor = Tensor::<i32>::from_shape(shape);

        tensor[&[1, 2]] = 1;
    }

    #[test]
    fn from_matrix() {
        let m1 = Matrix::new(3, 5, (0..15).collect()).unwrap();
        let t1 = m1.into_tensor();
        let ans = Tensor::new(shape![3, 5], (0..15).collect()).unwrap();

        assert_eq!(t1, ans);
    }

    #[test]
    fn can_create_scalar_tensor() {
        let t1 = Tensor::new(shape![], vec![8]).unwrap();
        assert_eq!(t1.shape(), shape![]);
        assert_eq!(t1.elements().len(), 1);
    }

    #[test]
    fn can_create_empty_tensor() {
        let t1 = Tensor::<usize>::new(shape![0], vec![]).unwrap();
        assert_eq!(t1.shape(), shape![0]);
        assert_eq!(t1.elements().len(), 0);
    }

    #[test]
    fn get() {
        let t1 = Tensor::new(shape![2, 3, 4], (0..24).collect()).unwrap();
        assert_eq!(t1.get(&[0, 2, 3]), Some(&11));
        assert_eq!(t1.get(&[10, 0, 0]), None);
        assert_eq!(t1.get(&[10]), None);

        let t2 = Tensor::new(shape![], vec![2]).unwrap();
        assert_eq!(t2.get(&[]), Some(&2));
        assert_eq!(t2.get(&[0, 2, 3]), None);
    }

    #[test]
    fn normal_indexing() {
        let t1 = Tensor::new(shape![2, 3], (0..6).collect()).unwrap();
        assert_eq!(t1[&[0, 0]], 0);
        assert_eq!(t1[&[1, 2]], 5);

        let t2 = Tensor::new(shape![], vec![2]).unwrap();
        assert_eq!(t2[&[]], 2);
    }

    #[test]
    fn new_elements_shape_and_rank_cover_scalar_and_empty_shapes() {
        let scalar = Tensor::new(shape![], vec![42]).unwrap();
        assert_eq!(scalar.rank(), 0);
        assert_eq!(scalar.shape(), shape![]);
        assert_eq!(scalar.elements(), &[42]);

        for dims in [shape![0], shape![0, 3], shape![3, 0], shape![0, 0]] {
            let tensor = Tensor::<i32>::new(dims.clone(), vec![]).unwrap();
            assert_eq!(tensor.shape(), dims);
            assert!(tensor.elements().is_empty());
        }
        assert_eq!(
            Tensor::<i32>::new(shape![], vec![]),
            Err(TensorErrors::ShapeSizeDoesNotMatch)
        );
    }

    #[test]
    fn elements_mut_updates_values_and_empty_tensor_is_a_valid_slice() {
        let mut tensor = Tensor::new(shape![2], vec![1, 2]).unwrap();
        tensor.elements_mut()[1] = 3;
        assert_eq!(tensor.elements(), &[1, 3]);

        let mut empty = Tensor::<i32>::new(shape![0, 2], vec![]).unwrap();
        assert!(empty.elements_mut().is_empty());
    }

    #[test]
    fn get_and_get_mut_validate_rank_and_each_dimension() {
        let mut tensor = Tensor::new(shape![2, 3], (0..6).collect()).unwrap();
        assert_eq!(tensor.get(&[1, 2]), Some(&5));
        assert_eq!(tensor.get(&[]), None);
        assert_eq!(tensor.get(&[1]), None);
        assert_eq!(tensor.get(&[2, 0]), None);
        assert_eq!(tensor.get(&[0, 3]), None);

        *tensor.get_mut(&[1, 1]).unwrap() = 20;
        assert_eq!(tensor.get(&[1, 1]), Some(&20));
        assert!(tensor.get_mut(&[]).is_none());
        assert!(tensor.get_mut(&[1]).is_none());
        assert!(tensor.get_mut(&[2, 0]).is_none());
        assert!(tensor.get_mut(&[0, 3]).is_none());

        let scalar = Tensor::new(shape![], vec![8]).unwrap();
        assert_eq!(scalar.get(&[]), Some(&8));

        let empty = Tensor::<i32>::new(shape![0, 2], vec![]).unwrap();
        assert_eq!(empty.get(&[0, 0]), None);
    }

    #[test]
    fn swap_handles_scalar_and_checks_ranks_and_indices() {
        let mut tensor = Tensor::new(shape![2, 2], vec![0, 1, 2, 3]).unwrap();
        tensor.swap(&[0, 1], &[1, 0]).unwrap();
        assert_eq!(tensor.elements(), &[0, 2, 1, 3]);
        tensor.swap(&[0, 0], &[0, 0]).unwrap();

        assert_eq!(
            tensor.swap(&[0], &[1, 0]),
            Err(TensorErrors::IndicesInvalidForRank(1, 2))
        );
        assert_eq!(
            tensor.swap(&[0, 0], &[0]),
            Err(TensorErrors::IndicesInvalidForRank(1, 2))
        );
        assert_eq!(
            tensor.swap(&[2, 0], &[0, 0]),
            Err(TensorErrors::IndexOutOfBounds {
                axis: 0,
                length: 2,
                index: 2
            })
        );
        assert_eq!(
            tensor.swap(&[0, 0], &[0, 2]),
            Err(TensorErrors::IndexOutOfBounds {
                axis: 1,
                length: 2,
                index: 2
            })
        );

        let mut scalar = Tensor::new(shape![], vec![7]).unwrap();
        scalar.swap(&[], &[]).unwrap();
        assert_eq!(scalar.elements(), &[7]);

        let mut empty = Tensor::<i32>::new(shape![0], vec![]).unwrap();
        assert!(empty.swap(&[0], &[0]).is_err());
    }

    #[test]
    fn checked_indexing_mutation_and_panics() {
        let mut tensor = Tensor::new(shape![2], vec![1, 2]).unwrap();
        tensor[&[1]] = 3;
        assert_eq!(tensor[&[1]], 3);
        let mut scalar = Tensor::new(shape![], vec![42]).unwrap();
        scalar[&[]] = 24;
        assert_eq!(scalar[&[]], 24);
    }

    #[test]
    #[should_panic]
    fn immutable_indexing_panics_for_empty_dimension() {
        let tensor = Tensor::<i32>::new(shape![0], vec![]).unwrap();
        let _ = tensor[&[0]];
    }

    #[test]
    #[should_panic]
    fn mutable_indexing_panics_for_empty_dimension() {
        let mut tensor = Tensor::<i32>::new(shape![0], vec![]).unwrap();
        tensor[&[0]] = 1;
    }

    #[test]
    fn parallel_and_consuming_iterators_cover_scalar_and_empty_tensors() {
        assert_eq!(
            Tensor::new(shape![2], vec![1, 2])
                .unwrap()
                .into_par_iter()
                .sum::<i32>(),
            3
        );
        assert_eq!(
            Tensor::<i32>::new(shape![0, 2], vec![])
                .unwrap()
                .into_par_iter()
                .count(),
            0
        );
        assert_eq!(
            Tensor::new(shape![], vec![7])
                .unwrap()
                .into_par_iter()
                .sum::<i32>(),
            7
        );
        assert_eq!(
            Tensor::new(shape![2], vec![1, 2])
                .unwrap()
                .into_iter()
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(
            Tensor::<i32>::new(shape![0], vec![])
                .unwrap()
                .into_iter()
                .count(),
            0
        );
    }

    #[test]
    fn matrix_conversion_checks_rank_and_preserves_empty_dimensions() {
        assert_eq!(
            Tensor::new(shape![], vec![1])
                .unwrap()
                .try_into_matrix()
                .unwrap_err(),
            TensorErrors::RanksDoNotMatch(0, 2)
        );
        assert_eq!(
            Tensor::new(shape![3], vec![1, 2, 3])
                .unwrap()
                .try_into_matrix()
                .unwrap_err(),
            TensorErrors::RanksDoNotMatch(1, 2)
        );
        assert_eq!(
            Tensor::new(shape![2, 2], vec![1, 2, 3, 4])
                .unwrap()
                .try_into_matrix()
                .unwrap(),
            Matrix::new(2, 2, vec![1, 2, 3, 4]).unwrap()
        );
        assert_eq!(
            Tensor::<i32>::new(shape![0, 3], vec![])
                .unwrap()
                .try_into_matrix()
                .unwrap(),
            Matrix::new(0, 3, vec![]).unwrap()
        );
    }

    #[test]
    fn vector_conversions_and_collections_handle_empty_and_nonempty_data() {
        assert_eq!(
            vec![1, 2].into_tensor(),
            Tensor::new(shape![2], vec![1, 2]).unwrap()
        );
        assert_eq!(
            Vec::<i32>::new().into_tensor(),
            Tensor::new(shape![0], vec![]).unwrap()
        );
        assert_eq!(
            [1, 2, 3].into_iter().collect::<Tensor<_>>(),
            Tensor::new(shape![3], vec![1, 2, 3]).unwrap()
        );
        assert_eq!(
            std::iter::empty().collect::<Tensor<i32>>(),
            Tensor::new(shape![0], vec![]).unwrap()
        );
        assert_eq!(
            (0..3).into_par_iter().collect::<Tensor<_>>(),
            Tensor::new(shape![3], vec![0, 1, 2]).unwrap()
        );
        assert_eq!(
            Vec::<i32>::new().into_par_iter().collect::<Tensor<_>>(),
            Tensor::new(shape![0], vec![]).unwrap()
        );
    }

    #[test]
    fn deref_and_deref_mut_expose_tensor_elements() {
        let mut tensor = Tensor::new(shape![2], vec![4, 5]).unwrap();
        assert_eq!(&*tensor, &[4, 5]);
        std::ops::DerefMut::deref_mut(&mut tensor)[0] = 10;
        assert_eq!(&*tensor, &[10, 5]);

        let empty = Tensor::<i32>::new(shape![0], vec![]).unwrap();
        assert!(empty.is_empty());
    }
}
