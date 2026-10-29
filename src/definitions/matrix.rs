use crate::definitions::errors::TensorErrors;
use crate::definitions::shape::Shape;
use crate::definitions::strides::Strides;
use crate::definitions::tensor::Tensor;
use crate::definitions::traits::{IntoMatrix, IntoTensor};
use crate::{mat_addr, shape};
use rayon::iter::{FromParallelIterator, IntoParallelIterator};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::vec::IntoIter;
/*
--------------------------------------------
* Matrix definition
--------------------------------------------
*/

/// This struct represents a matrix, i.e. a rank 2 tensor.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Matrix<T> {
    pub(crate) elements: Vec<T>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

impl<T> Matrix<T> {
    /// Returns a new matrix with the given rows and columns and specified elements.
    /// This fails if `elements.len() != rows * cols`.
    pub fn new(rows: usize, cols: usize, elements: Vec<T>) -> Result<Matrix<T>, TensorErrors> {
        if rows * cols != elements.len() {
            return Err(TensorErrors::ShapeSizeDoesNotMatch);
        }
        Ok(Matrix {
            elements,
            rows,
            cols,
        })
    }

    /// Returns the number of rows in the matrix.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns in the matrix.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Gives the shape of the matrix.
    pub fn shape(&self) -> Shape {
        shape![self.rows, self.cols]
    }

    /// Checks if the matrix is square.
    pub fn is_square(&self) -> bool {
        self.cols == self.rows
    }

    /// Returns the underlying elements.
    pub fn elements(&self) -> &[T] {
        &self.elements
    }

    /// Returns a mutable reference to the underlying elements.
    pub fn elements_mut(&mut self) -> &mut [T] {
        &mut self.elements
    }

    /// Gets the element at the specified indices, returning `None` if the indices are out of bounds.
    pub fn get(&self, indices: (usize, usize)) -> Option<&T> {
        if indices.0 >= self.rows || indices.1 >= self.cols {
            return None;
        }

        Some(unsafe { self.get_unchecked(indices) })
    }

    /// Gets the element at the specified indices without bounds checking.
    pub(crate) unsafe fn get_unchecked(&self, indices: (usize, usize)) -> &T {
        self.elements.get_unchecked(mat_addr!(indices, self.cols))
    }

    /// Gets a mutable reference to the element at the specified indices, returning `None` if the indices are out of bounds.
    pub fn get_mut(&mut self, indices: (usize, usize)) -> Option<&mut T> {
        if indices.0 >= self.rows || indices.1 >= self.cols {
            return None;
        }

        Some(unsafe { self.get_unchecked_mut(indices) })
    }

    /// Gets a mutable reference to the element at the specified indices without bounds checking.
    pub(crate) unsafe fn get_unchecked_mut(&mut self, indices: (usize, usize)) -> &mut T {
        self.elements
            .get_unchecked_mut(mat_addr!(indices, self.cols))
    }

    /// Consumes the matrix and returns a parallel iterator over its elements.
    pub fn into_par_iter(self) -> impl ParallelIterator<Item = T>
    where
        T: Send + Sync,
    {
        self.elements.into_par_iter()
    }

    /// Swaps two rows in the matrix.
    pub fn swap_rows(&mut self, row1: usize, row2: usize) -> Result<(), TensorErrors> {
        let max = row1.max(row2);
        if row1 >= self.rows || row2 >= self.rows {
            return Err(TensorErrors::RowOutOfBounds {
                row: max,
                op: "swap_rows",
            });
        }

        unsafe { self.swap_rows_unchecked(row1, row2) };
        Ok(())
    }

    /// Swaps two rows in the matrix without bounds checking.
    pub(crate) unsafe fn swap_rows_unchecked(&mut self, row1: usize, row2: usize) {
        let min = row1.min(row2);
        let max = row1.max(row2);
        let cols = self.cols;

        let (left, right) = self.split_at_mut_unchecked(max * cols);
        left.get_unchecked_mut(min * cols..min * cols + cols)
            .swap_with_slice(right.get_unchecked_mut(..cols));
    }

    /// Swap the values at the indices.
    pub fn swap(
        &mut self,
        index1: (usize, usize),
        index2: (usize, usize),
    ) -> Result<(), TensorErrors> {
        if index1.0 >= self.rows {
            return Err(TensorErrors::IndexOutOfBounds {
                index: index1.0,
                axis: 0,
                length: self.rows,
            });
        }

        if index1.1 >= self.cols {
            return Err(TensorErrors::IndexOutOfBounds {
                index: index1.1,
                axis: 1,
                length: self.cols,
            });
        }

        if index2.0 >= self.rows {
            return Err(TensorErrors::IndexOutOfBounds {
                index: index2.0,
                axis: 0,
                length: self.rows,
            });
        }

        if index2.1 >= self.cols {
            return Err(TensorErrors::IndexOutOfBounds {
                index: index2.1,
                axis: 1,
                length: self.cols,
            });
        }

        self.elements
            .swap(mat_addr!(index1, self.cols), mat_addr!(index2, self.cols));

        Ok(())
    }

    /// Returns an iterator over the rows.
    pub fn iter_rows(&self) -> impl ExactSizeIterator<Item = &[T]> + DoubleEndedIterator {
        let cols = self.cols;
        unsafe {
            (0..self.rows).map(move |row| self.elements.get_unchecked(row * cols..(row + 1) * cols))
        }
    }

    /// Returns a parallel iterator over the rows.
    pub fn par_iter_rows(&self) -> impl IndexedParallelIterator<Item = &[T]>
    where
        T: Send + Sync,
    {
        let cols = self.cols;
        unsafe {
            (0..self.rows)
                .into_par_iter()
                .map(move |row| self.elements.get_unchecked(row * cols..(row + 1) * cols))
        }
    }

    /// Returns a mutable iterator over the rows.
    pub fn iter_rows_mut(
        &mut self,
    ) -> impl ExactSizeIterator<Item = &mut [T]> + DoubleEndedIterator + '_ {
        let rows = self.rows;
        let cols = self.cols;
        let elements = self.elements.as_mut_ptr();
        (0..rows).map(move |row| {
            // The row ranges are disjoint; zero-width rows are empty slices.
            unsafe { std::slice::from_raw_parts_mut(elements.add(row * cols), cols) }
        })
    }

    /// Returns a parallel mutable iterator over the rows.
    pub fn par_iter_rows_mut(&mut self) -> impl IndexedParallelIterator<Item = &mut [T]>
    where
        T: Send + Sync,
    {
        let rows = self.rows;
        let cols = self.cols;
        let mut elements = self.elements.as_mut_slice();
        let mut row_slices = Vec::with_capacity(rows);

        for _ in 0..rows {
            let (row, remaining) = elements.split_at_mut(cols);
            row_slices.push(row);
            elements = remaining;
        }

        row_slices.into_par_iter()
    }
}

/*
--------------------------------------------
* Matrix indexing
--------------------------------------------
*/

impl<T> Index<&[usize; 2]> for Matrix<T> {
    type Output = T;

    fn index(&self, index: &[usize; 2]) -> &Self::Output {
        assert!(
            index[0] < self.rows && index[1] < self.cols,
            "Indices out of bounds: {:?}",
            index
        );

        unsafe {
            self.elements
                .get_unchecked(mat_addr!((index[0], index[1]), self.cols))
        }
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        assert!(
            index.0 < self.rows && index.1 < self.cols,
            "Indices out of bounds: {:?}",
            index
        );

        unsafe { self.elements.get_unchecked(mat_addr!(index, self.cols)) }
    }
}

impl<T> IndexMut<&[usize; 2]> for Matrix<T> {
    fn index_mut(&mut self, index: &[usize; 2]) -> &mut Self::Output {
        assert!(
            index[0] < self.rows && index[1] < self.cols,
            "Indices out of bounds: {:?}",
            index
        );

        unsafe {
            self.elements
                .get_unchecked_mut(mat_addr!((index[0], index[1]), self.cols))
        }
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        assert!(
            index.0 < self.rows && index.1 < self.cols,
            "Indices out of bounds: {:?}",
            index
        );

        unsafe { self.elements.get_unchecked_mut(mat_addr!(index, self.cols)) }
    }
}

/*
--------------------------------------------
* Conversion between Matrix and Tensor
--------------------------------------------
*/

impl<T> IntoTensor<T> for Matrix<T> {
    fn into_tensor(self) -> Tensor<T> {
        Tensor {
            shape: shape![self.rows, self.cols],
            strides: Strides(vec![self.cols, 1]),
            elements: self.elements,
        }
    }
}

/*
--------------------------------------------
* Conversion from Vector
--------------------------------------------
*/

impl<T> IntoMatrix<T> for Vec<T> {
    fn into_matrix(self) -> Matrix<T> {
        let cols = self.len();
        Matrix {
            elements: self,
            rows: 1,
            cols,
        }
    }
}

/*
--------------------------------------------
* Deref and DerefMut implementations
--------------------------------------------
*/

impl<T> Deref for Matrix<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.elements
    }
}

impl<T> DerefMut for Matrix<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.elements
    }
}

/*
--------------------------------------------
* Collection from iterators
--------------------------------------------
*/

impl<T> FromIterator<T> for Matrix<T> {
    /// Converts an iterator into a matrix of shape `(1, iter.len())`.
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let elems = iter.into_iter().collect::<Vec<_>>();

        Matrix {
            rows: 1,
            cols: elems.len(),
            elements: elems,
        }
    }
}

impl<T: Send> FromParallelIterator<T> for Matrix<T> {
    /// Converts a parallel iterator into a matrix of shape `(1, iter.len())`.
    fn from_par_iter<I>(par_iter: I) -> Self
    where
        I: IntoParallelIterator<Item = T>,
    {
        let elements: Vec<T> = par_iter.into_par_iter().collect();
        Matrix {
            cols: elements.len(),
            rows: 1,
            elements,
        }
    }
}

/*
--------------------------------------------
* Conversion into iterators
--------------------------------------------
*/

impl<T> IntoIterator for Matrix<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.into_iter()
    }
}

#[cfg(test)]
mod matrix_definition_tests {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    use tensor_math::definitions::errors::TensorErrors;
    use tensor_math::definitions::matrix::Matrix;
    use tensor_math::definitions::tensor::Tensor;
    use tensor_math::definitions::traits::{IntoMatrix, IntoTensor, TryIntoMatrix};
    use tensor_math::shape;

    #[test]
    fn invalid_shape_and_elements() {
        let err = Matrix::new(1, 1, vec![1, 2]).unwrap_err();
        assert_eq!(err, TensorErrors::ShapeSizeDoesNotMatch);
    }

    #[test]
    fn is_square() {
        let m1 = Matrix::<i32>::from_shape(2, 2);
        let m2 = Matrix::<i32>::from_shape(2, 3);

        assert!(m1.is_square());
        assert!(!m2.is_square());
    }

    #[test]
    #[should_panic]
    fn invalid_index() {
        let m1 = Matrix::<i32>::from_shape(2, 2);
        m1[(2, 2)];
    }

    #[test]
    fn get_is_safe_indexing() {
        let m1 = Matrix::<i32>::new(2, 2, vec![0, 1, 2, 3]).unwrap();

        assert_eq!(m1.get((1, 1)), Some(&3));
        assert_eq!(m1.get((2, 2)), None);
    }

    #[test]
    fn rows_cols_gives_shape() {
        let m1 = Matrix::<i32>::new(2, 3, vec![0, 1, 2, 3, 4, 5]).unwrap();

        assert_eq!(m1.rows(), 2);
        assert_eq!(m1.cols(), 3);
        assert_eq!(m1.shape(), shape![2, 3]);
        assert_eq!(
            Matrix::<i32>::new(0, 3, vec![]).unwrap().shape(),
            shape![0, 3]
        );
        assert_eq!(
            Matrix::<i32>::new(3, 0, vec![]).unwrap().shape(),
            shape![3, 0]
        );
    }

    #[test]
    fn new_accepts_all_zero_dimension_shapes() {
        assert_eq!(
            Matrix::<i32>::new(0, 0, vec![]).unwrap().elements(),
            &[] as &[i32]
        );
        assert_eq!(Matrix::<i32>::new(0, 3, vec![]).unwrap().rows(), 0);
        assert_eq!(Matrix::<i32>::new(3, 0, vec![]).unwrap().cols(), 0);
        assert_eq!(
            Matrix::<i32>::new(0, 1, vec![1]),
            Err(TensorErrors::ShapeSizeDoesNotMatch)
        );
        assert_eq!(
            Matrix::<i32>::new(1, 0, vec![1]),
            Err(TensorErrors::ShapeSizeDoesNotMatch)
        );
    }

    #[test]
    fn is_square_includes_empty_and_zero_dimension_matrices() {
        assert!(Matrix::<i32>::new(0, 0, vec![]).unwrap().is_square());
        assert!(!Matrix::<i32>::new(0, 3, vec![]).unwrap().is_square());
        assert!(!Matrix::<i32>::new(3, 0, vec![]).unwrap().is_square());
    }

    #[test]
    fn elements_and_elements_mut_expose_only_the_element_slice() {
        let mut matrix = Matrix::new(2, 2, vec![1, 2, 3, 4]).unwrap();
        assert_eq!(matrix.elements(), &[1, 2, 3, 4]);
        matrix.elements_mut()[1] = 20;
        assert_eq!(matrix.elements(), &[1, 20, 3, 4]);
        assert_eq!(
            Matrix::<i32>::new(0, 3, vec![]).unwrap().elements(),
            &[] as &[i32]
        );
    }

    #[test]
    fn get_and_get_mut_handle_each_out_of_bounds_axis_and_empty_shapes() {
        let mut matrix = Matrix::new(2, 2, vec![1, 2, 3, 4]).unwrap();
        assert_eq!(matrix.get((1, 0)), Some(&3));
        assert_eq!(matrix.get((2, 0)), None);
        assert_eq!(matrix.get((0, 2)), None);
        *matrix.get_mut((1, 0)).unwrap() = 30;
        assert_eq!(matrix.get((1, 0)), Some(&30));
        assert!(matrix.get_mut((2, 0)).is_none());
        assert!(matrix.get_mut((0, 2)).is_none());

        let mut no_rows = Matrix::<i32>::new(0, 3, vec![]).unwrap();
        let mut no_columns = Matrix::<i32>::new(3, 0, vec![]).unwrap();
        assert_eq!(no_rows.get((0, 0)), None);
        assert_eq!(no_columns.get((0, 0)), None);
        assert!(no_rows.get_mut((0, 0)).is_none());
        assert!(no_columns.get_mut((0, 0)).is_none());
    }

    #[test]
    fn swap_rows_swaps_distinct_valid_rows_and_checks_zero_column_rows() {
        let mut matrix = Matrix::new(3, 2, vec![1, 2, 3, 4, 5, 6]).unwrap();
        matrix.swap_rows(0, 2).unwrap();
        assert_eq!(matrix.elements(), &[5, 6, 3, 4, 1, 2]);

        let mut zero_columns = Matrix::<i32>::new(2, 0, vec![]).unwrap();
        assert_eq!(
            zero_columns.swap_rows(0, 2),
            Err(TensorErrors::RowOutOfBounds {
                row: 2,
                op: "swap_rows"
            })
        );
        assert_eq!(
            Matrix::<i32>::new(0, 0, vec![]).unwrap().swap_rows(0, 0),
            Err(TensorErrors::RowOutOfBounds {
                row: 0,
                op: "swap_rows"
            })
        );
        assert_eq!(
            matrix.swap_rows(0, 3),
            Err(TensorErrors::RowOutOfBounds {
                row: 3,
                op: "swap_rows"
            })
        );
    }

    #[test]
    fn swap_exchanges_elements_and_checks_both_indices() {
        let mut matrix = Matrix::new(2, 2, vec![1, 2, 3, 4]).unwrap();
        matrix.swap((0, 1), (1, 0)).unwrap();
        assert_eq!(matrix.elements(), &[1, 3, 2, 4]);
        matrix.swap((0, 0), (0, 0)).unwrap();

        assert_eq!(
            matrix.swap((2, 0), (0, 0)),
            Err(TensorErrors::IndexOutOfBounds {
                index: 2,
                axis: 0,
                length: 2
            })
        );
        assert_eq!(
            matrix.swap((0, 2), (0, 0)),
            Err(TensorErrors::IndexOutOfBounds {
                index: 2,
                axis: 1,
                length: 2
            })
        );
        assert_eq!(
            matrix.swap((0, 0), (2, 0)),
            Err(TensorErrors::IndexOutOfBounds {
                index: 2,
                axis: 0,
                length: 2
            })
        );
        assert_eq!(
            matrix.swap((0, 0), (0, 2)),
            Err(TensorErrors::IndexOutOfBounds {
                index: 2,
                axis: 1,
                length: 2
            })
        );

        let mut empty = Matrix::<i32>::new(0, 3, vec![]).unwrap();
        assert!(empty.swap((0, 0), (0, 0)).is_err());
    }

    #[test]
    fn row_iterators_return_rows_in_order_and_mutate_rows() {
        let mut matrix = Matrix::new(2, 3, vec![1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(matrix.iter_rows().next_back(), Some(&[4, 5, 6][..]));
        assert_eq!(
            matrix.par_iter_rows().collect::<Vec<_>>(),
            vec![&[1, 2, 3][..], &[4, 5, 6][..]]
        );
        for row in matrix.iter_rows_mut() {
            row[0] *= 10;
        }
        assert_eq!(matrix.elements(), &[10, 2, 3, 40, 5, 6]);
        matrix.par_iter_rows_mut().for_each(|row| row[1] += 1);
        assert_eq!(matrix.elements(), &[10, 3, 3, 40, 6, 6]);
    }

    #[test]
    fn parallel_iterator_consumes_empty_and_nonempty_matrices() {
        assert_eq!(
            Matrix::new(0, 0, Vec::<i32>::new())
                .unwrap()
                .into_par_iter()
                .count(),
            0
        );
        assert_eq!(
            Matrix::new(1, 3, vec![1, 2, 3])
                .unwrap()
                .into_par_iter()
                .sum::<i32>(),
            6
        );
    }

    #[test]
    fn indexing_and_mutable_indexing_support_slice_and_tuple_indices() {
        let mut matrix = Matrix::new(2, 2, vec![1, 2, 3, 4]).unwrap();
        assert_eq!(matrix[&[0, 1]], 2);
        assert_eq!(matrix[(1, 0)], 3);
        matrix[&[0, 1]] = 20;
        matrix[(1, 0)] = 30;
        assert_eq!(matrix.elements(), &[1, 20, 30, 4]);
    }

    #[test]
    #[should_panic]
    fn slice_indexing_panics_for_out_of_bounds_indices() {
        let matrix = Matrix::<i32>::new(0, 3, vec![]).unwrap();
        let _ = matrix[&[0, 0]];
    }

    #[test]
    #[should_panic]
    fn tuple_indexing_panics_for_out_of_bounds_indices() {
        let matrix = Matrix::<i32>::new(3, 0, vec![]).unwrap();
        let _ = matrix[(0, 0)];
    }

    #[test]
    #[should_panic]
    fn mutable_slice_indexing_panics_for_out_of_bounds_indices() {
        let mut matrix = Matrix::<i32>::new(0, 3, vec![]).unwrap();
        matrix[&[0, 0]] = 1;
    }

    #[test]
    #[should_panic]
    fn mutable_tuple_indexing_panics_for_out_of_bounds_indices() {
        let mut matrix = Matrix::<i32>::new(3, 0, vec![]).unwrap();
        matrix[(0, 0)] = 1;
    }

    #[test]
    fn convert_into_tensor() {
        let m1 = Matrix::<i32>::new(2, 3, vec![0, 1, 2, 3, 4, 5]).unwrap();
        let t1 = Tensor::<i32>::new(shape![2, 3], vec![0, 1, 2, 3, 4, 5]).unwrap();
        let m2 = Matrix::<usize>::new(0, 0, vec![]).unwrap();
        let t2 = Tensor::<usize>::new(shape![0, 0], vec![]).unwrap();
        let m3 = Matrix::<usize>::new(0, 3, vec![]).unwrap();
        let t3 = Tensor::<usize>::new(shape![0, 3], vec![]).unwrap();
        let m4 = Matrix::<usize>::new(3, 0, vec![]).unwrap();
        let t4 = Tensor::<usize>::new(shape![3, 0], vec![]).unwrap();

        assert_eq!(m1.into_tensor(), t1);
        assert_eq!(m2.into_tensor(), t2);
        assert_eq!(m3.into_tensor(), t3);
        assert_eq!(m4.into_tensor(), t4);
    }

    #[test]
    fn vec_into_matrix_preserves_empty_and_nonempty_vectors() {
        let matrix = vec![1, 2, 3].into_matrix();
        assert_eq!(matrix, Matrix::new(1, 3, vec![1, 2, 3]).unwrap());
        assert_eq!(
            Vec::<i32>::new().into_matrix(),
            Matrix::new(1, 0, vec![]).unwrap()
        );
    }

    #[test]
    fn deref_and_deref_mut_expose_the_underlying_vector() {
        let mut matrix = Matrix::new(1, 2, vec![1, 2]).unwrap();
        assert_eq!(&*matrix, &vec![1, 2]);
        matrix.push(3);
        assert_eq!(&*matrix, &vec![1, 2, 3]);
        assert_eq!(Matrix::<i32>::new(0, 0, vec![]).unwrap().len(), 0);
    }

    #[test]
    fn default_matrix() {
        let m1 = Matrix::<i32>::default();
        assert_eq!(m1, Matrix::new(1, 1, vec![0]).unwrap());
    }

    #[test]
    fn from_iter() {
        let v1 = vec![1, 2, 3, 4, 5, 6];
        let m1: Matrix<i32> = v1.iter().copied().collect();

        let ans = Matrix::new(1, 6, v1).unwrap();

        assert_eq!(m1, ans);
        let empty: Matrix<i32> = std::iter::empty().collect();
        assert_eq!(empty, Matrix::new(1, 0, vec![]).unwrap());
    }

    #[test]
    fn from_parallel_iterator_supports_empty_and_nonempty_iterators() {
        let matrix: Matrix<i32> = (0..6).into_par_iter().collect();
        assert_eq!(matrix, Matrix::new(1, 6, (0..6).collect()).unwrap());
        let empty: Matrix<i32> = Vec::<i32>::new().into_par_iter().collect();
        assert_eq!(empty, Matrix::new(1, 0, vec![]).unwrap());
    }

    #[test]
    fn into_iterator_yields_all_elements_and_handles_empty_matrix() {
        let matrix = Matrix::new(2, 2, vec![1, 2, 3, 4]).unwrap();
        assert_eq!(matrix.into_iter().collect::<Vec<_>>(), vec![1, 2, 3, 4]);
        assert_eq!(
            Matrix::<i32>::new(0, 3, vec![])
                .unwrap()
                .into_iter()
                .count(),
            0
        );
    }

    #[test]
    fn try_from_tensor() {
        let t1 = (0..6).collect::<Tensor<_>>().reshape(shape![2, 3]).unwrap();
        let m1 = t1.try_into_matrix().unwrap();
        let ans = Matrix::new(2, 3, vec![0, 1, 2, 3, 4, 5]).unwrap();
        assert_eq!(m1, ans);

        for shape in [shape![0, 0], shape![0, 3], shape![3, 0]] {
            let tensor = Tensor::<i32>::new(shape.clone(), vec![]).unwrap();
            let matrix = tensor.try_into_matrix().unwrap();
            assert_eq!(matrix.shape(), shape);
            assert!(matrix.is_empty());
        }
    }

    #[test]
    fn invalid_try_from_tensor() {
        let t1 = vec![1, 2, 3, 4, 5, 6].into_tensor();
        let err = t1.try_into_matrix().unwrap_err();
        assert!(matches!(err, TensorErrors::RanksDoNotMatch(..)));
    }

    #[test]
    fn can_create_empty_matrix() {
        let m1 = Matrix::<i32>::new(0, 0, vec![]).unwrap();
        assert_eq!(0, m1.rows());
        assert_eq!(0, m1.cols());
    }

    #[test]
    #[should_panic]
    fn invalid_indexing_empty_matrix() {
        let m1 = Matrix::<i32>::new(0, 0, vec![]).unwrap();
        m1[&[0, 0]];
    }

    #[test]
    fn zero_column_matrices_yield_one_empty_row_per_row() {
        let mut matrix = Matrix::new(3, 0, Vec::<i32>::new()).unwrap();

        assert_eq!(
            matrix.iter_rows().map(<[i32]>::len).collect::<Vec<_>>(),
            [0; 3]
        );
        assert_eq!(
            matrix.par_iter_rows().map(<[i32]>::len).collect::<Vec<_>>(),
            [0; 3]
        );
        assert_eq!(
            matrix
                .iter_rows_mut()
                .map(|row| row.len())
                .collect::<Vec<_>>(),
            [0; 3]
        );
        assert_eq!(
            matrix
                .par_iter_rows_mut()
                .map(|row| row.len())
                .collect::<Vec<_>>(),
            [0; 3]
        );
    }

    #[test]
    fn zero_row_matrices_yield_no_rows() {
        let mut matrix = Matrix::new(0, 3, Vec::<i32>::new()).unwrap();

        assert_eq!(matrix.iter_rows().count(), 0);
        assert_eq!(matrix.par_iter_rows().count(), 0);
        assert_eq!(matrix.iter_rows_mut().count(), 0);
        assert_eq!(matrix.par_iter_rows_mut().count(), 0);
    }
}
