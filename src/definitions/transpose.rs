use crate::definitions::errors::TensorErrors;
use crate::definitions::shape::Shape;

#[derive(Debug, Eq, PartialEq, Clone)]
pub struct Transpose {
    pub(crate) permutation: Vec<usize>,
    pub(crate) two_cycles: Vec<(usize, usize)>,
}

/// Returns the two-cycles of a permutation.
fn two_cycles_from_permutation(permutation: &[usize]) -> Vec<(usize, usize)> {
    let mut cycles = Vec::new();
    let mut visited = vec![false; permutation.len()];

    for i in 0..permutation.len() {
        if !visited[i] {
            let mut cycle = Vec::new();
            let mut j = i;

            while !visited[j] {
                visited[j] = true;
                cycle.push(j);
                j = permutation[j];
            }

            if cycle.len() > 1 {
                for k in 0..cycle.len() - 1 {
                    cycles.push((cycle[k], cycle[k + 1]));
                }
            }
        }
    }

    cycles
}

impl Transpose {
    /// Constructs a new transpose.
    /// This will fail if the permutation is not a rearrangement of `(0..n)`.
    pub fn new(permutation: &[usize]) -> Result<Self, TensorErrors> {
        let mut perm_copy = permutation.to_vec();
        perm_copy.sort();

        if perm_copy != (0..permutation.len()).collect::<Vec<usize>>() {
            return Err(TensorErrors::TransposePermutationInvalid);
        }

        Ok(Transpose {
            permutation: permutation.to_vec(),
            two_cycles: two_cycles_from_permutation(permutation),
        })
    }

    /// Returns the permutation vector of this transpose.
    pub fn permutation(&self) -> &Vec<usize> {
        &self.permutation
    }

    /// Returns the two-cycles of this transpose.
    pub fn two_cycles(&self) -> &Vec<(usize, usize)> {
        &self.two_cycles
    }

    /// Returns the identity transpose for rank `n` tensors.
    pub fn identity(n: usize) -> Self {
        Transpose::new(&(0..n).collect::<Vec<_>>()).unwrap()
    }

    /// Swaps two axes.
    /// This fails if either axis is out of bounds.
    pub fn swap_axes(self, axis1: usize, axis2: usize) -> Result<Self, TensorErrors> {
        if axis1 >= self.permutation.len() || axis2 >= self.permutation.len() {
            return Err(TensorErrors::TransposePermutationInvalid);
        }

        Ok(unsafe { self.swap_axes_unchecked(axis1, axis2) })
    }

    /// Swaps two axes without bounds checking.
    pub(crate) unsafe fn swap_axes_unchecked(mut self, axis1: usize, axis2: usize) -> Self {
        self.permutation.swap(axis1, axis2);

        Transpose {
            two_cycles: two_cycles_from_permutation(&self.permutation), // This allows for simplifying the two-cycles after the swap.
            permutation: self.permutation,
        }
    }

    /// Returns this transpose applied to `old_shape`.
    /// This fails if `old_shape.rank() != self.permutation().len()`.
    pub fn new_shape(&self, old_shape: Shape) -> Result<Shape, TensorErrors> {
        if old_shape.rank() != self.permutation.len() {
            return Err(TensorErrors::TransposeIncompatibleRank {
                rank: old_shape.rank(),
                trank: self.permutation.len(),
            });
        }

        Ok(unsafe { self.new_shape_unchecked(old_shape) })
    }

    /// Returns this transpose applied to `old_shape`, without validity checking.
    pub(crate) unsafe fn new_shape_unchecked(&self, mut old_shape: Shape) -> Shape {
        for &(i0, i1) in self.two_cycles.iter() {
            old_shape.0.swap(i0, i1);
        }

        old_shape
    }

    /// Returns the old shape that would have been transformed into `new_shape` by this transpose.
    /// This fails if `new_shape.rank() != self.permutation().len()`.
    pub fn old_shape(&self, new_shape: Shape) -> Result<Shape, TensorErrors> {
        if new_shape.rank() != self.permutation.len() {
            return Err(TensorErrors::TransposeIncompatibleRank {
                rank: new_shape.rank(),
                trank: self.permutation.len(),
            });
        }

        Ok(unsafe { self.old_shape_unchecked(new_shape) })
    }

    /// Returns the old shape that would have been transformed into `new_shape` by this transpose, without validity checking.
    pub(crate) unsafe fn old_shape_unchecked(&self, mut new_shape: Shape) -> Shape {
        for &(i0, i1) in self.two_cycles.iter().rev() {
            new_shape.0.swap(i0, i1);
        }

        new_shape
    }

    /// Returns this transpose applied to `old_index`.
    /// This fails if `old_index.len() != self.permutation().len()`.
    pub fn new_index(&self, old_index: &[usize]) -> Result<Vec<usize>, TensorErrors> {
        if old_index.len() != self.permutation.len() {
            return Err(TensorErrors::TransposeIncompatibleRank {
                rank: old_index.len(),
                trank: self.permutation.len(),
            });
        }

        Ok(unsafe { self.new_index_unchecked(old_index.to_vec()) })
    }

    /// Returns this transpose applied to `old_index`, without validity checking.
    pub(crate) unsafe fn new_index_unchecked(&self, mut old_index: Vec<usize>) -> Vec<usize> {
        for &(i0, i1) in self.two_cycles.iter() {
            old_index.swap(i0, i1);
        }

        old_index
    }

    /// Returns the old index that would have been transformed into `new_index` by this transpose.
    /// This fails if `new_index.len() != self.permutation().len()`.
    pub fn old_index(&self, new_index: &[usize]) -> Result<Vec<usize>, TensorErrors> {
        if new_index.len() != self.permutation.len() {
            return Err(TensorErrors::TransposeIncompatibleRank {
                rank: new_index.len(),
                trank: self.permutation.len(),
            });
        }

        Ok(unsafe { self.old_index_unchecked(new_index.to_vec()) })
    }

    /// Returns the old index that would have been transformed into `new_index` by this transpose, without validity checking.
    pub(crate) unsafe fn old_index_unchecked(&self, mut new_index: Vec<usize>) -> Vec<usize> {
        for &(i0, i1) in self.two_cycles.iter().rev() {
            new_index.swap(i0, i1);
        }

        new_index
    }

    /// Returns the inverse transpose.
    pub fn inverse(mut self) -> Transpose {
        self.permutation = (0..self.permutation.len()).collect::<Vec<_>>();

        for &(i0, i1) in self.two_cycles.iter().rev() {
            self.permutation.swap(i0, i1);
        }

        self.two_cycles.reverse();

        self
    }
}

#[cfg(test)]
mod transpose_tests {
    use tensor_math::definitions::errors::TensorErrors;
    use tensor_math::definitions::transpose::Transpose;
    use tensor_math::{shape, transpose};

    #[test]
    fn old_and_new_shapes() {
        let transpose = transpose![1, 0, 2];
        let shape = shape![3, 4, 5];

        let old_shape = transpose.old_shape(shape.clone()).unwrap();
        let ans = transpose.clone().inverse().new_shape(shape).unwrap();

        assert_eq!(old_shape, ans);

        let shape = shape![2, 3];
        let err = transpose.old_shape(shape.clone()).unwrap_err();
        assert!(matches!(
            err,
            TensorErrors::TransposeIncompatibleRank { .. }
        ));

        let err = transpose.inverse().new_shape(shape).unwrap_err();
        assert!(matches!(
            err,
            TensorErrors::TransposeIncompatibleRank { .. }
        ));
    }

    #[test]
    fn invalid_perm() {
        let err = Transpose::new(&vec![4, 1]).unwrap_err();
        assert!(matches!(err, TensorErrors::TransposePermutationInvalid));
    }

    #[test]
    fn indices_do_not_match_rank() {
        let transpose = transpose![1, 0, 2];
        let index = vec![1, 2];

        let err = transpose.new_index(&index).unwrap_err();
        assert!(matches!(
            err,
            TensorErrors::TransposeIncompatibleRank { .. }
        ));

        let err = transpose.old_index(&index).unwrap_err();
        assert!(matches!(
            err,
            TensorErrors::TransposeIncompatibleRank { .. }
        ));
    }

    #[test]
    fn default_perm() {
        let transpose = Transpose::identity(3);
        let ans = transpose![0, 1, 2];

        assert_eq!(transpose, ans);
    }

    #[test]
    fn swap_axes() {
        let transpose = transpose![1, 0, 2];

        let transpose = transpose.swap_axes(0, 1).unwrap();
        assert_eq!(transpose![0, 1, 2], transpose);

        let err = transpose.swap_axes(1, 10).unwrap_err();
        assert!(matches!(err, TensorErrors::TransposePermutationInvalid));
    }

    #[test]
    fn transpose_for_scalar_tensor() {
        let transpose = Transpose::identity(0);
        assert_eq!(transpose.permutation().len(), 0);

        let shape = shape![];
        let new_shape = transpose.new_shape(shape).unwrap();
        assert_eq!(new_shape, shape![]);

        let index = vec![];
        let new_index = transpose.new_index(&index).unwrap();
        assert_eq!(new_index.len(), 0);

        assert_eq!(transpose.clone().inverse(), transpose);
    }

    #[test]
    fn old_and_new_indices() {
        let transpose = transpose![1, 0, 2];
        let index = vec![1, 2, 3];

        let new_index = transpose.new_index(&index).unwrap();

        assert_eq!(new_index, vec![2, 1, 3]);

        let old_index = transpose.old_index(&new_index).unwrap();

        assert_eq!(index, old_index);
    }

    #[test]
    fn two_cycles() {
        let transpose = transpose![2, 0, 1];
        let two_cycles = transpose.two_cycles();

        for i in 0..3 {
            let mut pos = i;
            for cycle in two_cycles.iter().rev() {
                assert!(cycle.0 < 3 && cycle.1 < 3);
                if pos == cycle.0 {
                    pos = cycle.1;
                } else if pos == cycle.1 {
                    pos = cycle.0;
                }
            }
            assert_eq!(pos, transpose.permutation()[i]);
        }
    }
}
