use crate::definitions::errors::TensorErrors;
use crate::definitions::shape::Shape;

#[derive(Debug, Eq, PartialEq)]
pub struct Transpose {
    pub(crate) permutation: Vec<usize>,
}

impl Transpose {
    /// Constructs a new transpose.
    ///
    /// This will fail if the permutation is not a rearrangement of `(0..n)`.
    pub fn new(permutation: &Vec<usize>) -> Result<Self, TensorErrors> {
        let mut perm_copy = permutation.to_vec();
        perm_copy.sort();

        if perm_copy != (0..permutation.len()).collect::<Vec<usize>>() {
            return Err(TensorErrors::TransposePermutationInvalid);
        }

        Ok(Transpose {
            permutation: permutation.clone(),
        })
    }

    /// Returns the permutation vector of this transpose.
    pub fn permutation(&self) -> &Vec<usize> {
        &self.permutation
    }

    /// Returns the identity transpose for rank `n` tensors.
    pub fn default(n: usize) -> Self {
        Transpose::new(&(0..n).collect()).unwrap()
    }

    /// Swaps two axes.
    ///
    /// This fails if either axis is out of bounds.
    pub fn swap_axes(&self, axis1: usize, axis2: usize) -> Result<Self, TensorErrors> {
        if axis1 >= self.permutation.len() || axis2 >= self.permutation.len() {
            return Err(TensorErrors::TransposePermutationInvalid);
        }

        let mut new_perm = self.permutation.clone();
        new_perm.swap(axis1, axis2);

        Transpose::new(&new_perm)
    }

    /// Returns this transpose applied to `old_shape`.
    ///
    /// This fails if `old_shape.rank() != self.permutation().len()`.
    pub fn new_shape(&self, old_shape: &Shape) -> Result<Shape, TensorErrors> {
        if old_shape.rank() != self.permutation.len() {
            return Err(TensorErrors::TransposeIncompatibleRank {
                rank: old_shape.rank(),
                trank: self.permutation.len(),
            });
        }

        let mut new_shape_vec = Vec::with_capacity(old_shape.rank());

        for old_pos in self.permutation.iter() {
            new_shape_vec.push(old_shape[*old_pos]);
        }

        Ok(Shape::new(new_shape_vec))
    }

    /// Returns the old shape that would have been transformed into `new_shape` by this transpose.
    /// 
    /// This fails if `new_shape.rank() != self.permutation().len()`.
    pub fn old_shape(&self, new_shape: &Shape) -> Result<Shape, TensorErrors> {
        if new_shape.rank() != self.permutation.len() {
            return Err(TensorErrors::TransposeIncompatibleRank {
                rank: new_shape.rank(),
                trank: self.permutation.len(),
            });
        }

        let mut old_shape_vec = vec![0; new_shape.rank()];
        let mut count = 0;

        for old_pos in self.permutation.iter() {
            old_shape_vec[*old_pos] = new_shape[count];
            count += 1
        }

        Ok(Shape::new(old_shape_vec))
    }

    /// Returns this transpose applied to `old_index`.
    /// 
    /// This fails if `old_index.len() != self.permutation().len()`.
    pub fn new_index(&self, old_index: &[usize]) -> Result<Vec<usize>, TensorErrors> {
        if old_index.len() != self.permutation.len() {
            return Err(TensorErrors::TransposeIncompatibleRank {
                rank: old_index.len(),
                trank: self.permutation.len(),
            });
        }

        let mut new_index_vec = Vec::with_capacity(old_index.len());

        for old_pos in self.permutation.iter() {
            new_index_vec.push(old_index[*old_pos]);
        }

        Ok(new_index_vec)
    }

    /// Returns the old index that would have been transformed into `new_index` by this transpose.
    /// 
    /// This fails if `new_index.len() != self.permutation().len()`.
    pub fn old_index(&self, new_index: &[usize]) -> Result<Vec<usize>, TensorErrors> {
        if new_index.len() != self.permutation.len() {
            return Err(TensorErrors::TransposeIncompatibleRank {
                rank: new_index.len(),
                trank: self.permutation.len(),
            });
        }

        let mut old_index_vec = vec![0; new_index.len()];
        let mut count = 0;

        for old_pos in self.permutation.iter() {
            old_index_vec[*old_pos] = new_index[count];
            count += 1;
        }

        Ok(old_index_vec)
    }

    /// Returns the inverse transpose.
    pub fn inverse(&self) -> Transpose {
        let mut inv_vec = vec![0; self.permutation.len()];
        let mut count = 0;

        for i in self.permutation.iter() {
            inv_vec[*i] = count;
            count += 1;
        }

        Transpose::new(&inv_vec).unwrap()
    }
}

#[macro_export]
/// Constructs a transpose using the specified permutation.
/// Assumes the permutation is valid so will panic if it is not.
macro_rules! transpose {
    ($($x:expr),*$(,)?) => {
        Transpose::new(&vec![$($x),*]).unwrap()
    };
}
