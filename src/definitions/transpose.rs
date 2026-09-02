use crate::definitions::errors::TensorErrors;
use crate::definitions::shape::Shape;

#[derive(Debug, Eq, PartialEq)]
pub struct Transpose {
    pub(crate) permutation: Vec<usize>,
    pub(crate) two_cycles: Vec<(usize, usize)>,
}

/// Returns the two-cycles of a permutation.
fn two_cycles_from_permutation(permutation: &Vec<usize>) -> Vec<(usize, usize)> {
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
    pub fn new(permutation: &Vec<usize>) -> Result<Self, TensorErrors> {
        let mut perm_copy = permutation.to_vec();
        perm_copy.sort();

        if perm_copy != (0..permutation.len()).collect::<Vec<usize>>() {
            return Err(TensorErrors::TransposePermutationInvalid);
        }

        Ok(Transpose {
            permutation: permutation.clone(),
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
        Transpose::new(&(0..n).collect()).unwrap()
    }

    /// Swaps two axes.
    /// This fails if either axis is out of bounds.
    pub fn swap_axes(mut self, axis1: usize, axis2: usize) -> Result<Self, TensorErrors> {
        if axis1 >= self.permutation.len() || axis2 >= self.permutation.len() {
            return Err(TensorErrors::TransposePermutationInvalid);
        }

        self.permutation.swap(axis1, axis2);

        Ok(Transpose {
            two_cycles: two_cycles_from_permutation(&self.permutation), // This allows for simplifying the two-cycles after the swap.
            permutation: self.permutation,
        })
    }

    /// Returns this transpose applied to `old_shape`.
    /// This fails if `old_shape.rank() != self.permutation().len()`.
    pub fn new_shape(&self, mut old_shape: Shape) -> Result<Shape, TensorErrors> {
        if old_shape.rank() != self.permutation.len() {
            return Err(TensorErrors::TransposeIncompatibleRank {
                rank: old_shape.rank(),
                trank: self.permutation.len(),
            });
        }

        for cycle in self.two_cycles.iter() {
            old_shape.0.swap(cycle.0, cycle.1);
        }

        Ok(old_shape)
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
    pub fn old_shape(&self, mut new_shape: Shape) -> Result<Shape, TensorErrors> {
        if new_shape.rank() != self.permutation.len() {
            return Err(TensorErrors::TransposeIncompatibleRank {
                rank: new_shape.rank(),
                trank: self.permutation.len(),
            });
        }

        for &(i0, i1) in self.two_cycles.iter().rev() {
            new_shape.0.swap(i0, i1);
        }

        Ok(new_shape)
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

        let mut new_index_vec = old_index.to_vec();

        for &(i0, i1) in self.two_cycles.iter() {
            new_index_vec.swap(i0, i1);
        }

        Ok(new_index_vec)
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

        let mut old_index_vec = vec![0; new_index.len()];
        let mut count = 0;

        for old_pos in self.permutation.iter() {
            old_index_vec[*old_pos] = new_index[count];
            count += 1;
        }

        Ok(old_index_vec)
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
