use crate::definitions::shape::Shape;
use std::ops::Index;

/// Cache the strides required to index the tensor.
/// addr = dot_vectors(index_vector, strides)
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct Strides(pub(crate) Vec<usize>);
impl Strides {
    pub(crate) fn from_shape(shape: &Shape) -> Strides {
        let mut strides: Vec<usize> = vec![1; shape.rank()];

        for i in 1..shape.rank() {
            let current_index = shape.rank() - 1 - i;
            strides[current_index] = shape[current_index + 1] * strides[current_index + 1];
        }

        Strides(strides)
    }
}
impl Index<usize> for Strides {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
impl Into<Vec<usize>> for Strides {
    fn into(self) -> Vec<usize> {
        self.0
    }
}
