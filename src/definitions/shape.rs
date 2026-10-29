use crate::definitions::errors::TensorErrors;
use crate::definitions::strides::Strides;
use crate::utilities::internal_functions::dot_vectors;
use std::fmt::Display;
use std::ops::{Index, IndexMut};

/*
--------------------------------------------
* Shape definition
--------------------------------------------
*/

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Shape(pub(crate) Vec<usize>);

impl Shape {
    /// Constructs a new shape.
    pub fn new(shape: Vec<usize>) -> Self {
        Shape(shape)
    }

    /// This returns the number of dimensions of the shape.
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// The number of elements a tensor of this shape should have.
    pub fn element_count(&self) -> usize {
        self.0.iter().product()
    }

    /// This gives a list of all indices that are valid for a tensor with this shape.
    pub fn indices(&self) -> Vec<Vec<usize>> {
        (0..self.element_count())
            .map(|i| self.tensor_index(i).unwrap())
            .collect()
    }

    /// Sets the value at a given axis.
    /// This fails if the axis is out of bounds.
    pub fn set(&mut self, axis: usize, value: usize) -> Result<(), TensorErrors> {
        if axis >= self.rank() {
            return Err(TensorErrors::AxisOutOfBounds {
                axis,
                rank: self.rank(),
            });
        }

        self.0[axis] = value;

        Ok(())
    }

    /// Gets the length at a specified axis, if it exists, otherwise returns None.
    pub fn get(&self, axis: usize) -> Option<usize> {
        self.0.get(axis).copied()
    }

    /// This gives the address for a corresponding shape index.
    /// This fails if the index is out of bounds.
    pub fn address(&self, index: &[usize]) -> Result<usize, TensorErrors> {
        if index.len() != self.rank() {
            return Err(TensorErrors::IndicesInvalidForRank(
                index.len(),
                self.rank(),
            ));
        }

        for (i, &v) in index.iter().enumerate() {
            if v >= self[i] {
                return Err(TensorErrors::IndexOutOfBounds {
                    index: v,
                    axis: i,
                    length: self[i],
                });
            }
        }

        Ok(unsafe { self.address_unchecked(index) })
    }

    /// This gives the address for a corresponding shape index and trusts validity
    pub(crate) unsafe fn address_unchecked(&self, index: &[usize]) -> usize {
        dot_vectors(&Strides::from_shape(self).0, index)
    }

    /// Computes the tensor index for a given address.
    /// This fails if the address is out of bounds.
    pub fn tensor_index(&self, address: usize) -> Result<Vec<usize>, TensorErrors> {
        if address >= self.element_count() {
            return Err(TensorErrors::AddressOutOfBounds(address));
        }

        Ok(unsafe { self.tensor_index_unchecked(address) })
    }

    /// Computes the tensor index for a given address and trusts validity.
    pub(crate) unsafe fn tensor_index_unchecked(&self, address: usize) -> Vec<usize> {
        let mut index_vec = Vec::with_capacity(self.rank());
        let mut remainder = address;
        let strides = Strides::from_shape(self);

        for j in strides.0.iter() {
            let floored_div = remainder / j;
            index_vec.push(floored_div);
            remainder %= j;
        }

        index_vec
    }
}

/*
--------------------------------------------
* Shape indexing
--------------------------------------------
*/

impl Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for Shape {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

/*
--------------------------------------------
* Shape conversions
--------------------------------------------
*/

impl FromIterator<usize> for Shape {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        Shape(Vec::from_iter(iter))
    }
}

impl From<Vec<usize>> for Shape {
    fn from(shape: Vec<usize>) -> Self {
        Shape::new(shape)
    }
}

/*
--------------------------------------------
* Shape display
--------------------------------------------
*/

impl Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(")?;
        for (i, dim) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", dim)?;
        }
        write!(f, ")")
    }
}


#[cfg(test)]
mod shape_definition_tests {
    use tensor_math::definitions::errors::TensorErrors;
    use tensor_math::definitions::shape::Shape;
    use tensor_math::definitions::strides::Strides;
    use tensor_math::shape;

    #[test]
    fn can_create_empty_shape() {
        let shape = shape![];
        assert_eq!(shape.rank(), 0);
    }

    #[test]
    #[should_panic]
    fn invalid_index_empty_shape() {
        let shape = shape![];
        shape[0];
    }

    #[test]
    #[should_panic]
    fn invalid_index_mut_empty_shape() {
        let mut shape = shape![];
        shape[0] = 5;
    }

    #[test]
    #[should_panic]
    fn invalid_index() {
        shape![1, 3][2];
    }

    #[test]
    fn valid_index() {
        let shape = shape![1, 3];
        assert_eq!(shape[1], 3);
    }

    #[test]
    fn valid_index_mut() {
        let mut shape = shape![1, 3];
        shape[1] = 5;
        assert_eq!(shape[1], 5);
    }

    #[test]
    #[should_panic]
    fn invalid_index_mut() {
        let mut shape = shape![1, 3];
        shape[2] = 5;
    }

    #[test]
    fn strides() {
        let shape = shape![2, 3, 4];
        let index_products = Strides::from_shape(&shape);
        assert_eq!(vec![12, 4, 1], index_products.0);
    }

    #[test]
    fn display_impl() {
        let shape = shape![2, 3, 4];
        assert_eq!(format!("{}", shape), "(2, 3, 4)");
    }

    #[test]
    fn set_at_axis() {
        let mut shape = shape![1, 1, 1];
        shape.set(2, 2).unwrap();
        assert_eq!(shape, shape![1, 1, 2]);
        let err = shape.set(3, 2).unwrap_err();
        assert!(matches!(err, TensorErrors::AxisOutOfBounds { .. }));
    }

    #[test]
    fn get_is_safe_index() {
        let shape = shape![1, 2, 3];
        assert_eq!(shape.get(1), Some(2));
        assert_eq!(shape.get(5), None);
    }

    #[test]
    fn address() {
        let shape = shape![];
        let addr = shape.address(vec![].as_slice()).unwrap();
        assert_eq!(addr, 0);
        let shape = shape![4, 2, 3];
        let addr = shape.address(vec![2, 1, 2].as_slice()).unwrap();
        assert_eq!(addr, 17);
        let err = shape.address(vec![1, 2].as_slice()).unwrap_err();
        assert!(matches!(err, TensorErrors::IndicesInvalidForRank(_, _)));
        let err = shape.address(vec![4, 0, 0].as_slice()).unwrap_err();
        assert!(matches!(err, TensorErrors::IndexOutOfBounds {
            index: _,
            axis: _,
            length: _,
        }));
    }

    #[test]
    fn tensor_index() {
        let shape = shape![];
        let index = shape.tensor_index(0).unwrap();
        assert_eq!(index, Vec::<usize>::new());

        let shape = shape![0];
        let err = shape.tensor_index(0).unwrap_err();
        assert!(matches!(err, TensorErrors::AddressOutOfBounds(_)));

        let shape = shape![4, 2, 3];
        let index = shape.tensor_index(17).unwrap();
        assert_eq!(index, vec![2, 1, 2]);

        let err = shape.tensor_index(24).unwrap_err();
        assert!(matches!(err, TensorErrors::AddressOutOfBounds(_)));
    }

    #[test]
    fn conversion_from_vec() {
        let shape = shape![2, 3, 4];
        let vec: Vec<usize> = shape.0.clone();
        let shape_from_vec: Shape = vec.clone().into();
        assert_eq!(shape, shape_from_vec);
    }

    #[test]
    fn conversion_from_iterator() {
        let shape = shape![2, 3, 4];
        let vec: Vec<usize> = shape.0.clone();
        let shape_from_iter: Shape = vec.into_iter().collect();
        assert_eq!(shape, shape_from_iter);
    }

    #[test]
    fn indices_for_empty_shape() {
        let shape = shape![];
        let indices = shape.indices();
        assert_eq!(indices, vec![vec![]]);
    }

    #[test]
    fn indices_for_non_empty_shape() {
        let shape = shape![2, 3];
        let indices = shape.indices();
        assert_eq!(indices, vec![vec![0, 0], vec![0, 1], vec![0, 2], vec![1, 0], vec![1, 1], vec![1, 2]]);
    }

    #[test]
    fn indices_for_shape_with_zero_dimension() {
        let shape = shape![2, 0, 3];
        let indices = shape.indices();
        assert_eq!(indices, Vec::<Vec<usize>>::new());
    }
}
