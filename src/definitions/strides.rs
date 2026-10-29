use crate::definitions::shape::Shape;
use std::ops::Index;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
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

#[cfg(test)]
mod strides_tests {
    #[test]
    fn from_shape() {
        use crate::definitions::shape::Shape;
        use crate::definitions::strides::Strides;

        let shape = Shape(vec![2, 3, 4]);
        let strides = Strides::from_shape(&shape);
        assert_eq!(strides.0, vec![12, 4, 1]);
    }

    #[test]
    fn from_empty_shape() {
        use crate::definitions::shape::Shape;
        use crate::definitions::strides::Strides;

        let shape = Shape(vec![]);
        let strides = Strides::from_shape(&shape);
        assert_eq!(strides.0, vec![]);
    }

    #[test]
    fn from_shape_with_zero_dimension() {
        use crate::definitions::shape::Shape;
        use crate::definitions::strides::Strides;

        let shape = Shape(vec![2, 0, 4]);
        let strides = Strides::from_shape(&shape);
        assert_eq!(strides.0, vec![0, 4, 1]);
    }

    #[test]
    fn index() {
        use crate::definitions::shape::Shape;
        use crate::definitions::strides::Strides;

        let shape = Shape(vec![2, 3, 4]);
        let strides = Strides::from_shape(&shape);
        assert_eq!(strides[0], 12);
        assert_eq!(strides[1], 4);
        assert_eq!(strides[2], 1);
    }

    #[test]
    #[should_panic]
    fn invalid_index() {
        use crate::definitions::shape::Shape;
        use crate::definitions::strides::Strides;

        let shape = Shape(vec![2, 3, 4]);
        let strides = Strides::from_shape(&shape);
        strides[3];
    }
}
