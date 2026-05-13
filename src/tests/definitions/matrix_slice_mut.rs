#[cfg(test)]
mod matrix_slice_mut_tests {
    use crate::definitions::matrix::Matrix;
    use crate::definitions::traits::{IntoMatrix, TryIntoMatrix};

    #[test]
    fn empty_slice_mut() {
        let mut m1 = Matrix::<usize>::new(0, 0, vec![]).unwrap();
        let slice = m1.slice_mut(0..0, 0..0).unwrap();
        assert_eq!(slice.end, (0, 0));
        assert_eq!(slice.start, (0, 0));
    }

    #[test]
    fn index_mut_slice() {
        let mut m1 = Matrix::new(
            3, 3,
            (0..9).collect(),
        ).unwrap();

        let mut slice = m1.slice_mut(1..3, 1..3).unwrap();

        assert_eq!(slice[&[0, 0]], 4);
        assert_eq!(slice[(0, 1)], 5);

        slice[(1, 0)] = -1;
        slice[(1, 1)] = -2;

        let ans = Matrix::new(
            3, 3,
            vec![
                0, 1, 2,
                3, 4, 5,
                6, -1, -2,
            ]
        ).unwrap();

        assert_eq!(m1, ans);
    }

    #[test]
    fn convert_into_mat() {
        let mut m1 = Matrix::new(
            3, 3,
            (0..9).collect(),
        ).unwrap();

        let mut slice = m1.slice_mut(1..3, 1..3).unwrap();
        slice[(0, 1)] = 10;

        let ans = Matrix::new(
            2, 2,
            vec![
                4, 10,
                7, 8,
            ]
        ).unwrap();

        assert_eq!(slice.into_matrix(), ans);

        let slice = m1.slice_mut(1..3, 1..3).unwrap();

        assert_eq!(slice.try_into_matrix().unwrap(), ans);
    }

    #[test]
    fn get_in_slice() {
        let mut m1 = Matrix::new(
            3, 3,
            (0..9).collect(),
        ).unwrap();

        let slice = m1.slice_mut(1..3, 1..3).unwrap();

        assert_eq!(slice.get((0, 0)), Some(&4));
        assert_eq!(slice.get((2, 2)), None);
    }
}
