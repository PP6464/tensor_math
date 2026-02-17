#[cfg(test)]
mod householder_tests {
    use crate::definitions::shape::Shape;
use float_cmp::{ApproxEq, F64Margin, FloatMargin};
    use num::complex::{Complex64, ComplexFloat};
    use num::FromPrimitive;
    use crate::definitions::matrix::Matrix;
    use crate::definitions::tensor::Tensor;
    use crate::shape;
    use crate::utilities::matrix::eye;

    #[test]
    fn test_householder() {
        let m1: Matrix<f64> = Tensor::<i32>::new(
            &shape![9],
            vec![
                4, 1, 1,
                1, 3, 0,
                1, 0, 2,
            ],
        )
            .unwrap()
            .iter()
            .map(|x| f64::from_i32(*x).unwrap())
            .collect::<Matrix<f64>>()
            .reshape(3, 3)
            .unwrap();

        let (q1, r1) = m1.householder();

        for i in 0..r1.shape[0] {
            if i >= r1.shape[1] - 1 {
                continue
            }

            assert!(r1.slice(i+1..r1.shape[0], i..i+1).unwrap().iter().all(|x| { x.abs() <= 1e-10 }));
        }
        assert!(q1.contract_mul(&r1).unwrap().tensor.approx_eq(m1.clone().tensor, F64Margin::default().epsilon(1e-10)));

        let m2: Matrix<Complex64> = Tensor::<Complex64>::new(
            &shape![3, 2],
            vec![
                Complex64 { re: 4.0, im: 1.0 }, Complex64 { re: -5.0, im: -2.0 },
                Complex64 { re: 5.0, im: -4.0 }, Complex64 { re: 5.0, im: 3.0 },
                Complex64 { re: 0.0, im: 0.0 }, Complex64 { re: 1.0, im: -1.0 },
            ],
        ).unwrap().try_into().unwrap();
        let (q2, r2) = m2.householder();

        for i in 0..r2.shape[0] {
            if i >= r2.shape[1] - 1 {
                continue
            }

            assert!(r2.slice(i+1..r2.shape[0], i..i+1).unwrap().iter().all(|x| x.abs() <= 1e-10));
        }
        assert!(q2.contract_mul(&r2).unwrap().tensor.approx_eq(m2.tensor, F64Margin::default().epsilon(1e-10)));

        let m3: Matrix<Complex64> = Tensor::<Complex64>::new(
            &shape![2, 3],
            vec![
                Complex64 { re: -4.0, im: -1.0 }, Complex64 { re: 5.0, im: -3.0 }, Complex64 { re: 2.0, im: -4.0 },
                Complex64 { re: -5.0, im: 2.0 }, Complex64 { re: 2.0, im: -1.0 }, Complex64 { re: 4.0, im: -1.0 },
            ],
        ).unwrap().try_into().unwrap();
        let (q3, r3) = m3.householder();

        for i in 0..(r3.shape[0] - 1) {
            if i >= r3.shape[1] - 1 {
                continue
            }
            assert!(r3.slice(i+1..r3.shape[0], i..i+1).unwrap().iter().all(|x| x.abs() <= 1e-10));
        }
        assert!(q3.contract_mul(&r3).unwrap().tensor.approx_eq(m3.tensor, F64Margin::default().epsilon(1e-10)));
    }

    #[test]
    fn householder_of_zeros() {
        let m1 = Matrix::<f64>::zeros(3, 3).unwrap();
        let m2 = Matrix::<Complex64>::zeros(3, 3).unwrap();

        assert_eq!(m1.householder(), (eye(3).unwrap(), m1));
        assert_eq!(m2.householder(), (eye(3).unwrap(), m2));
    }
}