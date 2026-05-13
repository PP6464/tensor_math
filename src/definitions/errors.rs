use crate::definitions::shape::Shape;
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum TensorErrors {
    #[error("Shape vector indicates a size different to that of the elements provided")]
    ShapeSizeDoesNotMatch,
    #[error("Shape vector is not 1 on axis {0} so cannot flatten on this axis")]
    AxisIsNotOne(usize),
    #[error("Dimension {axis} greater than rank of shape: {rank}")]
    AxisOutOfBounds { axis: usize, rank: usize },
    #[error("Index {index} out of bounds for axis {axis} of length {length}")]
    IndexOutOfBounds {
        index: usize,
        axis: usize,
        length: usize,
    },
    #[error("Shapes are not compatible")]
    ShapesIncompatible,
    #[error("{0} indices not valid for tensor of rank {1}")]
    IndicesInvalidForRank(usize, usize),
    #[error("Transposition permutation invalid")]
    TransposePermutationInvalid,
    #[error("Determinant is zero")]
    DeterminantZero,
    #[error("Ranks do not match: {0} and {1}")]
    RanksDoNotMatch(usize, usize),
    #[error("Matrix is not square")]
    NonSquareMatrix,
    #[error("Slice indices out of bounds at axis {axis} length: {length}, range: {start}..{end}")]
    SliceIndicesOutOfBounds {
        axis: usize,
        length: usize,
        start: usize,
        end: usize,
    },
    #[error("Slice shape {slice_shape} incompatible with tensor shape {tensor_shape}")]
    SliceIncompatibleShape {
        slice_shape: Shape,
        tensor_shape: Shape,
    },
    #[error("Standard deviation ({0}) is not positive")]
    NonPositiveSigma(f64),
    #[error("Covariance matrix is not positive definite")]
    CovMatNotPositiveDefinite,
    #[error("Standard deviations are not all positive")]
    SigmaListNotAllPositive,
    #[error("The number of standard deviations ({0}) does not match the rank of the result ({1})")]
    SigmaListLengthIncompatible(usize, usize),
    #[error("Range has max ({max}) <= min ({min})")]
    InvalidInterval { min: f64, max: f64 },
    #[error("Polynomial has a leading coefficient that is 0")]
    PolynomialLeadingCoefficientZero,
    #[error("Transpose for tensor of rank {trank} is incompatible for tensor of rank {rank}")]
    TransposeIncompatibleRank { rank: usize, trank: usize },
    #[error("Address {0} out of bounds")]
    AddressOutOfBounds(usize),
    #[error("Shape contains zero where it cannot for this specific operation")]
    ShapeContainsZero,
    #[error("Tensor has rank zero which is not allowed for the operation {op}")]
    RankZero { op: &'static str },
}
