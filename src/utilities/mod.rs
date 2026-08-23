#[cfg(not(feature = "internal"))]
pub(crate) mod internal_functions;

#[cfg(feature = "internal")]
pub mod internal_functions;

pub mod matrix;
pub mod matrix_slice;
pub mod matrix_slice_mut;
pub mod tensor;
pub mod tensor_slice;
pub mod tensor_slice_mut;
