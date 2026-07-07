pub mod errors;
pub mod matrix;
pub mod matrix_slice_mut;
pub mod shape;

#[cfg(not(feature = "internal"))]
pub(crate) mod strides;

#[cfg(feature = "internal")]
pub mod strides;

pub mod tensor;
pub mod tensor_slice_mut;
pub mod traits;
pub mod transpose;
