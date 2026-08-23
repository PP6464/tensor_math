pub mod errors;
pub mod matrix;
pub mod matrix_slice;
pub mod shape;

#[cfg(not(feature = "internal"))]
pub(crate) mod strides;

#[cfg(feature = "internal")]
pub mod strides;

pub mod macros;
pub mod matrix_slice_mut;
pub mod tensor;
pub mod tensor_slice;
pub mod tensor_slice_mut;
pub mod traits;
pub mod transpose;
