#[cfg(not(feature = "internal"))]
pub(crate) mod internal_functions;

#[cfg(feature = "internal")]
pub mod internal_functions;

pub mod matrix;
pub mod tensor;
