//! Benchmarks for items declared in `src/utilities/`.

mod clip;
mod concat;
mod constructors;
mod flip;
mod iter;
mod pool;
mod reshape_flatten;
mod slice;
mod transpose;
mod map;

pub use clip::clip_benches;
pub use concat::concat_benches;
pub use constructors::constructors_benches;
pub use flip::flip_benches;
pub use iter::iter_benches;
pub use map::map_benches;
pub use pool::pool_benches;
pub use reshape_flatten::reshape_flatten_benches;
pub use slice::slice_benches;
pub use transpose::transpose_benches;
