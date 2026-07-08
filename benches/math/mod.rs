//! Benchmarks for items declared in `src/math/`.

mod eigen;
mod row_echelon;
mod dot;
mod fft;

pub use eigen::eigen_benches;
pub use row_echelon::row_echelon_benches;
pub use dot::dot_benches;
pub use fft::fft_benches;