//! Benchmarks for items declared in `src/math/`.

mod eigen;
mod row_echelon;

pub use eigen::eigen_benches;
pub use row_echelon::row_echelon_benches;