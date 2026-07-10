//! Benchmarks for items declared in `src/math/`.

mod eigen;
mod row_echelon;
mod dot;
mod fft;
mod fft_internal;
mod contract_mul;
mod corr_conv;
mod householder;
mod hessenberg;
mod det_inv;
mod transformation_rank;
mod sum;

pub use eigen::eigen_benches;
pub use row_echelon::row_echelon_benches;
pub use dot::dot_benches;
pub use fft::fft_benches;
pub use fft_internal::fft_internal_benches;
pub use contract_mul::contract_mul_benches;
pub use corr_conv::corr_conv_benches;
pub use householder::householder_benches;
pub use hessenberg::hessenberg_benches;
pub use det_inv::det_inv_benches;
pub use transformation_rank::transformation_rank_benches;
pub use sum::sum_benches;