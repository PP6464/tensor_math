//! Criterion benchmarks for `tensor_math`.
//!
//! The module tree mirrors `src/`: each top-level source directory has a
//! matching benches subdirectory holding the benchmarks for its public API.

use criterion::criterion_main;

mod utilities;
mod math;

use utilities::{
    clip_benches, concat_benches, constructors_benches, flip_benches, iter_benches,
    map_benches, pool_benches, reshape_flatten_benches, slice_benches, transpose_benches,
};

use math::{
    contract_mul_benches, corr_conv_benches, eigen_benches, fft_benches, fft_internal_benches,
    dot_benches, householder_benches, row_echelon_benches, hessenberg_benches,
    det_inv_benches, transformation_rank_benches, sum_benches,
};

criterion_main!(
    concat_benches,
    transpose_benches,
    clip_benches,
    map_benches,
    flip_benches,
    pool_benches,
    slice_benches,
    reshape_flatten_benches,
    iter_benches,
    constructors_benches,
    eigen_benches,
    row_echelon_benches,
    dot_benches,
    fft_benches,
    fft_internal_benches,
    contract_mul_benches,
    corr_conv_benches,
    householder_benches,
    hessenberg_benches,
    det_inv_benches,
    transformation_rank_benches,
    sum_benches,
);
