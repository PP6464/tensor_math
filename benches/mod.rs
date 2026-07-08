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
    eigen_benches, fft_benches,
};
use crate::math::{dot_benches, row_echelon_benches};

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
);
