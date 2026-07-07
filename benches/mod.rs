//! Criterion benchmarks for `tensor_math`.
//!
//! The module tree mirrors `src/`: each top-level source directory has a
//! matching benches subdirectory holding the benchmarks for its public API.

use criterion::criterion_main;

mod utilities;

use utilities::{
    clip_benches, concat_benches, constructors_benches, flip_benches, iter_benches,
    map_benches, pool_benches, reshape_flatten_benches, slice_benches, transpose_benches,
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
);
