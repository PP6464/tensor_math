//! Criterion benchmarks for `tensor_math`.
//!
//! The module tree mirrors `src/`: each top-level source directory has a
//! matching benches subdirectory holding the benchmarks for its public API.

use criterion::criterion_main;

mod utilities;

use utilities::{concat_benches, transpose_benches};

criterion_main!(concat_benches, transpose_benches);
