pub mod definitions;
pub mod math;
pub mod utilities;

extern crate self as tensor_math;

#[cfg(test)]
mod test_helpers {
    use std::panic::{catch_unwind, UnwindSafe};

    pub fn assert_panics(closure: impl FnOnce() + UnwindSafe) {
        assert!(catch_unwind(closure).is_err());
    }
}
