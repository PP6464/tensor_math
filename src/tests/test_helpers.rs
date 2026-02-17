use std::panic::{catch_unwind, UnwindSafe};

pub(crate) fn assert_panics(closure: impl FnOnce() + UnwindSafe) {
    assert!(catch_unwind(closure).is_err());
}