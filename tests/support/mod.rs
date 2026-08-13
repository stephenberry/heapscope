//! Test-only helpers shared between integration tests.
//!
//! Compiled into each test binary that declares `mod support;`, so not every
//! item is used by every binary; `dead_code` is allowed throughout.

pub mod dhat;
pub mod display;
pub mod fixture;
pub mod json;
pub mod native;
pub mod page;
pub mod snapshot;
