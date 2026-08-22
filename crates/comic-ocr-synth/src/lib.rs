//! Synthetic training crops whose labels are true by construction.
//!
//! Stage 1 of `docs/TRAINING_PATH.md`. Two properties make this the load-bearing
//! stage: it is unlimited (100k pairs is compute, not a data-collection
//! project) and it is unencumbered (no provider terms, no corpus agreement, no
//! third party's output), so weights trained here are ours without
//! qualification.
pub mod degrade;
pub mod render;
