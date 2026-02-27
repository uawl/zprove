//! `zk_proof` module — Circle STARK proof generation and verification.
//!
//! Split from the original monolithic `zk_proof.rs`.  All public items are
//! re-exported at this level so external callers see no change.

pub mod types;
pub mod batch;
pub mod preprocessed;
pub mod stack_ir;
pub mod lut;
pub mod stage_a;
pub mod memory;
pub mod storage;
pub mod stack_consistency;
pub mod keccak;
pub mod recursive;

pub use types::*;
pub use batch::*;
pub use preprocessed::*;
pub use stack_ir::*;
pub use lut::*;
pub use stage_a::*;
pub use memory::*;
pub use storage::*;
pub use stack_consistency::*;
pub use keccak::*;
pub use recursive::*;
