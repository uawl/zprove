//! `zk_proof` module — Circle STARK proof generation and verification.
//!
//! Split from the original monolithic `zk_proof.rs`.  All public items are
//! re-exported at this level so external callers see no change.

pub mod batch;
pub mod fri_air;
pub mod keccak;
pub mod lut;
pub mod m31ext3_air;
pub mod memory;
pub mod merkle_air;
pub mod poseidon2_air;
pub mod preprocessed;
pub mod recursive;
pub mod stack_consistency;
pub mod stack_ir;
pub mod stack_rw;
pub mod stage_a;
pub mod stark_verifier_air;
pub mod storage;
pub mod types;

pub use batch::*;
pub use fri_air::*;
pub use keccak::*;
pub use lut::*;
pub use m31ext3_air::*;
pub use memory::*;
pub use merkle_air::*;
pub use poseidon2_air::*;
pub use preprocessed::*;
pub use recursive::*;
pub use stack_consistency::*;
pub use stack_ir::*;
pub use stack_rw::*;
pub use stage_a::*;
pub use stark_verifier_air::*;
pub use storage::*;
pub use types::*;
