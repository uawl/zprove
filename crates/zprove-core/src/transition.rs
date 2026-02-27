//! EVM instruction state-transition proofs.
//!
//! Each EVM instruction is modeled as a state transition:
//!   `(opcode, stack_inputs) → stack_outputs`
//!
//! An [`InstructionTransitionProof`] bundles the opcode, inputs, outputs,
//! and a Hilbert-style semantic proof that the transition is correct.
//!
//! A [`TransactionProof`] is a sequence of instruction proofs covering
//! every step of a transaction's execution.

use crate::semantic_proof::ProofRow;
use crate::semantic_proof::{
  Proof, WFF, compile_proof, infer_proof, prove_add, prove_addmod, prove_and, prove_byte,
  prove_div, prove_eq, prove_exp, prove_gt, prove_iszero, prove_lt, prove_mod, prove_mul,
  prove_mulmod, prove_not, prove_or, prove_sar, prove_sdiv, prove_sgt, prove_shl, prove_shr,
  prove_signextend, prove_slt, prove_smod, prove_sub, prove_xor, verify_compiled, wff_add,
  wff_addmod, wff_and, wff_byte, wff_div, wff_eq, wff_exp, wff_gt, wff_iszero, wff_lt, wff_mod,
  wff_mul, wff_mulmod, wff_not, wff_or, wff_sar, wff_sdiv, wff_sgt, wff_shl, wff_shr,
  wff_signextend, wff_slt, wff_smod, wff_sub, wff_xor,
};
use crate::zk_proof::{
  BatchInstructionMeta, BatchProofRowsManifest, CircleStarkConfig, CircleStarkProof,
  KeccakConsistencyProof, MemoryConsistencyProof, StackConsistencyProof, StorageConsistencyProof,
  RECEIPT_BIND_TAG_LUT, RECEIPT_BIND_TAG_STACK,
  collect_byte_table_queries_from_rows, compute_wff_opcode_digest,
  make_batch_receipt_binding_public_values,
  make_receipt_binding_public_values, prove_batch_lut_with_prep, prove_keccak_consistency,
  prove_memory_consistency, prove_stack_consistency, prove_storage_consistency,
  prove_stack_ir_with_prep, setup_batch_proof_rows_preprocessed, setup_proof_rows_preprocessed,
  validate_manifest_rows, verify_batch_lut_with_prep, verify_keccak_consistency,
  verify_memory_consistency, verify_stack_consistency, verify_storage_consistency,
  verify_stack_ir_with_prep, validate_keccak_memory_cross_check,
  StateCommitment, commit_vm_state, prove_link_stark, verify_link_stark,
};
use p3_uni_stark::PreprocessedVerifierKey;
use revm::bytecode::opcode;

// ============================================================
// Types
// ============================================================

/// A single memory read or write claim from one EVM instruction.
///
/// Witnesses that "at global rw_counter N, address `addr` was [read/written]
/// with value `value`".  The `MemoryConsistencyAir` checks that the multiset
/// of all claims across the entire transaction is consistent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemAccessClaim {
  /// Global read-write counter (monotone across all instructions).
  pub rw_counter: u64,
  /// 32-byte aligned word address in EVM memory (in bytes, multiple of 32
  /// for MLOAD/MSTORE, 32-byte chunk containing the byte for MSTORE8).
  pub addr: u64,
  /// `true` for a write, `false` for a read.
  pub is_write: bool,
  /// The 32-byte word value.
  pub value: [u8; 32],
}

/// A single EVM storage access claim (SLOAD / SSTORE).
///
/// `contract` is the 20-byte address of the contract whose storage is accessed.
/// `slot`     is the 32-byte storage key (EVM U256, big-endian).
/// `value`    is the 32-byte storage value (EVM U256, big-endian).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageAccessClaim {
  /// Global read-write counter (monotone across all instructions).
  pub rw_counter: u64,
  /// 20-byte contract address.
  pub contract: [u8; 20],
  /// 32-byte storage slot key.
  pub slot: [u8; 32],
  /// `true` for SSTORE, `false` for SLOAD.
  pub is_write: bool,
  /// The 32-byte storage value.
  pub value: [u8; 32],
}

/// Return/revert data witness for RETURN and REVERT opcodes.
///
/// Records the memory slice returned by the terminating instruction so that
/// the verifier can confirm what the transaction produced without re-executing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReturnDataClaim {
  /// `true` for REVERT, `false` for RETURN.
  pub is_revert: bool,
  /// Memory byte offset of the return data.
  pub offset: u64,
  /// Length of the return data in bytes.
  pub size: u64,
  /// Actual return data bytes (length == `size`).
  pub data: Vec<u8>,
}

/// A single EVM stack push or pop claim from one instruction.
///
/// Every instruction that touches the stack emits a sequence of claims in
/// execution order (pops first, then pushes).  `StackConsistencyAir` verifies
/// that every pop value matches the most recent push to the same depth, and
/// that the depth sequence is monotone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StackAccessClaim {
  /// Global push/pop counter (monotone across all instructions).
  pub rw_counter: u64,
  /// Stack depth *after* this operation (0 = empty stack).
  pub depth_after: usize,
  /// `true` for a push, `false` for a pop.
  pub is_push: bool,
  /// The 32-byte word value pushed onto or popped from the stack.
  pub value: [u8; 32],
}

/// A witness for the KECCAK256 opcode.
///
/// Records the memory slice that was hashed and the resulting 32-byte digest
/// so that verifiers can re-check `keccak256(input_bytes) == output_hash`
/// without re-executing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeccakClaim {
  /// Memory byte offset of the input data (from `stack_inputs[0]`).
  pub offset: u64,
  /// Length of the input data in bytes (from `stack_inputs[1]`).
  pub size: u64,
  /// Raw input bytes (length == `size`).
  pub input_bytes: Vec<u8>,
  /// keccak256(input_bytes) — the 32-byte digest pushed to the stack.
  pub output_hash: [u8; 32],
}

/// A witness for a sub-call or contract-creation opcode.
///
/// Records the callee, value, return data, and success flag so the verifier
/// can confirm the sub-call result matches the success value pushed onto the
/// stack at batch level.
///
/// EVM maximum CALL nesting depth (EIP-150 gas limit enforces ≤ 1024).
pub const MAX_CALL_DEPTH: u16 = 1024;

/// `inner_proof` is `None` at Level-0 (oracle witness): the claim is accepted
/// without recursively verifying the callee trace. A non-None value enables
/// Level-1 recursive verification.
#[derive(Debug, Clone)]
pub struct SubCallClaim {
  /// Opcode: CALL, CALLCODE, DELEGATECALL, STATICCALL, CREATE, or CREATE2.
  pub opcode: u8,
  /// Callee address (20 bytes). For CREATE/CREATE2 this is the deployed address.
  pub callee: [u8; 20],
  /// ETH transferred (big-endian U256). Zero for DELEGATECALL/STATICCALL.
  pub value: [u8; 32],
  /// Return data bytes from the callee (cloned at call exit).
  pub return_data: Vec<u8>,
  /// Whether the sub-call succeeded (`true`) or reverted (`false`).
  pub success: bool,
  /// EVM call nesting depth at which this sub-call was made (0 = top-level TX).
  pub depth: u16,
  /// Optional recursive proof of the callee execution trace (Level-1+).
  pub inner_proof: Option<Box<TransactionProof>>,
}

/// A witness for external-state-reading opcodes: BLOCKHASH, EXTCODESIZE, EXTCODEHASH.
///
/// Records the query key (block number or account address) and the returned
/// value so the verifier can re-check consistency with the external state at
/// batch level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalStateClaim {
  /// The opcode that produced this claim.
  /// Supported: BLOCKHASH, EXTCODESIZE, EXTCODEHASH.
  pub opcode: u8,
  /// For BLOCKHASH: queried block number (32-byte big-endian U256).
  /// For EXTCODESIZE / EXTCODEHASH: queried contract address (20 bytes,
  /// zero-padded to 32 in big-endian).
  pub key: [u8; 32],
  /// The 32-byte value pushed to the stack.
  pub output_value: [u8; 32],
}

/// A witness for environment-reading opcodes: CALLER, CALLVALUE, CALLDATALOAD, CALLDATASIZE.
///
/// Records the value pushed to the stack so the verifier can re-check
/// consistency with the original call context at batch level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallContextClaim {
  /// The opcode that produced this claim.
  /// Supported: CALLER, CALLVALUE, CALLDATALOAD, CALLDATASIZE, PC, MSIZE, GAS.
  pub opcode: u8,
  /// For CALLDATALOAD: byte offset into calldata (from the stack input).
  /// Zero for all other opcodes.
  pub calldata_offset: u64,
  /// The 32-byte value pushed to the stack.
  pub output_value: [u8; 32],
}

/// Static block and transaction context captured at execution time.
///
/// These fields act as **public inputs** to the ZK proof: the verifier can
/// reconstruct them independently from the block header and transaction data,
/// then assert that every matching `CallContextClaim` in the trace is
/// consistent with these values.
///
/// All values are stored as 32-byte big-endian EVM words.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BlockTxContext {
  /// COINBASE (0x41): block beneficiary address, zero-padded to 32 bytes.
  pub coinbase: [u8; 32],
  /// TIMESTAMP (0x42): block timestamp as U256 big-endian.
  pub timestamp: [u8; 32],
  /// NUMBER (0x43): block number as U256 big-endian.
  pub block_number: [u8; 32],
  /// DIFFICULTY / PREVRANDAO (0x44): post-Merge = prevrandao (B256),
  /// pre-Merge = difficulty (U256).
  pub prevrandao: [u8; 32],
  /// GASLIMIT (0x45): block gas limit as U256 big-endian.
  pub gas_limit: [u8; 32],
  /// CHAINID (0x46): chain identifier as U256 big-endian.
  pub chain_id: [u8; 32],
  /// BASEFEE (0x48): block base fee as U256 big-endian.
  pub basefee: [u8; 32],
  /// ORIGIN (0x32): transaction origin address, zero-padded to 32 bytes.
  pub origin: [u8; 32],
  /// GASPRICE (0x3a): effective gas price as U256 big-endian.
  pub gas_price: [u8; 32],
}

/// A proof that one EVM instruction correctly transforms the stack.
#[derive(Debug, Clone)]
pub struct InstructionTransitionProof {
  /// The opcode executed (e.g. 0x01 = ADD).
  pub opcode: u8,
  /// Program counter before execution.
  pub pc: usize,
  /// Values consumed from the stack (big-endian `[u8; 32]` each).
  pub stack_inputs: Vec<[u8; 32]>,
  /// Values produced to the stack.
  pub stack_outputs: Vec<[u8; 32]>,
  /// Hilbert-style proof of semantic correctness.
  /// `None` for structural operations (PUSH/POP/DUP/SWAP/STOP/JUMPDEST).
  pub semantic_proof: Option<Proof>,
  /// Memory access claims from this instruction.
  /// Non-empty for MLOAD, MSTORE, MSTORE8.
  pub memory_claims: Vec<MemAccessClaim>,
  /// Storage access claims from this instruction.
  /// Non-empty for SLOAD, SSTORE.
  pub storage_claims: Vec<StorageAccessClaim>,
  /// Stack push/pop claims from this instruction.
  /// Non-empty for every instruction that touches the stack.
  pub stack_claims: Vec<StackAccessClaim>,
  /// Return/revert data witness.  `Some` only for RETURN and REVERT.
  pub return_data_claim: Option<ReturnDataClaim>,
  /// Call-context witness.  `Some` for CALLER, CALLVALUE, CALLDATALOAD, CALLDATASIZE.
  pub call_context_claim: Option<CallContextClaim>,
  /// KECCAK256 preimage witness.  `Some` only for KECCAK256.
  pub keccak_claim: Option<KeccakClaim>,
  /// External-state witness.  `Some` for BLOCKHASH, EXTCODESIZE, EXTCODEHASH.
  pub external_state_claim: Option<ExternalStateClaim>,
  /// Sub-call / create witness.  `Some` for CALL, CALLCODE, DELEGATECALL,
  /// STATICCALL, CREATE, CREATE2.
  pub sub_call_claim: Option<SubCallClaim>,
}

/// Complete proof for all steps of a transaction execution.
#[derive(Debug, Clone)]
pub struct TransactionProof {
  pub steps: Vec<InstructionTransitionProof>,
  /// Block and transaction context values, used as public inputs to verify
  /// that `CallContextClaim`s produced during execution are consistent
  /// with the actual on-chain context.
  pub block_tx_context: BlockTxContext,
}

#[derive(Debug, Clone)]
pub struct VmState {
  pub opcode: u8,
  pub pc: usize,
  pub sp: usize,
  pub stack: Vec<[u8; 32]>,
  pub memory_root: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessDomain {
  Stack,
  Memory,
  Storage,
  Calldata,
  Returndata,
  Code,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessKind {
  Read,
  Write,
  ReadWrite,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessRecord {
  pub rw_counter: u64,
  pub domain: AccessDomain,
  pub kind: AccessKind,
  pub addr: u64,
  pub width: u32,
  pub value_before: Option<[u8; 32]>,
  pub value_after: Option<[u8; 32]>,
  pub merkle_path_before: Vec<[u8; 32]>,
  pub merkle_path_after: Vec<[u8; 32]>,
}

#[derive(Debug, Clone)]
pub struct InstructionTransitionStatement {
  pub opcode: u8,
  pub s_i: VmState,
  pub s_next: VmState,
  pub accesses: Vec<AccessRecord>,
  /// Sub-call / create witness — `Some` for CALL, CALLCODE, DELEGATECALL,
  /// STATICCALL, CREATE, CREATE2.  Copied from the corresponding
  /// [`InstructionTransitionProof::sub_call_claim`] during batch proving.
  pub sub_call_claim: Option<SubCallClaim>,
}

pub struct InstructionTransitionZkReceipt {
  pub opcode: u8,
  pub stack_ir_proof: CircleStarkProof,
  pub lut_kernel_proof: CircleStarkProof,
  /// Companion LogUp byte-table proof for AND/OR/XOR operations.
  /// `None` for opcodes without byte-level bitwise operations.
  pub byte_table_proof: Option<p3_uni_stark::Proof<CircleStarkConfig>>,
  /// Preprocessed verifier key for the StackIR STARK (single-instruction setup).
  pub preprocessed_vk: PreprocessedVerifierKey<CircleStarkConfig>,
  /// Preprocessed verifier key for the LUT STARK (batch N=1 setup via
  /// [`BatchLutKernelAirWithPrep`]).
  pub lut_preprocessed_vk: PreprocessedVerifierKey<CircleStarkConfig>,
  pub expected_wff: WFF,
}

/// ZK receipt for a batch of N instructions proved in a single LUT STARK call.
///
/// Security model:
/// - `lut_proof`: batch LUT STARK covering arithmetic of ALL instructions' ProofRows.
/// - `preprocessed_vk`: VK for the shared batch preprocessed matrix.
/// - `manifest`: per-instruction metadata (opcode, WFF, row range in `all_rows`).
/// - `memory_proof`: STARK proof that memory accesses are consistent across the batch.
///
/// Verification requires:
/// 1. `verify_batch_lut_with_prep(lut_proof, preprocessed_vk, pis)` — STARK check.
/// 2. For each i: `infer_proof(pi_i) == manifest.entries[i].wff` — deterministic.
/// 3. `compute_batch_manifest_digest(&entries) == pis[2..10]` — deterministic.
/// 4. If `memory_proof` is Some: `verify_memory_consistency` — STARK check.
/// 5. If `stack_proof` is Some: `verify_stack_consistency` — STARK check.
pub struct BatchTransactionZkReceipt {
  /// Single LUT STARK proof covering arithmetic of all N instructions.
  pub lut_proof: CircleStarkProof,
  /// Shared preprocessed verifier key (batch manifest committed here).
  pub preprocessed_vk: PreprocessedVerifierKey<CircleStarkConfig>,
  /// Per-instruction metadata and concatenated ProofRows.
  pub manifest: BatchProofRowsManifest,
  /// Memory consistency proof.  `None` when the batch has no MLOAD/MSTORE/MSTORE8.
  pub memory_proof: Option<MemoryConsistencyProof>,
  /// Storage consistency proof.  `None` when the batch has no SLOAD/SSTORE.
  pub storage_proof: Option<StorageConsistencyProof>,
  /// Stack consistency proof.  `None` when the batch has no stack-touching instructions.
  pub stack_proof: Option<StackConsistencyProof>,
  /// KECCAK256 consistency proof.  `None` when the batch has no KECCAK256 instructions.
  pub keccak_proof: Option<KeccakConsistencyProof>,
  /// Return/revert data from the terminating instruction.
  /// `Some` when the transaction ended with RETURN or REVERT.
  pub return_data: Option<ReturnDataClaim>,
  // (logs are stored inside the individual proofs themselves)
}

// ============================================================
// Number of stack inputs/outputs per opcode
// ============================================================

/// How many values this opcode pops from the stack.
pub fn opcode_input_count(op: u8) -> usize {
  match op {
    opcode::STOP => 0,
    opcode::ADD
    | opcode::SUB
    | opcode::MUL
    | opcode::DIV
    | opcode::SDIV
    | opcode::MOD
    | opcode::SMOD => 2,
    opcode::ADDMOD | opcode::MULMOD => 3,
    opcode::EXP => 2,
    opcode::LT | opcode::GT | opcode::SLT | opcode::SGT | opcode::EQ => 2,
    opcode::ISZERO | opcode::NOT => 1,
    opcode::BYTE | opcode::SIGNEXTEND => 2,
    opcode::AND | opcode::OR | opcode::XOR => 2,
    opcode::SHL | opcode::SHR | opcode::SAR => 2,
    opcode::MLOAD => 1,
    opcode::MSTORE | opcode::MSTORE8 => 2,
    opcode::SLOAD => 1,
    opcode::SSTORE => 2,
    opcode::RETURN | opcode::REVERT => 2,
    opcode::KECCAK256 => 2,
    opcode::CALLER | opcode::CALLVALUE | opcode::CALLDATASIZE => 0,
    opcode::CALLDATALOAD => 1,
    opcode::PC | opcode::MSIZE | opcode::GAS => 0,
    opcode::ADDRESS
    | opcode::ORIGIN
    | opcode::GASPRICE
    | opcode::CODESIZE
    | opcode::RETURNDATASIZE
    | opcode::COINBASE
    | opcode::TIMESTAMP
    | opcode::NUMBER
    | opcode::DIFFICULTY
    | opcode::GASLIMIT
    | opcode::CHAINID
    | opcode::SELFBALANCE
    | opcode::BASEFEE
    | opcode::BLOBBASEFEE => 0,
    opcode::BLOCKHASH | opcode::EXTCODESIZE | opcode::EXTCODEHASH | opcode::BALANCE => 1,
    opcode::RETURNDATACOPY | opcode::MCOPY | opcode::CALLDATACOPY | opcode::CODECOPY => 3,
    opcode::EXTCODECOPY => 4,
    opcode::BLOBHASH => 1,
    opcode::CALL | opcode::CALLCODE => 7,
    opcode::DELEGATECALL | opcode::STATICCALL => 6,
    opcode::CREATE => 3,
    opcode::CREATE2 => 4,
    opcode::SELFDESTRUCT => 1,
    opcode::TLOAD => 1,
    opcode::TSTORE => 2,
    opcode::LOG0 => 2,
    opcode::LOG1 => 3,
    opcode::LOG2 => 4,
    opcode::LOG3 => 5,
    opcode::LOG4 => 6,
    opcode::INVALID => 0,
    opcode::POP => 1,
    opcode::JUMP => 1,
    opcode::JUMPI => 2,
    opcode::JUMPDEST => 0,
    op if (opcode::PUSH0..=opcode::PUSH32).contains(&op) => 0,
    op if (opcode::DUP1..=opcode::DUP16).contains(&op) => 0,
    op if (opcode::SWAP1..=opcode::SWAP16).contains(&op) => 0,
    _ => 0,
  }
}

/// How many values this opcode pushes onto the stack.
pub fn opcode_output_count(op: u8) -> usize {
  match op {
    opcode::STOP => 0,
    opcode::ADD
    | opcode::SUB
    | opcode::MUL
    | opcode::DIV
    | opcode::SDIV
    | opcode::MOD
    | opcode::SMOD => 1,
    opcode::ADDMOD | opcode::MULMOD => 1,
    opcode::EXP => 1,
    opcode::BYTE | opcode::SIGNEXTEND => 1,
    opcode::LT | opcode::GT | opcode::SLT | opcode::SGT | opcode::EQ => 1,
    opcode::ISZERO | opcode::NOT => 1,
    opcode::AND | opcode::OR | opcode::XOR => 1,
    opcode::SHL | opcode::SHR | opcode::SAR => 1,
    opcode::MLOAD => 1,
    opcode::MSTORE | opcode::MSTORE8 => 0,
    opcode::SLOAD => 1,
    opcode::SSTORE => 0,
    opcode::RETURN | opcode::REVERT => 0,
    opcode::KECCAK256 => 1,
    opcode::CALLER | opcode::CALLVALUE | opcode::CALLDATALOAD | opcode::CALLDATASIZE => 1,
    opcode::PC | opcode::MSIZE | opcode::GAS => 1,
    opcode::ADDRESS
    | opcode::ORIGIN
    | opcode::GASPRICE
    | opcode::CODESIZE
    | opcode::RETURNDATASIZE
    | opcode::COINBASE
    | opcode::TIMESTAMP
    | opcode::NUMBER
    | opcode::DIFFICULTY
    | opcode::GASLIMIT
    | opcode::CHAINID
    | opcode::SELFBALANCE
    | opcode::BASEFEE
    | opcode::BLOBBASEFEE => 1,
    opcode::BLOCKHASH | opcode::EXTCODESIZE | opcode::EXTCODEHASH | opcode::BALANCE => 1,
    opcode::RETURNDATACOPY | opcode::EXTCODECOPY | opcode::MCOPY
    | opcode::CALLDATACOPY | opcode::CODECOPY => 0,
    opcode::BLOBHASH => 1,
    opcode::CALL | opcode::CALLCODE | opcode::DELEGATECALL | opcode::STATICCALL => 1,
    opcode::CREATE | opcode::CREATE2 => 1,
    opcode::SELFDESTRUCT => 0,
    opcode::TLOAD => 1,
    opcode::TSTORE => 0,
    opcode::LOG0 | opcode::LOG1 | opcode::LOG2 | opcode::LOG3 | opcode::LOG4 => 0,
    opcode::INVALID => 0,
    opcode::POP => 0,
    opcode::JUMP | opcode::JUMPI => 0,
    opcode::JUMPDEST => 0,
    op if (opcode::PUSH0..=opcode::PUSH32).contains(&op) => 1,
    op if (opcode::DUP1..=opcode::DUP16).contains(&op) => 1,
    op if (opcode::SWAP1..=opcode::SWAP16).contains(&op) => 0,
    _ => 0,
  }
}

pub fn has_expected_stack_arity(op: u8, inputs: &[[u8; 32]], outputs: &[[u8; 32]]) -> bool {
  inputs.len() == opcode_input_count(op) && outputs.len() == opcode_output_count(op)
}

pub fn expected_sp_after(op: u8, sp_before: usize) -> Option<usize> {
  let pops = opcode_input_count(op);
  let pushes = opcode_output_count(op);
  sp_before.checked_sub(pops).map(|v| v + pushes)
}

pub fn has_valid_sp_transition(op: u8, sp_before: usize, sp_after: usize) -> bool {
  expected_sp_after(op, sp_before) == Some(sp_after)
}

fn is_structural_opcode(op: u8) -> bool {
  matches!(
    op,
    opcode::STOP | opcode::POP | opcode::JUMP | opcode::JUMPI | opcode::JUMPDEST | opcode::INVALID
  ) || (opcode::PUSH0..=opcode::PUSH32).contains(&op)
    || (opcode::DUP1..=opcode::DUP16).contains(&op)
    || (opcode::SWAP1..=opcode::SWAP16).contains(&op)
}

/// Memory opcodes have no per-instruction Hilbert proof;
/// their correctness is established at batch level by `MemoryConsistencyAir`.
fn is_memory_opcode(op: u8) -> bool {
  matches!(
    op,
    opcode::MLOAD
      | opcode::MSTORE
      | opcode::MSTORE8
      | opcode::RETURNDATACOPY
      | opcode::EXTCODECOPY
      | opcode::MCOPY
      | opcode::CALLDATACOPY
      | opcode::CODECOPY
  )
}

/// Storage opcodes have no per-instruction Hilbert proof;
/// their correctness is established at batch level by `StorageConsistencyAir`.
/// Includes transient storage (EIP-1153: TLOAD/TSTORE).
fn is_storage_opcode(op: u8) -> bool {
  matches!(op, opcode::SLOAD | opcode::SSTORE | opcode::TLOAD | opcode::TSTORE)
}

/// Terminating opcodes that carry return data from memory (RETURN / REVERT).
/// Like memory opcodes, they have no per-instruction Hilbert proof;
/// correctness is captured by the `ReturnDataClaim` witness.
fn is_terminating_data_opcode(op: u8) -> bool {
  matches!(op, opcode::RETURN | opcode::REVERT)
}

/// Keccak256 opcode (SHA3 / 0x20).
/// No per-instruction Hilbert proof; correctness is established at batch
/// level by re-computing `keccak256(preimage)` in `prove_keccak_consistency`.
fn is_keccak_opcode(op: u8) -> bool {
  op == opcode::KECCAK256
}

/// Inline keccak verification for `verify_proof`:
/// - opcode must be KECCAK256
/// - keccak_claim must be present
/// - keccak256(claim.input_bytes) must equal claim.output_hash
/// - claim.output_hash must match the stack output
fn verify_keccak_claim_inline(proof: &InstructionTransitionProof) -> bool {
  if !is_keccak_opcode(proof.opcode) {
    return false;
  }
  let Some(ref claim) = proof.keccak_claim else {
    return false;
  };
  // Verify preimage
  let computed = crate::zk_proof::keccak256_bytes(&claim.input_bytes);
  if computed != claim.output_hash {
    return false;
  }
  // Stack output must match the hash
  if proof.stack_outputs.len() != 1 {
    return false;
  }
  proof.stack_outputs[0] == claim.output_hash
}

/// Environment / execution-context opcodes whose output comes from the EVM state,
/// not from arithmetic over stack inputs.
/// Includes: CALLER, CALLVALUE, CALLDATALOAD, CALLDATASIZE, PC, MSIZE, GAS,
/// ADDRESS, ORIGIN, GASPRICE, CODESIZE, RETURNDATASIZE, COINBASE, TIMESTAMP,
/// NUMBER, DIFFICULTY (PREVRANDAO), GASLIMIT, CHAINID, SELFBALANCE, BASEFEE,
/// BLOBBASEFEE, BLOBHASH (EIP-4844).
/// Correctness is established at batch level by cross-checking `CallContextClaim`
/// against the original execution context.
fn is_env_opcode(op: u8) -> bool {
  matches!(
    op,
    opcode::CALLER
      | opcode::CALLVALUE
      | opcode::CALLDATALOAD
      | opcode::CALLDATASIZE
      | opcode::PC
      | opcode::MSIZE
      | opcode::GAS
      | opcode::ADDRESS
      | opcode::ORIGIN
      | opcode::GASPRICE
      | opcode::CODESIZE
      | opcode::RETURNDATASIZE
      | opcode::COINBASE
      | opcode::TIMESTAMP
      | opcode::NUMBER
      | opcode::DIFFICULTY
      | opcode::GASLIMIT
      | opcode::CHAINID
      | opcode::SELFBALANCE
      | opcode::BASEFEE
      | opcode::BLOBBASEFEE
      | opcode::BLOBHASH
  )
}

/// External-state-reading opcodes whose output comes from world state,
/// not from arithmetic over stack inputs.
/// Correctness is established at batch level via `ExternalStateClaim`.
fn is_external_state_opcode(op: u8) -> bool {
  matches!(
    op,
    opcode::BLOCKHASH | opcode::EXTCODESIZE | opcode::EXTCODEHASH | opcode::BALANCE
  )
}

/// Event log opcodes (LOG0-LOG4).
/// These consume memory data and topic stack items but produce no stack output.
/// The log content is captured in execution receipts; no per-instruction proof needed.
fn is_log_opcode(op: u8) -> bool {
  matches!(
    op,
    opcode::LOG0 | opcode::LOG1 | opcode::LOG2 | opcode::LOG3 | opcode::LOG4
  )
}

/// Sub-call / contract-creation opcodes.
/// Correctness of the success flag and return data is established at batch
/// level via `SubCallClaim`.
fn is_subcall_opcode(op: u8) -> bool {
  matches!(
    op,
    opcode::CALL
      | opcode::CALLCODE
      | opcode::DELEGATECALL
      | opcode::STATICCALL
      | opcode::CREATE
      | opcode::CREATE2
      | opcode::SELFDESTRUCT
  )
}

pub fn verify_statement_semantics(statement: &InstructionTransitionStatement) -> bool {
  if !has_valid_sp_transition(statement.opcode, statement.s_i.sp, statement.s_next.sp) {
    return false;
  }

  if !has_expected_stack_arity(
    statement.opcode,
    &statement.s_i.stack,
    &statement.s_next.stack,
  ) {
    return false;
  }

  true
}

// ============================================================
// Proof generation per instruction
// ============================================================

/// Generate a semantic proof for a single EVM instruction.
///
/// Returns `(proof, computed_outputs)`.  The caller should compare
/// `computed_outputs` against the actual EVM outputs to ensure consistency.
pub fn prove_instruction(op: u8, inputs: &[[u8; 32]], outputs: &[[u8; 32]]) -> Option<Proof> {
  if !has_expected_stack_arity(op, inputs, outputs) {
    return None;
  }

  match op {
    opcode::ADD => Some(prove_add(&inputs[0], &inputs[1], &outputs[0])),
    opcode::SUB => Some(prove_sub(&inputs[0], &inputs[1], &outputs[0])),
    opcode::MUL => Some(prove_mul(&inputs[0], &inputs[1], &outputs[0])),
    opcode::DIV => Some(prove_div(&inputs[0], &inputs[1], &outputs[0])),
    opcode::MOD => Some(prove_mod(&inputs[0], &inputs[1], &outputs[0])),
    opcode::SDIV => Some(prove_sdiv(&inputs[0], &inputs[1], &outputs[0])),
    opcode::SMOD => Some(prove_smod(&inputs[0], &inputs[1], &outputs[0])),
    opcode::AND => Some(prove_and(&inputs[0], &inputs[1], &outputs[0])),
    opcode::OR => Some(prove_or(&inputs[0], &inputs[1], &outputs[0])),
    opcode::XOR => Some(prove_xor(&inputs[0], &inputs[1], &outputs[0])),
    opcode::NOT => Some(prove_not(&inputs[0], &outputs[0])),
    opcode::LT => Some(prove_lt(&inputs[0], &inputs[1], &outputs[0])),
    opcode::GT => Some(prove_gt(&inputs[0], &inputs[1], &outputs[0])),
    opcode::SLT => Some(prove_slt(&inputs[0], &inputs[1], &outputs[0])),
    opcode::SGT => Some(prove_sgt(&inputs[0], &inputs[1], &outputs[0])),
    opcode::EQ => Some(prove_eq(&inputs[0], &inputs[1], &outputs[0])),
    opcode::ISZERO => Some(prove_iszero(&inputs[0], &outputs[0])),
    opcode::SHL => Some(prove_shl(&inputs[0], &inputs[1], &outputs[0])),
    opcode::SHR => Some(prove_shr(&inputs[0], &inputs[1], &outputs[0])),
    opcode::SAR => Some(prove_sar(&inputs[0], &inputs[1], &outputs[0])),
    opcode::BYTE => Some(prove_byte(&inputs[0], &inputs[1], &outputs[0])),
    opcode::SIGNEXTEND => Some(prove_signextend(&inputs[0], &inputs[1], &outputs[0])),
    opcode::ADDMOD => Some(prove_addmod(&inputs[0], &inputs[1], &inputs[2], &outputs[0])),
    opcode::MULMOD => Some(prove_mulmod(&inputs[0], &inputs[1], &inputs[2], &outputs[0])),
    opcode::EXP => Some(prove_exp(&inputs[0], &inputs[1], &outputs[0])),
    // Memory opcodes: no arithmetic proof in the proof tree.
    // Consistency is guaranteed by MemoryConsistencyAir at the batch level.
    opcode::MLOAD
    | opcode::MSTORE
    | opcode::MSTORE8
    | opcode::RETURNDATACOPY
    | opcode::EXTCODECOPY
    | opcode::MCOPY => None,
    // Storage opcodes: no arithmetic proof; correctness via StorageConsistencyAir.
    opcode::SLOAD | opcode::SSTORE => None,
    // RETURN/REVERT: no arithmetic proof; return data is captured via ReturnDataClaim.
    opcode::RETURN | opcode::REVERT => None,
    // Env opcodes: no arithmetic proof; value comes from execution context (CallContextClaim).
    opcode::CALLER
    | opcode::CALLVALUE
    | opcode::CALLDATALOAD
    | opcode::CALLDATASIZE
    | opcode::PC
    | opcode::MSIZE
    | opcode::GAS
    | opcode::ADDRESS
    | opcode::ORIGIN
    | opcode::GASPRICE
    | opcode::CODESIZE
    | opcode::RETURNDATASIZE
    | opcode::COINBASE
    | opcode::TIMESTAMP
    | opcode::NUMBER
    | opcode::DIFFICULTY
    | opcode::GASLIMIT
    | opcode::CHAINID
    | opcode::SELFBALANCE
    | opcode::BASEFEE
    | opcode::BLOBBASEFEE => None,
    // External state opcodes: no arithmetic proof; value comes from ExternalStateClaim.
    opcode::BLOCKHASH | opcode::EXTCODESIZE | opcode::EXTCODEHASH | opcode::BALANCE => None,
    // BLOBHASH: no arithmetic proof; value comes from transaction blob context (CallContextClaim).
    opcode::BLOBHASH => None,
    // CALLDATACOPY/CODECOPY: memory copy; correctness via MemoryConsistencyAir.
    opcode::CALLDATACOPY | opcode::CODECOPY => None,
    // Sub-call / create opcodes: no arithmetic proof; result captured via SubCallClaim.
    opcode::CALL
    | opcode::CALLCODE
    | opcode::DELEGATECALL
    | opcode::STATICCALL
    | opcode::CREATE
    | opcode::CREATE2
    | opcode::SELFDESTRUCT => None,
    // Transient storage (EIP-1153): no arithmetic proof; like SLOAD/SSTORE.
    opcode::TLOAD | opcode::TSTORE => None,
    // Event log opcodes: no arithmetic proof; data captured in execution receipt.
    opcode::LOG0 | opcode::LOG1 | opcode::LOG2 | opcode::LOG3 | opcode::LOG4 => None,
    // INVALID: terminates execution; no proof.
    opcode::INVALID => None,
    // KECCAK256: no arithmetic proof; preimage correctness via KeccakConsistencyProof.
    opcode::KECCAK256 => None,
    // Structural ops: no semantic proof needed
    _ => None,
  }
}

// ============================================================
// WFF generation
// ============================================================

pub fn wff_instruction(op: u8, inputs: &[[u8; 32]], outputs: &[[u8; 32]]) -> Option<WFF> {
  if !has_expected_stack_arity(op, inputs, outputs) {
    return None;
  }

  match op {
    opcode::ADD => Some(wff_add(&inputs[0], &inputs[1], &outputs[0])),
    opcode::SUB => Some(wff_sub(&inputs[0], &inputs[1], &outputs[0])),
    opcode::MUL => Some(wff_mul(&inputs[0], &inputs[1], &outputs[0])),
    opcode::DIV => Some(wff_div(&inputs[0], &inputs[1], &outputs[0])),
    opcode::MOD => Some(wff_mod(&inputs[0], &inputs[1], &outputs[0])),
    opcode::SDIV => Some(wff_sdiv(&inputs[0], &inputs[1], &outputs[0])),
    opcode::SMOD => Some(wff_smod(&inputs[0], &inputs[1], &outputs[0])),
    opcode::AND => Some(wff_and(&inputs[0], &inputs[1], &outputs[0])),
    opcode::OR => Some(wff_or(&inputs[0], &inputs[1], &outputs[0])),
    opcode::XOR => Some(wff_xor(&inputs[0], &inputs[1], &outputs[0])),
    opcode::NOT => Some(wff_not(&inputs[0], &outputs[0])),
    opcode::LT => Some(wff_lt(&inputs[0], &inputs[1], &outputs[0])),
    opcode::GT => Some(wff_gt(&inputs[0], &inputs[1], &outputs[0])),
    opcode::SLT => Some(wff_slt(&inputs[0], &inputs[1], &outputs[0])),
    opcode::SGT => Some(wff_sgt(&inputs[0], &inputs[1], &outputs[0])),
    opcode::EQ => Some(wff_eq(&inputs[0], &inputs[1], &outputs[0])),
    opcode::ISZERO => Some(wff_iszero(&inputs[0], &outputs[0])),
    opcode::SHL => Some(wff_shl(&inputs[0], &inputs[1], &outputs[0])),
    opcode::SHR => Some(wff_shr(&inputs[0], &inputs[1], &outputs[0])),
    opcode::SAR => Some(wff_sar(&inputs[0], &inputs[1], &outputs[0])),
    opcode::BYTE => Some(wff_byte(&inputs[0], &inputs[1], &outputs[0])),
    opcode::SIGNEXTEND => Some(wff_signextend(&inputs[0], &inputs[1], &outputs[0])),
    opcode::ADDMOD => Some(wff_addmod(&inputs[0], &inputs[1], &inputs[2], &outputs[0])),
    opcode::MULMOD => Some(wff_mulmod(&inputs[0], &inputs[1], &inputs[2], &outputs[0])),
    opcode::EXP => Some(wff_exp(&inputs[0], &inputs[1], &outputs[0])),
    // Memory opcodes: WFF is handled at batch level via MemoryConsistencyAir
    opcode::MLOAD
    | opcode::MSTORE
    | opcode::MSTORE8
    | opcode::RETURNDATACOPY
    | opcode::EXTCODECOPY
    | opcode::MCOPY => None,
    // Storage opcodes: no WFF; consistency via StorageConsistencyAir.
    opcode::SLOAD | opcode::SSTORE => None,
    // RETURN/REVERT: no WFF; return data captured via ReturnDataClaim.
    opcode::RETURN | opcode::REVERT => None,
    // Env opcodes: no WFF; value comes from execution context (CallContextClaim).
    opcode::CALLER
    | opcode::CALLVALUE
    | opcode::CALLDATALOAD
    | opcode::CALLDATASIZE
    | opcode::PC
    | opcode::MSIZE
    | opcode::GAS
    | opcode::ADDRESS
    | opcode::ORIGIN
    | opcode::GASPRICE
    | opcode::CODESIZE
    | opcode::RETURNDATASIZE
    | opcode::COINBASE
    | opcode::TIMESTAMP
    | opcode::NUMBER
    | opcode::DIFFICULTY
    | opcode::GASLIMIT
    | opcode::CHAINID
    | opcode::SELFBALANCE
    | opcode::BASEFEE
    | opcode::BLOBBASEFEE => None,
    // External state opcodes: no WFF; value comes from ExternalStateClaim.
    opcode::BLOCKHASH | opcode::EXTCODESIZE | opcode::EXTCODEHASH | opcode::BALANCE => None,
    // BLOBHASH: no WFF; value comes from transaction blob context (CallContextClaim).
    opcode::BLOBHASH => None,
    // CALLDATACOPY/CODECOPY: memory copy; correctness via MemoryConsistencyAir.
    opcode::CALLDATACOPY | opcode::CODECOPY => None,
    // Sub-call / create opcodes: no WFF; result captured via SubCallClaim.
    opcode::CALL
    | opcode::CALLCODE
    | opcode::DELEGATECALL
    | opcode::STATICCALL
    | opcode::CREATE
    | opcode::CREATE2
    | opcode::SELFDESTRUCT => None,
    // Transient storage (EIP-1153): no WFF; like SLOAD/SSTORE.
    opcode::TLOAD | opcode::TSTORE => None,
    // Event log opcodes: no WFF; data captured in execution receipt.
    opcode::LOG0 | opcode::LOG1 | opcode::LOG2 | opcode::LOG3 | opcode::LOG4 => None,
    // INVALID: terminates execution; no proof.
    opcode::INVALID => None,
    // KECCAK256: no WFF; preimage correctness via KeccakConsistencyProof.
    opcode::KECCAK256 => None,
    // Structural ops: no semantic proof needed
    _ => None,
  }
}

// ============================================================
// Verification (for testing)
// ============================================================

pub fn verify_proof(proof: &InstructionTransitionProof) -> bool {
  if !has_expected_stack_arity(proof.opcode, &proof.stack_inputs, &proof.stack_outputs) {
    return false;
  }

  let expected_wff = wff_instruction(proof.opcode, &proof.stack_inputs, &proof.stack_outputs);
  match (&proof.semantic_proof, expected_wff) {
    (Some(proof), Some(wff)) => {
      let Ok(wff_result) = infer_proof(proof) else {
        return false;
      };
      wff == wff_result
    }
    (None, None) => is_structural_opcode(proof.opcode)
      || is_memory_opcode(proof.opcode)
      || is_storage_opcode(proof.opcode)
      || is_terminating_data_opcode(proof.opcode)
      || is_env_opcode(proof.opcode)
      || is_external_state_opcode(proof.opcode)
      || is_subcall_opcode(proof.opcode)
      || is_log_opcode(proof.opcode)
      || verify_keccak_claim_inline(proof),
    _ => false, // Mismatch between proof presence and expected WFF
  }
}

/// Verify an instruction transition with both:
/// 1) Hilbert-style semantic proof checking, and
/// 2) compiled ProofRow verification (`compile_proof` + `verify_compiled`).
///
/// This is the main verification path for production transition checks.
pub fn verify_proof_with_rows(proof: &InstructionTransitionProof) -> bool {
  if !verify_proof(proof) {
    return false;
  }

  match &proof.semantic_proof {
    Some(semantic_proof) => {
      let rows = compile_proof(semantic_proof);
      verify_compiled(&rows).is_ok()
    }
    _ => true,
  }
}

/// Backward-compatible alias.
///
/// ZKP path uses a ProofRow-shaped trace in AIR.
pub fn verify_proof_with_zkp(proof: &InstructionTransitionProof) -> bool {
  let statement = InstructionTransitionStatement {
    opcode: proof.opcode,
    s_i: VmState {
      opcode: proof.opcode,
      pc: proof.pc,
      sp: proof.stack_inputs.len(),
      stack: proof.stack_inputs.clone(),
      memory_root: [0u8; 32],
    },
    s_next: VmState {
      opcode: proof.opcode,
      pc: proof.pc + 1,
      sp: proof.stack_outputs.len(),
      stack: proof.stack_outputs.clone(),
      memory_root: [0u8; 32],
    },
    accesses: Vec::new(),
    sub_call_claim: proof.sub_call_claim.clone(),
  };
  let receipt = match prove_instruction_zk_receipt(proof) {
    Ok(receipt) => receipt,
    Err(_) => return false,
  };
  verify_instruction_zk_receipt(&statement, &receipt)
}

pub fn prove_instruction_zk_receipt(
  proof: &InstructionTransitionProof,
) -> Result<InstructionTransitionZkReceipt, String> {
  let mut receipts = prove_instruction_zk_receipts_parallel(vec![proof.clone()], 0)?;
  receipts
    .pop()
    .ok_or_else(|| "missing receipt from single-proof proving path".to_string())
}

pub(crate) fn prove_instruction_zk_receipt_with_lut_override(
  proof: &InstructionTransitionProof,
) -> Result<InstructionTransitionZkReceipt, String> {
  if !verify_proof_with_rows(proof) {
    return Err("semantic/row verification failed before ZKP receipt generation".to_string());
  }

  let semantic_proof = proof
    .semantic_proof
    .as_ref()
    .ok_or_else(|| "missing semantic proof".to_string())?;

  let expected_wff = wff_instruction(proof.opcode, &proof.stack_inputs, &proof.stack_outputs)
    .ok_or_else(|| format!("unsupported opcode for ZKP receipt: 0x{:02x}", proof.opcode))?;

  let rows = compile_proof(semantic_proof);
  let stack_bind_pv =
    make_receipt_binding_public_values(RECEIPT_BIND_TAG_STACK, proof.opcode, &expected_wff);

  // StackIR proof: single-instruction setup (unchanged).
  let (prep_data, preprocessed_vk) = setup_proof_rows_preprocessed(&rows, &stack_bind_pv)?;
  let stack_ir_proof = prove_stack_ir_with_prep(&rows, &prep_data, &stack_bind_pv)?;

  // LUT proof: route through batch (BatchLutKernelAirWithPrep) for N=1.
  let lut_items: &[(u8, &Proof)] = &[(proof.opcode, semantic_proof)];
  let lut_manifest = build_batch_manifest(lut_items)?;
  let lut_bind_pv =
    make_batch_receipt_binding_public_values(RECEIPT_BIND_TAG_LUT, &lut_manifest.entries);
  let (lut_prep_data, lut_preprocessed_vk) =
    setup_batch_proof_rows_preprocessed(&lut_manifest, &lut_bind_pv)?;
  let lut_kernel_proof =
    prove_batch_lut_with_prep(&lut_manifest, &lut_prep_data, &lut_bind_pv)?;

  // Byte-table proof for AND/OR/XOR operations.
  let byte_queries = collect_byte_table_queries_from_rows(&lut_manifest.all_rows);
  let byte_table_proof = if byte_queries.is_empty() {
    None
  } else {
    Some(crate::byte_table::prove_byte_table(&byte_queries))
  };

  Ok(InstructionTransitionZkReceipt {
    opcode: proof.opcode,
    stack_ir_proof,
    lut_kernel_proof,
    byte_table_proof,
    preprocessed_vk,
    lut_preprocessed_vk,
    expected_wff,
  })
}

/// Like [`prove_instruction_zk_receipt`], but also returns per-component timings.
///
/// Returns `(receipt, (setup_µs, stack_ir_µs, lut_µs, logup_µs))`.
/// `lut_µs` is the pure LUT STARK prove time; `logup_µs` is the companion
/// byte-table (LogUp) prove time (non-zero for AND/OR/XOR only).
/// Useful for profiling which of the steps dominates.
pub fn prove_instruction_zk_receipt_timed(
  proof: &InstructionTransitionProof,
) -> Result<(InstructionTransitionZkReceipt, (f64, f64, f64, f64)), String> {
  use std::time::Instant;

  if !verify_proof_with_rows(proof) {
    return Err("semantic/row verification failed before ZKP receipt generation".to_string());
  }

  let semantic_proof = proof
    .semantic_proof
    .as_ref()
    .ok_or_else(|| "missing semantic proof".to_string())?;

  let expected_wff = wff_instruction(proof.opcode, &proof.stack_inputs, &proof.stack_outputs)
    .ok_or_else(|| format!("unsupported opcode for ZKP receipt: 0x{:02x}", proof.opcode))?;

  let rows = compile_proof(semantic_proof);
  let stack_bind_pv =
    make_receipt_binding_public_values(RECEIPT_BIND_TAG_STACK, proof.opcode, &expected_wff);

  let t0 = Instant::now();
  let (prep_data, preprocessed_vk) = setup_proof_rows_preprocessed(&rows, &stack_bind_pv)?;
  let setup_us = t0.elapsed().as_secs_f64() * 1e6;

  let t1 = Instant::now();
  let stack_ir_proof = prove_stack_ir_with_prep(&rows, &prep_data, &stack_bind_pv)?;
  let stack_ir_us = t1.elapsed().as_secs_f64() * 1e6;

  // LUT proof: batch path (N=1), includes batch setup + prove.
  let t2 = Instant::now();
  let lut_items: &[(u8, &Proof)] = &[(proof.opcode, semantic_proof)];
  let lut_manifest = build_batch_manifest(lut_items)?;
  let lut_bind_pv =
    make_batch_receipt_binding_public_values(RECEIPT_BIND_TAG_LUT, &lut_manifest.entries);
  let (lut_prep_data, lut_preprocessed_vk) =
    setup_batch_proof_rows_preprocessed(&lut_manifest, &lut_bind_pv)?;
  let lut_kernel_proof =
    prove_batch_lut_with_prep(&lut_manifest, &lut_prep_data, &lut_bind_pv)?;
  let lut_us = t2.elapsed().as_secs_f64() * 1e6;

  // Time the companion LogUp byte-table proof separately.
  let byte_queries = collect_byte_table_queries_from_rows(&lut_manifest.all_rows);
  let t3 = Instant::now();
  let byte_table_proof = if byte_queries.is_empty() {
    None
  } else {
    Some(crate::byte_table::prove_byte_table(&byte_queries))
  };
  let logup_us = t3.elapsed().as_secs_f64() * 1e6;

  let receipt = InstructionTransitionZkReceipt {
    opcode: proof.opcode,
    stack_ir_proof,
    lut_kernel_proof,
    byte_table_proof,
    preprocessed_vk,
    lut_preprocessed_vk,
    expected_wff,
  };
  Ok((receipt, (setup_us, stack_ir_us, lut_us, logup_us)))
}

/// Prove many instruction ZKP receipts in parallel using a thread pool.
///
/// Work items are distributed across worker threads and output order matches
/// input order.
pub fn prove_instruction_zk_receipts_parallel(
  proofs: Vec<InstructionTransitionProof>,
  worker_count: usize,
) -> Result<Vec<InstructionTransitionZkReceipt>, String> {
  use std::collections::VecDeque;
  use std::sync::{Arc, Mutex, mpsc};
  use std::thread;

  let task_count = proofs.len();
  if task_count == 0 {
    return Ok(Vec::new());
  }

  let workers = if worker_count == 0 {
    thread::available_parallelism()
      .map(|n| n.get())
      .unwrap_or(1)
      .min(task_count)
  } else {
    worker_count.max(1).min(task_count)
  };

  let queue: Arc<Mutex<VecDeque<(usize, InstructionTransitionProof)>>> =
    Arc::new(Mutex::new(proofs.into_iter().enumerate().collect()));

  let (tx, rx) = mpsc::channel::<(usize, Result<InstructionTransitionZkReceipt, String>)>();
  let mut handles = Vec::with_capacity(workers);

  for _ in 0..workers {
    let queue = Arc::clone(&queue);
    let tx = tx.clone();
    handles.push(thread::spawn(move || {
      loop {
        let item = queue.lock().unwrap().pop_front();
        let Some((idx, proof)) = item else { break };
        let result = prove_instruction_zk_receipt_with_lut_override(&proof);
        if tx.send((idx, result)).is_err() {
          break;
        }
      }
    }));
  }
  drop(tx);

  let mut ordered: Vec<Option<InstructionTransitionZkReceipt>> =
    std::iter::repeat_with(|| None).take(task_count).collect();
  let mut first_error: Option<String> = None;

  for _ in 0..task_count {
    let (orig_idx, result) = rx
      .recv()
      .map_err(|_| "parallel proving channel closed unexpectedly".to_string())?;
    match result {
      Ok(receipt) => ordered[orig_idx] = Some(receipt),
      Err(err) => {
        if first_error.is_none() {
          first_error = Some(err);
        }
      }
    }
  }

  for handle in handles {
    let _ = handle.join();
  }

  if let Some(err) = first_error {
    return Err(err);
  }

  let mut out = Vec::with_capacity(task_count);
  for receipt in ordered {
    out.push(
      receipt.ok_or_else(|| "failed to collect all receipts after parallel proving".to_string())?,
    );
  }
  Ok(out)
}

pub fn verify_instruction_zk_receipt(
  statement: &InstructionTransitionStatement,
  receipt: &InstructionTransitionZkReceipt,
) -> bool {
  if statement.s_i.opcode != statement.opcode {
    return false;
  }

  if statement.s_i.sp != statement.s_i.stack.len()
    || statement.s_next.sp != statement.s_next.stack.len()
  {
    return false;
  }

  if !verify_statement_semantics(statement) {
    return false;
  }

  if statement.opcode != receipt.opcode {
    return false;
  }

  let expected_wff = match wff_instruction(
    statement.opcode,
    &statement.s_i.stack,
    &statement.s_next.stack,
  ) {
    Some(wff) => wff,
    None => return false,
  };

  if expected_wff != receipt.expected_wff {
    return false;
  }

  let stack_bind_pv =
    make_receipt_binding_public_values(RECEIPT_BIND_TAG_STACK, statement.opcode, &expected_wff);

  if verify_stack_ir_with_prep(
    &receipt.stack_ir_proof,
    &receipt.preprocessed_vk,
    &stack_bind_pv,
  )
  .is_err()
  {
    return false;
  }

  // LUT verification: batch path (BatchLutKernelAirWithPrep).
  let batch_entries = [BatchInstructionMeta {
    opcode: statement.opcode,
    wff: expected_wff.clone(),
    wff_digest: compute_wff_opcode_digest(statement.opcode, &expected_wff),
    row_start: 0,
    row_count: 0,
  }];
  let lut_bind_pv =
    make_batch_receipt_binding_public_values(RECEIPT_BIND_TAG_LUT, &batch_entries);
  if verify_batch_lut_with_prep(
    &receipt.lut_kernel_proof,
    &receipt.lut_preprocessed_vk,
    &lut_bind_pv,
  )
  .is_err()
  {
    return false;
  }

  // Byte-table proof (AND/OR/XOR).
  if let Some(bp) = &receipt.byte_table_proof {
    if crate::byte_table::verify_byte_table(bp).is_err() {
      return false;
    }
  }

  true
}

// ============================================================
// Batch proving helpers
// ============================================================

/// Compile a slice of `(opcode, &Proof)` pairs into a [`BatchProofRowsManifest`].
///
/// Each proof tree is independently compiled via [`compile_proof`] and its
/// inferred WFF recorded.  All resulting [`ProofRow`]s are concatenated into
/// `all_rows` with per-instruction boundary information stored in `entries`.
///
/// Errors if `items` is empty or if any proof fails WFF inference.
pub fn build_batch_manifest(items: &[(u8, &Proof)]) -> Result<BatchProofRowsManifest, String> {
  if items.is_empty() {
    return Err("build_batch_manifest: empty item list".to_string());
  }

  let mut entries: Vec<BatchInstructionMeta> = Vec::with_capacity(items.len());
  let mut all_rows: Vec<ProofRow> = Vec::new();

  for (opcode, proof) in items {
    let wff = infer_proof(proof).map_err(|e| {
      format!(
        "build_batch_manifest: infer_proof failed for opcode 0x{:02x}: {e}",
        opcode
      )
    })?;
    let rows = compile_proof(proof);
    let row_start = all_rows.len();
    let row_count = rows.len();
    all_rows.extend(rows);
    let wff_digest = compute_wff_opcode_digest(*opcode, &wff);
    entries.push(BatchInstructionMeta {
      opcode: *opcode,
      wff,
      wff_digest,
      row_start,
      row_count,
    });
  }

  Ok(BatchProofRowsManifest { entries, all_rows })
}

/// Prove N instructions as a single batch ZK receipt.
///
/// Only instructions that carry a `semantic_proof` (i.e. arithmetic opcodes)
/// are included.  Structural opcodes (PUSH, POP, JUMP, …) are silently skipped.
/// Memory opcodes (MLOAD, MSTORE, MSTORE8) contribute `memory_claims` which
/// are proved separately via `MemoryConsistencyAir`.
///
/// **Proving steps:**
/// 1. Build [`BatchProofRowsManifest`] from the semantic proofs (`build_batch_manifest`).
/// 2. Setup shared batch preprocessed commitment (`setup_batch_proof_rows_preprocessed`).
/// 3. Prove batch LUT STARK in one call (`prove_batch_lut_with_prep`).
/// 4. Collect all memory claims from the batch and prove `MemoryConsistencyAir`.
///
/// Returns `Err` if `itps` contains no semantic proofs or if any proof step fails.
pub fn prove_batch_transaction_zk_receipt(
  itps: &[InstructionTransitionProof],
) -> Result<BatchTransactionZkReceipt, String> {
  let items: Vec<(u8, &Proof)> = itps
    .iter()
    .filter_map(|itp| itp.semantic_proof.as_ref().map(|p| (itp.opcode, p)))
    .collect();

  if items.is_empty() {
    return Err("prove_batch_transaction_zk_receipt: no semantic proofs in batch".to_string());
  }

  let manifest = build_batch_manifest(&items)?;
  let lut_bind_pv =
    make_batch_receipt_binding_public_values(RECEIPT_BIND_TAG_LUT, &manifest.entries);
  let (prep_data, preprocessed_vk) = setup_batch_proof_rows_preprocessed(&manifest, &lut_bind_pv)?;
  let lut_proof = prove_batch_lut_with_prep(&manifest, &prep_data, &lut_bind_pv)?;

  // Collect memory access claims from all instructions, assigning monotone rw_counters.
  let mut all_mem_claims: Vec<MemAccessClaim> = Vec::new();
  let mut all_stor_claims: Vec<StorageAccessClaim> = Vec::new();
  let mut all_keccak_claims: Vec<KeccakClaim> = Vec::new();
  let mut rw_counter: u64 = 0;
  for itp in itps {
    for claim in &itp.memory_claims {
      rw_counter += 1;
      all_mem_claims.push(MemAccessClaim {
        rw_counter,
        addr: claim.addr,
        is_write: claim.is_write,
        value: claim.value,
      });
    }
    for claim in &itp.storage_claims {
      rw_counter += 1;
      all_stor_claims.push(StorageAccessClaim {
        rw_counter,
        contract: claim.contract,
        slot: claim.slot,
        is_write: claim.is_write,
        value: claim.value,
      });
    }
    if let Some(ref kc) = itp.keccak_claim {
      all_keccak_claims.push(kc.clone());
    }
  }
  let memory_proof = if all_mem_claims.is_empty() {
    None
  } else {
    Some(prove_memory_consistency(&all_mem_claims)?)
  };
  let storage_proof = if all_stor_claims.is_empty() {
    None
  } else {
    Some(prove_storage_consistency(&all_stor_claims)?)
  };
  let keccak_proof = if all_keccak_claims.is_empty() {
    None
  } else {
    Some(prove_keccak_consistency(&all_keccak_claims)?)
  };

  // Collect stack claims from all instructions (rw_counter is shared and already monotone
  // since the claims are emitted in execution order by the inspector).
  let all_stack_claims: Vec<StackAccessClaim> = itps
    .iter()
    .flat_map(|itp| itp.stack_claims.iter().cloned())
    .collect();
  let stack_proof = if all_stack_claims.is_empty() {
    None
  } else {
    Some(prove_stack_consistency(&all_stack_claims)?)
  };

  // Collect return/revert data from the last terminating instruction (if any).
  let return_data = itps
    .iter()
    .rev()
    .find_map(|itp| itp.return_data_claim.clone());

  Ok(BatchTransactionZkReceipt {
    lut_proof,
    preprocessed_vk,
    manifest,
    memory_proof,
    storage_proof,
    keccak_proof,
    stack_proof,
    return_data,
  })
}

/// Verify a batch ZK receipt.
///
/// **Verification steps (all deterministic except step 4):**
/// 1. Check `statements.len() == receipt.manifest.entries.len()`.
/// 2. For each i: verify statement semantics (SP transition correctness).
/// 3. For each i: re-derive `wff_instruction(opcode, inputs, outputs)` and
///    confirm it equals `receipt.manifest.entries[i].wff`.  This ties the
///    manifest's WFF claims to the actual EVM execution.
/// 4. STARK: re-derive batch public values from the manifest and call
///    [`verify_batch_lut_with_prep`].  The STARK proof binds the manifest
///    digest (`pis[2..10]`) to every committed ProofRow, proving arithmetic
///    correctness for all N instructions.
///
/// Returns `false` on any mismatch; the error reason is intentionally not
/// surfaced to avoid oracle-style leakage.
pub fn verify_batch_transaction_zk_receipt(
  statements: &[InstructionTransitionStatement],
  receipt: &BatchTransactionZkReceipt,
) -> bool {
  // 1. Count check.
  if statements.len() != receipt.manifest.entries.len() {
    return false;
  }

  // 2 & 3. Per-instruction semantic and WFF checks.
  for (stmt, entry) in statements.iter().zip(receipt.manifest.entries.iter()) {
    // Opcode consistency.
    if stmt.opcode != entry.opcode {
      return false;
    }

    // Statement-level SP/arity coherence.
    if !verify_statement_semantics(stmt) {
      return false;
    }

    // Bind manifest WFF to actual execution output.
    let expected_wff = match wff_instruction(stmt.opcode, &stmt.s_i.stack, &stmt.s_next.stack) {
      Some(wff) => wff,
      None => return false,
    };
    if expected_wff != entry.wff {
      return false;
    }
  }

  // 4. Batch STARK verification.
  // Re-derive pis independently from the manifest so the verifier never
  // trusts the prover-supplied public values directly.
  let lut_bind_pv =
    make_batch_receipt_binding_public_values(RECEIPT_BIND_TAG_LUT, &receipt.manifest.entries);
  if verify_batch_lut_with_prep(&receipt.lut_proof, &receipt.preprocessed_vk, &lut_bind_pv).is_err()
  {
    return false;
  }

  // 4.5. Out-of-circuit manifest row validation (Gap 3 + Gap 4).
  //
  // Gap 3: byte AND/OR/XOR rows must commit the correct result value
  //   (row.value == scalar0 op scalar1).  The LUT AIR binds out0 to
  //   prep[PREP_COL_VALUE] but does not constrain out0 = in0 op in1.
  //
  // Gap 4: arithmetic operands (U29/U24AddEq, U15MulEq) must be within
  //   their declared bit-width bounds.  M31 field arithmetic can satisfy
  //   the AIR sum equation with out-of-range inputs, so the check is
  //   enforced here rather than (only) in the STARK.
  if !validate_manifest_rows(&receipt.manifest.all_rows) {
    return false;
  }

  // 5. Memory consistency STARK (if present).
  if let Some(mem_proof) = &receipt.memory_proof {
    if !verify_memory_consistency(mem_proof) {
      return false;
    }
  }

  // 6. Storage consistency STARK (if present).
  if let Some(stor_proof) = &receipt.storage_proof {
    if !verify_storage_consistency(stor_proof) {
      return false;
    }
  }

  // 7. Stack consistency STARK (if present).
  if let Some(stk_proof) = &receipt.stack_proof {
    if !verify_stack_consistency(stk_proof) {
      return false;
    }
  }

  // 8. Keccak consistency STARK (if present).
  if let Some(kec_proof) = &receipt.keccak_proof {
    if !verify_keccak_consistency(kec_proof) {
      return false;
    }
  }

  // 9. Keccak ↔ memory cross-check (BUG-MISS-3).
  //
  // verify_keccak_consistency only checks keccak256(input_bytes) == output_hash.
  // Without this step a malicious prover could supply a keccak claim whose
  // input_bytes differ from what the memory proof holds at [offset, offset+size).
  // We cross-check every keccak claim's bytes against the memory write/read logs.
  if let (Some(kec_proof), Some(mem_proof)) =
    (&receipt.keccak_proof, &receipt.memory_proof)
  {
    if !validate_keccak_memory_cross_check(
      &kec_proof.log,
      &mem_proof.write_log,
      &mem_proof.read_log,
    ) {
      return false;
    }
  }

  // 10. SubCall inner_proof recursive verification (Phase 1).
  //
  // For every CALL/CALLCODE/DELEGATECALL/STATICCALL/CREATE/CREATE2 instruction
  // that carries an `inner_proof`, recursively verify the callee's execution
  // trace.  This provides callee soundness and enforces the return_data
  // binding between the caller claim and the callee receipt.
  for stmt in statements {
    if let Some(sc) = &stmt.sub_call_claim {
      if verify_sub_call_claim(sc, statements).is_err() {
        return false;
      }
    }
  }

  true
}

// ============================================================
// Recursive proving types (Phase 1 & 2 skeleton)
// ============================================================

/// Proof for a single execution segment (≤ window_size instructions).
///
/// A [`LeafReceipt`] is the base case of the [`ExecutionReceipt`] tree: one
/// [`BatchTransactionZkReceipt`] plus the Poseidon2-committed VM state at the
/// segment's entry and exit boundaries.
pub struct LeafReceipt {
  /// VM state commitment at segment entry.
  pub s_in: StateCommitment,
  /// VM state commitment at segment exit.
  pub s_out: StateCommitment,
  /// The batch STARK proof covering all arithmetic instructions in this
  /// segment.  `None` when the segment has no arithmetic opcodes (pure
  /// structural instructions like JUMP/PUSH need no batch proof).
  pub batch_receipt: Option<BatchTransactionZkReceipt>,
  /// Raw step proofs for this segment — required by the verifier to
  /// reconstruct [`InstructionTransitionStatement`]s from stored data.
  pub steps: Vec<InstructionTransitionProof>,
}

/// Aggregation of two adjacent [`ExecutionReceipt`]s via a LinkAir STARK.
///
/// The `link_proof` asserts `left.s_out == right.s_in`, chaining the two
/// sub-receipts into a single commitment-chain receipt.
pub struct AggregationReceipt {
  /// Outermost entry state (== `left.s_in`).
  pub s_in: StateCommitment,
  /// Outermost exit state (== `right.s_out`).
  pub s_out: StateCommitment,
  /// LinkAir STARK proof asserting `left.s_out == right.s_in`.
  pub link_proof: CircleStarkProof,
  pub left: Box<ExecutionReceipt>,
  pub right: Box<ExecutionReceipt>,
}

/// Recursive execution-proof tree node.
///
/// - [`Leaf`](ExecutionReceipt::Leaf): a single batch segment.
/// - [`Aggregated`](ExecutionReceipt::Aggregated): two sub-receipts linked by a
///   [`LinkAir`](crate::zk_proof::LinkAir) STARK.
pub enum ExecutionReceipt {
  Leaf(LeafReceipt),
  Aggregated(AggregationReceipt),
}

impl ExecutionReceipt {
  /// Outermost entry commitment of this receipt subtree.
  pub fn s_in(&self) -> &StateCommitment {
    match self {
      ExecutionReceipt::Leaf(l) => &l.s_in,
      ExecutionReceipt::Aggregated(a) => &a.s_in,
    }
  }

  /// Outermost exit commitment of this receipt subtree.
  pub fn s_out(&self) -> &StateCommitment {
    match self {
      ExecutionReceipt::Leaf(l) => &l.s_out,
      ExecutionReceipt::Aggregated(a) => &a.s_out,
    }
  }
}

// ── Phase 1: SubCall recursive verification ──────────────────────────

/// Extract an [`InstructionTransitionStatement`] from an
/// [`InstructionTransitionProof`] step, mirroring the same logic used by the
/// batch-prove path in `execute.rs`.
fn statement_from_proof_step(
  step: &InstructionTransitionProof,
) -> InstructionTransitionStatement {
  InstructionTransitionStatement {
    opcode: step.opcode,
    s_i: VmState {
      opcode: step.opcode,
      pc: step.pc,
      sp: step.stack_inputs.len(),
      stack: step.stack_inputs.clone(),
      memory_root: [0u8; 32],
    },
    s_next: VmState {
      opcode: step.opcode,
      pc: step.pc + 1,
      sp: step.stack_outputs.len(),
      stack: step.stack_outputs.clone(),
      memory_root: [0u8; 32],
    },
    accesses: Vec::new(),
    sub_call_claim: step.sub_call_claim.clone(),
  }
}

/// Verify a [`SubCallClaim`]'s `inner_proof`, if present.
///
/// Steps:
/// 1. Enforce `claim.depth < MAX_CALL_DEPTH`.
/// 2. If `inner_proof` is `Some`, build [`InstructionTransitionStatement`]s
///    from the inner steps for structural consistency checking.
/// 3. Confirm the callee's `return_data` matches `claim.return_data`
///    (binding the caller's claim to the callee's RETURN/REVERT).
/// 4. Recursively verify any nested sub-calls inside the inner proof,
///    so callee soundness propagates transitively (depth-bounded by
///    `MAX_CALL_DEPTH`).
///
/// Note: Full STARK-receipt re-verification of the inner proof will be wired
/// in Phase 2 once `commit_vm_state` and `prove_execution_chain` are complete.
pub fn verify_sub_call_claim(
  claim: &SubCallClaim,
  _outer_statements: &[InstructionTransitionStatement],
) -> Result<(), String> {
  if claim.depth >= MAX_CALL_DEPTH {
    return Err(format!(
      "SubCall depth {} exceeds MAX_CALL_DEPTH {}",
      claim.depth, MAX_CALL_DEPTH
    ));
  }

  if let Some(inner) = &claim.inner_proof {
    if inner.steps.is_empty() {
      return Err("SubCall inner_proof has no steps".to_string());
    }

    // Build statements from the inner proof's steps (used for sub-call
    // recursion below; STARK receipt verification deferred to Phase 2).
    let inner_stmts: Vec<InstructionTransitionStatement> =
      inner.steps.iter().map(statement_from_proof_step).collect();

    // Return-data binding: the last RETURN/REVERT step's bytes must match.
    let inner_return = inner
      .steps
      .iter()
      .rev()
      .find_map(|s| s.return_data_claim.as_ref());
    match inner_return {
      Some(rd) => {
        if rd.data != claim.return_data {
          return Err(format!(
            "SubCall return_data mismatch: inner={} bytes, claim={} bytes",
            rd.data.len(),
            claim.return_data.len()
          ));
        }
      }
      None => {
        // STOP or out-of-gas: caller expects empty return data.
        if !claim.return_data.is_empty() {
          return Err(
            "SubCall claim has non-empty return_data but inner proof has no RETURN"
              .to_string(),
          );
        }
      }
    }

    // Recursively verify any nested sub-calls inside the inner proof.
    for step in &inner.steps {
      if let Some(sc) = &step.sub_call_claim {
        verify_sub_call_claim(sc, &inner_stmts)?;
      }
    }
  }

  Ok(())
}

// ── Phase 2: Execution chain proving / verification ───────────────────

/// Split execution into window-sized segments, prove each segment with a
/// [`BatchTransactionZkReceipt`] (skipping pure-structural segments), then
/// aggregates all leaves into a binary tree via [`LinkAir`] link STARKs.
///
/// # Arguments
/// - `vm_state_seq`: K+1 [`VmState`] boundary snapshots for K segments.
/// - `step_seqs`:    K segment step vectors (one per segment).
///
/// # Returns
/// The root [`ExecutionReceipt`] of the binary aggregation tree.
pub fn prove_execution_chain(
  vm_state_seq: &[VmState],
  step_seqs: Vec<Vec<InstructionTransitionProof>>,
) -> Result<ExecutionReceipt, String> {
  let n = step_seqs.len();
  if n == 0 {
    return Err("prove_execution_chain: no segments".to_string());
  }
  if vm_state_seq.len() != n + 1 {
    return Err(format!(
      "prove_execution_chain: expected {} VmState boundaries for {} segments, got {}",
      n + 1,
      n,
      vm_state_seq.len()
    ));
  }

  // Step 1: Commit all boundary states at once.
  let commitments: Vec<StateCommitment> =
    vm_state_seq.iter().map(commit_vm_state).collect();

  // Step 2: Build one LeafReceipt per segment.
  let mut nodes: Vec<ExecutionReceipt> = Vec::with_capacity(n);
  for (i, steps) in step_seqs.into_iter().enumerate() {
    let has_semantic = steps.iter().any(|s| s.semantic_proof.is_some());
    let batch_receipt = if has_semantic {
      Some(prove_batch_transaction_zk_receipt(&steps)?)
    } else {
      None
    };
    let leaf = LeafReceipt {
      s_in: commitments[i].clone(),
      s_out: commitments[i + 1].clone(),
      batch_receipt,
      steps,
    };
    nodes.push(ExecutionReceipt::Leaf(leaf));
  }

  // Step 3: Binary tree aggregation — repeatedly merge adjacent pairs.
  while nodes.len() > 1 {
    let mut new_nodes: Vec<ExecutionReceipt> = Vec::with_capacity((nodes.len() + 1) / 2);
    let mut iter = nodes.into_iter();
    loop {
      match (iter.next(), iter.next()) {
        (Some(left), Some(right)) => {
          // The link STARK binds the junction: left.s_out must equal right.s_in.
          let junction = [(left.s_out().clone(), right.s_in().clone())];
          let link_proof = prove_link_stark(&junction, left.s_out(), right.s_in());
          let s_in = left.s_in().clone();
          let s_out = right.s_out().clone();
          new_nodes.push(ExecutionReceipt::Aggregated(AggregationReceipt {
            s_in,
            s_out,
            link_proof,
            left: Box::new(left),
            right: Box::new(right),
          }));
        }
        (Some(odd), None) => {
          // Odd node carries forward unchanged.
          new_nodes.push(odd);
          break;
        }
        (None, _) => break,
      }
    }
    nodes = new_nodes;
  }

  Ok(nodes.remove(0))
}

/// Recursively verify an [`ExecutionReceipt`] tree.
///
/// - **Leaf**: reconstructs [`InstructionTransitionStatement`]s from the stored
///   steps and calls [`verify_batch_transaction_zk_receipt`] (when a batch
///   receipt is present).
/// - **Aggregated**: verifies the junction link proof via [`verify_link_stark`]
///   using the actual junction states (`left.s_out`, `right.s_in`), then
///   recurses into both sub-receipts.
pub fn verify_execution_receipt(receipt: &ExecutionReceipt) -> Result<(), String> {
  match receipt {
    ExecutionReceipt::Leaf(leaf) => {
      if let Some(batch_receipt) = &leaf.batch_receipt {
        let stmts: Vec<InstructionTransitionStatement> = leaf
          .steps
          .iter()
          .filter(|s| s.semantic_proof.is_some())
          .map(statement_from_proof_step)
          .collect();
        if !verify_batch_transaction_zk_receipt(&stmts, batch_receipt) {
          return Err("Leaf batch receipt verification failed".to_string());
        }
      }
      Ok(())
    }
    ExecutionReceipt::Aggregated(agg) => {
      // Verify the link at the junction (left.s_out ↔ right.s_in), NOT the
      // overall chain boundaries — that is the public value the proof encodes.
      verify_link_stark(&agg.link_proof, agg.left.s_out(), agg.right.s_in())
        .map_err(|e| format!("LinkAir verify failed: {:?}", e))?;
      verify_execution_receipt(&agg.left)?;
      verify_execution_receipt(&agg.right)?;
      Ok(())
    }
  }
}

// ============================================================
// Tests
// ============================================================
