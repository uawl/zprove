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
  Proof, Term, WFF, compile_proof, infer_proof, prove_addmod, prove_byte,
  prove_div, prove_eq, prove_exp, prove_gt, prove_iszero, prove_lt, prove_mod, prove_mul,
  prove_mulmod, prove_pc_step, prove_sar, prove_sdiv, prove_sgt, prove_shl,
  prove_shr, prove_signextend, prove_slt, prove_smod,
  prove_and_sym, prove_or_sym, prove_xor_sym, prove_not_sym, prove_add_limb_sym, prove_sub_limb_sym,
  verify_compiled, check_wff_io_values,
  wff_addmod, wff_byte, wff_div, wff_eq, wff_exp, wff_gt, wff_iszero, wff_lt,
  wff_mod, wff_mul, wff_mulmod, wff_pc_step, wff_sar, wff_sdiv, wff_sgt,
  wff_shl, wff_shr, wff_signextend, wff_slt, wff_smod, wff_stack_inputs, wff_stack_outputs,
  wff_and_sym, wff_or_sym, wff_xor_sym, wff_not_sym, wff_add_limb_sym, wff_sub_limb_sym,
  OP_BYTE_AND_SYM,
};
use crate::zk_proof::{
  BatchInstructionMeta, BatchProofRowsManifest, CircleStarkConfig, CircleStarkProof,
  KeccakConsistencyProof, MemLogEntry, MemoryConsistencyProof, RECEIPT_BIND_TAG_LUT,
  RECEIPT_BIND_TAG_STACK, StackConsistencyProof, StateCommitment, StorageConsistencyProof,
  collect_byte_table_queries_from_rows, commit_vm_state, compute_wff_opcode_digest,
  make_batch_receipt_binding_public_values, make_receipt_binding_public_values,
  prove_batch_lut_with_prep, prove_keccak_consistency, prove_link_stark, prove_memory_consistency,
  prove_batch_stack_ir_with_prep, prove_stack_consistency, prove_stack_ir_with_prep, prove_stack_rw,
  prove_storage_consistency,
  setup_batch_proof_rows_preprocessed, setup_proof_rows_preprocessed,
  validate_keccak_memory_cross_check, validate_manifest_rows, verify_batch_lut_with_prep,
  verify_keccak_consistency, verify_link_stark, verify_memory_consistency,
  verify_batch_stack_ir_with_prep, verify_stack_consistency, verify_stack_ir_with_prep, verify_stack_rw,
  verify_storage_consistency,
  StackRwProof,
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

/// `inner_proof` is `None` (oracle mode): the claim is accepted with structural
/// checks only — verifying return_data binding and call depth.
/// When `Some`, `verify_sub_call_claim` also performs STARK receipt
/// re-verification via `verify_batch_transaction_zk_receipt` if `batch_receipt`
/// is present on the inner proof.
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
  /// Optional recursive proof of the callee execution trace.
  /// When `Some`, full structural and STARK verification is performed by
  /// `verify_sub_call_claim`.
  pub inner_proof: Option<Box<TransactionProof>>,
}

/// Copy-consistency witness for the MCOPY opcode (EIP-5656).
///
/// Records the source and destination offsets, size, and actual bytes copied
/// so the batch verifier can confirm that every src read claim and every dst
/// write claim in `memory_claims` carry the same word values — i.e., the copy
/// was faithful.
///
/// `src_rw_start` and `dst_rw_start` are the **batch-scoped** `rw_counter`
/// values of the first read/write claim emitted by this instruction.  These
/// are filled in by `prove_batch_transaction_zk_receipt` when it renumbers
/// the claims into a single monotone sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemCopyClaim {
  /// Byte offset of the source region (not necessarily 32-aligned).
  pub src_offset: u64,
  /// Byte offset of the destination region (not necessarily 32-aligned).
  pub dst_offset: u64,
  /// Number of bytes to copy.
  pub size: u64,
  /// The exact `size` bytes copied from source (host-extracted, used for
  /// cross-checking src reads and dst writes during verification).
  pub data: Vec<u8>,
  /// Batch-scoped `rw_counter` of the first src-range read claim.
  /// Set to 0 until `prove_batch_transaction_zk_receipt` runs.
  pub src_rw_start: u64,
  /// Number of 32-byte-aligned words in the src read range.
  pub src_word_count: usize,
  /// Batch-scoped `rw_counter` of the first dst-range write claim.
  /// Set to 0 until `prove_batch_transaction_zk_receipt` runs.
  pub dst_rw_start: u64,
  /// Number of 32-byte-aligned words in the dst write range.
  pub dst_word_count: usize,
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

/// Proof that one EVM instruction correctly transforms the stack.
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
  /// Non-empty for MLOAD, MSTORE, MSTORE8, MCOPY, and copy-family opcodes.
  pub memory_claims: Vec<MemAccessClaim>,
  /// MCOPY copy-consistency witness.  `Some` only for MCOPY instructions.
  /// Binds the src read claims and dst write claims together so the batch
  /// verifier can confirm every copied word is faithfully transferred.
  pub mcopy_claim: Option<MemCopyClaim>,
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
pub struct TransactionProof {
  pub steps: Vec<InstructionTransitionProof>,
  /// Block and transaction context values, used as public inputs to verify
  /// that `CallContextClaim`s produced during execution are consistent
  /// with the actual on-chain context.
  pub block_tx_context: BlockTxContext,
  /// Optional pre-proved batch receipt for STARK re-verification of the
  /// inner callee's execution.  `None` = oracle mode (structural-only).
  /// Populated by `prove_batch_transaction_zk_receipt` at call time.
  pub batch_receipt: Option<BatchTransactionZkReceipt>,
}

impl core::fmt::Debug for TransactionProof {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("TransactionProof")
      .field("steps", &self.steps)
      .field("block_tx_context", &self.block_tx_context)
      .field(
        "batch_receipt",
        &self.batch_receipt.as_ref().map(|_| "<BatchTransactionZkReceipt>"),
      )
      .finish()
  }
}

impl Clone for TransactionProof {
  fn clone(&self) -> Self {
    Self {
      steps: self.steps.clone(),
      block_tx_context: self.block_tx_context.clone(),
      batch_receipt: self.batch_receipt.clone(),
    }
  }
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
  /// MCOPY copy-consistency witness — `Some` only for MCOPY instructions.
  /// Copied from [`InstructionTransitionProof::mcopy_claim`] during batch proving.
  pub mcopy_claim: Option<MemCopyClaim>,
}

#[derive(Clone)]
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
/// - `stack_ir_proof`: batch StackIR STARK covering the concatenated ProofRows of ALL instructions.
/// - `lut_proof`: batch LUT STARK covering arithmetic of ALL instructions' ProofRows.
/// - `preprocessed_vk`: VK for the shared batch preprocessed matrix (used by both STARKs).
/// - `manifest`: per-instruction metadata (opcode, WFF, row range in `all_rows`).
/// - `memory_proof`: STARK proof that memory accesses are consistent across the batch.
///
/// Verification requires:
/// 0. `verify_stack_ir_with_prep(stack_ir_proof, preprocessed_vk, pis)` — STARK check.
/// 1. `verify_batch_lut_with_prep(lut_proof, preprocessed_vk, pis)` — STARK check.
/// 2. For each i: `infer_proof(pi_i) == manifest.entries[i].wff` — deterministic.
/// 3. `compute_batch_manifest_digest(&entries) == pis[2..10]` — deterministic.
/// 4. If `memory_proof` is Some: `verify_memory_consistency` — STARK check.
/// 5. If `stack_proof` is Some: `verify_stack_consistency` — STARK check.
/// 6. If `stack_rw_proof` is Some: `verify_stack_rw` — STARK check (chronological RW log).
#[derive(Clone)]
pub struct BatchTransactionZkReceipt {
  /// Single StackIR STARK proof covering the concatenated ProofRows of all N instructions.
  /// Uses the same shared preprocessed matrix as `lut_proof` (tag = RECEIPT_BIND_TAG_STACK).
  pub stack_ir_proof: CircleStarkProof,
  /// Single LUT STARK proof covering arithmetic of all N instructions.
  pub lut_proof: CircleStarkProof,
  /// Shared preprocessed verifier key (batch manifest committed here; used by both STARKs).
  pub preprocessed_vk: PreprocessedVerifierKey<CircleStarkConfig>,
  /// Per-instruction metadata and concatenated ProofRows.
  pub manifest: BatchProofRowsManifest,
  /// Memory consistency proof.  `None` when the batch has no MLOAD/MSTORE/MSTORE8.
  pub memory_proof: Option<MemoryConsistencyProof>,
  /// Storage consistency proof.  `None` when the batch has no SLOAD/SSTORE.
  pub storage_proof: Option<StorageConsistencyProof>,
  /// Stack consistency proof.  `None` when the batch has no stack-touching instructions.
  pub stack_proof: Option<StackConsistencyProof>,
  /// Stack read/write chronological consistency proof (StackRwAir + LogUp).
  /// Proves that every pop (read) observes the value from the most-recent push
  /// (write) at the same depth, and commits the full RW log via Poseidon hashes.
  /// `None` when the batch has no stack-touching instructions.
  pub stack_rw_proof: Option<StackRwProof>,
  /// KECCAK256 consistency proof.  `None` when the batch has no KECCAK256 instructions.
  pub keccak_proof: Option<KeccakConsistencyProof>,
  /// MCOPY copy-consistency witnesses collected from all MCOPY instructions
  /// in this batch.  Empty when the batch contains no MCOPY opcodes.
  ///
  /// `src_rw_start` / `dst_rw_start` inside each entry are set to
  /// batch-scoped `rw_counter` values by `prove_batch_transaction_zk_receipt`.
  pub mcopy_claims: Vec<MemCopyClaim>,
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
    opcode::RETURNDATACOPY
    | opcode::EXTCODECOPY
    | opcode::MCOPY
    | opcode::CALLDATACOPY
    | opcode::CODECOPY => 0,
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

// ─── private opcode classifiers ──────────────────────────────────────────────

/// Keccak256 opcode (SHA3 / 0x20).
fn is_keccak_opcode(op: u8) -> bool {
  op == opcode::KECCAK256
}

/// Inline keccak verification for `verify_proof`:
/// - opcode must be KECCAK256
/// - keccak_claim must be present
/// - keccak256(preimage) must equal output_hash
/// - output_hash must match the single stack output
fn verify_keccak_claim_inline(proof: &InstructionTransitionProof) -> bool {
  if !is_keccak_opcode(proof.opcode) {
    return false;
  }
  let Some(ref claim) = proof.keccak_claim else {
    return false;
  };
  let computed = crate::zk_proof::keccak256_bytes(&claim.input_bytes);
  if computed != claim.output_hash {
    return false;
  }
  if proof.stack_outputs.len() != 1 {
    return false;
  }
  proof.stack_outputs[0] == claim.output_hash
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

  if !check_output_semantics(statement.opcode, &statement.s_i.stack, &statement.s_next.stack) {
    return false;
  }

  true
}

/// Pre-LogUp placeholder: check that the claimed outputs are arithmetically
/// consistent with the inputs for opcodes not yet bound via ZK value binding.
fn check_output_semantics(op: u8, inputs: &[[u8; 32]], outputs: &[[u8; 32]]) -> bool {
  match op {
    opcode::ADD => add_u256_mod(&inputs[0], &inputs[1]) == outputs[0],
    opcode::SUB => add_u256_mod(&inputs[0], &negate_u256(&inputs[1])) == outputs[0],
    opcode::AND => (0..32).all(|i| inputs[0][i] & inputs[1][i] == outputs[0][i]),
    opcode::OR  => (0..32).all(|i| inputs[0][i] | inputs[1][i] == outputs[0][i]),
    opcode::XOR => (0..32).all(|i| inputs[0][i] ^ inputs[1][i] == outputs[0][i]),
    opcode::NOT => (0..32).all(|i| !inputs[0][i] == outputs[0][i]),
    _ => true,
  }
}

fn add_u256_mod(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
  let mut out = [0u8; 32];
  let mut carry = 0u16;
  for i in (0..32).rev() {
    let total = a[i] as u16 + b[i] as u16 + carry;
    out[i] = total as u8;
    carry = total >> 8;
  }
  out
}

fn negate_u256(a: &[u8; 32]) -> [u8; 32] {
  let mut flipped = [0u8; 32];
  for i in 0..32 {
    flipped[i] = !a[i];
  }
  let one = { let mut b = [0u8; 32]; b[31] = 1; b };
  add_u256_mod(&flipped, &one)
}

// ============================================================
// Proof generation per instruction
// ============================================================

/// Number of bytes consumed by an instruction (opcode byte + immediate operands).
/// Used to derive `PcAfter = PcBefore + instr_size` in `prove_pc_step`.
fn opcode_instr_size(op: u8) -> u32 {
  if op >= opcode::PUSH1 && op <= opcode::PUSH32 {
    (op - opcode::PUSH1 + 2) as u32
  } else {
    1
  }
}

/// Returns `true` for instructions where `PcAfter = PcBefore + instr_size`.
/// Returns `false` for jumps (JUMP / JUMPI) and terminators
/// (STOP / RETURN / REVERT / INVALID / SELFDESTRUCT).
fn has_linear_pc(op: u8) -> bool {
  !matches!(
    op,
    opcode::JUMP
    | opcode::JUMPI
    | opcode::STOP
    | opcode::RETURN
    | opcode::REVERT
    | opcode::INVALID
    | opcode::SELFDESTRUCT
  )
}

/// Generate a semantic proof for a single EVM instruction.
///
/// Returns `(proof, computed_outputs)`.  The caller should compare
/// `computed_outputs` against the actual EVM outputs to ensure consistency.
pub fn prove_instruction(op: u8, inputs: &[[u8; 32]], outputs: &[[u8; 32]]) -> Option<Proof> {
  if !has_expected_stack_arity(op, inputs, outputs) {
    return None;
  }
  // Stack I/O bindings are committed via the WFF hash in public inputs.
  // InputEq/OutputEq and PcStep rows are omitted; only the instruction core logic is proven.
  prove_instruction_core(op, inputs, outputs)
}

/// Core semantic proof (instruction-specific logic only, no stack/PC bindings).
fn prove_instruction_core(op: u8, inputs: &[[u8; 32]], outputs: &[[u8; 32]]) -> Option<Proof> {
  match op {
    opcode::ADD => Some(prove_add_limb_sym()),
    opcode::SUB => Some(prove_sub_limb_sym()),
    opcode::MUL => Some(prove_mul(&inputs[0], &inputs[1], &outputs[0])),
    opcode::DIV => Some(prove_div(&inputs[0], &inputs[1], &outputs[0])),
    opcode::MOD => Some(prove_mod(&inputs[0], &inputs[1], &outputs[0])),
    opcode::SDIV => Some(prove_sdiv(&inputs[0], &inputs[1], &outputs[0])),
    opcode::SMOD => Some(prove_smod(&inputs[0], &inputs[1], &outputs[0])),
    opcode::AND => Some(prove_and_sym(&inputs[0], &inputs[1])),
    opcode::OR  => Some(prove_or_sym(&inputs[0], &inputs[1])),
    opcode::XOR => Some(prove_xor_sym(&inputs[0], &inputs[1])),
    opcode::NOT => Some(prove_not_sym(&inputs[0])),
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
    opcode::ADDMOD => Some(prove_addmod(
      &inputs[0],
      &inputs[1],
      &inputs[2],
      &outputs[0],
    )),
    opcode::MULMOD => Some(prove_mulmod(
      &inputs[0],
      &inputs[1],
      &inputs[2],
      &outputs[0],
    )),
    opcode::EXP => Some(prove_exp(&inputs[0], &inputs[1], &outputs[0])),

    // ── Per-opcode dedicated axioms ─────────────────────────────────────────

    // Memory read/write
    opcode::MLOAD => Some(Proof::MloadAxiom),
    opcode::MSTORE => Some(Proof::MstoreAxiom {
      opcode: op,
    }),
    opcode::MSTORE8 => Some(Proof::MstoreAxiom {
      opcode: op,
    }),

    // Memory-copy family
    opcode::RETURNDATACOPY | opcode::EXTCODECOPY | opcode::MCOPY => {
      Some(Proof::MemCopyAxiom { opcode: op })
    }
    opcode::CALLDATACOPY | opcode::CODECOPY => {
      Some(Proof::MemCopyAxiom { opcode: op })
    }

    // Storage
    opcode::SLOAD => Some(Proof::SloadAxiom),
    opcode::SSTORE => Some(Proof::SstoreAxiom),
    opcode::TLOAD => Some(Proof::TransientAxiom {
      opcode: op,
    }),
    opcode::TSTORE => Some(Proof::TransientAxiom {
      opcode: op,
    }),

    // Keccak
    opcode::KECCAK256 => Some(Proof::KeccakAxiom),

    // Environment / call-context opcodes
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
    | opcode::BLOBBASEFEE => Some(Proof::EnvAxiom {
      opcode: op,
    }),

    // External state
    opcode::BLOCKHASH | opcode::EXTCODESIZE | opcode::EXTCODEHASH | opcode::BALANCE => {
      Some(Proof::ExternalStateAxiom {
        opcode: op,
      })
    }
    opcode::BLOBHASH => Some(Proof::ExternalStateAxiom {
      opcode: op,
    }),

    // Terminate
    opcode::RETURN | opcode::REVERT => Some(Proof::TerminateAxiom {
      opcode: op,
    }),

    // Sub-call / create
    opcode::CALL | opcode::CALLCODE | opcode::DELEGATECALL | opcode::STATICCALL => {
      Some(Proof::CallAxiom {
        opcode: op,
      })
    }
    opcode::CREATE | opcode::CREATE2 => Some(Proof::CreateAxiom {
      opcode: op,
    }),
    opcode::SELFDESTRUCT => Some(Proof::SelfdestructAxiom),

    // Log
    opcode::LOG0 | opcode::LOG1 | opcode::LOG2 | opcode::LOG3 | opcode::LOG4 => {
      Some(Proof::LogAxiom {
        opcode: op,
      })
    }

    // INVALID
    opcode::INVALID => Some(Proof::StructuralAxiom { opcode: op }),

    // Structural: STOP, POP, JUMP, JUMPI, JUMPDEST, PUSH*, DUP*, SWAP*
    op if op == opcode::STOP
      || op == opcode::POP
      || op == opcode::JUMP
      || op == opcode::JUMPI
      || op == opcode::JUMPDEST
      || (opcode::PUSH0..=opcode::PUSH32).contains(&op) => {
      Some(Proof::PushAxiom)
    }
    op if (opcode::DUP1..=opcode::DUP16).contains(&op) => {
      let depth = op - opcode::DUP1 + 1;
      Some(Proof::DupAxiom { depth })
    }
    op if (opcode::SWAP1..=opcode::SWAP16).contains(&op) => {
      let depth = op - opcode::SWAP1 + 1;
      Some(Proof::SwapAxiom { depth })
    }

    _ => Some(Proof::StructuralAxiom { opcode: op }),
  }
}

// ============================================================
// WFF generation
// ============================================================

pub fn wff_instruction(op: u8, inputs: &[[u8; 32]], outputs: &[[u8; 32]]) -> Option<WFF> {
  if !has_expected_stack_arity(op, inputs, outputs) {
    return None;
  }
  let core = wff_instruction_core(op, inputs, outputs)?;
  let with_outputs = WFF::And(
    Box::new(wff_stack_outputs(outputs)),
    Box::new(core),
  );
  let with_pc = if has_linear_pc(op) {
    WFF::And(
      Box::new(wff_pc_step(opcode_instr_size(op))),
      Box::new(with_outputs),
    )
  } else {
    with_outputs
  };
  Some(WFF::And(
    Box::new(wff_stack_inputs(inputs)),
    Box::new(with_pc),
  ))
}

/// Core WFF (instruction-specific logic only, no stack/PC bindings).
fn wff_instruction_core(op: u8, inputs: &[[u8; 32]], outputs: &[[u8; 32]]) -> Option<WFF> {
  match op {
    opcode::ADD => Some(wff_add_limb_sym()),
    opcode::SUB => Some(wff_sub_limb_sym()),
    opcode::MUL => Some(wff_mul(&inputs[0], &inputs[1], &outputs[0])),
    opcode::DIV => Some(wff_div(&inputs[0], &inputs[1], &outputs[0])),
    opcode::MOD => Some(wff_mod(&inputs[0], &inputs[1], &outputs[0])),
    opcode::SDIV => Some(wff_sdiv(&inputs[0], &inputs[1], &outputs[0])),
    opcode::SMOD => Some(wff_smod(&inputs[0], &inputs[1], &outputs[0])),
    opcode::AND => Some(wff_and_sym(&inputs[0], &inputs[1], &outputs[0])),
    opcode::OR  => Some(wff_or_sym(&inputs[0], &inputs[1], &outputs[0])),
    opcode::XOR => Some(wff_xor_sym(&inputs[0], &inputs[1], &outputs[0])),
    opcode::NOT => Some(wff_not_sym(&inputs[0], &outputs[0])),
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

    // ── Per-opcode axiom WFFs: Equal(OutputTerm{0,0}, OutputTerm{0,0}) ──────────────
    opcode::MLOAD
    | opcode::MSTORE
    | opcode::MSTORE8
    | opcode::RETURNDATACOPY
    | opcode::EXTCODECOPY
    | opcode::MCOPY
    | opcode::CALLDATACOPY
    | opcode::CODECOPY
    | opcode::SLOAD
    | opcode::SSTORE
    | opcode::TLOAD
    | opcode::TSTORE
    | opcode::KECCAK256
    | opcode::CALLER
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
    | opcode::BLOCKHASH
    | opcode::EXTCODESIZE
    | opcode::EXTCODEHASH
    | opcode::BALANCE
    | opcode::BLOBHASH
    | opcode::RETURN
    | opcode::REVERT
    | opcode::CALL
    | opcode::CALLCODE
    | opcode::DELEGATECALL
    | opcode::STATICCALL
    | opcode::CREATE
    | opcode::CREATE2
    | opcode::SELFDESTRUCT
    | opcode::LOG0
    | opcode::LOG1
    | opcode::LOG2
    | opcode::LOG3
    | opcode::LOG4
    | opcode::INVALID => Some(WFF::Equal(
      Box::new(Term::OutputTerm { stack_idx: 0, byte_idx: 0, value: 0 }),
      Box::new(Term::OutputTerm { stack_idx: 0, byte_idx: 0, value: 0 }),
    )),
    _ => Some(WFF::Equal(
      Box::new(Term::OutputTerm { stack_idx: 0, byte_idx: 0, value: 0 }),
      Box::new(Term::OutputTerm { stack_idx: 0, byte_idx: 0, value: 0 }),
    )),
  }
}

/// WFF containing only the PcStep derivation and the instruction-specific core
/// formula — without stack I/O binding terms (`InputEq`/`OutputEq`).
///
/// This exactly matches `infer_proof(prove_instruction(op, inputs, outputs))`,
/// which is useful for tests that verify the proof tree is internally consistent.
pub fn wff_instruction_core_only(
  op: u8,
  inputs: &[[u8; 32]],
  outputs: &[[u8; 32]],
) -> Option<WFF> {
  if !has_expected_stack_arity(op, inputs, outputs) {
    return None;
  }
  // Returns only the instruction-specific core formula —
  // no PcStep, no InputEq/OutputEq wrappers.
  // Matches infer_proof(prove_instruction(op, inputs, outputs)).
  wff_instruction_core(op, inputs, outputs)
}

// ============================================================
// Verification (for testing)
// ============================================================

pub fn verify_proof(proof: &InstructionTransitionProof) -> bool {
  if !has_expected_stack_arity(proof.opcode, &proof.stack_inputs, &proof.stack_outputs) {
    return false;
  }

  // Pre-LogUp placeholder: reject proofs whose stated outputs are
  // arithmetically inconsistent with the inputs.
  if !check_output_semantics(proof.opcode, &proof.stack_inputs, &proof.stack_outputs) {
    return false;
  }

  // prove_instruction no longer emits InputEq/OutputEq rows; stack I/O binding is
  // enforced via the WFF hash in public inputs during ZK receipt verification.
  // Therefore we compare infer_proof(..) against wff_instruction_core_only(..).
  let expected_core_wff =
    wff_instruction_core_only(proof.opcode, &proof.stack_inputs, &proof.stack_outputs);
  match (&proof.semantic_proof, expected_core_wff) {
    (Some(sem_proof), Some(wff)) => {
      let Ok(wff_result) = infer_proof(sem_proof) else {
        return false;
      };
      // Verify every InputTerm/OutputTerm claim in the inferred WFF against
      // the declared stack inputs/outputs.
      if !check_wff_io_values(&wff_result, &proof.stack_inputs, &proof.stack_outputs) {
        return false;
      }
      wff == wff_result
    }
    (None, None) => false, // All opcodes now have a WFF; proof must be present or auto-generated.
    (None, Some(expected_wff)) => {
      // Auto-generation of the proof is only allowed for axiom opcodes.
      // Arithmetic / logic opcodes (ADD, SUB, …) require an explicit semantic_proof.
      let Some(auto_proof) = prove_instruction(
        proof.opcode,
        &proof.stack_inputs,
        &proof.stack_outputs,
      ) else {
        return false;
      };
      // Check that the auto-generated proof contains no symbolic op rows.
      // Symbolic rows (OP_BYTE_AND_SYM, OP_ADD_BYTE_SYM, OP_CARRY_EQ, …)
      // are used only in arithmetic/logic proofs, which must be supplied
      // explicitly via semantic_proof.  Axiom/structural opcodes (PUSH*, LOG*,
      // ENV*, …) produce only structural rows.
      let auto_rows = compile_proof(&auto_proof);
      if auto_rows.iter().any(|r| r.op >= OP_BYTE_AND_SYM) {
        return false; // Arithmetic/logic proofs must be supplied explicitly.
      }
      let Ok(wff_result) = infer_proof(&auto_proof) else {
        return false;
      };
      if !check_wff_io_values(&wff_result, &proof.stack_inputs, &proof.stack_outputs) {
        return false;
      }
      if expected_wff != wff_result {
        return false;
      }
      // KECCAK256 additionally requires a valid keccak_claim that
      // binds the stack output to the hash of the preimage bytes.
      if proof.opcode == opcode::KECCAK256 {
        verify_keccak_claim_inline(proof)
      } else {
        true
      }
    }
    (Some(_), None) => false, // Proof present but no WFF expected — reject
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
  if !has_expected_stack_arity(proof.opcode, &proof.stack_inputs, &proof.stack_outputs) {
    return false;
  }

  // KECCAK256 additionally requires a valid keccak_claim that binds the
  // stack output to the hash of the preimage bytes.
  if proof.opcode == opcode::KECCAK256 && !verify_keccak_claim_inline(proof) {
    return false;
  }

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
    mcopy_claim: proof.mcopy_claim.clone(),
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

  // Per-opcode axiom: if semantic_proof is None, auto-generate the canonical
  // per-opcode axiom proof so the LUT STARK sees a concrete proof term.
  let oracle_proof_holder;
  let semantic_proof: &Proof = match &proof.semantic_proof {
    Some(p) => p,
    None => {
      if let Some(auto) = prove_instruction(proof.opcode, &proof.stack_inputs, &proof.stack_outputs) {
        oracle_proof_holder = auto;
        &oracle_proof_holder
      } else {
        return Err(format!(
          "opcode 0x{:02x} has no semantic proof and no axiom proof",
          proof.opcode
        ));
      }
    }
  };

  let expected_wff = wff_instruction_core_only(proof.opcode, &proof.stack_inputs, &proof.stack_outputs)
    .ok_or_else(|| format!("unsupported opcode for ZKP receipt: 0x{:02x}", proof.opcode))?;

  let rows = compile_proof(semantic_proof);
  let stack_bind_pv =
    make_receipt_binding_public_values(RECEIPT_BIND_TAG_STACK, proof.opcode, &expected_wff);

  // StackIR proof: single-instruction setup (unchanged).
  let (prep_data, preprocessed_vk) = setup_proof_rows_preprocessed(&rows, &stack_bind_pv)?;
  let stack_ir_proof = prove_stack_ir_with_prep(&rows, &prep_data, &stack_bind_pv)?;

  // LUT proof: route through batch (BatchLutKernelAirWithPrep) for N=1.
  // Use the authoritative WFF from wff_instruction (includes stack I/O bindings).
  let lut_manifest =
    build_batch_manifest_from_wffs(&[(proof.opcode, expected_wff.clone(), semantic_proof)])?;
  let lut_bind_pv =
    make_batch_receipt_binding_public_values(RECEIPT_BIND_TAG_LUT, &lut_manifest.entries);
  let (lut_prep_data, lut_preprocessed_vk) =
    setup_batch_proof_rows_preprocessed(&lut_manifest, &lut_bind_pv)?;
  let lut_kernel_proof = prove_batch_lut_with_prep(&lut_manifest, &lut_prep_data, &lut_bind_pv)?;

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
#[allow(clippy::type_complexity)]
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

  let expected_wff = wff_instruction_core_only(proof.opcode, &proof.stack_inputs, &proof.stack_outputs)
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
  let lut_manifest =
    build_batch_manifest_from_wffs(&[(proof.opcode, expected_wff.clone(), semantic_proof)])?;
  let lut_bind_pv =
    make_batch_receipt_binding_public_values(RECEIPT_BIND_TAG_LUT, &lut_manifest.entries);
  let (lut_prep_data, lut_preprocessed_vk) =
    setup_batch_proof_rows_preprocessed(&lut_manifest, &lut_bind_pv)?;
  let lut_kernel_proof = prove_batch_lut_with_prep(&lut_manifest, &lut_prep_data, &lut_bind_pv)?;
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

  let expected_wff = match wff_instruction_core_only(
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
  let lut_bind_pv = make_batch_receipt_binding_public_values(RECEIPT_BIND_TAG_LUT, &batch_entries);
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
  if let Some(bp) = &receipt.byte_table_proof
    && crate::byte_table::verify_byte_table(bp).is_err() {
      return false;
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

/// Like [`build_batch_manifest`] but accepts an explicit WFF per instruction
/// instead of inferring it from the proof tree.
///
/// Used internally by proving paths that already have the authoritative WFF
/// from [`wff_instruction`] — in particular when the proof no longer contains
/// `InputEq`/`OutputEq` axiom rows and `infer_proof` would yield a smaller WFF
/// that does not include the stack I/O bindings.
fn build_batch_manifest_from_wffs(
  items: &[(u8, WFF, &Proof)],
) -> Result<BatchProofRowsManifest, String> {
  if items.is_empty() {
    return Err("build_batch_manifest_from_wffs: empty item list".to_string());
  }

  let mut entries: Vec<BatchInstructionMeta> = Vec::with_capacity(items.len());
  let mut all_rows: Vec<ProofRow> = Vec::new();

  for (opcode, wff, proof) in items {
    let rows = compile_proof(proof);
    let row_start = all_rows.len();
    let row_count = rows.len();
    all_rows.extend(rows);
    let wff_digest = compute_wff_opcode_digest(*opcode, wff);
    entries.push(BatchInstructionMeta {
      opcode: *opcode,
      wff: wff.clone(),
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
  let items: Vec<(u8, WFF, &Proof)> = itps
    .iter()
    .filter_map(|itp| {
      let wff = wff_instruction(itp.opcode, &itp.stack_inputs, &itp.stack_outputs)?;
      itp.semantic_proof.as_ref().map(|p| (itp.opcode, wff, p))
    })
    .collect();

  if items.is_empty() {
    return Err("prove_batch_transaction_zk_receipt: no semantic proofs in batch".to_string());
  }

  let manifest = build_batch_manifest_from_wffs(&items)?;
  let lut_bind_pv =
    make_batch_receipt_binding_public_values(RECEIPT_BIND_TAG_LUT, &manifest.entries);
  let (prep_data, preprocessed_vk) = setup_batch_proof_rows_preprocessed(&manifest, &lut_bind_pv)?;

  // StackIR batch proof: prove all instructions' ProofRows in one STARK call.
  // Reuses the same `prep_data` as the LUT proof — both AIRs share the same
  // preprocessed matrix; only the tag in `pis[0]` differs.
  let stack_bind_pv =
    make_batch_receipt_binding_public_values(RECEIPT_BIND_TAG_STACK, &manifest.entries);
  let stack_ir_proof = prove_batch_stack_ir_with_prep(&manifest, &prep_data, &stack_bind_pv)?;

  let lut_proof = prove_batch_lut_with_prep(&manifest, &prep_data, &lut_bind_pv)?;

  // Collect memory access claims from all instructions, assigning monotone rw_counters.
  // MCOPY claims are translated so that their src_rw_start / dst_rw_start reflect the
  // batch-scoped counter used when building the MemoryConsistencyAir trace.
  let mut all_mem_claims: Vec<MemAccessClaim> = Vec::new();
  let mut all_stor_claims: Vec<StorageAccessClaim> = Vec::new();
  let mut all_keccak_claims: Vec<KeccakClaim> = Vec::new();
  let mut all_mcopy_claims: Vec<MemCopyClaim> = Vec::new();
  let mut rw_counter: u64 = 0;
  for itp in itps {
    // Track the batch rw_counter for the first read/write claim emitted by
    // this instruction so MCOPY claims can be pinned to the correct log entries.
    let rw_before = rw_counter;
    for claim in &itp.memory_claims {
      rw_counter += 1;
      all_mem_claims.push(MemAccessClaim {
        rw_counter,
        addr: claim.addr,
        is_write: claim.is_write,
        value: claim.value,
      });
    }
    // Translate this instruction’s MCOPY claim into batch-scoped rw_counters.
    if let Some(ref mc) = itp.mcopy_claim {
      // Read claims precede write claims in the claim list (see build_memory_and_return_claims).
      // Count the number of reads (src) and writes (dst) in itp.memory_claims.
      let src_count = itp.memory_claims.iter().filter(|c| !c.is_write).count();
      let dst_count = itp.memory_claims.iter().filter(|c| c.is_write).count();
      // Batch rw_counters: first src read = rw_before + 1; first dst write = rw_before + src_count + 1.
      all_mcopy_claims.push(MemCopyClaim {
        src_offset: mc.src_offset,
        dst_offset: mc.dst_offset,
        size: mc.size,
        data: mc.data.clone(),
        src_rw_start: rw_before + 1,
        src_word_count: src_count,
        dst_rw_start: rw_before + src_count as u64 + 1,
        dst_word_count: dst_count,
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
  // Stack RW chronological proof: proves that every pop sees the most-recent
  // push at the same depth, with the full ordered log committed via Poseidon.
  let stack_rw_proof = if all_stack_claims.is_empty() {
    None
  } else {
    Some(prove_stack_rw(&all_stack_claims)?)
  };

  // Collect return/revert data from the last terminating instruction (if any).
  let return_data = itps
    .iter()
    .rev()
    .find_map(|itp| itp.return_data_claim.clone());

  Ok(BatchTransactionZkReceipt {
    stack_ir_proof,
    lut_proof,
    preprocessed_vk,
    manifest,
    memory_proof,
    storage_proof,
    keccak_proof,
    stack_proof,
    stack_rw_proof,
    mcopy_claims: all_mcopy_claims,
    return_data,
  })
}

// ── MCOPY copy-consistency helpers ───────────────────────────────────────────

/// Extract exactly `size` bytes starting at `offset` from a slice of
/// memory-log entries (sorted or unsorted — we use a HashMap).
///
/// Only entries whose `rw_counter` falls in `[rw_start, rw_start + word_count)`
/// are considered, which pins the extraction to a single instruction's claims.
///
/// Returns `None` if any required 32-byte word is missing from the log.
fn extract_bytes_from_mem_log(
  log: &[MemLogEntry],
  offset: u64,
  size: u64,
  rw_start: u64,
  word_count: usize,
) -> Option<Vec<u8>> {
  if size == 0 {
    return Some(vec![]);
  }
  let rw_end = rw_start + word_count as u64;
  let word_map: std::collections::HashMap<u64, [u8; 32]> = log
    .iter()
    .filter(|e| e.rw_counter >= rw_start && e.rw_counter < rw_end)
    .map(|e| (e.addr, e.value))
    .collect();

  let start = offset as usize;
  let end = start + size as usize;
  let mut result = vec![0u8; size as usize];
  let start_word = (start / 32) * 32;
  let end_word = end.div_ceil(32) * 32;
  let mut word_addr = start_word;
  while word_addr < end_word {
    let word = word_map.get(&(word_addr as u64))?;
    let copy_start = start.max(word_addr);
    let copy_end = end.min(word_addr + 32);
    if copy_start < copy_end {
      let result_off = copy_start - start;
      let word_off = copy_start - word_addr;
      result[result_off..result_off + (copy_end - copy_start)]
        .copy_from_slice(&word[word_off..word_off + (copy_end - copy_start)]);
    }
    word_addr += 32;
  }
  Some(result)
}

// ── Oracle cross-validation helpers ─────────────────────────────────────────

/// Convert a 32-byte big-endian EVM word to a u64 memory address.
///
/// EVM memory addresses fit in 64 bits in practice; we take the low 8 bytes.
#[inline]
fn evm_word_to_addr(word: &[u8; 32]) -> u64 {
  u64::from_be_bytes(word[24..32].try_into().expect("slice is 8 bytes"))
}

/// For every MLOAD statement, verify the committed output value appears as a
/// matching `(addr, value)` pair in the memory consistency proof's read log.
///
/// Uses a multiset countdown so duplicate MLOAD reads to the same address with
/// the same value are each matched to a distinct log entry.
///
/// Returns `false` if any MLOAD output is not present in the read log.
fn validate_oracle_mload_reads(
  statements: &[InstructionTransitionStatement],
  mem_proof: &MemoryConsistencyProof,
) -> bool {
  use std::collections::HashMap;
  let mut counts: HashMap<(u64, [u8; 32]), usize> = HashMap::new();
  for e in &mem_proof.read_log {
    *counts.entry((e.addr, e.value)).or_default() += 1;
  }
  for stmt in statements.iter().filter(|s| s.opcode == opcode::MLOAD) {
    if stmt.s_i.stack.is_empty() || stmt.s_next.stack.is_empty() {
      return false;
    }
    let key = (evm_word_to_addr(&stmt.s_i.stack[0]), stmt.s_next.stack[0]);
    match counts.get_mut(&key) {
      Some(c) if *c > 0 => *c -= 1,
      _ => return false,
    }
  }
  true
}

/// For every MSTORE / MSTORE8 statement, verify the committed (addr, value)
/// pair appears in the memory consistency proof's write log.
fn validate_oracle_mstore_writes(
  statements: &[InstructionTransitionStatement],
  mem_proof: &MemoryConsistencyProof,
) -> bool {
  use std::collections::HashMap;
  let mut counts: HashMap<(u64, [u8; 32]), usize> = HashMap::new();
  for e in &mem_proof.write_log {
    *counts.entry((e.addr, e.value)).or_default() += 1;
  }
  for stmt in statements
    .iter()
    .filter(|s| s.opcode == opcode::MSTORE || s.opcode == opcode::MSTORE8)
  {
    if stmt.s_i.stack.len() < 2 {
      return false;
    }
    let key = (evm_word_to_addr(&stmt.s_i.stack[0]), stmt.s_i.stack[1]);
    match counts.get_mut(&key) {
      Some(c) if *c > 0 => *c -= 1,
      _ => return false,
    }
  }
  true
}

/// For every SLOAD statement, verify the committed output value appears as a
/// matching `(slot, value)` pair in the storage consistency proof's read log.
fn validate_oracle_sload_reads(
  statements: &[InstructionTransitionStatement],
  stor_proof: &StorageConsistencyProof,
) -> bool {
  use std::collections::HashMap;
  // Key is (slot[32], value[32]); contract is omitted here because the
  // statement does not carry the contract address — the StorageConsistencyAir
  // already binds the contract via its rw_counter multiset.
  let mut counts: HashMap<([u8; 32], [u8; 32]), usize> = HashMap::new();
  for e in &stor_proof.read_log {
    *counts.entry((e.slot, e.value)).or_default() += 1;
  }
  for stmt in statements.iter().filter(|s| s.opcode == opcode::SLOAD) {
    if stmt.s_i.stack.is_empty() || stmt.s_next.stack.is_empty() {
      return false;
    }
    let key = (stmt.s_i.stack[0], stmt.s_next.stack[0]);
    match counts.get_mut(&key) {
      Some(c) if *c > 0 => *c -= 1,
      _ => return false,
    }
  }
  true
}

/// For every SSTORE statement, verify the committed (slot, value) pair appears
/// in the storage consistency proof's write log.
fn validate_oracle_sstore_writes(
  statements: &[InstructionTransitionStatement],
  stor_proof: &StorageConsistencyProof,
) -> bool {
  use std::collections::HashMap;
  let mut counts: HashMap<([u8; 32], [u8; 32]), usize> = HashMap::new();
  for e in &stor_proof.write_log {
    *counts.entry((e.slot, e.value)).or_default() += 1;
  }
  for stmt in statements.iter().filter(|s| s.opcode == opcode::SSTORE) {
    if stmt.s_i.stack.len() < 2 {
      return false;
    }
    let key = (stmt.s_i.stack[0], stmt.s_i.stack[1]);
    match counts.get_mut(&key) {
      Some(c) if *c > 0 => *c -= 1,
      _ => return false,
    }
  }
  true
}

/// For every KECCAK256 statement, verify that the committed output hash
/// matches the corresponding entry in the keccak consistency proof's log.
///
/// The keccak log is ordered by execution (= rw_counter order), matching the
/// sequential order in which KECCAK256 statements appear.
fn validate_oracle_keccak_outputs(
  statements: &[InstructionTransitionStatement],
  kec_proof: &KeccakConsistencyProof,
) -> bool {
  let keccak_stmts: Vec<_> = statements
    .iter()
    .filter(|s| s.opcode == opcode::KECCAK256)
    .collect();
  if keccak_stmts.len() != kec_proof.log.len() {
    return false;
  }
  for (stmt, entry) in keccak_stmts.iter().zip(kec_proof.log.iter()) {
    // stack_outputs[0] must equal the hash recorded by KeccakConsistencyProof.
    if stmt.s_next.stack.is_empty() {
      return false;
    }
    if stmt.s_next.stack[0] != entry.output_hash {
      return false;
    }
  }
  true
}

/// Verify that an MCOPY instruction faithfully copied its source bytes to
/// the destination — i.e., that every src read and dst write in the memory
/// proof carry the same data as `mc.data`.
///
/// Security guarantee: combined with `MemoryConsistencyAir` (which ensures
/// each read reflects the most-recent write), this prevents a malicious
/// prover from supplying mismatched src/dst values in a MCOPY claim.
fn validate_mcopy_copy_consistency(
  mc: &MemCopyClaim,
  read_log: &[MemLogEntry],
  write_log: &[MemLogEntry],
) -> bool {
  let src_bytes = extract_bytes_from_mem_log(
    read_log,
    mc.src_offset,
    mc.size,
    mc.src_rw_start,
    mc.src_word_count,
  );
  let dst_bytes = extract_bytes_from_mem_log(
    write_log,
    mc.dst_offset,
    mc.size,
    mc.dst_rw_start,
    mc.dst_word_count,
  );
  match (src_bytes, dst_bytes) {
    (Some(s), Some(d)) => s == mc.data && d == mc.data,
    _ => false,
  }
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

  // 4a. StackIR STARK: all instructions' ProofRows committed in one proof.
  let stack_ir_bind_pv =
    make_batch_receipt_binding_public_values(RECEIPT_BIND_TAG_STACK, &receipt.manifest.entries);
  if verify_batch_stack_ir_with_prep(
    &receipt.stack_ir_proof,
    &receipt.preprocessed_vk,
    &stack_ir_bind_pv,
  )
  .is_err()
  {
    return false;
  }

  // 4b. LUT STARK: arithmetic correctness for all ProofRows.
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
  if let Some(mem_proof) = &receipt.memory_proof
    && !verify_memory_consistency(mem_proof) {
      return false;
    }

  // 6. Storage consistency STARK (if present).
  if let Some(stor_proof) = &receipt.storage_proof
    && !verify_storage_consistency(stor_proof) {
      return false;
    }

  // 7. Stack consistency STARK (if present).
  if let Some(stk_proof) = &receipt.stack_proof
    && !verify_stack_consistency(stk_proof) {
      return false;
    }

  // 7b. Stack RW chronological STARK (if present).
  if let Some(rw_proof) = &receipt.stack_rw_proof
    && !verify_stack_rw(rw_proof) {
      return false;
    }

  // 8. Keccak consistency STARK (if present).
  if let Some(kec_proof) = &receipt.keccak_proof
    && !verify_keccak_consistency(kec_proof) {
      return false;
    }

  // 9. Keccak ↔ memory cross-check (BUG-MISS-3).
  //
  // verify_keccak_consistency only checks keccak256(input_bytes) == output_hash.
  // Without this step a malicious prover could supply a keccak claim whose
  // input_bytes differ from what the memory proof holds at [offset, offset+size).
  // We cross-check every keccak claim's bytes against the memory write/read logs.
  if let (Some(kec_proof), Some(mem_proof)) = (&receipt.keccak_proof, &receipt.memory_proof)
    && !validate_keccak_memory_cross_check(
      &kec_proof.log,
      &mem_proof.write_log,
      &mem_proof.read_log,
    ) {
      return false;
    }

  // 10. MCOPY copy-consistency cross-check.
  //
  // MemoryConsistencyAir validates each read/write claim independently, but
  // does NOT assert that the value read from src equals the value written to
  // dst.  Without this step a malicious prover could produce a MCOPY whose
  // destination bytes differ from the source bytes while both read and write
  // claims individually satisfy the memory multiset.
  //
  // For each MemCopyClaim we locate the exact memory-log entries (pinpointed
  // by batch-scoped rw_counter) and verify:
  //   extract_bytes(read_log[src_rw_start..], src_offset, size) == mc.data
  //   extract_bytes(write_log[dst_rw_start..], dst_offset, size) == mc.data
  if !receipt.mcopy_claims.is_empty() {
    let mem_proof = match &receipt.memory_proof {
      Some(p) => p,
      None => return false, // mcopy_claims require a memory proof
    };
    for mc in &receipt.mcopy_claims {
      if !validate_mcopy_copy_consistency(mc, &mem_proof.read_log, &mem_proof.write_log) {
        return false;
      }
    }
  }

  // 11. Oracle cross-validation: bind oracle WFF output values to their
  //     respective consistency AIR proofs (closes Gaps 1, 5-read, 5-write).

  // 11a. MLOAD oracle ↔ MemoryConsistencyProof.read_log
  //
  // The oracle WFF for MLOAD encodes (addr, value_read) in the manifest
  // digest.  We verify each MLOAD statement's committed output appears in
  // the memory read log, preventing a prover from claiming a different value
  // than what MemoryConsistencyAir proved was read.
  let has_mload = statements.iter().any(|s| s.opcode == opcode::MLOAD);
  if has_mload {
    let mem_proof = match &receipt.memory_proof {
      Some(p) => p,
      None => return false,
    };
    if !validate_oracle_mload_reads(statements, mem_proof) {
      return false;
    }
  }

  // 11b. MSTORE/MSTORE8 oracle ↔ MemoryConsistencyProof.write_log
  //
  // Verifies that every store instruction's (addr, value) pair ends up in
  // the memory write log, binding the prover's claimed write to the AIR.
  let has_mstore = statements
    .iter()
    .any(|s| s.opcode == opcode::MSTORE || s.opcode == opcode::MSTORE8);
  if has_mstore {
    let mem_proof = match &receipt.memory_proof {
      Some(p) => p,
      None => return false,
    };
    if !validate_oracle_mstore_writes(statements, mem_proof) {
      return false;
    }
  }

  // 11c. SLOAD oracle ↔ StorageConsistencyProof.read_log
  let has_sload = statements.iter().any(|s| s.opcode == opcode::SLOAD);
  if has_sload {
    let stor_proof = match &receipt.storage_proof {
      Some(p) => p,
      None => return false,
    };
    if !validate_oracle_sload_reads(statements, stor_proof) {
      return false;
    }
  }

  // 11d. SSTORE oracle ↔ StorageConsistencyProof.write_log
  let has_sstore = statements.iter().any(|s| s.opcode == opcode::SSTORE);
  if has_sstore {
    let stor_proof = match &receipt.storage_proof {
      Some(p) => p,
      None => return false,
    };
    if !validate_oracle_sstore_writes(statements, stor_proof) {
      return false;
    }
  }

  // 11e. KECCAK256 oracle ↔ KeccakConsistencyProof.log output hashes (Gap 7)
  //
  // The existing step 8 verifies keccak256(preimage) == output_hash inside
  // KeccakConsistencyProof; step 9 binds the preimage bytes to memory.
  // This step closes the remaining gap by verifying that the hash committed
  // in the oracle WFF (= stack output) equals what KeccakConsistencyProof
  // recorded.
  let has_keccak = statements.iter().any(|s| s.opcode == opcode::KECCAK256);
  if has_keccak {
    let kec_proof = match &receipt.keccak_proof {
      Some(p) => p,
      None => return false,
    };
    if !validate_oracle_keccak_outputs(statements, kec_proof) {
      return false;
    }
  }

  // 12. SubCall inner_proof recursive verification.
  //
  // For every CALL/CALLCODE/DELEGATECALL/STATICCALL/CREATE/CREATE2 instruction
  // that carries an `inner_proof`, recursively verify the callee's execution
  // trace.  This provides callee soundness and enforces the return_data
  // binding between the caller claim and the callee receipt.
  for stmt in statements {
    if let Some(sc) = &stmt.sub_call_claim
      && verify_sub_call_claim(sc, statements).is_err() {
        return false;
      }
  }

  true
}

// ============================================================
// Recursive proving types
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
#[allow(clippy::large_enum_variant)]
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

// ── SubCall recursive verification ──────────────────────────────────────────

/// Extract an [`InstructionTransitionStatement`] from an
/// [`InstructionTransitionProof`] step, mirroring the same logic used by the
/// batch-prove path in `execute.rs`.
fn statement_from_proof_step(step: &InstructionTransitionProof) -> InstructionTransitionStatement {
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
    mcopy_claim: step.mcopy_claim.clone(),
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
/// When `inner_proof.batch_receipt` is `Some`, the receipt is cryptographically
/// verified via `verify_batch_transaction_zk_receipt`. A `None` receipt is also
/// accepted (oracle mode), which is useful for tests and tooling that omit the
/// full STARK prover.
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

    // Build statements from the inner proof's steps.
    // Only include steps that have a semantic_proof — `prove_batch_transaction_zk_receipt`
    // filters the same way, so the manifest entry count matches.
    let inner_stmts: Vec<InstructionTransitionStatement> = inner
      .steps
      .iter()
      .filter(|s| s.semantic_proof.is_some())
      .map(statement_from_proof_step)
      .collect();

    // STARK receipt re-verification.
    // When the inner proof carries a pre-computed batch receipt, verify it
    // cryptographically.  `None` = oracle mode (structural-only), which is
    // still accepted — it lets tests and tooling omit the full STARK prover.
    if let Some(receipt) = &inner.batch_receipt
      && !verify_batch_transaction_zk_receipt(&inner_stmts, receipt) {
        return Err("SubCall inner STARK receipt verification failed".to_string());
      }

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
            "SubCall claim has non-empty return_data but inner proof has no RETURN".to_string(),
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

// ── Execution chain proving / verification ──────────────────────────────────

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
  let commitments: Vec<StateCommitment> = vm_state_seq.iter().map(commit_vm_state).collect();

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
    let mut new_nodes: Vec<ExecutionReceipt> = Vec::with_capacity(nodes.len().div_ceil(2));
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
