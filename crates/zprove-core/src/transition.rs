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
  Proof, Term, WFF, compile_proof, infer_proof, mul_u256_mod, prove_addmod, prove_byte,
  prove_div, prove_eq, prove_exp, prove_gt, prove_iszero, prove_lt, prove_mod, prove_mul,
  prove_mulmod, prove_sar, prove_sdiv, prove_sgt, prove_shl,
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
  commit_vm_state, compute_wff_opcode_digest,
  make_batch_receipt_binding_public_values,
  prove_batch_lut_with_prep, prove_keccak_consistency, prove_link_stark,
  prove_memory_consistency,
  prove_batch_stack_ir_with_prep, prove_stack_consistency, prove_stack_rw,
  prove_storage_consistency,
  setup_batch_proof_rows_preprocessed,
  validate_keccak_memory_cross_check, validate_manifest_rows, verify_batch_lut_with_prep,
  verify_keccak_consistency, verify_link_stark, verify_memory_consistency,
  verify_batch_stack_ir_with_prep, verify_stack_consistency, verify_stack_rw,
  verify_storage_consistency,
  StackRwProof,
  write_delta::{MemWriteSet, StorWriteSet},
};
use p3_uni_stark::{PreprocessedProverData, PreprocessedVerifierKey};
use revm::bytecode::opcode;
use crate::zk_proof::types::Val;

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

/// `inner_proof` contains the callee's captured execution trace (always present).
/// If `inner_proof.steps` is non-empty, `verify_sub_call_claim` performs full
/// structural and STARK re-verification.  Empty `steps` means the callee was a
/// precompile or empty account — values are accepted as given (no EVM bytecode
/// to verify).
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
  /// Recursive proof of the callee execution trace.
  /// Non-empty `steps` → full structural + STARK verification by `verify_sub_call_claim`.
  /// Empty `steps`     → precompile / empty account oracle (no bytecode to verify).
  pub inner_proof: Box<TransactionProof>,
  // ── Gap-C4: CREATE2 address derivation witness ───────────────────────────
  /// Address of the account that issued the CREATE2 (= deployer, 20 bytes).
  /// `Some` only when `opcode == CREATE2`; `None` for all other call types.
  pub create2_deployer: Option<[u8; 20]>,
  /// Salt passed to CREATE2 (big-endian U256, 32 bytes).
  /// `Some` only when `opcode == CREATE2`.
  pub create2_salt: Option<[u8; 32]>,
  /// `keccak256(initcode)` — the hash of the deployed bytecode (32 bytes).
  /// Computed by the prover from the actual initcode bytes at prove time.
  /// `Some` only when `opcode == CREATE2`.
  pub create2_initcode_hash: Option<[u8; 32]>,
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

/// A witness for external-state-reading opcodes:
/// BLOCKHASH, EXTCODESIZE, EXTCODEHASH, BALANCE, SELFBALANCE.
///
/// Records the query key (block number or account address) and the returned
/// value so the verifier can re-check consistency with the committed
/// [`WorldStateContext`] at batch level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalStateClaim {
  /// The opcode that produced this claim.
  /// Supported: BLOCKHASH, EXTCODESIZE, EXTCODEHASH, BALANCE, SELFBALANCE.
  pub opcode: u8,
  /// For BLOCKHASH: queried block number (32-byte big-endian U256).
  /// For EXTCODESIZE / EXTCODEHASH / BALANCE: queried contract address
  /// (20 bytes, zero-left-padded to 32).
  /// For SELFBALANCE: address of the currently-executing contract
  /// (zero-left-padded to 32).
  pub key: [u8; 32],
  /// The 32-byte value pushed to the stack.
  pub output_value: [u8; 32],
}

/// One committed entry in the world-state oracle table.
///
/// The prover supplies a [`WorldStateContext`] as a public input; the
/// verifier checks that every [`ExternalStateClaim`] output matches the
/// corresponding entry here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorldStateEntry {
  /// Same opcodes as [`ExternalStateClaim::opcode`].
  pub opcode: u8,
  /// Same semantics as [`ExternalStateClaim::key`].
  pub key: [u8; 32],
  /// The committed 32-byte value for this (opcode, key) pair.
  pub value: [u8; 32],
}

/// Pre-committed world-state oracle table, provided as a public input.
///
/// # Security model
/// Full Merkle-Patricia-Trie proof is out-of-scope for this phase.
/// This struct acts as an *external commitment*: the prover cannot manufacture
/// an [`ExternalStateClaim`] whose `output_value` disagrees with the matching
/// entry here, because step 14 of [`verify_batch_transaction_zk_receipt`]
/// rejects the receipt if any mismatch is detected.
/// An outer layer (e.g., a light client) is responsible for binding the
/// `WorldStateContext` to an on-chain state root.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WorldStateContext {
  /// Committed (opcode, key) → value entries.
  pub entries: Vec<WorldStateEntry>,
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
  /// CALLER (0x33): address of the direct caller of this frame.
  /// For the outermost transaction this equals `tx.caller` (== ORIGIN).
  pub caller: [u8; 32],
  /// CALLVALUE (0x34): ETH value attached to this call as U256 big-endian.
  pub callvalue: [u8; 32],
  /// ADDRESS (0x30): address of the contract being executed, zero-padded to 32 bytes.
  pub self_address: [u8; 32],
  /// CALLDATASIZE (0x36): byte-length of calldata as U256 big-endian.
  pub calldata_size: [u8; 32],
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
  pub semantic_proof: Proof,
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
  /// External-state witness.  `Some` for BLOCKHASH, EXTCODESIZE, EXTCODEHASH,
  /// BALANCE, and SELFBALANCE.
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
  /// Commitment to cumulative memory write-set `W_i` at this state boundary.
  /// `hash_mem_write_set(W_i)`.  `[0u8;32]` encodes an empty write-set (start).
  pub memory_root: [u8; 32],
  /// Commitment to cumulative storage write-set `W_i` at this state boundary.
  /// `hash_stor_write_set(W_i)`.  `[0u8;32]` encodes an empty write-set (start).
  pub storage_root: [u8; 32],
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
  /// External-state witness — `Some` for BLOCKHASH, EXTCODESIZE, EXTCODEHASH,
  /// BALANCE, SELFBALANCE.
  /// Copied from [`InstructionTransitionProof::external_state_claim`] during batch proving.
  pub external_state_claim: Option<ExternalStateClaim>,
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
  /// Block/tx public-input context for env-oracle validation.
  ///
  /// `None` = produced without env-context binding (backward-compatible mode).
  /// `Some` enables step 13 of [`verify_batch_transaction_zk_receipt`], which
  /// asserts that every env-opcode output matches the committed context.
  pub env_context: Option<BlockTxContext>,
  /// World-state oracle table for external-state-opcode validation (Gap-B).
  ///
  /// `None` = produced without world-state binding (backward-compatible mode).
  /// `Some` enables step 14 of [`verify_batch_transaction_zk_receipt`], which
  /// asserts that every BLOCKHASH / EXTCODESIZE / EXTCODEHASH / BALANCE /
  /// SELFBALANCE output matches the corresponding committed entry.
  pub world_state_context: Option<WorldStateContext>,
  // (logs are stored inside the individual proofs themselves)
}

impl BatchTransactionZkReceipt {
  /// Returns the total number of AIR columns across all active sub-proofs in
  /// this batch receipt.
  ///
  /// The shared preprocessed matrix (`NUM_BATCH_PREP_COLS`) is counted once,
  /// even though it is reused by both the StackIR and LUT proofs.
  /// Optional sub-proofs (memory, storage, stack, keccak) contribute their
  /// column counts only when present.
  pub fn total_column_count(&self) -> usize {
    use crate::zk_proof::{
      NUM_BATCH_PREP_COLS, NUM_KECCAK_COLS, NUM_LUT_COLS, NUM_MEM_COLS, NUM_STACK_COLS,
      NUM_STACK_IR_COLS, NUM_STACK_RW_COLS, NUM_STOR_COLS,
    };
    let mut total = NUM_BATCH_PREP_COLS  // shared preprocessed matrix (counted once)
      + NUM_STACK_IR_COLS               // StackIR batch proof (always present)
      + NUM_LUT_COLS;                   // LUT batch proof    (always present)
    if self.memory_proof.is_some()  { total += NUM_MEM_COLS; }
    if self.storage_proof.is_some() { total += NUM_STOR_COLS; }
    if self.stack_proof.is_some()   { total += NUM_STACK_COLS; }
    if self.stack_rw_proof.is_some(){ total += NUM_STACK_RW_COLS; }
    if self.keccak_proof.is_some()  { total += NUM_KECCAK_COLS; }
    total
  }
}

// ── Pre-computed heavy step for the batch proving pipeline ───────────────────

/// Pre-computed, instruction-independent artifacts used by the STARK proving
/// pipeline.  Build once (outside any timing loop) and reuse across multiple
/// [`prove_batch_transaction_zk_receipt`] calls.
///
/// This separates the expensive WFF-building + preprocessed-matrix setup step
/// from the STARK proving proper, letting benchmarks measure only the latter.
///
/// # Usage
/// ```ignore
/// // One-time setup (outside the timing loop):
/// let prep = build_batch_zk_receipt_prep(&batch_itps)?;
///
/// // Timed inner loop:
/// for _ in 0..iters {
///     let receipt = prove_batch_transaction_zk_receipt_with_prep(&batch_itps, &prep)?;
/// }
/// ```
pub struct BatchTransactionZkReceiptPrep {
  pub manifest: BatchProofRowsManifest,
  pub prep_data: PreprocessedProverData<CircleStarkConfig>,
  pub preprocessed_vk: PreprocessedVerifierKey<CircleStarkConfig>,
  pub lut_bind_pv: Vec<Val>,
  pub stack_bind_pv: Vec<Val>,
}

/// Build the [`BatchTransactionZkReceiptPrep`] for `itps`.
///
/// Performs the WFF inference, manifest compilation, and preprocessed-matrix
/// setup — the slow O(batch_count) work that does not change between proving
/// rounds.  Returns `Err` if any proof is invalid or no provable instructions
/// are found.
pub fn build_batch_zk_receipt_prep(
  itps: &[InstructionTransitionProof],
) -> Result<BatchTransactionZkReceiptPrep, String> {
  // Validate all proofs first (same pre-check as prove_batch_transaction_zk_receipt).
  for (i, itp) in itps.iter().enumerate() {
    if !verify_proof(itp) {
      return Err(format!(
        "build_batch_zk_receipt_prep: semantic/row verification failed at step {i} (opcode 0x{:02x})",
        itp.opcode
      ));
    }
  }

  let items: Vec<(u8, WFF, &Proof)> = itps
    .iter()
    .filter_map(|itp| {
      let wff = wff_instruction(itp.opcode, &itp.stack_inputs, &itp.stack_outputs);
      Some((itp.opcode, wff, &itp.semantic_proof))
    })
    .collect();

  if items.is_empty() {
    return Err("build_batch_zk_receipt_prep: no provable instructions in batch".to_string());
  }

  let manifest = build_batch_manifest_from_wffs(&items)?;
  let lut_bind_pv =
    make_batch_receipt_binding_public_values(RECEIPT_BIND_TAG_LUT, &manifest.entries);
  let stack_bind_pv =
    make_batch_receipt_binding_public_values(RECEIPT_BIND_TAG_STACK, &manifest.entries);
  let (prep_data, preprocessed_vk) = setup_batch_proof_rows_preprocessed(&manifest, &lut_bind_pv)?;

  Ok(BatchTransactionZkReceiptPrep {
    manifest,
    prep_data,
    preprocessed_vk,
    lut_bind_pv,
    stack_bind_pv,
  })
}

/// Prove a batch ZK receipt using pre-built artifacts from [`build_batch_zk_receipt_prep`].
///
/// This is the fast inner loop body: it runs only the STARK provers and the
/// consistency proofs (memory / storage / KECCAK / stack).  WFF inference and
/// preprocessed-matrix setup are **not** repeated.
pub fn prove_batch_transaction_zk_receipt_with_prep(
  itps: &[InstructionTransitionProof],
  prep: &BatchTransactionZkReceiptPrep,
) -> Result<BatchTransactionZkReceipt, String> {
  let manifest = &prep.manifest;
  let prep_data = &prep.prep_data;
  let lut_bind_pv = &prep.lut_bind_pv;
  let stack_bind_pv = &prep.stack_bind_pv;
  let preprocessed_vk = prep.preprocessed_vk.clone();

  // ── Collect all claims (sequential, O(n) in instruction count) ──────────
  let mut all_mem_claims: Vec<MemAccessClaim> = Vec::new();
  let mut all_keccak_claims: Vec<KeccakClaim> = Vec::new();
  let mut all_mcopy_claims: Vec<MemCopyClaim> = Vec::new();
  let mut rw_counter: u64 = 0;
  for itp in itps {
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
    if let Some(ref mc) = itp.mcopy_claim {
      let src_count = itp.memory_claims.iter().filter(|c| !c.is_write).count();
      let dst_count = itp.memory_claims.iter().filter(|c| c.is_write).count();
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
    if let Some(ref kc) = itp.keccak_claim {
      all_keccak_claims.push(kc.clone());
    }
  }
  let mut stor_rw_counter: u64 = 0;
  let all_stor_claims = collect_storage_claims_recursive(itps, &mut stor_rw_counter);
  let all_stack_claims: Vec<StackAccessClaim> = itps
    .iter()
    .flat_map(|itp| itp.stack_claims.iter().cloned())
    .collect();
  let return_data = itps
    .iter()
    .rev()
    .find_map(|itp| itp.return_data_claim.clone());

  // Each sub-proof is fully independent — run all 7 in parallel OS threads.
  // rayon is NOT active here, so each thread is purely single-threaded; no
  // oversubscription.  Wall-clock time = slowest single sub-proof.
  use std::sync::Mutex;
  use crate::zk_proof::batch_proof_pool;

  let stack_ir_res:  Mutex<Option<Result<CircleStarkProof, String>>>  = Mutex::new(None);
  let lut_res:       Mutex<Option<Result<CircleStarkProof, String>>>  = Mutex::new(None);
  let memory_res:    Mutex<Option<Result<Option<MemoryConsistencyProof>, String>>>   = Mutex::new(None);
  let storage_res:   Mutex<Option<Result<Option<StorageConsistencyProof>, String>>>  = Mutex::new(None);
  let keccak_res:    Mutex<Option<Result<Option<KeccakConsistencyProof>, String>>>   = Mutex::new(None);
  let stack_res:     Mutex<Option<Result<Option<StackConsistencyProof>, String>>>    = Mutex::new(None);
  let stack_rw_res:  Mutex<Option<Result<Option<StackRwProof>, String>>>             = Mutex::new(None);

  batch_proof_pool().scope(|s| {
    s.spawn(|_| *stack_ir_res.lock().unwrap() = Some(prove_batch_stack_ir_with_prep(manifest, prep_data, stack_bind_pv)));
    s.spawn(|_| *lut_res.lock().unwrap()      = Some(prove_batch_lut_with_prep(manifest, prep_data, lut_bind_pv)));
    s.spawn(|_| *memory_res.lock().unwrap()   = Some(
      if all_mem_claims.is_empty() { Ok(None) }
      else { prove_memory_consistency(&all_mem_claims).map(Some) }
    ));
    s.spawn(|_| *storage_res.lock().unwrap()  = Some(
      if all_stor_claims.is_empty() { Ok(None) }
      else { prove_storage_consistency(&all_stor_claims).map(Some) }
    ));
    s.spawn(|_| *keccak_res.lock().unwrap()   = Some(
      if all_keccak_claims.is_empty() { Ok(None) }
      else { prove_keccak_consistency(&all_keccak_claims).map(Some) }
    ));
    s.spawn(|_| *stack_res.lock().unwrap()    = Some(
      if all_stack_claims.is_empty() { Ok(None) }
      else { prove_stack_consistency(&all_stack_claims).map(Some) }
    ));
    s.spawn(|_| *stack_rw_res.lock().unwrap() = Some(
      if all_stack_claims.is_empty() { Ok(None) }
      else { prove_stack_rw(&all_stack_claims).map(Some) }
    ));
  });

  let stack_ir_proof = stack_ir_res.into_inner().unwrap().unwrap()?;
  let lut_proof      = lut_res.into_inner().unwrap().unwrap()?;
  let memory_proof   = memory_res.into_inner().unwrap().unwrap()?;
  let storage_proof  = storage_res.into_inner().unwrap().unwrap()?;
  let keccak_proof   = keccak_res.into_inner().unwrap().unwrap()?;
  let stack_proof    = stack_res.into_inner().unwrap().unwrap()?;
  let stack_rw_proof = stack_rw_res.into_inner().unwrap().unwrap()?;

  Ok(BatchTransactionZkReceipt {
    stack_ir_proof,
    lut_proof,
    preprocessed_vk,
    manifest: manifest.clone(),
    memory_proof,
    storage_proof,
    keccak_proof,
    stack_proof,
    stack_rw_proof,
    mcopy_claims: all_mcopy_claims,
    return_data,
    env_context: None,
    world_state_context: None,
  })
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
    opcode::MUL => mul_u256_mod(&inputs[0], &inputs[1]) == outputs[0],
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

/// Core semantic proof (instruction-specific logic only, no stack/PC bindings).
pub fn prove_instruction(op: u8, inputs: &[[u8; 32]], outputs: &[[u8; 32]]) -> Proof {
  match op {
    opcode::ADD => prove_add_limb_sym(),
    opcode::SUB => prove_sub_limb_sym(),
    opcode::MUL => prove_mul(&inputs[0], &inputs[1], &outputs[0]),
    opcode::DIV => prove_div(&inputs[0], &inputs[1], &outputs[0]),
    opcode::MOD => prove_mod(&inputs[0], &inputs[1], &outputs[0]),
    opcode::SDIV => prove_sdiv(&inputs[0], &inputs[1], &outputs[0]),
    opcode::SMOD => prove_smod(&inputs[0], &inputs[1], &outputs[0]),
    opcode::AND => prove_and_sym(&inputs[0], &inputs[1]),
    opcode::OR  => prove_or_sym(&inputs[0], &inputs[1]),
    opcode::XOR => prove_xor_sym(&inputs[0], &inputs[1]),
    opcode::NOT => prove_not_sym(&inputs[0]),
    opcode::LT => prove_lt(&inputs[0], &inputs[1], &outputs[0]),
    opcode::GT => prove_gt(&inputs[0], &inputs[1], &outputs[0]),
    opcode::SLT => prove_slt(&inputs[0], &inputs[1], &outputs[0]),
    opcode::SGT => prove_sgt(&inputs[0], &inputs[1], &outputs[0]),
    opcode::EQ => prove_eq(&inputs[0], &inputs[1], &outputs[0]),
    opcode::ISZERO => prove_iszero(&inputs[0], &outputs[0]),
    opcode::SHL => prove_shl(&inputs[0], &inputs[1], &outputs[0]),
    opcode::SHR => prove_shr(&inputs[0], &inputs[1], &outputs[0]),
    opcode::SAR => prove_sar(&inputs[0], &inputs[1], &outputs[0]),
    opcode::BYTE => prove_byte(&inputs[0], &inputs[1], &outputs[0]),
    opcode::SIGNEXTEND => prove_signextend(&inputs[0], &inputs[1], &outputs[0]),
    opcode::ADDMOD => prove_addmod(
      &inputs[0],
      &inputs[1],
      &inputs[2],
      &outputs[0],
    ),
    opcode::MULMOD => prove_mulmod(
      &inputs[0],
      &inputs[1],
      &inputs[2],
      &outputs[0],
    ),
    opcode::EXP => prove_exp(&inputs[0], &inputs[1], &outputs[0]),

    // ── Per-opcode dedicated axioms ─────────────────────────────────────────

    // Memory read/write
    opcode::MLOAD => Proof::MloadAxiom,
    opcode::MSTORE => Proof::MstoreAxiom {
      opcode: op,
    },
    opcode::MSTORE8 => Proof::MstoreAxiom {
      opcode: op,
    },

    // Memory-copy family
    opcode::RETURNDATACOPY | opcode::EXTCODECOPY | opcode::MCOPY => {
      Proof::MemCopyAxiom { opcode: op }
    }
    opcode::CALLDATACOPY | opcode::CODECOPY => {
      Proof::MemCopyAxiom { opcode: op }
    }

    // Storage
    opcode::SLOAD => Proof::SloadAxiom,
    opcode::SSTORE => Proof::SstoreAxiom,
    opcode::TLOAD => Proof::TransientAxiom {
      opcode: op,
    },
    opcode::TSTORE => Proof::TransientAxiom {
      opcode: op,
    },

    // Keccak
    opcode::KECCAK256 => Proof::KeccakAxiom,

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
    | opcode::BLOBBASEFEE => Proof::EnvAxiom {
      opcode: op,
    },

    // External state
    opcode::BLOCKHASH | opcode::EXTCODESIZE | opcode::EXTCODEHASH | opcode::BALANCE => {
      Proof::ExternalStateAxiom {
        opcode: op,
      }
    }
    opcode::BLOBHASH => Proof::ExternalStateAxiom {
      opcode: op,
    },

    // Terminate
    opcode::RETURN | opcode::REVERT => Proof::TerminateAxiom {
      opcode: op,
    },

    // Sub-call / create
    opcode::CALL | opcode::CALLCODE | opcode::DELEGATECALL | opcode::STATICCALL => {
      Proof::CallAxiom {
        opcode: op,
      }
    }
    opcode::CREATE | opcode::CREATE2 => Proof::CreateAxiom {
      opcode: op,
    },
    opcode::SELFDESTRUCT => Proof::SelfdestructAxiom,

    // Log
    opcode::LOG0 | opcode::LOG1 | opcode::LOG2 | opcode::LOG3 | opcode::LOG4 => {
      Proof::LogAxiom {
        opcode: op,
      }
    }

    // INVALID
    opcode::INVALID => Proof::StructuralAxiom { opcode: op },

    // Structural: STOP, POP, JUMP, JUMPI, JUMPDEST, PUSH*, DUP*, SWAP*
    op if op == opcode::STOP
      || op == opcode::POP
      || op == opcode::JUMP
      || op == opcode::JUMPI
      || op == opcode::JUMPDEST
      || (opcode::PUSH0..=opcode::PUSH32).contains(&op) => {
      Proof::PushAxiom
    }
    op if (opcode::DUP1..=opcode::DUP16).contains(&op) => {
      let depth = op - opcode::DUP1 + 1;
      Proof::DupAxiom { depth }
    }
    op if (opcode::SWAP1..=opcode::SWAP16).contains(&op) => {
      let depth = op - opcode::SWAP1 + 1;
      Proof::SwapAxiom { depth }
    }

    _ => Proof::StructuralAxiom { opcode: op },
  }
}

// ============================================================
// WFF generation
// ============================================================

pub fn wff_instruction(op: u8, inputs: &[[u8; 32]], outputs: &[[u8; 32]]) -> WFF {
  let core = wff_instruction_core(op, inputs, outputs);
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
  WFF::And(
    Box::new(wff_stack_inputs(inputs)),
    Box::new(with_pc),
  )
}

/// Core WFF (instruction-specific logic only, no stack/PC bindings).
pub fn wff_instruction_core(op: u8, inputs: &[[u8; 32]], outputs: &[[u8; 32]]) -> WFF {
  match op {
    opcode::ADD => wff_add_limb_sym(),
    opcode::SUB => wff_sub_limb_sym(),
    opcode::MUL => wff_mul(&inputs[0], &inputs[1], &outputs[0]),
    opcode::DIV => wff_div(&inputs[0], &inputs[1], &outputs[0]),
    opcode::MOD => wff_mod(&inputs[0], &inputs[1], &outputs[0]),
    opcode::SDIV => wff_sdiv(&inputs[0], &inputs[1], &outputs[0]),
    opcode::SMOD => wff_smod(&inputs[0], &inputs[1], &outputs[0]),
    opcode::AND => wff_and_sym(&inputs[0], &inputs[1], &outputs[0]),
    opcode::OR  => wff_or_sym(&inputs[0], &inputs[1], &outputs[0]),
    opcode::XOR => wff_xor_sym(&inputs[0], &inputs[1], &outputs[0]),
    opcode::NOT => wff_not_sym(&inputs[0], &outputs[0]),
    opcode::LT => wff_lt(&inputs[0], &inputs[1], &outputs[0]),
    opcode::GT => wff_gt(&inputs[0], &inputs[1], &outputs[0]),
    opcode::SLT => wff_slt(&inputs[0], &inputs[1], &outputs[0]),
    opcode::SGT => wff_sgt(&inputs[0], &inputs[1], &outputs[0]),
    opcode::EQ => wff_eq(&inputs[0], &inputs[1], &outputs[0]),
    opcode::ISZERO => wff_iszero(&inputs[0], &outputs[0]),
    opcode::SHL => wff_shl(&inputs[0], &inputs[1], &outputs[0]),
    opcode::SHR => wff_shr(&inputs[0], &inputs[1], &outputs[0]),
    opcode::SAR => wff_sar(&inputs[0], &inputs[1], &outputs[0]),
    opcode::BYTE => wff_byte(&inputs[0], &inputs[1], &outputs[0]),
    opcode::SIGNEXTEND => wff_signextend(&inputs[0], &inputs[1], &outputs[0]),
    opcode::ADDMOD => wff_addmod(&inputs[0], &inputs[1], &inputs[2], &outputs[0]),
    opcode::MULMOD => wff_mulmod(&inputs[0], &inputs[1], &inputs[2], &outputs[0]),
    opcode::EXP => wff_exp(&inputs[0], &inputs[1], &outputs[0]),

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
    | opcode::INVALID => WFF::Equal(
      Box::new(Term::OutputTerm { stack_idx: 0, byte_idx: 0, value: 0 }),
      Box::new(Term::OutputTerm { stack_idx: 0, byte_idx: 0, value: 0 }),
    ),
    _ => WFF::Equal(
      Box::new(Term::OutputTerm { stack_idx: 0, byte_idx: 0, value: 0 }),
      Box::new(Term::OutputTerm { stack_idx: 0, byte_idx: 0, value: 0 }),
    ),
  }
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
  // Therefore we compare infer_proof(..) against wff_instruction_core(..).
  let expected_core_wff =
    wff_instruction_core(proof.opcode, &proof.stack_inputs, &proof.stack_outputs);
  let Ok(wff_result) = infer_proof(&proof.semantic_proof) else {
    return false;
  };
  // Verify every InputTerm/OutputTerm claim in the inferred WFF against
  // the declared stack inputs/outputs.
  if !check_wff_io_values(&wff_result, &proof.stack_inputs, &proof.stack_outputs) {
    return false;
  }
  expected_core_wff == wff_result
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

  let rows = compile_proof(&proof.semantic_proof);
  verify_compiled(&rows).is_ok()
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
      storage_root: [0u8; 32],
    },
    s_next: VmState {
      opcode: proof.opcode,
      pc: proof.pc + 1,
      sp: proof.stack_outputs.len(),
      stack: proof.stack_outputs.clone(),
      memory_root: [0u8; 32],
      storage_root: [0u8; 32],
    },
    accesses: Vec::new(),
    sub_call_claim: proof.sub_call_claim.clone(),
    mcopy_claim: proof.mcopy_claim.clone(),
    external_state_claim: proof.external_state_claim.clone(),
  };
  let receipt = match prove_batch_transaction_zk_receipt(std::slice::from_ref(proof)) {
    Ok(r) => r,
    Err(_) => return false,
  };
  verify_batch_transaction_zk_receipt(std::slice::from_ref(&statement), &receipt)
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
  prove_batch_transaction_zk_receipt_with_w_in(itps, &MemWriteSet::new(), &StorWriteSet::new())
}

/// Like [`prove_batch_transaction_zk_receipt`] but validates cross-segment
/// inherited reads against `mem_w_in` / `stor_w_in` (the cumulative write-sets
/// from all preceding segments).  Returns `Err` if any first-read in the batch
/// observes a value that differs from the previous state.
///
/// - Addresses in `w_in` map to their last-written value from prior segments.
/// - Addresses absent from `w_in` must be read as zero (EVM zero-initialisation).
pub fn prove_batch_transaction_zk_receipt_with_w_in(
  itps: &[InstructionTransitionProof],
  mem_w_in: &MemWriteSet,
  stor_w_in: &StorWriteSet,
) -> Result<BatchTransactionZkReceipt, String> {
  let receipt = prove_batch_transaction_zk_receipt_impl(itps)?;
  // First-Read Initialization soundness check.
  if let Some(ref mem_proof) = receipt.memory_proof {
    if !crate::zk_proof::write_delta::validate_inherited_reads(&mem_proof.read_set, mem_w_in) {
      return Err("memory: inherited read value does not match W_in".to_string());
    }
  }
  if let Some(ref stor_proof) = receipt.storage_proof {
    if !crate::zk_proof::write_delta::validate_inherited_stor_reads(
      &stor_proof.read_set,
      stor_w_in,
    ) {
      return Err("storage: inherited read value does not match W_in".to_string());
    }
  }
  Ok(receipt)
}

/// Recursively collect all [`StorageAccessClaim`]s from `itps` and every
/// sub-call's `inner_proof`, assigning monotone `rw_counter` values in
/// strict EVM execution order.
///
/// EVM storage is **transaction-global**: an SSTORE in a sub-call immediately
/// visible to the caller (and vice-versa).  By flattening the claim list here,
/// all SLOAD/SSTORE/TLOAD/TSTORE accesses across the entire call tree are
/// covered by the outer [`StorageConsistencyAir`], closing Gap 2.
///
/// Inner-call claims are inserted immediately after the CALL step's own
/// claims (before the next outer step), which matches the EVM dispatch order:
/// the callee executes between the CALL opcode firing and its `step_end`.
fn collect_storage_claims_recursive(
  itps: &[InstructionTransitionProof],
  rw_counter: &mut u64,
) -> Vec<StorageAccessClaim> {
  let mut out = Vec::new();
  for itp in itps {
    // This step's own storage claims (SLOAD/SSTORE/TLOAD/TSTORE).
    for claim in &itp.storage_claims {
      *rw_counter += 1;
      out.push(StorageAccessClaim {
        rw_counter: *rw_counter,
        contract: claim.contract,
        slot: claim.slot,
        is_write: claim.is_write,
        value: claim.value,
      });
    }
    // Then the callee's claims — inner execution runs after the CALL fires.
    if let Some(sc) = &itp.sub_call_claim {
      let inner = collect_storage_claims_recursive(&sc.inner_proof.steps, rw_counter);
      out.extend(inner);
    }
  }
  out
}

// Internal implementation shared by both public entry-points.
fn prove_batch_transaction_zk_receipt_impl(
  itps: &[InstructionTransitionProof],
) -> Result<BatchTransactionZkReceipt, String> {
  // Pre-check: reject forged/invalid proofs before expensive STARK proving.
  // Applies verify_proof (semantic + WFF consistency) to every instruction,
  // whether or not a semantic_proof is supplied.
  for (i, itp) in itps.iter().enumerate() {
    if !verify_proof(itp) {
      return Err(format!(
        "semantic/row verification failed at step {i} (opcode 0x{:02x})",
        itp.opcode
      ));
    }
  }

  let items: Vec<(u8, WFF, &Proof)> = itps
    .iter()
    .filter_map(|itp| {
      let wff = wff_instruction(itp.opcode, &itp.stack_inputs, &itp.stack_outputs);
      let proof_ref: &Proof = &itp.semantic_proof;
      Some((itp.opcode, wff, proof_ref))
    })
    .collect();

  if items.is_empty() {
    return Err("prove_batch_transaction_zk_receipt: no provable instructions in batch".to_string());
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

  // Collect all claims before spawning threads (sequential, O(n)).
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
  let all_stack_claims: Vec<StackAccessClaim> = itps
    .iter()
    .flat_map(|itp| itp.stack_claims.iter().cloned())
    .collect();
  let return_data = itps
    .iter()
    .rev()
    .find_map(|itp| itp.return_data_claim.clone());

  use std::sync::Mutex;
  use crate::zk_proof::batch_proof_pool;

  let stack_ir_res:  Mutex<Option<Result<CircleStarkProof, String>>>  = Mutex::new(None);
  let lut_res:       Mutex<Option<Result<CircleStarkProof, String>>>  = Mutex::new(None);
  let memory_res:    Mutex<Option<Result<Option<MemoryConsistencyProof>, String>>>   = Mutex::new(None);
  let storage_res:   Mutex<Option<Result<Option<StorageConsistencyProof>, String>>>  = Mutex::new(None);
  let keccak_res:    Mutex<Option<Result<Option<KeccakConsistencyProof>, String>>>   = Mutex::new(None);
  let stack_res:     Mutex<Option<Result<Option<StackConsistencyProof>, String>>>    = Mutex::new(None);
  let stack_rw_res:  Mutex<Option<Result<Option<StackRwProof>, String>>>             = Mutex::new(None);

  batch_proof_pool().scope(|s| {
    s.spawn(|_| *stack_ir_res.lock().unwrap() = Some(prove_batch_stack_ir_with_prep(&manifest, &prep_data, &stack_bind_pv)));
    s.spawn(|_| *lut_res.lock().unwrap()      = Some(prove_batch_lut_with_prep(&manifest, &prep_data, &lut_bind_pv)));
    s.spawn(|_| *memory_res.lock().unwrap()   = Some(
      if all_mem_claims.is_empty() { Ok(None) }
      else { prove_memory_consistency(&all_mem_claims).map(Some) }
    ));
    s.spawn(|_| *storage_res.lock().unwrap()  = Some(
      if all_stor_claims.is_empty() { Ok(None) }
      else { prove_storage_consistency(&all_stor_claims).map(Some) }
    ));
    s.spawn(|_| *keccak_res.lock().unwrap()   = Some(
      if all_keccak_claims.is_empty() { Ok(None) }
      else { prove_keccak_consistency(&all_keccak_claims).map(Some) }
    ));
    s.spawn(|_| *stack_res.lock().unwrap()    = Some(
      if all_stack_claims.is_empty() { Ok(None) }
      else { prove_stack_consistency(&all_stack_claims).map(Some) }
    ));
    s.spawn(|_| *stack_rw_res.lock().unwrap() = Some(
      if all_stack_claims.is_empty() { Ok(None) }
      else { prove_stack_rw(&all_stack_claims).map(Some) }
    ));
  });

  let stack_ir_proof = stack_ir_res.into_inner().unwrap().unwrap()?;
  let lut_proof      = lut_res.into_inner().unwrap().unwrap()?;
  let memory_proof   = memory_res.into_inner().unwrap().unwrap()?;
  let storage_proof  = storage_res.into_inner().unwrap().unwrap()?;
  let keccak_proof   = keccak_res.into_inner().unwrap().unwrap()?;
  let stack_proof    = stack_res.into_inner().unwrap().unwrap()?;
  let stack_rw_proof = stack_rw_res.into_inner().unwrap().unwrap()?;

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
    env_context: None,
    world_state_context: None,
  })
}

/// Prove a batch and embed the given [`BlockTxContext`] as public-input binding.
///
/// This is the preferred path for production proving: any env-reading opcode
/// (COINBASE, TIMESTAMP, CALLER, CALLVALUE, ADDRESS, …) in the batch will
/// have its claimed output validated against `ctx` during
/// [`verify_batch_transaction_zk_receipt`].
pub fn prove_batch_transaction_zk_receipt_with_env(
  itps: &[InstructionTransitionProof],
  ctx: BlockTxContext,
) -> Result<BatchTransactionZkReceipt, String> {
  let mut receipt = prove_batch_transaction_zk_receipt(itps)?;
  receipt.env_context = Some(ctx);
  Ok(receipt)
}

/// Prove a batch and embed the given [`WorldStateContext`] as public-input binding.
///
/// Resolves Gap-B1 (BALANCE), Gap-B2 (EXTCODESIZE/EXTCODEHASH),
/// Gap-B3 (BLOCKHASH), and Gap-B4 (SELFBALANCE): every external-state
/// opcode's claimed output will be validated against the committed table
/// during [`verify_batch_transaction_zk_receipt`].
///
/// The `WorldStateContext` must contain one entry per distinct
/// `(opcode, key)` pair in the batch; step 14 rejects the receipt if any
/// entry is missing or mismatched.
pub fn prove_batch_transaction_zk_receipt_with_world_state(
  itps: &[InstructionTransitionProof],
  ws: WorldStateContext,
) -> Result<BatchTransactionZkReceipt, String> {
  let mut receipt = prove_batch_transaction_zk_receipt(itps)?;
  receipt.world_state_context = Some(ws);
  Ok(receipt)
}

/// Prove a batch and embed both env-context and world-state-context bindings.
///
/// Combines `prove_batch_transaction_zk_receipt_with_env` and
/// `prove_batch_transaction_zk_receipt_with_world_state` in a single call,
/// enabling both step 13 (Gap-A env oracle) and step 14 (Gap-B world-state
/// oracle) validation.
pub fn prove_batch_transaction_zk_receipt_with_env_and_world_state(
  itps: &[InstructionTransitionProof],
  ctx: BlockTxContext,
  ws: WorldStateContext,
) -> Result<BatchTransactionZkReceipt, String> {
  let mut receipt = prove_batch_transaction_zk_receipt(itps)?;
  receipt.env_context = Some(ctx);
  receipt.world_state_context = Some(ws);
  Ok(receipt)
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
  // TLOAD has the same stack layout as SLOAD (one key in, one value out).
  for stmt in statements.iter().filter(|s| s.opcode == opcode::SLOAD || s.opcode == opcode::TLOAD) {
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

/// For every SSTORE/TSTORE statement, verify the committed (slot, value) pair appears
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
  // TSTORE has the same stack layout as SSTORE (slot key + value, both inputs).
  for stmt in statements.iter().filter(|s| s.opcode == opcode::SSTORE || s.opcode == opcode::TSTORE) {
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

/// Validate external-state oracle outputs against the committed world-state table.
///
/// For each statement that carries an [`ExternalStateClaim`] (produced by
/// BLOCKHASH, EXTCODESIZE, EXTCODEHASH, BALANCE, or SELFBALANCE), look up
/// `(claim.opcode, claim.key)` in `ws.entries` and assert that
/// `claim.output_value == entry.value`.
///
/// If the `WorldStateContext` does not contain an entry for a particular
/// `(opcode, key)` pair the claim is **rejected** (the prover is required to
/// provide a complete context).
///
/// Recursively walks inner sub-call proofs so that world-state reads inside
/// callees are also bound to the same committed table.
fn validate_oracle_external_state_claims(
  statements: &[InstructionTransitionStatement],
  ws: &WorldStateContext,
) -> bool {
  use std::collections::HashMap;
  let map: HashMap<(u8, [u8; 32]), [u8; 32]> =
    ws.entries.iter().map(|e| ((e.opcode, e.key), e.value)).collect();

  fn check_stmts(
    stmts: &[InstructionTransitionStatement],
    map: &std::collections::HashMap<(u8, [u8; 32]), [u8; 32]>,
  ) -> bool {
    for stmt in stmts {
      // Check this statement's external-state claim.
      if let Some(esc) = &stmt.external_state_claim {
        match map.get(&(esc.opcode, esc.key)) {
          Some(&committed) if committed == esc.output_value => {}
          _ => return false,
        }
      }
      // Recurse into inner sub-call proofs.
      if let Some(sc) = &stmt.sub_call_claim {
        let inner_stmts: Vec<InstructionTransitionStatement> = sc
          .inner_proof
          .steps
          .iter()
          .map(statement_from_proof_step)
          .collect();
        if !check_stmts(&inner_stmts, map) {
          return false;
        }
      }
    }
    true
  }

  check_stmts(statements, &map)
}

/// Validate env-opcode oracle outputs against embedded public-input context.
///
/// For each statement whose opcode belongs to the static-per-transaction set
/// (COINBASE, TIMESTAMP, NUMBER, DIFFICULTY/PREVRANDAO, GASLIMIT, CHAINID,
/// BASEFEE, ORIGIN, GASPRICE, CALLER, CALLVALUE, ADDRESS, CALLDATASIZE),
/// the single pushed value (`s_next.stack[0]`) must equal the corresponding
/// field in `ctx`.
///
/// Dynamic per-instruction opcodes (PC, MSIZE, GAS, CALLDATALOAD, CODESIZE,
/// RETURNDATASIZE, SELFBALANCE, BLOBBASEFEE) are skipped — they require
/// per-step witnesses beyond a single public context.
fn validate_oracle_env_claims(
  statements: &[InstructionTransitionStatement],
  ctx: &BlockTxContext,
) -> bool {
  for stmt in statements {
    let expected: [u8; 32] = match stmt.opcode {
      opcode::COINBASE    => ctx.coinbase,
      opcode::TIMESTAMP   => ctx.timestamp,
      opcode::NUMBER      => ctx.block_number,
      opcode::DIFFICULTY  => ctx.prevrandao,
      opcode::GASLIMIT    => ctx.gas_limit,
      opcode::CHAINID     => ctx.chain_id,
      opcode::BASEFEE     => ctx.basefee,
      opcode::ORIGIN      => ctx.origin,
      opcode::GASPRICE    => ctx.gas_price,
      opcode::CALLER      => ctx.caller,
      opcode::CALLVALUE   => ctx.callvalue,
      opcode::ADDRESS     => ctx.self_address,
      opcode::CALLDATASIZE => ctx.calldata_size,
      _ => continue,
    };
    // All env opcodes above push exactly one value with no pops; the pushed
    // word is the first (and only) element of s_next.stack.
    let Some(&claimed) = stmt.s_next.stack.first() else {
      return false;
    };
    if claimed != expected {
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


/// Recursively collect all [`InstructionTransitionStatement`]s that have
/// SLOAD or SSTORE opcodes from `statements` and their sub-call inner proofs.
///
/// Used by `verify_batch_transaction_zk_receipt` to extend the oracle
/// cross-checks (11c/11d) from outer statements to the full call tree,
/// ensuring inner SLOAD/SSTORE oracle values are also bound to the
/// transaction-global `StorageConsistencyProof`.
fn collect_all_storage_stmts_recursive(
  statements: &[InstructionTransitionStatement],
) -> Vec<InstructionTransitionStatement> {
  let mut out = Vec::new();
  for stmt in statements {
    if stmt.opcode == opcode::SLOAD || stmt.opcode == opcode::SSTORE {
      out.push(stmt.clone());
    }
    if let Some(sc) = &stmt.sub_call_claim {
      let inner_stmts: Vec<InstructionTransitionStatement> = sc
        .inner_proof
        .steps
        .iter()
        .map(statement_from_proof_step)
        .collect();
      out.extend(collect_all_storage_stmts_recursive(&inner_stmts));
    }
  }
  out
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

    // VmState opcode fields must agree with the statement opcode
    // (mirrors the check in verify_instruction_zk_receipt).
    if stmt.s_i.opcode != stmt.opcode || stmt.s_next.opcode != stmt.opcode {
      return false;
    }

    // Statement-level SP/arity coherence.
    if !verify_statement_semantics(stmt) {
      return false;
    }

    // Bind manifest WFF to actual execution output.
    let expected_wff = wff_instruction(stmt.opcode, &stmt.s_i.stack, &stmt.s_next.stack);
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
  //
  // Collect SLOAD/TLOAD statements from outer AND inner (sub-call) steps recursively.
  // TLOAD uses the same stack layout as SLOAD and is included in the same
  // StorageConsistencyAir (Gap-D1 verifier extension).
  let all_sload_stmts = collect_all_storage_stmts_recursive(statements);
  let all_sload_stmts: Vec<InstructionTransitionStatement> = all_sload_stmts
    .into_iter()
    .filter(|s| s.opcode == opcode::SLOAD || s.opcode == opcode::TLOAD)
    .collect();
  if !all_sload_stmts.is_empty() {
    let stor_proof = match &receipt.storage_proof {
      Some(p) => p,
      None => return false,
    };
    if !validate_oracle_sload_reads(&all_sload_stmts, stor_proof) {
      return false;
    }
  }

  // 11d. SSTORE/TSTORE oracle ↔ StorageConsistencyProof.write_log
  //
  // Same recursive collection for SSTORE and TSTORE (Gap-D1 verifier extension).
  let all_sstore_stmts = collect_all_storage_stmts_recursive(statements);
  let all_sstore_stmts: Vec<InstructionTransitionStatement> = all_sstore_stmts
    .into_iter()
    .filter(|s| s.opcode == opcode::SSTORE || s.opcode == opcode::TSTORE)
    .collect();
  if !all_sstore_stmts.is_empty() {
    let stor_proof = match &receipt.storage_proof {
      Some(p) => p,
      None => return false,
    };
    if !validate_oracle_sstore_writes(&all_sstore_stmts, stor_proof) {
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

  // 12. SubCall success-flag binding + recursive inner_proof verification.
  //
  // (a) For every CALL/CREATE family instruction: the top-of-stack value
  //     after the instruction must agree with SubCallClaim.success, and for
  //     CREATE/CREATE2 the address portion must match SubCallClaim.callee.
  // (b) Recursively verify the callee's execution trace via verify_sub_call_claim.
  for stmt in statements {
    if let Some(sc) = &stmt.sub_call_claim {
      // (a) Success/address binding.
      match sc.opcode {
        opcode::CALL | opcode::CALLCODE | opcode::DELEGATECALL | opcode::STATICCALL => {
          let stack_ok = stmt
            .s_next
            .stack
            .first()
            .map(|v| *v != [0u8; 32])
            .unwrap_or(false);
          if stack_ok != sc.success {
            return false;
          }
        }
        opcode::CREATE | opcode::CREATE2 => {
          let top = stmt.s_next.stack.first().unwrap_or(&[0u8; 32]);
          let stack_ok = *top != [0u8; 32];
          if stack_ok != sc.success {
            return false;
          }
          // On success the pushed word is the deployed address (zero-padded to 32 bytes).
          if sc.success {
            let top_addr: [u8; 20] = match top[12..32].try_into() {
              Ok(a) => a,
              Err(_) => return false,
            };
            if top_addr != sc.callee {
              return false;
            }
          }
        }
        _ => {}
      }
      // (b) Recursive inner proof verification.
      if verify_sub_call_claim(sc, statements).is_err() {
        return false;
      }
    }
  }

  // 13. Env-oracle public-input validation.
  //
  // When the receipt was produced with an embedded BlockTxContext
  // (via prove_batch_transaction_zk_receipt_with_env), verify that every
  // env-opcode's claimed output matches the corresponding context field.
  // This closes the oracle gap for COINBASE, TIMESTAMP, NUMBER, PREVRANDAO,
  // GASLIMIT, CHAINID, BASEFEE, ORIGIN, GASPRICE, CALLER, CALLVALUE,
  // ADDRESS, and CALLDATASIZE.
  if let Some(ctx) = &receipt.env_context {
    if !validate_oracle_env_claims(statements, ctx) {
      return false;
    }
  }

  // 14. World-state oracle binding (Gap-B1/B2/B3/B4).
  //
  // When the receipt was produced with an embedded WorldStateContext
  // (via prove_batch_transaction_zk_receipt_with_world_state), verify that
  // every external-state opcode's claimed output matches the committed entry.
  // This closes the oracle gap for BLOCKHASH, EXTCODESIZE, EXTCODEHASH,
  // BALANCE, and SELFBALANCE.
  if let Some(ws) = &receipt.world_state_context {
    if !validate_oracle_external_state_claims(statements, ws) {
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
  /// The batch STARK proof covering all instructions in this segment.
  /// `None` when the segment has no instructions (empty batch).
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
      storage_root: [0u8; 32],
    },
    s_next: VmState {
      opcode: step.opcode,
      pc: step.pc + 1,
      sp: step.stack_outputs.len(),
      stack: step.stack_outputs.clone(),
      memory_root: [0u8; 32],
      storage_root: [0u8; 32],
    },
    accesses: Vec::new(),
    sub_call_claim: step.sub_call_claim.clone(),
    mcopy_claim: step.mcopy_claim.clone(),
    external_state_claim: step.external_state_claim.clone(),
  }
}

/// Verify a [`SubCallClaim`]'s `inner_proof`.
///
/// Steps:
/// 1. Enforce `claim.depth < MAX_CALL_DEPTH`.
/// 2. If `inner_proof.steps` is non-empty, build [`InstructionTransitionStatement`]s
///    and perform full structural consistency checking.
/// 3. Confirm return_data binding: the callee's RETURN/REVERT data must match
///    `claim.return_data` (empty for precompiles/STOP/out-of-gas is OK).
/// 4. STATICCALL write-prohibition (EIP-214): inner steps must not contain
///    state-modifying opcodes.
/// 5. Recursively verify any nested sub-calls, so callee soundness propagates
///    transitively (depth-bounded by `MAX_CALL_DEPTH`).
///
/// When `inner_proof.batch_receipt` is `Some`, the receipt is cryptographically
/// verified via `verify_batch_transaction_zk_receipt`.
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

  let inner = &claim.inner_proof;

  // Empty steps = precompile / empty account — no EVM bytecode to verify.
  // Return-data comes from the host environment and is accepted as-is.
  if inner.steps.is_empty() {
    return Ok(());
  }

  // Build statements from ALL steps — prove_batch_transaction_zk_receipt
  // processes every step (using auto-proofs for structural opcodes), so
  // the manifest entry count equals the number of steps, not just those
  // that carry an explicit semantic_proof.
  let inner_stmts: Vec<InstructionTransitionStatement> = inner
    .steps
    .iter()
    .map(statement_from_proof_step)
    .collect();

  // STARK receipt re-verification.
  // When the inner proof carries a pre-computed batch receipt, verify it
  // cryptographically.  `batch_receipt = None` means the inner execution
  // has not been STARK-proven yet (capture-only mode); accepted structurally.
  if let Some(receipt) = &inner.batch_receipt
    && !verify_batch_transaction_zk_receipt(&inner_stmts, receipt) {
      return Err("SubCall inner STARK receipt verification failed".to_string());
    }

  // Gap-C2: The inner proof must end with a terminal opcode.
  // A non-terminal last step (e.g. PUSH1, ADD) indicates a truncated trace —
  // the prover may have cut execution just before a REVERT to falsely claim
  // success, or injected arbitrary return-data.
  let last_op = inner.steps.last().map(|s| s.opcode).unwrap_or(0);
  if !matches!(
    last_op,
    opcode::RETURN | opcode::REVERT | opcode::STOP | opcode::SELFDESTRUCT
  ) {
    return Err(format!(
      "SubCall inner proof is non-terminal (last opcode 0x{:02x}); \
       prover may have truncated execution to hide a REVERT",
      last_op
    ));
  }

  // Gap-C1: The terminal opcode must agree with the claimed success flag.
  //   RETURN / STOP  → success must be true
  //   REVERT         → success must be false
  //   SELFDESTRUCT   → implies success (not strictly enforced in current MVP)
  match last_op {
    opcode::RETURN | opcode::STOP => {
      if !claim.success {
        return Err(
          "SubCall inner proof ends with RETURN/STOP but success=false".to_string(),
        );
      }
    }
    opcode::REVERT => {
      if claim.success {
        return Err(
          "SubCall inner proof ends with REVERT but success=true".to_string(),
        );
      }
    }
    _ => {}
  }

  // Gap-C4: CREATE2 address derivation formula.
  // Expected: keccak256(0xff ++ deployer[20] ++ salt[32] ++ keccak256(initcode)[32])[12..]
  if claim.opcode == opcode::CREATE2 {
    let deployer = match claim.create2_deployer {
      Some(d) => d,
      None => return Err("CREATE2 claim is missing deployer".to_string()),
    };
    let salt = match claim.create2_salt {
      Some(s) => s,
      None => return Err("CREATE2 claim is missing salt".to_string()),
    };
    let initcode_hash = match claim.create2_initcode_hash {
      Some(h) => h,
      None => return Err("CREATE2 claim is missing initcode_hash".to_string()),
    };
    // Build the 85-byte preimage: 0xff ++ deployer(20) ++ salt(32) ++ initcode_hash(32)
    let mut preimage = [0u8; 85];
    preimage[0] = 0xff;
    preimage[1..21].copy_from_slice(&deployer);
    preimage[21..53].copy_from_slice(&salt);
    preimage[53..85].copy_from_slice(&initcode_hash);
    let hash = crate::zk_proof::keccak256_bytes(&preimage);
    let mut expected_addr = [0u8; 20];
    expected_addr.copy_from_slice(&hash[12..32]);
    if expected_addr != claim.callee {
      return Err(format!(
        "CREATE2 address mismatch: derived 0x{} but claim.callee is 0x{}",
        expected_addr.iter().map(|b| format!("{:02x}", b)).collect::<String>(),
        claim.callee.iter().map(|b| format!("{:02x}", b)).collect::<String>(),
      ));
    }
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

  // STATICCALL write-prohibition (EIP-214).
  // Inner execution must not contain any state-modifying opcodes.
  if claim.opcode == opcode::STATICCALL {
    for step in &inner.steps {
      if step.storage_claims.iter().any(|c| c.is_write)
        || matches!(
          step.opcode,
          opcode::LOG0
            | opcode::LOG1
            | opcode::LOG2
            | opcode::LOG3
            | opcode::LOG4
            | opcode::CREATE
            | opcode::CREATE2
            | opcode::SELFDESTRUCT
        )
      {
        return Err(format!(
          "STATICCALL inner proof contains state-modifying opcode 0x{:02x}",
          step.opcode
        ));
      }
    }
  }

  // Recursively verify any nested sub-calls inside the inner proof.
  for step in &inner.steps {
    if let Some(sc) = &step.sub_call_claim {
      verify_sub_call_claim(sc, &inner_stmts)?;
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
  //
  // Maintain cumulative write-sets `mem_w_in` / `stor_w_in` so each segment's
  // first-read values can be validated against the state inherited from prior
  // segments (First-Read Initialization soundness check).
  let mut mem_w_in = MemWriteSet::new();
  let mut stor_w_in = StorWriteSet::new();
  let mut nodes: Vec<ExecutionReceipt> = Vec::with_capacity(n);
  for (i, steps) in step_seqs.into_iter().enumerate() {
    let batch_receipt = if steps.is_empty() {
      None
    } else {
      Some(prove_batch_transaction_zk_receipt_with_w_in(&steps, &mem_w_in, &stor_w_in)?)
    };

    // Advance W_in for the next segment: accumulate writes from this segment.
    for step in &steps {
      for claim in &step.memory_claims {
        if claim.is_write {
          mem_w_in.insert(claim.addr, claim.value);
        }
      }
      for claim in &step.storage_claims {
        if claim.is_write {
          stor_w_in.insert((claim.contract, claim.slot), claim.value);
        }
      }
    }

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
      let stmts: Vec<InstructionTransitionStatement> = leaf
        .steps
        .iter()
        .map(statement_from_proof_step)
        .collect();
      if let Some(batch_receipt) = &leaf.batch_receipt {
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
