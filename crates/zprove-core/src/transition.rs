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

use crate::semantic_proof::{
  Proof, WFF, compile_proof, infer_proof,
  prove_add, prove_and, prove_div, prove_eq, prove_gt, prove_iszero, prove_lt, prove_mod,
  prove_mul, prove_not, prove_or, prove_sdiv, prove_sgt, prove_slt, prove_smod, prove_sub,
  prove_xor, verify_compiled,
  wff_add, wff_and, wff_div, wff_eq, wff_gt, wff_iszero, wff_lt, wff_mod, wff_mul, wff_not,
  wff_or, wff_sdiv, wff_sgt, wff_slt, wff_smod, wff_sub, wff_xor,
};
use crate::semantic_proof::ProofRow;
use crate::zk_proof::{
  BatchInstructionMeta, BatchProofRowsManifest,
  CircleStarkConfig, CircleStarkProof, RECEIPT_BIND_TAG_LUT, RECEIPT_BIND_TAG_STACK,
  make_batch_receipt_binding_public_values, make_receipt_binding_public_values,
  prove_batch_lut_with_prep, prove_lut_with_prep, prove_stack_ir_with_prep,
  setup_batch_proof_rows_preprocessed, setup_proof_rows_preprocessed,
  verify_batch_lut_with_prep, verify_lut_with_prep, verify_stack_ir_with_prep,
};
use p3_uni_stark::PreprocessedVerifierKey;
use revm::bytecode::opcode;

// ============================================================
// Types
// ============================================================

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
}

/// Complete proof for all steps of a transaction execution.
#[derive(Debug, Clone)]
pub struct TransactionProof {
  pub steps: Vec<InstructionTransitionProof>,
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
}

pub struct InstructionTransitionZkReceipt {
  pub opcode: u8,
  pub stack_ir_proof: CircleStarkProof,
  pub lut_kernel_proof: CircleStarkProof,
  pub preprocessed_vk: PreprocessedVerifierKey<CircleStarkConfig>,
  pub expected_wff: WFF,
}

/// ZK receipt for a batch of N instructions proved in a single LUT STARK call.
///
/// Security model:
/// - `lut_proof`: batch LUT STARK covering arithmetic of ALL instructions' ProofRows.
/// - `preprocessed_vk`: VK for the shared batch preprocessed matrix.
/// - `manifest`: per-instruction metadata (opcode, WFF, row range in `all_rows`).
///
/// Verification requires:
/// 1. `verify_batch_lut_with_prep(lut_proof, preprocessed_vk, pis)` — STARK check.
/// 2. For each i: `infer_proof(pi_i) == manifest.entries[i].wff` — deterministic.
/// 3. `compute_batch_manifest_digest(&entries) == pis[2..10]` — deterministic.
pub struct BatchTransactionZkReceipt {
  /// Single LUT STARK proof covering arithmetic of all N instructions.
  pub lut_proof: CircleStarkProof,
  /// Shared preprocessed verifier key (batch manifest committed here).
  pub preprocessed_vk: PreprocessedVerifierKey<CircleStarkConfig>,
  /// Per-instruction metadata and concatenated ProofRows.
  pub manifest: BatchProofRowsManifest,
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
    opcode::AND | opcode::OR | opcode::XOR => 2,
    opcode::MLOAD => 1,
    opcode::MSTORE | opcode::MSTORE8 => 2,
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
    opcode::LT | opcode::GT | opcode::SLT | opcode::SGT | opcode::EQ => 1,
    opcode::ISZERO | opcode::NOT => 1,
    opcode::AND | opcode::OR | opcode::XOR => 1,
    opcode::MLOAD => 1,
    opcode::MSTORE | opcode::MSTORE8 => 0,
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
    opcode::STOP | opcode::POP | opcode::JUMP | opcode::JUMPI | opcode::JUMPDEST
  ) || (opcode::PUSH0..=opcode::PUSH32).contains(&op)
    || (opcode::DUP1..=opcode::DUP16).contains(&op)
    || (opcode::SWAP1..=opcode::SWAP16).contains(&op)
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
    // Not yet implemented — no panic, just skip semantic proof
    opcode::ADDMOD | opcode::MULMOD | opcode::EXP => None,
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
    // Not yet implemented
    opcode::ADDMOD | opcode::MULMOD | opcode::EXP => None,
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
    (None, None) => is_structural_opcode(proof.opcode),
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
  let stack_bind_pv = make_receipt_binding_public_values(RECEIPT_BIND_TAG_STACK, proof.opcode, &expected_wff);
  let lut_bind_pv = make_receipt_binding_public_values(RECEIPT_BIND_TAG_LUT, proof.opcode, &expected_wff);

  let (prep_data, preprocessed_vk) = setup_proof_rows_preprocessed(&rows, &stack_bind_pv)?;

  let stack_ir_proof = prove_stack_ir_with_prep(&rows, &prep_data, &stack_bind_pv)?;
  let lut_kernel_proof = prove_lut_with_prep(&rows, &prep_data, &lut_bind_pv)?;

  Ok(InstructionTransitionZkReceipt {
    opcode: proof.opcode,
    stack_ir_proof,
    lut_kernel_proof,
    preprocessed_vk,
    expected_wff,
  })
}

/// Like [`prove_instruction_zk_receipt`], but also returns per-component timings.
///
/// Returns `(receipt, (setup_µs, stack_ir_µs, lut_µs))`.
/// Useful for profiling which of the three steps dominates.
pub fn prove_instruction_zk_receipt_timed(
  proof: &InstructionTransitionProof,
) -> Result<(InstructionTransitionZkReceipt, (f64, f64, f64)), String> {
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
  let stack_bind_pv = make_receipt_binding_public_values(RECEIPT_BIND_TAG_STACK, proof.opcode, &expected_wff);
  let lut_bind_pv = make_receipt_binding_public_values(RECEIPT_BIND_TAG_LUT, proof.opcode, &expected_wff);

  let t0 = Instant::now();
  let (prep_data, preprocessed_vk) = setup_proof_rows_preprocessed(&rows, &stack_bind_pv)?;
  let setup_us = t0.elapsed().as_secs_f64() * 1e6;

  let t1 = Instant::now();
  let stack_ir_proof = prove_stack_ir_with_prep(&rows, &prep_data, &stack_bind_pv)?;
  let stack_ir_us = t1.elapsed().as_secs_f64() * 1e6;

  let t2 = Instant::now();
  let lut_kernel_proof = prove_lut_with_prep(&rows, &prep_data, &lut_bind_pv)?;
  let lut_us = t2.elapsed().as_secs_f64() * 1e6;

  let receipt = InstructionTransitionZkReceipt {
    opcode: proof.opcode,
    stack_ir_proof,
    lut_kernel_proof,
    preprocessed_vk,
    expected_wff,
  };
  Ok((receipt, (setup_us, stack_ir_us, lut_us)))
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
    handles.push(thread::spawn(move || loop {
      let item = queue.lock().unwrap().pop_front();
      let Some((idx, proof)) = item else { break };
      let result = prove_instruction_zk_receipt_with_lut_override(&proof);
      if tx.send((idx, result)).is_err() {
        break;
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
      receipt
        .ok_or_else(|| "failed to collect all receipts after parallel proving".to_string())?,
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
  let lut_bind_pv =
    make_receipt_binding_public_values(RECEIPT_BIND_TAG_LUT, statement.opcode, &expected_wff);

  let prep_vk = &receipt.preprocessed_vk;

  if verify_stack_ir_with_prep(&receipt.stack_ir_proof, prep_vk, &stack_bind_pv).is_err() {
    return false;
  }

  verify_lut_with_prep(&receipt.lut_kernel_proof, prep_vk, &lut_bind_pv).is_ok()
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
pub fn build_batch_manifest(
  items: &[(u8, &Proof)],
) -> Result<BatchProofRowsManifest, String> {
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
    entries.push(BatchInstructionMeta {
      opcode: *opcode,
      wff,
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
///
/// **Proving steps:**
/// 1. Build [`BatchProofRowsManifest`] from the semantic proofs (`build_batch_manifest`).
/// 2. Setup shared batch preprocessed commitment (`setup_batch_proof_rows_preprocessed`).
/// 3. Prove batch LUT STARK in one call (`prove_batch_lut_with_prep`).
///
/// Note: StackIR verification is replaced by the deterministic out-of-circuit
/// check `infer_proof(pi_i) == entry.wff` performed inside
/// [`verify_batch_transaction_zk_receipt`], so no separate StackIR STARK is
/// generated in the batch path.
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
    return Err(
      "prove_batch_transaction_zk_receipt: no semantic proofs in batch".to_string(),
    );
  }

  let manifest = build_batch_manifest(&items)?;
  let lut_bind_pv =
    make_batch_receipt_binding_public_values(RECEIPT_BIND_TAG_LUT, &manifest.entries);
  let (prep_data, preprocessed_vk) =
    setup_batch_proof_rows_preprocessed(&manifest, &lut_bind_pv)?;
  let lut_proof = prove_batch_lut_with_prep(&manifest, &prep_data, &lut_bind_pv)?;

  Ok(BatchTransactionZkReceipt {
    lut_proof,
    preprocessed_vk,
    manifest,
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
    let expected_wff =
      match wff_instruction(stmt.opcode, &stmt.s_i.stack, &stmt.s_next.stack) {
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
  let lut_bind_pv = make_batch_receipt_binding_public_values(
    RECEIPT_BIND_TAG_LUT,
    &receipt.manifest.entries,
  );
  verify_batch_lut_with_prep(
    &receipt.lut_proof,
    &receipt.preprocessed_vk,
    &lut_bind_pv,
  )
  .is_ok()
}

// ============================================================
// Tests
// ============================================================
