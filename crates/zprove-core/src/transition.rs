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

use crate::memory_proof::{verify_memory_access_commitments, verify_memory_semantics};
use crate::semantic_proof::{
  Proof, WFF, compile_proof, infer_proof,
  prove_add, prove_and, prove_div, prove_eq, prove_gt, prove_iszero, prove_lt, prove_mod,
  prove_mul, prove_not, prove_or, prove_sdiv, prove_sgt, prove_slt, prove_smod, prove_sub,
  prove_xor, verify_compiled,
  wff_add, wff_and, wff_div, wff_eq, wff_gt, wff_iszero, wff_lt, wff_mod, wff_mul, wff_not,
  wff_or, wff_sdiv, wff_sgt, wff_slt, wff_smod, wff_sub, wff_xor,
};
use crate::zk_proof::{
  CircleStarkProof, RECEIPT_BIND_TAG_LUT, RECEIPT_BIND_TAG_STACK, RECEIPT_BIND_TAG_WFF,
  LutStep,
  build_lut_steps_from_rows_add_family, build_lut_steps_from_rows_bit_family,
  build_lut_steps_from_rows_mul_family, make_receipt_binding_public_values,
  prove_expected_wff_match_stark_with_public_values,
  prove_lut_kernel_stark_with_public_values,
  prove_stack_ir_scaffold_stark_with_public_values,
  verify_lut_kernel_stark_with_public_values, verify_stack_ir_scaffold_stark_with_public_values,
  verify_wff_match_stark_with_public_values,
};
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
  pub wff_match_proof: CircleStarkProof,
  pub expected_wff: WFF,
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

  verify_memory_semantics(statement)
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
  lut_steps_override: Option<Vec<LutStep>>,
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
  let wff_bind_pv = make_receipt_binding_public_values(RECEIPT_BIND_TAG_WFF, proof.opcode, &expected_wff);

  let lut_steps = if let Some(steps) = lut_steps_override {
    if steps.is_empty() {
      return Err("provided LUT step override is empty".to_string());
    }
    match proof.opcode {
      opcode::ADD => steps,
      opcode::AND | opcode::OR | opcode::XOR | opcode::NOT => steps,
      _ => {
        return Err(format!(
          "LUT step override is only supported for ADD/bit-family opcodes (got 0x{:02x})",
          proof.opcode
        ));
      }
    }
  } else {
    match proof.opcode {
      opcode::ADD | opcode::SUB => build_lut_steps_from_rows_add_family(&rows)?,
      opcode::MUL | opcode::DIV | opcode::MOD | opcode::SDIV | opcode::SMOD => {
        build_lut_steps_from_rows_mul_family(&rows)?
      }
      opcode::AND | opcode::OR | opcode::XOR | opcode::NOT => {
        build_lut_steps_from_rows_bit_family(&rows)?
      }
      _ => {
        return Err(format!(
          "unsupported opcode family for LUT kernel receipt: 0x{:02x}",
          proof.opcode
        ));
      }
    }
  };

  let stack_ir_proof =
    prove_stack_ir_scaffold_stark_with_public_values(&rows, &stack_bind_pv)?;
  let lut_kernel_proof =
    prove_lut_kernel_stark_with_public_values(&lut_steps, &lut_bind_pv)?;
  let wff_match_proof = prove_expected_wff_match_stark_with_public_values(
    semantic_proof,
    &expected_wff,
    &wff_bind_pv,
  )?;

  Ok(InstructionTransitionZkReceipt {
    opcode: proof.opcode,
    stack_ir_proof,
    lut_kernel_proof,
    wff_match_proof,
    expected_wff,
  })
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
      let result = prove_instruction_zk_receipt_with_lut_override(&proof, None);
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

  if !verify_memory_access_commitments(statement) {
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
  let wff_bind_pv =
    make_receipt_binding_public_values(RECEIPT_BIND_TAG_WFF, statement.opcode, &expected_wff);

  if verify_stack_ir_scaffold_stark_with_public_values(&receipt.stack_ir_proof, &stack_bind_pv)
    .is_err()
  {
    return false;
  }

  if verify_lut_kernel_stark_with_public_values(&receipt.lut_kernel_proof, &lut_bind_pv).is_err()
  {
    return false;
  }

  verify_wff_match_stark_with_public_values(&receipt.wff_match_proof, &wff_bind_pv).is_ok()
}

// ============================================================
// Tests
// ============================================================
