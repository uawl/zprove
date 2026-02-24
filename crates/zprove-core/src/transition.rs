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

use crate::sementic_proof::{
  Proof,
  WFF,
  compile_proof,
  infer_proof,
  prove_add,
  prove_div,
  prove_and,
  prove_mod,
  prove_mul,
  prove_not,
  prove_or,
  prove_sdiv,
  prove_smod,
  prove_sub,
  prove_xor,
  verify_compiled,
  wff_and,
  wff_add,
  wff_div,
  wff_mod,
  wff_mul,
  wff_not,
  wff_or,
  wff_sdiv,
  wff_smod,
  wff_sub,
  wff_xor,
};
use crate::memory_proof::{verify_memory_access_commitments, verify_memory_semantics};
use crate::zk_proof::{
  CircleStarkProof,
  build_lut_steps_from_rows_add_family,
  build_lut_steps_from_rows_bit_family,
  build_lut_steps_from_rows_mul_family,
  prove_expected_wff_match_stark,
  prove_lut_kernel_stark,
  prove_stack_ir_scaffold_stark,
  verify_lut_kernel_stark,
  verify_stack_ir_scaffold_stark,
  verify_wff_match_stark,
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
    opcode::STOP
      | opcode::POP
      | opcode::JUMP
      | opcode::JUMPI
      | opcode::JUMPDEST
  ) || (opcode::PUSH0..=opcode::PUSH32).contains(&op)
    || (opcode::DUP1..=opcode::DUP16).contains(&op)
    || (opcode::SWAP1..=opcode::SWAP16).contains(&op)
}

pub fn verify_statement_semantics(statement: &InstructionTransitionStatement) -> bool {
  if !has_valid_sp_transition(statement.opcode, statement.s_i.sp, statement.s_next.sp) {
    return false;
  }

  if !has_expected_stack_arity(statement.opcode, &statement.s_i.stack, &statement.s_next.stack)
  {
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
    opcode::ADDMOD
    | opcode::MULMOD
    | opcode::EXP
    | opcode::LT
    | opcode::GT
    | opcode::SLT
    | opcode::SGT
    | opcode::EQ
    | opcode::ISZERO
    => todo!("semantic proof path not implemented yet for opcode 0x{op:02x}"),
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
    opcode::ADDMOD
    | opcode::MULMOD
    | opcode::EXP
    | opcode::LT
    | opcode::GT
    | opcode::SLT
    | opcode::SGT
    | opcode::EQ
    | opcode::ISZERO
    => todo!("WFF path not implemented yet for opcode 0x{op:02x}"),
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
    },
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
  let lut_steps = match proof.opcode {
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
  };

  let stack_ir_proof = prove_stack_ir_scaffold_stark(&rows)?;
  let lut_kernel_proof = prove_lut_kernel_stark(&lut_steps)?;
  let wff_match_proof = prove_expected_wff_match_stark(semantic_proof, &expected_wff)?;

  Ok(InstructionTransitionZkReceipt {
    opcode: proof.opcode,
    stack_ir_proof,
    lut_kernel_proof,
    wff_match_proof,
    expected_wff,
  })
}

pub fn verify_instruction_zk_receipt(
  statement: &InstructionTransitionStatement,
  receipt: &InstructionTransitionZkReceipt,
) -> bool {
  if statement.s_i.opcode != statement.opcode {
    return false;
  }

  if statement.s_i.sp != statement.s_i.stack.len() || statement.s_next.sp != statement.s_next.stack.len() {
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

  if verify_stack_ir_scaffold_stark(&receipt.stack_ir_proof).is_err() {
    return false;
  }

  if verify_lut_kernel_stark(&receipt.lut_kernel_proof).is_err() {
    return false;
  }

  verify_wff_match_stark(&receipt.wff_match_proof).is_ok()
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
  use super::*;
  use crate::memory_proof::compute_memory_root;
  use crate::sementic_proof::Term;

  fn u256_bytes(val: u128) -> [u8; 32] {
    let mut b = [0u8; 32];
    b[16..32].copy_from_slice(&val.to_be_bytes());
    b
  }

  fn i256_bytes(val: i128) -> [u8; 32] {
    let mut b = if val < 0 { [0xFF; 32] } else { [0u8; 32] };
    b[16..32].copy_from_slice(&val.to_be_bytes());
    b
  }

  #[test]
  fn test_sub_transition_via_add_form() {
    // SUB(a, b) = c is proven as a = b + c
    let a = u256_bytes(3000);
    let b = u256_bytes(2000);
    let c = u256_bytes(1000);

    let proof = prove_instruction(opcode::SUB, &[a, b], &[c]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::SUB,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![c],
      semantic_proof: Some(proof),
    };
    assert!(verify_proof(&itp));
    assert!(verify_proof_with_rows(&itp));
    assert!(verify_proof_with_zkp(&itp));
  }

  #[test]
  fn test_zkp_rejects_missing_semantic_proof() {
    let itp = InstructionTransitionProof {
      opcode: opcode::PUSH1,
      pc: 0,
      stack_inputs: vec![],
      stack_outputs: vec![u256_bytes(1)],
      semantic_proof: None,
    };
    assert!(!verify_proof_with_zkp(&itp));
  }

  #[test]
  fn test_zkp_rejects_forged_semantic_proof_shape() {
    let a = u256_bytes(1000);
    let b = u256_bytes(2000);
    let c = u256_bytes(3000);

    // Malicious prover tries to attach an unrelated trivial proof.
    let forged = Proof::EqRefl(Term::Byte(0));
    let itp = InstructionTransitionProof {
      opcode: opcode::ADD,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![c],
      semantic_proof: Some(forged),
    };

    assert!(!verify_proof(&itp));
    assert!(!verify_proof_with_rows(&itp));
    assert!(!verify_proof_with_zkp(&itp));
  }

  #[test]
  fn test_zkp_rejects_forged_opcode_proof_mismatch() {
    let a = u256_bytes(1000);
    let b = u256_bytes(2000);
    let c = u256_bytes(3000);

    // Malicious prover reuses ADD proof under wrong opcode claim.
    let forged = prove_add(&a, &b, &c);
    let itp = InstructionTransitionProof {
      opcode: opcode::MUL,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![c],
      semantic_proof: Some(forged),
    };

    assert!(!verify_proof(&itp));
    assert!(!verify_proof_with_rows(&itp));
    assert!(!verify_proof_with_zkp(&itp));
  }

  #[test]
  fn test_receipt_roundtrip_add() {
    let a = u256_bytes(1234);
    let b = u256_bytes(5678);
    let c = u256_bytes(6912);

    let semantic = prove_instruction(opcode::ADD, &[a, b], &[c]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::ADD,
      pc: 7,
      stack_inputs: vec![a, b],
      stack_outputs: vec![c],
      semantic_proof: Some(semantic),
    };

    let statement = InstructionTransitionStatement {
      opcode: itp.opcode,
      s_i: VmState {
        opcode: itp.opcode,
        pc: itp.pc,
        sp: itp.stack_inputs.len(),
        stack: itp.stack_inputs.clone(),
        memory_root: [0u8; 32],
      },
      s_next: VmState {
        opcode: itp.opcode,
        pc: itp.pc + 1,
        sp: itp.stack_outputs.len(),
        stack: itp.stack_outputs.clone(),
        memory_root: [0u8; 32],
      },
      accesses: Vec::new(),
    };

    let receipt = prove_instruction_zk_receipt(&itp).expect("receipt proving should succeed");
    assert!(verify_instruction_zk_receipt(&statement, &receipt));
  }

  #[test]
  fn test_receipt_rejects_statement_mismatch() {
    let a = u256_bytes(1234);
    let b = u256_bytes(5678);
    let c = u256_bytes(6912);

    let semantic = prove_instruction(opcode::ADD, &[a, b], &[c]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::ADD,
      pc: 8,
      stack_inputs: vec![a, b],
      stack_outputs: vec![c],
      semantic_proof: Some(semantic),
    };

    let receipt = prove_instruction_zk_receipt(&itp).expect("receipt proving should succeed");

    let mut wrong_c = c;
    wrong_c[31] ^= 1;
    let wrong_statement = InstructionTransitionStatement {
      opcode: opcode::ADD,
      s_i: VmState {
        opcode: opcode::ADD,
        pc: 8,
        sp: 2,
        stack: vec![a, b],
        memory_root: [0u8; 32],
      },
      s_next: VmState {
        opcode: opcode::ADD,
        pc: 9,
        sp: 1,
        stack: vec![wrong_c],
        memory_root: [0u8; 32],
      },
      accesses: Vec::new(),
    };

    assert!(!verify_instruction_zk_receipt(&wrong_statement, &receipt));
  }

  #[test]
  fn test_receipt_proving_rejects_forged_semantic_shape() {
    let a = u256_bytes(1000);
    let b = u256_bytes(2000);
    let c = u256_bytes(3000);

    let forged = Proof::EqRefl(Term::Byte(0));
    let itp = InstructionTransitionProof {
      opcode: opcode::ADD,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![c],
      semantic_proof: Some(forged),
    };

    assert!(prove_instruction_zk_receipt(&itp).is_err());
  }

  #[test]
  fn test_receipt_rejects_wrong_sp_transition() {
    let a = u256_bytes(1234);
    let b = u256_bytes(5678);
    let c = u256_bytes(6912);

    let semantic = prove_instruction(opcode::ADD, &[a, b], &[c]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::ADD,
      pc: 9,
      stack_inputs: vec![a, b],
      stack_outputs: vec![c],
      semantic_proof: Some(semantic),
    };

    let receipt = prove_instruction_zk_receipt(&itp).expect("receipt proving should succeed");
    let wrong_sp_statement = InstructionTransitionStatement {
      opcode: opcode::ADD,
      s_i: VmState {
        opcode: opcode::ADD,
        pc: 9,
        sp: 1,
        stack: vec![a, b],
        memory_root: [0u8; 32],
      },
      s_next: VmState {
        opcode: opcode::ADD,
        pc: 10,
        sp: 1,
        stack: vec![c],
        memory_root: [0u8; 32],
      },
      accesses: Vec::new(),
    };

    assert!(!verify_instruction_zk_receipt(&wrong_sp_statement, &receipt));
  }

  #[test]
  fn test_receipt_rejects_mismatched_state_opcode() {
    let a = u256_bytes(1234);
    let b = u256_bytes(5678);
    let c = u256_bytes(6912);

    let semantic = prove_instruction(opcode::ADD, &[a, b], &[c]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::ADD,
      pc: 10,
      stack_inputs: vec![a, b],
      stack_outputs: vec![c],
      semantic_proof: Some(semantic),
    };

    let receipt = prove_instruction_zk_receipt(&itp).expect("receipt proving should succeed");
    let wrong_opcode_statement = InstructionTransitionStatement {
      opcode: opcode::ADD,
      s_i: VmState {
        opcode: opcode::MUL,
        pc: 10,
        sp: 2,
        stack: vec![a, b],
        memory_root: [0u8; 32],
      },
      s_next: VmState {
        opcode: opcode::MUL,
        pc: 11,
        sp: 1,
        stack: vec![c],
        memory_root: [0u8; 32],
      },
      accesses: Vec::new(),
    };

    assert!(!verify_instruction_zk_receipt(&wrong_opcode_statement, &receipt));
  }

  #[test]
  fn test_receipt_rejects_memory_read_without_proof_path() {
    let a = u256_bytes(1234);
    let b = u256_bytes(5678);
    let c = u256_bytes(6912);

    let semantic = prove_instruction(opcode::ADD, &[a, b], &[c]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::ADD,
      pc: 11,
      stack_inputs: vec![a, b],
      stack_outputs: vec![c],
      semantic_proof: Some(semantic),
    };

    let receipt = prove_instruction_zk_receipt(&itp).expect("receipt proving should succeed");
    let statement = InstructionTransitionStatement {
      opcode: opcode::ADD,
      s_i: VmState {
        opcode: opcode::ADD,
        pc: 11,
        sp: 2,
        stack: vec![a, b],
        memory_root: [7u8; 32],
      },
      s_next: VmState {
        opcode: opcode::ADD,
        pc: 12,
        sp: 1,
        stack: vec![c],
        memory_root: [7u8; 32],
      },
      accesses: vec![AccessRecord {
        rw_counter: 1,
        domain: AccessDomain::Memory,
        kind: AccessKind::Read,
        addr: 0,
        width: 32,
        value_before: Some([0u8; 32]),
        value_after: None,
        merkle_path_before: Vec::new(),
        merkle_path_after: Vec::new(),
      }],
    };

    assert!(!verify_instruction_zk_receipt(&statement, &receipt));
  }

  #[test]
  fn test_mstore_commitment_update_verifies() {
    let offset = u256_bytes(64);
    let value = [9u8; 32];
    let word = value;
    let before_value = [0u8; 32];
    let after_value = value;
    let path_before = vec![[3u8; 32], [4u8; 32]];
    let path_after = path_before.clone();
    let root_before = compute_memory_root(64, 32, &before_value, &path_before);
    let root_after = compute_memory_root(64, 32, &after_value, &path_after);

    let statement = InstructionTransitionStatement {
      opcode: opcode::MSTORE,
      s_i: VmState {
        opcode: opcode::MSTORE,
        pc: 12,
        sp: 2,
        stack: vec![offset, word],
        memory_root: root_before,
      },
      s_next: VmState {
        opcode: opcode::MSTORE,
        pc: 13,
        sp: 0,
        stack: vec![],
        memory_root: root_after,
      },
      accesses: vec![AccessRecord {
        rw_counter: 1,
        domain: AccessDomain::Memory,
        kind: AccessKind::Write,
        addr: 64,
        width: 32,
        value_before: Some(before_value),
        value_after: Some(after_value),
        merkle_path_before: path_before,
        merkle_path_after: path_after,
      }],
    };

    assert!(verify_statement_semantics(&statement));
    assert!(verify_memory_access_commitments(&statement));
  }

  #[test]
  fn test_mload_statement_semantics_passes() {
    let offset = u256_bytes(64);
    let loaded = u256_bytes(0xABCD);
    let statement = InstructionTransitionStatement {
      opcode: opcode::MLOAD,
      s_i: VmState {
        opcode: opcode::MLOAD,
        pc: 0,
        sp: 1,
        stack: vec![offset],
        memory_root: [0u8; 32],
      },
      s_next: VmState {
        opcode: opcode::MLOAD,
        pc: 1,
        sp: 1,
        stack: vec![loaded],
        memory_root: [0u8; 32],
      },
      accesses: vec![AccessRecord {
        rw_counter: 1,
        domain: AccessDomain::Memory,
        kind: AccessKind::Read,
        addr: 64,
        width: 32,
        value_before: Some(loaded),
        value_after: None,
        merkle_path_before: vec![[1u8; 32]],
        merkle_path_after: Vec::new(),
      }],
    };
    assert!(verify_statement_semantics(&statement));
  }

  #[test]
  fn test_mstore_statement_semantics_passes() {
    let offset = u256_bytes(96);
    let value = u256_bytes(0xFEED);
    let statement = InstructionTransitionStatement {
      opcode: opcode::MSTORE,
      s_i: VmState {
        opcode: opcode::MSTORE,
        pc: 0,
        sp: 2,
        stack: vec![offset, value],
        memory_root: [0u8; 32],
      },
      s_next: VmState {
        opcode: opcode::MSTORE,
        pc: 1,
        sp: 0,
        stack: vec![],
        memory_root: [0u8; 32],
      },
      accesses: vec![AccessRecord {
        rw_counter: 1,
        domain: AccessDomain::Memory,
        kind: AccessKind::Write,
        addr: 96,
        width: 32,
        value_before: Some([0u8; 32]),
        value_after: Some(value),
        merkle_path_before: vec![[2u8; 32]],
        merkle_path_after: vec![[2u8; 32]],
      }],
    };
    assert!(verify_statement_semantics(&statement));
  }

  #[test]
  fn test_mstore8_statement_semantics_rejects_wrong_byte_projection() {
    let offset = u256_bytes(130);
    let mut word = [0u8; 32];
    word[31] = 0xAA;
    let mut before_chunk = [0u8; 32];
    before_chunk[2] = 0x11;
    let mut wrong_after = [0u8; 32];
    wrong_after[31] = 0xBB;

    let statement = InstructionTransitionStatement {
      opcode: opcode::MSTORE8,
      s_i: VmState {
        opcode: opcode::MSTORE8,
        pc: 0,
        sp: 2,
        stack: vec![offset, word],
        memory_root: [0u8; 32],
      },
      s_next: VmState {
        opcode: opcode::MSTORE8,
        pc: 1,
        sp: 0,
        stack: vec![],
        memory_root: [0u8; 32],
      },
      accesses: vec![AccessRecord {
        rw_counter: 1,
        domain: AccessDomain::Memory,
        kind: AccessKind::Write,
        addr: 128,
        width: 32,
        value_before: Some(before_chunk),
        value_after: Some(wrong_after),
        merkle_path_before: vec![[3u8; 32]],
        merkle_path_after: vec![[3u8; 32]],
      }],
    };
    assert!(!verify_statement_semantics(&statement));
  }

  #[test]
  fn test_mstore8_statement_semantics_passes_on_single_byte_patch() {
    let offset = u256_bytes(130);
    let mut word = [0u8; 32];
    word[31] = 0xAB;
    let mut before_chunk = [0u8; 32];
    before_chunk[2] = 0x11;
    let mut after_chunk = before_chunk;
    after_chunk[2] = 0xAB;

    let statement = InstructionTransitionStatement {
      opcode: opcode::MSTORE8,
      s_i: VmState {
        opcode: opcode::MSTORE8,
        pc: 0,
        sp: 2,
        stack: vec![offset, word],
        memory_root: [0u8; 32],
      },
      s_next: VmState {
        opcode: opcode::MSTORE8,
        pc: 1,
        sp: 0,
        stack: vec![],
        memory_root: [0u8; 32],
      },
      accesses: vec![AccessRecord {
        rw_counter: 1,
        domain: AccessDomain::Memory,
        kind: AccessKind::Write,
        addr: 128,
        width: 32,
        value_before: Some(before_chunk),
        value_after: Some(after_chunk),
        merkle_path_before: vec![[3u8; 32]],
        merkle_path_after: vec![[3u8; 32]],
      }],
    };
    assert!(verify_statement_semantics(&statement));
  }

  #[test]
  fn test_non_memory_opcode_rejects_memory_access_log() {
    let a = u256_bytes(1);
    let b = u256_bytes(2);
    let c = u256_bytes(3);
    let statement = InstructionTransitionStatement {
      opcode: opcode::ADD,
      s_i: VmState {
        opcode: opcode::ADD,
        pc: 0,
        sp: 2,
        stack: vec![a, b],
        memory_root: [0u8; 32],
      },
      s_next: VmState {
        opcode: opcode::ADD,
        pc: 1,
        sp: 1,
        stack: vec![c],
        memory_root: [0u8; 32],
      },
      accesses: vec![AccessRecord {
        rw_counter: 1,
        domain: AccessDomain::Memory,
        kind: AccessKind::Read,
        addr: 0,
        width: 32,
        value_before: Some([0u8; 32]),
        value_after: None,
        merkle_path_before: vec![[1u8; 32]],
        merkle_path_after: Vec::new(),
      }],
    };
    assert!(!verify_statement_semantics(&statement));
  }

  #[test]
  fn test_sub_transition_wrong_output_fails() {
    let a = u256_bytes(3000);
    let b = u256_bytes(2000);
    let wrong_c = u256_bytes(999);

    let proof = prove_instruction(opcode::SUB, &[a, b], &[wrong_c]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::SUB,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![wrong_c],
      semantic_proof: Some(proof),
    };
    assert!(!verify_proof(&itp));
    assert!(!verify_proof_with_rows(&itp));
    assert!(!verify_proof_with_zkp(&itp));
  }

  #[test]
  fn test_mul_transition_byte_local_success() {
    let a = u256_bytes(123);
    let b = u256_bytes(45);
    let c = u256_bytes(5535);

    let proof = prove_instruction(opcode::MUL, &[a, b], &[c]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::MUL,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![c],
      semantic_proof: Some(proof),
    };
    assert!(verify_proof(&itp));
    assert!(verify_proof_with_rows(&itp));
    assert!(verify_proof_with_zkp(&itp));
  }

  #[test]
  fn test_mul_transition_byte_local_wrong_output_fails() {
    let a = u256_bytes(123);
    let b = u256_bytes(45);
    let wrong = u256_bytes(5534);

    let proof = prove_instruction(opcode::MUL, &[a, b], &[wrong]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::MUL,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![wrong],
      semantic_proof: Some(proof),
    };
    assert!(!verify_proof(&itp));
    assert!(!verify_proof_with_rows(&itp));
    assert!(!verify_proof_with_zkp(&itp));
  }

  #[test]
  fn test_div_transition_pair_form_success() {
    let a = u256_bytes(1000);
    let b = u256_bytes(30);
    let q = u256_bytes(33);

    let proof = prove_instruction(opcode::DIV, &[a, b], &[q]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::DIV,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![q],
      semantic_proof: Some(proof),
    };
    assert!(verify_proof(&itp));
    assert!(verify_proof_with_rows(&itp));
    assert!(verify_proof_with_zkp(&itp));
  }

  #[test]
  fn test_div_transition_pair_form_wrong_output_fails() {
    let a = u256_bytes(1000);
    let b = u256_bytes(30);
    let wrong_q = u256_bytes(34);

    let proof = prove_instruction(opcode::DIV, &[a, b], &[wrong_q]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::DIV,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![wrong_q],
      semantic_proof: Some(proof),
    };
    assert!(!verify_proof(&itp));
    assert!(!verify_proof_with_rows(&itp));
    assert!(!verify_proof_with_zkp(&itp));
  }

  #[test]
  fn test_mod_transition_pair_form_success() {
    let a = u256_bytes(1000);
    let b = u256_bytes(30);
    let r = u256_bytes(10);

    let proof = prove_instruction(opcode::MOD, &[a, b], &[r]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::MOD,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![r],
      semantic_proof: Some(proof),
    };
    assert!(verify_proof(&itp));
    assert!(verify_proof_with_rows(&itp));
    assert!(verify_proof_with_zkp(&itp));
  }

  #[test]
  fn test_mod_transition_pair_form_wrong_output_fails() {
    let a = u256_bytes(1000);
    let b = u256_bytes(30);
    let wrong_r = u256_bytes(11);

    let proof = prove_instruction(opcode::MOD, &[a, b], &[wrong_r]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::MOD,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![wrong_r],
      semantic_proof: Some(proof),
    };
    assert!(!verify_proof(&itp));
    assert!(!verify_proof_with_rows(&itp));
    assert!(!verify_proof_with_zkp(&itp));
  }

  #[test]
  fn test_div_mod_zero_divisor_output_zero() {
    let a = u256_bytes(1000);
    let b = u256_bytes(0);
    let z = u256_bytes(0);

    let div_proof = prove_instruction(opcode::DIV, &[a, b], &[z]).unwrap();
    let div_itp = InstructionTransitionProof {
      opcode: opcode::DIV,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![z],
      semantic_proof: Some(div_proof),
    };
    assert!(verify_proof(&div_itp));
    assert!(verify_proof_with_zkp(&div_itp));

    let mod_proof = prove_instruction(opcode::MOD, &[a, b], &[z]).unwrap();
    let mod_itp = InstructionTransitionProof {
      opcode: opcode::MOD,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![z],
      semantic_proof: Some(mod_proof),
    };
    assert!(verify_proof(&mod_itp));
    assert!(verify_proof_with_zkp(&mod_itp));
  }

  #[test]
  fn test_sdiv_transition_pair_form_success() {
    let a = i256_bytes(-1000);
    let b = i256_bytes(30);
    let q = i256_bytes(-33);

    let proof = prove_instruction(opcode::SDIV, &[a, b], &[q]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::SDIV,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![q],
      semantic_proof: Some(proof),
    };
    assert!(verify_proof(&itp));
    assert!(verify_proof_with_rows(&itp));
    assert!(verify_proof_with_zkp(&itp));
  }

  #[test]
  fn test_sdiv_transition_pair_form_wrong_output_fails() {
    let a = i256_bytes(-1000);
    let b = i256_bytes(30);
    let wrong_q = i256_bytes(-34);

    let proof = prove_instruction(opcode::SDIV, &[a, b], &[wrong_q]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::SDIV,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![wrong_q],
      semantic_proof: Some(proof),
    };
    assert!(!verify_proof(&itp));
    assert!(!verify_proof_with_rows(&itp));
    assert!(!verify_proof_with_zkp(&itp));
  }

  #[test]
  fn test_smod_transition_pair_form_success() {
    let a = i256_bytes(-1000);
    let b = i256_bytes(30);
    let r = i256_bytes(-10);

    let proof = prove_instruction(opcode::SMOD, &[a, b], &[r]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::SMOD,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![r],
      semantic_proof: Some(proof),
    };
    assert!(verify_proof(&itp));
    assert!(verify_proof_with_rows(&itp));
    assert!(verify_proof_with_zkp(&itp));
  }

  #[test]
  fn test_smod_transition_pair_form_wrong_output_fails() {
    let a = i256_bytes(-1000);
    let b = i256_bytes(30);
    let wrong_r = i256_bytes(-11);

    let proof = prove_instruction(opcode::SMOD, &[a, b], &[wrong_r]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::SMOD,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![wrong_r],
      semantic_proof: Some(proof),
    };
    assert!(!verify_proof(&itp));
    assert!(!verify_proof_with_rows(&itp));
    assert!(!verify_proof_with_zkp(&itp));
  }

  #[test]
  fn test_sdiv_smod_zero_divisor_output_zero() {
    let a = i256_bytes(-1000);
    let b = i256_bytes(0);
    let z = i256_bytes(0);

    let sdiv_proof = prove_instruction(opcode::SDIV, &[a, b], &[z]).unwrap();
    let sdiv_itp = InstructionTransitionProof {
      opcode: opcode::SDIV,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![z],
      semantic_proof: Some(sdiv_proof),
    };
    assert!(verify_proof(&sdiv_itp));
    assert!(verify_proof_with_zkp(&sdiv_itp));

    let smod_proof = prove_instruction(opcode::SMOD, &[a, b], &[z]).unwrap();
    let smod_itp = InstructionTransitionProof {
      opcode: opcode::SMOD,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![z],
      semantic_proof: Some(smod_proof),
    };
    assert!(verify_proof(&smod_itp));
    assert!(verify_proof_with_zkp(&smod_itp));
  }

  #[test]
  fn test_sdiv_int_min_overflow_case_fixed() {
    // EVM SDIV special case: INT_MIN / -1 = INT_MIN
    let mut int_min = [0u8; 32];
    int_min[0] = 0x80;
    let neg_one = [0xFF; 32];
    let expected = int_min;

    let proof = prove_instruction(opcode::SDIV, &[int_min, neg_one], &[expected]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::SDIV,
      pc: 0,
      stack_inputs: vec![int_min, neg_one],
      stack_outputs: vec![expected],
      semantic_proof: Some(proof),
    };
    assert!(verify_proof(&itp));
    assert!(verify_proof_with_rows(&itp));
    assert!(verify_proof_with_zkp(&itp));
  }

  #[test]
  fn test_smod_int_min_by_neg_one_is_zero() {
    // EVM SMOD special case companion: INT_MIN % -1 = 0
    let mut int_min = [0u8; 32];
    int_min[0] = 0x80;
    let neg_one = [0xFF; 32];
    let zero = [0u8; 32];

    let proof = prove_instruction(opcode::SMOD, &[int_min, neg_one], &[zero]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::SMOD,
      pc: 0,
      stack_inputs: vec![int_min, neg_one],
      stack_outputs: vec![zero],
      semantic_proof: Some(proof),
    };
    assert!(verify_proof(&itp));
    assert!(verify_proof_with_rows(&itp));
    assert!(verify_proof_with_zkp(&itp));
  }

  #[test]
  fn test_and_transition_success() {
    let a = [0xAA; 32];
    let b = [0x55; 32];
    let c = [0x00; 32];

    let proof = prove_instruction(opcode::AND, &[a, b], &[c]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::AND,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![c],
      semantic_proof: Some(proof),
    };
    assert!(verify_proof(&itp));
    assert!(verify_proof_with_rows(&itp));
    assert!(verify_proof_with_zkp(&itp));
  }

  #[test]
  fn test_or_transition_success() {
    let a = [0xAA; 32];
    let b = [0x55; 32];
    let c = [0xFF; 32];

    let proof = prove_instruction(opcode::OR, &[a, b], &[c]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::OR,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![c],
      semantic_proof: Some(proof),
    };
    assert!(verify_proof(&itp));
    assert!(verify_proof_with_rows(&itp));
    assert!(verify_proof_with_zkp(&itp));
  }

  #[test]
  fn test_xor_transition_success() {
    let a = [0xAA; 32];
    let b = [0x55; 32];
    let c = [0xFF; 32];

    let proof = prove_instruction(opcode::XOR, &[a, b], &[c]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::XOR,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![c],
      semantic_proof: Some(proof),
    };
    assert!(verify_proof(&itp));
    assert!(verify_proof_with_rows(&itp));
    assert!(verify_proof_with_zkp(&itp));
  }

  #[test]
  fn test_not_transition_success() {
    let a = [0xAA; 32];
    let c = [0x55; 32];

    let proof = prove_instruction(opcode::NOT, &[a], &[c]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::NOT,
      pc: 0,
      stack_inputs: vec![a],
      stack_outputs: vec![c],
      semantic_proof: Some(proof),
    };
    assert!(verify_proof(&itp));
    assert!(verify_proof_with_rows(&itp));
    assert!(verify_proof_with_zkp(&itp));
  }

  #[test]
  fn test_bitwise_wrong_output_fails() {
    let a = [0xAA; 32];
    let b = [0x55; 32];
    let wrong = [0x01; 32];

    let and_itp = InstructionTransitionProof {
      opcode: opcode::AND,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![wrong],
      semantic_proof: prove_instruction(opcode::AND, &[a, b], &[wrong]),
    };
    assert!(!verify_proof(&and_itp));
    assert!(!verify_proof_with_rows(&and_itp));
    assert!(!verify_proof_with_zkp(&and_itp));

    let not_itp = InstructionTransitionProof {
      opcode: opcode::NOT,
      pc: 0,
      stack_inputs: vec![a],
      stack_outputs: vec![wrong],
      semantic_proof: prove_instruction(opcode::NOT, &[a], &[wrong]),
    };
    assert!(!verify_proof(&not_itp));
    assert!(!verify_proof_with_rows(&not_itp));
    assert!(!verify_proof_with_zkp(&not_itp));
  }

  #[test]
  fn test_binary_arith_simple_transition() {
    let a = u256_bytes(1000);
    let b = u256_bytes(2000);
    let c = u256_bytes(3000);
    let proof = prove_instruction(opcode::ADD, &[a, b], &[c]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::ADD,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![c],
      semantic_proof: Some(proof),
    };
    assert!(verify_proof(&itp));
  }

  #[test]
  fn test_add_transition_zkp_passes_with_stack_lut_kernel() {
    let a = u256_bytes(1000);
    let b = u256_bytes(2000);
    let c = u256_bytes(3000);
    let proof = prove_instruction(opcode::ADD, &[a, b], &[c]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::ADD,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![c],
      semantic_proof: Some(proof),
    };
    assert!(verify_proof_with_zkp(&itp));
  }

  #[test]
  fn test_binary_arith_with_carry_transition() {
    // 0xFF...FF + 1 = 0 (overflow)
    let a = [0xFF; 32];
    let mut b = [0u8; 32];
    b[31] = 1;
    let c = [0u8; 32]; // wraps to 0
    let proof = prove_instruction(opcode::ADD, &[a, b], &[c]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::ADD,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![c],
      semantic_proof: Some(proof),
    };
    assert!(verify_proof(&itp));
  }

  #[test]
  fn test_binary_arith_large_values_transition() {
    // 0x80...00 + 0x80...00 = 0 (two large values)
    let mut a = [0u8; 32];
    a[0] = 0x80;
    let b = a;
    let c = [0u8; 32];
    let proof = prove_instruction(opcode::ADD, &[a, b], &[c]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::ADD,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![c],
      semantic_proof: Some(proof),
    };
    assert!(verify_proof(&itp));
  }

  #[test]
  fn test_binary_arith_wrong_output_fails() {
    let a = u256_bytes(100);
    let b = u256_bytes(200);
    let wrong = u256_bytes(999); // incorrect result
    let proof = prove_instruction(opcode::ADD, &[a, b], &[wrong]);
    // ByteAddEq should fail since the result is wrong
    assert!(proof.is_none() || {
      let itp = InstructionTransitionProof {
        opcode: opcode::ADD,
        pc: 0,
        stack_inputs: vec![a, b],
        stack_outputs: vec![wrong],
        semantic_proof: proof,
      };
      !verify_proof(&itp)
    });
  }

  #[test]
  fn test_structural_op_no_proof() {
    let itp = InstructionTransitionProof {
      opcode: opcode::POP,
      pc: 0,
      stack_inputs: vec![u256_bytes(42)],
      stack_outputs: vec![],
      semantic_proof: None,
    };
    assert!(verify_proof(&itp));
  }

  #[test]
  fn test_push_no_proof() {
    let itp = InstructionTransitionProof {
      opcode: opcode::PUSH1,
      pc: 0,
      stack_inputs: vec![],
      stack_outputs: vec![u256_bytes(42)],
      semantic_proof: None,
    };
    assert!(verify_proof(&itp));
  }

  #[test]
  fn test_structural_pop_wrong_arity_fails() {
    let itp = InstructionTransitionProof {
      opcode: opcode::POP,
      pc: 0,
      stack_inputs: vec![],
      stack_outputs: vec![],
      semantic_proof: None,
    };
    assert!(!verify_proof(&itp));
  }

  #[test]
  fn test_structural_push_wrong_arity_fails() {
    let itp = InstructionTransitionProof {
      opcode: opcode::PUSH1,
      pc: 0,
      stack_inputs: vec![u256_bytes(1)],
      stack_outputs: vec![u256_bytes(42)],
      semantic_proof: None,
    };
    assert!(!verify_proof(&itp));
  }

  #[test]
  fn test_prove_instruction_rejects_wrong_arity() {
    let a = u256_bytes(10);
    let b = u256_bytes(20);
    let c = u256_bytes(30);
    assert!(prove_instruction(opcode::ADD, &[a], &[c]).is_none());
    assert!(prove_instruction(opcode::ADD, &[a, b], &[]).is_none());
  }

  #[test]
  fn test_transition_semantic_and_proofrow_verification() {
    let a = u256_bytes(12345);
    let b = u256_bytes(67890);
    let c = u256_bytes(80235);

    let proof = prove_instruction(opcode::ADD, &[a, b], &[c]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::ADD,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![c],
      semantic_proof: Some(proof),
    };

    assert!(verify_proof_with_rows(&itp));
  }

  #[test]
  fn test_transition_semantic_or_proofrow_fail_on_wrong_output() {
    let a = u256_bytes(100);
    let b = u256_bytes(200);
    let wrong = u256_bytes(999);

    let maybe_proof = prove_instruction(opcode::ADD, &[a, b], &[wrong]);
    if let Some(proof) = maybe_proof {
      let itp = InstructionTransitionProof {
        opcode: opcode::ADD,
        pc: 0,
        stack_inputs: vec![a, b],
        stack_outputs: vec![wrong],
        semantic_proof: Some(proof),
      };
      assert!(!verify_proof_with_rows(&itp));
    }
  }

  #[test]
  fn test_transaction_arithmetic_chain() {
    let a = u256_bytes(100);
    let b = u256_bytes(200);
    let c = u256_bytes(300);
    let d = u256_bytes(50);
    let e = u256_bytes(350);

    // Step 1: ADD(100, 200) = 300
    let p1 = prove_instruction(opcode::ADD, &[a, b], &[c]).unwrap();
    let step1 = InstructionTransitionProof {
      opcode: opcode::ADD, pc: 0,
      stack_inputs: vec![a, b], stack_outputs: vec![c],
      semantic_proof: Some(p1),
    };

    // Step 2: ADD(300, 50) = 350
    let p2 = prove_instruction(opcode::ADD, &[c, d], &[e]).unwrap();
    let step2 = InstructionTransitionProof {
      opcode: opcode::ADD, pc: 1,
      stack_inputs: vec![c, d], stack_outputs: vec![e],
      semantic_proof: Some(p2),
    };

    let tx = TransactionProof { steps: vec![step1, step2] };
    for step in &tx.steps {
      assert!(verify_proof(step));
    }
  }
}
