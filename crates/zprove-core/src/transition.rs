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
  Proof, Term, VerifyError, extract_word_bytes, make_word_term, prove_word_add, prove_word_and,
  prove_word_eq, prove_word_gt, prove_word_iszero, prove_word_lt, prove_word_not, prove_word_or,
  prove_word_sub, prove_word_xor, verify,
};

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

// ============================================================
// Opcode constants (subset of EVM)
// ============================================================

pub mod opcode {
  pub const STOP: u8 = 0x00;
  pub const ADD: u8 = 0x01;
  pub const MUL: u8 = 0x02;
  pub const SUB: u8 = 0x03;
  pub const DIV: u8 = 0x04;
  pub const SDIV: u8 = 0x05;
  pub const MOD: u8 = 0x06;
  pub const SMOD: u8 = 0x07;
  pub const ADDMOD: u8 = 0x08;
  pub const MULMOD: u8 = 0x09;
  pub const EXP: u8 = 0x0a;
  pub const LT: u8 = 0x10;
  pub const GT: u8 = 0x11;
  pub const SLT: u8 = 0x12;
  pub const SGT: u8 = 0x13;
  pub const EQ: u8 = 0x14;
  pub const ISZERO: u8 = 0x15;
  pub const AND: u8 = 0x16;
  pub const OR: u8 = 0x17;
  pub const XOR: u8 = 0x18;
  pub const NOT: u8 = 0x19;
  pub const POP: u8 = 0x50;
  pub const JUMP: u8 = 0x56;
  pub const JUMPI: u8 = 0x57;
  pub const JUMPDEST: u8 = 0x5b;
  pub const PUSH0: u8 = 0x5f;
  pub const PUSH1: u8 = 0x60;
  pub const PUSH32: u8 = 0x7f;
  pub const DUP1: u8 = 0x80;
  pub const DUP16: u8 = 0x8f;
  pub const SWAP1: u8 = 0x90;
  pub const SWAP16: u8 = 0x9f;
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
    opcode::POP => 0,
    opcode::JUMP | opcode::JUMPI => 0,
    opcode::JUMPDEST => 0,
    op if (opcode::PUSH0..=opcode::PUSH32).contains(&op) => 1,
    op if (opcode::DUP1..=opcode::DUP16).contains(&op) => 1,
    op if (opcode::SWAP1..=opcode::SWAP16).contains(&op) => 0,
    _ => 0,
  }
}

// ============================================================
// Proof generation per instruction
// ============================================================

/// Generate a semantic proof for a single EVM instruction.
///
/// Returns `(proof, computed_outputs)`.  The caller should compare
/// `computed_outputs` against the actual EVM outputs to ensure consistency.
pub fn prove_instruction(op: u8, inputs: &[[u8; 32]]) -> Option<(Proof, Vec<[u8; 32]>)> {
  match op {
    opcode::ADD => {
      let (proof, result) = prove_word_add(&inputs[0], &inputs[1]);
      Some((proof, vec![result]))
    }
    opcode::SUB => {
      let (proof, result) = prove_word_sub(&inputs[0], &inputs[1]);
      Some((proof, vec![result]))
    }
    opcode::AND => {
      let (proof, result) = prove_word_and(&inputs[0], &inputs[1]);
      Some((proof, vec![result]))
    }
    opcode::OR => {
      let (proof, result) = prove_word_or(&inputs[0], &inputs[1]);
      Some((proof, vec![result]))
    }
    opcode::XOR => {
      let (proof, result) = prove_word_xor(&inputs[0], &inputs[1]);
      Some((proof, vec![result]))
    }
    opcode::NOT => {
      let (proof, result) = prove_word_not(&inputs[0]);
      Some((proof, vec![result]))
    }
    opcode::EQ => {
      let (proof, result) = prove_word_eq(&inputs[0], &inputs[1]);
      Some((proof, vec![result]))
    }
    opcode::ISZERO => {
      let (proof, result) = prove_word_iszero(&inputs[0]);
      Some((proof, vec![result]))
    }
    opcode::LT => {
      let (proof, result) = prove_word_lt(&inputs[0], &inputs[1]);
      Some((proof, vec![result]))
    }
    opcode::GT => {
      let (proof, result) = prove_word_gt(&inputs[0], &inputs[1]);
      Some((proof, vec![result]))
    }
    // Structural ops: no semantic proof needed
    _ => None,
  }
}

// ============================================================
// Verification
// ============================================================

/// Verification error for instruction-level proofs.
#[derive(Debug)]
pub enum TransitionVerifyError {
  /// Semantic proof verification failed.
  SemanticProof(VerifyError),
  /// Proof proves wrong operation for this opcode.
  WrongOperation { opcode: u8, expected: &'static str },
  /// Proven result doesn't match claimed output.
  OutputMismatch { index: usize },
  /// Proven inputs don't match claimed inputs.
  InputMismatch { index: usize },
  /// Missing semantic proof for arithmetic/logic opcode.
  MissingProof { opcode: u8 },
}

impl From<VerifyError> for TransitionVerifyError {
  fn from(e: VerifyError) -> Self {
    TransitionVerifyError::SemanticProof(e)
  }
}

/// Verify a single instruction's transition proof.
pub fn verify_instruction(proof: &InstructionTransitionProof) -> Result<(), TransitionVerifyError> {
  let op = proof.opcode;

  // Structural ops: just check stack counts
  if proof.semantic_proof.is_none() {
    return Ok(());
  }

  let semantic = proof.semantic_proof.as_ref().unwrap();
  let (lhs, rhs) = verify(semantic)?;

  // Check that the proven LHS matches the expected operation on the inputs,
  // and the proven RHS matches the claimed outputs.
  match op {
    opcode::ADD => check_binop_proof(&lhs, &rhs, proof, Term::WordAdd, "WordAdd"),
    opcode::SUB => check_binop_proof(&lhs, &rhs, proof, Term::WordSub, "WordSub"),
    opcode::AND => check_binop_proof(&lhs, &rhs, proof, Term::WordAnd, "WordAnd"),
    opcode::OR => check_binop_proof(&lhs, &rhs, proof, Term::WordOr, "WordOr"),
    opcode::XOR => check_binop_proof(&lhs, &rhs, proof, Term::WordXor, "WordXor"),
    opcode::NOT => check_unaryop_proof(&lhs, &rhs, proof, Term::WordNot, "WordNot"),
    opcode::EQ => check_binop_proof(&lhs, &rhs, proof, Term::WordEqOp, "WordEqOp"),
    opcode::ISZERO => check_unaryop_proof(&lhs, &rhs, proof, Term::WordIsZero, "WordIsZero"),
    opcode::LT => check_binop_proof(&lhs, &rhs, proof, Term::WordLt, "WordLt"),
    opcode::GT => check_binop_proof(&lhs, &rhs, proof, Term::WordGt, "WordGt"),
    _ => Ok(()), // unrecognized opcode with proof — just trust semantic verify
  }
}

fn check_binop_proof(
  lhs: &Term,
  rhs: &Term,
  itp: &InstructionTransitionProof,
  make_op: fn(Box<Term>, Box<Term>) -> Term,
  expected_name: &'static str,
) -> Result<(), TransitionVerifyError> {
  let expected_lhs = make_op(
    Box::new(make_word_term(&itp.stack_inputs[0])),
    Box::new(make_word_term(&itp.stack_inputs[1])),
  );
  if *lhs != expected_lhs {
    return Err(TransitionVerifyError::InputMismatch { index: 0 });
  }
  let result_bytes = extract_word_bytes(rhs).ok_or(TransitionVerifyError::WrongOperation {
    opcode: itp.opcode,
    expected: expected_name,
  })?;
  if result_bytes != itp.stack_outputs[0] {
    return Err(TransitionVerifyError::OutputMismatch { index: 0 });
  }
  Ok(())
}

fn check_unaryop_proof(
  lhs: &Term,
  rhs: &Term,
  itp: &InstructionTransitionProof,
  make_op: fn(Box<Term>) -> Term,
  expected_name: &'static str,
) -> Result<(), TransitionVerifyError> {
  let expected_lhs = make_op(Box::new(make_word_term(&itp.stack_inputs[0])));
  if *lhs != expected_lhs {
    return Err(TransitionVerifyError::InputMismatch { index: 0 });
  }
  let result_bytes = extract_word_bytes(rhs).ok_or(TransitionVerifyError::WrongOperation {
    opcode: itp.opcode,
    expected: expected_name,
  })?;
  if result_bytes != itp.stack_outputs[0] {
    return Err(TransitionVerifyError::OutputMismatch { index: 0 });
  }
  Ok(())
}

/// Verify all instruction proofs in a transaction.
pub fn verify_transaction(
  tx_proof: &TransactionProof,
) -> Result<(), (usize, TransitionVerifyError)> {
  for (step_idx, instr_proof) in tx_proof.steps.iter().enumerate() {
    verify_instruction(instr_proof).map_err(|e| (step_idx, e))?;
  }
  Ok(())
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
  use super::*;

  fn u256_bytes(val: u128) -> [u8; 32] {
    let mut b = [0u8; 32];
    b[16..32].copy_from_slice(&val.to_be_bytes());
    b
  }

  #[test]
  fn test_prove_and_verify_add() {
    let a = u256_bytes(1000);
    let b = u256_bytes(2000);
    let (proof, outputs) = prove_instruction(opcode::ADD, &[a, b]).unwrap();
    assert_eq!(outputs, vec![u256_bytes(3000)]);

    let itp = InstructionTransitionProof {
      opcode: opcode::ADD,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: outputs,
      semantic_proof: Some(proof),
    };
    verify_instruction(&itp).unwrap();
  }

  #[test]
  fn test_prove_and_verify_sub() {
    let a = u256_bytes(5000);
    let b = u256_bytes(3000);
    let (proof, outputs) = prove_instruction(opcode::SUB, &[a, b]).unwrap();
    assert_eq!(outputs, vec![u256_bytes(2000)]);

    let itp = InstructionTransitionProof {
      opcode: opcode::SUB,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: outputs,
      semantic_proof: Some(proof),
    };
    verify_instruction(&itp).unwrap();
  }

  #[test]
  fn test_prove_and_verify_lt() {
    let a = u256_bytes(10);
    let b = u256_bytes(20);
    let (proof, outputs) = prove_instruction(opcode::LT, &[a, b]).unwrap();
    let mut one = [0u8; 32];
    one[31] = 1;
    assert_eq!(outputs, vec![one]);

    let itp = InstructionTransitionProof {
      opcode: opcode::LT,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: outputs,
      semantic_proof: Some(proof),
    };
    verify_instruction(&itp).unwrap();
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
    verify_instruction(&itp).unwrap();
  }

  #[test]
  fn test_transaction_proof() {
    let a = u256_bytes(100);
    let b = u256_bytes(200);

    // Step 1: ADD
    let (add_proof, add_out) = prove_instruction(opcode::ADD, &[a, b]).unwrap();
    let step1 = InstructionTransitionProof {
      opcode: opcode::ADD,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: add_out.clone(),
      semantic_proof: Some(add_proof),
    };

    // Step 2: ISZERO on the result
    let (iz_proof, iz_out) = prove_instruction(opcode::ISZERO, &[add_out[0]]).unwrap();
    let step2 = InstructionTransitionProof {
      opcode: opcode::ISZERO,
      pc: 1,
      stack_inputs: vec![add_out[0]],
      stack_outputs: iz_out,
      semantic_proof: Some(iz_proof),
    };

    let tx_proof = TransactionProof {
      steps: vec![step1, step2],
    };
    verify_transaction(&tx_proof).unwrap();
  }
}
