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
  Proof, WFF, compile_proof, infer_proof, prove_add, verify_compiled, wff_add
};
use crate::zk_proof::{
  prove_and_verify_expected_wff_match_stark,
  prove_and_verify_inferred_wff_stark,
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
pub fn prove_instruction(op: u8, inputs: &[[u8; 32]], outputs: &[[u8; 32]]) -> Option<Proof> {
  match op {
    opcode::ADD => Some(prove_add(&inputs[0], &inputs[1], &outputs[0])),
    opcode::SUB
    | opcode::MUL
    | opcode::DIV
    | opcode::SDIV
    | opcode::MOD
    | opcode::SMOD
    | opcode::ADDMOD
    | opcode::MULMOD
    | opcode::EXP
    | opcode::LT
    | opcode::GT
    | opcode::SLT
    | opcode::SGT
    | opcode::EQ
    | opcode::ISZERO
    | opcode::AND
    | opcode::OR
    | opcode::XOR
    | opcode::NOT => todo!("semantic proof path not implemented yet for opcode 0x{op:02x}"),
    // Structural ops: no semantic proof needed
    _ => None,
  }
}

// ============================================================
// WFF generation
// ============================================================

pub fn wff_instruction(op: u8, inputs: &[[u8; 32]], outputs: &[[u8; 32]]) -> Option<WFF> {
  match op {
    opcode::ADD => Some(wff_add(&inputs[0], &inputs[1], &outputs[0])),
    opcode::SUB
    | opcode::MUL
    | opcode::DIV
    | opcode::SDIV
    | opcode::MOD
    | opcode::SMOD
    | opcode::ADDMOD
    | opcode::MULMOD
    | opcode::EXP
    | opcode::LT
    | opcode::GT
    | opcode::SLT
    | opcode::SGT
    | opcode::EQ
    | opcode::ISZERO
    | opcode::AND
    | opcode::OR
    | opcode::XOR
    | opcode::NOT => todo!("WFF path not implemented yet for opcode 0x{op:02x}"),
    // Structural ops: no semantic proof needed
    _ => None,
  }
}

// ============================================================
// Verification (for testing)
// ============================================================

pub fn verify_proof(proof: &InstructionTransitionProof) -> bool {
  let expected_wff = wff_instruction(proof.opcode, &proof.stack_inputs, &proof.stack_outputs);
  match (&proof.semantic_proof, expected_wff) {
    (Some(proof), Some(wff)) => {
      let Ok(wff_result) = infer_proof(proof) else {
        return false;
      };
      wff == wff_result
    },
    (None, None) => true, // No proof needed for structural ops
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
  match &proof.semantic_proof {
    Some(semantic_proof) => {
      let rows = compile_proof(semantic_proof);
      let infer_wff_ok = prove_and_verify_inferred_wff_stark(&rows);

      let expected_wff_ok = match proof.opcode {
        opcode::ADD => wff_instruction(proof.opcode, &proof.stack_inputs, &proof.stack_outputs)
          .map(|public_wff| prove_and_verify_expected_wff_match_stark(semantic_proof, &public_wff))
          .unwrap_or(false),
        opcode::SUB
        | opcode::MUL
        | opcode::DIV
        | opcode::SDIV
        | opcode::MOD
        | opcode::SMOD
        | opcode::ADDMOD
        | opcode::MULMOD
        | opcode::EXP
        | opcode::LT
        | opcode::GT
        | opcode::SLT
        | opcode::SGT
        | opcode::EQ
        | opcode::ISZERO
        | opcode::AND
        | opcode::OR
        | opcode::XOR
        | opcode::NOT => false,
        _ => true,
      };

      infer_wff_ok && expected_wff_ok
    }
    None => true,
  }
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
