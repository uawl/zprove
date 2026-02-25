// moved from src/transition.rs

#[cfg(test)]
mod tests {
  use zprove_core::memory_proof::{compute_memory_root, verify_memory_access_commitments};
  use zprove_core::semantic_proof::{infer_proof, prove_add, Proof, Term};
  use zprove_core::transition::*;
  use revm::bytecode::opcode;

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
  fn test_opcode_target_wff_matches_inferred_proof_for_supported_ops() {
    let add_a = u256_bytes(1234);
    let add_b = u256_bytes(5678);
    let add_c = u256_bytes(6912);

    let sub_a = u256_bytes(3000);
    let sub_b = u256_bytes(2000);
    let sub_c = u256_bytes(1000);

    let mul_a = u256_bytes(123);
    let mul_b = u256_bytes(45);
    let mul_c = u256_bytes(5535);

    let div_a = u256_bytes(1000);
    let div_b = u256_bytes(30);
    let div_q = u256_bytes(33);

    let mod_a = u256_bytes(1000);
    let mod_b = u256_bytes(30);
    let mod_r = u256_bytes(10);

    let sdiv_a = i256_bytes(-100);
    let sdiv_b = i256_bytes(30);
    let sdiv_q = i256_bytes(-3);

    let smod_a = i256_bytes(-100);
    let smod_b = i256_bytes(30);
    let smod_r = i256_bytes(-10);

    let mut bit_a = [0u8; 32];
    let mut bit_b = [0u8; 32];
    bit_a[31] = 0b1010_1100;
    bit_b[31] = 0b0110_1010;
    let mut and_c = [0u8; 32];
    and_c[31] = bit_a[31] & bit_b[31];
    let mut or_c = [0u8; 32];
    or_c[31] = bit_a[31] | bit_b[31];
    let mut xor_c = [0u8; 32];
    xor_c[31] = bit_a[31] ^ bit_b[31];
    let mut not_c = [0u8; 32];
    not_c[31] = !bit_a[31];
    for item in not_c.iter_mut().take(31) {
      *item = 0xFF;
    }

    let cases = vec![
      (opcode::ADD, vec![add_a, add_b], vec![add_c]),
      (opcode::SUB, vec![sub_a, sub_b], vec![sub_c]),
      (opcode::MUL, vec![mul_a, mul_b], vec![mul_c]),
      (opcode::DIV, vec![div_a, div_b], vec![div_q]),
      (opcode::MOD, vec![mod_a, mod_b], vec![mod_r]),
      (opcode::SDIV, vec![sdiv_a, sdiv_b], vec![sdiv_q]),
      (opcode::SMOD, vec![smod_a, smod_b], vec![smod_r]),
      (opcode::AND, vec![bit_a, bit_b], vec![and_c]),
      (opcode::OR, vec![bit_a, bit_b], vec![or_c]),
      (opcode::XOR, vec![bit_a, bit_b], vec![xor_c]),
      (opcode::NOT, vec![bit_a], vec![not_c]),
    ];

    for (op, inputs, outputs) in cases {
      let proof = prove_instruction(op, &inputs, &outputs).expect("supported opcode must prove");
      let inferred = infer_proof(&proof).expect("inference must succeed for generated proof");
      let expected =
        wff_instruction(op, &inputs, &outputs).expect("supported opcode must have target WFF");

      assert_eq!(
        inferred, expected,
        "opcode 0x{op:02x} target WFF must match inferred WFF"
      );
    }
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

    assert!(!verify_instruction_zk_receipt(
      &wrong_sp_statement,
      &receipt
    ));
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

    assert!(!verify_instruction_zk_receipt(
      &wrong_opcode_statement,
      &receipt
    ));
  }

  #[test]
  fn test_receipt_rejects_tampered_receipt_opcode() {
    let a = u256_bytes(4321);
    let b = u256_bytes(1234);
    let c = u256_bytes(5555);

    let semantic = prove_instruction(opcode::ADD, &[a, b], &[c]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::ADD,
      pc: 17,
      stack_inputs: vec![a, b],
      stack_outputs: vec![c],
      semantic_proof: Some(semantic),
    };

    let statement = InstructionTransitionStatement {
      opcode: opcode::ADD,
      s_i: VmState {
        opcode: opcode::ADD,
        pc: 17,
        sp: 2,
        stack: vec![a, b],
        memory_root: [0u8; 32],
      },
      s_next: VmState {
        opcode: opcode::ADD,
        pc: 18,
        sp: 1,
        stack: vec![c],
        memory_root: [0u8; 32],
      },
      accesses: Vec::new(),
    };

    let mut receipt = prove_instruction_zk_receipt(&itp).expect("receipt proving should succeed");
    receipt.opcode = opcode::MUL;

    assert!(!verify_instruction_zk_receipt(&statement, &receipt));
  }

  #[test]
  fn test_receipt_rejects_tampered_expected_wff() {
    let a = u256_bytes(4321);
    let b = u256_bytes(1234);
    let c = u256_bytes(5555);
    let mut wrong_c = c;
    wrong_c[31] ^= 1;

    let semantic = prove_instruction(opcode::ADD, &[a, b], &[c]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::ADD,
      pc: 18,
      stack_inputs: vec![a, b],
      stack_outputs: vec![c],
      semantic_proof: Some(semantic),
    };

    let statement = InstructionTransitionStatement {
      opcode: opcode::ADD,
      s_i: VmState {
        opcode: opcode::ADD,
        pc: 18,
        sp: 2,
        stack: vec![a, b],
        memory_root: [0u8; 32],
      },
      s_next: VmState {
        opcode: opcode::ADD,
        pc: 19,
        sp: 1,
        stack: vec![c],
        memory_root: [0u8; 32],
      },
      accesses: Vec::new(),
    };

    let mut receipt = prove_instruction_zk_receipt(&itp).expect("receipt proving should succeed");
    receipt.expected_wff =
      wff_instruction(opcode::ADD, &[a, b], &[wrong_c]).expect("wff build should succeed");

    assert!(!verify_instruction_zk_receipt(&statement, &receipt));
  }

  #[test]
  fn test_receipt_splice_attack_stack_ir_rejected() {
    let a1 = u256_bytes(4321);
    let b1 = u256_bytes(1234);
    let c1 = u256_bytes(5555);
    let a2 = u256_bytes(1111);
    let b2 = u256_bytes(2222);
    let c2 = u256_bytes(3333);

    let semantic1 = prove_instruction(opcode::ADD, &[a1, b1], &[c1]).unwrap();
    let semantic2 = prove_instruction(opcode::ADD, &[a2, b2], &[c2]).unwrap();

    let itp1 = InstructionTransitionProof {
      opcode: opcode::ADD,
      pc: 19,
      stack_inputs: vec![a1, b1],
      stack_outputs: vec![c1],
      semantic_proof: Some(semantic1),
    };
    let itp2 = InstructionTransitionProof {
      opcode: opcode::ADD,
      pc: 20,
      stack_inputs: vec![a2, b2],
      stack_outputs: vec![c2],
      semantic_proof: Some(semantic2),
    };

    let statement1 = InstructionTransitionStatement {
      opcode: opcode::ADD,
      s_i: VmState {
        opcode: opcode::ADD,
        pc: 19,
        sp: 2,
        stack: vec![a1, b1],
        memory_root: [0u8; 32],
      },
      s_next: VmState {
        opcode: opcode::ADD,
        pc: 20,
        sp: 1,
        stack: vec![c1],
        memory_root: [0u8; 32],
      },
      accesses: Vec::new(),
    };

    let mut receipt1 = prove_instruction_zk_receipt(&itp1).expect("receipt proving should succeed");
    let mut receipt2 = prove_instruction_zk_receipt(&itp2).expect("receipt proving should succeed");
    std::mem::swap(&mut receipt1.stack_ir_proof, &mut receipt2.stack_ir_proof);

    assert!(!verify_instruction_zk_receipt(&statement1, &receipt1));
  }

  #[test]
  fn test_receipt_splice_attack_lut_rejected() {
    let a1 = u256_bytes(4321);
    let b1 = u256_bytes(1234);
    let c1 = u256_bytes(5555);
    let a2 = u256_bytes(1111);
    let b2 = u256_bytes(2222);
    let c2 = u256_bytes(3333);

    let semantic1 = prove_instruction(opcode::ADD, &[a1, b1], &[c1]).unwrap();
    let semantic2 = prove_instruction(opcode::ADD, &[a2, b2], &[c2]).unwrap();

    let itp1 = InstructionTransitionProof {
      opcode: opcode::ADD,
      pc: 21,
      stack_inputs: vec![a1, b1],
      stack_outputs: vec![c1],
      semantic_proof: Some(semantic1),
    };
    let itp2 = InstructionTransitionProof {
      opcode: opcode::ADD,
      pc: 22,
      stack_inputs: vec![a2, b2],
      stack_outputs: vec![c2],
      semantic_proof: Some(semantic2),
    };

    let statement1 = InstructionTransitionStatement {
      opcode: opcode::ADD,
      s_i: VmState {
        opcode: opcode::ADD,
        pc: 21,
        sp: 2,
        stack: vec![a1, b1],
        memory_root: [0u8; 32],
      },
      s_next: VmState {
        opcode: opcode::ADD,
        pc: 22,
        sp: 1,
        stack: vec![c1],
        memory_root: [0u8; 32],
      },
      accesses: Vec::new(),
    };

    let mut receipt1 = prove_instruction_zk_receipt(&itp1).expect("receipt proving should succeed");
    let mut receipt2 = prove_instruction_zk_receipt(&itp2).expect("receipt proving should succeed");
    std::mem::swap(&mut receipt1.lut_kernel_proof, &mut receipt2.lut_kernel_proof);

    assert!(!verify_instruction_zk_receipt(&statement1, &receipt1));
  }

  #[test]
  fn test_receipt_splice_attack_wff_match_rejected() {
    let a1 = u256_bytes(4321);
    let b1 = u256_bytes(1234);
    let c1 = u256_bytes(5555);
    let a2 = u256_bytes(1111);
    let b2 = u256_bytes(2222);
    let c2 = u256_bytes(3333);

    let semantic1 = prove_instruction(opcode::ADD, &[a1, b1], &[c1]).unwrap();
    let semantic2 = prove_instruction(opcode::ADD, &[a2, b2], &[c2]).unwrap();

    let itp1 = InstructionTransitionProof {
      opcode: opcode::ADD,
      pc: 23,
      stack_inputs: vec![a1, b1],
      stack_outputs: vec![c1],
      semantic_proof: Some(semantic1),
    };
    let itp2 = InstructionTransitionProof {
      opcode: opcode::ADD,
      pc: 24,
      stack_inputs: vec![a2, b2],
      stack_outputs: vec![c2],
      semantic_proof: Some(semantic2),
    };

    let statement1 = InstructionTransitionStatement {
      opcode: opcode::ADD,
      s_i: VmState {
        opcode: opcode::ADD,
        pc: 23,
        sp: 2,
        stack: vec![a1, b1],
        memory_root: [0u8; 32],
      },
      s_next: VmState {
        opcode: opcode::ADD,
        pc: 24,
        sp: 1,
        stack: vec![c1],
        memory_root: [0u8; 32],
      },
      accesses: Vec::new(),
    };

    let mut receipt1 = prove_instruction_zk_receipt(&itp1).expect("receipt proving should succeed");
    let mut receipt2 = prove_instruction_zk_receipt(&itp2).expect("receipt proving should succeed");
    std::mem::swap(&mut receipt1.wff_match_proof, &mut receipt2.wff_match_proof);

    assert!(!verify_instruction_zk_receipt(&statement1, &receipt1));
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
    assert!(
      proof.is_none() || {
        let itp = InstructionTransitionProof {
          opcode: opcode::ADD,
          pc: 0,
          stack_inputs: vec![a, b],
          stack_outputs: vec![wrong],
          semantic_proof: proof,
        };
        !verify_proof(&itp)
      }
    );
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
      opcode: opcode::ADD,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![c],
      semantic_proof: Some(p1),
    };

    // Step 2: ADD(300, 50) = 350
    let p2 = prove_instruction(opcode::ADD, &[c, d], &[e]).unwrap();
    let step2 = InstructionTransitionProof {
      opcode: opcode::ADD,
      pc: 1,
      stack_inputs: vec![c, d],
      stack_outputs: vec![e],
      semantic_proof: Some(p2),
    };

    let tx = TransactionProof {
      steps: vec![step1, step2],
    };
    for step in &tx.steps {
      assert!(verify_proof(step));
    }
  }

  // ============================================================
  // Comparison / equality opcode tests
  // ============================================================

  fn bool_word(v: bool) -> [u8; 32] {
    let mut w = [0u8; 32];
    w[31] = v as u8;
    w
  }

  #[test]
  fn test_eq_true_when_equal() {
    let a = u256_bytes(12345);
    let proof = prove_instruction(opcode::EQ, &[a, a], &[bool_word(true)]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::EQ,
      pc: 0,
      stack_inputs: vec![a, a],
      stack_outputs: vec![bool_word(true)],
      semantic_proof: Some(proof),
    };
    assert!(verify_proof(&itp));
    assert!(verify_proof_with_rows(&itp));
  }

  #[test]
  fn test_eq_false_when_different() {
    let a = u256_bytes(100);
    let b = u256_bytes(200);
    let proof = prove_instruction(opcode::EQ, &[a, b], &[bool_word(false)]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::EQ,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![bool_word(false)],
      semantic_proof: Some(proof),
    };
    assert!(verify_proof(&itp));
    assert!(verify_proof_with_rows(&itp));
  }

  #[test]
  fn test_eq_rejects_wrong_output() {
    let a = u256_bytes(42);
    // Claim EQ(a, a) = 0  — should fail
    let proof = prove_instruction(opcode::EQ, &[a, a], &[bool_word(false)]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::EQ,
      pc: 0,
      stack_inputs: vec![a, a],
      stack_outputs: vec![bool_word(false)],
      semantic_proof: Some(proof),
    };
    assert!(!verify_proof(&itp));
  }

  #[test]
  fn test_iszero_true_for_zero() {
    let zero = u256_bytes(0);
    let proof = prove_instruction(opcode::ISZERO, &[zero], &[bool_word(true)]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::ISZERO,
      pc: 0,
      stack_inputs: vec![zero],
      stack_outputs: vec![bool_word(true)],
      semantic_proof: Some(proof),
    };
    assert!(verify_proof(&itp));
    assert!(verify_proof_with_rows(&itp));
  }

  #[test]
  fn test_iszero_false_for_nonzero() {
    let val = u256_bytes(1);
    let proof = prove_instruction(opcode::ISZERO, &[val], &[bool_word(false)]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::ISZERO,
      pc: 0,
      stack_inputs: vec![val],
      stack_outputs: vec![bool_word(false)],
      semantic_proof: Some(proof),
    };
    assert!(verify_proof(&itp));
    assert!(verify_proof_with_rows(&itp));
  }

  #[test]
  fn test_lt_true_when_less() {
    let a = u256_bytes(5);
    let b = u256_bytes(10);
    let proof = prove_instruction(opcode::LT, &[a, b], &[bool_word(true)]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::LT,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![bool_word(true)],
      semantic_proof: Some(proof),
    };
    assert!(verify_proof(&itp));
    assert!(verify_proof_with_rows(&itp));
  }

  #[test]
  fn test_lt_false_when_greater_or_equal() {
    let a = u256_bytes(10);
    let b = u256_bytes(5);
    let proof = prove_instruction(opcode::LT, &[a, b], &[bool_word(false)]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::LT,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![bool_word(false)],
      semantic_proof: Some(proof),
    };
    assert!(verify_proof(&itp));
    assert!(verify_proof_with_rows(&itp));
  }

  #[test]
  fn test_lt_false_when_equal() {
    let a = u256_bytes(7);
    let proof = prove_instruction(opcode::LT, &[a, a], &[bool_word(false)]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::LT,
      pc: 0,
      stack_inputs: vec![a, a],
      stack_outputs: vec![bool_word(false)],
      semantic_proof: Some(proof),
    };
    assert!(verify_proof(&itp));
  }

  #[test]
  fn test_lt_rejects_wrong_output() {
    let a = u256_bytes(5);
    let b = u256_bytes(10);
    // Claim LT(5, 10) = 0 — wrong
    let proof = prove_instruction(opcode::LT, &[a, b], &[bool_word(false)]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::LT,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![bool_word(false)],
      semantic_proof: Some(proof),
    };
    assert!(!verify_proof(&itp));
  }

  #[test]
  fn test_gt_true_when_greater() {
    let a = u256_bytes(10);
    let b = u256_bytes(3);
    let proof = prove_instruction(opcode::GT, &[a, b], &[bool_word(true)]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::GT,
      pc: 0,
      stack_inputs: vec![a, b],
      stack_outputs: vec![bool_word(true)],
      semantic_proof: Some(proof),
    };
    assert!(verify_proof(&itp));
    assert!(verify_proof_with_rows(&itp));
  }

  #[test]
  fn test_slt_negative_less_than_positive() {
    let neg = i256_bytes(-1);
    let pos = u256_bytes(1);
    let proof = prove_instruction(opcode::SLT, &[neg, pos], &[bool_word(true)]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::SLT,
      pc: 0,
      stack_inputs: vec![neg, pos],
      stack_outputs: vec![bool_word(true)],
      semantic_proof: Some(proof),
    };
    assert!(verify_proof(&itp));
    assert!(verify_proof_with_rows(&itp));
  }

  #[test]
  fn test_slt_positive_not_less_than_negative() {
    let pos = u256_bytes(1);
    let neg = i256_bytes(-1);
    let proof = prove_instruction(opcode::SLT, &[pos, neg], &[bool_word(false)]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::SLT,
      pc: 0,
      stack_inputs: vec![pos, neg],
      stack_outputs: vec![bool_word(false)],
      semantic_proof: Some(proof),
    };
    assert!(verify_proof(&itp));
    assert!(verify_proof_with_rows(&itp));
  }

  #[test]
  fn test_sgt_positive_greater_than_negative() {
    let pos = u256_bytes(1);
    let neg = i256_bytes(-1);
    let proof = prove_instruction(opcode::SGT, &[pos, neg], &[bool_word(true)]).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::SGT,
      pc: 0,
      stack_inputs: vec![pos, neg],
      stack_outputs: vec![bool_word(true)],
      semantic_proof: Some(proof),
    };
    assert!(verify_proof(&itp));
    assert!(verify_proof_with_rows(&itp));
  }

  #[test]
  fn test_comparison_opcodes_in_wff_target_match() {
    // Verify infer_proof(prove_*(..)) == wff_*(..)) for all new ops
    let small = u256_bytes(5);
    let large = u256_bytes(100);
    let neg = i256_bytes(-10);
    let zero = u256_bytes(0);

    let cases: &[(u8, Vec<[u8; 32]>, [u8; 32])] = &[
      (opcode::EQ,     vec![small, small], bool_word(true)),
      (opcode::EQ,     vec![small, large], bool_word(false)),
      (opcode::ISZERO, vec![zero],         bool_word(true)),
      (opcode::ISZERO, vec![small],        bool_word(false)),
      (opcode::LT,     vec![small, large], bool_word(true)),
      (opcode::LT,     vec![large, small], bool_word(false)),
      (opcode::GT,     vec![large, small], bool_word(true)),
      (opcode::GT,     vec![small, large], bool_word(false)),
      (opcode::SLT,    vec![neg, small],   bool_word(true)),
      (opcode::SLT,    vec![small, neg],   bool_word(false)),
      (opcode::SGT,    vec![small, neg],   bool_word(true)),
      (opcode::SGT,    vec![neg, small],   bool_word(false)),
    ];

    for (op, inputs, output) in cases {
      let proof = prove_instruction(*op, inputs, &[*output])
        .unwrap_or_else(|| panic!("prove_instruction failed for opcode 0x{op:02x}"));
      let inferred = infer_proof(&proof)
        .unwrap_or_else(|e| panic!("infer_proof failed for opcode 0x{op:02x}: {e}"));
      let expected = wff_instruction(*op, inputs, &[*output])
        .unwrap_or_else(|| panic!("wff_instruction returned None for opcode 0x{op:02x}"));
      assert_eq!(
        inferred, expected,
        "WFF mismatch for opcode 0x{op:02x}"
      );
    }
  }
}
