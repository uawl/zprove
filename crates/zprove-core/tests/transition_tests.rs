// moved from src/transition.rs

#[cfg(test)]
mod tests {
  use revm::bytecode::opcode;
  use zprove_core::semantic_proof::{Proof, Term, infer_proof, prove_add};
  use zprove_core::transition::*;

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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
    };
    let itp2 = InstructionTransitionProof {
      opcode: opcode::ADD,
      pc: 20,
      stack_inputs: vec![a2, b2],
      stack_outputs: vec![c2],
      semantic_proof: Some(semantic2),
      memory_claims: vec![],
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
      memory_claims: vec![],
    };
    let itp2 = InstructionTransitionProof {
      opcode: opcode::ADD,
      pc: 22,
      stack_inputs: vec![a2, b2],
      stack_outputs: vec![c2],
      semantic_proof: Some(semantic2),
      memory_claims: vec![],
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
    std::mem::swap(
      &mut receipt1.lut_kernel_proof,
      &mut receipt2.lut_kernel_proof,
    );

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
      memory_claims: vec![],
    };
    let itp2 = InstructionTransitionProof {
      opcode: opcode::ADD,
      pc: 24,
      stack_inputs: vec![a2, b2],
      stack_outputs: vec![c2],
      semantic_proof: Some(semantic2),
      memory_claims: vec![],
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
    std::mem::swap(&mut receipt1.preprocessed_vk, &mut receipt2.preprocessed_vk);

    assert!(!verify_instruction_zk_receipt(&statement1, &receipt1));
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
          memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
        memory_claims: vec![],
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
      memory_claims: vec![],
    };

    // Step 2: ADD(300, 50) = 350
    let p2 = prove_instruction(opcode::ADD, &[c, d], &[e]).unwrap();
    let step2 = InstructionTransitionProof {
      opcode: opcode::ADD,
      pc: 1,
      stack_inputs: vec![c, d],
      stack_outputs: vec![e],
      semantic_proof: Some(p2),
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      memory_claims: vec![],
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
      (opcode::EQ, vec![small, small], bool_word(true)),
      (opcode::EQ, vec![small, large], bool_word(false)),
      (opcode::ISZERO, vec![zero], bool_word(true)),
      (opcode::ISZERO, vec![small], bool_word(false)),
      (opcode::LT, vec![small, large], bool_word(true)),
      (opcode::LT, vec![large, small], bool_word(false)),
      (opcode::GT, vec![large, small], bool_word(true)),
      (opcode::GT, vec![small, large], bool_word(false)),
      (opcode::SLT, vec![neg, small], bool_word(true)),
      (opcode::SLT, vec![small, neg], bool_word(false)),
      (opcode::SGT, vec![small, neg], bool_word(true)),
      (opcode::SGT, vec![neg, small], bool_word(false)),
    ];

    for (op, inputs, output) in cases {
      let proof = prove_instruction(*op, inputs, &[*output])
        .unwrap_or_else(|| panic!("prove_instruction failed for opcode 0x{op:02x}"));
      let inferred = infer_proof(&proof)
        .unwrap_or_else(|e| panic!("infer_proof failed for opcode 0x{op:02x}: {e}"));
      let expected = wff_instruction(*op, inputs, &[*output])
        .unwrap_or_else(|| panic!("wff_instruction returned None for opcode 0x{op:02x}"));
      assert_eq!(inferred, expected, "WFF mismatch for opcode 0x{op:02x}");
    }
  }

  #[test]
  fn test_cross_family_opcodes_zkp_receipt_passes() {
    // verify_proof_with_zkp previously failed for LT/GT/SLT/SGT/EQ/ISZERO
    // because their proof rows mix add-family and bit-family LUT ops.
    // After routing through build_lut_steps_from_rows_auto, all should succeed.
    let small = u256_bytes(5);
    let large = u256_bytes(100);
    let neg = i256_bytes(-10);
    let zero = u256_bytes(0);

    let cases: &[(u8, Vec<[u8; 32]>, [u8; 32])] = &[
      (opcode::LT, vec![small, large], bool_word(true)),
      (opcode::LT, vec![large, small], bool_word(false)),
      (opcode::GT, vec![large, small], bool_word(true)),
      (opcode::GT, vec![small, large], bool_word(false)),
      (opcode::SLT, vec![neg, small], bool_word(true)),
      (opcode::SLT, vec![small, neg], bool_word(false)),
      (opcode::SGT, vec![small, neg], bool_word(true)),
      (opcode::SGT, vec![neg, small], bool_word(false)),
      (opcode::EQ, vec![small, small], bool_word(true)),
      (opcode::EQ, vec![small, large], bool_word(false)),
      (opcode::ISZERO, vec![zero], bool_word(true)),
      (opcode::ISZERO, vec![small], bool_word(false)),
    ];

    for (op, inputs, output) in cases {
      let proof = prove_instruction(*op, inputs, &[*output])
        .unwrap_or_else(|| panic!("prove_instruction returned None for opcode 0x{op:02x}"));
      let itp = InstructionTransitionProof {
        opcode: *op,
        pc: 0,
        stack_inputs: inputs.clone(),
        stack_outputs: vec![*output],
        semantic_proof: Some(proof),
        memory_claims: vec![],
      };
      assert!(
        verify_proof(&itp),
        "verify_proof failed for opcode 0x{op:02x}"
      );
      assert!(
        verify_proof_with_rows(&itp),
        "verify_proof_with_rows failed for opcode 0x{op:02x}"
      );
      assert!(
        verify_proof_with_zkp(&itp),
        "verify_proof_with_zkp (ZK receipt) failed for opcode 0x{op:02x}"
      );
    }
  }
}

#[test]
fn test_and_debug_internals() {
  use revm::bytecode::opcode;
  use zprove_core::semantic_proof::compile_proof;
  use zprove_core::transition::{prove_instruction, wff_instruction};
  use zprove_core::zk_proof::{
    LutKernelAir, RECEIPT_BIND_TAG_LUT, RECEIPT_BIND_TAG_STACK,
    build_lut_steps_from_rows_bit_family, build_lut_trace_from_proof_rows, make_circle_config,
    make_receipt_binding_public_values, prove_lut_kernel_stark_with_public_values,
    prove_lut_with_prep_and_logup, prove_stack_ir_with_prep, setup_proof_rows_preprocessed,
    verify_lut_kernel_stark_with_public_values, verify_lut_with_prep,
    verify_lut_with_prep_and_logup, verify_stack_ir_with_prep,
  };

  // ── VERY EARLY byte_table fresh proof (before ANY other STARK calls) ────
  {
    use zprove_core::byte_table::{
      BYTE_OP_AND, ByteTableQuery, prove_byte_table, verify_byte_table,
    };
    let queries = vec![ByteTableQuery {
      a: 0xAA,
      b: 0x55,
      op: BYTE_OP_AND,
      result: 0x00,
      multiplicity: 1,
    }];
    let fresh_proof_early = prove_byte_table(&queries);
    let fresh_result_early = verify_byte_table(&fresh_proof_early);
    eprintln!(
      "VERY EARLY fresh byte_table proof verify: {:?}",
      fresh_result_early.is_ok()
    );
  }

  let a = [0xAAu8; 32];
  let b = [0x55u8; 32];
  let c = [0x00u8; 32];

  let proof = prove_instruction(opcode::AND, &[a, b], &[c]).unwrap();
  let rows = compile_proof(&proof);
  eprintln!("rows.len() = {}", rows.len());

  let expected_wff = wff_instruction(opcode::AND, &[a, b], &[c]).unwrap();
  let stack_bind_pv =
    make_receipt_binding_public_values(RECEIPT_BIND_TAG_STACK, opcode::AND, &expected_wff);
  let lut_bind_pv =
    make_receipt_binding_public_values(RECEIPT_BIND_TAG_LUT, opcode::AND, &expected_wff);

  // OLD path: bit-family steps => standalone LUT
  let lut_steps = build_lut_steps_from_rows_bit_family(&rows).unwrap();
  eprintln!("lut_steps.len() = {}", lut_steps.len());
  let old_lut_proof = prove_lut_kernel_stark_with_public_values(&lut_steps, &lut_bind_pv).unwrap();
  let old_lut_result = verify_lut_kernel_stark_with_public_values(&old_lut_proof, &lut_bind_pv);
  eprintln!("OLD LUT verify: {:?}", old_lut_result.is_ok());
  assert!(
    old_lut_result.is_ok(),
    "OLD LUT verify failed: {:?}",
    old_lut_result.err()
  );

  // ── byte_table fresh proof AFTER old LUT prove ────
  {
    use zprove_core::byte_table::{
      BYTE_OP_AND, ByteTableQuery, prove_byte_table, verify_byte_table,
    };
    let queries = vec![ByteTableQuery {
      a: 0xAA,
      b: 0x55,
      op: BYTE_OP_AND,
      result: 0x00,
      multiplicity: 1,
    }];
    let fresh_proof = prove_byte_table(&queries);
    eprintln!(
      "After OLD LUT prove: fresh byte_table verify: {:?}",
      verify_byte_table(&fresh_proof).is_ok()
    );
  }

  let (prep_data, prep_vk) = setup_proof_rows_preprocessed(&rows, &stack_bind_pv).unwrap();

  // ── byte_table fresh proof AFTER setup_prep ────
  {
    use zprove_core::byte_table::{
      BYTE_OP_AND, ByteTableQuery, prove_byte_table, verify_byte_table,
    };
    let queries = vec![ByteTableQuery {
      a: 0xAA,
      b: 0x55,
      op: BYTE_OP_AND,
      result: 0x00,
      multiplicity: 1,
    }];
    let fresh_proof = prove_byte_table(&queries);
    eprintln!(
      "After setup_prep: fresh byte_table verify: {:?}",
      verify_byte_table(&fresh_proof).is_ok()
    );
  }

  let stack_proof = prove_stack_ir_with_prep(&rows, &prep_data, &stack_bind_pv).unwrap();
  let stack_result = verify_stack_ir_with_prep(&stack_proof, &prep_vk, &stack_bind_pv);
  eprintln!("Stack IR verify: {:?}", stack_result.is_ok());
  assert!(
    stack_result.is_ok(),
    "stack IR verify failed: {:?}",
    stack_result.err()
  );

  // ── byte_table fresh proof AFTER stack IR prove ────
  {
    use zprove_core::byte_table::{
      BYTE_OP_AND, ByteTableQuery, prove_byte_table, verify_byte_table,
    };
    let queries = vec![ByteTableQuery {
      a: 0xAA,
      b: 0x55,
      op: BYTE_OP_AND,
      result: 0x00,
      multiplicity: 1,
    }];
    let fresh_proof = prove_byte_table(&queries);
    eprintln!(
      "After stack IR prove: fresh byte_table verify: {:?}",
      verify_byte_table(&fresh_proof).is_ok()
    );
  }

  // DIAGNOSTIC: LutKernelAir + build_lut_trace_from_proof_rows + prove/verify (no prep)
  // NOTE: This runs AFTER setup_prep and stack IR prove to test isolation
  let diag_proof_for_lut = {
    let trace = build_lut_trace_from_proof_rows(&rows).unwrap();
    let config = make_circle_config();
    let diag_proof = p3_uni_stark::prove(&config, &LutKernelAir, trace, &lut_bind_pv);
    let diag_result = p3_uni_stark::verify(&config, &LutKernelAir, &diag_proof, &lut_bind_pv);
    eprintln!(
      "DIAG LUT verify (after stack prove, old air + new trace): {:?}",
      diag_result.is_ok()
    );
    diag_proof
  };

  // ── byte_table fresh proof AFTER DIAG prove ────
  {
    use zprove_core::byte_table::{
      BYTE_OP_AND, ByteTableQuery, prove_byte_table, verify_byte_table,
    };
    let queries = vec![ByteTableQuery {
      a: 0xAA,
      b: 0x55,
      op: BYTE_OP_AND,
      result: 0x00,
      multiplicity: 1,
    }];
    let fresh_proof = prove_byte_table(&queries);
    eprintln!(
      "After DIAG prove: fresh byte_table verify: {:?}",
      verify_byte_table(&fresh_proof).is_ok()
    );
  }

  let (lut_proof, byte_table_proof) =
    prove_lut_with_prep_and_logup(&rows, &prep_data, &lut_bind_pv).unwrap();

  // ── byte_table fresh proof AFTER prove_lut_with_prep_and_logup ────
  {
    use zprove_core::byte_table::{
      BYTE_OP_AND, ByteTableQuery, prove_byte_table, verify_byte_table,
    };
    let queries = vec![ByteTableQuery {
      a: 0xAA,
      b: 0x55,
      op: BYTE_OP_AND,
      result: 0x00,
      multiplicity: 1,
    }];
    let fresh_proof = prove_byte_table(&queries);
    eprintln!(
      "After prove_lut: fresh byte_table verify: {:?}",
      verify_byte_table(&fresh_proof).is_ok()
    );
  }

  // Compare proofs
  eprintln!(
    "diag_proof.degree_bits = {}",
    diag_proof_for_lut.degree_bits
  );
  eprintln!("lut_proof.degree_bits  = {}", lut_proof.degree_bits);
  eprintln!(
    "diag trace_local len = {}",
    diag_proof_for_lut.opened_values.trace_local.len()
  );
  eprintln!(
    "lut  trace_local len = {}",
    lut_proof.opened_values.trace_local.len()
  );

  // Try verifying DIAG proof with LUT verifier
  let config = make_circle_config();
  let diag_as_lut = p3_uni_stark::verify_with_preprocessed(
    &config,
    &LutKernelAir,
    &diag_proof_for_lut,
    &lut_bind_pv,
    None,
  );
  eprintln!(
    "DIAG proof verified with LUT verifier (after stack): {:?}",
    diag_as_lut.is_ok()
  );

  // Try verifying LUT proof with p3_uni_stark::verify directly
  let config2 = make_circle_config();
  let lut_direct = p3_uni_stark::verify(&config2, &LutKernelAir, &lut_proof, &lut_bind_pv);
  eprintln!(
    "LUT proof verified with p3::verify directly: {:?}",
    lut_direct.is_ok()
  );

  // Try verifying LUT proof with verify_lut_with_prep (which uses same args)
  let lut_prep_result = verify_lut_with_prep(&lut_proof, &prep_vk, &lut_bind_pv);
  eprintln!(
    "LUT proof verify_lut_with_prep: {:?}",
    lut_prep_result.is_ok()
  );

  // Directly test byte_table verify
  eprintln!("byte_table_proof is Some: {:?}", byte_table_proof.is_some());
  if let Some(ref bp) = byte_table_proof {
    // Rebuild queries from rows and re-prove independently (BEFORE bt_result)
    use zprove_core::zk_proof::collect_byte_table_queries_from_rows;
    let queries = collect_byte_table_queries_from_rows(&rows);
    eprintln!("queries count: {}", queries.len());
    let fresh_proof_before = zprove_core::byte_table::prove_byte_table(&queries);
    let fresh_result_before = zprove_core::byte_table::verify_byte_table(&fresh_proof_before);
    eprintln!(
      "fresh byte_table proof verify BEFORE bt_result: {:?}",
      fresh_result_before.is_ok()
    );

    let bt_result = zprove_core::byte_table::verify_byte_table(bp);
    eprintln!("byte_table verify directly: {:?}", bt_result.is_ok());
    eprintln!("byte_table verify err: {:?}", bt_result.err());

    let fresh_proof = zprove_core::byte_table::prove_byte_table(&queries);
    let fresh_result = zprove_core::byte_table::verify_byte_table(&fresh_proof);
    eprintln!(
      "fresh byte_table proof verify AFTER bt_result: {:?}",
      fresh_result.is_ok()
    );
  }

  let lut_result = verify_lut_with_prep_and_logup(
    &lut_proof,
    byte_table_proof.as_ref(),
    &prep_vk,
    &lut_bind_pv,
  );
  eprintln!("LUT verify: {:?}", lut_result.is_ok());
  assert!(
    lut_result.is_ok(),
    "LUT verify failed: {:?}",
    lut_result.err()
  );
}

// ============================================================
// Memory consistency proof tests
// ============================================================

#[cfg(test)]
mod memory_consistency_tests {
  use std::collections::HashMap;
  use zprove_core::transition::MemAccessClaim;
  use zprove_core::zk_proof::{prove_memory_consistency, verify_memory_consistency};

  /// Compute correct `write_version` values for a sequence of claims.
  ///
  /// Each claim's wv = number of writes to the same address *before* this
  /// claim in the sequence (i.e. zero-indexed write count so far).
  fn assign_write_versions(claims: &mut Vec<MemAccessClaim>) {
    let mut wv_map: HashMap<u64, u32> = HashMap::new();
    for c in claims.iter_mut() {
      let wv = *wv_map.get(&c.addr).unwrap_or(&0);
      c.write_version = wv;
      if c.is_write {
        *wv_map.entry(c.addr).or_insert(0) += 1;
      }
    }
  }

  fn mwrite(addr: u64, rw: u64, val: u8) -> MemAccessClaim {
    let mut value = [0u8; 32];
    value[31] = val;
    MemAccessClaim {
      rw_counter: rw,
      addr,
      is_write: true,
      value,
      write_version: 0,
    }
  }

  fn mread(addr: u64, rw: u64, val: u8) -> MemAccessClaim {
    let mut value = [0u8; 32];
    value[31] = val;
    MemAccessClaim {
      rw_counter: rw,
      addr,
      is_write: false,
      value,
      write_version: 0,
    }
  }

  #[test]
  fn test_memory_consistency_single_write() {
    let mut claims = vec![mwrite(0, 1, 42)];
    assign_write_versions(&mut claims);
    let proof = prove_memory_consistency(&claims).expect("prove failed");
    assert!(verify_memory_consistency(&proof, &claims));
  }

  #[test]
  fn test_memory_consistency_write_then_read_same_value() {
    let mut claims = vec![mwrite(0, 1, 99), mread(0, 2, 99)];
    assign_write_versions(&mut claims);
    let proof = prove_memory_consistency(&claims).expect("prove failed");
    assert!(verify_memory_consistency(&proof, &claims));
  }

  #[test]
  fn test_memory_consistency_read_uninitialized_zero() {
    // Reading an address never written: value must be 0.
    let mut claims = vec![mread(64, 1, 0)];
    assign_write_versions(&mut claims);
    let proof = prove_memory_consistency(&claims).expect("prove failed");
    assert!(verify_memory_consistency(&proof, &claims));
  }

  #[test]
  fn test_memory_consistency_multiple_addresses() {
    let mut claims = vec![
      mwrite(0, 1, 11),
      mwrite(32, 2, 22),
      mread(0, 3, 11),
      mread(32, 4, 22),
    ];
    assign_write_versions(&mut claims);
    let proof = prove_memory_consistency(&claims).expect("prove failed");
    assert!(verify_memory_consistency(&proof, &claims));
  }

  #[test]
  fn test_memory_consistency_overwrite_same_address() {
    let mut claims = vec![
      mwrite(0, 1, 10),
      mwrite(0, 2, 20),
      mread(0, 3, 20), // reads the overwritten value
    ];
    assign_write_versions(&mut claims);
    let proof = prove_memory_consistency(&claims).expect("prove failed");
    assert!(verify_memory_consistency(&proof, &claims));
  }

  #[test]
  fn test_memory_consistency_proof_fails_bad_read() {
    // Read reports wrong value for an unwritten address.
    let mut claims = vec![mread(0, 1, 7)]; // 7 != 0 (uninitialised)
    assign_write_versions(&mut claims);
    let result = prove_memory_consistency(&claims);
    assert!(
      result.is_err(),
      "expected prove to fail for inconsistent read"
    );
  }

  #[test]
  fn test_memory_consistency_proof_fails_stale_read() {
    // Read after newer overwrite sees old value — should fail.
    let mut claims = vec![
      mwrite(0, 1, 10),
      mwrite(0, 2, 20),
      mread(0, 3, 10), // stale: should read 20
    ];
    assign_write_versions(&mut claims);
    let result = prove_memory_consistency(&claims);
    assert!(result.is_err(), "expected prove to fail for stale read");
  }

  #[test]
  fn test_memory_consistency_proof_fails_wrong_read_after_write() {
    // Write 5, then read 6 (wrong).
    let mut claims = vec![mwrite(0, 1, 5), mread(0, 2, 6)];
    assign_write_versions(&mut claims);
    let result = prove_memory_consistency(&claims);
    assert!(
      result.is_err(),
      "expected prove to fail for wrong read value"
    );
  }

  #[test]
  fn test_memory_consistency_proof_roundtrip_larger() {
    // Prove → verify roundtrip with several addresses and values.
    let mut claims = vec![
      mwrite(0, 1, 1),
      mwrite(32, 2, 2),
      mwrite(0, 3, 3), // overwrite addr 0
      mread(0, 4, 3),  // reads overwritten value
      mread(32, 5, 2),
    ];
    assign_write_versions(&mut claims);
    let proof = prove_memory_consistency(&claims).expect("prove failed");
    assert!(verify_memory_consistency(&proof, &claims));
  }

  #[test]
  fn test_memory_consistency_empty_claims() {
    let claims: Vec<MemAccessClaim> = vec![];
    let proof = prove_memory_consistency(&claims).expect("prove failed");
    assert!(verify_memory_consistency(&proof, &claims));
  }
}
