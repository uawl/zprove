// moved from src/zk_proof.rs

#[cfg(test)]
mod memory_bus_tests {
  use p3_matrix::Matrix;
  use zprove_core::memory_proof::{CqMemoryEvent, CqRw};
  use zprove_core::zk_proof::{
    build_memory_bus_steps_from_events, build_memory_bus_trace_from_steps,
    prove_and_verify_memory_bus_stark_from_events,
  };

  #[test]
  fn test_memory_bus_steps_sort_and_same_cell_flag() {
    let events = vec![
      CqMemoryEvent {
        addr: 10,
        step: 2,
        value: [2u8; 32],
        rw: CqRw::Write,
        width: 32,
      },
      CqMemoryEvent {
        addr: 10,
        step: 2,
        value: [2u8; 32],
        rw: CqRw::Read,
        width: 32,
      },
      CqMemoryEvent {
        addr: 11,
        step: 1,
        value: [1u8; 32],
        rw: CqRw::Read,
        width: 32,
      },
    ];

    let steps = build_memory_bus_steps_from_events(&events).expect("step build should succeed");
    assert_eq!(steps.len(), 3);
    assert_eq!(steps[0].addr, 10);
    assert_eq!(steps[0].rw, 0);
    assert_eq!(steps[0].same_cell_next, 1);
    assert_eq!(steps[1].addr, 10);
    assert_eq!(steps[1].rw, 1);
    assert_eq!(steps[1].same_cell_next, 0);
  }

  #[test]
  fn test_memory_bus_trace_builds() {
    let events = vec![CqMemoryEvent {
      addr: 5,
      step: 1,
      value: [0xAB; 32],
      rw: CqRw::Read,
      width: 32,
    }];

    let steps = build_memory_bus_steps_from_events(&events).expect("step build should succeed");
    let trace = build_memory_bus_trace_from_steps(&steps).expect("trace build should succeed");
    assert!(trace.height() >= 4);
  }

  #[test]
  fn test_memory_bus_stark_roundtrip_from_events() {
    let events = vec![
      CqMemoryEvent {
        addr: 64,
        step: 9,
        value: [0x11; 32],
        rw: CqRw::Read,
        width: 32,
      },
      CqMemoryEvent {
        addr: 64,
        step: 9,
        value: [0x11; 32],
        rw: CqRw::Write,
        width: 32,
      },
    ];

    assert!(prove_and_verify_memory_bus_stark_from_events(&events));
  }
}

#[cfg(test)]
mod tests {
  use zprove_core::semantic_proof::{
    NUM_PROOF_COLS, OP_BYTE_ADD_EQ, OP_U24_ADD_EQ, OP_U29_ADD_EQ, Proof, ProofRow, RET_WFF_EQ,
    Term, compile_proof, infer_proof, prove_add, prove_mul, wff_add, wff_mul,
  };
  use zprove_core::zk_proof::*;
  use p3_matrix::Matrix;

  fn tamper_first_mul_related_leaf(proof: &mut Proof) -> bool {
    match proof {
      Proof::ByteMulLowEq(a, b) => {
        *proof = Proof::ByteMulLowEq(*a ^ 1, *b);
        true
      }
      Proof::ByteMulHighEq(a, b) => {
        *proof = Proof::ByteMulHighEq(*a ^ 1, *b);
        true
      }
      Proof::ByteAddEq(a, b, c) => {
        *proof = Proof::ByteAddEq(*a, *b, !*c);
        true
      }
      Proof::U15MulEq(a, b) => {
        *proof = Proof::U15MulEq(*a ^ 1, *b);
        true
      }
      Proof::U29AddEq(a, b, cin, c) => {
        *proof = Proof::U29AddEq(*a ^ 1, *b, *cin, *c);
        true
      }
      Proof::U24AddEq(a, b, cin, c) => {
        *proof = Proof::U24AddEq(*a ^ 1, *b, *cin, *c);
        true
      }
      Proof::U16AddEq(a, b, cin, c) => {
        *proof = Proof::U16AddEq(*a, *b, !*cin, *c);
        true
      }
      Proof::AndIntro(p1, p2) => {
        if tamper_first_mul_related_leaf(p1) {
          true
        } else {
          tamper_first_mul_related_leaf(p2)
        }
      }
      Proof::EqSym(p) => tamper_first_mul_related_leaf(p),
      Proof::EqTrans(p1, p2) => {
        if tamper_first_mul_related_leaf(p1) {
          true
        } else {
          tamper_first_mul_related_leaf(p2)
        }
      }
      Proof::ByteAddThirdCongruence(p, _, _) => tamper_first_mul_related_leaf(p),
      Proof::ByteAddCarryThirdCongruence(p, _, _) => tamper_first_mul_related_leaf(p),
      _ => false,
    }
  }

  fn u256_bytes(val: u128) -> [u8; 32] {
    let mut b = [0u8; 32];
    b[16..32].copy_from_slice(&val.to_be_bytes());
    b
  }

  fn compute_add(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    let mut result = [0u8; 32];
    let mut carry = 0u16;
    for i in (0..32).rev() {
      let sum = a[i] as u16 + b[i] as u16 + carry;
      result[i] = (sum & 0xFF) as u8;
      carry = sum >> 8;
    }
    result
  }

  #[test]
  fn test_stage_a_semantic_trace_generation() {
    let a = u256_bytes(1000);
    let b = u256_bytes(2000);
    let c = compute_add(&a, &b);
    let semantic = prove_add(&a, &b, &c);
    let rows = compile_proof(&semantic);
    let trace =
      generate_stage_a_semantic_trace(&rows).expect("semantic trace generation should succeed");
    assert_eq!(trace.width(), NUM_PROOF_COLS);
    assert_eq!(trace.height(), 16);
  }

  #[test]
  fn test_prove_and_verify_stage1_simple_semantic_rows() {
    let a = u256_bytes(1000);
    let b = u256_bytes(2000);
    let c = compute_add(&a, &b);

    let semantic = prove_add(&a, &b, &c);
    let rows = compile_proof(&semantic);
    let proof = prove_inferred_wff_stark(&rows).expect("stage-1 proof should succeed");
    verify_inferred_wff_stark(&proof).expect("stage-1 verification failed");
  }

  #[test]
  fn test_prove_and_verify_stage1_overflow_semantic_rows() {
    let a = [0xFF; 32];
    let mut b = [0u8; 32];
    b[31] = 1;
    let c = compute_add(&a, &b);

    let semantic = prove_add(&a, &b, &c);
    let rows = compile_proof(&semantic);
    let proof = prove_inferred_wff_stark(&rows).expect("stage-1 proof should succeed");
    verify_inferred_wff_stark(&proof).expect("stage-1 verification failed");
  }

  #[test]
  fn test_prove_and_verify_stage1_large_semantic_rows() {
    let mut a = [0xABu8; 32];
    a[0] = 0x7F;
    let mut b = [0xCDu8; 32];
    b[0] = 0x3E;
    let c = compute_add(&a, &b);

    let semantic = prove_add(&a, &b, &c);
    let rows = compile_proof(&semantic);
    let proof = prove_inferred_wff_stark(&rows).expect("stage-1 proof should succeed");
    verify_inferred_wff_stark(&proof).expect("stage-1 verification failed");
  }

  #[test]
  fn test_prove_and_verify_stage1_zero_semantic_rows() {
    let a = [0u8; 32];
    let b = [0u8; 32];
    let c = [0u8; 32];

    let semantic = prove_add(&a, &b, &c);
    let rows = compile_proof(&semantic);
    let proof = prove_inferred_wff_stark(&rows).expect("stage-1 proof should succeed");
    verify_inferred_wff_stark(&proof).expect("stage-1 verification failed");
  }

  #[test]
  fn test_prove_and_verify_stage1_from_compiled_rows() {
    let a = u256_bytes(1234);
    let b = u256_bytes(5678);
    let c = compute_add(&a, &b);

    let semantic = prove_add(&a, &b, &c);
    let rows = compile_proof(&semantic);
    assert!(prove_and_verify_compiled_rows_stark(&rows));
  }

  #[test]
  fn test_prove_and_verify_stage2_wff_match() {
    let a = u256_bytes(1111);
    let b = u256_bytes(2222);
    let c = compute_add(&a, &b);

    let semantic = prove_add(&a, &b, &c);
    let public_wff = wff_add(&a, &b, &c);
    assert!(prove_and_verify_wff_match_stark(&semantic, &public_wff));
  }

  #[test]
  fn test_wff_match_fails_on_wrong_output() {
    let a = u256_bytes(100);
    let b = u256_bytes(200);
    let c = compute_add(&a, &b);
    let mut wrong = c;
    wrong[31] ^= 1;

    let semantic = prove_add(&a, &b, &c);
    let wrong_public_wff = wff_add(&a, &b, &wrong);
    assert!(!prove_and_verify_wff_match_stark(
      &semantic,
      &wrong_public_wff
    ));
  }

  #[test]
  fn test_stage_a_success() {
    let a = u256_bytes(777);
    let b = u256_bytes(888);
    let c = compute_add(&a, &b);

    let public_wff = wff_add(&a, &b, &c);
    let private_pi = prove_add(&a, &b, &c);
    assert!(prove_and_verify_stage_a(&public_wff, &private_pi));
  }

  #[test]
  fn test_stage_a_fails_on_wrong_public_wff() {
    let a = u256_bytes(10);
    let b = u256_bytes(20);
    let c = compute_add(&a, &b);
    let mut wrong = c;
    wrong[31] ^= 1;

    let private_pi = prove_add(&a, &b, &c);
    let wrong_public_wff = wff_add(&a, &b, &wrong);
    assert!(!prove_and_verify_stage_a(&wrong_public_wff, &private_pi));
  }

  #[test]
  fn test_stage_a_supports_non_add_infer_path() {
    let private_pi = Proof::EqRefl(Term::Byte(42));
    let public_wff = infer_proof(&private_pi).expect("infer_proof should succeed");
    assert!(!prove_and_verify_stage_a(&public_wff, &private_pi));
  }

  #[test]
  fn test_build_lut_steps_from_rows_byte_add_eq_only() {
    let proof = Proof::ByteAddEq(10, 20, true);
    let rows = compile_proof(&proof);
    let steps = build_lut_steps_from_rows(&rows).expect("lut steps should build");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].op, LutOpcode::ByteAddEq);
    assert_eq!(steps[0].in0, 10);
    assert_eq!(steps[0].in1, 20);
    assert_eq!(steps[0].in2, 1);
    assert_eq!(steps[0].out0, 31);
    assert_eq!(steps[0].out1, 0);

    let trace = build_lut_trace_from_steps(&steps).expect("lut trace should build");
    assert_eq!(trace.width(), 6);
    assert!(trace.height() >= 4);
  }

  #[test]
  fn test_build_lut_steps_rejects_structural_rows() {
    let proof = Proof::EqRefl(Term::Byte(7));
    let rows = compile_proof(&proof);
    let err = build_lut_steps_from_rows(&rows).expect_err("structural rows must be rejected");
    assert!(err.contains("not LUT-step encodable"));
  }

  #[test]
  fn test_build_lut_trace_rejects_out_of_range_u16_add_step() {
    let steps = vec![LutStep {
      op: LutOpcode::U16AddEq,
      in0: 70_000,
      in1: 1,
      in2: 0,
      out0: 1,
      out1: 0,
    }];

    let err =
      build_lut_trace_from_steps(&steps).expect_err("out-of-range u16-add input must be rejected");
    assert!(err.contains("out of range"));
  }

  #[test]
  fn test_build_lut_trace_rejects_out_of_range_byte_add_step() {
    let steps = vec![LutStep {
      op: LutOpcode::ByteAddEq,
      in0: 300,
      in1: 1,
      in2: 0,
      out0: 45,
      out1: 0,
    }];

    let err =
      build_lut_trace_from_steps(&steps).expect_err("out-of-range byte-add input must be rejected");
    assert!(err.contains("out of range"));
  }

  #[test]
  fn test_stage_a_trace_rejects_out_of_range_row() {
    let rows = vec![ProofRow {
      op: OP_BYTE_ADD_EQ,
      scalar0: 300,
      scalar1: 1,
      scalar2: 0,
      arg0: 0,
      arg1: 45,
      arg2: 0,
      value: 45,
      ret_ty: RET_WFF_EQ,
    }];

    let err = generate_stage_a_semantic_trace(&rows)
      .expect_err("out-of-range stage-a row must be rejected");
    assert!(err.contains("out of range"));
  }

  #[test]
  fn test_stage_a_rejects_forged_u29_carry_chain_break() {
    let a = [0xAA; 32];
    let b = [0x55; 32];
    let c = compute_add(&a, &b);

    let proof = prove_add(&a, &b, &c);
    let mut rows = compile_proof(&proof);
    let row_idx = rows
      .iter()
      .position(|r| r.op == OP_U29_ADD_EQ)
      .expect("U29 rows must exist for add proof");

    rows[row_idx].scalar2 ^= 1;
    assert!(!prove_and_verify_compiled_rows_stark(&rows));
  }

  #[test]
  fn test_stage_a_rejects_forged_u29_to_u24_opcode_downgrade() {
    let a = [0xFF; 32];
    let b = [0xFF; 32];
    let c = compute_add(&a, &b);

    let proof = prove_add(&a, &b, &c);
    let mut rows = compile_proof(&proof);
    let row_idx = rows
      .iter()
      .position(|r| r.op == OP_U29_ADD_EQ && r.arg1 == 1)
      .expect("expected at least one U29 row with carry-out = 1");

    rows[row_idx].op = OP_U24_ADD_EQ;
    assert!(!prove_and_verify_compiled_rows_stark(&rows));
  }

  #[test]
  fn test_build_stack_ir_steps_from_rows_eq_refl() {
    let proof = Proof::EqRefl(Term::Byte(7));
    let rows = compile_proof(&proof);
    let steps = build_stack_ir_steps_from_rows(&rows).expect("stack-ir steps should build");

    assert_eq!(steps.len(), rows.len());
    assert_eq!(steps[0].sp_before, 0);
    assert_eq!(steps[0].sp_after, 1);
    assert_eq!(steps[1].sp_before, 1);
    assert_eq!(steps[1].sp_after, 1);

    let trace = build_stack_ir_trace_from_steps(&steps).expect("stack-ir trace should build");
    assert_eq!(trace.width(), 10);
    assert!(trace.height() >= 4);
  }

  #[test]
  fn test_build_stack_ir_steps_underflow_rejected() {
    let rows = vec![ProofRow {
      op: zprove_core::semantic_proof::OP_EQ_TRANS,
      ret_ty: RET_WFF_EQ,
      ..Default::default()
    }];

    let err = build_stack_ir_steps_from_rows(&rows).expect_err("underflow must be rejected");
    assert!(err.contains("stack underflow"));
  }

  #[test]
  fn test_stack_ir_scaffold_stark_eq_refl_roundtrip() {
    let proof = Proof::EqRefl(Term::Byte(7));
    let rows = compile_proof(&proof);
    assert!(prove_and_verify_stack_ir_scaffold_stark(&rows));
  }

  #[test]
  fn test_stack_ir_scaffold_supports_add_rows() {
    let a = u256_bytes(10);
    let b = u256_bytes(20);
    let c = compute_add(&a, &b);
    let proof = prove_add(&a, &b, &c);
    let rows = compile_proof(&proof);
    assert!(prove_and_verify_stack_ir_scaffold_stark(&rows));
  }

  #[test]
  fn test_lut_kernel_stark_byte_add_eq_roundtrip() {
    let proof = Proof::ByteAddEq(100, 200, false);
    let rows = compile_proof(&proof);
    let steps = build_lut_steps_from_rows(&rows).expect("lut steps should build");
    assert!(prove_and_verify_lut_kernel_stark_from_steps(&steps));
  }

  #[test]
  fn test_lut_kernel_stark_byte_add_carry_eq_roundtrip() {
    let proof = Proof::ByteAddCarryEq(200, 100, false);
    let rows = compile_proof(&proof);
    let steps = build_lut_steps_from_rows(&rows).expect("lut steps should build");
    assert!(prove_and_verify_lut_kernel_stark_from_steps(&steps));
  }

  #[test]
  fn test_lut_kernel_stark_supports_mul_ops() {
    let proof = Proof::ByteMulLowEq(3, 7);
    let rows = compile_proof(&proof);
    let steps = build_lut_steps_from_rows(&rows).expect("lut steps should build");
    assert!(prove_and_verify_lut_kernel_stark_from_steps(&steps));
  }

  #[test]
  fn test_build_lut_trace_accepts_bit_level_bit_ops_inputs() {
    let steps = vec![LutStep {
      op: LutOpcode::ByteAndEq,
      in0: 1,
      in1: 0,
      in2: 0,
      out0: 0,
      out1: 0,
    }];

    let trace = build_lut_trace_from_steps(&steps)
      .expect("bit-level bitwise steps should be trace-encodable");
    assert_eq!(trace.width(), 6);
  }

  #[test]
  fn test_build_lut_trace_rejects_non_bool_bit_ops_inputs() {
    let steps = vec![LutStep {
      op: LutOpcode::ByteAndEq,
      in0: 2,
      in1: 1,
      in2: 0,
      out0: 0,
      out1: 0,
    }];

    let err =
      build_lut_trace_from_steps(&steps).expect_err("non-bool bitwise step should be rejected");
    assert!(err.contains("out of range"));
  }

  #[test]
  fn test_add_composite_stack_lut_roundtrip() {
    let a = u256_bytes(1000);
    let b = u256_bytes(2000);
    let c = compute_add(&a, &b);
    let proof = prove_add(&a, &b, &c);
    let rows = compile_proof(&proof);
    assert!(prove_and_verify_add_stack_lut_stark(&rows));
  }

  #[test]
  fn test_add_composite_stack_lut_rejects_non_add_rowset() {
    let proof = Proof::EqRefl(Term::Byte(7));
    let rows = compile_proof(&proof);
    assert!(!prove_and_verify_add_stack_lut_stark(&rows));
  }

  #[test]
  fn test_add_family_lut_builder_rejects_foreign_row() {
    let rows = vec![
      ProofRow {
        op: zprove_core::semantic_proof::OP_BYTE_ADD_EQ,
        scalar0: 1,
        scalar1: 2,
        arg0: 0,
        value: 3,
        scalar2: 0,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      },
      ProofRow {
        op: zprove_core::semantic_proof::OP_BYTE_MUL_LOW_EQ,
        scalar0: 2,
        scalar1: 3,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      },
    ];

    let err =
      build_lut_steps_from_rows_add_family(&rows).expect_err("foreign rows must be rejected");
    assert!(err.contains("non-add-family row"));
  }

  #[test]
  fn test_mul_composite_stack_lut_roundtrip() {
    let mut a = [0u8; 32];
    let mut b = [0u8; 32];
    a[31] = 123;
    b[31] = 45;
    let c = {
      let mut out = [0u8; 32];
      out[31] = 159;
      out[30] = 21;
      out
    };

    let proof = zprove_core::semantic_proof::prove_mul(&a, &b, &c);
    let rows = compile_proof(&proof);
    assert!(prove_and_verify_mul_stack_lut_stark(&rows));
  }

  #[test]
  fn test_mul_family_lut_builder_rejects_foreign_row() {
    let rows = vec![
      ProofRow {
        op: zprove_core::semantic_proof::OP_BYTE_MUL_LOW_EQ,
        scalar0: 3,
        scalar1: 7,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      },
      ProofRow {
        op: zprove_core::semantic_proof::OP_BYTE_OR_EQ,
        scalar0: 1,
        scalar1: 2,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      },
    ];

    let err =
      build_lut_steps_from_rows_mul_family(&rows).expect_err("foreign rows must be rejected");
    assert!(err.contains("non-mul-family row"));
  }

  #[test]
  fn test_mul_leaf_tamper_rejected_by_wff_match() {
    let mut a = [0u8; 32];
    let mut b = [0u8; 32];
    a[31] = 123;
    b[31] = 45;
    let c = {
      let mut out = [0u8; 32];
      out[31] = 159;
      out[30] = 21;
      out
    };

    let mut proof = prove_mul(&a, &b, &c);
    assert!(tamper_first_mul_related_leaf(&mut proof));

    let public_wff = wff_mul(&a, &b, &c);
    assert!(!prove_and_verify_expected_wff_match_stark(
      &proof,
      &public_wff
    ));
  }
}
