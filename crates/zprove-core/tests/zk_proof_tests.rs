// moved from src/zk_proof.rs

#[cfg(test)]
mod tests {
  use p3_matrix::Matrix;
  use zprove_core::semantic_proof::{
    NUM_PROOF_COLS, OP_U24_ADD_EQ, OP_U29_ADD_EQ, Proof, ProofRow, RET_WFF_EQ, Term, compile_proof,
    infer_proof, prove_add, prove_mul, wff_add, wff_mul,
  };
  use zprove_core::zk_proof::*;

  fn tamper_first_mul_related_leaf(proof: &mut Proof) -> bool {
    match proof {
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
    assert_eq!(trace.height(), 4); // CSE reduces 1000+2000 proof to 4 unique rows
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
  fn test_build_lut_steps_rejects_structural_rows() {
    let proof = Proof::EqRefl(Term::Byte(7));
    let rows = compile_proof(&proof);
    let err = build_lut_steps_from_rows(&rows).expect_err("structural rows must be rejected");
    assert!(err.contains("not LUT-step encodable"));
  }

  #[test]
  fn test_stage_a_trace_rejects_out_of_range_row() {
    let rows = vec![ProofRow {
      op: OP_U29_ADD_EQ,
      scalar0: 1 << 29,
      scalar1: 1,
      scalar2: 0,
      arg0: 0,
      arg1: 0,
      arg2: 0,
      value: 1,
      ret_ty: RET_WFF_EQ,
    }];

    let err = generate_stage_a_semantic_trace(&rows)
      .expect_err("out-of-range stage-a row must be rejected");
    assert!(err.contains("unsupported Stage A"));
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
    // Row 0: Byte(7) — leaf node, no args
    assert_eq!(steps[0].arg0, 0);
    assert_eq!(steps[0].arg1, 0);
    assert_eq!(steps[0].arg2, 0);
    // Row 1: EqRefl — arg0 points to row 0 (the Byte term)
    assert_eq!(steps[1].arg0, 0);
    assert_eq!(steps[1].arg1, 0);
    assert_eq!(steps[1].arg2, 0);

    let trace = build_stack_ir_trace_from_steps(&steps).expect("stack-ir trace should build");
    assert_eq!(trace.width(), zprove_core::zk_proof::NUM_STACK_IR_COLS);
    assert!(trace.height() >= 4);
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
    assert_eq!(trace.width(), zprove_core::zk_proof::NUM_LUT_COLS);
  }

  #[test]
  fn test_build_lut_trace_accepts_byte_level_bit_ops_inputs() {
    // ByteAndEq/OrEq/XorEq now operate at byte level (0..=255), not bit level.
    // in0=2, in1=1 are valid byte inputs; soundness is delegated to the LogUp byte table argument.
    let steps = vec![
      LutStep {
        op: LutOpcode::ByteAndEq,
        in0: 0xAB,
        in1: 0xCD,
        in2: 0,
        out0: (0xAB_u8 & 0xCD_u8) as u32,
        out1: 0,
      },
      LutStep {
        op: LutOpcode::ByteOrEq,
        in0: 0xAB,
        in1: 0xCD,
        in2: 0,
        out0: (0xAB_u8 | 0xCD_u8) as u32,
        out1: 0,
      },
      LutStep {
        op: LutOpcode::ByteXorEq,
        in0: 0xAB,
        in1: 0xCD,
        in2: 0,
        out0: (0xAB_u8 ^ 0xCD_u8) as u32,
        out1: 0,
      },
    ];

    let trace = build_lut_trace_from_steps(&steps)
      .expect("byte-level bitwise steps should be trace-encodable");
    assert_eq!(trace.width(), zprove_core::zk_proof::NUM_LUT_COLS);
  }

  #[test]
  fn test_build_lut_trace_rejects_out_of_byte_range_bit_ops_inputs() {
    // Values > 255 are invalid even for byte-level operations.
    let steps = vec![LutStep {
      op: LutOpcode::ByteAndEq,
      in0: 256,
      in1: 1,
      in2: 0,
      out0: 0,
      out1: 0,
    }];

    let err =
      build_lut_trace_from_steps(&steps).expect_err("out-of-byte-range input should be rejected");
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
        op: zprove_core::semantic_proof::OP_U29_ADD_EQ,
        scalar0: 1,
        scalar1: 2,
        scalar2: 0,
        arg0: 0,
        arg1: 0,
        value: 3,
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

// ============================================================
// Phase 1: Shared preprocessed rows infrastructure tests
// ============================================================

#[cfg(test)]
mod preprocessed_rows_tests {
  use p3_matrix::Matrix;
  use zprove_core::semantic_proof::{ProofRow, compile_proof};
  use zprove_core::zk_proof::{
    NUM_PREP_COLS, PREP_COL_OP, PREP_COL_SCALAR0, PREP_COL_SCALAR1, PREP_COL_VALUE,
    build_proof_rows_preprocessed_matrix, prove_stack_ir_with_prep, setup_proof_rows_preprocessed,
    stack_ir_scaffold_public_values, verify_stack_ir_with_prep,
  };

  // ---------------------------------------------------------------------------
  // Helper: a minimal add proof compiled into rows.
  // ---------------------------------------------------------------------------
  fn add_proof_rows() -> Vec<ProofRow> {
    use zprove_core::semantic_proof::prove_add;
    compile_proof(&prove_add(&[0u8; 32], &[0u8; 32], &[0u8; 32]))
  }

  // ---------------------------------------------------------------------------
  // Test 1: matrix dimensions
  // ---------------------------------------------------------------------------
  #[test]
  fn test_preprocessed_matrix_dimensions() {
    let rows = add_proof_rows();
    let matrix = build_proof_rows_preprocessed_matrix(&rows, &stack_ir_scaffold_public_values());

    // Height must be a power of two and at least 4.
    let h = matrix.height();
    assert!(h >= 4, "height must be >= 4");
    assert!(h.is_power_of_two(), "height must be power of two");
    assert!(h >= rows.len(), "height must cover all rows");

    assert_eq!(
      matrix.width(),
      NUM_PREP_COLS,
      "width must equal NUM_PREP_COLS"
    );
  }

  // ---------------------------------------------------------------------------
  // Test 2: real row values are preserved in the matrix.
  // ---------------------------------------------------------------------------
  #[test]
  fn test_preprocessed_matrix_row_values() {
    let rows = add_proof_rows();
    let matrix = build_proof_rows_preprocessed_matrix(&rows, &stack_ir_scaffold_public_values());

    for (i, row) in rows.iter().enumerate() {
      use p3_field::PrimeCharacteristicRing;
      use p3_mersenne_31::Mersenne31;
      let row_data = matrix.row_slice(i).unwrap();
      assert_eq!(
        row_data[PREP_COL_OP],
        Mersenne31::from_u32(row.op),
        "row {i}: op mismatch"
      );
      assert_eq!(
        row_data[PREP_COL_SCALAR0],
        Mersenne31::from_u32(row.scalar0),
        "row {i}: scalar0 mismatch"
      );
      assert_eq!(
        row_data[PREP_COL_SCALAR1],
        Mersenne31::from_u32(row.scalar1),
        "row {i}: scalar1 mismatch"
      );
      assert_eq!(
        row_data[PREP_COL_VALUE],
        Mersenne31::from_u32(row.value),
        "row {i}: value mismatch"
      );
    }
  }

  // ---------------------------------------------------------------------------
  // Test 3: setup_proof_rows_preprocessed returns without error.
  // ---------------------------------------------------------------------------
  #[test]
  fn test_setup_proof_rows_preprocessed_ok() {
    let rows = add_proof_rows();
    let result = setup_proof_rows_preprocessed(&rows, &stack_ir_scaffold_public_values());
    assert!(result.is_ok(), "setup must succeed: {:?}", result.err());
  }

  // ---------------------------------------------------------------------------
  // Test 4: prove + verify round-trip with shared preprocessed commitment.
  // ---------------------------------------------------------------------------
  #[test]
  fn test_prove_verify_stack_ir_with_prep_roundtrip() {
    let rows = add_proof_rows();

    // Build the public values expected by StackIrAir (tag + zeros for scaffold).
    let pv = stack_ir_scaffold_public_values();

    // Commit the compiled rows as a preprocessed trace.
    let (prep_data, prep_vk) =
      setup_proof_rows_preprocessed(&rows, &pv).expect("setup must succeed");

    // Prove.
    let proof = prove_stack_ir_with_prep(&rows, &prep_data, &pv).expect("prove must succeed");

    // Verify.
    let result = verify_stack_ir_with_prep(&proof, &prep_vk, &pv);
    assert!(result.is_ok(), "verify must succeed: {:?}", result.err());
  }

  // ---------------------------------------------------------------------------
  // Test 5: VK from a *different* row set must fail verification.
  // ---------------------------------------------------------------------------
  #[test]
  fn test_verify_fails_with_wrong_prep_vk() {
    let rows = add_proof_rows();

    // Real commitment.
    let pv = stack_ir_scaffold_public_values();
    let (prep_data, _real_vk) =
      setup_proof_rows_preprocessed(&rows, &pv).expect("setup must succeed");

    // Manufacture a *different* row set (just one extra row).
    let mut other_rows = rows.clone();
    // Mutate the op of the first row to produce a different commitment.
    other_rows[0].op = other_rows[0].op.wrapping_add(1);
    let (_other_data, other_vk) =
      setup_proof_rows_preprocessed(&other_rows, &pv).expect("other setup must succeed");

    let pv = stack_ir_scaffold_public_values();
    let proof = prove_stack_ir_with_prep(&rows, &prep_data, &pv).expect("prove must succeed");

    // Verifying with the wrong VK should fail.
    let result = verify_stack_ir_with_prep(&proof, &other_vk, &pv);
    assert!(
      result.is_err(),
      "verification with wrong VK must fail, but succeeded"
    );
  }
}

// ============================================================
// Phase 3: LUT preprocessed binding tests
// ============================================================

#[cfg(test)]
mod lut_prep_tests {
  use p3_matrix::Matrix;
  use zprove_core::semantic_proof::{ProofRow, compile_proof};
  use zprove_core::transition::build_batch_manifest;
  use zprove_core::zk_proof::{
    NUM_LUT_COLS, RECEIPT_BIND_TAG_LUT, RECEIPT_BIND_TAG_STACK, build_lut_trace_from_proof_rows,
    make_batch_receipt_binding_public_values, prove_batch_lut_with_prep, prove_stack_ir_with_prep,
    setup_batch_proof_rows_preprocessed, setup_proof_rows_preprocessed,
    stack_ir_scaffold_public_values, verify_batch_lut_with_prep, verify_stack_ir_with_prep,
  };

  // ---------------------------------------------------------------------------
  // Helper: a minimal add proof compiled into rows.
  // ---------------------------------------------------------------------------
  fn add_rows() -> Vec<ProofRow> {
    use zprove_core::semantic_proof::prove_add;
    compile_proof(&prove_add(&[0u8; 32], &[0u8; 32], &[0u8; 32]))
  }

  // ---------------------------------------------------------------------------
  // Test 1: build_lut_trace_from_proof_rows dimensions match preprocessed.
  // ---------------------------------------------------------------------------
  #[test]
  fn test_lut_trace_from_proof_rows_dimensions() {
    let rows = add_rows();
    let trace = build_lut_trace_from_proof_rows(&rows).expect("build must succeed");
    let expected_height = rows.len().max(4).next_power_of_two();
    assert_eq!(
      trace.height(),
      expected_height,
      "height must match preprocessed"
    );
    assert_eq!(trace.width(), NUM_LUT_COLS, "width must equal NUM_LUT_COLS");
  }

  // Helper: build a batch manifest for a single ADD instruction.
  fn add_manifest() -> zprove_core::zk_proof::BatchProofRowsManifest {
    use revm::bytecode::opcode;
    use zprove_core::semantic_proof::prove_add;
    let proof = prove_add(&[0u8; 32], &[0u8; 32], &[0u8; 32]);
    let items: &[(u8, &zprove_core::semantic_proof::Proof)] = &[(opcode::ADD, &proof)];
    build_batch_manifest(items).expect("manifest build must succeed")
  }

  // ---------------------------------------------------------------------------
  // Test 2: Prove + verify batch LUT (N=1) roundtrip.
  // ---------------------------------------------------------------------------
  #[test]
  fn test_prove_verify_lut_with_prep_roundtrip() {
    let manifest = add_manifest();
    let pv = make_batch_receipt_binding_public_values(RECEIPT_BIND_TAG_LUT, &manifest.entries);
    let (prep_data, prep_vk) =
      setup_batch_proof_rows_preprocessed(&manifest, &pv).expect("setup must succeed");
    let proof = prove_batch_lut_with_prep(&manifest, &prep_data, &pv).expect("prove must succeed");
    let result = verify_batch_lut_with_prep(&proof, &prep_vk, &pv);
    assert!(result.is_ok(), "verify must succeed: {:?}", result.err());
  }

  // ---------------------------------------------------------------------------
  // Test 3: Wrong VK fails batch LUT verification.
  // ---------------------------------------------------------------------------
  #[test]
  fn test_lut_verify_fails_with_wrong_prep_vk() {
    use revm::bytecode::opcode;
    use zprove_core::semantic_proof::prove_add;

    let manifest = add_manifest();
    let pv = make_batch_receipt_binding_public_values(RECEIPT_BIND_TAG_LUT, &manifest.entries);
    let (prep_data, _vk) =
      setup_batch_proof_rows_preprocessed(&manifest, &pv).expect("setup must succeed");

    // Different manifest: mutate first row's op to get a different VK.
    let mut other_manifest = add_manifest();
    other_manifest.all_rows[0].op = other_manifest.all_rows[0].op.wrapping_add(1);
    let (_, other_vk) =
      setup_batch_proof_rows_preprocessed(&other_manifest, &pv).expect("other setup must succeed");

    let proof = prove_batch_lut_with_prep(&manifest, &prep_data, &pv).expect("prove must succeed");
    let result = verify_batch_lut_with_prep(&proof, &other_vk, &pv);
    assert!(result.is_err(), "verify with wrong VK must fail");
  }

  // ---------------------------------------------------------------------------
  // Test 4: StackIR (single-instruction VK) and batch LUT (batch VK) both
  // verify their respective proofs with the correct VKs.
  // ---------------------------------------------------------------------------
  #[test]
  fn test_stack_ir_and_batch_lut_verify_with_correct_vks() {
    let rows = add_rows();
    let stack_pv = stack_ir_scaffold_public_values();
    let (stack_prep_data, stack_prep_vk) =
      setup_proof_rows_preprocessed(&rows, &stack_pv).expect("setup must succeed");
    let stack_proof = prove_stack_ir_with_prep(&rows, &stack_prep_data, &stack_pv)
      .expect("stack prove must succeed");
    assert!(
      verify_stack_ir_with_prep(&stack_proof, &stack_prep_vk, &stack_pv).is_ok(),
      "StackIR verify must succeed"
    );

    let manifest = add_manifest();
    let lut_pv = make_batch_receipt_binding_public_values(RECEIPT_BIND_TAG_LUT, &manifest.entries);
    let (lut_prep_data, lut_prep_vk) =
      setup_batch_proof_rows_preprocessed(&manifest, &lut_pv).expect("lut setup must succeed");
    let lut_proof = prove_batch_lut_with_prep(&manifest, &lut_prep_data, &lut_pv)
      .expect("lut prove must succeed");
    assert!(
      verify_batch_lut_with_prep(&lut_proof, &lut_prep_vk, &lut_pv).is_ok(),
      "Batch LUT verify must succeed"
    );
  }
}

// =============================================================================
// Batch LUT STARK tests (Phase 1-5)
// =============================================================================
mod batch_lut_tests {
  use revm::bytecode::opcode;
  use zprove_core::semantic_proof::compile_proof;
  use zprove_core::transition::{
    InstructionTransitionProof, InstructionTransitionStatement, VmState, build_batch_manifest,
    prove_batch_transaction_zk_receipt, prove_instruction, verify_batch_transaction_zk_receipt,
  };
  use zprove_core::zk_proof::compute_batch_manifest_digest;

  fn a256(lo: u128) -> [u8; 32] {
    let mut b = [0u8; 32];
    b[16..].copy_from_slice(&lo.to_be_bytes());
    b
  }

  fn make_itp(op: u8, inputs: &[[u8; 32]], output: [u8; 32]) -> InstructionTransitionProof {
    let proof = prove_instruction(op, inputs, &[output]);
    InstructionTransitionProof {
      opcode: op,
      pc: 0,
      stack_inputs: inputs.to_vec(),
      stack_outputs: vec![output],
      semantic_proof: proof,
      memory_claims: vec![],
      storage_claims: vec![],
      stack_claims: vec![],
      return_data_claim: None,
      call_context_claim: None,
      keccak_claim: None,

      external_state_claim: None,
      mcopy_claim: None,
      sub_call_claim: None,
    }
  }

  fn make_stmt(op: u8, inputs: &[[u8; 32]], output: [u8; 32]) -> InstructionTransitionStatement {
    InstructionTransitionStatement {
      opcode: op,
      s_i: VmState {
        opcode: op,
        pc: 0,
        sp: inputs.len(),
        stack: inputs.to_vec(),
        memory_root: [0u8; 32],
        storage_root: [0u8; 32],
      },
      s_next: VmState {
        opcode: op,
        pc: 1,
        sp: 1,
        stack: vec![output],
        memory_root: [0u8; 32],
        storage_root: [0u8; 32],
      },
      accesses: Vec::new(),
      mcopy_claim: None,
      external_state_claim: None,
      sub_call_claim: None,
    }
  }

  // ---------------------------------------------------------------------------
  // Test 1: same inputs produce the same batch manifest digest (determinism).
  // ---------------------------------------------------------------------------
  #[test]
  fn test_batch_manifest_digest_is_deterministic() {
    let add_proof =
      prove_instruction(opcode::ADD, &[a256(5), a256(3)], &[a256(8)]);
    let and_proof =
      prove_instruction(opcode::AND, &[a256(0xF0), a256(0x0F)], &[a256(0)]);

    let items = vec![(opcode::ADD, &add_proof), (opcode::AND, &and_proof)];
    let manifest_a = build_batch_manifest(&items).expect("manifest_a");
    let manifest_b = build_batch_manifest(&items).expect("manifest_b");

    let digest_a = compute_batch_manifest_digest(&manifest_a.entries);
    let digest_b = compute_batch_manifest_digest(&manifest_b.entries);
    assert_eq!(digest_a, digest_b, "same inputs must yield the same digest");
  }

  // ---------------------------------------------------------------------------
  // Test 2: row boundaries are contiguous and cover every row.
  // ---------------------------------------------------------------------------
  #[test]
  fn test_batch_manifest_row_boundaries_are_contiguous() {
    let specs: &[(u8, u128, u128, u128)] =
      &[(opcode::ADD, 7, 2, 9), (opcode::AND, 0xFF, 0x0F, 0x0F)];
    let proofs: Vec<(u8, _)> = specs
      .iter()
      .map(|(op, a, b, out)| {
        let p = prove_instruction(*op, &[a256(*a), a256(*b)], &[a256(*out)]);
        (*op, p)
      })
      .collect();

    let items: Vec<_> = proofs.iter().map(|(op, p)| (*op, p)).collect();
    let manifest = build_batch_manifest(&items).expect("manifest");
    let total = manifest.all_rows.len();

    for (i, entry) in manifest.entries.iter().enumerate() {
      let next_start = if i + 1 < manifest.entries.len() {
        manifest.entries[i + 1].row_start
      } else {
        total
      };
      assert!(entry.row_count > 0, "entry {i} must have at least one row");
      assert_eq!(
        entry.row_start + entry.row_count,
        next_start,
        "entry {i} row range must be contiguous"
      );
    }
    // Verify compile_proof row counts match the manifest.
    for (i, ((_op, proof), entry)) in proofs.iter().zip(manifest.entries.iter()).enumerate() {
      let rows = compile_proof(proof);
      assert_eq!(
        rows.len(),
        entry.row_count,
        "entry {i} row_count must equal compile_proof len"
      );
    }
  }

  // ---------------------------------------------------------------------------
  // Test 3: prove + verify round-trip for a 2-instruction batch (ADD + AND).
  // ---------------------------------------------------------------------------
  #[test]
  fn test_batch_lut_prove_verify_roundtrip() {
    let add_in = [a256(10), a256(20)];
    let and_in = [a256(0xFF), a256(0x0F)];

    let itps = vec![
      make_itp(opcode::ADD, &add_in, a256(30)),
      make_itp(opcode::AND, &and_in, a256(0x0F)),
    ];
    let stmts = vec![
      make_stmt(opcode::ADD, &add_in, a256(30)),
      make_stmt(opcode::AND, &and_in, a256(0x0F)),
    ];

    let receipt = prove_batch_transaction_zk_receipt(&itps)
      .expect("prove_batch_transaction_zk_receipt must succeed");

    assert_eq!(
      receipt.manifest.entries.len(),
      2,
      "manifest must have exactly 2 entries"
    );
    assert!(
      verify_batch_transaction_zk_receipt(&stmts, &receipt),
      "verify_batch_transaction_zk_receipt must return true"
    );
  }

  // ---------------------------------------------------------------------------
  // Test 4: verify rejects when statement count doesn't match manifest.
  // ---------------------------------------------------------------------------
  #[test]
  fn test_batch_verify_rejects_wrong_statement_count() {
    let ins = [a256(3), a256(4)];
    let itp = make_itp(opcode::ADD, &ins, a256(7));
    let receipt = prove_batch_transaction_zk_receipt(&[itp]).expect("prove must succeed");

    // Empty statements — count mismatch ⇒ false.
    assert!(
      !verify_batch_transaction_zk_receipt(&[], &receipt),
      "mismatched statement count must fail"
    );
  }

  // ---------------------------------------------------------------------------
  // Test 5: verify rejects when opcode in statement doesn't match manifest.
  // ---------------------------------------------------------------------------
  #[test]
  fn test_batch_verify_rejects_opcode_mismatch() {
    let ins = [a256(5), a256(6)];
    let itp = make_itp(opcode::ADD, &ins, a256(11));
    let receipt = prove_batch_transaction_zk_receipt(&[itp]).expect("prove must succeed");

    // Statement claims AND but manifest has ADD.
    let bad_stmt = make_stmt(opcode::AND, &[a256(5), a256(6)], a256(4));
    assert!(
      !verify_batch_transaction_zk_receipt(&[bad_stmt], &receipt),
      "opcode mismatch must fail verification"
    );
  }

  // ---------------------------------------------------------------------------
  // Gap 3 test: validate_manifest_rows rejects ByteAnd/Or/Xor with wrong value.
  // ---------------------------------------------------------------------------
  #[test]
  fn test_validate_manifest_rows_gap3_byte_op_wrong_value() {
    use zprove_core::semantic_proof::{OP_BYTE_AND_EQ, OP_BYTE_OR_EQ, OP_BYTE_XOR_EQ, ProofRow};
    use zprove_core::zk_proof::validate_manifest_rows;

    // Correct ByteAnd rows pass.
    let good = vec![
      ProofRow {
        op: OP_BYTE_AND_EQ,
        scalar0: 0xAB,
        scalar1: 0xCD,
        value: 0xAB & 0xCD,
        ..Default::default()
      },
      ProofRow {
        op: OP_BYTE_OR_EQ,
        scalar0: 0x12,
        scalar1: 0x34,
        value: 0x12 | 0x34,
        ..Default::default()
      },
      ProofRow {
        op: OP_BYTE_XOR_EQ,
        scalar0: 0xFF,
        scalar1: 0x0F,
        value: 0xFF ^ 0x0F,
        ..Default::default()
      },
    ];
    assert!(
      validate_manifest_rows(&good),
      "correct byte-op rows should pass"
    );

    // Wrong AND result fails.
    let bad_and = vec![ProofRow {
      op: OP_BYTE_AND_EQ,
      scalar0: 0xAB,
      scalar1: 0xCD,
      value: 0x42, // incorrect: should be 0xAB & 0xCD = 0x89
      ..Default::default()
    }];
    assert!(
      !validate_manifest_rows(&bad_and),
      "wrong AND result must be rejected"
    );

    // Wrong OR result fails.
    let bad_or = vec![ProofRow {
      op: OP_BYTE_OR_EQ,
      scalar0: 0x12,
      scalar1: 0x34,
      value: 0, // incorrect: should be 0x36
      ..Default::default()
    }];
    assert!(
      !validate_manifest_rows(&bad_or),
      "wrong OR result must be rejected"
    );

    // Wrong XOR result fails.
    let bad_xor = vec![ProofRow {
      op: OP_BYTE_XOR_EQ,
      scalar0: 0xFF,
      scalar1: 0xFF,
      value: 0xFF, // incorrect: 0xFF ^ 0xFF = 0x00
      ..Default::default()
    }];
    assert!(
      !validate_manifest_rows(&bad_xor),
      "wrong XOR result must be rejected"
    );

    // Out-of-range input fails.
    let bad_input = vec![ProofRow {
      op: OP_BYTE_AND_EQ,
      scalar0: 256,
      scalar1: 1,
      value: 0,
      ..Default::default()
    }];
    assert!(
      !validate_manifest_rows(&bad_input),
      "out-of-byte-range input must be rejected"
    );
  }

  // ---------------------------------------------------------------------------
  // Gap 4 test: validate_manifest_rows rejects out-of-range arithmetic operands.
  // ---------------------------------------------------------------------------
  #[test]
  fn test_validate_manifest_rows_gap4_arithmetic_range() {
    use zprove_core::semantic_proof::{OP_U24_ADD_EQ, OP_U29_ADD_EQ, ProofRow};
    use zprove_core::zk_proof::validate_manifest_rows;

    const MAX29: u32 = (1u32 << 29) - 1;
    const MAX24: u32 = (1u32 << 24) - 1;

    // Correct U29AddEq row passes.
    let good_u29 = vec![ProofRow {
      op: OP_U29_ADD_EQ,
      scalar0: MAX29,
      scalar1: 0,
      scalar2: 0,
      value: MAX29,
      arg1: 0,
      ..Default::default()
    }];
    assert!(validate_manifest_rows(&good_u29), "valid U29 row must pass");

    // in0 exceeds 2^29 - 1 → rejected.
    let bad_in0 = vec![ProofRow {
      op: OP_U29_ADD_EQ,
      scalar0: MAX29 + 1,
      scalar1: 0,
      scalar2: 0,
      value: 1,
      arg1: 0,
      ..Default::default()
    }];
    assert!(
      !validate_manifest_rows(&bad_in0),
      "U29 in0 > max must be rejected"
    );

    // out0 exceeds range → rejected.
    let bad_out0 = vec![ProofRow {
      op: OP_U29_ADD_EQ,
      scalar0: 0,
      scalar1: 0,
      scalar2: 0,
      value: MAX29 + 1,
      arg1: 0,
      ..Default::default()
    }];
    assert!(
      !validate_manifest_rows(&bad_out0),
      "U29 out0 > max must be rejected"
    );

    // carry-in > 1 → rejected.
    let bad_carry = vec![ProofRow {
      op: OP_U29_ADD_EQ,
      scalar0: 0,
      scalar1: 0,
      scalar2: 2, // carry-in = 2
      value: 0,
      arg1: 0,
      ..Default::default()
    }];
    assert!(
      !validate_manifest_rows(&bad_carry),
      "U29 carry-in > 1 must be rejected"
    );

    // Correct U24AddEq row passes.
    let good_u24 = vec![ProofRow {
      op: OP_U24_ADD_EQ,
      scalar0: MAX24,
      scalar1: 0,
      scalar2: 0,
      value: MAX24,
      arg1: 0,
      ..Default::default()
    }];
    assert!(validate_manifest_rows(&good_u24), "valid U24 row must pass");

    // U24 in1 exceeds 2^24 - 1 → rejected.
    let bad_u24 = vec![ProofRow {
      op: OP_U24_ADD_EQ,
      scalar0: 0,
      scalar1: MAX24 + 1,
      scalar2: 0,
      value: 0,
      arg1: 0,
      ..Default::default()
    }];
    assert!(
      !validate_manifest_rows(&bad_u24),
      "U24 in1 > max must be rejected"
    );
  }

  // ---------------------------------------------------------------------------
  // Gap 3 end-to-end: AND batch receipt roundtrip uses correct byte-op values.
  // ---------------------------------------------------------------------------
  #[test]
  fn test_batch_and_receipt_roundtrip_correct_values() {
    // AND of two non-trivial 32-byte values.
    let a: [u8; 32] = {
      let mut v = [0u8; 32];
      for i in 0..32 {
        v[i] = ((i as u8).wrapping_mul(17)).wrapping_add(3);
      }
      v
    };
    let b: [u8; 32] = {
      let mut v = [0u8; 32];
      for i in 0..32 {
        v[i] = ((i as u8).wrapping_mul(31)).wrapping_add(7);
      }
      v
    };
    let expected: [u8; 32] = std::array::from_fn(|i| a[i] & b[i]);

    let itp = make_itp(opcode::AND, &[a, b], expected);
    let receipt = prove_batch_transaction_zk_receipt(&[itp]).expect("prove must succeed");

    // All manifest rows must pass validation (Gap 3 + Gap 4).
    assert!(
      zprove_core::zk_proof::validate_manifest_rows(&receipt.manifest.all_rows),
      "AND batch manifest rows must pass validation"
    );

    let stmt = make_stmt(opcode::AND, &[a, b], expected);
    assert!(
      verify_batch_transaction_zk_receipt(&[stmt], &receipt),
      "AND batch receipt roundtrip must verify"
    );
  }
}

// ============================================================
// Keccak ↔ memory cross-check tests  (BUG-MISS-3)
// ============================================================

#[cfg(test)]
mod keccak_memory_cross_check_tests {
  use zprove_core::transition::KeccakClaim;
  use zprove_core::zk_proof::{
    KeccakLogEntry, MemLogEntry, keccak256_bytes, prove_keccak_consistency,
    validate_keccak_memory_cross_check, verify_keccak_consistency,
  };

  fn mem_entry(addr: u64, value: [u8; 32]) -> MemLogEntry {
    MemLogEntry {
      rw_counter: 0,
      addr,
      value,
    }
  }

  fn keccak_claim(offset: u64, input: &[u8]) -> KeccakClaim {
    KeccakClaim {
      offset,
      size: input.len() as u64,
      input_bytes: input.to_vec(),
      output_hash: keccak256_bytes(input),
    }
  }

  // ── prove / verify roundtrip ──────────────────────────────────────

  #[test]
  fn keccak_prove_verify_roundtrip() {
    let claims = vec![keccak_claim(0, b"hello world")];
    let proof = prove_keccak_consistency(&claims).expect("prove must succeed");
    assert!(verify_keccak_consistency(&proof), "roundtrip must verify");
  }

  #[test]
  fn keccak_empty_input_roundtrip() {
    let claims = vec![keccak_claim(0, b"")];
    let proof = prove_keccak_consistency(&claims).expect("prove must succeed");
    assert!(verify_keccak_consistency(&proof));
  }

  #[test]
  fn keccak_multiple_claims_roundtrip() {
    let claims = vec![keccak_claim(0, b"foo"), keccak_claim(32, b"bar")];
    let proof = prove_keccak_consistency(&claims).expect("prove must succeed");
    assert!(verify_keccak_consistency(&proof));
  }

  // ── cross-check: correct memory ──────────────────────────────────

  #[test]
  fn cross_check_single_word_exact_match() {
    // keccak of exactly one 32-byte aligned word (offset=0, size=32)
    let mut value = [0u8; 32];
    value.copy_from_slice(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ012345");
    let log = vec![KeccakLogEntry {
      offset: 0,
      size: 32,
      input_bytes: value.to_vec(),
      output_hash: keccak256_bytes(&value),
      memory_words: vec![],
    }];
    let write_log = vec![mem_entry(0, value)];
    assert!(validate_keccak_memory_cross_check(&log, &write_log, &[]));
  }

  #[test]
  fn cross_check_subword_range() {
    // keccak of bytes [4, 20) inside a single 32-byte word at addr 0
    let mut word_val = [0u8; 32];
    for i in 0..32 {
      word_val[i] = i as u8;
    }
    let input_bytes = word_val[4..20].to_vec();
    let log = vec![KeccakLogEntry {
      offset: 4,
      size: 16,
      input_bytes: input_bytes.clone(),
      output_hash: keccak256_bytes(&input_bytes),
      memory_words: vec![],
    }];
    let write_log = vec![mem_entry(0, word_val)];
    assert!(validate_keccak_memory_cross_check(&log, &write_log, &[]));
  }

  #[test]
  fn cross_check_spanning_two_words() {
    // keccak spans bytes [28, 36) — last 4 bytes of word 0 + first 4 bytes of word 32
    let mut w0 = [0u8; 32];
    for i in 0..32 {
      w0[i] = i as u8;
    }
    let mut w1 = [0u8; 32];
    for i in 0..32 {
      w1[i] = (i + 32) as u8;
    }
    let mut input_bytes = Vec::new();
    input_bytes.extend_from_slice(&w0[28..32]);
    input_bytes.extend_from_slice(&w1[0..4]);
    let log = vec![KeccakLogEntry {
      offset: 28,
      size: 8,
      input_bytes: input_bytes.clone(),
      output_hash: keccak256_bytes(&input_bytes),
      memory_words: vec![],
    }];
    let write_log = vec![mem_entry(0, w0), mem_entry(32, w1)];
    assert!(validate_keccak_memory_cross_check(&log, &write_log, &[]));
  }

  #[test]
  fn cross_check_empty_keccak_log() {
    // No keccak claims → always passes
    assert!(validate_keccak_memory_cross_check(&[], &[], &[]));
  }

  // ── cross-check: tampered memory — must reject ───────────────────

  #[test]
  fn cross_check_wrong_memory_value_fails() {
    // Memory has value X but keccak claims to have hashed Y
    let correct_bytes: Vec<u8> = (0u8..32).collect();
    let mut wrong_word = [0u8; 32];
    wrong_word.copy_from_slice(&(32u8..64).collect::<Vec<_>>()[..32]);

    let log = vec![KeccakLogEntry {
      offset: 0,
      size: 32,
      input_bytes: correct_bytes.clone(),
      output_hash: keccak256_bytes(&correct_bytes),
      memory_words: vec![],
    }];
    let write_log = vec![mem_entry(0, wrong_word)]; // wrong bytes in memory
    assert!(
      !validate_keccak_memory_cross_check(&log, &write_log, &[]),
      "tampered memory must fail cross-check"
    );
  }

  #[test]
  fn cross_check_missing_memory_word_fails() {
    // Memory log doesn't contain the word keccak reads
    let input_bytes: Vec<u8> = (0u8..32).collect();
    let log = vec![KeccakLogEntry {
      offset: 0,
      size: 32,
      input_bytes: input_bytes.clone(),
      output_hash: keccak256_bytes(&input_bytes),
      memory_words: vec![],
    }];
    // write_log / read_log are both empty
    assert!(
      !validate_keccak_memory_cross_check(&log, &[], &[]),
      "missing memory word must fail cross-check"
    );
  }

  #[test]
  fn cross_check_tampered_input_bytes_fails() {
    // KeccakLogEntry.input_bytes tampered AFTER proof — cross-check catches it
    let real_bytes: Vec<u8> = (0u8..32).collect();
    let mut real_word = [0u8; 32];
    real_word.copy_from_slice(&real_bytes);

    let mut tampered_bytes: Vec<u8> = real_bytes.clone();
    tampered_bytes[0] = 0xFF; // tamper

    let log = vec![KeccakLogEntry {
      offset: 0,
      size: 32,
      input_bytes: tampered_bytes, // doesn't match memory
      output_hash: keccak256_bytes(&real_bytes),
      memory_words: vec![],
    }];
    let write_log = vec![mem_entry(0, real_word)];
    assert!(
      !validate_keccak_memory_cross_check(&log, &write_log, &[]),
      "tampered input_bytes must fail cross-check"
    );
  }

  #[test]
  fn cross_check_uses_read_log_for_cross_batch_reads() {
    // Word is provided via read_log (cross-batch), not write_log
    let value: Vec<u8> = (0u8..32).collect();
    let mut word_val = [0u8; 32];
    word_val.copy_from_slice(&value);
    let log = vec![KeccakLogEntry {
      offset: 0,
      size: 32,
      input_bytes: value.clone(),
      output_hash: keccak256_bytes(&value),
      memory_words: vec![],
    }];
    let read_log = vec![mem_entry(0, word_val)];
    assert!(validate_keccak_memory_cross_check(&log, &[], &read_log));
  }

  #[test]
  fn cross_check_write_overrides_read_log() {
    // Same address in both read_log and write_log — write_log value wins
    let mut read_val = [0u8; 32]; // "old" value
    read_val[0] = 0xAA;
    let mut write_val = [0u8; 32]; // "new" value after write
    write_val[0] = 0xBB;

    // keccak reads the WRITTEN (new) value
    let log = vec![KeccakLogEntry {
      offset: 0,
      size: 1,
      input_bytes: vec![0xBB],
      output_hash: keccak256_bytes(&[0xBB]),
      memory_words: vec![],
    }];
    let write_log = vec![mem_entry(0, write_val)];
    let read_log = vec![mem_entry(0, read_val)];
    assert!(validate_keccak_memory_cross_check(
      &log, &write_log, &read_log
    ));
  }
}
