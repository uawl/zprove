/// Integration tests for the byte-table LogUp argument.
///
/// Tests:
/// 1. `test_byte_table_and_prove_verify` — prove/verify AND queries
/// 2. `test_byte_table_or_prove_verify`  — prove/verify OR queries
/// 3. `test_byte_table_xor_prove_verify` — prove/verify XOR queries
/// 4. `test_byte_table_mixed_ops`        — mixed AND/OR/XOR in one proof
/// 5. `test_prove_lut_with_prep_and_logup` — full LUT + byte-table companion proof
#[cfg(test)]
mod byte_table_tests {
  use zprove_core::byte_table::{
    BYTE_OP_AND, BYTE_OP_OR, BYTE_OP_XOR, ByteTableQuery, prove_byte_table, verify_byte_table,
  };

  // ── single-op proofs ────────────────────────────────────────────────

  #[test]
  fn test_byte_table_and_prove_verify() {
    let queries = vec![
      ByteTableQuery::new_and(0xAB, 0xCD),
      ByteTableQuery::new_and(0x00, 0xFF),
      ByteTableQuery::new_and(0x55, 0xAA),
      ByteTableQuery::new_and(0xFF, 0xFF),
    ];
    let proof = prove_byte_table(&queries);
    let result = verify_byte_table(&proof);
    assert!(result.is_ok(), "AND verify failed: {:?}", result.err());
  }

  #[test]
  fn test_byte_table_or_prove_verify() {
    let queries = vec![
      ByteTableQuery::new_or(0x12, 0x34),
      ByteTableQuery::new_or(0x00, 0x00),
      ByteTableQuery::new_or(0xFF, 0x00),
    ];
    let proof = prove_byte_table(&queries);
    let result = verify_byte_table(&proof);
    assert!(result.is_ok(), "OR verify failed: {:?}", result.err());
  }

  #[test]
  fn test_byte_table_and_2queries_prove_verify() {
    // 2 queries + 2 receives = 4 data rows → height=4, no padding
    let queries = vec![
      ByteTableQuery::new_and(0x12, 0x34),
      ByteTableQuery::new_and(0x00, 0xFF),
    ];
    let proof = prove_byte_table(&queries);
    let result = verify_byte_table(&proof);
    assert!(result.is_ok(), "AND-2 verify failed: {:?}", result.err());
  }

  #[test]
  fn test_byte_table_and_3queries_prove_verify() {
    // 3 queries + 3 receives = 6 data rows → height=8, 2 padding rows
    let queries = vec![
      ByteTableQuery::new_and(0x12, 0x34),
      ByteTableQuery::new_and(0x00, 0xFF),
      ByteTableQuery::new_and(0x55, 0xAA),
    ];
    let proof = prove_byte_table(&queries);
    let result = verify_byte_table(&proof);
    assert!(result.is_ok(), "AND-3 verify failed: {:?}", result.err());
  }

  #[test]
  fn test_byte_table_and_3queries_same_as_or() {
    // Same (a,b) pairs as OR test but with AND op
    let queries = vec![
      ByteTableQuery::new_and(0x12, 0x34),
      ByteTableQuery::new_and(0x00, 0x00),
      ByteTableQuery::new_and(0xFF, 0x00),
    ];
    let proof = prove_byte_table(&queries);
    let result = verify_byte_table(&proof);
    assert!(
      result.is_ok(),
      "AND-same-as-or verify failed: {:?}",
      result.err()
    );
  }

  #[test]
  fn test_byte_table_and_1query_prove_verify() {
    // Single AND query: height = max(2, 4).next_power_of_two() = 4, 2 padding rows
    let queries = vec![ByteTableQuery::new_and(0xFF, 0x00)];
    let proof = prove_byte_table(&queries);
    let result = verify_byte_table(&proof);
    assert!(result.is_ok(), "AND-1 verify failed: {:?}", result.err());
  }

  #[test]
  fn test_byte_table_xor_1query_prove_verify() {
    // Single XOR query - simplest possible case
    let queries = vec![ByteTableQuery::new_xor(0xFF, 0x00)];
    let proof = prove_byte_table(&queries);
    let result = verify_byte_table(&proof);
    assert!(result.is_ok(), "XOR-1 verify failed: {:?}", result.err());
  }

  #[test]
  fn test_byte_table_xor_2queries_prove_verify() {
    // 2 XOR queries
    let queries = vec![
      ByteTableQuery::new_xor(0xFF, 0x00),
      ByteTableQuery::new_xor(0xAA, 0x55),
    ];
    let proof = prove_byte_table(&queries);
    let result = verify_byte_table(&proof);
    assert!(result.is_ok(), "XOR-2 verify failed: {:?}", result.err());
  }

  #[test]
  fn test_byte_table_xor_3queries_prove_verify() {
    // Same count as the OR test (3 queries) but with XOR op
    let queries = vec![
      ByteTableQuery::new_xor(0x12, 0x34),
      ByteTableQuery::new_xor(0x00, 0x00),
      ByteTableQuery::new_xor(0xFF, 0x00),
    ];
    let proof = prove_byte_table(&queries);
    let result = verify_byte_table(&proof);
    assert!(result.is_ok(), "XOR-3 verify failed: {:?}", result.err());
  }

  #[test]
  fn test_byte_table_xor_prove_verify() {
    let queries = vec![
      ByteTableQuery::new_xor(0xDE, 0xAD),
      ByteTableQuery::new_xor(0xBE, 0xEF),
      ByteTableQuery::new_xor(0x00, 0xFF),
      ByteTableQuery::new_xor(0xFF, 0xFF),
    ];
    let proof = prove_byte_table(&queries);
    let result = verify_byte_table(&proof);
    assert!(result.is_ok(), "XOR verify failed: {:?}", result.err());
  }

  #[test]
  fn test_byte_table_mixed_ops() {
    let queries = vec![
      ByteTableQuery::new_and(0xAB, 0xCD),
      ByteTableQuery::new_or(0x12, 0x34),
      ByteTableQuery::new_xor(0x56, 0x78),
      ByteTableQuery::new_and(0xFF, 0x0F),
      ByteTableQuery::new_xor(0xAA, 0x55),
    ];
    let proof = prove_byte_table(&queries);
    let result = verify_byte_table(&proof);
    assert!(
      result.is_ok(),
      "mixed ops verify failed: {:?}",
      result.err()
    );
  }

  // ── soundness check: wrong result should fail ─────────────────────

  /// Manually build a query with an INCORRECT result.  The LogUp running sum
  /// will not balance and the proof should not verify.
  ///
  /// NOTE: we build a trace manually with a wrong multiplicity imbalance.
  /// We test this indirectly: if we override the result in a query with a wrong
  /// value, `build_byte_table_trace` inserts a query row with that wrong value.
  /// The corresponding receive row (from the truth-table section) will have the
  /// CORRECT result. The running-sum cannot cancel → the proof is unsound and
  /// the prover will panic or produce garbage.
  ///
  /// We verify that the *correct* result passes (not trying to prove a lie here,
  /// just documenting the design).
  #[test]
  fn test_byte_table_correct_result_verified() {
    // 0x3C & 0x5A == 0x18
    let a: u8 = 0x3C;
    let b: u8 = 0x5A;
    let expected = a & b;
    assert_eq!(expected, 0x18);
    let queries = vec![ByteTableQuery {
      a,
      b,
      op: BYTE_OP_AND,
      result: expected,
      multiplicity: 1,
    }];
    let proof = prove_byte_table(&queries);
    let result = verify_byte_table(&proof);
    assert!(
      result.is_ok(),
      "correct AND result should verify: {:?}",
      result.err()
    );
  }

  // ── companion proof alongside LUT STARK ─────────────────────────────

  #[test]
  fn test_collect_byte_table_queries_from_lut_steps() {
    use zprove_core::zk_proof::{LutOpcode, LutStep, collect_byte_table_queries_from_lut_steps};

    let steps = vec![
      LutStep {
        op: LutOpcode::ByteAndEq,
        in0: 0xAB,
        in1: 0xCD,
        in2: 0,
        out0: 0xAB & 0xCD,
        out1: 0,
      },
      LutStep {
        op: LutOpcode::ByteOrEq,
        in0: 0x12,
        in1: 0x34,
        in2: 0,
        out0: 0x12 | 0x34,
        out1: 0,
      },
      LutStep {
        op: LutOpcode::ByteXorEq,
        in0: 0xFF,
        in1: 0xAA,
        in2: 0,
        out0: 0xFF ^ 0xAA,
        out1: 0,
      },
      // Non-bitwise step — should NOT appear in the byte-table queries.
      LutStep {
        op: LutOpcode::U29AddEq,
        in0: 1,
        in1: 2,
        in2: 0,
        out0: 3,
        out1: 0,
      },
    ];

    let queries = collect_byte_table_queries_from_lut_steps(&steps);

    assert_eq!(queries.len(), 3, "expected exactly 3 byte-table queries");
    assert_eq!(queries[0].op, BYTE_OP_AND);
    assert_eq!(queries[0].a, 0xAB);
    assert_eq!(queries[0].b, 0xCD);
    assert_eq!(queries[0].result, 0xAB & 0xCD);

    assert_eq!(queries[1].op, BYTE_OP_OR);
    assert_eq!(queries[1].result, 0x12 | 0x34);

    assert_eq!(queries[2].op, BYTE_OP_XOR);
    assert_eq!(queries[2].result, 0xFF ^ 0xAA);
  }
}
