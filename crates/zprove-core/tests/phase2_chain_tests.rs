/// Phase 2 통합 테스트 — Segment Splitting & LinkAir STARK Commitment Chain
///
/// 테스트 대상:
/// - `prove_execution_chain` / `verify_execution_receipt` (transition.rs)
/// - `commit_vm_state` / `prove_link_stark` / `verify_link_stark` (recursive.rs)
/// - `execute_bytecode_and_prove_chain` (execute.rs)
#[cfg(test)]
mod phase2_chain_tests {
  use revm::primitives::{Bytes, U256};
  use zprove_core::execute::execute_bytecode_and_prove_chain;
  use zprove_core::transition::{
    ExecutionReceipt, InstructionTransitionProof, VmState, prove_execution_chain,
    verify_execution_receipt,
  };
  use zprove_core::zk_proof::{commit_vm_state, prove_link_stark, verify_link_stark};

  // --------------------------------------------------------------------------
  // 헬퍼 함수
  // --------------------------------------------------------------------------

  /// 최소 VmState — 스택 0개, 빈 메모리 루트
  fn make_vm_state(pc: usize, sp: usize) -> VmState {
    VmState {
      opcode: 0x01, // ADD
      pc,
      sp,
      stack: vec![[0u8; 32]; sp],
      memory_root: [0u8; 32],
    }
  }

  /// 빈 단계 시퀀스 (arithmetic 없이 structural-only) — batch_receipt = None
  fn empty_steps() -> Vec<InstructionTransitionProof> {
    vec![]
  }

  // --------------------------------------------------------------------------
  // Test 1: StateCommitment 직렬화 / 일관성
  // --------------------------------------------------------------------------

  #[test]
  fn test_state_commitment_to_fields_consistency() {
    let state = make_vm_state(10, 2);
    let comm = commit_vm_state(&state);

    // pc / sp 가 StateCommitment에 반영됐는지 확인
    assert_eq!(comm.pc, 10u32);
    assert_eq!(comm.sp, 2u32);

    // to_fields() 길이 = 18 (pc, sp, stack_hash[8], memory_root[8])
    let fields = comm.to_fields();
    assert_eq!(fields.len(), 18);

    // 동일 VmState → 동일 commitment
    let comm2 = commit_vm_state(&state);
    assert_eq!(comm.pc, comm2.pc);
    assert_eq!(comm.sp, comm2.sp);
    assert_eq!(comm.stack_hash, comm2.stack_hash);
    assert_eq!(comm.memory_root, comm2.memory_root);

    // 다른 pc → 다른 commitment
    let state_diff = make_vm_state(99, 2);
    let comm_diff = commit_vm_state(&state_diff);
    assert_ne!(comm.pc, comm_diff.pc);
  }

  // --------------------------------------------------------------------------
  // Test 2: prove_link_stark / verify_link_stark 라운드트립
  // --------------------------------------------------------------------------

  #[test]
  fn test_link_stark_roundtrip() {
    let s_out = commit_vm_state(&make_vm_state(42, 3));
    let s_in = s_out.clone(); // 이어진 세그먼트 → s_out == s_in

    let junction = [(s_out.clone(), s_in.clone())];
    let proof = prove_link_stark(&junction, &s_out, &s_in);

    let result = verify_link_stark(&proof, &s_out, &s_in);
    assert!(result.is_ok(), "LinkAir verify 실패: {:?}", result.err());
  }

  // --------------------------------------------------------------------------
  // Test 3: 단일 세그먼트 → LeafReceipt (집계 없음)
  // --------------------------------------------------------------------------

  #[test]
  fn test_single_segment_yields_leaf_receipt() {
    let s0 = make_vm_state(0, 0);
    let s1 = make_vm_state(1, 0);

    let receipt = prove_execution_chain(&[s0, s1], vec![empty_steps()])
      .expect("단일 세그먼트 prove 실패");

    match &receipt {
      ExecutionReceipt::Leaf(_) => {} // 기대값
      ExecutionReceipt::Aggregated(_) => panic!("단일 세그먼트는 Leaf여야 합니다"),
    }

    // 검증도 통과해야 함
    verify_execution_receipt(&receipt)
      .expect("단일 세그먼트 Leaf 검증 실패");
  }

  // --------------------------------------------------------------------------
  // Test 4: 두 세그먼트 → AggregationReceipt (LinkAir STARK 포함)
  // --------------------------------------------------------------------------

  #[test]
  fn test_two_segment_chain_yields_aggregation_receipt() {
    let s0 = make_vm_state(0, 2);
    let s1 = make_vm_state(5, 2);
    let s2 = make_vm_state(10, 2);

    let receipt = prove_execution_chain(
      &[s0.clone(), s1.clone(), s2.clone()],
      vec![empty_steps(), empty_steps()],
    )
    .expect("2-세그먼트 prove 실패");

    match &receipt {
      ExecutionReceipt::Aggregated(agg) => {
        // 경계 상태 확인 (pc 기준)
        assert_eq!(agg.s_in.pc, 0u32);
        assert_eq!(agg.s_out.pc, 10u32);
      }
      ExecutionReceipt::Leaf(_) => panic!("2-세그먼트는 Aggregated여야 합니다"),
    }

    // end-to-end 검증
    verify_execution_receipt(&receipt)
      .expect("2-세그먼트 AggregationReceipt 검증 실패");
  }

  // --------------------------------------------------------------------------
  // Test 5: 세 세그먼트 → 트리 집계 (홀수 노드 carry-forward)
  // --------------------------------------------------------------------------

  #[test]
  fn test_three_segment_chain_aggregation() {
    let states: Vec<VmState> = (0..=3).map(|i| make_vm_state(i * 10, 0)).collect();
    let step_seqs = vec![empty_steps(), empty_steps(), empty_steps()];

    let receipt = prove_execution_chain(&states, step_seqs)
      .expect("3-세그먼트 prove 실패");

    // 결과는 반드시 Aggregated여야 함 (2쌍 → 1 + carry → 최종 Aggregated)
    match &receipt {
      ExecutionReceipt::Aggregated(agg) => {
        assert_eq!(agg.s_in.pc, 0u32);
        assert_eq!(agg.s_out.pc, 30u32);
      }
      ExecutionReceipt::Leaf(_) => panic!("3-세그먼트는 Aggregated여야 합니다"),
    }

    verify_execution_receipt(&receipt)
      .expect("3-세그먼트 receipt 검증 실패");
  }

  // --------------------------------------------------------------------------
  // Test 6: 네 세그먼트 → 완전 이진 트리
  // --------------------------------------------------------------------------

  #[test]
  fn test_four_segment_chain_full_binary_tree() {
    let states: Vec<VmState> = (0..=4).map(|i| make_vm_state(i * 5, 0)).collect();
    let step_seqs = vec![
      empty_steps(),
      empty_steps(),
      empty_steps(),
      empty_steps(),
    ];

    let receipt = prove_execution_chain(&states, step_seqs)
      .expect("4-세그먼트 prove 실패");

    match &receipt {
      ExecutionReceipt::Aggregated(agg) => {
        assert_eq!(agg.s_in.pc, 0u32);
        assert_eq!(agg.s_out.pc, 20u32);
      }
      ExecutionReceipt::Leaf(_) => panic!("4-세그먼트는 Aggregated여야 합니다"),
    }

    verify_execution_receipt(&receipt)
      .expect("4-세그먼트 receipt 검증 실패");
  }

  // --------------------------------------------------------------------------
  // Test 7: 입력 오류 처리 — 세그먼트 0개
  // --------------------------------------------------------------------------

  #[test]
  fn test_prove_execution_chain_empty_segments_error() {
    let s0 = make_vm_state(0, 0);
    let result = prove_execution_chain(&[s0], vec![]);
    assert!(result.is_err(), "빈 세그먼트는 에러여야 합니다");
  }

  // --------------------------------------------------------------------------
  // Test 8: 입력 오류 처리 — 경계 상태 수 불일치
  // --------------------------------------------------------------------------

  #[test]
  fn test_prove_execution_chain_boundary_mismatch_error() {
    let s0 = make_vm_state(0, 0);
    let s1 = make_vm_state(1, 0);
    // step_seqs 2개인데 vm_state_seq는 2개 (n+1=3 필요)
    let result = prove_execution_chain(&[s0, s1], vec![empty_steps(), empty_steps()]);
    assert!(result.is_err(), "경계 수 불일치는 에러여야 합니다");
  }

  // --------------------------------------------------------------------------
  // Test 9: execute_bytecode_and_prove_chain — STOP 바이트코드 (window_size=1)
  // --------------------------------------------------------------------------

  #[test]
  fn test_execute_bytecode_stop_window_1() {
    // EVM STOP (0x00) — 한 스텝만 실행됨
    let bytecode = Bytes::from(vec![0x00]);
    let result = execute_bytecode_and_prove_chain(bytecode, Bytes::new(), U256::ZERO, 1);
    // STOP은 스텝이 없을 수 있으므로 에러 허용, 아니면 Leaf
    match result {
      Ok(receipt) => {
        verify_execution_receipt(&receipt)
          .expect("STOP bytecode receipt 검증 실패");
      }
      Err(_) => {
        // execute_bytecode_and_prove_chain: no execution steps — 정상 에러
      }
    }
  }

  // --------------------------------------------------------------------------
  // Test 10: execute_bytecode_and_prove_chain — PUSH1 + PUSH1 + ADD (window_size=2)
  // --------------------------------------------------------------------------

  #[test]
  fn test_execute_bytecode_add_window_size_2() {
    // PUSH1 0x03, PUSH1 0x04, ADD, POP, STOP
    let bytecode = Bytes::from(vec![
      0x60, 0x03, // PUSH1 3
      0x60, 0x04, // PUSH1 4
      0x01, // ADD
      0x50, // POP
      0x00, // STOP
    ]);

    let receipt = execute_bytecode_and_prove_chain(
      bytecode,
      Bytes::new(),
      U256::ZERO,
      2, // window_size = 2 → 여러 세그먼트로 분할
    )
    .expect("PUSH+ADD+POP 바이트코드 prove 실패");

    verify_execution_receipt(&receipt)
      .expect("PUSH+ADD+POP receipt 검증 실패");
  }

  // --------------------------------------------------------------------------
  // Test 11: execute_bytecode_and_prove_chain — window_size=1 (최대 분할)
  // --------------------------------------------------------------------------

  #[test]
  fn test_execute_bytecode_add_window_size_1() {
    let bytecode = Bytes::from(vec![
      0x60, 0x05, // PUSH1 5
      0x60, 0x06, // PUSH1 6
      0x01, // ADD
      0x50, // POP
      0x00, // STOP
    ]);

    let receipt = execute_bytecode_and_prove_chain(
      bytecode,
      Bytes::new(),
      U256::ZERO,
      1, // 스텝당 1개 세그먼트
    )
    .expect("window_size=1 prove 실패");

    verify_execution_receipt(&receipt)
      .expect("window_size=1 receipt 검증 실패");
  }
}
