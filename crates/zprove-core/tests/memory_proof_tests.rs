// moved from src/memory_proof.rs

#[cfg(test)]
mod tests {
  use revm::bytecode::opcode;
  use zprove_core::memory_proof::{
    CqRw, build_memory_cq_events, compute_memory_root, verify_memory_access_commitments,
    verify_memory_cq_constraints,
  };
  use zprove_core::transition::{
    AccessDomain, AccessKind, AccessRecord, InstructionTransitionStatement, VmState,
  };

  fn u256_bytes(val: u128) -> [u8; 32] {
    let mut b = [0u8; 32];
    b[16..32].copy_from_slice(&val.to_be_bytes());
    b
  }

  fn make_statement(
    accesses: Vec<AccessRecord>,
    root_before: [u8; 32],
    root_after: [u8; 32],
  ) -> InstructionTransitionStatement {
    InstructionTransitionStatement {
      opcode: opcode::MSTORE,
      s_i: VmState {
        opcode: opcode::MSTORE,
        pc: 0,
        sp: 2,
        stack: vec![u256_bytes(64), [9u8; 32]],
        memory_root: root_before,
      },
      s_next: VmState {
        opcode: opcode::MSTORE,
        pc: 1,
        sp: 0,
        stack: vec![],
        memory_root: root_after,
      },
      accesses,
    }
  }

  #[test]
  fn test_build_memory_cq_events_for_read_write() {
    let before = [1u8; 32];
    let after = [2u8; 32];
    let statement = make_statement(
      vec![AccessRecord {
        rw_counter: 7,
        domain: AccessDomain::Memory,
        kind: AccessKind::ReadWrite,
        addr: 64,
        width: 32,
        value_before: Some(before),
        value_after: Some(after),
        merkle_path_before: vec![[3u8; 32]],
        merkle_path_after: vec![[3u8; 32]],
      }],
      [0u8; 32],
      [0u8; 32],
    );

    let events = build_memory_cq_events(&statement).expect("cq event build should succeed");
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].rw, CqRw::Read);
    assert_eq!(events[0].step, 7);
    assert_eq!(events[1].rw, CqRw::Write);
    assert_eq!(events[1].step, 7);
    assert_eq!(events[1].value, after);
  }

  #[test]
  fn test_verify_memory_cq_constraints_rejects_discontinuous_read() {
    let first_written = [7u8; 32];
    let wrong_read = [8u8; 32];
    let accesses = vec![
      AccessRecord {
        rw_counter: 1,
        domain: AccessDomain::Memory,
        kind: AccessKind::Write,
        addr: 64,
        width: 32,
        value_before: Some([0u8; 32]),
        value_after: Some(first_written),
        merkle_path_before: vec![[1u8; 32]],
        merkle_path_after: vec![[1u8; 32]],
      },
      AccessRecord {
        rw_counter: 2,
        domain: AccessDomain::Memory,
        kind: AccessKind::Read,
        addr: 64,
        width: 32,
        value_before: Some(wrong_read),
        value_after: None,
        merkle_path_before: vec![[2u8; 32]],
        merkle_path_after: Vec::new(),
      },
    ];

    let statement = make_statement(accesses, [0u8; 32], [0u8; 32]);
    assert!(!verify_memory_cq_constraints(&statement));
  }

  #[test]
  fn test_verify_memory_access_commitments_enforces_cq_constraints() {
    let path = vec![[9u8; 32], [10u8; 32]];
    let zero = [0u8; 32];
    let first = [0xAA; 32];
    let bad_read = [0xBB; 32];

    let root_before = compute_memory_root(64, 32, &zero, &path);
    let root_after_first_write = compute_memory_root(64, 32, &first, &path);

    let statement = make_statement(
      vec![
        AccessRecord {
          rw_counter: 1,
          domain: AccessDomain::Memory,
          kind: AccessKind::Write,
          addr: 64,
          width: 32,
          value_before: Some(zero),
          value_after: Some(first),
          merkle_path_before: path.clone(),
          merkle_path_after: path.clone(),
        },
        AccessRecord {
          rw_counter: 2,
          domain: AccessDomain::Memory,
          kind: AccessKind::Read,
          addr: 64,
          width: 32,
          value_before: Some(bad_read),
          value_after: None,
          merkle_path_before: path,
          merkle_path_after: Vec::new(),
        },
      ],
      root_before,
      root_after_first_write,
    );

    assert!(!verify_memory_access_commitments(&statement));
  }
}
