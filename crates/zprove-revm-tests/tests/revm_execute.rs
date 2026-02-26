use revm::{
  bytecode::opcode,
  primitives::{Bytes, U256},
};
use zprove_core::execute::{
  execute_bytecode_and_prove, execute_bytecode_and_prove_with_zkp_parallel, execute_bytecode_trace,
};

fn u256_bytes_u128(v: u128) -> [u8; 32] {
  let mut out = [0u8; 32];
  out[16..].copy_from_slice(&v.to_be_bytes());
  out
}

fn i256_bytes_i128(v: i128) -> [u8; 32] {
  let mut out = if v < 0 { [0xFFu8; 32] } else { [0u8; 32] };
  out[16..].copy_from_slice(&v.to_be_bytes());
  out
}

fn i256_int_min() -> [u8; 32] {
  let mut out = [0u8; 32];
  out[0] = 0x80;
  out
}

fn bytecode_for_binary_op(op: u8, a: [u8; 32], b: [u8; 32]) -> Bytes {
  let mut code = Vec::with_capacity(1 + 32 + 1 + 32 + 2);
  code.push(opcode::PUSH32);
  code.extend_from_slice(&a);
  code.push(opcode::PUSH32);
  code.extend_from_slice(&b);
  code.push(op);
  code.push(opcode::STOP);
  Bytes::from(code)
}

fn find_step_by_opcode(
  steps: &[zprove_core::transition::InstructionTransitionProof],
  op: u8,
) -> Option<&zprove_core::transition::InstructionTransitionProof> {
  steps.iter().find(|step| step.opcode == op)
}

#[test]
fn executes_add_program_with_revm_and_proves() {
  let bytecode = Bytes::from(vec![
    opcode::PUSH1,
    0x02,
    opcode::PUSH1,
    0x03,
    opcode::ADD,
    opcode::STOP,
  ]);

  let proof = execute_bytecode_and_prove(bytecode, Bytes::default(), U256::ZERO)
    .expect("revm execution and proof generation should succeed");

  assert!(
    !proof.steps.is_empty(),
    "expected at least one executed step"
  );

  let add_step = find_step_by_opcode(&proof.steps, opcode::ADD).expect("ADD step should exist");
  assert_eq!(add_step.stack_inputs.len(), 2);
  assert_eq!(add_step.stack_outputs.len(), 1);
  assert_eq!(add_step.stack_outputs[0][31], 5);
}

#[test]
fn executes_bitwise_program_with_revm_and_proves() {
  let bytecode = Bytes::from(vec![
    opcode::PUSH1,
    0xF0,
    opcode::PUSH1,
    0x0F,
    opcode::XOR,
    opcode::STOP,
  ]);

  let proof = execute_bytecode_and_prove(bytecode, Bytes::default(), U256::ZERO)
    .expect("revm execution and proof generation should succeed");

  let xor_step = find_step_by_opcode(&proof.steps, opcode::XOR).expect("XOR step should exist");
  assert_eq!(xor_step.stack_outputs.len(), 1);
  assert_eq!(xor_step.stack_outputs[0][31], 0xFF);
}

#[test]
fn executes_add_program_with_revm_and_parallel_zkp_verify() {
  let bytecode = Bytes::from(vec![
    opcode::PUSH1,
    0x02,
    opcode::PUSH1,
    0x03,
    opcode::ADD,
    opcode::STOP,
  ]);

  let proof =
    execute_bytecode_and_prove_with_zkp_parallel(bytecode, Bytes::default(), U256::ZERO, 0)
      .expect("revm execution and parallel zkp verification should succeed");

  let add_step = find_step_by_opcode(&proof.steps, opcode::ADD).expect("ADD step should exist");
  assert_eq!(add_step.stack_outputs.len(), 1);
  assert_eq!(add_step.stack_outputs[0][31], 5);
}

#[test]
fn executes_memory_mstore_mload_with_revm_trace() {
  let bytecode = Bytes::from(vec![
    opcode::PUSH1,
    0x2A,
    opcode::PUSH1,
    0x00,
    opcode::MSTORE,
    opcode::PUSH1,
    0x00,
    opcode::MLOAD,
    opcode::STOP,
  ]);

  let trace = execute_bytecode_trace(bytecode, Bytes::default(), U256::ZERO)
    .expect("revm execution trace should succeed");

  let mstore_step =
    find_step_by_opcode(&trace.steps, opcode::MSTORE).expect("MSTORE step should exist");
  assert_eq!(mstore_step.stack_inputs.len(), 2);
  assert_eq!(mstore_step.stack_outputs.len(), 0);

  let mload_step =
    find_step_by_opcode(&trace.steps, opcode::MLOAD).expect("MLOAD step should exist");
  assert_eq!(mload_step.stack_outputs.len(), 1);
  assert_eq!(mload_step.stack_outputs[0][31], 0x2A);
}

#[test]
fn executes_memory_mstore8_with_revm_trace() {
  let bytecode = Bytes::from(vec![
    opcode::PUSH1,
    0xAB,
    opcode::PUSH1,
    0x00,
    opcode::MSTORE8,
    opcode::PUSH1,
    0x00,
    opcode::MLOAD,
    opcode::STOP,
  ]);

  let trace = execute_bytecode_trace(bytecode, Bytes::default(), U256::ZERO)
    .expect("revm execution trace should succeed");

  let mstore8_step =
    find_step_by_opcode(&trace.steps, opcode::MSTORE8).expect("MSTORE8 step should exist");
  assert_eq!(mstore8_step.stack_inputs.len(), 2);
  assert_eq!(mstore8_step.stack_outputs.len(), 0);

  let mload_step =
    find_step_by_opcode(&trace.steps, opcode::MLOAD).expect("MLOAD step should exist");
  assert_eq!(mload_step.stack_outputs.len(), 1);
  assert_eq!(mload_step.stack_outputs[0][0], 0xAB);
}

#[test]
fn revm_oracle_validates_div_family_proofs_on_edge_cases() {
  let unsigned_cases = [
    (u256_bytes_u128(0), u256_bytes_u128(1)),
    (u256_bytes_u128(1), u256_bytes_u128(1)),
    (u256_bytes_u128(1), u256_bytes_u128(0)),
    (u256_bytes_u128(u128::MAX), u256_bytes_u128(3)),
    (u256_bytes_u128(u128::MAX), u256_bytes_u128(u128::MAX - 1)),
    (u256_bytes_u128(1 << 127), u256_bytes_u128((1 << 64) + 1)),
    (
      {
        let mut x = [0u8; 32];
        x[0] = 0x7F;
        x[31] = 0xFF;
        x
      },
      {
        let mut y = [0u8; 32];
        y[0] = 0x01;
        y[31] = 0x11;
        y
      },
    ),
  ];

  let signed_cases = [
    (i256_bytes_i128(0), i256_bytes_i128(1)),
    (i256_bytes_i128(-1), i256_bytes_i128(1)),
    (i256_bytes_i128(1), i256_bytes_i128(-1)),
    (i256_bytes_i128(-1), i256_bytes_i128(-1)),
    (i256_bytes_i128(123456789), i256_bytes_i128(-97)),
    (i256_bytes_i128(-123456789), i256_bytes_i128(97)),
    (i256_int_min(), i256_bytes_i128(-1)),
    (i256_int_min(), i256_bytes_i128(0)),
  ];

  for (op, cases) in [
    (opcode::DIV, &unsigned_cases[..]),
    (opcode::MOD, &unsigned_cases[..]),
    (opcode::SDIV, &signed_cases[..]),
    (opcode::SMOD, &signed_cases[..]),
  ] {
    for (a, b) in cases {
      let bytecode = bytecode_for_binary_op(op, *a, *b);
      let proof = execute_bytecode_and_prove(bytecode, Bytes::default(), U256::ZERO)
        .expect("revm oracle execution/proving should succeed for edge case");
      let step = find_step_by_opcode(&proof.steps, op).expect("expected arithmetic opcode step");
      assert_eq!(step.stack_outputs.len(), 1);
    }
  }
}
