use revm::{
  bytecode::opcode,
  primitives::{Bytes, U256},
};
use zprove_core::execute::{execute_bytecode_and_prove, execute_bytecode_trace};

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

  assert!(!proof.steps.is_empty(), "expected at least one executed step");

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

  let mstore_step = find_step_by_opcode(&trace.steps, opcode::MSTORE).expect("MSTORE step should exist");
  assert_eq!(mstore_step.stack_inputs.len(), 2);
  assert_eq!(mstore_step.stack_outputs.len(), 0);

  let mload_step = find_step_by_opcode(&trace.steps, opcode::MLOAD).expect("MLOAD step should exist");
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

  let mstore8_step = find_step_by_opcode(&trace.steps, opcode::MSTORE8).expect("MSTORE8 step should exist");
  assert_eq!(mstore8_step.stack_inputs.len(), 2);
  assert_eq!(mstore8_step.stack_outputs.len(), 0);

  let mload_step = find_step_by_opcode(&trace.steps, opcode::MLOAD).expect("MLOAD step should exist");
  assert_eq!(mload_step.stack_outputs.len(), 1);
  assert_eq!(mload_step.stack_outputs[0][0], 0xAB);
}
