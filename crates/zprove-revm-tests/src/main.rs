use revm::{
  bytecode::opcode,
  primitives::{Bytes, U256},
};
use zprove_core::execute::execute_bytecode_and_prove;

fn main() -> Result<(), String> {
  let bytecode = Bytes::from(vec![
    opcode::PUSH1,
    0x02,
    opcode::PUSH1,
    0x03,
    opcode::ADD,
    opcode::STOP,
  ]);

  let proof = execute_bytecode_and_prove(bytecode, Bytes::default(), U256::ZERO)?;
  println!("revm execution proof steps: {}", proof.steps.len());
  Ok(())
}
