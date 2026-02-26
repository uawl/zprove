use revm::{
  bytecode::opcode,
  primitives::{Bytes, U256},
};
use zprove_core::execute::execute_bytecode_and_prove_with_zkp_parallel;

fn main() -> Result<(), String> {
  let bytecode = Bytes::from(vec![
    opcode::PUSH1,
    0x02,
    opcode::PUSH1,
    0x03,
    opcode::MUL,
    opcode::STOP,
  ]);

  let proof =
    execute_bytecode_and_prove_with_zkp_parallel(bytecode, Bytes::default(), U256::ZERO, 12)?;
  println!("revm execution proof steps: {}", proof.steps.len());
  Ok(())
}
