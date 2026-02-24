use crate::transition::{
  InstructionTransitionProof, TransactionProof, opcode_input_count, opcode_output_count,
  prove_instruction, verify_proof, verify_proof_with_rows,
};
use revm::{
  Context, InspectEvm, Inspector, MainBuilder, MainContext,
  context::TxEnv,
  database::BenchmarkDB,
  database_interface::{BENCH_CALLER, BENCH_TARGET},
  interpreter::interpreter_types::StackTr,
  interpreter::{Interpreter, InterpreterTypes, interpreter_types::Jumps},
  state::Bytecode,
  primitives::{Address, Bytes, TxKind, U256},
};

// ============================================================
// U256 ↔ [u8; 32] helpers
// ============================================================

fn u256_to_bytes(val: U256) -> [u8; 32] {
  val.to_be_bytes::<32>()
}

// ============================================================
// Proving inspector
// ============================================================

/// An EVM inspector that captures stack snapshots around every instruction
/// and generates Hilbert-style transition proofs.
#[derive(Default)]
struct ProvingInspector {
  /// Opcode of the instruction about to execute.
  pending_opcode: u8,
  /// Program counter before execution.
  pending_pc: usize,
  /// Stack values consumed by the instruction (read in `step()`).
  pending_inputs: Vec<[u8; 32]>,
  /// Stack depth before the instruction.
  pending_stack_depth: usize,
  /// Accumulated instruction transition proofs.
  pub proofs: Vec<InstructionTransitionProof>,
}

impl<CTX, INTR: InterpreterTypes> Inspector<CTX, INTR> for ProvingInspector {
  /// Called **before** each instruction executes.
  /// Captures the opcode, PC, and relevant stack inputs.
  fn step(&mut self, interp: &mut Interpreter<INTR>, _context: &mut CTX) {
    let opcode = interp.bytecode.opcode();
    let pc = interp.bytecode.pc();
    let stack_data = interp.stack.data(); // &[U256], index 0 = bottom
    let stack_len = stack_data.len();

    self.pending_opcode = opcode;
    self.pending_pc = pc;
    self.pending_stack_depth = stack_len;

    // Read the top N values that this opcode consumes
    let n_inputs = opcode_input_count(opcode);
    self.pending_inputs.clear();
    for i in 0..n_inputs.min(stack_len) {
      let val = stack_data[stack_len - 1 - i];
      self.pending_inputs.push(u256_to_bytes(val));
    }
  }

  /// Called **after** each instruction executes.
  /// Captures outputs and generates the transition proof.
  fn step_end(&mut self, interp: &mut Interpreter<INTR>, _context: &mut CTX) {
    let stack_data = interp.stack.data();
    let stack_len = stack_data.len();

    let n_outputs = opcode_output_count(self.pending_opcode);
    let mut outputs = Vec::with_capacity(n_outputs);
    for i in 0..n_outputs.min(stack_len) {
      let val = stack_data[stack_len - 1 - i];
      outputs.push(u256_to_bytes(val));
    }

    // Generate semantic proof (if applicable)
    let semantic_proof = if !self.pending_inputs.is_empty() || n_outputs > 0 {
      prove_instruction(self.pending_opcode, &self.pending_inputs, &outputs)
    } else {
      None
    };

    self.proofs.push(InstructionTransitionProof {
      opcode: self.pending_opcode,
      pc: self.pending_pc,
      stack_inputs: self.pending_inputs.clone(),
      stack_outputs: outputs,
      semantic_proof,
    });
  }
}

// ============================================================
// Public API
// ============================================================

/// Execute an EVM transaction and produce a Hilbert-style transition proof
/// for every instruction.
pub fn execute_and_prove(
  caller: Address,
  transact_to: Address,
  data: Bytes,
  value: U256,
) -> Result<TransactionProof, String> {
  let ctx = Context::mainnet();
  let inspector = ProvingInspector::default();
  let mut evm = ctx.build_mainnet_with_inspector(inspector);

  let tx = TxEnv::builder()
    .caller(caller)
    .kind(TxKind::Call(transact_to))
    .data(data)
    .value(value)
    .gas_limit(1_000_000)
    .build()
    .map_err(|err| format!("failed to build tx env: {err:?}"))?;

  let _execution_result = evm
    .inspect_one_tx(tx)
    .map_err(|err| format!("failed to execute tx: {err}"))?;

  let proof = TransactionProof {
    steps: evm.inspector.proofs.clone(),
  };

  verify_transaction_proof(&proof)?;

  Ok(proof)
}

/// Execute an EVM transaction and prove each instruction transition,
/// requiring semantic proof validity and compiled ProofRow validity.
pub fn execute_and_prove_with_rows(
  caller: Address,
  transact_to: Address,
  data: Bytes,
  value: U256,
) -> Result<TransactionProof, String> {
  execute_and_prove(caller, transact_to, data, value)
}

/// Backward-compatible alias.
///
/// Despite the name, this now uses the ProofRow verification path.
pub fn execute_and_prove_with_zkp(
  caller: Address,
  transact_to: Address,
  data: Bytes,
  value: U256,
) -> Result<TransactionProof, String> {
  execute_and_prove_with_rows(caller, transact_to, data, value)
}

fn verify_transaction_proof(proof: &TransactionProof) -> Result<(), String> {
  for (i, step) in proof.steps.iter().enumerate() {
    if !verify_proof(step) {
      return Err(format!("proof verification failed at step {i} (opcode 0x{:02x})", step.opcode));
    }
    if !verify_proof_with_rows(step) {
      return Err(format!(
        "proof row verification failed at step {i} (opcode 0x{:02x})",
        step.opcode
      ));
    }
  }
  Ok(())
}

/// Execute provided EVM bytecode in a benchmark DB and return transition traces.
///
/// This path performs real revm execution and captures per-opcode transition data,
/// but does not enforce semantic-proof verification for each opcode.
pub fn execute_bytecode_trace(
  bytecode: Bytes,
  data: Bytes,
  value: U256,
) -> Result<TransactionProof, String> {
  let bytecode = Bytecode::new_legacy(bytecode);
  let ctx = Context::mainnet().with_db(BenchmarkDB::new_bytecode(bytecode));
  let inspector = ProvingInspector::default();
  let mut evm = ctx.build_mainnet_with_inspector(inspector);

  let tx = TxEnv::builder()
    .caller(BENCH_CALLER)
    .kind(TxKind::Call(BENCH_TARGET))
    .data(data)
    .value(value)
    .gas_limit(1_000_000)
    .build()
    .map_err(|err| format!("failed to build tx env: {err:?}"))?;

  let _execution_result = evm
    .inspect_one_tx(tx)
    .map_err(|err| format!("failed to execute tx: {err}"))?;

  Ok(TransactionProof {
    steps: evm.inspector.proofs.clone(),
  })
}

/// Execute provided EVM bytecode in a benchmark DB and produce transition proofs.
///
/// This is intended for deterministic revm-backed integration tests.
pub fn execute_bytecode_and_prove(
  bytecode: Bytes,
  data: Bytes,
  value: U256,
) -> Result<TransactionProof, String> {
  let proof = execute_bytecode_trace(bytecode, data, value)?;
  verify_transaction_proof(&proof)?;
  Ok(proof)
}
