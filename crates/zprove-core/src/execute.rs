use crate::transition::{
  InstructionTransitionProof, InstructionTransitionStatement, TransactionProof, VmState,
  opcode_input_count, opcode_output_count, prove_instruction,
  prove_batch_transaction_zk_receipt, prove_instruction_zk_receipts_parallel,
  verify_batch_transaction_zk_receipt, verify_instruction_zk_receipt, verify_proof,
  verify_proof_with_rows,
};
use revm::bytecode::opcode;
use revm::{
  Context, InspectEvm, Inspector, MainBuilder, MainContext,
  context::TxEnv,
  database::BenchmarkDB,
  database_interface::{BENCH_CALLER, BENCH_TARGET},
  interpreter::interpreter_types::StackTr,
  interpreter::{Interpreter, InterpreterTypes, interpreter_types::Jumps},
  primitives::{Address, Bytes, TxKind, U256},
  state::Bytecode,
};

// ============================================================
// U256 ↔ [u8; 32] helpers
// ============================================================

const DEFAULT_ZKP_BATCH_CAPACITY: usize = 256;

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
  execute_and_prove_with_zkp_parallel(caller, transact_to, data, value, 0)
}

/// Execute an EVM transaction and verify ZKP receipts for supported semantic opcodes.
///
/// ZKP receipt proving is parallelized with a lock-free queue worker pool.
/// `worker_count = 0` means auto-select from available CPU parallelism.
pub fn execute_and_prove_with_zkp_parallel(
  caller: Address,
  transact_to: Address,
  data: Bytes,
  value: U256,
  worker_count: usize,
) -> Result<TransactionProof, String> {
  execute_and_prove_with_zkp_parallel_batched(
    caller,
    transact_to,
    data,
    value,
    worker_count,
    DEFAULT_ZKP_BATCH_CAPACITY,
  )
}

/// Execute an EVM transaction and verify ZKP receipts in buffered parallel batches.
///
/// Steps are accumulated in memory and flushed to proving when `batch_capacity`
/// is reached, then verified on CPU.
pub fn execute_and_prove_with_zkp_parallel_batched(
  caller: Address,
  transact_to: Address,
  data: Bytes,
  value: U256,
  worker_count: usize,
  batch_capacity: usize,
) -> Result<TransactionProof, String> {
  let proof = execute_and_prove(caller, transact_to, data, value)?;
  verify_transaction_proof_with_zkp_parallel(&proof, worker_count, batch_capacity)?;
  Ok(proof)
}

fn verify_transaction_proof(proof: &TransactionProof) -> Result<(), String> {
  for (i, step) in proof.steps.iter().enumerate() {
    if !verify_proof(step) {
      return Err(format!(
        "proof verification failed at step {i} (opcode 0x{:02x})",
        step.opcode
      ));
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

fn supports_zkp_receipt(op: u8) -> bool {
  matches!(
    op,
    opcode::ADD
      | opcode::SUB
      | opcode::MUL
      | opcode::DIV
      | opcode::MOD
      | opcode::SDIV
      | opcode::SMOD
      | opcode::AND
      | opcode::OR
      | opcode::XOR
      | opcode::NOT
  )
}

fn statement_from_step(step: &InstructionTransitionProof) -> InstructionTransitionStatement {
  InstructionTransitionStatement {
    opcode: step.opcode,
    s_i: VmState {
      opcode: step.opcode,
      pc: step.pc,
      sp: step.stack_inputs.len(),
      stack: step.stack_inputs.clone(),
      memory_root: [0u8; 32],
    },
    s_next: VmState {
      opcode: step.opcode,
      pc: step.pc + 1,
      sp: step.stack_outputs.len(),
      stack: step.stack_outputs.clone(),
      memory_root: [0u8; 32],
    },
    accesses: Vec::new(),
  }
}

fn flush_and_verify_zkp_batch(
  statements: &mut Vec<InstructionTransitionStatement>,
  steps: &mut Vec<InstructionTransitionProof>,
  worker_count: usize,
) -> Result<(), String> {
  if steps.is_empty() {
    return Ok(());
  }

  let proving_steps = std::mem::take(steps);
  let batch_statements = std::mem::take(statements);
  let receipts = prove_instruction_zk_receipts_parallel(proving_steps, worker_count)?;
  for (local_idx, (statement, receipt)) in batch_statements.iter().zip(receipts.iter()).enumerate() {
    if !verify_instruction_zk_receipt(statement, receipt) {
      return Err(format!(
        "zkp receipt verification failed at batched item {local_idx} (opcode 0x{:02x})",
        statement.opcode
      ));
    }
  }

  Ok(())
}

fn verify_transaction_proof_with_zkp_parallel(
  proof: &TransactionProof,
  worker_count: usize,
  batch_capacity: usize,
) -> Result<(), String> {
  verify_transaction_proof(proof)?;

  let cap = batch_capacity.max(1);
  let mut statements = Vec::with_capacity(cap);
  let mut steps = Vec::with_capacity(cap);

  for step in &proof.steps {
    if step.semantic_proof.is_some() && supports_zkp_receipt(step.opcode) {
      statements.push(statement_from_step(step));
      steps.push(step.clone());
      if steps.len() >= cap {
        flush_and_verify_zkp_batch(&mut statements, &mut steps, worker_count)?;
      }
    }
  }

  flush_and_verify_zkp_batch(&mut statements, &mut steps, worker_count)?;

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

// ============================================================
// Phase 6: Batch ZKP path
//
// Unlike the parallel path (N × individual receipt), the batch path proves
// all arithmetic opcodes in a single LUT STARK call:
//
//   flush_and_verify_batch_zkp_single():
//     all N steps → build_batch_manifest → setup_batch_prep
//                 → prove_batch_lut_with_prep (1 STARK)
//                 → verify_batch_transaction_zk_receipt
//
// batch_capacity still caps how many instructions share one STARK invocation
// (useful to bound trace height and proof size).
// ============================================================

fn flush_and_verify_batch_zkp_single(
  statements: &mut Vec<InstructionTransitionStatement>,
  steps: &mut Vec<InstructionTransitionProof>,
) -> Result<(), String> {
  if steps.is_empty() {
    return Ok(());
  }

  let proving_steps = std::mem::take(steps);
  let batch_statements = std::mem::take(statements);

  let receipt = prove_batch_transaction_zk_receipt(&proving_steps)?;
  if !verify_batch_transaction_zk_receipt(&batch_statements, &receipt) {
    return Err("batch ZKP receipt verification failed".to_string());
  }

  Ok(())
}

fn verify_transaction_proof_with_batch_zkp(
  proof: &TransactionProof,
  batch_capacity: usize,
) -> Result<(), String> {
  verify_transaction_proof(proof)?;

  let cap = batch_capacity.max(1);
  let mut statements: Vec<InstructionTransitionStatement> = Vec::with_capacity(cap);
  let mut steps: Vec<InstructionTransitionProof> = Vec::with_capacity(cap);

  for step in &proof.steps {
    if step.semantic_proof.is_some() && supports_zkp_receipt(step.opcode) {
      statements.push(statement_from_step(step));
      steps.push(step.clone());
      if steps.len() >= cap {
        flush_and_verify_batch_zkp_single(&mut statements, &mut steps)?;
      }
    }
  }

  flush_and_verify_batch_zkp_single(&mut statements, &mut steps)
}

/// Execute provided EVM bytecode and verify ZKP receipts using the batch path.
///
/// All arithmetic opcodes within each batch window share a single LUT STARK
/// proof, significantly reducing total proving overhead compared to the
/// per-instruction parallel path.
///
/// `batch_capacity` controls the maximum number of instructions per STARK
/// call; `0` or `DEFAULT_ZKP_BATCH_CAPACITY` is a sensible default.
pub fn execute_bytecode_and_prove_with_batch_zkp(
  bytecode: Bytes,
  data: Bytes,
  value: U256,
) -> Result<TransactionProof, String> {
  execute_bytecode_and_prove_with_batch_zkp_batched(
    bytecode,
    data,
    value,
    DEFAULT_ZKP_BATCH_CAPACITY,
  )
}

/// Execute provided EVM bytecode and verify ZKP receipts using the batch path
/// with a custom `batch_capacity`.
///
/// Steps with semantic proofs are accumulated and flushed to
/// [`prove_batch_transaction_zk_receipt`] / [`verify_batch_transaction_zk_receipt`]
/// when `batch_capacity` is reached or at the end of the transaction.
pub fn execute_bytecode_and_prove_with_batch_zkp_batched(
  bytecode: Bytes,
  data: Bytes,
  value: U256,
  batch_capacity: usize,
) -> Result<TransactionProof, String> {
  let proof = execute_bytecode_trace(bytecode, data, value)?;
  verify_transaction_proof_with_batch_zkp(&proof, batch_capacity)?;
  Ok(proof)
}

/// Execute provided EVM bytecode and verify ZKP receipts for supported opcodes.
///
/// ZKP receipt proving is parallelized with a lock-free queue worker pool.
/// `worker_count = 0` means auto-select from available CPU parallelism.
pub fn execute_bytecode_and_prove_with_zkp_parallel(
  bytecode: Bytes,
  data: Bytes,
  value: U256,
  worker_count: usize,
) -> Result<TransactionProof, String> {
  execute_bytecode_and_prove_with_zkp_parallel_batched(
    bytecode,
    data,
    value,
    worker_count,
    DEFAULT_ZKP_BATCH_CAPACITY,
  )
}

/// Execute provided EVM bytecode and verify ZKP receipts using buffered batches.
pub fn execute_bytecode_and_prove_with_zkp_parallel_batched(
  bytecode: Bytes,
  data: Bytes,
  value: U256,
  worker_count: usize,
  batch_capacity: usize,
) -> Result<TransactionProof, String> {
  let proof = execute_bytecode_trace(bytecode, data, value)?;
  verify_transaction_proof_with_zkp_parallel(&proof, worker_count, batch_capacity)?;
  Ok(proof)
}
