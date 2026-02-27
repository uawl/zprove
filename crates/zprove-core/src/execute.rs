use crate::transition::{
  BlockTxContext, CallContextClaim, ExecutionReceipt, ExternalStateClaim,
  InstructionTransitionProof, InstructionTransitionStatement, KeccakClaim, MemAccessClaim,
  ReturnDataClaim, StackAccessClaim, StorageAccessClaim, SubCallClaim, TransactionProof, VmState,
  opcode_input_count, opcode_output_count, prove_batch_transaction_zk_receipt,
  prove_execution_chain, prove_instruction, prove_instruction_zk_receipts_parallel,
  verify_batch_transaction_zk_receipt, verify_instruction_zk_receipt, verify_proof,
  verify_proof_with_rows,
};
use crate::zk_proof::{prove_memory_consistency, verify_memory_consistency};
use revm::bytecode::opcode;
use revm::{
  Context, InspectEvm, Inspector, MainBuilder, MainContext,
  context::TxEnv,
  database::BenchmarkDB,
  database_interface::{BENCH_CALLER, BENCH_TARGET},
  interpreter::interpreter_types::StackTr,
  interpreter::interpreter_types::InputsTr,
  interpreter::{
    Interpreter, InterpreterTypes,
    interpreter_types::{Jumps, MemoryTr},
    CallInputs, CallOutcome, CreateInputs, CreateOutcome,
  },
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
  /// Snapshot of the relevant memory word *before* the instruction executes.
  /// Used to build MemAccessClaim for MSTORE/MSTORE8 (we need value_before).
  pending_memory_before: Option<[u8; 32]>,
  /// Contract address executing the current instruction (20 bytes, big-endian).
  /// Captured in `step()` to be used in `step_end()` for SLOAD/SSTORE claims.
  pending_contract: [u8; 20],
  /// Global monotone rw_counter for memory accesses.
  rw_counter: u64,
  /// Global monotone rw_counter for stack accesses.
  stack_rw_counter: u64,
  /// Accumulated instruction transition proofs.
  pub proofs: Vec<InstructionTransitionProof>,
  /// Stack of sub-call info captured in `call()` / `create()` hooks,
  /// pending consumption in the matching `call_end()` / `create_end()`.
  pending_sub_call_stack: Vec<(u8, [u8; 20], [u8; 32])>, // (opcode, callee, value)
  /// Sub-call claim ready to be attached to the next ITP in `step_end()`.
  pending_sub_call: Option<SubCallClaim>,
}

/// Read a 32-byte word from a raw memory slice at a given byte offset.
/// Returns `[0u8; 32]` if the range is out of bounds.
fn read_word_from_mem(mem: &[u8], offset: usize) -> [u8; 32] {
  let mut word = [0u8; 32];
  let available = mem.len().saturating_sub(offset);
  let copy_len = available.min(32);
  if copy_len > 0 {
    word[..copy_len].copy_from_slice(&mem[offset..offset + copy_len]);
  }
  word
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
    self.pending_memory_before = None;

    // Read the top N values that this opcode consumes
    let n_inputs = opcode_input_count(opcode);
    self.pending_inputs.clear();
    for i in 0..n_inputs.min(stack_len) {
      let val = stack_data[stack_len - 1 - i];
      self.pending_inputs.push(u256_to_bytes(val));
    }

    // For memory opcodes, snapshot the 32-byte word at the target address
    // *before* the instruction mutates memory.
    match opcode {
      opcode::MSTORE | opcode::MSTORE8 => {
        // inputs[0] = offset (stack top before execute)
        if stack_len >= 1 {
          let offset_u256 = stack_data[stack_len - 1];
          // Clamp to usize; EVM guarantees gas will prevent unreasonably large offsets
          if let Some(offset_usize) = offset_u256
            .try_into()
            .ok()
            .filter(|&v: &usize| v <= 4 * 1024 * 1024)
          {
            // Align down to 32-byte word boundary
            let word_addr = (offset_usize / 32) * 32;
            let local_off = interp.memory.local_memory_offset();
            let local_size = interp.memory.size();
            let mem_bytes: Vec<u8> = interp
              .memory
              .global_slice(local_off..local_off + local_size)
              .to_vec();
            self.pending_memory_before = Some(read_word_from_mem(&mem_bytes, word_addr));
          }
        }
      }
      opcode::SLOAD | opcode::SSTORE => {
        // Capture the contract address once per instruction for storage claims.
        let addr = interp.input.target_address();
        self.pending_contract = addr.into_array();
      }
      _ => {}
    }
  }

  /// Called **after** each instruction executes.
  /// Captures outputs, memory claims, and generates the transition proof.
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

    // Build memory access claims for MLOAD / MSTORE / MSTORE8 / RETURN / REVERT.
    let (memory_claims, return_data_claim) = self.build_memory_and_return_claims(interp, &outputs);

    // Build storage access claims for SLOAD / SSTORE.
    let storage_claims = self.build_storage_claims(&outputs);

    // Build call-context claim for CALLER / CALLVALUE / CALLDATALOAD / CALLDATASIZE.
    let call_context_claim = self.build_call_context_claim(interp, &outputs);

    // Build KECCAK256 claim (preimage + hash witness).
    let keccak_claim = self.build_keccak_claim(interp, &outputs);

    // Build external-state claim (BLOCKHASH, EXTCODESIZE, EXTCODEHASH).
    let external_state_claim = self.build_external_state_claim(&outputs);

    // Take the sub-call claim captured in call_end/create_end.
    let sub_call_claim = self.pending_sub_call.take();

    // Build stack access claims.
    let stack_claims = self.build_stack_claims(&outputs);

    self.proofs.push(InstructionTransitionProof {
      opcode: self.pending_opcode,
      pc: self.pending_pc,
      stack_inputs: self.pending_inputs.clone(),
      stack_outputs: outputs,
      semantic_proof,
      memory_claims,
      storage_claims,
      stack_claims,
      return_data_claim,
      call_context_claim,
      keccak_claim,
      external_state_claim,
      sub_call_claim,
    });
  }

  /// Called when a CALL / CALLCODE / DELEGATECALL / STATICCALL is about to execute.
  /// Pushes metadata onto the sub-call stack so `call_end` can build the claim.
  fn call(&mut self, _context: &mut CTX, inputs: &mut CallInputs) -> Option<CallOutcome> {
    use revm::interpreter::CallScheme;
    let op = match inputs.scheme {
      CallScheme::Call => opcode::CALL,
      CallScheme::CallCode => opcode::CALLCODE,
      CallScheme::DelegateCall => opcode::DELEGATECALL,
      CallScheme::StaticCall => opcode::STATICCALL,
    };
    let callee = inputs.target_address.into_array();
    let value = inputs
      .transfer_value()
      .unwrap_or_else(|| inputs.apparent_value().unwrap_or(U256::ZERO))
      .to_be_bytes::<32>();
    self.pending_sub_call_stack.push((op, callee, value));
    None
  }

  /// Called after a CALL / CALLCODE / DELEGATECALL / STATICCALL finishes.
  /// Builds the `SubCallClaim` from the recorded info + outcome.
  fn call_end(
    &mut self,
    _context: &mut CTX,
    _inputs: &CallInputs,
    outcome: &mut CallOutcome,
  ) {
    if let Some((op, callee, value)) = self.pending_sub_call_stack.pop() {
      // After popping, len() equals the call nesting depth (0 = top-level TX).
      let depth = self.pending_sub_call_stack.len() as u16;
      self.pending_sub_call = Some(SubCallClaim {
        opcode: op,
        callee,
        value,
        return_data: outcome.result.output.to_vec(),
        success: outcome.result.is_ok(),
        depth,
        inner_proof: None,
      });
    }
  }

  /// Called when a CREATE / CREATE2 is about to execute.
  fn create(
    &mut self,
    _context: &mut CTX,
    inputs: &mut CreateInputs,
  ) -> Option<CreateOutcome> {
    use revm::interpreter::CreateScheme;
    let op = match inputs.scheme() {
      CreateScheme::Create => opcode::CREATE,
      CreateScheme::Create2 { .. } => opcode::CREATE2,
      CreateScheme::Custom { .. } => opcode::CREATE,
    };
    // Callee address is not known until after deployment; use zero placeholder.
    self.pending_sub_call_stack.push((op, [0u8; 20], inputs.value().to_be_bytes::<32>()));
    None
  }

  /// Called after a CREATE / CREATE2 finishes.
  fn create_end(
    &mut self,
    _context: &mut CTX,
    _inputs: &CreateInputs,
    outcome: &mut CreateOutcome,
  ) {
    if let Some((op, _, value)) = self.pending_sub_call_stack.pop() {
      // After popping, len() equals the call nesting depth (0 = top-level TX).
      let depth = self.pending_sub_call_stack.len() as u16;
      let callee = outcome.address.map(|a| a.into_array()).unwrap_or([0u8; 20]);
      self.pending_sub_call = Some(SubCallClaim {
        opcode: op,
        callee,
        value,
        return_data: outcome.result.output.to_vec(),
        success: outcome.result.is_ok(),
        depth,
        inner_proof: None,
      });
    }
  }
}

impl ProvingInspector {
  /// Build `MemAccessClaim`s and (optionally) a `ReturnDataClaim` for the current instruction.
  ///
  /// Returns `(memory_claims, return_data_claim)`.
  /// `return_data_claim` is `Some` only for RETURN and REVERT.
  fn build_memory_and_return_claims<INTR: InterpreterTypes>(
    &mut self,
    interp: &Interpreter<INTR>,
    outputs: &[[u8; 32]],
  ) -> (Vec<MemAccessClaim>, Option<ReturnDataClaim>) {
    let mut claims = Vec::new();
    let mut return_data_claim: Option<ReturnDataClaim> = None;

    match self.pending_opcode {
      opcode::MLOAD => {
        // inputs[0] = offset; outputs[0] = loaded 32-byte word
        if self.pending_inputs.len() >= 1 && outputs.len() >= 1 {
          let offset_bytes = self.pending_inputs[0];
          let offset_u256 = U256::from_be_bytes::<32>(offset_bytes);
          if let Some(word_addr) = offset_u256
            .try_into()
            .ok()
            .filter(|&v: &usize| v <= 4 * 1024 * 1024)
          {
            let aligned = (word_addr / 32) * 32;
            self.rw_counter += 1;
            claims.push(MemAccessClaim {
              rw_counter: self.rw_counter,
              addr: aligned as u64,
              is_write: false,
              value: outputs[0],
            });
          }
        }
      }
      opcode::MSTORE => {
        // inputs[0] = offset, inputs[1] = value
        if self.pending_inputs.len() >= 2 {
          let offset_bytes = self.pending_inputs[0];
          let offset_u256 = U256::from_be_bytes::<32>(offset_bytes);
          let value_after = self.pending_inputs[1];
          if let Some(word_addr) = offset_u256
            .try_into()
            .ok()
            .filter(|&v: &usize| v <= 4 * 1024 * 1024)
          {
            let aligned = (word_addr / 32) * 32;
            self.rw_counter += 1;
            claims.push(MemAccessClaim {
              rw_counter: self.rw_counter,
              addr: aligned as u64,
              is_write: true,
              value: value_after,
            });
          }
        }
      }
      opcode::MSTORE8 => {
        // inputs[0] = offset, inputs[1] = value (only LSB written)
        if self.pending_inputs.len() >= 2 {
          let offset_bytes = self.pending_inputs[0];
          let offset_u256 = U256::from_be_bytes::<32>(offset_bytes);
          let byte_val = self.pending_inputs[1][31]; // LSB
          if let Some(byte_addr) = offset_u256
            .try_into()
            .ok()
            .filter(|&v: &usize| v <= 4 * 1024 * 1024)
          {
            let aligned = (byte_addr / 32) * 32;
            // Read the updated 32-byte word from memory after the write.
            let local_off = interp.memory.local_memory_offset();
            let local_size = interp.memory.size();
            let mem_after: Vec<u8> = interp
              .memory
              .global_slice(local_off..local_off + local_size)
              .to_vec();
            let mut word_after = self.pending_memory_before.unwrap_or([0u8; 32]);
            let byte_offset_in_word = byte_addr % 32;
            word_after[byte_offset_in_word] = byte_val;
            // Cross-check against actual memory content.
            let actual = read_word_from_mem(&mem_after, aligned);
            let _ = (actual, word_after); // both available for debugging
            self.rw_counter += 1;
            claims.push(MemAccessClaim {
              rw_counter: self.rw_counter,
              addr: aligned as u64,
              is_write: true,
              // Use actual memory after write (word_after may differ if memory
              // was already expanded with non-zero bytes from prior writes).
              value: actual,
            });
          }
        }
      }
      op @ (opcode::RETURN | opcode::REVERT) => {
        // inputs[0] = offset, inputs[1] = size
        if self.pending_inputs.len() >= 2 {
          let offset_u256 = U256::from_be_bytes::<32>(self.pending_inputs[0]);
          let size_u256 = U256::from_be_bytes::<32>(self.pending_inputs[1]);
          if let (Some(offset), Some(size)) = (
            offset_u256.try_into().ok().filter(|&v: &usize| v <= 4 * 1024 * 1024),
            size_u256.try_into().ok().filter(|&v: &usize| v <= 4 * 1024 * 1024),
          ) {
            // Read the current memory snapshot.
            let local_off = interp.memory.local_memory_offset();
            let local_size = interp.memory.size();
            let mem_bytes: Vec<u8> = interp
              .memory
              .global_slice(local_off..local_off + local_size)
              .to_vec();

            // Emit one MemAccessClaim per 32-byte aligned chunk touched.
            if size > 0 {
              let start_aligned = (offset / 32) * 32;
              let end_byte = offset + size;
              let end_aligned = ((end_byte + 31) / 32) * 32;
              let mut addr = start_aligned;
              while addr < end_aligned {
                self.rw_counter += 1;
                let word = read_word_from_mem(&mem_bytes, addr);
                claims.push(MemAccessClaim {
                  rw_counter: self.rw_counter,
                  addr: addr as u64,
                  is_write: false,
                  value: word,
                });
                addr += 32;
              }
            }

            // Capture the actual return data bytes.
            let data = if size == 0 {
              vec![]
            } else {
              let start = offset.min(mem_bytes.len());
              let end = (offset + size).min(mem_bytes.len());
              let mut d = mem_bytes[start..end].to_vec();
              // Zero-pad if memory was shorter than offset+size.
              if d.len() < size {
                d.resize(size, 0u8);
              }
              d
            };

            return_data_claim = Some(ReturnDataClaim {
              is_revert: op == opcode::REVERT,
              offset: offset as u64,
              size: size as u64,
              data,
            });
          }
        }
      }
      // RETURNDATACOPY: (dest_offset, src_offset, size) → 0
      // Emits one write MemAccessClaim per 32-byte aligned chunk in dest range.
      opcode::RETURNDATACOPY => {
        if self.pending_inputs.len() >= 3 {
          let dest_u256 = U256::from_be_bytes::<32>(self.pending_inputs[0]);
          let size_u256 = U256::from_be_bytes::<32>(self.pending_inputs[2]);
          if let (Some(dest), Some(size)) = (
            dest_u256.try_into().ok().filter(|&v: &usize| v <= 4 * 1024 * 1024),
            size_u256.try_into().ok().filter(|&v: &usize| v <= 4 * 1024 * 1024),
          ) {
            if size > 0 {
              let local_off = interp.memory.local_memory_offset();
              let local_size = interp.memory.size();
              let mem: Vec<u8> = interp
                .memory
                .global_slice(local_off..local_off + local_size)
                .to_vec();
              let start_aligned = (dest / 32) * 32;
              let end_aligned = ((dest + size + 31) / 32) * 32;
              let mut addr = start_aligned;
              while addr < end_aligned {
                self.rw_counter += 1;
                claims.push(MemAccessClaim {
                  rw_counter: self.rw_counter,
                  addr: addr as u64,
                  is_write: true,
                  value: read_word_from_mem(&mem, addr),
                });
                addr += 32;
              }
            }
          }
        }
      }

      // EXTCODECOPY: (address, dest_offset, src_offset, size) → 0
      // Emits one write MemAccessClaim per 32-byte aligned chunk in dest range.
      opcode::EXTCODECOPY => {
        if self.pending_inputs.len() >= 4 {
          let dest_u256 = U256::from_be_bytes::<32>(self.pending_inputs[1]);
          let size_u256 = U256::from_be_bytes::<32>(self.pending_inputs[3]);
          if let (Some(dest), Some(size)) = (
            dest_u256.try_into().ok().filter(|&v: &usize| v <= 4 * 1024 * 1024),
            size_u256.try_into().ok().filter(|&v: &usize| v <= 4 * 1024 * 1024),
          ) {
            if size > 0 {
              let local_off = interp.memory.local_memory_offset();
              let local_size = interp.memory.size();
              let mem: Vec<u8> = interp
                .memory
                .global_slice(local_off..local_off + local_size)
                .to_vec();
              let start_aligned = (dest / 32) * 32;
              let end_aligned = ((dest + size + 31) / 32) * 32;
              let mut addr = start_aligned;
              while addr < end_aligned {
                self.rw_counter += 1;
                claims.push(MemAccessClaim {
                  rw_counter: self.rw_counter,
                  addr: addr as u64,
                  is_write: true,
                  value: read_word_from_mem(&mem, addr),
                });
                addr += 32;
              }
            }
          }
        }
      }

      // MCOPY: (dest_offset, src_offset, size) → 0
      // Emits read claims for src range then write claims for dest range.
      opcode::MCOPY => {
        if self.pending_inputs.len() >= 3 {
          let dest_u256 = U256::from_be_bytes::<32>(self.pending_inputs[0]);
          let src_u256 = U256::from_be_bytes::<32>(self.pending_inputs[1]);
          let size_u256 = U256::from_be_bytes::<32>(self.pending_inputs[2]);
          if let (Some(dest), Some(src), Some(size)) = (
            dest_u256.try_into().ok().filter(|&v: &usize| v <= 4 * 1024 * 1024),
            src_u256.try_into().ok().filter(|&v: &usize| v <= 4 * 1024 * 1024),
            size_u256.try_into().ok().filter(|&v: &usize| v <= 4 * 1024 * 1024),
          ) {
            if size > 0 {
              let local_off = interp.memory.local_memory_offset();
              let local_size = interp.memory.size();
              let mem: Vec<u8> = interp
                .memory
                .global_slice(local_off..local_off + local_size)
                .to_vec();
              // Read claims for source range.
              let src_start = (src / 32) * 32;
              let src_end = ((src + size + 31) / 32) * 32;
              let mut addr = src_start;
              while addr < src_end {
                self.rw_counter += 1;
                claims.push(MemAccessClaim {
                  rw_counter: self.rw_counter,
                  addr: addr as u64,
                  is_write: false,
                  value: read_word_from_mem(&mem, addr),
                });
                addr += 32;
              }
              // Write claims for destination range (after copy).
              let dst_start = (dest / 32) * 32;
              let dst_end = ((dest + size + 31) / 32) * 32;
              let mut addr = dst_start;
              while addr < dst_end {
                self.rw_counter += 1;
                claims.push(MemAccessClaim {
                  rw_counter: self.rw_counter,
                  addr: addr as u64,
                  is_write: true,
                  value: read_word_from_mem(&mem, addr),
                });
                addr += 32;
              }
            }
          }
        }
      }

      _ => {}
    }

    (claims, return_data_claim)
  }

  /// Build a `KeccakClaim` for KECCAK256.
  ///
  /// - `pending_inputs[0]` = memory offset (U256, big-endian).
  /// - `pending_inputs[1]` = size in bytes (U256, big-endian).
  /// - `outputs[0]`        = keccak256 digest pushed to the stack.
  ///
  /// Reads `size` bytes from memory at `offset` to record the full preimage.
  fn build_keccak_claim<INTR: InterpreterTypes>(
    &self,
    interp: &Interpreter<INTR>,
    outputs: &[[u8; 32]],
  ) -> Option<KeccakClaim> {
    if self.pending_opcode != opcode::KECCAK256 {
      return None;
    }
    if self.pending_inputs.len() < 2 || outputs.is_empty() {
      return None;
    }

    let offset_u256 = U256::from_be_bytes::<32>(self.pending_inputs[0]);
    let size_u256 = U256::from_be_bytes::<32>(self.pending_inputs[1]);

    // Clamp to usize (gas enforcement prevents unreasonable values in practice).
    let offset: u64 = offset_u256.try_into().unwrap_or(u64::MAX);
    let size: u64 = size_u256.try_into().unwrap_or(u64::MAX);

    let output_hash = outputs[0];

    // Read `size` bytes from memory.
    let input_bytes = if size == 0 {
      vec![]
    } else {
      let local_off = interp.memory.local_memory_offset();
      let local_size = interp.memory.size();
      let mem: Vec<u8> = interp
        .memory
        .global_slice(local_off..local_off + local_size)
        .to_vec();
      let start = offset as usize;
      let end = start.saturating_add(size as usize).min(mem.len());
      let mut bytes = vec![0u8; size as usize];
      if start < mem.len() {
        let copy_len = (end - start).min(size as usize);
        bytes[..copy_len].copy_from_slice(&mem[start..start + copy_len]);
      }
      bytes
    };

    Some(KeccakClaim { offset, size, input_bytes, output_hash })
  }

  /// Build a `CallContextClaim` for CALLER / CALLVALUE / CALLDATALOAD / CALLDATASIZE.
  ///
  /// - CALLER:        `output_value` = caller address left-padded to 32 bytes.
  /// - CALLVALUE:     `output_value` = ETH value as big-endian U256.
  /// - CALLDATALOAD:  `calldata_offset` = `pending_inputs[0]` (clamped), `output_value` = `outputs[0]`.
  /// - CALLDATASIZE:  `output_value` = calldata length as big-endian U256.
  fn build_call_context_claim<INTR: InterpreterTypes>(
    &self,
    interp: &Interpreter<INTR>,
    outputs: &[[u8; 32]],
  ) -> Option<CallContextClaim> {
    match self.pending_opcode {
      opcode::CALLER => {
        let addr = interp.input.caller_address();
        let mut value = [0u8; 32];
        value[12..32].copy_from_slice(&addr.into_array());
        Some(CallContextClaim { opcode: opcode::CALLER, calldata_offset: 0, output_value: value })
      }
      opcode::CALLVALUE => {
        let val = interp.input.call_value();
        Some(CallContextClaim {
          opcode: opcode::CALLVALUE,
          calldata_offset: 0,
          output_value: val.to_be_bytes::<32>(),
        })
      }
      opcode::CALLDATALOAD => {
        if outputs.is_empty() {
          return None;
        }
        let offset = if self.pending_inputs.is_empty() {
          0u64
        } else {
          U256::from_be_bytes::<32>(self.pending_inputs[0])
            .try_into()
            .unwrap_or(u64::MAX)
        };
        Some(CallContextClaim {
          opcode: opcode::CALLDATALOAD,
          calldata_offset: offset,
          output_value: outputs[0],
        })
      }
      opcode::CALLDATASIZE => {
        let size = interp.input.input().len();
        let mut value = [0u8; 32];
        value[24..32].copy_from_slice(&(size as u64).to_be_bytes());
        Some(CallContextClaim {
          opcode: opcode::CALLDATASIZE,
          calldata_offset: 0,
          output_value: value,
        })
      }
      opcode::PC => {
        // PC pushes the program counter of the PC opcode itself.
        if outputs.is_empty() {
          return None;
        }
        Some(CallContextClaim {
          opcode: opcode::PC,
          calldata_offset: 0,
          output_value: outputs[0],
        })
      }
      opcode::MSIZE => {
        // MSIZE pushes current memory size in bytes (always a multiple of 32).
        if outputs.is_empty() {
          return None;
        }
        Some(CallContextClaim {
          opcode: opcode::MSIZE,
          calldata_offset: 0,
          output_value: outputs[0],
        })
      }
      opcode::GAS => {
        // GAS pushes remaining gas after deducting the cost of the GAS opcode.
        if outputs.is_empty() {
          return None;
        }
        Some(CallContextClaim {
          opcode: opcode::GAS,
          calldata_offset: 0,
          output_value: outputs[0],
        })
      }
      // Block / tx context opcodes: revm already computed and pushed the value;
      // we simply capture outputs[0] as the witness.
      opcode::ADDRESS
      | opcode::ORIGIN
      | opcode::GASPRICE
      | opcode::CODESIZE
      | opcode::RETURNDATASIZE
      | opcode::COINBASE
      | opcode::TIMESTAMP
      | opcode::NUMBER
      | opcode::DIFFICULTY
      | opcode::GASLIMIT
      | opcode::CHAINID
      | opcode::SELFBALANCE
      | opcode::BASEFEE
      | opcode::BLOBBASEFEE => {
        if outputs.is_empty() {
          return None;
        }
        Some(CallContextClaim {
          opcode: self.pending_opcode,
          calldata_offset: 0,
          output_value: outputs[0],
        })
      }
      _ => None,
    }
  }

  /// Build `StorageAccessClaim`s for the current instruction (SLOAD / SSTORE).
  ///
  /// - SLOAD:  `pending_inputs[0]` = slot key, `outputs[0]` = loaded value.
  /// - SSTORE: `pending_inputs[0]` = slot key, `pending_inputs[1]` = written value.
  fn build_storage_claims(&mut self, outputs: &[[u8; 32]]) -> Vec<StorageAccessClaim> {
    match self.pending_opcode {
      opcode::SLOAD => {
        if self.pending_inputs.len() >= 1 && outputs.len() >= 1 {
          self.rw_counter += 1;
          vec![StorageAccessClaim {
            rw_counter: self.rw_counter,
            contract: self.pending_contract,
            slot: self.pending_inputs[0],
            is_write: false,
            value: outputs[0],
          }]
        } else {
          vec![]
        }
      }
      opcode::SSTORE => {
        if self.pending_inputs.len() >= 2 {
          self.rw_counter += 1;
          vec![StorageAccessClaim {
            rw_counter: self.rw_counter,
            contract: self.pending_contract,
            slot: self.pending_inputs[0],
            is_write: true,
            value: self.pending_inputs[1],
          }]
        } else {
          vec![]
        }
      }
      _ => vec![],
    }
  }

  /// Build an `ExternalStateClaim` for BLOCKHASH, EXTCODESIZE, EXTCODEHASH.
  ///
  /// `key` is the stack input: block number for BLOCKHASH, contract address
  /// (zero-padded to 32 bytes) for EXTCODESIZE / EXTCODEHASH.
  fn build_external_state_claim(&self, outputs: &[[u8; 32]]) -> Option<ExternalStateClaim> {
    match self.pending_opcode {
      opcode::BLOCKHASH | opcode::EXTCODESIZE | opcode::EXTCODEHASH => {
        if self.pending_inputs.is_empty() || outputs.is_empty() {
          return None;
        }
        Some(ExternalStateClaim {
          opcode: self.pending_opcode,
          key: self.pending_inputs[0],
          output_value: outputs[0],
        })
      }
      _ => None,
    }
  }

  /// Build `StackAccessClaim`s for the current instruction.
  ///
  /// Call order: one pop claim per consumed value (inputs),
  /// one push claim per produced value (outputs).
  ///
  /// `depth_after` for a pop:   stack_len_before - (i + 1)
  ///   = depth after having popped this and all previous inputs.
  /// `depth_after` for a push:  depth_after_all_pops + (j + 1)
  fn build_stack_claims(&mut self, outputs: &[[u8; 32]]) -> Vec<StackAccessClaim> {
    let mut claims = Vec::new();
    let n_pops = self.pending_inputs.len();
    let sp_before = self.pending_stack_depth;

    for (i, val) in self.pending_inputs.iter().enumerate() {
      self.stack_rw_counter += 1;
      claims.push(StackAccessClaim {
        rw_counter: self.stack_rw_counter,
        depth_after: sp_before.saturating_sub(i + 1),
        is_push: false,
        value: *val,
      });
    }

    let sp_after_pops = sp_before.saturating_sub(n_pops);
    for (j, val) in outputs.iter().enumerate() {
      self.stack_rw_counter += 1;
      claims.push(StackAccessClaim {
        rw_counter: self.stack_rw_counter,
        depth_after: sp_after_pops + (j + 1),
        is_push: true,
        value: *val,
      });
    }

    claims
  }
}

// ============================================================
// Public API
// ============================================================

/// Extract a [`BlockTxContext`] from the revm execution environment.
///
/// Should be called after `inspect_one_tx` so the context holds the
/// actual values used during execution.
fn extract_block_tx_context<CFG>(
  block: &revm::context::BlockEnv,
  tx: &revm::context::TxEnv,
  cfg: &CFG,
) -> BlockTxContext
where
  CFG: revm::context_interface::Cfg,
{
  // COINBASE: Address (20 bytes) zero-padded to 32
  let mut coinbase = [0u8; 32];
  coinbase[12..].copy_from_slice(block.beneficiary.as_slice());

  // TIMESTAMP / NUMBER / DIFFICULTY: U256 → big-endian [u8; 32]
  let timestamp = block.timestamp.to_be_bytes::<32>();
  let block_number = block.number.to_be_bytes::<32>();

  // PREVRANDAO: prefer post-Merge prevrandao (B256), fall back to difficulty
  let prevrandao = block
    .prevrandao
    .map(|r| *r)
    .unwrap_or_else(|| block.difficulty.to_be_bytes::<32>());

  // GASLIMIT: u64 → big-endian [u8; 32]
  let mut gas_limit = [0u8; 32];
  gas_limit[24..].copy_from_slice(&block.gas_limit.to_be_bytes());

  // CHAINID: from cfg (u64) → big-endian [u8; 32]
  let mut chain_id = [0u8; 32];
  chain_id[24..].copy_from_slice(&cfg.chain_id().to_be_bytes());

  // BASEFEE: u64 → big-endian [u8; 32]
  let mut basefee = [0u8; 32];
  basefee[24..].copy_from_slice(&block.basefee.to_be_bytes());

  // ORIGIN: tx.caller (Address, 20 bytes) zero-padded to 32
  let mut origin = [0u8; 32];
  origin[12..].copy_from_slice(tx.caller.as_slice());

  // GASPRICE: u128 → big-endian [u8; 32]
  let mut gas_price = [0u8; 32];
  gas_price[16..].copy_from_slice(&tx.gas_price.to_be_bytes());

  BlockTxContext {
    coinbase,
    timestamp,
    block_number,
    prevrandao,
    gas_limit,
    chain_id,
    basefee,
    origin,
    gas_price,
  }
}

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

  let block_tx_context =
    extract_block_tx_context(&evm.ctx.block, &evm.ctx.tx, &evm.ctx.cfg);

  let proof = TransactionProof {
    steps: evm.inspector.proofs.clone(),
    block_tx_context,
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

  // Collect all memory access claims and prove consistency via MemoryConsistencyAir.
  let all_mem_claims: Vec<MemAccessClaim> = proof
    .steps
    .iter()
    .flat_map(|s| s.memory_claims.iter().cloned())
    .collect();
  if !all_mem_claims.is_empty() {
    let mem_proof = prove_memory_consistency(&all_mem_claims)
      .map_err(|e| format!("memory consistency proving failed: {e}"))?;
    if !verify_memory_consistency(&mem_proof) {
      return Err("memory consistency verification failed".to_string());
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
    sub_call_claim: step.sub_call_claim.clone(),
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
  for (local_idx, (statement, receipt)) in batch_statements.iter().zip(receipts.iter()).enumerate()
  {
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

  let block_tx_context =
    extract_block_tx_context(&evm.ctx.block, &evm.ctx.tx, &evm.ctx.cfg);

  Ok(TransactionProof {
    steps: evm.inspector.proofs.clone(),
    block_tx_context,
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

/// Execute `bytecode` and produce a recursive [`ExecutionReceipt`] via the
/// Phase-2 segment-chain prover.
///
/// Steps are chunked into windows of `window_size` instructions.  Each window
/// is proved independently with a [`BatchTransactionZkReceipt`] (omitted when
/// the window contains no arithmetic opcodes), and all windows are linked into
/// a binary aggregation tree using [`LinkAir`](crate::zk_proof::LinkAir)
/// link STARKs.
///
/// The `memory_root` in synthesised [`VmState`] boundaries is set to all-zeros
/// for Phase 2 (full Merkle-memory integration is deferred to Phase 3).
pub fn execute_bytecode_and_prove_chain(
  bytecode: Bytes,
  data: Bytes,
  value: U256,
  window_size: usize,
) -> Result<ExecutionReceipt, String> {
  let proof = execute_bytecode_trace(bytecode, data, value)?;
  verify_transaction_proof(&proof)?;

  let steps = &proof.steps;
  if steps.is_empty() {
    return Err("execute_bytecode_and_prove_chain: no execution steps".to_string());
  }

  let wsize = window_size.max(1);

  // Split steps into window-sized segments.
  let windows: Vec<Vec<InstructionTransitionProof>> =
    steps.chunks(wsize).map(|c| c.to_vec()).collect();

  // Build VmState boundary for the start of each window, then append the
  // terminal state after the last step.
  let mut vm_states: Vec<VmState> = windows
    .iter()
    .map(|w| {
      let first = w.first().unwrap();
      VmState {
        opcode: first.opcode,
        pc: first.pc,
        sp: first.stack_inputs.len(),
        stack: first.stack_inputs.clone(),
        memory_root: [0u8; 32],
      }
    })
    .collect();

  // Terminal boundary: state as-of the instruction *after* the last step.
  let last = steps.last().unwrap();
  vm_states.push(VmState {
    opcode: last.opcode,
    pc: last.pc.saturating_add(1),
    sp: last.stack_outputs.len(),
    stack: last.stack_outputs.clone(),
    memory_root: [0u8; 32],
  });

  prove_execution_chain(&vm_states, windows)
}
