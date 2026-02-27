//! Batch instruction metadata and manifest digest helpers.

use super::types::{RECEIPT_BIND_PUBLIC_VALUES_LEN, Val, default_poseidon_sponge};
use crate::semantic_proof::WFF;
use p3_field::PrimeCharacteristicRing;
use p3_symmetric::CryptographicHasher;

/// Per-instruction metadata within a [`BatchProofRowsManifest`].
///
/// Stores the EVM opcode, inferred WFF, and the half-open row range
/// `[row_start, row_start + row_count)` inside
/// [`BatchProofRowsManifest::all_rows`].
#[derive(Debug, Clone)]
pub struct BatchInstructionMeta {
  /// EVM opcode (e.g. `0x01` = ADD).
  pub opcode: u8,
  /// Well-formed formula inferred from the instruction's semantic proof.
  pub wff: WFF,
  /// Poseidon digest of `(opcode, wff_bytes)` — precomputed once in
  /// `build_batch_manifest` to avoid redundant hashing in the proving pipeline.
  pub wff_digest: [Val; 8],
  /// Index of the first `ProofRow` in `BatchProofRowsManifest::all_rows`.
  pub row_start: usize,
  /// Number of ProofRows contributed by this instruction.
  pub row_count: usize,
}

/// Concatenated [`ProofRow`]s for N instructions with per-instruction
/// boundary metadata.
///
/// Produced by `transition::build_batch_manifest`; consumed by the batch
/// STARK proving functions in this module.
#[derive(Debug, Clone)]
pub struct BatchProofRowsManifest {
  /// Per-instruction metadata in execution order.
  pub entries: Vec<BatchInstructionMeta>,
  /// All ProofRows from all instructions, in order.
  pub all_rows: Vec<crate::semantic_proof::ProofRow>,
}

// ============================================================
// Phase 2: Batch public values and manifest digest
// ============================================================

/// Compute the **batch manifest digest**: a Poseidon hash of all N instructions'
/// `(opcode, wff_digest)` pairs plus the instruction count.
///
/// Input layout fed to the sponge:
/// `[N, opcode₀, digest₀[0..7], opcode₁, digest₁[0..7], …]`
///
/// The resulting 8-element M31 digest is tag-independent and stored in
/// `pis[2..10]` for both the batch StackIR and batch LUT STARKs.
/// Both STARKs use the same preprocessed matrix so only one digest is needed.
pub fn compute_batch_manifest_digest(entries: &[BatchInstructionMeta]) -> [Val; 8] {
  let sponge = default_poseidon_sponge();
  let mut input: Vec<Val> = Vec::new();
  input.push(Val::from_u32(entries.len() as u32));
  for entry in entries {
    input.push(Val::from_u32(entry.opcode as u32));
    input.extend_from_slice(&entry.wff_digest);
  }
  sponge.hash_iter(input)
}

/// Build the public-values vector for a batch receipt-binding STARK.
///
/// Layout (length = `RECEIPT_BIND_PUBLIC_VALUES_LEN = 10`):
/// `[tag, N, batch_digest[0..8]]`
///
/// - `pis[0]` = tag (`RECEIPT_BIND_TAG_STACK` or `RECEIPT_BIND_TAG_LUT`)
/// - `pis[1]` = N (number of instructions in the batch)
/// - `pis[2..10]` = `compute_batch_manifest_digest(entries)` — tag-independent
///   so both the StackIR and LUT AIRs can reference it via the same prep column.
pub fn make_batch_receipt_binding_public_values(
  tag: u32,
  entries: &[BatchInstructionMeta],
) -> Vec<Val> {
  let digest = compute_batch_manifest_digest(entries);
  let mut pv = Vec::with_capacity(RECEIPT_BIND_PUBLIC_VALUES_LEN);
  pv.push(Val::from_u32(tag));
  pv.push(Val::from_u32(entries.len() as u32));
  pv.extend_from_slice(&digest);
  pv
}
