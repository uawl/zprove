//! Shared preprocessed matrix infrastructure for the StackIR and LUT STARKs.
//!
//! `rows = compile_proof(semantic_proof)` is committed as a preprocessed trace
//! shared by the StackIR and LUT STARKs.  Both proofs include this commitment
//! in their Fiat-Shamir transcript, binding them to the same ProofRow set.
//!
//! Column layout in the preprocessed matrix (NUM_PREP_COLS = 18):
//!   0 op  1 scalar0  2 scalar1  3 scalar2  4 arg0  5 arg1  6 arg2  7 value  8 ret_ty
//!   9 evm_opcode  10..17 wff_digest[0..7]
//!
//! Columns 9-17 are identical across all rows of a given proof.  They store the
//! EVM opcode (`pis[1]`) and the tag-independent WFF Poseidon digest (`pis[2..10]`).
//! Both StackIrAirWithPrep and LutKernelAirWithPrep bind these to the corresponding
//! public-input slots on the first row, closing the gap that previously let a forger
//! prove valid-arithmetic rows for a different WFF claim.

use super::types::{CircleStarkConfig, Val, make_circle_config};
use super::batch::BatchProofRowsManifest;
use crate::semantic_proof::{OP_EQ_REFL, ProofRow, RET_WFF_EQ};

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_uni_stark::{PreprocessedProverData, PreprocessedVerifierKey, setup_preprocessed};

pub const PREP_COL_OP: usize = 0;
pub const PREP_COL_SCALAR0: usize = 1;
pub const PREP_COL_SCALAR1: usize = 2;
pub const PREP_COL_SCALAR2: usize = 3;
pub const PREP_COL_ARG0: usize = 4;
pub const PREP_COL_ARG1: usize = 5;
pub const PREP_COL_ARG2: usize = 6;
pub const PREP_COL_VALUE: usize = 7;
pub const PREP_COL_RET_TY: usize = 8;
/// EVM opcode column — must equal `pis[1]` (checked by both AIR evals via `when_first_row`).
pub const PREP_COL_EVM_OPCODE: usize = 9;
/// First of 8 columns holding the tag-independent WFF Poseidon digest.
/// `PREP_COL_WFF_DIGEST_START + k` (k = 0..7) must equal `pis[2 + k]`.
pub const PREP_COL_WFF_DIGEST_START: usize = 10;
/// Total columns in the **single-instruction** preprocessed matrix.
pub const NUM_PREP_COLS: usize = 18;

// ============================================================
// Phase 3: Batch preprocessed matrix — constants and builder
//
// Extends the single-instruction layout (NUM_PREP_COLS = 18) with two
// batch-level columns that are identical across all rows:
//
//   Col 18  PREP_COL_BATCH_N            — number of instructions in the batch
//   Col 19  PREP_COL_BATCH_DIGEST_START — first of 8 M31 manifest-digest values
//   ...     (cols 19..26)
//
// pis layout for batch STARKs (same RECEIPT_BIND_PUBLIC_VALUES_LEN = 10):
//   pis[0] = tag
//   pis[1] = N            ← bound to PREP_COL_BATCH_N   at row 0
//   pis[2..10] = digest   ← bound to PREP_COL_BATCH_DIGEST_START+k at row 0
//
// Per-instruction PREP_COL_EVM_OPCODE and PREP_COL_WFF_DIGEST_* are still
// stored in the batch matrix (cols 9-17) and change per instruction segment,
// but are NOT directly bound to pis — the verifier checks them out-of-circuit
// via `compute_batch_manifest_digest`.
// ============================================================

/// Column holding the batch instruction count N; same value in every row.
pub const PREP_COL_BATCH_N: usize = 18;
/// First of 8 columns holding the batch manifest Poseidon digest.
/// `PREP_COL_BATCH_DIGEST_START + k` (k = 0..7) bound to `pis[2 + k]`.
pub const PREP_COL_BATCH_DIGEST_START: usize = 19;
/// Total columns in the **batch** preprocessed matrix.
pub const NUM_BATCH_PREP_COLS: usize = 27;

/// Build the batch preprocessed matrix from a [`BatchProofRowsManifest`] and
/// its corresponding batch public-values vector.
///
/// **Layout per row (NUM_BATCH_PREP_COLS = 27 columns):**
/// - Cols 0..8 : ProofRow fields (`op`, `scalar0`–`scalar2`, `arg0`–`arg2`,
///              `value`, `ret_ty`) — identical to the single-instruction matrix.
/// - Col 9    : `PREP_COL_EVM_OPCODE` — the EVM opcode for the owning
///              instruction segment; changes at instruction boundaries.
/// - Cols 10..17 : `PREP_COL_WFF_DIGEST_START + k` — per-instruction WFF
///              digest; changes at instruction boundaries.
/// - Col 18   : `PREP_COL_BATCH_N` — N, replicated in every row.
/// - Cols 19..26 : `PREP_COL_BATCH_DIGEST_START + k` — batch manifest digest,
///              replicated in every row.
///
/// Height = `all_rows.len().max(4).next_power_of_two()` (power-of-2 FRI requirement).
/// Padding rows beyond `all_rows.len()` use `op = OP_EQ_REFL`, `ret_ty =
/// RET_WFF_EQ`, and replicate the batch metadata columns.
pub fn build_batch_proof_rows_preprocessed_matrix(
  manifest: &BatchProofRowsManifest,
  pv: &[Val],
) -> RowMajorMatrix<Val> {
  let rows = &manifest.all_rows;
  let n_rows = rows.len().max(4).next_power_of_two();
  let mut matrix = RowMajorMatrix::new(
    Val::zero_vec(n_rows * NUM_BATCH_PREP_COLS),
    NUM_BATCH_PREP_COLS,
  );

  // Batch-level values from pv: pv[1] = N, pv[2..10] = batch_digest.
  let batch_n = if pv.len() > 1 { pv[1] } else { Val::ZERO };
  let batch_digest: &[Val] = if pv.len() >= 10 { &pv[2..10] } else { &[] };

  // Cursor-based per-instruction opcode/digest lookup.
  let mut entry_idx = 0_usize;

  for (i, row) in rows.iter().enumerate() {
    while entry_idx + 1 < manifest.entries.len()
      && i >= manifest.entries[entry_idx + 1].row_start
    {
      entry_idx += 1;
    }
    let entry = &manifest.entries[entry_idx];
    let opcode_val = Val::from_u32(entry.opcode as u32);
    let per_digest = entry.wff_digest;
    let base = i * NUM_BATCH_PREP_COLS;
    matrix.values[base + PREP_COL_OP] = Val::from_u32(row.op);
    matrix.values[base + PREP_COL_SCALAR0] = Val::from_u32(row.scalar0);
    matrix.values[base + PREP_COL_SCALAR1] = Val::from_u32(row.scalar1);
    matrix.values[base + PREP_COL_SCALAR2] = Val::from_u32(row.scalar2);
    matrix.values[base + PREP_COL_ARG0] = Val::from_u32(row.arg0);
    matrix.values[base + PREP_COL_ARG1] = Val::from_u32(row.arg1);
    matrix.values[base + PREP_COL_ARG2] = Val::from_u32(row.arg2);
    matrix.values[base + PREP_COL_VALUE] = Val::from_u32(row.value);
    matrix.values[base + PREP_COL_RET_TY] = Val::from_u32(row.ret_ty);
    matrix.values[base + PREP_COL_EVM_OPCODE] = opcode_val;
    for k in 0..8 {
      matrix.values[base + PREP_COL_WFF_DIGEST_START + k] = per_digest[k];
    }
    matrix.values[base + PREP_COL_BATCH_N] = batch_n;
    for k in 0..8 {
      matrix.values[base + PREP_COL_BATCH_DIGEST_START + k] = if k < batch_digest.len() {
        batch_digest[k]
      } else {
        Val::ZERO
      };
    }
  }

  // Padding rows: op = OP_EQ_REFL, ret_ty = RET_WFF_EQ; batch metadata replicated.
  let (last_opcode, last_per_digest) = manifest
    .entries
    .last()
    .map(|e| (Val::from_u32(e.opcode as u32), e.wff_digest))
    .unwrap_or((Val::ZERO, [Val::ZERO; 8]));
  for i in rows.len()..n_rows {
    let base = i * NUM_BATCH_PREP_COLS;
    matrix.values[base + PREP_COL_OP] = Val::from_u32(OP_EQ_REFL);
    matrix.values[base + PREP_COL_RET_TY] = Val::from_u32(RET_WFF_EQ);
    matrix.values[base + PREP_COL_EVM_OPCODE] = last_opcode;
    for k in 0..8 {
      matrix.values[base + PREP_COL_WFF_DIGEST_START + k] = last_per_digest[k];
    }
    matrix.values[base + PREP_COL_BATCH_N] = batch_n;
    for k in 0..8 {
      matrix.values[base + PREP_COL_BATCH_DIGEST_START + k] = if k < batch_digest.len() {
        batch_digest[k]
      } else {
        Val::ZERO
      };
    }
  }

  matrix
}

/// Commit the batch manifest as a preprocessed trace shared by the batch
/// StackIR and batch LUT STARKs.
///
/// Returns `(ProverData, VerifierKey)`.  Both proving functions
/// (`prove_batch_stack_ir_with_prep` and `prove_batch_lut_with_prep`) must
/// receive the same pair to share an identical Fiat-Shamir transcript.
pub fn setup_batch_proof_rows_preprocessed(
  manifest: &BatchProofRowsManifest,
  pv: &[Val],
) -> Result<
  (
    PreprocessedProverData<CircleStarkConfig>,
    PreprocessedVerifierKey<CircleStarkConfig>,
  ),
  String,
> {
  let matrix = build_batch_proof_rows_preprocessed_matrix(manifest, pv);
  let n_rows = matrix.height();
  let degree_bits = n_rows.trailing_zeros() as usize;
  let config = make_circle_config();
  let holder = ProofRowsPreprocessedHolder { matrix };
  setup_preprocessed(&config, &holder, degree_bits)
    .ok_or_else(|| "batch preprocessed matrix was empty (zero width)".to_string())
}

/// Build the preprocessed matrix from compiled proof rows and public values.
///
/// `pv` must follow the layout of [`make_receipt_binding_public_values`]:
/// `[tag, opcode, digest[0..7]]`.  The EVM opcode (`pv[1]`) and WFF digest
/// (`pv[2..9]`) are replicated into columns 9-17 of every row, enabling both
/// the StackIR and LUT AIRs to bind `pis[1]` and `pis[2..10]` at row 0.
///
/// Height = `rows.len().max(4).next_power_of_two()` — identical to the height
/// used by [`build_stack_ir_trace_from_rows`], ensuring dimension compatibility
/// with `p3_uni_stark::prove_with_preprocessed`.
///
/// Padding rows (beyond `rows.len()`) replicate the StackIR padding convention:
/// `op = OP_EQ_REFL`, all other ProofRow columns zero; opcode/digest columns
/// carry the same values as live rows.
pub fn build_proof_rows_preprocessed_matrix(rows: &[ProofRow], pv: &[Val]) -> RowMajorMatrix<Val> {
  let n_rows = rows.len().max(4).next_power_of_two();
  let mut matrix = RowMajorMatrix::new(Val::zero_vec(n_rows * NUM_PREP_COLS), NUM_PREP_COLS);

  let evm_opcode = if pv.len() > 1 { pv[1] } else { Val::ZERO };
  let digest_slice: &[Val] = if pv.len() >= 10 { &pv[2..10] } else { &[] };

  for (i, row) in rows.iter().enumerate() {
    let base = i * NUM_PREP_COLS;
    matrix.values[base + PREP_COL_OP] = Val::from_u32(row.op);
    matrix.values[base + PREP_COL_SCALAR0] = Val::from_u32(row.scalar0);
    matrix.values[base + PREP_COL_SCALAR1] = Val::from_u32(row.scalar1);
    matrix.values[base + PREP_COL_SCALAR2] = Val::from_u32(row.scalar2);
    matrix.values[base + PREP_COL_ARG0] = Val::from_u32(row.arg0);
    matrix.values[base + PREP_COL_ARG1] = Val::from_u32(row.arg1);
    matrix.values[base + PREP_COL_ARG2] = Val::from_u32(row.arg2);
    matrix.values[base + PREP_COL_VALUE] = Val::from_u32(row.value);
    matrix.values[base + PREP_COL_RET_TY] = Val::from_u32(row.ret_ty);
    matrix.values[base + PREP_COL_EVM_OPCODE] = evm_opcode;
    for k in 0..8 {
      matrix.values[base + PREP_COL_WFF_DIGEST_START + k] = if k < digest_slice.len() {
        digest_slice[k]
      } else {
        Val::ZERO
      };
    }
  }

  // Padding: op = OP_EQ_REFL, ret_ty = RET_WFF_EQ; opcode/digest same as live rows.
  for i in rows.len()..n_rows {
    let base = i * NUM_PREP_COLS;
    matrix.values[base + PREP_COL_OP] = Val::from_u32(OP_EQ_REFL);
    matrix.values[base + PREP_COL_RET_TY] = Val::from_u32(RET_WFF_EQ);
    matrix.values[base + PREP_COL_EVM_OPCODE] = evm_opcode;
    for k in 0..8 {
      matrix.values[base + PREP_COL_WFF_DIGEST_START + k] = if k < digest_slice.len() {
        digest_slice[k]
      } else {
        Val::ZERO
      };
    }
  }

  matrix
}

/// Dummy AIR used exclusively to drive `p3_uni_stark::setup_preprocessed`.
///
/// It carries no main-trace columns and no constraints; its only purpose is to
/// return the dynamic preprocessed matrix via `BaseAir::preprocessed_trace`.
pub(super) struct ProofRowsPreprocessedHolder {
  pub(super) matrix: RowMajorMatrix<Val>,
}

impl BaseAir<Val> for ProofRowsPreprocessedHolder {
  /// No main-trace columns — this AIR exists only as a preprocessed-trace holder.
  fn width(&self) -> usize {
    0
  }

  fn preprocessed_trace(&self) -> Option<RowMajorMatrix<Val>> {
    Some(self.matrix.clone())
  }
}

/// Blanket Air impl: no constraints, any base-field builder.
impl<AB: AirBuilder<F = Val>> Air<AB> for ProofRowsPreprocessedHolder {
  fn eval(&self, _builder: &mut AB) {}
}

/// Commit the compiled proof rows as a preprocessed trace.
///
/// Returns `(ProverData, VerifierKey)` bound to the specific `rows` content.
/// Both are tied to `degree_bits = log2(rows.len().max(4).next_power_of_two())`.
///
/// **Both** [`prove_stack_ir_with_prep`] and the analogous LUT prove function
/// must receive the *same* `ProverData` / `VerifierKey` pair so that their
/// Fiat-Shamir transcripts commit to an identical preprocessed matrix.
pub fn setup_proof_rows_preprocessed(
  rows: &[ProofRow],
  pv: &[Val],
) -> Result<
  (
    PreprocessedProverData<CircleStarkConfig>,
    PreprocessedVerifierKey<CircleStarkConfig>,
  ),
  String,
> {
  let matrix = build_proof_rows_preprocessed_matrix(rows, pv);
  let n_rows = matrix.height();
  // n_rows is already a power of two (guaranteed by build_proof_rows_preprocessed_matrix).
  let degree_bits = n_rows.trailing_zeros() as usize;

  let config = make_circle_config();
  let holder = ProofRowsPreprocessedHolder { matrix };

  setup_preprocessed(&config, &holder, degree_bits)
    .ok_or_else(|| "preprocessed matrix was empty (zero width)".to_string())
}
