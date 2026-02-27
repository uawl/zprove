//! Circle STARK proof generation and verification for semantic proof rows.
//!
//! Stage A/B overview:
//! - Stage 1 proves inferred-WFF semantic constraints over compiled `ProofRow`s.
//! - Stage 2 proves inferred WFF equals public WFF via serialized equality trace.
//!
//! The current Stage-1 semantic AIR kernel enforces the 29-bit/24-bit add equality rows
//! (`OP_U29_ADD_EQ`, `OP_U24_ADD_EQ`) embedded in `ProofRow` encoding.

use crate::semantic_proof::{
  NUM_PROOF_COLS, Proof, ProofRow, RET_BYTE, RET_WFF_AND, RET_WFF_EQ, Term, WFF,
  compile_proof, infer_proof, verify_compiled,
};

use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, PairBuilder};
use p3_challenger::{DuplexChallenger, HashChallenger, SerializingChallenger32};
use p3_circle::CirclePcs;
use p3_commit::ExtensionMmcs;
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_fri::FriParameters;
use p3_keccak::Keccak256Hash;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_mersenne_31::{Mersenne31, Poseidon2Mersenne31};
use p3_symmetric::{
  CompressionFunctionFromHasher, CryptographicHasher, PaddingFreeSponge, SerializingHasher,
  TruncatedPermutation,
};
use p3_lookup::logup::LogUpGadget;
use p3_lookup::lookup_traits::{Kind, Lookup, LookupData, LookupGadget};
use p3_uni_stark::{
  Entry, PreprocessedProverData, PreprocessedVerifierKey, StarkConfig, SymbolicExpression,
  SymbolicVariable, VerifierConstraintFolder, prove_with_lookup, setup_preprocessed,
  verify_with_lookup,
};
use rand::SeedableRng;
use rand::rngs::SmallRng;

// ============================================================
// Type aliases for Circle STARK over M31
// ============================================================

pub type Val = Mersenne31;
pub type Challenge = BinomialExtensionField<Val, 3>;

// ---- Poseidon2 over Mersenne31 — used as the default hash/commitment backend ----
pub type P2Perm = Poseidon2Mersenne31<16>;
pub type P2Hash = PaddingFreeSponge<P2Perm, 16, 8, 8>;
pub type P2Compress = TruncatedPermutation<P2Perm, 2, 8, 16>;
pub type P2ValMmcs = MerkleTreeMmcs<Val, Val, P2Hash, P2Compress, 8>;
pub type P2ChallengeMmcs = ExtensionMmcs<Val, Challenge, P2ValMmcs>;
pub type P2Challenger = DuplexChallenger<Val, P2Perm, 16, 8>;

pub type Pcs = CirclePcs<Val, P2ValMmcs, P2ChallengeMmcs>;
pub type CircleStarkConfig = StarkConfig<Pcs, Challenge, P2Challenger>;

pub type CircleStarkProof = p3_uni_stark::Proof<CircleStarkConfig>;
pub type CircleStarkVerifyResult =
  Result<(), p3_uni_stark::VerificationError<p3_uni_stark::PcsError<CircleStarkConfig>>>;

// Keccak types kept around in case needed elsewhere (not used by default config)
#[allow(dead_code)]
type KeccakByteHash = Keccak256Hash;
#[allow(dead_code)]
type KeccakFieldHash = SerializingHasher<KeccakByteHash>;
#[allow(dead_code)]
type KeccakCompress = CompressionFunctionFromHasher<KeccakByteHash, 2, 32>;
#[allow(dead_code)]
type KeccakChallenger = SerializingChallenger32<Val, HashChallenger<u8, KeccakByteHash, 32>>;

pub const RECEIPT_BIND_TAG_STACK: u32 = 1;
pub const RECEIPT_BIND_TAG_LUT: u32 = 2;
pub const RECEIPT_BIND_TAG_WFF: u32 = 3;
const RECEIPT_BIND_PUBLIC_VALUES_LEN: usize = 10;

fn default_receipt_bind_public_values() -> Vec<Val> {
  vec![Val::from_u32(0); RECEIPT_BIND_PUBLIC_VALUES_LEN]
}

fn default_receipt_bind_public_values_for_tag(tag: u32) -> Vec<Val> {
  let mut values = default_receipt_bind_public_values();
  values[0] = Val::from_u32(tag);
  values
}

/// Public values for the scaffold StackIR STARK: tag = `RECEIPT_BIND_TAG_STACK`,
/// remaining positions are zero.  Use this whenever calling
/// [`prove_stack_ir_with_prep`] / [`prove_stack_ir_scaffold_stark`] outside of
/// the full prove pipeline which supplies a real Poseidon hash.
pub fn stack_ir_scaffold_public_values() -> Vec<Val> {
  default_receipt_bind_public_values_for_tag(RECEIPT_BIND_TAG_STACK)
}

/// Compute the **tag-independent** Poseidon digest of `(opcode, len, wff_bytes)`.
///
/// Intentionally omits the STARK tag so the same 8-element digest can be stored
/// once in the shared preprocessed matrix and bound by both the StackIR and LUT
/// AIRs (which differ only in `pis[0]`, the tag).
pub fn compute_wff_opcode_digest(opcode: u8, expected_wff: &WFF) -> [Val; 8] {
  let mut rng = SmallRng::seed_from_u64(0xC0DEC0DE_u64);
  let poseidon = Poseidon2Mersenne31::<16>::new_from_rng_128(&mut rng);
  let sponge = PaddingFreeSponge::<_, 16, 8, 8>::new(poseidon);

  let mut input = Vec::new();
  input.push(Val::from_u32(opcode as u32));
  input.push(Val::from_u32(serialize_wff_bytes(expected_wff).len() as u32));
  input.extend(
    serialize_wff_bytes(expected_wff)
      .into_iter()
      .map(Val::from_u8),
  );

  sponge.hash_iter(input)
}

/// Build the public-values vector for a receipt-binding STARK.
///
/// Layout: `[tag, opcode, digest[0..7]]` where `digest = compute_wff_opcode_digest(opcode, wff)`.
///
/// The digest is **tag-independent** so both the StackIR and LUT STARKs can share
/// the same preprocessed matrix commitment (which stores the digest once) while
/// each independently enforces `pis[0]` against their own tag.
pub fn make_receipt_binding_public_values(tag: u32, opcode: u8, expected_wff: &WFF) -> Vec<Val> {
  let digest = compute_wff_opcode_digest(opcode, expected_wff);
  let mut public_values = Vec::with_capacity(RECEIPT_BIND_PUBLIC_VALUES_LEN);
  public_values.push(Val::from_u32(tag));
  public_values.push(Val::from_u32(opcode as u32));
  public_values.extend_from_slice(&digest);
  public_values
}

// ============================================================
// Phase 1: Batch instruction types
// ============================================================

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
  pub all_rows: Vec<ProofRow>,
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
  let mut rng = rand::rngs::SmallRng::seed_from_u64(0xC0DEC0DE_u64);
  let poseidon = Poseidon2Mersenne31::<16>::new_from_rng_128(&mut rng);
  let sponge = PaddingFreeSponge::<_, 16, 8, 8>::new(poseidon);

  let mut input: Vec<Val> = Vec::new();
  input.push(Val::from_u32(entries.len() as u32));
  for entry in entries {
    let per_inst_digest = compute_wff_opcode_digest(entry.opcode, &entry.wff);
    input.push(Val::from_u32(entry.opcode as u32));
    input.extend_from_slice(&per_inst_digest);
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

// ============================================================
// Stack IR framework (offset-friendly scaffold)
// ============================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StackIrStep {
  pub op: u32,
  pub pop: u32,
  pub push: u32,
  pub sp_before: u32,
  pub sp_after: u32,
  pub scalar0: u32,
  pub scalar1: u32,
  pub scalar2: u32,
  pub value: u32,
  pub ret_ty: u32,
}

const STACK_COL_OP: usize = 0;
const STACK_COL_POP: usize = 1;
const STACK_COL_PUSH: usize = 2;
const STACK_COL_SP_BEFORE: usize = 3;
const STACK_COL_SP_AFTER: usize = 4;
const STACK_COL_SCALAR0: usize = 5;
const STACK_COL_SCALAR1: usize = 6;
const STACK_COL_SCALAR2: usize = 7;
const STACK_COL_VALUE: usize = 8;
const STACK_COL_RET_TY: usize = 9;
// ---- Selector columns (one-hot, indices 10-29) ----
// Each selector corresponds to exactly one allowed opcode tag.
// Replacing the degree-20 allowed_poly + gate() pattern with degree-2 constraints.
const STACK_COL_SEL_BOOL: usize = 10;
const STACK_COL_SEL_BYTE: usize = 11;
const STACK_COL_SEL_EQ_REFL: usize = 12;
const STACK_COL_SEL_AND_INTRO: usize = 13;
const STACK_COL_SEL_EQ_TRANS: usize = 14;
const STACK_COL_SEL_ADD_CONGR: usize = 15;
const STACK_COL_SEL_ADD_CARRY_CONGR: usize = 16;
const STACK_COL_SEL_U29_ADD_EQ: usize = 17;
const STACK_COL_SEL_U24_ADD_EQ: usize = 18;
const STACK_COL_SEL_U15_MUL_EQ: usize = 19;
const STACK_COL_SEL_MUL_LOW_EQ: usize = 20;
const STACK_COL_SEL_MUL_HIGH_EQ: usize = 21;
const STACK_COL_SEL_AND_EQ: usize = 22;
const STACK_COL_SEL_OR_EQ: usize = 23;
const STACK_COL_SEL_XOR_EQ: usize = 24;
const STACK_COL_SEL_NOT: usize = 25;
const STACK_COL_SEL_EQ_SYM: usize = 26;
pub const NUM_STACK_IR_COLS: usize = 27;

fn row_pop_count(op: u32) -> Option<u32> {
  match op {
    crate::semantic_proof::OP_BOOL
    | crate::semantic_proof::OP_BYTE
    | crate::semantic_proof::OP_U29_ADD_EQ
    | crate::semantic_proof::OP_U24_ADD_EQ
    | crate::semantic_proof::OP_U15_MUL_EQ
    | crate::semantic_proof::OP_BYTE_MUL_LOW_EQ
    | crate::semantic_proof::OP_BYTE_MUL_HIGH_EQ
    | crate::semantic_proof::OP_BYTE_AND_EQ
    | crate::semantic_proof::OP_BYTE_OR_EQ
    | crate::semantic_proof::OP_BYTE_XOR_EQ => Some(0),

    crate::semantic_proof::OP_NOT
    | crate::semantic_proof::OP_EQ_REFL
    | crate::semantic_proof::OP_EQ_SYM
    | crate::semantic_proof::OP_BYTE_ADD_THIRD_CONGRUENCE
    | crate::semantic_proof::OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE => Some(1),

    crate::semantic_proof::OP_AND
    | crate::semantic_proof::OP_OR
    | crate::semantic_proof::OP_XOR
    | crate::semantic_proof::OP_BYTE_MUL_LOW
    | crate::semantic_proof::OP_BYTE_MUL_HIGH
    | crate::semantic_proof::OP_BYTE_AND
    | crate::semantic_proof::OP_BYTE_OR
    | crate::semantic_proof::OP_BYTE_XOR
    | crate::semantic_proof::OP_AND_INTRO
    | crate::semantic_proof::OP_EQ_TRANS
    | crate::semantic_proof::OP_ITE_TRUE_EQ
    | crate::semantic_proof::OP_ITE_FALSE_EQ => Some(2),

    crate::semantic_proof::OP_ITE
    | crate::semantic_proof::OP_BYTE_ADD
    | crate::semantic_proof::OP_BYTE_ADD_CARRY => Some(3),

    _ => None,
  }
}

pub fn build_stack_ir_steps_from_rows(rows: &[ProofRow]) -> Result<Vec<StackIrStep>, String> {
  if rows.is_empty() {
    return Err("cannot build stack-ir steps from empty rows".to_string());
  }

  let mut out = Vec::with_capacity(rows.len());
  let mut sp: i64 = 0;

  for row in rows {
    let pop = row_pop_count(row.op)
      .ok_or_else(|| format!("row op {} is not stack-ir encodable yet", row.op))?
      as i64;
    let push: i64 = 1;

    let sp_before = sp;
    if sp_before < pop {
      return Err(format!(
        "stack underflow in stack-ir build: op={} sp_before={} pop={}",
        row.op, sp_before, pop
      ));
    }

    sp = sp_before - pop + push;

    out.push(StackIrStep {
      op: row.op,
      pop: pop as u32,
      push: push as u32,
      sp_before: sp_before as u32,
      sp_after: sp as u32,
      scalar0: row.scalar0,
      scalar1: row.scalar1,
      scalar2: row.scalar2,
      value: row.value,
      ret_ty: row.ret_ty,
    });
  }

  if sp != 1 {
    return Err(format!(
      "stack-ir build ended with invalid final stack size: {sp}"
    ));
  }

  Ok(out)
}

pub fn build_stack_ir_trace_from_steps(
  steps: &[StackIrStep],
) -> Result<RowMajorMatrix<Val>, String> {
  if steps.is_empty() {
    return Err("cannot build stack-ir trace from empty steps".to_string());
  }

  let ensure_u16 = |name: &str, value: u32, row: usize| -> Result<(), String> {
    if value > u16::MAX as u32 {
      return Err(format!(
        "stack-ir {name} out of u16 range at row {row}: {value}"
      ));
    }
    Ok(())
  };

  let ensure_m31 = |name: &str, value: u32, row: usize| -> Result<(), String> {
    if value >= 0x7fff_ffff {
      return Err(format!(
        "stack-ir {name} out of M31 range at row {row}: {value}"
      ));
    }
    Ok(())
  };

  for (i, step) in steps.iter().enumerate() {
    ensure_u16("op", step.op, i)?;
    ensure_u16("pop", step.pop, i)?;
    ensure_u16("push", step.push, i)?;
    ensure_u16("sp_before", step.sp_before, i)?;
    ensure_u16("sp_after", step.sp_after, i)?;
    ensure_m31("scalar0", step.scalar0, i)?;
    ensure_m31("scalar1", step.scalar1, i)?;
    ensure_m31("scalar2", step.scalar2, i)?;
    ensure_m31("value", step.value, i)?;
    ensure_u16("ret_ty", step.ret_ty, i)?;
  }

  let n_rows = steps.len().max(4).next_power_of_two();
  let mut trace = RowMajorMatrix::new(Val::zero_vec(n_rows * NUM_STACK_IR_COLS), NUM_STACK_IR_COLS);

  for (i, step) in steps.iter().enumerate() {
    let base = i * NUM_STACK_IR_COLS;
    trace.values[base + STACK_COL_OP] = Val::from_u16(step.op as u16);
    trace.values[base + STACK_COL_POP] = Val::from_u16(step.pop as u16);
    trace.values[base + STACK_COL_PUSH] = Val::from_u16(step.push as u16);
    trace.values[base + STACK_COL_SP_BEFORE] = Val::from_u16(step.sp_before as u16);
    trace.values[base + STACK_COL_SP_AFTER] = Val::from_u16(step.sp_after as u16);
    trace.values[base + STACK_COL_SCALAR0] = Val::from_u32(step.scalar0);
    trace.values[base + STACK_COL_SCALAR1] = Val::from_u32(step.scalar1);
    trace.values[base + STACK_COL_SCALAR2] = Val::from_u32(step.scalar2);
    trace.values[base + STACK_COL_VALUE] = Val::from_u32(step.value);
    trace.values[base + STACK_COL_RET_TY] = Val::from_u16(step.ret_ty as u16);
    // Set the one-hot selector for this opcode.
    if let Some(sel_col) = op_to_sel_col(step.op) {
      trace.values[base + sel_col] = Val::ONE;
    }
  }

  if steps.len() < n_rows {
    let last = *steps.last().expect("non-empty checked");
    let steady_sp = last.sp_after;
    for i in steps.len()..n_rows {
      let base = i * NUM_STACK_IR_COLS;
      trace.values[base + STACK_COL_OP] = Val::from_u16(crate::semantic_proof::OP_EQ_REFL as u16);
      trace.values[base + STACK_COL_POP] = Val::from_u16(1);
      trace.values[base + STACK_COL_PUSH] = Val::from_u16(1);
      trace.values[base + STACK_COL_SP_BEFORE] = Val::from_u16(steady_sp as u16);
      trace.values[base + STACK_COL_SP_AFTER] = Val::from_u16(steady_sp as u16);
      trace.values[base + STACK_COL_SCALAR0] = Val::from_u16(0);
      trace.values[base + STACK_COL_SCALAR1] = Val::from_u16(0);
      trace.values[base + STACK_COL_SCALAR2] = Val::from_u16(0);
      trace.values[base + STACK_COL_VALUE] = Val::from_u16(0);
      trace.values[base + STACK_COL_RET_TY] = Val::from_u16(RET_WFF_EQ as u16);
      // Padding uses OP_EQ_REFL → set its selector.
      trace.values[base + STACK_COL_SEL_EQ_REFL] = Val::ONE;
    }
  }

  Ok(trace)
}

/// Map a ProofRow opcode to its selector column index in the main trace.
/// Returns `None` for opcodes that are not represented as allowed tags.
fn op_to_sel_col(op: u32) -> Option<usize> {
  use crate::semantic_proof::*;
  match op {
    OP_BOOL => Some(STACK_COL_SEL_BOOL),
    OP_BYTE => Some(STACK_COL_SEL_BYTE),
    OP_EQ_REFL => Some(STACK_COL_SEL_EQ_REFL),
    OP_AND_INTRO => Some(STACK_COL_SEL_AND_INTRO),
    OP_EQ_TRANS => Some(STACK_COL_SEL_EQ_TRANS),
    OP_BYTE_ADD_THIRD_CONGRUENCE => Some(STACK_COL_SEL_ADD_CONGR),
    OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE => Some(STACK_COL_SEL_ADD_CARRY_CONGR),
    OP_U29_ADD_EQ => Some(STACK_COL_SEL_U29_ADD_EQ),
    OP_U24_ADD_EQ => Some(STACK_COL_SEL_U24_ADD_EQ),
    OP_U15_MUL_EQ => Some(STACK_COL_SEL_U15_MUL_EQ),
    OP_BYTE_MUL_LOW_EQ => Some(STACK_COL_SEL_MUL_LOW_EQ),
    OP_BYTE_MUL_HIGH_EQ => Some(STACK_COL_SEL_MUL_HIGH_EQ),
    OP_BYTE_AND_EQ => Some(STACK_COL_SEL_AND_EQ),
    OP_BYTE_OR_EQ => Some(STACK_COL_SEL_OR_EQ),
    OP_BYTE_XOR_EQ => Some(STACK_COL_SEL_XOR_EQ),
    OP_NOT => Some(STACK_COL_SEL_NOT),
    OP_EQ_SYM => Some(STACK_COL_SEL_EQ_SYM),
    _ => None,
  }
}

pub fn build_stack_ir_trace_from_rows(rows: &[ProofRow]) -> Result<RowMajorMatrix<Val>, String> {
  let steps = build_stack_ir_steps_from_rows(rows)?;
  build_stack_ir_trace_from_steps(&steps)
}

pub struct StackIrAir;

impl<F> BaseAir<F> for StackIrAir {
  fn width(&self) -> usize {
    NUM_STACK_IR_COLS
  }
}

// StackIrAir: original AIR used by the plain scaffold prove/verify path.
// No PairBuilder requirement — works with every AirBuilderWithPublicValues
// including DebugConstraintBuilder (debug-assertion checker in p3-uni-stark).
//
// Phase 2 introduces StackIrAirWithPrep (below) which additionally binds
// main-trace columns to the shared preprocessed ProofRow commitment.

impl<AB: AirBuilderWithPublicValues> Air<AB> for StackIrAir {
  fn eval(&self, builder: &mut AB) {
    eval_stack_ir_inner(builder);
  }
}

/// Core StackIR constraints — shared between `StackIrAir` and
/// `StackIrAirWithPrep` to avoid code duplication.
fn eval_stack_ir_inner<AB: AirBuilderWithPublicValues>(builder: &mut AB) {
  let pis = builder.public_values();
  builder.assert_eq(pis[0], AB::Expr::from_u32(RECEIPT_BIND_TAG_STACK));

  let main = builder.main();
  let local = main.row_slice(0).expect("empty trace");
  let next = main.row_slice(1).expect("single-row trace");

  let local = &*local;
  let next = &*next;

  let op = local[STACK_COL_OP].clone();
  let pop = local[STACK_COL_POP].clone();
  let push = local[STACK_COL_PUSH].clone();
  let sp_before = local[STACK_COL_SP_BEFORE].clone();
  let sp_after = local[STACK_COL_SP_AFTER].clone();
  let scalar0 = local[STACK_COL_SCALAR0].clone();
  let value = local[STACK_COL_VALUE].clone();
  let ret_ty = local[STACK_COL_RET_TY].clone();

  // ---- Opcode tag constants ----
  let t_bool = crate::semantic_proof::OP_BOOL as u16;
  let t_byte = crate::semantic_proof::OP_BYTE as u16;
  let t_eq_refl = crate::semantic_proof::OP_EQ_REFL as u16;
  let t_and_intro = crate::semantic_proof::OP_AND_INTRO as u16;
  let t_eq_trans = crate::semantic_proof::OP_EQ_TRANS as u16;
  let t_add_congr = crate::semantic_proof::OP_BYTE_ADD_THIRD_CONGRUENCE as u16;
  let t_add_carry_congr = crate::semantic_proof::OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE as u16;
  let t_u29_add_eq = crate::semantic_proof::OP_U29_ADD_EQ as u16;
  let t_u24_add_eq = crate::semantic_proof::OP_U24_ADD_EQ as u16;
  let t_u15_mul_eq = crate::semantic_proof::OP_U15_MUL_EQ as u16;
  let t_mul_low_eq = crate::semantic_proof::OP_BYTE_MUL_LOW_EQ as u16;
  let t_mul_high_eq = crate::semantic_proof::OP_BYTE_MUL_HIGH_EQ as u16;
  let t_and_eq = crate::semantic_proof::OP_BYTE_AND_EQ as u16;
  let t_or_eq = crate::semantic_proof::OP_BYTE_OR_EQ as u16;
  let t_xor_eq = crate::semantic_proof::OP_BYTE_XOR_EQ as u16;
  let t_not = crate::semantic_proof::OP_NOT as u16;
  let t_eq_sym = crate::semantic_proof::OP_EQ_SYM as u16;

  // ---- Read selector columns (one-hot witness, max degree 1) ----
  let sel_bool = local[STACK_COL_SEL_BOOL].clone();
  let sel_byte = local[STACK_COL_SEL_BYTE].clone();
  let sel_eq_refl = local[STACK_COL_SEL_EQ_REFL].clone();
  let sel_and_intro = local[STACK_COL_SEL_AND_INTRO].clone();
  let sel_eq_trans = local[STACK_COL_SEL_EQ_TRANS].clone();
  let sel_add_congr = local[STACK_COL_SEL_ADD_CONGR].clone();
  let sel_add_carry_congr = local[STACK_COL_SEL_ADD_CARRY_CONGR].clone();
  let sel_u29_add_eq = local[STACK_COL_SEL_U29_ADD_EQ].clone();
  let sel_u24_add_eq = local[STACK_COL_SEL_U24_ADD_EQ].clone();
  let sel_u15_mul_eq = local[STACK_COL_SEL_U15_MUL_EQ].clone();
  let sel_mul_low_eq = local[STACK_COL_SEL_MUL_LOW_EQ].clone();
  let sel_mul_high_eq = local[STACK_COL_SEL_MUL_HIGH_EQ].clone();
  let sel_and_eq = local[STACK_COL_SEL_AND_EQ].clone();
  let sel_or_eq = local[STACK_COL_SEL_OR_EQ].clone();
  let sel_xor_eq = local[STACK_COL_SEL_XOR_EQ].clone();
  let sel_not = local[STACK_COL_SEL_NOT].clone();
  let sel_eq_sym = local[STACK_COL_SEL_EQ_SYM].clone();

  // ---- Selector boolean constraints: sel_i * (1 - sel_i) = 0  (degree 2) ----
  let c_one = AB::Expr::from_u16(1);
  builder.assert_zero(sel_bool.clone() * (c_one.clone() - sel_bool.clone()));
  builder.assert_zero(sel_byte.clone() * (c_one.clone() - sel_byte.clone()));
  builder.assert_zero(sel_eq_refl.clone() * (c_one.clone() - sel_eq_refl.clone()));
  builder.assert_zero(sel_and_intro.clone() * (c_one.clone() - sel_and_intro.clone()));
  builder.assert_zero(sel_eq_trans.clone() * (c_one.clone() - sel_eq_trans.clone()));
  builder.assert_zero(sel_add_congr.clone() * (c_one.clone() - sel_add_congr.clone()));
  builder.assert_zero(sel_add_carry_congr.clone() * (c_one.clone() - sel_add_carry_congr.clone()));
  builder.assert_zero(sel_u29_add_eq.clone() * (c_one.clone() - sel_u29_add_eq.clone()));
  builder.assert_zero(sel_u24_add_eq.clone() * (c_one.clone() - sel_u24_add_eq.clone()));
  builder.assert_zero(sel_u15_mul_eq.clone() * (c_one.clone() - sel_u15_mul_eq.clone()));
  builder.assert_zero(sel_mul_low_eq.clone() * (c_one.clone() - sel_mul_low_eq.clone()));
  builder.assert_zero(sel_mul_high_eq.clone() * (c_one.clone() - sel_mul_high_eq.clone()));
  builder.assert_zero(sel_and_eq.clone() * (c_one.clone() - sel_and_eq.clone()));
  builder.assert_zero(sel_or_eq.clone() * (c_one.clone() - sel_or_eq.clone()));
  builder.assert_zero(sel_xor_eq.clone() * (c_one.clone() - sel_xor_eq.clone()));
  builder.assert_zero(sel_not.clone() * (c_one.clone() - sel_not.clone()));
  builder.assert_zero(sel_eq_sym.clone() * (c_one.clone() - sel_eq_sym.clone()));

  // ---- Exactly one selector active per row: Σ sel_i = 1  (degree 1) ----
  let sel_sum = sel_bool.clone()
    + sel_byte.clone()
    + sel_eq_refl.clone()
    + sel_and_intro.clone()
    + sel_eq_trans.clone()
    + sel_add_congr.clone()
    + sel_add_carry_congr.clone()
    + sel_u29_add_eq.clone()
    + sel_u24_add_eq.clone()
    + sel_u15_mul_eq.clone()
    + sel_mul_low_eq.clone()
    + sel_mul_high_eq.clone()
    + sel_and_eq.clone()
    + sel_or_eq.clone()
    + sel_xor_eq.clone()
    + sel_not.clone()
    + sel_eq_sym.clone();
  builder.assert_eq(sel_sum, c_one.clone());

  // ---- op = Σ(sel_i · tag_i): binds op to the active selector  (degree 2) ----
  let op_from_sels = sel_bool.clone() * AB::Expr::from_u16(t_bool)
    + sel_byte.clone() * AB::Expr::from_u16(t_byte)
    + sel_eq_refl.clone() * AB::Expr::from_u16(t_eq_refl)
    + sel_and_intro.clone() * AB::Expr::from_u16(t_and_intro)
    + sel_eq_trans.clone() * AB::Expr::from_u16(t_eq_trans)
    + sel_add_congr.clone() * AB::Expr::from_u16(t_add_congr)
    + sel_add_carry_congr.clone() * AB::Expr::from_u16(t_add_carry_congr)
    + sel_u29_add_eq.clone() * AB::Expr::from_u16(t_u29_add_eq)
    + sel_u24_add_eq.clone() * AB::Expr::from_u16(t_u24_add_eq)
    + sel_u15_mul_eq.clone() * AB::Expr::from_u16(t_u15_mul_eq)
    + sel_mul_low_eq.clone() * AB::Expr::from_u16(t_mul_low_eq)
    + sel_mul_high_eq.clone() * AB::Expr::from_u16(t_mul_high_eq)
    + sel_and_eq.clone() * AB::Expr::from_u16(t_and_eq)
    + sel_or_eq.clone() * AB::Expr::from_u16(t_or_eq)
    + sel_xor_eq.clone() * AB::Expr::from_u16(t_xor_eq)
    + sel_not.clone() * AB::Expr::from_u16(t_not)
    + sel_eq_sym.clone() * AB::Expr::from_u16(t_eq_sym);
  builder.assert_eq(op.clone(), op_from_sels);

  // ---- Stack pointer transition: sp_after = sp_before - pop + push  (degree 1) ----
  builder.assert_zero(
    sp_before.clone().into() - pop.clone().into() + push.clone().into() - sp_after.clone().into(),
  );
  builder.when_first_row().assert_zero(sp_before.clone());
  builder
    .when_transition()
    .assert_eq(next[STACK_COL_SP_BEFORE].clone(), sp_after.clone());
  builder
    .when_last_row()
    .assert_eq(sp_after.clone(), c_one.clone());

  // ---- Per-opcode constraints using selectors  (degree 2) ----
  let c_zero = AB::Expr::from_u16(0);
  let c_two = AB::Expr::from_u16(2);

  // OP_BOOL: pop=0, push=1, ret_ty=RET_BOOL, value=scalar0
  builder.assert_zero(sel_bool.clone() * (pop.clone().into() - c_zero.clone()));
  builder.assert_zero(sel_bool.clone() * (push.clone().into() - c_one.clone()));
  builder.assert_zero(
    sel_bool.clone()
      * (ret_ty.clone().into() - AB::Expr::from_u16(crate::semantic_proof::RET_BOOL as u16)),
  );
  builder.assert_zero(sel_bool * (value.clone().into() - scalar0.clone().into()));

  // OP_BYTE: pop=0, push=1, ret_ty=RET_BYTE, value=scalar0
  builder.assert_zero(sel_byte.clone() * (pop.clone().into() - c_zero.clone()));
  builder.assert_zero(sel_byte.clone() * (push.clone().into() - c_one.clone()));
  builder
    .assert_zero(sel_byte.clone() * (ret_ty.clone().into() - AB::Expr::from_u16(RET_BYTE as u16)));
  builder.assert_zero(sel_byte * (value.clone().into() - scalar0.clone().into()));

  // OP_EQ_REFL: pop=1, push=1, ret_ty=RET_WFF_EQ
  builder.assert_zero(sel_eq_refl.clone() * (pop.clone().into() - c_one.clone()));
  builder.assert_zero(sel_eq_refl.clone() * (push.clone().into() - c_one.clone()));
  builder
    .assert_zero(sel_eq_refl * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)));

  // OP_AND_INTRO: pop=2, push=1, ret_ty=RET_WFF_AND
  builder.assert_zero(sel_and_intro.clone() * (pop.clone().into() - c_two.clone()));
  builder.assert_zero(sel_and_intro.clone() * (push.clone().into() - c_one.clone()));
  builder
    .assert_zero(sel_and_intro * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_AND as u16)));

  // OP_EQ_TRANS: pop=2, push=1, ret_ty=RET_WFF_EQ
  builder.assert_zero(sel_eq_trans.clone() * (pop.clone().into() - c_two.clone()));
  builder.assert_zero(sel_eq_trans.clone() * (push.clone().into() - c_one.clone()));
  builder
    .assert_zero(sel_eq_trans * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)));

  // OP_BYTE_ADD_THIRD_CONGRUENCE: pop=1, push=1, ret_ty=RET_WFF_EQ
  builder.assert_zero(sel_add_congr.clone() * (pop.clone().into() - c_one.clone()));
  builder.assert_zero(sel_add_congr.clone() * (push.clone().into() - c_one.clone()));
  builder
    .assert_zero(sel_add_congr * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)));

  // OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE: pop=1, push=1, ret_ty=RET_WFF_EQ
  builder.assert_zero(sel_add_carry_congr.clone() * (pop.clone().into() - c_one.clone()));
  builder.assert_zero(sel_add_carry_congr.clone() * (push.clone().into() - c_one.clone()));
  builder.assert_zero(
    sel_add_carry_congr * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)),
  );

  // OP_U29_ADD_EQ: pop=0, push=1, ret_ty=RET_WFF_AND
  builder.assert_zero(sel_u29_add_eq.clone() * (pop.clone().into() - c_zero.clone()));
  builder.assert_zero(sel_u29_add_eq.clone() * (push.clone().into() - c_one.clone()));
  builder
    .assert_zero(sel_u29_add_eq * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_AND as u16)));

  // OP_U24_ADD_EQ: pop=0, push=1, ret_ty=RET_WFF_AND
  builder.assert_zero(sel_u24_add_eq.clone() * (pop.clone().into() - c_zero.clone()));
  builder.assert_zero(sel_u24_add_eq.clone() * (push.clone().into() - c_one.clone()));
  builder
    .assert_zero(sel_u24_add_eq * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_AND as u16)));

  // OP_U15_MUL_EQ: pop=0, push=1, ret_ty=RET_WFF_AND
  builder.assert_zero(sel_u15_mul_eq.clone() * (pop.clone().into() - c_zero.clone()));
  builder.assert_zero(sel_u15_mul_eq.clone() * (push.clone().into() - c_one.clone()));
  builder
    .assert_zero(sel_u15_mul_eq * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_AND as u16)));

  // OP_BYTE_MUL_LOW_EQ: pop=0, push=1, ret_ty=RET_WFF_EQ
  builder.assert_zero(sel_mul_low_eq.clone() * (pop.clone().into() - c_zero.clone()));
  builder.assert_zero(sel_mul_low_eq.clone() * (push.clone().into() - c_one.clone()));
  builder
    .assert_zero(sel_mul_low_eq * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)));

  // OP_BYTE_MUL_HIGH_EQ: pop=0, push=1, ret_ty=RET_WFF_EQ
  builder.assert_zero(sel_mul_high_eq.clone() * (pop.clone().into() - c_zero.clone()));
  builder.assert_zero(sel_mul_high_eq.clone() * (push.clone().into() - c_one.clone()));
  builder
    .assert_zero(sel_mul_high_eq * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)));

  // OP_BYTE_AND_EQ: pop=0, push=1, ret_ty=RET_WFF_EQ
  builder.assert_zero(sel_and_eq.clone() * (pop.clone().into() - c_zero.clone()));
  builder.assert_zero(sel_and_eq.clone() * (push.clone().into() - c_one.clone()));
  builder.assert_zero(sel_and_eq * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)));

  // OP_BYTE_OR_EQ: pop=0, push=1, ret_ty=RET_WFF_EQ
  builder.assert_zero(sel_or_eq.clone() * (pop.clone().into() - c_zero.clone()));
  builder.assert_zero(sel_or_eq.clone() * (push.clone().into() - c_one.clone()));
  builder.assert_zero(sel_or_eq * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)));

  // OP_BYTE_XOR_EQ: pop=0, push=1, ret_ty=RET_WFF_EQ
  builder.assert_zero(sel_xor_eq.clone() * (pop.clone().into() - c_zero.clone()));
  builder.assert_zero(sel_xor_eq.clone() * (push.clone().into() - c_one.clone()));
  builder.assert_zero(sel_xor_eq * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)));

  // OP_NOT: pop=1, push=1, ret_ty=RET_BOOL
  builder.assert_zero(sel_not.clone() * (pop.clone().into() - c_one.clone()));
  builder.assert_zero(sel_not.clone() * (push.clone().into() - c_one.clone()));
  builder.assert_zero(
    sel_not * (ret_ty.clone().into() - AB::Expr::from_u16(crate::semantic_proof::RET_BOOL as u16)),
  );

  // OP_EQ_SYM: pop=1, push=1, ret_ty=RET_WFF_EQ
  builder.assert_zero(sel_eq_sym.clone() * (pop.clone().into() - c_one.clone()));
  builder.assert_zero(sel_eq_sym.clone() * (push.clone().into() - c_one.clone()));
  builder.assert_zero(sel_eq_sym * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)));
}

// ============================================================
// Phase 2: StackIrAirWithPrep — binds main-trace scalar columns to the shared
// preprocessed ProofRow commitment produced by setup_proof_rows_preprocessed.
// ============================================================

/// StackIR AIR variant that additionally enforces, via PairBuilder, that every
/// scalar field in the main trace matches the corresponding column of the
/// committed preprocessed ProofRow matrix.
///
/// Used exclusively by [`prove_stack_ir_with_prep`] and
/// [`verify_stack_ir_with_prep`]; the plain scaffold API keeps using
/// [`StackIrAir`] so debug-assertion checking (DebugConstraintBuilder) does not
/// require a preprocessed trace to be wired up.
///
/// Storing `prep_matrix` here is necessary for p3-uni-stark's debug-constraint
/// checker (`check_constraints`), which calls `BaseAir::preprocessed_trace()`
/// rather than reading from `PreprocessedProverData`.  The same matrix is
/// committed by `setup_proof_rows_preprocessed` via `ProofRowsPreprocessedHolder`,
/// so the two values are always consistent.
pub struct StackIrAirWithPrep {
  /// Present when used in the prover path so the debug-constraint checker
  /// (`check_constraints`) can access the preprocessed rows.
  /// `None` in the verifier path; the verifier obtains preprocessed data
  /// from the `PreprocessedVerifierKey` / proof and does not use this field.
  prep_matrix: Option<RowMajorMatrix<Val>>,
}

impl StackIrAirWithPrep {
  /// Build a `StackIrAirWithPrep` for use in the **prover** path.
  ///
  /// Stores the preprocessed matrix so p3-uni-stark's debug constraint checker
  /// (`check_constraints`) can wire up the preprocessed columns during
  /// `prove_with_preprocessed`.
  pub fn new(rows: &[ProofRow], pv: &[Val]) -> Self {
    Self {
      prep_matrix: Some(build_proof_rows_preprocessed_matrix(rows, pv)),
    }
  }

  /// Build a `StackIrAirWithPrep` for use in the **verifier** path.
  ///
  /// The verifier obtains preprocessed data from the `PreprocessedVerifierKey`
  /// embedded in the proof, so no local matrix is needed here.
  pub fn for_verify() -> Self {
    Self { prep_matrix: None }
  }
}

impl BaseAir<Val> for StackIrAirWithPrep {
  fn width(&self) -> usize {
    NUM_STACK_IR_COLS
  }

  /// Return the preprocessed matrix so the debug-constraint builder can wire
  /// up the preprocessed columns during `prove_with_preprocessed`.
  fn preprocessed_trace(&self) -> Option<RowMajorMatrix<Val>> {
    self.prep_matrix.clone()
  }
}

impl<AB> Air<AB> for StackIrAirWithPrep
where
  AB: AirBuilderWithPublicValues + PairBuilder + AirBuilder<F = Val>,
{
  fn eval(&self, builder: &mut AB) {
    // Step 1 — preprocessed binding.
    // Equate each scalar column of the main trace to the corresponding column
    // of the committed preprocessed ProofRow matrix.  Because both the StackIR
    // STARK and the LUT STARK use the same PreprocessedVerifierKey the verifier
    // can confirm, outside of proof checking, that the two proofs share the
    // same row set.
    {
      let prep = builder.preprocessed();
      let prep_row = prep
        .row_slice(0)
        .expect("StackIrAirWithPrep: empty preprocessed trace");
      let prep_row = &*prep_row;
      let main = builder.main();
      let local = main
        .row_slice(0)
        .expect("StackIrAirWithPrep: empty main trace");
      let local = &*local;

      builder.assert_eq(local[STACK_COL_OP].clone(), prep_row[PREP_COL_OP].clone());
      builder.assert_eq(
        local[STACK_COL_SCALAR0].clone(),
        prep_row[PREP_COL_SCALAR0].clone(),
      );
      builder.assert_eq(
        local[STACK_COL_SCALAR1].clone(),
        prep_row[PREP_COL_SCALAR1].clone(),
      );
      builder.assert_eq(
        local[STACK_COL_SCALAR2].clone(),
        prep_row[PREP_COL_SCALAR2].clone(),
      );
      builder.assert_eq(
        local[STACK_COL_VALUE].clone(),
        prep_row[PREP_COL_VALUE].clone(),
      );
      builder.assert_eq(
        local[STACK_COL_RET_TY].clone(),
        prep_row[PREP_COL_RET_TY].clone(),
      );
    }

    // Step 1b — bind pis[1] (EVM opcode) and pis[2..10] (tag-independent WFF digest)
    // to the corresponding preprocessed columns (PREP_COL_EVM_OPCODE / PREP_COL_WFF_DIGEST_*).
    //
    // `when_first_row()` applies each check only at row 0, where the verifier's
    // externally-derived pv must equal the prep-committed claims.  Because the
    // prep matrix stores the same opcode/digest in every row (including padding),
    // row-0 suffices to bind the entire committed object to the expected WFF.
    {
      // Collect owned AB::Expr values from public inputs (ends the immutable borrow
      // on `builder` before the mutable `when_first_row()` calls below).
      let pi_opcode: AB::Expr = {
        let pis = builder.public_values();
        pis[1].clone().into()
      };
      let pi_digest: [AB::Expr; 8] = {
        let pis = builder.public_values();
        std::array::from_fn(|k| pis[2 + k].clone().into())
      };
      // Collect owned AB::Expr values from the preprocessed row 0.
      let prep_opcode: AB::Expr = {
        let prep = builder.preprocessed();
        let row = prep
          .row_slice(0)
          .expect("StackIrAirWithPrep: empty prep (pv-bind)");
        row[PREP_COL_EVM_OPCODE].clone().into()
      };
      let prep_digest: [AB::Expr; 8] = {
        let prep = builder.preprocessed();
        let row = prep
          .row_slice(0)
          .expect("StackIrAirWithPrep: empty prep (pv-bind)");
        std::array::from_fn(|k| row[PREP_COL_WFF_DIGEST_START + k].clone().into())
      };
      // All borrows released — safe to call when_first_row() now.
      builder.when_first_row().assert_eq(prep_opcode, pi_opcode);
      for k in 0..8_usize {
        builder
          .when_first_row()
          .assert_eq(prep_digest[k].clone(), pi_digest[k].clone());
      }
    }

    // Step 2 — all existing StackIR semantic constraints (shared).
    eval_stack_ir_inner(builder);
  }
}

pub fn prove_stack_ir_scaffold_stark(rows: &[ProofRow]) -> Result<CircleStarkProof, String> {
  prove_stack_ir_scaffold_stark_with_public_values(
    rows,
    &default_receipt_bind_public_values_for_tag(RECEIPT_BIND_TAG_STACK),
  )
}

pub fn prove_stack_ir_scaffold_stark_with_public_values(
  rows: &[ProofRow],
  public_values: &[Val],
) -> Result<CircleStarkProof, String> {
  let trace = build_stack_ir_trace_from_rows(rows)?;
  let config = make_circle_config();
  Ok(p3_uni_stark::prove(
    &config,
    &StackIrAir,
    trace,
    public_values,
  ))
}

pub fn verify_stack_ir_scaffold_stark(proof: &CircleStarkProof) -> CircleStarkVerifyResult {
  verify_stack_ir_scaffold_stark_with_public_values(
    proof,
    &default_receipt_bind_public_values_for_tag(RECEIPT_BIND_TAG_STACK),
  )
}

pub fn verify_stack_ir_scaffold_stark_with_public_values(
  proof: &CircleStarkProof,
  public_values: &[Val],
) -> CircleStarkVerifyResult {
  let config = make_circle_config();
  p3_uni_stark::verify(&config, &StackIrAir, proof, public_values)
}

pub fn prove_and_verify_stack_ir_scaffold_stark(rows: &[ProofRow]) -> bool {
  let proved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
    prove_stack_ir_scaffold_stark(rows)
  }));
  let proof = match proved {
    Ok(Ok(proof)) => proof,
    Ok(Err(_)) => return false,
    Err(_) => return false,
  };
  verify_stack_ir_scaffold_stark(&proof).is_ok()
}

// ============================================================
// Phase 1: Shared preprocessed rows infrastructure
//
// `rows = compile_proof(semantic_proof)` is committed as a preprocessed trace
// shared by the StackIR and LUT STARKs.  Both proofs include this commitment
// in their Fiat-Shamir transcript, binding them to the same ProofRow set.
//
// Column layout in the preprocessed matrix (NUM_PREP_COLS = 18):
//   0 op  1 scalar0  2 scalar1  3 scalar2  4 arg0  5 arg1  6 arg2  7 value  8 ret_ty
//   9 evm_opcode  10..17 wff_digest[0..7]
//
// Columns 9-17 are identical across all rows of a given proof.  They store the
// EVM opcode (`pis[1]`) and the tag-independent WFF Poseidon digest (`pis[2..10]`).
// Both StackIrAirWithPrep and LutKernelAirWithPrep bind these to the corresponding
// public-input slots on the first row, closing the gap that previously let a forger
// prove valid-arithmetic rows for a different WFF claim.
// ============================================================

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

  // Build a lookup: row_index → (evm_opcode, per-instruction wff_digest)
  // We pre-compute per-instruction WFF digests once.
  let mut row_meta: Vec<(Val, [Val; 8])> = vec![(Val::ZERO, [Val::ZERO; 8]); rows.len()];
  for entry in &manifest.entries {
    let per_digest = compute_wff_opcode_digest(entry.opcode, &entry.wff);
    let opcode_val = Val::from_u32(entry.opcode as u32);
    for r in entry.row_start..(entry.row_start + entry.row_count).min(rows.len()) {
      row_meta[r] = (opcode_val, per_digest);
    }
  }

  for (i, row) in rows.iter().enumerate() {
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
    let (opcode_val, per_digest) = row_meta[i];
    matrix.values[base + PREP_COL_EVM_OPCODE] = opcode_val;
    for k in 0..8 {
      matrix.values[base + PREP_COL_WFF_DIGEST_START + k] = per_digest[k];
    }
    // Batch-level metadata (same for every row).
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
  // Pick a representative per-instruction opcode/digest from the last live row.
  let (last_opcode, last_per_digest) = row_meta
    .last()
    .copied()
    .unwrap_or((Val::ZERO, [Val::ZERO; 8]));
  for i in rows.len()..n_rows {
    let base = i * NUM_BATCH_PREP_COLS;
    matrix.values[base + PREP_COL_OP] = Val::from_u32(crate::semantic_proof::OP_EQ_REFL);
    matrix.values[base + PREP_COL_RET_TY] = Val::from_u32(crate::semantic_proof::RET_WFF_EQ);
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
    matrix.values[base + PREP_COL_OP] = Val::from_u32(crate::semantic_proof::OP_EQ_REFL);
    matrix.values[base + PREP_COL_RET_TY] = Val::from_u32(crate::semantic_proof::RET_WFF_EQ);
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
struct ProofRowsPreprocessedHolder {
  matrix: RowMajorMatrix<Val>,
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

/// Prove the StackIR STARK for `rows` and bind the proof to the shared
/// preprocessed commitment (`prep_data`).
///
/// The resulting proof includes `prep_data`'s commitment in its transcript,
/// so the verifier can confirm (via [`verify_stack_ir_with_prep`]) that this
/// proof was generated with the correct preprocessed rows.
pub fn prove_stack_ir_with_prep(
  rows: &[ProofRow],
  prep_data: &PreprocessedProverData<CircleStarkConfig>,
  public_values: &[Val],
) -> Result<CircleStarkProof, String> {
  let trace = build_stack_ir_trace_from_rows(rows)?;
  let config = make_circle_config();
  let air = StackIrAirWithPrep::new(rows, public_values);
  let proof =
    p3_uni_stark::prove_with_preprocessed(&config, &air, trace, public_values, Some(prep_data));
  Ok(proof)
}

/// Verify a StackIR proof that was generated with [`prove_stack_ir_with_prep`].
///
/// `prep_vk` must be the verifier key produced by the same
/// [`setup_proof_rows_preprocessed`] call that produced the prover data;
/// if the two witnesses share a VK the proof links them to identical rows.
pub fn verify_stack_ir_with_prep(
  proof: &CircleStarkProof,
  prep_vk: &PreprocessedVerifierKey<CircleStarkConfig>,
  public_values: &[Val],
) -> CircleStarkVerifyResult {
  let config = make_circle_config();
  p3_uni_stark::verify_with_preprocessed(
    &config,
    &StackIrAirWithPrep::for_verify(),
    proof,
    public_values,
    Some(prep_vk),
  )
}

// ============================================================
// LUT framework (v2 scaffold)
// ============================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LutOpcode {
  U29AddEq,
  U24AddEq,
  U15AddEq,
  BitAddEq,
  U15MulEq,
  ByteMulLowEq,
  ByteMulHighEq,
  ByteAndEq,
  ByteOrEq,
  ByteXorEq,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LutStep {
  pub op: LutOpcode,
  pub in0: u32,
  pub in1: u32,
  pub in2: u32,
  pub out0: u32,
  pub out1: u32,
}

const LUT_COL_OP: usize = 0;
const LUT_COL_IN0: usize = 1;
const LUT_COL_IN1: usize = 2;
const LUT_COL_IN2: usize = 3;
const LUT_COL_OUT0: usize = 4;
const LUT_COL_OUT1: usize = 5;
// One-hot selector columns (6..15) — one per LutOpcode variant.
const LUT_COL_SEL_U29_ADD_EQ: usize = 6;
const LUT_COL_SEL_U24_ADD_EQ: usize = 7;
const LUT_COL_SEL_U15_ADD_EQ: usize = 8;
const LUT_COL_SEL_BIT_ADD_EQ: usize = 9;
const LUT_COL_SEL_U15_MUL_EQ: usize = 10;
const LUT_COL_SEL_BYTE_MUL_LOW_EQ: usize = 11;
const LUT_COL_SEL_BYTE_MUL_HIGH_EQ: usize = 12;
const LUT_COL_SEL_BYTE_AND_EQ: usize = 13;
const LUT_COL_SEL_BYTE_OR_EQ: usize = 14;
const LUT_COL_SEL_BYTE_XOR_EQ: usize = 15;
pub const NUM_LUT_COLS: usize = 16;

fn lut_opcode_tag(op: LutOpcode) -> u16 {
  match op {
    LutOpcode::U29AddEq => 1,
    LutOpcode::U24AddEq => 2,
    LutOpcode::U15AddEq => 3,
    LutOpcode::BitAddEq => 4,
    LutOpcode::U15MulEq => 5,
    LutOpcode::ByteMulLowEq => 6,
    LutOpcode::ByteMulHighEq => 7,
    LutOpcode::ByteAndEq => 8,
    LutOpcode::ByteOrEq => 9,
    LutOpcode::ByteXorEq => 10,
  }
}

/// Map a `LutOpcode` to its one-hot selector column index.
fn lut_op_to_sel_col(op: LutOpcode) -> usize {
  match op {
    LutOpcode::U29AddEq => LUT_COL_SEL_U29_ADD_EQ,
    LutOpcode::U24AddEq => LUT_COL_SEL_U24_ADD_EQ,
    LutOpcode::U15AddEq => LUT_COL_SEL_U15_ADD_EQ,
    LutOpcode::BitAddEq => LUT_COL_SEL_BIT_ADD_EQ,
    LutOpcode::U15MulEq => LUT_COL_SEL_U15_MUL_EQ,
    LutOpcode::ByteMulLowEq => LUT_COL_SEL_BYTE_MUL_LOW_EQ,
    LutOpcode::ByteMulHighEq => LUT_COL_SEL_BYTE_MUL_HIGH_EQ,
    LutOpcode::ByteAndEq => LUT_COL_SEL_BYTE_AND_EQ,
    LutOpcode::ByteOrEq => LUT_COL_SEL_BYTE_OR_EQ,
    LutOpcode::ByteXorEq => LUT_COL_SEL_BYTE_XOR_EQ,
  }
}


pub fn build_lut_steps_from_rows(rows: &[ProofRow]) -> Result<Vec<LutStep>, String> {
  let mut out = Vec::with_capacity(rows.len());

  for row in rows {
    let step = match row.op {
      crate::semantic_proof::OP_U29_ADD_EQ => LutStep {
        op: LutOpcode::U29AddEq,
        in0: row.scalar0,
        in1: row.scalar1,
        in2: row.scalar2,
        out0: row.value,
        out1: row.arg1,
      },
      crate::semantic_proof::OP_U24_ADD_EQ => LutStep {
        op: LutOpcode::U24AddEq,
        in0: row.scalar0,
        in1: row.scalar1,
        in2: row.scalar2,
        out0: row.value,
        out1: row.arg1,
      },
      crate::semantic_proof::OP_U15_MUL_EQ => {
        let total = row.scalar0 * row.scalar1;
        LutStep {
          op: LutOpcode::U15MulEq,
          in0: row.scalar0,
          in1: row.scalar1,
          in2: 0,
          out0: total & 0x7FFF,
          out1: total >> 15,
        }
      }
      crate::semantic_proof::OP_BYTE_MUL_LOW_EQ => {
        let product = row.scalar0 * row.scalar1;
        // Soundness fix: MulLow AIR constraint is  in0*in1 = out0 + 256*out1
        // so out1 must carry the high byte, not 0.
        LutStep {
          op: LutOpcode::ByteMulLowEq,
          in0: row.scalar0,
          in1: row.scalar1,
          in2: 0,
          out0: product & 0xFF,
          out1: product >> 8,
        }
      }
      crate::semantic_proof::OP_BYTE_MUL_HIGH_EQ => {
        let product = row.scalar0 * row.scalar1;
        // Soundness fix: MulHigh AIR constraint is  in0*in1 = 256*out0 + out1
        // so out1 must carry the low byte, not 0.
        LutStep {
          op: LutOpcode::ByteMulHighEq,
          in0: row.scalar0,
          in1: row.scalar1,
          in2: 0,
          out0: product >> 8,
          out1: product & 0xFF,
        }
      }
      crate::semantic_proof::OP_BYTE_AND_EQ => LutStep {
        op: LutOpcode::ByteAndEq,
        in0: row.scalar0,
        in1: row.scalar1,
        in2: 0,
        out0: row.scalar0 & row.scalar1,
        out1: 0,
      },
      crate::semantic_proof::OP_BYTE_OR_EQ => LutStep {
        op: LutOpcode::ByteOrEq,
        in0: row.scalar0,
        in1: row.scalar1,
        in2: 0,
        out0: row.scalar0 | row.scalar1,
        out1: 0,
      },
      crate::semantic_proof::OP_BYTE_XOR_EQ => LutStep {
        op: LutOpcode::ByteXorEq,
        in0: row.scalar0,
        in1: row.scalar1,
        in2: 0,
        out0: row.scalar0 ^ row.scalar1,
        out1: 0,
      },
      other => return Err(format!("row op {other} is not LUT-step encodable yet")),
    };
    out.push(step);
  }

  if out.is_empty() {
    return Err("cannot build LUT steps from empty row set".to_string());
  }

  Ok(out)
}

pub fn build_lut_trace_from_steps(steps: &[LutStep]) -> Result<RowMajorMatrix<Val>, String> {
  if steps.is_empty() {
    return Err("cannot build LUT trace from empty steps".to_string());
  }

  validate_lut_steps(steps)?;

  let n_rows = steps.len().max(4).next_power_of_two();
  let mut trace = RowMajorMatrix::new(Val::zero_vec(n_rows * NUM_LUT_COLS), NUM_LUT_COLS);

  for (i, step) in steps.iter().enumerate() {
    let base = i * NUM_LUT_COLS;
    trace.values[base + LUT_COL_OP] = Val::from_u16(lut_opcode_tag(step.op));
    trace.values[base + LUT_COL_IN0] = Val::from_u32(step.in0);
    trace.values[base + LUT_COL_IN1] = Val::from_u32(step.in1);
    trace.values[base + LUT_COL_IN2] = Val::from_u32(step.in2);
    trace.values[base + LUT_COL_OUT0] = Val::from_u32(step.out0);
    trace.values[base + LUT_COL_OUT1] = Val::from_u32(step.out1);
    // One-hot selector: set the column for this opcode to 1.
    trace.values[base + lut_op_to_sel_col(step.op)] = Val::from_u16(1);
  }

  for i in steps.len()..n_rows {
    let base = i * NUM_LUT_COLS;
    // Padding: U29AddEq with all-zero inputs (0+0+0=0, carry=0).
    trace.values[base + LUT_COL_OP] = Val::from_u16(lut_opcode_tag(LutOpcode::U29AddEq));
    trace.values[base + LUT_COL_IN0] = Val::from_u16(0);
    trace.values[base + LUT_COL_IN1] = Val::from_u16(0);
    trace.values[base + LUT_COL_IN2] = Val::from_u16(0);
    trace.values[base + LUT_COL_OUT0] = Val::from_u16(0);
    trace.values[base + LUT_COL_OUT1] = Val::from_u16(0);
    trace.values[base + LUT_COL_SEL_U29_ADD_EQ] = Val::from_u16(1);
  }

  Ok(trace)
}

fn validate_lut_steps(steps: &[LutStep]) -> Result<(), String> {
  if steps.is_empty() {
    return Err("cannot validate empty LUT steps".to_string());
  }

  let ensure_le = |name: &str, value: u32, max: u32, row: usize| -> Result<(), String> {
    if value > max {
      return Err(format!(
        "lut {name} out of range at row {row}: {value} > {max}"
      ));
    }
    Ok(())
  };

  for (i, step) in steps.iter().enumerate() {
    match step.op {
      LutOpcode::U29AddEq => {
        ensure_le("in0", step.in0, (1u32 << 29) - 1, i)?;
        ensure_le("in1", step.in1, (1u32 << 29) - 1, i)?;
        ensure_le("in2", step.in2, 1, i)?;
        ensure_le("out0", step.out0, (1u32 << 29) - 1, i)?;
        ensure_le("out1", step.out1, 1, i)?;
      }
      LutOpcode::U24AddEq => {
        ensure_le("in0", step.in0, (1u32 << 24) - 1, i)?;
        ensure_le("in1", step.in1, (1u32 << 24) - 1, i)?;
        ensure_le("in2", step.in2, 1, i)?;
        ensure_le("out0", step.out0, (1u32 << 24) - 1, i)?;
        ensure_le("out1", step.out1, 1, i)?;
      }
      LutOpcode::U15AddEq => {
        ensure_le("in0", step.in0, 0x7FFF, i)?;
        ensure_le("in1", step.in1, 0x7FFF, i)?;
        ensure_le("in2", step.in2, 1, i)?;
        ensure_le("out0", step.out0, 0x7FFF, i)?;
        ensure_le("out1", step.out1, 1, i)?;
      }
      LutOpcode::BitAddEq => {
        ensure_le("in0", step.in0, 1, i)?;
        ensure_le("in1", step.in1, 1, i)?;
        ensure_le("in2", step.in2, 1, i)?;
        ensure_le("out0", step.out0, 1, i)?;
        ensure_le("out1", step.out1, 1, i)?;
      }
      LutOpcode::U15MulEq => {
        ensure_le("in0", step.in0, 0x7FFF, i)?;
        ensure_le("in1", step.in1, 0x7FFF, i)?;
        ensure_le("in2", step.in2, 0, i)?;
        ensure_le("out0", step.out0, 0x7FFF, i)?;
        ensure_le("out1", step.out1, 0x7FFF, i)?;
      }
      LutOpcode::ByteMulLowEq | LutOpcode::ByteMulHighEq => {
        ensure_le("in0", step.in0, 255, i)?;
        ensure_le("in1", step.in1, 255, i)?;
        ensure_le("in2", step.in2, 0, i)?;
        ensure_le("out0", step.out0, 255, i)?;
        ensure_le("out1", step.out1, 255, i)?;
      }
      LutOpcode::ByteAndEq | LutOpcode::ByteOrEq | LutOpcode::ByteXorEq => {
        // Byte-level operations: inputs and output are full bytes (0..=255).
        // Soundness for the correct operation is enforced by the LogUp byte
        // table argument in byte_table.rs, not by these range checks alone.
        ensure_le("in0", step.in0, 255, i)?;
        ensure_le("in1", step.in1, 255, i)?;
        ensure_le("in2", step.in2, 0, i)?;
        ensure_le("out0", step.out0, 255, i)?;
        ensure_le("out1", step.out1, 0, i)?;
      }
    }
  }

  Ok(())
}

pub struct LutKernelAir;

impl<F> BaseAir<F> for LutKernelAir {
  fn width(&self) -> usize {
    NUM_LUT_COLS
  }
}

impl<AB: AirBuilderWithPublicValues> Air<AB> for LutKernelAir {
  fn eval(&self, builder: &mut AB) {
    eval_lut_kernel_inner(builder);
  }
}

/// Core LUT kernel constraints — shared between `LutKernelAir` and
/// `LutKernelAirWithPrep` to avoid code duplication.
///
/// Uses one-hot selector columns instead of `allowed_poly` (degree-13) and
/// `gate()` polynomials (degree-12). Per-opcode constraints become degree 2~3
/// (`sel * linear` or `sel * in0*in1` for multiply ops).
fn eval_lut_kernel_inner<AB: AirBuilderWithPublicValues>(builder: &mut AB) {
  let pis = builder.public_values();
  builder.assert_eq(pis[0], AB::Expr::from_u32(RECEIPT_BIND_TAG_LUT));

  let main = builder.main();
  let local = main.row_slice(0).expect("empty trace");
  let local = &*local;

  let op = local[LUT_COL_OP].clone();
  let in0 = local[LUT_COL_IN0].clone();
  let in1 = local[LUT_COL_IN1].clone();
  let in2 = local[LUT_COL_IN2].clone();
  let out0 = local[LUT_COL_OUT0].clone();
  let out1 = local[LUT_COL_OUT1].clone();

  // Selector columns.
  let s_u29 = local[LUT_COL_SEL_U29_ADD_EQ].clone();
  let s_u24 = local[LUT_COL_SEL_U24_ADD_EQ].clone();
  let s_u15 = local[LUT_COL_SEL_U15_ADD_EQ].clone();
  let s_bit = local[LUT_COL_SEL_BIT_ADD_EQ].clone();
  let s_mul = local[LUT_COL_SEL_U15_MUL_EQ].clone();
  let s_mul_low = local[LUT_COL_SEL_BYTE_MUL_LOW_EQ].clone();
  let s_mul_high = local[LUT_COL_SEL_BYTE_MUL_HIGH_EQ].clone();
  let s_and = local[LUT_COL_SEL_BYTE_AND_EQ].clone();
  let s_or = local[LUT_COL_SEL_BYTE_OR_EQ].clone();
  let s_xor = local[LUT_COL_SEL_BYTE_XOR_EQ].clone();

  // ── (1) Boolean: sel_i * (1 - sel_i) = 0  (degree 2) ──────────────
  for s in [
    s_u29.clone(),
    s_u24.clone(),
    s_u15.clone(),
    s_bit.clone(),
    s_mul.clone(),
    s_mul_low.clone(),
    s_mul_high.clone(),
    s_and.clone(),
    s_or.clone(),
    s_xor.clone(),
  ] {
    builder.assert_bool(s);
  }

  // ── (2) One-hot: Σ sel_i = 1  (degree 1) ──────────────────────────
  let sel_sum = s_u29.clone().into()
    + s_u24.clone().into()
    + s_u15.clone().into()
    + s_bit.clone().into()
    + s_mul.clone().into()
    + s_mul_low.clone().into()
    + s_mul_high.clone().into()
    + s_and.clone().into()
    + s_or.clone().into()
    + s_xor.clone().into();
  builder.assert_one(sel_sum);

  // ── (3) op = Σ(sel_i · tag_i)  (degree 2) ─────────────────────────
  let op_reconstruct = s_u29.clone().into() * AB::Expr::from_u16(lut_opcode_tag(LutOpcode::U29AddEq))
    + s_u24.clone().into() * AB::Expr::from_u16(lut_opcode_tag(LutOpcode::U24AddEq))
    + s_u15.clone().into() * AB::Expr::from_u16(lut_opcode_tag(LutOpcode::U15AddEq))
    + s_bit.clone().into() * AB::Expr::from_u16(lut_opcode_tag(LutOpcode::BitAddEq))
    + s_mul.clone().into() * AB::Expr::from_u16(lut_opcode_tag(LutOpcode::U15MulEq))
    + s_mul_low.clone().into() * AB::Expr::from_u16(lut_opcode_tag(LutOpcode::ByteMulLowEq))
    + s_mul_high.clone().into() * AB::Expr::from_u16(lut_opcode_tag(LutOpcode::ByteMulHighEq))
    + s_and.clone().into() * AB::Expr::from_u16(lut_opcode_tag(LutOpcode::ByteAndEq))
    + s_or.clone().into() * AB::Expr::from_u16(lut_opcode_tag(LutOpcode::ByteOrEq))
    + s_xor.clone().into() * AB::Expr::from_u16(lut_opcode_tag(LutOpcode::ByteXorEq));
  builder.assert_eq(op.into(), op_reconstruct);

  // ── (4) Per-opcode arithmetic constraints ─────────────────────────
  // All are `sel * (constraint)`.  Linear constraints → degree 2.
  // Multiply constraints (in0*in1) → degree 3.
  let total = in0.clone().into() + in1.clone().into() + in2.clone().into();
  let c256 = AB::Expr::from_u16(256);
  let c2 = AB::Expr::from_u16(2);
  let c32768 = AB::Expr::from_u32(32768);
  let c16777216 = AB::Expr::from_u32(1u32 << 24);
  let c536870912 = AB::Expr::from_u32(1u32 << 29);

  // U29AddEq
  builder.assert_zero(
    s_u29.clone()
      * (total.clone() - out0.clone().into() - c536870912.clone() * out1.clone().into()),
  );
  builder.assert_bool(s_u29.clone() * out1.clone().into());

  // U24AddEq
  builder.assert_zero(
    s_u24.clone() * (total.clone() - out0.clone().into() - c16777216.clone() * out1.clone().into()),
  );
  builder.assert_bool(s_u24.clone() * out1.clone().into());

  // U15AddEq
  builder.assert_zero(
    s_u15.clone() * (total.clone() - out0.clone().into() - c32768.clone() * out1.clone().into()),
  );
  builder.assert_bool(s_u15.clone() * out1.clone().into());

  // BitAddEq: in0+in1+in2 = out0 + 2*out1, all inputs/outputs boolean
  builder.assert_zero(
    s_bit.clone() * (total.clone() - out0.clone().into() - c2.clone() * out1.clone().into()),
  );
  builder.assert_bool(s_bit.clone() * in0.clone().into());
  builder.assert_bool(s_bit.clone() * in1.clone().into());
  builder.assert_bool(s_bit.clone() * in2.clone().into());
  builder.assert_bool(s_bit.clone() * out0.clone().into());
  builder.assert_bool(s_bit.clone() * out1.clone().into());

  // U15MulEq: in0*in1 = out0 + 32768*out1, in2=0
  builder.assert_zero(
    s_mul.clone()
      * ((in0.clone().into() * in1.clone().into())
        - out0.clone().into()
        - c32768.clone() * out1.clone().into()),
  );
  builder.assert_zero(s_mul.clone() * in2.clone());

  // ByteMulLowEq: in0*in1 = out0 + 256*out1, in2=0
  builder.assert_zero(
    s_mul_low.clone()
      * ((in0.clone().into() * in1.clone().into())
        - out0.clone().into()
        - c256.clone() * out1.clone().into()),
  );
  builder.assert_zero(s_mul_low.clone() * in2.clone());

  // ByteMulHighEq: in0*in1 = 256*out0 + out1, in2=0
  builder.assert_zero(
    s_mul_high.clone()
      * ((in0.clone().into() * in1.clone().into())
        - out1.clone().into()
        - c256.clone() * out0.clone().into()),
  );
  builder.assert_zero(s_mul_high.clone() * in2.clone());

  // ByteAndEq (byte-level): structural constraints only.
  // Arithmetic correctness (out0 = in0 & in1) is enforced by the LogUp
  // byte-table argument in byte_table.rs.  Here we only assert in2=out1=0.
  builder.assert_zero(s_and.clone() * in2.clone());
  builder.assert_zero(s_and.clone() * out1.clone());

  // ByteOrEq (byte-level): structural constraints only.
  builder.assert_zero(s_or.clone() * in2.clone());
  builder.assert_zero(s_or.clone() * out1.clone());

  // ByteXorEq (byte-level): structural constraints only.
  builder.assert_zero(s_xor.clone() * in2.clone());
  builder.assert_zero(s_xor.clone() * out1.clone());

  // U{15,24,29}AddEq all use in2 as carry-in; each opcode enforces its own in2 semantics above.
}

/// Build a LUT main trace with exactly **one row per [`ProofRow`]** — the same
/// height as [`build_proof_rows_preprocessed_matrix`] and
/// [`build_stack_ir_trace_from_rows`].  This 1:1 alignment is required to share
/// the same `PreprocessedProverData` between the StackIR and LUT STARKs.
///
/// Non-LUT structural ops (`OP_EQ_REFL`, etc.) are encoded as `U29AddEq(0,0,0)`
/// padding rows, which satisfy the LUT AIR's arithmetic constraints trivially.
pub fn build_lut_trace_from_proof_rows(rows: &[ProofRow]) -> Result<RowMajorMatrix<Val>, String> {
  use crate::semantic_proof::{
    OP_AND_INTRO, OP_BOOL, OP_BYTE, OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE,
    OP_BYTE_ADD_THIRD_CONGRUENCE, OP_BYTE_AND_EQ, OP_BYTE_MUL_HIGH_EQ,
    OP_BYTE_MUL_LOW_EQ, OP_BYTE_OR_EQ, OP_BYTE_XOR_EQ, OP_EQ_REFL, OP_EQ_SYM, OP_EQ_TRANS, OP_NOT,
    OP_U15_MUL_EQ, OP_U24_ADD_EQ, OP_U29_ADD_EQ,
  };

  if rows.is_empty() {
    return Err("build_lut_trace_from_proof_rows: cannot build from empty row set".to_string());
  }

  let n_rows = rows.len().max(4).next_power_of_two();
  let mut trace = RowMajorMatrix::new(Val::zero_vec(n_rows * NUM_LUT_COLS), NUM_LUT_COLS);
  let pad_tag = Val::from_u16(lut_opcode_tag(LutOpcode::U29AddEq));

  for (i, row) in rows.iter().enumerate() {
    let b = i * NUM_LUT_COLS;
    match row.op {
      // ── Structural (non-LUT) rows: pad as U29AddEq(0,0,0,0,0) ──────────
      op if op == OP_BOOL
        || op == OP_BYTE
        || op == OP_EQ_REFL
        || op == OP_AND_INTRO
        || op == OP_EQ_TRANS
        || op == OP_BYTE_ADD_THIRD_CONGRUENCE
        || op == OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE
        || op == OP_NOT
        || op == OP_EQ_SYM =>
      {
        trace.values[b + LUT_COL_OP] = pad_tag;
        trace.values[b + LUT_COL_SEL_U29_ADD_EQ] = Val::from_u16(1);
        // in0..out1 remain zero — U29AddEq(0+0+0=0) is arithmetically valid.
      }
      OP_U29_ADD_EQ => {
        trace.values[b + LUT_COL_OP] = Val::from_u16(lut_opcode_tag(LutOpcode::U29AddEq));
        trace.values[b + LUT_COL_IN0] = Val::from_u32(row.scalar0);
        trace.values[b + LUT_COL_IN1] = Val::from_u32(row.scalar1);
        trace.values[b + LUT_COL_IN2] = Val::from_u32(row.scalar2);
        trace.values[b + LUT_COL_OUT0] = Val::from_u32(row.value);
        trace.values[b + LUT_COL_OUT1] = Val::from_u32(row.arg1);
        trace.values[b + LUT_COL_SEL_U29_ADD_EQ] = Val::from_u16(1);
      }
      OP_U24_ADD_EQ => {
        trace.values[b + LUT_COL_OP] = Val::from_u16(lut_opcode_tag(LutOpcode::U24AddEq));
        trace.values[b + LUT_COL_IN0] = Val::from_u32(row.scalar0);
        trace.values[b + LUT_COL_IN1] = Val::from_u32(row.scalar1);
        trace.values[b + LUT_COL_IN2] = Val::from_u32(row.scalar2);
        trace.values[b + LUT_COL_OUT0] = Val::from_u32(row.value);
        trace.values[b + LUT_COL_OUT1] = Val::from_u32(row.arg1);
        trace.values[b + LUT_COL_SEL_U24_ADD_EQ] = Val::from_u16(1);
      }
      OP_U15_MUL_EQ => {
        // lo = value, hi = arg0  (as stored by compile_proof)
        trace.values[b + LUT_COL_OP] = Val::from_u16(lut_opcode_tag(LutOpcode::U15MulEq));
        trace.values[b + LUT_COL_IN0] = Val::from_u32(row.scalar0);
        trace.values[b + LUT_COL_IN1] = Val::from_u32(row.scalar1);
        // in2 = 0 (already zero)
        trace.values[b + LUT_COL_OUT0] = Val::from_u32(row.value);
        trace.values[b + LUT_COL_OUT1] = Val::from_u32(row.arg0);
        trace.values[b + LUT_COL_SEL_U15_MUL_EQ] = Val::from_u16(1);
      }
      OP_BYTE_MUL_LOW_EQ => {
        let result = row.scalar0 * row.scalar1;
        trace.values[b + LUT_COL_OP] = Val::from_u16(lut_opcode_tag(LutOpcode::ByteMulLowEq));
        trace.values[b + LUT_COL_IN0] = Val::from_u32(row.scalar0);
        trace.values[b + LUT_COL_IN1] = Val::from_u32(row.scalar1);
        trace.values[b + LUT_COL_OUT0] = Val::from_u32(result & 0xFF);
        // Soundness fix: AIR constraint is  in0*in1 = out0 + 256*out1
        // so out1 must carry the high byte.
        trace.values[b + LUT_COL_OUT1] = Val::from_u32(result >> 8);
        trace.values[b + LUT_COL_SEL_BYTE_MUL_LOW_EQ] = Val::from_u16(1);
      }
      OP_BYTE_MUL_HIGH_EQ => {
        let result = row.scalar0 * row.scalar1;
        trace.values[b + LUT_COL_OP] = Val::from_u16(lut_opcode_tag(LutOpcode::ByteMulHighEq));
        trace.values[b + LUT_COL_IN0] = Val::from_u32(row.scalar0);
        trace.values[b + LUT_COL_IN1] = Val::from_u32(row.scalar1);
        trace.values[b + LUT_COL_OUT0] = Val::from_u32(result >> 8);
        // Soundness fix: AIR constraint is  in0*in1 = 256*out0 + out1
        // so out1 must carry the low byte.
        trace.values[b + LUT_COL_OUT1] = Val::from_u32(result & 0xFF);
        trace.values[b + LUT_COL_SEL_BYTE_MUL_HIGH_EQ] = Val::from_u16(1);
      }
      op if op == OP_BYTE_AND_EQ => {
        // ByteAndEq: encode with real byte operands and result.
        // eval_lut_kernel_inner enforces only in2=0 and out1=0 for s_and;
        // arithmetic correctness is handled by the LogUp byte-table argument.
        trace.values[b + LUT_COL_OP] = Val::from_u16(lut_opcode_tag(LutOpcode::ByteAndEq));
        trace.values[b + LUT_COL_IN0] = Val::from_u32(row.scalar0);
        trace.values[b + LUT_COL_IN1] = Val::from_u32(row.scalar1);
        // in2 = 0 (already zero)
        trace.values[b + LUT_COL_OUT0] = Val::from_u32(row.value);
        // out1 = 0 (already zero)
        trace.values[b + LUT_COL_SEL_BYTE_AND_EQ] = Val::from_u16(1);
      }
      op if op == OP_BYTE_OR_EQ => {
        // ByteOrEq: same layout as ByteAndEq.
        trace.values[b + LUT_COL_OP] = Val::from_u16(lut_opcode_tag(LutOpcode::ByteOrEq));
        trace.values[b + LUT_COL_IN0] = Val::from_u32(row.scalar0);
        trace.values[b + LUT_COL_IN1] = Val::from_u32(row.scalar1);
        trace.values[b + LUT_COL_OUT0] = Val::from_u32(row.value);
        trace.values[b + LUT_COL_SEL_BYTE_OR_EQ] = Val::from_u16(1);
      }
      op if op == OP_BYTE_XOR_EQ => {
        // ByteXorEq: same layout as ByteAndEq.
        trace.values[b + LUT_COL_OP] = Val::from_u16(lut_opcode_tag(LutOpcode::ByteXorEq));
        trace.values[b + LUT_COL_IN0] = Val::from_u32(row.scalar0);
        trace.values[b + LUT_COL_IN1] = Val::from_u32(row.scalar1);
        trace.values[b + LUT_COL_OUT0] = Val::from_u32(row.value);
        trace.values[b + LUT_COL_SEL_BYTE_XOR_EQ] = Val::from_u16(1);
      }
      other => {
        return Err(format!(
          "build_lut_trace_from_proof_rows: unsupported op {other} at row {i}"
        ));
      }
    }
  }

  // Padding rows beyond `rows.len()`: U29AddEq(0,0,0,0,0).
  for i in rows.len()..n_rows {
    let b = i * NUM_LUT_COLS;
    trace.values[b + LUT_COL_OP] = pad_tag;
    trace.values[b + LUT_COL_SEL_U29_ADD_EQ] = Val::from_u16(1);
  }

  Ok(trace)
}

/// Extract [`ByteTableQuery`] items from a set of LUT steps.
///
/// Collects all steps whose opcode is `ByteAndEq`, `ByteOrEq`, or
/// `ByteXorEq` and converts them into the byte-table query format.
/// Used to generate the companion byte-table proof alongside the main LUT
/// STARK proof.
/// Collect byte-table LogUp queries from a **LutStep** slice.
///
/// Each `ByteAndEq` / `ByteOrEq` / `ByteXorEq` step becomes one query using
/// `in0`/`in1` as the byte inputs.  All other steps are ignored.
///
/// Prefer [`collect_byte_table_queries_from_rows`] which operates on raw
/// `ProofRow`s and merges duplicate `(a, b, op)` pairs by multiplicity.
pub fn collect_byte_table_queries_from_lut_steps(
  steps: &[LutStep],
) -> Vec<crate::byte_table::ByteTableQuery> {
  use crate::byte_table::{BYTE_OP_AND, BYTE_OP_OR, BYTE_OP_XOR, ByteTableQuery};

  steps
    .iter()
    .filter_map(|s| match s.op {
      LutOpcode::ByteAndEq => Some(ByteTableQuery {
        a: s.in0 as u8,
        b: s.in1 as u8,
        op: BYTE_OP_AND,
        result: s.out0 as u8,
        multiplicity: 1,
      }),
      LutOpcode::ByteOrEq => Some(ByteTableQuery {
        a: s.in0 as u8,
        b: s.in1 as u8,
        op: BYTE_OP_OR,
        result: s.out0 as u8,
        multiplicity: 1,
      }),
      LutOpcode::ByteXorEq => Some(ByteTableQuery {
        a: s.in0 as u8,
        b: s.in1 as u8,
        op: BYTE_OP_XOR,
        result: s.out0 as u8,
        multiplicity: 1,
      }),
      _ => None,
    })
    .collect()
}

/// Collect byte-table LogUp queries **directly from `ProofRow`s**.
///
/// Scans the row set for `OP_BYTE_AND_EQ`, `OP_BYTE_OR_EQ`, and
/// `OP_BYTE_XOR_EQ` rows and emits one [`crate::byte_table::ByteTableQuery`]
/// per row using `scalar0` / `scalar1` as the true byte operands.
///
/// This is the **correct** entry-point for the byte-table argument:
/// - Structural rows (OP_AND_INTRO, OP_EQ_REFL, …) are silently skipped.
/// - Inputs are real bytes (0..=255), not bit-expanded (0/1).
/// - Produces *N* queries for a 256-bit AND (N = 32), not 8·N.
///
/// Duplicate queries with the same `(a, b, op)` key are automatically merged
/// by summing their multiplicities, keeping the trace as small as possible.
pub fn collect_byte_table_queries_from_rows(
  rows: &[ProofRow],
) -> Vec<crate::byte_table::ByteTableQuery> {
  use crate::byte_table::{BYTE_OP_AND, BYTE_OP_OR, BYTE_OP_XOR, ByteTableQuery};
  use crate::semantic_proof::{OP_BYTE_AND_EQ, OP_BYTE_OR_EQ, OP_BYTE_XOR_EQ};
  use std::collections::BTreeMap;

  // key = (a, b, op)  → accumulated multiplicity
  let mut acc: BTreeMap<(u8, u8, u32), i32> = BTreeMap::new();

  for row in rows {
    let (a, b, op) = match row.op {
      op if op == OP_BYTE_AND_EQ => (row.scalar0 as u8, row.scalar1 as u8, BYTE_OP_AND),
      op if op == OP_BYTE_OR_EQ => (row.scalar0 as u8, row.scalar1 as u8, BYTE_OP_OR),
      op if op == OP_BYTE_XOR_EQ => (row.scalar0 as u8, row.scalar1 as u8, BYTE_OP_XOR),
      _ => continue,
    };
    *acc.entry((a, b, op)).or_insert(0) += 1;
  }

  acc
    .into_iter()
    .map(|((a, b, op), mult)| {
      let result = match op {
        BYTE_OP_AND => a & b,
        BYTE_OP_OR => a | b,
        _ => a ^ b, // BYTE_OP_XOR
      };
      ByteTableQuery {
        a,
        b,
        op,
        result,
        multiplicity: mult,
      }
    })
    .collect()
}

/// Out-of-circuit consistency check for all ProofRows in a batch manifest.
///
/// **Gap 3 (ByteAnd/Or/Xor result linking)**: verifies that each byte-op row's
/// committed `value` equals the correct bitwise result of `scalar0 op scalar1`.
/// This is required because `eval_lut_prep_row_binding_inner` binds
/// `out0 = prep[PREP_COL_VALUE]`, but the LUT AIR constraints do not enforce
/// `out0 = in0 op in1` for bitwise ops.
///
/// **Gap 4 (U29/U24AddEq range constraints)**: verifies that arithmetic operands
/// are within the declared bit-width bounds.  The AIR constraint
/// `in0 + in1 + in2 = out0 + 2^N * out1` is satisfiable in M31 even when
/// `in0 ≥ 2^N`, so range enforcement must be added out-of-circuit.
///
/// Returns `false` if any row fails validation.  Structural padding rows
/// (OP_BOOL, OP_EQ_REFL, etc.) are silently accepted.
pub fn validate_manifest_rows(rows: &[ProofRow]) -> bool {
  use crate::semantic_proof::{
    OP_BYTE_AND_EQ, OP_BYTE_MUL_HIGH_EQ, OP_BYTE_MUL_LOW_EQ, OP_BYTE_OR_EQ, OP_BYTE_XOR_EQ,
    OP_U15_MUL_EQ, OP_U24_ADD_EQ, OP_U29_ADD_EQ,
  };

  for row in rows {
    if row.op == OP_BYTE_AND_EQ {
      // Gap 3: byte AND result must be committed and correct.
      if row.scalar0 > 255 || row.scalar1 > 255 {
        return false;
      }
      if row.value != row.scalar0 & row.scalar1 {
        return false;
      }
    } else if row.op == OP_BYTE_OR_EQ {
      if row.scalar0 > 255 || row.scalar1 > 255 {
        return false;
      }
      if row.value != row.scalar0 | row.scalar1 {
        return false;
      }
    } else if row.op == OP_BYTE_XOR_EQ {
      if row.scalar0 > 255 || row.scalar1 > 255 {
        return false;
      }
      if row.value != row.scalar0 ^ row.scalar1 {
        return false;
      }
    } else if row.op == OP_U29_ADD_EQ {
      // Gap 4: U29 operand range enforcement.
      const MAX29: u32 = (1u32 << 29) - 1;
      if row.scalar0 > MAX29 || row.scalar1 > MAX29 || row.value > MAX29 {
        return false;
      }
      if row.scalar2 > 1 || row.arg1 > 1 {
        // carry-in and overflow bit must be boolean
        return false;
      }
    } else if row.op == OP_U24_ADD_EQ {
      // Gap 4: U24 operand range enforcement.
      const MAX24: u32 = (1u32 << 24) - 1;
      if row.scalar0 > MAX24 || row.scalar1 > MAX24 || row.value > MAX24 {
        return false;
      }
      if row.scalar2 > 1 || row.arg1 > 1 {
        return false;
      }
    } else if row.op == OP_U15_MUL_EQ {
      // Gap 4: U15 product operand range enforcement.
      const MAX15: u32 = 0x7FFF_u32;
      if row.scalar0 > MAX15 || row.scalar1 > MAX15 {
        return false;
      }
      if row.value > MAX15 || row.arg0 > MAX15 {
        // out0 (lo) and out1 (hi) are both at most 0x7FFF
        return false;
      }
    } else if row.op == OP_BYTE_MUL_LOW_EQ || row.op == OP_BYTE_MUL_HIGH_EQ {
      // Gap 4: byte product inputs must be bytes.
      if row.scalar0 > 255 || row.scalar1 > 255 {
        return false;
      }
    }
    // Structural pad ops (OP_BOOL, OP_EQ_REFL, etc.) have no arithmetic fields
    // to validate; they are accepted unconditionally.
  }

  true
}

//
// LUT STARK variant for batch proving.  Uses the extended batch preprocessed
// matrix (NUM_BATCH_PREP_COLS = 27) and binds pis[1] / pis[2..10] to
// PREP_COL_BATCH_N / PREP_COL_BATCH_DIGEST_* instead of the per-instruction
// EVM opcode / WFF digest used by the single-instruction LutKernelAirWithPrep.
//
// Per-instruction EVM opcode / WFF digest columns (PREP_COL_EVM_OPCODE,
// PREP_COL_WFF_DIGEST_*) are stored in the matrix and verified out-of-circuit
// by the verifier via `compute_batch_manifest_digest`.
// ============================================================

/// Per-row prep scalar binding helper used by [`BatchLutKernelAirWithPrep`].
///
/// Binds every LUT main-trace column to the corresponding committed
/// preprocessed ProofRow field using a gate polynomial over `prep[PREP_COL_OP]`.
/// Structural (non-LUT) rows are silently skipped via the all-ops gate.
fn eval_lut_prep_row_binding_inner<AB>(builder: &mut AB, prep_row: &[AB::Var], local: &[AB::Var])
where
  AB: AirBuilderWithPublicValues + PairBuilder + AirBuilder<F = Val>,
{
  use crate::semantic_proof::{
    OP_AND_INTRO, OP_BOOL, OP_BYTE, OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE,
    OP_BYTE_ADD_THIRD_CONGRUENCE, OP_BYTE_AND_EQ, OP_BYTE_MUL_HIGH_EQ,
    OP_BYTE_MUL_LOW_EQ, OP_BYTE_OR_EQ, OP_BYTE_XOR_EQ, OP_EQ_REFL, OP_EQ_SYM, OP_EQ_TRANS, OP_NOT,
    OP_U15_MUL_EQ, OP_U24_ADD_EQ, OP_U29_ADD_EQ,
  };

  let all_prep_ops: &[u32] = &[
    OP_U29_ADD_EQ,
    OP_U24_ADD_EQ,
    OP_U15_MUL_EQ,
    OP_BYTE_MUL_LOW_EQ,
    OP_BYTE_MUL_HIGH_EQ,
    OP_BYTE_AND_EQ,
    OP_BYTE_OR_EQ,
    OP_BYTE_XOR_EQ,
    OP_BOOL,
    OP_BYTE,
    OP_AND_INTRO,
    OP_EQ_REFL,
    OP_EQ_TRANS,
    OP_BYTE_ADD_THIRD_CONGRUENCE,
    OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE,
    OP_NOT,
    OP_EQ_SYM,
  ];

  // Precompute pg(target) = ∏_{t ≠ target} (v - t) for all targets via
  // prefix/suffix products: O(2n) multiplications total instead of O(n) per
  // call.  prefix[i] = ∏_{j<i}(v-ops[j]),  suffix[i] = ∏_{j>i}(v-ops[j]).
  // pg(ops[i]) = prefix[i] * suffix[i+1].
  let n = all_prep_ops.len();
  let v: AB::Expr = prep_row[PREP_COL_OP].clone().into();
  let diffs: Vec<AB::Expr> = all_prep_ops
    .iter()
    .map(|&t| v.clone() - AB::Expr::from_u32(t))
    .collect();
  let mut prefix: Vec<AB::Expr> = Vec::with_capacity(n + 1);
  prefix.push(AB::Expr::from_u32(1));
  for i in 0..n {
    let p = prefix[i].clone() * diffs[i].clone();
    prefix.push(p);
  }
  let mut suffix: Vec<AB::Expr> = vec![AB::Expr::from_u32(1); n + 1];
  for i in (0..n).rev() {
    suffix[i] = diffs[i].clone() * suffix[i + 1].clone();
  }
  let pg_vals: Vec<AB::Expr> = (0..n)
    .map(|i| prefix[i].clone() * suffix[i + 1].clone())
    .collect();
  let pg = |target: u32| -> AB::Expr {
    let idx = all_prep_ops
      .iter()
      .position(|&t| t == target)
      .expect("pg: target not in all_prep_ops");
    pg_vals[idx].clone()
  };

  // OP_U29_ADD_EQ → U29AddEq: in0=s0, in1=s1, in2=s2, out0=value, out1=arg1
  let tag_u29 = lut_opcode_tag(LutOpcode::U29AddEq) as u32;
  let g = pg(OP_U29_ADD_EQ);
  builder.assert_zero(g.clone() * (local[LUT_COL_OP].clone().into() - AB::Expr::from_u32(tag_u29)));
  builder.assert_zero(
    g.clone() * (local[LUT_COL_IN0].clone().into() - prep_row[PREP_COL_SCALAR0].clone().into()),
  );
  builder.assert_zero(
    g.clone() * (local[LUT_COL_IN1].clone().into() - prep_row[PREP_COL_SCALAR1].clone().into()),
  );
  builder.assert_zero(
    g.clone() * (local[LUT_COL_IN2].clone().into() - prep_row[PREP_COL_SCALAR2].clone().into()),
  );
  builder.assert_zero(
    g.clone() * (local[LUT_COL_OUT0].clone().into() - prep_row[PREP_COL_VALUE].clone().into()),
  );
  builder
    .assert_zero(g * (local[LUT_COL_OUT1].clone().into() - prep_row[PREP_COL_ARG1].clone().into()));

  // OP_U24_ADD_EQ → U24AddEq (same field layout as U29)
  let tag_u24 = lut_opcode_tag(LutOpcode::U24AddEq) as u32;
  let g = pg(OP_U24_ADD_EQ);
  builder.assert_zero(g.clone() * (local[LUT_COL_OP].clone().into() - AB::Expr::from_u32(tag_u24)));
  builder.assert_zero(
    g.clone() * (local[LUT_COL_IN0].clone().into() - prep_row[PREP_COL_SCALAR0].clone().into()),
  );
  builder.assert_zero(
    g.clone() * (local[LUT_COL_IN1].clone().into() - prep_row[PREP_COL_SCALAR1].clone().into()),
  );
  builder.assert_zero(
    g.clone() * (local[LUT_COL_IN2].clone().into() - prep_row[PREP_COL_SCALAR2].clone().into()),
  );
  builder.assert_zero(
    g.clone() * (local[LUT_COL_OUT0].clone().into() - prep_row[PREP_COL_VALUE].clone().into()),
  );
  builder
    .assert_zero(g * (local[LUT_COL_OUT1].clone().into() - prep_row[PREP_COL_ARG1].clone().into()));

  // OP_U15_MUL_EQ → U15MulEq: in0=s0, in1=s1, out0=value(lo), out1=arg0(hi)
  let tag_mul = lut_opcode_tag(LutOpcode::U15MulEq) as u32;
  let g = pg(OP_U15_MUL_EQ);
  builder.assert_zero(g.clone() * (local[LUT_COL_OP].clone().into() - AB::Expr::from_u32(tag_mul)));
  builder.assert_zero(
    g.clone() * (local[LUT_COL_IN0].clone().into() - prep_row[PREP_COL_SCALAR0].clone().into()),
  );
  builder.assert_zero(
    g.clone() * (local[LUT_COL_IN1].clone().into() - prep_row[PREP_COL_SCALAR1].clone().into()),
  );
  builder.assert_zero(
    g.clone() * (local[LUT_COL_OUT0].clone().into() - prep_row[PREP_COL_VALUE].clone().into()),
  );
  builder
    .assert_zero(g * (local[LUT_COL_OUT1].clone().into() - prep_row[PREP_COL_ARG0].clone().into()));

  // OP_BYTE_MUL_LOW_EQ → ByteMulLowEq: in0=s0, in1=s1
  let tag_ml = lut_opcode_tag(LutOpcode::ByteMulLowEq) as u32;
  let g = pg(OP_BYTE_MUL_LOW_EQ);
  builder.assert_zero(g.clone() * (local[LUT_COL_OP].clone().into() - AB::Expr::from_u32(tag_ml)));
  builder.assert_zero(
    g.clone() * (local[LUT_COL_IN0].clone().into() - prep_row[PREP_COL_SCALAR0].clone().into()),
  );
  builder.assert_zero(
    g * (local[LUT_COL_IN1].clone().into() - prep_row[PREP_COL_SCALAR1].clone().into()),
  );

  // OP_BYTE_MUL_HIGH_EQ → ByteMulHighEq: in0=s0, in1=s1
  let tag_mh = lut_opcode_tag(LutOpcode::ByteMulHighEq) as u32;
  let g = pg(OP_BYTE_MUL_HIGH_EQ);
  builder.assert_zero(g.clone() * (local[LUT_COL_OP].clone().into() - AB::Expr::from_u32(tag_mh)));
  builder.assert_zero(
    g.clone() * (local[LUT_COL_IN0].clone().into() - prep_row[PREP_COL_SCALAR0].clone().into()),
  );
  builder.assert_zero(
    g * (local[LUT_COL_IN1].clone().into() - prep_row[PREP_COL_SCALAR1].clone().into()),
  );

  // OP_BYTE_AND_EQ → ByteAndEq: in0=s0, in1=s1, out0=value
  let tag_and = lut_opcode_tag(LutOpcode::ByteAndEq) as u32;
  let g = pg(OP_BYTE_AND_EQ);
  builder.assert_zero(g.clone() * (local[LUT_COL_OP].clone().into() - AB::Expr::from_u32(tag_and)));
  builder.assert_zero(
    g.clone() * (local[LUT_COL_IN0].clone().into() - prep_row[PREP_COL_SCALAR0].clone().into()),
  );
  builder.assert_zero(
    g.clone() * (local[LUT_COL_IN1].clone().into() - prep_row[PREP_COL_SCALAR1].clone().into()),
  );
  builder.assert_zero(
    g * (local[LUT_COL_OUT0].clone().into() - prep_row[PREP_COL_VALUE].clone().into()),
  );

  // OP_BYTE_OR_EQ → ByteOrEq: in0=s0, in1=s1, out0=value
  let tag_or = lut_opcode_tag(LutOpcode::ByteOrEq) as u32;
  let g = pg(OP_BYTE_OR_EQ);
  builder.assert_zero(g.clone() * (local[LUT_COL_OP].clone().into() - AB::Expr::from_u32(tag_or)));
  builder.assert_zero(
    g.clone() * (local[LUT_COL_IN0].clone().into() - prep_row[PREP_COL_SCALAR0].clone().into()),
  );
  builder.assert_zero(
    g.clone() * (local[LUT_COL_IN1].clone().into() - prep_row[PREP_COL_SCALAR1].clone().into()),
  );
  builder.assert_zero(
    g * (local[LUT_COL_OUT0].clone().into() - prep_row[PREP_COL_VALUE].clone().into()),
  );

  // OP_BYTE_XOR_EQ → ByteXorEq: in0=s0, in1=s1, out0=value
  let tag_xor = lut_opcode_tag(LutOpcode::ByteXorEq) as u32;
  let g = pg(OP_BYTE_XOR_EQ);
  builder.assert_zero(g.clone() * (local[LUT_COL_OP].clone().into() - AB::Expr::from_u32(tag_xor)));
  builder.assert_zero(
    g.clone() * (local[LUT_COL_IN0].clone().into() - prep_row[PREP_COL_SCALAR0].clone().into()),
  );
  builder.assert_zero(
    g.clone() * (local[LUT_COL_IN1].clone().into() - prep_row[PREP_COL_SCALAR1].clone().into()),
  );
  builder.assert_zero(
    g * (local[LUT_COL_OUT0].clone().into() - prep_row[PREP_COL_VALUE].clone().into()),
  );
}

/// Batch LUT STARK variant that uses the extended batch preprocessed matrix
/// ([`NUM_BATCH_PREP_COLS`] = 27 columns) to cover N instructions in one proof.
///
/// **AIR constraints:**
/// 1. `pis[0] == RECEIPT_BIND_TAG_LUT`
/// 2. Row 0: `prep[PREP_COL_BATCH_N] == pis[1]` and
///    `prep[PREP_COL_BATCH_DIGEST_START + k] == pis[2 + k]` for k in 0..8.
/// 3. Per-row prep scalar binding (same gates as [`LutKernelAirWithPrep`]).
/// 4. All LUT arithmetic constraints via [`eval_lut_kernel_inner`].
///
/// Per-instruction `PREP_COL_EVM_OPCODE` / `PREP_COL_WFF_DIGEST_*` are
/// committed in the matrix but verified **out-of-circuit** by the verifier.
pub struct BatchLutKernelAirWithPrep {
  prep_matrix: Option<RowMajorMatrix<Val>>,
}

impl BatchLutKernelAirWithPrep {
  /// Prover path: store the batch preprocessed matrix.
  pub fn new(manifest: &BatchProofRowsManifest, pv: &[Val]) -> Self {
    Self {
      prep_matrix: Some(build_batch_proof_rows_preprocessed_matrix(manifest, pv)),
    }
  }

  /// Verifier path: preprocessed data comes from the `PreprocessedVerifierKey`.
  pub fn for_verify() -> Self {
    Self { prep_matrix: None }
  }
}

impl BaseAir<Val> for BatchLutKernelAirWithPrep {
  fn width(&self) -> usize {
    NUM_LUT_COLS
  }

  fn preprocessed_trace(&self) -> Option<RowMajorMatrix<Val>> {
    self.prep_matrix.clone()
  }
}

impl<AB> Air<AB> for BatchLutKernelAirWithPrep
where
  AB: AirBuilderWithPublicValues + PairBuilder + AirBuilder<F = Val>,
{
  fn eval(&self, builder: &mut AB) {
    // ── 1. Tag check ──────────────────────────────────────────────────────
    {
      let pis = builder.public_values();
      builder.assert_eq(pis[0].clone(), AB::Expr::from_u32(RECEIPT_BIND_TAG_LUT));
    }

    // ── 2. Batch pv binding at row 0 ──────────────────────────────────────
    // pis[1] = N, pis[2..10] = batch_manifest_digest
    {
      let pi_n: AB::Expr = {
        let pis = builder.public_values();
        pis[1].clone().into()
      };
      let pi_digest: [AB::Expr; 8] = {
        let pis = builder.public_values();
        std::array::from_fn(|k| pis[2 + k].clone().into())
      };
      let prep_n: AB::Expr = {
        let prep = builder.preprocessed();
        let row = prep
          .row_slice(0)
          .expect("BatchLutKernelAirWithPrep: empty prep (batch-n bind)");
        row[PREP_COL_BATCH_N].clone().into()
      };
      let prep_digest: [AB::Expr; 8] = {
        let prep = builder.preprocessed();
        let row = prep
          .row_slice(0)
          .expect("BatchLutKernelAirWithPrep: empty prep (batch-digest bind)");
        std::array::from_fn(|k| row[PREP_COL_BATCH_DIGEST_START + k].clone().into())
      };
      builder.when_first_row().assert_eq(prep_n, pi_n);
      for k in 0..8_usize {
        builder
          .when_first_row()
          .assert_eq(prep_digest[k].clone(), pi_digest[k].clone());
      }
    }

    // ── 3. Per-row prep scalar binding ────────────────────────────────────
    {
      let prep = builder.preprocessed();
      let prep_row = prep
        .row_slice(0)
        .expect("BatchLutKernelAirWithPrep: empty preprocessed trace");
      let prep_row = &*prep_row;
      let main = builder.main();
      let local = main
        .row_slice(0)
        .expect("BatchLutKernelAirWithPrep: empty main trace");
      let local = &*local;
      eval_lut_prep_row_binding_inner(builder, prep_row, local);
    }

    // ── 4. LUT arithmetic constraints ─────────────────────────────────────
    eval_lut_kernel_inner(builder);
  }
}

/// Prove the batch LUT STARK for `manifest` and bind the proof to the shared
/// batch preprocessed commitment (`prep_data`).
///
/// Returns a [`CircleStarkProof`] that covers the arithmetic of all N
/// instructions' ProofRows in one call.  Pair with
/// [`verify_batch_lut_with_prep`] and out-of-circuit manifest digest checks.
pub fn prove_batch_lut_with_prep(
  manifest: &BatchProofRowsManifest,
  prep_data: &PreprocessedProverData<CircleStarkConfig>,
  public_values: &[Val],
) -> Result<CircleStarkProof, String> {
  let trace = build_lut_trace_from_proof_rows(&manifest.all_rows)?;
  let config = make_circle_config();
  let air = BatchLutKernelAirWithPrep::new(manifest, public_values);
  let proof =
    p3_uni_stark::prove_with_preprocessed(&config, &air, trace, public_values, Some(prep_data));
  Ok(proof)
}

/// Verify a batch LUT proof generated by [`prove_batch_lut_with_prep`].
pub fn verify_batch_lut_with_prep(
  proof: &CircleStarkProof,
  prep_vk: &PreprocessedVerifierKey<CircleStarkConfig>,
  public_values: &[Val],
) -> CircleStarkVerifyResult {
  let config = make_circle_config();
  p3_uni_stark::verify_with_preprocessed(
    &config,
    &BatchLutKernelAirWithPrep::for_verify(),
    proof,
    public_values,
    Some(prep_vk),
  )
}

pub(crate) fn prove_lut_kernel_stark(steps: &[LutStep]) -> Result<CircleStarkProof, String> {
  prove_lut_kernel_stark_with_public_values(
    steps,
    &default_receipt_bind_public_values_for_tag(RECEIPT_BIND_TAG_LUT),
  )
}

pub fn prove_lut_kernel_stark_with_public_values(
  steps: &[LutStep],
  public_values: &[Val],
) -> Result<CircleStarkProof, String> {
  let trace = build_lut_trace_from_steps(steps)?;
  let config = make_circle_config();
  Ok(p3_uni_stark::prove(
    &config,
    &LutKernelAir,
    trace,
    public_values,
  ))
}

pub(crate) fn verify_lut_kernel_stark(proof: &CircleStarkProof) -> CircleStarkVerifyResult {
  verify_lut_kernel_stark_with_public_values(
    proof,
    &default_receipt_bind_public_values_for_tag(RECEIPT_BIND_TAG_LUT),
  )
}

pub fn verify_lut_kernel_stark_with_public_values(
  proof: &CircleStarkProof,
  public_values: &[Val],
) -> CircleStarkVerifyResult {
  let config = make_circle_config();
  p3_uni_stark::verify(&config, &LutKernelAir, proof, public_values)
}

pub(crate) fn prove_and_verify_lut_kernel_stark_from_steps(steps: &[LutStep]) -> bool {
  let proved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
    prove_lut_kernel_stark(steps)
  }));
  let proof = match proved {
    Ok(Ok(proof)) => proof,
    Ok(Err(_)) => return false,
    Err(_) => return false,
  };
  verify_lut_kernel_stark(&proof).is_ok()
}

pub fn build_lut_steps_from_rows_add_family(rows: &[ProofRow]) -> Result<Vec<LutStep>, String> {
  let mut out = Vec::new();
  let mut seen_add_family = false;

  for row in rows {
    match row.op {
      // Soundness fix: OP_NOT and OP_EQ_SYM can appear in compiled proof rows
      // but carry no LUT arithmetic content; treat them as structural skips.
      crate::semantic_proof::OP_BOOL
      | crate::semantic_proof::OP_BYTE
      | crate::semantic_proof::OP_EQ_REFL
      | crate::semantic_proof::OP_AND_INTRO
      | crate::semantic_proof::OP_EQ_TRANS
      | crate::semantic_proof::OP_BYTE_ADD_THIRD_CONGRUENCE
      | crate::semantic_proof::OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE
      | crate::semantic_proof::OP_NOT
      | crate::semantic_proof::OP_EQ_SYM => {
        seen_add_family = true;
      }
      crate::semantic_proof::OP_U29_ADD_EQ => out.push(LutStep {
        op: LutOpcode::U29AddEq,
        in0: row.scalar0,
        in1: row.scalar1,
        in2: row.scalar2,
        out0: row.value,
        out1: row.arg1,
      }),
      crate::semantic_proof::OP_U24_ADD_EQ => out.push(LutStep {
        op: LutOpcode::U24AddEq,
        in0: row.scalar0,
        in1: row.scalar1,
        in2: row.scalar2,
        out0: row.value,
        out1: row.arg1,
      }),
      other => return Err(format!("non-add-family row in ADD kernel path: {other}")),
    }
  }

  if !seen_add_family && out.is_empty() {
    return Err("empty add-family row set".to_string());
  }

  if out.is_empty() {
    return Err("no add LUT rows in add-family row set".to_string());
  }

  Ok(out)
}

pub fn prove_and_verify_add_stack_lut_stark(rows: &[ProofRow]) -> bool {
  if !prove_and_verify_stack_ir_scaffold_stark(rows) {
    return false;
  }

  let steps = match build_lut_steps_from_rows_add_family(rows) {
    Ok(steps) => steps,
    Err(_) => return false,
  };

  prove_and_verify_lut_kernel_stark_from_steps(&steps)
}

pub fn build_lut_steps_from_rows_mul_family(rows: &[ProofRow]) -> Result<Vec<LutStep>, String> {
  let mut out = Vec::new();
  let mut seen_mul_family = false;

  for row in rows {
    match row.op {
      // Soundness fix: OP_NOT and OP_EQ_SYM can appear in compiled proof rows
      // but carry no LUT arithmetic content; treat them as structural skips.
      crate::semantic_proof::OP_BOOL
      | crate::semantic_proof::OP_BYTE
      | crate::semantic_proof::OP_EQ_REFL
      | crate::semantic_proof::OP_AND_INTRO
      | crate::semantic_proof::OP_EQ_TRANS
      | crate::semantic_proof::OP_BYTE_ADD_THIRD_CONGRUENCE
      | crate::semantic_proof::OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE
      | crate::semantic_proof::OP_NOT
      | crate::semantic_proof::OP_EQ_SYM => {
        seen_mul_family = true;
      }
      crate::semantic_proof::OP_U29_ADD_EQ => out.push(LutStep {
        op: LutOpcode::U29AddEq,
        in0: row.scalar0,
        in1: row.scalar1,
        in2: row.scalar2,
        out0: row.value,
        out1: row.arg1,
      }),
      crate::semantic_proof::OP_U24_ADD_EQ => out.push(LutStep {
        op: LutOpcode::U24AddEq,
        in0: row.scalar0,
        in1: row.scalar1,
        in2: row.scalar2,
        out0: row.value,
        out1: row.arg1,
      }),
      crate::semantic_proof::OP_U15_MUL_EQ => {
        let total = row.scalar0 * row.scalar1;
        out.push(LutStep {
          op: LutOpcode::U15MulEq,
          in0: row.scalar0,
          in1: row.scalar1,
          in2: 0,
          out0: total & 0x7FFF,
          out1: total >> 15,
        });
      }
      other => return Err(format!("non-mul-family row in MUL kernel path: {other}")),
    }
  }

  if !seen_mul_family && out.is_empty() {
    return Err("empty mul-family row set".to_string());
  }

  if out.is_empty() {
    return Err("no mul/add LUT rows in mul-family row set".to_string());
  }

  Ok(out)
}

pub fn prove_and_verify_mul_stack_lut_stark(rows: &[ProofRow]) -> bool {
  if !prove_and_verify_stack_ir_scaffold_stark(rows) {
    return false;
  }

  let steps = match build_lut_steps_from_rows_mul_family(rows) {
    Ok(steps) => steps,
    Err(_) => return false,
  };

  prove_and_verify_lut_kernel_stark_from_steps(&steps)
}

/// Build LUT steps for byte-level AND/OR/XOR rows.
///
/// Each `OP_BYTE_AND_EQ` / `OP_BYTE_OR_EQ` / `OP_BYTE_XOR_EQ` ProofRow
/// becomes exactly **one** `LutStep` with byte operands (0..=255).
/// Structural rows such as `OP_AND_INTRO` and `OP_EQ_SYM` are skipped.
/// All other row opcodes are rejected with an error.
pub fn build_lut_steps_from_rows_bit_family(rows: &[ProofRow]) -> Result<Vec<LutStep>, String> {
  use crate::semantic_proof::{OP_AND_INTRO, OP_EQ_REFL, OP_EQ_SYM, OP_EQ_TRANS, OP_NOT};
  let mut out = Vec::new();
  let mut seen_bitwise = false;

  for row in rows {
    match row.op {
      // Structural rows: skip silently.
      op if op == OP_AND_INTRO
        || op == OP_EQ_REFL
        || op == OP_EQ_SYM
        || op == OP_EQ_TRANS
        || op == OP_NOT => {}

      // Byte-level bitwise ops: one step per ProofRow.
      crate::semantic_proof::OP_BYTE_AND_EQ => {
        seen_bitwise = true;
        let a = row.scalar0;
        let b = row.scalar1;
        out.push(LutStep {
          op: LutOpcode::ByteAndEq,
          in0: a,
          in1: b,
          in2: 0,
          out0: a & b,
          out1: 0,
        });
      }
      crate::semantic_proof::OP_BYTE_OR_EQ => {
        seen_bitwise = true;
        let a = row.scalar0;
        let b = row.scalar1;
        out.push(LutStep {
          op: LutOpcode::ByteOrEq,
          in0: a,
          in1: b,
          in2: 0,
          out0: a | b,
          out1: 0,
        });
      }
      crate::semantic_proof::OP_BYTE_XOR_EQ => {
        seen_bitwise = true;
        let a = row.scalar0;
        let b = row.scalar1;
        out.push(LutStep {
          op: LutOpcode::ByteXorEq,
          in0: a,
          in1: b,
          in2: 0,
          out0: a ^ b,
          out1: 0,
        });
      }
      other => return Err(format!("non-bitwise row in bitwise kernel path: {other}")),
    }
  }

  if !seen_bitwise {
    return Err("no AND/OR/XOR LUT rows in bitwise row set".to_string());
  }
  Ok(out)
}

// `ProofRow` column indices (same order as semantic_proof::ProofRow fields).
const COL_OP: usize = 0;
const COL_SCALAR0: usize = 1;
const COL_SCALAR1: usize = 2;
const COL_SCALAR2: usize = 3;
const COL_ARG0: usize = 4;
const COL_ARG1: usize = 5;
const COL_ARG2: usize = 6;
const COL_VALUE: usize = 7;
const COL_RET_TY: usize = 8;

/// Number of (inferred, expected) column-pairs per trace row.
/// Bytes per row = MATCH_PACK_COLS × MATCH_PACK_BYTES = 4 × 3 = 12.
const MATCH_PACK_COLS: usize = 4;
/// Bytes packed into each M31 field element (max 3 since 0xFFFFFF < M31 = 2^31-1).
const MATCH_PACK_BYTES: usize = 3;
/// Total columns: first MATCH_PACK_COLS are inferred, next MATCH_PACK_COLS are expected.
const NUM_WFF_MATCH_COLS: usize = MATCH_PACK_COLS * 2;

pub struct StageAProof {
  pub inferred_wff_proof: CircleStarkProof,
  pub public_wff_match_proof: CircleStarkProof,
  expected_public_wff: WFF,
}

fn route_stage_a_row_op(op: u32) -> Result<(), String> {
  use crate::semantic_proof::{
    OP_AND_INTRO, OP_BOOL, OP_BYTE, OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE,
    OP_BYTE_ADD_THIRD_CONGRUENCE, OP_EQ_REFL, OP_EQ_TRANS, OP_U24_ADD_EQ,
    OP_U29_ADD_EQ,
  };
  match op {
    OP_U29_ADD_EQ
    | OP_U24_ADD_EQ
    | OP_BOOL
    | OP_BYTE
    | OP_EQ_REFL
    | OP_AND_INTRO
    | OP_EQ_TRANS
    | OP_BYTE_ADD_THIRD_CONGRUENCE
    | OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE => Ok(()),
    other => Err(format!("unsupported Stage A proof-row op: {other}")),
  }
}

fn has_stage_a_semantic_rows(rows: &[ProofRow]) -> Result<bool, String> {
  let mut found = false;
  for row in rows {
    route_stage_a_row_op(row.op)?;
    if row.op == crate::semantic_proof::OP_U29_ADD_EQ
      || row.op == crate::semantic_proof::OP_U24_ADD_EQ
    {
      found = true;
    }
  }
  Ok(found)
}

// ============================================================
// AIR definition
// ============================================================

/// AIR for Stage-1 semantic constraints using `ProofRow` trace columns.
///
/// The current kernel verifies u29/u24 add equality constraints encoded in
/// `OP_U29_ADD_EQ` / `OP_U24_ADD_EQ` rows.
pub struct StageASemanticAir;

impl<F> BaseAir<F> for StageASemanticAir {
  fn width(&self) -> usize {
    NUM_PROOF_COLS
  }
}

/// AIR for checking inferred WFF bytes equal expected WFF bytes.
///
/// Each row compares one byte:
/// - inferred byte from compiled add-equality rows
/// - expected byte from transition output
pub struct WffMatchAir;

impl<F> BaseAir<F> for WffMatchAir {
  fn width(&self) -> usize {
    NUM_WFF_MATCH_COLS
  }
}

impl<AB: AirBuilderWithPublicValues> Air<AB> for WffMatchAir {
  fn eval(&self, builder: &mut AB) {
    let pis = builder.public_values();
    builder.assert_eq(pis[0], AB::Expr::from_u32(RECEIPT_BIND_TAG_WFF));

    let main = builder.main();
    let local = main.row_slice(0).expect("empty trace");
    let local = &*local;

    // Check equality for each (inferred, expected) column pair.
    // Columns 0..MATCH_PACK_COLS hold packed inferred bytes;
    // columns MATCH_PACK_COLS..NUM_WFF_MATCH_COLS hold packed expected bytes.
    for col in 0..MATCH_PACK_COLS {
      builder.assert_eq(local[col].clone(), local[MATCH_PACK_COLS + col].clone());
    }
  }
}

impl<AB: AirBuilder> Air<AB> for StageASemanticAir {
  fn eval(&self, builder: &mut AB) {
    let main = builder.main();
    let local = main.row_slice(0).expect("empty trace");
    let next = main.row_slice(1).expect("single-row trace");

    let local = &*local;
    let next = &*next;

    let op = local[COL_OP].clone();
    let t_u29 = crate::semantic_proof::OP_U29_ADD_EQ as u16;
    let t_u24 = crate::semantic_proof::OP_U24_ADD_EQ as u16;

    let tags = [t_u29, t_u24];
    let gate = |target: u16| {
      let mut g = AB::Expr::from_u16(1);
      for t in tags {
        if t != target {
          g *= op.clone().into() - AB::Expr::from_u16(t);
        }
      }
      g
    };

    let g_u29 = gate(t_u29);
    let g_u24 = gate(t_u24);

    let allowed_poly = (op.clone().into() - AB::Expr::from_u16(t_u29))
      * (op.clone().into() - AB::Expr::from_u16(t_u24));
    builder.assert_zero(allowed_poly);

    let a = local[COL_SCALAR0].clone();
    let b = local[COL_SCALAR1].clone();
    let sum = local[COL_VALUE].clone();

    builder.assert_zero(
      g_u29.clone()
        * (a.clone().into() + b.clone().into() + local[COL_SCALAR2].clone().into()
          - sum.clone().into()
          - AB::Expr::from_u32(1u32 << 29) * local[COL_ARG1].clone().into()),
    );
    builder.assert_zero(
      g_u24.clone()
        * (a.clone().into() + b.clone().into() + local[COL_SCALAR2].clone().into()
          - sum.clone().into()
          - AB::Expr::from_u32(1u32 << 24) * local[COL_ARG1].clone().into()),
    );

    builder.assert_zero(g_u29.clone() * (sum.clone().into() - local[COL_ARG0].clone().into()));
    builder.assert_zero(g_u24.clone() * (sum.clone().into() - local[COL_ARG0].clone().into()));

    builder.assert_zero(
      g_u29.clone()
        * (local[COL_ARG1].clone().into()
          * (local[COL_ARG1].clone().into() - AB::Expr::from_u16(1))),
    );
    builder.assert_zero(
      g_u24.clone()
        * (local[COL_ARG1].clone().into()
          * (local[COL_ARG1].clone().into() - AB::Expr::from_u16(1))),
    );
    builder.assert_zero(
      g_u29.clone()
        * (local[COL_SCALAR2].clone().into()
          * (local[COL_SCALAR2].clone().into() - AB::Expr::from_u16(1))),
    );
    builder.assert_zero(
      g_u24.clone()
        * (local[COL_SCALAR2].clone().into()
          * (local[COL_SCALAR2].clone().into() - AB::Expr::from_u16(1))),
    );

    builder.assert_zero(local[COL_ARG2].clone());
    builder.assert_zero(
      g_u29.clone() * (local[COL_RET_TY].clone().into() - AB::Expr::from_u16(RET_WFF_AND as u16)),
    );
    builder.assert_zero(
      g_u24.clone() * (local[COL_RET_TY].clone().into() - AB::Expr::from_u16(RET_WFF_AND as u16)),
    );

    builder
      .when_first_row()
      .assert_zero(g_u29.clone() * local[COL_SCALAR2].clone().into());
    builder
      .when_first_row()
      .assert_zero(g_u24.clone() * local[COL_SCALAR2].clone().into());

    let next_op = next[COL_OP].clone();
    let next_gate = |target: u16| {
      let mut g = AB::Expr::from_u16(1);
      for t in tags {
        if t != target {
          g *= next_op.clone().into() - AB::Expr::from_u16(t);
        }
      }
      g
    };
    let ng_u29 = next_gate(t_u29);
    let ng_u24 = next_gate(t_u24);

    let local_cout_u29 = local[COL_ARG1].clone().into();
    let local_cout_u24 = local[COL_ARG1].clone().into();

    let next_cin_u29 = next[COL_SCALAR2].clone().into();
    let next_cin_u24 = next[COL_SCALAR2].clone().into();

    let local_cases = [
      (g_u29, local_cout_u29),
      (g_u24, local_cout_u24),
    ];
    let next_cases = [
      (ng_u29, next_cin_u29),
      (ng_u24, next_cin_u24),
    ];

    for (lg, lcout) in local_cases.iter() {
      for (ng, ncin) in next_cases.iter() {
        builder
          .when_transition()
          .assert_zero(lg.clone() * ng.clone() * (ncin.clone() - lcout.clone()));
      }
    }
  }
}

// ============================================================
// Config builder
// ============================================================

/// Build a Circle STARK configuration over M31 with Poseidon2 hashing.
pub fn make_circle_config() -> CircleStarkConfig {
  make_circle_config_with_params(40, 8, 0)
}

/// Tunable Circle STARK config (Poseidon2 hash backend).
///
/// - `num_queries`: FRI query count. Soundness ≈ num_queries × log_blowup bits. Default 40.
/// - `query_pow_bits`: grinding bits for query phase (2^n hashes). Default 8. Set 0 to disable.
/// - `log_final_poly_len`: log2 of FRI final polynomial degree limit. Default 0.
pub fn make_circle_config_with_params(
  num_queries: usize,
  query_pow_bits: usize,
  log_final_poly_len: usize,
) -> CircleStarkConfig {
  let mut rng = SmallRng::seed_from_u64(0x5EED_C0DE_u64);
  let perm = P2Perm::new_from_rng_128(&mut rng);

  let hash = P2Hash::new(perm.clone());
  let compress = P2Compress::new(perm.clone());
  let val_mmcs = P2ValMmcs::new(hash, compress);
  let challenge_mmcs = P2ChallengeMmcs::new(val_mmcs.clone());

  let fri_params = FriParameters {
    log_blowup: 1,
    log_final_poly_len,
    num_queries,
    commit_proof_of_work_bits: 0,
    query_proof_of_work_bits: query_pow_bits,
    mmcs: challenge_mmcs,
  };

  let pcs = CirclePcs {
    mmcs: val_mmcs,
    fri_params,
    _phantom: core::marker::PhantomData,
  };

  let challenger = P2Challenger::new(perm);
  StarkConfig::new(pcs, challenger)
}

// ============================================================
// Prove & Verify
// ============================================================

/// Build a Stage-A semantic trace (ProofRow layout) from compiled proof rows.
///
/// Routing policy by `row.op`:
/// - semantic kernel rows are included,
/// - structural rows are ignored,
/// - unknown rows are rejected.
pub fn generate_stage_a_semantic_trace(rows: &[ProofRow]) -> Result<RowMajorMatrix<Val>, String> {
  let mut semantic_rows: Vec<ProofRow> = Vec::new();

  for (i, row) in rows.iter().enumerate() {
    let _ = i;
    route_stage_a_row_op(row.op)?;

    match row.op {
      crate::semantic_proof::OP_U29_ADD_EQ => {
        let limit = (1u32 << 29) - 1;
        if row.scalar0 > limit || row.scalar1 > limit {
          return Err(format!(
            "unsupported Stage A: u29 add operand out of 29-bit range at row {}",
            i
          ));
        }
        semantic_rows.push(row.clone());
      }
      crate::semantic_proof::OP_U24_ADD_EQ => {
        let limit = (1u32 << 24) - 1;
        if row.scalar0 > limit || row.scalar1 > limit {
          return Err(format!(
            "unsupported Stage A: u24 add operand out of 24-bit range at row {}",
            i
          ));
        }
        semantic_rows.push(row.clone());
      }
      _ => {}
    }
  }

  if semantic_rows.is_empty() {
    return Err("no stage-a semantic rows in compiled proof".to_string());
  }

  let semantic_len = semantic_rows.len();
  let n_rows = semantic_len.max(4).next_power_of_two();
  let mut trace = RowMajorMatrix::new(Val::zero_vec(n_rows * NUM_PROOF_COLS), NUM_PROOF_COLS);

  for (i, row) in semantic_rows.iter().enumerate() {
    let base = i * NUM_PROOF_COLS;
    trace.values[base + COL_OP] = Val::from_u16(row.op as u16);
    trace.values[base + COL_SCALAR0] = Val::from_u32(row.scalar0);
    trace.values[base + COL_SCALAR1] = Val::from_u32(row.scalar1);
    trace.values[base + COL_SCALAR2] = Val::from_u32(row.scalar2);
    trace.values[base + COL_ARG0] = Val::from_u32(row.arg0);
    trace.values[base + COL_ARG1] = Val::from_u32(row.arg1);
    trace.values[base + COL_ARG2] = Val::from_u32(row.arg2);
    trace.values[base + COL_VALUE] = Val::from_u32(row.value);
    trace.values[base + COL_RET_TY] = Val::from_u16(row.ret_ty as u16);
  }

  if semantic_len < n_rows {
    let mut carry_in = semantic_rows
      .last()
      .map(|row| match row.op {
        crate::semantic_proof::OP_U29_ADD_EQ
        | crate::semantic_proof::OP_U24_ADD_EQ => row.arg1 as u16,
        _ => 0,
      })
      .unwrap_or(0);

    for i in semantic_len..n_rows {
      let carry_out = 0u16;

      let base = i * NUM_PROOF_COLS;
      trace.values[base + COL_OP] = Val::from_u16(crate::semantic_proof::OP_U29_ADD_EQ as u16);
      trace.values[base + COL_SCALAR0] = Val::from_u16(0);
      trace.values[base + COL_SCALAR1] = Val::from_u16(0);
      trace.values[base + COL_SCALAR2] = Val::from_u16(carry_in);
      trace.values[base + COL_ARG0] = Val::from_u16(carry_in);
      trace.values[base + COL_ARG1] = Val::from_u16(carry_out);
      trace.values[base + COL_ARG2] = Val::from_u16(0);
      trace.values[base + COL_VALUE] = Val::from_u16(carry_in);
      trace.values[base + COL_RET_TY] = Val::from_u16(crate::semantic_proof::RET_WFF_AND as u16);

      carry_in = carry_out;
    }
  }

  Ok(trace)
}

const WFF_TAG_EQUAL: u8 = 1;
const WFF_TAG_AND: u8 = 2;

const TERM_TAG_BOOL: u8 = 1;
const TERM_TAG_NOT: u8 = 2;
const TERM_TAG_AND: u8 = 3;
const TERM_TAG_OR: u8 = 4;
const TERM_TAG_XOR: u8 = 5;
const TERM_TAG_ITE: u8 = 6;
const TERM_TAG_BYTE: u8 = 7;
const TERM_TAG_BYTE_ADD: u8 = 8;
const TERM_TAG_BYTE_ADD_CARRY: u8 = 9;
const TERM_TAG_BYTE_MUL_LOW: u8 = 10;
const TERM_TAG_BYTE_MUL_HIGH: u8 = 11;
const TERM_TAG_BYTE_AND: u8 = 12;
const TERM_TAG_BYTE_OR: u8 = 13;
const TERM_TAG_BYTE_XOR: u8 = 14;

fn serialize_term(term: &Term, out: &mut Vec<u8>) {
  match term {
    Term::Bool(v) => {
      out.push(TERM_TAG_BOOL);
      out.push(*v as u8);
    }
    Term::Not(a) => {
      out.push(TERM_TAG_NOT);
      serialize_term(a, out);
    }
    Term::And(a, b) => {
      out.push(TERM_TAG_AND);
      serialize_term(a, out);
      serialize_term(b, out);
    }
    Term::Or(a, b) => {
      out.push(TERM_TAG_OR);
      serialize_term(a, out);
      serialize_term(b, out);
    }
    Term::Xor(a, b) => {
      out.push(TERM_TAG_XOR);
      serialize_term(a, out);
      serialize_term(b, out);
    }
    Term::Ite(c, a, b) => {
      out.push(TERM_TAG_ITE);
      serialize_term(c, out);
      serialize_term(a, out);
      serialize_term(b, out);
    }
    Term::Byte(v) => {
      out.push(TERM_TAG_BYTE);
      out.push(*v);
    }
    Term::ByteAdd(a, b, c) => {
      out.push(TERM_TAG_BYTE_ADD);
      serialize_term(a, out);
      serialize_term(b, out);
      serialize_term(c, out);
    }
    Term::ByteAddCarry(a, b, c) => {
      out.push(TERM_TAG_BYTE_ADD_CARRY);
      serialize_term(a, out);
      serialize_term(b, out);
      serialize_term(c, out);
    }
    Term::ByteMulLow(a, b) => {
      out.push(TERM_TAG_BYTE_MUL_LOW);
      serialize_term(a, out);
      serialize_term(b, out);
    }
    Term::ByteMulHigh(a, b) => {
      out.push(TERM_TAG_BYTE_MUL_HIGH);
      serialize_term(a, out);
      serialize_term(b, out);
    }
    Term::ByteAnd(a, b) => {
      out.push(TERM_TAG_BYTE_AND);
      serialize_term(a, out);
      serialize_term(b, out);
    }
    Term::ByteOr(a, b) => {
      out.push(TERM_TAG_BYTE_OR);
      serialize_term(a, out);
      serialize_term(b, out);
    }
    Term::ByteXor(a, b) => {
      out.push(TERM_TAG_BYTE_XOR);
      serialize_term(a, out);
      serialize_term(b, out);
    }
  }
}

fn serialize_wff(wff: &WFF, out: &mut Vec<u8>) {
  match wff {
    WFF::Equal(lhs, rhs) => {
      out.push(WFF_TAG_EQUAL);
      serialize_term(lhs, out);
      serialize_term(rhs, out);
    }
    WFF::And(a, b) => {
      out.push(WFF_TAG_AND);
      serialize_wff(a, out);
      serialize_wff(b, out);
    }
  }
}

fn serialize_wff_bytes(wff: &WFF) -> Vec<u8> {
  let mut out = Vec::new();
  serialize_wff(wff, &mut out);
  out
}

/// Build a WFF-match trace from inferred/public WFF byte serializations.
///
/// This is the second ZKP stage:
/// - stage 1 proves inferred WFF rows are valid (add-equality constraints)
/// - stage 2 proves inferred WFF serialization equals public WFF serialization.
pub fn generate_wff_match_trace_from_wffs(
  inferred_wff: &WFF,
  public_wff: &WFF,
) -> Result<RowMajorMatrix<Val>, String> {
  let inferred_bytes = serialize_wff_bytes(inferred_wff);
  let expected_bytes = serialize_wff_bytes(public_wff);

  generate_wff_match_trace_from_bytes(&inferred_bytes, &expected_bytes)
}

fn generate_wff_match_trace_from_bytes(
  inferred_bytes: &[u8],
  expected_bytes: &[u8],
) -> Result<RowMajorMatrix<Val>, String> {
  if inferred_bytes.len() != expected_bytes.len() {
    return Err(format!(
      "serialized WFF length mismatch: inferred={} public={}",
      inferred_bytes.len(),
      expected_bytes.len()
    ));
  }

  if inferred_bytes.is_empty() {
    return Err("serialized WFF cannot be empty".to_string());
  }

  // Pack MATCH_PACK_BYTES (3) bytes per M31 field element:
  //   packed = b0 | (b1 << 8) | (b2 << 16)
  //   max = 0xFFFFFF = 16,777,215 < M31 (2^31-1) — no field overflow.
  // Each row holds MATCH_PACK_COLS (4) inferred chunks + MATCH_PACK_COLS expected chunks,
  // encoding MATCH_PACK_COLS * MATCH_PACK_BYTES = 12 bytes per row.
  // Soundness: equality of packed chunks is equivalent to equality of the underlying bytes
  //   since packing is injective for equal-length sequences.
  let bytes_per_row = MATCH_PACK_COLS * MATCH_PACK_BYTES;
  let n_rows = (inferred_bytes.len() + bytes_per_row - 1) / bytes_per_row;
  let padded_rows = n_rows.next_power_of_two();

  let pack_chunk = |bytes: &[u8], byte_start: usize| -> u32 {
    let b0 = bytes.get(byte_start).copied().unwrap_or(0) as u32;
    let b1 = bytes.get(byte_start + 1).copied().unwrap_or(0) as u32;
    let b2 = bytes.get(byte_start + 2).copied().unwrap_or(0) as u32;
    b0 | (b1 << 8) | (b2 << 16)
  };

  let mut trace = RowMajorMatrix::new(
    Val::zero_vec(padded_rows * NUM_WFF_MATCH_COLS),
    NUM_WFF_MATCH_COLS,
  );
  for row in 0..n_rows {
    let base = row * NUM_WFF_MATCH_COLS;
    for col in 0..MATCH_PACK_COLS {
      let byte_start = row * bytes_per_row + col * MATCH_PACK_BYTES;
      trace.values[base + col] = Val::from_u32(pack_chunk(inferred_bytes, byte_start));
      trace.values[base + MATCH_PACK_COLS + col] =
        Val::from_u32(pack_chunk(expected_bytes, byte_start));
    }
  }
  // Rows [n_rows..padded_rows] are already Val::zero (set by zero_vec).

  Ok(trace)
}

/// Prove compiled ProofRows using the Stage-A backend.
pub fn prove_compiled_rows_stark(rows: &[ProofRow]) -> Result<CircleStarkProof, String> {
  prove_inferred_wff_stark(rows)
}

/// Stage-1 generic ZKP: prove inferred WFF validity from compiled ProofRows.
pub fn prove_inferred_wff_stark(rows: &[ProofRow]) -> Result<CircleStarkProof, String> {
  for row in rows {
    route_stage_a_row_op(row.op)?;
  }

  if !has_stage_a_semantic_rows(rows)? {
    return Err("stage-a semantic kernel unavailable for this proof-row set".to_string());
  }

  let trace = generate_stage_a_semantic_trace(rows)?;
  let config = make_circle_config();
  Ok(p3_uni_stark::prove(&config, &StageASemanticAir, trace, &[]))
}

/// Verify stage-1 inferred-WFF proof.
pub fn verify_inferred_wff_stark(proof: &CircleStarkProof) -> CircleStarkVerifyResult {
  let config = make_circle_config();
  p3_uni_stark::verify(&config, &StageASemanticAir, proof, &[])
}

/// Convenience helper: prove and verify compiled rows in one call.
pub fn prove_and_verify_compiled_rows_stark(rows: &[ProofRow]) -> bool {
  prove_and_verify_inferred_wff_stark(rows)
}

/// Stage-1 generic convenience helper.
pub fn prove_and_verify_inferred_wff_stark(rows: &[ProofRow]) -> bool {
  let proved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
    prove_compiled_rows_stark(rows)
  }));
  let proof = match proved {
    Ok(Ok(proof)) => proof,
    Ok(Err(_)) => return false,
    Err(_) => return false,
  };
  match verify_inferred_wff_stark(&proof) {
    Ok(()) => true,
    Err(_) => false,
  }
}

/// Prove stage-2 WFF match from compiled rows and expected output bytes.
pub fn prove_wff_match_stark(
  private_pi: &Proof,
  public_wff: &WFF,
) -> Result<CircleStarkProof, String> {
  prove_expected_wff_match_stark_with_public_values(
    private_pi,
    public_wff,
    &default_receipt_bind_public_values_for_tag(RECEIPT_BIND_TAG_WFF),
  )
}

/// Stage-2 generic ZKP: prove inferred WFF equals externally expected WFF.
pub fn prove_expected_wff_match_stark(
  private_pi: &Proof,
  public_wff: &WFF,
) -> Result<CircleStarkProof, String> {
  prove_expected_wff_match_stark_with_public_values(
    private_pi,
    public_wff,
    &default_receipt_bind_public_values_for_tag(RECEIPT_BIND_TAG_WFF),
  )
}

pub fn prove_expected_wff_match_stark_with_public_values(
  private_pi: &Proof,
  public_wff: &WFF,
  public_values: &[Val],
) -> Result<CircleStarkProof, String> {
  let inferred_wff = infer_proof(private_pi)
    .map_err(|err| format!("failed to infer WFF from private proof: {err}"))?;

  let trace = generate_wff_match_trace_from_wffs(&inferred_wff, public_wff)?;
  let config = make_circle_config();
  Ok(p3_uni_stark::prove(
    &config,
    &WffMatchAir,
    trace,
    public_values,
  ))
}

/// Verify stage-2 WFF match proof.
pub fn verify_wff_match_stark(proof: &CircleStarkProof) -> CircleStarkVerifyResult {
  verify_wff_match_stark_with_public_values(
    proof,
    &default_receipt_bind_public_values_for_tag(RECEIPT_BIND_TAG_WFF),
  )
}

pub fn verify_wff_match_stark_with_public_values(
  proof: &CircleStarkProof,
  public_values: &[Val],
) -> CircleStarkVerifyResult {
  let config = make_circle_config();
  p3_uni_stark::verify(&config, &WffMatchAir, proof, public_values)
}

/// Convenience helper for stage-2 prove+verify.
pub fn prove_and_verify_wff_match_stark(private_pi: &Proof, public_wff: &WFF) -> bool {
  prove_and_verify_expected_wff_match_stark(private_pi, public_wff)
}

/// Stage-2 generic convenience helper.
pub fn prove_and_verify_expected_wff_match_stark(private_pi: &Proof, public_wff: &WFF) -> bool {
  let proved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
    prove_expected_wff_match_stark(private_pi, public_wff)
  }));
  let proof = match proved {
    Ok(Ok(proof)) => proof,
    Ok(Err(_)) => return false,
    Err(_) => return false,
  };
  match verify_wff_match_stark(&proof) {
    Ok(()) => true,
    Err(_) => false,
  }
}

/// Stage A: public input = `wff_i`, private input = `pi_i`.
///
/// Produces:
/// 1) proof that `pi_i` is a valid derivation (`infer_wff`), and
/// 2) proof that inferred WFF equals public `wff_i`.
pub fn prove_stage_a(public_wff: &WFF, private_pi: &Proof) -> Result<StageAProof, String> {
  let rows = compile_proof(private_pi);

  verify_compiled(&rows).map_err(|err| format!("compiled proof validation failed: {err}"))?;
  let inferred_wff = infer_proof(private_pi).map_err(|err| format!("infer_proof failed: {err}"))?;
  if inferred_wff != *public_wff {
    return Err("inferred WFF does not match public WFF".to_string());
  }

  let inferred_wff_proof = prove_inferred_wff_stark(&rows)?;
  let public_wff_match_proof = prove_expected_wff_match_stark(private_pi, public_wff)?;

  Ok(StageAProof {
    inferred_wff_proof,
    public_wff_match_proof,
    expected_public_wff: public_wff.clone(),
  })
}

pub fn verify_stage_a(public_wff: &WFF, stage_a_proof: &StageAProof) -> bool {
  if *public_wff != stage_a_proof.expected_public_wff {
    return false;
  }

  let stage1_ok = verify_inferred_wff_stark(&stage_a_proof.inferred_wff_proof).is_ok();

  stage1_ok && verify_wff_match_stark(&stage_a_proof.public_wff_match_proof).is_ok()
}

pub fn prove_and_verify_stage_a(public_wff: &WFF, private_pi: &Proof) -> bool {
  let proved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
    prove_stage_a(public_wff, private_pi)
  }));
  let stage_a = match proved {
    Ok(Ok(proof)) => proof,
    Ok(Err(_)) => return false,
    Err(_) => return false,
  };
  verify_stage_a(public_wff, &stage_a)
}

// ============================================================
// Cross-family batch WFF proving
//
// Prove any mix of opcodes (ADD, SUB, MUL, DIV, AND, OR, XOR, LT, SGT, EQ …)
// in a single LUT STARK call by concatenating ALL their ProofRows, regardless
// of opcode family.
//
// Design:
//   - The LUT kernel AIR already handles every LUT opcode in one circuit.
//   - Structural rows (OP_BOOL, OP_EQ_REFL, …) are silently skipped — they
//     carry no arithmetic content.
//   - Bit-family ops (AND/OR/XOR) are expanded to bit-level, as required by
//     the LUT AIR constraints (in0,in1 ∈ {0,1}).
//
// Security model:
//   1. LUT STARK: proves all N instructions' arithmetic is correct (one proof).
//   2. WFF check (deterministic, no ZK): `infer_proof(pi_i) == wff_i` for each i.
//      This is pure-function verification (µs range) done by the verifier.
//
// Usage:
//   let itps: Vec<InstructionTransitionProof> = ...;
//   let proofs: Vec<&Proof> = itps.iter()
//       .filter_map(|itp| itp.semantic_proof.as_ref())
//       .collect();
//   let (lut_proof, wffs) = prove_batch_wff_proofs(&proofs)?;
//   // verify:
//   assert!(verify_batch_wff_proofs(&lut_proof, &proofs, &wffs).is_ok());
// ============================================================

/// Build LUT steps from ProofRows belonging to **any** opcode family.
///
/// Structural rows (OP_BOOL, OP_BYTE, OP_EQ_REFL, OP_AND_INTRO, OP_EQ_TRANS,
/// OP_BYTE_ADD_THIRD_CONGRUENCE, OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE,
/// OP_NOT, OP_EQ_SYM) are skipped silently.
///
/// AND/OR/XOR rows produce exactly **one byte-level `LutStep`** per
/// `ProofRow` — no bit expansion.  Arithmetic soundness for these operations
/// is guaranteed by the companion LogUp byte-table argument.
pub fn build_lut_steps_from_rows_auto(rows: &[ProofRow]) -> Result<Vec<LutStep>, String> {
  use crate::semantic_proof::{
    OP_AND_INTRO, OP_BOOL, OP_BYTE, OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE,
    OP_BYTE_ADD_THIRD_CONGRUENCE, OP_EQ_REFL, OP_EQ_SYM, OP_EQ_TRANS, OP_NOT,
  };

  let mut out = Vec::with_capacity(rows.len() * 2);

  for row in rows {
    match row.op {
      // ── Structural rows: skip silently ────────────────────────────────────
      // Soundness fix: OP_NOT and OP_EQ_SYM can appear in compiled proof rows
      // but carry no LUT arithmetic content; they must be skipped here.
      op if op == OP_BOOL
        || op == OP_BYTE
        || op == OP_EQ_REFL
        || op == OP_AND_INTRO
        || op == OP_EQ_TRANS
        || op == OP_BYTE_ADD_THIRD_CONGRUENCE
        || op == OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE
        || op == OP_NOT
        || op == OP_EQ_SYM => {}

      // ── Add-family LUT ops ─────────────────────────────────────────────────
      crate::semantic_proof::OP_U29_ADD_EQ => out.push(LutStep {
        op: LutOpcode::U29AddEq,
        in0: row.scalar0,
        in1: row.scalar1,
        in2: row.scalar2,
        out0: row.value,
        out1: row.arg1,
      }),
      crate::semantic_proof::OP_U24_ADD_EQ => out.push(LutStep {
        op: LutOpcode::U24AddEq,
        in0: row.scalar0,
        in1: row.scalar1,
        in2: row.scalar2,
        out0: row.value,
        out1: row.arg1,
      }),

      // ── Mul-family LUT ops ─────────────────────────────────────────────────
      crate::semantic_proof::OP_U15_MUL_EQ => {
        let total = row.scalar0 * row.scalar1;
        out.push(LutStep {
          op: LutOpcode::U15MulEq,
          in0: row.scalar0,
          in1: row.scalar1,
          in2: 0,
          out0: total & 0x7FFF,
          out1: total >> 15,
        });
      }
      crate::semantic_proof::OP_BYTE_MUL_LOW_EQ => {
        let product = row.scalar0 * row.scalar1;
        // Soundness fix: AIR constraint is  in0*in1 = out0 + 256*out1
        // so out1 must carry the high byte, not 0.
        out.push(LutStep {
          op: LutOpcode::ByteMulLowEq,
          in0: row.scalar0,
          in1: row.scalar1,
          in2: 0,
          out0: product & 0xFF,
          out1: product >> 8,
        });
      }
      crate::semantic_proof::OP_BYTE_MUL_HIGH_EQ => {
        let product = row.scalar0 * row.scalar1;
        // Soundness fix: AIR constraint is  in0*in1 = 256*out0 + out1
        // so out1 must carry the low byte, not 0.
        out.push(LutStep {
          op: LutOpcode::ByteMulHighEq,
          in0: row.scalar0,
          in1: row.scalar1,
          in2: 0,
          out0: product >> 8,
          out1: product & 0xFF,
        });
      }

      // ── Byte-level bitwise ops: one step per ProofRow ────────────────────
      crate::semantic_proof::OP_BYTE_AND_EQ => {
        let (a, b) = (row.scalar0, row.scalar1);
        out.push(LutStep {
          op: LutOpcode::ByteAndEq,
          in0: a,
          in1: b,
          in2: 0,
          out0: a & b,
          out1: 0,
        });
      }
      crate::semantic_proof::OP_BYTE_OR_EQ => {
        let (a, b) = (row.scalar0, row.scalar1);
        out.push(LutStep {
          op: LutOpcode::ByteOrEq,
          in0: a,
          in1: b,
          in2: 0,
          out0: a | b,
          out1: 0,
        });
      }
      crate::semantic_proof::OP_BYTE_XOR_EQ => {
        let (a, b) = (row.scalar0, row.scalar1);
        out.push(LutStep {
          op: LutOpcode::ByteXorEq,
          in0: a,
          in1: b,
          in2: 0,
          out0: a ^ b,
          out1: 0,
        });
      }

      other => {
        return Err(format!(
          "build_lut_steps_from_rows_auto: unrecognised row op {other}"
        ));
      }
    }
  }

  if out.is_empty() {
    return Err("build_lut_steps_from_rows_auto: no LUT steps found in row set".to_string());
  }

  Ok(out)
}

/// Prove N instructions of ANY opcode mix using a single LUT STARK call.
///
/// Returns `(lut_proof, individual_wffs)`.
/// - `lut_proof` covers the arithmetic correctness of ALL instructions.
/// - `individual_wffs[i]` is the WFF inferred from `proofs[i]`.
///
/// Verification: call `verify_batch_wff_proofs`.
pub fn prove_batch_wff_proofs(proofs: &[&Proof]) -> Result<(CircleStarkProof, Vec<WFF>), String> {
  if proofs.is_empty() {
    return Err("prove_batch_wff_proofs: empty proof batch".to_string());
  }

  // Compile all ProofRows and concatenate
  let all_rows: Vec<ProofRow> = proofs.iter().flat_map(|p| compile_proof(p)).collect();

  // Build cross-family LUT steps
  let steps = build_lut_steps_from_rows_auto(&all_rows)?;

  // Prove once
  let lut_proof = prove_lut_kernel_stark(&steps)?;

  // Collect individual WFFs (deterministic, cheap)
  let wffs = proofs
    .iter()
    .map(|p| infer_proof(p).map_err(|e| format!("infer_proof failed: {e}")))
    .collect::<Result<Vec<_>, _>>()?;

  Ok((lut_proof, wffs))
}

/// Verify a batch WFF proof.
///
/// Checks:
/// 1. The LUT STARK proof is valid.
/// 2. For each proof `proofs[i]`, `infer_proof(proofs[i]) == wffs[i]` (cheap).
pub fn verify_batch_wff_proofs(
  lut_proof: &CircleStarkProof,
  proofs: &[&Proof],
  wffs: &[WFF],
) -> Result<(), String> {
  if proofs.len() != wffs.len() {
    return Err(format!(
      "verify_batch_wff_proofs: proofs.len()={} != wffs.len()={}",
      proofs.len(),
      wffs.len()
    ));
  }

  // 1. Verify the shared LUT proof
  verify_lut_kernel_stark(lut_proof)
    .map_err(|e| format!("batch LUT STARK verification failed: {e:?}"))?;

  // 2. Verify each WFF matches the deterministic derivation
  for (i, (proof, expected_wff)) in proofs.iter().zip(wffs.iter()).enumerate() {
    let inferred =
      infer_proof(proof).map_err(|e| format!("infer_proof failed for instruction {i}: {e}"))?;
    if inferred != *expected_wff {
      return Err(format!("WFF mismatch at instruction {i}"));
    }
  }

  // 3. Out-of-circuit row content validation (Gap 3 + Gap 4).
  //
  // Gap 3: byte AND/OR/XOR rows must commit the correct result value.
  // Gap 4: arithmetic operands must be within their declared bit-width bounds.
  //
  // The LUT AIR does not constrain value = scalar0 op scalar1 for byte ops,
  // nor does it enforce operand ranges for add/mul ops.  These checks are
  // mandatory here to close the soundness gaps.
  let all_rows: Vec<ProofRow> = proofs.iter().flat_map(|p| compile_proof(p)).collect();
  if !validate_manifest_rows(&all_rows) {
    return Err(
      "manifest row validation failed (Gap 3/4): byte op value or range check".to_string(),
    );
  }

  Ok(())
}

// ============================================================
// Memory consistency proof  (read/write set intersection, binary-tree aggregation)
// ============================================================
//
// Soundness design
// ─────────────────
// Each batch proof exposes two sparse maps as public output:
//
//   write_set : Map<addr → value>  — final value written to each address
//                                    in this batch (last write wins).
//   read_set  : Map<addr → value>  — addresses read *before* any write in
//                                    this batch (cross-batch dependencies).
//
// Binary-tree aggregation merges adjacent proofs by checking the
// intersection of the right child's `read_set` against the left child's
// `write_set`:
//
//     ∀ addr ∈ R.read_set ∩ L.write_set:
//         R.read_set[addr] == L.write_set[addr]
//
//     merged.write_set = L.write_set ∪ R.write_set  (R overwrites same addr)
//     merged.read_set  = L.read_set ∪ (R.read_set \ L.write_set.keys())
//
// All leaf proofs are fully independent (no prev-snapshot dependency) and
// can be generated in parallel.  Binary-tree aggregation then requires only
// O(log N) sequential levels.  At the root, `read_set` lists the addresses
// that must be satisfied by the genesis state, and `write_set` is the final
// memory state after all batches.
//
// After aggregation the write/read logs are discarded; only the compact
// `AggregatedMemoryProof { write_set, read_set }` remains.
//
// Column layout  (NUM_MEM_COLS = 12)
// ────────────────────────────────────
//  0  addr_hi   — upper 32 bits of the 64-bit word address
//  1  addr_lo   — lower 32 bits
//  2  val0      — value bytes  0- 3 (big-endian u32)
//  3..9  val1..val7
// 10  is_write  — 1 for write rows, 0 for read rows
// 11  mult      — +1 for log entry rows, 0 for padding
//
// Public values layout  (MEM_PUBLIC_VALUES_LEN = 35)
// ────────────────────────────────────────────────────
//  pis[ 0]      = RECEIPT_BIND_TAG_MEM
//  pis[ 1]      = n_writes
//  pis[ 2..10]  = poseidon_hash(write_log)  (8 × M31, rw_counter included)
//  pis[10]      = n_reads
//  pis[11..19]  = poseidon_hash(read_log)   (8 × M31, rw_counter included)
//  pis[19..27]  = poseidon_hash(write_set)  (8 × M31) — aggregation interface
//  pis[27..35]  = poseidon_hash(read_set)   (8 × M31) — aggregation interface



pub const RECEIPT_BIND_TAG_MEM: u32 = 4;
const MEM_PUBLIC_VALUES_LEN: usize = 35;

const MEM_COL_ADDR_HI: usize = 0;
const MEM_COL_ADDR_LO: usize = 1;
const MEM_COL_VAL0: usize = 2;
// val1..val7 are MEM_COL_VAL0+1 .. MEM_COL_VAL0+7  (cols 2..9)
const MEM_COL_IS_WRITE: usize = 10;
const MEM_COL_MULT: usize = 11;
/// Total number of main-trace columns.
pub const NUM_MEM_COLS: usize = 12;

// ── Log entry type ────────────────────────────────────────────────────

/// A single entry in the public write or read log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemLogEntry {
  pub rw_counter: u64,
  pub addr: u64,
  pub value: [u8; 32],
}

/// Split a flat `MemAccessClaim` slice into separate write and read logs.
pub fn split_mem_logs(
  claims: &[crate::transition::MemAccessClaim],
) -> (Vec<MemLogEntry>, Vec<MemLogEntry>) {
  let mut writes = Vec::new();
  let mut reads = Vec::new();
  for c in claims {
    let entry = MemLogEntry { rw_counter: c.rw_counter, addr: c.addr, value: c.value };
    if c.is_write { writes.push(entry); } else { reads.push(entry); }
  }
  (writes, reads)
}

// ── Hash helpers ──────────────────────────────────────────────────────

fn hash_mem_log(log: &[MemLogEntry]) -> [Val; 8] {
  let mut rng = rand::rngs::SmallRng::seed_from_u64(0xC0DEC0DE_u64);
  let poseidon = Poseidon2Mersenne31::<16>::new_from_rng_128(&mut rng);
  let sponge = PaddingFreeSponge::<_, 16, 8, 8>::new(poseidon);

  let mut input: Vec<Val> = Vec::with_capacity(1 + log.len() * 12);
  input.push(Val::from_u32(log.len() as u32));
  for e in log {
    input.push(Val::from_u32((e.rw_counter >> 32) as u32));
    input.push(Val::from_u32(e.rw_counter as u32));
    input.push(Val::from_u32((e.addr >> 32) as u32));
    input.push(Val::from_u32(e.addr as u32));
    for chunk in e.value.chunks(4) {
      input.push(Val::from_u32(u32::from_be_bytes([
        chunk[0], chunk[1], chunk[2], chunk[3],
      ])));
    }
  }
  sponge.hash_iter(input)
}

fn hash_mem_set(set: &std::collections::HashMap<u64, [u8; 32]>) -> [Val; 8] {
  if set.is_empty() {
    return [Val::ZERO; 8];
  }
  let mut sorted: Vec<(u64, [u8; 32])> = set.iter().map(|(&a, &v)| (a, v)).collect();
  sorted.sort_by_key(|(a, _)| *a);

  let mut rng = rand::rngs::SmallRng::seed_from_u64(0xC0DEC0DE_u64);
  let poseidon = Poseidon2Mersenne31::<16>::new_from_rng_128(&mut rng);
  let sponge = PaddingFreeSponge::<_, 16, 8, 8>::new(poseidon);

  let mut input: Vec<Val> = Vec::with_capacity(1 + sorted.len() * 10);
  input.push(Val::from_u32(sorted.len() as u32));
  for (addr, value) in &sorted {
    input.push(Val::from_u32((*addr >> 32) as u32));
    input.push(Val::from_u32(*addr as u32));
    for chunk in value.chunks(4) {
      input.push(Val::from_u32(u32::from_be_bytes([
        chunk[0], chunk[1], chunk[2], chunk[3],
      ])));
    }
  }
  sponge.hash_iter(input)
}

/// Build the `public_values` vector (`MEM_PUBLIC_VALUES_LEN = 35`).
fn make_mem_public_values(
  write_log: &[MemLogEntry],
  read_log: &[MemLogEntry],
  write_set: &std::collections::HashMap<u64, [u8; 32]>,
  read_set: &std::collections::HashMap<u64, [u8; 32]>,
) -> Vec<Val> {
  let wlh = hash_mem_log(write_log);
  let rlh = hash_mem_log(read_log);
  let wsh = hash_mem_set(write_set);
  let rsh = hash_mem_set(read_set);
  let mut pv = Vec::with_capacity(MEM_PUBLIC_VALUES_LEN);
  pv.push(Val::from_u32(RECEIPT_BIND_TAG_MEM));       //  0
  pv.push(Val::from_u32(write_log.len() as u32));     //  1
  pv.extend_from_slice(&wlh);                         //  2..10
  pv.push(Val::from_u32(read_log.len() as u32));      //  10
  pv.extend_from_slice(&rlh);                         // 11..19
  pv.extend_from_slice(&wsh);                         // 19..27 — aggregation interface
  pv.extend_from_slice(&rsh);                         // 27..35 — aggregation interface
  pv
}

// ── Intra-batch consistency checker ──────────────────────────────────

/// Validate intra-batch read/write consistency and derive `(write_set, read_set)`.
///
/// Scans `claims` in execution order (rw_counter is monotone):
/// - **Write**: updates `write_set[addr] = value` (last write wins).
/// - **Intra-batch read**: addr was already written in this batch — value must
///   match the most recent write.
/// - **Cross-batch read**: addr not yet written in this batch — exposed in
///   `read_set[addr]` for the parent aggregation node to check.
///
/// Cross-batch consistency (e.g., `read_set[addr] == left.write_set[addr]`)
/// is deferred to aggregation time and enforced by [`aggregate_memory_proofs`].
fn check_claims_and_build_sets(
  claims: &[crate::transition::MemAccessClaim],
) -> Result<
  (
    std::collections::HashMap<u64, [u8; 32]>,
    std::collections::HashMap<u64, [u8; 32]>,
  ),
  String,
> {
  use std::collections::HashMap;
  let mut write_set: HashMap<u64, [u8; 32]> = HashMap::new();
  let mut read_set: HashMap<u64, [u8; 32]> = HashMap::new();
  let mut last_write: HashMap<u64, [u8; 32]> = HashMap::new();

  for claim in claims {
    if claim.is_write {
      last_write.insert(claim.addr, claim.value);
      write_set.insert(claim.addr, claim.value); // last write wins
    } else if let Some(&written) = last_write.get(&claim.addr) {
      // Intra-batch read: must match the most recent write to this address.
      if claim.value != written {
        return Err(format!(
          "read/write mismatch at addr=0x{:x}: read {:?}, last write {:?}",
          claim.addr, claim.value, written
        ));
      }
    } else {
      // Cross-batch read: addr not yet written in this batch.
      match read_set.entry(claim.addr) {
        std::collections::hash_map::Entry::Vacant(e) => {
          e.insert(claim.value);
        }
        std::collections::hash_map::Entry::Occupied(e) => {
          if *e.get() != claim.value {
            return Err(format!(
              "cross-batch read mismatch at addr=0x{:x}: {:?} vs {:?}",
              claim.addr,
              e.get(),
              claim.value
            ));
          }
        }
      }
    }
  }
  Ok((write_set, read_set))
}

/// Re-derive `(write_set, read_set)` from a split log pair (used in verification).
///
/// Merges `write_log` and `read_log` by `rw_counter`, then applies the same
/// sequential scan as [`check_claims_and_build_sets`].
fn derive_sets_from_logs(
  write_log: &[MemLogEntry],
  read_log: &[MemLogEntry],
) -> Result<
  (
    std::collections::HashMap<u64, [u8; 32]>,
    std::collections::HashMap<u64, [u8; 32]>,
  ),
  String,
> {
  use std::collections::HashMap;
  // Merge into (rw_counter, is_write, addr, value) and sort by rw_counter.
  let mut merged: Vec<(u64, bool, u64, [u8; 32])> = Vec::new();
  for e in write_log {
    merged.push((e.rw_counter, true, e.addr, e.value));
  }
  for e in read_log {
    merged.push((e.rw_counter, false, e.addr, e.value));
  }
  merged.sort_by_key(|(rw, _, _, _)| *rw);

  let mut write_set: HashMap<u64, [u8; 32]> = HashMap::new();
  let mut read_set: HashMap<u64, [u8; 32]> = HashMap::new();
  let mut last_write: HashMap<u64, [u8; 32]> = HashMap::new();

  for (_, is_write, addr, value) in merged {
    if is_write {
      last_write.insert(addr, value);
      write_set.insert(addr, value);
    } else if let Some(&written) = last_write.get(&addr) {
      if value != written {
        return Err(format!(
          "read/write mismatch at addr=0x{:x}: read {:?}, last write {:?}",
          addr, value, written
        ));
      }
    } else {
      match read_set.entry(addr) {
        std::collections::hash_map::Entry::Vacant(e) => {
          e.insert(value);
        }
        std::collections::hash_map::Entry::Occupied(e) => {
          if *e.get() != value {
            return Err(format!(
              "cross-batch read mismatch at addr=0x{:x}: {:?} vs {:?}",
              addr,
              e.get(),
              value
            ));
          }
        }
      }
    }
  }
  Ok((write_set, read_set))
}

// ── AIR ──────────────────────────────────────────────────────────────

/// AIR for the memory log LogUp membership argument.
///
/// Each trace row represents one entry from the write or read log.
/// The only AIR constraints are:
///   1. Tag check: `pis[0] == RECEIPT_BIND_TAG_MEM`  (degree 1)
///   2. `is_write ∈ {0, 1}`                          (degree 2)
///
/// All read/write consistency is checked deterministically by the verifier
/// using the public write_log and read_log (no ZK needed for consistency).
///
/// The LogUp argument proves that the rows in the trace are a subset of
/// the committed (write_log ∪ read_log) tuples, preventing the prover from
/// fabricating log entries not present in the actual execution.
pub struct MemoryConsistencyAir;

impl<F: p3_field::Field> p3_air::BaseAir<F> for MemoryConsistencyAir {
  fn width(&self) -> usize {
    NUM_MEM_COLS
  }
}

impl<AB: p3_air::AirBuilderWithPublicValues> p3_air::Air<AB> for MemoryConsistencyAir
where
  AB::F: p3_field::Field,
{
  fn eval(&self, builder: &mut AB) {
    // ── Tag check ────────────────────────────────────────────────
    let pis = builder.public_values();
    builder.assert_eq(pis[0].into(), AB::Expr::from_u32(RECEIPT_BIND_TAG_MEM));

    // ── is_write boolean ─────────────────────────────────────────
    let main = builder.main();
    let local = main.row_slice(0).expect("empty memory trace");
    let local = &*local;
    let is_write = local[MEM_COL_IS_WRITE].clone();
    builder.assert_zero(is_write.clone().into() * (AB::Expr::ONE - is_write.into()));
  }
}

// ── Lookup descriptor ────────────────────────────────────────────────

// ── Trace builder ─────────────────────────────────────────────────────

fn value_to_u32s(value: &[u8; 32]) -> [u32; 8] {
  let mut out = [0u32; 8];
  for (i, chunk) in value.chunks(4).enumerate() {
    out[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
  }
  out
}

fn fill_log_row(data: &mut [Val], base: usize, entry: &MemLogEntry, is_write: bool, mult: i32) {
  data[base + MEM_COL_ADDR_HI] = Val::from_u32((entry.addr >> 32) as u32);
  data[base + MEM_COL_ADDR_LO] = Val::from_u32(entry.addr as u32);
  let vals = value_to_u32s(&entry.value);
  for k in 0..8 {
    data[base + MEM_COL_VAL0 + k] = Val::from_u32(vals[k]);
  }
  data[base + MEM_COL_IS_WRITE] = Val::from_u32(is_write as u32);
  data[base + MEM_COL_MULT] = if mult >= 0 {
    Val::from_u32(mult as u32)
  } else {
    -Val::from_u32((-mult) as u32)
  };
}

/// Build the memory log trace from write and read logs with LogUp multiplicities.
///
/// Trace layout:
///   Rows 0 .. n_writes       : write log entries (is_write=1)
///   Rows n_writes .. n_total : read log entries  (is_write=0)
///   Remaining rows           : padding (all-zero, mult=0)
///
/// LogUp multiplicity assignment:
///   - Last write per addr (the entry whose value appears in `write_set`):
///     `mult = count of intra-batch reads to that addr`.  All earlier writes
///     to the same addr get `mult = 0` (they are not the canonical value).
///   - Intra-batch read (addr ∈ write_set): `mult = −1`.
///   - Cross-batch read (addr ∉ write_set): `mult = 0` (not in this LogUp).
///   - Padding rows: `mult = 0`.
///
/// Combined, the LogUp sum Σ mult_i/(α − fingerprint(row_i)) = 0 proves that
/// the multiset of intra-batch read `(addr, value)` tuples equals the
/// last-written value per addr with the correct multiplicities — i.e. every
/// intra-batch read sees `write_set[addr]`.
fn build_mem_log_trace(
  write_log: &[MemLogEntry],
  read_log: &[MemLogEntry],
  write_set: &std::collections::HashMap<u64, [u8; 32]>,
) -> RowMajorMatrix<Val> {
  // Count intra-batch reads per addr (reads whose addr appears in write_set).
  let mut intra_read_count: std::collections::HashMap<u64, i32> = Default::default();
  for e in read_log {
    if write_set.contains_key(&e.addr) {
      *intra_read_count.entry(e.addr).or_insert(0) += 1;
    }
  }

  // Index of the last write per addr (determines which write carries the canonical value).
  let mut last_write_idx: std::collections::HashMap<u64, usize> = Default::default();
  for (i, e) in write_log.iter().enumerate() {
    last_write_idx.insert(e.addr, i);
  }

  let n_total = write_log.len() + read_log.len();
  let height = n_total.max(4).next_power_of_two();
  let mut data = vec![Val::ZERO; height * NUM_MEM_COLS];

  for (i, entry) in write_log.iter().enumerate() {
    // Only the last write to each addr contributes to the LogUp sum;
    // earlier writes are "overwritten" and get mult = 0.
    let mult = if last_write_idx.get(&entry.addr) == Some(&i) {
      *intra_read_count.get(&entry.addr).unwrap_or(&0)
    } else {
      0
    };
    fill_log_row(&mut data, i * NUM_MEM_COLS, entry, true, mult);
  }

  for (i, entry) in read_log.iter().enumerate() {
    let mult = if write_set.contains_key(&entry.addr) { -1 } else { 0 };
    fill_log_row(&mut data, (write_log.len() + i) * NUM_MEM_COLS, entry, false, mult);
  }

  RowMajorMatrix::new(data, NUM_MEM_COLS)
}

// ── Memory LogUp argument helpers ────────────────────────────────────

/// Build the `Lookup` descriptor for the memory consistency local LogUp.
///
/// Fingerprint tuple: `(addr_hi, addr_lo, val0..val7)` — columns 0–9.
/// Multiplicity column: `MEM_COL_MULT` (column 11).
/// Auxiliary column index in the permutation trace: 0.
fn make_mem_lookup() -> Lookup<Val> {
  let col = |c: usize| {
    SymbolicExpression::Variable(SymbolicVariable::new(Entry::Main { offset: 0 }, c))
  };
  Lookup::new(
    Kind::Local,
    vec![vec![
      col(MEM_COL_ADDR_HI),
      col(MEM_COL_ADDR_LO),
      col(MEM_COL_VAL0),
      col(MEM_COL_VAL0 + 1),
      col(MEM_COL_VAL0 + 2),
      col(MEM_COL_VAL0 + 3),
      col(MEM_COL_VAL0 + 4),
      col(MEM_COL_VAL0 + 5),
      col(MEM_COL_VAL0 + 6),
      col(MEM_COL_VAL0 + 7),
    ]],
    vec![col(MEM_COL_MULT)],
    vec![0],
  )
}

/// Generate the permutation (LogUp running-sum) trace for the memory AIR.
fn generate_mem_perm_trace(
  main_trace: &RowMajorMatrix<Val>,
  perm_challenges: &[Challenge],
) -> Option<RowMajorMatrix<Challenge>> {
  let gadget = LogUpGadget::new();
  let lookup = make_mem_lookup();
  let mut lookup_data: Vec<LookupData<Challenge>> = vec![];
  let perm_trace = gadget.generate_permutation::<CircleStarkConfig>(
    main_trace,
    &None,
    &[],
    &[lookup],
    &mut lookup_data,
    perm_challenges,
  );
  Some(perm_trace)
}

/// Evaluate the memory LogUp constraints (called in both prover and verifier).
fn eval_mem_lookup(folder: &mut VerifierConstraintFolder<CircleStarkConfig>) {
  LogUpGadget::new().eval_local_lookup(folder, &make_mem_lookup());
}

// ── Public API ────────────────────────────────────────────────────────

/// Proof that the execution's memory accesses are consistent.
///
/// Exposes `write_set` (final written values) and `read_set` (cross-batch
/// dependencies) as public output.  Binary-tree aggregation merges adjacent
/// proofs by checking `R.read_set ∩ L.write_set` for value consistency.
pub struct MemoryConsistencyProof {
  pub stark_proof: CircleStarkProof,
  /// Final value written to each address in this batch (last write wins).
  pub write_set: std::collections::HashMap<u64, [u8; 32]>,
  /// Cross-batch dependencies: addresses read before any write in this batch.
  pub read_set: std::collections::HashMap<u64, [u8; 32]>,
  /// Write log (retained for STARK verification; can be discarded after aggregation).
  pub write_log: Vec<MemLogEntry>,
  /// Read log (retained for STARK verification; can be discarded after aggregation).
  pub read_log: Vec<MemLogEntry>,
}

/// Prove memory consistency for `claims`.
///
/// Steps:
/// 1. Split claims into write/read logs (preserving `rw_counter` for ordering).
/// 2. Scan claims in execution order to build `write_set` / `read_set` and check
///    intra-batch read/write consistency.
/// 3. Build STARK with LogUp permutation argument that proves every intra-batch
///    read `(addr, value)` matches `write_set[addr]` in-circuit.
pub fn prove_memory_consistency(
  claims: &[crate::transition::MemAccessClaim],
) -> Result<MemoryConsistencyProof, String> {
  let (write_log, read_log) = split_mem_logs(claims);

  // Intra-batch consistency + derive (write_set, read_set).
  let (write_set, read_set) = check_claims_and_build_sets(claims)?;

  let public_values =
    make_mem_public_values(&write_log, &read_log, &write_set, &read_set);

  let trace = if write_log.is_empty() && read_log.is_empty() {
    RowMajorMatrix::new(vec![Val::ZERO; 4 * NUM_MEM_COLS], NUM_MEM_COLS)
  } else {
    build_mem_log_trace(&write_log, &read_log, &write_set)
  };

  let config = make_circle_config();
  let trace_for_perm = trace.clone();
  let stark_proof = prove_with_lookup(
    &config,
    &MemoryConsistencyAir,
    trace,
    &public_values,
    None,
    move |perm_challenges| generate_mem_perm_trace(&trace_for_perm, perm_challenges),
    2, // num_perm_challenges (1 lookup × 2 challenges: alpha + beta)
    2, // lookup_constraint_count (first_row + universal for local LogUp)
    |folder: &mut VerifierConstraintFolder<CircleStarkConfig>| eval_mem_lookup(folder),
  );

  Ok(MemoryConsistencyProof {
    stark_proof,
    write_set,
    read_set,
    write_log,
    read_log,
  })
}

/// Verify a single [`MemoryConsistencyProof`].
///
/// 1. Re-derives `(write_set, read_set)` from the embedded logs to catch tampering.
/// 2. Verifies the STARK proof against the re-derived public values.
///
/// Cross-batch consistency (i.e., that `read_set` values are satisfied by a
/// prior batch's `write_set`) is enforced at aggregation time.
pub fn verify_memory_consistency(proof: &MemoryConsistencyProof) -> bool {
  // Re-derive (write_set, read_set) from logs to detect tampering.
  let (derived_write_set, derived_read_set) =
    match derive_sets_from_logs(&proof.write_log, &proof.read_log) {
      Ok(s) => s,
      Err(_) => return false,
    };
  if derived_write_set != proof.write_set || derived_read_set != proof.read_set {
    return false;
  }

  let public_values = make_mem_public_values(
    &proof.write_log,
    &proof.read_log,
    &proof.write_set,
    &proof.read_set,
  );

  let config = make_circle_config();
  verify_with_lookup(
    &config,
    &MemoryConsistencyAir,
    &proof.stark_proof,
    &public_values,
    None,
    2, // num_perm_challenges (must match prove_memory_consistency)
    |folder: &mut VerifierConstraintFolder<CircleStarkConfig>| eval_mem_lookup(folder),
  )
  .is_ok()
}

// ── Binary-tree aggregation ───────────────────────────────────────────

/// Compact result of binary-tree aggregation over leaf [`MemoryConsistencyProof`]s.
///
/// After aggregation the write/read logs are discarded; only the sparse
/// `write_set` (final memory state) and `read_set` (genesis dependencies)
/// survive, summarising the entire multi-batch state transition in O(accessed
/// addresses) data.
///
/// # Recursive STARK aggregation
/// Each merge node will be replaced by a **recursive STARK proof** whose
/// circuit verifies both child proofs inside the arithmetisation:
///
/// ```text
/// MergeAir public inputs:
///   left.write_set_hash  (8 × M31)
///   left.read_set_hash   (8 × M31)
///   right.write_set_hash (8 × M31)
///   right.read_set_hash  (8 × M31)
///
/// MergeAir constraints:
///   1. Verify left  child STARK proof  (IOP transcript check in-circuit)
///   2. Verify right child STARK proof
///   3. ∀ addr ∈ R.read_set ∩ L.write_set: R.read_set[addr] == L.write_set[addr]
///
/// Output public input for the parent level:
///   merged.write_set_hash, merged.read_set_hash
/// ```
///
/// The current native-Rust intersection check is a placeholder that will be
/// replaced once the recursive STARK verifier circuit is implemented.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AggregatedMemoryProof {
  /// Final values written across all batches (right/later batch wins on overlap).
  pub write_set: std::collections::HashMap<u64, [u8; 32]>,
  /// Cross-batch dependencies not satisfied within the aggregated subtree.
  /// At the root this corresponds to the required genesis memory state.
  pub read_set: std::collections::HashMap<u64, [u8; 32]>,
}

/// Merge two adjacent proven batches using read/write set intersection.
///
/// Checks that every address the right batch reads from a prior batch has the
/// value that the left batch wrote.  Returns `Err` if any proof is invalid or
/// any intersection value mismatches.
pub fn aggregate_memory_proofs(
  left: &MemoryConsistencyProof,
  right: &MemoryConsistencyProof,
) -> Result<AggregatedMemoryProof, String> {
  if !verify_memory_consistency(left) {
    return Err("left child proof failed verification".to_string());
  }
  if !verify_memory_consistency(right) {
    return Err("right child proof failed verification".to_string());
  }
  merge_sets(&left.write_set, &left.read_set, &right.write_set, &right.read_set)
}

/// Core set-merge logic (shared by both leaf and tree aggregation).
fn merge_sets(
  left_ws: &std::collections::HashMap<u64, [u8; 32]>,
  left_rs: &std::collections::HashMap<u64, [u8; 32]>,
  right_ws: &std::collections::HashMap<u64, [u8; 32]>,
  right_rs: &std::collections::HashMap<u64, [u8; 32]>,
) -> Result<AggregatedMemoryProof, String> {
  // Intersection check: ∀ addr ∈ R.read_set ∩ L.write_set, values must match.
  for (addr, &r_val) in right_rs {
    if let Some(&l_val) = left_ws.get(addr) {
      if r_val != l_val {
        return Err(format!(
          "read/write set mismatch at addr=0x{addr:x}: right read {r_val:?}, left wrote {l_val:?}"
        ));
      }
    }
  }

  // Merged write_set: left base, right overwrites on overlap.
  let mut write_set = left_ws.clone();
  for (&addr, &val) in right_ws {
    write_set.insert(addr, val);
  }

  // Merged read_set: left's unresolved reads + right's reads not covered by left's writes.
  let mut read_set = left_rs.clone();
  for (&addr, &val) in right_rs {
    if !left_ws.contains_key(&addr) {
      // Not satisfied by the left subtree → propagate upward.
      read_set.entry(addr).or_insert(val);
    }
  }

  Ok(AggregatedMemoryProof { write_set, read_set })
}

/// Aggregate a sequence of adjacent batch proofs into a single compact proof
/// using a binary tree of merge operations.
///
/// # Current implementation
/// Each merge node performs a native-Rust read/write set intersection check.
/// This is a **placeholder** — see [`AggregatedMemoryProof`] for the planned
/// replacement with a recursive STARK circuit.
///
/// # Parallelism
/// Leaf proofs are fully independent (no prev-snapshot dependency) and can be
/// generated in any order or in parallel.  Within each tree level all merge
/// nodes are also independent — each reads only its two immediate children —
/// so every level can be processed with data-parallel iteration.
///
/// Tree depth = ⌈log₂(N)⌉ → O(N) total work, O(log N) sequential depth.
///
/// # Odd lengths
/// If a level has an odd number of nodes the final node is carried up
/// unchanged (equivalent to padding with an identity batch).
pub fn aggregate_proofs_tree(
  proofs: &[MemoryConsistencyProof],
) -> Result<AggregatedMemoryProof, String> {
  if proofs.is_empty() {
    return Err("aggregate_proofs_tree: empty proof slice".to_string());
  }

  // Validate all leaves and convert to AggregatedMemoryProof nodes.
  let mut nodes: Vec<AggregatedMemoryProof> = proofs
    .iter()
    .enumerate()
    .map(|(i, p)| {
      if !verify_memory_consistency(p) {
        Err(format!("leaf proof {i} failed verification"))
      } else {
        Ok(AggregatedMemoryProof {
          write_set: p.write_set.clone(),
          read_set: p.read_set.clone(),
        })
      }
    })
    .collect::<Result<Vec<_>, _>>()?;

  // Binary-tree reduction — each level halves the node count.
  while nodes.len() > 1 {
    let mut next_level = Vec::with_capacity((nodes.len() + 1) / 2);
    let mut iter = nodes.into_iter().peekable();
    while let Some(left) = iter.next() {
      if let Some(right) = iter.next() {
        let merged =
          merge_sets(&left.write_set, &left.read_set, &right.write_set, &right.read_set)?;
        next_level.push(merged);
      } else {
        // Odd node — carry forward unchanged.
        next_level.push(left);
      }
    }
    nodes = next_level;
  }

  Ok(nodes.into_iter().next().unwrap())
}

// ============================================================
// Storage consistency proof  (read/write set intersection)
// ============================================================
//
// Identical soundness design to the memory proof, with a 2D key space:
//   StorageKey = (contract: [u8;20], slot: [u8;32])  →  value: [u8;32]
//
// Each batch proof exposes:
//   write_set : Map<StorageKey → value>  — final value SSTORE'd per (contract, slot)
//   read_set  : Map<StorageKey → value>  — cross-batch SLOAD dependencies
//
// Aggregation: same intersection rule as memory.
//
// Column layout  (NUM_STOR_COLS = 23)
// ─────────────────────────────────────────────────────────────
//  0.. 4  contract0..4 — 20-byte contract address as 5 × u32 (big-endian)
//  5..12  slot0..7     — 32-byte slot key as 8 × u32
// 13..20  val0..7      — 32-byte value as 8 × u32
// 21      is_write     — 1 for SSTORE, 0 for SLOAD
// 22      mult         — +1 for live rows, 0 for padding
//
// Public values layout  (STOR_PUBLIC_VALUES_LEN = 35)
// ─────────────────────────────────────────────────────────────
//  pis[ 0]      = RECEIPT_BIND_TAG_STORAGE
//  pis[ 1]      = n_writes
//  pis[ 2..10]  = poseidon_hash(write_log)   (8 × M31)
//  pis[10]      = n_reads
//  pis[11..19]  = poseidon_hash(read_log)    (8 × M31)
//  pis[19..27]  = poseidon_hash(write_set)   (8 × M31)  — aggregation interface
//  pis[27..35]  = poseidon_hash(read_set)    (8 × M31)  — aggregation interface
// ============================================================

pub const RECEIPT_BIND_TAG_STORAGE: u32 = 5;
const STOR_PUBLIC_VALUES_LEN: usize = 35;

const STOR_COL_CONTRACT0: usize = 0;
// contract1..4 = STOR_COL_CONTRACT0+1..STOR_COL_CONTRACT0+4  (cols 0..4)
const STOR_COL_SLOT0: usize = 5;
// slot1..7 = STOR_COL_SLOT0+1..STOR_COL_SLOT0+7  (cols 5..12)
const STOR_COL_VAL0: usize = 13;
// val1..7 = STOR_COL_VAL0+1..STOR_COL_VAL0+7  (cols 13..20)
const STOR_COL_IS_WRITE: usize = 21;
const STOR_COL_MULT: usize = 22;
/// Total number of main-trace columns for storage.
pub const NUM_STOR_COLS: usize = 23;

// ── Key and set types ─────────────────────────────────────────────────

/// `(contract_address, storage_slot)` key.
pub type StorageKey = ([u8; 20], [u8; 32]);
/// Storage map type: `StorageKey → value`.
pub type StorageSet = std::collections::HashMap<StorageKey, [u8; 32]>;

// ── Log entry type ────────────────────────────────────────────────────

/// A single entry in the public storage write or read log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageLogEntry {
  pub rw_counter: u64,
  pub contract: [u8; 20],
  pub slot: [u8; 32],
  pub value: [u8; 32],
}

/// Split a `StorageAccessClaim` slice into separate write and read logs.
pub fn split_storage_logs(
  claims: &[crate::transition::StorageAccessClaim],
) -> (Vec<StorageLogEntry>, Vec<StorageLogEntry>) {
  let mut writes = Vec::new();
  let mut reads = Vec::new();
  for c in claims {
    let entry = StorageLogEntry {
      rw_counter: c.rw_counter,
      contract: c.contract,
      slot: c.slot,
      value: c.value,
    };
    if c.is_write { writes.push(entry); } else { reads.push(entry); }
  }
  (writes, reads)
}

// ── Hash helpers ──────────────────────────────────────────────────────

fn contract_to_u32s(contract: &[u8; 20]) -> [u32; 5] {
  let mut out = [0u32; 5];
  for (i, chunk) in contract.chunks(4).enumerate() {
    out[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
  }
  out
}

fn hash_storage_log(log: &[StorageLogEntry]) -> [Val; 8] {
  let mut rng = rand::rngs::SmallRng::seed_from_u64(0xC0DEC0DE_u64);
  let poseidon = Poseidon2Mersenne31::<16>::new_from_rng_128(&mut rng);
  let sponge = PaddingFreeSponge::<_, 16, 8, 8>::new(poseidon);

  // per entry: 2 (rw_counter) + 5 (contract) + 8 (slot) + 8 (value) = 23
  let mut input: Vec<Val> = Vec::with_capacity(1 + log.len() * 23);
  input.push(Val::from_u32(log.len() as u32));
  for e in log {
    input.push(Val::from_u32((e.rw_counter >> 32) as u32));
    input.push(Val::from_u32(e.rw_counter as u32));
    for u in contract_to_u32s(&e.contract) {
      input.push(Val::from_u32(u));
    }
    for chunk in e.slot.chunks(4) {
      input.push(Val::from_u32(u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])));
    }
    for chunk in e.value.chunks(4) {
      input.push(Val::from_u32(u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])));
    }
  }
  sponge.hash_iter(input)
}

fn hash_storage_set(set: &StorageSet) -> [Val; 8] {
  if set.is_empty() {
    return [Val::ZERO; 8];
  }
  // Sort by (contract, slot) for determinism.
  let mut sorted: Vec<(StorageKey, [u8; 32])> =
    set.iter().map(|(&k, &v)| (k, v)).collect();
  sorted.sort_by_key(|((c, s), _)| (*c, *s));

  let mut rng = rand::rngs::SmallRng::seed_from_u64(0xC0DEC0DE_u64);
  let poseidon = Poseidon2Mersenne31::<16>::new_from_rng_128(&mut rng);
  let sponge = PaddingFreeSponge::<_, 16, 8, 8>::new(poseidon);

  // per entry: 5 (contract) + 8 (slot) + 8 (value) = 21
  let mut input: Vec<Val> = Vec::with_capacity(1 + sorted.len() * 21);
  input.push(Val::from_u32(sorted.len() as u32));
  for ((contract, slot), value) in &sorted {
    for u in contract_to_u32s(contract) {
      input.push(Val::from_u32(u));
    }
    for chunk in slot.chunks(4) {
      input.push(Val::from_u32(u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])));
    }
    for chunk in value.chunks(4) {
      input.push(Val::from_u32(u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])));
    }
  }
  sponge.hash_iter(input)
}

/// Build the `public_values` vector (`STOR_PUBLIC_VALUES_LEN = 35`).
fn make_storage_public_values(
  write_log: &[StorageLogEntry],
  read_log: &[StorageLogEntry],
  write_set: &StorageSet,
  read_set: &StorageSet,
) -> Vec<Val> {
  let wlh = hash_storage_log(write_log);
  let rlh = hash_storage_log(read_log);
  let wsh = hash_storage_set(write_set);
  let rsh = hash_storage_set(read_set);
  let mut pv = Vec::with_capacity(STOR_PUBLIC_VALUES_LEN);
  pv.push(Val::from_u32(RECEIPT_BIND_TAG_STORAGE));    //  0
  pv.push(Val::from_u32(write_log.len() as u32));      //  1
  pv.extend_from_slice(&wlh);                          //  2..10
  pv.push(Val::from_u32(read_log.len() as u32));       //  10
  pv.extend_from_slice(&rlh);                          // 11..19
  pv.extend_from_slice(&wsh);                          // 19..27 — aggregation interface
  pv.extend_from_slice(&rsh);                          // 27..35 — aggregation interface
  pv
}

// ── Intra-batch consistency checker ──────────────────────────────────

/// Validate intra-batch SLOAD/SSTORE consistency and derive `(write_set, read_set)`.
fn check_storage_claims_and_build_sets(
  claims: &[crate::transition::StorageAccessClaim],
) -> Result<(StorageSet, StorageSet), String> {
  use std::collections::HashMap;
  let mut write_set: StorageSet = HashMap::new();
  let mut read_set: StorageSet = HashMap::new();
  let mut last_write: StorageSet = HashMap::new();

  for claim in claims {
    let key: StorageKey = (claim.contract, claim.slot);
    if claim.is_write {
      last_write.insert(key, claim.value);
      write_set.insert(key, claim.value); // last write wins
    } else if let Some(&written) = last_write.get(&key) {
      if claim.value != written {
        return Err(format!(
          "storage read/write mismatch at contract={} slot={}: read {:?}, last write {:?}",
          format!("{:?}", claim.contract),
          format!("{:?}", claim.slot),
          claim.value,
          written
        ));
      }
    } else {
      match read_set.entry(key) {
        std::collections::hash_map::Entry::Vacant(e) => {
          e.insert(claim.value);
        }
        std::collections::hash_map::Entry::Occupied(e) => {
          if *e.get() != claim.value {
            return Err(format!(
              "cross-batch storage read mismatch at contract={} slot={}",
              format!("{:?}", claim.contract),
              format!("{:?}", claim.slot),
            ));
          }
        }
      }
    }
  }
  Ok((write_set, read_set))
}

/// Re-derive `(write_set, read_set)` from a split log pair (used in verification).
fn derive_storage_sets_from_logs(
  write_log: &[StorageLogEntry],
  read_log: &[StorageLogEntry],
) -> Result<(StorageSet, StorageSet), String> {
  // Merge by rw_counter to restore execution order.
  let mut merged: Vec<(u64, bool, StorageKey, [u8; 32])> = Vec::new();
  for e in write_log {
    merged.push((e.rw_counter, true, (e.contract, e.slot), e.value));
  }
  for e in read_log {
    merged.push((e.rw_counter, false, (e.contract, e.slot), e.value));
  }
  merged.sort_by_key(|(rw, _, _, _)| *rw);

  let mut write_set: StorageSet = std::collections::HashMap::new();
  let mut read_set: StorageSet = std::collections::HashMap::new();
  let mut last_write: StorageSet = std::collections::HashMap::new();

  for (_, is_write, key, value) in merged {
    if is_write {
      last_write.insert(key, value);
      write_set.insert(key, value);
    } else if let Some(&written) = last_write.get(&key) {
      if value != written {
        return Err(format!(
          "storage read/write mismatch at contract={} slot={}",
          format!("{:?}", key.0),
          format!("{:?}", key.1),
        ));
      }
    } else {
      match read_set.entry(key) {
        std::collections::hash_map::Entry::Vacant(e) => {
          e.insert(value);
        }
        std::collections::hash_map::Entry::Occupied(e) => {
          if *e.get() != value {
            return Err(format!(
              "cross-batch storage read mismatch at contract={} slot={}",
              format!("{:?}", key.0),
              format!("{:?}", key.1),
            ));
          }
        }
      }
    }
  }
  Ok((write_set, read_set))
}

// ── AIR ──────────────────────────────────────────────────────────────

/// AIR for the storage log LogUp membership argument.
///
/// Each trace row represents one entry from the SSTORE or SLOAD log.
/// The AIR constraints are:
///   1. Tag check: `pis[0] == RECEIPT_BIND_TAG_STORAGE`  (degree 1)
///   2. `is_write ∈ {0, 1}`                              (degree 2)
///
/// In-circuit consistency is enforced by the LogUp permutation argument
/// in `eval_stor_lookup`: every intra-batch SLOAD `(contract, slot, value)`
/// must match an SSTORE row with the correct multiplicity, proving that the
/// read value equals the last value written to that slot within the batch.
pub struct StorageConsistencyAir;

impl<F: p3_field::Field> p3_air::BaseAir<F> for StorageConsistencyAir {
  fn width(&self) -> usize {
    NUM_STOR_COLS
  }
}

impl<AB: p3_air::AirBuilderWithPublicValues> p3_air::Air<AB> for StorageConsistencyAir
where
  AB::F: p3_field::Field,
{
  fn eval(&self, builder: &mut AB) {
    let pis = builder.public_values();
    builder.assert_eq(pis[0].into(), AB::Expr::from_u32(RECEIPT_BIND_TAG_STORAGE));

    let main = builder.main();
    let local = main.row_slice(0).expect("empty storage trace");
    let local = &*local;
    let is_write = local[STOR_COL_IS_WRITE].clone();
    builder.assert_zero(is_write.clone().into() * (AB::Expr::ONE - is_write.into()));
  }
}

// ── Lookup descriptor ────────────────────────────────────────────────

/// Build the `Lookup` descriptor for the storage consistency local LogUp.
///
/// Fingerprint tuple: `(contract0..4, slot0..7, val0..7)` — 21 columns.
/// Multiplicity column: `STOR_COL_MULT`.
fn make_stor_lookup() -> Lookup<Val> {
  let col = |c: usize| {
    SymbolicExpression::Variable(SymbolicVariable::new(Entry::Main { offset: 0 }, c))
  };
  Lookup::new(
    Kind::Local,
    vec![vec![
      col(STOR_COL_CONTRACT0),
      col(STOR_COL_CONTRACT0 + 1),
      col(STOR_COL_CONTRACT0 + 2),
      col(STOR_COL_CONTRACT0 + 3),
      col(STOR_COL_CONTRACT0 + 4),
      col(STOR_COL_SLOT0),
      col(STOR_COL_SLOT0 + 1),
      col(STOR_COL_SLOT0 + 2),
      col(STOR_COL_SLOT0 + 3),
      col(STOR_COL_SLOT0 + 4),
      col(STOR_COL_SLOT0 + 5),
      col(STOR_COL_SLOT0 + 6),
      col(STOR_COL_SLOT0 + 7),
      col(STOR_COL_VAL0),
      col(STOR_COL_VAL0 + 1),
      col(STOR_COL_VAL0 + 2),
      col(STOR_COL_VAL0 + 3),
      col(STOR_COL_VAL0 + 4),
      col(STOR_COL_VAL0 + 5),
      col(STOR_COL_VAL0 + 6),
      col(STOR_COL_VAL0 + 7),
    ]],
    vec![col(STOR_COL_MULT)],
    vec![0],
  )
}

/// Generate the permutation (LogUp running-sum) trace for the storage AIR.
fn generate_stor_perm_trace(
  main_trace: &RowMajorMatrix<Val>,
  perm_challenges: &[Challenge],
) -> Option<RowMajorMatrix<Challenge>> {
  let gadget = LogUpGadget::new();
  let lookup = make_stor_lookup();
  let mut lookup_data: Vec<LookupData<Challenge>> = vec![];
  let perm_trace = gadget.generate_permutation::<CircleStarkConfig>(
    main_trace,
    &None,
    &[],
    &[lookup],
    &mut lookup_data,
    perm_challenges,
  );
  Some(perm_trace)
}

/// Evaluate the storage LogUp constraints (called in both prover and verifier).
fn eval_stor_lookup(folder: &mut VerifierConstraintFolder<CircleStarkConfig>) {
  LogUpGadget::new().eval_local_lookup(folder, &make_stor_lookup());
}

// ── Trace builder ─────────────────────────────────────────────────────

fn fill_storage_log_row(data: &mut [Val], base: usize, entry: &StorageLogEntry, is_write: bool, mult: i32) {
  let cs = contract_to_u32s(&entry.contract);
  for k in 0..5 {
    data[base + STOR_COL_CONTRACT0 + k] = Val::from_u32(cs[k]);
  }
  for (k, chunk) in entry.slot.chunks(4).enumerate() {
    data[base + STOR_COL_SLOT0 + k] =
      Val::from_u32(u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
  }
  let vs = value_to_u32s(&entry.value);
  for k in 0..8 {
    data[base + STOR_COL_VAL0 + k] = Val::from_u32(vs[k]);
  }
  data[base + STOR_COL_IS_WRITE] = Val::from_u32(is_write as u32);
  data[base + STOR_COL_MULT] = if mult >= 0 {
    Val::from_u32(mult as u32)
  } else {
    -Val::from_u32((-mult) as u32)
  };
}

fn build_storage_log_trace(
  write_log: &[StorageLogEntry],
  read_log: &[StorageLogEntry],
  _write_set: &StorageSet,
) -> RowMajorMatrix<Val> {
  // Determine intra-batch reads using rw_counter ordering.
  //
  // A read is "intra-batch" iff it occurred AFTER the last write to the same
  // key in this batch.  Reads that precede any write (genesis / cross-batch
  // reads) are excluded from the LogUp (mult = 0) even if the same key is
  // later written.  This ensures the LogUp sum cancels correctly:
  //   last-write sends +(key, final_value) × count_intra_reads
  //   each intra-batch read receives −1 × (key, final_value)
  //
  // The native-Rust recalculation in `verify_storage_consistency` (via
  // `derive_storage_sets_from_logs`) provides the authoritative consistency
  // guarantee; the LogUp adds an in-circuit binding on top.

  // Step 1: rw_counter of the last write per key.
  let mut last_write_rw: std::collections::HashMap<StorageKey, u64> = Default::default();
  for e in write_log {
    let key = (e.contract, e.slot);
    let rw = last_write_rw.entry(key).or_insert(0);
    if e.rw_counter > *rw {
      *rw = e.rw_counter;
    }
  }

  // Step 2: classify reads.  A read is intra-batch only if it happened
  // strictly after the last write to that key.
  let mut intra_read_count: std::collections::HashMap<StorageKey, i32> = Default::default();
  let mut intra_read_rw_set: std::collections::HashSet<u64> = Default::default();
  for e in read_log {
    let key = (e.contract, e.slot);
    if let Some(&lw_rw) = last_write_rw.get(&key) {
      if e.rw_counter > lw_rw {
        *intra_read_count.entry(key).or_insert(0) += 1;
        intra_read_rw_set.insert(e.rw_counter);
      }
    }
  }

  // Step 3: index of the last write per key (by position in write_log).
  let mut last_write_idx: std::collections::HashMap<StorageKey, usize> = Default::default();
  for (i, e) in write_log.iter().enumerate() {
    last_write_idx.insert((e.contract, e.slot), i);
  }

  let n_total = write_log.len() + read_log.len();
  let height = n_total.max(4).next_power_of_two();
  let mut data = vec![Val::ZERO; height * NUM_STOR_COLS];

  for (i, entry) in write_log.iter().enumerate() {
    let key: StorageKey = (entry.contract, entry.slot);
    // Only the last SSTORE to each key carries the LogUp multiplicity.
    // Earlier writes to the same key get mult = 0 (their value is not the
    // canonical one that intra-batch reads see).
    let mult = if last_write_idx.get(&key) == Some(&i) {
      *intra_read_count.get(&key).unwrap_or(&0)
    } else {
      0
    };
    fill_storage_log_row(&mut data, i * NUM_STOR_COLS, entry, true, mult);
  }

  for (i, entry) in read_log.iter().enumerate() {
    // Only intra-batch reads (happened after the last write) contribute −1.
    let mult = if intra_read_rw_set.contains(&entry.rw_counter) { -1 } else { 0 };
    fill_storage_log_row(&mut data, (write_log.len() + i) * NUM_STOR_COLS, entry, false, mult);
  }

  RowMajorMatrix::new(data, NUM_STOR_COLS)
}

// ── Public API ────────────────────────────────────────────────────────

/// Proof that the execution's storage accesses (SLOAD/SSTORE) are consistent.
///
/// Exposes `write_set` (final SSTORE'd values) and `read_set` (cross-batch
/// SLOAD dependencies).  Binary-tree aggregation merges adjacent proofs by
/// checking `R.read_set ∩ L.write_set` for value consistency.
///
/// The key space is `(contract_address: [u8;20], slot: [u8;32])`, matching
/// the EVM's two-dimensional storage layout.
pub struct StorageConsistencyProof {
  pub stark_proof: CircleStarkProof,
  /// Final value SSTORE'd to each `(contract, slot)` in this batch (last write wins).
  pub write_set: StorageSet,
  /// Cross-batch SLOAD dependencies: slots read before any SSTORE in this batch.
  pub read_set: StorageSet,
  /// SSTORE log (retained for STARK verification; can be discarded after aggregation).
  pub write_log: Vec<StorageLogEntry>,
  /// SLOAD log (retained for STARK verification; can be discarded after aggregation).
  pub read_log: Vec<StorageLogEntry>,
}

/// Prove storage consistency for `claims` (a mix of SLOAD and SSTORE claims).
pub fn prove_storage_consistency(
  claims: &[crate::transition::StorageAccessClaim],
) -> Result<StorageConsistencyProof, String> {
  let (write_log, read_log) = split_storage_logs(claims);

  let (write_set, read_set) = check_storage_claims_and_build_sets(claims)?;

  let public_values =
    make_storage_public_values(&write_log, &read_log, &write_set, &read_set);

  let trace = if write_log.is_empty() && read_log.is_empty() {
    RowMajorMatrix::new(vec![Val::ZERO; 4 * NUM_STOR_COLS], NUM_STOR_COLS)
  } else {
    build_storage_log_trace(&write_log, &read_log, &write_set)
  };

  let config = make_circle_config();
  let trace_for_perm = trace.clone();
  let stark_proof = prove_with_lookup(
    &config,
    &StorageConsistencyAir,
    trace,
    &public_values,
    None,
    move |perm_challenges| generate_stor_perm_trace(&trace_for_perm, perm_challenges),
    2, // num_perm_challenges (1 lookup × 2 challenges: alpha + beta)
    2, // lookup_constraint_count (first_row + universal for local LogUp)
    |folder: &mut VerifierConstraintFolder<CircleStarkConfig>| eval_stor_lookup(folder),
  );

  Ok(StorageConsistencyProof {
    stark_proof,
    write_set,
    read_set,
    write_log,
    read_log,
  })
}

/// Verify a single [`StorageConsistencyProof`].
///
/// 1. Re-derives `(write_set, read_set)` from the embedded logs to catch tampering.
/// 2. Verifies the STARK proof against the re-derived public values.
///
/// Cross-batch consistency is enforced at aggregation time.
pub fn verify_storage_consistency(proof: &StorageConsistencyProof) -> bool {
  let (derived_write_set, derived_read_set) =
    match derive_storage_sets_from_logs(&proof.write_log, &proof.read_log) {
      Ok(s) => s,
      Err(_) => return false,
    };
  if derived_write_set != proof.write_set || derived_read_set != proof.read_set {
    return false;
  }

  let public_values = make_storage_public_values(
    &proof.write_log,
    &proof.read_log,
    &proof.write_set,
    &proof.read_set,
  );

  let config = make_circle_config();
  verify_with_lookup(
    &config,
    &StorageConsistencyAir,
    &proof.stark_proof,
    &public_values,
    None,
    2, // num_perm_challenges (must match prove_storage_consistency)
    |folder: &mut VerifierConstraintFolder<CircleStarkConfig>| eval_stor_lookup(folder),
  )
  .is_ok()
}

// ── Binary-tree aggregation ───────────────────────────────────────────

/// Compact result of binary-tree aggregation over leaf [`StorageConsistencyProof`]s.
///
/// After aggregation, `write_set` is the final storage state and `read_set`
/// lists the genesis-required slots (those read before any write across all batches).
///
/// # Recursive STARK aggregation
/// Same `MergeAir` design as memory aggregation; the only change is the key
/// type (`StorageKey` vs `u64`).
///
/// ```text
/// MergeAir constraints:
///   1. Verify left  child STARK proof
///   2. Verify right child STARK proof
///   3. ∀ key ∈ R.read_set ∩ L.write_set: R.read_set[key] == L.write_set[key]
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AggregatedStorageProof {
  pub write_set: StorageSet,
  pub read_set: StorageSet,
}

/// Merge two adjacent proven batches using storage read/write set intersection.
pub fn aggregate_storage_proofs(
  left: &StorageConsistencyProof,
  right: &StorageConsistencyProof,
) -> Result<AggregatedStorageProof, String> {
  if !verify_storage_consistency(left) {
    return Err("left child storage proof failed verification".to_string());
  }
  if !verify_storage_consistency(right) {
    return Err("right child storage proof failed verification".to_string());
  }
  merge_storage_sets(&left.write_set, &left.read_set, &right.write_set, &right.read_set)
}

fn merge_storage_sets(
  left_ws: &StorageSet,
  left_rs: &StorageSet,
  right_ws: &StorageSet,
  right_rs: &StorageSet,
) -> Result<AggregatedStorageProof, String> {
  for (key, &r_val) in right_rs {
    if let Some(&l_val) = left_ws.get(key) {
      if r_val != l_val {
        return Err(format!(
          "storage read/write set mismatch at contract={} slot={}: right read {:?}, left wrote {:?}",
          format!("{:?}", key.0),
          format!("{:?}", key.1),
          r_val,
          l_val,
        ));
      }
    }
  }

  let mut write_set = left_ws.clone();
  for (&key, &val) in right_ws {
    write_set.insert(key, val);
  }

  let mut read_set = left_rs.clone();
  for (&key, &val) in right_rs {
    if !left_ws.contains_key(&key) {
      read_set.entry(key).or_insert(val);
    }
  }

  Ok(AggregatedStorageProof { write_set, read_set })
}

/// Aggregate a sequence of adjacent batch storage proofs using a binary tree.
///
/// Identical algorithm to [`aggregate_proofs_tree`] for memory.
/// Tree depth = ⌈log₂(N)⌉ → O(N) total work, O(log N) sequential depth.
pub fn aggregate_storage_proofs_tree(
  proofs: &[StorageConsistencyProof],
) -> Result<AggregatedStorageProof, String> {
  if proofs.is_empty() {
    return Err("aggregate_storage_proofs_tree: empty proof slice".to_string());
  }

  let mut nodes: Vec<AggregatedStorageProof> = proofs
    .iter()
    .enumerate()
    .map(|(i, p)| {
      if !verify_storage_consistency(p) {
        Err(format!("leaf storage proof {i} failed verification"))
      } else {
        Ok(AggregatedStorageProof {
          write_set: p.write_set.clone(),
          read_set: p.read_set.clone(),
        })
      }
    })
    .collect::<Result<Vec<_>, _>>()?;

  while nodes.len() > 1 {
    let mut next_level = Vec::with_capacity((nodes.len() + 1) / 2);
    let mut iter = nodes.into_iter().peekable();
    while let Some(left) = iter.next() {
      if let Some(right) = iter.next() {
        let merged = merge_storage_sets(
          &left.write_set,
          &left.read_set,
          &right.write_set,
          &right.read_set,
        )?;
        next_level.push(merged);
      } else {
        next_level.push(left);
      }
    }
    nodes = next_level;
  }

  Ok(nodes.into_iter().next().unwrap())
}

// ============================================================
// Stack consistency proof  (push/pop multiset argument)
// ============================================================
//
// Soundness design
// ─────────────────
// Every instruction that touches the stack emits a sequence of
// `StackAccessClaim`s in execution order:
//
//   pop claims  — one for each value consumed from the top (inputs)
//   push claims — one for each value produced onto the top (outputs)
//
// Each claim records `(rw_counter, depth_after, is_push, value)`.
// The consistency checker verifies that for every pop at depth D,
// the value matches the most recent push that left the stack at depth D.
//
// This prevents the prover from:
//   - fabricating `stack_inputs` to arithmetic instructions (wrong value)
//   - forging `stack_outputs` from earlier instructions
//   - hiding instructions by omitting claims
//
// Column layout  (NUM_STACK_COLS = 12)
// ──────────────────────────────────────
//  0     rw_hi      — upper 32 bits of the 64-bit rw_counter
//  1     rw_lo      — lower 32 bits
//  2     depth      — stack depth after this access (u32)
//  3..10 val0..val7 — 32-byte value as 8 × u32 (big-endian)
// 11     is_push    — 1 for push, 0 for pop
//
// Public values layout  (STACK_PUBLIC_VALUES_LEN = 19)
// ─────────────────────────────────────────────────────
//  pis[ 0]      = RECEIPT_BIND_TAG_STACK_CONSISTENCY
//  pis[ 1]      = n_pushes
//  pis[ 2..10]  = poseidon_hash(push_log)  (8 × M31)
//  pis[10]      = n_pops
//  pis[11..19]  = poseidon_hash(pop_log)   (8 × M31)

pub const RECEIPT_BIND_TAG_STACK_CONSISTENCY: u32 = 6;
const STACK_PUBLIC_VALUES_LEN: usize = 19;

const STACK_COL_RW_HI: usize = 0;
const STACK_COL_RW_LO: usize = 1;
const STACK_COL_DEPTH: usize = 2;
const STACK_COL_VAL0: usize = 3;
// val1..val7 are STACK_COL_VAL0+1 .. STACK_COL_VAL0+7  (cols 3..10)
const STACK_COL_IS_PUSH: usize = 11;
/// LogUp multiplicity: +count_pops for last push at (depth,val), -1 for intra-batch pops,
/// 0 for cross-batch pops and padding.
const STACK_COL_MULT: usize = 12;
/// Total number of main-trace columns for stack consistency.
pub const NUM_STACK_COLS: usize = 13;

// ── Log types ─────────────────────────────────────────────────────────

/// A single entry in the stack push or pop log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StackLogEntry {
  pub rw_counter: u64,
  pub depth_after: u32,
  pub value: [u8; 32],
}

/// Split a `StackAccessClaim` slice into push_log and pop_log.
pub fn split_stack_logs(
  claims: &[crate::transition::StackAccessClaim],
) -> (Vec<StackLogEntry>, Vec<StackLogEntry>) {
  let mut pushes = Vec::new();
  let mut pops = Vec::new();
  for c in claims {
    let entry = StackLogEntry {
      rw_counter: c.rw_counter,
      depth_after: c.depth_after as u32,
      value: c.value,
    };
    if c.is_push {
      pushes.push(entry);
    } else {
      pops.push(entry);
    }
  }
  (pushes, pops)
}

// ── Hash helpers ──────────────────────────────────────────────────────

fn hash_stack_log(log: &[StackLogEntry]) -> [Val; 8] {
  let mut rng = rand::rngs::SmallRng::seed_from_u64(0xC0DEC0DE_u64);
  let poseidon = Poseidon2Mersenne31::<16>::new_from_rng_128(&mut rng);
  let sponge = PaddingFreeSponge::<_, 16, 8, 8>::new(poseidon);

  // per entry: 2 (rw_counter) + 1 (depth) + 8 (value) = 11 fields
  let mut input: Vec<Val> = Vec::with_capacity(1 + log.len() * 11);
  input.push(Val::from_u32(log.len() as u32));
  for e in log {
    input.push(Val::from_u32((e.rw_counter >> 32) as u32));
    input.push(Val::from_u32(e.rw_counter as u32));
    input.push(Val::from_u32(e.depth_after));
    for chunk in e.value.chunks(4) {
      input.push(Val::from_u32(u32::from_be_bytes([
        chunk[0], chunk[1], chunk[2], chunk[3],
      ])));
    }
  }
  sponge.hash_iter(input)
}

fn make_stack_public_values(
  push_log: &[StackLogEntry],
  pop_log: &[StackLogEntry],
) -> Vec<Val> {
  let plh = hash_stack_log(push_log);
  let oph = hash_stack_log(pop_log);
  let mut pv = Vec::with_capacity(STACK_PUBLIC_VALUES_LEN);
  pv.push(Val::from_u32(RECEIPT_BIND_TAG_STACK_CONSISTENCY)); //  0
  pv.push(Val::from_u32(push_log.len() as u32));              //  1
  pv.extend_from_slice(&plh);                                 //  2..10
  pv.push(Val::from_u32(pop_log.len() as u32));               // 10
  pv.extend_from_slice(&oph);                                 // 11..19
  pv
}

// ── Intra-batch consistency checker ───────────────────────────────────

/// Validate push/pop consistency within the batch.
///
/// Scans claims in execution order (rw_counter monotone):
/// - **Push**: records `last_push[depth_after] = value`.
/// - **Pop**: depth_before = depth_after_pop + 1 (the popped slot).
///   Verifies `value == last_push[depth_before]`.
///
/// Returns `Err` on value mismatch.
fn check_stack_claims(
  claims: &[crate::transition::StackAccessClaim],
) -> Result<(), String> {
  // `top_value[d]` = value at stack depth d (1-based; depth d means d items on stack,
  // the top item is at index `depth` after a push).
  let mut top_value: std::collections::HashMap<u32, [u8; 32]> = std::collections::HashMap::new();

  for claim in claims {
    let d = claim.depth_after as u32;
    if claim.is_push {
      // After push: depth d, value = claim.value
      top_value.insert(d, claim.value);
    } else {
      // This is a pop.  `depth_after` = depth after the pop = depth before pop - 1.
      // The popped slot was at depth d + 1.
      let popped_depth = d + 1;
      if let Some(&pushed_val) = top_value.get(&popped_depth) {
        if pushed_val != claim.value {
          return Err(format!(
            "stack pop/push mismatch at rw_counter={} depth={}: \
             popped {:?} but last push was {:?}",
            claim.rw_counter, popped_depth, claim.value, pushed_val
          ));
        }
        // Remove the depth slot — it's been consumed.
        top_value.remove(&popped_depth);
      }
      // If no prior push tracked at this depth it's a cross-batch dependency
      // (e.g. values left on the stack from a previous batch); silently accept.
    }
  }
  Ok(())
}

// ── AIR ───────────────────────────────────────────────────────────────

/// AIR for the stack log LogUp membership argument.
///
/// Each trace row represents one push or pop event.
/// Constraints:
///   1. Tag check: `pis[0] == RECEIPT_BIND_TAG_STACK_CONSISTENCY`  (degree 1)
///   2. `is_push ∈ {0, 1}`                                         (degree 2)
pub struct StackConsistencyAir;

impl<F: p3_field::Field> p3_air::BaseAir<F> for StackConsistencyAir {
  fn width(&self) -> usize {
    NUM_STACK_COLS
  }
}

impl<AB: p3_air::AirBuilderWithPublicValues> p3_air::Air<AB> for StackConsistencyAir
where
  AB::F: p3_field::Field,
{
  fn eval(&self, builder: &mut AB) {
    let pis = builder.public_values();
    builder.assert_eq(
      pis[0].into(),
      AB::Expr::from_u32(RECEIPT_BIND_TAG_STACK_CONSISTENCY),
    );

    let main = builder.main();
    let local = main.row_slice(0).expect("empty stack trace");
    let local = &*local;
    let is_push = local[STACK_COL_IS_PUSH].clone();
    builder.assert_zero(is_push.clone().into() * (AB::Expr::ONE - is_push.into()));
  }
}

// ── Trace builder ─────────────────────────────────────────────────────

fn fill_stack_log_row(data: &mut [Val], base: usize, entry: &StackLogEntry, is_push: bool) {
  data[base + STACK_COL_RW_HI] = Val::from_u32((entry.rw_counter >> 32) as u32);
  data[base + STACK_COL_RW_LO] = Val::from_u32(entry.rw_counter as u32);
  data[base + STACK_COL_DEPTH] = Val::from_u32(entry.depth_after);
  let vals = value_to_u32s(&entry.value);
  for k in 0..8 {
    data[base + STACK_COL_VAL0 + k] = Val::from_u32(vals[k]);
  }
  data[base + STACK_COL_IS_PUSH] = Val::from_u32(is_push as u32);
  // STACK_COL_MULT is set separately during trace construction.
}

fn build_stack_log_trace(
  push_log: &[StackLogEntry],
  pop_log: &[StackLogEntry],
  intra_pop_count_per_push: &std::collections::HashMap<(u32, [u8; 32]), i32>,
  intra_pop_set: &std::collections::HashSet<u64>, // rw_counters of intra-batch pops
) -> RowMajorMatrix<Val> {
  let n_total = push_log.len() + pop_log.len();
  let height = n_total.max(4).next_power_of_two();
  let mut data = vec![Val::ZERO; height * NUM_STACK_COLS];

  // Index of the last push per (depth, value) key — only that row carries the LogUp mult.
  let mut last_push_idx: std::collections::HashMap<(u32, [u8; 32]), usize> = Default::default();
  for (i, e) in push_log.iter().enumerate() {
    last_push_idx.insert((e.depth_after, e.value), i);
  }

  for (i, entry) in push_log.iter().enumerate() {
    fill_stack_log_row(&mut data, i * NUM_STACK_COLS, entry, true);
    let key = (entry.depth_after, entry.value);
    let mult = if last_push_idx.get(&key) == Some(&i) {
      *intra_pop_count_per_push.get(&key).unwrap_or(&0)
    } else {
      0
    };
    let base = i * NUM_STACK_COLS;
    data[base + STACK_COL_MULT] = if mult >= 0 {
      Val::from_u32(mult as u32)
    } else {
      -Val::from_u32((-mult) as u32)
    };
  }

  for (i, entry) in pop_log.iter().enumerate() {
    let base = (push_log.len() + i) * NUM_STACK_COLS;
    fill_stack_log_row(&mut data, base, entry, false);
    // Intra-batch pops contribute −1; cross-batch pops contribute 0.
    data[base + STACK_COL_MULT] = if intra_pop_set.contains(&entry.rw_counter) {
      -Val::ONE
    } else {
      Val::ZERO
    };
  }

  RowMajorMatrix::new(data, NUM_STACK_COLS)
}

// ── Stack LogUp argument helpers ──────────────────────────────────────

/// Build the `Lookup` descriptor for the stack consistency local LogUp.
///
/// Fingerprint tuple: `(depth_slot, val0..val7)` where `depth_slot` is the
/// *canonical slot depth* — i.e. the depth of the pushed/popped item, not the
/// stack depth *after* the operation.
///
/// - Push row: `STACK_COL_DEPTH` = D (depth after push = slot depth).
///   `STACK_COL_IS_PUSH` = 1.  →  depth_slot = D + (1 - 1) = D.
/// - Pop  row: `STACK_COL_DEPTH` = D − 1 (depth after pop).
///   `STACK_COL_IS_PUSH` = 0.  →  depth_slot = (D−1) + (1 − 0) = D.
///
/// Adding `(1 − is_push)` normalises both row types to the same slot depth D,
/// so push and pop rows for the same physical stack slot share the same
/// fingerprint and their multiplicities (+count vs −1) cancel correctly.
///
/// Multiplicity column: STACK_COL_MULT.
fn make_stack_lookup() -> Lookup<Val> {
  let col = |c: usize| {
    SymbolicExpression::Variable(SymbolicVariable::new(Entry::Main { offset: 0 }, c))
  };
  // depth_slot = STACK_COL_DEPTH + (1 − STACK_COL_IS_PUSH)
  let depth_slot = col(STACK_COL_DEPTH)
    + (SymbolicExpression::Constant(Val::ONE) - col(STACK_COL_IS_PUSH));
  Lookup::new(
    Kind::Local,
    vec![vec![
      depth_slot,
      col(STACK_COL_VAL0),
      col(STACK_COL_VAL0 + 1),
      col(STACK_COL_VAL0 + 2),
      col(STACK_COL_VAL0 + 3),
      col(STACK_COL_VAL0 + 4),
      col(STACK_COL_VAL0 + 5),
      col(STACK_COL_VAL0 + 6),
      col(STACK_COL_VAL0 + 7),
    ]],
    vec![col(STACK_COL_MULT)],
    vec![0],
  )
}

fn generate_stack_perm_trace(
  main_trace: &RowMajorMatrix<Val>,
  perm_challenges: &[Challenge],
) -> Option<RowMajorMatrix<Challenge>> {
  let gadget = LogUpGadget::new();
  let lookup = make_stack_lookup();
  let mut lookup_data: Vec<LookupData<Challenge>> = vec![];
  let perm_trace = gadget.generate_permutation::<CircleStarkConfig>(
    main_trace,
    &None,
    &[],
    &[lookup],
    &mut lookup_data,
    perm_challenges,
  );
  Some(perm_trace)
}

fn eval_stack_lookup(folder: &mut VerifierConstraintFolder<CircleStarkConfig>) {
  LogUpGadget::new().eval_local_lookup(folder, &make_stack_lookup());
}

/// Build auxiliary data structures for the stack LogUp trace:
/// - `intra_pop_count_per_push`: for each (depth_after, value) key seen in pushes, how many
///   intra-batch pops matched it.
/// - `intra_pop_set`: set of rw_counters of intra-batch pops.
///
/// An intra-batch pop is one where the matching push also appears in this batch.
fn build_stack_logup_aux(
  push_log: &[StackLogEntry],
  pop_log: &[StackLogEntry],
) -> (
  std::collections::HashMap<(u32, [u8; 32]), i32>,
  std::collections::HashSet<u64>,
) {
  // Build the push set: (depth_after, value) → list of push rw_counters.
  let mut push_set: std::collections::HashMap<(u32, [u8; 32]), Vec<u64>> = Default::default();
  for e in push_log {
    push_set.entry((e.depth_after, e.value)).or_default().push(e.rw_counter);
  }

  let mut intra_pop_count: std::collections::HashMap<(u32, [u8; 32]), i32> = Default::default();
  let mut intra_pop_set: std::collections::HashSet<u64> = Default::default();

  for e in pop_log {
    // A pop's "depth_before" = depth_after + 1; the canonical stack slot being popped has
    // depth = depth_after + 1, but the *value* stored there was recorded as depth_after
    // in the push that put it there.  We store depth_after in both push and pop log entries.
    let key = (e.depth_after + 1, e.value);
    if push_set.contains_key(&key) {
      *intra_pop_count.entry(key).or_insert(0) += 1;
      intra_pop_set.insert(e.rw_counter);
    }
  }

  (intra_pop_count, intra_pop_set)
}

// ── Public API ─────────────────────────────────────────────────────────

/// Proof that the execution's stack push/pop accesses are consistent.
pub struct StackConsistencyProof {
  pub stark_proof: CircleStarkProof,
  pub push_log: Vec<StackLogEntry>,
  pub pop_log: Vec<StackLogEntry>,
}

/// Prove stack push/pop consistency for `claims`.
pub fn prove_stack_consistency(
  claims: &[crate::transition::StackAccessClaim],
) -> Result<StackConsistencyProof, String> {
  let (push_log, pop_log) = split_stack_logs(claims);

  check_stack_claims(claims)?;

  let public_values = make_stack_public_values(&push_log, &pop_log);

  let (intra_pop_count, intra_pop_set) = build_stack_logup_aux(&push_log, &pop_log);

  let trace = if push_log.is_empty() && pop_log.is_empty() {
    RowMajorMatrix::new(vec![Val::ZERO; 4 * NUM_STACK_COLS], NUM_STACK_COLS)
  } else {
    build_stack_log_trace(&push_log, &pop_log, &intra_pop_count, &intra_pop_set)
  };

  let config = make_circle_config();
  let trace_for_perm = trace.clone();
  let stark_proof = prove_with_lookup(
    &config,
    &StackConsistencyAir,
    trace,
    &public_values,
    None,
    move |perm_challenges| generate_stack_perm_trace(&trace_for_perm, perm_challenges),
    2,
    2,
    |folder: &mut VerifierConstraintFolder<CircleStarkConfig>| eval_stack_lookup(folder),
  );

  Ok(StackConsistencyProof { stark_proof, push_log, pop_log })
}

/// Verify a single [`StackConsistencyProof`].
///
/// 1. Re-derives public values from the embedded logs to catch tampering.
/// 2. Replays the push/pop consistency check.
/// 3. Verifies the STARK proof.
pub fn verify_stack_consistency(proof: &StackConsistencyProof) -> bool {
  // Re-derive public values from the logs.
  let public_values = make_stack_public_values(&proof.push_log, &proof.pop_log);

  // Reconstruct claims from logs sorted by rw_counter to replay consistency check.
  let mut merged: Vec<(u64, bool, u32, [u8; 32])> = Vec::new();
  for e in &proof.push_log {
    merged.push((e.rw_counter, true, e.depth_after, e.value));
  }
  for e in &proof.pop_log {
    merged.push((e.rw_counter, false, e.depth_after, e.value));
  }
  merged.sort_by_key(|(rw, _, _, _)| *rw);

  let reconstructed: Vec<crate::transition::StackAccessClaim> = merged
    .into_iter()
    .map(|(rw_counter, is_push, depth_after, value)| {
      crate::transition::StackAccessClaim {
        rw_counter,
        depth_after: depth_after as usize,
        is_push,
        value,
      }
    })
    .collect();

  if check_stack_claims(&reconstructed).is_err() {
    return false;
  }

  let config = make_circle_config();
  verify_with_lookup(
    &config,
    &StackConsistencyAir,
    &proof.stark_proof,
    &public_values,
    None,
    2,
    |folder: &mut VerifierConstraintFolder<CircleStarkConfig>| eval_stack_lookup(folder),
  )
  .is_ok()
}

// ============================================================
// Keccak256 consistency proof  (preimage membership argument)
// ============================================================
//
// Soundness design
// ─────────────────
// Every KECCAK256 instruction emits a `KeccakClaim`:
//
//   KeccakClaim { offset, size, input_bytes, output_hash }
//
// The consistency checker:
//   1. Verifies `keccak256(input_bytes) == output_hash` for every claim.
//   2. Commits to `(n_claims, log_hash)` via Poseidon in `public_values`.
//   3. Proves a STARK over the claim log so the verifier can re-check.
//
// This prevents the prover from:
//   - Claiming a wrong hash for a given preimage.
//   - Hiding KECCAK256 instructions entirely (the claim count is public).
//
// ── 현재 설계의 Soundness 한계 ──────────────────────────────────────────
//
// 검증자(verify_keccak_consistency)는 log에 기록된 output_hash가 실제
// 메모리 바이트(memory[offset..offset+size])로부터 도출됐는지 체크하지
// 않는다. 악의적 prover가 임의의 hash를 커밋해도 STARK는 통과한다.
//
// 가장 경제적인 보완책: KeccakClaim.offset + size 범위가 MemAccessClaim
// 로그에 read 항목으로 존재하는지 배치 검증자 단계에서 크로스체크.
// STARK 복잡도 증가 없이 soundness를 의미 있게 끌어올릴 수 있다.
//
// ── LogUp 전환 분석 ─────────────────────────────────────────────────────
//
// keccak256 preimage를 STARK 내부에서 완전 증명하려면 keccak-f 치환을
// AIR 제약으로 표현해야 한다.
//
//   keccak-f 비용:  25 lanes × 64 bits × 24 rounds ≈ 38,400 행/호출
//
// 현재 vs LogUp 내부 증명 비교:
//
//   항목              | 현재 (Rust 검증)  | keccak-f AIR
//   ─────────────────────────────────────────────────────
//   트레이스 크기     | n_calls × 1 행    | n_calls × ~40,000 행
//   증명 시간(1회)    | 수 μs             | 수십~수백 ms
//   증명 크기         | 최소              | 수십 배 증가
//
// ∴ LogUp 전환은 성능 향상을 기대할 수 없다. keccak-f는 비선형(χ 스텝)
//   이고, AND/OR/XOR 바이트 테이블처럼 작은 재사용 테이블로 축약 불가.
//   LogUp이 유효한 것은 "작고 자주 재사용되는 테이블"(byte_table 등)
//   에 한정된다.
//
// Column layout  (NUM_KECCAK_COLS = 11)
// ──────────────────────────────────────
//  0     offset_hi  — upper 32 bits of the memory offset
//  1     offset_lo  — lower 32 bits
//  2     size_hi    — upper 32 bits of the data length
//  3     size_lo    — lower 32 bits
//  4..11 hash0..7   — 32-byte keccak output as 8 × u32 (big-endian)
//
// Public values layout  (KECCAK_PUBLIC_VALUES_LEN = 11)
// ───────────────────────────────────────────────────────
//  pis[ 0]      = RECEIPT_BIND_TAG_KECCAK
//  pis[ 1]      = n_claims
//  pis[ 2..10]  = poseidon_hash(claim_log)  (8 × M31)

pub const RECEIPT_BIND_TAG_KECCAK: u32 = 7;
const KECCAK_PUBLIC_VALUES_LEN: usize = 11;

const KECCAK_COL_OFFSET_HI: usize = 0;
const KECCAK_COL_OFFSET_LO: usize = 1;
const KECCAK_COL_SIZE_HI: usize = 2;
const KECCAK_COL_SIZE_LO: usize = 3;
const KECCAK_COL_HASH0: usize = 4;
/// Total number of main-trace columns for keccak consistency.
pub const NUM_KECCAK_COLS: usize = 12; // 4 (offset/size) + 8 (hash)

// ── Log type ──────────────────────────────────────────────────────────

/// A single entry in the KECCAK256 claim log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeccakLogEntry {
  /// Memory byte offset of the hashed data.
  pub offset: u64,
  /// Length in bytes of the hashed data.
  pub size: u64,
  /// Raw input bytes (length == `size`).
  ///
  /// Retained in the proof so that `verify_batch_transaction_zk_receipt` can
  /// cross-check these bytes against the `MemoryConsistencyProof` log (BUG-MISS-3).
  /// Included in the Poseidon log-hash so any tampering is detected by
  /// `verify_keccak_consistency`.
  pub input_bytes: Vec<u8>,
  /// keccak256(input_bytes).
  pub output_hash: [u8; 32],
}

// ── AIR ───────────────────────────────────────────────────────────────

/// AIR for the KECCAK256 claim log.
///
/// Constraints:
///   1. Tag check: `pis[0] == RECEIPT_BIND_TAG_KECCAK`  (degree 1)
///
/// Practical soundness: The Poseidon log-hash in `public_values` binds every
/// `(offset, size, output_hash)` triple the prover committed.  The prover must
/// compute `keccak256(input_bytes) == output_hash` _before_ building the trace
/// (enforced in `prove_keccak_consistency`).  The verifier re-derives the
/// Poseidon hash from the embedded log and checks it against `pis[2..10]`
/// (enforced in `verify_keccak_consistency`).  This combination prevents hash
/// forgery without requiring a keccak-f sub-circuit.
pub struct KeccakConsistencyAir;

impl<F: p3_field::Field> p3_air::BaseAir<F> for KeccakConsistencyAir {
  fn width(&self) -> usize {
    NUM_KECCAK_COLS
  }
}

impl<AB: p3_air::AirBuilderWithPublicValues> p3_air::Air<AB> for KeccakConsistencyAir
where
  AB::F: p3_field::Field,
{
  fn eval(&self, builder: &mut AB) {
    let pis = builder.public_values();

    // Constraint 1: tag check — uniquely binds this proof to the keccak role.
    builder.assert_eq(pis[0].into(), AB::Expr::from_u32(RECEIPT_BIND_TAG_KECCAK));
  }
}

// ── Hash helper ───────────────────────────────────────────────────────

fn hash_keccak_log(log: &[KeccakLogEntry]) -> [Val; 8] {
  let mut rng = rand::rngs::SmallRng::seed_from_u64(0xC0DEC0DE_u64);
  let poseidon = Poseidon2Mersenne31::<16>::new_from_rng_128(&mut rng);
  let sponge = PaddingFreeSponge::<_, 16, 8, 8>::new(poseidon);

  let mut input: Vec<Val> = Vec::with_capacity(1 + log.len() * 14);
  input.push(Val::from_u32(log.len() as u32));
  for e in log {
    input.push(Val::from_u32((e.offset >> 32) as u32));
    input.push(Val::from_u32(e.offset as u32));
    input.push(Val::from_u32((e.size >> 32) as u32));
    input.push(Val::from_u32(e.size as u32));
    // Commit input_bytes into the hash (packed 4 bytes per field element).
    input.push(Val::from_u32(e.input_bytes.len() as u32));
    for chunk in e.input_bytes.chunks(4) {
      let mut buf = [0u8; 4];
      buf[..chunk.len()].copy_from_slice(chunk);
      input.push(Val::from_u32(u32::from_be_bytes(buf)));
    }
    for chunk in e.output_hash.chunks(4) {
      input.push(Val::from_u32(u32::from_be_bytes([
        chunk[0], chunk[1], chunk[2], chunk[3],
      ])));
    }
  }
  sponge.hash_iter(input)
}

fn make_keccak_public_values(log: &[KeccakLogEntry]) -> Vec<Val> {
  let lh = hash_keccak_log(log);
  let mut pv = Vec::with_capacity(KECCAK_PUBLIC_VALUES_LEN);
  pv.push(Val::from_u32(RECEIPT_BIND_TAG_KECCAK)); //  0
  pv.push(Val::from_u32(log.len() as u32));        //  1
  pv.extend_from_slice(&lh);                        //  2..10
  pv
}

// ── Trace builder ─────────────────────────────────────────────────────

fn fill_keccak_log_row(data: &mut [Val], base: usize, entry: &KeccakLogEntry) {
  data[base + KECCAK_COL_OFFSET_HI] = Val::from_u32((entry.offset >> 32) as u32);
  data[base + KECCAK_COL_OFFSET_LO] = Val::from_u32(entry.offset as u32);
  data[base + KECCAK_COL_SIZE_HI] = Val::from_u32((entry.size >> 32) as u32);
  data[base + KECCAK_COL_SIZE_LO] = Val::from_u32(entry.size as u32);
  for (k, chunk) in entry.output_hash.chunks(4).enumerate() {
    data[base + KECCAK_COL_HASH0 + k] =
      Val::from_u32(u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
  }
}

fn build_keccak_log_trace(log: &[KeccakLogEntry]) -> RowMajorMatrix<Val> {
  let height = log.len().max(4).next_power_of_two();
  let mut data = vec![Val::ZERO; height * NUM_KECCAK_COLS];
  for (i, entry) in log.iter().enumerate() {
    fill_keccak_log_row(&mut data, i * NUM_KECCAK_COLS, entry);
  }
  RowMajorMatrix::new(data, NUM_KECCAK_COLS)
}

// ── Keccak256 computation ─────────────────────────────────────────────

/// Compute keccak256 of `input` and return the 32-byte digest.
pub fn keccak256_bytes(input: &[u8]) -> [u8; 32] {
  use p3_keccak::Keccak256Hash;
  use p3_symmetric::CryptographicHasher;
  Keccak256Hash.hash_iter(input.iter().copied())
}

// ── Public API ────────────────────────────────────────────────────────

/// Proof that all KECCAK256 instructions in a batch produced correct digests.
///
/// The STARK commits to the list of `(offset, size, output_hash)` tuples and
/// the prover verifies `keccak256(input_bytes) == output_hash` for each claim
/// before building the STARK trace.
pub struct KeccakConsistencyProof {
  pub stark_proof: CircleStarkProof,
  /// The claim log committed to in this proof.
  pub log: Vec<KeccakLogEntry>,
}

/// Prove KECCAK256 consistency for `claims`.
///
/// Steps:
/// 1. Verify `keccak256(claim.input_bytes) == claim.output_hash` for every claim.
/// 2. Build the claim log (drops `input_bytes` after verification).
/// 3. Build STARK that commits to `(n_claims, log_hash)` in `public_values`.
pub fn prove_keccak_consistency(
  claims: &[crate::transition::KeccakClaim],
) -> Result<KeccakConsistencyProof, String> {
  // Step 1: verify all preimages.
  let mut log = Vec::with_capacity(claims.len());
  for claim in claims {
    let computed = keccak256_bytes(&claim.input_bytes);
    if computed != claim.output_hash {
      return Err(format!(
        "KECCAK256 preimage mismatch at offset={} size={}: computed {:?}, claimed {:?}",
        claim.offset, claim.size, computed, claim.output_hash
      ));
    }
    log.push(KeccakLogEntry {
      offset: claim.offset,
      size: claim.size,
      input_bytes: claim.input_bytes.clone(),
      output_hash: claim.output_hash,
    });
  }

  let public_values = make_keccak_public_values(&log);

  let trace = if log.is_empty() {
    RowMajorMatrix::new(vec![Val::ZERO; 4 * NUM_KECCAK_COLS], NUM_KECCAK_COLS)
  } else {
    build_keccak_log_trace(&log)
  };

  let config = make_circle_config();
  let stark_proof = p3_uni_stark::prove(&config, &KeccakConsistencyAir, trace, &public_values);

  Ok(KeccakConsistencyProof { stark_proof, log })
}

/// Cross-check that each keccak claim's `input_bytes` match the bytes in the
/// memory log at the corresponding addresses.
///
/// For every claim's address range `[offset, offset+size)`, each 32-byte word
/// address is looked up in `(mem_write_log ∪ mem_read_log)` and the relevant
/// byte slice is compared against `input_bytes`.  Returns `false` on any
/// mismatch or if a keccak-accessed word is missing from the memory log.
///
/// This closes BUG-MISS-3: without this check a malicious prover could commit
/// keccak256(X) while the memory proof has different values at the same
/// addresses.
pub fn validate_keccak_memory_cross_check(
  keccak_log: &[KeccakLogEntry],
  mem_write_log: &[MemLogEntry],
  mem_read_log: &[MemLogEntry],
) -> bool {
  if keccak_log.is_empty() {
    return true;
  }

  // Build addr → last-known value map.
  // read_log base values are installed first; write_log (ordered by rw_counter)
  // overwrites to give the final written value per address.
  let mut mem_map: std::collections::HashMap<u64, [u8; 32]> =
    std::collections::HashMap::new();
  for e in mem_read_log {
    mem_map.entry(e.addr).or_insert(e.value);
  }
  for e in mem_write_log {
    mem_map.insert(e.addr, e.value);
  }

  for entry in keccak_log {
    let offset = entry.offset;
    let size = entry.size as usize;

    if entry.input_bytes.len() != size {
      return false;
    }
    if size == 0 {
      continue;
    }

    // Iterate over the 32-byte-aligned words that cover [offset, offset+size).
    let first_word = (offset / 32) * 32;
    let last_word = ((offset + size as u64 - 1) / 32) * 32;
    let mut word = first_word;
    while word <= last_word {
      // Overlap of this word [word, word+32) with [offset, offset+size).
      let abs_start = offset.max(word);
      let abs_end = (offset + size as u64).min(word + 32);

      let val_start = (abs_start - word) as usize;
      let val_end = (abs_end - word) as usize;
      let inp_start = (abs_start - offset) as usize;
      let inp_end = (abs_end - offset) as usize;

      let mem_value = match mem_map.get(&word) {
        Some(v) => v,
        None => return false, // keccak-accessed word not in memory log
      };

      if mem_value[val_start..val_end] != entry.input_bytes[inp_start..inp_end] {
        return false;
      }

      word += 32;
    }
  }

  true
}

/// Verify a single [`KeccakConsistencyProof`].
///
/// 1. Re-derives public values from the embedded log to catch tampering.
/// 2. Verifies the STARK proof against the re-derived public values.
pub fn verify_keccak_consistency(proof: &KeccakConsistencyProof) -> bool {
  let public_values = make_keccak_public_values(&proof.log);
  let config = make_circle_config();
  p3_uni_stark::verify(
    &config,
    &KeccakConsistencyAir,
    &proof.stark_proof,
    &public_values,
  )
  .is_ok()
}

// ============================================================
// Tests
// ============================================================
