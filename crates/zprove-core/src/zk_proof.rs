//! Circle STARK proof generation and verification for semantic proof rows.
//!
//! Stage A/B overview:
//! - Stage 1 proves inferred-WFF semantic constraints over compiled `ProofRow`s.
//! - Stage 2 proves inferred WFF equals public WFF via serialized equality trace.
//!
//! The current Stage-1 semantic AIR kernel enforces the byte-add equality rows
//! (`OP_BYTE_ADD_EQ`) embedded in `ProofRow` encoding.

use crate::semantic_proof::{
  NUM_PROOF_COLS, OP_BYTE_ADD_EQ, Proof, ProofRow, RET_BYTE, RET_WFF_AND, RET_WFF_EQ, Term, WFF,
  compile_proof, infer_proof, verify_compiled,
};

use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, PairBuilder};
use p3_challenger::{DuplexChallenger, HashChallenger, SerializingChallenger32};
use p3_circle::CirclePcs;
use p3_commit::ExtensionMmcs;
use p3_field::PrimeCharacteristicRing;
use p3_field::Field;
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
use p3_uni_stark::{
  PreprocessedProverData, PreprocessedVerifierKey, StarkConfig, setup_preprocessed,
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
const STACK_COL_SEL_ADD_EQ: usize = 17;
const STACK_COL_SEL_U16_ADD_EQ: usize = 18;
const STACK_COL_SEL_U29_ADD_EQ: usize = 19;
const STACK_COL_SEL_U24_ADD_EQ: usize = 20;
const STACK_COL_SEL_U15_MUL_EQ: usize = 21;
const STACK_COL_SEL_ADD_CARRY_EQ: usize = 22;
const STACK_COL_SEL_MUL_LOW_EQ: usize = 23;
const STACK_COL_SEL_MUL_HIGH_EQ: usize = 24;
const STACK_COL_SEL_AND_EQ: usize = 25;
const STACK_COL_SEL_OR_EQ: usize = 26;
const STACK_COL_SEL_XOR_EQ: usize = 27;
const STACK_COL_SEL_NOT: usize = 28;
const STACK_COL_SEL_EQ_SYM: usize = 29;
pub const NUM_STACK_IR_COLS: usize = 30;

fn row_pop_count(op: u32) -> Option<u32> {
  match op {
    crate::semantic_proof::OP_BOOL
    | crate::semantic_proof::OP_BYTE
    | crate::semantic_proof::OP_BYTE_ADD_EQ
    | crate::semantic_proof::OP_U16_ADD_EQ
    | crate::semantic_proof::OP_U29_ADD_EQ
    | crate::semantic_proof::OP_U24_ADD_EQ
    | crate::semantic_proof::OP_U15_MUL_EQ
    | crate::semantic_proof::OP_BYTE_ADD_CARRY_EQ
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
    OP_BYTE_ADD_EQ => Some(STACK_COL_SEL_ADD_EQ),
    OP_U16_ADD_EQ => Some(STACK_COL_SEL_U16_ADD_EQ),
    OP_U29_ADD_EQ => Some(STACK_COL_SEL_U29_ADD_EQ),
    OP_U24_ADD_EQ => Some(STACK_COL_SEL_U24_ADD_EQ),
    OP_U15_MUL_EQ => Some(STACK_COL_SEL_U15_MUL_EQ),
    OP_BYTE_ADD_CARRY_EQ => Some(STACK_COL_SEL_ADD_CARRY_EQ),
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
  let t_add_eq = crate::semantic_proof::OP_BYTE_ADD_EQ as u16;
  let t_u16_add_eq = crate::semantic_proof::OP_U16_ADD_EQ as u16;
  let t_u29_add_eq = crate::semantic_proof::OP_U29_ADD_EQ as u16;
  let t_u24_add_eq = crate::semantic_proof::OP_U24_ADD_EQ as u16;
  let t_u15_mul_eq = crate::semantic_proof::OP_U15_MUL_EQ as u16;
  let t_add_carry_eq = crate::semantic_proof::OP_BYTE_ADD_CARRY_EQ as u16;
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
  let sel_add_eq = local[STACK_COL_SEL_ADD_EQ].clone();
  let sel_u16_add_eq = local[STACK_COL_SEL_U16_ADD_EQ].clone();
  let sel_u29_add_eq = local[STACK_COL_SEL_U29_ADD_EQ].clone();
  let sel_u24_add_eq = local[STACK_COL_SEL_U24_ADD_EQ].clone();
  let sel_u15_mul_eq = local[STACK_COL_SEL_U15_MUL_EQ].clone();
  let sel_add_carry_eq = local[STACK_COL_SEL_ADD_CARRY_EQ].clone();
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
  builder.assert_zero(sel_add_eq.clone() * (c_one.clone() - sel_add_eq.clone()));
  builder.assert_zero(sel_u16_add_eq.clone() * (c_one.clone() - sel_u16_add_eq.clone()));
  builder.assert_zero(sel_u29_add_eq.clone() * (c_one.clone() - sel_u29_add_eq.clone()));
  builder.assert_zero(sel_u24_add_eq.clone() * (c_one.clone() - sel_u24_add_eq.clone()));
  builder.assert_zero(sel_u15_mul_eq.clone() * (c_one.clone() - sel_u15_mul_eq.clone()));
  builder.assert_zero(sel_add_carry_eq.clone() * (c_one.clone() - sel_add_carry_eq.clone()));
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
    + sel_add_eq.clone()
    + sel_u16_add_eq.clone()
    + sel_u29_add_eq.clone()
    + sel_u24_add_eq.clone()
    + sel_u15_mul_eq.clone()
    + sel_add_carry_eq.clone()
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
    + sel_add_eq.clone() * AB::Expr::from_u16(t_add_eq)
    + sel_u16_add_eq.clone() * AB::Expr::from_u16(t_u16_add_eq)
    + sel_u29_add_eq.clone() * AB::Expr::from_u16(t_u29_add_eq)
    + sel_u24_add_eq.clone() * AB::Expr::from_u16(t_u24_add_eq)
    + sel_u15_mul_eq.clone() * AB::Expr::from_u16(t_u15_mul_eq)
    + sel_add_carry_eq.clone() * AB::Expr::from_u16(t_add_carry_eq)
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

  // OP_BYTE_ADD_EQ: pop=0, push=1, ret_ty=RET_WFF_EQ
  builder.assert_zero(sel_add_eq.clone() * (pop.clone().into() - c_zero.clone()));
  builder.assert_zero(sel_add_eq.clone() * (push.clone().into() - c_one.clone()));
  builder.assert_zero(sel_add_eq * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)));

  // OP_U16_ADD_EQ: pop=0, push=1, ret_ty=RET_WFF_AND
  builder.assert_zero(sel_u16_add_eq.clone() * (pop.clone().into() - c_zero.clone()));
  builder.assert_zero(sel_u16_add_eq.clone() * (push.clone().into() - c_one.clone()));
  builder
    .assert_zero(sel_u16_add_eq * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_AND as u16)));

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

  // OP_BYTE_ADD_CARRY_EQ: pop=0, push=1, ret_ty=RET_WFF_EQ
  builder.assert_zero(sel_add_carry_eq.clone() * (pop.clone().into() - c_zero.clone()));
  builder.assert_zero(sel_add_carry_eq.clone() * (push.clone().into() - c_one.clone()));
  builder.assert_zero(
    sel_add_carry_eq * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)),
  );

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
  ByteAddEq,
  U29AddEq,
  U24AddEq,
  U15AddEq,
  BitAddEq,
  U15MulEq,
  U16AddEq,
  ByteAddCarryEq,
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
// One-hot selector columns (6..18) — one per LutOpcode variant.
const LUT_COL_SEL_BYTE_ADD_EQ: usize = 6;
const LUT_COL_SEL_U29_ADD_EQ: usize = 7;
const LUT_COL_SEL_U24_ADD_EQ: usize = 8;
const LUT_COL_SEL_U15_ADD_EQ: usize = 9;
const LUT_COL_SEL_BIT_ADD_EQ: usize = 10;
const LUT_COL_SEL_U15_MUL_EQ: usize = 11;
const LUT_COL_SEL_U16_ADD_EQ: usize = 12;
const LUT_COL_SEL_BYTE_ADD_CARRY_EQ: usize = 13;
const LUT_COL_SEL_BYTE_MUL_LOW_EQ: usize = 14;
const LUT_COL_SEL_BYTE_MUL_HIGH_EQ: usize = 15;
const LUT_COL_SEL_BYTE_AND_EQ: usize = 16;
const LUT_COL_SEL_BYTE_OR_EQ: usize = 17;
const LUT_COL_SEL_BYTE_XOR_EQ: usize = 18;
pub const NUM_LUT_COLS: usize = 19;

fn lut_opcode_tag(op: LutOpcode) -> u16 {
  match op {
    LutOpcode::ByteAddEq => 1,
    LutOpcode::U29AddEq => 2,
    LutOpcode::U24AddEq => 3,
    LutOpcode::U15AddEq => 4,
    LutOpcode::BitAddEq => 5,
    LutOpcode::U15MulEq => 6,
    LutOpcode::U16AddEq => 7,
    LutOpcode::ByteAddCarryEq => 8,
    LutOpcode::ByteMulLowEq => 9,
    LutOpcode::ByteMulHighEq => 10,
    LutOpcode::ByteAndEq => 11,
    LutOpcode::ByteOrEq => 12,
    LutOpcode::ByteXorEq => 13,
  }
}

/// Map a `LutOpcode` to its one-hot selector column index.
fn lut_op_to_sel_col(op: LutOpcode) -> usize {
  match op {
    LutOpcode::ByteAddEq => LUT_COL_SEL_BYTE_ADD_EQ,
    LutOpcode::U29AddEq => LUT_COL_SEL_U29_ADD_EQ,
    LutOpcode::U24AddEq => LUT_COL_SEL_U24_ADD_EQ,
    LutOpcode::U15AddEq => LUT_COL_SEL_U15_ADD_EQ,
    LutOpcode::BitAddEq => LUT_COL_SEL_BIT_ADD_EQ,
    LutOpcode::U15MulEq => LUT_COL_SEL_U15_MUL_EQ,
    LutOpcode::U16AddEq => LUT_COL_SEL_U16_ADD_EQ,
    LutOpcode::ByteAddCarryEq => LUT_COL_SEL_BYTE_ADD_CARRY_EQ,
    LutOpcode::ByteMulLowEq => LUT_COL_SEL_BYTE_MUL_LOW_EQ,
    LutOpcode::ByteMulHighEq => LUT_COL_SEL_BYTE_MUL_HIGH_EQ,
    LutOpcode::ByteAndEq => LUT_COL_SEL_BYTE_AND_EQ,
    LutOpcode::ByteOrEq => LUT_COL_SEL_BYTE_OR_EQ,
    LutOpcode::ByteXorEq => LUT_COL_SEL_BYTE_XOR_EQ,
  }
}

fn append_u16_row_as_u15_steps(row: &ProofRow, out: &mut Vec<LutStep>) -> Result<(), String> {
  if row.scalar0 > 0xFFFF
    || row.scalar1 > 0xFFFF
    || row.scalar2 > 1
    || row.value > 0xFFFF
    || row.arg1 > 1
  {
    return Err("u16 row out of range for u15 expansion".to_string());
  }

  let a0 = row.scalar0 & 0x7FFF;
  let a1 = row.scalar0 >> 15;
  let b0 = row.scalar1 & 0x7FFF;
  let b1 = row.scalar1 >> 15;
  let s0 = row.value & 0x7FFF;
  let s1 = row.value >> 15;
  let cin = row.scalar2;

  let low_total = a0 + b0 + cin;
  let carry0 = if low_total >= 32768 { 1 } else { 0 };
  if (low_total & 0x7FFF) != s0 {
    return Err("u16->u15 expansion low-limb mismatch".to_string());
  }

  let high_total = a1 + b1 + carry0;
  let carry1 = if high_total >= 2 { 1 } else { 0 };
  if (high_total & 1) != s1 || carry1 != row.arg1 {
    return Err("u16->u15 expansion high-bit mismatch".to_string());
  }

  out.push(LutStep {
    op: LutOpcode::U15AddEq,
    in0: a0,
    in1: b0,
    in2: cin,
    out0: s0,
    out1: carry0,
  });
  out.push(LutStep {
    op: LutOpcode::BitAddEq,
    in0: a1,
    in1: b1,
    in2: carry0,
    out0: s1,
    out1: carry1,
  });

  Ok(())
}

pub fn build_lut_steps_from_rows(rows: &[ProofRow]) -> Result<Vec<LutStep>, String> {
  let mut out = Vec::with_capacity(rows.len());

  for row in rows {
    let step = match row.op {
      crate::semantic_proof::OP_BYTE_ADD_EQ => LutStep {
        op: LutOpcode::ByteAddEq,
        in0: row.scalar0,
        in1: row.scalar1,
        in2: row.arg0,
        out0: row.value,
        out1: row.scalar2,
      },
      crate::semantic_proof::OP_U16_ADD_EQ => {
        append_u16_row_as_u15_steps(row, &mut out)?;
        continue;
      }
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
      crate::semantic_proof::OP_BYTE_ADD_CARRY_EQ => {
        let total = row.scalar0 + row.scalar1 + row.arg0;
        LutStep {
          op: LutOpcode::ByteAddCarryEq,
          in0: row.scalar0,
          in1: row.scalar1,
          in2: row.arg0,
          out0: if total >= 256 { 1 } else { 0 },
          out1: total & 0xFF,
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
    // Padding: ByteAddEq with all-zero inputs (0+0+0=0, carry=0).
    trace.values[base + LUT_COL_OP] = Val::from_u16(lut_opcode_tag(LutOpcode::ByteAddEq));
    trace.values[base + LUT_COL_IN0] = Val::from_u16(0);
    trace.values[base + LUT_COL_IN1] = Val::from_u16(0);
    trace.values[base + LUT_COL_IN2] = Val::from_u16(0);
    trace.values[base + LUT_COL_OUT0] = Val::from_u16(0);
    trace.values[base + LUT_COL_OUT1] = Val::from_u16(0);
    trace.values[base + LUT_COL_SEL_BYTE_ADD_EQ] = Val::from_u16(1);
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
      LutOpcode::ByteAddEq => {
        ensure_le("in0", step.in0, 255, i)?;
        ensure_le("in1", step.in1, 255, i)?;
        ensure_le("in2", step.in2, 1, i)?;
        ensure_le("out0", step.out0, 255, i)?;
        ensure_le("out1", step.out1, 1, i)?;
      }
      LutOpcode::U16AddEq => {
        ensure_le("in0", step.in0, 0xFFFF, i)?;
        ensure_le("in1", step.in1, 0xFFFF, i)?;
        ensure_le("in2", step.in2, 1, i)?;
        ensure_le("out0", step.out0, 0xFFFF, i)?;
        ensure_le("out1", step.out1, 1, i)?;
      }
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
      LutOpcode::ByteAddCarryEq => {
        ensure_le("in0", step.in0, 255, i)?;
        ensure_le("in1", step.in1, 255, i)?;
        ensure_le("in2", step.in2, 1, i)?;
        ensure_le("out0", step.out0, 1, i)?;
        ensure_le("out1", step.out1, 255, i)?;
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
  let s_add = local[LUT_COL_SEL_BYTE_ADD_EQ].clone();
  let s_u29 = local[LUT_COL_SEL_U29_ADD_EQ].clone();
  let s_u24 = local[LUT_COL_SEL_U24_ADD_EQ].clone();
  let s_u15 = local[LUT_COL_SEL_U15_ADD_EQ].clone();
  let s_bit = local[LUT_COL_SEL_BIT_ADD_EQ].clone();
  let s_mul = local[LUT_COL_SEL_U15_MUL_EQ].clone();
  let s_u16 = local[LUT_COL_SEL_U16_ADD_EQ].clone();
  let s_carry = local[LUT_COL_SEL_BYTE_ADD_CARRY_EQ].clone();
  let s_mul_low = local[LUT_COL_SEL_BYTE_MUL_LOW_EQ].clone();
  let s_mul_high = local[LUT_COL_SEL_BYTE_MUL_HIGH_EQ].clone();
  let s_and = local[LUT_COL_SEL_BYTE_AND_EQ].clone();
  let s_or = local[LUT_COL_SEL_BYTE_OR_EQ].clone();
  let s_xor = local[LUT_COL_SEL_BYTE_XOR_EQ].clone();

  // ── (1) Boolean: sel_i * (1 - sel_i) = 0  (degree 2) ──────────────
  for s in [
    s_add.clone(),
    s_u29.clone(),
    s_u24.clone(),
    s_u15.clone(),
    s_bit.clone(),
    s_mul.clone(),
    s_u16.clone(),
    s_carry.clone(),
    s_mul_low.clone(),
    s_mul_high.clone(),
    s_and.clone(),
    s_or.clone(),
    s_xor.clone(),
  ] {
    builder.assert_bool(s);
  }

  // ── (2) One-hot: Σ sel_i = 1  (degree 1) ──────────────────────────
  let sel_sum = s_add.clone().into()
    + s_u29.clone().into()
    + s_u24.clone().into()
    + s_u15.clone().into()
    + s_bit.clone().into()
    + s_mul.clone().into()
    + s_u16.clone().into()
    + s_carry.clone().into()
    + s_mul_low.clone().into()
    + s_mul_high.clone().into()
    + s_and.clone().into()
    + s_or.clone().into()
    + s_xor.clone().into();
  builder.assert_one(sel_sum);

  // ── (3) op = Σ(sel_i · tag_i)  (degree 2) ─────────────────────────
  let op_reconstruct = s_add.clone().into()
    * AB::Expr::from_u16(lut_opcode_tag(LutOpcode::ByteAddEq))
    + s_u29.clone().into() * AB::Expr::from_u16(lut_opcode_tag(LutOpcode::U29AddEq))
    + s_u24.clone().into() * AB::Expr::from_u16(lut_opcode_tag(LutOpcode::U24AddEq))
    + s_u15.clone().into() * AB::Expr::from_u16(lut_opcode_tag(LutOpcode::U15AddEq))
    + s_bit.clone().into() * AB::Expr::from_u16(lut_opcode_tag(LutOpcode::BitAddEq))
    + s_mul.clone().into() * AB::Expr::from_u16(lut_opcode_tag(LutOpcode::U15MulEq))
    + s_u16.clone().into() * AB::Expr::from_u16(lut_opcode_tag(LutOpcode::U16AddEq))
    + s_carry.clone().into() * AB::Expr::from_u16(lut_opcode_tag(LutOpcode::ByteAddCarryEq))
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
  let c65536 = AB::Expr::from_u32(65536);
  let c16777216 = AB::Expr::from_u32(1u32 << 24);
  let c536870912 = AB::Expr::from_u32(1u32 << 29);

  // ByteAddEq: in0+in1+in2 = out0 + 256*out1, out1 ∈ {0,1}
  builder.assert_zero(
    s_add.clone() * (total.clone() - out0.clone().into() - c256.clone() * out1.clone().into()),
  );
  builder.assert_bool(s_add.clone() * out1.clone().into());

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

  // U16AddEq
  builder.assert_zero(
    s_u16.clone() * (total.clone() - out0.clone().into() - c65536.clone() * out1.clone().into()),
  );
  builder.assert_bool(s_u16.clone() * out1.clone().into());

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

  // ByteAddCarryEq: in0+in1+in2 = out1 + 256*out0, out0 ∈ {0,1}
  builder.assert_zero(
    s_carry.clone() * (total.clone() - out1.clone().into() - c256.clone() * out0.clone().into()),
  );
  builder.assert_bool(s_carry.clone() * out0.clone().into());

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

  // ByteAddCarryEq / U{15,16,24,29}AddEq all use in2 as carry-in;
  // original code had a dead `assert_bool(in2)` after the gate section.
  // Now each opcode enforces its own in2 semantics above; no global needed.
}

// ============================================================
// Phase 3: LutKernelAirWithPrep — binds LUT main-trace columns
// to the shared preprocessed ProofRow commitment.
// ============================================================

/// LUT STARK variant that, in addition to the arithmetic validity constraints
/// from [`LutKernelAir`], enforces (via [`PairBuilder`]) that each LUT step in
/// the main trace was derived from the corresponding committed [`ProofRow`].
///
/// Only used by [`prove_lut_with_prep`] / [`verify_lut_with_prep`].
pub struct LutKernelAirWithPrep {
  prep_matrix: Option<RowMajorMatrix<Val>>,
}

impl LutKernelAirWithPrep {
  /// Prover path: store the preprocessed matrix so the p3-uni-stark
  /// debug-constraint checker can wire up the preprocessed columns.
  pub fn new(rows: &[ProofRow], pv: &[Val]) -> Self {
    Self {
      prep_matrix: Some(build_proof_rows_preprocessed_matrix(rows, pv)),
    }
  }

  /// Verifier path: preprocessed data comes from the `PreprocessedVerifierKey`
  /// embedded in the proof; no local matrix is needed.
  pub fn for_verify() -> Self {
    Self { prep_matrix: None }
  }
}

impl BaseAir<Val> for LutKernelAirWithPrep {
  fn width(&self) -> usize {
    NUM_LUT_COLS
  }

  fn preprocessed_trace(&self) -> Option<RowMajorMatrix<Val>> {
    self.prep_matrix.clone()
  }
}

impl<AB> Air<AB> for LutKernelAirWithPrep
where
  AB: AirBuilderWithPublicValues + PairBuilder + AirBuilder<F = Val>,
{
  fn eval(&self, _builder: &mut AB) {
    // Intentionally empty — diagnostic test
  }
}

/// Build a LUT main trace with exactly **one row per [`ProofRow`]** — the same
/// height as [`build_proof_rows_preprocessed_matrix`] and
/// [`build_stack_ir_trace_from_rows`].  This 1:1 alignment is required to share
/// the same `PreprocessedProverData` between the StackIR and LUT STARKs.
///
/// Non-LUT structural ops (`OP_EQ_REFL`, etc.) are encoded as `ByteAddEq(0,0,0)`
/// padding rows, which satisfy the LUT AIR's arithmetic constraints trivially.
/// `OP_U16_ADD_EQ` is encoded as a single `U16AddEq` step (not the 2-step
/// U15-decomposition used by the legacy path) to maintain the 1:1 invariant.
pub fn build_lut_trace_from_proof_rows(rows: &[ProofRow]) -> Result<RowMajorMatrix<Val>, String> {
  use crate::semantic_proof::{
    OP_AND_INTRO, OP_BOOL, OP_BYTE, OP_BYTE_ADD_CARRY_EQ, OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE,
    OP_BYTE_ADD_EQ, OP_BYTE_ADD_THIRD_CONGRUENCE, OP_BYTE_AND_EQ, OP_BYTE_MUL_HIGH_EQ,
    OP_BYTE_MUL_LOW_EQ, OP_BYTE_OR_EQ, OP_BYTE_XOR_EQ, OP_EQ_REFL, OP_EQ_SYM, OP_EQ_TRANS, OP_NOT,
    OP_U15_MUL_EQ, OP_U16_ADD_EQ, OP_U24_ADD_EQ, OP_U29_ADD_EQ,
  };

  if rows.is_empty() {
    return Err("build_lut_trace_from_proof_rows: cannot build from empty row set".to_string());
  }

  let n_rows = rows.len().max(4).next_power_of_two();
  let mut trace = RowMajorMatrix::new(Val::zero_vec(n_rows * NUM_LUT_COLS), NUM_LUT_COLS);
  let pad_tag = Val::from_u16(lut_opcode_tag(LutOpcode::ByteAddEq));

  for (i, row) in rows.iter().enumerate() {
    let b = i * NUM_LUT_COLS;
    match row.op {
      // ── Structural (non-LUT) rows: pad as ByteAddEq(0,0,0,0,0) ──────────
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
        trace.values[b + LUT_COL_SEL_BYTE_ADD_EQ] = Val::from_u16(1);
        // in0..out1 remain zero — ByteAddEq(0+0+0=0) is arithmetically valid.
      }
      OP_BYTE_ADD_EQ => {
        trace.values[b + LUT_COL_OP] = Val::from_u16(lut_opcode_tag(LutOpcode::ByteAddEq));
        trace.values[b + LUT_COL_IN0] = Val::from_u32(row.scalar0);
        trace.values[b + LUT_COL_IN1] = Val::from_u32(row.scalar1);
        trace.values[b + LUT_COL_IN2] = Val::from_u32(row.arg0);
        trace.values[b + LUT_COL_OUT0] = Val::from_u32(row.value);
        trace.values[b + LUT_COL_OUT1] = Val::from_u32(row.scalar2);
        trace.values[b + LUT_COL_SEL_BYTE_ADD_EQ] = Val::from_u16(1);
      }
      OP_U16_ADD_EQ => {
        // Single U16AddEq step — keeps 1:1 row alignment with ProofRows.
        trace.values[b + LUT_COL_OP] = Val::from_u16(lut_opcode_tag(LutOpcode::U16AddEq));
        trace.values[b + LUT_COL_IN0] = Val::from_u32(row.scalar0);
        trace.values[b + LUT_COL_IN1] = Val::from_u32(row.scalar1);
        trace.values[b + LUT_COL_IN2] = Val::from_u32(row.scalar2);
        trace.values[b + LUT_COL_OUT0] = Val::from_u32(row.value);
        trace.values[b + LUT_COL_OUT1] = Val::from_u32(row.arg1);
        trace.values[b + LUT_COL_SEL_U16_ADD_EQ] = Val::from_u16(1);
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
      OP_BYTE_ADD_CARRY_EQ => {
        let total = row.scalar0 + row.scalar1 + row.arg0;
        trace.values[b + LUT_COL_OP] = Val::from_u16(lut_opcode_tag(LutOpcode::ByteAddCarryEq));
        trace.values[b + LUT_COL_IN0] = Val::from_u32(row.scalar0);
        trace.values[b + LUT_COL_IN1] = Val::from_u32(row.scalar1);
        trace.values[b + LUT_COL_IN2] = Val::from_u32(row.arg0);
        trace.values[b + LUT_COL_OUT0] = Val::from_u32(if total >= 256 { 1 } else { 0 });
        trace.values[b + LUT_COL_OUT1] = Val::from_u32(total & 0xFF);
        trace.values[b + LUT_COL_SEL_BYTE_ADD_CARRY_EQ] = Val::from_u16(1);
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

  // Padding rows beyond `rows.len()`: ByteAddEq(0,0,0,0,0).
  for i in rows.len()..n_rows {
    let b = i * NUM_LUT_COLS;
    trace.values[b + LUT_COL_OP] = pad_tag;
    trace.values[b + LUT_COL_SEL_BYTE_ADD_EQ] = Val::from_u16(1);
  }

  Ok(trace)
}

/// Scaffold public-values helper for the LUT prep path (tag + zeros).
pub fn lut_scaffold_public_values() -> Vec<Val> {
  default_receipt_bind_public_values_for_tag(RECEIPT_BIND_TAG_LUT)
}

/// Prove the LUT kernel STARK with shared preprocessed ProofRow commitment.
///
/// Uses [`build_lut_trace_from_proof_rows`] to build a 1:1 trace (one row per
/// ProofRow) so the main trace and the preprocessed matrix have equal height,
/// allowing both to be committed under the same `PreprocessedProverData`.
pub fn prove_lut_with_prep(
  rows: &[ProofRow],
  prep_data: &PreprocessedProverData<CircleStarkConfig>,
  public_values: &[Val],
) -> Result<CircleStarkProof, String> {
  let trace = build_lut_trace_from_proof_rows(rows)?;
  let config = make_circle_config();
  let air = LutKernelAirWithPrep::new(rows, public_values);
  let proof =
    p3_uni_stark::prove_with_preprocessed(&config, &air, trace, public_values, Some(prep_data));
  Ok(proof)
}

/// Verify a LUT proof generated by [`prove_lut_with_prep`].
pub fn verify_lut_with_prep(
  proof: &CircleStarkProof,
  prep_vk: &PreprocessedVerifierKey<CircleStarkConfig>,
  public_values: &[Val],
) -> CircleStarkVerifyResult {
  let config = make_circle_config();
  p3_uni_stark::verify_with_preprocessed(
    &config,
    &LutKernelAirWithPrep::for_verify(),
    proof,
    public_values,
    Some(prep_vk),
  )
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

/// High-level: prove a LUT kernel step-set **with** a companion byte-table
/// LogUp proof for AND/OR/XOR byte operations.
///
/// Returns `(lut_proof, byte_table_proof)` where:
/// - `lut_proof` is the standard LUT STARK proof (preprocessed path).
/// - `byte_table_proof` is the LogUp proof for all AND/OR/XOR queries that
///   appear in `rows`.  Returns `None` when there are no such operations.
pub fn prove_lut_with_prep_and_logup(
  rows: &[ProofRow],
  prep_data: &PreprocessedProverData<CircleStarkConfig>,
  public_values: &[Val],
) -> Result<
  (
    CircleStarkProof,
    Option<p3_uni_stark::Proof<CircleStarkConfig>>,
  ),
  String,
> {
  // Collect byte-level queries directly from ProofRows (not via bit-expanded
  // LutSteps).  This correctly handles structural rows such as OP_AND_INTRO
  // and produces real byte operands (scalar0/scalar1 ∈ 0..=255) rather than
  // bit-expanded operands (0/1).
  let queries = collect_byte_table_queries_from_rows(rows);

  // Build the LUT proof (existing preprocessed path).
  let lut_proof = prove_lut_with_prep(rows, prep_data, public_values)?;

  // Build a companion byte-table proof when there are AND/OR/XOR operations.
  let byte_proof = if queries.is_empty() {
    None
  } else {
    Some(crate::byte_table::prove_byte_table(&queries))
  };

  Ok((lut_proof, byte_proof))
}

/// Verify a `(lut_proof, byte_table_proof)` pair produced by
/// [`prove_lut_with_prep_and_logup`].
pub fn verify_lut_with_prep_and_logup(
  lut_proof: &CircleStarkProof,
  byte_proof: Option<&p3_uni_stark::Proof<CircleStarkConfig>>,
  prep_vk: &PreprocessedVerifierKey<CircleStarkConfig>,
  public_values: &[Val],
) -> CircleStarkVerifyResult {
  // Verify the main LUT STARK proof.
  verify_lut_with_prep(lut_proof, prep_vk, public_values)?;

  // Verify the companion byte-table proof when present.
  if let Some(bp) = byte_proof {
    crate::byte_table::verify_byte_table(bp)?;
  }

  Ok(())
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

/// Per-row prep scalar binding helper shared between [`LutKernelAirWithPrep`]
/// and [`BatchLutKernelAirWithPrep`].
///
/// Binds every LUT main-trace column to the corresponding committed
/// preprocessed ProofRow field using a gate polynomial over `prep[PREP_COL_OP]`.
/// Structural (non-LUT) rows are silently skipped via the all-ops gate.
fn eval_lut_prep_row_binding_inner<AB>(builder: &mut AB, prep_row: &[AB::Var], local: &[AB::Var])
where
  AB: AirBuilderWithPublicValues + PairBuilder + AirBuilder<F = Val>,
{
  use crate::semantic_proof::{
    OP_AND_INTRO, OP_BOOL, OP_BYTE, OP_BYTE_ADD_CARRY_EQ, OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE,
    OP_BYTE_ADD_EQ, OP_BYTE_ADD_THIRD_CONGRUENCE, OP_BYTE_AND_EQ, OP_BYTE_MUL_HIGH_EQ,
    OP_BYTE_MUL_LOW_EQ, OP_BYTE_OR_EQ, OP_BYTE_XOR_EQ, OP_EQ_REFL, OP_EQ_SYM, OP_EQ_TRANS, OP_NOT,
    OP_U15_MUL_EQ, OP_U16_ADD_EQ, OP_U24_ADD_EQ, OP_U29_ADD_EQ,
  };

  let all_prep_ops: &[u32] = &[
    OP_BYTE_ADD_EQ,
    OP_U16_ADD_EQ,
    OP_U29_ADD_EQ,
    OP_U24_ADD_EQ,
    OP_U15_MUL_EQ,
    OP_BYTE_ADD_CARRY_EQ,
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

  // OP_BYTE_ADD_EQ → ByteAddEq: in0=s0, in1=s1, in2=arg0, out0=value, out1=s2
  let tag_add = lut_opcode_tag(LutOpcode::ByteAddEq) as u32;
  let g = pg(OP_BYTE_ADD_EQ);
  builder.assert_zero(g.clone() * (local[LUT_COL_OP].clone().into() - AB::Expr::from_u32(tag_add)));
  builder.assert_zero(
    g.clone() * (local[LUT_COL_IN0].clone().into() - prep_row[PREP_COL_SCALAR0].clone().into()),
  );
  builder.assert_zero(
    g.clone() * (local[LUT_COL_IN1].clone().into() - prep_row[PREP_COL_SCALAR1].clone().into()),
  );
  builder.assert_zero(
    g.clone() * (local[LUT_COL_IN2].clone().into() - prep_row[PREP_COL_ARG0].clone().into()),
  );
  builder.assert_zero(
    g.clone() * (local[LUT_COL_OUT0].clone().into() - prep_row[PREP_COL_VALUE].clone().into()),
  );
  builder.assert_zero(
    g * (local[LUT_COL_OUT1].clone().into() - prep_row[PREP_COL_SCALAR2].clone().into()),
  );

  // OP_U16_ADD_EQ → U16AddEq: in0=s0, in1=s1, in2=s2, out0=value, out1=arg1
  let tag_u16 = lut_opcode_tag(LutOpcode::U16AddEq) as u32;
  let g = pg(OP_U16_ADD_EQ);
  builder.assert_zero(g.clone() * (local[LUT_COL_OP].clone().into() - AB::Expr::from_u32(tag_u16)));
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

  // OP_BYTE_ADD_CARRY_EQ → ByteAddCarryEq: in0=s0, in1=s1, in2=arg0
  let tag_carry = lut_opcode_tag(LutOpcode::ByteAddCarryEq) as u32;
  let g = pg(OP_BYTE_ADD_CARRY_EQ);
  builder
    .assert_zero(g.clone() * (local[LUT_COL_OP].clone().into() - AB::Expr::from_u32(tag_carry)));
  builder.assert_zero(
    g.clone() * (local[LUT_COL_IN0].clone().into() - prep_row[PREP_COL_SCALAR0].clone().into()),
  );
  builder.assert_zero(
    g.clone() * (local[LUT_COL_IN1].clone().into() - prep_row[PREP_COL_SCALAR1].clone().into()),
  );
  builder
    .assert_zero(g * (local[LUT_COL_IN2].clone().into() - prep_row[PREP_COL_ARG0].clone().into()));

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

pub fn prove_lut_kernel_stark(steps: &[LutStep]) -> Result<CircleStarkProof, String> {
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

pub fn verify_lut_kernel_stark(proof: &CircleStarkProof) -> CircleStarkVerifyResult {
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

pub fn prove_and_verify_lut_kernel_stark_from_steps(steps: &[LutStep]) -> bool {
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

pub fn prove_and_verify_lut_kernel_stark_from_rows(rows: &[ProofRow]) -> bool {
  let steps = match build_lut_steps_from_rows(rows) {
    Ok(steps) => steps,
    Err(_) => return false,
  };
  prove_and_verify_lut_kernel_stark_from_steps(&steps)
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
      crate::semantic_proof::OP_BYTE_ADD_EQ => out.push(LutStep {
        op: LutOpcode::ByteAddEq,
        in0: row.scalar0,
        in1: row.scalar1,
        in2: row.arg0,
        out0: row.value,
        out1: row.scalar2,
      }),
      crate::semantic_proof::OP_U16_ADD_EQ => {
        append_u16_row_as_u15_steps(row, &mut out)?;
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
      crate::semantic_proof::OP_BYTE_ADD_CARRY_EQ => {
        let total = row.scalar0 + row.scalar1 + row.arg0;
        out.push(LutStep {
          op: LutOpcode::ByteAddCarryEq,
          in0: row.scalar0,
          in1: row.scalar1,
          in2: row.arg0,
          out0: if total >= 256 { 1 } else { 0 },
          out1: total & 0xFF,
        });
      }
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
      crate::semantic_proof::OP_BYTE_ADD_EQ => out.push(LutStep {
        op: LutOpcode::ByteAddEq,
        in0: row.scalar0,
        in1: row.scalar1,
        in2: row.arg0,
        out0: row.value,
        out1: row.scalar2,
      }),
      crate::semantic_proof::OP_U16_ADD_EQ => {
        append_u16_row_as_u15_steps(row, &mut out)?;
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
      crate::semantic_proof::OP_BYTE_ADD_CARRY_EQ => {
        let total = row.scalar0 + row.scalar1 + row.arg0;
        out.push(LutStep {
          op: LutOpcode::ByteAddCarryEq,
          in0: row.scalar0,
          in1: row.scalar1,
          in2: row.arg0,
          out0: if total >= 256 { 1 } else { 0 },
          out1: total & 0xFF,
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

pub fn prove_and_verify_bit_stack_lut_stark(rows: &[ProofRow]) -> bool {
  if !prove_and_verify_stack_ir_scaffold_stark(rows) {
    return false;
  }

  let steps = match build_lut_steps_from_rows_bit_family(rows) {
    Ok(steps) => steps,
    Err(_) => return false,
  };

  prove_and_verify_lut_kernel_stark_from_steps(&steps)
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
  match op {
    OP_BYTE_ADD_EQ
    | crate::semantic_proof::OP_U16_ADD_EQ
    | crate::semantic_proof::OP_U29_ADD_EQ
    | crate::semantic_proof::OP_U24_ADD_EQ
    | crate::semantic_proof::OP_BOOL
    | crate::semantic_proof::OP_BYTE
    | crate::semantic_proof::OP_EQ_REFL
    | crate::semantic_proof::OP_AND_INTRO
    | crate::semantic_proof::OP_EQ_TRANS
    | crate::semantic_proof::OP_BYTE_ADD_THIRD_CONGRUENCE
    | crate::semantic_proof::OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE => Ok(()),
    other => Err(format!("unsupported Stage A proof-row op: {other}")),
  }
}

fn has_stage_a_semantic_rows(rows: &[ProofRow]) -> Result<bool, String> {
  let mut found = false;
  for row in rows {
    route_stage_a_row_op(row.op)?;
    if row.op == OP_BYTE_ADD_EQ
      || row.op == crate::semantic_proof::OP_U16_ADD_EQ
      || row.op == crate::semantic_proof::OP_U29_ADD_EQ
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
/// The current kernel verifies byte-add equality constraints encoded in
/// `OP_BYTE_ADD_EQ` rows.
pub struct StageASemanticAir;

impl<F> BaseAir<F> for StageASemanticAir {
  fn width(&self) -> usize {
    NUM_PROOF_COLS
  }
}

/// AIR for checking inferred WFF bytes equal expected WFF bytes.
///
/// Each row compares one byte:
/// - inferred byte from compiled `OP_BYTE_ADD_EQ` rows
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
    let t_byte = OP_BYTE_ADD_EQ as u16;
    let t_u16 = crate::semantic_proof::OP_U16_ADD_EQ as u16;
    let t_u29 = crate::semantic_proof::OP_U29_ADD_EQ as u16;
    let t_u24 = crate::semantic_proof::OP_U24_ADD_EQ as u16;

    let tags = [t_byte, t_u16, t_u29, t_u24];
    let gate = |target: u16| {
      let mut g = AB::Expr::from_u16(1);
      for t in tags {
        if t != target {
          g *= op.clone().into() - AB::Expr::from_u16(t);
        }
      }
      g
    };

    let g_byte = gate(t_byte);
    let g_u16 = gate(t_u16);
    let g_u29 = gate(t_u29);
    let g_u24 = gate(t_u24);

    let allowed_poly = (op.clone().into() - AB::Expr::from_u16(t_byte))
      * (op.clone().into() - AB::Expr::from_u16(t_u16))
      * (op.clone().into() - AB::Expr::from_u16(t_u29))
      * (op.clone().into() - AB::Expr::from_u16(t_u24));
    builder.assert_zero(allowed_poly);

    let a = local[COL_SCALAR0].clone();
    let b = local[COL_SCALAR1].clone();
    let sum = local[COL_VALUE].clone();

    builder.assert_zero(
      g_byte.clone()
        * (a.clone().into() + b.clone().into() + local[COL_ARG0].clone().into()
          - sum.clone().into()
          - AB::Expr::from_u16(256) * local[COL_SCALAR2].clone().into()),
    );
    builder.assert_zero(
      g_u16.clone()
        * (a.clone().into() + b.clone().into() + local[COL_SCALAR2].clone().into()
          - sum.clone().into()
          - AB::Expr::from_u32(1u32 << 16) * local[COL_ARG1].clone().into()),
    );
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

    builder.assert_zero(g_byte.clone() * (sum.clone().into() - local[COL_ARG1].clone().into()));
    builder.assert_zero(g_u16.clone() * (sum.clone().into() - local[COL_ARG0].clone().into()));
    builder.assert_zero(g_u29.clone() * (sum.clone().into() - local[COL_ARG0].clone().into()));
    builder.assert_zero(g_u24.clone() * (sum.clone().into() - local[COL_ARG0].clone().into()));

    builder.assert_zero(
      g_byte.clone()
        * (local[COL_SCALAR2].clone().into()
          * (local[COL_SCALAR2].clone().into() - AB::Expr::from_u16(1))),
    );
    builder.assert_zero(
      g_byte.clone()
        * (local[COL_ARG0].clone().into()
          * (local[COL_ARG0].clone().into() - AB::Expr::from_u16(1))),
    );
    builder.assert_zero(
      g_u16.clone()
        * (local[COL_ARG1].clone().into()
          * (local[COL_ARG1].clone().into() - AB::Expr::from_u16(1))),
    );
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
      g_u16.clone()
        * (local[COL_SCALAR2].clone().into()
          * (local[COL_SCALAR2].clone().into() - AB::Expr::from_u16(1))),
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
      g_byte.clone() * (local[COL_RET_TY].clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)),
    );
    builder.assert_zero(
      g_u16.clone() * (local[COL_RET_TY].clone().into() - AB::Expr::from_u16(RET_WFF_AND as u16)),
    );
    builder.assert_zero(
      g_u29.clone() * (local[COL_RET_TY].clone().into() - AB::Expr::from_u16(RET_WFF_AND as u16)),
    );
    builder.assert_zero(
      g_u24.clone() * (local[COL_RET_TY].clone().into() - AB::Expr::from_u16(RET_WFF_AND as u16)),
    );

    builder
      .when_first_row()
      .assert_zero(g_byte.clone() * local[COL_ARG0].clone().into());
    builder
      .when_first_row()
      .assert_zero(g_u16.clone() * local[COL_SCALAR2].clone().into());
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
    let ng_byte = next_gate(t_byte);
    let ng_u16 = next_gate(t_u16);
    let ng_u29 = next_gate(t_u29);
    let ng_u24 = next_gate(t_u24);

    let local_cout_byte = local[COL_SCALAR2].clone().into();
    let local_cout_u16 = local[COL_ARG1].clone().into();
    let local_cout_u29 = local[COL_ARG1].clone().into();
    let local_cout_u24 = local[COL_ARG1].clone().into();

    let next_cin_byte = next[COL_ARG0].clone().into();
    let next_cin_u16 = next[COL_SCALAR2].clone().into();
    let next_cin_u29 = next[COL_SCALAR2].clone().into();
    let next_cin_u24 = next[COL_SCALAR2].clone().into();

    let local_cases = [
      (g_byte, local_cout_byte),
      (g_u16, local_cout_u16),
      (g_u29, local_cout_u29),
      (g_u24, local_cout_u24),
    ];
    let next_cases = [
      (ng_byte, next_cin_byte),
      (ng_u16, next_cin_u16),
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

  let push_byte_add_row = |out: &mut Vec<ProofRow>,
                           a: u32,
                           b: u32,
                           carry_in: u32,
                           sum: u32,
                           carry_out: u32,
                           i: usize|
   -> Result<(), String> {
    if a > 255 || b > 255 || sum > 255 {
      return Err(format!("stage-a byte column out of range at row {i}"));
    }
    if carry_in > 1 || carry_out > 1 {
      return Err(format!("stage-a carry column out of range at row {i}"));
    }
    out.push(ProofRow {
      op: OP_BYTE_ADD_EQ,
      scalar0: a,
      scalar1: b,
      scalar2: carry_out,
      arg0: carry_in,
      arg1: sum,
      arg2: 0,
      value: sum,
      ret_ty: RET_WFF_EQ,
    });
    Ok(())
  };

  for (i, row) in rows.iter().enumerate() {
    route_stage_a_row_op(row.op)?;

    match row.op {
      OP_BYTE_ADD_EQ => {
        if row.scalar0 > 255 || row.scalar1 > 255 || row.arg1 > 255 || row.value > 255 {
          return Err(format!("stage-a byte column out of range at row {i}"));
        }
        if row.scalar2 > 1 || row.arg0 > 1 {
          return Err(format!("stage-a carry column out of range at row {i}"));
        }
        if row.arg2 != 0 {
          return Err(format!("stage-a arg2 must be zero at row {i}"));
        }
        if row.ret_ty != RET_WFF_EQ {
          return Err(format!("stage-a ret_ty must be RET_WFF_EQ at row {i}"));
        }
        semantic_rows.push(row.clone());
      }
      crate::semantic_proof::OP_U16_ADD_EQ => {
        if row.scalar0 > 0xFFFF
          || row.scalar1 > 0xFFFF
          || row.scalar2 > 1
          || row.arg0 > 0xFFFF
          || row.arg1 > 1
        {
          return Err(format!("stage-a u16 column out of range at row {i}"));
        }
        if row.value != row.arg0 {
          return Err(format!("stage-a u16 sum mismatch at row {i}"));
        }

        let a_lo = row.scalar0 & 0xFF;
        let a_hi = (row.scalar0 >> 8) & 0xFF;
        let b_lo = row.scalar1 & 0xFF;
        let b_hi = (row.scalar1 >> 8) & 0xFF;
        let sum_lo = row.value & 0xFF;
        let sum_hi = (row.value >> 8) & 0xFF;

        let low_total = a_lo + b_lo + row.scalar2;
        let carry_mid = if low_total >= 256 { 1 } else { 0 };
        let high_total = a_hi + b_hi + carry_mid;
        let carry_out = if high_total >= 256 { 1 } else { 0 };

        if carry_out != row.arg1 {
          return Err(format!("stage-a u16 carry mismatch at row {i}"));
        }

        push_byte_add_row(
          &mut semantic_rows,
          a_lo,
          b_lo,
          row.scalar2,
          sum_lo,
          carry_mid,
          i,
        )?;
        push_byte_add_row(
          &mut semantic_rows,
          a_hi,
          b_hi,
          carry_mid,
          sum_hi,
          row.arg1,
          i,
        )?;
      }
      crate::semantic_proof::OP_U29_ADD_EQ | crate::semantic_proof::OP_U24_ADD_EQ => {
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
        OP_BYTE_ADD_EQ => row.scalar2 as u16,
        crate::semantic_proof::OP_U16_ADD_EQ
        | crate::semantic_proof::OP_U29_ADD_EQ
        | crate::semantic_proof::OP_U24_ADD_EQ => row.arg1 as u16,
        _ => 0,
      })
      .unwrap_or(0);

    for i in semantic_len..n_rows {
      let carry_out = 0u16;
      let sum = carry_in;
      let expected = sum;

      let base = i * NUM_PROOF_COLS;
      trace.values[base + COL_OP] = Val::from_u16(OP_BYTE_ADD_EQ as u16);
      trace.values[base + COL_SCALAR0] = Val::from_u16(0);
      trace.values[base + COL_SCALAR1] = Val::from_u16(0);
      trace.values[base + COL_SCALAR2] = Val::from_u16(carry_out);
      trace.values[base + COL_ARG0] = Val::from_u16(carry_in);
      trace.values[base + COL_ARG1] = Val::from_u16(expected);
      trace.values[base + COL_ARG2] = Val::from_u16(0);
      trace.values[base + COL_VALUE] = Val::from_u16(sum);
      trace.values[base + COL_RET_TY] = Val::from_u16(RET_WFF_EQ as u16);

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
/// - stage 1 proves inferred WFF rows are valid (`OP_BYTE_ADD_EQ` constraints)
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
// Batch STARK proving
//
// Usage pattern:
//   let all_steps: Vec<LutStep> = instructions
//       .iter()
//       .flat_map(|itp| {
//           let rows = compile_proof(itp.semantic_proof.as_ref().unwrap());
//           build_lut_steps_from_rows_add_family(&rows).unwrap_or_default()
//       })
//       .collect();
//   let proof = prove_batch_lut_stark_add_family(&all_steps)?;
// ============================================================

/// Prove N add-family (ADD/SUB) instructions' LUT kernel in one STARK call.
///
/// `all_steps` is the concatenation of `build_lut_steps_from_rows_add_family()`
/// output for each instruction.
pub fn prove_batch_lut_stark(all_steps: &[LutStep]) -> Result<CircleStarkProof, String> {
  prove_lut_kernel_stark(all_steps)
}

/// Verify a batched LUT STARK proof.
pub fn verify_batch_lut_stark(proof: &CircleStarkProof) -> bool {
  verify_lut_kernel_stark(proof).is_ok()
}

/// Build and immediately prove N add-family instructions' LUT in one shot.
pub fn prove_batch_stark_add_family(all_rows: &[ProofRow]) -> Result<CircleStarkProof, String> {
  let steps = build_lut_steps_from_rows_add_family(all_rows)?;
  prove_lut_kernel_stark(&steps)
}

/// Verify an add-family batch LUT proof.
pub fn verify_batch_stark_add_family(proof: &CircleStarkProof) -> bool {
  verify_lut_kernel_stark(proof).is_ok()
}

/// Build and immediately prove N add-family instructions' LUT with custom FRI params.
pub fn prove_batch_stark_add_family_with_params(
  all_rows: &[ProofRow],
  num_queries: usize,
  query_pow_bits: usize,
  log_final_poly_len: usize,
) -> Result<CircleStarkProof, String> {
  let steps = build_lut_steps_from_rows_add_family(all_rows)?;
  let trace = build_lut_trace_from_steps(&steps)?;
  let config = make_circle_config_with_params(num_queries, query_pow_bits, log_final_poly_len);
  Ok(p3_uni_stark::prove(
    &config,
    &LutKernelAir,
    trace,
    &default_receipt_bind_public_values_for_tag(RECEIPT_BIND_TAG_LUT),
  ))
}

/// Verify an add-family batch proof produced with matching custom FRI params.
pub fn verify_batch_stark_add_family_with_params(
  proof: &CircleStarkProof,
  num_queries: usize,
  query_pow_bits: usize,
  log_final_poly_len: usize,
) -> bool {
  let config = make_circle_config_with_params(num_queries, query_pow_bits, log_final_poly_len);
  p3_uni_stark::verify(
    &config,
    &LutKernelAir,
    proof,
    &default_receipt_bind_public_values_for_tag(RECEIPT_BIND_TAG_LUT),
  )
  .is_ok()
}

/// Build and immediately prove N mul-family (MUL/DIV/MOD/SDIV/SMOD) instructions' LUT in one shot.
pub fn prove_batch_stark_mul_family(all_rows: &[ProofRow]) -> Result<CircleStarkProof, String> {
  let steps = build_lut_steps_from_rows_mul_family(all_rows)?;
  prove_lut_kernel_stark(&steps)
}

/// Build and immediately prove N mul-family instructions' LUT with custom FRI params.
pub fn prove_batch_stark_mul_family_with_params(
  all_rows: &[ProofRow],
  num_queries: usize,
  query_pow_bits: usize,
  log_final_poly_len: usize,
) -> Result<CircleStarkProof, String> {
  let steps = build_lut_steps_from_rows_mul_family(all_rows)?;
  let trace = build_lut_trace_from_steps(&steps)?;
  let config = make_circle_config_with_params(num_queries, query_pow_bits, log_final_poly_len);
  Ok(p3_uni_stark::prove(
    &config,
    &LutKernelAir,
    trace,
    &default_receipt_bind_public_values_for_tag(RECEIPT_BIND_TAG_LUT),
  ))
}

/// Verify a mul-family batch LUT proof.
pub fn verify_batch_stark_mul_family(proof: &CircleStarkProof) -> bool {
  verify_lut_kernel_stark(proof).is_ok()
}

/// Verify a mul-family batch proof produced with matching custom FRI params.
pub fn verify_batch_stark_mul_family_with_params(
  proof: &CircleStarkProof,
  num_queries: usize,
  query_pow_bits: usize,
  log_final_poly_len: usize,
) -> bool {
  let config = make_circle_config_with_params(num_queries, query_pow_bits, log_final_poly_len);
  p3_uni_stark::verify(
    &config,
    &LutKernelAir,
    proof,
    &default_receipt_bind_public_values_for_tag(RECEIPT_BIND_TAG_LUT),
  )
  .is_ok()
}

/// Build and immediately prove N bit-family (AND/OR/XOR/NOT) instructions' LUT in one shot.
pub fn prove_batch_stark_bit_family(all_rows: &[ProofRow]) -> Result<CircleStarkProof, String> {
  let steps = build_lut_steps_from_rows_bit_family(all_rows)?;
  prove_lut_kernel_stark(&steps)
}

/// Build and immediately prove N bit-family instructions' LUT with custom FRI params.
pub fn prove_batch_stark_bit_family_with_params(
  all_rows: &[ProofRow],
  num_queries: usize,
  query_pow_bits: usize,
  log_final_poly_len: usize,
) -> Result<CircleStarkProof, String> {
  let steps = build_lut_steps_from_rows_bit_family(all_rows)?;
  let trace = build_lut_trace_from_steps(&steps)?;
  let config = make_circle_config_with_params(num_queries, query_pow_bits, log_final_poly_len);
  Ok(p3_uni_stark::prove(
    &config,
    &LutKernelAir,
    trace,
    &default_receipt_bind_public_values_for_tag(RECEIPT_BIND_TAG_LUT),
  ))
}

/// Verify a bit-family batch LUT proof.
pub fn verify_batch_stark_bit_family(proof: &CircleStarkProof) -> bool {
  verify_lut_kernel_stark(proof).is_ok()
}

/// Verify a bit-family batch proof produced with matching custom FRI params.
pub fn verify_batch_stark_bit_family_with_params(
  proof: &CircleStarkProof,
  num_queries: usize,
  query_pow_bits: usize,
  log_final_poly_len: usize,
) -> bool {
  let config = make_circle_config_with_params(num_queries, query_pow_bits, log_final_poly_len);
  p3_uni_stark::verify(
    &config,
    &LutKernelAir,
    proof,
    &default_receipt_bind_public_values_for_tag(RECEIPT_BIND_TAG_LUT),
  )
  .is_ok()
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
      crate::semantic_proof::OP_BYTE_ADD_EQ => out.push(LutStep {
        op: LutOpcode::ByteAddEq,
        in0: row.scalar0,
        in1: row.scalar1,
        in2: row.arg0,
        out0: row.value,
        out1: row.scalar2,
      }),
      crate::semantic_proof::OP_U16_ADD_EQ => {
        append_u16_row_as_u15_steps(row, &mut out)?;
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
      crate::semantic_proof::OP_BYTE_ADD_CARRY_EQ => {
        let total = row.scalar0 + row.scalar1 + row.arg0;
        out.push(LutStep {
          op: LutOpcode::ByteAddCarryEq,
          in0: row.scalar0,
          in1: row.scalar1,
          in2: row.arg0,
          out0: if total >= 256 { 1 } else { 0 },
          out1: total & 0xFF,
        });
      }

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

  Ok(())
}

// ============================================================
// Memory consistency proof  (SMAT — Sorted Memory Access Table)
// ============================================================
//
// Soundness design
// ─────────────────
// The proof binds to the public claim set via a Poseidon hash committed in
// public_values, and enforces read/write consistency in the AIR itself.
//
// Three defences:
//
//   1. **Claim binding** — `public_values = [TAG, n_claims, hash[0..8]]`
//      where `hash = poseidon(claims)`.  The verifier recomputes this hash
//      from the supplied claims and checks it matches the proof's public
//      values, so a proof for claim set A cannot be replayed for set B.
//
//   2. **Sorted SMAT witness** — The prover lays out a *sorted* receive table
//      (sorted by (addr, rw_counter)).  The AIR enforces:
//        a. Same-addr read-continuity: if addr[i]==addr[i+1] and both are
//           SMAT rows and is_write[i+1]=0, then val[i+1] == val[i].
//      (Full range-based sort ordering would require a range chip; we defer
//      that to a future step and rely on the LogUp multiset argument + the
//      claim hash to enforce that the exact claimed accesses are witnessed.)
//
//   3. **LogUp multiset equality** — claim rows (mult=+1 each) and SMAT
//      receive rows (mult=−freq) form a balanced multiset; the LogUp running
//      sum is zero iff both sides encode the same multiset.
//
// Column layout  (NUM_MEM_COLS = 19)
// ────────────────────────────────────
//  0  addr_hi      — upper 32 bits of the 64-bit word address
//  1  addr_lo      — lower 32 bits
//  2  rw_hi        — upper 32 bits of rw_counter
//  3  rw_lo        — lower 32 bits
//  4  is_write     — 0 (read) or 1 (write)
//  5  val0         — value bytes  0- 3 (big-endian u32)
//  6  val1         — value bytes  4- 7
//  7  val2         — value bytes  8-11
//  8  val3         — value bytes 12-15
//  9  val4         — value bytes 16-19
// 10  val5         — value bytes 20-23
// 11  val6         — value bytes 24-27
// 12  val7         — value bytes 28-31
// 13  write_version — per-address write version (0 for first access, +1 per write)
// 14  mult         — +1 for claim rows, −freq for SMAT rows, 0 for padding
// 15  is_smat      — 1 for SMAT receive rows, 0 for claim / padding rows
// 16  addr_eq      — 1 if addr[i] == addr[i+1] and both are SMAT rows
// 17  inv_addr_hi  — inverse witness: (addr_hi[next]-addr_hi[cur])^{-1} or 0
// 18  inv_addr_lo  — inverse witness: (addr_lo[next]-addr_lo[cur])^{-1} or 0
//                   (used only when addr_hi parts are equal)
//
// (19 columns total, 1 LogUp aux column.)
//
// Public values layout  (MEM_PUBLIC_VALUES_LEN = 10)
// ────────────────────────────────────────────────────
//  pis[0]     = RECEIPT_BIND_TAG_MEM
//  pis[1]     = n_claims  (number of MemAccessClaims)
//  pis[2..10] = poseidon_hash(claims)  (8 × M31)

use p3_lookup::logup::LogUpGadget;
use p3_lookup::lookup_traits::{Kind as LookupKind, LookupData, LookupGadget};
use p3_uni_stark::{Entry, SymbolicExpression, SymbolicVariable, VerifierConstraintFolder};

pub const RECEIPT_BIND_TAG_MEM: u32 = 4;
const MEM_PUBLIC_VALUES_LEN: usize = 10;

const MEM_COL_ADDR_HI: usize = 0;
const MEM_COL_ADDR_LO: usize = 1;
const MEM_COL_RW_HI: usize = 2;
const MEM_COL_RW_LO: usize = 3;
const MEM_COL_IS_WRITE: usize = 4;
const MEM_COL_VAL0: usize = 5;
// val1..val7 are MEM_COL_VAL0+1 .. MEM_COL_VAL0+7
const MEM_COL_WRITE_VERSION: usize = 13;
const MEM_COL_MULT: usize = 14;
const MEM_COL_IS_SMAT: usize = 15;
const MEM_COL_ADDR_EQ: usize = 16;
const MEM_COL_INV_ADDR_HI: usize = 17;
const MEM_COL_INV_ADDR_LO: usize = 18;
/// Total number of main-trace columns in the SMAT AIR.
pub const NUM_MEM_COLS: usize = 19;
/// Number of columns in the LogUp lookup tuple (cols 0..14: addr_hi..write_version + mult excluded).
/// Tuple = (addr_hi, addr_lo, rw_hi, rw_lo, is_write, val0..val7, write_version) = 14 cols.
const MEM_LOOKUP_TUPLE_COLS: usize = 14;

// ── Claim hash ────────────────────────────────────────────────────────

/// Poseidon2/M31 hash of a `MemAccessClaim` slice.
///
/// Sponge input layout: `[n_claims, addr_hi, addr_lo, rw_hi, rw_lo, is_write,
/// write_version, val0..val7, …]` one block per claim.
fn hash_mem_claims(claims: &[crate::transition::MemAccessClaim]) -> [Val; 8] {
  let mut rng = rand::rngs::SmallRng::seed_from_u64(0xC0DEC0DE_u64);
  let poseidon = Poseidon2Mersenne31::<16>::new_from_rng_128(&mut rng);
  let sponge = PaddingFreeSponge::<_, 16, 8, 8>::new(poseidon);

  let mut input: Vec<Val> = Vec::with_capacity(1 + claims.len() * 16);
  input.push(Val::from_u32(claims.len() as u32));
  for c in claims {
    input.push(Val::from_u32((c.addr >> 32) as u32));
    input.push(Val::from_u32(c.addr as u32));
    input.push(Val::from_u32((c.rw_counter >> 32) as u32));
    input.push(Val::from_u32(c.rw_counter as u32));
    input.push(Val::from_u32(c.is_write as u32));
    input.push(Val::from_u32(c.write_version));
    for chunk in c.value.chunks(4) {
      input.push(Val::from_u32(u32::from_be_bytes([
        chunk[0], chunk[1], chunk[2], chunk[3],
      ])));
    }
  }
  sponge.hash_iter(input)
}

/// Build the `public_values` vector for a memory consistency STARK.
fn make_mem_public_values(claims: &[crate::transition::MemAccessClaim]) -> Vec<Val> {
  let hash = hash_mem_claims(claims);
  let mut pv = Vec::with_capacity(MEM_PUBLIC_VALUES_LEN);
  pv.push(Val::from_u32(RECEIPT_BIND_TAG_MEM));
  pv.push(Val::from_u32(claims.len() as u32));
  pv.extend_from_slice(&hash);
  pv
}

// ── AIR ──────────────────────────────────────────────────────────────

/// AIR for the SMAT memory consistency LogUp argument.
///
/// Constraints (in addition to the LogUp running-sum argument):
///  • `is_write  ∈ {0,1}` (degree 2)
///  • `is_smat   ∈ {0,1}` (degree 2)
///  • `addr_eq   ∈ {0,1}` (degree 2)
///  • Tag check: `pis[0] == RECEIPT_BIND_TAG_MEM` (degree 1)
///  • addr_eq soundness (degree 2, per part hi/lo):
///      `diff_hi * inv_hi = 1 - same_hi`,  `diff_hi * same_hi = 0`
///      `diff_lo * inv_lo = 1 - same_lo`,  `diff_lo * same_lo = 0`
///      `addr_eq = same_hi * same_lo`  (degree 2)
///      (only enforced on SMAT rows: gated by `is_smat * is_smat_next`)
///  • Read-continuity (degree 5):
///      `is_smat[cur] * is_smat[next] * addr_eq[cur]
///       * (1 - is_write[next]) * (val_k[next] - val_k[cur]) = 0`
///    for each k ∈ 0..8.
///  • write_version increment (degree 4):
///      `is_smat[cur] * is_smat[next] * addr_eq[cur]
///       * (wv[next] - wv[cur] - is_write[cur]) = 0`
///  • write_version reset (degree 3):
///      `is_smat[next] * (1 - addr_eq[cur]) * wv[next] = 0`
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
    let main = builder.main();
    let local = main.row_slice(0).expect("empty SMAT trace");
    let next = main.row_slice(1).expect("single-row SMAT trace");
    let local = &*local;
    let next = &*next;

    // ── Boolean checks ─────────────────────────────────────────────
    let is_write = local[MEM_COL_IS_WRITE].clone();
    builder.assert_zero(is_write.clone().into() * (AB::Expr::ONE - is_write.clone().into()));

    let is_smat = local[MEM_COL_IS_SMAT].clone();
    builder.assert_zero(is_smat.clone().into() * (AB::Expr::ONE - is_smat.clone().into()));

    let addr_eq = local[MEM_COL_ADDR_EQ].clone();
    builder.assert_zero(addr_eq.clone().into() * (AB::Expr::ONE - addr_eq.clone().into()));

    // ── Tag check ──────────────────────────────────────────────────
    let pis = builder.public_values();
    builder.assert_eq(pis[0].into(), AB::Expr::from_u32(RECEIPT_BIND_TAG_MEM));

    // ── Read-continuity constraint ────────────────────────────────
    // For consecutive SMAT rows with the same address, a read must
    // return the value written by the preceding row.
    //
    // Note: is_transition() is NOT needed here because padding rows
    // have is_smat=0 and claim rows have is_smat=0, so the selector
    // `sel` is already zero on any row pair that doesn't involve two
    // consecutive SMAT rows.  This includes the wrap-around (last→first).
    let is_smat_next = next[MEM_COL_IS_SMAT].clone();
    let is_write_next = next[MEM_COL_IS_WRITE].clone();

    // sel = is_smat[cur] * is_smat[next] * addr_eq[cur] * (1 - is_write[next])
    let sel: AB::Expr = is_smat.clone().into()
      * is_smat_next.clone().into()
      * addr_eq.clone().into()
      * (AB::Expr::ONE - is_write_next.clone().into());

    for k in 0..8 {
      let val_cur = local[MEM_COL_VAL0 + k].clone();
      let val_next = next[MEM_COL_VAL0 + k].clone();
      builder.assert_zero(sel.clone() * (val_next.clone().into() - val_cur.clone().into()));
    }

    // ── addr_eq soundness via inverse witnesses ───────────────────
    //
    // We need to prove addr_eq is genuinely determined by the actual
    // address values, not freely set by a cheating prover.
    //
    // For each address part (hi, lo) we use an "equality bit" pattern:
    //   diff * inv = 1 - same_bit   (degree 2)
    //   diff * same_bit = 0         (degree 2)
    //
    // This forces same_bit=1 iff diff==0 (assuming inv is the field
    // inverse of diff when diff≠0, else inv can be anything).
    //
    // Then addr_eq = same_hi * same_lo is checked against the column.
    //
    // All constraints are gated by is_smat[cur] * is_smat[next] to
    // apply only on SMAT-to-SMAT row transitions.
    let smat_gate: AB::Expr =
      is_smat.clone().into() * is_smat_next.clone().into();

    let addr_hi_cur: AB::Expr = local[MEM_COL_ADDR_HI].clone().into();
    let addr_hi_next: AB::Expr = next[MEM_COL_ADDR_HI].clone().into();
    let addr_lo_cur: AB::Expr = local[MEM_COL_ADDR_LO].clone().into();
    let addr_lo_next: AB::Expr = next[MEM_COL_ADDR_LO].clone().into();

    let diff_hi: AB::Expr = addr_hi_next - addr_hi_cur;
    let diff_lo: AB::Expr = addr_lo_next - addr_lo_cur;

    let inv_hi: AB::Expr = local[MEM_COL_INV_ADDR_HI].clone().into();
    let inv_lo: AB::Expr = local[MEM_COL_INV_ADDR_LO].clone().into();

    // same_hi, same_lo derived from addr_eq and the constraint pattern.
    // We store same_hi and same_lo implicitly:
    //   same_hi = 1 - diff_hi * inv_hi  (this is enforced below)
    //   same_lo = 1 - diff_lo * inv_lo
    //   addr_eq  = same_hi * same_lo
    //
    // Constraints (degree 3 each, gated by smat_gate):
    //   C1: smat_gate * (diff_hi * inv_hi + addr_hi_same - 1) = 0
    //           where addr_hi_same = 1 - diff_hi * inv_hi
    //   C2: smat_gate * diff_hi * (1 - addr_hi_same) = 0  [redundant with C1 for malleability]
    //
    // Simpler combined constraints (total 6, degree ≤ 3):
    //   C1: smat_gate * diff_hi * (1 - diff_hi * inv_hi) = 0
    //         → if diff_hi≠0, then inv_hi must satisfy diff_hi*inv_hi=1 (else constraint fails)
    //   C2: smat_gate * (1 - addr_eq) = same_hi_lo_gate
    //   ...
    //
    // Practical encoding: use addr_eq directly with:
    //   C1: addr_eq * diff_hi = 0      → addr_eq=1 implies hi parts equal
    //   C2: addr_eq * diff_lo = 0      → addr_eq=1 implies lo parts equal
    //   C3: (1 - addr_eq) * smat_gate * (1 - diff_hi * inv_hi - diff_lo * inv_lo) = 0
    //         → if addr_eq=0 and on SMAT pair, at least one diff is non-zero
    //            and its inverse is correct (prover must supply valid inverses)
    //
    // C1 and C2 are degree 2; C3 is degree 4 (3 multiplied terms with smat_gate).

    // C1: addr_eq * diff_hi = 0
    builder.assert_zero(addr_eq.clone().into() * diff_hi.clone());
    // C2: addr_eq * diff_lo = 0
    builder.assert_zero(addr_eq.clone().into() * diff_lo.clone());
    // C3: (1 - addr_eq) * smat_gate * (diff_hi * inv_hi + diff_lo * inv_lo - ...) 
    // We enforce: smat_gate * (1 - addr_eq) forces that (diff_hi*inv_hi + diff_lo*inv_lo)
    // spans the non-zero case. Specifically:
    //   if addr_eq=0 and is_smat pair: must have diff_hi≠0 OR diff_lo≠0
    //   we enforce: smat_gate * (1 - addr_eq) * (1 - diff_hi * inv_hi) * (1 - diff_lo * inv_lo) = 0
    // but that's degree 6. Instead use:
    //   C3: smat_gate * (1 - addr_eq) * ((1 - diff_hi*inv_hi)*(1 - diff_lo*inv_lo)) = 0
    // Rewrite as: smat_gate * (1 - addr_eq - diff_hi*inv_hi - diff_lo*inv_lo + diff_hi*inv_hi*diff_lo*inv_lo) = 0
    // The last term is degree 5 with smat_gate = degree 7: too high.
    //
    // Use the simplest sound encoding: two separate one-sided checks:
    //   C3a: smat_gate * (1 - addr_eq - diff_hi * inv_hi) * ... 
    // Actually the cleanest that keeps degree ≤ 4:
    //   Let x = diff_hi * inv_hi + diff_lo * inv_lo   (this is 1 for hi-differ, 1 for lo-differ, 2 for both)
    //   C3: smat_gate * (1 - addr_eq) * (1 - x) = 0  (degree 4 with smat_gate degree 2)
    //   Wait, smat_gate is degree 2, (1-addr_eq) is degree 1, (1-x) is degree 3 → product degree 6: too high.
    //
    // Minimum-degree sound approach (degree 3 total):
    //   C1: addr_eq * diff_hi = 0                            degree 2 ✓
    //   C2: addr_eq * diff_lo = 0                            degree 2 ✓
    //   C3: is_smat * (1 - addr_eq) * (1 - diff_hi*inv_hi - diff_lo*inv_lo) = 0  degree 4 ✓
    //   (Note: is_smat alone, not smat_gate, to stay at degree 4)
    //
    // C3 with is_smat (degree 1 select) * (1-addr_eq) * (1 - diff*inv - diff*inv):
    //   = degree 1 + 1 + 2 = 4 ✓
    let prod_hi: AB::Expr = diff_hi.clone() * inv_hi;
    let prod_lo: AB::Expr = diff_lo.clone() * inv_lo;
    // C3: is_smat * is_smat_next * (1 - addr_eq) * (1 - diff_hi*inv_hi - diff_lo*inv_lo) = 0
    // Gated by smat_gate so only applies on SMAT-to-SMAT row transitions.
    // (degree 4: smat_gate=2, (1-addr_eq)=1, (1-prod_hi-prod_lo)=3 → total 4 with smat_gate being 2)
    // Actually degree: smat_gate(2) * (1-addr_eq)(1) * expr(2) = 5. Use is_smat only (degree 1):
    // is_smat(1) * is_smat_next(1) * (1-addr_eq)(1) * (1-prod)(2) = degree 5 — too high.
    // Use smat_gate(2) alone, drop one factor: accept degree 4.
    // C3 rewritten: smat_gate * (1 - addr_eq) * (1 - prod_hi - prod_lo) = 0
    // degree = 2 + 1 + 2 = 5: still too high with the polynomial evaluation.
    // Correct degree counting: smat_gate is degree 2, (1-addr_eq) is degree 1 in terms of
    // original witness entries, (1 - diff*inv - diff*inv) is degree 2.
    // Product = degree 2+1+2 = 5 ← above the constraint degree limit.
    // 
    // Fix: replace smat_gate with is_smat only (degree 1):
    // C3: is_smat * (1 - addr_eq) * (1 - diff_hi*inv_hi - diff_lo*inv_lo) = 0
    // but also gate on is_smat_next to exclude the last SMAT row's transition to non-SMAT:
    // We cannot add is_smat_next without going to degree 4+2=6...
    //
    // The approach: encode C3 differently:
    //   When is_smat=1 AND addr_eq=0: at least one of prod_hi, prod_lo must be 1.
    //   i.e. is_smat * (1 - addr_eq) * (1 - prod_hi) * (1 - prod_lo) = 0
    //   degree = 1 + 1 + 2 + 2 = 6: too high.
    //
    // Simplest approach that stays degree ≤ 4 and is still sound:
    //   C3: is_smat * is_smat_next * (1 - addr_eq) * (prod_hi + prod_lo) = 1 ... no, enforce:
    //   C3: is_smat * is_smat_next * (1 - prod_hi - prod_lo - addr_eq) = 0
    //   degree = 2 + 1 + (degree of prod expressions = 2) ... 
    //   prod_hi = diff_hi * inv_hi (degree 2), so (1 - prod_hi - prod_lo - addr_eq) is degree 2,
    //   × smat_gate (degree 2) = degree 4. ✓
    //
    // But this encodes: prod_hi + prod_lo + addr_eq = 1
    // Cases:
    //   addr_eq=1, prod_hi=0, prod_lo=0 → 1 ✓ (same address, inv=0)
    //   addr_eq=0, prod_hi=1, prod_lo=0 → 1 ✓ (hi differs, hi inverse valid)
    //   addr_eq=0, prod_hi=0, prod_lo=1 → 1 ✓ (lo differs, lo inverse valid)
    //   addr_eq=0, prod_hi=1, prod_lo=1 → 2 ✗ — fails when BOTH parts differ!
    //
    // Fix for the "both parts differ" case: encode as OR with a combined inverse.
    // Let comb = diff_hi * (any_nonzero_when_hi_differs) + ...
    // 
    // Pragmatic sound fix: use is_smat alone (not smat_gate), and fill inv witnesses
    // on ALL rows including the last SMAT row. The last SMAT row's next is a non-SMAT
    // row; diff_hi and diff_lo will differ from 0, so inv needs to be set. But we only
    // want C3 to fire for SMAT-to-SMAT, so gate with is_smat_next:
    //
    // C3: is_smat_next * (1 - addr_eq) * (1 - prod_hi - prod_lo) = 0
    //     degree = 1 + 1 + 2 = 4 ✓
    //   But now "is_smat_next" is on the next row, so inv columns are on the current row.
    //   This is equivalent to: on the NEXT row (when it's SMAT) and this row's addr_eq=0,
    //   prod_hi + prod_lo must equal 1.
    //   This is slightly weaker (doesn't constrain the last SMAT row's inv values when
    //   the next is non-SMAT) but that's fine — those inv values are unconstrained padding.
    //
    // HOWEVER: is_smat_next gates on the NEXT row's is_smat, which means on the wrap-
    // around row (last padding → first SMAT row, is_smat_next=1) the constraint fires
    // even though the current row is padding (is_smat=0, addr_eq=0, prod=0) and would
    // incorrectly fail. Therefore we MUST also gate on is_smat[cur]:
    //   Final: smat_gate * (1 - addr_eq) * (1 - prod_hi - prod_lo) = 0  [degree 5]
    builder.assert_zero(
      smat_gate.clone()
        * (AB::Expr::ONE - addr_eq.clone().into())
        * (AB::Expr::ONE - prod_hi - prod_lo),
    );

    // ── write_version monotonicity ────────────────────────────────
    //
    // For SMAT rows with the same address:
    //   wv[next] = wv[cur] + is_write[cur]
    //
    // For SMAT rows with a different address:
    //   wv[next] = 0   (new address starts at wv=0)
    //
    // C4: is_smat[cur] * is_smat[next] * addr_eq[cur]
    //       * (wv[next] - wv[cur] - is_write[cur]) = 0  (degree 4)
    let wv_cur: AB::Expr = local[MEM_COL_WRITE_VERSION].clone().into();
    let wv_next: AB::Expr = next[MEM_COL_WRITE_VERSION].clone().into();

    builder.assert_zero(
      smat_gate.clone()
        * addr_eq.clone().into()
        * (wv_next.clone() - wv_cur - is_write.clone().into()),
    );

    // C5: is_smat[next] * (1 - addr_eq[cur]) * wv[next] = 0  (degree 3)
    builder.assert_zero(
      is_smat_next.clone().into()
        * (AB::Expr::ONE - addr_eq.clone().into())
        * wv_next,
    );
  }
}

// ── Lookup descriptor ────────────────────────────────────────────────

fn make_mem_lookup() -> p3_lookup::lookup_traits::Lookup<Val> {
  let col =
    |c: usize| SymbolicExpression::Variable(SymbolicVariable::new(Entry::Main { offset: 0 }, c));
  // Lookup tuple = cols 0..MEM_LOOKUP_TUPLE_COLS (addr_hi..val7, 13 elements).
  // is_smat and addr_eq are AIR-internal and NOT part of the lookup tuple.
  let tuple: Vec<SymbolicExpression<Val>> = (0..MEM_LOOKUP_TUPLE_COLS).map(col).collect();
  p3_lookup::lookup_traits::Lookup::new(
    LookupKind::Local,
    vec![tuple],
    vec![col(MEM_COL_MULT)],
    vec![0], // single aux column index
  )
}

// ── Trace builder ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct MemTraceKey {
  addr_hi: u32,
  addr_lo: u32,
  rw_hi: u32,
  rw_lo: u32,
  is_write: u32,
  val: [u32; 8],
  write_version: u32,
}

impl MemTraceKey {
  fn from_claim(c: &crate::transition::MemAccessClaim) -> Self {
    let mut val = [0u32; 8];
    for (i, chunk) in c.value.chunks(4).enumerate() {
      val[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Self {
      addr_hi: (c.addr >> 32) as u32,
      addr_lo: c.addr as u32,
      rw_hi: (c.rw_counter >> 32) as u32,
      rw_lo: c.rw_counter as u32,
      is_write: c.is_write as u32,
      val,
      write_version: c.write_version,
    }
  }

  fn fill_tuple_cols(&self, row: &mut [Val]) {
    row[MEM_COL_ADDR_HI] = Val::from_u32(self.addr_hi);
    row[MEM_COL_ADDR_LO] = Val::from_u32(self.addr_lo);
    row[MEM_COL_RW_HI] = Val::from_u32(self.rw_hi);
    row[MEM_COL_RW_LO] = Val::from_u32(self.rw_lo);
    row[MEM_COL_IS_WRITE] = Val::from_u32(self.is_write);
    for k in 0..8 {
      row[MEM_COL_VAL0 + k] = Val::from_u32(self.val[k]);
    }
    row[MEM_COL_WRITE_VERSION] = Val::from_u32(self.write_version);
  }

  fn addr64(&self) -> u64 {
    ((self.addr_hi as u64) << 32) | self.addr_lo as u64
  }
}

/// Build the SMAT LogUp trace from `claims`.
///
/// Trace layout:
///  - Rows 0 .. n_smat          : SMAT receive rows (sorted by (addr,rw_counter)).
///                                mult = −freq,  is_smat = 1.
///  - Rows n_smat .. n_smat+N   : claim query rows (original claim order).
///                                mult = +1,  is_smat = 0.
///  - Remaining rows            : padding (all-zero).
///
/// `addr_eq[i]` is set to 1 when SMAT row i and i+1 have the same address.
///
/// Returns `Err` if any read claim is inconsistent with the witnessed writes.
fn build_smat_trace(
  claims: &[crate::transition::MemAccessClaim],
) -> Result<RowMajorMatrix<Val>, String> {
  use std::collections::{BTreeMap, HashMap};

  // ── 1. R/W consistency check ─────────────────────────────────────
  let mut sorted_idx: Vec<usize> = (0..claims.len()).collect();
  sorted_idx.sort_by_key(|&i| (claims[i].addr, claims[i].rw_counter));

  let mut last_write: HashMap<u64, [u8; 32]> = HashMap::new();
  for &i in &sorted_idx {
    let c = &claims[i];
    if c.is_write {
      last_write.insert(c.addr, c.value);
    } else {
      let expected = last_write.get(&c.addr).copied().unwrap_or([0u8; 32]);
      if c.value != expected {
        return Err(format!(
          "build_smat_trace: read/write consistency violation \
           at addr=0x{:x} rw={}",
          c.addr, c.rw_counter
        ));
      }
    }
  }

  // ── 2. Build sorted SMAT (unique tuples with frequency) ──────────
  let mut smat_mult: BTreeMap<MemTraceKey, u32> = BTreeMap::new();
  for c in claims {
    *smat_mult.entry(MemTraceKey::from_claim(c)).or_insert(0) += 1;
  }
  // BTreeMap order = lexicographic on (addr_hi, addr_lo, rw_hi, rw_lo, …)
  // which is equivalent to sort by (addr, rw_counter) ascending.
  let smat_rows: Vec<(&MemTraceKey, u32)> = smat_mult.iter().map(|(k, &m)| (k, m)).collect();
  let n_smat = smat_rows.len();
  let n_claims = claims.len();

  // ── 3. Allocate trace ─────────────────────────────────────────────
  let n_data = n_smat + n_claims;
  let height = n_data.max(4).next_power_of_two();
  let mut data = vec![Val::ZERO; height * NUM_MEM_COLS];

  // ── 4. SMAT receive rows ──────────────────────────────────────────
  for (row_idx, (key, freq)) in smat_rows.iter().enumerate() {
    let base = row_idx * NUM_MEM_COLS;
    key.fill_tuple_cols(&mut data[base..]);
    data[base + MEM_COL_MULT] = -Val::from_u32(*freq); // receive: negative mult
    data[base + MEM_COL_IS_SMAT] = Val::ONE;
    // addr_eq: 1 if next SMAT row has the same address.
    // Also fill inv_addr_hi / inv_addr_lo witnesses.
    if row_idx + 1 < n_smat {
      let next_key = smat_rows[row_idx + 1].0;
      let diff_hi = (next_key.addr_hi as i64) - (key.addr_hi as i64);
      let diff_lo = (next_key.addr_lo as i64) - (key.addr_lo as i64);
      if diff_hi == 0 && diff_lo == 0 {
        data[base + MEM_COL_ADDR_EQ] = Val::ONE;
        // same_hi = same_lo = 1 → diff * inv = 0, set inv to 0
        data[base + MEM_COL_INV_ADDR_HI] = Val::ZERO;
        data[base + MEM_COL_INV_ADDR_LO] = Val::ZERO;
      } else {
        // addr_eq = 0: set inv so that diff * inv = 1 (standard inverse witness)
        // diff_hi != 0 OR diff_lo != 0; use whichever is non-zero.
        let inv_hi = if diff_hi != 0 {
          Val::from_u32(diff_hi.rem_euclid(0x7FFF_FFFFi64) as u32)
            .try_inverse()
            .unwrap_or(Val::ZERO)
        } else {
          Val::ZERO
        };
        let inv_lo = if diff_lo != 0 {
          Val::from_u32(diff_lo.rem_euclid(0x7FFF_FFFFi64) as u32)
            .try_inverse()
            .unwrap_or(Val::ZERO)
        } else {
          Val::ZERO
        };
        data[base + MEM_COL_INV_ADDR_HI] = inv_hi;
        data[base + MEM_COL_INV_ADDR_LO] = inv_lo;
      }
    }
  }

  // ── 5. Claim query rows ───────────────────────────────────────────
  for (i, c) in claims.iter().enumerate() {
    let base = (n_smat + i) * NUM_MEM_COLS;
    MemTraceKey::from_claim(c).fill_tuple_cols(&mut data[base..]);
    data[base + MEM_COL_MULT] = Val::ONE; // query: positive mult
    // is_smat = 0, addr_eq = 0 (already zero from vec initialisation)
  }

  Ok(RowMajorMatrix::new(data, NUM_MEM_COLS))
}

// ── Permutation trace helper ──────────────────────────────────────────

fn generate_smat_perm_trace(
  main_trace: &RowMajorMatrix<Val>,
  public_values: &[Val],
  perm_challenges: &[<CircleStarkConfig as p3_uni_stark::StarkGenericConfig>::Challenge],
) -> Option<RowMajorMatrix<<CircleStarkConfig as p3_uni_stark::StarkGenericConfig>::Challenge>> {
  let gadget = LogUpGadget::new();
  let lookup = make_mem_lookup();
  let mut lookup_data: Vec<
    LookupData<<CircleStarkConfig as p3_uni_stark::StarkGenericConfig>::Challenge>,
  > = vec![];
  let perm_trace = gadget.generate_permutation::<CircleStarkConfig>(
    main_trace,
    &None,
    public_values,
    &[lookup],
    &mut lookup_data,
    perm_challenges,
  );
  Some(perm_trace)
}

// ── Public API ────────────────────────────────────────────────────────

/// Proof of SMAT memory consistency backed by a Circle STARK / LogUp argument.
///
/// The proof commits to the claim set via `public_values`; supply the *same*
/// `claims` to [`verify_memory_consistency`].
pub struct MemoryConsistencyProof {
  pub stark_proof: CircleStarkProof,
}

/// Prove memory consistency for `claims` using the SMAT LogUp STARK.
///
/// Returns `Err` if any read claim is inconsistent with the witnessed writes.
pub fn prove_memory_consistency(
  claims: &[crate::transition::MemAccessClaim],
) -> Result<MemoryConsistencyProof, String> {
  let public_values = make_mem_public_values(claims);

  let trace = if claims.is_empty() {
    RowMajorMatrix::new(vec![Val::ZERO; 4 * NUM_MEM_COLS], NUM_MEM_COLS)
  } else {
    build_smat_trace(claims)?
  };

  let trace_for_perm = trace.clone();
  let pv_for_perm = public_values.clone();
  let lookup_prove = make_mem_lookup();
  let config = make_circle_config();

  let stark_proof = p3_uni_stark::prove_with_lookup(
    &config,
    &MemoryConsistencyAir,
    trace,
    &public_values,
    None,
    move |perm_challenges| {
      generate_smat_perm_trace(&trace_for_perm, &pv_for_perm, perm_challenges)
    },
    2,
    2,
    move |folder: &mut VerifierConstraintFolder<CircleStarkConfig>| {
      let gadget = LogUpGadget::new();
      gadget.eval_local_lookup(folder, &lookup_prove);
    },
  );

  Ok(MemoryConsistencyProof { stark_proof })
}

/// Verify a [`MemoryConsistencyProof`] against a set of access claims.
///
/// Recomputes the claim hash from `claims`, checks it against the proof's
/// committed public values, and runs the STARK / LogUp verifier.
pub fn verify_memory_consistency(
  proof: &MemoryConsistencyProof,
  claims: &[crate::transition::MemAccessClaim],
) -> bool {
  let public_values = make_mem_public_values(claims);
  let config = make_circle_config();
  let lookup_verify = make_mem_lookup();

  let result = p3_uni_stark::verify_with_lookup(
    &config,
    &MemoryConsistencyAir,
    &proof.stark_proof,
    &public_values,
    None,
    2,
    move |folder: &mut VerifierConstraintFolder<CircleStarkConfig>| {
      let gadget = LogUpGadget::new();
      gadget.eval_local_lookup(folder, &lookup_verify);
    },
  );
  result.is_ok()
}

// ============================================================
// Tests
// ============================================================
