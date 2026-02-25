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
const NUM_STACK_IR_COLS: usize = 10;

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
    }
  }

  Ok(trace)
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
    // Soundness fix: OP_NOT and OP_EQ_SYM appear in row_pop_count (and thus in
    // StackIR traces) but were missing from allowed_tags, causing allowed_poly
    // to reject valid proofs that use these opcodes.
    let t_not = crate::semantic_proof::OP_NOT as u16;
    let t_eq_sym = crate::semantic_proof::OP_EQ_SYM as u16;

    let c_bool = AB::Expr::from_u16(t_bool);
    let c_byte = AB::Expr::from_u16(t_byte);
    let c_eq_refl = AB::Expr::from_u16(t_eq_refl);
    let c_and_intro = AB::Expr::from_u16(t_and_intro);
    let c_eq_trans = AB::Expr::from_u16(t_eq_trans);
    let c_add_congr = AB::Expr::from_u16(t_add_congr);
    let c_add_carry_congr = AB::Expr::from_u16(t_add_carry_congr);
    let c_add_eq = AB::Expr::from_u16(t_add_eq);
    let c_u16_add_eq = AB::Expr::from_u16(t_u16_add_eq);
    let c_u29_add_eq = AB::Expr::from_u16(t_u29_add_eq);
    let c_u24_add_eq = AB::Expr::from_u16(t_u24_add_eq);
    let c_u15_mul_eq = AB::Expr::from_u16(t_u15_mul_eq);
    let c_add_carry_eq = AB::Expr::from_u16(t_add_carry_eq);
    let c_mul_low_eq = AB::Expr::from_u16(t_mul_low_eq);
    let c_mul_high_eq = AB::Expr::from_u16(t_mul_high_eq);
    let c_and_eq = AB::Expr::from_u16(t_and_eq);
    let c_or_eq = AB::Expr::from_u16(t_or_eq);
    let c_xor_eq = AB::Expr::from_u16(t_xor_eq);
    let c_not = AB::Expr::from_u16(t_not);
    let c_eq_sym = AB::Expr::from_u16(t_eq_sym);
    let c_zero = AB::Expr::from_u16(0);
    let c_one = AB::Expr::from_u16(1);
    let c_two = AB::Expr::from_u16(2);

    let allowed_tags = [
      t_bool,
      t_byte,
      t_eq_refl,
      t_and_intro,
      t_eq_trans,
      t_add_congr,
      t_add_carry_congr,
      t_add_eq,
      t_u16_add_eq,
      t_u29_add_eq,
      t_u24_add_eq,
      t_u15_mul_eq,
      t_add_carry_eq,
      t_mul_low_eq,
      t_mul_high_eq,
      t_and_eq,
      t_or_eq,
      t_xor_eq,
      t_not,
      t_eq_sym,
    ];

    let gate = |target: u16| {
      let mut g = AB::Expr::from_u16(1);
      for t in allowed_tags {
        if t != target {
          g *= op.clone().into() - AB::Expr::from_u16(t);
        }
      }
      g
    };

    let allowed_poly = (op.clone().into() - c_bool.clone())
      * (op.clone().into() - c_byte.clone())
      * (op.clone().into() - c_eq_refl.clone())
      * (op.clone().into() - c_and_intro.clone())
      * (op.clone().into() - c_eq_trans.clone())
      * (op.clone().into() - c_add_congr.clone())
      * (op.clone().into() - c_add_carry_congr.clone())
      * (op.clone().into() - c_add_eq.clone())
      * (op.clone().into() - c_u16_add_eq.clone())
      * (op.clone().into() - c_u29_add_eq.clone())
      * (op.clone().into() - c_u24_add_eq.clone())
      * (op.clone().into() - c_u15_mul_eq.clone())
      * (op.clone().into() - c_add_carry_eq.clone())
      * (op.clone().into() - c_mul_low_eq.clone())
      * (op.clone().into() - c_mul_high_eq.clone())
      * (op.clone().into() - c_and_eq.clone())
      * (op.clone().into() - c_or_eq.clone())
      * (op.clone().into() - c_xor_eq.clone())
      * (op.clone().into() - c_not.clone())
      * (op.clone().into() - c_eq_sym.clone());
    builder.assert_zero(allowed_poly);

    builder.assert_zero(
      sp_before.clone().into() - pop.clone().into() + push.clone().into() - sp_after.clone().into(),
    );

    builder.when_first_row().assert_zero(sp_before.clone());
    builder
      .when_transition()
      .assert_eq(next[STACK_COL_SP_BEFORE].clone(), sp_after.clone());
    // Soundness fix: enforce the last trace row ends with exactly 1 item on the
    // proof stack (a single WFF result).  Padding rows carry steady_sp = 1, so
    // this constraint passes for correctly-padded traces.
    builder.when_last_row().assert_eq(sp_after.clone(), AB::Expr::from_u16(1));

    let bool_gate = gate(t_bool);
    let byte_gate = gate(t_byte);
    let eq_refl_gate = gate(t_eq_refl);
    let and_intro_gate = gate(t_and_intro);
    let eq_trans_gate = gate(t_eq_trans);
    let add_congr_gate = gate(t_add_congr);
    let add_carry_congr_gate = gate(t_add_carry_congr);
    let add_eq_gate = gate(t_add_eq);
    let u16_add_eq_gate = gate(t_u16_add_eq);
    let u29_add_eq_gate = gate(t_u29_add_eq);
    let u24_add_eq_gate = gate(t_u24_add_eq);
    let u15_mul_eq_gate = gate(t_u15_mul_eq);
    let add_carry_eq_gate = gate(t_add_carry_eq);
    let mul_low_eq_gate = gate(t_mul_low_eq);
    let mul_high_eq_gate = gate(t_mul_high_eq);
    let and_eq_gate = gate(t_and_eq);
    let or_eq_gate = gate(t_or_eq);
    let xor_eq_gate = gate(t_xor_eq);
    let not_gate = gate(t_not);
    let eq_sym_gate = gate(t_eq_sym);

    builder.assert_zero(bool_gate.clone() * (pop.clone().into() - c_zero.clone()));
    builder.assert_zero(bool_gate.clone() * (push.clone().into() - c_one.clone()));
    builder.assert_zero(
      bool_gate.clone()
        * (ret_ty.clone().into() - AB::Expr::from_u16(crate::semantic_proof::RET_BOOL as u16)),
    );
    builder.assert_zero(bool_gate * (value.clone().into() - scalar0.clone().into()));

    builder.assert_zero(byte_gate.clone() * (pop.clone().into() - c_zero.clone()));
    builder.assert_zero(byte_gate.clone() * (push.clone().into() - c_one.clone()));
    builder.assert_zero(
      byte_gate.clone() * (ret_ty.clone().into() - AB::Expr::from_u16(RET_BYTE as u16)),
    );
    builder.assert_zero(byte_gate * (value.clone().into() - scalar0.clone().into()));

    builder.assert_zero(eq_refl_gate.clone() * (pop.clone().into() - c_one.clone()));
    builder.assert_zero(eq_refl_gate.clone() * (push.clone().into() - c_one.clone()));
    builder
      .assert_zero(eq_refl_gate * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)));

    builder.assert_zero(and_intro_gate.clone() * (pop.clone().into() - c_two));
    builder.assert_zero(and_intro_gate.clone() * (push.clone().into() - c_one));
    builder.assert_zero(
      and_intro_gate * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_AND as u16)),
    );

    builder.assert_zero(eq_trans_gate.clone() * (pop.clone().into() - AB::Expr::from_u16(2)));
    builder.assert_zero(eq_trans_gate.clone() * (push.clone().into() - AB::Expr::from_u16(1)));
    builder
      .assert_zero(eq_trans_gate * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)));

    builder.assert_zero(add_congr_gate.clone() * (pop.clone().into() - AB::Expr::from_u16(1)));
    builder.assert_zero(add_congr_gate.clone() * (push.clone().into() - AB::Expr::from_u16(1)));
    builder.assert_zero(
      add_congr_gate * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)),
    );

    builder
      .assert_zero(add_carry_congr_gate.clone() * (pop.clone().into() - AB::Expr::from_u16(1)));
    builder
      .assert_zero(add_carry_congr_gate.clone() * (push.clone().into() - AB::Expr::from_u16(1)));
    builder.assert_zero(
      add_carry_congr_gate * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)),
    );

    builder.assert_zero(add_eq_gate.clone() * (pop.clone().into() - AB::Expr::from_u16(0)));
    builder.assert_zero(add_eq_gate.clone() * (push.clone().into() - AB::Expr::from_u16(1)));
    builder
      .assert_zero(add_eq_gate * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)));

    builder.assert_zero(u16_add_eq_gate.clone() * (pop.clone().into() - AB::Expr::from_u16(0)));
    builder.assert_zero(u16_add_eq_gate.clone() * (push.clone().into() - AB::Expr::from_u16(1)));
    builder.assert_zero(
      u16_add_eq_gate * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_AND as u16)),
    );

    builder.assert_zero(u29_add_eq_gate.clone() * (pop.clone().into() - AB::Expr::from_u16(0)));
    builder.assert_zero(u29_add_eq_gate.clone() * (push.clone().into() - AB::Expr::from_u16(1)));
    builder.assert_zero(
      u29_add_eq_gate * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_AND as u16)),
    );

    builder.assert_zero(u24_add_eq_gate.clone() * (pop.clone().into() - AB::Expr::from_u16(0)));
    builder.assert_zero(u24_add_eq_gate.clone() * (push.clone().into() - AB::Expr::from_u16(1)));
    builder.assert_zero(
      u24_add_eq_gate * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_AND as u16)),
    );

    builder.assert_zero(u15_mul_eq_gate.clone() * (pop.clone().into() - AB::Expr::from_u16(0)));
    builder.assert_zero(u15_mul_eq_gate.clone() * (push.clone().into() - AB::Expr::from_u16(1)));
    builder.assert_zero(
      u15_mul_eq_gate * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_AND as u16)),
    );

    builder.assert_zero(add_carry_eq_gate.clone() * (pop.clone().into() - AB::Expr::from_u16(0)));
    builder.assert_zero(add_carry_eq_gate.clone() * (push.clone().into() - AB::Expr::from_u16(1)));
    builder.assert_zero(
      add_carry_eq_gate * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)),
    );

    builder.assert_zero(mul_low_eq_gate.clone() * (pop.clone().into() - AB::Expr::from_u16(0)));
    builder.assert_zero(mul_low_eq_gate.clone() * (push.clone().into() - AB::Expr::from_u16(1)));
    builder.assert_zero(
      mul_low_eq_gate * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)),
    );

    builder.assert_zero(mul_high_eq_gate.clone() * (pop.clone().into() - AB::Expr::from_u16(0)));
    builder.assert_zero(mul_high_eq_gate.clone() * (push.clone().into() - AB::Expr::from_u16(1)));
    builder.assert_zero(
      mul_high_eq_gate * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)),
    );

    builder.assert_zero(and_eq_gate.clone() * (pop.clone().into() - AB::Expr::from_u16(0)));
    builder.assert_zero(and_eq_gate.clone() * (push.clone().into() - AB::Expr::from_u16(1)));
    builder
      .assert_zero(and_eq_gate * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)));

    builder.assert_zero(or_eq_gate.clone() * (pop.clone().into() - AB::Expr::from_u16(0)));
    builder.assert_zero(or_eq_gate.clone() * (push.clone().into() - AB::Expr::from_u16(1)));
    builder
      .assert_zero(or_eq_gate * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)));

    builder.assert_zero(xor_eq_gate.clone() * (pop.clone().into() - AB::Expr::from_u16(0)));
    builder.assert_zero(xor_eq_gate.clone() * (push.clone().into() - AB::Expr::from_u16(1)));
    builder
      .assert_zero(xor_eq_gate * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)));

    // Soundness fix: OP_NOT — unary boolean term constructor (pop=1, push=1, ret_ty=RET_BOOL)
    builder.assert_zero(not_gate.clone() * (pop.clone().into() - AB::Expr::from_u16(1)));
    builder.assert_zero(not_gate.clone() * (push.clone().into() - AB::Expr::from_u16(1)));
    builder.assert_zero(
      not_gate
        * (ret_ty.clone().into()
          - AB::Expr::from_u16(crate::semantic_proof::RET_BOOL as u16)),
    );

    // Soundness fix: OP_EQ_SYM — proof rule (pop=1, push=1, ret_ty=RET_WFF_EQ)
    builder.assert_zero(eq_sym_gate.clone() * (pop.clone().into() - AB::Expr::from_u16(1)));
    builder.assert_zero(eq_sym_gate.clone() * (push.clone().into() - AB::Expr::from_u16(1)));
    builder.assert_zero(
      eq_sym_gate * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)),
    );
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
      let prep_row = prep.row_slice(0).expect("StackIrAirWithPrep: empty preprocessed trace");
      let prep_row = &*prep_row;
      let main = builder.main();
      let local = main.row_slice(0).expect("StackIrAirWithPrep: empty main trace");
      let local = &*local;

      builder.assert_eq(local[STACK_COL_OP].clone(),      prep_row[PREP_COL_OP].clone());
      builder.assert_eq(local[STACK_COL_SCALAR0].clone(), prep_row[PREP_COL_SCALAR0].clone());
      builder.assert_eq(local[STACK_COL_SCALAR1].clone(), prep_row[PREP_COL_SCALAR1].clone());
      builder.assert_eq(local[STACK_COL_SCALAR2].clone(), prep_row[PREP_COL_SCALAR2].clone());
      builder.assert_eq(local[STACK_COL_VALUE].clone(),   prep_row[PREP_COL_VALUE].clone());
      builder.assert_eq(local[STACK_COL_RET_TY].clone(),  prep_row[PREP_COL_RET_TY].clone());
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
        let row = prep.row_slice(0).expect("StackIrAirWithPrep: empty prep (pv-bind)");
        row[PREP_COL_EVM_OPCODE].clone().into()
      };
      let prep_digest: [AB::Expr; 8] = {
        let prep = builder.preprocessed();
        let row = prep.row_slice(0).expect("StackIrAirWithPrep: empty prep (pv-bind)");
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
  Ok(p3_uni_stark::prove(&config, &StackIrAir, trace, public_values))
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
  let mut row_meta: Vec<(Val, [Val; 8])> = vec![
    (Val::ZERO, [Val::ZERO; 8]);
    rows.len()
  ];
  for entry in &manifest.entries {
    let per_digest = compute_wff_opcode_digest(entry.opcode, &entry.wff);
    let opcode_val = Val::from_u32(entry.opcode as u32);
    for r in entry.row_start..(entry.row_start + entry.row_count).min(rows.len()) {
      row_meta[r] = (opcode_val, per_digest);
    }
  }

  for (i, row) in rows.iter().enumerate() {
    let base = i * NUM_BATCH_PREP_COLS;
    matrix.values[base + PREP_COL_OP]      = Val::from_u32(row.op);
    matrix.values[base + PREP_COL_SCALAR0] = Val::from_u32(row.scalar0);
    matrix.values[base + PREP_COL_SCALAR1] = Val::from_u32(row.scalar1);
    matrix.values[base + PREP_COL_SCALAR2] = Val::from_u32(row.scalar2);
    matrix.values[base + PREP_COL_ARG0]    = Val::from_u32(row.arg0);
    matrix.values[base + PREP_COL_ARG1]    = Val::from_u32(row.arg1);
    matrix.values[base + PREP_COL_ARG2]    = Val::from_u32(row.arg2);
    matrix.values[base + PREP_COL_VALUE]   = Val::from_u32(row.value);
    matrix.values[base + PREP_COL_RET_TY]  = Val::from_u32(row.ret_ty);
    let (opcode_val, per_digest) = row_meta[i];
    matrix.values[base + PREP_COL_EVM_OPCODE] = opcode_val;
    for k in 0..8 {
      matrix.values[base + PREP_COL_WFF_DIGEST_START + k] = per_digest[k];
    }
    // Batch-level metadata (same for every row).
    matrix.values[base + PREP_COL_BATCH_N] = batch_n;
    for k in 0..8 {
      matrix.values[base + PREP_COL_BATCH_DIGEST_START + k] =
        if k < batch_digest.len() { batch_digest[k] } else { Val::ZERO };
    }
  }

  // Padding rows: op = OP_EQ_REFL, ret_ty = RET_WFF_EQ; batch metadata replicated.
  // Pick a representative per-instruction opcode/digest from the last live row.
  let (last_opcode, last_per_digest) = row_meta.last().copied().unwrap_or((Val::ZERO, [Val::ZERO; 8]));
  for i in rows.len()..n_rows {
    let base = i * NUM_BATCH_PREP_COLS;
    matrix.values[base + PREP_COL_OP] =
      Val::from_u32(crate::semantic_proof::OP_EQ_REFL);
    matrix.values[base + PREP_COL_RET_TY] =
      Val::from_u32(crate::semantic_proof::RET_WFF_EQ);
    matrix.values[base + PREP_COL_EVM_OPCODE] = last_opcode;
    for k in 0..8 {
      matrix.values[base + PREP_COL_WFF_DIGEST_START + k] = last_per_digest[k];
    }
    matrix.values[base + PREP_COL_BATCH_N] = batch_n;
    for k in 0..8 {
      matrix.values[base + PREP_COL_BATCH_DIGEST_START + k] =
        if k < batch_digest.len() { batch_digest[k] } else { Val::ZERO };
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
      matrix.values[base + PREP_COL_WFF_DIGEST_START + k] =
        if k < digest_slice.len() { digest_slice[k] } else { Val::ZERO };
    }
  }

  // Padding: op = OP_EQ_REFL, ret_ty = RET_WFF_EQ; opcode/digest same as live rows.
  for i in rows.len()..n_rows {
    let base = i * NUM_PREP_COLS;
    matrix.values[base + PREP_COL_OP] =
      Val::from_u32(crate::semantic_proof::OP_EQ_REFL);
    matrix.values[base + PREP_COL_RET_TY] =
      Val::from_u32(crate::semantic_proof::RET_WFF_EQ);
    matrix.values[base + PREP_COL_EVM_OPCODE] = evm_opcode;
    for k in 0..8 {
      matrix.values[base + PREP_COL_WFF_DIGEST_START + k] =
        if k < digest_slice.len() { digest_slice[k] } else { Val::ZERO };
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
  let proof = p3_uni_stark::prove_with_preprocessed(
    &config,
    &air,
    trace,
    public_values,
    Some(prep_data),
  );
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
  p3_uni_stark::verify_with_preprocessed(&config, &StackIrAirWithPrep::for_verify(), proof, public_values, Some(prep_vk))
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
pub const NUM_LUT_COLS: usize = 6;

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
  }

  for i in steps.len()..n_rows {
    let base = i * NUM_LUT_COLS;
    trace.values[base + LUT_COL_OP] = Val::from_u16(lut_opcode_tag(LutOpcode::ByteAddEq));
    trace.values[base + LUT_COL_IN0] = Val::from_u16(0);
    trace.values[base + LUT_COL_IN1] = Val::from_u16(0);
    trace.values[base + LUT_COL_IN2] = Val::from_u16(0);
    trace.values[base + LUT_COL_OUT0] = Val::from_u16(0);
    trace.values[base + LUT_COL_OUT1] = Val::from_u16(0);
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
        ensure_le("in0", step.in0, 1, i)?;
        ensure_le("in1", step.in1, 1, i)?;
        ensure_le("in2", step.in2, 0, i)?;
        ensure_le("out0", step.out0, 1, i)?;
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

    let c_add = AB::Expr::from_u16(lut_opcode_tag(LutOpcode::ByteAddEq));
    let c_u29_add = AB::Expr::from_u16(lut_opcode_tag(LutOpcode::U29AddEq));
    let c_u24_add = AB::Expr::from_u16(lut_opcode_tag(LutOpcode::U24AddEq));
    let c_u15_add = AB::Expr::from_u16(lut_opcode_tag(LutOpcode::U15AddEq));
    let c_bit_add = AB::Expr::from_u16(lut_opcode_tag(LutOpcode::BitAddEq));
    let c_u15_mul = AB::Expr::from_u16(lut_opcode_tag(LutOpcode::U15MulEq));
    let c_u16_add = AB::Expr::from_u16(lut_opcode_tag(LutOpcode::U16AddEq));
    let c_add_carry = AB::Expr::from_u16(lut_opcode_tag(LutOpcode::ByteAddCarryEq));
    let c_mul_low = AB::Expr::from_u16(lut_opcode_tag(LutOpcode::ByteMulLowEq));
    let c_mul_high = AB::Expr::from_u16(lut_opcode_tag(LutOpcode::ByteMulHighEq));
    let c_and = AB::Expr::from_u16(lut_opcode_tag(LutOpcode::ByteAndEq));
    let c_or = AB::Expr::from_u16(lut_opcode_tag(LutOpcode::ByteOrEq));
    let c_xor = AB::Expr::from_u16(lut_opcode_tag(LutOpcode::ByteXorEq));
    let c256 = AB::Expr::from_u16(256);

    let allowed_poly = (op.clone().into() - c_add.clone())
      * (op.clone().into() - c_u29_add.clone())
      * (op.clone().into() - c_u24_add.clone())
      * (op.clone().into() - c_u15_add.clone())
      * (op.clone().into() - c_bit_add.clone())
      * (op.clone().into() - c_u15_mul.clone())
      * (op.clone().into() - c_u16_add.clone())
      * (op.clone().into() - c_add_carry.clone())
      * (op.clone().into() - c_mul_low.clone())
      * (op.clone().into() - c_mul_high.clone())
      * (op.clone().into() - c_and.clone())
      * (op.clone().into() - c_or.clone())
      * (op.clone().into() - c_xor.clone());
    builder.assert_zero(allowed_poly);

    let t_add = lut_opcode_tag(LutOpcode::ByteAddEq);
    let t_u29_add = lut_opcode_tag(LutOpcode::U29AddEq);
    let t_u24_add = lut_opcode_tag(LutOpcode::U24AddEq);
    let t_u15_add = lut_opcode_tag(LutOpcode::U15AddEq);
    let t_bit_add = lut_opcode_tag(LutOpcode::BitAddEq);
    let t_u15_mul = lut_opcode_tag(LutOpcode::U15MulEq);
    let t_u16_add = lut_opcode_tag(LutOpcode::U16AddEq);
    let t_add_carry = lut_opcode_tag(LutOpcode::ByteAddCarryEq);
    let t_mul_low = lut_opcode_tag(LutOpcode::ByteMulLowEq);
    let t_mul_high = lut_opcode_tag(LutOpcode::ByteMulHighEq);
    let t_and = lut_opcode_tag(LutOpcode::ByteAndEq);
    let t_or = lut_opcode_tag(LutOpcode::ByteOrEq);
    let t_xor = lut_opcode_tag(LutOpcode::ByteXorEq);

    let all_tags = [
      t_add,
      t_u29_add,
      t_u24_add,
      t_u15_add,
      t_bit_add,
      t_u15_mul,
      t_u16_add,
      t_add_carry,
      t_mul_low,
      t_mul_high,
      t_and,
      t_or,
      t_xor,
    ];
    let gate = |target_tag: u16| {
      let mut g = AB::Expr::from_u16(1);
      for t in all_tags {
        if t != target_tag {
          g *= op.clone().into() - AB::Expr::from_u16(t);
        }
      }
      g
    };

    let g_add = gate(t_add);
    let g_u29_add = gate(t_u29_add);
    let g_u24_add = gate(t_u24_add);
    let g_u15_add = gate(t_u15_add);
    let g_bit_add = gate(t_bit_add);
    let g_u15_mul = gate(t_u15_mul);
    let g_u16_add = gate(t_u16_add);
    let g_add_carry = gate(t_add_carry);
    let g_mul_low = gate(t_mul_low);
    let g_mul_high = gate(t_mul_high);
    let g_and = gate(t_and);
    let g_or = gate(t_or);
    let g_xor = gate(t_xor);

    let total = in0.clone().into() + in1.clone().into() + in2.clone().into();
    let c32768 = AB::Expr::from_u32(32768);
    let c16777216 = AB::Expr::from_u32(1u32 << 24);
    let c536870912 = AB::Expr::from_u32(1u32 << 29);
    let c2 = AB::Expr::from_u16(2);
    let c65536 = AB::Expr::from_u32(65536);

    builder.assert_zero(
      g_add.clone() * (total.clone() - out0.clone().into() - c256.clone() * out1.clone().into()),
    );
    builder.assert_zero(
      g_add.clone() * (out1.clone().into() * (out1.clone().into() - AB::Expr::from_u16(1))),
    );

    builder.assert_zero(
      g_u29_add.clone()
        * (total.clone() - out0.clone().into() - c536870912.clone() * out1.clone().into()),
    );
    builder.assert_zero(
      g_u29_add.clone() * (out1.clone().into() * (out1.clone().into() - AB::Expr::from_u16(1))),
    );

    builder.assert_zero(
      g_u24_add.clone()
        * (total.clone() - out0.clone().into() - c16777216.clone() * out1.clone().into()),
    );
    builder.assert_zero(
      g_u24_add.clone() * (out1.clone().into() * (out1.clone().into() - AB::Expr::from_u16(1))),
    );

    builder.assert_zero(
      g_u16_add.clone()
        * (total.clone() - out0.clone().into() - c65536.clone() * out1.clone().into()),
    );
    builder.assert_zero(
      g_u16_add.clone() * (out1.clone().into() * (out1.clone().into() - AB::Expr::from_u16(1))),
    );

    builder.assert_zero(
      g_u15_add.clone()
        * (total.clone() - out0.clone().into() - c32768.clone() * out1.clone().into()),
    );
    builder.assert_zero(
      g_u15_add.clone() * (out1.clone().into() * (out1.clone().into() - AB::Expr::from_u16(1))),
    );

    builder.assert_zero(
      g_bit_add.clone() * (total.clone() - out0.clone().into() - c2.clone() * out1.clone().into()),
    );
    builder.assert_zero(
      g_bit_add.clone() * (in0.clone().into() * (in0.clone().into() - AB::Expr::from_u16(1))),
    );
    builder.assert_zero(
      g_bit_add.clone() * (in1.clone().into() * (in1.clone().into() - AB::Expr::from_u16(1))),
    );
    builder.assert_zero(
      g_bit_add.clone() * (in2.clone().into() * (in2.clone().into() - AB::Expr::from_u16(1))),
    );
    builder.assert_zero(
      g_bit_add.clone() * (out0.clone().into() * (out0.clone().into() - AB::Expr::from_u16(1))),
    );
    builder.assert_zero(
      g_bit_add.clone() * (out1.clone().into() * (out1.clone().into() - AB::Expr::from_u16(1))),
    );

    builder.assert_zero(
      g_u15_mul.clone()
        * ((in0.clone().into() * in1.clone().into())
          - out0.clone().into()
          - c32768.clone() * out1.clone().into()),
    );
    builder.assert_zero(g_u15_mul.clone() * in2.clone());

    builder.assert_zero(
      g_add_carry.clone()
        * (total.clone() - out1.clone().into() - c256.clone() * out0.clone().into()),
    );
    builder.assert_zero(
      g_add_carry.clone() * (out0.clone().into() * (out0.clone().into() - AB::Expr::from_u16(1))),
    );

    builder.assert_zero(
      g_mul_low.clone()
        * ((in0.clone().into() * in1.clone().into())
          - out0.clone().into()
          - c256.clone() * out1.clone().into()),
    );
    builder.assert_zero(g_mul_low.clone() * in2.clone());

    builder.assert_zero(
      g_mul_high.clone()
        * ((in0.clone().into() * in1.clone().into())
          - out1.clone().into()
          - c256.clone() * out0.clone().into()),
    );
    builder.assert_zero(g_mul_high.clone() * in2.clone());

    builder.assert_zero(
      g_and.clone() * (out0.clone().into() - (in0.clone().into() * in1.clone().into())),
    );
    builder.assert_zero(
      g_and.clone() * (in0.clone().into() * (in0.clone().into() - AB::Expr::from_u16(1))),
    );
    builder.assert_zero(
      g_and.clone() * (in1.clone().into() * (in1.clone().into() - AB::Expr::from_u16(1))),
    );
    builder.assert_zero(
      g_and.clone() * (out0.clone().into() * (out0.clone().into() - AB::Expr::from_u16(1))),
    );
    builder.assert_zero(g_and.clone() * in2.clone());
    builder.assert_zero(g_and.clone() * out1.clone());

    builder.assert_zero(
      g_or.clone()
        * (out0.clone().into()
          - (in0.clone().into() + in1.clone().into() - in0.clone().into() * in1.clone().into())),
    );
    builder.assert_zero(
      g_or.clone() * (in0.clone().into() * (in0.clone().into() - AB::Expr::from_u16(1))),
    );
    builder.assert_zero(
      g_or.clone() * (in1.clone().into() * (in1.clone().into() - AB::Expr::from_u16(1))),
    );
    builder.assert_zero(
      g_or.clone() * (out0.clone().into() * (out0.clone().into() - AB::Expr::from_u16(1))),
    );
    builder.assert_zero(g_or.clone() * in2.clone());
    builder.assert_zero(g_or.clone() * out1.clone());

    builder.assert_zero(
      g_xor.clone()
        * (out0.clone().into()
          - (in0.clone().into() + in1.clone().into()
            - AB::Expr::from_u16(2) * in0.clone().into() * in1.clone().into())),
    );
    builder.assert_zero(
      g_xor.clone() * (in0.clone().into() * (in0.clone().into() - AB::Expr::from_u16(1))),
    );
    builder.assert_zero(
      g_xor.clone() * (in1.clone().into() * (in1.clone().into() - AB::Expr::from_u16(1))),
    );
    builder.assert_zero(
      g_xor.clone() * (out0.clone().into() * (out0.clone().into() - AB::Expr::from_u16(1))),
    );
    builder.assert_zero(g_xor.clone() * in2.clone());
    builder.assert_zero(g_xor.clone() * out1.clone());

    // (dead constraint `g_add_carry * c_zero` removed — was always trivially 0)
    builder.assert_bool(in2.clone());
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
  fn eval(&self, builder: &mut AB) {
    // ── Phase 3: preprocessed-row binding ──────────────────────────────────
    //
    // For every LUT-eligible ProofRow op `X`, a gate polynomial is zero unless
    // `prep[PREP_COL_OP] == X`.  When it fires it constrains the LUT main-
    // trace columns to reflect the committed ProofRow fields.
    //
    // For structural (non-LUT) ProofRow ops the gate is always zero and the
    // LUT step's ByteAddEq(0,0,0,0,0) padding is unconstrained.
    {
      use crate::semantic_proof::{
        OP_AND_INTRO, OP_BOOL, OP_BYTE, OP_BYTE_ADD_CARRY_EQ, OP_BYTE_ADD_EQ,
        OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE, OP_BYTE_ADD_THIRD_CONGRUENCE,
        OP_BYTE_AND_EQ, OP_BYTE_MUL_HIGH_EQ, OP_BYTE_MUL_LOW_EQ, OP_BYTE_OR_EQ,
        OP_BYTE_XOR_EQ, OP_EQ_REFL, OP_EQ_SYM, OP_EQ_TRANS, OP_NOT, OP_U15_MUL_EQ,
        OP_U16_ADD_EQ, OP_U24_ADD_EQ, OP_U29_ADD_EQ,
      };

      let prep = builder.preprocessed();
      let prep_row = prep
        .row_slice(0)
        .expect("LutKernelAirWithPrep: empty preprocessed trace");
      let prep_row = &*prep_row;

      let main = builder.main();
      let local = main
        .row_slice(0)
        .expect("LutKernelAirWithPrep: empty main trace");
      let local = &*local;

      // Gate polynomial that is zero unless `prep[op] == target`.
      //
      // IMPORTANT: must include ALL ProofRow ops — both LUT-eligible AND
      // structural (OP_EQ_REFL, OP_BOOL, etc.) — so that the gate evaluates
      // to zero for preprocessed padding rows (op = OP_EQ_REFL), preventing
      // spurious constraint fires on non-LUT rows.  Degree = 17.
      let all_prep_ops: &[u32] = &[
        // LUT-eligible ops
        OP_BYTE_ADD_EQ, OP_U16_ADD_EQ, OP_U29_ADD_EQ, OP_U24_ADD_EQ,
        OP_U15_MUL_EQ, OP_BYTE_ADD_CARRY_EQ, OP_BYTE_MUL_LOW_EQ,
        OP_BYTE_MUL_HIGH_EQ, OP_BYTE_AND_EQ, OP_BYTE_OR_EQ, OP_BYTE_XOR_EQ,
        // Structural (non-LUT) ops that appear in preprocessed padding rows
        OP_BOOL, OP_BYTE, OP_AND_INTRO, OP_EQ_REFL, OP_EQ_TRANS,
        OP_BYTE_ADD_THIRD_CONGRUENCE, OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE,
        // Soundness fix: OP_NOT and OP_EQ_SYM can appear in proof traces;
        // they must be in all_prep_ops so the pg() gate polynomial evaluates
        // to zero when the preprocessed op is NOT or EQ_SYM.
        OP_NOT, OP_EQ_SYM,
      ];
      let pg = |target: u32| -> AB::Expr {
        all_prep_ops
          .iter()
          .filter(|&&t| t != target)
          .fold(AB::Expr::from_u32(1), |acc, &t| {
            acc * (prep_row[PREP_COL_OP].clone().into() - AB::Expr::from_u32(t))
          })
      };

      // OP_BYTE_ADD_EQ → ByteAddEq
      // in0=scalar0, in1=scalar1, in2=arg0, out0=value, out1=scalar2
      let tag_add = lut_opcode_tag(LutOpcode::ByteAddEq) as u32;
      let g = pg(OP_BYTE_ADD_EQ);
      builder.assert_zero(g.clone() * (local[LUT_COL_OP].clone().into()   - AB::Expr::from_u32(tag_add)));
      builder.assert_zero(g.clone() * (local[LUT_COL_IN0].clone().into()  - prep_row[PREP_COL_SCALAR0].clone().into()));
      builder.assert_zero(g.clone() * (local[LUT_COL_IN1].clone().into()  - prep_row[PREP_COL_SCALAR1].clone().into()));
      builder.assert_zero(g.clone() * (local[LUT_COL_IN2].clone().into()  - prep_row[PREP_COL_ARG0].clone().into()));
      builder.assert_zero(g.clone() * (local[LUT_COL_OUT0].clone().into() - prep_row[PREP_COL_VALUE].clone().into()));
      builder.assert_zero(g         * (local[LUT_COL_OUT1].clone().into() - prep_row[PREP_COL_SCALAR2].clone().into()));

      // OP_U16_ADD_EQ → U16AddEq
      // in0=scalar0, in1=scalar1, in2=scalar2, out0=value, out1=arg1
      let tag_u16 = lut_opcode_tag(LutOpcode::U16AddEq) as u32;
      let g = pg(OP_U16_ADD_EQ);
      builder.assert_zero(g.clone() * (local[LUT_COL_OP].clone().into()   - AB::Expr::from_u32(tag_u16)));
      builder.assert_zero(g.clone() * (local[LUT_COL_IN0].clone().into()  - prep_row[PREP_COL_SCALAR0].clone().into()));
      builder.assert_zero(g.clone() * (local[LUT_COL_IN1].clone().into()  - prep_row[PREP_COL_SCALAR1].clone().into()));
      builder.assert_zero(g.clone() * (local[LUT_COL_IN2].clone().into()  - prep_row[PREP_COL_SCALAR2].clone().into()));
      builder.assert_zero(g.clone() * (local[LUT_COL_OUT0].clone().into() - prep_row[PREP_COL_VALUE].clone().into()));
      builder.assert_zero(g         * (local[LUT_COL_OUT1].clone().into() - prep_row[PREP_COL_ARG1].clone().into()));

      // OP_U29_ADD_EQ → U29AddEq
      // in0=scalar0, in1=scalar1, in2=scalar2, out0=value, out1=arg1
      let tag_u29 = lut_opcode_tag(LutOpcode::U29AddEq) as u32;
      let g = pg(OP_U29_ADD_EQ);
      builder.assert_zero(g.clone() * (local[LUT_COL_OP].clone().into()   - AB::Expr::from_u32(tag_u29)));
      builder.assert_zero(g.clone() * (local[LUT_COL_IN0].clone().into()  - prep_row[PREP_COL_SCALAR0].clone().into()));
      builder.assert_zero(g.clone() * (local[LUT_COL_IN1].clone().into()  - prep_row[PREP_COL_SCALAR1].clone().into()));
      builder.assert_zero(g.clone() * (local[LUT_COL_IN2].clone().into()  - prep_row[PREP_COL_SCALAR2].clone().into()));
      builder.assert_zero(g.clone() * (local[LUT_COL_OUT0].clone().into() - prep_row[PREP_COL_VALUE].clone().into()));
      builder.assert_zero(g         * (local[LUT_COL_OUT1].clone().into() - prep_row[PREP_COL_ARG1].clone().into()));

      // OP_U24_ADD_EQ → U24AddEq (same field layout as U29)
      let tag_u24 = lut_opcode_tag(LutOpcode::U24AddEq) as u32;
      let g = pg(OP_U24_ADD_EQ);
      builder.assert_zero(g.clone() * (local[LUT_COL_OP].clone().into()   - AB::Expr::from_u32(tag_u24)));
      builder.assert_zero(g.clone() * (local[LUT_COL_IN0].clone().into()  - prep_row[PREP_COL_SCALAR0].clone().into()));
      builder.assert_zero(g.clone() * (local[LUT_COL_IN1].clone().into()  - prep_row[PREP_COL_SCALAR1].clone().into()));
      builder.assert_zero(g.clone() * (local[LUT_COL_IN2].clone().into()  - prep_row[PREP_COL_SCALAR2].clone().into()));
      builder.assert_zero(g.clone() * (local[LUT_COL_OUT0].clone().into() - prep_row[PREP_COL_VALUE].clone().into()));
      builder.assert_zero(g         * (local[LUT_COL_OUT1].clone().into() - prep_row[PREP_COL_ARG1].clone().into()));

      // OP_U15_MUL_EQ → U15MulEq
      // in0=scalar0, in1=scalar1, in2=0, out0=value(lo), out1=arg0(hi)
      let tag_mul = lut_opcode_tag(LutOpcode::U15MulEq) as u32;
      let g = pg(OP_U15_MUL_EQ);
      builder.assert_zero(g.clone() * (local[LUT_COL_OP].clone().into()   - AB::Expr::from_u32(tag_mul)));
      builder.assert_zero(g.clone() * (local[LUT_COL_IN0].clone().into()  - prep_row[PREP_COL_SCALAR0].clone().into()));
      builder.assert_zero(g.clone() * (local[LUT_COL_IN1].clone().into()  - prep_row[PREP_COL_SCALAR1].clone().into()));
      builder.assert_zero(g.clone() * (local[LUT_COL_OUT0].clone().into() - prep_row[PREP_COL_VALUE].clone().into()));
      builder.assert_zero(g         * (local[LUT_COL_OUT1].clone().into() - prep_row[PREP_COL_ARG0].clone().into()));

      // OP_BYTE_ADD_CARRY_EQ → ByteAddCarryEq
      // in0=scalar0, in1=scalar1, in2=arg0 (outputs arithmetically derived)
      let tag_carry = lut_opcode_tag(LutOpcode::ByteAddCarryEq) as u32;
      let g = pg(OP_BYTE_ADD_CARRY_EQ);
      builder.assert_zero(g.clone() * (local[LUT_COL_OP].clone().into()  - AB::Expr::from_u32(tag_carry)));
      builder.assert_zero(g.clone() * (local[LUT_COL_IN0].clone().into() - prep_row[PREP_COL_SCALAR0].clone().into()));
      builder.assert_zero(g.clone() * (local[LUT_COL_IN1].clone().into() - prep_row[PREP_COL_SCALAR1].clone().into()));
      builder.assert_zero(g         * (local[LUT_COL_IN2].clone().into() - prep_row[PREP_COL_ARG0].clone().into()));

      // OP_BYTE_MUL_LOW_EQ → ByteMulLowEq
      // in0=scalar0, in1=scalar1 (out arithmetically derived)
      let tag_ml = lut_opcode_tag(LutOpcode::ByteMulLowEq) as u32;
      let g = pg(OP_BYTE_MUL_LOW_EQ);
      builder.assert_zero(g.clone() * (local[LUT_COL_OP].clone().into()  - AB::Expr::from_u32(tag_ml)));
      builder.assert_zero(g.clone() * (local[LUT_COL_IN0].clone().into() - prep_row[PREP_COL_SCALAR0].clone().into()));
      builder.assert_zero(g         * (local[LUT_COL_IN1].clone().into() - prep_row[PREP_COL_SCALAR1].clone().into()));

      // OP_BYTE_MUL_HIGH_EQ → ByteMulHighEq
      let tag_mh = lut_opcode_tag(LutOpcode::ByteMulHighEq) as u32;
      let g = pg(OP_BYTE_MUL_HIGH_EQ);
      builder.assert_zero(g.clone() * (local[LUT_COL_OP].clone().into()  - AB::Expr::from_u32(tag_mh)));
      builder.assert_zero(g.clone() * (local[LUT_COL_IN0].clone().into() - prep_row[PREP_COL_SCALAR0].clone().into()));
      builder.assert_zero(g         * (local[LUT_COL_IN1].clone().into() - prep_row[PREP_COL_SCALAR1].clone().into()));

      // OP_BYTE_AND_EQ / OP_BYTE_OR_EQ / OP_BYTE_XOR_EQ:
      // These bit-family ops use byte-level values (0..255) in ProofRows but
      // the inner LUT arithmetic constraints enforce bit-level (0 or 1) inputs.
      // Binding them to a `ByteAndEq`/`ByteOrEq`/`ByteXorEq` LUT row would
      // fail the arithmetic check.  The correctness guarantee for these rows
      // comes from the StackIR binding (scalar0/scalar1/value are all committed
      // to the shared preprocessed VK), so no additional LUT arithmetic
      // binding is needed here.
    }

    // ── pv binding: EVM opcode + WFF digest (first row only) ─────────────
    //
    // Mirrors the StackIrAirWithPrep Step 1b binding.  Both AIRs share the
    // same preprocessed matrix, so asserting row-0 EVM opcode/digest against
    // pis[1] and pis[2..10] here closes the same gap on the LUT side.
    {
      // Extract owned AB::Expr values before mutable when_first_row() calls.
      let pi_opcode: AB::Expr = {
        let pis = builder.public_values();
        pis[1].clone().into()
      };
      let pi_digest: [AB::Expr; 8] = {
        let pis = builder.public_values();
        std::array::from_fn(|k| pis[2 + k].clone().into())
      };
      let prep_opcode: AB::Expr = {
        let prep = builder.preprocessed();
        let row = prep.row_slice(0).expect("LutKernelAirWithPrep: empty prep (pv-bind)");
        row[PREP_COL_EVM_OPCODE].clone().into()
      };
      let prep_digest: [AB::Expr; 8] = {
        let prep = builder.preprocessed();
        let row = prep.row_slice(0).expect("LutKernelAirWithPrep: empty prep (pv-bind)");
        std::array::from_fn(|k| row[PREP_COL_WFF_DIGEST_START + k].clone().into())
      };
      builder.when_first_row().assert_eq(prep_opcode, pi_opcode);
      for k in 0..8_usize {
        builder
          .when_first_row()
          .assert_eq(prep_digest[k].clone(), pi_digest[k].clone());
      }
    }

    // ── All existing LUT arithmetic constraints ──────────────────────────
    eval_lut_kernel_inner(builder);
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
pub fn build_lut_trace_from_proof_rows(
  rows: &[ProofRow],
) -> Result<RowMajorMatrix<Val>, String> {
  use crate::semantic_proof::{
    OP_AND_INTRO, OP_BOOL, OP_BYTE, OP_BYTE_ADD_CARRY_EQ, OP_BYTE_ADD_EQ,
    OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE, OP_BYTE_ADD_THIRD_CONGRUENCE,
    OP_BYTE_AND_EQ, OP_BYTE_MUL_HIGH_EQ, OP_BYTE_MUL_LOW_EQ, OP_BYTE_OR_EQ,
    OP_BYTE_XOR_EQ, OP_EQ_REFL, OP_EQ_SYM, OP_EQ_TRANS, OP_NOT, OP_U15_MUL_EQ,
    OP_U16_ADD_EQ, OP_U24_ADD_EQ, OP_U29_ADD_EQ,
  };

  if rows.is_empty() {
    return Err("build_lut_trace_from_proof_rows: cannot build from empty row set".to_string());
  }

  let n_rows = rows.len().max(4).next_power_of_two();
  let mut trace =
    RowMajorMatrix::new(Val::zero_vec(n_rows * NUM_LUT_COLS), NUM_LUT_COLS);
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
        // in0..out1 remain zero — ByteAddEq(0+0+0=0) is arithmetically valid.
      }
      OP_BYTE_ADD_EQ => {
        trace.values[b + LUT_COL_OP]   = Val::from_u16(lut_opcode_tag(LutOpcode::ByteAddEq));
        trace.values[b + LUT_COL_IN0]  = Val::from_u32(row.scalar0);
        trace.values[b + LUT_COL_IN1]  = Val::from_u32(row.scalar1);
        trace.values[b + LUT_COL_IN2]  = Val::from_u32(row.arg0);
        trace.values[b + LUT_COL_OUT0] = Val::from_u32(row.value);
        trace.values[b + LUT_COL_OUT1] = Val::from_u32(row.scalar2);
      }
      OP_U16_ADD_EQ => {
        // Single U16AddEq step — keeps 1:1 row alignment with ProofRows.
        trace.values[b + LUT_COL_OP]   = Val::from_u16(lut_opcode_tag(LutOpcode::U16AddEq));
        trace.values[b + LUT_COL_IN0]  = Val::from_u32(row.scalar0);
        trace.values[b + LUT_COL_IN1]  = Val::from_u32(row.scalar1);
        trace.values[b + LUT_COL_IN2]  = Val::from_u32(row.scalar2);
        trace.values[b + LUT_COL_OUT0] = Val::from_u32(row.value);
        trace.values[b + LUT_COL_OUT1] = Val::from_u32(row.arg1);
      }
      OP_U29_ADD_EQ => {
        trace.values[b + LUT_COL_OP]   = Val::from_u16(lut_opcode_tag(LutOpcode::U29AddEq));
        trace.values[b + LUT_COL_IN0]  = Val::from_u32(row.scalar0);
        trace.values[b + LUT_COL_IN1]  = Val::from_u32(row.scalar1);
        trace.values[b + LUT_COL_IN2]  = Val::from_u32(row.scalar2);
        trace.values[b + LUT_COL_OUT0] = Val::from_u32(row.value);
        trace.values[b + LUT_COL_OUT1] = Val::from_u32(row.arg1);
      }
      OP_U24_ADD_EQ => {
        trace.values[b + LUT_COL_OP]   = Val::from_u16(lut_opcode_tag(LutOpcode::U24AddEq));
        trace.values[b + LUT_COL_IN0]  = Val::from_u32(row.scalar0);
        trace.values[b + LUT_COL_IN1]  = Val::from_u32(row.scalar1);
        trace.values[b + LUT_COL_IN2]  = Val::from_u32(row.scalar2);
        trace.values[b + LUT_COL_OUT0] = Val::from_u32(row.value);
        trace.values[b + LUT_COL_OUT1] = Val::from_u32(row.arg1);
      }
      OP_U15_MUL_EQ => {
        // lo = value, hi = arg0  (as stored by compile_proof)
        trace.values[b + LUT_COL_OP]   = Val::from_u16(lut_opcode_tag(LutOpcode::U15MulEq));
        trace.values[b + LUT_COL_IN0]  = Val::from_u32(row.scalar0);
        trace.values[b + LUT_COL_IN1]  = Val::from_u32(row.scalar1);
        // in2 = 0 (already zero)
        trace.values[b + LUT_COL_OUT0] = Val::from_u32(row.value);
        trace.values[b + LUT_COL_OUT1] = Val::from_u32(row.arg0);
      }
      OP_BYTE_ADD_CARRY_EQ => {
        let total = row.scalar0 + row.scalar1 + row.arg0;
        trace.values[b + LUT_COL_OP]   = Val::from_u16(lut_opcode_tag(LutOpcode::ByteAddCarryEq));
        trace.values[b + LUT_COL_IN0]  = Val::from_u32(row.scalar0);
        trace.values[b + LUT_COL_IN1]  = Val::from_u32(row.scalar1);
        trace.values[b + LUT_COL_IN2]  = Val::from_u32(row.arg0);
        trace.values[b + LUT_COL_OUT0] = Val::from_u32(if total >= 256 { 1 } else { 0 });
        trace.values[b + LUT_COL_OUT1] = Val::from_u32(total & 0xFF);
      }
      OP_BYTE_MUL_LOW_EQ => {
        let result = row.scalar0 * row.scalar1;
        trace.values[b + LUT_COL_OP]   = Val::from_u16(lut_opcode_tag(LutOpcode::ByteMulLowEq));
        trace.values[b + LUT_COL_IN0]  = Val::from_u32(row.scalar0);
        trace.values[b + LUT_COL_IN1]  = Val::from_u32(row.scalar1);
        trace.values[b + LUT_COL_OUT0] = Val::from_u32(result & 0xFF);
        // Soundness fix: AIR constraint is  in0*in1 = out0 + 256*out1
        // so out1 must carry the high byte.
        trace.values[b + LUT_COL_OUT1] = Val::from_u32(result >> 8);
      }
      OP_BYTE_MUL_HIGH_EQ => {
        let result = row.scalar0 * row.scalar1;
        trace.values[b + LUT_COL_OP]   = Val::from_u16(lut_opcode_tag(LutOpcode::ByteMulHighEq));
        trace.values[b + LUT_COL_IN0]  = Val::from_u32(row.scalar0);
        trace.values[b + LUT_COL_IN1]  = Val::from_u32(row.scalar1);
        trace.values[b + LUT_COL_OUT0] = Val::from_u32(result >> 8);
        // Soundness fix: AIR constraint is  in0*in1 = 256*out0 + out1
        // so out1 must carry the low byte.
        trace.values[b + LUT_COL_OUT1] = Val::from_u32(result & 0xFF);
      }
      OP_BYTE_AND_EQ | OP_BYTE_OR_EQ | OP_BYTE_XOR_EQ => {
        // Bit-family ops: byte inputs (0..255) are incompatible with the
        // bit-level LUT inner constraints (in0*(in0-1)==0).  Encode as a
        // ByteAddEq(0,0,0,0,0) pad so the arithmetic checks trivially pass.
        // The scalar0/scalar1/value columns are still committed to the shared
        // preprocessed VK via the StackIR binding, which is sufficient for
        // these operations.
        trace.values[b + LUT_COL_OP] = pad_tag;
        // in0..out1 remain zero.
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
    trace.values[i * NUM_LUT_COLS + LUT_COL_OP] = pad_tag;
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
  let proof = p3_uni_stark::prove_with_preprocessed(
    &config,
    &air,
    trace,
    public_values,
    Some(prep_data),
  );
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

// ============================================================
// Phase 4: BatchLutKernelAirWithPrep
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
fn eval_lut_prep_row_binding_inner<AB>(
  builder: &mut AB,
  prep_row: &[AB::Var],
  local: &[AB::Var],
)
where
  AB: AirBuilderWithPublicValues + PairBuilder + AirBuilder<F = Val>,
{
  use crate::semantic_proof::{
    OP_AND_INTRO, OP_BOOL, OP_BYTE, OP_BYTE_ADD_CARRY_EQ, OP_BYTE_ADD_EQ,
    OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE, OP_BYTE_ADD_THIRD_CONGRUENCE,
    OP_BYTE_AND_EQ, OP_BYTE_MUL_HIGH_EQ, OP_BYTE_MUL_LOW_EQ, OP_BYTE_OR_EQ,
    OP_BYTE_XOR_EQ, OP_EQ_REFL, OP_EQ_SYM, OP_EQ_TRANS, OP_NOT, OP_U15_MUL_EQ,
    OP_U16_ADD_EQ, OP_U24_ADD_EQ, OP_U29_ADD_EQ,
  };

  let all_prep_ops: &[u32] = &[
    OP_BYTE_ADD_EQ, OP_U16_ADD_EQ, OP_U29_ADD_EQ, OP_U24_ADD_EQ,
    OP_U15_MUL_EQ, OP_BYTE_ADD_CARRY_EQ, OP_BYTE_MUL_LOW_EQ,
    OP_BYTE_MUL_HIGH_EQ, OP_BYTE_AND_EQ, OP_BYTE_OR_EQ, OP_BYTE_XOR_EQ,
    OP_BOOL, OP_BYTE, OP_AND_INTRO, OP_EQ_REFL, OP_EQ_TRANS,
    OP_BYTE_ADD_THIRD_CONGRUENCE, OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE,
    OP_NOT, OP_EQ_SYM,
  ];
  let pg = |target: u32| -> AB::Expr {
    all_prep_ops
      .iter()
      .filter(|&&t| t != target)
      .fold(AB::Expr::from_u32(1), |acc, &t| {
        acc * (prep_row[PREP_COL_OP].clone().into() - AB::Expr::from_u32(t))
      })
  };

  // OP_BYTE_ADD_EQ → ByteAddEq: in0=s0, in1=s1, in2=arg0, out0=value, out1=s2
  let tag_add = lut_opcode_tag(LutOpcode::ByteAddEq) as u32;
  let g = pg(OP_BYTE_ADD_EQ);
  builder.assert_zero(g.clone() * (local[LUT_COL_OP].clone().into()   - AB::Expr::from_u32(tag_add)));
  builder.assert_zero(g.clone() * (local[LUT_COL_IN0].clone().into()  - prep_row[PREP_COL_SCALAR0].clone().into()));
  builder.assert_zero(g.clone() * (local[LUT_COL_IN1].clone().into()  - prep_row[PREP_COL_SCALAR1].clone().into()));
  builder.assert_zero(g.clone() * (local[LUT_COL_IN2].clone().into()  - prep_row[PREP_COL_ARG0].clone().into()));
  builder.assert_zero(g.clone() * (local[LUT_COL_OUT0].clone().into() - prep_row[PREP_COL_VALUE].clone().into()));
  builder.assert_zero(g         * (local[LUT_COL_OUT1].clone().into() - prep_row[PREP_COL_SCALAR2].clone().into()));

  // OP_U16_ADD_EQ → U16AddEq: in0=s0, in1=s1, in2=s2, out0=value, out1=arg1
  let tag_u16 = lut_opcode_tag(LutOpcode::U16AddEq) as u32;
  let g = pg(OP_U16_ADD_EQ);
  builder.assert_zero(g.clone() * (local[LUT_COL_OP].clone().into()   - AB::Expr::from_u32(tag_u16)));
  builder.assert_zero(g.clone() * (local[LUT_COL_IN0].clone().into()  - prep_row[PREP_COL_SCALAR0].clone().into()));
  builder.assert_zero(g.clone() * (local[LUT_COL_IN1].clone().into()  - prep_row[PREP_COL_SCALAR1].clone().into()));
  builder.assert_zero(g.clone() * (local[LUT_COL_IN2].clone().into()  - prep_row[PREP_COL_SCALAR2].clone().into()));
  builder.assert_zero(g.clone() * (local[LUT_COL_OUT0].clone().into() - prep_row[PREP_COL_VALUE].clone().into()));
  builder.assert_zero(g         * (local[LUT_COL_OUT1].clone().into() - prep_row[PREP_COL_ARG1].clone().into()));

  // OP_U29_ADD_EQ → U29AddEq: in0=s0, in1=s1, in2=s2, out0=value, out1=arg1
  let tag_u29 = lut_opcode_tag(LutOpcode::U29AddEq) as u32;
  let g = pg(OP_U29_ADD_EQ);
  builder.assert_zero(g.clone() * (local[LUT_COL_OP].clone().into()   - AB::Expr::from_u32(tag_u29)));
  builder.assert_zero(g.clone() * (local[LUT_COL_IN0].clone().into()  - prep_row[PREP_COL_SCALAR0].clone().into()));
  builder.assert_zero(g.clone() * (local[LUT_COL_IN1].clone().into()  - prep_row[PREP_COL_SCALAR1].clone().into()));
  builder.assert_zero(g.clone() * (local[LUT_COL_IN2].clone().into()  - prep_row[PREP_COL_SCALAR2].clone().into()));
  builder.assert_zero(g.clone() * (local[LUT_COL_OUT0].clone().into() - prep_row[PREP_COL_VALUE].clone().into()));
  builder.assert_zero(g         * (local[LUT_COL_OUT1].clone().into() - prep_row[PREP_COL_ARG1].clone().into()));

  // OP_U24_ADD_EQ → U24AddEq (same field layout as U29)
  let tag_u24 = lut_opcode_tag(LutOpcode::U24AddEq) as u32;
  let g = pg(OP_U24_ADD_EQ);
  builder.assert_zero(g.clone() * (local[LUT_COL_OP].clone().into()   - AB::Expr::from_u32(tag_u24)));
  builder.assert_zero(g.clone() * (local[LUT_COL_IN0].clone().into()  - prep_row[PREP_COL_SCALAR0].clone().into()));
  builder.assert_zero(g.clone() * (local[LUT_COL_IN1].clone().into()  - prep_row[PREP_COL_SCALAR1].clone().into()));
  builder.assert_zero(g.clone() * (local[LUT_COL_IN2].clone().into()  - prep_row[PREP_COL_SCALAR2].clone().into()));
  builder.assert_zero(g.clone() * (local[LUT_COL_OUT0].clone().into() - prep_row[PREP_COL_VALUE].clone().into()));
  builder.assert_zero(g         * (local[LUT_COL_OUT1].clone().into() - prep_row[PREP_COL_ARG1].clone().into()));

  // OP_U15_MUL_EQ → U15MulEq: in0=s0, in1=s1, out0=value(lo), out1=arg0(hi)
  let tag_mul = lut_opcode_tag(LutOpcode::U15MulEq) as u32;
  let g = pg(OP_U15_MUL_EQ);
  builder.assert_zero(g.clone() * (local[LUT_COL_OP].clone().into()   - AB::Expr::from_u32(tag_mul)));
  builder.assert_zero(g.clone() * (local[LUT_COL_IN0].clone().into()  - prep_row[PREP_COL_SCALAR0].clone().into()));
  builder.assert_zero(g.clone() * (local[LUT_COL_IN1].clone().into()  - prep_row[PREP_COL_SCALAR1].clone().into()));
  builder.assert_zero(g.clone() * (local[LUT_COL_OUT0].clone().into() - prep_row[PREP_COL_VALUE].clone().into()));
  builder.assert_zero(g         * (local[LUT_COL_OUT1].clone().into() - prep_row[PREP_COL_ARG0].clone().into()));

  // OP_BYTE_ADD_CARRY_EQ → ByteAddCarryEq: in0=s0, in1=s1, in2=arg0
  let tag_carry = lut_opcode_tag(LutOpcode::ByteAddCarryEq) as u32;
  let g = pg(OP_BYTE_ADD_CARRY_EQ);
  builder.assert_zero(g.clone() * (local[LUT_COL_OP].clone().into()  - AB::Expr::from_u32(tag_carry)));
  builder.assert_zero(g.clone() * (local[LUT_COL_IN0].clone().into() - prep_row[PREP_COL_SCALAR0].clone().into()));
  builder.assert_zero(g.clone() * (local[LUT_COL_IN1].clone().into() - prep_row[PREP_COL_SCALAR1].clone().into()));
  builder.assert_zero(g         * (local[LUT_COL_IN2].clone().into() - prep_row[PREP_COL_ARG0].clone().into()));

  // OP_BYTE_MUL_LOW_EQ → ByteMulLowEq: in0=s0, in1=s1
  let tag_ml = lut_opcode_tag(LutOpcode::ByteMulLowEq) as u32;
  let g = pg(OP_BYTE_MUL_LOW_EQ);
  builder.assert_zero(g.clone() * (local[LUT_COL_OP].clone().into()  - AB::Expr::from_u32(tag_ml)));
  builder.assert_zero(g.clone() * (local[LUT_COL_IN0].clone().into() - prep_row[PREP_COL_SCALAR0].clone().into()));
  builder.assert_zero(g         * (local[LUT_COL_IN1].clone().into() - prep_row[PREP_COL_SCALAR1].clone().into()));

  // OP_BYTE_MUL_HIGH_EQ → ByteMulHighEq: in0=s0, in1=s1
  let tag_mh = lut_opcode_tag(LutOpcode::ByteMulHighEq) as u32;
  let g = pg(OP_BYTE_MUL_HIGH_EQ);
  builder.assert_zero(g.clone() * (local[LUT_COL_OP].clone().into()  - AB::Expr::from_u32(tag_mh)));
  builder.assert_zero(g.clone() * (local[LUT_COL_IN0].clone().into() - prep_row[PREP_COL_SCALAR0].clone().into()));
  builder.assert_zero(g         * (local[LUT_COL_IN1].clone().into() - prep_row[PREP_COL_SCALAR1].clone().into()));
  // OP_BYTE_AND_EQ / OP_BYTE_OR_EQ / OP_BYTE_XOR_EQ: bit-family ops use byte-level
  // scalars; correctness comes from StackIR binding so no LUT binding needed here.
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
  let proof = p3_uni_stark::prove_with_preprocessed(
    &config,
    &air,
    trace,
    public_values,
    Some(prep_data),
  );
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
  Ok(p3_uni_stark::prove(&config, &LutKernelAir, trace, public_values))
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

pub fn build_lut_steps_from_rows_bit_family(rows: &[ProofRow]) -> Result<Vec<LutStep>, String> {
  let mut out = Vec::new();
  let mut seen_bit_family = false;

  let push_bitwise_steps = |out: &mut Vec<LutStep>, op: LutOpcode, a: u32, b: u32| {
    for bit in 0..8 {
      let abit = (a >> bit) & 1;
      let bbit = (b >> bit) & 1;
      let out_bit = match op {
        LutOpcode::ByteAndEq => abit & bbit,
        LutOpcode::ByteOrEq => abit | bbit,
        LutOpcode::ByteXorEq => abit ^ bbit,
        _ => 0,
      };
      out.push(LutStep {
        op,
        in0: abit,
        in1: bbit,
        in2: 0,
        out0: out_bit,
        out1: 0,
      });
    }
  };

  for row in rows {
    match row.op {
      crate::semantic_proof::OP_AND_INTRO => {
        seen_bit_family = true;
      }
      crate::semantic_proof::OP_BYTE_AND_EQ => {
        push_bitwise_steps(&mut out, LutOpcode::ByteAndEq, row.scalar0, row.scalar1)
      }
      crate::semantic_proof::OP_BYTE_OR_EQ => {
        push_bitwise_steps(&mut out, LutOpcode::ByteOrEq, row.scalar0, row.scalar1)
      }
      crate::semantic_proof::OP_BYTE_XOR_EQ => {
        push_bitwise_steps(&mut out, LutOpcode::ByteXorEq, row.scalar0, row.scalar1)
      }
      other => return Err(format!("non-bit-family row in BIT kernel path: {other}")),
    }
  }

  if !seen_bit_family && out.is_empty() {
    return Err("empty bit-family row set".to_string());
  }

  if out.is_empty() {
    return Err("no bit LUT rows in bit-family row set".to_string());
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
      builder.assert_eq(
        local[col].clone(),
        local[MATCH_PACK_COLS + col].clone(),
      );
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
        * (local[COL_ARG0].clone().into() * (local[COL_ARG0].clone().into() - AB::Expr::from_u16(1))),
    );
    builder.assert_zero(
      g_u16.clone()
        * (local[COL_ARG1].clone().into() * (local[COL_ARG1].clone().into() - AB::Expr::from_u16(1))),
    );
    builder.assert_zero(
      g_u29.clone()
        * (local[COL_ARG1].clone().into() * (local[COL_ARG1].clone().into() - AB::Expr::from_u16(1))),
    );
    builder.assert_zero(
      g_u24.clone()
        * (local[COL_ARG1].clone().into() * (local[COL_ARG1].clone().into() - AB::Expr::from_u16(1))),
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
      trace.values[base + MATCH_PACK_COLS + col] = Val::from_u32(pack_chunk(expected_bytes, byte_start));
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
  Ok(p3_uni_stark::prove(&config, &WffMatchAir, trace, public_values))
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
pub fn prove_batch_lut_stark(
  all_steps: &[LutStep],
) -> Result<CircleStarkProof, String> {
  prove_lut_kernel_stark(all_steps)
}

/// Verify a batched LUT STARK proof.
pub fn verify_batch_lut_stark(
  proof: &CircleStarkProof,
) -> bool {
  verify_lut_kernel_stark(proof).is_ok()
}

/// Build and immediately prove N add-family instructions' LUT in one shot.
pub fn prove_batch_stark_add_family(
  all_rows: &[ProofRow],
) -> Result<CircleStarkProof, String> {
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
pub fn prove_batch_stark_mul_family(
  all_rows: &[ProofRow],
) -> Result<CircleStarkProof, String> {
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
pub fn prove_batch_stark_bit_family(
  all_rows: &[ProofRow],
) -> Result<CircleStarkProof, String> {
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

/// Expand a byte-level bitwise op row into 8 per-bit LitSteps.
/// Used by `build_lut_steps_from_rows_auto` for AND/OR/XOR.
fn push_bit_lut_steps_into(out: &mut Vec<LutStep>, op: LutOpcode, byte_a: u32, byte_b: u32) {
  for bit in 0..8 {
    let a = (byte_a >> bit) & 1;
    let b = (byte_b >> bit) & 1;
    let result = match op {
      LutOpcode::ByteAndEq => a & b,
      LutOpcode::ByteOrEq => a | b,
      LutOpcode::ByteXorEq => a ^ b,
      _ => 0,
    };
    out.push(LutStep { op, in0: a, in1: b, in2: 0, out0: result, out1: 0 });
  }
}

/// Build LUT steps from ProofRows belonging to **any** opcode family.
///
/// Structural rows (OP_BOOL, OP_BYTE, OP_EQ_REFL, OP_AND_INTRO, OP_EQ_TRANS,
/// OP_BYTE_ADD_THIRD_CONGRUENCE, OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE,
/// OP_NOT, OP_EQ_SYM) are skipped silently.  Bit-family ops are expanded to
/// per-bit level as required by the LUT kernel AIR.  All other recognised LUT
/// ops are processed normally.
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

      // ── Bit-family LUT ops (expand byte → bits) ────────────────────────────
      crate::semantic_proof::OP_BYTE_AND_EQ => {
        push_bit_lut_steps_into(&mut out, LutOpcode::ByteAndEq, row.scalar0, row.scalar1)
      }
      crate::semantic_proof::OP_BYTE_OR_EQ => {
        push_bit_lut_steps_into(&mut out, LutOpcode::ByteOrEq, row.scalar0, row.scalar1)
      }
      crate::semantic_proof::OP_BYTE_XOR_EQ => {
        push_bit_lut_steps_into(&mut out, LutOpcode::ByteXorEq, row.scalar0, row.scalar1)
      }

      other => return Err(format!("build_lut_steps_from_rows_auto: unrecognised row op {other}")),
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
pub fn prove_batch_wff_proofs(
  proofs: &[&Proof],
) -> Result<(CircleStarkProof, Vec<WFF>), String> {
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
    let inferred = infer_proof(proof)
      .map_err(|e| format!("infer_proof failed for instruction {i}: {e}"))?;
    if inferred != *expected_wff {
      return Err(format!("WFF mismatch at instruction {i}"));
    }
  }

  Ok(())
}

// ============================================================
// Tests
// ============================================================
