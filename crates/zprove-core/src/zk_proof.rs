//! Circle STARK proof generation and verification for semantic proof rows.
//!
//! Stage A/B overview:
//! - Stage 1 proves inferred-WFF semantic constraints over compiled `ProofRow`s.
//! - Stage 2 proves inferred WFF equals public WFF via serialized equality trace.
//!
//! The current Stage-1 semantic AIR kernel enforces the byte-add equality rows
//! (`OP_BYTE_ADD_EQ`) embedded in `ProofRow` encoding.

use crate::sementic_proof::{
  NUM_PROOF_COLS,
  OP_AND_INTRO,
  OP_BYTE_ADD_CARRY_EQ,
  OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE,
  OP_BYTE_ADD_EQ,
  OP_BYTE_ADD_THIRD_CONGRUENCE,
  OP_BYTE_AND_EQ,
  OP_BYTE_MUL_HIGH_EQ,
  OP_BYTE_MUL_LOW_EQ,
  OP_BYTE_OR_EQ,
  OP_BYTE_XOR_EQ,
  OP_BYTE_XOR,
  OP_EQ_REFL,
  OP_EQ_SYM,
  OP_EQ_TRANS,
  OP_ITE_FALSE_EQ,
  OP_ITE_TRUE_EQ,
  infer_proof,
  Proof,
  ProofRow,
  Term,
  RET_WFF_EQ,
  WFF,
  compile_proof,
  verify_compiled,
};

use p3_air::{Air, AirBuilder, BaseAir};
use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_circle::CirclePcs;
use p3_commit::ExtensionMmcs;
use p3_field::extension::BinomialExtensionField;
use p3_field::PrimeCharacteristicRing;
use p3_fri::FriParameters;
use p3_keccak::Keccak256Hash;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_mersenne_31::Mersenne31;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use p3_uni_stark::StarkConfig;

// ============================================================
// Type aliases for Circle STARK over M31
// ============================================================

pub type Val = Mersenne31;
pub type Challenge = BinomialExtensionField<Val, 3>;

type ByteHash = Keccak256Hash;
type FieldHash = SerializingHasher<ByteHash>;
type Compress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
type ValMmcs = MerkleTreeMmcs<Val, u8, FieldHash, Compress, 32>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;

pub type Pcs = CirclePcs<Val, ValMmcs, ChallengeMmcs>;
pub type CircleStarkConfig = StarkConfig<Pcs, Challenge, Challenger>;

pub type CircleStarkProof = p3_uni_stark::Proof<CircleStarkConfig>;
pub type CircleStarkVerifyResult = Result<(), p3_uni_stark::VerificationError<p3_uni_stark::PcsError<CircleStarkConfig>>>;

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

const MATCH_COL_INFERRED: usize = 0;
const MATCH_COL_EXPECTED: usize = 1;
const NUM_WFF_MATCH_COLS: usize = 2;

pub struct StageAProof {
  pub inferred_wff_proof: CircleStarkProof,
  pub public_wff_match_proof: CircleStarkProof,
  expected_public_wff: WFF,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StageARowClass {
  SemanticConstraint,
  Structural,
}

fn route_stage_a_row_op(op: u32) -> Result<StageARowClass, String> {
  match op {
    OP_BYTE_ADD_EQ => Ok(StageARowClass::SemanticConstraint),
    op if op <= OP_BYTE_XOR => Ok(StageARowClass::Structural),
    OP_AND_INTRO
    | OP_EQ_TRANS
    | OP_EQ_REFL
    | OP_EQ_SYM
    | OP_BYTE_ADD_CARRY_EQ
    | OP_BYTE_ADD_THIRD_CONGRUENCE
    | OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE
    | OP_BYTE_MUL_LOW_EQ
    | OP_BYTE_MUL_HIGH_EQ
    | OP_BYTE_AND_EQ
    | OP_BYTE_OR_EQ
    | OP_BYTE_XOR_EQ
    | OP_ITE_TRUE_EQ
    | OP_ITE_FALSE_EQ => Ok(StageARowClass::Structural),
    other => Err(format!("unsupported Stage A proof-row op: {other}")),
  }
}

fn has_stage_a_semantic_rows(rows: &[ProofRow]) -> Result<bool, String> {
  let mut has_semantic = false;
  for row in rows {
    match route_stage_a_row_op(row.op)? {
      StageARowClass::SemanticConstraint => has_semantic = true,
      StageARowClass::Structural => {}
    }
  }
  Ok(has_semantic)
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

impl<AB: AirBuilder> Air<AB> for WffMatchAir {
  fn eval(&self, builder: &mut AB) {
    let main = builder.main();
    let local = main.row_slice(0).expect("empty trace");
    let local = &*local;

    builder.assert_eq(local[MATCH_COL_INFERRED].clone(), local[MATCH_COL_EXPECTED].clone());
  }
}

impl<AB: AirBuilder> Air<AB> for StageASemanticAir {
  fn eval(&self, builder: &mut AB) {
    let main = builder.main();
    let local = main.row_slice(0).expect("empty trace");
    let next = main.row_slice(1).expect("single-row trace");

    let local = &*local;
    let next = &*next;

    let op_const = AB::Expr::from_u16(OP_BYTE_ADD_EQ as u16);
    let ret_ty_const = AB::Expr::from_u16(RET_WFF_EQ as u16);

    let c256 = AB::Expr::from_u16(256);

    let a = local[COL_SCALAR0].clone();
    let b = local[COL_SCALAR1].clone();
    let carry_out = local[COL_SCALAR2].clone();
    let carry_in = local[COL_ARG0].clone();
    let expected = local[COL_ARG1].clone();
    let sum = local[COL_VALUE].clone();

    // ── Constraint 1: a + b + carry_in = sum + 256 * carry_out ──
    // Equivalently: a + b + carry_in - sum - 256 * carry_out = 0
    builder.assert_zero(
      a.clone().into()
        + b.clone().into()
        + carry_in.clone().into()
        - sum.clone().into()
        - c256 * carry_out.clone().into(),
    );

    // ── Constraint 2: sum = expected ──
    builder.assert_eq(sum.clone(), expected.clone());

    // ── Constraint 3: carry_out ∈ {0, 1} ──
    builder.assert_bool(carry_out.clone());

    // ── Constraint 4: carry_in ∈ {0, 1} ──
    builder.assert_bool(carry_in.clone());

    // ── Constraint 5: fixed opcode/ret-ty tags for ProofRow encoding ──
    builder.assert_zero(local[COL_OP].clone().into() - op_const);
    builder.assert_zero(local[COL_RET_TY].clone().into() - ret_ty_const);
    builder.assert_zero(local[COL_ARG2].clone());

    // ── Constraint 6: First row carry_in = 0 ──
    builder.when_first_row().assert_zero(carry_in.clone());

    // ── Constraint 7: Transition carry chain ──
    // next.carry_in = local.carry_out
    builder
      .when_transition()
      .assert_eq(next[COL_ARG0].clone(), carry_out.clone());
  }
}

// ============================================================
// Config builder
// ============================================================

/// Build a Circle STARK configuration over M31 with Keccak hashing.
pub fn make_circle_config() -> CircleStarkConfig {
  let byte_hash = Keccak256Hash {};
  let field_hash = SerializingHasher::new(byte_hash);
  let compress = CompressionFunctionFromHasher::new(byte_hash);
  let val_mmcs = MerkleTreeMmcs::new(field_hash, compress);
  let challenge_mmcs = ExtensionMmcs::new(val_mmcs.clone());

  let fri_params = FriParameters {
    log_blowup: 1,
    log_final_poly_len: 0,
    num_queries: 40,
    commit_proof_of_work_bits: 0,
    query_proof_of_work_bits: 8,
    mmcs: challenge_mmcs,
  };

  let pcs = CirclePcs {
    mmcs: val_mmcs,
    fri_params,
    _phantom: core::marker::PhantomData,
  };

  let challenger = SerializingChallenger32::from_hasher(vec![], byte_hash);
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
  let mut semantic_rows: Vec<&ProofRow> = Vec::new();
  for row in rows {
    match route_stage_a_row_op(row.op)? {
      StageARowClass::SemanticConstraint => semantic_rows.push(row),
      StageARowClass::Structural => {}
    }
  }

  if semantic_rows.is_empty() {
    return Err("no stage-a semantic rows in compiled proof".to_string());
  }

  let n_rows = semantic_rows.len();
  let mut trace = RowMajorMatrix::new(Val::zero_vec(n_rows * NUM_PROOF_COLS), NUM_PROOF_COLS);

  for (i, row) in semantic_rows.iter().enumerate() {
    let base = i * NUM_PROOF_COLS;
    trace.values[base + COL_OP] = Val::from_u16(row.op as u16);
    trace.values[base + COL_SCALAR0] = Val::from_u16(row.scalar0 as u16);
    trace.values[base + COL_SCALAR1] = Val::from_u16(row.scalar1 as u16);
    trace.values[base + COL_SCALAR2] = Val::from_u16(row.scalar2 as u16);
    trace.values[base + COL_ARG0] = Val::from_u16(row.arg0 as u16);
    trace.values[base + COL_ARG1] = Val::from_u16(row.arg1 as u16);
    trace.values[base + COL_ARG2] = Val::from_u16(row.arg2 as u16);
    trace.values[base + COL_VALUE] = Val::from_u16(row.value as u16);
    trace.values[base + COL_RET_TY] = Val::from_u16(row.ret_ty as u16);
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

  let padded_rows = inferred_bytes.len().next_power_of_two();
  let mut trace = RowMajorMatrix::new(Val::zero_vec(padded_rows * NUM_WFF_MATCH_COLS), NUM_WFF_MATCH_COLS);
  for row in 0..inferred_bytes.len() {
    let base = row * NUM_WFF_MATCH_COLS;
    trace.values[base + MATCH_COL_INFERRED] = Val::from_u16(inferred_bytes[row] as u16);
    trace.values[base + MATCH_COL_EXPECTED] = Val::from_u16(expected_bytes[row] as u16);
  }

  for row in inferred_bytes.len()..padded_rows {
    let base = row * NUM_WFF_MATCH_COLS;
    trace.values[base + MATCH_COL_INFERRED] = Val::from_u16(0);
    trace.values[base + MATCH_COL_EXPECTED] = Val::from_u16(0);
  }

  Ok(trace)
}

/// Prove compiled ProofRows using the Stage-A backend.
pub fn prove_compiled_rows_stark(
  rows: &[ProofRow],
) -> Result<CircleStarkProof, String> {
  prove_inferred_wff_stark(rows)
}

/// Stage-1 generic ZKP: prove inferred WFF validity from compiled ProofRows.
pub fn prove_inferred_wff_stark(rows: &[ProofRow]) -> Result<CircleStarkProof, String> {
  for row in rows {
    let _ = route_stage_a_row_op(row.op)?;
  }

  if !has_stage_a_semantic_rows(rows)? {
    return Err("stage-a semantic kernel unavailable for this proof-row set".to_string());
  }

  let config = make_circle_config();
  let trace = generate_stage_a_semantic_trace(rows)?;
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
  let proved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| prove_compiled_rows_stark(rows)));
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
  prove_expected_wff_match_stark(private_pi, public_wff)
}

/// Stage-2 generic ZKP: prove inferred WFF equals externally expected WFF.
pub fn prove_expected_wff_match_stark(
  private_pi: &Proof,
  public_wff: &WFF,
) -> Result<CircleStarkProof, String> {
  let inferred_wff = infer_proof(private_pi)
    .map_err(|err| format!("failed to infer WFF from private proof: {err}"))?;

  let rows = compile_proof(private_pi);
  for row in rows {
    let _ = route_stage_a_row_op(row.op)?;
  }

  let config = make_circle_config();
  let trace = generate_wff_match_trace_from_wffs(&inferred_wff, public_wff)?;
  Ok(p3_uni_stark::prove(&config, &WffMatchAir, trace, &[]))
}

/// Verify stage-2 WFF match proof.
pub fn verify_wff_match_stark(proof: &CircleStarkProof) -> CircleStarkVerifyResult {
  let config = make_circle_config();
  p3_uni_stark::verify(&config, &WffMatchAir, proof, &[])
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
// Tests
// ============================================================

#[cfg(test)]
mod tests {
  use super::*;
  use crate::sementic_proof::{compile_proof, infer_proof, prove_add, wff_add};

  fn u256_bytes(val: u128) -> [u8; 32] {
    let mut b = [0u8; 32];
    b[16..32].copy_from_slice(&val.to_be_bytes());
    b
  }

  fn compute_add(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    let mut result = [0u8; 32];
    let mut carry = 0u16;
    for i in (0..32).rev() {
      let sum = a[i] as u16 + b[i] as u16 + carry;
      result[i] = (sum & 0xFF) as u8;
      carry = sum >> 8;
    }
    result
  }

  #[test]
  fn test_stage_a_semantic_trace_generation() {
    let a = u256_bytes(1000);
    let b = u256_bytes(2000);
    let c = compute_add(&a, &b);
    let semantic = prove_add(&a, &b, &c);
    let rows = compile_proof(&semantic);
    let trace = generate_stage_a_semantic_trace(&rows).expect("semantic trace generation should succeed");
    assert_eq!(trace.width(), NUM_PROOF_COLS);
    assert_eq!(trace.height(), 32);
  }

  #[test]
  fn test_prove_and_verify_stage1_simple_semantic_rows() {
    let a = u256_bytes(1000);
    let b = u256_bytes(2000);
    let c = compute_add(&a, &b);

    let semantic = prove_add(&a, &b, &c);
    let rows = compile_proof(&semantic);
    let proof = prove_inferred_wff_stark(&rows).expect("stage-1 proof should succeed");
    verify_inferred_wff_stark(&proof).expect("stage-1 verification failed");
  }

  #[test]
  fn test_prove_and_verify_stage1_overflow_semantic_rows() {
    let a = [0xFF; 32];
    let mut b = [0u8; 32];
    b[31] = 1;
    let c = compute_add(&a, &b);

    let semantic = prove_add(&a, &b, &c);
    let rows = compile_proof(&semantic);
    let proof = prove_inferred_wff_stark(&rows).expect("stage-1 proof should succeed");
    verify_inferred_wff_stark(&proof).expect("stage-1 verification failed");
  }

  #[test]
  fn test_prove_and_verify_stage1_large_semantic_rows() {
    let mut a = [0xABu8; 32];
    a[0] = 0x7F;
    let mut b = [0xCDu8; 32];
    b[0] = 0x3E;
    let c = compute_add(&a, &b);

    let semantic = prove_add(&a, &b, &c);
    let rows = compile_proof(&semantic);
    let proof = prove_inferred_wff_stark(&rows).expect("stage-1 proof should succeed");
    verify_inferred_wff_stark(&proof).expect("stage-1 verification failed");
  }

  #[test]
  fn test_prove_and_verify_stage1_zero_semantic_rows() {
    let a = [0u8; 32];
    let b = [0u8; 32];
    let c = [0u8; 32];

    let semantic = prove_add(&a, &b, &c);
    let rows = compile_proof(&semantic);
    let proof = prove_inferred_wff_stark(&rows).expect("stage-1 proof should succeed");
    verify_inferred_wff_stark(&proof).expect("stage-1 verification failed");
  }

  #[test]
  fn test_prove_and_verify_stage1_from_compiled_rows() {
    let a = u256_bytes(1234);
    let b = u256_bytes(5678);
    let c = compute_add(&a, &b);

    let semantic = prove_add(&a, &b, &c);
    let rows = compile_proof(&semantic);
    assert!(prove_and_verify_compiled_rows_stark(&rows));
  }

  #[test]
  fn test_prove_and_verify_stage2_wff_match() {
    let a = u256_bytes(1111);
    let b = u256_bytes(2222);
    let c = compute_add(&a, &b);

    let semantic = prove_add(&a, &b, &c);
    let public_wff = wff_add(&a, &b, &c);
    assert!(prove_and_verify_wff_match_stark(&semantic, &public_wff));
  }

  #[test]
  fn test_wff_match_fails_on_wrong_output() {
    let a = u256_bytes(100);
    let b = u256_bytes(200);
    let c = compute_add(&a, &b);
    let mut wrong = c;
    wrong[31] ^= 1;

    let semantic = prove_add(&a, &b, &c);
    let wrong_public_wff = wff_add(&a, &b, &wrong);
    assert!(!prove_and_verify_wff_match_stark(&semantic, &wrong_public_wff));
  }

  #[test]
  fn test_stage_a_success() {
    let a = u256_bytes(777);
    let b = u256_bytes(888);
    let c = compute_add(&a, &b);

    let public_wff = wff_add(&a, &b, &c);
    let private_pi = prove_add(&a, &b, &c);
    assert!(prove_and_verify_stage_a(&public_wff, &private_pi));
  }

  #[test]
  fn test_stage_a_fails_on_wrong_public_wff() {
    let a = u256_bytes(10);
    let b = u256_bytes(20);
    let c = compute_add(&a, &b);
    let mut wrong = c;
    wrong[31] ^= 1;

    let private_pi = prove_add(&a, &b, &c);
    let wrong_public_wff = wff_add(&a, &b, &wrong);
    assert!(!prove_and_verify_stage_a(&wrong_public_wff, &private_pi));
  }

  #[test]
  fn test_stage_a_supports_non_add_infer_path() {
    let private_pi = Proof::EqRefl(Term::Byte(42));
    let public_wff = infer_proof(&private_pi).expect("infer_proof should succeed");
    assert!(!prove_and_verify_stage_a(&public_wff, &private_pi));
  }
}
