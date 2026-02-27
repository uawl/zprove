//! Stage A semantic AIR and prove/verify functions.
//!
//! Stage 1: prove inferred-WFF semantic constraints over compiled ProofRows.
//! Stage 2: prove inferred WFF equals the public WFF via serialized equality trace.

use super::types::{
  CircleStarkProof, CircleStarkVerifyResult, RECEIPT_BIND_TAG_WFF, Val,
  default_receipt_bind_public_values_for_tag, make_circle_config,
};

use crate::semantic_proof::{
  NUM_PROOF_COLS, Proof, ProofRow, RET_WFF_AND, WFF, compile_proof, infer_proof,
  verify_compiled,
};

use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;

// ============================================================
// Stage-A column layout  (mirrors semantic_proof::ProofRow)
// ============================================================

const COL_OP: usize = 0;
const COL_SCALAR0: usize = 1;
const COL_SCALAR1: usize = 2;
const COL_SCALAR2: usize = 3;
const COL_ARG0: usize = 4;
const COL_ARG1: usize = 5;
const COL_ARG2: usize = 6;
const COL_VALUE: usize = 7;
const COL_RET_TY: usize = 8;

/// Number of (inferred, expected) column-pairs per WFF-match trace row.
/// Bytes per row = MATCH_PACK_COLS × MATCH_PACK_BYTES = 4 × 3 = 12.
const MATCH_PACK_COLS: usize = 4;
/// Bytes packed into each M31 field element (max 3 since 0xFFFFFF < M31 = 2^31-1).
const MATCH_PACK_BYTES: usize = 3;
/// Total columns: first MATCH_PACK_COLS are inferred, next MATCH_PACK_COLS are expected.
const NUM_WFF_MATCH_COLS: usize = MATCH_PACK_COLS * 2;

// ============================================================
// Proof wrapper
// ============================================================

pub struct StageAProof {
  pub inferred_wff_proof: CircleStarkProof,
  pub public_wff_match_proof: CircleStarkProof,
  pub expected_public_wff: WFF,
}

// ============================================================
// Routing helpers
// ============================================================

fn route_stage_a_row_op(op: u32) -> Result<(), String> {
  use crate::semantic_proof::{
    OP_AND_INTRO, OP_BOOL, OP_BYTE, OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE,
    OP_BYTE_ADD_THIRD_CONGRUENCE, OP_EQ_REFL, OP_EQ_TRANS, OP_U24_ADD_EQ, OP_U29_ADD_EQ,
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
// AIR definitions
// ============================================================

/// AIR for Stage-1 semantic constraints using `ProofRow` trace columns.
///
/// The kernel verifies u29/u24 add equality constraints encoded in
/// `OP_U29_ADD_EQ` / `OP_U24_ADD_EQ` rows.
pub struct StageASemanticAir;

impl<F> BaseAir<F> for StageASemanticAir {
  fn width(&self) -> usize {
    NUM_PROOF_COLS
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

    let local_cases = [(g_u29, local_cout_u29), (g_u24, local_cout_u24)];
    let next_cases = [(ng_u29, next_cin_u29), (ng_u24, next_cin_u24)];

    for (lg, lcout) in local_cases.iter() {
      for (ng, ncin) in next_cases.iter() {
        builder
          .when_transition()
          .assert_zero(lg.clone() * ng.clone() * (ncin.clone() - lcout.clone()));
      }
    }
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
    for col in 0..MATCH_PACK_COLS {
      builder.assert_eq(local[col].clone(), local[MATCH_PACK_COLS + col].clone());
    }
  }
}

// ============================================================
// Trace builders
// ============================================================

pub fn generate_stage_a_semantic_trace(rows: &[ProofRow]) -> Result<RowMajorMatrix<Val>, String> {
  let mut semantic_rows: Vec<ProofRow> = Vec::new();

  for (i, row) in rows.iter().enumerate() {
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
        crate::semantic_proof::OP_U29_ADD_EQ | crate::semantic_proof::OP_U24_ADD_EQ => {
          row.arg1 as u16
        }
        _ => 0,
      })
      .unwrap_or(0);

    for i in semantic_len..n_rows {
      let carry_out = 0u16;
      let _ = carry_out;

      let base = i * NUM_PROOF_COLS;
      trace.values[base + COL_OP] = Val::from_u16(crate::semantic_proof::OP_U29_ADD_EQ as u16);
      trace.values[base + COL_SCALAR0] = Val::from_u16(0);
      trace.values[base + COL_SCALAR1] = Val::from_u16(0);
      trace.values[base + COL_SCALAR2] = Val::from_u16(carry_in);
      trace.values[base + COL_ARG0] = Val::from_u16(carry_in);
      trace.values[base + COL_ARG1] = Val::from_u16(0);
      trace.values[base + COL_ARG2] = Val::from_u16(0);
      trace.values[base + COL_VALUE] = Val::from_u16(carry_in);
      trace.values[base + COL_RET_TY] = Val::from_u16(RET_WFF_AND as u16);

      carry_in = 0;
    }
  }

  Ok(trace)
}

/// Build a WFF-match trace from inferred/public WFF byte serializations.
pub fn generate_wff_match_trace_from_wffs(
  inferred_wff: &WFF,
  public_wff: &WFF,
) -> Result<RowMajorMatrix<Val>, String> {
  let inferred_bytes = super::types::serialize_wff_bytes(inferred_wff);
  let expected_bytes = super::types::serialize_wff_bytes(public_wff);
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

  let bytes_per_row = MATCH_PACK_COLS * MATCH_PACK_BYTES;
  let n_rows = inferred_bytes.len().div_ceil(bytes_per_row);
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

  Ok(trace)
}

// ============================================================
// Prove & Verify
// ============================================================

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

pub fn prove_and_verify_inferred_wff_stark(rows: &[ProofRow]) -> bool {
  let proved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
    prove_compiled_rows_stark(rows)
  }));
  let proof = match proved {
    Ok(Ok(proof)) => proof,
    Ok(Err(_)) => return false,
    Err(_) => return false,
  };
  verify_inferred_wff_stark(&proof).is_ok()
}

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

pub fn prove_and_verify_wff_match_stark(private_pi: &Proof, public_wff: &WFF) -> bool {
  prove_and_verify_expected_wff_match_stark(private_pi, public_wff)
}

pub fn prove_and_verify_expected_wff_match_stark(private_pi: &Proof, public_wff: &WFF) -> bool {
  let proved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
    prove_expected_wff_match_stark(private_pi, public_wff)
  }));
  let proof = match proved {
    Ok(Ok(proof)) => proof,
    Ok(Err(_)) => return false,
    Err(_) => return false,
  };
  verify_wff_match_stark(&proof).is_ok()
}

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
