//! LUT (Look-Up Table) kernel AIR, trace builders, and prove/verify functions.
//!
//! Covers single-instruction and batch LUT STARKs, byte-table query helpers,
//! cross-family batch WFF proving, and manifest row validation.

use super::batch::BatchProofRowsManifest;
use super::preprocessed::{
  PREP_COL_ARG0, PREP_COL_ARG1, PREP_COL_BATCH_DIGEST_START, PREP_COL_BATCH_N, PREP_COL_OP, PREP_COL_SCALAR0, PREP_COL_SCALAR1, PREP_COL_SCALAR2,
  PREP_COL_VALUE, build_batch_proof_rows_preprocessed_matrix,
};
use super::stack_ir::prove_and_verify_stack_ir_scaffold_stark;
use super::types::{
  CircleStarkConfig, CircleStarkProof, CircleStarkVerifyResult, RECEIPT_BIND_TAG_LUT, Val,
  default_receipt_bind_public_values_for_tag, make_circle_config,
};
use crate::semantic_proof::{Proof, ProofRow, compile_proof, infer_proof};

use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, PairBuilder};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{PreprocessedProverData, PreprocessedVerifierKey};

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
/// Axiom selector: set for per-opcode axiom rows (prove_instruction axiom proofs).
const LUT_COL_SEL_AXIOM: usize = 16;
pub const NUM_LUT_COLS: usize = 17;

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

/// Op tag used in the LUT trace for axiom rows.
const LUT_AXIOM_TAG: u16 = 11;

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
    trace.values[base + lut_op_to_sel_col(step.op)] = Val::from_u16(1);
  }

  for i in steps.len()..n_rows {
    let base = i * NUM_LUT_COLS;
    trace.values[base + LUT_COL_OP] = Val::from_u16(lut_opcode_tag(LutOpcode::U29AddEq));
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
/// `BatchLutKernelAirWithPrep` to avoid code duplication.
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
  let s_axiom = local[LUT_COL_SEL_AXIOM].clone();

  // ── (1) Boolean: sel_i * (1 - sel_i) = 0 ──
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
    s_axiom.clone(),
  ] {
    builder.assert_bool(s);
  }

  // ── (2) One-hot: Σ sel_i = 1 ──
  let sel_sum = s_u29.clone().into()
    + s_u24.clone().into()
    + s_u15.clone().into()
    + s_bit.clone().into()
    + s_mul.clone().into()
    + s_mul_low.clone().into()
    + s_mul_high.clone().into()
    + s_and.clone().into()
    + s_or.clone().into()
    + s_xor.clone().into()
    + s_axiom.clone().into();
  builder.assert_one(sel_sum);

  // ── (3) op = Σ(sel_i · tag_i) ──
  let op_reconstruct = s_u29.clone().into()
    * AB::Expr::from_u16(lut_opcode_tag(LutOpcode::U29AddEq))
    + s_u24.clone().into() * AB::Expr::from_u16(lut_opcode_tag(LutOpcode::U24AddEq))
    + s_u15.clone().into() * AB::Expr::from_u16(lut_opcode_tag(LutOpcode::U15AddEq))
    + s_bit.clone().into() * AB::Expr::from_u16(lut_opcode_tag(LutOpcode::BitAddEq))
    + s_mul.clone().into() * AB::Expr::from_u16(lut_opcode_tag(LutOpcode::U15MulEq))
    + s_mul_low.clone().into() * AB::Expr::from_u16(lut_opcode_tag(LutOpcode::ByteMulLowEq))
    + s_mul_high.clone().into() * AB::Expr::from_u16(lut_opcode_tag(LutOpcode::ByteMulHighEq))
    + s_and.clone().into() * AB::Expr::from_u16(lut_opcode_tag(LutOpcode::ByteAndEq))
    + s_or.clone().into() * AB::Expr::from_u16(lut_opcode_tag(LutOpcode::ByteOrEq))
    + s_xor.clone().into() * AB::Expr::from_u16(lut_opcode_tag(LutOpcode::ByteXorEq))
    + s_axiom.clone().into() * AB::Expr::from_u16(LUT_AXIOM_TAG);
  builder.assert_eq(op.into(), op_reconstruct);

  // ── (4) Per-opcode arithmetic constraints ──
  let total = in0.clone().into() + in1.clone().into() + in2.clone().into();
  let c256 = AB::Expr::from_u16(256);
  let c2 = AB::Expr::from_u16(2);
  let c32768 = AB::Expr::from_u32(32768);
  let c16777216 = AB::Expr::from_u32(1u32 << 24);
  let c536870912 = AB::Expr::from_u32(1u32 << 29);

  builder.assert_zero(
    s_u29.clone()
      * (total.clone() - out0.clone().into() - c536870912.clone() * out1.clone().into()),
  );
  builder.assert_bool(s_u29.clone() * out1.clone().into());

  builder.assert_zero(
    s_u24.clone() * (total.clone() - out0.clone().into() - c16777216.clone() * out1.clone().into()),
  );
  builder.assert_bool(s_u24.clone() * out1.clone().into());

  builder.assert_zero(
    s_u15.clone() * (total.clone() - out0.clone().into() - c32768.clone() * out1.clone().into()),
  );
  builder.assert_bool(s_u15.clone() * out1.clone().into());

  builder.assert_zero(
    s_bit.clone() * (total.clone() - out0.clone().into() - c2.clone() * out1.clone().into()),
  );
  builder.assert_bool(s_bit.clone() * in0.clone().into());
  builder.assert_bool(s_bit.clone() * in1.clone().into());
  builder.assert_bool(s_bit.clone() * in2.clone().into());
  builder.assert_bool(s_bit.clone() * out0.clone().into());
  builder.assert_bool(s_bit.clone() * out1.clone().into());

  builder.assert_zero(
    s_mul.clone()
      * ((in0.clone().into() * in1.clone().into())
        - out0.clone().into()
        - c32768.clone() * out1.clone().into()),
  );
  builder.assert_zero(s_mul.clone() * in2.clone());

  builder.assert_zero(
    s_mul_low.clone()
      * ((in0.clone().into() * in1.clone().into())
        - out0.clone().into()
        - c256.clone() * out1.clone().into()),
  );
  builder.assert_zero(s_mul_low.clone() * in2.clone());

  builder.assert_zero(
    s_mul_high.clone()
      * ((in0.clone().into() * in1.clone().into())
        - out1.clone().into()
        - c256.clone() * out0.clone().into()),
  );
  builder.assert_zero(s_mul_high.clone() * in2.clone());

  builder.assert_zero(s_and.clone() * in2.clone());
  builder.assert_zero(s_and.clone() * out1.clone());
  builder.assert_zero(s_or.clone() * in2.clone());
  builder.assert_zero(s_or.clone() * out1.clone());
  builder.assert_zero(s_xor.clone() * in2.clone());
  builder.assert_zero(s_xor.clone() * out1.clone());
}

/// Build a LUT main trace with exactly **one row per [`ProofRow`]** — the same
/// height as [`build_proof_rows_preprocessed_matrix`] and
/// [`build_stack_ir_trace_from_rows`].
pub fn build_lut_trace_from_proof_rows(rows: &[ProofRow]) -> Result<RowMajorMatrix<Val>, String> {
  use crate::semantic_proof::{
    OP_AND_INTRO, OP_BOOL, OP_BYTE, OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE,
    OP_BYTE_ADD_THIRD_CONGRUENCE, OP_BYTE_AND_EQ, OP_BYTE_MUL_HIGH_EQ, OP_BYTE_MUL_LOW_EQ,
    OP_BYTE_OR_EQ, OP_BYTE_XOR_EQ, OP_EQ_REFL, OP_EQ_SYM, OP_EQ_TRANS, OP_NOT, OP_U15_MUL_EQ,
    OP_U24_ADD_EQ, OP_U29_ADD_EQ, OP_PUSH_AXIOM,
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
        trace.values[b + LUT_COL_OP] = Val::from_u16(lut_opcode_tag(LutOpcode::U15MulEq));
        trace.values[b + LUT_COL_IN0] = Val::from_u32(row.scalar0);
        trace.values[b + LUT_COL_IN1] = Val::from_u32(row.scalar1);
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
        trace.values[b + LUT_COL_OUT1] = Val::from_u32(result >> 8);
        trace.values[b + LUT_COL_SEL_BYTE_MUL_LOW_EQ] = Val::from_u16(1);
      }
      OP_BYTE_MUL_HIGH_EQ => {
        let result = row.scalar0 * row.scalar1;
        trace.values[b + LUT_COL_OP] = Val::from_u16(lut_opcode_tag(LutOpcode::ByteMulHighEq));
        trace.values[b + LUT_COL_IN0] = Val::from_u32(row.scalar0);
        trace.values[b + LUT_COL_IN1] = Val::from_u32(row.scalar1);
        trace.values[b + LUT_COL_OUT0] = Val::from_u32(result >> 8);
        trace.values[b + LUT_COL_OUT1] = Val::from_u32(result & 0xFF);
        trace.values[b + LUT_COL_SEL_BYTE_MUL_HIGH_EQ] = Val::from_u16(1);
      }
      op if op == OP_BYTE_AND_EQ => {
        trace.values[b + LUT_COL_OP] = Val::from_u16(lut_opcode_tag(LutOpcode::ByteAndEq));
        trace.values[b + LUT_COL_IN0] = Val::from_u32(row.scalar0);
        trace.values[b + LUT_COL_IN1] = Val::from_u32(row.scalar1);
        trace.values[b + LUT_COL_OUT0] = Val::from_u32(row.value);
        trace.values[b + LUT_COL_SEL_BYTE_AND_EQ] = Val::from_u16(1);
      }
      op if op == OP_BYTE_OR_EQ => {
        trace.values[b + LUT_COL_OP] = Val::from_u16(lut_opcode_tag(LutOpcode::ByteOrEq));
        trace.values[b + LUT_COL_IN0] = Val::from_u32(row.scalar0);
        trace.values[b + LUT_COL_IN1] = Val::from_u32(row.scalar1);
        trace.values[b + LUT_COL_OUT0] = Val::from_u32(row.value);
        trace.values[b + LUT_COL_SEL_BYTE_OR_EQ] = Val::from_u16(1);
      }
      op if op == OP_BYTE_XOR_EQ => {
        trace.values[b + LUT_COL_OP] = Val::from_u16(lut_opcode_tag(LutOpcode::ByteXorEq));
        trace.values[b + LUT_COL_IN0] = Val::from_u32(row.scalar0);
        trace.values[b + LUT_COL_IN1] = Val::from_u32(row.scalar1);
        trace.values[b + LUT_COL_OUT0] = Val::from_u32(row.value);
        trace.values[b + LUT_COL_SEL_BYTE_XOR_EQ] = Val::from_u16(1);
      }
      // Axiom rows (op >= OP_PUSH_AXIOM) connect to the AIR via the dedicated
      // LUT_COL_SEL_AXIOM selector.  Their correctness is enforced by the
      // consistency AIR at batch level; the LUT AIR just records their presence.
      other if other >= OP_PUSH_AXIOM => {
        trace.values[b + LUT_COL_OP] = Val::from_u16(LUT_AXIOM_TAG);
        trace.values[b + LUT_COL_IN0] = Val::from_u32(row.scalar0);
        trace.values[b + LUT_COL_SEL_AXIOM] = Val::from_u16(1);
      }
      other => {
        return Err(format!(
          "build_lut_trace_from_proof_rows: unsupported op {other} at row {i}"
        ));
      }
    }
  }

  for i in rows.len()..n_rows {
    let b = i * NUM_LUT_COLS;
    trace.values[b + LUT_COL_OP] = pad_tag;
    trace.values[b + LUT_COL_SEL_U29_ADD_EQ] = Val::from_u16(1);
  }

  Ok(trace)
}

/// Collect byte-table LogUp queries from a **LutStep** slice.
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
pub fn collect_byte_table_queries_from_rows(
  rows: &[ProofRow],
) -> Vec<crate::byte_table::ByteTableQuery> {
  use crate::byte_table::{BYTE_OP_AND, BYTE_OP_OR, BYTE_OP_XOR};
  use crate::semantic_proof::{
    OP_BYTE_AND_EQ, OP_BYTE_AND_SYM, OP_BYTE_NOT_SYM, OP_BYTE_OR_EQ, OP_BYTE_OR_SYM,
    OP_BYTE_XOR_EQ, OP_BYTE_XOR_SYM,
  };
  use std::collections::BTreeMap;

  let mut acc: BTreeMap<(u8, u8, u32), i32> = BTreeMap::new();

  for row in rows {
    let (a, b, op) = match row.op {
      op if op == OP_BYTE_AND_EQ => (row.scalar0 as u8, row.scalar1 as u8, BYTE_OP_AND),
      op if op == OP_BYTE_OR_EQ => (row.scalar0 as u8, row.scalar1 as u8, BYTE_OP_OR),
      op if op == OP_BYTE_XOR_EQ => (row.scalar0 as u8, row.scalar1 as u8, BYTE_OP_XOR),
      // Symbolic variants: concrete bytes in scalar1/scalar2.
      op if op == OP_BYTE_AND_SYM => (row.scalar1 as u8, row.scalar2 as u8, BYTE_OP_AND),
      op if op == OP_BYTE_OR_SYM => (row.scalar1 as u8, row.scalar2 as u8, BYTE_OP_OR),
      op if op == OP_BYTE_XOR_SYM => (row.scalar1 as u8, row.scalar2 as u8, BYTE_OP_XOR),
      // NOT(a) = a XOR 0xFF — uses XOR table.
      op if op == OP_BYTE_NOT_SYM => (row.scalar1 as u8, 0xFF, BYTE_OP_XOR),
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
        _ => a ^ b,
      };
      crate::byte_table::ByteTableQuery {
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
pub fn validate_manifest_rows(rows: &[ProofRow]) -> bool {
  use crate::semantic_proof::{
    OP_BYTE_AND_EQ, OP_BYTE_MUL_HIGH_EQ, OP_BYTE_MUL_LOW_EQ, OP_BYTE_OR_EQ, OP_BYTE_XOR_EQ,
    OP_U15_MUL_EQ, OP_U24_ADD_EQ, OP_U29_ADD_EQ,
  };

  for row in rows {
    if row.op == OP_BYTE_AND_EQ {
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
      const MAX29: u32 = (1u32 << 29) - 1;
      if row.scalar0 > MAX29 || row.scalar1 > MAX29 || row.value > MAX29 {
        return false;
      }
      if row.scalar2 > 1 || row.arg1 > 1 {
        return false;
      }
    } else if row.op == OP_U24_ADD_EQ {
      const MAX24: u32 = (1u32 << 24) - 1;
      if row.scalar0 > MAX24 || row.scalar1 > MAX24 || row.value > MAX24 {
        return false;
      }
      if row.scalar2 > 1 || row.arg1 > 1 {
        return false;
      }
    } else if row.op == OP_U15_MUL_EQ {
      const MAX15: u32 = 0x7FFF_u32;
      if row.scalar0 > MAX15 || row.scalar1 > MAX15 {
        return false;
      }
      if row.value > MAX15 || row.arg0 > MAX15 {
        return false;
      }
    } else if (row.op == OP_BYTE_MUL_LOW_EQ || row.op == OP_BYTE_MUL_HIGH_EQ)
      && (row.scalar0 > 255 || row.scalar1 > 255) {
        return false;
      }
  }
  true
}

/// Per-row prep scalar binding helper used by [`BatchLutKernelAirWithPrep`].
fn eval_lut_prep_row_binding_inner<AB>(builder: &mut AB, prep_row: &[AB::Var], local: &[AB::Var])
where
  AB: AirBuilderWithPublicValues + PairBuilder + AirBuilder<F = Val>,
{
  use crate::semantic_proof::{
    OP_ADD24, OP_ADD29, OP_ADD29_CARRY, OP_ADD_BYTE_SYM, OP_AND_INTRO, OP_BOOL, OP_BYTE,
    OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE, OP_BYTE_ADD_THIRD_CONGRUENCE, OP_BYTE_AND_EQ,
    OP_BYTE_AND_SYM, OP_BYTE_MUL_HIGH_EQ, OP_BYTE_MUL_LOW_EQ, OP_BYTE_NOT_SYM,
    OP_BYTE_OR_EQ, OP_BYTE_OR_SYM, OP_BYTE_XOR_EQ, OP_BYTE_XOR_SYM,
    OP_CALL_AXIOM, OP_CARRY_EQ, OP_CARRY_LIMB, OP_CARRY_LIMB_ZERO, OP_CARRY_TERM,
    OP_CREATE_AXIOM, OP_DUP_AXIOM,
    OP_ENV_AXIOM, OP_EQ_REFL, OP_EQ_SYM, OP_EQ_TRANS, OP_EXTERNAL_STATE_AXIOM,
    OP_INPUT_LIMB24, OP_INPUT_LIMB29, OP_INPUT_TERM, OP_KECCAK_AXIOM, OP_LOG_AXIOM,
    OP_MEM_COPY_AXIOM, OP_MLOAD_AXIOM, OP_MSTORE_AXIOM, OP_NOT, OP_OUTPUT_LIMB24,
    OP_OUTPUT_LIMB29, OP_OUTPUT_TERM, OP_PC_AFTER, OP_PC_BEFORE, OP_PUSH_AXIOM,
    OP_SELFDESTRUCT_AXIOM, OP_SLOAD_AXIOM, OP_SSTORE_AXIOM, OP_STRUCTURAL_AXIOM,
    OP_SUB_BYTE_SYM, OP_SWAP_AXIOM, OP_TERMINATE_AXIOM, OP_TRANSIENT_AXIOM, OP_U15_MUL_EQ,
    OP_U24_ADD_EQ, OP_U24_ADD_SYM, OP_U24_SUB_SYM, OP_U29_ADD_EQ, OP_U29_ADD_SYM,
    OP_U29_SUB_SYM,
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
    // Axiom leaf rows — included so their product-gate selectors evaluate to
    // zero, preventing spurious constraint enforcement on axiom rows.
    OP_PUSH_AXIOM, OP_DUP_AXIOM, OP_SWAP_AXIOM, OP_STRUCTURAL_AXIOM,
    OP_MLOAD_AXIOM, OP_MSTORE_AXIOM, OP_MEM_COPY_AXIOM,
    OP_SLOAD_AXIOM, OP_SSTORE_AXIOM, OP_TRANSIENT_AXIOM, OP_KECCAK_AXIOM,
    OP_ENV_AXIOM, OP_EXTERNAL_STATE_AXIOM, OP_TERMINATE_AXIOM,
    OP_CALL_AXIOM, OP_CREATE_AXIOM, OP_SELFDESTRUCT_AXIOM, OP_LOG_AXIOM,
    OP_INPUT_TERM, OP_OUTPUT_TERM, OP_PC_BEFORE, OP_PC_AFTER,
    // 29/24-bit limb symbolic terms — product-gate selectors must evaluate to zero.
    OP_INPUT_LIMB29, OP_OUTPUT_LIMB29, OP_INPUT_LIMB24, OP_OUTPUT_LIMB24,
    // Limb arithmetic terms (syntactic; LogUp provides values).
    OP_CARRY_LIMB, OP_ADD29, OP_ADD29_CARRY, OP_ADD24,
    // Syntactic limb-level proof variants.
    OP_CARRY_LIMB_ZERO, OP_U29_ADD_SYM, OP_U24_ADD_SYM, OP_U29_SUB_SYM, OP_U24_SUB_SYM,
    // Symbolic byte-level op leaves (AND/OR/XOR/NOT/ADD/SUB sym proofs).
    OP_BYTE_AND_SYM, OP_BYTE_OR_SYM, OP_BYTE_XOR_SYM, OP_BYTE_NOT_SYM,
    OP_ADD_BYTE_SYM, OP_SUB_BYTE_SYM, OP_CARRY_TERM, OP_CARRY_EQ,
  ];

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

  let tag_ml = lut_opcode_tag(LutOpcode::ByteMulLowEq) as u32;
  let g = pg(OP_BYTE_MUL_LOW_EQ);
  builder.assert_zero(g.clone() * (local[LUT_COL_OP].clone().into() - AB::Expr::from_u32(tag_ml)));
  builder.assert_zero(
    g.clone() * (local[LUT_COL_IN0].clone().into() - prep_row[PREP_COL_SCALAR0].clone().into()),
  );
  builder.assert_zero(
    g * (local[LUT_COL_IN1].clone().into() - prep_row[PREP_COL_SCALAR1].clone().into()),
  );

  let tag_mh = lut_opcode_tag(LutOpcode::ByteMulHighEq) as u32;
  let g = pg(OP_BYTE_MUL_HIGH_EQ);
  builder.assert_zero(g.clone() * (local[LUT_COL_OP].clone().into() - AB::Expr::from_u32(tag_mh)));
  builder.assert_zero(
    g.clone() * (local[LUT_COL_IN0].clone().into() - prep_row[PREP_COL_SCALAR0].clone().into()),
  );
  builder.assert_zero(
    g * (local[LUT_COL_IN1].clone().into() - prep_row[PREP_COL_SCALAR1].clone().into()),
  );

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
pub struct BatchLutKernelAirWithPrep {
  prep_matrix: Option<RowMajorMatrix<Val>>,
}

impl BatchLutKernelAirWithPrep {
  pub fn new(manifest: &BatchProofRowsManifest, pv: &[Val]) -> Self {
    Self {
      prep_matrix: cfg!(debug_assertions)
        .then(|| build_batch_proof_rows_preprocessed_matrix(manifest, pv)),
    }
  }

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
    // ── 1. Tag check ──
    {
      let pis = builder.public_values();
      builder.assert_eq(pis[0], AB::Expr::from_u32(RECEIPT_BIND_TAG_LUT));
    }

    // ── 2. Batch pv binding at row 0 ──
    {
      let pi_n: AB::Expr = {
        let pis = builder.public_values();
        pis[1].into()
      };
      let pi_digest: [AB::Expr; 8] = {
        let pis = builder.public_values();
        std::array::from_fn(|k| pis[2 + k].into())
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

    // ── 3. Per-row prep scalar binding ──
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

    // ── 4. LUT arithmetic constraints ──
    eval_lut_kernel_inner(builder);
  }
}

/// Prove the batch LUT STARK for `manifest`.
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
pub fn build_lut_steps_from_rows_bit_family(rows: &[ProofRow]) -> Result<Vec<LutStep>, String> {
  use crate::semantic_proof::{
    OP_AND_INTRO, OP_BYTE_AND_SYM, OP_BYTE_NOT_SYM, OP_BYTE_OR_SYM, OP_BYTE_XOR_SYM,
    OP_CARRY_EQ, OP_CARRY_TERM, OP_EQ_REFL, OP_EQ_SYM, OP_EQ_TRANS, OP_INPUT_TERM,
    OP_NOT, OP_OUTPUT_TERM, OP_PC_AFTER, OP_PC_BEFORE,
  };
  let mut out = Vec::new();
  let mut seen_bitwise = false;

  for row in rows {
    match row.op {
      op if op == OP_AND_INTRO
        || op == OP_EQ_REFL
        || op == OP_EQ_SYM
        || op == OP_EQ_TRANS
        || op == OP_NOT
        // Symbolic leaf rows — produce LUT entries for the underlying byte ops.
        || op == OP_CARRY_TERM
        || op == OP_CARRY_EQ
        || op == OP_INPUT_TERM
        || op == OP_OUTPUT_TERM
        || op == OP_PC_BEFORE
        || op == OP_PC_AFTER => {}
      op if op == OP_BYTE_AND_SYM => {
        seen_bitwise = true;
        let (a, b) = (row.scalar1, row.scalar2);
        out.push(LutStep {
          op: LutOpcode::ByteAndEq,
          in0: a,
          in1: b,
          in2: 0,
          out0: a & b,
          out1: 0,
        });
      }
      op if op == OP_BYTE_OR_SYM => {
        seen_bitwise = true;
        let (a, b) = (row.scalar1, row.scalar2);
        out.push(LutStep {
          op: LutOpcode::ByteOrEq,
          in0: a,
          in1: b,
          in2: 0,
          out0: a | b,
          out1: 0,
        });
      }
      op if op == OP_BYTE_XOR_SYM => {
        seen_bitwise = true;
        let (a, b) = (row.scalar1, row.scalar2);
        out.push(LutStep {
          op: LutOpcode::ByteXorEq,
          in0: a,
          in1: b,
          in2: 0,
          out0: a ^ b,
          out1: 0,
        });
      }
      op if op == OP_BYTE_NOT_SYM => {
        seen_bitwise = true;
        let a = row.scalar1;
        out.push(LutStep {
          op: LutOpcode::ByteXorEq,
          in0: a,
          in1: 0xFF,
          in2: 0,
          out0: a ^ 0xFF,
          out1: 0,
        });
      }
      crate::semantic_proof::OP_BYTE_AND_EQ => {
        seen_bitwise = true;
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
        seen_bitwise = true;
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
        seen_bitwise = true;
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
      other => return Err(format!("non-bitwise row in bitwise kernel path: {other}")),
    }
  }

  if !seen_bitwise {
    return Err("no AND/OR/XOR LUT rows in bitwise row set".to_string());
  }
  Ok(out)
}

// ============================================================
// Cross-family batch WFF proving
// ============================================================

/// Build LUT steps from ProofRows belonging to **any** opcode family.
pub fn build_lut_steps_from_rows_auto(rows: &[ProofRow]) -> Result<Vec<LutStep>, String> {
  use crate::semantic_proof::{
    OP_AND_INTRO, OP_BOOL, OP_BYTE, OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE,
    OP_BYTE_ADD_THIRD_CONGRUENCE, OP_EQ_REFL, OP_EQ_SYM, OP_EQ_TRANS, OP_NOT,
  };

  let mut out = Vec::with_capacity(rows.len() * 2);

  for row in rows {
    match row.op {
      op if op == OP_BOOL
        || op == OP_BYTE
        || op == OP_EQ_REFL
        || op == OP_AND_INTRO
        || op == OP_EQ_TRANS
        || op == OP_BYTE_ADD_THIRD_CONGRUENCE
        || op == OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE
        || op == OP_NOT
        || op == OP_EQ_SYM => {}

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
      crate::semantic_proof::OP_BYTE_MUL_LOW_EQ => {
        let product = row.scalar0 * row.scalar1;
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
        out.push(LutStep {
          op: LutOpcode::ByteMulHighEq,
          in0: row.scalar0,
          in1: row.scalar1,
          in2: 0,
          out0: product >> 8,
          out1: product & 0xFF,
        });
      }
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
pub fn prove_batch_wff_proofs(
  proofs: &[&Proof],
) -> Result<(CircleStarkProof, Vec<crate::semantic_proof::WFF>), String> {
  if proofs.is_empty() {
    return Err("prove_batch_wff_proofs: empty proof batch".to_string());
  }
  let all_rows: Vec<ProofRow> = proofs.iter().flat_map(|p| compile_proof(p)).collect();
  let steps = build_lut_steps_from_rows_auto(&all_rows)?;
  let lut_proof = prove_lut_kernel_stark(&steps)?;
  let wffs = proofs
    .iter()
    .map(|p| infer_proof(p).map_err(|e| format!("infer_proof failed: {e}")))
    .collect::<Result<Vec<_>, _>>()?;
  Ok((lut_proof, wffs))
}

/// Verify a batch WFF proof.
pub fn verify_batch_wff_proofs(
  lut_proof: &CircleStarkProof,
  proofs: &[&Proof],
  wffs: &[crate::semantic_proof::WFF],
) -> Result<(), String> {
  if proofs.len() != wffs.len() {
    return Err(format!(
      "verify_batch_wff_proofs: proofs.len()={} != wffs.len()={}",
      proofs.len(),
      wffs.len()
    ));
  }
  verify_lut_kernel_stark(lut_proof)
    .map_err(|e| format!("batch LUT STARK verification failed: {e:?}"))?;
  for (i, (proof, expected_wff)) in proofs.iter().zip(wffs.iter()).enumerate() {
    let inferred =
      infer_proof(proof).map_err(|e| format!("infer_proof failed for instruction {i}: {e}"))?;
    if inferred != *expected_wff {
      return Err(format!("WFF mismatch at instruction {i}"));
    }
  }
  let all_rows: Vec<ProofRow> = proofs.iter().flat_map(|p| compile_proof(p)).collect();
  if !validate_manifest_rows(&all_rows) {
    return Err(
      "manifest row validation failed (Gap 3/4): byte op value or range check".to_string(),
    );
  }
  Ok(())
}
