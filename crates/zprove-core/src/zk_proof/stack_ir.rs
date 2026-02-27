//! StackIR (register-machine IR) trace builder, AIR, and prove/verify functions.

use super::preprocessed::{
  PREP_COL_ARG0, PREP_COL_ARG1, PREP_COL_ARG2, PREP_COL_EVM_OPCODE, PREP_COL_OP, PREP_COL_RET_TY,
  PREP_COL_SCALAR0, PREP_COL_SCALAR1, PREP_COL_SCALAR2, PREP_COL_VALUE, PREP_COL_WFF_DIGEST_START,
  build_proof_rows_preprocessed_matrix,
};
use super::types::{
  CircleStarkConfig, CircleStarkProof, CircleStarkVerifyResult, RECEIPT_BIND_TAG_STACK, Val,
  default_receipt_bind_public_values_for_tag, make_circle_config,
};
use crate::semantic_proof::{OP_EQ_REFL, ProofRow, RET_WFF_EQ};

use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, PairBuilder};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{PreprocessedProverData, PreprocessedVerifierKey};

// ============================================================
// Stack IR framework (offset-friendly scaffold)
// ============================================================

/// Register-machine IR step.  Each row stores explicit operand row indices
/// (`arg0/arg1/arg2`) instead of implicit stack-pointer deltas, enabling
/// Common Sub-expression Elimination (CSE) in `compile_proof_inner`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StackIrStep {
  pub op: u32,
  pub arg0: u32,
  pub arg1: u32,
  pub arg2: u32,
  pub scalar0: u32,
  pub scalar1: u32,
  pub scalar2: u32,
  pub value: u32,
  pub ret_ty: u32,
}

const STACK_COL_OP: usize = 0;
const STACK_COL_ARG0: usize = 1;
const STACK_COL_ARG1: usize = 2;
const STACK_COL_ARG2: usize = 3;
const STACK_COL_SCALAR0: usize = 4;
const STACK_COL_SCALAR1: usize = 5;
const STACK_COL_SCALAR2: usize = 6;
const STACK_COL_VALUE: usize = 7;
const STACK_COL_RET_TY: usize = 8;
// Columns reduced from 10 to 9: pop/push/sp_before/sp_after removed,
// replaced by arg0/arg1/arg2.  Semantics now enforced via preprocessed binding.
pub const NUM_STACK_IR_COLS: usize = 9;

pub fn build_stack_ir_steps_from_rows(rows: &[ProofRow]) -> Result<Vec<StackIrStep>, String> {
  if rows.is_empty() {
    return Err("cannot build reg-ir steps from empty rows".to_string());
  }
  Ok(
    rows
      .iter()
      .map(|row| StackIrStep {
        op: row.op,
        arg0: row.arg0,
        arg1: row.arg1,
        arg2: row.arg2,
        scalar0: row.scalar0,
        scalar1: row.scalar1,
        scalar2: row.scalar2,
        value: row.value,
        ret_ty: row.ret_ty,
      })
      .collect(),
  )
}

pub fn build_stack_ir_trace_from_steps(
  steps: &[StackIrStep],
) -> Result<RowMajorMatrix<Val>, String> {
  if steps.is_empty() {
    return Err("cannot build reg-ir trace from empty steps".to_string());
  }

  let ensure_m31 = |name: &str, value: u32, row: usize| -> Result<(), String> {
    if value >= 0x7fff_ffff {
      return Err(format!(
        "reg-ir {name} out of M31 range at row {row}: {value}"
      ));
    }
    Ok(())
  };

  for (i, step) in steps.iter().enumerate() {
    ensure_m31("op", step.op, i)?;
    ensure_m31("arg0", step.arg0, i)?;
    ensure_m31("arg1", step.arg1, i)?;
    ensure_m31("arg2", step.arg2, i)?;
    ensure_m31("scalar0", step.scalar0, i)?;
    ensure_m31("scalar1", step.scalar1, i)?;
    ensure_m31("scalar2", step.scalar2, i)?;
    ensure_m31("value", step.value, i)?;
    ensure_m31("ret_ty", step.ret_ty, i)?;
  }

  let n_rows = steps.len().max(4).next_power_of_two();
  let mut trace = RowMajorMatrix::new(Val::zero_vec(n_rows * NUM_STACK_IR_COLS), NUM_STACK_IR_COLS);

  for (i, step) in steps.iter().enumerate() {
    let base = i * NUM_STACK_IR_COLS;
    trace.values[base + STACK_COL_OP] = Val::from_u32(step.op);
    trace.values[base + STACK_COL_ARG0] = Val::from_u32(step.arg0);
    trace.values[base + STACK_COL_ARG1] = Val::from_u32(step.arg1);
    trace.values[base + STACK_COL_ARG2] = Val::from_u32(step.arg2);
    trace.values[base + STACK_COL_SCALAR0] = Val::from_u32(step.scalar0);
    trace.values[base + STACK_COL_SCALAR1] = Val::from_u32(step.scalar1);
    trace.values[base + STACK_COL_SCALAR2] = Val::from_u32(step.scalar2);
    trace.values[base + STACK_COL_VALUE] = Val::from_u32(step.value);
    trace.values[base + STACK_COL_RET_TY] = Val::from_u32(step.ret_ty);
  }

  // Padding rows: op = OP_EQ_REFL, ret_ty = RET_WFF_EQ, all arg/scalar/value = 0.
  if steps.len() < n_rows {
    for i in steps.len()..n_rows {
      let base = i * NUM_STACK_IR_COLS;
      trace.values[base + STACK_COL_OP] = Val::from_u32(OP_EQ_REFL);
      trace.values[base + STACK_COL_RET_TY] = Val::from_u32(RET_WFF_EQ);
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

impl<AB: AirBuilderWithPublicValues> Air<AB> for StackIrAir {
  fn eval(&self, builder: &mut AB) {
    eval_stack_ir_inner(builder);
  }
}

/// Core RegIR constraints — shared between `StackIrAir` and
/// `StackIrAirWithPrep` to avoid code duplication.
fn eval_stack_ir_inner<AB: AirBuilderWithPublicValues>(builder: &mut AB) {
  let pis = builder.public_values();
  builder.assert_eq(pis[0], AB::Expr::from_u32(RECEIPT_BIND_TAG_STACK));
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
  pub fn new(rows: &[ProofRow], pv: &[Val]) -> Self {
    Self {
      prep_matrix: cfg!(debug_assertions).then(|| build_proof_rows_preprocessed_matrix(rows, pv)),
    }
  }

  /// Build a `StackIrAirWithPrep` for use in the **verifier** path.
  pub fn for_verify() -> Self {
    Self { prep_matrix: None }
  }
}

impl BaseAir<Val> for StackIrAirWithPrep {
  fn width(&self) -> usize {
    NUM_STACK_IR_COLS
  }

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
        local[STACK_COL_ARG0].clone(),
        prep_row[PREP_COL_ARG0].clone(),
      );
      builder.assert_eq(
        local[STACK_COL_ARG1].clone(),
        prep_row[PREP_COL_ARG1].clone(),
      );
      builder.assert_eq(
        local[STACK_COL_ARG2].clone(),
        prep_row[PREP_COL_ARG2].clone(),
      );
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

    // Step 1b — bind pis[1] (EVM opcode) and pis[2..10] (tag-independent WFF digest).
    {
      let pi_opcode: AB::Expr = {
        let pis = builder.public_values();
        pis[1].into()
      };
      let pi_digest: [AB::Expr; 8] = {
        let pis = builder.public_values();
        std::array::from_fn(|k| pis[2 + k].into())
      };
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
      builder.when_first_row().assert_eq(prep_opcode, pi_opcode);
      for k in 0..8_usize {
        builder
          .when_first_row()
          .assert_eq(prep_digest[k].clone(), pi_digest[k].clone());
      }
    }

    // Step 2 — shared tag check.
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

/// Prove the StackIR STARK for `rows` and bind the proof to the shared
/// preprocessed commitment (`prep_data`).
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
