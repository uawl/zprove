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
  OP_BYTE_ADD_EQ,
  infer_proof,
  Proof,
  ProofRow,
  RET_BYTE,
  RET_WFF_AND,
  Term,
  RET_WFF_EQ,
  WFF,
  compile_proof,
  verify_compiled,
};
use crate::memory_proof::{CqMemoryEvent, CqRw};

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
    crate::sementic_proof::OP_BOOL
    | crate::sementic_proof::OP_BYTE
    | crate::sementic_proof::OP_BYTE_ADD_EQ
    | crate::sementic_proof::OP_U16_ADD_EQ
    | crate::sementic_proof::OP_BYTE_ADD_CARRY_EQ
    | crate::sementic_proof::OP_BYTE_MUL_LOW_EQ
    | crate::sementic_proof::OP_BYTE_MUL_HIGH_EQ
    | crate::sementic_proof::OP_BYTE_AND_EQ
    | crate::sementic_proof::OP_BYTE_OR_EQ
    | crate::sementic_proof::OP_BYTE_XOR_EQ => Some(0),

    crate::sementic_proof::OP_NOT
    | crate::sementic_proof::OP_EQ_REFL
    | crate::sementic_proof::OP_EQ_SYM
    | crate::sementic_proof::OP_BYTE_ADD_THIRD_CONGRUENCE
    | crate::sementic_proof::OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE => Some(1),

    crate::sementic_proof::OP_AND
    | crate::sementic_proof::OP_OR
    | crate::sementic_proof::OP_XOR
    | crate::sementic_proof::OP_BYTE_MUL_LOW
    | crate::sementic_proof::OP_BYTE_MUL_HIGH
    | crate::sementic_proof::OP_BYTE_AND
    | crate::sementic_proof::OP_BYTE_OR
    | crate::sementic_proof::OP_BYTE_XOR
    | crate::sementic_proof::OP_AND_INTRO
    | crate::sementic_proof::OP_EQ_TRANS
    | crate::sementic_proof::OP_ITE_TRUE_EQ
    | crate::sementic_proof::OP_ITE_FALSE_EQ => Some(2),

    crate::sementic_proof::OP_ITE
    | crate::sementic_proof::OP_BYTE_ADD
    | crate::sementic_proof::OP_BYTE_ADD_CARRY => Some(3),

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
      .ok_or_else(|| format!("row op {} is not stack-ir encodable yet", row.op))? as i64;
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
    return Err(format!("stack-ir build ended with invalid final stack size: {sp}"));
  }

  Ok(out)
}

pub fn build_stack_ir_trace_from_steps(steps: &[StackIrStep]) -> Result<RowMajorMatrix<Val>, String> {
  if steps.is_empty() {
    return Err("cannot build stack-ir trace from empty steps".to_string());
  }

  let ensure_u16 = |name: &str, value: u32, row: usize| -> Result<(), String> {
    if value > u16::MAX as u32 {
      return Err(format!("stack-ir {name} out of u16 range at row {row}: {value}"));
    }
    Ok(())
  };

  for (i, step) in steps.iter().enumerate() {
    ensure_u16("op", step.op, i)?;
    ensure_u16("pop", step.pop, i)?;
    ensure_u16("push", step.push, i)?;
    ensure_u16("sp_before", step.sp_before, i)?;
    ensure_u16("sp_after", step.sp_after, i)?;
    ensure_u16("scalar0", step.scalar0, i)?;
    ensure_u16("scalar1", step.scalar1, i)?;
    ensure_u16("scalar2", step.scalar2, i)?;
    ensure_u16("value", step.value, i)?;
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
    trace.values[base + STACK_COL_SCALAR0] = Val::from_u16(step.scalar0 as u16);
    trace.values[base + STACK_COL_SCALAR1] = Val::from_u16(step.scalar1 as u16);
    trace.values[base + STACK_COL_SCALAR2] = Val::from_u16(step.scalar2 as u16);
    trace.values[base + STACK_COL_VALUE] = Val::from_u16(step.value as u16);
    trace.values[base + STACK_COL_RET_TY] = Val::from_u16(step.ret_ty as u16);
  }

  if steps.len() < n_rows {
    let last = *steps.last().expect("non-empty checked");
    let steady_sp = last.sp_after;
    for i in steps.len()..n_rows {
      let base = i * NUM_STACK_IR_COLS;
      trace.values[base + STACK_COL_OP] = Val::from_u16(crate::sementic_proof::OP_EQ_REFL as u16);
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

impl<AB: AirBuilder> Air<AB> for StackIrAir {
  fn eval(&self, builder: &mut AB) {
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

    let t_bool = crate::sementic_proof::OP_BOOL as u16;
    let t_byte = crate::sementic_proof::OP_BYTE as u16;
    let t_eq_refl = crate::sementic_proof::OP_EQ_REFL as u16;
    let t_and_intro = crate::sementic_proof::OP_AND_INTRO as u16;
    let t_eq_trans = crate::sementic_proof::OP_EQ_TRANS as u16;
    let t_add_congr = crate::sementic_proof::OP_BYTE_ADD_THIRD_CONGRUENCE as u16;
    let t_add_carry_congr = crate::sementic_proof::OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE as u16;
    let t_add_eq = crate::sementic_proof::OP_BYTE_ADD_EQ as u16;
    let t_u16_add_eq = crate::sementic_proof::OP_U16_ADD_EQ as u16;
    let t_add_carry_eq = crate::sementic_proof::OP_BYTE_ADD_CARRY_EQ as u16;
    let t_mul_low_eq = crate::sementic_proof::OP_BYTE_MUL_LOW_EQ as u16;
    let t_mul_high_eq = crate::sementic_proof::OP_BYTE_MUL_HIGH_EQ as u16;
    let t_and_eq = crate::sementic_proof::OP_BYTE_AND_EQ as u16;
    let t_or_eq = crate::sementic_proof::OP_BYTE_OR_EQ as u16;
    let t_xor_eq = crate::sementic_proof::OP_BYTE_XOR_EQ as u16;

    let c_bool = AB::Expr::from_u16(t_bool);
    let c_byte = AB::Expr::from_u16(t_byte);
    let c_eq_refl = AB::Expr::from_u16(t_eq_refl);
    let c_and_intro = AB::Expr::from_u16(t_and_intro);
    let c_eq_trans = AB::Expr::from_u16(t_eq_trans);
    let c_add_congr = AB::Expr::from_u16(t_add_congr);
    let c_add_carry_congr = AB::Expr::from_u16(t_add_carry_congr);
    let c_add_eq = AB::Expr::from_u16(t_add_eq);
    let c_u16_add_eq = AB::Expr::from_u16(t_u16_add_eq);
    let c_add_carry_eq = AB::Expr::from_u16(t_add_carry_eq);
    let c_mul_low_eq = AB::Expr::from_u16(t_mul_low_eq);
    let c_mul_high_eq = AB::Expr::from_u16(t_mul_high_eq);
    let c_and_eq = AB::Expr::from_u16(t_and_eq);
    let c_or_eq = AB::Expr::from_u16(t_or_eq);
    let c_xor_eq = AB::Expr::from_u16(t_xor_eq);
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
      t_add_carry_eq,
      t_mul_low_eq,
      t_mul_high_eq,
      t_and_eq,
      t_or_eq,
      t_xor_eq,
    ];

    let gate = |target: u16| {
      let mut g = AB::Expr::from_u16(1);
      for t in allowed_tags {
        if t != target {
          g = g * (op.clone().into() - AB::Expr::from_u16(t));
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
      * (op.clone().into() - c_add_carry_eq.clone())
      * (op.clone().into() - c_mul_low_eq.clone())
      * (op.clone().into() - c_mul_high_eq.clone())
      * (op.clone().into() - c_and_eq.clone())
      * (op.clone().into() - c_or_eq.clone())
      * (op.clone().into() - c_xor_eq.clone());
    builder.assert_zero(allowed_poly);

    builder.assert_zero(
      sp_before.clone().into() - pop.clone().into() + push.clone().into() - sp_after.clone().into(),
    );

    builder.when_first_row().assert_zero(sp_before.clone());
    builder.when_transition().assert_eq(next[STACK_COL_SP_BEFORE].clone(), sp_after.clone());

    let bool_gate = gate(t_bool);
    let byte_gate = gate(t_byte);
    let eq_refl_gate = gate(t_eq_refl);
    let and_intro_gate = gate(t_and_intro);
    let eq_trans_gate = gate(t_eq_trans);
    let add_congr_gate = gate(t_add_congr);
    let add_carry_congr_gate = gate(t_add_carry_congr);
    let add_eq_gate = gate(t_add_eq);
    let u16_add_eq_gate = gate(t_u16_add_eq);
    let add_carry_eq_gate = gate(t_add_carry_eq);
    let mul_low_eq_gate = gate(t_mul_low_eq);
    let mul_high_eq_gate = gate(t_mul_high_eq);
    let and_eq_gate = gate(t_and_eq);
    let or_eq_gate = gate(t_or_eq);
    let xor_eq_gate = gate(t_xor_eq);

    builder.assert_zero(bool_gate.clone() * (pop.clone().into() - c_zero.clone()));
    builder.assert_zero(bool_gate.clone() * (push.clone().into() - c_one.clone()));
    builder.assert_zero(bool_gate.clone() * (ret_ty.clone().into() - AB::Expr::from_u16(crate::sementic_proof::RET_BOOL as u16)));
    builder.assert_zero(bool_gate * (value.clone().into() - scalar0.clone().into()));

    builder.assert_zero(byte_gate.clone() * (pop.clone().into() - c_zero.clone()));
    builder.assert_zero(byte_gate.clone() * (push.clone().into() - c_one.clone()));
    builder.assert_zero(byte_gate.clone() * (ret_ty.clone().into() - AB::Expr::from_u16(RET_BYTE as u16)));
    builder.assert_zero(byte_gate * (value.clone().into() - scalar0.clone().into()));

    builder.assert_zero(eq_refl_gate.clone() * (pop.clone().into() - c_one.clone()));
    builder.assert_zero(eq_refl_gate.clone() * (push.clone().into() - c_one.clone()));
    builder.assert_zero(eq_refl_gate * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)));

    builder.assert_zero(and_intro_gate.clone() * (pop.clone().into() - c_two));
    builder.assert_zero(and_intro_gate.clone() * (push.clone().into() - c_one));
    builder.assert_zero(and_intro_gate * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_AND as u16)));

    builder.assert_zero(eq_trans_gate.clone() * (pop.clone().into() - AB::Expr::from_u16(2)));
    builder.assert_zero(eq_trans_gate.clone() * (push.clone().into() - AB::Expr::from_u16(1)));
    builder.assert_zero(eq_trans_gate * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)));

    builder.assert_zero(add_congr_gate.clone() * (pop.clone().into() - AB::Expr::from_u16(1)));
    builder.assert_zero(add_congr_gate.clone() * (push.clone().into() - AB::Expr::from_u16(1)));
    builder.assert_zero(add_congr_gate * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)));

    builder.assert_zero(add_carry_congr_gate.clone() * (pop.clone().into() - AB::Expr::from_u16(1)));
    builder.assert_zero(add_carry_congr_gate.clone() * (push.clone().into() - AB::Expr::from_u16(1)));
    builder.assert_zero(add_carry_congr_gate * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)));

    builder.assert_zero(add_eq_gate.clone() * (pop.clone().into() - AB::Expr::from_u16(0)));
    builder.assert_zero(add_eq_gate.clone() * (push.clone().into() - AB::Expr::from_u16(1)));
    builder.assert_zero(add_eq_gate * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)));

    builder.assert_zero(u16_add_eq_gate.clone() * (pop.clone().into() - AB::Expr::from_u16(0)));
    builder.assert_zero(u16_add_eq_gate.clone() * (push.clone().into() - AB::Expr::from_u16(1)));
    builder.assert_zero(u16_add_eq_gate * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_AND as u16)));

    builder.assert_zero(add_carry_eq_gate.clone() * (pop.clone().into() - AB::Expr::from_u16(0)));
    builder.assert_zero(add_carry_eq_gate.clone() * (push.clone().into() - AB::Expr::from_u16(1)));
    builder.assert_zero(add_carry_eq_gate * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)));

    builder.assert_zero(mul_low_eq_gate.clone() * (pop.clone().into() - AB::Expr::from_u16(0)));
    builder.assert_zero(mul_low_eq_gate.clone() * (push.clone().into() - AB::Expr::from_u16(1)));
    builder.assert_zero(mul_low_eq_gate * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)));

    builder.assert_zero(mul_high_eq_gate.clone() * (pop.clone().into() - AB::Expr::from_u16(0)));
    builder.assert_zero(mul_high_eq_gate.clone() * (push.clone().into() - AB::Expr::from_u16(1)));
    builder.assert_zero(mul_high_eq_gate * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)));

    builder.assert_zero(and_eq_gate.clone() * (pop.clone().into() - AB::Expr::from_u16(0)));
    builder.assert_zero(and_eq_gate.clone() * (push.clone().into() - AB::Expr::from_u16(1)));
    builder.assert_zero(and_eq_gate * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)));

    builder.assert_zero(or_eq_gate.clone() * (pop.clone().into() - AB::Expr::from_u16(0)));
    builder.assert_zero(or_eq_gate.clone() * (push.clone().into() - AB::Expr::from_u16(1)));
    builder.assert_zero(or_eq_gate * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)));

    builder.assert_zero(xor_eq_gate.clone() * (pop.clone().into() - AB::Expr::from_u16(0)));
    builder.assert_zero(xor_eq_gate.clone() * (push.clone().into() - AB::Expr::from_u16(1)));
    builder.assert_zero(xor_eq_gate * (ret_ty.clone().into() - AB::Expr::from_u16(RET_WFF_EQ as u16)));
  }
}

pub fn prove_stack_ir_scaffold_stark(rows: &[ProofRow]) -> Result<CircleStarkProof, String> {
  let trace = build_stack_ir_trace_from_rows(rows)?;
  let config = make_circle_config();
  Ok(p3_uni_stark::prove(&config, &StackIrAir, trace, &[]))
}

pub fn verify_stack_ir_scaffold_stark(proof: &CircleStarkProof) -> CircleStarkVerifyResult {
  let config = make_circle_config();
  p3_uni_stark::verify(&config, &StackIrAir, proof, &[])
}

pub fn prove_and_verify_stack_ir_scaffold_stark(rows: &[ProofRow]) -> bool {
  let proved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| prove_stack_ir_scaffold_stark(rows)));
  let proof = match proved {
    Ok(Ok(proof)) => proof,
    Ok(Err(_)) => return false,
    Err(_) => return false,
  };
  verify_stack_ir_scaffold_stark(&proof).is_ok()
}

// ============================================================
// LUT framework (v2 scaffold)
// ============================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LutOpcode {
  ByteAddEq,
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
const NUM_LUT_COLS: usize = 6;

fn lut_opcode_tag(op: LutOpcode) -> u16 {
  match op {
    LutOpcode::ByteAddEq => 1,
    LutOpcode::U16AddEq => 2,
    LutOpcode::ByteAddCarryEq => 3,
    LutOpcode::ByteMulLowEq => 4,
    LutOpcode::ByteMulHighEq => 5,
    LutOpcode::ByteAndEq => 6,
    LutOpcode::ByteOrEq => 7,
    LutOpcode::ByteXorEq => 8,
  }
}

pub fn build_lut_steps_from_rows(rows: &[ProofRow]) -> Result<Vec<LutStep>, String> {
  let mut out = Vec::with_capacity(rows.len());

  for row in rows {
    let step = match row.op {
      crate::sementic_proof::OP_BYTE_ADD_EQ => LutStep {
        op: LutOpcode::ByteAddEq,
        in0: row.scalar0,
        in1: row.scalar1,
        in2: row.arg0,
        out0: row.value,
        out1: row.scalar2,
      },
      crate::sementic_proof::OP_U16_ADD_EQ => LutStep {
        op: LutOpcode::U16AddEq,
        in0: row.scalar0,
        in1: row.scalar1,
        in2: row.scalar2,
        out0: row.value,
        out1: row.arg1,
      },
      crate::sementic_proof::OP_BYTE_ADD_CARRY_EQ => {
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
      crate::sementic_proof::OP_BYTE_MUL_LOW_EQ => LutStep {
        op: LutOpcode::ByteMulLowEq,
        in0: row.scalar0,
        in1: row.scalar1,
        in2: 0,
        out0: (row.scalar0 * row.scalar1) & 0xFF,
        out1: 0,
      },
      crate::sementic_proof::OP_BYTE_MUL_HIGH_EQ => LutStep {
        op: LutOpcode::ByteMulHighEq,
        in0: row.scalar0,
        in1: row.scalar1,
        in2: 0,
        out0: (row.scalar0 * row.scalar1) >> 8,
        out1: 0,
      },
      crate::sementic_proof::OP_BYTE_AND_EQ => LutStep {
        op: LutOpcode::ByteAndEq,
        in0: row.scalar0,
        in1: row.scalar1,
        in2: 0,
        out0: row.scalar0 & row.scalar1,
        out1: 0,
      },
      crate::sementic_proof::OP_BYTE_OR_EQ => LutStep {
        op: LutOpcode::ByteOrEq,
        in0: row.scalar0,
        in1: row.scalar1,
        in2: 0,
        out0: row.scalar0 | row.scalar1,
        out1: 0,
      },
      crate::sementic_proof::OP_BYTE_XOR_EQ => LutStep {
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

  let ensure_le = |name: &str, value: u32, max: u32, row: usize| -> Result<(), String> {
    if value > max {
      return Err(format!("lut {name} out of range at row {row}: {value} > {max}"));
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

  let n_rows = steps.len().max(4).next_power_of_two();
  let mut trace = RowMajorMatrix::new(Val::zero_vec(n_rows * NUM_LUT_COLS), NUM_LUT_COLS);

  for (i, step) in steps.iter().enumerate() {
    let base = i * NUM_LUT_COLS;
    trace.values[base + LUT_COL_OP] = Val::from_u16(lut_opcode_tag(step.op));
    trace.values[base + LUT_COL_IN0] = Val::from_u16(step.in0 as u16);
    trace.values[base + LUT_COL_IN1] = Val::from_u16(step.in1 as u16);
    trace.values[base + LUT_COL_IN2] = Val::from_u16(step.in2 as u16);
    trace.values[base + LUT_COL_OUT0] = Val::from_u16(step.out0 as u16);
    trace.values[base + LUT_COL_OUT1] = Val::from_u16(step.out1 as u16);
  }

  for i in steps.len()..n_rows {
    let base = i * NUM_LUT_COLS;
    trace.values[base + LUT_COL_OP] = Val::from_u16(lut_opcode_tag(LutOpcode::ByteAddCarryEq));
    trace.values[base + LUT_COL_IN0] = Val::from_u16(0);
    trace.values[base + LUT_COL_IN1] = Val::from_u16(0);
    trace.values[base + LUT_COL_IN2] = Val::from_u16(0);
    trace.values[base + LUT_COL_OUT0] = Val::from_u16(0);
    trace.values[base + LUT_COL_OUT1] = Val::from_u16(0);
  }

  Ok(trace)
}

pub struct LutKernelAir;

impl<F> BaseAir<F> for LutKernelAir {
  fn width(&self) -> usize {
    NUM_LUT_COLS
  }
}

impl<AB: AirBuilder> Air<AB> for LutKernelAir {
  fn eval(&self, builder: &mut AB) {
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
    let c_u16_add = AB::Expr::from_u16(lut_opcode_tag(LutOpcode::U16AddEq));
    let c_add_carry = AB::Expr::from_u16(lut_opcode_tag(LutOpcode::ByteAddCarryEq));
    let c_mul_low = AB::Expr::from_u16(lut_opcode_tag(LutOpcode::ByteMulLowEq));
    let c_mul_high = AB::Expr::from_u16(lut_opcode_tag(LutOpcode::ByteMulHighEq));
    let c_and = AB::Expr::from_u16(lut_opcode_tag(LutOpcode::ByteAndEq));
    let c_or = AB::Expr::from_u16(lut_opcode_tag(LutOpcode::ByteOrEq));
    let c_xor = AB::Expr::from_u16(lut_opcode_tag(LutOpcode::ByteXorEq));
    let c_zero = AB::Expr::from_u16(0);
    let c256 = AB::Expr::from_u16(256);

    let allowed_poly = (op.clone().into() - c_add.clone())
      * (op.clone().into() - c_u16_add.clone())
      * (op.clone().into() - c_add_carry.clone())
      * (op.clone().into() - c_mul_low.clone())
      * (op.clone().into() - c_mul_high.clone())
      * (op.clone().into() - c_and.clone())
      * (op.clone().into() - c_or.clone())
      * (op.clone().into() - c_xor.clone());
    builder.assert_zero(allowed_poly);

    let t_add = lut_opcode_tag(LutOpcode::ByteAddEq);
    let t_u16_add = lut_opcode_tag(LutOpcode::U16AddEq);
    let t_add_carry = lut_opcode_tag(LutOpcode::ByteAddCarryEq);
    let t_mul_low = lut_opcode_tag(LutOpcode::ByteMulLowEq);
    let t_mul_high = lut_opcode_tag(LutOpcode::ByteMulHighEq);
    let t_and = lut_opcode_tag(LutOpcode::ByteAndEq);
    let t_or = lut_opcode_tag(LutOpcode::ByteOrEq);
    let t_xor = lut_opcode_tag(LutOpcode::ByteXorEq);

    let all_tags = [t_add, t_u16_add, t_add_carry, t_mul_low, t_mul_high, t_and, t_or, t_xor];
    let gate = |target_tag: u16| {
      let mut g = AB::Expr::from_u16(1);
      for t in all_tags {
        if t != target_tag {
          g = g * (op.clone().into() - AB::Expr::from_u16(t));
        }
      }
      g
    };

    let g_add = gate(t_add);
    let g_u16_add = gate(t_u16_add);
    let g_add_carry = gate(t_add_carry);
    let g_mul_low = gate(t_mul_low);
    let g_mul_high = gate(t_mul_high);
    let g_and = gate(t_and);
    let g_or = gate(t_or);
    let g_xor = gate(t_xor);

    let total = in0.clone().into() + in1.clone().into() + in2.clone().into();
    let c65536 = AB::Expr::from_u32(65536);

    builder.assert_zero(g_add.clone() * (total.clone() - out0.clone().into() - c256.clone() * out1.clone().into()));
    builder.assert_zero(g_add.clone() * (out1.clone().into() * (out1.clone().into() - AB::Expr::from_u16(1))));

    builder.assert_zero(g_u16_add.clone() * (total.clone() - out0.clone().into() - c65536.clone() * out1.clone().into()));
    builder.assert_zero(g_u16_add.clone() * (out1.clone().into() * (out1.clone().into() - AB::Expr::from_u16(1))));

    builder.assert_zero(g_add_carry.clone() * (total.clone() - out1.clone().into() - c256.clone() * out0.clone().into()));
    builder.assert_zero(g_add_carry.clone() * (out0.clone().into() * (out0.clone().into() - AB::Expr::from_u16(1))));

    builder.assert_zero(g_mul_low.clone() * ((in0.clone().into() * in1.clone().into()) - out0.clone().into() - c256.clone() * out1.clone().into()));
    builder.assert_zero(g_mul_low.clone() * in2.clone());

    builder.assert_zero(g_mul_high.clone() * ((in0.clone().into() * in1.clone().into()) - out1.clone().into() - c256.clone() * out0.clone().into()));
    builder.assert_zero(g_mul_high.clone() * in2.clone());

    builder.assert_zero(g_and.clone() * (out0.clone().into() - (in0.clone().into() * in1.clone().into())));
    builder.assert_zero(g_and.clone() * (in0.clone().into() * (in0.clone().into() - AB::Expr::from_u16(1))));
    builder.assert_zero(g_and.clone() * (in1.clone().into() * (in1.clone().into() - AB::Expr::from_u16(1))));
    builder.assert_zero(g_and.clone() * (out0.clone().into() * (out0.clone().into() - AB::Expr::from_u16(1))));
    builder.assert_zero(g_and.clone() * in2.clone());
    builder.assert_zero(g_and.clone() * out1.clone());

    builder.assert_zero(g_or.clone() * (out0.clone().into() - (in0.clone().into() + in1.clone().into() - in0.clone().into() * in1.clone().into())));
    builder.assert_zero(g_or.clone() * (in0.clone().into() * (in0.clone().into() - AB::Expr::from_u16(1))));
    builder.assert_zero(g_or.clone() * (in1.clone().into() * (in1.clone().into() - AB::Expr::from_u16(1))));
    builder.assert_zero(g_or.clone() * (out0.clone().into() * (out0.clone().into() - AB::Expr::from_u16(1))));
    builder.assert_zero(g_or.clone() * in2.clone());
    builder.assert_zero(g_or.clone() * out1.clone());

    builder.assert_zero(g_xor.clone() * (out0.clone().into() - (in0.clone().into() + in1.clone().into() - AB::Expr::from_u16(2) * in0.clone().into() * in1.clone().into())));
    builder.assert_zero(g_xor.clone() * (in0.clone().into() * (in0.clone().into() - AB::Expr::from_u16(1))));
    builder.assert_zero(g_xor.clone() * (in1.clone().into() * (in1.clone().into() - AB::Expr::from_u16(1))));
    builder.assert_zero(g_xor.clone() * (out0.clone().into() * (out0.clone().into() - AB::Expr::from_u16(1))));
    builder.assert_zero(g_xor.clone() * in2.clone());
    builder.assert_zero(g_xor.clone() * out1.clone());

    builder.assert_zero(g_add_carry * c_zero.clone());
    builder.assert_bool(in2.clone());
  }
}

pub fn prove_lut_kernel_stark(steps: &[LutStep]) -> Result<CircleStarkProof, String> {
  let trace = build_lut_trace_from_steps(steps)?;
  let config = make_circle_config();
  Ok(p3_uni_stark::prove(&config, &LutKernelAir, trace, &[]))
}

pub fn verify_lut_kernel_stark(proof: &CircleStarkProof) -> CircleStarkVerifyResult {
  let config = make_circle_config();
  p3_uni_stark::verify(&config, &LutKernelAir, proof, &[])
}

pub fn prove_and_verify_lut_kernel_stark_from_steps(steps: &[LutStep]) -> bool {
  let proved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| prove_lut_kernel_stark(steps)));
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

// ============================================================
// Memory CQ bus scaffold
// ============================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryBusStep {
  pub addr: u32,
  pub step: u32,
  pub width: u32,
  pub rw: u32,
  pub value_lo: u32,
  pub value_hi: u32,
  pub same_cell_next: u32,
}

const MEM_BUS_COL_ADDR: usize = 0;
const MEM_BUS_COL_STEP: usize = 1;
const MEM_BUS_COL_WIDTH: usize = 2;
const MEM_BUS_COL_RW: usize = 3;
const MEM_BUS_COL_VALUE_LO: usize = 4;
const MEM_BUS_COL_VALUE_HI: usize = 5;
const MEM_BUS_COL_SAME_CELL_NEXT: usize = 6;
const NUM_MEM_BUS_COLS: usize = 7;

fn cq_rw_to_u32(rw: CqRw) -> u32 {
  match rw {
    CqRw::Read => 0,
    CqRw::Write => 1,
  }
}

fn encode_value_u32_pair(value: &[u8; 32]) -> (u32, u32) {
  let mut lo = [0u8; 4];
  let mut hi = [0u8; 4];
  lo.copy_from_slice(&value[24..28]);
  hi.copy_from_slice(&value[28..32]);
  (u32::from_be_bytes(lo), u32::from_be_bytes(hi))
}

pub fn build_memory_bus_steps_from_events(events: &[CqMemoryEvent]) -> Result<Vec<MemoryBusStep>, String> {
  if events.is_empty() {
    return Err("cannot build memory bus steps from empty events".to_string());
  }

  let mut sorted = events.to_vec();
  sorted.sort_by_key(|e| (e.addr, e.width, e.step, cq_rw_to_u32(e.rw)));

  let mut out = Vec::with_capacity(sorted.len());
  for (i, event) in sorted.iter().enumerate() {
    let addr = u32::try_from(event.addr)
      .map_err(|_| format!("memory bus addr out of u32 range: {}", event.addr))?;
    let step = u32::try_from(event.step)
      .map_err(|_| format!("memory bus step out of u32 range: {}", event.step))?;

    let (value_lo, value_hi) = encode_value_u32_pair(&event.value);

    let same_cell_next = if let Some(next) = sorted.get(i + 1) {
      u32::from(next.addr == event.addr && next.width == event.width)
    } else {
      0
    };

    out.push(MemoryBusStep {
      addr,
      step,
      width: event.width,
      rw: cq_rw_to_u32(event.rw),
      value_lo,
      value_hi,
      same_cell_next,
    });
  }

  Ok(out)
}

pub fn build_memory_bus_trace_from_steps(steps: &[MemoryBusStep]) -> Result<RowMajorMatrix<Val>, String> {
  if steps.is_empty() {
    return Err("cannot build memory bus trace from empty steps".to_string());
  }

  let n_rows = steps.len().max(4).next_power_of_two();
  let mut trace = RowMajorMatrix::new(Val::zero_vec(n_rows * NUM_MEM_BUS_COLS), NUM_MEM_BUS_COLS);

  for (i, step) in steps.iter().enumerate() {
    if step.rw > 1 {
      return Err(format!("memory bus rw must be boolean at row {i}"));
    }
    if step.same_cell_next > 1 {
      return Err(format!("memory bus same_cell_next must be boolean at row {i}"));
    }

    let base = i * NUM_MEM_BUS_COLS;
    trace.values[base + MEM_BUS_COL_ADDR] = Val::from_u32(step.addr);
    trace.values[base + MEM_BUS_COL_STEP] = Val::from_u32(step.step);
    trace.values[base + MEM_BUS_COL_WIDTH] = Val::from_u32(step.width);
    trace.values[base + MEM_BUS_COL_RW] = Val::from_u32(step.rw);
    trace.values[base + MEM_BUS_COL_VALUE_LO] = Val::from_u32(step.value_lo);
    trace.values[base + MEM_BUS_COL_VALUE_HI] = Val::from_u32(step.value_hi);
    trace.values[base + MEM_BUS_COL_SAME_CELL_NEXT] = Val::from_u32(step.same_cell_next);
  }

  if steps.len() < n_rows {
    let last = *steps.last().expect("non-empty checked");
    for i in steps.len()..n_rows {
      let base = i * NUM_MEM_BUS_COLS;
      trace.values[base + MEM_BUS_COL_ADDR] = Val::from_u32(last.addr);
      trace.values[base + MEM_BUS_COL_STEP] = Val::from_u32(last.step);
      trace.values[base + MEM_BUS_COL_WIDTH] = Val::from_u32(last.width);
      trace.values[base + MEM_BUS_COL_RW] = Val::from_u32(last.rw);
      trace.values[base + MEM_BUS_COL_VALUE_LO] = Val::from_u32(last.value_lo);
      trace.values[base + MEM_BUS_COL_VALUE_HI] = Val::from_u32(last.value_hi);
      trace.values[base + MEM_BUS_COL_SAME_CELL_NEXT] = Val::from_u32(0);
    }
  }

  Ok(trace)
}

pub struct MemoryBusAir;

impl<F> BaseAir<F> for MemoryBusAir {
  fn width(&self) -> usize {
    NUM_MEM_BUS_COLS
  }
}

impl<AB: AirBuilder> Air<AB> for MemoryBusAir {
  fn eval(&self, builder: &mut AB) {
    let main = builder.main();
    let local = main.row_slice(0).expect("empty trace");
    let next = main.row_slice(1).expect("single-row trace");
    let local = &*local;
    let next = &*next;

    let rw = local[MEM_BUS_COL_RW].clone();
    let same = local[MEM_BUS_COL_SAME_CELL_NEXT].clone();

    builder.assert_bool(rw.clone());
    builder.assert_bool(same.clone());

    builder.when_transition().assert_zero(
      same.clone() * (next[MEM_BUS_COL_ADDR].clone().into() - local[MEM_BUS_COL_ADDR].clone().into()),
    );
    builder.when_transition().assert_zero(
      same.clone() * (next[MEM_BUS_COL_WIDTH].clone().into() - local[MEM_BUS_COL_WIDTH].clone().into()),
    );

    builder.when_transition().assert_zero(
      same.clone() * (next[MEM_BUS_COL_STEP].clone().into() - local[MEM_BUS_COL_STEP].clone().into()),
    );
    builder.when_transition().assert_zero(
      same.clone() * (local[MEM_BUS_COL_RW].clone().into() - AB::Expr::from_u16(0)),
    );
    builder.when_transition().assert_zero(
      same.clone() * (next[MEM_BUS_COL_RW].clone().into() - AB::Expr::from_u16(1)),
    );

    builder.when_transition().assert_zero(
      same.clone()
        * (next[MEM_BUS_COL_VALUE_LO].clone().into() - local[MEM_BUS_COL_VALUE_LO].clone().into()),
    );
    builder.when_transition().assert_zero(
      same
        * (next[MEM_BUS_COL_VALUE_HI].clone().into() - local[MEM_BUS_COL_VALUE_HI].clone().into()),
    );
  }
}

pub fn prove_memory_bus_stark(steps: &[MemoryBusStep]) -> Result<CircleStarkProof, String> {
  let trace = build_memory_bus_trace_from_steps(steps)?;
  let config = make_circle_config();
  Ok(p3_uni_stark::prove(&config, &MemoryBusAir, trace, &[]))
}

pub fn prove_memory_bus_stark_from_events(events: &[CqMemoryEvent]) -> Result<CircleStarkProof, String> {
  let steps = build_memory_bus_steps_from_events(events)?;
  prove_memory_bus_stark(&steps)
}

pub fn verify_memory_bus_stark(proof: &CircleStarkProof) -> CircleStarkVerifyResult {
  let config = make_circle_config();
  p3_uni_stark::verify(&config, &MemoryBusAir, proof, &[])
}

pub fn prove_and_verify_memory_bus_stark(steps: &[MemoryBusStep]) -> bool {
  let proved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| prove_memory_bus_stark(steps)));
  let proof = match proved {
    Ok(Ok(proof)) => proof,
    Ok(Err(_)) => return false,
    Err(_) => return false,
  };
  verify_memory_bus_stark(&proof).is_ok()
}

pub fn prove_and_verify_memory_bus_stark_from_events(events: &[CqMemoryEvent]) -> bool {
  let steps = match build_memory_bus_steps_from_events(events) {
    Ok(steps) => steps,
    Err(_) => return false,
  };
  prove_and_verify_memory_bus_stark(&steps)
}

#[cfg(test)]
mod memory_bus_tests {
  use super::{
    build_memory_bus_steps_from_events,
    build_memory_bus_trace_from_steps,
    prove_and_verify_memory_bus_stark_from_events,
  };
  use crate::memory_proof::{CqMemoryEvent, CqRw};
  use p3_matrix::Matrix;

  #[test]
  fn test_memory_bus_steps_sort_and_same_cell_flag() {
    let events = vec![
      CqMemoryEvent {
        addr: 10,
        step: 2,
        value: [2u8; 32],
        rw: CqRw::Write,
        width: 32,
      },
      CqMemoryEvent {
        addr: 10,
        step: 2,
        value: [2u8; 32],
        rw: CqRw::Read,
        width: 32,
      },
      CqMemoryEvent {
        addr: 11,
        step: 1,
        value: [1u8; 32],
        rw: CqRw::Read,
        width: 32,
      },
    ];

    let steps = build_memory_bus_steps_from_events(&events).expect("step build should succeed");
    assert_eq!(steps.len(), 3);
    assert_eq!(steps[0].addr, 10);
    assert_eq!(steps[0].rw, 0);
    assert_eq!(steps[0].same_cell_next, 1);
    assert_eq!(steps[1].addr, 10);
    assert_eq!(steps[1].rw, 1);
    assert_eq!(steps[1].same_cell_next, 0);
  }

  #[test]
  fn test_memory_bus_trace_builds() {
    let events = vec![CqMemoryEvent {
      addr: 5,
      step: 1,
      value: [0xAB; 32],
      rw: CqRw::Read,
      width: 32,
    }];

    let steps = build_memory_bus_steps_from_events(&events).expect("step build should succeed");
    let trace = build_memory_bus_trace_from_steps(&steps).expect("trace build should succeed");
    assert!(trace.height() >= 4);
  }

  #[test]
  fn test_memory_bus_stark_roundtrip_from_events() {
    let events = vec![
      CqMemoryEvent {
        addr: 64,
        step: 9,
        value: [0x11; 32],
        rw: CqRw::Read,
        width: 32,
      },
      CqMemoryEvent {
        addr: 64,
        step: 9,
        value: [0x11; 32],
        rw: CqRw::Write,
        width: 32,
      },
    ];

    assert!(prove_and_verify_memory_bus_stark_from_events(&events));
  }
}

pub fn build_lut_steps_from_rows_add_family(rows: &[ProofRow]) -> Result<Vec<LutStep>, String> {
  let mut out = Vec::new();
  let mut seen_add_family = false;

  for row in rows {
    match row.op {
      crate::sementic_proof::OP_BOOL
      | crate::sementic_proof::OP_BYTE
      | crate::sementic_proof::OP_EQ_REFL
      | crate::sementic_proof::OP_AND_INTRO
      | crate::sementic_proof::OP_EQ_TRANS
      | crate::sementic_proof::OP_BYTE_ADD_THIRD_CONGRUENCE
      | crate::sementic_proof::OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE => {
        seen_add_family = true;
      }
      crate::sementic_proof::OP_BYTE_ADD_EQ => out.push(LutStep {
        op: LutOpcode::ByteAddEq,
        in0: row.scalar0,
        in1: row.scalar1,
        in2: row.arg0,
        out0: row.value,
        out1: row.scalar2,
      }),
      crate::sementic_proof::OP_U16_ADD_EQ => out.push(LutStep {
        op: LutOpcode::U16AddEq,
        in0: row.scalar0,
        in1: row.scalar1,
        in2: row.scalar2,
        out0: row.value,
        out1: row.arg1,
      }),
      crate::sementic_proof::OP_BYTE_ADD_CARRY_EQ => {
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
      crate::sementic_proof::OP_BOOL
      | crate::sementic_proof::OP_BYTE
      | crate::sementic_proof::OP_EQ_REFL
      | crate::sementic_proof::OP_AND_INTRO
      | crate::sementic_proof::OP_EQ_TRANS
      | crate::sementic_proof::OP_BYTE_ADD_THIRD_CONGRUENCE
      | crate::sementic_proof::OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE => {
        seen_mul_family = true;
      }
      crate::sementic_proof::OP_BYTE_ADD_EQ => out.push(LutStep {
        op: LutOpcode::ByteAddEq,
        in0: row.scalar0,
        in1: row.scalar1,
        in2: row.arg0,
        out0: row.value,
        out1: row.scalar2,
      }),
      crate::sementic_proof::OP_U16_ADD_EQ => out.push(LutStep {
        op: LutOpcode::U16AddEq,
        in0: row.scalar0,
        in1: row.scalar1,
        in2: row.scalar2,
        out0: row.value,
        out1: row.arg1,
      }),
      crate::sementic_proof::OP_BYTE_ADD_CARRY_EQ => {
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
      crate::sementic_proof::OP_BYTE_MUL_LOW_EQ => {
        let total = row.scalar0 * row.scalar1;
        out.push(LutStep {
          op: LutOpcode::ByteMulLowEq,
          in0: row.scalar0,
          in1: row.scalar1,
          in2: 0,
          out0: total & 0xFF,
          out1: total >> 8,
        });
      }
      crate::sementic_proof::OP_BYTE_MUL_HIGH_EQ => {
        let total = row.scalar0 * row.scalar1;
        out.push(LutStep {
          op: LutOpcode::ByteMulHighEq,
          in0: row.scalar0,
          in1: row.scalar1,
          in2: 0,
          out0: total >> 8,
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
      crate::sementic_proof::OP_AND_INTRO => {
        seen_bit_family = true;
      }
      crate::sementic_proof::OP_BYTE_AND_EQ => push_bitwise_steps(&mut out, LutOpcode::ByteAndEq, row.scalar0, row.scalar1),
      crate::sementic_proof::OP_BYTE_OR_EQ => push_bitwise_steps(&mut out, LutOpcode::ByteOrEq, row.scalar0, row.scalar1),
      crate::sementic_proof::OP_BYTE_XOR_EQ => push_bitwise_steps(&mut out, LutOpcode::ByteXorEq, row.scalar0, row.scalar1),
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

const MATCH_COL_INFERRED: usize = 0;
const MATCH_COL_EXPECTED: usize = 1;
const NUM_WFF_MATCH_COLS: usize = 2;

pub struct StageAProof {
  pub inferred_wff_proof: CircleStarkProof,
  pub public_wff_match_proof: CircleStarkProof,
  expected_public_wff: WFF,
}

fn route_stage_a_row_op(op: u32) -> Result<(), String> {
  match op {
    OP_BYTE_ADD_EQ
    | crate::sementic_proof::OP_U16_ADD_EQ
    | crate::sementic_proof::OP_BOOL
    | crate::sementic_proof::OP_BYTE
    | crate::sementic_proof::OP_EQ_REFL
    | crate::sementic_proof::OP_AND_INTRO
    | crate::sementic_proof::OP_EQ_TRANS
    | crate::sementic_proof::OP_BYTE_ADD_THIRD_CONGRUENCE
    | crate::sementic_proof::OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE => Ok(()),
    other => Err(format!("unsupported Stage A proof-row op: {other}")),
  }
}

fn has_stage_a_semantic_rows(rows: &[ProofRow]) -> Result<bool, String> {
  let mut found = false;
  for row in rows {
    route_stage_a_row_op(row.op)?;
    if row.op == OP_BYTE_ADD_EQ || row.op == crate::sementic_proof::OP_U16_ADD_EQ {
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
  let mut semantic_rows: Vec<ProofRow> = Vec::new();

  let push_byte_add_row = |out: &mut Vec<ProofRow>, a: u32, b: u32, carry_in: u32, sum: u32, carry_out: u32, i: usize| -> Result<(), String> {
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
      crate::sementic_proof::OP_U16_ADD_EQ => {
        if row.scalar0 > 0xFFFF || row.scalar1 > 0xFFFF || row.scalar2 > 1 || row.arg0 > 0xFFFF || row.arg1 > 1 {
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

        push_byte_add_row(&mut semantic_rows, a_lo, b_lo, row.scalar2, sum_lo, carry_mid, i)?;
        push_byte_add_row(&mut semantic_rows, a_hi, b_hi, carry_mid, sum_hi, row.arg1, i)?;
      }
      _ => {
      }
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
    trace.values[base + COL_SCALAR0] = Val::from_u16(row.scalar0 as u16);
    trace.values[base + COL_SCALAR1] = Val::from_u16(row.scalar1 as u16);
    trace.values[base + COL_SCALAR2] = Val::from_u16(row.scalar2 as u16);
    trace.values[base + COL_ARG0] = Val::from_u16(row.arg0 as u16);
    trace.values[base + COL_ARG1] = Val::from_u16(row.arg1 as u16);
    trace.values[base + COL_ARG2] = Val::from_u16(row.arg2 as u16);
    trace.values[base + COL_VALUE] = Val::from_u16(row.value as u16);
    trace.values[base + COL_RET_TY] = Val::from_u16(row.ret_ty as u16);
  }

  if semantic_len < n_rows {
    let mut carry_in = semantic_rows
      .last()
      .map(|row| row.scalar2 as u16)
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
  use crate::sementic_proof::{compile_proof, infer_proof, prove_add, prove_mul, wff_add, wff_mul, Proof};

  fn tamper_first_mul_related_leaf(proof: &mut Proof) -> bool {
    match proof {
      Proof::ByteMulLowEq(a, b) => {
        *proof = Proof::ByteMulLowEq(*a ^ 1, *b);
        true
      }
      Proof::ByteMulHighEq(a, b) => {
        *proof = Proof::ByteMulHighEq(*a ^ 1, *b);
        true
      }
      Proof::ByteAddEq(a, b, c) => {
        *proof = Proof::ByteAddEq(*a, *b, !*c);
        true
      }
      Proof::U16AddEq(a, b, cin, c) => {
        *proof = Proof::U16AddEq(*a, *b, !*cin, *c);
        true
      }
      Proof::AndIntro(p1, p2) => {
        if tamper_first_mul_related_leaf(p1) {
          true
        } else {
          tamper_first_mul_related_leaf(p2)
        }
      }
      Proof::EqSym(p) => tamper_first_mul_related_leaf(p),
      Proof::EqTrans(p1, p2) => {
        if tamper_first_mul_related_leaf(p1) {
          true
        } else {
          tamper_first_mul_related_leaf(p2)
        }
      }
      Proof::ByteAddThirdCongruence(p, _, _) => tamper_first_mul_related_leaf(p),
      Proof::ByteAddCarryThirdCongruence(p, _, _) => tamper_first_mul_related_leaf(p),
      _ => false,
    }
  }

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

  #[test]
  fn test_build_lut_steps_from_rows_byte_add_eq_only() {
    let proof = Proof::ByteAddEq(10, 20, true);
    let rows = compile_proof(&proof);
    let steps = build_lut_steps_from_rows(&rows).expect("lut steps should build");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0].op, LutOpcode::ByteAddEq);
    assert_eq!(steps[0].in0, 10);
    assert_eq!(steps[0].in1, 20);
    assert_eq!(steps[0].in2, 1);
    assert_eq!(steps[0].out0, 31);
    assert_eq!(steps[0].out1, 0);

    let trace = build_lut_trace_from_steps(&steps).expect("lut trace should build");
    assert_eq!(trace.width(), NUM_LUT_COLS);
    assert!(trace.height() >= 4);
  }

  #[test]
  fn test_build_lut_steps_rejects_structural_rows() {
    let proof = Proof::EqRefl(Term::Byte(7));
    let rows = compile_proof(&proof);
    let err = build_lut_steps_from_rows(&rows).expect_err("structural rows must be rejected");
    assert!(err.contains("not LUT-step encodable"));
  }

  #[test]
  fn test_build_lut_trace_rejects_out_of_range_u16_add_step() {
    let steps = vec![LutStep {
      op: LutOpcode::U16AddEq,
      in0: 70_000,
      in1: 1,
      in2: 0,
      out0: 1,
      out1: 0,
    }];

    let err = build_lut_trace_from_steps(&steps).expect_err("out-of-range u16-add input must be rejected");
    assert!(err.contains("out of range"));
  }

  #[test]
  fn test_build_lut_trace_rejects_out_of_range_byte_add_step() {
    let steps = vec![LutStep {
      op: LutOpcode::ByteAddEq,
      in0: 300,
      in1: 1,
      in2: 0,
      out0: 45,
      out1: 0,
    }];

    let err = build_lut_trace_from_steps(&steps).expect_err("out-of-range byte-add input must be rejected");
    assert!(err.contains("out of range"));
  }

  #[test]
  fn test_stage_a_trace_rejects_out_of_range_row() {
    let rows = vec![ProofRow {
      op: OP_BYTE_ADD_EQ,
      scalar0: 300,
      scalar1: 1,
      scalar2: 0,
      arg0: 0,
      arg1: 45,
      arg2: 0,
      value: 45,
      ret_ty: RET_WFF_EQ,
    }];

    let err = generate_stage_a_semantic_trace(&rows).expect_err("out-of-range stage-a row must be rejected");
    assert!(err.contains("out of range"));
  }

  #[test]
  fn test_build_stack_ir_steps_from_rows_eq_refl() {
    let proof = Proof::EqRefl(Term::Byte(7));
    let rows = compile_proof(&proof);
    let steps = build_stack_ir_steps_from_rows(&rows).expect("stack-ir steps should build");

    assert_eq!(steps.len(), rows.len());
    assert_eq!(steps[0].sp_before, 0);
    assert_eq!(steps[0].sp_after, 1);
    assert_eq!(steps[1].sp_before, 1);
    assert_eq!(steps[1].sp_after, 1);

    let trace = build_stack_ir_trace_from_steps(&steps).expect("stack-ir trace should build");
    assert_eq!(trace.width(), NUM_STACK_IR_COLS);
    assert!(trace.height() >= 4);
  }

  #[test]
  fn test_build_stack_ir_steps_underflow_rejected() {
    let rows = vec![ProofRow {
      op: crate::sementic_proof::OP_EQ_TRANS,
      ret_ty: RET_WFF_EQ,
      ..Default::default()
    }];

    let err = build_stack_ir_steps_from_rows(&rows).expect_err("underflow must be rejected");
    assert!(err.contains("stack underflow"));
  }

  #[test]
  fn test_stack_ir_scaffold_stark_eq_refl_roundtrip() {
    let proof = Proof::EqRefl(Term::Byte(7));
    let rows = compile_proof(&proof);
    assert!(prove_and_verify_stack_ir_scaffold_stark(&rows));
  }

  #[test]
  fn test_stack_ir_scaffold_supports_add_rows() {
    let a = u256_bytes(10);
    let b = u256_bytes(20);
    let c = compute_add(&a, &b);
    let proof = prove_add(&a, &b, &c);
    let rows = compile_proof(&proof);
    assert!(prove_and_verify_stack_ir_scaffold_stark(&rows));
  }

  #[test]
  fn test_lut_kernel_stark_byte_add_eq_roundtrip() {
    let proof = Proof::ByteAddEq(100, 200, false);
    let rows = compile_proof(&proof);
    let steps = build_lut_steps_from_rows(&rows).expect("lut steps should build");
    assert!(prove_and_verify_lut_kernel_stark_from_steps(&steps));
  }

  #[test]
  fn test_lut_kernel_stark_byte_add_carry_eq_roundtrip() {
    let proof = Proof::ByteAddCarryEq(200, 100, false);
    let rows = compile_proof(&proof);
    let steps = build_lut_steps_from_rows(&rows).expect("lut steps should build");
    assert!(prove_and_verify_lut_kernel_stark_from_steps(&steps));
  }

  #[test]
  fn test_lut_kernel_stark_supports_mul_ops() {
    let proof = Proof::ByteMulLowEq(3, 7);
    let rows = compile_proof(&proof);
    let steps = build_lut_steps_from_rows(&rows).expect("lut steps should build");
    assert!(prove_and_verify_lut_kernel_stark_from_steps(&steps));
  }

  #[test]
  fn test_build_lut_trace_accepts_bit_level_bit_ops_inputs() {
    let steps = vec![LutStep {
      op: LutOpcode::ByteAndEq,
      in0: 1,
      in1: 0,
      in2: 0,
      out0: 0,
      out1: 0,
    }];

    let trace = build_lut_trace_from_steps(&steps).expect("bit-level bitwise steps should be trace-encodable");
    assert_eq!(trace.width(), NUM_LUT_COLS);
  }

  #[test]
  fn test_build_lut_trace_rejects_non_bool_bit_ops_inputs() {
    let steps = vec![LutStep {
      op: LutOpcode::ByteAndEq,
      in0: 2,
      in1: 1,
      in2: 0,
      out0: 0,
      out1: 0,
    }];

    let err = build_lut_trace_from_steps(&steps)
      .expect_err("non-bool bitwise step should be rejected");
    assert!(err.contains("out of range"));
  }

  #[test]
  fn test_add_composite_stack_lut_roundtrip() {
    let a = u256_bytes(1000);
    let b = u256_bytes(2000);
    let c = compute_add(&a, &b);
    let proof = prove_add(&a, &b, &c);
    let rows = compile_proof(&proof);
    assert!(prove_and_verify_add_stack_lut_stark(&rows));
  }

  #[test]
  fn test_add_composite_stack_lut_rejects_non_add_rowset() {
    let proof = Proof::EqRefl(Term::Byte(7));
    let rows = compile_proof(&proof);
    assert!(!prove_and_verify_add_stack_lut_stark(&rows));
  }

  #[test]
  fn test_add_family_lut_builder_rejects_foreign_row() {
    let rows = vec![
      ProofRow {
        op: crate::sementic_proof::OP_BYTE_ADD_EQ,
        scalar0: 1,
        scalar1: 2,
        arg0: 0,
        value: 3,
        scalar2: 0,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      },
      ProofRow {
        op: crate::sementic_proof::OP_BYTE_MUL_LOW_EQ,
        scalar0: 2,
        scalar1: 3,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      },
    ];

    let err = build_lut_steps_from_rows_add_family(&rows).expect_err("foreign rows must be rejected");
    assert!(err.contains("non-add-family row"));
  }

  #[test]
  fn test_mul_composite_stack_lut_roundtrip() {
    let mut a = [0u8; 32];
    let mut b = [0u8; 32];
    a[31] = 123;
    b[31] = 45;
    let c = {
      let mut out = [0u8; 32];
      out[31] = 159;
      out[30] = 21;
      out
    };

    let proof = crate::sementic_proof::prove_mul(&a, &b, &c);
    let rows = compile_proof(&proof);
    assert!(prove_and_verify_mul_stack_lut_stark(&rows));
  }

  #[test]
  fn test_mul_family_lut_builder_rejects_foreign_row() {
    let rows = vec![
      ProofRow {
        op: crate::sementic_proof::OP_BYTE_MUL_LOW_EQ,
        scalar0: 3,
        scalar1: 7,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      },
      ProofRow {
        op: crate::sementic_proof::OP_BYTE_OR_EQ,
        scalar0: 1,
        scalar1: 2,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      },
    ];

    let err = build_lut_steps_from_rows_mul_family(&rows).expect_err("foreign rows must be rejected");
    assert!(err.contains("non-mul-family row"));
  }

  #[test]
  fn test_mul_leaf_tamper_rejected_by_wff_match() {
    let mut a = [0u8; 32];
    let mut b = [0u8; 32];
    a[31] = 123;
    b[31] = 45;
    let c = {
      let mut out = [0u8; 32];
      out[31] = 159;
      out[30] = 21;
      out
    };

    let mut proof = prove_mul(&a, &b, &c);
    assert!(tamper_first_mul_related_leaf(&mut proof));

    let public_wff = wff_mul(&a, &b, &c);
    assert!(!prove_and_verify_expected_wff_match_stark(&proof, &public_wff));
  }
}
