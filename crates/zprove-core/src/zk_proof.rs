//! Circle STARK proof generation and verification for semantic proof rows.
//!
//! Stage A/B overview:
//! - Stage 1 proves inferred-WFF semantic constraints over compiled `ProofRow`s.
//! - Stage 2 proves inferred WFF equals public WFF via serialized equality trace.
//!
//! The current Stage-1 semantic AIR kernel enforces the byte-add equality rows
//! (`OP_BYTE_ADD_EQ`) embedded in `ProofRow` encoding.

use crate::memory_proof::{CqMemoryEvent, CqRw};
use crate::semantic_proof::{
  NUM_PROOF_COLS, OP_BYTE_ADD_EQ, Proof, ProofRow, RET_BYTE, RET_WFF_AND, RET_WFF_EQ, Term, WFF,
  compile_proof, infer_proof, verify_compiled,
};

use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
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
use p3_uni_stark::StarkConfig;
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

pub fn make_receipt_binding_public_values(tag: u32, opcode: u8, expected_wff: &WFF) -> Vec<Val> {
  let mut rng = SmallRng::seed_from_u64(0xC0DEC0DE_u64);
  let poseidon = Poseidon2Mersenne31::<16>::new_from_rng_128(&mut rng);
  let sponge = PaddingFreeSponge::<_, 16, 8, 8>::new(poseidon);

  let mut input = Vec::new();
  input.push(Val::from_u32(tag));
  input.push(Val::from_u32(opcode as u32));
  input.push(Val::from_u32(serialize_wff_bytes(expected_wff).len() as u32));
  input.extend(
    serialize_wff_bytes(expected_wff)
      .into_iter()
      .map(Val::from_u8),
  );

  let digest = sponge.hash_iter(input);
  let mut public_values = Vec::with_capacity(RECEIPT_BIND_PUBLIC_VALUES_LEN);
  public_values.push(Val::from_u32(tag));
  public_values.push(Val::from_u32(opcode as u32));
  public_values.extend_from_slice(&digest);
  public_values
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

impl<AB: AirBuilderWithPublicValues> Air<AB> for StackIrAir {
  fn eval(&self, builder: &mut AB) {
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
      * (op.clone().into() - c_xor_eq.clone());
    builder.assert_zero(allowed_poly);

    builder.assert_zero(
      sp_before.clone().into() - pop.clone().into() + push.clone().into() - sp_after.clone().into(),
    );

    builder.when_first_row().assert_zero(sp_before.clone());
    builder
      .when_transition()
      .assert_eq(next[STACK_COL_SP_BEFORE].clone(), sp_after.clone());

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
const NUM_LUT_COLS: usize = 6;

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
      crate::semantic_proof::OP_BYTE_MUL_LOW_EQ => LutStep {
        op: LutOpcode::ByteMulLowEq,
        in0: row.scalar0,
        in1: row.scalar1,
        in2: 0,
        out0: (row.scalar0 * row.scalar1) & 0xFF,
        out1: 0,
      },
      crate::semantic_proof::OP_BYTE_MUL_HIGH_EQ => LutStep {
        op: LutOpcode::ByteMulHighEq,
        in0: row.scalar0,
        in1: row.scalar1,
        in2: 0,
        out0: (row.scalar0 * row.scalar1) >> 8,
        out1: 0,
      },
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
    let c_zero = AB::Expr::from_u16(0);
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

    builder.assert_zero(g_add_carry * c_zero.clone());
    builder.assert_bool(in2.clone());
  }
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

pub fn build_memory_bus_steps_from_events(
  events: &[CqMemoryEvent],
) -> Result<Vec<MemoryBusStep>, String> {
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

pub fn build_memory_bus_trace_from_steps(
  steps: &[MemoryBusStep],
) -> Result<RowMajorMatrix<Val>, String> {
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
      return Err(format!(
        "memory bus same_cell_next must be boolean at row {i}"
      ));
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
      same.clone()
        * (next[MEM_BUS_COL_ADDR].clone().into() - local[MEM_BUS_COL_ADDR].clone().into()),
    );
    builder.when_transition().assert_zero(
      same.clone()
        * (next[MEM_BUS_COL_WIDTH].clone().into() - local[MEM_BUS_COL_WIDTH].clone().into()),
    );

    builder.when_transition().assert_zero(
      same.clone()
        * (next[MEM_BUS_COL_STEP].clone().into() - local[MEM_BUS_COL_STEP].clone().into()),
    );
    builder
      .when_transition()
      .assert_zero(same.clone() * (local[MEM_BUS_COL_RW].clone().into() - AB::Expr::from_u16(0)));
    builder
      .when_transition()
      .assert_zero(same.clone() * (next[MEM_BUS_COL_RW].clone().into() - AB::Expr::from_u16(1)));

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

pub fn prove_memory_bus_stark_from_events(
  events: &[CqMemoryEvent],
) -> Result<CircleStarkProof, String> {
  let steps = build_memory_bus_steps_from_events(events)?;
  prove_memory_bus_stark(&steps)
}

pub fn verify_memory_bus_stark(proof: &CircleStarkProof) -> CircleStarkVerifyResult {
  let config = make_circle_config();
  p3_uni_stark::verify(&config, &MemoryBusAir, proof, &[])
}

pub fn prove_and_verify_memory_bus_stark(steps: &[MemoryBusStep]) -> bool {
  let proved = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
    prove_memory_bus_stark(steps)
  }));
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
      | crate::semantic_proof::OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE => {
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
      crate::semantic_proof::OP_BOOL
      | crate::semantic_proof::OP_BYTE
      | crate::semantic_proof::OP_EQ_REFL
      | crate::semantic_proof::OP_AND_INTRO
      | crate::semantic_proof::OP_EQ_TRANS
      | crate::semantic_proof::OP_BYTE_ADD_THIRD_CONGRUENCE
      | crate::semantic_proof::OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE => {
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

    builder.assert_eq(
      local[MATCH_COL_INFERRED].clone(),
      local[MATCH_COL_EXPECTED].clone(),
    );
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

  let padded_rows = inferred_bytes.len().next_power_of_two();
  let mut trace = RowMajorMatrix::new(
    Val::zero_vec(padded_rows * NUM_WFF_MATCH_COLS),
    NUM_WFF_MATCH_COLS,
  );
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
/// OP_BYTE_ADD_THIRD_CONGRUENCE, OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE) are skipped
/// silently.  Bit-family ops are expanded to per-bit level as required by the
/// LUT kernel AIR.  All other recognised LUT ops are processed normally.
pub fn build_lut_steps_from_rows_auto(rows: &[ProofRow]) -> Result<Vec<LutStep>, String> {
  use crate::semantic_proof::{
    OP_AND_INTRO, OP_BOOL, OP_BYTE, OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE,
    OP_BYTE_ADD_THIRD_CONGRUENCE, OP_EQ_REFL, OP_EQ_TRANS,
  };

  let mut out = Vec::with_capacity(rows.len() * 2);

  for row in rows {
    match row.op {
      // ── Structural rows: skip silently ────────────────────────────────────
      op if op == OP_BOOL
        || op == OP_BYTE
        || op == OP_EQ_REFL
        || op == OP_AND_INTRO
        || op == OP_EQ_TRANS
        || op == OP_BYTE_ADD_THIRD_CONGRUENCE
        || op == OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE => {}

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
      crate::semantic_proof::OP_BYTE_MUL_LOW_EQ => out.push(LutStep {
        op: LutOpcode::ByteMulLowEq,
        in0: row.scalar0,
        in1: row.scalar1,
        in2: 0,
        out0: (row.scalar0 * row.scalar1) & 0xFF,
        out1: 0,
      }),
      crate::semantic_proof::OP_BYTE_MUL_HIGH_EQ => out.push(LutStep {
        op: LutOpcode::ByteMulHighEq,
        in0: row.scalar0,
        in1: row.scalar1,
        in2: 0,
        out0: (row.scalar0 * row.scalar1) >> 8,
        out1: 0,
      }),

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
