
use std::{array, fmt};
use serde::Serialize;

// ============================================================
// Types
// ============================================================

/// Types in the proof system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Ty {
  Bool,
  Byte,
}

// ============================================================
// Terms
// ============================================================

/// Terms represent values and computations in the proof language.
///
/// Byte-level terms are the atoms; word-level terms compose 32 bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum Term {
  // ---- Boolean-level ----
  /// Boolean constant (0 or 1).
  Bool(bool),
  /// `Not(a) = !a`.
  Not(Box<Term>),
  /// `And(a, b) = a && b`.
  And(Box<Term>, Box<Term>),
  /// `Or(a, b) = a || b`.
  Or(Box<Term>, Box<Term>),
  /// `Xor(a, b) = a != b`.
  Xor(Box<Term>, Box<Term>),
  /// `ite(c, a, b) = if c then a else b`.
  Ite(Box<Term>, Box<Term>, Box<Term>),
  // ---- Byte-level ----
  /// Concrete byte value.
  Byte(u8),
  /// `ByteAdd(a, b, c) = (a + b + c) mod 256` where c ∈ {0,1}.
  ByteAdd(Box<Term>, Box<Term>, Box<Term>),
  /// `ByteAddCarry(a, b, c) = (a + b + c) / 256` — carry output (0 or 1).
  ByteAddCarry(Box<Term>, Box<Term>, Box<Term>),
  /// `ByteMulLow(a, b) = (a * b) mod 256`.
  ByteMulLow(Box<Term>, Box<Term>),
  /// `ByteMulHigh(a, b) = (a * b) / 256`.
  ByteMulHigh(Box<Term>, Box<Term>),
  /// `ByteAnd(a, b) = a & b`.
  ByteAnd(Box<Term>, Box<Term>),
  /// `ByteOr(a, b) = a | b`.
  ByteOr(Box<Term>, Box<Term>),
  /// `ByteXor(a, b) = a ^ b`.
  ByteXor(Box<Term>, Box<Term>)
}

// ============================================================
// Well-formed formulas
// ============================================================

/// Well-formed formulas in the proof system.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum WFF {
  /// `t1 = t2`.
  Equal(Box<Term>, Box<Term>),
  /// `φ ∧ ψ`.
  And(Box<WFF>, Box<WFF>),
}

// ============================================================
// Proof terms
// ============================================================

/// Hilbert-style proof terms.
///
/// Each variant is either an **axiom schema** (leaf) or an **inference rule**
/// (interior node with sub-proofs).
#[derive(Debug, Clone, Serialize)]
pub enum Proof {
  // ======== Logical inference rules ========
  /// From `φ` and `ψ`, derive `φ ∧ ψ`.
  AndIntro(Box<Proof>, Box<Proof>),
  // ======== Structural equality rules ========
  /// Axiom: `t = t` (reflexivity).
  EqRefl(Term),
  /// From `(a = b)`, derive `(b = a)` (symmetry).
  EqSym(Box<Proof>),
  /// From `(a = b)` and `(b = c)`, derive `(a = c)` (transitivity).
  EqTrans(Box<Proof>, Box<Proof>),

  // ======== Byte axioms (lookup-table verifiable) ========
  /// `ByteAdd(Byte(a), Byte(b), Bool(c)) = Byte((a+b+c) mod 256)`.
  ByteAddEq(u8, u8, bool),
  /// `ByteAddCarry(Byte(a), Byte(b), Bool(c)) = Bool((a+b+c) >= 256)`.
  ByteAddCarryEq(u8, u8, bool),
  /// `ByteMulLow(Byte(a), Byte(b)) = Byte((a*b) mod 256)`.
  ByteMulLowEq(u8, u8),
  /// `ByteMulHigh(Byte(a), Byte(b)) = Byte((a*b) / 256)`.
  ByteMulHighEq(u8, u8),
  /// `ByteAnd(Byte(a), Byte(b)) = Byte(a & b)`.
  ByteAndEq(u8, u8),
  /// `ByteOr(Byte(a), Byte(b)) = Byte(a | b)`.
  ByteOrEq(u8, u8),
  /// `ByteXor(Byte(a), Byte(b)) = Byte(a ^ b)`.
  ByteXorEq(u8, u8),
  /// From `(c1 = c2)`, derive
  /// `ByteAdd(Byte(a), Byte(b), c1) = ByteAdd(Byte(a), Byte(b), c2)`.
  ByteAddThirdCongruence(Box<Proof>, u8, u8),
  /// From `(c1 = c2)`, derive
  /// `ByteAddCarry(Byte(a), Byte(b), c1) = ByteAddCarry(Byte(a), Byte(b), c2)`.
  ByteAddCarryThirdCongruence(Box<Proof>, u8, u8),
  /// `ite(true, a, b) = a`.
  IteTrueEq(Term, Term),
  /// `ite(false, a, b) = b`.
  IteFalseEq(Term, Term),
}

// ============================================================
// Verification errors
// ============================================================

/// Errors produced during Hilbert-style proof verification.
#[derive(Debug, Clone)]
pub enum VerifyError {
  /// Expected a different term variant.
  UnexpectedTermVariant { expected: &'static str },
  /// Expected a different proof variant.
  UnexpectedProofVariant { expected: &'static str },
  /// Transitivity: intermediate terms don't match.
  TransitivityMismatch,
  /// Decide failed: expected equal bytes but got different values.
  ByteDecideFailed,
}

impl fmt::Display for VerifyError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::UnexpectedTermVariant { expected } => {
        write!(f, "expected term variant: {expected}")
      }
      Self::UnexpectedProofVariant { expected } => {
        write!(f, "expected proof variant: {expected}")
      }
      Self::TransitivityMismatch => {
        write!(f, "transitivity: intermediate terms don't match")
      }
      Self::ByteDecideFailed => {
        write!(f, "decide failed: irreducible terms are found")
      }
    }
  }
}

// ============================================================
// Compiling Proofs — opcode table
// ============================================================

// ---- Term opcodes (0..13) ----
pub const OP_BOOL: u32            = 0;
pub const OP_NOT: u32             = 1;
pub const OP_AND: u32             = 2;
pub const OP_OR: u32              = 3;
pub const OP_XOR: u32             = 4;
pub const OP_ITE: u32             = 5;
pub const OP_BYTE: u32            = 6;
pub const OP_BYTE_ADD: u32        = 7;
pub const OP_BYTE_ADD_CARRY: u32  = 8;
pub const OP_BYTE_MUL_LOW: u32    = 9;
pub const OP_BYTE_MUL_HIGH: u32   = 10;
pub const OP_BYTE_AND: u32        = 11;
pub const OP_BYTE_OR: u32         = 12;
pub const OP_BYTE_XOR: u32        = 13;
// ---- Proof opcodes (14..24) ----
pub const OP_AND_INTRO: u32       = 14;
pub const OP_EQ_REFL: u32         = 15;
pub const OP_EQ_SYM: u32          = 16;
pub const OP_EQ_TRANS: u32        = 17;
pub const OP_BYTE_ADD_EQ: u32     = 18;
pub const OP_BYTE_ADD_CARRY_EQ: u32 = 19;
pub const OP_BYTE_MUL_LOW_EQ: u32 = 20;
pub const OP_BYTE_MUL_HIGH_EQ: u32 = 21;
pub const OP_BYTE_AND_EQ: u32     = 22;
pub const OP_BYTE_OR_EQ: u32      = 23;
pub const OP_BYTE_XOR_EQ: u32     = 24;
pub const OP_BYTE_ADD_THIRD_CONGRUENCE: u32 = 25;
pub const OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE: u32 = 26;
pub const OP_ITE_TRUE_EQ: u32 = 27;
pub const OP_ITE_FALSE_EQ: u32 = 28;

// ---- Return-type tags ----
pub const RET_BOOL: u32    = 0;
pub const RET_BYTE: u32    = 1;
pub const RET_WFF_EQ: u32  = 2;
pub const RET_WFF_AND: u32 = 3;

/// Number of columns in the compiled proof row (= STARK trace width).
pub const NUM_PROOF_COLS: usize = 9;

/// Flat representation of a proof / term node.
///
/// Each node in the proof tree becomes one row.
/// Rows are in **post-order**: children always appear before parents.
///
/// | Column    | Description                                              |
/// |-----------|----------------------------------------------------------|
/// | `op`      | Opcode (`OP_*` constant)                                 |
/// | `scalar0` | First immediate (byte or bool-as-int)                    |
/// | `scalar1` | Second immediate (for ByteAndEq, etc.)                   |
/// | `scalar2` | Third immediate (for ByteAddEq carry, etc.)              |
/// | `arg0`    | Row index of 1st child                                   |
/// | `arg1`    | Row index of 2nd child                                   |
/// | `arg2`    | Row index of 3rd child                                   |
/// | `value`   | Computed value (reduce result for terms; 0 for proofs)   |
/// | `ret_ty`  | Return type (`RET_BOOL`, `RET_BYTE`, `RET_WFF_*`)       |
#[derive(Clone, Debug, Default)]
pub struct ProofRow {
  pub op: u32,
  pub scalar0: u32,
  pub scalar1: u32,
  pub scalar2: u32,
  pub arg0: u32,
  pub arg1: u32,
  pub arg2: u32,
  pub value: u32,
  pub ret_ty: u32,
}

// ============================================================
// Checking Proofs
// ============================================================

pub fn infer_ty(term: &Term) -> Result<Ty, VerifyError> {
  match term {
    Term::Bool(_) => Ok(Ty::Bool),
    Term::Not(a) =>
      if infer_ty(&*a)? == Ty::Bool {
        Ok(Ty::Bool)
      } else {
        Err(VerifyError::UnexpectedTermVariant { expected: "boolean subterm" })
      },
    Term::And(a, b) |
    Term::Or(a, b) |
    Term::Xor(a, b) => {
      if infer_ty(&*a)? == Ty::Bool && infer_ty(&*b)? == Ty::Bool {
        Ok(Ty::Bool)
      } else {
        Err(VerifyError::UnexpectedTermVariant { expected: "boolean subterm" })
      }
    },
    Term::Ite(c, a, b) => {
      if infer_ty(&*c)? == Ty::Bool {
        let ty_a = infer_ty(&*a)?;
        let ty_b = infer_ty(&*b)?;
        if ty_a == ty_b {
          Ok(ty_a)
        } else {
          Err(VerifyError::UnexpectedTermVariant { expected: "matching type subterms" })
        }
      } else {
        Err(VerifyError::UnexpectedTermVariant { expected: "boolean condition subterm" })
      }
    }
    Term::Byte(_) => Ok(Ty::Byte),
    Term::ByteAdd(a, b, c) =>
      if infer_ty(&*a)? == Ty::Byte && infer_ty(&*b)? == Ty::Byte && infer_ty(&*c)? == Ty::Bool {
        Ok(Ty::Byte)
      } else {
        Err(VerifyError::UnexpectedTermVariant { expected: "byte subterm" })
      },
    Term::ByteAddCarry(a, b, c) =>
      if infer_ty(&*a)? == Ty::Byte && infer_ty(&*b)? == Ty::Byte && infer_ty(&*c)? == Ty::Bool {
        Ok(Ty::Bool)
      } else {
        Err(VerifyError::UnexpectedTermVariant { expected: "bool subterm" })
      },
    Term::ByteMulLow(a, b) |
    Term::ByteMulHigh(a, b) |
    Term::ByteAnd(a, b) |
    Term::ByteOr(a, b) |
    Term::ByteXor(a, b) => {
      if infer_ty(&*a)? == Ty::Byte && infer_ty(&*b)? == Ty::Byte {
        Ok(Ty::Byte)
      } else {
        Err(VerifyError::UnexpectedTermVariant { expected: "byte subterm" })
      }
    },
  }
}

pub fn infer_proof(proof: &Proof) -> Result<WFF, VerifyError> {
  match proof {
    Proof::AndIntro(p1, p2) => {
      let wff1 = infer_proof(p1)?;
      let wff2 = infer_proof(p2)?;
      Ok(WFF::And(Box::new(wff1), Box::new(wff2)))
    },
    Proof::EqRefl(t) => Ok(WFF::Equal(Box::new(t.clone()), Box::new(t.clone()))),
    Proof::EqSym(p) => {
      if let WFF::Equal(a, b) = infer_proof(p)? {
        Ok(WFF::Equal(b, a))
      } else {
        Err(VerifyError::UnexpectedProofVariant { expected: "equality proof" })
      }
    },
    Proof::EqTrans(p1, p2) => {
      if let WFF::Equal(a, b) = infer_proof(p1)? {
        if let WFF::Equal(b2, c) = infer_proof(p2)? {
          if *b == *b2 {
            Ok(WFF::Equal(a, c))
          } else {
            Err(VerifyError::TransitivityMismatch)
          }
        } else {
          Err(VerifyError::UnexpectedProofVariant { expected: "equality proof" })
        }
      } else {
        Err(VerifyError::UnexpectedProofVariant { expected: "equality proof" })
      }
    },
    Proof::ByteAddEq(a, b, c) => {
      let cv = *c as u16;
      let total = *a as u16 + *b as u16 + cv;
      Ok(WFF::Equal(
        Box::new(Term::ByteAdd(
          Box::new(Term::Byte(*a)), Box::new(Term::Byte(*b)), Box::new(Term::Bool(*c)))),
        Box::new(Term::Byte((total & 0xFF) as u8)),
      ))
    }
    Proof::ByteAddCarryEq(a, b, c) => {
      let cv = *c as u16;
      let total = *a as u16 + *b as u16 + cv;
      Ok(WFF::Equal(
        Box::new(Term::ByteAddCarry(
          Box::new(Term::Byte(*a)), Box::new(Term::Byte(*b)), Box::new(Term::Bool(*c)))),
        Box::new(Term::Bool(total >= 256)),
      ))
    }
    Proof::ByteMulLowEq(a, b) =>
      Ok(WFF::Equal(
        Box::new(Term::ByteMulLow(Box::new(Term::Byte(*a)), Box::new(Term::Byte(*b)))),
        Box::new(Term::Byte(((*a as u16 * *b as u16) & 0xFF) as u8)),
      )),
    Proof::ByteMulHighEq(a, b) =>
      Ok(WFF::Equal(
        Box::new(Term::ByteMulHigh(Box::new(Term::Byte(*a)), Box::new(Term::Byte(*b)))),
        Box::new(Term::Byte(((*a as u16 * *b as u16) >> 8) as u8)),
      )),
    Proof::ByteAndEq(a, b) =>
      Ok(WFF::Equal(
        Box::new(Term::ByteAnd(Box::new(Term::Byte(*a)), Box::new(Term::Byte(*b)))),
        Box::new(Term::Byte(*a & *b))
      )),
    Proof::ByteOrEq(a, b) =>
      Ok(WFF::Equal(
        Box::new(Term::ByteOr(Box::new(Term::Byte(*a)), Box::new(Term::Byte(*b)))),
        Box::new(Term::Byte(*a | *b))
      )),
    Proof::ByteXorEq(a, b) =>
      Ok(WFF::Equal(
        Box::new(Term::ByteXor(Box::new(Term::Byte(*a)), Box::new(Term::Byte(*b)))),
        Box::new(Term::Byte(*a ^ *b))
      )),
    Proof::ByteAddThirdCongruence(p, a, b) => {
      if let WFF::Equal(c1, c2) = infer_proof(p)? {
        Ok(WFF::Equal(
          Box::new(Term::ByteAdd(
            Box::new(Term::Byte(*a)),
            Box::new(Term::Byte(*b)),
            c1,
          )),
          Box::new(Term::ByteAdd(
            Box::new(Term::Byte(*a)),
            Box::new(Term::Byte(*b)),
            c2,
          )),
        ))
      } else {
        Err(VerifyError::UnexpectedProofVariant { expected: "equality proof" })
      }
    }
    Proof::ByteAddCarryThirdCongruence(p, a, b) => {
      if let WFF::Equal(c1, c2) = infer_proof(p)? {
        Ok(WFF::Equal(
          Box::new(Term::ByteAddCarry(
            Box::new(Term::Byte(*a)),
            Box::new(Term::Byte(*b)),
            c1,
          )),
          Box::new(Term::ByteAddCarry(
            Box::new(Term::Byte(*a)),
            Box::new(Term::Byte(*b)),
            c2,
          )),
        ))
      } else {
        Err(VerifyError::UnexpectedProofVariant { expected: "equality proof" })
      }
    }
    Proof::IteTrueEq(a, b) => {
      let ty_a = infer_ty(a)?;
      let ty_b = infer_ty(b)?;
      if ty_a != ty_b {
        return Err(VerifyError::UnexpectedTermVariant { expected: "matching type subterms" });
      }
      Ok(WFF::Equal(
        Box::new(Term::Ite(
          Box::new(Term::Bool(true)),
          Box::new(a.clone()),
          Box::new(b.clone()),
        )),
        Box::new(a.clone()),
      ))
    }
    Proof::IteFalseEq(a, b) => {
      let ty_a = infer_ty(a)?;
      let ty_b = infer_ty(b)?;
      if ty_a != ty_b {
        return Err(VerifyError::UnexpectedTermVariant { expected: "matching type subterms" });
      }
      Ok(WFF::Equal(
        Box::new(Term::Ite(
          Box::new(Term::Bool(false)),
          Box::new(a.clone()),
          Box::new(b.clone()),
        )),
        Box::new(b.clone()),
      ))
    }
  }
}

// ============================================================
// Word-level functions
// ============================================================

pub fn word_add(a: &[u8; 32], b: &[u8; 32]) -> [Box<Term>; 32] {
  let mut carry = Box::new(Term::Bool(false));
  // Build symbolic carry chain LSB-first (byte 31 → byte 0) for big-endian layout
  let mut terms: Vec<Box<Term>> = vec![Box::new(Term::Byte(0)); 32];
  for i in (0..32).rev() {
    let ai = Box::new(Term::Byte(a[i]));
    let bi = Box::new(Term::Byte(b[i]));
    terms[i] = Box::new(Term::ByteAdd(
      ai.clone(),
      bi.clone(),
      carry.clone(),
    ));
    carry = Box::new(Term::ByteAddCarry(ai, bi, carry));
  }
  array::from_fn(|i| terms[i].clone())
}

#[derive(Debug, Clone)]
pub struct WordAddHybridWitness {
  pub output: [u8; 32],
  pub add_steps: Vec<ByteAddWitnessStep>,
}

fn word_add_hybrid_internal(
  a: &[u8; 32],
  b: &[u8; 32],
  mut witness_steps: Option<&mut Vec<ByteAddWitnessStep>>,
) -> ([Box<Term>; 32], [u8; 32]) {
  let mut carry = false;
  let mut terms: Vec<Box<Term>> = vec![Box::new(Term::Byte(0)); 32];
  let mut out = [0u8; 32];

  for i in (0..32).rev() {
    let total = a[i] as u16 + b[i] as u16 + carry as u16;
    let sum = (total & 0xFF) as u8;
    let carry_out = total >= 256;

    let ai = Box::new(Term::Byte(a[i]));
    let bi = Box::new(Term::Byte(b[i]));
    terms[i] = Box::new(Term::ByteAdd(ai, bi, Box::new(Term::Bool(carry))));
    out[i] = sum;

    if let Some(steps) = witness_steps.as_mut() {
      steps.push(ByteAddWitnessStep {
        byte_index: i,
        lhs: a[i],
        rhs: b[i],
        carry_in: carry,
        sum,
        carry_out,
      });
    }

    carry = carry_out;
  }

  (array::from_fn(|i| terms[i].clone()), out)
}

pub fn word_add_with_hybrid_witness(a: &[u8; 32], b: &[u8; 32]) -> ([Box<Term>; 32], WordAddHybridWitness) {
  let mut steps = Vec::new();
  let (terms, output) = word_add_hybrid_internal(a, b, Some(&mut steps));
  (
    terms,
    WordAddHybridWitness {
      output,
      add_steps: steps,
    },
  )
}

fn add_u256_mod(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
  let mut out = [0u8; 32];
  let mut carry = 0u16;
  for i in (0..32).rev() {
    let total = a[i] as u16 + b[i] as u16 + carry;
    out[i] = (total & 0xFF) as u8;
    carry = total >> 8;
  }
  out
}

pub fn verify_word_add_hybrid_witness(a: &[u8; 32], b: &[u8; 32], witness: &WordAddHybridWitness) -> bool {
  if witness.add_steps.len() != 32 {
    return false;
  }

  let mut expected_idx = 31usize;
  for step in &witness.add_steps {
    if step.byte_index != expected_idx {
      return false;
    }
    if step.lhs != a[step.byte_index] || step.rhs != b[step.byte_index] {
      return false;
    }
    let total = step.lhs as u16 + step.rhs as u16 + step.carry_in as u16;
    if (total & 0xFF) as u8 != step.sum {
      return false;
    }
    if (total >= 256) != step.carry_out {
      return false;
    }

    if step.sum != witness.output[step.byte_index] {
      return false;
    }

    if expected_idx == 0 {
      break;
    }
    expected_idx -= 1;
  }

  let expected = add_u256_mod(a, b);
  witness.output == expected
}

pub fn word_mul(a: &[u8; 32], b: &[u8; 32]) -> [Box<Term>; 32] {
  word_mul_internal(a, b, None).0
}

#[derive(Debug, Clone)]
pub struct ByteAddWitnessStep {
  pub byte_index: usize,
  pub lhs: u8,
  pub rhs: u8,
  pub carry_in: bool,
  pub sum: u8,
  pub carry_out: bool,
}

#[derive(Debug, Clone)]
pub struct WordMulHybridWitness {
  pub output: [u8; 32],
  pub add_steps: Vec<ByteAddWitnessStep>,
}

fn word_mul_internal(
  a: &[u8; 32],
  b: &[u8; 32],
  mut witness_steps: Option<&mut Vec<ByteAddWitnessStep>>,
) -> ([Box<Term>; 32], [u8; 32]) {
  fn add_term_at(
    acc_terms: &mut [Option<Box<Term>>; 32],
    acc_vals: &mut [u8; 32],
    idx: usize,
    term: Box<Term>,
    term_val: u8,
    witness_steps: &mut Option<&mut Vec<ByteAddWitnessStep>>,
  ) {
    fn is_zero_byte(t: &Term) -> bool {
      matches!(t, Term::Byte(0))
    }

    fn as_byte_const(t: &Term) -> Option<u8> {
      match t {
        Term::Byte(v) => Some(*v),
        _ => None,
      }
    }

    fn as_bool_const(t: &Term) -> Option<bool> {
      match t {
        Term::Bool(v) => Some(*v),
        _ => None,
      }
    }

    fn mk_byte_add(lhs: Box<Term>, rhs: Box<Term>, carry: Box<Term>) -> Box<Term> {
      if let (Some(a), Some(b), Some(c)) = (as_byte_const(lhs.as_ref()), as_byte_const(rhs.as_ref()), as_bool_const(carry.as_ref())) {
        let total = a as u16 + b as u16 + c as u16;
        return Box::new(Term::Byte((total & 0xFF) as u8));
      }
      if matches!(carry.as_ref(), Term::Bool(false)) && is_zero_byte(rhs.as_ref()) {
        return lhs;
      }
      if matches!(carry.as_ref(), Term::Bool(false)) && is_zero_byte(lhs.as_ref()) {
        return rhs;
      }
      Box::new(Term::ByteAdd(lhs, rhs, carry))
    }

    let mut carry: u16 = 0;

    for i in (0..=idx).rev() {
      let rhs_val = if i == idx { term_val } else { 0 };
      if rhs_val == 0 && carry == 0 {
        break;
      }

      let carry_in = carry;
      let lhs_val = acc_vals[i] as u16;
      let total = lhs_val + rhs_val as u16 + carry_in;
      let sum_val = (total & 0xFF) as u8;
      carry = total >> 8;

      if let Some(steps) = witness_steps.as_mut() {
        steps.push(ByteAddWitnessStep {
          byte_index: i,
          lhs: lhs_val as u8,
          rhs: rhs_val,
          carry_in: carry_in != 0,
          sum: sum_val,
          carry_out: carry != 0,
        });
      }

      if i == idx && lhs_val == 0 && carry_in == 0 {
        acc_terms[i] = Some(term.clone());
        acc_vals[i] = term_val;
        continue;
      }

      let lhs_term = acc_terms[i].take().unwrap_or_else(|| Box::new(Term::Byte(acc_vals[i])));
      let rhs_term = if i == idx { term.clone() } else { Box::new(Term::Byte(0)) };
      let carry_term = Box::new(Term::Bool(carry_in != 0));
      let sum_term = mk_byte_add(lhs_term, rhs_term, carry_term);

      acc_terms[i] = Some(sum_term);
      acc_vals[i] = sum_val;
    }
  }

  let mut acc_terms: [Option<Box<Term>>; 32] = array::from_fn(|_| None);
  let mut acc_vals = [0u8; 32];

  // Comba-style diagonal accumulation over base-256 columns (k = i + j),
  // reduced modulo 2^256 (only columns 0..31 are materialized).
  // i,j,k are little-endian limb indices; mapped to big-endian byte positions.
  for k in 0..32 {
    for i in 0..=k {
      let j = k - i;

      let av = a[31 - i];
      let bv = b[31 - j];
      if av == 0 || bv == 0 {
        continue;
      }

      let a_term = Box::new(Term::Byte(av));
      let b_term = Box::new(Term::Byte(bv));
      let low = Box::new(Term::ByteMulLow(a_term.clone(), b_term.clone()));
      let high = Box::new(Term::ByteMulHigh(a_term, b_term));
      let prod = av as u16 * bv as u16;
      let low_val = (prod & 0xFF) as u8;
      let high_val = (prod >> 8) as u8;

      let low_be = 31 - k;
      add_term_at(&mut acc_terms, &mut acc_vals, low_be, low, low_val, &mut witness_steps);

      if k + 1 < 32 {
        let high_be = 31 - (k + 1);
        add_term_at(&mut acc_terms, &mut acc_vals, high_be, high, high_val, &mut witness_steps);
      }
    }
  }

  (
    array::from_fn(|i| acc_terms[i].clone().unwrap_or_else(|| Box::new(Term::Byte(0)))),
    acc_vals,
  )
}

pub fn word_mul_with_hybrid_witness(a: &[u8; 32], b: &[u8; 32]) -> ([Box<Term>; 32], WordMulHybridWitness) {
  let mut steps = Vec::new();
  let (terms, output) = word_mul_internal(a, b, Some(&mut steps));
  (
    terms,
    WordMulHybridWitness {
      output,
      add_steps: steps,
    },
  )
}

fn mul_u256_mod(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
  let mut limbs = [0u32; 64];
  for i in 0..32 {
    let ai = a[31 - i] as u32;
    for j in 0..32 {
      let bj = b[31 - j] as u32;
      limbs[i + j] += ai * bj;
    }
  }
  for k in 0..63 {
    let carry = limbs[k] >> 8;
    limbs[k] &= 0xFF;
    limbs[k + 1] += carry;
  }
  let mut out = [0u8; 32];
  for k in 0..32 {
    out[31 - k] = limbs[k] as u8;
  }
  out
}

pub fn verify_word_mul_hybrid_witness(a: &[u8; 32], b: &[u8; 32], witness: &WordMulHybridWitness) -> bool {
  let mut replay = [0u8; 32];
  for step in &witness.add_steps {
    if step.byte_index >= 32 {
      return false;
    }
    if replay[step.byte_index] != step.lhs {
      return false;
    }
    let total = step.lhs as u16 + step.rhs as u16 + step.carry_in as u16;
    if (total & 0xFF) as u8 != step.sum {
      return false;
    }
    if (total >= 256) != step.carry_out {
      return false;
    }
    replay[step.byte_index] = step.sum;
  }

  if replay != witness.output {
    return false;
  }

  let expected = mul_u256_mod(a, b);
  witness.output == expected
}

// ============================================================
// Core WFFs 
// ============================================================

pub fn wff_add(a: &[u8; 32], b: &[u8; 32], c: &[u8; 32]) -> WFF {
  let sums = word_add(a, b);
  let mut wff = None;
  for i in (0..32).rev() {
    let cur_wff = WFF::Equal(
      sums[i].clone(),
      Box::new(Term::Byte(c[i])),
    );

    wff = Some(match wff {
      None => cur_wff,
      Some(p) => WFF::And(Box::new(p), Box::new(cur_wff)),
    });
  }
  wff.unwrap()
}

// ============================================================
// Proving Instructions
// ============================================================

pub fn prove_add(a: &[u8; 32], b: &[u8; 32], _c: &[u8; 32]) -> Proof {
  let mut prf = None;
  let mut carry_term = Box::new(Term::Bool(false));
  let mut carry_bool = false;
  let mut carry_eq_proof = Proof::EqRefl(Term::Bool(false));
  // LSB-first: byte 31 → byte 0 (big-endian layout)
  for i in (0..32).rev() {
    let av = a[i];
    let bv = b[i];
    let total = av as u16 + bv as u16 + carry_bool as u16;

    let add_congr = Proof::ByteAddThirdCongruence(
      Box::new(carry_eq_proof.clone()),
      av,
      bv,
    );
    let add_axiom = Proof::ByteAddEq(av, bv, carry_bool);
    let cur_prf = Proof::EqTrans(Box::new(add_congr), Box::new(add_axiom));

    let carry_congr = Proof::ByteAddCarryThirdCongruence(
      Box::new(carry_eq_proof),
      av,
      bv,
    );
    let carry_axiom = Proof::ByteAddCarryEq(av, bv, carry_bool);
    carry_eq_proof = Proof::EqTrans(Box::new(carry_congr), Box::new(carry_axiom));

    let ai = Box::new(Term::Byte(av));
    let bi = Box::new(Term::Byte(bv));
    carry_term = Box::new(Term::ByteAddCarry(ai, bi, carry_term));
    carry_bool = total >= 256;

    prf = Some(match prf {
      None => cur_prf,
      Some(p) => Proof::AndIntro(Box::new(p), Box::new(cur_prf)),
    });
  }
  prf.expect("prove_add must produce 32-byte conjunction proof")
}

// ============================================================
// Compiling: Tree → Vec<ProofRow>
// ============================================================

fn compile_term_inner(term: &Term, rows: &mut Vec<ProofRow>) -> u32 {
  match term {
    Term::Bool(v) => {
      let idx = rows.len() as u32;
      let val = *v as u32;
      rows.push(ProofRow { op: OP_BOOL, scalar0: val, value: val, ret_ty: RET_BOOL, ..Default::default() });
      idx
    }
    Term::Not(a) => {
      let ai = compile_term_inner(a, rows);
      let val = 1 - rows[ai as usize].value;
      let idx = rows.len() as u32;
      rows.push(ProofRow { op: OP_NOT, arg0: ai, value: val, ret_ty: RET_BOOL, ..Default::default() });
      idx
    }
    Term::And(a, b) => {
      let ai = compile_term_inner(a, rows);
      let bi = compile_term_inner(b, rows);
      let val = rows[ai as usize].value * rows[bi as usize].value;
      let idx = rows.len() as u32;
      rows.push(ProofRow { op: OP_AND, arg0: ai, arg1: bi, value: val, ret_ty: RET_BOOL, ..Default::default() });
      idx
    }
    Term::Or(a, b) => {
      let ai = compile_term_inner(a, rows);
      let bi = compile_term_inner(b, rows);
      let (av, bv) = (rows[ai as usize].value, rows[bi as usize].value);
      let val = av + bv - av * bv;
      let idx = rows.len() as u32;
      rows.push(ProofRow { op: OP_OR, arg0: ai, arg1: bi, value: val, ret_ty: RET_BOOL, ..Default::default() });
      idx
    }
    Term::Xor(a, b) => {
      let ai = compile_term_inner(a, rows);
      let bi = compile_term_inner(b, rows);
      let (av, bv) = (rows[ai as usize].value, rows[bi as usize].value);
      let val = av + bv - 2 * av * bv;
      let idx = rows.len() as u32;
      rows.push(ProofRow { op: OP_XOR, arg0: ai, arg1: bi, value: val, ret_ty: RET_BOOL, ..Default::default() });
      idx
    }
    Term::Ite(c, a, b) => {
      let ci = compile_term_inner(c, rows);
      let ai = compile_term_inner(a, rows);
      let bi = compile_term_inner(b, rows);
      let cv = rows[ci as usize].value;
      let (av, bv) = (rows[ai as usize].value, rows[bi as usize].value);
      let val = cv * av + (1 - cv) * bv;
      let ret = rows[ai as usize].ret_ty; // branches have same type
      let idx = rows.len() as u32;
      rows.push(ProofRow { op: OP_ITE, arg0: ci, arg1: ai, arg2: bi, value: val, ret_ty: ret, ..Default::default() });
      idx
    }
    Term::Byte(v) => {
      let idx = rows.len() as u32;
      rows.push(ProofRow { op: OP_BYTE, scalar0: *v as u32, value: *v as u32, ret_ty: RET_BYTE, ..Default::default() });
      idx
    }
    Term::ByteAdd(a, b, c) => {
      let ai = compile_term_inner(a, rows);
      let bi = compile_term_inner(b, rows);
      let ci = compile_term_inner(c, rows);
      let total = rows[ai as usize].value + rows[bi as usize].value + rows[ci as usize].value;
      let val = total & 0xFF;
      let idx = rows.len() as u32;
      rows.push(ProofRow { op: OP_BYTE_ADD, arg0: ai, arg1: bi, arg2: ci, value: val, ret_ty: RET_BYTE, ..Default::default() });
      idx
    }
    Term::ByteAddCarry(a, b, c) => {
      let ai = compile_term_inner(a, rows);
      let bi = compile_term_inner(b, rows);
      let ci = compile_term_inner(c, rows);
      let total = rows[ai as usize].value + rows[bi as usize].value + rows[ci as usize].value;
      let val = if total >= 256 { 1 } else { 0 };
      let idx = rows.len() as u32;
      rows.push(ProofRow { op: OP_BYTE_ADD_CARRY, arg0: ai, arg1: bi, arg2: ci, value: val, ret_ty: RET_BOOL, ..Default::default() });
      idx
    }
    Term::ByteMulLow(a, b) => {
      let ai = compile_term_inner(a, rows);
      let bi = compile_term_inner(b, rows);
      let val = (rows[ai as usize].value * rows[bi as usize].value) & 0xFF;
      let idx = rows.len() as u32;
      rows.push(ProofRow { op: OP_BYTE_MUL_LOW, arg0: ai, arg1: bi, value: val, ret_ty: RET_BYTE, ..Default::default() });
      idx
    }
    Term::ByteMulHigh(a, b) => {
      let ai = compile_term_inner(a, rows);
      let bi = compile_term_inner(b, rows);
      let val = (rows[ai as usize].value * rows[bi as usize].value) >> 8;
      let idx = rows.len() as u32;
      rows.push(ProofRow { op: OP_BYTE_MUL_HIGH, arg0: ai, arg1: bi, value: val, ret_ty: RET_BYTE, ..Default::default() });
      idx
    }
    Term::ByteAnd(a, b) => {
      let ai = compile_term_inner(a, rows);
      let bi = compile_term_inner(b, rows);
      let val = rows[ai as usize].value & rows[bi as usize].value;
      let idx = rows.len() as u32;
      rows.push(ProofRow { op: OP_BYTE_AND, arg0: ai, arg1: bi, value: val, ret_ty: RET_BYTE, ..Default::default() });
      idx
    }
    Term::ByteOr(a, b) => {
      let ai = compile_term_inner(a, rows);
      let bi = compile_term_inner(b, rows);
      let val = rows[ai as usize].value | rows[bi as usize].value;
      let idx = rows.len() as u32;
      rows.push(ProofRow { op: OP_BYTE_OR, arg0: ai, arg1: bi, value: val, ret_ty: RET_BYTE, ..Default::default() });
      idx
    }
    Term::ByteXor(a, b) => {
      let ai = compile_term_inner(a, rows);
      let bi = compile_term_inner(b, rows);
      let val = rows[ai as usize].value ^ rows[bi as usize].value;
      let idx = rows.len() as u32;
      rows.push(ProofRow { op: OP_BYTE_XOR, arg0: ai, arg1: bi, value: val, ret_ty: RET_BYTE, ..Default::default() });
      idx
    }
  }
}

fn compile_proof_inner(proof: &Proof, rows: &mut Vec<ProofRow>) -> u32 {
  match proof {
    Proof::AndIntro(p1, p2) => {
      let p1i = compile_proof_inner(p1, rows);
      let p2i = compile_proof_inner(p2, rows);
      let idx = rows.len() as u32;
      rows.push(ProofRow { op: OP_AND_INTRO, arg0: p1i, arg1: p2i, ret_ty: RET_WFF_AND, ..Default::default() });
      idx
    }
    Proof::EqRefl(t) => {
      let ti = compile_term_inner(t, rows);
      let idx = rows.len() as u32;
      rows.push(ProofRow { op: OP_EQ_REFL, arg0: ti, ret_ty: RET_WFF_EQ, ..Default::default() });
      idx
    }
    Proof::EqSym(p) => {
      let pi = compile_proof_inner(p, rows);
      let idx = rows.len() as u32;
      rows.push(ProofRow { op: OP_EQ_SYM, arg0: pi, ret_ty: RET_WFF_EQ, ..Default::default() });
      idx
    }
    Proof::EqTrans(p1, p2) => {
      let p1i = compile_proof_inner(p1, rows);
      let p2i = compile_proof_inner(p2, rows);
      let idx = rows.len() as u32;
      rows.push(ProofRow { op: OP_EQ_TRANS, arg0: p1i, arg1: p2i, ret_ty: RET_WFF_EQ, ..Default::default() });
      idx
    }
    Proof::ByteAddEq(a, b, c) => {
      let carry_in = *c as u32;
      let total = *a as u32 + *b as u32 + carry_in;
      let sum = total & 0xFF;
      let carry_out = if total >= 256 { 1 } else { 0 };
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_BYTE_ADD_EQ,
        scalar0: *a as u32,
        scalar1: *b as u32,
        scalar2: carry_out,
        arg0: carry_in,
        arg1: sum,
        value: sum,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      });
      idx
    }
    Proof::ByteAddCarryEq(a, b, c) => {
      let idx = rows.len() as u32;
      rows.push(ProofRow { op: OP_BYTE_ADD_CARRY_EQ, scalar0: *a as u32, scalar1: *b as u32, arg0: *c as u32, ret_ty: RET_WFF_EQ, ..Default::default() });
      idx
    }
    Proof::ByteMulLowEq(a, b) => {
      let idx = rows.len() as u32;
      rows.push(ProofRow { op: OP_BYTE_MUL_LOW_EQ, scalar0: *a as u32, scalar1: *b as u32, ret_ty: RET_WFF_EQ, ..Default::default() });
      idx
    }
    Proof::ByteMulHighEq(a, b) => {
      let idx = rows.len() as u32;
      rows.push(ProofRow { op: OP_BYTE_MUL_HIGH_EQ, scalar0: *a as u32, scalar1: *b as u32, ret_ty: RET_WFF_EQ, ..Default::default() });
      idx
    }
    Proof::ByteAndEq(a, b) => {
      let idx = rows.len() as u32;
      rows.push(ProofRow { op: OP_BYTE_AND_EQ, scalar0: *a as u32, scalar1: *b as u32, ret_ty: RET_WFF_EQ, ..Default::default() });
      idx
    }
    Proof::ByteOrEq(a, b) => {
      let idx = rows.len() as u32;
      rows.push(ProofRow { op: OP_BYTE_OR_EQ, scalar0: *a as u32, scalar1: *b as u32, ret_ty: RET_WFF_EQ, ..Default::default() });
      idx
    }
    Proof::ByteXorEq(a, b) => {
      let idx = rows.len() as u32;
      rows.push(ProofRow { op: OP_BYTE_XOR_EQ, scalar0: *a as u32, scalar1: *b as u32, ret_ty: RET_WFF_EQ, ..Default::default() });
      idx
    }
    Proof::ByteAddThirdCongruence(p, a, b) => {
      let pi = compile_proof_inner(p, rows);
      let idx = rows.len() as u32;
      rows.push(ProofRow { op: OP_BYTE_ADD_THIRD_CONGRUENCE, scalar0: *a as u32, scalar1: *b as u32, arg0: pi, ret_ty: RET_WFF_EQ, ..Default::default() });
      idx
    }
    Proof::ByteAddCarryThirdCongruence(p, a, b) => {
      let pi = compile_proof_inner(p, rows);
      let idx = rows.len() as u32;
      rows.push(ProofRow { op: OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE, scalar0: *a as u32, scalar1: *b as u32, arg0: pi, ret_ty: RET_WFF_EQ, ..Default::default() });
      idx
    }
    Proof::IteTrueEq(a, b) => {
      let ai = compile_term_inner(a, rows);
      let bi = compile_term_inner(b, rows);
      let idx = rows.len() as u32;
      rows.push(ProofRow { op: OP_ITE_TRUE_EQ, arg0: ai, arg1: bi, ret_ty: RET_WFF_EQ, ..Default::default() });
      idx
    }
    Proof::IteFalseEq(a, b) => {
      let ai = compile_term_inner(a, rows);
      let bi = compile_term_inner(b, rows);
      let idx = rows.len() as u32;
      rows.push(ProofRow { op: OP_ITE_FALSE_EQ, arg0: ai, arg1: bi, ret_ty: RET_WFF_EQ, ..Default::default() });
      idx
    }
  }
}

/// Flatten a [`Proof`] tree into a `Vec<ProofRow>` (post-order).
pub fn compile_proof(proof: &Proof) -> Vec<ProofRow> {
  let mut rows = Vec::new();
  compile_proof_inner(proof, &mut rows);
  rows
}

/// Flatten a [`Term`] tree into a `Vec<ProofRow>` (post-order).
pub fn compile_term(term: &Term) -> Vec<ProofRow> {
  let mut rows = Vec::new();
  compile_term_inner(term, &mut rows);
  rows
}

// ============================================================
// Flat verification (walk the compiled array)
// ============================================================

/// Verify a compiled proof row-by-row.
///
/// Checks term values are correct and proof nodes are well-formed.
/// Returns `Ok(())` if every row passes, otherwise the first error.
pub fn verify_compiled(rows: &[ProofRow]) -> Result<(), VerifyError> {
  for (i, row) in rows.iter().enumerate() {
    // Helper: fetch a child row with bounds check.
    let arg = |idx: u32| -> Result<&ProofRow, VerifyError> {
      let j = idx as usize;
      if j >= i {
        return Err(VerifyError::UnexpectedTermVariant { expected: "valid child index" });
      }
      Ok(&rows[j])
    };

    match row.op {
      // ── Leaf terms ──
      OP_BOOL => {
        if row.value != row.scalar0 || row.value > 1 {
          return Err(VerifyError::ByteDecideFailed);
        }
      }
      OP_BYTE => {
        if row.value != row.scalar0 || row.value > 255 {
          return Err(VerifyError::ByteDecideFailed);
        }
      }
      // ── Unary bool ──
      OP_NOT => {
        let a = arg(row.arg0)?;
        if row.value != 1 - a.value {
          return Err(VerifyError::ByteDecideFailed);
        }
      }
      // ── Binary bool ──
      OP_AND => {
        let (a, b) = (arg(row.arg0)?, arg(row.arg1)?);
        if row.value != a.value * b.value {
          return Err(VerifyError::ByteDecideFailed);
        }
      }
      OP_OR => {
        let (a, b) = (arg(row.arg0)?, arg(row.arg1)?);
        if row.value != a.value + b.value - a.value * b.value {
          return Err(VerifyError::ByteDecideFailed);
        }
      }
      OP_XOR => {
        let (a, b) = (arg(row.arg0)?, arg(row.arg1)?);
        if row.value != a.value + b.value - 2 * a.value * b.value {
          return Err(VerifyError::ByteDecideFailed);
        }
      }
      // ── Ternary: if-then-else ──
      OP_ITE => {
        let (c, a, b) = (arg(row.arg0)?, arg(row.arg1)?, arg(row.arg2)?);
        if row.value != c.value * a.value + (1 - c.value) * b.value {
          return Err(VerifyError::ByteDecideFailed);
        }
      }
      // ── Byte arithmetic ──
      OP_BYTE_ADD => {
        let (a, b, c) = (arg(row.arg0)?, arg(row.arg1)?, arg(row.arg2)?);
        if row.value != (a.value + b.value + c.value) & 0xFF {
          return Err(VerifyError::ByteDecideFailed);
        }
      }
      OP_BYTE_ADD_CARRY => {
        let (a, b, c) = (arg(row.arg0)?, arg(row.arg1)?, arg(row.arg2)?);
        let exp = if a.value + b.value + c.value >= 256 { 1 } else { 0 };
        if row.value != exp {
          return Err(VerifyError::ByteDecideFailed);
        }
      }
      OP_BYTE_MUL_LOW => {
        let (a, b) = (arg(row.arg0)?, arg(row.arg1)?);
        if row.value != (a.value * b.value) & 0xFF {
          return Err(VerifyError::ByteDecideFailed);
        }
      }
      OP_BYTE_MUL_HIGH => {
        let (a, b) = (arg(row.arg0)?, arg(row.arg1)?);
        if row.value != (a.value * b.value) >> 8 {
          return Err(VerifyError::ByteDecideFailed);
        }
      }
      OP_BYTE_AND => {
        let (a, b) = (arg(row.arg0)?, arg(row.arg1)?);
        if row.value != (a.value & b.value) {
          return Err(VerifyError::ByteDecideFailed);
        }
      }
      OP_BYTE_OR => {
        let (a, b) = (arg(row.arg0)?, arg(row.arg1)?);
        if row.value != (a.value | b.value) {
          return Err(VerifyError::ByteDecideFailed);
        }
      }
      OP_BYTE_XOR => {
        let (a, b) = (arg(row.arg0)?, arg(row.arg1)?);
        if row.value != (a.value ^ b.value) {
          return Err(VerifyError::ByteDecideFailed);
        }
      }
      // ── Proof: byte-add axiom ──
      OP_BYTE_ADD_EQ => {
        if row.scalar0 > 255 || row.scalar1 > 255 || row.arg0 > 1 {
          return Err(VerifyError::ByteDecideFailed);
        }
      }
      OP_BYTE_ADD_CARRY_EQ => {
        if row.scalar0 > 255 || row.scalar1 > 255 || row.arg0 > 1 {
          return Err(VerifyError::ByteDecideFailed);
        }
      }
      OP_BYTE_MUL_LOW_EQ | OP_BYTE_MUL_HIGH_EQ => {
        if row.scalar0 > 255 || row.scalar1 > 255 {
          return Err(VerifyError::ByteDecideFailed);
        }
      }
      // ── Proof: conjunction ──
      OP_AND_INTRO => {
        let (a, b) = (arg(row.arg0)?, arg(row.arg1)?);
        if a.ret_ty < RET_WFF_EQ || b.ret_ty < RET_WFF_EQ {
          return Err(VerifyError::UnexpectedProofVariant { expected: "proof arguments" });
        }
      }
      // ── Proof: reflexivity ──
      OP_EQ_REFL => {
        let a = arg(row.arg0)?;
        if a.ret_ty > RET_BYTE {
          return Err(VerifyError::UnexpectedProofVariant { expected: "term argument" });
        }
      }
      // ── Proof: symmetry ──
      OP_EQ_SYM => {
        let a = arg(row.arg0)?;
        if a.ret_ty != RET_WFF_EQ {
          return Err(VerifyError::UnexpectedProofVariant { expected: "equality proof" });
        }
      }
      // ── Proof: transitivity ──
      OP_EQ_TRANS => {
        let (a, b) = (arg(row.arg0)?, arg(row.arg1)?);
        if a.ret_ty != RET_WFF_EQ || b.ret_ty != RET_WFF_EQ {
          return Err(VerifyError::UnexpectedProofVariant { expected: "equality proofs" });
        }
      }
      // ── Proof: byte bitwise axioms (lookup-verifiable) ──
      OP_BYTE_AND_EQ | OP_BYTE_OR_EQ | OP_BYTE_XOR_EQ => {
        if row.scalar0 > 255 || row.scalar1 > 255 {
          return Err(VerifyError::ByteDecideFailed);
        }
      }
      OP_BYTE_ADD_THIRD_CONGRUENCE | OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE => {
        if row.scalar0 > 255 || row.scalar1 > 255 {
          return Err(VerifyError::ByteDecideFailed);
        }
        let p = arg(row.arg0)?;
        if p.ret_ty != RET_WFF_EQ {
          return Err(VerifyError::UnexpectedProofVariant { expected: "equality proof" });
        }
      }
      OP_ITE_TRUE_EQ | OP_ITE_FALSE_EQ => {
        let (a, b) = (arg(row.arg0)?, arg(row.arg1)?);
        if a.ret_ty > RET_BYTE || b.ret_ty > RET_BYTE || a.ret_ty != b.ret_ty {
          return Err(VerifyError::UnexpectedProofVariant { expected: "matching term arguments" });
        }
      }
      _ => return Err(VerifyError::UnexpectedTermVariant { expected: "valid opcode" }),
    }
  }
  Ok(())
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
  use super::*;

  fn term_node_count(term: &Term) -> usize {
    match term {
      Term::Bool(_) | Term::Byte(_) => 1,
      Term::Not(a) => 1 + term_node_count(a),
      Term::And(a, b)
      | Term::Or(a, b)
      | Term::Xor(a, b)
      | Term::ByteMulLow(a, b)
      | Term::ByteMulHigh(a, b)
      | Term::ByteAnd(a, b)
      | Term::ByteOr(a, b)
      | Term::ByteXor(a, b) => 1 + term_node_count(a) + term_node_count(b),
      Term::Ite(c, a, b)
      | Term::ByteAdd(a, b, c)
      | Term::ByteAddCarry(a, b, c) => 1 + term_node_count(c) + term_node_count(a) + term_node_count(b),
    }
  }

  fn word_term_sizes(terms: &[Box<Term>; 32]) -> [usize; 32] {
    array::from_fn(|i| term_node_count(terms[i].as_ref()))
  }

  #[test]
  fn test_compile_and_verify_simple_add() {
    let mut a = [0u8; 32];
    let mut b = [0u8; 32];
    a[31] = 100;
    b[31] = 50;
    let mut c = [0u8; 32];
    c[31] = 150;

    let proof = prove_add(&a, &b, &c);
    let rows = compile_proof(&proof);
    verify_compiled(&rows).expect("compiled verification should pass");
  }

  #[test]
  fn test_compile_and_verify_add_with_carry() {
    let mut a = [0u8; 32];
    let mut b = [0u8; 32];
    a[31] = 200;
    b[31] = 100;
    let mut c = [0u8; 32];
    c[31] = 44;
    c[30] = 1;

    let proof = prove_add(&a, &b, &c);
    let rows = compile_proof(&proof);
    verify_compiled(&rows).expect("compiled verification should pass");
  }

  #[test]
  fn test_compile_row_count() {
    let a = [0u8; 32];
    let b = [0u8; 32];
    let c = [0u8; 32];

    let proof = prove_add(&a, &b, &c);
    let rows = compile_proof(&proof);
    assert!(rows.len() > 32, "need at least one row per byte + overhead");
    eprintln!("zero-add row count: {}", rows.len());
  }

  #[test]
  fn test_compiled_corrupted_value_fails() {
    let mut a = [0u8; 32];
    let mut b = [0u8; 32];
    a[31] = 10;
    b[31] = 20;
    let mut c = [0u8; 32];
    c[31] = 30;

    let proof = prove_add(&a, &b, &c);
    let mut rows = compile_proof(&proof);

    // Corrupt the first ByteAddEq axiom's scalar → range check fails.
    for r in rows.iter_mut() {
      if r.op == OP_BYTE_ADD_EQ {
        r.scalar0 = 300; // > 255, triggers range check error
        break;
      }
    }
    assert!(verify_compiled(&rows).is_err(), "corrupted value should fail");
  }

  #[test]
  fn test_compile_term_standalone() {
    let term = Term::ByteAdd(
      Box::new(Term::Byte(200)),
      Box::new(Term::Byte(100)),
      Box::new(Term::Bool(false)),
    );
    let rows = compile_term(&term);
    assert_eq!(rows.last().unwrap().value, 44); // (200+100+0) & 0xFF
    assert_eq!(rows.last().unwrap().ret_ty, RET_BYTE);
    verify_compiled(&rows).unwrap();
  }

  #[test]
  fn test_compile_bool_ops() {
    let term = Term::Xor(
      Box::new(Term::Bool(true)),
      Box::new(Term::Bool(false)),
    );
    let rows = compile_term(&term);
    assert_eq!(rows.last().unwrap().value, 1);
    assert_eq!(rows.last().unwrap().ret_ty, RET_BOOL);
    verify_compiled(&rows).unwrap();
  }

  #[test]
  fn test_all_bytes_max() {
    let a = [0xFF; 32];
    let mut b = [0u8; 32];
    b[31] = 1;
    // 0xFF..FF + 1 = 0x00..00 (mod 2^256)
    let mut c = [0u8; 32];
    c[0] = 0; // MSB wraps
    // compute expected
    let mut carry = 0u16;
    for i in (0..32).rev() {
      let sum = a[i] as u16 + b[i] as u16 + carry;
      c[i] = (sum & 0xFF) as u8;
      carry = sum >> 8;
    }

    let proof = prove_add(&a, &b, &c);
    let rows = compile_proof(&proof);
    verify_compiled(&rows).unwrap();
  }

  #[test]
  fn test_compile_and_verify_ite_axioms() {
    let p_true = Proof::IteTrueEq(Term::Byte(7), Term::Byte(99));
    let w_true = infer_proof(&p_true).expect("infer_proof ite true should succeed");
    match w_true {
      WFF::Equal(lhs, rhs) => {
        assert_eq!(*rhs, Term::Byte(7));
        match *lhs {
          Term::Ite(c, _, _) => assert_eq!(*c, Term::Bool(true)),
          _ => panic!("lhs should be ite"),
        }
      }
      _ => panic!("expected equality"),
    }
    let rows_true = compile_proof(&p_true);
    verify_compiled(&rows_true).expect("compiled verification should pass for ite true axiom");

    let p_false = Proof::IteFalseEq(Term::Byte(7), Term::Byte(99));
    let w_false = infer_proof(&p_false).expect("infer_proof ite false should succeed");
    match w_false {
      WFF::Equal(lhs, rhs) => {
        assert_eq!(*rhs, Term::Byte(99));
        match *lhs {
          Term::Ite(c, _, _) => assert_eq!(*c, Term::Bool(false)),
          _ => panic!("lhs should be ite"),
        }
      }
      _ => panic!("expected equality"),
    }
    let rows_false = compile_proof(&p_false);
    verify_compiled(&rows_false).expect("compiled verification should pass for ite false axiom");
  }

  #[test]
  fn test_word_mul_hybrid_witness_verifies() {
    let mut a = [0u8; 32];
    let mut b = [0u8; 32];
    a[31] = 123;
    b[31] = 45;

    let (_terms, witness) = word_mul_with_hybrid_witness(&a, &b);
    assert!(verify_word_mul_hybrid_witness(&a, &b, &witness));
  }

  #[test]
  fn test_word_mul_hybrid_witness_tamper_detected() {
    let mut a = [0u8; 32];
    let mut b = [0u8; 32];
    a[31] = 200;
    b[31] = 100;

    let (_terms, mut witness) = word_mul_with_hybrid_witness(&a, &b);
    if let Some(first) = witness.add_steps.first_mut() {
      first.sum ^= 1;
    }
    assert!(!verify_word_mul_hybrid_witness(&a, &b, &witness));
  }

  #[test]
  fn test_word_add_hybrid_witness_verifies() {
    let mut a = [0u8; 32];
    let mut b = [0u8; 32];
    a[31] = 200;
    b[31] = 100;

    let (_terms, witness) = word_add_with_hybrid_witness(&a, &b);
    assert!(verify_word_add_hybrid_witness(&a, &b, &witness));
  }

  #[test]
  fn test_word_add_hybrid_witness_tamper_detected() {
    let mut a = [0u8; 32];
    let mut b = [0u8; 32];
    a[31] = 77;
    b[31] = 99;

    let (_terms, mut witness) = word_add_with_hybrid_witness(&a, &b);
    if let Some(first) = witness.add_steps.first_mut() {
      first.carry_out = !first.carry_out;
    }
    assert!(!verify_word_add_hybrid_witness(&a, &b, &witness));
  }

  #[test]
  #[ignore = "diagnostic size measurement; run explicitly with -- --ignored --nocapture"]
  fn test_measure_word_mul_term_size() {
    let mut sparse_a = [0u8; 32];
    let mut sparse_b = [0u8; 32];
    sparse_a[31] = 100;
    sparse_b[31] = 200;

    let mut dense_a = [0u8; 32];
    let mut dense_b = [0u8; 32];
    for i in 30..32 {
      dense_a[i] = 0xFF;
      dense_b[i] = 0xFF;
    }

    let sparse_terms = word_mul(&sparse_a, &sparse_b);
    let sparse_sizes = word_term_sizes(&sparse_terms);
    let sparse_total: usize = sparse_sizes.iter().sum();

    let dense_terms = word_mul(&dense_a, &dense_b);
    let dense_sizes = word_term_sizes(&dense_terms);
    let dense_total: usize = dense_sizes.iter().sum();

    eprintln!("word_mul term-size sparse total: {}", sparse_total);
    eprintln!("word_mul term-size sparse per-byte: {:?}", sparse_sizes);
    eprintln!("word_mul term-size dense total: {}", dense_total);
    eprintln!("word_mul term-size dense per-byte: {:?}", dense_sizes);

    assert!(sparse_total > 0);
    assert!(dense_total >= sparse_total);
  }
}

