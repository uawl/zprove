
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

// ---- Return-type tags ----
pub const RET_BOOL: u32    = 0;
pub const RET_BYTE: u32    = 1;
pub const RET_WFF_EQ: u32  = 2;
pub const RET_WFF_AND: u32 = 3;

/// Number of columns in the compiled proof row (= STARK trace width).
pub const NUM_PROOF_COLS: usize = 8;

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

pub fn reduce_byte(term: &Term) -> Result<u8, VerifyError> {
  match term {
    Term::Byte(b) => Ok(*b),
    Term::ByteAdd(a, b, c) => {
      let av = reduce_byte(&*a)?;
      let bv = reduce_byte(&*b)?;
      let cv = if reduce_bool(&*c)? { 1u16 } else { 0 };
      Ok(((av as u16 + bv as u16 + cv) & 0xFF) as u8)
    },
    Term::ByteMulHigh(a, b) => {
      let av = reduce_byte(&*a)?;
      let bv = reduce_byte(&*b)?;
      Ok(((av as u16 * bv as u16) >> 8) as u8)
    },
    Term::ByteMulLow(a, b) => {
      let av = reduce_byte(&*a)?;
      let bv = reduce_byte(&*b)?;
      Ok(((av as u16 * bv as u16) & 0xFF) as u8)
    },
    Term::Ite(c, a, b) => {
      let cv = reduce_bool(&*c)?;
      if cv {
        reduce_byte(&*a)
      } else {
        reduce_byte(&*b)
      }
    }
    _ => Err(VerifyError::UnexpectedTermVariant { expected: "irreducible term" })
  }
}

pub fn reduce_bool(term: &Term) -> Result<bool, VerifyError> {
  match term {
    Term::Bool(b) => Ok(*b),
    Term::Not(a) => Ok(!reduce_bool(&*a)?),
    Term::And(a, b) => Ok(reduce_bool(&*a)? && reduce_bool(&*b)?),
    Term::Or(a, b) => Ok(reduce_bool(&*a)? || reduce_bool(&*b)?),
    Term::Xor(a, b) => Ok(reduce_bool(&*a)? != reduce_bool(&*b)?),
    Term::Ite(c, a, b) => {
      let cv = reduce_bool(&*c)?;
      if cv {
        reduce_bool(&*a)
      } else {
        reduce_bool(&*b)
      }
    }
    Term::ByteAddCarry(a, b, c) => {
      let av = reduce_byte(&*a)? as u16;
      let bv = reduce_byte(&*b)? as u16;
      let cv = if reduce_bool(&*c)? { 1u16 } else { 0 };
      Ok((av + bv + cv) >= 256)
    }
    _ => Err(VerifyError::UnexpectedTermVariant { expected: "irreducible term" })
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
  }
}

// ============================================================
// Word-level functions
// ============================================================

pub fn word_add(a: &[u8; 32], b: &[u8; 32]) -> [Box<Term>; 32] {
  let mut carry = Box::new(Term::Bool(false));
  // Build the carry chain LSB-first (byte 31 → byte 0) for big-endian layout
  let mut terms: Vec<Box<Term>> = vec![Box::new(Term::Byte(0)); 32];
  for i in (0..32).rev() {
    let ai = Box::new(Term::Byte(a[i]));
    let bi = Box::new(Term::Byte(b[i]));
    terms[i] = Box::new(Term::ByteAdd(ai.clone(), bi.clone(), carry.clone()));
    carry = Box::new(Term::ByteAddCarry(ai, bi, carry));
  }
  array::from_fn(|i| terms[i].clone())
}

// ============================================================
// Core WFFs 
// ============================================================

pub fn wff_add(a: &[u8; 32], b: &[u8; 32], c: &[u8; 32]) -> WFF {
  let mut wff = None;
  let mut carry = false;
  for i in (0..32).rev() {
    let av = a[i];
    let bv = b[i];
    let total = av as u16 + bv as u16 + carry as u16;
    // Use the *claimed* output byte c[i], not the computed sum.
    // If c is correct, this matches the proof's WFF;
    // if c is wrong, verification fails because WFFs diverge.
    let cur_wff = WFF::Equal(
      Box::new(Term::ByteAdd(
        Box::new(Term::Byte(av)),
        Box::new(Term::Byte(bv)),
        Box::new(Term::Bool(carry)),
      )),
      Box::new(Term::Byte(c[i])),
    );
    carry = total >= 256;
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

pub fn prove_add(a: &[u8; 32], b: &[u8; 32], _c: &[u8; 32]) -> Option<Proof> {
  let mut prf = None;
  let mut carry = false;
  // LSB-first: byte 31 → byte 0 (big-endian layout)
  for i in (0..32).rev() {
    let av = a[i];
    let bv = b[i];
    let total = av as u16 + bv as u16 + carry as u16;
    let cur_prf = Proof::ByteAddEq(av, bv, carry);
    carry = total >= 256;
    prf = Some(match prf {
      None => cur_prf,
      Some(p) => Proof::AndIntro(Box::new(p), Box::new(cur_prf)),
    });
  }
  prf
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
      let idx = rows.len() as u32;
      rows.push(ProofRow { op: OP_BYTE_ADD_EQ, scalar0: *a as u32, scalar1: *b as u32, arg0: *c as u32, ret_ty: RET_WFF_EQ, ..Default::default() });
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

  #[test]
  fn test_compile_and_verify_simple_add() {
    let mut a = [0u8; 32];
    let mut b = [0u8; 32];
    a[31] = 100;
    b[31] = 50;
    let mut c = [0u8; 32];
    c[31] = 150;

    let proof = prove_add(&a, &b, &c).unwrap();
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

    let proof = prove_add(&a, &b, &c).unwrap();
    let rows = compile_proof(&proof);
    verify_compiled(&rows).expect("compiled verification should pass");
  }

  #[test]
  fn test_compile_row_count() {
    let a = [0u8; 32];
    let b = [0u8; 32];
    let c = [0u8; 32];

    let proof = prove_add(&a, &b, &c).unwrap();
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

    let proof = prove_add(&a, &b, &c).unwrap();
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

    let proof = prove_add(&a, &b, &c).unwrap();
    let rows = compile_proof(&proof);
    verify_compiled(&rows).unwrap();
  }
}

