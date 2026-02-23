//! Hilbert-style proof system for EVM instruction semantics.
//!
//! # Architecture
//!
//! The proof system operates at two levels:
//! - **Byte level**: Axioms for 8-bit arithmetic (lookup-table verifiable in O(1))
//! - **Word level**: 256-bit operations composed from 32 byte-level sub-proofs
//!
//! Each [`Proof`] term is a deduction tree of axioms and inference rules.
//! The [`verify`] function traverses this tree, checks every step, and returns
//! the proven equation `(lhs, rhs)` such that `lhs = rhs`.
//!
//! # ZKP Integration
//!
//! These proofs serve as witnesses for ZKP circuits:
//! - Byte axioms → lookup table constraints (2^8 entries)
//! - Carry chains → sequential arithmetic constraints
//! - Word construction → byte packing (free in circuit)
//! - The verifier logic maps directly to a simple arithmetic circuit

use std::fmt;

// ============================================================
// Types
// ============================================================

/// Types in the proof system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ty {
  Byte,
  Word,
}

// ============================================================
// Terms
// ============================================================

/// Terms represent values and computations in the proof language.
///
/// Byte-level terms are the atoms; word-level terms compose 32 bytes.
/// Abstract operation terms (`WordAdd`, `WordSub`, …) represent EVM
/// operations whose semantics are established through proof rules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Term {
  // ---- Byte-level ----
  /// Concrete byte value.
  Byte(u8),
  /// `ByteAdd(a, b, c) = (a + b + c) mod 256` where c ∈ {0,1}.
  ByteAdd(Box<Term>, Box<Term>, Box<Term>),
  /// `ByteAddCarry(a, b, c) = (a + b + c) / 256` — carry output (0 or 1).
  ByteAddCarry(Box<Term>, Box<Term>, Box<Term>),
  /// `ByteNeg(a) = !a` — bitwise NOT.
  ByteNeg(Box<Term>),
  /// `ByteMulLow(a, b) = (a * b) mod 256`.
  ByteMulLow(Box<Term>, Box<Term>),
  /// `ByteMulHigh(a, b) = (a * b) / 256`.
  ByteMulHigh(Box<Term>, Box<Term>),
  /// `ByteInv(a)` = modular multiplicative inverse of byte.
  ByteInv(Box<Term>),
  /// `ByteAnd(a, b) = a & b`.
  ByteAnd(Box<Term>, Box<Term>),
  /// `ByteOr(a, b) = a | b`.
  ByteOr(Box<Term>, Box<Term>),
  /// `ByteXor(a, b) = a ^ b`.
  ByteXor(Box<Term>, Box<Term>),

  // ---- Word-level (256-bit, 32 bytes big-endian: index 0 = MSB) ----
  /// `Word([b0, …, b31])` — concrete word from 32 byte terms.
  Word(Box<[Term; 32]>),

  // ---- Abstract word operations (map 1-to-1 to EVM opcodes) ----
  /// 256-bit modular addition.
  WordAdd(Box<Term>, Box<Term>),
  /// 256-bit modular subtraction.
  WordSub(Box<Term>, Box<Term>),
  /// 256-bit modular multiplication.
  WordMul(Box<Term>, Box<Term>),
  /// 256-bit bitwise AND.
  WordAnd(Box<Term>, Box<Term>),
  /// 256-bit bitwise OR.
  WordOr(Box<Term>, Box<Term>),
  /// 256-bit bitwise XOR.
  WordXor(Box<Term>, Box<Term>),
  /// 256-bit bitwise NOT.
  WordNot(Box<Term>),
  /// Unsigned less-than → Word(1) or Word(0).
  WordLt(Box<Term>, Box<Term>),
  /// Unsigned greater-than → Word(1) or Word(0).
  WordGt(Box<Term>, Box<Term>),
  /// Equality check → Word(1) or Word(0).
  WordEqOp(Box<Term>, Box<Term>),
  /// Is-zero check → Word(1) or Word(0).
  WordIsZero(Box<Term>),
}

// ============================================================
// Well-formed formulas
// ============================================================

/// Well-formed formulas in the proof system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WFF {
  /// `t1 = t2`.
  Eq(Box<Term>, Box<Term>),
  /// `φ ∧ ψ`.
  And(Box<WFF>, Box<WFF>),
  /// `φ ∨ ψ`.
  Or(Box<WFF>, Box<WFF>),
  /// `¬φ`.
  Not(Box<WFF>),
}

// ============================================================
// Proof terms
// ============================================================

/// Hilbert-style proof terms.
///
/// Each variant is either an **axiom schema** (leaf) or an **inference rule**
/// (interior node with sub-proofs).  [`verify`] checks validity and extracts
/// the proven equation.
#[derive(Debug, Clone)]
pub enum Proof {
  // ======== Structural equality rules ========
  /// Axiom: `t = t` (reflexivity).
  EqRefl(Term),
  /// From `(a = b)`, derive `(b = a)` (symmetry).
  EqSym(Box<Proof>),
  /// From `(a = b)` and `(b = c)`, derive `(a = c)` (transitivity).
  EqTrans(Box<Proof>, Box<Proof>),

  // ======== Byte axioms (lookup-table verifiable) ========
  /// `ByteAdd(Byte(a), Byte(b), Byte(c)) = Byte((a+b+c) mod 256)`, c ∈ {0,1}.
  ByteAddEq(u8, u8, u8),
  /// `ByteAddCarry(Byte(a), Byte(b), Byte(c)) = Byte((a+b+c) / 256)`.
  ByteAddCarryEq(u8, u8, u8),
  /// `ByteNeg(Byte(a)) = Byte(!a)`.
  ByteNegEq(u8),
  /// `ByteMulLow(Byte(a), Byte(b)) = Byte((a*b) mod 256)`.
  ByteMulLowEq(u8, u8),
  /// `ByteMulHigh(Byte(a), Byte(b)) = Byte((a*b) / 256)`.
  ByteMulHighEq(u8, u8),
  /// `ByteInv(Byte(a)) = Byte(inv(a))`.
  ByteInvEq(u8),
  /// `ByteAnd(Byte(a), Byte(b)) = Byte(a & b)`.
  ByteAndEq(u8, u8),
  /// `ByteOr(Byte(a), Byte(b)) = Byte(a | b)`.
  ByteOrEq(u8, u8),
  /// `ByteXor(Byte(a), Byte(b)) = Byte(a ^ b)`.
  ByteXorEq(u8, u8),

  // ======== Word inference rules ========
  /// **Bytewise word equality**: from 32 byte-equality proofs, derive
  /// `Word([a_0,…,a_31]) = Word([b_0,…,b_31])`.
  WordBytewiseEq(Box<[Proof; 32]>),

  /// **256-bit ripple-carry addition**.
  ///
  /// - `byte_proofs[i]` must be `ByteAddEq(a_i, b_i, c_i)`
  /// - `carry_proofs[i]` must be `ByteAddCarryEq(a_i, b_i, c_i)`
  /// - Carries propagate from LSB (index 31) toward MSB (index 0).
  /// - Initial carry at index 31 is 0.
  ///
  /// Proves `WordAdd(Word(a), Word(b)) = Word(r)`.
  WordAddRule {
    byte_proofs: Box<[Proof; 32]>,
    carry_proofs: Box<[Proof; 32]>,
  },

  /// **256-bit subtraction** via two's complement: `a − b = a + NOT(b) + 1`.
  ///
  /// - `neg_proofs[i]`: `ByteNegEq(b_i)`
  /// - `byte_add_proofs[i]`: `ByteAddEq(a_i, !b_i, c_i)`
  /// - `carry_proofs[i]`: `ByteAddCarryEq(a_i, !b_i, c_i)`
  /// - Initial carry at index 31 is **1** (the +1 of two's complement).
  ///
  /// Proves `WordSub(Word(a), Word(b)) = Word(r)`.
  WordSubRule {
    neg_proofs: Box<[Proof; 32]>,
    byte_add_proofs: Box<[Proof; 32]>,
    carry_proofs: Box<[Proof; 32]>,
  },

  /// **Bytewise AND**. Proves `WordAnd(Word(a), Word(b)) = Word(r)`.
  WordAndRule(Box<[Proof; 32]>),
  /// **Bytewise OR**. Proves `WordOr(Word(a), Word(b)) = Word(r)`.
  WordOrRule(Box<[Proof; 32]>),
  /// **Bytewise XOR**. Proves `WordXor(Word(a), Word(b)) = Word(r)`.
  WordXorRule(Box<[Proof; 32]>),
  /// **Bytewise NOT**. Proves `WordNot(Word(a)) = Word(r)`.
  WordNotRule(Box<[Proof; 32]>),

  /// **Word equality check** (EQ opcode).
  /// Proves `WordEqOp(Word(a), Word(b)) = Word(0 or 1)`.
  WordEqCheckRule {
    a_bytes: [u8; 32],
    b_bytes: [u8; 32],
  },
  /// **Word is-zero** (ISZERO opcode).
  /// Proves `WordIsZero(Word(a)) = Word(0 or 1)`.
  WordIsZeroRule { a_bytes: [u8; 32] },
  /// **Word less-than** (LT opcode).
  /// Proves `WordLt(Word(a), Word(b)) = Word(0 or 1)`.
  WordLtRule {
    a_bytes: [u8; 32],
    b_bytes: [u8; 32],
  },
  /// **Word greater-than** (GT opcode).
  /// Proves `WordGt(Word(a), Word(b)) = Word(0 or 1)`.
  WordGtRule {
    a_bytes: [u8; 32],
    b_bytes: [u8; 32],
  },
}

// ============================================================
// Verification errors
// ============================================================

/// Errors produced during Hilbert-style proof verification.
#[derive(Debug, Clone)]
pub enum VerifyError {
  /// Carry byte must be 0 or 1.
  InvalidCarry(u8),
  /// Carry chain inconsistency.
  CarryMismatch {
    byte_index: usize,
    expected: u8,
    got: u8,
  },
  /// Sub-proof operands don't match expected values.
  InputMismatch { byte_index: usize },
  /// Expected a different proof variant.
  UnexpectedProofVariant { expected: &'static str },
  /// Transitivity: intermediate terms don't match.
  TransitivityMismatch,
  /// Subtraction neg byte doesn't match add operand.
  SubNegMismatch { byte_index: usize },
}

impl fmt::Display for VerifyError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::InvalidCarry(c) => write!(f, "invalid carry value: {c} (must be 0 or 1)"),
      Self::CarryMismatch {
        byte_index,
        expected,
        got,
      } => write!(
        f,
        "carry mismatch at byte {byte_index}: expected {expected}, got {got}"
      ),
      Self::InputMismatch { byte_index } => {
        write!(f, "input mismatch at byte {byte_index}")
      }
      Self::UnexpectedProofVariant { expected } => {
        write!(f, "expected proof variant: {expected}")
      }
      Self::TransitivityMismatch => write!(f, "transitivity: intermediate terms differ"),
      Self::SubNegMismatch { byte_index } => {
        write!(f, "subtraction negation mismatch at byte {byte_index}")
      }
    }
  }
}

// ============================================================
// Term construction helpers
// ============================================================

/// Box a byte term.
fn bt(val: u8) -> Box<Term> {
  Box::new(Term::Byte(val))
}

/// Construct `Word([Byte(b0), …, Byte(b31)])` from concrete bytes.
pub fn make_word_term(bytes: &[u8; 32]) -> Term {
  Term::Word(Box::new(std::array::from_fn(|i| Term::Byte(bytes[i]))))
}

/// Word(0).
pub fn word_zero() -> Term {
  make_word_term(&[0u8; 32])
}

/// Word(1) — big-endian: `[0,…,0,1]`.
pub fn word_one() -> Term {
  let mut b = [0u8; 32];
  b[31] = 1;
  make_word_term(&b)
}

/// Extract concrete byte values from a `Word([Byte(b0),…,Byte(b31)])`.
pub fn extract_word_bytes(term: &Term) -> Option<[u8; 32]> {
  match term {
    Term::Word(bytes) => {
      let mut out = [0u8; 32];
      for i in 0..32 {
        match &bytes[i] {
          Term::Byte(b) => out[i] = *b,
          _ => return None,
        }
      }
      Some(out)
    }
    _ => None,
  }
}

// ============================================================
// Verification
// ============================================================

/// Verify a Hilbert-style proof and return the proven equation `(lhs, rhs)`.
///
/// Returns `Err` if any axiom instance is invalid or any inference rule
/// is incorrectly applied.
pub fn verify(proof: &Proof) -> Result<(Term, Term), VerifyError> {
  match proof {
    // ---- Structural equality ----
    Proof::EqRefl(t) => Ok((t.clone(), t.clone())),

    Proof::EqSym(p) => {
      let (lhs, rhs) = verify(p)?;
      Ok((rhs, lhs))
    }

    Proof::EqTrans(p1, p2) => {
      let (a, b1) = verify(p1)?;
      let (b2, c) = verify(p2)?;
      if b1 != b2 {
        return Err(VerifyError::TransitivityMismatch);
      }
      Ok((a, c))
    }

    // ---- Byte axioms ----
    Proof::ByteAddEq(a, b, c) => {
      if *c > 1 {
        return Err(VerifyError::InvalidCarry(*c));
      }
      let sum = *a as u16 + *b as u16 + *c as u16;
      let result = (sum & 0xFF) as u8;
      Ok((Term::ByteAdd(bt(*a), bt(*b), bt(*c)), Term::Byte(result)))
    }

    Proof::ByteAddCarryEq(a, b, c) => {
      if *c > 1 {
        return Err(VerifyError::InvalidCarry(*c));
      }
      let sum = *a as u16 + *b as u16 + *c as u16;
      let carry = (sum >> 8) as u8;
      Ok((
        Term::ByteAddCarry(bt(*a), bt(*b), bt(*c)),
        Term::Byte(carry),
      ))
    }

    Proof::ByteNegEq(a) => Ok((Term::ByteNeg(bt(*a)), Term::Byte(!*a))),

    Proof::ByteMulLowEq(a, b) => {
      let low = (((*a as u16) * (*b as u16)) & 0xFF) as u8;
      Ok((Term::ByteMulLow(bt(*a), bt(*b)), Term::Byte(low)))
    }

    Proof::ByteMulHighEq(a, b) => {
      let high = (((*a as u16) * (*b as u16)) >> 8) as u8;
      Ok((Term::ByteMulHigh(bt(*a), bt(*b)), Term::Byte(high)))
    }

    Proof::ByteInvEq(a) => {
      let inv = if *a == 0 {
        0
      } else {
        (0u16..256)
          .find(|&x| ((*a as u16 * x) & 0xFF) == 1)
          .unwrap_or(0) as u8
      };
      Ok((Term::ByteInv(bt(*a)), Term::Byte(inv)))
    }

    Proof::ByteAndEq(a, b) => Ok((Term::ByteAnd(bt(*a), bt(*b)), Term::Byte(*a & *b))),
    Proof::ByteOrEq(a, b) => Ok((Term::ByteOr(bt(*a), bt(*b)), Term::Byte(*a | *b))),
    Proof::ByteXorEq(a, b) => Ok((Term::ByteXor(bt(*a), bt(*b)), Term::Byte(*a ^ *b))),

    // ---- Word bytewise equality ----
    Proof::WordBytewiseEq(proofs) => {
      let mut lhs_arr: Vec<Term> = Vec::with_capacity(32);
      let mut rhs_arr: Vec<Term> = Vec::with_capacity(32);
      for i in 0..32 {
        let (l, r) = verify(&proofs[i])?;
        lhs_arr.push(l);
        rhs_arr.push(r);
      }
      let la: [Term; 32] = lhs_arr.try_into().expect("32 elements");
      let ra: [Term; 32] = rhs_arr.try_into().expect("32 elements");
      Ok((Term::Word(Box::new(la)), Term::Word(Box::new(ra))))
    }

    // ---- Word addition (ripple-carry) ----
    Proof::WordAddRule {
      byte_proofs,
      carry_proofs,
    } => verify_word_add(byte_proofs, carry_proofs, 0),

    // ---- Word subtraction (two's complement) ----
    Proof::WordSubRule {
      neg_proofs,
      byte_add_proofs,
      carry_proofs,
    } => {
      // 1. Verify negation proofs and extract b[i], !b[i]
      let mut b_bytes = [0u8; 32];
      let mut neg_b_bytes = [0u8; 32];
      for i in 0..32 {
        match &neg_proofs[i] {
          Proof::ByteNegEq(b) => {
            b_bytes[i] = *b;
            neg_b_bytes[i] = !*b;
          }
          _ => {
            return Err(VerifyError::UnexpectedProofVariant {
              expected: "ByteNegEq",
            });
          }
        }
      }

      // 2. Verify the addition a + NOT(b) with initial carry = 1
      let (_, rhs) = verify_word_add(byte_add_proofs, carry_proofs, 1)?;

      // 3. Verify that the add operand b-side matches neg_b
      let mut a_bytes = [0u8; 32];
      for i in 0..32 {
        match &byte_add_proofs[i] {
          Proof::ByteAddEq(a, nb, _) => {
            if *nb != neg_b_bytes[i] {
              return Err(VerifyError::SubNegMismatch { byte_index: i });
            }
            a_bytes[i] = *a;
          }
          _ => {
            return Err(VerifyError::UnexpectedProofVariant {
              expected: "ByteAddEq",
            });
          }
        }
      }

      Ok((
        Term::WordSub(
          Box::new(make_word_term(&a_bytes)),
          Box::new(make_word_term(&b_bytes)),
        ),
        rhs,
      ))
    }

    // ---- Bytewise bitwise operations ----
    Proof::WordAndRule(proofs) => verify_bytewise_binop(proofs, ByteOp::And),
    Proof::WordOrRule(proofs) => verify_bytewise_binop(proofs, ByteOp::Or),
    Proof::WordXorRule(proofs) => verify_bytewise_binop(proofs, ByteOp::Xor),

    Proof::WordNotRule(proofs) => {
      let mut a_bytes = [0u8; 32];
      let mut r_bytes = [0u8; 32];
      for i in 0..32 {
        match &proofs[i] {
          Proof::ByteNegEq(a) => {
            a_bytes[i] = *a;
            r_bytes[i] = !*a;
          }
          _ => {
            return Err(VerifyError::UnexpectedProofVariant {
              expected: "ByteNegEq",
            });
          }
        }
      }
      Ok((
        Term::WordNot(Box::new(make_word_term(&a_bytes))),
        make_word_term(&r_bytes),
      ))
    }

    // ---- Comparison / check rules (axiom-style) ----
    Proof::WordEqCheckRule { a_bytes, b_bytes } => {
      let eq = a_bytes == b_bytes;
      let result = if eq { word_one() } else { word_zero() };
      Ok((
        Term::WordEqOp(
          Box::new(make_word_term(a_bytes)),
          Box::new(make_word_term(b_bytes)),
        ),
        result,
      ))
    }

    Proof::WordIsZeroRule { a_bytes } => {
      let zero = a_bytes.iter().all(|&b| b == 0);
      let result = if zero { word_one() } else { word_zero() };
      Ok((Term::WordIsZero(Box::new(make_word_term(a_bytes))), result))
    }

    Proof::WordLtRule { a_bytes, b_bytes } => {
      // Big-endian byte array comparison = unsigned integer comparison
      let lt = *a_bytes < *b_bytes;
      let result = if lt { word_one() } else { word_zero() };
      Ok((
        Term::WordLt(
          Box::new(make_word_term(a_bytes)),
          Box::new(make_word_term(b_bytes)),
        ),
        result,
      ))
    }

    Proof::WordGtRule { a_bytes, b_bytes } => {
      let gt = *a_bytes > *b_bytes;
      let result = if gt { word_one() } else { word_zero() };
      Ok((
        Term::WordGt(
          Box::new(make_word_term(a_bytes)),
          Box::new(make_word_term(b_bytes)),
        ),
        result,
      ))
    }
  }
}

// ---- Internal verification helpers ----

/// Verify a ripple-carry word addition with the given initial carry (0 or 1).
fn verify_word_add(
  byte_proofs: &[Proof; 32],
  carry_proofs: &[Proof; 32],
  initial_carry: u8,
) -> Result<(Term, Term), VerifyError> {
  let mut carry = initial_carry;
  let mut a_bytes = [0u8; 32];
  let mut b_bytes = [0u8; 32];
  let mut r_bytes = [0u8; 32];

  for i in (0..32).rev() {
    match &byte_proofs[i] {
      Proof::ByteAddEq(a, b, c) => {
        if *c != carry {
          return Err(VerifyError::CarryMismatch {
            byte_index: i,
            expected: carry,
            got: *c,
          });
        }
        a_bytes[i] = *a;
        b_bytes[i] = *b;
        let sum = *a as u16 + *b as u16 + *c as u16;
        r_bytes[i] = (sum & 0xFF) as u8;
      }
      _ => {
        return Err(VerifyError::UnexpectedProofVariant {
          expected: "ByteAddEq",
        });
      }
    }

    match &carry_proofs[i] {
      Proof::ByteAddCarryEq(a, b, c) => {
        if *a != a_bytes[i] || *b != b_bytes[i] || *c != carry {
          return Err(VerifyError::InputMismatch { byte_index: i });
        }
        let sum = *a as u16 + *b as u16 + *c as u16;
        carry = (sum >> 8) as u8;
      }
      _ => {
        return Err(VerifyError::UnexpectedProofVariant {
          expected: "ByteAddCarryEq",
        });
      }
    }
  }

  let word_a = make_word_term(&a_bytes);
  let word_b = make_word_term(&b_bytes);
  let word_r = make_word_term(&r_bytes);

  Ok((Term::WordAdd(Box::new(word_a), Box::new(word_b)), word_r))
}

#[derive(Clone, Copy)]
enum ByteOp {
  And,
  Or,
  Xor,
}

/// Verify a bytewise binary operation (AND / OR / XOR).
fn verify_bytewise_binop(proofs: &[Proof; 32], op: ByteOp) -> Result<(Term, Term), VerifyError> {
  let mut a_bytes = [0u8; 32];
  let mut b_bytes = [0u8; 32];
  let mut r_bytes = [0u8; 32];

  let expected = match op {
    ByteOp::And => "ByteAndEq",
    ByteOp::Or => "ByteOrEq",
    ByteOp::Xor => "ByteXorEq",
  };

  for i in 0..32 {
    match (&proofs[i], op) {
      (Proof::ByteAndEq(a, b), ByteOp::And) => {
        a_bytes[i] = *a;
        b_bytes[i] = *b;
        r_bytes[i] = *a & *b;
      }
      (Proof::ByteOrEq(a, b), ByteOp::Or) => {
        a_bytes[i] = *a;
        b_bytes[i] = *b;
        r_bytes[i] = *a | *b;
      }
      (Proof::ByteXorEq(a, b), ByteOp::Xor) => {
        a_bytes[i] = *a;
        b_bytes[i] = *b;
        r_bytes[i] = *a ^ *b;
      }
      _ => {
        return Err(VerifyError::UnexpectedProofVariant { expected });
      }
    }
  }

  let wa = make_word_term(&a_bytes);
  let wb = make_word_term(&b_bytes);
  let wr = make_word_term(&r_bytes);

  let lhs = match op {
    ByteOp::And => Term::WordAnd(Box::new(wa), Box::new(wb)),
    ByteOp::Or => Term::WordOr(Box::new(wa), Box::new(wb)),
    ByteOp::Xor => Term::WordXor(Box::new(wa), Box::new(wb)),
  };
  Ok((lhs, wr))
}

// ============================================================
// Proof generation
// ============================================================

/// Generate a Hilbert-style proof for 256-bit addition and compute the result.
pub fn prove_word_add(a: &[u8; 32], b: &[u8; 32]) -> (Proof, [u8; 32]) {
  let mut result = [0u8; 32];
  let mut byte_proofs = Vec::with_capacity(32);
  let mut carry_proofs = Vec::with_capacity(32);
  let mut carry: u8 = 0;

  for i in (0..32).rev() {
    let sum = a[i] as u16 + b[i] as u16 + carry as u16;
    result[i] = (sum & 0xFF) as u8;
    let new_carry = (sum >> 8) as u8;
    byte_proofs.push(Proof::ByteAddEq(a[i], b[i], carry));
    carry_proofs.push(Proof::ByteAddCarryEq(a[i], b[i], carry));
    carry = new_carry;
  }
  byte_proofs.reverse();
  carry_proofs.reverse();

  let proof = Proof::WordAddRule {
    byte_proofs: Box::new(byte_proofs.try_into().expect("32 elements")),
    carry_proofs: Box::new(carry_proofs.try_into().expect("32 elements")),
  };
  (proof, result)
}

/// Generate a Hilbert-style proof for 256-bit subtraction (`a − b`).
pub fn prove_word_sub(a: &[u8; 32], b: &[u8; 32]) -> (Proof, [u8; 32]) {
  let mut neg_b = [0u8; 32];
  let neg_proofs: [Proof; 32] = std::array::from_fn(|i| {
    neg_b[i] = !b[i];
    Proof::ByteNegEq(b[i])
  });

  let mut result = [0u8; 32];
  let mut byte_add_proofs = Vec::with_capacity(32);
  let mut carry_proofs = Vec::with_capacity(32);
  let mut carry: u8 = 1; // +1 of two's complement

  for i in (0..32).rev() {
    let sum = a[i] as u16 + neg_b[i] as u16 + carry as u16;
    result[i] = (sum & 0xFF) as u8;
    let new_carry = (sum >> 8) as u8;
    byte_add_proofs.push(Proof::ByteAddEq(a[i], neg_b[i], carry));
    carry_proofs.push(Proof::ByteAddCarryEq(a[i], neg_b[i], carry));
    carry = new_carry;
  }
  byte_add_proofs.reverse();
  carry_proofs.reverse();

  let proof = Proof::WordSubRule {
    neg_proofs: Box::new(neg_proofs),
    byte_add_proofs: Box::new(byte_add_proofs.try_into().expect("32")),
    carry_proofs: Box::new(carry_proofs.try_into().expect("32")),
  };
  (proof, result)
}

/// Bitwise AND proof.
pub fn prove_word_and(a: &[u8; 32], b: &[u8; 32]) -> (Proof, [u8; 32]) {
  let mut result = [0u8; 32];
  let proofs: [Proof; 32] = std::array::from_fn(|i| {
    result[i] = a[i] & b[i];
    Proof::ByteAndEq(a[i], b[i])
  });
  (Proof::WordAndRule(Box::new(proofs)), result)
}

/// Bitwise OR proof.
pub fn prove_word_or(a: &[u8; 32], b: &[u8; 32]) -> (Proof, [u8; 32]) {
  let mut result = [0u8; 32];
  let proofs: [Proof; 32] = std::array::from_fn(|i| {
    result[i] = a[i] | b[i];
    Proof::ByteOrEq(a[i], b[i])
  });
  (Proof::WordOrRule(Box::new(proofs)), result)
}

/// Bitwise XOR proof.
pub fn prove_word_xor(a: &[u8; 32], b: &[u8; 32]) -> (Proof, [u8; 32]) {
  let mut result = [0u8; 32];
  let proofs: [Proof; 32] = std::array::from_fn(|i| {
    result[i] = a[i] ^ b[i];
    Proof::ByteXorEq(a[i], b[i])
  });
  (Proof::WordXorRule(Box::new(proofs)), result)
}

/// Bitwise NOT proof.
pub fn prove_word_not(a: &[u8; 32]) -> (Proof, [u8; 32]) {
  let mut result = [0u8; 32];
  let proofs: [Proof; 32] = std::array::from_fn(|i| {
    result[i] = !a[i];
    Proof::ByteNegEq(a[i])
  });
  (Proof::WordNotRule(Box::new(proofs)), result)
}

/// EQ check proof.
pub fn prove_word_eq(a: &[u8; 32], b: &[u8; 32]) -> (Proof, [u8; 32]) {
  let equal = a == b;
  let mut result = [0u8; 32];
  if equal {
    result[31] = 1;
  }
  (
    Proof::WordEqCheckRule {
      a_bytes: *a,
      b_bytes: *b,
    },
    result,
  )
}

/// ISZERO proof.
pub fn prove_word_iszero(a: &[u8; 32]) -> (Proof, [u8; 32]) {
  let zero = a.iter().all(|&b| b == 0);
  let mut result = [0u8; 32];
  if zero {
    result[31] = 1;
  }
  (Proof::WordIsZeroRule { a_bytes: *a }, result)
}

/// LT proof.
pub fn prove_word_lt(a: &[u8; 32], b: &[u8; 32]) -> (Proof, [u8; 32]) {
  let lt = *a < *b;
  let mut result = [0u8; 32];
  if lt {
    result[31] = 1;
  }
  (
    Proof::WordLtRule {
      a_bytes: *a,
      b_bytes: *b,
    },
    result,
  )
}

/// GT proof.
pub fn prove_word_gt(a: &[u8; 32], b: &[u8; 32]) -> (Proof, [u8; 32]) {
  let gt = *a > *b;
  let mut result = [0u8; 32];
  if gt {
    result[31] = 1;
  }
  (
    Proof::WordGtRule {
      a_bytes: *a,
      b_bytes: *b,
    },
    result,
  )
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
  use super::*;

  fn u256_bytes(val: u128) -> [u8; 32] {
    let mut b = [0u8; 32];
    b[16..32].copy_from_slice(&val.to_be_bytes());
    b
  }

  #[test]
  fn test_byte_add_eq_verify() {
    let proof = Proof::ByteAddEq(200, 100, 0);
    let (lhs, rhs) = verify(&proof).unwrap();
    assert_eq!(rhs, Term::Byte(44));
    assert_eq!(lhs, Term::ByteAdd(bt(200), bt(100), bt(0)));
  }

  #[test]
  fn test_byte_add_carry_verify() {
    let proof = Proof::ByteAddCarryEq(200, 100, 0);
    let (_, rhs) = verify(&proof).unwrap();
    assert_eq!(rhs, Term::Byte(1));
  }

  #[test]
  fn test_word_add_simple() {
    let a = u256_bytes(100);
    let b = u256_bytes(200);
    let (proof, result) = prove_word_add(&a, &b);
    assert_eq!(result, u256_bytes(300));
    let (lhs, rhs) = verify(&proof).unwrap();
    assert_eq!(rhs, make_word_term(&u256_bytes(300)));
    assert_eq!(
      lhs,
      Term::WordAdd(Box::new(make_word_term(&a)), Box::new(make_word_term(&b)))
    );
  }

  #[test]
  fn test_word_add_overflow() {
    let a = [0xFF; 32];
    let b = u256_bytes(1);
    let (proof, result) = prove_word_add(&a, &b);
    assert_eq!(result, [0u8; 32]);
    verify(&proof).unwrap();
  }

  #[test]
  fn test_word_sub_simple() {
    let a = u256_bytes(300);
    let b = u256_bytes(100);
    let (proof, result) = prove_word_sub(&a, &b);
    assert_eq!(result, u256_bytes(200));
    let (lhs, rhs) = verify(&proof).unwrap();
    assert_eq!(rhs, make_word_term(&u256_bytes(200)));
    assert_eq!(
      lhs,
      Term::WordSub(Box::new(make_word_term(&a)), Box::new(make_word_term(&b)))
    );
  }

  #[test]
  fn test_word_sub_underflow() {
    let a = u256_bytes(0);
    let b = u256_bytes(1);
    let (proof, result) = prove_word_sub(&a, &b);
    assert_eq!(result, [0xFF; 32]);
    verify(&proof).unwrap();
  }

  #[test]
  fn test_word_and() {
    let a = u256_bytes(0xFF00);
    let b = u256_bytes(0x0FF0);
    let (proof, result) = prove_word_and(&a, &b);
    assert_eq!(result, u256_bytes(0x0F00));
    verify(&proof).unwrap();
  }

  #[test]
  fn test_word_or() {
    let a = u256_bytes(0xFF00);
    let b = u256_bytes(0x0FF0);
    let (proof, result) = prove_word_or(&a, &b);
    assert_eq!(result, u256_bytes(0xFFF0));
    verify(&proof).unwrap();
  }

  #[test]
  fn test_word_xor() {
    let a = u256_bytes(0xFF00);
    let b = u256_bytes(0x0FF0);
    let (proof, result) = prove_word_xor(&a, &b);
    assert_eq!(result, u256_bytes(0xF0F0));
    verify(&proof).unwrap();
  }

  #[test]
  fn test_word_not() {
    let a = [0u8; 32];
    let (proof, result) = prove_word_not(&a);
    assert_eq!(result, [0xFF; 32]);
    verify(&proof).unwrap();
  }

  #[test]
  fn test_word_eq_true() {
    let a = u256_bytes(42);
    let (proof, result) = prove_word_eq(&a, &a);
    let mut expected = [0u8; 32];
    expected[31] = 1;
    assert_eq!(result, expected);
    verify(&proof).unwrap();
  }

  #[test]
  fn test_word_eq_false() {
    let a = u256_bytes(1);
    let b = u256_bytes(2);
    let (proof, result) = prove_word_eq(&a, &b);
    assert_eq!(result, [0u8; 32]);
    verify(&proof).unwrap();
  }

  #[test]
  fn test_word_iszero() {
    let zero = [0u8; 32];
    let (proof, result) = prove_word_iszero(&zero);
    let mut expected = [0u8; 32];
    expected[31] = 1;
    assert_eq!(result, expected);
    verify(&proof).unwrap();

    let nonzero = u256_bytes(1);
    let (proof2, result2) = prove_word_iszero(&nonzero);
    assert_eq!(result2, [0u8; 32]);
    verify(&proof2).unwrap();
  }

  #[test]
  fn test_word_lt() {
    let a = u256_bytes(10);
    let b = u256_bytes(20);
    let (proof, result) = prove_word_lt(&a, &b);
    let mut expected = [0u8; 32];
    expected[31] = 1;
    assert_eq!(result, expected);
    verify(&proof).unwrap();
  }

  #[test]
  fn test_word_gt() {
    let a = u256_bytes(20);
    let b = u256_bytes(10);
    let (proof, result) = prove_word_gt(&a, &b);
    let mut expected = [0u8; 32];
    expected[31] = 1;
    assert_eq!(result, expected);
    verify(&proof).unwrap();
  }

  #[test]
  fn test_eq_sym() {
    let a = u256_bytes(5);
    let b = u256_bytes(10);
    let (add_proof, _) = prove_word_add(&a, &b);
    let sym = Proof::EqSym(Box::new(add_proof));
    let (lhs, rhs) = verify(&sym).unwrap();
    assert_eq!(lhs, make_word_term(&u256_bytes(15)));
    assert_eq!(
      rhs,
      Term::WordAdd(Box::new(make_word_term(&a)), Box::new(make_word_term(&b)))
    );
  }

  #[test]
  fn test_invalid_carry_rejected() {
    let bad = Proof::ByteAddEq(1, 2, 3);
    assert!(verify(&bad).is_err());
  }
}
