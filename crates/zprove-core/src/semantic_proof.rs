use revm_primitives::U256;
use serde::Serialize;
use std::{array, collections::HashMap, fmt};

/// CSE memo for `compile_proof_inner` / `compile_term_inner`.
///
/// Key: `(op, scalar0, scalar1, scalar2)` — uniquely identifies any *leaf* row
/// (those with no child-row operands).  Non-leaf rows (AndIntro, EqTrans, …)
/// are never cached: their identity depends on child row indices which shift
/// whenever the rows vec grows.
/// CSE key: (op, arg0, arg1, arg2, scalar0, scalar1, scalar2)
type Memo = HashMap<(u32, u32, u32, u32, u32, u32, u32), u32>;

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
  ByteXor(Box<Term>, Box<Term>),

  // ---- Symbolic variable terms (resolved at batch-verification time) ----
  /// The `byte_idx`-th byte (big-endian, 0 = MSB) of the `stack_idx`-th stack input word.
  InputTerm { stack_idx: u8, byte_idx: u8 },
  /// The `byte_idx`-th byte of the `stack_idx`-th stack output word.
  OutputTerm { stack_idx: u8, byte_idx: u8 },
  /// The `byte_idx`-th byte of the program counter before execution (4-byte big-endian u32).
  PcBefore { byte_idx: u8 },
  /// The `byte_idx`-th byte of the program counter after execution.
  PcAfter { byte_idx: u8 },
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

  // ── Per-opcode axiom formulas ─────────────────────────────────────────
  // Each variant represents the proposition that the axiom asserts.
  // Soundness of the claim is enforced by the corresponding consistency
  // AIR at the batch level; the axiom proof term is the per-instruction
  // witness that the claim is well-formed and matches the stack I/O.

  /// PUSH: the pushed value equals `value`.
  PushAxiom,
  /// DUP(depth): the top of stack equals the item at `depth`.
  DupAxiom { depth: u8 },
  /// SWAP(depth): top and item at `depth` are exchanged.
  SwapAxiom { depth: u8 },
  /// Structural no-output opcodes: STOP, POP, JUMP, JUMPI, JUMPDEST, INVALID.
  StructuralAxiom { opcode: u8 },
  /// MLOAD: reading address returns a value from memory.
  MloadAxiom,
  /// MSTORE / MSTORE8: writing a value to memory.
  MstoreAxiom { opcode: u8 },
  /// Memory-copy opcodes (CALLDATACOPY, CODECOPY, RETURNDATACOPY, EXTCODECOPY, MCOPY).
  MemCopyAxiom { opcode: u8 },
  /// SLOAD: reading a storage slot returns a value.
  SloadAxiom,
  /// SSTORE: writing a value to a storage slot.
  SstoreAxiom,
  /// TLOAD / TSTORE (transient storage, EIP-1153).
  TransientAxiom { opcode: u8 },
  /// KECCAK256: hash of memory region.
  KeccakAxiom,
  /// Environment / context opcodes: the value equals what the EVM state provides.
  EnvAxiom { opcode: u8 },
  /// External-state opcodes (BLOCKHASH, EXTCODESIZE, BALANCE, EXTCODEHASH).
  ExternalStateAxiom { opcode: u8 },
  /// RETURN / REVERT: specifies offset and size of return data.
  TerminateAxiom { opcode: u8 },
  /// CALL / CALLCODE / DELEGATECALL / STATICCALL.
  CallAxiom { opcode: u8 },
  /// CREATE / CREATE2: deploys a contract.
  CreateAxiom { opcode: u8 },
  /// SELFDESTRUCT.
  SelfdestructAxiom,
  /// LOG0-LOG4: emits a log entry.
  LogAxiom { opcode: u8 },
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
  /// `ByteAnd(Byte(a), Byte(b)) = Byte(a & b)`.
  ByteAndEq(u8, u8),
  /// `ByteOr(Byte(a), Byte(b)) = Byte(a | b)`.
  ByteOrEq(u8, u8),
  /// `ByteXor(Byte(a), Byte(b)) = Byte(a ^ b)`.
  ByteXorEq(u8, u8),
  /// 29-bit chunk add axiom:
  /// `(a29 + b29 + cin) mod 2^29 = c29`.
  U29AddEq(u32, u32, bool, u32),
  /// 24-bit chunk add axiom (top chunk for 256-bit words with 29-bit radix):
  /// `(a24 + b24 + cin) mod 2^24 = c24`.
  U24AddEq(u32, u32, bool, u32),
  /// 15-bit chunk mul axiom:
  /// `a15 * b15 = lo15 + 2^15 * hi15`.
  U15MulEq(u16, u16),
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
  /// `ByteMulLow(Byte(a), Byte(b)) = Byte((a * b) & 0xFF)`.
  ByteMulLowEq(u8, u8),
  /// `ByteMulHigh(Byte(a), Byte(b)) = Byte((a * b) >> 8)`.
  ByteMulHighEq(u8, u8),

  // ── Per-opcode axiom proof terms ──────────────────────────────────────
  // Leaf nodes: no sub-proofs.  Each encodes exactly the WFF it witnesses.
  PushAxiom,
  DupAxiom { depth: u8 },
  SwapAxiom { depth: u8 },
  StructuralAxiom { opcode: u8 },
  MloadAxiom,
  MstoreAxiom { opcode: u8 },
  MemCopyAxiom { opcode: u8 },
  SloadAxiom,
  SstoreAxiom,
  TransientAxiom { opcode: u8 },
  KeccakAxiom,
  EnvAxiom { opcode: u8 },
  ExternalStateAxiom { opcode: u8 },
  TerminateAxiom { opcode: u8 },
  CallAxiom { opcode: u8 },
  CreateAxiom { opcode: u8 },
  SelfdestructAxiom,
  LogAxiom { opcode: u8 },
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
pub const OP_BOOL: u32 = 0;
pub const OP_NOT: u32 = 1;
pub const OP_AND: u32 = 2;
pub const OP_OR: u32 = 3;
pub const OP_XOR: u32 = 4;
pub const OP_ITE: u32 = 5;
pub const OP_BYTE: u32 = 6;
pub const OP_BYTE_ADD: u32 = 7;
pub const OP_BYTE_ADD_CARRY: u32 = 8;
pub const OP_BYTE_MUL_LOW: u32 = 9;
pub const OP_BYTE_MUL_HIGH: u32 = 10;
pub const OP_BYTE_AND: u32 = 11;
pub const OP_BYTE_OR: u32 = 12;
pub const OP_BYTE_XOR: u32 = 13;
// ---- Proof opcodes (14..24) ----
pub const OP_AND_INTRO: u32 = 14;
pub const OP_EQ_REFL: u32 = 15;
pub const OP_EQ_SYM: u32 = 16;
pub const OP_EQ_TRANS: u32 = 17;
pub const OP_BYTE_MUL_LOW_EQ: u32 = 18;
pub const OP_BYTE_MUL_HIGH_EQ: u32 = 19;
pub const OP_BYTE_AND_EQ: u32 = 20;
pub const OP_BYTE_OR_EQ: u32 = 21;
pub const OP_BYTE_XOR_EQ: u32 = 22;
pub const OP_BYTE_ADD_THIRD_CONGRUENCE: u32 = 23;
pub const OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE: u32 = 24;
pub const OP_ITE_TRUE_EQ: u32 = 25;
pub const OP_ITE_FALSE_EQ: u32 = 26;
pub const OP_U15_MUL_EQ: u32 = 27;
pub const OP_U29_ADD_EQ: u32 = 28;
pub const OP_U24_ADD_EQ: u32 = 29;

// ---- Per-opcode axiom opcodes (30..) ----
pub const OP_PUSH_AXIOM: u32 = 30;
pub const OP_DUP_AXIOM: u32 = 31;
pub const OP_SWAP_AXIOM: u32 = 32;
pub const OP_STRUCTURAL_AXIOM: u32 = 33;
pub const OP_MLOAD_AXIOM: u32 = 34;
pub const OP_MSTORE_AXIOM: u32 = 35;
pub const OP_MEM_COPY_AXIOM: u32 = 36;
pub const OP_SLOAD_AXIOM: u32 = 37;
pub const OP_SSTORE_AXIOM: u32 = 38;
pub const OP_TRANSIENT_AXIOM: u32 = 39;
pub const OP_KECCAK_AXIOM: u32 = 40;
pub const OP_ENV_AXIOM: u32 = 41;
pub const OP_EXTERNAL_STATE_AXIOM: u32 = 42;
pub const OP_TERMINATE_AXIOM: u32 = 43;
pub const OP_CALL_AXIOM: u32 = 44;
pub const OP_CREATE_AXIOM: u32 = 45;
pub const OP_SELFDESTRUCT_AXIOM: u32 = 46;
pub const OP_LOG_AXIOM: u32 = 47;

// ---- Symbolic variable term opcodes ----
pub const OP_INPUT_TERM: u32 = 48;
pub const OP_OUTPUT_TERM: u32 = 49;
pub const OP_PC_BEFORE: u32 = 50;
pub const OP_PC_AFTER: u32 = 51;

// ---- Return-type tags ----
pub const RET_BOOL: u32 = 0;
pub const RET_BYTE: u32 = 1;
pub const RET_WFF_EQ: u32 = 2;
pub const RET_WFF_AND: u32 = 3;
/// Axiom WFF: an opcode-specific axiom whose correctness is enforced by the
/// consistency AIR at batch level, not by inline term evaluation.
pub const RET_WFF_AXIOM: u32 = 4;

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
/// | `scalar2` | Third immediate (carry-in for add opcodes, etc.)          |
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
    Term::Not(a) => {
      if infer_ty(a)? == Ty::Bool {
        Ok(Ty::Bool)
      } else {
        Err(VerifyError::UnexpectedTermVariant {
          expected: "boolean subterm",
        })
      }
    }
    Term::And(a, b) | Term::Or(a, b) | Term::Xor(a, b) => {
      if infer_ty(a)? == Ty::Bool && infer_ty(b)? == Ty::Bool {
        Ok(Ty::Bool)
      } else {
        Err(VerifyError::UnexpectedTermVariant {
          expected: "boolean subterm",
        })
      }
    }
    Term::Ite(c, a, b) => {
      if infer_ty(c)? == Ty::Bool {
        let ty_a = infer_ty(a)?;
        let ty_b = infer_ty(b)?;
        if ty_a == ty_b {
          Ok(ty_a)
        } else {
          Err(VerifyError::UnexpectedTermVariant {
            expected: "matching type subterms",
          })
        }
      } else {
        Err(VerifyError::UnexpectedTermVariant {
          expected: "boolean condition subterm",
        })
      }
    }
    Term::Byte(_) => Ok(Ty::Byte),
    Term::ByteAdd(a, b, c) => {
      if infer_ty(a)? == Ty::Byte && infer_ty(b)? == Ty::Byte && infer_ty(c)? == Ty::Bool {
        Ok(Ty::Byte)
      } else {
        Err(VerifyError::UnexpectedTermVariant {
          expected: "byte subterm",
        })
      }
    }
    Term::ByteAddCarry(a, b, c) => {
      if infer_ty(a)? == Ty::Byte && infer_ty(b)? == Ty::Byte && infer_ty(c)? == Ty::Bool {
        Ok(Ty::Bool)
      } else {
        Err(VerifyError::UnexpectedTermVariant {
          expected: "bool subterm",
        })
      }
    }
    Term::ByteMulLow(a, b)
    | Term::ByteMulHigh(a, b)
    | Term::ByteAnd(a, b)
    | Term::ByteOr(a, b)
    | Term::ByteXor(a, b) => {
      if infer_ty(a)? == Ty::Byte && infer_ty(b)? == Ty::Byte {
        Ok(Ty::Byte)
      } else {
        Err(VerifyError::UnexpectedTermVariant {
          expected: "byte subterm",
        })
      }
    }
    Term::InputTerm { .. }
    | Term::OutputTerm { .. }
    | Term::PcBefore { .. }
    | Term::PcAfter { .. } => Ok(Ty::Byte),
  }
}

pub fn infer_proof(proof: &Proof) -> Result<WFF, VerifyError> {
  match proof {
    Proof::AndIntro(p1, p2) => {
      let wff1 = infer_proof(p1)?;
      let wff2 = infer_proof(p2)?;
      Ok(WFF::And(Box::new(wff1), Box::new(wff2)))
    }
    Proof::EqRefl(t) => Ok(WFF::Equal(Box::new(t.clone()), Box::new(t.clone()))),
    Proof::EqSym(p) => {
      if let WFF::Equal(a, b) = infer_proof(p)? {
        Ok(WFF::Equal(b, a))
      } else {
        Err(VerifyError::UnexpectedProofVariant {
          expected: "equality proof",
        })
      }
    }
    Proof::EqTrans(p1, p2) => {
      if let WFF::Equal(a, b) = infer_proof(p1)? {
        if let WFF::Equal(b2, c) = infer_proof(p2)? {
          if *b == *b2 {
            Ok(WFF::Equal(a, c))
          } else {
            Err(VerifyError::TransitivityMismatch)
          }
        } else {
          Err(VerifyError::UnexpectedProofVariant {
            expected: "equality proof",
          })
        }
      } else {
        Err(VerifyError::UnexpectedProofVariant {
          expected: "equality proof",
        })
      }
    }
    Proof::ByteAndEq(a, b) => Ok(WFF::Equal(
      Box::new(Term::ByteAnd(
        Box::new(Term::Byte(*a)),
        Box::new(Term::Byte(*b)),
      )),
      Box::new(Term::Byte(*a & *b)),
    )),
    Proof::ByteOrEq(a, b) => Ok(WFF::Equal(
      Box::new(Term::ByteOr(
        Box::new(Term::Byte(*a)),
        Box::new(Term::Byte(*b)),
      )),
      Box::new(Term::Byte(*a | *b)),
    )),
    Proof::ByteXorEq(a, b) => Ok(WFF::Equal(
      Box::new(Term::ByteXor(
        Box::new(Term::Byte(*a)),
        Box::new(Term::Byte(*b)),
      )),
      Box::new(Term::Byte(*a ^ *b)),
    )),
    Proof::U29AddEq(a, b, _cin, c) => {
      if *a >= (1u32 << 29) || *b >= (1u32 << 29) || *c >= (1u32 << 29) {
        return Err(VerifyError::UnexpectedProofVariant {
          expected: "u29 inputs",
        });
      }
      let total = *a + *b + (*_cin as u32);
      let sum = total & ((1u32 << 29) - 1);
      let pairs = vec![
        ((sum & 0xFF) as u8, (*c & 0xFF) as u8),
        (((sum >> 8) & 0xFF) as u8, ((*c >> 8) & 0xFF) as u8),
        (((sum >> 16) & 0xFF) as u8, ((*c >> 16) & 0xFF) as u8),
        (((sum >> 24) & 0x1F) as u8, ((*c >> 24) & 0x1F) as u8),
      ];
      Ok(and_wffs(
        pairs
          .into_iter()
          .map(|(lhs, rhs)| WFF::Equal(Box::new(Term::Byte(lhs)), Box::new(Term::Byte(rhs))))
          .collect(),
      ))
    }
    Proof::U24AddEq(a, b, _cin, c) => {
      if *a >= (1u32 << 24) || *b >= (1u32 << 24) || *c >= (1u32 << 24) {
        return Err(VerifyError::UnexpectedProofVariant {
          expected: "u24 inputs",
        });
      }
      let total = *a + *b + (*_cin as u32);
      let sum = total & ((1u32 << 24) - 1);
      let pairs = vec![
        ((sum & 0xFF) as u8, (*c & 0xFF) as u8),
        (((sum >> 8) & 0xFF) as u8, ((*c >> 8) & 0xFF) as u8),
        (((sum >> 16) & 0xFF) as u8, ((*c >> 16) & 0xFF) as u8),
      ];
      Ok(and_wffs(
        pairs
          .into_iter()
          .map(|(lhs, rhs)| WFF::Equal(Box::new(Term::Byte(lhs)), Box::new(Term::Byte(rhs))))
          .collect(),
      ))
    }
    Proof::U15MulEq(a, b) => {
      if *a > 0x7FFF || *b > 0x7FFF {
        return Err(VerifyError::UnexpectedProofVariant {
          expected: "u15 inputs",
        });
      }
      let a_lo = (*a & 0xFF) as u8;
      let a_hi = ((*a >> 8) & 0x7F) as u8;
      let b_lo = (*b & 0xFF) as u8;
      let b_hi = ((*b >> 8) & 0x7F) as u8;

      let p00 = a_lo as u16 * b_lo as u16;
      let p01 = a_lo as u16 * b_hi as u16;
      let p10 = a_hi as u16 * b_lo as u16;
      let p11 = a_hi as u16 * b_hi as u16;

      let leaves = vec![
        WFF::Equal(
          Box::new(Term::ByteMulLow(
            Box::new(Term::Byte(a_lo)),
            Box::new(Term::Byte(b_lo)),
          )),
          Box::new(Term::Byte((p00 & 0xFF) as u8)),
        ),
        WFF::Equal(
          Box::new(Term::ByteMulHigh(
            Box::new(Term::Byte(a_lo)),
            Box::new(Term::Byte(b_lo)),
          )),
          Box::new(Term::Byte((p00 >> 8) as u8)),
        ),
        WFF::Equal(
          Box::new(Term::ByteMulLow(
            Box::new(Term::Byte(a_lo)),
            Box::new(Term::Byte(b_hi)),
          )),
          Box::new(Term::Byte((p01 & 0xFF) as u8)),
        ),
        WFF::Equal(
          Box::new(Term::ByteMulHigh(
            Box::new(Term::Byte(a_lo)),
            Box::new(Term::Byte(b_hi)),
          )),
          Box::new(Term::Byte((p01 >> 8) as u8)),
        ),
        WFF::Equal(
          Box::new(Term::ByteMulLow(
            Box::new(Term::Byte(a_hi)),
            Box::new(Term::Byte(b_lo)),
          )),
          Box::new(Term::Byte((p10 & 0xFF) as u8)),
        ),
        WFF::Equal(
          Box::new(Term::ByteMulHigh(
            Box::new(Term::Byte(a_hi)),
            Box::new(Term::Byte(b_lo)),
          )),
          Box::new(Term::Byte((p10 >> 8) as u8)),
        ),
        WFF::Equal(
          Box::new(Term::ByteMulLow(
            Box::new(Term::Byte(a_hi)),
            Box::new(Term::Byte(b_hi)),
          )),
          Box::new(Term::Byte((p11 & 0xFF) as u8)),
        ),
        WFF::Equal(
          Box::new(Term::ByteMulHigh(
            Box::new(Term::Byte(a_hi)),
            Box::new(Term::Byte(b_hi)),
          )),
          Box::new(Term::Byte((p11 >> 8) as u8)),
        ),
      ];

      Ok(and_wffs(leaves))
    }
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
        Err(VerifyError::UnexpectedProofVariant {
          expected: "equality proof",
        })
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
        Err(VerifyError::UnexpectedProofVariant {
          expected: "equality proof",
        })
      }
    }
    Proof::IteTrueEq(a, b) => {
      let ty_a = infer_ty(a)?;
      let ty_b = infer_ty(b)?;
      if ty_a != ty_b {
        return Err(VerifyError::UnexpectedTermVariant {
          expected: "matching type subterms",
        });
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
        return Err(VerifyError::UnexpectedTermVariant {
          expected: "matching type subterms",
        });
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
    Proof::ByteMulLowEq(a, b) => Ok(WFF::Equal(
      Box::new(Term::ByteMulLow(
        Box::new(Term::Byte(*a)),
        Box::new(Term::Byte(*b)),
      )),
      Box::new(Term::Byte(((*a as u16 * *b as u16) & 0xFF) as u8)),
    )),
    Proof::ByteMulHighEq(a, b) => Ok(WFF::Equal(
      Box::new(Term::ByteMulHigh(
        Box::new(Term::Byte(*a)),
        Box::new(Term::Byte(*b)),
      )),
      Box::new(Term::Byte(((*a as u16 * *b as u16) >> 8) as u8)),
    )),

    // ── Per-opcode axiom proofs: each infers the matching WFF variant. ──────
    Proof::PushAxiom => Ok(WFF::PushAxiom),
    Proof::DupAxiom { depth } => Ok(WFF::DupAxiom { depth: *depth }),
    Proof::SwapAxiom { depth } => Ok(WFF::SwapAxiom { depth: *depth }),
    Proof::StructuralAxiom { opcode } => Ok(WFF::StructuralAxiom { opcode: *opcode }),
    Proof::MloadAxiom => Ok(WFF::MloadAxiom),
    Proof::MstoreAxiom { opcode } => Ok(WFF::MstoreAxiom { opcode: *opcode }),
    Proof::MemCopyAxiom { opcode } => Ok(WFF::MemCopyAxiom { opcode: *opcode }),
    Proof::SloadAxiom => Ok(WFF::SloadAxiom),
    Proof::SstoreAxiom => Ok(WFF::SstoreAxiom),
    Proof::TransientAxiom { opcode } => Ok(WFF::TransientAxiom { opcode: *opcode }),
    Proof::KeccakAxiom => Ok(WFF::KeccakAxiom),
    Proof::EnvAxiom { opcode } => Ok(WFF::EnvAxiom { opcode: *opcode }),
    Proof::ExternalStateAxiom { opcode } => Ok(WFF::ExternalStateAxiom { opcode: *opcode }),
    Proof::TerminateAxiom { opcode } => Ok(WFF::TerminateAxiom { opcode: *opcode }),
    Proof::CallAxiom { opcode } => Ok(WFF::CallAxiom { opcode: *opcode }),
    Proof::CreateAxiom { opcode } => Ok(WFF::CreateAxiom { opcode: *opcode }),
    Proof::SelfdestructAxiom => Ok(WFF::SelfdestructAxiom),
    Proof::LogAxiom { opcode } => Ok(WFF::LogAxiom { opcode: *opcode }),
  }
}

// ============================================================
// Word-level functions
// ============================================================

/// Build a 32-element array of `InputTerm` bytes for the `stack_idx`-th stack input word.
///
/// `result[j]` = byte `j` (big-endian, 0 = MSB) of the `stack_idx`-th input.
pub fn input_word(stack_idx: u8) -> [Box<Term>; 32] {
  array::from_fn(|j| Box::new(Term::InputTerm { stack_idx, byte_idx: j as u8 }))
}

/// Build a 32-element array of `OutputTerm` bytes for the `stack_idx`-th stack output word.
pub fn output_word(stack_idx: u8) -> [Box<Term>; 32] {
  array::from_fn(|j| Box::new(Term::OutputTerm { stack_idx, byte_idx: j as u8 }))
}

pub fn word_add(a: &[u8; 32], b: &[u8; 32]) -> [Box<Term>; 32] {
  let mut carry = Box::new(Term::Bool(false));
  // Build symbolic carry chain LSB-first (byte 31 → byte 0) for big-endian layout
  let mut terms: Vec<Box<Term>> = vec![Box::new(Term::Byte(0)); 32];
  for i in (0..32).rev() {
    let ai = Box::new(Term::Byte(a[i]));
    let bi = Box::new(Term::Byte(b[i]));
    *terms[i] = Term::ByteAdd(ai.clone(), bi.clone(), carry.clone());
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
    *terms[i] = Term::ByteAdd(ai, bi, Box::new(Term::Bool(carry)));
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

pub fn word_add_with_hybrid_witness(
  a: &[u8; 32],
  b: &[u8; 32],
) -> ([Box<Term>; 32], WordAddHybridWitness) {
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

fn extract_limb_le_bits(word: &[u8; 32], base_bit: usize, width: usize) -> u32 {
  let mut limb = 0u32;
  for bit in 0..width {
    let abs_bit = base_bit + bit;
    if abs_bit >= 256 {
      break;
    }
    let byte_from_lsb = abs_bit / 8;
    let bit_in_byte = abs_bit % 8;
    let byte = word[31 - byte_from_lsb];
    let bit_val = (byte >> bit_in_byte) & 1;
    limb |= (bit_val as u32) << bit;
  }
  limb
}

pub fn verify_word_add_hybrid_witness(
  a: &[u8; 32],
  b: &[u8; 32],
  witness: &WordAddHybridWitness,
) -> bool {
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
      if let (Some(a), Some(b), Some(c)) = (
        as_byte_const(lhs.as_ref()),
        as_byte_const(rhs.as_ref()),
        as_bool_const(carry.as_ref()),
      ) {
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

      let lhs_term = acc_terms[i]
        .take()
        .unwrap_or_else(|| Box::new(Term::Byte(acc_vals[i])));
      let rhs_term = if i == idx {
        term.clone()
      } else {
        Box::new(Term::Byte(0))
      };
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
      add_term_at(
        &mut acc_terms,
        &mut acc_vals,
        low_be,
        low,
        low_val,
        &mut witness_steps,
      );

      if k + 1 < 32 {
        let high_be = 31 - (k + 1);
        add_term_at(
          &mut acc_terms,
          &mut acc_vals,
          high_be,
          high,
          high_val,
          &mut witness_steps,
        );
      }
    }
  }

  (
    array::from_fn(|i| {
      acc_terms[i]
        .clone()
        .unwrap_or_else(|| Box::new(Term::Byte(0)))
    }),
    acc_vals,
  )
}

pub fn word_mul_with_hybrid_witness(
  a: &[u8; 32],
  b: &[u8; 32],
) -> ([Box<Term>; 32], WordMulHybridWitness) {
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

#[cfg(debug_assertions)]
fn mul_u256_mod_u64_ref(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
  fn to_u64_le_words(word: &[u8; 32]) -> [u64; 4] {
    let mut out = [0u64; 4];
    for (i, out_word) in out.iter_mut().enumerate() {
      let start = 32 - ((i + 1) * 8);
      let mut chunk = [0u8; 8];
      chunk.copy_from_slice(&word[start..start + 8]);
      *out_word = u64::from_be_bytes(chunk);
    }
    out
  }

  fn from_u64_le_words(words: &[u64; 4]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, word) in words.iter().enumerate() {
      let start = 32 - ((i + 1) * 8);
      out[start..start + 8].copy_from_slice(&word.to_be_bytes());
    }
    out
  }

  let aw = to_u64_le_words(a);
  let bw = to_u64_le_words(b);
  let mut out = [0u64; 4];

  for (i, &ai) in aw.iter().enumerate() {
    let mut carry = 0u128;
    for (j, &bj) in bw.iter().enumerate() {
      let k = i + j;
      if k >= 4 {
        break;
      }
      let t = out[k] as u128 + (ai as u128) * (bj as u128) + carry;
      out[k] = t as u64;
      carry = t >> 64;
    }
  }

  from_u64_le_words(&out)
}

#[cfg(debug_assertions)]
fn debug_assert_mul_consistency(a: &[u8; 32], b: &[u8; 32], expected: &[u8; 32]) {
  let reference = mul_u256_mod_u64_ref(a, b);
  debug_assert_eq!(
    &reference, expected,
    "mul_u256_mod mismatch against debug reference implementation"
  );
}

#[cfg(debug_assertions)]
fn cmp_word_be(a: &[u8; 32], b: &[u8; 32]) -> core::cmp::Ordering {
  for i in 0..32 {
    if a[i] < b[i] {
      return core::cmp::Ordering::Less;
    }
    if a[i] > b[i] {
      return core::cmp::Ordering::Greater;
    }
  }
  core::cmp::Ordering::Equal
}

#[cfg(debug_assertions)]
fn shl1_word_be(x: &mut [u8; 32]) {
  let mut carry = 0u8;
  for i in (0..32).rev() {
    let next_carry = (x[i] >> 7) & 1;
    x[i] = (x[i] << 1) | carry;
    carry = next_carry;
  }
}

#[cfg(debug_assertions)]
fn sub_word_be_in_place(x: &mut [u8; 32], y: &[u8; 32]) {
  let mut borrow = 0u16;
  for i in (0..32).rev() {
    let xi = x[i] as u16;
    let yi = y[i] as u16;
    let rhs = yi + borrow;
    if xi >= rhs {
      x[i] = (xi - rhs) as u8;
      borrow = 0;
    } else {
      x[i] = (xi + 256 - rhs) as u8;
      borrow = 1;
    }
  }
}

#[cfg(debug_assertions)]
fn div_mod_u256_bitwise_ref(a: &[u8; 32], b: &[u8; 32]) -> Option<([u8; 32], [u8; 32])> {
  if is_zero_word(b) {
    return None;
  }

  let mut q = [0u8; 32];
  let mut r = [0u8; 32];

  for bit_idx in 0..256 {
    shl1_word_be(&mut r);
    let src_byte = bit_idx / 8;
    let src_bit = 7 - (bit_idx % 8);
    let incoming = (a[src_byte] >> src_bit) & 1;
    r[31] |= incoming;

    shl1_word_be(&mut q);
    if cmp_word_be(&r, b) != core::cmp::Ordering::Less {
      sub_word_be_in_place(&mut r, b);
      q[31] |= 1;
    }
  }

  Some((q, r))
}

#[cfg(debug_assertions)]
fn div_mod_s256_bitwise_ref(a: &[u8; 32], b: &[u8; 32]) -> Option<([u8; 32], [u8; 32])> {
  if is_zero_word(b) {
    return None;
  }

  let a_neg = is_negative_word(a);
  let b_neg = is_negative_word(b);
  let abs_a = abs_word(a);
  let abs_b = abs_word(b);
  let (q_mag, r_mag) = div_mod_u256_bitwise_ref(&abs_a, &abs_b)?;

  let q = if a_neg ^ b_neg {
    twos_complement_word(&q_mag)
  } else {
    q_mag
  };
  let r = if a_neg && !is_zero_word(&r_mag) {
    twos_complement_word(&r_mag)
  } else {
    r_mag
  };
  Some((q, r))
}

#[cfg(debug_assertions)]
fn debug_assert_div_mod_consistency(a: &[u8; 32], b: &[u8; 32], signed: bool) {
  if signed {
    let fast = div_mod_s256(a, b);
    let reference = div_mod_s256_bitwise_ref(a, b);
    debug_assert_eq!(
      fast, reference,
      "div_mod_s256 mismatch against debug reference implementation"
    );
  } else {
    let fast = div_mod_u256(a, b);
    let reference = div_mod_u256_bitwise_ref(a, b);
    debug_assert_eq!(
      fast, reference,
      "div_mod_u256 mismatch against debug reference implementation"
    );
  }
}

fn is_zero_word(x: &[u8; 32]) -> bool {
  x.iter().all(|&v| v == 0)
}

fn div_mod_u256(a: &[u8; 32], b: &[u8; 32]) -> Option<([u8; 32], [u8; 32])> {
  if is_zero_word(b) {
    return None;
  }

  let ua = U256::from_be_slice(a);
  let ub = U256::from_be_slice(b);
  let q = ua / ub;
  let r = ua % ub;
  Some((q.to_be_bytes::<32>(), r.to_be_bytes::<32>()))
}

fn is_negative_word(x: &[u8; 32]) -> bool {
  (x[0] & 0x80) != 0
}

fn twos_complement_word(x: &[u8; 32]) -> [u8; 32] {
  let mut out = [0u8; 32];
  for i in 0..32 {
    out[i] = !x[i];
  }

  let mut carry = 1u16;
  for i in (0..32).rev() {
    let sum = out[i] as u16 + carry;
    out[i] = (sum & 0xFF) as u8;
    carry = sum >> 8;
  }

  out
}

fn abs_word(x: &[u8; 32]) -> [u8; 32] {
  if is_negative_word(x) {
    twos_complement_word(x)
  } else {
    *x
  }
}

fn div_mod_s256(a: &[u8; 32], b: &[u8; 32]) -> Option<([u8; 32], [u8; 32])> {
  if is_zero_word(b) {
    return None;
  }

  let a_neg = is_negative_word(a);
  let b_neg = is_negative_word(b);

  let abs_a = abs_word(a);
  let abs_b = abs_word(b);
  let (q_mag, r_mag) = div_mod_u256(&abs_a, &abs_b).expect("non-zero divisor");

  let q = if a_neg ^ b_neg {
    twos_complement_word(&q_mag)
  } else {
    q_mag
  };
  let r = if a_neg && !is_zero_word(&r_mag) {
    twos_complement_word(&r_mag)
  } else {
    r_mag
  };

  Some((q, r))
}

pub fn verify_word_mul_hybrid_witness(
  a: &[u8; 32],
  b: &[u8; 32],
  witness: &WordMulHybridWitness,
) -> bool {
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
  let mut leaves = Vec::with_capacity(9);
  let mut carry = false;

  for limb in 0..8 {
    let base = 29 * limb;
    let a29 = extract_limb_le_bits(a, base, 29);
    let b29 = extract_limb_le_bits(b, base, 29);
    let c29 = extract_limb_le_bits(c, base, 29);
    let leaf = Proof::U29AddEq(a29, b29, carry, c29);
    leaves.push(infer_proof(&leaf).expect("u29 add leaf must infer"));

    let total = a29 + b29 + carry as u32;
    carry = total >= (1u32 << 29);
  }

  let a24 = extract_limb_le_bits(a, 232, 24);
  let b24 = extract_limb_le_bits(b, 232, 24);
  let c24 = extract_limb_le_bits(c, 232, 24);
  let top_leaf = Proof::U24AddEq(a24, b24, carry, c24);
  leaves.push(infer_proof(&top_leaf).expect("u24 add leaf must infer"));

  and_wffs(leaves)
}

pub fn wff_sub(a: &[u8; 32], b: &[u8; 32], c: &[u8; 32]) -> WFF {
  wff_add(b, c, a)
}

pub fn wff_mul(a: &[u8; 32], b: &[u8; 32], c: &[u8; 32]) -> WFF {
  let expected = mul_u256_mod(a, b);
  #[cfg(debug_assertions)]
  debug_assert_mul_consistency(a, b, &expected);

  let mut leaves: Vec<WFF> = mul_local_proof_leaves(a, b)
    .into_iter()
    .map(|leaf| infer_proof(&leaf).expect("u15 mul leaf must infer"))
    .collect();
  leaves.reserve(9);

  for limb in 0..8 {
    let base = 29 * limb;
    let expected29 = extract_limb_le_bits(&expected, base, 29);
    let c29 = extract_limb_le_bits(c, base, 29);
    let leaf = Proof::U29AddEq(expected29, 0, false, c29);
    leaves.push(infer_proof(&leaf).expect("u29 output-check leaf must infer"));
  }

  let expected24 = extract_limb_le_bits(&expected, 232, 24);
  let c24 = extract_limb_le_bits(c, 232, 24);
  let top_leaf = Proof::U24AddEq(expected24, 0, false, c24);
  leaves.push(infer_proof(&top_leaf).expect("u24 output-check leaf must infer"));

  and_wffs(leaves)
}

pub fn wff_div(a: &[u8; 32], b: &[u8; 32], c: &[u8; 32]) -> WFF {
  if is_zero_word(b) {
    let zero = [0u8; 32];
    return wff_add(&zero, &zero, c);
  }

  #[cfg(debug_assertions)]
  debug_assert_div_mod_consistency(a, b, false);

  let (_q, r) = div_mod_u256(a, b).expect("checked non-zero divisor");
  let bc = mul_u256_mod(b, c);
  WFF::And(Box::new(wff_mul(b, c, &bc)), Box::new(wff_add(&bc, &r, a)))
}

pub fn wff_mod(a: &[u8; 32], b: &[u8; 32], r: &[u8; 32]) -> WFF {
  if is_zero_word(b) {
    let zero = [0u8; 32];
    return wff_add(&zero, &zero, r);
  }

  #[cfg(debug_assertions)]
  debug_assert_div_mod_consistency(a, b, false);

  let (q, _r) = div_mod_u256(a, b).expect("checked non-zero divisor");
  let bq = mul_u256_mod(b, &q);
  WFF::And(Box::new(wff_mul(b, &q, &bq)), Box::new(wff_add(&bq, r, a)))
}

pub fn wff_sdiv(a: &[u8; 32], b: &[u8; 32], c: &[u8; 32]) -> WFF {
  if is_zero_word(b) {
    let zero = [0u8; 32];
    return wff_add(&zero, &zero, c);
  }

  #[cfg(debug_assertions)]
  debug_assert_div_mod_consistency(a, b, true);

  let (_q, r) = div_mod_s256(a, b).expect("checked non-zero divisor");
  let bc = mul_u256_mod(b, c);
  WFF::And(Box::new(wff_mul(b, c, &bc)), Box::new(wff_add(&bc, &r, a)))
}

pub fn wff_smod(a: &[u8; 32], b: &[u8; 32], r: &[u8; 32]) -> WFF {
  if is_zero_word(b) {
    let zero = [0u8; 32];
    return wff_add(&zero, &zero, r);
  }

  #[cfg(debug_assertions)]
  debug_assert_div_mod_consistency(a, b, true);

  let (q, _r) = div_mod_s256(a, b).expect("checked non-zero divisor");
  let bq = mul_u256_mod(b, &q);
  WFF::And(Box::new(wff_mul(b, &q, &bq)), Box::new(wff_add(&bq, r, a)))
}

pub fn wff_and(a: &[u8; 32], b: &[u8; 32], c: &[u8; 32]) -> WFF {
  let mut leaves = Vec::with_capacity(32);
  for i in (0..32).rev() {
    leaves.push(WFF::Equal(
      Box::new(Term::ByteAnd(
        Box::new(Term::Byte(a[i])),
        Box::new(Term::Byte(b[i])),
      )),
      Box::new(Term::Byte(c[i])),
    ));
  }
  and_wffs(leaves)
}

pub fn wff_or(a: &[u8; 32], b: &[u8; 32], c: &[u8; 32]) -> WFF {
  let mut leaves = Vec::with_capacity(32);
  for i in (0..32).rev() {
    leaves.push(WFF::Equal(
      Box::new(Term::ByteOr(
        Box::new(Term::Byte(a[i])),
        Box::new(Term::Byte(b[i])),
      )),
      Box::new(Term::Byte(c[i])),
    ));
  }
  and_wffs(leaves)
}

pub fn wff_xor(a: &[u8; 32], b: &[u8; 32], c: &[u8; 32]) -> WFF {
  let mut leaves = Vec::with_capacity(32);
  for i in (0..32).rev() {
    leaves.push(WFF::Equal(
      Box::new(Term::ByteXor(
        Box::new(Term::Byte(a[i])),
        Box::new(Term::Byte(b[i])),
      )),
      Box::new(Term::Byte(c[i])),
    ));
  }
  and_wffs(leaves)
}

pub fn wff_not(a: &[u8; 32], c: &[u8; 32]) -> WFF {
  let mut leaves = Vec::with_capacity(32);
  for i in (0..32).rev() {
    leaves.push(WFF::Equal(
      Box::new(Term::ByteXor(
        Box::new(Term::Byte(a[i])),
        Box::new(Term::Byte(0xFF)),
      )),
      Box::new(Term::Byte(c[i])),
    ));
  }
  and_wffs(leaves)
}

// ============================================================
// Comparison helpers
// ============================================================

/// The EVM boolean true word: 0x000...001.
pub const ONE_WORD: [u8; 32] = {
  let mut w = [0u8; 32];
  w[31] = 1;
  w
};

pub const ZERO_WORD: [u8; 32] = [0u8; 32];

/// Big-endian unsigned compare for 256-bit words.
fn compare_u256_be(a: &[u8; 32], b: &[u8; 32]) -> std::cmp::Ordering {
  for i in 0..32 {
    if a[i] < b[i] {
      return std::cmp::Ordering::Less;
    }
    if a[i] > b[i] {
      return std::cmp::Ordering::Greater;
    }
  }
  std::cmp::Ordering::Equal
}

/// Compute `a - b` (big-endian, wrapping on underflow like u256 subtraction).
fn sub_u256_be(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
  let mut out = [0u8; 32];
  let mut borrow: u16 = 0;
  for i in (0..32).rev() {
    let ai = a[i] as u16;
    let bi = b[i] as u16 + borrow;
    if ai >= bi {
      out[i] = (ai - bi) as u8;
      borrow = 0;
    } else {
      out[i] = (ai + 256 - bi) as u8;
      borrow = 1;
    }
  }
  out
}

/// The sign-bit mask for two's-complement signed comparison: [0x80, 0, ..., 0].
fn sign_mask() -> [u8; 32] {
  let mut m = [0u8; 32];
  m[0] = 0x80;
  m
}

/// Flip the MSB of a 256-bit big-endian word (used for signed→unsigned reduction).
fn flip_sign_bit(x: &[u8; 32]) -> [u8; 32] {
  let mut out = *x;
  out[0] ^= 0x80;
  out
}

// ============================================================
// Comparison / equality WFFs
//
// Design principle:
//   wff_cmp(inputs..., c) = AND(arithmetic_wff, output_binding_wff)
//   where output_binding_wff = wff_xor(c, expected_c, ZERO_WORD)
//   This is sound: the inferred XOR of c vs expected_c must be zero,
//   so c == expected_c is enforced by the WFF equality check.
// ============================================================

/// WFF for EQ(a, b) = c.
///
/// Structure:
///   AND(wff_xor(a, b, a^b),            // proves each byte XOR
///       wff_xor(c, expected_c, ZERO))   // pins output c to the correct boolean
pub fn wff_eq(a: &[u8; 32], b: &[u8; 32], c: &[u8; 32]) -> WFF {
  let mut xor_result = [0u8; 32];
  for i in 0..32 {
    xor_result[i] = a[i] ^ b[i];
  }
  let expected_c = if xor_result == ZERO_WORD {
    ONE_WORD
  } else {
    ZERO_WORD
  };
  WFF::And(
    Box::new(wff_xor(a, b, &xor_result)),
    Box::new(wff_xor(c, &expected_c, &ZERO_WORD)),
  )
}

/// WFF for ISZERO(a) = c.  Reduces to EQ(a, 0) = c.
pub fn wff_iszero(a: &[u8; 32], c: &[u8; 32]) -> WFF {
  wff_eq(a, &ZERO_WORD, c)
}

/// WFF for LT(a, b) = c  (unsigned).
///
/// Arithmetic witness:
///   a < b → prove  b = a + (b−a)   via wff_sub(b, a, b−a)
///   a ≥ b → prove  a = b + (a−b)   via wff_sub(a, b, a−b)
pub fn wff_lt(a: &[u8; 32], b: &[u8; 32], c: &[u8; 32]) -> WFF {
  let a_lt_b = compare_u256_be(a, b) == std::cmp::Ordering::Less;
  let expected_c = if a_lt_b { ONE_WORD } else { ZERO_WORD };

  let arith_wff = if a_lt_b {
    let diff = sub_u256_be(b, a);
    wff_sub(b, a, &diff)
  } else {
    let diff = sub_u256_be(a, b);
    wff_sub(a, b, &diff)
  };

  WFF::And(
    Box::new(arith_wff),
    Box::new(wff_xor(c, &expected_c, &ZERO_WORD)),
  )
}

/// WFF for GT(a, b) = c.  Reduces to LT(b, a) = c.
pub fn wff_gt(a: &[u8; 32], b: &[u8; 32], c: &[u8; 32]) -> WFF {
  wff_lt(b, a, c)
}

/// WFF for SLT(a, b) = c  (signed, two's-complement).
///
/// Reduction: flip the sign bit of both operands, then do unsigned LT.
/// This works because: a <_s b  ↔  (a ^ (1<<255)) <_u (b ^ (1<<255)).
pub fn wff_slt(a: &[u8; 32], b: &[u8; 32], c: &[u8; 32]) -> WFF {
  let sm = sign_mask();
  let a_adj = flip_sign_bit(a);
  let b_adj = flip_sign_bit(b);

  WFF::And(
    Box::new(WFF::And(
      Box::new(wff_xor(a, &sm, &a_adj)),
      Box::new(wff_xor(b, &sm, &b_adj)),
    )),
    Box::new(wff_lt(&a_adj, &b_adj, c)),
  )
}

/// WFF for SGT(a, b) = c.  Reduces to SLT(b, a) = c.
pub fn wff_sgt(a: &[u8; 32], b: &[u8; 32], c: &[u8; 32]) -> WFF {
  wff_slt(b, a, c)
}

// ============================================================
// Proving Instructions
// ============================================================

pub fn prove_add(a: &[u8; 32], b: &[u8; 32], _c: &[u8; 32]) -> Proof {
  let mut leaves = Vec::with_capacity(9);
  let mut carry = false;

  for limb in 0..8 {
    let base = 29 * limb;
    let a29 = extract_limb_le_bits(a, base, 29);
    let b29 = extract_limb_le_bits(b, base, 29);
    let total = a29 + b29 + carry as u32;
    let c29 = total & ((1u32 << 29) - 1);
    leaves.push(Proof::U29AddEq(a29, b29, carry, c29));
    carry = total >= (1u32 << 29);
  }

  let a24 = extract_limb_le_bits(a, 232, 24);
  let b24 = extract_limb_le_bits(b, 232, 24);
  let total_top = a24 + b24 + carry as u32;
  let c24 = total_top & ((1u32 << 24) - 1);
  leaves.push(Proof::U24AddEq(a24, b24, carry, c24));

  and_proofs(leaves)
}

pub fn prove_sub(a: &[u8; 32], b: &[u8; 32], c: &[u8; 32]) -> Proof {
  prove_add(b, c, a)
}

fn and_proofs(mut leaves: Vec<Proof>) -> Proof {
  let first = leaves.remove(0);
  leaves
    .into_iter()
    .fold(first, |acc, p| Proof::AndIntro(Box::new(acc), Box::new(p)))
}

fn and_wffs(mut leaves: Vec<WFF>) -> WFF {
  let first = leaves.remove(0);
  leaves
    .into_iter()
    .fold(first, |acc, w| WFF::And(Box::new(acc), Box::new(w)))
}

pub fn prove_mul(a: &[u8; 32], b: &[u8; 32], _c: &[u8; 32]) -> Proof {
  let expected = mul_u256_mod(a, b);
  #[cfg(debug_assertions)]
  debug_assert_mul_consistency(a, b, &expected);

  let mut leaves = mul_local_proof_leaves(a, b);
  leaves.reserve(9);

  for limb in 0..8 {
    let base = 29 * limb;
    let expected29 = extract_limb_le_bits(&expected, base, 29);
    leaves.push(Proof::U29AddEq(expected29, 0, false, expected29));
  }

  let expected24 = extract_limb_le_bits(&expected, 232, 24);
  leaves.push(Proof::U24AddEq(expected24, 0, false, expected24));

  and_proofs(leaves)
}

fn word_to_u15_limbs(word: &[u8; 32]) -> [u16; 18] {
  let mut limbs = [0u16; 18];
  for (limb_idx, limb_slot) in limbs.iter_mut().enumerate() {
    let mut limb = 0u16;
    let base_bit = limb_idx * 15;
    for bit in 0..15 {
      let abs_bit = base_bit + bit;
      if abs_bit >= 256 {
        break;
      }
      let byte_from_lsb = abs_bit / 8;
      let bit_in_byte = abs_bit % 8;
      let byte = word[31 - byte_from_lsb];
      let bit_val = (byte >> bit_in_byte) & 1;
      limb |= (bit_val as u16) << bit;
    }
    *limb_slot = limb;
  }
  limbs
}

fn mul_local_proof_leaves(a: &[u8; 32], b: &[u8; 32]) -> Vec<Proof> {
  let mut leaves = Vec::new();
  let a_limbs = word_to_u15_limbs(a);
  let b_limbs = word_to_u15_limbs(b);

  for (i, &av) in a_limbs.iter().enumerate() {
    for (j, &bv) in b_limbs.iter().enumerate() {
      if i + j > 17 {
        continue;
      }
      if av == 0 || bv == 0 {
        continue;
      }
      leaves.push(Proof::U15MulEq(av, bv));
    }
  }

  leaves
}

pub fn prove_div(a: &[u8; 32], b: &[u8; 32], c: &[u8; 32]) -> Proof {
  if is_zero_word(b) {
    let zero = [0u8; 32];
    return prove_add(&zero, &zero, c);
  }

  #[cfg(debug_assertions)]
  debug_assert_div_mod_consistency(a, b, false);

  let (_q, r) = div_mod_u256(a, b).expect("checked non-zero divisor");
  let bc = mul_u256_mod(b, c);
  let p_mul = prove_mul(b, c, &bc);
  let p_add = prove_add(&bc, &r, a);
  Proof::AndIntro(Box::new(p_mul), Box::new(p_add))
}

pub fn prove_mod(a: &[u8; 32], b: &[u8; 32], r: &[u8; 32]) -> Proof {
  if is_zero_word(b) {
    let zero = [0u8; 32];
    return prove_add(&zero, &zero, r);
  }

  #[cfg(debug_assertions)]
  debug_assert_div_mod_consistency(a, b, false);

  let (q, _r) = div_mod_u256(a, b).expect("checked non-zero divisor");
  let bq = mul_u256_mod(b, &q);
  let p_mul = prove_mul(b, &q, &bq);
  let p_add = prove_add(&bq, r, a);
  Proof::AndIntro(Box::new(p_mul), Box::new(p_add))
}

pub fn prove_sdiv(a: &[u8; 32], b: &[u8; 32], c: &[u8; 32]) -> Proof {
  if is_zero_word(b) {
    let zero = [0u8; 32];
    return prove_add(&zero, &zero, c);
  }

  #[cfg(debug_assertions)]
  debug_assert_div_mod_consistency(a, b, true);

  let (_q, r) = div_mod_s256(a, b).expect("checked non-zero divisor");
  let bc = mul_u256_mod(b, c);
  let p_mul = prove_mul(b, c, &bc);
  let p_add = prove_add(&bc, &r, a);
  Proof::AndIntro(Box::new(p_mul), Box::new(p_add))
}

pub fn prove_smod(a: &[u8; 32], b: &[u8; 32], r: &[u8; 32]) -> Proof {
  if is_zero_word(b) {
    let zero = [0u8; 32];
    return prove_add(&zero, &zero, r);
  }

  #[cfg(debug_assertions)]
  debug_assert_div_mod_consistency(a, b, true);

  let (q, _r) = div_mod_s256(a, b).expect("checked non-zero divisor");
  let bq = mul_u256_mod(b, &q);
  let p_mul = prove_mul(b, &q, &bq);
  let p_add = prove_add(&bq, r, a);
  Proof::AndIntro(Box::new(p_mul), Box::new(p_add))
}

pub fn prove_and(a: &[u8; 32], b: &[u8; 32], _c: &[u8; 32]) -> Proof {
  let mut leaves = Vec::with_capacity(32);
  for i in (0..32).rev() {
    leaves.push(Proof::ByteAndEq(a[i], b[i]));
  }
  and_proofs(leaves)
}

pub fn prove_or(a: &[u8; 32], b: &[u8; 32], _c: &[u8; 32]) -> Proof {
  let mut leaves = Vec::with_capacity(32);
  for i in (0..32).rev() {
    leaves.push(Proof::ByteOrEq(a[i], b[i]));
  }
  and_proofs(leaves)
}

pub fn prove_xor(a: &[u8; 32], b: &[u8; 32], _c: &[u8; 32]) -> Proof {
  let mut leaves = Vec::with_capacity(32);
  for i in (0..32).rev() {
    leaves.push(Proof::ByteXorEq(a[i], b[i]));
  }
  and_proofs(leaves)
}

pub fn prove_not(a: &[u8; 32], _c: &[u8; 32]) -> Proof {
  let mut leaves = Vec::with_capacity(32);
  for i in (0..32).rev() {
    leaves.push(Proof::ByteXorEq(a[i], 0xFF));
  }
  and_proofs(leaves)
}

// ============================================================
// Shift helpers
// ============================================================

/// Decompose a U256 shift amount into (byte_shift n, bit_shift m).
///
/// EVM spec: if shift >= 256, the result is 0 (SHL/SHR) or all-ones (SAR neg).
/// Returns `None` when shift >= 256 (caller handles the overflow case).
fn decompose_shift(shift: &[u8; 32]) -> Option<(usize, u8)> {
  if shift[..31].iter().any(|&b| b != 0) {
    return None; // shift >= 256
  }
  let s = shift[31];
  Some(((s / 8) as usize, s % 8))
}

/// Byte-aligned intermediate for SHL: mid[k] = value[k + n] (or 0).
fn shl_mid(value: &[u8; 32], n: usize) -> [u8; 32] {
  let mut mid = [0u8; 32];
  for k in 0..32usize {
    mid[k] = if k + n < 32 { value[k + n] } else { 0 };
  }
  mid
}

/// Byte-aligned intermediate for SHR: mid[k] = value[k - n] (or 0).
fn shr_mid(value: &[u8; 32], n: usize) -> [u8; 32] {
  let mut mid = [0u8; 32];
  for k in 0..32usize {
    mid[k] = if k >= n { value[k - n] } else { 0 };
  }
  mid
}

// ============================================================
// WFF for shift operations  (uses U15MulEq)
//
// Key identities for m ∈ 1..=7, factor = (1 << m):
//   a << m  =  (a * factor) & 0xFF   → U15MulEq low-half
//   a >> m  =  (a * (256 / factor)) >> 8
//            = (a * (1 << inv)) >> 8  → U15MulEq high-half  (inv = 8-m)
//
// Both fit within 15-bit × 8-bit ≤ 15-bit products because a ≤ 255 < 2^8
// and factor ≤ 128 < 2^8, so a * factor ≤ 255 * 128 = 32640 < 2^15.
// ============================================================

// ── internal WFF leaf helpers ─────────────────────────────────────────

/// WFF leaf: ByteMulLow(a, factor) = lo.
fn wff_mul_lo(a: u8, factor: u8, lo: u8) -> WFF {
  WFF::Equal(
    Box::new(Term::ByteMulLow(
      Box::new(Term::Byte(a)),
      Box::new(Term::Byte(factor)),
    )),
    Box::new(Term::Byte(lo)),
  )
}

/// WFF leaf: ByteMulHigh(a, factor) = hi.
fn wff_mul_hi(a: u8, factor: u8, hi: u8) -> WFF {
  WFF::Equal(
    Box::new(Term::ByteMulHigh(
      Box::new(Term::Byte(a)),
      Box::new(Term::Byte(factor)),
    )),
    Box::new(Term::Byte(hi)),
  )
}

/// WFF for `SHL(shift, value) = result`.
pub fn wff_shl(shift: &[u8; 32], value: &[u8; 32], result: &[u8; 32]) -> WFF {
  match decompose_shift(shift) {
    None => wff_xor(result, &ZERO_WORD, &ZERO_WORD),
    Some((n, 0)) => {
      let mid = shl_mid(value, n);
      let leaves: Vec<WFF> = (0..32)
        .rev()
        .map(|k| {
          WFF::Equal(
            Box::new(Term::ByteXor(
              Box::new(Term::Byte(mid[k])),
              Box::new(Term::Byte(0x00)),
            )),
            Box::new(Term::Byte(result[k])),
          )
        })
        .collect();
      and_wffs(leaves)
    }
    Some((n, m)) => {
      let mid = shl_mid(value, n);
      let factor: u8 = 1 << m; // multiply by 2^m
      let mut leaves = Vec::with_capacity(32 * 3);
      for k in 0..32usize {
        let next = if k < 31 { mid[k + 1] } else { 0 };
        // lo = (mid[k] * 2^m) & 0xFF  = mid[k] << m
        let lo = ((mid[k] as u16 * factor as u16) & 0xFF) as u8;
        // hi = (next * 2^m) >> 8      = next >> (8-m) = next >> inv
        let hi = ((next as u16 * factor as u16) >> 8) as u8;
        leaves.push(wff_mul_lo(mid[k], factor, lo));
        leaves.push(wff_mul_hi(next, factor, hi));
        leaves.push(WFF::Equal(
          Box::new(Term::ByteOr(
            Box::new(Term::Byte(lo)),
            Box::new(Term::Byte(hi)),
          )),
          Box::new(Term::Byte(result[k])),
        ));
      }
      and_wffs(leaves)
    }
  }
}

/// WFF for `SHR(shift, value) = result` (logical).
pub fn wff_shr(shift: &[u8; 32], value: &[u8; 32], result: &[u8; 32]) -> WFF {
  match decompose_shift(shift) {
    None => wff_xor(result, &ZERO_WORD, &ZERO_WORD),
    Some((n, 0)) => {
      let mid = shr_mid(value, n);
      let leaves: Vec<WFF> = (0..32)
        .rev()
        .map(|k| {
          WFF::Equal(
            Box::new(Term::ByteXor(
              Box::new(Term::Byte(mid[k])),
              Box::new(Term::Byte(0x00)),
            )),
            Box::new(Term::Byte(result[k])),
          )
        })
        .collect();
      and_wffs(leaves)
    }
    Some((n, m)) => {
      let mid = shr_mid(value, n);
      // For SHR by m: a >> m = (a * 2^(8-m)) >> 8
      let inv: u8 = 8 - m;
      let factor: u8 = 1 << inv; // multiply by 2^inv then take high byte
      let mut leaves = Vec::with_capacity(32 * 3);
      for k in 0..32usize {
        let prev = if k > 0 { mid[k - 1] } else { 0 };
        // lo = mid[k] >> m = (mid[k] * 2^inv) >> 8
        let lo = ((mid[k] as u16 * factor as u16) >> 8) as u8;
        // hi = prev << inv = (prev * 2^inv) & 0xFF
        let hi = ((prev as u16 * factor as u16) & 0xFF) as u8;
        leaves.push(wff_mul_hi(mid[k], factor, lo));
        leaves.push(wff_mul_lo(prev, factor, hi));
        leaves.push(WFF::Equal(
          Box::new(Term::ByteOr(
            Box::new(Term::Byte(lo)),
            Box::new(Term::Byte(hi)),
          )),
          Box::new(Term::Byte(result[k])),
        ));
      }
      and_wffs(leaves)
    }
  }
}

/// WFF for `SAR(shift, value) = result` (arithmetic).
pub fn wff_sar(shift: &[u8; 32], value: &[u8; 32], result: &[u8; 32]) -> WFF {
  let sign_fill = if (value[0] & 0x80) != 0 { 0xFFu8 } else { 0u8 };
  match decompose_shift(shift) {
    None => {
      let fill_word = [sign_fill; 32];
      wff_xor(result, &fill_word, &ZERO_WORD)
    }
    Some((n, 0)) => {
      let mut mid = [sign_fill; 32];
      for k in 0..32usize {
        if k >= n {
          mid[k] = value[k - n];
        }
      }
      let leaves: Vec<WFF> = (0..32)
        .rev()
        .map(|k| {
          WFF::Equal(
            Box::new(Term::ByteXor(
              Box::new(Term::Byte(mid[k])),
              Box::new(Term::Byte(0x00)),
            )),
            Box::new(Term::Byte(result[k])),
          )
        })
        .collect();
      and_wffs(leaves)
    }
    Some((n, m)) => {
      let mut mid = [sign_fill; 32];
      for k in 0..32usize {
        if k >= n {
          mid[k] = value[k - n];
        }
      }
      let inv: u8 = 8 - m;
      let factor_shr: u8 = 1 << inv; // used for logical right-shift via mul-high
      let factor_shl: u8 = 1 << inv; // same value, used for left-shift via mul-low
      let fill_mask: u8 = (0xFF00u16 >> m) as u8; // sign bits to OR in for negative k=0
      let mut leaves = Vec::new();
      for k in 0..32usize {
        let prev = if k > 0 { mid[k - 1] } else { sign_fill };
        let lo = ((mid[k] as u16 * factor_shr as u16) >> 8) as u8; // logical SHR
        let hi = ((prev as u16 * factor_shl as u16) & 0xFF) as u8; // prev << inv
        if k == 0 {
          // For the top byte, sign-extend: OR in fill_mask to restore sign bits set to 1.
          let sar_val = ((mid[0] as i8) >> m) as u8;
          let shr_val = ((mid[0] as u16 * factor_shr as u16) >> 8) as u8;
          leaves.push(wff_mul_hi(mid[0], factor_shr, shr_val));
          if sign_fill == 0xFF {
            // negative: OR the shifted byte with fill_mask
            leaves.push(WFF::Equal(
              Box::new(Term::ByteOr(
                Box::new(Term::Byte(shr_val)),
                Box::new(Term::Byte(fill_mask)),
              )),
              Box::new(Term::Byte(sar_val)),
            ));
          }
          // hi comes from virtual sign_fill byte left-shifted
          leaves.push(wff_mul_lo(sign_fill, factor_shl, hi));
          leaves.push(WFF::Equal(
            Box::new(Term::ByteOr(
              Box::new(Term::Byte(sar_val)),
              Box::new(Term::Byte(hi)),
            )),
            Box::new(Term::Byte(result[k])),
          ));
        } else {
          leaves.push(wff_mul_hi(mid[k], factor_shr, lo));
          leaves.push(wff_mul_lo(prev, factor_shl, hi));
          leaves.push(WFF::Equal(
            Box::new(Term::ByteOr(
              Box::new(Term::Byte(lo)),
              Box::new(Term::Byte(hi)),
            )),
            Box::new(Term::Byte(result[k])),
          ));
        }
      }
      and_wffs(leaves)
    }
  }
}

// ============================================================
// Prove functions for shift operations  (uses U15MulEq)
// ============================================================

/// Prove `SHL(shift, value) = result`.
pub fn prove_shl(shift: &[u8; 32], value: &[u8; 32], result: &[u8; 32]) -> Proof {
  match decompose_shift(shift) {
    None => prove_xor(result, &ZERO_WORD, &ZERO_WORD),
    Some((n, 0)) => {
      let mid = shl_mid(value, n);
      let leaves: Vec<Proof> = (0..32)
        .rev()
        .map(|k| Proof::ByteXorEq(mid[k], 0x00))
        .collect();
      and_proofs(leaves)
    }
    Some((n, m)) => {
      let mid = shl_mid(value, n);
      let factor: u8 = 1 << m;
      let mut leaves = Vec::with_capacity(32 * 3);
      for k in 0..32usize {
        let next = if k < 31 { mid[k + 1] } else { 0 };
        let lo = ((mid[k] as u16 * factor as u16) & 0xFF) as u8;
        let hi = ((next as u16 * factor as u16) >> 8) as u8;
        leaves.push(Proof::ByteMulLowEq(mid[k], factor));
        leaves.push(Proof::ByteMulHighEq(next, factor));
        leaves.push(Proof::ByteOrEq(lo, hi));
      }
      and_proofs(leaves)
    }
  }
}

/// Prove `SHR(shift, value) = result` (logical).
pub fn prove_shr(shift: &[u8; 32], value: &[u8; 32], result: &[u8; 32]) -> Proof {
  match decompose_shift(shift) {
    None => prove_xor(result, &ZERO_WORD, &ZERO_WORD),
    Some((n, 0)) => {
      let mid = shr_mid(value, n);
      let leaves: Vec<Proof> = (0..32)
        .rev()
        .map(|k| Proof::ByteXorEq(mid[k], 0x00))
        .collect();
      and_proofs(leaves)
    }
    Some((n, m)) => {
      let mid = shr_mid(value, n);
      let inv: u8 = 8 - m;
      let factor: u8 = 1 << inv;
      let mut leaves = Vec::with_capacity(32 * 3);
      for k in 0..32usize {
        let prev = if k > 0 { mid[k - 1] } else { 0 };
        let lo = ((mid[k] as u16 * factor as u16) >> 8) as u8;
        let hi = ((prev as u16 * factor as u16) & 0xFF) as u8;
        leaves.push(Proof::ByteMulHighEq(mid[k], factor));
        leaves.push(Proof::ByteMulLowEq(prev, factor));
        leaves.push(Proof::ByteOrEq(lo, hi));
      }
      and_proofs(leaves)
    }
  }
}

/// Prove `SAR(shift, value) = result` (arithmetic).
pub fn prove_sar(shift: &[u8; 32], value: &[u8; 32], result: &[u8; 32]) -> Proof {
  let sign_fill = if (value[0] & 0x80) != 0 { 0xFFu8 } else { 0u8 };
  match decompose_shift(shift) {
    None => {
      let fill_word = [sign_fill; 32];
      prove_xor(result, &fill_word, &ZERO_WORD)
    }
    Some((n, 0)) => {
      let mut mid = [sign_fill; 32];
      for k in 0..32usize {
        if k >= n {
          mid[k] = value[k - n];
        }
      }
      let leaves: Vec<Proof> = (0..32)
        .rev()
        .map(|k| Proof::ByteXorEq(mid[k], 0x00))
        .collect();
      and_proofs(leaves)
    }
    Some((n, m)) => {
      let mut mid = [sign_fill; 32];
      for k in 0..32usize {
        if k >= n {
          mid[k] = value[k - n];
        }
      }
      let inv: u8 = 8 - m;
      let factor: u8 = 1 << inv;
      let fill_mask: u8 = (0xFF00u16 >> m) as u8;
      let mut leaves = Vec::new();
      for k in 0..32usize {
        let prev = if k > 0 { mid[k - 1] } else { sign_fill };
        let hi = ((prev as u16 * factor as u16) & 0xFF) as u8;
        if k == 0 {
          let shr_val = ((mid[0] as u16 * factor as u16) >> 8) as u8;
          let sar_val = ((mid[0] as i8) >> m) as u8;
          leaves.push(Proof::ByteMulHighEq(mid[0], factor));
          if sign_fill == 0xFF {
            leaves.push(Proof::ByteOrEq(shr_val, fill_mask));
          }
          leaves.push(Proof::ByteMulLowEq(sign_fill, factor));
          leaves.push(Proof::ByteOrEq(sar_val, hi));
        } else {
          let lo = ((mid[k] as u16 * factor as u16) >> 8) as u8;
          leaves.push(Proof::ByteMulHighEq(mid[k], factor));
          leaves.push(Proof::ByteMulLowEq(prev, factor));
          leaves.push(Proof::ByteOrEq(lo, hi));
        }
      }
      and_proofs(leaves)
    }
  }
}

/// Prove EQ(a, b) = c.
///
/// Part 1: ByteXorEq for each byte of a vs b.
/// Part 2: ByteXorEq for each byte of c vs expected_c (pins the output).
pub fn prove_eq(a: &[u8; 32], b: &[u8; 32], c: &[u8; 32]) -> Proof {
  let mut xor_result = [0u8; 32];
  for i in 0..32 {
    xor_result[i] = a[i] ^ b[i];
  }
  let expected_c = if xor_result == ZERO_WORD {
    ONE_WORD
  } else {
    ZERO_WORD
  };
  let p_xor = prove_xor(a, b, &xor_result);
  let p_output = prove_xor(c, &expected_c, &ZERO_WORD);
  Proof::AndIntro(Box::new(p_xor), Box::new(p_output))
}

/// Prove ISZERO(a) = c.  Reduces to EQ(a, 0) = c.
pub fn prove_iszero(a: &[u8; 32], c: &[u8; 32]) -> Proof {
  prove_eq(a, &ZERO_WORD, c)
}

/// Prove LT(a, b) = c  (unsigned).
///
/// Arithmetic witness: b = a + diff  (if a < b) or  a = b + diff  (if a ≥ b).
/// Output pin: ByteXorEq chain proving c == expected_c.
pub fn prove_lt(a: &[u8; 32], b: &[u8; 32], c: &[u8; 32]) -> Proof {
  let a_lt_b = compare_u256_be(a, b) == std::cmp::Ordering::Less;
  let expected_c = if a_lt_b { ONE_WORD } else { ZERO_WORD };

  let p_arith = if a_lt_b {
    let diff = sub_u256_be(b, a);
    prove_sub(b, a, &diff)
  } else {
    let diff = sub_u256_be(a, b);
    prove_sub(a, b, &diff)
  };

  let p_output = prove_xor(c, &expected_c, &ZERO_WORD);
  Proof::AndIntro(Box::new(p_arith), Box::new(p_output))
}

/// Prove GT(a, b) = c.  Reduces to LT(b, a) = c.
pub fn prove_gt(a: &[u8; 32], b: &[u8; 32], c: &[u8; 32]) -> Proof {
  prove_lt(b, a, c)
}

/// Prove SLT(a, b) = c  (signed).
///
/// Flip the MSB of both operands (sign-extension trick) then prove unsigned LT.
pub fn prove_slt(a: &[u8; 32], b: &[u8; 32], c: &[u8; 32]) -> Proof {
  let sm = sign_mask();
  let a_adj = flip_sign_bit(a);
  let b_adj = flip_sign_bit(b);
  let p_flip_a = prove_xor(a, &sm, &a_adj);
  let p_flip_b = prove_xor(b, &sm, &b_adj);
  let p_lt = prove_lt(&a_adj, &b_adj, c);
  Proof::AndIntro(
    Box::new(Proof::AndIntro(Box::new(p_flip_a), Box::new(p_flip_b))),
    Box::new(p_lt),
  )
}

/// Prove SGT(a, b) = c.  Reduces to SLT(b, a) = c.
pub fn prove_sgt(a: &[u8; 32], b: &[u8; 32], c: &[u8; 32]) -> Proof {
  prove_slt(b, a, c)
}

// ============================================================
// Arithmetic helpers for output-binding proofs
// ============================================================

/// Compute `(a + b) mod n`, where `a, b < n`, handling the 257-bit intermediate.
///
/// Since `a, b < n`, the sum is at most `2n - 2 < 2^257`.
/// If it overflows 256 bits, the quotient is always 1, so the result is
/// `sum + 2^256 - n` (which fits in 256 bits because `sum < n`).
fn addmod_inner(a: U256, b: U256, n: U256) -> U256 {
  let (sum, overflow) = a.overflowing_add(b);
  if overflow {
    // a + b = sum + 2^256 ≥ 2^256 ≥ n, so quotient = 1.
    // result = sum + 2^256 - n  (fits since sum < n)
    sum.wrapping_add(U256::MAX.wrapping_sub(n).wrapping_add(U256::from(1u64)))
  } else if sum >= n {
    sum - n
  } else {
    sum
  }
}

/// `ADDMOD(a, b, n) = (a + b) mod n`.  Returns `0` when `n = 0`.
pub fn compute_addmod(a: &[u8; 32], b: &[u8; 32], n: &[u8; 32]) -> [u8; 32] {
  if is_zero_word(n) {
    return ZERO_WORD;
  }
  let ua = U256::from_be_slice(a);
  let ub = U256::from_be_slice(b);
  let un = U256::from_be_slice(n);
  // Reduce both inputs first so addmod_inner precondition (< n) holds.
  addmod_inner(ua % un, ub % un, un).to_be_bytes::<32>()
}

/// `MULMOD(a, b, n) = (a * b) mod n`.  Returns `0` when `n = 0`.
///
/// Implemented via binary (Russian peasant) multiplication to avoid a 512-bit
/// intermediate; each doubling step uses `addmod_inner`.
pub fn compute_mulmod(a: &[u8; 32], b: &[u8; 32], n: &[u8; 32]) -> [u8; 32] {
  if is_zero_word(n) {
    return ZERO_WORD;
  }
  let ua = U256::from_be_slice(a);
  let ub = U256::from_be_slice(b);
  let un = U256::from_be_slice(n);
  let mut result = U256::ZERO;
  let mut base = ua % un;
  let mut exp = ub;
  while exp > U256::ZERO {
    if exp.bit(0) {
      result = addmod_inner(result, base, un);
    }
    base = addmod_inner(base, base, un);
    exp >>= 1;
  }
  result.to_be_bytes::<32>()
}

/// `EXP(a, b) = a ** b mod 2^256`.  Uses binary exponentiation with wrapping mul.
pub fn compute_exp(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
  let ua = U256::from_be_slice(a);
  let ub = U256::from_be_slice(b);
  let mut result = U256::from(1u64);
  let mut base = ua;
  let mut exp = ub;
  while exp > U256::ZERO {
    if exp.bit(0) {
      result = result.wrapping_mul(base);
    }
    base = base.wrapping_mul(base);
    exp >>= 1;
  }
  result.to_be_bytes::<32>()
}

/// `BYTE(i, x)`: the `i`-th byte of `x` (0 = MSB), zero-padded to 32 bytes.
/// Returns `0` when `i >= 32`.
pub fn compute_byte(i: &[u8; 32], x: &[u8; 32]) -> [u8; 32] {
  if i[..31].iter().any(|&v| v != 0) || i[31] >= 32 {
    return ZERO_WORD;
  }
  let mut result = ZERO_WORD;
  result[31] = x[i[31] as usize];
  result
}

/// `SIGNEXTEND(b, x)`: sign-extend `x` from byte `b` (0 = LSB).
/// Returns `x` unchanged when `b >= 31`.
pub fn compute_signextend(b: &[u8; 32], x: &[u8; 32]) -> [u8; 32] {
  if b[..31].iter().any(|&v| v != 0) || b[31] >= 31 {
    return *x;
  }
  let b_val = b[31] as usize;
  let byte_sign = 31 - b_val; // index into x[] from MSB
  let fill = if (x[byte_sign] & 0x80) != 0 {
    0xFF
  } else {
    0x00
  };
  let mut result = *x;
  for byte in result[..byte_sign].iter_mut() {
    *byte = fill;
  }
  result
}

// ============================================================
// Prove / WFF for remaining arithmetic opcodes
//
// All use **output-binding**: prove_xor(out, expected, ZERO_WORD) certifies
// that `out == expected` because it asserts `out XOR expected = 0` byte-by-byte.
// ============================================================

/// Prove `BYTE(i, x) = out`.
pub fn prove_byte(i: &[u8; 32], x: &[u8; 32], out: &[u8; 32]) -> Proof {
  prove_xor(out, &compute_byte(i, x), &ZERO_WORD)
}
/// WFF for `BYTE(i, x) = out`.
pub fn wff_byte(i: &[u8; 32], x: &[u8; 32], out: &[u8; 32]) -> WFF {
  wff_xor(out, &compute_byte(i, x), &ZERO_WORD)
}

/// Prove `SIGNEXTEND(b, x) = out`.
pub fn prove_signextend(b: &[u8; 32], x: &[u8; 32], out: &[u8; 32]) -> Proof {
  prove_xor(out, &compute_signextend(b, x), &ZERO_WORD)
}
/// WFF for `SIGNEXTEND(b, x) = out`.
pub fn wff_signextend(b: &[u8; 32], x: &[u8; 32], out: &[u8; 32]) -> WFF {
  wff_xor(out, &compute_signextend(b, x), &ZERO_WORD)
}

/// Prove `ADDMOD(a, b, n) = out`.
pub fn prove_addmod(a: &[u8; 32], b: &[u8; 32], n: &[u8; 32], out: &[u8; 32]) -> Proof {
  prove_xor(out, &compute_addmod(a, b, n), &ZERO_WORD)
}
/// WFF for `ADDMOD(a, b, n) = out`.
pub fn wff_addmod(a: &[u8; 32], b: &[u8; 32], n: &[u8; 32], out: &[u8; 32]) -> WFF {
  wff_xor(out, &compute_addmod(a, b, n), &ZERO_WORD)
}

/// Prove `MULMOD(a, b, n) = out`.
pub fn prove_mulmod(a: &[u8; 32], b: &[u8; 32], n: &[u8; 32], out: &[u8; 32]) -> Proof {
  prove_xor(out, &compute_mulmod(a, b, n), &ZERO_WORD)
}
/// WFF for `MULMOD(a, b, n) = out`.
pub fn wff_mulmod(a: &[u8; 32], b: &[u8; 32], n: &[u8; 32], out: &[u8; 32]) -> WFF {
  wff_xor(out, &compute_mulmod(a, b, n), &ZERO_WORD)
}

/// Prove `EXP(a, b) = out`.
pub fn prove_exp(a: &[u8; 32], b: &[u8; 32], out: &[u8; 32]) -> Proof {
  prove_xor(out, &compute_exp(a, b), &ZERO_WORD)
}
/// WFF for `EXP(a, b) = out`.
pub fn wff_exp(a: &[u8; 32], b: &[u8; 32], out: &[u8; 32]) -> WFF {
  wff_xor(out, &compute_exp(a, b), &ZERO_WORD)
}

// ============================================================
// Compiling: Tree → Vec<ProofRow>
// ============================================================

fn compile_term_inner(term: &Term, rows: &mut Vec<ProofRow>, memo: &mut Memo) -> u32 {
  match term {
    Term::Bool(v) => {
      let val = *v as u32;
      let key = (OP_BOOL, 0, 0, 0, val, 0, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_BOOL,
        scalar0: val,
        value: val,
        ret_ty: RET_BOOL,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Term::Not(a) => {
      let ai = compile_term_inner(a, rows, memo);
      let key = (OP_NOT, ai, 0, 0, 0, 0, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let val = 1 - rows[ai as usize].value;
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_NOT,
        arg0: ai,
        value: val,
        ret_ty: RET_BOOL,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Term::And(a, b) => {
      let ai = compile_term_inner(a, rows, memo);
      let bi = compile_term_inner(b, rows, memo);
      let key = (OP_AND, ai, bi, 0, 0, 0, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let val = rows[ai as usize].value * rows[bi as usize].value;
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_AND,
        arg0: ai,
        arg1: bi,
        value: val,
        ret_ty: RET_BOOL,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Term::Or(a, b) => {
      let ai = compile_term_inner(a, rows, memo);
      let bi = compile_term_inner(b, rows, memo);
      let key = (OP_OR, ai, bi, 0, 0, 0, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let (av, bv) = (rows[ai as usize].value, rows[bi as usize].value);
      let val = av + bv - av * bv;
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_OR,
        arg0: ai,
        arg1: bi,
        value: val,
        ret_ty: RET_BOOL,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Term::Xor(a, b) => {
      let ai = compile_term_inner(a, rows, memo);
      let bi = compile_term_inner(b, rows, memo);
      let key = (OP_XOR, ai, bi, 0, 0, 0, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let (av, bv) = (rows[ai as usize].value, rows[bi as usize].value);
      let val = av + bv - 2 * av * bv;
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_XOR,
        arg0: ai,
        arg1: bi,
        value: val,
        ret_ty: RET_BOOL,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Term::Ite(c, a, b) => {
      let ci = compile_term_inner(c, rows, memo);
      let ai = compile_term_inner(a, rows, memo);
      let bi = compile_term_inner(b, rows, memo);
      let key = (OP_ITE, ci, ai, bi, 0, 0, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let cv = rows[ci as usize].value;
      let (av, bv) = (rows[ai as usize].value, rows[bi as usize].value);
      let val = cv * av + (1 - cv) * bv;
      let ret = rows[ai as usize].ret_ty;
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_ITE,
        arg0: ci,
        arg1: ai,
        arg2: bi,
        value: val,
        ret_ty: ret,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Term::Byte(v) => {
      let key = (OP_BYTE, 0, 0, 0, *v as u32, 0, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_BYTE,
        scalar0: *v as u32,
        value: *v as u32,
        ret_ty: RET_BYTE,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Term::ByteAdd(a, b, c) => {
      let ai = compile_term_inner(a, rows, memo);
      let bi = compile_term_inner(b, rows, memo);
      let ci = compile_term_inner(c, rows, memo);
      let key = (OP_BYTE_ADD, ai, bi, ci, 0, 0, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let total = rows[ai as usize].value + rows[bi as usize].value + rows[ci as usize].value;
      let val = total & 0xFF;
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_BYTE_ADD,
        arg0: ai,
        arg1: bi,
        arg2: ci,
        value: val,
        ret_ty: RET_BYTE,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Term::ByteAddCarry(a, b, c) => {
      let ai = compile_term_inner(a, rows, memo);
      let bi = compile_term_inner(b, rows, memo);
      let ci = compile_term_inner(c, rows, memo);
      let key = (OP_BYTE_ADD_CARRY, ai, bi, ci, 0, 0, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let total = rows[ai as usize].value + rows[bi as usize].value + rows[ci as usize].value;
      let val = if total >= 256 { 1 } else { 0 };
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_BYTE_ADD_CARRY,
        arg0: ai,
        arg1: bi,
        arg2: ci,
        value: val,
        ret_ty: RET_BOOL,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Term::ByteMulLow(a, b) => {
      let ai = compile_term_inner(a, rows, memo);
      let bi = compile_term_inner(b, rows, memo);
      let key = (OP_BYTE_MUL_LOW, ai, bi, 0, 0, 0, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let val = (rows[ai as usize].value * rows[bi as usize].value) & 0xFF;
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_BYTE_MUL_LOW,
        arg0: ai,
        arg1: bi,
        value: val,
        ret_ty: RET_BYTE,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Term::ByteMulHigh(a, b) => {
      let ai = compile_term_inner(a, rows, memo);
      let bi = compile_term_inner(b, rows, memo);
      let key = (OP_BYTE_MUL_HIGH, ai, bi, 0, 0, 0, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let val = (rows[ai as usize].value * rows[bi as usize].value) >> 8;
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_BYTE_MUL_HIGH,
        arg0: ai,
        arg1: bi,
        value: val,
        ret_ty: RET_BYTE,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Term::ByteAnd(a, b) => {
      let ai = compile_term_inner(a, rows, memo);
      let bi = compile_term_inner(b, rows, memo);
      let key = (OP_BYTE_AND, ai, bi, 0, 0, 0, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let val = rows[ai as usize].value & rows[bi as usize].value;
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_BYTE_AND,
        arg0: ai,
        arg1: bi,
        value: val,
        ret_ty: RET_BYTE,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Term::ByteOr(a, b) => {
      let ai = compile_term_inner(a, rows, memo);
      let bi = compile_term_inner(b, rows, memo);
      let key = (OP_BYTE_OR, ai, bi, 0, 0, 0, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let val = rows[ai as usize].value | rows[bi as usize].value;
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_BYTE_OR,
        arg0: ai,
        arg1: bi,
        value: val,
        ret_ty: RET_BYTE,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Term::ByteXor(a, b) => {
      let ai = compile_term_inner(a, rows, memo);
      let bi = compile_term_inner(b, rows, memo);
      let key = (OP_BYTE_XOR, ai, bi, 0, 0, 0, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let val = rows[ai as usize].value ^ rows[bi as usize].value;
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_BYTE_XOR,
        arg0: ai,
        arg1: bi,
        value: val,
        ret_ty: RET_BYTE,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    // ── Symbolic variable terms (value = 0 placeholder; resolved by consistency AIR) ──
    Term::InputTerm { stack_idx, byte_idx } => {
      let key = (OP_INPUT_TERM, 0, 0, 0, *stack_idx as u32, *byte_idx as u32, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_INPUT_TERM,
        scalar0: *stack_idx as u32,
        scalar1: *byte_idx as u32,
        ret_ty: RET_BYTE,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Term::OutputTerm { stack_idx, byte_idx } => {
      let key = (OP_OUTPUT_TERM, 0, 0, 0, *stack_idx as u32, *byte_idx as u32, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_OUTPUT_TERM,
        scalar0: *stack_idx as u32,
        scalar1: *byte_idx as u32,
        ret_ty: RET_BYTE,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Term::PcBefore { byte_idx } => {
      let key = (OP_PC_BEFORE, 0, 0, 0, *byte_idx as u32, 0, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_PC_BEFORE,
        scalar0: *byte_idx as u32,
        ret_ty: RET_BYTE,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Term::PcAfter { byte_idx } => {
      let key = (OP_PC_AFTER, 0, 0, 0, *byte_idx as u32, 0, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_PC_AFTER,
        scalar0: *byte_idx as u32,
        ret_ty: RET_BYTE,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
  }
}

fn compile_proof_inner(proof: &Proof, rows: &mut Vec<ProofRow>, memo: &mut Memo) -> u32 {
  match proof {
    Proof::AndIntro(p1, p2) => {
      let p1i = compile_proof_inner(p1, rows, memo);
      let p2i = compile_proof_inner(p2, rows, memo);
      let key = (OP_AND_INTRO, p1i, p2i, 0, 0, 0, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_AND_INTRO,
        arg0: p1i,
        arg1: p2i,
        ret_ty: RET_WFF_AND,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Proof::EqRefl(t) => {
      let ti = compile_term_inner(t, rows, memo);
      let key = (OP_EQ_REFL, ti, 0, 0, 0, 0, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_EQ_REFL,
        arg0: ti,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Proof::EqSym(p) => {
      let pi = compile_proof_inner(p, rows, memo);
      let key = (OP_EQ_SYM, pi, 0, 0, 0, 0, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_EQ_SYM,
        arg0: pi,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Proof::EqTrans(p1, p2) => {
      let p1i = compile_proof_inner(p1, rows, memo);
      let p2i = compile_proof_inner(p2, rows, memo);
      let key = (OP_EQ_TRANS, p1i, p2i, 0, 0, 0, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_EQ_TRANS,
        arg0: p1i,
        arg1: p2i,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Proof::ByteAndEq(a, b) => {
      let key = (OP_BYTE_AND_EQ, 0, 0, 0, *a as u32, *b as u32, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_BYTE_AND_EQ,
        scalar0: *a as u32,
        scalar1: *b as u32,
        value: (*a & *b) as u32,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Proof::ByteOrEq(a, b) => {
      let key = (OP_BYTE_OR_EQ, 0, 0, 0, *a as u32, *b as u32, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_BYTE_OR_EQ,
        scalar0: *a as u32,
        scalar1: *b as u32,
        value: (*a | *b) as u32,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Proof::ByteXorEq(a, b) => {
      let key = (OP_BYTE_XOR_EQ, 0, 0, 0, *a as u32, *b as u32, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_BYTE_XOR_EQ,
        scalar0: *a as u32,
        scalar1: *b as u32,
        value: (*a ^ *b) as u32,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Proof::U29AddEq(a, b, cin, c) => {
      let carry_in = *cin as u32;
      let key = (OP_U29_ADD_EQ, 0, 0, 0, *a, *b, carry_in);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let total = *a + *b + carry_in;
      let sum29 = total & ((1u32 << 29) - 1);
      let carry_out = if total >= (1u32 << 29) { 1 } else { 0 };
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_U29_ADD_EQ,
        scalar0: *a,
        scalar1: *b,
        scalar2: carry_in,
        arg0: *c,
        arg1: carry_out,
        value: sum29,
        ret_ty: RET_WFF_AND,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Proof::U24AddEq(a, b, cin, c) => {
      let carry_in = *cin as u32;
      let key = (OP_U24_ADD_EQ, 0, 0, 0, *a, *b, carry_in);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let total = *a + *b + carry_in;
      let sum24 = total & ((1u32 << 24) - 1);
      let carry_out = if total >= (1u32 << 24) { 1 } else { 0 };
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_U24_ADD_EQ,
        scalar0: *a,
        scalar1: *b,
        scalar2: carry_in,
        arg0: *c,
        arg1: carry_out,
        value: sum24,
        ret_ty: RET_WFF_AND,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Proof::U15MulEq(a, b) => {
      let key = (OP_U15_MUL_EQ, 0, 0, 0, *a as u32, *b as u32, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let total = *a as u32 * *b as u32;
      let lo = total & 0x7FFF;
      let hi = total >> 15;
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_U15_MUL_EQ,
        scalar0: *a as u32,
        scalar1: *b as u32,
        scalar2: 0,
        arg0: hi,
        value: lo,
        ret_ty: RET_WFF_AND,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Proof::ByteAddThirdCongruence(p, a, b) => {
      let pi = compile_proof_inner(p, rows, memo);
      let key = (
        OP_BYTE_ADD_THIRD_CONGRUENCE,
        pi,
        0,
        0,
        *a as u32,
        *b as u32,
        0,
      );
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_BYTE_ADD_THIRD_CONGRUENCE,
        scalar0: *a as u32,
        scalar1: *b as u32,
        arg0: pi,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Proof::ByteAddCarryThirdCongruence(p, a, b) => {
      let pi = compile_proof_inner(p, rows, memo);
      let key = (
        OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE,
        pi,
        0,
        0,
        *a as u32,
        *b as u32,
        0,
      );
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE,
        scalar0: *a as u32,
        scalar1: *b as u32,
        arg0: pi,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Proof::IteTrueEq(a, b) => {
      let ai = compile_term_inner(a, rows, memo);
      let bi = compile_term_inner(b, rows, memo);
      let key = (OP_ITE_TRUE_EQ, ai, bi, 0, 0, 0, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_ITE_TRUE_EQ,
        arg0: ai,
        arg1: bi,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Proof::IteFalseEq(a, b) => {
      let ai = compile_term_inner(a, rows, memo);
      let bi = compile_term_inner(b, rows, memo);
      let key = (OP_ITE_FALSE_EQ, ai, bi, 0, 0, 0, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_ITE_FALSE_EQ,
        arg0: ai,
        arg1: bi,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Proof::ByteMulLowEq(a, b) => {
      let key = (OP_BYTE_MUL_LOW_EQ, 0, 0, 0, *a as u32, *b as u32, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_BYTE_MUL_LOW_EQ,
        scalar0: *a as u32,
        scalar1: *b as u32,
        value: (*a as u32 * *b as u32) & 0xFF,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Proof::ByteMulHighEq(a, b) => {
      let key = (OP_BYTE_MUL_HIGH_EQ, 0, 0, 0, *a as u32, *b as u32, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_BYTE_MUL_HIGH_EQ,
        scalar0: *a as u32,
        scalar1: *b as u32,
        value: (*a as u32 * *b as u32) >> 8,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }

    // ── Per-opcode axiom leaf rows ──────────────────────────────────────────
    // Each axiom compiles to a single row with:
    //   op      = OP_*_AXIOM
    //   scalar0 = EVM opcode byte
    //   ret_ty  = RET_WFF_AXIOM
    // The row carries no child indices — it is a leaf.  Its soundness is
    // enforced by the consistency AIR at batch level, not by verify_compiled.
    Proof::PushAxiom { .. } => push_axiom_row(OP_PUSH_AXIOM, 0, rows, memo),
    Proof::DupAxiom { depth, .. } => push_axiom_row(OP_DUP_AXIOM, *depth as u32, rows, memo),
    Proof::SwapAxiom { depth, .. } => push_axiom_row(OP_SWAP_AXIOM, *depth as u32, rows, memo),
    Proof::StructuralAxiom { opcode } => push_axiom_row(OP_STRUCTURAL_AXIOM, *opcode as u32, rows, memo),
    Proof::MloadAxiom { .. } => push_axiom_row(OP_MLOAD_AXIOM, 0, rows, memo),
    Proof::MstoreAxiom { opcode, .. } => push_axiom_row(OP_MSTORE_AXIOM, *opcode as u32, rows, memo),
    Proof::MemCopyAxiom { opcode, .. } => push_axiom_row(OP_MEM_COPY_AXIOM, *opcode as u32, rows, memo),
    Proof::SloadAxiom { .. } => push_axiom_row(OP_SLOAD_AXIOM, 0, rows, memo),
    Proof::SstoreAxiom { .. } => push_axiom_row(OP_SSTORE_AXIOM, 0, rows, memo),
    Proof::TransientAxiom { opcode, .. } => push_axiom_row(OP_TRANSIENT_AXIOM, *opcode as u32, rows, memo),
    Proof::KeccakAxiom { .. } => push_axiom_row(OP_KECCAK_AXIOM, 0, rows, memo),
    Proof::EnvAxiom { opcode, .. } => push_axiom_row(OP_ENV_AXIOM, *opcode as u32, rows, memo),
    Proof::ExternalStateAxiom { opcode, .. } => push_axiom_row(OP_EXTERNAL_STATE_AXIOM, *opcode as u32, rows, memo),
    Proof::TerminateAxiom { opcode, .. } => push_axiom_row(OP_TERMINATE_AXIOM, *opcode as u32, rows, memo),
    Proof::CallAxiom { opcode, .. } => push_axiom_row(OP_CALL_AXIOM, *opcode as u32, rows, memo),
    Proof::CreateAxiom { opcode, .. } => push_axiom_row(OP_CREATE_AXIOM, *opcode as u32, rows, memo),
    Proof::SelfdestructAxiom { .. } => push_axiom_row(OP_SELFDESTRUCT_AXIOM, 0, rows, memo),
    Proof::LogAxiom { opcode, .. } => push_axiom_row(OP_LOG_AXIOM, *opcode as u32, rows, memo),
  }
}

/// Emit a single axiom leaf row, deduplicating by (op, scalar0).
fn push_axiom_row(op: u32, scalar0: u32, rows: &mut Vec<ProofRow>, memo: &mut Memo) -> u32 {
  let key = (op, scalar0, 0, 0, 0, 0, 0);
  if let Some(&cached) = memo.get(&key) {
    return cached;
  }
  let idx = rows.len() as u32;
  rows.push(ProofRow {
    op,
    scalar0,
    ret_ty: RET_WFF_AXIOM,
    ..Default::default()
  });
  memo.insert(key, idx);
  idx
}

/// Flatten a [`Proof`] tree into a `Vec<ProofRow>` (post-order).
pub fn compile_proof(proof: &Proof) -> Vec<ProofRow> {
  let mut rows = Vec::new();
  let mut memo = Memo::new();
  compile_proof_inner(proof, &mut rows, &mut memo);
  rows
}

/// Flatten a [`Term`] tree into a `Vec<ProofRow>` (post-order, CSE-deduplicated).
pub fn compile_term(term: &Term) -> Vec<ProofRow> {
  let mut rows = Vec::new();
  let mut memo = Memo::new();
  compile_term_inner(term, &mut rows, &mut memo);
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
        return Err(VerifyError::UnexpectedTermVariant {
          expected: "valid child index",
        });
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
        let exp = if a.value + b.value + c.value >= 256 {
          1
        } else {
          0
        };
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
      OP_U15_MUL_EQ => {
        if row.scalar0 > 0x7FFF || row.scalar1 > 0x7FFF || row.arg0 > 0x7FFF || row.value > 0x7FFF {
          return Err(VerifyError::ByteDecideFailed);
        }
        let prod = row.scalar0 * row.scalar1;
        if row.value != (prod & 0x7FFF) || row.arg0 != (prod >> 15) {
          return Err(VerifyError::ByteDecideFailed);
        }
        if row.ret_ty != RET_WFF_AND {
          return Err(VerifyError::UnexpectedProofVariant {
            expected: "u15-mul proof row returns conjunction",
          });
        }
      }
      OP_U29_ADD_EQ => {
        if row.scalar0 >= (1u32 << 29)
          || row.scalar1 >= (1u32 << 29)
          || row.scalar2 > 1
          || row.arg0 >= (1u32 << 29)
          || row.arg1 > 1
          || row.value >= (1u32 << 29)
        {
          return Err(VerifyError::ByteDecideFailed);
        }
        let total = row.scalar0 + row.scalar1 + row.scalar2;
        if row.value != (total & ((1u32 << 29) - 1)) || row.arg0 != row.value {
          return Err(VerifyError::ByteDecideFailed);
        }
        if row.arg1 != if total >= (1u32 << 29) { 1 } else { 0 } {
          return Err(VerifyError::ByteDecideFailed);
        }
        if row.ret_ty != RET_WFF_AND {
          return Err(VerifyError::UnexpectedProofVariant {
            expected: "u29-add proof row returns conjunction",
          });
        }
      }
      OP_U24_ADD_EQ => {
        if row.scalar0 >= (1u32 << 24)
          || row.scalar1 >= (1u32 << 24)
          || row.scalar2 > 1
          || row.arg0 >= (1u32 << 24)
          || row.arg1 > 1
          || row.value >= (1u32 << 24)
        {
          return Err(VerifyError::ByteDecideFailed);
        }
        let total = row.scalar0 + row.scalar1 + row.scalar2;
        if row.value != (total & ((1u32 << 24) - 1)) || row.arg0 != row.value {
          return Err(VerifyError::ByteDecideFailed);
        }
        if row.arg1 != if total >= (1u32 << 24) { 1 } else { 0 } {
          return Err(VerifyError::ByteDecideFailed);
        }
        if row.ret_ty != RET_WFF_AND {
          return Err(VerifyError::UnexpectedProofVariant {
            expected: "u24-add proof row returns conjunction",
          });
        }
      }
      // ── Proof: conjunction ──
      OP_AND_INTRO => {
        let (a, b) = (arg(row.arg0)?, arg(row.arg1)?);
        if a.ret_ty < RET_WFF_EQ || b.ret_ty < RET_WFF_EQ {
          return Err(VerifyError::UnexpectedProofVariant {
            expected: "proof arguments",
          });
        }
      }
      // ── Proof: reflexivity ──
      OP_EQ_REFL => {
        let a = arg(row.arg0)?;
        if a.ret_ty > RET_BYTE {
          return Err(VerifyError::UnexpectedProofVariant {
            expected: "term argument",
          });
        }
      }
      // ── Proof: symmetry ──
      OP_EQ_SYM => {
        let a = arg(row.arg0)?;
        if a.ret_ty != RET_WFF_EQ {
          return Err(VerifyError::UnexpectedProofVariant {
            expected: "equality proof",
          });
        }
      }
      // ── Proof: transitivity ──
      OP_EQ_TRANS => {
        let (a, b) = (arg(row.arg0)?, arg(row.arg1)?);
        if a.ret_ty != RET_WFF_EQ || b.ret_ty != RET_WFF_EQ {
          return Err(VerifyError::UnexpectedProofVariant {
            expected: "equality proofs",
          });
        }
      }
      // ── Proof: byte bitwise axioms (lookup-verifiable) ──
      OP_BYTE_AND_EQ => {
        if row.scalar0 > 255 || row.scalar1 > 255 {
          return Err(VerifyError::ByteDecideFailed);
        }
        if row.value != row.scalar0 & row.scalar1 {
          return Err(VerifyError::ByteDecideFailed);
        }
      }
      OP_BYTE_OR_EQ => {
        if row.scalar0 > 255 || row.scalar1 > 255 {
          return Err(VerifyError::ByteDecideFailed);
        }
        if row.value != row.scalar0 | row.scalar1 {
          return Err(VerifyError::ByteDecideFailed);
        }
      }
      OP_BYTE_XOR_EQ => {
        if row.scalar0 > 255 || row.scalar1 > 255 {
          return Err(VerifyError::ByteDecideFailed);
        }
        if row.value != row.scalar0 ^ row.scalar1 {
          return Err(VerifyError::ByteDecideFailed);
        }
      }
      // ── Proof: byte mul axioms (lookup-verifiable) ──
      OP_BYTE_MUL_LOW_EQ => {
        if row.scalar0 > 255 || row.scalar1 > 255 {
          return Err(VerifyError::ByteDecideFailed);
        }
        if row.value != (row.scalar0 * row.scalar1) & 0xFF {
          return Err(VerifyError::ByteDecideFailed);
        }
      }
      OP_BYTE_MUL_HIGH_EQ => {
        if row.scalar0 > 255 || row.scalar1 > 255 {
          return Err(VerifyError::ByteDecideFailed);
        }
        if row.value != (row.scalar0 * row.scalar1) >> 8 {
          return Err(VerifyError::ByteDecideFailed);
        }
      }
      OP_BYTE_ADD_THIRD_CONGRUENCE | OP_BYTE_ADD_CARRY_THIRD_CONGRUENCE => {
        if row.scalar0 > 255 || row.scalar1 > 255 {
          return Err(VerifyError::ByteDecideFailed);
        }
        let p = arg(row.arg0)?;
        if p.ret_ty != RET_WFF_EQ {
          return Err(VerifyError::UnexpectedProofVariant {
            expected: "equality proof",
          });
        }
      }
      OP_ITE_TRUE_EQ | OP_ITE_FALSE_EQ => {
        let (a, b) = (arg(row.arg0)?, arg(row.arg1)?);
        if a.ret_ty > RET_BYTE || b.ret_ty > RET_BYTE || a.ret_ty != b.ret_ty {
          return Err(VerifyError::UnexpectedProofVariant {
            expected: "matching term arguments",
          });
        }
      }
      // ── Per-opcode axiom rows: leaf nodes, no children to validate ──────
      // Soundness is guaranteed by the consistency AIR at batch level.
      OP_PUSH_AXIOM
      | OP_DUP_AXIOM
      | OP_SWAP_AXIOM
      | OP_STRUCTURAL_AXIOM
      | OP_MLOAD_AXIOM
      | OP_MSTORE_AXIOM
      | OP_MEM_COPY_AXIOM
      | OP_SLOAD_AXIOM
      | OP_SSTORE_AXIOM
      | OP_TRANSIENT_AXIOM
      | OP_KECCAK_AXIOM
      | OP_ENV_AXIOM
      | OP_EXTERNAL_STATE_AXIOM
      | OP_TERMINATE_AXIOM
      | OP_CALL_AXIOM
      | OP_CREATE_AXIOM
      | OP_SELFDESTRUCT_AXIOM
      | OP_LOG_AXIOM => {
        if row.ret_ty != RET_WFF_AXIOM {
          return Err(VerifyError::UnexpectedProofVariant {
            expected: "axiom row must have RET_WFF_AXIOM",
          });
        }
        // No child rows to check; accepted unconditionally here.
      }
      _ => {
        return Err(VerifyError::UnexpectedTermVariant {
          expected: "valid opcode",
        });
      }
    }
  }
  Ok(())
}

// ============================================================
// Oracle I/O axiom helpers
// ============================================================

/// Build the WFF for an oracle axiom that commits concrete `inputs` and
/// `outputs` values to the batch manifest.
///
/// Produces a right-fold conjunction of `Equal(Byte(b), Byte(b))` over every
/// byte in `inputs ++ outputs` (in word-order: word[0..32] then word[1..32]…).
/// Each equality is trivially true, so `infer_proof(prove_oracle_io(…))` always
/// succeeds — soundness comes from the cross-validation in
/// `verify_batch_transaction_zk_receipt`, which checks the committed values
/// against the appropriate consistency AIR (memory, storage, keccak).
///
/// For no-input-no-output calls (future extension), returns a trivial
/// `Equal(Bool(true), Bool(true))`.
pub fn wff_oracle_io(inputs: &[[u8; 32]], outputs: &[[u8; 32]]) -> WFF {
  let all_bytes: Vec<u8> = inputs
    .iter()
    .chain(outputs.iter())
    .flat_map(|w| w.iter().copied())
    .collect();
  if all_bytes.is_empty() {
    return WFF::Equal(Box::new(Term::Bool(true)), Box::new(Term::Bool(true)));
  }
  let n = all_bytes.len();
  let byte_refl = |b: u8| WFF::Equal(Box::new(Term::Byte(b)), Box::new(Term::Byte(b)));
  let mut result = byte_refl(all_bytes[n - 1]);
  for &b in all_bytes[..n - 1].iter().rev() {
    result = WFF::And(Box::new(byte_refl(b)), Box::new(result));
  }
  result
}

/// Build a trivially-satisfied oracle proof that encodes concrete `inputs` and
/// `outputs` values in the proof tree.
///
/// Compiles to a right-fold `AndIntro` chain of `EqRefl(Byte(b))` nodes —
/// one per byte across all input and output words.  These become non-arithmetic
/// padding rows in the LUT STARK (they satisfy the U29AddEq constraint with all
/// zeros), while the byte values appear in the preprocessed matrix and are
/// covered by the batch manifest digest.
///
/// `infer_proof(prove_oracle_io(inputs, outputs))` returns exactly
/// `wff_oracle_io(inputs, outputs)`.
pub fn prove_oracle_io(inputs: &[[u8; 32]], outputs: &[[u8; 32]]) -> Proof {
  let all_bytes: Vec<u8> = inputs
    .iter()
    .chain(outputs.iter())
    .flat_map(|w| w.iter().copied())
    .collect();
  if all_bytes.is_empty() {
    return Proof::EqRefl(Term::Bool(true));
  }
  let n = all_bytes.len();
  let mut result = Proof::EqRefl(Term::Byte(all_bytes[n - 1]));
  for &b in all_bytes[..n - 1].iter().rev() {
    result = Proof::AndIntro(Box::new(Proof::EqRefl(Term::Byte(b))), Box::new(result));
  }
  result
}

// ============================================================
// Tests
// ============================================================
