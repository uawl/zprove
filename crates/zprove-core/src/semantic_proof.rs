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
  /// 15-bit unsigned integer (limb of a 256-bit word in 15-bit Comba MUL).
  U15,
  /// 29-bit unsigned integer (limb of a 256-bit word in the 29+24-radix decomposition).
  U29,
  /// 24-bit unsigned integer (top limb of a 256-bit word).
  U24,
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
  /// `value` is the **claimed** concrete byte — accepted optimistically here and
  /// verified later via a separate stack-log AIR + LogUp multiset argument.
  InputTerm { stack_idx: u8, byte_idx: u8, value: u8 },
  /// The `byte_idx`-th byte of the `stack_idx`-th stack output word.
  /// `value` carries the same optimistic claim as `InputTerm`.
  OutputTerm { stack_idx: u8, byte_idx: u8, value: u8 },
  /// The `byte_idx`-th byte of the program counter before execution (4-byte big-endian u32).
  PcBefore { byte_idx: u8 },
  /// The `byte_idx`-th byte of the program counter after execution.
  PcAfter { byte_idx: u8 },

  // ---- Intermediate witness terms (carry / borrow) ----
  /// The carry bit OUT of byte `byte_idx` during an ADD or SUB (0 or 1).
  /// For ADD: carry_out[j] = (inputs[0][j] + inputs[1][j] + carry_out[j+1]) >> 8.
  /// For SUB (`a = b + c`): same carry from adding `b` and `c`.
  CarryTerm { byte_idx: u8 },

  // ---- 29/24-bit limb symbolic terms ----
  // A 256-bit EVM word is decomposed as 8 limbs of 29 bits (little-endian)
  // plus one 24-bit top limb: word = Σ_{j=0}^{7} limb29[j] * 2^(29*j) + limb24 * 2^232.
  /// The `limb_idx`-th 29-bit LE limb of the `stack_idx`-th stack input word.
  /// `limb_idx` ∈ 0..=7: limb 0 covers bits 0..29, limb 7 covers bits 203..232.
  InputLimb29 { stack_idx: u8, limb_idx: u8 },
  /// The `limb_idx`-th 29-bit LE limb of the `stack_idx`-th stack output word.
  OutputLimb29 { stack_idx: u8, limb_idx: u8 },
  /// The 24-bit top limb of the `stack_idx`-th stack input (bits 232..256).
  InputLimb24 { stack_idx: u8 },
  /// The 24-bit top limb of the `stack_idx`-th stack output (bits 232..256).
  OutputLimb24 { stack_idx: u8 },

  // ---- Limb-level arithmetic terms (syntactic; values from LogUp) ----
  /// The carry bit OUT of the `limb_idx`-th 29-bit limb addition (Bool: 0 or 1).
  /// `CarryLimb{0} = false` (no incoming carry for the LSB limb of ADD/SUB).
  CarryLimb { limb_idx: u8 },
  /// `(a + b + cin) mod 2^29` — 29-bit modular sum (U29).
  Add29(Box<Term>, Box<Term>, Box<Term>),
  /// `(a + b + cin) >> 29` — carry out of a 29-bit addition (Bool: 0 or 1).
  Add29Carry(Box<Term>, Box<Term>, Box<Term>),
  /// `(a + b + cin) mod 2^24` — 24-bit modular sum (U24, top-limb).
  Add24(Box<Term>, Box<Term>, Box<Term>),

  // ---- 15-bit MUL intermediate terms ----
  /// Concrete 15-bit literal value (analogous to `Byte(u8)` at U15 granularity).
  U15(u16),
  /// Lower 15 bits of a 15-bit × 15-bit product: `(a * b) & 0x7FFF`.
  /// Both operands must be U15 terms.
  U15MulLow(Box<Term>, Box<Term>),
  /// Upper 15 bits of a 15-bit × 15-bit product: `(a * b) >> 15`.
  /// Both operands must be U15 terms.
  U15MulHigh(Box<Term>, Box<Term>),
  /// Prover-supplied carry witness INTO 15-bit Comba column `col_idx`.
  ///
  /// Represents the overflow accumulated from all columns with index < `col_idx`.
  /// Typed U15 at any given proof step; the STARK verifier checks the committed
  /// value is consistent with the LUT column-sum constraint.
  PartialMul15 { col_idx: u8 },
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
  /// `U15MulLow(U15(a), U15(b)) = U15((a * b) & 0x7FFF)`.
  U15MulLowEq(u16, u16),
  /// `U15MulHigh(U15(a), U15(b)) = U15((a * b) >> 15)`.
  U15MulHighEq(u16, u16),
  /// `PartialMul15{col_idx} = U15(value mod 2^15) + 2^15 * U15(value >> 15)`.
  ///
  /// Binds the prover-supplied carry witness `PartialMul15{col_idx}` to its
  /// concrete lo/hi 15-bit decomposition so the carry chain can be verified.
  PartialMul15Bind { col_idx: u8, lo: u16, hi: u16 },

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

  // ── Stack I/O and PC binding axioms ──────────────────────────────────
  /// `InputTerm{stack_idx, byte_idx} = Byte(value)` — stack input binding.
  InputEq { stack_idx: u8, byte_idx: u8, value: u8 },
  /// `OutputTerm{stack_idx, byte_idx} = Byte(value)` — stack output binding.
  OutputEq { stack_idx: u8, byte_idx: u8, value: u8 },
  /// `PcBefore{byte_idx} = Byte(value)` — PC-before binding.
  PcBeforeEq { byte_idx: u8, value: u8 },
  /// `PcAfter{j} = ByteAdd(PcBefore{j}, Byte(instr_size_bytes[j]), carry_j)` for all `j`.
  ///
  /// Proves the 4-byte big-endian relationship `PcAfter = PcBefore + instr_size`
  /// using a carry chain over bytes 3..=0.
  PcStep { instr_size: u32 },

  // ── Symbolic byte-level operation axioms ──────────────────────────────
  //
  // These produce WFFs that reference `InputTerm` / `OutputTerm` symbolically,
  // so the WFF structure is *identical for all executions of the same opcode*.
  // The concrete byte values `a`, `b` are carried only as witnesses for the
  // lookup-table check in `verify_compiled`; they do NOT appear in the WFF.
  //
  // Soundness: these axioms are used only inside `prove_instruction`, which
  // wraps them in an outer `AndIntro` with `prove_stack_inputs` /
  // `prove_stack_outputs` bindings.  The STARK enforces that the witness
  // values match the binding values column-by-column.

  /// `ByteAnd(InputTerm{0,j}, InputTerm{1,j}) = OutputTerm{0,j}`
  ByteAndSym { byte_idx: u8, a: u8, b: u8 },
  /// `ByteOr(InputTerm{0,j}, InputTerm{1,j}) = OutputTerm{0,j}`
  ByteOrSym { byte_idx: u8, a: u8, b: u8 },
  /// `ByteXor(InputTerm{0,j}, InputTerm{1,j}) = OutputTerm{0,j}`
  ByteXorSym { byte_idx: u8, a: u8, b: u8 },
  /// `ByteXor(InputTerm{0,j}, Byte(0xFF)) = OutputTerm{0,j}`  (NOT is XOR with constant)
  ByteNotSym { byte_idx: u8, a: u8 },
  /// `ByteAdd(InputTerm{0,j}, InputTerm{1,j}, CarryTerm{j}) = OutputTerm{0,j}`
  ///
  /// Carry is now a symbolic `CarryTerm{j}`, not a concrete `Byte(c)`.  The
  /// carry value is provided separately via `CarryEq{j, value}`.  This makes
  /// the WFF for every byte leaf identical for all ADD executions.
  AddByteSym { byte_idx: u8, a: u8, b: u8 },
  /// `ByteAdd(InputTerm{1,j}, OutputTerm{0,j}, CarryTerm{j}) = InputTerm{0,j}`
  ///
  /// Verifies `a - b = c` as `b + c = a` byte-wise, reusing `CarryTerm` for
  /// the carry of the addition `b + c`.  No separate borrow machinery needed.
  SubByteSym { byte_idx: u8, b: u8, c: u8 },

  /// `CarryTerm{byte_idx} = Byte(value)` — binds the carry witness to a concrete byte.
  CarryEq { byte_idx: u8, value: u8 },

  // ---- Syntactic (value-free) limb-level proof variants -----------------
  // These reference symbolic limb terms (InputLimb29, OutputLimb29, etc.).
  // Concrete correctness is deferred to the LogUp binding layer; these proofs
  // are purely structural and encode a FIXED WFF per opcode.

  /// `CarryLimb{0} = Bool(false)` — the ADD/SUB carry chain starts with carry = 0.
  CarryLimbZero,
  /// `Add29(InputLimb29{0,j}, InputLimb29{1,j}, CarryLimb{j}) = OutputLimb29{0,j}`
  /// AND `Add29Carry(InputLimb29{0,j}, InputLimb29{1,j}, CarryLimb{j}) = CarryLimb{j+1}`.
  U29AddSym { limb_idx: u8 },
  /// `Add24(InputLimb24{0}, InputLimb24{1}, CarryLimb{8}) = OutputLimb24{0}`.
  U24AddSym,
  /// `Add29(InputLimb29{1,j}, OutputLimb29{0,j}, CarryLimb{j}) = InputLimb29{0,j}`
  /// AND `Add29Carry(InputLimb29{1,j}, OutputLimb29{0,j}, CarryLimb{j}) = CarryLimb{j+1}`.
  /// Verifies SUB (`a - b = c`) as `b + c = a` at the limb level.
  U29SubSym { limb_idx: u8 },
  /// `Add24(InputLimb24{1}, OutputLimb24{0}, CarryLimb{8}) = InputLimb24{0}`.
  U24SubSym,
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

// ---- Stack I/O and PC binding proof opcodes ----
pub const OP_INPUT_EQ: u32 = 52;
pub const OP_OUTPUT_EQ: u32 = 53;
pub const OP_PC_BEFORE_EQ: u32 = 54;
pub const OP_PC_STEP: u32 = 55;

// ---- Symbolic byte-level operation proof opcodes ----
pub const OP_BYTE_AND_SYM: u32 = 56;
pub const OP_BYTE_OR_SYM: u32 = 57;
pub const OP_BYTE_XOR_SYM: u32 = 58;
pub const OP_BYTE_NOT_SYM: u32 = 59;
pub const OP_ADD_BYTE_SYM: u32 = 60;
pub const OP_SUB_BYTE_SYM: u32 = 61;
// ---- Intermediate witness term opcodes ----
pub const OP_CARRY_TERM: u32 = 62;
// ---- Intermediate witness binding proof opcodes ----
pub const OP_CARRY_EQ: u32 = 64;
// ---- 29/24-bit limb symbolic term opcodes ----
pub const OP_INPUT_LIMB29: u32 = 65;
pub const OP_OUTPUT_LIMB29: u32 = 66;
pub const OP_INPUT_LIMB24: u32 = 67;
pub const OP_OUTPUT_LIMB24: u32 = 68;
// ---- Limb-level arithmetic term opcodes ----
pub const OP_CARRY_LIMB: u32 = 69;
pub const OP_ADD29: u32 = 70;
pub const OP_ADD29_CARRY: u32 = 71;
pub const OP_ADD24: u32 = 72;
// ---- Limb-level symbolic proof opcodes ----
pub const OP_U29_ADD_SYM: u32 = 73;
pub const OP_U24_ADD_SYM: u32 = 74;
pub const OP_CARRY_LIMB_ZERO: u32 = 75;
pub const OP_U29_SUB_SYM: u32 = 76;
pub const OP_U24_SUB_SYM: u32 = 77;
// ---- 15-bit MUL intermediate term opcodes ----
pub const OP_U15_LIT: u32 = 78;
pub const OP_U15_MUL_LOW: u32 = 79;
pub const OP_U15_MUL_HIGH: u32 = 80;
pub const OP_PARTIAL_MUL15: u32 = 81;
// ---- 15-bit MUL axiom proof opcodes ----
pub const OP_U15_MUL_LOW_EQ: u32 = 82;
pub const OP_U15_MUL_HIGH_EQ: u32 = 83;
pub const OP_PARTIAL_MUL15_BIND: u32 = 84;
pub const RET_BOOL: u32 = 0;
pub const RET_BYTE: u32 = 1;
pub const RET_WFF_EQ: u32 = 2;
pub const RET_WFF_AND: u32 = 3;
/// 29-bit limb value return type.
pub const RET_U29: u32 = 4;
/// 24-bit limb value return type.
pub const RET_U24: u32 = 5;
/// 15-bit limb value return type.
pub const RET_U15: u32 = 6;

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

/// Walk a WFF tree and verify every `InputTerm` / `OutputTerm` claim against
/// the supplied concrete stack arrays.
///
/// Returns `true` iff every claim is consistent.
/// Tautological self-equalities `Equal(T, T)` are skipped — they are used as
/// placeholder WFFs for axiom opcodes and carry no meaningful constraints.
pub fn check_wff_io_values(
  wff: &WFF,
  stack_inputs: &[[u8; 32]],
  stack_outputs: &[[u8; 32]],
) -> bool {
  match wff {
    WFF::And(a, b) => {
      check_wff_io_values(a, stack_inputs, stack_outputs)
        && check_wff_io_values(b, stack_inputs, stack_outputs)
    }
    WFF::Equal(a, b) => {
      // Skip tautological self-equality (axiom placeholder, e.g. Equal(OutputTerm{0,0,0}, OutputTerm{0,0,0})).
      if a == b {
        return true;
      }
      check_term_io_values(a, stack_inputs, stack_outputs)
        && check_term_io_values(b, stack_inputs, stack_outputs)
    }
  }
}

/// Walk a Term tree and verify every `InputTerm` / `OutputTerm` claim.
fn check_term_io_values(
  term: &Term,
  stack_inputs: &[[u8; 32]],
  stack_outputs: &[[u8; 32]],
) -> bool {
  match term {
    Term::InputTerm { stack_idx, byte_idx, value } => {
      let s = *stack_idx as usize;
      let j = *byte_idx as usize;
      s < stack_inputs.len() && j < 32 && stack_inputs[s][j] == *value
    }
    Term::OutputTerm { stack_idx, byte_idx, value } => {
      let s = *stack_idx as usize;
      let j = *byte_idx as usize;
      s < stack_outputs.len() && j < 32 && stack_outputs[s][j] == *value
    }
    // Recurse into sub-terms.
    Term::Not(a) => check_term_io_values(a, stack_inputs, stack_outputs),
    Term::And(a, b)
    | Term::Or(a, b)
    | Term::Xor(a, b)
    | Term::ByteMulLow(a, b)
    | Term::ByteMulHigh(a, b)
    | Term::U15MulLow(a, b)
    | Term::U15MulHigh(a, b)
    | Term::ByteAnd(a, b)
    | Term::ByteOr(a, b)
    | Term::ByteXor(a, b) => {
      check_term_io_values(a, stack_inputs, stack_outputs)
        && check_term_io_values(b, stack_inputs, stack_outputs)
    }
    Term::Ite(c, a, b) => {
      check_term_io_values(c, stack_inputs, stack_outputs)
        && check_term_io_values(a, stack_inputs, stack_outputs)
        && check_term_io_values(b, stack_inputs, stack_outputs)
    }
    Term::ByteAdd(a, b, c)
    | Term::ByteAddCarry(a, b, c)
    | Term::Add29(a, b, c)
    | Term::Add29Carry(a, b, c)
    | Term::Add24(a, b, c) => {
      check_term_io_values(a, stack_inputs, stack_outputs)
        && check_term_io_values(b, stack_inputs, stack_outputs)
        && check_term_io_values(c, stack_inputs, stack_outputs)
    }
    // Pure leaves — no IO term claims.
    Term::Byte(_)
    | Term::Bool(_)
    | Term::PcBefore { .. }
    | Term::PcAfter { .. }
    | Term::CarryTerm { .. }
    | Term::CarryLimb { .. }
    | Term::InputLimb29 { .. }
    | Term::OutputLimb29 { .. }
    | Term::InputLimb24 { .. }
    | Term::OutputLimb24 { .. }
    | Term::PartialMul15 { .. }
    | Term::U15(_) => true,
  }
}

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
    | Term::PcAfter { .. }
    | Term::CarryTerm { .. } => Ok(Ty::Byte),
    // Limb-level symbolic terms.
    Term::InputLimb29 { .. } | Term::OutputLimb29 { .. } => Ok(Ty::U29),
    Term::InputLimb24 { .. } | Term::OutputLimb24 { .. } => Ok(Ty::U24),
    Term::CarryLimb { .. } => Ok(Ty::Bool),
    Term::Add29(a, b, cin) => {
      if infer_ty(a)? == Ty::U29 && infer_ty(b)? == Ty::U29 && infer_ty(cin)? == Ty::Bool {
        Ok(Ty::U29)
      } else {
        Err(VerifyError::UnexpectedTermVariant { expected: "u29 subterm" })
      }
    }
    Term::Add29Carry(a, b, cin) => {
      if infer_ty(a)? == Ty::U29 && infer_ty(b)? == Ty::U29 && infer_ty(cin)? == Ty::Bool {
        Ok(Ty::Bool)
      } else {
        Err(VerifyError::UnexpectedTermVariant { expected: "u29 subterm" })
      }
    }
    Term::Add24(a, b, cin) => {
      if infer_ty(a)? == Ty::U24 && infer_ty(b)? == Ty::U24 && infer_ty(cin)? == Ty::Bool {
        Ok(Ty::U24)
      } else {
        Err(VerifyError::UnexpectedTermVariant { expected: "u24 subterm" })
      }
    }
    Term::U15MulLow(a, b) | Term::U15MulHigh(a, b) => {
      if infer_ty(a)? == Ty::U15 && infer_ty(b)? == Ty::U15 {
        Ok(Ty::U15)
      } else {
        Err(VerifyError::UnexpectedTermVariant { expected: "u15 subterm" })
      }
    }
    Term::U15(_) | Term::PartialMul15 { .. } => Ok(Ty::U15),
  }
}

pub fn infer_proof(proof: &Proof) -> Result<WFF, VerifyError> {
  // Iterative (stack-based) implementation avoids call-stack overflow on deeply
  // nested proofs (e.g. large DIV/MUL witnesses in debug mode).
  enum Task<'a> {
    /// Evaluate a proof node and push its WFF result onto `wffs`.
    Eval(&'a Proof),
    /// Pop two WFFs and push `And(wff1, wff2)`.
    AndIntro,
    /// Pop one WFF (must be `Equal(a,b)`) and push `Equal(b,a)`.
    EqSym,
    /// Pop two WFFs (`Equal(a,b)` then `Equal(b2,c)`) and push `Equal(a,c)`.
    EqTrans,
    /// Pop one WFF (must be `Equal(c1,c2)`) and push the ByteAdd-congruence.
    ByteAddThird { a: u8, b: u8 },
    /// Same shape but for `ByteAddCarry`.
    ByteAddCarryThird { a: u8, b: u8 },
  }

  let mut tasks: Vec<Task> = vec![Task::Eval(proof)];
  let mut wffs: Vec<WFF> = Vec::new();

  while let Some(task) = tasks.pop() {
    match task {
      // ── Evaluation ────────────────────────────────────────────────────
      Task::Eval(p) => match p {
        // Recursive cases: schedule continuation + children.
        Proof::AndIntro(p1, p2) => {
          tasks.push(Task::AndIntro);
          tasks.push(Task::Eval(p2));
          tasks.push(Task::Eval(p1));
        }
        Proof::EqSym(p) => {
          tasks.push(Task::EqSym);
          tasks.push(Task::Eval(p));
        }
        Proof::EqTrans(p1, p2) => {
          tasks.push(Task::EqTrans);
          tasks.push(Task::Eval(p2));
          tasks.push(Task::Eval(p1));
        }
        Proof::ByteAddThirdCongruence(p, a, b) => {
          tasks.push(Task::ByteAddThird { a: *a, b: *b });
          tasks.push(Task::Eval(p));
        }
        Proof::ByteAddCarryThirdCongruence(p, a, b) => {
          tasks.push(Task::ByteAddCarryThird { a: *a, b: *b });
          tasks.push(Task::Eval(p));
        }

        // Leaf cases: compute WFF and push directly.
        Proof::EqRefl(t) => {
          wffs.push(WFF::Equal(Box::new(t.clone()), Box::new(t.clone())));
        }
Proof::ByteAndEq(a, b) => {
          wffs.push(WFF::Equal(
            Box::new(Term::ByteAnd(Box::new(Term::Byte(*a)), Box::new(Term::Byte(*b)))),
            Box::new(Term::Byte(*a & *b)),
          ));
        }
        Proof::ByteOrEq(a, b) => {
          wffs.push(WFF::Equal(
            Box::new(Term::ByteOr(Box::new(Term::Byte(*a)), Box::new(Term::Byte(*b)))),
            Box::new(Term::Byte(*a | *b)),
          ));
        }
        Proof::ByteXorEq(a, b) => {
          wffs.push(WFF::Equal(
            Box::new(Term::ByteXor(Box::new(Term::Byte(*a)), Box::new(Term::Byte(*b)))),
            Box::new(Term::Byte(*a ^ *b)),
          ));
        }
        Proof::U29AddEq(a, b, cin, c) => {
          if *a >= (1u32 << 29) || *b >= (1u32 << 29) || *c >= (1u32 << 29) {
            return Err(VerifyError::UnexpectedProofVariant { expected: "u29 inputs" });
          }
          let total = *a + *b + (*cin as u32);
          let sum = total & ((1u32 << 29) - 1);
          let pairs = [
            ((sum & 0xFF) as u8,          (*c & 0xFF) as u8),
            (((sum >> 8) & 0xFF) as u8,   ((*c >> 8) & 0xFF) as u8),
            (((sum >> 16) & 0xFF) as u8,  ((*c >> 16) & 0xFF) as u8),
            (((sum >> 24) & 0x1F) as u8,  ((*c >> 24) & 0x1F) as u8),
          ];
          wffs.push(and_wffs(
            pairs
              .into_iter()
              .map(|(lhs, rhs)| WFF::Equal(Box::new(Term::Byte(lhs)), Box::new(Term::Byte(rhs))))
              .collect(),
          ));
        }
        Proof::U24AddEq(a, b, cin, c) => {
          if *a >= (1u32 << 24) || *b >= (1u32 << 24) || *c >= (1u32 << 24) {
            return Err(VerifyError::UnexpectedProofVariant { expected: "u24 inputs" });
          }
          let total = *a + *b + (*cin as u32);
          let sum = total & ((1u32 << 24) - 1);
          let pairs = [
            ((sum & 0xFF) as u8,         (*c & 0xFF) as u8),
            (((sum >> 8) & 0xFF) as u8,  ((*c >> 8) & 0xFF) as u8),
            (((sum >> 16) & 0xFF) as u8, ((*c >> 16) & 0xFF) as u8),
          ];
          wffs.push(and_wffs(
            pairs
              .into_iter()
              .map(|(lhs, rhs)| WFF::Equal(Box::new(Term::Byte(lhs)), Box::new(Term::Byte(rhs))))
              .collect(),
          ));
        }
        Proof::U15MulEq(a, b) => {
          if *a > 0x7FFF || *b > 0x7FFF {
            return Err(VerifyError::UnexpectedProofVariant { expected: "u15 inputs" });
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
              Box::new(Term::ByteMulLow(Box::new(Term::Byte(a_lo)), Box::new(Term::Byte(b_lo)))),
              Box::new(Term::Byte((p00 & 0xFF) as u8)),
            ),
            WFF::Equal(
              Box::new(Term::ByteMulHigh(Box::new(Term::Byte(a_lo)), Box::new(Term::Byte(b_lo)))),
              Box::new(Term::Byte((p00 >> 8) as u8)),
            ),
            WFF::Equal(
              Box::new(Term::ByteMulLow(Box::new(Term::Byte(a_lo)), Box::new(Term::Byte(b_hi)))),
              Box::new(Term::Byte((p01 & 0xFF) as u8)),
            ),
            WFF::Equal(
              Box::new(Term::ByteMulHigh(Box::new(Term::Byte(a_lo)), Box::new(Term::Byte(b_hi)))),
              Box::new(Term::Byte((p01 >> 8) as u8)),
            ),
            WFF::Equal(
              Box::new(Term::ByteMulLow(Box::new(Term::Byte(a_hi)), Box::new(Term::Byte(b_lo)))),
              Box::new(Term::Byte((p10 & 0xFF) as u8)),
            ),
            WFF::Equal(
              Box::new(Term::ByteMulHigh(Box::new(Term::Byte(a_hi)), Box::new(Term::Byte(b_lo)))),
              Box::new(Term::Byte((p10 >> 8) as u8)),
            ),
            WFF::Equal(
              Box::new(Term::ByteMulLow(Box::new(Term::Byte(a_hi)), Box::new(Term::Byte(b_hi)))),
              Box::new(Term::Byte((p11 & 0xFF) as u8)),
            ),
            WFF::Equal(
              Box::new(Term::ByteMulHigh(Box::new(Term::Byte(a_hi)), Box::new(Term::Byte(b_hi)))),
              Box::new(Term::Byte((p11 >> 8) as u8)),
            ),
          ];
          wffs.push(and_wffs(leaves));
        }
        Proof::IteTrueEq(a, b) => {
          let ty_a = infer_ty(a)?;
          let ty_b = infer_ty(b)?;
          if ty_a != ty_b {
            return Err(VerifyError::UnexpectedTermVariant { expected: "matching type subterms" });
          }
          wffs.push(WFF::Equal(
            Box::new(Term::Ite(Box::new(Term::Bool(true)), Box::new(a.clone()), Box::new(b.clone()))),
            Box::new(a.clone()),
          ));
        }
        Proof::IteFalseEq(a, b) => {
          let ty_a = infer_ty(a)?;
          let ty_b = infer_ty(b)?;
          if ty_a != ty_b {
            return Err(VerifyError::UnexpectedTermVariant { expected: "matching type subterms" });
          }
          wffs.push(WFF::Equal(
            Box::new(Term::Ite(Box::new(Term::Bool(false)), Box::new(a.clone()), Box::new(b.clone()))),
            Box::new(b.clone()),
          ));
        }
        Proof::ByteMulLowEq(a, b) => {
          wffs.push(WFF::Equal(
            Box::new(Term::ByteMulLow(Box::new(Term::Byte(*a)), Box::new(Term::Byte(*b)))),
            Box::new(Term::Byte(((*a as u16 * *b as u16) & 0xFF) as u8)),
          ));
        }
        Proof::ByteMulHighEq(a, b) => {
          wffs.push(WFF::Equal(
            Box::new(Term::ByteMulHigh(Box::new(Term::Byte(*a)), Box::new(Term::Byte(*b)))),
            Box::new(Term::Byte(((*a as u16 * *b as u16) >> 8) as u8)),
          ));
        }
        Proof::U15MulLowEq(a, b) => {
          if *a > 0x7FFF || *b > 0x7FFF {
            return Err(VerifyError::UnexpectedProofVariant { expected: "u15 inputs" });
          }
          let lo = ((*a as u32 * *b as u32) & 0x7FFF) as u16;
          wffs.push(WFF::Equal(
            Box::new(Term::U15MulLow(Box::new(Term::U15(*a)), Box::new(Term::U15(*b)))),
            Box::new(Term::U15(lo)),
          ));
        }
        Proof::U15MulHighEq(a, b) => {
          if *a > 0x7FFF || *b > 0x7FFF {
            return Err(VerifyError::UnexpectedProofVariant { expected: "u15 inputs" });
          }
          let hi = ((*a as u32 * *b as u32) >> 15) as u16;
          wffs.push(WFF::Equal(
            Box::new(Term::U15MulHigh(Box::new(Term::U15(*a)), Box::new(Term::U15(*b)))),
            Box::new(Term::U15(hi)),
          ));
        }
        Proof::PartialMul15Bind { col_idx, lo, hi } => {
          wffs.push(WFF::And(
            Box::new(WFF::Equal(
              Box::new(Term::PartialMul15 { col_idx: *col_idx }),
              Box::new(Term::U15(*lo)),
            )),
            Box::new(WFF::Equal(
              Box::new(Term::PartialMul15 { col_idx: col_idx.wrapping_add(128) }),
              Box::new(Term::U15(*hi)),
            )),
          ));
        }
        // ── Per-opcode axiom proofs: each infers Equal(OutputTerm{0,0}, OutputTerm{0,0}). ──
        Proof::PushAxiom
        | Proof::DupAxiom { .. }
        | Proof::SwapAxiom { .. }
        | Proof::StructuralAxiom { .. }
        | Proof::MloadAxiom
        | Proof::MstoreAxiom { .. }
        | Proof::MemCopyAxiom { .. }
        | Proof::SloadAxiom
        | Proof::SstoreAxiom
        | Proof::TransientAxiom { .. }
        | Proof::KeccakAxiom
        | Proof::EnvAxiom { .. }
        | Proof::ExternalStateAxiom { .. }
        | Proof::TerminateAxiom { .. }
        | Proof::CallAxiom { .. }
        | Proof::CreateAxiom { .. }
        | Proof::SelfdestructAxiom
        | Proof::LogAxiom { .. } => {
          wffs.push(WFF::Equal(
            Box::new(Term::OutputTerm { stack_idx: 0, byte_idx: 0, value: 0 }),
            Box::new(Term::OutputTerm { stack_idx: 0, byte_idx: 0, value: 0 }),
          ));
        }
        Proof::InputEq { stack_idx, byte_idx, value } => {
          wffs.push(WFF::Equal(
            Box::new(Term::InputTerm { stack_idx: *stack_idx, byte_idx: *byte_idx, value: *value }),
            Box::new(Term::Byte(*value)),
          ));
        }
        Proof::OutputEq { stack_idx, byte_idx, value } => {
          wffs.push(WFF::Equal(
            Box::new(Term::OutputTerm { stack_idx: *stack_idx, byte_idx: *byte_idx, value: *value }),
            Box::new(Term::Byte(*value)),
          ));
        }
        Proof::PcBeforeEq { byte_idx, value } => {
          wffs.push(WFF::Equal(
            Box::new(Term::PcBefore { byte_idx: *byte_idx }),
            Box::new(Term::Byte(*value)),
          ));
        }
        Proof::PcStep { instr_size } => {
          wffs.push(wff_pc_step(*instr_size));
        }
        // ── Symbolic byte-level ops ─────────────────────────────────────
        Proof::ByteAndSym { byte_idx, a, b } => {
          wffs.push(WFF::Equal(
            Box::new(Term::ByteAnd(
              Box::new(Term::InputTerm { stack_idx: 0, byte_idx: *byte_idx, value: *a }),
              Box::new(Term::InputTerm { stack_idx: 1, byte_idx: *byte_idx, value: *b }),
            )),
            Box::new(Term::OutputTerm { stack_idx: 0, byte_idx: *byte_idx, value: *a & *b }),
          ));
        }
        Proof::ByteOrSym { byte_idx, a, b } => {
          wffs.push(WFF::Equal(
            Box::new(Term::ByteOr(
              Box::new(Term::InputTerm { stack_idx: 0, byte_idx: *byte_idx, value: *a }),
              Box::new(Term::InputTerm { stack_idx: 1, byte_idx: *byte_idx, value: *b }),
            )),
            Box::new(Term::OutputTerm { stack_idx: 0, byte_idx: *byte_idx, value: *a | *b }),
          ));
        }
        Proof::ByteXorSym { byte_idx, a, b } => {
          wffs.push(WFF::Equal(
            Box::new(Term::ByteXor(
              Box::new(Term::InputTerm { stack_idx: 0, byte_idx: *byte_idx, value: *a }),
              Box::new(Term::InputTerm { stack_idx: 1, byte_idx: *byte_idx, value: *b }),
            )),
            Box::new(Term::OutputTerm { stack_idx: 0, byte_idx: *byte_idx, value: *a ^ *b }),
          ));
        }
        Proof::ByteNotSym { byte_idx, a } => {
          wffs.push(WFF::Equal(
            Box::new(Term::ByteXor(
              Box::new(Term::InputTerm { stack_idx: 0, byte_idx: *byte_idx, value: *a }),
              Box::new(Term::Byte(0xFF)),
            )),
            Box::new(Term::OutputTerm { stack_idx: 0, byte_idx: *byte_idx, value: *a ^ 0xFF }),
          ));
        }
        Proof::AddByteSym { byte_idx, a, b } => {
          wffs.push(WFF::Equal(
            Box::new(Term::ByteAdd(
              Box::new(Term::InputTerm { stack_idx: 0, byte_idx: *byte_idx, value: *a }),
              Box::new(Term::InputTerm { stack_idx: 1, byte_idx: *byte_idx, value: *b }),
              Box::new(Term::CarryTerm { byte_idx: *byte_idx }),
            )),
            // Output value is (a + b + carry) % 256; carry is unknown at this point
            // (provided by a separate CarryEq proof), so use 0 as placeholder.
            Box::new(Term::OutputTerm { stack_idx: 0, byte_idx: *byte_idx, value: 0 }),
          ));
        }
        Proof::SubByteSym { byte_idx, b, c } => {
          wffs.push(WFF::Equal(
            Box::new(Term::ByteAdd(
              Box::new(Term::InputTerm { stack_idx: 1, byte_idx: *byte_idx, value: *b }),
              Box::new(Term::OutputTerm { stack_idx: 0, byte_idx: *byte_idx, value: *c }),
              Box::new(Term::CarryTerm { byte_idx: *byte_idx }),
            )),
            // a = b + c + carry; carry unknown, use 0 as placeholder.
            Box::new(Term::InputTerm { stack_idx: 0, byte_idx: *byte_idx, value: 0 }),
          ));
        }
        Proof::CarryEq { byte_idx, value } => {
          wffs.push(WFF::Equal(
            Box::new(Term::CarryTerm { byte_idx: *byte_idx }),
            Box::new(Term::Byte(*value)),
          ));
        }
        // ── Syntactic (value-free) limb-level proofs ────────────────────
        Proof::CarryLimbZero => {
          wffs.push(WFF::Equal(
            Box::new(Term::CarryLimb { limb_idx: 0 }),
            Box::new(Term::Bool(false)),
          ));
        }
        Proof::U29AddSym { limb_idx: j } => {
          let mk_a = || Box::new(Term::InputLimb29 { stack_idx: 0, limb_idx: *j });
          let mk_b = || Box::new(Term::InputLimb29 { stack_idx: 1, limb_idx: *j });
          let mk_cin = || Box::new(Term::CarryLimb { limb_idx: *j });
          wffs.push(WFF::And(
            Box::new(WFF::Equal(
              Box::new(Term::Add29(mk_a(), mk_b(), mk_cin())),
              Box::new(Term::OutputLimb29 { stack_idx: 0, limb_idx: *j }),
            )),
            Box::new(WFF::Equal(
              Box::new(Term::Add29Carry(mk_a(), mk_b(), mk_cin())),
              Box::new(Term::CarryLimb { limb_idx: j + 1 }),
            )),
          ));
        }
        Proof::U24AddSym => {
          wffs.push(WFF::Equal(
            Box::new(Term::Add24(
              Box::new(Term::InputLimb24 { stack_idx: 0 }),
              Box::new(Term::InputLimb24 { stack_idx: 1 }),
              Box::new(Term::CarryLimb { limb_idx: 8 }),
            )),
            Box::new(Term::OutputLimb24 { stack_idx: 0 }),
          ));
        }
        Proof::U29SubSym { limb_idx: j } => {
          // Verifies a - b = c as b + c = a at the limb level.
          let mk_b = || Box::new(Term::InputLimb29 { stack_idx: 1, limb_idx: *j });
          let mk_c = || Box::new(Term::OutputLimb29 { stack_idx: 0, limb_idx: *j });
          let mk_cin = || Box::new(Term::CarryLimb { limb_idx: *j });
          wffs.push(WFF::And(
            Box::new(WFF::Equal(
              Box::new(Term::Add29(mk_b(), mk_c(), mk_cin())),
              Box::new(Term::InputLimb29 { stack_idx: 0, limb_idx: *j }),
            )),
            Box::new(WFF::Equal(
              Box::new(Term::Add29Carry(mk_b(), mk_c(), mk_cin())),
              Box::new(Term::CarryLimb { limb_idx: j + 1 }),
            )),
          ));
        }
        Proof::U24SubSym => {
          wffs.push(WFF::Equal(
            Box::new(Term::Add24(
              Box::new(Term::InputLimb24 { stack_idx: 1 }),
              Box::new(Term::OutputLimb24 { stack_idx: 0 }),
              Box::new(Term::CarryLimb { limb_idx: 8 }),
            )),
            Box::new(Term::InputLimb24 { stack_idx: 0 }),
          ));
        }
      },

      // ── Continuations ─────────────────────────────────────────────────
      Task::AndIntro => {
        let wff2 = wffs.pop().expect("AndIntro: result stack underflow");
        let wff1 = wffs.pop().expect("AndIntro: result stack underflow");
        wffs.push(WFF::And(Box::new(wff1), Box::new(wff2)));
      }
      Task::EqSym => {
        let wff = wffs.pop().expect("EqSym: result stack underflow");
        match wff {
          WFF::Equal(a, b) => wffs.push(WFF::Equal(b, a)),
          _ => return Err(VerifyError::UnexpectedProofVariant { expected: "equality proof" }),
        }
      }
      Task::EqTrans => {
        let wff2 = wffs.pop().expect("EqTrans: result stack underflow");
        let wff1 = wffs.pop().expect("EqTrans: result stack underflow");
        match (wff1, wff2) {
          (WFF::Equal(a, b), WFF::Equal(b2, c)) if *b == *b2 => {
            wffs.push(WFF::Equal(a, c));
          }
          (WFF::Equal(_, _), WFF::Equal(_, _)) => {
            return Err(VerifyError::TransitivityMismatch);
          }
          _ => return Err(VerifyError::UnexpectedProofVariant { expected: "equality proof" }),
        }
      }
      Task::ByteAddThird { a, b } => {
        let wff = wffs.pop().expect("ByteAddThird: result stack underflow");
        match wff {
          WFF::Equal(c1, c2) => wffs.push(WFF::Equal(
            Box::new(Term::ByteAdd(Box::new(Term::Byte(a)), Box::new(Term::Byte(b)), c1)),
            Box::new(Term::ByteAdd(Box::new(Term::Byte(a)), Box::new(Term::Byte(b)), c2)),
          )),
          _ => return Err(VerifyError::UnexpectedProofVariant { expected: "equality proof" }),
        }
      }
      Task::ByteAddCarryThird { a, b } => {
        let wff = wffs.pop().expect("ByteAddCarryThird: result stack underflow");
        match wff {
          WFF::Equal(c1, c2) => wffs.push(WFF::Equal(
            Box::new(Term::ByteAddCarry(Box::new(Term::Byte(a)), Box::new(Term::Byte(b)), c1)),
            Box::new(Term::ByteAddCarry(Box::new(Term::Byte(a)), Box::new(Term::Byte(b)), c2)),
          )),
          _ => return Err(VerifyError::UnexpectedProofVariant { expected: "equality proof" }),
        }
      }
    }
  }

  wffs.pop().ok_or(VerifyError::UnexpectedProofVariant { expected: "proof result" })
}

// ============================================================
// Word-level functions
// ============================================================

/// Build a 32-element array of `InputTerm` bytes for the `stack_idx`-th stack input word.
///
/// `result[j]` = byte `j` (big-endian, 0 = MSB) of the `stack_idx`-th input.
/// `word[j]` is the claimed concrete byte value, committed in the proof trace.
pub fn input_word(stack_idx: u8, word: &[u8; 32]) -> [Box<Term>; 32] {
  array::from_fn(|j| Box::new(Term::InputTerm { stack_idx, byte_idx: j as u8, value: word[j] }))
}

/// Build a 32-element array of `OutputTerm` bytes for the `stack_idx`-th stack output word.
pub fn output_word(stack_idx: u8, word: &[u8; 32]) -> [Box<Term>; 32] {
  array::from_fn(|j| Box::new(Term::OutputTerm { stack_idx, byte_idx: j as u8, value: word[j] }))
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

pub fn mul_u256_mod(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
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

/// Directly build the WFF for `Proof::U29AddEq(a29, b29, cin, c29)` without
/// going through `infer_proof`.  Produces the same four `WFF::Equal(Byte(x),
/// Byte(y))` leaves that `infer_proof(U29AddEq {...})` returns.
#[inline]
fn wff_u29_add_eq_direct(a29: u32, b29: u32, cin: bool, c29: u32) -> WFF {
  let sum = (a29 + b29 + cin as u32) & ((1u32 << 29) - 1);
  and_wffs(vec![
    WFF::Equal(Box::new(Term::Byte((sum & 0xFF) as u8)),           Box::new(Term::Byte((c29 & 0xFF) as u8))),
    WFF::Equal(Box::new(Term::Byte(((sum >> 8) & 0xFF) as u8)),    Box::new(Term::Byte(((c29 >> 8) & 0xFF) as u8))),
    WFF::Equal(Box::new(Term::Byte(((sum >> 16) & 0xFF) as u8)),   Box::new(Term::Byte(((c29 >> 16) & 0xFF) as u8))),
    WFF::Equal(Box::new(Term::Byte(((sum >> 24) & 0x1F) as u8)),   Box::new(Term::Byte(((c29 >> 24) & 0x1F) as u8))),
  ])
}

/// Directly build the WFF for `Proof::U24AddEq(a24, b24, cin, c24)`.
#[inline]
fn wff_u24_add_eq_direct(a24: u32, b24: u32, cin: bool, c24: u32) -> WFF {
  let sum = (a24 + b24 + cin as u32) & ((1u32 << 24) - 1);
  and_wffs(vec![
    WFF::Equal(Box::new(Term::Byte((sum & 0xFF) as u8)),           Box::new(Term::Byte((c24 & 0xFF) as u8))),
    WFF::Equal(Box::new(Term::Byte(((sum >> 8) & 0xFF) as u8)),    Box::new(Term::Byte(((c24 >> 8) & 0xFF) as u8))),
    WFF::Equal(Box::new(Term::Byte(((sum >> 16) & 0xFF) as u8)),   Box::new(Term::Byte(((c24 >> 16) & 0xFF) as u8))),
  ])
}

pub fn wff_add(a: &[u8; 32], b: &[u8; 32], c: &[u8; 32]) -> WFF {
  let mut leaves = Vec::with_capacity(9);
  let mut carry = false;

  for limb in 0..8 {
    let base = 29 * limb;
    let a29 = extract_limb_le_bits(a, base, 29);
    let b29 = extract_limb_le_bits(b, base, 29);
    let c29 = extract_limb_le_bits(c, base, 29);
    let cin = carry;
    let total = a29 + b29 + cin as u32;
    carry = total >= (1u32 << 29);
    leaves.push(wff_u29_add_eq_direct(a29, b29, cin, c29));
  }

  let a24 = extract_limb_le_bits(a, 232, 24);
  let b24 = extract_limb_le_bits(b, 232, 24);
  let c24 = extract_limb_le_bits(c, 232, 24);
  leaves.push(wff_u24_add_eq_direct(a24, b24, carry, c24));

  and_wffs(leaves)
}

pub fn wff_sub(a: &[u8; 32], b: &[u8; 32], c: &[u8; 32]) -> WFF {
  wff_add(b, c, a)
}

/// Directly build the WFF for `Proof::U15MulEq(a, b)` without going through
/// `infer_proof`.  Produces the same eight `WFF::Equal` nodes that
/// `infer_proof(U15MulEq { a, b })` returns.
#[inline]
fn wff_u15_mul_eq_direct(a: u16, b: u16) -> WFF {
  let a_lo = (a & 0xFF) as u8;
  let a_hi = ((a >> 8) & 0x7F) as u8;
  let b_lo = (b & 0xFF) as u8;
  let b_hi = ((b >> 8) & 0x7F) as u8;
  let p00 = a_lo as u16 * b_lo as u16;
  let p01 = a_lo as u16 * b_hi as u16;
  let p10 = a_hi as u16 * b_lo as u16;
  let p11 = a_hi as u16 * b_hi as u16;
  and_wffs(vec![
    WFF::Equal(Box::new(Term::ByteMulLow(Box::new(Term::Byte(a_lo)), Box::new(Term::Byte(b_lo)))),  Box::new(Term::Byte((p00 & 0xFF) as u8))),
    WFF::Equal(Box::new(Term::ByteMulHigh(Box::new(Term::Byte(a_lo)), Box::new(Term::Byte(b_lo)))), Box::new(Term::Byte((p00 >> 8)  as u8))),
    WFF::Equal(Box::new(Term::ByteMulLow(Box::new(Term::Byte(a_lo)), Box::new(Term::Byte(b_hi)))),  Box::new(Term::Byte((p01 & 0xFF) as u8))),
    WFF::Equal(Box::new(Term::ByteMulHigh(Box::new(Term::Byte(a_lo)), Box::new(Term::Byte(b_hi)))), Box::new(Term::Byte((p01 >> 8)  as u8))),
    WFF::Equal(Box::new(Term::ByteMulLow(Box::new(Term::Byte(a_hi)), Box::new(Term::Byte(b_lo)))),  Box::new(Term::Byte((p10 & 0xFF) as u8))),
    WFF::Equal(Box::new(Term::ByteMulHigh(Box::new(Term::Byte(a_hi)), Box::new(Term::Byte(b_lo)))), Box::new(Term::Byte((p10 >> 8)  as u8))),
    WFF::Equal(Box::new(Term::ByteMulLow(Box::new(Term::Byte(a_hi)), Box::new(Term::Byte(b_hi)))),  Box::new(Term::Byte((p11 & 0xFF) as u8))),
    WFF::Equal(Box::new(Term::ByteMulHigh(Box::new(Term::Byte(a_hi)), Box::new(Term::Byte(b_hi)))), Box::new(Term::Byte((p11 >> 8)  as u8))),
  ])
}

pub fn wff_mul(a: &[u8; 32], b: &[u8; 32], _c: &[u8; 32]) -> WFF {
  let expected = mul_u256_mod(a, b);
  #[cfg(debug_assertions)]
  debug_assert_mul_consistency(a, b, &expected);

  let a_limbs = word_to_u15_limbs(a);
  let b_limbs = word_to_u15_limbs(b);
  let b_top = b_limbs
    .iter()
    .rposition(|&v| v != 0)
    .map(|idx| idx + 1)
    .unwrap_or(0);

  if b_top == 0 {
    // a * b = 0: single trivially-true leaf mirrors prove_mul's fallback.
    return wff_u29_add_eq_direct(0, 0, false, 0);
  }

  let mut leaves = Vec::new();
  for (i, &av) in a_limbs.iter().enumerate() {
    if av == 0 {
      continue;
    }
    let j_max = (17 - i + 1).min(b_top);
    for j in 0..j_max {
      let bv = b_limbs[j];
      if bv == 0 {
        continue;
      }
      leaves.push(wff_u15_mul_eq_direct(av, bv));
    }
  }

  if leaves.is_empty() {
    // a * b = 0: single trivially-true leaf mirrors prove_mul's fallback.
    wff_u29_add_eq_direct(0, 0, false, 0)
  } else {
    and_wffs(leaves)
  }
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

  let leaves = mul_local_proof_leaves(a, b);
  if leaves.is_empty() {
    // a * b = 0 (one operand is zero): emit a single trivially-true leaf.
    Proof::U29AddEq(0, 0, false, 0)
  } else {
    and_proofs(leaves)
  }
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
  let a_limbs = word_to_u15_limbs(a);
  let b_limbs = word_to_u15_limbs(b);

  // b의 최고 유효 limb 인덱스를 미리 계산해 내부 루프 상한을 줄인다.
  let b_top = b_limbs
    .iter()
    .rposition(|&v| v != 0)
    .map(|idx| idx + 1)
    .unwrap_or(0);

  if b_top == 0 {
    return Vec::new();
  }

  let mut leaves = Vec::new();

  for (i, &av) in a_limbs.iter().enumerate() {
    if av == 0 {
      continue;
    }
    // i + j <= 17 이고 j < b_top 이어야 하므로 내부 루프 상한을 좁힌다.
    let j_max = (17 - i + 1).min(b_top);
    for j in 0..j_max {
      let bv = b_limbs[j];
      if bv == 0 {
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
    // ── 15-bit MUL intermediates ──────────────────────────────────────────
    Term::U15(v) => {
      let key = (OP_U15_LIT, 0, 0, 0, *v as u32, 0, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_U15_LIT,
        scalar0: *v as u32,
        value: *v as u32,
        ret_ty: RET_U15,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Term::U15MulLow(a, b) => {
      let ai = compile_term_inner(a, rows, memo);
      let bi = compile_term_inner(b, rows, memo);
      let key = (OP_U15_MUL_LOW, ai, bi, 0, 0, 0, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let val = (rows[ai as usize].value * rows[bi as usize].value) & 0x7FFF;
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_U15_MUL_LOW,
        arg0: ai,
        arg1: bi,
        value: val,
        ret_ty: RET_U15,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Term::U15MulHigh(a, b) => {
      let ai = compile_term_inner(a, rows, memo);
      let bi = compile_term_inner(b, rows, memo);
      let key = (OP_U15_MUL_HIGH, ai, bi, 0, 0, 0, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let val = (rows[ai as usize].value * rows[bi as usize].value) >> 15;
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_U15_MUL_HIGH,
        arg0: ai,
        arg1: bi,
        value: val,
        ret_ty: RET_U15,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Term::PartialMul15 { col_idx } => {
      let key = (OP_PARTIAL_MUL15, 0, 0, 0, *col_idx as u32, 0, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_PARTIAL_MUL15,
        scalar0: *col_idx as u32,
        ret_ty: RET_U15,
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
    // ── Symbolic variable terms (value = claimed byte; verified by stack-log LogUp AIR) ──
    Term::InputTerm { stack_idx, byte_idx, value } => {
      let key = (OP_INPUT_TERM, 0, 0, 0, *stack_idx as u32, *byte_idx as u32, *value as u32);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_INPUT_TERM,
        scalar0: *stack_idx as u32,
        scalar1: *byte_idx as u32,
        value: *value as u32,
        ret_ty: RET_BYTE,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Term::OutputTerm { stack_idx, byte_idx, value } => {
      let key = (OP_OUTPUT_TERM, 0, 0, 0, *stack_idx as u32, *byte_idx as u32, *value as u32);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_OUTPUT_TERM,
        scalar0: *stack_idx as u32,
        scalar1: *byte_idx as u32,
        value: *value as u32,
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
    Term::CarryTerm { byte_idx } => {
      let key = (OP_CARRY_TERM, 0, 0, 0, *byte_idx as u32, 0, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_CARRY_TERM,
        scalar0: *byte_idx as u32,
        ret_ty: RET_BYTE,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Term::InputLimb29 { stack_idx, limb_idx } => {
      let key = (OP_INPUT_LIMB29, 0, 0, 0, *stack_idx as u32, *limb_idx as u32, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_INPUT_LIMB29,
        scalar0: *stack_idx as u32,
        scalar1: *limb_idx as u32,
        ret_ty: RET_U29,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Term::OutputLimb29 { stack_idx, limb_idx } => {
      let key = (OP_OUTPUT_LIMB29, 0, 0, 0, *stack_idx as u32, *limb_idx as u32, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_OUTPUT_LIMB29,
        scalar0: *stack_idx as u32,
        scalar1: *limb_idx as u32,
        ret_ty: RET_U29,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Term::InputLimb24 { stack_idx } => {
      let key = (OP_INPUT_LIMB24, 0, 0, 0, *stack_idx as u32, 0, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_INPUT_LIMB24,
        scalar0: *stack_idx as u32,
        ret_ty: RET_U24,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Term::OutputLimb24 { stack_idx } => {
      let key = (OP_OUTPUT_LIMB24, 0, 0, 0, *stack_idx as u32, 0, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_OUTPUT_LIMB24,
        scalar0: *stack_idx as u32,
        ret_ty: RET_U24,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Term::CarryLimb { limb_idx } => {
      let key = (OP_CARRY_LIMB, 0, 0, 0, *limb_idx as u32, 0, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_CARRY_LIMB,
        scalar0: *limb_idx as u32,
        ret_ty: RET_BOOL,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Term::Add29(a, b, cin) => {
      let ai = compile_term_inner(a, rows, memo);
      let bi = compile_term_inner(b, rows, memo);
      let ci = compile_term_inner(cin, rows, memo);
      let key = (OP_ADD29, ai, bi, ci, 0, 0, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_ADD29,
        arg0: ai,
        arg1: bi,
        arg2: ci,
        ret_ty: RET_U29,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Term::Add29Carry(a, b, cin) => {
      let ai = compile_term_inner(a, rows, memo);
      let bi = compile_term_inner(b, rows, memo);
      let ci = compile_term_inner(cin, rows, memo);
      let key = (OP_ADD29_CARRY, ai, bi, ci, 0, 0, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_ADD29_CARRY,
        arg0: ai,
        arg1: bi,
        arg2: ci,
        ret_ty: RET_BOOL,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Term::Add24(a, b, cin) => {
      let ai = compile_term_inner(a, rows, memo);
      let bi = compile_term_inner(b, rows, memo);
      let ci = compile_term_inner(cin, rows, memo);
      let key = (OP_ADD24, ai, bi, ci, 0, 0, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_ADD24,
        arg0: ai,
        arg1: bi,
        arg2: ci,
        ret_ty: RET_U24,
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
      // AND_INTRO is a pure structural node that adds no LUT or StackIR
      // constraints. Skipping the row emit halves the trace for leaf-only proofs
      // (e.g. MUL: 171 U15MulEq rows instead of 341), packing ~2x more
      // instructions per STARK batch. Both children MUST still be compiled
      // so their leaf rows are emitted and memoised.
      compile_proof_inner(p1, rows, memo);
      compile_proof_inner(p2, rows, memo)
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

    Proof::U15MulLowEq(a, b) => {
      let key = (OP_U15_MUL_LOW_EQ, 0, 0, 0, *a as u32, *b as u32, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_U15_MUL_LOW_EQ,
        scalar0: *a as u32,
        scalar1: *b as u32,
        value: (*a as u32 * *b as u32) & 0x7FFF,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Proof::U15MulHighEq(a, b) => {
      let key = (OP_U15_MUL_HIGH_EQ, 0, 0, 0, *a as u32, *b as u32, 0);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_U15_MUL_HIGH_EQ,
        scalar0: *a as u32,
        scalar1: *b as u32,
        value: (*a as u32 * *b as u32) >> 15,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Proof::PartialMul15Bind { col_idx, lo, hi } => {
      let key = (OP_PARTIAL_MUL15_BIND, 0, 0, 0, *col_idx as u32, *lo as u32, *hi as u32);
      if let Some(&cached) = memo.get(&key) {
        return cached;
      }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_PARTIAL_MUL15_BIND,
        scalar0: *col_idx as u32,
        scalar1: *lo as u32,
        scalar2: *hi as u32,
        value: *lo as u32,
        ret_ty: RET_WFF_AND,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }

    // ── Per-opcode axiom leaf rows: compile as EqRefl(OutputTerm{0,0}) ────────────────
    Proof::PushAxiom
    | Proof::DupAxiom { .. }
    | Proof::SwapAxiom { .. }
    | Proof::StructuralAxiom { .. }
    | Proof::MloadAxiom
    | Proof::MstoreAxiom { .. }
    | Proof::MemCopyAxiom { .. }
    | Proof::SloadAxiom
    | Proof::SstoreAxiom
    | Proof::TransientAxiom { .. }
    | Proof::KeccakAxiom
    | Proof::EnvAxiom { .. }
    | Proof::ExternalStateAxiom { .. }
    | Proof::TerminateAxiom { .. }
    | Proof::CallAxiom { .. }
    | Proof::CreateAxiom { .. }
    | Proof::SelfdestructAxiom
    | Proof::LogAxiom { .. } => push_output_eq_row(rows, memo),
    Proof::InputEq { stack_idx, byte_idx, value } => {
      let key = (OP_INPUT_EQ, 0, 0, 0, *stack_idx as u32, *byte_idx as u32, *value as u32);
      if let Some(&cached) = memo.get(&key) { return cached; }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_INPUT_EQ,
        scalar0: *stack_idx as u32,
        scalar1: *byte_idx as u32,
        scalar2: *value as u32,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Proof::OutputEq { stack_idx, byte_idx, value } => {
      let key = (OP_OUTPUT_EQ, 0, 0, 0, *stack_idx as u32, *byte_idx as u32, *value as u32);
      if let Some(&cached) = memo.get(&key) { return cached; }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_OUTPUT_EQ,
        scalar0: *stack_idx as u32,
        scalar1: *byte_idx as u32,
        scalar2: *value as u32,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Proof::PcBeforeEq { byte_idx, value } => {
      let key = (OP_PC_BEFORE_EQ, 0, 0, 0, *byte_idx as u32, *value as u32, 0);
      if let Some(&cached) = memo.get(&key) { return cached; }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_PC_BEFORE_EQ,
        scalar0: *byte_idx as u32,
        scalar1: *value as u32,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Proof::PcStep { instr_size } => {
      let key = (OP_PC_STEP, 0, 0, 0, *instr_size, 0, 0);
      if let Some(&cached) = memo.get(&key) { return cached; }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_PC_STEP,
        scalar0: *instr_size,
        ret_ty: RET_WFF_AND,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    // ── Symbolic byte-level ops ──────────────────────────────────────────────
    Proof::ByteAndSym { byte_idx, a, b } => {
      let key = (OP_BYTE_AND_SYM, 0, 0, 0, *byte_idx as u32, *a as u32, *b as u32);
      if let Some(&cached) = memo.get(&key) { return cached; }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_BYTE_AND_SYM,
        scalar0: *byte_idx as u32,
        scalar1: *a as u32,
        scalar2: *b as u32,
        value: (*a & *b) as u32,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Proof::ByteOrSym { byte_idx, a, b } => {
      let key = (OP_BYTE_OR_SYM, 0, 0, 0, *byte_idx as u32, *a as u32, *b as u32);
      if let Some(&cached) = memo.get(&key) { return cached; }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_BYTE_OR_SYM,
        scalar0: *byte_idx as u32,
        scalar1: *a as u32,
        scalar2: *b as u32,
        value: (*a | *b) as u32,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Proof::ByteXorSym { byte_idx, a, b } => {
      let key = (OP_BYTE_XOR_SYM, 0, 0, 0, *byte_idx as u32, *a as u32, *b as u32);
      if let Some(&cached) = memo.get(&key) { return cached; }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_BYTE_XOR_SYM,
        scalar0: *byte_idx as u32,
        scalar1: *a as u32,
        scalar2: *b as u32,
        value: (*a ^ *b) as u32,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Proof::ByteNotSym { byte_idx, a } => {
      let key = (OP_BYTE_NOT_SYM, 0, 0, 0, *byte_idx as u32, *a as u32, 0);
      if let Some(&cached) = memo.get(&key) { return cached; }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_BYTE_NOT_SYM,
        scalar0: *byte_idx as u32,
        scalar1: *a as u32,
        value: (*a ^ 0xFF) as u32,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Proof::AddByteSym { byte_idx, a, b } => {
      let key = (OP_ADD_BYTE_SYM, 0, 0, 0, *byte_idx as u32, *a as u32, *b as u32);
      if let Some(&cached) = memo.get(&key) { return cached; }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_ADD_BYTE_SYM,
        scalar0: *byte_idx as u32,
        scalar1: *a as u32,
        scalar2: *b as u32,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Proof::SubByteSym { byte_idx, b, c } => {
      let key = (OP_SUB_BYTE_SYM, 0, 0, 0, *byte_idx as u32, *b as u32, *c as u32);
      if let Some(&cached) = memo.get(&key) { return cached; }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_SUB_BYTE_SYM,
        scalar0: *byte_idx as u32,
        scalar1: *b as u32,
        scalar2: *c as u32,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Proof::CarryEq { byte_idx, value } => {
      let key = (OP_CARRY_EQ, 0, 0, 0, *byte_idx as u32, *value as u32, 0);
      if let Some(&cached) = memo.get(&key) { return cached; }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_CARRY_EQ,
        scalar0: *byte_idx as u32,
        scalar1: *value as u32,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    // ── Syntactic (value-free) limb-level proof variants ─────────────────
    Proof::CarryLimbZero => {
      let carry_idx = compile_term_inner(&Term::CarryLimb { limb_idx: 0 }, rows, memo);
      let zero_idx = compile_term_inner(&Term::Bool(false), rows, memo);
      let key = (OP_CARRY_LIMB_ZERO, carry_idx, zero_idx, 0, 0, 0, 0);
      if let Some(&cached) = memo.get(&key) { return cached; }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_CARRY_LIMB_ZERO,
        arg0: carry_idx,
        arg1: zero_idx,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Proof::U29AddSym { limb_idx } => {
      // arg0=InputLimb29{0,j}, arg1=InputLimb29{1,j}, arg2=CarryLimb{j}
      // scalar0=OutputLimb29{0,j} row ref, scalar1=CarryLimb{j+1} row ref, scalar2=limb_idx
      let a_idx = compile_term_inner(&Term::InputLimb29 { stack_idx: 0, limb_idx: *limb_idx }, rows, memo);
      let b_idx = compile_term_inner(&Term::InputLimb29 { stack_idx: 1, limb_idx: *limb_idx }, rows, memo);
      let cin_idx = compile_term_inner(&Term::CarryLimb { limb_idx: *limb_idx }, rows, memo);
      let c_idx = compile_term_inner(&Term::OutputLimb29 { stack_idx: 0, limb_idx: *limb_idx }, rows, memo);
      let cout_idx = compile_term_inner(&Term::CarryLimb { limb_idx: limb_idx + 1 }, rows, memo);
      let key = (OP_U29_ADD_SYM, a_idx, b_idx, cin_idx, c_idx, cout_idx, 0);
      if let Some(&cached) = memo.get(&key) { return cached; }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_U29_ADD_SYM,
        scalar0: c_idx,
        scalar1: cout_idx,
        scalar2: *limb_idx as u32,
        arg0: a_idx,
        arg1: b_idx,
        arg2: cin_idx,
        ret_ty: RET_WFF_AND,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Proof::U24AddSym => {
      let a_idx = compile_term_inner(&Term::InputLimb24 { stack_idx: 0 }, rows, memo);
      let b_idx = compile_term_inner(&Term::InputLimb24 { stack_idx: 1 }, rows, memo);
      let cin_idx = compile_term_inner(&Term::CarryLimb { limb_idx: 8 }, rows, memo);
      let c_idx = compile_term_inner(&Term::OutputLimb24 { stack_idx: 0 }, rows, memo);
      let key = (OP_U24_ADD_SYM, a_idx, b_idx, cin_idx, c_idx, 0, 0);
      if let Some(&cached) = memo.get(&key) { return cached; }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_U24_ADD_SYM,
        scalar0: c_idx,
        arg0: a_idx,
        arg1: b_idx,
        arg2: cin_idx,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Proof::U29SubSym { limb_idx } => {
      // b + c = a at limb level: arg0=InputLimb29{1,j}(b), arg1=OutputLimb29{0,j}(c), arg2=CarryLimb{j}
      // scalar0=InputLimb29{0,j}(a) result, scalar1=CarryLimb{j+1}, scalar2=limb_idx
      let b_idx = compile_term_inner(&Term::InputLimb29 { stack_idx: 1, limb_idx: *limb_idx }, rows, memo);
      let c_idx = compile_term_inner(&Term::OutputLimb29 { stack_idx: 0, limb_idx: *limb_idx }, rows, memo);
      let cin_idx = compile_term_inner(&Term::CarryLimb { limb_idx: *limb_idx }, rows, memo);
      let a_idx = compile_term_inner(&Term::InputLimb29 { stack_idx: 0, limb_idx: *limb_idx }, rows, memo);
      let cout_idx = compile_term_inner(&Term::CarryLimb { limb_idx: limb_idx + 1 }, rows, memo);
      let key = (OP_U29_SUB_SYM, b_idx, c_idx, cin_idx, a_idx, cout_idx, 0);
      if let Some(&cached) = memo.get(&key) { return cached; }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_U29_SUB_SYM,
        scalar0: a_idx,
        scalar1: cout_idx,
        scalar2: *limb_idx as u32,
        arg0: b_idx,
        arg1: c_idx,
        arg2: cin_idx,
        ret_ty: RET_WFF_AND,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
    Proof::U24SubSym => {
      let b_idx = compile_term_inner(&Term::InputLimb24 { stack_idx: 1 }, rows, memo);
      let c_idx = compile_term_inner(&Term::OutputLimb24 { stack_idx: 0 }, rows, memo);
      let cin_idx = compile_term_inner(&Term::CarryLimb { limb_idx: 8 }, rows, memo);
      let a_idx = compile_term_inner(&Term::InputLimb24 { stack_idx: 0 }, rows, memo);
      let key = (OP_U24_SUB_SYM, b_idx, c_idx, cin_idx, a_idx, 0, 0);
      if let Some(&cached) = memo.get(&key) { return cached; }
      let idx = rows.len() as u32;
      rows.push(ProofRow {
        op: OP_U24_SUB_SYM,
        scalar0: a_idx,
        arg0: b_idx,
        arg1: c_idx,
        arg2: cin_idx,
        ret_ty: RET_WFF_EQ,
        ..Default::default()
      });
      memo.insert(key, idx);
      idx
    }
  }
}

/// Emit an EqRefl row for OutputTerm{stack_idx: 0, byte_idx: 0}, deduplicating via memo.
fn push_output_eq_row(rows: &mut Vec<ProofRow>, memo: &mut Memo) -> u32 {
  let ti = compile_term_inner(
    &Term::OutputTerm { stack_idx: 0, byte_idx: 0, value: 0 },
    rows,
    memo,
  );
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
      // ── Symbolic term leaves (value is a stack/PC slot; not checked here) ──
      OP_INPUT_TERM | OP_OUTPUT_TERM | OP_PC_BEFORE | OP_PC_AFTER => {}
      // ── 29/24-bit limb symbolic term leaves (value resolved at batch-verification time) ──
      OP_INPUT_LIMB29 | OP_OUTPUT_LIMB29 | OP_INPUT_LIMB24 | OP_OUTPUT_LIMB24 => {}
      // ── Limb arithmetic term leaves (syntactic; LogUp provides values) ──
      OP_CARRY_LIMB | OP_ADD29 | OP_ADD29_CARRY | OP_ADD24 => {}
      // ── Syntactic limb-level proof rows (correctness via LogUp) ──
      OP_CARRY_LIMB_ZERO | OP_U29_ADD_SYM | OP_U24_ADD_SYM | OP_U29_SUB_SYM | OP_U24_SUB_SYM => {}
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
      OP_INPUT_EQ | OP_OUTPUT_EQ => {
        if row.scalar0 > 255 || row.scalar1 > 255 || row.scalar2 > 255 {
          return Err(VerifyError::ByteDecideFailed);
        }
      }
      OP_PC_BEFORE_EQ => {
        if row.scalar0 > 3 || row.scalar1 > 255 {
          return Err(VerifyError::ByteDecideFailed);
        }
      }
      OP_PC_STEP => {
        // instr_size must be a plausible EVM instruction size (1-based, ≤ 33 for PUSH32)
        if row.scalar0 == 0 || row.scalar0 > 33 {
          return Err(VerifyError::ByteDecideFailed);
        }
      }
      OP_BYTE_AND_SYM => {
        if row.scalar0 > 31 || row.scalar1 > 255 || row.scalar2 > 255 { return Err(VerifyError::ByteDecideFailed); }
        if row.value != (row.scalar1 & row.scalar2) { return Err(VerifyError::ByteDecideFailed); }
      }
      OP_BYTE_OR_SYM => {
        if row.scalar0 > 31 || row.scalar1 > 255 || row.scalar2 > 255 { return Err(VerifyError::ByteDecideFailed); }
        if row.value != (row.scalar1 | row.scalar2) { return Err(VerifyError::ByteDecideFailed); }
      }
      OP_BYTE_XOR_SYM => {
        if row.scalar0 > 31 || row.scalar1 > 255 || row.scalar2 > 255 { return Err(VerifyError::ByteDecideFailed); }
        if row.value != (row.scalar1 ^ row.scalar2) { return Err(VerifyError::ByteDecideFailed); }
      }
      OP_BYTE_NOT_SYM => {
        if row.scalar0 > 31 || row.scalar1 > 255 { return Err(VerifyError::ByteDecideFailed); }
        if row.value != (row.scalar1 ^ 0xFF) { return Err(VerifyError::ByteDecideFailed); }
      }
      OP_ADD_BYTE_SYM => {
        // scalar0=byte_idx, scalar1=a, scalar2=b (witnesses, not in WFF)
        if row.scalar0 > 31 || row.scalar1 > 255 || row.scalar2 > 255 {
          return Err(VerifyError::ByteDecideFailed);
        }
      }
      OP_SUB_BYTE_SYM => {
        if row.scalar0 > 31 || row.scalar1 > 255 || row.scalar2 > 255 {
          return Err(VerifyError::ByteDecideFailed);
        }
      }
      OP_CARRY_EQ => {
        if row.scalar0 > 31 || row.scalar1 > 1 {
          return Err(VerifyError::ByteDecideFailed);
        }
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
// Stack I/O and PC binding proofs
// ============================================================

/// Conjunction of `InputEq` axioms for every byte of every input word.
///
/// Proves `InputTerm{s,j} = Byte(inputs[s][j])` for all `s` in `0..inputs.len()`
/// and `j` in `0..32`.  Soundness is enforced by the stack consistency AIR.
pub fn prove_stack_inputs(inputs: &[[u8; 32]]) -> Proof {
  let leaves: Vec<Proof> = inputs
    .iter()
    .enumerate()
    .flat_map(|(s, word)| {
      (0u8..32).map(move |j| Proof::InputEq {
        stack_idx: s as u8,
        byte_idx: j,
        value: word[j as usize],
      })
    })
    .collect();
  if leaves.is_empty() {
    return Proof::EqRefl(Term::Bool(true));
  }
  and_proofs(leaves)
}

/// Conjunction of `OutputEq` axioms for every byte of every output word.
pub fn prove_stack_outputs(outputs: &[[u8; 32]]) -> Proof {
  let leaves: Vec<Proof> = outputs
    .iter()
    .enumerate()
    .flat_map(|(s, word)| {
      (0u8..32).map(move |j| Proof::OutputEq {
        stack_idx: s as u8,
        byte_idx: j,
        value: word[j as usize],
      })
    })
    .collect();
  if leaves.is_empty() {
    return Proof::EqRefl(Term::Bool(true));
  }
  and_proofs(leaves)
}

/// Conjunction of `PcBeforeEq` axioms for the 4-byte big-endian representation of `pc`.
pub fn prove_pc_before(pc: u32) -> Proof {
  let bytes = pc.to_be_bytes();
  let leaves: Vec<Proof> = (0u8..4)
    .map(|j| Proof::PcBeforeEq { byte_idx: j, value: bytes[j as usize] })
    .collect();
  and_proofs(leaves)
}

/// Proof that `PcAfter = PcBefore + instr_size` via carry-chain arithmetic.
///
/// `instr_size` must match the number of bytes consumed by the instruction (1 for
/// most opcodes, 2..=33 for PUSH1..PUSH32).
pub fn prove_pc_step(instr_size: u32) -> Proof {
  Proof::PcStep { instr_size }
}

/// WFF for stack input bindings: `AND` over `InputTerm{s,j,v} = Byte(v)` where `v = inputs[s][j]`.
pub fn wff_stack_inputs(inputs: &[[u8; 32]]) -> WFF {
  let leaves: Vec<WFF> = inputs
    .iter()
    .enumerate()
    .flat_map(|(s, word)| {
      (0u8..32).map(move |j| WFF::Equal(
        Box::new(Term::InputTerm { stack_idx: s as u8, byte_idx: j, value: word[j as usize] }),
        Box::new(Term::Byte(word[j as usize])),
      ))
    })
    .collect();
  if leaves.is_empty() {
    return WFF::Equal(Box::new(Term::Bool(true)), Box::new(Term::Bool(true)));
  }
  and_wffs(leaves)
}

/// WFF for stack output bindings: `AND` over `OutputTerm{s,j,v} = Byte(v)` where `v = outputs[s][j]`.
pub fn wff_stack_outputs(outputs: &[[u8; 32]]) -> WFF {
  let leaves: Vec<WFF> = outputs
    .iter()
    .enumerate()
    .flat_map(|(s, word)| {
      (0u8..32).map(move |j| WFF::Equal(
        Box::new(Term::OutputTerm { stack_idx: s as u8, byte_idx: j, value: word[j as usize] }),
        Box::new(Term::Byte(word[j as usize])),
      ))
    })
    .collect();
  if leaves.is_empty() {
    return WFF::Equal(Box::new(Term::Bool(true)), Box::new(Term::Bool(true)));
  }
  and_wffs(leaves)
}

/// WFF for PC-before binding: `AND` over `PcBefore{j} = Byte(pc_bytes[j])`.
pub fn wff_pc_before(pc: u32) -> WFF {
  let bytes = pc.to_be_bytes();
  let leaves: Vec<WFF> = (0u8..4)
    .map(|j| WFF::Equal(
      Box::new(Term::PcBefore { byte_idx: j }),
      Box::new(Term::Byte(bytes[j as usize])),
    ))
    .collect();
  and_wffs(leaves)
}

/// WFF expressing `PcAfter = PcBefore + instr_size` via a 4-byte carry chain.
///
/// For each byte index `j` (0 = MSB, 3 = LSB):
/// `PcAfter{j} = ByteAdd(PcBefore{j}, Byte(instr_size_bytes[j]), carry_into_j)`
/// where `carry_into_j` is `ByteAddCarry` from byte `j+1` (with `carry_into_3 = Byte(0)`).
pub fn wff_pc_step(instr_size: u32) -> WFF {
  let sb = instr_size.to_be_bytes(); // sb[0]=MSB … sb[3]=LSB

  // Build carry terms from LSB toward MSB.
  let mk_carry = |j: u8, c_in: Box<Term>| -> Term {
    Term::ByteAddCarry(
      Box::new(Term::PcBefore { byte_idx: j }),
      Box::new(Term::Byte(sb[j as usize])),
      c_in,
    )
  };

  let c3_in = Box::new(Term::Byte(0));
  let c2_in = Box::new(mk_carry(3, c3_in.clone()));
  let c1_in = Box::new(mk_carry(2, c2_in.clone()));
  let c0_in = Box::new(mk_carry(1, c1_in.clone()));

  let eq_j = |j: u8, c_in: Box<Term>| WFF::Equal(
    Box::new(Term::PcAfter { byte_idx: j }),
    Box::new(Term::ByteAdd(
      Box::new(Term::PcBefore { byte_idx: j }),
      Box::new(Term::Byte(sb[j as usize])),
      c_in,
    )),
  );

  WFF::And(
    Box::new(eq_j(0, c0_in)),
    Box::new(WFF::And(
      Box::new(eq_j(1, c1_in)),
      Box::new(WFF::And(
        Box::new(eq_j(2, c2_in)),
        Box::new(eq_j(3, c3_in)),
      )),
    )),
  )
}

// ============================================================
// Symbolic prove_* / wff_* (value-free WFF, witnesses only for verification)
// ============================================================
//
// Design: all input/output references in the WFF use InputTerm / OutputTerm
// symbolic variables.  Concrete byte values are passed ONLY as witnesses so
// that `verify_compiled` and the STARK can check lookup tables.
// Two executions of the same opcode produce *identical* WFF trees, enabling
// opcode-level WFF caching and digest reuse.

/// Symbolic AND — `ByteAnd(InputTerm{0,j}, InputTerm{1,j}) = OutputTerm{0,j}` for j 31..=0.
pub fn prove_and_sym(a: &[u8; 32], b: &[u8; 32]) -> Proof {
  let leaves: Vec<Proof> = (0..32u8)
    .rev()
    .map(|j| Proof::ByteAndSym { byte_idx: j, a: a[j as usize], b: b[j as usize] })
    .collect();
  and_proofs(leaves)
}

/// Symbolic OR — `ByteOr(InputTerm{0,j}, InputTerm{1,j}) = OutputTerm{0,j}` for j 31..=0.
pub fn prove_or_sym(a: &[u8; 32], b: &[u8; 32]) -> Proof {
  let leaves: Vec<Proof> = (0..32u8)
    .rev()
    .map(|j| Proof::ByteOrSym { byte_idx: j, a: a[j as usize], b: b[j as usize] })
    .collect();
  and_proofs(leaves)
}

/// Symbolic XOR — `ByteXor(InputTerm{0,j}, InputTerm{1,j}) = OutputTerm{0,j}` for j 31..=0.
pub fn prove_xor_sym(a: &[u8; 32], b: &[u8; 32]) -> Proof {
  let leaves: Vec<Proof> = (0..32u8)
    .rev()
    .map(|j| Proof::ByteXorSym { byte_idx: j, a: a[j as usize], b: b[j as usize] })
    .collect();
  and_proofs(leaves)
}

/// Symbolic NOT — `ByteXor(InputTerm{0,j}, Byte(0xFF)) = OutputTerm{0,j}` for j 31..=0.
pub fn prove_not_sym(a: &[u8; 32]) -> Proof {
  let leaves: Vec<Proof> = (0..32u8)
    .rev()
    .map(|j| Proof::ByteNotSym { byte_idx: j, a: a[j as usize] })
    .collect();
  and_proofs(leaves)
}

/// Symbolic ADD — `ByteAdd(InputTerm{0,j}, InputTerm{1,j}, Byte(carry_j)) = OutputTerm{0,j}`.
///
/// Carry chain is computed from the concrete `a[j]`, `b[j]` bytes (big-endian, j=0 is MSB).
pub fn prove_add_sym(a: &[u8; 32], b: &[u8; 32]) -> Proof {
  // carry_out[j] = carry OUT of byte j = carry INTO byte j-1 (big-endian).
  // carry_out[31] (LSB) = carry from bit position below, which doesn't exist = 0.
  let mut carry_out = [0u8; 32]; // carry_out[j] = carry produced by byte j
  for j in (0..32usize).rev() {
    let prev_carry = if j < 31 { carry_out[j + 1] } else { 0u8 };
    let s = a[j] as u16 + b[j] as u16 + prev_carry as u16;
    carry_out[j] = (s >> 8) as u8;
  }
  // For each byte j: CarryEq{j, carry_out[j]} ∧ AddByteSym{j, a[j], b[j]}
  // All bundled as a single AND conjunction.
  let mut leaves: Vec<Proof> = Vec::with_capacity(64);
  for j in (0..32u8).rev() {
    // CarryTerm{j} is the carry INTO byte j = carry_out[j+1] (or 0 for LSB)
    let carry_into_j = if j < 31 { carry_out[j as usize + 1] } else { 0u8 };
    leaves.push(Proof::CarryEq { byte_idx: j, value: carry_into_j });
    leaves.push(Proof::AddByteSym { byte_idx: j, a: a[j as usize], b: b[j as usize] });
  }
  and_proofs(leaves)
}

/// Symbolic SUB — `ByteAdd(InputTerm{1,j}, OutputTerm{0,j}, Byte(borrow_j)) = InputTerm{0,j}`.
///
/// Proves `a - b = c` as `b + c + borrow = a` byte-wise.
pub fn prove_sub_sym(a: &[u8; 32], b: &[u8; 32]) -> Proof {
  // Verify a - b = c  as  b + c = a,  reusing CarryTerm.
  let c = sub_u256_be(a, b);
  let mut carry_out = [0u8; 32];
  for j in (0..32usize).rev() {
    let prev_carry = if j < 31 { carry_out[j + 1] } else { 0u8 };
    let s = b[j] as u16 + c[j] as u16 + prev_carry as u16;
    carry_out[j] = (s >> 8) as u8;
  }
  let mut leaves: Vec<Proof> = Vec::with_capacity(64);
  for j in (0..32u8).rev() {
    let carry_into_j = if j < 31 { carry_out[j as usize + 1] } else { 0u8 };
    leaves.push(Proof::CarryEq { byte_idx: j, value: carry_into_j });
    leaves.push(Proof::SubByteSym { byte_idx: j, b: b[j as usize], c: c[j as usize] });
  }
  and_proofs(leaves)
}

// ---- Corresponding wff_*_sym functions ----
//
// All return value-free WFFs (no Byte(concrete) for inputs/outputs).

/// WFF for AND with concrete claimed values.
/// `a[j]` = claimed byte j of input 0, `b[j]` = input 1, `c[j]` = output 0 (= a[j] & b[j]).
/// Values are accepted optimistically and will be verified via stack-log LogUp.
pub fn wff_and_sym(a: &[u8; 32], b: &[u8; 32], c: &[u8; 32]) -> WFF {
  let leaves: Vec<WFF> = (0..32u8)
    .rev()
    .map(|j| WFF::Equal(
      Box::new(Term::ByteAnd(
        Box::new(Term::InputTerm { stack_idx: 0, byte_idx: j, value: a[j as usize] }),
        Box::new(Term::InputTerm { stack_idx: 1, byte_idx: j, value: b[j as usize] }),
      )),
      Box::new(Term::OutputTerm { stack_idx: 0, byte_idx: j, value: c[j as usize] }),
    ))
    .collect();
  and_wffs(leaves)
}

/// WFF for OR with concrete claimed values.
pub fn wff_or_sym(a: &[u8; 32], b: &[u8; 32], c: &[u8; 32]) -> WFF {
  let leaves: Vec<WFF> = (0..32u8)
    .rev()
    .map(|j| WFF::Equal(
      Box::new(Term::ByteOr(
        Box::new(Term::InputTerm { stack_idx: 0, byte_idx: j, value: a[j as usize] }),
        Box::new(Term::InputTerm { stack_idx: 1, byte_idx: j, value: b[j as usize] }),
      )),
      Box::new(Term::OutputTerm { stack_idx: 0, byte_idx: j, value: c[j as usize] }),
    ))
    .collect();
  and_wffs(leaves)
}

/// WFF for XOR with concrete claimed values.
pub fn wff_xor_sym(a: &[u8; 32], b: &[u8; 32], c: &[u8; 32]) -> WFF {
  let leaves: Vec<WFF> = (0..32u8)
    .rev()
    .map(|j| WFF::Equal(
      Box::new(Term::ByteXor(
        Box::new(Term::InputTerm { stack_idx: 0, byte_idx: j, value: a[j as usize] }),
        Box::new(Term::InputTerm { stack_idx: 1, byte_idx: j, value: b[j as usize] }),
      )),
      Box::new(Term::OutputTerm { stack_idx: 0, byte_idx: j, value: c[j as usize] }),
    ))
    .collect();
  and_wffs(leaves)
}

/// WFF for NOT — `ByteXor(InputTerm{0,j,a[j]}, Byte(0xFF)) = OutputTerm{0,j,a[j]^0xFF}`.
/// `a[j]` = claimed input byte; `c[j]` = claimed output byte (should equal `a[j] ^ 0xFF`).
pub fn wff_not_sym(a: &[u8; 32], c: &[u8; 32]) -> WFF {
  let leaves: Vec<WFF> = (0..32u8)
    .rev()
    .map(|j| WFF::Equal(
      Box::new(Term::ByteXor(
        Box::new(Term::InputTerm { stack_idx: 0, byte_idx: j, value: a[j as usize] }),
        Box::new(Term::Byte(0xFF)),
      )),
      Box::new(Term::OutputTerm { stack_idx: 0, byte_idx: j, value: c[j as usize] }),
    ))
    .collect();
  and_wffs(leaves)
}

/// WFF for ADD: `ByteAdd(InputTerm{0,j,a}, InputTerm{1,j,b}, CarryTerm{j}) = OutputTerm{0,j,out}`
/// combined with `CarryTerm{j} = Byte(carry_into_j)` bindings.
///
/// `InputTerm` / `OutputTerm` carry concrete claimed values (verified later via LogUp).
pub fn wff_add_sym(a: &[u8; 32], b: &[u8; 32]) -> WFF {
  let mut carry_out = [0u8; 32];
  for j in (0..32usize).rev() {
    let prev_carry = if j < 31 { carry_out[j + 1] } else { 0u8 };
    let s = a[j] as u16 + b[j] as u16 + prev_carry as u16;
    carry_out[j] = (s >> 8) as u8;
  }
  let mut leaves: Vec<WFF> = Vec::with_capacity(64);
  for j in (0..32u8).rev() {
    let carry_into_j = if j < 31 { carry_out[j as usize + 1] } else { 0u8 };
    let out_j = ((a[j as usize] as u16 + b[j as usize] as u16 + carry_into_j as u16) % 256) as u8;
    // CarryTerm{j} = Byte(carry_into_j)
    leaves.push(WFF::Equal(
      Box::new(Term::CarryTerm { byte_idx: j }),
      Box::new(Term::Byte(carry_into_j)),
    ));
    // ByteAdd(InputTerm{0,j,a[j]}, InputTerm{1,j,b[j]}, CarryTerm{j}) = OutputTerm{0,j,out_j}
    leaves.push(WFF::Equal(
      Box::new(Term::ByteAdd(
        Box::new(Term::InputTerm { stack_idx: 0, byte_idx: j, value: a[j as usize] }),
        Box::new(Term::InputTerm { stack_idx: 1, byte_idx: j, value: b[j as usize] }),
        Box::new(Term::CarryTerm { byte_idx: j }),
      )),
      Box::new(Term::OutputTerm { stack_idx: 0, byte_idx: j, value: out_j }),
    ));
  }
  and_wffs(leaves)
}

/// WFF for SUB with borrow witnesses (same structure as `wff_add_sym`).
pub fn wff_sub_sym(a: &[u8; 32], b: &[u8; 32]) -> WFF {
  // a = b + c  →  ByteAdd(Input{1,j,b[j]}, Output{0,j,c[j]}, CarryTerm{j}) = Input{0,j,a[j]}
  let c = sub_u256_be(a, b);
  let mut carry_out = [0u8; 32];
  for j in (0..32usize).rev() {
    let prev_carry = if j < 31 { carry_out[j + 1] } else { 0u8 };
    let s = b[j] as u16 + c[j] as u16 + prev_carry as u16;
    carry_out[j] = (s >> 8) as u8;
  }
  let mut leaves: Vec<WFF> = Vec::with_capacity(64);
  for j in (0..32u8).rev() {
    let carry_into_j = if j < 31 { carry_out[j as usize + 1] } else { 0u8 };
    leaves.push(WFF::Equal(
      Box::new(Term::CarryTerm { byte_idx: j }),
      Box::new(Term::Byte(carry_into_j)),
    ));
    leaves.push(WFF::Equal(
      Box::new(Term::ByteAdd(
        Box::new(Term::InputTerm { stack_idx: 1, byte_idx: j, value: b[j as usize] }),
        Box::new(Term::OutputTerm { stack_idx: 0, byte_idx: j, value: c[j as usize] }),
        Box::new(Term::CarryTerm { byte_idx: j }),
      )),
      Box::new(Term::InputTerm { stack_idx: 0, byte_idx: j, value: a[j as usize] }),
    ));
  }
  and_wffs(leaves)
}

// ============================================================
// Syntactic (value-free) limb-level prove_* / wff_* for ADD and SUB
// ============================================================
//
// These functions produce a FIXED proof / WFF that is identical for EVERY
// execution of the same opcode.  Concrete word values are not embedded here;
// correctness is enforced at batch-verification time via LogUp binding that
// ties each `InputLimb29` / `OutputLimb29` symbolic term to the actual 29-bit
// chunk of the corresponding stack word.

/// Syntactic proof for ADD — no concrete values.
/// The proof tree is identical for every ADD execution.
///
/// Structure:
/// ```text
/// And(CarryLimbZero, And(U29AddSym{0}, And(U29AddSym{1}, … And(U29AddSym{7}, U24AddSym))))
/// ```
pub fn prove_add_limb_sym() -> Proof {
  let mut proofs = Vec::with_capacity(10);
  proofs.push(Proof::CarryLimbZero);
  for j in 0u8..8 {
    proofs.push(Proof::U29AddSym { limb_idx: j });
  } 
  proofs.push(Proof::U24AddSym);
  and_proofs(proofs)
}

/// Fixed WFF for ADD using symbolic limb terms — identical for every ADD execution.
pub fn wff_add_limb_sym() -> WFF {
  infer_proof(&prove_add_limb_sym()).expect("wff_add_limb_sym: infer_proof failed")
}

/// Syntactic proof for SUB (`a - b = c` verified as `b + c = a`) — no concrete values.
pub fn prove_sub_limb_sym() -> Proof {
  let mut proofs = Vec::with_capacity(10);
  proofs.push(Proof::CarryLimbZero);
  for j in 0u8..8 {
    proofs.push(Proof::U29SubSym { limb_idx: j });
  }
  proofs.push(Proof::U24SubSym);
  and_proofs(proofs)
}

/// Fixed WFF for SUB using symbolic limb terms.
pub fn wff_sub_limb_sym() -> WFF {
  infer_proof(&prove_sub_limb_sym()).expect("wff_sub_limb_sym: infer_proof failed")
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
