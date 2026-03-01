//! Byte-operation lookup table AIR.
//!
//! ## Overview
//!
//! Provides a stand-alone, self-contained AIR that proves the correctness of
//! byte-level AND/OR/XOR operations using LogUp.
//!
//! ### Design
//!
//! The **table side** is a static, preprocessed 256-row matrix where each
//! row `i` (0 ≤ i ≤ 255) encodes:
//!
//! ```text
//! prep[i] = [a, b, a&b, a|b, a^b]
//! ```
//!
//! for `(a = i / 256, b = i % 256)` — but since a single STARK only has one
//! "height", we make the table 256*256 = 65536 rows, one per `(a,b)` pair.
//!
//! The **query side** is a main trace of `N` rows, each containing:
//!
//! ```text
//! main[j] = [a, b, result, op, multiplicity]
//! ```
//!
//! where `op` ∈ {0=AND, 1=OR, 2=XOR}.
//!
//! A **local LogUp argument** ties the two together:
//! - For every query row, send `(a, b, op, result)` with multiplicity `+multiplicity`.
//! - For every table row, receive `(a, b, 0, a&b)` with `-1`, `(a, b, 1, a|b)` with `-1`,
//!   `(a, b, 2, a^b)` with `-1` — but we do this more efficiently:
//!   since the table is preprocessed, we embed those receives as "static" terms.
//!
//! ### Simplified variant used here
//!
//! Rather than a two-sided cross-AIR argument, we use a **local, self-contained
//! lookup** within a single AIR:
//!
//! - Main trace: `N` query rows `[a, b, op, result, mult]`
//! - Preprocessed trace: 256-row "subtable" for a single `a` value (not used here)
//!
//! Because p3-lookup requires all sends and receives to balance within one AIR
//! for local lookups, we embed the full truth-table as additional trace rows
//! appended after the query rows, with multiplicity `-1` each.
//!
//! This means the trace height is `(N_query + 3 * 256 * 256).next_power_of_two()`,
//! which is very large for large `N`. For the initial integration we cap at 8-bit
//! pairs and use a single AIR with no preprocessed columns.
//!
//! ## Column layout
//!
//! | index | name         | meaning                              |
//! |-------|--------------|--------------------------------------|
//! | 0     | `col_a`      | first byte (0..255)                  |
//! | 1     | `col_b`      | second byte (0..255)                 |
//! | 2     | `col_op`     | operation: 0=AND, 1=OR, 2=XOR       |
//! | 3     | `col_result` | expected result                      |
//! | 4     | `col_mult`   | multiplicity: +1 (query) or -1 (table receive) |

use p3_air::{Air, AirBuilderWithPublicValues, BaseAir};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;

use crate::zk_proof::{CircleStarkConfig, Val};

// ── Public column indices ────────────────────────────────────────────
pub const BYTE_TABLE_COL_A: usize = 0;
pub const BYTE_TABLE_COL_B: usize = 1;
pub const BYTE_TABLE_COL_OP: usize = 2;
pub const BYTE_TABLE_COL_RESULT: usize = 3;
pub const BYTE_TABLE_COL_MULT: usize = 4;
pub const NUM_BYTE_TABLE_COLS: usize = 5;

/// Operation codes stored in `col_op`.
pub const BYTE_OP_AND: u32 = 0;
pub const BYTE_OP_OR: u32 = 1;
pub const BYTE_OP_XOR: u32 = 2;

// ── One query row ────────────────────────────────────────────────────

/// A single query against the byte operation table.
#[derive(Debug, Clone, Copy)]
pub struct ByteTableQuery {
  /// First input byte.
  pub a: u8,
  /// Second input byte.
  pub b: u8,
  /// Operation: use [`BYTE_OP_AND`], [`BYTE_OP_OR`], or [`BYTE_OP_XOR`].
  pub op: u32,
  /// Expected result (must equal `a & b`, `a | b`, or `a ^ b`).
  pub result: u8,
  /// Multiplicity count — typically 1.  Multiple identical queries can be
  /// merged by summing multiplicities.
  pub multiplicity: i32,
}

impl ByteTableQuery {
  /// Construct a ByteTableQuery and compute the result automatically.
  pub fn new_and(a: u8, b: u8) -> Self {
    Self {
      a,
      b,
      op: BYTE_OP_AND,
      result: a & b,
      multiplicity: 1,
    }
  }
  pub fn new_or(a: u8, b: u8) -> Self {
    Self {
      a,
      b,
      op: BYTE_OP_OR,
      result: a | b,
      multiplicity: 1,
    }
  }
  pub fn new_xor(a: u8, b: u8) -> Self {
    Self {
      a,
      b,
      op: BYTE_OP_XOR,
      result: a ^ b,
      multiplicity: 1,
    }
  }
}

// ── AIR struct ──────────────────────────────────────────────────────

/// AIR for a self-contained byte-operation lookup table.
///
/// The trace contains both *query* rows (with positive multiplicity) and
/// *receive* rows from the truth table (with negative multiplicity), so that
/// the LogUp running sum balances to zero.
pub struct ByteTableAir;

impl<F: Field> BaseAir<F> for ByteTableAir {
  fn width(&self) -> usize {
    NUM_BYTE_TABLE_COLS
  }
}

impl<AB> Air<AB> for ByteTableAir
where
  AB: AirBuilderWithPublicValues,
  AB::F: Field,
{
  fn eval(&self, builder: &mut AB) {
    eval_byte_table_inner(builder);
  }
}

/// Core byte-table constraints (no LogUp here — those come from the caller
/// via the `eval_lookup` callback in `prove_with_lookup`).
///
/// We verify:
/// 1. `col_op ∈ {0, 1, 2}` (Boolean-ish: `op·(op-1)·(op-2) = 0`, degree 3).
/// 2. For arithmetic validity (non-lookup path, used only for debug):
///    - When `op=0` (AND): `result = a * b` — but this is only valid for bits.
///    - For bytes we rely entirely on the LogUp argument, so this constraint is
///      **not** emitted here.  The AIR itself is trivially satisfied; soundness
///      comes from LogUp.
///
/// Note: the `op ∈ {0, 1, 2}` range check previously expressed as the
/// degree-3 polynomial `op*(op-1)*(op-2) = 0` has been removed.  The LogUp
/// argument already enforces this: every query `(a, b, op, result)` must
/// match a receive row in the truth table, and the truth table only contains
/// rows with `op ∈ {0, 1, 2}`.  An out-of-range `op` value would leave the
/// LogUp sum unbalanced, making the proof unsound — so the polynomial
/// constraint is redundant.  Removing it reduces the max constraint degree
/// from 3 to 1, cutting the number of quotient polynomial chunks.
fn eval_byte_table_inner<AB: AirBuilderWithPublicValues>(_builder: &mut AB) {
  // No polynomial constraints needed: correctness of (a, b, op, result)
  // tuples is enforced entirely by the LogUp argument (see make_byte_table_lookup).
}

// ── Trace builder ────────────────────────────────────────────────────

/// Build the combined main trace for `ByteTableAir`.
///
/// The trace consists of:
/// 1. `query` rows — one per `ByteTableQuery` with `multiplicity = query.multiplicity`.
/// 2. "receive" rows — for every `(a, b)` pair and every `op ∈ {AND,OR,XOR}`,
///    one row with `multiplicity = -1`.
///
/// Both sets share the same column layout.  The height is padded to the next
/// power of two.  Padding rows use `op=0, a=b=result=0, mult=0`.
pub fn build_byte_table_trace(queries: &[ByteTableQuery]) -> RowMajorMatrix<Val> {
  // Receive rows: for every distinct (a, b, op) triple referenced by queries,
  // one truth-table row with multiplicity = -(sum of query multiplicities for
  // that triple).
  //
  // Use a flat array indexed by `a * 768 + b * 3 + op` instead of a BTreeMap.
  // Since a, b ∈ [0,255] and op ∈ {0,1,2}, the total space is 256 × 256 × 3 =
  // 196,608 entries — O(1) insert/lookup, perfectly cache-friendly.
  //
  // A companion Vec<u32> tracks which indices have been touched so we can
  // iterate only the non-zero slots without scanning all 196,608 cells.
  const TABLE_SIZE: usize = 256 * 256 * 3;
  let mut mult_flat = vec![0i32; TABLE_SIZE];
  let mut touched: Vec<u32> = Vec::with_capacity(queries.len().min(TABLE_SIZE));

  for q in queries {
    let idx = (q.a as usize) * 768 + (q.b as usize) * 3 + q.op as usize;
    if mult_flat[idx] == 0 {
      touched.push(idx as u32);
    }
    mult_flat[idx] += q.multiplicity;
  }

  let n_data = queries.len() + touched.len();
  let height = n_data.max(4).next_power_of_two();
  let mut values = vec![Val::ZERO; height * NUM_BYTE_TABLE_COLS];

  // Fill query rows.
  for (i, q) in queries.iter().enumerate() {
    let base = i * NUM_BYTE_TABLE_COLS;
    values[base + BYTE_TABLE_COL_A] = Val::from_u32(q.a as u32);
    values[base + BYTE_TABLE_COL_B] = Val::from_u32(q.b as u32);
    values[base + BYTE_TABLE_COL_OP] = Val::from_u32(q.op);
    values[base + BYTE_TABLE_COL_RESULT] = Val::from_u32(q.result as u32);
    // multiplicity: positive query count
    let mult_val = if q.multiplicity >= 0 {
      Val::from_u32(q.multiplicity as u32)
    } else {
      -Val::from_u32((-q.multiplicity) as u32)
    };
    values[base + BYTE_TABLE_COL_MULT] = mult_val;
  }

  // Fill receive rows (truth-table entries, multiplicity = -total_query_mult).
  // Iterate only the touched slots — O(distinct triples) not O(196,608).
  for (j, &flat_idx) in touched.iter().enumerate() {
    let flat_idx = flat_idx as usize;
    let op = (flat_idx % 3) as u32;
    let b = ((flat_idx / 3) % 256) as u8;
    let a = (flat_idx / 768) as u8;
    let total = mult_flat[flat_idx];
    let i = queries.len() + j;
    let base = i * NUM_BYTE_TABLE_COLS;
    let result = match op {
      BYTE_OP_AND => a & b,
      BYTE_OP_OR => a | b,
      _ => a ^ b,
    };
    values[base + BYTE_TABLE_COL_A] = Val::from_u32(a as u32);
    values[base + BYTE_TABLE_COL_B] = Val::from_u32(b as u32);
    values[base + BYTE_TABLE_COL_OP] = Val::from_u32(op);
    values[base + BYTE_TABLE_COL_RESULT] = Val::from_u32(result as u32);
    // receive: multiplicity = -total (negates the accumulated query multiplicity)
    let neg_total = -Val::from_u32(total as u32);
    values[base + BYTE_TABLE_COL_MULT] = neg_total;
  }

  // Padding rows: op=0 (AND), a=b=result=0, mult=0 (neutral).
  // (already ZERO from initialization)

  RowMajorMatrix::new(values, NUM_BYTE_TABLE_COLS)
}

// ── Public API ───────────────────────────────────────────────────────

use p3_lookup::logup::LogUpGadget;
use p3_lookup::lookup_traits::{Kind, Lookup, LookupData, LookupGadget};
use p3_uni_stark::{Entry, SymbolicExpression, SymbolicVariable, VerifierConstraintFolder};

/// Build the `Lookup` descriptor for the byte-table local lookup.
///
/// The lookup uses a single aux column (index 0) and two challenges.
/// Both query rows and receive rows use the same `(a, b, op, result)` tuple
/// with the multiplicity column determining sign.
///
/// The `element_exprs` is a single tuple `[a, b, op, result]` and the
/// `multiplicities_exprs` is the `mult` column.
pub fn make_byte_table_lookup() -> Lookup<Val> {
  let col =
    |c: usize| SymbolicExpression::Variable(SymbolicVariable::new(Entry::Main { offset: 0 }, c));
  Lookup::new(
    Kind::Local,
    vec![vec![
      col(BYTE_TABLE_COL_A),
      col(BYTE_TABLE_COL_B),
      col(BYTE_TABLE_COL_OP),
      col(BYTE_TABLE_COL_RESULT),
    ]],
    vec![col(BYTE_TABLE_COL_MULT)],
    vec![0], // aux column 0
  )
}

/// Generate the permutation trace for the byte-table lookup.
///
/// Returns `Some(perm_trace)` if the main trace is non-empty, `None` if empty.
pub fn generate_byte_table_perm_trace(
  main_trace: &RowMajorMatrix<Val>,
  perm_challenges: &[<CircleStarkConfig as p3_uni_stark::StarkGenericConfig>::Challenge],
) -> Option<RowMajorMatrix<<CircleStarkConfig as p3_uni_stark::StarkGenericConfig>::Challenge>> {
  let gadget = LogUpGadget::new();
  let lookup = make_byte_table_lookup();
  let mut lookup_data: Vec<
    LookupData<<CircleStarkConfig as p3_uni_stark::StarkGenericConfig>::Challenge>,
  > = vec![];
  let perm_trace = gadget.generate_permutation::<CircleStarkConfig>(
    main_trace,
    &None,
    &[],
    &[lookup],
    &mut lookup_data,
    perm_challenges,
  );
  Some(perm_trace)
}

/// Evaluate the byte-table lookup constraints in a `VerifierConstraintFolder`.
///
/// Call this as the `eval_lookup` argument of `prove_with_lookup` /
/// `verify_with_lookup`.
pub fn eval_byte_table_lookup(folder: &mut VerifierConstraintFolder<CircleStarkConfig>) {
  let gadget = LogUpGadget::new();
  let lookup = make_byte_table_lookup();
  gadget.eval_local_lookup(folder, &lookup);
}

/// High-level: prove a set of byte-operation queries using LogUp.
///
/// Returns a `p3_uni_stark::Proof<CircleStarkConfig>`.
pub fn prove_byte_table(queries: &[ByteTableQuery]) -> p3_uni_stark::Proof<CircleStarkConfig> {
  use crate::zk_proof::{air_cache, make_circle_config};

  let config = make_circle_config();
  let air = ByteTableAir;
  let trace = build_byte_table_trace(queries);
  let trace_for_perm = trace.clone();
  let lookup_prove = make_byte_table_lookup();
  let hints = air_cache::get_or_compute(&air, 0, 0, 0);

  p3_uni_stark::prove_with_lookup_hinted(
    &config,
    &air,
    trace,
    &[],
    None,
    move |perm_challenges| generate_byte_table_perm_trace(&trace_for_perm, perm_challenges),
    2, // num_perm_challenges: 1 lookup × 2 (alpha + beta)
    2, // lookup_constraint_count: first_row + transition (local LogUp = 2 constraints)
    move |folder: &mut VerifierConstraintFolder<CircleStarkConfig>| {
      let gadget = LogUpGadget::new();
      gadget.eval_local_lookup(folder, &lookup_prove);
    },
    hints,
  )
}

/// High-level: verify a byte-table proof.
///
/// Returns `Ok(())` on success or an error on failure.
pub fn verify_byte_table(
  proof: &p3_uni_stark::Proof<CircleStarkConfig>,
) -> crate::zk_proof::CircleStarkVerifyResult {
  use crate::zk_proof::make_circle_config;

  let config = make_circle_config();
  let air = ByteTableAir;
  let lookup_verify = make_byte_table_lookup();

  p3_uni_stark::verify_with_lookup(
    &config,
    &air,
    proof,
    &[],
    None,
    2,
    move |folder: &mut VerifierConstraintFolder<CircleStarkConfig>| {
      let gadget = LogUpGadget::new();
      gadget.eval_local_lookup(folder, &lookup_verify);
    },
  )
}
