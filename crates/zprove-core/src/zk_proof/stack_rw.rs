//! Stack read/write chronological consistency AIR with LogUp.
//!
//! Every EVM stack access — push (write) and pop (read) — is recorded as a
//! [`StackRwEntry`].  The witness trace is sorted by `(access_depth,
//! rw_counter)`.  Transition constraints then enforce that each read sees the
//! most-recent prior write at the same depth.  A LogUp multiset argument ties
//! the `(depth, value)` tuples back to the execution trace.
//!
//! # Column layout  (NUM_STACK_RW_COLS = 15)
//!
//! ```text
//!  0  DEPTH           u32 access-depth (= depth_after for push, depth_after+1 for pop)
//!  1  RW_HI           rw_counter >> 32
//!  2  RW_LO           rw_counter & 0xFFFF_FFFF
//!  3  VAL0            ┐
//!  4  VAL1            │
//!  5  VAL2            │  32-byte value as 8 × u32 (big-endian)
//!  6  VAL3            │
//!  7  VAL4            │
//!  8  VAL5            │
//!  9  VAL6            │
//! 10  VAL7            ┘
//! 11  IS_WRITE        boolean — 1 = push (write), 0 = pop (read)
//! 12  IS_SAME_DEPTH   boolean — 1 if this row's depth == next row's depth
//! 13  MULT            LogUp multiplicity (M31-encoded signed integer)
//! 14  IS_READ_CONT    = IS_SAME_DEPTH × (1 − next.IS_WRITE); helper column that
//!                     caps the AIR constraint degree at 2 (without it, read-continuity
//!                     constraints would be degree 3, requiring 2 quotient chunks).
//! ```
//!
//! # AIR constraints  (13 main + 2 LogUp = 15 total, max degree = 2)
//!
//! | # | Expression | Kind |
//! |---|---|---|
//! | 1 | `pv[0] = TAG_STACK_RW` | public-values |
//! | 2 | `is_write · (1 − is_write) = 0` | boolean |
//! | 3 | `is_same_depth · (1 − is_same_depth) = 0` | boolean |
//! | 4 | `is_same_depth · (next.depth − this.depth) = 0` | transition |
//! | 5 | `is_read_cont − is_same_depth · (1 − next.is_write) = 0` | helper definition |
//! | 6‥13 | `is_read_cont · (next.val_k − this.val_k) = 0` ∀k | read-continuity |
//!
//! Constraints 4–12 are written *without* `when_transition()` so they do not
//! introduce an `IsTransition` factor (which has `degree_multiple = 0` in the
//! Circle STARK and causes `OodEvaluationMismatch` for high-degree products).
//! Correctness at the wrap-around row is guaranteed by `IS_SAME_DEPTH = 0`
//! on the last data row and on every padding row.

use super::memory::value_to_u32s;
use super::types::{
  Challenge, CircleStarkConfig, CircleStarkProof, Val, default_poseidon_sponge, make_circle_config,
};

use p3_air::{Air, AirBuilderWithPublicValues, BaseAir};
use p3_field::PrimeCharacteristicRing;
use p3_lookup::logup::LogUpGadget;
use p3_lookup::lookup_traits::{Kind, Lookup, LookupData, LookupGadget};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_symmetric::CryptographicHasher;
use p3_uni_stark::{
  Entry, SymbolicExpression, SymbolicVariable, VerifierConstraintFolder,
  prove_with_lookup_hinted, verify_with_lookup,
};

use super::air_cache;

// ── Tag ───────────────────────────────────────────────────────────────────────

pub const RECEIPT_BIND_TAG_STACK_RW: u32 = 8;

// ── Column indices ────────────────────────────────────────────────────────────

const COL_DEPTH: usize = 0;
const COL_RW_HI: usize = 1;
const COL_RW_LO: usize = 2;
const COL_VAL0: usize = 3; // VAL0..VAL7 occupy columns 3–10
const COL_IS_WRITE: usize = 11;
const COL_IS_SAME_DEPTH: usize = 12;
const COL_MULT: usize = 13;
/// Helper witness: `IS_SAME_DEPTH × (1 − next.IS_WRITE)`.  Pre-computing this
/// product as a column lets read-continuity constraints stay at degree 2 instead
/// of degree 3, halving the number of quotient polynomial chunks.
const COL_IS_READ_CONT: usize = 14;

pub const NUM_STACK_RW_COLS: usize = 15;

/// Length of the public-values vector: tag + n_writes + hash_write[8] + n_reads + hash_read[8].
const STACK_RW_PUBLIC_VALUES_LEN: usize = 19;

// ── Log entry ─────────────────────────────────────────────────────────────────

/// One push (write) or pop (read) stack access, as recorded by the EVM executor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StackRwEntry {
  /// Monotone global push/pop counter across all instructions in the batch.
  pub rw_counter: u64,
  /// Stack depth *after* the operation (0 = empty stack after a pop that
  /// left nothing below).
  pub depth_after: u32,
  /// `true` = push (write to depth `depth_after`),
  /// `false` = pop (read from depth `depth_after + 1`).
  pub is_write: bool,
  /// 32-byte value pushed to or popped from the stack.
  pub value: [u8; 32],
}

impl StackRwEntry {
  /// The depth slot that is *accessed* by this operation:
  /// - push: writes the new top at `depth_after`
  /// - pop:  reads the current top at `depth_after + 1`
  #[inline]
  pub fn access_depth(&self) -> u32 {
    if self.is_write {
      self.depth_after
    } else {
      self.depth_after + 1
    }
  }
}

/// Convert a slice of [`StackAccessClaim`]s (from the executor) into
/// [`StackRwEntry`]s suitable for this module.
pub fn stack_rw_entries_from_claims(
  claims: &[crate::transition::StackAccessClaim],
) -> Vec<StackRwEntry> {
  claims
    .iter()
    .map(|c| StackRwEntry {
      rw_counter: c.rw_counter,
      depth_after: c.depth_after as u32,
      is_write: c.is_push,
      value: c.value,
    })
    .collect()
}

// ── Poseidon hash helpers ─────────────────────────────────────────────────────

/// Hash a log of [`StackRwEntry`]s to 8 M31 field elements.
fn hash_rw_log(log: &[StackRwEntry]) -> [Val; 8] {
  let sponge = default_poseidon_sponge();
  // capacity: 1 (len) + len * (2 rw + 1 depth + 8 value) fields
  let mut input: Vec<Val> = Vec::with_capacity(1 + log.len() * 11);
  input.push(Val::from_u32(log.len() as u32));
  for e in log {
    input.push(Val::from_u32((e.rw_counter >> 32) as u32));
    input.push(Val::from_u32(e.rw_counter as u32));
    input.push(Val::from_u32(e.access_depth()));
    for chunk in e.value.chunks(4) {
      input.push(Val::from_u32(u32::from_be_bytes([
        chunk[0], chunk[1], chunk[2], chunk[3],
      ])));
    }
  }
  sponge.hash_iter(input)
}

fn make_stack_rw_public_values(
  write_log: &[StackRwEntry],
  read_log: &[StackRwEntry],
) -> Vec<Val> {
  let wh = hash_rw_log(write_log);
  let rh = hash_rw_log(read_log);
  let mut pv = Vec::with_capacity(STACK_RW_PUBLIC_VALUES_LEN);
  pv.push(Val::from_u32(RECEIPT_BIND_TAG_STACK_RW));
  pv.push(Val::from_u32(write_log.len() as u32));
  pv.extend_from_slice(&wh);
  pv.push(Val::from_u32(read_log.len() as u32));
  pv.extend_from_slice(&rh);
  pv
}

// ── Consistency checker ───────────────────────────────────────────────────────

/// Validate that every intra-batch read sees the most-recent prior write at
/// the same depth.  Cross-batch reads (no prior write in this batch) are
/// allowed and left for the outer LogUp join.
fn check_stack_rw_consistency(entries: &[StackRwEntry]) -> Result<(), String> {
  let mut sorted = entries.to_vec();
  sorted.sort_by_key(|e| e.rw_counter);

  let mut last_write: std::collections::HashMap<u32, [u8; 32]> = Default::default();
  for e in &sorted {
    let d = e.access_depth();
    if e.is_write {
      last_write.insert(d, e.value);
    } else if let Some(&w_val) = last_write.get(&d) {
      if e.value != w_val {
        return Err(format!(
          "stack_rw: read at depth={} rw={} value mismatch \
           (got {:?}, last write was {:?})",
          d, e.rw_counter, e.value, w_val
        ));
      }
    }
    // No prior write → cross-batch read, tolerated at this layer.
  }
  Ok(())
}

// ── Trace builder ─────────────────────────────────────────────────────────────

/// Compute LogUp multiplicities for the sorted trace.
///
/// Write multiplicity = number of consecutive same-depth reads that follow
/// before the next write (or a depth change).  Read multiplicity = −1 if the
/// depth was written **before the current read** in this batch, 0 otherwise (cross-batch).
///
/// Bug fix: reads that precede their depth's first write in the sorted (depth, rw_counter)
/// order are cross-batch reads and must get MULT=0.  The old pre-computed `depths_written`
/// set incorrectly flagged them as intra-batch (MULT=-1), breaking the LogUp sum.
fn compute_multiplicities(sorted_rows: &[&StackRwEntry]) -> Vec<i32> {
  let n = sorted_rows.len();
  let mut mults = vec![0i32; n];
  // sorted_rows is ordered by (depth ASC, rw_counter ASC), so all entries at
  // the same depth are consecutive.  A simple per-group flag replaces the
  // HashSet used previously — no hashing, pure cache-friendly linear scan.
  let mut current_depth = u32::MAX;
  let mut depth_had_write = false;
  for i in 0..n {
    let e = sorted_rows[i];
    let d = e.access_depth();
    if d != current_depth {
      current_depth = d;
      depth_had_write = false;
    }
    if e.is_write {
      depth_had_write = true;
      // Count immediately-following same-depth reads before any next write.
      let mut count = 0i32;
      let mut j = i + 1;
      while j < n && sorted_rows[j].access_depth() == d && !sorted_rows[j].is_write {
        count += 1;
        j += 1;
      }
      mults[i] = count;
    } else {
      // Read: −1 if a write for this depth group has already appeared (intra-batch);
      //        0 if this is the first access for this depth (cross-batch read).
      mults[i] = if depth_had_write { -1 } else { 0 };
    }
  }
  mults
}

fn build_stack_rw_trace(
  all_entries: &[StackRwEntry],
) -> RowMajorMatrix<Val> {
  // Sort by (access_depth ASC, rw_counter ASC) so that within each depth,
  // entries are in chronological order.
  let mut rows: Vec<&StackRwEntry> = all_entries.iter().collect();
  rows.sort_by_key(|e| (e.access_depth(), e.rw_counter));

  let n = rows.len();
  let height = n.max(4).next_power_of_two();
  let mults = compute_multiplicities(&rows);

  let mut data = vec![Val::ZERO; height * NUM_STACK_RW_COLS];

  for (i, (e, &mult)) in rows.iter().zip(mults.iter()).enumerate() {
    let base = i * NUM_STACK_RW_COLS;
    let d = e.access_depth();
    let vals = value_to_u32s(&e.value);

    // IS_SAME_DEPTH = 1 iff the next data row has the same depth.
    // The last data row and all padding rows get IS_SAME_DEPTH = 0.
    let is_same_depth: u32 = if i + 1 < n && rows[i + 1].access_depth() == d {
      1
    } else {
      0
    };

    // IS_READ_CONT = IS_SAME_DEPTH × (1 − IS_WRITE_next): 1 iff the next row is a
    // same-depth read that must inherit this row's value.
    let is_read_cont: u32 =
      if is_same_depth == 1 && i + 1 < n && !rows[i + 1].is_write { 1 } else { 0 };

    data[base + COL_DEPTH] = Val::from_u32(d);
    data[base + COL_RW_HI] = Val::from_u32((e.rw_counter >> 32) as u32);
    data[base + COL_RW_LO] = Val::from_u32(e.rw_counter as u32);
    for k in 0..8 {
      data[base + COL_VAL0 + k] = Val::from_u32(vals[k]);
    }
    data[base + COL_IS_WRITE] = Val::from_u32(e.is_write as u32);
    data[base + COL_IS_SAME_DEPTH] = Val::from_u32(is_same_depth);
    data[base + COL_MULT] = if mult >= 0 {
      Val::from_u32(mult as u32)
    } else {
      -Val::from_u32((-mult) as u32)
    };
    data[base + COL_IS_READ_CONT] = Val::from_u32(is_read_cont);
  }
  // Padding rows are all-zero: IS_WRITE=0, IS_SAME_DEPTH=0, IS_READ_CONT=0, MULT=0 —
  // all constraints evaluate to 0 on padding.

  RowMajorMatrix::new(data, NUM_STACK_RW_COLS)
}

// ── AIR ───────────────────────────────────────────────────────────────────────

pub struct StackRwAir;

impl<F: p3_field::Field> BaseAir<F> for StackRwAir {
  fn width(&self) -> usize {
    NUM_STACK_RW_COLS
  }
}

impl<AB: AirBuilderWithPublicValues> Air<AB> for StackRwAir
where
  AB::F: p3_field::Field,
{
  fn eval(&self, builder: &mut AB) {
    // ── Constraint 1: receipt-binding tag ─────────────────────────────────
    let pis = builder.public_values();
    builder.assert_eq(
      pis[0].into(),
      AB::Expr::from_u32(RECEIPT_BIND_TAG_STACK_RW),
    );

    let main = builder.main();
    let local = main.row_slice(0).expect("empty stack-rw trace");
    let local = &*local;
    let next = main.row_slice(1).expect("stack-rw trace too short");
    let next = &*next;

    let is_write = local[COL_IS_WRITE].clone();
    let is_same_depth = local[COL_IS_SAME_DEPTH].clone();

    // ── Constraint 2: is_write is boolean ─────────────────────────────────
    builder.assert_zero(
      is_write.clone().into() * (AB::Expr::ONE - is_write.clone().into()),
    );

    // ── Constraint 3: is_same_depth is boolean ────────────────────────────
    builder.assert_zero(
      is_same_depth.clone().into() * (AB::Expr::ONE - is_same_depth.clone().into()),
    );

    // ── Constraint 4: if is_same_depth=1 → depths must match ──────────────
    //
    // Written without `when_transition()` to avoid the IsTransition
    // degree_multiple=0 bug in Circle STARK.  The IS_SAME_DEPTH selector is
    // 0 on the last data row and every padding row, so this evaluates to 0 at
    // the wrap-around point (last row → first row on the circle).
    builder.assert_zero(
      is_same_depth.clone().into()
        * (next[COL_DEPTH].clone().into() - local[COL_DEPTH].clone().into()),
    );

    // ── Constraint 5: IS_READ_CONT definition (degree 2) ────────────────────
    //
    // Enforce that the helper column equals IS_SAME_DEPTH × (1 − IS_WRITE_next).
    // Pre-computing this product as a witness column lets the read-continuity
    // constraints below stay at degree 2 rather than degree 3, which halves the
    // number of FRI quotient chunks from 2 to 1.
    let is_read_cont = local[COL_IS_READ_CONT].clone();
    builder.assert_zero(
      is_read_cont.clone().into()
        - is_same_depth.clone().into()
          * (AB::Expr::ONE - next[COL_IS_WRITE].clone().into()),
    );

    // ── Constraints 6–13: read-after-write continuity (degree 2) ─────────────
    //
    // If IS_READ_CONT=1 (same depth AND next is a read) → values must match.
    // The IS_SAME_DEPTH=0 invariant on the last data/padding rows ensures
    // IS_READ_CONT=0 there, so these constraints evaluate to 0 at wrap-around.
    for k in 0..8 {
      builder.assert_zero(
        is_read_cont.clone().into()
          * (next[COL_VAL0 + k].clone().into() - local[COL_VAL0 + k].clone().into()),
      );
    }
  }
}

// ── LogUp argument ────────────────────────────────────────────────────────────

/// Build the Lookup descriptor: tuple = (depth, val0..val7), multiplicity = MULT.
fn make_stack_rw_lookup() -> Lookup<Val> {
  let col =
    |c: usize| SymbolicExpression::Variable(SymbolicVariable::new(Entry::Main { offset: 0 }, c));
  Lookup::new(
    Kind::Local,
    vec![vec![
      col(COL_DEPTH),
      col(COL_VAL0),
      col(COL_VAL0 + 1),
      col(COL_VAL0 + 2),
      col(COL_VAL0 + 3),
      col(COL_VAL0 + 4),
      col(COL_VAL0 + 5),
      col(COL_VAL0 + 6),
      col(COL_VAL0 + 7),
    ]],
    vec![col(COL_MULT)],
    vec![0],
  )
}

fn generate_stack_rw_perm_trace(
  main_trace: &RowMajorMatrix<Val>,
  perm_challenges: &[Challenge],
) -> Option<RowMajorMatrix<Challenge>> {
  let gadget = LogUpGadget::new();
  let lookup = make_stack_rw_lookup();
  let mut lookup_data: Vec<LookupData<Challenge>> = vec![];
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

fn eval_stack_rw_lookup(folder: &mut VerifierConstraintFolder<CircleStarkConfig>) {
  LogUpGadget::new().eval_local_lookup(folder, &make_stack_rw_lookup());
}

// ── Public API ────────────────────────────────────────────────────────────────

/// A complete stack read/write consistency proof.
#[derive(Clone)]
pub struct StackRwProof {
  pub stark_proof: CircleStarkProof,
  /// All push (write) entries, in rw_counter order.
  pub write_log: Vec<StackRwEntry>,
  /// All pop (read) entries, in rw_counter order.
  pub read_log: Vec<StackRwEntry>,
}

/// Generate a [`StackRwProof`] from a slice of [`StackAccessClaim`]s.
///
/// Returns `Err` if the claims are chronologically inconsistent (a read at
/// depth *d* sees a value different from the last write to depth *d*).
pub fn prove_stack_rw(
  claims: &[crate::transition::StackAccessClaim],
) -> Result<StackRwProof, String> {
  let entries = stack_rw_entries_from_claims(claims);
  check_stack_rw_consistency(&entries)?;

  let mut write_log: Vec<StackRwEntry> = Vec::new();
  let mut read_log: Vec<StackRwEntry> = Vec::new();
  for e in &entries {
    if e.is_write {
      write_log.push(e.clone());
    } else {
      read_log.push(e.clone());
    }
  }

  let public_values = make_stack_rw_public_values(&write_log, &read_log);

  let trace = if entries.is_empty() {
    RowMajorMatrix::new(vec![Val::ZERO; 4 * NUM_STACK_RW_COLS], NUM_STACK_RW_COLS)
  } else {
    build_stack_rw_trace(&entries)
  };

  let config = make_circle_config();
  let hints = air_cache::get_or_compute(&StackRwAir, 0, public_values.len(), 0);
  let trace_for_perm = trace.clone();
  let stark_proof = prove_with_lookup_hinted(
    &config,
    &StackRwAir,
    trace,
    &public_values,
    None,
    move |perm_challenges| generate_stack_rw_perm_trace(&trace_for_perm, perm_challenges),
    2,
    2,
    |folder: &mut VerifierConstraintFolder<CircleStarkConfig>| eval_stack_rw_lookup(folder),
    hints,
  );

  Ok(StackRwProof {
    stark_proof,
    write_log,
    read_log,
  })
}

/// Verify a [`StackRwProof`].
///
/// Returns `true` iff the STARK proof is valid *and* the committed logs are
/// themselves mutually consistent.
pub fn verify_stack_rw(proof: &StackRwProof) -> bool {
  // Reconstruct all entries and re-check consistency.
  let all_entries: Vec<StackRwEntry> = proof
    .write_log
    .iter()
    .chain(proof.read_log.iter())
    .cloned()
    .collect();
  if check_stack_rw_consistency(&all_entries).is_err() {
    return false;
  }

  let public_values = make_stack_rw_public_values(&proof.write_log, &proof.read_log);
  let config = make_circle_config();
  verify_with_lookup(
    &config,
    &StackRwAir,
    &proof.stark_proof,
    &public_values,
    None,
    2,
    |folder: &mut VerifierConstraintFolder<CircleStarkConfig>| eval_stack_rw_lookup(folder),
  )
  .is_ok()
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;
  use crate::transition::StackAccessClaim;

  fn claim(rw: u64, depth_after: usize, is_push: bool, byte_val: u8) -> StackAccessClaim {
    let mut value = [0u8; 32];
    value[31] = byte_val;
    StackAccessClaim { rw_counter: rw, depth_after, is_push, value }
  }

  #[test]
  fn test_empty_log() {
    let proof = prove_stack_rw(&[]).expect("prove failed");
    assert!(verify_stack_rw(&proof), "verify failed");
  }

  #[test]
  fn test_single_push() {
    let claims = vec![claim(1, 1, true, 42)];
    let proof = prove_stack_rw(&claims).expect("prove failed");
    assert!(verify_stack_rw(&proof), "verify failed");
  }

  #[test]
  fn test_push_then_pop() {
    // push 42 onto depth 1, then pop 42 from depth 1
    let claims = vec![
      claim(1, 1, true, 42),  // push: depth_after=1, access_depth=1
      claim(2, 0, false, 42), // pop:  depth_after=0, access_depth=1
    ];
    let proof = prove_stack_rw(&claims).expect("prove failed");
    assert!(verify_stack_rw(&proof), "verify failed");
  }

  #[test]
  fn test_two_depths_interleaved() {
    // push to depth 1, push to depth 2, pop from 2, pop from 1
    let claims = vec![
      claim(1, 1, true, 0xAA),  // push 0xAA, depth_after=1, access_depth=1
      claim(2, 2, true, 0xBB),  // push 0xBB, depth_after=2, access_depth=2
      claim(3, 1, false, 0xBB), // pop  0xBB, depth_after=1, access_depth=2
      claim(4, 0, false, 0xAA), // pop  0xAA, depth_after=0, access_depth=1
    ];
    let proof = prove_stack_rw(&claims).expect("prove failed");
    assert!(verify_stack_rw(&proof), "verify failed");
  }

  #[test]
  fn test_multiple_reads_same_depth() {
    // one write, two reads at the same depth
    let claims = vec![
      claim(1, 1, true, 0x99),   // push 0x99, access_depth=1
      claim(2, 0, false, 0x99),  // pop, access_depth=1
      claim(3, 1, true, 0x77),   // push 0x77 again, access_depth=1
      claim(4, 0, false, 0x77),  // pop 0x77, access_depth=1
    ];
    let proof = prove_stack_rw(&claims).expect("prove failed");
    assert!(verify_stack_rw(&proof), "verify failed");
  }

  #[test]
  fn test_consistency_check_catches_mismatch() {
    // push 0xAA but pop 0xBB → should fail
    let claims = vec![
      claim(1, 1, true, 0xAA),
      claim(2, 0, false, 0xBB),
    ];
    assert!(prove_stack_rw(&claims).is_err(), "should have failed");
  }
}
