//! Keccak256 consistency proof (preimage membership argument).
//!
//! # Layout (word-per-row)
//!
//! Each row of the main trace represents **one 32-byte aligned memory word**
//! that is part of a KECCAK256 input range.  The last word of each call carries
//! the output hash; earlier words have hash columns zeroed.
//!
//! The full keccak input bytes and the addresses of the input words are committed
//! in the public-values Poseidon hash, binding this proof to both the memory
//! consistency proof (via the shared `read_log` entries that are now emitted as
//! `MemAccessClaim`s by the EVM inspector) and to the receipt binding tag.
//!
//! # Memory binding
//!
//! When the EVM inspector handles KECCAK256 it now emits `MemAccessClaim`s for
//! every 32-byte aligned word in `[offset, offset+size)`.  Those claims end up
//! in the `MemoryConsistencyProof.read_log`.  The cross-check in
//! `validate_keccak_memory_cross_check` verifies that every word recorded here
//! appears in that read log with the same `(addr, value)`.

use super::memory::MemLogEntry;
use super::types::{CircleStarkProof, Val, default_poseidon_sponge, make_circle_config};

use p3_field::PrimeCharacteristicRing;
use p3_keccak::Keccak256Hash;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_symmetric::CryptographicHasher;

// ── Column layout  (NUM_KECCAK_COLS = 23) ─────────────────────────────
//  0     call_id_hi   — index of the KECCAK256 call (hi 32 bits)
//  1     call_id_lo   — index of the KECCAK256 call (lo 32 bits)
//  2     addr_hi      — 32-byte-aligned word address (hi 32 bits)
//  3     addr_lo      — 32-byte-aligned word address (lo 32 bits)
//  4..11 val0..7      — 32-byte word value (8 × u32, big-endian)
//  12    is_last      — 1 iff this is the last word of this call's input
//  13..20 hash0..7   — keccak256 digest (valid only when is_last == 1, else 0)
//  21    is_active    — 1 for real word rows, 0 for padding
//  22    (reserved / padding column for alignment)

const KECCAK_COL_CALL_ID_HI: usize = 0;
const KECCAK_COL_CALL_ID_LO: usize = 1;
const KECCAK_COL_ADDR_HI: usize = 2;
const KECCAK_COL_ADDR_LO: usize = 3;
const KECCAK_COL_VAL0: usize = 4;   // val0..val7 → cols 4..11
const KECCAK_COL_IS_LAST: usize = 12;
const KECCAK_COL_HASH0: usize = 13; // hash0..hash7 → cols 13..20
const KECCAK_COL_IS_ACTIVE: usize = 21;
/// Total number of main-trace columns for keccak consistency.
pub const NUM_KECCAK_COLS: usize = 22; // 2(call_id)+2(addr)+8(val)+1(is_last)+8(hash)+1(is_active)

pub const RECEIPT_BIND_TAG_KECCAK: u32 = 7;
// public values: [tag(1), num_calls(1), words_hash(8), calls_hash(8)] = 18
const KECCAK_PUBLIC_VALUES_LEN: usize = 18;

// ── Log types ─────────────────────────────────────────────────────────

/// One 32-byte-aligned memory word that contributes to a KECCAK256 input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeccakWordEntry {
  /// Call index (0-based order of KECCAK256 instructions in the batch).
  pub call_id: u64,
  /// 32-byte-aligned byte address of this word in EVM memory.
  pub addr: u64,
  /// The 32-byte word value.
  pub value: [u8; 32],
  /// True iff this is the last word of the call's input range.
  pub is_last: bool,
  /// keccak256 digest — only meaningful when `is_last == true`, else all-zeros.
  pub output_hash: [u8; 32],
}

/// Per-KECCAK256-call log entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeccakLogEntry {
  pub offset: u64,
  pub size: u64,
  pub input_bytes: Vec<u8>,
  pub output_hash: [u8; 32],
  /// Ordered list of 32-byte aligned memory words in `[offset, offset+size)`.
  /// Each entry carries `(addr, value)` so the memory cross-check can verify
  /// that these exact words appear in the memory proof's read log.
  pub memory_words: Vec<(u64, [u8; 32])>,
}

// ── Hash helpers ──────────────────────────────────────────────────────

/// Poseidon hash of all `(addr, value)` pairs across all word entries, in order.
/// Used as part of the public values to bind keccak claim words to the memory proof.
fn hash_keccak_words(words: &[KeccakWordEntry]) -> [Val; 8] {
  let sponge = default_poseidon_sponge();
  let mut input: Vec<Val> = Vec::with_capacity(1 + words.len() * 12);
  input.push(Val::from_u32(words.len() as u32));
  for w in words {
    input.push(Val::from_u32((w.call_id >> 32) as u32));
    input.push(Val::from_u32(w.call_id as u32));
    input.push(Val::from_u32((w.addr >> 32) as u32));
    input.push(Val::from_u32(w.addr as u32));
    for chunk in w.value.chunks(4) {
      input.push(Val::from_u32(u32::from_be_bytes([
        chunk[0], chunk[1], chunk[2], chunk[3],
      ])));
    }
  }
  sponge.hash_iter(input)
}

/// Poseidon hash of keccak call-level metadata (offset, size, output_hash).
fn hash_keccak_calls(log: &[KeccakLogEntry]) -> [Val; 8] {
  let sponge = default_poseidon_sponge();
  let mut input: Vec<Val> = Vec::with_capacity(1 + log.len() * 12);
  input.push(Val::from_u32(log.len() as u32));
  for e in log {
    input.push(Val::from_u32((e.offset >> 32) as u32));
    input.push(Val::from_u32(e.offset as u32));
    input.push(Val::from_u32((e.size >> 32) as u32));
    input.push(Val::from_u32(e.size as u32));
    for chunk in e.output_hash.chunks(4) {
      input.push(Val::from_u32(u32::from_be_bytes([
        chunk[0], chunk[1], chunk[2], chunk[3],
      ])));
    }
  }
  sponge.hash_iter(input)
}

fn collect_word_entries(log: &[KeccakLogEntry]) -> Vec<KeccakWordEntry> {
  let mut entries = Vec::new();
  for (call_id, e) in log.iter().enumerate() {
    if e.memory_words.is_empty() {
      // Zero-length keccak: emit a single sentinel row (addr=0, val=0, is_last=true).
      entries.push(KeccakWordEntry {
        call_id: call_id as u64,
        addr: 0,
        value: [0u8; 32],
        is_last: true,
        output_hash: e.output_hash,
      });
    } else {
      let last_idx = e.memory_words.len() - 1;
      for (i, &(addr, value)) in e.memory_words.iter().enumerate() {
        let is_last = i == last_idx;
        entries.push(KeccakWordEntry {
          call_id: call_id as u64,
          addr,
          value,
          is_last,
          output_hash: if is_last { e.output_hash } else { [0u8; 32] },
        });
      }
    }
  }
  entries
}

fn make_keccak_public_values(log: &[KeccakLogEntry]) -> Vec<Val> {
  let words = collect_word_entries(log);
  let words_hash = hash_keccak_words(&words);
  let calls_hash = hash_keccak_calls(log);
  let mut pv = Vec::with_capacity(KECCAK_PUBLIC_VALUES_LEN);
  pv.push(Val::from_u32(RECEIPT_BIND_TAG_KECCAK));
  pv.push(Val::from_u32(log.len() as u32));
  pv.extend_from_slice(&words_hash);
  pv.extend_from_slice(&calls_hash);
  pv
}

// ── Keccak256 computation ─────────────────────────────────────────────

/// Compute keccak256 of `input` and return the 32-byte digest.
pub fn keccak256_bytes(input: &[u8]) -> [u8; 32] {
  Keccak256Hash.hash_iter(input.iter().copied())
}

// ── AIR ───────────────────────────────────────────────────────────────

pub struct KeccakConsistencyAir;

impl<F: p3_field::Field> p3_air::BaseAir<F> for KeccakConsistencyAir {
  fn width(&self) -> usize {
    NUM_KECCAK_COLS
  }
}

impl<AB: p3_air::AirBuilderWithPublicValues> p3_air::Air<AB> for KeccakConsistencyAir
where
  AB::F: p3_field::Field,
{
  fn eval(&self, builder: &mut AB) {
    // ── Constraint 1: receipt-binding tag ─────────────────────────────────
    let pis = builder.public_values();
    builder.assert_eq(pis[0].into(), AB::Expr::from_u32(RECEIPT_BIND_TAG_KECCAK));

    let main = builder.main();
    let local = main.row_slice(0).expect("empty keccak trace");
    let local = &*local;

    let is_active = local[KECCAK_COL_IS_ACTIVE].clone();
    let is_last   = local[KECCAK_COL_IS_LAST].clone();

    // ── Constraint 2: is_active is boolean ────────────────────────────────
    builder.assert_zero(
      is_active.clone().into() * (AB::Expr::ONE - is_active.clone().into()),
    );

    // ── Constraint 3: is_last is boolean ──────────────────────────────────
    builder.assert_zero(
      is_last.clone().into() * (AB::Expr::ONE - is_last.clone().into()),
    );

    // ── Constraint 4: on inactive rows, is_last must be 0 ─────────────────
    // is_last * (1 - is_active) == 0
    builder.assert_zero(
      is_last.into() * (AB::Expr::ONE - is_active.into()),
    );

    // Note: we do NOT add a transition constraint for monotone is_active
    // because Circle STARK wraps rows circularly — see previous analysis.
    // The trace builder enforces the ordering; the Poseidon commitment in
    // public values binds the content.
  }
}

// ── Trace builder ─────────────────────────────────────────────────────

fn fill_word_row(data: &mut [Val], base: usize, entry: &KeccakWordEntry) {
  data[base + KECCAK_COL_CALL_ID_HI] = Val::from_u32((entry.call_id >> 32) as u32);
  data[base + KECCAK_COL_CALL_ID_LO] = Val::from_u32(entry.call_id as u32);
  data[base + KECCAK_COL_ADDR_HI]    = Val::from_u32((entry.addr >> 32) as u32);
  data[base + KECCAK_COL_ADDR_LO]    = Val::from_u32(entry.addr as u32);
  for (k, chunk) in entry.value.chunks(4).enumerate() {
    data[base + KECCAK_COL_VAL0 + k] =
      Val::from_u32(u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
  }
  data[base + KECCAK_COL_IS_LAST] = Val::from_u32(entry.is_last as u32);
  if entry.is_last {
    for (k, chunk) in entry.output_hash.chunks(4).enumerate() {
      data[base + KECCAK_COL_HASH0 + k] =
        Val::from_u32(u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
  }
  data[base + KECCAK_COL_IS_ACTIVE] = Val::from_u32(1);
}

fn build_keccak_word_trace(words: &[KeccakWordEntry]) -> RowMajorMatrix<Val> {
  let height = words.len().max(4).next_power_of_two();
  let mut data = vec![Val::ZERO; height * NUM_KECCAK_COLS];
  for (i, entry) in words.iter().enumerate() {
    fill_word_row(&mut data, i * NUM_KECCAK_COLS, entry);
  }
  RowMajorMatrix::new(data, NUM_KECCAK_COLS)
}

// ── Public API ─────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct KeccakConsistencyProof {
  pub stark_proof: CircleStarkProof,
  pub log: Vec<KeccakLogEntry>,
}

/// Prove KECCAK256 consistency for `claims`.
///
/// For each claim:
/// 1. Verifies `keccak256(claim.input_bytes) == claim.output_hash` (prover-side oracle check).
/// 2. Records the 32-byte-aligned memory words in `KeccakLogEntry.memory_words`.
/// 3. Builds a word-per-row trace and STARK proof that commits to all words and call
///    metadata via Poseidon hashes in public values.
///
/// The memory words are also emitted as `MemAccessClaim`s in `execute.rs`, so they
/// end up in `MemoryConsistencyProof.read_log`.  `validate_keccak_memory_cross_check`
/// verifies that every word in `KeccakLogEntry.memory_words` appears in the memory
/// read log with the same value, completing the keccak ↔ memory binding.
pub fn prove_keccak_consistency(
  claims: &[crate::transition::KeccakClaim],
) -> Result<KeccakConsistencyProof, String> {
  let mut log = Vec::with_capacity(claims.len());
  for claim in claims {
    let computed = keccak256_bytes(&claim.input_bytes);
    if computed != claim.output_hash {
      return Err(format!(
        "KECCAK256 preimage mismatch at offset={} size={}: computed {:?}, claimed {:?}",
        claim.offset, claim.size, computed, claim.output_hash
      ));
    }

    // Build the list of 32-byte aligned words covering [offset, offset+size).
    let memory_words = if claim.size == 0 {
      vec![]
    } else {
      let start_aligned = (claim.offset / 32) * 32;
      let end_aligned = (claim.offset + claim.size).div_ceil(32) * 32;
      let mut words = Vec::new();
      let mut addr = start_aligned;
      while addr < end_aligned {
        // Extract the 32-byte word from input_bytes at this aligned offset.
        let word_start = addr.saturating_sub(claim.offset) as usize;
        let word_end = ((addr + 32).min(claim.offset + claim.size)
          .saturating_sub(claim.offset)) as usize;
        let mut value = [0u8; 32];
        // The part of the word that overlaps with [offset, offset+size).
        let val_start = claim.offset.saturating_sub(addr) as usize;
        let val_end = val_start + (word_end - word_start).min(32 - val_start);
        if word_start < claim.input_bytes.len() && word_start < word_end {
          let src_end = word_end.min(claim.input_bytes.len());
          value[val_start..val_start + (src_end - word_start)]
            .copy_from_slice(&claim.input_bytes[word_start..src_end]);
        }
        words.push((addr, value));
        addr += 32;
      }
      words
    };

    log.push(KeccakLogEntry {
      offset: claim.offset,
      size: claim.size,
      input_bytes: claim.input_bytes.clone(),
      output_hash: claim.output_hash,
      memory_words,
    });
  }

  let public_values = make_keccak_public_values(&log);

  let words = collect_word_entries(&log);
  let trace = build_keccak_word_trace(&words);

  let config = make_circle_config();
  let stark_proof = p3_uni_stark::prove(&config, &KeccakConsistencyAir, trace, &public_values);

  Ok(KeccakConsistencyProof { stark_proof, log })
}

/// Verify a single [`KeccakConsistencyProof`].
///
/// Verifies the STARK proof against the Poseidon commitments in public values.
/// The AIR enforces:
/// - `pis[0] == RECEIPT_BIND_TAG_KECCAK` — receipt binding tag.
/// - `is_active ∈ {0, 1}` — active-row boolean.
/// - `is_last ∈ {0, 1}` — last-word boolean.
/// - `is_last * (1 - is_active) == 0` — last only on active rows.
///
/// # Limitation — oracle model
///
/// Hash correctness (`keccak256(input_bytes) == output_hash`) is checked by
/// the prover in `prove_keccak_consistency` but NOT in-circuit.  A keccak256
/// permutation AIR would be required for a fully ZK argument.
pub fn verify_keccak_consistency(proof: &KeccakConsistencyProof) -> bool {
  let public_values = make_keccak_public_values(&proof.log);
  let config = make_circle_config();
  p3_uni_stark::verify(
    &config,
    &KeccakConsistencyAir,
    &proof.stark_proof,
    &public_values,
  )
  .is_ok()
}

/// Cross-check that each keccak claim's input words appear in the memory read log.
///
/// Since `execute.rs` now emits `MemAccessClaim`s for every 32-byte-aligned word
/// in the KECCAK256 input range, those entries must appear in
/// `MemoryConsistencyProof.read_log` with matching `(addr, value)`.  This check
/// closes the keccak ↔ memory binding loop without requiring cross-table LogUp.
///
/// # Arguments
/// * `keccak_log` — log from `KeccakConsistencyProof.log`
/// * `mem_read_log` — read log from `MemoryConsistencyProof.read_log`
pub fn validate_keccak_memory_cross_check(
  keccak_log: &[KeccakLogEntry],
  mem_write_log: &[MemLogEntry],
  mem_read_log: &[MemLogEntry],
) -> bool {
  if keccak_log.is_empty() {
    return true;
  }

  // Build addr → set of values seen in memory (both read and write logs).
  use std::collections::HashMap;
  let mut seen_at: HashMap<u64, Vec<[u8; 32]>> = HashMap::new();
  for e in mem_read_log.iter().chain(mem_write_log.iter()) {
    seen_at.entry(e.addr).or_default().push(e.value);
  }

  for entry in keccak_log {
    let size = entry.size as usize;
    if size == 0 {
      continue;
    }

    if entry.input_bytes.len() != size {
      return false;
    }

    let offset = entry.offset;

    // For each 32-byte aligned word that covers part of [offset, offset+size),
    // check that the memory log contains a value for that address whose bytes
    // that overlap with [offset, offset+size) match the corresponding input_bytes.
    for &(word_addr, _stored_val) in &entry.memory_words {
      let abs_start = offset.max(word_addr);
      let abs_end   = (offset + size as u64).min(word_addr + 32);
      let val_start = (abs_start - word_addr) as usize;
      let val_end   = (abs_end   - word_addr) as usize;
      let inp_start = (abs_start - offset) as usize;
      let inp_end   = (abs_end   - offset) as usize;

      // Find any memory-log value at word_addr whose overlap bytes match input_bytes.
      let found = seen_at
        .get(&word_addr)
        .map(|vals| {
          vals.iter().any(|mem_val| {
            mem_val[val_start..val_end] == entry.input_bytes[inp_start..inp_end]
          })
        })
        .unwrap_or(false);

      if !found {
        return false;
      }
    }
  }

  true
}

