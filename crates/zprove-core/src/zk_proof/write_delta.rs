//! Write-set management for cross-segment state transition proofs.
//!
//! # Design
//!
//! Each execution segment tracks which memory/storage addresses it **wrote**
//! and what their final values are.  This per-segment *write delta* `D_seg` is
//! committed as `d_seg_root = Poseidon2(sorted (addr, value) pairs)`.
//!
//! The cumulative write-set `W_i` at the boundary between segments is:
//! ```text
//! W_0 = ∅   (empty at execution start)
//! W_{i+1} = merge(W_i, D_seg_i)   (D_seg overrides W_in for conflicting addrs)
//! ```
//!
//! The final `W_N` is the public output: every address that the execution
//! wrote, with its last-written value.
//!
//! # Soundness constraint
//!
//! For the scheme to be sound, every **first read** of an address in segment `i`
//! (a read that has no preceding write in the same segment) must match the value
//! in `W_{i-1}`.  This is the "First-Read Initialization" check implemented by
//! [`validate_inherited_reads`] / [`validate_inherited_stor_reads`].
//!
//! These functions are called by the `_with_w_in` proving/verification variants
//! in `memory.rs` and `storage.rs`.

use super::types::{Val, default_poseidon_sponge};
use p3_field::PrimeCharacteristicRing;
use p3_symmetric::CryptographicHasher;
use std::collections::BTreeMap;

// ── Type aliases ──────────────────────────────────────────────────────────────

/// Cumulative write-set for EVM memory: `addr → last-written 32-byte word`.
///
/// `addr` is a 32-byte-aligned memory word address (EVM byte offset / 32 * 32).
/// BTreeMap ensures sorted iteration for deterministic Poseidon2 hashing.
pub type MemWriteSet = BTreeMap<u64, [u8; 32]>;

/// Cumulative write-set for EVM storage: `(contract, slot) → last-written value`.
pub type StorWriteSet = BTreeMap<([u8; 20], [u8; 32]), [u8; 32]>;

// ── Hash helpers ──────────────────────────────────────────────────────────────

/// Hash a [`MemWriteSet`] into a 32-byte commitment.
///
/// `W_root = Poseidon2(len ‖ addr_0_hi ‖ addr_0_lo ‖ val_0[0..8] ‖ … )` where
/// entries are sorted by addr (BTreeMap order).  Returns `[0u8; 32]` for empty sets.
pub fn hash_mem_write_set(ws: &MemWriteSet) -> [u8; 32] {
  if ws.is_empty() {
    return [0u8; 32];
  }
  let sponge = default_poseidon_sponge();
  let mut input: Vec<Val> = Vec::with_capacity(1 + ws.len() * 10);
  input.push(Val::from_u32(ws.len() as u32));
  for (&addr, value) in ws.iter() {
    input.push(Val::from_u32((addr >> 32) as u32));
    input.push(Val::from_u32(addr as u32));
    for chunk in value.chunks(4) {
      input.push(Val::from_u32(u32::from_be_bytes([
        chunk[0], chunk[1], chunk[2], chunk[3],
      ])));
    }
  }
  let hash: [Val; 8] = sponge.hash_iter(input);
  vals_to_bytes(&hash)
}

/// Hash a [`StorWriteSet`] into a 32-byte commitment.
///
/// Entries sorted by (contract, slot) (BTreeMap order).
pub fn hash_stor_write_set(ws: &StorWriteSet) -> [u8; 32] {
  if ws.is_empty() {
    return [0u8; 32];
  }
  let sponge = default_poseidon_sponge();
  let mut input: Vec<Val> = Vec::with_capacity(1 + ws.len() * 21);
  input.push(Val::from_u32(ws.len() as u32));
  for ((contract, slot), value) in ws.iter() {
    for i in 0..5usize {
      let base = i * 4;
      let end = (base + 4).min(20);
      let mut arr = [0u8; 4];
      arr[..end - base].copy_from_slice(&contract[base..end]);
      input.push(Val::from_u32(u32::from_be_bytes(arr)));
    }
    for chunk in slot.chunks(4) {
      input.push(Val::from_u32(u32::from_be_bytes([
        chunk[0], chunk[1], chunk[2], chunk[3],
      ])));
    }
    for chunk in value.chunks(4) {
      input.push(Val::from_u32(u32::from_be_bytes([
        chunk[0], chunk[1], chunk[2], chunk[3],
      ])));
    }
  }
  let hash: [Val; 8] = sponge.hash_iter(input);
  vals_to_bytes(&hash)
}

// ── Set operations ────────────────────────────────────────────────────────────

/// Merge `W_in` and `D_seg` into `W_out`.
///
/// When the same address appears in both, `D_seg` (new writes) takes priority.
pub fn merge_mem_write_sets(w_in: &MemWriteSet, d_seg: &MemWriteSet) -> MemWriteSet {
  let mut w_out = w_in.clone();
  for (&addr, &val) in d_seg.iter() {
    w_out.insert(addr, val);
  }
  w_out
}

/// Merge `W_in` and `D_seg` into `W_out` for storage.
pub fn merge_stor_write_sets(w_in: &StorWriteSet, d_seg: &StorWriteSet) -> StorWriteSet {
  let mut w_out = w_in.clone();
  for (key, &val) in d_seg.iter() {
    w_out.insert(*key, val);
  }
  w_out
}

/// Build a [`MemWriteSet`] (write delta) from an unordered `HashMap<u64, [u8;32]>`.
///
/// Used to convert `MemoryConsistencyProof::write_set` into a sortable form.
pub fn mem_write_set_from_map(
  map: &std::collections::HashMap<u64, [u8; 32]>,
) -> MemWriteSet {
  map.iter().map(|(&k, &v)| (k, v)).collect()
}

/// Build a [`StorWriteSet`] from an unordered HashMap.
pub fn stor_write_set_from_map(
  map: &std::collections::HashMap<([u8; 20], [u8; 32]), [u8; 32]>,
) -> StorWriteSet {
  map.iter().map(|(k, &v)| (*k, v)).collect()
}

// ── Soundness checks ──────────────────────────────────────────────────────────

/// Validate that all cross-segment reads are consistent with `W_in`.
///
/// `read_set` contains addresses that were **read in this segment without a
/// preceding write** (i.e., cross-segment inherited reads from earlier segments).
/// Each such read must observe the value that `W_in` recorded for that address.
///
/// If an address is absent from `W_in`, the only valid observation is `[0u8;32]`
/// — EVM memory starts zeroed and storage slots default to zero.
///
/// # Security
///
/// Without this check a malicious prover could supply arbitrary `value_before`
/// for the first read of an address, effectively forging a cross-segment state.
/// This function is the "First-Read Initialization" constraint described in the
/// design document.
pub fn validate_inherited_reads(
  read_set: &std::collections::HashMap<u64, [u8; 32]>,
  w_in: &MemWriteSet,
) -> bool {
  for (addr, &read_val) in read_set {
    match w_in.get(addr) {
      Some(&w_val) if w_val == read_val => {}
      Some(_) => return false, // prover claims a different value than W_in
      None => {
        // Address never written before: EVM memory is zero-initialised.
        if read_val != [0u8; 32] {
          return false;
        }
      }
    }
  }
  true
}

/// Validate inherited storage reads against `W_in`.
///
/// Storage slots default to zero when never written, just like memory.
pub fn validate_inherited_stor_reads(
  read_set: &std::collections::HashMap<([u8; 20], [u8; 32]), [u8; 32]>,
  w_in: &StorWriteSet,
) -> bool {
  for (key, &read_val) in read_set {
    match w_in.get(key) {
      Some(&w_val) if w_val == read_val => {}
      Some(_) => return false,
      None => {
        if read_val != [0u8; 32] {
          return false;
        }
      }
    }
  }
  true
}

// ── VmState memory root helpers ───────────────────────────────────────────────

/// Build a [`MemWriteSet`] from the memory claims of a sequence of steps,
/// applying only write operations in execution order.
///
/// Used in `execute.rs` to compute `VmState.memory_root` at segment boundaries.
pub fn compute_mem_write_set_from_claims(
  claims: impl Iterator<Item = crate::transition::MemAccessClaim>,
) -> MemWriteSet {
  let mut ws = MemWriteSet::new();
  for claim in claims {
    if claim.is_write {
      ws.insert(claim.addr, claim.value);
    }
  }
  ws
}

/// Build a [`StorWriteSet`] from storage claims of a sequence of steps.
pub fn compute_stor_write_set_from_claims(
  claims: impl Iterator<Item = crate::transition::StorageAccessClaim>,
) -> StorWriteSet {
  let mut ws = StorWriteSet::new();
  for claim in claims {
    if claim.is_write {
      ws.insert((claim.contract, claim.slot), claim.value);
    }
  }
  ws
}

// ── Private helpers ───────────────────────────────────────────────────────────

fn vals_to_bytes(vals: &[Val; 8]) -> [u8; 32] {
  use p3_field::PrimeField32;
  let mut out = [0u8; 32];
  for (i, v) in vals.iter().enumerate() {
    out[i * 4..(i + 1) * 4].copy_from_slice(&v.as_canonical_u32().to_be_bytes());
  }
  out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn empty_write_set_hash_is_zero() {
    assert_eq!(hash_mem_write_set(&MemWriteSet::new()), [0u8; 32]);
    assert_eq!(hash_stor_write_set(&StorWriteSet::new()), [0u8; 32]);
  }

  #[test]
  fn merge_overrides_w_in() {
    let mut w_in = MemWriteSet::new();
    w_in.insert(0, [1u8; 32]);
    w_in.insert(32, [2u8; 32]);

    let mut d_seg = MemWriteSet::new();
    d_seg.insert(0, [9u8; 32]); // override
    d_seg.insert(64, [3u8; 32]); // new

    let w_out = merge_mem_write_sets(&w_in, &d_seg);
    assert_eq!(w_out[&0], [9u8; 32]);
    assert_eq!(w_out[&32], [2u8; 32]);
    assert_eq!(w_out[&64], [3u8; 32]);
  }

  #[test]
  fn validate_inherited_reads_ok_if_matches_w_in() {
    let mut w_in = MemWriteSet::new();
    w_in.insert(0, [42u8; 32]);

    let mut read_set = std::collections::HashMap::new();
    read_set.insert(0u64, [42u8; 32]);

    assert!(validate_inherited_reads(&read_set, &w_in));
  }

  #[test]
  fn validate_inherited_reads_fail_on_mismatch() {
    let mut w_in = MemWriteSet::new();
    w_in.insert(0, [42u8; 32]);

    let mut read_set = std::collections::HashMap::new();
    read_set.insert(0u64, [7u8; 32]); // wrong value

    assert!(!validate_inherited_reads(&read_set, &w_in));
  }

  #[test]
  fn validate_inherited_reads_unwritten_addr_must_be_zero() {
    let w_in = MemWriteSet::new();

    let mut read_set = std::collections::HashMap::new();
    read_set.insert(0u64, [0u8; 32]); // ok: zero
    assert!(validate_inherited_reads(&read_set, &w_in));

    let mut read_set2 = std::collections::HashMap::new();
    read_set2.insert(0u64, [1u8; 32]); // not ok: non-zero from unwritten addr
    assert!(!validate_inherited_reads(&read_set2, &w_in));
  }
}
