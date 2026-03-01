//! Storage consistency proof: SLOAD/SSTORE read/write set intersection + aggregation.

pub const RECEIPT_BIND_TAG_STORAGE: u32 = 5;

use super::memory::value_to_u32s;
use super::types::{
  Challenge, CircleStarkConfig, CircleStarkProof, Val, default_poseidon_sponge, make_circle_config,
};

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

// ── Column layout  (NUM_STOR_COLS = 23) ──────────────────────────────

const STOR_COL_CONTRACT0: usize = 0;
const STOR_COL_SLOT0: usize = 5;
const STOR_COL_VAL0: usize = 13;
const STOR_COL_IS_WRITE: usize = 21;
const STOR_COL_MULT: usize = 22;
pub const NUM_STOR_COLS: usize = 23;
const STOR_PUBLIC_VALUES_LEN: usize = 35;

// ── Key and set types ─────────────────────────────────────────────────

pub type StorageKey = ([u8; 20], [u8; 32]);
pub type StorageSet = std::collections::HashMap<StorageKey, [u8; 32]>;

// ── Log entry ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageLogEntry {
  pub rw_counter: u64,
  pub contract: [u8; 20],
  pub slot: [u8; 32],
  pub value: [u8; 32],
}

pub fn split_storage_logs(
  claims: &[crate::transition::StorageAccessClaim],
) -> (Vec<StorageLogEntry>, Vec<StorageLogEntry>) {
  let mut writes = Vec::new();
  let mut reads = Vec::new();
  for c in claims {
    let entry = StorageLogEntry {
      rw_counter: c.rw_counter,
      contract: c.contract,
      slot: c.slot,
      value: c.value,
    };
    if c.is_write {
      writes.push(entry);
    } else {
      reads.push(entry);
    }
  }
  (writes, reads)
}

// ── Hash helpers ──────────────────────────────────────────────────────

fn contract_to_u32s(contract: &[u8; 20]) -> [u32; 5] {
  let mut out = [0u32; 5];
  for (i, chunk) in contract.chunks(4).enumerate() {
    out[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
  }
  out
}

fn hash_storage_log(log: &[StorageLogEntry]) -> [Val; 8] {
  let sponge = default_poseidon_sponge();
  let mut input: Vec<Val> = Vec::with_capacity(1 + log.len() * 23);
  input.push(Val::from_u32(log.len() as u32));
  for e in log {
    input.push(Val::from_u32((e.rw_counter >> 32) as u32));
    input.push(Val::from_u32(e.rw_counter as u32));
    for u in contract_to_u32s(&e.contract) {
      input.push(Val::from_u32(u));
    }
    for chunk in e.slot.chunks(4) {
      input.push(Val::from_u32(u32::from_be_bytes([
        chunk[0], chunk[1], chunk[2], chunk[3],
      ])));
    }
    for chunk in e.value.chunks(4) {
      input.push(Val::from_u32(u32::from_be_bytes([
        chunk[0], chunk[1], chunk[2], chunk[3],
      ])));
    }
  }
  sponge.hash_iter(input)
}

fn hash_storage_set(set: &StorageSet) -> [Val; 8] {
  if set.is_empty() {
    return [Val::ZERO; 8];
  }
  let mut sorted: Vec<(StorageKey, [u8; 32])> = set.iter().map(|(&k, &v)| (k, v)).collect();
  sorted.sort_by_key(|((c, s), _)| (*c, *s));

  let sponge = default_poseidon_sponge();
  let mut input: Vec<Val> = Vec::with_capacity(1 + sorted.len() * 21);
  input.push(Val::from_u32(sorted.len() as u32));
  for ((contract, slot), value) in &sorted {
    for u in contract_to_u32s(contract) {
      input.push(Val::from_u32(u));
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
  sponge.hash_iter(input)
}

fn make_storage_public_values(
  write_log: &[StorageLogEntry],
  read_log: &[StorageLogEntry],
  write_set: &StorageSet,
  read_set: &StorageSet,
) -> Vec<Val> {
  let wlh = hash_storage_log(write_log);
  let rlh = hash_storage_log(read_log);
  let wsh = hash_storage_set(write_set);
  let rsh = hash_storage_set(read_set);
  let mut pv = Vec::with_capacity(STOR_PUBLIC_VALUES_LEN);
  pv.push(Val::from_u32(RECEIPT_BIND_TAG_STORAGE));
  pv.push(Val::from_u32(write_log.len() as u32));
  pv.extend_from_slice(&wlh);
  pv.push(Val::from_u32(read_log.len() as u32));
  pv.extend_from_slice(&rlh);
  pv.extend_from_slice(&wsh);
  pv.extend_from_slice(&rsh);
  pv
}

// ── Intra-batch consistency checker ──────────────────────────────────

fn check_storage_claims_and_build_sets(
  claims: &[crate::transition::StorageAccessClaim],
) -> Result<(StorageSet, StorageSet), String> {
  use std::collections::HashMap;
  let mut write_set: StorageSet = HashMap::new();
  let mut read_set: StorageSet = HashMap::new();
  let mut last_write: StorageSet = HashMap::new();

  for claim in claims {
    let key: StorageKey = (claim.contract, claim.slot);
    if claim.is_write {
      last_write.insert(key, claim.value);
      write_set.insert(key, claim.value);
    } else if let Some(&written) = last_write.get(&key) {
      if claim.value != written {
        return Err(format!(
          "storage read/write mismatch at contract={:?} slot={:?}: read {:?}, last write {:?}",
          claim.contract, claim.slot, claim.value, written
        ));
      }
    } else {
      match read_set.entry(key) {
        std::collections::hash_map::Entry::Vacant(e) => {
          e.insert(claim.value);
        }
        std::collections::hash_map::Entry::Occupied(e) => {
          if *e.get() != claim.value {
            return Err(format!(
              "cross-batch storage read mismatch at contract={:?} slot={:?}",
              claim.contract, claim.slot
            ));
          }
        }
      }
    }
  }
  Ok((write_set, read_set))
}

fn derive_storage_sets_from_logs(
  write_log: &[StorageLogEntry],
  read_log: &[StorageLogEntry],
) -> Result<(StorageSet, StorageSet), String> {
  let mut merged: Vec<(u64, bool, StorageKey, [u8; 32])> = Vec::new();
  for e in write_log {
    merged.push((e.rw_counter, true, (e.contract, e.slot), e.value));
  }
  for e in read_log {
    merged.push((e.rw_counter, false, (e.contract, e.slot), e.value));
  }
  merged.sort_by_key(|(rw, _, _, _)| *rw);

  let mut write_set: StorageSet = std::collections::HashMap::new();
  let mut read_set: StorageSet = std::collections::HashMap::new();
  let mut last_write: StorageSet = std::collections::HashMap::new();

  for (_, is_write, key, value) in merged {
    if is_write {
      last_write.insert(key, value);
      write_set.insert(key, value);
    } else if let Some(&written) = last_write.get(&key) {
      if value != written {
        return Err(format!(
          "storage read/write mismatch at contract={:?} slot={:?}",
          key.0, key.1
        ));
      }
    } else {
      match read_set.entry(key) {
        std::collections::hash_map::Entry::Vacant(e) => {
          e.insert(value);
        }
        std::collections::hash_map::Entry::Occupied(e) => {
          if *e.get() != value {
            return Err(format!(
              "cross-batch storage read mismatch at contract={:?} slot={:?}",
              key.0, key.1
            ));
          }
        }
      }
    }
  }
  Ok((write_set, read_set))
}

// ── AIR ──────────────────────────────────────────────────────────────

pub struct StorageConsistencyAir;

impl<F: p3_field::Field> p3_air::BaseAir<F> for StorageConsistencyAir {
  fn width(&self) -> usize {
    NUM_STOR_COLS
  }
}

impl<AB: p3_air::AirBuilderWithPublicValues> p3_air::Air<AB> for StorageConsistencyAir
where
  AB::F: p3_field::Field,
{
  fn eval(&self, builder: &mut AB) {
    let pis = builder.public_values();
    builder.assert_eq(pis[0].into(), AB::Expr::from_u32(RECEIPT_BIND_TAG_STORAGE));
    let main = builder.main();
    let local = main.row_slice(0).expect("empty storage trace");
    let local = &*local;
    let is_write = local[STOR_COL_IS_WRITE].clone();
    builder.assert_zero(is_write.clone().into() * (AB::Expr::ONE - is_write.into()));
  }
}

// ── Lookup descriptor ────────────────────────────────────────────────

fn make_stor_lookup() -> Lookup<Val> {
  let col =
    |c: usize| SymbolicExpression::Variable(SymbolicVariable::new(Entry::Main { offset: 0 }, c));
  Lookup::new(
    Kind::Local,
    vec![vec![
      col(STOR_COL_CONTRACT0),
      col(STOR_COL_CONTRACT0 + 1),
      col(STOR_COL_CONTRACT0 + 2),
      col(STOR_COL_CONTRACT0 + 3),
      col(STOR_COL_CONTRACT0 + 4),
      col(STOR_COL_SLOT0),
      col(STOR_COL_SLOT0 + 1),
      col(STOR_COL_SLOT0 + 2),
      col(STOR_COL_SLOT0 + 3),
      col(STOR_COL_SLOT0 + 4),
      col(STOR_COL_SLOT0 + 5),
      col(STOR_COL_SLOT0 + 6),
      col(STOR_COL_SLOT0 + 7),
      col(STOR_COL_VAL0),
      col(STOR_COL_VAL0 + 1),
      col(STOR_COL_VAL0 + 2),
      col(STOR_COL_VAL0 + 3),
      col(STOR_COL_VAL0 + 4),
      col(STOR_COL_VAL0 + 5),
      col(STOR_COL_VAL0 + 6),
      col(STOR_COL_VAL0 + 7),
    ]],
    vec![col(STOR_COL_MULT)],
    vec![0],
  )
}

fn generate_stor_perm_trace(
  main_trace: &RowMajorMatrix<Val>,
  perm_challenges: &[Challenge],
) -> Option<RowMajorMatrix<Challenge>> {
  let gadget = LogUpGadget::new();
  let lookup = make_stor_lookup();
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

fn eval_stor_lookup(folder: &mut VerifierConstraintFolder<CircleStarkConfig>) {
  LogUpGadget::new().eval_local_lookup(folder, &make_stor_lookup());
}

// ── Trace builder ─────────────────────────────────────────────────────

fn fill_storage_log_row(
  data: &mut [Val],
  base: usize,
  entry: &StorageLogEntry,
  is_write: bool,
  mult: i32,
) {
  let cs = contract_to_u32s(&entry.contract);
  for k in 0..5 {
    data[base + STOR_COL_CONTRACT0 + k] = Val::from_u32(cs[k]);
  }
  for (k, chunk) in entry.slot.chunks(4).enumerate() {
    data[base + STOR_COL_SLOT0 + k] =
      Val::from_u32(u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
  }
  let vs = value_to_u32s(&entry.value);
  for k in 0..8 {
    data[base + STOR_COL_VAL0 + k] = Val::from_u32(vs[k]);
  }
  data[base + STOR_COL_IS_WRITE] = Val::from_u32(is_write as u32);
  data[base + STOR_COL_MULT] = if mult >= 0 {
    Val::from_u32(mult as u32)
  } else {
    -Val::from_u32((-mult) as u32)
  };
}

fn build_storage_log_trace(
  write_log: &[StorageLogEntry],
  read_log: &[StorageLogEntry],
  _write_set: &StorageSet,
) -> RowMajorMatrix<Val> {
  // ── Step 1: last-write rw_counter and original index per (contract, slot) ─
  // Sort writes by (key ASC, rw_counter ASC).  The final entry in each key
  // group carries the highest rw_counter and the original "last write" index.
  let mut writes_sorted: Vec<(StorageKey, u64, usize)> = write_log
    .iter()
    .enumerate()
    .map(|(i, e)| ((e.contract, e.slot), e.rw_counter, i))
    .collect();
  writes_sorted.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
  let last_writes: Vec<(StorageKey, u64, usize)> = {
    let mut v: Vec<(StorageKey, u64, usize)> = Vec::new();
    for (key, rw, idx) in writes_sorted {
      match v.last_mut() {
        Some((k, prev_rw, prev_idx)) if *k == key => {
          *prev_rw = rw;   // rw is non-decreasing within the key group
          *prev_idx = idx;
        }
        _ => v.push((key, rw, idx)),
      }
    }
    v
  };

  // ── Step 2: classify reads as intra-batch or cross-batch ─────────────────
  // A read is intra-batch when its rw_counter > last write's rw for the same
  // (contract, slot).  Collect their rw_counters (for per-read MULT = -1) and
  // their keys (for per-key count → last-write MULT).
  let mut intra_rw: Vec<u64> = Vec::new();
  let mut intra_keys: Vec<StorageKey> = Vec::new();
  for e in read_log {
    let key = (e.contract, e.slot);
    if let Ok(pos) = last_writes.binary_search_by(|(k, _, _)| k.cmp(&key)) {
      if e.rw_counter > last_writes[pos].1 {
        intra_rw.push(e.rw_counter);
        intra_keys.push(key);
      }
    }
  }

  // Count intra-batch reads per key (assigned as multiplicity on the last write).
  intra_keys.sort_unstable();
  let key_counts: Vec<(StorageKey, i32)> = {
    let mut v: Vec<(StorageKey, i32)> = Vec::new();
    for key in intra_keys {
      match v.last_mut() {
        Some((k, c)) if *k == key => *c += 1,
        _ => v.push((key, 1)),
      }
    }
    v
  };

  // Sort intra rw_counters for O(log n) membership test on the read side.
  intra_rw.sort_unstable();

  // ── Fill trace ────────────────────────────────────────────────────────────
  let n_total = write_log.len() + read_log.len();
  let height = n_total.max(4).next_power_of_two();
  let mut data = vec![Val::ZERO; height * NUM_STOR_COLS];

  for (i, entry) in write_log.iter().enumerate() {
    let key: StorageKey = (entry.contract, entry.slot);
    let mult = match last_writes.binary_search_by(|(k, _, _)| k.cmp(&key)) {
      Ok(pos) if last_writes[pos].2 == i => {
        key_counts
          .binary_search_by(|(k, _)| k.cmp(&key))
          .map_or(0, |cp| key_counts[cp].1)
      }
      _ => 0,
    };
    fill_storage_log_row(&mut data, i * NUM_STOR_COLS, entry, true, mult);
  }
  for (i, entry) in read_log.iter().enumerate() {
    let mult = if intra_rw.binary_search(&entry.rw_counter).is_ok() { -1 } else { 0 };
    fill_storage_log_row(
      &mut data,
      (write_log.len() + i) * NUM_STOR_COLS,
      entry,
      false,
      mult,
    );
  }
  RowMajorMatrix::new(data, NUM_STOR_COLS)
}

// ── Public API ────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct StorageConsistencyProof {
  pub stark_proof: CircleStarkProof,
  pub write_set: StorageSet,
  pub read_set: StorageSet,
  pub write_log: Vec<StorageLogEntry>,
  pub read_log: Vec<StorageLogEntry>,
  /// Commitment to `write_set` — i.e. `D_seg_root` for storage.
  /// Equals `hash_stor_write_set(BTreeMap from write_set)`. `[0u8;32]` for empty segments.
  pub d_seg_root: [u8; 32],
}

pub fn prove_storage_consistency(
  claims: &[crate::transition::StorageAccessClaim],
) -> Result<StorageConsistencyProof, String> {
  prove_storage_consistency_inner(claims, None)
}

/// Like [`prove_storage_consistency`] but also validates cross-segment reads.
///
/// `w_in` is the cumulative storage write-set `W_{i-1}` at the start of this segment.
pub fn prove_storage_consistency_with_w_in(
  claims: &[crate::transition::StorageAccessClaim],
  w_in: &super::write_delta::StorWriteSet,
) -> Result<StorageConsistencyProof, String> {
  prove_storage_consistency_inner(claims, Some(w_in))
}

fn prove_storage_consistency_inner(
  claims: &[crate::transition::StorageAccessClaim],
  w_in: Option<&super::write_delta::StorWriteSet>,
) -> Result<StorageConsistencyProof, String> {
  let (write_log, read_log) = split_storage_logs(claims);
  let (write_set, read_set) = check_storage_claims_and_build_sets(claims)?;

  // First-Read Initialization soundness check.
  if let Some(w_in) = w_in {
    if !super::write_delta::validate_inherited_stor_reads(&read_set, w_in) {
      return Err(
        "prove_storage_consistency: inherited read value does not match W_in".to_string(),
      );
    }
  }

  let d_seg_root = {
    let bmap = super::write_delta::stor_write_set_from_map(&write_set);
    super::write_delta::hash_stor_write_set(&bmap)
  };

  let public_values = make_storage_public_values(&write_log, &read_log, &write_set, &read_set);

  let trace = if write_log.is_empty() && read_log.is_empty() {
    RowMajorMatrix::new(vec![Val::ZERO; 4 * NUM_STOR_COLS], NUM_STOR_COLS)
  } else {
    build_storage_log_trace(&write_log, &read_log, &write_set)
  };

  let config = make_circle_config();
  let hints =
    air_cache::get_or_compute(&StorageConsistencyAir, 0, public_values.len(), 0);
  let trace_for_perm = trace.clone();
  let stark_proof = prove_with_lookup_hinted(
    &config,
    &StorageConsistencyAir,
    trace,
    &public_values,
    None,
    move |perm_challenges| generate_stor_perm_trace(&trace_for_perm, perm_challenges),
    2,
    2,
    |folder: &mut VerifierConstraintFolder<CircleStarkConfig>| eval_stor_lookup(folder),
    hints,
  );
  Ok(StorageConsistencyProof {
    stark_proof,
    write_set,
    read_set,
    write_log,
    read_log,
    d_seg_root,
  })
}

pub fn verify_storage_consistency(proof: &StorageConsistencyProof) -> bool {
  verify_storage_consistency_inner(proof, None)
}

/// Like [`verify_storage_consistency`] but also re-checks the inherited-read constraint.
pub fn verify_storage_consistency_with_w_in(
  proof: &StorageConsistencyProof,
  w_in: &super::write_delta::StorWriteSet,
) -> bool {
  verify_storage_consistency_inner(proof, Some(w_in))
}

fn verify_storage_consistency_inner(
  proof: &StorageConsistencyProof,
  w_in: Option<&super::write_delta::StorWriteSet>,
) -> bool {
  let (derived_write_set, derived_read_set) =
    match derive_storage_sets_from_logs(&proof.write_log, &proof.read_log) {
      Ok(s) => s,
      Err(_) => return false,
    };
  if derived_write_set != proof.write_set || derived_read_set != proof.read_set {
    return false;
  }
  // Verify d_seg_root commitment.
  let expected_d_seg_root = {
    let bmap = super::write_delta::stor_write_set_from_map(&proof.write_set);
    super::write_delta::hash_stor_write_set(&bmap)
  };
  if proof.d_seg_root != expected_d_seg_root {
    return false;
  }
  // First-Read Initialization check (optional).
  if let Some(w_in) = w_in {
    if !super::write_delta::validate_inherited_stor_reads(&proof.read_set, w_in) {
      return false;
    }
  }
  let public_values = make_storage_public_values(
    &proof.write_log,
    &proof.read_log,
    &proof.write_set,
    &proof.read_set,
  );
  let config = make_circle_config();
  verify_with_lookup(
    &config,
    &StorageConsistencyAir,
    &proof.stark_proof,
    &public_values,
    None,
    2,
    |folder: &mut VerifierConstraintFolder<CircleStarkConfig>| eval_stor_lookup(folder),
  )
  .is_ok()
}

// ── Binary-tree aggregation ───────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AggregatedStorageProof {
  pub write_set: StorageSet,
  pub read_set: StorageSet,
}

pub fn aggregate_storage_proofs(
  left: &StorageConsistencyProof,
  right: &StorageConsistencyProof,
) -> Result<AggregatedStorageProof, String> {
  if !verify_storage_consistency(left) {
    return Err("left child storage proof failed verification".to_string());
  }
  if !verify_storage_consistency(right) {
    return Err("right child storage proof failed verification".to_string());
  }
  merge_storage_sets(
    &left.write_set,
    &left.read_set,
    &right.write_set,
    &right.read_set,
  )
}

fn merge_storage_sets(
  left_ws: &StorageSet,
  left_rs: &StorageSet,
  right_ws: &StorageSet,
  right_rs: &StorageSet,
) -> Result<AggregatedStorageProof, String> {
  for (key, &r_val) in right_rs {
    if let Some(&l_val) = left_ws.get(key) {
      // Slot was written by the left segment — right must read that value.
      if r_val != l_val {
        return Err(format!(
          "storage read/write set mismatch at contract={:?} slot={:?}: right read {:?}, left wrote {:?}",
          key.0, key.1, r_val, l_val,
        ));
      }
    } else if let Some(&inherited) = left_rs.get(key) {
      // Slot was inherited by left from an even earlier segment.
      if r_val != inherited {
        return Err(format!(
          "storage inherited read mismatch at contract={:?} slot={:?}: right read {:?}, left inherited {:?}",
          key.0, key.1, r_val, inherited,
        ));
      }
    }
    // Otherwise: slot not yet resolved in the left subtree.  The unresolved
    // read propagates upward into the merged read_set, where it will be checked
    // against the stor_w_in passed to prove_batch_transaction_zk_receipt_with_w_in.
    // Zero-init enforcement is done there, not here, so that binary-tree
    // aggregation can work correctly when left and right come from different
    // subtrees whose writes are not yet combined.
  }
  let mut write_set = left_ws.clone();
  for (&key, &val) in right_ws {
    write_set.insert(key, val);
  }
  let mut read_set = left_rs.clone();
  for (&key, &val) in right_rs {
    if !left_ws.contains_key(&key) {
      read_set.entry(key).or_insert(val);
    }
  }
  Ok(AggregatedStorageProof {
    write_set,
    read_set,
  })
}

pub fn aggregate_storage_proofs_tree(
  proofs: &[StorageConsistencyProof],
) -> Result<AggregatedStorageProof, String> {
  if proofs.is_empty() {
    return Err("aggregate_storage_proofs_tree: empty proof slice".to_string());
  }
  let mut nodes: Vec<AggregatedStorageProof> = proofs
    .iter()
    .enumerate()
    .map(|(i, p)| {
      if !verify_storage_consistency(p) {
        Err(format!("leaf storage proof {i} failed verification"))
      } else {
        Ok(AggregatedStorageProof {
          write_set: p.write_set.clone(),
          read_set: p.read_set.clone(),
        })
      }
    })
    .collect::<Result<Vec<_>, _>>()?;

  while nodes.len() > 1 {
    let mut next_level = Vec::with_capacity(nodes.len().div_ceil(2));
    let mut iter = nodes.into_iter().peekable();
    while let Some(left) = iter.next() {
      if let Some(right) = iter.next() {
        let merged = merge_storage_sets(
          &left.write_set,
          &left.read_set,
          &right.write_set,
          &right.read_set,
        )?;
        next_level.push(merged);
      } else {
        next_level.push(left);
      }
    }
    nodes = next_level;
  }
  Ok(nodes.into_iter().next().unwrap())
}
