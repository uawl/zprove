//! Storage consistency proof: SLOAD/SSTORE read/write set intersection + aggregation.

pub const RECEIPT_BIND_TAG_STORAGE: u32 = 5;

use super::types::{
  Challenge, CircleStarkConfig, CircleStarkProof, Val,
  default_poseidon_sponge, make_circle_config,
};
use super::memory::value_to_u32s;

use p3_field::PrimeCharacteristicRing;
use p3_lookup::logup::LogUpGadget;
use p3_lookup::lookup_traits::{Kind, Lookup, LookupData, LookupGadget};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_symmetric::CryptographicHasher;
use p3_uni_stark::{
  Entry, SymbolicExpression, SymbolicVariable, VerifierConstraintFolder, prove_with_lookup,
  verify_with_lookup,
};

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
      rw_counter: c.rw_counter, contract: c.contract, slot: c.slot, value: c.value,
    };
    if c.is_write { writes.push(entry); } else { reads.push(entry); }
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
    for u in contract_to_u32s(&e.contract) { input.push(Val::from_u32(u)); }
    for chunk in e.slot.chunks(4) {
      input.push(Val::from_u32(u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])));
    }
    for chunk in e.value.chunks(4) {
      input.push(Val::from_u32(u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])));
    }
  }
  sponge.hash_iter(input)
}

fn hash_storage_set(set: &StorageSet) -> [Val; 8] {
  if set.is_empty() { return [Val::ZERO; 8]; }
  let mut sorted: Vec<(StorageKey, [u8; 32])> = set.iter().map(|(&k, &v)| (k, v)).collect();
  sorted.sort_by_key(|((c, s), _)| (*c, *s));

  let sponge = default_poseidon_sponge();
  let mut input: Vec<Val> = Vec::with_capacity(1 + sorted.len() * 21);
  input.push(Val::from_u32(sorted.len() as u32));
  for ((contract, slot), value) in &sorted {
    for u in contract_to_u32s(contract) { input.push(Val::from_u32(u)); }
    for chunk in slot.chunks(4) {
      input.push(Val::from_u32(u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])));
    }
    for chunk in value.chunks(4) {
      input.push(Val::from_u32(u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])));
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
        std::collections::hash_map::Entry::Vacant(e) => { e.insert(claim.value); }
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
  for e in write_log { merged.push((e.rw_counter, true, (e.contract, e.slot), e.value)); }
  for e in read_log  { merged.push((e.rw_counter, false, (e.contract, e.slot), e.value)); }
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
          "storage read/write mismatch at contract={:?} slot={:?}", key.0, key.1
        ));
      }
    } else {
      match read_set.entry(key) {
        std::collections::hash_map::Entry::Vacant(e) => { e.insert(value); }
        std::collections::hash_map::Entry::Occupied(e) => {
          if *e.get() != value {
            return Err(format!(
              "cross-batch storage read mismatch at contract={:?} slot={:?}", key.0, key.1
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
  fn width(&self) -> usize { NUM_STOR_COLS }
}

impl<AB: p3_air::AirBuilderWithPublicValues> p3_air::Air<AB> for StorageConsistencyAir
where AB::F: p3_field::Field,
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
  let col = |c: usize| {
    SymbolicExpression::Variable(SymbolicVariable::new(Entry::Main { offset: 0 }, c))
  };
  Lookup::new(
    Kind::Local,
    vec![vec![
      col(STOR_COL_CONTRACT0), col(STOR_COL_CONTRACT0 + 1), col(STOR_COL_CONTRACT0 + 2),
      col(STOR_COL_CONTRACT0 + 3), col(STOR_COL_CONTRACT0 + 4),
      col(STOR_COL_SLOT0), col(STOR_COL_SLOT0 + 1), col(STOR_COL_SLOT0 + 2),
      col(STOR_COL_SLOT0 + 3), col(STOR_COL_SLOT0 + 4), col(STOR_COL_SLOT0 + 5),
      col(STOR_COL_SLOT0 + 6), col(STOR_COL_SLOT0 + 7),
      col(STOR_COL_VAL0), col(STOR_COL_VAL0 + 1), col(STOR_COL_VAL0 + 2),
      col(STOR_COL_VAL0 + 3), col(STOR_COL_VAL0 + 4), col(STOR_COL_VAL0 + 5),
      col(STOR_COL_VAL0 + 6), col(STOR_COL_VAL0 + 7),
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
    main_trace, &None, &[], &[lookup], &mut lookup_data, perm_challenges,
  );
  Some(perm_trace)
}

fn eval_stor_lookup(folder: &mut VerifierConstraintFolder<CircleStarkConfig>) {
  LogUpGadget::new().eval_local_lookup(folder, &make_stor_lookup());
}

// ── Trace builder ─────────────────────────────────────────────────────

fn fill_storage_log_row(data: &mut [Val], base: usize, entry: &StorageLogEntry, is_write: bool, mult: i32) {
  let cs = contract_to_u32s(&entry.contract);
  for k in 0..5 { data[base + STOR_COL_CONTRACT0 + k] = Val::from_u32(cs[k]); }
  for (k, chunk) in entry.slot.chunks(4).enumerate() {
    data[base + STOR_COL_SLOT0 + k] =
      Val::from_u32(u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
  }
  let vs = value_to_u32s(&entry.value);
  for k in 0..8 { data[base + STOR_COL_VAL0 + k] = Val::from_u32(vs[k]); }
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
  let mut last_write_rw: std::collections::HashMap<StorageKey, u64> = Default::default();
  for e in write_log {
    let key = (e.contract, e.slot);
    let rw = last_write_rw.entry(key).or_insert(0);
    if e.rw_counter > *rw { *rw = e.rw_counter; }
  }

  let mut intra_read_count: std::collections::HashMap<StorageKey, i32> = Default::default();
  let mut intra_read_rw_set: std::collections::HashSet<u64> = Default::default();
  for e in read_log {
    let key = (e.contract, e.slot);
    if let Some(&lw_rw) = last_write_rw.get(&key) {
      if e.rw_counter > lw_rw {
        *intra_read_count.entry(key).or_insert(0) += 1;
        intra_read_rw_set.insert(e.rw_counter);
      }
    }
  }

  let mut last_write_idx: std::collections::HashMap<StorageKey, usize> = Default::default();
  for (i, e) in write_log.iter().enumerate() {
    last_write_idx.insert((e.contract, e.slot), i);
  }

  let n_total = write_log.len() + read_log.len();
  let height = n_total.max(4).next_power_of_two();
  let mut data = vec![Val::ZERO; height * NUM_STOR_COLS];

  for (i, entry) in write_log.iter().enumerate() {
    let key: StorageKey = (entry.contract, entry.slot);
    let mult = if last_write_idx.get(&key) == Some(&i) {
      *intra_read_count.get(&key).unwrap_or(&0)
    } else { 0 };
    fill_storage_log_row(&mut data, i * NUM_STOR_COLS, entry, true, mult);
  }
  for (i, entry) in read_log.iter().enumerate() {
    let mult = if intra_read_rw_set.contains(&entry.rw_counter) { -1 } else { 0 };
    fill_storage_log_row(&mut data, (write_log.len() + i) * NUM_STOR_COLS, entry, false, mult);
  }
  RowMajorMatrix::new(data, NUM_STOR_COLS)
}

// ── Public API ────────────────────────────────────────────────────────

pub struct StorageConsistencyProof {
  pub stark_proof: CircleStarkProof,
  pub write_set: StorageSet,
  pub read_set: StorageSet,
  pub write_log: Vec<StorageLogEntry>,
  pub read_log: Vec<StorageLogEntry>,
}

pub fn prove_storage_consistency(
  claims: &[crate::transition::StorageAccessClaim],
) -> Result<StorageConsistencyProof, String> {
  let (write_log, read_log) = split_storage_logs(claims);
  let (write_set, read_set) = check_storage_claims_and_build_sets(claims)?;
  let public_values = make_storage_public_values(&write_log, &read_log, &write_set, &read_set);

  let trace = if write_log.is_empty() && read_log.is_empty() {
    RowMajorMatrix::new(vec![Val::ZERO; 4 * NUM_STOR_COLS], NUM_STOR_COLS)
  } else {
    build_storage_log_trace(&write_log, &read_log, &write_set)
  };

  let config = make_circle_config();
  let trace_for_perm = trace.clone();
  let stark_proof = prove_with_lookup(
    &config, &StorageConsistencyAir, trace, &public_values, None,
    move |perm_challenges| generate_stor_perm_trace(&trace_for_perm, perm_challenges),
    2, 2,
    |folder: &mut VerifierConstraintFolder<CircleStarkConfig>| eval_stor_lookup(folder),
  );
  Ok(StorageConsistencyProof { stark_proof, write_set, read_set, write_log, read_log })
}

pub fn verify_storage_consistency(proof: &StorageConsistencyProof) -> bool {
  let (derived_write_set, derived_read_set) =
    match derive_storage_sets_from_logs(&proof.write_log, &proof.read_log) {
      Ok(s) => s,
      Err(_) => return false,
    };
  if derived_write_set != proof.write_set || derived_read_set != proof.read_set { return false; }
  let public_values = make_storage_public_values(
    &proof.write_log, &proof.read_log, &proof.write_set, &proof.read_set,
  );
  let config = make_circle_config();
  verify_with_lookup(
    &config, &StorageConsistencyAir, &proof.stark_proof, &public_values, None,
    2,
    |folder: &mut VerifierConstraintFolder<CircleStarkConfig>| eval_stor_lookup(folder),
  ).is_ok()
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
  merge_storage_sets(&left.write_set, &left.read_set, &right.write_set, &right.read_set)
}

fn merge_storage_sets(
  left_ws: &StorageSet, left_rs: &StorageSet,
  right_ws: &StorageSet, right_rs: &StorageSet,
) -> Result<AggregatedStorageProof, String> {
  for (key, &r_val) in right_rs {
    if let Some(&l_val) = left_ws.get(key) {
      if r_val != l_val {
        return Err(format!(
          "storage read/write set mismatch at contract={:?} slot={:?}: right read {:?}, left wrote {:?}",
          key.0, key.1, r_val, l_val,
        ));
      }
    }
  }
  let mut write_set = left_ws.clone();
  for (&key, &val) in right_ws { write_set.insert(key, val); }
  let mut read_set = left_rs.clone();
  for (&key, &val) in right_rs {
    if !left_ws.contains_key(&key) { read_set.entry(key).or_insert(val); }
  }
  Ok(AggregatedStorageProof { write_set, read_set })
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
        Ok(AggregatedStorageProof { write_set: p.write_set.clone(), read_set: p.read_set.clone() })
      }
    })
    .collect::<Result<Vec<_>, _>>()?;

  while nodes.len() > 1 {
    let mut next_level = Vec::with_capacity((nodes.len() + 1) / 2);
    let mut iter = nodes.into_iter().peekable();
    while let Some(left) = iter.next() {
      if let Some(right) = iter.next() {
        let merged = merge_storage_sets(&left.write_set, &left.read_set, &right.write_set, &right.read_set)?;
        next_level.push(merged);
      } else {
        next_level.push(left);
      }
    }
    nodes = next_level;
  }
  Ok(nodes.into_iter().next().unwrap())
}
