//! Memory consistency proof: read/write set intersection + binary-tree aggregation.

pub const RECEIPT_BIND_TAG_MEM: u32 = 4;

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
  Entry, SymbolicExpression, SymbolicVariable, VerifierConstraintFolder, prove_with_lookup,
  verify_with_lookup,
};

// ── Column layout  (NUM_MEM_COLS = 12) ───────────────────────────────

const MEM_COL_ADDR_HI: usize = 0;
const MEM_COL_ADDR_LO: usize = 1;
const MEM_COL_VAL0: usize = 2;
const MEM_COL_IS_WRITE: usize = 10;
const MEM_COL_MULT: usize = 11;
pub const NUM_MEM_COLS: usize = 12;
const MEM_PUBLIC_VALUES_LEN: usize = 35;

// ── Log entry type ────────────────────────────────────────────────────

/// A single entry in the public write or read log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemLogEntry {
  pub rw_counter: u64,
  pub addr: u64,
  pub value: [u8; 32],
}

/// Convert a 32-byte value to 8 × `u32` in big-endian order.
pub(super) fn value_to_u32s(value: &[u8; 32]) -> [u32; 8] {
  let mut out = [0u32; 8];
  for (i, chunk) in value.chunks(4).enumerate() {
    out[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
  }
  out
}

pub fn split_mem_logs(
  claims: &[crate::transition::MemAccessClaim],
) -> (Vec<MemLogEntry>, Vec<MemLogEntry>) {
  let mut writes = Vec::new();
  let mut reads = Vec::new();
  for c in claims {
    let entry = MemLogEntry {
      rw_counter: c.rw_counter,
      addr: c.addr,
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

fn hash_mem_log(log: &[MemLogEntry]) -> [Val; 8] {
  let sponge = default_poseidon_sponge();
  let mut input: Vec<Val> = Vec::with_capacity(1 + log.len() * 12);
  input.push(Val::from_u32(log.len() as u32));
  for e in log {
    input.push(Val::from_u32((e.rw_counter >> 32) as u32));
    input.push(Val::from_u32(e.rw_counter as u32));
    input.push(Val::from_u32((e.addr >> 32) as u32));
    input.push(Val::from_u32(e.addr as u32));
    for chunk in e.value.chunks(4) {
      input.push(Val::from_u32(u32::from_be_bytes([
        chunk[0], chunk[1], chunk[2], chunk[3],
      ])));
    }
  }
  sponge.hash_iter(input)
}

fn hash_mem_set(set: &std::collections::HashMap<u64, [u8; 32]>) -> [Val; 8] {
  if set.is_empty() {
    return [Val::ZERO; 8];
  }
  let mut sorted: Vec<(u64, [u8; 32])> = set.iter().map(|(&a, &v)| (a, v)).collect();
  sorted.sort_by_key(|(a, _)| *a);

  let sponge = default_poseidon_sponge();
  let mut input: Vec<Val> = Vec::with_capacity(1 + sorted.len() * 10);
  input.push(Val::from_u32(sorted.len() as u32));
  for (addr, value) in &sorted {
    input.push(Val::from_u32((*addr >> 32) as u32));
    input.push(Val::from_u32(*addr as u32));
    for chunk in value.chunks(4) {
      input.push(Val::from_u32(u32::from_be_bytes([
        chunk[0], chunk[1], chunk[2], chunk[3],
      ])));
    }
  }
  sponge.hash_iter(input)
}

fn make_mem_public_values(
  write_log: &[MemLogEntry],
  read_log: &[MemLogEntry],
  write_set: &std::collections::HashMap<u64, [u8; 32]>,
  read_set: &std::collections::HashMap<u64, [u8; 32]>,
) -> Vec<Val> {
  let wlh = hash_mem_log(write_log);
  let rlh = hash_mem_log(read_log);
  let wsh = hash_mem_set(write_set);
  let rsh = hash_mem_set(read_set);
  let mut pv = Vec::with_capacity(MEM_PUBLIC_VALUES_LEN);
  pv.push(Val::from_u32(RECEIPT_BIND_TAG_MEM));
  pv.push(Val::from_u32(write_log.len() as u32));
  pv.extend_from_slice(&wlh);
  pv.push(Val::from_u32(read_log.len() as u32));
  pv.extend_from_slice(&rlh);
  pv.extend_from_slice(&wsh);
  pv.extend_from_slice(&rsh);
  pv
}

// ── Intra-batch consistency checker ──────────────────────────────────

#[allow(clippy::type_complexity)]
fn check_claims_and_build_sets(
  claims: &[crate::transition::MemAccessClaim],
) -> Result<
  (
    std::collections::HashMap<u64, [u8; 32]>,
    std::collections::HashMap<u64, [u8; 32]>,
  ),
  String,
> {
  use std::collections::HashMap;
  let mut write_set: HashMap<u64, [u8; 32]> = HashMap::new();
  let mut read_set: HashMap<u64, [u8; 32]> = HashMap::new();
  let mut last_write: HashMap<u64, [u8; 32]> = HashMap::new();

  for claim in claims {
    if claim.is_write {
      last_write.insert(claim.addr, claim.value);
      write_set.insert(claim.addr, claim.value);
    } else if let Some(&written) = last_write.get(&claim.addr) {
      if claim.value != written {
        return Err(format!(
          "read/write mismatch at addr=0x{:x}: read {:?}, last write {:?}",
          claim.addr, claim.value, written
        ));
      }
    } else {
      match read_set.entry(claim.addr) {
        std::collections::hash_map::Entry::Vacant(e) => {
          e.insert(claim.value);
        }
        std::collections::hash_map::Entry::Occupied(e) => {
          if *e.get() != claim.value {
            return Err(format!(
              "cross-batch read mismatch at addr=0x{:x}: {:?} vs {:?}",
              claim.addr,
              e.get(),
              claim.value
            ));
          }
        }
      }
    }
  }
  Ok((write_set, read_set))
}

#[allow(clippy::type_complexity)]
fn derive_sets_from_logs(
  write_log: &[MemLogEntry],
  read_log: &[MemLogEntry],
) -> Result<
  (
    std::collections::HashMap<u64, [u8; 32]>,
    std::collections::HashMap<u64, [u8; 32]>,
  ),
  String,
> {
  use std::collections::HashMap;
  let mut merged: Vec<(u64, bool, u64, [u8; 32])> = Vec::new();
  for e in write_log {
    merged.push((e.rw_counter, true, e.addr, e.value));
  }
  for e in read_log {
    merged.push((e.rw_counter, false, e.addr, e.value));
  }
  merged.sort_by_key(|(rw, _, _, _)| *rw);

  let mut write_set: HashMap<u64, [u8; 32]> = HashMap::new();
  let mut read_set: HashMap<u64, [u8; 32]> = HashMap::new();
  let mut last_write: HashMap<u64, [u8; 32]> = HashMap::new();

  for (_, is_write, addr, value) in merged {
    if is_write {
      last_write.insert(addr, value);
      write_set.insert(addr, value);
    } else if let Some(&written) = last_write.get(&addr) {
      if value != written {
        return Err(format!(
          "read/write mismatch at addr=0x{:x}: read {:?}, last write {:?}",
          addr, value, written
        ));
      }
    } else {
      match read_set.entry(addr) {
        std::collections::hash_map::Entry::Vacant(e) => {
          e.insert(value);
        }
        std::collections::hash_map::Entry::Occupied(e) => {
          if *e.get() != value {
            return Err(format!(
              "cross-batch read mismatch at addr=0x{:x}: {:?} vs {:?}",
              addr,
              e.get(),
              value
            ));
          }
        }
      }
    }
  }
  Ok((write_set, read_set))
}

// ── AIR ──────────────────────────────────────────────────────────────

pub struct MemoryConsistencyAir;

impl<F: p3_field::Field> p3_air::BaseAir<F> for MemoryConsistencyAir {
  fn width(&self) -> usize {
    NUM_MEM_COLS
  }
}

impl<AB: p3_air::AirBuilderWithPublicValues> p3_air::Air<AB> for MemoryConsistencyAir
where
  AB::F: p3_field::Field,
{
  fn eval(&self, builder: &mut AB) {
    let pis = builder.public_values();
    builder.assert_eq(pis[0].into(), AB::Expr::from_u32(RECEIPT_BIND_TAG_MEM));
    let main = builder.main();
    let local = main.row_slice(0).expect("empty memory trace");
    let local = &*local;
    let is_write = local[MEM_COL_IS_WRITE].clone();
    builder.assert_zero(is_write.clone().into() * (AB::Expr::ONE - is_write.into()));
  }
}

// ── Trace and LogUp helpers ───────────────────────────────────────────

fn fill_log_row(data: &mut [Val], base: usize, entry: &MemLogEntry, is_write: bool, mult: i32) {
  data[base + MEM_COL_ADDR_HI] = Val::from_u32((entry.addr >> 32) as u32);
  data[base + MEM_COL_ADDR_LO] = Val::from_u32(entry.addr as u32);
  let vals = value_to_u32s(&entry.value);
  for k in 0..8 {
    data[base + MEM_COL_VAL0 + k] = Val::from_u32(vals[k]);
  }
  data[base + MEM_COL_IS_WRITE] = Val::from_u32(is_write as u32);
  data[base + MEM_COL_MULT] = if mult >= 0 {
    Val::from_u32(mult as u32)
  } else {
    -Val::from_u32((-mult) as u32)
  };
}

fn build_mem_log_trace(
  write_log: &[MemLogEntry],
  read_log: &[MemLogEntry],
  write_set: &std::collections::HashMap<u64, [u8; 32]>,
) -> RowMajorMatrix<Val> {
  let mut intra_read_count: std::collections::HashMap<u64, i32> = Default::default();
  for e in read_log {
    if write_set.contains_key(&e.addr) {
      *intra_read_count.entry(e.addr).or_insert(0) += 1;
    }
  }
  let mut last_write_idx: std::collections::HashMap<u64, usize> = Default::default();
  for (i, e) in write_log.iter().enumerate() {
    last_write_idx.insert(e.addr, i);
  }

  let n_total = write_log.len() + read_log.len();
  let height = n_total.max(4).next_power_of_two();
  let mut data = vec![Val::ZERO; height * NUM_MEM_COLS];

  for (i, entry) in write_log.iter().enumerate() {
    let mult = if last_write_idx.get(&entry.addr) == Some(&i) {
      *intra_read_count.get(&entry.addr).unwrap_or(&0)
    } else {
      0
    };
    fill_log_row(&mut data, i * NUM_MEM_COLS, entry, true, mult);
  }
  for (i, entry) in read_log.iter().enumerate() {
    let mult = if write_set.contains_key(&entry.addr) {
      -1
    } else {
      0
    };
    fill_log_row(
      &mut data,
      (write_log.len() + i) * NUM_MEM_COLS,
      entry,
      false,
      mult,
    );
  }
  RowMajorMatrix::new(data, NUM_MEM_COLS)
}

fn make_mem_lookup() -> Lookup<Val> {
  let col =
    |c: usize| SymbolicExpression::Variable(SymbolicVariable::new(Entry::Main { offset: 0 }, c));
  Lookup::new(
    Kind::Local,
    vec![vec![
      col(MEM_COL_ADDR_HI),
      col(MEM_COL_ADDR_LO),
      col(MEM_COL_VAL0),
      col(MEM_COL_VAL0 + 1),
      col(MEM_COL_VAL0 + 2),
      col(MEM_COL_VAL0 + 3),
      col(MEM_COL_VAL0 + 4),
      col(MEM_COL_VAL0 + 5),
      col(MEM_COL_VAL0 + 6),
      col(MEM_COL_VAL0 + 7),
    ]],
    vec![col(MEM_COL_MULT)],
    vec![0],
  )
}

fn generate_mem_perm_trace(
  main_trace: &RowMajorMatrix<Val>,
  perm_challenges: &[Challenge],
) -> Option<RowMajorMatrix<Challenge>> {
  let gadget = LogUpGadget::new();
  let lookup = make_mem_lookup();
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

fn eval_mem_lookup(folder: &mut VerifierConstraintFolder<CircleStarkConfig>) {
  LogUpGadget::new().eval_local_lookup(folder, &make_mem_lookup());
}

// ── Public types and API ──────────────────────────────────────────────

#[derive(Clone)]
pub struct MemoryConsistencyProof {
  pub stark_proof: CircleStarkProof,
  pub write_set: std::collections::HashMap<u64, [u8; 32]>,
  pub read_set: std::collections::HashMap<u64, [u8; 32]>,
  pub write_log: Vec<MemLogEntry>,
  pub read_log: Vec<MemLogEntry>,
}

pub fn prove_memory_consistency(
  claims: &[crate::transition::MemAccessClaim],
) -> Result<MemoryConsistencyProof, String> {
  let (write_log, read_log) = split_mem_logs(claims);
  let (write_set, read_set) = check_claims_and_build_sets(claims)?;
  let public_values = make_mem_public_values(&write_log, &read_log, &write_set, &read_set);

  let trace = if write_log.is_empty() && read_log.is_empty() {
    RowMajorMatrix::new(vec![Val::ZERO; 4 * NUM_MEM_COLS], NUM_MEM_COLS)
  } else {
    build_mem_log_trace(&write_log, &read_log, &write_set)
  };

  let config = make_circle_config();
  let trace_for_perm = trace.clone();
  let stark_proof = prove_with_lookup(
    &config,
    &MemoryConsistencyAir,
    trace,
    &public_values,
    None,
    move |perm_challenges| generate_mem_perm_trace(&trace_for_perm, perm_challenges),
    2,
    2,
    |folder: &mut VerifierConstraintFolder<CircleStarkConfig>| eval_mem_lookup(folder),
  );

  Ok(MemoryConsistencyProof {
    stark_proof,
    write_set,
    read_set,
    write_log,
    read_log,
  })
}

pub fn verify_memory_consistency(proof: &MemoryConsistencyProof) -> bool {
  let (derived_write_set, derived_read_set) =
    match derive_sets_from_logs(&proof.write_log, &proof.read_log) {
      Ok(s) => s,
      Err(_) => return false,
    };
  if derived_write_set != proof.write_set || derived_read_set != proof.read_set {
    return false;
  }
  let public_values = make_mem_public_values(
    &proof.write_log,
    &proof.read_log,
    &proof.write_set,
    &proof.read_set,
  );
  let config = make_circle_config();
  verify_with_lookup(
    &config,
    &MemoryConsistencyAir,
    &proof.stark_proof,
    &public_values,
    None,
    2,
    |folder: &mut VerifierConstraintFolder<CircleStarkConfig>| eval_mem_lookup(folder),
  )
  .is_ok()
}

// ── Binary-tree aggregation ───────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AggregatedMemoryProof {
  pub write_set: std::collections::HashMap<u64, [u8; 32]>,
  pub read_set: std::collections::HashMap<u64, [u8; 32]>,
}

pub fn aggregate_memory_proofs(
  left: &MemoryConsistencyProof,
  right: &MemoryConsistencyProof,
) -> Result<AggregatedMemoryProof, String> {
  if !verify_memory_consistency(left) {
    return Err("left child proof failed verification".to_string());
  }
  if !verify_memory_consistency(right) {
    return Err("right child proof failed verification".to_string());
  }
  merge_sets(
    &left.write_set,
    &left.read_set,
    &right.write_set,
    &right.read_set,
  )
}

fn merge_sets(
  left_ws: &std::collections::HashMap<u64, [u8; 32]>,
  left_rs: &std::collections::HashMap<u64, [u8; 32]>,
  right_ws: &std::collections::HashMap<u64, [u8; 32]>,
  right_rs: &std::collections::HashMap<u64, [u8; 32]>,
) -> Result<AggregatedMemoryProof, String> {
  for (addr, &r_val) in right_rs {
    if let Some(&l_val) = left_ws.get(addr)
      && r_val != l_val {
        return Err(format!(
          "read/write set mismatch at addr=0x{addr:x}: right read {r_val:?}, left wrote {l_val:?}"
        ));
      }
  }
  let mut write_set = left_ws.clone();
  for (&addr, &val) in right_ws {
    write_set.insert(addr, val);
  }
  let mut read_set = left_rs.clone();
  for (&addr, &val) in right_rs {
    if !left_ws.contains_key(&addr) {
      read_set.entry(addr).or_insert(val);
    }
  }
  Ok(AggregatedMemoryProof {
    write_set,
    read_set,
  })
}

pub fn aggregate_proofs_tree(
  proofs: &[MemoryConsistencyProof],
) -> Result<AggregatedMemoryProof, String> {
  if proofs.is_empty() {
    return Err("aggregate_proofs_tree: empty proof slice".to_string());
  }
  let mut nodes: Vec<AggregatedMemoryProof> = proofs
    .iter()
    .enumerate()
    .map(|(i, p)| {
      if !verify_memory_consistency(p) {
        Err(format!("leaf proof {i} failed verification"))
      } else {
        Ok(AggregatedMemoryProof {
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
        let merged = merge_sets(
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
