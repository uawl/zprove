//! Stack consistency proof: push/pop multiset argument.

use super::types::{
  Challenge, CircleStarkConfig, CircleStarkProof, Val, default_poseidon_sponge, make_circle_config,
};

pub const RECEIPT_BIND_TAG_STACK_CONSISTENCY: u32 = 6;
use super::memory::value_to_u32s;

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

// ── Column layout  (NUM_STACK_COLS = 13) ─────────────────────────────

const STACK_COL_RW_HI: usize = 0;
const STACK_COL_RW_LO: usize = 1;
const STACK_COL_DEPTH: usize = 2;
const STACK_COL_VAL0: usize = 3;
const STACK_COL_IS_PUSH: usize = 11;
const STACK_COL_MULT: usize = 12;
pub const NUM_STACK_COLS: usize = 13;
const STACK_PUBLIC_VALUES_LEN: usize = 19;

// ── Log types ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StackLogEntry {
  pub rw_counter: u64,
  pub depth_after: u32,
  pub value: [u8; 32],
}

pub fn split_stack_logs(
  claims: &[crate::transition::StackAccessClaim],
) -> (Vec<StackLogEntry>, Vec<StackLogEntry>) {
  let mut pushes = Vec::new();
  let mut pops = Vec::new();
  for c in claims {
    let entry = StackLogEntry {
      rw_counter: c.rw_counter,
      depth_after: c.depth_after as u32,
      value: c.value,
    };
    if c.is_push {
      pushes.push(entry);
    } else {
      pops.push(entry);
    }
  }
  (pushes, pops)
}

// ── Hash helpers ──────────────────────────────────────────────────────

fn hash_stack_log(log: &[StackLogEntry]) -> [Val; 8] {
  let sponge = default_poseidon_sponge();
  let mut input: Vec<Val> = Vec::with_capacity(1 + log.len() * 11);
  input.push(Val::from_u32(log.len() as u32));
  for e in log {
    input.push(Val::from_u32((e.rw_counter >> 32) as u32));
    input.push(Val::from_u32(e.rw_counter as u32));
    input.push(Val::from_u32(e.depth_after));
    for chunk in e.value.chunks(4) {
      input.push(Val::from_u32(u32::from_be_bytes([
        chunk[0], chunk[1], chunk[2], chunk[3],
      ])));
    }
  }
  sponge.hash_iter(input)
}

fn make_stack_public_values(push_log: &[StackLogEntry], pop_log: &[StackLogEntry]) -> Vec<Val> {
  let plh = hash_stack_log(push_log);
  let oph = hash_stack_log(pop_log);
  let mut pv = Vec::with_capacity(STACK_PUBLIC_VALUES_LEN);
  pv.push(Val::from_u32(RECEIPT_BIND_TAG_STACK_CONSISTENCY));
  pv.push(Val::from_u32(push_log.len() as u32));
  pv.extend_from_slice(&plh);
  pv.push(Val::from_u32(pop_log.len() as u32));
  pv.extend_from_slice(&oph);
  pv
}

// ── Intra-batch consistency checker ───────────────────────────────────

fn check_stack_claims(claims: &[crate::transition::StackAccessClaim]) -> Result<(), String> {
  let mut top_value: std::collections::HashMap<u32, [u8; 32]> = std::collections::HashMap::new();
  for claim in claims {
    let d = claim.depth_after as u32;
    if claim.is_push {
      top_value.insert(d, claim.value);
    } else {
      let popped_depth = d + 1;
      if let Some(&pushed_val) = top_value.get(&popped_depth) {
        if pushed_val != claim.value {
          return Err(format!(
            "stack pop/push mismatch at rw_counter={} depth={}: popped {:?} but last push was {:?}",
            claim.rw_counter, popped_depth, claim.value, pushed_val
          ));
        }
        top_value.remove(&popped_depth);
      }
    }
  }
  Ok(())
}

// ── AIR ───────────────────────────────────────────────────────────────

pub struct StackConsistencyAir;

impl<F: p3_field::Field> p3_air::BaseAir<F> for StackConsistencyAir {
  fn width(&self) -> usize {
    NUM_STACK_COLS
  }
}

impl<AB: p3_air::AirBuilderWithPublicValues> p3_air::Air<AB> for StackConsistencyAir
where
  AB::F: p3_field::Field,
{
  fn eval(&self, builder: &mut AB) {
    let pis = builder.public_values();
    builder.assert_eq(
      pis[0].into(),
      AB::Expr::from_u32(RECEIPT_BIND_TAG_STACK_CONSISTENCY),
    );
    let main = builder.main();
    let local = main.row_slice(0).expect("empty stack trace");
    let local = &*local;
    let is_push = local[STACK_COL_IS_PUSH].clone();
    builder.assert_zero(is_push.clone().into() * (AB::Expr::ONE - is_push.into()));
  }
}

// ── Trace builder ─────────────────────────────────────────────────────

fn fill_stack_log_row(data: &mut [Val], base: usize, entry: &StackLogEntry, is_push: bool) {
  data[base + STACK_COL_RW_HI] = Val::from_u32((entry.rw_counter >> 32) as u32);
  data[base + STACK_COL_RW_LO] = Val::from_u32(entry.rw_counter as u32);
  data[base + STACK_COL_DEPTH] = Val::from_u32(entry.depth_after);
  let vals = value_to_u32s(&entry.value);
  for k in 0..8 {
    data[base + STACK_COL_VAL0 + k] = Val::from_u32(vals[k]);
  }
  data[base + STACK_COL_IS_PUSH] = Val::from_u32(is_push as u32);
}

fn build_stack_log_trace(
  push_log: &[StackLogEntry],
  pop_log: &[StackLogEntry],
  intra_pop_count_per_push: &[((u32, [u8; 32]), i32)], // sorted by key
  intra_pop_rw: &[u64],                                  // sorted
) -> RowMajorMatrix<Val> {
  let n_total = push_log.len() + pop_log.len();
  let height = n_total.max(4).next_power_of_two();
  let mut data = vec![Val::ZERO; height * NUM_STACK_COLS];

  // ── last push index per (depth_after, value) key (sorted array) ──────────
  let mut push_key_idx: Vec<((u32, [u8; 32]), usize)> = push_log
    .iter()
    .enumerate()
    .map(|(i, e)| ((e.depth_after, e.value), i))
    .collect();
  push_key_idx.sort_unstable_by(|a, b| a.0.cmp(&b.0));
  let last_push: Vec<((u32, [u8; 32]), usize)> = {
    let mut v: Vec<((u32, [u8; 32]), usize)> = Vec::new();
    for (key, idx) in push_key_idx {
      match v.last_mut() {
        Some((k, i)) if *k == key => *i = idx,
        _ => v.push((key, idx)),
      }
    }
    v
  };

  for (i, entry) in push_log.iter().enumerate() {
    fill_stack_log_row(&mut data, i * NUM_STACK_COLS, entry, true);
    let key = (entry.depth_after, entry.value);
    let mult = match last_push.binary_search_by(|(k, _)| k.cmp(&key)) {
      Ok(pos) if last_push[pos].1 == i => {
        intra_pop_count_per_push
          .binary_search_by(|(k, _)| k.cmp(&key))
          .map_or(0, |rp| intra_pop_count_per_push[rp].1)
      }
      _ => 0,
    };
    let base = i * NUM_STACK_COLS;
    data[base + STACK_COL_MULT] = if mult >= 0 {
      Val::from_u32(mult as u32)
    } else {
      -Val::from_u32((-mult) as u32)
    };
  }

  for (i, entry) in pop_log.iter().enumerate() {
    let base = (push_log.len() + i) * NUM_STACK_COLS;
    fill_stack_log_row(&mut data, base, entry, false);
    data[base + STACK_COL_MULT] =
      if intra_pop_rw.binary_search(&entry.rw_counter).is_ok() {
        -Val::ONE
      } else {
        Val::ZERO
      };
  }
  RowMajorMatrix::new(data, NUM_STACK_COLS)
}

// ── LogUp argument ────────────────────────────────────────────────────

fn make_stack_lookup() -> Lookup<Val> {
  let col =
    |c: usize| SymbolicExpression::Variable(SymbolicVariable::new(Entry::Main { offset: 0 }, c));
  let depth_slot =
    col(STACK_COL_DEPTH) + (SymbolicExpression::Constant(Val::ONE) - col(STACK_COL_IS_PUSH));
  Lookup::new(
    Kind::Local,
    vec![vec![
      depth_slot,
      col(STACK_COL_VAL0),
      col(STACK_COL_VAL0 + 1),
      col(STACK_COL_VAL0 + 2),
      col(STACK_COL_VAL0 + 3),
      col(STACK_COL_VAL0 + 4),
      col(STACK_COL_VAL0 + 5),
      col(STACK_COL_VAL0 + 6),
      col(STACK_COL_VAL0 + 7),
    ]],
    vec![col(STACK_COL_MULT)],
    vec![0],
  )
}

fn generate_stack_perm_trace(
  main_trace: &RowMajorMatrix<Val>,
  perm_challenges: &[Challenge],
) -> Option<RowMajorMatrix<Challenge>> {
  let gadget = LogUpGadget::new();
  let lookup = make_stack_lookup();
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

fn eval_stack_lookup(folder: &mut VerifierConstraintFolder<CircleStarkConfig>) {
  LogUpGadget::new().eval_local_lookup(folder, &make_stack_lookup());
}

fn build_stack_logup_aux(
  push_log: &[StackLogEntry],
  pop_log: &[StackLogEntry],
) -> (
  Vec<((u32, [u8; 32]), i32)>, // intra-pop count per push key, sorted by key
  Vec<u64>,                    // rw_counters of intra-batch pops, sorted
) {
  // Sorted, deduplicated push keys for O(log n) intra-batch pop detection.
  let mut push_keys: Vec<(u32, [u8; 32])> = push_log
    .iter()
    .map(|e| (e.depth_after, e.value))
    .collect();
  push_keys.sort_unstable();
  push_keys.dedup();

  // Collect the push-perspective key and rw_counter of each intra-batch pop.
  let mut intra_key_list: Vec<(u32, [u8; 32])> = Vec::new();
  let mut intra_pop_rw: Vec<u64> = Vec::new();
  for e in pop_log {
    let key = (e.depth_after + 1, e.value);
    if push_keys.binary_search(&key).is_ok() {
      intra_key_list.push(key);
      intra_pop_rw.push(e.rw_counter);
    }
  }

  // Aggregate count per key.
  intra_key_list.sort_unstable();
  let intra_pop_count: Vec<((u32, [u8; 32]), i32)> = {
    let mut v: Vec<((u32, [u8; 32]), i32)> = Vec::new();
    for key in intra_key_list {
      match v.last_mut() {
        Some((k, c)) if *k == key => *c += 1,
        _ => v.push((key, 1)),
      }
    }
    v
  };

  // Sort rw_counters for O(log n) membership test in build_stack_log_trace.
  intra_pop_rw.sort_unstable();
  (intra_pop_count, intra_pop_rw)
}

// ── Public API ─────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct StackConsistencyProof {
  pub stark_proof: CircleStarkProof,
  pub push_log: Vec<StackLogEntry>,
  pub pop_log: Vec<StackLogEntry>,
}

pub fn prove_stack_consistency(
  claims: &[crate::transition::StackAccessClaim],
) -> Result<StackConsistencyProof, String> {
  let (push_log, pop_log) = split_stack_logs(claims);
  check_stack_claims(claims)?;
  let public_values = make_stack_public_values(&push_log, &pop_log);
  let (intra_pop_count, intra_pop_set) = build_stack_logup_aux(&push_log, &pop_log);

  let trace = if push_log.is_empty() && pop_log.is_empty() {
    RowMajorMatrix::new(vec![Val::ZERO; 4 * NUM_STACK_COLS], NUM_STACK_COLS)
  } else {
    build_stack_log_trace(&push_log, &pop_log, &intra_pop_count, &intra_pop_set)
  };

  let config = make_circle_config();
  let hints =
    air_cache::get_or_compute(&StackConsistencyAir, 0, public_values.len(), 0);
  let trace_for_perm = trace.clone();
  let stark_proof = prove_with_lookup_hinted(
    &config,
    &StackConsistencyAir,
    trace,
    &public_values,
    None,
    move |perm_challenges| generate_stack_perm_trace(&trace_for_perm, perm_challenges),
    2,
    2,
    |folder: &mut VerifierConstraintFolder<CircleStarkConfig>| eval_stack_lookup(folder),
    hints,
  );
  Ok(StackConsistencyProof {
    stark_proof,
    push_log,
    pop_log,
  })
}

pub fn verify_stack_consistency(proof: &StackConsistencyProof) -> bool {
  let public_values = make_stack_public_values(&proof.push_log, &proof.pop_log);

  let mut merged: Vec<(u64, bool, u32, [u8; 32])> = Vec::new();
  for e in &proof.push_log {
    merged.push((e.rw_counter, true, e.depth_after, e.value));
  }
  for e in &proof.pop_log {
    merged.push((e.rw_counter, false, e.depth_after, e.value));
  }
  merged.sort_by_key(|(rw, _, _, _)| *rw);

  let reconstructed: Vec<crate::transition::StackAccessClaim> = merged
    .into_iter()
    .map(
      |(rw_counter, is_push, depth_after, value)| crate::transition::StackAccessClaim {
        rw_counter,
        depth_after: depth_after as usize,
        is_push,
        value,
      },
    )
    .collect();

  if check_stack_claims(&reconstructed).is_err() {
    return false;
  }

  let config = make_circle_config();
  verify_with_lookup(
    &config,
    &StackConsistencyAir,
    &proof.stark_proof,
    &public_values,
    None,
    2,
    |folder: &mut VerifierConstraintFolder<CircleStarkConfig>| eval_stack_lookup(folder),
  )
  .is_ok()
}
