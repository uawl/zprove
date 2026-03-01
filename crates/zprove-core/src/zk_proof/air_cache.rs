//! Global cache for [`AirConstraintHints`] keyed by AIR type.
//!
//! Symbolic-constraint evaluation (constraint count + log_num_quotient_chunks)
//! is deterministic for a given AIR struct.  Computing it on every `prove()`
//! call wastes ~15-70 ms per invocation.  This module stores the result in a
//! process-wide [`OnceLock`]-protected [`HashMap`] keyed by
//! `(TypeId, preprocessed_width, num_public_values, is_zk)`.
//!
//! # Usage
//!
//! Hot modules (stack_rw, memory, storage, …) should call
//! [`get_or_compute`] instead of naked `prove_with_lookup` so that the first
//! call pays the symbolic-eval cost and all subsequent calls are a map lookup.
//!
//! [`warm_up`] pre-populates every well-known AIR type; call it once at
//! benchmark / application startup.

use std::any::TypeId;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use p3_air::Air;
use p3_uni_stark::{AirConstraintHints, SymbolicAirBuilder, compute_air_constraint_hints};

use crate::zk_proof::types::Val;

// ── cache storage ────────────────────────────────────────────────────────────

type CacheKey = (TypeId, usize, usize, usize); // (type, pp_width, num_pv, is_zk)

static CACHE: OnceLock<Mutex<HashMap<CacheKey, AirConstraintHints>>> = OnceLock::new();

fn cache() -> &'static Mutex<HashMap<CacheKey, AirConstraintHints>> {
  CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

// ── public API ───────────────────────────────────────────────────────────────

/// Return cached [`AirConstraintHints`] for the given AIR, computing them on
/// the first call and caching the result for all future calls.
pub fn get_or_compute<A>(
  air: &A,
  preprocessed_width: usize,
  num_public_values: usize,
  is_zk: usize,
) -> AirConstraintHints
where
  A: Air<SymbolicAirBuilder<Val>> + 'static,
{
  let key: CacheKey = (TypeId::of::<A>(), preprocessed_width, num_public_values, is_zk);

  // Fast path: already cached.
  {
    let guard = cache().lock().expect("air_cache mutex poisoned");
    if let Some(&h) = guard.get(&key) {
      return h;
    }
  }

  // Slow path: compute, then insert.
  let hints =
    compute_air_constraint_hints::<Val, A>(air, preprocessed_width, num_public_values, is_zk);
  {
    let mut guard = cache().lock().expect("air_cache mutex poisoned");
    guard.entry(key).or_insert(hints);
  }
  hints
}

/// Pre-populate the cache for every well-known AIR type used in this crate.
///
/// Call this once at application / benchmark startup so that the first proof
/// of each type does not pay the symbolic-evaluation cost.
pub fn warm_up() {
  use crate::byte_table::ByteTableAir;
  use crate::zk_proof::{
    BatchLutKernelAirWithPrep, KeccakConsistencyAir, LutKernelAir, MemoryConsistencyAir,
    StackConsistencyAir, StackIrAir, StackRwAir, StorageConsistencyAir,
  };
  use crate::zk_proof::preprocessed::NUM_BATCH_PREP_COLS;

  // is_zk = 0 (no ZK padding) — matches the default config used everywhere.
  const IS_ZK: usize = 0;

  // num_public_values from each module's *_PUBLIC_VALUES_LEN constant,
  // preprocessed_width = 0 for all (no preprocessed trace for these AIRs).
  get_or_compute(&StackRwAir, 0, 19, IS_ZK);         // STACK_RW_PUBLIC_VALUES_LEN
  get_or_compute(&MemoryConsistencyAir, 0, 35, IS_ZK);  // MEM_PUBLIC_VALUES_LEN
  get_or_compute(&StorageConsistencyAir, 0, 35, IS_ZK); // STOR_PUBLIC_VALUES_LEN
  get_or_compute(&KeccakConsistencyAir, 0, 11, IS_ZK);  // KECCAK_PUBLIC_VALUES_LEN
  get_or_compute(&StackConsistencyAir, 0, 19, IS_ZK);   // STACK_PUBLIC_VALUES_LEN
  get_or_compute(&StackIrAir, 0, 10, IS_ZK);           // RECEIPT_BIND_PUBLIC_VALUES_LEN

  // LUT-related AIRs.
  // ByteTableAir: prove_with_lookup(&[], …) → pv_len = 0, pp_width = 0.
  get_or_compute(&ByteTableAir, 0, 0, IS_ZK);
  // LutKernelAir: prove(…, pv) where pv = RECEIPT_BIND_PUBLIC_VALUES_LEN = 10.
  get_or_compute(&LutKernelAir, 0, 10, IS_ZK);
  // BatchLutKernelAirWithPrep: prove_with_preprocessed(pp_width=NUM_BATCH_PREP_COLS, pv_len=10).
  get_or_compute(&BatchLutKernelAirWithPrep::for_verify(), NUM_BATCH_PREP_COLS, 10, IS_ZK);
  // StackIrBatchAirWithPrep: same preprocessed width and public-value count.
  use crate::zk_proof::stack_ir::StackIrBatchAirWithPrep;
  get_or_compute(&StackIrBatchAirWithPrep::for_verify(), NUM_BATCH_PREP_COLS, 10, IS_ZK);
}
