//! Keccak256 consistency proof (preimage membership argument).

use super::memory::MemLogEntry;
use super::types::{CircleStarkProof, Val, default_poseidon_sponge, make_circle_config};

use p3_field::PrimeCharacteristicRing;
use p3_keccak::Keccak256Hash;
use p3_matrix::dense::RowMajorMatrix;
use p3_symmetric::CryptographicHasher;

// ── Column layout  (NUM_KECCAK_COLS = 12) ─────────────────────────────
//  0     offset_hi
//  1     offset_lo
//  2     size_hi
//  3     size_lo
//  4..11 hash0..7  (8 × u32, big-endian)

const KECCAK_COL_OFFSET_HI: usize = 0;
const KECCAK_COL_OFFSET_LO: usize = 1;
const KECCAK_COL_SIZE_HI: usize = 2;
const KECCAK_COL_SIZE_LO: usize = 3;
const KECCAK_COL_HASH0: usize = 4;
/// Total number of main-trace columns for keccak consistency.
pub const NUM_KECCAK_COLS: usize = 12; // 4 (offset/size) + 8 (hash)

pub const RECEIPT_BIND_TAG_KECCAK: u32 = 7;
const KECCAK_PUBLIC_VALUES_LEN: usize = 11;

// ── Log type ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeccakLogEntry {
  pub offset: u64,
  pub size: u64,
  pub input_bytes: Vec<u8>,
  pub output_hash: [u8; 32],
}

// ── Hash helpers ──────────────────────────────────────────────────────

fn hash_keccak_log(log: &[KeccakLogEntry]) -> [Val; 8] {
  let sponge = default_poseidon_sponge();
  let mut input: Vec<Val> = Vec::with_capacity(1 + log.len() * 14);
  input.push(Val::from_u32(log.len() as u32));
  for e in log {
    input.push(Val::from_u32((e.offset >> 32) as u32));
    input.push(Val::from_u32(e.offset as u32));
    input.push(Val::from_u32((e.size >> 32) as u32));
    input.push(Val::from_u32(e.size as u32));
    // Commit input_bytes into the hash (packed 4 bytes per field element).
    input.push(Val::from_u32(e.input_bytes.len() as u32));
    for chunk in e.input_bytes.chunks(4) {
      let mut buf = [0u8; 4];
      buf[..chunk.len()].copy_from_slice(chunk);
      input.push(Val::from_u32(u32::from_be_bytes(buf)));
    }
    for chunk in e.output_hash.chunks(4) {
      input.push(Val::from_u32(u32::from_be_bytes([
        chunk[0], chunk[1], chunk[2], chunk[3],
      ])));
    }
  }
  sponge.hash_iter(input)
}

fn make_keccak_public_values(log: &[KeccakLogEntry]) -> Vec<Val> {
  let h = hash_keccak_log(log);
  let mut pv = Vec::with_capacity(KECCAK_PUBLIC_VALUES_LEN);
  pv.push(Val::from_u32(RECEIPT_BIND_TAG_KECCAK));
  pv.push(Val::from_u32(log.len() as u32));
  pv.extend_from_slice(&h);
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
    let pis = builder.public_values();
    builder.assert_eq(pis[0].into(), AB::Expr::from_u32(RECEIPT_BIND_TAG_KECCAK));
  }
}

// ── Trace builder ─────────────────────────────────────────────────────

fn fill_keccak_log_row(data: &mut [Val], base: usize, entry: &KeccakLogEntry) {
  data[base + KECCAK_COL_OFFSET_HI] = Val::from_u32((entry.offset >> 32) as u32);
  data[base + KECCAK_COL_OFFSET_LO] = Val::from_u32(entry.offset as u32);
  data[base + KECCAK_COL_SIZE_HI] = Val::from_u32((entry.size >> 32) as u32);
  data[base + KECCAK_COL_SIZE_LO] = Val::from_u32(entry.size as u32);
  for (k, chunk) in entry.output_hash.chunks(4).enumerate() {
    data[base + KECCAK_COL_HASH0 + k] =
      Val::from_u32(u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
  }
}

fn build_keccak_log_trace(log: &[KeccakLogEntry]) -> RowMajorMatrix<Val> {
  let height = log.len().max(4).next_power_of_two();
  let mut data = vec![Val::ZERO; height * NUM_KECCAK_COLS];
  for (i, entry) in log.iter().enumerate() {
    fill_keccak_log_row(&mut data, i * NUM_KECCAK_COLS, entry);
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
/// 1. Verifies `keccak256(claim.input_bytes) == claim.output_hash` for every claim.
/// 2. Builds the claim log and STARK proof.
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
    log.push(KeccakLogEntry {
      offset: claim.offset,
      size: claim.size,
      input_bytes: claim.input_bytes.clone(),
      output_hash: claim.output_hash,
    });
  }

  let public_values = make_keccak_public_values(&log);

  let trace = if log.is_empty() {
    RowMajorMatrix::new(vec![Val::ZERO; 4 * NUM_KECCAK_COLS], NUM_KECCAK_COLS)
  } else {
    build_keccak_log_trace(&log)
  };

  let config = make_circle_config();
  let stark_proof = p3_uni_stark::prove(&config, &KeccakConsistencyAir, trace, &public_values);

  Ok(KeccakConsistencyProof { stark_proof, log })
}

/// Verify a single [`KeccakConsistencyProof`].
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

/// Cross-check that each keccak claim's `input_bytes` match the bytes in the
/// memory log at the corresponding addresses.
pub fn validate_keccak_memory_cross_check(
  keccak_log: &[KeccakLogEntry],
  mem_write_log: &[MemLogEntry],
  mem_read_log: &[MemLogEntry],
) -> bool {
  if keccak_log.is_empty() {
    return true;
  }

  // Build addr → last-known value map.
  // read_log base values are installed first; write_log overwrites to give
  // the final written value per address.
  let mut mem_map: std::collections::HashMap<u64, [u8; 32]> = std::collections::HashMap::new();
  for e in mem_read_log {
    mem_map.entry(e.addr).or_insert(e.value);
  }
  for e in mem_write_log {
    mem_map.insert(e.addr, e.value);
  }

  for entry in keccak_log {
    let offset = entry.offset;
    let size = entry.size as usize;

    if entry.input_bytes.len() != size {
      return false;
    }
    if size == 0 {
      continue;
    }

    // Iterate over the 32-byte-aligned words that cover [offset, offset+size).
    let first_word = (offset / 32) * 32;
    let last_word = ((offset + size as u64 - 1) / 32) * 32;
    let mut word = first_word;
    while word <= last_word {
      // Overlap of this word [word, word+32) with [offset, offset+size).
      let abs_start = offset.max(word);
      let abs_end = (offset + size as u64).min(word + 32);

      let val_start = (abs_start - word) as usize;
      let val_end = (abs_end - word) as usize;
      let inp_start = (abs_start - offset) as usize;
      let inp_end = (abs_end - offset) as usize;

      let mem_value = match mem_map.get(&word) {
        Some(v) => v,
        None => return false, // keccak-accessed word not in memory log
      };

      if mem_value[val_start..val_end] != entry.input_bytes[inp_start..inp_end] {
        return false;
      }

      word += 32;
    }
  }

  true
}
