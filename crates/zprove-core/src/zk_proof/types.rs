//! Type aliases, STARK configuration, shared constants, and WFF serialization helpers.

use crate::semantic_proof::{Term, WFF};

use p3_challenger::{DuplexChallenger, HashChallenger, SerializingChallenger32};
use p3_circle::CirclePcs;
use p3_commit::ExtensionMmcs;
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_fri::FriParameters;
use p3_keccak::Keccak256Hash;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_mersenne_31::{Mersenne31, Poseidon2Mersenne31};
use p3_symmetric::{
  CompressionFunctionFromHasher, CryptographicHasher, PaddingFreeSponge, SerializingHasher,
  TruncatedPermutation,
};
use p3_uni_stark::StarkConfig;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use std::sync::OnceLock;

// ============================================================
// Type aliases for Circle STARK over M31
// ============================================================

pub type Val = Mersenne31;
pub type Challenge = BinomialExtensionField<Val, 3>;

// ---- Poseidon2 over Mersenne31 — used as the default hash/commitment backend ----
pub type P2Perm = Poseidon2Mersenne31<16>;
pub type P2Hash = PaddingFreeSponge<P2Perm, 16, 8, 8>;
pub type P2Compress = TruncatedPermutation<P2Perm, 2, 8, 16>;
pub type P2ValMmcs = MerkleTreeMmcs<Val, Val, P2Hash, P2Compress, 8>;
pub type P2ChallengeMmcs = ExtensionMmcs<Val, Challenge, P2ValMmcs>;
pub type P2Challenger = DuplexChallenger<Val, P2Perm, 16, 8>;

pub type Pcs = CirclePcs<Val, P2ValMmcs, P2ChallengeMmcs>;
pub type CircleStarkConfig = StarkConfig<Pcs, Challenge, P2Challenger>;

pub type CircleStarkProof = p3_uni_stark::Proof<CircleStarkConfig>;
pub type CircleStarkVerifyResult =
  Result<(), p3_uni_stark::VerificationError<p3_uni_stark::PcsError<CircleStarkConfig>>>;

// Keccak types kept around in case needed elsewhere (not used by default config)
#[allow(dead_code)]
type KeccakByteHash = Keccak256Hash;
#[allow(dead_code)]
type KeccakFieldHash = SerializingHasher<KeccakByteHash>;
#[allow(dead_code)]
type KeccakCompress = CompressionFunctionFromHasher<KeccakByteHash, 2, 32>;
#[allow(dead_code)]
type KeccakChallenger = SerializingChallenger32<Val, HashChallenger<u8, KeccakByteHash, 32>>;

pub const RECEIPT_BIND_TAG_STACK: u32 = 1;
pub const RECEIPT_BIND_TAG_LUT: u32 = 2;
pub const RECEIPT_BIND_TAG_WFF: u32 = 3;
pub(super) const RECEIPT_BIND_PUBLIC_VALUES_LEN: usize = 10;

pub(super) fn default_receipt_bind_public_values() -> Vec<Val> {
  vec![Val::from_u32(0); RECEIPT_BIND_PUBLIC_VALUES_LEN]
}

pub(super) fn default_receipt_bind_public_values_for_tag(tag: u32) -> Vec<Val> {
  let mut values = default_receipt_bind_public_values();
  values[0] = Val::from_u32(tag);
  values
}

/// Public values for the scaffold StackIR STARK: tag = `RECEIPT_BIND_TAG_STACK`,
/// remaining positions are zero.  Use this whenever calling
/// [`prove_stack_ir_with_prep`] / [`prove_stack_ir_scaffold_stark`] outside of
/// the full prove pipeline which supplies a real Poseidon hash.
pub fn stack_ir_scaffold_public_values() -> Vec<Val> {
  default_receipt_bind_public_values_for_tag(RECEIPT_BIND_TAG_STACK)
}

/// Lazily-initialised Poseidon2 sponge shared across all hash helpers.
///
/// `Poseidon2Mersenne31` round constants are derived deterministically from a
/// fixed seed, so building the permutation once and reusing it is equivalent
/// to rebuilding it on every call — with no observable difference to callers.
pub(super) fn default_poseidon_sponge() -> &'static PaddingFreeSponge<P2Perm, 16, 8, 8> {
  static SPONGE: OnceLock<PaddingFreeSponge<P2Perm, 16, 8, 8>> = OnceLock::new();
  SPONGE.get_or_init(|| {
    let mut rng = SmallRng::seed_from_u64(0xC0DEC0DE_u64);
    let poseidon = Poseidon2Mersenne31::<16>::new_from_rng_128(&mut rng);
    PaddingFreeSponge::<_, 16, 8, 8>::new(poseidon)
  })
}

/// Compute the **tag-independent** Poseidon digest of `(opcode, len, wff_bytes)`.
///
/// Intentionally omits the STARK tag so the same 8-element digest can be stored
/// once in the shared preprocessed matrix and bound by both the StackIR and LUT
/// AIRs (which differ only in `pis[0]`, the tag).
pub fn compute_wff_opcode_digest(opcode: u8, expected_wff: &WFF) -> [Val; 8] {
  let sponge = default_poseidon_sponge();
  let wff_bytes = serialize_wff_bytes(expected_wff);
  let mut input = Vec::with_capacity(2 + wff_bytes.len());
  input.push(Val::from_u32(opcode as u32));
  input.push(Val::from_u32(wff_bytes.len() as u32));
  input.extend(wff_bytes.into_iter().map(Val::from_u8));
  sponge.hash_iter(input)
}

/// Build the public-values vector for a receipt-binding STARK.
///
/// Layout: `[tag, opcode, digest[0..7]]` where `digest = compute_wff_opcode_digest(opcode, wff)`.
///
/// The digest is **tag-independent** so both the StackIR and LUT STARKs can share
/// the same preprocessed matrix commitment (which stores the digest once) while
/// each independently enforces `pis[0]` against their own tag.
pub fn make_receipt_binding_public_values(tag: u32, opcode: u8, expected_wff: &WFF) -> Vec<Val> {
  let digest = compute_wff_opcode_digest(opcode, expected_wff);
  let mut public_values = Vec::with_capacity(RECEIPT_BIND_PUBLIC_VALUES_LEN);
  public_values.push(Val::from_u32(tag));
  public_values.push(Val::from_u32(opcode as u32));
  public_values.extend_from_slice(&digest);
  public_values
}

// ============================================================
// Config builder
// ============================================================

/// Build a Circle STARK configuration over M31 with Poseidon2 hashing.
pub fn make_circle_config() -> CircleStarkConfig {
  make_circle_config_with_params(40, 8, 0)
}

/// Tunable Circle STARK config (Poseidon2 hash backend).
///
/// - `num_queries`: FRI query count. Soundness ≈ num_queries × log_blowup bits. Default 40.
/// - `query_pow_bits`: grinding bits for query phase (2^n hashes). Default 8. Set 0 to disable.
/// - `log_final_poly_len`: log2 of FRI final polynomial degree limit. Default 0.
pub fn make_circle_config_with_params(
  num_queries: usize,
  query_pow_bits: usize,
  log_final_poly_len: usize,
) -> CircleStarkConfig {
  let mut rng = SmallRng::seed_from_u64(0x5EED_C0DE_u64);
  let perm = P2Perm::new_from_rng_128(&mut rng);

  let hash = P2Hash::new(perm.clone());
  let compress = P2Compress::new(perm.clone());
  let val_mmcs = P2ValMmcs::new(hash, compress);
  let challenge_mmcs = P2ChallengeMmcs::new(val_mmcs.clone());

  let fri_params = FriParameters {
    log_blowup: 1,
    log_final_poly_len,
    num_queries,
    commit_proof_of_work_bits: 0,
    query_proof_of_work_bits: query_pow_bits,
    mmcs: challenge_mmcs,
  };

  let pcs = CirclePcs {
    mmcs: val_mmcs,
    fri_params,
    _phantom: core::marker::PhantomData,
  };

  let challenger = P2Challenger::new(perm);
  StarkConfig::new(pcs, challenger)
}

// ============================================================
// WFF / Term serialization helpers
// ============================================================

const WFF_TAG_EQUAL: u8 = 1;
const WFF_TAG_AND: u8 = 2;

const TERM_TAG_BOOL: u8 = 1;
const TERM_TAG_NOT: u8 = 2;
const TERM_TAG_AND: u8 = 3;
const TERM_TAG_OR: u8 = 4;
const TERM_TAG_XOR: u8 = 5;
const TERM_TAG_ITE: u8 = 6;
const TERM_TAG_BYTE: u8 = 7;
const TERM_TAG_BYTE_ADD: u8 = 8;
const TERM_TAG_BYTE_ADD_CARRY: u8 = 9;
const TERM_TAG_BYTE_MUL_LOW: u8 = 10;
const TERM_TAG_BYTE_MUL_HIGH: u8 = 11;
const TERM_TAG_BYTE_AND: u8 = 12;
const TERM_TAG_BYTE_OR: u8 = 13;
const TERM_TAG_BYTE_XOR: u8 = 14;

pub(super) fn serialize_term(term: &Term, out: &mut Vec<u8>) {
  match term {
    Term::Bool(v) => {
      out.push(TERM_TAG_BOOL);
      out.push(*v as u8);
    }
    Term::Not(a) => {
      out.push(TERM_TAG_NOT);
      serialize_term(a, out);
    }
    Term::And(a, b) => {
      out.push(TERM_TAG_AND);
      serialize_term(a, out);
      serialize_term(b, out);
    }
    Term::Or(a, b) => {
      out.push(TERM_TAG_OR);
      serialize_term(a, out);
      serialize_term(b, out);
    }
    Term::Xor(a, b) => {
      out.push(TERM_TAG_XOR);
      serialize_term(a, out);
      serialize_term(b, out);
    }
    Term::Ite(c, a, b) => {
      out.push(TERM_TAG_ITE);
      serialize_term(c, out);
      serialize_term(a, out);
      serialize_term(b, out);
    }
    Term::Byte(v) => {
      out.push(TERM_TAG_BYTE);
      out.push(*v);
    }
    Term::ByteAdd(a, b, c) => {
      out.push(TERM_TAG_BYTE_ADD);
      serialize_term(a, out);
      serialize_term(b, out);
      serialize_term(c, out);
    }
    Term::ByteAddCarry(a, b, c) => {
      out.push(TERM_TAG_BYTE_ADD_CARRY);
      serialize_term(a, out);
      serialize_term(b, out);
      serialize_term(c, out);
    }
    Term::ByteMulLow(a, b) => {
      out.push(TERM_TAG_BYTE_MUL_LOW);
      serialize_term(a, out);
      serialize_term(b, out);
    }
    Term::ByteMulHigh(a, b) => {
      out.push(TERM_TAG_BYTE_MUL_HIGH);
      serialize_term(a, out);
      serialize_term(b, out);
    }
    Term::ByteAnd(a, b) => {
      out.push(TERM_TAG_BYTE_AND);
      serialize_term(a, out);
      serialize_term(b, out);
    }
    Term::ByteOr(a, b) => {
      out.push(TERM_TAG_BYTE_OR);
      serialize_term(a, out);
      serialize_term(b, out);
    }
    Term::ByteXor(a, b) => {
      out.push(TERM_TAG_BYTE_XOR);
      serialize_term(a, out);
      serialize_term(b, out);
    }
  }
}

pub(super) fn serialize_wff(wff: &WFF, out: &mut Vec<u8>) {
  match wff {
    WFF::Equal(lhs, rhs) => {
      out.push(WFF_TAG_EQUAL);
      serialize_term(lhs, out);
      serialize_term(rhs, out);
    }
    WFF::And(a, b) => {
      out.push(WFF_TAG_AND);
      serialize_wff(a, out);
      serialize_wff(b, out);
    }
  }
}

pub(super) fn serialize_wff_bytes(wff: &WFF) -> Vec<u8> {
  let mut out = Vec::new();
  serialize_wff(wff, &mut out);
  out
}
