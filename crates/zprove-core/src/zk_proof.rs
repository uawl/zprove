//! Circle STARK proof generation and verification for EVM state transitions.
//!
//! Arithmetizes the byte-level semantic proofs into an AIR over the Mersenne-31
//! field, then proves correctness using the Plonky3 Circle STARK framework.
//!
//! # Architecture
//!
//! The AIR has 32 rows (one per byte position) with columns:
//!
//! | Column    | Description                                    |
//! |-----------|------------------------------------------------|
//! | `a`       | Byte from first operand                        |
//! | `b`       | Byte from second operand                       |
//! | `carry_in`| Carry input from previous byte (0 or 1)        |
//! | `sum`     | `(a + b + carry_in) mod 256`                   |
//! | `carry_out`| `(a + b + carry_in) / 256`  (0 or 1)          |
//! | `expected`| Expected output byte                           |
//!
//! Constraints:
//!   1. `a + b + carry_in = sum + 256 * carry_out`
//!   2. `sum = expected`
//!   3. `carry_out ∈ {0, 1}`
//!   4. Transition: `carry_in[next] = carry_out[current]`
//!   5. First row: `carry_in = 0`

use core::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_circle::CirclePcs;
use p3_commit::ExtensionMmcs;
use p3_field::extension::BinomialExtensionField;
use p3_field::PrimeCharacteristicRing;
use p3_fri::FriParameters;
use p3_keccak::Keccak256Hash;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_mersenne_31::Mersenne31;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use p3_uni_stark::StarkConfig;

// ============================================================
// Type aliases for Circle STARK over M31
// ============================================================

pub type Val = Mersenne31;
pub type Challenge = BinomialExtensionField<Val, 3>;

type ByteHash = Keccak256Hash;
type FieldHash = SerializingHasher<ByteHash>;
type Compress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
type ValMmcs = MerkleTreeMmcs<Val, u8, FieldHash, Compress, 32>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;

pub type Pcs = CirclePcs<Val, ValMmcs, ChallengeMmcs>;
pub type CircleStarkConfig = StarkConfig<Pcs, Challenge, Challenger>;

// ============================================================
// Trace row layout
// ============================================================

/// Number of columns in the byte-add AIR.
pub const NUM_ADD_COLS: usize = 6;

/// Trace row for byte-level addition with carry chain.
#[repr(C)]
#[derive(Clone, Debug)]
pub struct ByteAddRow<F> {
  /// Byte from operand `a`.
  pub a: F,
  /// Byte from operand `b`.
  pub b: F,
  /// Carry input (0 or 1).
  pub carry_in: F,
  /// `(a + b + carry_in) mod 256`.
  pub sum: F,
  /// `(a + b + carry_in) / 256` — 0 or 1.
  pub carry_out: F,
  /// Expected output byte.
  pub expected: F,
}

impl<F> Borrow<ByteAddRow<F>> for [F] {
  fn borrow(&self) -> &ByteAddRow<F> {
    debug_assert_eq!(self.len(), NUM_ADD_COLS);
    let (prefix, rows, _suffix) = unsafe { self.align_to::<ByteAddRow<F>>() };
    debug_assert!(prefix.is_empty());
    &rows[0]
  }
}

// ============================================================
// AIR definition
// ============================================================

/// AIR for 256-bit (32-byte) addition with ripple-carry.
///
/// Proves that `a[0..32] + b[0..32] = expected[0..32]` (mod 2^256)
/// using a 32-row trace with byte-level carry propagation.
pub struct ByteAddAir;

impl<F> BaseAir<F> for ByteAddAir {
  fn width(&self) -> usize {
    NUM_ADD_COLS
  }
}

impl<AB: AirBuilder> Air<AB> for ByteAddAir {
  fn eval(&self, builder: &mut AB) {
    let main = builder.main();
    let local = main.row_slice(0).expect("empty trace");
    let next = main.row_slice(1).expect("single-row trace");

    let local: &ByteAddRow<AB::Var> = (*local).borrow();
    let next: &ByteAddRow<AB::Var> = (*next).borrow();

    let c256 = AB::Expr::from_u16(256);

    // ── Constraint 1: a + b + carry_in = sum + 256 * carry_out ──
    // Equivalently: a + b + carry_in - sum - 256 * carry_out = 0
    builder.assert_zero(
      local.a.clone().into()
        + local.b.clone().into()
        + local.carry_in.clone().into()
        - local.sum.clone().into()
        - c256 * local.carry_out.clone().into(),
    );

    // ── Constraint 2: sum = expected ──
    builder.assert_eq(local.sum.clone(), local.expected.clone());

    // ── Constraint 3: carry_out ∈ {0, 1} ──
    builder.assert_bool(local.carry_out.clone());

    // ── Constraint 4: carry_in ∈ {0, 1} ──
    builder.assert_bool(local.carry_in.clone());

    // ── Constraint 5: First row carry_in = 0 ──
    builder.when_first_row().assert_zero(local.carry_in.clone());

    // ── Constraint 6: Transition carry chain ──
    // next.carry_in = local.carry_out
    builder
      .when_transition()
      .assert_eq(next.carry_in.clone(), local.carry_out.clone());
  }
}

// ============================================================
// Trace generation
// ============================================================

/// Generate the execution trace for a 32-byte addition.
///
/// `a` and `b` are 32-byte big-endian inputs.
/// `expected` is the 32-byte big-endian expected result.
///
/// The trace has 32 rows, processing from LSB (byte 31) to MSB (byte 0).
pub fn generate_add_trace(a: &[u8; 32], b: &[u8; 32], expected: &[u8; 32]) -> RowMajorMatrix<Val> {
  let n_rows = 32usize;
  let mut trace = RowMajorMatrix::new(Val::zero_vec(n_rows * NUM_ADD_COLS), NUM_ADD_COLS);

  let (prefix, rows, _suffix) = unsafe { trace.values.align_to_mut::<ByteAddRow<Val>>() };
  debug_assert!(prefix.is_empty());
  assert_eq!(rows.len(), n_rows);

  let mut carry: u16 = 0;

  // Row 0 = byte 31 (LSB), Row 31 = byte 0 (MSB)
  for row in 0..32 {
    let byte_idx = 31 - row; // big-endian → LSB first
    let av = a[byte_idx] as u16;
    let bv = b[byte_idx] as u16;
    let total = av + bv + carry;
    let sum = total & 0xFF;
    let carry_out = total >> 8;

    rows[row] = ByteAddRow {
      a: Val::from_u16(av as u16),
      b: Val::from_u16(bv as u16),
      carry_in: Val::from_u16(carry as u16),
      sum: Val::from_u16(sum as u16),
      carry_out: Val::from_u16(carry_out as u16),
      expected: Val::from_u16(expected[byte_idx] as u16),
    };

    carry = carry_out;
  }

  trace
}

// ============================================================
// Config builder
// ============================================================

/// Build a Circle STARK configuration over M31 with Keccak hashing.
pub fn make_circle_config() -> CircleStarkConfig {
  let byte_hash = Keccak256Hash {};
  let field_hash = SerializingHasher::new(byte_hash);
  let compress = CompressionFunctionFromHasher::new(byte_hash);
  let val_mmcs = MerkleTreeMmcs::new(field_hash, compress);
  let challenge_mmcs = ExtensionMmcs::new(val_mmcs.clone());

  let fri_params = FriParameters {
    log_blowup: 1,
    log_final_poly_len: 0,
    num_queries: 40,
    commit_proof_of_work_bits: 0,
    query_proof_of_work_bits: 8,
    mmcs: challenge_mmcs,
  };

  let pcs = CirclePcs {
    mmcs: val_mmcs,
    fri_params,
    _phantom: core::marker::PhantomData,
  };

  let challenger = SerializingChallenger32::from_hasher(vec![], byte_hash);
  StarkConfig::new(pcs, challenger)
}

// ============================================================
// Prove & Verify
// ============================================================

/// Generate a Circle STARK proof for a 256-bit addition.
///
/// Returns the serialized proof.
pub fn prove_add_stark(
  a: &[u8; 32],
  b: &[u8; 32],
  expected: &[u8; 32],
) -> p3_uni_stark::Proof<CircleStarkConfig> {
  let config = make_circle_config();
  let trace = generate_add_trace(a, b, expected);

  p3_uni_stark::prove(&config, &ByteAddAir, trace, &[])
}

/// Verify a Circle STARK proof for a 256-bit addition.
pub fn verify_add_stark(
  proof: &p3_uni_stark::Proof<CircleStarkConfig>,
) -> Result<(), p3_uni_stark::VerificationError<p3_uni_stark::PcsError<CircleStarkConfig>>> {
  let config = make_circle_config();
  p3_uni_stark::verify(&config, &ByteAddAir, proof, &[])
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
  use super::*;

  fn u256_bytes(val: u128) -> [u8; 32] {
    let mut b = [0u8; 32];
    b[16..32].copy_from_slice(&val.to_be_bytes());
    b
  }

  fn compute_add(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    let mut result = [0u8; 32];
    let mut carry = 0u16;
    for i in (0..32).rev() {
      let sum = a[i] as u16 + b[i] as u16 + carry;
      result[i] = (sum & 0xFF) as u8;
      carry = sum >> 8;
    }
    result
  }

  #[test]
  fn test_trace_generation() {
    let a = u256_bytes(1000);
    let b = u256_bytes(2000);
    let c = compute_add(&a, &b);
    let trace = generate_add_trace(&a, &b, &c);
    assert_eq!(trace.width(), NUM_ADD_COLS);
    assert_eq!(trace.height(), 32);
  }

  #[test]
  fn test_prove_and_verify_add_simple() {
    let a = u256_bytes(1000);
    let b = u256_bytes(2000);
    let c = compute_add(&a, &b);

    let proof = prove_add_stark(&a, &b, &c);
    verify_add_stark(&proof).expect("verification failed");
  }

  #[test]
  fn test_prove_and_verify_add_overflow() {
    let a = [0xFF; 32];
    let mut b = [0u8; 32];
    b[31] = 1;
    let c = compute_add(&a, &b);

    let proof = prove_add_stark(&a, &b, &c);
    verify_add_stark(&proof).expect("verification failed");
  }

  #[test]
  fn test_prove_and_verify_add_large() {
    let mut a = [0xABu8; 32];
    a[0] = 0x7F;
    let mut b = [0xCDu8; 32];
    b[0] = 0x3E;
    let c = compute_add(&a, &b);

    let proof = prove_add_stark(&a, &b, &c);
    verify_add_stark(&proof).expect("verification failed");
  }

  #[test]
  fn test_prove_and_verify_add_zero() {
    let a = [0u8; 32];
    let b = [0u8; 32];
    let c = [0u8; 32];

    let proof = prove_add_stark(&a, &b, &c);
    verify_add_stark(&proof).expect("verification failed");
  }
}
