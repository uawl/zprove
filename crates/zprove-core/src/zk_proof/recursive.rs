//! Segment-chain recursive proving skeleton (Phase 2).
//!
//! Building blocks for the commitment-chain recursive prover described in
//! `plans/recursive-proving.md §4`.
//!
//! Current status: **skeleton only** — AIR constraints are complete, but the
//! prover functions are stubs (`todo!()`) until `commit_vm_state` is wired up.
//!
//! ```text
//! Layer 1 (done):   SubCall inner_proof depth field (transition.rs)
//! Layer 2 (here):   LinkAir — segment boundary commitment chain
//! Layer 3 (future): StarkVerifierAir — full in-circuit STARK recursion
//! ```

use super::types::{
  CircleStarkProof, CircleStarkVerifyResult, Val, make_circle_config,
};
use crate::transition::VmState;

use p3_air::{Air, AirBuilderWithPublicValues, BaseAir};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

// ── Column layout (NUM_LINK_COLS = 36) ────────────────────────────────
//
//  Each row encodes one (left_out, right_in) segment-boundary pair.
//
//  Col   Name
//   0    left_pc
//   1    left_sp
//   2..9  left_stack_hash[0..8]
//  10..17 left_mem_root[0..8]
//  18   right_pc
//  19   right_sp
//  20..27 right_stack_hash[0..8]
//  28..35 right_mem_root[0..8]
// ─────────────────────────────────────────────────────────────────────

/// Total main-trace columns for the link AIR.
pub const NUM_LINK_COLS: usize = 36;
const LINK_PUBLIC_VALUES_LEN: usize = 36;

const LCOL_L_PC: usize = 0;
const LCOL_L_SP: usize = 1;
const LCOL_L_STACK: usize = 2;  // [2, 10)
const LCOL_L_MEM: usize = 10;   // [10, 18)
const LCOL_R_PC: usize = 18;
const LCOL_R_SP: usize = 19;
const LCOL_R_STACK: usize = 20; // [20, 28)
const LCOL_R_MEM: usize = 28;   // [28, 36)

// ── StateCommitment ───────────────────────────────────────────────────

/// Poseidon2-compressed snapshot of VM state at a segment boundary.
///
/// This is stored as STARK public inputs (18 M31 elements per boundary,
/// 36 total for a link proof) so the verifier can reconstruct commitments
/// independently and compare them against the prover-supplied trace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateCommitment {
  /// Program counter.
  pub pc: u32,
  /// Stack pointer (current depth).
  pub sp: u32,
  /// Poseidon2 digest of `stack[0..sp]` packed as 32-bit limbs.
  pub stack_hash: [Val; 8],
  /// Poseidon2 digest of the memory Merkle root bytes.
  pub memory_root: [Val; 8],
}

impl StateCommitment {
  /// Serialize into 18 field elements for use in a trace row or public-values vector.
  fn to_fields(&self) -> [Val; 18] {
    let mut out = [Val::ZERO; 18];
    out[0] = Val::from_u32(self.pc);
    out[1] = Val::from_u32(self.sp);
    out[2..10].copy_from_slice(&self.stack_hash);
    out[10..18].copy_from_slice(&self.memory_root);
    out
  }
}

/// Derive a [`StateCommitment`] from a VM execution snapshot.
///
/// **Stub — not yet implemented.**
/// See `plans/recursive-proving.md §4-1` for the full design:
/// - `stack_hash  = Poseidon2(stack[0..sp] packed as 32-bit limbs)`
/// - `memory_root = Poseidon2(existing 32-byte Merkle root bytes)`
pub fn commit_vm_state(_state: &VmState) -> StateCommitment {
  todo!("Phase 2: hash stack and memory_root via Poseidon2 sponge")
}

// ── Public-value helpers ──────────────────────────────────────────────

/// Encode `(s_in, s_out)` into the 36-element public-values vector for [`LinkAir`].
pub fn link_public_values(s_in: &StateCommitment, s_out: &StateCommitment) -> Vec<Val> {
  let mut pv = Vec::with_capacity(LINK_PUBLIC_VALUES_LEN);
  pv.extend_from_slice(&s_in.to_fields());
  pv.extend_from_slice(&s_out.to_fields());
  pv
}

// ── AIR ───────────────────────────────────────────────────────────────

/// Segment-linking AIR (36 columns).
///
/// Each trace row asserts that the exit state of segment *left* equals the
/// entry state of segment *right*, forming a valid segment-boundary chain.
///
/// Constraints (20 total, all degree ≤ 2):
/// - `left_stack_hash[i] == right_stack_hash[i]`  for i ∈ 0..8   (8 constraints)
/// - `left_mem_root[i]   == right_mem_root[i]`    for i ∈ 0..8   (8 constraints)
/// - `left_pc == right_pc`                                         (1 constraint)
/// - `left_sp == right_sp`                                         (1 constraint)
///
/// Public-value constraints pin the outermost segment boundaries so the verifier
/// can re-derive them from the block header independently.
pub struct LinkAir;

impl<F: p3_field::Field> BaseAir<F> for LinkAir {
  fn width(&self) -> usize {
    NUM_LINK_COLS
  }
}

impl<AB: AirBuilderWithPublicValues> Air<AB> for LinkAir
where
  AB::F: p3_field::Field,
{
  fn eval(&self, builder: &mut AB) {
    // Collect public values into owned expressions first to avoid
    // holding the immutable borrow on `builder` while calling assert_eq.
    let pv: Vec<AB::Expr> = builder
      .public_values()
      .iter()
      .map(|&p| Into::<AB::Expr>::into(p))
      .collect();

    // Collect row values into owned copies (drops the main-matrix borrow).
    let (l_pc, l_sp, l_stack, l_mem, r_pc, r_sp, r_stack, r_mem) = {
      let main = builder.main();
      let row = main.row_slice(0).expect("empty link trace");
      let row = &*row;
      (
        row[LCOL_L_PC].clone(),
        row[LCOL_L_SP].clone(),
        std::array::from_fn::<_, 8, _>(|i| row[LCOL_L_STACK + i].clone()),
        std::array::from_fn::<_, 8, _>(|i| row[LCOL_L_MEM   + i].clone()),
        row[LCOL_R_PC].clone(),
        row[LCOL_R_SP].clone(),
        std::array::from_fn::<_, 8, _>(|i| row[LCOL_R_STACK + i].clone()),
        std::array::from_fn::<_, 8, _>(|i| row[LCOL_R_MEM   + i].clone()),
      )
    };

    // stack_hash continuity: left[i] == right[i]
    for i in 0..8 {
      builder.assert_eq(l_stack[i].clone().into(), r_stack[i].clone().into());
    }
    // mem_root continuity: left[i] == right[i]
    for i in 0..8 {
      builder.assert_eq(l_mem[i].clone().into(), r_mem[i].clone().into());
    }
    // PC and SP continuity
    builder.assert_eq(l_pc.clone().into(), r_pc.clone().into());
    builder.assert_eq(l_sp.clone().into(), r_sp.clone().into());

    // Pin public values to the first row so the verifier can independently re-derive them.
    // pv layout: [l_pc, l_sp, l_stack[8], l_mem[8], r_pc, r_sp, r_stack[8], r_mem[8]]
    builder.assert_eq(pv[0].clone(), l_pc.into());
    builder.assert_eq(pv[1].clone(), l_sp.into());
    for i in 0..8 {
      builder.assert_eq(pv[2  + i].clone(), l_stack[i].clone().into());
      builder.assert_eq(pv[10 + i].clone(), l_mem[i].clone().into());
    }
    builder.assert_eq(pv[18].clone(), r_pc.into());
    builder.assert_eq(pv[19].clone(), r_sp.into());
    for i in 0..8 {
      builder.assert_eq(pv[20 + i].clone(), r_stack[i].clone().into());
      builder.assert_eq(pv[28 + i].clone(), r_mem[i].clone().into());
    }
  }
}

// ── Trace builder ─────────────────────────────────────────────────────

fn build_link_trace(links: &[(StateCommitment, StateCommitment)]) -> RowMajorMatrix<Val> {
  let height = links.len().max(4).next_power_of_two();
  let mut data = vec![Val::ZERO; height * NUM_LINK_COLS];
  for (i, (left_out, right_in)) in links.iter().enumerate() {
    let base = i * NUM_LINK_COLS;
    let lf = left_out.to_fields();
    let rf = right_in.to_fields();
    data[base..base + 18].copy_from_slice(&lf);
    data[base + 18..base + 36].copy_from_slice(&rf);
  }
  RowMajorMatrix::new(data, NUM_LINK_COLS)
}

// ── Public API ─────────────────────────────────────────────────────────

/// Prove that adjacent segment boundaries form a valid chain.
///
/// Each `(left_out, right_in)` pair in `links` must satisfy
/// `left_out == right_in` — the AIR enforces this.
///
/// `s_in` and `s_out` are the chain's outermost boundaries bound into
/// the STARK's public inputs.
///
/// **Prerequisite**: `commit_vm_state` must be implemented.
pub fn prove_link_stark(
  links: &[(StateCommitment, StateCommitment)],
  s_in: &StateCommitment,
  s_out: &StateCommitment,
) -> CircleStarkProof {
  let public_values = link_public_values(s_in, s_out);
  let trace = build_link_trace(links);
  let config = make_circle_config();
  p3_uni_stark::prove(&config, &LinkAir, trace, &public_values)
}

/// Verify a [`prove_link_stark`] proof.
pub fn verify_link_stark(
  proof: &CircleStarkProof,
  s_in: &StateCommitment,
  s_out: &StateCommitment,
) -> CircleStarkVerifyResult {
  let public_values = link_public_values(s_in, s_out);
  let config = make_circle_config();
  p3_uni_stark::verify(&config, &LinkAir, proof, &public_values)
}
