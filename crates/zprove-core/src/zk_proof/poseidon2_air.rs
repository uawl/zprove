//! Poseidon2 permutation AIR for M31 (Phase 3).
//!
//! `p3-poseidon2-air`를 M31 파라미터로 타입 별칭화하고, prove/verify 편의 함수를
//! 제공한다.
//!
//! ## M31 파라미터 (width = 16, S-box = x^5)
//!
//! `poseidon2_round_numbers_128`에서 M31 × (width=16, degree=5) → (8, 14):
//! - `HALF_FULL_ROUNDS` = 4  (external rounds 총 8개)
//! - `PARTIAL_ROUNDS`   = 14 (optimal for 128-bit security, M31 31-bit prime)
//! - `SBOX_DEGREE`      = 5  (x^5, gcd(5, p-1)=1 for M31)
//! - `SBOX_REGISTERS`   = 1  (t = x² 중간변수 1개로 x^5 계산)
//!
//! ## Round constant 일치성
//!
//! `matching_p2_round_constants()`는 `SmallRng::seed_from_u64(0xC0DEC0DE)`로
//! 시드를 동일하게 설정하므로, `default_poseidon_sponge()`가 내부에서 사용하는
//! `Poseidon2Mersenne31::<16>::new_from_rng_128`와 동일한 라운드 상수를 생성한다.

use super::types::{CircleStarkProof, CircleStarkVerifyResult, Val, make_circle_config};
use p3_mersenne_31::GenericPoseidon2LinearLayersMersenne31;
use p3_poseidon2_air::{Poseidon2Air, RoundConstants, generate_vectorized_trace_rows};
use p3_symmetric::Permutation;
use rand::SeedableRng;
use rand::rngs::SmallRng;

// ── M31 Poseidon2 파라미터 상수 ───────────────────────────────────────────────

/// Poseidon2 상태 너비.
pub const P2_WIDTH: usize = 16;
/// S-box 다항식 차수 (x^5, M31에서 gcd(5, p-1)=1).
pub const P2_SBOX_DEGREE: u64 = 5;
/// S-box 중간 레지스터 개수 (x^5: t=x², 1개 충분).
pub const P2_SBOX_REGISTERS: usize = 1;
/// 절반 ext round 수 (총 external rounds = 8).
pub const P2_HALF_FULL_ROUNDS: usize = 4;
/// Partial round 수 (128-bit 보안, M31 width=16 degree=5).
pub const P2_PARTIAL_ROUNDS: usize = 14;

// ── 타입 별칭 ─────────────────────────────────────────────────────────────────

/// M31 Poseidon2 AIR 타입. 한 행(row)이 하나의 permutation을 검증한다.
///
/// `Air<AB>` 구현은 `AB::F: Algebra<Val>` 바운드를 가지므로
/// `SymbolicAirBuilder`, `ProverConstraintFolder`, `VerifierConstraintFolder`
/// 모두에 적용된다.
pub type P2Air = Poseidon2Air<
  Val,
  GenericPoseidon2LinearLayersMersenne31,
  P2_WIDTH,
  P2_SBOX_DEGREE,
  P2_SBOX_REGISTERS,
  P2_HALF_FULL_ROUNDS,
  P2_PARTIAL_ROUNDS,
>;

/// M31 Poseidon2 AIR 라운드 상수 타입.
pub type P2RoundConstants = RoundConstants<Val, P2_WIDTH, P2_HALF_FULL_ROUNDS, P2_PARTIAL_ROUNDS>;

// ── 상수 생성 헬퍼 ────────────────────────────────────────────────────────────

/// `default_poseidon_sponge()`와 **동일한 라운드 상수**를 생성한다.
///
/// 두 생성자가 같은 시드(`0xC0DEC0DE`)와 동일한 RNG 소비 순서를 사용하므로
/// AIR 제약과 퍼뮤테이션 실행이 반드시 일치한다.
pub fn matching_p2_round_constants() -> P2RoundConstants {
  let mut rng = SmallRng::seed_from_u64(0xC0DE_C0DE_u64);
  P2RoundConstants::from_rng(&mut rng)
}

/// 기본 Poseidon2 AIR 인스턴스를 생성한다.
pub fn default_p2_air() -> P2Air {
  P2Air::new(matching_p2_round_constants())
}

// ── Trace 생성 헬퍼 ───────────────────────────────────────────────────────────

/// `inputs` 를 2의 거듭제곱으로 패딩 후 Poseidon2 trace 행렬을 반환한다.
///
/// 각 행은 하나의 permutation 상태 전이를 인코딩한다 (VECTOR_LEN = 1).
fn build_p2_trace(mut inputs: Vec<[Val; P2_WIDTH]>) -> p3_matrix::dense::RowMajorMatrix<Val> {
  let n = inputs.len().max(4).next_power_of_two();
  inputs.resize(n, [Val::new(0); P2_WIDTH]);
  let constants = matching_p2_round_constants();
  generate_vectorized_trace_rows::<
    Val,
    GenericPoseidon2LinearLayersMersenne31,
    P2_WIDTH,
    P2_SBOX_DEGREE,
    P2_SBOX_REGISTERS,
    P2_HALF_FULL_ROUNDS,
    P2_PARTIAL_ROUNDS,
    1, // VECTOR_LEN
  >(inputs, &constants, 0)
}

// ── 공개 API ──────────────────────────────────────────────────────────────────

/// Poseidon2 permutation 배치를 STARK으로 증명한다.
///
/// `inputs`의 각 원소가 하나의 폭-16 입력 상태이다.
/// 길이가 2의 거듭제곱이 아니면 영-벡터로 자동 패딩된다.
pub fn prove_poseidon2_permutations(inputs: Vec<[Val; P2_WIDTH]>) -> CircleStarkProof {
  let air = default_p2_air();
  let trace = build_p2_trace(inputs);
  let config = make_circle_config();
  p3_uni_stark::prove(&config, &air, trace, &[])
}

/// [`prove_poseidon2_permutations`] 결과를 검증한다.
pub fn verify_poseidon2_permutations(proof: &CircleStarkProof) -> CircleStarkVerifyResult {
  let air = default_p2_air();
  let config = make_circle_config();
  p3_uni_stark::verify(&config, &air, proof, &[])
}

/// Poseidon2 permutation 배치를 STARK으로 증명하고 **입출력 쌍**을 함께 반환한다.
///
/// `ChallengerTranscriptAir`가 "permutation이 올바르게 계산됐다"는 사실을
/// 공개 입력을 통해 외부 STARK과 결합(cross-proof linking)할 때 사용한다.
///
/// ## 반환값
/// `(proof, outputs)` — `proof`의 public values는 `flatten([in_i || out_i])`,
/// `outputs[i]`는 `P2Perm(inputs[i])`의 첫 16원소 상태.
///
/// ## 공개 입력(public values) 레이아웃
/// `pv = [in_0[0..16], out_0[0..16], in_1[0..16], out_1[0..16], ...]`
/// 패딩 행(영-벡터)의 IO도 포함되므로 `pv.len() == 32 * n_padded`.
pub fn prove_poseidon2_permutations_io(
  inputs: Vec<[Val; P2_WIDTH]>,
) -> (CircleStarkProof, Vec<[Val; P2_WIDTH]>) {
  use p3_mersenne_31::Poseidon2Mersenne31;

  let air = default_p2_air();

  // Native permutation with the same seed
  let perm = {
    let mut rng = SmallRng::seed_from_u64(0xC0DE_C0DE_u64);
    Poseidon2Mersenne31::<16>::new_from_rng_128(&mut rng)
  };

  let n_padded = inputs.len().max(4).next_power_of_two();
  let mut padded = inputs.clone();
  padded.resize(n_padded, [Val::new(0); P2_WIDTH]);

  let n_orig = inputs.len();
  let outputs_full: Vec<[Val; P2_WIDTH]> = padded
    .iter()
    .map(|inp| {
      let mut state = *inp;
      state = perm.permute(state);
      state
    })
    .collect();
  let outputs: Vec<[Val; P2_WIDTH]> = outputs_full[..n_orig].to_vec();

  // public values: [in_0, out_0, in_1, out_1, ...] (padded rows 포함)
  let mut pv: Vec<Val> = Vec::with_capacity(32 * n_padded);
  for (inp, out) in padded.iter().zip(outputs_full.iter()) {
    pv.extend_from_slice(inp);
    pv.extend_from_slice(out);
  }

  let trace = build_p2_trace(padded);
  let config = make_circle_config();
  let proof = p3_uni_stark::prove(&config, &air, trace, &pv);
  (proof, outputs)
}

/// [`prove_poseidon2_permutations_io`] 결과를 검증한다.
///
/// `expected_ios`는 `(input, output)` 쌍의 슬라이스.
/// public values와 일치하는지 검사 후 STARK 증명을 검증한다.
pub fn verify_poseidon2_permutations_io(
  proof: &CircleStarkProof,
  expected_ios: &[([Val; P2_WIDTH], [Val; P2_WIDTH])],
) -> CircleStarkVerifyResult {
  let air = default_p2_air();
  let config = make_circle_config();
  // public values 재결합
  let pv: Vec<Val> = expected_ios
    .iter()
    .flat_map(|(inp, out)| inp.iter().chain(out.iter()).copied())
    .collect();
  p3_uni_stark::verify(&config, &air, proof, &pv)
}
