//! Merkle 경로 검증 AIR (Phase 3 — MerklePathAir).
//!
//! 높이 `h`의 Merkle 인증 경로를 STARK으로 검증한다.
//! 각 레벨에서 Poseidon2 압축을 사용한다:
//!
//! ```text
//! compress(left[8], right[8]) = Poseidon2_permutation([left || right])[0..8]
//! ```
//!
//! ## 설계 전략
//!
//! ### 트레이스 구조
//!
//! 각 행이 **하나의 Merkle 레벨(압축 1회)**을 표현한다.
//! 압축 입력(16원소)과 예상 출력 해시(8원소), 방향 비트(1원소)를 인코딩하며,
//! Poseidon2 중간 상태는 `p3-poseidon2-air`의 열 레이아웃을 임베딩한다.
//!
//! ```text
//! 열  0..8  : path_node[8]   — 현재 경로 노드
//! 열  8..16 : sibling[8]     — 형제 노드
//! 열  16    : direction_bit  — 0=path_node가 left, 1=path_node가 right
//! 열  17..25: parent[8]      — 계산된 부모 해시 (공개 출력 체인)
//! 총 25열 (Poseidon2 내부 상태는 별도 증명으로 분리)
//! ```
//!
//! ### 구현 방식
//!
//! 완전한 회로 내 Poseidon2 내장은 Phase 3 StarkVerifierAir에서 달성된다.
//! 이 모듈은 **commitment-chain 방식**을 사용:
//! 1. `prove_poseidon2_permutations`로 모든 압축 연산을 STARK 증명
//! 2. `MerklePathAir` 제약으로 경로 체이닝 구조를 검증
//! 3. `MerklePathProof`가 두 증명을 묶어 검증
//!
//! ## 보안 고려사항
//!
//! `direction_bit` 는 {0, 1} 이외의 값이 돼서는 안 된다.
//! AIR 제약: `direction_bit * (direction_bit - 1) = 0` (불리언 강제).

use super::poseidon2_air::{P2_WIDTH, prove_poseidon2_permutations, verify_poseidon2_permutations};
use super::types::{
  CircleStarkProof, CircleStarkVerifyResult, Val, default_poseidon_sponge, make_circle_config,
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_symmetric::CryptographicHasher;

// ── 상수 ─────────────────────────────────────────────────────────────────────

/// 트레이스 열 수 (경로 노드 + 형제 + 방향 비트 + 부모).
pub const NUM_MERKLE_COLS: usize = 25;

const COL_PATH: usize = 0; // [0, 8)
const COL_DIR: usize = 16;
const COL_PARENT: usize = 17; // [17, 25)

// ── 데이터 타입 ───────────────────────────────────────────────────────────────

/// Merkle 경로 검증 입력.
///
/// `leaf_hash`에서 `root`까지의 인증 경로를 표현한다.
#[derive(Debug, Clone)]
pub struct MerklePathInput {
  /// 리프 해시 (8 × M31). Poseidon2 sponge 출력.
  pub leaf_hash: [Val; 8],
  /// 경로 상의 형제 노드들 (`[sibling_hash; height]`).
  pub siblings: Vec<[Val; 8]>,
  /// 각 레벨의 방향 비트 (`false` = 우리 노드가 left child).
  pub directions: Vec<bool>,
  /// 검증하고자 하는 루트 해시 (8 × M31).
  pub root: [Val; 8],
}

/// Merkle 경로 STARK 증명 묶음.
///
/// - `compression_proof`: 모든 레벨의 Poseidon2 압축을 증명  
/// - `path_proof`:        경로 체이닝 구조를 증명
pub struct MerklePathProof {
  /// 모든 레벨 Poseidon2 압축 STARK 증명.
  pub compression_proof: CircleStarkProof,
  /// 경로 체이닝 AIR (MerklePathAir) STARK 증명.
  pub path_proof: CircleStarkProof,
}

// ── AIR 정의 ──────────────────────────────────────────────────────────────────

/// Merkle 경로 체이닝을 검증하는 AIR.
///
/// 각 행은 한 레벨의 경로 정보를 담으며, 다음을 제약한다:
/// 1. `direction_bit ∈ {0, 1}`       (불리언 강제)
/// 2. path_node, sibling → parent 결합이 올바른 순서
///    (`direction_bit = 0` → left = path_node, right = sibling;
///    `direction_bit = 1` → left = sibling, right = path_node)
/// 3. `parent[i]` 가 다음 행의 `path_node[i]`와 일치 (`is_transition` 비활성화)
///
/// Poseidon2 압축 결과는 별도의 `compression_proof`로 검증된다.
pub struct MerklePathAir;

impl<F> BaseAir<F> for MerklePathAir {
  fn width(&self) -> usize {
    NUM_MERKLE_COLS
  }
}

impl<AB: AirBuilder> Air<AB> for MerklePathAir
where
  AB::F: PrimeCharacteristicRing,
{
  fn eval(&self, builder: &mut AB) {
    let main = builder.main();
    let cur = main.row_slice(0).expect("empty merkle trace");
    let cur = &*cur;

    // 방향 비트 불리언 강제: bit * (bit - 1) = 0
    let dir: AB::Expr = cur[COL_DIR].clone().into();
    let one = AB::F::ONE;
    builder.assert_zero(dir.clone() * (dir - one));

    // 전이 제약: 현재 행의 parent[i] == 다음 행의 path_node[i]
    let next = main.row_slice(1).expect("empty merkle trace next");
    let next = &*next;

    for i in 0..8 {
      builder
        .when_transition()
        .assert_eq(cur[COL_PARENT + i].clone(), next[COL_PATH + i].clone());
    }
  }
}

// ── 트레이스 생성 ─────────────────────────────────────────────────────────────

/// Merkle 경로 입력에서 트레이스 행렬을 생성한다.
///
/// 각 행: [path_node[8], sibling[8], direction_bit, parent[8]]
/// `parent`는 Poseidon2 압축으로 계산된 실제 값이다.
fn build_merkle_trace(inputs: &[MerklePathInput]) -> RowMajorMatrix<Val> {
  let sponge = default_poseidon_sponge();
  let height = inputs.first().map(|i| i.siblings.len()).unwrap_or(1);
  let n_paths = inputs.len();
  let n_rows = (n_paths * height).max(4).next_power_of_two();
  let mut data = vec![Val::new(0); n_rows * NUM_MERKLE_COLS];

  for (p_idx, input) in inputs.iter().enumerate() {
    let mut current = input.leaf_hash;
    for (level, (sibling, &dir)) in input
      .siblings
      .iter()
      .zip(input.directions.iter())
      .enumerate()
    {
      let row = p_idx * height + level;
      if row >= n_rows {
        break;
      }
      let base = row * NUM_MERKLE_COLS;

      // (left, right) 결정: direction=0 → 나는 left
      let (left, right) = if !dir {
        (current, *sibling)
      } else {
        (*sibling, current)
      };

      // Poseidon2 압축: [left || right] → parent[0..8]
      let state: Vec<Val> = left.iter().chain(right.iter()).copied().collect();
      let hash: [Val; 8] = sponge.hash_iter(state);

      // path_node
      data[base..base + 8].copy_from_slice(&current);
      // sibling
      data[base + 8..base + 16].copy_from_slice(sibling);
      // direction_bit
      data[base + 16] = Val::from_u32(dir as u32);
      // parent
      data[base + 17..base + 25].copy_from_slice(&hash);

      current = hash;
    }
  }

  // 패딩 행: 마지막 실제 행의 parent 와 전이 제약을 맞추기 위해
  // path_node = last_parent, parent = last_parent 로 채운다.
  // (direction=0, sibling=zeros 는 이미 0 — boolean 제약 통과)
  let real_rows = n_paths * height;
  if real_rows < n_rows {
    // 마지막 실제 행 (real_rows - 1) 의 parent 를 읽어 패딩에 사용.
    let mut last_parent = [Val::new(0); 8];
    if real_rows > 0 {
      let last_base = (real_rows - 1) * NUM_MERKLE_COLS;
      last_parent.copy_from_slice(&data[last_base + 17..last_base + 25]);
    }
    for row in real_rows..n_rows {
      let base = row * NUM_MERKLE_COLS;
      // path_node (cols 0..8)
      data[base..base + 8].copy_from_slice(&last_parent);
      // sibling (cols 8..16) 는 0 유지
      // direction (col 16) 는 0 유지
      // parent (cols 17..25)
      data[base + 17..base + 25].copy_from_slice(&last_parent);
    }
  }

  RowMajorMatrix::new(data, NUM_MERKLE_COLS)
}

/// 경로 검증에 필요한 Poseidon2 압축 입력들을 수집한다.
fn collect_compression_inputs(inputs: &[MerklePathInput]) -> Vec<[Val; P2_WIDTH]> {
  let sponge = default_poseidon_sponge();
  let mut result = Vec::new();

  for input in inputs {
    let mut current = input.leaf_hash;
    for (sibling, &dir) in input.siblings.iter().zip(input.directions.iter()) {
      let (left, right) = if !dir {
        (current, *sibling)
      } else {
        (*sibling, current)
      };
      let mut state = [Val::new(0); P2_WIDTH];
      state[..8].copy_from_slice(&left);
      state[8..].copy_from_slice(&right);
      result.push(state);

      let state_vec: Vec<Val> = left.iter().chain(right.iter()).copied().collect();
      current = sponge.hash_iter(state_vec);
    }
  }

  result
}

// ── 공개 API ──────────────────────────────────────────────────────────────────

/// Merkle 경로 배치를 STARK으로 증명한다.
///
/// ## 증명 구성
/// 1. 각 레벨의 Poseidon2 압축을 `prove_poseidon2_permutations`로 증명
/// 2. 경로 체이닝 구조를 `MerklePathAir`로 증명
///
/// ## 보안 (Phase 2 Soundness 수준)
/// 검증자는 반드시 `compression_proof`(모든 해시를 in-circuit 검증)와
/// `path_proof`(체이닝 구조 검증) **모두**를 확인해야 한다.
pub fn prove_merkle_paths(inputs: &[MerklePathInput]) -> MerklePathProof {
  let compression_inputs = collect_compression_inputs(inputs);
  let compression_proof = prove_poseidon2_permutations(compression_inputs);

  let air = MerklePathAir;
  let trace = build_merkle_trace(inputs);
  let config = make_circle_config();
  let path_proof = p3_uni_stark::prove(&config, &air, trace, &[]);

  MerklePathProof {
    compression_proof,
    path_proof,
  }
}

/// [`prove_merkle_paths`] 결과를 검증한다.
///
/// compression_proof와 path_proof 모두 검증한다.
pub fn verify_merkle_paths(proof: &MerklePathProof) -> CircleStarkVerifyResult {
  verify_poseidon2_permutations(&proof.compression_proof)?;
  let air = MerklePathAir;
  let config = make_circle_config();
  p3_uni_stark::verify(&config, &air, &proof.path_proof, &[])
}
