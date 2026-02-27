//! GF(M31³) 곱셈 AIR (Phase 3 — M31Ext3MulAir).
//!
//! `BinomialExtensionField<M31, 3>` = M31[X] / (X³ − 5) 에서의 곱셈을 STARK으로
//! 검증한다. 기약다항식 `X³ = 5` 를 이용한 곱셈 공식:
//!
//! ```text
//! (a₀ + a₁X + a₂X²) × (b₀ + b₁X + b₂X²) = c₀ + c₁X + c₂X²
//!
//! c₀ = a₀b₀ + 5(a₁b₂ + a₂b₁)
//! c₁ = a₀b₁ + a₁b₀ + 5a₂b₂
//! c₂ = a₀b₂ + a₁b₁ + a₂b₀
//! ```
//!
//! ## 트레이스 레이아웃 (9열, 차수-2 제약 3개)
//!
//! | 열  | 이름 | 역할        |
//! |-----|------|-------------|
//! |  0  | a₀   | 입력 a의 상수항 |
//! |  1  | a₁   | 입력 a의 X 계수 |
//! |  2  | a₂   | 입력 a의 X² 계수 |
//! |  3  | b₀   | 입력 b의 상수항 |
//! |  4  | b₁   | 입력 b의 X 계수 |
//! |  5  | b₂   | 입력 b의 X² 계수 |
//! |  6  | c₀   | 출력 (검증) |
//! |  7  | c₁   | 출력 (검증) |
//! |  8  | c₂   | 출력 (검증) |

use super::types::{Challenge, CircleStarkProof, CircleStarkVerifyResult, Val, make_circle_config};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;

// ── 상수 ─────────────────────────────────────────────────────────────────────

/// 트레이스 열 수.
pub const NUM_EXT3_MUL_COLS: usize = 9;

// M31[X]/(X³−5) 에서 X³ = 5 이므로 non-residue W = 5
const W: u32 = 5;

// ── AIR 정의 ──────────────────────────────────────────────────────────────────

/// GF(M31³) 곱셈을 검증하는 AIR.
///
/// 각 행은 독립적인 `a × b = c` 확인이다.
/// 모든 제약이 차수-2 이므로 성능 페널티가 없다.
pub struct M31Ext3MulAir;

impl<F> BaseAir<F> for M31Ext3MulAir {
  fn width(&self) -> usize {
    NUM_EXT3_MUL_COLS
  }
}

impl<AB: AirBuilder> Air<AB> for M31Ext3MulAir
where
  AB::F: PrimeCharacteristicRing,
{
  fn eval(&self, builder: &mut AB) {
    let main = builder.main();
    let row = main.row_slice(0).expect("empty M31Ext3Mul trace");
    let row = &*row;

    // AB::Var: Clone (not Copy) — row[i].clone() each use.
    // AB::Var * AB::Var → AB::Expr  (Mul<Var, Output=Expr>)
    // AB::Expr * AB::F → AB::Expr  (Algebra<F>: Mul<F, Output=Self>)

    let w = AB::F::from_u32(W); // PrimeCharacteristicRing::from_u32

    // 제약 0: c₀ − a₀b₀ − 5·a₁b₂ − 5·a₂b₁ = 0
    let a0b0: AB::Expr = row[0].clone() * row[3].clone();
    let a1b2: AB::Expr = row[1].clone() * row[5].clone();
    let a2b1: AB::Expr = row[2].clone() * row[4].clone();
    let lhs0 = row[6].clone().into() - a0b0 - a1b2 * w.clone() - a2b1 * w.clone();
    builder.assert_zero(lhs0);

    // 제약 1: c₁ − a₀b₁ − a₁b₀ − 5·a₂b₂ = 0
    let a0b1: AB::Expr = row[0].clone() * row[4].clone();
    let a1b0: AB::Expr = row[1].clone() * row[3].clone();
    let a2b2: AB::Expr = row[2].clone() * row[5].clone();
    let lhs1 = row[7].clone().into() - a0b1 - a1b0 - a2b2 * w.clone();
    builder.assert_zero(lhs1);

    // 제약 2: c₂ − a₀b₂ − a₁b₁ − a₂b₀ = 0
    let a0b2: AB::Expr = row[0].clone() * row[5].clone();
    let a1b1: AB::Expr = row[1].clone() * row[4].clone();
    let a2b0: AB::Expr = row[2].clone() * row[3].clone();
    let lhs2 = row[8].clone().into() - a0b2 - a1b1 - a2b0;
    builder.assert_zero(lhs2);
  }
}

// ── 트레이스 생성 ─────────────────────────────────────────────────────────────

/// `(a, b)` 쌍 목록으로부터 M31Ext3 곱셈 트레이스를 생성한다.
///
/// 각 행: [a₀, a₁, a₂, b₀, b₁, b₂, c₀, c₁, c₂]
/// 결과 `c`는 실제 GF(M31³) 곱셈으로 계산한다.
fn build_ext3_mul_trace(pairs: &[(Challenge, Challenge)]) -> RowMajorMatrix<Val> {
  use p3_field::BasedVectorSpace;
  let height = pairs.len().max(4).next_power_of_two();
  let mut data = vec![Val::new(0); height * NUM_EXT3_MUL_COLS];

  for (i, (a, b)) in pairs.iter().enumerate() {
    let base = i * NUM_EXT3_MUL_COLS;
    // BasedVectorSpace::as_basis_coefficients_slice returns &[Val; 3] for Challenge.
    let a_coeffs = a.as_basis_coefficients_slice();
    let b_coeffs = b.as_basis_coefficients_slice();
    let c = *a * *b;
    let c_coeffs = c.as_basis_coefficients_slice();

    data[base] = a_coeffs[0];
    data[base + 1] = a_coeffs[1];
    data[base + 2] = a_coeffs[2];
    data[base + 3] = b_coeffs[0];
    data[base + 4] = b_coeffs[1];
    data[base + 5] = b_coeffs[2];
    data[base + 6] = c_coeffs[0];
    data[base + 7] = c_coeffs[1];
    data[base + 8] = c_coeffs[2];
  }

  RowMajorMatrix::new(data, NUM_EXT3_MUL_COLS)
}

// ── 공개 API ──────────────────────────────────────────────────────────────────

/// GF(M31³) 곱셈 배치를 STARK으로 증명한다.
///
/// `pairs`의 각 `(a, b)`에 대해 `a × b = c` 를 계산하고 회로 제약을 증명한다.
pub fn prove_m31ext3_mul(pairs: &[(Challenge, Challenge)]) -> CircleStarkProof {
  let air = M31Ext3MulAir;
  let trace = build_ext3_mul_trace(pairs);
  let config = make_circle_config();
  p3_uni_stark::prove(&config, &air, trace, &[])
}

/// [`prove_m31ext3_mul`] 결과를 검증한다.
pub fn verify_m31ext3_mul(proof: &CircleStarkProof) -> CircleStarkVerifyResult {
  let air = M31Ext3MulAir;
  let config = make_circle_config();
  p3_uni_stark::verify(&config, &air, proof, &[])
}
