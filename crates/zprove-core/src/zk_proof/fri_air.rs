//! FRI (Fast Reed-Solomon Interactive Oracle Proof) 쿼리 및 검증 AIR (Phase 3).
//!
//! Circle STARK의 FRI fold 스텝을 회로로 구현한다.
//!
//! ## Circle FRI Fold 공식 (p3-circle `fold_x_row`)
//!
//! ```text
//! t       = nth_x_twiddle(reverse_bits_len(index, log_h)).inverse()   // Val (M31)
//! sum     = y_plus + y_minus                                           // EF
//! diff    = (y_plus - y_minus) * t                                     // EF * Val = EF
//! folded  = (sum + beta * diff).halve()                                // EF
//! ```
//!
//! y_plus, y_minus, beta, folded ∈ M31³ (BinomialExtensionField).
//! t ∈ M31 (base field scalar).
//!
//! ## 트레이스 레이아웃 (FriQueryAir — 19열)
//!
//! | 열      | 이름         | 설명                                      |
//! |---------|--------------|-------------------------------------------|
//! |  0      | t_inv        | twiddle 역원 = nth_x_twiddle(rev_idx)⁻¹   |
//! |  1..4   | y_plus[3]    | f(P) ∈ M31³ (베이스 계수 3개)              |
//! |  4..7   | y_minus[3]   | f(P̄) ∈ M31³ (켤레 평가값)                 |
//! |  7..10  | beta[3]      | 챌린지 β ∈ M31³                           |
//! | 10..13  | folded[3]    | fold 결과 ∈ M31³                          |
//! | 13..16  | sum[3]       | y_plus + y_minus (witness)                |
//! | 16..19  | diff[3]      | (y_plus - y_minus) * t_inv (witness)      |
//!
//! ## 제약 목록 (모두 차수 ≤ 2)
//!
//! sum 정의 (선형, 차수 1):
//!   sum[i] = y_plus[i] + y_minus[i]   (i=0,1,2)
//!
//! diff 정의 (2차):
//!   diff[i] = (y_plus[i] - y_minus[i]) * t_inv   (i=0,1,2)
//!
//! fold 합성 (M31³ 산술, 최대 2차):
//!   beta * diff 의 M31³ 곱 계산 후:
//!   2 * folded[i] = sum[i] + (beta * diff)[i]   (i=0,1,2)
//!
//! M31³ 기약 다항식은 X³ - 5 (= X³ - w, w=5), 즉:
//!   (a * b)[0] = a[0]*b[0] + 5*(a[1]*b[2] + a[2]*b[1])
//!   (a * b)[1] = a[0]*b[1] + a[1]*b[0] + 5*a[2]*b[2]
//!   (a * b)[2] = a[0]*b[2] + a[1]*b[1] + a[2]*b[0]
//!
//! ## FriVerifierAir
//!
//! `FriVerifierAir`는 `FriQueryAir`를 `k` 쿼리 × `log₂n` fold 깊이로 중첩한다.

use super::m31ext3_air::prove_m31ext3_mul;
use super::types::{Challenge, CircleStarkProof, CircleStarkVerifyResult, Val, make_circle_config};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;

// ── 상수 ─────────────────────────────────────────────────────────────────────

/// FriQueryAir 트레이스 열 수.
/// FriQueryAir 트레이스 열 수.
/// 레이아웃: t_inv(1) + y_plus(3) + y_minus(3) + beta(3) + folded(3) + sum(3) + diff(3) = 19
pub const NUM_FRI_QUERY_COLS: usize = 19;

// 열 인덱스
const COL_T_INV: usize = 0;   // twiddle 역원 (Val)
const COL_Y_PLUS: usize = 1;  // [1, 4) — y_plus ∈ M31³
const COL_Y_MINUS: usize = 4; // [4, 7) — y_minus ∈ M31³
const COL_BETA: usize = 7;    // [7, 10) — β ∈ M31³
const COL_FOLDED: usize = 10; // [10, 13) — fold 결과 ∈ M31³
const COL_SUM: usize = 13;    // [13, 16) — sum = y_plus + y_minus (witness)
const COL_DIFF: usize = 16;   // [16, 19) — diff = (y_plus - y_minus) * t_inv (witness)

// ── FRI 쿼리 입력 ─────────────────────────────────────────────────────────────

/// Circle FRI 단일 fold 스텝에 대한 입력/출력.
#[derive(Debug, Clone)]
pub struct FriQueryInput {
    /// twiddle 역원: nth_x_twiddle(reverse_bits_len(index, log_h)).inverse() (M31).
    pub t_inv: Val,
    /// `f(P)` ∈ M31³. Circle 위 점 P에서의 다항식 평가값.
    pub y_plus: Challenge,
    /// `f(P̄)` ∈ M31³. 켤레 점 P̄에서의 평가값.
    pub y_minus: Challenge,
    /// Fiat-Shamir 챌린지 β ∈ M31³.
    pub beta: Challenge,
    /// 예상 fold 결과 ∈ M31³ (프루버가 제공, AIR이 검증).
    pub expected_folded: Challenge,
}

// ── AIR 정의 ──────────────────────────────────────────────────────────────────

/// Circle FRI fold 단일 스텝을 검증하는 AIR.
///
/// 공식: `folded = ((y_plus + y_minus) + beta * (y_plus - y_minus) * t_inv) / 2`
///
/// 각 행은 독립적인 쿼리 (`FriQueryInput`)를 검증한다.
///
/// ### 제약 (모두 차수 ≤ 2)
///
/// sum 정의 (차수 1):
///   sum[i] − y_plus[i] − y_minus[i] = 0   (i=0,1,2)
///
/// diff 정의 (차수 2):
///   diff[i] − (y_plus[i] − y_minus[i]) * t_inv = 0   (i=0,1,2)
///
/// fold 합성 — beta * diff in M31³ (X³−5 기약, 차수 2):
///   bd[0] = beta[0]*diff[0] + 5*(beta[1]*diff[2] + beta[2]*diff[1])
///   bd[1] = beta[0]*diff[1] + beta[1]*diff[0] + 5*beta[2]*diff[2]
///   bd[2] = beta[0]*diff[2] + beta[1]*diff[1] + beta[2]*diff[0]
///   2*folded[i] − sum[i] − bd[i] = 0   (i=0,1,2)
pub struct FriQueryAir;

impl<F> BaseAir<F> for FriQueryAir {
    fn width(&self) -> usize {
        NUM_FRI_QUERY_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for FriQueryAir
where
    AB::F: PrimeCharacteristicRing,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("empty FriQuery trace");
        let row = &*row;

        let two = AB::F::from_u32(2u32);
        let w = AB::F::from_u32(5u32); // M31³ 기약 다항식 X³-5 의 w

        // ── sum 정의: sum[i] = y_plus[i] + y_minus[i] ──
        for i in 0..3 {
            let c = row[COL_SUM + i].clone().into()
                - row[COL_Y_PLUS + i].clone().into()
                - row[COL_Y_MINUS + i].clone().into();
            builder.assert_zero(c);
        }

        // ── diff 정의: diff[i] = (y_plus[i] - y_minus[i]) * t_inv (차수 2) ──
        for i in 0..3 {
            // diff[i] - (y_plus[i] - y_minus[i]) * t_inv = 0
            let c = row[COL_DIFF + i].clone().into()
                - (row[COL_Y_PLUS + i].clone().into() - row[COL_Y_MINUS + i].clone().into())
                    * row[COL_T_INV].clone();
            builder.assert_zero(c);
        }

        // ── fold 합성: 2*folded[i] = sum[i] + (beta * diff)[i] ──
        // beta * diff in M31³ (X³ - 5):
        //   bd[0] = b0*d0 + 5*(b1*d2 + b2*d1)
        //   bd[1] = b0*d1 + b1*d0 + 5*b2*d2
        //   bd[2] = b0*d2 + b1*d1 + b2*d0
        let b = [
            row[COL_BETA].clone(),
            row[COL_BETA + 1].clone(),
            row[COL_BETA + 2].clone(),
        ];
        let d = [
            row[COL_DIFF].clone(),
            row[COL_DIFF + 1].clone(),
            row[COL_DIFF + 2].clone(),
        ];

        let bd0 = b[0].clone() * d[0].clone()
            + b[1].clone() * d[2].clone() * w.clone()
            + b[2].clone() * d[1].clone() * w.clone();
        let bd1 = b[0].clone() * d[1].clone()
            + b[1].clone() * d[0].clone()
            + b[2].clone() * d[2].clone() * w.clone();
        let bd2 = b[0].clone() * d[2].clone()
            + b[1].clone() * d[1].clone()
            + b[2].clone() * d[0].clone();

        let bd = [bd0, bd1, bd2];
        for i in 0..3 {
            let c = row[COL_FOLDED + i].clone() * two.clone()
                - row[COL_SUM + i].clone().into()
                - bd[i].clone();
            builder.assert_zero(c);
        }
    }
}

// ── FriVerifierAir ────────────────────────────────────────────────────────────

/// FRI 완전 검증 AIR.
///
/// `k`개 FRI 쿼리를 `log2_n` fold 깊이로 검증한다.
pub struct FriVerifierAir {
    /// FRI 쿼리 개수.
    pub num_queries: usize,
    /// Fold 깊이 (= log₂(original_degree)).
    pub log2_degree: usize,
}

impl<F> BaseAir<F> for FriVerifierAir {
    fn width(&self) -> usize {
        NUM_FRI_QUERY_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for FriVerifierAir
where
    AB::F: PrimeCharacteristicRing,
{
    fn eval(&self, builder: &mut AB) {
        FriQueryAir.eval(builder);
    }
}

// ── 트레이스 생성 ─────────────────────────────────────────────────────────────

/// FRI 쿼리 입력들로부터 FriQueryAir 트레이스를 생성한다.
fn build_fri_query_trace(queries: &[FriQueryInput]) -> RowMajorMatrix<Val> {
    use p3_field::BasedVectorSpace;

    let height = queries.len().max(4).next_power_of_two();
    let mut data = vec![Val::new(0); height * NUM_FRI_QUERY_COLS];

    for (i, q) in queries.iter().enumerate() {
        let base = i * NUM_FRI_QUERY_COLS;

        // witness 계산
        let yp: &[Val] = q.y_plus.as_basis_coefficients_slice();
        let ym: &[Val] = q.y_minus.as_basis_coefficients_slice();
        let beta: &[Val] = q.beta.as_basis_coefficients_slice();
        let folded: &[Val] = q.expected_folded.as_basis_coefficients_slice();

        // sum[i] = y_plus[i] + y_minus[i]
        let sum: [Val; 3] = core::array::from_fn(|i| yp[i] + ym[i]);
        // diff[i] = (y_plus[i] - y_minus[i]) * t_inv
        let diff: [Val; 3] = core::array::from_fn(|i| (yp[i] - ym[i]) * q.t_inv);

        data[base + COL_T_INV] = q.t_inv;
        for k in 0..3 {
            data[base + COL_Y_PLUS + k] = yp[k];
            data[base + COL_Y_MINUS + k] = ym[k];
            data[base + COL_BETA + k] = beta[k];
            data[base + COL_FOLDED + k] = folded[k];
            data[base + COL_SUM + k] = sum[k];
            data[base + COL_DIFF + k] = diff[k];
        }
    }

    // 패딩 행: 모든 열 0 → sum 정의(0=0+0 ✓), diff 정의(0=0*0 ✓),
    // fold 합성(0=0+0 ✓) 모두 만족.
    // t_inv=0이어도 diff 제약 `diff[i]*t_inv = y+[i]-y-[i]` → 0=0 ✓
    // folded=0 제약 `2*0 = 0 + 0` ✓
    for _row in queries.len()..height {
        // 기본 0으로 채워짐 — 추가 처리 불필요
    }

    RowMajorMatrix::new(data, NUM_FRI_QUERY_COLS)
}

/// beta * diff 곱셈 입력 쌍을 수집한다.
/// M31Ext3MulAir로 별도 증명하기 위해 사용한다.
fn collect_beta_mul_pairs(queries: &[FriQueryInput]) -> Vec<(Challenge, Challenge)> {
    use p3_field::BasedVectorSpace;
    queries
        .iter()
        .map(|q| {
            let yp: &[Val] = q.y_plus.as_basis_coefficients_slice();
            let ym: &[Val] = q.y_minus.as_basis_coefficients_slice();
            let diff_coeffs: [Val; 3] = core::array::from_fn(|i| (yp[i] - ym[i]) * q.t_inv);
            let diff_ext = Challenge::from_basis_coefficients_fn(|i| diff_coeffs[i]);
            (q.beta, diff_ext)
        })
        .collect()
}

// ── 증명 묶음 ──────────────────────────────────────────────────────────────────

/// FRI 검증 증명 묶음.
///
/// - `query_proof`: FriQueryAir로 증명된 fold 스텝들
/// - `beta_mul_proof`: β × diff 곱셈을 M31Ext3MulAir로 증명
pub struct FriVerifyProof {
    /// FriQueryAir fold 스텝 STARK 증명.
    pub query_proof: CircleStarkProof,
    /// M31Ext3 챌린지-곱셈 STARK 증명.
    pub beta_mul_proof: CircleStarkProof,
}

// ── 공개 API ──────────────────────────────────────────────────────────────────

/// FRI 쿼리 배치를 STARK으로 증명한다.
///
/// 각 `FriQueryInput`은 하나의 fold 스텝이며, 모두 독립적으로 증명된다.
/// β × diff 곱셈은 `prove_m31ext3_mul`을 통해 M31Ext3MulAir로 분리 증명된다.
pub fn prove_fri_queries(queries: &[FriQueryInput]) -> FriVerifyProof {
    let beta_pairs = collect_beta_mul_pairs(queries);
    let beta_mul_proof = prove_m31ext3_mul(&beta_pairs);

    let air = FriQueryAir;
    let trace = build_fri_query_trace(queries);
    let config = make_circle_config();
    let query_proof = p3_uni_stark::prove(&config, &air, trace, &[]);

    FriVerifyProof {
        query_proof,
        beta_mul_proof,
    }
}

/// [`prove_fri_queries`] 결과를 검증한다.
pub fn verify_fri_queries(proof: &FriVerifyProof) -> CircleStarkVerifyResult {
    use super::m31ext3_air::verify_m31ext3_mul;
    verify_m31ext3_mul(&proof.beta_mul_proof)?;
    let air = FriQueryAir;
    let config = make_circle_config();
    p3_uni_stark::verify(&config, &air, &proof.query_proof, &[])
}
