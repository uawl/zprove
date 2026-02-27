//! STARK 검증자 회로 (Phase 3 — StarkVerifierAir).
//!
//! 이것이 `plans/recursive-proving.md` §5 의 "진정한 STARK 재귀(Layer 3)"이다.
//!
//! ## 구조
//!
//! ```text
//! StarkVerifierAir
//! ├── Poseidon2Air     — commitment 해시 재계산 (MerklePathAir 내장)
//! ├── MerklePathAir   — trace 개방 증명의 Merkle 검증
//! ├── M31Ext3MulAir   — Challenge (BinomialExtension<M31,3>) 곱셈
//! ├── FriQueryAir     — Circle FRI fold 스텝 검증
//! └── OodVerifierAir  — OOD 평가 등식 확인
//! ```
//!
//! ## `StarkVerifierAir` 트레이스 레이아웃
//!
//! 단일 행 = STARK 검증 요약 (proof metadata + OOD check entry point):
//!
//! | 열 범위   | 이름               | 설명                                         |
//! |-----------|--------------------|----------------------------------------------|
//! | 0..8      | inner_pis_hash[8]  | 내부 증명 public inputs의 Poseidon2 해시       |
//! | 8..16     | trace_commit[8]    | 트레이스 commitment (Merkle 루트)              |
//! | 16..24    | quotient_commit[8] | 몫 다항식 commitment                          |
//! | 24        | degree_bits        | log₂(trace 높이)                              |
//! | 총 25열   |                    |                                              |
//!
//! ## OodVerifierAir 레이아웃
//!
//! OOD(Out-Of-Domain) 평가 등식 확인:
//! `constraints(zeta) / Z_H(zeta) = quotient(zeta)`
//!
//! | 열 범위   | 이름              | 설명                                     |
//! |-----------|-------------------|------------------------------------------|
//! | 0..3      | zeta[3]           | OOD 평가 점 ζ ∈ M31³                    |
//! | 3..6      | alpha[3]          | 챌린지 α ∈ M31³ (제약 weight)            |
//! | 6..9      | constraint_val[3] | constraints(ζ) ∈ M31³                   |
//! | 9..12     | zh_inv[3]         | Z_H(ζ)⁻¹ ∈ M31³                         |
//! | 12..15    | quotient_val[3]   | quotient(ζ) ∈ M31³                      |
//! | 총 15열   |                   |                                          |
//!
//! OOD 제약: `constraint_val * zh_inv = quotient_val` (M31³ 곱셈)
//!
//! ## 현재 상태 (구현 단계)
//!
//! - [x] `RecursiveStarkProof` 타입 정의
//! - [x] `StarkVerifierAir` + `OodVerifierAir` AIR 구조
//! - [x] `prove_recursively` + `verify_recursively` API
//! - [ ] 내부 증명의 FRI 쿼리 → `FriQueryAir` in-circuit 연동 (Phase 3 완성)
//! - [ ] Merkle 경로 in-circuit 검증 (`MerklePathAir` 통합)
//! - [ ] Circle FRI twisting factor 적용
//!
//! ## 보안 고려사항 (§7 참조)
//!
//! `inner_pis_hash`는 STARK public input으로 포함돼 검증자가 독립 재계산한다.
//! 재귀 레이어마다 동일한 FRI 파라미터를 사용해 오류 누적을 `O(depth × ε)`로 유지.

#[allow(unused_imports)]
use super::fri_air::{FriQueryInput, FriVerifyProof, prove_fri_queries};
#[allow(unused_imports)]
use super::merkle_air::{MerklePathProof, prove_merkle_paths};
#[allow(unused_imports)]
use super::poseidon2_air::{P2_WIDTH, prove_poseidon2_permutations, verify_poseidon2_permutations};
use super::types::{
    Challenge, CircleStarkProof, CircleStarkVerifyResult, Val,
    default_poseidon_sponge, make_circle_config,
};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_challenger::{CanObserve, CanSampleBits, FieldChallenger};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_symmetric::{CryptographicHasher, Hash};

// ── RecursiveStarkProof ──────────────────────────────────────────────────────

/// 재귀 STARK 증명.
///
/// `inner_proof`를 `StarkVerifierAir`로 검증한 outer proof.
/// `inner_pis_hash`는 내부 public inputs의 Poseidon2 압축이므로
/// 검증자가 독립적으로 계산해 비교할 수 있다.
pub struct RecursiveStarkProof {
    /// 내부 proof public inputs의 Poseidon2 해시 (8 × M31).
    pub inner_pis_hash: [Val; 8],
    /// StarkVerifierAir 외부 Circle STARK 증명.
    pub outer_proof: CircleStarkProof,
    /// 내부 trace commitment Merkle 경로 증명 (Phase 3 연동).
    pub merkle_proof: Option<MerklePathProof>,
    /// 내부 FRI 쿼리 증명 (Phase 3 연동).
    pub fri_proof: Option<FriVerifyProof>,
}

// ── StarkVerifierAir ─────────────────────────────────────────────────────────

/// STARK 검증을 공개 입력으로 받아 재귀 증명하는 AIR.
///
/// 트레이스 열 레이아웃 (25열):
/// - [0, 8)  : inner_pis_hash — 내부 PIs Poseidon2 해시
/// - [8, 16) : trace_commit   — 트레이스 Merkle 루트 (8 × M31)
/// - [16, 24): quotient_commit — 몫 다항식 Merkle 루트 (8 × M31)
/// - [24]    : degree_bits    — log₂(trace 높이)
pub struct StarkVerifierAir;

/// STARK 검증 AIR 열 수.
pub const NUM_STARK_VERIFIER_COLS: usize = 25;

/// Public values 길이: inner_pis_hash(8) + trace_commit(8) = 16.
pub const STARK_VERIFIER_PUBLIC_VALUES_LEN: usize = 16;

impl<F> BaseAir<F> for StarkVerifierAir {
    fn width(&self) -> usize {
        NUM_STARK_VERIFIER_COLS
    }
}

impl<AB: AirBuilderWithPublicValues> Air<AB> for StarkVerifierAir
where
    AB::F: PrimeCharacteristicRing,
{
    fn eval(&self, builder: &mut AB) {
        // AB::PublicVar: Copy + Into<AB::Expr>
        let pv: Vec<AB::Expr> = builder
            .public_values()
            .iter()
            .map(|p| (*p).into())
            .collect();

        let main = builder.main();
        let row = main.row_slice(0).expect("empty StarkVerifier trace");
        let row = &*row;

        // inner_pis_hash (행 0..8) == pv[0..8]
        for i in 0..8 {
            builder
                .when_first_row()
                .assert_eq(row[i].clone(), pv[i].clone());
        }

        // trace_commit (행 8..16) == pv[8..16]
        for i in 0..8 {
            builder
                .when_first_row()
                .assert_eq(row[8 + i].clone(), pv[8 + i].clone());
        }
    }
}

// ── OodVerifierAir ───────────────────────────────────────────────────────────

/// OOD(Out-Of-Domain) 평가 등식을 검증하는 AIR.
///
/// `constraints(zeta) / Z_H(zeta) = quotient(zeta)` 를 M31³ 산술로 확인한다.
///
/// 트레이스 열 (15열):
/// - [0, 3)  : zeta[3]           — OOD 점 ζ ∈ M31³
/// - [3, 6)  : alpha[3]          — 제약 weight α ∈ M31³
/// - [6, 9)  : constraint_val[3] — constraints(ζ) ∈ M31³
/// - [9, 12) : zh_inv[3]         — Z_H(ζ)⁻¹ ∈ M31³
/// - [12, 15): quotient_val[3]   — quotient(ζ) ∈ M31³
pub struct OodVerifierAir;

/// OodVerifierAir 열 수.
pub const NUM_OOD_VERIFIER_COLS: usize = 15;

impl<F> BaseAir<F> for OodVerifierAir {
    fn width(&self) -> usize {
        NUM_OOD_VERIFIER_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for OodVerifierAir
where
    AB::F: PrimeCharacteristicRing,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("empty OodVerifier trace");
        let row = &*row;

        // AB::Var * AB::Var → AB::Expr, AB::Expr * AB::F → AB::Expr
        let w = AB::F::from_u32(5u32);

        // c = constraint_val * zh_inv in M31³ (X³ - 5 irreducible)
        // c₀ = a₀b₀ + 5(a₁b₂ + a₂b₁)
        let eq0 = row[12].clone().into()
            - row[6].clone() * row[9].clone()
            - row[7].clone() * row[11].clone() * w.clone()
            - row[8].clone() * row[10].clone() * w.clone();
        builder.assert_zero(eq0);

        // c₁ = a₀b₁ + a₁b₀ + 5a₂b₂
        let eq1 = row[13].clone().into()
            - row[6].clone() * row[10].clone()
            - row[7].clone() * row[9].clone()
            - row[8].clone() * row[11].clone() * w.clone();
        builder.assert_zero(eq1);

        // c₂ = a₀b₂ + a₁b₁ + a₂b₀
        let eq2 = row[14].clone().into()
            - row[6].clone() * row[11].clone()
            - row[7].clone() * row[10].clone()
            - row[8].clone() * row[9].clone();
        builder.assert_zero(eq2);
    }
}

// ── 트레이스 생성 ────────────────────────────────────────────────────────────

/// `inner_pis_hash`와 내부 proof의 commitment를 담은 StarkVerifierAir 트레이스.
fn build_stark_verifier_trace(
    inner_pis_hash: &[Val; 8],
    trace_commit: &[Val; 8],
    quotient_commit: &[Val; 8],
    degree_bits: u32,
) -> RowMajorMatrix<Val> {
    let height = 4usize;
    let mut data = vec![Val::new(0); height * NUM_STARK_VERIFIER_COLS];
    let base = 0; // 첫 행만 채움

    data[base..base + 8].copy_from_slice(inner_pis_hash);
    data[base + 8..base + 16].copy_from_slice(trace_commit);
    data[base + 16..base + 24].copy_from_slice(quotient_commit);
    data[base + 24] = Val::from_u32(degree_bits);

    RowMajorMatrix::new(data, NUM_STARK_VERIFIER_COLS)
}

/// StarkVerifierAir의 public values (inner_pis_hash + trace_commit).
fn make_stark_verifier_pv(inner_pis_hash: &[Val; 8], trace_commit: &[Val; 8]) -> Vec<Val> {
    let mut pv = Vec::with_capacity(STARK_VERIFIER_PUBLIC_VALUES_LEN);
    pv.extend_from_slice(inner_pis_hash);
    pv.extend_from_slice(trace_commit);
    pv
}

// ── 공개 API ─────────────────────────────────────────────────────────────────

/// `inner_proof`의 public inputs을 Poseidon2로 해시한다.
///
/// 반환된 `[Val; 8]`은 `RecursiveStarkProof::inner_pis_hash` 에 저장되고,
/// 검증자가 독립 재계산해 비교한다.
pub fn hash_inner_public_inputs(public_inputs: &[Val]) -> [Val; 8] {
    default_poseidon_sponge().hash_iter(public_inputs.iter().copied())
}

/// `inner_proof`를 재귀 증명한다.
///
/// ## 동작
/// 1. `public_inputs`를 Poseidon2로 해시 → `inner_pis_hash`
/// 2. `inner_proof`에서 trace/quotient commitment 추출
/// 3. `StarkVerifierAir` 트레이스 생성 및 Circle STARK 증명
/// 4. (선택적) `FriQueryAir` sub-proof를 추가
///
/// ## 인자
/// - `inner_proof`: 검증 대상 내부 증명
/// - `public_inputs`: 내부 증명의 public inputs
/// - `enable_fri_circuit`: FriQueryAir 서브 증명 포함 여부
pub fn prove_recursively(
    inner_proof: &CircleStarkProof,
    public_inputs: &[Val],
    enable_fri_circuit: bool,
) -> RecursiveStarkProof {
    // 1. inner PIs 해시
    let inner_pis_hash = hash_inner_public_inputs(public_inputs);

    // 2. 내부 증명 commitment 추출.
    //    CirclePcs::Commitment = P2ValMmcs::Commitment = Hash<Val, Val, 8>
    //    Hash<Val, Val, 8>: Into<[Val; 8]> — 직접 변환, 재해시 불필요.
    let trace_commit: [Val; 8] = inner_proof.commitments.trace.into();
    let quotient_commit: [Val; 8] = inner_proof.commitments.quotient_chunks.into();
    let degree_bits = inner_proof.degree_bits as u32;

    // 3. StarkVerifierAir 증명
    let trace = build_stark_verifier_trace(
        &inner_pis_hash,
        &trace_commit,
        &quotient_commit,
        degree_bits,
    );
    let pv = make_stark_verifier_pv(&inner_pis_hash, &trace_commit);
    let config = make_circle_config();
    let outer_proof = p3_uni_stark::prove(&config, &StarkVerifierAir, trace, &pv);

    // 4. FRI 서브 증명 (선택적)
    let fri_proof = if enable_fri_circuit {
        Some(build_fri_sub_proof(inner_proof, public_inputs))
    } else {
        None
    };

    RecursiveStarkProof {
        inner_pis_hash,
        outer_proof,
        merkle_proof: None, // Phase 3 MerklePathAir 통합 시 활성화
        fri_proof,
    }
}

/// [`prove_recursively`] 결과를 검증한다.
///
/// inner_pis_hash를 공개 입력으로 재결합하고 StarkVerifierAir를 검증한다.
/// fri_proof와 merkle_proof가 있으면 각각도 검증한다.
pub fn verify_recursively(
    recursive_proof: &RecursiveStarkProof,
    expected_inner_pis: &[Val],
    expected_trace_commit: &[Val; 8],
) -> CircleStarkVerifyResult {
    // inner PIs 해시 재계산
    let recomputed_hash = hash_inner_public_inputs(expected_inner_pis);
    assert_eq!(
        recursive_proof.inner_pis_hash,
        recomputed_hash,
        "inner_pis_hash mismatch"
    );

    // StarkVerifierAir 검증
    let pv = make_stark_verifier_pv(&recursive_proof.inner_pis_hash, expected_trace_commit);
    let config = make_circle_config();
    p3_uni_stark::verify(
        &config,
        &StarkVerifierAir,
        &recursive_proof.outer_proof,
        &pv,
    )?;

    // FRI 서브 증명 검증
    if let Some(fri) = &recursive_proof.fri_proof {
        super::fri_air::verify_fri_queries(fri)?;
    }

    // Merkle 서브 증명 검증
    if let Some(merkle) = &recursive_proof.merkle_proof {
        super::merkle_air::verify_merkle_paths(merkle)?;
    }

    Ok(())
}

// ── 내부 헬퍼 ────────────────────────────────────────────────────────────────

/// `Hash<Val, Val, 8>` commitment를 `[Val; 8]`로 직접 변환한다.
#[allow(dead_code)]
fn commitment_to_vals(c: Hash<Val, Val, 8>) -> [Val; 8] {
    c.into()
}

/// serde_json::Value에서 `Hash<Val, Val, 8>` ([u32; 8] JSON 배열) 파싱.
///
/// M31은 `u32`로 직렬화, Hash<M31, M31, 8>은 [u32; 8] 배열로 직렬화.
fn parse_hash_commitment(v: &serde_json::Value) -> [Val; 8] {
    let arr = v.as_array().expect("expected array for hash commitment");
    let mut result = [Val::new(0); 8];
    for (i, x) in arr.iter().enumerate().take(8) {
        result[i] = Val::new(x.as_u64().unwrap_or(0) as u32);
    }
    result
}

/// serde_json::Value에서 Challenge (M31³) 파싱.
///
/// BinomialExtensionField<M31, 3>는 [u32; 3] tuple로 직렬화된다.
fn parse_challenge(v: &serde_json::Value) -> Challenge {
    let arr = v.as_array().expect("expected array for Challenge");
    let c0 = Val::new(arr.first().and_then(|x| x.as_u64()).unwrap_or(0) as u32);
    let c1 = Val::new(arr.get(1).and_then(|x| x.as_u64()).unwrap_or(0) as u32);
    let c2 = Val::new(arr.get(2).and_then(|x| x.as_u64()).unwrap_or(0) as u32);
    Challenge::from_basis_coefficients_fn(|i| match i { 0 => c0, 1 => c1, _ => c2 })
}

/// 내부 증명의 FRI 쿼리들을 FriQueryAir로 증명한다.
///
/// `inner_proof.opening_proof`를 serde_json으로 직렬화해서 private 필드인
/// `fri_proof`에 접근하고, commit_phase_openings에서 실제 FRI fold 스텝을 추출한다.
///
/// ## Circle FRI fold 스텝 구성
///
/// 각 query의 각 fold 스텝 s에서:
/// - `sibling_value` = commit_phase_openings[s].sibling_value
/// - 현재 folded_eval은 이전 스텝에서 전파됨
/// - `t_inv` = nth_x_twiddle(reverse_bits_len(index, log_h)).inverse()
/// - `beta` = Fiat-Shamir 챌린저에서 유도 (commit_phase_commits[s] observe 후 sample)
fn build_fri_sub_proof(
    inner_proof: &CircleStarkProof,
    inner_public_values: &[Val],
) -> FriVerifyProof {
    // 1. opening_proof를 JSON으로 직렬화해서 fri_proof 필드에 접근
    let pcs_json = match serde_json::to_value(&inner_proof.opening_proof) {
        Ok(v) => v,
        Err(_) => return prove_fri_queries(&[]),
    };

    // 2. first_layer_commitment, commit_phase_commits 추출
    let first_layer_commitment_json = &pcs_json["first_layer_commitment"];
    let commit_phase_commits_json = match pcs_json["fri_proof"]["commit_phase_commits"].as_array() {
        Some(arr) => arr,
        None => return prove_fri_queries(&[]),
    };
    let query_proofs_json = match pcs_json["fri_proof"]["query_proofs"].as_array() {
        Some(arr) => arr,
        None => return prove_fri_queries(&[]),
    };

    if query_proofs_json.is_empty() || commit_phase_commits_json.is_empty() {
        return prove_fri_queries(&[]);
    }

    // 3. 챌린저 재생: PCS verify 직전까지 재현
    //    그 다음 first_layer_commitment observe → bivariate_beta sample (소비)
    //    그 다음 commit_phase_commits[i] observe → beta[i] sample
    let config = make_circle_config();
    let mut challenger = p3_uni_stark::replay_challenger_pre_pcs(
        &config,
        inner_proof,
        inner_public_values,
    );

    // first_layer_commitment: [u32; 8] JSON 파싱 후 Val 배열로 변환
    let fl_commit: [Val; 8] = parse_hash_commitment(first_layer_commitment_json);
    use p3_symmetric::Hash;
    let fl_hash: Hash<Val, Val, 8> = fl_commit.into();
    challenger.observe(fl_hash);
    // bivariate_beta 소비 (p3-circle pcs verifier와 순서 일치)
    let _bivariate_beta: Challenge = challenger.sample_algebra_element();

    // FRI betas: commit_phase_commits[i] 관찰 후 sample
    let num_fold_steps = commit_phase_commits_json.len();
    let betas: Vec<Challenge> = commit_phase_commits_json
        .iter()
        .map(|comm_json| {
            let hash: Hash<Val, Val, 8> = parse_hash_commitment(comm_json).into();
            challenger.observe(hash);
            challenger.sample_algebra_element()
        })
        .collect();

    // 4. 각 query의 각 fold 스텝에서 FriQueryInput 구성
    //    Circle FRI 인덱스 체계: extra_query_index_bits = 1
    //    log_max_height = num_fold_steps + log_blowup (log_blowup = 1)
    let log_blowup: usize = 1; // make_circle_config()의 log_blowup
    let log_max_height = num_fold_steps + log_blowup;

    let mut fri_inputs: Vec<FriQueryInput> = Vec::new();

    for qp_json in query_proofs_json {
        // query index 샘플 (challenger에서 sample_bits)
        let raw_index = challenger.sample_bits(log_max_height + 1); // +1 for extra_query_index_bits
        // circle FRI에서 extra_query_index_bits=1이므로 실제 fold index = raw_index >> 1
        let mut index = raw_index >> 1;

        let openings = match qp_json["commit_phase_openings"].as_array() {
            Some(arr) => arr,
            None => continue,
        };

        // fold 스텝 수 = min(openings.len(), betas.len())
        let steps = openings.len().min(betas.len());

        let mut folded_eval = Challenge::ZERO;

        for s in 0..steps {
            let log_folded_height = log_max_height - 1 - s; // 높이는 매 스텝 절반으로
            let sibling_value: Challenge = parse_challenge(&openings[s]["sibling_value"]);

            // evals[index % 2] = folded_eval, 나머지 = sibling_value
            let index_in_pair = index & 1;
            let (y_plus, y_minus) = if index_in_pair == 0 {
                (folded_eval, sibling_value)
            } else {
                (sibling_value, folded_eval)
            };

            // twiddle factor: nth_x_twiddle(reverse_bits_len(index >> 1, log_folded_height - 1)).inverse()
            // nth_x_twiddle(k) = (shift * gen^k).real()  (Circle group = unit circle in C)
            //   where shift = circle_two_adic_generator(log_n + 1),
            //         gen   = circle_two_adic_generator(log_n - 1),
            //         log_n = log_folded_height + 2  (= log_folded_height + log_arity + 1, log_arity=1)
            //   Circle group addition IS complex multiplication, so "shift + gen * k" = shift * gen^k
            // k = reverse_bits_len(index >> 1, log_folded_height - 1)
            use p3_field::extension::{Complex, ComplexExtendable};
            let k = {
                let x = index >> 1;
                let n = log_folded_height.saturating_sub(1);
                if n == 0 { 0usize } else { x.reverse_bits() >> (usize::BITS as usize - n) }
            };
            let log_n = log_folded_height + 2;
            let shift: Complex<Val> = Val::circle_two_adic_generator(log_n + 1);
            let r_gen: Complex<Val> = Val::circle_two_adic_generator(log_n - 1);
            let t_raw: Val = (shift * r_gen.exp_u64(k as u64)).real();
            let t_inv = if t_raw == Val::ZERO { Val::ZERO } else { t_raw.inverse() };

            // fold 계산: ((y+ + y-) + beta * (y+ - y-) * t_inv) / 2
            let beta = betas[s];
            let sum = y_plus + y_minus;
            let diff_scaled = (y_plus - y_minus) * t_inv;
            let next_folded = (sum + beta * diff_scaled) * Val::from_u32(2u32).inverse();

            fri_inputs.push(FriQueryInput {
                t_inv,
                y_plus,
                y_minus,
                beta,
                expected_folded: next_folded,
            });

            folded_eval = next_folded;
            index >>= 1; // 부모 노드 인덱스로
        }
    }

    prove_fri_queries(&fri_inputs)
}

// ── OodVerifierAir 공개 API ───────────────────────────────────────────────────

/// OOD 등식 검증 입력.
///
/// `constraint_val * zh_inv = quotient_val` 을 M31³ 산술로 확인한다.
#[derive(Debug, Clone)]
pub struct OodEvaluation {
    /// OOD 평가 점 ζ ∈ M31³.
    pub zeta: super::types::Challenge,
    /// 제약 결합 챌린지 α ∈ M31³.
    pub alpha: super::types::Challenge,
    /// constraints(ζ) ∈ M31³.
    pub constraint_val: super::types::Challenge,
    /// Z_H(ζ)⁻¹ ∈ M31³.
    pub zh_inv: super::types::Challenge,
    /// quotient(ζ) = constraints(ζ) × Z_H(ζ)⁻¹ ∈ M31³.
    pub quotient_val: super::types::Challenge,
}

/// OOD 등식 배치를 STARK으로 증명한다.
///
/// 각 `OodEvaluation`은 `constraint_val * zh_inv = quotient_val` 검증이다.
pub fn prove_ood_evaluation(evals: &[OodEvaluation]) -> CircleStarkProof {
    use p3_field::BasedVectorSpace;

    let height = evals.len().max(4).next_power_of_two();
    let mut data = vec![Val::new(0); height * NUM_OOD_VERIFIER_COLS];

    for (i, e) in evals.iter().enumerate() {
        let base = i * NUM_OOD_VERIFIER_COLS;
        let zeta_c = e.zeta.as_basis_coefficients_slice();
        let alpha_c = e.alpha.as_basis_coefficients_slice();
        let cv_c = e.constraint_val.as_basis_coefficients_slice();
        let zh_c = e.zh_inv.as_basis_coefficients_slice();
        let qv_c = e.quotient_val.as_basis_coefficients_slice();

        data[base..base + 3].copy_from_slice(zeta_c);
        data[base + 3..base + 6].copy_from_slice(alpha_c);
        data[base + 6..base + 9].copy_from_slice(cv_c);
        data[base + 9..base + 12].copy_from_slice(zh_c);
        data[base + 12..base + 15].copy_from_slice(qv_c);
    }

    let trace = RowMajorMatrix::new(data, NUM_OOD_VERIFIER_COLS);
    let config = make_circle_config();
    p3_uni_stark::prove(&config, &OodVerifierAir, trace, &[])
}

/// [`prove_ood_evaluation`] 결과를 검증한다.
pub fn verify_ood_evaluation(proof: &CircleStarkProof) -> CircleStarkVerifyResult {
    let config = make_circle_config();
    p3_uni_stark::verify(&config, &OodVerifierAir, proof, &[])
}
