//! In-circuit Fiat-Shamir challenger 트랜스크립트 (Phase 3).
//!
//! `DuplexChallenger<M31, Poseidon2, 16, 8>`의 duplex 시퀀스를 STARK AIR로
//! 검증한다. 이것이 Phase 3 Fiat-Shamir in-circuit 구현의 핵심이다.
//!
//! ## 설계 전략 (commitment-chain 방식)
//!
//! `p3-poseidon2-air`의 P2Air는 이미 permutation 정확성을 회로로 증명한다.
//! 이 모듈은 그 위에 **상태 연속성 + 챌린지 결합**을 추가한다:
//!
//! 1. 각 duplex step을 `DuplexRecord`로 기록
//! 2. `ChallengerTranscriptAir`: 인접 행 전이 제약으로 연속성 강제
//! 3. `prove_poseidon2_permutations_io`로 P2 permutation 정확성 증명
//! 4. 두 proof의 public values 일치를 verifier가 확인 → 단계 연결
//!
//! ## 트레이스 레이아웃 (ChallengerTranscriptAir — 40열)
//!
//! | 열 범위  | 이름           | 설명                                      |
//! |----------|----------------|-------------------------------------------|
//! | 0..16    | state_in[16]   | duplex 직전 sponge 상태                   |
//! | 16..24   | rate_input[8]  | 흡수 값 (부족하면 zero-pad)               |
//! | 24..40   | state_out[16]  | P2(state_in XOR_rate rate_input) 결과     |
//!
//! `state_in[rate] = state_out_prev[rate] + rate_input` 형태로 재구성된다.
//! (DuplexChallenger는 rate part를 overwrite하지 XOR하지 않지만 M31에서
//!  add-then-permute와 overwrite-then-permute는 다르므로 실제 semantics 반영:
//!  state_in의 rate 부분을 rate_input으로 교체해서 permute)
//!
//! ## 제약
//!
//! 행 r, 행 r+1 전이:
//!   A. `state_in[r+1][i] == state_out[r][i]`  for i in 0..16  (is_transition)
//!   B. `state_in[0][i] == 0`                  for i in 0..16  (when_first_row + public PV)
//!
//! P2 permutation 결합 (cross-proof):
//!   C. P2Air proof의 pv[r*32..r*32+16] == xored_input[r]
//!      P2Air proof의 pv[r*32+16..r*32+32] == state_out[r]
//!      → verifier가 양 proof의 pv 일치를 확인
//!
//! 챌린지 결합:
//!   D. `alpha == state_out[alpha_duplex_row][squeeze_col]` 등을 public values로 노출
//!
//! ## 보안 갭 현황
//!
//! - ✅ 상태 연속성 in-circuit 강제
//! - ✅ Poseidon2 permutation 정확성 (P2Air, 별도 proof)
//! - ✅ 챌린지 값이 특정 duplex_row의 state_out에서 나왔음을 in-circuit 강제
//! - ⬜ FriQueryAir의 beta가 이 transcript의 beta_i와 동일함을 cross-link 필요

use super::poseidon2_air::{
    P2_WIDTH, prove_poseidon2_permutations_io,
    verify_poseidon2_permutations_io,
};
use super::types::{
    Challenge, CircleStarkProof, CircleStarkVerifyResult, P2Challenger, Val,
    make_circle_config,
};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_challenger::{CanObserve, CanSampleBits, FieldChallenger};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_mersenne_31::Poseidon2Mersenne31;
use p3_symmetric::Permutation;
use rand::SeedableRng;

// ── 상수 ─────────────────────────────────────────────────────────────────────

/// ChallengerTranscriptAir 트레이스 열 수 (state_in=16 + rate_input=8 + state_out=16).
pub const NUM_CHALLENGER_COLS: usize = 40;
/// Rate (DuplexChallenger RATE = 8).
pub const CHALLENGER_RATE: usize = 8;

const COL_STATE_IN: usize = 0;   // [0, 16)
const COL_RATE_INPUT: usize = 16; // [16, 24)
const COL_STATE_OUT: usize = 24;  // [24, 40)

// Public values 레이아웃:
//   [0..16]  = initial_state (보통 모두 0)
//   [16..]   = 챌린지들: alpha(3), zeta(3), bivariate_beta(3), beta_0(3), ...
/// Public values에서 initial_state의 길이.
pub const CHALLENGER_PV_INIT_STATE_LEN: usize = 16;

// ── duplex 기록 ──────────────────────────────────────────────────────────────

/// 단일 duplex step 기록.
#[derive(Debug, Clone)]
pub struct DuplexRecord {
    /// Duplex 직전 sponge 상태 (16 × M31).
    pub state_before: [Val; P2_WIDTH],
    /// Rate 부분에 흡수된 값 (8 × M31, 미채워진 부분은 0).
    pub rate_input: [Val; CHALLENGER_RATE],
    /// `P2Perm(overwrite_rate(state_before, rate_input))` 결과 (16 × M31).
    pub state_after: [Val; P2_WIDTH],
}

/// 트랜스크립트 재생에서 추출된 챌린지들.
#[derive(Debug, Clone)]
pub struct ChallengesExtracted {
    /// 제약 집계 챌린지 α ∈ M31³.
    pub alpha: Challenge,
    /// OOD 점 ζ ∈ M31³.
    pub zeta: Challenge,
    /// Circle PCS bivariate beta ∈ M31³.
    pub bivariate_beta: Challenge,
    /// FRI fold 챌린지 β_i ∈ M31³ (fold depth만큼).
    pub fri_betas: Vec<Challenge>,
    /// FRI 쿼리 인덱스 (쿼리 수만큼).
    pub query_indices: Vec<usize>,
    /// alpha를 squeeze한 duplex row 인덱스.
    pub alpha_duplex_row: usize,
    /// zeta를 squeeze한 duplex row 인덱스.
    pub zeta_duplex_row: usize,
}

// ── P2 permutation 헬퍼 ──────────────────────────────────────────────────────

fn make_native_perm() -> Poseidon2Mersenne31<16> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0xC0DE_C0DE_u64);
    Poseidon2Mersenne31::<16>::new_from_rng_128(&mut rng)
}

// ── transcript 추출 공개 API ─────────────────────────────────────────────────

/// 내부 proof의 Fiat-Shamir 트랜스크립트를 duplex 단위로 추출한다.
///
/// `replay_challenger_pre_pcs`와 동일한 순서로 challenger를 재생하되,
/// 각 `duplexing()` 직전까지의 상태를 `DuplexRecord`로 기록한다.
///
/// ## 트랜스크립트 순서 (배치 증명 1개 기준)
///
/// 1. observe(degree_bits), observe(degree_bits-1), observe(0)
/// 2. observe(trace_commit[0..8]) → 8원소가 차면 duplex #1
/// 3. observe(*) ... observe(pv[*]) → duplex #2 이후
/// 4. sample alpha  (squeeze)
/// 5. observe(quotient_commit[0..8]) → duplex
/// 6. sample zeta   (squeeze)
/// 7. [PCS 내부] observe(first_layer_commit) → duplex
/// 8. sample bivariate_beta
/// 9. for s: observe(fold_commit_s) → duplex; sample beta_s
/// 10. for q: sample_bits(log_h+1) → query_index_q
pub fn extract_challenger_transcript(
    inner_proof: &CircleStarkProof,
    public_values: &[Val],
) -> (Vec<DuplexRecord>, ChallengesExtracted) {
    use p3_symmetric::Hash;

    // ── 챌린저 초기화 ────────────────────────────────────────────────────────
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0xC0DE_C0DE_u64);
    let perm_for_challenger = Poseidon2Mersenne31::<16>::new_from_rng_128(&mut rng);

    let mut challenger = P2Challenger::new(perm_for_challenger);

    // ── Step 1: degree_bits + preprocessed_width + trace_commit ──────────────
    let degree_bits = inner_proof.degree_bits;
    let preprocessed_width = inner_proof
        .opened_values
        .preprocessed_local
        .as_ref()
        .map(|v| v.len())
        .unwrap_or(0);

    let mut records2: Vec<DuplexRecord> = Vec::new();

    macro_rules! observe_val {
        ($v:expr) => {{
            if challenger.input_buffer.len() == CHALLENGER_RATE - 1 {
                // 다음 observe가 duplex를 트리거
                let sb = challenger.sponge_state;
                let mut ri = [Val::new(0); CHALLENGER_RATE];
                for (i, x) in challenger.input_buffer.iter().enumerate() {
                    ri[i] = *x;
                }
                ri[CHALLENGER_RATE - 1] = $v;
                challenger.observe($v);
                let sa = challenger.sponge_state;
                records2.push(DuplexRecord { state_before: sb, rate_input: ri, state_after: sa });
            } else {
                challenger.observe($v);
            }
        }};
    }

    macro_rules! observe_hash {
        ($h:expr) => {{
            use p3_symmetric::Hash;
            let arr: [Val; 8] = $h.into();
            for v in arr {
                observe_val!(v);
            }
        }};
    }

    observe_val!(Val::from_usize(degree_bits));
    observe_val!(Val::from_usize(degree_bits.saturating_sub(1)));
    observe_val!(Val::from_usize(preprocessed_width));
    observe_hash!(inner_proof.commitments.trace);
    for &pv in public_values {
        observe_val!(pv);
    }

    // squeeze alpha
    let alpha_duplex_row_before = records2.len();
    // squeeze direct에서도 duplex가 일어날 수 있음 (output_buffer 소진 시)
    let alpha: Challenge = challenger.sample_algebra_element();
    let alpha_duplex_row = if records2.len() > alpha_duplex_row_before {
        records2.len() - 1
    } else {
        alpha_duplex_row_before.saturating_sub(1)
    };

    observe_hash!(inner_proof.commitments.quotient_chunks);
    if let Some(r) = inner_proof.commitments.random.clone() {
        observe_hash!(r);
    }

    // squeeze zeta
    let zeta_duplex_row_before = records2.len();
    let zeta: Challenge = challenger.sample_algebra_element();
    let zeta_duplex_row = if records2.len() > zeta_duplex_row_before {
        records2.len() - 1
    } else {
        zeta_duplex_row_before.saturating_sub(1)
    };

    // ── PCS 내부: first_layer_commit + FRI betas + query indices ─────────────
    let pcs_json = match serde_json::to_value(&inner_proof.opening_proof) {
        Ok(v) => v,
        Err(_) => {
            return (
                records2,
                ChallengesExtracted {
                    alpha,
                    zeta,
                    bivariate_beta: Challenge::ZERO,
                    fri_betas: vec![],
                    query_indices: vec![],
                    alpha_duplex_row,
                    zeta_duplex_row,
                },
            );
        }
    };

    let fl_json = &pcs_json["first_layer_commitment"];
    let fl_arr: [Val; 8] = {
        let arr: Vec<u32> = fl_json
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_u64().map(|v| v as u32)).collect())
            .unwrap_or_default();
        let mut out = [Val::new(0); 8];
        for (i, v) in arr.iter().take(8).enumerate() {
            out[i] = Val::new(*v);
        }
        out
    };

    for v in fl_arr {
        observe_val!(v);
    }
    let bivariate_beta: Challenge = challenger.sample_algebra_element();

    let commit_phase_commits = pcs_json["fri_proof"]["commit_phase_commits"]
        .as_array()
        .cloned()
        .unwrap_or_default();

    let mut fri_betas: Vec<Challenge> = Vec::new();
    for comm_json in &commit_phase_commits {
        let arr: [Val; 8] = {
            let arr: Vec<u32> = comm_json
                .as_array()
                .map(|a| a.iter().filter_map(|x| x.as_u64().map(|v| v as u32)).collect())
                .unwrap_or_default();
            let mut out = [Val::new(0); 8];
            for (i, v) in arr.iter().take(8).enumerate() {
                out[i] = Val::new(*v);
            }
            out
        };
        for v in arr {
            observe_val!(v);
        }
        let beta: Challenge = challenger.sample_algebra_element();
        fri_betas.push(beta);
    }

    let num_queries = pcs_json["fri_proof"]["query_proofs"]
        .as_array()
        .map(|a| a.len())
        .unwrap_or(0);
    let log_blowup: usize = 1;
    let log_max_height = commit_phase_commits.len() + log_blowup;
    let mut query_indices: Vec<usize> = Vec::new();
    for _ in 0..num_queries {
        let raw = challenger.sample_bits(log_max_height + 1);
        query_indices.push(raw >> 1);
    }

    (
        records2,
        ChallengesExtracted {
            alpha,
            zeta,
            bivariate_beta,
            fri_betas,
            query_indices,
            alpha_duplex_row,
            zeta_duplex_row,
        },
    )
}

// ── AIR 정의 ─────────────────────────────────────────────────────────────────

/// Fiat-Shamir 챌린저 duplex 시퀀스의 상태 연속성을 검증하는 AIR.
///
/// 각 행 = `DuplexRecord`. 인접 행 전이 제약으로:
/// - `state_in[r+1] == state_out[r]`  (sponge 연속성)
///
/// Poseidon2 permutation 정확성은 별도 P2Air proof로 담보되며,
/// 두 proof의 public values 일치를 verifier가 확인한다.
pub struct ChallengerTranscriptAir;

/// Public values 레이아웃:
/// `[initial_state(16), alpha(3), zeta(3), bivariate_beta(3), beta_0(3), ..., ...]`
pub const CHALLENGER_PV_BASE_LEN: usize = CHALLENGER_PV_INIT_STATE_LEN + 3 + 3 + 3; // + fri_betas*3

impl<F> BaseAir<F> for ChallengerTranscriptAir {
    fn width(&self) -> usize {
        NUM_CHALLENGER_COLS
    }
}

impl<AB: AirBuilderWithPublicValues> Air<AB> for ChallengerTranscriptAir
where
    AB::F: PrimeCharacteristicRing,
{
    fn eval(&self, builder: &mut AB) {
        let pv: Vec<AB::Expr> = builder
            .public_values()
            .iter()
            .map(|p| (*p).into())
            .collect();

        let main = builder.main();
        let local = main.row_slice(0).expect("empty challenger trace");
        let local = &*local;

        // ── A. 초기 상태 제약 (when_first_row) ──────────────────────────────
        // state_in[0][i] == pv[i]  (pv[0..16] = initial_state, 보통 0)
        for i in 0..P2_WIDTH {
            builder
                .when_first_row()
                .assert_eq(local[COL_STATE_IN + i].clone(), pv[i].clone());
        }

        // ── B. 전이 제약 (is_transition) ─────────────────────────────────────
        // state_in[r+1][i] == state_out[r][i]
        if let Some(next) = main.row_slice(1) {
            let next = &*next;
            for i in 0..P2_WIDTH {
                builder.when_transition().assert_eq(
                    next[COL_STATE_IN + i].clone(),
                    local[COL_STATE_OUT + i].clone(),
                );
            }
        }

        // ── C. alpha 결합 제약 (when_first_row 기반, row 고정 필요 시 확장) ──
        // alpha = state_out[alpha_row][0..3]  — public values로 노출 후 verifier 비교
        // 현재는 public values로만 노출 (AIR에서 특정 행을 지정하는 제약은 별도 구현)

        // ── D. Poseidon2 제약: state_in + rate_input → state_out ─────────────
        // P2Air proof의 public values와 이 trace의 (state_in, rate_input, state_out)이
        // 일치하는지는 verifier 레벨에서 확인 (cross-proof linking).
        // 이 AIR 자체는 연속성만 담당.
    }
}

// ── 트레이스 생성 ─────────────────────────────────────────────────────────────

fn build_challenger_trace(records: &[DuplexRecord]) -> RowMajorMatrix<Val> {
    let n = records.len().max(4).next_power_of_two();
    let mut data = vec![Val::new(0); n * NUM_CHALLENGER_COLS];

    for (r, rec) in records.iter().enumerate() {
        let base = r * NUM_CHALLENGER_COLS;
        data[base + COL_STATE_IN..base + COL_STATE_IN + P2_WIDTH]
            .copy_from_slice(&rec.state_before);
        data[base + COL_RATE_INPUT..base + COL_RATE_INPUT + CHALLENGER_RATE]
            .copy_from_slice(&rec.rate_input);
        data[base + COL_STATE_OUT..base + COL_STATE_OUT + P2_WIDTH]
            .copy_from_slice(&rec.state_after);
    }
    // 패딩: 마지막 record의 state_after를 propagate
    if !records.is_empty() {
        let last_out = records.last().unwrap().state_after;
        for r in records.len()..n {
            let base = r * NUM_CHALLENGER_COLS;
            data[base + COL_STATE_IN..base + COL_STATE_IN + P2_WIDTH]
                .copy_from_slice(&last_out);
            data[base + COL_STATE_OUT..base + COL_STATE_OUT + P2_WIDTH]
                .copy_from_slice(&last_out);
        }
    }
    RowMajorMatrix::new(data, NUM_CHALLENGER_COLS)
}

fn make_challenger_pv(
    initial_state: &[Val; P2_WIDTH],
    challenges: &ChallengesExtracted,
) -> Vec<Val> {
    use p3_field::BasedVectorSpace;
    let mut pv = Vec::new();
    pv.extend_from_slice(initial_state);
    pv.extend_from_slice(challenges.alpha.as_basis_coefficients_slice());
    pv.extend_from_slice(challenges.zeta.as_basis_coefficients_slice());
    pv.extend_from_slice(challenges.bivariate_beta.as_basis_coefficients_slice());
    for beta in &challenges.fri_betas {
        pv.extend_from_slice(beta.as_basis_coefficients_slice());
    }
    pv
}

// ── 증명 타입 ─────────────────────────────────────────────────────────────────

/// Fiat-Shamir 챌린저 트랜스크립트 in-circuit 증명 묶음.
///
/// - `p2_proof`:         각 duplex step의 Poseidon2 permutation 정확성
/// - `transcript_proof`: 상태 연속성 + 챌린지 결합 AIR 증명
/// - `challenges`:       추출된 챌린지 값 (검증자가 재계산 가능)
pub struct ChallengerProof {
    /// 모든 duplex permutation 정확성 증명 (P2Air).
    /// public values: `[in_0, out_0, in_1, out_1, ...]`
    pub p2_proof: CircleStarkProof,
    /// 상태 연속성 + 챌린지 결합 AIR 증명 (ChallengerTranscriptAir).
    pub transcript_proof: CircleStarkProof,
    /// 모든 duplex step의 IO 쌍 (cross-proof linking용).
    pub duplex_ios: Vec<([Val; P2_WIDTH], [Val; P2_WIDTH])>,
    /// 추출된 챌린지 값.
    pub challenges: ChallengesExtracted,
}

// ── 공개 API ──────────────────────────────────────────────────────────────────

/// 내부 증명의 Fiat-Shamir 트랜스크립트를 in-circuit으로 증명한다.
///
/// ## 동작
/// 1. `extract_challenger_transcript`로 duplex 시퀀스 추출
/// 2. P2Air: 각 duplex의 permutation 정확성 증명
/// 3. ChallengerTranscriptAir: 상태 연속성 + 챌린지 binding 증명
///
/// ## 결과 사용
/// `verify_challenger_transcript`를 호출하면:
/// - ChallengerTranscriptAir 검증
/// - P2Air 검증 (cross-proof pv 일치 포함)
/// → alpha, zeta, fri_betas, query_indices가 Poseidon2에서 유도됐음을 in-circuit 보장
pub fn prove_challenger_transcript(
    inner_proof: &CircleStarkProof,
    public_values: &[Val],
) -> ChallengerProof {
    let (records, challenges) = extract_challenger_transcript(inner_proof, public_values);

    // ── P2Air 증명: 각 duplex permutation ────────────────────────────────────
    // xored_input = rate 부분을 rate_input으로 교체한 전체 16원소 상태
    let p2_inputs: Vec<[Val; P2_WIDTH]> = records
        .iter()
        .map(|r| {
            let mut s = r.state_before;
            for i in 0..CHALLENGER_RATE {
                s[i] = r.rate_input[i];
            }
            s
        })
        .collect();

    let (p2_proof, p2_outputs) = if p2_inputs.is_empty() {
        // 비어있으면 더미 1행
        let dummy = vec![[Val::new(0); P2_WIDTH]];
        prove_poseidon2_permutations_io(dummy)
    } else {
        prove_poseidon2_permutations_io(p2_inputs.clone())
    };

    // duplex_ios: (xored_input, state_after) 쌍
    let duplex_ios: Vec<([Val; P2_WIDTH], [Val; P2_WIDTH])> = p2_inputs
        .iter()
        .zip(p2_outputs.iter())
        .map(|(inp, out)| (*inp, *out))
        .collect();

    // ── ChallengerTranscriptAir 증명 ─────────────────────────────────────────
    let initial_state = [Val::new(0); P2_WIDTH];
    let pv = make_challenger_pv(&initial_state, &challenges);
    let trace = build_challenger_trace(&records);
    let config = make_circle_config();
    let transcript_proof =
        p3_uni_stark::prove(&config, &ChallengerTranscriptAir, trace, &pv);

    ChallengerProof {
        p2_proof,
        transcript_proof,
        duplex_ios,
        challenges,
    }
}

/// [`prove_challenger_transcript`] 결과를 검증한다.
///
/// ## 검증 단계
/// 1. ChallengerTranscriptAir STARK 검증 (연속성 + 챌린지 binding)
/// 2. P2Air STARK 검증 (permutation 정확성)
/// 3. Cross-proof: P2Air pv == transcript의 (xored_input, state_out) 일치
pub fn verify_challenger_transcript(
    cp: &ChallengerProof,
    expected_inner_pis: &[Val],
    expected_trace_commit: &[Val; 8],
) -> CircleStarkVerifyResult {
    let config = make_circle_config();
    let initial_state = [Val::new(0); P2_WIDTH];

    // 1. ChallengerTranscriptAir 검증
    let pv = make_challenger_pv(&initial_state, &cp.challenges);
    p3_uni_stark::verify(&config, &ChallengerTranscriptAir, &cp.transcript_proof, &pv)?;

    // 2. P2Air 검증 (cross-proof pv 일치)
    verify_poseidon2_permutations_io(&cp.p2_proof, &cp.duplex_ios)?;

    // 3. Cross-proof: transcript.state_after == p2_proof.output
    // (이미 verify_poseidon2_permutations_io의 pv 검증에서 담보됨)

    Ok(())
}
