# Soundness Gap 수정 계획

분석 일시: 2026-02-27

## 개요

코드베이스 전체 분석 결과 총 5개의 soundness gap이 발견되었다.  
아래에 심각도 순으로 정리한다.

---

## Gap 1 — `LutKernelAirWithPrep::eval` 미구현 (CRITICAL)

### 위치

`crates/zprove-core/src/zk_proof.rs` — `LutKernelAirWithPrep` impl

### 문제

```rust
impl<AB> Air<AB> for LutKernelAirWithPrep
where AB: AirBuilderWithPublicValues + PairBuilder + AirBuilder<F = Val>,
{
  fn eval(&self, _builder: &mut AB) {
    // Intentionally empty — diagnostic test
  }
}
```

단일 instruction 경로(`prove_lut_with_prep` → `InstructionTransitionZkReceipt`)에서
LUT STARK가 **아무 제약도 검증하지 않는다**.

- 산술 제약 (`U29AddEq`, `ByteMulLow`, 등) 없음
- preprocessed ProofRow 바인딩 없음

배치 경로(`BatchLutKernelAirWithPrep`)는 정상 구현되어 있어 배치 증명은 안전하다.

### 공격 벡터

`verify_instruction_zk_receipt`를 통해 검증하면 완전히 가짜 산술 결과로 구성한
LUT 트레이스도 통과한다.

### 수정

`LutKernelAirWithPrep::eval` 본문에 두 함수 호출을 추가한다.

```rust
fn eval(&self, builder: &mut AB) {
    // 1. Preprocessed ProofRow 바인딩
    {
        let prep = builder.preprocessed();
        let prep_row = prep.row_slice(0)
            .expect("LutKernelAirWithPrep: empty preprocessed trace");
        let prep_row = &*prep_row;
        let main = builder.main();
        let local = main.row_slice(0)
            .expect("LutKernelAirWithPrep: empty main trace");
        let local = &*local;
        eval_lut_prep_row_binding_inner(builder, prep_row, local);
    }

    // 2. Tag 체크 + 단일 instruction pv 바인딩 (pis[1]=opcode, pis[2..10]=wff_digest)
    {
        let pi_opcode: AB::Expr = { let pis = builder.public_values(); pis[1].clone().into() };
        let pi_digest: [AB::Expr; 8] = {
            let pis = builder.public_values();
            std::array::from_fn(|k| pis[2 + k].clone().into())
        };
        let prep_opcode: AB::Expr = {
            let prep = builder.preprocessed();
            let row = prep.row_slice(0).unwrap();
            row[PREP_COL_EVM_OPCODE].clone().into()
        };
        let prep_digest: [AB::Expr; 8] = {
            let prep = builder.preprocessed();
            let row = prep.row_slice(0).unwrap();
            std::array::from_fn(|k| row[PREP_COL_WFF_DIGEST_START + k].clone().into())
        };
        builder.when_first_row().assert_eq(prep_opcode, pi_opcode);
        for k in 0..8_usize {
            builder.when_first_row().assert_eq(prep_digest[k].clone(), pi_digest[k].clone());
        }
    }

    // 3. LUT 산술 제약
    eval_lut_kernel_inner(builder);
}
```

`BatchLutKernelAirWithPrep::eval`과 대칭 구조로 맞추면 된다.

---

## Gap 2 — Memory / Stack / Keccak 일관성 AIR 제약 미비 (CRITICAL)

### 위치

- `MemoryConsistencyAir::eval` — `crates/zprove-core/src/zk_proof.rs` 약 L4137
- `StackConsistencyAir::eval` — 약 L5155
- `KeccakConsistencyAir::eval` — 약 L5380

### 문제

각 AIR의 `eval`이 검증하는 것:

| AIR | 현재 검증 내용 |
|---|---|
| `MemoryConsistencyAir` | 태그 체크 + `is_write ∈ {0,1}` |
| `StackConsistencyAir` | 태그 체크 + `is_push ∈ {0,1}` |
| `KeccakConsistencyAir` | 태그 체크만 |

실제 일관성 검사(`check_claims_and_build_sets`, `check_stack_claims`)는 prover-side
native Rust로만 실행된다. 검증자는 STARK를 통과해도 다음을 보장받지 못한다:

- 메모리: 같은 주소에 write → read 순서 강제, read 값 = 직전 write 값
- 스택: push/pop 쌍의 값 일치, depth 단조성
- Keccak: `output_hash = keccak256(input_bytes)`

### 공격 벡터

조작된 `write_log`/`read_log`가 포함된 `MemoryConsistencyProof`를 수동 구성하면
`verify_memory_consistency`의 STARK 검증은 통과하지만
read-before-write 위반, 자의적 메모리 값 변조가 가능하다.

### 수정 방향

각 AIR에 다음 multi-set 또는 LogUp 제약을 추가한다.

**MemoryConsistencyAir**  
- `(addr, rw_counter, value)` 튜플을 LogUp으로 연결:  
  write 행의 send = read 행의 receive (같은 addr의 최종 write 값을 소비)
- 또는 `rw_counter`에 따른 정렬 + 인접 행 전이 제약

**StackConsistencyAir**  
- push 행과 대응 pop 행의 `(depth_after, value)` 일치 제약  
  (LogUp 또는 permutation argument)

**KeccakConsistencyAir**  
- 각 행에 `keccak256(input_bytes) == output_hash` 체크를 위한
  keccak precompile 전용 lookup 테이블 또는 witness commitment 제약

단기 완화책으로, `verify_memory_consistency` / `verify_stack_consistency`에서
현재의 native Rust 체크를 verifier 측에서도 강제로 재실행하도록 한다  
(현재는 이미 있지만 STARK 검증과의 연결이 끊겨 있음).

---

## Gap 3 — ByteAnd/Or/Xor 결과 링킹 누락 (SERIOUS)

### 위치

`eval_lut_prep_row_binding_inner` 및 `eval_lut_kernel_inner`  
— `crates/zprove-core/src/zk_proof.rs`

### 문제

AND/OR/XOR 연산의 soundness 구조:

1. **LUT STARK**: `out0 = prep[PREP_COL_VALUE]` 바인딩 (prover-committed 값)
2. **byte-table 증명**: `scalar0 op scalar1 = 자체 계산값` LogUp 검증
3. **두 증명 간 링크 없음**: `prep[PREP_COL_VALUE] = scalar0 op scalar1` 을 어느 STARK도 강제하지 않음

`eval_lut_kernel_inner`의 AND 제약:
```rust
// 현재: 구조적 제약만
builder.assert_zero(s_and.clone() * in2.clone());   // in2 = 0
builder.assert_zero(s_and.clone() * out1.clone());  // out1 = 0
// 빠진 것: s_and * (out0 - in0 & in1) = 0  ← 이 제약이 없음
```

### 공격 벡터

AND 연산에서 `out0`을 임의 값으로 설정해도 LUT STARK와 byte-table STARK 각각 통과한다.
두 증명이 실제로 같은 연산을 다루는지 보장이 없다.

### 수정

**방법 A (단기)**: `eval_lut_kernel_inner`에 byte-op 직접 제약 추가.
AND/OR/XOR는 M31 필드에서 다항식으로 표현 불가하므로 범위를 제한한 후
LogUp으로 처리하거나, verifier에서 out-of-circuit 체크를 강제한다.

**방법 B (권장)**: `verify_lut_with_prep_and_logup`에서 검증 후
`collect_byte_table_queries_from_rows`로 쿼리를 재구성하여
byte-table의 `result` 값과 `prep[PREP_COL_VALUE]`가 일치하는지
out-of-circuit으로 확인하는 단계를 추가한다.

```rust
pub fn verify_lut_with_prep_and_logup(...) -> CircleStarkVerifyResult {
    verify_lut_with_prep(lut_proof, prep_vk, public_values)?;

    if let Some(bp) = byte_proof {
        crate::byte_table::verify_byte_table(bp)?;
    }

    // 추가: byte-table result가 prep committed value와 일치하는지 확인
    // (out-of-circuit, verifier가 prep commitment에서 재구성)
    verify_byte_op_result_consistency(prep_vk, byte_proof)?;

    Ok(())
}
```

---

## Gap 4 — U29/U24AddEq 입력 범위 제약 없음 (SERIOUS)

### 위치

`eval_lut_kernel_inner` — `crates/zprove-core/src/zk_proof.rs`

### 문제

U29AddEq 제약:
```rust
// in0 + in1 + in2 = out0 + 2^29 * out1
// out1 ∈ {0,1}
```

M31 = 2^31 − 1 필드에서 `in0 ≥ 2^29` 인 값을 넣어도 동일 방정식이 성립될 수 있다.
`validate_lut_steps`의 범위 체크는 prover 경로에서만 실행되며 AIR 제약에는 포함되어 있지 않다.

### 수정

각 operand에 대한 range-check 제약을 AIR에 추가한다.  
Mersenne31 환경에서는 다음 방식을 고려한다:

**방법 A**: 각 29-bit 입력을 8-bit 청크 3개 + 5-bit 청크 1개로 분해하여
byte-table lookup으로 범위 강제 (기존 byte_table 인프라 재활용).

**방법 B**: `in0 ≤ 2^29 - 1` 을 보조 witness `overflow_bit`와 함께  
`in0 + overflow_bit * 2^29 < 2^30` 형태로 2단계로 강제한다.

U24AddEq, U15AddEq도 동일한 방식으로 처리한다.

---

## Gap 5 — Sub-call / 외부 상태 오라클 위트니스 (DESIGN)

### 위치

`SubCallClaim` — `crates/zprove-core/src/transition.rs`  
`ExternalStateClaim`, `CallContextClaim`, `KeccakClaim`

### 문제

```rust
pub struct SubCallClaim {
    ...
    pub inner_proof: Option<Box<TransactionProof>>,  // Level-0: None → oracle
}
```

- CALL / CREATE / DELEGATECALL 결과를 `SubCallClaim` oracle로 받아들임
- `inner_proof = None` 이면 callee 실행 전체를 신뢰
- BALANCE, BLOCKHASH, EXTCODESIZE 등 외부 상태 쿼리도 oracle (검증 없음)

### 현재 수용 가능한 이유

Level-0 oracle witness는 설계상 인지된 한계이며,  
서브콜이 포함된 트랜잭션을 ZK로 완전 보장하려면 recursive proof가 필요하다.  
단일 컨트랙트 / 단일 tx 환경에서는 영향 없음.

### 수정 방향 (장기)

1. `inner_proof: Some(...)` 경로(Level-1 재귀 검증)를 `verify_proof`에서 실제로 검증
2. 외부 상태에 대해 state commitment (Merkle proof 등)를 `ExternalStateClaim`에 포함
3. `SubCallClaim` 없는 트랜잭션에만 현재 ZK 보장 범위를 명시적으로 제한

---

## 수정 우선순위 요약

| 우선순위 | Gap | 영향 경로 | 예상 난이도 |
|---|---|---|---|
| 1 | `LutKernelAirWithPrep::eval` 미구현 | 단일 instruction zk receipt | 낮음 (코드 복사 수준) |
| 2 | Memory/Stack/Keccak AIR 제약 미비 | 모든 메모리/스택 ops | 높음 (LogUp/perm 설계 필요) |
| 3 | AND/OR/XOR 결과 링킹 | bitwise 연산 전체 | 중간 |
| 4 | 범위 제약 누락 | 산술 연산 전체 | 중간 |
| 5 | Sub-call oracle | CALL/CREATE 포함 tx | 높음 (recursive proof) |

Gap 1은 즉시 수정 가능하며 반드시 먼저 처리해야 한다.  
Gap 2-4는 AIR 재설계가 필요하므로 별도 브랜치에서 진행을 권장한다.  
Gap 5는 장기 로드맵 항목으로 관리한다.
