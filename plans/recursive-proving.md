# ZK 재귀 증명 계획

> 작성일: 2026-02-27  
> 대상 브랜치: `main`  
> 현재 테스트 현황: 344개 통과 / 0 실패

---

## 1. 현재 상태 및 문제점

| 구성 요소 | 현재 상태 | 문제점 |
|-----------|-----------|--------|
| `SubCallClaim.inner_proof` | oracle 저장만, STARK 미검증 | CALL/CREATE 내부 실행 증명 불가 |
| `aggregate_proofs_tree` | native Rust read/write set 교차 검사 | STARK 보장 없음 — 집계자를 신뢰해야 함 |
| `LinkAir` / `MergeAir` | 주석·설계만 존재 | 미구현 |
| 긴 트랜잭션 분할 | 단일 배치로만 처리 | 수백만 row는 메모리 초과 |
| 블록 전체 증명 | 없음 | 여러 TX를 하나의 루트 proof로 묶을 수단 없음 |

---

## 2. 재귀 증명의 세 가지 레이어

```
Layer 3 (장기):  StarkVerifierAir — 자식 STARK 증명을 회로 내부에서 검증
                  ↑ 진정한 ZK 재귀 (4–6개월)
Layer 2 (중기):  LinkAir — 세그먼트 경계 StateCommitment를 STARK로 링킹
                  ↑ 커밋먼트 체인 접근 (6–8주)
Layer 1 (단기):  SubCall inner_proof 재귀 검증 — 기존 파이프라인 확장
                  ↑ 가장 빠른 soundness 향상 (1–2주)
```

Layer 1·2는 **외부에서 자식 proof를 네이티브 검증**하고, 그 결과를 
STARK public input으로 연결하는 *commitment-chain* 방식이다.  
Layer 3에서 비로소 자식 검증 로직 자체가 AIR 제약 안에 포함된다.

---

## 3. Phase 1 — SubCall 재귀 검증 (Gap 5 해결)

**목표**: `inner_proof`가 `Some`일 때 `verify_batch_transaction_zk_receipt`를 재귀 호출해 callee 실행을 검증한다.

### 3-1. `SubCallClaim`에 재귀 깊이 필드 추가

**파일**: `crates/zprove-core/src/transition.rs`

```rust
pub struct SubCallClaim {
    pub opcode:       u8,
    pub callee:       [u8; 20],
    pub value:        [u8; 32],
    pub return_data:  Vec<u8>,
    pub success:      bool,
    /// EVM 재귀 깊이 (0 = 최상위 TX, 최대 1023).
    pub depth:        u16,
    /// Level-1+ 재귀 증명 (None = oracle 허용).
    pub inner_proof:  Option<Box<TransactionProof>>,
}

/// EVM 최대 CALL 중첩 깊이.
pub const MAX_CALL_DEPTH: u16 = 1024;
```

### 3-2. `verify_sub_call_claim` 신규 함수

**파일**: `crates/zprove-core/src/transition.rs` (또는 `zk_proof.rs`)

```rust
/// SubCallClaim 재귀 검증.
///
/// 1. depth 상한 확인 (depth < MAX_CALL_DEPTH).
/// 2. inner_proof が Some 이면 verify_batch_transaction_zk_receipt 재귀 호출.
/// 3. callee의 return_data == caller SubCallClaim.return_data 바이트 일치.
/// 4. callee의 마지막 스택 depth == 0 (새 호출 프레임).
pub fn verify_sub_call_claim(
    claim:               &SubCallClaim,
    caller_mem_proof:    &MemoryConsistencyProof,
) -> Result<(), String>
```

### 3-3. `verify_batch_transaction_zk_receipt` Step 10 추가

**파일**: `crates/zprove-core/src/transition.rs`  
현재 9단계 끝에 추가:

```rust
// 10. SubCall inner_proof 재귀 검증.
for stmt in statements {
    if let Some(sc) = &stmt.sub_call_claim {
        if let Some(inner) = &sc.inner_proof {
            if sc.depth >= MAX_CALL_DEPTH {
                return false;
            }
            // inner TX의 모든 단계를 재귀 검증
            let inner_stmts = collect_inner_statements(inner);
            let inner_receipt = collect_inner_receipt(inner);
            if !verify_batch_transaction_zk_receipt(&inner_stmts, &inner_receipt) {
                return false;
            }
            // return_data 일치 검사
            if !verify_return_data_binding(sc, inner) {
                return false;
            }
        }
    }
}
```

### 3-4. Phase 1 작업 목록

- [ ] `SubCallClaim`에 `depth: u16` 추가 + `MAX_CALL_DEPTH = 1024` 상수
- [ ] `verify_sub_call_claim` 구현 (depth 검사 + 재귀 호출 + return_data 일치)
- [ ] `verify_batch_transaction_zk_receipt` Step 10 삽입
- [ ] `prove_batch_transaction_zk_receipt`에서 `depth` 전파 로직 추가
- [ ] Gap 5 테스트 케이스: CALL → callee 정상/리버트 케이스 2개

---

## 4. Phase 2 — 세그먼트 분할 + LinkAir STARK

### 4-1. StateCommitment 타입

**파일**: `crates/zprove-core/src/transition.rs`

```rust
use crate::zk_proof::Val;

/// 세그먼트 경계에서의 VM 상태 커밋먼트.
///
/// Poseidon2 해시로 압축되므로 STARK public input (각 8개 M31 원소)에 담을 수 있다.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateCommitment {
    /// 프로그램 카운터.
    pub pc:          u32,
    /// 스택 포인터 (depth).
    pub sp:          u32,
    /// 남은 가스.
    pub gas_left:    u64,
    /// Poseidon2(stack[0..sp] concat으로 정렬된 32-byte words).
    /// 길이 8의 M31 배열.
    pub stack_hash:  [Val; 8],
    /// Poseidon2(write_set 주소 오름차순 정렬 후 [addr || value] 직렬화).
    /// 길이 8의 M31 배열.
    pub memory_root: [Val; 8],
}

/// `VmState`로부터 `StateCommitment`를 계산한다.
///
/// Poseidon2 sponge는 `zk_proof::poseidon2_hash_m31_words`를 재사용한다.
pub fn commit_vm_state(s: &VmState, gas_left: u64) -> StateCommitment
```

`commit_vm_state` 구현 노트:
- `stack_hash`: `s.stack[0..s.sp]`를 4-byte limb로 펼쳐 Poseidon2 sponge 흡수
- `memory_root`: `s.memory_root`를 그대로 Poseidon2에 흡수 (이미 해시된 값)
- 결과를 `[Val; 8]`로 반환 (기존 `poseidon2_hash_m31_words` 시그니처 참조)

### 4-2. 새 Receipt 타입 계층

**파일**: `crates/zprove-core/src/transition.rs`

```rust
/// 단일 세그먼트 (≤ window_size 명령어)의 증명.
#[derive(Debug, Clone)]
pub struct LeafReceipt {
    pub s_in:         StateCommitment,
    pub s_out:        StateCommitment,
    pub batch_receipt: BatchTransactionZkReceipt,
}

/// 두 `ExecutionReceipt`를 하나로 병합한 집계 증명.
#[derive(Debug, Clone)]
pub struct AggregationReceipt {
    pub s_in:        StateCommitment,
    pub s_out:       StateCommitment,
    /// LinkAir STARK proof — s_left.s_out == s_right.s_in 을 증명.
    pub link_proof:  CircleStarkProof,
    pub left:        Box<ExecutionReceipt>,
    pub right:       Box<ExecutionReceipt>,
}

/// 실행 증명 트리의 노드.
#[derive(Debug, Clone)]
pub enum ExecutionReceipt {
    Leaf(LeafReceipt),
    Aggregated(AggregationReceipt),
}
```

### 4-3. LinkAir STARK 설계

**목적**: 인접한 두 세그먼트의 경계 상태가 `s_left.s_out == s_right.s_in`임을 STARK로 증명

**파일**: `crates/zprove-core/src/zk_proof.rs`

#### Trace 레이아웃

행 1개 = 링크 쌍 1개 (binary tree 한 노드)

```
열 번호  이름                 설명
  0      left_pc              left 세그먼트 출구 PC
  1      left_sp              left 세그먼트 출구 SP
  2..10  left_stack_hash[8]   left 세그먼트 출구 스택 해시 (M31 × 8)
  10..18 left_mem_root[8]     left 세그먼트 출구 메모리 루트 (M31 × 8)
  18     right_pc             right 세그먼트 입구 PC
  19     right_sp             right 세그먼트 입구 SP
  20..28 right_stack_hash[8]  right 세그먼트 입구 스택 해시 (M31 × 8)
  28..36 right_mem_root[8]    right 세그먼트 입구 메모리 루트 (M31 × 8)
  총 36열
```

#### AIR 제약

```
연속성(18개):
  ∀ i ∈ 0..8: left_stack_hash[i] == right_stack_hash[i]
  ∀ i ∈ 0..8: left_mem_root[i]   == right_mem_root[i]
PC·SP 연속성(2개):
  left_pc == right_pc    (세그먼트 분할 지점에서 PC 동일)
  left_sp == right_sp    (스택 깊이 보존)
총 20개 제약 (모두 차수 2 이하, is_transition 불필요)
```

#### Public inputs

```
[in_pc, in_sp, in_stack_hash[8], in_mem_root[8],
 out_pc, out_sp, out_stack_hash[8], out_mem_root[8]]
= 총 20개 M31 원소
```

**LinkAir 구현 스켈레톤**:

```rust
pub struct LinkAir;

impl BaseAir<Val> for LinkAir {
    fn width(&self) -> usize { 36 }
}

impl<AB: AirBuilderWithPublicValues<F = Val>> Air<AB> for LinkAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0);

        // left_stack_hash[i] == right_stack_hash[i]
        for i in 0..8 {
            builder.assert_eq(row[2 + i], row[20 + i]);
        }
        // left_mem_root[i] == right_mem_root[i]
        for i in 0..8 {
            builder.assert_eq(row[10 + i], row[28 + i]);
        }
        // PC·SP 연속성
        builder.assert_eq(row[0], row[18]); // pc
        builder.assert_eq(row[1], row[19]); // sp
    }
}

pub fn prove_link_stark(
    links: &[(StateCommitment, StateCommitment)], // (left_out, right_in) 쌍
) -> Result<CircleStarkProof, String>

pub fn verify_link_stark(
    proof:  &CircleStarkProof,
    s_in:   &StateCommitment,
    s_out:  &StateCommitment,
) -> CircleStarkVerifyResult
```

### 4-4. prove_execution_chain

**파일**: `crates/zprove-core/src/transition.rs`

```rust
/// 트랜잭션 실행 전체를 window 단위로 분할·증명하고
/// binary tree LinkAir로 집계한다.
///
/// # 인자
/// - `vm_state_seq`: 실행 중 수집한 VmState 스냅샷 (window 경계마다 1개)
/// - `receipts`:     window별 BatchTransactionZkReceipt (기존 경로 재사용)
/// - `window_size`:  한 Leaf 세그먼트의 최대 명령어 수 (권장 256)
///
/// # 반환
/// - 루트 `ExecutionReceipt` (Leaf 또는 Aggregated)
pub fn prove_execution_chain(
    vm_state_seq: &[VmState],
    gas_seq:      &[u64],
    receipts:     Vec<BatchTransactionZkReceipt>,
    window_size:  usize,
) -> Result<ExecutionReceipt, String>

/// ExecutionReceipt 트리를 루트부터 재귀적으로 검증한다.
///
/// 각 Aggregated 노드에서 verify_link_stark를 호출하고,
/// 각 Leaf 노드에서 verify_batch_transaction_zk_receipt를 호출한다.
pub fn verify_execution_receipt(
    receipt: &ExecutionReceipt,
) -> Result<(), String>
```

내부 집계 흐름:
```
Leaf 노드 생성 (각 window):
  commit_vm_state(vm_state_seq[i])   → LeafReceipt.s_in
  commit_vm_state(vm_state_seq[i+1]) → LeafReceipt.s_out

Binary tree 집계 (level-by-level):
  while nodes.len() > 1:
    pair (left, right) 묶기
    prove_link_stark([(left.s_out, right.s_in)])
    → AggregationReceipt { link_proof, left, right }
```

### 4-5. execute.rs 통합

**파일**: `crates/zprove-core/src/execute.rs`

`execute_bytecode_and_prove_batch` 에서 window 경계마다 `VmState` 스냅샷 수집 후 `prove_execution_chain` 호출하는 새 함수 `execute_bytecode_and_prove_chain` 추가.

### 4-6. Phase 2 작업 목록

- [ ] `StateCommitment` + `commit_vm_state` 구현
- [ ] `LeafReceipt`, `AggregationReceipt`, `ExecutionReceipt` 타입 정의
- [ ] `LinkAir` (36열, 20 제약) + `prove_link_stark` / `verify_link_stark`
- [ ] `prove_execution_chain` — window 분할 + binary tree 집계
- [ ] `verify_execution_receipt` — 재귀 검증자
- [ ] `execute_bytecode_and_prove_chain` — VmState 수집 통합
- [ ] `zprove-bench`에 체인 증명 벤치마크 케이스 추가
- [ ] Phase 2 통합 테스트: 256/512/1024 명령어 체인 케이스

---

## 5. Phase 3 — 진정한 STARK 재귀 (장기)

이 단계에서 `LinkAir`가 자식 STARK 증명을 **회로(AIR) 내부에서 검증**하는 
완전한 재귀 구조로 전환된다.

### 5-1. 필요한 하위 회로

Circle STARK (M31 필드, Poseidon2 해시) 검증자를 AIR로 구현하려면 다음 서브서킷이 필요하다.

| 회로 이름 | 역할 | 주요 제약 |
|-----------|------|-----------|
| `Poseidon2Air` | M31×16 치환 1회 | ~2,400 (full round) |
| `MerklePathAir(h)` | 높이 h Merkle 경로 검증 | `h × 2 × Poseidon2` |
| `M31Ext3MulAir` | GF(M31³) 곱셈 | ~30 |
| `FriQueryAir` | FRI fold 1단계 | ~50 + M31Ext3Mul |
| `FriVerifierAir` | FRI 전체 (k 쿼리 × log₂n fold) | k × log₂n × FriQueryAir |
| `OodVerifierAir` | OOD 평가 등식 확인 | AIR 너비에 비례 |
| `StarkVerifierAir` | 위 조합 (AIR + PCS 검증) | 위 합산 |

**필드 선택 이점**: Circle STARK가 M31을 사용하므로 검증자 회로도 M31에서 실행 가능. 필드 전환(field embedding) 오버헤드 없음.

### 5-2. Poseidon2Air 설계

Plonky3 `Poseidon2Mersenne31<16>` 의 한 치환 = 내부 round × 8 + 외부 round × 8.

```
Trace: 16행 (각 round) × (16 state + 16 after-linear-layer) = 32열
제약: 각 행마다 S-box(x^5) + MDS 행렬 = ~150 제약 / round
총: (8+8) round × 150 ≈ 2,400 제약
Public input: [state_in[0..16], state_out[0..16]]
```

### 5-3. RecursiveStarkProof 타입

```rust
/// 재귀 STARK 증명.
/// `outer_proof` 는 StarkVerifierAir 실행 결과이며,
/// inner_pis_hash 는 자식 proof의 public inputs를 압축한 것이다.
#[derive(Debug, Clone)]
pub struct RecursiveStarkProof {
    /// 검증된 자식 public inputs의 Poseidon2 해시 (8 × M31).
    pub inner_pis_hash:  [Val; 8],
    /// 외부 StarkVerifierAir 증명.
    pub outer_proof:     CircleStarkProof,
}
```

### 5-4. 기술적 주의사항

1. **OodEvaluationMismatch gotcha** (기존 문서 참조): `is_transition()` 을 재귀 AIR에서 사용하지 않는다. 모든 "enable 조건"은 별도 selector 열로 처리.

2. **FRI 파라미터 고정**: 재귀 레이어마다 동일한 BLOWUP_FACTOR / NUM_QUERIES 를 사용해 검증자 회로 크기를 일정하게 유지.

3. **증명 크기**: 레이어마다 outer_proof가 생성되지만 inner tree는 버려도 됨. 최종 검증자는 루트 `RecursiveStarkProof` 하나만 확인.

4. **트레이드오프 — Phase 2 vs 3**:
   - Phase 2 (commitment chain): 검증자가 각 세그먼트 proof도 보유해야 함 (O(N) proof 크기)
   - Phase 3 (true recursion): 루트 proof 1개만으로 전체 검증 (O(log N) 크기, 그러나 증명 시간은 N × StarkVerifierAir)

### 5-5. Phase 3 작업 목록

- [ ] `Poseidon2Air` 구현 및 단위 테스트
- [ ] `MerklePathAir` 구현 및 단위 테스트
- [ ] `M31Ext3MulAir` 구현
- [ ] `FriQueryAir` 구현
- [ ] `FriVerifierAir` 통합 및 테스트
- [ ] `StarkVerifierAir` 통합 (LinkAir 내부에서 자식 proof 검증)
- [ ] `RecursiveStarkProof` 타입 + end-to-end 테스트

---

## 6. 전체 구현 마일스톤

| 단계 | 파일 | 핵심 작업 | 우선순위 | 예상 소요 |
|------|------|-----------|----------|-----------|
| 1a | `transition.rs` | `SubCallClaim.depth` + `MAX_CALL_DEPTH` | **P0** | 0.5일 |
| 1b | `transition.rs` | `verify_sub_call_claim` 구현 | **P0** | 1일 |
| 1c | `transition.rs` | `verify_batch_transaction_zk_receipt` Step 10 | **P0** | 0.5일 |
| 1d | `tests/` | CALL 재귀 테스트 케이스 2개 | **P0** | 1일 |
| 2a | `transition.rs` | `StateCommitment` + `commit_vm_state` | **P1** | 1일 |
| 2b | `transition.rs` | `LeafReceipt`, `AggregationReceipt`, `ExecutionReceipt` | **P1** | 0.5일 |
| 2c | `zk_proof.rs` | `LinkAir` (36열, 20제약) | **P1** | 2일 |
| 2d | `zk_proof.rs` | `prove_link_stark` / `verify_link_stark` | **P1** | 1일 |
| 2e | `transition.rs` | `prove_execution_chain` + `verify_execution_receipt` | **P1** | 2일 |
| 2f | `execute.rs` | `execute_bytecode_and_prove_chain` | **P1** | 1일 |
| 2g | `zprove-bench` | 체인 증명 벤치마크 | **P2** | 1일 |
| 3a | `zk_proof.rs` | `Poseidon2Air` 회로 | **P3** | 5일 |
| 3b | 신규 파일 | `MerklePathAir` | **P3** | 3일 |
| 3c | 신규 파일 | `FriQueryAir` + `FriVerifierAir` | **P3** | 7일 |
| 3d | 신규 파일 | `StarkVerifierAir` 통합 | **P3** | 5일 |

---

## 7. 보안 고려사항

### 7-1. Depth overflow
`MAX_CALL_DEPTH = 1024` 체크를 검증 측에서도 반드시 수행. 증명자가 `depth` 필드를 조작해 제한을 우회하지 못하도록 `depth` 값을 public input으로 포함.

### 7-2. Phase 2 Soundness
Phase 2의 `LinkAir`는 *상태 해시 일치*만 확인하고 자식 STARK를 회로 내부에서 검증하지 않는다. 따라서:
- 검증자는 반드시 각 `LeafReceipt.batch_receipt`도 `verify_batch_transaction_zk_receipt`로 검증해야 함
- `verify_execution_receipt`는 트리를 DFS로 순회하며 **모든 leaf·link proof를 검증**

### 7-3. StateCommitment 도메인 분리
`stack_hash`와 `memory_root` Poseidon2 흡수 시 도메인 태그를 prefix로 포함해 두 해시 간 혼용 공격(collision) 방지:
```rust
const DOMAIN_STACK_HASH:  Val = Val::from_u32(0x53544B5F); // "STK_"
const DOMAIN_MEMORY_ROOT: Val = Val::from_u32(0x4D454D5F); // "MEM_"
```

### 7-4. Phase 3 Soundness
진정한 재귀에서는 `StarkVerifierAir` 자체의 soundness에 의존.  
- FRI soundness error: `ε ≤ (blowup_factor)^{-num_queries}`
- 재귀 레이어마다 동일한 파라미터를 사용해 누적 오류가 `O(depth × ε)`에 머물도록 설정

---

## 8. 현황 업데이트 (로드맵 대비)

| 항목 | 로드맵 상태 | 이 계획 단계 |
|------|-------------|-------------|
| Gap 5: SubCall 재귀 증명 | 🔴 미완료 | Phase 1 (P0) |
| Gap 6: MergeAir 집계 STARK | 🔴 미완료 | Phase 2 LinkAir (P1) |
| GPU 병렬 증명 LeafReceipt | 🔴 미완료 | Phase 2 LeafReceipt 타입 선행 필요 |
| 진정한 STARK 재귀 | 🔴 (신규) | Phase 3 (P3, 장기) |
