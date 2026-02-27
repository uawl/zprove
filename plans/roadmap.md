# zprove 개발 로드맵

> 마지막 갱신: 2025-02  
> 테스트 현황: **344개 통과 / 0 실패**  
> 단일 컨트랙트 TX 기준 완성도: **~90%**  
> CALL/CREATE 포함 TX 기준 완성도: **~40–50%**

---

## 1. 완료된 항목

### 1-1. Soundness Gap 수정 (Gap 1–4)

| Gap | 증상 | 수정 내용 |
|-----|------|-----------|
| **Gap 1** | `LutKernelAirWithPrep::eval` 빈 구현 | `LutKernelAirWithPrep` 구조체 제거. `BatchLutKernelAirWithPrep`으로 단일/배치 통합 |
| **Gap 2** | Memory/Stack/Keccak AIR 제약 불완전 | `derive_sets_from_logs` 독립 재구성 + LogUp 멀티셋 + Keccak native Rust 재검증 |
| **Gap 3** | AND/OR/XOR 결과 링킹 누락 | `validate_manifest_rows`에서 `row.value == scalar0 op scalar1` 검증 추가 |
| **Gap 4** | U29/U24 범위 제약 누락 | `validate_manifest_rows`에서 U29≤2²⁹−1, U24≤2²⁴−1 비트 폭 검증 추가 |

**핵심 AIR**: `MemoryConsistencyAir`, `StackConsistencyAir`, `KeccakConsistencyAir` — 각각 tag + is_write/is_push boolean + LogUp 멀티셋 멤버십 제약 적용 중.

### 1-2. Opt-5: StackIR 컬럼 수 최적화 (27 → 9)

계획(27 → 15, LogUp 테이블)을 초과 달성. **전처리 commitment** 방식으로 27 → 9 컬럼 구현.

| 컬럼 | 설명 |
|------|------|
| `STACK_COL_OP` | 연산 종류 |
| `STACK_COL_ARG0/1/2` | 입력 인수 |
| `STACK_COL_SCALAR0/1/2` | 스칼라 보조값 |
| `STACK_COL_VALUE` | 결과 값 |
| `STACK_COL_RET_TY` | 반환 타입 |

`eval_stack_ir_inner`: 태그(타입) 체크 1개 제약만 존재. one-hot 셀렉터 컬럼 완전 제거.  
전처리 행(preprocessed row)과 1:1 등식 제약으로 대체.  
`NUM_STACK_IR_COLS = 9`, `NUM_LUT_COLS = 16` (현재 코드 기준).

### 1-3. 검증 파이프라인 (`verify_batch_transaction_zk_receipt`)

9단계 완전 구현:
1. tx hash 검증
2. receipt binding public values 검증
3. memory consistency STARK 검증
4. stack consistency STARK 검증
5. storage consistency STARK 검증
6. keccak consistency STARK 검증
7. batch LUT STARK 검증
8. StackIR STARK 검증
9. keccak↔memory cross-check (`validate_keccak_memory_cross_check`)

---

## 2. 미완료 항목 (단기)

### 2-1. Gap 5: SubCall 재귀 증명

**파일**: `crates/zprove-core/src/transition.rs`  
**현상태**: `SubCallClaim.inner_proof: Option<Box<TransactionProof>>` — oracle 상태 (단순 중첩 저장, STARK 미적용)

**목표**: CALL/CREATE 내부 실행을 완전한 `TransactionProof`로 귀납 증명.  
검증자는 외부 TX proof 검증 시 inner_proof를 재귀적으로 검증해야 함.

**작업**:
- [ ] `verify_batch_transaction_zk_receipt`에 재귀 inner_proof 검증 로직 추가
- [ ] `SubCallClaim` Keccak/Memory 경계 linking (외부↔내부 메모리 스냅샷 일치)
- [ ] 재귀 깊이 상한 및 가스 소비 제약

---

### 2-2. Gap 6: `aggregate_proofs_tree` → `MergeAir` STARK

**파일**: `crates/zprove-core/src/zk_proof.rs`  
**현상태**: `aggregate_proofs_tree` — native Rust로 단순 결합, STARK 미적용

**목표**: 여러 LeafReceipt를 binary tree로 압축하는 `MergeAir` STARK 구현.  
`verify_execution_receipt` 로 재귀 검증 가능.

**설계**: `LinkAir` (소형 STARK, 4–16행):
```
columns: [s_left_out_pc, s_left_out_hash[0..8],
          s_right_in_pc, s_right_in_hash[0..8]]
constraints:
    s_left_out_hash == s_right_in_hash   (상태 연결)
    s_left_out_pc + 1 == s_right_in_pc  (PC 연속성)
public_inputs: [s_in_commit, s_out_commit]
```

**작업**:
- [ ] `LinkAir` 구현 (`zk_proof.rs`)
- [ ] `prove_link_stark` / `verify_link_stark` 함수 추가
- [ ] `AggregationReceipt`, `ExecutionReceipt` enum 타입 정의 (`transition.rs`)
- [ ] `aggregate_proofs_tree` → STARK 기반으로 교체

---

## 3. 미완료 항목 (장기)

### 3-1. GPU 병렬 증명

**타겟 하드웨어**: RTX 2060 SUPER (VRAM 8GiB, SM 34개) + Ryzen 5 3600X (12스레드)

**설계 개요**:

```
전체 실행 (750,000 instr, ~45M rows)
    ↓ window 분할 (256 instr = ~15,360 rows = 2¹⁴)
[Leaf₀] [Leaf₁] ... [Leaf₂₉₂₉]   ← GPU 34 SM 동시 병렬
    ↓ binary tree 집계 (12 레벨, LinkAir)
[Agg₀₋₁] [Agg₂₋₃] ...           ← CPU 12스레드
    ↓
ExecutionReceipt (단일 root proof)
```

**성능 예측 (15M gas 블록 기준)**:

| 단계 | 시간 | 병목 |
|------|------|------|
| Host → GPU 전송 (PCIe Gen3) | 0.45 s | PCIe Gen3 |
| GPU Leaf 증명 (34 SM) | 0.22 s | SM 할당 |
| GPU → Host 전송 | 0.05 s | — |
| CPU Aggregation (12 레벨) | 0.35 s | CPU 12스레드 |
| **합계** | **~1.1 s** | PCIe 전송 지배 |

현재 CPU 전용 시간 ~3.8 s 대비 약 3.5× 가속, 이더리움 블록 간격 12 s의 9%.

**window 크기 2,048으로 올리면**: PCIe 전송 0.45 s → 0.06 s, 전체 ~0.7 s 달성 가능.

**새로 추가할 타입**:
```rust
pub struct StateCommitment {
    pub pc:          usize,
    pub stack_depth: usize,
    pub stack_hash:  [u8; 32],   // Poseidon2(stack[0..sp])
    pub memory_root: [u8; 32],
}

pub struct LeafReceipt {
    pub s_in:        StateCommitment,
    pub s_out:       StateCommitment,
    pub stark_proof: CircleStarkProof,
}

pub struct AggregationReceipt {
    pub s_in:       StateCommitment,
    pub s_out:      StateCommitment,
    pub link_proof: CircleStarkProof,
    pub left:       Box<ExecutionReceipt>,
    pub right:      Box<ExecutionReceipt>,
}

pub enum ExecutionReceipt {
    Leaf(LeafReceipt),
    Aggregated(AggregationReceipt),
}
```

**구현 순서**:

| 단계 | 파일 | 작업 |
|------|------|------|
| 1 | `transition.rs` | `StateCommitment` + `commit_vm_state` |
| 2 | `transition.rs` | `LeafReceipt`, `AggregationReceipt`, `ExecutionReceipt` 타입 |
| 3 | `zk_proof.rs` | `LinkAir` + `prove_link_stark` / `verify_link_stark` |
| 4 | `transition.rs` | `prove_leaf`, `link_receipts`, `prove_execution_chain_parallel`, `verify_execution_receipt` |
| 5 | `execute.rs` | `VmState` 시퀀스 수집 + `execute_bytecode_and_prove_chain` 신규 함수 |
| 6 | `zprove-bench` | 새 경로 벤치마크 케이스 추가 |

**PCIe 병목 완화 방안 (추후 검토)**:
- window 크기 256 → 2,048: 전송 횟수 ÷8
- GPU 상에서 trace 빌드 자체 실행 (Host 메모리 생략)
- Pinned memory + async transfer: compile_proof와 GPU 전송 오버랩

---

## 4. 현황 요약

| 항목 | 상태 |
|------|------|
| Gap 1: BatchLutKernelAirWithPrep 통합 | ✅ 완료 |
| Gap 2: Memory/Stack/Keccak LogUp 제약 | ✅ 완료 |
| Gap 3: AND/OR/XOR 결과 링킹 | ✅ 완료 |
| Gap 4: U29/U24 범위 제약 | ✅ 완료 |
| Opt-5: StackIR 컬럼 27 → 9 | ✅ 완료 (계획 초과 달성) |
| 9단계 검증 파이프라인 | ✅ 완료 |
| Gap 5: SubCall 재귀 증명 | 🔴 미완료 |
| Gap 6: MergeAir 집계 STARK | 🔴 미완료 |
| GPU 병렬 증명 (LinkAir + 병렬화) | 🔴 미완료 |
| 전이 제약 (ordering/value-continuity) | 🔴 미완료 |

**테스트**: 344개 전체 통과. CALL/CREATE 포함 TX는 inner_proof oracle 상태로 end-to-end 증명 미완성.
