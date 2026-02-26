# GPU 병렬 증명 설계 계획

## 배경

현재 `zprove`의 ZK 증명 파이프라인은 instruction 단위 또는 배치 단위로
`BatchTransactionZkReceipt`를 생성하며, 내부적으로 Circle STARK (Plonky3, M31)를 사용한다.

### 현재 구조의 제약

- `prove_batch_transaction_zk_receipt`는 N개 instruction의 ProofRow를 **순차 concatenation**하여
  하나의 STARK trace를 만든다.
- 블록 전체(750K instruction)를 하나의 trace로 만들 경우 rows ≈ 45M → 메모리 및 NTT 시간이 선형 증가.
- 병렬화가 instruction 단위 CPU 멀티스레드에만 의존하며, GPU 분산이 불가능하다.

---

## 목표

> EVM 블록 전체 실행(S₀ → S_N)을 **완전 독립적인 window 단위 leaf 증명**으로 분해하고,
> **binary tree 재귀 aggregation**으로 압축하여 GPU 병렬 처리가 가능한 구조로 전환한다.

- Leaf 증명: window별 독립 → GPU core 1개(SM) 당 1 window 병렬 처리
- Aggregation: 인접 증명 간 `s_out(i) == s_in(i+1)` 조건만 LinkAir로 강제
- 최종 증명: `ExecutionReceipt` 하나 — 전체 블록을 커버

---

## 핵심 아이디어: 재귀 상태 연결

```
기존:  S0 → S1 → S2 → ... → SN  (순차 의존 / 하나의 거대 STARK)

신규:
  [S0→S1]  [S1→S2]  [S2→S3]  [S3→S4]   ← GPU 병렬 Leaf 증명
      \      /           \      /
   [S0→S2]             [S2→S4]           ← Aggregation (level 1, 병렬)
          \              /
            [S0→S4]                      ← Aggregation (level 2)
```

각 노드의 public input = `(s_in_commit, s_out_commit)`.
aggregation 노드는 `left.s_out == right.s_in`만 STARK로 강제하면 전체 체인이 증명된다.

---

## 성능 예측 (이 기기 기준: RTX 2060 SUPER)

### 하드웨어

| 항목 | 사양 |
|---|---|
| CPU | AMD Ryzen 5 3600X (6코어 12스레드, boost 4.4GHz) |
| RAM | 11 GiB |
| GPU | NVIDIA GeForce RTX 2060 SUPER (Turing) |
| CUDA 코어 | 2,176개 — SM 34개 |
| VRAM | 8 GiB GDDR6 |
| 메모리 대역폭 | 448 GB/s |
| PCIe | Gen 3 × 16 (실효 ~12 GB/s) |

### 파라미터 (15M gas 이더리움 블록)

| 항목 | 값 |
|---|---|
| 증명 대상 instruction 수 | ~750,000개 |
| 평균 ProofRow 수/instruction | ~60행 |
| leaf window 크기 | 256 instructions (~15,360 rows = 2¹⁴) |
| window 총 수 (N_windows) | 750,000 / 256 ≈ **2,930개** |
| aggregation 깊이 | log₂(2,930) ≈ **12 레벨** |

### Phase 1: Leaf 증명 (GPU 병렬)

window 1개 메모리:
```
256 instr × 60 rows × 30 cols × 4B       = 1.84 MiB (trace)
LDE blowup ×8                             = 14.7 MiB
Merkle 오버헤드 포함                      = ~29 MiB/window
```

GPU 동시 처리 window 수 (VRAM 제한):
```
8,192 MiB / 29 MiB ≈ 282개 → 실 여유 150개
SM 34개 기준 NTT 병렬 → 34개 동시
```

window 1개 GPU 처리 시간:
| 단계 | 시간 |
|---|---|
| NTT (LDE, 30 columns × 2¹⁴) | ~0.5ms |
| Merkle Poseidon2 commit | ~0.8ms |
| 제약 평가 + quotient | ~0.5ms |
| FRI prove | ~0.8ms |
| **window 1개 합계** | **~2.6ms** |

Host→GPU 전송 (PCIe Gen3):
```
2,930 windows × 1.84 MiB = 5.39 GiB
12 GB/s 실효 → 0.45초
```

Leaf 총 시간:
```
GPU 연산: 2,930 / 34 SM ≈ 86 배치 × 2.6ms = 0.22초
PCIe 전송: 0.45초
→ Phase 1 합계: ~0.45초 (PCIe 병목)
```

### Phase 2: Aggregation tree (CPU)

LinkAir = 상태 연결 조건만 체크하는 소형 STARK (4~16행).
CPU 12스레드, 노드당 ~0.8ms:
```
총 노드: 2,929개
12 병렬: 2,929 / 12 × 0.8ms / 레벨당 ≈ 0.35초
```

### 전체 시간 요약

| 단계 | 시간 | 병목 |
|---|---|---|
| Host → GPU 전송 | **0.45초** | PCIe Gen3 |
| GPU Leaf 증명 | **0.22초** | SM 34개 |
| GPU → Host 전송 | **0.05초** | proof 크기 소량 |
| CPU Aggregation (12레벨) | **0.35초** | CPU 12스레드 |
| **합계** | **~1.1초** | PCIe 전송 지배 |

### 환경별 비교

| 환경 | 블록 시간 | 이더리움 12초 대비 |
|---|---|---|
| CPU 12스레드 (현재, GPU 미사용) | ~3.8초 | 32% |
| **RTX 2060 SUPER (이 기기)** | **~1.1초** | **9%** |
| RTX 4090 + PCIe Gen4 | ~0.35초 | 3% |
| A100 SXM (NVLink, PCIe 무관) | ~0.15초 | 1.3% |

---

## 구현 계획

### 추가할 타입 (`transition.rs`)

```rust
/// EVM 상태 스냅샷의 commitment. AIR public input으로 사용.
pub struct StateCommitment {
    pub pc:          usize,
    pub stack_depth: usize,
    pub stack_hash:  [u8; 32],   // Poseidon2(stack[0..sp])
    pub memory_root: [u8; 32],
}

/// Leaf: window 하나(256 instrs)에 대한 독립 증명.
/// public input = (s_in_commit, s_out_commit)
pub struct LeafReceipt {
    pub s_in:        StateCommitment,
    pub s_out:       StateCommitment,
    pub stark_proof: CircleStarkProof,
}

/// Aggregation: 두 인접 증명을 LinkAir로 압축.
/// 조건: left.s_out == right.s_in
pub struct AggregationReceipt {
    pub s_in:       StateCommitment,   // = left.s_in
    pub s_out:      StateCommitment,   // = right.s_out
    pub link_proof: CircleStarkProof,  // LinkAir STARK
    pub left:       Box<ExecutionReceipt>,
    pub right:      Box<ExecutionReceipt>,
}

pub enum ExecutionReceipt {
    Leaf(LeafReceipt),
    Aggregated(AggregationReceipt),
}
```

### 추가할 AIR (`zk_proof.rs` 내부: `LinkAir`)

```
columns: [s_left_out_pc, s_left_out_hash[0..8],
          s_right_in_pc, s_right_in_hash[0..8]]
constraints:
    s_left_out_hash == s_right_in_hash   (연결 조건)
    s_left_out_pc + 1 == s_right_in_pc  (PC 연속성)
public_inputs: [s_in_commit, s_out_commit]  (leaf와 동일 레이아웃)
```

현재 `make_receipt_binding_public_values` 패턴과 동일하게
`pis`에 두 StateCommitment를 직렬화.

### 추가할 함수

```rust
// VmState → StateCommitment (Poseidon2 해시)
fn commit_vm_state(state: &VmState) -> StateCommitment;

// window 1개 → LeafReceipt (현재 prove_batch_transaction_zk_receipt 래핑)
fn prove_leaf(
    itps: &[InstructionTransitionProof],
    s_in: StateCommitment,
    s_out: StateCommitment,
) -> Result<LeafReceipt, String>;

// 인접 두 receipt 연결 (s_out == s_in 강제)
fn link_receipts(
    left: ExecutionReceipt,
    right: ExecutionReceipt,
) -> Result<AggregationReceipt, String>;

// 전체 실행 → window 분할 → GPU 병렬 leaf → binary tree fold
fn prove_execution_chain_parallel(
    itps: Vec<InstructionTransitionProof>,
    states: Vec<VmState>,    // len = itps.len() + 1
    window_size: usize,      // 256 권장
    worker_count: usize,
) -> Result<ExecutionReceipt, String>;

// 재귀 검증
fn verify_execution_receipt(
    receipt: &ExecutionReceipt,
    expected_s_in: &StateCommitment,
    expected_s_out: &StateCommitment,
) -> bool;
```

### `execute.rs` 변경

`ProvingInspector::step()` 에서 이미 스택 스냅샷을 보유 중.
`VmState` 시퀀스를 `proofs`와 함께 `states: Vec<VmState>`로 추가 수집하기만 하면 됨.
추가 EVM 실행 비용 없음.

---

## 구현 순서

| 단계 | 파일 | 작업 |
|---|---|---|
| 1 | `transition.rs` | `StateCommitment` + `commit_vm_state` |
| 2 | `transition.rs` | `LeafReceipt`, `AggregationReceipt`, `ExecutionReceipt` 타입 |
| 3 | `zk_proof.rs` | `LinkAir` + `prove_link_stark` / `verify_link_stark` |
| 4 | `transition.rs` | `prove_leaf`, `link_receipts`, `prove_execution_chain_parallel`, `verify_execution_receipt` |
| 5 | `execute.rs` | `VmState` 시퀀스 수집 + `execute_bytecode_and_prove_chain` 신규 함수 |
| 6 | `zprove-bench` | 새 경로 벤치마크 케이스 추가 |

---

## PCIe 병목 완화 방안 (추후 검토)

1. **window 크기 증가** (256 → 2,048): 전송 횟수 ↓, GPU 점유율 ↑, VRAM 압박 ↑
2. **GPU 상에서 compile_proof 실행**: trace 빌드 자체를 GPU에서 수행하면 Host 메모리를 거치지 않음
3. **Pinned memory + async transfer**: CPU compile_proof와 GPU 전송을 오버랩

가장 단기 효과: window 크기를 2,048으로 올리면 전송 횟수 ÷8,
PCIe 전송 0.45초 → 0.06초로 단축, **전체 블록 시간 ~0.7초** 달성 가능.
