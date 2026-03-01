# R/W 로그 기반 상태 전이 증명 설계

## 핵심 아이디어

**쓰기 로그의 마지막 항목 = 최종 상태**

메모리/스토리지의 최종 값은 각 주소에 대한 R/W 로그 중 **마지막 WRITE의 값**이다.  
이 사실은 이미 `MemoryConsistencyAir`(LogUp)가 보장한다 — 읽기는 반드시 직전 쓰기를 참조해야 한다.  
따라서 R/W 로그를 (`addr`, `rw_counter`) 기준으로 정렬했을 때,  
각 주소 그룹의 **마지막 행 값** = 해당 주소의 최종 상태다.

이것을 해시로 커밋해서 공개하면 SMT 경로 증명 없이도 최종 상태를 증명할 수 있다.

---

## 목표

$$
\text{공개: } W_{in},\ W_{out},\ \pi
\quad\text{비공개: } \text{R/W 로그 전체,\ 중간 상태}
$$

- $W_{in}$ = 세그먼트 시작 시점의 write-set 커밋 (이전 세그먼트에서 상속)
- $W_{out}$ = 세그먼트 종료 시점의 write-set 커밋 (= $W_{in}$ + 이번 세그먼트의 last-writes)
- 재귀 합성 후 최종적으로 공개되는 것 = $(W_0, W_N, \pi)$ 한 쌍

---

## 현재 구조와의 연결

```
현재 VmState.memory_root = [0u8; 32]  ← 항상 0, 아직 미구현

MemoryConsistencyAir (LogUp)  ← 이미 R/W 일관성 보장 (배치 내부)
AccessRecord { rw_counter, addr, value_before, value_after, is_write }
           ← 로그의 모든 원소가 이미 수집됨

StateCommitment.memory_root  ← LinkAir에 포함됨, memory_root 채우면 자동 활성화
LeafReceipt / AggregationReceipt  ← 재귀 트리, LinkAir로 s_out==s_in 강제
```

새로 만들 것: **WriteDeltaAir** (write-set 병합 증명)  
기존 SMT, Merkle path witness, per-update 경로 증명 — **불필요**

---

## 상태 표현 설계

### Write-set W

```
W = 정렬된 [(addr, value)] 리스트, addr 기준 오름차순, 각 addr는 한 번만 등장
W_root = Poseidon2(addr_0 ‖ val_0 ‖ addr_1 ‖ val_1 ‖ … ‖ addr_k ‖ val_k)
```

- **메모리 W_mem**: EVM 메모리 주소(32-byte) → 값(32-byte)
- **스토리지 W_stor**: `(contract_addr ‖ slot)` → 값(32-byte)

두 도메인을 분리해서 커밋하거나, 도메인 태그를 붙여 단일 리스트로 합칠 수 있다.

### VmState 변경

```rust
pub struct VmState {
    pub opcode:       u8,
    pub pc:           usize,
    pub sp:           usize,
    pub stack:        Vec<[u8; 32]>,
    pub memory_root:  [u8; 32],   // ← W_mem 루트 (현재: all-zero)
    pub storage_root: [u8; 32],   // 신규: W_stor 루트
}
```

`StateCommitment`도 `storage_root` 추가 커밋.

---

## R/W 로그에서 최종 상태 추출

### 정렬 기준

```
AccessRecord 정렬 순서: (addr, rw_counter) 오름차순

같은 addr 그룹에서:
  - 마지막 행 = is_last = true
  - 마지막 WRITE의 value_after = 이 주소의 최종 값
```

### 세그먼트의 쓰기 델타 D

```
D_seg = { (addr, last_written_value) |
          addr가 이번 세그먼트에서 WRITE된 주소,
          last_written_value = 해당 addr의 가장 높은 rw_counter에서의 value_after }

D_seg_root = Poseidon2(정렬된 D_seg)
```

`MemoryConsistencyAir`는 이미 이 구조를 내부적으로 갖고 있다.  
**추가할 것**: 각 addr 그룹의 마지막 WRITE를 추출해서 `D_seg_root`를 공개 출력으로 내보내는 AIR 제약.

---

## WriteDeltaAir 설계

### 역할

$$
W_{out\_root} = \text{merge}(W_{in\_root},\ D_{seg\_root})
$$

를 인-서킷으로 강제한다. 여기서 merge = "주소 충돌 시 D_seg 쪽 값이 우선, 정렬 유지".

### 입력 / 출력

```
public input:
  W_in_root    ← 이전 세그먼트가 전달한 write-set 커밋
  D_seg_root   ← 이번 세그먼트 MemoryConsistencyAir의 공개 출력

public output:
  W_out_root   ← 다음 세그먼트로 전달 (= s_out.memory_root)

비밀 trace:
  W_in 전체와 D_seg 전체를 addr 순으로 인터리브한 행들
```

### AIR trace 구조 (행 1개 = 하나의 (addr, value) 항목)

| 컬럼 | 설명 |
|------|------|
| `addr[0..8]` | 32-byte 주소 (M31 필드 8개) |
| `value[0..8]` | 32-byte 값 |
| `src` | 0 = W_in에서 옴, 1 = D_seg에서 옴 |
| `keep` | 1 = 이 행이 W_out에 포함됨 |
| `running_hash[0..8]` | 현재까지의 누적 Poseidon2 스펀지 상태 |

### AIR 제약

1. **정렬**: `addr[i] ≤ addr[i+1]` (주소 오름차순)
2. **중복 처리**: 동일 addr가 연속으로 두 번 나오면:
   - `src[i]=0, src[i+1]=1` (W_in 다음 D_seg 순)
   - `keep[i]=0, keep[i+1]=1` (D_seg가 우선)
3. **유일성**: 같은 addr에 대해 D_seg 항목이 없으면 W_in 항목을 유지 (`keep=1`)
4. **누적 해시**: `running_hash`가 keep=1인 행들에 대해 Poseidon2 스펀지를 누적
5. **첫 행**: 스펀지 초기화
6. **마지막 행**: `running_hash == W_out_root`
7. **서브-증명 연결**:
   - W_in 항목들의 해시 = W_in_root (별도 로오업 또는 직렬 해시)
   - D_seg 항목들의 해시 = D_seg_root (MemoryConsistencyAir와 공유)

---

## MemoryConsistencyAir 확장

현재 `MemoryConsistencyAir`가 이미 수행하는 것:
- R/W 로그를 (addr, rw_counter) 기준으로 정렬 (LogUp 인수)
- 각 읽기가 직전 쓰기를 참조함을 강제

다음 두 가지 제약을 추가한다.

### 제약 1: D_seg_root 공개 출력 (쓰기 방향)

각 addr 그룹의 마지막 행이 WRITE일 경우, 그 `(addr, value_after)`를  
`D_seg` 리스트에 포함시키고, `D_seg_root = Poseidon2(D_seg)`를 **공개 출력**으로 내보낸다.

```
is_last[i] = (addr[i] ≠ addr[i+1]) 또는 마지막 행
is_write_last[i] = is_last[i] AND is_write[i]
```
이 is_write_last=1 인 행들을 addr 순으로 해시한 것이 D_seg_root.

### 제약 2: First-Read Initialization (읽기 방향) ← Soundness 필수

세그먼트 경계를 넘어오는 읽기를 제약하지 않으면 **sound하지 않다**.

```
반례:
  세그먼트 j: MSTORE(A, 42)  → W_j[A] = 42, W_j → W_{j+1}
  세그먼트 i (i > j): MLOAD(A) → 프루버가 value=0 으로 위조

  MemoryConsistencyAir(세그먼트 i) 내부에는 A의 이전 쓰기가 없으므로
  value_before 에 무슨 값을 넣어도 내부 AIR를 통과함
```

**fix**: 세그먼트 내에서 **처음으로 등장하는 접근(is_first_access=1)** 이 READ인 경우,  
그 `value_before`가 W_in[addr]와 일치함을 인-서킷으로 강제한다.

```
is_first_access[i] = (i == 0) OR (addr[i] ≠ addr[i-1])
is_inherited_read[i] = is_first_access[i] AND (is_write[i] == 0)
```

is_inherited_read=1 인 행은 **W_in lookup** 을 거쳐야 한다:

```
(addr[i], value_before[i]) ∈ W_in_table
```

W_in_table은 세그먼트 STARK의 **auxiliary public table**로 제공된다.  
WriteDeltaAir가 이미 W_in 전체를 witness로 갖고 있으므로, 같은 데이터를 LogUp 테이블로 재사용한다.

> 단, 첫 번째 세그먼트(W_in = ∅)에서 is_inherited_read=1이면 value_before가 반드시 0이어야 한다  
> (EVM 초기 메모리는 모두 0).

비용: 컬럼 3개(`is_last`, `is_write_last`, `is_first_access`) + running hash + W_in LogUp 인수.

---

## 세그먼트 증명 구조

### LeafReceipt 확장

```rust
pub struct LeafReceipt {
    pub s_in:           StateCommitment,  // s_in.memory_root = W_in_root
    pub s_out:          StateCommitment,  // s_out.memory_root = W_out_root
    pub batch_receipt:  Option<BatchTransactionZkReceipt>,
    pub steps:          Vec<InstructionTransitionProof>,
    // 신규:
    pub write_delta_proof: Option<WriteDeltaProof>,  // WriteDeltaAir STARK 증명
}
```

### BatchTransactionZkReceipt 확장

```rust
pub struct BatchTransactionZkReceipt {
    ...
    pub env_context:    Option<BlockTxContext>,  // 기존
    pub d_seg_root:     Option<[u8; 32]>,        // 신규: 이번 배치의 쓰기 델타 커밋
}
```

`verify_batch_transaction_zk_receipt` step 14: `d_seg_root`가 R/W 로그의 last-writes와 일치하는지 검증.

---

## 공개 입력 (Public Interface)

```
공개:
  W_0 = []  (빈 write-set, 실행 시작 전)
  W_N = [(addr_0, val_0), …, (addr_k, val_k)]  ← 모든 쓰여진 주소의 최종 값
  π   = ExecutionReceipt (재귀 트리)

비공개:
  중간 write-set W_1 … W_{N-1}
  R/W 로그 전체 (배치 내부 AIR에만 존재)
  실행 trace
```

검증자 알고리즘:
1. `verify_execution_receipt(π)`
2. `π.s_in().memory_root == H(W_0) = H([])` 검증
3. `π.s_out().memory_root == H(W_N)` 검증
4. 필요시 `W_N`의 특정 주소 값 조회 (W_N은 공개 리스트이므로 직접 확인)

---

## 재귀 합성 흐름

```
세그먼트 0        세그먼트 1        세그먼트 2        세그먼트 3
W_0 → W_n        W_n → W_2n       W_2n → W_3n      W_3n → W_N
(D_0 적용)       (D_1 적용)       (D_2 적용)       (D_3 적용)
  │                  │                  │                  │
  └────── Link ──────┘                  └────── Link ───────┘
      W_0 → W_2n                            W_2n → W_N
  (LinkAir: left.s_out.memory_root          (동일)
         == right.s_in.memory_root)
          └──────────────── Link ──────────────────┘
                           W_0 → W_N
```

`LinkAir`는 기존 그대로 `left.s_out.memory_root == right.s_in.memory_root`를 강제한다.  
`memory_root`가 실제 W_root로 채워지면 **추가 변경 없이 전이 연결이 보장**된다.

---

## 구현 단계

### Phase 3-A: MemoryConsistencyAir 확장

| 파일 | 작업 |
|------|------|
| `zk_proof/memory_consistency.rs` | `is_last`, `is_write_last`, `is_first_access` 컬럼 추가 |
| | `d_seg_root` 공개 출력 계산 (running Poseidon2 over last-writes) |
| | `MemoryConsistencyProof`에 `d_seg_root: [u8; 32]` 필드 추가 |
| | is_inherited_read=1 행에 대해 `(addr, value_before) ∈ W_in_table` LogUp 인수 추가 |
| `zk_proof/write_delta.rs` | W_in 항목들을 LogUp 테이블로도 노출 (MemoryConsistencyAir와 공유) |

### Phase 3-B: WriteDeltaAir

| 파일 | 작업 |
|------|------|
| `zk_proof/write_delta.rs` (신규) | `WriteDeltaAir` — merge(W_in, D_seg) → W_out |
| | `prove_write_delta(w_in_root, d_seg, w_in_entries) -> WriteDeltaProof` |
| | `verify_write_delta(proof, w_in_root, d_seg_root, w_out_root) -> bool` |

### Phase 3-C: VmState / StateCommitment 확장

| 파일 | 작업 |
|------|------|
| `transition.rs` | `VmState.storage_root: [u8; 32]` 추가 |
| `transition.rs` | `BatchTransactionZkReceipt.d_seg_root` 추가 |
| `transition.rs` | `LeafReceipt.write_delta_proof` 추가 |
| `zk_proof/recursive.rs` | `StateCommitment`에 `storage_root` 추가 |
| `zk_proof/recursive.rs` | `LinkAir` 컬럼 확장 (36 → 52, storage_root[8] 포함) |

### Phase 3-D: Inspector와 실행 경로 연동

| 파일 | 작업 |
|------|------|
| `execute.rs` | `ProvingInspector`에 `write_set: BTreeMap<[u8;32], [u8;32]>` 보관 |
| | MSTORE/SSTORE 시 write_set 업데이트 |
| | 세그먼트 경계에서 `VmState.memory_root = H(write_set)` 채우기 |
| | `d_seg` 계산 후 `BatchTransactionZkReceipt.d_seg_root` 세팅 |

### Phase 3-E: prove_batch / verify_batch 통합

| 파일 | 작업 |
|------|------|
| `transition.rs` | `verify_batch_transaction_zk_receipt` step 14: `d_seg_root` 검증 |
| `transition.rs` | `prove_batch_transaction_zk_receipt`에서 `d_seg_root` 생성 |
| `transition.rs` | `prove_leaf_receipt`에서 `WriteDeltaProof` 생성 |

### Phase 3-F: 최상위 BlockProof 래퍼

| 파일 | 작업 |
|------|------|
| `transition.rs` (신규 함수) | `BlockProof { s_initial, s_final, receipt }` |
| | `prove_block_proof(receipts) -> BlockProof` |
| | `verify_block_proof(proof, w_final: &[(addr, value)]) -> bool` |

---

## MemoryConsistencyAir와의 관계

| | 현재 MemoryConsistencyAir | 확장 후 |
|---|---|---|
| **역할** | R/W 일관성 (읽기 = 직전 쓰기, 세그먼트 내부) | 동일 + D_seg_root 출력 + 크로스-세그먼트 읽기 초기화 |
| **공개 값** | 없음 | `d_seg_root` |
| **비용 증가** | — | 컬럼 3개 + running hash + W_in LogUp 인수 |
| **StorageConsistencyAir** | 동일 구조 | 동일하게 확장 |

WriteDeltaAir는 `d_seg_root`를 입력으로 받아 W_in + D_seg → W_out를 증명하고,  
W_in 항목들을 LogUp 테이블로 노출해 MemoryConsistencyAir의 First-Read 검증에 사용한다.

### AIR 간 연결 흐름

```
 WriteDeltaAir
   W_in_table (public aux)  ──LogUp──►  MemoryConsistencyAir
         │                               (is_inherited_read 행의
         │                                value_before 검증)
         │
         ▼
   D_seg_root (public output) ◄── MemoryConsistencyAir
         │
         ▼
   W_out_root (public output)
         │
         ▼
   s_out.memory_root  ──LinkAir──►  다음 세그먼트의 s_in.memory_root
                                     (= W_in of next segment)
```

---

## 로드맵 연결

```
oracle-removal-progress.md 기준:

Phase 3-A/B (MemoryConsistencyAir 확장 + WriteDeltaAir)
  → MLOAD/MSTORE/SLOAD/SSTORE 의 "나머지 한계" 해소 (SMT 경로 증명 없이)
  → TLOAD/TSTORE: 별도 TransientDeltaAir 또는 동일 구조 재사용

Phase 3-C/D (VmState 확장)
  → memory_root / storage_root 가 실제 W_root가 되어
    LinkAir 체인이 완전한 상태 전이 증명으로 격상

Phase 3-E/F (배치 + BlockProof 통합)
  → 단일 배치도 d_seg_root로 독립적 쓰기 정확성 보장
  → 최종 W_N이 공개되어 블록 실행 결과 외부 검증 가능
```
