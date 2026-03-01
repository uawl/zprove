# 오라클 제거 진행 현황

> 최종 목표: 모든 EVM opcode의 실행 결과가 인-서킷(STARK AIR)으로 검증되어
> "외부 증인(oracle)"에 의존하지 않는 완전한 ZK-EVM 증명 시스템 구축.

---

## Soundness Gap 마스터 테이블

> 마지막 업데이트: 2026-03-01

| Gap ID | 분류 | Opcode(s) | 공격 시나리오 요약 | 상태 | 닫힌 위치 |
|--------|------|-----------|-----------------|------|----------|
| Gap-1/5 | Memory | MLOAD/MSTORE | 메모리 오라클 divergence | ✅ | step 11a/11b |
| Gap-2 | Storage | SLOAD/SSTORE + sub-call | call tree 전체 스토리지 일관성 | ✅ | `collect_storage_claims_recursive` + step 11c/11d |
| Gap-3 | Arithmetic | AND/OR/XOR | 결과값 unconstrained | ✅ | `validate_manifest_rows` |
| Gap-4 | Arithmetic | U29/U24/U15 | 연산자 비트 범위 | ✅ | `validate_manifest_rows` |
| Gap-7 | Crypto | KECCAK256 | output hash binding | ✅ | step 8/9/11e |
| Gap-8 | Memory | MCOPY | src≠dst 복사 | ✅ | step 10 |
| Gap-10 | Env | 13 static opcodes | 블록/TX 컨텍스트 오라클 | ✅ | step 13, `BlockTxContext` |
| Gap-10b | Control | STATICCALL | EIP-214 쓰기 금지 위반 | ✅ | `verify_sub_call_claim` |
| Gap-C5 | Control | CALL/CREATE | depth ≥ 1024 미거부 | ✅ | `verify_sub_call_claim` |
| Gap-11a | Control | CALL 계열 | stack top ↔ success flag 불일치 | ✅ | step 12(a) |
| Gap-11b | Control | CREATE/CREATE2 | stack pushed addr ↔ `callee` 불일치 | ✅ | step 12(a) |
| Gap-A3 | Control | inner_proof optional | oracle mode 허용 (inner_proof: None) | ✅ | `Box<TransactionProof>` 필수화 |
| Gap-C1 | Control | CALL/CREATE | inner proof RETURN vs REVERT ↔ success 불일치 | ✅ | `verify_sub_call_claim` step 3b |
| Gap-C2 | Control | CALL/CREATE | inner proof 비종료 (mid-exec 에서 잘림) | ✅ | `verify_sub_call_claim` step 0 |
| Gap-D1 | Storage | TLOAD/TSTORE | transient storage TX-start reset 미검증 | ✅ | `validate_oracle_sload_reads` + `validate_oracle_sstore_writes` TLOAD/TSTORE 확장 |
| Gap-C4 | Create | CREATE2 | `keccak256(0xff‖deployer‖salt‖keccak256(initcode))` 공식 미검증 | ✅ | `verify_sub_call_claim` + `SubCallClaim.create2_{deployer,salt,initcode_hash}` |
| Gap-11c | Create | CREATE | nonce-based 주소 유도 공식 미검증 | 🔴 OPEN | — |
| Gap-C3 | Control | CALL | ETH 값 전송 세계 상태 미바인딩 | 🔴 OPEN | — |
| Gap-D2 | Env | PC | PC 오라클 (EnvAxiom tautology) | 🔴 OPEN | — |
| Gap-A1 | Memory | MSIZE | 메모리 크기 오라클 | 🔴 OPEN | — |
| Gap-A2 | Env | GAS | 가스 소모 accounting 없음 | 🔴 OPEN | — |
| Gap-B1 | World State | BALANCE | 계정 잔고 오라클 | 🔴 OPEN | — |
| Gap-B2 | World State | EXTCODESIZE/EXTCODEHASH | 코드 크기/해시 오라클 | 🔴 OPEN | — |
| Gap-B3 | World State | BLOCKHASH | 블록 해시 오라클 | 🔴 OPEN | — |
| Gap-B4 | World State | SELFBALANCE | 자신 잔고 오라클 (mutable) | 🔴 OPEN | — |
| Gap-E1 | Event | LOG0–LOG4 | 메모리 → 토픽/data 바인딩 없음 | 🔴 OPEN | — |
| Gap-F1 | State | SELFDESTRUCT | 수혜자 잔고 이전 증명 없음 | 🔴 OPEN | — |

---

## 닫힌 갭 상세

### ✅ Gap-1/5: 메모리 일관성 (MLOAD / MSTORE / MSTORE8)

| 항목 | 상태 |
|------|------|
| `MemoryConsistencyAir` LogUp | ✅ |
| `validate_oracle_mload_reads` | ✅ step 11a |
| `validate_oracle_mstore_writes` | ✅ step 11b |
| MCOPY copy-consistency (Gap-8) | ✅ step 10 |

---

### ✅ Gap-2: 스토리지 일관성 — outer + sub-call 통합 (2026-03-01)

| 항목 | 상태 |
|------|------|
| `StorageConsistencyAir` LogUp | ✅ |
| `collect_storage_claims_recursive`: outer + inner call 재귀 수집 | ✅ |
| `validate_oracle_sload_reads` (outer + inner) | ✅ step 11c |
| `validate_oracle_sstore_writes` (outer + inner) | ✅ step 11d |
| TLOAD/TSTORE → 동일 AIR 포함 | ✅ (reset 불변식은 Gap-D1로 별도) |

---

### ✅ Gap-7: KECCAK256

| 항목 | 상태 |
|------|------|
| `KeccakConsistencyAir` | ✅ step 8 |
| 메모리 ↔ keccak 크로스체크 | ✅ step 9 |
| oracle output ↔ consistency proof | ✅ step 11e |

---

### ✅ Gap-10: Static env opcode 13개 (2026-03-01)

`BlockTxContext`를 public input. `prove_batch_transaction_zk_receipt_with_env` →
step 13 에서 모든 스택 출력값을 `BlockTxContext` 필드와 1:1 대조.

| opcode | hex | 필드 |
|--------|-----|------|
| ADDRESS | 0x30 | `self_address` |
| ORIGIN | 0x32 | `origin` |
| CALLER | 0x33 | `caller` |
| CALLVALUE | 0x34 | `callvalue` |
| CALLDATASIZE | 0x36 | `calldata_size` |
| GASPRICE | 0x3a | `gas_price` |
| COINBASE | 0x41 | `coinbase` |
| TIMESTAMP | 0x42 | `timestamp` |
| NUMBER | 0x43 | `block_number` |
| DIFFICULTY/PREVRANDAO | 0x44 | `prevrandao` |
| GASLIMIT | 0x45 | `gas_limit` |
| CHAINID | 0x46 | `chain_id` |
| BASEFEE | 0x48 | `basefee` |

---

### ✅ Gap-10b / Gap-A3 / Gap-11a,b / Gap-C1 / Gap-C2: Sub-call soundness (Phase A, 2026-03-01)

| 항목 | 상태 |
|------|------|
| `inner_proof: Option` → `Box<TransactionProof>` (oracle mode 제거) | ✅ |
| `pending_sub_call_stack`에 `inner_start` 추가, `call_end()` drain | ✅ |
| STATICCALL write prohibition (EIP-214) | ✅ `verify_sub_call_claim` |
| stack top ↔ `sc.success` binding (CALL/CALLCODE/DELEGATECALL/STATICCALL) | ✅ step 12(a) |
| CREATE success → pushed addr == `sc.callee` | ✅ step 12(a) |
| precompile/empty account: empty `inner.steps` 허용 | ✅ |
| **Gap-C1**: `sc.success` ↔ inner 최종 opcode (RETURN vs REVERT) | ✅ `verify_sub_call_claim` step 3b |
| **Gap-C2**: inner proof 비종료(잘림) 거부 | ✅ `verify_sub_call_claim` step 0 |

---

## 미해결 갭 상세

### ✅ Gap-D1: TLOAD/TSTORE transient storage reset 불변식 (2026-03-01)

| 항목 | 상태 |
|------|------|
| TLOAD/TSTORE claim → `StorageConsistencyAir` 포함 (SLOAD/SSTORE와 공유) | ✅ |
| `validate_oracle_sload_reads` 에 TLOAD 필터 추가 | ✅ |
| `validate_oracle_sstore_writes` 에 TSTORE 필터 추가 | ✅ |
| reset 불변식 (TX 시작 시 `stor_w_in = ∅`) 이 STARK 수준에서 강제됨 | ✅ |

**설명**: `prove_batch_transaction_zk_receipt_with_w_in` 에서 `stor_w_in = ∅` 이 이미 public input으로 강제되므로, StorageConsistencyAir는 TLOAD의 첫 번째 읽기가 0이어야 함을 AIR에서 증명합니다. 검증자 측에서 TLOAD/TSTORE 오라클 값을 consistency proof에 바인딩하는 코드가 누락되어 있었으며 이를 확장했습니다.

---

### ✅ Gap-C4: CREATE2 주소 공식 검증 (2026-03-01)

| 항목 | 상태 |
|------|------|
| `SubCallClaim.create2_deployer: Option<[u8; 20]>` | ✅ |
| `SubCallClaim.create2_salt: Option<[u8; 32]>` | ✅ |
| `SubCallClaim.create2_initcode_hash: Option<[u8; 32]>` (keccak256(initcode)) | ✅ |
| `execute.rs create()`: `CreateScheme::Create2 { salt }` 에서 deployer/salt/initcode_hash 취득 | ✅ |
| `verify_sub_call_claim`: `keccak256(0xff‖deployer‖salt‖initcode_hash)[12..]` ↔ `claim.callee` 구연 | ✅ |
| 센지내심/증인 누락 시 Err | ✅ |

---

### ⚠️ Gap-D1 잔여 한계 (TODO for later)

### ⚠️ Gap-D1 잔여 한계 (TODO for later)

- **현황**: TLOAD/TSTORE 오라클 값이 StorageConsistencyProof에 바인딩됨 ✔︎
- **재실 가능한 이해**: 컨트랙트가 다를 때 TLOAD 직접 성능 데이터 제공의 좌요표를 리보크 -> 전용 TransientStorageConsistencyAir

### 🔴 Gap-C4 잔여 한계 (TODO for later)

### 🔴 Gap-C4 잔여 한계 (TODO for later)

- initcode_hash는 execute.rs에서 `keccak256(init_code)` 직접 계산되지만, KeccakConsistencyAir에 데이터를 컨밋하지 않음. 즉 증명자가 잘못된 initcode_hash를 제공하면 주소 유도는 실패하지만 실제로 배포된 bytecode가 initcode와 다를 수 있음.
- 완전한 해결: inner proof의 initcode Keccak claim을 `create2_initcode_hash`와 교차 검증

### 🔴 Gap-11c: CREATE 주소 공식 미검증

`keccak256(rlp(sender, nonce))` 공식 미검증.
**선행 조건**: RLP 인코딩 + KeccakConsistencyAir.

### 🔴 Gap-C3: CALL ETH 값 전송 미바인딩

`SubCallClaim.value > 0` 일 때 caller 잔고 감소 / callee 증가가 월드 상태 증명으로 검증되지 않음.
**선행 조건**: Gap-B1 (BALANCE MPT).

### 🔴 Group A (동적 env): PC / MSIZE / GAS / CODESIZE / RETURNDATASIZE / SELFBALANCE / BLOBBASEFEE

| opcode | hex | 필요 작업 |
|--------|-----|----------|
| PC | 0x58 | `VmState.pc` 유도 또는 PCCounterAir |
| MSIZE | 0x59 | MemoryConsistencyAir max address에서 유도 |
| GAS | 0x5a | GasAccountingAir |
| CODESIZE | 0x38 | public input 확장 (비교적 쉬움) |
| RETURNDATASIZE | 0x3d | `ReturnDataClaim.size` 필드 추가 |
| SELFBALANCE | 0x47 | 월드 상태 Merkle |
| BLOBBASEFEE | 0x4a | 블록 헤더 public input |

### 🔴 Group B (외부 상태): BALANCE / EXTCODESIZE / EXTCODEHASH / BLOCKHASH / BLOBHASH

전부 MPT 검증 AIR 필요. 가장 복잡한 작업군.

### 🔴 Group E / F: LOG0–LOG4, SELFDESTRUCT

현재 tautology WFF. 각각 메모리 크로스체크 AIR, 월드 상태 전이 증명 필요.

---

## 우선순위 로드맵

```
Phase 1   ✅ static env opcode — BlockTxContext public input
Phase 2   ✅ Sub-call Phase A — inner_proof 필수화, success binding, STATICCALL EIP-214
Phase 2b  ✅ Gap-2 — sub-call storage 통합 (collect_storage_claims_recursive)
Phase 2c  ✅ Gap-C1/C2 — inner proof 종료 검증 (RETURN/REVERT ↔ success)
Phase 2d  ✅ Gap-D1 — TLOAD/TSTORE 오라클 값 검증 확장
Phase 2e  ✅ Gap-C4 — CREATE2 주소 공식 검증
Phase 3   🔴 Gap-11c — CREATE nonce-based 주소 공식
Phase 5   🔴 Group A 일부 — CODESIZE, RETURNDATASIZE, BLOBBASEFEE (public input 확장)
Phase 6   🔴 Group A 나머지 — PC, MSIZE, GAS (전용 AIR)
Phase 7   🔴 Log AIR — LOG0–LOG4 메모리 바인딩
Phase 8   🔴 Group B — MPT 검증 AIR (BALANCE, EXTCODE*, BLOCKHASH)
Phase 9   🔴 Gap-C3 — CALL 값 전송 (Phase 8 선행 필요)
Phase 10  🔴 SELFDESTRUCT — 월드 상태 전이 증명
```

---

## 전체 진행률

| 분류 | 총 Gap 수 | 완료 | 잔여 |
|------|----------|------|------|
| 메모리 (MLOAD/MSTORE/MCOPY) | 3 | 3 | 0 |
| 스토리지 (SLOAD/SSTORE + sub-call) | 2 | 2 | 0 |
| KECCAK256 | 1 | 1 | 0 |
| 산술/논리 결과·범위 | 2 | 2 | 0 |
| Static env (13 opcodes) | 1 | 1 | 0 |
| Sub-call/Create soundness | 8 | 8 | 0 |
| CREATE 주소 공식 | 2 | 1 (C4) | 1 (11c, CREATE nonce) |
| Dynamic env (PC/MSIZE/GAS 등) | 7 | 0 | 7 |
| External state (BALANCE 등) | 5 | 0 | 5 |
| ETH 값 전송 | 1 | 0 | 1 |
| LOG0–LOG4 | 1 | 0 | 1 |
| SELFDESTRUCT | 1 | 0 | 1 |
| **합계** | **34** | **17** | **17** |
