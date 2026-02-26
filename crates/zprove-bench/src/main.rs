use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use revm::bytecode::opcode;
use zprove_core::byte_table::{prove_byte_table, verify_byte_table};
use zprove_core::semantic_proof::{Proof, compile_proof};
use zprove_core::transition::{
  BatchTransactionZkReceipt, InstructionTransitionProof, InstructionTransitionStatement,
  InstructionTransitionZkReceipt, MemAccessClaim, VmState, prove_batch_transaction_zk_receipt,
  prove_instruction, prove_instruction_zk_receipt, prove_instruction_zk_receipt_timed,
  prove_instruction_zk_receipts_parallel, verify_batch_transaction_zk_receipt,
  verify_instruction_zk_receipt, verify_proof,
};
use zprove_core::zk_proof::{
  CircleStarkProof, build_lut_steps_from_rows_auto, collect_byte_table_queries_from_rows,
  prove_batch_wff_proofs, prove_memory_consistency, verify_batch_wff_proofs,
  verify_memory_consistency,
};

fn proof_bytes(p: &CircleStarkProof) -> usize {
  bincode::serialize(p).map(|v| v.len()).unwrap_or(0)
}

// ============================================================
// Baseline save/compare
// ============================================================

/// Per-opcode timing snapshot stored in the baseline file.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct BaselineEntry {
  rows: usize,
  setup_us: f64,
  stack_ir_us: f64,
  lut_us: f64,
  logup_us: f64,
  total_us: f64,
  zkp_prove_us: f64,
  zkp_verify_us: f64,
}

type Baseline = BTreeMap<String, BaselineEntry>;

fn load_baseline(path: &str) -> Result<Baseline, String> {
  let text =
    std::fs::read_to_string(path).map_err(|e| format!("cannot read baseline '{path}': {e}"))?;
  serde_json::from_str(&text).map_err(|e| format!("cannot parse baseline '{path}': {e}"))
}

fn save_baseline(path: &str, baseline: &Baseline) -> Result<(), String> {
  let text = serde_json::to_string_pretty(baseline)
    .map_err(|e| format!("cannot serialize baseline: {e}"))?;
  std::fs::write(path, text).map_err(|e| format!("cannot write baseline '{path}': {e}"))
}

/// Format a percentage delta: positive = slower (red), negative = faster (green).
/// Always shows sign and one decimal place, e.g. "+12.3%" or "-34.1%".
fn fmt_delta(new_val: f64, old_val: f64) -> String {
  if old_val <= 0.0 {
    return "  n/a ".to_string();
  }
  let pct = (new_val - old_val) / old_val * 100.0;
  if pct >= 1.0 {
    format!("\x1b[31m{:+.1}%\x1b[0m", pct) // red = slower
  } else if pct <= -1.0 {
    format!("\x1b[32m{:+.1}%\x1b[0m", pct) // green = faster
  } else {
    format!("{:+.1}%", pct) // neutral
  }
}

fn parse_arg_opt_str<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
  let mut i = 0;
  while i < args.len() {
    if args[i] == key {
      return args.get(i + 1).map(|s| s.as_str());
    }
    i += 1;
  }
  None
}

#[derive(Clone)]
struct BenchCase {
  name: String,
  group: &'static str,
  opcode: u8,
  inputs: Vec<[u8; 32]>,
  output: [u8; 32],
}

fn u256_bytes(val: u128) -> [u8; 32] {
  let mut b = [0u8; 32];
  b[16..32].copy_from_slice(&val.to_be_bytes());
  b
}

fn i256_bytes(val: i128) -> [u8; 32] {
  let mut b = if val < 0 { [0xFF; 32] } else { [0u8; 32] };
  b[16..32].copy_from_slice(&val.to_be_bytes());
  b
}

fn mul_u256_mod(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
  let mut limbs = [0u32; 64];
  for i in 0..32 {
    let ai = a[31 - i] as u32;
    for j in 0..32 {
      let bj = b[31 - j] as u32;
      limbs[i + j] += ai * bj;
    }
  }
  for k in 0..63 {
    let carry = limbs[k] >> 8;
    limbs[k] &= 0xFF;
    limbs[k + 1] += carry;
  }
  let mut out = [0u8; 32];
  for k in 0..32 {
    out[31 - k] = limbs[k] as u8;
  }
  out
}

fn add_u256_mod(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
  let mut out = [0u8; 32];
  let mut carry = 0u16;
  for i in (0..32).rev() {
    let total = a[i] as u16 + b[i] as u16 + carry;
    out[i] = (total & 0xFF) as u8;
    carry = total >> 8;
  }
  out
}

fn sub_u256_mod(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
  let mut out = [0u8; 32];
  let mut borrow = 0i16;
  for i in (0..32).rev() {
    let ai = a[i] as i16;
    let bi = b[i] as i16;
    let mut diff = ai - bi - borrow;
    if diff < 0 {
      diff += 256;
      borrow = 1;
    } else {
      borrow = 0;
    }
    out[i] = diff as u8;
  }
  out
}

fn random_bytes32(rng: &mut StdRng) -> [u8; 32] {
  let mut out = [0u8; 32];
  rng.fill(&mut out);
  out
}

fn random_nonzero_u128(rng: &mut StdRng) -> u128 {
  loop {
    let v = rng.random::<u128>();
    if v != 0 {
      return v;
    }
  }
}

fn random_sdiv_pair(rng: &mut StdRng) -> (i128, i128) {
  loop {
    let a = rng.random::<i128>();
    let b = rng.random::<i128>();
    if b == 0 {
      continue;
    }
    if a == i128::MIN && b == -1 {
      continue;
    }
    return (a, b);
  }
}

fn bool_word(v: bool) -> [u8; 32] {
  let mut w = [0u8; 32];
  w[31] = v as u8;
  w
}

// ============================================================
// Table-driven case generators
//
// Adding a new opcode:
//   1. Write `fn make_<OP>_case(rng: &mut StdRng, idx: usize) -> BenchCase`
//   2. Append it to CASE_GENERATORS
//   That's it — every other bench section picks it up automatically.
// ============================================================

type CaseGen = fn(&mut StdRng, usize) -> BenchCase;

fn make_add_case(rng: &mut StdRng, idx: usize) -> BenchCase {
  let a = u256_bytes(rng.random::<u128>());
  let b = u256_bytes(rng.random::<u128>());
  BenchCase {
    name: format!("ADD#{:02}", idx + 1),
    group: "ADD",
    opcode: opcode::ADD,
    inputs: vec![a, b],
    output: add_u256_mod(&a, &b),
  }
}
fn make_sub_case(rng: &mut StdRng, idx: usize) -> BenchCase {
  let a = u256_bytes(rng.random::<u128>());
  let b = u256_bytes(rng.random::<u128>());
  BenchCase {
    name: format!("SUB#{:02}", idx + 1),
    group: "SUB",
    opcode: opcode::SUB,
    inputs: vec![a, b],
    output: sub_u256_mod(&a, &b),
  }
}
fn make_mul_sparse_case(rng: &mut StdRng, idx: usize) -> BenchCase {
  let a = u256_bytes(rng.random::<u16>() as u128);
  let b = u256_bytes(rng.random::<u16>() as u128);
  BenchCase {
    name: format!("MULs#{:02}", idx + 1),
    group: "MUL (sparse)",
    opcode: opcode::MUL,
    inputs: vec![a, b],
    output: mul_u256_mod(&a, &b),
  }
}
fn make_mul_dense_case(rng: &mut StdRng, idx: usize) -> BenchCase {
  let a = random_bytes32(rng);
  let b = random_bytes32(rng);
  BenchCase {
    name: format!("MULd#{:02}", idx + 1),
    group: "MUL (dense)",
    opcode: opcode::MUL,
    inputs: vec![a, b],
    output: mul_u256_mod(&a, &b),
  }
}
fn make_div_case(rng: &mut StdRng, idx: usize) -> BenchCase {
  let a = rng.random::<u128>();
  let b = random_nonzero_u128(rng);
  BenchCase {
    name: format!("DIV#{:02}", idx + 1),
    group: "DIV",
    opcode: opcode::DIV,
    inputs: vec![u256_bytes(a), u256_bytes(b)],
    output: u256_bytes(a / b),
  }
}
fn make_mod_case(rng: &mut StdRng, idx: usize) -> BenchCase {
  let a = rng.random::<u128>();
  let b = random_nonzero_u128(rng);
  BenchCase {
    name: format!("MOD#{:02}", idx + 1),
    group: "MOD",
    opcode: opcode::MOD,
    inputs: vec![u256_bytes(a), u256_bytes(b)],
    output: u256_bytes(a % b),
  }
}
fn make_sdiv_case(rng: &mut StdRng, idx: usize) -> BenchCase {
  let (a, b) = random_sdiv_pair(rng);
  BenchCase {
    name: format!("SDIV#{:02}", idx + 1),
    group: "SDIV",
    opcode: opcode::SDIV,
    inputs: vec![i256_bytes(a), i256_bytes(b)],
    output: i256_bytes(a / b),
  }
}
fn make_smod_case(rng: &mut StdRng, idx: usize) -> BenchCase {
  let (a, b) = random_sdiv_pair(rng);
  BenchCase {
    name: format!("SMOD#{:02}", idx + 1),
    group: "SMOD",
    opcode: opcode::SMOD,
    inputs: vec![i256_bytes(a), i256_bytes(b)],
    output: i256_bytes(a % b),
  }
}
fn make_and_case(rng: &mut StdRng, idx: usize) -> BenchCase {
  let a = random_bytes32(rng);
  let b = random_bytes32(rng);
  let mut c = [0u8; 32];
  for i in 0..32 {
    c[i] = a[i] & b[i];
  }
  BenchCase {
    name: format!("AND#{:02}", idx + 1),
    group: "AND",
    opcode: opcode::AND,
    inputs: vec![a, b],
    output: c,
  }
}
fn make_or_case(rng: &mut StdRng, idx: usize) -> BenchCase {
  let a = random_bytes32(rng);
  let b = random_bytes32(rng);
  let mut c = [0u8; 32];
  for i in 0..32 {
    c[i] = a[i] | b[i];
  }
  BenchCase {
    name: format!("OR#{:02}", idx + 1),
    group: "OR",
    opcode: opcode::OR,
    inputs: vec![a, b],
    output: c,
  }
}
fn make_xor_case(rng: &mut StdRng, idx: usize) -> BenchCase {
  let a = random_bytes32(rng);
  let b = random_bytes32(rng);
  let mut c = [0u8; 32];
  for i in 0..32 {
    c[i] = a[i] ^ b[i];
  }
  BenchCase {
    name: format!("XOR#{:02}", idx + 1),
    group: "XOR",
    opcode: opcode::XOR,
    inputs: vec![a, b],
    output: c,
  }
}
fn make_not_case(rng: &mut StdRng, idx: usize) -> BenchCase {
  let a = random_bytes32(rng);
  let mut c = [0u8; 32];
  for i in 0..32 {
    c[i] = !a[i];
  }
  BenchCase {
    name: format!("NOT#{:02}", idx + 1),
    group: "NOT",
    opcode: opcode::NOT,
    inputs: vec![a],
    output: c,
  }
}
fn make_lt_case(rng: &mut StdRng, idx: usize) -> BenchCase {
  let a = rng.random::<u128>();
  let b = rng.random::<u128>();
  BenchCase {
    name: format!("LT#{:02}", idx + 1),
    group: "LT",
    opcode: opcode::LT,
    inputs: vec![u256_bytes(a), u256_bytes(b)],
    output: bool_word(a < b),
  }
}
fn make_gt_case(rng: &mut StdRng, idx: usize) -> BenchCase {
  let a = rng.random::<u128>();
  let b = rng.random::<u128>();
  BenchCase {
    name: format!("GT#{:02}", idx + 1),
    group: "GT",
    opcode: opcode::GT,
    inputs: vec![u256_bytes(a), u256_bytes(b)],
    output: bool_word(a > b),
  }
}
fn make_slt_case(rng: &mut StdRng, idx: usize) -> BenchCase {
  let (a, b) = random_sdiv_pair(rng);
  BenchCase {
    name: format!("SLT#{:02}", idx + 1),
    group: "SLT",
    opcode: opcode::SLT,
    inputs: vec![i256_bytes(a), i256_bytes(b)],
    output: bool_word(a < b),
  }
}
fn make_sgt_case(rng: &mut StdRng, idx: usize) -> BenchCase {
  let (a, b) = random_sdiv_pair(rng);
  BenchCase {
    name: format!("SGT#{:02}", idx + 1),
    group: "SGT",
    opcode: opcode::SGT,
    inputs: vec![i256_bytes(a), i256_bytes(b)],
    output: bool_word(a > b),
  }
}
fn make_eq_case(rng: &mut StdRng, idx: usize) -> BenchCase {
  let same = rng.random::<bool>();
  let a = u256_bytes(rng.random::<u128>());
  let b = if same {
    a
  } else {
    u256_bytes(rng.random::<u128>())
  };
  BenchCase {
    name: format!("EQ#{:02}", idx + 1),
    group: "EQ",
    opcode: opcode::EQ,
    inputs: vec![a, b],
    output: bool_word(a == b),
  }
}
fn make_iszero_case(rng: &mut StdRng, idx: usize) -> BenchCase {
  let v: u128 = if rng.random::<bool>() {
    0
  } else {
    rng.random::<u128>()
  };
  BenchCase {
    name: format!("ISZERO#{:02}", idx + 1),
    group: "ISZERO",
    opcode: opcode::ISZERO,
    inputs: vec![u256_bytes(v)],
    output: bool_word(v == 0),
  }
}

/// Per-sample case generator table.
/// To add a new opcode: write `fn make_X_case` above and append it here.
static CASE_GENERATORS: &[CaseGen] = &[
  make_add_case,
  make_sub_case,
  make_mul_sparse_case,
  make_mul_dense_case,
  make_div_case,
  make_mod_case,
  make_sdiv_case,
  make_smod_case,
  make_and_case,
  make_or_case,
  make_xor_case,
  make_not_case,
  make_lt_case,
  make_gt_case,
  make_slt_case,
  make_sgt_case,
  make_eq_case,
  make_iszero_case,
];

/// One-off edge cases appended regardless of `samples`.
fn edge_cases() -> Vec<BenchCase> {
  let mut int_min = [0u8; 32];
  int_min[0] = 0x80;
  vec![BenchCase {
    name: "SDIV_INT_MIN".to_string(),
    group: "SDIV_INT_MIN",
    opcode: opcode::SDIV,
    inputs: vec![int_min, [0xFF; 32]],
    output: int_min,
  }]
}

fn bench_cases(samples: usize, seed: u64) -> Vec<BenchCase> {
  let mut rng = StdRng::seed_from_u64(seed);
  let mut cases = Vec::new();
  for sample_idx in 0..samples {
    for make_case in CASE_GENERATORS {
      cases.push(make_case(&mut rng, sample_idx));
    }
  }
  cases.extend(edge_cases());
  cases
}

fn parse_arg_usize(args: &[String], key: &str, default: usize) -> usize {
  let mut i = 0;
  while i < args.len() {
    if args[i] == key
      && let Some(v) = args.get(i + 1).and_then(|x| x.parse::<usize>().ok())
    {
      return v;
    }
    i += 1;
  }
  default
}

fn parse_arg_opt_u64(args: &[String], key: &str) -> Option<u64> {
  let mut i = 0;
  while i < args.len() {
    if args[i] == key {
      return args.get(i + 1).and_then(|x| x.parse::<u64>().ok());
    }
    i += 1;
  }
  None
}

fn has_flag(args: &[String], key: &str) -> bool {
  args.iter().any(|arg| arg == key)
}

fn build_itp(case: &BenchCase) -> InstructionTransitionProof {
  let proof = prove_instruction(case.opcode, &case.inputs, &[case.output])
    .expect("supported benchmark opcode");

  InstructionTransitionProof {
    opcode: case.opcode,
    pc: 0,
    stack_inputs: case.inputs.clone(),
    stack_outputs: vec![case.output],
    semantic_proof: Some(proof),
    memory_claims: vec![],
  }
}

fn build_statement(case: &BenchCase) -> InstructionTransitionStatement {
  InstructionTransitionStatement {
    opcode: case.opcode,
    s_i: VmState {
      opcode: case.opcode,
      pc: 0,
      sp: case.inputs.len(),
      stack: case.inputs.clone(),
      memory_root: [0u8; 32],
    },
    s_next: VmState {
      opcode: case.opcode,
      pc: 1,
      sp: 1,
      stack: vec![case.output],
      memory_root: [0u8; 32],
    },
    accesses: Vec::new(),
  }
}

fn bench_zkp_prove(
  itp: &InstructionTransitionProof,
  iters: usize,
  workers: usize,
) -> Result<f64, String> {
  let batch = vec![itp.clone(); iters.max(1)];
  let start = Instant::now();
  let receipts = prove_instruction_zk_receipts_parallel(black_box(batch), workers)?;
  black_box(receipts);
  Ok(start.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64)
}

fn warm_up(itp: &InstructionTransitionProof, workers: usize) -> Result<(), String> {
  let warmup_batch = vec![itp.clone()];
  let receipts = prove_instruction_zk_receipts_parallel(warmup_batch, workers)?;
  black_box(receipts);
  Ok(())
}

fn bench_zkp_verify(
  statement: &InstructionTransitionStatement,
  receipt: &InstructionTransitionZkReceipt,
  iters: usize,
) -> f64 {
  let start = Instant::now();
  for _ in 0..iters {
    black_box(verify_instruction_zk_receipt(
      black_box(statement),
      black_box(receipt),
    ));
  }
  start.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64
}

/// Returns true if the ITP can produce a ZK receipt.
/// Auto-detected at runtime — no hard-coded opcode list.
/// New opcodes are supported as soon as their proof compiles.
fn supports_zkp_receipt(itp: &InstructionTransitionProof) -> bool {
  let Some(proof) = itp.semantic_proof.as_ref() else {
    return false;
  };
  let rows = compile_proof(proof);
  build_lut_steps_from_rows_auto(&rows).is_ok()
}

fn bench_wff_prove(case: &BenchCase, iters: usize) -> f64 {
  let start = Instant::now();
  for _ in 0..iters {
    let p = prove_instruction(
      black_box(case.opcode),
      black_box(&case.inputs),
      black_box(&[case.output]),
    );
    black_box(p);
  }
  start.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64
}

fn bench_wff_verify(itp: &InstructionTransitionProof, iters: usize) -> f64 {
  let start = Instant::now();
  for _ in 0..iters {
    black_box(verify_proof(black_box(itp)));
  }
  start.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64
}

fn batch_rows_for_itps(
  itps: &[InstructionTransitionProof],
) -> Vec<zprove_core::semantic_proof::ProofRow> {
  itps
    .iter()
    .flat_map(|itp| {
      itp
        .semantic_proof
        .as_ref()
        .map(|p| compile_proof(p))
        .unwrap_or_default()
    })
    .collect()
}

/// Prove `batch_size` identical copies of `itp` as a single LUT STARK call.
/// Cross-family — no opcode-family dispatch required.
/// Returns `(prove_µs/instr, verify_µs/instr, total_rows, lut_steps)`.
fn bench_batch_wff(
  itp: &InstructionTransitionProof,
  batch_size: usize,
) -> Option<(f64, f64, usize, usize)> {
  let batch: Vec<_> = std::iter::repeat(itp.clone()).take(batch_size).collect();
  let all_rows = batch_rows_for_itps(&batch);
  let total_rows = all_rows.len();
  let lut_steps = build_lut_steps_from_rows_auto(&all_rows).ok()?.len();

  let proofs: Vec<&Proof> = batch
    .iter()
    .filter_map(|i| i.semantic_proof.as_ref())
    .collect();
  if proofs.is_empty() {
    return None;
  }

  let start = Instant::now();
  let (lut_proof, wffs) = prove_batch_wff_proofs(black_box(&proofs)).ok()?;
  black_box(&lut_proof);
  black_box(&wffs);
  let prove_us = start.elapsed().as_secs_f64() * 1_000_000.0 / batch_size as f64;

  let start = Instant::now();
  let _ = black_box(verify_batch_wff_proofs(
    black_box(&lut_proof),
    black_box(&proofs),
    black_box(&wffs),
  ));
  let verify_us = start.elapsed().as_secs_f64() * 1_000_000.0 / batch_size as f64;

  Some((prove_us, verify_us, total_rows, lut_steps))
}

/// Measure the LogUp byte-table overhead for a single instruction proof.
///
/// Returns `(lut_rows, byte_queries, byte_prove_µs, byte_verify_µs)`.
/// Returns `None` when the instruction has no AND/OR/XOR byte operations.
fn bench_logup_overhead(
  itp: &InstructionTransitionProof,
  iters: usize,
) -> Option<(usize, usize, f64, f64)> {
  let semantic_proof = itp.semantic_proof.as_ref()?;
  let rows = compile_proof(semantic_proof);
  let lut_rows = rows.len();
  // Collect byte-level queries directly from ProofRows (correct path).
  // scalar0/scalar1 hold the real byte operands (0..=255); identical
  // (a, b, op) pairs are merged by multiplicity.
  let queries = collect_byte_table_queries_from_rows(&rows);
  if queries.is_empty() {
    return None;
  }
  let n_queries = queries.len();

  // Time prove_byte_table (LogUp overhead).
  let t = Instant::now();
  for _ in 0..iters {
    let proof = black_box(prove_byte_table(black_box(&queries)));
    let _ = proof;
  }
  let byte_prove_us = t.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64;

  // Produce one proof for verify timing.
  let proof = prove_byte_table(&queries);

  // Time verify_byte_table.
  let t = Instant::now();
  for _ in 0..iters {
    let _ = black_box(verify_byte_table(black_box(&proof)));
  }
  let byte_verify_us = t.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64;

  Some((lut_rows, n_queries, byte_prove_us, byte_verify_us))
}

fn main() {
  let args: Vec<String> = std::env::args().collect();
  let wff_iters = parse_arg_usize(&args, "--wff-iters", 20);
  let zkp_prove_iters = parse_arg_usize(&args, "--zkp-prove-iters", 1);
  let zkp_verify_iters = parse_arg_usize(&args, "--zkp-verify-iters", 1);
  let zkp_workers = parse_arg_usize(&args, "--zkp-workers", 0);
  let warmup_only = has_flag(&args, "--warmup-only");
  let no_warmup = has_flag(&args, "--no-warmup");
  let no_stark = has_flag(&args, "--no-stark");
  let samples = parse_arg_usize(&args, "--samples", 3);
  let seed_opt = parse_arg_opt_u64(&args, "--seed");
  let seed = seed_opt.unwrap_or_else(rand::random::<u64>);
  let save_baseline_path = parse_arg_opt_str(&args, "--save-baseline").map(str::to_owned);
  let compare_path = parse_arg_opt_str(&args, "--compare").map(str::to_owned);

  // Load existing baseline for comparison (optional).
  let cmp_baseline: Option<Baseline> = compare_path.as_deref().map(|p| match load_baseline(p) {
    Ok(b) => {
      println!("  compare vs    : {p} ({} entries)", b.len());
      b
    }
    Err(e) => {
      eprintln!("[warn] {e}");
      BTreeMap::new()
    }
  });
  // Accumulate current run's timings for optional save.
  let mut current_baseline: Baseline = BTreeMap::new();

  println!("zprove benchmark");
  match seed_opt {
    Some(s) => println!("  seed          : {s} (fixed)"),
    None => println!("  seed          : {seed} (random)"),
  }
  println!("  samples       : {samples}");
  println!("  wff iters     : {wff_iters}");
  println!("  zkp prove iter: {zkp_prove_iters}");
  println!("  zkp verify iter: {zkp_verify_iters}");
  println!(
    "  zkp workers   : {}",
    if zkp_workers == 0 {
      "auto".to_string()
    } else {
      zkp_workers.to_string()
    }
  );
  println!(
    "  stark bench   : {}",
    if no_stark { "disabled" } else { "enabled" }
  );
  if let Some(ref p) = save_baseline_path {
    println!("  save baseline : {p}");
  }

  // ---- warmup ----
  let warmup_seed = seed ^ 0xA5A5_5A5A_D3C3_B1B1;
  let warmup_itp = bench_cases(1, warmup_seed)
    .into_iter()
    .map(|c| build_itp(&c))
    .find(|itp| supports_zkp_receipt(itp))
    .expect("benchmark suite includes at least one opcode with ZKP receipt support");

  if !no_warmup {
    warm_up(&warmup_itp, zkp_workers).unwrap_or_else(|err| panic!("warm-up failed: {err}"));
  }
  if warmup_only {
    println!("warmup complete");
    return;
  }

  // ---- aggregation structs ----
  #[derive(Default)]
  struct Agg {
    count: usize,
    proof_rows_sum: usize,
    wff_prove_sum: f64,
    wff_verify_sum: f64,
    // STARK (only populated when supports_zkp_receipt)
    stark_count: usize,
    lut_steps_sum: usize,
    zkp_prove_sum: f64,
    zkp_verify_sum: f64,
  }

  let mut aggs: BTreeMap<&'static str, Agg> = BTreeMap::new();

  for case in bench_cases(samples, seed) {
    let itp = build_itp(&case);
    let statement = build_statement(&case);

    let proof_rows = itp
      .semantic_proof
      .as_ref()
      .map(|p| compile_proof(p).len())
      .unwrap_or(0);
    let wff_prove_us = bench_wff_prove(&case, wff_iters);
    let wff_verify_us = bench_wff_verify(&itp, wff_iters);

    let agg = aggs.entry(case.group).or_default();
    agg.count += 1;
    agg.proof_rows_sum += proof_rows;
    agg.wff_prove_sum += wff_prove_us;
    agg.wff_verify_sum += wff_verify_us;

    if no_stark || !supports_zkp_receipt(&itp) {
      continue;
    }

    // LUT step count — auto via cross-family builder, no opcode-family dispatch
    let lut_steps = itp
      .semantic_proof
      .as_ref()
      .and_then(|p| {
        let rows = compile_proof(p);
        build_lut_steps_from_rows_auto(&rows).ok().map(|s| s.len())
      })
      .unwrap_or(0);

    let receipt = match prove_instruction_zk_receipt(&itp) {
      Ok(r) => r,
      Err(_) => continue,
    };
    let zkp_prove_us = match bench_zkp_prove(&itp, zkp_prove_iters, zkp_workers) {
      Ok(v) => v,
      Err(_) => continue,
    };
    let zkp_verify_us = bench_zkp_verify(&statement, &receipt, zkp_verify_iters);

    agg.stark_count += 1;
    agg.lut_steps_sum += lut_steps;
    agg.zkp_prove_sum += zkp_prove_us;
    agg.zkp_verify_sum += zkp_verify_us;
  }

  // ---- WFF layer table ----
  println!();
  println!("── WFF layer (all opcodes) ──────────────────────────────────────────");
  println!(
    "{:<10} {:>7}  {:>10}  {:>14}  {:>14}",
    "opcode", "samples", "proof_rows", "wff_prove(µs)", "wff_verify(µs)"
  );
  println!("{}", "─".repeat(62));
  for (group, agg) in &aggs {
    let n = agg.count as f64;
    println!(
      "{:<10} {:>7}  {:>10.1}  {:>14.3}  {:>14.3}",
      group,
      agg.count,
      agg.proof_rows_sum as f64 / n,
      agg.wff_prove_sum / n,
      agg.wff_verify_sum / n,
    );
  }

  // ---- STARK layer table ----
  if !no_stark {
    let stark_aggs: Vec<_> = aggs.iter().filter(|(_, a)| a.stark_count > 0).collect();
    if !stark_aggs.is_empty() {
      let has_cmp = cmp_baseline.is_some();
      println!();
      println!(
        "── STARK layer (receipt-capable opcodes) ─────────────────────────────────────────────"
      );
      if has_cmp {
        println!(
          "{:<10} {:>7}  {:>10}  {:>16}  {:>16}  {:>12}  {:>12}",
          "opcode",
          "samples",
          "lut_steps",
          "zkp_prove(µs)",
          "zkp_verify(µs)",
          "prove Δ",
          "verify Δ"
        );
        println!("{}", "─".repeat(92));
      } else {
        println!(
          "{:<10} {:>7}  {:>10}  {:>16}  {:>16}",
          "opcode", "samples", "lut_steps", "zkp_prove(µs)", "zkp_verify(µs)"
        );
        println!("{}", "─".repeat(66));
      }
      for (group, agg) in stark_aggs {
        let n = agg.stark_count as f64;
        let prove_us = agg.zkp_prove_sum / n;
        let verify_us = agg.zkp_verify_sum / n;
        if has_cmp {
          let (pd, vd) = cmp_baseline
            .as_ref()
            .and_then(|b| b.get(*group))
            .map(|e| {
              (
                fmt_delta(prove_us, e.zkp_prove_us),
                fmt_delta(verify_us, e.zkp_verify_us),
              )
            })
            .unwrap_or_else(|| ("  new  ".to_string(), "  new  ".to_string()));
          println!(
            "{:<10} {:>7}  {:>10.1}  {:>16.2}  {:>16.2}  {}  {}",
            group,
            agg.stark_count,
            agg.lut_steps_sum as f64 / n,
            prove_us,
            verify_us,
            pd,
            vd,
          );
        } else {
          println!(
            "{:<10} {:>7}  {:>10.1}  {:>16.2}  {:>16.2}",
            group,
            agg.stark_count,
            agg.lut_steps_sum as f64 / n,
            prove_us,
            verify_us,
          );
        }
        // Accumulate into current baseline (will be enriched with component data below).
        current_baseline
          .entry(group.to_string())
          .or_insert_with(|| BaselineEntry {
            rows: 0,
            setup_us: 0.0,
            stack_ir_us: 0.0,
            lut_us: 0.0,
            logup_us: 0.0,
            total_us: 0.0,
            zkp_prove_us: prove_us,
            zkp_verify_us: verify_us,
          });
      }
    }
  }

  // ---- Per-component STARK timing ----
  // For each receipt-capable group, break the prove time into:
  //   StackIR STARK | LUT STARK | WFFMatch STARK
  if !no_stark {
    // Re-use the first case of each receipt-capable group.
    let comp_cases = bench_cases(samples.max(1), seed);
    let mut seen: std::collections::HashSet<&'static str> = std::collections::HashSet::new();
    let mut comp_reps: Vec<(&'static str, InstructionTransitionProof)> = Vec::new();
    for c in &comp_cases {
      let itp = build_itp(c);
      if supports_zkp_receipt(&itp) && seen.insert(c.group) {
        comp_reps.push((c.group, itp));
      }
    }

    if !comp_reps.is_empty() {
      let has_cmp = cmp_baseline.is_some();
      println!();
      println!(
        "── Per-component STARK timing (single instruction, single run) ──────────────────────────────"
      );
      if has_cmp {
        println!(
          "{:<14}  {:>9}  {:>10}  {:>12}  {:>10}  {:>10}  {:>10}  {:>12}  {:>12}",
          "group",
          "rows",
          "setup µs",
          "StackIR µs",
          "LUT µs",
          "LogUp µs",
          "total µs",
          "StackIR Δ",
          "total Δ"
        );
        println!("{}", "─".repeat(107));
      } else {
        println!(
          "{:<14}  {:>9}  {:>10}  {:>12}  {:>10}  {:>10}  {:>10}",
          "group", "rows", "setup µs", "StackIR µs", "LUT µs", "LogUp µs", "total µs"
        );
        println!("{}", "─".repeat(82));
      }
      for (name, itp) in &comp_reps {
        let rows = itp
          .semantic_proof
          .as_ref()
          .map(|p| compile_proof(p).len())
          .unwrap_or(0);
        match prove_instruction_zk_receipt_timed(itp) {
          Ok((_, (setup_us, stack_us, lut_us, logup_us))) => {
            let total_us = setup_us + stack_us + lut_us + logup_us;
            // Update current_baseline with component breakdown.
            let entry = current_baseline.entry(name.to_string()).or_default();
            entry.rows = rows;
            entry.setup_us = setup_us;
            entry.stack_ir_us = stack_us;
            entry.lut_us = lut_us;
            entry.logup_us = logup_us;
            entry.total_us = total_us;
            if has_cmp {
              let (sd, td) = cmp_baseline
                .as_ref()
                .and_then(|b| b.get(*name))
                .map(|e| {
                  (
                    fmt_delta(stack_us, e.stack_ir_us),
                    fmt_delta(total_us, e.total_us),
                  )
                })
                .unwrap_or_else(|| ("  new  ".to_string(), "  new  ".to_string()));
              println!(
                "{:<14}  {:>9}  {:>10.1}  {:>12.1}  {:>10.1}  {:>10.1}  {:>10.1}  {}  {}",
                name, rows, setup_us, stack_us, lut_us, logup_us, total_us, sd, td,
              );
            } else {
              println!(
                "{:<14}  {:>9}  {:>10.1}  {:>12.1}  {:>10.1}  {:>10.1}  {:>10.1}",
                name, rows, setup_us, stack_us, lut_us, logup_us, total_us
              );
            }
          }
          Err(e) => println!("{:<14}  — ({e})", name),
        }
      }
    }
  }

  // ── LogUp byte-table overhead (AND / OR / XOR opcodes) ──────────────────────────────────
  // For each AND/OR/XOR group, compare the standard LUT STARK prove time against the
  // additional cost of producing the companion byte-table LogUp argument.
  //
  // Columns:
  //   group        — opcode family (AND, OR, XOR)
  //   lut_rows     — number of ProofRows in the LUT trace
  //   byte_queries — AND/OR/XOR LutSteps that feed the byte-table proof
  //   lut_us       — standard LUT STARK prove time (from per-component bench above)
  //   bt_prove_us  — prove_byte_table time (LogUp overhead)
  //   overhead%    — bt_prove_us / lut_us × 100
  //   bt_verify_us — verify_byte_table time
  if !no_stark {
    // Derive one ITP per distinct bit-op group.
    let logup_cases = bench_cases(samples.max(1), seed);
    let logup_groups: &[&str] = &["AND", "OR", "XOR"];
    let mut seen_lg: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut logup_reps: Vec<(&'static str, InstructionTransitionProof)> = Vec::new();
    for c in &logup_cases {
      if logup_groups.contains(&c.group)
        && supports_zkp_receipt(&build_itp(c))
        && seen_lg.insert(c.group)
      {
        logup_reps.push((c.group, build_itp(c)));
      }
    }

    if !logup_reps.is_empty() {
      println!();
      println!(
        "── LogUp byte-table overhead (AND / OR / XOR) ──────────────────────────────────────────"
      );
      println!(
        "{:<6}  {:>8}  {:>12}  {:>10}  {:>12}  {:>10}  {:>13}",
        "group", "lut_rows", "byte_queries", "lut_us", "bt_prove_us", "overhead%", "bt_verify_us"
      );
      println!("{}", "─".repeat(79));

      for (name, itp) in &logup_reps {
        // LUT prove time from the timed receipt path (pure LUT STARK, without LogUp).
        let lut_us_opt = prove_instruction_zk_receipt_timed(itp)
          .ok()
          .map(|(_, (_, _, lut, _))| lut);

        match bench_logup_overhead(itp, zkp_prove_iters.max(1)) {
          Some((lut_rows, byte_queries, bt_prove_us, bt_verify_us)) => {
            let (lut_us_str, overhead_str) = match lut_us_opt {
              Some(lut_us) => {
                let pct = bt_prove_us / lut_us * 100.0;
                (format!("{:>10.2}", lut_us), format!("{:>9.1}%", pct))
              }
              None => ("      n/a".to_string(), "      n/a".to_string()),
            };
            println!(
              "{:<6}  {:>8}  {:>12}  {}  {:>12.2}  {}  {:>13.2}",
              name, lut_rows, byte_queries, lut_us_str, bt_prove_us, overhead_str, bt_verify_us,
            );
          }
          None => println!("{:<6}  — (no byte queries)", name),
        }
      }
    }
  }

  // ---- Batch LUT STARK + Cross-family table ----
  // Representative ITPs: one per distinct group (first case of each group that
  // supports ZKP receipt). No opcode-family enum, no manual dispatch.
  if !no_stark {
    let all_cases = bench_cases(samples.max(1), seed);
    let mut seen_groups: std::collections::HashSet<&'static str> = std::collections::HashSet::new();
    struct Rep {
      name: &'static str,
      itp: InstructionTransitionProof,
      wff_verify_us: f64,
    }
    let mut reps: Vec<Rep> = Vec::new();
    for case in &all_cases {
      let itp = build_itp(case);
      if supports_zkp_receipt(&itp) && seen_groups.insert(case.group) {
        let wff_verify_us = bench_wff_verify(&itp, wff_iters);
        reps.push(Rep {
          name: case.group,
          itp,
          wff_verify_us,
        });
        if reps.len() >= 5 {
          break;
        }
      }
    }

    if !reps.is_empty() {
      let batch_sizes = [1usize, 2, 4, 8, 16];
      println!();
      println!(
        "── Batch LUT STARK (amortized µs per instruction) ───────────────────────────────────────"
      );
      println!(
        "{:<14}  {:>5}  {:>12}  {:>12}  {:>12}  {:>12}",
        "group", "batch", "total_rows", "prove/instr", "verify/instr", "×WFF_verify"
      );
      println!("{}", "─".repeat(76));

      for rep in &reps {
        for &bs in &batch_sizes {
          match bench_batch_wff(&rep.itp, bs) {
            Some((prove_us, verify_us, total_rows, _lut)) => {
              let ratio = if rep.wff_verify_us > 0.0 {
                verify_us / rep.wff_verify_us
              } else {
                0.0
              };
              println!(
                "{:<14}  {:>5}  {:>12}  {:>12.2}  {:>12.2}  {:>12.1}×",
                rep.name, bs, total_rows, prove_us, verify_us, ratio,
              );
            }
            None => println!("{:<14}  {:>5}  — (failed)", rep.name, bs),
          }
        }
      }
    }

    // ── Cross-family batch: mix N distinct groups into one LUT STARK call ──
    let cross_proofs: Vec<&Proof> = reps
      .iter()
      .filter_map(|r| r.itp.semantic_proof.as_ref())
      .collect();
    if cross_proofs.len() >= 2 {
      println!();
      println!(
        "── Cross-family batch (mixed opcodes → single LUT STARK) ────────────────────────────────"
      );
      println!(
        "{:<36}  {:>5}  {:>14}  {:>14}  {:>14}",
        "mix", "n", "prove_total µs", "verify_total µs", "verify/instr µs"
      );
      println!("{}", "─".repeat(84));

      for n in 2..=cross_proofs.len() {
        let batch: Vec<&Proof> = cross_proofs.iter().take(n).copied().collect();
        let mix_name: String = reps
          .iter()
          .take(n)
          .map(|r| r.name)
          .collect::<Vec<_>>()
          .join(" + ");
        let prove_start = Instant::now();
        match prove_batch_wff_proofs(&batch) {
          Ok((ref lut_proof, ref wffs)) => {
            let prove_us = prove_start.elapsed().as_secs_f64() * 1e6;
            let verify_start = Instant::now();
            let _ = verify_batch_wff_proofs(lut_proof, &batch, wffs);
            let verify_us = verify_start.elapsed().as_secs_f64() * 1e6;
            println!(
              "{:<36}  {:>5}  {:>14.2}  {:>14.2}  {:>14.2}",
              mix_name,
              n,
              prove_us,
              verify_us,
              verify_us / n as f64
            );
          }
          Err(e) => println!("{:<36}  {:>5}  prove failed: {e}", mix_name, n),
        }
      }
    }
  }

  // ── Proof size per opcode ───────────────────────────────────────────────────
  // Reports the serialized byte size of each STARK proof component:
  //   - StackIR proof  (bincode of CircleStarkProof)
  //   - LUT proof      (bincode of CircleStarkProof)
  //   - VK             (width + degree_bits + 8×M31 commitment = fixed 48 B)
  // The VK is constant per (Air, degree_bits) so we also show whether two
  // opcodes with the same lut_steps share the same degree_bits.
  if !no_stark {
    // Re-derive one receipt per distinct ZKP-capable group.
    let all_cases = bench_cases(samples.max(1), seed);
    let mut seen: std::collections::HashSet<&'static str> = std::collections::HashSet::new();
    let mut size_reps: Vec<(&'static str, usize, InstructionTransitionZkReceipt)> = Vec::new();
    for case in &all_cases {
      let itp = build_itp(case);
      if supports_zkp_receipt(&itp) && seen.insert(case.group) {
        let rows = itp
          .semantic_proof
          .as_ref()
          .map(|p| compile_proof(p).len())
          .unwrap_or(0);
        if let Ok(receipt) = prove_instruction_zk_receipt(&itp) {
          size_reps.push((case.group, rows, receipt));
        }
      }
    }

    if !size_reps.is_empty() {
      // VK = 2×usize (8B each) + commitment (8×M31 = 32B) = 48B (constant)
      let vk_bytes = 8 + 8 + 8 * 4;

      println!();
      println!(
        "── Proof sizes (bincode serialized) ─────────────────────────────────────────────────────"
      );
      println!(
        "{:<14}  {:>6}  {:>12}  {:>10}  {:>9}  {:>10}  {:>9}",
        "opcode", "rows", "stack_ir B", "lut B", "vk B", "total B", "total kB"
      );
      println!("{}", "─".repeat(78));
      let mut total_bytes_all: Vec<(usize, usize, usize, usize)> = Vec::new();
      for (name, rows, receipt) in &size_reps {
        let sir_b = proof_bytes(&receipt.stack_ir_proof);
        let lut_b = proof_bytes(&receipt.lut_kernel_proof);
        let total = sir_b + lut_b + vk_bytes;
        println!(
          "{:<14}  {:>6}  {:>12}  {:>10}  {:>9}  {:>10}  {:>9.2}",
          name,
          rows,
          sir_b,
          lut_b,
          vk_bytes,
          total,
          total as f64 / 1024.0
        );
        total_bytes_all.push((*rows, sir_b, lut_b, total));
      }

      // Batch size vs proof size comparison
      if let Some((_, _, rep_receipt)) = size_reps.first() {
        let rep_itp = build_itp(
          bench_cases(1, seed)
            .iter()
            .find(|c| supports_zkp_receipt(&build_itp(c)))
            .expect("at least one case"),
        );
        println!();
        println!(
          "── Batch proof size vs N×single (ADD as representative) ─────────────────────────────"
        );
        println!(
          "  {:>5}  {:>14}  {:>14}  {:>10}  {:>10}",
          "N", "N×single B", "batch B", "saving B", "saving %"
        );
        println!("  {}", "─".repeat(58));
        let single_lut_b = proof_bytes(&rep_receipt.lut_kernel_proof);
        let single_sir_b = proof_bytes(&rep_receipt.stack_ir_proof);
        let single_total = single_lut_b + single_sir_b + vk_bytes;
        for &n in &[1usize, 2, 4, 8, 16] {
          let batch_itps: Vec<_> = std::iter::repeat(rep_itp.clone()).take(n).collect();
          if let Ok(batch_receipt) = prove_batch_transaction_zk_receipt(&batch_itps) {
            let batch_b = proof_bytes(&batch_receipt.lut_proof) + vk_bytes;
            let n_single_b = single_total * n;
            let saving = n_single_b as i64 - batch_b as i64;
            let pct = saving as f64 / n_single_b as f64 * 100.0;
            println!(
              "  {:>5}  {:>14}  {:>14}  {:>10}  {:>9.1}%",
              n, n_single_b, batch_b, saving, pct
            );
          }
        }
      }
    }
  }

  // ── Parallel vs Batch STARK comparison ──────────────────────────────────────────────────────
  // Compares two paths for proving N identical instructions:
  //   Parallel: N independent STARK proofs run in a thread-pool.
  //   Batch:    1 shared STARK proof covering all N rows (amortises setup + FRI).
  // Reports per-instruction latency and the batch speedup ratio.
  if !no_stark {
    let par_vs_batch_case = bench_cases(1, seed)
      .into_iter()
      .map(|c| (c.clone(), build_itp(&c)))
      .find(|(_, itp)| supports_zkp_receipt(itp));

    if let Some((rep_case, rep_itp)) = par_vs_batch_case {
      let rep_stmt = build_statement(&rep_case);
      let sizes = [1usize, 2, 4, 8];

      println!();
      println!(
        "── Parallel vs Batch STARK ({} as representative) ──────────────────────────────────────",
        rep_case.group
      );
      println!(
        "  {:>5}  {:>14}  {:>14}  {:>9}  {:>14}  {:>14}",
        "N", "par µs/N", "batch µs/N", "speedup", "batch tot µs", "bv tot µs",
      );
      println!("  {}", "─".repeat(78));

      for &n in &sizes {
        let batch_itps: Vec<_> = std::iter::repeat(rep_itp.clone()).take(n).collect();

        // --- parallel path ---
        let t_par = Instant::now();
        let par_ok = match prove_instruction_zk_receipts_parallel(
          black_box(batch_itps.clone()),
          zkp_workers,
        ) {
          Ok(r) => {
            black_box(r);
            true
          }
          Err(_) => false,
        };
        let par_us = t_par.elapsed().as_secs_f64() * 1_000_000.0;

        if !par_ok {
          println!("  {:>5}  — (parallel failed)", n);
          continue;
        }

        // --- batch path ---
        let t_batch = Instant::now();
        let batch_receipt = match prove_batch_transaction_zk_receipt(black_box(&batch_itps)) {
          Ok(r) => r,
          Err(e) => {
            println!("  {:>5}  — batch prove failed: {e}", n);
            continue;
          }
        };
        let batch_us = t_batch.elapsed().as_secs_f64() * 1_000_000.0;

        // --- batch verify ---
        let batch_stmts: Vec<_> = std::iter::repeat(rep_stmt.clone()).take(n).collect();
        let t_bv = Instant::now();
        let bv_ok = black_box(verify_batch_transaction_zk_receipt(
          black_box(&batch_stmts),
          black_box(&batch_receipt),
        ));
        let bv_us = t_bv.elapsed().as_secs_f64() * 1_000_000.0;

        let speedup = if batch_us > 0.0 {
          par_us / batch_us
        } else {
          f64::INFINITY
        };
        println!(
          "  {:>5}  {:>14.1}  {:>14.1}  {:>8.2}×  {:>14.1}  {:>14.1}  verify_ok={bv_ok}",
          n,
          par_us / n as f64,
          batch_us / n as f64,
          speedup,
          batch_us,
          bv_us,
        );

        // Suppress unused-variable lint on the typed receipt.
        let _: BatchTransactionZkReceipt = batch_receipt;
      }
    }
  }

  // ── Memory consistency (SMAT LogUp STARK) ─────────────────────────────────────────────────
  // Benchmarks prove_memory_consistency / verify_memory_consistency for growing N.
  // Each test case is a write-then-read sequence: N writes to distinct 32-byte-aligned
  // addresses followed by N reads, all with matching values.
  if !no_stark {
    fn make_mem_claims(n: usize, rng: &mut StdRng) -> Vec<MemAccessClaim> {
      let mut claims = Vec::with_capacity(n * 2);
      for i in 0..n {
        let addr = (i as u64) * 32;
        let mut value = [0u8; 32];
        rng.fill(&mut value);
        // write first: wv=0 (first write to this address)
        claims.push(MemAccessClaim {
          rw_counter: (i * 2) as u64 + 1,
          addr,
          is_write: true,
          value,
          write_version: 0,
        });
        // read after write: wv=1 (reads the value written at version 0,
        // so the read's write_version = number of writes so far = 1)
        claims.push(MemAccessClaim {
          rw_counter: (i * 2) as u64 + 2,
          addr,
          is_write: false,
          value,
          write_version: 1,
        });
      }
      claims
    }

    let claim_sizes = [1usize, 4, 8, 16, 32, 64];
    let mem_seed = seed ^ 0xDEAD_BEEF_u64;
    let mut mem_rng = StdRng::seed_from_u64(mem_seed);

    println!();
    println!(
      "── Memory consistency STARK (SMAT LogUp AIR) ────────────────────────────────────────────"
    );
    println!(
      "{:>8}  {:>14}  {:>14}  {:>12}",
      "n_claims", "prove_us", "verify_us", "verify_ok"
    );
    println!("{}", "─".repeat(56));

    for &n in &claim_sizes {
      let claims = make_mem_claims(n, &mut mem_rng);
      let n_claims = claims.len(); // 2*n (write+read pairs)

      let t_prove = Instant::now();
      let proof = match prove_memory_consistency(black_box(&claims)) {
        Ok(p) => p,
        Err(e) => {
          println!("{:>8}  — prove failed: {e}", n_claims);
          continue;
        }
      };
      let prove_us = t_prove.elapsed().as_secs_f64() * 1_000_000.0;

      let t_verify = Instant::now();
      let ok = black_box(verify_memory_consistency(black_box(&proof), black_box(&claims)));
      let verify_us = t_verify.elapsed().as_secs_f64() * 1_000_000.0;

      println!(
        "{:>8}  {:>14.2}  {:>14.2}  {:>12}",
        n_claims, prove_us, verify_us, ok
      );
    }
  }

  // ---- Save baseline ----
  if let Some(ref path) = save_baseline_path {
    if current_baseline.is_empty() {
      println!();
      println!("[warn] no timing data collected — baseline not saved (run without --no-stark)");
    } else {
      match save_baseline(path, &current_baseline) {
        Ok(()) => println!(
          "\n  baseline saved  : {path} ({} entries)",
          current_baseline.len()
        ),
        Err(e) => eprintln!("[error] {e}"),
      }
    }
  }
}
