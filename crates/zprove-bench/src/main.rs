use std::hint::black_box;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use revm::bytecode::opcode;

use zprove_core::transition::{
  InstructionTransitionProof, InstructionTransitionStatement, VmState, prove_instruction,
  prove_batch_transaction_zk_receipt, verify_batch_transaction_zk_receipt,
};

// ============================================================
// Report data model
// ============================================================

#[derive(Default)]
struct BenchReport {
  timestamp: String,
  seed: u64,
  samples: usize,
  rows: Vec<BatchBenchRow>,
}

struct BatchBenchRow {
  n: usize,
  prove_us: f64,
  verify_us: f64,
  verify_ok: bool,
}

// ============================================================
// HTML report
// ============================================================

fn render_html(r: &BenchReport) -> String {
  let css = r#"
    body { font-family: monospace; background: #1e1e2e; color: #cdd6f4; padding: 2em; }
    h1   { color: #89b4fa; }
    h2   { color: #74c7ec; border-bottom: 1px solid #45475a; padding-bottom: 0.3em; }
    p    { color: #a6adc8; margin: 0.3em 0; }
    table { border-collapse: collapse; margin-bottom: 2em; }
    th   { background: #313244; color: #89b4fa; padding: 0.5em 1.2em; text-align: right; }
    td   { padding: 0.4em 1.2em; text-align: right; border-top: 1px solid #313244; }
    .ok  { color: #a6e3a1; }
    .fail { color: #f38ba8; }
  "#;

  let meta = format!(
    "<p>timestamp: <b>{}</b></p><p>seed: <b>{}</b> | samples per opcode: <b>{}</b></p>",
    r.timestamp, r.seed, r.samples
  );

  let headers = ["N", "prove µs", "prove/N µs", "verify µs", "verify/N µs", "verify_ok"];
  let th_row: String = headers.iter().map(|h| format!("<th>{h}</th>")).collect::<String>();

  let rows: String = r
    .rows
    .iter()
    .map(|row| {
      let cls = if row.verify_ok { "ok" } else { "fail" };
      let ok_str = if row.verify_ok { "true" } else { "false" };
      format!(
        "<tr><td>{n}</td><td>{p:.1}</td><td>{pp:.1}</td><td>{v:.1}</td><td>{vp:.1}</td>\
         <td class=\"{cls}\">{ok}</td></tr>",
        n = row.n,
        p = row.prove_us,
        pp = row.prove_us / row.n as f64,
        v = row.verify_us,
        vp = row.verify_us / row.n as f64,
        cls = cls,
        ok = ok_str,
      )
    })
    .collect::<String>();

  format!(
    "<!DOCTYPE html><html><head><meta charset=\"UTF-8\">\
     <title>zprove batch bench</title><style>{css}</style></head>\
     <body><h1>zprove batch bench</h1>{meta}\
     <h2>prove_batch_transaction_zk_receipt / verify_batch_transaction_zk_receipt</h2>\
     <table><thead><tr>{th_row}</tr></thead><tbody>{rows}</tbody></table>\
     </body></html>"
  )
}

fn write_html_report(report: &BenchReport) {
  let path = format!("report/report-{}.html", report.timestamp);
  if let Some(parent) = std::path::Path::new(&path).parent() {
    let _ = std::fs::create_dir_all(parent);
  }
  match std::fs::write(&path, render_html(report)) {
    Ok(()) => println!("\n  report written : {path}"),
    Err(e) => eprintln!("[error] cannot write report: {e}"),
  }
}

// ============================================================
// Math helpers (used by case generators)
// ============================================================

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
    let mut diff = a[i] as i16 - b[i] as i16 - borrow;
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
// Bench case generators
//
// To add a new opcode:
//   1. Write `fn make_<OP>_case(rng: &mut StdRng, idx: usize) -> BenchCase`
//   2. Append it to CASE_GENERATORS
// ============================================================

#[derive(Clone)]
struct BenchCase {
  #[allow(dead_code)]
  group: &'static str,
  opcode: u8,
  inputs: Vec<[u8; 32]>,
  output: [u8; 32],
}

type CaseGen = fn(&mut StdRng, usize) -> BenchCase;

fn make_add_case(rng: &mut StdRng, _idx: usize) -> BenchCase {
  let a = u256_bytes(rng.random::<u128>());
  let b = u256_bytes(rng.random::<u128>());
  BenchCase { group: "ADD", opcode: opcode::ADD, inputs: vec![a, b], output: add_u256_mod(&a, &b) }
}

fn make_sub_case(rng: &mut StdRng, _idx: usize) -> BenchCase {
  let a = u256_bytes(rng.random::<u128>());
  let b = u256_bytes(rng.random::<u128>());
  BenchCase { group: "SUB", opcode: opcode::SUB, inputs: vec![a, b], output: sub_u256_mod(&a, &b) }
}

fn make_mul_sparse_case(rng: &mut StdRng, _idx: usize) -> BenchCase {
  let a = u256_bytes(rng.random::<u16>() as u128);
  let b = u256_bytes(rng.random::<u16>() as u128);
  BenchCase {
    group: "MUL(sparse)",
    opcode: opcode::MUL,
    inputs: vec![a, b],
    output: mul_u256_mod(&a, &b),
  }
}

fn make_mul_dense_case(rng: &mut StdRng, _idx: usize) -> BenchCase {
  let a = random_bytes32(rng);
  let b = random_bytes32(rng);
  BenchCase {
    group: "MUL(dense)",
    opcode: opcode::MUL,
    inputs: vec![a, b],
    output: mul_u256_mod(&a, &b),
  }
}

fn make_div_case(rng: &mut StdRng, _idx: usize) -> BenchCase {
  let a = rng.random::<u128>();
  let b = random_nonzero_u128(rng);
  BenchCase {
    group: "DIV",
    opcode: opcode::DIV,
    inputs: vec![u256_bytes(a), u256_bytes(b)],
    output: u256_bytes(a / b),
  }
}

fn make_mod_case(rng: &mut StdRng, _idx: usize) -> BenchCase {
  let a = rng.random::<u128>();
  let b = random_nonzero_u128(rng);
  BenchCase {
    group: "MOD",
    opcode: opcode::MOD,
    inputs: vec![u256_bytes(a), u256_bytes(b)],
    output: u256_bytes(a % b),
  }
}

fn make_sdiv_case(rng: &mut StdRng, _idx: usize) -> BenchCase {
  let (a, b) = random_sdiv_pair(rng);
  BenchCase {
    group: "SDIV",
    opcode: opcode::SDIV,
    inputs: vec![i256_bytes(a), i256_bytes(b)],
    output: i256_bytes(a / b),
  }
}

fn make_smod_case(rng: &mut StdRng, _idx: usize) -> BenchCase {
  let (a, b) = random_sdiv_pair(rng);
  BenchCase {
    group: "SMOD",
    opcode: opcode::SMOD,
    inputs: vec![i256_bytes(a), i256_bytes(b)],
    output: i256_bytes(a % b),
  }
}

fn make_and_case(rng: &mut StdRng, _idx: usize) -> BenchCase {
  let a = random_bytes32(rng);
  let b = random_bytes32(rng);
  let mut c = [0u8; 32];
  for i in 0..32 {
    c[i] = a[i] & b[i];
  }
  BenchCase { group: "AND", opcode: opcode::AND, inputs: vec![a, b], output: c }
}

fn make_or_case(rng: &mut StdRng, _idx: usize) -> BenchCase {
  let a = random_bytes32(rng);
  let b = random_bytes32(rng);
  let mut c = [0u8; 32];
  for i in 0..32 {
    c[i] = a[i] | b[i];
  }
  BenchCase { group: "OR", opcode: opcode::OR, inputs: vec![a, b], output: c }
}

fn make_xor_case(rng: &mut StdRng, _idx: usize) -> BenchCase {
  let a = random_bytes32(rng);
  let b = random_bytes32(rng);
  let mut c = [0u8; 32];
  for i in 0..32 {
    c[i] = a[i] ^ b[i];
  }
  BenchCase { group: "XOR", opcode: opcode::XOR, inputs: vec![a, b], output: c }
}

fn make_not_case(rng: &mut StdRng, _idx: usize) -> BenchCase {
  let a = random_bytes32(rng);
  let mut c = [0u8; 32];
  for i in 0..32 {
    c[i] = !a[i];
  }
  BenchCase { group: "NOT", opcode: opcode::NOT, inputs: vec![a], output: c }
}

fn make_lt_case(rng: &mut StdRng, _idx: usize) -> BenchCase {
  let a = rng.random::<u128>();
  let b = rng.random::<u128>();
  BenchCase {
    group: "LT",
    opcode: opcode::LT,
    inputs: vec![u256_bytes(a), u256_bytes(b)],
    output: bool_word(a < b),
  }
}

fn make_gt_case(rng: &mut StdRng, _idx: usize) -> BenchCase {
  let a = rng.random::<u128>();
  let b = rng.random::<u128>();
  BenchCase {
    group: "GT",
    opcode: opcode::GT,
    inputs: vec![u256_bytes(a), u256_bytes(b)],
    output: bool_word(a > b),
  }
}

fn make_slt_case(rng: &mut StdRng, _idx: usize) -> BenchCase {
  let (a, b) = random_sdiv_pair(rng);
  BenchCase {
    group: "SLT",
    opcode: opcode::SLT,
    inputs: vec![i256_bytes(a), i256_bytes(b)],
    output: bool_word(a < b),
  }
}

fn make_sgt_case(rng: &mut StdRng, _idx: usize) -> BenchCase {
  let (a, b) = random_sdiv_pair(rng);
  BenchCase {
    group: "SGT",
    opcode: opcode::SGT,
    inputs: vec![i256_bytes(a), i256_bytes(b)],
    output: bool_word(a > b),
  }
}

fn make_eq_case(rng: &mut StdRng, _idx: usize) -> BenchCase {
  let same = rng.random::<bool>();
  let a = u256_bytes(rng.random::<u128>());
  let b = if same { a } else { u256_bytes(rng.random::<u128>()) };
  BenchCase {
    group: "EQ",
    opcode: opcode::EQ,
    inputs: vec![a, b],
    output: bool_word(a == b),
  }
}

fn make_iszero_case(rng: &mut StdRng, _idx: usize) -> BenchCase {
  let v: u128 = if rng.random::<bool>() { 0 } else { rng.random::<u128>() };
  BenchCase {
    group: "ISZERO",
    opcode: opcode::ISZERO,
    inputs: vec![u256_bytes(v)],
    output: bool_word(v == 0),
  }
}

fn evm_shl_bytes(value: &[u8; 32], shift: u64) -> [u8; 32] {
  if shift >= 256 {
    return [0u8; 32];
  }
  let byte_shift = (shift / 8) as usize;
  let bit_shift = (shift % 8) as u32;
  let mut out = [0u8; 32];
  for k in 0..32usize {
    let src = k + byte_shift;
    if src < 32 {
      out[k] = if bit_shift == 0 {
        value[src]
      } else {
        let lo = value[src] << bit_shift;
        let hi = if src + 1 < 32 { value[src + 1] >> (8 - bit_shift) } else { 0 };
        lo | hi
      };
    }
  }
  out
}

fn evm_shr_u128(shift: u64, value: u128) -> u128 {
  if shift >= 128 { 0 } else { value.wrapping_shr(shift as u32) }
}

fn evm_sar_u128(shift: u64, value: i128) -> i128 {
  if shift >= 128 { if value < 0 { -1 } else { 0 } } else { value >> shift }
}

fn make_shl_case(rng: &mut StdRng, _idx: usize) -> BenchCase {
  let shift = (rng.random::<u8>() % 64) as u64;
  let value256 = u256_bytes(rng.random::<u128>());
  let result = evm_shl_bytes(&value256, shift);
  BenchCase {
    group: "SHL",
    opcode: opcode::SHL,
    inputs: vec![u256_bytes(shift as u128), value256],
    output: result,
  }
}

fn make_shr_case(rng: &mut StdRng, _idx: usize) -> BenchCase {
  let shift = (rng.random::<u8>() % 64) as u64;
  let value = rng.random::<u128>();
  BenchCase {
    group: "SHR",
    opcode: opcode::SHR,
    inputs: vec![u256_bytes(shift as u128), u256_bytes(value)],
    output: u256_bytes(evm_shr_u128(shift, value)),
  }
}

fn make_sar_case(rng: &mut StdRng, _idx: usize) -> BenchCase {
  let shift = (rng.random::<u8>() % 64) as u64;
  let value = rng.random::<i128>();
  BenchCase {
    group: "SAR",
    opcode: opcode::SAR,
    inputs: vec![u256_bytes(shift as u128), i256_bytes(value)],
    output: i256_bytes(evm_sar_u128(shift, value)),
  }
}

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
  make_shl_case,
  make_shr_case,
  make_sar_case,
];

fn bench_cases(samples: usize, seed: u64) -> Vec<BenchCase> {
  let mut rng = StdRng::seed_from_u64(seed);
  let mut cases = Vec::new();
  for sample_idx in 0..samples {
    for make_case in CASE_GENERATORS {
      cases.push(make_case(&mut rng, sample_idx));
    }
  }
  cases
}

// ============================================================
// ITP / Statement builders
// ============================================================

fn build_itp(case: &BenchCase) -> Option<InstructionTransitionProof> {
  let proof = prove_instruction(case.opcode, &case.inputs, &[case.output])?;
  Some(InstructionTransitionProof {
    opcode: case.opcode,
    pc: 0,
    stack_inputs: case.inputs.clone(),
    stack_outputs: vec![case.output],
    semantic_proof: Some(proof),
    memory_claims: vec![],
    storage_claims: vec![],
    stack_claims: vec![],
    return_data_claim: None,
    call_context_claim: None,
    keccak_claim: None,
    external_state_claim: None,
    sub_call_claim: None,
  })
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

// ============================================================
// Arg parsing
// ============================================================

fn parse_arg_usize(args: &[String], key: &str, default: usize) -> usize {
  let mut i = 0;
  while i < args.len() {
    if args[i] == key {
      if let Some(v) = args.get(i + 1).and_then(|x| x.parse::<usize>().ok()) {
        return v;
      }
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

// ============================================================
// Timestamp helper
// ============================================================

fn utc_timestamp() -> String {
  let now = std::time::SystemTime::now()
    .duration_since(std::time::UNIX_EPOCH)
    .unwrap_or_default()
    .as_secs();
  let s = now % 60;
  let m = (now / 60) % 60;
  let h = (now / 3600) % 24;
  let days = now / 86400;
  let mut y = 1970u64;
  let mut rem = days;
  loop {
    let leap: u64 =
      if y % 400 == 0 { 1 } else if y % 100 == 0 { 0 } else if y % 4 == 0 { 1 } else { 0 };
    let ydays = 365 + leap;
    if rem < ydays {
      break;
    }
    rem -= ydays;
    y += 1;
  }
  let leap: u64 =
    if y % 400 == 0 { 1 } else if y % 100 == 0 { 0 } else if y % 4 == 0 { 1 } else { 0 };
  let month_days: [u64; 12] = [31, 28 + leap, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  let mut mo = 1u64;
  for &md in &month_days {
    if rem < md {
      break;
    }
    rem -= md;
    mo += 1;
  }
  let d = rem + 1;
  format!("{y}{mo:02}{d:02}_{h:02}{m:02}{s:02}")
}

// ============================================================
// main
// ============================================================

fn main() {
  let args: Vec<String> = std::env::args().collect();
  let samples = parse_arg_usize(&args, "--samples", 3);
  let iters = parse_arg_usize(&args, "--iters", 1);
  let seed_opt = parse_arg_opt_u64(&args, "--seed");
  let seed = seed_opt.unwrap_or_else(rand::random::<u64>);

  println!("zprove batch bench");
  match seed_opt {
    Some(s) => println!("  seed    : {s} (fixed)"),
    None => println!("  seed    : {seed} (random)"),
  }
  println!("  samples : {samples}  (per opcode)");
  println!("  iters   : {iters}  (repetitions per batch size)");

  // ── Build ITPs ─────────────────────────────────────────────────────────────
  let cases = bench_cases(samples, seed);
  let pairs: Vec<(InstructionTransitionProof, InstructionTransitionStatement)> = cases
    .iter()
    .filter_map(|c| Some((build_itp(c)?, build_statement(c))))
    .collect();

  if pairs.is_empty() {
    eprintln!("no valid ITPs — check opcode implementations");
    return;
  }
  println!("  itps    : {} (from {} cases)", pairs.len(), cases.len());

  // ── Batch prove / verify ───────────────────────────────────────────────────
  let batch_sizes = [1usize, 2, 4, 8, 16, 32];

  println!();
  println!("── prove_batch_transaction_zk_receipt / verify_batch_transaction_zk_receipt ──");
  println!(
    "  {:>5}  {:>12}  {:>12}  {:>12}  {:>12}  {:>10}",
    "N", "prove µs", "prove/N µs", "verify µs", "verify/N µs", "verify_ok"
  );
  println!("  {}", "─".repeat(70));

  let mut report = BenchReport {
    timestamp: utc_timestamp(),
    seed,
    samples,
    rows: Vec::new(),
  };

  for &n in &batch_sizes {
    let batch_itps: Vec<InstructionTransitionProof> =
      pairs.iter().map(|(itp, _)| itp.clone()).cycle().take(n).collect();
    let batch_stmts: Vec<InstructionTransitionStatement> =
      pairs.iter().map(|(_, stmt)| stmt.clone()).cycle().take(n).collect();

    // ── prove (best of iters) ──────────────────────────────────────────────
    let mut prove_us = f64::MAX;
    let mut last_receipt = None;
    for _ in 0..iters {
      let t = Instant::now();
      match prove_batch_transaction_zk_receipt(black_box(&batch_itps)) {
        Ok(receipt) => {
          let us = t.elapsed().as_secs_f64() * 1_000_000.0;
          if us < prove_us {
            prove_us = us;
          }
          last_receipt = Some(receipt);
        }
        Err(e) => {
          eprintln!("  n={n}: prove failed: {e}");
          break;
        }
      }
    }
    let receipt = match last_receipt {
      Some(r) => r,
      None => continue,
    };

    // ── verify (best of iters) ─────────────────────────────────────────────
    let mut verify_us = f64::MAX;
    let mut verify_ok = false;
    for _ in 0..iters {
      let t = Instant::now();
      let ok = black_box(verify_batch_transaction_zk_receipt(
        black_box(&batch_stmts),
        black_box(&receipt),
      ));
      let us = t.elapsed().as_secs_f64() * 1_000_000.0;
      if us < verify_us {
        verify_us = us;
        verify_ok = ok;
      }
    }

    println!(
      "  {:>5}  {:>12.1}  {:>12.1}  {:>12.1}  {:>12.1}  {:>10}",
      n,
      prove_us,
      prove_us / n as f64,
      verify_us,
      verify_us / n as f64,
      verify_ok,
    );
    report.rows.push(BatchBenchRow { n, prove_us, verify_us, verify_ok });
  }

  write_html_report(&report);
}
