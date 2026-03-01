use std::hint::black_box;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use revm::bytecode::opcode;
use revm::primitives::{Bytes, U256};

use zprove_core::execute::{
  execute_bytecode_and_prove_with_batch_zkp, execute_bytecode_trace,
  prove_transaction_proof_with_batch_zkp,
  reset_proof_phase_timings, read_proof_phase_timings,
};
use zprove_core::transition::TransactionProof;
use zprove_core::semantic_proof::{
  compile_proof, compute_addmod, compute_byte, compute_exp, compute_mulmod, compute_signextend,
};
use zprove_core::transition::{
  InstructionTransitionProof, InstructionTransitionStatement, VmState,
  build_batch_zk_receipt_prep, prove_batch_transaction_zk_receipt_with_prep,
  prove_instruction, verify_batch_transaction_zk_receipt,
};

// ============================================================
// Report data model
// ============================================================

#[derive(Default)]
struct BenchReport {
  timestamp: String,
  seed: u64,
  samples: usize,
  log_rows: usize,
  rows: Vec<BatchBenchRow>,
  tps: Option<TpsSection>,
}

struct TpsSection {
  ncpus: usize,
  tx_count: usize,
  seq_us: f64,
  par_us: f64,
  /// If `--split` was passed: (seq_trace_us, seq_prove_us, par_trace_us, par_prove_us)
  split: Option<(f64, f64, f64, f64)>,
  /// If `--split` was passed: phase breakdown of the sequential prove run.
  /// (seq_logup_ns, seq_fri_commit_ns, par_logup_ns, par_fri_commit_ns)
  phase: Option<(u64, u64, u64, u64)>,
}

struct BatchBenchRow {
  group: String,
  rows_per_instr: usize,
  batch_count: usize,
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
    "<p>timestamp: <b>{}</b></p><p>seed: <b>{}</b> | samples per opcode: <b>{}</b> | log_rows (N): <b>{}</b> (max rows/batch = 2^N = {})</p>",
    r.timestamp,
    r.seed,
    r.samples,
    r.log_rows,
    1usize << r.log_rows
  );

  let headers = [
    "opcode group",
    "rows/instr",
    "batch count",
    "prove µs",
    "prove/instr µs",
    "verify µs",
    "verify/instr µs",
    "verify_ok",
  ];
  let th_row: String = headers
    .iter()
    .map(|h| format!("<th>{h}</th>"))
    .collect::<String>();

  let rows: String = r
    .rows
    .iter()
    .map(|row| {
      let cls = if row.verify_ok { "ok" } else { "fail" };
      let ok_str = if row.verify_ok { "true" } else { "false" };
      let n = row.batch_count;
      format!(
        "<tr><td style=\"text-align:left\">{g}</td><td>{ri}</td><td>{n}</td>\
         <td>{p:.1}</td><td>{pp:.1}</td><td>{v:.1}</td><td>{vp:.1}</td>\
         <td class=\"{cls}\">{ok}</td></tr>",
        g = row.group,
        ri = row.rows_per_instr,
        n = n,
        p = row.prove_us,
        pp = row.prove_us / n.max(1) as f64,
        v = row.verify_us,
        vp = row.verify_us / n.max(1) as f64,
        cls = cls,
        ok = ok_str,
      )
    })
    .collect::<String>();

  let tps_section = if let Some(tps) = &r.tps {
    let seq_ms = tps.seq_us / 1000.0;
    let par_ms = tps.par_us / 1000.0;
    let seq_tps = tps.tx_count as f64 / (tps.seq_us / 1_000_000.0);
    let par_tps = tps.tx_count as f64 / (tps.par_us / 1_000_000.0);
    let speedup = tps.seq_us / tps.par_us;
    let base = format!(
      "<h2>ERC-20 transfer TPS</h2>\
       <p>bytecode: GT + ISZERO + SUB + MSTORE + ADD + MSTORE &mdash; 6 proven ops / transfer</p>\
       <p>tx_count = {} &nbsp;|&nbsp; pool threads = {ncpus}</p>\
       <table><thead><tr><th style='text-align:left'>mode</th>\
       <th>tx count</th><th>wall ms</th><th>ms / tx</th><th>TPS</th></tr></thead><tbody>\
       <tr><td style='text-align:left'>sequential</td><td>{n}</td>\
       <td>{sm:.1}</td><td>{spt:.2}</td><td>{st:.0}</td></tr>\
       <tr><td style='text-align:left'>parallel ({ncpus} threads)</td><td>{n}</td>\
       <td>{pm:.1}</td><td>{ppt:.2}</td><td>{pt:.0}</td></tr>\
       </tbody></table>\
       <p>speedup: <b>{sp:.2}&times;</b></p>",
      tps.tx_count,
      ncpus = tps.ncpus,
      n = tps.tx_count,
      sm = seq_ms,
      spt = seq_ms / tps.tx_count as f64,
      st = seq_tps,
      pm = par_ms,
      ppt = par_ms / tps.tx_count as f64,
      pt = par_tps,
      sp = speedup,
    );

    let split_html = if let Some((st_us, sp_us, pt_us, pp_us)) = tps.split {
      let n = tps.tx_count as f64;
      let phase_html = if let Some((sl_ns, sf_ns, pl_ns, pf_ns)) = tps.phase {
        let sl_ms = sl_ns as f64 / 1_000_000.0;
        let sf_ms = sf_ns as f64 / 1_000_000.0;
        let pl_ms = pl_ns as f64 / 1_000_000.0;
        let pf_ms = pf_ns as f64 / 1_000_000.0;
        let total_s = sl_ms + sf_ms;
        format!(
          "<h2>Sub-phase breakdown: LogUp vs FRI+Commit</h2>\
           <table><thead><tr>\
           <th style='text-align:left'>mode</th>\
           <th>logup ms</th><th>logup ms/tx</th>\
           <th>fri+com ms</th><th>fri+com ms/tx</th>\
           <th>logup %</th><th>fri+com %</th>\
           </tr></thead><tbody>\
           <tr><td style='text-align:left'>sequential</td>\
           <td>{sl:.3}</td><td>{slt:.4}</td>\
           <td>{sf:.1}</td><td>{sft:.2}</td>\
           <td>{slp:.1}%</td><td>{sfp:.1}%</td></tr>\
           <tr><td style='text-align:left'>parallel ({ncpus} threads, amortized)</td>\
           <td>{pl:.3}</td><td>{plt:.4}</td>\
           <td>{pf_:.1}</td><td>{pft:.2}</td>\
           <td>—</td><td>—</td></tr>\
           </tbody></table>",
          ncpus = tps.ncpus,
          sl = sl_ms, slt = sl_ms / n,
          sf = sf_ms, sft = sf_ms / n,
          slp = sl_ms / total_s * 100.0,
          sfp = sf_ms / total_s * 100.0,
          pl = pl_ms, plt = pl_ms / n,
          pf_ = pf_ms, pft = pf_ms / n,
        )
      } else {
        String::new()
      };
      format!(
        "<h2>ERC-20 phase breakdown (--split)</h2>\
         <table><thead><tr>\
         <th style='text-align:left'>mode</th>\
         <th>trace ms</th><th>trace ms/tx</th>\
         <th>prove ms</th><th>prove ms/tx</th>\
         <th>total ms</th><th>TPS</th>\
         </tr></thead><tbody>\
         <tr><td style='text-align:left'>sequential</td>\
         <td>{stm:.1}</td><td>{stpt:.2}</td>\
         <td>{spm:.1}</td><td>{sppt:.2}</td>\
         <td>{stot:.1}</td><td>{stps:.0}</td></tr>\
         <tr><td style='text-align:left'>parallel ({ncpus} threads)</td>\
         <td>{ptm:.1}</td><td>{ptpt:.2}</td>\
         <td>{ppm:.1}</td><td>{pppt:.2}</td>\
         <td>{ptot:.1}</td><td>{ptps:.0}</td></tr>\
         </tbody></table>\
         <p>prove / total (seq): <b>{pr_frac:.0}%</b> &nbsp; trace: <b>{tr_frac:.0}%</b></p>\
         {phase_html}",
        ncpus = tps.ncpus,
        stm  = st_us / 1000.0,
        stpt = st_us / 1000.0 / n,
        spm  = sp_us / 1000.0,
        sppt = sp_us / 1000.0 / n,
        stot = (st_us + sp_us) / 1000.0,
        stps = n / ((st_us + sp_us) / 1_000_000.0),
        ptm  = pt_us / 1000.0,
        ptpt = pt_us / 1000.0 / n,
        ppm  = pp_us / 1000.0,
        pppt = pp_us / 1000.0 / n,
        ptot = (pt_us + pp_us) / 1000.0,
        ptps = n / ((pt_us + pp_us) / 1_000_000.0),
        pr_frac = sp_us / (st_us + sp_us) * 100.0,
        tr_frac = st_us / (st_us + sp_us) * 100.0,
        phase_html = phase_html,
      )
    } else {
      String::new()
    };

    base + &split_html
  } else {
    String::new()
  };

  format!(
    "<!DOCTYPE html><html><head><meta charset=\"UTF-8\">\
     <title>zprove batch bench</title><style>{css}</style></head>\
     <body><h1>zprove batch bench</h1>{meta}\
     <h2>prove_batch_transaction_zk_receipt / verify_batch_transaction_zk_receipt</h2>\
     <table><thead><tr>{th_row}</tr></thead><tbody>{rows}</tbody></table>\
     {tps_section}\
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

// ============================================================
// ERC-20 transfer bytecode helpers
// ============================================================

/// Push a u128 value as a 32-byte big-endian word (PUSH32 opcode).
fn push32_u128(code: &mut Vec<u8>, v: u128) {
  code.push(opcode::PUSH32);
  let mut b = [0u8; 32];
  b[16..].copy_from_slice(&v.to_be_bytes());
  code.extend_from_slice(&b);
}

/// Minimal ERC-20 transfer: overflow check + debit sender + credit recipient.
///
/// Proven ops (per transfer):
///   GT      – amount > sender_balance?
///   ISZERO  – invert result (1 = valid)
///   SUB     – new_sender  = sender_balance - amount
///   MSTORE  – write new_sender  to mem[0]
///   ADD     – new_recip   = recipient_balance + amount
///   MSTORE  – write new_recip   to mem[32]
fn erc20_transfer_bytecode(sender_balance: u128, recipient_balance: u128, amount: u128) -> Bytes {
  let mut code: Vec<u8> = Vec::new();

  // ── Overflow check ────────────────────────────────────────────────────────
  push32_u128(&mut code, sender_balance);  // [sender_bal]
  push32_u128(&mut code, amount);           // [amount, sender_bal]
  code.push(opcode::GT);                    // [amount > sender_bal]
  code.push(opcode::ISZERO);                // [valid (1 = ok)]
  code.push(opcode::POP);                   // []

  // ── Debit sender ─────────────────────────────────────────────────────────
  push32_u128(&mut code, amount);           // [amount]
  push32_u128(&mut code, sender_balance);   // [sender_bal, amount]
  code.push(opcode::SUB);                   // [sender_bal - amount]
  code.push(opcode::PUSH1); code.push(0x00);
  code.push(opcode::MSTORE);                // mem[0]  = new_sender_bal

  // ── Credit recipient ─────────────────────────────────────────────────────
  push32_u128(&mut code, recipient_balance); // [recipient_bal]
  push32_u128(&mut code, amount);            // [amount, recipient_bal]
  code.push(opcode::ADD);                    // [recipient_bal + amount]
  code.push(opcode::PUSH1); code.push(0x20);
  code.push(opcode::MSTORE);                 // mem[32] = new_recipient_bal

  code.push(opcode::STOP);
  Bytes::from(code)
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
  BenchCase {
    group: "ADD",
    opcode: opcode::ADD,
    inputs: vec![a, b],
    output: add_u256_mod(&a, &b),
  }
}

fn make_sub_case(rng: &mut StdRng, _idx: usize) -> BenchCase {
  let a = u256_bytes(rng.random::<u128>());
  let b = u256_bytes(rng.random::<u128>());
  BenchCase {
    group: "SUB",
    opcode: opcode::SUB,
    inputs: vec![a, b],
    output: sub_u256_mod(&a, &b),
  }
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
  BenchCase {
    group: "AND",
    opcode: opcode::AND,
    inputs: vec![a, b],
    output: c,
  }
}

fn make_or_case(rng: &mut StdRng, _idx: usize) -> BenchCase {
  let a = random_bytes32(rng);
  let b = random_bytes32(rng);
  let mut c = [0u8; 32];
  for i in 0..32 {
    c[i] = a[i] | b[i];
  }
  BenchCase {
    group: "OR",
    opcode: opcode::OR,
    inputs: vec![a, b],
    output: c,
  }
}

fn make_xor_case(rng: &mut StdRng, _idx: usize) -> BenchCase {
  let a = random_bytes32(rng);
  let b = random_bytes32(rng);
  let mut c = [0u8; 32];
  for i in 0..32 {
    c[i] = a[i] ^ b[i];
  }
  BenchCase {
    group: "XOR",
    opcode: opcode::XOR,
    inputs: vec![a, b],
    output: c,
  }
}

fn make_not_case(rng: &mut StdRng, _idx: usize) -> BenchCase {
  let a = random_bytes32(rng);
  let mut c = [0u8; 32];
  for i in 0..32 {
    c[i] = !a[i];
  }
  BenchCase {
    group: "NOT",
    opcode: opcode::NOT,
    inputs: vec![a],
    output: c,
  }
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
  let b = if same {
    a
  } else {
    u256_bytes(rng.random::<u128>())
  };
  BenchCase {
    group: "EQ",
    opcode: opcode::EQ,
    inputs: vec![a, b],
    output: bool_word(a == b),
  }
}

fn make_iszero_case(rng: &mut StdRng, _idx: usize) -> BenchCase {
  let v: u128 = if rng.random::<bool>() {
    0
  } else {
    rng.random::<u128>()
  };
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
        let hi = if src + 1 < 32 {
          value[src + 1] >> (8 - bit_shift)
        } else {
          0
        };
        lo | hi
      };
    }
  }
  out
}

fn evm_shr_u128(shift: u64, value: u128) -> u128 {
  if shift >= 128 {
    0
  } else {
    value.wrapping_shr(shift as u32)
  }
}

fn evm_sar_u128(shift: u64, value: i128) -> i128 {
  if shift >= 128 {
    if value < 0 { -1 } else { 0 }
  } else {
    value >> shift
  }
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

fn make_addmod_case(rng: &mut StdRng, _idx: usize) -> BenchCase {
  let a = random_bytes32(rng);
  let b = random_bytes32(rng);
  let n = random_bytes32(rng);
  let out = compute_addmod(&a, &b, &n);
  BenchCase {
    group: "ADDMOD",
    opcode: opcode::ADDMOD,
    inputs: vec![a, b, n],
    output: out,
  }
}

fn make_mulmod_case(rng: &mut StdRng, _idx: usize) -> BenchCase {
  let a = random_bytes32(rng);
  let b = random_bytes32(rng);
  let n = random_bytes32(rng);
  let out = compute_mulmod(&a, &b, &n);
  BenchCase {
    group: "MULMOD",
    opcode: opcode::MULMOD,
    inputs: vec![a, b, n],
    output: out,
  }
}

fn make_exp_case(rng: &mut StdRng, _idx: usize) -> BenchCase {
  // Small base/exponent to keep bench runtime reasonable.
  let base = u256_bytes(rng.random::<u16>() as u128);
  let exp = u256_bytes(rng.random::<u8>() as u128);
  let out = compute_exp(&base, &exp);
  BenchCase {
    group: "EXP",
    opcode: opcode::EXP,
    inputs: vec![base, exp],
    output: out,
  }
}

fn make_byte_case(rng: &mut StdRng, _idx: usize) -> BenchCase {
  let i = u256_bytes((rng.random::<u8>() % 32) as u128);
  let x = random_bytes32(rng);
  let out = compute_byte(&i, &x);
  BenchCase {
    group: "BYTE",
    opcode: opcode::BYTE,
    inputs: vec![i, x],
    output: out,
  }
}

fn make_signextend_case(rng: &mut StdRng, _idx: usize) -> BenchCase {
  let b = u256_bytes((rng.random::<u8>() % 32) as u128);
  let x = random_bytes32(rng);
  let out = compute_signextend(&b, &x);
  BenchCase {
    group: "SIGNEXTEND",
    opcode: opcode::SIGNEXTEND,
    inputs: vec![b, x],
    output: out,
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
  make_addmod_case,
  make_mulmod_case,
  make_exp_case,
  make_byte_case,
  make_signextend_case,
];

// ============================================================
// ITP / Statement builders
// ============================================================

fn build_itp(case: &BenchCase) -> InstructionTransitionProof {
  let proof = prove_instruction(case.opcode, &case.inputs, &[case.output]);
  InstructionTransitionProof {
    opcode: case.opcode,
    pc: 0,
    stack_inputs: case.inputs.clone(),
    stack_outputs: vec![case.output],
    semantic_proof: proof,
    memory_claims: vec![],
    storage_claims: vec![],
    stack_claims: vec![],
    mcopy_claim: None,
    return_data_claim: None,
    call_context_claim: None,
    keccak_claim: None,
    external_state_claim: None,
    sub_call_claim: None,
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
      storage_root: [0u8; 32],
    },
    s_next: VmState {
      opcode: case.opcode,
      pc: 1,
      sp: 1,
      stack: vec![case.output],
      memory_root: [0u8; 32],
      storage_root: [0u8; 32],
    },
    accesses: Vec::new(),
    sub_call_claim: None,
    mcopy_claim: None,
    external_state_claim: None,
  }
}

// ============================================================
// Arg parsing
// ============================================================

fn parse_arg_bool(args: &[String], key: &str) -> bool {
  args.iter().any(|a| a == key)
}

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
    let leap: u64 = if y % 400 == 0 {
      1
    } else if y % 100 == 0 {
      0
    } else if y % 4 == 0 {
      1
    } else {
      0
    };
    let ydays = 365 + leap;
    if rem < ydays {
      break;
    }
    rem -= ydays;
    y += 1;
  }
  let leap: u64 = if y % 400 == 0 {
    1
  } else if y % 100 == 0 {
    0
  } else if y % 4 == 0 {
    1
  } else {
    0
  };
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
  let log_rows = parse_arg_usize(&args, "--log-rows", 10);
  let seed_opt = parse_arg_opt_u64(&args, "--seed");
  let seed = seed_opt.unwrap_or_else(rand::random::<u64>);
  // --split  : also measure trace-gen and FRI/commitment phases separately
  let do_split = parse_arg_bool(&args, "--split");
  let max_rows: usize = 1 << log_rows;

  // ── Warm up AIR constraint hint caches ───────────────────────────────────
  //
  // Pre-compute symbolic constraint metadata (constraint count +
  // log_num_quotient_chunks) for all well-known AIR types.  Without this, the
  // first `prove_*` call for each AIR type pays ~15-70 ms of symbolic
  // evaluation overhead; subsequent calls use the cached hints for free.
  zprove_core::zk_proof::air_cache::warm_up();

  println!("zprove batch bench");
  match seed_opt {
    Some(s) => println!("  seed     : {s} (fixed)"),
    None => println!("  seed     : {seed} (random)"),
  }
  println!("  samples  : {samples}  (per opcode)");
  println!("  iters    : {iters}  (repetitions per bench point)");
  println!("  log-rows : {log_rows}  (max rows/batch = 2^N = {max_rows})");

  println!();
  println!("── prove_batch_transaction_zk_receipt / verify_batch_transaction_zk_receipt ──");
  println!("   (N = {log_rows}, max rows/batch = {max_rows})");
  println!(
    "  {:<16}  {:>10}  {:>6}  {:>12}  {:>12}  {:>12}  {:>12}  {:>10}",
    "group",
    "rows/instr",
    "batch",
    "prove µs",
    "prove/instr",
    "verify µs",
    "verify/instr",
    "verify_ok"
  );
  println!("  {}", "─".repeat(100));

  let mut report = BenchReport {
    timestamp: utc_timestamp(),
    seed,
    samples,
    log_rows,
    rows: Vec::new(),
    tps: None,
  };

  for (gen_idx, make_case) in CASE_GENERATORS.iter().enumerate() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ (gen_idx as u64));

    // ── Generate `samples` cases for this generator, cycling for batch ────
    let mut seed_cases: Vec<BenchCase> = (0..samples.max(1)).map(|i| make_case(&mut rng, i)).collect();
    if seed_cases.is_empty() {
      seed_cases.push(make_case(&mut rng, 0));
    }

    // ── Build batch online: add instructions until rows would overflow ─────
    //
    // We prove each sample instruction once, check its real row contribution,
    // and stop (discarding the overflow instruction) the moment the cumulative
    // row total would exceed max_rows.  This handles input-dependent row
    // counts (e.g. ISZERO: v=0 → 2 rows, v≠0 → 18 rows) without any upfront
    // probing or worst-case estimation.
    let mut batch_itps: Vec<InstructionTransitionProof> = Vec::new();
    let mut batch_stmts: Vec<InstructionTransitionStatement> = Vec::new();
    let mut total_rows = 0usize;
    for c in seed_cases.iter().cycle().take(max_rows) {
      let itp = build_itp(c);
      let row_count = compile_proof(&itp.semantic_proof).len().max(1);
      // Always include the very first instruction even if it alone exceeds
      // max_rows; otherwise stop before overflow.
      if !batch_itps.is_empty() && total_rows + row_count > max_rows {
        break;
      }
      total_rows += row_count;
      batch_itps.push(itp);
      batch_stmts.push(build_statement(c));
    }
    if batch_itps.is_empty() {
      eprintln!("  {:<16} : skipped (no valid ITPs)", seed_cases[0].group);
      continue;
    }
    let actual_count = batch_itps.len();
    let rows_per_instr = total_rows / actual_count.max(1);

    // ── pre-build receipt prep (WFF + manifest + preprocessed setup) ──────
    // This work is amortised outside the timing loop so the bench measures
    // only the STARK proving cost, not WFF tree construction.
    let receipt_prep = match build_batch_zk_receipt_prep(&batch_itps) {
      Ok(p) => p,
      Err(e) => {
        eprintln!("  {:<16} : prep failed: {e}", seed_cases[0].group);
        continue;
      }
    };

    // ── prove (best of iters) ─────────────────────────────────────────────
    let mut prove_us = f64::MAX;
    let mut last_receipt = None;
    for _ in 0..iters {
      let t = Instant::now();
      match prove_batch_transaction_zk_receipt_with_prep(black_box(&batch_itps), &receipt_prep) {
        Ok(receipt) => {
          let us = t.elapsed().as_secs_f64() * 1_000_000.0;
          if us < prove_us {
            prove_us = us;
          }
          last_receipt = Some(receipt);
        }
        Err(e) => {
          eprintln!("  {:<16} : prove failed: {e}", seed_cases[0].group);
          break;
        }
      }
    }
    let receipt = match last_receipt {
      Some(r) => r,
      None => continue,
    };

    // ── verify (best of iters) ────────────────────────────────────────────
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
      "  {:<16}  {:>10}  {:>6}  {:>12.1}  {:>12.1}  {:>12.1}  {:>12.1}  {:>10}",
      seed_cases[0].group,
      rows_per_instr,
      actual_count,
      prove_us,
      prove_us / actual_count as f64,
      verify_us,
      verify_us / actual_count as f64,
      verify_ok,
    );
    report.rows.push(BatchBenchRow {
      group: seed_cases[0].group.to_string(),
      rows_per_instr,
      batch_count: actual_count,
      prove_us,
      verify_us,
      verify_ok,
    });
  }

  // ── ERC-20 transfer TPS ────────────────────────────────────────────────────
  println!();
  println!("── ERC-20 transfer TPS ────────────────────────────────────────────────────────");
  println!("   (GT + ISZERO + SUB + MSTORE + ADD + MSTORE per transfer)");

  let ncpus = std::thread::available_parallelism()
    .map(|n| n.get())
    .unwrap_or(4);
  let tx_count = ncpus * 2; // two waves to keep pool busy

  // Generate varied ERC-20 transfers to avoid caching effects.
  let mut tps_rng = rand::rngs::StdRng::seed_from_u64(seed ^ 0xEE20EE20u64);
  let bytecodes: Vec<Bytes> = (0..tx_count)
    .map(|_| {
      let sender_bal: u128 = tps_rng.random::<u64>() as u128 + 100_000;
      let recipient_bal: u128 = tps_rng.random::<u32>() as u128;
      let amount: u128 = (tps_rng.random::<u32>() as u128 % 1000) + 1;
      erc20_transfer_bytecode(sender_bal, recipient_bal, amount)
    })
    .collect();

  // Warm-up: one tx to initialise pool, AIR caches, JIT, etc.
  let warmup = erc20_transfer_bytecode(1_000_000, 0, 100);
  let _ = execute_bytecode_and_prove_with_batch_zkp(warmup, Bytes::default(), U256::ZERO);

  // ── Sequential ──────────────────────────────────────────────────────────
  let t_seq = Instant::now();
  for bc in &bytecodes {
    execute_bytecode_and_prove_with_batch_zkp(bc.clone(), Bytes::default(), U256::ZERO)
      .expect("erc20 seq prove failed");
  }
  let seq_us = t_seq.elapsed().as_secs_f64() * 1_000_000.0;
  let seq_tps = tx_count as f64 / (seq_us / 1_000_000.0);

  // ── Parallel ────────────────────────────────────────────────────────────
  let (done_tx, done_rx) = std::sync::mpsc::channel::<Result<(), String>>();
  let t_par = Instant::now();
  for bc in &bytecodes {
    let bc = bc.clone();
    let tx = done_tx.clone();
    std::thread::spawn(move || {
      let r = execute_bytecode_and_prove_with_batch_zkp(bc, Bytes::default(), U256::ZERO)
        .map(|_| ());
      tx.send(r).ok();
    });
  }
  drop(done_tx);
  for r in done_rx {
    r.expect("erc20 parallel prove failed");
  }
  let par_us = t_par.elapsed().as_secs_f64() * 1_000_000.0;
  let par_tps = tx_count as f64 / (par_us / 1_000_000.0);

  println!(
    "  {:<28}  {:>8}  {:>10.1}  {:>10.2}  {:>8.0}",
    "mode", "tx", "wall ms", "ms/tx", "TPS"
  );
  println!("  {}", "─".repeat(70));
  println!(
    "  {:<28}  {:>8}  {:>10.1}  {:>10.2}  {:>8.0}",
    "sequential",
    tx_count,
    seq_us / 1000.0,
    seq_us / 1000.0 / tx_count as f64,
    seq_tps,
  );
  println!(
    "  {:<28}  {:>8}  {:>10.1}  {:>10.2}  {:>8.0}",
    format!("parallel ({ncpus} threads)"),
    tx_count,
    par_us / 1000.0,
    par_us / 1000.0 / tx_count as f64,
    par_tps,
  );
  println!("  speedup: {:.2}×", seq_us / par_us);

  // ── Phase breakdown (--split) ────────────────────────────────────────────
  let split_timings: Option<(f64, f64, f64, f64, u64, u64, u64, u64)> = if do_split {
    println!();
    println!("── ERC-20 phase breakdown (--split) ──────────────────────────────────────────");
    println!("   Phase 1: EVM trace generation    Phase 2: Commitment + FRI (batch STARK)");

    // Pre-generate all traces (shared by both sequential and parallel prove runs).
    let traces: Vec<TransactionProof> = bytecodes
      .iter()
      .map(|bc| {
        execute_bytecode_trace(bc.clone(), Bytes::default(), U256::ZERO)
          .expect("erc20 trace failed")
      })
      .collect();

    // ── sequential trace ──
    let t = Instant::now();
    for bc in &bytecodes {
      execute_bytecode_trace(bc.clone(), Bytes::default(), U256::ZERO)
        .expect("trace failed");
    }
    let seq_trace_us = t.elapsed().as_secs_f64() * 1_000_000.0;

    // ── sequential prove ──
    reset_proof_phase_timings();
    let t = Instant::now();
    for tr in &traces {
      prove_transaction_proof_with_batch_zkp(tr)
        .expect("erc20 prove failed");
    }
    let seq_prove_us = t.elapsed().as_secs_f64() * 1_000_000.0;
    let seq_phase = read_proof_phase_timings();

    // ── parallel trace ──
    let (tx_c, rx_c) = std::sync::mpsc::channel::<f64>();
    let t = Instant::now();
    for bc in &bytecodes {
      let bc = bc.clone();
      let s = tx_c.clone();
      std::thread::spawn(move || {
        let t2 = Instant::now();
        execute_bytecode_trace(bc, Bytes::default(), U256::ZERO)
          .expect("par trace failed");
        s.send(t2.elapsed().as_secs_f64() * 1_000_000.0).ok();
      });
    }
    drop(tx_c);
    let _: Vec<_> = rx_c.iter().collect();
    let par_trace_us = t.elapsed().as_secs_f64() * 1_000_000.0;

    // ── parallel prove ──
    reset_proof_phase_timings();
    let (done_tx2, done_rx2) = std::sync::mpsc::channel::<Result<(), String>>();
    let t = Instant::now();
    for tr in traces {
      let s = done_tx2.clone();
      std::thread::spawn(move || {
        s.send(prove_transaction_proof_with_batch_zkp(&tr)).ok();
      });
    }
    drop(done_tx2);
    for r in done_rx2 {
      r.expect("par prove failed");
    }
    let par_prove_us = t.elapsed().as_secs_f64() * 1_000_000.0;
    let par_phase = read_proof_phase_timings();

    println!(
      "  {:<28}  {:>12}  {:>12}  {:>12}  {:>12}",
      "mode", "trace ms", "trace ms/tx", "prove ms", "prove ms/tx"
    );
    println!("  {}", "─".repeat(80));
    let n = tx_count as f64;
    println!(
      "  {:<28}  {:>12.2}  {:>12.3}  {:>12.1}  {:>12.2}",
      "sequential",
      seq_trace_us / 1000.0, seq_trace_us / 1000.0 / n,
      seq_prove_us / 1000.0, seq_prove_us / 1000.0 / n,
    );
    println!(
      "  {:<28}  {:>12.2}  {:>12.3}  {:>12.1}  {:>12.2}",
      format!("parallel ({ncpus} threads)"),
      par_trace_us / 1000.0, par_trace_us / 1000.0 / n,
      par_prove_us / 1000.0, par_prove_us / 1000.0 / n,
    );
    println!(
      "  prove / total (seq): {:.1}%   trace: {:.1}%",
      seq_prove_us / (seq_trace_us + seq_prove_us) * 100.0,
      seq_trace_us / (seq_trace_us + seq_prove_us) * 100.0,
    );

    // ── phase breakdown (LogUp vs FRI+Commit) ──
    println!();
    println!("── Sub-phase breakdown (LogUp vs FRI+Commitment) ─────────────────────────────");
    println!(
      "  {:<34}  {:>12}  {:>12}  {:>12}  {:>12}",
      "mode", "logup ms", "logup ms/tx", "fri+com ms", "fri+com ms/tx"
    );
    println!("  {}", "─".repeat(86));
    let logup_ms_s   = seq_phase.logup_eval_ns() as f64 / 1_000_000.0;
    let fri_ms_s     = seq_phase.fri_commit_ns()  as f64 / 1_000_000.0;
    let total_prove_ms = logup_ms_s + fri_ms_s;
    let logup_frac   = logup_ms_s / total_prove_ms;
    let fri_frac     = 1.0 - logup_frac;

    // Parallel amortized wall-clock per tx = par_wall_clock / n * phase_fraction.
    // This is the "throughput" view: how fast each tx is processed end-to-end,
    // with the phase split estimated from the sequential ratios.
    let par_prove_ms = par_prove_us / 1000.0;
    let logup_ms_p   = par_prove_ms * logup_frac / n;
    let fri_ms_p     = par_prove_ms * fri_frac   / n;

    println!(
      "  {:<34}  {:>12.3}  {:>12.4}  {:>12.1}  {:>12.2}",
      "sequential",
      logup_ms_s, logup_ms_s / n,
      fri_ms_s, fri_ms_s / n,
    );
    println!(
      "  {:<34}  {:>12.3}  {:>12.4}  {:>12.1}  {:>12.2}",
      format!("parallel ({ncpus} threads, amortized)"),
      logup_ms_p * n, logup_ms_p,
      fri_ms_p   * n, fri_ms_p,
    );
    println!(
      "  logup / prove (seq): {:.1}%   fri+commit: {:.1}%",
      logup_frac * 100.0,
      fri_frac   * 100.0,
    );

    Some((seq_trace_us, seq_prove_us, par_trace_us, par_prove_us,
          seq_phase.logup_eval_ns(), seq_phase.fri_commit_ns(),
          // Amortized parallel wall-clock ns for each phase.
          (logup_ms_p * n * 1_000_000.0) as u64,
          (fri_ms_p   * n * 1_000_000.0) as u64))
  } else {
    None
  };

  let (split_tup, phase_tup) = match split_timings {
    Some((a, b, c, d, e, f, g, h)) => (Some((a, b, c, d)), Some((e, f, g, h))),
    None => (None, None),
  };
  report.tps = Some(TpsSection { ncpus, tx_count, seq_us, par_us, split: split_tup, phase: phase_tup });

  write_html_report(&report);
}
