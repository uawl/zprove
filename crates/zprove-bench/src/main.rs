use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use revm::bytecode::opcode;
use zprove_core::semantic_proof::compile_proof;
use zprove_core::transition::{
  InstructionTransitionProof, InstructionTransitionStatement, InstructionTransitionZkReceipt,
  VmState, prove_instruction, prove_instruction_zk_receipt,
  prove_instruction_zk_receipts_parallel, verify_instruction_zk_receipt, verify_proof,
};
use zprove_core::zk_proof::{
  build_lut_steps_from_rows_add_family, build_lut_steps_from_rows_bit_family,
  build_lut_steps_from_rows_mul_family, prove_batch_stark_add_family,
  prove_batch_stark_add_family_with_params, prove_batch_stark_bit_family,
  prove_batch_stark_bit_family_with_params, prove_batch_stark_mul_family,
  prove_batch_stark_mul_family_with_params, prove_batch_wff_proofs,
  verify_batch_stark_add_family, verify_batch_stark_add_family_with_params,
  verify_batch_stark_bit_family, verify_batch_stark_bit_family_with_params,
  verify_batch_stark_mul_family, verify_batch_stark_mul_family_with_params,
  verify_batch_wff_proofs,
};

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

fn cmp_u128(a: u128, b: u128) -> bool {
  a < b
}

fn cmp_i128_lt(a: i128, b: i128) -> bool {
  a < b
}

fn bench_cases(samples: usize, seed: u64) -> Vec<BenchCase> {
  let mut rng = StdRng::seed_from_u64(seed);
  let mut cases = Vec::new();

  for sample_idx in 0..samples {
    let add_a = u256_bytes(rng.random::<u128>());
    let add_b = u256_bytes(rng.random::<u128>());
    let add_c = add_u256_mod(&add_a, &add_b);
    cases.push(BenchCase {
      name: format!("ADD#{:02}", sample_idx + 1),
      group: "ADD",
      opcode: opcode::ADD,
      inputs: vec![add_a, add_b],
      output: add_c,
    });

    let sub_a = u256_bytes(rng.random::<u128>());
    let sub_b = u256_bytes(rng.random::<u128>());
    let sub_c = sub_u256_mod(&sub_a, &sub_b);
    cases.push(BenchCase {
      name: format!("SUB#{:02}", sample_idx + 1),
      group: "SUB",
      opcode: opcode::SUB,
      inputs: vec![sub_a, sub_b],
      output: sub_c,
    });

    let mul_sparse_a = u256_bytes(rng.random::<u16>() as u128);
    let mul_sparse_b = u256_bytes(rng.random::<u16>() as u128);
    cases.push(BenchCase {
      name: format!("MULs#{:02}", sample_idx + 1),
      group: "MUL (sparse)",
      opcode: opcode::MUL,
      inputs: vec![mul_sparse_a, mul_sparse_b],
      output: mul_u256_mod(&mul_sparse_a, &mul_sparse_b),
    });

    let mul_dense_a = random_bytes32(&mut rng);
    let mul_dense_b = random_bytes32(&mut rng);
    cases.push(BenchCase {
      name: format!("MULd#{:02}", sample_idx + 1),
      group: "MUL (dense)",
      opcode: opcode::MUL,
      inputs: vec![mul_dense_a, mul_dense_b],
      output: mul_u256_mod(&mul_dense_a, &mul_dense_b),
    });

    let div_a = rng.random::<u128>();
    let div_b = random_nonzero_u128(&mut rng);
    cases.push(BenchCase {
      name: format!("DIV#{:02}", sample_idx + 1),
      group: "DIV",
      opcode: opcode::DIV,
      inputs: vec![u256_bytes(div_a), u256_bytes(div_b)],
      output: u256_bytes(div_a / div_b),
    });

    let mod_a = rng.random::<u128>();
    let mod_b = random_nonzero_u128(&mut rng);
    cases.push(BenchCase {
      name: format!("MOD#{:02}", sample_idx + 1),
      group: "MOD",
      opcode: opcode::MOD,
      inputs: vec![u256_bytes(mod_a), u256_bytes(mod_b)],
      output: u256_bytes(mod_a % mod_b),
    });

    let (sdiv_a, sdiv_b) = random_sdiv_pair(&mut rng);
    cases.push(BenchCase {
      name: format!("SDIV#{:02}", sample_idx + 1),
      group: "SDIV",
      opcode: opcode::SDIV,
      inputs: vec![i256_bytes(sdiv_a), i256_bytes(sdiv_b)],
      output: i256_bytes(sdiv_a / sdiv_b),
    });

    let (smod_a, smod_b) = random_sdiv_pair(&mut rng);
    cases.push(BenchCase {
      name: format!("SMOD#{:02}", sample_idx + 1),
      group: "SMOD",
      opcode: opcode::SMOD,
      inputs: vec![i256_bytes(smod_a), i256_bytes(smod_b)],
      output: i256_bytes(smod_a % smod_b),
    });

    let and_a = random_bytes32(&mut rng);
    let and_b = random_bytes32(&mut rng);
    let mut and_c = [0u8; 32];
    let mut or_c = [0u8; 32];
    let mut xor_c = [0u8; 32];
    let mut not_c = [0u8; 32];
    for i in 0..32 {
      and_c[i] = and_a[i] & and_b[i];
      or_c[i] = and_a[i] | and_b[i];
      xor_c[i] = and_a[i] ^ and_b[i];
      not_c[i] = !and_a[i];
    }

    cases.push(BenchCase {
      name: format!("AND#{:02}", sample_idx + 1),
      group: "AND",
      opcode: opcode::AND,
      inputs: vec![and_a, and_b],
      output: and_c,
    });
    cases.push(BenchCase {
      name: format!("OR#{:02}", sample_idx + 1),
      group: "OR",
      opcode: opcode::OR,
      inputs: vec![and_a, and_b],
      output: or_c,
    });
    cases.push(BenchCase {
      name: format!("XOR#{:02}", sample_idx + 1),
      group: "XOR",
      opcode: opcode::XOR,
      inputs: vec![and_a, and_b],
      output: xor_c,
    });
    cases.push(BenchCase {
      name: format!("NOT#{:02}", sample_idx + 1),
      group: "NOT",
      opcode: opcode::NOT,
      inputs: vec![and_a],
      output: not_c,
    });

    // ---- Comparison / equality opcodes (new) ----

    let cmp_a = rng.random::<u128>();
    let cmp_b = rng.random::<u128>();
    cases.push(BenchCase {
      name: format!("LT#{:02}", sample_idx + 1),
      group: "LT",
      opcode: opcode::LT,
      inputs: vec![u256_bytes(cmp_a), u256_bytes(cmp_b)],
      output: bool_word(cmp_u128(cmp_a, cmp_b)),
    });
    cases.push(BenchCase {
      name: format!("GT#{:02}", sample_idx + 1),
      group: "GT",
      opcode: opcode::GT,
      inputs: vec![u256_bytes(cmp_a), u256_bytes(cmp_b)],
      output: bool_word(cmp_u128(cmp_b, cmp_a)),
    });

    let (slt_a, slt_b) = random_sdiv_pair(&mut rng);
    cases.push(BenchCase {
      name: format!("SLT#{:02}", sample_idx + 1),
      group: "SLT",
      opcode: opcode::SLT,
      inputs: vec![i256_bytes(slt_a), i256_bytes(slt_b)],
      output: bool_word(cmp_i128_lt(slt_a, slt_b)),
    });
    cases.push(BenchCase {
      name: format!("SGT#{:02}", sample_idx + 1),
      group: "SGT",
      opcode: opcode::SGT,
      inputs: vec![i256_bytes(slt_a), i256_bytes(slt_b)],
      output: bool_word(cmp_i128_lt(slt_b, slt_a)),
    });

    let eq_same: bool = rng.random::<bool>();
    let eq_a = u256_bytes(rng.random::<u128>());
    let eq_b = if eq_same { eq_a } else { u256_bytes(rng.random::<u128>()) };
    cases.push(BenchCase {
      name: format!("EQ#{:02}", sample_idx + 1),
      group: "EQ",
      opcode: opcode::EQ,
      inputs: vec![eq_a, eq_b],
      output: bool_word(eq_a == eq_b),
    });

    let iszero_val: u128 = if rng.random::<bool>() { 0 } else { rng.random::<u128>() };
    cases.push(BenchCase {
      name: format!("ISZERO#{:02}", sample_idx + 1),
      group: "ISZERO",
      opcode: opcode::ISZERO,
      inputs: vec![u256_bytes(iszero_val)],
      output: bool_word(iszero_val == 0),
    });
  }

  let mut int_min = [0u8; 32];
  int_min[0] = 0x80;
  cases.push(BenchCase {
    name: "SDIV_INT_MIN".to_string(),
    group: "SDIV_INT_MIN",
    opcode: opcode::SDIV,
    inputs: vec![int_min, [0xFF; 32]],
    output: int_min,
  });

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

fn warm_up(
  itp: &InstructionTransitionProof,
  workers: usize,
) -> Result<(), String> {
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

fn supports_zkp_receipt(op: u8) -> bool {
  matches!(
    op,
    opcode::ADD
      | opcode::SUB
      | opcode::MUL
      | opcode::DIV
      | opcode::MOD
      | opcode::SDIV
      | opcode::SMOD
      | opcode::AND
      | opcode::OR
      | opcode::XOR
      | opcode::NOT
  )
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

// ---- batch STARK helpers ----

#[derive(Clone, Copy, PartialEq, Eq)]
enum OpcodeFamily {
  Add,
  Mul,
  Bit,
}

#[allow(dead_code)]
fn opcode_family(op: u8) -> Option<OpcodeFamily> {
  match op {
    opcode::ADD | opcode::SUB => Some(OpcodeFamily::Add),
    opcode::MUL | opcode::DIV | opcode::MOD | opcode::SDIV | opcode::SMOD => {
      Some(OpcodeFamily::Mul)
    }
    opcode::AND | opcode::OR | opcode::XOR | opcode::NOT => Some(OpcodeFamily::Bit),
    _ => None,
  }
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

/// Prove `batch_size` identical copies of `itp` as a single batched LUT STARK.
/// Returns (prove_µs_per_instr, verify_µs_per_instr, total_rows, total_lut_steps).
fn bench_batch_stark(
  itp: &InstructionTransitionProof,
  family: OpcodeFamily,
  batch_size: usize,
) -> Option<(f64, f64, usize, usize)> {
  let batch: Vec<_> = std::iter::repeat(itp.clone()).take(batch_size).collect();
  let all_rows = batch_rows_for_itps(&batch);
  let total_rows = all_rows.len();

  let lut_len = match family {
    OpcodeFamily::Add => build_lut_steps_from_rows_add_family(&all_rows).ok()?.len(),
    OpcodeFamily::Mul => build_lut_steps_from_rows_mul_family(&all_rows).ok()?.len(),
    OpcodeFamily::Bit => build_lut_steps_from_rows_bit_family(&all_rows).ok()?.len(),
  };

  let start = Instant::now();
  let proof = match family {
    OpcodeFamily::Add => prove_batch_stark_add_family(black_box(&all_rows)).ok()?,
    OpcodeFamily::Mul => prove_batch_stark_mul_family(black_box(&all_rows)).ok()?,
    OpcodeFamily::Bit => prove_batch_stark_bit_family(black_box(&all_rows)).ok()?,
  };
  black_box(&proof);
  let prove_us = start.elapsed().as_secs_f64() * 1_000_000.0 / batch_size as f64;

  let start = Instant::now();
  let ok = match family {
    OpcodeFamily::Add => verify_batch_stark_add_family(black_box(&proof)),
    OpcodeFamily::Mul => verify_batch_stark_mul_family(black_box(&proof)),
    OpcodeFamily::Bit => verify_batch_stark_bit_family(black_box(&proof)),
  };
  black_box(ok);
  let verify_us = start.elapsed().as_secs_f64() * 1_000_000.0 / batch_size as f64;

  Some((prove_us, verify_us, total_rows, lut_len))
}

#[derive(Clone, Copy)]
struct FriPreset {
  label: &'static str,
  num_queries: usize,
  query_pow_bits: usize,
  log_final_poly_len: usize,
}

/// Prove a batch with custom FRI params, return (prove_µs/instr, verify_µs/instr).
fn bench_batch_stark_fri(
  itp: &InstructionTransitionProof,
  family: OpcodeFamily,
  batch_size: usize,
  fri: FriPreset,
) -> Option<(f64, f64)> {
  let batch: Vec<_> = std::iter::repeat(itp.clone()).take(batch_size).collect();
  let all_rows = batch_rows_for_itps(&batch);

  let (nq, pw, lf) = (fri.num_queries, fri.query_pow_bits, fri.log_final_poly_len);

  let start = Instant::now();
  let proof = match family {
    OpcodeFamily::Add => prove_batch_stark_add_family_with_params(black_box(&all_rows), nq, pw, lf),
    OpcodeFamily::Mul => prove_batch_stark_mul_family_with_params(black_box(&all_rows), nq, pw, lf),
    OpcodeFamily::Bit => prove_batch_stark_bit_family_with_params(black_box(&all_rows), nq, pw, lf),
  }.ok()?;
  black_box(&proof);
  let prove_us = start.elapsed().as_secs_f64() * 1_000_000.0 / batch_size as f64;

  let start = Instant::now();
  let ok = match family {
    OpcodeFamily::Add => verify_batch_stark_add_family_with_params(black_box(&proof), nq, pw, lf),
    OpcodeFamily::Mul => verify_batch_stark_mul_family_with_params(black_box(&proof), nq, pw, lf),
    OpcodeFamily::Bit => verify_batch_stark_bit_family_with_params(black_box(&proof), nq, pw, lf),
  };
  black_box(ok);
  let verify_us = start.elapsed().as_secs_f64() * 1_000_000.0 / batch_size as f64;

  Some((prove_us, verify_us))
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
    if zkp_workers == 0 { "auto".to_string() } else { zkp_workers.to_string() }
  );
  println!("  stark bench   : {}", if no_stark { "disabled" } else { "enabled" });

  // ---- warmup ----
  let warmup_seed = seed ^ 0xA5A5_5A5A_D3C3_B1B1;
  let warmup_case = bench_cases(1, warmup_seed)
    .into_iter()
    .find(|case| supports_zkp_receipt(case.opcode))
    .expect("benchmark case generator should include supported opcodes");
  let warmup_itp = build_itp(&warmup_case);

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

    let proof_rows = itp.semantic_proof.as_ref().map(|p| compile_proof(p).len()).unwrap_or(0);
    let wff_prove_us = bench_wff_prove(&case, wff_iters);
    let wff_verify_us = bench_wff_verify(&itp, wff_iters);

    let agg = aggs.entry(case.group).or_default();
    agg.count += 1;
    agg.proof_rows_sum += proof_rows;
    agg.wff_prove_sum += wff_prove_us;
    agg.wff_verify_sum += wff_verify_us;

    if no_stark || !supports_zkp_receipt(case.opcode) {
      continue;
    }

    let lut_steps = itp.semantic_proof.as_ref().map(|p| {
      let rows = compile_proof(p);
      let maybe = match case.opcode {
        opcode::ADD | opcode::SUB => build_lut_steps_from_rows_add_family(&rows),
        opcode::MUL | opcode::DIV | opcode::MOD | opcode::SDIV | opcode::SMOD => {
          build_lut_steps_from_rows_mul_family(&rows)
        }
        opcode::AND | opcode::OR | opcode::XOR | opcode::NOT => {
          build_lut_steps_from_rows_bit_family(&rows)
        }
        _ => return 0,
      };
      maybe.map(|s| s.len()).unwrap_or(0)
    }).unwrap_or(0);

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
      println!();
      println!("── STARK layer (receipt-capable opcodes) ─────────────────────────────────────────────");
      println!(
        "{:<10} {:>7}  {:>10}  {:>16}  {:>16}",
        "opcode", "samples", "lut_steps", "zkp_prove(µs)", "zkp_verify(µs)"
      );
      println!("{}", "─".repeat(66));
      for (group, agg) in stark_aggs {
        let n = agg.stark_count as f64;
        println!(
          "{:<10} {:>7}  {:>10.1}  {:>16.2}  {:>16.2}",
          group,
          agg.stark_count,
          agg.lut_steps_sum as f64 / n,
          agg.zkp_prove_sum / n,
          agg.zkp_verify_sum / n,
        );
      }
    }
  }

  // ---- Batch STARK layer table ----
  // Use one representative case per family (first sample of the cheapest opcode).
  if !no_stark {
    // Pick one representative ITP per family from the generated cases.
    let all_cases = bench_cases(samples.max(1), seed);
    let rep_add = all_cases.iter().find(|c| c.opcode == opcode::ADD).map(build_itp);
    let rep_mul = all_cases.iter().find(|c| c.opcode == opcode::MUL).map(build_itp);
    let rep_bit = all_cases.iter().find(|c| c.opcode == opcode::AND).map(build_itp);

    let batch_sizes = [1usize, 2, 4, 8, 16];

    println!();
    println!("── Batch STARK layer (amortized µs per instruction) ─────────────────────────────────────");
    println!(
      "{:<12}  {:>5}  {:>12}  {:>12}  {:>12}  {:>12}",
      "family", "batch", "total_rows", "prove/instr", "verify/instr", "×WFF_verify"
    );
    println!("{}", "─".repeat(74));

    struct FamilyRep {
      name: &'static str,
      family: OpcodeFamily,
      itp: InstructionTransitionProof,
      wff_verify_us: f64,
    }

    let mut reps: Vec<FamilyRep> = Vec::new();

    if let Some(itp) = rep_add {
      let wff_verify_us = bench_wff_verify(&itp, wff_iters);
      reps.push(FamilyRep { name: "ADD (add-fam)", family: OpcodeFamily::Add, itp, wff_verify_us });
    }
    if let Some(itp) = rep_mul {
      let wff_verify_us = bench_wff_verify(&itp, wff_iters);
      reps.push(FamilyRep { name: "MUL (mul-fam)", family: OpcodeFamily::Mul, itp, wff_verify_us });
    }
    if let Some(itp) = rep_bit {
      let wff_verify_us = bench_wff_verify(&itp, wff_iters);
      reps.push(FamilyRep { name: "AND (bit-fam)", family: OpcodeFamily::Bit, itp, wff_verify_us });
    }

    for rep in &reps {
      for &bs in &batch_sizes {
        match bench_batch_stark(&rep.itp, rep.family, bs) {
          Some((prove_us, verify_us, total_rows, _lut_steps)) => {
            let ratio = if rep.wff_verify_us > 0.0 { verify_us / rep.wff_verify_us } else { 0.0 };
            println!(
              "{:<12}  {:>5}  {:>12}  {:>12.2}  {:>12.2}  {:>12.1}×",
              rep.name, bs, total_rows, prove_us, verify_us, ratio,
            );
          }
          None => {
            println!("{:<12}  {:>5}  — (failed)", rep.name, bs);
          }
        }
      }
    }

    // ---- FRI param sweep table ----
    // Fixed batch=8, vary FRI params to show cost of each knob.
    let fri_batch_size = 8usize;
    let fri_presets = [
      FriPreset { label: "default (40q,pow8,f0)", num_queries: 40, query_pow_bits: 8, log_final_poly_len: 0 },
      FriPreset { label: "no pow  (40q,pow0,f0)", num_queries: 40, query_pow_bits: 0, log_final_poly_len: 0 },
      FriPreset { label: "20q     (20q,pow0,f0)", num_queries: 20, query_pow_bits: 0, log_final_poly_len: 0 },
      FriPreset { label: "10q     (10q,pow0,f0)", num_queries: 10, query_pow_bits: 0, log_final_poly_len: 0 },
      FriPreset { label: "10q+f2  (10q,pow0,f2)", num_queries: 10, query_pow_bits: 0, log_final_poly_len: 2 },
    ];

    println!();
    println!("── FRI param sweep  (batch={fri_batch_size}, single run each) ─────────────────────────────");
    println!(
      "{:<25}  {:>12}  {:>12}  {:>12}  {:>12}",
      "preset", "ADD prove", "ADD verify", "AND prove", "AND verify"
    );
    println!("{}", "─".repeat(80));

    for preset in &fri_presets {
      let add_res = reps.iter()
        .find(|r| matches!(r.family, OpcodeFamily::Add))
        .and_then(|r| bench_batch_stark_fri(&r.itp, r.family, fri_batch_size, *preset));
      let bit_res = reps.iter()
        .find(|r| matches!(r.family, OpcodeFamily::Bit))
        .and_then(|r| bench_batch_stark_fri(&r.itp, r.family, fri_batch_size, *preset));

      let fmt = |v: Option<f64>| match v {
        Some(x) => format!("{x:>10.2}µs"),
        None => "     —    ".to_string(),
      };

      println!(
        "{:<25}  {}  {}  {}  {}",
        preset.label,
        fmt(add_res.map(|(p, _)| p)),
        fmt(add_res.map(|(_, v)| v)),
        fmt(bit_res.map(|(p, _)| p)),
        fmt(bit_res.map(|(_, v)| v)),
      );
    }

    // ── Cross-family batch WFF prove/verify ────────────────────────────────
    // Mix ADD + MUL + AND in a single LUT STARK call. Proofs come from the
    // representative instruction of each family (the one used for the batch
    // table above). Batch size = 1 rep of each present family.
    //
    // Each row of the table combines N instructions (one per family) into one
    // prove call and one verify call, then shows the amortized cost per instruction.

    // Collect (name, itp) pairs for cross-family mix
    let cross_cases: Vec<(&str, &InstructionTransitionProof)> = reps
      .iter()
      .map(|r| (r.name, &r.itp))
      .collect();

    if cross_cases.len() >= 2 {
      println!();
      println!("── Cross-family WFF batch (mixed opcodes → single LUT STARK) ────────────────────────────");
      println!(
        "{:<30}  {:>5}  {:>14}  {:>14}  {:>14}",
        "mix", "n", "prove_total µs", "verify_total µs", "verify/instr µs"
      );
      println!("{}", "─".repeat(80));

      // Collect semantic proofs for each mix size
      let sem_proofs: Vec<_> = cross_cases
        .iter()
        .filter_map(|(_, itp)| itp.semantic_proof.as_ref())
        .collect();

      for n in 2..=cross_cases.len() {
        let batch: Vec<_> = sem_proofs.iter().take(n).copied().collect();
        let mix_name: String = cross_cases.iter().take(n).map(|(nm, _)| *nm).collect::<Vec<_>>().join(" + ");

        // Prove (single run for timing)
        let prove_start = std::time::Instant::now();
        let result = prove_batch_wff_proofs(&batch);
        let prove_us = prove_start.elapsed().as_secs_f64() * 1e6;

        match result {
          Ok((ref lut_proof, ref wffs)) => {
            // Verify
            let verify_start = std::time::Instant::now();
            let _ = verify_batch_wff_proofs(lut_proof, &batch, wffs);
            let verify_us = verify_start.elapsed().as_secs_f64() * 1e6;

            println!(
              "{:<30}  {:>5}  {:>14.2}  {:>14.2}  {:>14.2}",
              mix_name, n, prove_us, verify_us, verify_us / n as f64
            );
          }
          Err(e) => {
            println!("{:<30}  {:>5}  prove failed: {e}", mix_name, n);
          }
        }
      }
    }
  }
}
