use std::hint::black_box;
use std::time::Instant;
use std::collections::BTreeMap;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use revm::bytecode::opcode;
use zprove_core::sementic_proof::compile_proof;
use zprove_core::transition::{
  prove_instruction,
  prove_instruction_zk_receipt,
  verify_instruction_zk_receipt,
  InstructionTransitionZkReceipt,
  InstructionTransitionStatement,
  InstructionTransitionProof,
  VmState,
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

fn bench_zkp_prove(itp: &InstructionTransitionProof, iters: usize) -> f64 {
  let start = Instant::now();
  for _ in 0..iters {
    let receipt = prove_instruction_zk_receipt(black_box(itp))
      .expect("zkp receipt generation should succeed during benchmark");
    black_box(receipt);
  }
  start.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64
}

fn bench_zkp_verify(
  statement: &InstructionTransitionStatement,
  receipt: &InstructionTransitionZkReceipt,
  iters: usize,
) -> f64 {
  let start = Instant::now();
  for _ in 0..iters {
    black_box(verify_instruction_zk_receipt(black_box(statement), black_box(receipt)));
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

fn main() {
  let args: Vec<String> = std::env::args().collect();
  let zkp_prove_iters = parse_arg_usize(&args, "--zkp-prove-iters", 1);
  let zkp_verify_iters = parse_arg_usize(&args, "--zkp-verify-iters", 1);
  let samples = parse_arg_usize(&args, "--samples", 10);
  let seed_opt = parse_arg_opt_u64(&args, "--seed");
  let seed = seed_opt.unwrap_or_else(rand::random::<u64>);

  println!("zprove benchmark bin");
  match seed_opt {
    Some(s) => println!("- seed: {s} (fixed)"),
    None => println!("- seed: {seed} (random)"),
  }
  println!("- random samples per opcode: {samples}");
  println!("- zkp prove iterations: {zkp_prove_iters}");
  println!("- zkp verify iterations: {zkp_verify_iters}");
  println!();
  println!("{:<18} {:>8} {:>10} {:>16} {:>16}", "opcode", "samples", "proof_rows", "zkp_prove(µs)", "zkp_verify(µs)");
  println!("{}", "-".repeat(78));

  #[derive(Default)]
  struct Agg {
    count: usize,
    proof_rows_sum: usize,
    zkp_prove_sum: f64,
    zkp_verify_sum: f64,
  }

  let mut aggs: BTreeMap<&'static str, Agg> = BTreeMap::new();
  let mut skipped: BTreeMap<&'static str, usize> = BTreeMap::new();

  for case in bench_cases(samples, seed) {
    if !supports_zkp_receipt(case.opcode) {
      *skipped.entry(case.group).or_default() += 1;
      continue;
    }

    let itp = build_itp(&case);
    let statement = build_statement(&case);
    let proof_rows = itp
      .semantic_proof
      .as_ref()
      .map(|p| compile_proof(p).len())
      .unwrap_or(0);
    let receipt = match prove_instruction_zk_receipt(&itp) {
      Ok(receipt) => receipt,
      Err(_) => {
        *skipped.entry(case.group).or_default() += 1;
        continue;
      }
    };

    let zkp_prove_us = bench_zkp_prove(&itp, zkp_prove_iters);
    let zkp_verify_us = bench_zkp_verify(&statement, &receipt, zkp_verify_iters);

    let agg = aggs.entry(case.group).or_default();
    agg.count += 1;
    agg.proof_rows_sum += proof_rows;
    agg.zkp_prove_sum += zkp_prove_us;
    agg.zkp_verify_sum += zkp_verify_us;
    black_box(case.name);
  }

  for (group, agg) in aggs {
    let n = agg.count as f64;
    println!(
      "{:<18} {:>8} {:>10.1} {:>16.2} {:>16.2}",
      group,
      agg.count,
      agg.proof_rows_sum as f64 / n,
      agg.zkp_prove_sum / n,
      agg.zkp_verify_sum / n,
    );
  }

  if !skipped.is_empty() {
    println!();
    println!("skipped unsupported opcode groups:");
    for (group, count) in skipped {
      println!("- {group}: {count}");
    }
  }
}
