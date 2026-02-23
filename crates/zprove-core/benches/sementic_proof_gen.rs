//! Benchmarks for semantic proof generation and verification per EVM instruction.
//!
//! Measures:
//!   - Proof generation time
//!   - Proof verification time
//!   - Proof tree size (node count)
//!
//! Run with: `cargo bench --bench sementic_proof_gen`

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use zprove_core::sementic_proof::{Proof, Term, WFF, infer_proof, prove_add, wff_add};
use zprove_core::transition::{opcode, verify_proof, InstructionTransitionProof};
use serde_json;
use bincode;

// ============================================================
// Helpers
// ============================================================

fn u256_bytes(val: u128) -> [u8; 32] {
  let mut b = [0u8; 32];
  b[16..32].copy_from_slice(&val.to_be_bytes());
  b
}

/// Count the number of nodes in a Proof tree.
fn proof_node_count(proof: &Proof) -> usize {
  match proof {
    Proof::AndIntro(a, b) => 1 + proof_node_count(a) + proof_node_count(b),
    Proof::EqRefl(_) => 1,
    Proof::EqSym(p) => 1 + proof_node_count(p),
    Proof::EqTrans(a, b) => 1 + proof_node_count(a) + proof_node_count(b),
    Proof::ByteAddEq(_, _, _) | Proof::ByteAddCarryEq(_, _, _) => 1,
    Proof::ByteMulLowEq(_, _) | Proof::ByteMulHighEq(_, _) => 1,
    Proof::ByteAndEq(_, _) | Proof::ByteOrEq(_, _) | Proof::ByteXorEq(_, _) => 1,
  }
}

/// Count the number of nodes in a Term tree.
fn term_node_count(term: &Term) -> usize {
  match term {
    Term::Bool(_) | Term::Byte(_) => 1,
    Term::Not(a) => 1 + term_node_count(a),
    Term::And(a, b) | Term::Or(a, b) | Term::Xor(a, b) => {
      1 + term_node_count(a) + term_node_count(b)
    }
    Term::Ite(c, a, b) => 1 + term_node_count(c) + term_node_count(a) + term_node_count(b),
    Term::ByteAdd(a, b, c) | Term::ByteAddCarry(a, b, c) => {
      1 + term_node_count(a) + term_node_count(b) + term_node_count(c)
    }
    Term::ByteMulLow(a, b)
    | Term::ByteMulHigh(a, b)
    | Term::ByteAnd(a, b)
    | Term::ByteOr(a, b)
    | Term::ByteXor(a, b) => 1 + term_node_count(a) + term_node_count(b),
  }
}

/// Count the number of nodes in a WFF tree.
fn wff_node_count(wff: &WFF) -> usize {
  match wff {
    WFF::Equal(a, b) => 1 + term_node_count(a) + term_node_count(b),
    WFF::And(a, b) => 1 + wff_node_count(a) + wff_node_count(b),
  }
}

/// Estimate memory size of a Proof tree in bytes (rough approximation).
fn proof_mem_size(proof: &Proof) -> usize {
  let self_size = std::mem::size_of::<Proof>();
  match proof {
    Proof::AndIntro(a, b) => self_size + proof_mem_size(a) + proof_mem_size(b),
    Proof::EqRefl(t) => self_size + term_mem_size(t),
    Proof::EqSym(p) => self_size + proof_mem_size(p),
    Proof::EqTrans(a, b) => self_size + proof_mem_size(a) + proof_mem_size(b),
    Proof::ByteAddEq(_, _, _) | Proof::ByteAddCarryEq(_, _, _)
    | Proof::ByteMulLowEq(_, _) | Proof::ByteMulHighEq(_, _)
    | Proof::ByteAndEq(_, _) | Proof::ByteOrEq(_, _) | Proof::ByteXorEq(_, _) => self_size,
  }
}

/// Estimate memory size of a Term tree in bytes.
fn term_mem_size(term: &Term) -> usize {
  let self_size = std::mem::size_of::<Term>();
  match term {
    Term::Bool(_) | Term::Byte(_) => self_size,
    Term::Not(a) => self_size + term_mem_size(a),
    Term::And(a, b) | Term::Or(a, b) | Term::Xor(a, b) => {
      self_size + term_mem_size(a) + term_mem_size(b)
    }
    Term::Ite(c, a, b) => {
      self_size + term_mem_size(c) + term_mem_size(a) + term_mem_size(b)
    }
    Term::ByteAdd(a, b, c) | Term::ByteAddCarry(a, b, c) => {
      self_size + term_mem_size(a) + term_mem_size(b) + term_mem_size(c)
    }
    Term::ByteMulLow(a, b)
    | Term::ByteMulHigh(a, b)
    | Term::ByteAnd(a, b)
    | Term::ByteOr(a, b)
    | Term::ByteXor(a, b) => self_size + term_mem_size(a) + term_mem_size(b),
  }
}

// ============================================================
// Test vectors
// ============================================================

struct TestCase {
  name: &'static str,
  a: [u8; 32],
  b: [u8; 32],
}

fn test_cases() -> Vec<TestCase> {
  vec![
    TestCase {
      name: "small",
      a: u256_bytes(1000),
      b: u256_bytes(2000),
    },
    TestCase {
      name: "large",
      a: {
        let mut v = [0xABu8; 32];
        v[0] = 0x7F;
        v
      },
      b: {
        let mut v = [0xCDu8; 32];
        v[0] = 0x3E;
        v
      },
    },
    TestCase {
      name: "overflow",
      a: [0xFF; 32],
      b: {
        let mut v = [0u8; 32];
        v[31] = 1;
        v
      },
    },
    TestCase {
      name: "zero",
      a: [0u8; 32],
      b: [0u8; 32],
    },
  ]
}

fn compute_add(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
  let mut result = [0u8; 32];
  let mut carry = 0u16;
  for i in (0..32).rev() {
    let sum = a[i] as u16 + b[i] as u16 + carry;
    result[i] = (sum & 0xFF) as u8;
    carry = sum >> 8;
  }
  result
}

// ============================================================
// Benchmarks
// ============================================================

fn bench_add_prove(c: &mut Criterion) {
  let mut group = c.benchmark_group("ADD/prove");

  for tc in test_cases() {
    let output = compute_add(&tc.a, &tc.b);
    group.bench_function(tc.name, |bencher| {
      bencher.iter(|| {
        black_box(prove_add(
          black_box(&tc.a),
          black_box(&tc.b),
          black_box(&output),
        ))
      })
    });
  }

  group.finish();
}

fn bench_add_verify(c: &mut Criterion) {
  let mut group = c.benchmark_group("ADD/verify");

  for tc in test_cases() {
    let output = compute_add(&tc.a, &tc.b);
    let proof = prove_add(&tc.a, &tc.b, &output).unwrap();
    let itp = InstructionTransitionProof {
      opcode: opcode::ADD,
      pc: 0,
      stack_inputs: vec![tc.a, tc.b],
      stack_outputs: vec![output],
      semantic_proof: Some(proof),
    };

    group.bench_function(tc.name, |bencher| {
      bencher.iter(|| black_box(verify_proof(black_box(&itp))))
    });
  }

  group.finish();
}

fn bench_add_infer(c: &mut Criterion) {
  let mut group = c.benchmark_group("ADD/infer_proof");

  for tc in test_cases() {
    let output = compute_add(&tc.a, &tc.b);
    let proof = prove_add(&tc.a, &tc.b, &output).unwrap();

    group.bench_function(tc.name, |bencher| {
      bencher.iter(|| black_box(infer_proof(black_box(&proof))))
    });
  }

  group.finish();
}

fn bench_add_wff(c: &mut Criterion) {
  let mut group = c.benchmark_group("ADD/wff_gen");

  for tc in test_cases() {
    let output = compute_add(&tc.a, &tc.b);

    group.bench_function(tc.name, |bencher| {
      bencher.iter(|| {
        black_box(wff_add(
          black_box(&tc.a),
          black_box(&tc.b),
          black_box(&output),
        ))
      })
    });
  }

  group.finish();
}

// ============================================================
// Size report (printed once, not benchmarked)
// ============================================================

fn format_size(bytes: usize) -> String {
  if bytes >= 1024 * 1024 {
    format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
  } else if bytes >= 1024 {
    format!("{:.1} KB", bytes as f64 / 1024.0)
  } else {
    format!("{} B", bytes)
  }
}

fn print_proof_sizes() {
  println!("\n============================================================================");
  println!("  Semantic Proof Size Report");
  println!("============================================================================\n");
  println!(
    "{:<12} {:>8} {:>12} {:>12} {:>12} {:>10}",
    "case", "nodes", "json", "bincode", "heap(est)", "verified"
  );
  println!("{}", "-".repeat(70));

  for tc in test_cases() {
    let output = compute_add(&tc.a, &tc.b);
    let proof = prove_add(&tc.a, &tc.b, &output).unwrap();

    let json_bytes = serde_json::to_vec(&proof).unwrap();
    let bincode_bytes = bincode::serialize(&proof).unwrap();
    let heap_est = proof_mem_size(&proof);
    let nodes = proof_node_count(&proof);

    let itp = InstructionTransitionProof {
      opcode: opcode::ADD,
      pc: 0,
      stack_inputs: vec![tc.a, tc.b],
      stack_outputs: vec![output],
      semantic_proof: Some(proof.clone()),
    };
    let verified = verify_proof(&itp);

    println!(
      "{:<12} {:>8} {:>12} {:>12} {:>12} {:>10}",
      tc.name,
      nodes,
      format_size(json_bytes.len()),
      format_size(bincode_bytes.len()),
      format_size(heap_est),
      if verified { "OK" } else { "FAIL" },
    );
  }
  println!();
}

fn bench_sizes(c: &mut Criterion) {
  // Print size report once during benchmark run
  static ONCE: std::sync::Once = std::sync::Once::new();
  ONCE.call_once(print_proof_sizes);

  // Trivial benchmark so criterion doesn't complain about empty group
  c.bench_function("ADD/proof_size_report", |bencher| {
    let a = u256_bytes(1000);
    let b = u256_bytes(2000);
    let output = compute_add(&a, &b);
    bencher.iter(|| {
      let proof = prove_add(&a, &b, &output).unwrap();
      black_box(proof_node_count(&proof))
    })
  });
}

criterion_group!(
  benches,
  bench_add_prove,
  bench_add_verify,
  bench_add_infer,
  bench_add_wff,
  bench_sizes,
);
criterion_main!(benches);
