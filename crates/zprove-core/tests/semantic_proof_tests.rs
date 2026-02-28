// moved from src/semantic_proof.rs

#[cfg(test)]
mod tests {
  use std::array;
  use zprove_core::semantic_proof::*;

  fn term_node_count(term: &Term) -> usize {
    match term {
      Term::Bool(_) | Term::Byte(_) => 1,
      Term::Not(a) => 1 + term_node_count(a),
      Term::And(a, b)
      | Term::Or(a, b)
      | Term::Xor(a, b)
      | Term::ByteMulLow(a, b)
      | Term::ByteMulHigh(a, b)
      | Term::ByteAnd(a, b)
      | Term::ByteOr(a, b)
      | Term::ByteXor(a, b) => 1 + term_node_count(a) + term_node_count(b),
      Term::Ite(c, a, b) | Term::ByteAdd(a, b, c) | Term::ByteAddCarry(a, b, c) => {
        1 + term_node_count(c) + term_node_count(a) + term_node_count(b)
      }
      Term::InputTerm { .. } | Term::OutputTerm { .. } | Term::PcBefore { .. } | Term::PcAfter { .. } => 1,
    }
  }

  fn word_term_sizes(terms: &[Box<Term>; 32]) -> [usize; 32] {
    array::from_fn(|i| term_node_count(terms[i].as_ref()))
  }

  #[test]
  fn test_compile_and_verify_simple_add() {
    let mut a = [0u8; 32];
    let mut b = [0u8; 32];
    a[31] = 100;
    b[31] = 50;
    let mut c = [0u8; 32];
    c[31] = 150;

    let proof = prove_add(&a, &b, &c);
    let rows = compile_proof(&proof);
    verify_compiled(&rows).expect("compiled verification should pass");
  }

  #[test]
  fn test_compile_and_verify_add_with_carry() {
    let mut a = [0u8; 32];
    let mut b = [0u8; 32];
    a[31] = 200;
    b[31] = 100;
    let mut c = [0u8; 32];
    c[31] = 44;
    c[30] = 1;

    let proof = prove_add(&a, &b, &c);
    let rows = compile_proof(&proof);
    verify_compiled(&rows).expect("compiled verification should pass");
  }

  #[test]
  fn test_compile_row_count() {
    let a = [0u8; 32];
    let b = [0u8; 32];
    let c = [0u8; 32];

    let proof = prove_add(&a, &b, &c);
    let rows = compile_proof(&proof);
    assert!(!rows.is_empty(), "compiled proof must contain rows");
    eprintln!("zero-add row count: {}", rows.len());
  }

  #[test]
  fn test_compiled_corrupted_value_fails() {
    let mut a = [0u8; 32];
    let mut b = [0u8; 32];
    a[31] = 10;
    b[31] = 20;
    let mut c = [0u8; 32];
    c[31] = 30;

    let proof = prove_add(&a, &b, &c);
    let mut rows = compile_proof(&proof);

    // Corrupt the first ADD-family axiom scalar/range so compiled verification fails.
    let mut corrupted = false;
    for r in rows.iter_mut() {
      if r.op == OP_U29_ADD_EQ {
        r.scalar0 = 1 << 29; // out of range for u29
        corrupted = true;
        break;
      }
      if r.op == OP_U24_ADD_EQ {
        r.scalar0 = 1 << 24; // out of range for u24
        corrupted = true;
        break;
      }
    }
    assert!(corrupted, "expected at least one add-family row to corrupt");
    assert!(
      verify_compiled(&rows).is_err(),
      "corrupted value should fail"
    );
  }

  #[test]
  fn test_compile_term_standalone() {
    let term = Term::ByteAdd(
      Box::new(Term::Byte(200)),
      Box::new(Term::Byte(100)),
      Box::new(Term::Bool(false)),
    );
    let rows = compile_term(&term);
    assert_eq!(rows.last().unwrap().value, 44); // (200+100+0) & 0xFF
    assert_eq!(rows.last().unwrap().ret_ty, RET_BYTE);
    verify_compiled(&rows).unwrap();
  }

  #[test]
  fn test_compile_bool_ops() {
    let term = Term::Xor(Box::new(Term::Bool(true)), Box::new(Term::Bool(false)));
    let rows = compile_term(&term);
    assert_eq!(rows.last().unwrap().value, 1);
    assert_eq!(rows.last().unwrap().ret_ty, RET_BOOL);
    verify_compiled(&rows).unwrap();
  }

  #[test]
  fn test_all_bytes_max() {
    let a = [0xFF; 32];
    let mut b = [0u8; 32];
    b[31] = 1;
    // 0xFF..FF + 1 = 0x00..00 (mod 2^256)
    let mut c = [0u8; 32];
    c[0] = 0; // MSB wraps
    // compute expected
    let mut carry = 0u16;
    for i in (0..32).rev() {
      let sum = a[i] as u16 + b[i] as u16 + carry;
      c[i] = (sum & 0xFF) as u8;
      carry = sum >> 8;
    }

    let proof = prove_add(&a, &b, &c);
    let rows = compile_proof(&proof);
    verify_compiled(&rows).unwrap();
  }

  #[test]
  fn test_compile_and_verify_ite_axioms() {
    let p_true = Proof::IteTrueEq(Term::Byte(7), Term::Byte(99));
    let w_true = infer_proof(&p_true).expect("infer_proof ite true should succeed");
    match w_true {
      WFF::Equal(lhs, rhs) => {
        assert_eq!(*rhs, Term::Byte(7));
        match *lhs {
          Term::Ite(c, _, _) => assert_eq!(*c, Term::Bool(true)),
          _ => panic!("lhs should be ite"),
        }
      }
      _ => panic!("expected equality"),
    }
    let rows_true = compile_proof(&p_true);
    verify_compiled(&rows_true).expect("compiled verification should pass for ite true axiom");

    let p_false = Proof::IteFalseEq(Term::Byte(7), Term::Byte(99));
    let w_false = infer_proof(&p_false).expect("infer_proof ite false should succeed");
    match w_false {
      WFF::Equal(lhs, rhs) => {
        assert_eq!(*rhs, Term::Byte(99));
        match *lhs {
          Term::Ite(c, _, _) => assert_eq!(*c, Term::Bool(false)),
          _ => panic!("lhs should be ite"),
        }
      }
      _ => panic!("expected equality"),
    }
    let rows_false = compile_proof(&p_false);
    verify_compiled(&rows_false).expect("compiled verification should pass for ite false axiom");
  }

  #[test]
  fn test_word_mul_hybrid_witness_verifies() {
    let mut a = [0u8; 32];
    let mut b = [0u8; 32];
    a[31] = 123;
    b[31] = 45;

    let (_terms, witness) = word_mul_with_hybrid_witness(&a, &b);
    assert!(verify_word_mul_hybrid_witness(&a, &b, &witness));
  }

  #[test]
  fn test_word_mul_hybrid_witness_tamper_detected() {
    let mut a = [0u8; 32];
    let mut b = [0u8; 32];
    a[31] = 200;
    b[31] = 100;

    let (_terms, mut witness) = word_mul_with_hybrid_witness(&a, &b);
    if let Some(first) = witness.add_steps.first_mut() {
      first.sum ^= 1;
    }
    assert!(!verify_word_mul_hybrid_witness(&a, &b, &witness));
  }

  #[test]
  fn test_mul_u15_leaf_multiplicity_not_deduped() {
    let a = [0xFF; 32];
    let b = [0xFF; 32];
    let c = {
      let mut out = [0u8; 32];
      let mut carry = 0u16;
      for i in (0..32).rev() {
        let total = a[i] as u16 * b[i] as u16 + carry;
        out[i] = (total & 0xFF) as u8;
        carry = total >> 8;
      }
      out
    };

    let proof = prove_mul(&a, &b, &c);
    let rows = compile_proof(&proof);
    let u15_rows = rows.iter().filter(|r| r.op == OP_U15_MUL_EQ).count();

    assert!(u15_rows > 1, "U15 MUL leaves should preserve multiplicity");
  }

  #[test]
  fn test_word_add_hybrid_witness_verifies() {
    let mut a = [0u8; 32];
    let mut b = [0u8; 32];
    a[31] = 200;
    b[31] = 100;

    let (_terms, witness) = word_add_with_hybrid_witness(&a, &b);
    assert!(verify_word_add_hybrid_witness(&a, &b, &witness));
  }

  #[test]
  fn test_word_add_hybrid_witness_tamper_detected() {
    let mut a = [0u8; 32];
    let mut b = [0u8; 32];
    a[31] = 77;
    b[31] = 99;

    let (_terms, mut witness) = word_add_with_hybrid_witness(&a, &b);
    if let Some(first) = witness.add_steps.first_mut() {
      first.carry_out = !first.carry_out;
    }
    assert!(!verify_word_add_hybrid_witness(&a, &b, &witness));
  }

  #[test]
  #[ignore = "diagnostic size measurement; run explicitly with -- --ignored --nocapture"]
  fn test_measure_word_mul_term_size() {
    let mut sparse_a = [0u8; 32];
    let mut sparse_b = [0u8; 32];
    sparse_a[31] = 100;
    sparse_b[31] = 200;

    let mut dense_a = [0u8; 32];
    let mut dense_b = [0u8; 32];
    for i in 30..32 {
      dense_a[i] = 0xFF;
      dense_b[i] = 0xFF;
    }

    let sparse_terms = word_mul(&sparse_a, &sparse_b);
    let sparse_sizes = word_term_sizes(&sparse_terms);
    let sparse_total: usize = sparse_sizes.iter().sum();

    let dense_terms = word_mul(&dense_a, &dense_b);
    let dense_sizes = word_term_sizes(&dense_terms);
    let dense_total: usize = dense_sizes.iter().sum();

    eprintln!("word_mul term-size sparse total: {}", sparse_total);
    eprintln!("word_mul term-size sparse per-byte: {:?}", sparse_sizes);
    eprintln!("word_mul term-size dense total: {}", dense_total);
    eprintln!("word_mul term-size dense per-byte: {:?}", dense_sizes);

    assert!(sparse_total > 0);
    assert!(dense_total >= sparse_total);
  }
}
