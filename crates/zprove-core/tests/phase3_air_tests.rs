/// Phase 3 통합 테스트 — 재귀 STARK 서브 회로
///
/// 테스트 대상:
/// - `Poseidon2Air` (poseidon2_air.rs)
/// - `M31Ext3MulAir` (m31ext3_air.rs)
/// - `MerklePathAir` (merkle_air.rs)
/// - `FriQueryAir` (fri_air.rs)
/// - `OodVerifierAir` + `StarkVerifierAir` (stark_verifier_air.rs)
#[cfg(test)]
mod phase3_air_tests {
  use p3_field::BasedVectorSpace;
  use zprove_core::zk_proof::{
    Challenge, MerklePathInput, OodEvaluation, P2_WIDTH, Val, FriQueryInput,
    hash_inner_public_inputs, prove_fri_queries, prove_m31ext3_mul,
    prove_merkle_paths, prove_ood_evaluation, prove_poseidon2_permutations,
    prove_recursively, verify_fri_queries, verify_m31ext3_mul,
    verify_merkle_paths, verify_ood_evaluation, verify_poseidon2_permutations,
    verify_recursively,
  };

  // ── 헬퍼 ───────────────────────────────────────────────────────────────────

  /// `(a0, a1, a2)` 계수로 M31³ 원소를 생성한다.
  fn ch(a0: u32, a1: u32, a2: u32) -> Challenge {
    Challenge::from_basis_coefficients_fn(|d| match d {
      0 => Val::new(a0),
      1 => Val::new(a1),
      2 => Val::new(a2),
      _ => unreachable!(),
    })
  }

  /// Circle FRI fold 공식으로 올바른 `expected_folded`를 계산해 `FriQueryInput`을 만든다.
  ///
  /// fold(f)(β, t_inv) = (y+ + y-)/2 + β * (y+ - y-) * t_inv / 2
  /// y_plus, y_minus는 M31³ (Challenge)
  fn make_fri_input(t_inv: Val, y_plus: Challenge, y_minus: Challenge, beta: Challenge) -> FriQueryInput {
    use p3_field::Field;
    let two_inv: Val = Val::new(2).inverse();

    let sum = y_plus + y_minus;
    let diff = (y_plus - y_minus) * t_inv;
    let expected_folded = (sum + beta * diff) * two_inv;
    FriQueryInput { t_inv, y_plus, y_minus, beta, expected_folded }
  }

  /// `constraint_val * zh_inv = quotient_val` 을 만족하는 OodEvaluation을 만든다.
  fn make_ood(constraint_val: Challenge, zh_inv: Challenge) -> OodEvaluation {
    OodEvaluation {
      zeta: ch(1, 0, 0),
      alpha: ch(2, 0, 0),
      constraint_val,
      zh_inv,
      quotient_val: constraint_val * zh_inv,
    }
  }

  // ── Poseidon2Air ────────────────────────────────────────────────────────────

  #[test]
  fn test_poseidon2_air_roundtrip_single() {
    let input = vec![[Val::new(0); P2_WIDTH]];
    let proof = prove_poseidon2_permutations(input);
    let r = verify_poseidon2_permutations(&proof);
    assert!(r.is_ok(), "Poseidon2 single: {:?}", r.err());
  }

  #[test]
  fn test_poseidon2_air_roundtrip_batch() {
    let inputs: Vec<[Val; P2_WIDTH]> = (0u32..4)
      .map(|i| {
        let mut v = [Val::new(0); P2_WIDTH];
        v[0] = Val::new(i + 1);
        v
      })
      .collect();
    let proof = prove_poseidon2_permutations(inputs);
    let r = verify_poseidon2_permutations(&proof);
    assert!(r.is_ok(), "Poseidon2 batch: {:?}", r.err());
  }

  // ── M31Ext3MulAir ───────────────────────────────────────────────────────────

  #[test]
  fn test_m31ext3_mul_roundtrip_identity() {
    // a * 1 = a (항등원)
    let a = ch(5, 7, 11);
    let one = Challenge::from(Val::new(1));
    let proof = prove_m31ext3_mul(&[(a, one)]);
    let r = verify_m31ext3_mul(&proof);
    assert!(r.is_ok(), "M31Ext3 identity: {:?}", r.err());
  }

  #[test]
  fn test_m31ext3_mul_roundtrip_batch() {
    // (1,1,0) * (1,0,1) 과 (2,0,0) * (3,0,0) 배치
    let pairs = vec![(ch(1, 1, 0), ch(1, 0, 1)), (ch(2, 0, 0), ch(3, 0, 0))];
    let proof = prove_m31ext3_mul(&pairs);
    let r = verify_m31ext3_mul(&proof);
    assert!(r.is_ok(), "M31Ext3 batch: {:?}", r.err());
  }

  // ── MerklePathAir ───────────────────────────────────────────────────────────

  fn vhash(seed: u32) -> [Val; 8] {
    let mut h = [Val::new(0); 8];
    h[0] = Val::new(seed);
    h
  }

  #[test]
  fn test_merkle_path_air_roundtrip_depth1_left() {
    // leaf → left child, sibling → right child
    let input = MerklePathInput {
      leaf_hash: vhash(1),
      siblings: vec![vhash(2)],
      directions: vec![false], // leaf = left child
      root: [Val::new(0); 8], // root는 현재 AIR에서 검증 안 함
    };
    let proof = prove_merkle_paths(&[input]);
    let r = verify_merkle_paths(&proof);
    assert!(r.is_ok(), "Merkle depth1 left: {:?}", r.err());
  }

  #[test]
  fn test_merkle_path_air_roundtrip_depth1_right() {
    // leaf → right child
    let input = MerklePathInput {
      leaf_hash: vhash(5),
      siblings: vec![vhash(3)],
      directions: vec![true], // leaf = right child
      root: [Val::new(0); 8],
    };
    let proof = prove_merkle_paths(&[input]);
    let r = verify_merkle_paths(&proof);
    assert!(r.is_ok(), "Merkle depth1 right: {:?}", r.err());
  }

  #[test]
  fn test_merkle_path_air_roundtrip_depth2() {
    let input = MerklePathInput {
      leaf_hash: vhash(10),
      siblings: vec![vhash(20), vhash(30)],
      directions: vec![false, true],
      root: [Val::new(0); 8],
    };
    let proof = prove_merkle_paths(&[input]);
    let r = verify_merkle_paths(&proof);
    assert!(r.is_ok(), "Merkle depth2: {:?}", r.err());
  }

  // ── FriQueryAir ─────────────────────────────────────────────────────────────

  #[test]
  fn test_fri_query_air_roundtrip_single() {
    // t_inv=1 (twiddle^{-1}), y+=ch(4,0,0), y-=ch(2,0,0), β=(3,0,0)
    let q = make_fri_input(Val::new(1), ch(4, 0, 0), ch(2, 0, 0), ch(3, 0, 0));
    let proof = prove_fri_queries(&[q]);
    let r = verify_fri_queries(&proof);
    assert!(r.is_ok(), "FRI single: {:?}", r.err());
  }

  #[test]
  fn test_fri_query_air_roundtrip_batch() {
    let qs = vec![
      make_fri_input(Val::new(1), ch(4, 0, 0), ch(2, 0, 0), ch(3, 0, 0)),
      make_fri_input(Val::new(2), ch(10, 0, 0), ch(6, 0, 0), ch(1, 2, 3)),
    ];
    let proof = prove_fri_queries(&qs);
    let r = verify_fri_queries(&proof);
    assert!(r.is_ok(), "FRI batch: {:?}", r.err());
  }

  // ── OodVerifierAir ──────────────────────────────────────────────────────────

  #[test]
  fn test_ood_verifier_air_roundtrip_single() {
    // constraint_val=(2,0,0), zh_inv=(3,0,0) → quotient=(6,0,0)
    let ev = make_ood(ch(2, 0, 0), ch(3, 0, 0));
    let proof = prove_ood_evaluation(&[ev]);
    let r = verify_ood_evaluation(&proof);
    assert!(r.is_ok(), "OOD single: {:?}", r.err());
  }

  #[test]
  fn test_ood_verifier_air_roundtrip_batch() {
    let evals = vec![
      make_ood(ch(1, 0, 0), ch(1, 0, 0)),   // 1*1 = 1
      make_ood(ch(2, 1, 0), ch(1, 1, 0)),   // 비자명한 M31³ 곱
      make_ood(ch(0, 1, 0), ch(0, 0, 1)),   // 순수 확장 원소 곱
    ];
    let proof = prove_ood_evaluation(&evals);
    let r = verify_ood_evaluation(&proof);
    assert!(r.is_ok(), "OOD batch: {:?}", r.err());
  }

  // ── StarkVerifierAir (재귀 증명) ─────────────────────────────────────────────

  #[test]
  fn test_recursive_stark_proof_roundtrip() {
    // 내부 증명 = M31Ext3Mul (public inputs 없음)
    let inner_proof = prove_m31ext3_mul(&[(ch(2, 0, 0), ch(3, 0, 0))]);
    let expected_tc: [Val; 8] = inner_proof.commitments.trace.into();

    let recursive = prove_recursively(&inner_proof, &[], false);
    let r = verify_recursively(&recursive, &[], &expected_tc);
    assert!(r.is_ok(), "recursive roundtrip: {:?}", r.err());
  }

  #[test]
  #[should_panic(expected = "inner_pis_hash mismatch")]
  fn test_recursive_stark_pis_hash_mismatch_panics() {
    // 잘못된 expected_inner_pis → assert_eq 패닉이 발생해야 함
    let inner_proof = prove_m31ext3_mul(&[(ch(1, 0, 0), ch(1, 0, 0))]);
    let expected_tc: [Val; 8] = inner_proof.commitments.trace.into();

    let recursive = prove_recursively(&inner_proof, &[], false);
    // 원래 공개 입력은 `&[]` 인데 `&[Val::new(99)]` 로 잘못 전달
    verify_recursively(&recursive, &[Val::new(99)], &expected_tc).unwrap();
  }

  #[test]
  fn test_recursive_stark_proof_hash_determinism() {
    // 동일 입력 → 동일 해시
    let h1 = hash_inner_public_inputs(&[Val::new(1), Val::new(2)]);
    let h2 = hash_inner_public_inputs(&[Val::new(1), Val::new(2)]);
    assert_eq!(h1, h2);

    // 다른 입력 → 다른 해시
    let h3 = hash_inner_public_inputs(&[Val::new(1), Val::new(3)]);
    assert_ne!(h1, h3);
  }

  #[test]
  fn test_recursive_stark_proof_over_m31ext3_mul() {
    // 비자명한 M31³ 곱을 내부 증명으로 재귀 증명
    let a = ch(1, 1, 0);
    let b = ch(1, 0, 1);
    let inner_proof = prove_m31ext3_mul(&[(a, b)]);
    let expected_tc: [Val; 8] = inner_proof.commitments.trace.into();

    let recursive = prove_recursively(&inner_proof, &[], false);
    let r = verify_recursively(&recursive, &[], &expected_tc);
    assert!(r.is_ok(), "recursive m31ext3: {:?}", r.err());
  }
}
