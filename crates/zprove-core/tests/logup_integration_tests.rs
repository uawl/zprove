/// End-to-end integration test for the `prove_with_lookup` / `verify_with_lookup`
/// path in our patched `p3-uni-stark`.
#[cfg(test)]
mod logup_integration_tests {
  use p3_air::{Air, AirBuilderWithPublicValues, BaseAir, PairBuilder};
  use p3_field::PrimeCharacteristicRing;
  use p3_lookup::logup::LogUpGadget;
  use p3_lookup::lookup_traits::{Kind, Lookup, LookupData, LookupGadget};
  use p3_matrix::dense::RowMajorMatrix;
  use p3_uni_stark::{
    Entry, SymbolicExpression, SymbolicVariable, VerifierConstraintFolder, prove_with_lookup,
    verify_with_lookup,
  };
  use zprove_core::zk_proof::make_circle_config;

  type SC = zprove_core::zk_proof::CircleStarkConfig;
  type F = p3_mersenne_31::Mersenne31;
  type EF = <SC as p3_uni_stark::StarkGenericConfig>::Challenge;

  struct LocalSumAir {
    width: usize,
  }

  impl<F: PrimeCharacteristicRing + Clone> BaseAir<F> for LocalSumAir {
    fn width(&self) -> usize {
      self.width
    }
  }

  impl<AB> Air<AB> for LocalSumAir
  where
    AB: AirBuilderWithPublicValues + PairBuilder,
  {
    fn eval(&self, _builder: &mut AB) {}
  }

  fn build_trace() -> RowMajorMatrix<F> {
    let one = F::ONE;
    let neg_one = -F::ONE;
    let two = F::from_u8(2);
    RowMajorMatrix::new(
      vec![
        one, one, // row 0: value=1, mult=+1
        two, one, // row 1: value=2, mult=+1
        one, neg_one, // row 2: value=1, mult=-1
        two, neg_one, // row 3: value=2, mult=-1
      ],
      2,
    )
  }

  fn make_lookup() -> Lookup<F> {
    let val_expr =
      SymbolicExpression::Variable(SymbolicVariable::new(Entry::Main { offset: 0 }, 0));
    let mult_expr =
      SymbolicExpression::Variable(SymbolicVariable::new(Entry::Main { offset: 0 }, 1));
    Lookup::new(Kind::Local, vec![vec![val_expr]], vec![mult_expr], vec![0])
  }

  #[test]
  fn test_logup_local_prove_verify_roundtrip() {
    let config = make_circle_config();
    let air = LocalSumAir { width: 2 };
    let trace = build_trace();

    let trace_for_perm = trace.clone();
    let lookup_perm = make_lookup();
    let lookup_prove_eval = make_lookup();
    let lookup_verify_eval = make_lookup();

    // --- prove ---
    let proof = prove_with_lookup(
      &config,
      &air,
      trace,
      &[],
      None,
      move |perm_challenges| {
        let gadget = LogUpGadget::new();
        let mut lookup_data: Vec<LookupData<EF>> = vec![];
        let perm_trace = gadget.generate_permutation::<SC>(
          &trace_for_perm,
          &None,
          &[],
          &[lookup_perm.clone()],
          &mut lookup_data,
          perm_challenges,
        );
        Some(perm_trace)
      },
      2, // num_perm_challenges (1 lookup × 2 challenges)
      3, // lookup_constraint_count (first_row + transition + last_row)
      move |folder: &mut VerifierConstraintFolder<SC>| {
        let gadget = LogUpGadget::new();
        gadget.eval_local_lookup(folder, &lookup_prove_eval);
      },
    );

    // --- verify ---
    let result = verify_with_lookup(
      &config,
      &air,
      &proof,
      &[],
      None,
      2,
      move |folder: &mut VerifierConstraintFolder<SC>| {
        let gadget = LogUpGadget::new();
        gadget.eval_local_lookup(folder, &lookup_verify_eval);
      },
    );

    assert!(
      result.is_ok(),
      "verify_with_lookup failed: {:?}",
      result.err()
    );
  }
}
