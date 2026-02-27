//! See [`crate::prover`] for an overview of the protocol and a more detailed soundness analysis.

use alloc::string::String;
use alloc::vec::Vec;
use alloc::{format, vec};

use itertools::Itertools;
use p3_air::Air;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::VerticalPair;
use p3_util::zip_eq::zip_eq;
use thiserror::Error;
use tracing::instrument;

use crate::symbolic_builder::{SymbolicAirBuilder, get_log_num_quotient_chunks};
use crate::{
  Domain, PcsError, PreprocessedVerifierKey, Proof, StarkGenericConfig, Val,
  VerifierConstraintFolder,
};

/// Recomposes the quotient polynomial from its chunks evaluated at a point.
///
/// Given quotient chunks and their domains, this computes the Lagrange
/// interpolation coefficients (zps) and reconstructs quotient(zeta).
pub fn recompose_quotient_from_chunks<SC>(
  quotient_chunks_domains: &[Domain<SC>],
  quotient_chunks: &[Vec<SC::Challenge>],
  zeta: SC::Challenge,
) -> SC::Challenge
where
  SC: StarkGenericConfig,
{
  let zps = quotient_chunks_domains
    .iter()
    .enumerate()
    .map(|(i, domain)| {
      quotient_chunks_domains
        .iter()
        .enumerate()
        .filter(|(j, _)| *j != i)
        .map(|(_, other_domain)| {
          other_domain.vanishing_poly_at_point(zeta)
            * other_domain
              .vanishing_poly_at_point(domain.first_point())
              .inverse()
        })
        .product::<SC::Challenge>()
    })
    .collect_vec();

  quotient_chunks
    .iter()
    .enumerate()
    .map(|(ch_i, ch)| {
      // We checked in valid_shape the length of "ch" is equal to
      // <SC::Challenge as BasedVectorSpace<Val<SC>>>::DIMENSION. Hence
      // the unwrap() will never panic.
      zps[ch_i]
        * ch
          .iter()
          .enumerate()
          .map(|(e_i, &c)| SC::Challenge::ith_basis_element(e_i).unwrap() * c)
          .sum::<SC::Challenge>()
    })
    .sum::<SC::Challenge>()
}

/// Verifies that the folded constraints match the quotient polynomial at zeta.
///
/// This evaluates the [`Air`] constraints at the out-of-domain point and checks
/// that constraints(zeta) / Z_H(zeta) = quotient(zeta).
#[allow(clippy::too_many_arguments)]
pub fn verify_constraints<SC, A, PcsErr>(
  air: &A,
  trace_local: &[SC::Challenge],
  trace_next: &[SC::Challenge],
  preprocessed_local: Option<&[SC::Challenge]>,
  preprocessed_next: Option<&[SC::Challenge]>,
  public_values: &[Val<SC>],
  trace_domain: Domain<SC>,
  zeta: SC::Challenge,
  alpha: SC::Challenge,
  quotient: SC::Challenge,
) -> Result<(), VerificationError<PcsErr>>
where
  SC: StarkGenericConfig,
  A: for<'a> Air<VerifierConstraintFolder<'a, SC>>,
  PcsErr: core::fmt::Debug,
{
  let sels = trace_domain.selectors_at_point(zeta);

  let main = VerticalPair::new(
    RowMajorMatrixView::new_row(trace_local),
    RowMajorMatrixView::new_row(trace_next),
  );

  let preprocessed = match (preprocessed_local, preprocessed_next) {
    (Some(local), Some(next)) => Some(VerticalPair::new(
      RowMajorMatrixView::new_row(local),
      RowMajorMatrixView::new_row(next),
    )),
    _ => None,
  };

  let mut folder = VerifierConstraintFolder {
    main,
    preprocessed,
    permutation: None,
    permutation_challenges: &[],
    public_values,
    is_first_row: sels.is_first_row,
    is_last_row: sels.is_last_row,
    is_transition: sels.is_transition,
    alpha,
    accumulator: SC::Challenge::ZERO,
  };
  air.eval(&mut folder);
  let folded_constraints = folder.accumulator;

  // Check that constraints(zeta) / Z_H(zeta) = quotient(zeta)
  let lhs = folded_constraints * sels.inv_vanishing;
  if lhs != quotient {
    return Err(VerificationError::OodEvaluationMismatch { index: None });
  }

  Ok(())
}

/// Validates and commits the preprocessed trace if present.
/// Returns the preprocessed width and its commitment hash (available iff width > 0).
#[allow(clippy::type_complexity)]
fn process_preprocessed_trace<SC, A>(
  air: &A,
  opened_values: &crate::proof::OpenedValues<SC::Challenge>,
  preprocessed_vk: Option<&PreprocessedVerifierKey<SC>>,
) -> Result<
  (
    usize,
    Option<<SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Commitment>,
  ),
  VerificationError<PcsError<SC>>,
>
where
  SC: StarkGenericConfig,
  A: for<'a> Air<VerifierConstraintFolder<'a, SC>>,
{
  // Determine expected preprocessed width.
  // - If a verifier key is provided, trust its width.
  // - Otherwise, derive width from the AIR's preprocessed trace (if any).
  let preprocessed_width = preprocessed_vk
    .map(|vk| vk.width)
    .or_else(|| air.preprocessed_trace().as_ref().map(|m| m.width))
    .unwrap_or(0);

  // Check that the proof's opened preprocessed values match the expected width.
  let preprocessed_local_len = opened_values
    .preprocessed_local
    .as_ref()
    .map_or(0, |v| v.len());
  let preprocessed_next_len = opened_values
    .preprocessed_next
    .as_ref()
    .map_or(0, |v| v.len());
  if preprocessed_width != preprocessed_local_len || preprocessed_width != preprocessed_next_len {
    // Verifier expects preprocessed trace while proof does not have it, or vice versa
    return Err(VerificationError::InvalidProofShape);
  }

  // Validate consistency between width, verifier key, and zk settings.
  match (preprocessed_width, preprocessed_vk) {
    // Case: No preprocessed columns.
    //
    // Valid only if no verifier key is provided.
    (0, None) => Ok((0, None)),

    // Case: Preprocessed columns exist.
    //
    // Valid only if VK exists, widths match, and we are NOT in zk mode.
    (w, Some(vk)) if w == vk.width => Ok((w, Some(vk.commitment.clone()))),

    // Catch-all for invalid states, such as:
    // - Width is 0 but VK is provided.
    // - Width > 0 but VK is missing.
    // - Width > 0 but VK width mismatches the expected width.
    _ => Err(VerificationError::InvalidProofShape),
  }
}

/// Replays the Fiat-Shamir challenger up to (but not including) the final `pcs.verify()` call.
///
/// This allows callers to then observe the PCS proof's `commit_phase_commits` to
/// derive the FRI fold betas, without needing to know the AIR definition.
///
/// The shape of `opened_values` is inferred from the proof itself:
/// - `trace_width` = `opened_values.trace_local.len()`
/// - `num_quotient_chunks` = `opened_values.quotient_chunks.len()`
/// - `preprocessed_width` = `opened_values.preprocessed_local.as_ref().map(|v| v.len()).unwrap_or(0)`
///
/// Returns the challenger in the state it would be in just before `pcs.verify()` is called.
#[allow(dead_code)]
pub fn replay_challenger_pre_pcs<SC>(
  config: &SC,
  proof: &Proof<SC>,
  public_values: &[Val<SC>],
) -> SC::Challenger
where
  SC: StarkGenericConfig,
{
  use p3_challenger::CanObserve;

  let Proof {
    commitments,
    opened_values,
    degree_bits,
    ..
  } = proof;

  let preprocessed_width = opened_values
    .preprocessed_local
    .as_ref()
    .map(|v| v.len())
    .unwrap_or(0);

  let mut challenger = config.initialise_challenger();

  challenger.observe(Val::<SC>::from_usize(*degree_bits));
  challenger.observe(Val::<SC>::from_usize(*degree_bits - config.is_zk()));
  challenger.observe(Val::<SC>::from_usize(preprocessed_width));
  challenger.observe(commitments.trace.clone());
  if preprocessed_width > 0 {
    // We don't have the preprocessed commitment here; skip for now.
    // In practice, recursive proofs don't use preprocessed traces.
  }
  challenger.observe_slice(public_values);

  // alpha (constraint aggregation challenge)
  let _alpha: SC::Challenge = challenger.sample_algebra_element();
  challenger.observe(commitments.quotient_chunks.clone());

  if let Some(r_commit) = commitments.random.clone() {
    challenger.observe(r_commit);
  }

  // zeta (OOD evaluation point) — sampled but not needed here
  let _zeta: SC::Challenge = challenger.sample_algebra_element();

  // The challenger is now in the state where pcs.verify() receives it.
  challenger
}

#[instrument(skip_all)]
pub fn verify<SC, A>(
  config: &SC,
  air: &A,
  proof: &Proof<SC>,
  public_values: &[Val<SC>],
) -> Result<(), VerificationError<PcsError<SC>>>
where
  SC: StarkGenericConfig,
  A: Air<SymbolicAirBuilder<Val<SC>>> + for<'a> Air<VerifierConstraintFolder<'a, SC>>,
{
  verify_with_preprocessed(config, air, proof, public_values, None)
}

#[instrument(skip_all)]
pub fn verify_with_preprocessed<SC, A>(
  config: &SC,
  air: &A,
  proof: &Proof<SC>,
  public_values: &[Val<SC>],
  preprocessed_vk: Option<&PreprocessedVerifierKey<SC>>,
) -> Result<(), VerificationError<PcsError<SC>>>
where
  SC: StarkGenericConfig,
  A: Air<SymbolicAirBuilder<Val<SC>>> + for<'a> Air<VerifierConstraintFolder<'a, SC>>,
{
  let Proof {
    commitments,
    opened_values,
    opening_proof,
    degree_bits,
  } = proof;

  let pcs = config.pcs();
  let degree = 1 << degree_bits;
  let trace_domain = pcs.natural_domain_for_degree(degree);
  // TODO: allow moving preprocessed commitment to preprocess time, if known in advance
  let (preprocessed_width, preprocessed_commit) =
    process_preprocessed_trace::<SC, A>(air, opened_values, preprocessed_vk)?;

  // Ensure the preprocessed trace and main trace have the same height.
  if let Some(vk) = preprocessed_vk
    && preprocessed_width > 0
    && vk.degree_bits != *degree_bits
  {
    return Err(VerificationError::InvalidProofShape);
  }

  let log_num_quotient_chunks = get_log_num_quotient_chunks::<Val<SC>, A>(
    air,
    preprocessed_width,
    public_values.len(),
    config.is_zk(),
  );
  let num_quotient_chunks = 1 << (log_num_quotient_chunks + config.is_zk());
  let mut challenger = config.initialise_challenger();
  let init_trace_domain = pcs.natural_domain_for_degree(degree >> (config.is_zk()));

  let quotient_domain =
    trace_domain.create_disjoint_domain(1 << (degree_bits + log_num_quotient_chunks));
  let quotient_chunks_domains = quotient_domain.split_domains(num_quotient_chunks);

  let randomized_quotient_chunks_domains = quotient_chunks_domains
    .iter()
    .map(|domain| pcs.natural_domain_for_degree(domain.size() << (config.is_zk())))
    .collect_vec();
  // Check that the random commitments are/are not present depending on the ZK setting.
  // - If ZK is enabled, the prover should have random commitments.
  // - If ZK is not enabled, the prover should not have random commitments.
  if (opened_values.random.is_some() != SC::Pcs::ZK)
    || (commitments.random.is_some() != SC::Pcs::ZK)
  {
    return Err(VerificationError::RandomizationError);
  }

  let air_width = A::width(air);
  let valid_shape = opened_values.trace_local.len() == air_width
        && opened_values.trace_next.len() == air_width
        && opened_values.quotient_chunks.len() == num_quotient_chunks
        && opened_values
            .quotient_chunks
            .iter()
            .all(|qc| qc.len() == SC::Challenge::DIMENSION)
        // We've already checked that opened_values.random is present if and only if ZK is enabled.
        && opened_values.random.as_ref().is_none_or(|r_comm| r_comm.len() == SC::Challenge::DIMENSION);
  if !valid_shape {
    return Err(VerificationError::InvalidProofShape);
  }

  // Observe the instance.
  challenger.observe(Val::<SC>::from_usize(proof.degree_bits));
  challenger.observe(Val::<SC>::from_usize(proof.degree_bits - config.is_zk()));
  challenger.observe(Val::<SC>::from_usize(preprocessed_width));
  // TODO: Might be best practice to include other instance data here in the transcript, like some
  // encoding of the AIR. This protects against transcript collisions between distinct instances.
  // Practically speaking though, the only related known attack is from failing to include public
  // values. It's not clear if failing to include other instance data could enable a transcript
  // collision, since most such changes would completely change the set of satisfying witnesses.
  challenger.observe(commitments.trace.clone());
  if preprocessed_width > 0 {
    challenger.observe(preprocessed_commit.as_ref().unwrap().clone());
  }
  challenger.observe_slice(public_values);

  // Get the first Fiat Shamir challenge which will be used to combine all constraint polynomials
  // into a single polynomial.
  //
  // Soundness Error: n/|EF| where n is the number of constraints.
  let alpha = challenger.sample_algebra_element();
  challenger.observe(commitments.quotient_chunks.clone());

  // We've already checked that commitments.random is present if and only if ZK is enabled.
  // Observe the random commitment if it is present.
  if let Some(r_commit) = commitments.random.clone() {
    challenger.observe(r_commit);
  }

  // Get an out-of-domain point to open our values at.
  //
  // Soundness Error: dN/|EF| where `N` is the trace length and our constraint polynomial has degree `d`.
  let zeta = challenger.sample_algebra_element();
  let zeta_next = init_trace_domain
    .next_point(zeta)
    .ok_or(VerificationError::NextPointUnavailable)?;

  // We've already checked that commitments.random and opened_values.random are present if and only if ZK is enabled.
  let mut coms_to_verify = if let Some(random_commit) = &commitments.random {
    let random_values = opened_values
      .random
      .as_ref()
      .ok_or(VerificationError::RandomizationError)?;
    vec![(
      random_commit.clone(),
      vec![(trace_domain, vec![(zeta, random_values.clone())])],
    )]
  } else {
    vec![]
  };
  coms_to_verify.extend(vec![
    (
      commitments.trace.clone(),
      vec![(
        trace_domain,
        vec![
          (zeta, opened_values.trace_local.clone()),
          (zeta_next, opened_values.trace_next.clone()),
        ],
      )],
    ),
    (
      commitments.quotient_chunks.clone(),
      // Check the commitment on the randomized domains.
      zip_eq(
        randomized_quotient_chunks_domains.iter(),
        &opened_values.quotient_chunks,
        VerificationError::InvalidProofShape,
      )?
      .map(|(domain, values)| (*domain, vec![(zeta, values.clone())]))
      .collect_vec(),
    ),
  ]);

  // Add preprocessed commitment verification if present
  if preprocessed_width > 0 {
    coms_to_verify.push((
      preprocessed_commit.unwrap(),
      vec![(
        trace_domain,
        vec![
          (zeta, opened_values.preprocessed_local.clone().unwrap()),
          (zeta_next, opened_values.preprocessed_next.clone().unwrap()),
        ],
      )],
    ));
  }

  pcs
    .verify(coms_to_verify, opening_proof, &mut challenger)
    .map_err(VerificationError::InvalidOpeningArgument)?;

  let quotient = recompose_quotient_from_chunks::<SC>(
    &quotient_chunks_domains,
    &opened_values.quotient_chunks,
    zeta,
  );

  verify_constraints::<SC, A, PcsError<SC>>(
    air,
    &opened_values.trace_local,
    &opened_values.trace_next,
    opened_values.preprocessed_local.as_deref(),
    opened_values.preprocessed_next.as_deref(),
    public_values,
    init_trace_domain,
    zeta,
    alpha,
    quotient,
  )?;

  Ok(())
}

/// Defines errors that can occur during lookup verification.
#[derive(Debug)]
pub enum LookupError {
  /// Error indicating that the global cumulative sum is incorrect.
  GlobalCumulativeMismatch(Option<String>),
}

#[derive(Debug, Error)]
pub enum VerificationError<PcsErr>
where
  PcsErr: core::fmt::Debug,
{
  #[error("invalid proof shape")]
  InvalidProofShape,
  /// An error occurred while verifying the claimed openings.
  #[error("invalid opening argument: {0:?}")]
  InvalidOpeningArgument(PcsErr),
  /// Out-of-domain evaluation mismatch, i.e. `constraints(zeta)` did not match
  /// `quotient(zeta) Z_H(zeta)`.
  #[error("out-of-domain evaluation mismatch{}", .index.map(|i| format!(" at index {}", i)).unwrap_or_default())]
  OodEvaluationMismatch { index: Option<usize> },
  /// The FRI batch randomization does not correspond to the ZK setting.
  #[error("randomization error: FRI batch randomization does not match ZK setting")]
  RandomizationError,
  /// The domain does not support computing the next point algebraically.
  #[error("next point unavailable: domain does not support computing the next point algebraically")]
  NextPointUnavailable,
  /// Lookup related error
  #[error("lookup error: {0:?}")]
  LookupError(LookupError),
}

/// Verifies a STARK proof that includes a permutation/lookup argument (e.g. LogUp).
///
/// The transcript must match the one produced by [`prove_with_lookup`]:
/// permutation challenges are sampled *before* alpha, and the permutation commitment
/// (if any) is observed between perm-challenge sampling and alpha sampling.
///
/// # Parameters
/// - `num_perm_challenges`: must equal the value used in `prove_with_lookup`.
/// - `eval_lookup`: evaluates lookup constraints given a `VerifierConstraintFolder`
///   that already has `permutation` and `permutation_challenges` populated.
///   It should call `folder.assert_zero_ext(...)` for each lookup constraint.
#[instrument(skip_all)]
pub fn verify_with_lookup<SC, A>(
  config: &SC,
  air: &A,
  proof: &Proof<SC>,
  public_values: &[Val<SC>],
  preprocessed_vk: Option<&PreprocessedVerifierKey<SC>>,
  num_perm_challenges: usize,
  eval_lookup: impl for<'a> Fn(&mut VerifierConstraintFolder<'a, SC>),
) -> Result<(), VerificationError<PcsError<SC>>>
where
  SC: StarkGenericConfig,
  A: Air<SymbolicAirBuilder<Val<SC>>> + for<'a> Air<VerifierConstraintFolder<'a, SC>>,
{
  let Proof {
    commitments,
    opened_values,
    opening_proof,
    degree_bits,
  } = proof;

  let pcs = config.pcs();
  let degree = 1 << degree_bits;
  let trace_domain = pcs.natural_domain_for_degree(degree);
  let (preprocessed_width, preprocessed_commit) =
    process_preprocessed_trace::<SC, A>(air, opened_values, preprocessed_vk)?;

  if let Some(vk) = preprocessed_vk
    && preprocessed_width > 0
    && vk.degree_bits != *degree_bits
  {
    return Err(VerificationError::InvalidProofShape);
  }

  let log_num_quotient_chunks = get_log_num_quotient_chunks::<Val<SC>, A>(
    air,
    preprocessed_width,
    public_values.len(),
    config.is_zk(),
  );
  let num_quotient_chunks = 1 << (log_num_quotient_chunks + config.is_zk());
  let mut challenger = config.initialise_challenger();
  let init_trace_domain = pcs.natural_domain_for_degree(degree >> (config.is_zk()));

  let quotient_domain =
    trace_domain.create_disjoint_domain(1 << (degree_bits + log_num_quotient_chunks));
  let quotient_chunks_domains = quotient_domain.split_domains(num_quotient_chunks);
  let randomized_quotient_chunks_domains = quotient_chunks_domains
    .iter()
    .map(|domain| pcs.natural_domain_for_degree(domain.size() << (config.is_zk())))
    .collect_vec();

  if (opened_values.random.is_some() != SC::Pcs::ZK)
    || (commitments.random.is_some() != SC::Pcs::ZK)
  {
    return Err(VerificationError::RandomizationError);
  }

  // ── proof shape check ────────────────────────────────────────────────────
  let d = <SC::Challenge as BasedVectorSpace<Val<SC>>>::DIMENSION;
  let air_width = A::width(air);
  let has_perm = commitments.perm.is_some();

  // Perm local/next must be present iff perm commitment is present.
  if opened_values.perm_local.is_some() != has_perm || opened_values.perm_next.is_some() != has_perm
  {
    return Err(VerificationError::InvalidProofShape);
  }

  // If perm is present, both rows must have the same length, and length must
  // be a multiple of D (EF dimension).
  let perm_ef_width = if has_perm {
    let local_len = opened_values.perm_local.as_ref().unwrap().len();
    let next_len = opened_values.perm_next.as_ref().unwrap().len();
    if local_len != next_len || local_len % d != 0 {
      return Err(VerificationError::InvalidProofShape);
    }
    local_len / d
  } else {
    0
  };

  let valid_shape = opened_values.trace_local.len() == air_width
    && opened_values.trace_next.len() == air_width
    && opened_values.quotient_chunks.len() == num_quotient_chunks
    && opened_values.quotient_chunks.iter().all(|qc| qc.len() == d)
    && opened_values.random.as_ref().is_none_or(|r| r.len() == d);
  if !valid_shape {
    return Err(VerificationError::InvalidProofShape);
  }

  // ── transcript ────────────────────────────────────────────────────────────
  challenger.observe(Val::<SC>::from_usize(proof.degree_bits));
  challenger.observe(Val::<SC>::from_usize(proof.degree_bits - config.is_zk()));
  challenger.observe(Val::<SC>::from_usize(preprocessed_width));
  challenger.observe(commitments.trace.clone());
  if preprocessed_width > 0 {
    challenger.observe(preprocessed_commit.as_ref().unwrap().clone());
  }
  challenger.observe_slice(public_values);

  // Sample permutation challenges (matches prove_with_lookup transcript).
  challenger.observe(Val::<SC>::from_usize(num_perm_challenges));
  let perm_challenges: Vec<SC::Challenge> = (0..num_perm_challenges)
    .map(|_| challenger.sample_algebra_element())
    .collect();

  // Observe perm commitment if present.
  if let Some(perm_commit) = &commitments.perm {
    challenger.observe(perm_commit.clone());
  }

  let alpha: SC::Challenge = challenger.sample_algebra_element();
  challenger.observe(commitments.quotient_chunks.clone());
  if let Some(r_commit) = commitments.random.clone() {
    challenger.observe(r_commit);
  }

  let zeta: SC::Challenge = challenger.sample_algebra_element();
  let zeta_next = init_trace_domain
    .next_point(zeta)
    .ok_or(VerificationError::NextPointUnavailable)?;

  // ── build coms_to_verify ─────────────────────────────────────────────────
  let mut coms_to_verify = if let Some(random_commit) = &commitments.random {
    let random_values = opened_values
      .random
      .as_ref()
      .ok_or(VerificationError::RandomizationError)?;
    vec![(
      random_commit.clone(),
      vec![(trace_domain, vec![(zeta, random_values.clone())])],
    )]
  } else {
    vec![]
  };

  coms_to_verify.extend(vec![
    (
      commitments.trace.clone(),
      vec![(
        trace_domain,
        vec![
          (zeta, opened_values.trace_local.clone()),
          (zeta_next, opened_values.trace_next.clone()),
        ],
      )],
    ),
    (
      commitments.quotient_chunks.clone(),
      zip_eq(
        randomized_quotient_chunks_domains.iter(),
        &opened_values.quotient_chunks,
        VerificationError::InvalidProofShape,
      )?
      .map(|(domain, values)| (*domain, vec![(zeta, values.clone())]))
      .collect_vec(),
    ),
  ]);

  if preprocessed_width > 0 {
    coms_to_verify.push((
      preprocessed_commit.unwrap(),
      vec![(
        trace_domain,
        vec![
          (zeta, opened_values.preprocessed_local.clone().unwrap()),
          (zeta_next, opened_values.preprocessed_next.clone().unwrap()),
        ],
      )],
    ));
  }

  // Add perm trace verification if perm commitment is present.
  if let Some(perm_commit) = &commitments.perm {
    coms_to_verify.push((
      perm_commit.clone(),
      vec![(
        trace_domain,
        vec![
          (zeta, opened_values.perm_local.clone().unwrap()),
          (zeta_next, opened_values.perm_next.clone().unwrap()),
        ],
      )],
    ));
  }

  pcs
    .verify(coms_to_verify, opening_proof, &mut challenger)
    .map_err(VerificationError::InvalidOpeningArgument)?;

  // ── constraint check ─────────────────────────────────────────────────────
  let quotient = recompose_quotient_from_chunks::<SC>(
    &quotient_chunks_domains,
    &opened_values.quotient_chunks,
    zeta,
  );

  let sels = init_trace_domain.selectors_at_point(zeta);

  let main = VerticalPair::new(
    RowMajorMatrixView::new_row(&opened_values.trace_local),
    RowMajorMatrixView::new_row(&opened_values.trace_next),
  );

  let preprocessed = match (
    opened_values.preprocessed_local.as_deref(),
    opened_values.preprocessed_next.as_deref(),
  ) {
    (Some(local), Some(next)) => Some(VerticalPair::new(
      RowMajorMatrixView::new_row(local),
      RowMajorMatrixView::new_row(next),
    )),
    _ => None,
  };

  // Reconstruct EF perm rows from flattened base-field opened values.
  // perm_local_bf has perm_ef_width * D elements (each an EF value = base poly at zeta).
  // ef_perm_local[c] = Σ_{k=0}^{D-1} ith_basis_element(k) * perm_local_bf[c*D + k]
  let perm_view = if has_perm {
    let perm_local_bf = opened_values.perm_local.as_ref().unwrap();
    let perm_next_bf = opened_values.perm_next.as_ref().unwrap();
    let recon = |flat: &[SC::Challenge]| -> Vec<SC::Challenge> {
      (0..perm_ef_width)
        .map(|c| {
          (0..d)
            .map(|k| SC::Challenge::ith_basis_element(k).unwrap() * flat[c * d + k])
            .sum::<SC::Challenge>()
        })
        .collect()
    };
    let local_ef = recon(perm_local_bf);
    let next_ef = recon(perm_next_bf);
    // Keep the vecs alive for the duration of folder use, store in a Box.
    Some((local_ef, next_ef))
  } else {
    None
  };

  let (perm_view_pair, p_challenges_ref): (Option<VerticalPair<_, _>>, &[SC::Challenge]) =
    if let Some((ref local, ref next)) = perm_view {
      (
        Some(VerticalPair::new(
          RowMajorMatrixView::new_row(local),
          RowMajorMatrixView::new_row(next),
        )),
        &perm_challenges,
      )
    } else {
      (None, &[])
    };

  let mut folder = VerifierConstraintFolder {
    main,
    preprocessed,
    permutation: perm_view_pair,
    permutation_challenges: p_challenges_ref,
    public_values,
    is_first_row: sels.is_first_row,
    is_last_row: sels.is_last_row,
    is_transition: sels.is_transition,
    alpha,
    accumulator: SC::Challenge::ZERO,
  };

  // Evaluate main constraints, then extend with lookup constraints.
  // Both use the same Horner accumulator, giving the combined polynomial
  // P(alpha) = alpha^{lookup_count} * main_horner + lookup_horner,
  // which matches the prover's quotient construction.
  air.eval(&mut folder);
  if has_perm {
    eval_lookup(&mut folder);
  }

  let folded_constraints = folder.accumulator;
  if folded_constraints * sels.inv_vanishing != quotient {
    return Err(VerificationError::OodEvaluationMismatch { index: None });
  }

  Ok(())
}
