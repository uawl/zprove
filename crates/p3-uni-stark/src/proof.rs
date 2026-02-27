use alloc::vec::Vec;

use p3_commit::Pcs;
use serde::{Deserialize, Serialize};

use crate::StarkGenericConfig;

type Com<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
  <SC as StarkGenericConfig>::Challenge,
  <SC as StarkGenericConfig>::Challenger,
>>::Commitment;
type PcsProof<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
  <SC as StarkGenericConfig>::Challenge,
  <SC as StarkGenericConfig>::Challenger,
>>::Proof;

#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Proof<SC: StarkGenericConfig> {
  pub commitments: Commitments<Com<SC>>,
  pub opened_values: OpenedValues<SC::Challenge>,
  pub opening_proof: PcsProof<SC>,
  pub degree_bits: usize,
}

impl<SC: StarkGenericConfig> Clone for Proof<SC>
where
  Com<SC>: Clone,
  PcsProof<SC>: Clone,
  SC::Challenge: Clone,
{
  fn clone(&self) -> Self {
    Self {
      commitments: self.commitments.clone(),
      opened_values: self.opened_values.clone(),
      opening_proof: self.opening_proof.clone(),
      degree_bits: self.degree_bits,
    }
  }
}

impl<SC: StarkGenericConfig> core::fmt::Debug for Proof<SC>
where
  Com<SC>: core::fmt::Debug,
  PcsProof<SC>: core::fmt::Debug,
  SC::Challenge: core::fmt::Debug,
{
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("Proof")
      .field("commitments", &self.commitments)
      .field("opened_values", &self.opened_values)
      .field("degree_bits", &self.degree_bits)
      .finish_non_exhaustive()
  }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Commitments<Com> {
  pub trace: Com,
  pub quotient_chunks: Com,
  pub random: Option<Com>,
  /// Permutation (lookup) trace commitment — `None` if no lookup argument.
  pub perm: Option<Com>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenedValues<Challenge> {
  pub trace_local: Vec<Challenge>,
  pub trace_next: Vec<Challenge>,
  pub preprocessed_local: Option<Vec<Challenge>>,
  pub preprocessed_next: Option<Vec<Challenge>>, // may not always be necessary
  pub quotient_chunks: Vec<Vec<Challenge>>,
  pub random: Option<Vec<Challenge>>,
  /// Opened values for the permutation trace at `zeta` — each element is a
  /// base-field value (the perm trace is flattened EF → base field).
  pub perm_local: Option<Vec<Challenge>>,
  /// Opened values for the permutation trace at `zeta_next`.
  pub perm_next: Option<Vec<Challenge>>,
}
