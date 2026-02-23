# Stwo Circle STARK Library — API Reference for Custom AIR Components

> Based on `stwo v2.1.0`, `stwo-constraint-framework v2.1.0`, and `stwo-air-utils v2.1.0`.

## Table of Contents

1. [M31 Field Types](#1-m31-field-types)
2. [Key Traits: `FrameworkEval`, `EvalAtRow`, `Component`](#2-key-traits)
3. [Trace & Column Building](#3-trace--column-building)
4. [Adding Constraints (equality, lookup / logup)](#4-adding-constraints)
5. [Prove & Verify](#5-prove--verify)
6. [Minimal End-to-End Example (wide Fibonacci)](#6-minimal-end-to-end-example)

---

## Cargo Dependencies

```toml
[dependencies]
stwo = { version = "2.1", features = ["prover", "parallel"] }
stwo-constraint-framework = "2.1"
stwo-air-utils = "2.1"
```

> **Important:** The `prover` feature flag gates `CommitmentSchemeProver`, `prove()`, `ComponentProver`, 
> and all SIMD backend prover functionality. Without it you only get the verifier.

---

## 1. M31 Field Types

The field tower is:

| Type | Alias | Degree over M31 | Description |
|------|-------|-----------------|-------------|
| `M31` | `BaseField` | 1 | Mersenne-31 field, $P = 2^{31} - 1$ |
| `CM31` | — | 2 | Complex extension $\text{M31}[i]$ where $i^2 + 1 = 0$ |
| `QM31` | `SecureField` | 4 | Quartic extension (pair of `CM31`) |

### Creating M31 Values

```rust
use stwo::core::fields::m31::{M31, BaseField, P};

// From Rust integer types
let a = M31::from(42u32);          // From<u32>  (auto-reduces mod P)
let b = M31::from(-1i32);          // From<i32>  => M31(P - 1)
let c = BaseField::from(100usize); // From<usize>

// Unchecked (caller guarantees value < P)
let d = M31::from_u32_unchecked(7);

// From a u64 with explicit modular reduction
let e = M31::reduce(123456789u64);
```

### Arithmetic

All standard ring ops are implemented: `Add`, `Sub`, `Mul`, `Neg`, `AddAssign`, `SubAssign`, `MulAssign`.

```rust
use stwo::core::fields::FieldExpOps;

let inv   = a.inverse();          // modular inverse
let sq    = a.square();           // a * a (optimised)
let power = a.pow(1000u128);      // square-and-multiply
let doubled = a.double();         // a + a (Field trait)
```

Batch operations:
```rust
use stwo::core::fields::batch_inverse;
let inverses = batch_inverse(&[a, b, c]); // Montgomery batch-inverse
```

### Constants

```rust
use stwo::core::fields::m31::P;               // 2^31 - 1
use stwo::core::fields::m31::MODULUS_BITS;     // 31
use num_traits::{Zero, One};
let zero = M31::zero();
let one  = M31::one();
```

### SecureField (QM31)

```rust
use stwo::core::fields::qm31::{QM31, SecureField};
// SecureField = QM31 = (CM31, CM31) = ((M31, M31), (M31, M31))
```

---

## 2. Key Traits

### 2.1 `FrameworkEval` — Define Your AIR Component

```rust
// In stwo_constraint_framework
pub trait FrameworkEval {
    fn log_size(&self) -> u32;
    fn max_constraint_log_degree_bound(&self) -> u32;
    fn evaluate<E: EvalAtRow>(&self, eval: E) -> E;
}
```

| Method | Purpose |
|--------|---------|
| `log_size()` | Returns $\log_2$ of the number of rows in this component's trace |
| `max_constraint_log_degree_bound()` | Returns the max $\log_2$ degree of any constraint polynomial. Typically `log_size() + 1` for degree-2 constraints |
| `evaluate()` | The heart of your AIR: read trace cells and add constraints using the `EvalAtRow` interface |

**Implementing `FrameworkEval` automatically provides `Component` and `ComponentProver` (for the SIMD backend) via `FrameworkComponent<YourEval>`.**

### 2.2 `EvalAtRow` — The Constraint DSL

```rust
pub trait EvalAtRow {
    type F: Clone + Debug + Zero + Neg<Output=Self::F> + Add<...> + Sub<...> + Mul<...>;
    type EF: Clone + Debug + Zero + Neg<Output=Self::EF> + Add<...> + Sub<...> + Mul<...>;

    // Required
    fn next_interaction_mask(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N];
    fn add_constraint<G>(&mut self, constraint: G);
    fn combine_ef(values: [Self::F; 4]) -> Self::EF;

    // Provided (commonly used)
    fn next_trace_mask(&mut self) -> Self::F;                // shorthand: interaction=0, offset=[0]
    fn get_preprocessed_column(&mut self, column: PreprocessedColumn) -> Self::F;
    fn add_to_relation<R: Relation<Self::F, Self::EF>>(
        &mut self,
        entry: RelationEntry<'_, Self::F, Self::EF, R>,
    );
    fn write_logup_frac(&mut self, fraction: Fraction<Self::EF, Self::EF>);
    fn finalize_logup(&mut self);
}
```

Key methods:

| Method | What it does |
|--------|-------------|
| `next_trace_mask()` | Returns the value of the next unread column at the current row (original trace, offset 0) |
| `next_interaction_mask(interaction, offsets)` | Read from a specific interaction trace at given offsets |
| `add_constraint(expr)` | Assert that `expr == 0` at every row |
| `add_to_relation(entry)` | Add a logup relation entry (for lookup arguments) |
| `write_logup_frac(frac)` | Directly write a logup fraction numerator/denominator |
| `finalize_logup()` | Must be called at end of `evaluate()` if logup is used |

### 2.3 `Component` (low-level trait)

```rust
pub trait Component {
    fn n_constraints(&self) -> usize;
    fn max_constraint_log_degree_bound(&self) -> u32;
    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>>;
    fn mask_points(&self, point: CirclePoint<SecureField>) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>;
    fn preprocessed_column_indices(&self) -> Vec<usize>;
    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &TreeVec<ColumnVec<Vec<SecureField>>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
        interaction_elements: &InteractionElements,
        lookup_values: &LookupValues,
    );
}
```

You normally **don't** implement this directly. Instead, implement `FrameworkEval` and wrap it in `FrameworkComponent`.

### 2.4 `FrameworkComponent<C: FrameworkEval>`

```rust
pub struct FrameworkComponent<C: FrameworkEval> { /* ... */ }

impl<C: FrameworkEval> FrameworkComponent<C> {
    // Standard constructor. `claimed_sum` is (sum, Option<logup_shift>).
    pub fn new(
        location_allocator: &mut TraceLocationAllocator,
        eval: C,
        claimed_sum: (SecureField, Option<SecureField>),
    ) -> Self;

    // For a disabled (no-op) component
    pub fn disabled(
        location_allocator: &mut TraceLocationAllocator,
        eval: C,
    ) -> Self;
}
```

- Wrapping `FrameworkEval` implementor in `FrameworkComponent` gives you a full `Component` + `ComponentProver<SimdBackend>`.
- `claimed_sum`: For components with no logup, use `(SecureField::zero(), None)`.

---

## 3. Trace & Column Building

### 3.1 Building Trace Columns

Traces are `ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>`.

```rust
use stwo::core::backend::simd::SimdBackend;
use stwo::core::backend::{Col, Column};
use stwo::core::fields::m31::BaseField;
use stwo::core::poly::circle::{CanonicCoset, CircleEvaluation};
use stwo::core::poly::BitReversedOrder;
use stwo::core::ColumnVec;

fn build_trace(log_size: u32) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    let n_rows = 1 << log_size;
    let n_columns = 3; // example: 3 columns

    // Allocate columns
    let mut columns: Vec<Col<SimdBackend, BaseField>> = (0..n_columns)
        .map(|_| Col::<SimdBackend, BaseField>::zeros(n_rows))
        .collect();

    // Fill columns with your data (row-by-row or however you like)
    for row in 0..n_rows {
        let a = BaseField::from(row as u32);
        let b = BaseField::from((row * 2) as u32);
        let c = a + b;
        columns[0].set(row, a);
        columns[1].set(row, b);
        columns[2].set(row, c);
    }

    // Wrap in CircleEvaluation over the canonic coset domain
    let domain = CanonicCoset::new(log_size).circle_domain();
    columns
        .into_iter()
        .map(|col| CircleEvaluation::new(domain, col))
        .collect()
}
```

### 3.2 SIMD-Optimised Trace Building

For better performance with SIMD, work with `PackedBaseField` (packs 16 `M31` values):

```rust
use stwo::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES}; // LOG_N_LANES = 4 (16 lanes)

// Each PackedBaseField holds 16 M31 values.
// Column data is stored as Vec<PackedBaseField>, accessed via col.data[vec_index].
// vec_index = row / 16, lane = row % 16

let mut col = Col::<SimdBackend, BaseField>::zeros(1 << log_size);
col.data[0] = PackedBaseField::from_array(std::array::from_fn(|j| BaseField::from(j as u32)));
```

### 3.3 Committing Traces via `CommitmentSchemeProver`

```rust
use stwo::core::pcs::{CommitmentSchemeProver, PcsConfig, TreeVec};
use stwo::core::backend::simd::SimdBackend;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::channel::Blake2sChannel;
use stwo::core::vcs::blake2_merkle::Blake2sMerkleChannel;

let config = PcsConfig::default();
let log_size: u32 = 8;

// Precompute twiddles for the evaluation domain
let twiddles = SimdBackend::precompute_twiddles(
    CanonicCoset::new(log_size + 1 + config.fri_config.log_blowup_factor)
        .circle_domain()
        .half_coset,
);

let prover_channel = &mut Blake2sChannel::default();
let mut commitment_scheme =
    CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(config, &twiddles);

// 1. Preprocessed trace (empty if none)
let mut tree_builder = commitment_scheme.tree_builder();
tree_builder.extend_evals([]); // no preprocessed columns
tree_builder.commit(prover_channel);

// 2. Original trace
let trace = build_trace(log_size);
let mut tree_builder = commitment_scheme.tree_builder();
tree_builder.extend_evals(trace);
tree_builder.commit(prover_channel);

// 3. (Optional) Interaction trace for logup — commit after drawing lookup elements
// let mut tree_builder = commitment_scheme.tree_builder();
// tree_builder.extend_evals(interaction_trace);
// tree_builder.commit(prover_channel);
```

**Tree ordering matters:**
| Index | Constant | Content |
|-------|----------|---------|
| 0 | `PREPROCESSED_TRACE_IDX` | Preprocessed columns (can be empty) |
| 1 | `ORIGINAL_TRACE_IDX` | Your main execution trace |
| 2+ | `INTERACTION_TRACE_IDX` | Interaction / logup trace columns |

---

## 4. Adding Constraints

### 4.1 Algebraic Equality Constraints

Inside `FrameworkEval::evaluate()`:

```rust
fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
    // Read consecutive columns from the original trace
    let a = eval.next_trace_mask();
    let b = eval.next_trace_mask();
    let c = eval.next_trace_mask();

    // Constraint: c = a + b  ⟹  c - a - b = 0
    eval.add_constraint(c - a.clone() - b.clone());

    // Quadratic constraint: c = a² + b²
    // eval.add_constraint(c - (a.square() + b.square()));

    eval
}
```

Constraints are evaluated point-by-point. The prover divides by the vanishing polynomial to get the quotient; verifier checks at a random point.

### 4.2 Reading Columns at Offsets

```rust
// Read column at offsets [-1, 0, 1] from interaction trace 1
let [prev, curr, next] = eval.next_interaction_mask(1, [-1, 0, 1]);
eval.add_constraint(next - curr.clone() - prev);
```

### 4.3 Logup Lookup Arguments

Logup enforces that a multiset of "used" values equals a multiset of "provided" values via the sum:
$$\sum_i \frac{m_i}{z + \sum_j \alpha^j \cdot x_{i,j}} = 0$$

#### Defining a Relation

```rust
use stwo_constraint_framework::{relation, Relation, RelationEntry};

// The `relation!` macro defines a struct implementing `Relation<F, EF>`.
// Argument: relation name, number of columns.
relation!(MyLookupRelation, 2); // 2-column lookup
```

This creates a struct wrapping `LookupElements<2>`.

#### `LookupElements<N>`

```rust
use stwo_constraint_framework::logup::LookupElements;

pub struct LookupElements<const N: usize> {
    pub z: SecureField,
    pub alpha: SecureField,
    pub alpha_powers: [SecureField; N],
}

impl<const N: usize> LookupElements<N> {
    /// Draw random z, alpha from the Fiat-Shamir channel (after trace commitment).
    pub fn draw(channel: &mut impl Channel) -> Self;

    /// Compute z + sum(alpha^j * values[j]) — the denominator for a lookup entry.
    pub fn combine<F>(&self, values: &[F]) -> EF;

    /// Dummy instance for testing.
    pub fn dummy() -> Self;
}
```

#### `LogupAtRow`

```rust
use stwo_constraint_framework::logup::LogupAtRow;

pub struct LogupAtRow<E: EvalAtRow> {
    pub interaction: usize,
    pub cumsum_shift: SecureField,
    pub fracs: Vec<Fraction<E::EF, E::EF>>,
    pub is_finalized: bool,
    pub log_size: u32,
}

impl<E: EvalAtRow> LogupAtRow<E> {
    /// Create a new LogupAtRow.
    /// `interaction`: the trace interaction index for the logup columns.
    /// `claimed_sum`: the claimed total sum of all fractions.
    /// `log_size`: log2 of the trace length.
    pub fn new(interaction: usize, claimed_sum: SecureField, log_size: u32) -> Self;
}
```

#### Using Logup in `evaluate()`

```rust
fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
    let a = eval.next_trace_mask();
    let b = eval.next_trace_mask();

    // Multiplicity +1: "this row uses the lookup"
    eval.add_to_relation(RelationEntry::new(
        &self.lookup_relation,   // implements Relation
        E::EF::one(),            // multiplicity (+1)
        &[a.clone(), b.clone()], // values to combine
    ));

    // Multiplicity -1: "this row provides the lookup"
    eval.add_to_relation(RelationEntry::new(
        &self.lookup_relation,
        -E::EF::one(),           // multiplicity (-1)
        &[a, b],
    ));

    eval.finalize_logup(); // MUST call at end
    eval
}
```

The `finalize_logup()` call writes the cumulative sum constraint columns into the interaction trace. If you forget it, a `Drop` assertion will fire.

---

## 5. Prove & Verify

### 5.1 Prove (requires `prover` feature)

```rust
use stwo::core::prover::prove;

pub fn prove<B: BackendForChannel<MC>, MC: MerkleChannel>(
    components: &[&dyn ComponentProver<B>],
    channel: &mut MC::C,
    commitment_scheme: CommitmentSchemeProver<'_, B, MC>,
) -> Result<StarkProof<MC::H>, ProvingError>;
```

| Parameter | Description |
|-----------|-------------|
| `B` | Backend type, typically `SimdBackend` |
| `MC` | Merkle channel, e.g. `Blake2sMerkleChannel` or `Poseidon252MerkleChannel` |
| `components` | Slice of references to your `FrameworkComponent`s (they impl `ComponentProver`) |
| `channel` | Fiat-Shamir channel (same instance used during trace commitment) |
| `commitment_scheme` | The `CommitmentSchemeProver` with all traces already committed |

Returns `StarkProof<MC::H>` on success.

### 5.2 Verify

```rust
use stwo::core::prover::verify; // in the prover crate
// or (verifier-only, from stwo crate):
use stwo::core::verifier::verify;

pub fn verify<MC: MerkleChannel>(
    components: &[&dyn Component],
    channel: &mut MC::C,
    commitment_scheme: &mut CommitmentSchemeVerifier<MC>,
    proof: StarkProof<MC::H>,
) -> Result<(), VerificationError>;
```

The verifier reconstructs the commitment scheme from proof commitments:

```rust
use stwo::core::pcs::CommitmentSchemeVerifier;

let verifier_channel = &mut Blake2sChannel::default(); // fresh channel
let mut commitment_scheme = CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);

// Replay commitments from the proof
let sizes = component.trace_log_degree_bounds();
commitment_scheme.commit(proof.commitments[0], &sizes[0], verifier_channel); // preprocessed
commitment_scheme.commit(proof.commitments[1], &sizes[1], verifier_channel); // original trace
// commitment_scheme.commit(proof.commitments[2], &sizes[2], verifier_channel); // interaction trace (if logup)

verify::<Blake2sMerkleChannel>(&[&component], verifier_channel, &mut commitment_scheme, proof)?;
```

### 5.3 `PcsConfig`

```rust
use stwo::core::pcs::PcsConfig;

let config = PcsConfig::default();
// Default FRI config has log_blowup_factor, n_queries, etc.
// For testing, default is fine.
```

### 5.4 Error Types

```rust
pub enum ProvingError {
    ConstraintsNotSatisfied,
}

pub enum VerificationError {
    InvalidStructure(String),
    InvalidLookup(String),
    Merkle(MerkleVerificationError),
    OodsNotMatching,
    Fri(FriVerificationError),
    ProofOfWork,
}
```

---

## 6. Minimal End-to-End Example

This is a simplified version of the **wide Fibonacci** example from the stwo repo (`crates/prover/src/examples/wide_fibonacci/mod.rs`).

### 6.1 Define the AIR

```rust
use stwo_constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval};
use stwo::core::fields::FieldExpOps;

// Type alias for convenience
pub type FibComponent = FrameworkComponent<FibEval>;

#[derive(Clone)]
pub struct FibEval {
    pub log_n_rows: u32,
}

impl FrameworkEval for FibEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        // Degree-2 constraints (squaring) → degree bound = 2 * trace_size
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        // 3 columns per row: a, b, c
        let a = eval.next_trace_mask();
        let b = eval.next_trace_mask();
        let c = eval.next_trace_mask();

        // Constraint: c = a² + b²
        eval.add_constraint(c - (a.square() + b.square()));

        eval
    }
}
```

### 6.2 Generate Trace

```rust
use itertools::Itertools;
use stwo::core::backend::simd::SimdBackend;
use stwo::core::backend::{Col, Column};
use stwo::core::fields::m31::BaseField;
use stwo::core::poly::circle::{CanonicCoset, CircleEvaluation};
use stwo::core::poly::BitReversedOrder;
use stwo::core::ColumnVec;

fn generate_trace(
    log_size: u32,
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    let n_rows = 1 << log_size;

    let mut col_a = Col::<SimdBackend, BaseField>::zeros(n_rows);
    let mut col_b = Col::<SimdBackend, BaseField>::zeros(n_rows);
    let mut col_c = Col::<SimdBackend, BaseField>::zeros(n_rows);

    let mut a = BaseField::from(1u32);
    let mut b = BaseField::from(1u32);
    for row in 0..n_rows {
        let c = a * a + b * b; // a² + b² (mod P)
        col_a.set(row, a);
        col_b.set(row, b);
        col_c.set(row, c);
        a = b;
        b = c;
    }

    let domain = CanonicCoset::new(log_size).circle_domain();
    vec![col_a, col_b, col_c]
        .into_iter()
        .map(|col| CircleEvaluation::new(domain, col))
        .collect()
}
```

### 6.3 Prove & Verify

```rust
use stwo::core::backend::simd::SimdBackend;
use stwo::core::channel::Blake2sChannel;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::{CommitmentSchemeProver, CommitmentSchemeVerifier, PcsConfig};
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::prover::{prove, verify};
use stwo::core::vcs::blake2_merkle::Blake2sMerkleChannel;
use stwo_constraint_framework::TraceLocationAllocator;

fn prove_and_verify() {
    let log_size: u32 = 8; // 256 rows
    let config = PcsConfig::default();

    // Precompute twiddles
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_size + 1 + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );

    // --- PROVER ---
    let prover_channel = &mut Blake2sChannel::default();
    let mut commitment_scheme =
        CommitmentSchemeProver::<SimdBackend, Blake2sMerkleChannel>::new(config, &twiddles);

    // Commit preprocessed trace (empty)
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals([]);
    tree_builder.commit(prover_channel);

    // Commit execution trace
    let trace = generate_trace(log_size);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace);
    tree_builder.commit(prover_channel);

    // Build component
    let component = FibComponent::new(
        &mut TraceLocationAllocator::default(),
        FibEval { log_n_rows: log_size },
        (SecureField::zero(), None), // no logup → zero claimed sum
    );

    // Generate proof
    let proof = prove::<SimdBackend, Blake2sMerkleChannel>(
        &[&component],
        prover_channel,
        commitment_scheme,
    )
    .unwrap();

    // --- VERIFIER ---
    let verifier_channel = &mut Blake2sChannel::default();
    let mut verifier_commitment_scheme =
        CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);

    // Replay commitments
    let sizes = component.trace_log_degree_bounds();
    verifier_commitment_scheme.commit(proof.commitments[0], &sizes[0], verifier_channel);
    verifier_commitment_scheme.commit(proof.commitments[1], &sizes[1], verifier_channel);

    // Verify
    verify::<Blake2sMerkleChannel>(
        &[&component],
        verifier_channel,
        &mut verifier_commitment_scheme,
        proof,
    )
    .unwrap();

    println!("Proof verified successfully!");
}
```

---

## Quick Reference Cheat Sheet

```
┌──────────────────────────────────────────────────────────┐
│                    YOUR CODE                             │
│                                                          │
│  1. impl FrameworkEval for MyEval { ... }                │
│     - log_size()                                         │
│     - max_constraint_log_degree_bound()                  │
│     - evaluate<E: EvalAtRow>(eval) → E                   │
│       ├─ eval.next_trace_mask()        ← read column     │
│       ├─ eval.add_constraint(expr)     ← assert expr=0   │
│       ├─ eval.add_to_relation(entry)   ← logup entry     │
│       └─ eval.finalize_logup()         ← end logup       │
│                                                          │
│  2. type MyComponent = FrameworkComponent<MyEval>;        │
│                                                          │
│  3. Build trace: Vec<CircleEvaluation<SimdBackend, ...>> │
│                                                          │
│  4. CommitmentSchemeProver → tree_builder → commit(chan)  │
│     ├─ tree 0: preprocessed (empty or precomputed)       │
│     ├─ tree 1: original execution trace                  │
│     └─ tree 2: interaction trace (logup, optional)       │
│                                                          │
│  5. prove(&[&component], channel, commitment_scheme)     │
│     → StarkProof                                         │
│                                                          │
│  6. CommitmentSchemeVerifier + verify(...)                │
└──────────────────────────────────────────────────────────┘
```

## Important Notes

- **`num_traits`**: Import `Zero` and `One` from `num_traits` for `M31::zero()`, `M31::one()`.
- **Column read order**: `next_trace_mask()` calls must match the exact column order committed in the trace.
- **Degree bounds**: If constraint degree is $d$ and trace size is $2^n$, set `max_constraint_log_degree_bound()` to $n + \lceil\log_2 d\rceil$. For degree-2 (quadratic) constraints, that's $n + 1$.
- **`SecureField::zero()` for no-logup**: When creating a `FrameworkComponent` for an AIR with no lookup arguments, pass `(SecureField::zero(), None)` as the claimed sum.
- **Blake2s vs Poseidon252**: `Blake2sMerkleChannel` for standard CPU/x86. `Poseidon252MerkleChannel` for Poseidon-based (e.g. Cairo-compatible).
- **Serialization**: `StarkProof` implements `Serialize` / `Deserialize` (serde) and can be serialized with bincode, JSON, etc.
