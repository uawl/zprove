# Plonky3 Circle STARK Library — API Reference for Custom AIR Components

> Based on `p3-*` crates at version **0.4.2**. Sourced from docs.rs and the official Plonky3 GitHub test suite.

## Table of Contents

1. [M31 Field Type & Arithmetic](#1-m31-field-type--arithmetic)
2. [Air Trait & AirBuilder](#2-air-trait--airbuilder)
3. [Building a Trace Matrix](#3-building-a-trace-matrix)
4. [Prove & Verify (p3-uni-stark)](#4-prove--verify-p3-uni-stark)
5. [Circle STARK Config with M31](#5-circle-stark-config-with-m31)
6. [LogUp Lookup Arguments (p3-lookup)](#6-logup-lookup-arguments-p3-lookup)
7. [Minimal End-to-End Example (Fibonacci)](#7-minimal-end-to-end-example-fibonacci)
8. [Cargo Dependencies & Feature Flags](#8-cargo-dependencies--feature-flags)

---

## 1. M31 Field Type & Arithmetic

**Crate:** `p3-mersenne-31` (module `p3_mersenne_31`)

The Mersenne-31 prime field $F_p$ where $p = 2^{31} - 1 = 2\,147\,483\,647$.

### Type

```rust
use p3_mersenne_31::Mersenne31;

// Mersenne31 is a Copy + Clone + Debug + Send + Sync type
// It implements: Field, PrimeField32, PrimeField64, PrimeCharacteristicRing, ComplexExtendable
```

### Construction

```rust
// From u32 (auto-reduced mod p):
let a = Mersenne31::new(42);

// Checked (returns None if value > p):
let b = Mersenne31::new_checked(100);

// From various integer types (via PrimeCharacteristicRing trait):
let c = Mersenne31::from_u64(123);
let d = Mersenne31::from_u32(456);
let e = Mersenne31::from_usize(789);

// Array construction:
let arr = Mersenne31::new_array([1, 2, 3, 4]);

// Constants:
let zero = Mersenne31::ZERO;
let one  = Mersenne31::ONE;
let neg1 = Mersenne31::NEG_ONE;

// Allocate a zero vector:
let zeros: Vec<Mersenne31> = Mersenne31::zero_vec(1024);
```

### Arithmetic

Standard `+`, `-`, `*`, `+=`, `-=`, `*=`, `Neg` are all implemented.

```rust
let x = Mersenne31::new(10);
let y = Mersenne31::new(20);

let sum  = x + y;
let diff = x - y;
let prod = x * y;
let neg  = -x;

// Inverse (returns Option<Self>):
let inv = x.try_inverse();       // Some(...) if x != 0
let inv = x.inverse();           // panics if x == 0

// Exponentiation:
let x_pow = x.exp_u64(17);

// Boolean check:
assert!(x.is_zero() == false);
assert!(Mersenne31::ZERO.is_zero());
```

### Field trait details

| Trait | Key methods |
|---|---|
| `PrimeCharacteristicRing` | `ZERO`, `ONE`, `NEG_ONE`, `from_u8/u16/u32/u64/u128/usize`, `from_i32/i64`, `exp_u64`, `powers()`, `dot_product`, `zero_vec` |
| `Field` | `GENERATOR`, `try_inverse()`, `inverse()`, `is_zero()`, `order()`, `bits()` |
| `PrimeField32` | `ORDER_U32 = 2_147_483_647`, `as_canonical_u32()`, `to_unique_u32()` |
| `PrimeField64` | `ORDER_U64 = 2_147_483_647`, `as_canonical_u64()`, `to_unique_u64()` |
| `ComplexExtendable` | `CIRCLE_TWO_ADICITY = 31`, `circle_two_adic_generator(bits)` — enables Circle STARK |

### Extension field for challenges

Mersenne31 supports degree-3 binomial extension:

```rust
use p3_field::extension::BinomialExtensionField;

type Challenge = BinomialExtensionField<Mersenne31, 3>;
```

---

## 2. Air Trait & AirBuilder

**Crate:** `p3-air` (module `p3_air`)

### Trait hierarchy

```
BaseAir<F>                 — declares the trace width
  └─ Air<AB: AirBuilder>   — defines the constraints via eval()
```

### `BaseAir<F>` trait

```rust
pub trait BaseAir<F>: Sync {
    /// Number of columns in this AIR.
    fn width(&self) -> usize;

    /// Optional preprocessed trace matrix.
    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> { None }
}
```

### `BaseAirWithPublicValues<F>` trait

Extends `BaseAir<F>` with public values support:

```rust
pub trait BaseAirWithPublicValues<F>: BaseAir<F> {
    fn num_public_values(&self) -> usize { 0 }
}
```

### `Air<AB>` trait

```rust
pub trait Air<AB: AirBuilder>: BaseAir<AB::F> {
    /// Evaluate all AIR constraints using the provided builder.
    fn eval(&self, builder: &mut AB);
}
```

### `AirBuilder` trait (key methods)

```rust
pub trait AirBuilder: Sized {
    type F: PrimeCharacteristicRing + Sync;
    type Expr: Algebra<Self::F> + Algebra<Self::Var>;
    type Var: Into<Self::Expr> + Clone + Send + Sync + Add/Sub/Mul ops...;
    type M: Matrix<Self::Var>;

    // Trace access
    fn main(&self) -> Self::M;

    // Row selectors (return expressions that are 1 on specific rows, 0 elsewhere)
    fn is_first_row(&self) -> Self::Expr;
    fn is_last_row(&self) -> Self::Expr;
    fn is_transition(&self) -> Self::Expr;       // 1 except last row
    fn is_transition_window(&self, size: usize) -> Self::Expr;

    // Constraint assertion
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I);
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, array: [I; N]);
    fn assert_one<I: Into<Self::Expr>>(&mut self, x: I);
    fn assert_eq<I1: Into<Self::Expr>, I2: Into<Self::Expr>>(&mut self, x: I1, y: I2);
    fn assert_bool<I: Into<Self::Expr>>(&mut self, x: I);
    fn assert_bools<const N: usize, I: Into<Self::Expr>>(&mut self, array: [I; N]);

    // Conditional builders — returns FilteredAirBuilder
    fn when<I: Into<Self::Expr>>(&mut self, condition: I) -> FilteredAirBuilder<'_, Self>;
    fn when_first_row(&mut self) -> FilteredAirBuilder<'_, Self>;
    fn when_last_row(&mut self) -> FilteredAirBuilder<'_, Self>;
    fn when_transition(&mut self) -> FilteredAirBuilder<'_, Self>;
    fn when_ne<I1, I2>(&mut self, x: I1, y: I2) -> FilteredAirBuilder<'_, Self>;
}
```

### `AirBuilderWithPublicValues` trait

```rust
pub trait AirBuilderWithPublicValues: AirBuilder {
    fn public_values(&self) -> &[Self::F];
}
```

### Reading trace rows

```rust
fn eval(&self, builder: &mut AB) {
    let main = builder.main();              // returns AB::M (a Matrix)
    let local = main.row_slice(0);          // current row
    let next  = main.row_slice(1);          // next row

    // Access individual columns:
    let col_0_local: AB::Var = local[0].clone();
}
```

---

## 3. Building a Trace Matrix

**Crate:** `p3-matrix` (module `p3_matrix`)

### `RowMajorMatrix<F>`

A type alias for `DenseMatrix<F, Vec<F>>` — the standard trace format.

```rust
use p3_matrix::dense::RowMajorMatrix;
use p3_mersenne_31::Mersenne31;

type F = Mersenne31;

fn build_trace(n: usize) -> RowMajorMatrix<F> {
    let num_cols = 2;
    // Allocate: n rows × num_cols columns, initialized to zero
    let mut trace = RowMajorMatrix::new(F::zero_vec(n * num_cols), num_cols);

    // Fill rows: trace.values is a flat Vec<F> in row-major order
    // Row i, column j is at index: i * num_cols + j
    trace.values[0 * num_cols + 0] = F::from_u64(0); // row 0, col 0
    trace.values[0 * num_cols + 1] = F::from_u64(1); // row 0, col 1

    for i in 1..n {
        let a = trace.values[(i - 1) * num_cols + 0];
        let b = trace.values[(i - 1) * num_cols + 1];
        trace.values[i * num_cols + 0] = b;
        trace.values[i * num_cols + 1] = a + b;
    }

    trace
}
```

### Using `#[repr(C)]` row structs (zero-copy pattern from Plonky3 tests)

```rust
use core::borrow::Borrow;

const NUM_COLS: usize = 2;

#[repr(C)]
pub struct MyRow<F> {
    pub left: F,
    pub right: F,
}

impl<F> Borrow<MyRow<F>> for [F] {
    fn borrow(&self) -> &MyRow<F> {
        debug_assert_eq!(self.len(), NUM_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<MyRow<F>>() };
        debug_assert!(prefix.is_empty());
        debug_assert!(suffix.is_empty());
        &shorts[0]
    }
}

// In eval(), cast row slices to your struct:
let local: &MyRow<AB::Var> = (*main.row_slice(0).unwrap()).borrow();
let next:  &MyRow<AB::Var> = (*main.row_slice(1).unwrap()).borrow();
```

### Key `Matrix<T>` trait methods

| Method | Description |
|---|---|
| `height()` | Number of rows |
| `width()` | Number of columns |
| `row_slice(r)` | Returns the r-th row as a slice (`Result`) |

### Trace size requirement

The number of rows **must be a power of two** for both univariate and Circle STARKs.

---

## 4. Prove & Verify (p3-uni-stark)

**Crate:** `p3-uni-stark` (module `p3_uni_stark`)

The prove/verify functions are **generic** over the PCS (polynomial commitment scheme). They work with both `TwoAdicFriPcs` (standard FRI) and `CirclePcs` (Circle STARK).

### `prove()`

```rust
pub fn prove<SC, A>(
    config: &SC,
    air: &A,
    trace: RowMajorMatrix<Val<SC>>,
    public_values: &[Val<SC>],
) -> Proof<SC>
where
    SC: StarkGenericConfig,
    A: Air<SymbolicAirBuilder<Val<SC>>>
     + for<'a> Air<ProverConstraintFolder<'a, SC>>
     + for<'a> Air<DebugConstraintBuilder<'a, Val<SC>>>,
```

### `verify()`

```rust
pub fn verify<SC, A>(
    config: &SC,
    air: &A,
    proof: &Proof<SC>,
    public_values: &[Val<SC>],
) -> Result<(), VerificationError<PcsError<SC>>>
where
    SC: StarkGenericConfig,
    A: Air<SymbolicAirBuilder<Val<SC>>>
     + for<'a> Air<VerifierConstraintFolder<'a, SC>>,
```

### `StarkConfig`

```rust
use p3_uni_stark::StarkConfig;

// StarkConfig<Pcs, Challenge, Challenger>
let config = StarkConfig::new(pcs, challenger);
```

### Usage pattern

```rust
use p3_uni_stark::{prove, verify, StarkConfig};

let config = make_config();         // see §5 for Circle config
let air = MyAir {};
let trace = build_trace(1 << 3);    // 8 rows
let pis = vec![/* public values */];

let proof = prove(&config, &air, trace, &pis);
verify(&config, &air, &proof, &pis).expect("verification failed");
```

### With preprocessed trace

```rust
use p3_uni_stark::{prove_with_preprocessed, verify_with_preprocessed, setup_preprocessed};

let (ppd, pvk) = setup_preprocessed(&config, &air, 1 << 3);  // degree
let proof = prove_with_preprocessed(&config, &air, trace, &pis, &ppd);
verify_with_preprocessed(&config, &air, &proof, &pis, &pvk)?;
```

---

## 5. Circle STARK Config with M31

This is the canonical setup from Plonky3's own test suite (`uni-stark/tests/fib_air.rs`). It uses **Keccak** for hashing and a **degree-3 extension** over Mersenne31.

### Full type aliases

```rust
use p3_mersenne_31::Mersenne31;
use p3_field::extension::BinomialExtensionField;
use p3_circle::CirclePcs;
use p3_commit::ExtensionMmcs;
use p3_keccak::Keccak256Hash;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_fri::FriParameters;
use p3_uni_stark::StarkConfig;

type Val = Mersenne31;
type Challenge = BinomialExtensionField<Val, 3>;

type ByteHash    = Keccak256Hash;
type FieldHash   = SerializingHasher<ByteHash>;
type Compress    = CompressionFunctionFromHasher<ByteHash, 2, 32>;
type ValMmcs     = MerkleTreeMmcs<Val, u8, FieldHash, Compress, 32>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger  = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;

type Pcs = CirclePcs<Val, ValMmcs, ChallengeMmcs>;
type CircleConfig = StarkConfig<Pcs, Challenge, Challenger>;
```

### Configuration constructor

```rust
fn make_circle_config() -> CircleConfig {
    let byte_hash = Keccak256Hash {};
    let field_hash = SerializingHasher::new(byte_hash);
    let compress = CompressionFunctionFromHasher::new(byte_hash);
    let val_mmcs = MerkleTreeMmcs::new(field_hash, compress, 0);
    let challenge_mmcs = ExtensionMmcs::new(val_mmcs.clone());

    let fri_params = FriParameters {
        log_blowup: 1,
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 40,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
    };

    let pcs = CirclePcs {
        mmcs: val_mmcs,
        fri_params,
        _phantom: core::marker::PhantomData,
    };

    let challenger = SerializingChallenger32::from_hasher(vec![], byte_hash);
    StarkConfig::new(pcs, challenger)
}
```

### Key FRI parameter choices

| Parameter | Typical Value | Meaning |
|---|---|---|
| `log_blowup` | 1–2 | Rate = $2^{-\text{log\_blowup}}$. Higher = more security, larger proof |
| `log_final_poly_len` | 0 | Log of the final polynomial length in FRI |
| `max_log_arity` | 1 | Max folding arity |
| `num_queries` | 28–100 | Number of FRI query rounds |
| `commit_proof_of_work_bits` | 0–16 | PoW bits during commit phase |
| `query_proof_of_work_bits` | 0–16 | PoW bits during query phase |

---

## 6. LogUp Lookup Arguments (p3-lookup)

**Crate:** `p3-lookup` (module `p3_lookup`)

LogUp transforms standard lookup equations using logarithmic derivatives:

$$\sum_i \frac{m_i}{\alpha - a_i} = \sum_j \frac{m'_j}{\alpha - b_j}$$

where $\alpha$ is a random challenge and $m_i, m'_j$ are multiplicities.

### Modules

| Module | Contents |
|---|---|
| `p3_lookup::logup` | `LogUpGadget` — core gadget implementing lookup via log derivatives |
| `p3_lookup::lookup_traits` | Traits and types for defining lookups |
| `p3_lookup::folder` | Constraint folder integration |

### Key types in `lookup_traits`

```rust
// Direction of data flow in the lookup
pub enum Direction { Send, Receive }

// Whether lookup is local to one AIR or global across AIRs
pub enum Kind { Local, Global }

// Lookup definition — shared between prover and verifier
pub struct Lookup { /* ... */ }

// Builds the permutation trace for the lookup argument
pub struct LookupTraceBuilder { /* ... */ }

// Data for global lookup arguments in multi-STARK proofs
pub struct LookupData { /* ... */ }

// Input tuple type alias
pub type LookupInput = (Direction, Kind, Vec<SymbolicExpression<F>>, SymbolicExpression<F>);
```

### Key traits

```rust
/// AIR that handles lookup arguments
pub trait AirLookupHandler {
    // Defines the lookup interactions
}

/// Lookup gadget trait (LogUpGadget implements this)
pub trait LookupGadget {
    // Generates and verifies the lookup argument
}
```

### Wrapper for AIRs without lookups

```rust
use p3_lookup::lookup_traits::AirNoLookup;

let wrapped = AirNoLookup::new(my_air);
```

### PermutationAirBuilder (from p3-air)

For AIRs that use lookups, the builder must implement:

```rust
pub trait PermutationAirBuilder: AirBuilder {
    // Provides access to permutation (lookup) trace columns
}
```

---

## 7. Minimal End-to-End Example (Fibonacci)

This is the **complete, working** example from the Plonky3 test suite demonstrating a Circle STARK proof over M31. It proves that `fib(n) == x` where `fib(0)=a, fib(1)=b`.

### AIR Definition

```rust
use core::borrow::Borrow;
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_matrix::Matrix;
use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

const NUM_FIBONACCI_COLS: usize = 2;

#[repr(C)]
pub struct FibonacciRow<F> {
    pub left: F,
    pub right: F,
}

impl<F> Borrow<FibonacciRow<F>> for [F] {
    fn borrow(&self) -> &FibonacciRow<F> {
        debug_assert_eq!(self.len(), NUM_FIBONACCI_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<FibonacciRow<F>>() };
        debug_assert!(prefix.is_empty());
        debug_assert!(suffix.is_empty());
        &shorts[0]
    }
}

pub struct FibonacciAir {}

impl<F> BaseAir<F> for FibonacciAir {
    fn width(&self) -> usize {
        NUM_FIBONACCI_COLS
    }
}

impl<AB: AirBuilderWithPublicValues> Air<AB> for FibonacciAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let pis = builder.public_values();
        let a = pis[0];   // fib(0)
        let b = pis[1];   // fib(1)
        let x = pis[2];   // expected result

        let (local, next) = (
            main.row_slice(0).expect("Matrix is empty?"),
            main.row_slice(1).expect("Matrix only has 1 row?"),
        );
        let local: &FibonacciRow<AB::Var> = (*local).borrow();
        let next:  &FibonacciRow<AB::Var> = (*next).borrow();

        // First row: left == a, right == b
        let mut when_first_row = builder.when_first_row();
        when_first_row.assert_eq(local.left.clone(), a);
        when_first_row.assert_eq(local.right.clone(), b);

        // Transition: next.left = local.right, next.right = local.left + local.right
        let mut when_transition = builder.when_transition();
        when_transition.assert_eq(local.right.clone(), next.left.clone());
        when_transition.assert_eq(
            local.left.clone() + local.right.clone(),
            next.right.clone(),
        );

        // Last row: right == x
        builder.when_last_row().assert_eq(local.right.clone(), x);
    }
}
```

### Trace Generation

```rust
pub fn generate_fib_trace<F: PrimeField64>(a: u64, b: u64, n: usize) -> RowMajorMatrix<F> {
    assert!(n.is_power_of_two());

    let mut trace = RowMajorMatrix::new(
        F::zero_vec(n * NUM_FIBONACCI_COLS),
        NUM_FIBONACCI_COLS,
    );

    let (prefix, rows, suffix) = unsafe {
        trace.values.align_to_mut::<FibonacciRow<F>>()
    };
    assert!(prefix.is_empty());
    assert!(suffix.is_empty());
    assert_eq!(rows.len(), n);

    rows[0] = FibonacciRow { left: F::from_u64(a), right: F::from_u64(b) };

    for i in 1..n {
        rows[i].left  = rows[i - 1].right;
        rows[i].right = rows[i - 1].left + rows[i - 1].right;
    }

    trace
}
```

### Circle STARK Prove & Verify

```rust
use p3_mersenne_31::Mersenne31;
use p3_field::extension::BinomialExtensionField;
use p3_circle::CirclePcs;
use p3_commit::ExtensionMmcs;
use p3_keccak::Keccak256Hash;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_fri::FriParameters;
use p3_uni_stark::{StarkConfig, prove, verify};

type Val = Mersenne31;
type Challenge = BinomialExtensionField<Val, 3>;
type ByteHash = Keccak256Hash;
type FieldHash = SerializingHasher<ByteHash>;
type Compress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
type ValMmcs = MerkleTreeMmcs<Val, u8, FieldHash, Compress, 32>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;
type Pcs = CirclePcs<Val, ValMmcs, ChallengeMmcs>;
type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;

fn main() {
    // Build config
    let byte_hash = Keccak256Hash {};
    let field_hash = SerializingHasher::new(byte_hash);
    let compress = CompressionFunctionFromHasher::new(byte_hash);
    let val_mmcs = MerkleTreeMmcs::new(field_hash, compress, 0);
    let challenge_mmcs = ExtensionMmcs::new(val_mmcs.clone());

    let fri_params = FriParameters {
        log_blowup: 1,
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 40,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
    };

    let pcs = CirclePcs {
        mmcs: val_mmcs,
        fri_params,
        _phantom: core::marker::PhantomData,
    };
    let challenger = SerializingChallenger32::from_hasher(vec![], byte_hash);
    let config = StarkConfig::new(pcs, challenger);

    // Generate trace: fib(0)=0, fib(1)=1, 8 rows → fib(7)=21
    let trace = generate_fib_trace::<Val>(0, 1, 1 << 3);
    let pis = vec![
        Val::from_u64(0),     // a
        Val::from_u64(1),     // b
        Val::from_u64(21),    // expected fib(7)
    ];

    // Prove
    let proof = prove(&config, &FibonacciAir {}, trace, &pis);

    // Verify
    verify(&config, &FibonacciAir {}, &proof, &pis)
        .expect("verification failed");

    println!("Circle STARK proof verified!");
}
```

---

## 8. Cargo Dependencies & Feature Flags

### All required crates (version 0.4.2)

```toml
[dependencies]
# Core field
p3-mersenne-31 = "0.4.2"
p3-field        = "0.4.2"

# AIR definitions
p3-air          = "0.4.2"

# Matrix / trace
p3-matrix       = "0.4.2"

# Circle STARK PCS
p3-circle       = "0.4.2"

# STARK prove/verify framework
p3-uni-stark    = "0.4.2"

# Lookup arguments
p3-lookup       = "0.4.2"

# FRI protocol
p3-fri          = "0.4.2"

# Commitment infrastructure
p3-commit       = "0.4.2"

# Hashing (choose one or more)
p3-keccak       = "0.4.2"          # Keccak-based (good for non-recursive, larger proofs)
# p3-poseidon2  = "0.4.2"         # Poseidon2-based (good for recursion)

# Symmetric crypto utilities
p3-symmetric    = "0.4.2"

# Merkle tree MMCS
p3-merkle-tree  = "0.4.2"

# Challenger
p3-challenger   = "0.4.2"
```

### Feature flags

Most p3 crates have minimal or no feature flags. Key ones:

- **`p3-mersenne-31`**: no required features
- **`p3-uni-stark`**: no required features  
- **`p3-circle`**: no required features
- **`p3-keccak`**: no required features

For SIMD/AVX optimizations, the crates auto-detect CPU features at compile time. Use `RUSTFLAGS="-C target-cpu=native"` for best performance.

### Comparison with Stwo

| Aspect | Plonky3 (p3) | Stwo |
|---|---|---|
| Field | `Mersenne31` (same M31) | `M31` / `QM31` |
| AIR definition | `impl Air<AB> for MyAir` | `impl FrameworkEval for MyAir` |
| Constraint style | `builder.assert_eq(a, b)` | `eval.add_constraint(a - b)` |
| Trace building | `RowMajorMatrix<F>` flat vec | Column-oriented `CircleEvaluation` |
| Prove API | `prove(&config, &air, trace, &pis)` | `stwo_prover::prove(components, channel)` |
| Lookup | `p3-lookup` (LogUp) | Built-in LogUp via `eval.add_to_relation` |
| Extension field | `BinomialExtensionField<M31, 3>` | `QM31` (degree-4 extension) |

---

## Quick Reference: Constraint Patterns

```rust
// Inside impl Air<AB> for MyAir:
fn eval(&self, builder: &mut AB) {
    let main = builder.main();
    let local = main.row_slice(0).unwrap();
    let next  = main.row_slice(1).unwrap();

    // ── Boundary constraint (first row) ──
    builder.when_first_row().assert_eq(local[0].clone(), AB::F::from_u64(42));

    // ── Boundary constraint (last row) ──
    builder.when_last_row().assert_eq(local[0].clone(), AB::F::from_u64(100));

    // ── Transition constraint ──
    builder.when_transition().assert_eq(
        next[0].clone(),
        local[0].clone() + local[1].clone(),
    );

    // ── Boolean constraint ──
    builder.assert_bool(local[2].clone());

    // ── Conditional constraint ──
    builder.when(local[2].clone()).assert_eq(
        local[0].clone(),
        local[1].clone(),
    );

    // ── General zero constraint ──
    let expr = local[0].clone() * local[1].clone() - next[0].clone();
    builder.assert_zero(expr);
}
```
