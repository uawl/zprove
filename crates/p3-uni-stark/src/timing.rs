//! Per-phase timing counters for the STARK prover.
//!
//! Enabled only when the `timing` Cargo feature is active. The counters are
//! global `AtomicU64` nanosecond accumulators so that parallel provers running
//! on different OS threads all contribute to the same totals.
//!
//! # Usage
//! ```rust,ignore
//! p3_uni_stark::timing::reset();
//! prove_with_preprocessed_hinted(...);
//! let snap = p3_uni_stark::timing::snapshot();
//! println!("quotient eval: {} ns", snap.quotient_eval_ns);
//! ```

use core::sync::atomic::{AtomicU64, Ordering};

// ── Global nanosecond counters ────────────────────────────────────────────────

/// Nanoseconds spent committing the main execution trace (Merkle tree build).
pub static TRACE_COMMIT_NS: AtomicU64 = AtomicU64::new(0);

/// Nanoseconds spent evaluating AIR constraints + LogUp sums (quotient poly eval).
pub static QUOTIENT_EVAL_NS: AtomicU64 = AtomicU64::new(0);

/// Nanoseconds spent committing the quotient polynomial chunks (Merkle tree build).
pub static QUOTIENT_COMMIT_NS: AtomicU64 = AtomicU64::new(0);

/// Nanoseconds spent in the FRI opening phase (queries + opening proof).
pub static OPEN_NS: AtomicU64 = AtomicU64::new(0);

/// Number of [`prove_with_preprocessed_hinted`] calls recorded since last reset.
pub static PROOF_COUNT: AtomicU64 = AtomicU64::new(0);

// ── Snapshot struct ───────────────────────────────────────────────────────────

/// A snapshot of the accumulated phase timings.
#[derive(Clone, Copy, Debug, Default)]
pub struct Snapshot {
    pub trace_commit_ns: u64,
    pub quotient_eval_ns: u64,
    pub quotient_commit_ns: u64,
    pub open_ns: u64,
    /// Number of sub-proofs included in the totals above.
    pub proof_count: u64,
}

impl Snapshot {
    /// Total ns across all four phases.
    pub fn total_ns(&self) -> u64 {
        self.trace_commit_ns
            + self.quotient_eval_ns
            + self.quotient_commit_ns
            + self.open_ns
    }

    /// Nanoseconds in the "LogUp / constraint eval" bucket (= quotient eval only).
    pub fn logup_eval_ns(&self) -> u64 {
        self.quotient_eval_ns
    }

    /// Nanoseconds in the "FRI + Commitment" bucket (trace commit + quotient commit + FRI open).
    pub fn fri_commit_ns(&self) -> u64 {
        self.trace_commit_ns + self.quotient_commit_ns + self.open_ns
    }
}

// ── API ───────────────────────────────────────────────────────────────────────

/// Zero all counters.
pub fn reset() {
    TRACE_COMMIT_NS.store(0, Ordering::Relaxed);
    QUOTIENT_EVAL_NS.store(0, Ordering::Relaxed);
    QUOTIENT_COMMIT_NS.store(0, Ordering::Relaxed);
    OPEN_NS.store(0, Ordering::Relaxed);
    PROOF_COUNT.store(0, Ordering::Relaxed);
}

/// Read a consistent snapshot of the counters (best-effort; not transactional).
pub fn snapshot() -> Snapshot {
    Snapshot {
        trace_commit_ns: TRACE_COMMIT_NS.load(Ordering::Relaxed),
        quotient_eval_ns: QUOTIENT_EVAL_NS.load(Ordering::Relaxed),
        quotient_commit_ns: QUOTIENT_COMMIT_NS.load(Ordering::Relaxed),
        open_ns: OPEN_NS.load(Ordering::Relaxed),
        proof_count: PROOF_COUNT.load(Ordering::Relaxed),
    }
}

// ── Internal helper used by the prover ────────────────────────────────────────

/// Time the closure `f` and add the elapsed nanoseconds to `counter`.
#[inline]
pub(crate) fn record<T, F: FnOnce() -> T>(counter: &AtomicU64, f: F) -> T {
    let t0 = std::time::Instant::now();
    let result = f();
    counter.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    result
}
