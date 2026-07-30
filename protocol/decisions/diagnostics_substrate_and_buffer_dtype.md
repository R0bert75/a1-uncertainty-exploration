# Decision memo — diagnostics substrate + replay storage dtype

**Status: options + recommendation, awaiting decision. Not a protocol amendment.**
Neither item changes a frozen value. Item 2 is shown below to be *behaviourally inert*
(bit-exact), so it is not even a Class-1 tuning choice.

---

## 1. "Disagreement logging" — the audit's framing was wrong

`audits/code_completeness_session3.md` listed this as *unscheduled, needs a scope decision*.
Re-reading the spec, both halves of that are incorrect:

* **It is scheduled.** Implementation order item 5 is "switchboard + backbone-tuning pass +
  **disagreement logging**". The switchboard is done; this is the current item.
* **It is not unscoped.** Spec §3.3 / prereg "Uncertainty diagnostics" freeze **all nine**
  diagnostics mathematically — including tie-breaking (lowest action index), the exact
  quantile method (`numpy.quantile(..., method="linear")`), the statistic per diagnostic
  (Spearman ρ, rank-biserial r = 2(AUC−0.5), OLS slope), and the undefined-value policy.

So there is no scope question. The real question is **architectural**: where do the nine
statistics get computed?

### The scheduling fact that settles it

Item 5 (disagreement logging) precedes item 6 (DeepSea integration) and item 9 (RQ2-Q
battery). Diagnostics 1, 2, 3, 4 and 7 all require **Q\*, which exists only on DeepSea** —
so at item 5 there is nothing to compare against. Item-5 "disagreement logging" therefore
cannot mean the Q\*-dependent statistics. It can only mean the **recording substrate**: the
per-checkpoint value samples from which every statistic is later computed.

### Recommendation: persist raw samples, compute statistics offline

Log the **`[S, M, A]` sample tensor** per checkpoint (M = K heads for ensembles; M = 30
i.i.d. draws for NoisyNet at measurement only; M = 1 for DDQN) to a per-run sidecar.
Compute all nine statistics in `analysis/rq2l/` from that artifact. Do **not** compute them
inside the training loop.

Four reasons, in order of weight:

1. **§3.6 makes an in-loop bug catastrophic.** "Implementation bug → the confirmatory block
   is void, a new pre-registered iteration follows." A statistics bug inside the run path
   voids 2,500–3,000 runs. The same bug in `analysis/` costs a re-analysis. Moving the
   statistics out of the run path shrinks the surface that can void the confirmatory block
   to the substrate alone.
2. **The statistics are frozen; the code that implements them is not yet written.** Raw
   samples are the representation from which the frozen definitions are *derivable*, so the
   analysis code can be corrected without touching a single run.
3. **Q\* arrives later than the substrate.** Raw samples are Q\*-agnostic and work for both
   parts; statistics 1–4, 7 are DeepSea-only.
4. **C1 byte-identity.** Spearman/OLS/AUC in the run path add float operations to the
   sequence that must reproduce bit-for-bit. A forward pass does not.

`[S, M, A]` is the minimum sufficient representation — marginal (Q̄, σ) per (s,a) is **not**
enough: diagnostic 2 needs the std of the *difference* `Q_m(a₁) − Q_m(a₂)` (a joint
quantity), 3 needs per-sample argmaxes, and 7 needs empirical quantiles over m.

### What already exists and should be reused

`src/diagnostics/temporal_persistence.py` already ships the `GreedySampler` protocol with
`EnsembleHeadSampler` / `NoisyNetSampler`, and already enforces the invariant this needs:
resampling draws from a **measurement** generator, never the agent's operational
`_head_rng` / `_noise_gen`, so measuring cannot perturb the run. The `probe_set` RNG stream
is already reserved in the prereg stream list. The substrate is a value-returning sibling of
the existing greedy-action sampler, not new machinery.

### Measured cost (8 threads, |S| probe states, A=6, K=10)

| \|S\| | ensemble, M=K=10 (one pass) | NoisyNet, M=30 (30 passes) | tensor, M=30, float32 |
|---|---|---|---|
| 200 | 41 ms | 2.3 s | 0.14 MB |
| 1000 | 58 ms | 6.2 s | 0.72 MB |
| 4000 | 373 ms | 8.4 s | 2.88 MB |

Negligible against the ~380 s/seed measured on the Breakout smoke run. Note the ~100×
ensemble/NoisyNet asymmetry: ensembles get all K heads from one forward pass, NoisyNet needs
M=30 separate passes. This is inherent to the frozen M definition, not an implementation
choice, and it is a **compute-trigger input** (freeze item 4 is method-specific and keyed on
per-method median wall-clock) — so the battery must be enabled during the pilot tier that
measures those medians, or the trigger will be computed from runs cheaper than the real ones.

### The blocking gap is in the protocol, not the code

**Freeze item 7 is "probe-set construction + weighting", and only the weighting is
written down** ("uniform primary; visitation-weighted secondary"). The *construction* rule —
how many probe states, sampled how — appears nowhere in the spec or the prereg. `|S|` sets
both the storage volume and the per-checkpoint cost in the table above, so it must be
written before the freeze. This is a protocol drafting item for the freeze list, and it
gates the substrate's parameters (not its structure — the substrate can be built against a
configurable `|S|` now).

Also worth stating in the paper: **DDQN's entry is structurally σ ≡ 0** (M=1). Per
diagnostic 9, "σ = 0 is a substantive measurement where it is a value rather than a
divisor" — so DDQN is a genuine zero, not a missing row.

---

## 2. Replay storage dtype — recommend `uint8`, and it is behaviourally inert

### Measured facts

Every MinAtar game emits observations in exactly `{0.0, 1.0}` — the planes are binary
indicator masks. DeepSea observations are one-hot, likewise `{0.0, 1.0}`.

| game | C | buffer @100k, float32 | uint8 |
|---|---|---|---|
| breakout, asterix | 4 | 320 MB | 80 MB |
| space_invaders | 6 | 480 MB | 120 MB |
| freeway | 7 | 560 MB | 140 MB |
| seaquest | 10 | 800 MB | 200 MB |

`float32 → uint8 → float32` is **exactly lossless** on observed frames (verified over 5,000
steps per game, all five games). A 1,500-step / 501-update DDQN-on-Breakout run under each
dtype produced **identical actions, rewards, losses, and final network weights**
(max |Δ| = 0.0).

### This reframes the decision

Because the change is provably bit-exact, it is **not a Class-1 tuning choice** — Class 1 is
"tuned once on the baseline and inherited identically", which presumes a value that *can*
affect behaviour. A lossless representation change has no behaviour to inherit. It should
still be applied uniformly, but for C11 code-path purity, not because it is a tuned constant.
It does not consume a freeze item.

### The real argument is throughput, not headroom

One run fits at float32. The study is ~2,500–3,000 runs on a single workstation, so
**per-run memory is the constraint on how many runs go in parallel** — and parallelism is
the throughput bottleneck for the whole study.

| game | 8 concurrent runs, float32 | uint8 |
|---|---|---|
| breakout | 2.56 GB | 0.64 GB |
| space_invaders | 3.84 GB | 0.96 GB |
| freeway | 4.48 GB | 1.12 GB |
| seaquest | 6.40 GB | 1.60 GB |

Against ~8 GiB available, float32 Seaquest at 8-way saturates memory and will thrash;
uint8 leaves the machine comfortable. The saving buys 8-way parallelism outright — which is
also why this should be settled *before* the pilot tier, since the wall-clock medians
measured there feed the frozen compute trigger (freeze item 4).

### Recommendation

**`uint8`, with the cast owned entirely by the buffer.** `ReplayBuffer` already accepts
`obs_dtype`, so no API change is needed — but today `gather()` returns the storage dtype,
which would silently hand agents a `uint8` tensor. Make `add()` cast to the storage dtype
and `gather()` cast back to `float32` unconditionally. Then the dtype is entirely
encapsulated: **no agent changes, no agent can forget the cast, and C11 purity is
strengthened** rather than complicated (today `_flat_obs`/`_net_obs` in each agent would
each need to know about it).

`uint8` over `bool`: numpy `bool_` is also 1 byte, so `bool` saves nothing further, and
arithmetic on torch bool tensors is a silent-error hazard. `uint8` is the standard DQN
frame convention and tolerates a future non-binary plane.

**Guard rather than assume.** The losslessness is a property of *these* environments, not of
the cast. Add a test that enumerates all six environments, steps each a few thousand times,
and asserts every observed value survives the round trip — a one-time certification with
zero run-time cost. Pair it with the existing run-scale C1 byte-identity check.
