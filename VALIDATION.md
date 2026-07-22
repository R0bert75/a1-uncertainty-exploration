# VALIDATION.md — correctness & validation gates (C0–C13)

Living checklist. Every gate is an explicit check run per session; **any failure is a
hard stop**. In confirmatory blocks, failures are governed by the operational failure
policy (spec §3.6). The reviewer agent is a second opinion, never a replacement for
these gates or the advisor gate.

Legend: `[ ]` not yet · `[~]` in progress · `[x]` passed (with session + evidence path).

---

- [ ] **C0 — No ranking-conditioned gates.** No gate, checkpoint, or stopping rule is
  ever conditioned on the ranking of the studied methods. Success/failure of any
  uncertainty-aware method is a *result*, never a correctness condition.

- [ ] **C1 — Determinism (incl. the derived-stream scheme).** Same `(config, seed)`
  reproduces the same curve bit-for-bit, or within documented nondeterminism. Every random
  stream is derived as `hash(master_seed, cell_id, stream_name, seed_index)` with no stream
  reused across cells — implemented in `src/utils/conventions.py` (`derive_seed`) and
  unit-tested in `tests/test_conventions.py` (pinned regression value + cross-cell
  non-overlap). _Utility + tests: Session 0. Full-run determinism: Session 3._

- [ ] **C2 — Logging integrity.** Every metric in a figure exists in a committed CSV;
  `make figures` reads only those CSVs. No figure sourced from in-memory state. _Session 3._

- [ ] **C3 — Baseline sanity.** Double DQN reaches published-ballpark scores on MinAtar.
  A silently broken baseline makes every method look good by comparison. _Session 3._

- [ ] **C4 — Ensemble diversity / masking.** Bootstrapped heads are genuinely diverse
  (disagreement > 0, not collapsed from step 0); bootstrap masking is correct. Posterior
  collapse is a *finding to measure*, not a bug to hide. _Session 4._

- [ ] **C5 — DeepSea environment / implementation validity.** Small-N transitions,
  rewards, terminal conditions, and Q\* match hand calculations and reference behaviour;
  a known-capable oracle solves it. Environment-and-implementation only — never a
  hypothesis gate. _Session 5._

- [ ] **C6 — NoisyNet correctness.** Noise injected during native action selection;
  learned σ parameters behave as intended. _Session 5._

- [ ] **C7 — Ground truth + battery formulas.** DeepSea Q\* and uncertainty intervals
  validated on small N; the uncertainty battery computed exactly per the frozen formulas
  (spec §3.3), including the undefined-value policy. MinAtar diagnostics remain approximate.
  _Session 6._

- [ ] **C8 — No contamination (both splits).** MinAtar: evaluation games never used in
  tuning. DeepSea: development sizes never used as confirmatory. Primary figures use the
  correct split; all-task/secondary views labelled. _Session 7._

- [ ] **C9 — Equal-tuning audit.** Every method receives the identical pre-registered
  search budget within each family; no method gets extra configs or seeds. _Session 7._

- [ ] **C10 — Aggregation discipline; hierarchy honoured.** Probability of improvement,
  raw per-game differences, and performance profiles lead RQ1 (no ratios anywhere).
  The §1.1 estimand hierarchy is reported exactly as frozen — primary first, fixed-sequence
  secondaries with the stopping rule, everything else descriptive. _Session 10._

- [ ] **C11 — Per-contrast code-path purity.** For every reported contrast, the two cells
  differ only in the varied factor along the code path (no incidental branch differences).
  _Sessions 4–6b._

- [ ] **C12 — Frozen-policy / checkpoint extraction.** Budget-axis checkpoints
  (100k/500k/1M) are read from the same unconditional 1M runs (Variant B); extraction
  rules are the frozen ones. _Session 7._

- [ ] **C13 — Configuration-identity audit.** For every reported contrast pair, the two
  cells' *fully resolved* configs differ only in the varied factor and its pre-registered
  factor-specific parameters (three-class parameter rule). Audit output committed to
  `audits/c13/` alongside the figures. _Sessions 4, 5, 6b, 7._

---

## Session tick log

| Session | Gates checked | Result | Evidence |
|---|---|---|---|
| 0 (bootstrap) | — (no science yet) | env + skeleton + CI green | `build_manifest/versions.json`, this repo |
| infra (pre-freeze) | C1 (agent-level) | DDQN baseline + replay + shared Q-net; stream-derived init/replay/action_noise reproduce a full run bit-for-bit; cross-cell divergence verified | `src/ddqn.py`, `src/networks.py`, `src/replay_buffer.py`, `tests/test_ddqn.py` (34 tests pass) |
| infra (pre-freeze) | C5/C7 groundwork (not the gates) | DeepSea env + exact Q\* solver; Q\* matched to brute-force enumeration (γ∈{1,0.99}) and the 0.99 optimal return; per-row mapping bound to frozen `env_mapping` stream + hash; DDQN solves N=5 (mean return 0.05→0.88, best 0.99). C5/C7 remain the Session-5/6 gates. | `src/deep_sea.py`, `tests/test_deep_sea.py` (12 tests pass) |
| infra (pre-freeze) | C13 groundwork | Run-config loader: validates the config vocabulary + `arm`↔cell_id agreement, derives `size_class`, computes the `config_sha256` identity, serializes `resolved_config.json`, mints per-seed `RunContext`s, and builds env+agent from a frozen YAML (a whole DDQN-on-DeepSea run executes from config alone). Freeze discipline enforced in code: confirmatory configs may not inherit development placeholders; no frozen numeric value baked in; `master_seed` is a required config field. | `src/config.py`, `tests/test_config.py` (24 tests pass) |
| infra (pre-freeze) | C1 + C11 (ensemble) | Bootstrapped-DQN ensemble on the **same** `MLPQNetwork`/`ReplayBuffer`/Double-DQN code path as the baseline (only `n_heads=K` + a head-masked loss differ — the C11 purity claim in code). Frozen Class-2 mechanics implemented + tested: per-head Ber(mask_prob) bootstrap masks off the `bootstrap_mask` stream (Osband 2018 §3.1); 1/K shared-trunk gradient normalization verified exact (Osband 2016 §6.1); K per-head Double-DQN targets; three use-rules (episodic/per_step posterior sampling, ensemble_mean ε-greedy); stream-derived, reproduces bit-for-bit from the RNG triple with no global-RNG dependence. Wired into the config loader (`IMPLEMENTED_METHODS` gains `bdqn`, prior=off); prior=on (RP-BDQN randomized prior functions) is a documented seam, not yet built. | `src/bdqn.py`, `tests/test_bdqn.py` (19 tests pass); `src/networks.py`, `src/replay_buffer.py` (trunk/head + index-sampling seams) |
