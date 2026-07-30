# Code-completeness audit — pre-freeze (Session 3)

**Scope.** Every component the frozen spec (`docs/a1-requirements-and-alternatives-v6.3.md`)
and the pre-registration (`protocol/preregistration.md`) require, checked against what exists
in this repo. The question answered is *"is the code in place, is anything missing, is
everything committed"* — not *"are the gates green"* (gates C3–C10 are, by design, later
sessions' evidence, not code).

**Audit basis.** Spec §2.1 (switchboard), §3.1–§3.4 (protocol, evaluation, diagnostics,
budgets), §4 (repository requirements), §6 (deliverables), §8 (implementation order);
`VALIDATION.md` C0–C13. Test counts from `pytest --collect-only` at the audited commit.

---

## 1. Summary

The pipeline spine is complete on both environment families. All four canonical MinAtar
methods build and run end-to-end from config alone, on the step-budgeted lane, logging both
frozen reporting axes into the frozen CSV schema, with a figure rebuilt from logs alone.

**276 tests pass; ruff clean.** Nothing required for a pre-freeze state is missing. What
remains unbuilt is, in every case, either (a) gated behind the freeze by design, or (b)
scheduled for a later session in the spec's own §8 implementation order.

| Area | State |
| --- | --- |
| Part A (DeepSea) path | complete — env + exact Q\*, episode lane, 3 methods |
| Part B (MinAtar) path | complete — env adapter, conv trunk, step lane, 4 methods |
| Switchboard + 3 parameter classes | complete, enforced in the loader |
| RNG derivation (8 streams) | complete, CI-regression-pinned |
| CSV schema + logger | complete, frozen header enforced |
| Frozen-policy extraction (both axes) | complete — gap closed this session |
| Uncertainty battery (§3.3, 9 items) | 1 of 9 implemented (by design: pre-freeze-safe subset) |
| Analysis (§1.1 hierarchy, RQ2-L) | placeholder figure only; dirs exist, empty |
| QR-DQN | deliberately deferred (exploratory, §8 item 16) |

---

## 2. Implemented — spec-required components present

| Spec requirement | Where | Tests |
| --- | --- | --- |
| §2.1 switchboard (`use_rule` × `prior` × K), cell_id ↔ `arm` agreement | `src/config.py` | 35 |
| Three parameter classes; confirmatory configs may not inherit dev placeholders | `src/config.py` | 35 |
| Determinism from `(config, seed)` via 8 cell-specific derived streams | `src/utils/conventions.py` | 21 |
| CSV logging, frozen header, `axis` ∈ {online, frozen_policy}, `is_t0` | `src/utils/conventions.py` | 21 |
| DeepSea env + **exact Q\*** solver (brute-force matched, γ ∈ {1, 0.99}) | `src/deep_sea.py` | 12 |
| ε-greedy Double DQN baseline (method 1) | `src/ddqn.py` | 11 |
| NoisyNet-DQN (method 2), factorized Gaussian, no ε schedule | `src/noisynet.py` | 22 |
| Bootstrapped DQN (method 3) + randomized priors → RP-BDQN (method 4) | `src/bdqn.py` | 28 |
| 1/K shared-trunk gradient normalization (Osband 2016 §6.1) | `src/bdqn.py` | 28 |
| MinAtar env adapter, gymnasium-style, pinned sticky/ramp defaults | `src/minatar_env.py` | 23 |
| MinAtar conv trunk + K heads + noisy variant | `src/networks.py` | 30 |
| MinAtar config branch (all 5 games, 4 methods, step budgets) | `src/config.py` | 46 |
| Episode-budgeted trainer lane (Part A) | `src/trainer.py` | 9 |
| **Step-budgeted trainer lane (Part B), both axes** | `src/trainer.py` | 20 |
| Temporal persistence (§3.3 #6) — the pre-freeze-safe battery member | `src/diagnostics/temporal_persistence.py` | 17 |
| `make figures` rebuilt from `logs/` alone | `analysis/make_figures.py`, `Makefile` | 2 |
| C13 configuration-identity audit script | `audits/c13_audit.py` | — |
| Committed-config schema check (§4 CI clause) | `audits/config_schema_check.py` | 1 |
| Per-seed wall-clock reporting (compute sidecar) | `src/trainer.py` | 1 |
| CI: lint, tests, derived-stream regression, figures-from-logs, C13 | `.github/workflows/smoke.yml` | — |

### Defects found and fixed during this session

Listed in descending order of consequence. The first is the one that would have corrupted
results rather than merely inconveniencing analysis.

1. **The two Part-B baselines shared every RNG stream seed.** `configs/example_noisynet_
   breakout_dev.yaml` carried `arm: episodic|off|K1` — the DDQN reference cell id. Streams are
   keyed on `cell_id` alone, so the NoisyNet and DDQN Breakout runs drew *identical*
   `init`, `replay`, and `action_noise` seeds: same network initialization, same replay index
   sequence, correlated rather than independent runs. The loader already had a guard against
   exactly this, with reasoning that is part-independent — but it was scoped `if part == "A"`,
   so Part B slipped through. Guard widened to all parts; config fixed to `arm: noisynet`;
   two regression tests added (one asserting the rejection, one asserting stream separation
   across all three streams).
2. **`BDQNAgent.mean_action` was missing.** Spec §5 pins the frozen-policy extraction per
   method: DDQN → `greedy(Q)`, NoisyNet → noise-off greedy, **bootstrapped cells → greedy
   w.r.t. ensemble-mean Q**. DDQN and NoisyNet each exposed their entry point; the ensemble
   agent had only `greedy_action_of_head` and the use-rule-dependent `select_action` (whose
   `ensemble_mean` branch applies ε, so it is not the extraction). Without it, half the
   canonical four had no secondary axis. Added, with the distinction documented: the
   extraction applies no ε and draws from no stream.
3. **The figure grouping key omitted `axis`.** Both reporting axes share the metric name
   `episode_return`, so grouping by `(part, env, metric)` overplotted two different estimands
   on one panel. `axis` is now part of the key and the filename.
4. **`example_rpbdqn_deepsea_dev.yaml` declared `method: bdqn`** while its Breakout twin
   declared `method: rp_bdqn`. Both resolve to the same agent (the names are aliases), but the
   C13 identity audit compares method names across contrast pairs, so the same cell appearing
   under two names is a trap. Normalized to `rp_bdqn`; test added.

### Gaps closed during this session

- **G9 (wall-clock).** Per-seed wall-clock now recorded in both lanes and written to
  `<out_dir>/<run_id>.compute.json`. Deliberately **not** in the metrics CSV: wall-clock is
  machine-dependent and gate C1 requires that CSV to reproduce byte-for-byte. Test asserts
  both the sidecar's presence and its absence from the CSV.
- **G12 (config-schema CI).** `audits/config_schema_check.py` walks `configs/*.yaml` through
  the loader; wired into CI and as `make schema` (now part of `make smoke`). This script is
  what surfaced defects 1 and 4 above.
- **G14 (README).** Layout section now names every module and states explicitly that
  `analysis/hierarchy/` and `analysis/rq2l/` are reserved and empty, so the README no longer
  reads as more complete than the code.
- **CSVLogger docstring** corrected to state that the header is frozen and extra keys raise.

---

## 3. Gap list — not implemented, with disposition

Each row states *why* it is absent, so the list is actionable rather than a bare diff.

| # | Component | Spec ref | Why absent | Disposition |
| --- | --- | --- | --- | --- |
| G1 | Battery items §3.3 #1–#5, #7 (marginal alignment, action-gap alignment, incorrect-argmax rank-biserial, optimal-path σ, visitation-conditioned decay, empirical containment) | §3.3 | All six reference **Q\*** or a frozen constant, i.e. learned-vs-truth comparisons. Building them pre-freeze risks fitting the estimator to the diagnostic. | **Session 6 (C7 gate)**, per §8 item 9. Deliberate. |
| G2 | Battery item §3.3 #8 (MinAtar clone/restore reproduction spike, 100 tests) | §3.3 #8 | Exploratory, appendix-only, deterministic-conditional. Requires the MinAtar path to exist first — it now does. | Now unblocked; schedule with the Part-B pilot. |
| G3 | Battery item §3.3 #9 (undefined-value / NA policy as shared code) | §3.3 #9 | Battery-wide policy; there is no battery yet to apply it to. | With G1. |
| G4 | Frozen diagnostic **probe-set generator** | §3.3 notation ("probe set S"); `probe_set` stream is registered | The stream name is reserved in the registry, but no generator exists. Needed by G1. | With G1; the reserved stream is the seam. |
| G5 | §1.1 primary-estimand hierarchy + fixed-sequence testing | §1.1, §6 v1.0 | `analysis/hierarchy/` exists and is **empty**. Analysis code operates on committed CSVs, so it can be written any time — but the estimand is frozen and the tests are pre-registered, so writing it now buys nothing. | Post-freeze, pre-Gate-A. |
| G6 | RQ2-L concordance + permutation analysis | §3.3 #1, §6 | `analysis/rq2l/` exists and is **empty**. Same reasoning as G5. | Post-freeze. |
| G7 | Real (non-placeholder) figures; the EWRL-grade figure | §6 M1 | `analysis/make_figures.py` is explicitly a placeholder: mean±band per method, titled `[PLACEHOLDER]`. Correct for a pipeline spine. | Post-pilot. |
| G8 | QR-DQN | §8 item 16 | Distributional control, exploratory follow-up, descope rung 6. `VALID_METHODS` includes it; `IMPLEMENTED_METHODS` does not, and `build_agent` raises `NotImplementedError` — a deliberate, tested boundary. | Last, or documented descope. |
| G9 | Per-method **wall-clock / compute recording** | §8 item 4, §6 v1.0 ("compute reported") | Was absent entirely. | **CLOSED this session** — `<run_id>.compute.json` sidecar, both lanes. |
| G10 | Ensemble **disagreement logging** | §8 item 5 | Named in the implementation order alongside the switchboard. No `disagreement` metric is emitted. Partly subsumed by the battery (G1), but §8 lists it separately at item 5, i.e. *before* the battery. | Clarify whether it is a battery member or a standalone metric. |
| G11 | C11 per-contrast purity as an **automated** check | §4 ("CI smoke + C11 per-contrast purity + factorial config schema") | C11 is currently argued in code comments and asserted per-method in tests, not checked per *contrast pair* in CI. C13 has a script; C11 does not. | Recommend a script mirroring `c13_audit.py`. |
| G12 | Factorial **config-schema** check in CI | §4 (same clause) | The loader validated a config only when a run loaded it; nothing walked the committed set. | **CLOSED this session** — `audits/config_schema_check.py`, in CI + `make schema`. |
| G13 | MinAtar replay-buffer **dtype** | not spec-pinned | `ReplayBuffer` supports `obs_dtype`, but the agents construct it with the default `float32`. At the pre-registered 100k capacity and 10×10×C observations this is 320–800 MB per run depending on the game (uint8 would be 80–200 MB). Correctness is unaffected; parallel-seed memory is. | Class-1 nuisance; decide before the pilot. |
| G14 | `README.md` layout section | §6 ("honest README") | Described `analysis/` as containing hierarchy/rq2l analysis that does not exist yet. | **CLOSED this session** — every module named; empty dirs marked reserved. |

### One documentation defect worth recording

`CSVLogger.log()`'s docstring states that `extra` keys "become extra columns (only in the
header of a fresh file)". They do not: `_fieldnames` is fixed to `BASE_FIELDS` at
construction and `csv.DictWriter` raises `ValueError` on any unknown key. The frozen header
(gate C2) is the correct behavior and wins; the docstring is wrong. Worked around in the step
lane by emitting `episodes_in_window` as its own metric row. **Fix the docstring, not the
code.**

---

## 4. Remaining pre-freeze recommendations

After this session's fixes, the unscheduled-and-missing list is down to two items, neither of
which touches a frozen value:

1. **G11 — C11 per-contrast purity script.** §4 names it in the same CI clause as C13 and the
   config schema, both of which now have scripts. C11 is the harder one: it needs a
   contrast-pair manifest to check against (which pairs are reported, and which factor each
   pair varies), so the manifest is the real design question, not the checking code. Worth
   scoping before the freeze even if the script lands after it.
2. **G10 — disagreement logging.** §8 lists it at item 5, *before* the battery at item 9, which
   suggests it is meant as a standalone online metric rather than a battery member. Needs a
   one-line decision: if it is `d(s) = 1 − modal fraction` over a probe set, it is battery item
   §3.3 #3 and belongs with G1; if it is a cheap per-checkpoint ensemble spread, it is a
   separate metric row and can be added now.

**G13 (replay-buffer dtype)** is worth a decision before any multi-seed MinAtar run: the
buffer stores `float32` at the pre-registered 100k capacity, i.e. 320–800 MB per run depending
on the game, where `uint8` would be 80–200 MB. `ReplayBuffer` already accepts `obs_dtype`, so
this is a one-line change in the agent constructors. Correctness is unaffected; the constraint
is how many seeds fit in memory concurrently. Class-1 nuisance, so no freeze implication.

## 5. Verification at the audited commit

- `pytest -q` → **276 passed** (157 at the start of the session)
- `ruff check .` → clean
- `python audits/config_schema_check.py` → 8/8 committed configs resolve
- `python audits/c13_audit.py` → passes
- `make figures` rebuilds every figure from `logs/` alone
- **DDQN-on-Breakout smoke run** (3 seeds × 30k steps, step lane): both axes populated at
  all three checkpoints; online return 0.76 → 4.43, frozen-policy return 2.13 → 5.60;
  `steps_to_first_reward` 11–17 steps, 0 censored. Re-run at the same seeds reproduces the
  CSV byte-for-byte (C1 at 30k-step scale, not just unit-test scale).
- Freeze tags (`prereg-draft`, `session-0-bootstrap`) untouched

**Bottom line.** From the code and infrastructure side the project is ready to run. Both
environment families, all four canonical methods, both reporting axes, and the figure path
are wired and tested end-to-end. What is not built is deliberately not built — the
uncertainty battery and the confirmatory analysis are the frozen-spec's own post-freeze work.
