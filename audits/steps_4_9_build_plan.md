# Steps 4–9: what must be built

**Audited 2026-07-31 against `docs/a1-requirements-and-alternatives-v6.3.md` §8 (implementation
order, 17 steps) and `protocol/preregistration.md` freeze items 2, 3, 12, 20 and §3.3.**
Method: read each step's spec text, then grep the tree for the component it names. Every "missing"
below is an absent symbol or absent config, not an impression.

![Steps 4–9 gap]({{artifact:art_c30b4b4c-251d-4826-869f-9f5af74d0f5b}})

---

## Summary

| Step | Required | Exists | Blocking gap |
|---|---|---|---|
| 4. 5-seed baseline + wall-clock | 2 | 2 | none — code complete, never run |
| 5. Switchboard + backbone tuning + disagreement | 4 | 1 | **sweep driver**; tuning objective unspecified; disagreement logging; 6 of 10 cells |
| 6. DeepSea dev sizes + solve-vs-depth | 2 | 1 | figure; needs step-5 runs first |
| 7. Priors + `prior_scale` mini-search | 2 | 1 | mini-search (reuses step-5 driver) |
| 8. NoisyNet + `eps_schedule` mini-search | 3 | 1 | mini-search **+ its objective's input cell does not exist** |
| 9. RQ2-Q battery | 10 | 3 | 6 of 9 diagnostics; trainer never calls the substrate |

**One artefact unblocks four steps.** Steps 5, 7, 8 and (transitively) 6 all wait on a single
missing module: something that takes a config template plus a parameter grid, runs the cells, and
feeds per-seed scores to `selection.score_candidates`. Step 9 is independent and is the largest
single body of work.

---

## Step 4 — 5-seed baseline; per-method wall-clock

**Spec:** "5-seed baseline; per-method wall-clock recorded (trigger inputs)."

**State: code complete, never executed.** `trainer.train()` runs every committed seed;
`_write_compute_sidecar` writes per-seed wall-clock to `<run_id>.compute.json`, deliberately outside
the metrics CSV so gate C1's byte-exact re-run property survives.

**Gap: not code — data.** `logs/` contains only `dummy_smoke.csv`. Per-method wall-clock is a
**descope-ladder trigger input** (spec §8 item 4) and feeds the freeze item 4 cap-X formula, so
until a real 5-seed run exists both triggers are un-evaluable. This is the cheapest item on the
list and should be run first, on DeepSea N=10.

---

## Step 5 — Switchboard + backbone-tuning pass + disagreement logging

The largest step, and it decomposes into four independent pieces.

### 5a. Sweep driver — MISSING (the critical artefact)

No `src/search.py`, `src/sweep.py`, or equivalent. `selection.score_candidates(points, score_fn,
sort_key_fn)` is a *pure* function by design — it consumes already-collected scores. Nothing
supplies `score_fn`. What must exist:

- sample `n_backbone = 12` points from the frozen class-1 distributions using the new
  `hparam_search` RNG stream;
- materialize each as a `RunConfig` override over a template, run 3 seeds × 2 dev sizes
  (N ∈ {10, 20}) = 6 runs per candidate;
- reduce each candidate's per-seed scores through `iqm`, tie-break on the lower parameter value,
  and emit the winner as a committed config plus an auditable search record.

Budget already approved: 12 candidates × 6 runs = **72 runs** of the 120 allotted to all tuning
(the remaining 48 are the two mini-searches, steps 7 and 8).

### 5b. Tuning objective metric — UNSPECIFIED IN THE PROTOCOL

Freeze item 2 pins the *statistic* (IQM, item 3) and the *tie-break* (lower parameter value) for the
backbone search, and pins full objectives for both class-3 mini-searches — `prior_scale` by "IQM of
the canonical prior-on cell `(episodic, on, 10)` on development sizes", `eps_schedule` by "IQM of
`(mean_eps, off, 10)`". **The backbone search names no metric at all.** "IQM" is a reduction; IQM
*of what* is undefined.

This is a genuine protocol gap, and it is not cosmetic, because the obvious default is a poor
objective. DeepSea's primary outcome is binary per seed (`discovered`, `trainer.py` L350). On 6
binary seeds IQM takes only **5 distinct values** — it collapses 0/6 with 1/6, and 5/6 with 6/6:

| successes k | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| IQM | 0.000 | **0.000** | 0.167 | 0.500 | 0.833 | **1.000** | 1.000 |

Simulated over a 12-candidate field (20 000 trials), IQM-of-terminal-discovery ties at the top in
**44 %** of searches in a low-signal regime and **98 %** once several candidates are good — where
the tie-break ("lower parameter value") is uncorrelated with performance, making the selection
close to arbitrary. Using the **area under the discovery curve** instead — already recoverable from
the logged data, since `discovery_prob` is written at every checkpoint, so a seed that discovers
early scores above one that discovers late — drops the tie rate to **3–11 %** and raises
P(select the truly best candidate) from 0.44 to 0.61 in the high-signal regime:

| regime | objective | P(tie at top) | P(pick true best) |
|---|---|---|---|
| hard (p ∈ 0.00–0.25) | terminal binary | 0.438 | 0.470 |
| hard | discovery AUC | **0.112** | 0.466 |
| mixed (p ∈ 0.05–0.60) | terminal binary | 0.445 | 0.587 |
| mixed | discovery AUC | **0.027** | 0.587 |
| easy (p ∈ 0.40–0.90) | terminal binary | 0.980 | 0.439 |
| easy | discovery AUC | **0.053** | **0.610** |

**This is a stage-3 protocol item, not a code decision.** It must be written into freeze item 2
before the search runs, or the choice of objective becomes a post-hoc degree of freedom. It is the
fourth staged gap and should join `protocol/decisions/staged_stage3_protocol_fixes.md`.

### 5c. Disagreement logging — MISSING

Spec step 5 requires it explicitly. No `sigma`, `head_std`, or disagreement symbol exists in
`bdqn.py` or `trainer.py`. This is the per-checkpoint ensemble-spread quantity that RQ2-L's σ(s,a)
diagnostics consume, so it is also a step-9 prerequisite.

### 5d. Six of ten Part-A cells have no config

Freeze item 12's 10-cell structured partial factorial: `use_rule × prior` at K=10 (6 cells) plus
`episodic_head` at K ∈ {5, 20} × both prior levels (4 cells). Committed DeepSea configs cover only
`episodic|off|K10`, `episodic|on|K10`, `episodic|off|K1` (the DDQN reference, not one of the 10),
and a NoisyNet arm.

**`ensemble_mean` has no DeepSea config at all** — so the **C-USE contrast** (episodic_head vs.
capacity-matched ensemble-mean ε-greedy) currently has no instantiated comparison arm, and neither
does `per_step`, which is the temporal-coherence comparator.

---

## Step 6 — DeepSea integration, development sizes; pilot solve-vs-depth

Env, exact `q_star`, and both trainer lanes exist. Missing: the **pilot solve-vs-depth figure**
(`make_figures.py` has one generic `_plot_group` and no depth-axis figure). It needs real runs
across N, so it follows 5a rather than blocking it. Must carry a "pilot" label per spec.

---

## Step 7 — Randomized priors + `prior_scale` mini-search

`rp_bdqn` is implemented and `prior_scale: 3.0` sits in the configs as a **declared placeholder**.
The mini-search is `n_mini = 4` candidates and its objective **is** pinned, so once 5a exists this
step is nearly free. Its input cell `(episodic, on, 10)` is one of the two that already has a config.

---

## Step 8 — NoisyNet + `eps_schedule` mini-search

NoisyNet is implemented with its own arm string. This step is blocked **harder than step 7**: its
pinned objective is "IQM of `(mean_eps, off, 10)` on development sizes", and per 5d that cell has no
config. The objective's input population does not exist. Fixing 5d is a prerequisite, not a parallel
task.

---

## Step 9 — RQ2-Q battery on development sizes

The nine frozen §3.3 diagnostics, against the tree:

| # | Diagnostic | State |
|---|---|---|
| 1 | Marginal alignment — Spearman ρ over (s,a) between σ and \|Q̄ − Q*\| (**RQ2-L primary**) | missing |
| 2 | Action-gap alignment (top-2 by Q̄, ties by lowest action index) | missing |
| 3 | Incorrect-argmax flagging | missing |
| 4 | Optimal-path uncertainty (per depth; AUC over depth) | missing |
| 5 | Visitation-conditioned decay (OLS of log σ on log(1+v)) | missing |
| 6 | Temporal persistence | **implemented** (`temporal_persistence.py`, both variants + samplers) |
| 7 | Empirical containment (central 80 % interval) | missing |
| 8 | MinAtar behavior-policy analogue — deterministic conditional (freeze item 20) | missing |
| 9 | Undefined-value policy (NA, counted, published; σ = 0 substantive) | missing |

Plus two structural pieces:

- **`q_star` — done and validated.** `tests/test_deep_sea.py` checks it against brute-force
  enumeration, satisfying "Q* validated on hand-checkable N".
- **The substrate is never called.** `src/diagnostics/substrate.py` exists and is cap-agnostic, but
  grep finds no reference to it in `trainer.py`. Nothing collects `{Q_m(s,a)}` samples during a run,
  so no diagnostic can be computed even once written. **Wiring the substrate into the checkpoint
  path is the first step-9 task**, and it depends on 5c (disagreement logging) touching the same code.

### Correction to a prior claim

An earlier session recorded that "MinAtar |S| does not exist — the battery is DeepSea-only." The
first half stands: diagnostics 1–5 and 7 all reference `Q*` and are DeepSea-only, and no MinAtar
probe-set size is specified anywhere. But **diagnostic 8 is a MinAtar clause**, via freeze item 20:
100 clone/restore reproduction tests deciding between full probe rollouts, episode-start-only, or
dropping the analogue. It is exploratory and appendix-only and must never be called "Q-error", but
it is a required component of step 9. `src/minatar_env.py` has **no clone/restore/get_state
support**, so the conditional cannot currently be evaluated in any direction.

---

## Budget arithmetic (checked, reconciles)

The approved 120 runs covers **all 20 tuning candidates**, not the backbone alone:
12 backbone + 4 `prior_scale` + 4 `eps_schedule` = 20, at 3 seeds × 2 development sizes = 6 runs
each. The backbone search's own share is **72 runs**; the two mini-searches are 24 each. Development
tier total 110 + 120 = **230**, inside the 150–250 envelope. No discrepancy — the figure in the
sign-off note is right, and step 5a's cost is 72 of it.

---

## Recommended order

1. **Run step 4** on DeepSea N=10 — cheapest, and unblocks both compute triggers.
2. **Add the 6 missing cell configs** (5d) — pure YAML + schema check; unblocks step 8's objective.
3. **Stage the tuning-objective gap** (5b) as staged fix #4; get sign-off on discovery-AUC.
4. **Build the sweep driver** (5a) — the one artefact four steps wait on.
5. **Disagreement logging + substrate wiring** (5c + step 9 structural) — same code path.
6. **Run the backbone tuning pass**, then steps 7 and 8's mini-searches, then step 6's figure.
7. **Diagnostics 1–5, 7, 9**, then the item-20 clone conditional (8).

Steps 1–3 are hours and need no new science. Step 4 is the real build. Step 9 is the long tail and
can proceed in parallel with the tuning runs.
