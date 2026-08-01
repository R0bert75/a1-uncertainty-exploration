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
| 4. 5-seed baseline + wall-clock | 2 | 2 | none — code complete; no 5-seed run committed |
| 5. Switchboard + backbone tuning + disagreement | 4 | 4 | ~~none — all four closed 2026-08-01~~ (see §5; the row as first written said "6 of 10 cells", the true count was 8 of 10) |
| 6. DeepSea dev sizes + solve-vs-depth | 2 | 1 | figure; needs step-5 runs first |
| 7. Priors + `prior_scale` mini-search | 2 | 2 | none — code complete; no runs committed |
| 8. NoisyNet + `eps_schedule` mini-search | 3 | 3 | none — code complete; no runs committed. ~~input cell does not exist~~ (committed `b0f63ef`) |
| 9. RQ2-Q battery | 10 | 4 | 6 of 9 diagnostics; #8 needs MinAtar clone/restore; ~~trainer never calls the substrate~~ **wired at `0d6df0b`; this row was stale** |

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

**Gap: not code — data.** The committed `logs/` holds only `dummy_smoke.csv`; no run at the
required 5 seeds exists, and no per-method wall-clock is persisted in the repository for **any**
method.

One real measurement was taken this session and then lost to the workspace sweep — the
DDQN-on-Breakout smoke run (`smoke_ddqn_breakout`, cell `episodic|off|K1`, 3 seeds × 30 000 steps).
Recording it here because it is the project's only empirical timing figure and it feeds the cap-X
formula:

| quantity | value |
|---|---|
| per-seed wall-clock | 405.7 / 371.3 / 358.6 s (mean **378.5 s**) |
| total | 1135.6 s for 90 000 env steps |
| throughput | **≈ 79 env-steps/s per seed**, 8-core CPU, no GPU |
| extrapolated | 0.35 h/seed at 100 k steps; **3.5 h/seed at 1 M steps** |

That is one method on one MinAtar game at 3 seeds. It does **not** satisfy step 4, which needs
5 seeds and *per-method* figures across all four arms — but it does mean the cap-X input is no
longer a total unknown, and the 1 M-step extrapolation is worth reading before the confirmatory
tier is scheduled. The step-4 run itself is the cheapest item on this list and should go first,
on DeepSea N=10.

**Process note:** the smoke run's CSV and `.compute.json` went to `dev_battery/`, which is
`.gitignore`d, so the numbers survived only in the session transcript. Whatever produces the step-4
figures must commit them under `logs/`.

---

## Step 5 — Switchboard + backbone-tuning pass + disagreement logging

The largest step, and it decomposes into four independent pieces.

### 5a. Sweep driver — **BUILT 2026-08-01** (`src/search.py`, 28 tests)

> **Status update 2026-08-01.** `src/search.py` now exists and does all four bullets below.
> `make search-dry` prints the frozen 12-candidate field without executing anything;
> `make search-backbone` runs the pass. The original entry is kept verbatim underneath because
> it is the specification the module was built against.
>
> Two RNG facts surfaced during the build that the entry below did not anticipate, and both are
> now pinned by mutation-verified tests:
>
> 1. **Tuning needs its own `cell_id` namespace.** Streams key on `cell_id` alone, so a tuning
>    run reusing the reference cell's arm (`episodic|off|K1`) would draw byte-identical
>    `init`/`env_mapping`/`replay`/`action_noise` at the same `seed_index` as that cell's
>    *evaluation* runs — the backbone would be selected on the very environment instances it is
>    later measured on. `src/config.py` gained a `tune|` arm branch, restricted to
>    `role: exploratory` so a reported config cannot use it to escape the factorial identity check.
> 2. **Candidates must *share* streams with each other.** All 12 use the same tuning `cell_id`, so
>    at a given seed index every candidate gets the same DeepSea mapping — common random numbers,
>    which is what makes the candidate contrast a hyperparameter contrast. The `cell_id` therefore
>    encodes neither the candidate index nor the size; `run_id` and the config fingerprint carry
>    those, so the 72 runs stay separable.
>
> Known limitation, deliberate: `deepsea_action_mapping` derives from a size-independent stream, so
> the N=20 mapping's first 10 entries equal the N=10 mapping's. The two sizes of one candidate are
> not independent draws. Left as-is — it is what the committed cell configs already do, and
> diverging here would tune under a different mapping convention than the runs being tuned for.

<details>
<summary>Original entry as first written (2026-07-31) — the spec the module was built against</summary>

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

</details>

### 5b. Tuning objective metric — **RESOLVED 2026-08-01** (Gap 4, owner sign-off)

> **Status update 2026-08-01.** Owner chose the discovery-AUC objective, and the scope was
> *widened*: this entry claimed only the backbone search lacked an objective while "both class-3
> mini-searches" had theirs pinned. That was wrong — grepping every "IQM of" clause showed each
> names a *cell* and a *size set* but never an outcome, and both mini-searches run at the same
> 6-runs-per-candidate scale, so they inherit the identical tie pathology. One sub-clause now
> governs all three searches. See Gap 4 in `protocol/decisions/staged_stage3_protocol_fixes.md`
> for the corrected simulation (the accuracy figures quoted below were recomputed: the original
> model drew discovery *time* independently of candidate quality, so AUC could only break ties by
> noise; under a coupled hazard AUC weakly dominates, never worse, 29 % vs 20 % top-1 in the easy
> regime). Implemented as `search.discovery_auc`.

<details>
<summary>Original entry as first written (2026-07-31)</summary>


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

</details>

### 5c. Disagreement logging — **BUILT 2026-08-01** (`src/diagnostics/`, main `0d6df0b`)

> **Status update 2026-08-01.** Three modules, each with one concern: `substrate.py` (record type
> + npz writer, no agent knowledge), `samplers.py` (`ValueSampler` adapters + probe set, no run
> knowledge), `recorder.py` (per-run probe set, visitation histogram, spec). Exposed as
> `trainer --diagnostics`, default OFF, episode lane only — the whole battery references Q\*, which
> MinAtar does not have. **Load-bearing invariant:** `--diagnostics` leaves the metrics CSV
> byte-identical; that is what lets it be a CLI flag rather than a config field, keeping it out of
> the config fingerprint without weakening the reproducibility gate. Verified by mutation.

<details>
<summary>Original entry as first written (2026-07-31)</summary>


Spec step 5 requires it explicitly. No `sigma`, `head_std`, or disagreement symbol exists in
`bdqn.py` or `trainer.py`. This is the per-checkpoint ensemble-spread quantity that RQ2-L's σ(s,a)
diagnostics consume, so it is also a step-9 prerequisite.

</details>

### 5d. ~~Six~~ **Eight** of ten Part-A cells have no config — **CLOSED 2026-08-01** (main `b0f63ef`)

> **Status update + correction 2026-08-01.** All 10 factorial cells now have committed DeepSea
> configs, so `C-USE` and `C-COHERENCE` have instantiated comparison arms and step 8's pinned
> objective cell (`ensemble_mean|off|K10`) exists. **The heading's count was wrong:** it said six
> missing, but `episodic|off|K1` is the DDQN *reference* and sits outside the switchboard, so only
> two of the ten were covered and **eight** were missing. The paragraph below states this correctly
> in its own text — the arithmetic in the heading did not follow it.

<details>
<summary>Original entry as first written (2026-07-31)</summary>


Freeze item 12's 10-cell structured partial factorial: `use_rule × prior` at K=10 (6 cells) plus
`episodic_head` at K ∈ {5, 20} × both prior levels (4 cells). Committed DeepSea configs cover only
`episodic|off|K10`, `episodic|on|K10`, `episodic|off|K1` (the DDQN reference, not one of the 10),
and a NoisyNet arm.

**`ensemble_mean` has no DeepSea config at all** — so the **C-USE contrast** (episodic_head vs.
capacity-matched ensemble-mean ε-greedy) currently has no instantiated comparison arm, and neither
does `per_step`, which is the temporal-coherence comparator.

</details>

---

## Step 6 — DeepSea integration, development sizes; pilot solve-vs-depth

Env, exact `q_star`, and both trainer lanes exist. Missing: the **pilot solve-vs-depth figure**
(`make_figures.py` has one generic `_plot_group` and no depth-axis figure). It needs real runs
across N, so it follows 5a rather than blocking it. Must carry a "pilot" label per spec.

---

## Step 7 — Randomized priors + `prior_scale` mini-search

**BUILT 2026-08-01** — `make search-prior-scale`. `run_mini_search(template, "prior_scale")` draws
`n_mini = 4` from log-uniform `[0.1, 10.0]` and routes them through the *same* `run_candidate`,
objective, pooling and selection path as the backbone search, so the three searches cannot diverge
in method. `prior_scale: 3.0` remains a declared placeholder in the configs until the pass runs.

> As predicted, "nearly free once 5a exists" — but the entry's claim that its objective **is**
> pinned was wrong in the same way step 8's was: the clause names a cell and a size set, not an
> outcome. Gap 4's sub-clause supplies the outcome for all three searches.

---

## Step 8 — NoisyNet + `eps_schedule` mini-search

> **CORRECTED 2026-08-01.** The "blocked harder than step 7" claim below is **no longer true and
> its reasoning was also incomplete**. `configs/cell_ensemble_mean_off_K10_deepsea_dev.yaml` was
> committed at `b0f63ef`, so the objective's input population exists and steps 7 and 8 are now
> **symmetric** — each needs only the `n_mini = 4` class-3 driver. Separately, the entry said this
> step's objective "is pinned"; Gap 4 later established that the `(mean_eps, off, 10)` clause names
> a cell and a size set but **never an outcome**, exactly like the backbone clause — so its
> objective was *not* pinned either, and it is the Gap 4 discovery-AUC sub-clause that supplies one.

**BUILT 2026-08-01** — `make search-eps-schedule`. Draws final ε from log-uniform `[0.005, 0.1]`;
`eps_decay_steps` is *derived*, not drawn (Gap 2 fixes decay at "the first 10% of the budget"), and
is computed per size as `0.10 × episodes × size`. That product is exact rather than an estimate:
every DeepSea episode runs to the bottom row, so an episode is exactly `size` steps — pinned by
`test_deep_sea_episode_length_is_exactly_size`, because if the env ever gains truncation the ε
schedule silently changes meaning. `config.step_budget` is *not* usable here; it reads
`env_budget.total_steps`, which only MinAtar runs carry.

> **A defect this step surfaced, now fixed.** `_build_bdqn` never read
> `factor_specific.eps_schedule`. Every `ensemble_mean` cell asking for `eps_decay_steps: 3000`
> silently ran `BDQNConfig`'s 10,000 default. Since `eps_schedule` is exactly what this
> mini-search tunes, the four candidates would have behaved identically and the search would have
> measured nothing but seed noise — while still emitting a confident winner. Fixed as the mirror of
> the `prior_scale` handling directly above it in `_build_bdqn`: `ensemble_mean` (the only use_rule
> whose acting consumes ε) now *requires* a schedule, and the others must leave it unset rather
> than carry an inert value. Four regression tests in `tests/test_config.py`.

<details>
<summary>Original entry as first written (2026-07-31)</summary>

NoisyNet is implemented with its own arm string. This step is blocked **harder than step 7**: its
pinned objective is "IQM of `(mean_eps, off, 10)` on development sizes", and per 5d that cell has no
config. The objective's input population does not exist. Fixing 5d is a prerequisite, not a parallel
task.

</details>

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
- ~~**The substrate is never called.**~~ **DONE 2026-08-01 (`0d6df0b`).** As predicted, this landed
  with 5c on the shared code path: `trainer.py` imports `diagnostics.recorder`, builds one per run
  under `--diagnostics`, calls `observe_state(obs)` each step for the visitation counts diagnostic
  #5 needs, and `record(step)` at every checkpoint. `{Q_m(s,a)}` is now collected, so the six
  missing diagnostics are **pure post-hoc analysis over committed `.npz` files** — none of them
  requires another training run or another trainer change. That is the main thing this table
  understates: the remaining step-9 work no longer touches the run path at all.

  <details><summary>Original entry (2026-07-31)</summary>

  `src/diagnostics/substrate.py` exists and is cap-agnostic, but grep finds no reference to it in
  `trainer.py`. Nothing collects `{Q_m(s,a)}` samples during a run, so no diagnostic can be computed
  even once written. **Wiring the substrate into the checkpoint path is the first step-9 task**, and
  it depends on 5c (disagreement logging) touching the same code.

  </details>

- **Diagnostic 8 remains hard-blocked, and it is the only step-9 item that is.** Freeze item 20's
  clone/restore conditional needs `MinAtarEnv` to expose state save/restore; grep finds no
  `clone`/`get_state`/`set_state` in `src/minatar_env.py`, so the conditional cannot be evaluated in
  any direction — including the direction that *drops* the analogue.

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
each. The backbone search's own share is **72 runs**; the two mini-searches are 24 each. No
discrepancy — the figure in the sign-off note is right, and step 5a's cost is 72 of it.

The **dev-cell** side of the total, however, was wrong here and everywhere it appeared, and is
corrected as of `b0f63ef`: it is **120**, not 110. The old count charged only ten arms because it
treated the ε-greedy DDQN reference `(episodic|off|K1)` as one of the ten factorial cells; it is
an eleventh arm outside the switchboard, so nine non-rule-input cells are charged at 2 sizes × 5
seeds and two rule-input arms at 10 + 5. Development tier total **120 + 120 = 240**, still inside
the 150–250 envelope but with 10 runs of slack rather than 20.

---

## Recommended order

> **Superseded 2026-08-01 — items 1–5 are all done** (`ce383ff` closes the last of them). Current
> order below; the original is kept underneath because the ordering *rationale* still holds.

**Setup (code) is complete for steps 5, 7 and 8 except one shared driver, and for step 9 except
post-hoc analysis. Nothing further is blocked on a decision.** What remains:

1. **Run the backbone tuning pass** — `make search-backbone`, 72 runs. Compute, not code. Produces
   the tuned DDQN backbone every later comparison is defined against, so it gates 6, 7 and 8.
2. ~~**Generalize the driver to the two class-3 mini-searches.**~~ **DONE 2026-08-01** —
   `make search-prior-scale`, `make search-eps-schedule`, `make search-all`. All three searches
   share one execution/objective/selection path.
3. **Step 6's solve-vs-depth figure** — needs the runs from (1), must carry a "pilot" label.
4. **Step 9's six missing diagnostics** — pure post-hoc analysis over the committed `.npz` files;
   no trainer change, no extra runs. Diagnostic 1 (marginal alignment) is the RQ2-L primary and
   should come first.
5. **MinAtar clone/restore** — the one genuinely missing *capability* in the tree, and the only
   hard blocker left. Needed to evaluate freeze item 20's conditional in any direction.
6. **Cut the final freeze tag**, land the staged protocol fixes, mirror to OSF, then pilot.

<details>
<summary>Original order as first written (2026-07-31) — all five items now complete</summary>

1. **Run step 4** on DeepSea N=10 — cheapest, and unblocks both compute triggers.
2. **Add the 6 missing cell configs** (5d) — pure YAML + schema check; unblocks step 8's objective.
3. **Stage the tuning-objective gap** (5b) as staged fix #4; get sign-off on discovery-AUC.
4. **Build the sweep driver** (5a) — the one artefact four steps wait on.
5. **Disagreement logging + substrate wiring** (5c + step 9 structural) — same code path.

</details>
6. **Run the backbone tuning pass**, then steps 7 and 8's mini-searches, then step 6's figure.
7. **Diagnostics 1–5, 7, 9**, then the item-20 clone conditional (8).

Steps 1–3 are hours and need no new science. Step 4 is the real build. Step 9 is the long tail and
can proceed in parallel with the tuning runs.
