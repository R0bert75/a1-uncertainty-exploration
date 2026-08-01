# Staged Stage-3 protocol fixes — proposed text, NOT yet in the pre-registration

**Drafted 2026-07-30. Corrected 2026-07-30 (second pass).
Owner sign-off 2026-07-30 — all recommendations below are APPROVED.**

Status: **approved, staged, not yet applied.** The values are settled; what remains is
mechanical application at stage 3. `preregistration.md` is deliberately still unchanged — see
the next section for why the sign-off does not license an edit today.

**Approved values:**

| Freeze item | Value | Status |
|---|---|---|
| 7 — probe set | Exhaustive, `\|S\| = N(N+1)/2`, no cap; no MinAtar probe set | approved |
| 2 — search budget | `n_backbone = 12`, `n_mini = 4` (each of two) | approved |
| 1 — seed counts | **3 seeds per tuning candidate** (new sub-clause, see Gap 3) | approved |
| 2 — backbone search *objective* | **IQM of per-seed discovery-curve AUC** (new sub-clause, see Gap 4) | ✅ approved 2026-08-01; extended by Fix #4 (exact formulation + tie-breaker + N=20 caveat) |

**All five gaps are now owner-decided.** Gap 4 was surfaced on 2026-07-31 and owner-approved
2026-08-01 (discovery AUC). Fix #4 (2026-08-01, from external reviewer) extends Gap 4 with the
exact mathematical formulation, tie-breaker cascade, and N=20 zero-stratum disclosure.
Fixes #5 and #6 (2026-08-01, from external reviewer) are staged and **require owner sign-off**.

## Why this file exists instead of an edit to `preregistration.md`

Freeze item 18 defines a three-stage sequence: **(1)** fill all freeze-list values →
`prereg-draft` tag; **(2)** external methodological pass *on the valued draft*; **(3)** fixes
→ final tag + OSF mirror. We are inside stage 2. The reviewer is reading `prereg-draft`
(`38db441`), and stage 3 is the designated channel for changes.

Editing `preregistration.md` now would mean the document a reviewer is reading is not the
document they were sent — which defeats the purpose of running the pass against a stable
reference. So the corrected text is drafted here, and applied to `preregistration.md` at
stage 3, alongside whatever the reviewer returns.

**This is not a discretionary deferral.** Both items below are *defects in the draft*: item 18
stage 1 requires all freeze-list values to be filled before the draft tag, and these two were
not. They must be disclosed to the reviewer now rather than quietly fixed, because a reviewer
assessing a "valued draft" is entitled to know which values were in fact missing. The
recommended action is to send both to the reviewer as an erratum against `prereg-draft`.

---

## Gap 1 — Freeze item 7: probe-set construction rule and |S|

### The defect

Item 7 currently reads:

> **7. Probe-set construction + weighting.** **Uniform weighting primary; visitation-weighted
> secondary.** Probe set S constructed per the frozen diagnostics spec; the `probe_set` RNG
> stream governs any sampling.

The **weighting** half is valued. The **construction** half delegates to "the frozen
diagnostics spec" — but §3.3 introduces `S` only as notation ("Notation: probe set S; M value
samples..."). No construction rule and no `|S|` appear in §3.3 or anywhere else in either
document. The cross-reference is dangling, so item 7 is only half-valued.

This matters beyond bookkeeping. Six of the nine frozen diagnostics are computed **over** `S`,
so `S` is the sample frame for every uncertainty-quality statistic in the study. Two
consequences: statistics are only comparable across runs if `S` is identical, and `|S|` sets
both the per-checkpoint diagnostic cost and the substrate storage volume (`|S| × M × A × 4`
bytes per checkpoint).

### Proposed replacement text

> **7. Probe-set construction + weighting.** **Uniform weighting primary; visitation-weighted
> secondary.** The probe set `S` is constructed **once per (environment, size) at run start**,
> before any learning, and held **fixed for the entire run** — every checkpoint's diagnostics
> are computed on the same states, so within-run trajectories are comparable and the
> `probe_set` stream is drawn exactly once per run.
>
> **Construction (DeepSea — exhaustive).** DeepSea's state space is the reachable lower-left
> triangle of the `N × N` grid, of size `N(N+1)/2` — 55 states at `N = 10`, 210 at `N = 20`,
> 1,275 at `N = 50`. The probe set is the **exhaustive reachable set**: `S = {all reachable
> (row, column)}`, `|S| = N(N+1)/2`, at every size. No sampling is performed and the `probe_set`
> stream is therefore **drawn zero times**; it remains defined so that sampling is available
> without a protocol amendment should a future size make enumeration impractical. Consequences:
> every diagnostic in §3.3 is an **exact** quantity rather than an estimate at every size, `|S|`
> is a deterministic function of `N` with nothing to pre-specify, and the probe set is
> method-independent trivially rather than by construction.
>
> **MinAtar.** No probe set. The §3.3 battery is defined against `Q*`, which is computable only
> on DeepSea; MinAtar's role is the RQ1 descriptive comparison on the two reporting axes. No
> uncertainty diagnostics are computed on MinAtar and none are reported.
>
> **Visitation counts.** The secondary visitation weighting and diagnostic §3.3 #5 both need
> `v(s)` per probe state. `v(s)` is accumulated **during the run** as the count of visits to each
> reachable DeepSea state (exact state match), recorded at each checkpoint alongside the value
> samples. It is a within-run quantity and cannot be reconstructed afterwards. Because the probe
> set is exhaustive, `v` is simply the run's full state-visitation histogram.

### Values that need owner sign-off

> **CORRECTED 2026-07-30 (second pass).** The first draft of this section proposed a *fixed
> cap* of `|S| = 256` for DeepSea and `|S| = 512` for MinAtar. Both were wrong when checked
> against the documents rather than against intuition. The corrected recommendation is
> **exhaustive enumeration on DeepSea and no MinAtar probe set at all.** The superseded
> reasoning is retained at the end of this section.

| Value | Proposed | Note |
|---|---|---|
| DeepSea `\|S\|` | **exhaustive: `\|S\| = N(N+1)/2`, no cap** | Every diagnostic exact at every size; ~1.35 GB for the whole confirmatory sweep |
| MinAtar `\|S\|` | **not applicable — no MinAtar probe set** | The battery needs `Q*`, which exists only on DeepSea |
| Sampling | Enumeration; the `probe_set` stream is a **no-op** | Method-independent by construction, trivially |

**On DeepSea — enumerate, do not cap.** Freeze item 7 says the `probe_set` stream "governs
**any** sampling" — permissive phrasing that is satisfied, not violated, by there being no
sampling to govern. So exhaustive enumeration is *already* compliant with the frozen text; no
cap has to be chosen at all. And it is affordable, which is the part the first draft never
checked. Reachable-set sizes are `N(N+1)/2`: 55 at `N = 10`, 210 at `N = 20`, up to 1,275 at
`N = 50` and 1,830 at `N = 60` (the largest size any confirmatory set can select). At the
worst case `M = 20` heads and `A = 2`, that is 199 KB per checkpoint at `N = 50` and 286 KB at
`N = 60` — and **1.35 GB for the entire default confirmatory sweep** (10 cells × 5 sizes × 20
seeds × ~10 checkpoints), 2.08 GB under the up-shifted set. That is not a budget that needs
managing on a workstation study whose replay buffers alone are 80–200 MB per run.

Against that, a fixed cap buys nothing and costs real precision. `|S| = 256` would probe only
**55% of reachable states at `N = 30`, falling to 20% at `N = 50`** — and those are the
confirmatory sizes, where RQ2-L is a **v1.0 submission-gate item** (spec §"v1.0 (submission
gate)": "RQ2-L per spec"). So the cap's only effect is to convert an exactly-computable
confirmatory statistic into a subsample estimate, to save on the order of one gigabyte. That is
the wrong trade in the wrong direction.

Enumeration also removes the item from the freeze list rather than filling it in, which is
strictly better under a pre-registration: **there is no `|S|` to pre-specify, no sampling
seed-dependence to audit, and no reviewer question about whether 256 was chosen after seeing
data.** The `probe_set` stream stays in the protocol text so that the *option* of sampling
remains available if a future size makes enumeration impractical, but it is drawn zero times in
this study.

**On MinAtar — the first draft invented a requirement.** Every diagnostic in §3.3 is defined
against `Q*` (marginal alignment, action-gap alignment, and the rest all reference
`|Q̄ − Q*|`), and `Q*` is computable only on DeepSea. RQ2-M is scoped "(mechanism, DeepSea,
confirmatory)" and RQ2-L is a DeepSea statistic; **`grep` for a MinAtar diagnostics or
probe-set clause across both frozen documents returns nothing.** MinAtar's role is RQ1
descriptive performance on the two reporting axes. So there is no MinAtar probe set to size,
and the first draft's 512 was answering a question the protocol never asked. Removing it also
deletes the "uniform-random-policy reference set" construction, which was the most
implementation-heavy and least specified part of the original proposal.

<details>
<summary>Superseded first-draft reasoning (kept for provenance)</summary>

The first draft argued for `|S| = 256` over 128 on the grounds that 128 makes the two
development sizes straddle the exhaustive threshold (`N = 10` exact at 55 states, `N = 20`
subsampled at 128-of-210), whereas 256 puts the threshold at `N = 22` and makes both dev sizes
exhaustive — "paying 20 KB/checkpoint for exactness where the Part-A mechanism claims are
made." That reasoning was *directionally* right and stopped one step short: if exactness is
worth 20 KB at the development sizes, it is worth ~200 KB at the confirmatory sizes, where the
submission-gate statistic lives. Once the total was actually priced (1.35 GB, not per-checkpoint
KB), the cap had no remaining justification. The draft also asserted a confirmatory reachable-set
sequence of "465 / 561 / 666 / 780 / 903" for `{30,35,40,45,50}`; the correct values are
**465 / 630 / 820 / 1035 / 1275**. The first three of those five numbers were wrong.

</details>

---

## Gap 2 — Freeze item 2: search distributions and the backbone-tuning budget

### The defect

Item 2 is titled "**Search distributions per hyperparameter; backbone-tuning budget; two
factor-specific mini-budgets**". The mini-searches' *selection objectives* and *tie-breakers*
are fully valued (IQM of the named cells; ties → lower parameter value). The **distributions**
and the **budget count** are not. Item 2 says they are "frozen in the parameter table below";
the parameter table's Class-1 entry says they are "frozen with the pipeline config"; and the
committed configs contain no search distributions at all. The three references form a cycle
with no values in it.

Unlike gap 1, this one is load-bearing for the *equal-search-budget* claim (C-iii, spec §"Equal
search budget"), which is stated as an inherited rigor standard: "identical tiers/counts/statistic
across the canonical four". The claim is not checkable unless the count is written down.

### Proposed replacement text

> **2. Search distributions per hyperparameter; backbone-tuning budget; two factor-specific
> mini-budgets with pinned selection objectives and tie-breakers.** Backbone nuisance (class 1)
> tuned once on the ε-greedy DDQN backbone over development sizes by **random search with
> `n_backbone = 12` draws**, each evaluated at **3 seeds on each of the two development sizes**
> (6 runs per candidate), selected by IQM per item 3. Distributions (drawn with the
> `hparam_search` stream):
>
> | Parameter | Distribution |
> |---|---|
> | learning rate | log-uniform `[1e-4, 1e-2]` |
> | batch size | uniform over `{32, 64, 128}` |
> | target-update cadence (steps) | uniform over `{100, 500, 1000}` |
> | network width (FC units) | uniform over `{64, 128, 256}` |
> | replay capacity | **fixed at 100,000** (not searched; memory-bound) |
> | optimizer | **fixed at Adam** (not searched) |
>
> Two factor-specific mini-searches (class 3), **`n_mini = 4` draws each** at the same 3 seeds ×
> 2 development sizes, identical count for
> every method so the equal-search-budget standard holds: **`prior_scale`** — log-uniform
> `[0.1, 10.0]`, selected by IQM of the canonical prior-on cell `(episodic, on, 10)` on
> development sizes, value shared by all `prior=on` cells; **`eps_schedule`** for
> `ensemble_mean_eps` — final-ε log-uniform `[0.005, 0.1]` with linear decay over the first 10%
> of the budget, selected by IQM of `(mean_eps, off, 10)` on development sizes, shared by its
> cells at both prior levels. **Ties broken by the lower parameter value.** Nothing else is
> tunable per cell.

### Consistency check against the frozen run budget

> **CORRECTED 2026-07-30 (second pass).** The first draft justified `n_backbone = 24` with the
> arithmetic `192 + 32 + 12 = 236 ≈ 240`, where 192 was "4 methods × 2 tuning games × 24
> draws". **That table was wrong in three independent ways** and its agreement with 240 was a
> coincidence of compensating errors. The corrections:
>
> 1. **The backbone is not tuned per method.** Spec §"Parameter classes" item 1 and prereg item
>    2 both say backbone nuisance is "tuned **once** on the ε-greedy DDQN backbone… inherited
>    identically by all cells". The factor of 4 was fabricated. Tuning it per method would in
>    fact **violate** the equal-search-budget standard this section is supposed to protect.
> 2. **The 240 counts runs, not configs.** Spec §3.4's budget list is in runs throughout
>    ("≈ 240 pilot + 800 final", "≈ 120 runs", "≈ 1,100"). MinAtar pilot seeds are **5** (spec
>    §Seeds: "3 dev → 5 pilot → 10 held-out"), so 240 runs = **48 configs**.
> 3. **The backbone search does not run on MinAtar.** Prereg item 2 says it is tuned "over
>    **development sizes**" — i.e. DeepSea `N ∈ {10, 20}`. It is therefore charged to the
>    **DeepSea dev** budget (≈ 150–250 runs), not to MinAtar's 240.

**MinAtar side — what the 240 actually implies.** The only searches charged to MinAtar's pilot
tier are the `K_shared` sweep and the per-game backbone confirmation:

| Block | Configs | Runs (× 5 pilot seeds) |
|---|---|---|
| `K_shared`: 2 ensemble methods × `K ∈ {5,10,20}` × 2 tuning games | 12 | 60 |
| Remaining pilot capacity | 36 | 180 |
| **Total** | **48** | **240** |

36 remaining configs over 2 tuning games is **18 per game**. So if a MinAtar-side backbone
confirmation pass is run at all, the frozen 240 implies `n_backbone_minatar = 18` — **exactly**,
not approximately.

**DeepSea side — where the class-1 search is actually charged.**

> **CORRECTED (second pass), 2026-08-01.** The dev-cell subtotal below was **110** and is
> **120**. The error: the old expression `8 cells × 2 sizes × 5 seeds + 2 rule-input cells ×
> (10 + 5)` treated the ε-greedy DDQN reference as one of the ten factorial cells. It is not
> — the reference arm is `(episodic|off|K1)`, which is outside the switchboard (confirmed
> when the eight missing cell configs were written, main `b0f63ef`). So the reference is an
> **eleventh** arm charged on top of the ten, and the count of non-rule-input factorial cells
> is **9**, not 8. The recommendation is unchanged, but the slack is half what this section
> previously claimed.

The ten factorial cells plus the reference arm consume
`9 non-rule-input cells × 2 sizes × 5 seeds + 2 rule-input arms × (10 + 5) seeds`
= `90 + 30` = **120 runs** of the 150–250 dev budget, leaving **30–130 runs for all tuning**.
At 3 seeds × 2 dev sizes = 6 runs per candidate:

| `n_backbone` | `n_mini` (each of 2) | Candidates | Runs | Dev total | Verdict |
|---|---|---|---|---|---|
| 8 | 3 | 14 | 84 | 204 | fits comfortably |
| **12** | **4** | **20** | **120** | **240** | **fits; recommended (10 runs of slack)** |
| 24 | 8 | 40 | 240 | 360 | **44% over the ceiling** |

At 5 seeds per candidate nothing fits: even `n_backbone = 8, n_mini = 3` costs 140 tuning runs
for a dev total of 260, over the ceiling. Under the corrected count the 3-seeds-per-candidate
rule (freeze item 1, owner-approved) is not merely preferable but **load-bearing**.

<details>
<summary>Superseded first-pass reasoning (retained for provenance)</summary>

The first pass wrote: *"The 10 dev cells consume `8 cells × 2 sizes × 5 seeds + 2 rule-input
cells × (10 + 5) seeds` = 110 runs … leaving 40–140 runs for all tuning"*, and tabulated dev
totals of 194 / 230 / 350 with the note that at 5 seeds per candidate `n_backbone = 8,
n_mini = 3` still fits at exactly 250. Every one of those figures is 10 runs low, and the
5-seed claim was wrong outright: it fits only if the reference arm is not charged. The
`8 + 2 = 10` decomposition happens to sum to the same 110 as a genuine ten-cell count would,
which is why the error survived the first arithmetic check.

</details>

**So the first draft's `n_backbone = 24` was not "nearly implied by the budget" — it overspends
the budget it was charged against by 40%.** The corrected recommendation is `n_backbone = 12`,
`n_mini = 4`, at **3 seeds per tuning candidate**, which lands the DeepSea dev tier at 240 runs
inside its 150–250 envelope. Note that the tuning seed count is itself a value freeze item 1
should state and does not; it is added to the proposed text below.

If the owner prefers a larger search, the honest move is to raise §3.4's DeepSea dev budget
explicitly rather than let the search silently exceed it — and cap X's arithmetic (Σ over
methods of final-tier count × median pilot wall-clock ≤ X GPU-h) should be re-checked either
way, since pilot wall-clock is its input.

---

## Gap 3 — Freeze item 1: the per-candidate tuning seed count (surfaced by the Gap 2 audit)

### The defect

Freeze item 1 pins seed counts for every *tier*: DeepSea development (10 for the two rule-input
runs, 5 elsewhere), DeepSea confirmatory (20 per cell), MinAtar (3 dev → 5 pilot → 10 held-out).
It has **no category for a tuning candidate.** A random-search draw is not a development cell and
not a pilot run; it is a third thing the item does not name.

This was invisible until Gap 2 was recomputed, because the first draft never priced the search in
runs. It matters now for a concrete reason: the seed count is a **multiplier on the entire tuning
budget**, so item 2's `n_backbone` is not even well-defined without it. At 3 seeds the approved
`n_backbone = 12` costs 120 runs; at 5 it costs 200 and breaks the ceiling. Leaving it unstated
would mean pre-registering a search budget whose actual cost is unspecified — the same class of
defect as items 2 and 7.

### Proposed addition to item 1

> **Tuning-candidate seeds.** Each random-search candidate (backbone and both factor-specific
> mini-searches) is evaluated at **3 seeds on each of the two development sizes** — 6 runs per
> candidate. Selection is by IQM over those 6 runs per item 3. This count is deliberately lower
> than the 5 used for development *cells*: a tuning candidate's IQM feeds a selection decision,
> not a reported estimand, and no confidence interval is published for it. The winning
> configuration is then run at the full development seed count as a normal cell.

### Consequence for the run budget

`(12 + 2 × 4) × 6 = 120` tuning runs; with the 120 dev-cell runs (see the corrected count above
— the ε-greedy DDQN reference is an eleventh arm, not one of the ten cells) the DeepSea
development tier totals **240**, inside its frozen ≈ 150–250 envelope with 10 runs of slack.
No other frozen count moves.

---

## Gap 4 — Freeze item 2: no search names its objective *outcome*

**Surfaced 2026-07-31 by the steps 4–9 build audit (`audits/steps_4_9_build_plan.md`).**
**Scope corrected and resolved 2026-08-01 (owner-decided).**
Status: **approved, staged, not yet applied.**

**Owner decision (2026-08-01): discovery AUC, applied to all three searches.** The proposed
sub-clause below is the text to apply at stage 3. Rationale and the evidence behind it follow.

### The defect

Freeze item 2 pins selection objectives for three searches — the class-1 backbone search and the two
class-3 mini-searches (`prior_scale`, `eps_schedule`) — and item 3 pins the statistic (IQM) and
tie-break (lower parameter value) "throughout". IQM is a *reduction*; IQM **of what** is nowhere
stated. Two candidates could be ranked by terminal discovery probability, by area under the discovery
curve, by mean episode return, or by episodes-to-first-discovery, all of which are "IQM" and all of
which give different winners.

**Scope correction (2026-08-01).** As first written, this entry asserted that item 2 "pins a full
objective for both class-3 mini-searches" and that only the class-1 search was defective. That is
wrong, and the error was in reading a *cell* specification as an *outcome* specification. Grepping
every `IQM of …` occurrence across both `preregistration.md` and the v6.3 requirements document
returns six clauses; each names a cell and a size set, and **none names an outcome**:

| clause | cell pinned | sizes pinned | **outcome pinned** |
|---|---|---|---|
| class-1 backbone (item 2) | — (the ε-greedy DDQN backbone) | ✓ dev | **✗** |
| class-3 `prior_scale` (item 2, param table Class 3) | ✓ `(episodic, on, 10)` | ✓ dev | **✗** |
| class-3 `eps_schedule` (item 2, param table Class 3) | ✓ `(mean_eps, off, 10)` | ✓ dev | **✗** |

Item 2 pins *more* of the class-3 objectives than the class-1 one, not all of it. Since both class-3
mini-searches are DeepSea searches at the same 3 seeds × 2 development sizes = 6 runs per candidate,
they inherit the identical tie pathology described below. Fixing class-1 alone would leave the freeze
item carrying two different selection outcomes with only one of them stated — the harder thing to
defend at review than the original gap. **The sub-clause therefore governs all three searches.**

### Why it is not cosmetic

DeepSea's primary outcome is **binary per seed** (`discovered`, `src/trainer.py`). At the approved
3 seeds × 2 development sizes = 6 runs per candidate, IQM of a 6-vector of 0/1 takes only **five
distinct values** — it collapses 0/6 with 1/6, and 5/6 with 6/6:

| successes k | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| IQM | 0.000 | **0.000** | 0.167 | 0.500 | 0.833 | **1.000** | 1.000 |

That table is not a hand calculation: it is `src/selection.iqm` — the frozen implementation that will
actually score the search — evaluated on all seven 6-vectors. Its fractional-rank weighting does not
rescue the binary case, because with n = 6 the two boundary observations carry weight 0.5 each and
the middle four are all identical whenever k ∈ {0,1} or k ∈ {5,6}.

Item 3's tie-break is "the lower parameter value", which is uncorrelated with performance — so in the
regime the search is *supposed* to resolve, selection is close to arbitrary.

### Recommendation: area under the discovery curve

`trainer.py` logs `discovery_prob` at **every checkpoint** as a *cumulative* indicator
(`float(discovered)`, monotone 0 → 1), so the per-seed mean of that curve is recoverable from the
existing CSV with no new instrumentation and no re-run — including retroactively, on tuning runs
already executed. Two consequences follow from the cumulative encoding:

* **It is continuous** where the terminal indicator is five-valued.
* **It is exactly a time-to-discovery measure.** If discovery happens at checkpoint *j* of *C*, then
  AUC = 1 − *j*/*C* identically, strictly decreasing in *j*, with never-discovered mapping to 0 by
  construction. So this objective already carries the information the episodes-to-first-discovery
  alternative would supply, *without* needing the censoring convention that alternative would require
  the protocol to invent and defend for never-discovering seeds.

Simulation, 12-candidate field, 3 seeds × 2 development sizes = 6 runs per candidate, 20 checkpoints,
scored through `src/selection.iqm`. Candidate quality is a per-checkpoint discovery hazard *h* with
terminal p = 1 − (1 − *h*)^C, so a better candidate also discovers earlier — the coupling a backbone
search actually faces. Regret = (true p of the best candidate) − (true p of the selected candidate):

| regime (true discovery prob) | objective | P(tie at top) | P(pick true best) | mean regret |
|---|---|---|---|---|
| hard (p ∈ 0.00–0.25) | terminal binary | 0.442 | 0.214 | 0.051 |
| hard | **discovery AUC** | **0.105** | 0.212 | 0.051 |
| mixed (p ∈ 0.05–0.60) | terminal binary | 0.433 | 0.297 | 0.077 |
| mixed | **discovery AUC** | **0.011** | 0.286 | 0.080 |
| easy (p ∈ 0.40–0.90) | terminal binary | 0.976 | 0.203 | 0.112 |
| easy | **discovery AUC** | **0.024** | **0.295** | **0.078** |

AUC weakly dominates: the tie rate collapses everywhere, accuracy and regret are within noise in the
hard and mixed regimes, and in the easy regime — the one a *well-tuned* search is most likely to
produce, and the one where terminal-binary ties 98 % of the time — it picks the true best 0.295 vs
0.203 of the time at two-thirds the regret.

<details>
<summary>Superseded: the accuracy column as first tabulated (2026-07-31) — retained for provenance</summary>

The original table reported P(pick true best) of 0.470 / 0.587 / 0.439 for terminal binary against
0.466 / 0.587 / **0.610** for AUC. Those absolute levels do not reproduce, and the AUC advantage they
showed in the easy regime was real but arrived at through a model that could not have produced it
honestly: **the original simulation drew each run's discovery *time* independently of the candidate's
quality.** Under that model a better candidate discovers no earlier than a worse one, so AUC carries
no signal terminal-binary lacks and can only break ties by noise — which is exactly what the
recomputation showed (identical mean regret, 0.052 vs 0.053, in every regime).

Correcting the model to couple discovery time to quality through a per-checkpoint hazard is what
produces the honest advantage now tabulated above. The tie-rate column, which is the load-bearing
one, reproduced within Monte-Carlo error throughout (0.44 / 0.43 / 0.98 against the original
0.44 / 0.45 / 0.98) and was never in doubt. **The recommendation is unchanged; its justification is
now the tie rate plus a smaller, real accuracy gain in the easy regime, rather than the large
accuracy gain originally claimed.**
</details>

### Proposed item 2 sub-clause (owner-approved 2026-08-01)

> **Search objective (all pre-registered searches).** The selection objective for the class-1
> backbone search and for both class-3 mini-searches is the **IQM across tuning seeds of the per-seed
> area under the online discovery-probability curve** — the unweighted mean of the `discovery_prob`
> metric over the run's checkpoints — pooled across the two development sizes. The cell and size set
> for each search are as already stated in this item; this sub-clause supplies only the outcome that
> IQM reduces, which was previously unstated for all three. Ties are broken by the lower parameter
> value (item 3). This objective is a **selection input, not a reported estimand**: the primary
> outcome for all reported results remains terminal discovery probability (§1.1), unchanged.

### Why this must be pre-specified rather than chosen at run time

Both objectives are computable from the same logs. Choosing between them after seeing the search
results would be a post-hoc degree of freedom over the backbone that every cell in the study
inherits — the single largest such freedom in the design. It costs nothing to fix now and cannot be
fixed later.

**Blocks:** step 5a (sweep driver) — which metric it reduces. *As first written this entry added
"does not block steps 7 or 8, whose objectives are already pinned"; the scope correction above
retracts that — steps 7 and 8 are the two class-3 mini-searches and their outcome was equally
unstated, so they were blocked on this decision too. All three are unblocked by the one sub-clause.*

---

### New RNG stream required

The distributions above need a `hparam_search` stream. `STREAM_NAMES` in
`src/utils/conventions.py` does not have one. Adding a stream name is explicitly a safe
operation there ("names, not positions, key the derivation"), so nothing already derived shifts —
but it is a code change that must land with this text, and the stream-registry test needs its
expected tuple updated.

---

## Gap 5 — Freeze item 5: the DeepSea per-size episode budget was never committed

**Surfaced 2026-08-01, by sizing the tuning sweep before launching it.**

Item 5 says: "Episode budget per N: frozen per size (the pre-registered DeepSea episode budget;
the size-scaled budget committed with the pipeline)." The clause defers the actual numbers to a
mapping committed alongside the code — and **no such mapping was ever committed.** `grep` finds no
size→episodes table anywhere in the tree. What exists is a convention nobody wrote down: all eight
committed factorial cells use 2000 episodes at N=30, and the two example configs use 500 at N=8.

This is load-bearing for more than the sweep. Four things read this quantity:

1. the three tuning searches (`src/search.py`),
2. item 6's reporting window ("DeepSea reported at the frozen per-size episode budget"),
3. item 11's t₀ landmark ("t₀ = 10% of the episode budget"),
4. the class-3 ε-decay derivation, which is 10% of the budget in steps.

**How it surfaced.** `src/search.py`'s materializers override `deep_sea_size` but inherited
`episodes` from whichever template each search points at. Since the three searches point at
*different* templates, the backbone would have been tuned at **500 episodes** and the two
mini-searches at **2000** — a 4× asymmetry, in exactly the equal-search-budget standard (C-iii)
that this document's Gap 2 consistency check exists to protect. Neither figure was size-scaled.

### Proposed replacement text for item 5's budget sentence

> **Episode budget per N: 2000 episodes at every N**, constant in N, for all DeepSea runs —
> development and confirmatory sizes alike, tuning and evaluation alike. Env *steps* scale with
> N (a DeepSea episode is exactly N steps, so the step budget is 2000·N), which is the sense in
> which the budget is size-scaled; the *episode* count is not.

**Why constant episodes and not constant steps** (owner decision 2026-08-01). One episode is one
attempt at the treasure. Holding attempts fixed across N is what makes ε-greedy's failure onset
with depth a property of *the method* rather than of the budget. Under a constant-*step* budget
the small sizes receive proportionally more attempts — 6000 episodes at N=10 against 2000 at
N=30 — which flatters ε-greedy at precisely the sizes where its failure is the claim (§ item 5's
own rationale cites "published ε-greedy failure onset near N≈10–15"). A linear-in-N reading was
also rejected: 667 episodes at N=10 risks near-zero discovery for every method, so the
development sizes would stop discriminating between tuning candidates.

**2000 is not a new number.** It is what the eight committed cells already use, so no committed
config changes and the tuning runs match the runs they tune for.

**Wording note.** This *narrows* item 5 rather than filling a blank: the phrase "the size-scaled
budget" must be amended, since on this resolution the episode count is explicitly not size-scaled.
That is why this is a staged protocol fix and not merely a committed constant.

**Landed in code as** `config.DEEP_SEA_EPISODE_BUDGET = 2000`, applied by both search
materializers and asserted against the committed cells by
`test_committed_deepsea_cells_agree_with_the_pinned_budget`. The materializers accept an
`episodes` override used **only** by tests that need a sweep finishing in seconds; a guard test
asserts no protocol entry point (including the CLI) exposes it.

**Blocks:** nothing that is not already blocked — but it silently *mis-specified* every tuning
run, so it had to land before the sweep, not after.

---

## Fix #4 — Backbone tuning objective: exact formulation (external review feedback, 2026-08-01)

**Incorporates: external reviewer feedback 2026-08-01. Owner sign-off required before this is
applied at stage 3.**

The external reviewer confirmed that the Gap 4 sub-clause (owner-decided 2026-08-01) is
correct in its discovery-AUC choice, but identified three further requirements before the text
is complete:

1. **The exact per-run quantity must be defined mathematically**, not just named.
2. **A tie-breaker cascade must be specified** (even at 3–11% tie rate, any tie that occurs
   must resolve mechanically, not by researcher judgment post-results).
3. **The N=20 zero-stratum situation must be documented** — this is no longer hypothetical.
   The completed backbone sweep (2026-08-01, `logs/search/backbone/search_record.json`) showed
   all 12 candidates returning 0/3 discovery at N=20. The winner (c01) was therefore selected
   on the N=10 stratum alone.
4. **The sub-clause must cover all three searches**, not just the backbone (Gap 4 already
   resolved this; the reviewer confirmed the same pathology applies to both class-3 searches).

### Proposed replacement sub-clause for item 2 (supersedes the sub-clause in Gap 4 above)

> **Search objective (all pre-registered searches).** The selection objective for the class-1
> backbone search and for both class-3 mini-searches is defined as follows.
>
> **Per-run score.** For a run r with budget B_N and cumulative first-discovery indicator
> D_r(t) = 1(T_r ≤ t), the normalized discovery AUC is:
>
>     A_r = (1/B_N) ∫₀^{B_N} D_r(t) dt
>
> In the discrete checkpoint case (C checkpoints at equal spacing):
>
>     A_r = (1 − j/C)  if discovery occurs at checkpoint j  (j ∈ {1, …, C})
>     A_r = 0           if no discovery occurs within the budget
>
> where j is the index of the first checkpoint at which `discovery_prob = 1`. Earlier discovery
> yields a higher score; a run that never discovers scores 0. This is equivalent to (1 − T_r /
> B_N) normalized to [0, 1], which is a direct measure of sample efficiency.
>
> **Candidate score.** The candidate score is the IQM of {A_r} over the six development runs
> (3 seeds × 2 development sizes). Both sizes contribute exactly three runs each; because the
> per-run scores are normalized to [0, 1] and the counts are equal, this is already
> stratum-balanced without an explicit weighting step.
>
> **Tie-breaker cascade.** If two candidates share the same IQM to floating-point precision:
> (1) higher terminal discovery rate (fraction of the 6 runs with T_r ≤ B_N); (2) lower
> primary hyperparameter value (the varied parameter for class-3 searches; frozen candidate
> index for the class-1 backbone search, whose parameters have no natural order). This cascade
> is purely mechanical and requires no researcher judgment after results are observed.
>
> **Zero-stratum behaviour.** If all runs at one development size score 0 (no discovery in any
> of the three seeds at that size), the pooled IQM is computed over the full six-run vector as
> stated; the uninformative stratum contributes zeros and does not receive special treatment.
> This situation occurred in the completed backbone sweep (N=20, all 12 candidates, 0/3
> discovery), so the selected backbone configuration (c01: lr=6.054e-4, batch_size=128,
> hidden_width=64, target_update_period=100; IQM=0.1583) was chosen on the N=10 stratum alone.
> This is a substantive caveat: any cross-size comparison built on this backbone should note
> that the backbone was never observed to discover the goal at N=20 during tuning. The selection
> rule is unchanged; this note is disclosure, not a correction.
>
> **What this objective is not.** This is a selection input, not a reported estimand. The
> primary outcome for all reported results remains terminal discovery probability (§1.1),
> unchanged. The tuning objective must not be used as a confirmatory outcome or as a diagnostic
> input — doing so would partially optimize the estimator against the quantity it is later used
> to study (§3.3 uncertainty diagnostics reference the same logs).

### Reviewer's staging note

The reviewer noted that staged fixes must not become a hidden intermediate layer that an
external reviewer never saw. **Fix #4 is a methodological decision, not a typo fix**, and the
final freeze candidate should be presented to whoever closes Gate 1 (substitute reviewer or
waiver process) as the explicit composite:

    prereg-draft (38db441)
    + staged fixes #1–#6
    = candidate final preregistration

Before cutting the final tag, this diff should be disclosed as a formal erratum against
`prereg-draft`, so that the Gate-1 record shows the reviewer (or waiver) was shown the full
candidate text, not just the original draft.

---

## Fix #5 — Probe-set details: three quantities not frozen (external review feedback, 2026-08-01)

**Incorporates: external reviewer feedback 2026-08-01. Owner sign-off required.**

The reviewer identified three probe-set details that are not explicitly frozen anywhere in the
current documents, despite being load-bearing for the §3.3 diagnostic battery:

### The three unresolved details

**5a. Terminal-state inclusion.** DeepSea's reachable lower-left triangle includes the
goal cell (row N−1, col N−1). It is a terminal state: the episode ends on entering it, no
action is taken from it, and Q*(s_terminal, a) = r_terminal / (1−γ) for all a under
standard conventions. The diagnostic battery applies Q*-based alignment measures over S; the
question is whether s_terminal ∈ S and whether the alignment is meaningful there (where the
agent has no policy to observe). **Must be decided before `analysis/diagnostics_battery.py` is
run on real data.**

**5b. S × A scope.** Six diagnostics (marginal alignment, action-gap alignment,
incorrect-argmax rank, empirical containment, optimal-path σ, visitation-conditioned decay)
compute a statistic over pairs (s, a). DeepSea's branching factor is 2 at every non-terminal
state (left/right), but the action space has A=2 regardless of state. The question is whether
diagnostics iterate over all (s, a) with s ∈ S, or only over (s, a*) where a* = argmax Q*(s),
or only over reachable (s, a) pairs (excluding actions that transition outside the grid). This
determines |S × A| and the interpretation of every alignment coefficient.

**5c. Q* mapped to the run's action permutation.** DeepSea's action labels (0/1 = left/right)
are assigned by a per-run permutation drawn from the `env_mapping` stream and stored as a hash
in the run record. The brute-force Q* solver produces Q* values indexed by the solver's own
action ordering, not necessarily by the run's permutation. If the hash is not stored alongside
the diagnostic data and the mapping is not applied, Q*-based alignment diagnostics silently
compute Spearman ρ between the model's value-function ordering and the *wrong* Q* column.
Given the cell-specific RNG derivation scheme, this is a real risk: the per-run action
mapping is non-trivial across cells.

### Proposed additions to item 7 (to be staged and applied at stage 3)

> **Probe-set details (three frozen quantities for the §3.3 battery).**
>
> (a) **Terminal states:** s_terminal is **excluded** from the probe set. The agent takes no
> action from s_terminal (the episode terminates before the policy is queried), so σ(s,a) is
> not defined there and Q*-alignment diagnostics have no behavioral referent. The probe set is
> the strictly non-terminal reachable states: S = {(row, col) : row + col < N−1}, giving
> |S| = N(N+1)/2 − 1 states. (The exhaustive-enumeration rule in Gap 1 is otherwise
> unchanged; this clarification reduces |S| by one entry per size.)
>
> (b) **S × A scope:** diagnostics iterate over **all (s, a) pairs with s ∈ S and a ∈ {0,1}**
> (full Cartesian product, both actions per state). This is the natural scope for a measure
> of uncertainty quality: the diagnostic asks whether the model's uncertainty is well-calibrated
> globally, not just along the optimal path.
>
> (c) **Action-mapping provenance:** the `env_mapping` hash stored in the run's
> `resolved_config.json` is recorded alongside every diagnostic output file. Before computing
> any Q*-based alignment statistic, the diagnostic driver reads the hash, retrieves the
> permutation from the run's `env_mapping` stream seed, and applies it to index into Q*
> correctly. A guard assertion checks that the loaded Q* array's argmax pattern matches the
> permuted reference before any correlation is computed.

---

## Fix #6 — Diagnostic 8: MinAtar state-selection rule unresolved (external review feedback, 2026-08-01)

**Incorporates: external reviewer feedback 2026-08-01. Owner sign-off required.**

The reviewer identified a gap that was created by resolving a different gap. Freeze item 20's
conditional (which MinAtar Diagnostic 8 probe-rollout variant to run) was resolved to "full
probe rollouts" at commit `288834b`. The `MinAtarEnv.clone_state()` / `.restore_state()`
implementation that unblocked it is committed and tested. But:

**Closing "which variant" did not close "from which states."**

Diagnostic 8 is the MinAtar behavior-policy analogue: 100 clone/restore reproduction tests
whose outcome decides between full probe rollouts, episode-start-only, or dropping the
analogue. "Full probe rollouts" means: restore a state, roll out the behavior policy, measure
whether the trajectory is bit-exact. But which states are restored? The item-20 conditional
names the variant; it says nothing about the state population.

At the time item 20 was written, MinAtar was known to have no clone/restore. Now that it does,
the state-selection question surfaces as a genuine open item that was never asked.

### The unresolved question

Candidate state populations for Diagnostic 8:

- **On-policy encountered states**: states visited during a training run, sampled from the
  replay buffer or a logged trajectory snapshot. Depends on the policy; varies across seeds and
  checkpoints.
- **Episode-start states**: states at the beginning of each episode (`env.reset()`). For
  MinAtar these are fully deterministic given the seed, so they are reproducible.
- **Fixed evaluation trajectories**: a frozen set of (env, seed) pairs that define starting
  configurations, evaluated by rolling out the behavior policy from each.
- **Uniform snapshots at fixed step intervals**: save every K steps during training, restore
  and roll out from each.

The choice determines what "full probe rollouts" means in practice, how reproducible the
diagnostic is across implementations, and how it relates to the DeepSea probe set (which
is exhaustive over reachable states).

### Status

**This fix requires owner decision before Diagnostic 8 is implemented.** The implementation
in `src/diagnostics/` should not begin until the state-selection rule is written here and
signed off. The recommended approach, consistent with the spirit of the DeepSea battery
(exhaustive where feasible, probe-set-stream governed otherwise), is:

> **Diagnostic 8 state population:** episode-start states from the first 100 distinct seeds
> applied to the MinAtar `env.reset()` call, drawn from the `probe_set` stream. This is
> reproducible, independent of any learned policy, and directly analogous to the DeepSea
> probe set's goal of sampling the reachable state distribution representatively. The 100
> rollouts mandated by item 20 correspond to these 100 states.

**Owner decision needed: accept this recommendation, or choose a different state population
and document the rationale here before implementation begins.**

---



---

## Recommended sequence

1. ~~Owner signs off on the corrected recommendations.~~ **DONE 2026-07-30 — approved.** For the
   record, what was approved: **`|S|` exhaustive / no MinAtar probe set**, **`n_backbone = 12`,
   `n_mini = 4`**, and **3 seeds per tuning candidate** (Gap 3). Detail retained below:
   - **DeepSea `|S|`: exhaustive, no cap.** Not a value at all — `|S| = N(N+1)/2` follows from
     `N`. Compliant with item 7 as frozen (which permits rather than requires sampling), exact at
     every size including the confirmatory sizes where RQ2-L is a submission-gate deliverable,
     and ~1.35 GB for the whole sweep. **Nothing to pre-specify and nothing for a reviewer to
     question.** MinAtar `|S|` does not exist — the battery needs `Q*`.
   - **`n_backbone = 12`, `n_mini = 4`, 3 seeds per candidate** — 120 tuning runs, landing the
     DeepSea dev tier at 240 of its 150–250 envelope. The first draft's 24 overspent it by 44%.
     If a MinAtar-side backbone confirmation is also run, the frozen 240 pilot runs imply
     **18 draws per tuning game** exactly.
   The one genuinely discretionary item left is the **tuning seed count** (3 recommended; 5 forces
   `n_backbone` down to 8), which freeze item 1 should state and currently does not.
2. Send both gaps to the reviewer as an **erratum against `prereg-draft`** — they are missing
   stage-1 values, and disclosing them is cheaper than having the reviewer find them.
3. At stage 3, apply **all six** gap texts to `preregistration.md` (Gaps 1–5 plus Fix #4's extended formulation; Fixes #5 and #6 once owner sign-off received), add the `hparam_search`
   stream, and cut the final tag + OSF mirror. (Gap 5 amends item 5's wording rather than filling
   a blank — it must not be applied as a pure insertion.)

Until step 3, `preregistration.md` is unchanged and `prereg-draft` remains a valid stable
reference for the pass in progress.
