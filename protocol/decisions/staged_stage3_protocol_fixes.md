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

**DeepSea side — where the class-1 search is actually charged.** The 10 dev cells consume
`8 cells × 2 sizes × 5 seeds + 2 rule-input cells × (10 + 5) seeds` = **110 runs** of the
150–250 dev budget, leaving **40–140 runs for all tuning**. At 3 seeds × 2 dev sizes = 6 runs
per candidate:

| `n_backbone` | `n_mini` (each of 2) | Candidates | Runs | Dev total | Verdict |
|---|---|---|---|---|---|
| 8 | 3 | 14 | 84 | 194 | fits comfortably |
| **12** | **4** | **20** | **120** | **230** | **fits; recommended** |
| 24 | 8 | 40 | 240 | 350 | **40% over the ceiling** |

At 5 seeds per candidate the ceiling binds much harder: only `n_backbone = 8, n_mini = 3` fits
(140 runs, dev total exactly 250).

**So the first draft's `n_backbone = 24` was not "nearly implied by the budget" — it overspends
the budget it was charged against by 40%.** The corrected recommendation is `n_backbone = 12`,
`n_mini = 4`, at **3 seeds per tuning candidate**, which lands the DeepSea dev tier at 230 runs
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

`(12 + 2 × 4) × 6 = 120` tuning runs; with the 110 dev-cell runs the DeepSea development tier
totals **230**, inside its frozen ≈ 150–250 envelope. No other frozen count moves.

---

### New RNG stream required

The distributions above need a `hparam_search` stream. `STREAM_NAMES` in
`src/utils/conventions.py` does not have one. Adding a stream name is explicitly a safe
operation there ("names, not positions, key the derivation"), so nothing already derived shifts —
but it is a code change that must land with this text, and the stream-registry test needs its
expected tuple updated.

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
     DeepSea dev tier at 230 of its 150–250 envelope. The first draft's 24 overspent it by 40%.
     If a MinAtar-side backbone confirmation is also run, the frozen 240 pilot runs imply
     **18 draws per tuning game** exactly.
   The one genuinely discretionary item left is the **tuning seed count** (3 recommended; 5 forces
   `n_backbone` down to 8), which freeze item 1 should state and currently does not.
2. Send both gaps to the reviewer as an **erratum against `prereg-draft`** — they are missing
   stage-1 values, and disclosing them is cheaper than having the reviewer find them.
3. At stage 3, apply both texts to `preregistration.md`, add the `hparam_search` stream, and cut
   the final tag + OSF mirror.

Until step 3, `preregistration.md` is unchanged and `prereg-draft` remains a valid stable
reference for the pass in progress.
