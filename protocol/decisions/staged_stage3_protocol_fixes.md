# Staged Stage-3 protocol fixes — proposed text, NOT yet in the pre-registration

**Drafted 2026-07-30.** Status: **staged, not applied.**

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
> **Construction (DeepSea, |S| = 128).** DeepSea's state space is the reachable lower-left
> triangle of the `N × N` grid, of size `N(N+1)/2` — 55 states at `N = 10`, 1,275 at `N = 50`.
> The probe set is drawn **without replacement, uniformly over reachable states**, using the
> `probe_set` stream. When `N(N+1)/2 ≤ 128` the probe set is the **exhaustive** reachable set
> (`|S| = N(N+1)/2`, sampling is a no-op and the diagnostics are exact rather than estimated);
> otherwise 128 states are sampled. Rationale: exhaustive coverage at the development sizes
> where the mechanism claims are made, and a fixed cost ceiling at the confirmatory sizes.
> Uniform-over-reachable — not on-policy — is deliberate: an on-policy probe set would be
> endogenous to the exploration method under test, which is precisely the quantity being
> compared, and would make the diagnostics incomparable across methods.
>
> **Construction (MinAtar, |S| = 512).** MinAtar's state space is not enumerable, so `S` is
> collected as a **fixed reference set from a uniform-random behavior policy**: run
> uniform-random episodes with the `probe_set` stream until 512 distinct observations are
> collected, taking every observation in order and deduplicating exactly. The random policy is
> chosen for the same reason as uniform-over-reachable on DeepSea — it is identical across all
> methods, so it does not privilege any method's own visitation distribution. The set is
> generated **per (game, master_seed)** and serialized with the run so it can be audited and
> reused. *(MinAtar diagnostics are the deterministic-conditional analogue, item 20 / §3.3 #8 —
> exploratory and appendix-only, with no confirmatory claims attached.)*
>
> **Visitation counts.** The secondary visitation weighting and diagnostic §3.3 #5 both need
> `v(s)` per probe state. `v(s)` is accumulated **during the run** as the count of visits to each
> probe state (DeepSea: exact state match; MinAtar: exact observation match), recorded at each
> checkpoint alongside the value samples. It is a within-run quantity and cannot be
> reconstructed afterwards.

### Values that need owner sign-off

| Value | Proposed | Note |
|---|---|---|
| DeepSea `\|S\|` | **256** (exhaustive when `N(N+1)/2 ≤ 256`) | Exhaustive through `N = 22`, i.e. **both** dev sizes |
| MinAtar `\|S\|` | 512 | 512 × 20 × 6 × 4 B = 240 KB per checkpoint |
| Sampling | Uniform w/o replacement over reachable (DeepSea); uniform-random-policy reference set (MinAtar) | Method-independent by construction |

**On the DeepSea value — 256, not 128.** Reachable-set sizes are 55 at `N = 10` and 210 at
`N = 20` (the two development sizes), then 465 / 561 / 666 / 780 / 903 at the default
confirmatory sizes `{30, 35, 40, 45, 50}`. At `|S| = 128` the dev sizes *straddle* the
exhaustive threshold: `N = 10` is exact, `N = 20` is sampled. At `|S| = 256` both dev sizes are
exhaustive (the threshold is `N = 22`), so every diagnostic on which a Part-A mechanism claim
rests is computed **exactly** rather than on a 128-of-210 subsample, while the confirmatory
sizes stay capped. Since the Part-A mechanism claims are the study's strongest contribution
(C-i, C-ii), paying 20 KB/checkpoint for exactness there is the right trade. This is the one
genuinely discretionary choice in this document, and it is a recommendation rather than a
derivation — 128 remains defensible if storage or diagnostic cost turns out to bind.

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
> `n_backbone = 24` draws**, selected by IQM per item 3. Distributions (drawn with the
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
> Two factor-specific mini-searches (class 3), **`n_mini = 8` draws each**, identical count for
> every method so the equal-search-budget standard holds: **`prior_scale`** — log-uniform
> `[0.1, 10.0]`, selected by IQM of the canonical prior-on cell `(episodic, on, 10)` on
> development sizes, value shared by all `prior=on` cells; **`eps_schedule`** for
> `ensemble_mean_eps` — final-ε log-uniform `[0.005, 0.1]` with linear decay over the first 10%
> of the budget, selected by IQM of `(mean_eps, off, 10)` on development sizes, shared by its
> cells at both prior levels. **Ties broken by the lower parameter value.** Nothing else is
> tunable per cell.

### Consistency check against the frozen run budget

Spec §3.4 pins MinAtar tuning at **≈ 240 pilot** runs. Counting the proposed values:

| Block | Configs |
|---|---|
| Backbone: 4 methods × 2 tuning games × 24 draws | 192 |
| Mini-searches: 2 searches × 8 draws × 2 games | 32 |
| `K ∈ {5,10,20}`: 2 ensemble methods × 3 K × 2 games | 12 |
| **Total** | **236** |

236 against a frozen ≈ 240 — the proposed `n_backbone = 24` is **not a free parameter**, it is
very nearly what the already-frozen pilot budget implies. That agreement is the main argument
for these numbers over any other set. If the owner prefers a different count, §3.4's 240 has to
move with it and cap X's arithmetic (Σ over methods of final-tier count × median pilot
wall-clock ≤ 120 GPU-h) should be re-checked, since pilot wall-clock is its input.

### New RNG stream required

The distributions above need a `hparam_search` stream. `STREAM_NAMES` in
`src/utils/conventions.py` does not have one. Adding a stream name is explicitly a safe
operation there ("names, not positions, key the derivation"), so nothing already derived shifts —
but it is a code change that must land with this text, and the stream-registry test needs its
expected tuple updated.

---

## Recommended sequence

1. Owner decides the discretionary values: DeepSea `|S|` (128 vs 256) and `n_backbone` (24, or a
   different count with §3.4 moved to match).
2. Send both gaps to the reviewer as an **erratum against `prereg-draft`** — they are missing
   stage-1 values, and disclosing them is cheaper than having the reviewer find them.
3. At stage 3, apply both texts to `preregistration.md`, add the `hparam_search` stream, and cut
   the final tag + OSF mirror.

Until step 3, `preregistration.md` is unchanged and `prereg-draft` remains a valid stable
reference for the pass in progress.
