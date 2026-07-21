# Pre-registration — A1

**Working title:** *When Do Randomized Value Estimates Buy Exploration? Separating the
Estimator from the Use Rule under Low Interaction Budgets.*

> **STATUS: VALUED DRAFT (Session 1, Stage 1) — about to be tagged `prereg-draft`.**
> This file was committed as a stub first (Session 0) so its commit history visibly
> precedes any result commit. Session 1 fills every freeze-list item with its numeric/verbatim
> value from **spec v6.3** (`a1-requirements-and-alternatives-v6.3.md`) and **plan v4.3**.
> The **two-stage freeze**:
>
> 1. Fill all freeze-list items → commit under a **`prereg-draft`** tag (internal timestamp). ← *this draft*
> 2. External methodological pass on the valued draft (one-week time-box; silence → substitute
>    reviewer; none within a further week → documented methodological-review waiver).
> 3. Incorporate fixes → commit the **final pre-registration**: new tag + **OSF mirror**. *This is the freeze.*
>
> **Scientific runs (pipeline spine, Session 3 onward) begin only after the final freeze.**
> Infrastructure work (environment, throwaway benchmark, skeleton, CI) may precede it.
> Post-freeze/pre-confirmatory changes are versioned amendments (new tag + OSF, reasons,
> changed-item list, motivating development outcomes disclosed).
>
> **Two numeric values are reserved to the owner's risk judgment and are set at draft review,
> not invented here:** the compute caps **X** and **Y** (GPU-hours). Their *rules, provenance,
> and order of application* are frozen below (items 4, 17); only the two scalars await the
> owner's call, informed by the Stage-A benchmark forecast in `compute/` (v1.1). They are
> filled before the `prereg-draft` tag is cut.

---

## Two-part identity

- **Part A — controlled mechanism study (DeepSea):** pre-registered **structured partial
  factorial** replication and extension of Osband et al. (2016). Confirmatory.
- **Part B — external performance evaluation (MinAtar):** descriptive whole-algorithm
  comparison under equal search budgets. No mechanism attribution; no transfer claim; no
  confirmatory uncertainty-quality claim.

---

## §1.1 Primary estimand hierarchy (frozen — the anti-flexibility device)

**Primary mechanism estimand (single):** **C-COHERENCE** at `prior=off, K=10`, shared backbone
config. **Primary outcome:** discovery probability within the pre-registered episode budget
(first strictly positive terminal reward). **Aggregation:** unweighted mean of the per-size
effect (difference in discovery probability) over the five confirmatory sizes.

**Seed-independence rule (v6.3):** every random stream in every run (network initialization,
DeepSea action-direction mapping, replay sampling, action/exploration noise, and the extended
streams) is derived as `hash(master_seed, cell_id, stream_name, seed_index)` with non-overlapping
derived streams; **no underlying random stream is reused across cells.** Seed labels may share an
index for bookkeeping, but runs in different cells are independent by construction. *(Implementation,
pinned: BLAKE2b(digest_size=16) → `numpy.random.SeedSequence` over a canonical `0x1F`-separated
UTF-8 payload of the four fields; the builtin salted `hash()` is never used. Eight registered
streams: `init, env_mapping, replay, action_noise, bootstrap_mask, eval_episodes, probe_set,
noisynet_diag`. Pinned regression value: `derive_seed(0,"episodic|off|K10","init",0) ==
8011425302454941550`. See `src/utils/conventions.py`; guarded by CI.)*

**Bootstrap structure:** independent stratified bootstrap — seeds resampled independently within
each cell × size; sizes are fixed strata; **no common-random-number design; no paired claims
anywhere.**

**Interpretation rule with SESOI:** the 95% stratified-bootstrap CI on the aggregated difference is
the confirmatory statement. **SESOI: Δ = 0.10** absolute difference in aggregated discovery
probability. Report as: a **directionally confirmed** effect (CI excludes zero); a **practically
negligible** effect (CI lies entirely within (−Δ, +Δ) — equivalence-style conclusion); or an
**inconclusive** estimate (neither). The many-seed design (20/cell) is described as substantially
higher-powered than the original ablations; a **prospective MDE report** computed from
development-pilot variance is a pre-registered deliverable of the mechanism pilot (informs
interpretation; never alters the frozen Δ or the design).

**Fixed-sequence gatekeeping — written literally:**
1. **C-COHERENCE** (primary)
2. **C-USE** at `prior=off, K=10`
3. **C-PRIOR** at `use_rule=episodic, K=10`
4. **C-K** trend within `use_rule=episodic, prior=off`

Each test receives confirmatory interpretation **only if every preceding test rejected its null at
two-sided α = 0.05**; at the first non-rejection, all subsequent tests are reported descriptively
with CIs and no test language. Rejection for tests 1–3 = the 95% stratified-bootstrap CI on the
aggregated difference excluding zero. **Test 4 (C-K):** Cochran–Armitage trend statistic on per-seed
discovery indicators with equally spaced ordinal scores (1, 2, 3) for K ∈ {5, 10, 20}, stratified by
size (CMH-style combination); inference by permutation of K-cell labels within size strata
(**10,000 permutations**); two-sided α = 0.05. **Rung-2 conditional (frozen):** if descope rung 2 is
triggered before confirmatory execution, C-K is removed from the confirmatory sequence, which then
terminates after C-PRIOR; any two-level K comparison is reported descriptively.

**Everything else is descriptive** (CIs, no test language, explicitly labeled): the same contrasts on
time-to-first-success and max-depth; `prior=on` replications of C-COHERENCE/C-USE; all battery
relationships; all MinAtar comparisons; RQ3. **Stated limitation (v6.3):** the structured partial
factorial cannot estimate K × use_rule or three-way interactions; C-COHERENCE and C-USE are estimated
at K = 10 only.

No contrast, prior level, K, size subset, outcome, or diagnostic may be promoted after results exist.

---

## Contrasts and the switchboard

One bootstrapped-ensemble implementation; every cell is a point in the factor space, selected by
config:

```text
use_rule:  episodic_head | per_step_head | ensemble_mean_eps
prior:     off | on
K:         5 | 10 | 20
```

**The 10-cell structured partial factorial (v6.3):** the full `use_rule × prior` core at K = 10
(6 cells), **augmented** with K ∈ {5, 20} for `episodic_head` at both prior levels (4 cells). This is
**not** a full 3×2×3 factorial: K × use_rule and three-way interactions are not estimable, and
C-COHERENCE / C-USE are estimated at K = 10 only (limitation stated in the paper).

| Contrast | Varies | Held fixed | Identifies |
|---|---|---|---|
| **C-USE** | episodic_head vs. ensemble_mean_eps | prior, K | Effect of the pre-registered episodic-head rule vs. a **capacity-matched ensemble-mean ε-greedy rule** |
| **C-COHERENCE** | episodic_head vs. per_step_head | prior, K | Effect of **temporal coherence** of the sampled value function at matched estimator |
| **C-PRIOR** | prior on vs. off | use_rule, K | Effect of the randomized-prior **estimator intervention** as practiced (incl. tuned scale) |
| **C-K** | K ∈ {5,10,20} within episodic_head | prior level | Ensemble-size **estimator intervention** |

Figure aliases only: E1 = (episodic, off, 10), E2 = (episodic, on, 10), E3 = (per_step, off, 10),
E4 = (mean_eps, off, 10). Causal language is permitted only for **C-USE** and **C-COHERENCE**;
**C-PRIOR** and **C-K** are described as *estimator interventions* ("associated/anticipates," never
causal "predicts").

---

## THE FREEZE LIST (20 items — VALUED)

**1. Config/seed counts per tier and stage + the cell-specific RNG derivation scheme.**
DeepSea development: **exactly 10 seeds** for the two rule-input runs (the canonical episodic cell
`(episodic_head, prior=off, K=10)` and the ε-greedy DDQN reference), **5 seeds** for every other
development cell. DeepSea confirmatory: **20 seeds per cell**. MinAtar: **3 dev → 5 pilot → 10
held-out**. RNG derivation scheme = the §1.1 seed-independence rule (8 non-overlapping derived
streams; no reuse across cells; pinned implementation and regression value as in §1.1).
*(Reviewer Fix 3 — ride-along, owner-decided 2026-07-21:)* the DDQN and NoisyNet baselines ride
along in the DeepSea **confirmatory** sweep at **10 seeds each** across the five confirmatory sizes
(2 methods × 5 sizes × 10 seeds = 100 runs), included in the ≈1,100 confirmatory total (see item 17);
this is a descriptive reference, not part of any confirmatory contrast, and changes no other count.

**2. Search distributions per hyperparameter; backbone-tuning budget; two factor-specific
mini-budgets with pinned selection objectives and tie-breakers.** Backbone nuisance (class 1) tuned
once on the ε-greedy DDQN backbone over development sizes under a pre-registered random-search budget
(distributions and budget frozen in the parameter table below). Two factor-specific mini-searches
(class 3): **`prior_scale`** — one mini-search selected by **IQM of the canonical prior-on cell
`(episodic, on, 10)` on development sizes**, value shared by all `prior=on` cells; **`eps_schedule`**
for `ensemble_mean_eps` — one mini-search selected by **IQM of `(mean_eps, off, 10)` on development
sizes**, shared by its cells at both prior levels. **Ties broken by the lower parameter value.**
Nothing else is tunable per cell.

**3. Selection statistic + tie-breaking.** **IQM** (interquartile mean) throughout; ties broken by
the lower parameter value.

**4. Final-tier trigger (method-specific formula); cap X GPU-hours.** The final tier runs **iff**
Σ over the four methods of *(final-tier run count × that method's median pilot-tier per-run
wall-clock)* **≤ X GPU-hours**. X is fixed at freeze; inputs come from each method's own pilot-tier
runs. **[X — owner to set at draft review; rule + provenance frozen. Forecast context in
`compute/A1-compute-cost-and-free-tier-execution-plan-v1.1.md` §3.3/§11.]**

**5. DeepSea development sizes; confirmatory-size selection rule + its three size sets; episode
budget per N.** Development sizes: **N = 10 and N = 20**. Confirmatory-size selection rule
(numerically exact; inputs measured before Gate A on **exactly 10 development seeds each** of the
ε-greedy DDQN reference (a) and the canonical episodic cell (b), at N = 20):
- **if (b) ≤ 2/10** → shift **down** to **{22, 26, 30, 34, 38}**;
- **else if (a) ≥ 9/10** → shift **up** to **{40, 45, 50, 55, 60}**;
- **else default {30, 35, 40, 45, 50}**.

Inclusive integer thresholds; raw proportions; **down-shift takes precedence**. Episode budget per N:
frozen per size (the pre-registered DeepSea episode budget; the size-scaled budget committed with the
pipeline). Rationale: published ε-greedy failure onset near N≈10–15; ensemble solvability through
N≈50 at bsuite budgets.

**6. Reporting windows.** MinAtar checkpoints at **100k / 500k (/1M)** (Variant B — same runs);
DeepSea reported at the frozen per-size episode budget. Reporting windows frozen with the pipeline.

**7. Probe-set construction + weighting.** **Uniform weighting primary; visitation-weighted
secondary.** Probe set S constructed per the frozen diagnostics spec; the `probe_set` RNG stream
governs any sampling.

**8. Operational failure policy (§3.6) incl. divergence criterion + infra/algorithmic
classification.** Infrastructure failure → rerun the **same (config, seed)**, classified by error
type. Algorithmic failure (incl. divergence per the frozen divergence criterion) → *is* the result.
Implementation bug → the **confirmatory block is void**, a new pre-registered iteration follows.
Reproducibility reruns are excluded from analysis. **Seed counts never increase post-results.**

**9. Bootstrap units, stratification, CI construction; §1.1 aggregation + fixed-sequence; RQ2-L
statistic in full.** Independent stratified bootstrap (units = seeds; strata = sizes; no CRN, no
pairing); 95% CI on the aggregated difference; §1.1 aggregation and fixed-sequence as above.
**RQ2-L:** primary diagnostic = marginal uncertainty–error rank alignment at t₀; standardize the
diagnostic within each (cell × N) stratum (z-score across seeds); outcome = time-to-first-success
after t₀, censored at budget end; statistic = within-stratum **Harrell's-C-style concordance** over
comparable seed pairs (a pair is comparable if the shorter time is uncensored; ties in time or
diagnostic excluded), combined across strata by weights proportional to comparable-pair counts;
direction: better alignment → shorter time → concordance > 0.5; inference by **permutation
(10,000)** within stratum; strata fixed; **a stratum with < 10 comparable pairs is excluded and
reported**. Supporting, never causal.

**10. Frozen-policy extraction rules per method.** Frozen per method (C12 governs the extraction);
committed with the pipeline.

**11. t₀ landmark rule + post-t₀ outcome definitions.** t₀ = **10% of the episode budget**; fallback
to **5%** iff **the canonical episodic cell's 10 development seeds** show **> 2/10 succeeding by
t₀** (v6.3-pinned population). Post-t₀ outcomes: time-to-first-success after t₀ (censored at budget
end) and the post-t₀ diagnostic definitions.

**12. The 10 cells, four contrasts, shared-configuration rule, factor-specific parameter list, stated
interaction limitations.** As in the switchboard section above and the parameter table below. The
interaction limitation (no K × use_rule; no three-way; C-COHERENCE/C-USE at K = 10 only) is stated in
the paper.

**13. Primary estimand hierarchy (§1.1).** Frozen verbatim in §1.1 above — contrasts, settings,
outcomes, aggregation, test order, SESOI.

**14. Battery formulas, sample counts, aggregation, micro-conventions (§3.3).** Frozen verbatim in
the **Uncertainty diagnostics** section below — including the action-gap **lowest-action-index tie
rule**, quantile interpolation **`numpy.quantile(..., method="linear")`**, the **rank-biserial
r = 2·(AUC − 0.5)** formula and sign, and the C-K ordinal-score coding (1, 2, 3) + permutation
procedure. Samples: **M = K heads**; **M = 30 i.i.d. NoisyNet draws at measurement only**.

**15. Canonical MinAtar four with defining parameters + K_shared joint rule.** (1) **ε-greedy Double
DQN** (tuned ε schedule); (2) **NoisyNet-DQN**; (3) **Bootstrapped DQN** = `(episodic_head,
prior=off, K = K_shared)`; (4) **RP-BDQN** = `(episodic_head, prior=on, K = K_shared, tuned prior
scale)`. Identical pre-registered search budget each. **K_shared selection rule (frozen):** during
the pilot tier, K ∈ {5, 10, 20} is searched by both ensemble methods; **K_shared = argmax over K of
the mean of the two ensemble methods' best-config IQM on the tuning games (Breakout + Asterix) at
that K**; both inherit K_shared unchanged.

**16. QR-DQN exploratory status; hypothesis + test (neutrality rule).** Pre-specified **exploratory
follow-up**, implemented and run **only after Gate B**; identical search budget for comparability;
**excluded from confirmatory RQ1 aggregates**; labeled exploratory everywhere; its
epistemic-vs-distributional hypothesis and test pre-registered under the neutrality rule.

**17. Descope ladder + objective compute-cap trigger Y GPU-hours; before-Gate-A execution point.**
After pilot-tier wall-clock measurement, projected total compute is computed; **if it exceeds the
frozen cap Y GPU-hours, rungs apply in fixed order until ≤ Y**, executed **before Gate A and before
any confirmatory outcome** (ladder application after confirmatory execution is prohibited).
**Order:** (1) pre-launch horizon → 500k (RQ3 descoped; **pre-launch only**); (2) K-axis → {10, 20}
(**C-K exits the confirmatory sequence per §1.1**); (3) final-tier configs down (identical across
methods); (4) held-out seeds 10 → 5 with sensitivity note; (5) drop the MinAtar E-cell spot-check;
(6) drop the QR-DQN exploratory follow-up. **Never cut:** the **six-cell `use_rule × prior` core at
K = 10**, the dual axes, the EWRL figure, the protocol, the DeepSea confirmatory seeds.
**[Y — owner to set at draft review; rule + order frozen. Forecast context in `compute/` v1.1.]**

**18. Two-stage freeze + amendment policy incl. external-pass time-box + substitute/waiver rule.**
(1) fill all freeze-list values → `prereg-draft` tag; (2) external pass on the valued draft
(**one week**; silence → **substitute reviewer**; none within a further week → **documented
methodological-review waiver**); (3) fixes → **final tag + OSF mirror** (the freeze); (4) scientific
runs only after the final freeze. **Amendment policy:** any change post-final-freeze but
pre-confirmatory-execution = versioned amendment (new tag + OSF timestamp, reasons, enumerated
changed items, disclosure of motivating development outcomes); confirmatory runs only after the
amendment freeze. Post-confirmatory changes are governed by the confirmatory-integrity rule (§3.6);
amendments cannot rescue them.

**19. Parameter classes; K_shared joint rule; SESOI Δ = 0.10.** Three classes (backbone-tuned /
literature-fixed ensemble-shared parameter table / factor-specific) as in the parameter table below;
K_shared joint rule as in item 15; **SESOI Δ = 0.10** with the equivalence-style interpretation of
the 95% CI (item / §1.1).

**20. Deterministic MinAtar-cloning conditional + undefined-value policy (§3.3).** MinAtar
behavior-policy analogue: **100 clone/restore reproduction tests** (stored state + RNG state, fixed
action sequence, two replays): **full probe rollouts iff all 100 bit-exact**; **episode-start-only
iff cloning fails but seeded fresh-reset rollouts are bit-reproducible**; **drop otherwise**.
Exploratory, appendix-only; never called "Q-error." **Undefined-value policy (battery-wide):**
undefined statistics recorded **NA**, excluded from aggregation, exclusions counted and published;
nothing imputed; **σ = 0 is a substantive measurement** where it is a value rather than a divisor
(the RQ2-L < 10-comparable-pairs rule is an instance).

---

## Parameter table (the three shared-configuration classes)

**Class 1 — Backbone nuisance (tuned once on the ε-greedy DDQN backbone; inherited identically by all
cells).** Learning rate, optimizer, replay size, batch size, target-update cadence, network width.
Tuned over development sizes under the pre-registered random-search budget (distributions frozen with
the pipeline config). *Bootstrap-specific parameters cannot be tuned on a single-head DDQN and are
excluded from this class.*

**Class 2 — Ensemble-shared nuisance (fixed to explicitly listed values, NOT tuned; identical across
all cells).** The final pre-registration carries a table — **parameter | value | source or
justification** — covering: bootstrap **mask probability**, **head-loss aggregation/normalization**,
**shared-trunk convention**, **head initialization**, **per-head target handling**. Each value cites
the specific section/experiment of **Osband et al. (2016)**, or is labeled a **documented design
choice inspired by Osband et al.** where no exact published default exists (their Atari experiments
used **mask p = 1**; there is no universal "Osband default" for DeepSea). *(The exact values are
verified against the source papers in Session 2 and filled here before the final freeze.)*

**Class 3 — Factor-specific parameters (the only per-factor tuning; objectives pinned).**
`prior_scale` (one mini-search; selected by IQM of `(episodic, on, 10)` on dev sizes; shared by all
`prior=on` cells) and `eps_schedule` for `ensemble_mean_eps` (one mini-search; selected by IQM of
`(mean_eps, off, 10)` on dev sizes; shared by its cells at both prior levels). **Ties → lower
parameter value.** C13 audits resolved configs against exactly this three-class structure.

---

## Uncertainty diagnostics — mathematical definitions (§3.3; all frozen)

Notation: probe set S; M value samples {Q_m} (**M = K heads; M = 30 i.i.d. NoisyNet draws at
measurement only**); Q̄ = mean; σ(s,a) = sample std; Q* exact on DeepSea.

1. **Marginal alignment (RQ2-L primary):** Spearman ρ over (s,a) ∈ S × A between σ(s,a) and
   |Q̄(s,a) − Q*(s,a)|.
2. **Action-gap alignment:** a₁, a₂ = top-2 by Q̄ (**ties by lowest action index**);
   ĝ(s) = Q̄(s,a₁) − Q̄(s,a₂); **g*(s) = Q*(s,a₁) − Q*(s,a₂) using the same a₁, a₂**;
   u_g(s) = std over m of [Q_m(s,a₁) − Q_m(s,a₂)]. Statistic: Spearman ρ between u_g(s) and
   |ĝ(s) − g*(s)|.
3. **Incorrect-argmax flagging:** optimal set = Argmax_a Q*(s,·); e(s) = 1[argmax Q̄ ∉ Argmax Q*];
   modal-action ties among samples by lowest action index; d(s) = 1 − modal fraction. **Statistic:
   rank-biserial r = 2·(AUC − 0.5)** (AUC = probability a uniformly drawn incorrect-argmax state
   carries higher d(s)); positive = greater disagreement at incorrect states.
4. **Optimal-path uncertainty:** mean σ(s, a*(s)) along the optimal path (per depth; AUC over depth
   as summary).
5. **Visitation-conditioned decay:** OLS slope of log σ(s, a*(s)) on log(1 + v(s)) over raw probe
   states (bins for display only); unweighted; per checkpoint; **σ = 0 excluded and counted**.
6. **Temporal persistence:** within-episode fraction of probe states whose greedy action under the
   current sample equals the episode-start sample's; per-episode mean, per checkpoint (across
   consecutive samples for per-step rules; ~1 by construction for episodic — descriptive).
7. **Empirical containment:** central 80% interval from empirical quantiles of {Q_m(s,a)} —
   **`numpy.quantile(..., method="linear")`**; containment = fraction of (s,a) with Q* inside.
   Descriptive; level fixed.
8. **MinAtar behavior-policy analogue — deterministic conditional:** as in freeze item 20.
9. **Undefined-value policy (battery-wide):** as in freeze item 20.

---

## Run budget, descope ladder, and the Session-7 confirmatory block (Reviewer Fix 2)

**Run budget (spec §3.4):** MinAtar tuning ≈ **240 pilot + 800 final** (formula-gated) @500k;
MinAtar held-out ≈ **120** runs to the pre-launch frozen horizon; DeepSea dev ≈ **150–250** +
confirmatory ≈ **1,100**; QR-DQN exploratory ≈ **300**. Order of **2,500–3,000** runs total;
per-method wall-clock measured pre-final drives the frozen trigger.

**DeepSea confirmatory block (≈1,100 runs) — cell accounting (Fix 2 + Fix 3):**

| Component | Cells / methods | Sizes | Seeds | Runs |
|---|---|---:|---:|---:|
| **Structured partial factorial core** | **10 cells** (default) | 5 | 20 | **1,000** |
| Baseline ride-along (Fix 3) | DDQN + NoisyNet | 5 | 10 each | **100** |
| **Total (default)** | | | | **≈ 1,100** |

**Rung-2 (descope) variant of the block:** if descope **rung 2** is triggered before confirmatory
execution, the design reduces to the **mandatory six-cell `use_rule × prior` core at K = 10 plus the
surviving K cells** — **C-K exits the confirmatory sequence per §1.1**, so the factorial runs
**8 cells** (6 core @ K=10 + 2 episodic @ K=20), and the block total is reduced accordingly; any
two-level K comparison is reported descriptively. The run-budget table above therefore reads
**"10 cells default, or 8 cells under rung 2,"** with the confirmatory total **≈1,100 default /
reduced under rung 2**. This is the only path by which the confirmatory cell count changes, and it
can only be taken **before Gate A / before any confirmatory outcome**.

**Never cut** (spec §3.4): the six-cell `use_rule × prior` core at K = 10, the dual axes, the EWRL
figure, the protocol, the DeepSea confirmatory seeds.

---

## §3.6 Operational failure policy (frozen)

Infrastructure failure → rerun the **same (config, seed)**, classified by error type. Algorithmic
failure → *is* the result. Implementation bug → **confirmatory block void**, new pre-registered
iteration. Reproducibility reruns excluded. **Seed counts never increase post-results.** This is the
operational arm of the confirmatory-integrity rule: post-result design changes on either environment
family demote affected analyses to exploratory, permanently.

---

*Sources of truth: `a1-requirements-and-alternatives-v6.3.md` (§1.1, §2/§2.1, §3.1–§3.4, §6, §7) and
`A1-claude-science-execution-plan-v4.3.md` (§7 Session 1 brief, §8 gates, Appendices C/D). Every value
above is quoted or computed from those two documents; where a value is reserved to the owner (caps X,
Y) or verified against primary papers in Session 2 (the class-2 parameter table), that is stated
inline.*
