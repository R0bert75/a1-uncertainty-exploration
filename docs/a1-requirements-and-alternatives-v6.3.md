# A1 Research Project: Requirements Specification & Topic Alternatives Assessment

**Companion to pivot plan v4 · July 2026 · v6.3 (external review #7 + corrected novelty audit incorporated; architecture unchanged)**

**Changelog v1→v6.1:** as recorded in prior versions (RQ operationalization; two-part mechanism restructure; development/confirmatory split; factorial switchboard with named contrasts; C0; estimand hierarchy; Variant B; failure policy; parameter classes; K_shared rule; two-stage freeze; amendment policy; SESOI; deterministic conditionals; battery formulas).

**Changelog v6.1 → v6.2:** claim-level novelty audit incorporated (C-i/C-ii contributions; C-iii/C-iv inherited/design); horizon wording (pre-launch frozen horizon); rung-2/C-K conditional; ensemble-shared parameter table; C-K test fully specified; battery micro-freezes; substitute-reviewer/waiver rule.

**Changelog v6.2 → v6.3 (review #7 — precision of design naming, seed independence, and novelty claims):**
1. **Design renamed honestly:** the 10-cell design is a **structured partial factorial** — a full `use_rule × prior` core at K = 10, augmented with a prior × K axis for episodic sampling. It is **not** a full 3×2×3 factorial (18 cells) and cannot estimate K × use_rule or three-way interactions; this is stated as a limitation. C-i reworded accordingly.
2. **Seed independence repaired (the v6.2 claim was wrong):** identical integer seeds *can* create cross-cell dependence (shared init weights, shared DeepSea action-direction mappings, shared replay streams). New rule (Variant A): **cell-specific derived RNG streams** — every random stream (network init, environment mapping, replay sampling, action noise) is derived as `hash(master_seed, cell_id, stream_name, seed_index)`; no underlying stream is reused across cells. Independence holds by construction; the independent stratified bootstrap is thereby justified. Seed *labels* may share an index for bookkeeping only.
3. **t₀ and mini-search input populations pinned:** t₀ rule input = the solve fraction among **exactly the 10 development seeds of the canonical episodic cell `(episodic_head, prior=off, K=10)` at N = 20** (the same runs as size-rule input (b)); `prior_scale` selected by IQM of `(episodic, on, 10)` on development sizes; `eps_schedule` by IQM of `(mean_eps, off, 10)` on development sizes; ties broken by the lower parameter value.
4. **Novelty claims tightened (corrected audit):** C-i now reads "we did not identify any prior study combining a controlled `use_rule × prior` factorial with a pre-specified ensemble-size axis, exact-Q\* diagnostics, and a matched confirmatory protocol" — never "no work crosses use_rule × prior × K." C-ii restricted to **exact-Q\*-referenced error, decision-margin, and incorrect-action ground truth** for randomized-value estimates (not "first continuous measurement on DeepSea"). **"First application" removed from C-iii.** MinAtar null narrowed to the exact canonical set, phrased "we did not identify," never "no paper exists."
5. **Neighbor set corrected:** the **BDQN scaling paper promoted to the top tier** (it already occupies much of the prior × K axis on DeepSea); **EVE (Exploration via Epistemic Value Estimation)** added — continuous epistemic uncertainty on DeepSea, acting-rule and bootstrapping ablations, so neither "all prior work compares whole algorithms" nor "all DeepSea work is pass/fail" may be claimed; **Priors Matter (2025)** added for C-PRIOR framing; **Auditing the Risk Claims of Distributional RL (July 2026; existence verified 2026-07-15)** added — ground-truth audits of QR-DQN-family risk claims on MinAtar, confirming QR-DQN's exploratory status and restricting C-ii to randomized-value estimates on DeepSea. Detailed characterizations of all four carry a **verify-at-Session-2** flag.
6. **DeepSea origin corrected** where lineage is stated: introduced in the randomized-value/prior-functions line of work, later standardized as part of bsuite.
7. **Safe one-pager novelty formulation** adopted into §9. The internal novelty deep dive is retired as a standalone document; its corrected content and a reproducible-search appendix fold into Session 2's `positioning.md` (single source of truth).

All changes land **before protocol freeze**; after freeze, changes become documented deviations or versioned amendments per §3.1.

---

# Part I — Requirements specification

## 0. Review-closure status

Reviews #2–#6 — closed in v4–v6.2. Review #7 + the corrected novelty audit — closed in v6.3 per the changelog. Judgment call this round: the three newly surfaced neighbors enter with verify-at-Session-2 discipline (existence of the July-2026 audit paper independently confirmed; content characterizations to be read against the papers before related-work text is written), and the novelty deep dive folds into `positioning.md` rather than persisting as a third synchronized document.

## 1. Research questions

Two-part identity: **Part A — controlled mechanism study (DeepSea)**, a pre-registered structured-partial-factorial replication-and-extension; **Part B — external performance evaluation (MinAtar)**, descriptive whole-algorithm comparison under equal search budgets. No confirmatory uncertainty-quality claims on MinAtar; no formal transfer claim.

- **RQ1 (descriptive, MinAtar + DeepSea):** which uncertainty-aware methods beat well-tuned ε-greedy on both reporting axes and on direct exploration outcomes, under equal pre-registered search budgets; capacity/compute reported; no mechanism attribution. **Reporting boundary:** on MinAtar, the primary output is the three per-game results; the aggregate is a compact summary; the paper makes **no claim about a general ranking of uncertainty-aware methods**. *(RQ1 can constitute a complete descriptive study; whether it is publishable alone depends on outcomes, statistical power, and baseline quality — not a methodological guarantee.)*

- **RQ2-M (mechanism, DeepSea, confirmatory):**
  - **RQ2-M1 (causal core):** effects of the pre-registered action-selection contrasts, stated as rule-vs-rule comparisons:
    - **C-COHERENCE** — per-episode vs. per-step head resampling at fixed estimator (prior level, K, shared config): the effect of temporal coherence of the sampled value function. *(Replicates and extends the Bootstrapped-DQN vs. "Thompson DQN" comparison of Osband et al. 2016 as a many-seed, exact-ground-truth experiment with a pre-registered MDE report.)*
    - **C-USE** — the pre-registered episodic-head rule vs. a **capacity-matched ensemble-mean ε-greedy rule** (with its own pre-registered tuned ε schedule): the effect of the full action-selection rule, **not** an abstract "uncertainty used vs. ignored." *(Replicates and extends the ensemble-policy control of Osband et al. 2016.)*
  - **RQ2-M2 (estimator interventions, supporting):** effects of the randomized-prior intervention **as practiced** (C-PRIOR, incl. its tuned scale) and of ensemble size (C-K) on (i) measured uncertainty properties and (ii) discovery outcomes, and whether (i) and (ii) move together. Association, not causation-on-quality. *(Framing sentence, v6.3: A1 studies randomized prior functions as one controlled intervention inside the bootstrapped-value family, complementing recent work on prior and likelihood misspecification in Bayesian deep Q-learning and on the scaling of bootstrapped ensembles.)*

### §1.1 Primary estimand hierarchy (frozen; the anti-flexibility device)

- **Primary mechanism estimand (single):** **C-COHERENCE** at `prior=off, K=10`, shared backbone config. **Primary outcome:** discovery probability within the pre-registered episode budget (first strictly positive terminal reward). **Aggregation:** unweighted mean of the per-size effect (difference in discovery probability) over the five confirmatory sizes.
- **Seed-independence rule (v6.3 — replaces the v6.2 claim):** every random stream in every run (network initialization, DeepSea action-direction mapping, replay sampling, action/exploration noise) is derived as `hash(master_seed, cell_id, stream_name, seed_index)` with non-overlapping derived streams; **no underlying random stream is reused across cells.** Seed labels may share an index for bookkeeping, but runs in different cells are independent by construction. **Bootstrap structure:** independent stratified bootstrap — seeds resampled independently within each cell × size; sizes are fixed strata; no common-random-number design; no paired claims anywhere.
- **Interpretation rule with SESOI:** the 95% stratified-bootstrap CI on the aggregated difference is the confirmatory statement. Smallest effect size of interest: **Δ = 0.10** absolute difference in aggregated discovery probability. The result is reported as: a directionally confirmed effect (CI excludes zero); a **practically negligible** effect (CI lies entirely within (−Δ, +Δ) — an equivalence-style conclusion); or an inconclusive estimate (neither). The many-seed design (20/cell) is described as substantially higher-powered than the original ablations, and a **prospective MDE report** computed from development-pilot variance is a pre-registered deliverable of the mechanism pilot (informs interpretation; never alters the frozen Δ or the design).
- **Fixed-sequence gatekeeping — written literally:** the sequence is **(1) C-COHERENCE (primary) → (2) C-USE at `prior=off, K=10` → (3) C-PRIOR at `use_rule=episodic, K=10` → (4) C-K trend within `use_rule=episodic, prior=off`**. The primary test is position 1. Each test receives confirmatory interpretation **only if every preceding test rejected its null at two-sided α = 0.05**; at the first non-rejection, all subsequent tests are reported descriptively with CIs and no test language. Rejection for tests 1–3 = the 95% stratified-bootstrap CI on the aggregated difference excluding zero. **Test 4:** Cochran–Armitage trend statistic on per-seed discovery indicators with equally spaced ordinal scores (1, 2, 3) for K ∈ {5, 10, 20}, stratified by size (CMH-style combination); inference by permutation of K-cell labels within size strata (10,000 permutations); two-sided α = 0.05. **Rung-2 conditional (frozen):** if descope rung 2 is triggered before confirmatory execution, C-K is removed from the confirmatory sequence, which then terminates after C-PRIOR; any two-level K comparison is reported descriptively.
- **Everything else is descriptive:** the same contrasts on time-to-first-success and max-depth; prior=on replications of C-COHERENCE/C-USE; all battery relationships; all MinAtar comparisons; RQ3. Reported with CIs, no hypothesis-test language, explicitly labeled. **Limitation stated (v6.3):** the structured partial factorial cannot estimate K × use_rule or three-way interactions; C-COHERENCE and C-USE are estimated at K = 10 only.
- The hierarchy, outcomes, aggregation, SESOI, seed-derivation and bootstrap rules, and test procedures are freeze-list items. No contrast, prior level, K, size subset, outcome, or diagnostic may be promoted after results exist.

- **RQ2-Q (battery):** scope unchanged; mathematically defined in §3.3.

- **RQ2-L (supporting associational analysis — fully specified):**
  - **Primary diagnostic (single):** marginal uncertainty–error rank alignment at t₀ (§3.3.1); action-gap alignment is the first secondary diagnostic.
  - **Procedure:** standardize the diagnostic within each (cell × N) stratum (z-score across seeds); outcome = time-to-first-success after t₀, censored at budget end. **Statistic:** within-stratum Harrell's-C-style concordance over comparable seed pairs (a pair is comparable if the shorter time is uncensored; ties in time or diagnostic excluded), combined across strata by weights proportional to comparable-pair counts. **Direction:** better alignment → shorter time-to-success = concordance > 0.5. **Inference:** permutation test, diagnostics permuted within stratum, 10,000 permutations. **Strata are fixed.** **Exclusion rule:** a stratum with fewer than 10 comparable pairs is excluded and reported.
  - Status: supporting; never causal; cross-method scatter remains a descriptive appendix.

- **RQ3 (pre-specified secondary analysis):** budget-axis data exist by design (§3.2 Variant B); analysis pre-specified and secondary, with the tune-once-at-500k transfer confound stated. Gate B occurring after these data exist is acknowledged and harmless. Under a rung-1 horizon, RQ3 is not answerable and is reported as descoped.

**Thesis sentence (v6.3):** "Uncertainty-aware exploration methods bundle an uncertainty *estimator* with a rule for *using* it. The original Bootstrapped DQN work sketched ablations separating the two; we replicate and extend them in a pre-registered structured design with exact ground truth — per-episode vs. per-step sampling and randomized-value vs. ensemble-mean rules as causal contrasts at a fixed ensemble size, randomized priors and ensemble size as estimator interventions — with a decision-relevant uncertainty-quality battery referenced to exact Q\*, and separately evaluate the same methods' whole-algorithm performance on standard small-scale benchmarks under an equal-search-budget protocol."

**Terminology, NoisyNet status, language rules:** unchanged, plus (v6.3): the design is a **structured partial factorial** ("full factorial" is banned); novelty statements use "we did not identify," never "no paper exists" or "first"; "replication and extension" remains the sanctioned frame; contrast names cited for every mechanism claim; "associated/anticipates," never causal "predicts."

## 2. Scope

### Milestone 1

| Dimension | Requirement |
|---|---|
| **MinAtar method set (canonical four — frozen):** | (1) ε-greedy Double DQN (tuned ε schedule); (2) NoisyNet-DQN; (3) Bootstrapped DQN = `(episodic_head, prior=off, K = K_shared)`; (4) RP-BDQN = `(episodic_head, prior=on, K = K_shared, tuned prior scale)`. Each with the identical pre-registered search budget. **K_shared selection rule (frozen):** during the pilot tier, K ∈ {5, 10, 20} is searched by both ensemble methods; K_shared = argmax over K of the **mean of the two ensemble methods' best-config IQM** on the tuning games at that K; both inherit K_shared unchanged. |
| Mechanism design (DeepSea) | The 10-cell **structured partial factorial** (§2.1): development pilots pre-Gate-A; single confirmatory sweep post-Gate-A |
| Environments | MinAtar (5 games; tuning = Breakout + Asterix, evaluation = remaining 3) · DeepSea (development N = 10, 20; confirmatory sizes per the §3.1 selection rule, default {30, 35, 40, 45, 50}) |
| Budgets | MinAtar: tuning @500k; **held-out evaluation runs proceed unconditionally to the pre-launch frozen horizon — 1M by default, or 500k if descope rung 1 was triggered before launch — with 100k/500k(/1M) checkpoints (Variant B)**. DeepSea: frozen episode budget per size |
| Seeds | MinAtar: 3 dev → 5 pilot → 10 held-out. DeepSea: development — **exactly 10 seeds for the canonical episodic cell and the ε-greedy reference (rule inputs), 5 seeds for the remaining cells**; **20/cell confirmatory** |
| QR-DQN | **Pre-specified exploratory follow-up:** implemented and run only after Gate B; identical search budget for comparability; excluded from confirmatory RQ1 aggregates; labeled exploratory everywhere; its epistemic-vs-distributional hypothesis and test pre-registered under the neutrality rule. *(v6.3 note: the July-2026 distributional-risk audit paper independently ground-truth-audits QR-family risk claims on MinAtar, confirming that exploratory status is the right weight for this block.)* |

### §2.1 The structured partial factorial switchboard

One bootstrapped-ensemble implementation; every cell is a point in the factor space, selected by config:

```text
use_rule:  episodic_head | per_step_head | ensemble_mean_eps
prior:     off | on
K:         5 | 10 | 20
```

**Pre-registered design (10 cells — a structured partial factorial, v6.3):** the full `use_rule × prior` core at K = 10 (6 cells), **augmented** with K ∈ {5, 20} for `episodic_head` at both prior levels (4 cells). This is not a full 3×2×3 factorial: K × use_rule and three-way interactions are not estimable, and C-COHERENCE/C-USE are estimated at K = 10 only (limitation stated in the paper).

**Named pre-registered contrasts — each varies exactly one factor:**

| Contrast | Varies | Held fixed | Identifies |
|---|---|---|---|
| **C-USE** | episodic_head vs. ensemble_mean_eps | prior, K | Effect of the pre-registered episodic-head rule relative to a capacity-matched ensemble-mean ε-greedy rule |
| **C-COHERENCE** | episodic_head vs. per_step_head | prior, K | Effect of temporal coherence of the sampled value function at matched estimator |
| **C-PRIOR** | prior on vs. off | use_rule, K | Effect of the randomized-prior **estimator intervention** as practiced (incl. tuned scale) |
| **C-K** | K ∈ {5,10,20} within episodic_head | prior level | Ensemble-size **estimator intervention**; engages the BDQN-scaling diversity findings directly |

Aliases for figures only: E1 = (episodic, off, 10), E2 = (episodic, on, 10), E3 = (per_step, off, 10), E4 = (mean_eps, off, 10).

- **Shared-configuration rule (three parameter classes):**
  1. **Backbone nuisance — tuned once on the ε-greedy DDQN backbone** (development sizes, pre-registered budget): learning rate, optimizer, replay size, batch size, target-update cadence, network width. Inherited identically by all cells. *(Bootstrap-specific parameters cannot be tuned on a single-head DDQN and are excluded from this class.)*
  2. **Ensemble-shared nuisance — fixed to explicitly listed values, not tuned.** The pre-registration contains a table — parameter | value | source or justification — covering bootstrap mask probability, head-loss aggregation/normalization, shared-trunk convention, head initialization, and per-head target handling. Each value cites the specific section/experiment of Osband et al. (2016), or is labeled a **documented design choice inspired by Osband et al.** where no exact published default exists (their Atari experiments used mask p = 1 — there is no universal "Osband default" for DeepSea). Identical across all cells.
  3. **Factor-specific parameters (the only per-factor tuning; objectives pinned, v6.3):** `prior_scale` — one mini-search, selected by **IQM of the canonical prior-on cell `(episodic, on, 10)` on development sizes**, value shared by all prior=on cells; `eps_schedule` for ensemble_mean_eps — one mini-search, selected by **IQM of `(mean_eps, off, 10)` on development sizes**, shared by its cells at both prior levels. **Ties broken by the lower parameter value.**
  Nothing else is tunable per cell. C13 audits resolved configs against exactly this three-class structure.
- **Contrast definitions:** rule-vs-rule; interventions-as-practiced (§1).
- **Ceiling cell:** `ensemble_mean_greedy` (no ε) as a pure capacity-only control — exploratory, behind Gate B.

### Ceiling (behind Gate B; top-down; all exploratory): E5 UCB + spread-temperature · ensemble_mean_greedy cell · per-budget re-tune spot check · MiniGrid + Classic Control · ensemble-Thompson/PSRL · bonus anchor.

### Out of scope — unchanged (no novel algorithm; no ALE; no continuous control; no model-based; no LLM environments; no environment building; no full factorial across *families* — and, v6.3, no completion of the 18-cell full factorial: the missing cells are a documented limitation, not a to-do).

## 3. Methodology requirements

### 3.1 Pre-registered protocol

- **Frozen-rule + measured-input pattern (governing):** decision rules with later-measured inputs are permitted (wall-clock, dev-pilot solve rates), provided rule, thresholds, and input provenance (development/tuning data only) are frozen. No deferred choices.
- **Equal search budget** — the *defined* matched-search-effort device (identical tiers/counts/statistic across the canonical four); per-method random-search sensitivity curves reported. **We apply the matched-search-budget evaluation standard established by Ceron (2021) and Taïga (2021) to randomized-value exploration methods** — never framed as an original idea, and (v6.3) never framed as a "first application."
- **DeepSea development/confirmatory split**, plus the **confirmatory-size selection rule (numerically exact):** inputs = development-pilot solve counts at N=20, on **exactly 10 development seeds each**, of (a) the ε-greedy DDQN reference and (b) the canonical episodic cell `(episodic_head, prior=off, K=10)`, measured before Gate A. Frozen mapping with **down-shift precedence**: if (b) ≤ 2/10 → shift down to {22, 26, 30, 34, 38}; else if (a) ≥ 9/10 → shift up to {40, 45, 50, 55, 60}; else default {30, 35, 40, 45, 50}. Inclusive integer thresholds; raw proportions. Rationale: published ε-greedy failure onset near N≈10–15; ensemble solvability through N≈50 at bsuite budgets.
- **MinAtar tuning/evaluation split and claims template** — unchanged.
- **Two-stage freeze:** (1) fill all freeze-list values → `prereg-draft` tag; (2) external pass on the valued draft (one week; silence → substitute reviewer; none within a further week → documented methodological-review waiver); (3) fixes → **final tag + OSF mirror** (the freeze); (4) scientific runs only after the final freeze (infrastructure work may precede it; the pipeline spine may not).
- **Pre-registration amendment policy (Gate A's formal channel):** any change post-final-freeze but pre-confirmatory-execution = versioned amendment (new tag + OSF timestamp, reasons, enumerated changed items, disclosure of motivating development outcomes); confirmatory runs only after the amendment freeze. Post-confirmatory changes remain governed by the confirmatory-integrity rule (§3.6) — amendments cannot rescue them.
- **THE FREEZE LIST (20 items):**
  1. Config/seed counts per tier and stage (DeepSea dev: 10 seeds for the two rule-input runs, 5 elsewhere; 20/cell confirmatory; MinAtar held-out 10) **and the cell-specific RNG derivation scheme (§1.1 seed-independence rule)**.
  2. Search distributions per hyperparameter; the backbone-tuning budget; the two factor-specific mini-budgets **with their pinned selection objectives and tie-breakers**.
  3. Selection statistic (IQM) + tie-breaking.
  4. **Final-tier trigger (method-specific):** final tier runs iff Σ over the four methods of (final-tier run count × that method's median pilot-tier per-run wall-clock) ≤ X GPU-hours; X fixed at freeze; inputs from each method's own pilot-tier runs.
  5. DeepSea development sizes; the confirmatory-size selection rule with its three size sets; episode budget per N.
  6. Reporting windows.
  7. Probe-set construction + weighting (uniform primary; visitation-weighted secondary).
  8. **Operational failure policy** (§3.6) incl. divergence criterion and infrastructure/algorithmic classification.
  9. Bootstrap units, stratification, CI construction; the §1.1 aggregation and fixed-sequence procedure; the RQ2-L statistic in full.
  10. Frozen-policy extraction rules per method.
  11. **t₀ landmark rule** (10% of episode budget; fallback to 5% iff **the canonical episodic cell's 10 development seeds** show >2/10 succeeding by t₀ — v6.3 pinned population) and post-t₀ outcome definitions.
  12. **The 10-cell structured partial factorial**, four contrasts, shared-configuration rule (three classes), factor-specific parameter list, and the stated interaction limitations.
  13. **The primary estimand hierarchy (§1.1)** — contrasts, settings, outcomes, aggregation, test order, SESOI.
  14. **Battery formulas, sample counts, aggregation, and micro-conventions (§3.3)** — incl. the action-gap tie rule, quantile interpolation method, rank-biserial formula and sign, and the C-K score coding + permutation procedure.
  15. **The canonical MinAtar four** with defining parameters and the K_shared joint rule.
  16. QR-DQN exploratory status, hypothesis, and test (neutrality rule).
  17. The descope ladder **and its objective trigger:** after pilot-tier wall-clock measurement, projected total compute is computed; if it exceeds the frozen cap **Y GPU-hours**, rungs apply in fixed order until ≤ Y; executed **before Gate A and before any confirmatory outcome**. Rung 1 sets the pre-launch horizon to 500k (pre-launch only); **rung 2 removes C-K from the confirmatory sequence per §1.1** and reduces the design to the mandatory six-cell core + surviving K cells. Ladder application after confirmatory execution is prohibited.
  18. **The two-stage freeze and amendment policy**, incl. the external-pass time-box and substitute/waiver rule.
  19. **Parameter classes** (backbone-tuned / literature-fixed ensemble-shared **parameter table** / factor-specific), the K_shared rule, and the SESOI Δ = 0.10 with its equivalence interpretation.
  20. **The deterministic MinAtar-cloning conditional and the undefined-value policy** (§3.3).

### 3.2 Evaluation methodology

- **Two reporting axes:** (1) online return at budget checkpoints (primary; acting policy is the method under its own definition); (2) standardized frozen-policy evaluation (secondary; pre-registered extraction: DDQN → greedy(Q); NoisyNet → noise-off greedy; bootstrapped cells → greedy w.r.t. ensemble-mean Q; QR-DQN → greedy(quantile mean)).
- **Direct exploration outcomes:** DeepSea — solve-probability vs. depth, discovery probability, episodes-to-first-success, max depth; MinAtar (exploratory proxies) — time-to-first-reward, hashed-observation coverage.
- **Budget axis (Variant B):** every MinAtar held-out run proceeds **unconditionally to the pre-launch frozen horizon — 1M by default, or 500k under a pre-launch rung 1.** Once launched: no selective continuation, no early stopping, no schedule changes, no dropping of runs (failures governed solely by §3.6). RQ3 is a pre-specified secondary analysis on data that exist by design (or reported as descoped).
- **Primary RQ1 statistics:** probability of improvement over the tuned baseline (rliable); raw per-game score differences with stratified bootstrap CIs; performance profiles. **Ratio metrics deleted entirely.**
- **MinAtar reporting boundary:** three per-game results primary; aggregate as compact summary; no general-ranking claim.
- **Capacity/compute table** next to the RQ1 headline. **Evidence tiers:** pilot-labeled plots never cited; paper claims rliable-only.

### 3.3 Uncertainty diagnostics — mathematical definitions (all frozen)

Notation: probe set S; M value samples {Q_m} (M = K heads; M = 30 i.i.d. NoisyNet draws at measurement only); Q̄ = mean; σ(s,a) = sample std; Q\* exact on DeepSea.

1. **Marginal alignment:** Spearman ρ over (s,a) ∈ S × A between σ(s,a) and |Q̄(s,a) − Q\*(s,a)|. *(RQ2-L primary diagnostic.)*
2. **Action-gap alignment:** a₁, a₂ = top-2 by Q̄ (**ties by lowest action index**); ĝ(s) = Q̄(s,a₁) − Q̄(s,a₂); **g\*(s) = Q\*(s,a₁) − Q\*(s,a₂) using the same a₁, a₂**; u_g(s) = std over m of [Q_m(s,a₁) − Q_m(s,a₂)]. Statistic: Spearman ρ between u_g(s) and |ĝ(s) − g\*(s)|.
3. **Incorrect-argmax flagging:** optimal set = Argmax_a Q\*(s,·); e(s) = 1[argmax Q̄ ∉ Argmax Q\*]; modal-action ties among samples by lowest action index; d(s) = 1 − modal fraction. **Statistic: rank-biserial r = 2·(AUC − 0.5)** (AUC = probability a uniformly drawn incorrect-argmax state carries higher d(s)); positive = greater disagreement at incorrect states.
4. **Optimal-path uncertainty:** mean σ(s, a\*(s)) along the optimal path (per depth; AUC over depth as summary).
5. **Visitation-conditioned decay:** OLS slope of log σ(s, a\*(s)) on log(1 + v(s)) over raw probe states (bins for display only); unweighted; per checkpoint; σ = 0 excluded and counted.
6. **Temporal persistence:** within-episode fraction of probe states whose greedy action under the current sample equals the episode-start sample's; per-episode mean, per checkpoint (across consecutive samples for per-step rules; ~1 by construction for episodic — descriptive).
7. **Empirical containment:** central 80% interval from empirical quantiles of {Q_m(s,a)} — **`numpy.quantile(..., method="linear")`**; containment = fraction of (s,a) with Q\* inside. Descriptive; level fixed.
8. **MinAtar behavior-policy analogue — deterministic conditional:** 100 clone/restore reproduction tests (stored state + RNG state, fixed action sequence, two replays): full probe rollouts iff all 100 bit-exact; episode-start-only iff cloning fails but seeded fresh-reset rollouts are bit-reproducible; drop otherwise. Spike outputs committed. Exploratory, appendix-only; never "Q-error."
9. **Undefined-value policy (battery-wide):** undefined statistics recorded NA, excluded from aggregation, exclusions counted and published; nothing imputed; σ = 0 is a substantive measurement where it is a value rather than a divisor. The RQ2-L <10-comparable-pairs rule is an instance.

### 3.4 Compute, run budget, descope ladder

Single-workstation reproducibility of reported results remains the identity claim. **Run budget:** MinAtar tuning ≈ 240 pilot + 800 final (formula-gated) @500k; MinAtar held-out ≈ 120 runs to the pre-launch frozen horizon; DeepSea dev ≈ 150–250 + confirmatory ≈ 1,100 (10 cells × 5 sizes × 20 seeds + baseline/NoisyNet ride-along); QR-DQN exploratory ≈ 300. Order of 2,500–3,000 runs; per-method wall-clock measured pre-final drives the frozen trigger.

**Descope ladder (trigger per freeze item 17):** applied only by the frozen compute-cap rule, before Gate A and before any confirmatory outcome; rung 1 pre-launch only. Order: (1) pre-launch horizon → 500k (RQ3 descoped) → (2) K-axis → {10, 20} — *C-K exits the confirmatory sequence per §1.1* → (3) final-tier configs down (identical across methods) → (4) held-out seeds 10 → 5 with sensitivity note → (5) drop MinAtar E-cell spot-check → (6) drop the QR-DQN exploratory follow-up. **Never cut:** the **six-cell `use_rule × prior` core at K = 10**, dual axes, the EWRL figure, the protocol, DeepSea confirmatory seeds.

**Environment pinning + bsuite fallback:** unchanged.

### 3.5 Negative-results clause — unchanged (informative iff diagnostics + contrasts distinguish absence of effect from insensitivity; sensitivity curves; direct exploration outcomes; MDE statement with any null).

### 3.6 Operational failure policy — unchanged (infrastructure failure → rerun same (config, seed), classified by error type; algorithmic failure → is the result; implementation bug → confirmatory block void, new pre-registered iteration; reproducibility reruns excluded; seed counts never increase post-results).

## 4. Repository requirements — unchanged (CleanRL-style; determinism from (config, seed) **via the v6.3 cell-specific derived streams**; CSV; `make figures`; two-stage tags; CI smoke + C11 per-contrast purity + factorial config schema; C13 configuration-identity audit committed with figures; `VALIDATION.md` covers C0–C13; public at M1).

## 5. Paper requirements

**Working title:** *"When Do Randomized Value Estimates Buy Exploration? Separating the Estimator from the Use Rule under Low Interaction Budgets."* Structure: motivation → protocol → **Part A: DeepSea mechanism study** (hierarchy-shaped) → **Part B: external performance evaluation (MinAtar)** → RQ3 → honest limitations (incl. the partial-factorial interaction limitation). Replication-and-extension framing; the Taïga pre-emption paragraph; QR-DQN in a marked exploratory section. Claims discipline: scoped claims; contrast-name citations; no causal "predicts"; no "transfer"; no "first"; no original-fairness; **no "full factorial."** Venue: RLC 2027 primary (verify CFP — the site currently publishes RLC 2026 only; Q1 2027 is a working assumption); workshop tracks and EWRL 2027 fallbacks. Related work from the A2 corpus (**structured bibliometric survey**). Authorship: self + HSE advisor; slot open for a Lille contributor.

## 6. Deliverables & gates

**Milestone 1 / v0.1:** 4 canonical methods on MinAtar (≥5-seed pilots) · switchboard under the shared-configuration rule (C11+C13 green) · development mechanism pilot complete (pilot-labeled figure) · confirmatory DeepSea sweep executed once on the rule-selected sizes, post-Gate-A; confirmatory mechanism figure + §1.1 primary estimand computed first · MinAtar held-out evaluation to the pre-launch frozen horizon, both axes · 1 EWRL-grade figure · honest README · repo public.

**GATE A — advisor protocol review (BEFORE both confirmatory blocks):** frozen protocol + freeze list + pilot evidence from development sizes and tuning games only; the rule-selected size set reviewed as an application of a frozen rule. Time-box: 3 weeks → documented deviation + substitute review. Design changes enacted through the amendment policy; confirmatory runs only after the amendment freeze.

**CONFIRMATORY-INTEGRITY RULE:** post-result design changes on either environment family demote affected analyses to exploratory, permanently, with §3.6 as the operational arm. Gate B shapes interpretation and exploratory extensions only.

**GATE B — advisor results review (before any expansion):** interpretation + exploratory extensions only; sees budget-axis data by design (harmless); no new methods enter any confirmatory analysis. Both gates are letter-production activity (pivot plan §6).

**v1.0 (submission gate):** protocol executed as pre-registered (deviations/amendments documented) · structured partial factorial complete on the selected confirmatory sizes — incl. C-K, or with the rung-2 substitution documented · §1.1 hierarchy reported exactly as frozen · RQ2-L per spec · both axes at all checkpoints · sensitivity curves + capacity/compute table · C13 outputs committed · QR-DQN exploratory section (or descope documented) · RQ3 secondary analysis (or descope documented) · `make figures` end-to-end · compute reported · zero untraced claims · limitations advisor-signed.

## 7. Positioning (claim-level contribution map — corrected, v6.3)

**Stated contributions (phrased as "we did not identify," never "no one has"):**
- **C-i — the structured partial factorial.** We did not identify any prior study combining a controlled `use_rule × prior` factorial with a pre-specified ensemble-size axis, exact-Q\* diagnostics, and a matched confirmatory protocol. The foundational papers (Osband 2016/2018/2019 — the RVF paper A1 explicitly extends) establish the components; Osband 2016's ensemble-policy and per-step "Thompson DQN" variants are single-setting chain sketches — the honest lineage of C-USE and C-COHERENCE. **The BDQN-scaling preprint already occupies much of the prior × K axis on DeepSea** (reward-discovery vs. hardness and ensemble size; posterior collapse; prior functions sustaining diversity — verify review status before citing); **EVE (Exploration via Epistemic Value Estimation)** already ablates acting rule and value bootstrapping and measures continuous epistemic uncertainty on DeepSea. A1's residual C-i novelty is therefore the *composition*: the controlled use-rule axis inside one purity-audited design, crossed with prior and K interventions, under a pre-registered estimand hierarchy. The different-axis disentanglements (MULEX — explore-vs-exploit; Clements 2019 — epistemic-vs-aleatoric; Plappert 2018 — action- vs. parameter-space noise, without an estimator axis) remain adjacent, not occupying.
- **C-ii — exact Q\* as error ground truth.** Not "the first continuous measurement on DeepSea" (EVE measures continuous epistemic uncertainty there): the contribution is **exact-Q\*-referenced measurement of estimator error, decision-margin error, and incorrect-action risk across a controlled randomized-value design**. DeepSea was introduced in the randomized-value/prior-functions line of work and later standardized in bsuite; its users largely keep the pass/fail spirit (bsuite; HyperAgent's episode-scaling results). **Janz 2019 is the analytic precursor that motivates rather than pre-empts the design**; Russo 2019 supplies the tabular regret theory. The **July-2026 distributional-risk audit** (existence verified 2026-07-15; read at Session 2) ground-truth-audits QR-family risk claims on MinAtar — methodologically kindred, which is exactly why C-ii is scoped to randomized-value estimates on DeepSea and QR-DQN stays exploratory.

**Inherited standards (claimed as application, never as ideas, never as "first"):**
- **C-iii — matched-budget fair baselines.** We apply the matched-search-budget evaluation standard established by Revisiting Rainbow (Ceron 2021) and Taïga (2021) to randomized-value exploration methods.
- **Taïga pre-emption (frozen paragraph):** Taïga's finding — fair evaluation shrinks exploration-method advantages — motivates asking the same question for randomized-value estimators, **where the estimator/use-rule decomposition lets A1 say *why* an advantage appears or disappears rather than only *that* it does.**

**Stated design choices (never novelty):**
- **C-iv — MinAtar at low budgets** (Young 2019; Clements 2019 precedent). Sensible experimental design.
- **C-v — the matched head-to-head** (DDQN, NoisyNet, BDQN, RP-BDQN; QR-DQN exploratory). **Narrowed null (v6.3):** *we did not identify* a study comparing this exact canonical set under one matched MinAtar tuning-and-evaluation protocol — MinAtar exploration and uncertainty comparisons do exist in general. Execution-quality contribution, coupled to Part A.

**C-PRIOR framing:** A1 studies randomized prior functions as one controlled intervention inside the bootstrapped-value family, complementing **Priors Matter (2025)** on prior/likelihood misspecification in Bayesian deep Q-learning (MinAtar + DeepSea experiments; verify at Session 2) and the BDQN-scaling diversity findings.

**Remaining neighbor notes:** UA-DQN (tune-on-Breakout protocol precedent); UPER; β-DQN (prior on modest bootstrapped gains; feeds §3.5); Neural Testbed line (design input for the battery).

**Scoop-check (v6.3):** the narrowed null results are recorded with "we did not identify under this search protocol" phrasing and **re-run at final freeze and at submission**. Session 2's `positioning.md` absorbs the corrected novelty audit and must include a **reproducible-search appendix**: search date, databases, exact queries and filters, anchor papers, forward/backward citation sources, retrieval counts before/after deduplication, screening rules, full-text review list, null-result queries, known limitations.

**Evidence hygiene:** no corpus counts as gap evidence; A2 = structured bibliometric survey.

## 8. Implementation order

1. DDQN on MinAtar Breakout, 1 seed, CSV logs — preceded in Session 0 by the throwaway infrastructure benchmark (informs caps X/Y)
2. Determinism/config/logging discipline — **incl. the cell-specific RNG derivation scheme**
3. `make figures` one curve
4. 5-seed baseline; per-method wall-clock recorded (trigger inputs)
5. Structured-partial-factorial switchboard + backbone-tuning pass (class 1 frozen; class 2 fixed via the parameter table) + disagreement logging
6. DeepSea integration — development sizes only; pilot solve-vs-depth (pilot-labeled)
7. Randomized priors (`prior_scale` mini-search per its pinned objective)
8. NoisyNet-DQN (+ `eps_schedule` mini-search per its pinned objective)
9. RQ2-Q battery on development sizes (Q\* validated on hand-checkable N)
10. Mechanism pilot on development sizes: **10 dev seeds for the canonical episodic cell and the ε-greedy reference (rule inputs), 5 for other cells**; t₀ input measured on the canonical cell; size-rule inputs measured; MDE report; pilot mechanism figure
11. MinAtar pilot + pre-registered tuning (K_shared rule; trigger + descope formulas in the open)
12. **GATE A**
13. **Confirmatory block:** DeepSea confirmatory sweep (once; §1.1 primary computed first) + MinAtar held-out runs to the pre-launch frozen horizon → **M1 release; repo public**
14. **GATE B**
15. RQ3 secondary analysis (no new runs)
16. QR-DQN exploratory follow-up
17. Ceiling, top-down

## 9. Advisor packaging

The advisor artifact is the 1-page protocol doc (touch #2): mechanism framing + the contrast table + the primary-estimand sentence + replication-and-extension framing + the two-part design + venue logic + Milestone 1 scope + the fairness protocol (as inherited standard) + the nearest-neighbor map + the two-gate structure + the precise ask. **Safe novelty formulation (v6.3, adopted verbatim):** *"Existing work has separately demonstrated the importance of episodic commitment, randomized priors, ensemble size, and epistemic-value estimation. This project brings these strands into one pre-registered structured design and evaluates randomized value estimates against exact, decision-relevant ground truth on DeepSea, followed by a matched external performance evaluation on MinAtar."* **Content rule:** include — the title; the bundling problem; the honest lineage; the contribution (C-i + C-ii as composition); the primary estimand (C-COHERENCE); the exact-ground-truth DeepSea design; the MinAtar evaluation; the precise ask. Exclude — caps X/Y, the descope ladder, clone/restore tests, C13, battery formulas, QR-DQN, the full gatekeeping sequence. **Banned:** corpus counts, expected rankings, "any negative result is publishable," "first," "no one has," "full factorial," original-fairness claims.

---

# Part II — Alternatives assessment

## 1–2. Criteria (C1–C7) and candidates — unchanged.

## 3. Recommendation

**A1 survives review #7 and the corrected novelty audit with naming, statistics, and claims-precision fixes only.** The strongest-defended contribution is the exact-Q\*-referenced, decision-relevant battery applied across a controlled bootstrapped-value design; the moderately defended contribution is the systematic replication-and-decomposition composition (use-rule effects × prior and ensemble-size interventions); matched fairness, MinAtar, low budgets, and pre-registration are quality attributes, not contributions — and the documents now say exactly that. Status: **final architectural version**; remaining loop = fill values → draft tag → external pass → final freeze; the one-pager is unblocked and its safe novelty sentence is written.

**Refined one-liner (v6.3):** *Bring episodic commitment, randomized priors, ensemble size, and epistemic-value measurement into one pre-registered structured partial factorial with exact decision-relevant ground truth on DeepSea — one primary contrast, one shared configuration, one pass over the confirmatory data — then evaluate the same methods head-to-head on MinAtar under the inherited matched-budget standard.*

**Action items:**
- [ ] Fill freeze-list values → `prereg-draft` tag → external pass (substitute/waiver rule) → **final tag + OSF** → runs.
- [ ] Build the one-pager per §9 (safe novelty sentence verbatim); send to the advisor as touch #2.
- [ ] Session 2: verify EVE, Priors Matter, the distributional-risk audit, and the Osband 2016 ablation/parameter details against the papers; fold the corrected novelty audit + reproducible-search appendix into `positioning.md`; re-run the narrowed null searches.
- [ ] Feasibility spike (deterministic conditional); BDQN-scaling status check.
