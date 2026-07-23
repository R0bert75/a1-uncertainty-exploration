# Positioning & prior-work map — A1

*Single source of truth for how* **"When Do Randomized Value Estimates Buy Exploration?
Separating the Estimator from the Use Rule under Low Interaction Budgets"** *sits in the
literature. This file absorbs the corrected claim-level novelty audit
(`novelty_deep_dive.md`, `novelty_claims.md`, `novelty_threat_matrix.csv`) and extends it
with a newer neighbor tier verified paper-by-paper in Session 2. The reproducible-search
appendix (§A) documents exactly how the picture was assembled. Every DOI/arXiv id below was
retrieved and its abstract read; nothing here is asserted from memory.*

> **Language discipline (enforced throughout, and a review-safety rule for the paper).**
> No first-ness language ("first," "no one has," "novel benchmark"). No "full factorial"
> (the design is a **structured partial factorial** — K × use_rule and three-way
> interactions are not estimable). No original-fairness language (the matched-budget
> baseline is an **inherited standard**, not an invention). Null results are phrased
> "we did not identify, under this search protocol," never "there is no such work."

---

## 1. What A1 claims, and how each claim survives

A1 makes five claims. The audit's severity rubric: **3** = a paper does essentially this
(A1 would be a replication of it); **2** = adjacent (uses the same testbed or algorithms but
not the same separation/matching); **1** = touches one component; **0** = unrelated. A1
survives on a claim if the best threat is **≤ 2** and we can say exactly what that work did
*not* do.

| Claim | Statement | Best prior overlap | Verdict |
|---|---|---|---|
| **C-i** | Factorial **separation of estimator** (randomized/bootstrapped value ensemble ± prior functions, size *K*) **from use rule** (episodic deep-exploration commitment vs per-step action noise) as independent, pre-registered crossed axes | 2 (adjacent disentanglements on *other* axes) | **Clear — load-bearing** |
| **C-ii** | **DeepSea's exact optimal Q\*** used as *continuous estimator-quality ground truth* across the factorial, not as a pass/fail scaling probe | 2 (DeepSea used, but pass/fail) | **Clear, partial overlap** |
| **C-iii** | A genuinely **well-tuned ε-greedy Double DQN** on a matched search budget | 3 (established methodology) | **At-risk → reframe as adopted standard** |
| **C-iv** | **MinAtar at low interaction budgets** as the external-validity axis | 3 (already common practice) | **Not a novelty claim — venue choice** |
| **C-v** | **One matched protocol** across ε-greedy DDQN, NoisyNet, Bootstrapped DQN, RP-BDQN (+ QR-DQN exploratory) | 2 (common-framework re-eval exists for other method sets) | **Clear, partial** |

**The two contributions to state as contributions are C-i and C-ii.** C-iii and C-iv are
presented as *adopted rigor* and *sensible experimental design*; C-v derives its weight from
being coupled to the Part A mechanism findings, not from the head-to-head existing in
isolation.

### C-i — estimator vs use rule (the core)

The field's exploration comparisons are overwhelmingly *algorithm-level*: whole named agents
pitted against each other rather than decomposed into "how the value randomness is generated"
vs "how that randomness drives behaviour." The foundational randomized-value work establishes
the components A1 varies but never crosses them as a designed factorial. Osband 2016
(arXiv:1602.04621) introduces bootstrapped ensembles for temporally-extended (episodic)
exploration; Osband 2018 (arXiv:1806.03335) adds randomized prior functions and shows they
matter; Osband 2019 (arXiv:1703.07608), the paper A1 explicitly extends, unifies randomized
value functions under a regret argument. In each, the estimator and its episodic use are
bundled into a single proposed method — there is no on/off × use-rule × K switchboard.

The papers that *do* say "disentangle" disentangle something else. **MULEX** (Beyer 2019,
arXiv:1907.00868) separates exploration from exploitation via parallel losses — a real
decomposition, but on the explore-vs-exploit axis. **Clements 2019** (arXiv:1905.09638)
disentangles epistemic from aleatoric uncertainty — a real separation, but of *uncertainty
types*. **Plappert 2018** (arXiv:1706.01905) is the closest thing to a use-rule contrast in
the classical literature (action-space vs parameter-space noise) — a "how is the perturbation
applied" question, but not crossed with an ensemble/prior estimator axis, and run on
continuous control / Atari, not a controlled chain. A1's crossing of `use_rule × prior × K`
as pre-registered independent factors is, to the depth of this search, unoccupied.

### C-ii — DeepSea exact-Q\* as a mechanism testbed

DeepSea is popular, which initially looks threatening, but the *way* A1 uses it is not.
DeepSea originates in bsuite (Osband 2020, arXiv:1908.03568), which by design reports a scalar
pass/fail scaling score — the size of chain still solvable — not a continuous diagnostic of
value-estimate quality. The strongest DeepSea users keep that pass/fail spirit: **HyperAgent**
(Li 2024, arXiv:2402.10228) reports optimal episode-count scaling with problem size;
**Successor Uncertainties** (Janz 2019, arXiv:1810.06530) uses sparse-reward chains to show
that many randomized-value-function-plus-NN methods *lack* the posterior-sampling properties
that make them explore, and provably fail. Janz is the most intellectually adjacent paper to
A1's mechanism question — a diagnostic of estimator quality against an exploration use case —
but it proceeds analytically and proposes a new algorithm; it does not use the known optimal
Q\* as continuous ground truth to score estimator error cell-by-cell across a crossed design.
The regret theory behind this (Russo 2019, arXiv:1906.02870) is tabular and non-empirical.
C-ii overlaps on the *testbed* but not on the *measurement*, and the measurement is the
contribution. **Cite Janz as the analytic precursor that motivates the design, not one that
pre-empts it.**

### C-iii — well-tuned baseline (reframe, do not claim)

A1's fairness thesis is **not new as a methodological idea**, and A1 is stronger for saying
so. That a carefully tuned strong baseline plus a common evaluation framework can overturn
published component-importance conclusions is an established result. **Ceron 2021**
("Revisiting Rainbow," arXiv:2011.14826) makes exactly this argument on small/mid-scale
benchmarks; **Taïga 2021** ("On Bonus-Based Exploration Methods in the ALE,"
arXiv:2109.11052) is the most direct precedent — it re-evaluates popular exploration methods
in one common framework on a strong Rainbow baseline and finds the specialised bonuses'
advantage largely evaporates outside the few hard-exploration games they were designed for.
That is A1's fairness logic, already published — but for *bonus-based* exploration on the
full ALE, not for randomized-value estimators crossed with a use rule on DeepSea and MinAtar.
Mitigation costs nothing scientifically: cite Ceron and Taïga as the inherited precedent,
frame the matched-budget baseline as "applying an accepted rigor standard to a method family
it has not yet been applied to," and drop any language implying the fair-baseline idea is
original.

### C-iv — MinAtar low-budget venue (demote to design)

This should not be advanced as novelty; doing so would be a review liability. MinAtar exists
precisely to make thorough, reproducible small-scale Atari-like experiments cheap (Young
2019, arXiv:1903.03176), and uncertainty-aware DQN variants have already been evaluated on it
(Clements 2019 reports its uncertainty-aware DQN outperforming other DQN variants on MinAtar).
Low-interaction-budget evaluation is likewise standard. The novelty is not that MinAtar is
used or that budgets are small — it is *what* is run there: the four/five methods that fall
out of the Part A factorial under a single matched protocol.

### C-v — matched head-to-head (couple to Part A)

No paper in the search places NoisyNet, Bootstrapped DQN, RP-BDQN, and QR-DQN head-to-head
under a single matched training/tuning/seed/metric protocol. The component anchors are each
canonical — NoisyNet (Fortunato 2018, arXiv:1706.10295), QR-DQN (Dabney 2018,
doi:10.1609/aaai.v32i1.11791) — but were introduced in separate papers each with its own
baseline and evaluation. Two targeted searches for a bootstrapped-vs-NoisyNet exploration
comparison, and for any MinAtar study crossing bootstrap/noisy/ensemble exploration methods,
returned zero. The nearest precedent is again the *style* of Taïga's common-framework
re-evaluation, applied to a different method set. C-v is an execution-quality contribution;
its weight comes from coupling to the Part A mechanism findings.

---

## 2. Closest prior work (classical tier)

| Paper | What it did | Claims touched | What it did **not** do |
|---|---|---|---|
| Osband 2019 — Deep Exploration via RVF (arXiv:1703.07608) | Unified randomized value functions + regret bound; the paper A1 extends | C-i, C-ii | No estimator×use-rule factorial; tabular ground truth not used as a continuous estimator-quality metric |
| Janz 2019 — Successor Uncertainties (arXiv:1810.06530) | Showed RVF+NN methods lack PSRL properties, fail in sparse chains | C-i, C-ii, C-v | Analytic + one new algorithm; not a crossed pre-registered design |
| Beyer 2019 — MULEX (arXiv:1907.00868) | Disentangles exploration from exploitation via parallel losses | C-i | Wrong axis: explore-vs-exploit, not estimator-vs-use-rule |
| Plappert 2018 — Parameter-Space Noise (arXiv:1706.01905) | Contrasts action-space vs parameter-space noise | C-i | Use-rule contrast only; no ensemble/prior estimator axis; not DeepSea/MinAtar |
| Taïga 2021 — Bonus-Based Exploration in ALE (arXiv:2109.11052) | Common-framework re-eval vs strong Rainbow; the gap shrinks | C-iii, C-v | Bonus methods on full ALE, not randomized-value estimator×use-rule on DeepSea/MinAtar |
| Ceron 2021 — Revisiting Rainbow (arXiv:2011.14826) | Strong baselines on small benchmarks overturn component-importance claims | C-iii | Rainbow components, not exploration mechanisms |
| Clements 2019 — Estimating Risk & Uncertainty (arXiv:1905.09638) | Disentangles epistemic/aleatoric uncertainty; evaluated on MinAtar | C-i, C-iv | Wrong disentanglement (uncertainty types); one algorithm, no factorial |
| Li 2024 — HyperAgent (arXiv:2402.10228) | Q\*-posterior approximation; solves DeepSea with optimal episode scaling | C-ii | Uses DeepSea for scaling, not exact-Q estimator diagnostics; one new algorithm |

Full severity scoring is in `novelty_threat_matrix.csv` (16 papers × 5 claims).

---

## 3. Newer neighbor tier (verified in Session 2, 2026-07-23)

Four newer neighbors were named in the Session-2 brief for direct verification. Three were
located and their abstracts read; the fourth was not located under this search protocol.

### 3.1 EVE — Exploration via Epistemic Value Estimation (arXiv:2303.04012, 2023; AAAI-2023 class)

EVE proposes a recipe giving an agent a tractable posterior over *all* its parameters, from
which epistemic value uncertainty is computed and used to drive exploration; it reports
competitive performance on hard-exploration benchmarks (DeepSea among them). **Relation to
A1:** EVE is a *single new estimator* with an epistemic-uncertainty acting rule; it studies
acting-rule and bootstrapping ablations *within its own method*, but it does not cross an
estimator axis against a use-rule axis as independent pre-registered factors, and it uses
DeepSea for capability, not for exact-Q\* estimator-error diagnostics. It is important for one
reason: it means **A1 must not claim "all prior work compares whole algorithms" nor "all
DeepSea work is pass/fail."** EVE ablates within a method and uses DeepSea as a hard task —
A1's distinction is the *crossed, pre-registered* separation plus the *continuous* Q\*
diagnostic, and the positioning sentence must be that precise.

### 3.2 Priors Matter — Misspecification in Bayesian Deep Q-Learning (arXiv:2508.21488, Aug 2025; preprint)

Shifts attention from posterior-approximation accuracy to the accuracy of the *prior and
likelihood assumptions*. Demonstrates a **cold-posterior effect** in Bayesian deep Q-learning
(performance improves when the posterior temperature is reduced, contrary to theory), shows
via statistical tests that the common Gaussian-likelihood assumption is frequently violated,
and studies prior distributions directly; experiments span MinAtar and DeepSea. **Relation to
A1:** complementary, not competing. It interrogates *whether the Bayesian assumptions hold*;
A1 asks *when the randomized-value estimator, as practiced, buys exploration behaviour, and
under which use rule.* It **frames C-PRIOR as complementary** — A1's `prior on/off` axis
measures the behavioural payoff of the prior intervention, while Priors Matter explains, at
the assumption level, why priors can help or hurt. Cite it as motivation for treating the
prior as a *factor to measure* rather than a setting to assume.

### 3.3 Auditing the Risk Claims of Distributional RL (arXiv:2607.11607, Jul 2026; preprint)

Audits whether the risk claims of trained distributional agents are *true*, using a
decision-relevant screening metric (the excess-Wasserstein gap between the top two actions),
snapshot-restart Monte-Carlo ground truth, and a permutation/bootstrap/FDR statistical
harness. Across **QR-DQN, C51, IQN on MinAtar (33 runs)**, it finds 40–95% of the strongest
claimed risk trade-offs refuted at 95% confidence. **Relation to A1:** it independently
**confirms QR-DQN's exploratory (non-confirmatory) weight** in A1 and scopes C-ii — its
message is that distributional agents' return-distribution claims often fail a direct audit,
which is exactly why A1 keeps QR-DQN as a *pre-specified exploratory distributional control*
rather than a confirmatory arm, and confines the confirmatory estimator-quality claims to
randomized-value estimates on DeepSea. Its rigor apparatus (permutation nulls, bootstrap
refutation, FDR) is a methodological sibling to A1's own confirmatory-integrity design.

### 3.4 BDQN-scaling preprint (prior × K on DeepSea; posterior collapse / diversity) — **not located**

The Session-2 brief flagged, as the top neighbor tier, a preprint studying `prior × K` on
DeepSea with posterior-collapse and ensemble-diversity findings, review status to be verified.
**Under this search protocol it was not located** — it was never pinned to a concrete
arXiv/DOI identifier in any prior session, and the targeted queries in §A.4 returned no
matching record. This is recorded as a **null result, not an absence claim**: if this preprint
is a genuine near-twin of A1's `prior × K` sub-design, it would be a **severity-3 threat to
C-i and C-ii on the K axis specifically** and must be located and characterized before the
final freeze. **Action carried to the external-review window / Session-3 pilot freeze:** ask
the reviewer to name it if they know it, and re-run the neighbor search closer to freeze.

---

## 4. The Taïga pre-emption paragraph (drop-in for the paper's related work)

> A reviewer will raise Taïga et al. (2021), and A1 should pre-empt it. Taïga et al. show that
> when bonus-based exploration methods are re-evaluated in one common framework against a
> strong, equally-tuned Rainbow baseline, most of their advantage over the baseline
> disappears outside the handful of hard-exploration games they were designed for. We take
> that finding as the motivation for our question rather than as a result we contest: it
> establishes that fair, matched-budget evaluation is the right lens, and it did so for
> bonus-based methods on the Arcade Learning Environment. We ask the same question of a
> different family — randomized-value estimators (bootstrapped ensembles, with and without
> randomized prior functions) and the use rule that turns their value uncertainty into
> behaviour — and we add an ingredient Taïga et al. did not have: on DeepSea the exact optimal
> Q\* is known, so when a method's advantage appears or disappears we can say *why* in terms of
> estimator quality, not only *that* it does. Our matched-budget ε-greedy Double DQN baseline
> follows the standard set by Ceron et al. (2021) and Taïga et al. (2021); we claim its
> application to this method family, not the standard itself.

---

## 5. Net positioning statement (for the abstract / intro)

A1 is a **replication and extension** of the Osband et al. randomized-value ablations. Its
stated contributions are (C-i) separating the *estimator* from the *use rule* in a
pre-registered structured partial factorial, and (C-ii) using DeepSea's exact Q\* as a
continuous estimator-quality diagnostic. It **adopts** — and cites as inherited standard — the
matched-budget fair-baseline methodology (Ceron, Taïga) and the MinAtar low-budget venue
(Young; Clements). It positions the newer neighbors as complementary: EVE ablates within one
epistemic-value method (A1 crosses factors across methods); Priors Matter interrogates the
Bayesian assumptions A1's `prior` axis measures behaviourally; the July-2026 distributional
audit confirms QR-DQN's exploratory status. One `prior × K` neighbor preprint remains to be
located before the final freeze.

---

# §A — Reproducible-search appendix

*This appendix records how the prior-work map was assembled, to the standard a methodological
reviewer can re-run. It combines the original broad Bayesian-RL meta-analysis, the claim-level
re-mine that produced the audit, and the Session-2 neighbor verification.*

## A.1 Search dates and databases

- **Original corpus build & broad meta-analysis:** Session 0 (project onboarding task).
  Databases: **arXiv** (`export.arxiv.org` API) and **OpenAlex** (`api.openalex.org`).
- **Claim-level re-mine (the audit):** produced `novelty_deep_dive.md` /
  `novelty_threat_matrix.csv`. Databases: arXiv + OpenAlex, plus a forward/backward
  **citation walk** around ten anchor papers.
- **Newer-neighbor verification:** **2026-07-23** (this session). Database: arXiv API
  (title/abstract queries); OpenAlex attempted for venue/review status.

## A.2 Anchor papers (citation-walk seeds)

Osband 2016 (1602.04621), Osband 2018 (1806.03335), Osband 2019 (1703.07608), bsuite /
Osband 2020 (1908.03568), NoisyNets / Fortunato 2018 (1706.10295), QR-DQN / Dabney 2018,
MinAtar / Young 2019 (1903.03176), RND, Janz 2019 (1810.06530), and the Rainbow-revisiting
line (Ceron 2011.14826, Taïga 2109.11052). Forward walks over these collectively touch several
hundred citing works; multi-anchor intersections surfaced surveys and the disentanglement
papers rather than a hidden twin of A1.

## A.3 Retrieval counts

| Stage | Retrieved | After dedup / screening | Notes |
|---|---:|---:|---|
| Original corpus | **1,009 papers** | corpus re-mined at claim level | broad Bayesian-RL meta-analysis base |
| Targeted claim-level re-mine | **~350 new records** | abstracts read for the closest tier | arXiv + OpenAlex targeted queries |
| Exploration subset (shared) | **75 papers** | `corpus_exploration_subset.csv` | hand-tag `hs` (heuristic-signal) labels |
| Session-2 neighbor verification | 4 targets | 3 located, 1 not located | see A.4 |

## A.4 Exact queries — Session-2 neighbor verification (arXiv API, 2026-07-23)

| Target | Query string | Result |
|---|---|---|
| EVE | `all:"epistemic value estimation" AND all:exploration` | **2303.04012** (2023-03-07) |
| Priors Matter | `all:"priors matter" AND all:reinforcement` | **2508.21488** (2025-08-29) |
| Auditing distributional risk | `all:"distributional reinforcement learning" AND all:risk AND all:auditing` | **2607.11607** (2026-07-13) |
| BDQN-scaling (attempt 1) | `abs:bootstrapped AND abs:"prior functions" AND abs:diversity AND abs:DeepSea` | 0 hits |
| BDQN-scaling (attempt 2) | `abs:"Bootstrapped DQN" AND abs:"ensemble size" AND abs:DeepSea` | 0 hits |
| BDQN-scaling (attempt 3) | `all:"randomized prior" AND all:"ensemble size" AND all:diversity` | 0 hits |
| BDQN-scaling (attempt 4) | `abs:"Bootstrapped DQN" AND abs:"posterior collapse"` | 0 hits |
| BDQN-scaling (attempt 5) | `ti:"deep exploration" AND abs:"ensemble size"` | 0 hits |

## A.5 Load-bearing null queries (the audit)

- Title/abstract search for a paper jointly about **bootstrapped *and* NoisyNet** exploration
  → **0 matches**.
- Search for an exploration-method comparison (bootstrap / noisy / ensemble) **on MinAtar**
  → **0 matches**.
- The **BDQN-scaling `prior × K`** preprint (A.4) → **not located under this protocol**.

Null queries support C-i and C-v (no bootstrapped-vs-noisy comparison; no MinAtar exploration
cross-method study) and flag one open verification (A.4 / §3.4).

## A.6 Screening rules

Included if the record (a) proposes or evaluates a randomized/ensemble/epistemic value
estimator **or** a use-rule/perturbation mechanism for exploration, **and** (b) reports on a
deep-exploration chain (DeepSea/bsuite) **or** MinAtar/ALE **or** a matched-baseline
re-evaluation. Excluded: pure policy-gradient exploration bonuses without a value-estimator
axis, model-based planning-only work, and LLM/self-play "diversity" papers surfaced by lexical
collision (e.g. the cs.CL hits returned by the `diversity+collapse` query in A.4 — screened
out on full-text scope).

## A.7 Review-status verification

EVE (2303.04012) is a published conference paper (AAAI-2023 class). **Priors Matter**
(2508.21488, Aug 2025) and **Auditing** (2607.11607, Jul 2026) are treated as **arXiv
preprints**: OpenAlex title-search returned mismatched records for both (a famous survey with
wrong year and citation count), so peer-review/venue status is **not reliably verified** and
the arXiv listing is the authoritative record used here. Given their dates, non-peer-reviewed
preprint status is the conservative assumption.

## A.8 Limitations

Null results cannot prove absence; they bound the search protocol, not the literature. arXiv
API relevance ranking and OpenAlex fuzzy title-matching both introduce recall gaps — the
BDQN-scaling preprint (§3.4) is the concrete instance and is carried as an open item to the
external-review window and the Session-3 pilot freeze. Venue/review status for the two 2025–26
preprints is unverified (A.7). This appendix should be re-run closer to the final freeze so the
neighbor tier reflects the state of the literature at registration.

---

*Provenance: absorbs `novelty_deep_dive.md`, `novelty_claims.md`, `novelty_threat_matrix.csv`
(claim-level audit, Session 0/1) and the Session-2 neighbor verification run (2026-07-23).
Sources of truth for the design being positioned: `a1-requirements-and-alternatives-v6.3.md`
and `A1-claude-science-execution-plan-v4.3.md` (§7 Session-2 brief).*
