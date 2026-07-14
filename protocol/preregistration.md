# Pre-registration — A1

**Working title:** *When Do Randomized Value Estimates Buy Exploration? Separating the
Estimator from the Use Rule under Low Interaction Budgets.*

> **STATUS: STUB — NOT YET FROZEN.** This file is committed first (Session 0) so that its
> commit history visibly precedes any result commit. It is *filled with numeric values*
> and frozen in Session 1 via the **two-stage freeze**:
>
> 1. Fill all freeze-list items → commit under a **`prereg-draft`** tag (internal timestamp).
> 2. External methodological pass on the valued draft (1-week time-box; silence → promote with a note).
> 3. Incorporate fixes → commit the **final pre-registration**: new tag + **OSF mirror**. *This is the freeze.*
>
> **Scientific runs (pipeline spine, Session 3 onward) begin only after the final freeze.**
> Infrastructure work (environment, throwaway benchmark, skeleton, CI) may precede it.
> Post-freeze/pre-confirmatory changes are versioned amendments (new tag + OSF, reasons,
> changed-item list, motivating development outcomes disclosed).

---

## Two-part identity

- **Part A — controlled mechanism study (DeepSea):** pre-registered factorial replication
  and extension of Osband et al. (2016).
- **Part B — external performance evaluation (MinAtar):** descriptive whole-algorithm
  comparison under equal search budgets. No mechanism attribution; no transfer claim.

## The freeze list (20 items — VALUES TO BE FILLED in Session 1)

1. [ ] Config/seed counts per tier and stage (DeepSea 20/cell confirmatory; MinAtar held-out 10).
2. [ ] Search distributions per hyperparameter; backbone-tuning budget; two factor-specific mini-budgets.
3. [ ] Selection statistic + tie-breaking.
4. [ ] Final-tier trigger (method-specific formula); cap **X** GPU-hours.
5. [ ] DeepSea development sizes; confirmatory-size selection rule + its three size sets; episode budget per N.
6. [ ] Reporting windows.
7. [ ] Probe-set construction + weighting.
8. [ ] Operational failure policy (§3.6) incl. divergence criterion + infra/algorithmic classification.
9. [ ] Bootstrap units, stratification, CI construction; §1.1 aggregation + fixed-sequence; RQ2-L statistic in full.
10. [ ] Frozen-policy extraction rules.
11. [ ] t₀ landmark rule; post-t₀ outcome definitions.
12. [ ] The 10 cells, four contrasts, shared-configuration rule, factor-specific parameter list.
13. [ ] Primary estimand hierarchy (§1.1) — contrasts, settings, outcomes, aggregation, test order.
14. [ ] Battery formulas, sample counts, aggregation (§3.3).
15. [ ] Canonical MinAtar four with defining parameters.
16. [ ] QR-DQN exploratory status; its hypothesis + test (neutrality rule).
17. [ ] Descope ladder + objective compute-cap trigger **Y** GPU-hours; before-Gate-A execution point.
18. [ ] Two-stage freeze + amendment policy incl. external-pass time-box.
19. [ ] Parameter classes (backbone-tuned / literature-fixed ensemble-shared / factor-specific); K_shared joint rule; SESOI **Δ = 0.10**.
20. [ ] Deterministic MinAtar-cloning conditional + undefined-value policy.

## §1.1 Primary estimand hierarchy (to be stated verbatim in Session 1)

- **Primary:** C-COHERENCE at `prior=off, K=10`; outcome = discovery probability within
  the episode budget; aggregation = unweighted mean of per-size effect over the five
  confirmatory sizes; independent stratified bootstrap (no pairing, no CRN).
- **SESOI:** Δ = 0.10 absolute discovery-probability difference; equivalence-style
  interpretation of the 95% CI.
- **Fixed-sequence gatekeeping:** (1) C-COHERENCE → (2) C-USE → (3) C-PRIOR → (4) C-K trend;
  each confirmatory only if all preceding rejected at two-sided α = 0.05.
- **Everything else is descriptive with CIs.**

## Contrasts

- **C-COHERENCE** — per-episode vs. per-step head resampling (temporal coherence of the sampled value function).
- **C-USE** — episodic-head rule vs. capacity-matched ensemble-mean ε-greedy rule.
- **C-PRIOR** — randomized-prior intervention as practiced (incl. tuned scale).
- **C-K** — ensemble-size trend over K ∈ {5, 10, 20}.
