# A1 — advisor one-pager (STUB)

> **STATUS: STUB.** Drafted in full in Session 1 and sent as touch #2 alongside the
> `prereg-draft`. Initiates the Gate A track.

## The question

Uncertainty-aware exploration methods bundle an uncertainty *estimator* with a rule for
*using* it. We separate the two, as a pre-registered, exact-ground-truth replication and
extension of the Bootstrapped-DQN mechanism ablations of Osband et al. (2016).

## Primary estimand (one sentence)

_To be filled:_ the effect of temporal coherence of the sampled value function
(C-COHERENCE, per-episode vs. per-step resampling) on discovery probability within the
episode budget on DeepSea, at `prior=off, K=10`, aggregated over the five confirmatory sizes.

## Contrast table

| Contrast | Varies | Fixed | Question |
|---|---|---|---|
| C-COHERENCE | episode vs. step resampling | prior, K, config | does temporal coherence matter? |
| C-USE | episodic-head vs. ensemble-mean ε rule | estimator | does the use rule matter? |
| C-PRIOR | randomized prior on/off (tuned scale) | use rule, K | does the prior intervention help? |
| C-K | K ∈ {5, 10, 20} | use rule, prior | does ensemble size trend? |

## Why publishable

Systematic **replication and extension** at modern standards: pre-registered factorial,
named contrasts, fixed estimand hierarchy, exact Q\*, 20 seeds/cell, shared-configuration
purity audits, a decision-relevant uncertainty-quality battery, and a separate
equal-search-budget MinAtar performance evaluation. A recognised, valued contribution class at RLC.

## The ask

> "Would you be willing to advise/co-author this as an RLC-style empirical paper if I
> implement the repo and run the experiments? I'd value your feedback first on the
> protocol and the related-work framing."
