# Positioning & nearest-neighbour table (STUB)

> **STATUS: STUB.** Populated in Session 1 (protocol) and refined in Session 11 (writing).
> Purpose: state precisely what is replicated, what is extended, and what is new in
> *evidence* (not in first-ness). No novelty claims stronger than the evidence supports.

## Framing

This work is a systematic **replication and extension** of the mechanism ablations in
Osband et al. (2016, *Deep Exploration via Bootstrapped DQN*), brought to modern
empirical standards (pre-registration, exact ground truth, many seeds, named contrasts,
configuration-identity audits, uncertainty-quality battery).

## Nearest-neighbour table (to be filled)

| Work | Estimator | Use rule | Ground truth | Pre-reg | Seeds/cell | Separates estimator vs. use rule? | Budget regime |
|---|---|---|---|---|---|---|---|
| Osband et al. 2016 (Bootstrapped DQN) | ensemble | episodic head | partial (chain) | no | few | partially | — |
| Osband et al. 2018 (randomized priors) | ensemble + prior | episodic head | — | no | — | no | — |
| Fortunato et al. 2018 (NoisyNet) | param noise | native | — | no | — | no | — |
| Dabney et al. 2018 (QR-DQN) | distributional | greedy | — | no | — | no | — |
| **This work** | ensemble (±prior), noisy, distributional (exploratory) | episodic / per-step / ensemble-mean | **exact Q\* (DeepSea)** | **yes (two-stage)** | **20 (DeepSea conf.)** | **yes — the point** | **low (100k–1M)** |

## What we explicitly do NOT claim

- Not the first to study any of these methods.
- No general ranking of uncertainty-aware exploration methods.
- No "transfer" claim between DeepSea and MinAtar — they answer different questions
  (mechanism vs. external performance).
