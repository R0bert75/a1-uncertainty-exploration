# When Do Randomized Value Estimates Buy Exploration?

**Separating the Estimator from the Use Rule under Low Interaction Budgets**

A pre-registered, reproducible empirical study of uncertainty-aware exploration in
value-based deep RL under low interaction budgets (100k–1M steps).

> **Status:** repository skeleton (Session 0 bootstrap). No scientific results yet.
> Method implementations, protocol values, and figures are added in later sessions,
> and **only after the pre-registration is finally frozen** (see `protocol/`).

---

## Research question

Uncertainty-aware exploration methods bundle an uncertainty **estimator** with a rule
for **using** it. This study separates the two:

- **Part A — controlled mechanism study (DeepSea).** A pre-registered structured
  partial factorial (use-rule × prior × ensemble-size; `use_rule × prior` core at
  K=10 plus a prior × K axis for the episodic rule — K × use_rule and three-way
  interactions are not estimable) with exact ground-truth Q\*, run as a
  many-seed replication and extension of the Bootstrapped-DQN mechanism ablations of
  Osband et al. (2016).
- **Part B — external performance evaluation (MinAtar).** A descriptive,
  equal-search-budget comparison of four whole-algorithm methods on standard
  small-scale benchmarks. No mechanism attribution; no general-ranking claim.

The primary estimand, the fixed-sequence secondary hierarchy, and the smallest effect
size of interest are frozen in the pre-registration before any scientific run.

## Methods (canonical four — MinAtar)

1. ε-greedy Double DQN (tuned ε schedule) — baseline
2. NoisyNet-DQN
3. Bootstrapped DQN — `(episodic_head, prior=off, K=K_shared)`
4. RP-BDQN — `(episodic_head, prior=on, K=K_shared, tuned prior scale)`

QR-DQN is a **pre-specified exploratory follow-up** (post-Gate-B), excluded from all
confirmatory aggregates.

## Environments

- **MinAtar** (5 games; tuning = Breakout + Asterix, evaluation = the remaining 3).
  Package version and stochasticity settings are pinned in configs and named in the paper.
- **DeepSea** — implemented as a single tested file validated against the published
  specification on small N (the `bsuite` dependency is unmaintained and incompatible
  with the pinned stack; this fallback is dependency hygiene, not a new benchmark).

## Reproducibility

- Every result reproducible from `(config, seed)`; configs and seeds of every reported
  run are committed.
- All metrics logged to CSV — nothing lives only in a dashboard.
- `make figures` rebuilds every figure from `logs/*.csv` alone.
- Pinned environment (`requirements.txt` + lockfile) attached to every run.

```bash
make env       # create/refresh the pinned environment
make test      # run the smoke tests
make figures   # rebuild every figure from logs/ only
```

## Repository layout

```
protocol/        pre-registration + advisor one-pager (committed BEFORE any result)
related_work/    positioning / nearest-neighbour table
configs/         one file per (method, env, budget); seeds explicit
src/             ddqn.py · noisynet.py · bdqn.py (+ randomized priors = RP-BDQN)
                 deep_sea.py (Part A env + exact Q*) · minatar_env.py (Part B adapter)
                 networks.py (MLP + conv trunks, K heads, noisy variants)
                 replay_buffer.py · trainer.py (episode lane + step lane)
                 utils/conventions.py (RNG streams, frozen CSV schema)
                 diagnostics/ (frozen battery members; §3.3 #6 implemented)
analysis/        make_figures.py (rebuilds every figure from logs/ alone).
                 hierarchy/ (§1.1 primary + fixed-sequence) and rq2l/ (concordance +
                 permutation) are reserved and EMPTY — that analysis is written after the
                 protocol freeze, on committed CSVs.
audits/          c13_audit.py (configuration identity) · config_schema_check.py (§4 CI)
                 c13/ audit outputs · code_completeness_session3.md (pre-freeze audit)
logs/            CSV per run — the source of truth for every figure
figures/         outputs of `make figures`
tests/           unit + convention + determinism tests (271 at the current commit)
VALIDATION.md    C0–C13 correctness checklist, ticked per session
```

## Limitations (scope of every claim)

Claims are scoped to **value-based methods, discrete actions, low interaction budgets,
and the specific environment families studied here**. MinAtar results are reported
per-game; the paper makes no claim about a general ranking of uncertainty-aware methods.

## License

MIT — see [LICENSE](LICENSE).

## Citation

Working title: *"When Do Randomized Value Estimates Buy Exploration? Separating the
Estimator from the Use Rule under Low Interaction Budgets."* R. Meliksetyan, 2026.
Framed as a systematic **replication and extension** of Osband et al. (2016).
