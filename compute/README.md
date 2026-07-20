# compute/

Compute-cost and free-tier execution planning for the A1 study.

- `A1-compute-cost-and-free-tier-execution-plan-v1.1.md` — **current** plan.
  Re-costed against spec v6.1 / execution plan v4.1: adds the DeepSea CPU forecast
  (~1,100 confirmatory runs), the one-hardware-per-contrast fairness rule (C0/C1/C13),
  the frozen v6.1 descope ladder, the two-stage-freeze gating of all sweeps, and wires
  caps X and Y to the Stage-A benchmark.
- `compute-plan-review-v1.0.md` — gap analysis of v1.0 against the current source of truth.
- `A1-compute-cost-and-free-tier-execution-plan-v1.0.md` — original (written against the
  superseded v4 / v1.2 docs); kept for provenance.

Sweeps remain gated by the final pre-registration freeze; only the Stage-A infrastructure
benchmark and free-account setup are pre-freeze-eligible.
