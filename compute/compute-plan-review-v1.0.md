# Review — A1 Compute Cost & Free-Tier Execution Plan v1.0

**Reviewed against the current source of truth:** requirements spec **v6.1**
(`a1-requirements-and-alternatives-v6.1.md`) and execution plan **v4.1**
(`A1-claude-science-execution-plan-v4.1.md`).
**Reviewer:** Claude Science · July 2026.

---

## Verdict

The plan is **architecturally sound and worth keeping** — the provider-independent
CLI + run ledger + checkpoint discipline + scope-reduction ladder is exactly the right
skeleton for a zero-cash, multi-provider study, and it internally checks out (the run
arithmetic and the Modal credit math are correct).

But it was written against the **superseded v4 / v1.2 documents** (it says so in its own
header). Between v4 and v6.1 the study was restructured, and three of those changes break
assumptions the compute plan is built on. **It needs a v1.1 revision before it drives any
spending or account setup.** The gaps below are ordered by severity.

---

## A. CRITICAL — the workload counts predate v6.1 and undercount by ~3× (GPU) and omit the confirmatory DeepSea budget

The plan's headline target is **≈390 MinAtar runs** (§3.2), from: M1 pilot 192 + held-out
120 + QR-DQN 78. Those counts are the **v4** workload. The **v6.1** run budget (spec §3.4,
"Run budget updated") is materially larger:

| Workload | Compute plan v1.0 (from v4) | Spec v6.1 | Gap |
|---|---:|---:|---|
| MinAtar tuning | 192 pilot | **≈ 240 + 800 formula-gated** | plan omits the 800-run final tier |
| MinAtar held-out | 120 @1M | ≈ 120 @1M | matches (Variant B) |
| DeepSea dev | ~not budgeted for GPU (CPU) | ≈ 150–250 | not costed |
| **DeepSea confirmatory** | **not budgeted** | **≈ 1,100** | **entirely missing from any forecast** |
| QR-DQN | 78 (submission gate) | ≈ 300 (exploratory, post-Gate-B) | ~4× undercount |

Two consequences:

1. **The GPU forecast is ~3× too small** if the final-tier trigger fires. The plan
   *deliberately* excludes the 20-config / 800-run final tier ("does not assume the
   optional final tuning tier") — a legitimate scoping choice — but it must state this
   against the v6.1 numbers and note that **if cap X's trigger fires, the free-tier plan
   cannot absorb +800 runs** and the descope ladder is invoked immediately.
2. **The ≈1,100 confirmatory DeepSea runs are the scientific core (Part A) and are never
   costed.** The plan treats DeepSea as "small, CPU, cheap" and moves on. On our actual
   box (8 cores), 1,100 runs of a K≤20 ensemble DQN at N=30–60 is **not** free wall-clock:
   at even 10 min/run that is ~183 CPU-hours ≈ 23 h wall-clock at 8× parallelism — and it
   is on the confirmatory critical path, not overflow. **v1.1 must add a DeepSea CPU
   benchmark and a CPU-hour forecast**, symmetric to the MinAtar GPU benchmark.

## B. CRITICAL — heterogeneous hardware is a fairness/reproducibility confound vs gates C0, C1, C13

This is the most important scientific gap. The plan spreads runs across Modal T4, Kaggle
(T4/P100), Colab (variable), Studio Lab, and local CPU, and assigns providers by
*convenience* (whoever has quota). But different GPUs / CUDA / cuDNN versions produce
different numerical trajectories, and the whole study rests on:

- **C1 (determinism):** same `(config, seed)` reproduces the same curve "or within
  documented nondeterminism" — cross-hardware runs are *not* bit-reproducible.
- **C0 / C13 (identity of contrast cells):** a reported contrast pair must differ **only**
  in the varied factor. If method A's seeds ran on Kaggle-P100 and method B's on Modal-T4,
  **hardware is an uncontrolled second factor** aliased onto the comparison.

The plan gestures at this ("output hash", "validate one config on an independent image")
but has **no hardware-assignment rule**. v1.1 must add one. The minimal fix:

- **Every cell of a given contrast/comparison runs on one hardware class.** Never split a
  contrast across providers. Assign at the *comparison* level, not the run level.
- **The held-out MinAtar evaluation runs on a single declared reference hardware** (the
  plan already implies "the reference hardware" in §13 — operationalize it).
- **Log `provider` + `hardware` on every row** (the manifest already has these — good) and
  **add a C1-style cross-hardware agreement report** as an explicit deliverable (the spec's
  §3.6 "local reproducibility rerun, agreement reported separately" is the natural home).
- Where a family must span providers for capacity, **balance hardware across methods
  within the family and record it**, so hardware is at worst a randomized nuisance, never
  confounded with method.

Without this rule the free-tier distribution silently violates the equal-treatment
guarantees the paper's credibility depends on.

## C. HIGH — two different descope ladders now exist; the protocol's is frozen and governs

The compute plan has its own scope-reduction ladder (§10). Spec v6.1 **freezes** a descope
ladder (freeze item 17, triggered by compute cap **Y**) with a specific order:
cap held-out at 500k → K-axis to {10,20} → final-tier configs down → held-out seeds 10→5 →
drop the MinAtar E-cell spot-check → drop the QR-DQN follow-up; **never cut** the factorial,
dual axes, EWRL figure, protocol, or DeepSea confirmatory seeds.

The compute plan's ladder orders things differently (e.g. it lists "report 100k/500k, defer
1M" and "reduce held-out games — do not do") and predates the freeze. **These must be
reconciled: the pre-registered ladder is authoritative once frozen.** v1.1's ladder should
*be* the frozen ladder (or explicitly defer to it), not a parallel one — otherwise a
capacity cut becomes an unregistered protocol deviation.

## D. HIGH — the plan ignores the two-stage freeze gate; scientific sweeps can't start yet

Execution plan v4.1 adds the **two-stage freeze**: `prereg-draft` tag → 1-week external
pass → final tag + OSF, and **Session 3 (pipeline spine) onward requires the final
freeze**. The compute plan's Stage B pilot sweep is Session 7 — it **cannot run until the
final freeze**. The plan never mentions this gate.

What *is* pre-freeze-eligible (and should be pulled forward): the **§7 Stage A benchmark**
is exactly the v4.1 Session-0 "throwaway infra benchmark that informs compute cap X." So
the benchmark + free-account setup are the parts that can proceed now; the sweeps wait for
the freeze. v1.1 should mark this boundary explicitly.

## E. MEDIUM — the "existing 8-vCPU / 24-GB VPS" may not exist; the sandbox is ephemeral

The plan repeatedly assigns all DeepSea + all analysis + canonical log storage to an
"existing local machine / VPS (8 vCPU / 24 GB)." Our actual environment is **8 CPU / 15 GiB
RAM / ~1–3 GiB free disk, and ephemeral** (the workspace is swept; durable state lives in
the artifact store). If there is a real persistent VPS the user controls, the plan is fine;
if "local" means this sandbox, then:

- **~1,100 confirmatory DeepSea runs + canonical CSV storage + "two backups" have no
  durable home here** — they must land in the artifact store (or the user's own machine),
  not `logs/` on the sandbox disk.
- The 24 GB figure is wrong for planning; use the real spec.

**This needs a one-line answer from the user** (see questions below) because it decides
where the Part-A critical path actually runs.

## F. MEDIUM — the benchmark isn't mapped to *both* caps (X and Y)

v6.1 has two compute caps: **X** (final-tier trigger, freeze item 4) and **Y** (descope
trigger, freeze item 17). The plan's §11 worksheet + go/no-go thresholds are a good input
to setting them, but it only informally connects to "the compute budget." v1.1 should state:
the Stage-A benchmark → forecast → **sets X and Y numerically** at the Session-1 freeze,
and the §11.3 thresholds map to the descope ladder's rungs.

## G. MEDIUM — only Modal is automatable from Claude Science; the rest are manual toil

The plan calls Modal a "native fit… jobs launched directly from Claude Science" — correct,
via the Modal BYOC backend. But **no compute backend is connected yet** (`list_compute`
is empty), and **Kaggle / Colab / Studio Lab are not Claude Science backends** — they are
hand-run notebooks. That is a sound portability design, but it means the "reduce calendar
time" benefit is bounded by **the user's manual hours** babysitting notebooks, not by
automation. Two implications worth stating:

- To get the automated Modal lane the plan leans on, **connect Modal** (Customize →
  Compute → BYOC Modal). Until then even Modal is manual.
- Realistically, **Modal is the burst/parallel lane I can drive; Kaggle/Colab/Studio Lab
  are the user's manual bulk lane.** The plan should own this division explicitly so the
  calendar estimate is honest.

## Smaller items / nits

- **Working title is stale.** The plan uses "Which Uncertainty Estimates Buy Exploration?";
  v6.1's title is "When Do Randomized Value Estimates Buy Exploration? Separating the
  Estimator from the Use Rule…". Cosmetic, but update it.
- **QR-DQN framing drift.** The plan treats QR-DQN as a "submission-gate" method; v6.1
  demotes it to a **pre-specified exploratory** follow-up after Gate B, excluded from
  confirmatory aggregates (every row `role: exploratory`). Its runs are bonus, not gate.
- **GitHub Actions minutes on a *private* repo.** "Free for public repos" is right, but if
  the repo stays private during the external-pass window, CI draws on the limited
  private-repo minute allowance (2,000/mo). Decide repo visibility (see below) or budget the
  minutes.
- **Ledger schema vs our logger.** The plan's run-manifest/ledger fields
  (`evidence_tier: pilot|primary|secondary`) don't match the frozen logging contract we
  already shipped in Session 0 (`role: confirmatory|development|exploratory` +
  `config_sha256`). **Align the ledger to the committed schema**, don't introduce a second
  vocabulary.
- **Academic-credit net could be wider:** Modal grants (mentioned) + AWS Cloud Credit for
  Research, Google for Education/research credits, NVIDIA Academic — all long shots, none on
  the critical path, but cheap to apply for in parallel.

---

## What to add for v1.1 (checklist)

1. Re-cost the workload from **v6.1** numbers (tuning 240 + 800-gated; DeepSea dev 150–250 +
   **confirmatory 1,100**; held-out 120; QR-DQN exploratory ~300).
2. Add a **DeepSea CPU benchmark + CPU-hour forecast** alongside the MinAtar GPU benchmark.
3. Add a **hardware-assignment rule** (one hardware class per contrast; single reference
   hardware for held-out; balance+log otherwise) and a **cross-hardware agreement report**.
4. **Replace the plan's ladder with the frozen descope ladder** (cap Y) or explicitly defer.
5. Mark the **freeze gate**: benchmark + accounts are pre-freeze; sweeps wait for the final
   freeze (Session 3+).
6. Map the benchmark → **caps X and Y** numerically at the Session-1 freeze.
7. State the **Modal-automated vs manual-notebook** division and connect Modal to enable it.
8. **Align the ledger schema** to the committed `role`/`config_sha256` contract.
9. Fix the **title** and **QR-DQN role** wording.
10. Resolve the **VPS-vs-sandbox** storage/compute question.

None of these touch the plan's architecture — they update its numbers and bolt on the
fairness rule the multi-provider design implies but doesn't yet state.
