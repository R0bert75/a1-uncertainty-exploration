# A1 Research Project — Compute Cost and Free-Tier Execution Plan

**Project:** *When Do Randomized Value Estimates Buy Exploration? Separating the Estimator
from the Use Rule under Low Interaction Budgets*
**Owner:** Robert Meliksetyan
**Version:** 1.1
**Date:** July 2026
**Companion documents (current source of truth):** `a1-requirements-and-alternatives-v6.1.md`
and `A1-claude-science-execution-plan-v4.1.md`

> **What changed from v1.0 (read this first).** v1.0 was written against the superseded
> v4 / v1.2 documents. Its provider architecture, ledger, checkpoint discipline, and
> zero-cost controls are **retained unchanged** — they were correct and remain the backbone
> of this plan. v1.1 fixes the parts that the v4 → v6.1 restructure invalidated:
>
> 1. **Workload re-costed from v6.1** — the run counts in v1.0 §3 undercounted; the ≈1,100
>    confirmatory DeepSea runs (Part A, the scientific core) were never costed at all.
>    (§3, new.)
> 2. **DeepSea CPU forecast added** — a measured per-step benchmark shows Part A is a
>    multi-day critical-path cost, not a free afterthought. (§3.3, new.)
> 3. **Hardware-assignment rule added** — the multi-provider design creates a
>    fairness/reproducibility confound against gates C0/C1/C13 unless contrasts are
>    hardware-controlled. This is the most important scientific fix. (§5.5, new.)
> 4. **Descope ladder replaced by the frozen v6.1 ladder** — v1.0 had a parallel ladder;
>    the pre-registered one governs. (§10, rewritten.)
> 5. **Two-stage freeze gate honored** — only the Stage-A benchmark and account setup are
>    pre-freeze; all sweeps wait for the final freeze (Session 3+). (§7, gated.)
> 6. **Caps X and Y wired to the benchmark** — the benchmark output *sets* both frozen caps
>    numerically at the Session-1 freeze. (§11, rewritten.)
> 7. **Automation reality stated** — only Modal is a Claude Science backend; Kaggle / Colab /
>    Studio Lab are manual notebooks. And no backend is connected yet. (§2, §4.1, §6.)
> 8. **Ledger schema aligned** to the committed Session-0 logging contract
>    (`role` + `config_sha256`), replacing v1.0's `evidence_tier`. (§5.2–5.3.)

---

## 1. Purpose

This document defines how to execute the A1 research project with **zero out-of-pocket
cloud-compute cost**, using a coordinated mix of local CPU compute, Modal Starter monthly
credits, Kaggle free GPU quotas, Google Colab free runtimes, Amazon SageMaker Studio Lab,
optional Lightning AI free allocations, and GitHub Actions for CPU-only CI.

The goal is not to pretend free resources are guaranteed. The goal is a provider-independent
workflow that (1) can move jobs between providers without changing the experiment,
(2) prioritizes the most reliable free resource per task, (3) **preserves the preregistered
fairness and reproducibility rules — including hardware control across contrast cells**,
(4) prevents accidental charges, (5) records true compute used, and (6) degrades gracefully
under the *frozen* descope ladder.

The zero-cost plan is realistic for the **core study and submission gate** only if the
project runs DeepSea and statistics on CPU, uses the two-tier tuning protocol
conservatively, does not assume the formula-gated final tuning tier fires, reads
100k/500k/1M checkpoints from single 1M runs (Variant B), treats the QR-DQN exploratory
follow-up as optional, and **measures real runtime (both GPU and CPU) before freezing caps
X and Y.**

---

## 2. Executive recommendation

### Recommended provider hierarchy

| Priority | Provider | Main project role | Automatable from Claude Science? |
|---:|---|---|---|
| 1 | **Local / user VPS (CPU)** | Development, **all DeepSea**, tests, diagnostics, aggregation, figures | Local kernel |
| 2 | **Kaggle free GPU** | Bulk MinAtar GPU hours | **No — manual notebook** |
| 3 | **Modal Starter** | Parallel tuning bursts, deadline jobs, failed-run recovery | **Yes — BYOC backend** |
| 4 | **Google Colab Free** | Interactive debugging, overflow | No — manual notebook |
| 5 | **SageMaker Studio Lab** | Predictable daily backup GPU sessions | No — manual notebook |
| 6 | **Lightning AI** | Optional overflow after dashboard check | No — manual |
| 7 | **GitHub Actions** | CI, smoke tests, lint — never research sweeps | Yes (CI) |

### Core strategy

- **DeepSea is CPU-first — but it is not free wall-clock.** Its ≈1,100 confirmatory runs
  are the largest single CPU cost in the project and sit on the confirmatory critical path
  (§3.3). Budget them explicitly.
- **Kaggle supplies GPU volume** for MinAtar batches; **Modal supplies orchestration** (the
  one lane that can be driven programmatically). **Colab / Studio Lab supply resilience.**
- **No experiment is provider-specific** — one CLI, one lockfile, one YAML, one Git commit.
- **A contrast never spans hardware classes** (§5.5) — this is a fairness requirement, not a
  convenience preference.
- **No paid-billing upgrade** is required; Google Cloud GPU VMs are excluded.

### Automation reality (new in v1.1)

Only **Modal** is a Claude Science compute backend — jobs launched programmatically. Kaggle,
Colab, and Studio Lab are **hand-run notebooks**: the "reduce calendar time" benefit from
those three is bounded by the owner's manual hours, not by automation.
**As of this writing no compute backend is connected** (`list_compute` is empty). To use the
automated Modal lane this plan relies on, connect Modal BYOC (Customize → Compute). Until
then, even Modal is manual.

---

## 3. Workload assumptions — re-costed from v6.1

The v6.1 specification defines the **canonical four** methods for Part B (MinAtar external
performance evaluation): ε-greedy Double DQN (tuned ε); NoisyNet-DQN; Bootstrapped DQN
(= episodic, prior=off, K_shared); RP-BDQN (= episodic, prior=on, K_shared, tuned prior
scale). QR-DQN is a **pre-specified exploratory** follow-up run only after Gate B, excluded
from confirmatory aggregates (every row `role: exploratory`).

Part A (the mechanism study) is the **10-cell DeepSea factorial switchboard**
(use_rule × prior × K; K ∈ {5,10,20}), with **20 seeds/cell** in the single confirmatory
sweep.

### 3.1 Run budget (v6.1 §3.4, verbatim source)

> *MinAtar tuning ≈ 240 + 800 (formula-gated); MinAtar held-out ≈ 120 runs @1M
> (Variant B — the 500k eval and the extension are the same runs); DeepSea dev ≈ 150–250 +
> confirmatory ≈ 1,100; QR-DQN exploratory share ≈ 300 (post-Gate-B).*

| Workload | v6.1 count | Hardware | Costed in v1.0? |
|---|---:|---|---|
| MinAtar tuning (pilot tier) | ≈ 240 | GPU | partially (192) |
| MinAtar tuning (final tier, **formula-gated by cap X**) | ≈ 800 | GPU | **no — omitted** |
| MinAtar held-out @1M (Variant B) | ≈ 120 | GPU (reference HW) | yes |
| DeepSea development | ≈ 150–250 | CPU | no |
| **DeepSea confirmatory (Part A core)** | **≈ 1,100** | **CPU** | **no — never costed** |
| QR-DQN exploratory (post-Gate-B) | ≈ 300 | GPU/CPU | ~78 (undercount) |

**Two consequences vs v1.0:**

1. **The GPU forecast can be ~3× larger** if the final tier fires. The final tier is
   **method-specific and formula-gated** (v6.1 §freeze-item-4): it runs iff
   Σ over the four methods of (final-tier run count × that method's *median pilot-tier
   per-run wall-clock*) ≤ **cap X GPU-hours**, with X fixed at freeze. The free-tier plan
   **deliberately does not assume the final tier fires** — a legitimate scoping choice — but
   must state it against the v6.1 number: **if the trigger fires, the free tier cannot
   absorb +800 runs**, and the frozen descope ladder (§10) is the response, not silent
   omission.
2. **The ≈1,100 confirmatory DeepSea runs are costed for the first time in §3.3.**

### 3.2 MinAtar GPU forecast (unchanged method — measure, then forecast)

The GPU forecast is unchanged from v1.0 in *method*: measure a real 500k-step run per
backend, then apply the forecast formulas (§11). Only the counts change (use §3.1). The 1M
held-out runs use Variant B — checkpoints at 100k/500k/1M read from the *same* runs.

### 3.3 DeepSea CPU forecast (new — the fix for v1.0's biggest gap)

DeepSea is CPU, but 1,100 confirmatory runs of a K≤20 ensemble DQN is a real,
multi-day cost. A measured per-gradient-step probe on the reference box
(8 shared cores, torch CPU, K=20 ensemble, 64-batch) gives **≈ 25 ms/grad-step**. That
yields the following order-of-magnitude forecast (illustrative — the real per-run grad-step
count comes from the actual agent at freeze; this is a *frozen-method, measured-input*
estimate, not a commitment):

| Grad-steps / run | min / run | 1,100 runs (CPU-h, serial) | Wall-clock @ 4× | Wall-clock @ 6× |
|---:|---:|---:|---:|---:|
| 20,000 | 8.3 | 153 | ~38 h | ~25 h |
| 50,000 | 20.8 | 382 | ~95 h | ~64 h |
| 100,000 | 41.7 | 764 | ~191 h | ~127 h |

**Reading:** even the optimistic corner is **~1 day of dedicated wall-clock**; the middle is
**2.5–4 days**; the heavy corner is **over a week**. This is on the confirmatory critical
path and cannot be treated as free. Implications for the plan:

- **Add a DeepSea CPU benchmark to Stage A** (§7), symmetric to the MinAtar GPU benchmark:
  measure the *real* agent's grad-steps/run and s/step, then forecast 1,100 runs.
- **DeepSea confirmatory runs feed cap Y** (the descope trigger, §11) alongside GPU hours —
  the projected-total-compute number that trips the ladder must include CPU wall-clock, not
  just GPU-hours.
- **If the user's "VPS" is this ephemeral sandbox, Part A has no durable home** — see §12.

### 3.4 Runtime scenarios

Actual runtime must be measured on both GPU (MinAtar) and CPU (DeepSea). The v1.0 GPU
scenario table (5–60 min/500k-run) still illustrates why the benchmark is the cost gate; the
CPU table above is its DeepSea counterpart.

---

## 4. Provider comparison

Provider terms below are as recorded in v1.0 (verified July 13, 2026); **recheck each
dashboard before a frozen sweep**, since free allocations change. Only the automation and
academic-credit notes are updated for v1.1.

### 4.1 Modal Starter — the one automatable lane

- Starter plan: $0 subscription; **$30/month free compute credits**; up to 10 concurrent
  GPUs / 100 containers; per-second billing; T4 ≈ $0.59/hr before CPU/mem.
- Nominal $30 ≈ 50 T4-hours on the GPU line alone; **budget 35–45 *effective* T4-hours/month**
  after CPU/mem/startup/failure overhead.
- **Best A1 uses:** parallel pilot tuning, deadline sweeps, failed-seed recovery,
  reproducibility reruns, provider benchmarks, and **jobs launched directly from Claude
  Science via the Modal BYOC backend.**
- **Zero-cost controls (unchanged, all retained):** no payment method unless required;
  budget/usage alerts; cap concurrency at 2 during tests then 5–10 after gates pass; hard
  function timeouts; launch only manifest run IDs; stop mapping if the first two jobs fail;
  reserve ≥20% credit for recovery.
- **New:** apply for **Modal academic grants** (advertised up to $10,000; acceptance not
  assumed, never a critical-path dependency).

### 4.2–4.8 Kaggle / Colab / Studio Lab / Lightning / Google Cloud / GitHub Actions / others

**Retained from v1.0 unchanged in substance** (allocations, best uses, workflows, zero-cost
controls). The v1.1-relevant summary:

| Provider | Free allocation (recheck) | Role | Automatable here |
|---|---|---|---|
| Kaggle | ~30 GPU-h/week, resets weekly, HW not guaranteed | Bulk MinAtar GPU | No — manual notebook |
| Colab Free | Unpublished/dynamic; ≤~12 h runtime | Debug + overflow | No — manual notebook |
| Studio Lab | 4 GPU-h/session, 4 GPU-h/24h; persistent storage | Daily backup batch | No — manual notebook |
| Lightning AI | Account-specific; **count as zero until dashboard-verified** | Bonus overflow | No |
| Google Cloud trial | $300/90d but **no GPU on non-billable trial** | **Excluded** from zero-risk plan | n/a |
| GitHub Actions | Free CPU runners on **public** repos (repo is public) | CI / smoke / lint only | Yes |
| HF Spaces / AWS free tier / Oracle / Codespaces / paid marketplaces | see v1.0 §4.8 | not for batch RL | — |

**GitHub Actions note (updated):** the repo is **public**
(`github.com/R0bert75/a1-uncertainty-exploration`), so GitHub-hosted CPU runners are free
and unmetered — no private-repo minute budget applies. Do not use Actions for RL sweeps or
as a checkpoint store.

**Wider academic-credit net (new, all long shots, none on critical path):** Modal grants,
AWS Cloud Credit for Research, Google research credits, NVIDIA Academic. Apply in parallel;
assume nothing.

---

## 5. Provider-independent experiment contract

The project must never contain separate Kaggle/Modal/Colab versions of an algorithm. §5.1
and §5.4 are **retained from v1.0 unchanged**; §5.2–5.3 are aligned to the committed logging
schema; §5.5 is new and load-bearing.

### 5.1 Single execution interface (unchanged)

Every experiment runs through one command:

```bash
python -m src.run \
  --config configs/minatar/bootstrapped_dqn/breakout.yaml \
  --seed 7 \
  --run-id m1-minatar-bdqn-breakout-c03-s07 \
  --output-dir logs/raw
```

Notebook and cloud scripts only call this interface.

### 5.2 Required run metadata — aligned to the Session-0 logging contract

v1.0 used a `task_role` / `evidence_tier: pilot|primary|secondary` vocabulary. **That is
replaced** by the frozen contract already shipped in `src/utils/conventions.py` (Appendix C):
every row carries `role: confirmatory|development|exploratory`, `part: A|B`, and
`config_sha256`, and every run's **resolved config is serialized and committed** as the
C13 audit input. The run manifest is:

```yaml
run_id:
provider:              # modal | kaggle | colab | studiolab | local
hardware:              # e.g. modal-T4, kaggle-P100, local-cpu-8core  (NEW: load-bearing, see §5.5)
git_commit:
lockfile_hash:
config_sha256:         # matches the committed resolved config (C13)
seed:
part:                  # A | B
role:                  # confirmatory | development | exploratory
method:                # ddqn_egreedy | noisynet | bdqn | rp_bdqn | qrdqn
env:                   # deep_sea | minatar/<game>
budget_steps:
start_time_utc:
end_time_utc:
wall_clock_seconds:
status:                # complete | failed | interrupted
checkpoint_paths:
log_path:
```

### 5.3 Central run ledger (aligned)

Keep `runs/ledger.csv` in the public repo (no large binaries). Minimum columns:

```text
run_id, provider, hardware, method, env, part, role, config_sha256, seed,
budget_steps, status, artifact_location, validation_status
```

Rules (retained from v1.0, with the schema alignment): globally unique run IDs; no two
providers run the same ID unless it is a preregistered reproducibility rerun; the ledger
decides remaining work; cloud output is not accepted until imported and validated; failed
runs stay recorded. **The ledger vocabulary is the conventions-module vocabulary — no second
vocabulary is introduced.**

### 5.4 Artifact flow (unchanged from v1.0)

Repo/tagged commit → provider runner reads batch manifest → common CLI → CSV + checkpoint +
manifest → provider temp storage → download/sync to canonical logs → validation/import →
ledger marked complete → final raw logs archived in OSF/Zenodo. Do not use Actions artifacts
or Git LFS as the primary checkpoint store.

### 5.5 Hardware-assignment rule (NEW — fairness/reproducibility, gates C0/C1/C13)

**This is the most important addition in v1.1.** Spreading runs across Modal-T4, Kaggle
(T4/P100), Colab (variable), Studio Lab, and local CPU by *convenience* introduces
**hardware as an uncontrolled second factor**. Different GPU/CUDA/cuDNN stacks produce
different numerical trajectories, which collides with:

- **C1 (determinism):** same `(config, seed)` reproduces the same curve, or within
  *documented* nondeterminism — cross-hardware runs are not bit-reproducible.
- **C0 / C13 (contrast-cell identity):** a reported contrast pair must differ **only** in the
  varied factor. If method A's seeds ran on Kaggle-P100 and method B's on Modal-T4, hardware
  is aliased onto the comparison.

**The rule:**

1. **One hardware class per contrast.** Assign hardware at the *comparison* level, never the
   run level. All cells of a given contrast (or method comparison) run on one hardware class.
   Never split a contrast across providers.
2. **The MinAtar held-out evaluation runs on a single declared reference hardware.** Pick one
   backend as "reference HW" at freeze; all 120 held-out @1M runs use it. (v6.1 §13 already
   assumes "the reference hardware" for held-out accounting — this operationalizes it.)
3. **Log `provider` + `hardware` on every row** (already in the manifest) and produce a
   **cross-hardware agreement report** as an explicit deliverable — the natural home is the
   v6.1 §3.6 "local reproducibility rerun, agreement reported separately."
4. **Where a family must span providers for capacity, balance hardware across methods within
   the family and record it**, so hardware is at worst a randomized nuisance, never
   confounded with method.

Without this rule the free-tier distribution silently violates the equal-treatment guarantees
the paper's credibility depends on.

---

## 6. Provider-specific operational workflows

**Retained from v1.0 unchanged** (Modal adapter `cloud/modal_runner.py`; thin
`notebooks/kaggle_runner.ipynb`, `colab_runner.ipynb`; `studio_lab_runner.sh`; local/VPS
`tmux`/`systemd-run`), with three v1.1 amendments:

- **Modal is the only lane launched from Claude Science**; the notebook lanes are manual and
  their calendar benefit is bounded by the owner's hours. Connect Modal BYOC to enable the
  Modal lane at all.
- **Every runner prints and logs its `hardware` string** and refuses to run a contrast whose
  other cells were logged on a different hardware class (§5.5 enforcement).
- **Every runner writes the resolved config** (`config_sha256`) and `role`/`part` fields per
  §5.2 — not the retired `evidence_tier`.

Recommended Modal allocation (unchanged): 60% MinAtar pilot tuning, 20% QR-DQN exploratory,
20% failed-run + deadline reserve.

---

## 7. Zero-cost execution schedule — gated by the two-stage freeze (NEW gating)

**Critical addition:** execution plan v4.1 imposes a **two-stage freeze** — `prereg-draft`
tag → time-boxed external methodological pass → amendments → **final tag + OSF mirror** →
**only then scientific runs**. **Session 3 (pipeline spine) onward requires the final
freeze.** Therefore the schedule splits into pre-freeze and post-freeze work:

### Pre-freeze-eligible (can run now, during the external-pass window)

- **Stage A — development & benchmarking** (retained from v1.0), which *is* the v4.1
  Session-0 "throwaway infra benchmark that informs compute cap X":
  - **Local/CPU:** environment setup (done — env `a1-rl` pinned), Double DQN one-seed run,
    DeepSea implementation + tests, CI fixtures, the common CLI, ledger/import scripts.
  - **GPU benchmark:** the same 500k-step Double-DQN/Breakout run on Modal T4 / Kaggle /
    Colab / Studio Lab; record GPU type, wall-clock, setup time, steps/s, output hash,
    cost/quota.
  - **CPU benchmark (new):** the real DeepSea agent's per-run grad-steps and s/step, then the
    1,100-run forecast (§3.3).
  - **Gate:** compute the full GPU + CPU forecast → set caps **X and Y** (§11) → this is the
    input to the Session-1 freeze.
- Account creation, quota recording, smoke tests everywhere.

### Post-freeze only (blocked until the final freeze / Session 3+)

- **Stage B — Milestone-1 pilot tuning** (Modal first half + Kaggle remainder; DeepSea pilot
  local; Colab/Studio Lab for failures). Gate: 240 MinAtar pilot runs + DeepSea dev
  completed, or the frozen reduced budget reached **equally for every method**.
- **Stage C — held-out M1 evaluation** on the single reference hardware (§5.5): 1M runs,
  checkpoints at 100k/500k/1M (Variant B), 5+5 staged seeds treated as one final set.
  Primary Kaggle, deadline reserve Modal, overflow Studio Lab/Colab.
- **Stage D — QR-DQN exploratory** (post-Gate-B): Modal or Kaggle for the runs, DeepSea on
  CPU; every row `role: exploratory`; skip if capacity is tight (last rung of the ladder).
- **Stage E — analysis & manuscript**: entirely local/CPU + GitHub Actions; no GPU for
  statistics or plotting.

---

## 8. Free-capacity allocation plan

**Retained from v1.0 unchanged** (Modal 35–45 effective T4-h/mo; Kaggle ~30 GPU-h/wk;
Colab dynamic; Studio Lab ~4 GPU-h/day; Lightning zero-until-verified; local CPU;
Actions CPU-only). Plan around **50–60% usable realization** of nominal quotas. **v1.1 adds
the CPU line:** the local box must also supply the DeepSea 150–760 CPU-hours (§3.3) —
account for it as a first-class resource, not spare cycles.

---

## 9. Cost-saving rules in priority order

**Retained from v1.0 unchanged** (measure first; DeepSea on CPU; one 1M final run +
checkpoints; progressive 5→10 seeds; pilot tier can stand; cut configs before paper seeds;
held-out before all-task; QR-DQN before ceiling; no idle notebook time; fail fast;
checkpoint only where useful; no GPU analysis; no duplicate runs; strategic monthly resets;
apply for academic credits). **Two v1.1 amendments:** "DeepSea on CPU" does **not** mean
"DeepSea is free" — budget its wall-clock; and rule ordering for capacity cuts now defers to
the **frozen descope ladder (§10)**, not this convenience list.

---

## 10. Descope ladder — the FROZEN v6.1 ladder governs (rewritten)

v1.0 carried its own scope-reduction ladder. **It is superseded.** Spec v6.1 **freezes** a
descope ladder and its objective trigger (freeze item 17); once frozen it is authoritative,
and applying any *other* ladder is an unregistered protocol deviation. The frozen ladder,
verbatim from v6.1 §3.4:

> **Trigger:** applied only by the frozen compute-cap rule, before Gate A and before any
> confirmatory outcome; after the pilot-tier wall-clock measurement, if projected total
> compute > cap **Y**, rungs apply in order until ≤ Y. Rung 1 applies only if invoked
> **before held-out runs launch** (once launched, runs complete to 1M). Ladder application
> after confirmatory execution is prohibited.
>
> **Order:**
> 1. Cap held-out runs at 500k (drop the 1M leg → **RQ3 dies first**).
> 2. K-axis to {10, 20}.
> 3. Final-tier configs down (identical across methods).
> 4. Held-out seeds 10 → 5 (with sensitivity note).
> 5. Drop the MinAtar E-cell spot-check.
> 6. Drop the QR-DQN exploratory follow-up.
>
> **Never cut:** the factorial, dual axes, EWRL figure, the protocol, DeepSea confirmatory
> seeds.

**Never (retained from v1.0, consistent with the freeze):** give one method more tuning
because it looks promising; remove divergent seeds; move a held-out game into tuning after
seeing results; buy power by mixing tuning games into the primary aggregate.

The free-tier plan's job is to *avoid* invoking this ladder by measuring first and setting
realistic caps — but if capacity forces a cut, **this is the only cut order permitted.**

---

## 11. Compute budget worksheet — sets caps X and Y (rewritten)

Fill after the Stage-A benchmark; the outputs **are** the frozen caps.

### 11.1 Measured runtime table

| Provider/GPU | 500k min | 1M min | Setup min | Cost/quota per run |
|---|---:|---:|---:|---:|
| Modal T4 |  |  |  |  |
| Kaggle GPU |  |  |  |  |
| Colab GPU |  |  |  |  |
| Studio Lab GPU |  |  |  |  |
| Local CPU (MinAtar) |  |  |  |  |
| **Local CPU (DeepSea, per run)** |  | — |  | CPU-h |

### 11.2 Forecast formulas (updated to v6.1 counts)

```text
MinAtar pilot GPU-hours      = 240 × measured_500k_min / 60
MinAtar final-tier GPU-hours = Σ_method (final_run_count × method_median_pilot_min) / 60   # gated by cap X
MinAtar held-out GPU-hours   = 120 × measured_1M_min / 60          # reference hardware only
DeepSea dev CPU-hours        = (150..250) × deepsea_run_min / 60
DeepSea confirmatory CPU-h   = 1100 × deepsea_run_min / 60          # Part A core — measured in §3.3
QR-DQN exploratory hours     = ~300 × measured_run_min / 60         # optional, post-Gate-B

planned_compute = raw × 1.15    # 15% failure/setup reserve
```

### 11.3 Setting the two caps

- **Cap X (final-tier trigger, method-specific):** fix X at freeze; the final tier runs iff
  Σ_method(final_run_count × method_median_pilot_wall_clock) ≤ X GPU-hours. Set X to the
  largest final-tier GPU budget the free lanes can absorb *without* touching the confirmatory
  critical path. If the trigger would exceed X, the final tier simply does not run — no
  descope needed.
- **Cap Y (descope trigger):** fix Y at freeze as the total projected compute
  (**GPU-hours + DeepSea CPU-hours**, §3.3) the free plan can realistically deliver at
  50–60% quota realization. If the projected full plan > Y, apply the §10 ladder in order
  until ≤ Y — before Gate A, before any confirmatory outcome.

### 11.4 Go/no-go thresholds (retained from v1.0, GPU side)

≤50 GPU-h → Modal-nominal but keep Kaggle reserve · 50–150 → Kaggle+Modal (normal) ·
150–300 → add Studio Lab, spread weeks, cut optional scope · 300–500 → reduce pilot configs,
inspect utilization, reconsider 1M axis · >500 → not a credible free-tier project as-is;
redesign before running. **The DeepSea CPU forecast (§3.3) is a parallel go/no-go on the CPU
side:** >~400 CPU-h of confirmatory DeepSea on a shared 8-core box is a multi-day serial
bottleneck — plan a dedicated CPU box or accept the calendar cost.

---

## 12. Data storage and archival plan

**Retained from v1.0** (source/configs on GitHub; canonical CSV logs local with two backups;
provider temp output on Modal Volume / Kaggle output / Drive / Studio Lab storage; ledger on
GitHub+local; selected checkpoints local; OSF/Zenodo immutable release at submission).

**v1.1 caveat — the "existing VPS" assumption must be resolved.** v1.0 assigns all DeepSea +
analysis + canonical log storage + "two backups" to an "existing 8-vCPU/24-GB VPS." The
Claude Science box is **8 CPU / 15 GiB RAM / ~1–3 GiB free disk and ephemeral** (workspace
is swept; durable state lives in the artifact store). If a real persistent user VPS exists,
v1.0's storage plan stands. If "local" means this sandbox, then the ≈1,100 confirmatory
DeepSea runs + canonical CSVs + backups **have no durable home here** and must land in the
artifact store or the user's own machine. **This needs a one-line answer from the owner**
(see §14).

---

## 13. Accounting required by the paper

**Retained from v1.0 unchanged**, with the working title updated: report total tuning
GPU-hours by provider/hardware; total CPU-hours for DeepSea + analysis; **final held-out
GPU-hours on the single reference hardware** (§5.5); wall-clock calendar time; estimated
list-price value; actual cash paid (target **$0**); reproduction time for selected finals and
for the full pipeline. The approved wording (one modest workstation reproduces every final;
free-tier parallelism only reduced calendar time of the preregistered search) stands.

---

## 14. Immediate actions

Pre-freeze-eligible now:

1. **Connect Modal BYOC** (Customize → Compute) — without it there is no automated lane.
2. Create free accounts for Kaggle, Colab, Studio Lab; record each quota + reset date in
   `compute/providers.yaml`.
3. Confirm the run-manifest carries `provider` + `hardware` + `config_sha256` + `role` +
   `part` (align to `src/utils/conventions.py`); create `runs/ledger.csv` with the §5.3
   columns.
4. Implement the common CLI and thin Modal/Kaggle/Colab/Studio-Lab launchers; each prints
   its `hardware` string and enforces the §5.5 one-hardware-per-contrast rule.
5. Run one 100-step smoke everywhere.
6. **Run the Stage-A benchmarks — both:** the 500k Double-DQN/Breakout GPU benchmark on each
   available backend, **and** the real DeepSea agent CPU benchmark (§3.3).
7. Complete the §11 worksheet → **set caps X and Y** → feed them into the Session-1 freeze.
8. Apply for Modal academic credits (and the wider net) without making approval a dependency.
9. **Answer the storage question (§12):** is there a persistent user VPS, or is Part A's
   ≈1,100-run CPU workload to be hosted elsewhere?

Post-freeze only (blocked until the final prereg tag): Stages B–E (all sweeps).

---

## 15. Decision summary

The core A1 study can plausibly complete at **$0 cash cloud spend**, but v1.1 makes three
things explicit that v1.0 did not:

1. **The scientific core is a CPU cost, not a free afterthought** — ≈1,100 confirmatory
   DeepSea runs = 150–760 CPU-hours = 1–8 days wall-clock on this box (§3.3).
2. **Multi-provider distribution requires a hardware-control rule** or it breaks the
   equal-treatment guarantees the paper rests on (§5.5).
3. **Sweeps are gated by the two-stage freeze** — only the Stage-A benchmark and account
   setup can proceed now; everything else waits for the final freeze (§7).

Robustness still comes from provider portability, the central ledger, checkpointing, and —
now — the **frozen** descope ladder (§10). The first compute milestone is **not** a sweep: it
is one reproducible 500k-step GPU benchmark per backend **plus** the DeepSea CPU benchmark,
followed by a measured forecast that sets caps X and Y at the Session-1 freeze.
