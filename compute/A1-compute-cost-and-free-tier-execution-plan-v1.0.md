# A1 Research Project — Compute Cost and Free-Tier Execution Plan

**Project:** *Which Uncertainty Estimates Buy Exploration? A Reproducible Study of Uncertainty-Aware Exploration under Low Interaction Budgets*  
**Owner:** Robert Meliksetyan  
**Version:** 1.0  
**Date:** July 13, 2026  
**Companion documents:** `a1-requirements-and-alternatives-v4.md` and `A1-claude-science-execution-plan-v1.2.md`

---

## 1. Purpose

This document defines how to execute the A1 research project with **zero out-of-pocket cloud-compute cost**, using a coordinated mix of:

- local/VPS CPU compute;
- Modal Starter monthly credits;
- Kaggle free GPU quotas;
- Google Colab free runtimes;
- Amazon SageMaker Studio Lab;
- optional Lightning AI free allocations;
- GitHub Actions for CPU-only validation and continuous integration.

The goal is not to pretend that free resources are guaranteed. Their availability and quotas can change. The goal is to design a provider-independent workflow that:

1. can move jobs between providers without changing the experiment;
2. prioritizes the most reliable free resource for each task;
3. preserves the study's preregistered fairness and reproducibility rules;
4. prevents accidental charges;
5. records the true compute used, including free cloud compute;
6. can be reduced gracefully if quotas are insufficient.

The zero-cost plan is realistic for the **core study and submission gate** if the project:

- runs DeepSea and statistical analysis on CPU;
- uses the two-tier tuning protocol conservatively;
- does not assume the optional 20-configuration final tuning tier;
- runs selected final MinAtar experiments once to 1M steps and reads 100k/500k/1M checkpoints from the same runs;
- treats all-task secondary reruns and ceiling methods as optional;
- measures actual runtime before freezing the final compute budget.

---

## 2. Executive recommendation

### Recommended provider hierarchy

| Priority | Provider | Main project role |
|---:|---|---|
| 1 | **Existing local machine / VPS** | Development, DeepSea, tests, diagnostics, aggregation, figures |
| 2 | **Kaggle free GPU** | Main source of bulk MinAtar GPU hours |
| 3 | **Modal Starter** | Parallel tuning bursts, deadline-critical jobs, failed-run recovery |
| 4 | **Google Colab Free** | Interactive debugging and overflow batches |
| 5 | **Amazon SageMaker Studio Lab** | Predictable daily backup GPU sessions |
| 6 | **Lightning AI** | Optional overflow only after checking the account's current free allocation |
| 7 | **GitHub Actions** | CI, smoke tests, formatting, tiny CPU checks—not research sweeps |

### Core strategy

- **DeepSea is CPU-first.** Run its tuning, exact \(Q^*\), oracle checks, and uncertainty diagnostics locally or on the existing 8-vCPU/24-GB VPS.
- **Kaggle supplies volume.** Use it for batches of independent MinAtar seeds/configurations.
- **Modal supplies orchestration.** Reserve the $30 monthly credit for parallel tuning jobs and time-sensitive reruns.
- **Colab and Studio Lab supply resilience.** They keep the project moving when Kaggle quota or Modal credit is temporarily exhausted.
- **No experiment is provider-specific.** Every run is produced by the same CLI, lockfile, YAML configuration, and Git commit.
- **No paid-billing upgrade is required.** Google Cloud GPU VMs are specifically excluded from the zero-risk free plan.

---

## 3. Workload assumptions

The current A1 specification defines four Milestone-1 methods:

1. ε-greedy Double DQN;
2. NoisyNet-DQN;
3. Bootstrapped DQN;
4. Bootstrapped DQN with randomized priors.

QR-DQN is added after the advisor gate.

### 3.1 Core MinAtar workload

The following counts are planning estimates, not a commitment. The final count is frozen only after a real 500k-step benchmark.

#### Milestone-1 pilot tuning

\[
4\ methods \times 8\ configurations \times 3\ seeds \times 2\ tuning\ games = 192\ runs
\]

The protocol permits 8–12 pilot configurations. The free-only plan starts with **8**, not 12.

#### Held-out final evaluation

\[
4\ methods \times 10\ seeds \times 3\ held\text{-}out\ games = 120\ runs
\]

Run these directly to **1M steps** once the configuration is frozen, saving checkpoints at 100k and 500k. This avoids three separate run sets.

#### Optional all-task secondary reruns

\[
4\ methods \times 10\ seeds \times 2\ tuning\ games = 80\ runs
\]

These are optional. They are the first final-evaluation runs to cut if free capacity is tight.

#### QR-DQN addition

Pilot tuning:

\[
1\ method \times 8\ configurations \times 3\ seeds \times 2\ tuning\ games = 48\ runs
\]

Held-out evaluation:

\[
1\ method \times 10\ seeds \times 3\ held\text{-}out\ games = 30\ runs
\]

Optional all-task secondary reruns add 20 more runs.

### 3.2 Free-tier core versus maximum scope

| Scope | Approximate MinAtar GPU runs | Status |
|---|---:|---|
| M1 pilot tuning + four-method held-out evaluation | 312 | **Core** |
| Add QR-DQN tuning + held-out evaluation | +78 | **Submission gate** |
| Add all-task secondary reruns | +100 | Optional |
| Add 12-config rather than 8-config pilot | +96 for M1, +24 for QR-DQN | Optional |
| Add 20-config × 5-seed final tuning tier | Very large increase | **Not assumed in free plan** |
| MiniGrid, ensemble-Thompson, RND/bonus anchor | Additional | Ceiling only |

The practical free-only target is therefore approximately **390 MinAtar runs**, excluding optional all-task reruns and ceiling items.

### 3.3 Runtime scenarios

Actual runtime must be measured. The table below illustrates why that benchmark is the main cost gate.

| Typical 500k-step runtime | 192-run pilot | Interpretation |
|---:|---:|---|
| 5 min/run | 16 GPU-hours | Easy on free resources |
| 10 min/run | 32 GPU-hours | Fits one Modal month nominally or roughly one Kaggle week |
| 20 min/run | 64 GPU-hours | Use Kaggle + Modal |
| 30 min/run | 96 GPU-hours | Spread across providers/weeks |
| 60 min/run | 192 GPU-hours | Reduce configs and investigate GPU utilization |

A 1M-step selected evaluation will usually cost roughly twice a 500k-step run, though setup and evaluation overhead may alter the ratio.

---

## 4. Provider comparison

Provider terms are verified as of July 13, 2026. Free allocations can change, so recheck the dashboard and official documentation before a preregistered sweep.

### 4.1 Modal Starter

**Current free allocation**

- Starter plan: $0 monthly subscription.
- $30/month in free compute credits.
- Up to 10 concurrent GPUs and 100 containers.
- Serverless per-second billing.
- Current listed T4 rate is approximately $0.000164/second, or about $0.59/hour, before CPU and memory charges.

A nominal $30 credit therefore corresponds to about 50.8 T4 GPU-hours if only the GPU line item mattered. In practice, budget **35–45 effective T4 hours/month** after CPU, memory, startup, and failed-job overhead.

**Best A1 uses**

- parallel pilot hyperparameter jobs;
- deadline-critical sweeps;
- failed-seed recovery;
- exact reproducibility reruns;
- provider benchmark tests;
- jobs launched directly from Claude Science.

**Strengths**

- best automation and concurrency;
- pay only while jobs run;
- easy mapping over configuration manifests;
- logs and programmatic status;
- native fit with the current Claude Science execution plan.

**Weaknesses**

- credit is finite;
- a programming bug can consume many parallel GPU hours quickly;
- GPU plus CPU/memory charges make the nominal-hour estimate optimistic;
- free credit is monthly and should not be exhausted on development.

**Zero-cost control**

- do not attach a payment method unless required;
- set Modal budget/usage alerts;
- cap GPU concurrency at 2 during initial tests, then 5–10 only after correctness gates pass;
- set hard function timeouts;
- launch only run IDs present in the frozen manifest;
- stop mapping immediately if the first two jobs fail;
- reserve at least 20% of the monthly credit for recovery jobs.

**Official sources**

- [Modal pricing](https://modal.com/pricing)
- [Modal billing guide](https://modal.com/docs/guide/billing)
- [Modal academic grants](https://modal.com/pricing#academia)

---

### 4.2 Kaggle Notebooks

**Current free allocation**

Kaggle's official efficient-GPU documentation states that the weekly GPU quota is generally **30 hours, sometimes higher depending on demand and resources**. The quota resets weekly. Hardware availability is not guaranteed.

**Best A1 uses**

- bulk MinAtar seed batches;
- held-out final evaluation;
- 1M-step selected runs;
- QR-DQN batches;
- secondary all-task reruns if capacity remains.

**Strengths**

- largest recurring source of free GPU time in the plan;
- no direct cloud bill;
- notebook outputs can be downloaded as artifacts;
- suitable for independent, embarrassingly parallel RL runs.

**Weaknesses**

- manual or semi-manual orchestration;
- GPU availability and queues vary;
- notebook session and quota constraints;
- storage/output handling must be disciplined;
- not a native Claude Science backend.

**Recommended workflow**

1. The research repo remains the source of truth.
2. A thin `notebooks/kaggle_runner.ipynb`:
   - clones a fixed Git commit;
   - installs from the lockfile;
   - downloads or embeds a run manifest;
   - executes only the provider-independent CLI;
   - writes CSV logs, checkpoints, and a run summary.
3. One Kaggle session runs a bounded batch that finishes comfortably within the session.
4. Download the output archive immediately.
5. Import outputs locally through a validation script.
6. Mark run IDs complete in the central ledger.

**Zero-cost controls**

- do not store logic only in the notebook;
- save every completed run immediately;
- checkpoint long 1M-step runs;
- leave 15–20% quota headroom for retries;
- never rely on Kaggle as the only copy of raw logs;
- use weekly quota early rather than waiting until a deadline.

**Official sources**

- [Kaggle notebooks documentation](https://www.kaggle.com/docs/notebooks)
- [Kaggle efficient GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage)

---

### 4.3 Google Colab Free

**Current free allocation**

Google does not publish a fixed free GPU-hour quota. Free GPU access, GPU type, idle timeout, and overall limits are dynamic. Free notebooks can run for **at most approximately 12 hours**, depending on availability and usage.

**Best A1 uses**

- interactive CUDA debugging;
- one-off smoke tests;
- small seed batches;
- emergency overflow;
- validating a result on a second cloud image.

**Strengths**

- fastest ad hoc setup;
- easy Google Drive persistence;
- familiar notebook environment;
- often provides a T4-class GPU, though not guaranteed.

**Weaknesses**

- unpublished and variable quota;
- runtimes can terminate unexpectedly;
- GPU type can change;
- poor fit for deadline-critical batch scheduling;
- no guaranteed background execution on free tier.

**Recommended workflow**

1. Use `notebooks/colab_runner.ipynb` as a launcher only.
2. Mount a dedicated Google Drive folder.
3. Clone a fixed Git commit.
4. Install the lockfile.
5. Read a small batch manifest.
6. Save CSVs after every run and checkpoints periodically.
7. Sync outputs to Drive before the next run begins.
8. Import to the central ledger locally.

**Zero-cost controls**

- never schedule the only copy of a long final run on Colab;
- use batch sizes small enough to survive a runtime reset;
- checkpoint at least every 50k–100k environment steps;
- avoid interactive idle periods while a GPU is allocated;
- do not design the protocol around a particular GPU model.

**Official sources**

- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Colab plan comparison](https://colab.research.google.com/signup)

---

### 4.4 Amazon SageMaker Studio Lab

**Current free allocation**

Studio Lab is a free JupyterLab environment and does not require a normal AWS account or credit card. Current official documentation states:

- GPU sessions: up to 4 hours at a time and 4 hours in a 24-hour period;
- CPU sessions: up to 4 hours at a time and 8 hours in a 24-hour period;
- persistent project storage;
- no distributed training or managed pipelines.

**Best A1 uses**

- one daily four-hour MinAtar batch;
- backup when Kaggle or Colab is unavailable;
- independent reproducibility checks;
- persistent development environment;
- 1M-step runs that fit within four hours or support checkpointing.

**Strengths**

- no bill and no regular AWS account required;
- persistent environment;
- predictable daily session ceiling;
- terminal and Git integration.

**Weaknesses**

- only four GPU hours per day;
- limited automation;
- hardware may not match Kaggle/Modal;
- no large-scale distributed orchestration.

**Recommended workflow**

- clone the same fixed commit;
- run `studio_lab_runner.sh` or the common CLI;
- use 3–3.5-hour bounded batches;
- store outputs in persistent storage;
- download/sync logs after each daily session;
- use it as backup capacity, not primary orchestration.

**Official sources**

- [SageMaker Studio Lab guide](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab.html)
- [Studio Lab resource limits](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab-overview.html)

---

### 4.5 Lightning AI

**Current free allocation**

Lightning's public materials and account experiences currently indicate a free CPU Studio and, for many accounts, a monthly promotional GPU-credit allocation. The exact GPU allowance can be account-specific and has changed over time. It must be confirmed in the user's dashboard before being included in capacity planning.

**Best A1 uses**

- optional overflow GPU;
- persistent cloud development environment;
- small independent verification batches;
- CPU-side analysis if the local VPS is unavailable.

**Planning rule**

Count Lightning capacity as **zero** until the account dashboard confirms:

- current monthly credits;
- eligible GPU types;
- expiration/reset date;
- whether a payment method is required.

Any confirmed allocation is treated as bonus capacity, never as a critical path dependency.

**Official sources**

- [Lightning AI platform FAQ](https://lightning.ai/docs/platform/overview/faq)
- [Lightning AI](https://lightning.ai/)

---

### 4.6 Google Cloud Free Trial

**Current status**

New eligible users may receive $300 of credits for 90 days. However, while the account remains a non-billable Free Trial account, Google currently states that users **cannot add GPUs to Compute Engine VMs**. Upgrading unlocks GPU access and preserves remaining credit, but enables billing for usage beyond the credits.

**Decision for A1**

Google Cloud GPU VMs are **not part of the zero-risk free plan**.

They may become an emergency option only when all of the following are true:

- the user is eligible for the trial;
- the account is deliberately upgraded;
- hard budget alerts and quotas are configured;
- the user accepts billing risk;
- the remaining free credit is sufficient.

This document assumes none of those steps.

The always-free `e2-micro` CPU VM is too small for MinAtar training but could run a lightweight status page, manifest service, or tiny DeepSea tests. It is unnecessary because local/VPS and GitHub Actions already cover those functions.

**Official source**

- [Google Cloud free-program documentation](https://docs.cloud.google.com/free/docs/free-cloud-features)

---

### 4.7 GitHub Actions

**Current free allocation**

Standard GitHub-hosted runners are free for public repositories. They are CPU runners, not free GPU runners. Larger runners remain billable.

**Best A1 uses**

- import and dependency checks;
- 100–1,000-step smoke tests;
- unit tests for replay buffers, bootstrap masks, DeepSea transitions, \(Q^*\), and logging;
- `make figures` test on fixture data;
- linting and formatting;
- validating configuration and run manifests;
- checking that every expected log has a unique run ID.

**Do not use for**

- full RL tuning;
- long MinAtar experiments;
- storing large checkpoints or raw experiment archives.

**Official source**

- [GitHub Actions billing](https://docs.github.com/billing/managing-billing-for-github-actions/about-billing-for-github-actions)

---

### 4.8 Other providers

| Provider | Free option | A1 recommendation |
|---|---|---|
| Hugging Face Spaces | Free CPU; GPU generally requires payment or a community grant; ZeroGPU is optimized for interactive Spaces | Do not use for batch RL. Possible later demo only. |
| AWS SageMaker regular free tier | CPU notebook/training allowances; no durable free-GPU pool comparable to Studio Lab | Use Studio Lab instead. |
| Oracle Cloud Free Tier | CPU VMs, account/region availability varies | Optional DeepSea CPU only; unnecessary if VPS exists. |
| GitHub Codespaces | Limited free CPU development quota depending plan | Development only; no GPU value. |
| Paid GPU marketplaces (RunPod, Vast.ai, Lambda) | No dependable always-free allocation | Excluded from zero-cost plan; possible low-cost emergency fallback only. |

---

## 5. Provider-independent experiment contract

The project must never contain separate Kaggle, Modal, and Colab versions of an algorithm.

### 5.1 Single execution interface

Every experiment runs through one command, for example:

```bash
python -m src.run \
  --config configs/minatar/bootstrapped_dqn/breakout.yaml \
  --seed 7 \
  --run-id m1-minatar-bdqn-breakout-c03-s07 \
  --output-dir logs/raw
```

Notebook and cloud scripts only call this interface.

### 5.2 Required run metadata

Every run writes a manifest containing:

```yaml
run_id:
provider:
hardware:
git_commit:
lockfile_hash:
config_hash:
seed:
environment_family:
environment:
task_role: tuning | held_out
evidence_tier: pilot | primary | secondary
budget_steps:
start_time_utc:
end_time_utc:
wall_clock_seconds:
status: complete | failed | interrupted
checkpoint_paths:
log_path:
```

### 5.3 Central run ledger

Keep `runs/ledger.csv` locally and in the public repo without large binary artifacts.

Minimum columns:

```text
run_id, provider, method, environment, config_hash, seed,
budget, status, artifact_location, validation_status
```

Rules:

- a run ID is globally unique;
- no two providers intentionally run the same run ID unless it is a preregistered reproducibility rerun;
- the ledger decides what work remains;
- cloud notebook output is not accepted until imported and validated;
- failed/interrupted runs remain recorded.

### 5.4 Artifact flow

```text
GitHub repo + tagged commit
        ↓
Provider runner reads batch manifest
        ↓
Provider executes common CLI
        ↓
CSV + checkpoint + run manifest
        ↓
Provider-specific temporary storage
        ↓
Download/sync to local canonical logs
        ↓
Validation/import script
        ↓
Central ledger marked complete
        ↓
Final raw logs archived in OSF/Zenodo release
```

Do not use GitHub Actions artifacts or Git LFS as the primary store for large model checkpoints. Keep only paper-relevant checkpoints; most training checkpoints can be deleted after logs and final reproducibility checks are secured.

---

## 6. Provider-specific operational workflows

### 6.1 Modal workflow

Create a provider adapter such as `cloud/modal_runner.py`.

1. Read `runs/modal_batch_YYYYMMDD.yaml`.
2. Validate that all run IDs are unstarted.
3. Build a pinned Modal image from the project lockfile.
4. Upload or mount the fixed Git commit.
5. Map a function across configurations.
6. Limit early concurrency to 2.
7. After two successful jobs, raise concurrency to 5–10.
8. Write outputs to a Modal Volume.
9. Download completed artifacts to the canonical local log store.
10. Import through `scripts/import_runs.py`.
11. Check the Modal credit balance.
12. Stop the batch when the reserved monthly budget is reached.

Recommended Modal allocation:

- 60%: MinAtar pilot tuning;
- 20%: QR-DQN tuning;
- 20%: failed runs and deadline reserve.

### 6.2 Kaggle workflow

Maintain one thin notebook, `notebooks/kaggle_runner.ipynb`.

1. Enable GPU.
2. Clone the tagged repo commit.
3. Install dependencies from the lockfile.
4. Upload/select a batch-manifest dataset.
5. Print GPU model and environment versions.
6. Execute a bounded set of run IDs.
7. Save after every run.
8. Produce `batch_summary.csv`.
9. Commit notebook output.
10. Download the archive.
11. Import and validate locally.
12. Update the ledger before launching another notebook.

Batch sizing rule:

\[
batch\ duration \leq 0.75 \times expected\ session\ ceiling
\]

This leaves room for installation, validation, and output persistence.

### 6.3 Colab workflow

Maintain `notebooks/colab_runner.ipynb`.

1. Mount Google Drive.
2. Clone tagged commit.
3. Install dependencies.
4. Inspect GPU.
5. Load a small batch manifest from Drive.
6. Resume from checkpoints where needed.
7. Sync CSV/checkpoint after every completed run.
8. Stop when the runtime becomes unstable or quota warnings appear.
9. Import outputs locally.

Colab batches should be smaller than Kaggle batches because quota and runtime behavior are less predictable.

### 6.4 SageMaker Studio Lab workflow

1. Start the daily GPU session.
2. Pull the tagged commit.
3. Activate the persistent environment.
4. Select a batch estimated below 3.5 hours.
5. Run the common CLI.
6. Save outputs in persistent storage.
7. Download/sync before storage accumulates.
8. Shut down or allow the session to end.
9. Import locally.

### 6.5 Local/VPS workflow

Use the existing VPS/local CPU for:

- all DeepSea sweeps;
- tabular oracle/PSRL validation;
- \(Q^*\) calculation;
- uncertainty diagnostics;
- rliable/bootstrap analysis;
- `make figures`;
- manuscript data tables;
- provider output validation.

Use `tmux` or `systemd-run` for long CPU jobs. Record the VPS CPU model and wall-clock like any cloud provider.

---

## 7. Zero-cost execution schedule

The schedule below assumes the free quotas continue to be available. It is organized by project gates rather than calendar promises.

### Stage A — Development and benchmarking

**Local/VPS**

- environment setup;
- Double DQN one-seed run;
- DeepSea implementation and tests;
- all CI fixtures;
- provider-independent CLI;
- ledger/import scripts.

**Cloud benchmark**

Run the same 500k-step Double DQN/Breakout configuration on:

- Modal T4;
- Kaggle GPU;
- Colab GPU;
- Studio Lab GPU if access is available.

Record:

- GPU type;
- wall-clock;
- environment setup time;
- steps/second;
- output hash;
- cost or quota hours.

**Gate:** calculate the complete compute forecast before launching the pilot sweep.

### Stage B — Milestone-1 pilot tuning

**Modal**

- first half of the MinAtar pilot sweep;
- configurations where rapid parallel comparison is valuable.

**Kaggle**

- remaining pilot configurations/seeds;
- bulk execution using weekly quota.

**Local/VPS**

- full DeepSea pilot tuning;
- validation and diagnostics.

**Colab/Studio Lab**

- replace failed or queued Kaggle jobs;
- verify one configuration on an independent image.

**Gate:** 192 MinAtar pilot runs and separate DeepSea pilot completed, or the preregistered reduced budget reached equally for every method.

### Stage C — Held-out M1 evaluation

After configurations are frozen:

- run selected configurations directly to 1M steps where possible;
- save 100k, 500k, and 1M checkpoints;
- initially complete 5 seeds for the EWRL/M1 gate;
- later add 5 more seeds using exactly the same protocol;
- treat staged 5+5 completion as one final seed set, not as two analyses.

**Primary provider:** Kaggle.  
**Deadline reserve:** Modal.  
**Overflow:** Studio Lab and Colab.

Do not perform optional all-task secondary reruns until all held-out runs are complete.

### Stage D — QR-DQN

- use Modal for the 48-run pilot if monthly credit has reset;
- otherwise use Kaggle;
- run DeepSea QR-DQN on CPU;
- run the 30 held-out final MinAtar seeds on Kaggle/Modal;
- skip its all-task secondary reruns unless free capacity remains.

### Stage E — Analysis and manuscript

Entirely local/VPS plus GitHub Actions:

- bootstrap statistics;
- probability of improvement;
- RQ2 exploration-link analysis;
- figure regeneration;
- claim-to-log trace checks;
- manuscript tables.

No GPU should be allocated for manuscript work or plotting.

---

## 8. Free-capacity allocation plan

### Conservative monthly allocation

| Resource | Planning capacity | Reserved use |
|---|---:|---|
| Modal Starter | 35–45 effective T4 hours/month | Parallel tuning and deadline recovery |
| Kaggle | Up to ~30 GPU hours/week, availability dependent | Main bulk MinAtar compute |
| Colab Free | Unpublished/dynamic | Debug and overflow only |
| Studio Lab | Up to 4 GPU hours/day | Backup daily batch |
| Lightning AI | Count as zero until dashboard verified | Bonus only |
| Existing VPS | Existing fixed cost; no incremental cloud charge | DeepSea and analysis |
| GitHub Actions public repo | Standard CPU runners free | CI only |

A realistic recurring pool—without counting Colab or Lightning—is therefore:

- approximately 120 nominal Kaggle GPU hours per four weeks;
- approximately 35–45 effective Modal T4 hours per month;
- up to approximately 120 Studio Lab GPU hours per 30 days if used daily, though daily availability and practical use will be lower.

Theoretical quota totals should not be mistaken for guaranteed capacity. Plan around **50–60% usable realization**.

---

## 9. Cost-saving rules in priority order

1. **Measure first.** Never estimate the project from intuition once one real run is possible.
2. **DeepSea on CPU.** Do not spend GPU credits on tiny environments.
3. **One 1M final run, multiple checkpoints.** Never run separate 100k/500k/1M final experiments.
4. **Progressive seeds.** Run 5 final seeds first, then extend to 10 with unchanged configs and protocol.
5. **Pilot tier can stand.** The optional 20-config final tuning tier is not required if runtime is too high.
6. **Cut configurations before paper seeds.** Final uncertainty intervals are more valuable than exhaustive tuning.
7. **Held-out results before all-task secondary views.**
8. **QR-DQN before ceiling methods.**
9. **Provider batches contain no idle notebook time.**
10. **Fail fast.** Every provider batch starts with one smoke run.
11. **Checkpoint only where useful.** Keep final/diagnostic checkpoints; discard unnecessary intermediates after validation.
12. **No GPU analysis.** Statistics and plotting run on CPU.
13. **No duplicate runs.** The ledger prevents accidental duplication.
14. **Use monthly resets strategically.** Modal tuning can be split over two billing cycles without changing scientific validity.
15. **Apply for academic credits.** Modal advertises academic grants of up to $10,000; acceptance is not assumed.

---

## 10. Scope-reduction ladder if free compute is insufficient

Reductions must preserve equal treatment across methods and be documented before results influence the decision.

| Order | Reduction | Scientific cost |
|---:|---|---|
| 1 | Drop optional all-task secondary reruns | Very low |
| 2 | Use 8 rather than 12 pilot configurations | Low; already protocol-compatible |
| 3 | Do not activate the 20-config final tuning tier | Low to moderate |
| 4 | Keep 10 final seeds only on primary held-out tasks | Low |
| 5 | Run QR-DQN on held-out tasks only | Low |
| 6 | Report 100k and 500k only; defer 1M | Moderate; weakens RQ3 |
| 7 | Reduce final seeds from 10 to 8 equally | Moderate; last-resort |
| 8 | Defer QR-DQN | High; weakens mechanism control |
| 9 | Reduce held-out MinAtar games | **Do not do** without redesigning the protocol |

Never:

- give one method more tuning because it appears promising;
- remove divergent seeds;
- move a held-out game into tuning after seeing results;
- buy statistical power by mixing tuning games into the primary aggregate.

---

## 11. Compute budget worksheet

Fill this after the provider benchmark.

### 11.1 Measured runtime table

| Provider/GPU | 500k run minutes | 1M run minutes | Setup minutes | Effective cost/quota per run |
|---|---:|---:|---:|---:|
| Modal T4 |  |  |  |  |
| Kaggle GPU |  |  |  |  |
| Colab GPU |  |  |  |  |
| Studio Lab GPU |  |  |  |  |
| Local/VPS CPU |  |  |  |  |

### 11.2 Forecast formulas

```text
M1 pilot GPU-hours =
192 × measured_500k_minutes / 60

M1 held-out GPU-hours =
120 × measured_1M_minutes / 60

QR-DQN pilot GPU-hours =
48 × measured_500k_minutes / 60

QR-DQN held-out GPU-hours =
30 × measured_1M_minutes / 60

Optional all-task GPU-hours =
100 × measured_1M_minutes / 60
```

Add a 15% failure/setup reserve:

```text
planned_gpu_hours = raw_gpu_hours × 1.15
```

### 11.3 Go/no-go thresholds

| Forecast | Action |
|---:|---|
| ≤50 GPU-hours | Modal alone could nominally cover it, but still use Kaggle to preserve reserve |
| 50–150 hours | Kaggle + Modal; normal free-only plan |
| 150–300 hours | Add Studio Lab, spread over multiple weeks/months, cut optional scope |
| 300–500 hours | Reduce pilot configs, inspect utilization, reconsider 1M axis |
| >500 hours | Current implementation/protocol is not a credible free-tier project; redesign before running |

---

## 12. Data storage and archival plan

### During execution

- source/configs: GitHub;
- canonical raw CSV logs: local/VPS with two backups;
- provider temporary output:
  - Modal Volume;
  - Kaggle notebook output;
  - Google Drive for Colab;
  - Studio Lab persistent storage;
- central run ledger: GitHub + local;
- selected checkpoints only: local/VPS.

### At submission

Archive:

- tagged source release;
- lockfile;
- frozen configs and seeds;
- raw CSV logs used in figures;
- run manifests and compute ledger;
- figure-generation scripts;
- selected model checkpoints needed for diagnostics/reproduction;
- protocol and deviations log.

Use OSF or Zenodo for the immutable research release. Do not rely on free notebook storage as the long-term archive.

---

## 13. Accounting required by the paper

Free compute is still compute. Report:

1. total search/tuning GPU-hours by provider and hardware;
2. total CPU-hours for DeepSea and analysis;
3. final held-out evaluation GPU-hours on the reference hardware;
4. wall-clock calendar time;
5. estimated provider list-price value, even when credits made the cash cost zero;
6. actual cash paid: target **$0**;
7. time to reproduce selected final experiments;
8. time to reproduce the full tuning and evaluation process.

Approved wording:

> Every reported final evaluation is reproducible on one modest workstation. Parallel free-tier cloud resources were used to reduce the calendar time of the preregistered hyperparameter search. Total compute, provider hardware, and the full selection cost are reported separately.

---

## 14. Immediate actions

1. Create free accounts for Modal, Kaggle, Colab, and SageMaker Studio Lab.
2. Confirm identity/phone verification requirements.
3. Record each provider's current quota and reset date in `compute/providers.yaml`.
4. Add provider name and hardware fields to the run manifest.
5. Implement the common experiment CLI.
6. Create `runs/ledger.csv`.
7. Create thin Modal, Kaggle, Colab, and Studio Lab launchers.
8. Run one 100-step smoke test everywhere.
9. Run the same 500k-step Double DQN/Breakout benchmark on the available GPU providers.
10. Complete the cost worksheet.
11. Freeze the free-tier tuning counts only after this benchmark.
12. Apply for Modal academic credits, without making approval a project dependency.

---

## 15. Decision summary

The whole core A1 study can plausibly be completed with **$0 cash cloud spend**, but only through disciplined distribution of work:

- **VPS/local:** all CPU work;
- **Kaggle:** bulk GPU volume;
- **Modal Starter:** controlled parallel sweeps and recovery;
- **Colab:** debugging/overflow;
- **SageMaker Studio Lab:** backup daily GPU capacity;
- **Lightning AI:** uncommitted bonus;
- **GitHub Actions:** CI and validation.

The plan must not assume every nominal free hour will be available. Its robustness comes from provider portability, the central run ledger, checkpointing, and the scope-reduction ladder.

The first compute milestone is therefore not a full sweep. It is one reproducible 500k-step benchmark run on each available backend, followed by a measured forecast and a go/no-go decision.
