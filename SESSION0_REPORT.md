# Session 0 — Bootstrap report

**Project:** *When Do Randomized Value Estimates Buy Exploration? Separating the Estimator
from the Use Rule under Low Interaction Budgets.*
**Source of truth:** requirements spec **v6.1**, execution plan **v4.1**.
**Date:** July 2026 · **Git tag:** `session-0-bootstrap` (HEAD `09624d3`)

---

## 1. What Session 0 produced

A reproducible, CI-checked repository skeleton under local git history — **no science yet**.
Deliverables:

| Deliverable | Location |
|---|---|
| Pinned Python 3.11 environment | conda env `a1-rl`; `requirements.txt` + `build_manifest/pip_freeze_full.txt` |
| Repo skeleton (Appendix B + v4.1 dirs) | `a1-uncertainty-exploration/` |
| Determinism + CSV-logging conventions (Appendix C) | `src/utils/conventions.py` |
| `make figures` on a dummy CSV | `analysis/make_figures.py`, `analysis/make_dummy_logs.py` |
| CI smoke workflow + VALIDATION C0–C13 | `.github/workflows/smoke.yml`, `VALIDATION.md` |
| Local git history (protocol committed first) | 6 commits, root = `protocol/` |

## 2. Pinned versions (reference environment)

| Package | Version | Note |
|---|---|---|
| python | 3.11.15 | |
| torch | 2.13.0+cpu | **CPU-only** wheel; `torch.cuda.is_available()` is `False` |
| gymnasium | 1.3.0 | |
| minatar | 1.0.15 | |
| rliable | 1.2.0 | |
| numpy | 2.4.6 | |
| pandas | 3.0.3 | |
| scipy | 1.17.1 | |
| matplotlib | 3.11.0 | |
| seaborn | 0.13.2 | |
| pyyaml | 6.0.3 | |
| tyro | 1.0.15 | |
| pytest | 9.1.1 | dev |
| ruff | 0.15.21 | dev |

Full lock: 71 packages in `build_manifest/pip_freeze_full.txt`
(sha256 `791eadee88f9dbdefeb1f10caf7b95e49c4662b8cb5f0df7276f439a6a12ccbe`).

**torch install note:** the CPU wheel is not on PyPI; it comes from
`download.pytorch.org/whl/cpu` (which redirects to `download-r2.pytorch.org`). Both hosts
had to be allow-listed in this sandbox. `make env` and the CI workflow install it via
`--index-url https://download.pytorch.org/whl/cpu`.

## 3. Acceptance gate — what passed

Local smoke check, all green:

- **ruff** — clean (`ruff check src analysis audits tests`).
- **pytest** — **12/12 pass** (`tests/test_conventions.py`, `tests/test_figures.py`):
  determinism reproducibility (C1), CSV frozen-schema + role enforcement incl.
  QR-DQN→exploratory (C2), config-identity hashing + resolved-config serialization (C13),
  and the dummy-CSV → `make figures` → PNG round-trip.
- **`make figures`** — rebuilt `figures/partA_deep_sea_discovery_prob.png` from
  `logs/dummy_smoke.csv` **alone** (60 rows, 2 methods × 3 seeds). Data→figure path is the
  only path (C2 contract).
- **`make audit`** — C13 audit runs; reports "no contrasts registered yet" (expected — the
  registry is filled at protocol freeze).
- **CPU-only assertion** — CI asserts `torch.cuda.is_available() is False`.

The environment smoke also confirmed: MinAtar Breakout returns a `[10,10,4]` state, a torch
`Conv2d` runs on it, and `bsuite` is **absent** (see §4).

## 4. Decisions & deviations

- **`bsuite` not installed → single-file DeepSea.** `bsuite` is unmaintained and pins an
  old `gym` incompatible with the pinned gymnasium 1.3.0 / numpy 2.x. DeepSea will be a
  single tested file validated against the published spec on small N (gate C5). The spec
  sanctions this as dependency hygiene, not a cut-list exception.
- **CPU-only torch.** The box has no GPU and MinAtar/DeepSea are CPU-scale; the CUDA wheel
  (~2.5 GB) also would not fit the ~3 GB free disk. The pinned build is `2.13.0+cpu`.
- **Working title / framing updated to v6.1:** "replication and extension" of Osband et al.
  (2016); no first-ness language; "transfer" retired.

## 5. Deferred (not done in Session 0, by design or by blocker)

- **GitHub push — DONE.** Repo pushed public:
  `https://github.com/R0bert75/a1-uncertainty-exploration` (branch `main`, tag
  `session-0-bootstrap`), protocol-first commit history preserved as the public
  pre-registration provenance. **CI smoke workflow passed on GitHub runners**
  (CPU-torch assertion + ruff + pytest 12/12 + figure rebuild + C13 audit), independently
  confirming the pinned environment reproduces off this box.
- **OSF mirror + two-stage freeze tags (`prereg-draft` → final) — deferred to Session 1**,
  after the protocol values are filled and frozen.
- **Protocol values — deferred to Session 1** (the 20-item freeze list is a stub; the
  two-stage freeze and the final OSF mirror are Session-1 work). Scientific runs
  (Session 3 onward) require the final freeze.
- **Throwaway infra benchmark for compute cap X** — a Session-0-eligible item, deferred
  pending the user's multi-cloud compute context (local-only for now).
- **Method implementations, real figures, C13 contrast registry** — Sessions 3+.

## 6. Compute posture

Local-only for now (8 CPU / ~15 GiB RAM / no GPU). No compute providers connected. Pre-tuning
work runs locally; the Session-7+ tuning sweeps are the calendar bottleneck that will want a
burst lane once the user provides multi-cloud context.
