# Stage-A infrastructure benchmark — results

**Status:** partial (CPU lane only). GPU lanes (Modal / Kaggle / Colab / Studio Lab) are
**not yet measured** — no GPU backend is connected. Re-run `stage_a_benchmark.py --device cuda`
on each backend to fill the GPU rows of the §11.1 worksheet.

**Nature:** throwaway infrastructure benchmark (compute plan v1.1 §7, §11; execution plan v4.1
Session 0). The networks are **shape-representative proxies**, not the scientific agents
(which arrive in Session 3+). Purpose: measure per-gradient-step wall-clock to seed the
compute-cap forecast, not to produce any scientific result.

## Local CPU measurement (`results_local_cpu.json`)

Box: 8 cores, torch 2.13.0+cpu, 4 threads, batch 64, Python 3.11.15.

| Loop | ms / grad-step | steps/s |
|---|---:|---:|
| MinAtar DQN (ch4, a6) | 5.3 | 189 |
| DeepSea ensemble N20 K10 | 4.8 | 207 |
| DeepSea ensemble N20 K20 | 7.1 | 142 |
| DeepSea ensemble N40 K20 | 8.2 | 121 |

## Key finding — load contention is the dominant variable

The **same** K=20 DeepSea shape measured **~7 ms/step idle** here but **~25 ms/step** in an
earlier run while the shared box was under concurrent load — a **3–5× swing** from contention
alone, same code, same shapes. On a shared box this band, not the point estimate, is what the
forecast must carry.

## DeepSea Part-A confirmatory forecast (1,100 runs)

| grad-steps/run | idle (~7 ms) CPU-h | loaded (~25 ms) CPU-h | loaded, wall @6× |
|---:|---:|---:|---:|
| 20k | 43 | 153 | ~1.1 d |
| 50k | 108 | 382 | ~2.7 d |
| 100k | 216 | 764 | ~5.3 d |

**Reading:** even idle, the confirmatory DeepSea sweep is tens-to-hundreds of CPU-hours; under
realistic shared-box contention it is **1–5 days of wall-clock** and sits on the confirmatory
critical path. This confirms compute-plan v1.1 §3.3: Part A is a first-class CPU cost, not a
free afterthought, and argues for a **dedicated (unshared) CPU box** for the confirmatory sweep.

## What this does NOT yet give

- No GPU per-run wall-clock (needs a 500k/1M run on a real backend) → cap X and the MinAtar
  side of cap Y are still open.
- The real per-run grad-step count for DeepSea (the agent doesn't exist yet) → the three
  columns above are illustrative brackets, not a committed number.

## Next

1. Connect a GPU backend (Modal BYOC) and run `--device cuda`.
2. Once the real agents exist (Session 3+), re-run with true grad-steps/run to finalize the
   §11 worksheet and set caps X, Y at the Session-1 freeze.
