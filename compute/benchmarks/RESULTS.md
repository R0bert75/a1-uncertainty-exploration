# Stage-A infrastructure benchmark — results

**Status:** CPU lane + Modal A10G GPU lane measured. Kaggle / Colab / Studio Lab GPU lanes
still open — re-run `stage_a_benchmark.py --device cuda` on each to fill the remaining §11.1
worksheet rows.

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

## Modal A10G GPU measurement (`results_modal_a10g.json`)

Backend: Modal `byoc:modal`, 1× NVIDIA A10 (A10G, 24 GB), torch 2.13.0+cu130, batch 64.
Job wall-clock: 9 s. Image `im-pqNlq9TZZ3dXRbXErJnH1C` (cuda 12.4.1 runtime + torch + numpy).

| Loop | CPU idle (ms) | A10G (ms) | speedup |
|---|---:|---:|---:|
| MinAtar DQN (ch4, a6) | 5.3 | 1.6 | 3.3× |
| DeepSea ensemble N20 K10 | 4.8 | 2.5 | 1.9× |
| DeepSea ensemble N20 K20 | 7.1 | 3.8 | 1.9× |
| DeepSea ensemble N40 K20 | 8.2 | 3.8 | 2.2× |

**Finding — the GPU speedup is modest (1.9–3.3×) because these nets are tiny.** MinAtar is a
10×10×4 conv and the DeepSea agents are small ensemble MLPs; at batch 64 they don't saturate an
A10G, so kernel-launch overhead dominates and the GPU wins only ~2–3×. A bigger GPU tier
(A100/H100) would not help — the bottleneck is problem size, not FLOPs. The operational reading:
**a GPU is not the high-value lever for Part A. Escaping CPU contention is.**

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

- The A10G per-step number is a **shape-proxy** rate, not a real-agent per-run wall-clock. The
  real per-run grad-step count for the scientific agents (Session 3+) is still unknown, so the
  forecast columns above are illustrative brackets, not committed cap-X / cap-Y numbers.
- No MinAtar full-length (500k/1M) run has been timed on GPU — the `minatar_gpu_hours_hint`
  (1.6 ms/grad-step) still needs multiplying by real grad-steps/run × run counts to set cap X.

## Next

1. ~~Connect a GPU backend (Modal BYOC) and run `--device cuda`.~~ **Done — A10G lane above.**
2. (Optional) run the remaining hand-run notebook backends (Kaggle / Colab / Studio Lab) with
   `--device cuda` to fill their §11.1 rows; expect similar modest speedups for these shapes.
3. Once the real agents exist (Session 3+), re-run with true grad-steps/run to finalize the
   §11 worksheet and set caps X, Y at the Session-1 freeze.

**Provisioning implication (new):** since the A10G speedup is only ~2–3× and the dominant cost
driver is CPU contention (up to ~25 ms/step loaded vs ~7 idle vs ~3.8 on A10G), the
cost-effective home for Part A's 1,100-run confirmatory sweep is a **dedicated, unshared CPU
box** (or many cheap CPU workers), not GPU. GPU is worth reserving for the MinAtar Part-B
long-horizon runs where the conv net and 1M-step budget make it pay off.
