"""Stage-A infrastructure benchmark (throwaway, informs compute caps X and Y).

This is NOT a scientific run. It is the pre-freeze-eligible infrastructure
benchmark called for by the compute plan (v1.1 §7, §11) and execution plan v4.1
Session 0. It measures per-gradient-step and per-env-step wall-clock for
representative training loops so the §11 worksheet can forecast:

  * MinAtar GPU-hours   (Part B tuning + held-out)   -> cap X, cap Y
  * DeepSea CPU-hours   (Part A confirmatory, ~1,100 runs) -> cap Y

The nets here are SHAPE-representative proxies, not the real agents (which land
in Session 3+). Re-run this exact script unchanged on each real backend
(Modal T4, Kaggle, Colab, Studio Lab) to fill the §11.1 measured-runtime table;
the `--device` flag targets CPU or CUDA. Results are emitted as JSON so they can
be committed next to the plan and diffed across backends.

Usage:
    python compute/benchmarks/stage_a_benchmark.py --device cpu --out results_local_cpu.json
    python compute/benchmarks/stage_a_benchmark.py --device cuda --out results_modal_t4.json
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch


# ----------------------------------------------------------------------------- #
# Representative networks (shape-matched proxies, NOT the scientific agents)
# ----------------------------------------------------------------------------- #
class MinAtarDQN(torch.nn.Module):
    """MinAtar-shaped conv net: (B, C=in_ch, 10, 10) -> Q over `n_actions`.

    Matches the MinAtar convention (10x10 grid, per-game channel count) and the
    standard MinAtar DQN torso (single 16-filter 3x3 conv + 128-unit head).
    """

    def __init__(self, in_ch: int = 4, n_actions: int = 6, hidden: int = 128):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, 16, kernel_size=3, stride=1)
        # 10x10 -> 8x8 after a valid 3x3 conv
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(16 * 8 * 8, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.fc(torch.relu(self.conv(x)))


class DeepSeaEnsemble(torch.nn.Module):
    """DeepSea-shaped bootstrapped ensemble MLP: state (2N one-hot-ish) -> K heads x 2 actions."""

    def __init__(self, n: int, k: int, hidden: int = 64):
        super().__init__()
        self.torso = torch.nn.Sequential(
            torch.nn.Linear(2 * n, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
        )
        self.heads = torch.nn.ModuleList([torch.nn.Linear(hidden, 2) for _ in range(k)])

    def forward(self, x):
        h = self.torso(x)
        return torch.stack([head(h) for head in self.heads], dim=1)  # (B, K, 2)


# ----------------------------------------------------------------------------- #
# Benchmark kernels
# ----------------------------------------------------------------------------- #
@dataclass
class BenchResult:
    label: str
    device: str
    batch: int
    warmup_steps: int
    timed_steps: int
    ms_per_step: float
    steps_per_sec: float
    extra: dict


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def bench_minatar(device: str, in_ch: int, n_actions: int, batch: int,
                  warmup: int, steps: int) -> BenchResult:
    net = MinAtarDQN(in_ch, n_actions).to(device)
    tgt = MinAtarDQN(in_ch, n_actions).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=2.5e-4)
    gen = torch.Generator(device="cpu").manual_seed(0)

    def batch_of():
        s = torch.rand(batch, in_ch, 10, 10, generator=gen).to(device)
        s2 = torch.rand(batch, in_ch, 10, 10, generator=gen).to(device)
        a = torch.randint(0, n_actions, (batch,), generator=gen).to(device)
        r = torch.rand(batch, generator=gen).to(device)
        return s, s2, a, r

    for _ in range(warmup):
        s, s2, a, r = batch_of()
        q = net(s).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            tgt_q = r + 0.99 * tgt(s2).max(1).values
        loss = torch.nn.functional.smooth_l1_loss(q, tgt_q)
        opt.zero_grad()
        loss.backward()
        opt.step()
    _sync(device)

    t0 = time.perf_counter()
    for _ in range(steps):
        s, s2, a, r = batch_of()
        q = net(s).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            tgt_q = r + 0.99 * tgt(s2).max(1).values
        loss = torch.nn.functional.smooth_l1_loss(q, tgt_q)
        opt.zero_grad()
        loss.backward()
        opt.step()
    _sync(device)
    dt = time.perf_counter() - t0
    return BenchResult(
        label=f"minatar_dqn_ch{in_ch}_a{n_actions}", device=device, batch=batch,
        warmup_steps=warmup, timed_steps=steps,
        ms_per_step=dt / steps * 1000, steps_per_sec=steps / dt,
        extra={"in_ch": in_ch, "n_actions": n_actions},
    )


def bench_deepsea(device: str, n: int, k: int, batch: int,
                  warmup: int, steps: int) -> BenchResult:
    net = DeepSeaEnsemble(n, k).to(device)
    tgt = DeepSeaEnsemble(n, k).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    gen = torch.Generator(device="cpu").manual_seed(0)

    for _ in range(warmup):
        s = torch.randn(batch, 2 * n, generator=gen).to(device)
        q = net(s)
        with torch.no_grad():
            qn = tgt(torch.randn(batch, 2 * n, generator=gen).to(device)).max(-1).values
        a = torch.randint(0, 2, (batch,), generator=gen).to(device)
        qa = q.gather(-1, a[:, None, None].expand(batch, k, 1)).squeeze(-1)
        mask = (torch.rand(batch, k, generator=gen).to(device) > 0.5).float()
        loss = (mask * (qa - 0.99 * qn) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    _sync(device)

    t0 = time.perf_counter()
    for _ in range(steps):
        s = torch.randn(batch, 2 * n, generator=gen).to(device)
        q = net(s)
        with torch.no_grad():
            qn = tgt(torch.randn(batch, 2 * n, generator=gen).to(device)).max(-1).values
        a = torch.randint(0, 2, (batch,), generator=gen).to(device)
        qa = q.gather(-1, a[:, None, None].expand(batch, k, 1)).squeeze(-1)
        mask = (torch.rand(batch, k, generator=gen).to(device) > 0.5).float()
        loss = (mask * (qa - 0.99 * qn) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    _sync(device)
    dt = time.perf_counter() - t0
    return BenchResult(
        label=f"deepsea_ens_N{n}_K{k}", device=device, batch=batch,
        warmup_steps=warmup, timed_steps=steps,
        ms_per_step=dt / steps * 1000, steps_per_sec=steps / dt,
        extra={"N": n, "K": k},
    )


# ----------------------------------------------------------------------------- #
# Forecast helper (turns ms/step into the §11 worksheet numbers)
# ----------------------------------------------------------------------------- #
def forecast(results: dict) -> dict:
    """Map measured ms/step to the v6.1 run-budget forecast (illustrative grad-step counts)."""
    out = {"deepsea_confirmatory_1100_runs": {}, "minatar_gpu_hours_hint": {}}
    # DeepSea confirmatory: 1,100 runs, sweep over plausible grad-steps/run.
    ds = next((r for r in results["benchmarks"] if r["label"].startswith("deepsea")), None)
    if ds:
        ms = ds["ms_per_step"]
        for gs in (20_000, 50_000, 100_000):
            cpu_h = 1100 * gs * ms / 1000 / 3600
            out["deepsea_confirmatory_1100_runs"][f"{gs}_gradsteps"] = {
                "cpu_hours_serial": round(cpu_h, 1),
                "wall_h_at_4x": round(cpu_h / 4, 1),
                "wall_h_at_6x": round(cpu_h / 6, 1),
            }
    # MinAtar: report ms/step; the run-level forecast needs the real 500k/1M
    # per-run wall-clock measured on a GPU backend, so only give the per-step hint here.
    mn = next((r for r in results["benchmarks"] if r["label"].startswith("minatar")), None)
    if mn:
        ms = mn["ms_per_step"]
        out["minatar_gpu_hours_hint"] = {
            "ms_per_gradstep": round(ms, 3),
            "note": (
                "multiply by real grad-steps-per-run (from a 500k/1M run) then by run "
                "counts (240 pilot / 800 gated / 120 held-out) for cap X, Y"
            ),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (ignored on cuda)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", default="stage_a_results.json")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False.")
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    benches = []
    # MinAtar: Breakout has 4 channels / 6 actions; representative of the game set.
    benches.append(bench_minatar(args.device, in_ch=4, n_actions=6,
                                 batch=args.batch, warmup=args.warmup, steps=args.steps))
    # DeepSea: the K=20 ensemble at N in {20,40} bounds the confirmatory cost.
    for n, k in [(20, 10), (20, 20), (40, 20)]:
        benches.append(bench_deepsea(args.device, n=n, k=k,
                                     batch=args.batch, warmup=args.warmup, steps=args.steps))

    results = {
        "meta": {
            "device": args.device,
            "threads": args.threads if args.device == "cpu" else None,
            "torch_version": torch.__version__,
            "torch_cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "batch": args.batch,
            "note": (
                "Throwaway infra benchmark; shape-representative proxies, "
                "not scientific agents."
            ),
        },
        "benchmarks": [asdict(b) for b in benches],
    }
    results["forecast"] = forecast(results)

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"device={args.device} torch={torch.__version__} cuda={torch.cuda.is_available()}")
    for b in benches:
        print(f"  {b.label:<24} {b.ms_per_step:7.3f} ms/step  {b.steps_per_sec:8.1f} step/s")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
