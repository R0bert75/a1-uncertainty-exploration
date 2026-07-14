"""Emit a schema-correct DUMMY log CSV so ``make figures`` and CI have input in Session 0.

The numbers are meaningless synthetic values — the point is to exercise the data→figure
path end to end before any real training exists. Writes ``logs/dummy_smoke.csv`` using the
exact frozen schema from ``utils.conventions`` (so it can never drift from the real logger).

Usage:
    python analysis/make_dummy_logs.py --out logs/dummy_smoke.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Import the real logger so the dummy shares the frozen schema (single source of truth).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.conventions import (  # noqa: E402
    CSVLogger,
    RunContext,
    config_hash,
    seed_everything,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("logs/dummy_smoke.csv"))
    args = ap.parse_args(argv)

    seed_everything(0)
    steps = np.arange(0, 5000, 500)

    # Two methods, three seeds, one metric — enough for a mean±band placeholder figure.
    if args.out.exists():
        args.out.unlink()
    for method, base in [("ddqn_egreedy", 0.2), ("bdqn", 0.5)]:
        cfg = {"method": method, "env": "deep_sea", "use_rule": "episodic",
               "prior": "off", "K": 10}
        h = config_hash(cfg)
        for seed in range(3):
            ctx = RunContext(run_id=f"dummy_{method}_s{seed}", role="development",
                             part="A", method=method, env="deep_sea", seed=seed,
                             config_sha256=h)
            rng = np.random.default_rng(seed)
            with CSVLogger(args.out, ctx) as log:
                for s in steps:
                    val = float(np.clip(base * (s / steps.max()) + rng.normal(0, 0.05), 0, 1))
                    log.log(step=int(s), metric="discovery_prob", value=val)

    n = sum(1 for _ in args.out.open()) - 1
    print(f"wrote {args.out} ({n} data rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
