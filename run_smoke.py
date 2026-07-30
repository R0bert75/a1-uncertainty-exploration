"""Session-3 pipeline-spine smoke run: DDQN on MinAtar Breakout, step lane.

Not a pre-registered run — a smoke-scale exercise of the full Part-B path (conv trunk ->
step-budgeted trainer -> frozen CSV schema -> figure). Budget is deliberately far below the
pre-registered ladder; the output demonstrates plumbing, not learning.
"""

from pathlib import Path

import yaml

from src.config import resolve_config
from src.trainer import train

OUT = Path("dev_battery/smoke_breakout")


def main() -> int:
    d = yaml.safe_load(open("configs/example_ddqn_breakout_dev.yaml"))
    d["run_id"] = "smoke_ddqn_breakout"
    d["role"] = "exploratory"  # smoke-scale, not development-tier evidence
    d["env_budget"] = {"total_steps": 30_000, "checkpoint_steps": [10_000, 20_000, 30_000]}
    d["factor_specific"]["eps_schedule"]["eps_decay_steps"] = 10_000  # scaled to the budget
    d["seeds"] = [0, 1, 2]
    cfg = resolve_config(d)
    path = train(cfg, OUT, n_eval_episodes=5)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
