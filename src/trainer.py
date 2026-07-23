"""End-to-end training loop for Part A (DeepSea) — the run entrypoint.

This module is *execution* infrastructure: it wires the already-frozen pieces
(:mod:`src.config` factories → env + agent, :class:`src.utils.conventions.CSVLogger`)
into an episode loop and writes ``logs/<run_id>.csv``. It decides **no** scientific
parameter — every number comes from the resolved YAML config and the pinned seed
streams — so it is safe to add after the ``prereg-draft`` freeze.

What it logs (matching the frozen primary outcome, protocol/preregistration.md §1.1):

* ``discovery_prob`` — the per-seed discovery **indicator** (0 until the first episode
  with a strictly positive return, 1 thereafter). Averaged across seeds downstream, the
  per-step mean is the discovery *probability* curve the confirmatory analysis consumes.
* ``episode_return`` — mean episode return over the checkpoint window (a diagnostic).

Both are logged on ``axis="online"`` at a fixed number of checkpoints spread over the
episode budget. DeepSea episodes are exactly ``size`` env-steps long, so the cumulative
env-step at a checkpoint is identical across seeds — a shared ``step`` grid that
``analysis/make_figures.py`` groups on directly.

CLI::

    python -m src.trainer --config configs/example_bdqn_deepsea_dev.yaml --out logs

Determinism: all randomness flows through the derived seed streams inside the env and
agent (built by the config factories from ``master_seed``/``cell_id``/``seed_index``);
the loop itself introduces none. Re-running a (config, seed) reproduces the CSV
bit-for-bit (gate C1).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src import config as config_mod
from src.utils.conventions import CSVLogger

# Number of evenly spaced checkpoints logged over the episode budget. This is a
# *reporting* cadence (how densely the online curve is sampled), not a scientific
# parameter — the primary outcome is the end-of-budget indicator, which is the last
# checkpoint regardless of density.
DEFAULT_CHECKPOINTS = 20


def _checkpoint_episodes(n_episodes: int, n_checkpoints: int) -> list[int]:
    """The 1-based episode indices at which a checkpoint is logged.

    Always includes the final episode so the end-of-budget indicator is captured. Evenly
    spaced; de-duplicated when ``n_checkpoints`` exceeds ``n_episodes``.
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")
    k = max(1, min(n_checkpoints, n_episodes))
    # Evenly spaced endpoints e_1..e_k with e_k == n_episodes.
    marks = sorted({round(n_episodes * (i + 1) / k) for i in range(k)})
    marks = [m for m in marks if m >= 1]
    if marks[-1] != n_episodes:
        marks.append(n_episodes)
    return marks


def run_seed(
    cfg: config_mod.RunConfig,
    seed_index: int,
    log: CSVLogger,
    *,
    n_checkpoints: int = DEFAULT_CHECKPOINTS,
) -> dict[str, float]:
    """Run one seed end-to-end and log its online metrics. Returns a small summary dict.

    The agent and env are built by the frozen config factories (which seed every stream
    from ``master_seed``/``cell_id``/``seed_index``); this function adds no RNG of its own.
    """
    env = config_mod.build_env(cfg, seed_index)
    agent = config_mod.build_agent(cfg, seed_index)

    budget = cfg.data["env_budget"]
    n_episodes = int(budget["episodes"])
    size = int(budget["deep_sea_size"])

    checkpoints = _checkpoint_episodes(n_episodes, n_checkpoints)
    checkpoint_set = set(checkpoints)

    step = 0  # cumulative env-interaction steps (the budget axis)
    discovered = False  # becomes True at the first strictly-positive-return episode
    ck_index = 0
    window_returns: list[float] = []  # returns since the last checkpoint

    for episode in range(1, n_episodes + 1):
        obs, _ = env.reset()
        obs = obs.reshape(-1)  # DeepSea yields a 2D one-hot grid; the network wants it flat
        if hasattr(agent, "on_episode_start"):
            agent.on_episode_start()  # BDQN: resample the episode's active head

        done = False
        ep_return = 0.0
        while not done:
            action = agent.select_action(obs, step)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_obs = next_obs.reshape(-1)
            agent.observe(obs, action, reward, next_obs, terminated)
            agent.learn_step()
            obs = next_obs
            ep_return += float(reward)
            step += 1
            done = terminated or truncated

        window_returns.append(ep_return)
        if ep_return > 0.0:  # first strictly positive terminal reward → discovery
            discovered = True

        if episode in checkpoint_set:
            mean_return = sum(window_returns) / len(window_returns)
            is_t0 = ck_index == 0
            log.log(
                step=step,
                metric="discovery_prob",
                value=float(discovered),
                checkpoint=ck_index,
                is_t0=is_t0,
                axis="online",
            )
            log.log(
                step=step,
                metric="episode_return",
                value=mean_return,
                checkpoint=ck_index,
                is_t0=is_t0,
                axis="online",
            )
            ck_index += 1
            window_returns = []

    return {
        "seed": float(seed_index),
        "discovered": float(discovered),
        "total_steps": float(step),
        "episodes": float(n_episodes),
        "size": float(size),
    }


def train(
    cfg: config_mod.RunConfig,
    out_dir: str | Path = "logs",
    *,
    n_checkpoints: int = DEFAULT_CHECKPOINTS,
) -> Path:
    """Run every committed seed of ``cfg`` into a single ``logs/<run_id>.csv``.

    Also serializes the resolved config to ``<out_dir>/resolved_config.json`` (C13 input),
    so the run's identity fingerprint is committed alongside its logs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.write_resolved(out_dir)

    csv_path = out_dir / f"{cfg.run_id}.csv"
    if csv_path.exists():
        csv_path.unlink()  # append-only logger; start fresh for a clean re-run

    summaries: list[dict[str, float]] = []
    for seed_index in cfg.seeds:
        ctx = cfg.run_context(seed_index)
        with CSVLogger(csv_path, ctx) as log:
            summaries.append(run_seed(cfg, seed_index, log, n_checkpoints=n_checkpoints))

    n_discovered = int(sum(s["discovered"] for s in summaries))
    print(
        f"{cfg.run_id}: {len(summaries)} seeds, "
        f"{n_discovered}/{len(summaries)} discovered → {csv_path}"
    )
    return csv_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path, help="path to a resolved run YAML")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("logs"),
        help="output dir for CSV + resolved config",
    )
    ap.add_argument(
        "--checkpoints",
        type=int,
        default=DEFAULT_CHECKPOINTS,
        help="number of evenly spaced online checkpoints over the episode budget",
    )
    args = ap.parse_args(argv)

    cfg = config_mod.load_config(args.config)
    train(cfg, args.out, n_checkpoints=args.checkpoints)
    return 0


if __name__ == "__main__":
    sys.exit(main())
