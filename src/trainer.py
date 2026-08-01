"""End-to-end training loop for both parts — the run entrypoint.

This module is *execution* infrastructure: it wires the already-frozen pieces
(:mod:`src.config` factories → env + agent, :class:`src.utils.conventions.CSVLogger`)
into a training loop and writes ``logs/<run_id>.csv``. Two lanes share every frozen piece
and differ only in the budget axis: **Part A (DeepSea)** is episode-budgeted
(:func:`run_seed`), **Part B (MinAtar)** is step-budgeted on the pre-registered checkpoint
grid (:func:`run_seed_steps`); ``train`` dispatches on the env family. It decides **no** scientific
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
import json
import sys
import time
from pathlib import Path

from src import config as config_mod
from src.diagnostics import recorder as diag_recorder
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


#: Frozen-policy evaluation episodes per checkpoint (Part B secondary axis). A *reporting*
#: cadence, not a scientific parameter: the extraction rule itself is pre-registered, this is
#: only how many episodes it is averaged over. Overridable from the CLI.
DEFAULT_EVAL_EPISODES = 5

#: Hard cap on env steps per frozen-policy evaluation episode, so a degenerate deterministic
#: policy (e.g. one that never fires in Breakout) cannot stall a run. Diagnostic guard only.
EVAL_EPISODE_STEP_CAP = 5_000


def frozen_policy_action(agent, obs) -> int:
    """The pre-registered deterministic-policy extraction for the ``frozen_policy`` axis.

    Spec §5 "two reporting axes" pins one extraction per method: DDQN → ``greedy(Q)``;
    NoisyNet → noise-off greedy; bootstrapped cells → greedy w.r.t. ensemble-mean Q. Each
    agent exposes exactly one of those two entry points, so dispatch is by capability rather
    than by method name — which keeps this correct for the ``bdqn``/``rp_bdqn`` alias pair and
    for any use-rule, and draws no randomness (measuring cannot perturb the run; gate C1).
    """
    if hasattr(agent, "mean_action"):  # NoisyNet (noise-off) and BDQN/RP-BDQN (ensemble mean)
        return agent.mean_action(obs)
    return agent.greedy_action(obs)  # DDQN: greedy(Q)


def _evaluate_frozen_policy(
    cfg: config_mod.RunConfig,
    agent,
    seed_index: int,
    n_episodes: int,
) -> float:
    """Mean episode return of the extracted deterministic policy, on a throwaway env.

    A **separate** env instance is built for evaluation so the training env's RNG state is
    untouched — measurement must not perturb the run being measured (gate C1). Evaluation adds
    no RNG of its own: the policy is deterministic and the eval env is seeded by the same
    frozen derivation as the training env, so this returns the same value on a re-run.
    """
    eval_env = config_mod.build_env(cfg, seed_index)
    flatten = not config_mod.is_minatar(cfg.env)
    returns: list[float] = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        if flatten:
            obs = obs.reshape(-1)
        ep_return = 0.0
        for _ in range(EVAL_EPISODE_STEP_CAP):
            action = frozen_policy_action(agent, obs)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            if flatten:
                obs = obs.reshape(-1)
            ep_return += float(reward)
            if terminated or truncated:
                break
        returns.append(ep_return)
    return sum(returns) / len(returns)


def run_seed_steps(
    cfg: config_mod.RunConfig,
    seed_index: int,
    log: CSVLogger,
    *,
    n_eval_episodes: int = DEFAULT_EVAL_EPISODES,
) -> dict[str, float]:
    """Run one seed of a **step-budgeted** (MinAtar / Part B) config end-to-end.

    The Part-B budget axis is cumulative **env steps** with checkpoints at the pre-registered
    step grid (spec §5 Variant B: 100k/500k/1M), not episodes — so a checkpoint can fall
    mid-episode and the step grid is identical across seeds by construction. Both frozen
    reporting axes are logged at every checkpoint:

    * ``episode_return`` on ``axis="online"`` — mean return of episodes *completed* in the
      window since the previous checkpoint; the acting policy is the method under its own
      definition (primary axis).
    * ``episode_return`` on ``axis="frozen_policy"`` — mean return of the extracted
      deterministic policy over ``n_eval_episodes`` episodes (secondary axis).

    Also logged once, on the first checkpoint at or after it occurs:

    * ``steps_to_first_reward`` — the exploratory time-to-first-reward proxy (spec §5 "direct
      exploration outcomes"). Logged as the total budget if no reward was ever seen, so the
      column is never missing and the censoring is explicit.

    Adds no RNG of its own; every stream is derived inside the env and agent.
    """
    env = config_mod.build_env(cfg, seed_index)
    agent = config_mod.build_agent(cfg, seed_index)
    t_start = time.perf_counter()

    total_steps, checkpoints = config_mod.step_budget(cfg)
    ck_index = 0
    next_ck = checkpoints[0]

    step = 0
    steps_to_first_reward: int | None = None
    first_reward_logged = False
    window_returns: list[float] = []  # episodes COMPLETED since the last checkpoint
    ep_return = 0.0
    episodes_completed = 0

    obs, _ = env.reset()
    if hasattr(agent, "on_episode_start"):
        agent.on_episode_start()

    while step < total_steps:
        action = agent.select_action(obs, step)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        agent.observe(obs, action, reward, next_obs, terminated)
        agent.learn_step()
        obs = next_obs
        ep_return += float(reward)
        step += 1
        if reward > 0.0 and steps_to_first_reward is None:
            steps_to_first_reward = step

        if terminated or truncated:
            window_returns.append(ep_return)
            episodes_completed += 1
            ep_return = 0.0
            obs, _ = env.reset()
            if hasattr(agent, "on_episode_start"):
                agent.on_episode_start()  # BDQN: resample the episode's active head

        # Checkpoints are keyed on cumulative steps, so they may fall mid-episode.
        while ck_index < len(checkpoints) and step >= next_ck:
            is_t0 = ck_index == 0
            # Mean over episodes completed in this window; None-safe when a window closes
            # before any episode terminated (possible at a dense early checkpoint).
            mean_return = (
                sum(window_returns) / len(window_returns) if window_returns else float("nan")
            )
            log.log(
                step=step,
                metric="episode_return",
                value=mean_return,
                checkpoint=ck_index,
                is_t0=is_t0,
                axis="online",
            )
            # The window's episode count travels as its own metric row, not an extra column:
            # the CSV header is frozen (gate C2) and takes no additional fields.
            log.log(
                step=step,
                metric="episodes_in_window",
                value=float(len(window_returns)),
                checkpoint=ck_index,
                is_t0=is_t0,
                axis="online",
            )
            log.log(
                step=step,
                metric="episode_return",
                value=_evaluate_frozen_policy(cfg, agent, seed_index, n_eval_episodes),
                checkpoint=ck_index,
                is_t0=is_t0,
                axis="frozen_policy",
            )
            if not first_reward_logged and steps_to_first_reward is not None:
                log.log(
                    step=step,
                    metric="steps_to_first_reward",
                    value=float(steps_to_first_reward),
                    checkpoint=ck_index,
                    is_t0=is_t0,
                    axis="online",
                )
                first_reward_logged = True
            window_returns = []
            ck_index += 1
            if ck_index < len(checkpoints):
                next_ck = checkpoints[ck_index]

    if not first_reward_logged:
        # Explicitly censored at the budget rather than absent from the CSV.
        log.log(
            step=step,
            metric="steps_to_first_reward",
            value=float(total_steps),
            checkpoint=max(0, ck_index - 1),
            is_t0=False,
            axis="online",
        )

    # Per-seed wall-clock: a descope-ladder trigger input (spec §8 item 4) and a v1.0
    # reporting requirement ("compute reported"). It is deliberately NOT written to the
    # metrics CSV: wall-clock is machine-dependent, and gate C1 requires a (config, seed)
    # re-run to reproduce that CSV byte-for-byte. It travels in a sidecar instead.
    wall_clock_s = time.perf_counter() - t_start

    return {
        "seed": float(seed_index),
        "total_steps": float(step),
        "episodes": float(episodes_completed),
        "wall_clock_s": wall_clock_s,
        "steps_to_first_reward": float(
            steps_to_first_reward if steps_to_first_reward is not None else total_steps
        ),
        "first_reward_censored": float(steps_to_first_reward is None),
    }


def run_seed(
    cfg: config_mod.RunConfig,
    seed_index: int,
    log: CSVLogger,
    *,
    n_checkpoints: int = DEFAULT_CHECKPOINTS,
    diagnostics_dir: Path | None = None,
) -> dict[str, float]:
    """Run one seed end-to-end and log its online metrics. Returns a small summary dict.

    The agent and env are built by the frozen config factories (which seed every stream
    from ``master_seed``/``cell_id``/``seed_index``); this function adds no RNG of its own.

    When ``diagnostics_dir`` is given, the §3.3 value-sample substrate is recorded at every
    online checkpoint into that directory. This does **not** alter the run: measurement draws
    from the ``noisynet_diag`` stream (or, for ensembles, from no stream at all), and nothing
    diagnostic is written to the metrics CSV — so the CSV is byte-identical with and without
    the flag, and gate C1 keeps its meaning. See :mod:`src.diagnostics.recorder`.
    """
    env = config_mod.build_env(cfg, seed_index)
    agent = config_mod.build_agent(cfg, seed_index)
    recorder = (
        diag_recorder.make_recorder(agent, cfg, seed_index)
        if diagnostics_dir is not None
        else None
    )
    t_start = time.perf_counter()

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
            if recorder is not None:
                recorder.observe_state(obs)  # v(s) for diagnostic #5; no RNG, no policy effect
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
            if recorder is not None:
                # Deliberately NOT logged to the CSV — see src/diagnostics/recorder.py.
                recorder.record(step)
            ck_index += 1
            window_returns = []

    # Per-seed wall-clock: a descope-ladder trigger input (spec §8 item 4) and a v1.0
    # reporting requirement ("compute reported"). It is deliberately NOT written to the
    # metrics CSV: wall-clock is machine-dependent, and gate C1 requires a (config, seed)
    # re-run to reproduce that CSV byte-for-byte. It travels in a sidecar instead.
    wall_clock_s = time.perf_counter() - t_start

    if recorder is not None:
        recorder.write(diagnostics_dir)

    return {
        "seed": float(seed_index),
        "discovered": float(discovered),
        "total_steps": float(step),
        "episodes": float(n_episodes),
        "size": float(size),
        "wall_clock_s": wall_clock_s,
    }


def _write_compute_sidecar(
    out_dir: Path, cfg: config_mod.RunConfig, summaries: list[dict[str, float]]
) -> Path:
    """Write per-seed wall-clock to ``<out_dir>/<run_id>.compute.json``.

    Spec §8 item 4 makes per-method wall-clock a **descope-ladder trigger input**, and §6
    v1.0 requires compute reported. It is kept out of the metrics CSV on purpose: wall-clock
    is machine-dependent, and gate C1 requires a ``(config, seed)`` re-run to reproduce that
    CSV byte-for-byte. A sidecar keeps the requirement satisfied without weakening the gate.
    """
    payload = {
        "run_id": cfg.run_id,
        "method": cfg.method,
        "env": cfg.env,
        "cell_id": cfg.cell_id,
        "config_sha256": cfg.config_sha256,
        "per_seed_wall_clock_s": {
            str(int(s["seed"])): round(s["wall_clock_s"], 3) for s in summaries
        },
        "total_wall_clock_s": round(sum(s["wall_clock_s"] for s in summaries), 3),
        "total_env_steps": int(sum(s["total_steps"] for s in summaries)),
    }
    path = out_dir / f"{cfg.run_id}.compute.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def train(
    cfg: config_mod.RunConfig,
    out_dir: str | Path = "logs",
    *,
    n_checkpoints: int = DEFAULT_CHECKPOINTS,
    n_eval_episodes: int = DEFAULT_EVAL_EPISODES,
    diagnostics: bool = False,
) -> Path:
    """Run every committed seed of ``cfg`` into a single ``logs/<run_id>.csv``.

    Dispatches on the env family: MinAtar (Part B) configs run the **step-budgeted** lane
    (:func:`run_seed_steps`, checkpoints on the pre-registered step grid, both reporting
    axes), DeepSea (Part A) configs run the **episode-budgeted** lane (:func:`run_seed`).
    The two lanes share the env/agent factories, the seed derivation, and the CSV schema —
    only the budget axis and the logged metric set differ.

    Also serializes the resolved config to ``<out_dir>/resolved_config.json`` (C13 input),
    so the run's identity fingerprint is committed alongside its logs.

    ``diagnostics=True`` additionally records the §3.3 value-sample substrate to
    ``<out_dir>/diagnostics/``. It is a flag rather than a config field on purpose: it is not
    part of the run's scientific identity, must not enter ``config_sha256``, and — because it
    writes nothing to the CSV — leaves gate C1 exactly as strong as it was. It is a no-op for
    MinAtar runs and for the ε-greedy DDQN reference, neither of which has a sample
    distribution to record.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.write_resolved(out_dir)

    csv_path = out_dir / f"{cfg.run_id}.csv"
    if csv_path.exists():
        csv_path.unlink()  # append-only logger; start fresh for a clean re-run

    step_lane = config_mod.is_minatar(cfg.env)
    diag_dir = (out_dir / "diagnostics") if diagnostics else None
    summaries: list[dict[str, float]] = []
    for seed_index in cfg.seeds:
        ctx = cfg.run_context(seed_index)
        with CSVLogger(csv_path, ctx) as log:
            if step_lane:
                summaries.append(
                    run_seed_steps(cfg, seed_index, log, n_eval_episodes=n_eval_episodes)
                )
            else:
                summaries.append(
                    run_seed(
                        cfg,
                        seed_index,
                        log,
                        n_checkpoints=n_checkpoints,
                        diagnostics_dir=diag_dir,
                    )
                )

    _write_compute_sidecar(out_dir, cfg, summaries)

    if step_lane:
        mean_steps = sum(s["steps_to_first_reward"] for s in summaries) / len(summaries)
        n_censored = int(sum(s["first_reward_censored"] for s in summaries))
        print(
            f"{cfg.run_id}: {len(summaries)} seeds x {int(summaries[0]['total_steps'])} steps, "
            f"mean steps-to-first-reward {mean_steps:.0f} "
            f"({n_censored} censored) → {csv_path}"
        )
    else:
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
    ap.add_argument(
        "--eval-episodes",
        type=int,
        default=DEFAULT_EVAL_EPISODES,
        help="frozen-policy evaluation episodes per checkpoint (MinAtar/Part-B lane only)",
    )
    ap.add_argument(
        "--diagnostics",
        action="store_true",
        help=(
            "record the §3.3 value-sample substrate to <out>/diagnostics/ (DeepSea only; "
            "no-op for the DDQN reference). Does not change the run or the metrics CSV."
        ),
    )
    args = ap.parse_args(argv)

    cfg = config_mod.load_config(args.config)
    train(
        cfg,
        args.out,
        n_checkpoints=args.checkpoints,
        n_eval_episodes=args.eval_episodes,
        diagnostics=args.diagnostics,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
