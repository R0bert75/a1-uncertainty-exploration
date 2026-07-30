"""Tests for the step-budgeted (Part B / MinAtar) trainer lane.

Budgets here are deliberately tiny (a few thousand steps) — these test the *plumbing*
(budget axis, checkpoint grid, both reporting axes, determinism), not learning.
"""

from __future__ import annotations

import csv

import pytest

from src.config import resolve_config
from src.trainer import (
    frozen_policy_action,
    run_seed_steps,
    train,
)
from src.utils.conventions import BASE_FIELDS, CSVLogger

pytest.importorskip("torch")


def _cfg(method: str = "ddqn_egreedy", env: str = "breakout", **budget) -> dict:
    prior = "on" if method == "rp_bdqn" else "off"
    k = 10 if method in ("bdqn", "rp_bdqn") else 1
    factor: dict = {"prior_scale": 3.0 if prior == "on" else None}
    if method == "ddqn_egreedy":
        factor["eps_schedule"] = {"eps_start": 1.0, "eps_end": 0.05, "eps_decay_steps": 500}
    elif method == "noisynet":
        factor["sigma0"] = 0.5
    b = {"total_steps": 600, "checkpoint_steps": [300, 600]}
    b.update(budget)
    return {
        "run_id": f"t_steps_{method}_{env}",
        "role": "development",
        "part": "B",
        "method": method,
        "env": env,
        "master_seed": 0,
        "use_rule": "episodic",
        "prior": prior,
        "K": k,
        "arm": "noisynet" if method == "noisynet" else f"episodic|{prior}|K{k}",
        "backbone": {
            "lr": 5e-4,
            "batch_size": 4,
            "gamma": 0.99,
            "hidden_sizes": [32],
            "min_buffer": 16,
            "target_update_period": 100,
        },
        "factor_specific": factor,
        "env_budget": b,
        "seeds": [0],
    }


def _rows(path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------- #
# Budget axis
# --------------------------------------------------------------------------- #
def test_run_seed_steps_stops_exactly_at_the_step_budget(tmp_path) -> None:
    cfg = resolve_config(_cfg())
    with CSVLogger(tmp_path / "r.csv", cfg.run_context(0)) as log:
        summary = run_seed_steps(cfg, 0, log, n_eval_episodes=1)
    assert summary["total_steps"] == 600.0


def test_checkpoints_land_on_the_configured_step_grid(tmp_path) -> None:
    cfg = resolve_config(_cfg(total_steps=900, checkpoint_steps=[300, 600, 900]))
    with CSVLogger(tmp_path / "r.csv", cfg.run_context(0)) as log:
        run_seed_steps(cfg, 0, log, n_eval_episodes=1)
    rows = _rows(tmp_path / "r.csv")
    online = [r for r in rows if r["metric"] == "episode_return" and r["axis"] == "online"]
    assert [int(r["step"]) for r in online] == [300, 600, 900]
    assert [int(r["checkpoint"]) for r in online] == [0, 1, 2]


def test_first_checkpoint_is_marked_t0(tmp_path) -> None:
    cfg = resolve_config(_cfg())
    with CSVLogger(tmp_path / "r.csv", cfg.run_context(0)) as log:
        run_seed_steps(cfg, 0, log, n_eval_episodes=1)
    rows = [r for r in _rows(tmp_path / "r.csv") if r["metric"] == "episode_return"]
    t0 = [r for r in rows if r["is_t0"] == "True"]
    assert t0 and all(int(r["checkpoint"]) == 0 for r in t0)


# --------------------------------------------------------------------------- #
# Both reporting axes
# --------------------------------------------------------------------------- #
def test_both_reporting_axes_are_logged_at_every_checkpoint(tmp_path) -> None:
    """Spec §5: online (primary) + frozen_policy (secondary) — dual axes are never cut."""
    cfg = resolve_config(_cfg())
    with CSVLogger(tmp_path / "r.csv", cfg.run_context(0)) as log:
        run_seed_steps(cfg, 0, log, n_eval_episodes=1)
    rows = [r for r in _rows(tmp_path / "r.csv") if r["metric"] == "episode_return"]
    for ck in ("0", "1"):
        axes = {r["axis"] for r in rows if r["checkpoint"] == ck}
        assert axes == {"online", "frozen_policy"}


def test_steps_to_first_reward_is_always_logged(tmp_path) -> None:
    """The exploration proxy is never missing — censored at the budget if never seen."""
    cfg = resolve_config(_cfg())
    with CSVLogger(tmp_path / "r.csv", cfg.run_context(0)) as log:
        summary = run_seed_steps(cfg, 0, log, n_eval_episodes=1)
    rows = [r for r in _rows(tmp_path / "r.csv") if r["metric"] == "steps_to_first_reward"]
    assert len(rows) == 1
    v = float(rows[0]["value"])
    assert 0.0 < v <= 600.0
    if summary["first_reward_censored"]:
        assert v == 600.0


def test_csv_header_stays_frozen(tmp_path) -> None:
    """Gate C2: the step lane must not widen the CSV schema."""
    cfg = resolve_config(_cfg())
    with CSVLogger(tmp_path / "r.csv", cfg.run_context(0)) as log:
        run_seed_steps(cfg, 0, log, n_eval_episodes=1)
    with open(tmp_path / "r.csv", newline="") as fh:
        header = next(csv.reader(fh))
    assert tuple(header) == tuple(BASE_FIELDS)


def test_episodes_in_window_travels_as_a_metric_row(tmp_path) -> None:
    cfg = resolve_config(_cfg())
    with CSVLogger(tmp_path / "r.csv", cfg.run_context(0)) as log:
        run_seed_steps(cfg, 0, log, n_eval_episodes=1)
    rows = [r for r in _rows(tmp_path / "r.csv") if r["metric"] == "episodes_in_window"]
    assert len(rows) == 2
    assert all(float(r["value"]) >= 0.0 for r in rows)


# --------------------------------------------------------------------------- #
# Frozen-policy extraction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "method,expect_mean_action",
    [("ddqn_egreedy", False), ("noisynet", True), ("bdqn", True), ("rp_bdqn", True)],
)
def test_frozen_policy_extraction_dispatch(method: str, expect_mean_action: bool) -> None:
    """Each method exposes exactly the pre-registered extraction entry point (spec §5)."""
    from src.config import build_agent, build_env

    cfg = resolve_config(_cfg(method))
    agent = build_agent(cfg, 0)
    env = build_env(cfg, 0)
    assert hasattr(agent, "mean_action") is expect_mean_action
    obs, _ = env.reset()
    a = frozen_policy_action(agent, obs)
    assert 0 <= a < 6


def test_frozen_policy_action_is_deterministic() -> None:
    """The extracted policy must be deterministic and draw no stream randomness (C1)."""
    from src.config import build_agent, build_env

    for method in ("ddqn_egreedy", "noisynet", "bdqn", "rp_bdqn"):
        cfg = resolve_config(_cfg(method))
        agent = build_agent(cfg, 0)
        env = build_env(cfg, 0)
        obs, _ = env.reset()
        picks = {frozen_policy_action(agent, obs) for _ in range(8)}
        assert len(picks) == 1, f"{method} frozen-policy action is not deterministic"


# --------------------------------------------------------------------------- #
# Determinism (gate C1) and the four methods end-to-end
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["ddqn_egreedy", "noisynet", "bdqn", "rp_bdqn"])
def test_every_method_runs_the_step_lane(method: str, tmp_path) -> None:
    cfg = resolve_config(_cfg(method))
    path = train(cfg, tmp_path, n_eval_episodes=1)
    rows = _rows(path)
    assert rows
    assert {r["method"] for r in rows} == {method}
    assert {r["env"] for r in rows} == {"breakout"}
    assert {r["part"] for r in rows} == {"B"}


def test_rerun_reproduces_the_csv_bit_for_bit(tmp_path) -> None:
    """Gate C1: a (config, seed) re-run is byte-identical."""
    cfg = resolve_config(_cfg())
    a = (tmp_path / "a").resolve()
    b = (tmp_path / "b").resolve()
    pa = train(cfg, a, n_eval_episodes=1)
    pb = train(cfg, b, n_eval_episodes=1)
    assert pa.read_bytes() == pb.read_bytes()


def test_different_seed_index_diverges(tmp_path) -> None:
    d = _cfg()
    d["seeds"] = [0]
    cfg0 = resolve_config(d)
    d2 = _cfg()
    d2["seeds"] = [1]
    cfg1 = resolve_config(d2)
    p0 = train(cfg0, tmp_path / "s0", n_eval_episodes=1)
    p1 = train(cfg1, tmp_path / "s1", n_eval_episodes=1)
    assert p0.read_bytes() != p1.read_bytes()


def test_deepsea_config_still_takes_the_episode_lane(tmp_path) -> None:
    """Part A must be routed to the episode-budgeted lane, unchanged."""
    d = _cfg()
    d.update(
        part="A",
        env="deep_sea",
        run_id="t_steps_deepsea",
        env_budget={"deep_sea_size": 5, "episodes": 20},
    )
    cfg = resolve_config(d)
    path = train(cfg, tmp_path)
    metrics = {r["metric"] for r in _rows(path)}
    assert "discovery_prob" in metrics  # the Part-A primary outcome
    assert "steps_to_first_reward" not in metrics  # a Part-B-only proxy


# --------------------------------------------------------------------------- #
# Compute reporting (spec §8 item 4 trigger input; §6 v1.0 "compute reported")
# --------------------------------------------------------------------------- #
def test_compute_sidecar_is_written_and_excluded_from_the_csv(tmp_path) -> None:
    """Wall-clock is reported in a sidecar, never in the metrics CSV — the CSV must stay
    byte-reproducible (gate C1) and wall-clock is machine-dependent."""
    import json

    cfg = resolve_config(_cfg())
    csv_path = train(cfg, tmp_path, n_eval_episodes=1)
    sidecar = tmp_path / f"{cfg.run_id}.compute.json"
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert payload["run_id"] == cfg.run_id
    assert payload["method"] == "ddqn_egreedy"
    assert set(payload["per_seed_wall_clock_s"]) == {"0"}
    assert payload["total_wall_clock_s"] > 0.0
    assert payload["total_env_steps"] == 600
    assert "wall_clock_s" not in {r["metric"] for r in _rows(csv_path)}
