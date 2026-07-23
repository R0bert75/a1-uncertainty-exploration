"""Tests for the Part-A training loop (:mod:`src.trainer`).

These assert the loop's *contracts* — not the science — since the trainer is execution
infrastructure: schema-correct rows, a shared cross-seed step grid, a monotone discovery
indicator, resolved-config serialization (C13), and bit-for-bit reproducibility (C1).
"""

from __future__ import annotations

import csv
import filecmp
import shutil
from pathlib import Path

import pytest

from src import config as config_mod
from src.trainer import _checkpoint_episodes, train
from src.utils.conventions import BASE_FIELDS


def _cfg(method: str = "ddqn_egreedy", episodes: int = 40, size: int = 6):
    """A minimal schema-valid development config that runs in well under a second."""
    data = {
        "run_id": f"test_{method}",
        "role": "development",
        "part": "A",
        "method": method,
        "env": "deep_sea",
        "master_seed": 0,
        "use_rule": "episodic",
        "prior": "off",
        "K": 5,
        "arm": "episodic|off|K5",
        "backbone": {
            "lr": 5e-4,
            "batch_size": 16,
            "gamma": 0.99,
            "min_buffer": 40,
            "target_update_period": 40,
            "buffer_capacity": 4000,
        },
        "factor_specific": {
            "eps_schedule": {"eps_start": 1.0, "eps_end": 0.05, "eps_decay_steps": 200}
        },
        "env_budget": {"deep_sea_size": size, "episodes": episodes},
        "seeds": [0, 1],
    }
    return config_mod.resolve_config(data)


def _rows(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------- #
# Checkpoint scheduling
# --------------------------------------------------------------------------- #
def test_checkpoints_include_final_episode():
    marks = _checkpoint_episodes(n_episodes=100, n_checkpoints=10)
    assert marks[-1] == 100  # end-of-budget indicator is always captured
    assert marks == sorted(marks)
    assert all(1 <= m <= 100 for m in marks)


def test_checkpoints_clamped_when_more_than_episodes():
    marks = _checkpoint_episodes(n_episodes=3, n_checkpoints=20)
    assert marks[-1] == 3
    assert len(marks) == len(set(marks))  # de-duplicated
    assert marks == [1, 2, 3]


def test_checkpoints_rejects_zero_episodes():
    with pytest.raises(ValueError):
        _checkpoint_episodes(0, 5)


# --------------------------------------------------------------------------- #
# End-to-end loop contracts
# --------------------------------------------------------------------------- #
def test_train_writes_schema_correct_rows(tmp_path):
    cfg = _cfg()
    path = train(cfg, tmp_path, n_checkpoints=5)
    assert path.exists()
    rows = _rows(path)
    assert rows, "trainer produced no rows"
    # Header is exactly the frozen schema (gate C2).
    assert list(rows[0].keys()) == list(BASE_FIELDS)
    metrics = {r["metric"] for r in rows}
    assert metrics == {"discovery_prob", "episode_return"}
    # config_sha256 ties every row to the committed resolved config (C13).
    assert all(r["config_sha256"] == cfg.config_sha256 for r in rows)


def test_resolved_config_serialized(tmp_path):
    train(_cfg(), tmp_path, n_checkpoints=4)
    assert (tmp_path / "resolved_config.json").exists()


def test_step_grid_identical_across_seeds(tmp_path):
    # DeepSea episodes are exactly `size` steps long, so cumulative step at each
    # checkpoint must match across seeds — the shared grid make_figures groups on.
    path = train(_cfg(size=6, episodes=40), tmp_path, n_checkpoints=5)
    rows = _rows(path)
    grids = {}
    for r in rows:
        if r["metric"] == "discovery_prob":
            grids.setdefault(r["seed"], []).append(int(r["step"]))
    seeds = list(grids)
    assert len(seeds) == 2
    assert grids[seeds[0]] == grids[seeds[1]]
    # Every checkpoint step is a multiple of the episode length (size).
    assert all(s % 6 == 0 for s in grids[seeds[0]])


def test_discovery_indicator_is_monotone(tmp_path):
    path = train(_cfg(), tmp_path, n_checkpoints=6)
    rows = _rows(path)
    for seed in {r["seed"] for r in rows}:
        curve = [
            float(r["value"])
            for r in rows
            if r["metric"] == "discovery_prob" and r["seed"] == seed
        ]
        pairs = zip(curve, curve[1:], strict=False)  # offset pairs; length mismatch is intended
        assert all(b >= a for a, b in pairs), f"seed {seed} not monotone"
        assert all(v in (0.0, 1.0) for v in curve), "discovery is a 0/1 indicator"


def test_train_is_bit_for_bit_reproducible(tmp_path):
    # Gate C1 through the full loop: same (config, seed) -> identical CSV.
    cfg = _cfg()
    first = train(cfg, tmp_path, n_checkpoints=5)
    snapshot = shutil.copy(first, tmp_path / "first.csv")
    second = train(cfg, tmp_path, n_checkpoints=5)
    assert filecmp.cmp(snapshot, second, shallow=False)


def test_bdqn_path_runs(tmp_path):
    # The ensemble path exercises on_episode_start (episodic head resampling).
    path = train(_cfg(method="bdqn"), tmp_path, n_checkpoints=4)
    assert _rows(path)
