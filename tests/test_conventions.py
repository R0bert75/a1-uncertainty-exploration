"""Smoke tests for the A1 conventions module (gates C1, C2, C13)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.conventions import (  # noqa: E402
    BASE_FIELDS,
    CSVLogger,
    RunContext,
    config_hash,
    seed_everything,
    serialize_resolved_config,
)

# --- C1: determinism ------------------------------------------------------- #

def test_seed_everything_reproducible():
    seed_everything(123)
    a = np.random.rand(5)
    seed_everything(123)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_seed_everything_returns_seed_and_rejects_bad_type():
    assert seed_everything(7) == 7
    with pytest.raises(TypeError):
        seed_everything(1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        seed_everything(True)  # bool is not a valid seed


def test_torch_determinism_optional():
    torch = pytest.importorskip("torch")
    seed_everything(0)
    x = torch.randn(4)
    seed_everything(0)
    y = torch.randn(4)
    assert torch.equal(x, y)


# --- C13: config identity -------------------------------------------------- #

def test_config_hash_stable_and_order_independent():
    h1 = config_hash({"a": 1, "b": {"c": 2, "d": 3}})
    h2 = config_hash({"b": {"d": 3, "c": 2}, "a": 1})
    assert h1 == h2 and len(h1) == 64


def test_config_hash_sensitive_to_value():
    assert config_hash({"K": 10}) != config_hash({"K": 20})


def test_serialize_resolved_config(tmp_path):
    p = serialize_resolved_config({"method": "bdqn", "K": 10}, tmp_path)
    assert p.exists()
    import json
    payload = json.loads(p.read_text())
    assert payload["_config_sha256"] == config_hash({"method": "bdqn", "K": 10})


# --- C2: logging schema + role enforcement --------------------------------- #

def _ctx(**kw):
    base = dict(run_id="r0", role="development", part="A", method="bdqn",
                env="deep_sea", seed=0, config_sha256="deadbeef")
    base.update(kw)
    return RunContext(**base)


def test_csv_logger_writes_frozen_header(tmp_path):
    out = tmp_path / "run.csv"
    with CSVLogger(out, _ctx()) as log:
        log.log(step=100, metric="discovery_prob", value=0.5)
    rows = list(csv.DictReader(out.open()))
    assert list(rows[0].keys())[: len(BASE_FIELDS)] == list(BASE_FIELDS)
    assert rows[0]["role"] == "development"
    assert rows[0]["metric"] == "discovery_prob"
    assert float(rows[0]["value"]) == 0.5


def test_qrdqn_must_be_exploratory():
    # Spec v6.1: QR-DQN rows are exploratory by construction.
    with pytest.raises(ValueError):
        _ctx(method="qrdqn", role="confirmatory")
    ok = _ctx(method="qrdqn", role="exploratory")
    assert ok.role == "exploratory"


def test_invalid_role_and_part_rejected():
    with pytest.raises(ValueError):
        _ctx(role="production")
    with pytest.raises(ValueError):
        _ctx(part="C")


def test_csv_append_preserves_single_header(tmp_path):
    out = tmp_path / "run.csv"
    for seed in range(2):
        with CSVLogger(out, _ctx(seed=seed)) as log:
            log.log(step=0, metric="x", value=float(seed))
    lines = out.read_text().splitlines()
    assert lines[0].startswith("run_id,")
    assert sum(1 for ln in lines if ln.startswith("run_id,")) == 1  # exactly one header
