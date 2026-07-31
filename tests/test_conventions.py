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
    STREAM_NAMES,
    CSVLogger,
    RunContext,
    config_hash,
    deepsea_action_mapping,
    deepsea_mapping_hash,
    derive_cell_seeds,
    derive_numpy_generator,
    derive_seed,
    derive_seed_sequence,
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


# --- C1: cell-specific RNG derivation (spec v6.3 / plan v4.3 freeze item) --- #

def test_derive_seed_is_deterministic_and_platform_stable():
    # Same (master, cell, stream, index) -> same seed, every run, every platform.
    assert derive_seed(0, "episodic|off|K10", "init", 0) == derive_seed(
        0, "episodic|off|K10", "init", 0
    )
    # Pinned regression value guards the exact byte derivation (BLAKE2b, digest_size=16,
    # big-endian, 63-bit mask; canonical 0x1F-separated payload). If this changes, every
    # downstream stream changes — that is a versioned, documented event, never an accident.
    assert derive_seed(0, "episodic|off|K10", "init", 0) == 8011425302454941550


def test_derive_seed_streams_are_independent_across_every_axis():
    base = derive_seed(0, "cellA", "init", 0)
    # different seed_index, cell, stream, and master_seed each give a different stream
    assert derive_seed(0, "cellA", "init", 1) != base           # seed label
    assert derive_seed(0, "cellB", "init", 0) != base           # cell (no cross-cell reuse)
    assert derive_seed(0, "cellA", "replay", 0) != base         # stream name
    assert derive_seed(1, "cellA", "init", 0) != base           # master seed


def test_no_stream_reused_across_cells_bulk():
    # Every (cell, stream, seed) triple across a small grid must map to a unique seed.
    seeds = [
        derive_seed(0, cell, stream, idx)
        for cell in ("episodic|off|K10", "per_step|off|K10", "episodic|on|K10")
        for stream in STREAM_NAMES
        for idx in range(10)
    ]
    assert len(seeds) == len(set(seeds)), "a stream seed collided across cells"


def test_derive_cell_seeds_covers_all_streams():
    d = derive_cell_seeds(0, "episodic|off|K10", 3)
    assert set(d) == set(STREAM_NAMES)
    assert all(isinstance(v, int) and v >= 0 for v in d.values())


def test_derive_numpy_generator_reproducible_and_cell_separated():
    a = derive_numpy_generator(0, "cellA", "action_noise", 0).random(4)
    b = derive_numpy_generator(0, "cellA", "action_noise", 0).random(4)
    c = derive_numpy_generator(0, "cellB", "action_noise", 0).random(4)
    assert np.array_equal(a, b)          # reproducible
    assert not np.array_equal(a, c)      # different cell -> different draw


def test_derive_seed_rejects_unknown_stream_and_bad_types():
    with pytest.raises(ValueError):
        derive_seed(0, "cellA", "not_a_stream", 0)
    with pytest.raises(TypeError):
        derive_seed(1.0, "cellA", "init", 0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        derive_seed(0, "cellA", "init", True)  # bool is not a valid index


def test_stream_registry_is_the_full_nine():
    # Reviewer Fix 1: the stream registry must cover every named stream, in order.
    # Eight through 2026-07-30; ``hparam_search`` added with the approved freeze item 2
    # (random search needs its own stream). Adding a NAME is safe -- names, not positions,
    # key the derivation -- so nothing already derived shifts. The pre-registration's
    # item-1 phrase "8 non-overlapping derived streams" must move to 9 at stage 3.
    assert STREAM_NAMES == (
        "init",
        "env_mapping",
        "replay",
        "action_noise",
        "bootstrap_mask",
        "eval_episodes",
        "probe_set",
        "noisynet_diag",
        "hparam_search",
    )


def test_new_stream_is_independent_and_disturbs_nothing():
    """Adding ``hparam_search`` must not perturb any pre-existing stream.

    The registry comment claims "names, not positions, key the derivation", and the safety of
    adding a stream mid-project rests entirely on that claim. These are the pinned bytes of the
    eight original streams; if a future edit ever makes derivation position-sensitive, this
    fails rather than silently invalidating every seed already drawn.
    """
    # Literal bytes, recorded when the registry held eight names. These are NOT recomputed
    # from the registry -- hardcoding is the whole point, since a computed expectation would
    # move in lockstep with any regression and assert nothing.
    pinned = {
        "init": 8011425302454941550,
        "env_mapping": 176538044932436750,
        "replay": 8159814950184219800,
        "action_noise": 5052484805708001628,
        "bootstrap_mask": 2620863740542659760,
        "eval_episodes": 4132105731905996667,
        "probe_set": 2582891540777874814,
        "noisynet_diag": 8467507595424508135,
    }
    for name, expected in pinned.items():
        assert derive_seed(0, "episodic|off|K10", name, 0) == expected, (
            f"stream {name!r} changed -- every seed already derived is invalidated"
        )

    # And the new stream collides with none of them.
    new = derive_seed(0, "episodic|off|K10", "hparam_search", 0)
    assert new not in set(pinned.values())


def test_derive_seed_sequence_is_reproducible_and_independent():
    # SeedSequence is the canonical numpy entry point (reviewer-pinned BLAKE2b path).
    a = derive_numpy_generator(0, "episodic|off|K10", "replay", 0).random(4)
    b = derive_numpy_generator(0, "episodic|off|K10", "replay", 0).random(4)
    c = derive_numpy_generator(0, "episodic|off|K10", "replay", 1).random(4)
    assert np.array_equal(a, b)          # reproducible across calls
    assert not np.array_equal(a, c)      # different seed_index -> independent stream
    ss = derive_seed_sequence(0, "episodic|off|K10", "replay", 0)
    assert isinstance(ss, np.random.SeedSequence)


def test_deepsea_mapping_is_bound_to_env_mapping_stream():
    # Reviewer Fix 4: Q* is per-run; the mapping is reproducible and per-seed distinct.
    m0 = deepsea_action_mapping(0, "episodic|off|K10", 0, 8)
    m0b = deepsea_action_mapping(0, "episodic|off|K10", 0, 8)
    m1 = deepsea_action_mapping(0, "episodic|off|K10", 1, 8)
    assert np.array_equal(m0, m0b)                 # reproducible from (seed, cell, size)
    assert not np.array_equal(m0, m1)              # different seed -> different mapping
    assert m0.shape == (8,) and m0.dtype == bool


def test_deepsea_mapping_hash_is_stable_and_discriminating():
    m0 = deepsea_action_mapping(0, "episodic|off|K10", 0, 8)
    m0b = deepsea_action_mapping(0, "episodic|off|K10", 0, 8)
    m1 = deepsea_action_mapping(0, "episodic|off|K10", 1, 8)
    assert deepsea_mapping_hash(m0) == deepsea_mapping_hash(m0b)   # same mapping -> same hash
    assert deepsea_mapping_hash(m0) != deepsea_mapping_hash(m1)    # different mapping -> different


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


def test_size_class_validation():
    # bad value rejected
    with pytest.raises(ValueError):
        _ctx(size_class="pilot")
    # confirmatory size_class requires confirmatory role
    with pytest.raises(ValueError):
        _ctx(role="development", size_class="confirmatory")
    ok = _ctx(role="confirmatory", size_class="confirmatory")
    assert ok.size_class == "confirmatory"


def test_csv_append_preserves_single_header(tmp_path):
    out = tmp_path / "run.csv"
    for seed in range(2):
        with CSVLogger(out, _ctx(seed=seed)) as log:
            log.log(step=0, metric="x", value=float(seed))
    lines = out.read_text().splitlines()
    assert lines[0].startswith("run_id,")
    assert sum(1 for ln in lines if ln.startswith("run_id,")) == 1  # exactly one header
