"""Tests for the per-run diagnostics recorder and its trainer wiring.

The load-bearing test here is :func:`test_diagnostics_do_not_change_the_csv`. The recorder's
entire justification for being a command-line flag rather than a config field is that it
cannot affect the run: if enabling diagnostics changed a single byte of the metrics CSV, then
gate C1 ("re-running a ``(config, seed)`` pair reproduces its CSV byte-for-byte") would no
longer distinguish a code regression from a difference in how the run was invoked, and the
flag would have to become part of ``config_sha256``. That test is the guard on the claim, so
it compares whole file bytes rather than parsed rows — a reordering or a float-formatting
change is exactly the kind of drift it exists to catch.

The rest cover the recorder's own contracts: the visitation histogram is a snapshot at each
checkpoint rather than a shared mutable array, unreachable states are counted as misses
instead of aborting a long run, and the factory declines the three combinations that have no
sample distribution to record.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src import config as config_mod
from src import trainer
from src.diagnostics.recorder import RunRecorder, make_recorder, probe_set_id

SIZE = 6


#: Each method's own committed DeepSea example. Deriving a test config by rewriting another
#: method's ``method`` field does not work and should not: the config layer couples
#: ``method`` to ``arm``/``cell_id`` and rejects the mismatch (a NoisyNet arm is not an
#: ensemble-factorial cell). Shrinking an already-valid config of the right method is the
#: only edit that keeps it valid.
_EXAMPLES = {
    "bdqn": "configs/example_bdqn_deepsea_dev.yaml",
    "rp_bdqn": "configs/example_rpbdqn_deepsea_dev.yaml",
    "noisynet": "configs/example_noisynet_deepsea_dev.yaml",
    "ddqn_egreedy": "configs/example_ddqn_deepsea_dev.yaml",
}


def _cfg(tmp_path: Path, method: str, *, size: int = SIZE, episodes: int = 40):
    """A committed DeepSea example for ``method``, shrunk to one seed and a tiny budget."""
    base = yaml.safe_load(Path(_EXAMPLES[method]).read_text())
    base["env_budget"]["deep_sea_size"] = size
    base["env_budget"]["episodes"] = episodes
    base["seeds"] = [0]
    path = tmp_path / f"{method}.yaml"
    path.write_text(yaml.safe_dump(base, sort_keys=False))
    return config_mod.load_config(path)


# --------------------------------------------------------------------------------------
# The C1 invariant: diagnostics must not perturb the run.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["bdqn", "rp_bdqn", "noisynet"])
def test_diagnostics_do_not_change_the_csv(tmp_path, method):
    """Same config, same seed, diagnostics on vs off -> byte-identical metrics CSV.

    This is the guard on the design commitment that lets ``--diagnostics`` stay out of
    ``config_sha256``. It must hold for NoisyNet in particular, whose sampler draws 30 noise
    vectors per checkpoint: those draws come from the measurement stream, so they cannot
    advance the operational one.
    """
    off = tmp_path / "off"
    on = tmp_path / "on"
    cfg_a = _cfg(tmp_path, method)
    trainer.train(cfg_a, off, diagnostics=False)
    trainer.train(cfg_a, on, diagnostics=True)

    csv_off = (off / f"{cfg_a.run_id}.csv").read_bytes()
    csv_on = (on / f"{cfg_a.run_id}.csv").read_bytes()
    assert csv_off == csv_on
    assert len(csv_off) > 0


def test_diagnostics_off_writes_no_diagnostics_dir(tmp_path):
    cfg = _cfg(tmp_path, "bdqn")
    out = tmp_path / "out"
    trainer.train(cfg, out, diagnostics=False)
    assert not (out / "diagnostics").exists()


def test_diagnostics_on_writes_npz_and_sidecars(tmp_path):
    cfg = _cfg(tmp_path, "bdqn")
    out = tmp_path / "out"
    trainer.train(cfg, out, diagnostics=True)
    d = out / "diagnostics"
    key = f"{cfg.run_id}.seed0"
    assert (d / f"{key}.value_samples.npz").exists()
    assert (d / f"{key}.value_samples.json").exists()
    assert (d / f"{key}.diagnostics.json").exists()

    with np.load(d / f"{key}.value_samples.npz") as z:
        samples = z["samples"]  # [T, S, M, A]
        steps = z["steps"]
        visitation = z["visitation"]
    n_probe = SIZE * (SIZE + 1) // 2
    assert samples.shape[1:] == (n_probe, cfg.data["K"], 2)
    assert samples.dtype == np.float32
    assert samples.shape[0] == steps.shape[0] == visitation.shape[0]
    assert np.all(np.diff(steps) > 0), "checkpoint steps must be strictly increasing"


def test_ddqn_reference_records_nothing(tmp_path):
    """A point estimator has no sample distribution: the flag is a no-op, not an error."""
    cfg = _cfg(tmp_path, "ddqn_egreedy")
    out = tmp_path / "out"
    trainer.train(cfg, out, diagnostics=True)
    assert not (out / "diagnostics").exists()


# --------------------------------------------------------------------------------------
# Recorder contracts.
# --------------------------------------------------------------------------------------


class _StubSampler:
    """Returns a constant tensor; lets recorder mechanics be tested without an agent."""

    kind = "ensemble_heads"

    def __init__(self, m=3, fill=1.0):
        self._m = m
        self._fill = fill

    @property
    def n_samples(self):
        return self._m

    def value_samples(self, probe_states):
        s = np.asarray(probe_states).shape[0]
        return np.full((s, self._m, 2), self._fill, dtype=np.float32)

    def spec(self, *, n_probe_states, n_actions, probe_set_id):
        from src.diagnostics.substrate import SubstrateSpec

        return SubstrateSpec(
            n_probe_states=n_probe_states,
            n_samples=self._m,
            n_actions=n_actions,
            sampler_kind=self.kind,
            probe_set_id=probe_set_id,
        )


def _recorder(size=SIZE, m=3):
    return RunRecorder(_StubSampler(m=m), size=size, n_actions=2, run_id="r", seed=0)


def test_visitation_is_snapshotted_per_checkpoint(tmp_path):
    """Each record must hold v(s) as of ITS checkpoint, not a shared mutable array.

    Diagnostic #5 regresses log-sigma on log(1+v) at each checkpoint; if the recorder stored a
    reference instead of a copy, every checkpoint would silently carry the run's FINAL
    visitation and the regression would be computed against the wrong covariate.
    """
    rec = _recorder()
    obs = np.zeros(SIZE * SIZE, dtype=np.float32)
    obs[0] = 1.0  # cell (0, 0)
    rec.observe_state(obs)
    rec.record(step=10)
    for _ in range(5):
        rec.observe_state(obs)
    rec.record(step=20)

    rec.write(tmp_path)
    with np.load(tmp_path / "r.seed0.value_samples.npz") as z:
        v = z["visitation"]
    assert v[0, 0] == 1, "first checkpoint must see 1 visit"
    assert v[1, 0] == 6, "second must see 6 — not the first count, not a shared array"


def test_off_probe_observation_is_counted_not_raised(tmp_path):
    """An unreachable state is a wiring bug, but must not abort a multi-hour run."""
    rec = _recorder()
    bad = np.zeros(SIZE * SIZE, dtype=np.float32)
    bad[SIZE - 1] = 1.0  # cell (0, SIZE-1): col > row, unreachable
    rec.observe_state(bad)
    rec.record(step=1)
    rec.write(tmp_path)
    side = json.loads((tmp_path / "r.seed0.diagnostics.json").read_text())
    assert side["off_probe_observations"] == 1
    assert side["visitation_total"] == 0


def test_record_returns_disagreement_summary():
    rec = _recorder()
    out = rec.record(step=1)
    assert set(out) == {"mean_sigma", "max_sigma", "mean_sigma_greedy"}
    assert out["mean_sigma"] == 0.0, "identical samples have zero dispersion"


def test_summary_rides_the_sidecar_not_the_csv(tmp_path):
    rec = _recorder()
    rec.record(step=1)
    rec.record(step=2)
    rec.write(tmp_path)
    side = json.loads((tmp_path / "r.seed0.diagnostics.json").read_text())
    assert [d["step"] for d in side["disagreement_by_checkpoint"]] == [1, 2]


def test_write_returns_none_when_no_checkpoints(tmp_path):
    assert _recorder().write(tmp_path) is None


def test_spec_pins_probe_set_and_sampler_kind():
    rec = _recorder(m=7)
    assert rec.spec.probe_set_id == probe_set_id(SIZE)
    assert rec.spec.sampler_kind == "ensemble_heads"
    assert rec.spec.n_samples == 7
    assert rec.spec.n_probe_states == SIZE * (SIZE + 1) // 2


def test_probe_set_id_encodes_size():
    assert probe_set_id(10) != probe_set_id(20)
    assert "10" in probe_set_id(10)


# --------------------------------------------------------------------------------------
# Factory declines.
# --------------------------------------------------------------------------------------


def test_make_recorder_declines_minatar(tmp_path):
    """The battery needs Q*, which MinAtar does not have."""
    cfg = config_mod.load_config(Path("configs/example_bdqn_breakout_dev.yaml"))
    agent = config_mod.build_agent(cfg, 0)
    assert make_recorder(agent, cfg, 0) is None


def test_make_recorder_declines_ddqn(tmp_path):
    cfg = _cfg(tmp_path, "ddqn_egreedy")
    agent = config_mod.build_agent(cfg, 0)
    assert make_recorder(agent, cfg, 0) is None


@pytest.mark.parametrize("method", ["bdqn", "rp_bdqn", "noisynet"])
def test_make_recorder_accepts_the_three_uncertainty_methods(tmp_path, method):
    cfg = _cfg(tmp_path, method)
    agent = config_mod.build_agent(cfg, 0)
    rec = make_recorder(agent, cfg, 0)
    assert rec is not None
    assert rec.spec.n_probe_states == SIZE * (SIZE + 1) // 2
