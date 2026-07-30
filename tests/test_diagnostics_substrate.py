"""Tests for the value-sample recording substrate (§8 item 5)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.diagnostics.substrate import (
    SubstrateSpec,
    ValueSampleRecord,
    ValueSampleWriter,
    record_checkpoint,
)


def _spec(S=6, M=4, A=3, kind="ensemble_heads", pid="probe-v1"):
    return SubstrateSpec(
        n_probe_states=S, n_samples=M, n_actions=A, sampler_kind=kind, probe_set_id=pid
    )


class _FakeSampler:
    """Deterministic sampler: value = state index + sample index / 10 + action / 100."""

    def __init__(self, S, M, A):
        self.shape = (S, M, A)
        self.calls = 0

    def value_samples(self, probe_states):
        self.calls += 1
        S, M, A = self.shape
        s = np.arange(S).reshape(S, 1, 1)
        m = np.arange(M).reshape(1, M, 1)
        a = np.arange(A).reshape(1, 1, A)
        return (s + m / 10.0 + a / 100.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Spec validation
# --------------------------------------------------------------------------- #
def test_spec_shape():
    assert _spec(S=8, M=20, A=6).shape == (8, 20, 6)


@pytest.mark.parametrize("field", ["n_probe_states", "n_samples", "n_actions"])
def test_spec_rejects_nonpositive_dims(field):
    kwargs = dict(
        n_probe_states=4, n_samples=4, n_actions=4,
        sampler_kind="ensemble_heads", probe_set_id="p",
    )
    kwargs[field] = 0
    with pytest.raises(ValueError, match=f"{field} must be >= 1"):
        SubstrateSpec(**kwargs)


def test_spec_rejects_unknown_sampler_kind():
    with pytest.raises(ValueError, match="sampler_kind must be one of"):
        _spec(kind="magic")


def test_spec_accepts_noisynet_m30():
    """NoisyNet uses M = 30 i.i.d. draws at measurement only (§3.3 notation)."""
    spec = _spec(S=16, M=30, A=6, kind="noisynet_draws")
    assert spec.shape == (16, 30, 6)


def test_configurable_probe_set_size():
    """|S| is a parameter: freeze item 7's construction rule is not yet written down."""
    for S in (1, 8, 64, 512):
        assert _spec(S=S).shape[0] == S


# --------------------------------------------------------------------------- #
# record_checkpoint
# --------------------------------------------------------------------------- #
def test_record_checkpoint_shapes_and_dtype():
    spec = _spec()
    sampler = _FakeSampler(*spec.shape)
    probes = np.zeros((spec.n_probe_states, 4), dtype=np.float32)
    rec = record_checkpoint(sampler, probes, spec, step=1000)
    assert rec.samples.shape == spec.shape
    assert rec.samples.dtype == np.float32
    assert rec.step == 1000
    assert sampler.calls == 1


def test_record_checkpoint_rejects_probe_set_size_mismatch():
    spec = _spec(S=6)
    sampler = _FakeSampler(*spec.shape)
    probes = np.zeros((5, 4), dtype=np.float32)  # 5 != 6
    with pytest.raises(ValueError, match=r"\|S\| = 6"):
        record_checkpoint(sampler, probes, spec, step=0)


def test_record_rejects_wrong_shape():
    spec = _spec(S=6, M=4, A=3)
    bad = np.zeros((6, 4, 99), dtype=np.float32)
    with pytest.raises(ValueError, match="does not match spec"):
        ValueSampleRecord(step=0, samples=bad, spec=spec)


def test_record_rejects_non_float32():
    spec = _spec()
    bad = np.zeros(spec.shape, dtype=np.float64)
    with pytest.raises(ValueError, match="must be float32"):
        ValueSampleRecord(step=0, samples=bad, spec=spec)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_record_rejects_non_finite_samples(bad):
    spec = _spec()
    arr = np.zeros(spec.shape, dtype=np.float32)
    arr[0, 0, 0] = bad
    with pytest.raises(ValueError, match="non-finite"):
        ValueSampleRecord(step=0, samples=arr, spec=spec)


def test_record_rejects_visitation_shape_mismatch():
    spec = _spec(S=6)
    arr = np.zeros(spec.shape, dtype=np.float32)
    with pytest.raises(ValueError, match="visitation shape"):
        ValueSampleRecord(step=0, samples=arr, spec=spec, visitation=np.zeros(5))


# --------------------------------------------------------------------------- #
# Sufficiency: the three diagnostics that marginals cannot serve
# --------------------------------------------------------------------------- #
def test_substrate_supports_action_gap_std_of_difference():
    """§3.3 #2 needs std_m of a DIFFERENCE, not a function of per-action stds."""
    spec = _spec(S=1, M=4, A=2)
    # Perfectly correlated samples: each action's std is large, the difference's std is 0.
    arr = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    rec = ValueSampleRecord(step=0, samples=arr, spec=spec)
    per_action_std = rec.samples[0].std(axis=0)
    diff_std = (rec.samples[0, :, 0] - rec.samples[0, :, 1]).std()
    assert per_action_std.min() > 2.0
    assert diff_std == pytest.approx(0.0)


def test_substrate_supports_modal_fraction():
    """§3.3 #3 needs per-sample argmaxes to form d(s) = 1 - modal fraction."""
    spec = _spec(S=1, M=4, A=3)
    arr = np.zeros(spec.shape, dtype=np.float32)
    arr[0, 0] = [3.0, 1.0, 0.0]   # argmax 0
    arr[0, 1] = [3.0, 1.0, 0.0]   # argmax 0
    arr[0, 2] = [0.0, 5.0, 0.0]   # argmax 1
    arr[0, 3] = [0.0, 0.0, 9.0]   # argmax 2
    rec = ValueSampleRecord(step=0, samples=arr, spec=spec)
    argmaxes = rec.samples[0].argmax(axis=1)
    modal_fraction = np.bincount(argmaxes, minlength=3).max() / len(argmaxes)
    assert modal_fraction == pytest.approx(0.5)


def test_substrate_supports_empirical_quantiles():
    """§3.3 #7 needs empirical quantiles over m with method='linear'."""
    spec = _spec(S=1, M=5, A=1)
    arr = np.array([[[0.0], [1.0], [2.0], [3.0], [4.0]]], dtype=np.float32)
    rec = ValueSampleRecord(step=0, samples=arr, spec=spec)
    lo, hi = np.quantile(rec.samples[0, :, 0], [0.1, 0.9], method="linear")
    assert lo == pytest.approx(0.4)
    assert hi == pytest.approx(3.6)


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #
def test_writer_roundtrip(tmp_path):
    spec = _spec()
    sampler = _FakeSampler(*spec.shape)
    probes = np.zeros((spec.n_probe_states, 4), dtype=np.float32)
    writer = ValueSampleWriter(spec)
    for step in (0, 500, 1000):
        writer.append(
            record_checkpoint(
                sampler, probes, spec, step=step,
                visitation=np.arange(spec.n_probe_states, dtype=np.int64),
            )
        )
    path = writer.write(tmp_path, "run_abc")

    loaded = np.load(path)
    assert loaded["samples"].shape == (3, *spec.shape)
    assert loaded["steps"].tolist() == [0, 500, 1000]
    assert loaded["visitation"].shape == (3, spec.n_probe_states)

    sidecar = json.loads((tmp_path / "run_abc.value_samples.json").read_text())
    assert sidecar["spec"]["n_probe_states"] == spec.n_probe_states
    assert sidecar["n_checkpoints"] == 3
    assert sidecar["has_visitation"] is True


def test_writer_rejects_spec_mismatch():
    writer = ValueSampleWriter(_spec(S=6))
    other = _spec(S=8)
    rec = ValueSampleRecord(step=0, samples=np.zeros(other.shape, dtype=np.float32), spec=other)
    with pytest.raises(ValueError, match="record spec differs"):
        writer.append(rec)


def test_writer_requires_increasing_steps():
    spec = _spec()
    writer = ValueSampleWriter(spec)
    arr = np.zeros(spec.shape, dtype=np.float32)
    writer.append(ValueSampleRecord(step=100, samples=arr, spec=spec))
    with pytest.raises(ValueError, match="strictly increasing"):
        writer.append(ValueSampleRecord(step=100, samples=arr, spec=spec))


def test_writer_refuses_empty_write(tmp_path):
    with pytest.raises(ValueError, match="no checkpoint records"):
        ValueSampleWriter(_spec()).write(tmp_path, "empty_run")


def test_writer_omits_visitation_when_partial(tmp_path):
    """Mixed visitation would silently misalign diagnostic #5; the key is dropped instead."""
    spec = _spec()
    writer = ValueSampleWriter(spec)
    arr = np.zeros(spec.shape, dtype=np.float32)
    writer.append(ValueSampleRecord(step=0, samples=arr, spec=spec, visitation=np.zeros(6)))
    writer.append(ValueSampleRecord(step=1, samples=arr, spec=spec))  # no visitation
    path = writer.write(tmp_path, "partial")
    assert "visitation" not in np.load(path)
