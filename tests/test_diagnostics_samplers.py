"""Tests for the ValueSampler adapters and the DeepSea probe set.

The substrate shipped with a ``ValueSampler`` *protocol* and no implementation, so the §3.3
battery had no input tensor. These tests pin the two adapters that supply it.

The load-bearing test here is :func:`test_measurement_does_not_perturb_training` — gate C1
requires that a run with diagnostics enabled and the same run without produce identical
training trajectories. If measurement ever draws from an operational stream, every
diagnostic-carrying run in the study is silently a different experiment from its
diagnostic-free counterpart.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.bdqn import BDQNAgent, BDQNConfig
from src.diagnostics.samplers import (
    NOISYNET_DIAG_SAMPLES,
    EnsembleValueSampler,
    NoisyNetValueSampler,
    deep_sea_probe_set_size,
    deep_sea_probe_states,
    disagreement_summary,
    make_value_sampler,
)
from src.diagnostics.substrate import ValueSampler, record_checkpoint
from src.noisynet import DIAG_SAMPLES, NoisyNetAgent, NoisyNetConfig

SIZE = 6
N_ACTIONS = 2


def _bdqn(k=5, use_rule="episodic"):
    cfg = BDQNConfig(
        obs_dim=SIZE * SIZE, n_actions=N_ACTIONS, K=k, use_rule=use_rule,
        hidden_sizes=(16,), min_buffer=16, batch_size=8,
    )
    return BDQNAgent(cfg, master_seed=0, cell_id="test|off|K", seed_index=0)


def _noisynet():
    cfg = NoisyNetConfig(
        obs_dim=SIZE * SIZE, n_actions=N_ACTIONS, hidden_sizes=(16,),
        min_buffer=16, batch_size=8, sigma0=0.5,
    )
    return NoisyNetAgent(cfg, master_seed=0, cell_id="test|off|K", seed_index=0)


# --------------------------------------------------------------------------- #
# probe set
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n", [1, 2, 5, 10, 20, 30])
def test_probe_set_is_the_exhaustive_reachable_set(n):
    """Freeze item 7 (approved): exhaustive reachable set, |S| = N(N+1)/2, no cap."""
    obs, idx = deep_sea_probe_states(n)
    assert len(obs) == n * (n + 1) // 2 == deep_sea_probe_set_size(n)
    # reachability: column never exceeds row
    assert np.all(idx[:, 1] <= idx[:, 0])
    # exhaustive: every reachable cell appears exactly once
    assert {tuple(c) for c in idx} == {(r, c) for r in range(n) for c in range(n) if c <= r}


def test_probe_default_encoding_is_the_agent_facing_flat_one():
    """run_seed flattens DeepSea obs before select_action; probes must match, or the
    battery measures the network off its training input distribution."""
    flat, idx = deep_sea_probe_states(SIZE)
    grid, _ = deep_sea_probe_states(SIZE, flatten=False)
    assert flat.shape == (len(idx), SIZE * SIZE)
    assert grid.shape == (len(idx), SIZE, SIZE)
    assert np.array_equal(flat, grid.reshape(len(idx), -1))


def test_probe_states_are_one_hot_at_their_index():
    obs, idx = deep_sea_probe_states(SIZE, flatten=False)
    for o, (r, c) in zip(obs, idx, strict=True):
        assert o[r, c] == 1.0
        assert o.sum() == 1.0
    assert obs.dtype == np.float32


def test_probe_ordering_is_row_major_and_stable():
    """Ordering is contractual: the substrate stores tensors positionally."""
    _, a = deep_sea_probe_states(SIZE)
    _, b = deep_sea_probe_states(SIZE)
    assert np.array_equal(a, b)
    assert list(map(tuple, a)) == sorted(map(tuple, a))


def test_probe_set_rejects_nonpositive_size():
    with pytest.raises(ValueError):
        deep_sea_probe_states(0)


# --------------------------------------------------------------------------- #
# adapters satisfy the protocol and the shape contract
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("k", [5, 10, 20])
def test_ensemble_sampler_shape_is_S_by_K_by_A(k):
    agent = _bdqn(k=k)
    obs, _ = deep_sea_probe_states(SIZE)
    out = EnsembleValueSampler(agent).value_samples(obs)
    assert out.shape == (deep_sea_probe_set_size(SIZE), k, N_ACTIONS)
    assert out.dtype == np.float32


def test_noisynet_sampler_shape_is_S_by_M_by_A():
    obs, _ = deep_sea_probe_states(SIZE)
    out = NoisyNetValueSampler(_noisynet()).value_samples(obs)
    assert out.shape == (deep_sea_probe_set_size(SIZE), NOISYNET_DIAG_SAMPLES, N_ACTIONS)


def test_noisynet_m_matches_the_frozen_measurement_convention():
    """Freeze item 14 pins M = 30 i.i.d. NoisyNet draws at measurement only."""
    assert NOISYNET_DIAG_SAMPLES == DIAG_SAMPLES == 30


def test_adapters_satisfy_the_runtime_protocol():
    assert isinstance(EnsembleValueSampler(_bdqn()), ValueSampler)
    assert isinstance(NoisyNetValueSampler(_noisynet()), ValueSampler)


def test_ensemble_sampling_is_exhaustive_and_deterministic():
    """M = K heads is a full enumeration, not a draw: repeat calls must be identical."""
    sampler = EnsembleValueSampler(_bdqn())
    obs, _ = deep_sea_probe_states(SIZE)
    assert np.array_equal(sampler.value_samples(obs), sampler.value_samples(obs))


def test_noisynet_sampling_is_stochastic_across_draws():
    """M i.i.d. draws must actually differ, or σ collapses to 0 and the battery is vacuous."""
    out = NoisyNetValueSampler(_noisynet(), m=8).value_samples(deep_sea_probe_states(SIZE)[0])
    spread = out.std(axis=1)
    assert spread.mean() > 0.0


def test_heads_are_distinguishable():
    """Independent head init: σ over heads must be > 0 or C-K measures nothing."""
    out = EnsembleValueSampler(_bdqn(k=10)).value_samples(deep_sea_probe_states(SIZE)[0])
    assert out.std(axis=1).mean() > 0.0


# --------------------------------------------------------------------------- #
# C1: measurement must not perturb training
# --------------------------------------------------------------------------- #

def test_measurement_does_not_perturb_training():
    """Gate C1. Two identically-seeded NoisyNet agents; one is measured 5 times between
    action selections, the other is not. Their action sequences must be identical.

    NoisyNet is the sharp case: its acting policy consumes noise draws, so a measurement
    that reset noise from the operational generator would shift every subsequent action.
    """
    obs, _ = deep_sea_probe_states(SIZE)
    probe = obs[:4]

    plain, measured = _noisynet(), _noisynet()
    sampler = NoisyNetValueSampler(measured, m=5)

    a_plain, a_measured = [], []
    for step in range(12):
        a_plain.append(plain.select_action(obs[step % len(obs)], step))
        sampler.value_samples(probe)  # measurement interleaved
        a_measured.append(measured.select_action(obs[step % len(obs)], step))

    assert a_plain == a_measured


def test_ensemble_measurement_draws_no_randomness():
    """The ensemble adapter holds no generator at all — sampling K heads is exhaustive."""
    agent = _bdqn()
    sampler = EnsembleValueSampler(agent)
    assert not any(
        isinstance(v, (np.random.Generator, torch.Generator))
        for v in vars(sampler).values()
    )
    obs, _ = deep_sea_probe_states(SIZE)
    before = agent._active_head
    sampler.value_samples(obs)
    assert agent._active_head == before


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "method,cls",
    [("bdqn", EnsembleValueSampler), ("rp_bdqn", EnsembleValueSampler),
     ("noisynet", NoisyNetValueSampler)],
)
def test_factory_dispatch(method, cls):
    agent = _noisynet() if method == "noisynet" else _bdqn()
    assert isinstance(make_value_sampler(agent, method), cls)


def test_factory_returns_none_for_the_point_estimator():
    """ddqn_egreedy has one Q-function and no sample distribution: sigma is undefined."""
    assert make_value_sampler(_bdqn(), "ddqn_egreedy") is None


# --------------------------------------------------------------------------- #
# integration with the substrate
# --------------------------------------------------------------------------- #

def test_record_checkpoint_accepts_adapter_output():
    agent = _bdqn(k=5)
    sampler = EnsembleValueSampler(agent)
    obs, _ = deep_sea_probe_states(SIZE)
    spec = sampler.spec(
        n_probe_states=len(obs), n_actions=N_ACTIONS, probe_set_id=f"deep_sea_exhaustive_N{SIZE}"
    )
    rec = record_checkpoint(sampler, obs, spec, step=100)
    assert rec.samples.shape == spec.shape
    assert spec.sampler_kind == "ensemble_heads"


def test_spec_records_the_method_specific_M():
    ens = EnsembleValueSampler(_bdqn(k=20)).spec(
        n_probe_states=3, n_actions=2, probe_set_id="p")
    noisy = NoisyNetValueSampler(_noisynet()).spec(
        n_probe_states=3, n_actions=2, probe_set_id="p")
    assert ens.n_samples == 20
    assert noisy.n_samples == 30
    assert noisy.sampler_kind == "noisynet_draws"


# --------------------------------------------------------------------------- #
# disagreement summary
# --------------------------------------------------------------------------- #

def test_disagreement_uses_population_std():
    """ddof=0: the M heads ARE the population of samples, not a draw from a larger one."""
    s = np.zeros((1, 4, 1), dtype=np.float32)
    s[0, :, 0] = [1.0, 2.0, 3.0, 4.0]
    got = disagreement_summary(s)["mean_sigma"]
    assert got == pytest.approx(np.std([1.0, 2.0, 3.0, 4.0], ddof=0))
    assert got != pytest.approx(np.std([1.0, 2.0, 3.0, 4.0], ddof=1))


def test_disagreement_zero_when_all_samples_agree():
    s = np.ones((5, 3, 2), dtype=np.float32)
    d = disagreement_summary(s)
    assert d["mean_sigma"] == 0.0
    assert d["max_sigma"] == 0.0
    assert d["mean_sigma_greedy"] == 0.0


def test_mean_sigma_greedy_picks_the_mean_greedy_action():
    # state 0: action 1 has the higher mean but zero spread; action 0 has spread 1.0
    s = np.zeros((1, 2, 2), dtype=np.float32)
    s[0, :, 0] = [0.0, 2.0]   # mean 1.0, sigma 1.0
    s[0, :, 1] = [5.0, 5.0]   # mean 5.0, sigma 0.0  <- greedy
    d = disagreement_summary(s)
    assert d["mean_sigma_greedy"] == pytest.approx(0.0)
    assert d["max_sigma"] == pytest.approx(1.0)


def test_greedy_ties_break_to_lowest_action_index():
    """§3.3 micro-convention: ties by lowest action index."""
    s = np.zeros((1, 2, 2), dtype=np.float32)
    s[0, :, 0] = [0.0, 2.0]   # mean 1.0, sigma 1.0
    s[0, :, 1] = [1.0, 1.0]   # mean 1.0, sigma 0.0  -- tie on mean
    assert disagreement_summary(s)["mean_sigma_greedy"] == pytest.approx(1.0)


def test_disagreement_rejects_point_estimates():
    with pytest.raises(ValueError, match="M >= 2"):
        disagreement_summary(np.ones((3, 1, 2), dtype=np.float32))


def test_disagreement_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"\[S, M, A\]"):
        disagreement_summary(np.ones((3, 2), dtype=np.float32))
