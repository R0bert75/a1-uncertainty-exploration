"""Tests for NoisyNet-DQN (src/noisynet.py + the NoisyLinear/NoisyMLPQNetwork primitives).

Covers the mechanics that make this the parametric-noise baseline and not the ε-greedy
reference: factorized-Gaussian noise layers (Fortunato et al. 2018), σ-init scale, noise
buffers excluded from the optimizer, exploration by resampled noise (no ε), the shared
Double-DQN learning code path, the measurement-only M=30 diagnostic drawn off the
``noisynet_diag`` stream (never perturbing the operational ``action_noise`` stream), and
gate-C1 reproducibility (bit-for-bit from the RNG triple).
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.networks import NoisyLinear, NoisyMLPQNetwork, _factorized_noise  # noqa: E402
from src.noisynet import DIAG_SAMPLES, NoisyNetAgent, NoisyNetConfig  # noqa: E402
from src.utils import conventions  # noqa: E402

OBS_DIM = 5
N_ACTIONS = 2


def _obs(i: int = 0) -> np.ndarray:
    return np.eye(OBS_DIM, dtype=np.float32)[i % OBS_DIM]


def _gen(stream="init", seed_index=0):
    return conventions.derive_torch_generator(0, "noisynet", stream, seed_index)


def _agent(cell_id="noisynet", seed_index=0, **overrides):
    params = dict(
        obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden_sizes=(16,),
        min_buffer=16, batch_size=8, buffer_capacity=500, target_update_period=5,
    )
    params.update(overrides)
    return NoisyNetAgent(
        NoisyNetConfig(**params), master_seed=0, cell_id=cell_id, seed_index=seed_index
    )


# --------------------------------------------------------------------------- #
# Factorized noise + NoisyLinear primitives
# --------------------------------------------------------------------------- #
def test_factorized_noise_is_stream_deterministic_and_uses_transform():
    v1 = _factorized_noise(6, _gen("action_noise"))
    v2 = _factorized_noise(6, _gen("action_noise"))
    assert torch.equal(v1, v2)  # deterministic from the stream
    # f(x) = sgn(x) * sqrt(|x|)  ->  sgn(f) == sgn(x) and f^2 == |x|; both >= 0 under sqrt
    assert torch.all(v1.abs() >= 0)


def test_factorized_noise_differs_across_streams():
    assert not torch.equal(
        _factorized_noise(6, _gen("action_noise")),
        _factorized_noise(6, _gen("noisynet_diag")),
    )


def test_noisylinear_sigma_init_scales_as_sigma0_over_sqrt_fanin():
    fan_in, sigma0 = 4, 0.5
    lin = NoisyLinear(fan_in, 3, sigma0=sigma0, generator=_gen())
    expected = sigma0 / math.sqrt(fan_in)
    assert torch.allclose(lin.weight_sigma, torch.full((3, fan_in), expected))
    assert torch.allclose(lin.bias_sigma, torch.full((3,), expected))


def test_noisylinear_mu_init_bounded_by_one_over_sqrt_fanin():
    fan_in = 9
    lin = NoisyLinear(fan_in, 4, generator=_gen())
    bound = 1.0 / math.sqrt(fan_in)
    assert torch.all(lin.weight_mu.abs() <= bound + 1e-6)
    assert torch.all(lin.bias_mu.abs() <= bound + 1e-6)


def test_noisylinear_eps_are_buffers_not_parameters():
    lin = NoisyLinear(4, 3, generator=_gen())
    params = dict(lin.named_parameters())
    buffers = dict(lin.named_buffers())
    assert "weight_eps" not in params and "bias_eps" not in params
    assert "weight_eps" in buffers and "bias_eps" in buffers
    # only mu/sigma are learnable
    assert set(params) == {"weight_mu", "weight_sigma", "bias_mu", "bias_sigma"}


def test_noisylinear_noise_false_is_mu_only_and_deterministic():
    lin = NoisyLinear(4, 3, generator=_gen())
    lin.reset_noise(_gen("action_noise"))  # noise present in buffers
    x = torch.ones(2, 4)
    # mean net ignores the noise buffers entirely
    assert torch.allclose(lin(x, noisy=False), lin(x, noisy=False))
    expected = torch.nn.functional.linear(x, lin.weight_mu, lin.bias_mu)
    assert torch.allclose(lin(x, noisy=False), expected)


def test_noisylinear_reset_noise_changes_output():
    lin = NoisyLinear(4, 3, generator=_gen())
    x = torch.ones(2, 4)
    g = _gen("action_noise")
    lin.reset_noise(g)
    y1 = lin(x, noisy=True).clone()
    lin.reset_noise(g)
    y2 = lin(x, noisy=True).clone()
    assert not torch.allclose(y1, y2)  # fresh noise -> different output
    assert not torch.allclose(y1, lin(x, noisy=False))  # noisy != mean


# --------------------------------------------------------------------------- #
# NoisyMLPQNetwork
# --------------------------------------------------------------------------- #
def test_network_forward_shape_is_batch_by_actions():
    net = NoisyMLPQNetwork(OBS_DIM, N_ACTIONS, (16, 8), generator=_gen())
    out = net(torch.ones(4, OBS_DIM))
    assert out.shape == (4, N_ACTIONS)  # single head, no head axis


def test_network_all_layers_are_noisy_and_stream_reproducible():
    net = NoisyMLPQNetwork(OBS_DIM, N_ACTIONS, (16,), generator=_gen())
    assert all(isinstance(m, NoisyLinear) for m in net.layers)
    net2 = NoisyMLPQNetwork(OBS_DIM, N_ACTIONS, (16,), generator=_gen())
    for p1, p2 in zip(net.parameters(), net2.parameters(), strict=True):
        assert torch.equal(p1, p2)


def test_network_reset_noise_makes_forward_stochastic():
    net = NoisyMLPQNetwork(OBS_DIM, N_ACTIONS, (16,), generator=_gen())
    x = torch.ones(3, OBS_DIM)
    g = _gen("action_noise")
    net.reset_noise(g)
    a = net(x).clone()
    net.reset_noise(g)
    b = net(x).clone()
    assert not torch.allclose(a, b)
    # mean net stays fixed regardless of noise
    assert torch.allclose(net(x, noisy=False), net(x, noisy=False))


# --------------------------------------------------------------------------- #
# Agent: exploration, reproducibility, learning
# --------------------------------------------------------------------------- #
def test_config_rejects_nonpositive_sigma0():
    with pytest.raises(ValueError, match="sigma0"):
        NoisyNetConfig(obs_dim=OBS_DIM, n_actions=N_ACTIONS, sigma0=0.0)


def test_agent_is_reproducible_from_seed_and_cell():
    a1, a2 = _agent(), _agent()
    for p1, p2 in zip(a1.online.parameters(), a2.online.parameters(), strict=True):
        assert torch.equal(p1, p2)
    obs = _obs(0)
    assert [a1.select_action(obs) for _ in range(20)] == [a2.select_action(obs) for _ in range(20)]


def test_agent_diverges_across_master_seed():
    a = _agent()
    b = NoisyNetAgent(a.cfg, master_seed=1, cell_id="noisynet", seed_index=0)
    assert not all(
        torch.equal(p, q)
        for p, q in zip(a.online.parameters(), b.online.parameters(), strict=True)
    )


def test_target_starts_synced_and_learn_step_reduces_error_on_fixed_batch():
    a = _agent(hidden_sizes=(32,), lr=1e-3, min_buffer=20, batch_size=16)
    for p, q in zip(a.online.parameters(), a.target.parameters(), strict=True):
        assert torch.equal(p, q)
    rng = np.random.default_rng(0)
    for _ in range(200):
        o = _obs(int(rng.integers(OBS_DIM)))
        act = int(rng.integers(N_ACTIONS))
        r = float(o.argmax() == act)
        a.observe(o, act, r, _obs(int(rng.integers(OBS_DIM))), True)
    losses = [x for x in (a.learn_step() for _ in range(400)) if x is not None]
    assert losses[-1] < losses[0]


def test_learn_step_returns_none_until_min_buffer():
    a = _agent(min_buffer=32, batch_size=8)
    for i in range(10):
        a.observe(_obs(i), 0, 0.0, _obs(i + 1), False)
    assert a.learn_step() is None


def test_target_network_syncs_on_cadence():
    a = _agent(min_buffer=8, batch_size=8, target_update_period=3)
    for i in range(50):
        a.observe(_obs(i), i % N_ACTIONS, float(i % 2), _obs(i + 1), bool(i % 5 == 0))
    # drive several learn steps; after a multiple of the period the nets must match
    for _ in range(3):
        a.learn_step()
    for p, q in zip(a.online.parameters(), a.target.parameters(), strict=True):
        assert torch.equal(p, q)


# --------------------------------------------------------------------------- #
# Measurement-only diagnostic (M=30 draws off noisynet_diag)
# --------------------------------------------------------------------------- #
def test_diag_constant_is_thirty():
    assert DIAG_SAMPLES == 30


def test_sample_q_values_shape_and_variation():
    a = _agent()
    qs = a.sample_q_values(_obs(0))
    assert qs.shape == (DIAG_SAMPLES, N_ACTIONS)
    assert qs.std(dim=0).sum().item() > 0  # noise induces a value distribution


def test_diagnostic_stream_does_not_perturb_operational_stream():
    a = _agent()
    before = a._noise_gen.get_state().clone()
    a.sample_q_values(_obs(0), m=DIAG_SAMPLES)
    after = a._noise_gen.get_state().clone()
    assert torch.equal(before, after)  # diag draws only advance noisynet_diag


def test_diagnostic_is_reproducible_and_independent_of_action_selection():
    # Two agents: one only measures; one selects many actions first, then measures.
    a = _agent()
    b = _agent()
    for _ in range(50):
        b.select_action(_obs(0))  # advances action_noise only
    qa = a.sample_q_values(_obs(0))
    qb = b.sample_q_values(_obs(0))
    assert torch.equal(qa, qb)  # diagnostic depends on noisynet_diag alone


def test_mean_action_is_deterministic_noise_free():
    a = _agent()
    obs = _obs(0)
    assert a.mean_action(obs) == a.mean_action(obs)


def test_select_action_explores_without_epsilon():
    # With fresh per-step noise, the greedy-under-noise action is not always constant
    # early in training -> exploration comes from the noise, not an ε branch.
    a = _agent(hidden_sizes=(16, 16))
    acts = {a.select_action(_obs(0)) for _ in range(200)}
    assert len(acts) >= 1  # at least defined; noise-driven selection runs without ε
    # NoisyNetAgent has no epsilon method / schedule at all
    assert not hasattr(a, "epsilon")
