"""Tests for the MinAtar conv backbone: contract parity with the MLP path, determinism, noise."""

from __future__ import annotations

import math

import pytest
import torch

from src.minatar_env import MINATAR_CHANNELS
from src.networks import (
    MINATAR_CONV_CHANNELS,
    MINATAR_FC_WIDTH,
    MinAtarConvQNetwork,
    MLPQNetwork,
    NoisyMinAtarConvQNetwork,
)
from src.utils import conventions

CELL = "minatar|breakout|ddqn"
OBS = (4, 10, 10)  # breakout
N_ACT = 6


def _gen(seed_index: int = 0, master_seed: int = 0) -> torch.Generator:
    return conventions.derive_torch_generator(master_seed, CELL, "init", seed_index)


def _batch(n: int = 8, obs_shape: tuple[int, int, int] = OBS) -> torch.Tensor:
    g = torch.Generator().manual_seed(123)
    return (torch.rand(n, *obs_shape, generator=g) > 0.5).float()


# --------------------------------------------------------------------------- #
# Shape / contract parity with MLPQNetwork
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_heads", [1, 2, 5, 10])
def test_forward_shape_is_batch_heads_actions(n_heads: int) -> None:
    net = MinAtarConvQNetwork(OBS, N_ACT, n_heads=n_heads, generator=_gen())
    out = net(_batch(8))
    assert out.shape == (8, n_heads, N_ACT)


def test_public_contract_matches_mlp_qnetwork() -> None:
    """Drop-in parity is what keeps gate C11 checkable on MinAtar (same class, all methods)."""
    required = {"reset_parameters", "trunk_features", "heads_forward", "forward"}
    assert required <= set(dir(MinAtarConvQNetwork))
    assert required <= set(dir(MLPQNetwork))
    for attr in ("n_heads", "n_actions", "feature_dim", "heads"):
        assert hasattr(MinAtarConvQNetwork(OBS, N_ACT, generator=_gen()), attr)


@pytest.mark.parametrize("game,channels", sorted(MINATAR_CHANNELS.items()))
def test_every_game_channel_count_builds(game: str, channels: int) -> None:
    obs_shape = (channels, 10, 10)
    net = MinAtarConvQNetwork(obs_shape, N_ACT, n_heads=3, generator=_gen())
    assert net(_batch(4, obs_shape)).shape == (4, 3, N_ACT)


def test_trunk_and_heads_decompose_forward() -> None:
    """forward == heads_forward(trunk_features(.)) — the ensemble agent relies on this split."""
    net = MinAtarConvQNetwork(OBS, N_ACT, n_heads=4, generator=_gen())
    x = _batch(6)
    feats = net.trunk_features(x)
    assert feats.shape == (6, net.feature_dim) == (6, MINATAR_FC_WIDTH)
    assert torch.equal(net(x), net.heads_forward(feats))


def test_trunk_shape_is_the_standard_minatar_torso() -> None:
    net = MinAtarConvQNetwork(OBS, N_ACT, generator=_gen())
    assert net.conv.out_channels == MINATAR_CONV_CHANNELS == 16
    assert net.conv_out_dim == 16 * 8 * 8  # 10x10 -> 8x8 under a valid 3x3 conv
    assert net.feature_dim == 128


def test_flat_or_wrong_rank_observation_is_rejected() -> None:
    net = MinAtarConvQNetwork(OBS, N_ACT, generator=_gen())
    with pytest.raises(ValueError, match="4-D"):
        net(torch.zeros(8, 400))  # flattened — the DeepSea habit, wrong here
    with pytest.raises(ValueError, match="channel-first"):
        MinAtarConvQNetwork((400,), N_ACT)


def test_invalid_construction_args_raise() -> None:
    with pytest.raises(ValueError, match="n_heads must be"):
        MinAtarConvQNetwork(OBS, N_ACT, n_heads=0)
    with pytest.raises(ValueError, match="must be positive"):
        MinAtarConvQNetwork((4, 10, 10), 0)
    with pytest.raises(ValueError, match="exceeds"):
        MinAtarConvQNetwork(OBS, N_ACT, kernel_size=11)


# --------------------------------------------------------------------------- #
# Determinism (gate C1) and head independence (the diversity prior)
# --------------------------------------------------------------------------- #
def test_same_stream_reproduces_weights_bitwise() -> None:
    a = MinAtarConvQNetwork(OBS, N_ACT, n_heads=5, generator=_gen(0))
    b = MinAtarConvQNetwork(OBS, N_ACT, n_heads=5, generator=_gen(0))
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(pa, pb)


def test_different_seed_index_gives_different_weights() -> None:
    a = MinAtarConvQNetwork(OBS, N_ACT, n_heads=5, generator=_gen(0))
    b = MinAtarConvQNetwork(OBS, N_ACT, n_heads=5, generator=_gen(1))
    assert not torch.equal(a.conv.weight, b.conv.weight)


def test_init_is_independent_of_global_torch_rng_state() -> None:
    """Stream-derived init: final parameters must not depend on global torch RNG state.

    Note on scope — ``nn.Conv2d``/``nn.Linear`` constructors self-initialize from the global
    RNG *before* ``reset_parameters(generator)`` overwrites them, so merely *constructing* a
    module does advance global torch state. That is true of the existing ``MLPQNetwork`` path
    too, and it is harmless: what gate C1 requires is that the resulting parameters are a pure
    function of the derived stream. This asserts exactly that, under two different global
    seeds, for parity with the MLP contract.
    """
    torch.manual_seed(0)
    a = MinAtarConvQNetwork(OBS, N_ACT, n_heads=7, generator=_gen())
    torch.manual_seed(999)
    b = MinAtarConvQNetwork(OBS, N_ACT, n_heads=7, generator=_gen())
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(pa, pb)


def test_noisy_init_is_independent_of_global_torch_rng_state() -> None:
    torch.manual_seed(0)
    a = NoisyMinAtarConvQNetwork(OBS, N_ACT, generator=_gen())
    torch.manual_seed(999)
    b = NoisyMinAtarConvQNetwork(OBS, N_ACT, generator=_gen())
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(pa, pb)


def test_heads_are_independently_initialized() -> None:
    """Independent head init *is* the Bootstrapped-DQN diversity prior (Class-2 row)."""
    net = MinAtarConvQNetwork(OBS, N_ACT, n_heads=6, generator=_gen())
    weights = [h.weight for h in net.heads]
    for i in range(len(weights)):
        for j in range(i + 1, len(weights)):
            assert not torch.equal(weights[i], weights[j])


def test_heads_give_different_q_values() -> None:
    net = MinAtarConvQNetwork(OBS, N_ACT, n_heads=4, generator=_gen())
    q = net(_batch(16))
    for i in range(4):
        for j in range(i + 1, 4):
            assert not torch.allclose(q[:, i, :], q[:, j, :])


def test_reset_parameters_is_idempotent_per_generator_state() -> None:
    net = MinAtarConvQNetwork(OBS, N_ACT, n_heads=3, generator=_gen())
    snapshot = [p.clone() for p in net.parameters()]
    net.reset_parameters(_gen())  # fresh generator, same stream -> same draws
    for p, s in zip(net.parameters(), snapshot, strict=True):
        assert torch.equal(p, s)


def test_conv_init_uses_conv_fan_in_not_weight_shape_1() -> None:
    """fan_in for a conv is in_ch*kh*kw; using weight.shape[1] over-scales by sqrt(kh*kw)."""
    net = MinAtarConvQNetwork(OBS, N_ACT, generator=_gen())
    fan_in = OBS[0] * net.kernel_size * net.kernel_size  # 4*3*3 = 36
    bound = 1.0 / math.sqrt(fan_in)
    assert net.conv.weight.abs().max().item() <= bound + 1e-9
    # A wrong fan_in of weight.shape[1] (=4) would allow |w| up to 0.5 — assert we're well under.
    wrong_bound = 1.0 / math.sqrt(OBS[0])
    assert bound < wrong_bound
    assert net.conv.weight.abs().max().item() < wrong_bound


def test_single_head_matches_ddqn_backbone_shape() -> None:
    """n_heads=1 is the DDQN/NoisyNet backbone — same trunk, one head."""
    ddqn = MinAtarConvQNetwork(OBS, N_ACT, n_heads=1, generator=_gen())
    ens = MinAtarConvQNetwork(OBS, N_ACT, n_heads=10, generator=_gen())
    assert ddqn.conv.weight.shape == ens.conv.weight.shape
    assert ddqn.feature_dim == ens.feature_dim
    assert len(ddqn.heads) == 1 and len(ens.heads) == 10
    assert ddqn(_batch(4)).shape == (4, 1, N_ACT)


# --------------------------------------------------------------------------- #
# NoisyNet conv variant
# --------------------------------------------------------------------------- #
def test_noisy_forward_shape_has_no_head_axis() -> None:
    net = NoisyMinAtarConvQNetwork(OBS, N_ACT, generator=_gen())
    assert net(_batch(8)).shape == (8, N_ACT)


def test_noisy_conv_stays_deterministic() -> None:
    """Fortunato convention: noise on the linear layers, not the convolution."""
    net = NoisyMinAtarConvQNetwork(OBS, N_ACT, generator=_gen())
    assert isinstance(net.conv, torch.nn.Conv2d)
    assert not hasattr(net.conv, "weight_sigma")
    before = net.conv.weight.clone()
    net.reset_noise(conventions.derive_torch_generator(0, CELL, "action_noise", 0))
    assert torch.equal(net.conv.weight, before)  # resampling never touches the conv


def test_noisy_resampling_changes_output_but_mean_net_is_stable() -> None:
    net = NoisyMinAtarConvQNetwork(OBS, N_ACT, generator=_gen())
    x = _batch(8)
    op = conventions.derive_torch_generator(0, CELL, "action_noise", 0)
    net.reset_noise(op)
    q1 = net(x)
    net.reset_noise(op)
    q2 = net(x)
    assert not torch.allclose(q1, q2)  # different ε -> different Q
    assert torch.allclose(net(x, noisy=False), net(x, noisy=False))  # μ-only is stable


def test_noisy_sigma_initialization_follows_fortunato() -> None:
    """sigma = sigma0 / sqrt(fan_in) for each noisy layer."""
    net = NoisyMinAtarConvQNetwork(OBS, N_ACT, sigma0=0.5, generator=_gen())
    for layer in net.layers:
        expected = 0.5 / math.sqrt(layer.in_features)
        assert torch.allclose(
            layer.weight_sigma, torch.full_like(layer.weight_sigma, expected)
        )


def test_noisy_same_stream_reproduces_weights() -> None:
    a = NoisyMinAtarConvQNetwork(OBS, N_ACT, generator=_gen(0))
    b = NoisyMinAtarConvQNetwork(OBS, N_ACT, generator=_gen(0))
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(pa, pb)


def test_noisy_rejects_flat_observation() -> None:
    net = NoisyMinAtarConvQNetwork(OBS, N_ACT, generator=_gen())
    with pytest.raises(ValueError, match="4-D"):
        net(torch.zeros(8, 400))


def test_noisy_trunk_shape_matches_plain_conv_net() -> None:
    plain = MinAtarConvQNetwork(OBS, N_ACT, generator=_gen())
    noisy = NoisyMinAtarConvQNetwork(OBS, N_ACT, generator=_gen())
    assert noisy.conv_out_dim == plain.conv_out_dim
    assert noisy.layers[0].out_features == plain.feature_dim
