"""Certification that uint8 observation storage is lossless on all six environments.

The `float32 -> uint8` replay-storage change is safe because every environment in this
study emits observations in exactly ``{0.0, 1.0}``. That is a property of *these
environments*, not of the cast — so it is certified by stepping each one and checking the
round trip, rather than assumed. This is a one-time cost at test time and zero at run time.

The test also pins the two invariants that make the storage dtype invisible to agents:
``gather()`` always returns ``float32``, and a non-representable observation raises rather
than being silently truncated.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.deep_sea import DeepSea
from src.minatar_env import MINATAR_GAMES, MinAtarEnv
from src.replay_buffer import ReplayBuffer

STEPS_PER_ENV = 2000


def _assert_binary_and_lossless(obs: np.ndarray, env_label: str) -> None:
    flat = np.asarray(obs, dtype=np.float32).ravel()
    unique = np.unique(flat)
    assert np.all(np.isin(unique, [0.0, 1.0])), (
        f"{env_label} emitted values outside {{0,1}}: {unique[:10]} — uint8 replay "
        "storage would not be lossless for this environment"
    )
    roundtrip = flat.astype(np.uint8).astype(np.float32)
    assert np.array_equal(flat, roundtrip), f"{env_label} float32->uint8->float32 not exact"


@pytest.mark.parametrize("game", MINATAR_GAMES)
def test_minatar_observations_are_binary_and_uint8_lossless(game):
    env = MinAtarEnv(game, master_seed=12345, cell_id="dtype_cert", seed_index=0)
    obs, _ = env.reset()
    _assert_binary_and_lossless(obs, f"minatar/{game}")
    rng = np.random.default_rng(11)
    for _ in range(STEPS_PER_ENV):
        obs, _, terminated, truncated, _ = env.step(int(rng.integers(env.n_actions)))
        _assert_binary_and_lossless(obs, f"minatar/{game}")
        if terminated or truncated:
            obs, _ = env.reset()
            _assert_binary_and_lossless(obs, f"minatar/{game}")


@pytest.mark.parametrize("size", [10, 20, 30])
def test_deep_sea_observations_are_binary_and_uint8_lossless(size):
    env = DeepSea(size=size, master_seed=999, cell_id="dtype_cert", seed_index=0)
    obs, _ = env.reset()
    _assert_binary_and_lossless(obs, f"deep_sea/N={size}")
    rng = np.random.default_rng(5)
    for _ in range(STEPS_PER_ENV):
        obs, _, terminated, truncated, _ = env.step(int(rng.integers(2)))
        _assert_binary_and_lossless(obs, f"deep_sea/N={size}")
        if terminated or truncated:
            obs, _ = env.reset()
            _assert_binary_and_lossless(obs, f"deep_sea/N={size}")


# --------------------------------------------------------------------------- #
# Buffer-level invariants
# --------------------------------------------------------------------------- #
def test_gather_returns_float32_from_uint8_storage():
    """The storage dtype must be invisible to agents: gather() always yields float32."""
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(64, (4,), rng, obs_dtype=np.uint8)
    for i in range(10):
        o = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        buf.add(o, i % 2, 1.0, o, False)
    batch = buf.gather(np.arange(4))
    assert batch.obs.dtype.__str__() == "torch.float32"
    assert batch.next_obs.dtype.__str__() == "torch.float32"


def test_uint8_and_float32_storage_gather_identical_values():
    """The representation change is bit-exact on binary observations."""
    obs_seq = [
        np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
    ]
    gathered = {}
    for dtype in (np.float32, np.uint8):
        buf = ReplayBuffer(16, (4,), np.random.default_rng(0), obs_dtype=dtype)
        for i, o in enumerate(obs_seq):
            buf.add(o, i, float(i), o, i == 3)
        gathered[str(np.dtype(dtype))] = buf.gather(np.arange(4)).obs.numpy()
    assert np.array_equal(gathered["float32"], gathered["uint8"])


def test_non_representable_observation_raises_rather_than_truncating():
    """A 0.5 must not silently become 0 — the failure mode this guard exists to catch."""
    buf = ReplayBuffer(8, (2,), np.random.default_rng(0), obs_dtype=np.uint8)
    bad = np.array([0.5, 1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="not representable in storage dtype"):
        buf.add(bad, 0, 0.0, bad, False)


def test_float32_storage_accepts_non_binary_observations():
    """The guard is scoped to integer storage; float32 storage is unchanged."""
    buf = ReplayBuffer(8, (2,), np.random.default_rng(0), obs_dtype=np.float32)
    o = np.array([0.5, 0.25], dtype=np.float32)
    buf.add(o, 0, 0.0, o, False)
    assert np.allclose(buf.gather(np.array([0])).obs.numpy(), [[0.5, 0.25]])


def test_uint8_storage_is_four_times_smaller():
    shape = (4, 10, 10)
    f32 = ReplayBuffer(1000, shape, np.random.default_rng(0), obs_dtype=np.float32)
    u8 = ReplayBuffer(1000, shape, np.random.default_rng(0), obs_dtype=np.uint8)
    assert f32._obs.nbytes == 4 * u8._obs.nbytes
