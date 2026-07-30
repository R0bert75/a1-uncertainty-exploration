"""Tests for the MinAtar adapter: shape/dtype contract, action space, determinism (gate C1)."""

from __future__ import annotations

import numpy as np
import pytest

from src.minatar_env import (
    GRID_SIZE,
    MINATAR_CHANNELS,
    MINATAR_GAMES,
    MINATAR_HELDOUT_GAMES,
    MINATAR_TUNING_GAMES,
    N_ACTIONS,
    MinAtarEnv,
)
from src.utils import conventions

CELL = "minatar|breakout|ddqn"


def _env(game: str = "breakout", seed_index: int = 0, master_seed: int = 0) -> MinAtarEnv:
    return MinAtarEnv(game, master_seed=master_seed, cell_id=CELL, seed_index=seed_index)


# --------------------------------------------------------------------------- #
# Observation contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("game", MINATAR_GAMES)
def test_observation_shape_is_channel_first_float32(game: str) -> None:
    env = _env(game)
    obs, info = env.reset()
    assert obs.shape == (MINATAR_CHANNELS[game], GRID_SIZE, GRID_SIZE)
    assert obs.dtype == np.float32
    assert obs.shape == env.obs_shape
    assert info["game"] == game
    # MinAtar states are binary planes; the adapter must not rescale them.
    assert set(np.unique(obs)).issubset({0.0, 1.0})


@pytest.mark.parametrize("game", MINATAR_GAMES)
def test_action_space_is_full_six_for_every_game(game: str) -> None:
    """Uniform action space across games: a per-game space would alias game onto method."""
    env = _env(game)
    assert env.n_actions == N_ACTIONS == 6


def test_step_returns_gymnasium_five_tuple() -> None:
    env = _env()
    env.reset()
    out = env.step(0)
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert obs.shape == env.obs_shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert truncated is False  # MinAtar has no internal time limit
    assert info["game"] == "breakout"


def test_obs_is_contiguous_for_torch_from_numpy() -> None:
    """A transposed view would be non-contiguous; torch.from_numpy on it is a silent perf trap."""
    env = _env()
    obs, _ = env.reset()
    assert obs.flags["C_CONTIGUOUS"]


# --------------------------------------------------------------------------- #
# Episode lifecycle
# --------------------------------------------------------------------------- #
def test_step_before_reset_raises() -> None:
    env = _env()
    with pytest.raises(RuntimeError, match="finished episode"):
        env.step(0)


def test_episode_terminates_and_can_be_reset() -> None:
    env = _env()
    env.reset()
    rng = np.random.RandomState(0)
    terminated = False
    for _ in range(5000):
        _, _, terminated, _, _ = env.step(int(rng.randint(N_ACTIONS)))
        if terminated:
            break
    assert terminated, "breakout did not terminate within 5000 random steps"
    with pytest.raises(RuntimeError):
        env.step(0)  # terminal state is enforced
    obs, _ = env.reset()
    assert obs.shape == env.obs_shape  # reset revives the env


def test_invalid_action_raises() -> None:
    env = _env()
    env.reset()
    with pytest.raises(ValueError, match="action must be in"):
        env.step(N_ACTIONS)


def test_unknown_game_raises() -> None:
    with pytest.raises(ValueError, match="game must be one of"):
        MinAtarEnv("pong", master_seed=0, cell_id=CELL, seed_index=0)


# --------------------------------------------------------------------------- #
# Determinism — gate C1
# --------------------------------------------------------------------------- #
def _trace(env: MinAtarEnv, actions: list[int]) -> tuple[list[float], list[float]]:
    """Reward trace and observation checksums for a fixed action sequence."""
    env.reset()
    rewards, sums = [], []
    for a in actions:
        obs, r, term, _, _ = env.step(a)
        rewards.append(r)
        sums.append(float(obs.sum()))
        if term:
            env.reset()
    return rewards, sums


def test_same_seed_index_reproduces_trace_bitwise() -> None:
    actions = [int(a) for a in np.random.RandomState(7).randint(0, N_ACTIONS, size=300)]
    a = _trace(_env(seed_index=0), actions)
    b = _trace(_env(seed_index=0), actions)
    assert a == b


def test_different_seed_index_diverges() -> None:
    actions = [int(a) for a in np.random.RandomState(7).randint(0, N_ACTIONS, size=300)]
    a = _trace(_env(seed_index=0), actions)
    b = _trace(_env(seed_index=1), actions)
    assert a != b, "different seed_index produced an identical trace — env not seed-dependent"


def test_env_seed_comes_from_env_mapping_stream() -> None:
    """The env seed must be exactly the frozen derivation — no ad-hoc seeding.

    MinAtar's legacy ``RandomState`` takes only 32-bit seeds, so the adapter narrows through
    ``SeedSequence.generate_state`` (numpy's own canonical 32-bit derivation) rather than
    masking the 63-bit ``derive_seed`` int. This asserts that exact path.
    """
    env = _env(seed_index=3, master_seed=11)
    expected_ss = conventions.derive_seed_sequence(11, CELL, "env_mapping", 3)
    expected = int(expected_ss.generate_state(1, dtype=np.uint32)[0])
    assert env.env_seed == expected
    assert 0 <= env.env_seed < 2**32


def test_env_seed_is_32_bit_for_every_game_and_seed_index() -> None:
    """Regression guard: a 63-bit seed reaches RandomState as a hard ValueError."""
    for game in MINATAR_GAMES:
        for si in range(4):
            env = MinAtarEnv(game, master_seed=0, cell_id=CELL, seed_index=si)
            assert 0 <= env.env_seed < 2**32


def test_cell_id_changes_the_env_seed() -> None:
    a = MinAtarEnv("breakout", master_seed=0, cell_id="cell|a", seed_index=0)
    b = MinAtarEnv("breakout", master_seed=0, cell_id="cell|b", seed_index=0)
    assert a.env_seed != b.env_seed


def test_pinned_env_settings_are_explicit() -> None:
    env = _env()
    assert env.sticky_action_prob == 0.1
    assert env.difficulty_ramping is True
    info = env.reset()[1]
    assert info["sticky_action_prob"] == 0.1
    assert info["difficulty_ramping"] is True


# --------------------------------------------------------------------------- #
# Frozen game split
# --------------------------------------------------------------------------- #
def test_tuning_and_heldout_split_partitions_the_game_set() -> None:
    assert MINATAR_TUNING_GAMES == ("breakout", "asterix")
    assert set(MINATAR_TUNING_GAMES) | set(MINATAR_HELDOUT_GAMES) == set(MINATAR_GAMES)
    assert not set(MINATAR_TUNING_GAMES) & set(MINATAR_HELDOUT_GAMES)
    assert len(MINATAR_HELDOUT_GAMES) == 3
