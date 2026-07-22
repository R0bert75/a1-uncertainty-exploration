"""Tests for the DeepSea environment and its exact ground-truth Q*.

Gate C5 (environment / implementation validity): correct transitions, rewards, terminal
condition, observation encoding, and the ``col <= row`` invariant that makes the treasure
fire exactly once. Gate C7 preview (ground truth): the exact Q* solver is checked against an
independent brute-force enumeration of every action sequence, and against the analytic 0.99
optimal return. Gate C1: the action mapping (and thus Q*) is reproducible from the derived
``env_mapping`` stream and separates across cells. No frozen protocol value is asserted.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from src.deep_sea import MOVE_COST_DEFAULT, DeepSea  # noqa: E402
from src.utils import conventions  # noqa: E402


def _env(size=6, cell_id="episodic|off|K10", master_seed=0, seed_index=0):
    return DeepSea(size, master_seed=master_seed, cell_id=cell_id, seed_index=seed_index)


# --------------------------------------------------------------------------- #
# Gymnasium contract
# --------------------------------------------------------------------------- #
def test_spaces_and_reset_observation():
    env = _env(size=5)
    assert env.action_space.n == 2
    assert env.observation_space.shape == (5, 5)
    obs, info = env.reset()
    assert obs.shape == (5, 5) and obs.dtype == np.float32
    assert obs[0, 0] == 1.0 and obs.sum() == 1.0          # one-hot at start cell
    assert info["mapping_hash"] == env.mapping_hash


def test_step_before_reset_raises():
    env = _env()
    with pytest.raises(RuntimeError):
        env.step(0)


def test_episode_terminates_after_exactly_size_steps():
    env = _env(size=7)
    env.reset()
    steps, terminated = 0, False
    while not terminated:
        _, _, terminated, truncated, _ = env.step(0)  # always "left-ish"; length is fixed
        steps += 1
        assert truncated is False
    assert steps == 7                                      # one row per step, N rows


# --------------------------------------------------------------------------- #
# Dynamics: invariants and the treasure
# --------------------------------------------------------------------------- #
def test_col_never_exceeds_row_and_stays_in_bounds():
    env = _env(size=8)
    env.reset()
    terminated = False
    rng = np.random.default_rng(0)
    while not terminated:
        _, _, terminated, _, info = env.step(int(rng.integers(0, 2)))
        # col can never run ahead of the row the agent has descended to
        if not terminated:
            assert 0 <= info["col"] <= info["row"]


def test_optimal_rollout_reaches_treasure_with_099_return():
    env = _env(size=6)
    env.reset()
    total, terminated = 0.0, False
    while not terminated:
        a = int(env.right_action[env._row])               # always take the right action
        _, r, terminated, _, _ = env.step(a)
        total += r
    assert total == pytest.approx(1.0 - MOVE_COST_DEFAULT)  # 0.99 optimal return
    assert env.optimal_return == pytest.approx(0.99)


def test_uninformed_reward_is_treasure_only_on_last_column():
    # A wrong action anywhere off the diagonal cannot collect treasure; verify treasure
    # is paid exactly once and only via a right action at the final column.
    env = _env(size=5)
    env.reset()
    treasures, terminated = 0, False
    while not terminated:
        at_last_col = env._col == env.size - 1
        a = int(env.right_action[env._row])
        _, r, terminated, _, _ = env.step(a)
        if r >= 0.9:                                       # +1 treasure (minus tiny move cost)
            treasures += 1
            assert at_last_col
    assert treasures == 1


# --------------------------------------------------------------------------- #
# Exact ground truth Q*
# --------------------------------------------------------------------------- #
def _brute_force_optimal_return(env, gamma):
    """Independent max over every action sequence from the start state."""
    n = env.size
    best = -np.inf
    for seq in itertools.product((0, 1), repeat=n):
        row, col, ret, discount = 0, 0, 0.0, 1.0
        for a in seq:
            is_right = a == env.right_action[row]
            reward = 0.0
            if col == n - 1 and is_right:
                reward += 1.0
            if is_right:
                col = min(col + 1, n - 1)
                reward -= env.move_cost / n
            else:
                col = max(col - 1, 0)
            ret += discount * reward
            discount *= gamma
            row += 1
        best = max(best, ret)
    return best


@pytest.mark.parametrize("gamma", [1.0, 0.99])
def test_q_star_matches_brute_force_enumeration(gamma):
    env = _env(size=7)
    q = env.q_star(gamma=gamma)
    v00 = q[0, 0].max()
    assert v00 == pytest.approx(_brute_force_optimal_return(env, gamma))


def test_q_star_undiscounted_value_equals_optimal_return():
    env = _env(size=6)
    q = env.q_star(gamma=1.0)
    assert q[0, 0].max() == pytest.approx(env.optimal_return)


def test_optimal_action_follows_the_mapping_on_the_diagonal():
    env = _env(size=6)
    # On the optimal-path diagonal cell (r, r), the optimal action is the row's right action.
    for r in range(env.size):
        assert env.optimal_action(r, r) == int(env.right_action[r])


# --------------------------------------------------------------------------- #
# Determinism (C1)
# --------------------------------------------------------------------------- #
def test_mapping_reproducible_and_cell_separated():
    a = _env(cell_id="episodic|off|K10")
    b = _env(cell_id="episodic|off|K10")
    c = _env(cell_id="per_step|on|K20")
    assert a.mapping_hash == b.mapping_hash
    np.testing.assert_array_equal(a.right_action, b.right_action)
    assert a.mapping_hash != c.mapping_hash                 # different cell -> different mapping


def test_mapping_hash_is_bound_to_the_frozen_generator():
    env = _env(size=6, cell_id="episodic|off|K10")
    flip = conventions.deepsea_action_mapping(0, "episodic|off|K10", 0, 6)
    assert env.mapping_hash == conventions.deepsea_mapping_hash(flip)
    np.testing.assert_array_equal(env.right_action, flip.astype(np.int64))
