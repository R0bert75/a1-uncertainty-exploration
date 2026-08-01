"""Freeze item 20: clone/restore capability and the decision procedure over it."""

from __future__ import annotations

import json

import numpy as np
import pytest

from analysis import clone_reproduction as cr
from src.minatar_env import MinAtarEnv


def _env(game="breakout", cell="test_item20"):
    e = MinAtarEnv(game, master_seed=0, cell_id=cell, seed_index=0)
    e.reset()
    return e


def _advance(env, n, rng):
    for _ in range(n):
        if env._done:
            env.reset()
        env.step(int(rng.randint(env.n_actions)))


# --------------------------------------------------------------------------- #
# Capability
# --------------------------------------------------------------------------- #
def test_clone_restore_reproduces_a_rollout_bit_exactly():
    rng = np.random.RandomState(0)
    env = _env()
    _advance(env, 25, rng)
    if env._done:
        env.reset()
    actions = [int(rng.randint(env.n_actions)) for _ in range(30)]

    snap = env.clone_state()
    env.restore_state(snap)
    t1 = cr._trace(env, actions)
    env.restore_state(snap)
    t2 = cr._trace(env, actions)
    assert t1 == t2


def test_snapshot_survives_being_replayed_many_times():
    """One snapshot must seed many replays: item 20 replays each twice, and probe rollouts
    would replay far more. A snapshot aliasing live env internals would drift after replay 1."""
    rng = np.random.RandomState(1)
    env = _env()
    _advance(env, 15, rng)
    if env._done:
        env.reset()
    actions = [int(rng.randint(env.n_actions)) for _ in range(25)]
    snap = env.clone_state()
    traces = []
    for _ in range(5):
        env.restore_state(snap)
        traces.append(cr._trace(env, actions))
    assert all(t == traces[0] for t in traces)


def test_restore_reproduces_the_ORIGINAL_trajectory_not_just_itself():
    """The check item 20's literal wording misses.

    MinAtar's ``seed()`` aliases ONE RandomState across the wrapper and the inner game. A
    ``deepcopy`` of the game splits it in two, after which both replays agree with each other
    (100/100) while neither reproduces the pre-snapshot trajectory. Replay-vs-replay alone
    cannot see that; this can.
    """
    rng = np.random.RandomState(3)
    env = _env()
    _advance(env, 20, rng)
    if env._done:
        env.reset()
    actions = [int(rng.randint(env.n_actions)) for _ in range(35)]

    snap = env.clone_state()
    original = cr._trace(env, actions)
    env.restore_state(snap)
    assert cr._trace(env, actions) == original


def test_restore_preserves_minatar_rng_aliasing():
    """The invariant behind the test above, pinned directly at the object level."""
    env = _env()
    assert env._env.random is env._env.env.random, "seed() should alias one generator"
    env.restore_state(env.clone_state())
    assert env._env.random is env._env.env.random, "restore de-aliased the shared generator"


def test_restore_reinstates_last_action():
    """Object-level pin. ``Environment.last_action`` is what a sticky repeat replays, so a
    restore that leaves it stale runs the wrong action on the next sticky event.

    Pinned at the object level because the *behavioural* signature is low-frequency: with
    sticky_action_prob=0.1 a wrong last_action changes the trajectory in only ~8% of 40-step
    rollouts (measured 16/200), so a single-seed round-trip test misses it ~92% of the time.
    """
    env = _env()
    snap = env.clone_state()
    env._env.last_action = (snap["last_action"] + 3) % env.n_actions
    env.restore_state(snap)
    assert env._env.last_action == snap["last_action"]


def test_a_stale_last_action_is_detectable_across_seeds():
    """The behavioural counterpart: over enough seeds, a wrong last_action must show up.

    This is what makes item 20's 100-test procedure adequate for this failure mode even though
    any single test is only ~8% sensitive to it.
    """
    rng = np.random.RandomState(0)
    diverged = 0
    for t in range(60):
        env = MinAtarEnv("breakout", master_seed=t, cell_id=f"stale_la_{t}", seed_index=0)
        env.reset()
        _advance(env, int(rng.randint(5, 40)), rng)
        if env._done:
            env.reset()
        actions = [int(rng.randint(env.n_actions)) for _ in range(40)]
        snap = env.clone_state()
        original = cr._trace(env, actions)
        env.restore_state(snap)
        env._env.last_action = (snap["last_action"] + 3) % env.n_actions  # simulate a stale value
        diverged += cr._trace(env, actions) != original
    assert diverged > 0, "a stale last_action never changed the trajectory -- guard is blind"


def test_snapshot_holds_no_generator_object():
    """The RNG travels as an explicit state tuple, never as a copied generator -- that copy is
    exactly what breaks the aliasing."""
    env = _env()
    snap = env.clone_state()
    assert snap["game"].random is None
    assert isinstance(snap["rng"], tuple) and snap["rng_aliased"] is True


def test_restore_rejects_an_incomplete_snapshot():
    env = _env()
    snap = env.clone_state()
    del snap["rng"]
    with pytest.raises(ValueError, match="rng"):
        env.restore_state(snap)


def test_snapshot_is_not_aliased_to_live_env_state():
    env = _env()
    rng = np.random.RandomState(2)
    snap = env.clone_state()
    before = snap["game"].brick_map.copy()
    _advance(env, 40, rng)
    assert np.array_equal(snap["game"].brick_map, before), "snapshot mutated by later stepping"


def test_done_flag_is_part_of_the_state():
    """Restoring must reinstate whether the episode was over; otherwise a restored terminal
    state would accept a step() the original refused."""
    env = _env()
    snap_live = env.clone_state()
    assert snap_live["done"] is False
    env._done = True
    env.restore_state(snap_live)
    assert env._done is False


# --------------------------------------------------------------------------- #
# The decision procedure
# --------------------------------------------------------------------------- #
def test_procedure_decides_full_on_the_installed_package():
    res = cr.decide("breakout", n_tests=12)
    assert res.decision == cr.DECISION_FULL
    assert res.clone_bit_exact == res.clone_tests == 12


def test_procedure_falls_back_to_episode_start_only_when_cloning_fails(monkeypatch):
    monkeypatch.setattr(cr, "run_clone_tests", lambda *a, **k: (7, 100))
    monkeypatch.setattr(cr, "run_fresh_reset_tests", lambda *a, **k: (100, 100))
    assert cr.decide("breakout").decision == cr.DECISION_EPISODE_START_ONLY


def test_procedure_drops_when_neither_branch_reproduces(monkeypatch):
    monkeypatch.setattr(cr, "run_clone_tests", lambda *a, **k: (7, 100))
    monkeypatch.setattr(cr, "run_fresh_reset_tests", lambda *a, **k: (91, 100))
    assert cr.decide("breakout").decision == cr.DECISION_DROP


def test_full_requires_ALL_tests_bit_exact_not_merely_most(monkeypatch):
    """Item 20 says "iff all 100"; 99/100 is not "full"."""
    monkeypatch.setattr(cr, "run_clone_tests", lambda *a, **k: (99, 100))
    monkeypatch.setattr(cr, "run_fresh_reset_tests", lambda *a, **k: (100, 100))
    assert cr.decide("breakout").decision != cr.DECISION_FULL


def test_item20_count_is_the_frozen_100():
    assert cr.N_REPRODUCTION_TESTS == 100


def test_cli_writes_an_auditable_record(tmp_path):
    out = tmp_path / "rec.json"
    rc = cr.main(["--game", "breakout", "--n-tests", "5", "--out", str(out)])
    assert rc == 0
    rec = json.loads(out.read_text())
    assert rec["decision"] == cr.DECISION_FULL
    assert rec["clone_tests"] == 5
    assert rec["rationale"]
