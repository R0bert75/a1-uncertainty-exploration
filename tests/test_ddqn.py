"""Tests for the DDQN baseline, its replay buffer, and the shared Q-network.

Focus: gate C1 (determinism from the derived streams — same ``(seed, cell)`` reproduces
init, sampling, and exploration bit-for-bit; different cells diverge), gate C11 (the
single code path — DDQN is ``MLPQNetwork`` with ``n_heads=1``, and the ensemble seam
exists), plus a learning smoke that the update actually descends TD error on a trivial
deterministic task. No frozen protocol value is asserted here; these are infrastructure
correctness checks, not scientific results.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.ddqn import DDQNAgent, DDQNConfig  # noqa: E402
from src.networks import MLPQNetwork  # noqa: E402
from src.replay_buffer import ReplayBuffer  # noqa: E402
from src.utils import conventions  # noqa: E402


# --------------------------------------------------------------------------- #
# MLPQNetwork
# --------------------------------------------------------------------------- #
def test_network_forward_shape_single_and_multi_head():
    net1 = MLPQNetwork(6, 2, (8,), n_heads=1)
    net5 = MLPQNetwork(6, 2, (8,), n_heads=5)
    x = torch.zeros(4, 6)
    assert net1(x).shape == (4, 1, 2)   # [batch, n_heads, n_actions]
    assert net5(x).shape == (4, 5, 2)   # same code path, K heads (C11 seam)


def test_network_init_is_stream_deterministic_and_no_global_rng():
    g1 = conventions.derive_torch_generator(0, "ddqn_egreedy", "init", 0)
    g2 = conventions.derive_torch_generator(0, "ddqn_egreedy", "init", 0)
    a = MLPQNetwork(6, 2, (8,), n_heads=1, generator=g1)
    b = MLPQNetwork(6, 2, (8,), n_heads=1, generator=g2)
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(pa, pb)      # same (seed, cell, stream) -> identical weights


def test_network_init_differs_across_cells_and_across_heads():
    ga = conventions.derive_torch_generator(0, "cellA", "init", 0)
    gb = conventions.derive_torch_generator(0, "cellB", "init", 0)
    a = MLPQNetwork(6, 2, (8,), n_heads=1, generator=ga)
    b = MLPQNetwork(6, 2, (8,), n_heads=1, generator=gb)
    # different cell -> different init (no cross-cell reuse)
    pairs = zip(a.parameters(), b.parameters(), strict=True)
    assert not all(torch.equal(pa, pb) for pa, pb in pairs)
    # within one ensemble, heads are independently initialized
    gc = conventions.derive_torch_generator(0, "cellA", "init", 0)
    ens = MLPQNetwork(6, 2, (8,), n_heads=2, generator=gc)
    assert not torch.equal(ens.heads[0].weight, ens.heads[1].weight)


# --------------------------------------------------------------------------- #
# ReplayBuffer
# --------------------------------------------------------------------------- #
def test_replay_ring_overwrites_and_reports_size():
    rng = conventions.derive_numpy_generator(0, "cellA", "replay", 0)
    buf = ReplayBuffer(3, (2,), rng)
    for i in range(5):
        buf.add(np.full(2, i, np.float32), i % 2, float(i), np.zeros(2, np.float32), False)
    assert len(buf) == 3 and buf.is_full          # capacity-bounded ring


def test_replay_sampling_is_stream_reproducible():
    def fill_and_sample(seed_cell):
        rng = conventions.derive_numpy_generator(0, seed_cell, "replay", 0)
        buf = ReplayBuffer(50, (2,), rng)
        for i in range(50):
            buf.add(np.full(2, i, np.float32), 0, 0.0, np.zeros(2, np.float32), False)
        return buf.sample(8).obs.numpy()
    np.testing.assert_array_equal(fill_and_sample("cellA"), fill_and_sample("cellA"))
    # different cell -> different minibatch indices
    assert not np.array_equal(fill_and_sample("cellA"), fill_and_sample("cellB"))


def test_replay_refuses_undersized_sample():
    rng = conventions.derive_numpy_generator(0, "cellA", "replay", 0)
    buf = ReplayBuffer(10, (2,), rng)
    buf.add(np.zeros(2, np.float32), 0, 0.0, np.zeros(2, np.float32), False)
    with pytest.raises(ValueError):
        buf.sample(4)


# --------------------------------------------------------------------------- #
# DDQNAgent
# --------------------------------------------------------------------------- #
def _agent(cell_id="ddqn_egreedy", seed_index=0, **overrides):
    params = dict(
        obs_dim=4, n_actions=2, hidden_sizes=(16,),
        min_buffer=8, batch_size=4, target_update_period=5, eps_decay_steps=20,
    )
    params.update(overrides)
    cfg = DDQNConfig(**params)
    return DDQNAgent(cfg, master_seed=0, cell_id=cell_id, seed_index=seed_index)


def test_epsilon_decays_monotonically_to_floor():
    ag = _agent()
    assert ag.epsilon(0) == pytest.approx(ag.cfg.eps_start)
    assert ag.epsilon(10) > ag.epsilon(15)
    assert ag.epsilon(10_000) == pytest.approx(ag.cfg.eps_end)


def test_agent_is_reproducible_from_seed_and_cell():
    a, b = _agent(), _agent()
    obs = np.arange(4, dtype=np.float32)
    acts_a = [a.select_action(obs, step=0) for _ in range(20)]  # step 0 -> eps=1, pure explore
    acts_b = [b.select_action(obs, step=0) for _ in range(20)]
    assert acts_a == acts_b                                     # action_noise stream matches
    # different cell -> different exploration draws
    c = _agent(cell_id="episodic|off|K10")
    acts_c = [c.select_action(obs, step=0) for _ in range(20)]
    assert acts_a != acts_c


def test_target_starts_synced_and_learn_step_reduces_error_on_fixed_batch():
    ag = _agent(gamma=0.0)  # gamma=0 -> target is just the reward; a clean regression check
    # target net initialized to a copy of online
    for po, pt in zip(ag.online.parameters(), ag.target.parameters(), strict=True):
        assert torch.equal(po, pt)
    # one deterministic transition repeated: obs->action 1 gives reward 1.0
    obs = np.ones(4, np.float32)
    nxt = np.zeros(4, np.float32)
    for _ in range(64):
        ag.observe(obs, 1, 1.0, nxt, True)
    losses = [ls for _ in range(1000) if (ls := ag.learn_step()) is not None]
    assert len(losses) > 0
    assert losses[-1] < losses[0]          # TD error descends
    # Q(obs, a=1) should approach the reward (1.0) under gamma=0
    with torch.no_grad():
        q = ag.online(torch.as_tensor(obs).unsqueeze(0))[0, 0, 1].item()
    assert abs(q - 1.0) < 0.1


def test_learn_step_returns_none_until_min_buffer():
    ag = _agent(min_buffer=16)
    obs = np.ones(4, np.float32)
    for _ in range(10):
        ag.observe(obs, 0, 0.0, obs, False)
    assert ag.learn_step() is None         # below min_buffer -> no learning yet


def test_target_network_syncs_on_cadence():
    ag = _agent(gamma=0.0, target_update_period=3)
    obs = np.ones(4, np.float32)
    for _ in range(32):
        ag.observe(obs, 1, 1.0, np.zeros(4, np.float32), True)
    # drive exactly 3 learn steps -> one sync boundary; target should re-match online
    done = 0
    while done < 3:
        if ag.learn_step() is not None:
            done += 1
    for po, pt in zip(ag.online.parameters(), ag.target.parameters(), strict=True):
        assert torch.equal(po, pt)
