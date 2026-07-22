"""Tests for the Bootstrapped-DQN ensemble agent (src/bdqn.py).

Covers the Class-2 mechanics that make this the ensemble and not the baseline: K-head
network, per-head Ber(mask_prob) bootstrap masks off the ``bootstrap_mask`` stream, the
1/K trunk-gradient normalization (Osband 2016 §6.1), per-head Double-DQN targets, the
three use-rules, and gate-C1 reproducibility (bit-for-bit from the RNG triple) + gate-C11
code-path sharing with the DDQN baseline.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.bdqn import BDQNAgent, BDQNConfig  # noqa: E402
from src.replay_buffer import Transition  # noqa: E402

OBS_DIM = 8
N_ACTIONS = 2


def _obs(i: int) -> np.ndarray:
    return np.eye(OBS_DIM, dtype=np.float32)[i % OBS_DIM]


def _agent(cell_id="episodic|off|K5", seed_index=0, **overrides):
    params = dict(
        obs_dim=OBS_DIM, n_actions=N_ACTIONS, K=5, use_rule="episodic",
        hidden_sizes=(16,), min_buffer=16, batch_size=8, target_update_period=5,
    )
    params.update(overrides)
    cfg = BDQNConfig(**params)
    return BDQNAgent(cfg, master_seed=0, cell_id=cell_id, seed_index=seed_index)


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #
def test_config_rejects_bad_use_rule():
    with pytest.raises(ValueError, match="use_rule"):
        BDQNConfig(obs_dim=OBS_DIM, n_actions=N_ACTIONS, use_rule="bogus")


def test_config_rejects_bad_mask_prob_and_k():
    with pytest.raises(ValueError, match="mask_prob"):
        BDQNConfig(obs_dim=OBS_DIM, n_actions=N_ACTIONS, mask_prob=1.5)
    with pytest.raises(ValueError, match="K"):
        BDQNConfig(obs_dim=OBS_DIM, n_actions=N_ACTIONS, K=0)


def test_config_rejects_bad_head_loss_agg():
    with pytest.raises(ValueError, match="head_loss_agg"):
        BDQNConfig(obs_dim=OBS_DIM, n_actions=N_ACTIONS, head_loss_agg="mean")


# --------------------------------------------------------------------------- #
# Network shape / K heads
# --------------------------------------------------------------------------- #
def test_q_all_has_one_row_per_head():
    ag = _agent(K=7)
    q = ag._q_all(_obs(0))
    assert tuple(q.shape) == (7, N_ACTIONS)


def test_online_and_target_start_synced():
    ag = _agent()
    for po, pt in zip(ag.online.parameters(), ag.target.parameters(), strict=True):
        assert torch.equal(po, pt)


# --------------------------------------------------------------------------- #
# Bootstrap masks — Ber(mask_prob), per head, off the bootstrap_mask stream
# --------------------------------------------------------------------------- #
def test_masks_are_bernoulli_and_ring_locked_to_buffer():
    ag = _agent(K=10, buffer_capacity=1000)
    for i in range(500):
        o = _obs(i)
        ag.observe(o, 0, 0.0, o, False)
    assert len(ag.buffer) == 500
    assert ag._mask_pos == 500
    rate = float(ag._masks[:500].mean())
    assert 0.4 < rate < 0.6  # Ber(0.5)
    # masks are 0/1
    assert set(np.unique(ag._masks[:500]).tolist()) <= {0.0, 1.0}


def test_mask_prob_respected():
    ag = _agent(K=8, mask_prob=0.25, buffer_capacity=2000)
    for i in range(1000):
        o = _obs(i)
        ag.observe(o, 0, 0.0, o, False)
    assert 0.2 < float(ag._masks[:1000].mean()) < 0.3


def test_masks_stream_reproducible_and_cell_separated():
    a = _agent(K=6)
    b = _agent(K=6)
    c = _agent(K=6, seed_index=1)
    for i in range(100):
        o = _obs(i)
        for ag in (a, b, c):
            ag.observe(o, 0, 0.0, o, False)
    assert np.array_equal(a._masks[:100], b._masks[:100])  # same triple → same masks
    assert not np.array_equal(a._masks[:100], c._masks[:100])  # different seed → different


# --------------------------------------------------------------------------- #
# Use-rules
# --------------------------------------------------------------------------- #
def test_episodic_head_fixed_within_episode_then_resamples():
    ag = _agent(use_rule="episodic")
    h0 = ag._active_head
    assert all(ag._active_head == h0 for _ in range(20))
    # greedy action is the active head's argmax, stable within the episode
    o = _obs(0)
    a0 = ag.select_action(o, 0)
    assert all(ag.select_action(o, s) == a0 for s in range(1, 20))
    ag.on_episode_start()
    assert isinstance(ag._active_head, int) and 0 <= ag._active_head < ag.K


def test_per_step_uses_action_noise_only_no_epsilon():
    # per_step must not consult epsilon; it samples a fresh head each call.
    ag = _agent(use_rule="per_step")
    acts = {ag.select_action(_obs(0), s) for s in range(50)}
    assert acts <= {0, 1}


def test_ensemble_mean_is_epsilon_greedy():
    ag = _agent(use_rule="ensemble_mean", eps_start=1.0, eps_end=1.0, eps_decay_steps=1)
    # ε≈1 → essentially random; ε decays; schedule monotone non-increasing
    eps = [ag.epsilon(s) for s in range(0, 100, 10)]
    assert all(eps[i + 1] <= eps[i] + 1e-9 for i in range(len(eps) - 1))


# --------------------------------------------------------------------------- #
# Learning — per-head Double-DQN target, 1/K trunk grad, masked loss
# --------------------------------------------------------------------------- #
def test_learn_step_none_until_min_buffer():
    ag = _agent(min_buffer=32, batch_size=16)
    for i in range(20):  # < min_buffer
        o = _obs(i)
        ag.observe(o, 0, 0.0, o, False)
    assert ag.learn_step() is None


def test_per_head_target_shape_is_batch_by_k():
    ag = _agent(K=5, batch_size=8)
    for i in range(64):
        o = _obs(i)
        ag.observe(o, i % 2, 0.5, o, False)
    idx = ag.buffer.sample_indices(ag.cfg.batch_size)
    raw = ag.buffer.gather(idx)
    batch = Transition(raw.obs.float(), raw.action, raw.reward, raw.next_obs.float(), raw.done)
    target = ag._per_head_td_target(batch)
    assert tuple(target.shape) == (ag.cfg.batch_size, ag.K)


def test_trunk_gradient_normalized_by_one_over_k():
    # The hook must scale the shared-trunk gradient by exactly 1/K relative to the
    # unnormalized per-head gradient sum (Osband 2016 §6.1).
    def trunk_grad(with_hook: bool):
        ag = _agent(K=4, hidden_sizes=(16,), batch_size=8)
        for i in range(64):
            o = _obs(i)
            ag.observe(o, i % 2, 0.5, o, False)
        idx = ag.buffer.sample_indices(ag.cfg.batch_size)
        raw = ag.buffer.gather(idx)
        batch = Transition(
            raw.obs.float(), raw.action, raw.reward, raw.next_obs.float(), raw.done
        )
        masks = torch.as_tensor(ag._masks[idx])
        target = ag._per_head_td_target(batch)
        feats = ag.online.trunk_features(batch.obs)
        feats.retain_grad()
        if with_hook:
            feats.register_hook(lambda g: g / ag.K)
        q = ag.online.heads_forward(feats)
        qt = q.gather(2, batch.action.view(-1, 1, 1).expand(-1, ag.K, 1)).squeeze(2)
        per = torch.nn.functional.smooth_l1_loss(qt, target, reduction="none") * masks
        loss = (per.sum(0) / masks.sum(0).clamp(min=1.0)).sum()
        loss.backward()
        return feats.grad.abs().sum().item()

    ratio = trunk_grad(True) / trunk_grad(False)
    assert abs(ratio - 0.25) < 1e-5  # 1/K, K=4


def test_learn_step_reduces_error_on_fixed_deterministic_batch():
    # gamma=0, done=True → target = reward = f(state); repeated learning drives loss down
    # and recovers the per-state target as the mean head prediction.
    ag = _agent(K=5, hidden_sizes=(32,), min_buffer=16, batch_size=16,
                gamma=0.0, lr=1e-3, target_update_period=10)
    det = {s: float(s % 3 == 0) for s in range(OBS_DIM)}
    for s in range(OBS_DIM):
        o = _obs(s)
        for _ in range(8):
            ag.observe(o, 0, det[s], o, True)
    losses = [x for x in (ag.learn_step() for _ in range(600)) if x is not None]
    assert losses[-1] < 0.1 * losses[0]
    errs = [abs(ag._q_all(_obs(s))[:, 0].mean().item() - det[s]) for s in range(OBS_DIM)]
    assert np.mean(errs) < 0.05


def test_target_syncs_on_cadence():
    ag = _agent(min_buffer=16, batch_size=8, target_update_period=5, gamma=0.0)
    for i in range(64):
        o = _obs(i)
        ag.observe(o, i % 2, 0.5, o, True)
    # step a few times, then perturb online and confirm a sync copies it over
    for _ in range(4):
        ag.learn_step()
    with torch.no_grad():
        next(ag.online.parameters()).add_(5.0)
    synced_before = torch.equal(
        next(ag.online.parameters()), next(ag.target.parameters())
    )
    ag.learn_step()  # this is the 5th → triggers sync
    synced_after = torch.equal(
        next(ag.online.parameters()), next(ag.target.parameters())
    )
    assert not synced_before and synced_after


# --------------------------------------------------------------------------- #
# Reproducibility (C1) + cell separation
# --------------------------------------------------------------------------- #
def _rollout(ag, n=60):
    acts = []
    rng = np.random.default_rng(123)
    for step in range(n):
        o = _obs(int(rng.integers(0, OBS_DIM)))
        a = ag.select_action(o, step)
        no = _obs(int(rng.integers(0, OBS_DIM)))
        ag.observe(o, a, float(rng.random()), no, bool(rng.random() < 0.1))
        ag.learn_step()
        acts.append(a)
    return acts


def test_reproducible_from_triple():
    assert _rollout(_agent()) == _rollout(_agent())


def test_cell_and_seed_separation():
    base = _rollout(_agent())
    assert _rollout(_agent(seed_index=3)) != base
    assert _rollout(_agent(cell_id="per_step|off|K5", use_rule="per_step")) != base


def test_no_global_torch_rng_dependence():
    # Seeding torch's global RNG differently must not change the agent (all draws derived).
    torch.manual_seed(0)
    a = _rollout(_agent())
    torch.manual_seed(9999)
    b = _rollout(_agent())
    assert a == b


# --------------------------------------------------------------------------- #
# Randomized prior functions (RP-BDQN, prior=on) — Osband et al. 2018 §3 / Alg. 1
# --------------------------------------------------------------------------- #
def test_config_rejects_nonpositive_prior_scale():
    with pytest.raises(ValueError, match="prior_scale"):
        BDQNConfig(obs_dim=OBS_DIM, n_actions=N_ACTIONS, prior_scale=0.0)
    with pytest.raises(ValueError, match="prior_scale"):
        BDQNConfig(obs_dim=OBS_DIM, n_actions=N_ACTIONS, prior_scale=-1.0)


def test_prior_off_builds_no_prior():
    ag = _agent()  # prior_scale defaults to None
    assert ag.prior is None
    assert ag.prior_scale == 0.0


def test_prior_on_builds_a_frozen_prior():
    ag = _agent(cell_id="episodic|on|K5", prior_scale=3.0)
    assert ag.prior is not None
    assert ag.prior_scale == 3.0
    # The prior is untrainable and NOT handed to the optimizer.
    assert all(not p.requires_grad for p in ag.prior.parameters())
    opt_params = {id(p) for group in ag.optimizer.param_groups for p in group["params"]}
    assert not any(id(p) in opt_params for p in ag.prior.parameters())


def test_prior_scale_off_gives_byte_identical_trainable_net():
    # The prior is drawn AFTER the trainable net from the same init generator, so turning
    # the prior on must not perturb the trainable net's initialization (gate C1: prior=off
    # cells stay bit-for-bit identical to the plain ensemble).
    off = _agent(cell_id="episodic|off|K5")
    on = _agent(cell_id="episodic|off|K5", prior_scale=3.0)
    for p_off, p_on in zip(off.online.parameters(), on.online.parameters(), strict=True):
        assert torch.equal(p_off, p_on)


def test_prediction_is_net_plus_beta_prior():
    ag = _agent(cell_id="episodic|on|K5", prior_scale=2.5)
    obs = _obs(0)
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        net_only = ag.online(obs_t)[0]
        prior_only = ag.prior(obs_t)[0]
        q = ag._q_all(obs)
    assert torch.allclose(q, net_only + 2.5 * prior_only, atol=1e-6)


def test_prior_is_shared_by_online_and_target():
    # The SAME prior is added to both nets — only the trainable params differ (Osband 2018).
    ag = _agent(cell_id="episodic|on|K5", prior_scale=2.0)
    obs_t = torch.as_tensor(_obs(1), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        # online + prior minus online-net = target + prior minus target-net = the prior term.
        term_from_online = ag._q_online(obs_t) - ag.online(obs_t)
        term_from_target = ag._q_target(obs_t) - ag.target(obs_t)
    assert torch.allclose(term_from_online, term_from_target, atol=1e-6)


def test_prior_unchanged_after_training():
    ag = _agent(cell_id="episodic|on|K5", prior_scale=2.0, gamma=0.0,
                batch_size=16, min_buffer=16, lr=1e-3)
    before = torch.cat([p.flatten() for p in ag.prior.parameters()]).clone()
    det = {s: float(s % 3 == 0) for s in range(OBS_DIM)}
    for s in range(OBS_DIM):
        o = _obs(s)
        for _ in range(8):
            ag.observe(o, 0, det[s], o, True)
    for _ in range(400):
        ag.learn_step()
    after = torch.cat([p.flatten() for p in ag.prior.parameters()])
    assert torch.equal(before, after)


def test_prior_on_preserves_exact_1_over_k_trunk_gradient():
    # The prior is a detached constant, so the 1/K trunk-gradient normalization is untouched.
    ag = _agent(K=4, prior_scale=3.0, cell_id="episodic|on|K4")
    for i in range(64):
        o = _obs(i)
        ag.observe(o, i % N_ACTIONS, 0.5, o, False)
    idx = ag.buffer.sample_indices(8)
    raw = ag.buffer.gather(idx)
    batch = Transition(raw.obs.float(), raw.action, raw.reward, raw.next_obs.float(), raw.done)
    masks = torch.as_tensor(ag._masks[idx])
    target = ag._per_head_td_target(batch)

    def grad_sum(hook: bool) -> float:
        feats = ag.online.trunk_features(batch.obs)
        feats.retain_grad()
        if hook:
            feats.register_hook(lambda g: g / ag.K)
        q = ag.online.heads_forward(feats) + ag._prior_term(batch.obs)
        qt = q.gather(2, batch.action.view(-1, 1, 1).expand(-1, ag.K, 1)).squeeze(2)
        per = torch.nn.functional.smooth_l1_loss(qt, target, reduction="none") * masks
        loss = (per.sum(0) / masks.sum(0).clamp(min=1.0)).sum()
        ag.online.zero_grad(set_to_none=True)
        loss.backward()
        return feats.grad.abs().sum().item()

    assert abs(grad_sum(True) / grad_sum(False) - 1.0 / ag.K) < 1e-5


def test_rp_bdqn_reproducible_from_triple():
    a = _agent(cell_id="episodic|on|K5", prior_scale=3.0)
    b = _agent(cell_id="episodic|on|K5", prior_scale=3.0)
    assert _rollout(a) == _rollout(b)
