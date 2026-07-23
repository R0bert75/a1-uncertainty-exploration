"""Tests for the temporal-persistence diagnostic (src/diagnostics/temporal_persistence.py).

Battery formula §3.3 #6. This is C7 *groundwork*, not the gate: the diagnostic references no
Q*, no frozen constant, and no learned-vs-truth comparison, so its reference implementation
is pre-freeze-safe. Tests cover (a) the pure frozen formula in isolation, (b) the driver over
a synthetic sampler, (c) the ~1-by-construction episodic ensemble case vs. the <1 per-step
case, (d) the NoisyNet case, and (e) the invariant that measuring never touches an
operational RNG stream.
"""

import numpy as np
import pytest

from src.diagnostics.temporal_persistence import (
    episode_temporal_persistence,
    temporal_persistence,
)

torch = pytest.importorskip("torch")

from src.bdqn import BDQNAgent, BDQNConfig  # noqa: E402
from src.diagnostics.temporal_persistence import (  # noqa: E402
    EnsembleHeadSampler,
    NoisyNetSampler,
)
from src.noisynet import NoisyNetAgent, NoisyNetConfig  # noqa: E402

OBS_DIM = 8
N_ACTIONS = 2


def _probes(n: int = OBS_DIM) -> np.ndarray:
    return np.eye(OBS_DIM, dtype=np.float32)[:n]


# --------------------------------------------------------------------------- #
# The pure frozen formula
# --------------------------------------------------------------------------- #
def test_formula_all_match_is_one():
    start = np.array([0, 1, 0, 1])
    steps = np.tile(start, (5, 1))            # every current sample reproduces the start
    assert temporal_persistence(start, steps) == 1.0


def test_formula_none_match_is_zero():
    start = np.array([0, 0, 0, 0])
    steps = np.ones((3, 4), dtype=int)        # every state disagrees on every sample
    assert temporal_persistence(start, steps) == 0.0


def test_formula_is_mean_of_per_step_fractions():
    start = np.array([0, 1, 0, 1])            # S = 4
    steps = np.array(
        [
            [0, 1, 0, 1],                     # 4/4 match
            [0, 1, 1, 1],                     # 3/4 match
            [1, 0, 1, 0],                     # 0/4 match
        ]
    )
    # per-step fractions [1.0, 0.75, 0.0] -> mean 0.5833...
    assert temporal_persistence(start, steps) == pytest.approx((1.0 + 0.75 + 0.0) / 3)


@pytest.mark.parametrize(
    "start,steps,frag",
    [
        (np.zeros((2, 2)), np.zeros((1, 2)), "1-D"),          # start not 1-D
        (np.zeros(3), np.zeros(3), "2-D"),                    # steps not 2-D
        (np.zeros(3), np.zeros((1, 4)), "mismatch"),          # probe-size mismatch
        (np.zeros(3), np.zeros((0, 3)), "at least one"),      # no within-episode samples
    ],
)
def test_formula_rejects_malformed_input(start, steps, frag):
    with pytest.raises(ValueError) as e:
        temporal_persistence(start, steps)
    assert frag in str(e.value)


# --------------------------------------------------------------------------- #
# The driver over a synthetic sampler
# --------------------------------------------------------------------------- #
class _FixedSampler:
    """Deterministic sampler: always reports the same greedy actions -> persistence 1.0."""

    def __init__(self, actions):
        self.actions = np.asarray(actions)
        self.resamples = 0

    def resample(self):
        self.resamples += 1

    def greedy_actions(self, probe_states):
        return self.actions[: len(probe_states)]


class _CyclingSampler:
    """Alternates between two action vectors on each resample."""

    def __init__(self, a, b):
        self.vectors = [np.asarray(a), np.asarray(b)]
        self.i = -1

    def resample(self):
        self.i += 1

    def greedy_actions(self, probe_states):
        return self.vectors[self.i % 2][: len(probe_states)]


def test_driver_fixed_sampler_is_one_and_resamples_correctly():
    s = _FixedSampler([0, 1, 0, 1])
    val = episode_temporal_persistence(s, _probes(4), n_within_episode_samples=3)
    assert val == 1.0
    assert s.resamples == 1 + 3           # one start sample + three within-episode samples


def test_driver_rejects_bad_args():
    s = _FixedSampler([0, 1])
    with pytest.raises(ValueError):
        episode_temporal_persistence(s, _probes(2), n_within_episode_samples=0)
    with pytest.raises(ValueError):
        episode_temporal_persistence(s, np.zeros(3), n_within_episode_samples=1)  # probes 1-D


def test_driver_cycling_sampler_below_one():
    # start = a; within-episode samples cycle a, b, a -> fractions 1, m(b vs a), 1
    a = [0, 0, 0, 0]
    b = [1, 1, 0, 0]                       # matches a on 2 of 4 states
    s = _CyclingSampler(a, b)
    val = episode_temporal_persistence(s, _probes(4), n_within_episode_samples=3)
    # start sample fixes a; then samples are b, a, b -> fractions 0.5, 1.0, 0.5 -> mean 2/3
    assert 0.0 < val < 1.0
    assert val == pytest.approx((0.5 + 1.0 + 0.5) / 3)


# --------------------------------------------------------------------------- #
# Ensemble agent: episodic ~1 by construction, per_step < 1
# --------------------------------------------------------------------------- #
def _bdqn(use_rule, cell_id):
    cfg = BDQNConfig(
        obs_dim=OBS_DIM, n_actions=N_ACTIONS, K=10, use_rule=use_rule,
        hidden_sizes=(16,), min_buffer=16, batch_size=8,
    )
    return BDQNAgent(cfg, master_seed=0, cell_id=cell_id, seed_index=0)


def test_ensemble_episodic_persistence_is_one_by_construction():
    agent = _bdqn("episodic", "episodic|off|K10")
    gen = np.random.default_rng(12345)
    sampler = EnsembleHeadSampler(agent, gen, use_rule="episodic")
    val = episode_temporal_persistence(sampler, _probes(), n_within_episode_samples=8)
    assert val == 1.0                     # the episode head is fixed -> every sample agrees


def test_ensemble_per_step_persistence_below_one_when_heads_disagree():
    # With freshly initialized (unlearned) heads and K=10, distinct heads generally induce
    # different greedy maps, so a per-step sampler should drop below 1 on some probe states.
    agent = _bdqn("per_step", "per_step|off|K10")
    gen = np.random.default_rng(7)
    sampler = EnsembleHeadSampler(agent, gen, use_rule="per_step")
    val = episode_temporal_persistence(sampler, _probes(), n_within_episode_samples=20)
    assert 0.0 <= val < 1.0


def test_ensemble_sampler_uses_measurement_generator_not_operational_stream():
    # Running the diagnostic must not advance the agent's operational head RNG.
    # The ensemble draws heads from its operational ``action_noise`` generator (_action_rng);
    # the diagnostic must use its own measurement generator and leave that state untouched.
    agent = _bdqn("per_step", "per_step|off|K10")
    before = agent._action_rng.bit_generator.state
    gen = np.random.default_rng(1)
    sampler = EnsembleHeadSampler(agent, gen, use_rule="per_step")
    episode_temporal_persistence(sampler, _probes(), n_within_episode_samples=10)
    after = agent._action_rng.bit_generator.state
    assert before == after


def test_ensemble_persistence_is_reproducible_from_measurement_seed():
    agent = _bdqn("per_step", "per_step|off|K10")
    v1 = episode_temporal_persistence(
        EnsembleHeadSampler(agent, np.random.default_rng(99), use_rule="per_step"),
        _probes(), n_within_episode_samples=15,
    )
    v2 = episode_temporal_persistence(
        EnsembleHeadSampler(agent, np.random.default_rng(99), use_rule="per_step"),
        _probes(), n_within_episode_samples=15,
    )
    assert v1 == v2


# --------------------------------------------------------------------------- #
# NoisyNet agent
# --------------------------------------------------------------------------- #
def _noisynet(cell_id="noisynet"):
    cfg = NoisyNetConfig(
        obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden_sizes=(16,),
        min_buffer=16, batch_size=8, sigma0=0.5,
    )
    return NoisyNetAgent(cfg, master_seed=0, cell_id=cell_id, seed_index=0)


def test_noisynet_persistence_in_unit_interval():
    agent = _noisynet()
    gen = torch.Generator().manual_seed(2024)
    sampler = NoisyNetSampler(agent, gen)
    val = episode_temporal_persistence(sampler, _probes(), n_within_episode_samples=20)
    assert 0.0 <= val <= 1.0


def test_noisynet_diagnostic_does_not_touch_operational_noise_stream():
    # The operational noise generator's state must be identical before and after measuring.
    agent = _noisynet()
    before = agent._noise_gen.get_state()
    meas = torch.Generator().manual_seed(5)
    sampler = NoisyNetSampler(agent, meas)
    episode_temporal_persistence(sampler, _probes(), n_within_episode_samples=12)
    after = agent._noise_gen.get_state()
    assert torch.equal(before, after)


def test_noisynet_persistence_reproducible_from_measurement_generator():
    agent = _noisynet()
    v1 = episode_temporal_persistence(
        NoisyNetSampler(agent, torch.Generator().manual_seed(313)),
        _probes(), n_within_episode_samples=15,
    )
    v2 = episode_temporal_persistence(
        NoisyNetSampler(agent, torch.Generator().manual_seed(313)),
        _probes(), n_within_episode_samples=15,
    )
    assert v1 == v2
