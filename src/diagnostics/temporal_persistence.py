"""Temporal-persistence diagnostic — frozen battery formula §3.3 #6.

This is the one member of the uncertainty-quality battery that is **pre-freeze-safe
infrastructure**: unlike the other diagnostics it references no ``Q*``, no frozen numeric
constant, and makes no learned-vs-truth value comparison. It is a purely agent-side ratio
computed from the agent's *own* greedy actions, so its reference implementation can be
written and unit-tested now without opening the Session-6 C7 battery gate.

Frozen definition (``protocol/preregistration.md`` §3.3, item 6):

    Temporal persistence: within-episode fraction of probe states whose greedy action under
    the current sample equals the episode-start sample's; per-episode mean, per checkpoint
    (across consecutive samples for per-step rules; ~1 by construction for episodic —
    descriptive).

A "sample" is one draw of the agent's value function:

* **episodic** ensemble rule — the sampled head is fixed for the whole episode, so every
  within-episode current sample *is* the episode-start sample and the persistence is 1.0 by
  construction (the descriptive baseline);
* **per_step** ensemble rule — a fresh head is drawn each step, so consecutive samples differ
  and persistence drops below 1;
* **NoisyNet** — the parameter noise is resampled each step, likewise below 1;
* a deterministic policy (plain DDQN) has a single sample, so persistence is trivially 1.

The two public entry points separate the frozen arithmetic from the agent plumbing:

* :func:`temporal_persistence` — the pure formula over pre-computed greedy-action arrays;
  no torch, no agent, exactly the frozen definition, so it can be checked in isolation.
* :func:`episode_temporal_persistence` — the driver: given a :class:`GreedySampler` (a thin
  adapter over an agent's existing sampling API) and a probe set, it fixes the episode-start
  sample, draws ``n_within_episode_samples`` consecutive current samples, and reduces them
  with the pure formula.

All sampling used by the diagnostic must draw from a *measurement-side* generator, never an
operational RNG stream, so measuring never perturbs the training trajectory (the same
invariant the NoisyNet ``noisynet_diag`` value diagnostic upholds). The adapters below take
an explicit measurement generator to enforce this.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "GreedySampler",
    "temporal_persistence",
    "episode_temporal_persistence",
    "EnsembleHeadSampler",
    "NoisyNetSampler",
]


# --------------------------------------------------------------------------- #
# The frozen formula (agent-agnostic, no torch)
# --------------------------------------------------------------------------- #
def temporal_persistence(
    episode_start_actions: np.ndarray,
    per_step_actions: np.ndarray,
) -> float:
    """Per-episode mean temporal persistence (frozen §3.3 #6).

    Parameters
    ----------
    episode_start_actions:
        Greedy action per probe state under the episode-start sample, shape ``[S]``.
    per_step_actions:
        Greedy action per probe state under each within-episode *current* sample, shape
        ``[T, S]`` for ``T`` consecutive samples. For per-step rules these are successive
        draws; for episodic rules every row equals ``episode_start_actions``.

    Returns
    -------
    float
        Mean over the ``T`` within-episode samples of the *fraction of probe states* whose
        current-sample greedy action equals the episode-start greedy action. In ``[0, 1]``;
        exactly 1.0 when every current sample reproduces the episode-start actions (the
        episodic / deterministic case).
    """
    start = np.asarray(episode_start_actions)
    steps = np.asarray(per_step_actions)
    if start.ndim != 1:
        raise ValueError(f"episode_start_actions must be 1-D [S], got shape {start.shape}")
    if steps.ndim != 2:
        raise ValueError(f"per_step_actions must be 2-D [T, S], got shape {steps.shape}")
    if steps.shape[1] != start.shape[0]:
        raise ValueError(
            f"probe-set size mismatch: per_step_actions has {steps.shape[1]} states, "
            f"episode_start_actions has {start.shape[0]}"
        )
    if steps.shape[0] == 0:
        raise ValueError("per_step_actions must contain at least one within-episode sample")
    # Fraction of probe states matching the episode-start sample, per within-episode step.
    per_step_fraction = (steps == start[None, :]).mean(axis=1)  # [T]
    # Per-episode mean across the consecutive samples.
    return float(per_step_fraction.mean())


# --------------------------------------------------------------------------- #
# Sampler protocol + the driver
# --------------------------------------------------------------------------- #
@runtime_checkable
class GreedySampler(Protocol):
    """Contract the temporal-persistence driver needs from an agent adapter.

    A sampler holds one value-function *sample* at a time. ``resample`` advances to a fresh
    sample using a measurement-side generator (so operational RNG is untouched);
    ``greedy_actions`` reports the greedy action per probe state under the current sample.
    """

    def resample(self) -> None:
        """Draw a fresh value-function sample from the measurement generator."""
        ...

    def greedy_actions(self, probe_states: np.ndarray) -> np.ndarray:
        """Greedy action per probe state under the current sample, shape ``[S]`` (int)."""
        ...


def episode_temporal_persistence(
    sampler: GreedySampler,
    probe_states: np.ndarray,
    n_within_episode_samples: int,
) -> float:
    """Drive one episode of the temporal-persistence diagnostic.

    Fixes the episode-start sample, draws ``n_within_episode_samples`` consecutive current
    samples, and reduces them with :func:`temporal_persistence`.

    Parameters
    ----------
    sampler:
        A :class:`GreedySampler` adapter over the agent under test.
    probe_states:
        Probe set ``S``, shape ``[S, obs_dim]`` (rows are individual observations).
    n_within_episode_samples:
        Number of consecutive within-episode current samples ``T`` (must be ≥ 1). For an
        episodic sampler every current sample reproduces the start sample, giving 1.0.
    """
    if n_within_episode_samples < 1:
        raise ValueError("n_within_episode_samples must be >= 1")
    probes = np.asarray(probe_states)
    if probes.ndim != 2:
        raise ValueError(f"probe_states must be 2-D [S, obs_dim], got shape {probes.shape}")

    # Episode-start sample.
    sampler.resample()
    start_actions = np.asarray(sampler.greedy_actions(probes))

    per_step = np.empty((n_within_episode_samples, probes.shape[0]), dtype=start_actions.dtype)
    for t in range(n_within_episode_samples):
        sampler.resample()
        per_step[t] = sampler.greedy_actions(probes)

    return temporal_persistence(start_actions, per_step)


# --------------------------------------------------------------------------- #
# Thin agent adapters
# --------------------------------------------------------------------------- #
class EnsembleHeadSampler:
    """Temporal-persistence sampler for the Bootstrapped-DQN ensemble agent.

    A "sample" is a head index. Which head each ``resample`` yields is governed by the
    agent's use-rule, exactly as at run time:

    * ``episodic`` — the episode-start head is fixed; ``resample`` keeps it, so every current
      sample equals the start sample and persistence is 1.0 by construction;
    * ``per_step`` — ``resample`` draws a fresh head, so consecutive samples differ.

    Head draws use a caller-supplied *measurement* generator, never the agent's operational
    ``_head_rng``, so running the diagnostic does not disturb the agent's action stream.
    """

    def __init__(self, agent, generator: np.random.Generator, *, use_rule: str | None = None):
        self.agent = agent
        self.generator = generator
        self.use_rule = use_rule if use_rule is not None else agent.cfg.use_rule
        self._head: int | None = None

    def resample(self) -> None:
        if self.use_rule == "episodic":
            # Fixed for the whole episode: draw once, then hold it.
            if self._head is None:
                self._head = int(self.generator.integers(0, self.agent.K))
        else:
            # per_step (and the ensemble_mean comparator): a fresh head each sample.
            self._head = int(self.generator.integers(0, self.agent.K))

    def greedy_actions(self, probe_states: np.ndarray) -> np.ndarray:
        if self._head is None:
            raise RuntimeError("resample() must be called before greedy_actions()")
        return np.array(
            [self.agent.greedy_action_of_head(s, self._head) for s in probe_states],
            dtype=np.int64,
        )


class NoisyNetSampler:
    """Temporal-persistence sampler for the NoisyNet-DQN agent.

    A "sample" is a draw of the network's factorized parameter noise. ``resample`` refreshes
    the *online* net's noise from a caller-supplied measurement generator — never the
    operational ``_noise_gen`` — so the diagnostic does not advance the operational noise
    stream. (Operational steps always ``reset_noise`` from ``_noise_gen`` before acting, so
    the mutated online buffers never leak into the training trajectory — the same invariant
    the M-sample value diagnostic relies on.)
    """

    def __init__(self, agent, generator):
        self.agent = agent
        self.generator = generator

    def resample(self) -> None:
        self.agent.online.reset_noise(self.generator)

    def greedy_actions(self, probe_states: np.ndarray) -> np.ndarray:
        return np.array(
            [self.agent.greedy_action(s) for s in probe_states], dtype=np.int64
        )
