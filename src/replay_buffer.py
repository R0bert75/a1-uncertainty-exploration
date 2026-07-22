"""Experience replay buffer shared by every value-based method.

A fixed-capacity ring buffer of ``(obs, action, reward, next_obs, done)`` transitions.
Uniform sampling is driven by a **numpy Generator seeded off the run's ``replay``
stream** (``conventions.derive_numpy_generator(..., "replay", ...)``), so the sequence of
sampled minibatches is reproducible from ``(master_seed, cell_id, seed_index)`` alone
(gate C1) and never touches global RNG.

This class is method-agnostic on purpose (gate C11 code-path purity): Double-DQN and
Bootstrapped-DQN draw the *same* minibatch indices from the *same* ``replay`` stream. The
per-head bootstrap masks that make the ensemble a bootstrap — Ber(0.5) inclusion per
transition per head (Osband et al. 2018 §3.1, a frozen Class-2 value) — are a separate
concern owned by the ensemble agent and drawn from the distinct ``bootstrap_mask``
stream; they are deliberately **not** implemented here so the buffer stays identical
across methods. No frozen numeric values live in this file.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Transition:
    """A sampled minibatch as torch tensors (float obs/reward, long actions, float done)."""

    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor


class ReplayBuffer:
    """Fixed-capacity uniform experience replay.

    Args:
        capacity: maximum stored transitions (Class-1 backbone nuisance; passed in).
        obs_shape: shape of a single observation (e.g. ``(obs_dim,)`` for flattened
            DeepSea/MinAtar inputs).
        rng: numpy ``Generator`` for the ``replay`` stream. Owned by the caller so the
            buffer never creates its own entropy source.
        obs_dtype: storage dtype for observations (default ``float32``).
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: Sequence[int],
        rng: np.random.Generator,
        *,
        obs_dtype: np.dtype | type = np.float32,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = int(capacity)
        self.obs_shape = tuple(int(d) for d in obs_shape)
        self._rng = rng

        self._obs = np.zeros((self.capacity, *self.obs_shape), dtype=obs_dtype)
        self._next_obs = np.zeros((self.capacity, *self.obs_shape), dtype=obs_dtype)
        self._action = np.zeros((self.capacity,), dtype=np.int64)
        self._reward = np.zeros((self.capacity,), dtype=np.float32)
        self._done = np.zeros((self.capacity,), dtype=np.float32)

        self._pos = 0      # next write index (ring)
        self._size = 0     # number of valid transitions

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition, overwriting the oldest when at capacity."""
        i = self._pos
        self._obs[i] = obs
        self._next_obs[i] = next_obs
        self._action[i] = action
        self._reward[i] = reward
        self._done[i] = float(done)
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def can_sample(self, batch_size: int) -> bool:
        return self._size >= batch_size

    def sample_indices(self, batch_size: int) -> np.ndarray:
        """Draw ``batch_size`` uniform row indices (with replacement) via the ``replay`` stream.

        The single deterministic ``rng.integers`` draw that defines a minibatch. Exposed so
        the ensemble agent can select the *same* transitions and then look up their stored
        per-head bootstrap masks — without the buffer needing to know about masks (gate
        C11: identical sampling code path across methods).
        """
        if not self.can_sample(batch_size):
            raise ValueError(
                f"cannot sample {batch_size} from a buffer of size {self._size}"
            )
        return self._rng.integers(0, self._size, size=batch_size)

    def gather(self, idx: np.ndarray) -> Transition:
        """Materialize the transitions at ``idx`` as torch tensors (no RNG draw)."""
        return Transition(
            obs=torch.as_tensor(self._obs[idx]),
            action=torch.as_tensor(self._action[idx]),
            reward=torch.as_tensor(self._reward[idx]),
            next_obs=torch.as_tensor(self._next_obs[idx]),
            done=torch.as_tensor(self._done[idx]),
        )

    def sample(self, batch_size: int) -> Transition:
        """Draw a uniform minibatch (with replacement) via the ``replay`` stream.

        Sampling with replacement is the standard DQN convention. Equivalent to
        ``gather(sample_indices(batch_size))`` — one deterministic ``rng.integers`` call —
        so DDQN and the ensemble consume the ``replay`` stream identically.
        """
        return self.gather(self.sample_indices(batch_size))
