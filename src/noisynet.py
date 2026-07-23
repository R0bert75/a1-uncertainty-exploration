"""NoisyNet-DQN — parametric-noise exploration baseline (Part B method 2).

NoisyNet (Fortunato et al. 2018, arXiv:1706.10295): replace the ε-greedy schedule with
learnable Gaussian noise injected into the network's linear layers. Each weight carries a
learnable mean ``μ`` and standard deviation ``σ``; the effective weight is ``μ + σ ⊙ ε``
with ``ε`` a fixed sample resampled between forward passes. The agent explores because the
noisy weights induce a distribution over greedy policies, and it *learns how much* to
explore — driving ``σ`` down where the value estimate is confident — with no hand-tuned
schedule. This is baseline method (2) in the MinAtar four (see freeze item 15) and rides
along with DDQN in the DeepSea confirmatory sweep (Reviewer Fix 3).

Everything below the noise mechanism is the **same Double-DQN code path** as
:class:`~src.ddqn.DDQNAgent` — the same :class:`~src.replay_buffer.ReplayBuffer`, the same
online-selects / target-evaluates TD target, the same Huber loss and hard target sync —
so what differs between this baseline and the reference is exactly the exploration
mechanism, not the learning machinery.

Randomness is stream-derived (gate C1), never global:

* network parameter init (μ/σ) → ``init`` stream (owned by :class:`~src.networks.NoisyMLPQNetwork`);
* minibatch sampling → ``replay`` stream (owned by :class:`~src.replay_buffer.ReplayBuffer`);
* operational exploration/training noise ε → ``action_noise`` stream (owned here);
* the **measurement-only** M-sample value diagnostic → ``noisynet_diag`` stream (owned here).

The operational and diagnostic noise draw from *separate* generators, so measuring the
value distribution on the probe set never perturbs the training trajectory — the M = 30
i.i.d. draws at measurement time (freeze item 14) are drawn off ``noisynet_diag`` alone.

**No frozen tuned value is baked in.** Learning rate, γ, batch size, buffer capacity, and
target cadence are Class-1 backbone nuisances; ``sigma0`` (the NoisyNet noise-init scale) is
this method's tunable, defaulting to the Fortunato factorized value 0.5 as a labelled
development placeholder. Only ``DIAG_SAMPLES`` = 30 is a frozen *measurement convention*
(item 14), not a run-defining parameter, so it lives here as a named constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.networks import NoisyMLPQNetwork
from src.replay_buffer import ReplayBuffer, Transition
from src.utils import conventions

# Frozen measurement convention (preregistration freeze item 14): the NoisyNet value
# diagnostic draws M = 30 i.i.d. noise samples on the probe set, at measurement only.
DIAG_SAMPLES = 30


@dataclass
class NoisyNetConfig:
    """Class-1 backbone nuisances + the NoisyNet noise-init scale (development placeholders).

    Every field is passed by the caller from the resolved run config. The values below are
    development placeholders that let a smoke run execute; the frozen tuned values are filled
    at the Session-3 search and never live in source. ``sigma0`` is NoisyNet's own tunable
    (analogous to the DDQN ε schedule), defaulting to Fortunato et al.'s factorized 0.5.
    """

    obs_dim: int
    n_actions: int
    hidden_sizes: tuple[int, ...] = (64, 64)
    lr: float = 5e-4
    gamma: float = 0.99
    batch_size: int = 32
    buffer_capacity: int = 100_000
    min_buffer: int = 1_000          # transitions before learning starts
    target_update_period: int = 500  # env steps between hard target syncs
    sigma0: float = 0.5              # NoisyNet noise-init scale (Class-3 placeholder)

    def __post_init__(self) -> None:
        if self.sigma0 <= 0.0:
            raise ValueError(f"sigma0 must be a positive float, got {self.sigma0}")


class NoisyNetAgent:
    """NoisyNet Double-DQN agent (parametric-noise exploration, hard target updates).

    Args:
        config: the :class:`NoisyNetConfig` backbone/noise parameters.
        master_seed: the run's master seed.
        cell_id: the RNG-derivation cell id (e.g. the ``"noisynet"`` reference arm).
        seed_index: bookkeeping seed label for this run.
        device: torch device (CPU for this project).
    """

    def __init__(
        self,
        config: NoisyNetConfig,
        *,
        master_seed: int,
        cell_id: str,
        seed_index: int,
        device: str | torch.device = "cpu",
    ) -> None:
        self.cfg = config
        self.master_seed = int(master_seed)
        self.cell_id = str(cell_id)
        self.seed_index = int(seed_index)
        self.device = torch.device(device)

        # init stream -> reproducible μ/σ (no global torch RNG).
        init_gen = conventions.derive_torch_generator(
            self.master_seed, self.cell_id, "init", self.seed_index
        )
        self.online = NoisyMLPQNetwork(
            config.obs_dim,
            config.n_actions,
            config.hidden_sizes,
            sigma0=config.sigma0,
            generator=init_gen,
        ).to(self.device)
        self.target = NoisyMLPQNetwork(
            config.obs_dim, config.n_actions, config.hidden_sizes, sigma0=config.sigma0
        ).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.lr)

        # replay stream -> reproducible minibatch draws.
        replay_rng = conventions.derive_numpy_generator(
            self.master_seed, self.cell_id, "replay", self.seed_index
        )
        self.buffer = ReplayBuffer(config.buffer_capacity, (config.obs_dim,), replay_rng)

        # action_noise stream -> operational (exploration + training) noise, drawn as torch.
        self._noise_gen = conventions.derive_torch_generator(
            self.master_seed, self.cell_id, "action_noise", self.seed_index
        )
        # noisynet_diag stream -> measurement-only value-diagnostic noise (never touches training).
        self._diag_gen = conventions.derive_torch_generator(
            self.master_seed, self.cell_id, "noisynet_diag", self.seed_index
        )

        self._learn_steps = 0

    # ------------------------------------------------------------------ #
    # Action selection — exploration IS the parameter noise (no ε).
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def greedy_action(self, obs: np.ndarray) -> int:
        """argmax_a Q(obs, a) under the current noisy online net (ties by lowest index)."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.online(obs_t)  # [1, n_actions], noisy
        return int(torch.argmax(q, dim=1).item())

    def select_action(self, obs: np.ndarray, step: int = 0) -> int:
        """Resample the online net's noise, then act greedily under it (``step`` unused)."""
        self.online.reset_noise(self._noise_gen)
        return self.greedy_action(obs)

    # ------------------------------------------------------------------ #
    # Learning — same Double-DQN update as DDQN, with per-step noise resamples.
    # ------------------------------------------------------------------ #
    def observe(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in replay."""
        self.buffer.add(obs, action, reward, next_obs, done)

    def _td_target(self, batch: Transition) -> torch.Tensor:
        """Double-DQN target under the current noise: online selects a', target evaluates it."""
        with torch.no_grad():
            next_online = self.online(batch.next_obs)                     # [B, A] noisy
            next_actions = torch.argmax(next_online, dim=1, keepdim=True)  # [B, 1]
            next_target = self.target(batch.next_obs)                     # [B, A] noisy
            next_q = next_target.gather(1, next_actions).squeeze(1)        # [B]
            return batch.reward + self.cfg.gamma * (1.0 - batch.done) * next_q

    def learn_step(self) -> float | None:
        """One gradient step on a replay minibatch; returns the TD loss (``None`` if not ready).

        Resamples the online and target nets' noise once per update (Fortunato et al. 2018),
        then runs the standard Double-DQN / Huber update and the periodic hard target sync.
        """
        if len(self.buffer) < max(self.cfg.min_buffer, self.cfg.batch_size):
            return None
        batch = self.buffer.sample(self.cfg.batch_size)
        batch = Transition(
            obs=batch.obs.to(self.device, torch.float32),
            action=batch.action.to(self.device),
            reward=batch.reward.to(self.device),
            next_obs=batch.next_obs.to(self.device, torch.float32),
            done=batch.done.to(self.device),
        )
        # Fresh operational noise for this update (online used for both Q(s,a) and a').
        self.online.reset_noise(self._noise_gen)
        self.target.reset_noise(self._noise_gen)

        target = self._td_target(batch)
        q = self.online(batch.obs)                                   # [B, A] noisy
        q_taken = q.gather(1, batch.action.unsqueeze(1)).squeeze(1)  # [B]
        loss = nn.functional.smooth_l1_loss(q_taken, target)         # Huber (DQN standard)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self._learn_steps += 1
        if self._learn_steps % self.cfg.target_update_period == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    # ------------------------------------------------------------------ #
    # Measurement-only value diagnostic (M i.i.d. noise draws, off noisynet_diag).
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample_q_values(self, obs: np.ndarray, m: int = DIAG_SAMPLES) -> torch.Tensor:
        """M i.i.d. noisy Q-value samples for a probe observation ``[m, n_actions]``.

        Draws each sample's noise from the dedicated ``noisynet_diag`` generator, so the
        M = 30 measurement draws (freeze item 14) never advance the operational noise stream
        and cannot perturb the training trajectory. Used to build the value distribution that
        makes NoisyNet comparable to the ensemble's M = K per-head samples.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        samples = []
        for _ in range(int(m)):
            self.online.reset_noise(self._diag_gen)
            samples.append(self.online(obs_t))  # [n_obs, n_actions]
        # For a single probe obs, squeeze the obs axis -> [m, n_actions].
        stacked = torch.stack(samples, dim=0)
        return stacked[:, 0, :] if stacked.shape[1] == 1 else stacked

    @torch.no_grad()
    def mean_action(self, obs: np.ndarray) -> int:
        """Greedy action under the *mean* (μ-only, noise-free) net — for frozen-policy eval."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.online(obs_t, noisy=False)
        return int(torch.argmax(q, dim=1).item())
