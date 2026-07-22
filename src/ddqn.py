"""ε-greedy Double DQN — the reference method (Part A baseline, Part B method 1).

Double DQN (van Hasselt et al. 2016): the **online** network selects the greedy next
action and the **target** network evaluates it, decoupling selection from evaluation to
reduce the maximization bias of vanilla DQN. This is the baseline the structured partial
factorial extends and the ride-along reference in the DeepSea confirmatory sweep
(see ``protocol/preregistration.md`` freeze items 1, 15).

Everything random is stream-derived (gate C1), never global:

* network initialization → ``init`` stream (owned by :class:`~src.networks.MLPQNetwork`);
* minibatch sampling → ``replay`` stream (owned by :class:`~src.replay_buffer.ReplayBuffer`);
* ε-greedy exploration draws → ``action_noise`` stream (owned here).

This is the single-head (``n_heads=1``) use of ``MLPQNetwork``. Keeping selection,
target construction, and the update on one code path shared with the ensemble is what
makes gate C11 (per-contrast code-path purity) checkable: Bootstrapped-DQN will be the
same agent with ``n_heads=K`` and a head-masked loss, not a fork.

**No frozen numeric values are baked in.** Learning rate, γ, batch size, buffer capacity,
target-update cadence, and hidden width are Class-1 backbone nuisances supplied through
:class:`DDQNConfig` by the caller; the ε schedule is a Class-3 factor-specific parameter
tuned at the Session-3 mini-search. Defaults here are clearly-labelled development
placeholders, not the protocol values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.networks import MLPQNetwork
from src.replay_buffer import ReplayBuffer, Transition
from src.utils import conventions


@dataclass
class DDQNConfig:
    """Class-1 backbone nuisances + the baseline ε schedule (all development placeholders).

    Every field is passed by the caller from the resolved run config. The values below
    are development placeholders that let a smoke run execute; the frozen tuned values
    are filled at the Session-3 backbone mini-search and never live in source.
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
    # ε schedule (Class-3; placeholder linear decay): eps(t) = max(eps_end,
    # eps_start - (eps_start-eps_end) * t / eps_decay_steps).
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 10_000


class DDQNAgent:
    """ε-greedy Double DQN agent with hard target-network updates.

    Args:
        config: the :class:`DDQNConfig` backbone/schedule parameters.
        master_seed: the run's master seed.
        cell_id: the factorial arm / cell id (e.g. ``"ddqn_egreedy"`` reference, or an
            ``"episodic|off|K10"``-style arm) — the RNG-derivation ``cell_id``.
        seed_index: bookkeeping seed label for this run.
        device: torch device (CPU for this project; asserted CPU-only in CI).
    """

    def __init__(
        self,
        config: DDQNConfig,
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

        # init stream -> reproducible network parameters (no global torch RNG).
        init_gen = conventions.derive_torch_generator(
            self.master_seed, self.cell_id, "init", self.seed_index
        )
        self.online = MLPQNetwork(
            config.obs_dim,
            config.n_actions,
            config.hidden_sizes,
            n_heads=1,
            generator=init_gen,
        ).to(self.device)
        self.target = MLPQNetwork(
            config.obs_dim, config.n_actions, config.hidden_sizes, n_heads=1
        ).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.lr)

        # replay stream -> reproducible minibatch draws.
        replay_rng = conventions.derive_numpy_generator(
            self.master_seed, self.cell_id, "replay", self.seed_index
        )
        self.buffer = ReplayBuffer(
            config.buffer_capacity, (config.obs_dim,), replay_rng
        )

        # action_noise stream -> reproducible ε-greedy exploration draws.
        self._action_rng = conventions.derive_numpy_generator(
            self.master_seed, self.cell_id, "action_noise", self.seed_index
        )

        self._learn_steps = 0

    # ------------------------------------------------------------------ #
    # Action selection
    # ------------------------------------------------------------------ #
    def epsilon(self, step: int) -> float:
        """Linearly-decayed exploration rate at environment ``step`` (placeholder schedule)."""
        cfg = self.cfg
        frac = min(1.0, step / max(1, cfg.eps_decay_steps))
        return float(cfg.eps_start - (cfg.eps_start - cfg.eps_end) * frac)

    @torch.no_grad()
    def greedy_action(self, obs: np.ndarray) -> int:
        """argmax_a Q(obs, a) under the online net (ties by lowest action index)."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.online(obs_t)[:, 0, :]  # [1, n_actions] — single head
        return int(torch.argmax(q, dim=1).item())

    def select_action(self, obs: np.ndarray, step: int) -> int:
        """ε-greedy: explore with prob ε(step) via the ``action_noise`` stream, else greedy."""
        if self._action_rng.random() < self.epsilon(step):
            return int(self._action_rng.integers(0, self.cfg.n_actions))
        return self.greedy_action(obs)

    # ------------------------------------------------------------------ #
    # Learning
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
        """Double-DQN target: online net selects a', target net evaluates it."""
        with torch.no_grad():
            next_online = self.online(batch.next_obs)[:, 0, :]        # [B, A]
            next_actions = torch.argmax(next_online, dim=1, keepdim=True)  # [B, 1]
            next_target = self.target(batch.next_obs)[:, 0, :]        # [B, A]
            next_q = next_target.gather(1, next_actions).squeeze(1)   # [B]
            return batch.reward + self.cfg.gamma * (1.0 - batch.done) * next_q

    def learn_step(self) -> float | None:
        """One gradient step on a replay minibatch; returns the TD loss (or ``None`` if not ready).

        Also performs the periodic hard target-network sync. Returns ``None`` until the
        buffer holds ``min_buffer`` transitions, so callers can log "learning started".
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
        target = self._td_target(batch)
        q = self.online(batch.obs)[:, 0, :]                          # [B, A]
        q_taken = q.gather(1, batch.action.unsqueeze(1)).squeeze(1)  # [B]
        loss = nn.functional.smooth_l1_loss(q_taken, target)         # Huber (DQN standard)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self._learn_steps += 1
        if self._learn_steps % self.cfg.target_update_period == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())
