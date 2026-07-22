"""Bootstrapped DQN — the ensemble method (Part A factorial; Part B method 3).

Bootstrapped DQN (Osband et al. 2016, *Deep Exploration via Bootstrapped DQN*,
arXiv:1602.04621) maintains ``K`` value heads on a shared feature trunk, each trained on
its own bootstrap of the replay data, and explores by **posterior sampling** — acting
greedily with respect to a sampled head rather than with an ε-greedy dither. It is the
same agent as the ε-greedy Double DQN baseline with ``n_heads=K`` and a head-masked loss,
**not a fork** — which is exactly what gate C11 (per-contrast code-path purity) checks:
the network (:class:`~src.networks.MLPQNetwork`), the replay buffer
(:class:`~src.replay_buffer.ReplayBuffer`), and the Double-DQN target construction are the
identical code paths the baseline uses; the ensemble adds only (a) per-head bootstrap
masks and (b) a use-rule for turning K heads into one action.

Frozen Class-2 mechanics (valued/cited in ``protocol/preregistration.md``; verified against
the source papers 2026-07-21):

* **Bootstrap mask** — each transition included in head *k*'s replay with probability
  ``mask_prob`` **independently per head**, fixed when the transition is stored
  (Osband et al. **2018** §3.1, the 50-50 ensemble buffer the DeepSea setting uses; the
  2016 Atari ``p=1`` compute compromise is *not* adopted). Drawn from the distinct
  ``bootstrap_mask`` stream.
* **Head-loss aggregation** — per-head TD losses are independent (each head's gradient
  modulated by its own mask); the gradients the K heads push **into the shared trunk are
  normalized by 1/K** (Osband et al. **2016** §6.1 / App. D.2). Realized with a backward
  hook on the shared trunk features, so head parameters keep their own unscaled gradient.
* **Per-head target networks** — head Qₖ(·;θ) trains against its own target Qₖ(·;θ⁻);
  realized as the K heads of one target ``MLPQNetwork`` synced at the shared
  target-update cadence (a Class-1 backbone parameter).
* **Head initialization is the diversity prior** — each head is initialized with
  independent random parameters (owned by ``MLPQNetwork.reset_parameters``, drawn head by
  head from the ``init`` stream).
* **Randomized prior functions (RP-BDQN, the ``prior=on`` cells)** — when ``prior_scale``
  is set, each head additionally carries a fixed, untrainable random prior ``p_k`` and the
  value function is ``Q_k = f_k + β·p_k`` with ``β = prior_scale`` (Osband et al. **2018**
  §3 / Alg. 1). The prior is a second K-head network drawn from the same ``init`` stream
  immediately after the trainable net (an independent random function), frozen and never
  optimized, and the SAME prior is shared by the online and target nets. Because it is a
  detached constant it shifts the predicted Q but contributes no gradient — the 1/K trunk
  normalization and every gradient path are identical to the plain ensemble. ``prior=off``
  (``prior_scale=None``) builds no prior and draws nothing extra, so those cells are
  bit-for-bit identical to plain Bootstrapped DQN. RP-BDQN is thus the SAME agent as the
  bootstrap ensemble plus one fixed additive term — the gate-C11 code-path-purity claim
  extends to the prior on/off contrast.

Use rules (how K heads become one action; the Part A ``use_rule`` factor):

* ``episodic``  — sample one head uniformly at each episode start, act greedily w.r.t. it
  for the whole episode (temporally coherent deep exploration; the pre-registered rule).
* ``per_step``  — sample a fresh head every step (the temporal-coherence comparator).
* ``ensemble_mean`` — act ε-greedy on the mean Q across heads (the capacity-matched
  ε-greedy comparator; the only use-rule that uses ε).

All exploration randomness (head sampling, and ε for ``ensemble_mean``) is drawn from the
``action_noise`` stream. **No frozen numeric value is baked in**: K, the backbone
nuisances, and the ε schedule come from the config; ``mask_prob`` and ``head_loss_agg``
default to clearly-labelled development placeholders (the confirmatory loader refuses to
let a confirmatory run inherit them).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.networks import MLPQNetwork
from src.replay_buffer import ReplayBuffer, Transition
from src.utils import conventions

VALID_USE_RULES = ("episodic", "per_step", "ensemble_mean")
VALID_HEAD_LOSS_AGG = ("grad_norm_1_over_k",)


@dataclass
class BDQNConfig:
    """Backbone nuisances (Class 1) + ensemble parameters (Class 2) + ε schedule (Class 3).

    Backbone fields mirror ``DDQNConfig`` (same shared code path). ``K``, ``use_rule``,
    ``mask_prob`` and ``head_loss_agg`` are the ensemble parameters; ``mask_prob`` and
    ``head_loss_agg`` default to development placeholders (their frozen values live in the
    protocol, not here). The ε schedule applies only to ``use_rule='ensemble_mean'``.
    """

    obs_dim: int
    n_actions: int
    K: int = 10
    use_rule: str = "episodic"
    hidden_sizes: tuple[int, ...] = (64, 64)
    lr: float = 5e-4
    gamma: float = 0.99
    batch_size: int = 32
    buffer_capacity: int = 100_000
    min_buffer: int = 1_000
    target_update_period: int = 500
    # Class-2 ensemble parameters (development placeholders; pinned in protocol).
    mask_prob: float = 0.5
    head_loss_agg: str = "grad_norm_1_over_k"
    # Class-3 factor-specific tunables. ``prior_scale`` turns the plain bootstrap ensemble
    # into RP-BDQN (randomized prior functions, Osband 2018): ``None`` = prior=off (plain
    # Bootstrapped DQN, no prior); a positive float = prior=on with that β scale. The ε
    # schedule is only used by use_rule='ensemble_mean' (placeholder linear decay).
    prior_scale: float | None = None
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 10_000

    def __post_init__(self) -> None:
        if self.use_rule not in VALID_USE_RULES:
            raise ValueError(f"use_rule must be one of {VALID_USE_RULES}, got {self.use_rule!r}")
        if self.head_loss_agg not in VALID_HEAD_LOSS_AGG:
            raise ValueError(
                f"head_loss_agg must be one of {VALID_HEAD_LOSS_AGG}, got {self.head_loss_agg!r}"
            )
        if self.K < 1:
            raise ValueError(f"K must be >= 1, got {self.K}")
        if not 0.0 < self.mask_prob <= 1.0:
            raise ValueError(f"mask_prob must be in (0, 1], got {self.mask_prob}")
        if self.prior_scale is not None and self.prior_scale <= 0.0:
            raise ValueError(
                f"prior_scale must be a positive float when set (prior=on), got {self.prior_scale}"
            )


class BDQNAgent:
    """Bootstrapped DQN with K heads, per-head bootstrap masks, and posterior-sampling.

    Args mirror :class:`~src.ddqn.DDQNAgent`: ``config`` plus the RNG-derivation triple
    ``(master_seed, cell_id, seed_index)`` and a ``device``. Every random draw is derived
    from a named stream — ``init`` (network parameters), ``replay`` (minibatch indices),
    ``bootstrap_mask`` (per-head masks), ``action_noise`` (head sampling + ε) — so the run
    reproduces bit-for-bit from the triple alone (gate C1).
    """

    def __init__(
        self,
        config: BDQNConfig,
        *,
        master_seed: int,
        cell_id: str,
        seed_index: int,
        device: str | torch.device = "cpu",
    ) -> None:
        self.cfg = config
        self.K = int(config.K)
        self.master_seed = int(master_seed)
        self.cell_id = str(cell_id)
        self.seed_index = int(seed_index)
        self.device = torch.device(device)

        # init stream -> reproducible K-head network (each head independently initialized).
        init_gen = conventions.derive_torch_generator(
            self.master_seed, self.cell_id, "init", self.seed_index
        )
        self.online = MLPQNetwork(
            config.obs_dim, config.n_actions, config.hidden_sizes,
            n_heads=self.K, generator=init_gen,
        ).to(self.device)
        self.target = MLPQNetwork(
            config.obs_dim, config.n_actions, config.hidden_sizes, n_heads=self.K,
        ).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        # Randomized prior function (RP-BDQN, Osband et al. 2018 §3 / Alg. 1). When
        # prior_scale is set, every head carries a FIXED, UNTRAINABLE random prior p_k drawn
        # from a K-head network; the value function is the trainable net plus β·p_k, with the
        # SAME prior shared by the online and target nets (only the trainable part differs).
        # The prior network is drawn from the SAME init generator, immediately AFTER the
        # online net, so it is an independent random function and the RNG sequence for the
        # trainable net is byte-identical to the plain (prior=off) ensemble. It is frozen
        # (no grad) and never handed to the optimizer, so the 1/K trunk hook and every
        # gradient path are unchanged. prior=off (prior_scale is None) builds no prior and
        # draws nothing extra — those cells are bit-for-bit identical to plain Bootstrapped DQN.
        self.prior_scale = 0.0 if config.prior_scale is None else float(config.prior_scale)
        self.prior: MLPQNetwork | None = None
        if config.prior_scale is not None:
            self.prior = MLPQNetwork(
                config.obs_dim, config.n_actions, config.hidden_sizes,
                n_heads=self.K, generator=init_gen,
            ).to(self.device)
            self.prior.eval()
            for p in self.prior.parameters():
                p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.lr)

        # replay stream -> reproducible minibatch draws (same buffer/stream as DDQN).
        replay_rng = conventions.derive_numpy_generator(
            self.master_seed, self.cell_id, "replay", self.seed_index
        )
        self.buffer = ReplayBuffer(config.buffer_capacity, (config.obs_dim,), replay_rng)

        # bootstrap_mask stream -> per-transition-per-head Ber(mask_prob) masks. The mask
        # ring mirrors the replay buffer's ring position for position exactly (adds are in
        # lockstep), so buffer index i always lines up with mask row i.
        self._mask_rng = conventions.derive_numpy_generator(
            self.master_seed, self.cell_id, "bootstrap_mask", self.seed_index
        )
        self._masks = np.zeros((config.buffer_capacity, self.K), dtype=np.float32)
        self._mask_pos = 0

        # action_noise stream -> head sampling (episodic/per_step) and ε (ensemble_mean).
        self._action_rng = conventions.derive_numpy_generator(
            self.master_seed, self.cell_id, "action_noise", self.seed_index
        )

        self._learn_steps = 0
        self._active_head = self._sample_head()  # for episodic: fixed until next episode

    # ------------------------------------------------------------------ #
    # Posterior-sampling exploration
    # ------------------------------------------------------------------ #
    def _sample_head(self) -> int:
        """Uniformly sample a head index from the ``action_noise`` stream."""
        return int(self._action_rng.integers(0, self.K))

    def on_episode_start(self) -> None:
        """Resample the active head at an episode boundary (used by ``use_rule='episodic'``)."""
        self._active_head = self._sample_head()

    def epsilon(self, step: int) -> float:
        """Linearly-decayed ε (only used by ``use_rule='ensemble_mean'``; placeholder schedule)."""
        cfg = self.cfg
        frac = min(1.0, step / max(1, cfg.eps_decay_steps))
        return float(cfg.eps_start - (cfg.eps_start - cfg.eps_end) * frac)

    def _prior_term(self, obs: torch.Tensor) -> torch.Tensor | float:
        """The fixed prior contribution ``β·p(obs)`` (detached), or ``0.0`` when prior=off.

        Shape matches the network output ``[..., K, A]``. Always evaluated under no_grad —
        the prior is untrainable, so it is a constant added to the trainable value function
        and never participates in a gradient (Osband et al. 2018 §3, Alg. 1).
        """
        if self.prior is None:
            return 0.0
        with torch.no_grad():
            return self.prior_scale * self.prior(obs)

    def _q_online(self, obs: torch.Tensor) -> torch.Tensor:
        """Trainable online net + prior (grad flows only through the trainable part)."""
        return self.online(obs) + self._prior_term(obs)

    def _q_target(self, obs: torch.Tensor) -> torch.Tensor:
        """Target net + the SAME prior (only the trainable target params differ)."""
        return self.target(obs) + self._prior_term(obs)

    @torch.no_grad()
    def _q_all(self, obs: np.ndarray) -> torch.Tensor:
        """Per-head Q-values for a single observation, shape ``[K, n_actions]``."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self._q_online(obs_t)[0]  # [K, A]

    def greedy_action_of_head(self, obs: np.ndarray, head: int) -> int:
        """argmax_a Qₕₑₐd(obs, a) (ties → lowest action index)."""
        q = self._q_all(obs)[head]  # [A]
        return int(torch.argmax(q).item())

    def select_action(self, obs: np.ndarray, step: int) -> int:
        """Choose an action under the configured use-rule.

        ``episodic``  → greedy w.r.t. the episode's fixed active head;
        ``per_step``  → greedy w.r.t. a freshly sampled head;
        ``ensemble_mean`` → ε-greedy (``action_noise``) on the mean Q across heads.
        """
        rule = self.cfg.use_rule
        if rule == "episodic":
            return self.greedy_action_of_head(obs, self._active_head)
        if rule == "per_step":
            return self.greedy_action_of_head(obs, self._sample_head())
        # ensemble_mean: ε-greedy on the mean over heads.
        if self._action_rng.random() < self.epsilon(step):
            return int(self._action_rng.integers(0, self.cfg.n_actions))
        q_mean = self._q_all(obs).mean(dim=0)  # [A]
        return int(torch.argmax(q_mean).item())

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #
    def observe(
        self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool
    ) -> None:
        """Store a transition and draw its fixed per-head bootstrap mask (Ber(mask_prob))."""
        self.buffer.add(obs, action, reward, next_obs, done)
        mask = (self._mask_rng.random(self.K) < self.cfg.mask_prob).astype(np.float32)
        self._masks[self._mask_pos] = mask
        self._mask_pos = (self._mask_pos + 1) % self.cfg.buffer_capacity

    def _per_head_td_target(self, batch: Transition) -> torch.Tensor:
        """Per-head Double-DQN target ``[B, K]``: online head selects a', target head evaluates."""
        with torch.no_grad():
            next_online = self._q_online(batch.next_obs)           # [B, K, A] (+prior)
            next_actions = torch.argmax(next_online, dim=2, keepdim=True)  # [B, K, 1]
            next_target = self._q_target(batch.next_obs)           # [B, K, A] (+same prior)
            next_q = next_target.gather(2, next_actions).squeeze(2)  # [B, K]
            reward = batch.reward.unsqueeze(1)                     # [B, 1]
            done = batch.done.unsqueeze(1)                         # [B, 1]
            return reward + self.cfg.gamma * (1.0 - done) * next_q  # [B, K]

    def learn_step(self) -> float | None:
        """One masked, 1/K-trunk-normalized gradient step; returns total loss or ``None``.

        Returns ``None`` until the buffer holds ``min_buffer`` transitions. Draws the
        minibatch indices from the ``replay`` stream (identical to DDQN) and looks up the
        stored per-head masks for those same transitions.
        """
        if len(self.buffer) < max(self.cfg.min_buffer, self.cfg.batch_size):
            return None
        idx = self.buffer.sample_indices(self.cfg.batch_size)
        raw = self.buffer.gather(idx)
        batch = Transition(
            obs=raw.obs.to(self.device, torch.float32),
            action=raw.action.to(self.device),
            reward=raw.reward.to(self.device),
            next_obs=raw.next_obs.to(self.device, torch.float32),
            done=raw.done.to(self.device),
        )
        masks = torch.as_tensor(self._masks[idx], device=self.device)  # [B, K]

        target = self._per_head_td_target(batch)                       # [B, K]

        # Shared forward with a 1/K gradient hook on the trunk features: the K heads'
        # gradients into the shared trunk are averaged (Osband 2016 §6.1), while each
        # head's own parameters keep their unscaled gradient.
        features = self.online.trunk_features(batch.obs)               # [B, F]
        if features.requires_grad:
            features.register_hook(lambda g: g / self.K)
        # Trainable head outputs + the fixed prior (a detached constant, so it shifts the
        # predicted Q but contributes no gradient — the 1/K trunk hook is untouched).
        q_all = self.online.heads_forward(features) + self._prior_term(batch.obs)  # [B, K, A]
        action_idx = batch.action.view(-1, 1, 1).expand(-1, self.K, 1)  # [B, K, 1]
        q_taken = q_all.gather(2, action_idx).squeeze(2)               # [B, K]

        # Per-head Huber TD error, each transition modulated by that head's mask. Normalize
        # by the number of included (masked-in) transitions per head so a head trained on
        # fewer transitions is not down-weighted purely by count.
        per_elem = nn.functional.smooth_l1_loss(q_taken, target, reduction="none")  # [B, K]
        masked = per_elem * masks                                      # [B, K]
        denom = masks.sum(dim=0).clamp(min=1.0)                        # [K]
        per_head_loss = masked.sum(dim=0) / denom                      # [K]
        loss = per_head_loss.sum()                                     # sum over heads

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        self._learn_steps += 1
        if self._learn_steps % self.cfg.target_update_period == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())
