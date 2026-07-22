"""Q-value networks for the value-based methods (Part A DeepSea, Part B MinAtar).

A single ``MLPQNetwork`` backs every method in the study. It is a shared feature
trunk followed by ``n_heads`` **independent** linear value heads that split after the
shared representation — the Osband et al. (2016) §6 / Fig. 1a convention (see the
Class-2 parameter table in ``protocol/preregistration.md``). The single-head case
(``n_heads=1``) is the plain Double-DQN backbone; ``n_heads=K`` is the Bootstrapped-DQN
ensemble. Keeping one class for both is what makes per-contrast code-path purity (gate
C11) checkable: the baseline and the ensemble differ only in ``n_heads`` (and, later, a
prior function), not in a separate implementation.

Parameter initialization is **stream-derived, not global**: parameters are filled from a
``torch.Generator`` seeded off the run's ``init`` stream
(``conventions.derive_torch_generator(..., "init", ...)``), so the same
``(master_seed, cell_id, seed_index)`` reproduces the same initial weights bit-for-bit
(gate C1) without touching the global torch RNG. Each head is drawn independently from
that generator; for the ensemble that independent initialization *is* the diversity prior.

No frozen numeric values live here — width (``hidden_sizes``) is a Class-1 backbone
nuisance tuned at the Session-3 mini-search and is always passed in by the caller.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


def _init_linear_(layer: nn.Linear, generator: torch.Generator) -> None:
    """Reproducibly initialize a ``nn.Linear`` from ``generator`` (no global RNG).

    Matches PyTorch's default ``nn.Linear`` scheme numerically — weight and bias are
    both drawn from ``U(-1/sqrt(fan_in), 1/sqrt(fan_in))`` — but every draw is taken
    from the passed generator so initialization is a function of the ``init`` stream
    alone. ``Tensor.uniform_`` accepts ``generator=`` on the pinned torch build.
    """
    fan_in = layer.weight.shape[1]
    bound = 1.0 / math.sqrt(fan_in)
    with torch.no_grad():
        layer.weight.uniform_(-bound, bound, generator=generator)
        if layer.bias is not None:
            layer.bias.uniform_(-bound, bound, generator=generator)


class MLPQNetwork(nn.Module):
    """Shared MLP trunk + ``n_heads`` independent linear value heads.

    Args:
        obs_dim: flattened observation dimension (DeepSea one-hot row/column, or a
            flattened MinAtar plane stack — the caller flattens before the network).
        n_actions: size of the discrete action space.
        hidden_sizes: trunk widths (Class-1 backbone nuisance; passed in, never frozen
            here). An empty sequence gives a linear (trunk-free) Q-function.
        n_heads: number of independent value heads. ``1`` = Double-DQN backbone;
            ``K`` = Bootstrapped-DQN ensemble.
        generator: seeded ``torch.Generator`` for the ``init`` stream. When ``None`` the
            module keeps torch's default (global-RNG) initialization — only for throwaway
            shape checks; every real run passes the derived generator.

    Forward returns a ``[batch, n_heads, n_actions]`` tensor. Single-head callers squeeze
    the head axis; ensemble callers keep it.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int],
        *,
        n_heads: int = 1,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if obs_dim <= 0 or n_actions <= 0:
            raise ValueError(f"obs_dim and n_actions must be positive, got {obs_dim}, {n_actions}")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be >= 1, got {n_heads}")
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.n_heads = int(n_heads)

        trunk_layers: list[nn.Module] = []
        in_dim = self.obs_dim
        for width in hidden_sizes:
            trunk_layers.append(nn.Linear(in_dim, int(width)))
            trunk_layers.append(nn.ReLU())
            in_dim = int(width)
        self.trunk = nn.Sequential(*trunk_layers)
        self.feature_dim = in_dim
        # K independent heads split after the shared representation (Osband 2016 §6).
        self.heads = nn.ModuleList(
            [nn.Linear(self.feature_dim, self.n_actions) for _ in range(self.n_heads)]
        )

        if generator is not None:
            self.reset_parameters(generator)

    def reset_parameters(self, generator: torch.Generator) -> None:
        """Re-initialize every parameter from ``generator`` (the ``init`` stream).

        Draw order is fixed — trunk layers in order, then each head in order — so the
        initialization is a deterministic function of the seed. Heads are drawn
        sequentially from the one generator, giving each head independent parameters.
        """
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                _init_linear_(layer, generator)
        for head in self.heads:
            _init_linear_(head, generator)

    def trunk_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Shared representation ``[batch, feature_dim]`` (the trunk output before the heads).

        Exposed so the ensemble agent can attach a gradient hook to the shared features —
        the Osband et al. (2016) 1/K trunk-gradient normalization is applied to *this*
        tensor's backward gradient, leaving each head's own gradient unscaled.
        """
        return self.trunk(obs)

    def heads_forward(self, features: torch.Tensor) -> torch.Tensor:
        """Per-head Q-values ``[batch, n_heads, n_actions]`` from precomputed features."""
        return torch.stack([head(features) for head in self.heads], dim=1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Map observations to per-head Q-values, shape ``[batch, n_heads, n_actions]``."""
        # trunk then heads; split out so the ensemble can hook the shared features.
        return self.heads_forward(self.trunk_features(obs))
