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


def _init_conv_(layer: nn.Conv2d, generator: torch.Generator) -> None:
    """Reproducibly initialize an ``nn.Conv2d`` from ``generator`` (no global RNG).

    Same scheme as :func:`_init_linear_` — ``U(-1/sqrt(fan_in), 1/sqrt(fan_in))`` for weight
    and bias, matching PyTorch's default — but ``fan_in`` for a convolution is
    ``in_channels * kernel_h * kernel_w``, **not** ``weight.shape[1]``. Using the linear
    helper on a conv layer would silently over-scale the initialization by a factor of
    ``sqrt(kernel_h * kernel_w)``.
    """
    fan_in = layer.weight.shape[1] * layer.weight.shape[2] * layer.weight.shape[3]
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


def _factorized_noise(size: int, generator: torch.Generator) -> torch.Tensor:
    """Factorized-Gaussian noise vector ``f(ε)`` with ``f(x)=sgn(x)·sqrt(|x|)``.

    The transformation is Fortunato et al. (2018) §3.2's factorized scheme: a single
    length-``size`` standard-normal draw is passed through ``f`` and later combined by
    outer product to build the weight-noise matrix, so a ``[out, in]`` layer needs only
    ``out + in`` unit-normal samples instead of ``out·in``. Drawn from ``generator`` so the
    noise is a deterministic function of the stream, never the global torch RNG (gate C1).
    """
    eps = torch.randn(size, generator=generator)
    return eps.sign() * eps.abs().sqrt()


class NoisyLinear(nn.Module):
    """Factorized-Gaussian noisy linear layer (Fortunato et al. 2018, the DQN variant).

    Replaces ``y = Wx + b`` with ``y = (μ_w + σ_w ⊙ ε_w)x + (μ_b + σ_b ⊙ ε_b)`` where the
    ``μ`` and ``σ`` are learnable and the ``ε`` are fixed samples resampled between forward
    passes. The learnable ``σ`` let the network *learn* how much noise (exploration) to
    inject per weight, and anneal it away where the value estimate has become confident —
    this is what replaces the ε-greedy schedule.

    Initialization (Fortunato §3.2, factorized case): ``μ ~ U(-1/√p, 1/√p)`` and
    ``σ = σ₀/√p`` with ``p = fan_in``; ``σ₀`` is passed in (development placeholder 0.5, the
    paper's factorized default — never frozen in source). All ``μ``/``σ`` draws come from the
    ``init``-stream ``generator``; the per-step ε samples come from a *separate* operational
    generator handed to :meth:`reset_noise`, so parameter init and exploration noise never
    share a draw sequence.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        sigma0: float = 0.5,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError(
                f"in/out features must be positive, got {in_features}, {out_features}"
            )
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.sigma0 = float(sigma0)

        self.weight_mu = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.weight_sigma = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.bias_mu = nn.Parameter(torch.empty(self.out_features))
        self.bias_sigma = nn.Parameter(torch.empty(self.out_features))
        # Noise buffers (not parameters — never optimized). Filled by reset_noise().
        self.register_buffer("weight_eps", torch.zeros(self.out_features, self.in_features))
        self.register_buffer("bias_eps", torch.zeros(self.out_features))

        if generator is not None:
            self.reset_parameters(generator)

    def reset_parameters(self, generator: torch.Generator) -> None:
        """Initialize μ/σ from the ``init``-stream ``generator`` (Fortunato §3.2, factorized)."""
        bound = 1.0 / math.sqrt(self.in_features)
        sigma_init = self.sigma0 / math.sqrt(self.in_features)
        with torch.no_grad():
            self.weight_mu.uniform_(-bound, bound, generator=generator)
            self.bias_mu.uniform_(-bound, bound, generator=generator)
            self.weight_sigma.fill_(sigma_init)
            self.bias_sigma.fill_(sigma_init)

    def reset_noise(self, generator: torch.Generator) -> None:
        """Resample the factorized ε buffers from an operational ``generator``.

        Draw order is fixed (input factor then output factor) so the noise is a
        deterministic function of the operational stream's state.
        """
        eps_in = _factorized_noise(self.in_features, generator)
        eps_out = _factorized_noise(self.out_features, generator)
        with torch.no_grad():
            self.weight_eps.copy_(torch.outer(eps_out, eps_in))
            self.bias_eps.copy_(eps_out)

    def forward(self, x: torch.Tensor, *, noisy: bool = True) -> torch.Tensor:
        """Linear map. ``noisy=True`` uses μ+σ⊙ε; ``noisy=False`` uses μ only (mean net)."""
        if noisy:
            weight = self.weight_mu + self.weight_sigma * self.weight_eps
            bias = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return nn.functional.linear(x, weight, bias)


class NoisyMLPQNetwork(nn.Module):
    """Single-head MLP Q-network whose linear layers are :class:`NoisyLinear` (NoisyNet-DQN).

    Structurally the ``n_heads=1`` Double-DQN backbone — same trunk-then-head shape and the
    same stream-derived, no-global-RNG initialization contract as :class:`MLPQNetwork` — but
    every ``nn.Linear`` is a :class:`NoisyLinear`. Exploration comes from the learnable
    parameter noise (Fortunato et al. 2018) instead of an ε-greedy schedule, so there is no
    head axis and no ε. ``hidden_sizes`` (Class-1 width) and ``sigma0`` are passed in.

    Forward returns ``[batch, n_actions]``. :meth:`reset_noise` resamples every layer's ε from
    the operational generator; a forward with ``noisy=False`` gives the mean (μ-only) network.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int],
        *,
        sigma0: float = 0.5,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if obs_dim <= 0 or n_actions <= 0:
            raise ValueError(f"obs_dim and n_actions must be positive, got {obs_dim}, {n_actions}")
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.sigma0 = float(sigma0)

        self.layers = nn.ModuleList()
        self.activations: list[bool] = []  # True where a ReLU follows the layer
        in_dim = self.obs_dim
        for width in hidden_sizes:
            self.layers.append(NoisyLinear(in_dim, int(width), sigma0=self.sigma0))
            self.activations.append(True)
            in_dim = int(width)
        self.layers.append(NoisyLinear(in_dim, self.n_actions, sigma0=self.sigma0))
        self.activations.append(False)

        if generator is not None:
            self.reset_parameters(generator)

    def reset_parameters(self, generator: torch.Generator) -> None:
        """Re-initialize every noisy layer's μ/σ from ``generator`` (the ``init`` stream)."""
        for layer in self.layers:
            layer.reset_parameters(generator)

    def reset_noise(self, generator: torch.Generator) -> None:
        """Resample the factorized noise of every layer (in order) from ``generator``."""
        for layer in self.layers:
            layer.reset_noise(generator)

    def forward(self, obs: torch.Tensor, *, noisy: bool = True) -> torch.Tensor:
        """Q-values ``[batch, n_actions]``. ``noisy=False`` evaluates the mean (μ-only) net."""
        x = obs
        for layer, has_relu in zip(self.layers, self.activations, strict=True):
            x = layer(x, noisy=noisy)
            if has_relu:
                x = nn.functional.relu(x)
        return x


# --------------------------------------------------------------------------- #
# MinAtar (Part B) conv backbone
#
# Same two-class shape as the MLP path above: one multi-head class backing every
# value-based method, plus a NoisyNet variant. The conv trunk shape is a **Class-1**
# backbone nuisance (network width), not a Class-2 frozen constant — see
# protocol/decisions/minatar_backbone.md §1. The Class-2 shared-trunk row *is* honoured:
# K independent heads split after the shared (here convolutional) representation, which
# is Osband et al. (2016) Fig. 1a's original conv formulation.
# --------------------------------------------------------------------------- #

#: Default MinAtar torso: a single 16-filter 3x3 valid convolution, then a 128-unit FC.
#: The standard MinAtar DQN shape (Young & Tian 2019) and the shape already used by
#: ``compute/benchmarks/stage_a_benchmark.py``, so the Stage-A compute forecast stays valid.
#: Class-1 defaults — the caller may override, and the backbone search may revise them.
MINATAR_CONV_CHANNELS = 16
MINATAR_CONV_KERNEL = 3
MINATAR_FC_WIDTH = 128


class MinAtarConvQNetwork(nn.Module):
    """Shared conv+FC trunk + ``n_heads`` independent linear value heads (MinAtar).

    The Part-B analogue of :class:`MLPQNetwork`, with an identical public contract —
    :meth:`reset_parameters`, :meth:`trunk_features`, :meth:`heads_forward`, and a
    ``[batch, n_heads, n_actions]`` forward — so every method (DDQN at ``n_heads=1``,
    Bootstrapped-DQN at ``n_heads=K``, RP-BDQN as the ``prior=on`` arm of the same agent)
    is served by one class on MinAtar exactly as on DeepSea. That parity is what keeps
    gate C11 (per-contrast code-path purity) checkable on Part B.

    Args:
        obs_shape: channel-first observation shape ``(C, H, W)`` — ``MinAtarEnv.obs_shape``.
            MinAtar is ``(C, 10, 10)`` with a per-game ``C``.
        n_actions: size of the discrete action space (6 for every MinAtar game).
        conv_channels, kernel_size, fc_width: Class-1 trunk widths. Defaults are the
            standard MinAtar torso; never frozen here.
        n_heads: ``1`` = Double-DQN backbone; ``K`` = Bootstrapped-DQN ensemble.
        generator: seeded ``torch.Generator`` for the ``init`` stream. ``None`` keeps
            torch's global-RNG default and is for throwaway shape checks only.
    """

    def __init__(
        self,
        obs_shape: Sequence[int],
        n_actions: int,
        *,
        conv_channels: int = MINATAR_CONV_CHANNELS,
        kernel_size: int = MINATAR_CONV_KERNEL,
        fc_width: int = MINATAR_FC_WIDTH,
        n_heads: int = 1,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        obs_shape = tuple(int(x) for x in obs_shape)
        if len(obs_shape) != 3:
            raise ValueError(
                f"obs_shape must be channel-first (C, H, W), got {obs_shape}; "
                "MinAtar observations are (C, 10, 10) — do not flatten for the conv trunk"
            )
        if any(d <= 0 for d in obs_shape) or n_actions <= 0:
            raise ValueError(
                f"obs_shape and n_actions must be positive, got {obs_shape}, {n_actions}"
            )
        if n_heads <= 0:
            raise ValueError(f"n_heads must be >= 1, got {n_heads}")
        in_ch, height, width = obs_shape
        if kernel_size > min(height, width):
            raise ValueError(
                f"kernel_size {kernel_size} exceeds the {height}x{width} MinAtar grid"
            )

        self.obs_shape = obs_shape
        self.n_actions = int(n_actions)
        self.n_heads = int(n_heads)
        self.conv_channels = int(conv_channels)
        self.kernel_size = int(kernel_size)
        self.fc_width = int(fc_width)

        # Shared trunk: valid conv -> ReLU -> flatten -> FC -> ReLU.
        self.conv = nn.Conv2d(in_ch, self.conv_channels, kernel_size=self.kernel_size, stride=1)
        conv_h = height - self.kernel_size + 1
        conv_w = width - self.kernel_size + 1
        self.conv_out_dim = self.conv_channels * conv_h * conv_w
        self.fc = nn.Linear(self.conv_out_dim, self.fc_width)
        self.feature_dim = self.fc_width
        # K independent heads split after the shared representation (Osband 2016 Fig. 1a).
        self.heads = nn.ModuleList(
            [nn.Linear(self.feature_dim, self.n_actions) for _ in range(self.n_heads)]
        )

        if generator is not None:
            self.reset_parameters(generator)

    def reset_parameters(self, generator: torch.Generator) -> None:
        """Re-initialize every parameter from ``generator`` (the ``init`` stream).

        Draw order is fixed — conv, then FC, then each head in order — so initialization is a
        deterministic function of the seed. Heads are drawn sequentially from the one
        generator, giving each head independent parameters (the ensemble diversity prior).
        """
        _init_conv_(self.conv, generator)
        _init_linear_(self.fc, generator)
        for head in self.heads:
            _init_linear_(head, generator)

    def trunk_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Shared representation ``[batch, feature_dim]``.

        Exposed for the same reason as the MLP version: the ensemble agent hooks this
        tensor's backward gradient to apply Osband et al. (2016)'s 1/K trunk-gradient
        normalization, leaving each head's own gradient unscaled.
        """
        if obs.dim() != 4:
            raise ValueError(
                f"expected a 4-D [batch, C, H, W] observation batch, got shape {tuple(obs.shape)}"
            )
        x = nn.functional.relu(self.conv(obs))
        x = torch.flatten(x, start_dim=1)
        return nn.functional.relu(self.fc(x))

    def heads_forward(self, features: torch.Tensor) -> torch.Tensor:
        """Per-head Q-values ``[batch, n_heads, n_actions]`` from precomputed features."""
        return torch.stack([head(features) for head in self.heads], dim=1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Map observations to per-head Q-values, shape ``[batch, n_heads, n_actions]``."""
        return self.heads_forward(self.trunk_features(obs))


class NoisyMinAtarConvQNetwork(nn.Module):
    """Single-head MinAtar conv Q-network whose **linear** layers are :class:`NoisyLinear`.

    The Part-B analogue of :class:`NoisyMLPQNetwork`. The convolution stays a plain
    deterministic ``nn.Conv2d``; noise is applied to the fully-connected and output layers
    only — the Fortunato et al. (2018) convention of putting the learnable noise on the
    linear layers of the value stream. Forward returns ``[batch, n_actions]`` (no head axis),
    and exploration comes from the learnable σ rather than an ε schedule.
    """

    def __init__(
        self,
        obs_shape: Sequence[int],
        n_actions: int,
        *,
        conv_channels: int = MINATAR_CONV_CHANNELS,
        kernel_size: int = MINATAR_CONV_KERNEL,
        fc_width: int = MINATAR_FC_WIDTH,
        sigma0: float = 0.5,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        obs_shape = tuple(int(x) for x in obs_shape)
        if len(obs_shape) != 3:
            raise ValueError(
                f"obs_shape must be channel-first (C, H, W), got {obs_shape}"
            )
        if any(d <= 0 for d in obs_shape) or n_actions <= 0:
            raise ValueError(
                f"obs_shape and n_actions must be positive, got {obs_shape}, {n_actions}"
            )
        in_ch, height, width = obs_shape
        if kernel_size > min(height, width):
            raise ValueError(
                f"kernel_size {kernel_size} exceeds the {height}x{width} MinAtar grid"
            )

        self.obs_shape = obs_shape
        self.n_actions = int(n_actions)
        self.sigma0 = float(sigma0)
        self.conv_channels = int(conv_channels)
        self.kernel_size = int(kernel_size)
        self.fc_width = int(fc_width)

        self.conv = nn.Conv2d(in_ch, self.conv_channels, kernel_size=self.kernel_size, stride=1)
        conv_h = height - self.kernel_size + 1
        conv_w = width - self.kernel_size + 1
        self.conv_out_dim = self.conv_channels * conv_h * conv_w
        # Noisy linear layers only (conv stays deterministic).
        self.layers = nn.ModuleList(
            [
                NoisyLinear(self.conv_out_dim, self.fc_width, sigma0=self.sigma0),
                NoisyLinear(self.fc_width, self.n_actions, sigma0=self.sigma0),
            ]
        )
        self.activations: list[bool] = [True, False]  # ReLU after the hidden layer only

        if generator is not None:
            self.reset_parameters(generator)

    def reset_parameters(self, generator: torch.Generator) -> None:
        """Re-initialize the conv and every noisy layer's μ/σ from the ``init`` stream.

        Draw order is fixed: conv first, then the noisy layers in order.
        """
        _init_conv_(self.conv, generator)
        for layer in self.layers:
            layer.reset_parameters(generator)

    def reset_noise(self, generator: torch.Generator) -> None:
        """Resample the factorized noise of every noisy layer (in order) from ``generator``.

        The convolution has no noise to resample — it is deterministic by design.
        """
        for layer in self.layers:
            layer.reset_noise(generator)

    def forward(self, obs: torch.Tensor, *, noisy: bool = True) -> torch.Tensor:
        """Q-values ``[batch, n_actions]``. ``noisy=False`` evaluates the mean (μ-only) net."""
        if obs.dim() != 4:
            raise ValueError(
                f"expected a 4-D [batch, C, H, W] observation batch, got shape {tuple(obs.shape)}"
            )
        x = nn.functional.relu(self.conv(obs))
        x = torch.flatten(x, start_dim=1)
        for layer, has_relu in zip(self.layers, self.activations, strict=True):
            x = layer(x, noisy=noisy)
            if has_relu:
                x = nn.functional.relu(x)
        return x


# --------------------------------------------------------------------------- #
# Backbone selection
# --------------------------------------------------------------------------- #
def build_q_network(
    obs_dim: int,
    n_actions: int,
    hidden_sizes: Sequence[int],
    *,
    obs_shape: Sequence[int] | None = None,
    n_heads: int = 1,
    generator: torch.Generator | None = None,
):
    """Return the right multi-head Q-network for the observation shape.

    ``obs_shape=None`` (the DeepSea / Part-A path) gives :class:`MLPQNetwork` built from the
    flattened ``obs_dim`` — byte-identical to constructing it directly, so Part-A runs are
    unaffected. A 3-element ``obs_shape`` (the MinAtar / Part-B path) gives
    :class:`MinAtarConvQNetwork` over channel-first ``(C, H, W)`` observations, and
    ``hidden_sizes`` is reinterpreted as the conv trunk's FC width: ``hidden_sizes[-1]`` if
    given, else the standard MinAtar 128.

    One selection point means the three agents (DDQN, NoisyNet, BDQN/RP-BDQN) stay identical
    apart from ``n_heads`` and the prior — which is what gate C11 checks.
    """
    if obs_shape is None:
        return MLPQNetwork(
            obs_dim, n_actions, hidden_sizes, n_heads=n_heads, generator=generator
        )
    obs_shape = tuple(int(x) for x in obs_shape)
    fc_width = int(hidden_sizes[-1]) if len(hidden_sizes) else MINATAR_FC_WIDTH
    return MinAtarConvQNetwork(
        obs_shape,
        n_actions,
        fc_width=fc_width,
        n_heads=n_heads,
        generator=generator,
    )


def build_noisy_q_network(
    obs_dim: int,
    n_actions: int,
    hidden_sizes: Sequence[int],
    *,
    obs_shape: Sequence[int] | None = None,
    sigma0: float = 0.5,
    generator: torch.Generator | None = None,
):
    """NoisyNet counterpart of :func:`build_q_network` (single-head by construction)."""
    if obs_shape is None:
        return NoisyMLPQNetwork(
            obs_dim, n_actions, hidden_sizes, sigma0=sigma0, generator=generator
        )
    obs_shape = tuple(int(x) for x in obs_shape)
    fc_width = int(hidden_sizes[-1]) if len(hidden_sizes) else MINATAR_FC_WIDTH
    return NoisyMinAtarConvQNetwork(
        obs_shape, n_actions, fc_width=fc_width, sigma0=sigma0, generator=generator
    )
