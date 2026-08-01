"""``ValueSampler`` adapters + the DeepSea probe set (§3.3 substrate inputs).

:mod:`src.diagnostics.substrate` defines the ``ValueSampler`` protocol and the record/writer
machinery, but shipped with **no implementation** of the protocol — so nothing could actually
produce a ``[S, M, A]`` value-sample tensor, and the whole §3.3 battery had no input. This
module supplies the two adapters the study needs and the probe set they are evaluated on.

Two invariants govern everything here, and both are the reason these adapters are separate
from the agents rather than methods on them:

**Measurement must not perturb training (gate C1).** Every draw taken here comes from a
measurement-side generator — the ``noisynet_diag`` stream for NoisyNet's M = 30 draws, and
for the ensembles no randomness at all, since sampling the K heads is exhaustive. Nothing in
this module touches an operational stream, so a run with diagnostics enabled and the same run
with diagnostics disabled produce byte-identical training trajectories.

**M is a property of the method, not a free parameter.** For the ensemble methods M = K and
the "samples" are the heads themselves in fixed index order (deterministic, exhaustive). For
NoisyNet M = 30 i.i.d. parameter-noise draws taken *at measurement time only* (freeze item 14).
Both are asserted against the ``SubstrateSpec`` on every record.
"""

from __future__ import annotations

import numpy as np
import torch

from .substrate import SubstrateSpec

#: NoisyNet measurement draws (freeze item 14: "M = 30 i.i.d. NoisyNet draws at measurement
#: only"). Mirrors ``src.noisynet.DIAG_SAMPLES``; asserted equal in tests.
NOISYNET_DIAG_SAMPLES = 30


# --------------------------------------------------------------------------- #
# Probe set
# --------------------------------------------------------------------------- #

def deep_sea_probe_states(size: int, *, flatten: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """The exhaustive DeepSea probe set and its ``(row, col)`` index.

    Freeze item 7 (owner-approved 2026-07-30) sets the DeepSea probe set to the **exhaustive
    reachable set** — no cap, no subsampling. In DeepSea the agent is at row ``t`` after ``t``
    steps and the column is bounded by the row, so the reachable cells are exactly
    ``{(r, c) : 0 <= c <= r < N}`` and ``|S| = N(N+1)/2``.

    Parameters
    ----------
    flatten:
        Default ``True``, which returns ``[S, N*N]`` — the encoding the **agent** sees. The
        DeepSea env yields a 2D one-hot grid and :func:`src.trainer.run_seed` flattens it
        before every ``select_action``, so a probe set in grid form would be a different
        input distribution from the one the network was trained on. Pass ``False`` for the
        ``[S, N, N]`` grid form when a consumer wants the spatial layout.

    Returns
    -------
    (observations, index)
        ``index`` is ``[S, 2]`` int64 of the ``(row, col)`` each probe state corresponds to,
        so a reducer can align a probe state with ``Q*[row, col]`` without re-deriving the
        ordering.

    The ordering is row-major over ``(row, col)`` and is part of the contract: the substrate
    stores tensors positionally, so records are only poolable if the probe ordering is fixed.
    """
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    cells = [(r, c) for r in range(size) for c in range(size) if c <= r]
    obs = np.zeros((len(cells), size, size), dtype=np.float32)
    for i, (r, c) in enumerate(cells):
        obs[i, r, c] = 1.0
    if flatten:
        obs = obs.reshape(len(cells), -1)
    return obs, np.asarray(cells, dtype=np.int64)


def deep_sea_probe_set_size(size: int) -> int:
    """``|S| = N(N+1)/2`` — the exhaustive reachable-set size."""
    return size * (size + 1) // 2


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #

class EnsembleValueSampler:
    """``ValueSampler`` for the Bootstrapped-DQN ensemble (``bdqn`` / ``rp_bdqn``).

    The K heads *are* the value samples, so ``M = K`` and sampling is exhaustive and
    deterministic: no generator is needed and none is held. Two calls at the same network
    state return identical tensors, which is what makes the ensemble arm of the battery
    exactly reproducible.

    Head order is the agent's own index order and is stable across checkpoints, so
    ``samples[:, k, :]`` refers to the same head throughout a run.
    """

    sampler_kind = "ensemble_heads"

    def __init__(self, agent):
        self.agent = agent

    @property
    def n_samples(self) -> int:
        return int(self.agent.cfg.K)

    @torch.no_grad()
    def value_samples(self, probe_states: np.ndarray) -> np.ndarray:
        """``[S, K, A]`` per-head action-values on the probe set."""
        probes = np.asarray(probe_states)
        out = np.stack(
            [np.asarray(self.agent._q_all(s).cpu(), dtype=np.float32) for s in probes]
        )
        return out.astype(np.float32, copy=False)

    def spec(self, *, n_probe_states: int, n_actions: int, probe_set_id: str) -> SubstrateSpec:
        return SubstrateSpec(
            n_probe_states=n_probe_states,
            n_samples=self.n_samples,
            n_actions=n_actions,
            sampler_kind=self.sampler_kind,
            probe_set_id=probe_set_id,
        )


class NoisyNetValueSampler:
    """``ValueSampler`` for NoisyNet-DQN: M = 30 i.i.d. parameter-noise draws.

    Unlike the ensemble, NoisyNet has no finite set of heads — a "value sample" is a draw of
    the factorized parameter noise. Those draws are taken from the caller-supplied
    measurement generator (the ``noisynet_diag`` stream), never from the operational noise
    generator, so measurement cannot shift the acting policy.

    Delegates to :meth:`src.noisynet.NoisyNetAgent.sample_q_values`, which owns the
    noise-reset loop and restores the agent's operational noise afterwards.
    """

    sampler_kind = "noisynet_draws"

    def __init__(self, agent, m: int = NOISYNET_DIAG_SAMPLES):
        if m < 1:
            raise ValueError(f"m must be >= 1, got {m}")
        self.agent = agent
        self.m = int(m)

    @property
    def n_samples(self) -> int:
        return self.m

    @torch.no_grad()
    def value_samples(self, probe_states: np.ndarray) -> np.ndarray:
        """``[S, M, A]`` action-values under M i.i.d. measurement-time noise draws."""
        probes = np.asarray(probe_states)
        out = np.stack(
            [
                np.asarray(self.agent.sample_q_values(s, m=self.m).cpu(), dtype=np.float32)
                for s in probes
            ]
        )
        return out.astype(np.float32, copy=False)

    def spec(self, *, n_probe_states: int, n_actions: int, probe_set_id: str) -> SubstrateSpec:
        return SubstrateSpec(
            n_probe_states=n_probe_states,
            n_samples=self.n_samples,
            n_actions=n_actions,
            sampler_kind=self.sampler_kind,
            probe_set_id=probe_set_id,
        )


def make_value_sampler(agent, method: str, *, m: int = NOISYNET_DIAG_SAMPLES):
    """Return the ``ValueSampler`` for ``method``, or ``None`` if it has no value samples.

    ``ddqn_egreedy`` is a point estimator: it has one Q-function and no sample distribution,
    so σ(s, a) is undefined for it and the §3.3 battery does not apply. Returning ``None``
    rather than raising lets the trainer treat "this method has no uncertainty to measure" as
    an ordinary branch instead of a special case at every call site.
    """
    if method in ("bdqn", "rp_bdqn"):
        return EnsembleValueSampler(agent)
    if method == "noisynet":
        return NoisyNetValueSampler(agent, m=m)
    return None


# --------------------------------------------------------------------------- #
# Disagreement summary (spec §8 step 5's logged quantity)
# --------------------------------------------------------------------------- #

def disagreement_summary(samples: np.ndarray) -> dict[str, float]:
    """Scalar ensemble-disagreement summaries from one ``[S, M, A]`` checkpoint tensor.

    Spec §8 step 5 requires per-run disagreement logging, and RQ2-L consumes σ(s, a). The
    full tensor goes to the substrate ``.npz``; these are the scalars that ride in the run
    CSV so disagreement is visible per checkpoint without loading the tensor.

    ``ddof=0`` — σ is the *sample std over the M value samples* as §3.3 defines it, and for
    the ensembles the M heads are the entire population of samples, not a draw from a larger
    one. Using ddof=1 would inflate σ by sqrt(K/(K-1)) — 5.4% at K = 10 — on a quantity whose
    cross-method comparison is the point.

    Returns ``mean_sigma`` (over all (s, a)), ``max_sigma``, and ``mean_sigma_greedy``
    (σ at each state's ensemble-mean-greedy action, ties → lowest index — the §3.3
    convention). All are NaN-free by construction: the substrate rejects non-finite samples.
    """
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"expected [S, M, A], got shape {arr.shape}")
    if arr.shape[1] < 2:
        raise ValueError(
            f"disagreement needs M >= 2 value samples, got M = {arr.shape[1]}; "
            "a point estimator has no sample distribution (see make_value_sampler)"
        )
    sigma = arr.std(axis=1, ddof=0)  # [S, A]
    greedy = arr.mean(axis=1).argmax(axis=1)  # [S], ties -> lowest index
    return {
        "mean_sigma": float(sigma.mean()),
        "max_sigma": float(sigma.max()),
        "mean_sigma_greedy": float(sigma[np.arange(sigma.shape[0]), greedy].mean()),
    }
