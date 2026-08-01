"""Diagnostic 8 — initial-state return-prediction alignment (MinAtar, appendix-only).

Signed off 2026-08-01 in Fix #6 of
``protocol/decisions/staged_stage3_protocol_fixes.md``.

What this measures
------------------
At a single network checkpoint, for each of a set of distinct episode-start observations
drawn from the ``probe_set`` stream:

* **V̄(s₀)** = max_a Q̄(s₀, a), where Q̄ = mean over the M value samples.  This is the
  agent's greedy state-value estimate at episode start.
* **σ_V(s₀)** = std_m[max_a Q_m(s₀, a)] with ddof = 0 — the spread of the per-sample
  state-value estimates at that start.
* **G(s₀)** = empirical discounted return from s₀ under the mean-greedy rollout policy
  (bit-exact from the post-reset clone, so every rollout is reproducible).

The diagnostic statistic is **Spearman ρ(σ_V, |V̄ − G|)**: does the agent's uncertainty
at episode start track how wrong its value prediction actually is?

Scope constraints (Fix #6, signed off 2026-08-01)
--------------------------------------------------
* **Appendix-only, exploratory.** Does *not* participate in primary contrasts C-i or C-ii.
* **No within-episode or general-state-distribution claim.**  State population = episode-start
  states only (observation returned by ``env.reset()``).
* **Uniqueness threshold**: if fewer than :data:`UNIQUENESS_THRESHOLD` distinct observations
  arise from the 100 seeds, the diagnostic is omitted for that game and reported as
  uninformative (Fix #6 decision; many MinAtar games have low reset-state diversity so 100
  seeds may yield far fewer distinct boards).
* σ_V = 0 is kept in the Spearman — it is a substantive claim (confident prediction) not a
  divisor — consistent with battery diagnostics 1, 2, 4, 7.

Measurement isolation
---------------------
This module creates its own ``MinAtarEnv`` keyed on the ``probe_set`` stream (never the
training env).  The rollout policy is derived from the ``ValueSampler``, which draws only
from measurement-side generators.  No operational RNG stream is advanced by running this
diagnostic.

Code provenance
---------------
Every serialised result should carry ``conventions.code_version()`` so that dirty trees are
flagged as non-reproducible.  :func:`diag8_to_record` adds this field automatically.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from src.utils import conventions

# --------------------------------------------------------------------------- #
# Frozen constants (Fix #6)
# --------------------------------------------------------------------------- #

#: Number of episode-start seeds drawn from the ``probe_set`` stream.
N_START_SEEDS: int = 100

#: Omit the diagnostic if fewer than this many unique start observations are found.
UNIQUENESS_THRESHOLD: int = 20

#: Default rollout horizon per start state. MinAtar episodes can be long; cap so that
#: 100 rollouts are cheap enough to run at every checkpoint.
DEFAULT_MAX_ROLLOUT_STEPS: int = 500

#: Default discount for empirical-return computation.
DEFAULT_GAMMA: float = 0.99

NA = float("nan")

# Degeneracy threshold — matches ``analysis.diagnostics_battery.is_degenerate`` exactly.
_DEGENERACY_EPS_MULTIPLE = 8.0


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Diag8Result:
    """Result of Diagnostic 8 for one game at one checkpoint.

    Field layout mirrors :class:`analysis.diagnostics_battery.Result` for consistency,
    extended with MinAtar-specific provenance fields.
    """

    game: str
    value: float          # Spearman ρ, or NaN if uninformative / undefined
    n_seeds: int          # episode-start seeds drawn from probe_set stream
    n_unique_starts: int  # distinct start observations found
    n_used: int           # states that entered the Spearman
    n_excluded: int       # states dropped (none expected; see σ convention above)
    defined: bool         # False iff value is NaN
    reason: str           # non-empty when undefined, explaining why
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """JSON-serialisable representation (for embedding in a run record)."""
        return asdict(self)


def _uninformative(
    game: str,
    reason: str,
    *,
    n_seeds: int,
    n_unique: int,
) -> Diag8Result:
    return Diag8Result(
        game=game,
        value=NA,
        n_seeds=n_seeds,
        n_unique_starts=n_unique,
        n_used=0,
        n_excluded=n_unique,
        defined=False,
        reason=reason,
    )


# --------------------------------------------------------------------------- #
# ValueSampler protocol (minimal restatement for this module)
# --------------------------------------------------------------------------- #

@runtime_checkable
class ValueSampler(Protocol):
    """Minimal interface required by this diagnostic.

    Both :class:`src.diagnostics.samplers.EnsembleValueSampler` and
    :class:`src.diagnostics.samplers.NoisyNetValueSampler` satisfy this protocol.
    """

    def value_samples(self, probe_states: np.ndarray) -> np.ndarray:
        """``[S, M, A]`` value samples for the given observations."""
        ...


# --------------------------------------------------------------------------- #
# Step 1 — collect episode-start states
# --------------------------------------------------------------------------- #

def collect_episode_starts(
    game: str,
    *,
    master_seed: int,
    cell_id: str,
    seed_index: int,
    n_seeds: int = N_START_SEEDS,
) -> tuple[list[np.ndarray], list[dict]]:
    """Draw ``n_seeds`` episode-start observations and their clone-state snapshots.

    Uses the ``probe_set`` stream (from :data:`src.utils.conventions.STREAM_NAMES`) so the
    start-state population is keyed on the run's own coordinates and is fully reproducible
    without touching any operational stream.

    Each reset seed is derived by spawning ``n_seeds`` child
    :class:`numpy.random.SeedSequence` instances from the ``probe_set`` SeedSequence and
    generating one uint32 value per child — the canonical path for 32-bit MinAtar seeds.

    Returns
    -------
    (observations, snapshots)
        ``observations[i]`` is the float32 channel-first ``(C, 10, 10)`` array returned by
        ``MinAtarEnv.reset(seed=...)``.  ``snapshots[i]`` is the ``clone_state()`` dict
        taken immediately after that reset — safe to use as a rollout starting point.
    """
    from src.minatar_env import MinAtarEnv

    # Derive reset seeds from the probe_set stream.  MinAtar's RandomState accepts only
    # 32-bit seeds; derive via spawn + generate_state rather than masking a 63-bit int.
    ss = conventions.derive_seed_sequence(master_seed, cell_id, "probe_set", seed_index)
    child_seqs = ss.spawn(n_seeds)
    reset_seeds = [int(cs.generate_state(1, dtype=np.uint32)[0]) for cs in child_seqs]

    # Diagnostic env — never the training env.
    env = MinAtarEnv(
        game,
        master_seed=master_seed,
        cell_id=cell_id,
        seed_index=seed_index,
    )

    observations: list[np.ndarray] = []
    snapshots: list[dict] = []
    for s in reset_seeds:
        obs, _ = env.reset(seed=s)
        snap = env.clone_state()
        observations.append(obs)
        snapshots.append(snap)

    return observations, snapshots


# --------------------------------------------------------------------------- #
# Step 2 — deduplicate
# --------------------------------------------------------------------------- #

def deduplicate_starts(
    observations: list[np.ndarray],
    snapshots: list[dict],
) -> tuple[np.ndarray, list[dict]]:
    """Return unique (observation, snapshot) pairs by byte-equality of the observation.

    Uniqueness is ``obs.tobytes()``-based: two float32 channel-first tensors that are
    byte-identical represent the same board (MinAtar state is boolean under the hood and
    the conversion to float32 is lossless for {0.0, 1.0}).  The **first** snapshot seen
    for each unique observation is retained (FIFO — preserves the probe_set seed order).

    Returns
    -------
    (unique_obs, unique_snaps)
        ``unique_obs`` is ``[N_unique, *obs_shape]`` stacked; ``unique_snaps`` is the
        corresponding list of ``clone_state`` dicts.  Both are empty when the input is empty.
    """
    seen: dict[bytes, int] = {}
    unique_obs: list[np.ndarray] = []
    unique_snaps: list[dict] = []
    for obs, snap in zip(observations, snapshots, strict=True):
        key = np.asarray(obs).tobytes()
        if key not in seen:
            seen[key] = len(unique_obs)
            unique_obs.append(np.asarray(obs))
            unique_snaps.append(snap)
    if not unique_obs:
        return np.empty((0,), dtype=np.float32), []
    return np.stack(unique_obs), unique_snaps


# --------------------------------------------------------------------------- #
# Step 3 — rollouts
# --------------------------------------------------------------------------- #

def rollout_from_snapshot(
    env,
    snapshot: dict,
    policy: Callable[[np.ndarray], int],
    *,
    max_steps: int = DEFAULT_MAX_ROLLOUT_STEPS,
    gamma: float = DEFAULT_GAMMA,
) -> float:
    """Discounted return from a clone-state snapshot under the given policy.

    Restores ``env`` to the post-reset state in ``snapshot`` via
    :meth:`~src.minatar_env.MinAtarEnv.restore_state` and steps forward with ``policy``
    until the episode terminates or ``max_steps`` is reached.

    The same snapshot can be replayed multiple times (``restore_state`` deep-copies the
    snapshot on the way in, so passing the same dict twice gives two independent rollouts).

    Parameters
    ----------
    env:
        A :class:`~src.minatar_env.MinAtarEnv` **dedicated to rollouts** — not the training
        env.  Its state is overwritten by this call.
    snapshot:
        A dict produced by ``MinAtarEnv.clone_state()``.
    policy:
        ``(obs: np.ndarray) -> int`` — the greedy action at each step.
    max_steps:
        Truncation horizon.
    gamma:
        Discount factor.

    Returns
    -------
    float
        :math:`G = \\sum_{t=0}^{T-1} \\gamma^t r_t`.
    """
    env.restore_state(snapshot)
    total = 0.0
    discount = 1.0
    for _ in range(max_steps):
        obs = env._observation()
        action = policy(obs)
        _next_obs, reward, terminated, _truncated, _info = env.step(action)
        total += discount * float(reward)
        discount *= gamma
        if terminated:
            break
    return total


# --------------------------------------------------------------------------- #
# Step 4 — value and uncertainty at start states
# --------------------------------------------------------------------------- #

def value_and_uncertainty(
    sampler: ValueSampler,
    start_obs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predicted state-value mean and uncertainty at each start observation.

    Parameters
    ----------
    sampler:
        Any :class:`ValueSampler` — ``EnsembleValueSampler`` or ``NoisyNetValueSampler``.
    start_obs:
        ``[N, *obs_shape]`` batch of episode-start observations.

    Returns
    -------
    (v_bar, sigma_v)
        ``v_bar[i]`` = max_a Q̄(s_i, a) = greedy state-value under the mean Q-function.

        ``sigma_v[i]`` = std_m[max_a Q_m(s_i, a)] with ddof = 0 = spread of the per-sample
        state-value estimates.

        Both are 1-D float64 arrays of length N.

    Notes
    -----
    ddof = 0 — the M value samples *are* the population (heads or draws), not a sub-sample
    of a larger one.  This matches ``analysis.diagnostics_battery.sigma_and_mean`` and
    ``src.diagnostics.samplers.disagreement_summary``.
    """
    samples = sampler.value_samples(np.asarray(start_obs))  # [N, M, A]
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"value_samples must return [N, M, A], got shape {arr.shape}")
    if arr.shape[1] < 2:
        raise ValueError(
            f"need M >= 2 value samples for std; got M = {arr.shape[1]}"
        )

    # Greedy state-value: argmax over mean Q, then the value at that action.
    qbar = arr.mean(axis=1)          # [N, A]
    v_bar = qbar.max(axis=1)         # [N]  = max_a Q̄(s, a)

    # Per-sample state value and its spread.
    v_samples = arr.max(axis=2)      # [N, M]
    sigma_v = v_samples.std(axis=1, ddof=0)  # [N]

    return v_bar, sigma_v


# --------------------------------------------------------------------------- #
# Step 5 — the statistic
# --------------------------------------------------------------------------- #

def _is_degenerate(x: np.ndarray) -> bool:
    """True when ``x`` has no spread beyond float32 rounding noise.

    Mirrors ``analysis.diagnostics_battery.is_degenerate`` exactly so the two modules share
    the same degeneracy convention (see that function's docstring for the failure this
    prevents — Spearman ranking pure float32 rounding noise).
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return True
    scale = max(1.0, float(np.abs(arr).max()))
    tol = _DEGENERACY_EPS_MULTIPLE * float(np.finfo(np.float32).eps) * scale
    return float(np.ptp(arr)) <= tol


def _spearman_rho(x: np.ndarray, y: np.ndarray, min_n: int = 3) -> tuple[float, str]:
    """Spearman ρ with item-20 degeneracy guard.

    Returns ``(rho, reason)`` where ``reason`` is an empty string on success and a
    human-readable explanation when ``rho`` is NaN.
    """
    from scipy import stats as _stats  # lazy import — scipy is analysis-side only

    n = int(x.size)
    if n < min_n:
        return NA, f"n = {n} < {min_n}; Spearman undefined"
    if _is_degenerate(x):
        return NA, "σ_V is constant over start states; Spearman undefined"
    if _is_degenerate(y):
        return NA, "|V̄ − G| is constant over start states; Spearman undefined"
    rho = float(_stats.spearmanr(x, y).statistic)
    if math.isnan(rho):
        return NA, "scipy.stats.spearmanr returned nan"
    return rho, ""


def initial_state_return_alignment(
    v_bar: np.ndarray,
    sigma_v: np.ndarray,
    empirical_returns: np.ndarray,
    game: str,
    *,
    n_seeds: int,
    n_unique_starts: int,
) -> Diag8Result:
    """Spearman ρ between σ_V(s₀) and |V̄(s₀) − G(s₀)| over unique start states.

    σ_V = 0 at a start state is **kept** in the Spearman, not excluded.  An agent that
    reports zero spread at a state where its value prediction is wrong is making a claim the
    diagnostic exists to catch, and dropping those states would remove the most diagnostic
    evidence.  This is the same convention as battery diagnostics 1, 2, 4, 7; only
    diagnostic 5 excludes σ = 0 (because log 0 is undefined).

    Parameters
    ----------
    v_bar:
        Predicted state values ``[N]`` (from :func:`value_and_uncertainty`).
    sigma_v:
        Uncertainty at each start state ``[N]`` (from :func:`value_and_uncertainty`).
    empirical_returns:
        Discounted empirical returns ``[N]`` (from :func:`rollout_from_snapshot`).
    game:
        MinAtar game name, for the result record.
    n_seeds, n_unique_starts:
        Provenance counts for the result record.

    Returns
    -------
    Diag8Result
    """
    vb = np.asarray(v_bar, dtype=np.float64).ravel()
    sv = np.asarray(sigma_v, dtype=np.float64).ravel()
    gr = np.asarray(empirical_returns, dtype=np.float64).ravel()

    n = len(vb)
    if n != len(sv) or n != len(gr):
        raise ValueError(
            f"v_bar ({len(vb)}), sigma_v ({len(sv)}), returns ({len(gr)}) must have the same length"
        )

    error = np.abs(vb - gr)
    rho, reason = _spearman_rho(sv, error)
    defined = not math.isnan(rho)

    return Diag8Result(
        game=game,
        value=rho,
        n_seeds=n_seeds,
        n_unique_starts=n_unique_starts,
        n_used=n if defined else 0,
        n_excluded=0,  # σ_V = 0 is kept; no exclusion criteria in this diagnostic
        defined=defined,
        reason=reason,
        extra={
            "mean_v_bar": float(vb.mean()),
            "mean_return": float(gr.mean()),
            "mean_sigma_v": float(sv.mean()),
        },
    )


# --------------------------------------------------------------------------- #
# Rollout policy helper
# --------------------------------------------------------------------------- #

def make_mean_greedy_policy(sampler: ValueSampler) -> Callable[[np.ndarray], int]:
    """Return a policy that takes argmax over the mean Q-function.

    Uses ``sampler.value_samples`` called on a single observation to evaluate the mean Q and
    returns the greedy action.  This is the natural "consensus" policy for both ensemble and
    NoisyNet agents:

    * **Ensemble**: argmax over mean Q across all K heads (the Bayes-optimal point estimate).
    * **NoisyNet**: argmax over mean Q from M measurement-time draws (never the operational
      noise stream, so running the diagnostic does not perturb the acting policy).

    The returned callable is stateless — safe to call concurrently with the sampler from
    any iteration order.
    """
    def policy(obs: np.ndarray) -> int:
        samples = sampler.value_samples(np.asarray(obs)[None])  # [1, M, A]
        return int(np.asarray(samples[0], dtype=np.float64).mean(axis=0).argmax())
    return policy


# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #

def run_diagnostic_8(
    sampler: ValueSampler,
    game: str,
    *,
    master_seed: int,
    cell_id: str,
    seed_index: int,
    n_seeds: int = N_START_SEEDS,
    uniqueness_threshold: int = UNIQUENESS_THRESHOLD,
    max_rollout_steps: int = DEFAULT_MAX_ROLLOUT_STEPS,
    gamma: float = DEFAULT_GAMMA,
    rollout_policy: Callable[[np.ndarray], int] | None = None,
) -> Diag8Result:
    """Full Diagnostic 8 pipeline for one game at one checkpoint.

    Collect → deduplicate → rollout → value+uncertainty → statistic.

    Parameters
    ----------
    sampler:
        A :class:`ValueSampler` (``EnsembleValueSampler`` or ``NoisyNetValueSampler``).
        DDQN has no sample distribution and should not call this function.
    game:
        One of :data:`src.minatar_env.MINATAR_GAMES`.
    master_seed, cell_id, seed_index:
        The run's frozen RNG coordinates.  The ``probe_set`` stream is derived from these.
    n_seeds:
        Episode-start seeds to draw (Fix #6 default: 100).
    uniqueness_threshold:
        Minimum distinct start observations required; result is ``defined=False`` below this
        (Fix #6 default: 20).
    max_rollout_steps:
        Truncation horizon per start state.
    gamma:
        Discount factor for empirical-return computation.
    rollout_policy:
        Optional explicit policy ``(obs: np.ndarray) -> int``.  If ``None`` (default),
        :func:`make_mean_greedy_policy` is used — the recommended choice.

    Returns
    -------
    Diag8Result
        ``defined=False`` when the game is uninformative (too few unique starts) or the
        Spearman statistic is undefined (constant σ_V or |V̄ − G|).
    """
    from src.minatar_env import MinAtarEnv

    if rollout_policy is None:
        rollout_policy = make_mean_greedy_policy(sampler)

    # 1. Collect episode starts from the probe_set stream.
    observations, snapshots = collect_episode_starts(
        game,
        master_seed=master_seed,
        cell_id=cell_id,
        seed_index=seed_index,
        n_seeds=n_seeds,
    )

    # 2. Deduplicate — Fix #6: uniqueness is by byte equality of the observation.
    unique_obs, unique_snaps = deduplicate_starts(observations, snapshots)
    n_unique = len(unique_snaps)

    if n_unique < uniqueness_threshold:
        return _uninformative(
            game,
            f"only {n_unique} unique start observations from {n_seeds} seeds "
            f"(threshold = {uniqueness_threshold}); diagnostic omitted as uninformative",
            n_seeds=n_seeds,
            n_unique=n_unique,
        )

    # 3. Rollouts — fresh env so the training env is untouched.
    rollout_env = MinAtarEnv(
        game,
        master_seed=master_seed,
        cell_id=cell_id,
        seed_index=seed_index,
    )
    # Initialise the env before restore_state (ensures internal game object is fully set up).
    rollout_env.reset()

    empirical_returns = np.array(
        [
            rollout_from_snapshot(
                rollout_env,
                snap,
                rollout_policy,
                max_steps=max_rollout_steps,
                gamma=gamma,
            )
            for snap in unique_snaps
        ],
        dtype=np.float64,
    )

    # 4. Value and uncertainty at start observations.
    v_bar, sigma_v = value_and_uncertainty(sampler, unique_obs)

    # 5. Alignment statistic.
    return initial_state_return_alignment(
        v_bar,
        sigma_v,
        empirical_returns,
        game,
        n_seeds=n_seeds,
        n_unique_starts=n_unique,
    )


# --------------------------------------------------------------------------- #
# Serialisation helper
# --------------------------------------------------------------------------- #

def diag8_to_record(result: Diag8Result, *, code_ver: dict | None = None) -> dict:
    """Serialise a :class:`Diag8Result` to a JSON-compatible dict with code provenance.

    ``code_ver`` should be ``conventions.code_version()``; callers pass it in so a trainer
    building a run record does not need a second import of ``conventions``.  If ``None``,
    :func:`~src.utils.conventions.code_version` is called here.
    """
    rec = result.as_dict()
    rec["code_version"] = code_ver if code_ver is not None else conventions.code_version()
    return rec
