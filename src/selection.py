"""Selection statistic and tie-breaking for the pre-registered searches.

Freeze-list item 3 pins the selection statistic for **every** search in the study:

    "**IQM** (interquartile mean) throughout; ties broken by the lower parameter value."

This module implements exactly that and nothing else. It is deliberately free of any
run-execution machinery so that the frozen statistic can be unit-tested against
hand-computed values without touching an environment, a network, or a config.

Three consumers, all pinned by the pre-registration:

* **Backbone tuning** (item 2 / Class 1) — one random search on the ε-greedy DDQN
  backbone over development sizes; the winner is inherited identically by all cells.
* **Factor-specific mini-searches** (item 2 / Class 3) — ``prior_scale`` selected by IQM
  of the canonical prior-on cell ``(episodic, on, 10)``; ``eps_schedule`` for
  ``ensemble_mean_eps`` selected by IQM of ``(mean_eps, off, 10)``.
* **K_shared** (item 19 / spec §"MinAtar method set") — a *joint* rule: for each
  ``K in {5, 10, 20}``, take each ensemble method's best-config IQM, average the two, and
  argmax over K. Implemented as :func:`select_k_shared`, which composes this module's
  primitives rather than re-deriving them.

Why IQM. The interquartile mean discards the top and bottom quartile of the sample and
averages the middle half. On RL learning curves it is markedly less sensitive to the
occasional diverged or jackpot seed than the mean, while unlike the median it still uses
most of the data. This is the statistic Agarwal et al. (2021) argue for in the
few-seed regime, which is the regime the pilot tier runs in.

Definition used here. For n values sorted ascending, IQM averages the values whose
*fractional* rank lies in [0.25, 0.75]. When n is not a multiple of 4 the boundary
observations are included with fractional weight, so the estimator is continuous in the
data and no observation is silently dropped by an integer-truncation accident. n = 4k
reduces to the plain "drop k from each end" form. See :func:`iqm` for the worked
convention and the exactness guarantees the tests pin.

Ties. Item 3's tie-breaker ("lower parameter value") is a *total* order requirement, so
ties are broken on the candidate's own sort key, never on run order or dict ordering —
both of which would make selection non-reproducible. :func:`select_best` therefore
requires each candidate to expose a sort key and compares ``(-iqm, key)`` lexicographically.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "iqm",
    "Candidate",
    "select_best",
    "select_k_shared",
    "SelectionResult",
]


# --------------------------------------------------------------------------- #
# The frozen selection statistic
# --------------------------------------------------------------------------- #
def iqm(values: Iterable[float]) -> float:
    """Interquartile mean — the frozen selection statistic (freeze item 3).

    Averages the middle half of the sample: the values whose fractional rank lies in
    ``[0.25, 0.75]``. Boundary observations enter with fractional weight, so the
    estimator is continuous in the data.

    Concretely, for ``n`` sorted values the ``i``-th observation (0-based) covers the
    fractional-rank interval ``[i/n, (i+1)/n]``. Its weight is the length of the overlap
    of that interval with ``[0.25, 0.75]``, normalized so the weights sum to 1. For
    ``n = 4k`` this puts weight exactly 0 on the lowest and highest ``k`` observations
    and equal weight on the middle ``2k`` — the familiar "trim a quartile from each end"
    form. For other ``n`` it splits the boundary observations rather than rounding.

    Parameters
    ----------
    values:
        Finite sample of per-seed scores. Must be non-empty. NaNs are rejected rather
        than silently dropped: a NaN here means an upstream run failed or a metric was
        undefined, and the §3.3 undefined-value policy requires such cases to be
        *counted and excluded upstream*, not absorbed into a statistic.

    Returns
    -------
    float
        The interquartile mean.

    Raises
    ------
    ValueError
        If ``values`` is empty or contains a non-finite entry.

    Examples
    --------
    >>> iqm([1.0, 2.0, 3.0, 4.0])          # n = 4: keeps the middle two
    2.5
    >>> iqm([0.0, 1.0, 2.0, 3.0, 100.0])   # the outlier is trimmed away
    1.5
    >>> iqm([5.0])                          # degenerate but well-defined
    5.0
    """
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("iqm() requires at least one value")
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            "iqm() received a non-finite value; undefined per-seed scores must be "
            "counted and excluded by the caller (§3.3 undefined-value policy), "
            "not passed into the selection statistic"
        )

    n = arr.size
    arr = np.sort(arr)

    # Overlap of each observation's fractional-rank interval with [0.25, 0.75].
    lo = np.arange(n, dtype=np.float64) / n
    hi = np.arange(1, n + 1, dtype=np.float64) / n
    weights = np.clip(np.minimum(hi, 0.75) - np.maximum(lo, 0.25), 0.0, None)

    total = weights.sum()
    if total <= 0.0:  # pragma: no cover - unreachable: [0.25,0.75] always overlaps some cell
        raise ValueError("degenerate IQM weights")
    return float(np.dot(arr, weights) / total)


# --------------------------------------------------------------------------- #
# Candidates and selection
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Candidate:
    """One point in a search space, with its per-seed scores.

    Attributes
    ----------
    label:
        Human-readable identifier, used in logs and in the selection record.
    params:
        The hyperparameter assignment this candidate represents.
    scores:
        Per-seed scores on the search objective. The IQM of these is the candidate's
        selection score.
    sort_key:
        The value the frozen tie-breaker orders on ("lower parameter value"). For a
        single-parameter search this is that parameter's numeric value. For a
        multi-parameter search it must be a tuple giving a deterministic total order —
        see :func:`select_best`, which refuses to guess one.
    """

    label: str
    params: Mapping[str, Any]
    scores: tuple[float, ...]
    sort_key: Any

    @property
    def iqm(self) -> float:
        """IQM of this candidate's per-seed scores."""
        return iqm(self.scores)


@dataclass(frozen=True)
class SelectionResult:
    """The outcome of one search, in a form that can be committed alongside the runs.

    ``ranking`` carries every candidate in decision order, so the selection is auditable
    after the fact rather than only its winner being recorded. ``tie_broken`` records
    whether item 3's tie-breaker actually fired, which is exactly the kind of event a
    reviewer should be able to see without re-running anything.
    """

    winner: Candidate
    ranking: tuple[Candidate, ...]
    tie_broken: bool

    @property
    def winning_iqm(self) -> float:
        return self.winner.iqm

    def as_record(self) -> dict[str, Any]:
        """A JSON-serializable summary for the selection record."""
        return {
            "winner": {
                "label": self.winner.label,
                "params": dict(self.winner.params),
                "iqm": self.winner.iqm,
                "n_seeds": len(self.winner.scores),
            },
            "tie_broken": self.tie_broken,
            "ranking": [
                {"label": c.label, "iqm": c.iqm, "params": dict(c.params)} for c in self.ranking
            ],
        }


def _require_total_order(candidates: Sequence[Candidate]) -> None:
    """Reject search spaces whose tie-breaker is not a total order.

    Item 3 says ties break on "the lower parameter value". If two distinct candidates
    share a sort key, that instruction is silent and selection would fall through to
    whatever order the candidates happened to arrive in — i.e. non-reproducible. We fail
    loudly instead of picking arbitrarily.
    """
    keys = [c.sort_key for c in candidates]
    try:
        duplicated = len(set(keys)) != len(keys)
    except TypeError as exc:  # unhashable sort key
        raise TypeError(
            "Candidate.sort_key must be hashable and comparable so the frozen "
            "tie-breaker is a total order; got an unhashable key"
        ) from exc
    if duplicated:
        raise ValueError(
            "two candidates share a sort_key, so the frozen tie-breaker "
            "('ties → lower parameter value') does not determine a winner; give each "
            "candidate a distinct sort_key (e.g. a tuple over all searched parameters)"
        )


def select_best(candidates: Iterable[Candidate]) -> SelectionResult:
    """Select by IQM, breaking ties on the lower parameter value (freeze item 3).

    Sorts on ``(-iqm, sort_key)``: highest IQM first, and among equal IQMs the lower
    parameter value. The comparison is total (see :func:`_require_total_order`), so the
    winner does not depend on the order candidates were generated or on any dict
    iteration order.

    Parameters
    ----------
    candidates:
        The evaluated search points. Must be non-empty and must have pairwise-distinct
        ``sort_key``s.

    Returns
    -------
    SelectionResult
        Winner plus the full ranking and whether the tie-breaker fired.

    Raises
    ------
    ValueError
        If ``candidates`` is empty, or two candidates share a ``sort_key``.
    """
    cands = list(candidates)
    if not cands:
        raise ValueError("select_best() requires at least one candidate")
    _require_total_order(cands)

    scored = sorted(cands, key=lambda c: (-c.iqm, c.sort_key))
    best = scored[0]
    # The tie-breaker "fired" iff some other candidate matched the winner's IQM exactly.
    tie_broken = any(
        other is not best and math.isclose(other.iqm, best.iqm, rel_tol=0.0, abs_tol=0.0)
        for other in scored
    )
    return SelectionResult(winner=best, ranking=tuple(scored), tie_broken=tie_broken)


# --------------------------------------------------------------------------- #
# K_shared — the joint rule (freeze item 19 / spec MinAtar method set)
# --------------------------------------------------------------------------- #
def select_k_shared(
    per_method_candidates: Mapping[str, Mapping[int, Sequence[Candidate]]],
    *,
    k_values: Sequence[int] = (5, 10, 20),
) -> tuple[int, dict[int, float], dict[str, dict[int, Candidate]]]:
    """Apply the frozen ``K_shared`` selection rule.

    The rule, quoted from the spec's MinAtar method-set row:

        "during the pilot tier, K ∈ {5, 10, 20} is searched by both ensemble methods;
        K_shared = argmax over K of the **mean of the two ensemble methods' best-config
        IQM** on the tuning games at that K; both inherit K_shared unchanged."

    This is *not* the same as selecting each method's own best K — it is a joint rule, and
    the whole point is that both ensemble methods are forced onto one shared value so the
    K axis is not confounded with per-method tuning. Note the order of operations: best
    config *within* each (method, K), then mean *across* methods, then argmax over K.

    Ties over K fall back to item 3's tie-breaker on K itself (the lower K wins), which is
    both the frozen rule and the compute-cheaper choice.

    Parameters
    ----------
    per_method_candidates:
        ``{method_name: {K: [candidates...]}}``. Exactly two methods are expected (the two
        ensemble methods); every method must supply candidates for every K in ``k_values``,
        because a missing cell would make the cross-method mean incomparable across K.
    k_values:
        The searched K axis. Defaults to the frozen ``{5, 10, 20}``.

    Returns
    -------
    (k_shared, mean_iqm_by_k, best_by_method_and_k)
        The selected K, the cross-method mean IQM at each K (the quantity argmaxed), and
        each method's winning candidate at each K for the audit record.

    Raises
    ------
    ValueError
        If the method count is not 2, or any (method, K) cell is missing or empty.
    """
    methods = sorted(per_method_candidates)
    if len(methods) != 2:
        raise ValueError(
            f"K_shared is a joint rule over the TWO ensemble methods; got {len(methods)}: {methods}"
        )

    best_by: dict[str, dict[int, Candidate]] = {m: {} for m in methods}
    for m in methods:
        for k in k_values:
            cell = per_method_candidates[m].get(k)
            if not cell:
                raise ValueError(
                    f"no candidates for method {m!r} at K={k}; the cross-method mean is "
                    "not comparable across K unless every (method, K) cell is populated"
                )
            best_by[m][k] = select_best(cell).winner

    mean_iqm_by_k = {k: float(np.mean([best_by[m][k].iqm for m in methods])) for k in k_values}
    # argmax over K, ties → lower K (item 3 tie-breaker applied to the K axis).
    k_shared = min(k_values, key=lambda k: (-mean_iqm_by_k[k], k))
    return k_shared, mean_iqm_by_k, best_by


# --------------------------------------------------------------------------- #
# Convenience: build candidates from a scoring callable
# --------------------------------------------------------------------------- #
def score_candidates(
    points: Sequence[Mapping[str, Any]],
    score_fn: Callable[[Mapping[str, Any]], Sequence[float]],
    *,
    sort_key_fn: Callable[[Mapping[str, Any]], Any],
    label_fn: Callable[[Mapping[str, Any]], str] | None = None,
) -> list[Candidate]:
    """Evaluate a list of search points into :class:`Candidate` objects.

    Kept separate from :func:`select_best` so that selection stays a pure function of
    already-collected scores — which is what lets the frozen statistic be tested, and
    audited later, without executing any runs.

    Parameters
    ----------
    points:
        The hyperparameter assignments to evaluate.
    score_fn:
        Maps one point to its per-seed scores on the search objective.
    sort_key_fn:
        Maps one point to its tie-break key. Required rather than inferred: for a
        multi-parameter search only the caller knows which ordering item 3's "lower
        parameter value" refers to.
    label_fn:
        Optional labeller; defaults to a sorted ``k=v`` join.
    """
    if label_fn is None:

        def label_fn(p: Mapping[str, Any]) -> str:
            return ",".join(f"{k}={p[k]}" for k in sorted(p))

    return [
        Candidate(
            label=label_fn(p),
            params=dict(p),
            scores=tuple(float(s) for s in score_fn(p)),
            sort_key=sort_key_fn(p),
        )
        for p in points
    ]
