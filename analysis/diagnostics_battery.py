"""The §3.3 uncertainty-diagnostics battery — reducers over committed substrate ``.npz`` files.

Diagnostics 1-5 and 7, frozen verbatim in the pre-registration ("Uncertainty diagnostics —
mathematical definitions"; freeze item 14). Diagnostic 6 (temporal persistence) is in
``src/diagnostics/temporal_persistence.py`` because it must observe the agent *during* an
episode; diagnostic 8 is the MinAtar analogue (``analysis/clone_reproduction.py``); diagnostic
9 is the undefined-value policy, implemented here as the shared :class:`Undefined` convention
rather than as a separate statistic.

Everything here is **post-hoc**: input is a ``[T, S, M, A]`` sample tensor plus its ``[T, S]``
visitation counts, and no reducer touches the training path. That is deliberate — the substrate
module's own docstring gives the reason (raw samples are persisted precisely so the reduction
can be revised without re-running training).

Conventions that are shared, and that all six inherit
-----------------------------------------------------

``σ = std over the M samples with ddof=0``. Not a stylistic choice: for the ensembles the M
heads *are* the population of samples, and ddof=1 would inflate σ by sqrt(K/(K-1)) — 5.4% at
K = 10 — on a quantity whose cross-method comparison is the entire point. Matches
``samplers.disagreement_summary``.

``Ties → lowest action index`` everywhere (diagnostics 2 and 3 state it explicitly; applying it
uniformly keeps argmaxes consistent across the battery). ``numpy.argmax`` already does this.

**Undefined statistics are recorded NA, excluded from aggregation, and counted** (item 20). No
imputation, ever. Each reducer returns a :class:`Result` carrying ``value`` (possibly ``nan``),
``n_used``, ``n_excluded``, and ``reason`` — so an excluded statistic is visible in the record
rather than silently absent. **σ = 0 is a substantive measurement wherever σ is a value rather
than a divisor**: it is kept in diagnostics 1, 2, 4 and 7, and excluded *only* in diagnostic 5,
where log σ is undefined — which is exactly where the pre-registration says to exclude it.

Probe alignment
---------------
Probe states are stored positionally in row-major ``(row, col)`` order with ``c <= r``
(``samplers.deep_sea_probe_states``); ``probe_index`` is the ``[S, 2]`` array of those
coordinates. Q* is indexed ``[row, col, action]``. Every reducer takes ``q_star_flat`` — Q*
gathered onto the probe ordering, ``[S, A]`` — built once by :func:`align_q_star`, so no
reducer re-derives the ordering.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats

#: Item 20's undefined-value marker. NA is represented as ``nan`` in numeric fields; the
#: accompanying ``reason`` says why, so "undefined" is never confused with "zero".
NA = float("nan")


@dataclass(frozen=True)
class Result:
    """One diagnostic at one checkpoint, with its exclusion accounting (item 20)."""

    name: str
    value: float
    n_used: int
    n_excluded: int = 0
    reason: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def defined(self) -> bool:
        return not np.isnan(self.value)


def _na(name: str, reason: str, n_excluded: int = 0) -> Result:
    return Result(name=name, value=NA, n_used=0, n_excluded=n_excluded, reason=reason)


def align_q_star(q_star: np.ndarray, probe_index: np.ndarray) -> np.ndarray:
    """Gather ``Q*[row, col, a]`` onto the probe ordering → ``[S, A]``.

    Called once per run; every reducer consumes the result. Keeping this in one place is what
    stops a reducer from re-deriving the probe ordering and silently mis-aligning σ with Q*.
    """
    q = np.asarray(q_star, dtype=np.float64)
    idx = np.asarray(probe_index, dtype=np.int64)
    if idx.ndim != 2 or idx.shape[1] != 2:
        raise ValueError(f"probe_index must be [S, 2], got {idx.shape}")
    if q.ndim != 3:
        raise ValueError(f"q_star must be [N, N, A], got {q.shape}")
    return q[idx[:, 0], idx[:, 1], :]


def sigma_and_mean(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(σ[S, A], Q̄[S, A])`` from one ``[S, M, A]`` checkpoint. ddof=0 — see module docstring."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"expected [S, M, A], got {arr.shape}")
    if arr.shape[1] < 2:
        raise ValueError(f"need M >= 2 value samples, got M = {arr.shape[1]}")
    return arr.std(axis=1, ddof=0), arr.mean(axis=1)


#: Multiple of float32 eps below which a spread is treated as zero. The substrate stores samples
#: as float32, so every quantity here is float32-derived and a mathematically constant vector
#: arrives with ptp on the order of eps·|scale| rather than exactly 0.
_DEGENERACY_EPS_MULTIPLE = 8.0


def is_degenerate(x: np.ndarray) -> bool:
    """True when ``x`` has no spread beyond float32 rounding noise.

    **This is a correctness guard, not a cosmetic tolerance, and it was added in response to an
    observed failure.** An exact ``ptp(x) == 0`` test never fires on float32-derived data: with
    per-action sample noise that is perfectly correlated across the two actions, u_g(s) is
    mathematically 0 at every state, yet the realized values differ in the last float32 bit.
    Spearman then ranks pure rounding noise and returns |ρ| ≈ 0.99 — a fabricated "strong
    alignment" that no downstream check would question, since the value is finite, in range, and
    of plausible magnitude.

    The threshold is relative to the data's own scale (``eps · max(1, max|x|)``) because σ and
    gap values in a real run range over several orders of magnitude. In a run with genuine
    heteroscedasticity ptp exceeds this by many orders of magnitude, so the guard fires only on
    the degenerate case it is for.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return True
    scale = max(1.0, float(np.abs(arr).max()))
    tol = _DEGENERACY_EPS_MULTIPLE * float(np.finfo(np.float32).eps) * scale
    return float(np.ptp(arr)) <= tol


def _spearman(x: np.ndarray, y: np.ndarray, name: str, min_n: int = 3) -> Result:
    """Spearman ρ with item-20 exclusion accounting.

    ρ is undefined when either input is constant (zero variance ⇒ 0/0 in the rank correlation),
    which scipy reports as nan with a warning. That is a genuine NA, not a zero: a run whose σ
    is identical everywhere has no monotone relationship to detect, and recording 0.0 would
    average into aggregates as though alignment had been measured and found absent.

    Constancy is tested with :func:`is_degenerate` rather than ``ptp == 0`` — see that function
    for the failure this prevents.
    """
    n = int(x.size)
    if n < min_n:
        return _na(name, f"n = {n} < {min_n} usable (s, a) pairs", n_excluded=n)
    if is_degenerate(x) or is_degenerate(y):
        which = "uncertainty" if is_degenerate(x) else "error"
        return _na(name, f"{which} is constant over the probe set; Spearman undefined", n)
    rho = float(stats.spearmanr(x, y).statistic)
    if np.isnan(rho):
        return _na(name, "scipy returned nan", n)
    return Result(name=name, value=rho, n_used=n)


# --------------------------------------------------------------------------- #
# 1. Marginal alignment (RQ2-L primary)
# --------------------------------------------------------------------------- #
def marginal_alignment(samples: np.ndarray, q_star_flat: np.ndarray) -> Result:
    """Spearman ρ over (s, a) between σ(s, a) and |Q̄(s, a) − Q*(s, a)|.

    The RQ2-L primary. Every (s, a) pair enters: σ = 0 is kept, because here σ is a *value*
    being correlated, not a divisor — an estimator that reports zero uncertainty at a
    high-error pair is making a claim the diagnostic exists to catch, and dropping those pairs
    would remove the most diagnostic evidence in the set.
    """
    sigma, qbar = sigma_and_mean(samples)
    err = np.abs(qbar - np.asarray(q_star_flat, dtype=np.float64))
    return _spearman(sigma.ravel(), err.ravel(), "marginal_alignment")


# --------------------------------------------------------------------------- #
# 2. Action-gap alignment
# --------------------------------------------------------------------------- #
def action_gap_alignment(samples: np.ndarray, q_star_flat: np.ndarray) -> Result:
    """Spearman ρ between u_g(s) and |ĝ(s) − g*(s)|, top-2 by Q̄, ties by lowest index.

    The frozen definition pins something easy to get wrong: **g\\* uses the same a₁, a₂ chosen
    by Q̄**, not Q*'s own top-2. It is the error in the *estimated* gap between the actions the
    agent would actually compare — recomputing the pair under Q* would measure a different
    quantity and would silently make the statistic look better.

    u_g is the std over m of the per-sample gap ``Q_m(s,a₁) − Q_m(s,a₂)``, which is *not*
    recoverable from σ(s,a₁) and σ(s,a₂): it depends on the correlation between the two
    actions' samples. This is the reason the substrate stores raw samples.
    """
    arr = np.asarray(samples, dtype=np.float64)
    qs = np.asarray(q_star_flat, dtype=np.float64)
    if arr.shape[2] < 2:
        return _na("action_gap_alignment", "needs A >= 2 actions for a top-2 gap")
    qbar = arr.mean(axis=1)  # [S, A]
    order = np.argsort(-qbar, axis=1, kind="stable")  # stable ⇒ ties by lowest index
    a1, a2 = order[:, 0], order[:, 1]
    rows = np.arange(arr.shape[0])

    g_hat = qbar[rows, a1] - qbar[rows, a2]
    g_star = qs[rows, a1] - qs[rows, a2]  # SAME a1, a2 — see docstring
    u_g = (arr[rows, :, a1] - arr[rows, :, a2]).std(axis=1, ddof=0)
    return _spearman(u_g, np.abs(g_hat - g_star), "action_gap_alignment")


# --------------------------------------------------------------------------- #
# 3. Incorrect-argmax flagging
# --------------------------------------------------------------------------- #
def incorrect_argmax_flagging(samples: np.ndarray, q_star_flat: np.ndarray) -> Result:
    """Rank-biserial r = 2·(AUC − 0.5) relating disagreement d(s) to argmax incorrectness e(s).

    ``optimal set = Argmax_a Q*(s,·)`` is a SET: a state whose greedy action ties for optimal
    under Q* is **not** incorrect. Using a scalar ``argmax(Q*)`` instead would flag ties as
    errors, and in DeepSea ties are common wherever the two actions lead to equal-value cells.

    d(s) = 1 − modal fraction of the per-sample greedy actions (modal ties by lowest index).
    AUC is computed via the Mann-Whitney U identity ``AUC = U / (n₁·n₀)`` with midranks, so
    tied d values contribute 0.5 exactly as the probability statement requires. Positive r =
    greater disagreement at incorrect states.
    """
    arr = np.asarray(samples, dtype=np.float64)
    qs = np.asarray(q_star_flat, dtype=np.float64)
    n_states, _, n_actions = arr.shape

    greedy_hat = arr.mean(axis=1).argmax(axis=1)  # [S], ties → lowest index
    # Argmax SET under Q*, with an exact-max comparison (Q* is computed by exact DP, so no
    # tolerance is warranted; a tolerance here would silently widen the optimal set).
    is_optimal = qs == qs.max(axis=1, keepdims=True)
    e = (~is_optimal[np.arange(n_states), greedy_hat]).astype(np.int64)

    per_sample_greedy = arr.argmax(axis=2)  # [S, M], ties → lowest index
    counts = np.stack(
        [(per_sample_greedy == a).sum(axis=1) for a in range(n_actions)], axis=1
    )  # [S, A]
    modal_fraction = counts.max(axis=1) / arr.shape[1]
    d = 1.0 - modal_fraction

    n1, n0 = int(e.sum()), int((1 - e).sum())
    if n1 == 0 or n0 == 0:
        which = "no incorrect-argmax states" if n1 == 0 else "no correct-argmax states"
        return _na("incorrect_argmax_flagging", f"{which}; AUC undefined", n_states)
    u = stats.mannwhitneyu(d[e == 1], d[e == 0], alternative="two-sided").statistic
    auc = float(u) / (n1 * n0)
    return Result(
        name="incorrect_argmax_flagging",
        value=2.0 * (auc - 0.5),
        n_used=n_states,
        extra={"auc": auc, "n_incorrect": n1, "n_correct": n0},
    )


# --------------------------------------------------------------------------- #
# 4. Optimal-path uncertainty
# --------------------------------------------------------------------------- #
def optimal_path_uncertainty(
    samples: np.ndarray, q_star_flat: np.ndarray, probe_index: np.ndarray, size: int
) -> Result:
    """Mean σ(s, a*(s)) along the optimal path, per depth; AUC over depth as the summary.

    The optimal path in DeepSea is the "always right" trajectory ``(r, r)`` for r = 0..N−1 —
    one state per depth, so the per-depth mean is a single σ and the depth profile has exactly
    N points. a*(s) is taken from Q* (ties → lowest index), not from Q̄: the diagnostic asks
    how uncertain the estimator is *along the optimal path*, which is a fixed set of (s, a)
    pairs independent of what the agent currently believes.

    The AUC summary is the mean over depth (trapezoid on a unit-spaced grid, normalized by
    N−1), so it is comparable across sizes.
    """
    sigma, _ = sigma_and_mean(samples)
    qs = np.asarray(q_star_flat, dtype=np.float64)
    idx = np.asarray(probe_index, dtype=np.int64)

    pos = {(int(r), int(c)): i for i, (r, c) in enumerate(idx)}
    depths, sig_path = [], []
    for r in range(int(size)):
        i = pos.get((r, r))  # diagonal = the always-right optimal path
        if i is None:
            continue
        a_star = int(np.argmax(qs[i]))  # ties → lowest index
        depths.append(r)
        sig_path.append(float(sigma[i, a_star]))
    if len(sig_path) < 2:
        return _na("optimal_path_uncertainty", f"only {len(sig_path)} path states resolved")

    arr = np.asarray(sig_path)
    auc = float(np.trapezoid(arr, dx=1.0) / (len(arr) - 1))
    return Result(
        name="optimal_path_uncertainty",
        value=auc,
        n_used=len(arr),
        extra={"per_depth": arr.tolist(), "depths": depths},
    )


# --------------------------------------------------------------------------- #
# 5. Visitation-conditioned decay
# --------------------------------------------------------------------------- #
def visitation_conditioned_decay(
    samples: np.ndarray, q_star_flat: np.ndarray, visitation: np.ndarray
) -> Result:
    """OLS slope of log σ(s, a*(s)) on log(1 + v(s)); raw probe states, unweighted.

    **This is the one diagnostic where σ = 0 is excluded** — log 0 is undefined, so those
    states are dropped and counted (item 20: excluded, counted, published, never imputed).
    Everywhere else in the battery σ = 0 is a substantive measurement and is kept.

    "Bins for display only" (frozen wording): the slope is fit on raw probe states. Binning
    before fitting would change the estimate.
    """
    sigma, _ = sigma_and_mean(samples)
    qs = np.asarray(q_star_flat, dtype=np.float64)
    v = np.asarray(visitation, dtype=np.float64).ravel()
    if v.shape[0] != sigma.shape[0]:
        raise ValueError(f"visitation [{v.shape[0]}] does not match S = {sigma.shape[0]}")

    a_star = np.argmax(qs, axis=1)  # ties → lowest index
    sig = sigma[np.arange(sigma.shape[0]), a_star]
    keep = sig > 0
    n_excluded = int((~keep).sum())
    if keep.sum() < 3:
        return _na(
            "visitation_conditioned_decay",
            f"only {int(keep.sum())} states with sigma > 0; slope undefined",
            n_excluded,
        )
    x = np.log1p(v[keep])
    y = np.log(sig[keep])
    if is_degenerate(x):
        return _na(
            "visitation_conditioned_decay",
            "log(1 + v) is constant over the probe set; OLS slope undefined",
            n_excluded,
        )
    fit = stats.linregress(x, y)
    return Result(
        name="visitation_conditioned_decay",
        value=float(fit.slope),
        n_used=int(keep.sum()),
        n_excluded=n_excluded,
        extra={"intercept": float(fit.intercept), "r_value": float(fit.rvalue)},
    )


# --------------------------------------------------------------------------- #
# 7. Empirical containment
# --------------------------------------------------------------------------- #
def empirical_containment(samples: np.ndarray, q_star_flat: np.ndarray) -> Result:
    """Fraction of (s, a) whose Q* lies in the central 80% empirical interval of {Q_m(s,a)}.

    Quantiles via ``numpy.quantile(..., method="linear")`` — the method is named in the frozen
    definition, so it is passed explicitly rather than left to the numpy default (which is
    "linear" today but is a library default, not a protocol commitment).

    Interval endpoints are inclusive. With M = K heads the 10th/90th percentiles are
    interpolated between order statistics, and a strict comparison would drop the exactly-on-
    the-boundary case that arises when Q* coincides with a sampled value.
    """
    arr = np.asarray(samples, dtype=np.float64)
    qs = np.asarray(q_star_flat, dtype=np.float64)
    lo = np.quantile(arr, 0.10, axis=1, method="linear")  # [S, A]
    hi = np.quantile(arr, 0.90, axis=1, method="linear")
    inside = (qs >= lo) & (qs <= hi)
    n = int(inside.size)
    return Result(
        name="empirical_containment",
        value=float(inside.mean()),
        n_used=n,
        extra={"nominal_level": 0.80},
    )


# --------------------------------------------------------------------------- #
# Battery driver
# --------------------------------------------------------------------------- #
#: Diagnostics computed here, in frozen order. 6 and 8 live elsewhere; 9 is the NA convention.
BATTERY = (
    "marginal_alignment",
    "action_gap_alignment",
    "incorrect_argmax_flagging",
    "optimal_path_uncertainty",
    "visitation_conditioned_decay",
    "empirical_containment",
)


def run_battery(
    samples: np.ndarray,
    q_star_flat: np.ndarray,
    probe_index: np.ndarray,
    size: int,
    visitation: np.ndarray | None = None,
) -> dict[str, Result]:
    """All six reducers on ONE checkpoint's ``[S, M, A]`` tensor."""
    out = {
        "marginal_alignment": marginal_alignment(samples, q_star_flat),
        "action_gap_alignment": action_gap_alignment(samples, q_star_flat),
        "incorrect_argmax_flagging": incorrect_argmax_flagging(samples, q_star_flat),
        "optimal_path_uncertainty": optimal_path_uncertainty(
            samples, q_star_flat, probe_index, size
        ),
        "empirical_containment": empirical_containment(samples, q_star_flat),
    }
    if visitation is None:
        out["visitation_conditioned_decay"] = _na(
            "visitation_conditioned_decay", "no visitation counts recorded for this checkpoint"
        )
    else:
        out["visitation_conditioned_decay"] = visitation_conditioned_decay(
            samples, q_star_flat, visitation
        )
    return {k: out[k] for k in BATTERY}


def run_battery_over_run(
    npz_path: str | Path,
    size: int,
    *,
    master_seed: int,
    cell_id: str,
    seed_index: int,
    gamma: float = 1.0,
) -> dict:
    """Run the battery at every checkpoint of one committed substrate ``.npz``.

    ``master_seed`` / ``cell_id`` / ``seed_index`` are **required, not conveniences**. DeepSea
    randomizes which action index advances toward the treasure *per row*, drawn from the frozen
    ``env_mapping`` stream keyed on exactly this triple (``deep_sea.DeepSea.__init__`` →
    ``conventions.deepsea_action_mapping``). Q* is therefore run-specific: constructing a
    DeepSea with a different key produces a Q* whose action axis is flipped on ~half the rows,
    and every reducer that indexes Q* by action — 1, 2, 3, 4, 5 — would then silently compare
    σ against the wrong ground truth. Nothing would raise; the alignment statistics would just
    drift toward zero. The returned record carries ``mapping_hash`` so a consumer can verify
    the Q* used here matches the mapping the run logged.

    ``gamma`` is likewise explicit: the comparison is only meaningful under matched discounting,
    so the caller supplies the agent's own discount rather than inheriting a default.
    """
    from src.deep_sea import DeepSea
    from src.diagnostics.samplers import deep_sea_probe_states

    data = np.load(Path(npz_path))
    samples = data["samples"]  # [T, S, M, A]
    steps = data["steps"]
    visitation = data["visitation"] if "visitation" in data.files else None

    _obs, probe_index = deep_sea_probe_states(int(size))
    env = DeepSea(
        size=int(size),
        master_seed=int(master_seed),
        cell_id=str(cell_id),
        seed_index=int(seed_index),
    )
    q_flat = align_q_star(env.q_star(gamma=gamma), probe_index)

    checkpoints = []
    for t in range(samples.shape[0]):
        res = run_battery(
            samples[t],
            q_flat,
            probe_index,
            int(size),
            None if visitation is None else visitation[t],
        )
        checkpoints.append(
            {"step": int(steps[t]), **{k: asdict(v) for k, v in res.items()}}
        )
    return {
        "npz": str(npz_path),
        "deep_sea_size": int(size),
        "master_seed": int(master_seed),
        "cell_id": str(cell_id),
        "seed_index": int(seed_index),
        "mapping_hash": env.mapping_hash,  # lets a consumer verify Q* matches the run
        "gamma": float(gamma),
        "n_checkpoints": len(checkpoints),
        "checkpoints": checkpoints,
    }


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Run the §3.3 diagnostics battery over a run's npz.")
    p.add_argument("npz", type=Path)
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--master-seed", type=int, required=True)
    p.add_argument("--cell-id", type=str, required=True)
    p.add_argument("--seed-index", type=int, required=True)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    rec = run_battery_over_run(
        args.npz,
        args.size,
        master_seed=args.master_seed,
        cell_id=args.cell_id,
        seed_index=args.seed_index,
        gamma=args.gamma,
    )
    text = json.dumps(rec, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"battery → {args.out} ({rec['n_checkpoints']} checkpoints)")
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
