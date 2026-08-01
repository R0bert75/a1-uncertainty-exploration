"""Pre-registered random search over the class-1 backbone (freeze item 2).

This is the *driver* that freeze item 2's backbone search needs and
:mod:`src.selection` deliberately does not provide. ``selection`` is a pure function of
already-collected scores — that purity is what lets the frozen statistic be unit-tested
without touching an environment — so something has to sample the search points, execute
them, and reduce their logs to the per-seed scores ``selection`` consumes. That is this
module, and it is the only place in the tree where those three concerns meet.

Four things are pinned by the protocol and are therefore not options here.

**The draws.** ``n_backbone = 12`` points from the frozen class-1 distributions
(Gap 2 in ``protocol/decisions/staged_stage3_protocol_fixes.md``), drawn from the
``hparam_search`` RNG stream. The distributions live in :data:`BACKBONE_SPACE`.

**The evaluation.** 3 seeds on each of the two development sizes (N ∈ {10, 20}) = 6 runs
per candidate; 12 × 6 = 72 runs, which is the budgeted share of the 120-run DeepSea tuning
allotment (the remaining 48 are the two class-3 mini-searches).

**The objective.** IQM across tuning seeds of the per-seed **area under the online
discovery-probability curve** — the unweighted mean of the ``discovery_prob`` metric over
the run's checkpoints — pooled across the two development sizes (Gap 4, owner-decided
2026-08-01). See :func:`discovery_auc` for why this and not the terminal indicator.

**The tie-break.** Lower parameter value, per freeze item 3, applied through
``selection.select_best`` on a sort key this module constructs (:func:`_sort_key`).

Two RNG facts drove the design and are pinned by tests; both are easy to get wrong.

1. **Tuning must not share randomness with evaluation.** ``derive_seed`` keys on
   ``(master_seed, cell_id, stream, seed_index)``. If a tuning run of the ε-greedy
   backbone reused the reference cell's ``cell_id`` (``episodic|off|K1``), then tuning
   seed 0 and the *evaluated* reference cell's seed 0 would receive byte-identical
   ``init``, ``env_mapping``, ``replay`` and ``action_noise`` streams — the backbone would
   be selected on exactly the environment instances it is later measured on. Tuning runs
   therefore live in their own ``cell_id`` namespace (:data:`TUNING_CELL_PREFIX`), which
   freeze item 1's "no reuse across cells" rule already requires.

2. **Candidates must share randomness with each other.** All candidates use the *same*
   tuning ``cell_id``, so at a given ``seed_index`` every candidate gets the same DeepSea
   action mapping and the same initialization stream. This is common random numbers across
   the search: the contrast between two candidates is then driven by their
   hyperparameters rather than by which environment instance they happened to draw. The
   ``cell_id`` deliberately does **not** encode the candidate index.

One consequence of (2) is a subtlety at the two development sizes. ``deepsea_action_mapping``
derives a length-``size`` mask from a size-independent stream, so the N=20 mapping's first
10 entries are *identical* to the N=10 mapping's. The two sizes of one candidate are
therefore not independent draws. That is left as-is on purpose: it is the behaviour the
committed cell configs already have, changing it here would make tuning runs use a
different mapping convention than the runs they are tuning for, and the pooled objective
treats the 6 runs as one sample rather than claiming independence between the two sizes.
"""

from __future__ import annotations

import functools
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src import config as config_mod
from src import selection
from src.utils import conventions

# --------------------------------------------------------------------------- #
# Frozen search space (Gap 2 / freeze item 2)
# --------------------------------------------------------------------------- #

#: Number of random-search draws for the class-1 backbone. Freeze item 2 via Gap 2.
N_BACKBONE = 12

#: Tuning seeds per candidate *per development size*. Freeze item 1 via Gap 3.
TUNING_SEEDS = (0, 1, 2)

#: DeepSea development sizes (freeze item 5).
DEV_SIZES = (10, 20)

#: The tuning namespace. Deliberately not a factorial ``cell_id``: tuning runs are not
#: cells and must not collide with one (see the module docstring, RNG fact 1).
TUNING_CELL_PREFIX = "tune"

#: Class-1 search distributions, frozen in Gap 2. ``kind`` is either ``log_uniform``
#: (continuous, sampled uniformly in log space) or ``choice`` (uniform over a finite set).
#: Replay capacity and optimizer are *fixed*, not searched, and so are absent here.
BACKBONE_SPACE: dict[str, dict[str, Any]] = {
    "lr": {"kind": "log_uniform", "low": 1e-4, "high": 1e-2},
    "batch_size": {"kind": "choice", "values": (32, 64, 128)},
    "target_update_period": {"kind": "choice", "values": (100, 500, 1000)},
    "hidden_width": {"kind": "choice", "values": (64, 128, 256)},
}

#: Parameters held fixed across the search (Gap 2: "not searched").
BACKBONE_FIXED: dict[str, Any] = {"buffer_capacity": 100_000, "optimizer": "adam"}

#: Order in which parameters are drawn. Fixed explicitly rather than relying on dict
#: insertion order, because the draw sequence *is* the search: a reordering would silently
#: produce a different set of 12 candidates from the same master seed.
DRAW_ORDER: tuple[str, ...] = ("lr", "batch_size", "target_update_period", "hidden_width")

# --------------------------------------------------------------------------- #
# Frozen class-3 mini-searches (Gap 2 / freeze item 2)
# --------------------------------------------------------------------------- #

#: Draws per class-3 mini-search. Gap 2: "``n_mini = 4`` draws each", at the same
#: 3 seeds × 2 development sizes as the backbone, "identical count for every method so the
#: equal-search-budget standard holds".
N_MINI = 4

#: The two frozen one-parameter searches. Each names the ``factor_specific`` key it varies,
#: its distribution, and the cell whose IQM selects it. ``arm`` is the *evaluation* arm of
#: that input cell, recorded for provenance only — the runs themselves execute under the
#: ``tune|`` namespace, never under the evaluation arm (RNG fact 1).
#:
#: Both searches vary a single parameter, so freeze item 3's "ties → the lower parameter
#: value" is literal here rather than the lexicographic reading the 4-D backbone needs.
MINI_SEARCHES: dict[str, dict[str, Any]] = {
    "prior_scale": {
        "param": "prior_scale",
        "space": {"kind": "log_uniform", "low": 0.1, "high": 10.0},
        "arm": "episodic|on|K10",
        "template": "configs/example_rpbdqn_deepsea_dev.yaml",
        "shared_by": "all prior=on cells",
    },
    "eps_schedule": {
        "param": "eps_end",
        "space": {"kind": "log_uniform", "low": 0.005, "high": 0.1},
        "arm": "ensemble_mean|off|K10",
        "template": "configs/cell_ensemble_mean_off_K10_deepsea_dev.yaml",
        "shared_by": "ensemble_mean_eps cells at both prior levels",
    },
}

#: Fraction of the budget over which ε decays linearly. Gap 2 fixes this ("linear decay over
#: the first 10% of the budget") and searches only the final ε, so it is a constant here, not
#: a second search dimension.
EPS_DECAY_BUDGET_FRACTION = 0.10

#: Bookkeeping key carrying a candidate's draw position through ``selection``. Not a
#: hyperparameter — excluded from :data:`DRAW_ORDER` so it never enters the tie-break, and
#: stripped before a point is written into a config.
_INDEX_KEY = "_draw_index"


def _sort_key(point: Mapping[str, Any]) -> tuple:
    """Total order for freeze item 3's ``ties → lower parameter value``.

    Item 3 names "the lower parameter value" in the singular, which is unambiguous only
    for the one-dimensional class-3 mini-searches. The backbone search is 4-dimensional,
    so "lower" is read lexicographically in the frozen :data:`DRAW_ORDER`. The order is
    part of the pre-registration, not a runtime choice, which is what keeps the tie-break
    total and reproducible; ``selection.select_best`` rejects a non-total key.
    """
    return tuple(point[name] for name in DRAW_ORDER)


def sample_backbone_points(
    master_seed: int,
    *,
    n: int = N_BACKBONE,
    space: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Draw ``n`` class-1 backbone candidates from the frozen distributions.

    Uses the ``hparam_search`` stream at ``seed_index=0`` under the tuning ``cell_id``, so
    the candidate field is a deterministic function of ``master_seed`` alone and can be
    regenerated for audit without re-running anything.

    Parameters are drawn in :data:`DRAW_ORDER`, one full parameter vector at a time, so
    that changing ``n`` extends the field rather than reshuffling it: the first 12 points
    of an ``n=24`` draw are exactly the ``n=12`` draw.
    """
    space = BACKBONE_SPACE if space is None else space
    rng = conventions.derive_numpy_generator(
        master_seed, f"{TUNING_CELL_PREFIX}|backbone", "hparam_search", 0
    )
    points: list[dict[str, Any]] = []
    for _ in range(n):
        point: dict[str, Any] = {}
        for name in DRAW_ORDER:
            spec = space[name]
            if spec["kind"] == "log_uniform":
                lo, hi = math.log(spec["low"]), math.log(spec["high"])
                point[name] = float(math.exp(rng.uniform(lo, hi)))
            elif spec["kind"] == "choice":
                values = spec["values"]
                point[name] = type(values[0])(values[int(rng.integers(len(values)))])
            else:  # pragma: no cover - guarded by test_search_space_kinds
                raise ValueError(f"unknown distribution kind {spec['kind']!r} for {name!r}")
        points.append(point)
    return points


# --------------------------------------------------------------------------- #
# The objective (Gap 4, owner-decided 2026-08-01)
# --------------------------------------------------------------------------- #


def discovery_auc(discovery_prob_by_checkpoint: Sequence[float]) -> float:
    """Per-seed area under the online discovery-probability curve.

    ``trainer.run_seed`` logs ``discovery_prob`` once per checkpoint as a **cumulative**
    indicator: ``float(discovered)``, which is 0 before the discovering episode and 1 at
    every checkpoint after it. The unweighted mean of that curve is therefore not merely
    "some continuous score" — for a run that discovers at checkpoint ``j`` of ``C`` it
    equals ``1 - j/C`` exactly, and 0 for a run that never discovers. The objective is a
    strictly decreasing function of *when* discovery happened, which is why it subsumes
    the episodes-to-first-discovery alternative without needing a censoring convention.

    Why not the terminal indicator: at 6 runs per candidate, IQM of a 0/1 6-vector takes
    only five distinct values (it collapses 0/6 with 1/6 and 5/6 with 6/6), so the search
    ties at the top of a 12-candidate field in 44–98 % of simulated draws, and freeze
    item 3's tie-break is uncorrelated with performance. Gap 4 has the full table.
    """
    values = list(discovery_prob_by_checkpoint)
    if not values:
        raise ValueError("discovery_prob curve is empty; a run logs at least one checkpoint")
    return float(np.mean(values))


def score_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, int], float]:
    """Reduce logged CSV rows to ``{(size, seed): discovery_auc}``.

    Consumes the committed log schema (``metric``, ``axis``, ``value``, ``seed``,
    ``checkpoint``, ``size_class``) rather than a bespoke format, so the objective is
    recomputable from any run already on disk — including retroactively, since
    ``discovery_prob`` has been logged at every checkpoint since the episode lane existed.
    """
    curves: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for row in rows:
        if row.get("metric") != "discovery_prob" or row.get("axis") != "online":
            continue
        key = (int(row["size"]), int(row["seed"]))
        curves.setdefault(key, []).append((int(row["checkpoint"]), float(row["value"])))
    return {
        key: discovery_auc([v for _, v in sorted(points)]) for key, points in curves.items()
    }


# --------------------------------------------------------------------------- #
# Candidate execution
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SearchRecord:
    """The auditable record of one completed search."""

    kind: str
    master_seed: int
    n_candidates: int
    seeds: tuple[int, ...]
    sizes: tuple[int, ...]
    objective: str
    points: list[dict[str, Any]] = field(default_factory=list)
    per_candidate: list[dict[str, Any]] = field(default_factory=list)
    winner_index: int = -1
    tie_broken: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "master_seed": self.master_seed,
            "n_candidates": self.n_candidates,
            "seeds": list(self.seeds),
            "sizes": list(self.sizes),
            "objective": self.objective,
            "runs_per_candidate": len(self.seeds) * len(self.sizes),
            "total_runs": self.n_candidates * len(self.seeds) * len(self.sizes),
            "points": self.points,
            "per_candidate": self.per_candidate,
            "winner_index": self.winner_index,
            "winner": self.points[self.winner_index] if self.points else None,
            "tie_broken": self.tie_broken,
        }

    def write(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n")
        return out


def candidate_config(
    template: config_mod.RunConfig,
    point: Mapping[str, Any],
    size: int,
    *,
    index: int,
    seeds: Sequence[int] = TUNING_SEEDS,
) -> config_mod.RunConfig:
    """Materialize one (candidate, size) pair as a validated :class:`RunConfig`.

    The candidate index appears in ``run_id`` (so logs are separable) but **not** in
    ``cell_id`` (so candidates share RNG streams — common random numbers, module docstring
    fact 2). The size appears in neither, for the same reason.

    ``hidden_width`` is expanded to the two-layer ``hidden_sizes`` the backbone actually
    takes; the search space carries a single width because the frozen distribution is over
    "network width (FC units)", one number, not a per-layer vector.
    """
    data = json.loads(json.dumps(template.data))  # deep copy through plain JSON types
    data["run_id"] = f"tune_backbone_c{index:02d}_N{size}"
    data["role"] = "exploratory"
    data["arm"] = f"{TUNING_CELL_PREFIX}|backbone"
    data.pop("cell_id", None)
    data.pop("size_class", None)

    backbone = dict(data.get("backbone") or {})
    backbone["lr"] = float(point["lr"])
    backbone["batch_size"] = int(point["batch_size"])
    backbone["target_update_period"] = int(point["target_update_period"])
    backbone["hidden_sizes"] = [int(point["hidden_width"])] * 2
    backbone["buffer_capacity"] = BACKBONE_FIXED["buffer_capacity"]
    data["backbone"] = backbone

    budget = dict(data.get("env_budget") or {})
    budget["deep_sea_size"] = int(size)
    data["env_budget"] = budget
    data["seeds"] = [int(s) for s in seeds]
    return config_mod.resolve_config(data)


def sample_mini_points(
    master_seed: int,
    which: str,
    *,
    n: int = N_MINI,
) -> list[dict[str, Any]]:
    """Draw the ``n`` class-3 candidates for mini-search ``which``.

    Each mini-search has its own ``hparam_search`` stream, keyed by its own tuning
    ``cell_id``, so the three searches are mutually independent draws: changing ``n_mini``
    for one cannot shift the other's field or the backbone's.
    """
    if which not in MINI_SEARCHES:
        raise ValueError(f"unknown mini-search {which!r}; expected one of {sorted(MINI_SEARCHES)}")
    spec = MINI_SEARCHES[which]["space"]
    rng = conventions.derive_numpy_generator(
        master_seed, f"{TUNING_CELL_PREFIX}|{which}", "hparam_search", 0
    )
    param = MINI_SEARCHES[which]["param"]
    points: list[dict[str, Any]] = []
    for _ in range(n):
        if spec["kind"] != "log_uniform":  # pragma: no cover - both frozen spaces are log-uniform
            raise ValueError(f"unsupported class-3 distribution kind {spec['kind']!r}")
        lo, hi = math.log(spec["low"]), math.log(spec["high"])
        points.append({param: float(math.exp(rng.uniform(lo, hi)))})
    return points


def mini_candidate_config(
    template: config_mod.RunConfig,
    which: str,
    point: Mapping[str, Any],
    size: int,
    *,
    index: int,
    seeds: Sequence[int] = TUNING_SEEDS,
) -> config_mod.RunConfig:
    """Materialize one (class-3 candidate, size) pair as a validated :class:`RunConfig`.

    Same identity discipline as :func:`candidate_config`: the candidate index rides in
    ``run_id`` only, and the tuning ``cell_id`` carries neither index nor size, so the four
    candidates of a mini-search share environment instances (common random numbers) while
    staying separable in the logs.

    The one asymmetry with the backbone search is the ε case. Gap 2 searches only the
    *final* ε and fixes the decay to "the first 10% of the budget", so ``eps_decay_steps``
    is *derived* from the run's own step budget rather than drawn. It is computed per size
    from :func:`config.step_budget`, which means the two sizes of one candidate get
    different decay lengths — that is the intent, since 10% of a budget is a budget-relative
    quantity, not a constant.
    """
    if which not in MINI_SEARCHES:
        raise ValueError(f"unknown mini-search {which!r}; expected one of {sorted(MINI_SEARCHES)}")
    data = json.loads(json.dumps(template.data))
    data["run_id"] = f"tune_{which}_c{index:02d}_N{size}"
    data["role"] = "exploratory"
    data["arm"] = f"{TUNING_CELL_PREFIX}|{which}"
    data.pop("cell_id", None)
    data.pop("size_class", None)

    budget = dict(data.get("env_budget") or {})
    budget["deep_sea_size"] = int(size)
    data["env_budget"] = budget
    data["seeds"] = [int(s) for s in seeds]

    factor = dict(data.get("factor_specific") or {})
    if which == "prior_scale":
        factor["prior_scale"] = float(point["prior_scale"])
    else:
        # DeepSea's step budget is exact, not estimated: every episode runs to the bottom
        # row and terminates there, so an episode is always exactly ``size`` steps
        # (test_deep_sea_episode_length_is_exactly_size pins this). ``config.step_budget``
        # is not usable here — it reads ``env_budget.total_steps``, which only MinAtar runs
        # carry; DeepSea is episode-budgeted.
        total_steps = int(budget["episodes"]) * int(size)
        eps_end = float(point["eps_end"])
        factor["eps_schedule"] = {
            "eps_start": 1.0,
            "eps_end": eps_end,
            "eps_decay_steps": max(1, int(round(EPS_DECAY_BUDGET_FRACTION * total_steps))),
        }
    data["factor_specific"] = factor
    return config_mod.resolve_config(data)


def run_candidate(
    template: config_mod.RunConfig,
    point: Mapping[str, Any],
    *,
    index: int,
    out_dir: str | Path,
    seeds: Sequence[int] = TUNING_SEEDS,
    sizes: Sequence[int] = DEV_SIZES,
    n_checkpoints: int = 20,
    config_fn: Callable[..., config_mod.RunConfig] = candidate_config,
) -> dict[str, Any]:
    """Execute one candidate over all (size, seed) pairs and score it.

    ``config_fn`` is the materializer — :func:`candidate_config` for the class-1 backbone,
    a partial of :func:`mini_candidate_config` for a class-3 mini-search. Everything after
    materialization (execution, per-run scoring, pooling, ordering) is identical across the
    three searches by construction, which is the point of routing them through one function:
    the mini-searches cannot silently acquire a different objective or pooling rule.

    Returns ``{"scores": [...], "per_run": {...}, "csvs": [...]}`` where ``scores`` is the
    pooled per-run objective vector — one entry per (size, seed) — in a deterministic
    ``(size, seed)`` order, which is what :func:`selection.score_candidates` consumes.

    Pooling is *unweighted across the two sizes*, per the Gap 4 sub-clause: the 6 runs form
    one sample, and IQM is taken over all of them rather than over per-size means. Taking
    per-size means first would halve the effective n before the interquartile trim and
    reintroduce exactly the coarseness the objective was chosen to avoid.
    """
    from src import trainer  # local import: trainer pulls torch, search's samplers do not

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_run: dict[str, float] = {}
    csvs: list[str] = []
    ordered: list[tuple[int, int]] = []
    for size in sizes:
        cfg = config_fn(template, point, size, index=index, seeds=seeds)
        csv_path = trainer.train(cfg, out, n_checkpoints=n_checkpoints)
        csvs.append(str(csv_path))
        rows = _read_rows(csv_path)
        scored = score_from_rows(rows)
        for seed in seeds:
            key = (int(size), int(seed))
            if key not in scored:
                raise RuntimeError(
                    f"candidate {index} produced no discovery_prob rows for "
                    f"size={size} seed={seed} in {csv_path}"
                )
            per_run[f"N{size}_s{seed}"] = scored[key]
            ordered.append(key)
    return {
        "scores": [per_run[f"N{s}_s{seed}"] for s, seed in ordered],
        "per_run": per_run,
        "csvs": csvs,
    }


def _read_rows(csv_path: str | Path) -> list[dict[str, Any]]:
    """Read a committed metrics CSV into row dicts, adding the ``size`` the run used.

    ``size`` is not a CSV column — the schema carries ``size_class`` (development /
    confirmatory), not the DeepSea N — so it is recovered from the run_id suffix this
    module writes. Kept here rather than widening the frozen log schema, which would
    change every run's identity fingerprint.
    """
    import csv as _csv

    rows: list[dict[str, Any]] = []
    with open(csv_path, newline="") as fh:
        for row in _csv.DictReader(fh):
            run_id = row.get("run_id", "")
            if "_N" not in run_id:
                raise ValueError(
                    f"run_id {run_id!r} carries no _N<size> suffix; "
                    "search CSVs must come from candidate_config()"
                )
            row["size"] = int(run_id.rsplit("_N", 1)[1])
            rows.append(row)
    return rows


def run_backbone_search(
    template: config_mod.RunConfig,
    *,
    out_dir: str | Path,
    n: int = N_BACKBONE,
    seeds: Sequence[int] = TUNING_SEEDS,
    sizes: Sequence[int] = DEV_SIZES,
    n_checkpoints: int = 20,
) -> tuple[selection.SelectionResult, SearchRecord]:
    """Run the full class-1 backbone search and select the winner.

    The whole search is a deterministic function of ``template.master_seed``: the candidate
    field comes from the ``hparam_search`` stream, and each run's streams come from the
    tuning ``cell_id`` and seed index. Re-running reproduces the winner exactly, which is
    what makes the emitted :class:`SearchRecord` auditable rather than merely descriptive.
    """
    points = sample_backbone_points(template.master_seed, n=n)
    # Index by the candidate's own draw position, never by dict equality: two draws can
    # collide on every categorical field and differ only in lr, and a `points.index(p)`
    # lookup would then silently score one candidate with the other's runs.
    indexed = [dict(p, **{_INDEX_KEY: i}) for i, p in enumerate(points)]
    results = [
        run_candidate(
            template,
            point,
            index=i,
            out_dir=out_dir,
            seeds=seeds,
            sizes=sizes,
            n_checkpoints=n_checkpoints,
        )
        for i, point in enumerate(points)
    ]
    candidates = selection.score_candidates(
        indexed,
        lambda p: results[p[_INDEX_KEY]]["scores"],
        sort_key_fn=_sort_key,
        label_fn=lambda p: f"c{p[_INDEX_KEY]:02d}",
    )
    best = selection.select_best(candidates)
    record = SearchRecord(
        kind="class1_backbone",
        master_seed=template.master_seed,
        n_candidates=n,
        seeds=tuple(int(s) for s in seeds),
        sizes=tuple(int(s) for s in sizes),
        objective="iqm_of_per_seed_discovery_auc",
        points=[dict(p) for p in points],
        per_candidate=[
            {
                "index": i,
                "point": dict(points[i]),
                "iqm": candidates[i].iqm,
                "scores": results[i]["scores"],
                "per_run": results[i]["per_run"],
            }
            for i in range(n)
        ],
        winner_index=int(best.winner.params[_INDEX_KEY]),
        tie_broken=best.tie_broken,
    )
    return best, record


def run_mini_search(
    template: config_mod.RunConfig,
    which: str,
    *,
    out_dir: str | Path,
    n: int = N_MINI,
    seeds: Sequence[int] = TUNING_SEEDS,
    sizes: Sequence[int] = DEV_SIZES,
    n_checkpoints: int = 20,
) -> tuple[selection.SelectionResult, SearchRecord]:
    """Run one class-3 mini-search and select the winner.

    Structurally identical to :func:`run_backbone_search` — same objective, same pooling,
    same tie-break machinery, same auditable record — differing only in the candidate
    field's dimensionality and the materializer. The tie-break key is the single varied
    parameter, which is freeze item 3's "lower parameter value" read literally.
    """
    spec = MINI_SEARCHES[which]
    param = spec["param"]
    points = sample_mini_points(template.master_seed, which, n=n)
    indexed = [dict(p, **{_INDEX_KEY: i}) for i, p in enumerate(points)]
    config_fn = functools.partial(mini_candidate_config, which=which)
    results = [
        run_candidate(
            template,
            point,
            index=i,
            out_dir=out_dir,
            seeds=seeds,
            sizes=sizes,
            n_checkpoints=n_checkpoints,
            config_fn=config_fn,
        )
        for i, point in enumerate(points)
    ]
    candidates = selection.score_candidates(
        indexed,
        lambda p: results[p[_INDEX_KEY]]["scores"],
        sort_key_fn=lambda p: (p[param],),
        label_fn=lambda p: f"c{p[_INDEX_KEY]:02d}",
    )
    best = selection.select_best(candidates)
    record = SearchRecord(
        kind=f"class3_{which}",
        master_seed=template.master_seed,
        n_candidates=n,
        seeds=tuple(int(s) for s in seeds),
        sizes=tuple(int(s) for s in sizes),
        objective="iqm_of_per_seed_discovery_auc",
        points=[dict(p) for p in points],
        per_candidate=[
            {
                "index": i,
                "point": dict(points[i]),
                "iqm": candidates[i].iqm,
                "scores": results[i]["scores"],
                "per_run": results[i]["per_run"],
            }
            for i in range(n)
        ],
        winner_index=int(best.winner.params[_INDEX_KEY]),
        tie_broken=best.tie_broken,
    )
    return best, record


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: run the class-1 backbone search and write the auditable record."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--kind",
        default="backbone",
        choices=["backbone", *sorted(MINI_SEARCHES)],
        help="which frozen search to run: the class-1 backbone or a class-3 mini-search",
    )
    ap.add_argument(
        "--template",
        default=None,
        help=(
            "config the search perturbs; defaults per --kind to the ε-greedy DDQN backbone "
            "template or the mini-search's own input cell"
        ),
    )
    ap.add_argument("--out", default=None, help="run-log directory (default logs/search/<kind>)")
    ap.add_argument(
        "--record",
        default=None,
        help="auditable search record (default <out>/search_record.json)",
    )
    ap.add_argument("--candidates", type=int, default=None)
    ap.add_argument("--checkpoints", type=int, default=20)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="draw and print the candidate field without executing any run",
    )
    args = ap.parse_args(argv)

    is_backbone = args.kind == "backbone"
    default_template = (
        "configs/example_ddqn_deepsea_dev.yaml"
        if is_backbone
        else MINI_SEARCHES[args.kind]["template"]
    )
    template_path = args.template or default_template
    out_dir = args.out or f"logs/search/{args.kind}"
    record_path = args.record or f"{out_dir}/search_record.json"
    n = args.candidates if args.candidates is not None else (N_BACKBONE if is_backbone else N_MINI)

    template = config_mod.load_config(template_path)
    if args.dry_run:
        if is_backbone:
            points = sample_backbone_points(template.master_seed, n=n)
            keys: tuple[str, ...] = DRAW_ORDER
        else:
            points = sample_mini_points(template.master_seed, args.kind, n=n)
            keys = (MINI_SEARCHES[args.kind]["param"],)
        for i, point in enumerate(points):
            print(f"c{i:02d} " + " ".join(f"{k}={point[k]}" for k in keys))
        return 0

    if is_backbone:
        best, record = run_backbone_search(
            template,
            out_dir=out_dir,
            n=n,
            n_checkpoints=args.checkpoints,
        )
    else:
        best, record = run_mini_search(
            template,
            args.kind,
            out_dir=out_dir,
            n=n,
            n_checkpoints=args.checkpoints,
        )
    path = record.write(record_path)
    print(
        f"winner {best.winner.label}: IQM={best.winning_iqm:.4f} "
        f"tie_broken={best.tie_broken} → {path}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
