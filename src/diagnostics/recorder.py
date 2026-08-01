"""Per-run diagnostics recorder — the bridge from the trainer's checkpoint loop to the
:mod:`src.diagnostics.substrate` writer.

:mod:`src.diagnostics.samplers` supplies the ``ValueSampler`` adapters; the substrate
supplies the record type and the ``.npz`` writer. Neither knows anything about a training
run. This module is the missing third piece: it owns the per-run state a checkpoint record
needs but a sampler cannot see — the probe set, the visitation histogram accumulated since
the run began, and the spec that pins every record to one probe set and one sampler kind.

Scope: **DeepSea (Part A) only.** Every diagnostic in the §3.3 battery references ``Q*``,
which MinAtar does not have, so the step-budgeted Part-B lane never builds a recorder.

Two design commitments, both load-bearing and both pinned by tests.

**1. The metrics CSV is never touched.** It would be natural to log the scalar disagreement
summary as extra CSV rows. This module deliberately does not: gate C1 requires that
re-running a ``(config, seed)`` pair reproduces its CSV byte-for-byte, and the diagnostics
switch is a *command-line* flag, not part of the config identity. Were the summary logged to
the CSV, two runs of the same frozen config would produce different CSVs depending on how
they were invoked, and C1 could no longer distinguish "the code changed" from "someone passed
a different flag". Everything this module produces goes to the ``.npz`` and its JSON sidecar,
so the CSV is byte-identical whether or not diagnostics ran — an unconditional guarantee,
which is strictly stronger than the conditional one and is what
``test_diagnostics_do_not_change_the_csv`` asserts.

**2. Measurement never touches an operational RNG stream.** The recorder holds no generator
of its own and calls only ``sampler.value_samples``, whose adapters draw from the
measurement-side stream (``noisynet_diag``) or from nothing at all (ensembles are exhaustive
over their heads). A run with diagnostics enabled therefore takes the same actions, in the
same order, as one without — which is what makes commitment 1 achievable rather than merely
aspirational.

Visitation counts (``v(s)``, needed by diagnostic #5's visitation-weighted secondary) are
accumulated by the recorder rather than by the sampler, because they are a property of the
trajectory, not of the value function: ``σ(s, a*)`` regressed on ``log(1 + v(s))`` needs the
``v`` that held *at that checkpoint*, and it cannot be reconstructed after the fact. The
trainer calls :meth:`RunRecorder.observe_state` once per environment step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.diagnostics.samplers import (
    deep_sea_probe_states,
    disagreement_summary,
    make_value_sampler,
)
from src.diagnostics.substrate import SubstrateSpec, ValueSampleWriter, record_checkpoint


def probe_set_id(size: int) -> str:
    """Stable identifier for the exhaustive DeepSea probe set at ``size``.

    Carried in every record's spec so that samples taken on different probe sets can never
    be pooled by the offline reducer. It encodes the construction rule and the size, not a
    hash of the states: the rule is frozen (freeze item 7, exhaustive reachable set), so two
    probe sets agreeing on ``size`` are identical by construction.
    """
    return f"deep_sea_exhaustive_reachable_n{int(size)}"


class RunRecorder:
    """Accumulates one seed's value-sample records and writes them at the end of the run.

    Construct via :func:`make_recorder`, which returns ``None`` for method/env combinations
    that have no sample distribution to record. A ``None`` recorder is the normal case for
    the ε-greedy DDQN reference and for every MinAtar run, so the trainer's call sites are
    written to tolerate it.
    """

    def __init__(self, sampler, *, size: int, n_actions: int, run_id: str, seed: int) -> None:
        self.size = int(size)
        self.run_id = run_id
        self.seed = int(seed)
        self._sampler = sampler

        obs, idx = deep_sea_probe_states(self.size)  # flat encoding: matches the agent input
        self._probe_obs = obs
        self._probe_idx = idx
        #: (row, col) -> position in the probe axis, for the visitation histogram.
        self._probe_pos = {(int(r), int(c)): i for i, (r, c) in enumerate(idx)}
        self._visitation = np.zeros(len(idx), dtype=np.int64)

        self.spec: SubstrateSpec = sampler.spec(
            n_probe_states=len(idx),
            n_actions=int(n_actions),
            probe_set_id=probe_set_id(self.size),
        )
        self._writer = ValueSampleWriter(self.spec)
        self._summaries: list[dict[str, Any]] = []

    def observe_state(self, obs: np.ndarray) -> None:
        """Count one visit, from the flat one-hot observation the agent was handed.

        Unreachable cells cannot appear in a DeepSea trajectory, so a state outside the probe
        set is a wiring bug rather than a case to tolerate silently — but raising here would
        abort a long run over a diagnostic, so it is counted as a miss and surfaced in the
        sidecar instead.
        """
        flat = np.asarray(obs).reshape(-1)
        hit = int(np.argmax(flat))
        rc = divmod(hit, self.size)
        pos = self._probe_pos.get(rc)
        if pos is None:
            self._off_probe = getattr(self, "_off_probe", 0) + 1
            return
        self._visitation[pos] += 1

    def record(self, step: int) -> dict[str, float]:
        """Take one checkpoint's samples. Returns the scalar disagreement summary.

        The summary is returned rather than logged so the caller decides where it goes; the
        recorder itself only ever writes the sidecar (see this module's commitment 1).
        """
        rec = record_checkpoint(
            self._sampler,
            self._probe_obs,
            self.spec,
            step=int(step),
            visitation=self._visitation.copy(),  # snapshot: v(s) as of THIS checkpoint
        )
        self._writer.append(rec)
        summary = disagreement_summary(rec.samples)
        self._summaries.append({"step": int(step), **summary})
        return summary

    def __len__(self) -> int:
        return len(self._writer)

    def write(self, out_dir: str | Path) -> Path | None:
        """Write ``<run_id>.seed<N>.value_samples.npz`` + sidecar. ``None`` if no records.

        One file per (run, seed): the substrate writer's identity is per-run, but a run here
        spans several seeds sharing one CSV, and pooling seeds into one tensor would lose the
        seed axis the reducer needs.
        """
        if len(self._writer) == 0:
            return None
        run_key = f"{self.run_id}.seed{self.seed}"
        npz = self._writer.write(out_dir, run_key)
        extra = {
            "run_id": self.run_id,
            "seed": self.seed,
            "deep_sea_size": self.size,
            "disagreement_by_checkpoint": self._summaries,
            "visitation_total": int(self._visitation.sum()),
            "probe_states_visited": int((self._visitation > 0).sum()),
            "off_probe_observations": int(getattr(self, "_off_probe", 0)),
        }
        Path(out_dir, f"{run_key}.diagnostics.json").write_text(
            json.dumps(extra, indent=2, sort_keys=True) + "\n"
        )
        return npz


def make_recorder(agent, cfg, seed_index: int) -> RunRecorder | None:
    """Build a recorder for this run, or ``None`` when there is nothing to record.

    ``None`` is returned for three distinct reasons, all legitimate:

    * the env is not DeepSea — the battery needs ``Q*``, which only DeepSea has;
    * the method is the ε-greedy DDQN reference — a point estimator has no sample
      distribution, so ``make_value_sampler`` declines it;
    * the config carries no ``deep_sea_size`` — nothing to build a probe set from.
    """
    if cfg.env != "deep_sea":
        return None
    sampler = make_value_sampler(agent, cfg.method)
    if sampler is None:
        return None
    size = int(cfg.data["env_budget"]["deep_sea_size"])
    return RunRecorder(
        sampler,
        size=size,
        n_actions=2,  # DeepSea is binary: left/right
        run_id=cfg.run_id,
        seed=seed_index,
    )
