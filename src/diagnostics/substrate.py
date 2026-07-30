"""Value-sample recording substrate for the uncertainty-quality battery (§8 item 5).

The implementation order puts "disagreement logging" at item 5, *before* Q\\* (item 6) and
before the battery itself (item 9). Since six of the nine frozen diagnostics
(§3.3 #1–#5, #7) are defined against Q\\*, item 5 cannot mean computing them. What it can
mean — and what this module is — is the **recording substrate**: at each checkpoint,
persist the raw per-sample value tensor from which all nine frozen statistics are later
computed offline.

Why raw samples rather than summary statistics
----------------------------------------------
Storing only the marginal ``(Q̄, σ)`` would be strictly insufficient. Three of the frozen
diagnostics cannot be recovered from marginals:

* **#2 action-gap alignment** needs ``std_m[Q_m(s,a₁) − Q_m(s,a₂)]`` — the std of a
  *difference*, which depends on the correlation between the two actions' samples and is
  not a function of their separate stds.
* **#3 incorrect-argmax flagging** needs each sample's ``argmax_a Q_m(s,·)`` to form the
  modal fraction — a per-sample quantity that no marginal preserves.
* **#7 empirical containment** needs empirical quantiles over ``m``
  (``numpy.quantile(..., method="linear")``), i.e. the whole sample distribution.

So ``[S, M, A]`` is the minimum-sufficient record, and it is what we store.

Why the statistics are computed offline
---------------------------------------
§3.6 makes an implementation bug in the confirmatory path void the confirmatory block. A
statistic computed inside the run path puts nine numerical routines inside that blast
radius, and a bug in any one of them would be discovered only after the runs are spent.
Persisting samples and reducing them in ``analysis/`` bounds the in-run risk to this
module — which has exactly one job, is dtype- and shape-checked on write, and is
re-runnable against the stored tensor as many times as needed.

What is frozen vs. what is not
------------------------------
The *statistics* are frozen (§3.3, all nine, including tie-breaking and the quantile
method). The *probe set construction* is *not* fully specified: freeze item 7 covers
"probe-set construction + weighting", but only the weighting is written down — ``|S|`` and
the sampling rule appear in neither document. This module therefore takes the probe set as
an **argument** and never constructs one, so it is correct under any construction rule the
protocol eventually pins. ``|S|`` is a caller-side parameter (see
:class:`SubstrateSpec`), which is why nothing here needs to change when item 7 lands.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "ValueSampler",
    "SubstrateSpec",
    "ValueSampleRecord",
    "record_checkpoint",
    "ValueSampleWriter",
]


@runtime_checkable
class ValueSampler(Protocol):
    """Contract the substrate needs from an agent adapter.

    Deliberately narrower than the agent interface: the substrate must not be able to
    perturb training. Implementations draw from a *measurement-side* generator (the
    ``noisynet_diag`` stream for NoisyNet's M = 30 draws; the K heads themselves for the
    ensemble methods, where sampling is exhaustive rather than random).
    """

    def value_samples(self, probe_states: np.ndarray) -> np.ndarray:
        """Return the per-sample action-values on the probe set.

        Parameters
        ----------
        probe_states:
            Probe set ``S``, shape ``[S, *obs_shape]``.

        Returns
        -------
        np.ndarray
            Shape ``[S, M, A]``: for each probe state, ``M`` value samples over ``A``
            actions. For the ensemble methods ``M = K`` (the heads, in fixed order). For
            NoisyNet ``M = 30`` i.i.d. draws taken at measurement time only.
        """
        ...


@dataclass(frozen=True)
class SubstrateSpec:
    """Shape and provenance contract for one run's value-sample record.

    ``n_probe_states`` (``|S|``) is a *parameter*, not a constant: freeze item 7's probe-set
    construction rule is not yet written down, so the substrate is built to be correct
    under whatever it turns out to be. Recording it here means every stored tensor carries
    the ``|S|`` it was produced under, so records made before and after item 7 lands are
    never silently pooled.

    Attributes
    ----------
    n_probe_states:
        ``|S|`` — number of probe states.
    n_samples:
        ``M`` — value samples per state. ``K`` for ensembles; 30 for NoisyNet.
    n_actions:
        ``A`` — action-space size.
    sampler_kind:
        Which sampling semantics produced the record (``"ensemble_heads"`` or
        ``"noisynet_draws"``). Stored so the offline reducer never has to infer it.
    probe_set_id:
        Identifier of the probe set the samples were taken on. Two records are only
        comparable if this matches.
    """

    n_probe_states: int
    n_samples: int
    n_actions: int
    sampler_kind: str
    probe_set_id: str

    VALID_SAMPLER_KINDS = ("ensemble_heads", "noisynet_draws")

    def __post_init__(self) -> None:
        for name in ("n_probe_states", "n_samples", "n_actions"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.sampler_kind not in self.VALID_SAMPLER_KINDS:
            raise ValueError(
                f"sampler_kind must be one of {self.VALID_SAMPLER_KINDS}, got {self.sampler_kind!r}"
            )

    @property
    def shape(self) -> tuple[int, int, int]:
        """The expected ``[S, M, A]`` tensor shape."""
        return (self.n_probe_states, self.n_samples, self.n_actions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_probe_states": self.n_probe_states,
            "n_samples": self.n_samples,
            "n_actions": self.n_actions,
            "sampler_kind": self.sampler_kind,
            "probe_set_id": self.probe_set_id,
        }


@dataclass(frozen=True)
class ValueSampleRecord:
    """One checkpoint's value samples, plus the visitation counts diagnostic #5 needs.

    Attributes
    ----------
    step:
        Environment step at which the checkpoint was taken.
    samples:
        ``[S, M, A]`` float32 array of value samples.
    visitation:
        ``[S]`` array of state-visitation counts ``v(s)``. Diagnostic #5 regresses
        ``log σ(s, a*(s))`` on ``log(1 + v(s))``, so ``v`` must be captured *at the same
        checkpoint* as the samples — it cannot be reconstructed afterwards. Optional only
        because the MinAtar analogue (#8) does not use it.
    spec:
        The shape/provenance contract these samples satisfy.
    """

    step: int
    samples: np.ndarray
    spec: SubstrateSpec
    visitation: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.samples.shape != self.spec.shape:
            raise ValueError(
                f"samples shape {self.samples.shape} does not match spec {self.spec.shape}"
            )
        if self.samples.dtype != np.float32:
            raise ValueError(f"samples must be float32, got {self.samples.dtype}")
        if not np.all(np.isfinite(self.samples)):
            raise ValueError(
                "value samples contain non-finite entries; a diverged network must be "
                "recorded as such by the caller and handled under the §3.3 "
                "undefined-value policy, not written into the substrate"
            )
        if self.visitation is not None and self.visitation.shape != (self.spec.n_probe_states,):
            raise ValueError(
                f"visitation shape {self.visitation.shape} does not match "
                f"[S] = ({self.spec.n_probe_states},)"
            )


def record_checkpoint(
    sampler: ValueSampler,
    probe_states: np.ndarray,
    spec: SubstrateSpec,
    *,
    step: int,
    visitation: np.ndarray | None = None,
) -> ValueSampleRecord:
    """Draw and validate one checkpoint's value samples.

    Pure with respect to training: it calls only :meth:`ValueSampler.value_samples`, which
    by contract draws from a measurement-side generator. The returned record is validated
    against ``spec`` on construction, so a shape or dtype regression surfaces at the
    checkpoint that produced it rather than months later in the reducer.

    Raises
    ------
    ValueError
        If the sampler's output does not match ``spec``, or contains non-finite values.
    """
    probes = np.asarray(probe_states)
    if probes.shape[0] != spec.n_probe_states:
        raise ValueError(
            f"probe set has {probes.shape[0]} states but spec declares "
            f"|S| = {spec.n_probe_states}"
        )
    raw = np.asarray(sampler.value_samples(probes), dtype=np.float32)
    return ValueSampleRecord(step=int(step), samples=raw, spec=spec, visitation=visitation)


class ValueSampleWriter:
    """Accumulates per-checkpoint records and writes one ``.npz`` per run.

    One file per run (not per checkpoint) keeps the artifact count proportional to runs
    rather than to runs × checkpoints, which matters at the ~2,500–3,000-run scale the
    budget contemplates. A sidecar JSON carries the spec so the tensor can be interpreted
    without loading it.

    Storage cost is ``S × M × A × 4`` bytes per checkpoint. At ``|S| = 512``, ``M = 20``,
    ``A = 6`` that is 246 KB per checkpoint — negligible against the replay buffer, which
    is why persisting raw samples is affordable in the first place.
    """

    def __init__(self, spec: SubstrateSpec) -> None:
        self.spec = spec
        self._records: list[ValueSampleRecord] = []

    def append(self, record: ValueSampleRecord) -> None:
        """Add a checkpoint record, enforcing spec identity across the run."""
        if record.spec != self.spec:
            raise ValueError(
                "record spec differs from the writer's spec; samples from different "
                "probe sets or sampler kinds must not be pooled in one run file"
            )
        if self._records and record.step <= self._records[-1].step:
            raise ValueError(
                f"checkpoint steps must be strictly increasing; got {record.step} "
                f"after {self._records[-1].step}"
            )
        self._records.append(record)

    def __len__(self) -> int:
        return len(self._records)

    def write(self, out_dir: str | Path, run_id: str) -> Path:
        """Write ``<run_id>.value_samples.npz`` plus its ``.json`` sidecar.

        Returns the path to the ``.npz``. Raises if no records were accumulated — an empty
        substrate file is far more likely to be a wiring bug than an intended outcome.
        """
        if not self._records:
            raise ValueError("no checkpoint records to write")
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        steps = np.array([r.step for r in self._records], dtype=np.int64)
        samples = np.stack([r.samples for r in self._records])  # [T, S, M, A]
        payload: dict[str, np.ndarray] = {"steps": steps, "samples": samples}
        if all(r.visitation is not None for r in self._records):
            payload["visitation"] = np.stack([r.visitation for r in self._records])

        npz_path = out / f"{run_id}.value_samples.npz"
        np.savez_compressed(npz_path, **payload)

        sidecar = {
            "run_id": run_id,
            "spec": self.spec.as_dict(),
            "n_checkpoints": len(self._records),
            "steps": steps.tolist(),
            "tensor_layout": "[T, S, M, A] float32; T = checkpoints",
            "has_visitation": "visitation" in payload,
        }
        (out / f"{run_id}.value_samples.json").write_text(json.dumps(sidecar, indent=2) + "\n")
        return npz_path
