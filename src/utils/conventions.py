"""A1 conventions — determinism, CSV logging, resolved-config serialization.

Appendix C of the execution plan (v4.1), grounded in requirements spec v6.1.

Three jobs, all frozen here so every run obeys the same contract:

1. **Determinism** (gate C1): one `seed_everything(seed)` call seeds Python, NumPy and
   torch, and puts torch in deterministic mode. A run is reproducible from ``(config, seed)``.
   Each *random stream* is then derived per cell as ``hash(master_seed, cell_id,
   stream_name, seed_index)`` (``derive_seed`` and friends) — no stream is reused across
   cells, so the cross-cell bootstrap is independent by construction (spec v6.3 / plan v4.3).
2. **CSV logging** (gate C2): every scalar that can appear in a figure is appended to a
   CSV with a fixed schema. Nothing a figure depends on lives only in memory or a
   dashboard. Every row carries a ``role`` field in {confirmatory, development,
   exploratory} — QR-DQN rows are ``exploratory`` by construction (spec v6.1 Appendix C).
3. **Resolved-config serialization** (gate C13): every run's *fully resolved* config is
   serialized to JSON and committed, so the configuration-identity audit can prove that
   two cells of a contrast differ only in the varied factor and its factor-specific params.

Deliberately dependency-light: stdlib + numpy + torch only. No project-specific science
lives here — this is plumbing shared by every method.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

try:  # torch is optional at import time so pure-analysis code can import this module
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch always present in the pinned env
    _HAS_TORCH = False


# --------------------------------------------------------------------------- #
# 1. Determinism (C1)
# --------------------------------------------------------------------------- #

Role = Literal["confirmatory", "development", "exploratory"]
VALID_ROLES = ("confirmatory", "development", "exploratory")


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> int:
    """Seed every RNG that can affect a run and return the seed.

    Seeds ``random``, ``numpy``, and (if present) ``torch`` CPU + CUDA generators, and
    enables torch deterministic algorithms. Call exactly once at the top of a run, before
    building environments or networks. The returned value is logged so the seed that
    actually took effect is on record.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            # cuBLAS reproducibility for any accidental GPU use; harmless on CPU.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    return seed


def torch_generator(seed: int):
    """Return a seeded ``torch.Generator`` for dataloaders / explicit sampling."""
    if not _HAS_TORCH:
        raise RuntimeError("torch is not available")
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# --------------------------------------------------------------------------- #
# 1b. Cell-specific RNG derivation (spec v6.3 / plan v4.3, freeze item)
# --------------------------------------------------------------------------- #
#
# Every random stream in a run is derived from a *stable, pinned* function of
#
#     (master_seed, cell_id, stream_name, seed_index)
#
# and *no stream is ever reused across cells*. The seed labels (0..N-1) are
# bookkeeping only — pairing across cells by "the same integer seed" carries no
# statistical meaning, which is why the independent stratified bootstrap is
# justified by construction. (This retires the v6.2 "same integer seed creates
# no pairing" wording, which was incorrect: the point is that streams are
# *derived*, so cells share no randomness even at equal ``seed_index``.)
#
# Pinned implementation (reviewer requirement, do NOT change without a version
# bump — every downstream stream shifts if you do):
#
#   payload  = canonical UTF-8 serialization "<master_seed>\x1f<cell_id>\x1f"
#              "<stream_name>\x1f<seed_index>"  (0x1F unit separator; the fields
#              never contain it, so the mapping is unambiguous)
#   digest   = BLAKE2b(payload, digest_size=16)          # stable across builds
#   ss       = numpy.random.SeedSequence(int(digest))    # high-quality spawn
#
# The Python builtin ``hash()`` is deliberately NOT used anywhere here: it is
# salted per process (PYTHONHASHSEED) and would break reproducibility. A pinned
# regression value guards the exact bytes (see tests).

# The canonical named streams. A run draws each of its random quantities from
# the stream named for its role; adding/removing/reordering is a versioned change
# and shifts nothing already derived (names, not positions, key the derivation).
STREAM_NAMES: tuple[str, ...] = (
    "init",            # network / ensemble-head parameter initialization
    "env_mapping",     # environment + action-direction mapping (DeepSea flip mask)
    "replay",          # replay-buffer sampling order
    "action_noise",    # exploration noise (ε-greedy tie-breaks, per-step noise)
    "bootstrap_mask",  # Bootstrapped-DQN per-transition head masks
    "eval_episodes",   # evaluation / rollout episode stochasticity
    "probe_set",       # frozen diagnostic probe-set generation
    "noisynet_diag",   # NoisyNet measurement-time diagnostic draws (M=30)
)

# 0x1F (ASCII unit separator) delimits fields in the canonical payload. cell_id /
# stream_name are ASCII tokens that never contain it, so serialization is injective.
_FIELD_SEP = "\x1f"
_DIGEST_SIZE = 16  # BLAKE2b digest bytes
# When an int seed is needed (torch), mask the digest to 63 bits: non-negative and
# safe for torch.Generator.manual_seed. numpy paths use the full SeedSequence.
_SEED_BITS = 63


def _canonical_payload(master_seed: int, cell_id: str, stream_name: str, seed_index: int) -> bytes:
    """Injective UTF-8 serialization of the four fields (pinned; see module notes)."""
    fields = (str(master_seed), cell_id, stream_name, str(seed_index))
    return _FIELD_SEP.join(fields).encode("utf-8")


def _validate_derive_args(master_seed: int, stream_name: str, seed_index: int) -> None:
    if not isinstance(master_seed, int) or isinstance(master_seed, bool):
        raise TypeError(f"master_seed must be an int, got {type(master_seed).__name__}")
    if not isinstance(seed_index, int) or isinstance(seed_index, bool):
        raise TypeError(f"seed_index must be an int, got {type(seed_index).__name__}")
    if stream_name not in STREAM_NAMES:
        raise ValueError(f"stream_name must be one of {STREAM_NAMES}, got {stream_name!r}")


def derive_seed_sequence(
    master_seed: int, cell_id: str, stream_name: str, seed_index: int
) -> np.random.SeedSequence:
    """The canonical ``numpy.random.SeedSequence`` for one (cell, stream, seed).

    Pinned: ``SeedSequence(int(BLAKE2b(canonical_payload, digest_size=16)))``.
    This is the preferred entry point for numpy RNGs (best-quality spawning).
    ``cell_id`` is the factorial arm (e.g. ``"episodic|off|K10"`` = the ``arm``
    alias); ``seed_index`` is the bookkeeping seed label.
    """
    _validate_derive_args(master_seed, stream_name, seed_index)
    digest = hashlib.blake2b(
        _canonical_payload(master_seed, cell_id, stream_name, seed_index),
        digest_size=_DIGEST_SIZE,
    ).digest()
    return np.random.SeedSequence(int.from_bytes(digest, "big"))


def derive_seed(master_seed: int, cell_id: str, stream_name: str, seed_index: int) -> int:
    """A 63-bit int seed for the named derived stream (for ``torch.manual_seed``).

    Deterministic and platform-stable (BLAKE2b over the canonical payload). Distinct
    ``(cell_id, stream_name, seed_index)`` triples yield independent streams; the same
    triple always reproduces the same seed. Prefer :func:`derive_seed_sequence` /
    :func:`derive_numpy_generator` for numpy; this int form exists for libraries
    (torch) that take a plain integer seed.
    """
    _validate_derive_args(master_seed, stream_name, seed_index)
    digest = hashlib.blake2b(
        _canonical_payload(master_seed, cell_id, stream_name, seed_index),
        digest_size=_DIGEST_SIZE,
    ).digest()
    return int.from_bytes(digest, "big") & ((1 << _SEED_BITS) - 1)


def derive_cell_seeds(
    master_seed: int, cell_id: str, seed_index: int, stream_names: tuple[str, ...] = STREAM_NAMES
) -> dict[str, int]:
    """Return ``{stream_name: derived_seed}`` for every stream of one (cell, seed)."""
    return {s: derive_seed(master_seed, cell_id, s, seed_index) for s in stream_names}


def derive_numpy_generator(
    master_seed: int, cell_id: str, stream_name: str, seed_index: int
) -> np.random.Generator:
    """A ``numpy`` Generator seeded from the derived SeedSequence (use this, not global state)."""
    ss = derive_seed_sequence(master_seed, cell_id, stream_name, seed_index)
    return np.random.default_rng(ss)


def derive_torch_generator(master_seed: int, cell_id: str, stream_name: str, seed_index: int):
    """A seeded ``torch.Generator`` for the named derived stream."""
    if not _HAS_TORCH:
        raise RuntimeError("torch is not available")
    g = torch.Generator()
    g.manual_seed(derive_seed(master_seed, cell_id, stream_name, seed_index))
    return g


# --------------------------------------------------------------------------- #
# 1c. Per-run DeepSea mapping identity (spec v6.3 / plan v4.3, reviewer Fix 4)
# --------------------------------------------------------------------------- #
#
# Each run derives its own DeepSea action-direction mapping from the ``env_mapping``
# stream, so the optimal-action labels and Q* are specific to THAT run's mapping.
# ``deepsea_mapping_hash`` fingerprints the mapping; store it with the run and make
# every Q*-referenced diagnostic (optimal action, action gap, optimal path) assert
# it is operating on the same mapping — otherwise a diagnostic could silently compare
# a learned Q against a Q* built for a different mapping.


def deepsea_mapping_hash(action_mapping: Any) -> str:
    """Stable SHA-256 fingerprint of a DeepSea action-direction mapping.

    Accepts a 1-D array / sequence of per-row action-to-direction flips (bool/int).
    The hash is committed with the run and re-checked by every Q*-referenced
    diagnostic so ground truth and learned values always share one mapping.
    """
    arr = np.asarray(action_mapping)
    payload = arr.shape, arr.dtype.str, arr.tobytes()
    blob = repr(payload).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def deepsea_action_mapping(
    master_seed: int, cell_id: str, seed_index: int, size: int
) -> np.ndarray:
    """The per-run DeepSea action-flip mapping (one bool per row) from ``env_mapping``.

    Bound to the run's derived ``env_mapping`` stream so the mapping — and therefore
    Q* — is reproducible from ``(master_seed, cell_id, seed_index, size)`` alone.
    """
    rng = derive_numpy_generator(master_seed, cell_id, "env_mapping", seed_index)
    return rng.integers(0, 2, size=size).astype(bool)


# --------------------------------------------------------------------------- #
# 2. Resolved config + identity hash (C13)
# --------------------------------------------------------------------------- #


def _jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


def config_hash(resolved_config: dict[str, Any]) -> str:
    """Stable SHA-256 over a resolved config (sorted keys). The C13 identity fingerprint."""
    blob = json.dumps(_jsonable(resolved_config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def serialize_resolved_config(resolved_config: dict[str, Any], out_dir: str | os.PathLike) -> Path:
    """Write the fully resolved config to ``<out_dir>/resolved_config.json`` (C13 input).

    Returns the path written. The file includes a ``_config_sha256`` identity fingerprint
    so the audit can group contrast cells and diff only the varied factor.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(_jsonable(resolved_config))
    payload["_config_sha256"] = config_hash(resolved_config)
    path = out / "resolved_config.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


# --------------------------------------------------------------------------- #
# 3. CSV metric logging (C2)
# --------------------------------------------------------------------------- #

# Fixed leading schema. Extra scalar metrics are appended as trailing columns but the
# header is frozen once the first row is written, so a run's CSV is self-describing.
BASE_FIELDS = (
    "run_id",       # unique id for this (config, seed) execution
    "role",         # confirmatory | development | exploratory  (spec v6.1 Appendix C)
    "part",         # A (DeepSea mechanism) | B (MinAtar performance)
    "method",       # ddqn_egreedy | noisynet | bdqn | rp_bdqn | qrdqn | ...
    "env",          # deep_sea | breakout | asterix | ...
    "size_class",   # development | confirmatory  (Part A tier; Part B pilot|held-out) (v6.3 App C)
    "seed",         # bookkeeping seed_index; streams are hash-derived (see derive_seed)
    "config_sha256",  # ties every row back to the committed resolved_config.json (C13)
    "step",         # environment interaction step (the budget axis)
    "checkpoint",   # checkpoint index at which this metric was measured (v6.3 App C)
    "is_t0",        # bool: the t0 checkpoint (first success / earliest measurement) (v6.3 App C)
    "axis",         # online | frozen_policy  (which evaluation axis) (v6.3 App C, C12)
    "metric",       # metric name, e.g. discovery_prob, episode_return, sigma_mean
    "value",
)

VALID_SIZE_CLASSES = ("development", "confirmatory")
VALID_AXES = ("online", "frozen_policy")


@dataclass
class RunContext:
    """Immutable per-run identity stamped onto every logged row."""

    run_id: str
    role: Role
    part: Literal["A", "B"]
    method: str
    env: str
    seed: int
    config_sha256: str
    size_class: Literal["development", "confirmatory"] = "development"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}, got {self.role!r}")
        if self.part not in ("A", "B"):
            raise ValueError(f"part must be 'A' or 'B', got {self.part!r}")
        if self.size_class not in VALID_SIZE_CLASSES:
            raise ValueError(
                f"size_class must be one of {VALID_SIZE_CLASSES}, got {self.size_class!r}"
            )
        # A confirmatory size_class only makes sense on a confirmatory run.
        if self.size_class == "confirmatory" and self.role != "confirmatory":
            raise ValueError(
                "size_class='confirmatory' requires role='confirmatory' "
                f"(got role={self.role!r})"
            )
        if self.method == "qrdqn" and self.role != "exploratory":
            # Spec v6.1: QR-DQN is exploratory by construction; never in confirmatory aggregates.
            raise ValueError("QR-DQN rows must carry role='exploratory' (spec v6.1 Appendix C)")


class CSVLogger:
    """Append-only CSV metric logger with a frozen header (gate C2).

    Usage::

        ctx = RunContext(run_id="...", role="development", part="A",
                         method="bdqn", env="deep_sea", seed=0, config_sha256=h)
        with CSVLogger("logs/run_XXX.csv", ctx) as log:
            log.log(step=1000, metric="discovery_prob", value=0.0)

    Only scalar (config, seed)-reproducible metrics belong here. Every figure is later
    rebuilt from these CSVs alone (``make figures``).
    """

    def __init__(self, path: str | os.PathLike, ctx: RunContext):
        self.path = Path(path)
        self.ctx = ctx
        self._fh = None
        self._writer: csv.DictWriter | None = None
        self._fieldnames: list[str] = list(BASE_FIELDS)

    def __enter__(self) -> CSVLogger:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists() or self.path.stat().st_size == 0
        self._fh = self.path.open("a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self._fieldnames)
        if new:
            self._writer.writeheader()
            self._fh.flush()

    def log(
        self,
        *,
        step: int,
        metric: str,
        value: float,
        checkpoint: int | None = None,
        is_t0: bool = False,
        axis: Literal["online", "frozen_policy"] = "online",
        **extra: Any,
    ) -> None:
        """Append one metric row. ``extra`` keys become extra columns (only in the header
        of a fresh file; for an existing file they must already be present).

        ``checkpoint`` is the checkpoint index the metric was measured at; ``is_t0`` marks
        the t0 checkpoint (spec §7 t0 rule); ``axis`` selects the evaluation axis
        (``online`` vs ``frozen_policy``, gate C12)."""
        if self._writer is None:
            raise RuntimeError("CSVLogger used before open()")
        if axis not in VALID_AXES:
            raise ValueError(f"axis must be one of {VALID_AXES}, got {axis!r}")
        row = {
            "run_id": self.ctx.run_id,
            "role": self.ctx.role,
            "part": self.ctx.part,
            "method": self.ctx.method,
            "env": self.ctx.env,
            "size_class": self.ctx.size_class,
            "seed": self.ctx.seed,
            "config_sha256": self.ctx.config_sha256,
            "step": step,
            "checkpoint": checkpoint if checkpoint is not None else "",
            "is_t0": bool(is_t0),
            "axis": axis,
            "metric": metric,
            "value": value,
        }
        row.update({k: _jsonable(v) for k, v in extra.items()})
        self._writer.writerow(row)
        self._fh.flush()  # crash-safe: a killed run keeps every row it already emitted

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None
