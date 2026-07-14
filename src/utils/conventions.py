"""A1 conventions — determinism, CSV logging, resolved-config serialization.

Appendix C of the execution plan (v4.1), grounded in requirements spec v6.1.

Three jobs, all frozen here so every run obeys the same contract:

1. **Determinism** (gate C1): one `seed_everything(seed)` call seeds Python, NumPy and
   torch, and puts torch in deterministic mode. A run is reproducible from ``(config, seed)``.
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
    "seed",
    "config_sha256",  # ties every row back to the committed resolved_config.json (C13)
    "step",         # environment interaction step (the budget axis)
    "metric",       # metric name, e.g. discovery_prob, episode_return, sigma_mean
    "value",
)


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
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}, got {self.role!r}")
        if self.part not in ("A", "B"):
            raise ValueError(f"part must be 'A' or 'B', got {self.part!r}")
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

    def log(self, *, step: int, metric: str, value: float, **extra: Any) -> None:
        """Append one metric row. ``extra`` keys become extra columns (only in the header
        of a fresh file; for an existing file they must already be present)."""
        if self._writer is None:
            raise RuntimeError("CSVLogger used before open()")
        row = {
            "run_id": self.ctx.run_id,
            "role": self.ctx.role,
            "part": self.ctx.part,
            "method": self.ctx.method,
            "env": self.ctx.env,
            "seed": self.ctx.seed,
            "config_sha256": self.ctx.config_sha256,
            "step": step,
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
