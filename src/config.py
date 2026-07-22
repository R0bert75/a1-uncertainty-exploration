"""Run-config loader — the glue that makes one run reproducible from a frozen YAML.

A config file describes exactly one ``(method, env, budget)`` cell of the study (one file
per cell; seeds are always explicit and committed). This module turns that file into:

* a **validated** :class:`RunConfig` (structural vocabularies + the same guards
  ``conventions.RunContext`` enforces, checked at load time so a bad config fails before
  any compute is spent);
* a stable **C13 identity** — ``config_sha256`` over the fully resolved config, written to
  ``resolved_config.json`` so every logged row ties back to the exact config that produced
  it (gate C13);
* a per-seed :class:`~src.utils.conventions.RunContext` for the CSV logger;
* **factories** (:func:`build_env`, :func:`build_agent`) that instantiate the env and agent
  from the resolved config — closing the loop "a whole run executes from a frozen config"
  for the components that exist today (DeepSea + ε-greedy Double DQN). Methods/envs not yet
  implemented raise a clear :class:`NotImplementedError`, never a silent wrong default.

**Freeze discipline, enforced in code rather than baked as values.** This loader hard-codes
no frozen numeric protocol value. Structural *vocabularies* (the legal role/part/method/env
/use-rule/prior strings) are validated because they are enumerations, not tuned quantities.
Numeric quantities (K, learning rate, grid size, episode budget, ε schedule) are read from
the config. Development configs may omit backbone/schedule numbers — the missing ones fall
back to the clearly-labelled development placeholders that live in ``DDQNConfig`` (their one
sanctioned home). **Confirmatory** configs may not: :func:`load_config` refuses a
``role: confirmatory`` config that leaves any run-defining number to a placeholder, so a
confirmatory run can never silently inherit a development default. The study-wide
``master_seed`` is likewise a required config field (its frozen value is set at the Session-1
freeze), not a source constant.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.utils import conventions

# Structural vocabularies (enumerations, not tuned values) — safe to validate.
VALID_ROLES = conventions.VALID_ROLES                       # confirmatory|development|exploratory
VALID_PARTS = ("A", "B")
VALID_METHODS = ("ddqn_egreedy", "noisynet", "bdqn", "rp_bdqn", "qrdqn")
VALID_USE_RULES = ("episodic", "per_step", "ensemble_mean")
VALID_PRIORS = ("on", "off")

# Envs: DeepSea (Part A) + the five MinAtar games (Part B). Recognized for validation even
# though only DeepSea is buildable today.
VALID_ENVS = ("deep_sea", "breakout", "asterix", "seaquest", "freeway", "space_invaders")

# What the factories can actually instantiate right now (the rest raise NotImplementedError).
IMPLEMENTED_METHODS = ("ddqn_egreedy",)
IMPLEMENTED_ENVS = ("deep_sea",)

# Ensemble methods carry a per-head bootstrap (Class-2 params apply); the baseline does not.
ENSEMBLE_METHODS = ("bdqn", "rp_bdqn")


class ConfigError(ValueError):
    """Raised when a run config is structurally invalid or violates the freeze discipline."""


# --------------------------------------------------------------------------- #
# RunConfig
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RunConfig:
    """A validated, fully-resolved run config.

    ``data`` is the resolved config dict (the YAML as written, plus derived identity fields
    ``cell_id`` and ``size_class`` filled in). It is what gets hashed and serialized, so it
    is the single source of truth for the run's identity.
    """

    data: dict[str, Any]
    source_path: str | None = None

    # --- typed identity accessors ---------------------------------------- #
    @property
    def run_id(self) -> str:
        return self.data["run_id"]

    @property
    def role(self) -> str:
        return self.data["role"]

    @property
    def part(self) -> str:
        return self.data["part"]

    @property
    def method(self) -> str:
        return self.data["method"]

    @property
    def env(self) -> str:
        return self.data["env"]

    @property
    def master_seed(self) -> int:
        return int(self.data["master_seed"])

    @property
    def seeds(self) -> list[int]:
        return [int(s) for s in self.data["seeds"]]

    @property
    def size_class(self) -> str:
        return self.data["size_class"]

    @property
    def cell_id(self) -> str:
        """Canonical RNG-derivation cell id (== the config's ``arm``)."""
        return self.data["cell_id"]

    @property
    def config_sha256(self) -> str:
        """C13 identity fingerprint over the fully resolved config."""
        return conventions.config_hash(self.data)

    # --- run context + serialization ------------------------------------- #
    def run_context(self, seed_index: int) -> conventions.RunContext:
        """A :class:`RunContext` for one seed. ``config_sha256`` is shared across seeds of
        this cell; only ``seed`` differs. Re-validates via ``RunContext.__post_init__``."""
        if seed_index not in self.seeds:
            raise ConfigError(
                f"seed_index {seed_index} not in this config's committed seeds {self.seeds}"
            )
        return conventions.RunContext(
            run_id=self.run_id,
            role=self.role,
            part=self.part,
            method=self.method,
            env=self.env,
            seed=seed_index,
            config_sha256=self.config_sha256,
            size_class=self.size_class,
            extra={"cell_id": self.cell_id, "master_seed": self.master_seed},
        )

    def write_resolved(self, out_dir: str | Path) -> Path:
        """Serialize the resolved config to ``<out_dir>/resolved_config.json`` (C13 input)."""
        return conventions.serialize_resolved_config(self.data, out_dir)


# --------------------------------------------------------------------------- #
# Loading + validation
# --------------------------------------------------------------------------- #
def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def _canonical_cell_id(use_rule: str, prior: str, k: int) -> str:
    """The frozen cell-id convention: ``<use_rule>|<prior>|K<K>`` (RNG derivation key)."""
    return f"{use_rule}|{prior}|K{k}"


def _normalize_prior(data: dict[str, Any]) -> None:
    """Coerce ``prior`` back to the ``on``/``off`` strings.

    YAML 1.1 (PyYAML ``safe_load``) parses the bare tokens ``on``/``off`` as booleans, so a
    file that reads ``prior: off`` yields Python ``False``. We canonicalize to the string
    form in-place so the cell_id, the committed hash, and validation all see ``on``/``off``.
    """
    if isinstance(data.get("prior"), bool):
        data["prior"] = "on" if data["prior"] else "off"


def _validate_part_a_factorial(data: dict[str, Any]) -> str:
    """Validate the Part A switchboard fields and return the canonical cell_id."""
    use_rule = data.get("use_rule")
    prior = data.get("prior")
    k = data.get("K")
    _require(
        use_rule in VALID_USE_RULES,
        f"use_rule must be one of {VALID_USE_RULES}, got {use_rule!r}",
    )
    _require(prior in VALID_PRIORS, f"prior must be one of {VALID_PRIORS}, got {prior!r}")
    _require(
        isinstance(k, int) and not isinstance(k, bool) and k >= 1,
        f"K must be a positive int, got {k!r}",
    )
    canonical = _canonical_cell_id(use_rule, prior, k)
    arm = data.get("arm")
    _require(
        arm == canonical,
        f"arm {arm!r} does not match canonical cell_id {canonical!r} derived from "
        f"use_rule/prior/K — the arm string is the RNG-derivation key and must agree.",
    )
    return canonical


def _confirmatory_placeholder_violations(data: dict[str, Any]) -> list[str]:
    """Fields a confirmatory run must specify explicitly (never a development placeholder)."""
    bad: list[str] = []
    backbone = data.get("backbone") or {}
    for key in ("lr", "batch_size", "gamma"):
        if backbone.get(key) is None:
            bad.append(f"backbone.{key}")
    factor = data.get("factor_specific") or {}
    # The varied factor's own parameter must be pinned for a confirmatory run.
    if data.get("prior") == "on" and factor.get("prior_scale") is None:
        bad.append("factor_specific.prior_scale (required when prior: on)")
    if data.get("method") == "ddqn_egreedy" and factor.get("eps_schedule") is None:
        bad.append("factor_specific.eps_schedule (required for the ε-greedy baseline)")
    if data.get("method") in ENSEMBLE_METHODS:
        shared = data.get("ensemble_shared") or {}
        for key in ("mask_prob", "head_loss_agg"):
            if shared.get(key) is None:
                bad.append(f"ensemble_shared.{key}")
    return bad


def resolve_config(data: dict[str, Any], *, source_path: str | None = None) -> RunConfig:
    """Validate a raw config dict and return a resolved :class:`RunConfig`.

    Fills the derived identity fields (``cell_id``, ``size_class``) into a copy of ``data``
    so the resolved dict — the thing that gets hashed and serialized — is self-contained.
    """
    data = copy.deepcopy(data)  # we may rewrite nested keys (e.g. normalize prior)
    _normalize_prior(data)

    # --- required scalar identity ---
    for key in ("run_id", "role", "part", "method", "env", "master_seed", "seeds"):
        _require(key in data, f"config missing required field {key!r}")
    _require(
        data["role"] in VALID_ROLES,
        f"role must be one of {VALID_ROLES}, got {data['role']!r}",
    )
    _require(data["part"] in VALID_PARTS, f"part must be 'A' or 'B', got {data['part']!r}")
    _require(
        data["method"] in VALID_METHODS,
        f"method must be one of {VALID_METHODS}, got {data['method']!r}",
    )
    _require(
        data["env"] in VALID_ENVS,
        f"env must be one of {VALID_ENVS}, got {data['env']!r}",
    )
    _require(
        isinstance(data["master_seed"], int) and not isinstance(data["master_seed"], bool),
        f"master_seed must be an int, got {data['master_seed']!r}",
    )
    seeds = data["seeds"]
    _require(
        isinstance(seeds, list)
        and len(seeds) > 0
        and all(isinstance(s, int) and not isinstance(s, bool) for s in seeds),
        f"seeds must be a non-empty list of ints, got {seeds!r}",
    )
    _require(len(set(seeds)) == len(seeds), f"seeds must be unique, got {seeds!r}")

    # --- cell_id (RNG-derivation key) ---
    if data["part"] == "A":
        data["cell_id"] = _validate_part_a_factorial(data)
    else:
        # Part B: the factorial switchboard is not the identity; require an explicit arm.
        arm = data.get("arm")
        _require(
            isinstance(arm, str) and arm,
            "Part B config must set an explicit 'arm' (cell_id) string",
        )
        data["cell_id"] = arm

    # --- size_class (derive from role; validate the RunContext consistency rule) ---
    size_class = data.get("size_class")
    if size_class is None:
        size_class = "confirmatory" if data["role"] == "confirmatory" else "development"
    _require(
        size_class in conventions.VALID_SIZE_CLASSES,
        f"size_class must be one of {conventions.VALID_SIZE_CLASSES}, got {size_class!r}",
    )
    _require(
        not (size_class == "confirmatory" and data["role"] != "confirmatory"),
        f"size_class='confirmatory' requires role='confirmatory' (got role={data['role']!r})",
    )
    data["size_class"] = size_class

    # --- freeze discipline: confirmatory runs may not rely on development placeholders ---
    if data["role"] == "confirmatory":
        violations = _confirmatory_placeholder_violations(data)
        _require(
            not violations,
            "confirmatory config must pin every run-defining value explicitly; these are "
            f"unset (would fall back to a development placeholder): {violations}",
        )

    cfg = RunConfig(data=data, source_path=source_path)
    # Final cross-check: minting a RunContext runs conventions' own guards (e.g. QR-DQN role).
    cfg.run_context(seeds[0])
    return cfg


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a YAML run config from ``path``."""
    path = Path(path)
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    _require(isinstance(raw, dict), f"config at {path} did not parse to a mapping")
    return resolve_config(raw, source_path=str(path))


# --------------------------------------------------------------------------- #
# Factories (instantiate env + agent from a resolved config)
# --------------------------------------------------------------------------- #
def _deep_sea_dims(cfg: RunConfig) -> tuple[int, int, int]:
    """(size, obs_dim, n_actions) for a DeepSea run from the config's env_budget."""
    budget = cfg.data.get("env_budget") or {}
    size = budget.get("deep_sea_size")
    _require(
        isinstance(size, int) and size >= 1,
        f"env_budget.deep_sea_size must be a positive int, got {size!r}",
    )
    return size, size * size, 2


def build_env(cfg: RunConfig, seed_index: int):
    """Instantiate the environment for one seed. Raises for envs not yet implemented."""
    if cfg.env not in IMPLEMENTED_ENVS:
        raise NotImplementedError(
            f"env {cfg.env!r} is recognized but not yet implemented "
            f"(only {IMPLEMENTED_ENVS} build today)"
        )
    from src.deep_sea import DeepSea

    size, _, _ = _deep_sea_dims(cfg)
    return DeepSea(size, master_seed=cfg.master_seed, cell_id=cfg.cell_id, seed_index=seed_index)


def build_agent(cfg: RunConfig, seed_index: int):
    """Instantiate the agent for one seed. Raises for methods not yet implemented.

    Backbone/schedule numbers present in the config are used; any absent number falls back
    to the labelled development placeholder in ``DDQNConfig`` (permitted for development
    configs, forbidden for confirmatory ones — enforced at load time).
    """
    if cfg.method not in IMPLEMENTED_METHODS:
        raise NotImplementedError(
            f"method {cfg.method!r} is recognized but not yet implemented "
            f"(only {IMPLEMENTED_METHODS} build today; the ensemble methods arrive with the "
            "Bootstrapped-DQN agent)"
        )
    _require(
        cfg.env == "deep_sea",
        f"build_agent currently wires deep_sea only, got env={cfg.env!r}",
    )
    from src.ddqn import DDQNAgent, DDQNConfig

    _, obs_dim, n_actions = _deep_sea_dims(cfg)
    backbone = cfg.data.get("backbone") or {}
    factor = cfg.data.get("factor_specific") or {}

    kwargs: dict[str, Any] = {"obs_dim": obs_dim, "n_actions": n_actions}
    # Backbone (Class-1) — pass through only what the config specifies.
    for src_key, dst_key in (("lr", "lr"), ("batch_size", "batch_size"), ("gamma", "gamma")):
        if backbone.get(src_key) is not None:
            kwargs[dst_key] = backbone[src_key]
    for opt in ("hidden_sizes", "buffer_capacity", "min_buffer", "target_update_period"):
        if backbone.get(opt) is not None:
            kwargs[opt] = tuple(backbone[opt]) if opt == "hidden_sizes" else backbone[opt]
    # ε schedule (Class-3) — a nested mapping when specified.
    eps = factor.get("eps_schedule")
    if isinstance(eps, dict):
        for key in ("eps_start", "eps_end", "eps_decay_steps"):
            if eps.get(key) is not None:
                kwargs[key] = eps[key]

    config = DDQNConfig(**kwargs)
    return DDQNAgent(
        config, master_seed=cfg.master_seed, cell_id=cfg.cell_id, seed_index=seed_index
    )
