"""MinAtar (Part B) behind the same gymnasium-style contract the trainer uses for DeepSea.

The MinAtar package's own API differs from gymnasium in three ways that matter, all verified
by introspection and recorded in ``protocol/decisions/minatar_backbone.md``:

* ``Environment.act(a)`` returns a **2-tuple** ``(reward, terminated)``, not the gymnasium
  5-tuple; ``reset()`` returns ``None`` and mutates in place.
* there is **no** ``random_seed`` constructor argument — seeding goes through ``seed(s)``,
  which installs an ``np.random.RandomState`` and propagates it to the inner game. ``reset()``
  does *not* reseed, so one ``seed()`` call after construction fixes the whole episode stream.
* ``state()`` is dtype ``bool`` with shape ``(10, 10, C)`` — **channel-last**.

This adapter normalizes all three: gymnasium ``reset() -> (obs, info)`` /
``step(a) -> (obs, reward, terminated, truncated, info)``, and observations as ``float32``
**channel-first** ``(C, 10, 10)`` arrays ready for the conv trunk (no flattening — that is the
one structural difference from the DeepSea path).

Determinism (gate C1): the env seed is derived **exclusively** from the frozen ``env_mapping``
stream via :func:`conventions.derive_seed`. The adapter draws no randomness of its own and adds
no new stream — the 8-name registry stays frozen.

Two env settings are pinned explicitly rather than inherited from the package defaults, so a
future package change cannot silently alter the environment: ``sticky_action_prob = 0.1`` and
``difficulty_ramping = True`` (both are the MinAtar defaults and the settings under which the
benchmark's published results were produced).

The **full 6-action set** is used for every game (``num_actions()`` is 6 across all five),
rather than the smaller per-game ``minimal_action_set()``: an action space that varied by game
would alias game onto method comparisons.
"""

from __future__ import annotations

import numpy as np

try:  # gymnasium is the pinned RL env API; keep the module importable without it for docs.
    import gymnasium as gym
    from gymnasium import spaces

    _BASE = gym.Env
    _HAS_GYM = True
except ImportError:  # pragma: no cover - exercised only in a gym-less environment
    _BASE = object
    _HAS_GYM = False

from src.utils import conventions

#: The five canonical MinAtar games. Tuning = breakout + asterix (spec §3.4); the held-out
#: evaluation set is the remaining three.
MINATAR_GAMES: tuple[str, ...] = (
    "breakout",
    "asterix",
    "seaquest",
    "space_invaders",
    "freeway",
)

#: Tuning / held-out split, frozen in the spec (freeze item: environments row).
MINATAR_TUNING_GAMES: tuple[str, ...] = ("breakout", "asterix")
MINATAR_HELDOUT_GAMES: tuple[str, ...] = ("seaquest", "space_invaders", "freeway")

#: Verified per-game observation channel counts (``state_shape() == (10, 10, C)``).
#: Asserted against the live package in :meth:`MinAtarEnv.__init__`, so a package change
#: surfaces as a loud failure instead of a silently reshaped network.
MINATAR_CHANNELS: dict[str, int] = {
    "breakout": 4,
    "asterix": 4,
    "seaquest": 10,
    "space_invaders": 6,
    "freeway": 7,
}

GRID_SIZE = 10  # MinAtar is a 10x10 grid for every game
N_ACTIONS = 6  # num_actions() is 6 for all five games (full set, not minimal_action_set)

# Pinned env settings — explicit, never inherited from package defaults.
STICKY_ACTION_PROB = 0.1
DIFFICULTY_RAMPING = True


class MinAtarEnv(_BASE):
    """One MinAtar game behind the gymnasium contract, seeded from ``env_mapping``.

    Args:
        game: one of :data:`MINATAR_GAMES`.
        master_seed, cell_id, seed_index: the frozen RNG-derivation coordinates. The env
            seed is ``conventions.derive_seed(master_seed, cell_id, "env_mapping", seed_index)``.
        sticky_action_prob, difficulty_ramping: pinned defaults; exposed so a diagnostic can
            construct a deliberately different env, never to be silently varied in a run.
    """

    def __init__(
        self,
        game: str,
        *,
        master_seed: int,
        cell_id: str,
        seed_index: int,
        sticky_action_prob: float = STICKY_ACTION_PROB,
        difficulty_ramping: bool = DIFFICULTY_RAMPING,
    ) -> None:
        if _HAS_GYM:
            super().__init__()
        if game not in MINATAR_GAMES:
            raise ValueError(f"game must be one of {MINATAR_GAMES}, got {game!r}")

        from minatar import Environment as _MinAtarEnvironment

        self.game = str(game)
        self.master_seed = int(master_seed)
        self.cell_id = str(cell_id)
        self.seed_index = int(seed_index)
        self.sticky_action_prob = float(sticky_action_prob)
        self.difficulty_ramping = bool(difficulty_ramping)

        self._env = _MinAtarEnvironment(
            self.game,
            sticky_action_prob=self.sticky_action_prob,
            difficulty_ramping=self.difficulty_ramping,
        )

        # Sole source of env randomness: the frozen env_mapping stream.
        #
        # MinAtar seeds a legacy ``np.random.RandomState``, which accepts only 32-bit seeds,
        # while ``conventions.derive_seed`` returns the frozen 63-bit int form. Rather than
        # masking that int ad hoc, narrow through the canonical numpy entry point:
        # ``derive_seed_sequence`` is documented as *the* preferred numpy path, and
        # ``SeedSequence.generate_state(1, uint32)`` is its own standard 32-bit derivation.
        # So the env seed remains a pure function of
        # ``(master_seed, cell_id, "env_mapping", seed_index)`` with no new randomness.
        self.seed_sequence = conventions.derive_seed_sequence(
            self.master_seed, self.cell_id, "env_mapping", self.seed_index
        )
        self.env_seed = int(self.seed_sequence.generate_state(1, dtype=np.uint32)[0])
        self._env.seed(self.env_seed)

        shape = tuple(int(x) for x in self._env.state_shape())
        expected = (GRID_SIZE, GRID_SIZE, MINATAR_CHANNELS[self.game])
        if shape != expected:
            raise RuntimeError(
                f"installed minatar reports state_shape {shape} for {self.game!r}, "
                f"expected {expected}; the pinned channel table in src/minatar_env.py "
                "and the conv-trunk input width would be wrong"
            )
        n_act = int(self._env.num_actions())
        if n_act != N_ACTIONS:
            raise RuntimeError(
                f"installed minatar reports num_actions {n_act} for {self.game!r}, "
                f"expected {N_ACTIONS}"
            )

        self.n_channels = MINATAR_CHANNELS[self.game]
        self.n_actions = N_ACTIONS
        #: Conv-trunk input shape, channel-first.
        self.obs_shape: tuple[int, int, int] = (self.n_channels, GRID_SIZE, GRID_SIZE)

        if _HAS_GYM:
            self.action_space = spaces.Discrete(self.n_actions)
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=self.obs_shape, dtype=np.float32
            )

        self._done = True  # force reset() before step()

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def _observation(self) -> np.ndarray:
        """``(C, 10, 10)`` float32 — MinAtar's ``(10, 10, C)`` bool state, channel-first."""
        state = self._env.state()  # (10, 10, C) bool
        return np.ascontiguousarray(np.transpose(state, (2, 0, 1)), dtype=np.float32)

    def _info(self) -> dict:
        return {
            "game": self.game,
            "env_seed": self.env_seed,
            "n_channels": self.n_channels,
            "sticky_action_prob": self.sticky_action_prob,
            "difficulty_ramping": self.difficulty_ramping,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Start a new episode. ``seed`` re-seeds only if explicitly passed (tests/diagnostics).

        In a run it is never passed: the env is seeded once at construction from the
        ``env_mapping`` stream, and MinAtar's ``reset()`` deliberately does not reseed, so the
        episode stream stays a single reproducible sequence.
        """
        if seed is not None:
            self.env_seed = int(seed)
            self._env.seed(self.env_seed)
        self._env.reset()
        self._done = False
        return self._observation(), self._info()

    def step(self, action: int):
        """Gymnasium 5-tuple. MinAtar has no time limit of its own, so ``truncated`` is False."""
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset() first")
        act = int(action)
        if not 0 <= act < self.n_actions:
            raise ValueError(f"action must be in [0, {self.n_actions}), got {act}")

        reward, terminated = self._env.act(act)  # MinAtar 2-tuple
        self._done = bool(terminated)
        return self._observation(), float(reward), bool(terminated), False, self._info()

    # ------------------------------------------------------------------ #
    # State clone / restore (freeze item 20, diagnostic 8)
    # ------------------------------------------------------------------ #
    # Freeze item 20 makes the MinAtar behavior-policy analogue conditional on 100 bit-exact
    # clone/restore reproduction tests. That conditional is only *evaluable* if the adapter can
    # snapshot and restore state at all, so the capability is a protocol requirement, not a
    # convenience. See ``analysis/clone_reproduction.py`` for the test procedure itself.
    #
    # MinAtar keeps simulator state in four places: the inner game object, the RNG,
    # ``Environment.last_action`` (the action a sticky repeat replays), and this adapter's
    # episode-finished flag. Snapshotting the inner game ALONE reproduces only ~4/100 rollouts,
    # so the wrapper-level pieces are load-bearing, not incidental.
    #
    # THE SUBTLE PART — RNG ALIASING. ``Environment.seed()`` does
    # ``self.random = RandomState(s); self.env.random = self.random``: wrapper and game share
    # ONE generator object, and the sticky-action coin flip in ``Environment.act()`` draws from
    # the same stream the game's own dynamics draw from. A naive ``deepcopy(self._env.env)``
    # copies that generator, so after restore the two hold SEPARATE generators and the single
    # interleaved stream becomes two independent ones.
    #
    # That failure is invisible to item 20's stated test. Two replays from a de-aliased
    # snapshot still match EACH OTHER perfectly (both are wrong in the same way) — measured
    # 100/100 — while neither reproduces the pre-snapshot trajectory. Restoring therefore
    # re-establishes the aliasing explicitly and seeds the shared generator once, and
    # ``analysis/clone_reproduction.py`` compares replays against the ORIGINAL rollout in
    # addition to each other, which is the check that actually detects this.

    def clone_state(self) -> dict:
        """Snapshot the full simulator state; see :meth:`restore_state`.

        Returns a deep-copied, self-contained dict — safe to hold across further stepping,
        and never aliased to live env internals. The RNG is captured once, as a state tuple,
        because wrapper and game share a single generator (see the note above).
        """
        import copy as _copy

        game = self._env.env
        rng_aliased = self._env.random is getattr(game, "random", None)
        # Detach the generator before copying so the snapshot holds no generator object at all;
        # the position is captured separately as an explicit state tuple.
        saved = getattr(game, "random", None)
        try:
            if hasattr(game, "random"):
                game.random = None
            game_copy = _copy.deepcopy(game)
        finally:
            if saved is not None:
                game.random = saved

        snap = {
            "game": game_copy,
            "rng": self._env.random.get_state(),
            "rng_aliased": rng_aliased,
            "last_action": self._env.last_action,
            "done": bool(self._done),
        }
        if not rng_aliased:  # defensive: an unseeded env can hold two distinct generators
            snap["game_rng"] = saved.get_state() if saved is not None else None
        return snap

    def restore_state(self, snapshot: dict) -> None:
        """Restore a :meth:`clone_state` snapshot in place.

        The snapshot is deep-copied on the way out as well as in, so one snapshot can seed many
        replays (item 20 replays each twice; probe rollouts would replay far more).
        """
        import copy as _copy

        missing = {"game", "rng", "rng_aliased", "last_action", "done"} - set(snapshot)
        if missing:
            raise ValueError(f"snapshot missing required key(s): {sorted(missing)}")

        game = _copy.deepcopy(snapshot["game"])
        self._env.random.set_state(snapshot["rng"])
        if snapshot["rng_aliased"]:
            # Re-establish MinAtar's shared-generator invariant. Without this the restored env
            # runs two independent streams where the original ran one.
            game.random = self._env.random
        else:
            game_rng = np.random.RandomState()
            if snapshot.get("game_rng") is not None:
                game_rng.set_state(snapshot["game_rng"])
            game.random = game_rng
        self._env.env = game
        self._env.last_action = snapshot["last_action"]
        self._done = bool(snapshot["done"])

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    @property
    def obs_dim(self) -> int:
        """Flattened observation size. Present for parity with the DeepSea/MLP path only —
        the conv trunk consumes :attr:`obs_shape`, unflattened."""
        return int(np.prod(self.obs_shape))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"MinAtarEnv(game={self.game!r}, obs_shape={self.obs_shape}, "
            f"n_actions={self.n_actions}, env_seed={self.env_seed})"
        )
