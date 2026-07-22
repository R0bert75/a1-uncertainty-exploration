"""DeepSea — the Part A controlled-mechanism environment, with exact ground-truth Q*.

DeepSea (Osband et al. 2016/2018; canonical implementation in DeepMind ``bsuite``) is the
hard-exploration diagnostic Part A replicates. On an N×N grid the agent starts top-left and
descends one row per step; at each step it chooses one of two actions, exactly one of which
moves it one column *toward* the single treasure at the bottom-right. The correct ("right")
action is scrambled per state by a fixed random mapping, so an uninformed agent reaches the
treasure with probability only ``2**(-N)`` — deep, directed exploration is required. Reward
is ``+1`` for the treasure and a small ``-move_cost/N`` per rightward move, giving an optimal
return of ``1 - move_cost`` (= 0.99 at the bsuite default ``move_cost = 0.01``).

**Two deliberate consistency choices with the frozen infrastructure and the pinned stack**
(both documented here so gate C5 review can see them):

1. **Per-row action mapping, not per-cell.** Canonical bsuite draws an N×N Bernoulli(0.5)
   mapping (one flip per *cell*). This repo's *frozen* ``conventions.deepsea_action_mapping``
   (reviewer Fix 4, bound to the ``env_mapping`` RNG stream, hashed, unit-tested, committed)
   returns **one flip per row**. This environment consumes that frozen mapping directly, so
   ground-truth Q* and the run's stored ``deepsea_mapping_hash`` always describe the *same*
   mapping the agent acts under. The exploration difficulty is preserved: because the agent
   only ever occupies cells with ``col <= row`` and can reach ``col == N-1`` only at the final
   row, exactly N distinct rows gate the optimal path, each an independent coin flip → the
   same ``2**(-N)`` uninformed-success probability as canonical DeepSea.
2. **``bsuite`` is not a dependency.** It pins an old ``gym`` incompatible with the pinned
   ``gymnasium==1.3.0`` / numpy 2.x stack (SESSION0_REPORT §"bsuite not installed"). This is a
   single-file re-implementation validated against the published mechanics on small N, not a
   new benchmark.

``move_cost = 0.01`` and treasure ``= +1`` are structural constants of the DeepSea definition
(the bsuite default that fixes the 0.99 optimal return the advisor one-pager cites), not
owner-reserved frozen protocol values; they are constructor arguments defaulting to the
canonical values. Grid ``size`` and the episode budget are Class-frozen elsewhere (freeze
item 5) and always supplied by the caller — none are baked in here.

The exact solver (:meth:`DeepSea.q_star`) is exposed now because "DeepSea with exact
ground-truth Q*" is the environment's defining feature for Part A; the uncertainty battery
that *consumes* Q* is Session-6 work and lives elsewhere. ``gamma`` is an explicit argument
(no hidden default value) so a diagnostic can request Q* under the same discount the agent
learns with.
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

MOVE_COST_DEFAULT = 0.01  # bsuite canonical: optimal return = 1 - move_cost = 0.99
TREASURE_REWARD = 1.0


class DeepSea(_BASE):
    """Deterministic N×N DeepSea as a Gymnasium environment (discrete 2-action).

    The action mapping is derived from the run's frozen ``env_mapping`` stream via
    ``conventions.deepsea_action_mapping(master_seed, cell_id, seed_index, size)`` and
    fingerprinted with ``conventions.deepsea_mapping_hash``; the hash is returned in every
    ``info`` dict and should be stored with the run so every Q*-referenced diagnostic can
    assert it is operating on the same mapping.

    Observation: an ``(size, size)`` float32 one-hot of the current cell (all-zeros null
    observation once the episode has ended), matching bsuite. The agent flattens it before
    the network (see :mod:`src.networks`). Action space: ``Discrete(2)``.

    Args:
        size: grid dimension N (Class-frozen tier value; passed in).
        master_seed, cell_id, seed_index: RNG-derivation identity for the action mapping.
        move_cost: per-right-move penalty scale (default the bsuite canonical 0.01).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        size: int,
        *,
        master_seed: int,
        cell_id: str,
        seed_index: int,
        move_cost: float = MOVE_COST_DEFAULT,
    ) -> None:
        if _HAS_GYM:
            super().__init__()
        if size < 1:
            raise ValueError(f"size must be >= 1, got {size}")
        self.size = int(size)
        self.master_seed = int(master_seed)
        self.cell_id = str(cell_id)
        self.seed_index = int(seed_index)
        self.move_cost = float(move_cost)

        # Per-row "right" action, from the frozen env_mapping stream. right_action[r] in {0,1}
        # is the action index that advances the agent toward the treasure at row r.
        self._row_flip = conventions.deepsea_action_mapping(
            self.master_seed, self.cell_id, self.seed_index, self.size
        )
        self.right_action = self._row_flip.astype(np.int64)  # shape (size,)
        self.mapping_hash = conventions.deepsea_mapping_hash(self._row_flip)

        if _HAS_GYM:
            self.action_space = spaces.Discrete(2)
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(self.size, self.size), dtype=np.float32
            )

        self._row = 0
        self._col = 0
        self._done = True  # force reset() before step()

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def _observation(self) -> np.ndarray:
        obs = np.zeros((self.size, self.size), dtype=np.float32)
        if self._row < self.size:  # else: end-of-episode null observation (all zeros)
            obs[self._row, self._col] = 1.0
        return obs

    def _info(self) -> dict:
        return {
            "mapping_hash": self.mapping_hash,
            "row": self._row,
            "col": self._col,
            "optimal_return": self.optimal_return,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset to the top-left cell. Transitions are deterministic; ``seed`` is unused
        for dynamics (the mapping is fixed at construction from the derived stream) and is
        forwarded only to satisfy the Gymnasium contract."""
        if _HAS_GYM:
            super().reset(seed=seed)
        self._row = 0
        self._col = 0
        self._done = False
        return self._observation(), self._info()

    def step(self, action: int):
        """Advance one row. Returns ``(obs, reward, terminated, truncated, info)``.

        Deterministic dynamics (bsuite ``deterministic=True``): a right action advances the
        column (clipped at N-1) and costs ``move_cost/size``; a wrong action retreats the
        column (clipped at 0) at no cost. The treasure ``+1`` is paid when a right action is
        taken while already at the last column — reachable only at the final row, so it fires
        at most once. The episode terminates when the agent steps off the last row.
        """
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset() first")
        action = int(action)
        if action not in (0, 1):
            raise ValueError(f"action must be 0 or 1, got {action}")

        reward = 0.0
        action_right = action == self.right_action[self._row]
        if self._col == self.size - 1 and action_right:
            reward += TREASURE_REWARD
        if action_right:
            self._col = min(self._col + 1, self.size - 1)
            reward -= self.move_cost / self.size
        else:
            self._col = max(self._col - 1, 0)
        self._row += 1

        terminated = self._row == self.size
        self._done = terminated
        obs = self._observation()
        return obs, reward, terminated, False, self._info()

    # ------------------------------------------------------------------ #
    # Exact ground truth (Part A's defining feature)
    # ------------------------------------------------------------------ #
    @property
    def optimal_return(self) -> float:
        """Undiscounted optimal episode return: ``1 - move_cost`` (deterministic DeepSea)."""
        return TREASURE_REWARD - self.move_cost

    def q_star(self, gamma: float = 1.0) -> np.ndarray:
        """Exact optimal action-value ``Q*[row, col, action]`` by backward induction.

        Computed over the full ``(size, size, 2)`` grid (unreachable cells included; they are
        simply never queried). ``gamma`` is explicit — pass the agent's discount when the
        diagnostic compares learned values against Q* under matched discounting; the default
        ``1.0`` is the environment's own undiscounted optimum (return ``1 - move_cost``).

        Deterministic dynamics make this a plain finite-horizon DP: from ``(row, col)`` each
        action leads to exactly one ``(row+1, col')`` with a known reward; the value of any
        state at ``row == size`` is 0 (episode over).
        """
        n = self.size
        # V[row, col]; row == n is the absorbing post-terminal layer with value 0.
        V = np.zeros((n + 1, n), dtype=np.float64)
        Q = np.zeros((n, n, 2), dtype=np.float64)
        for row in range(n - 1, -1, -1):
            right = self.right_action[row]
            for col in range(n):
                for action in (0, 1):
                    is_right = action == right
                    reward = 0.0
                    if col == n - 1 and is_right:
                        reward += TREASURE_REWARD
                    if is_right:
                        next_col = min(col + 1, n - 1)
                        reward -= self.move_cost / n
                    else:
                        next_col = max(col - 1, 0)
                    Q[row, col, action] = reward + gamma * V[row + 1, next_col]
                V[row, col] = Q[row, col].max()
        return Q

    def optimal_action(self, row: int, col: int) -> int:
        """Greedy optimal action at ``(row, col)`` under undiscounted Q* (ties → lower index)."""
        q = self.q_star(gamma=1.0)[row, col]
        return int(np.argmax(q))  # np.argmax returns the lowest index on ties
