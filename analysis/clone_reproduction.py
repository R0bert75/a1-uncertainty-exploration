"""Freeze item 20: the deterministic MinAtar-cloning conditional.

Item 20 does not state a fact about MinAtar; it states a **decision procedure** whose outcome
selects how diagnostic 8 (the MinAtar behavior-policy analogue, §3.3) is computed:

* **100 clone/restore reproduction tests** — stored state + RNG state, fixed action sequence,
  two replays;
* ``full`` probe rollouts **iff all 100 are bit-exact**;
* ``episode_start_only`` **iff cloning fails but seeded fresh-reset rollouts are
  bit-reproducible**;
* ``drop`` otherwise.

The wording is exhaustive over the three outcomes, so running it always yields a decision. This
module runs it and writes an auditable record; it decides nothing on its own authority.

Two properties of the procedure are worth stating because they are easy to get wrong.

**"Bit-exact" is checked over what a probe rollout would actually consume**, not over the
observation alone: the compared trace is ``(reward, terminated, observation-bytes)`` at every
step. A state-only comparison would pass while rewards diverged.

**A replay-vs-replay comparison alone is not sufficient, and this was demonstrated, not
assumed.** Item 20 says "two replays", which catches a snapshot that aliases live env internals
(replay 1 mutates it, replay 2 diverges). But it does *not* catch a restore that is
self-consistently wrong. MinAtar's ``seed()`` makes the wrapper and the inner game share ONE
``RandomState``; a ``deepcopy`` of the game silently splits that into two generators, after
which both replays match each other **100/100** while neither reproduces the pre-snapshot
trajectory. This module therefore checks both conditions and a test counts as bit-exact only if
``replay_1 == replay_2 == original``. The stated procedure is a floor, not a ceiling: reporting
``full`` on a test that the restore path can pass while broken would put a silently wrong
diagnostic in the appendix.

The analogue this gates is **exploratory and appendix-only, and is never called "Q-error"**
(item 20). This module deliberately reports no Q-values at all.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.minatar_env import MinAtarEnv

#: Item 20 fixes the count; it is not a tunable.
N_REPRODUCTION_TESTS = 100

#: Steps per replay. Long enough for a sticky-action desync to surface (the game-only-snapshot
#: probe diverged well inside this horizon), short enough that 100 trials x 2 replays is cheap.
ROLLOUT_STEPS = 40

#: Burn-in range before snapshotting, so snapshots are taken from varied mid-episode states
#: rather than all from a fresh reset.
BURN_IN_RANGE = (5, 60)

DECISION_FULL = "full"
DECISION_EPISODE_START_ONLY = "episode_start_only"
DECISION_DROP = "drop"


def _trace(env: MinAtarEnv, actions) -> list[tuple]:
    """``(reward, terminated, obs-bytes)`` per step — everything a probe rollout consumes."""
    out = []
    for a in actions:
        obs, reward, terminated, _truncated, _info = env.step(int(a))
        out.append((float(reward), bool(terminated), obs.tobytes()))
        if terminated:
            env.reset()
    return out


def run_clone_tests(
    game: str = "breakout",
    n_tests: int = N_REPRODUCTION_TESTS,
    *,
    master_seed: int = 0,
    steps: int = ROLLOUT_STEPS,
) -> tuple[int, int]:
    """The 100 clone/restore tests. Returns ``(n_bit_exact, n_tests)``."""
    rng = np.random.RandomState(master_seed)
    n_ok = 0
    for i in range(n_tests):
        env = MinAtarEnv(game, master_seed=master_seed, cell_id=f"item20_clone_{i}", seed_index=0)
        env.reset()
        for _ in range(int(rng.randint(*BURN_IN_RANGE))):
            if env._done:
                env.reset()
            env.step(int(rng.randint(env.n_actions)))
        if env._done:
            env.reset()
        actions = [int(rng.randint(env.n_actions)) for _ in range(steps)]

        snapshot = env.clone_state()
        original = _trace(env, actions)  # the trajectory the snapshot must reproduce
        env.restore_state(snapshot)
        replay_1 = _trace(env, actions)
        env.restore_state(snapshot)
        replay_2 = _trace(env, actions)
        # Both conditions: replays agree AND they reproduce the original. See the module
        # docstring -- a de-aliased RNG passes the first and fails the second.
        n_ok += replay_1 == replay_2 == original
    return n_ok, n_tests


def run_fresh_reset_tests(
    game: str = "breakout",
    n_tests: int = N_REPRODUCTION_TESTS,
    *,
    master_seed: int = 0,
    steps: int = ROLLOUT_STEPS,
) -> tuple[int, int]:
    """The fallback branch: are *seeded fresh-reset* rollouts bit-reproducible?

    Two independently constructed envs on identical derivation coordinates, reset and driven
    through the same action sequence, must produce identical traces. This is what
    ``episode_start_only`` probing would rely on.
    """
    rng = np.random.RandomState(master_seed + 1)
    n_ok = 0
    for i in range(n_tests):
        cell = f"item20_fresh_{i}"
        actions = None
        traces = []
        for _replica in range(2):
            env = MinAtarEnv(game, master_seed=master_seed, cell_id=cell, seed_index=0)
            env.reset()
            if actions is None:
                actions = [int(rng.randint(env.n_actions)) for _ in range(steps)]
            traces.append(_trace(env, actions))
        n_ok += traces[0] == traces[1]
    return n_ok, n_tests


@dataclass
class Item20Result:
    game: str
    decision: str
    clone_bit_exact: int
    clone_tests: int
    fresh_reset_bit_exact: int
    fresh_reset_tests: int
    rollout_steps: int
    rationale: str


def decide(game: str = "breakout", n_tests: int = N_REPRODUCTION_TESTS, **kw) -> Item20Result:
    """Run the procedure and return item 20's decision for ``game``."""
    clone_ok, clone_n = run_clone_tests(game, n_tests, **kw)
    # The fallback branch is evaluated unconditionally so the record shows both numbers; the
    # decision below still reads them in item 20's order.
    fresh_ok, fresh_n = run_fresh_reset_tests(game, n_tests, **kw)

    if clone_ok == clone_n:
        decision = DECISION_FULL
        rationale = (
            f"all {clone_n} clone/restore reproduction tests bit-exact -> full probe rollouts"
        )
    elif fresh_ok == fresh_n:
        decision = DECISION_EPISODE_START_ONLY
        rationale = (
            f"cloning failed ({clone_ok}/{clone_n}) but seeded fresh-reset rollouts are "
            f"bit-reproducible ({fresh_ok}/{fresh_n}) -> episode-start-only probing"
        )
    else:
        decision = DECISION_DROP
        rationale = (
            f"cloning failed ({clone_ok}/{clone_n}) and fresh-reset rollouts are not "
            f"bit-reproducible ({fresh_ok}/{fresh_n}) -> drop the analogue"
        )
    return Item20Result(
        game=game,
        decision=decision,
        clone_bit_exact=clone_ok,
        clone_tests=clone_n,
        fresh_reset_bit_exact=fresh_ok,
        fresh_reset_tests=fresh_n,
        rollout_steps=kw.get("steps", ROLLOUT_STEPS),
        rationale=rationale,
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run freeze item 20's MinAtar-cloning conditional.")
    p.add_argument("--game", default="breakout")
    p.add_argument("--n-tests", type=int, default=N_REPRODUCTION_TESTS)
    p.add_argument("--out", type=Path, default=Path("audits/item20_clone_reproduction.json"))
    args = p.parse_args(argv)

    result = decide(args.game, args.n_tests)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(result), indent=2) + "\n")
    print(f"item 20 [{result.game}]: {result.decision} — {result.rationale}")
    print(f"record → {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
