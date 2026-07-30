# Decision note — MinAtar backbone and env adapter

**Recorded 2026-07-24 · pre-freeze infrastructure · touches no Class-2 frozen value**

This note records the MinAtar wiring decisions so the choices are auditable and were made
*before* any MinAtar run, not fitted to one. It is an implementation record, not a protocol
amendment: nothing here changes a frozen value, and the pre-registration is untouched.

---

## 1. Classification: the conv trunk is Class-1, not Class-2

The pre-registration parameter table (§ "Parameter table", Class 1) defines **Class 1 — Backbone
nuisance** as "tuned once on the ε-greedy DDQN backbone; inherited identically by all cells …
Learning rate, optimizer, replay size, batch size, target-update cadence, **network width**."

**Network architecture/width is therefore Class 1.** The Class-2 table (ensemble-shared nuisance,
"fixed to explicitly listed values, NOT tuned") contains **no** convolutional-architecture row —
it constrains the *shared-trunk convention*, head initialization, and per-head targets, not the
trunk's internal shape. Consequently: choosing the MinAtar conv stack here **does not set,
override, or pre-empt any Class-2 frozen value**, and the shape below is a default that the
Class-1 backbone search may later revise (identically for all methods, per the class definition).

## 2. What *is* frozen and must be honoured

The Class-2 **shared-trunk convention** row is binding and explicitly anticipates the conv case:

> "Single shared feature trunk; **K independent value heads split after the shared
> representation** … Osband et al. **2016** §6 / Fig. 1a: *'split 10 separate bootstrap heads
> after the convolutional layer'*; architecture *'identical to DQN except we split K heads.'*
> Documented design choice for the DeepSea backbone scale (**no convolutional stack at DeepSea
> resolution**)."

The parenthetical is the tell: the absence of a conv stack is scoped *to DeepSea resolution*. On
MinAtar the same convention applies in its original form — heads split **after the convolutional
representation**. Also binding: **head initialization** independent per head (the diversity prior)
and **K per-head target networks**.

## 3. Adopted default shape

Single shared trunk, then `n_heads` independent linear value heads:

```
input  (B, C, 10, 10)          C = per-game channel count (below)
conv   Conv2d(C, 16, k=3, s=1) valid padding -> (B, 16, 8, 8)   + ReLU
flatten                        -> (B, 1024)
fc     Linear(1024, 128)       + ReLU                  <- end of SHARED trunk
heads  n_heads x Linear(128, n_actions)                <- split here (Osband 2016 Fig 1a)
```

**Provenance.** This is the standard MinAtar DQN torso from the benchmark's own reference
implementation (Young & Tian 2019, *MinAtar: An Atari-Inspired Testbed*), i.e. a single 16-filter
3×3 convolution followed by a 128-unit fully-connected layer. It is **already the shape used in
this repo** by `compute/benchmarks/stage_a_benchmark.py::MinAtarDQN`, which produced the Stage-A
compute forecast — so adopting it keeps the compute worksheet's timing basis valid rather than
invalidating a measurement we already paid for.

**Contract parity.** `MinAtarConvQNetwork` mirrors `MLPQNetwork`'s interface exactly
(`reset_parameters(generator)`, `trunk_features`, `heads_forward`, `forward` → `[batch, n_heads,
n_actions]`). `n_heads=1` is the DDQN/NoisyNet backbone; `n_heads=K` is Bootstrapped-DQN on the
*same class*. This is what keeps gate **C11 (code-path purity)** checkable on MinAtar exactly as
it is on DeepSea, and lets RP-BDQN remain the `prior=on` arm of the same agent.

**NoisyNet variant.** `NoisyMinAtarConvQNetwork` applies `NoisyLinear` to the **fully-connected
and head layers only**, leaving the convolution deterministic — the Fortunato et al. 2018
convention (noise on the linear layers of the value stream).

## 4. MinAtar package API — verified surface

Verified by introspection on 2026-07-24, because the adapter must not rely on assumptions:

- `Environment(env_name, sticky_action_prob=0.1, difficulty_ramping=True)` — **no** `random_seed`
  constructor argument (an earlier assumption; it does not exist in this version).
- `seed(seed)` sets `np.random.RandomState(seed)` and propagates it to the inner game
  (`self.env.random`). **`reset()` does not reseed** — so seeding once after construction yields a
  reproducible *episode stream*, which is the behaviour the trainer wants.
- `act(a) -> (reward, terminated)` — a 2-tuple, **not** the gymnasium 5-tuple. Sticky actions are
  applied inside `act` off the env's own RandomState.
- `reset() -> None` (mutates in place; the adapter must call `state()` afterwards).
- `state() -> np.ndarray` of dtype **bool**, shape **(10, 10, C)** — *channel-last*. The adapter
  transposes to channel-first `(C, 10, 10)` float32 for conv input.
- `num_actions()` returns **6 for all five games**; `minimal_action_set()` is smaller and
  game-specific. We use the **full 6-action set** so the action space is identical across games
  and across methods (any per-game action-set difference would alias game onto method comparisons).

**Per-game observation channels** (verified):

| Game | `state_shape` | channels | `num_actions()` | `minimal_action_set()` |
|---|---|---:|---:|---:|
| breakout | (10, 10, 4) | 4 | 6 | 3 |
| asterix | (10, 10, 4) | 4 | 6 | 5 |
| seaquest | (10, 10, 10) | 10 | 6 | 6 |
| space_invaders | (10, 10, 6) | 6 | 6 | 4 |
| freeway | (10, 10, 7) | 7 | 6 | 3 |

Per the spec, tuning games are **breakout + asterix**; the held-out evaluation set is the
remaining three (**seaquest, space_invaders, freeway**).

## 5. Pinned env settings (explicit, not package defaults)

Both are pinned in the adapter rather than inherited, so a future package-default change cannot
silently alter the environment:

- `sticky_action_prob = 0.1` — the MinAtar default and the standard setting in the benchmark's
  reported results.
- `difficulty_ramping = True` — the MinAtar default; ramping is part of the benchmark as published.

## 6. Determinism / gate C1

The env is seeded exclusively from the frozen **`env_mapping`** stream; the adapter draws no
randomness of its own. Verified empirically: identical seed + identical action sequence reproduces
the reward and state traces exactly; a different seed diverges. No new RNG stream is introduced —
the 8-name registry stays frozen.

**32-bit narrowing (implementation detail worth recording).** MinAtar seeds a *legacy*
`np.random.RandomState`, which accepts only 32-bit seeds, whereas `conventions.derive_seed`
returns the frozen **63-bit** int form (masked for `torch.manual_seed`). Passing the 63-bit value
through raises `ValueError: Seed must be between 0 and 2**32 - 1`. Rather than masking that int ad
hoc — which would invent a narrowing convention not in the frozen registry — the adapter narrows
through numpy's own canonical path:

```python
ss = conventions.derive_seed_sequence(master_seed, cell_id, "env_mapping", seed_index)
env_seed = int(ss.generate_state(1, dtype=np.uint32)[0])
```

`derive_seed_sequence` is documented in `conventions` as *the preferred entry point for numpy
RNGs*, and `SeedSequence.generate_state` is numpy's standard 32-bit state derivation. The env seed
therefore remains a pure, platform-stable function of
`(master_seed, cell_id, "env_mapping", seed_index)` — no new convention, no ad-hoc bit masking,
and gate C1 is unaffected. Asserted in `tests/test_minatar_env.py` for all five games.

## 7. What this note does *not* decide

Class-1 backbone values (learning rate, optimizer, replay size, batch size, target-update cadence,
and the trunk widths above) are **tuned once on the ε-greedy DDQN backbone** under the
pre-registered search budget and then inherited identically by all methods. The shape here is the
starting default for that search, not its result.
