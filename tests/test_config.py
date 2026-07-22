"""Tests for the run-config loader (src/config.py).

Covers: structural validation (vocabularies, required fields), the canonical cell_id / arm
agreement (RNG-derivation key), size_class derivation + the RunContext consistency rule, the
freeze-discipline guard (confirmatory configs may not inherit development placeholders), C13
identity stability + serialization, and the env/agent factory round-trip that instantiates
and steps a DDQN agent on DeepSea entirely from a config.
"""

from __future__ import annotations

import copy

import pytest

from src.config import (
    ConfigError,
    RunConfig,
    build_agent,
    build_env,
    load_config,
    resolve_config,
)
from src.utils import conventions


def _base_dev_ddqn() -> dict:
    return {
        "run_id": "t_ddqn_dev",
        "role": "development",
        "part": "A",
        "method": "ddqn_egreedy",
        "env": "deep_sea",
        "master_seed": 0,
        "use_rule": "episodic",
        "prior": "off",
        "K": 1,
        "arm": "episodic|off|K1",
        "backbone": {"lr": 5e-4, "batch_size": 32, "gamma": 0.99, "hidden_sizes": [32, 32]},
        "factor_specific": {
            "prior_scale": None,
            "eps_schedule": {"eps_start": 1.0, "eps_end": 0.05, "eps_decay_steps": 2000},
        },
        "env_budget": {"deep_sea_size": 5, "episodes": 100},
        "seeds": [0, 1, 2],
    }


# --------------------------------------------------------------------------- #
# Loading the committed example configs
# --------------------------------------------------------------------------- #
def test_example_configs_load_and_validate():
    for name in ("example_ddqn_deepsea_dev.yaml", "example_bdqn_deepsea_dev.yaml"):
        cfg = load_config(f"configs/{name}")
        assert isinstance(cfg, RunConfig)
        assert cfg.cell_id == cfg.data["arm"]
        assert len(cfg.config_sha256) == 64


# --------------------------------------------------------------------------- #
# Structural validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "mutate,frag",
    [
        (lambda d: d.pop("master_seed"), "master_seed"),
        (lambda d: d.update(role="bogus"), "role"),
        (lambda d: d.update(part="C"), "part"),
        (lambda d: d.update(method="dqn"), "method"),
        (lambda d: d.update(env="pong"), "env"),
        (lambda d: d.update(seeds=[]), "seeds"),
        (lambda d: d.update(seeds=[0, 0, 1]), "unique"),
        (lambda d: d.update(master_seed=1.5), "master_seed"),
    ],
)
def test_invalid_configs_are_rejected(mutate, frag):
    d = _base_dev_ddqn()
    mutate(d)
    with pytest.raises(ConfigError) as e:
        resolve_config(d)
    assert frag in str(e.value)


def test_arm_must_match_canonical_cell_id():
    d = _base_dev_ddqn()
    d["arm"] = "episodic|off|K10"          # disagrees with K=1
    with pytest.raises(ConfigError) as e:
        resolve_config(d)
    assert "canonical cell_id" in str(e.value)


def test_cell_id_is_derived_from_switchboard():
    d = _base_dev_ddqn()
    d.update(use_rule="per_step", prior="on", K=20, arm="per_step|on|K20")
    cfg = resolve_config(d)
    assert cfg.cell_id == "per_step|on|K20"


# --------------------------------------------------------------------------- #
# size_class derivation + RunContext consistency
# --------------------------------------------------------------------------- #
def test_size_class_defaults_from_role():
    assert resolve_config(_base_dev_ddqn()).size_class == "development"


def test_confirmatory_size_class_requires_confirmatory_role():
    d = _base_dev_ddqn()
    d["size_class"] = "confirmatory"        # but role is development
    with pytest.raises(ConfigError) as e:
        resolve_config(d)
    assert "requires role='confirmatory'" in str(e.value)


def test_run_context_shares_hash_across_seeds():
    cfg = resolve_config(_base_dev_ddqn())
    c0, c1 = cfg.run_context(0), cfg.run_context(1)
    assert c0.config_sha256 == c1.config_sha256 == cfg.config_sha256
    assert (c0.seed, c1.seed) == (0, 1)
    assert c0.extra["cell_id"] == cfg.cell_id
    with pytest.raises(ConfigError):
        cfg.run_context(99)                 # not in committed seeds


# --------------------------------------------------------------------------- #
# Freeze discipline: confirmatory may not inherit development placeholders
# --------------------------------------------------------------------------- #
def test_confirmatory_config_must_pin_every_value():
    d = _base_dev_ddqn()
    d.update(role="confirmatory")
    d["backbone"].pop("lr")                 # leave a backbone number unset
    d["factor_specific"]["eps_schedule"] = None
    with pytest.raises(ConfigError) as e:
        resolve_config(d)
    msg = str(e.value)
    assert "backbone.lr" in msg and "eps_schedule" in msg


def test_confirmatory_config_with_all_values_is_accepted():
    d = _base_dev_ddqn()
    d.update(role="confirmatory", run_id="t_ddqn_conf")
    # every value the guard checks for the ε-greedy baseline is present; size_class defaults
    # to 'confirmatory' from the role.
    cfg = resolve_config(d)
    assert cfg.size_class == "confirmatory"
    # setting it explicitly is equivalent
    d["size_class"] = "confirmatory"
    assert resolve_config(d).size_class == "confirmatory"


def test_confirmatory_ensemble_requires_class2_params():
    d = _base_dev_ddqn()
    d.update(role="confirmatory", method="bdqn", K=10, arm="episodic|off|K10")
    d["factor_specific"]["eps_schedule"] = None   # not required for bdqn
    # ensemble_shared missing entirely -> mask_prob / head_loss_agg unset
    with pytest.raises(ConfigError) as e:
        resolve_config(d)
    assert "ensemble_shared.mask_prob" in str(e.value)


# --------------------------------------------------------------------------- #
# C13 identity stability + serialization
# --------------------------------------------------------------------------- #
def test_config_hash_is_stable_and_order_independent():
    d = _base_dev_ddqn()
    h1 = resolve_config(d).config_sha256
    d2 = copy.deepcopy(d)
    d2["backbone"] = {"gamma": 0.99, "batch_size": 32, "lr": 5e-4, "hidden_sizes": [32, 32]}
    assert resolve_config(d2).config_sha256 == h1     # key order does not matter

    d3 = copy.deepcopy(d)
    d3["backbone"]["lr"] = 1e-4
    assert resolve_config(d3).config_sha256 != h1     # a real change moves the hash


def test_write_resolved_round_trip(tmp_path):
    cfg = resolve_config(_base_dev_ddqn())
    path = cfg.write_resolved(tmp_path)
    assert path.name == "resolved_config.json"
    import json

    payload = json.loads(path.read_text())
    assert payload["_config_sha256"] == cfg.config_sha256
    assert payload["cell_id"] == cfg.cell_id


# --------------------------------------------------------------------------- #
# Not-yet-implemented methods/envs fail loudly, not silently
# --------------------------------------------------------------------------- #
def _base_dev_bdqn() -> dict:
    """A runnable Bootstrapped-DQN (prior=off) development cell."""
    return {
        "run_id": "t_bdqn_dev",
        "role": "development",
        "part": "A",
        "method": "bdqn",
        "env": "deep_sea",
        "master_seed": 0,
        "use_rule": "episodic",
        "prior": "off",
        "K": 5,
        "arm": "episodic|off|K5",
        "backbone": {"lr": 5e-4, "batch_size": 16, "gamma": 0.99, "hidden_sizes": [16]},
        "ensemble_shared": {"mask_prob": 0.5, "head_loss_agg": "grad_norm_1_over_k"},
        "factor_specific": {"prior_scale": None, "eps_schedule": None},
        "env_budget": {"deep_sea_size": 5, "episodes": 50},
        "seeds": [0, 1, 2],
    }


def test_unimplemented_method_raises_not_implemented():
    d = _base_dev_ddqn()
    d.update(method="noisynet")  # recognized in the vocabulary, no factory branch yet
    cfg = resolve_config(d)
    with pytest.raises(NotImplementedError):
        build_agent(cfg, 0)


def test_rp_bdqn_prior_on_is_not_yet_implemented():
    # method 'bdqn' builds the ensemble for prior=off only; prior=on is RP-BDQN.
    d = _base_dev_bdqn()
    d.update(prior="on", arm="episodic|on|K5",
             factor_specific={"prior_scale": 3.0, "eps_schedule": None})
    cfg = resolve_config(d)
    with pytest.raises(ConfigError, match="RP-BDQN"):
        build_agent(cfg, 0)


def test_unimplemented_env_raises_not_implemented():
    d = _base_dev_ddqn()
    d.update(env="breakout", part="B")
    cfg = resolve_config(d)
    with pytest.raises(NotImplementedError):
        build_env(cfg, 0)


# --------------------------------------------------------------------------- #
# Factory round-trip: a whole run executes from a config
# --------------------------------------------------------------------------- #
def test_factories_build_and_step_a_ddqn_run_from_config():
    pytest.importorskip("torch")
    cfg = resolve_config(_base_dev_ddqn())
    env = build_env(cfg, seed_index=0)
    agent = build_agent(cfg, seed_index=0)

    obs, _ = env.reset()
    obs = obs.ravel().astype("float32")
    step, terminated = 0, False
    while not terminated:
        a = agent.select_action(obs, step=step)
        nobs, r, terminated, _, _ = env.step(a)
        nobs = nobs.ravel().astype("float32")
        agent.observe(obs, a, r, nobs, terminated)
        agent.learn_step()
        obs = nobs
        step += 1
    assert step == cfg.data["env_budget"]["deep_sea_size"]   # one row per step


def test_factories_build_and_step_a_bdqn_ensemble_from_config():
    pytest.importorskip("torch")
    cfg = resolve_config(_base_dev_bdqn())
    env = build_env(cfg, seed_index=0)
    agent = build_agent(cfg, seed_index=0)
    assert type(agent).__name__ == "BDQNAgent"
    assert agent.K == 5  # 10-head would be the default; the config wins

    obs, _ = env.reset()
    obs = obs.ravel().astype("float32")
    agent.on_episode_start()
    step, terminated = 0, False
    while not terminated:
        a = agent.select_action(obs, step=step)
        nobs, r, terminated, _, _ = env.step(a)
        nobs = nobs.ravel().astype("float32")
        agent.observe(obs, a, r, nobs, terminated)
        agent.learn_step()
        obs = nobs
        step += 1
    assert step == cfg.data["env_budget"]["deep_sea_size"]


def test_factory_run_is_reproducible_and_cell_separated():
    pytest.importorskip("torch")

    def one_episode(cfg, seed_index):
        env = build_env(cfg, seed_index)
        agent = build_agent(cfg, seed_index)
        obs, _ = env.reset()
        obs = obs.ravel().astype("float32")
        actions, terminated = [], False
        while not terminated:
            a = agent.select_action(obs, step=0)      # step 0 -> eps_start, deterministic draw
            nobs, _, terminated, _, _ = env.step(a)
            actions.append(a)
            obs = nobs.ravel().astype("float32")
        return actions

    cfg = resolve_config(_base_dev_ddqn())
    assert one_episode(cfg, 0) == one_episode(cfg, 0)         # same seed+cell -> identical

    d = _base_dev_ddqn()
    d.update(use_rule="per_step", arm="per_step|off|K1")
    cfg2 = resolve_config(d)
    # different cell_id -> different env mapping and init; behavior should differ somewhere
    assert (one_episode(cfg, 0), cfg.cell_id) != (one_episode(cfg2, 0), cfg2.cell_id)


def test_run_context_feeds_csv_logger(tmp_path):
    cfg = resolve_config(_base_dev_ddqn())
    ctx = cfg.run_context(0)
    csv_path = tmp_path / "run.csv"
    with conventions.CSVLogger(csv_path, ctx) as log:
        log.log(step=100, metric="episode_return", value=0.0)
    text = csv_path.read_text()
    assert cfg.config_sha256 in text and "episode_return" in text
