"""MinAtar (Part B) config factories: every method x game builds, step budgets validate."""

from __future__ import annotations

import pytest

from src.config import (
    IMPLEMENTED_ENVS,
    ConfigError,
    build_agent,
    build_env,
    is_minatar,
    resolve_config,
    step_budget,
)
from src.minatar_env import MINATAR_CHANNELS, MINATAR_GAMES

METHODS = ("ddqn_egreedy", "noisynet", "bdqn", "rp_bdqn")


def _base(method: str = "ddqn_egreedy", env: str = "breakout") -> dict:
    """A development Part-B config, step-budgeted."""
    prior = "on" if method == "rp_bdqn" else "off"
    k = 10 if method in ("bdqn", "rp_bdqn") else 1
    factor: dict = {"prior_scale": 3.0 if prior == "on" else None}
    if method == "ddqn_egreedy":
        factor["eps_schedule"] = {"eps_start": 1.0, "eps_end": 0.05, "eps_decay_steps": 2000}
    elif method == "noisynet":
        factor["sigma0"] = 0.5
    return {
        "run_id": f"t_{method}_{env}_dev",
        "role": "development",
        "part": "B",
        "method": method,
        "env": env,
        "master_seed": 0,
        "use_rule": "episodic",
        "prior": prior,
        "K": k,
        # NoisyNet is not a switchboard cell: it carries a method-named cell_id so its RNG
        # streams do not collide with the DDQN reference cell (episodic|off|K1).
        "arm": "noisynet" if method == "noisynet" else f"episodic|{prior}|K{k}",
        "backbone": {"lr": 5e-4, "batch_size": 32, "gamma": 0.99, "hidden_sizes": [128]},
        "factor_specific": factor,
        "env_budget": {"total_steps": 5_000, "checkpoint_steps": [1_000, 5_000]},
        "seeds": [0, 1, 2],
    }


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
def test_all_minatar_games_are_implemented_envs() -> None:
    for game in MINATAR_GAMES:
        assert game in IMPLEMENTED_ENVS
    assert "deep_sea" in IMPLEMENTED_ENVS  # Part A unaffected


def test_is_minatar_discriminates_env_family() -> None:
    assert all(is_minatar(g) for g in MINATAR_GAMES)
    assert not is_minatar("deep_sea")


# --------------------------------------------------------------------------- #
# Factories: every method x game
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("game", MINATAR_GAMES)
def test_every_method_and_game_builds(method: str, game: str) -> None:
    pytest.importorskip("torch")
    cfg = resolve_config(_base(method, game))
    env = build_env(cfg, seed_index=0)
    agent = build_agent(cfg, seed_index=0)
    assert env.obs_shape == (MINATAR_CHANNELS[game], 10, 10)
    assert agent.cfg.obs_shape == env.obs_shape
    assert agent.cfg.n_actions == env.n_actions == 6


@pytest.mark.parametrize("method", METHODS)
def test_built_agent_acts_and_learns_on_minatar(method: str) -> None:
    """A few real steps through the conv path: select_action -> observe -> learn_step."""
    pytest.importorskip("torch")
    d = _base(method)
    d["backbone"]["batch_size"] = 4
    cfg = resolve_config(d)
    env = build_env(cfg, seed_index=0)
    agent = build_agent(cfg, seed_index=0)
    agent.cfg.min_buffer = 8

    obs, _ = env.reset()
    for step in range(40):
        action = agent.select_action(obs, step)
        assert 0 <= action < 6
        next_obs, reward, terminated, _, _ = env.step(action)
        agent.observe(obs, action, reward, next_obs, terminated)
        obs = next_obs if not terminated else env.reset()[0]
    loss = agent.learn_step()
    assert loss is None or loss >= 0.0


def test_agent_network_is_the_conv_backbone() -> None:
    pytest.importorskip("torch")
    import torch

    cfg = resolve_config(_base("bdqn"))
    agent = build_agent(cfg, seed_index=0)
    assert isinstance(agent.online.conv, torch.nn.Conv2d)
    assert agent.online.n_heads == 10
    assert agent.online.feature_dim == 128  # hidden_sizes[-1] read as conv FC width


def test_deepsea_agent_still_uses_the_mlp_backbone() -> None:
    """Part A must be untouched: obs_shape=None keeps the flat MLP path."""
    pytest.importorskip("torch")
    from src.networks import MLPQNetwork

    d = _base()
    d.update(part="A", env="deep_sea", env_budget={"deep_sea_size": 5, "episodes": 100})
    d["backbone"]["hidden_sizes"] = [32, 32]
    cfg = resolve_config(d)
    agent = build_agent(cfg, seed_index=0)
    assert isinstance(agent.online, MLPQNetwork)
    assert agent.cfg.obs_shape is None


# --------------------------------------------------------------------------- #
# Step budget schema
# --------------------------------------------------------------------------- #
def test_step_budget_reads_total_and_checkpoints() -> None:
    cfg = resolve_config(_base())
    total, ckpts = step_budget(cfg)
    assert total == 5_000
    assert ckpts == (1_000, 5_000)


def test_variant_b_checkpoint_grid_validates() -> None:
    """The pre-registered 100k/500k/1M grid must pass validation as-is."""
    d = _base()
    d["env_budget"] = {"total_steps": 1_000_000, "checkpoint_steps": [100_000, 500_000, 1_000_000]}
    total, ckpts = step_budget(resolve_config(d))
    assert total == 1_000_000
    assert ckpts == (100_000, 500_000, 1_000_000)


@pytest.mark.parametrize(
    "budget,match",
    [
        ({"checkpoint_steps": [100]}, "total_steps"),
        ({"total_steps": 0, "checkpoint_steps": [100]}, "total_steps"),
        ({"total_steps": 1000}, "checkpoint_steps"),
        ({"total_steps": 1000, "checkpoint_steps": []}, "checkpoint_steps"),
        ({"total_steps": 1000, "checkpoint_steps": [500, 100]}, "sorted"),
        ({"total_steps": 1000, "checkpoint_steps": [100, 100]}, "sorted"),
        ({"total_steps": 1000, "checkpoint_steps": [100, 5000]}, "exceeds"),
        ({"total_steps": 1000, "checkpoint_steps": [0]}, "positive"),
    ],
)
def test_bad_step_budgets_are_rejected(budget: dict, match: str) -> None:
    d = _base()
    d["env_budget"] = budget
    with pytest.raises(ConfigError, match=match):
        step_budget(resolve_config(d))


def test_episode_budget_config_is_rejected_for_a_step_budget_read() -> None:
    """Supplying the DeepSea (episode) budget shape to a MinAtar run fails loudly."""
    d = _base()
    d["env_budget"] = {"deep_sea_size": 5, "episodes": 100}
    with pytest.raises(ConfigError, match="total_steps"):
        step_budget(resolve_config(d))


# --------------------------------------------------------------------------- #
# Determinism through the factories
# --------------------------------------------------------------------------- #
def test_same_seed_index_builds_identical_agents() -> None:
    pytest.importorskip("torch")
    import torch

    cfg = resolve_config(_base("bdqn"))
    a = build_agent(cfg, seed_index=0)
    b = build_agent(cfg, seed_index=0)
    for pa, pb in zip(a.online.parameters(), b.online.parameters(), strict=True):
        assert torch.equal(pa, pb)


def test_rp_bdqn_requires_prior_on() -> None:
    d = _base("rp_bdqn")
    d["prior"] = "off"
    d["factor_specific"]["prior_scale"] = None
    with pytest.raises(ConfigError, match="prior"):
        build_agent(resolve_config(d), 0)


def test_rp_bdqn_builds_a_prior_network_on_minatar() -> None:
    pytest.importorskip("torch")
    plain = build_agent(resolve_config(_base("bdqn")), 0)
    rp = build_agent(resolve_config(_base("rp_bdqn")), 0)
    assert plain.prior is None
    assert rp.prior is not None
    assert rp.prior.n_heads == rp.online.n_heads


# --------------------------------------------------------------------------- #
# RNG-stream separation of the two baselines (regression: the guard used to be
# scoped to Part A, letting the Part-B NoisyNet config share the DDQN arm)
# --------------------------------------------------------------------------- #
def test_part_b_noisynet_rejects_switchboard_style_arm() -> None:
    """The method-named-cell_id rule is part-independent: NoisyNet is one of the canonical
    four on MinAtar too, and the streams key on cell_id alone."""
    d = _base("noisynet")
    d["arm"] = "episodic|off|K1"  # the DDQN reference cell
    with pytest.raises(ConfigError, match="NoisyNet config must set an explicit 'arm'"):
        resolve_config(d)


def test_part_b_noisynet_cell_id_is_distinct_from_the_ddqn_baseline() -> None:
    from src.utils.conventions import derive_seed

    noisy = resolve_config(_base("noisynet"))
    ddqn = resolve_config(_base("ddqn_egreedy"))
    assert noisy.cell_id != ddqn.cell_id
    for stream in ("init", "replay", "action_noise"):
        a = derive_seed(noisy.data["master_seed"], noisy.cell_id, stream, 0)
        b = derive_seed(ddqn.data["master_seed"], ddqn.cell_id, stream, 0)
        assert a != b, f"{stream} seed collides between the two Part-B baselines"


def test_every_committed_config_resolves() -> None:
    """Spec §4 CI clause: the committed config set must satisfy the frozen schema."""
    from pathlib import Path

    from audits.config_schema_check import check_dir

    ok, failed = check_dir(Path("configs"))
    assert not failed, f"configs failed to resolve: {failed}"
    assert len(ok) >= 8


def test_rp_bdqn_configs_declare_the_rp_bdqn_method() -> None:
    """Both RP-BDQN example configs should resolve to the same method name, not one to its
    ``bdqn`` alias — the C13 identity audit compares method names across contrast pairs."""
    import yaml

    for name in ("example_rpbdqn_deepsea_dev.yaml", "example_rpbdqn_breakout_dev.yaml"):
        d = yaml.safe_load(open(f"configs/{name}"))
        assert d["method"] == "rp_bdqn", name
