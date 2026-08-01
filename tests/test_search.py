"""Tests for the pre-registered search driver (``src/search.py``).

Three of these are load-bearing and were each verified by mutation before being trusted:

* :func:`test_tuning_streams_are_disjoint_from_the_reference_cell` — the search must not be
  scored on the environment instances the selected backbone is later evaluated on.
* :func:`test_candidates_share_random_numbers` — the contrast between two candidates must
  come from their hyperparameters, not from which DeepSea mapping they drew.
* :func:`test_discovery_auc_is_strictly_decreasing_in_discovery_time` — the property that
  makes the Gap 4 objective subsume episodes-to-first-discovery.

The rest pin the frozen quantities (draw count, run budget, tie-break order) so a later
edit that changes the search cannot pass silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import config as config_mod
from src import search, selection
from src.utils.conventions import derive_seed

TEMPLATE = "configs/example_ddqn_deepsea_dev.yaml"


@pytest.fixture
def template():
    return config_mod.load_config(TEMPLATE)


# --------------------------------------------------------------------------- #
# Frozen quantities
# --------------------------------------------------------------------------- #


def test_frozen_search_budget():
    """freeze item 2 via Gap 2/3: 12 candidates x 3 seeds x 2 sizes = 72 runs."""
    assert search.N_BACKBONE == 12
    assert search.TUNING_SEEDS == (0, 1, 2)
    assert search.DEV_SIZES == (10, 20)
    assert search.N_BACKBONE * len(search.TUNING_SEEDS) * len(search.DEV_SIZES) == 72


def test_draw_order_covers_the_space_exactly():
    """A parameter in the space but not in DRAW_ORDER would never be drawn — and a name in
    DRAW_ORDER but not the space would raise only at draw time."""
    assert set(search.DRAW_ORDER) == set(search.BACKBONE_SPACE)
    assert len(search.DRAW_ORDER) == len(set(search.DRAW_ORDER))


def test_search_space_matches_the_staged_protocol_text():
    """The distributions are frozen in a *document*; the code is a transcription of it.

    Nothing else in the tree would catch a divergence — a config audit sees only the values
    a run used, never the distribution it was drawn from — so if someone widens the lr range
    in ``BACKBONE_SPACE`` without amending Gap 2, the search silently stops being the
    pre-registered one. This asserts the literal table entries are still present.
    """
    doc = Path("protocol/decisions/staged_stage3_protocol_fixes.md").read_text()
    for fragment in (
        "`n_backbone = 12` draws",
        "3 seeds on each of the two development sizes",
        "log-uniform `[1e-4, 1e-2]`",
        "uniform over `{32, 64, 128}`",
        "uniform over `{100, 500, 1000}`",
        "uniform over `{64, 128, 256}`",
        "fixed at 100,000",
    ):
        assert fragment in doc, (
            f"Gap 2 no longer states {fragment!r}; src/search.py may have drifted"
        )


def test_fixed_params_are_not_searched():
    """Gap 2 fixes replay capacity and optimizer; drawing them would be a protocol change."""
    assert not set(search.BACKBONE_FIXED) & set(search.BACKBONE_SPACE)


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def test_sampling_is_deterministic_in_master_seed(template):
    a = search.sample_backbone_points(template.master_seed)
    b = search.sample_backbone_points(template.master_seed)
    assert a == b
    assert search.sample_backbone_points(template.master_seed + 1) != a


def test_extending_n_extends_the_field(template):
    """Drawing one full parameter vector at a time means a larger n appends rather than
    reshuffles — so ``--candidates 24`` is a superset audit of the frozen 12, not a
    different search."""
    twelve = search.sample_backbone_points(template.master_seed, n=12)
    twentyfour = search.sample_backbone_points(template.master_seed, n=24)
    assert twentyfour[:12] == twelve


def test_samples_respect_the_frozen_support(template):
    for point in search.sample_backbone_points(template.master_seed):
        lr_spec = search.BACKBONE_SPACE["lr"]
        assert lr_spec["low"] <= point["lr"] <= lr_spec["high"]
        for name in ("batch_size", "target_update_period", "hidden_width"):
            assert point[name] in search.BACKBONE_SPACE[name]["values"]
            assert isinstance(point[name], int)


def test_unknown_distribution_kind_raises(template):
    with pytest.raises(ValueError, match="unknown distribution kind"):
        search.sample_backbone_points(
            template.master_seed, n=1, space={"lr": {"kind": "gaussian"}}
        )


# --------------------------------------------------------------------------- #
# RNG hygiene — the two load-bearing facts
# --------------------------------------------------------------------------- #


def test_tuning_streams_are_disjoint_from_the_reference_cell(template):
    """LOAD-BEARING. Tuning must not reuse the reference cell's streams.

    ``derive_seed`` keys on ``(master_seed, cell_id, stream, seed_index)``. If a tuning run
    carried ``cell_id='episodic|off|K1'``, tuning seed 0 and the reference cell's
    *evaluation* seed 0 would get byte-identical init / env_mapping / replay /
    action_noise draws — selection on the very instances later used for measurement.
    """
    point = search.sample_backbone_points(template.master_seed)[0]
    cfg = search.candidate_config(template, point, 10, index=0)
    assert cfg.cell_id != template.cell_id
    for stream in ("init", "env_mapping", "replay", "action_noise"):
        for seed_index in (0, 1, 2):
            assert derive_seed(cfg.master_seed, cfg.cell_id, stream, seed_index) != derive_seed(
                template.master_seed, template.cell_id, stream, seed_index
            )


def test_candidates_share_random_numbers(template):
    """LOAD-BEARING. Common random numbers ACROSS candidates.

    Candidates must differ by hyperparameters, not by which environment instance they drew,
    so the cell_id must not encode the candidate index or the size.
    """
    points = search.sample_backbone_points(template.master_seed)
    a = search.candidate_config(template, points[0], 10, index=0)
    b = search.candidate_config(template, points[1], 10, index=1)
    c = search.candidate_config(template, points[0], 20, index=0)
    assert a.cell_id == b.cell_id == c.cell_id
    for stream in ("init", "env_mapping"):
        assert derive_seed(a.master_seed, a.cell_id, stream, 0) == derive_seed(
            b.master_seed, b.cell_id, stream, 0
        )


def test_candidate_runs_are_separable_despite_shared_streams(template):
    """Shared RNG must not mean shared logs: run_id and the config fingerprint still differ,
    so the 72 runs cannot overwrite one another."""
    points = search.sample_backbone_points(template.master_seed)
    a = search.candidate_config(template, points[0], 10, index=0)
    b = search.candidate_config(template, points[1], 10, index=1)
    c = search.candidate_config(template, points[0], 20, index=0)
    assert len({a.run_id, b.run_id, c.run_id}) == 3
    assert len({a.config_sha256, b.config_sha256, c.config_sha256}) == 3


# --------------------------------------------------------------------------- #
# Config materialization
# --------------------------------------------------------------------------- #


def test_every_drawn_parameter_reaches_the_agent(template):
    """A parameter that is drawn but silently dropped by the config layer would make the
    search a no-op along that axis — the failure mode most likely to go unnoticed."""
    point = search.sample_backbone_points(template.master_seed)[0]
    cfg = search.candidate_config(template, point, 10, index=0)
    agent = config_mod.build_agent(cfg, 0)
    assert agent.cfg.lr == pytest.approx(point["lr"])
    assert agent.cfg.batch_size == point["batch_size"]
    assert agent.cfg.target_update_period == point["target_update_period"]
    assert tuple(agent.cfg.hidden_sizes) == (point["hidden_width"],) * 2
    assert agent.cfg.buffer_capacity == search.BACKBONE_FIXED["buffer_capacity"]


def test_tuning_arm_requires_exploratory_role(template):
    """The tuning namespace is an exemption from the factorial identity check; confining it
    to role='exploratory' keeps a reported config from opting out by renaming its arm."""
    cfg = search.candidate_config(template, search.sample_backbone_points(0)[0], 10, index=0)
    data = dict(cfg.data)
    data["role"] = "development"
    data.pop("cell_id")
    data.pop("size_class")
    with pytest.raises(config_mod.ConfigError, match="requires role: exploratory"):
        config_mod.resolve_config(data)


def test_committed_cells_still_require_a_matching_arm():
    """Negative control on the new branch: it must not have loosened the factorial check
    for anything that is not a tuning arm."""
    data = dict(config_mod.load_config("configs/cell_episodic_off_K5_deepsea_dev.yaml").data)
    data["arm"] = "episodic|off|K7"
    data.pop("cell_id")
    data.pop("size_class")
    with pytest.raises(config_mod.ConfigError, match="does not match canonical cell_id"):
        config_mod.resolve_config(data)


# --------------------------------------------------------------------------- #
# The objective
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_ck", [5, 20])
def test_discovery_auc_is_strictly_decreasing_in_discovery_time(n_ck):
    """LOAD-BEARING. AUC = 1 - j/C for discovery at checkpoint j, 0 if never.

    This exact identity is why the Gap 4 objective carries the time-to-discovery
    information without a censoring convention.
    """
    aucs = []
    for j in range(n_ck):
        curve = [0.0] * j + [1.0] * (n_ck - j)
        auc = search.discovery_auc(curve)
        assert auc == pytest.approx(1 - j / n_ck)
        aucs.append(auc)
    assert aucs == sorted(aucs, reverse=True)
    assert len(set(aucs)) == n_ck  # strictly decreasing, no collapsed values
    assert search.discovery_auc([0.0] * n_ck) == 0.0


def test_discovery_auc_beats_the_terminal_indicator_on_resolution():
    """The terminal indicator cannot distinguish two runs that both discovered; AUC can.
    That resolution is the whole reason for the Gap 4 change."""
    early = [0.0] + [1.0] * 19
    late = [0.0] * 18 + [1.0] * 2
    assert early[-1] == late[-1] == 1.0
    assert search.discovery_auc(early) > search.discovery_auc(late)


def test_discovery_auc_rejects_an_empty_curve():
    with pytest.raises(ValueError, match="empty"):
        search.discovery_auc([])


def test_score_from_rows_reads_the_committed_schema():
    def row(metric, axis, ck, value, seed, size):
        return {
            "metric": metric, "axis": axis, "checkpoint": ck,
            "value": value, "seed": seed, "size": size,
        }

    rows = [
        row("discovery_prob", "online", 2, 1.0, 0, 10),
        row("discovery_prob", "online", 1, 0.0, 0, 10),
        row("episode_return", "online", 1, 9.0, 0, 10),   # wrong metric: ignored
        row("discovery_prob", "frozen", 1, 1.0, 0, 10),   # wrong axis: ignored
        row("discovery_prob", "online", 1, 1.0, 1, 20),
    ]
    scored = search.score_from_rows(rows)
    assert scored == {(10, 0): pytest.approx(0.5), (20, 1): pytest.approx(1.0)}


def test_score_from_rows_sorts_by_checkpoint():
    """Rows arriving out of order must not change the curve; the mean is order-invariant
    but the sort protects the identity the AUC test relies on."""
    shuffled = [
        {
            "metric": "discovery_prob", "axis": "online",
            "checkpoint": c, "value": v, "seed": 0, "size": 10,
        }
        for c, v in [(4, 1.0), (1, 0.0), (3, 1.0), (2, 0.0)]
    ]
    assert search.score_from_rows(shuffled)[(10, 0)] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Tie-breaking
# --------------------------------------------------------------------------- #


def test_sort_key_is_lexicographic_in_draw_order():
    lo = {"lr": 1e-4, "batch_size": 128, "target_update_period": 1000, "hidden_width": 256}
    hi = {"lr": 1e-3, "batch_size": 32, "target_update_period": 100, "hidden_width": 64}
    assert search._sort_key(lo) < search._sort_key(hi)  # lr dominates, per DRAW_ORDER


def test_tie_is_broken_toward_the_lower_parameter_value():
    """freeze item 3 applied through selection.select_best on this module's sort key."""
    tied = [
        selection.Candidate(
            label=f"c{i}",
            params=p,
            scores=(1.0, 1.0, 1.0, 1.0),
            sort_key=search._sort_key(p),
        )
        for i, p in enumerate(
            [
                {"lr": 1e-3, "batch_size": 32, "target_update_period": 100, "hidden_width": 64},
                {"lr": 1e-4, "batch_size": 128, "target_update_period": 1000, "hidden_width": 256},
            ]
        )
    ]
    result = selection.select_best(tied)
    assert result.tie_broken
    assert result.winner.label == "c1"
    assert result.winner.params["lr"] == 1e-4


def test_draw_index_never_enters_the_tie_break():
    """The bookkeeping key must not become a de-facto tie-breaker on draw order."""
    assert search._INDEX_KEY not in search.DRAW_ORDER
    point = {"lr": 1e-3, "batch_size": 32, "target_update_period": 100, "hidden_width": 64}
    assert search._sort_key(dict(point, **{search._INDEX_KEY: 7})) == search._sort_key(point)


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #


@pytest.fixture
def tiny_template(template):
    data = dict(template.data)
    data["env_budget"] = {"deep_sea_size": 5, "episodes": 40}
    data.pop("cell_id")
    data.pop("size_class")
    return config_mod.resolve_config(data)


def test_end_to_end_search_writes_an_auditable_record(tiny_template, tmp_path):
    best, record = search.run_backbone_search(
        tiny_template,
        out_dir=tmp_path,
        n=3,
        seeds=(0, 1),
        sizes=(5, 6),
        n_checkpoints=5,
        episodes=40,
    )
    assert len(record.per_candidate) == 3
    for entry in record.per_candidate:
        assert len(entry["scores"]) == 4  # 2 seeds x 2 sizes
        assert entry["iqm"] == pytest.approx(selection.iqm(entry["scores"]))

    payload = json.loads(record.write(tmp_path / "rec.json").read_text())
    assert payload["total_runs"] == 12
    assert payload["runs_per_candidate"] == 4
    assert payload["objective"] == "iqm_of_per_seed_discovery_auc"
    assert payload["winner"] == record.points[payload["winner_index"]]
    # The record's winner and the SelectionResult's winner must be the same candidate.
    assert record.winner_index == int(best.winner.params[search._INDEX_KEY])
    assert best.winner.label == f"c{record.winner_index:02d}"

    # The winner really is the argmax of the recorded IQMs (tie-break aside).
    assert record.per_candidate[record.winner_index]["iqm"] == pytest.approx(
        max(e["iqm"] for e in record.per_candidate)
    )


def test_search_is_reproducible(tiny_template, tmp_path):
    """The whole search is a function of master_seed: same seed, same winner and scores."""
    a, ra = search.run_backbone_search(
        tiny_template,
        out_dir=tmp_path / "a",
        n=2,
        seeds=(0,),
        sizes=(5,),
        n_checkpoints=4,
        episodes=40,
    )
    b, rb = search.run_backbone_search(
        tiny_template,
        out_dir=tmp_path / "b",
        n=2,
        seeds=(0,),
        sizes=(5,),
        n_checkpoints=4,
        episodes=40,
    )
    assert a.winner.label == b.winner.label
    assert [e["scores"] for e in ra.per_candidate] == [e["scores"] for e in rb.per_candidate]


def test_read_rows_rejects_a_foreign_csv(tmp_path):
    """Guard against silently scoring a CSV that has no size in its run_id."""
    p = tmp_path / "x.csv"
    p.write_text("run_id,metric,axis,checkpoint,value,seed\nsomecell,discovery_prob,online,1,1.0,0\n")
    with pytest.raises(ValueError, match="no _N<size> suffix"):
        search._read_rows(p)


def test_cli_dry_run_prints_the_field(capsys):
    assert search.main(["--dry-run", "--candidates", "4", "--template", TEMPLATE]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("c")]
    assert len(lines) == 4
    assert all(name in lines[0] for name in search.DRAW_ORDER)


def test_search_module_does_not_import_torch_at_module_scope():
    """``--dry-run`` and the audit path must work without a torch import; the trainer and torch
    imports are deliberately function-local.

    Checked on the AST rather than by splitting the source text at a function name: the text
    heuristic silently stopped covering anything defined above that function, which is where
    the parallel worker (and its function-local ``import torch``) landed."""
    import ast

    tree = ast.parse(Path("src/search.py").read_text())
    module_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = set()
    for node in module_level:
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        else:
            names.add(node.module or "")
    assert "torch" not in names
    assert "src.trainer" not in names
    assert not any(n.startswith("torch") for n in names), sorted(names)


# --------------------------------------------------------------------------- #
# Class-3 mini-searches (steps 7 and 8)
# --------------------------------------------------------------------------- #


def test_mini_search_budget_and_shape():
    """Gap 2: ``n_mini = 4`` draws each, one varied parameter each, two searches only."""
    assert search.N_MINI == 4
    assert set(search.MINI_SEARCHES) == {"prior_scale", "eps_schedule"}
    for spec in search.MINI_SEARCHES.values():
        assert spec["space"]["kind"] == "log_uniform"


def test_mini_space_matches_the_staged_protocol_text():
    """Same drift guard as the backbone: the class-3 distributions live in a document."""
    doc = Path("protocol/decisions/staged_stage3_protocol_fixes.md").read_text()
    for fragment in (
        "`n_mini = 4` draws each",
        "log-uniform\n> `[0.1, 10.0]`",
        "final-ε log-uniform `[0.005, 0.1]`",
        "linear decay over the first 10%",
        "Ties broken by the lower parameter value.",
    ):
        assert fragment in doc, (
            f"Gap 2 no longer states {fragment!r}; src/search.py's class-3 block may have drifted"
        )
    assert search.MINI_SEARCHES["prior_scale"]["space"] == {
        "kind": "log_uniform",
        "low": 0.1,
        "high": 10.0,
    }
    assert search.MINI_SEARCHES["eps_schedule"]["space"] == {
        "kind": "log_uniform",
        "low": 0.005,
        "high": 0.1,
    }
    assert search.EPS_DECAY_BUDGET_FRACTION == 0.10


def test_mini_samples_respect_the_frozen_support():
    for which, spec in search.MINI_SEARCHES.items():
        param, dist = spec["param"], spec["space"]
        for point in search.sample_mini_points(0, which, n=32):
            assert dist["low"] <= point[param] <= dist["high"]
            assert set(point) == {param}


def test_the_three_searches_are_independent_draws():
    """Each search keys its own ``hparam_search`` stream, so changing ``n`` for one cannot
    shift another's field. Compared as raw floats: a shared stream would make these equal."""
    a = [p["prior_scale"] for p in search.sample_mini_points(0, "prior_scale", n=8)]
    b = [p["eps_end"] for p in search.sample_mini_points(0, "eps_schedule", n=8)]
    c = [p["lr"] for p in search.sample_backbone_points(0, n=8)]
    assert a != b and a != c and b != c


def test_mini_tuning_streams_are_disjoint_from_their_input_cell():
    """RNG fact 1, for the mini-searches: a candidate must not run under the evaluation arm
    of the cell whose IQM selects it, or the parameter is chosen on the instances it is
    later measured on."""
    for which, spec in search.MINI_SEARCHES.items():
        tmpl = config_mod.load_config(spec["template"])
        cfg = search.mini_candidate_config(
            tmpl,
            search.sample_mini_points(tmpl.master_seed, which)[0],
            10,
            which=which,
            index=0,
        )
        assert cfg.cell_id == f"tune|{which}"
        assert cfg.cell_id != spec["arm"]
        assert cfg.role == "exploratory"
        for stream in ("init", "env_mapping", "replay", "action_noise"):
            assert derive_seed(tmpl.master_seed, cfg.cell_id, stream, 0) != derive_seed(
                tmpl.master_seed, spec["arm"], stream, 0
            )


def test_mini_candidates_share_random_numbers_but_stay_separable():
    """Common random numbers across the 4 candidates (so the contrast is the parameter),
    while ``run_id`` still separates their logs."""
    for which, spec in search.MINI_SEARCHES.items():
        tmpl = config_mod.load_config(spec["template"])
        pts = search.sample_mini_points(tmpl.master_seed, which)
        cfgs = [
            search.mini_candidate_config(tmpl, p, 10, which=which, index=i)
            for i, p in enumerate(pts)
        ]
        assert len({c.cell_id for c in cfgs}) == 1
        assert len({c.run_id for c in cfgs}) == len(pts)
        assert len({c.config_sha256 for c in cfgs}) == len(pts)


def test_mini_cell_id_encodes_neither_index_nor_size():
    which = "prior_scale"
    spec = search.MINI_SEARCHES[which]
    tmpl = config_mod.load_config(spec["template"])
    pts = search.sample_mini_points(tmpl.master_seed, which)
    ids = {
        search.mini_candidate_config(tmpl, pts[i], size, which=which, index=i).cell_id
        for i in range(len(pts))
        for size in search.DEV_SIZES
    }
    assert ids == {f"tune|{which}"}


def test_eps_decay_is_ten_percent_of_each_size_budget():
    """The decay length is *derived*, not drawn: 10% of that run's own step budget. DeepSea
    episodes are exactly ``size`` steps, so the budget is ``episodes × size`` exactly."""
    which = "eps_schedule"
    tmpl = config_mod.load_config(search.MINI_SEARCHES[which]["template"])
    point = search.sample_mini_points(tmpl.master_seed, which)[0]
    for size in search.DEV_SIZES:
        cfg = search.mini_candidate_config(tmpl, point, size, which=which, index=0)
        episodes = cfg.data["env_budget"]["episodes"]
        sched = cfg.data["factor_specific"]["eps_schedule"]
        assert sched["eps_decay_steps"] == round(0.10 * episodes * size)
        assert sched["eps_end"] == point["eps_end"]
        assert sched["eps_start"] == 1.0


def test_deep_sea_episode_length_is_exactly_size():
    """The identity the derived decay depends on. If DeepSea ever gains truncation or a
    variable-length episode, ``episodes × size`` stops being the step budget and the ε
    schedule silently changes meaning."""
    from src.deep_sea import DeepSea

    for size in search.DEV_SIZES:
        env = DeepSea(size=size, master_seed=0, cell_id="tune|eps_schedule", seed_index=0)
        env.reset()
        steps, done = 0, False
        while not done:
            _, _, term, trunc, _ = env.step(steps % 2)
            steps += 1
            done = term or trunc
        assert steps == size


def test_mini_search_configs_build_real_agents():
    """End-to-end: the drawn value must reach the agent, not just the config dict."""
    tmpl = config_mod.load_config(search.MINI_SEARCHES["prior_scale"]["template"])
    point = search.sample_mini_points(tmpl.master_seed, "prior_scale")[0]
    cfg = search.mini_candidate_config(tmpl, point, 10, which="prior_scale", index=0)
    agent = config_mod.build_agent(cfg, 0)
    assert agent.cfg.prior_scale == pytest.approx(point["prior_scale"])

    tmpl = config_mod.load_config(search.MINI_SEARCHES["eps_schedule"]["template"])
    point = search.sample_mini_points(tmpl.master_seed, "eps_schedule")[0]
    cfg = search.mini_candidate_config(tmpl, point, 10, which="eps_schedule", index=0)
    agent = config_mod.build_agent(cfg, 0)
    assert agent.cfg.eps_end == pytest.approx(point["eps_end"])
    assert agent.cfg.eps_decay_steps == cfg.data["factor_specific"]["eps_schedule"][
        "eps_decay_steps"
    ]


def test_unknown_mini_search_name_raises():
    with pytest.raises(ValueError, match="unknown mini-search"):
        search.sample_mini_points(0, "not_a_search")


def test_mini_cli_dry_run(capsys):
    for which, spec in search.MINI_SEARCHES.items():
        assert search.main(["--kind", which, "--dry-run"]) == 0
        lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("c")]
        assert len(lines) == search.N_MINI
        assert spec["param"] in lines[0]


def test_all_three_searches_use_one_episode_budget():
    """C-iii equal-search-budget. Each search points at a DIFFERENT template, and the templates
    disagree (500 vs 2000 episodes). Inheriting would have tuned the backbone at a 4x smaller
    budget than the two mini-searches while the protocol claims identical search tiers."""
    seen = set()
    tmpl = config_mod.load_config(TEMPLATE)
    point = search.sample_backbone_points(tmpl.master_seed)[0]
    for size in search.DEV_SIZES:
        cfg = search.candidate_config(tmpl, point, size, index=0)
        seen.add(cfg.data["env_budget"]["episodes"])
    for which, spec in search.MINI_SEARCHES.items():
        t = config_mod.load_config(spec["template"])
        p = search.sample_mini_points(t.master_seed, which)[0]
        for size in search.DEV_SIZES:
            cfg = search.mini_candidate_config(t, p, size, which=which, index=0)
            seen.add(cfg.data["env_budget"]["episodes"])
    assert seen == {config_mod.DEEP_SEA_EPISODE_BUDGET}


def test_episode_budget_is_constant_in_n_not_inherited():
    """The budget must not vary with size (owner decision: constant episodes, not constant
    steps), and must not track whatever the template happened to declare."""
    tmpl = config_mod.load_config(TEMPLATE)
    assert tmpl.data["env_budget"]["episodes"] != config_mod.DEEP_SEA_EPISODE_BUDGET
    point = search.sample_backbone_points(tmpl.master_seed)[0]
    budgets = {
        search.candidate_config(tmpl, point, size, index=0).data["env_budget"]["episodes"]
        for size in (10, 20, 30)
    }
    assert budgets == {2000}


def test_committed_deepsea_cells_agree_with_the_pinned_budget():
    """2000 is not a new number — it is what the eight committed factorial cells already use.
    If a cell ever disagrees, the tuning runs stop matching the runs they tune for."""
    import glob

    for path in sorted(glob.glob("configs/cell_*deepsea*.yaml")):
        cfg = config_mod.load_config(path)
        assert cfg.data["env_budget"]["episodes"] == config_mod.DEEP_SEA_EPISODE_BUDGET, path


def test_episodes_override_is_test_only():
    """The override exists so the end-to-end tests run in seconds. If a protocol entry point
    ever grows one, the searches stop being comparable and this catches it."""
    source = Path("src/search.py").read_text()
    runner_bodies = source.split("def main(", 1)[0]
    assert "episodes=EPISODES" not in runner_bodies
    cli = source.split("def main(", 1)[1]
    assert "--episodes" not in cli, "CLI must not expose an episode-budget flag"
    assert "episodes=" not in cli, "main() must not pass an episode override"


def test_parallel_execution_is_byte_identical_to_serial(tiny_template, tmp_path):
    """The sweep may be parallelized ONLY because every stream derives from
    (master_seed, cell_id, stream, seed_index) — never from a global RNG, wall-clock time, or
    execution order. If that ever stops holding, the winner becomes schedule-dependent and the
    search stops being reproducible from its record. This test is the tripwire."""
    kw = dict(n=3, seeds=(0, 1), sizes=(5, 6), n_checkpoints=5, episodes=30)
    ser, rec_s = search.run_backbone_search(tiny_template, out_dir=tmp_path / "s", **kw)
    par, rec_p = search.run_backbone_search(
        tiny_template, out_dir=tmp_path / "p", max_workers=4, **kw
    )
    assert [e["scores"] for e in rec_p.per_candidate] == [e["scores"] for e in rec_s.per_candidate]
    assert par.winner.label == ser.winner.label
    assert rec_p.points == rec_s.points


def test_parallel_mini_search_is_byte_identical_to_serial(tmp_path):
    """Same tripwire for the class-3 path, which uses a different materializer."""
    template = config_mod.load_config("configs/example_rpbdqn_deepsea_dev.yaml")
    data = dict(template.data)
    data["env_budget"] = {"deep_sea_size": 5, "episodes": 30}
    data.pop("cell_id", None)
    data.pop("size_class", None)
    tiny = config_mod.resolve_config(data)
    kw = dict(n=2, seeds=(0,), sizes=(5,), n_checkpoints=4, episodes=30)
    ser, rec_s = search.run_mini_search(tiny, "prior_scale", out_dir=tmp_path / "s", **kw)
    par, rec_p = search.run_mini_search(
        tiny, "prior_scale", out_dir=tmp_path / "p", max_workers=2, **kw
    )
    assert [e["scores"] for e in rec_p.per_candidate] == [e["scores"] for e in rec_s.per_candidate]
    assert par.winner.label == ser.winner.label
