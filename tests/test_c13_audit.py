"""Tests for the C13 configuration-identity audit (audits/c13_audit.py).

The audit is the mechanism that enforces freeze item 12's class-3 rule — a reported
contrast pair may differ ONLY in the varied factor and its own factor-specific parameters.
It had no test coverage before the registry was filled; these tests pin both the diff
engine's semantics and the registry's structural agreement with the frozen 10-cell design.

The negative controls matter more than the positive ones: an audit that cannot fail is
not an audit, and this one is the sole guard against a silent confound in a reported
contrast.
"""

from __future__ import annotations

import pytest

from audits.c13_audit import (
    CONTRAST_REGISTRY,
    IDENTITY_KEYS,
    audit_pair,
    collect_cells,
    run_audit,
)
from src import config as config_mod

# The 10 cells of the structured partial factorial (freeze item 12): the full
# use_rule x prior core at K=10, augmented with K in {5,20} for episodic at both priors.
TEN_CELLS = frozenset(
    [f"{u}|{p}|K10" for u in ("episodic", "per_step", "ensemble_mean") for p in ("off", "on")]
    + [f"episodic|{p}|K{k}" for k in (5, 20) for p in ("off", "on")]
)


def _cfg(**over):
    base = {
        "run_id": "r", "role": "development", "part": "A", "method": "bdqn",
        "env": "deep_sea", "master_seed": 0,
        "use_rule": "episodic", "prior": "off", "K": 10, "arm": "episodic|off|K10",
        "backbone": {"lr": 0.0005, "batch_size": 32, "gamma": 0.99},
        "ensemble_shared": {"mask_prob": 0.5, "head_loss_agg": "grad_norm_1_over_k"},
        "factor_specific": {"prior_scale": None, "eps_schedule": None},
    }
    base.update(over)
    return base


# --------------------------------------------------------------------------- #
# diff engine
# --------------------------------------------------------------------------- #

def test_identical_configs_pass():
    assert audit_pair(_cfg(), _cfg(), ["use_rule"], [])["pass"]


def test_varied_factor_is_licensed():
    res = audit_pair(_cfg(), _cfg(use_rule="per_step"), ["use_rule"], [])
    assert res["pass"], res["illicit_diffs"]


def test_unlicensed_difference_fails():
    """NEGATIVE CONTROL: a backbone drift between two arms is exactly the confound C13 exists
    to catch. If this ever passes, the audit is inert."""
    a, b = _cfg(), _cfg()
    b["backbone"] = dict(b["backbone"], lr=0.001)
    res = audit_pair(a, b, ["use_rule"], [])
    assert not res["pass"]
    assert "backbone.lr" in res["illicit_diffs"]
    assert res["illicit_diffs"]["backbone.lr"] == {"a": 0.0005, "b": 0.001}


def test_licensed_key_licenses_its_whole_subtree():
    """eps_schedule is a DICT; _flatten explodes it into leaves. Licensing the parent must
    license the leaves, or the C-USE pair fails on a parameter the protocol licenses."""
    a = _cfg()
    b = _cfg(use_rule="ensemble_mean")
    b["factor_specific"] = dict(
        b["factor_specific"],
        eps_schedule={"eps_start": 1.0, "eps_end": 0.05, "eps_decay_steps": 3000},
    )
    res = audit_pair(a, b, ["use_rule"], ["factor_specific.eps_schedule"])
    assert res["pass"], res["illicit_diffs"]


def test_subtree_licensing_does_not_leak_to_siblings():
    """Licensing factor_specific.eps_schedule must NOT license factor_specific.prior_scale —
    prefix matching is on dotted segments, not raw string prefixes."""
    a = _cfg()
    b = _cfg()
    b["factor_specific"] = dict(b["factor_specific"], prior_scale=3.0)
    res = audit_pair(a, b, [], ["factor_specific.eps_schedule"])
    assert not res["pass"]
    assert "factor_specific.prior_scale" in res["illicit_diffs"]


def test_absent_vs_present_is_a_difference():
    a = _cfg()
    b = _cfg()
    del b["backbone"]["gamma"]
    res = audit_pair(a, b, [], [])
    assert not res["pass"]
    assert res["illicit_diffs"]["backbone.gamma"]["b"] == "<absent>"


def test_identity_keys_are_excluded():
    """run_id/arm/cell_id differ in every pair by construction; they are labels, not params."""
    a = _cfg()
    b = _cfg(run_id="other", arm="per_step|off|K10")
    b["cell_id"] = "per_step|off|K10"
    a["cell_id"] = "episodic|off|K10"
    assert audit_pair(a, b, [], [])["pass"]
    assert {"run_id", "arm", "cell_id"} <= IDENTITY_KEYS


# --------------------------------------------------------------------------- #
# registry structure vs. the frozen design
# --------------------------------------------------------------------------- #

def test_registry_covers_the_four_frozen_contrasts():
    assert set(CONTRAST_REGISTRY) == {"C-USE", "C-COHERENCE", "C-PRIOR", "C-K"}


def test_every_registered_cell_is_one_of_the_ten():
    for contrast, spec in CONTRAST_REGISTRY.items():
        for pair in spec["pairs"]:
            for cell in pair:
                assert cell in TEN_CELLS, f"{contrast} references non-cell {cell!r}"


def test_coherence_and_use_are_k10_only():
    """Freeze item 12: C-COHERENCE / C-USE are estimated at K=10 ONLY (no K x use_rule)."""
    for contrast in ("C-USE", "C-COHERENCE"):
        for pair in CONTRAST_REGISTRY[contrast]["pairs"]:
            assert all(c.endswith("|K10") for c in pair), (contrast, pair)


def test_ck_varies_only_within_episodic():
    for pair in CONTRAST_REGISTRY["C-K"]["pairs"]:
        assert all(c.startswith("episodic|") for c in pair), pair
        priors = {c.split("|")[1] for c in pair}
        assert len(priors) == 1, f"C-K must hold prior fixed, got {pair}"


def test_prior_scale_licensed_only_where_prior_varies():
    for contrast, spec in CONTRAST_REGISTRY.items():
        lic = set(spec["licensed"])
        if "factor_specific.prior_scale" in lic:
            assert "prior" in spec["varies"], contrast


# --------------------------------------------------------------------------- #
# end-to-end over the committed configs
# --------------------------------------------------------------------------- #

def test_committed_configs_pass_the_audit(tmp_path):
    """The real gate: every committed contrast arm must be class-3 clean, with no pair
    skipped for a missing arm."""
    from pathlib import Path

    cells = collect_cells(Path("configs"), "configs")
    report = run_audit(cells)
    assert report["n_fail"] == 0, report["contrasts"]
    assert report["n_skipped"] == 0, "a registered contrast has no committed B-arm"
    assert report["n_pass"] == sum(len(s["pairs"]) for s in CONTRAST_REGISTRY.values())


def test_collect_cells_keys_on_canonical_cell_id():
    from pathlib import Path

    cells = collect_cells(Path("configs"), "configs")
    assert TEN_CELLS <= set(cells), sorted(TEN_CELLS - set(cells))


@pytest.mark.parametrize("cell", sorted(TEN_CELLS))
def test_every_cell_has_a_resolvable_committed_config(cell):
    from pathlib import Path

    cells = collect_cells(Path("configs"), "configs")
    cfg = cells[cell]
    assert cfg["cell_id"] == cell
    assert config_mod._canonical_cell_id(cfg["use_rule"], cfg["prior"], cfg["K"]) == cell
