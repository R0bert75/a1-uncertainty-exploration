"""Tests for the frozen selection statistic and tie-breaking (freeze items 3, 19)."""

from __future__ import annotations

import numpy as np
import pytest

from src.selection import (
    Candidate,
    iqm,
    score_candidates,
    select_best,
    select_k_shared,
)


# --------------------------------------------------------------------------- #
# IQM: exact values, hand-computed
# --------------------------------------------------------------------------- #
def test_iqm_n4_drops_one_from_each_end():
    # n = 4k: exactly the "trim a quartile from each end" form.
    assert iqm([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5)


def test_iqm_n8_averages_middle_four():
    vals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert iqm(vals) == pytest.approx(np.mean([2.0, 3.0, 4.0, 5.0]))


def test_iqm_is_order_invariant():
    vals = [4.0, 1.0, 3.0, 2.0]
    assert iqm(vals) == pytest.approx(iqm(sorted(vals)))
    assert iqm(vals) == pytest.approx(iqm(sorted(vals, reverse=True)))


def test_iqm_singleton_is_the_value():
    assert iqm([7.5]) == pytest.approx(7.5)


def test_iqm_trims_outliers_that_would_move_the_mean():
    """The property that motivates IQM: a jackpot seed must not decide a search."""
    clean = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    spiked = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1000.0]
    assert iqm(clean) == pytest.approx(iqm(spiked))
    assert np.mean(clean) != pytest.approx(np.mean(spiked))


def test_iqm_weights_sum_to_one_for_all_n():
    """Continuity check: IQM of a constant sample is that constant for every n."""
    for n in range(1, 33):
        assert iqm([3.0] * n) == pytest.approx(3.0), f"failed at n={n}"


def test_iqm_is_monotone_in_the_middle_half():
    base = [1.0, 2.0, 3.0, 4.0]
    higher = [1.0, 2.0, 3.5, 4.0]
    assert iqm(higher) > iqm(base)


def test_iqm_rejects_empty():
    with pytest.raises(ValueError, match="at least one value"):
        iqm([])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_iqm_rejects_non_finite(bad):
    """Undefined values must be counted and excluded upstream (§3.3), not absorbed."""
    with pytest.raises(ValueError, match="non-finite"):
        iqm([1.0, 2.0, bad, 4.0])


# --------------------------------------------------------------------------- #
# select_best: IQM ordering + the frozen tie-breaker
# --------------------------------------------------------------------------- #
def _cand(label, value, scores):
    return Candidate(label=label, params={"p": value}, scores=tuple(scores), sort_key=value)


def test_select_best_picks_highest_iqm():
    a = _cand("a", 0.1, [1.0, 1.0, 1.0, 1.0])
    b = _cand("b", 0.2, [5.0, 5.0, 5.0, 5.0])
    result = select_best([a, b])
    assert result.winner.label == "b"
    assert result.tie_broken is False


def test_tie_breaks_to_lower_parameter_value():
    """Freeze item 3: 'ties broken by the lower parameter value'."""
    hi = _cand("hi", 0.9, [2.0, 2.0, 2.0, 2.0])
    lo = _cand("lo", 0.1, [2.0, 2.0, 2.0, 2.0])
    result = select_best([hi, lo])  # deliberately hi-first
    assert result.winner.label == "lo"
    assert result.tie_broken is True


def test_tie_break_is_independent_of_input_order():
    hi = _cand("hi", 0.9, [2.0] * 4)
    lo = _cand("lo", 0.1, [2.0] * 4)
    assert select_best([hi, lo]).winner.label == select_best([lo, hi]).winner.label == "lo"


def test_duplicate_sort_keys_are_rejected():
    """Without a total order the frozen tie-breaker is silent — fail rather than guess."""
    a = _cand("a", 0.5, [1.0] * 4)
    b = _cand("b", 0.5, [1.0] * 4)
    with pytest.raises(ValueError, match="share a sort_key"):
        select_best([a, b])


def test_select_best_rejects_empty():
    with pytest.raises(ValueError, match="at least one candidate"):
        select_best([])


def test_ranking_is_full_and_ordered():
    cands = [_cand("a", 0.1, [1.0] * 4), _cand("b", 0.2, [3.0] * 4), _cand("c", 0.3, [2.0] * 4)]
    result = select_best(cands)
    assert [c.label for c in result.ranking] == ["b", "c", "a"]


def test_as_record_is_json_shaped():
    result = select_best([_cand("a", 0.1, [1.0] * 4), _cand("b", 0.2, [3.0] * 4)])
    rec = result.as_record()
    assert rec["winner"]["label"] == "b"
    assert rec["winner"]["n_seeds"] == 4
    assert len(rec["ranking"]) == 2


# --------------------------------------------------------------------------- #
# K_shared: the joint rule
# --------------------------------------------------------------------------- #
def _k_cands(per_k):
    """{K: [(label, param, scores)]} -> {K: [Candidate]}"""
    return {
        k: [
            Candidate(label=lb, params={"lr": p}, scores=tuple(s), sort_key=p)
            for lb, p, s in rows
        ]
        for k, rows in per_k.items()
    }


def test_k_shared_argmaxes_the_cross_method_mean_not_each_method():
    """The rule is joint: a K that is best for one method but poor for the other loses.

    Method X peaks at K=5 (10.0) but method Y is terrible there (0.0) -> mean 5.0.
    At K=10 both are 6.0 -> mean 6.0, which wins. A per-method argmax would have said 5.
    """
    per_method = {
        "bdqn": _k_cands({
            5: [("a", 0.1, [10.0] * 4)],
            10: [("b", 0.1, [6.0] * 4)],
            20: [("c", 0.1, [1.0] * 4)],
        }),
        "rp_bdqn": _k_cands({
            5: [("d", 0.1, [0.0] * 4)],
            10: [("e", 0.1, [6.0] * 4)],
            20: [("f", 0.1, [1.0] * 4)],
        }),
    }
    k_shared, means, best_by = select_k_shared(per_method)
    assert k_shared == 10
    assert means[5] == pytest.approx(5.0)
    assert means[10] == pytest.approx(6.0)
    assert best_by["bdqn"][5].label == "a"


def test_k_shared_takes_best_config_within_each_cell_first():
    """Order of operations: best-within-(method,K), then mean across methods."""
    per_method = {
        "bdqn": _k_cands({
            5: [("weak", 0.1, [1.0] * 4), ("strong", 0.2, [9.0] * 4)],
            10: [("m", 0.1, [4.0] * 4)],
            20: [("n", 0.1, [4.0] * 4)],
        }),
        "rp_bdqn": _k_cands({
            5: [("p", 0.1, [9.0] * 4)],
            10: [("q", 0.1, [4.0] * 4)],
            20: [("r", 0.1, [4.0] * 4)],
        }),
    }
    k_shared, means, best_by = select_k_shared(per_method)
    assert best_by["bdqn"][5].label == "strong"
    assert means[5] == pytest.approx(9.0)
    assert k_shared == 5


def test_k_shared_ties_prefer_lower_k():
    flat = {5: [("a", 0.1, [4.0] * 4)], 10: [("b", 0.1, [4.0] * 4)], 20: [("c", 0.1, [4.0] * 4)]}
    per_method = {"bdqn": _k_cands(flat), "rp_bdqn": _k_cands(flat)}
    k_shared, _, _ = select_k_shared(per_method)
    assert k_shared == 5


def test_k_shared_requires_exactly_two_methods():
    one = {"bdqn": _k_cands({5: [("a", 0.1, [1.0] * 4)]})}
    with pytest.raises(ValueError, match="TWO ensemble methods"):
        select_k_shared(one, k_values=(5,))


def test_k_shared_rejects_missing_cell():
    per_method = {
        "bdqn": _k_cands({5: [("a", 0.1, [1.0] * 4)], 10: [("b", 0.1, [1.0] * 4)]}),
        "rp_bdqn": _k_cands({5: [("c", 0.1, [1.0] * 4)]}),  # missing K=10
    }
    with pytest.raises(ValueError, match="no candidates"):
        select_k_shared(per_method, k_values=(5, 10))


# --------------------------------------------------------------------------- #
# score_candidates
# --------------------------------------------------------------------------- #
def test_score_candidates_builds_evaluated_candidates():
    points = [{"lr": 0.001}, {"lr": 0.01}]
    cands = score_candidates(
        points,
        score_fn=lambda p: [p["lr"] * 1000] * 4,
        sort_key_fn=lambda p: p["lr"],
    )
    assert [c.label for c in cands] == ["lr=0.001", "lr=0.01"]
    assert select_best(cands).winner.params["lr"] == 0.01
