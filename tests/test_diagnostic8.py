"""Tests for Diagnostic 8 — initial-state return-prediction alignment (MinAtar, appendix-only).

Design principles:
* Slow tests that need a real MinAtar game / real model are guarded by ``SLOW = True`` and
  skipped in CI by default.  The marker is ``pytest.mark.slow``.
* All other tests run on synthetic data so CI stays fast.
* Tests mirror the existing pattern in ``test_diagnostics_battery.py``: check mathematical
  correctness, check the uninformative / undefined paths, check provenance fields.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.diagnostics.initial_state_alignment import (
    DEFAULT_GAMMA,
    DEFAULT_MAX_ROLLOUT_STEPS,
    N_START_SEEDS,
    UNIQUENESS_THRESHOLD,
    Diag8Result,
    _is_degenerate,
    _uninformative,
    deduplicate_starts,
    diag8_to_record,
    initial_state_return_alignment,
    make_mean_greedy_policy,
    value_and_uncertainty,
)

# --------------------------------------------------------------------------- #
# Helpers / synthetic fixtures
# --------------------------------------------------------------------------- #

def _make_rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_samples(
    n: int,
    m: int,
    n_actions: int,
    v_bar: np.ndarray,
    sigma_v: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Construct a synthetic ``[N, M, A]`` value-samples array.

    Action 0 is always greedy, so max_a Q_m = Q_m[:, 0] = v_bar + sigma_v * z_m,
    giving mean = v_bar and std(max_a Q_m) ≈ sigma_v (exact as M → ∞).
    """
    _Z = rng.standard_normal((m,))
    arr = np.zeros((n, m, n_actions), dtype=np.float64)
    for i in range(n):
        arr[i, :, 0] = v_bar[i] + sigma_v[i] * _Z
        for a in range(1, n_actions):
            arr[i, :, a] = arr[i, :, 0] - float(a)  # action a is always strictly dominated
    return arr.astype(np.float32)


class _SyntheticSampler:
    """Minimal ValueSampler backed by a fixed ``[N, M, A]`` array (for slice indexing)."""

    def __init__(self, samples: np.ndarray):
        self._samples = samples  # [N, M, A]

    def value_samples(self, probe_states: np.ndarray) -> np.ndarray:
        # probe_states is [batch, *obs_shape]; ignore obs content, use row index
        batch = probe_states.shape[0]
        return self._samples[:batch]


# --------------------------------------------------------------------------- #
# Unit tests for sub-functions
# --------------------------------------------------------------------------- #

class TestIsDegenerate:
    def test_all_same_returns_true(self):
        assert _is_degenerate(np.full(10, 3.14))

    def test_different_values_returns_false(self):
        assert not _is_degenerate(np.array([1.0, 2.0]))

    def test_empty_returns_true(self):
        assert _is_degenerate(np.array([]))

    def test_within_float32_noise_degenerate(self):
        # Values differing by less than 8 * eps * scale should be degenerate.
        x = np.float64(1.0)
        eps = np.finfo(np.float32).eps
        arr = np.array([x, x + 2.0 * eps])
        assert _is_degenerate(arr)

    def test_just_above_threshold_not_degenerate(self):
        x = np.float64(1.0)
        eps = np.finfo(np.float32).eps
        arr = np.array([x, x + 100.0 * eps])  # well above 8 * eps
        assert not _is_degenerate(arr)


class TestDeduplicateStarts:
    def _obs(self, val: float, size: int = 4) -> np.ndarray:
        return np.full((size,), val, dtype=np.float32)

    def test_all_unique(self):
        obs_list = [self._obs(float(i)) for i in range(5)]
        snaps = [{"snap": i} for i in range(5)]
        unique_obs, unique_snaps = deduplicate_starts(obs_list, snaps)
        assert unique_obs.shape[0] == 5
        assert len(unique_snaps) == 5

    def test_duplicates_removed(self):
        obs_list = [self._obs(1.0), self._obs(2.0), self._obs(1.0)]
        snaps = [{"snap": 0}, {"snap": 1}, {"snap": 2}]
        unique_obs, unique_snaps = deduplicate_starts(obs_list, snaps)
        assert unique_obs.shape[0] == 2
        # First occurrence of obs(1.0) is retained
        assert unique_snaps[0] == {"snap": 0}

    def test_all_identical(self):
        obs_list = [self._obs(3.14)] * 10
        snaps = [{} for _ in range(10)]
        unique_obs, unique_snaps = deduplicate_starts(obs_list, snaps)
        assert unique_obs.shape[0] == 1

    def test_empty_input(self):
        unique_obs, unique_snaps = deduplicate_starts([], [])
        assert unique_obs.shape[0] == 0
        assert unique_snaps == []


class TestValueAndUncertainty:
    """value_and_uncertainty must return correct v_bar and sigma_v for known inputs."""

    def _run(self, n=8, m=20, n_actions=4):
        rng = _make_rng(42)
        v_bar_true = rng.uniform(0.0, 5.0, size=n)
        sigma_v_true = rng.uniform(0.1, 2.0, size=n)
        samples = _make_samples(n, m, n_actions, v_bar_true, sigma_v_true, _make_rng(99))
        sampler = _SyntheticSampler(samples)
        # probe_states shape doesn't matter for _SyntheticSampler
        probe = np.zeros((n, 1), dtype=np.float32)
        v_bar_hat, sigma_v_hat = value_and_uncertainty(sampler, probe)
        return v_bar_true, sigma_v_true, v_bar_hat, sigma_v_hat

    def test_v_bar_close_to_true(self):
        vb_true, _, vb_hat, _ = self._run(n=10, m=200)
        # mean Q̄ over M draws → v_bar_true as M → ∞
        np.testing.assert_allclose(vb_hat, vb_true, atol=0.3)

    def test_sigma_v_close_to_true(self):
        _, sv_true, _, sv_hat = self._run(n=10, m=200)
        np.testing.assert_allclose(sv_hat, sv_true, atol=0.3)

    def test_sigma_v_nonneg(self):
        _, _, _, sv_hat = self._run()
        assert (sv_hat >= 0).all()

    def test_rejects_non_3d_samples(self):
        sampler = _SyntheticSampler(np.zeros((4, 4)))  # 2D instead of 3D
        with pytest.raises(ValueError, match=r"\[N, M, A\]"):
            value_and_uncertainty(sampler, np.zeros((4, 1)))

    def test_rejects_m_lt_2(self):
        sampler = _SyntheticSampler(np.zeros((4, 1, 2)))
        with pytest.raises(ValueError, match="M = 1"):
            value_and_uncertainty(sampler, np.zeros((4, 1)))


class TestInitialStateReturnAlignment:
    """Direct tests of the Spearman-ρ function with synthetic v_bar / sigma_v / G."""

    def _mk_result(self, n=20, *, positive=True, seed=0) -> Diag8Result:
        rng = _make_rng(seed)
        sigma_v = rng.uniform(0.1, 2.0, size=n)
        if positive:
            # error positively correlated with sigma_v
            error = sigma_v + rng.uniform(0.0, 0.3, size=n)
        else:
            # error negatively correlated
            error = (sigma_v.max() - sigma_v) + rng.uniform(0.0, 0.3, size=n)
        v_bar = rng.uniform(0, 5, size=n)
        G = v_bar - error  # so |V̄ − G| = error
        return initial_state_return_alignment(
            v_bar, sigma_v, G, "test_game",
            n_seeds=100, n_unique_starts=n,
        )

    def test_positive_correlation_gives_positive_rho(self):
        res = self._mk_result(n=30, positive=True)
        assert res.defined
        assert res.value > 0.5, f"expected rho > 0.5, got {res.value:.3f}"

    def test_negative_correlation_gives_negative_rho(self):
        res = self._mk_result(n=30, positive=False)
        assert res.defined
        assert res.value < -0.5, f"expected rho < -0.5, got {res.value:.3f}"

    def test_constant_sigma_gives_undefined(self):
        v_bar = np.ones(20)
        sigma_v = np.full(20, 1.0)  # constant
        G = np.linspace(0.0, 5.0, 20)
        res = initial_state_return_alignment(
            v_bar, sigma_v, G, "test_game",
            n_seeds=100, n_unique_starts=20,
        )
        assert not res.defined
        assert math.isnan(res.value)
        assert "constant" in res.reason.lower()

    def test_constant_error_gives_undefined(self):
        rng = _make_rng(7)
        sigma_v = rng.uniform(0.1, 2.0, size=20)
        v_bar = np.zeros(20)
        G = np.zeros(20)  # |V̄ − G| all 0.0
        res = initial_state_return_alignment(
            v_bar, sigma_v, G, "test_game",
            n_seeds=100, n_unique_starts=20,
        )
        assert not res.defined
        assert math.isnan(res.value)

    def test_n_below_3_gives_undefined(self):
        res = initial_state_return_alignment(
            np.array([1.0, 2.0]),
            np.array([0.5, 1.0]),
            np.array([0.5, 1.5]),
            "test_game",
            n_seeds=10, n_unique_starts=2,
        )
        assert not res.defined
        assert "n = 2" in res.reason

    def test_result_fields_populated(self):
        res = self._mk_result(n=20)
        assert res.game == "test_game"
        assert res.n_seeds == 100
        assert res.n_unique_starts == 20
        assert res.n_excluded == 0  # σ = 0 is not excluded in this diagnostic
        if res.defined:
            assert res.n_used == 20
        assert "mean_v_bar" in res.extra
        assert "mean_return" in res.extra
        assert "mean_sigma_v" in res.extra

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            initial_state_return_alignment(
                np.ones(5), np.ones(4), np.ones(5), "g",
                n_seeds=10, n_unique_starts=5,
            )


class TestUninformative:
    def test_uninformative_result_structure(self):
        res = _uninformative("breakout", "too few unique states", n_seeds=100, n_unique=12)
        assert not res.defined
        assert math.isnan(res.value)
        assert res.n_seeds == 100
        assert res.n_unique_starts == 12
        assert res.n_used == 0
        assert "too few" in res.reason


class TestDiag8ToRecord:
    def test_code_version_present(self):
        res = _uninformative("g", "test", n_seeds=10, n_unique=5)
        rec = diag8_to_record(res)
        assert "code_version" in rec
        assert isinstance(rec["code_version"], dict)

    def test_supplied_code_version_used(self):
        res = _uninformative("g", "test", n_seeds=10, n_unique=5)
        rec = diag8_to_record(res, code_ver={"sha": "abc", "dirty": False})
        assert rec["code_version"] == {"sha": "abc", "dirty": False}

    def test_all_fields_serialisable(self):
        import json
        rng = _make_rng(0)
        sigma_v = rng.uniform(0.1, 1.0, size=10)
        v_bar = rng.uniform(0, 3, size=10)
        G = v_bar - 0.3 * sigma_v
        res = initial_state_return_alignment(
            v_bar, sigma_v, G, "test_game", n_seeds=100, n_unique_starts=10
        )
        rec = diag8_to_record(res)
        # Replace NaN with None for JSON round-trip
        rec_str = json.dumps(rec, allow_nan=False if not math.isnan(res.value) else True)
        assert isinstance(rec_str, str)


class TestMakeMeanGreedyPolicy:
    def test_selects_highest_mean_q_action(self):
        # M=4 samples; action 2 has the highest mean
        samples = np.zeros((1, 4, 3), dtype=np.float32)
        samples[0, :, 0] = [0.1, 0.2, 0.3, 0.4]   # mean = 0.25
        samples[0, :, 1] = [0.0, 0.0, 0.0, 0.0]    # mean = 0.00
        samples[0, :, 2] = [0.9, 1.0, 1.1, 1.2]    # mean = 1.05  ← highest
        sampler = _SyntheticSampler(samples)
        policy = make_mean_greedy_policy(sampler)
        obs = np.zeros(4, dtype=np.float32)
        assert policy(obs) == 2

    def test_tie_broken_by_lowest_index(self):
        # Actions 0 and 1 have identical mean; action 0 should win (lowest index).
        samples = np.zeros((1, 4, 2), dtype=np.float32)
        samples[0, :, 0] = [1.0, 1.0, 1.0, 1.0]
        samples[0, :, 1] = [1.0, 1.0, 1.0, 1.0]
        sampler = _SyntheticSampler(samples)
        policy = make_mean_greedy_policy(sampler)
        obs = np.zeros(4, dtype=np.float32)
        assert policy(obs) == 0


# --------------------------------------------------------------------------- #
# Frozen-constant guard — these must never silently change
# --------------------------------------------------------------------------- #

class TestFrozenConstants:
    """Fix #6 pins these values; changing them changes the diagnostic definition."""

    def test_n_start_seeds(self):
        assert N_START_SEEDS == 100

    def test_uniqueness_threshold(self):
        assert UNIQUENESS_THRESHOLD == 20

    def test_default_gamma(self):
        assert DEFAULT_GAMMA == pytest.approx(0.99)

    def test_default_max_rollout_steps(self):
        assert DEFAULT_MAX_ROLLOUT_STEPS == 500
