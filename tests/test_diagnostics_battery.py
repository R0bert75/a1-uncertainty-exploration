"""§3.3 diagnostics battery (1-5, 7): known-answer and invariance tests.

The reducers are cheap and deterministic, so almost everything here is a *constructed* case
whose answer is known analytically rather than a regression against recorded output. Where a
statistic's sign is the scientific claim (1, 2, 3, 5), both signs are tested: a diagnostic that
cannot report "misaligned" is not measuring alignment.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from analysis import diagnostics_battery as db
from src.deep_sea import DeepSea
from src.diagnostics.samplers import deep_sea_probe_states

SIZE = 8
KEY = dict(master_seed=0, cell_id="c01", seed_index=0)


@pytest.fixture
def setup():
    _obs, idx = deep_sea_probe_states(SIZE)
    env = DeepSea(size=SIZE, **KEY)
    return env, idx, db.align_q_star(env.q_star(gamma=1.0), idx)


#: A zero-mean, unit-std (ddof=0) pattern of length 10, and its action-antisymmetric partner.
#: Using a fixed pattern rather than random draws makes the realized σ EXACTLY the requested one,
#: so known-answer tests are exact rather than approximate.
_Z = np.asarray([-1.0, 1.0] * 5)
_Z = (_Z - _Z.mean()) / _Z.std()


def _samples(qbar, sigma, *, antisymmetric=True):
    """Samples with exact per-(s, a) mean and std (ddof=0).

    ``antisymmetric=True`` gives action 1 the NEGATED pattern, so the two actions' samples are
    perfectly anti-correlated. This matters and is not cosmetic: with the SAME pattern on every
    action the samples are perfectly correlated, so the per-sample argmax can never flip, d(s) is
    identically 0, and u_g(s) is identically 0 — which silently disables diagnostics 2 and 3.
    Real ensemble heads have independent per-action error, so the shared-pattern fixture is the
    unrealistic one. Pass ``antisymmetric=False`` to construct the degenerate case deliberately.
    """
    q = np.asarray(qbar, dtype=np.float64)
    sig = np.asarray(sigma, dtype=np.float64)
    pattern = np.tile(_Z[None, :, None], (q.shape[0], 1, q.shape[1]))
    if antisymmetric and q.shape[1] >= 2:
        pattern[:, :, 1::2] *= -1.0
    return (q[:, None, :] + sig[:, None, :] * pattern).astype(np.float32)


# --------------------------------------------------------------------------- #
# Probe/Q* alignment — the wiring every reducer depends on
# --------------------------------------------------------------------------- #
def test_align_q_star_gathers_the_probe_ordering(setup):
    env, idx, q_flat = setup
    q = env.q_star(gamma=1.0)
    assert q_flat.shape == (len(idx), 2)
    for i, (r, c) in enumerate(idx):
        assert np.array_equal(q_flat[i], q[r, c])


def test_q_star_is_run_specific_so_the_mapping_key_matters():
    """The trap ``run_battery_over_run`` exists to prevent.

    DeepSea draws its per-row "which action goes right" mapping from the frozen env_mapping
    stream keyed on (master_seed, cell_id, seed_index). Two different keys give Q* tensors that
    differ by an action-axis flip on some rows -- so a battery run against the wrong key
    compares σ to the wrong ground truth and silently reports near-zero alignment.
    """
    a = DeepSea(size=SIZE, master_seed=0, cell_id="c01", seed_index=0)
    b = DeepSea(size=SIZE, master_seed=0, cell_id="c02", seed_index=0)
    if a.mapping_hash == b.mapping_hash:
        pytest.skip("the two keys happened to draw the same mapping")
    assert not np.array_equal(a.q_star(1.0), b.q_star(1.0))


# --------------------------------------------------------------------------- #
# 1. Marginal alignment
# --------------------------------------------------------------------------- #
def test_marginal_alignment_is_1_when_sigma_tracks_error_exactly(setup):
    """Perfect monotone alignment ⇒ ρ = 1 exactly (Spearman is rank-based)."""
    _env, idx, q_flat = setup
    S, A = q_flat.shape
    err = np.linspace(0.01, 1.0, S * A).reshape(S, A)
    got = db.marginal_alignment(_samples(q_flat + err, err), q_flat)
    assert got.value == pytest.approx(1.0)
    assert got.n_used == S * A and got.n_excluded == 0


def test_marginal_alignment_is_negative_when_sigma_is_anti_aligned(setup):
    """The diagnostic must be able to report *mis*alignment, not just detect alignment."""
    _env, idx, q_flat = setup
    S, A = q_flat.shape
    err = np.linspace(0.01, 1.0, S * A).reshape(S, A)
    got = db.marginal_alignment(_samples(q_flat + err, err[::-1, ::-1].copy()), q_flat)
    assert got.value == pytest.approx(-1.0)


def test_marginal_alignment_keeps_zero_sigma_pairs(setup):
    """σ = 0 is a substantive claim here (confident *and* wrong), not an exclusion.

    Only diagnostic 5 excludes σ = 0, because only there is log σ taken. If this ever starts
    excluding, the RQ2-L primary would drop its most diagnostic evidence.
    """
    _env, idx, q_flat = setup
    S, A = q_flat.shape
    sigma = np.linspace(0.1, 1.0, S * A).reshape(S, A)  # non-degenerate, so rho is defined
    sigma[0, 0] = 0.0
    bias = np.linspace(0.01, 0.5, S * A).reshape(S, A)  # non-degenerate |Q̄ − Q*| too
    got = db.marginal_alignment(_samples(q_flat + bias, sigma), q_flat)
    assert got.defined
    assert got.n_used == S * A and got.n_excluded == 0


def test_marginal_alignment_is_NA_not_zero_when_sigma_is_constant(setup):
    """Item 20: undefined ⇒ NA + reason, never an imputed 0.0 that averages into aggregates."""
    _env, idx, q_flat = setup
    S, A = q_flat.shape
    got = db.marginal_alignment(_samples(q_flat, np.full((S, A), 0.3)), q_flat)
    assert not got.defined and np.isnan(got.value)
    assert "constant" in got.reason


# --------------------------------------------------------------------------- #
# 2. Action-gap alignment
# --------------------------------------------------------------------------- #
def test_action_gap_uses_the_Qbar_top2_for_g_star_not_Q_stars_own_top2(setup):
    """The subtle clause in the frozen definition, pinned.

    Construct a state where Q̄'s ranking is the REVERSE of Q*'s. Then g* under Q̄'s (a₁, a₂) is
    the negative of g* under Q*'s own top-2, so an implementation that re-derives the pair from
    Q* gets a different |ĝ − g*| — and this test separates the two.
    """
    _env, idx, q_flat = setup
    S, _A = q_flat.shape
    qbar = -q_flat  # reverses the per-state action order wherever Q* is not tied
    sigma = np.linspace(0.05, 0.5, S)[:, None] * np.ones((1, 2))  # per-state spread ⇒ u_g varies
    samples = _samples(qbar, sigma)

    rows = np.arange(S)
    arr = samples.astype(np.float64)
    qbar_realized = arr.mean(axis=1)
    order = np.argsort(-qbar_realized, axis=1, kind="stable")
    a1, a2 = order[:, 0], order[:, 1]
    g_hat = qbar_realized[rows, a1] - qbar_realized[rows, a2]
    g_star_same_pair = q_flat[rows, a1] - q_flat[rows, a2]
    g_star_own_top2 = q_flat[rows, 0] - q_flat[rows, 1]  # Q*'s OWN action order, not Q̄'s
    err_same = np.abs(g_hat - g_star_same_pair)
    err_own = np.abs(g_hat - g_star_own_top2)
    assert not np.allclose(err_same, err_own), "fixture failed to separate the two readings"

    u_g = (arr[rows, :, a1] - arr[rows, :, a2]).std(axis=1, ddof=0)
    expected = db._spearman(u_g, err_same, "x")
    got = db.action_gap_alignment(samples, q_flat)
    assert expected.defined, "fixture produced a degenerate u_g"
    assert got.value == pytest.approx(expected.value)
    # ... and the Q*-own-top2 reading is a DIFFERENT number, so the test discriminates
    wrong = db._spearman(u_g, err_own, "x")
    assert not wrong.defined or wrong.value != pytest.approx(got.value)


def test_action_gap_u_g_is_not_recoverable_from_marginal_sigmas(setup):
    """Why the substrate stores raw samples rather than σ.

    Two sample sets with IDENTICAL σ(s,a₁), σ(s,a₂) but opposite cross-action correlation have
    different u_g (0 when the two actions move together, 2σ when they move oppositely). A
    reducer working from marginal σ alone could not tell them apart.
    """
    q = np.zeros((4, 2))
    z = np.asarray([-1.0, 1.0, -1.0, 1.0])
    z = (z - z.mean()) / z.std()
    same = np.stack([np.stack([0.3 * z, 0.3 * z], axis=1)] * 4).astype(np.float32)
    opp = np.stack([np.stack([0.3 * z, -0.3 * z], axis=1)] * 4).astype(np.float32)
    assert np.allclose(same.std(axis=1, ddof=0), opp.std(axis=1, ddof=0))
    u_same = (same[:, :, 0] - same[:, :, 1]).std(axis=1, ddof=0)
    u_opp = (opp[:, :, 0] - opp[:, :, 1]).std(axis=1, ddof=0)
    assert np.allclose(u_same, 0.0) and np.allclose(u_opp, 0.6)
    _ = q  # documents that Q* plays no role in this identity


# --------------------------------------------------------------------------- #
# 3. Incorrect-argmax flagging
# --------------------------------------------------------------------------- #
def test_incorrect_argmax_treats_Q_star_ties_as_correct():
    """``optimal set = Argmax_a Q*(s,·)`` is a SET.

    A state where both actions are exactly optimal under Q* cannot be an argmax error whichever
    action the agent prefers. A scalar-argmax implementation would flag half of them.
    """
    q_star = np.zeros((6, 2))  # every state fully tied ⇒ no state can be incorrect
    qbar = np.tile([1.0, 0.0], (6, 1))  # agent always prefers action 0
    got = db.incorrect_argmax_flagging(_samples(qbar, np.full((6, 2), 0.1)), q_star)
    assert not got.defined and "no incorrect-argmax states" in got.reason


def test_incorrect_argmax_tie_test_discriminates_against_a_scalar_argmax():
    """The tie case, built so a scalar ``argmax(Q*)`` gives the WRONG answer.

    A fixture where the agent prefers action 0 does not discriminate: ``argmax`` of a tied Q*
    also returns 0 (ties → lowest index), so the scalar and set readings agree by coincidence.
    Here the agent prefers action 1 at every fully-tied state — still not an error, since action
    1 is in the optimal set — but a scalar reading calls all of them incorrect. This gap was
    found by mutation testing, not by inspection.
    """
    n = 6
    q_star = np.zeros((n, 2))  # every state fully tied ⇒ nothing can be incorrect
    qbar = np.tile([0.0, 1.0], (n, 1))  # agent prefers action 1, the NON-argmax tied action
    sigma = np.linspace(0.05, 0.5, n)[:, None] * np.ones((1, 2))
    got = db.incorrect_argmax_flagging(_samples(qbar, sigma), q_star)
    assert not got.defined, "a tied-optimal state must never count as an argmax error"
    assert "no incorrect-argmax states" in got.reason


def test_incorrect_argmax_r_is_1_when_disagreement_perfectly_separates():
    """Perfect separation ⇒ AUC = 1 ⇒ r = 1. Built so d(s) is exactly 0 or 0.5."""
    n = 8
    q_star = np.tile([1.0, 0.0], (n, 1))  # action 0 optimal everywhere
    qbar = np.tile([1.0, 0.0], (n, 1))
    qbar[n // 2 :] = [0.0, 1.0]  # second half: agent wrong
    sigma = np.zeros((n, 2))
    sigma[n // 2 :] = 2.0  # wrong states get huge, gap-crossing spread ⇒ split modal action
    got = db.incorrect_argmax_flagging(_samples(qbar, sigma), q_star)
    assert got.extra["n_incorrect"] == n // 2 and got.extra["n_correct"] == n // 2
    assert got.value == pytest.approx(1.0)
    assert got.extra["auc"] == pytest.approx(1.0)


def test_incorrect_argmax_r_is_negative_when_disagreement_is_at_correct_states():
    """The reversed case must report a negative r, not an absolute effect size."""
    n = 8
    q_star = np.tile([1.0, 0.0], (n, 1))
    qbar = np.tile([1.0, 0.0], (n, 1))
    qbar[n // 2 :] = [0.0, 1.0]
    sigma = np.zeros((n, 2))
    sigma[: n // 2] = 2.0  # spread on the CORRECT states instead
    got = db.incorrect_argmax_flagging(_samples(qbar, sigma), q_star)
    assert got.value == pytest.approx(-1.0)


def test_incorrect_argmax_ties_in_d_contribute_half():
    """All-tied d ⇒ AUC = 0.5 ⇒ r = 0, via midranks rather than an arbitrary tie direction."""
    n = 8
    q_star = np.tile([1.0, 0.0], (n, 1))
    qbar = np.tile([1.0, 0.0], (n, 1))
    qbar[n // 2 :] = [0.0, 1.0]
    got = db.incorrect_argmax_flagging(_samples(qbar, np.full((n, 2), 0.1)), q_star)
    assert got.value == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# 4. Optimal-path uncertainty
# --------------------------------------------------------------------------- #
def test_optimal_path_reads_the_diagonal_and_Q_stars_action(setup):
    """One state per depth: the always-right path is (r, r). a*(s) comes from Q*, not Q̄."""
    env, idx, q_flat = setup
    S, _ = q_flat.shape
    sigma = np.full((S, 2), 0.01)
    pos = {(int(r), int(c)): i for i, (r, c) in enumerate(idx)}
    q = env.q_star(1.0)
    want = []
    for r in range(SIZE):
        i = pos[(r, r)]
        a_star = int(np.argmax(q[r, r]))
        sigma[i, a_star] = 0.1 * (r + 1)  # a known depth profile
        want.append(0.1 * (r + 1))
    got = db.optimal_path_uncertainty(_samples(q_flat, sigma), q_flat, idx, SIZE)
    assert got.n_used == SIZE
    assert got.extra["per_depth"] == pytest.approx(want)
    assert got.value == pytest.approx(np.trapezoid(want, dx=1.0) / (SIZE - 1))


def test_optimal_path_takes_a_star_from_Q_star_not_from_Qbar(setup):
    """a*(s) is Q*'s optimal action, not the agent's current greedy action.

    Constructed so the agent's Q̄ prefers the OPPOSITE action at every diagonal state. The
    diagnostic asks how uncertain the estimator is along the *optimal* path — a fixed set of
    (s, a) pairs — so reading a* from Q̄ would measure uncertainty along the agent's own
    (possibly wrong) path and would drift as the agent learned. Samples centred on Q* cannot
    detect this, because then the two argmaxes coincide; this gap was found by mutation testing.
    """
    env, idx, q_flat = setup
    S, _ = q_flat.shape
    pos = {(int(r), int(c)): i for i, (r, c) in enumerate(idx)}
    q = env.q_star(1.0)

    qbar = -q_flat  # reverses every non-tied argmax, diagonal included
    sigma = np.full((S, 2), 0.01)
    want = []
    for r in range(SIZE):
        i = pos[(r, r)]
        a_star = int(np.argmax(q[r, r]))
        assert int(np.argmax(qbar[i])) != a_star, "fixture must invert the diagonal argmax"
        sigma[i, a_star] = 0.1 * (r + 1)
        sigma[i, 1 - a_star] = 0.001  # a Q̄-based reading would pick this up instead
        want.append(0.1 * (r + 1))
    got = db.optimal_path_uncertainty(_samples(qbar, sigma), q_flat, idx, SIZE)
    assert got.extra["per_depth"] == pytest.approx(want)


def test_optimal_path_ignores_off_path_states(setup):
    """Only diagonal states enter; noise elsewhere must not move the summary."""
    env, idx, q_flat = setup
    S, _ = q_flat.shape
    sigma = np.full((S, 2), 0.05)
    base = db.optimal_path_uncertainty(_samples(q_flat, sigma), q_flat, idx, SIZE).value
    off = np.asarray([i for i, (r, c) in enumerate(idx) if r != c])
    sigma[off] = 9.0
    got = db.optimal_path_uncertainty(_samples(q_flat, sigma), q_flat, idx, SIZE)
    assert got.value == pytest.approx(base)


# --------------------------------------------------------------------------- #
# 5. Visitation-conditioned decay
# --------------------------------------------------------------------------- #
def test_visitation_decay_recovers_a_planted_slope(setup):
    """σ(s,a*) = exp(b · log(1+v)) ⇒ the OLS slope is exactly b."""
    env, idx, q_flat = setup
    S, _ = q_flat.shape
    v = np.arange(1, S + 1, dtype=float)
    b = -0.7
    sigma = np.full((S, 2), 0.05)
    a_star = np.argmax(q_flat, axis=1)
    sigma[np.arange(S), a_star] = np.exp(b * np.log1p(v))
    got = db.visitation_conditioned_decay(_samples(q_flat, sigma), q_flat, v)
    assert got.value == pytest.approx(b)
    assert got.n_excluded == 0


def test_visitation_decay_excludes_and_COUNTS_zero_sigma(setup):
    """The one place σ = 0 is excluded (log 0 undefined) — and item 20 requires it be counted."""
    env, idx, q_flat = setup
    S, _ = q_flat.shape
    v = np.arange(1, S + 1, dtype=float)
    sigma = np.full((S, 2), 0.3)
    a_star = np.argmax(q_flat, axis=1)
    sigma[np.arange(S), a_star] = np.exp(-0.5 * np.log1p(v))
    sigma[np.arange(3), a_star[:3]] = 0.0  # three states with zero spread
    got = db.visitation_conditioned_decay(_samples(q_flat, sigma), q_flat, v)
    assert got.n_excluded == 3
    assert got.n_used == S - 3
    assert np.isfinite(got.value)


def test_visitation_decay_is_unweighted_and_unbinned(setup):
    """"Raw probe states, unweighted; bins for display only."

    Duplicating a high-visitation state must change an unweighted fit (it changes the design
    matrix) -- this pins that the fit is over raw states rather than visitation-weighted or
    pre-binned, either of which would make the duplicate a no-op.
    """
    env, idx, q_flat = setup
    S, _ = q_flat.shape
    v = np.arange(1, S + 1, dtype=float)
    sigma = np.full((S, 2), 0.3)
    a_star = np.argmax(q_flat, axis=1)
    sigma[np.arange(S), a_star] = np.exp(-0.5 * np.log1p(v)) * np.linspace(0.5, 1.5, S)
    samples = _samples(q_flat, sigma)
    plain = db.visitation_conditioned_decay(samples, q_flat, v)
    dup = db.visitation_conditioned_decay(
        np.concatenate([samples, samples[-1:]]),
        np.concatenate([q_flat, q_flat[-1:]]),
        np.concatenate([v, v[-1:]]),
    )
    assert plain.value != pytest.approx(dup.value)


# --------------------------------------------------------------------------- #
# 7. Empirical containment
# --------------------------------------------------------------------------- #
def test_containment_is_1_when_Q_star_is_the_sample_centre(setup):
    _env, idx, q_flat = setup
    S, _ = q_flat.shape
    got = db.empirical_containment(_samples(q_flat, np.full((S, 2), 0.2)), q_flat)
    assert got.value == pytest.approx(1.0)
    assert got.extra["nominal_level"] == 0.80


def test_containment_is_0_when_every_sample_is_far_from_Q_star(setup):
    _env, idx, q_flat = setup
    S, _ = q_flat.shape
    got = db.empirical_containment(
        _samples(q_flat + 50.0, np.full((S, 2), 0.01)), q_flat
    )
    assert got.value == pytest.approx(0.0)


def test_containment_uses_linear_quantiles_explicitly(setup):
    """The frozen definition names ``method="linear"``; it is passed, not left to the default."""
    src = (db.__file__).replace(".pyc", ".py")
    text = open(src).read()
    assert text.count('method="linear"') >= 2


# --------------------------------------------------------------------------- #
# Battery driver + item-20 accounting
# --------------------------------------------------------------------------- #
def test_run_battery_returns_all_six_in_frozen_order(setup):
    _env, idx, q_flat = setup
    S, _ = q_flat.shape
    res = db.run_battery(
        _samples(q_flat, np.full((S, 2), 0.2)), q_flat, idx, SIZE, np.arange(1, S + 1)
    )
    assert tuple(res) == db.BATTERY
    assert len(db.BATTERY) == 6


def test_missing_visitation_yields_NA_for_diagnostic_5_only(setup):
    """A run without visitation counts loses diagnostic 5 and nothing else."""
    _env, idx, q_flat = setup
    S, _ = q_flat.shape
    sigma = np.linspace(0.05, 0.5, S)[:, None] * np.ones((1, 2))
    qbar = q_flat.copy()
    qbar[::3] = qbar[::3, ::-1]  # flip some argmaxes so diagnostic 3 has both classes
    res = db.run_battery(_samples(qbar, sigma), q_flat, idx, SIZE, None)
    assert not res["visitation_conditioned_decay"].defined
    assert "no visitation" in res["visitation_conditioned_decay"].reason
    for name in db.BATTERY:
        if name != "visitation_conditioned_decay":
            assert res[name].defined, name


def test_cli_writes_a_record_carrying_the_mapping_hash(tmp_path, setup):
    """The record must let a reader verify Q* matched the run's own action mapping."""
    env, idx, q_flat = setup
    S, _ = q_flat.shape
    samples = _samples(q_flat, np.full((S, 2), 0.2))
    npz = tmp_path / "r.value_samples.npz"
    np.savez_compressed(
        npz,
        steps=np.array([100, 200], dtype=np.int64),
        samples=np.stack([samples, samples]),
        visitation=np.stack([np.arange(1, S + 1)] * 2),
    )
    out = tmp_path / "battery.json"
    rc = subprocess.run(
        [
            sys.executable, "-m", "analysis.diagnostics_battery", str(npz),
            "--size", str(SIZE), "--master-seed", "0", "--cell-id", "c01",
            "--seed-index", "0", "--out", str(out),
        ],
        capture_output=True, text=True,
    )
    assert rc.returncode == 0, rc.stderr
    rec = json.loads(out.read_text())
    assert rec["mapping_hash"] == env.mapping_hash
    assert rec["n_checkpoints"] == 2
    assert tuple(rec["checkpoints"][0]) == ("step",) + db.BATTERY
