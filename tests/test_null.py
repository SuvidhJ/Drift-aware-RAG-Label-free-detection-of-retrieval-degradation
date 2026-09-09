"""Permutation-null calibration.

These are the tests whose absence let F13 through: the suite never checked what a
two-sample statistic reads when *nothing* has changed, so nobody noticed that the
raw statistic carries an n^(-1/2) level term and that tau = 0.15 sat 75x above the
largest null value ever observed.
"""
from __future__ import annotations

import numpy as np
import pytest

from drift.null import calibrate, null_scaling, permutation_null
from drift.wasserstein import sliced_w1_fast

W1 = lambda a, b: sliced_w1_fast(a, b, n_projections=48, seed=0)  # noqa: E731


def test_z_is_near_zero_under_exchangeability(same_dist):
    res = calibrate(*same_dist, W1, "w1", n_perm=199, seed=1)
    assert abs(res.z) < 3.0
    assert res.p_value > 0.01
    assert not res.alarm


def test_z_is_large_under_a_real_shift(shifted):
    res = calibrate(*shifted, W1, "w1", n_perm=199, seed=1)
    assert res.z > 5.0
    assert res.p_value <= 0.01
    assert res.alarm


def test_p_value_cannot_be_exactly_zero(shifted):
    """The observed arrangement is one of the possibilities, so p >= 1/(n_perm+1)."""
    res = calibrate(*shifted, W1, "w1", n_perm=99, seed=2)
    assert res.p_value >= 1 / 100


def test_false_alarm_rate_is_near_alpha(rng):
    """The point of calibrating: repeated null pairs alarm at about alpha."""
    alarms = 0
    trials = 40
    for t in range(trials):
        X = rng.standard_normal((150, 12))
        Y = rng.standard_normal((150, 12))
        alarms += calibrate(X, Y, W1, "w1", n_perm=99, alpha=0.05, seed=t).alarm
    assert alarms / trials <= 0.25, f"null alarmed {alarms}/{trials} times"


def test_threshold_tracks_window_size(rng):
    """The calibrated threshold must fall as n grows; a fixed tau cannot.

    This is the mechanism behind the transfer failure in FINDINGS 12.8, asserted
    directly: the null level of the statistic is a function of n.
    """
    pool = rng.standard_normal((4000, 12))
    small = calibrate(pool[:200], pool[200:400], W1, "w1", n_perm=99, seed=7)
    large = calibrate(pool[:1500], pool[1500:3000], W1, "w1", n_perm=99, seed=7)
    assert large.threshold < small.threshold


def test_permutation_null_preserves_window_sizes(same_dist):
    X, Y = same_dist
    null = permutation_null(X, Y, lambda a, b: float(len(a) * 1000 + len(b)), n_perm=5)
    assert np.all(null == len(X) * 1000 + len(Y))


@pytest.mark.slow
def test_null_level_scales_like_n_to_the_minus_half(rng):
    """F13: measured log-log slope was -0.47 against a theoretical -0.5."""
    pool = rng.standard_normal((6000, 12))
    res = null_scaling(pool, W1, sizes=(200, 400, 800, 1600), n_trials=8, seed=0)
    assert -0.75 < res["log_log_slope"] < -0.25, res
