"""Sliced-W1 estimator and its confidence interval.

The regression these guard is F12: the shipped CI re-ran the estimator with new
projection seeds and never resampled the data, so its width shrank like P^(-1/2)
without bound while the estimand's uncertainty plateaued, and it covered the true
value in 0 of 15 trials at a nominal 95%.
"""
from __future__ import annotations

import numpy as np
import pytest

from drift.wasserstein import _sliced_w1_on, _unit_directions, sliced_w1, sliced_w1_fast


def test_null_is_smaller_than_shift(same_dist, shifted):
    assert sliced_w1_fast(*same_dist) < sliced_w1_fast(*shifted)


def test_identical_inputs_are_not_treated_as_the_null(rng):
    """X vs X.copy() is degenerate and must not be mistaken for a calibrated null.

    The previous test suite compared X to X.copy() and asserted the distance was
    near zero, which is true and tells you nothing: it is the same sample, not a
    second draw. A genuine two-sample null is strictly positive at finite n.
    """
    X = rng.standard_normal((200, 16))
    Y = rng.standard_normal((200, 16))
    assert sliced_w1_fast(X, X.copy()) == pytest.approx(0.0, abs=1e-12)
    assert sliced_w1_fast(X, Y) > 1e-3


def test_handles_unequal_window_sizes(rng):
    X = rng.standard_normal((300, 16))
    Y = rng.standard_normal((137, 16))
    assert np.isfinite(sliced_w1_fast(X, Y))


def test_ci_width_does_not_shrink_with_projections(rng):
    """F12 regression: the CI is a data bootstrap, so projections must not narrow it."""
    X = rng.standard_normal((250, 16))
    Y = rng.standard_normal((250, 16))

    widths, mc = {}, {}
    for p in (32, 512):
        r = sliced_w1(X, Y, n_projections=p, n_bootstrap=80, seed=3)
        widths[p], mc[p] = r.ci_hi - r.ci_lo, r.mc_se

    # slicing error does fall with more projections ...
    assert mc[512] < 0.6 * mc[32]
    # ... but the reported interval reflects sampling uncertainty and stays put
    assert widths[512] > 0.5 * widths[32]


def test_point_estimate_lies_inside_its_own_interval(rng):
    X = rng.standard_normal((200, 16))
    Y = rng.standard_normal((200, 16)) + 0.2
    r = sliced_w1(X, Y, n_projections=128, n_bootstrap=80, seed=5)
    assert r.ci_lo <= r.w1 <= r.ci_hi


@pytest.mark.slow
def test_interval_covers_the_estimand_most_of_the_time(rng):
    """Coverage is ~90% at a nominal 95%; the shortfall is documented, not tuned away.

    Asserted loosely (>= 70%) so this is a guard against the estimator regressing to
    the 48-68% of the percentile and pivotal variants, not a precise coverage claim.
    """
    d, n, p = 16, 200, 128
    theta = _unit_directions(d, p, np.random.default_rng(11))
    truth = float(
        np.mean([
            _sliced_w1_on(rng.standard_normal((n, d)), rng.standard_normal((n, d)) + 0.3, theta, n)
            for _ in range(60)
        ])
    )

    covered = 0
    trials = 25
    for t in range(trials):
        X = rng.standard_normal((n, d))
        Y = rng.standard_normal((n, d)) + 0.3
        r = sliced_w1(X, Y, n_projections=p, n_bootstrap=80, seed=100 + t)
        covered += r.ci_lo <= truth <= r.ci_hi

    assert covered / trials >= 0.70, f"coverage collapsed to {covered}/{trials}"


def test_rejects_mismatched_dimensions(rng):
    with pytest.raises(ValueError, match="dimension mismatch"):
        sliced_w1(rng.standard_normal((10, 4)), rng.standard_normal((10, 5)))


def test_rejects_empty_input(rng):
    with pytest.raises(ValueError):
        sliced_w1(np.zeros((0, 4)), rng.standard_normal((10, 4)))
