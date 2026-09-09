"""MMD via the Gram identity.

Guards F7: the shipped implementation materialised an (n, m, d) difference tensor,
16.4 GB per kernel matrix at the shipped defaults, and OOMed on the first window.
"""
from __future__ import annotations

import numpy as np
import pytest

from drift.mmd import (
    centroid_shift,
    mmd_rbf,
    mmd_rbf_fast,
    median_heuristic_sigma,
    rbf_kernel,
    sq_dists,
)


def _naive_kernel(A: np.ndarray, B: np.ndarray, sigma: float) -> np.ndarray:
    """The original (n, m, d) formulation, at a size small enough to be affordable."""
    d2 = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
    return np.exp(-d2 / (2 * sigma**2))


def test_gram_identity_matches_the_naive_form(rng):
    A = rng.standard_normal((60, 24))
    B = rng.standard_normal((50, 24))
    assert np.abs(rbf_kernel(A, B, 2.0) - _naive_kernel(A, B, 2.0)).max() < 1e-10


def test_squared_distances_are_non_negative(rng):
    """Floating point can make the identity return small negatives; they are clipped."""
    A = rng.standard_normal((40, 8))
    assert sq_dists(A, A).min() >= 0.0
    assert np.diag(sq_dists(A, A)) == pytest.approx(np.zeros(40), abs=1e-8)


def test_gram_path_allocates_a_matrix_not_a_tensor(rng):
    """(n, m) not (n, m, d): at d=1024 the difference is 32 MB against 16 GB."""
    A = rng.standard_normal((200, 256))
    B = rng.standard_normal((200, 256))
    assert rbf_kernel(A, B, 3.0).shape == (200, 200)


def test_null_is_near_zero_and_shift_is_detected(same_dist, shifted):
    assert mmd_rbf(*same_dist).mmd < 0.05
    assert mmd_rbf(*shifted).mmd > mmd_rbf(*same_dist).mmd


def test_unbiased_estimator_may_be_negative_under_the_null(same_dist):
    """mmd_sq is reported unclipped: a permutation null needs its left tail."""
    r = mmd_rbf(*same_dist)
    assert r.mmd >= 0.0
    assert r.mmd_sq < 0.05  # can and often does go below zero


def test_fast_path_agrees_with_the_full_one(rng):
    A = rng.standard_normal((80, 16))
    B = rng.standard_normal((80, 16)) + 0.5
    sigma = median_heuristic_sigma(A, B)
    full = mmd_rbf(A, B, sigma=sigma, subsample=None)
    assert mmd_rbf_fast(A, B, sigma) == pytest.approx(full.mmd_sq, rel=1e-9)


def test_rejects_degenerate_inputs(rng):
    with pytest.raises(ValueError, match="at least 2 samples"):
        mmd_rbf(rng.standard_normal((1, 4)), rng.standard_normal((10, 4)))
    with pytest.raises(ValueError, match="bandwidth"):
        mmd_rbf(rng.standard_normal((10, 4)), rng.standard_normal((10, 4)), sigma=0.0)


def test_centroid_shift(rng):
    A = rng.standard_normal((100, 8))
    assert centroid_shift(A, A) == pytest.approx(0.0, abs=1e-12)
    assert centroid_shift(A, A + 1.0) == pytest.approx(np.sqrt(8), rel=0.02)
    assert centroid_shift(np.zeros((0, 8)), A) == 0.0
