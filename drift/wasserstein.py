"""Sliced Wasserstein-1 distance with an honest confidence interval.

Two sources of uncertainty are reported separately, because conflating them was a
real defect in the previous version (FINDINGS.md F12):

``ci_lo`` / ``ci_hi``
    Bootstrap over the *data*: X and Y are resampled with replacement at their own
    sizes and the estimator is recomputed on the SAME projection directions.  This
    is the sampling uncertainty of the estimand and it does NOT shrink as the number
    of projections grows.

``mc_se``
    Monte-Carlo error of the slicing: the data are held fixed and only the projection
    directions are redrawn.  This one does shrink like P^(-1/2).

The old code reported the second and called it the first, so the interval narrowed
without bound while the true uncertainty plateaued, and it covered the true value in
0 of 15 trials at a nominal 95%.

Pure numpy; POT is not required.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from configs.settings import settings


@dataclass
class W1Result:
    w1: float
    ci_lo: float
    ci_hi: float
    mc_se: float
    n_projections: int
    n_bootstrap: int
    n_x: int
    n_y: int


def _unit_directions(dim: int, n_projections: int, rng: np.random.Generator) -> np.ndarray:
    theta = rng.standard_normal((dim, n_projections)).astype(np.float64)
    theta /= np.maximum(np.linalg.norm(theta, axis=0, keepdims=True), 1e-12)
    return theta


def _sliced_w1_on(X: np.ndarray, Y: np.ndarray, theta: np.ndarray, n_grid: int) -> float:
    """Mean over projections of the 1-D W1, computed on a shared quantile grid.

    Comparing sorted projections directly only works when len(X) == len(Y); the grid
    makes unequal window sizes well defined without silently truncating either side.
    """
    px = np.sort(X @ theta, axis=0)
    py = np.sort(Y @ theta, axis=0)

    levels = (np.arange(n_grid) + 0.5) / n_grid
    qx = np.quantile(px, levels, axis=0)
    qy = np.quantile(py, levels, axis=0)
    return float(np.abs(qx - qy).mean())


def sliced_w1(
    X: np.ndarray,
    Y: np.ndarray,
    n_projections: int | None = None,
    n_bootstrap: int | None = None,
    seed: int = 42,
) -> W1Result:
    """Sliced Wasserstein-1 between two embedding matrices.

    Args:
        X: (n, d) embeddings from P
        Y: (m, d) embeddings from Q
        n_projections: number of random 1-D projections
        n_bootstrap: bootstrap resamples for the data CI (0 disables)
        seed: RNG seed

    Returns:
        W1Result. Note ``ci_*`` is a bootstrap interval over the data and ``mc_se``
        is the separate slicing error.
    """
    n_projections = n_projections or settings.drift.w1_n_projections
    n_bootstrap = settings.drift.w1_bootstrap_iters if n_bootstrap is None else n_bootstrap

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2-D embedding matrices")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"dimension mismatch: X.d={X.shape[1]}, Y.d={Y.shape[1]}")
    if len(X) == 0 or len(Y) == 0:
        raise ValueError("sliced_w1 needs non-empty X and Y")

    rng = np.random.default_rng(seed)
    theta = _unit_directions(X.shape[1], n_projections, rng)
    n_grid = min(len(X), len(Y))

    point = _sliced_w1_on(X, Y, theta, n_grid)

    # Monte-Carlo error of the slicing: fixed data, fresh directions.
    mc_reps = min(32, max(4, n_projections // 16))
    mc_vals = np.array(
        [
            _sliced_w1_on(X, Y, _unit_directions(X.shape[1], n_projections, rng), n_grid)
            for _ in range(mc_reps)
        ]
    )
    mc_se = float(mc_vals.std(ddof=1))

    # Sampling uncertainty: resample the data, hold the directions fixed.
    #
    # The bootstrap is used as a standard-ERROR estimator only, and the interval is
    # theta_hat +/- 1.96 * se.  Neither the percentile nor the pivotal interval is
    # valid for this statistic, and the reason is measurable rather than a matter of
    # taste.  Against the true sampling distribution of the estimator at n=250, d=16:
    #
    #     bootstrap sd / true sd   = 0.91      <- spread is reproduced
    #     bootstrap mean - theta_hat = +0.027  <- centre is not (about 1.6 true SDs)
    #
    # A bootstrap resample keeps only ~63% distinct points, which coarsens both
    # empirical measures and pushes them apart, so the whole bootstrap distribution
    # sits above theta_hat.  A percentile interval inherits that shift; a pivotal one
    # reflects it to the wrong side and does even worse.  Measured coverage of the
    # estimand over 60 trials at a nominal 95%:
    #
    #     percentile  68%      pivotal  57%      theta_hat +/- 1.96 se   90%
    #
    # 90% is still short of nominal -- the residual is the 0.91 spread ratio -- and
    # it is left uncorrected rather than tuned away.  Treat this interval as
    # descriptive; the inference that carries weight is the permutation null in
    # drift/null.py.  tests/test_wasserstein.py pins both the ordering and the floor.
    if n_bootstrap and n_bootstrap > 0:
        boot = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            xb = X[rng.integers(0, len(X), len(X))]
            yb = Y[rng.integers(0, len(Y), len(Y))]
            boot[i] = _sliced_w1_on(xb, yb, theta, n_grid)
        se = float(boot.std(ddof=1))
        ci_lo = float(point - 1.96 * se)
        ci_hi = float(point + 1.96 * se)
    else:
        ci_lo = ci_hi = float("nan")

    return W1Result(
        w1=point,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        mc_se=mc_se,
        n_projections=n_projections,
        n_bootstrap=int(n_bootstrap or 0),
        n_x=len(X),
        n_y=len(Y),
    )


def sliced_w1_fast(
    X: np.ndarray, Y: np.ndarray, n_projections: int = 128, seed: int = 0
) -> float:
    """Point estimate only. For permutation nulls, where it is called thousands of times."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    theta = _unit_directions(X.shape[1], n_projections, rng)
    return _sliced_w1_on(X, Y, theta, min(len(X), len(Y)))
