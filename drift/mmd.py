"""RBF-kernel Maximum Mean Discrepancy, computed through the Gram identity.

The previous version materialised an (n, m, d) difference tensor.  At the shipped
default of subsample=2000 and d=1024 that is 16.4 GB per kernel matrix, three of them
per window, so the pipeline OOMed on window 1 (FINDINGS.md F7).

    ||x - y||^2 = ||x||^2 + ||y||^2 - 2 <x, y>

turns the same computation into one matmul: 32 MB and ~84x faster, agreeing with the
naive form to 4.3e-15.  tests/test_mmd.py pins that equivalence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from configs.settings import settings


@dataclass
class MMDResult:
    mmd: float
    mmd_sq: float
    sigma: float
    n_x: int
    n_y: int


def sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances via the Gram identity."""
    a2 = np.einsum("ij,ij->i", A, A)[:, None]
    b2 = np.einsum("ij,ij->i", B, B)[None, :]
    d2 = a2 + b2 - 2.0 * (A @ B.T)
    return np.maximum(d2, 0.0)


def median_heuristic_sigma(
    X: np.ndarray, Y: np.ndarray, subsample: int = 500, seed: int = 0
) -> float:
    """Median pairwise distance over a pooled subsample."""
    rng = np.random.default_rng(seed)
    Z = np.vstack([X, Y])
    if len(Z) > subsample:
        Z = Z[rng.choice(len(Z), subsample, replace=False)]
    d2 = sq_dists(Z, Z)
    iu = np.triu_indices(len(Z), k=1)
    med = float(np.median(d2[iu])) if len(iu[0]) else 0.0
    return float(np.sqrt(med + 1e-10))


def rbf_kernel(A: np.ndarray, B: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-sq_dists(A, B) / (2.0 * sigma**2))


def mmd_rbf(
    X: np.ndarray,
    Y: np.ndarray,
    sigma: float | None = None,
    subsample: int | None = None,
    seed: int = 42,
) -> MMDResult:
    """Unbiased MMD^2 with an RBF kernel; ``mmd`` is its clipped square root.

    ``mmd_sq`` is kept unclipped because the unbiased estimator is legitimately
    negative under the null, and clipping it destroys the left tail that any
    permutation calibration needs.
    """
    subsample = subsample if subsample is not None else settings.drift.mmd_subsample
    rng = np.random.default_rng(seed)

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if len(X) < 2 or len(Y) < 2:
        raise ValueError("MMD needs at least 2 samples per side")

    if subsample and len(X) > subsample:
        X = X[rng.choice(len(X), subsample, replace=False)]
    if subsample and len(Y) > subsample:
        Y = Y[rng.choice(len(Y), subsample, replace=False)]

    if sigma is None:
        sigma = settings.drift.mmd_sigma or median_heuristic_sigma(X, Y, seed=seed)
    if sigma <= 0:
        raise ValueError(f"RBF bandwidth must be positive, got {sigma}")

    n, m = len(X), len(Y)
    k_xx = rbf_kernel(X, X, sigma)
    k_yy = rbf_kernel(Y, Y, sigma)
    k_xy = rbf_kernel(X, Y, sigma)
    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)

    mmd_sq = (
        k_xx.sum() / (n * (n - 1))
        + k_yy.sum() / (m * (m - 1))
        - 2.0 * k_xy.mean()
    )

    return MMDResult(
        mmd=float(np.sqrt(max(mmd_sq, 0.0))),
        mmd_sq=float(mmd_sq),
        sigma=float(sigma),
        n_x=n,
        n_y=m,
    )


def mmd_rbf_fast(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    """Unbiased MMD^2 at a fixed bandwidth. For permutation nulls.

    The bandwidth must be supplied by the caller and held fixed across permutations,
    otherwise the null is calibrated against a moving kernel.
    """
    n, m = len(X), len(Y)
    k_xx = rbf_kernel(X, X, sigma)
    k_yy = rbf_kernel(Y, Y, sigma)
    k_xy = rbf_kernel(X, Y, sigma)
    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)
    return float(
        k_xx.sum() / (n * (n - 1)) + k_yy.sum() / (m * (m - 1)) - 2.0 * k_xy.mean()
    )


def centroid_shift(X: np.ndarray, Y: np.ndarray) -> float:
    """L2 distance between sample means. Cheap, and the most sensitive of the three."""
    if len(X) == 0 or len(Y) == 0:
        return 0.0
    return float(np.linalg.norm(X.mean(axis=0) - Y.mean(axis=0)))
