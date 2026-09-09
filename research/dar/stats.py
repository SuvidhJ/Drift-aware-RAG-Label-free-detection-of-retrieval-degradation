"""
Core two-sample statistics + permutation calibration.

Fixes implemented here (vs the original repo):

  B4  MMD via the Gram identity  ||x-y||^2 = ||x||^2 + ||y||^2 - 2 <x,y>
      -> intermediate is (n x m), never (n x m x d).  The repo's version
      allocates 2000*2000*1024*4 B = 16 GB and OOMs.

  B5  bootstrap resamples the *data* (rows of X, Y), not the projection seed.
      The repo's "CI" only measured Monte-Carlo error of the slicing.

  S1  permutation_null(): the object the project never had.  Pool X and Y,
      re-split at the observed sizes, recompute the statistic.  This is what
      "no drift, at these exact sample sizes" looks like.  Everything the
      detector reports downstream is standardised against it.

  S4  sliced_w1_max(): optimise the projection direction instead of averaging
      random ones.  Random slices concentrate (Johnson-Lindenstrauss); a
      low-dimensional shift (a product line vanishing) is intersected by a
      random direction with probability ~ k/d, so averaging *dilutes* the
      signal.  Max-slice also returns the drift direction, which is decodable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ot


# --------------------------------------------------------------------------- #
# subsampling
# --------------------------------------------------------------------------- #
def _sub(A: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if A.shape[0] <= n:
        return A
    idx = rng.choice(A.shape[0], n, replace=False)
    return A[idx]


# --------------------------------------------------------------------------- #
# sliced Wasserstein-1
# --------------------------------------------------------------------------- #
def sliced_w1_mean(X: np.ndarray, Y: np.ndarray, n_proj: int = 128,
                   rng: np.random.Generator | None = None) -> float:
    rng = rng or np.random.default_rng(0)
    X = np.ascontiguousarray(X, dtype=np.float64)
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    return float(ot.sliced_wasserstein_distance(
        X, Y, n_projections=n_proj, p=1, seed=int(rng.integers(0, 2**31 - 1))
    ))


_QGRID_M = 200
_QGRID = (np.arange(_QGRID_M) + 0.5) / _QGRID_M


def _w1_1d_batch(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Vectorised 1-D W1 across columns. px:(nx,P) py:(ny,P) -> (P,) L1 quantile dist."""
    qx = np.quantile(px, _QGRID, axis=0)      # (M, P)
    qy = np.quantile(py, _QGRID, axis=0)      # (M, P)
    return np.abs(qx - qy).mean(axis=0)


def sliced_w1_max(X: np.ndarray, Y: np.ndarray, n_proj: int = 128,
                  rng: np.random.Generator | None = None,
                  return_dir: bool = False):
    """max_theta W1_1( <X,theta>, <Y,theta> ) over random unit theta.

    Power >> mean-slicing for low-rank shifts; the argmax direction is the
    'direction of maximum drift' and can be decoded to nearest documents.
    """
    rng = rng or np.random.default_rng(0)
    d = X.shape[1]
    thetas = rng.standard_normal((n_proj, d))
    thetas /= np.linalg.norm(thetas, axis=1, keepdims=True) + 1e-12
    w1_per_dir = _w1_1d_batch(X @ thetas.T, Y @ thetas.T)
    j = int(np.argmax(w1_per_dir))
    if return_dir:
        return float(w1_per_dir[j]), thetas[j]
    return float(w1_per_dir[j])


def sliced_w1_pair(X: np.ndarray, Y: np.ndarray, n_proj: int = 128,
                   rng: np.random.Generator | None = None) -> tuple[float, float]:
    """Return (mean-sliced, max-sliced) W1 from ONE set of projections (amortised)."""
    rng = rng or np.random.default_rng(0)
    d = X.shape[1]
    thetas = rng.standard_normal((n_proj, d))
    thetas /= np.linalg.norm(thetas, axis=1, keepdims=True) + 1e-12
    w = _w1_1d_batch(X @ thetas.T, Y @ thetas.T)
    return float(w.mean()), float(w.max())


# --------------------------------------------------------------------------- #
# MMD (Gram identity)  -- fixes B4
# --------------------------------------------------------------------------- #
def _sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    aa = np.einsum("ij,ij->i", A, A)[:, None]
    bb = np.einsum("ij,ij->i", B, B)[None, :]
    return np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)


def _median_sigma(Z: np.ndarray, rng: np.random.Generator, cap: int = 800) -> float:
    Z = _sub(Z, cap, rng)
    dsq = _sq_dists(Z, Z)
    iu = np.triu_indices(Z.shape[0], k=1)
    med = np.median(dsq[iu])
    return float(np.sqrt(med / 2.0 + 1e-12))  # sigma s.t. exp(-d^2/(2 sigma^2))


def mmd2_rbf(X: np.ndarray, Y: np.ndarray, sigma: float | None = None,
             rng: np.random.Generator | None = None) -> float:
    """Unbiased MMD^2, RBF kernel, O(n^2 + m^2 + nm) memory (no d factor)."""
    rng = rng or np.random.default_rng(0)
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    if sigma is None:
        sigma = _median_sigma(np.vstack([X, Y]), rng)
    g = 1.0 / (2.0 * sigma * sigma)
    Kxx = np.exp(-g * _sq_dists(X, X))
    Kyy = np.exp(-g * _sq_dists(Y, Y))
    Kxy = np.exp(-g * _sq_dists(X, Y))
    n, m = X.shape[0], Y.shape[0]
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    return float(
        Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1)) - 2.0 * Kxy.mean()
    )


# --------------------------------------------------------------------------- #
# centroid terms
# --------------------------------------------------------------------------- #
def centroid_gap(Q: np.ndarray, C: np.ndarray) -> float:
    """||mean(Q) - mean(C)||  -- a LEVEL.  The repo puts this in D(t) directly."""
    return float(np.linalg.norm(Q.mean(0) - C.mean(0)))


def centroid_gap_delta(Q_t, C_t, Q_p, C_p) -> float:
    """|gap_t - gap_{t-1}|  -- the CHANGE.  This is what belongs in a drift stat (B3)."""
    return abs(centroid_gap(Q_t, C_t) - centroid_gap(Q_p, C_p))


def centroid_shift(A_t: np.ndarray, A_p: np.ndarray) -> float:
    """||mean(A_t) - mean(A_{t-1})||  -- the 'centroid-only' baseline."""
    return float(np.linalg.norm(A_t.mean(0) - A_p.mean(0)))


# --------------------------------------------------------------------------- #
# permutation calibration  -- fixes S1 (and subsumes B5)
# --------------------------------------------------------------------------- #
@dataclass
class Calibrated:
    raw: float
    z: float           # (raw - null_mean) / null_sd
    p: float           # P_null( stat >= raw ), add-one smoothed
    null_mean: float
    null_sd: float


def _calib(raw: float, null: np.ndarray) -> Calibrated:
    mu, sd = float(null.mean()), float(null.std() + 1e-12)
    p = (1.0 + np.sum(null >= raw)) / (len(null) + 1.0)
    return Calibrated(raw=raw, z=(raw - mu) / sd, p=float(p), null_mean=mu, null_sd=sd)


def permutation_null(
    X: np.ndarray,
    Y: np.ndarray,
    B: int = 120,
    subsample: int = 1400,
    n_proj: int = 128,
    seed: int = 0,
    do_mmd: bool = True,
) -> dict[str, Calibrated]:
    """Pool X,Y; re-split at observed sizes B times.  Computes w1_mean, w1_max
    (shared projections) and mmd2 (fixed pooled sigma) on the SAME resamples."""
    rng = np.random.default_rng(seed)
    Xs = _sub(X, subsample, rng)
    Ys = _sub(Y, subsample, rng)
    nx = Xs.shape[0]
    pool = np.vstack([Xs, Ys]).astype(np.float64)
    N = pool.shape[0]
    d = pool.shape[1]

    sigma = _median_sigma(pool, rng) if do_mmd else 1.0
    g = 1.0 / (2.0 * sigma * sigma)

    def _stats(A, Bm, r):
        th = r.standard_normal((n_proj, d))
        th /= np.linalg.norm(th, axis=1, keepdims=True) + 1e-12
        w = _w1_1d_batch(A @ th.T, Bm @ th.T)
        if do_mmd:
            na, nb = A.shape[0], Bm.shape[0]
            Kxx = np.exp(-g * _sq_dists(A, A)); np.fill_diagonal(Kxx, 0.0)
            Kyy = np.exp(-g * _sq_dists(Bm, Bm)); np.fill_diagonal(Kyy, 0.0)
            Kxy = np.exp(-g * _sq_dists(A, Bm))
            m2 = Kxx.sum() / (na * (na - 1)) + Kyy.sum() / (nb * (nb - 1)) - 2.0 * Kxy.mean()
        else:
            m2 = 0.0
        return float(w.mean()), float(w.max()), float(m2)

    r0 = _stats(Xs, Ys, rng)
    nm = np.empty(B); nx_ = np.empty(B); nd = np.empty(B)
    for b in range(B):
        p = rng.permutation(N)
        a, bm = pool[p[:nx]], pool[p[nx:]]
        nm[b], nx_[b], nd[b] = _stats(a, bm, rng)

    return {
        "w1_mean": _calib(r0[0], nm),
        "w1_max": _calib(r0[1], nx_),
        "mmd2": _calib(r0[2], nd),
    }
