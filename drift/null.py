"""Permutation-null calibration for two-sample drift statistics.

Why this module exists
----------------------
The previous design compared a raw two-sample statistic against a fixed global
threshold (tau = 0.15).  Two measured facts make that indefensible:

1. Every two-sample statistic here carries an n^(-1/2) level term even when the two
   windows are drawn from the *same* distribution.  Measured null sliced-W1 falls
   0.001975 -> 0.000523 as n goes 300 -> 4800, a log-log slope of -0.47 against a
   theoretical -0.5 (FINDINGS.md F13).  A fixed threshold is therefore partly a
   measurement of traffic volume.

2. Consequently a threshold calibrated at one window size is worthless at another.
   Calibrate at n=4000 and deploy at n=400 and the false-alarm rate is 1.000;
   calibrate at n=400 and deploy at n=4000 and it is 0.000 -- 200 trials, three
   statistics, no exceptions (FINDINGS.md 12.8).

The fix is to stop thresholding the raw statistic.  For each window we build the
null by repeatedly re-splitting the pooled two windows at their own sizes, which
holds n fixed by construction, and report:

    z  = (observed - null_mean) / null_sd
    p  = permutation p-value
    threshold = the (1 - alpha) quantile of that null

``z`` and ``p`` are dimensionless and comparable across window sizes; the raw
statistic is not.  Everything downstream keys off ``z``.

This is calibration, not a cure: 12.9 shows that at the effect sizes real corpora
actually exhibit, power at a correctly calibrated 5% false-alarm rate is 0.000-0.100.
Calibrating the threshold makes the detector honest, not sensitive.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

StatFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass
class NullCalibration:
    """Result of calibrating one statistic on one pair of windows."""

    statistic: str
    observed: float
    null_mean: float
    null_sd: float
    threshold: float
    z: float
    p_value: float
    alarm: bool
    n_perm: int
    n_x: int
    n_y: int
    alpha: float

    def as_dict(self, prefix: str = "") -> dict:
        return {f"{prefix}{k}": v for k, v in asdict(self).items()}


def permutation_null(
    X: np.ndarray,
    Y: np.ndarray,
    stat_fn: StatFn,
    n_perm: int = 199,
    seed: int = 0,
) -> np.ndarray:
    """Null distribution of ``stat_fn`` under exchangeability of X and Y.

    Pools the two windows and re-splits at the original sizes, so the null is built
    at exactly the n and m of the observed comparison. That is the whole point: it
    is what makes the resulting z-score comparable across window sizes.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    n = len(X)
    pooled = np.vstack([X, Y])
    rng = np.random.default_rng(seed)

    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        perm = rng.permutation(len(pooled))
        null[i] = stat_fn(pooled[perm[:n]], pooled[perm[n:]])
    return null


def calibrate(
    X: np.ndarray,
    Y: np.ndarray,
    stat_fn: StatFn,
    name: str = "statistic",
    n_perm: int = 199,
    alpha: float = 0.05,
    seed: int = 0,
) -> NullCalibration:
    """Calibrate one statistic against its own permutation null."""
    observed = float(stat_fn(X, Y))
    null = permutation_null(X, Y, stat_fn, n_perm=n_perm, seed=seed)

    null_mean = float(null.mean())
    null_sd = float(null.std(ddof=1)) if n_perm > 1 else 0.0
    threshold = float(np.quantile(null, 1.0 - alpha))

    z = (observed - null_mean) / null_sd if null_sd > 1e-15 else 0.0
    # +1 in numerator and denominator: the observed value is one of the possible
    # arrangements, so a p-value of exactly 0 is not attainable.
    p_value = float((1 + int((null >= observed).sum())) / (n_perm + 1))

    return NullCalibration(
        statistic=name,
        observed=observed,
        null_mean=null_mean,
        null_sd=null_sd,
        threshold=threshold,
        z=float(z),
        p_value=p_value,
        alarm=bool(observed > threshold),
        n_perm=n_perm,
        n_x=len(X),
        n_y=len(Y),
        alpha=alpha,
    )


def null_scaling(
    pool: np.ndarray,
    stat_fn: StatFn,
    sizes: tuple[int, ...] = (300, 600, 1200, 2400, 4800),
    n_trials: int = 20,
    seed: int = 0,
) -> dict:
    """Measure how the null level of a statistic scales with window size.

    Draws both windows from the same pool, so any signal is pure sample-size effect.
    Returns the per-size mean and the fitted log-log slope, which should sit near
    -0.5 for a two-sample statistic. Used by evaluation/transfer.py to show *why*
    a fixed threshold cannot transfer.
    """
    pool = np.asarray(pool)
    rng = np.random.default_rng(seed)
    means: dict[int, float] = {}

    for n in sizes:
        if 2 * n > len(pool):
            continue
        vals = []
        for _ in range(n_trials):
            idx = rng.choice(len(pool), 2 * n, replace=False)
            vals.append(stat_fn(pool[idx[:n]], pool[idx[n:]]))
        means[n] = float(np.mean(vals))

    if len(means) >= 2:
        xs = np.log(np.array(sorted(means)))
        ys = np.log(np.array([means[n] for n in sorted(means)]))
        slope = float(np.polyfit(xs, ys, 1)[0])
    else:
        slope = float("nan")

    return {"null_mean_by_n": means, "log_log_slope": slope, "n_trials": n_trials}
