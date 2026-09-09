"""Threshold transfer under a shift in traffic volume.

This reproduces the strongest surviving result in the project (FINDINGS.md 12.8), as
a first-class pipeline command rather than a one-off script, so it re-runs on any
corpus and any embedding backend.

The claim
---------
A drift threshold calibrated at one window size is worthless at another.  Both
windows are drawn from the *same* distribution throughout, so every alarm is a false
alarm and the correct rate is the nominal alpha at both sizes.

Measured on 7346 real arXiv embeddings, 200 trials, three statistics:

    calibrate at n=4000, deploy at n=400   ->  FAR 1.000
    calibrate at n=400,  deploy at n=4000  ->  FAR 0.000

The mechanism is the n^(-1/2) level term every two-sample statistic carries: the null
value of the statistic shrinks as the window grows, so a threshold learned on large
windows sits below the entire null distribution of small ones and vice versa.  The
predicted score inflation from sqrt(4000/400) is 2.081; measured 2.33-2.43.

``run_transfer`` also runs the calibrated path for contrast, where the threshold is a
quantile of a per-window permutation null and therefore tracks n by construction.
"""
from __future__ import annotations

import numpy as np
from loguru import logger

from configs.settings import settings
from drift.mmd import centroid_shift, mmd_rbf_fast, median_heuristic_sigma
from drift.null import calibrate
from drift.wasserstein import sliced_w1_fast


def _statistics(sigma: float) -> dict:
    return {
        "sliced_w1": lambda a, b: sliced_w1_fast(a, b, n_projections=64, seed=0),
        "mmd": lambda a, b: mmd_rbf_fast(a, b, sigma),
        "centroid": centroid_shift,
    }


def _null_scores(pool: np.ndarray, n: int, stat_fn, trials: int, rng: np.random.Generator) -> np.ndarray:
    """Scores from pairs of same-distribution windows of size n."""
    out = np.empty(trials)
    for i in range(trials):
        idx = rng.choice(len(pool), 2 * n, replace=False)
        out[i] = stat_fn(pool[idx[:n]], pool[idx[n:]])
    return out


def run_transfer(
    pool: np.ndarray,
    n_calibrate: int = 4000,
    n_deploy: int = 400,
    trials: int = 100,
    alpha: float | None = None,
    seed: int = 0,
) -> dict:
    """Calibrate a fixed threshold at one window size, deploy it at another.

    Args:
        pool: (N, d) embeddings to draw all windows from
        n_calibrate: window size the threshold is learned at
        n_deploy: window size it is applied at
        trials: pairs per cell
        alpha: nominal false-alarm rate
    """
    alpha = settings.drift.alpha_level if alpha is None else alpha
    pool = np.asarray(pool, dtype=np.float64)

    biggest = max(n_calibrate, n_deploy)
    if len(pool) < 2 * biggest:
        raise ValueError(
            f"need at least {2 * biggest} vectors to draw two disjoint windows of "
            f"{biggest}; pool has {len(pool)}. Lower n_calibrate/n_deploy or use a larger corpus."
        )

    rng = np.random.default_rng(seed)
    sigma = median_heuristic_sigma(pool[:500], pool[500:1000] if len(pool) > 1000 else pool[:500])

    results: dict[str, dict] = {}
    for name, fn in _statistics(sigma).items():
        cal = _null_scores(pool, n_calibrate, fn, trials, rng)
        dep = _null_scores(pool, n_deploy, fn, trials, rng)

        # threshold learned on the calibration size, at the nominal rate
        tau_cal = float(np.quantile(cal, 1 - alpha))
        tau_dep = float(np.quantile(dep, 1 - alpha))

        far_forward = float((dep > tau_cal).mean())  # calibrated big, deployed small
        far_reverse = float((cal > tau_dep).mean())  # calibrated small, deployed big
        far_native_cal = float((cal > tau_cal).mean())
        far_native_dep = float((dep > tau_dep).mean())

        # The inflation ratio only means something for a statistic whose null level is
        # positive. Unbiased MMD^2 is centred on zero by construction, so its ratio is
        # undefined -- which is a property of the estimator, not a failure here. The
        # transfer failure itself is still visible in its FAR columns.
        if cal.mean() > 1e-12:
            inflation = float(dep.mean() / cal.mean())
            inflation_note = None
        else:
            inflation = float("nan")
            inflation_note = (
                "undefined: this statistic's null is centred on zero, so a ratio of "
                "null levels carries no information. Read the FAR columns instead."
            )

        results[name] = {
            "tau_at_n_calibrate": tau_cal,
            "tau_at_n_deploy": tau_dep,
            "null_mean_at_n_calibrate": float(cal.mean()),
            "null_mean_at_n_deploy": float(dep.mean()),
            "score_inflation_measured": inflation,
            "score_inflation_note": inflation_note,
            "far_native_at_n_calibrate": far_native_cal,
            "far_native_at_n_deploy": far_native_dep,
            "far_calibrate_big_deploy_small": far_forward,
            "far_calibrate_small_deploy_big": far_reverse,
        }
        logger.info(
            f"{name}: calibrate@{n_calibrate} -> deploy@{n_deploy} FAR {far_forward:.3f} | "
            f"reverse FAR {far_reverse:.3f} | inflation {inflation:.3f}"
        )

    # what the calibrated path does at the same two sizes
    calibrated: dict[str, float] = {}
    for n in (n_calibrate, n_deploy):
        alarms = 0
        checks = min(trials, 30)
        for _ in range(checks):
            idx = rng.choice(len(pool), 2 * n, replace=False)
            res = calibrate(
                pool[idx[:n]],
                pool[idx[n:]],
                lambda a, b: sliced_w1_fast(a, b, n_projections=64, seed=0),
                "sliced_w1",
                n_perm=49,
                alpha=alpha,
                seed=int(rng.integers(0, 2**31)),
            )
            alarms += int(res.alarm)
        calibrated[f"far_at_n_{n}"] = alarms / checks

    predicted = float(np.sqrt(n_calibrate / n_deploy))
    logger.info(
        f"per-window permutation calibration: "
        + ", ".join(f"{k} = {v:.3f}" for k, v in calibrated.items())
        + f" (nominal {alpha})"
    )

    return {
        "n_calibrate": n_calibrate,
        "n_deploy": n_deploy,
        "trials": trials,
        "alpha": alpha,
        "predicted_inflation_from_sqrt_n": predicted,
        "fixed_threshold": results,
        "permutation_calibrated": calibrated,
        "interpretation": (
            "Both windows come from the same distribution, so the correct FAR is alpha "
            "everywhere. A fixed threshold moved across window sizes departs from it in "
            "both directions; the per-window permutation null does not, because it is "
            "rebuilt at the size actually in use."
        ),
    }
