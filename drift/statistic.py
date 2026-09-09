"""The D(t) compositor.

What changed and why
--------------------
The previous definition was

    D(t) = a*W1(Q_t, Q_{t-d}) + b*W1(C_t, C_{t-d}) + g*||mu(Q_t) - mu(C_t)||

and the third term dominated it.  ``||mu(Q_t) - mu(C_t)||`` is the gap between two
*text genres* -- questions versus documents -- which is fixed at t=0 and is not a
change over time at all.  Its measured share of D(t) was 92.5% with no genre gap and
99.7% with a realistic one; the repo's own run recorded 0.8718.  The alarm rule
reduced to a constant decided before any drift occurred (FINDINGS.md F5).

So D(t) is now built only from the two terms that are actually temporal, and each is
expressed as a z-score against its own per-window permutation null:

    D(t) = w_q * z_q(t) + w_c * z_c(t)

The genre gap is still computed and reported as ``qc_genre_gap``, because it is a
useful diagnostic, but it no longer enters the alarm path.

Working in z units rather than raw distances is what makes D(t) comparable across
windows of different sizes -- see drift/null.py for the measurements that force this.
``d_raw`` retains the old unnormalised sum so evaluation/transfer.py can demonstrate
the failure it causes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from configs.settings import settings
from drift.mmd import centroid_shift, mmd_rbf
from drift.null import calibrate
from drift.wasserstein import sliced_w1, sliced_w1_fast


def _w1_stat_fn(n_projections: int, seed: int = 0):
    """A fixed-projection W1 closure, so every permutation sees the same directions."""

    def fn(a: np.ndarray, b: np.ndarray) -> float:
        return sliced_w1_fast(a, b, n_projections=n_projections, seed=seed)

    return fn


def compute_drift_series(
    corpus_embs: list[np.ndarray],
    query_embs: list[np.ndarray],
    lag: int | None = None,
    n_perm: int | None = None,
    alpha_level: float | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute the drift series with per-window null calibration.

    Args:
        corpus_embs: one (n_t, d) array per window
        query_embs: one (m_t, d) array per window
        lag: comparison lag in windows
        n_perm: permutations per window per channel
        alpha_level: false-alarm rate the per-window threshold targets

    Returns:
        One row per window t >= lag.
    """
    cfg = settings.drift
    lag = cfg.lag if lag is None else lag
    n_perm = cfg.n_perm if n_perm is None else n_perm
    alpha_level = cfg.alpha_level if alpha_level is None else alpha_level

    if len(query_embs) != len(corpus_embs):
        raise ValueError(
            f"corpus_embs and query_embs must have the same length; "
            f"got {len(corpus_embs)} and {len(query_embs)}"
        )

    stat_fn = _w1_stat_fn(cfg.null_n_projections, seed=seed)
    rows: list[dict] = []

    for t in range(lag, len(corpus_embs)):
        C_t, C_prev = corpus_embs[t], corpus_embs[t - lag]
        Q_t, Q_prev = query_embs[t], query_embs[t - lag]

        if min(len(C_t), len(C_prev), len(Q_t), len(Q_prev)) < 2:
            logger.warning(f"t={t}: a window has fewer than 2 samples; skipping")
            continue

        logger.info(f"t={t}: calibrating (C: {len(C_prev)}->{len(C_t)}, Q: {len(Q_prev)}->{len(Q_t)})")

        w1_q = sliced_w1(Q_t, Q_prev, seed=seed + t)
        w1_c = sliced_w1(C_t, C_prev, seed=seed + t)

        cal_q = calibrate(
            Q_t, Q_prev, stat_fn, "sliced_w1_query",
            n_perm=n_perm, alpha=alpha_level, seed=seed + t,
        )
        cal_c = calibrate(
            C_t, C_prev, stat_fn, "sliced_w1_corpus",
            n_perm=n_perm, alpha=alpha_level, seed=seed + 1000 + t,
        )

        mmd_q = mmd_rbf(Q_t, Q_prev, seed=seed + t)
        mmd_c = mmd_rbf(C_t, C_prev, seed=seed + t)

        d_z = cfg.w_query * cal_q.z + cfg.w_corpus * cal_c.z
        d_raw = w1_q.w1 + w1_c.w1 + 0.5 * centroid_shift(Q_t, C_t)

        rows.append(
            {
                "window": t,
                "n_query": len(Q_t),
                "n_corpus": len(C_t),
                # raw estimates, with an honest CI (data bootstrap) and slicing error
                "w1_q": w1_q.w1,
                "w1_q_ci_lo": w1_q.ci_lo,
                "w1_q_ci_hi": w1_q.ci_hi,
                "w1_q_mc_se": w1_q.mc_se,
                "w1_c": w1_c.w1,
                "w1_c_ci_lo": w1_c.ci_lo,
                "w1_c_ci_hi": w1_c.ci_hi,
                "w1_c_mc_se": w1_c.mc_se,
                # null calibration -- the part that transfers across window sizes
                "z_q": cal_q.z,
                "p_q": cal_q.p_value,
                "thr_q": cal_q.threshold,
                "null_mean_q": cal_q.null_mean,
                "null_sd_q": cal_q.null_sd,
                "alarm_q": cal_q.alarm,
                "z_c": cal_c.z,
                "p_c": cal_c.p_value,
                "thr_c": cal_c.threshold,
                "null_mean_c": cal_c.null_mean,
                "null_sd_c": cal_c.null_sd,
                "alarm_c": cal_c.alarm,
                # secondary statistics, reported not thresholded
                "mmd_q": mmd_q.mmd,
                "mmd_c": mmd_c.mmd,
                "mmd_q_sq": mmd_q.mmd_sq,
                "mmd_c_sq": mmd_c.mmd_sq,
                "centroid_q": centroid_shift(Q_t, Q_prev),
                "centroid_c": centroid_shift(C_t, C_prev),
                # diagnostic only: the query/corpus genre gap that used to dominate D(t)
                "qc_genre_gap": centroid_shift(Q_t, C_t),
                # the live statistic, and the legacy one kept for the transfer demo
                "d": d_z,
                "d_raw": d_raw,
            }
        )

        logger.info(
            f"  D(t={t}) = {d_z:+.3f}   z_q={cal_q.z:+.2f} (p={cal_q.p_value:.3f})  "
            f"z_c={cal_c.z:+.2f} (p={cal_c.p_value:.3f})"
        )

    if not rows:
        raise RuntimeError(
            "no windows produced a drift measurement; every window had fewer than 2 "
            "samples. Check data/snapshots and DATA_MIN_WINDOW_SAMPLES."
        )

    return pd.DataFrame(rows)


def centroid_share_of_d(drift_df: pd.DataFrame) -> float:
    """Share of the legacy raw statistic contributed by the genre gap.

    Reported so the F5 defect stays visible in every run rather than needing to be
    rediscovered. Values near 1 mean the legacy D(t) was measuring the genre gap.
    """
    if "d_raw" not in drift_df or not len(drift_df):
        return float("nan")
    gap = 0.5 * drift_df["qc_genre_gap"].to_numpy()
    total = drift_df["d_raw"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(np.abs(total) > 1e-12, gap / total, np.nan)
    return float(np.nanmean(share))


def save_drift_series(df: pd.DataFrame, results_dir: Path | None = None, name: str = "drift_series") -> Path:
    results_dir = results_dir or settings.drift.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"{name}.parquet"
    df.to_parquet(out, index=False)
    logger.info(f"drift series saved -> {out}  ({len(df)} windows)")
    return out
