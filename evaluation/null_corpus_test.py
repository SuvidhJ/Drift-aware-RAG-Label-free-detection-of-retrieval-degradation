"""False-alarm rate on the null slice.

The previous pipeline reported FAR = 3.1% and could not have computed it.
``main.py`` guarded on ``results/null_drift_series.parquet``; grepping the repository
for that name returned exactly one hit -- that guard.  No writer existed anywhere, and
``build_null_corpus_slice`` returned a directory path without computing anything, so
the branch never ran and FAR was silently ``None`` (FINDINGS.md F2).  A further
problem was granularity: with T ~= 12 windows the achievable rates near the 5% target
are 0% and 8.3%, so 3.1% ~= 1/32 was not even expressible (F21).

Both are fixed by ``data.windows.build_null_windows``, which writes a real null slice
of arbitrary length, and by counting alarms from the per-window permutation
calibration rather than a fixed threshold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from configs.settings import settings


def compute_far(
    null_drift_df: pd.DataFrame, alpha: float | None = None, use_calibrated: bool = True
) -> dict:
    """False-alarm rate over a slice where nothing drifts.

    Args:
        null_drift_df: drift series computed on the null slice
        alpha: the rate the calibration targets
        use_calibrated: count the null-calibrated alarms. When False, applies the
            legacy fixed threshold to ``d_raw`` instead, which is what
            evaluation/transfer.py uses to show the fixed threshold failing.
    """
    alpha = settings.drift.alpha_level if alpha is None else alpha
    n_windows = len(null_drift_df)
    if n_windows == 0:
        return {"far": None, "note": "null drift series is empty"}

    if use_calibrated:
        cols = [c for c in ("alarm_q", "alarm_c") if c in null_drift_df.columns]
        if not cols:
            raise KeyError("null drift series carries no alarm columns; recompute with drift.statistic")
        alarms = null_drift_df[cols].any(axis=1)
        method = f"per-window permutation null at alpha={alpha}"
    else:
        tau = settings.drift.threshold_tau
        alarms = null_drift_df["d_raw"] > tau
        method = f"legacy fixed threshold tau={tau} on the raw statistic"

    n_alarms = int(alarms.sum())
    far = n_alarms / n_windows

    # Each window is an independent Bernoulli trial; Wilson interval, since the
    # normal approximation is poor at the small counts a short run produces.
    z = 1.96
    denom = 1 + z**2 / n_windows
    centre = (far + z**2 / (2 * n_windows)) / denom
    half = z * np.sqrt(far * (1 - far) / n_windows + z**2 / (4 * n_windows**2)) / denom

    result = {
        "far": float(far),
        "ci_lo": float(max(centre - half, 0.0)),
        "ci_hi": float(min(centre + half, 1.0)),
        "n_alarms": n_alarms,
        "n_windows": n_windows,
        "target": float(alpha),
        "target_met": bool(far <= alpha),
        "method": method,
        "granularity": 1.0 / n_windows,
    }

    status = "PASS" if result["target_met"] else "FAIL"
    logger.info(
        f"null FAR: {far:.4f} ({n_alarms}/{n_windows}) "
        f"[{result['ci_lo']:.3f}, {result['ci_hi']:.3f}] target <= {alpha}  {status}"
    )
    if n_windows < 1 / alpha:
        logger.warning(
            f"only {n_windows} null windows: the finest expressible rate is "
            f"{1 / n_windows:.3f}, coarser than the {alpha} target. Increase DATA_N_WINDOWS."
        )
    return result
