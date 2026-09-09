"""Validation metrics relating D(t) to observed retrieval quality.

Three defects from the audit are addressed here.

F19 -- the AUROC label was ``delisted exactly at t`` while the docstring said ``at or
before t``, and the manifest produced exactly one positive window, so the metric was
computed over a single positive against eleven negatives.  Labels are now supplied
explicitly by the caller with the convention named, and ``auroc_from_churn`` builds a
balanced label from per-window churn instead of a one-off event.

F20 -- Spearman rho was reported over ~12 autocorrelated windows with no CI.  A
moving-block bootstrap CI is now returned alongside it, and the effective sample size
is stated so the reader can see how little is being claimed.

F25 -- ``save_validation_report`` formatted possibly-missing values with ``:.4f``,
which raises TypeError on the ``'N/A'`` string it also supplied.  Formatting is now
None-safe.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr


def _fmt(value, spec: str = ".4f") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "N/A"
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def moving_block_bootstrap_rho(
    x: np.ndarray, y: np.ndarray, n_boot: int = 2000, block: int | None = None, seed: int = 0
) -> tuple[float, float]:
    """95% CI for Spearman rho under serial dependence.

    Resampling individual windows would assume independence that a drift time series
    does not have. Blocks of length ~n^(1/3) preserve local autocorrelation.
    """
    n = len(x)
    block = block or max(2, int(round(n ** (1 / 3))))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))

    vals = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(n - block + 1, 1), n_blocks)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
        if len(np.unique(x[idx])) < 3 or len(np.unique(y[idx])) < 3:
            continue
        r = spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(r):
            vals.append(r)

    if len(vals) < 50:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def compute_spearman_rho(
    drift_df: pd.DataFrame, quality_df: pd.DataFrame, quality_col: str = "recall_at_k"
) -> dict:
    """Spearman rho between D(t) and quality loss (1 - quality)."""
    merged = drift_df[["window", "d"]].merge(quality_df[["window", quality_col]], on="window")
    n = len(merged)

    if n < 4:
        return {"spearman_rho": None, "n_windows": n, "note": "need at least 4 overlapping windows"}

    d_vals = merged["d"].to_numpy(dtype=float)
    loss = 1.0 - merged[quality_col].to_numpy(dtype=float)

    rho_res = spearmanr(d_vals, loss)
    rho = float(rho_res.statistic)
    ci_lo, ci_hi = moving_block_bootstrap_rho(d_vals, loss)

    if n < 20:
        logger.warning(
            f"Spearman rho computed over only {n} windows; the CI is wide and the "
            "point estimate should not be quoted on its own"
        )

    return {
        "spearman_rho": rho,
        "p_value": float(rho_res.pvalue),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n_windows": n,
        "ci_method": "moving-block bootstrap, block = n^(1/3)",
    }


def compute_auroc(scores: np.ndarray, labels: np.ndarray, label_meaning: str = "unspecified") -> dict:
    """AUROC of a score against a binary label, with the label convention recorded."""
    from sklearn.metrics import roc_auc_score

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)

    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return {
            "auroc": None,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "label_meaning": label_meaning,
            "note": "AUROC undefined: labels have only one class",
        }
    if min(n_pos, n_neg) < 3:
        logger.warning(f"AUROC over {n_pos} positives and {n_neg} negatives is not informative")

    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "label_meaning": label_meaning,
    }


def auroc_from_churn(drift_df: pd.DataFrame, churn: pd.DataFrame) -> dict:
    """AUROC of D(t) against 'this window had above-median corpus churn'.

    Balanced by construction, unlike the single-event label it replaces.
    """
    merged = drift_df[["window", "d"]].merge(churn[["window", "churn"]], on="window")
    if len(merged) < 4:
        return {"auroc": None, "note": f"only {len(merged)} windows"}
    labels = (merged["churn"].to_numpy() > np.median(merged["churn"].to_numpy())).astype(int)
    return compute_auroc(
        merged["d"].to_numpy(), labels, label_meaning="window churn above the median across windows"
    )


def compute_lead_time(
    drift_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    quality_drop: float = 0.95,
    quality_col: str = "recall_at_k",
) -> dict:
    """Windows between the first drift alarm and the first material quality drop.

    The alarm is the null-calibrated one (``alarm_c`` or ``alarm_q``), not a fixed
    threshold on the raw statistic. Quality drop is relative to the first window.
    """
    merged = drift_df.merge(quality_df[["window", quality_col]], on="window", how="inner")
    if len(merged) < 2:
        return {"lead_time_windows": None, "note": "need at least 2 overlapping windows"}

    alarm_cols = [c for c in ("alarm_q", "alarm_c") if c in merged.columns]
    if not alarm_cols:
        return {"lead_time_windows": None, "note": "drift_df carries no calibrated alarm columns"}
    alarmed = merged[alarm_cols].any(axis=1)

    base = float(merged[quality_col].iloc[0])
    dropped = merged[quality_col] < quality_drop * base

    if not alarmed.any():
        return {
            "lead_time_windows": None,
            "drift_alarm_count": 0,
            "quality_drop_count": int(dropped.sum()),
            "note": "detector never alarmed",
        }
    if not dropped.any():
        return {
            "lead_time_windows": None,
            "drift_alarm_count": int(alarmed.sum()),
            "quality_drop_count": 0,
            "note": f"quality never fell below {quality_drop:.0%} of its first-window value",
        }

    first_alarm = int(merged.loc[alarmed, "window"].iloc[0])
    first_drop = int(merged.loc[dropped, "window"].iloc[0])
    return {
        "lead_time_windows": first_drop - first_alarm,
        "first_alarm_window": first_alarm,
        "first_quality_drop_window": first_drop,
        "drift_alarm_count": int(alarmed.sum()),
        "quality_drop_count": int(dropped.sum()),
    }


def save_validation_report(report: dict, results_dir: Path, name: str = "validation_report") -> Path:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"{name}.json"
    out.write_text(json.dumps(report, indent=2, default=str))

    logger.info("=" * 62)
    logger.info("VALIDATION")
    logger.info("=" * 62)
    sp = report.get("spearman_rho", {}) or {}
    lt = report.get("lead_time", {}) or {}
    au = report.get("auroc", {}) or {}
    far = report.get("false_alarm_rate", {}) or {}
    logger.info(
        f"  Spearman rho:  {_fmt(sp.get('spearman_rho'))} "
        f"[{_fmt(sp.get('ci_lo'))}, {_fmt(sp.get('ci_hi'))}]  n={sp.get('n_windows', 'N/A')}"
    )
    logger.info(f"  Lead time:     {_fmt(lt.get('lead_time_windows'), 'd')} windows")
    logger.info(f"  AUROC:         {_fmt(au.get('auroc'))}")
    logger.info(f"  FAR:           {_fmt(far.get('far'))}  (target <= {_fmt(far.get('target'), '.2f')})")
    logger.info("=" * 62)
    logger.info(f"report -> {out}")
    return out
