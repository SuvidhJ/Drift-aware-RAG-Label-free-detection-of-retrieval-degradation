"""Regime classification on a genuine two-dimensional plane.

What was wrong
--------------
The previous classifier normalised the two channels into shares,

    alpha = W1_q / (W1_q + W1_c),   beta = W1_c / (W1_q + W1_c)

so alpha + beta == 1 by construction.  A two-dimensional taxonomy collapsed to a
one-dimensional partition of alpha, with the consequences (FINDINGS.md F11):

  * (W1_q, W1_c) = (0.9, 0.9) and (0.009, 0.009) received the *same* label, because
    the classifier was scale-free and could not tell a large drift from a tiny one;
  * "joint drift" meant "the two channels are similar in magnitude", which is the
    least informative case, yet it triggered the most expensive action;
  * ``return Regime.STABLE`` on the final line was unreachable.

The fix
-------
Classify on the two *absolute* null-standardised magnitudes (z_q, z_c) against
separate thresholds.  All four quadrants are now reachable and each means what its
name says:

              z_c >= tau_c  |  CORPUS_DRIFT   JOINT_DRIFT
              z_c <  tau_c  |  STABLE         QUERY_DRIFT
                            +-----------------------------
                               z_q < tau_q    z_q >= tau_q

alpha and beta are still reported, as descriptive shares, but they do not drive the
decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger

from configs.settings import settings


class Regime(str, Enum):
    STABLE = "stable"
    QUERY_DRIFT = "query_drift"
    CORPUS_DRIFT = "corpus_drift"
    JOINT_DRIFT = "joint_drift"


ACTION_MAP: dict[Regime, str] = {
    Regime.STABLE: "a1_no_op",
    Regime.QUERY_DRIFT: "a2_rerank",
    Regime.CORPUS_DRIFT: "a3_reindex",
    Regime.JOINT_DRIFT: "a4_finetune",
}


@dataclass
class RegimePoint:
    window: int
    z_q: float
    z_c: float
    alpha: float
    beta: float
    raw_regime: str
    gated_regime: str | None
    action_suggested: str | None
    consecutive_count: int


def classify_raw(z_q: float, z_c: float, tau_q: float, tau_c: float) -> Regime:
    """Quadrant of the (z_q, z_c) plane. Every branch is reachable."""
    q_hot = z_q >= tau_q
    c_hot = z_c >= tau_c
    if q_hot and c_hot:
        return Regime.JOINT_DRIFT
    if q_hot:
        return Regime.QUERY_DRIFT
    if c_hot:
        return Regime.CORPUS_DRIFT
    return Regime.STABLE


def decompose_and_classify(
    drift_df: pd.DataFrame,
    tau_q: float | None = None,
    tau_c: float | None = None,
    gate: int | None = None,
) -> pd.DataFrame:
    """Classify each window and apply a persistence gate.

    The gate suppresses costly actions until the same non-stable regime has held for
    ``persistence_gate`` consecutive windows. STABLE is never gated: standing down is
    free, so it takes effect immediately.
    """
    cfg = settings.drift
    tau_q = cfg.tau_z_query if tau_q is None else tau_q
    tau_c = cfg.tau_z_corpus if tau_c is None else tau_c
    gate = cfg.persistence_gate if gate is None else gate

    missing = {"window", "z_q", "z_c"} - set(drift_df.columns)
    if missing:
        raise KeyError(
            f"drift_df is missing {sorted(missing)}. It must come from "
            "drift.statistic.compute_drift_series, which supplies null-calibrated z-scores."
        )

    rows: list[RegimePoint] = []
    run_regime: Regime | None = None
    run_len = 0

    for _, row in drift_df.iterrows():
        z_q = float(row["z_q"])
        z_c = float(row["z_c"])

        # descriptive only -- never used for the decision
        w1_q = float(row.get("w1_q", 0.0))
        w1_c = float(row.get("w1_c", 0.0))
        total = w1_q + w1_c
        alpha = w1_q / total if total > 1e-12 else 0.5
        beta = w1_c / total if total > 1e-12 else 0.5

        regime = classify_raw(z_q, z_c, tau_q, tau_c)

        if regime == run_regime:
            run_len += 1
        else:
            run_regime, run_len = regime, 1

        gated: Regime | None = None
        action: str | None = None

        if regime == Regime.STABLE:
            gated, action = Regime.STABLE, ACTION_MAP[Regime.STABLE]
        elif run_len >= gate:
            gated, action = regime, ACTION_MAP[regime]
            logger.info(
                f"t={int(row['window'])}: {regime.value} CONFIRMED "
                f"({run_len}/{gate} windows, z_q={z_q:+.2f}, z_c={z_c:+.2f}) -> {action}"
            )
        else:
            logger.debug(
                f"t={int(row['window'])}: {regime.value} not yet gated ({run_len}/{gate})"
            )

        rows.append(
            RegimePoint(
                window=int(row["window"]),
                z_q=z_q,
                z_c=z_c,
                alpha=alpha,
                beta=beta,
                raw_regime=regime.value,
                gated_regime=gated.value if gated else None,
                action_suggested=action,
                consecutive_count=run_len,
            )
        )

    return pd.DataFrame([vars(r) for r in rows])


def reachability_report(tau_q: float | None = None, tau_c: float | None = None) -> dict[str, bool]:
    """Confirm all four regimes are reachable at the configured thresholds.

    tests/test_decomposition.py asserts on this. The previous classifier would have
    failed it, which is the point.
    """
    cfg = settings.drift
    tau_q = cfg.tau_z_query if tau_q is None else tau_q
    tau_c = cfg.tau_z_corpus if tau_c is None else tau_c

    probes = np.array(
        [
            [tau_q - 1.0, tau_c - 1.0],
            [tau_q + 1.0, tau_c - 1.0],
            [tau_q - 1.0, tau_c + 1.0],
            [tau_q + 1.0, tau_c + 1.0],
        ]
    )
    seen = {classify_raw(float(a), float(b), tau_q, tau_c) for a, b in probes}
    return {r.value: (r in seen) for r in Regime}
