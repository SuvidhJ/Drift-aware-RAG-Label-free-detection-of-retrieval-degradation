"""Cost-aware policy: argmin_a [ cost(a) + lambda * regret(a) ].

What was wrong
--------------
The previous engine computed no-op regret as ``d * (alpha + beta)`` where
alpha + beta == 1 by construction, so it was just ``d``.  It then compared that --
a faithfulness fraction in [0, 1] -- against costs expressed as compute multiples in
[1, 110], at lambda = 1.  The two quantities had no common unit, and the arithmetic
worked out so that no-op won everywhere:

    sweep D in [0, 0.14], alpha in [0, 1], lambda = 1  ->  {a1_no_op}
    to ever select a2_rerank:  lambda > 54 at D = 0.14, or D > 7.5 at lambda = 1

against an achievable maximum D of 0.139.  The repo's own policy.csv showed the
"repo regime" row bit-identical to "always no-op" at every lambda (FINDINGS.md F6).

The fix
-------
Put regret in compute-equivalent units.  ``regret_scale`` is the compute a team
would spend to avoid losing *all* retrieval quality for one window, so

    regret(a) = regret_scale * residual_quality_loss(a)

is directly comparable to cost(a) and lambda is a dimensionless preference rather
than a unit conversion smuggled in as a hyperparameter.

Expected loss comes from the null-calibrated z-scores, through a saturating map so
that an unbounded z cannot produce an unbounded loss (recall is bounded by 1).  The
per-action recovery fractions are grounded in the measured selective-re-index curves
rather than invented; see ``ActionModel``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from configs.settings import settings
from policy.costs import Action, action_cost


@dataclass
class ActionModel:
    """Fraction of the loss on each channel that an action recovers.

    a2 re-rank touches only the query side: MMR widens and reorders candidates, which
    helps when queries have moved relative to a still-accurate index, and does nothing
    for vectors that are simply stale.

    a3 re-index recovers corpus-side loss in proportion to the share of recoverable
    quality a budgeted refresh actually captures.  At a 20% budget the measured share
    of gain is ~0.55 for probe-kNN and ~0.55 for cluster-threshold (they tie), against
    ~0.38 for random -- FINDINGS.md 12.4-12.5.  ``reindex_share`` defaults to that.

    a4 fine-tune is modelled as recovering most of both channels; no measurement in
    this project supports a precise value, and ``FINDINGS.md`` does not claim one, so
    it is declared here as an assumption rather than buried in a formula.
    """

    rerank_query: float = 0.40
    rerank_corpus: float = 0.0
    reindex_query: float = 0.0
    reindex_share: float = 0.55
    finetune_query: float = 0.85
    finetune_corpus: float = 0.85
    escalate: float = 1.0


def expected_quality_loss(z_q: float, z_c: float, saturation: float = 6.0) -> tuple[float, float]:
    """Map null-calibrated z-scores to expected fractional recall loss per channel.

    ``1 - exp(-max(z, 0) / saturation)`` is monotone, zero at and below the null, and
    bounded by 1.  A z of 6 corresponds to ~63% of the maximum modelled loss.

    This is a monotone transfer function, not a fitted predictor.  Use
    ``fit_loss_model`` to replace it with coefficients measured on a real run before
    quoting any cost saving.
    """
    q = 1.0 - np.exp(-max(z_q, 0.0) / saturation)
    c = 1.0 - np.exp(-max(z_c, 0.0) / saturation)
    return float(q), float(c)


def residual_loss(
    action: Action, loss_q: float, loss_c: float, model: ActionModel, budget: float
) -> float:
    """Quality loss remaining after ``action``, as a fraction in [0, 1]."""
    if action is Action.NO_OP:
        return loss_q + loss_c
    if action is Action.RERANK:
        return loss_q * (1.0 - model.rerank_query) + loss_c * (1.0 - model.rerank_corpus)
    if action is Action.REINDEX:
        # a full-budget refresh recovers everything on the corpus side; a partial one
        # recovers the measured share of the available gain
        share = model.reindex_share if budget < 1.0 else 1.0
        return loss_q * (1.0 - model.reindex_query) + loss_c * (1.0 - share)
    if action is Action.FINETUNE:
        return loss_q * (1.0 - model.finetune_query) + loss_c * (1.0 - model.finetune_corpus)
    if action is Action.ESCALATE:
        return (loss_q + loss_c) * (1.0 - model.escalate)
    raise ValueError(f"unhandled action {action!r}")


def objective(
    action: Action, loss_q: float, loss_c: float, lam: float, model: ActionModel, budget: float
) -> float:
    cfg = settings.policy
    return action_cost(action, budget) + lam * cfg.regret_scale * residual_loss(
        action, loss_q, loss_c, model, budget
    )


def compute_policy(
    drift_df: pd.DataFrame,
    regime_df: pd.DataFrame | None = None,
    lam: float | None = None,
    model: ActionModel | None = None,
    budget: float | None = None,
) -> pd.DataFrame:
    """Select an action per window by minimising cost + lambda * regret."""
    cfg = settings.policy
    lam = cfg.lam if lam is None else lam
    budget = cfg.reindex_budget if budget is None else budget
    model = model or ActionModel()

    if {"z_q", "z_c"} - set(drift_df.columns):
        raise KeyError(
            "drift_df must carry null-calibrated z_q and z_c; it should come from "
            "drift.statistic.compute_drift_series"
        )

    df = drift_df[["window", "z_q", "z_c"]].copy()
    if regime_df is not None and "gated_regime" in regime_df.columns:
        df = df.merge(regime_df[["window", "gated_regime", "action_suggested"]], on="window", how="left")

    rows: list[dict] = []
    for _, row in df.iterrows():
        loss_q, loss_c = expected_quality_loss(float(row["z_q"]), float(row["z_c"]))

        scored = {a: objective(a, loss_q, loss_c, lam, model, budget) for a in Action}
        best = min(scored, key=scored.get)

        rows.append(
            {
                "window": int(row["window"]),
                "z_q": float(row["z_q"]),
                "z_c": float(row["z_c"]),
                "loss_query": loss_q,
                "loss_corpus": loss_c,
                "selected_action": best.value,
                "total_cost": float(scored[best]),
                "regret_no_op": float(cfg.regret_scale * (loss_q + loss_c)),
                "regime_suggested": row.get("action_suggested"),
                **{f"obj_{a.value}": float(scored[a]) for a in Action},
            }
        )
        logger.info(
            f"t={int(row['window'])}: loss=(q {loss_q:.3f}, c {loss_c:.3f}) -> "
            f"{best.value} @ {scored[best]:.2f}"
        )

    return pd.DataFrame(rows)


def sweep_lambda(
    lambdas: tuple[float, ...] = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0),
    z_grid: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0),
    model: ActionModel | None = None,
    budget: float | None = None,
) -> pd.DataFrame:
    """Which actions are reachable at all, over a grid of lambda and drift.

    This is the degeneracy check the previous engine would have failed: it selected
    ``a1_no_op`` at every point of its own reachable input range.  tests/test_policy.py
    asserts that more than one action appears here.
    """
    model = model or ActionModel()
    budget = settings.policy.reindex_budget if budget is None else budget

    rows: list[dict] = []
    for lam in lambdas:
        for z_q in z_grid:
            for z_c in z_grid:
                loss_q, loss_c = expected_quality_loss(z_q, z_c)
                scored = {a: objective(a, loss_q, loss_c, lam, model, budget) for a in Action}
                best = min(scored, key=scored.get)
                rows.append(
                    {"lam": lam, "z_q": z_q, "z_c": z_c, "selected_action": best.value}
                )
    return pd.DataFrame(rows)


def fit_loss_model(drift_df: pd.DataFrame, quality_df: pd.DataFrame) -> dict:
    """Regress observed recall loss on the drift z-scores.

    Returns OLS coefficients and R^2 for ``loss ~ a + b_q z_q + b_c z_c``, where loss
    is measured as ``1 - recall@k / recall@k at t=0``.  This is what turns
    ``expected_quality_loss`` from a declared transfer function into a calibrated one;
    until it is run and its R^2 inspected, any cost saving quoted from this engine is
    a model output, not a measurement.
    """
    merged = drift_df[["window", "z_q", "z_c"]].merge(
        quality_df[["window", "recall_at_k"]], on="window", how="inner"
    )
    if len(merged) < 4:
        return {"n": len(merged), "note": "need at least 4 overlapping windows", "r2": float("nan")}

    base = float(merged["recall_at_k"].iloc[0])
    if base <= 0:
        return {"n": len(merged), "note": "baseline recall is zero", "r2": float("nan")}

    y = 1.0 - merged["recall_at_k"].to_numpy() / base
    X = np.column_stack([np.ones(len(merged)), merged["z_q"].to_numpy(), merged["z_c"].to_numpy()])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 1e-15 else float("nan")

    return {
        "n": len(merged),
        "intercept": float(coef[0]),
        "beta_z_q": float(coef[1]),
        "beta_z_c": float(coef[2]),
        "r2": r2,
    }
