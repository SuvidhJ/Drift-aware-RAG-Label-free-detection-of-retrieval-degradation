"""Cost-aware policy engine.

Guards F6. The shipped engine computed no-op regret as ``d * (alpha + beta)`` with
alpha + beta == 1 by construction, then compared that fraction against compute
multiples in [1, 110] at lambda = 1. Over its entire reachable input range it
selected ``a1_no_op`` and nothing else; to reach ``a2_rerank`` needed lambda > 54 or
D > 7.5 against an achievable maximum D of 0.139.

The old test suite encoded the defect rather than catching it: test_policy.py needed
lambda = 1000 to see any action, and its own comment worked out the arithmetic
proving the engine was inert at the default.
"""
from __future__ import annotations

import pandas as pd
import pytest

from policy.costs import Action, action_cost
from policy.engine import (
    ActionModel,
    compute_policy,
    expected_quality_loss,
    fit_loss_model,
    objective,
    residual_loss,
    sweep_lambda,
)


def test_engine_is_not_degenerate_at_the_default_lambda():
    """More than one action must be selectable over the reachable input range."""
    chosen = set(sweep_lambda().selected_action.unique())
    assert len(chosen) > 1, f"engine collapsed to {chosen}"


def test_every_action_is_reachable_somewhere():
    chosen = set(sweep_lambda().selected_action.unique())
    missing = {a.value for a in Action} - chosen
    assert not missing, f"unreachable actions: {sorted(missing)}"


def test_no_op_wins_when_nothing_is_drifting():
    df = compute_policy(pd.DataFrame({"window": [0], "z_q": [0.0], "z_c": [0.0]}))
    assert df.selected_action.iloc[0] == "a1_no_op"


def test_corpus_drift_selects_reindex_not_rerank():
    """Re-ranking cannot fix stale vectors, so it must not win on a corpus-only shift."""
    df = compute_policy(pd.DataFrame({"window": [0], "z_q": [0.0], "z_c": [10.0]}))
    assert df.selected_action.iloc[0] == "a3_reindex"


def test_query_drift_prefers_rerank_over_reindex():
    df = compute_policy(pd.DataFrame({"window": [0], "z_q": [6.0], "z_c": [0.0]}))
    assert df.selected_action.iloc[0] == "a2_rerank"


def test_escalation_needs_extreme_drift_and_high_lambda():
    calm = compute_policy(pd.DataFrame({"window": [0], "z_q": [1.0], "z_c": [1.0]}), lam=1.0)
    assert calm.selected_action.iloc[0] != "a5_escalate"


def test_quality_loss_is_monotone_and_bounded():
    assert expected_quality_loss(0.0, 0.0) == (0.0, 0.0)
    assert expected_quality_loss(-5.0, -5.0) == (0.0, 0.0)  # below the null is not drift
    q_small, _ = expected_quality_loss(1.0, 0.0)
    q_big, _ = expected_quality_loss(10.0, 0.0)
    assert 0.0 < q_small < q_big < 1.0


def test_residual_loss_decreases_with_stronger_actions():
    m = ActionModel()
    losses = {
        a: residual_loss(a, 0.5, 0.5, m, budget=0.2)
        for a in (Action.NO_OP, Action.RERANK, Action.REINDEX, Action.FINETUNE, Action.ESCALATE)
    }
    assert losses[Action.NO_OP] > losses[Action.RERANK] > losses[Action.FINETUNE]
    assert losses[Action.ESCALATE] == pytest.approx(0.0)


def test_reindex_cost_scales_with_budget():
    """A 5% refresh must not cost what a full one does."""
    assert action_cost(Action.REINDEX, 0.05) < action_cost(Action.REINDEX, 0.5)
    assert action_cost(Action.REINDEX, 1.0) == pytest.approx(action_cost(Action.REINDEX, 1.0))


def test_escalate_cost_is_a_number_not_a_sentinel():
    """1e9 dominated every float comparison and made the argmin meaningless."""
    assert action_cost(Action.ESCALATE) < 1e6


def test_objective_is_cost_when_lambda_is_zero():
    for a in Action:
        assert objective(a, 0.9, 0.9, 0.0, ActionModel(), 0.2) == pytest.approx(action_cost(a, 0.2))


def test_rejects_uncalibrated_input():
    with pytest.raises(KeyError, match="z_q"):
        compute_policy(pd.DataFrame({"window": [0], "d": [0.5]}))


def test_loss_model_fit_reports_r2():
    drift = pd.DataFrame({"window": range(6), "z_q": [0, 1, 2, 3, 4, 5], "z_c": [0, 1, 2, 3, 4, 5]})
    quality = pd.DataFrame({"window": range(6), "recall_at_k": [1.0, 0.95, 0.9, 0.85, 0.8, 0.75]})
    fit = fit_loss_model(drift, quality)
    assert fit["n"] == 6
    assert fit["r2"] > 0.9


def test_loss_model_declines_to_fit_on_too_few_windows():
    drift = pd.DataFrame({"window": [0, 1], "z_q": [0.0, 1.0], "z_c": [0.0, 1.0]})
    quality = pd.DataFrame({"window": [0, 1], "recall_at_k": [1.0, 0.9]})
    assert "note" in fit_loss_model(drift, quality)
