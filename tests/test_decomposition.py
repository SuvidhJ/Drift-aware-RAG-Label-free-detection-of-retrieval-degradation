"""Regime classification.

Guards F11 and F18. The previous classifier normalised alpha and beta into shares
summing to 1, which collapsed the 2x2 taxonomy to a 1-D partition and left
``Regime.STABLE`` unreachable on the final branch. The previous *tests* then fed it
D values of 0.95 and 1.80 against an achievable maximum of 0.139, so they exercised
a region the pipeline could never enter.
"""
from __future__ import annotations

import pandas as pd
import pytest

from drift.decomposition import Regime, classify_raw, decompose_and_classify, reachability_report


def _frame(pairs: list[tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window": range(len(pairs)),
            "z_q": [p[0] for p in pairs],
            "z_c": [p[1] for p in pairs],
            "w1_q": [0.1] * len(pairs),
            "w1_c": [0.1] * len(pairs),
        }
    )


def test_all_four_regimes_are_reachable():
    report = reachability_report()
    assert all(report.values()), f"unreachable regimes: {[k for k, v in report.items() if not v]}"


def test_stable_branch_is_reachable():
    """The literal defect: the old classifier's final `return STABLE` was dead code."""
    assert classify_raw(0.0, 0.0, 2.0, 2.0) is Regime.STABLE


def test_quadrants_mean_what_they_say():
    assert classify_raw(5.0, 0.0, 2.0, 2.0) is Regime.QUERY_DRIFT
    assert classify_raw(0.0, 5.0, 2.0, 2.0) is Regime.CORPUS_DRIFT
    assert classify_raw(5.0, 5.0, 2.0, 2.0) is Regime.JOINT_DRIFT


def test_magnitude_is_distinguishable():
    """(0.9, 0.9) and (0.009, 0.009) received the same label under the old scheme."""
    big = classify_raw(9.0, 9.0, 2.0, 2.0)
    small = classify_raw(0.09, 0.09, 2.0, 2.0)
    assert big is Regime.JOINT_DRIFT
    assert small is Regime.STABLE
    assert big is not small


def test_persistence_gate_delays_costly_actions():
    df = _frame([(5.0, 0.0)] * 4)
    out = decompose_and_classify(df, tau_q=2.0, tau_c=2.0, gate=3)
    # ungated windows carry a null; pandas stores that as NaN in an object column
    assert out.gated_regime.isna().tolist() == [True, True, False, False]
    assert out.gated_regime.dropna().tolist() == ["query_drift", "query_drift"]
    assert out.consecutive_count.tolist() == [1, 2, 3, 4]


def test_stable_is_never_gated():
    """Standing down is free, so it should take effect immediately."""
    out = decompose_and_classify(_frame([(0.0, 0.0)]), tau_q=2.0, tau_c=2.0, gate=3)
    assert out.gated_regime.iloc[0] == "stable"


def test_run_resets_when_the_regime_changes():
    df = _frame([(5.0, 0.0), (5.0, 0.0), (0.0, 5.0), (0.0, 5.0)])
    out = decompose_and_classify(df, tau_q=2.0, tau_c=2.0, gate=3)
    assert out.consecutive_count.tolist() == [1, 2, 1, 2]
    assert out.gated_regime.isna().all()


def test_alpha_beta_are_reported_but_not_used_for_the_decision():
    df = _frame([(5.0, 0.0)])
    df.loc[0, "w1_q"] = 0.9
    df.loc[0, "w1_c"] = 0.1
    out = decompose_and_classify(df, tau_q=2.0, tau_c=2.0, gate=1)
    assert out.alpha.iloc[0] == pytest.approx(0.9)
    assert out.beta.iloc[0] == pytest.approx(0.1)
    assert out.alpha.iloc[0] + out.beta.iloc[0] == pytest.approx(1.0)
    # decision came from z, not from the shares
    assert out.raw_regime.iloc[0] == "query_drift"


def test_rejects_uncalibrated_input():
    bad = pd.DataFrame({"window": [0], "w1_q": [0.1], "w1_c": [0.1]})
    with pytest.raises(KeyError, match="z_q"):
        decompose_and_classify(bad)
