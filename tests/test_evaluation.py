"""Validation metrics, FAR, transfer and power.

Guards F19 (AUROC computed over a single positive window, with the code and its
docstring disagreeing about the label), F20 (Spearman rho over ~12 autocorrelated
windows with no CI), F25 (``'N/A'`` formatted with ``:.4f``, raising TypeError), and
F2/F21 (a false-alarm rate that had no writer and a granularity too coarse to express
the number that was reported).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.correlate import (
    _fmt,
    auroc_from_churn,
    compute_auroc,
    compute_lead_time,
    compute_spearman_rho,
    moving_block_bootstrap_rho,
    save_validation_report,
)
from evaluation.null_corpus_test import compute_far
from evaluation.power import corpus_operating_point, perturb, run_power_sweep
from evaluation.transfer import run_transfer


# ----------------------------------------------------------------- formatting


def test_fmt_is_none_safe():
    """F25: the old reporter formatted its own 'N/A' string with :.4f."""
    assert _fmt(None) == "N/A"
    assert _fmt(float("nan")) == "N/A"
    assert _fmt("N/A") == "N/A"
    assert _fmt(0.5) == "0.5000"
    assert _fmt(3, "d") == "3"


def test_report_writes_even_with_missing_sections(tmp_path):
    out = save_validation_report({"spearman_rho": {}, "auroc": None}, tmp_path)
    assert out.exists()


# ----------------------------------------------------------------- correlation


def test_spearman_returns_a_ci_and_the_window_count():
    drift = pd.DataFrame({"window": range(12), "d": np.arange(12.0)})
    quality = pd.DataFrame({"window": range(12), "recall_at_k": np.linspace(1.0, 0.5, 12)})
    res = compute_spearman_rho(drift, quality)
    assert res["spearman_rho"] > 0.9
    assert res["n_windows"] == 12
    assert "ci_lo" in res and "ci_hi" in res


def test_spearman_declines_on_too_few_windows():
    drift = pd.DataFrame({"window": [0, 1], "d": [0.1, 0.2]})
    quality = pd.DataFrame({"window": [0, 1], "recall_at_k": [1.0, 0.9]})
    assert compute_spearman_rho(drift, quality)["spearman_rho"] is None


def test_block_bootstrap_ci_brackets_the_estimate():
    rng = np.random.default_rng(0)
    x = np.arange(30.0)
    y = x + rng.standard_normal(30)
    lo, hi = moving_block_bootstrap_rho(x, y, n_boot=400)
    assert lo < hi
    assert -1.0 <= lo and hi <= 1.0


# ----------------------------------------------------------------------- AUROC


def test_auroc_is_undefined_for_a_single_class():
    """F19: the old label produced exactly one positive window."""
    res = compute_auroc(np.array([0.1, 0.2, 0.3]), np.array([1, 1, 1]))
    assert res["auroc"] is None
    assert "one class" in res["note"]


def test_auroc_records_its_label_convention():
    res = compute_auroc(np.array([0.1, 0.9, 0.2, 0.8]), np.array([0, 1, 0, 1]), "churn above median")
    assert res["auroc"] == pytest.approx(1.0)
    assert res["label_meaning"] == "churn above median"


def test_churn_label_is_balanced():
    drift = pd.DataFrame({"window": range(8), "d": np.arange(8.0)})
    churn = pd.DataFrame({"window": range(8), "churn": [0, 0, 0, 0, 5, 5, 5, 5]})
    res = auroc_from_churn(drift, churn)
    assert res["n_positive"] == 4 and res["n_negative"] == 4


# ------------------------------------------------------------------ lead time


def test_lead_time_uses_calibrated_alarms():
    drift = pd.DataFrame(
        {
            "window": range(6),
            "alarm_q": [False, False, True, True, True, True],
            "alarm_c": [False] * 6,
        }
    )
    quality = pd.DataFrame({"window": range(6), "recall_at_k": [1.0, 1.0, 1.0, 1.0, 0.5, 0.4]})
    res = compute_lead_time(drift, quality)
    assert res["first_alarm_window"] == 2
    assert res["first_quality_drop_window"] == 4
    assert res["lead_time_windows"] == 2


def test_lead_time_reports_when_nothing_alarmed():
    drift = pd.DataFrame({"window": range(4), "alarm_q": [False] * 4, "alarm_c": [False] * 4})
    quality = pd.DataFrame({"window": range(4), "recall_at_k": [1.0, 0.5, 0.4, 0.3]})
    assert compute_lead_time(drift, quality)["lead_time_windows"] is None


# ------------------------------------------------------------------------ FAR


def test_far_counts_calibrated_alarms_with_a_ci():
    null = pd.DataFrame({"alarm_q": [False] * 19 + [True], "alarm_c": [False] * 20})
    res = compute_far(null, alpha=0.05)
    assert res["far"] == pytest.approx(0.05)
    assert res["n_windows"] == 20
    assert res["target_met"]
    assert res["ci_lo"] <= res["far"] <= res["ci_hi"]


def test_far_reports_its_granularity():
    """F21: at T=12 the achievable rates near 5% are 0% and 8.3%."""
    null = pd.DataFrame({"alarm_q": [False] * 12, "alarm_c": [False] * 12})
    assert compute_far(null)["granularity"] == pytest.approx(1 / 12)


def test_far_requires_a_computed_null():
    """F2: there was no writer for the null series at all, so this branch never ran."""
    with pytest.raises(KeyError):
        compute_far(pd.DataFrame({"d": [0.1, 0.2]}))


def test_far_can_use_the_legacy_fixed_threshold_for_contrast():
    null = pd.DataFrame({"d_raw": [0.01, 0.02, 0.9, 0.95], "alarm_q": [False] * 4, "alarm_c": [False] * 4})
    legacy = compute_far(null, use_calibrated=False)
    assert legacy["far"] == pytest.approx(0.5)
    assert "fixed threshold" in legacy["method"]


# ------------------------------------------------------------- transfer, power


def test_transfer_shows_a_fixed_threshold_failing_both_ways(unit_pool):
    """FINDINGS 12.8, at a size that runs in a test."""
    res = run_transfer(unit_pool, n_calibrate=400, n_deploy=60, trials=40, alpha=0.05, seed=0)
    w1 = res["fixed_threshold"]["sliced_w1"]

    # native rates are correct by construction
    assert w1["far_native_at_n_calibrate"] == pytest.approx(0.05, abs=0.05)
    # the null level is higher at the smaller window, so the imported threshold over-fires
    assert w1["null_mean_at_n_deploy"] > w1["null_mean_at_n_calibrate"]
    assert w1["far_calibrate_big_deploy_small"] > 0.5
    assert w1["far_calibrate_small_deploy_big"] < 0.05
    assert res["predicted_inflation_from_sqrt_n"] > 1.0


def test_transfer_needs_enough_vectors(unit_pool):
    with pytest.raises(ValueError, match="need at least"):
        run_transfer(unit_pool, n_calibrate=5000, n_deploy=100, trials=2)


def test_perturbation_preserves_unit_norm(unit_pool):
    out = perturb(unit_pool, 0.5, 0.3, np.random.default_rng(0))
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    assert (np.linalg.norm(out - unit_pool, axis=1) > 1e-9).sum() == pytest.approx(
        0.3 * len(unit_pool), rel=0.05
    )


def test_power_rises_with_effect_size(unit_pool):
    res = run_power_sweep(unit_pool, scales=(0.02, 1.5), window=250, trials=20, seed=0)
    small, large = res["sweep"][0], res["sweep"][1]
    assert small["mean_displacement_all_docs"] < large["mean_displacement_all_docs"]
    assert small["power_sliced_w1"] <= large["power_sliced_w1"]
    assert small["power_sliced_w1"] < 0.4, "a negligible perturbation should sit near the floor"


def test_sweep_reports_both_displacement_conventions(unit_pool):
    """They differ by the revised fraction, and mixing them is how FINDINGS 12.9 went
    wrong: it swept the per-moved quantity but compared corpora against it using their
    per-all means."""
    frac = 0.30
    res = run_power_sweep(
        unit_pool, scales=(1.0,), window=250, revised_fraction=frac, trials=10, seed=1
    )
    row = res["sweep"][0]
    ratio = row["mean_displacement_all_docs"] / row["mean_displacement_moved_docs"]
    assert ratio == pytest.approx(frac, rel=0.05)


def test_corpus_operating_point_reports_both_conventions(unit_pool):
    rng = np.random.default_rng(3)
    fresh = unit_pool.copy()
    moved = rng.choice(len(fresh), len(fresh) // 4, replace=False)
    step = rng.standard_normal((len(moved), fresh.shape[1]))
    fresh[moved] = step / np.linalg.norm(step, axis=1, keepdims=True)

    op = corpus_operating_point(unit_pool, fresh)
    assert op["fraction_moved"] == pytest.approx(0.25, rel=0.02)
    # the per-all mean is the per-moved mean scaled by the moved fraction
    assert op["mean_displacement"] == pytest.approx(
        op["mean_displacement_moved"] * op["fraction_moved"], rel=0.02
    )


def test_corpus_operating_point(unit_pool):
    op = corpus_operating_point(unit_pool, unit_pool)
    assert op["mean_displacement"] == pytest.approx(0.0)
    assert op["fraction_moved"] == pytest.approx(0.0)
