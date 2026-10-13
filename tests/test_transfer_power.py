"""Threshold transfer and detection power.

Copied verbatim from the transfer/power section of the original
``tests/test_evaluation.py`` (now in ``quarantine/tests/``), whose other tests covered
quarantined modules.
"""
from __future__ import annotations

import numpy as np
import pytest

from evaluation.power import corpus_operating_point, perturb, run_power_sweep
from evaluation.transfer import run_transfer


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
