"""Online CUSUM.

Guards F14, where the detector could not fire at all: the reference mean was updated
with the current observation so it chased the signal, the slack was absolute (0.5)
against a statistic on the 0.1 scale so ``g_pos`` was pinned at zero, and the alarm
therefore degenerated to a plain threshold test.
"""
from __future__ import annotations

import numpy as np
import pytest

from streaming.online_detector import OnlineDriftDetector, calibrate_h, run_detector


def test_burn_in_never_alarms():
    det = OnlineDriftDetector(burn_in=8, h=3.0)
    states = [det.update(v) for v in np.linspace(0.10, 0.11, 8)]
    assert all(s["phase"] == "burn_in" for s in states)
    assert not any(s["alarm"] for s in states)


def test_flat_signal_rarely_alarms():
    """A false-alarm RATE, not a single run.

    One flat series proving nothing is exactly how the previous suite convinced
    itself the detector worked. At h=5, k=0.5 the in-control ARL is a few hundred
    windows, so an occasional alarm over 50 is expected; a majority is not.
    """
    alarmed = 0
    trials = 20
    for seed in range(trials):
        rng = np.random.default_rng(seed)
        series = 0.1 + 0.01 * rng.standard_normal(60)
        alarmed += any(s["alarm"] for s in run_detector(series, burn_in=20, h=5.0))
    assert alarmed / trials <= 0.35, f"flat signal alarmed in {alarmed}/{trials} runs"


def test_sustained_shift_alarms():
    rng = np.random.default_rng(1)
    series = np.concatenate([0.1 + 0.01 * rng.standard_normal(20), 0.2 + 0.01 * rng.standard_normal(20)])
    states = run_detector(series, burn_in=10, h=5.0)
    alarms = [s for s in states if s["alarm"]]
    assert alarms, "a 10-sigma sustained shift went undetected"
    assert alarms[0]["window"] >= 20
    assert alarms[0]["direction"] == "increase"


def test_slack_is_in_sigma_units_not_absolute():
    """The defect: k=0.5 on a 0.1-scale statistic pinned the statistic at zero.

    A shift of several sigma must accumulate g_pos regardless of the raw scale, so
    the same shift expressed in tiny units must still be detected.
    """
    rng = np.random.default_rng(2)
    tiny = np.concatenate([
        0.001 + 0.0001 * rng.standard_normal(20),
        0.002 + 0.0001 * rng.standard_normal(20),
    ])
    assert any(s["alarm"] for s in run_detector(tiny, burn_in=10, h=5.0))


def test_gradual_drift_is_not_absorbed_by_the_reference():
    """A moving reference hides gradual drift, which is what the detector is for."""
    rng = np.random.default_rng(5)
    series = np.concatenate([
        0.10 + 0.005 * rng.standard_normal(12),
        np.linspace(0.10, 0.30, 30) + 0.005 * rng.standard_normal(30),
    ])
    assert any(s["alarm"] for s in run_detector(series, burn_in=10, h=5.0, drift_reference=False))


def test_flat_burn_in_does_not_blind_the_detector():
    """sd=0 in the burn-in used to fall back to 1.0, which silenced the detector."""
    series = np.concatenate([np.full(12, 0.10), np.linspace(0.10, 0.30, 30)])
    assert any(s["alarm"] for s in run_detector(series, burn_in=10, h=5.0))


def test_both_cusum_arms_are_live():
    """g_neg was computed and never read."""
    rng = np.random.default_rng(3)
    series = np.concatenate([0.2 + 0.01 * rng.standard_normal(20), 0.1 + 0.01 * rng.standard_normal(20)])
    alarms = [s for s in run_detector(series, burn_in=10, h=5.0) if s["alarm"]]
    assert alarms and alarms[0]["direction"] == "decrease"


def test_higher_h_is_less_trigger_happy():
    rng = np.random.default_rng(4)
    series = np.concatenate([0.10 + 0.02 * rng.standard_normal(20), 0.13 + 0.02 * rng.standard_normal(30)])
    loose = sum(s["alarm"] for s in run_detector(series, burn_in=10, h=2.0))
    tight = sum(s["alarm"] for s in run_detector(series, burn_in=10, h=12.0))
    assert loose >= tight


def test_reset_clears_state():
    det = OnlineDriftDetector(burn_in=5, h=1.0)
    for v in [0.1] * 5 + [5.0, 5.0]:
        det.update(v)
    det.reset()
    assert det.update(0.1)["g_pos"] == 0.0


@pytest.mark.slow
def test_h_calibrates_to_a_target_arl0():
    """Without this, h is a constant with no operating characteristic attached."""
    res = calibrate_h(target_arl0=100, n_sim=300, max_run=1500, seed=0)
    assert 1.0 < res["h"] < 12.0
    assert res["arl0_achieved"] > 50
