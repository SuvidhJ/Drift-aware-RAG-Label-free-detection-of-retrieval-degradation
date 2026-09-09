"""Online CUSUM over the drift series.

What was wrong
--------------
The previous detector could not fire (FINDINGS.md F14), for three compounding
reasons:

  1. the reference mean was computed *after* appending the current observation, so
     the baseline chased the signal it was supposed to detect;
  2. the slack ``k = 0.5`` was absolute, applied to a statistic living on the 0.1
     scale, so ``g_pos = max(0, g_pos - ~0.5)`` was pinned at zero and the alarm
     degenerated to the plain threshold test ``d > tau``;
  3. the 10-window rolling reference absorbed gradual drift by construction -- the
     exact failure mode the detector exists to catch.

``g_neg`` was computed and never read, and no ARL0 calibration existed, so ``h`` was
an arbitrary constant.

The fix
-------
Freeze the reference on a burn-in prefix, standardise, express slack in sigma units,
and calibrate ``h`` to a target in-control average run length by simulation.  Both
CUSUM arms are reported: a sustained *drop* in drift is also a change worth seeing.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from loguru import logger


@dataclass
class CUSUMState:
    g_pos: float = 0.0
    g_neg: float = 0.0
    alarm: bool = False
    alarm_window: int | None = None
    direction: str | None = None


@dataclass
class OnlineDriftDetector:
    """Two-sided CUSUM on a standardised drift statistic.

    Args:
        burn_in: observations used to fix the in-control mean and sd. They are
            consumed as reference and never alarm.
        k_sigma: slack, in standard deviations. 0.5 detects a sustained 1-sigma shift
            efficiently, the textbook default.
        h: decision interval, in the same standardised units. Use ``calibrate_h`` to
            pick it for a target ARL0 instead of guessing.
        drift_reference: when True the reference updates on in-control observations
            only. Leave False to detect gradual drift, which a moving reference hides.
    """

    burn_in: int = 8
    k_sigma: float = 0.5
    h: float = 5.0
    drift_reference: bool = False

    _history: deque = field(default_factory=lambda: deque(maxlen=4096), repr=False)
    _ref_mean: float | None = field(default=None, repr=False)
    _ref_sd: float | None = field(default=None, repr=False)
    _state: CUSUMState = field(default_factory=CUSUMState, repr=False)
    _t: int = field(default=0, repr=False)

    def update(self, d_val: float) -> dict:
        """Feed one observation and return the detector state."""
        t = self._t
        self._t += 1

        # burn-in: accumulate reference, never alarm
        if self._ref_mean is None:
            self._history.append(d_val)
            if len(self._history) >= self.burn_in:
                arr = np.asarray(self._history, dtype=float)
                self._ref_mean = float(arr.mean())
                sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
                if sd > 1e-12:
                    self._ref_sd = sd
                else:
                    # A perfectly flat burn-in carries no scale information. Falling
                    # back to 1.0 here silently blinded the detector: on a statistic
                    # living near 0.1, every subsequent z was smaller than the slack,
                    # so g_pos stayed pinned at zero and even a 3x gradual drift went
                    # unreported. A small scale relative to the level is the honest
                    # choice -- against a constant reference, any movement really is
                    # surprising, and the warning says so.
                    self._ref_sd = max(abs(self._ref_mean) * 1e-3, 1e-9)
                    logger.warning(
                        f"CUSUM burn-in had no variability (sd=0 over {len(arr)} windows); "
                        f"using a nominal scale of {self._ref_sd:.2e}. Any deviation will "
                        "look large. Lengthen the burn-in if this is not intended."
                    )
                logger.info(
                    f"CUSUM reference fixed after {len(arr)} windows: "
                    f"mu={self._ref_mean:.4f} sd={self._ref_sd:.4f}"
                )
            return {
                "window": t,
                "d": d_val,
                "z": float("nan"),
                "g_pos": 0.0,
                "g_neg": 0.0,
                "alarm": False,
                "phase": "burn_in",
            }

        z = (d_val - self._ref_mean) / self._ref_sd
        self._state.g_pos = max(0.0, self._state.g_pos + z - self.k_sigma)
        self._state.g_neg = max(0.0, self._state.g_neg - z - self.k_sigma)

        fired_up = self._state.g_pos > self.h
        fired_down = self._state.g_neg > self.h
        alarm = fired_up or fired_down

        if alarm and not self._state.alarm:
            self._state.alarm = True
            self._state.alarm_window = t
            self._state.direction = "increase" if fired_up else "decrease"
            logger.warning(
                f"CUSUM alarm at t={t} ({self._state.direction}): "
                f"z={z:+.2f} g+={self._state.g_pos:.2f} g-={self._state.g_neg:.2f} h={self.h}"
            )
        elif not alarm and self.drift_reference:
            # only in-control observations may move the reference
            self._history.append(d_val)
            arr = np.asarray(self._history, dtype=float)
            self._ref_mean = float(arr.mean())

        return {
            "window": t,
            "d": d_val,
            "z": float(z),
            "g_pos": float(self._state.g_pos),
            "g_neg": float(self._state.g_neg),
            "alarm": bool(alarm),
            "direction": self._state.direction if alarm else None,
            "phase": "monitor",
        }

    def reset(self) -> None:
        self._state = CUSUMState()
        logger.info("CUSUM state reset")


def calibrate_h(
    target_arl0: int = 200,
    k_sigma: float = 0.5,
    n_sim: int = 2000,
    max_run: int = 5000,
    seed: int = 0,
) -> dict:
    """Choose the decision interval for a target in-control average run length.

    ARL0 is the mean number of windows before a false alarm when nothing is changing.
    Simulated on standard normal observations, which is what the standardised
    statistic looks like in control. Without this, ``h`` is a number with no operating
    characteristic attached, which is what it was before.
    """
    rng = np.random.default_rng(seed)

    def arl_for(h: float) -> float:
        runs = np.empty(n_sim)
        for i in range(n_sim):
            g = 0.0
            for step in range(1, max_run + 1):
                g = max(0.0, g + rng.standard_normal() - k_sigma)
                if g > h:
                    runs[i] = step
                    break
            else:
                runs[i] = max_run
        return float(runs.mean())

    lo, hi = 1.0, 12.0
    best = hi
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        if arl_for(mid) < target_arl0:
            lo = mid
        else:
            hi = mid
            best = mid

    achieved = arl_for(best)
    logger.info(f"CUSUM h={best:.3f} gives ARL0={achieved:.0f} (target {target_arl0}, k={k_sigma})")
    return {"h": float(best), "arl0_achieved": achieved, "target_arl0": target_arl0, "k_sigma": k_sigma}


def run_detector(d_series, **kwargs) -> "list[dict]":
    det = OnlineDriftDetector(**kwargs)
    return [det.update(float(v)) for v in d_series]
