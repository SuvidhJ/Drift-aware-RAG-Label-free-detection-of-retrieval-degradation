"""
Sequential change detection — the fix for S3.

Repo bug: mu_ref is a rolling mean over the last 10 D-values *including the
current one*, so the reference chases the drift and a gradual shift is absorbed
and never alarms.  Also: k is in raw-D units not sigma units; g_neg is dead code;
the `d_val > tau` clause lets a raw threshold short-circuit the sequential test.

Fix: freeze (mu0, sigma0) over a burn-in prefix, run a standardised one-sided
CUSUM with k in sigma0 units, and calibrate the decision interval h to a target
in-control Average Run Length (ARL0) by Monte-Carlo on the null.  ARL0 is the
principled version of "FAR <= 5%": ARL0 = 1 / FAR_per_window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CusumConfig:
    burn_in: int = 6
    k_sigma: float = 0.5          # reference shift / 2, in sigma0 units
    h: float = 5.0                # decision interval (calibrate via arl0_calibrate)


class FrozenCUSUM:
    def __init__(self, cfg: CusumConfig):
        self.cfg = cfg
        self._buf: list[float] = []
        self.mu0: float | None = None
        self.sd0: float | None = None
        self.g = 0.0
        self.t = 0
        self.alarm_t: int | None = None

    def update(self, x: float) -> dict:
        c = self.cfg
        self.t += 1
        if self.mu0 is None:
            self._buf.append(x)
            if len(self._buf) >= c.burn_in:
                arr = np.asarray(self._buf)
                self.mu0 = float(arr.mean())
                self.sd0 = float(arr.std() + 1e-9)
            return {"t": self.t, "g": 0.0, "alarm": False, "armed": self.mu0 is not None}
        z = (x - self.mu0) / self.sd0
        self.g = max(0.0, self.g + z - c.k_sigma)
        fired = self.g > c.h
        if fired and self.alarm_t is None:
            self.alarm_t = self.t
        return {"t": self.t, "g": self.g, "alarm": fired, "armed": True}


def rolling_cusum_repo(x_series, window: int = 10, k: float = 0.5, h: float = 2.0, tau: float = 0.15):
    """Faithful reproduction of streaming/online_detector.py for comparison."""
    from collections import deque
    hist: deque = deque(maxlen=window)
    g_pos = 0.0
    alarm_t = None
    for t, d in enumerate(x_series):
        hist.append(d)
        mu_ref = float(np.mean(hist)) if len(hist) > 1 else d
        dev = d - (mu_ref + k)
        g_pos = max(0.0, g_pos + dev)
        if (g_pos > h or d > tau) and alarm_t is None:
            alarm_t = t
    return alarm_t


def arl0_calibrate(null_series_sampler, cfg: CusumConfig, target_arl0: float = 100.0,
                   n_sim: int = 400, h_grid=None, seed: int = 0) -> float:
    """Pick the smallest h whose in-control ARL >= target_arl0.

    null_series_sampler(rng) -> 1-D array of in-control D values (long).
    """
    rng = np.random.default_rng(seed)
    if h_grid is None:
        h_grid = np.linspace(1.0, 12.0, 23)
    best_h = h_grid[-1]
    for h in h_grid:
        run_lengths = []
        for _ in range(n_sim):
            s = null_series_sampler(rng)
            det = FrozenCUSUM(CusumConfig(cfg.burn_in, cfg.k_sigma, float(h)))
            rl = None
            for i, v in enumerate(s):
                st = det.update(v)
                if st["alarm"]:
                    rl = i - cfg.burn_in
                    break
            run_lengths.append(rl if rl is not None else (len(s) - cfg.burn_in))
        arl0 = float(np.mean(run_lengths))
        if arl0 >= target_arl0:
            best_h = float(h)
            break
    return best_h
