"""
Gradual drift -- the actual "RAG fails silently" scenario.

The factorial injects a one-shot step at t_event.  With a fixed reference and a
large persistent shift, detection is easy for *everything* (AUROC ~1 for raw
distances too).  The regime the README actually motivates is a slow ramp, where
each window-to-window step is small.

Query drift ramps r: 0 -> r_max linearly over t_event..T-1.  Per window we compare:
  * adjacent-window raw sliced-W1  (the repo's lag=1 statistic, Q + C)
  * adjacent-window calibrated z   (permutation-standardised, lag=1)
  * fixed-reference calibrated z   (vs a pre-drift baseline window)
  * frozen CUSUM on the fixed-ref z-series  vs  repo rolling CUSUM
and report time-to-detection at a matched in-control false-alarm level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.dar.simulator import RunSpec, simulate
from research.dar.stats import permutation_null, sliced_w1_mean
from research.dar.streaming import CusumConfig, FrozenCUSUM, rolling_cusum_repo

REF = 1


def exp_gradual(seeds=(0, 1, 2, 3, 4), T=16, t_event=6, r_max=0.35,
                B=90, sub=1300, nproj=112) -> tuple[pd.DataFrame, dict]:
    rows = []
    for sd in seeds:
        spec = RunSpec(seed=3000 + sd, T=T, t_event=t_event,
                       query_newline_frac=r_max, query_newline_count=2,
                       query_newline_ramp=True)
        W = simulate(spec).windows
        rng = np.random.default_rng(sd)

        adj_raw, adj_z, fix_z = [], [], []
        for t in range(1, T):
            adj_raw.append(sliced_w1_mean(W[t].queries, W[t - 1].queries, 128, rng)
                           + sliced_w1_mean(W[t].corpus, W[t - 1].corpus, 128, rng))
            adj_z.append(permutation_null(W[t].queries, W[t - 1].queries, B, sub, nproj, sd + t)["w1_max"].z)
            if t >= REF + 1:
                fix_z.append(permutation_null(W[t].queries, W[REF].queries, B, sub, nproj, sd + 100 + t)["w1_max"].z)
            else:
                fix_z.append(np.nan)

        for i, t in enumerate(range(1, T)):
            rows.append(dict(seed=sd, t=t, post=int(t >= t_event),
                             unanswerable=W[t].unanswerable_frac,
                             adj_raw=adj_raw[i], adj_z=adj_z[i], fix_z=fix_z[i]))
    df = pd.DataFrame(rows)

    def ttd(col):
        pre = df[df.post == 0][col].dropna()
        if len(pre) < 3:
            return [None, None]
        thr = float(np.quantile(pre, 0.95))
        delays = []
        for sd in seeds:
            g = df[df.seed == sd].sort_values("t")
            hit = g[(g.t >= t_event) & (g[col] > thr)]
            delays.append(int(hit.t.iloc[0] - t_event) if len(hit) else None)
        got = [x for x in delays if x is not None]
        return [float(np.median(got)) if got else None, len(got) / len(seeds)]

    cfg = CusumConfig(burn_in=4, k_sigma=0.5, h=3.0)
    fro, repo = [], []
    for sd in seeds:
        g = df[df.seed == sd].sort_values("t")
        z = g["fix_z"].dropna().values
        det = FrozenCUSUM(cfg); fa = None
        for i, v in enumerate(z):
            if det.update(float(v))["alarm"]:
                fa = (REF + 1) + i; break
        fro.append((fa - t_event) if fa is not None and fa >= t_event else None)
        rr = rolling_cusum_repo(z, window=10, k=0.5, h=2.0, tau=1e9)
        repo.append(((REF + 1) + rr - t_event) if rr is not None and (REF + 1) + rr >= t_event else None)

    summ = {
        "r_max": r_max, "T": T, "t_event": t_event, "n_seeds": len(seeds),
        "ttd_adj_raw": ttd("adj_raw"),
        "ttd_adj_z": ttd("adj_z"),
        "ttd_fix_z": ttd("fix_z"),
        "cusum_frozen_delay_median": float(np.median([x for x in fro if x is not None])) if any(x is not None for x in fro) else None,
        "cusum_frozen_detected": len([x for x in fro if x is not None]) / len(seeds),
        "cusum_repo_delay_median": float(np.median([x for x in repo if x is not None])) if any(x is not None for x in repo) else None,
        "cusum_repo_detected": len([x for x in repo if x is not None]) / len(seeds),
    }
    # per-window trajectory means for the chart
    traj = df.groupby("t").agg(
        adj_raw=("adj_raw", "mean"), adj_z=("adj_z", "mean"),
        fix_z=("fix_z", "mean"), unanswerable=("unanswerable", "mean")).reset_index()
    return df, {"summary": summ, "trajectory": traj.round(5).to_dict(orient="records")}
