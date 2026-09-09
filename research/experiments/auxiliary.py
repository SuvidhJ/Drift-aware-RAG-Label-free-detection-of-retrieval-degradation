"""Auxiliary experiments: null vs sample size, CUSUM ARL0, KR bound, policy."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.dar import stats, theory
from research.dar.policy import evaluate_policies
from research.dar.simulator import RunSpec, simulate
from research.dar.streaming import CusumConfig, FrozenCUSUM, arl0_calibrate, rolling_cusum_repo


# --------------------------------------------------------------------------- #
# 1. raw W1 is a proxy for sample size; the calibrated z-score is not  (S1)
# --------------------------------------------------------------------------- #
def exp_null_vs_sample_size(d: int = 256, spread: float = 0.5,
                            sizes=(300, 600, 1000, 1600, 2400, 3500, 5000),
                            reps: int = 30, seed: int = 0) -> pd.DataFrame:
    """Two i.i.d. samples from the SAME distribution, size n each.  Report raw
    sliced-W1 (mean & max) and the permutation z-score, vs n.  Raw should decay
    ~ n^-1/2; z should hover around 0 regardless of n."""
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(d); mu /= np.linalg.norm(mu)
    rows = []
    for n in sizes:
        for _ in range(reps):
            per = spread / np.sqrt(d)
            X = mu + per * rng.standard_normal((n, d)); X /= np.linalg.norm(X, axis=1, keepdims=True)
            Y = mu + per * rng.standard_normal((n, d)); Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            wm, wx = stats.sliced_w1_pair(X, Y, 128, rng)
            cal = stats.permutation_null(X, Y, B=60, subsample=min(n, 2500),
                                        n_proj=96, seed=int(rng.integers(1 << 30)), do_mmd=False)
            rows.append(dict(n=n, w1_mean=wm, w1_max=wx,
                             z_mean=cal["w1_mean"].z, z_max=cal["w1_max"].z,
                             p_max=cal["w1_max"].p))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. CUSUM: rolling-reference (repo) never catches gradual drift; frozen does (S3)
# --------------------------------------------------------------------------- #
def _gradual_series(rng, T=40, t0=20, slope=0.12, noise=1.0, base=0.0):
    x = base + noise * rng.standard_normal(T)
    ramp = np.clip(np.arange(T) - t0, 0, None) * slope
    return x + ramp


def exp_cusum(seed: int = 0, n_runs: int = 300) -> dict:
    rng = np.random.default_rng(seed)

    # calibrate frozen-CUSUM h to ARL0 >= 100 on pure-noise series
    cfg = CusumConfig(burn_in=8, k_sigma=0.5, h=5.0)
    h = arl0_calibrate(lambda r: r.standard_normal(400), cfg, target_arl0=100.0,
                       n_sim=250, seed=seed + 7)
    cfg_cal = CusumConfig(cfg.burn_in, cfg.k_sigma, h)

    # in-control ARL for both detectors
    def arl0(run_detector):
        rls = []
        for _ in range(n_runs):
            s = rng.standard_normal(500)
            rl = run_detector(s)
            rls.append(rl if rl is not None else len(s))
        return float(np.mean(rls))

    def frozen_first_alarm(s):
        det = FrozenCUSUM(cfg_cal)
        for i, v in enumerate(s):
            if det.update(v)["alarm"]:
                return i - cfg_cal.burn_in
        return None

    arl0_frozen = arl0(frozen_first_alarm)
    arl0_repo = arl0(lambda s: rolling_cusum_repo(s, window=10, k=0.5, h=2.0, tau=1e9))  # tau off

    # detection delay on gradual drift (t0=20, slope so it reaches ~3 sigma by end)
    dd_frozen, dd_repo, miss_frozen, miss_repo = [], [], 0, 0
    for _ in range(n_runs):
        s = _gradual_series(rng, T=44, t0=20, slope=0.18, noise=1.0)
        a = frozen_first_alarm(s)
        if a is not None and a + cfg_cal.burn_in >= 20:
            dd_frozen.append(a + cfg_cal.burn_in - 20)
        else:
            miss_frozen += 1
        b = rolling_cusum_repo(s, window=10, k=0.5, h=2.0, tau=1e9)
        if b is not None and b >= 20:
            dd_repo.append(b - 20)
        else:
            miss_repo += 1

    return dict(
        h_calibrated=h,
        arl0_frozen=arl0_frozen,
        arl0_repo_rolling=arl0_repo,
        gradual_detect_delay_frozen_median=float(np.median(dd_frozen)) if dd_frozen else None,
        gradual_detect_delay_repo_median=float(np.median(dd_repo)) if dd_repo else None,
        gradual_miss_rate_frozen=miss_frozen / n_runs,
        gradual_miss_rate_repo=miss_repo / n_runs,
    )


# --------------------------------------------------------------------------- #
# 3. KR bound: holds with exact W1, violated with sliced W1  (theory section)
# --------------------------------------------------------------------------- #
def exp_kr_bound(seeds=(0, 1, 2, 3, 4), levels=(0.0, 0.12, 0.3, 0.45)) -> pd.DataFrame:
    rows = []
    for sd in seeds:
        for r in levels:
            spec = RunSpec(seed=2000 + sd, T=9, t_event=3,
                           query_newline_frac=r, query_newline_count=2,
                           corpus_delist_lines=0)
            sim = simulate(spec)
            for br in theory.check_kr_bound(sim.windows, ref=0, cap=1400, seed=sd):
                rows.append(dict(seed=sd, r=r, t=br.t, lhs=br.lhs,
                                 w1_exact=br.w1_exact, w1_sliced_mean=br.w1_sliced_mean,
                                 w1_sliced_max=br.w1_sliced_max,
                                 holds_exact=br.holds_exact,
                                 holds_sliced_mean=br.holds_sliced_mean,
                                 tightness_exact=br.lhs / (br.w1_exact + 1e-12)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 4. policy: realised cost of each policy vs the oracle, over the factorial  (policy section)
# --------------------------------------------------------------------------- #
def exp_policy(wdf: pd.DataFrame, rdf: pd.DataFrame, lams=(0.3, 1.0, 3.0)) -> pd.DataFrame:
    """Uses the already-computed per-window regime calls from the factorial run.
    Sweeps the cost/regret trade-off lambda."""
    from research.experiments.harness import RunCfg, _spec
    from research.experiments.harness import CONDITIONS
    rc = RunCfg()
    tau_z = float(np.quantile(wdf[wdf.true_regime == "stable"]["d_z"].values, 0.95))

    rows = []
    for lam in lams:
        for _, r in rdf.iterrows():
            cond = next(c for c in CONDITIONS if c["name"] == r.condition)
            spec = _spec(cond, r.seed, rc)
            g = wdf[(wdf.condition == r.condition) & (wdf.seed == r.seed)].sort_values("t")
            cal_regimes = g.regime_cal.tolist()
            cal_sig = [rr != "stable" for rr in cal_regimes]
            outs = evaluate_policies(spec, cal_regimes, cal_sig, g.d_z.values,
                                     g.regime_orig.tolist(), tau_z=tau_z, lam=lam)
            oracle = {o.name: o for o in outs}["oracle"].total_cost
            for o in outs:
                rows.append(dict(lam=lam, condition=r.condition, seed=r.seed,
                                 true_regime=r.true_regime, policy=o.name,
                                 total_cost=o.total_cost, action_cost=o.action_cost,
                                 harm_cost=o.harm_cost, regret_vs_oracle=o.total_cost - oracle))
    return pd.DataFrame(rows)
