"""
Experiment harness: run the factorial, collect a tidy long-format table of
per-(run, window) detector outputs + ground truth, and the analysis routines
that turn it into the numbers in the report.

Everything is deterministic in (condition, seed).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from research.dar import detect
from research.dar.simulator import RunSpec, simulate

# --------------------------------------------------------------------------- #
# factorial design
# --------------------------------------------------------------------------- #
# magnitude levels chosen from the detectability probe:
#   corpus: L in {0,1,3}  (0=none, 1=~1 line ~2% unanswerable, 3=~5% unanswerable)
#   query : r in {0, 0.12, 0.30}
CONDITIONS: list[dict] = [
    dict(name="stable",        corpus_delist_lines=0, query_newline_frac=0.00),
    dict(name="query_lo",      corpus_delist_lines=0, query_newline_frac=0.12),
    dict(name="query_hi",      corpus_delist_lines=0, query_newline_frac=0.30),
    dict(name="corpus_lo",     corpus_delist_lines=1, query_newline_frac=0.00),
    dict(name="corpus_hi",     corpus_delist_lines=3, query_newline_frac=0.00),
    dict(name="joint_lo",      corpus_delist_lines=1, query_newline_frac=0.12),
    dict(name="joint_hi",      corpus_delist_lines=3, query_newline_frac=0.30),
]

BASELINES = {
    "top1_cosine":    detect.baseline_top1_cosine,
    "below_floor":    detect.baseline_below_floor,
    "centroid_shift": detect.baseline_centroid_shift,
    "sliced_w1":      detect.baseline_sliced_w1,
    "mmd":            detect.baseline_mmd,
    "c2st":           detect.baseline_c2st,
    "pca_ks":         detect.baseline_pca_ks,
}


@dataclass
class RunCfg:
    T: int = 11
    t_event: int = 6
    cal_B: int = 90
    cal_sub: int = 1300
    cal_nproj: int = 112
    seeds: tuple = (0, 1, 2, 3, 4, 6)


def _spec(cond: dict, seed: int, rc: RunCfg) -> RunSpec:
    return RunSpec(
        seed=1000 + seed,
        T=rc.T,
        t_event=rc.t_event,
        corpus_delist_lines=cond["corpus_delist_lines"],
        query_newline_frac=cond["query_newline_frac"],
        query_newline_count=2,
    )


def run_factorial(rc: RunCfg, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (windows_df, runs_df).

    windows_df: one row per (condition, seed, window t>=1) with every detector's
                scalar score, the original & calibrated regime call, and truth.
    runs_df:    one row per (condition, seed) with run-level metadata.
    """
    wrows, rrows = [], []
    total = len(CONDITIONS) * len(rc.seeds)
    k = 0
    for cond in CONDITIONS:
        for seed in rc.seeds:
            k += 1
            t0 = time.time()
            spec = _spec(cond, seed, rc)
            sim = simulate(spec)
            W = sim.windows
            true_regime = spec.regime()

            REF = detect.REF
            orig = detect.d_original_series(W, seed=seed, ref=REF)
            cal = detect.d_calibrated_series(
                W, B=rc.cal_B, subsample=rc.cal_sub, n_proj=rc.cal_nproj, seed=seed, ref=REF
            )
            base_scores = {name: fn(W, seed=seed, ref=REF) for name, fn in BASELINES.items()}

            for i, t in enumerate(range(REF + 1, rc.T)):
                w = W[t]
                op = orig[i]
                cp = cal[i]
                row = dict(
                    condition=cond["name"], seed=seed, t=t,
                    true_regime=w.regime, drifted=int(w.drifted),
                    is_event=int(t == rc.t_event),
                    post_event=int(t >= rc.t_event),
                    unanswerable=w.unanswerable_frac,
                    n_q=w.queries.shape[0], n_c=w.corpus.shape[0],
                    # original
                    d_orig=op.d, w1q_orig=op.w1_q, w1c_orig=op.w1_c, cgap_orig=op.cgap,
                    alpha_hat=op.alpha_hat, beta_hat=op.beta_hat,
                    regime_orig=op.raw_regime,
                    # calibrated
                    d_z=cp.d_z, q_z=cp.q.z, c_z=cp.c.z, q_p=cp.q.p, c_p=cp.c.p,
                    cgap_delta_z=cp.cgap_delta_z, p_combined=cp.p_combined,
                    regime_cal=cp.regime,
                )
                for name, arr in base_scores.items():
                    row[f"base_{name}"] = float(arr[i])
                wrows.append(row)

            rrows.append(dict(condition=cond["name"], seed=seed,
                              true_regime=true_regime,
                              corpus_lines=cond["corpus_delist_lines"],
                              query_frac=cond["query_newline_frac"],
                              t_event=rc.t_event, T=rc.T))
            log(f"[{k:3d}/{total}] {cond['name']:10s} seed={seed}  "
                f"{time.time()-t0:5.1f}s  (true={true_regime})")

    return pd.DataFrame(wrows), pd.DataFrame(rrows)


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
DETECTOR_COLS = {
    "D_original":      "d_orig",
    "D_calibrated_z":  "d_z",
    "base_top1_cosine": "base_top1_cosine",
    "base_below_floor": "base_below_floor",
    "base_centroid_shift": "base_centroid_shift",
    "base_sliced_w1":  "base_sliced_w1",
    "base_mmd":        "base_mmd",
    "base_c2st":       "base_c2st",
    "base_pca_ks":     "base_pca_ks",
}


def auroc_table(wdf: pd.DataFrame) -> pd.DataFrame:
    """AUROC of each detector's score vs the `drifted` label, over ALL windows
    and broken out by which drift channel is active."""
    out = []
    subsets = {
        "all":          wdf,
        "vs_query":     wdf[wdf.true_regime.isin(["stable", "query_drift"])],
        "vs_corpus":    wdf[wdf.true_regime.isin(["stable", "corpus_drift"])],
        "vs_joint":     wdf[wdf.true_regime.isin(["stable", "joint_drift"])],
    }
    for det, col in DETECTOR_COLS.items():
        rec = {"detector": det}
        for sname, sub in subsets.items():
            y = sub["drifted"].values
            s = sub[col].values
            rec[sname] = roc_auc_score(y, s) if len(set(y)) == 2 else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def far_calibrated_threshold(wdf: pd.DataFrame, col: str, target_far: float = 0.05) -> float:
    """(1 - target_far) quantile of the score on STABLE runs (no injected drift),
    excluding the first post-burn window."""
    stable = wdf[(wdf.true_regime == "stable")][col].values
    return float(np.quantile(stable, 1.0 - target_far))


def far_and_power(wdf: pd.DataFrame, target_far: float = 0.05) -> pd.DataFrame:
    out = []
    for det, col in DETECTOR_COLS.items():
        thr = far_calibrated_threshold(wdf, col, target_far)
        stable = wdf[wdf.true_regime == "stable"]
        far = float((stable[col].values > thr).mean())
        rec = {"detector": det, "threshold": thr, "FAR_empirical": far}
        for reg in ["query_drift", "corpus_drift", "joint_drift"]:
            sub = wdf[(wdf.true_regime == reg) & (wdf.post_event == 1)]
            rec[f"power_{reg}"] = float((sub[col].values > thr).mean()) if len(sub) else np.nan
        # power by magnitude bucket
        for cond in ["query_lo", "query_hi", "corpus_lo", "corpus_hi", "joint_lo", "joint_hi"]:
            sub = wdf[(wdf.condition == cond) & (wdf.post_event == 1)]
            rec[f"power_{cond}"] = float((sub[col].values > thr).mean()) if len(sub) else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def detection_delay(wdf: pd.DataFrame, rdf: pd.DataFrame, target_far: float = 0.05) -> pd.DataFrame:
    """Windows between t_event and the first alarm, at a FAR-matched threshold.
    NaN = never alarmed within the horizon."""
    out = []
    for det, col in DETECTOR_COLS.items():
        thr = far_calibrated_threshold(wdf, col, target_far)
        delays = []
        fired = 0
        drifted_runs = rdf[rdf.true_regime != "stable"]
        for _, r in drifted_runs.iterrows():
            g = wdf[(wdf.condition == r.condition) & (wdf.seed == r.seed)].sort_values("t")
            te = r.t_event
            alarms = g[(g.t >= te) & (g[col].values > thr)]
            if len(alarms):
                delays.append(int(alarms.t.iloc[0] - te))
                fired += 1
            else:
                delays.append(np.nan)
        d = np.array(delays, float)
        out.append({
            "detector": det,
            "median_delay": np.nanmedian(d) if np.isfinite(d).any() else np.nan,
            "mean_delay": np.nanmean(d) if np.isfinite(d).any() else np.nan,
            "detected_frac": fired / len(drifted_runs),
        })
    return pd.DataFrame(out)


def regime_confusion(wdf: pd.DataFrame, which: str = "regime_cal", rows: str = "post") -> pd.DataFrame:
    """Confusion matrix: predicted regime vs true regime.
    rows='post' -> all post-event windows (sustained-correctness view)
    rows='event' -> only the change-point window (latency view)"""
    sel = wdf[wdf.is_event == 1] if rows == "event" else wdf[wdf.post_event == 1]
    order = ["stable", "query_drift", "corpus_drift", "joint_drift"]
    return pd.crosstab(sel.true_regime, sel[which]).reindex(index=order, columns=order, fill_value=0)


def regime_accuracy(wdf: pd.DataFrame) -> dict:
    out = {}
    for tag, sel in [("post_event", wdf[wdf.post_event == 1]), ("event_window", wdf[wdf.is_event == 1])]:
        out[tag] = {
            "original": float((sel.regime_orig == sel.true_regime).mean()),
            "calibrated": float((sel.regime_cal == sel.true_regime).mean()),
            "n_rows": int(len(sel)),
        }
    # binary "any drift detected" (regime != stable) vs "any drift present"
    pe = wdf[wdf.post_event == 1]
    st = wdf[wdf.true_regime == "stable"]
    out["binary_calibrated"] = {
        "sensitivity": float((pe.regime_cal != "stable").mean()),
        "false_alarm_rate": float((st.regime_cal != "stable").mean()),
    }
    out["binary_original"] = {
        "sensitivity": float((pe.regime_orig != "stable").mean()),
        "false_alarm_rate": float((st.regime_orig != "stable").mean()),
    }
    return out


def seasonality_confound(wdf: pd.DataFrame) -> dict:
    """On STABLE runs, does the detector score track query volume |Q_t|?
    A well-calibrated detector should not."""
    from scipy.stats import spearmanr
    st = wdf[wdf.true_regime == "stable"]
    out = {}
    for det, col in DETECTOR_COLS.items():
        v = st[col].values
        if np.ptp(v) < 1e-12 or np.ptp(st["n_q"].values) < 1e-12:
            out[det] = {"spearman_rho_vs_nq": None, "p": None, "note": "constant input"}
            continue
        rho, p = spearmanr(st["n_q"].values, v)
        out[det] = {"spearman_rho_vs_nq": float(rho), "p": float(p)}
    return out
