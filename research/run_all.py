"""
Run the full research programme end-to-end.  Writes results/*.csv + results/*.json
incrementally so a crash keeps progress.  ~30-45 min single-core.

    python research/run_all.py            # full
    python research/run_all.py --fast     # smaller grid / fewer perms (~8 min)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).parent / "results"
RES.mkdir(exist_ok=True)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def jdump(obj, name):
    (RES / name).write_text(json.dumps(obj, indent=2, default=str))
    log(f"  wrote {name}")


def main(fast: bool = False):
    from research.experiments.harness import (
        RunCfg, run_factorial, auroc_table, far_and_power, detection_delay,
        regime_confusion, regime_accuracy, seasonality_confound,
    )
    # 'aux' is a reserved DOS device name, so a file called aux.py makes the repo
    # unclonable on Windows without core.protectNTFS=false. Renamed to auxiliary.py.
    from research.experiments import auxiliary as aux

    t_start = time.time()
    rc = RunCfg(
        T=9 if fast else 11,
        t_event=5 if fast else 6,
        cal_B=45 if fast else 90,
        cal_sub=900 if fast else 1300,
        cal_nproj=64 if fast else 112,
        seeds=(0, 1, 2) if fast else (0, 1, 2, 3, 4, 6),
    )
    jdump(dict(fast=fast, **{k: getattr(rc, k) for k in
              ["T", "t_event", "cal_B", "cal_sub", "cal_nproj", "seeds"]}), "config.json")

    # ---- 1. factorial (the workhorse) ------------------------------------ #
    log("=== FACTORIAL ===")
    wdf, rdf = run_factorial(rc, log=log)
    wdf.to_csv(RES / "factorial_windows.csv", index=False)
    rdf.to_csv(RES / "factorial_runs.csv", index=False)
    log(f"  factorial done: {len(wdf)} window-rows, {len(rdf)} runs")

    # ---- 2. analysis of the factorial --------------------------------- #
    log("=== ANALYSIS ===")
    auroc_table(wdf).to_csv(RES / "auroc_by_channel.csv", index=False)
    far_and_power(wdf).to_csv(RES / "far_power.csv", index=False)
    detection_delay(wdf, rdf).to_csv(RES / "detection_delay.csv", index=False)
    regime_confusion(wdf, "regime_cal", "post").to_csv(RES / "confusion_calibrated_post.csv")
    regime_confusion(wdf, "regime_cal", "event").to_csv(RES / "confusion_calibrated_event.csv")
    regime_confusion(wdf, "regime_orig", "post").to_csv(RES / "confusion_original_post.csv")
    jdump(regime_accuracy(wdf), "regime_accuracy.json")
    jdump(seasonality_confound(wdf), "seasonality_confound.json")

    # original-D(t) alarm audit: does it EVER cross tau=0.15 ?
    d_orig_max = float(wdf.d_orig.max())
    ev = wdf[wdf.is_event == 1]
    jdump(dict(
        tau_orig=0.15,
        d_orig_max_anywhere=d_orig_max,
        d_orig_ever_alarms=bool(d_orig_max > 0.15),
        d_orig_at_event_mean=float(ev.d_orig.mean()),
        d_orig_at_event_max=float(ev.d_orig.max()),
        cgap_share_of_d_quiet=float((0.5 * wdf[wdf.post_event == 0].cgap_orig).mean()
                                    / wdf[wdf.post_event == 0].d_orig.mean()),
    ), "original_dt_audit.json")

    # ---- 3. null vs sample size (S1) -------------------------------- #
    log("=== NULL vs SAMPLE SIZE ===")
    nvs = aux.exp_null_vs_sample_size(reps=8 if fast else 25)
    nvs.to_csv(RES / "null_vs_sample_size.csv", index=False)
    g = nvs.groupby("n").agg(w1_mean=("w1_mean", "mean"), w1_max=("w1_max", "mean"),
                             z_mean=("z_mean", "mean"), z_max=("z_max", "mean")).reset_index()
    # slope of log(w1) vs log(n): expect ~ -0.5 ; slope of z vs n: expect ~ 0
    lw = np.polyfit(np.log(g.n), np.log(g.w1_max), 1)[0]
    lz = np.polyfit(g.n, g.z_max, 1)[0]
    jdump(dict(loglog_slope_w1max_vs_n=float(lw),
               slope_zmax_vs_n=float(lz),
               w1max_at_n300=float(g[g.n == 300].w1_max.iloc[0]),
               w1max_at_n5000=float(g[g.n == 5000].w1_max.iloc[0]),
               zmax_range=[float(g.z_max.min()), float(g.z_max.max())]),
          "null_vs_sample_size_summary.json")

    # ---- 4. CUSUM (S3) -------------------------------------------- #
    log("=== CUSUM ===")
    jdump(aux.exp_cusum(n_runs=120 if fast else 300), "cusum.json")

    # ---- 5. KR bound (theory) ----------------------------------- #
    log("=== KR BOUND ===")
    kr = aux.exp_kr_bound(seeds=(0, 1, 2) if fast else (0, 1, 2, 3, 4))
    kr.to_csv(RES / "kr_bound.csv", index=False)
    drift_rows = kr[kr.r > 0]
    jdump(dict(
        n_rows=int(len(kr)),
        holds_exact_rate=float(kr.holds_exact.mean()),
        holds_sliced_mean_rate=float(kr.holds_sliced_mean.mean()),
        holds_sliced_mean_rate_driftrows=float(drift_rows.holds_sliced_mean.mean()),
        median_tightness_exact=float(kr.tightness_exact.median()),
        note="LHS<=RHS should hold ~always for exact W1, and be violated for sliced-mean W1 (a lower bound on W1).",
    ), "kr_bound_summary.json")

    # ---- 6. policy ---------------------------------------------- #
    log("=== POLICY ===")
    pol = aux.exp_policy(wdf, rdf, lams=(0.3, 1.0, 3.0))
    pol.to_csv(RES / "policy.csv", index=False)
    piv = (pol.groupby(["lam", "policy"])
              .agg(mean_total_cost=("total_cost", "mean"),
                   mean_regret=("regret_vs_oracle", "mean"),
                   mean_action_cost=("action_cost", "mean"),
                   mean_harm_cost=("harm_cost", "mean"))
              .reset_index().sort_values(["lam", "mean_total_cost"]))
    piv.to_csv(RES / "policy_summary.csv", index=False)

    # ---- final console summary --------------------------------- #
    log(f"=== DONE in {(time.time()-t_start)/60:.1f} min ===")
    print("\n" + "=" * 78)
    print("REGIME-CALL ACCURACY ON EVENT WINDOWS")
    print(json.loads((RES / "regime_accuracy.json").read_text()))
    print("\nAUROC vs drift label (all windows):")
    print(auroc_table(wdf)[["detector", "all", "vs_query", "vs_corpus", "vs_joint"]]
          .to_string(index=False))
    print("\nFAR / power @ FAR=0.05:")
    print(pd.read_csv(RES / "far_power.csv")[
        ["detector", "FAR_empirical", "power_query_drift", "power_corpus_drift", "power_joint_drift"]
    ].to_string(index=False))
    print("\nPOLICY (mean over factorial):")
    print(piv.to_string(index=False))
    print("\nCONFUSION (calibrated, post-event windows): rows=true, cols=pred")
    print(regime_confusion(wdf, "regime_cal", "post").to_string())
    print("\nCUSUM:")
    print(json.loads((RES / "cusum.json").read_text()))
    print("\nKR BOUND:")
    print(json.loads((RES / "kr_bound_summary.json").read_text()))
    print("\nNULL vs SAMPLE SIZE:")
    print(json.loads((RES / "null_vs_sample_size_summary.json").read_text()))
    print("=" * 78)


if __name__ == "__main__":
    main(fast="--fast" in sys.argv)
