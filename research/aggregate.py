"""Collapse results/*.{csv,json} into one report_data.json for the write-up."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).parent / "results"


def _csv(name):
    p = RES / name
    return pd.read_csv(p) if p.exists() else None


def _json(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    out: dict = {}
    out["config"] = _json("config.json")

    wdf = _csv("factorial_windows.csv")
    rdf = _csv("factorial_runs.csv")
    out["n_runs"] = 0 if rdf is None else int(len(rdf))
    out["n_window_rows"] = 0 if wdf is None else int(len(wdf))

    # ---- regime accuracy + confusion ----
    out["regime_accuracy"] = _json("regime_accuracy.json")
    for tag in ["confusion_calibrated_post", "confusion_calibrated_event", "confusion_original_post"]:
        c = _csv(tag + ".csv")
        if c is not None:
            c = c.rename(columns={c.columns[0]: "true"})
            out[tag] = c.to_dict(orient="records")

    # ---- AUROC ----
    a = _csv("auroc_by_channel.csv")
    if a is not None:
        out["auroc"] = a.round(4).to_dict(orient="records")

    # ---- FAR / power ----
    fp = _csv("far_power.csv")
    if fp is not None:
        out["far_power"] = fp.round(4).to_dict(orient="records")

    # ---- detection delay ----
    dd = _csv("detection_delay.csv")
    if dd is not None:
        out["detection_delay"] = dd.round(3).to_dict(orient="records")

    # ---- original D(t) audit ----
    out["original_dt_audit"] = _json("original_dt_audit.json")

    # ---- null vs sample size ----
    nvs = _csv("null_vs_sample_size.csv")
    if nvs is not None:
        g = nvs.groupby("n").agg(
            w1_mean=("w1_mean", "mean"), w1_mean_sd=("w1_mean", "std"),
            w1_max=("w1_max", "mean"), w1_max_sd=("w1_max", "std"),
            z_mean=("z_mean", "mean"), z_max=("z_max", "mean"),
            z_max_sd=("z_max", "std"),
        ).reset_index()
        out["null_vs_sample_size"] = g.round(6).to_dict(orient="records")
    out["null_vs_sample_size_summary"] = _json("null_vs_sample_size_summary.json")

    # ---- seasonality confound ----
    out["seasonality_confound"] = _json("seasonality_confound.json")

    # ---- CUSUM ----
    out["cusum"] = _json("cusum.json")

    # ---- KR bound ----
    out["kr_bound_summary"] = _json("kr_bound_summary.json")
    kr = _csv("kr_bound.csv")
    if kr is not None:
        kr_d = kr[kr.r > 0]
        out["kr_scatter"] = {
            "lhs": kr_d.lhs.round(5).tolist(),
            "w1_exact": kr_d.w1_exact.round(5).tolist(),
            "w1_sliced_mean": kr_d.w1_sliced_mean.round(5).tolist(),
        }

    # ---- gradual drift ----
    gd = _json("gradual.json")
    if gd:
        out["gradual"] = gd

    # ---- policy ----
    ps = _csv("policy_summary.csv")
    if ps is not None:
        out["policy_summary"] = ps.round(2).to_dict(orient="records")
    pol = _csv("policy.csv")
    if pol is not None and "lam" in pol.columns:
        p1 = pol[pol.lam == 1.0]
        out["policy_by_regime"] = (
            p1.groupby(["true_regime", "policy"])
              .total_cost.mean().round(1).reset_index()
              .to_dict(orient="records")
        )

    # ---- power-vs-magnitude curves (from far_power wide cols) ----
    if fp is not None:
        curves = {}
        for _, row in fp.iterrows():
            curves[row["detector"]] = {
                "query":  [row.get("power_query_lo", np.nan), row.get("power_query_hi", np.nan)],
                "corpus": [row.get("power_corpus_lo", np.nan), row.get("power_corpus_hi", np.nan)],
                "joint":  [row.get("power_joint_lo", np.nan), row.get("power_joint_hi", np.nan)],
            }
        out["power_curves"] = curves

    def _clean(o):
        if isinstance(o, float):
            return o if np.isfinite(o) else None
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        return o

    txt = json.dumps(_clean(out), indent=2, default=str, allow_nan=False)
    (RES / "report_data.json").write_text(txt)
    print(f"wrote report_data.json  ({len(txt)} bytes)")
    print("keys:", sorted(out.keys()))


if __name__ == "__main__":
    main()
