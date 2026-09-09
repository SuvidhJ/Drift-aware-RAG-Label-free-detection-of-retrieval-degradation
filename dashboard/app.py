"""Streamlit dashboard.

Reads whatever the pipeline has produced under ``results/`` and renders it. Two
deliberate departures from the previous version:

  * the drift timeline plots the null-calibrated z-scores, not the raw statistic
    against a fixed tau. A fixed threshold is not comparable across window sizes, so
    drawing one as a horizontal line invited exactly the misreading that
    evaluation/transfer.py exists to demonstrate;
  * the regime scatter uses (z_q, z_c), which is a genuine plane. The old one
    plotted alpha against beta, which sum to 1 by construction, so every point lay
    on a single line and the "quadrant" chart was a picture of a 1-D partition.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from configs.settings import settings

RESULTS = settings.drift.results_dir
REGIME_COLOURS = {
    "stable": "#22C55E",
    "query_drift": "#3B82F6",
    "corpus_drift": "#EF4444",
    "joint_drift": "#F59E0B",
}


def _load(name: str) -> pd.DataFrame | None:
    path = RESULTS / name
    return pd.read_parquet(path) if path.exists() else None


def _load_json(name: str) -> dict | None:
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def _drift_timeline(drift: pd.DataFrame) -> None:
    st.subheader("Null-calibrated drift over time")
    st.caption(
        "z-scores against a per-window permutation null. Comparable across window "
        "sizes; the raw statistic is not."
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=drift.window, y=drift.z_q, mode="lines+markers", name="z query", line={"color": "#3B82F6"}))
    fig.add_trace(go.Scatter(x=drift.window, y=drift.z_c, mode="lines+markers", name="z corpus", line={"color": "#EF4444"}))
    fig.add_hline(y=settings.drift.tau_z_query, line_dash="dash", line_color="gray", annotation_text="tau_z")
    fig.add_hline(y=0.0, line_color="#555")
    fig.update_layout(xaxis_title="window", yaxis_title="z", template="plotly_dark", height=380,
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)


def _regime_plane(regime: pd.DataFrame) -> None:
    st.subheader("Regime plane")
    st.caption("Both axes are absolute magnitudes, so all four quadrants are reachable.")
    fig = px.scatter(
        regime, x="z_q", y="z_c", color="raw_regime", color_discrete_map=REGIME_COLOURS,
        hover_data=["window", "consecutive_count", "gated_regime"],
        template="plotly_dark", height=420,
        labels={"z_q": "z query drift", "z_c": "z corpus drift"},
    )
    fig.add_vline(x=settings.drift.tau_z_query, line_dash="dash", line_color="gray")
    fig.add_hline(y=settings.drift.tau_z_corpus, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)


def _quality(quality: pd.DataFrame) -> None:
    st.subheader("Closed loop: what the action actually recovered")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=quality.window, y=quality.recall_stale, mode="lines", name="stale index",
                             line={"dash": "dot", "color": "#EF4444"}))
    fig.add_trace(go.Scatter(x=quality.window, y=quality.recall_at_k, mode="lines+markers", name="after action",
                             line={"color": "#7C3AED", "width": 3}))
    fig.add_trace(go.Scatter(x=quality.window, y=quality.recall_full_reindex, mode="lines", name="full re-index",
                             line={"dash": "dot", "color": "#22C55E"}))
    fig.update_layout(xaxis_title="window", yaxis_title=f"recall@{settings.retrieval.top_k}",
                      template="plotly_dark", height=380, legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)

    if quality.headroom.abs().max() < 1e-4:
        st.warning(
            "Headroom is zero in every window: a full re-index recovers nothing, so "
            "selective re-index is vacuous on this slice. This usually means the "
            "corpus is small enough that retrieval is saturated."
        )


def _policy(policy: pd.DataFrame) -> None:
    st.subheader("Policy decisions")
    emoji = {"a1_no_op": "-", "a2_rerank": "rerank", "a3_reindex": "reindex",
             "a4_finetune": "finetune", "a5_escalate": "escalate"}
    view = policy[["window", "z_q", "z_c", "loss_query", "loss_corpus", "selected_action", "total_cost"]].copy()
    view["action"] = view.selected_action.map(lambda a: emoji.get(a, a))
    st.dataframe(view.drop(columns=["selected_action"]), use_container_width=True, hide_index=True)


def _bakeoff() -> None:
    path = RESULTS / "bakeoff.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    st.subheader("Selective re-index bake-off")
    st.caption(
        "Share of recoverable recall captured. probe_knn against cluster_threshold is "
        "the comparison that decides novelty."
    )
    fig = px.line(df, x="budget", y="share_of_gain", color="selector", markers=True,
                  template="plotly_dark", height=380)
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Drift-Aware RAG", layout="wide")
    st.title("Drift-Aware RAG")

    drift = _load("drift_series.parquet")
    if drift is None:
        st.error(f"No results in {RESULTS}. Run `python main.py run-all` first.")
        return

    regime = _load("regime_series.parquet")
    policy = _load("policy_decisions.parquet")
    quality = _load("rag_quality.parquet")
    report = _load_json("validation_report.json") or {}

    cols = st.columns(4)
    cols[0].metric("windows", len(drift))
    cols[1].metric("query alarms", int(drift.alarm_q.sum()))
    cols[2].metric("corpus alarms", int(drift.alarm_c.sum()))
    far = (report.get("false_alarm_rate") or {}).get("far")
    cols[3].metric("null FAR", "n/a" if far is None else f"{far:.3f}")

    _drift_timeline(drift)
    if regime is not None:
        _regime_plane(regime)
    if quality is not None:
        _quality(quality)
    if policy is not None:
        _policy(policy)
    _bakeoff()

    with st.expander("validation report"):
        st.json(report)


if __name__ == "__main__":
    # Streamlit executes the script with __name__ == "__main__", so this covers both
    # `streamlit run dashboard/app.py` and direct execution. Guarding it also keeps
    # the module importable (for a syntax or reference check) without rendering.
    main()
