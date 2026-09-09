"""
dashboard — Streamlit monitoring dashboard package
===================================================

**What it's for:**
A read-only visual interface for exploring pipeline results after running
``python main.py run-all``. Loads pre-computed parquet files from
``results/`` and renders interactive Plotly charts via Streamlit.

**Module: ``app.py``**

The dashboard has four panels and a top metrics bar:

*Metrics bar:*
    Shows window count, max D(t), alarm count (windows where D > τ),
    and average RAGAS faithfulness across all evaluated windows.

*D(t) Timeline:*
    Line chart of D(t) over time with three traces:
    - D(t) (purple) — the composite drift signal
    - W1_Q (blue dashed) — query distribution shift
    - W1_C (red dashed) — corpus distribution shift
    An orange dashed line marks the alarm threshold τ.

*Regime Quadrant Scatter:*
    Scatter of α̂ (x-axis) vs β̂ (y-axis) for each window, coloured by
    raw regime. Vertical line at τ_α, horizontal at τ_β, creating the
    four quadrant boundaries visually.

*Faithfulness Overlay:*
    Dual-axis chart: D(t) on the left axis, (1 − faithfulness) on the
    right. Visual inspection of how tightly they track.

*Policy Action Log:*
    Table of per-window action decisions with emoji indicators:
    🟢 no-op, 🔵 re-rank, 🟠 re-index, 🔴 fine-tune, 💀 escalate.

*Validation Results:*
    If ``results/validation_report.json`` exists, displays the four
    Phase 6 metrics (ρ, lead-time, AUROC, FAR) as metric cards.

**Launch:**
    ``python main.py dashboard``
    or: ``uv run streamlit run dashboard/app.py``
"""
