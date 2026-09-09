"""Drift-Aware RAG CLI.

Pipeline
    python main.py build-snapshots     real revision windows C(t), Q(t), and a null slice
    python main.py embed               encode every window (each distinct text once)
    python main.py compute-drift       D(t) with a per-window permutation null
    python main.py classify-regimes    the (z_q, z_c) regime plane
    python main.py run-policy          cost-aware action selection
    python main.py run-rag             closed loop: stale index + action -> recall@k
    python main.py evaluate            validation report
    python main.py run-all             all of the above

Experiments
    python main.py bakeoff             selective re-index selectors vs prior art
    python main.py transfer            threshold transfer under volume shift
    python main.py power               detection power swept through the floor
    python main.py audit               structural self-checks

Legacy Amazon path (not recommended, see data/download.py)
    python main.py download
    python main.py inject-drift
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from configs.settings import settings

app = typer.Typer(help="Drift-Aware RAG pipeline", rich_markup_mode="rich", no_args_is_help=True)
console = Console()


def _setup_logging() -> None:
    if settings.log.enabled:
        Path(settings.log.sink).parent.mkdir(parents=True, exist_ok=True)
        logger.add(settings.log.sink, level=settings.log.level, rotation=settings.log.rotation, enqueue=True)


def _slice_dirs() -> tuple[Path, Path]:
    base = settings.data.snapshots_dir
    return base / settings.data.corpus, base / f"{settings.data.corpus}_null"


def _results_dir() -> Path:
    d = settings.drift.results_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def _opt(value, default=None):
    """Resolve a parameter that may still hold its typer sentinel.

    ``run-all`` calls the command functions directly rather than through the CLI, so
    any parameter the caller does not pass arrives as a ``typer.models.OptionInfo``
    instead of its declared default. Coercing here keeps both entry points working
    without duplicating every command into a wrapper plus an implementation.
    """
    return default if isinstance(value, typer.models.OptionInfo) else value


def _require(path: Path, hint: str):
    if not path.exists():
        console.print(f"[red]missing:[/red] {path}\n  {hint}")
        raise typer.Exit(1)
    return path


# --------------------------------------------------------------------------- phases


@app.command(name="build-snapshots")
def build_snapshots(
    force: bool = typer.Option(False, "--force"),
    windows: int = typer.Option(None, help="Override DATA_N_WINDOWS"),
    queries: int = typer.Option(None, help="Override DATA_QUERIES_PER_WINDOW"),
) -> None:
    """Phase 0: build C(t), Q(t) and the null slice."""
    _setup_logging()
    force, windows, queries = _opt(force, False), _opt(windows), _opt(queries)
    real_dir, null_dir = _slice_dirs()

    if not force and any(real_dir.glob("corpus_t*.parquet")):
        console.print(f"[yellow]snapshots already exist at {real_dir} (use --force)[/yellow]")
        return

    if settings.data.source == "amazon":
        from data.build_snapshots import build_snapshots as legacy_build

        for cat in (settings.data.category, settings.data.null_category):
            legacy_build(category=cat, force=force)
        console.print("[green]legacy Amazon snapshots built[/green]")
        return

    from data.corpora import load_corpus
    from data.windows import build_null_windows, build_windows

    corpus = load_corpus()
    real = build_windows(corpus, n_windows=windows, queries_per_window=queries, out_dir=real_dir)
    build_null_windows(corpus, n_windows=windows, queries_per_window=queries, out_dir=null_dir)

    m = real["manifest"]
    console.print(
        f"[green]OK[/green] {m['corpus']}: {m['n_docs']} docs "
        f"({100 * m['revised_fraction']:.1f}% revised), {m['n_windows']} windows, "
        f"{m['queries_per_window']} queries/window"
    )
    console.print(f"     revision schedule: {m['revision_schedule']}")
    console.print(f"     real -> {real_dir}\n     null -> {null_dir}")


@app.command()
def embed(force: bool = typer.Option(False, "--force")) -> None:
    """Phase 1: encode every window."""
    _setup_logging()
    force = _opt(force, False)
    from embeddings.encoder import Encoder

    enc = Encoder()
    for d in _slice_dirs():
        _require(d, "run: python main.py build-snapshots")
        for prefix in ("corpus", "queries"):
            embs = enc.encode_snapshot_dir(d, prefix=prefix, force=force)
            console.print(f"[green]OK[/green] {d.name}/{prefix}: {len(embs)} windows x {embs[0].shape}")


@app.command(name="compute-drift")
def compute_drift(null: bool = typer.Option(True, help="Also compute the null slice")) -> None:
    """Phase 2: D(t) with per-window permutation calibration."""
    _setup_logging()
    null = _opt(null, True)
    from drift.statistic import centroid_share_of_d, compute_drift_series, save_drift_series
    from embeddings.encoder import Encoder

    enc = Encoder()
    real_dir, null_dir = _slice_dirs()
    out = _results_dir()

    for tag, d in (("drift_series", real_dir), ("null_drift_series", null_dir)):
        if tag.startswith("null") and not null:
            continue
        _require(d, "run: python main.py build-snapshots")
        # The corpus channel is measured on the DISJOINT sample view: overlapping
        # windows are not exchangeable and break the permutation null (see
        # data/windows.py). Vectors are gathered from the corpus embeddings, so the
        # sample view costs no extra forward passes.
        C_full = enc.encode_snapshot_dir(d, "corpus")
        Q = enc.encode_snapshot_dir(d, "queries")
        sample_frames = [pd.read_parquet(p) for p in sorted(d.glob("sample_t*.parquet"))]
        if len(sample_frames) != len(C_full):
            raise RuntimeError(
                f"{d}: {len(sample_frames)} sample windows but {len(C_full)} corpus windows; "
                "rebuild with `python main.py build-snapshots --force`"
            )
        C = [C_full[t][sf["corpus_row"].to_numpy()] for t, sf in enumerate(sample_frames)]
        df = compute_drift_series(C, Q)
        save_drift_series(df, out, name=tag)
        console.print(
            f"[green]OK[/green] {tag}: {len(df)} windows, "
            f"alarms q={int(df.alarm_q.sum())} c={int(df.alarm_c.sum())}"
        )
        if tag == "drift_series":
            console.print(
                f"     legacy genre-gap share of the old D(t): "
                f"[yellow]{centroid_share_of_d(df):.1%}[/yellow] (why it was removed, F5)"
            )


@app.command(name="classify-regimes")
def classify_regimes() -> None:
    """Phase 3: the (z_q, z_c) regime plane."""
    _setup_logging()
    from drift.decomposition import decompose_and_classify

    out = _results_dir()
    path = _require(out / "drift_series.parquet", "run: python main.py compute-drift")
    df = decompose_and_classify(pd.read_parquet(path))
    df.to_parquet(out / "regime_series.parquet", index=False)

    console.print(f"[green]OK[/green] regimes -> {out / 'regime_series.parquet'}")
    console.print(f"     raw:   {df.raw_regime.value_counts().to_dict()}")
    console.print(f"     gated: {df.gated_regime.value_counts(dropna=False).to_dict()}")


@app.command(name="run-policy")
def run_policy(lam: float = typer.Option(None, help="Override POLICY_LAM")) -> None:
    """Phase 4: cost-aware action selection."""
    _setup_logging()
    lam = _opt(lam)
    from policy.engine import compute_policy

    out = _results_dir()
    drift = pd.read_parquet(_require(out / "drift_series.parquet", "run: compute-drift"))
    regime_path = out / "regime_series.parquet"
    regime = pd.read_parquet(regime_path) if regime_path.exists() else None

    df = compute_policy(drift, regime, lam=lam)
    df.to_parquet(out / "policy_decisions.parquet", index=False)

    t = Table(title="Policy")
    for c, s in [("t", "cyan"), ("z_q", "white"), ("z_c", "white"), ("action", "magenta"), ("objective", "yellow")]:
        t.add_column(c, style=s)
    for _, r in df.iterrows():
        t.add_row(str(int(r.window)), f"{r.z_q:+.2f}", f"{r.z_c:+.2f}", r.selected_action, f"{r.total_cost:.2f}")
    console.print(t)


@app.command(name="run-rag")
def run_rag(
    max_questions: int = typer.Option(0, help="0 = all queries in the window"),
) -> None:
    """Phase 5: closed loop -- serve window t from a stale index, apply the action, measure recall@k."""
    _setup_logging()
    max_questions = _opt(max_questions, 0)
    from embeddings.encoder import Encoder
    from evaluation.retrieval_quality import evaluate as eval_retrieval
    from retrieval.index import NumpyIndex
    from retrieval.select import displacement, select

    real_dir, _ = _slice_dirs()
    out = _results_dir()
    enc = Encoder()

    C = enc.encode_snapshot_dir(real_dir, "corpus")
    Q = enc.encode_snapshot_dir(real_dir, "queries")
    frames = [pd.read_parquet(p) for p in sorted(real_dir.glob("corpus_t*.parquet"))]
    qframes = [pd.read_parquet(p) for p in sorted(real_dir.glob("queries_t*.parquet"))]

    policy_path = out / "policy_decisions.parquet"
    policy = pd.read_parquet(policy_path) if policy_path.exists() else pd.DataFrame()

    def action_for(t: int) -> str:
        if policy.empty:
            return "a1_no_op"
        row = policy[policy.window == t]
        return str(row.selected_action.iloc[0]) if len(row) else "a1_no_op"

    rows: list[dict] = []
    for t in range(1, len(C)):
        stale_embs, fresh_embs = C[t - 1], C[t]
        qe = Q[t]
        gold = qframes[t]["gold_row"].to_numpy()
        if max_questions:
            qe, gold = qe[:max_questions], gold[:max_questions]

        action = action_for(t)
        idx = NumpyIndex.from_frame(frames[t], stale_embs, name=f"t{t:03d}")

        stale_q = eval_retrieval(idx, qe, gold)
        n_refreshed = 0

        if action == "a3_reindex":
            sel = select(
                stale_embs,
                settings.policy.reindex_budget,
                lambda r: displacement(stale_embs, fresh_embs, r),
                seed=t,
            )
            n_refreshed = idx.reindex_rows(sel.rows, fresh_embs)
        elif action in ("a4_finetune", "a5_escalate"):
            n_refreshed = idx.reindex_rows(np.arange(len(stale_embs)), fresh_embs)

        after_q = eval_retrieval(idx, qe, gold)

        oracle = NumpyIndex.from_frame(frames[t], fresh_embs, name="oracle")
        fresh_q = eval_retrieval(oracle, qe, gold)

        rows.append(
            {
                "window": t,
                "action": action,
                "n_refreshed": n_refreshed,
                "refresh_fraction": n_refreshed / len(stale_embs),
                "recall_stale": stale_q.recall_at_k,
                "recall_at_k": after_q.recall_at_k,
                "recall_full_reindex": fresh_q.recall_at_k,
                "mrr": after_q.mrr,
                "headroom": fresh_q.recall_at_k - stale_q.recall_at_k,
                "n_queries": after_q.n_queries,
            }
        )
        logger.info(
            f"t={t}: {action} refreshed {n_refreshed} -> recall {stale_q.recall_at_k:.4f} "
            f"-> {after_q.recall_at_k:.4f} (full {fresh_q.recall_at_k:.4f})"
        )

    df = pd.DataFrame(rows)
    df.to_parquet(out / "rag_quality.parquet", index=False)

    t = Table(title="Closed loop: stale index -> action -> recall@k")
    for c in ["t", "action", "refreshed", "stale", "after", "full", "headroom"]:
        t.add_column(c)
    for _, r in df.iterrows():
        t.add_row(
            str(int(r.window)), r.action, f"{r.refresh_fraction:.0%}",
            f"{r.recall_stale:.4f}", f"{r.recall_at_k:.4f}",
            f"{r.recall_full_reindex:.4f}", f"{r.headroom:+.4f}",
        )
    console.print(t)


@app.command()
def evaluate() -> None:
    """Phase 6: validation report."""
    _setup_logging()
    from evaluation.correlate import (
        auroc_from_churn, compute_lead_time, compute_spearman_rho, save_validation_report,
    )
    from evaluation.null_corpus_test import compute_far
    from policy.engine import fit_loss_model

    out = _results_dir()
    drift = pd.read_parquet(_require(out / "drift_series.parquet", "run: compute-drift"))
    quality_path = out / "rag_quality.parquet"
    quality = pd.read_parquet(quality_path) if quality_path.exists() else pd.DataFrame()

    report: dict = {"n_windows": len(drift)}

    if len(quality):
        report["spearman_rho"] = compute_spearman_rho(drift, quality)
        report["lead_time"] = compute_lead_time(drift, quality)
        report["loss_model"] = fit_loss_model(drift, quality)
    else:
        report["note"] = "no rag_quality.parquet; run `python main.py run-rag` for rho and lead time"

    real_dir, _ = _slice_dirs()
    manifest_path = real_dir / "manifest.json"
    if manifest_path.exists():
        man = json.loads(manifest_path.read_text())
        churn = pd.DataFrame(
            [{"window": w["window"], "n_revised": w["n_revised_by_now"]} for w in man["windows"]]
        )
        churn["churn"] = churn["n_revised"].diff().fillna(0)
        report["auroc"] = auroc_from_churn(drift, churn)

    null_path = out / "null_drift_series.parquet"
    if null_path.exists():
        null_df = pd.read_parquet(null_path)
        report["false_alarm_rate"] = compute_far(null_df)
        report["false_alarm_rate_legacy_fixed_threshold"] = compute_far(null_df, use_calibrated=False)

    save_validation_report(report, out)
    console.print(json.dumps(report, indent=2, default=str))


# --------------------------------------------------------------------------- experiments


@app.command()
def bakeoff(
    budgets: str = typer.Option("0.05,0.10,0.20,0.35,0.50"),
    seeds: int = typer.Option(5),
) -> None:
    """Compare selective re-index selectors, including the closest deployed prior art."""
    _setup_logging()
    budgets, seeds = _opt(budgets, "0.05,0.10,0.20,0.35,0.50"), _opt(seeds, 5)
    from embeddings.encoder import Encoder
    from evaluation.retrieval_quality import evaluate as eval_retrieval, share_of_gain
    from retrieval.index import NumpyIndex
    from retrieval.select import displacement, select

    real_dir, _ = _slice_dirs()
    enc = Encoder()
    C = enc.encode_snapshot_dir(real_dir, "corpus")
    Q = enc.encode_snapshot_dir(real_dir, "queries")
    frames = [pd.read_parquet(p) for p in sorted(real_dir.glob("corpus_t*.parquet"))]
    qframes = [pd.read_parquet(p) for p in sorted(real_dir.glob("queries_t*.parquet"))]

    stale, fresh = C[0], C[-1]
    qe, gold = Q[-1], qframes[-1]["gold_row"].to_numpy()
    m = displacement(stale, fresh)
    edited = m > 1e-6

    rs = eval_retrieval(NumpyIndex.from_frame(frames[0], stale), qe, gold).recall_at_k
    rf = eval_retrieval(NumpyIndex.from_frame(frames[0], fresh), qe, gold).recall_at_k
    console.print(f"stale recall@k {rs:.4f} -> full re-index {rf:.4f}   headroom [bold]{rf - rs:+.4f}[/bold]")
    if rf - rs < 1e-4:
        console.print("[yellow]no headroom: selective re-index is vacuous on this slice[/yellow]")
        return

    rows: list[dict] = []
    for budget in [float(b) for b in budgets.split(",")]:
        for name in ["random", "probe_knn", "cluster_threshold", "changelog", "oracle"]:
            vals = []
            for s in range(seeds):
                sel = select(
                    stale, budget, lambda r: displacement(stale, fresh, r),
                    selector=name, true_magnitude=m, edited=edited, seed=s,
                )
                idx = NumpyIndex.from_frame(frames[0], stale)
                idx.reindex_rows(sel.rows, fresh)
                vals.append(eval_retrieval(idx, qe, gold).recall_at_k)
            rows.append(
                {
                    "budget": budget, "selector": name,
                    "recall_mean": float(np.mean(vals)), "recall_sd": float(np.std(vals, ddof=1)) if seeds > 1 else 0.0,
                    "share_of_gain": share_of_gain(float(np.mean(vals)), rs, rf), "seeds": seeds,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(_results_dir() / "bakeoff.csv", index=False)

    t = Table(title=f"Selective re-index ({seeds} seeds)")
    t.add_column("budget")
    for name in ["random", "probe_knn", "cluster_threshold", "changelog", "oracle"]:
        t.add_column(name)
    for budget, grp in df.groupby("budget"):
        g = grp.set_index("selector")
        t.add_row(f"{budget:.0%}", *[f"{g.loc[n, 'share_of_gain']:.3f}" for n in
                                     ["random", "probe_knn", "cluster_threshold", "changelog", "oracle"]])
    console.print(t)
    # One print per markup span: rich parses each call independently, so a tag opened
    # in one console.print and closed in the next raises MarkupError.
    console.print(
        "[dim]Share of recoverable recall captured. probe_knn vs cluster_threshold is "
        "the comparison that matters for novelty; FINDINGS.md 12.5 found them tied.[/dim]"
    )


@app.command()
def transfer(
    n_calibrate: int = typer.Option(None), n_deploy: int = typer.Option(None), trials: int = typer.Option(60)
) -> None:
    """Threshold transfer under a shift in traffic volume."""
    _setup_logging()
    n_calibrate, n_deploy, trials = _opt(n_calibrate), _opt(n_deploy), _opt(trials, 60)
    from embeddings.encoder import Encoder
    from evaluation.transfer import run_transfer

    real_dir, _ = _slice_dirs()
    pool = Encoder().encode_snapshot_dir(real_dir, "corpus")[0]
    n = len(pool)
    n_calibrate = n_calibrate or max(n // 2, 50)
    n_deploy = n_deploy or max(n // 20, 20)

    res = run_transfer(pool, n_calibrate=n_calibrate, n_deploy=n_deploy, trials=trials)
    (_results_dir() / "transfer.json").write_text(json.dumps(res, indent=2))

    t = Table(title=f"Fixed threshold: calibrate n={n_calibrate}, deploy n={n_deploy} (all windows i.i.d.)")
    for c in ["statistic", "FAR big->small", "FAR small->big", "FAR native", "inflation"]:
        t.add_column(c)
    for name, r in res["fixed_threshold"].items():
        t.add_row(name, f"{r['far_calibrate_big_deploy_small']:.3f}",
                  f"{r['far_calibrate_small_deploy_big']:.3f}",
                  f"{r['far_native_at_n_calibrate']:.3f}",
                  f"{r['score_inflation_measured']:.2f}x")
    console.print(t)
    console.print(f"predicted inflation from sqrt(n) ratio: {res['predicted_inflation_from_sqrt_n']:.2f}x")
    console.print(f"[green]permutation-calibrated, same sizes:[/green] {res['permutation_calibrated']} "
                  f"(nominal {res['alpha']})")


@app.command()
def power(window: int = typer.Option(None), trials: int = typer.Option(40)) -> None:
    """Detection power swept down through the false-alarm floor."""
    _setup_logging()
    window, trials = _opt(window), _opt(trials, 40)
    from embeddings.encoder import Encoder
    from evaluation.power import corpus_operating_point, run_power_sweep

    real_dir, _ = _slice_dirs()
    enc = Encoder()
    C = enc.encode_snapshot_dir(real_dir, "corpus")
    pool = C[0]
    window = window or max(len(pool) // 3, 30)

    res = run_power_sweep(pool, window=window, trials=trials)
    op = corpus_operating_point(C[0], C[-1])
    res["corpus_operating_point"] = op
    (_results_dir() / "power.json").write_text(json.dumps(res, indent=2))

    t = Table(title=f"Power at a calibrated {res['alpha']:.0%} FAR, n={window}")
    for c in ["scale", "displ (all docs)", "displ (moved)", "sliced_w1", "mmd", "centroid"]:
        t.add_column(c)
    for r in res["sweep"]:
        t.add_row(f"{r['scale']:.2f}", f"{r['mean_displacement_all_docs']:.3f}",
                  f"{r['mean_displacement_moved_docs']:.3f}",
                  f"{r['power_sliced_w1']:.3f}", f"{r['power_mmd']:.3f}", f"{r['power_centroid']:.3f}")
    console.print(t)
    console.print(
        f"[bold]This corpus sits at displacement {op['mean_displacement']:.3f} over all "
        f"documents ({op['mean_displacement_moved']:.3f} over the {op['fraction_moved']:.0%} "
        f"that moved).[/bold]"
    )
    console.print(
        "[dim]Read its power off the matching column. Comparing an all-docs corpus "
        "figure against the moved-docs sweep column understates it by the moved "
        "fraction -- that mix-up is live in FINDINGS 12.9.[/dim]"
    )


@app.command()
def audit() -> None:
    """Structural self-checks for the defects this rewrite addresses."""
    _setup_logging()
    from drift.decomposition import reachability_report
    from policy.engine import sweep_lambda

    t = Table(title="Structural checks")
    t.add_column("check"); t.add_column("result"); t.add_column("finding")

    reach = reachability_report()
    t.add_row("all 4 regimes reachable", "PASS" if all(reach.values()) else f"FAIL {reach}", "F11")

    actions = set(sweep_lambda().selected_action.unique())
    t.add_row(f"policy selects >1 action ({len(actions)})", "PASS" if len(actions) > 1 else "FAIL", "F6")

    out = settings.drift.results_dir
    dp = out / "drift_series.parquet"
    if dp.exists():
        from drift.statistic import centroid_share_of_d

        df = pd.read_parquet(dp)
        t.add_row("genre-gap share of legacy D(t)", f"{centroid_share_of_d(df):.1%}", "F5")
        w = df.iloc[0]
        t.add_row("W1 CI is a data bootstrap",
                  f"width {w.w1_c_ci_hi - w.w1_c_ci_lo:.5f} vs mc_se {w.w1_c_mc_se:.6f}", "F12")
    np_ = out / "null_drift_series.parquet"
    if np_.exists():
        from evaluation.null_corpus_test import compute_far

        far = compute_far(pd.read_parquet(np_))
        t.add_row("null FAR computable", f"{far['far']:.3f} over {far['n_windows']} windows", "F2, F21")
    console.print(t)


@app.command(name="run-all")
def run_all(force: bool = typer.Option(False, "--force")) -> None:
    """Run every phase end to end."""
    force = _opt(force, False)
    console.print("[bold blue]0 data[/bold blue]");        build_snapshots(force=force)
    # never force re-embedding here: the cache is content-addressed, so changed text
    # already misses and unchanged text is correct to reuse. Forcing would discard
    # tens of minutes of forward passes for nothing.
    console.print("[bold blue]1 embed[/bold blue]");       embed(force=False)
    console.print("[bold blue]2 drift[/bold blue]");       compute_drift(null=True)
    console.print("[bold blue]3 regimes[/bold blue]");     classify_regimes()
    console.print("[bold blue]4 policy[/bold blue]");      run_policy(lam=None)
    console.print("[bold blue]5 rag[/bold blue]");         run_rag(max_questions=0)
    console.print("[bold blue]6 evaluate[/bold blue]");    evaluate()
    console.print("[bold blue]7 audit[/bold blue]");       audit()
    console.print("\n[bold green]done[/bold green] results in " + str(settings.drift.results_dir))


# --------------------------------------------------------------------------- legacy


@app.command()
def download(force: bool = typer.Option(False, "--force")) -> None:
    """LEGACY: download Amazon Reviews 2023."""
    _setup_logging()
    force = _opt(force, False)
    from data.download import download as _dl

    _dl(force=force)


@app.command(name="inject-drift")
def inject_drift() -> None:
    """LEGACY: build the synthetic delisting slice."""
    _setup_logging()
    from data.inject_drift import build_labeled_slice

    res = build_labeled_slice()
    console.print(f"[green]OK[/green] {res['snapshots_dir']}")


@app.command()
def dashboard() -> None:
    """Launch the Streamlit dashboard."""
    import subprocess, sys

    subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"])


if __name__ == "__main__":
    app()
