"""Build the time-windowed corpus C(t) and query Q(t) snapshots.

Two views of the corpus, and why both are needed
------------------------------------------------
``corpus_t*.parquet`` holds *every* document at its live revision as of t.  A
document whose later revision lands in window w is represented by ``text0`` for
t < w and ``text1`` for t >= w.  This is the paired view an index actually has, and
it is what retrieval evaluation and selective re-index use: the same documents,
stale vectors versus fresh ones.

``sample_t*.parquet`` holds a *disjoint* random subset of documents per window, again
at their live revision.  This is the view the drift detector uses, and it must be
disjoint for a measurable reason.  The two-sample permutation null in drift/null.py
assumes the two windows are independent samples.  If consecutive windows contain the
same documents, the observed split pairs each document with itself and is therefore
far more balanced than any reshuffle of the pooled set, so the observed statistic
sits below the entire null distribution.  Measured on this corpus with overlapping
windows, the corpus-channel z-score was pinned near -5 in every window -- including
windows with real revisions -- while the query channel, whose windows are drawn
independently, behaved correctly at z ~ N(0, 1).  A detector that reports -5 when
nothing is happening cannot report anything useful when something is.

Partitioning fixes it exactly rather than approximately: with no shared documents the
windows are genuinely exchangeable under the null, and the systematic part of the
comparison is what it should be -- the growing fraction of documents serving their
revised text, plus the shifting topic mixture.

The query channel
-----------------
Queries are sentence windows drawn from the *live* text of a document, so each query
has a known gold document and the query distribution moves for two independent
reasons:

  1. content the queries are drawn from is revised over time, and
  2. the topic mixture shifts -- hub weights interpolate from one random simplex
     point to another across the run, controlled by ``query_topic_drift``.

(2) is what makes query drift separable from corpus drift, so the regime plane in
drift/decomposition.py has all four quadrants reachable in data rather than only in
principle.  The previous pipeline emitted the literal string "What do users say about
this product?" for every row, making W1(Q_t, Q_{t-1}) identically zero (F3).

The null corpus
---------------
``build_null_windows`` draws every window from the *stale* content only, so the two
windows really are exchangeable and any alarm is a genuine false alarm.  The previous
code had no null writer at all: ``main.py`` guarded on
``results/null_drift_series.parquet`` and nothing in the repository ever wrote it, yet
a false-alarm rate of 3.1% was reported (F2).  Window count is a free parameter here,
so FAR granularity is not stuck at 1/12 (F21).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from configs.settings import settings
from data.corpora import Document, RevisionCorpus, load_corpus, parse_timestamp

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
MIN_SENTENCE_CHARS = 40
MIN_QUERY_CHARS = 60


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if len(s.strip()) >= MIN_SENTENCE_CHARS]


def assign_revision_windows(
    corpus: RevisionCorpus, n_windows: int, seed: int
) -> np.ndarray:
    """Window index at which each document switches to its later revision.

    Unrevised documents get ``n_windows`` (never switches, since windows are 0..T-1).
    Revised documents are placed by real timestamp when available; otherwise they are
    spread uniformly at random and ``corpus.real_timestamps`` records that this
    happened, so it can never be mistaken for a measured schedule.
    """
    rng = np.random.default_rng(seed)
    n = len(corpus.docs)
    out = np.full(n, n_windows, dtype=int)

    revised_idx = [i for i, d in enumerate(corpus.docs) if d.revised]
    if not revised_idx:
        return out

    if corpus.real_timestamps:
        stamps = {i: parse_timestamp(corpus.docs[i].ts1) for i in revised_idx}
        dated = {i: t for i, t in stamps.items() if t is not None}
        if dated:
            times = np.array([dated[i].timestamp() for i in sorted(dated)])
            lo, hi = times.min(), times.max()
            span = max(hi - lo, 1.0)
            for i in sorted(dated):
                frac = (dated[i].timestamp() - lo) / span
                # windows 1..T-1: nothing can change before the first window is observed
                out[i] = int(np.clip(1 + frac * (n_windows - 1), 1, n_windows - 1))
        undated = [i for i in revised_idx if i not in dated]
        if undated:
            out[undated] = rng.integers(1, n_windows, size=len(undated))
    else:
        out[revised_idx] = rng.integers(1, n_windows, size=len(revised_idx))

    return out


def live_text(doc: Document, t: int, revision_window: int) -> str:
    return doc.text1 if t >= revision_window else doc.text0


def hub_weight_schedule(
    hubs: list[str], n_windows: int, drift: float, seed: int
) -> np.ndarray:
    """(n_windows, n_hubs) query topic mixture, interpolating between two simplex points."""
    rng = np.random.default_rng(seed)
    h = len(hubs)
    start = rng.dirichlet(np.ones(h) * 2.0)
    end = rng.dirichlet(np.ones(h) * 2.0)
    end = (1.0 - drift) * start + drift * end

    ts = np.linspace(0.0, 1.0, n_windows)[:, None]
    w = (1.0 - ts) * start[None, :] + ts * end[None, :]
    return w / w.sum(axis=1, keepdims=True)


def revision_only_sentences(doc: Document) -> list[str]:
    """Sentences present in the later revision but not the earlier one."""
    before = set(sentences(doc.text0))
    return [s for s in sentences(doc.text1) if s not in before]


def build_windows(
    corpus: RevisionCorpus | None = None,
    n_windows: int | None = None,
    queries_per_window: int | None = None,
    query_topic_drift: float = 0.8,
    query_targeted_frac: float | None = None,
    out_dir: Path | None = None,
    seed: int | None = None,
) -> dict:
    """Write corpus_t*.parquet and queries_t*.parquet plus a manifest.

    ``query_targeted_frac`` controls how often a query about an already-revised
    document is drawn from text that exists *only* in the revision.

    This matters more than it looks. Queries drawn from any live sentence are almost
    all answerable from the stale index too, because most of a revised document is
    unchanged - measured on the 7,346-document arXiv build, stale recall@10 and
    full-re-index recall@10 were both 0.9433, headroom 0.0000, which makes every
    selective-re-index comparison vacuous. Targeting the changed text creates the
    staleness the index is supposed to suffer from, and is what the bake-off in
    FINDINGS 12.4-12.5 used ("targeted" queries).
    """
    cfg = settings.data
    corpus = corpus or load_corpus()
    n_windows = cfg.n_windows if n_windows is None else n_windows
    queries_per_window = cfg.queries_per_window if queries_per_window is None else queries_per_window
    query_targeted_frac = (
        cfg.query_targeted_frac if query_targeted_frac is None else query_targeted_frac
    )
    seed = cfg.seed if seed is None else seed
    out_dir = Path(out_dir or cfg.snapshots_dir / corpus.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(corpus.docs)
    if n < cfg.min_window_samples:
        raise ValueError(
            f"corpus {corpus.name!r} has {n} documents but DATA_MIN_WINDOW_SAMPLES is "
            f"{cfg.min_window_samples}. Lower the setting or pick a larger corpus. "
            "(This check exists because min_window_samples was declared as a threat "
            "mitigation and then read by nothing -- FINDINGS.md F22.)"
        )

    rev_window = assign_revision_windows(corpus, n_windows, seed)
    hubs = sorted({d.hub for d in corpus.docs})
    hub_index = {h: j for j, h in enumerate(hubs)}
    weights = hub_weight_schedule(hubs, n_windows, query_topic_drift, seed + 1)

    docs_by_hub: dict[str, list[int]] = {h: [] for h in hubs}
    for i, d in enumerate(corpus.docs):
        docs_by_hub[d.hub].append(i)

    rng = np.random.default_rng(seed + 2)
    corpus_paths: list[Path] = []
    sample_paths: list[Path] = []
    query_paths: list[Path] = []
    per_window: list[dict] = []

    # disjoint partition for the drift-detection view (see the module docstring)
    partition = np.array_split(rng.permutation(n), n_windows)

    for t in range(n_windows):
        texts = [live_text(d, t, int(rev_window[i])) for i, d in enumerate(corpus.docs)]
        c_df = pd.DataFrame(
            {
                "doc_id": [d.doc_id for d in corpus.docs],
                "hub": [d.hub for d in corpus.docs],
                "text": texts,
                "revised_by_now": rev_window <= t,
                "revision_window": rev_window,
                "window": t,
            }
        )
        c_path = out_dir / f"corpus_t{t:03d}.parquet"
        c_df.to_parquet(c_path, index=False)
        corpus_paths.append(c_path)

        rows = np.sort(partition[t])
        s_df = c_df.iloc[rows].copy()
        s_df["corpus_row"] = rows
        s_path = out_dir / f"sample_t{t:03d}.parquet"
        s_df.to_parquet(s_path, index=False)
        sample_paths.append(s_path)

        # queries: hub sampled from the drifting mixture, text from the live revision
        q_rows: list[dict] = []
        hub_p = weights[t]
        attempts = 0
        while len(q_rows) < queries_per_window and attempts < queries_per_window * 40:
            attempts += 1
            hub = hubs[int(rng.choice(len(hubs), p=hub_p))]
            pool = docs_by_hub[hub]
            if not pool:
                continue
            i = int(rng.choice(pool))
            doc = corpus.docs[i]

            targeted = False
            sents = None
            if rev_window[i] <= t and rng.random() < query_targeted_frac:
                only_new = revision_only_sentences(doc)
                if only_new:
                    sents, targeted = only_new, True
            if sents is None:
                sents = sentences(texts[i])
            if not sents:
                continue

            j = int(rng.integers(0, len(sents)))
            q_text = " ".join(sents[j : j + 2])
            if len(q_text) < MIN_QUERY_CHARS:
                continue
            q_rows.append(
                {
                    "query_id": f"t{t:03d}_q{len(q_rows):05d}",
                    "text": q_text,
                    "gold_doc_id": doc.doc_id,
                    "gold_row": i,
                    "hub": hub,
                    "window": t,
                    "targeted": targeted,
                }
            )

        if len(q_rows) < max(10, queries_per_window // 4):
            raise RuntimeError(
                f"window {t}: only {len(q_rows)} queries could be built from "
                f"{len(hubs)} hubs. The corpus text may be too short to split into "
                f"sentences of >= {MIN_SENTENCE_CHARS} chars."
            )

        q_df = pd.DataFrame(q_rows)
        q_path = out_dir / f"queries_t{t:03d}.parquet"
        q_df.to_parquet(q_path, index=False)
        query_paths.append(q_path)

        per_window.append(
            {
                "window": t,
                "n_corpus": len(c_df),
                "n_sample": len(s_df),
                "n_queries": len(q_df),
                "n_revised_by_now": int((rev_window <= t).sum()),
                "n_targeted_queries": int(q_df["targeted"].sum()) if "targeted" in q_df else 0,
                "hub_weights": {h: float(hub_p[hub_index[h]]) for h in hubs},
            }
        )
        logger.info(
            f"t={t}: {len(c_df)} docs ({int((rev_window <= t).sum())} on their later "
            f"revision), {len(q_df)} queries"
        )

    manifest = {
        **corpus.summary(),
        "n_windows": n_windows,
        "queries_per_window": queries_per_window,
        "query_topic_drift": query_topic_drift,
        "query_targeted_frac": query_targeted_frac,
        "seed": seed,
        "revision_schedule": "real ts1 timestamps" if corpus.real_timestamps else "uniform random (corpus has no real timestamps)",
        "hubs": hubs,
        "windows": per_window,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return {
        "corpus": corpus_paths,
        "sample": sample_paths,
        "queries": query_paths,
        "manifest": manifest,
        "dir": out_dir,
    }


def build_null_windows(
    corpus: RevisionCorpus | None = None,
    n_windows: int | None = None,
    queries_per_window: int | None = None,
    subsample: float = 0.6,
    out_dir: Path | None = None,
    seed: int | None = None,
) -> dict:
    """Write a null slice in which nothing drifts.

    Every window draws a fresh random subsample of the *stale* corpus and queries
    built from stale text only. Successive windows are therefore exchangeable by
    construction, so any alarm the detector raises here is a false alarm and the rate
    at which it does so is the FAR.
    """
    cfg = settings.data
    corpus = corpus or load_corpus()
    n_windows = cfg.n_windows if n_windows is None else n_windows
    queries_per_window = cfg.queries_per_window if queries_per_window is None else queries_per_window
    seed = (cfg.seed + 7919) if seed is None else seed
    out_dir = Path(out_dir or cfg.snapshots_dir / f"{corpus.name}_null")
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(corpus.docs)
    rng = np.random.default_rng(seed)
    # Disjoint, for the same reason the real slice partitions: overlapping windows
    # are not exchangeable and drive the permutation z-score negative regardless of
    # whether anything drifted.
    partition = np.array_split(rng.permutation(n), n_windows)
    take = min(len(g) for g in partition)

    corpus_paths: list[Path] = []
    sample_paths: list[Path] = []
    query_paths: list[Path] = []

    for t in range(n_windows):
        idx = np.sort(partition[t])
        docs = [corpus.docs[i] for i in idx]
        c_df = pd.DataFrame(
            {
                "doc_id": [d.doc_id for d in docs],
                "hub": [d.hub for d in docs],
                "text": [d.text0 for d in docs],
                "revised_by_now": False,
                "revision_window": -1,
                "window": t,
            }
        )
        # In the null slice each window IS its own partition group, so the sample view
        # and the corpus view coincide and corpus_row is the identity within the frame.
        c_df["doc_index"] = idx
        c_df["corpus_row"] = np.arange(len(idx))
        c_path = out_dir / f"corpus_t{t:03d}.parquet"
        c_df.to_parquet(c_path, index=False)
        corpus_paths.append(c_path)

        s_path = out_dir / f"sample_t{t:03d}.parquet"
        c_df.to_parquet(s_path, index=False)
        sample_paths.append(s_path)

        # Queries are drawn from the WHOLE stale corpus, not from this window's
        # partition group. Drawing them per group would make consecutive query
        # windows samples of different document sets, which is a real distributional
        # difference, not a null -- measured that way the query channel alarmed in
        # 5 of 5 null windows at z up to +4.9. Sampling from one fixed pool makes
        # successive query windows i.i.d., which is what a null slice has to be.
        q_rows: list[dict] = []
        attempts = 0
        while len(q_rows) < queries_per_window and attempts < queries_per_window * 40:
            attempts += 1
            r = int(rng.integers(0, n))
            doc = corpus.docs[r]
            sents = sentences(doc.text0)
            if not sents:
                continue
            j = int(rng.integers(0, len(sents)))
            q_text = " ".join(sents[j : j + 2])
            if len(q_text) < MIN_QUERY_CHARS:
                continue
            q_rows.append(
                {
                    "query_id": f"null_t{t:03d}_q{len(q_rows):05d}",
                    "text": q_text,
                    "gold_doc_id": doc.doc_id,
                    "gold_row": r,
                    "hub": doc.hub,
                    "window": t,
                }
            )

        q_path = out_dir / f"queries_t{t:03d}.parquet"
        pd.DataFrame(q_rows).to_parquet(q_path, index=False)
        query_paths.append(q_path)

    manifest = {
        "corpus": f"{corpus.name}_null",
        "kind": "null slice: stale content only, disjoint partition per window",
        "n_windows": n_windows,
        "n_corpus_per_window": take,
        "queries_per_window": queries_per_window,
        "seed": seed,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info(f"null slice: {n_windows} windows x {take} docs -> {out_dir}")

    return {
        "corpus": corpus_paths,
        "sample": sample_paths,
        "queries": query_paths,
        "manifest": manifest,
        "dir": out_dir,
    }


def load_window_frames(snapshot_dir: Path, prefix: str) -> list[pd.DataFrame]:
    paths = sorted(Path(snapshot_dir).glob(f"{prefix}_t*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"no {prefix}_t*.parquet under {snapshot_dir}. Run: python main.py build-snapshots"
        )
    return [pd.read_parquet(p) for p in paths]
