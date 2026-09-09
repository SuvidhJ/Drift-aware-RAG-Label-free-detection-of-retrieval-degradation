"""LEGACY: synthetic delisting slice for the Amazon path.

Repairs
-------
F1  The labelled slice was written to ``snapshots/{category}_labeled/`` but only the
    *corpus* files were written there -- the query snapshots were left behind in
    ``snapshots/{category}/``.  ``main.py`` then read the corpus from the undrifted
    directory and the labels from the drifted one, so AUROC paired labels from one
    slice against a signal computed on the other.  The labelled directory is now
    self-contained: queries are copied alongside, and the function returns the
    directory the rest of the pipeline should read.

F19 The manifest emitted one row per (asin, window) with ``delisted`` flipping at the
    offset, which yields exactly one positive window under the "delisted at t"
    reading the evaluation code used.  It now records a per-window summary as well,
    so a balanced label can be derived.

The default DATA_SOURCE=revision path does not use this module: it has real
revisions and does not need injected ones.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from configs.settings import settings


def build_labeled_slice(
    category: str | None = None,
    snapshots_dir: Path | None = None,
    seed: int = 42,
) -> dict[str, Path]:
    """Freeze the corpus, delist a fraction of products at t0+k, keep real queries."""
    category = category or settings.data.category
    snapshots_dir = snapshots_dir or settings.data.snapshots_dir
    offset = 3
    frac = 0.10

    rng = np.random.default_rng(seed)
    cat_dir = snapshots_dir / category
    corpus_paths = sorted(cat_dir.glob("corpus_t*.parquet"))
    query_paths = sorted(cat_dir.glob("queries_t*.parquet"))

    if len(corpus_paths) < offset + 2:
        raise ValueError(
            f"need at least {offset + 2} windows for the labelled slice; found {len(corpus_paths)} "
            f"under {cat_dir}"
        )

    labeled_dir = snapshots_dir / f"{category}_labeled"
    labeled_dir.mkdir(parents=True, exist_ok=True)

    t0_df = pd.read_parquet(corpus_paths[0])
    id_col = "doc_id" if "doc_id" in t0_df.columns else (
        "parent_asin" if "parent_asin" in t0_df.columns else "asin"
    )
    all_ids = t0_df[id_col].unique()
    n_delist = max(1, int(len(all_ids) * frac))
    delisted = set(rng.choice(all_ids, size=n_delist, replace=False).tolist())

    logger.info(f"delisting {n_delist}/{len(all_ids)} documents from window t={offset}")

    per_window: list[dict] = []
    for i, path in enumerate(corpus_paths):
        df = pd.read_parquet(path)
        removed = 0
        if i >= offset:
            before = len(df)
            df = df[~df[id_col].isin(delisted)].copy()
            removed = before - len(df)
        df.to_parquet(labeled_dir / path.name, index=False)
        per_window.append(
            {"window": i, "n_docs": len(df), "n_removed": removed, "any_delisted": int(i >= offset)}
        )

    # F1: the labelled slice must carry its own queries, or downstream code silently
    # pairs this corpus with the undrifted one.
    for path in query_paths:
        shutil.copyfile(path, labeled_dir / path.name)
    if not query_paths:
        logger.warning(f"no queries_t*.parquet under {cat_dir}; the labelled slice has no query channel")

    manifest = pd.DataFrame(
        [{"doc_id": d, "delisted_from_window": offset} for d in sorted(delisted)]
    )
    manifest_path = labeled_dir / "delist_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    window_path = labeled_dir / "delist_by_window.csv"
    pd.DataFrame(per_window).to_csv(window_path, index=False)

    logger.info(f"labelled slice -> {labeled_dir} ({len(query_paths)} query windows copied)")
    return {
        "snapshots_dir": labeled_dir,
        "manifest": manifest_path,
        "by_window": window_path,
    }


def build_null_corpus_slice(
    null_category: str | None = None,
    snapshots_dir: Path | None = None,
) -> Path:
    """Return the unmodified second category, used as the null slice.

    This only locates a directory -- it computes nothing. The null *drift series* is
    what the FAR check needs, and it is produced by running the drift phase against
    this directory; on the revision path use data.windows.build_null_windows instead,
    which builds an exchangeable null rather than relying on a second category being
    quiet.
    """
    null_category = null_category or settings.data.null_category
    snapshots_dir = snapshots_dir or settings.data.snapshots_dir

    null_dir = snapshots_dir / null_category
    if not null_dir.exists():
        raise FileNotFoundError(
            f"null category snapshots not found at {null_dir}. "
            f"Run build-snapshots for {null_category} first."
        )
    if not sorted(null_dir.glob("corpus_t*.parquet")):
        raise FileNotFoundError(f"{null_dir} exists but holds no corpus_t*.parquet")

    logger.info(f"null slice: {null_category} as-is")
    return null_dir


if __name__ == "__main__":
    build_labeled_slice()
    build_null_corpus_slice()
