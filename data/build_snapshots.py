"""LEGACY: build time-windowed snapshots from the Amazon dumps.

Superseded by data/windows.py, which works on the real revision corpora. Kept and
repaired (F4, F22) so the original experiment stays reproducible.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from configs.settings import settings


def _parse_timestamps(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    """Normalise unix-seconds or unix-milliseconds to pandas Timestamp."""
    if col not in df.columns:
        candidates = ["timestamp", "unixReviewTime", "unix_time", "time"]
        for c in candidates:
            if c in df.columns:
                col = c
                break
        else:
            raise KeyError(f"No timestamp column found in {list(df.columns)}")

    ts = pd.to_numeric(df[col], errors="coerce")
    if ts.median() > 1e12:
        ts = ts / 1_000
    df = df.copy()
    df["_ts"] = pd.to_datetime(ts, unit="s", utc=True, errors="coerce")
    df = df.dropna(subset=["_ts"])
    return df


def _build_period_index(df: pd.DataFrame, unit: str) -> pd.DataFrame:
    freq = "ME" if unit == "monthly" else "W"
    df["_period"] = df["_ts"].dt.to_period(freq[0])
    return df


def _corpus_text(meta: pd.DataFrame) -> pd.DataFrame:
    """Build a product-level text representation for embedding."""
    text_parts = []
    for col in ["title", "description", "features", "details"]:
        if col in meta.columns:
            text_parts.append(meta[col].fillna("").astype(str))

    if not text_parts:
        raise ValueError("Meta dataframe has no usable text columns")

    meta = meta.copy()
    meta["text"] = text_parts[0]
    for part in text_parts[1:]:
        meta["text"] = meta["text"] + " " + part

    meta["text"] = meta["text"].str.strip().str[:2048]
    return meta


def _query_text(qa: pd.DataFrame) -> pd.DataFrame:
    qa = qa.copy()
    q_col = "question" if "question" in qa.columns else "questionText"
    if q_col not in qa.columns:
        raise KeyError(f"No question column in Q&A dataframe: {list(qa.columns)}")
    qa["text"] = qa[q_col].fillna("").astype(str).str[:512]
    return qa


def build_snapshots(
    category: str | None = None,
    raw_dir: Path | None = None,
    snapshots_dir: Path | None = None,
    force: bool = False,
) -> dict[str, list[Path]]:
    category = category or settings.data.category
    raw_dir = raw_dir or settings.data.raw_dir
    snapshots_dir = snapshots_dir or settings.data.snapshots_dir
    unit = settings.data.window_unit

    cat_snap_dir = snapshots_dir / category
    cat_snap_dir.mkdir(parents=True, exist_ok=True)

    if not force and any(cat_snap_dir.glob("corpus_t*.parquet")):
        logger.info(f"Snapshots for {category} already exist; skipping (use --force)")
        corpus_paths = sorted(cat_snap_dir.glob("corpus_t*.parquet"))
        query_paths = sorted(cat_snap_dir.glob("queries_t*.parquet"))
        return {"corpus": corpus_paths, "queries": query_paths}

    meta_path = raw_dir / f"meta_{category}.parquet"
    reviews_path = raw_dir / f"reviews_{category}.parquet"
    qa_path = raw_dir / f"qa_{category}.parquet"

    if not meta_path.exists():
        raise FileNotFoundError(f"Meta not found: {meta_path}. Run data/download.py first.")

    logger.info(f"Loading raw data for {category}")
    meta = pd.read_parquet(meta_path)
    reviews = pd.read_parquet(reviews_path)
    qa = pd.read_parquet(qa_path)

    reviews = _parse_timestamps(reviews)
    qa = _parse_timestamps(qa)

    reviews = _build_period_index(reviews, unit)
    qa = _build_period_index(qa, unit)

    meta = _corpus_text(meta)
    qa = _query_text(qa)

    asin_col = "parent_asin" if "parent_asin" in meta.columns else "asin"
    review_asin = "parent_asin" if "parent_asin" in reviews.columns else "asin"

    asin_first_seen = (
        reviews.groupby(review_asin)["_period"].min().rename("first_period")
    )
    asin_last_seen = (
        reviews.groupby(review_asin)["_period"].max().rename("last_period")
    )

    meta = meta.merge(asin_first_seen, left_on=asin_col, right_index=True, how="left")
    meta = meta.merge(asin_last_seen, left_on=asin_col, right_index=True, how="left")

    all_periods = sorted(set(reviews["_period"].unique()) | set(qa["_period"].unique()))
    logger.info(f"Found {len(all_periods)} {unit} windows")

    corpus_paths: list[Path] = []
    query_paths: list[Path] = []

    # F4: liveness is reconstructed by joining a truncated meta dump against a
    # truncated review dump, two independently ordered files whose ASIN overlap is
    # small, so most products end up with NaN bounds. Treating NaN as "alive" made
    # them alive in every window and the snapshots near-identical across time --
    # which is why corpus drift was structurally invisible on this path. Unknown
    # liveness is now excluded and counted, rather than silently defaulting to alive.
    unknown = int(meta["first_period"].isna().sum())
    if unknown:
        logger.warning(
            f"{unknown}/{len(meta)} products have no reconstructable liveness window "
            f"and are excluded. If this is most of the corpus, raise max_reviews so the "
            f"review dump covers the products in the meta dump."
        )
    meta = meta[meta["first_period"].notna() & meta["last_period"].notna()]
    if meta.empty:
        raise RuntimeError(
            "no product has a reconstructable liveness window: the meta and review "
            "prefixes do not overlap. Increase DATA_MAX_REVIEWS / DATA_MAX_PRODUCTS."
        )

    for idx, period in enumerate(all_periods):
        alive = meta[
            (meta["first_period"] <= period) & (meta["last_period"] >= period)
        ].copy()
        alive["window"] = idx
        alive["period"] = str(period)

        q_window = qa[qa["_period"] == period].copy()
        q_window["window"] = idx
        q_window["period"] = str(period)

        c_path = cat_snap_dir / f"corpus_t{idx:03d}.parquet"
        q_path = cat_snap_dir / f"queries_t{idx:03d}.parquet"

        # doc_id/hub keep this path's output schema-compatible with the revision path
        alive["doc_id"] = alive[asin_col].astype(str)
        alive["hub"] = alive.get("main_category", pd.Series("unknown", index=alive.index)).fillna("unknown")
        q_window["query_id"] = [f"t{idx:03d}_q{i:05d}" for i in range(len(q_window))]

        alive[["window", "period", "doc_id", "hub", "text"]].to_parquet(c_path, index=False)
        q_window[["window", "period", "query_id", "text"]].to_parquet(q_path, index=False)

        if len(alive) < settings.data.min_window_samples:
            logger.warning(
                f"t={idx:03d}: {len(alive)} products is below "
                f"min_window_samples={settings.data.min_window_samples}"
            )

        corpus_paths.append(c_path)
        query_paths.append(q_path)
        logger.debug(f"  t={idx:03d} ({period}): {len(alive)} products, {len(q_window)} queries")

    logger.info(f"Snapshots written to {cat_snap_dir}")
    return {"corpus": corpus_paths, "queries": query_paths}


if __name__ == "__main__":
    build_snapshots()
